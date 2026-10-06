//! Forward pass state management and entry point.
//!
//! [`ForwardPassState`] owns pre-allocated scratch buffers reused each iteration.
//! [`ForwardPassInputs`] bundles per-call borrowed inputs (no allocation on hot path).

use std::sync::mpsc::Sender;
use std::time::Instant;

use cobre_core::WorkerPhaseTimings;
use cobre_core::{TrainingEvent, WorkerTimingPhase};
use cobre_solver::ActiveProfile;
use cobre_solver::{SolverInterface, SolverStatistics, StageTemplate};
use cobre_stochastic::context::ClassSchemes;
#[cfg(test)]
use cobre_stochastic::select_transition_child;
use cobre_stochastic::{
    ClassSampleRequest, ForwardNoiseTables, ForwardSampler, ForwardSamplerConfig, SampleRequest,
    build_forward_sampler,
};
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator,
};

#[cfg(test)]
use crate::setup::node_graph::NodeId;
use crate::training_session::iteration_scratch::IterationScratch;
use crate::training_session::rank_distribution::RankDistribution;
use crate::training_session::runtime::RuntimeHandles;
use crate::workspace::WorkspacePool;
use crate::{
    context::{StageContext, TrainingContext},
    cut::FutureCostFunction,
    error::SddpError,
    forward::{
        EnumeratedForwardResult, EnumeratedForwardScratch, EnumeratedParams, ForwardResult,
        StageKey, run_enumerated_forward, run_forward_stage,
    },
    setup::node_graph::{EnumeratedPlan, NodePos, StageIdx, Traversal, advance_sampled_node},
    solve::partition,
    solver_phase::Phase,
    solver_stats::SolverStatsDelta,
    trajectory::TrajectoryRecord,
    workspace::{BasisStore, BasisStoreSliceMut, SolverWorkspace},
};

/// Per-iteration argument bundle for [`ForwardPassState::run`].
///
/// Groups all borrowed inputs that vary between calls: solver workspaces,
/// basis store, stage context, records, and the per-iteration batch scalars.
/// Owned scratch buffers live on [`ForwardPassState`] and are not repeated here.
pub(crate) struct ForwardPassInputs<'a, S: SolverInterface + Send> {
    /// Solver workspaces (one per rayon worker thread).
    pub workspaces: &'a mut [SolverWorkspace<S>],
    /// Basis warm-start store (one slot per `(scenario, stage)` pair).
    pub basis_store: &'a mut BasisStore,
    /// Stage-level LP context (templates, row counts, noise scales).
    pub ctx: &'a StageContext<'a>,
    /// Frozen LP templates including pre-appended prior-iteration cuts.
    pub frozen: &'a [StageTemplate],
    /// Future-cost function — read-only for the forward pass.
    pub fcf: &'a FutureCostFunction,
    /// Study-level training context (horizon, indexer, stochastic model).
    pub training_ctx: &'a TrainingContext<'a>,
    /// Trajectory output records; pre-allocated by the caller.
    ///
    /// Length must equal `local_forward_passes * num_stages`.
    pub records: &'a mut [TrajectoryRecord],

    // ── Per-iteration batch scalars ──────
    /// Number of forward-pass scenarios assigned to this rank.
    pub local_forward_passes: usize,
    /// Total forward passes across all MPI ranks.
    pub total_forward_passes: usize,
    /// Current training iteration index (1-based).
    pub iteration: u64,
    /// Global index of this rank's first forward pass for seed derivation.
    pub fwd_offset: usize,
    /// Optional channel for emitting [`TrainingEvent::WorkerTiming`] events.
    pub event_sender: Option<&'a Sender<TrainingEvent>>,
}

impl<'a, S: SolverInterface + Send> ForwardPassInputs<'a, S> {
    /// Construct inputs from the fields of a `TrainingSession`, minus `fwd_state`
    /// (which the caller takes separately as a disjoint field borrow).
    // RATIONALE: 9 args are disjoint borrows of `TrainingSession` fields required because
    // Rust NLL cannot split a single `&mut TrainingSession` borrow when `fwd_state` is also
    // borrowed mutably. Each arg maps to a distinct session field; no grouping is possible
    // without adding indirection or invalidating the disjoint-borrow design.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_session_fields(
        fwd_pool: &'a mut WorkspacePool<S>,
        basis_store: &'a mut BasisStore,
        ctx: &'a StageContext<'a>,
        scratch: &'a mut IterationScratch,
        fcf: &'a FutureCostFunction,
        training_ctx: &'a TrainingContext<'a>,
        ranks: &RankDistribution,
        runtime: &'a RuntimeHandles,
        iteration: u64,
    ) -> Self {
        let fwd_record_len = ranks.my_actual_fwd * training_ctx.horizon.num_stages();
        Self {
            workspaces: &mut fwd_pool.workspaces,
            basis_store,
            ctx,
            frozen: &scratch.frozen_templates,
            fcf,
            training_ctx,
            records: &mut scratch.records[..fwd_record_len],
            local_forward_passes: ranks.my_actual_fwd,
            total_forward_passes: ranks.num_total_forward_passes,
            iteration,
            fwd_offset: ranks.my_fwd_offset,
            event_sender: runtime.event_sender(),
        }
    }
}

/// Read-only captures shared across all rayon workers in the forward pass.
///
/// Built once before the parallel region and passed by shared reference to every
/// [`run_forward_worker`] invocation; no field is mutated inside the worker.
pub(crate) struct ForwardWorkerParams<'a> {
    /// Number of forward passes assigned to this rank (local partition size).
    pub forward_passes: usize,
    /// Total forward passes across all MPI ranks (for seed derivation).
    pub total_forward_passes: usize,
    /// Number of rayon worker threads on this rank.
    pub n_workers: usize,
    /// Current training iteration index (1-based).
    pub iteration: u64,
    /// Global index of this rank's first forward pass (for seed derivation).
    pub fwd_offset: usize,
    /// The stage-0 root's canonical `NodeGraph` position — every trajectory's
    /// walk starts here. A chain-degenerate graph's root is `nodes[0]`.
    pub root_node: NodePos,
    /// Stage-level LP context (templates, row counts, noise scales).
    pub ctx: &'a StageContext<'a>,
    /// Frozen LP templates including pre-appended prior-iteration cuts.
    pub frozen: &'a [StageTemplate],
    /// Future-cost function — read-only for the forward pass.
    pub fcf: &'a FutureCostFunction,
    /// Study-level training context (horizon, indexer, stochastic model).
    pub training_ctx: &'a TrainingContext<'a>,
    /// Forward sampler that drives per-scenario-per-stage noise generation.
    pub sampler: &'a ForwardSampler<'a>,
    /// Per-iteration scenario-invariant tables backing `sampler`'s
    /// `OutOfSample` draws.
    pub noise_tables: &'a ForwardNoiseTables,
}

/// Return bundle from [`run_forward_worker`].
pub(crate) struct ForwardWorkerResult {
    /// Per-scenario trajectory costs for the local worker partition.
    pub trajectory_costs: Vec<f64>,
    /// Number of LP solves performed by this worker.
    pub local_solves: u64,
    /// Per-stage solver-stats accumulators for this worker.
    pub per_stage_stats: Vec<SolverStatsDelta>,
}

/// Scalar context threaded from [`ForwardPassState::run`] into
/// [`ForwardPassState::post_process_worker_results`].
///
/// Bundles the scalar values that are computed before the parallel region and
/// consumed during sequential post-processing, keeping the post-process helper's
/// argument count within the 8-parameter budget.
struct PostProcessContext {
    /// Total number of rayon workers used in the parallel region.
    n_workers: usize,
    /// Wall-clock duration of the parallel region in milliseconds.
    parallel_wall_ms: u64,
    /// `Instant` captured at the start of the entire `run()` call.
    start: Instant,
}

/// Owned scratch buffers for the forward pass, allocated once and reused.
///
/// Pre-sized from the study dimensions and reused across every iteration; every
/// field is cleared and repopulated each `run()`, so no allocation occurs on the
/// hot path. Per-iteration inputs are passed via [`ForwardPassInputs`].
// The `worker_` prefix on the per-worker scratch fields is what struct_field_names flags.
#[allow(clippy::struct_field_names)]
pub(crate) struct ForwardPassState {
    /// Per-worker, per-stage solver-stats accumulators (`n_workers × num_stages`).
    worker_stage_stats: Vec<Vec<SolverStatsDelta>>,

    /// Per-worker solver statistics snapshot taken **before** the parallel region.
    worker_stats_before: Vec<SolverStatistics>,

    /// Per-worker solver statistics snapshot taken **after** the parallel region.
    worker_stats_after: Vec<SolverStatistics>,

    /// Per-worker solver-statistics delta (after − before).
    worker_deltas: Vec<SolverStatsDelta>,

    /// Per-worker wall-time total for load-imbalance decomposition.
    worker_totals: Vec<f64>,

    /// Per-iteration scenario cost accumulator. Taken via `std::mem::take` into
    /// the [`ForwardResult`] each run and left empty; capacity grows only when
    /// `forward_passes` exceeds the previous maximum.
    scenario_costs: Vec<f64>,

    /// Per-stage solver-stats accumulator for the merged forward result. Swapped
    /// out via `std::mem::replace` into the [`ForwardResult`] each run, leaving a
    /// pre-sized empty vec so the next iteration's resize does not allocate.
    stage_stats: Vec<SolverStatsDelta>,

    /// Cross-rank error-reconciliation scratch, reused each iteration by the
    /// pre-`sync_forward` reconcile so that reconciliation never allocates.
    pub(crate) reconcile_scratch: [i32; 1],

    /// Resolved forward-phase solver profile applied at [`Self::run`] entry.
    /// Defaults to `Phase::Forward.profile()`; override with
    /// [`Self::set_profile`] before the first `run()` call.
    profile: ActiveProfile,

    /// The resolved forward-traversal axis: [`Self::run`] dispatches on this
    /// directly (no separate flag), so "selected enumerated but carries no
    /// plan" is unrepresentable. Set once via [`Self::set_traversal`];
    /// defaults to `Traversal::Sampled { forward_passes: 0 }` — the
    /// `run_forward_pass` test shim never calls `set_traversal`, matching its
    /// existing sampled-only behavior.
    traversal: Traversal,

    /// Reused per-run mutable scratch for the enumerated engine (arena,
    /// per-worker capture buffers). Meaningful only while `traversal` is
    /// `Enumerated`; always present (never `Option`) and grows to size on the
    /// first enumerated `run()`, so no allocation occurs after warm-up.
    enumerated_scratch: EnumeratedForwardScratch,

    /// Scenario-invariant per-class noise tables, rebuilt once per iteration
    /// in [`Self::run`] and shared by reference across every worker's draws.
    noise_tables: ForwardNoiseTables,
}

impl ForwardPassState {
    /// Allocate all scratch buffers sized for the given study dimensions.
    ///
    /// # Parameters
    ///
    /// - `n_workers`: number of rayon worker threads on this rank.
    /// - `num_stages`: total number of stages in the study horizon.
    /// - `max_local_fwd`: maximum number of forward-pass scenarios assigned to
    ///   this rank across all iterations. Used to pre-size `scenario_costs`.
    pub(crate) fn new(n_workers: usize, num_stages: usize, max_local_fwd: usize) -> Self {
        let worker_stage_stats = (0..n_workers)
            .map(|_| {
                (0..num_stages)
                    .map(|_| SolverStatsDelta::default())
                    .collect()
            })
            .collect();
        let stage_stats = (0..num_stages)
            .map(|_| SolverStatsDelta::default())
            .collect();
        Self {
            worker_stage_stats,
            worker_stats_before: Vec::with_capacity(n_workers),
            worker_stats_after: Vec::with_capacity(n_workers),
            worker_deltas: Vec::with_capacity(n_workers),
            worker_totals: Vec::with_capacity(n_workers),
            scenario_costs: Vec::with_capacity(max_local_fwd),
            stage_stats,
            reconcile_scratch: [0_i32; 1],
            profile: Phase::Forward.profile(),
            traversal: Traversal::default(),
            enumerated_scratch: EnumeratedForwardScratch::default(),
            noise_tables: ForwardNoiseTables::default(),
        }
    }

    /// Overrides the forward-phase solver profile applied at [`Self::run`]
    /// entry (default: `Phase::Forward.profile()`). Call before `run()`.
    pub(crate) fn set_profile(&mut self, profile: ActiveProfile) {
        self.profile = profile;
    }

    /// Routes [`Self::run`] to the enumerated all-paths engine
    /// (`Traversal::Enumerated`) or the sampled by-scenario driver
    /// (`Traversal::Sampled`, the default). Set once from the resolved
    /// selection; there is no separate forward-scheduler knob.
    pub(crate) fn set_traversal(&mut self, traversal: Traversal) {
        self.traversal = traversal;
    }

    /// The resolved forward-traversal axis, for a caller (the training
    /// session) that needs to read it after `run()` returns — e.g.
    /// `Traversal::path_weights()` for the exact upper-bound reduction.
    pub(crate) fn traversal(&self) -> &Traversal {
        &self.traversal
    }

    /// The persisted per-node outgoing-state arena from the most recent
    /// enumerated forward pass, handed to `BackwardPassInputs` so the
    /// enumerated backward fork reads a cut-generating node's own trial state
    /// directly instead of `records`/`exchange.state_at`. Empty (never
    /// indexed) under `Traversal::Sampled`.
    pub(crate) fn enumerated_state(&self) -> &EnumeratedForwardScratch {
        &self.enumerated_scratch
    }

    /// Execute the forward pass for one training iteration on this rank.
    ///
    /// Simulates `inputs.local_forward_passes` scenario trajectories through
    /// the full stage horizon, solving the stage LP at each `(scenario, stage)`
    /// pair with the current Future Cost Function approximation.
    ///
    /// # Errors
    ///
    /// Returns `Err(SddpError::Infeasible { .. })` when a stage LP has no
    /// feasible solution. Returns `Err(SddpError::Solver(_))` for all other
    /// terminal LP solver failures. Returns `Err(SddpError::Stochastic(_))` if
    /// `build_forward_sampler` fails, or if `rebuild_noise_tables` rejects an
    /// out-of-sample class wider than the Sobol dimension capacity.
    ///
    /// # Panics (debug builds only)
    ///
    /// Panics if any of the following debug preconditions are violated:
    ///
    /// - `inputs.records.len() != inputs.local_forward_passes * num_stages`
    /// - `inputs.training_ctx.initial_state.len() != state.n_state`
    /// - `inputs.frozen.len() != n_pools`
    pub(crate) fn run<S>(
        &mut self,
        inputs: &mut ForwardPassInputs<'_, S>,
    ) -> Result<ForwardResult, SddpError>
    where
        S: SolverInterface<Profile = ActiveProfile> + Send,
    {
        let training_ctx = inputs.training_ctx;
        let TrainingContext {
            horizon,
            state,
            stochastic,
            initial_state,
            ..
        } = training_ctx;

        let num_stages = horizon.num_stages();
        let forward_passes = inputs.local_forward_passes;

        debug_assert_eq!(inputs.records.len(), forward_passes * num_stages);
        debug_assert_eq!(initial_state.len(), state.n_state);
        let n_pools = training_ctx.node_graph.n_pools;
        debug_assert_eq!(
            inputs.frozen.len(),
            n_pools,
            "frozen templates length mismatch: expected {n_pools} pools, got {}",
            inputs.frozen.len()
        );

        let sampler = build_forward_sampler(ForwardSamplerConfig {
            class_schemes: ClassSchemes {
                inflow: Some(training_ctx.inflow_scheme),
                load: Some(training_ctx.load_scheme),
                ncs: Some(training_ctx.ncs_scheme),
            },
            ctx: stochastic,
            forward_seed: stochastic.forward_seed(),
            stages: training_ctx.stages,
            historical_library: training_ctx.historical_library,
            external_inflow_library: training_ctx.external_inflow_library,
            external_load_library: training_ctx.external_load_library,
            external_ncs_library: training_ctx.external_ncs_library,
        })?;
        #[allow(clippy::cast_possible_truncation)]
        sampler.rebuild_noise_tables(
            inputs.iteration as u32,
            inputs.total_forward_passes as u32,
            inputs.ctx.noise_group_ids,
            &mut self.noise_tables,
        )?;

        // Taken out for the match's duration so `self` stays freely `&mut`-usable
        // inside both arms (the `Enumerated` arm's `plan` borrows this local, not
        // `self`); restored before returning. Zero-allocation: `Traversal::default()`
        // is a stack-only `Sampled` variant.
        let traversal = std::mem::take(&mut self.traversal);
        let result = match &traversal {
            Traversal::Enumerated(plan) => {
                // Taken out for the call's duration, mirroring `traversal` above: `self`
                // must stay `&mut`-usable for `run_enumerated`'s receiver while this
                // reference is also passed as an argument. Restored right after.
                let noise_tables = std::mem::take(&mut self.noise_tables);
                let r = self.run_enumerated(inputs, &sampler, &noise_tables, plan);
                self.noise_tables = noise_tables;
                r
            }
            Traversal::Sampled { .. } => self.run_sampled(inputs, &sampler),
        };
        self.traversal = traversal;
        result
    }

    /// Monte-Carlo forward for one iteration: dispatched from [`Self::run`]
    /// under `Traversal::Sampled`. Partitions `inputs.local_forward_passes`
    /// trajectories across rayon workers, each walking the sampled node
    /// graph one stage at a time via [`run_forward_worker`].
    ///
    /// # Errors
    ///
    /// Propagates `SddpError::Infeasible`/`SddpError::Solver` from any stage
    /// solve, and `SddpError::Validation` from [`Self::resolve_root_node`] if
    /// stage 0 or the terminal stage carries no alive node.
    fn run_sampled<S>(
        &mut self,
        inputs: &mut ForwardPassInputs<'_, S>,
        sampler: &ForwardSampler<'_>,
    ) -> Result<ForwardResult, SddpError>
    where
        S: SolverInterface<Profile = ActiveProfile> + Send,
    {
        let training_ctx = inputs.training_ctx;
        let TrainingContext { horizon, .. } = training_ctx;
        let num_stages = horizon.num_stages();
        let forward_passes = inputs.local_forward_passes;

        let n_workers = inputs.workspaces.len().max(1);
        let start = Instant::now();

        let mut remaining: &mut [TrajectoryRecord] = inputs.records;
        let mut record_slices: Vec<&mut [TrajectoryRecord]> = Vec::with_capacity(n_workers);
        for w in 0..n_workers {
            let (start_m, end_m) = partition(forward_passes, n_workers, w);
            let (slice, rest) = remaining.split_at_mut((end_m - start_m) * num_stages);
            record_slices.push(slice);
            remaining = rest;
        }
        let basis_slices = inputs.basis_store.split_workers_mut(n_workers);

        let root_node = Self::resolve_root_node(training_ctx)?;

        // The worker count may differ from `new()` if the pool shrank.
        let shape_matches = self.worker_stage_stats.len() == n_workers
            && self.worker_stage_stats.first().map_or(0, Vec::len) == num_stages;
        if shape_matches {
            for inner in &mut self.worker_stage_stats {
                for d in inner.iter_mut() {
                    d.reset_in_place();
                }
            }
        } else {
            self.worker_stage_stats.clear();
            for _ in 0..n_workers {
                self.worker_stage_stats.push(
                    (0..num_stages)
                        .map(|_| SolverStatsDelta::default())
                        .collect(),
                );
            }
        }

        self.worker_stats_before.clear();
        self.worker_stats_before
            .extend(inputs.workspaces.iter().map(|ws| ws.solver.statistics()));

        // `set_profile` is delta-tracked: it issues solver-option FFI calls only
        // for fields that differ from each solver's current state.
        let forward_profile = self.profile;
        for ws in inputs.workspaces.iter_mut() {
            ws.solver.set_profile(&forward_profile);
            debug_assert!(
                ws.solver.current_profile() == &forward_profile,
                "solver profile must equal the profile passed to set_profile"
            );
            ws.worker_timing_buf = WorkerPhaseTimings::default();
        }

        let parallel_start = Instant::now();
        // Drain `worker_stage_stats` into the parallel closure; the updated stats
        // come back via each `ForwardWorkerResult` so the allocation is recycled.
        let worker_stage_stats_for_par: Vec<Vec<SolverStatsDelta>> =
            std::mem::take(&mut self.worker_stage_stats);

        let params = ForwardWorkerParams {
            forward_passes,
            total_forward_passes: inputs.total_forward_passes,
            n_workers,
            iteration: inputs.iteration,
            fwd_offset: inputs.fwd_offset,
            root_node,
            ctx: inputs.ctx,
            frozen: inputs.frozen,
            fcf: inputs.fcf,
            training_ctx,
            sampler,
            noise_tables: &self.noise_tables,
        };
        let worker_results: Vec<Result<ForwardWorkerResult, SddpError>> = inputs
            .workspaces
            .par_iter_mut()
            .zip(record_slices.par_iter_mut())
            .zip(basis_slices.into_par_iter())
            .zip(worker_stage_stats_for_par.into_par_iter())
            .enumerate()
            .map(
                |(w, (((ws, worker_records), mut basis_slice), mut per_stage_stats))| {
                    run_forward_worker(
                        w,
                        ws,
                        worker_records,
                        &mut basis_slice,
                        &mut per_stage_stats,
                        &params,
                    )
                },
            )
            .collect();

        #[allow(clippy::cast_possible_truncation)]
        let parallel_wall_ms = parallel_start.elapsed().as_millis() as u64;

        let ppc = PostProcessContext {
            n_workers,
            parallel_wall_ms,
            start,
        };
        self.post_process_worker_results(inputs, worker_results, &ppc)
    }

    /// Enumerated all-paths forward for one iteration: dispatched from
    /// [`Self::run`] under `Traversal::Enumerated`. Applies the forward
    /// profile and drives the by-node engine against the already-resolved
    /// `plan` (built once, at `Traversal` resolution — never re-derived
    /// here), whose per-path costs feed the exact `Σ P(ℓ)·C(ℓ)` bound.
    ///
    /// # Errors
    ///
    /// Propagates `SddpError::Infeasible`/`SddpError::Solver` from any node solve.
    fn run_enumerated<S>(
        &mut self,
        inputs: &mut ForwardPassInputs<'_, S>,
        sampler: &ForwardSampler<'_>,
        noise_tables: &ForwardNoiseTables,
        plan: &EnumeratedPlan,
    ) -> Result<ForwardResult, SddpError>
    where
        S: SolverInterface<Profile = ActiveProfile> + Send,
    {
        let start = Instant::now();
        let training_ctx = inputs.training_ctx;

        let forward_profile = self.profile;
        for ws in inputs.workspaces.iter_mut() {
            ws.solver.set_profile(&forward_profile);
            ws.worker_timing_buf = WorkerPhaseTimings::default();
        }

        Self::require_terminal_node(training_ctx)?;

        let dcs_params = training_ctx.dcs.filter(|p| p.is_active(inputs.iteration));

        let params = EnumeratedParams {
            iteration: inputs.iteration,
            fwd_offset: inputs.fwd_offset,
            local_forward_passes: inputs.local_forward_passes,
            total_forward_passes: inputs.total_forward_passes,
            ctx: inputs.ctx,
            frozen: inputs.frozen,
            fcf: inputs.fcf,
            training_ctx,
            sampler,
            noise_tables,
            dcs: dcs_params,
            event_sender: inputs.event_sender,
        };

        let EnumeratedForwardResult {
            scenario_costs,
            lp_solves,
            stage_stats,
        } = run_enumerated_forward(
            plan,
            &mut self.enumerated_scratch,
            inputs.workspaces,
            inputs.basis_store,
            inputs.records,
            &params,
        )?;

        #[allow(clippy::cast_possible_truncation)]
        let elapsed_ms = start.elapsed().as_millis() as u64;
        Ok(ForwardResult {
            scenario_costs,
            elapsed_ms,
            lp_solves,
            setup_time_ms: 0,
            load_imbalance_ms: 0,
            scheduling_overhead_ms: 0,
            stage_stats,
        })
    }

    /// Validate that the terminal stage carries an alive node. `Ok(())` for a
    /// zero-stage horizon.
    ///
    /// # Errors
    ///
    /// Returns [`SddpError::Validation`] if the terminal stage carries no alive node.
    fn require_terminal_node(training_ctx: &TrainingContext<'_>) -> Result<(), SddpError> {
        let num_stages = training_ctx.horizon.num_stages();
        if num_stages == 0 {
            return Ok(());
        }
        training_ctx
            .node_graph
            .any_stage_node(StageIdx(num_stages - 1))
            .ok_or_else(|| {
                SddpError::Validation(
                    "forward pass: terminal stage carries no alive node".to_string(),
                )
            })?;
        Ok(())
    }

    /// Resolve the sampled traversal's root node, and validate that the
    /// terminal stage carries an alive node.
    ///
    /// # Errors
    ///
    /// Returns [`SddpError::Validation`] if stage 0 or the terminal stage
    /// carries no alive node.
    fn resolve_root_node(training_ctx: &TrainingContext<'_>) -> Result<NodePos, SddpError> {
        let root_node = training_ctx
            .node_graph
            .frontier_node(StageIdx(0))
            .ok_or_else(|| {
                SddpError::Validation("forward pass: stage 0 carries no alive node".to_string())
            })?;
        Self::require_terminal_node(training_ctx)?;
        Ok(root_node)
    }

    /// Sequential post-processing after the rayon parallel region.
    ///
    /// # Errors
    ///
    /// Returns `Err(SddpError::*)` if any worker result is an `Err`.
    fn post_process_worker_results<S: SolverInterface + Send>(
        &mut self,
        inputs: &mut ForwardPassInputs<'_, S>,
        worker_results: Vec<Result<ForwardWorkerResult, SddpError>>,
        ppc: &PostProcessContext,
    ) -> Result<ForwardResult, SddpError> {
        let PostProcessContext {
            n_workers,
            parallel_wall_ms,
            start,
        } = *ppc;
        let num_stages = inputs.training_ctx.horizon.num_stages();

        self.worker_stats_after.clear();
        self.worker_stats_after
            .extend(inputs.workspaces.iter().map(|ws| ws.solver.statistics()));

        self.worker_deltas.clear();
        self.worker_deltas.extend(
            self.worker_stats_before
                .iter()
                .zip(&self.worker_stats_after)
                .map(|(b, a)| SolverStatsDelta::from_snapshots(b, a)),
        );

        // setup_time_ms: total non-solve work (load_model + set_bounds + basis_set).
        let fwd_setup_ms: f64 = self
            .worker_deltas
            .iter()
            .map(|d| d.load_model_time_ms + d.set_bounds_time_ms + d.basis_set_time_ms)
            .sum();

        self.worker_totals.clear();
        self.worker_totals
            .extend(self.worker_deltas.iter().map(|d| {
                d.solve_time_ms + d.load_model_time_ms + d.set_bounds_time_ms + d.basis_set_time_ms
            }));

        #[allow(clippy::cast_precision_loss)]
        let n_workers_f = n_workers as f64;
        let max_worker_ms = self.worker_totals.iter().copied().fold(0.0_f64, f64::max);
        let avg_worker_ms = if self.worker_totals.is_empty() {
            0.0_f64
        } else {
            self.worker_totals.iter().sum::<f64>() / n_workers_f
        };

        let fwd_imbalance_ms = (max_worker_ms - avg_worker_ms).max(0.0);
        #[allow(
            clippy::cast_precision_loss,
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss
        )]
        let fwd_scheduling_ms = (parallel_wall_ms as f64 - max_worker_ms).max(0.0);

        for (ws, delta) in inputs.workspaces.iter_mut().zip(&self.worker_deltas) {
            ws.worker_timing_buf.fwd_setup_ms +=
                delta.load_model_time_ms + delta.set_bounds_time_ms + delta.basis_set_time_ms;
        }
        if let Some(sender) = inputs.event_sender {
            for ws in inputs.workspaces.iter() {
                let _ = sender.send(TrainingEvent::WorkerTiming {
                    rank: ws.rank,
                    worker_id: ws.worker_id,
                    iteration: inputs.iteration,
                    phase: WorkerTimingPhase::Forward,
                    timings: ws.worker_timing_buf,
                });
            }
        }

        // Merge per-worker cost vectors in canonical global scenario index order,
        // and per-stage stats by element-wise summation across workers. The
        // canonical order keeps the merged result rank-count-invariant.
        let mut scenario_costs = std::mem::take(&mut self.scenario_costs);
        scenario_costs.clear();

        if self.stage_stats.len() != num_stages {
            self.stage_stats
                .resize_with(num_stages, SolverStatsDelta::default);
        }
        for d in &mut self.stage_stats {
            d.reset_in_place();
        }

        let mut lp_solves = 0u64;
        self.worker_stage_stats.clear();
        for result in worker_results {
            let ForwardWorkerResult {
                trajectory_costs: worker_costs,
                local_solves: w_solves,
                per_stage_stats: worker_stage_stats,
            } = result?;
            scenario_costs.extend(worker_costs);
            lp_solves += w_solves;
            for (dst, src) in self.stage_stats.iter_mut().zip(&worker_stage_stats) {
                SolverStatsDelta::accumulate_into(dst, src);
            }
            // Move the buffer back so the next iteration can reset in place.
            self.worker_stage_stats.push(worker_stage_stats);
        }

        let stage_stats = std::mem::replace(&mut self.stage_stats, Vec::with_capacity(num_stages));

        #[allow(clippy::cast_possible_truncation)]
        let elapsed_ms = start.elapsed().as_millis() as u64;

        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        Ok(ForwardResult {
            scenario_costs,
            elapsed_ms,
            lp_solves,
            setup_time_ms: fwd_setup_ms as u64,
            load_imbalance_ms: fwd_imbalance_ms as u64,
            scheduling_overhead_ms: fwd_scheduling_ms as u64,
            stage_stats,
        })
    }
}

/// Execute the forward pass for one rayon worker's scenario partition, through
/// every stage, accumulating trajectory costs and per-stage solver statistics.
///
/// `per_stage_stats` (length `num_stages`) is accumulated in place and taken via
/// `mem::take` into the result at return, recycling the allocation across
/// iterations.
///
/// # Errors
///
/// Propagates `Err(SddpError::Stochastic(_))` from `sampler.sample(...)` and
/// `Err(SddpError::Infeasible/Solver(_))` from [`run_forward_stage`].
// RATIONALE: one sequential per-(stage, trajectory) pipeline — node/state
// resolution, LP solve, and the transition-draw advance are load-bearing
// ordering, not independent steps a split would clarify; the numerical work
// itself is already delegated to `run_forward_stage`/`advance_sampled_node`.
#[allow(clippy::too_many_lines)]
pub(crate) fn run_forward_worker<S: SolverInterface + Send>(
    w: usize,
    ws: &mut SolverWorkspace<S>,
    worker_records: &mut [TrajectoryRecord],
    basis_slice: &mut BasisStoreSliceMut<'_>,
    per_stage_stats: &mut Vec<SolverStatsDelta>,
    params: &ForwardWorkerParams<'_>,
) -> Result<ForwardWorkerResult, SddpError> {
    let worker_wall_start = Instant::now();
    // Snapshot the cumulative lazy-scoring accumulator; the region-end delta
    // attributes this pass's scoring to the forward phase. The accumulator is
    // never reset, so a snapshot-delta is the only correct attribution; it stays
    // zero on the frozen path.
    let scoring_seconds_before = ws.backward_accum.dcs_solve.scoring_time_seconds;
    let (start_m, end_m) = partition(params.forward_passes, params.n_workers, w);
    let n_local = end_m - start_m;
    let num_stages = params.training_ctx.horizon.num_stages();

    ws.scratch.trajectory_costs_buf.clear();
    ws.scratch.trajectory_costs_buf.resize(n_local, 0.0_f64);

    // Sampling scratch taken out of ws so it can stay live while
    // run_forward_stage borrows ws (and so the allocation is reused).
    let mut raw_noise_buf = std::mem::take(&mut ws.scratch.raw_noise_buf);
    raw_noise_buf.resize(params.training_ctx.stochastic.dim(), 0.0_f64);
    let mut corr_scratch = std::mem::take(&mut ws.scratch.corr_scratch);
    corr_scratch.resize(2 * params.training_ctx.stochastic.dim(), 0.0_f64);

    // Per-trajectory sampled-walk node carrier, root-initialized: each
    // trajectory advances its own entry by the transition draw at the end of
    // every stage below. On a chain the root is the only node at every stage
    // (single out-edge, short-circuited — see the transition-draw call site),
    // so this reduces to reading `nodes[t]` byte-for-byte.
    let mut current_node_buf = std::mem::take(&mut ws.scratch.current_node_buf);
    current_node_buf.clear();
    current_node_buf.resize(n_local, params.root_node);

    let local_solve_count_before = ws.solver.statistics().solve_count;
    #[allow(clippy::cast_possible_truncation)]
    let total_scenarios_u32 = params.total_forward_passes as u32;

    // DCS decision for this pass: `Some` only when configured AND active at this
    // iteration; constant across all stages and scenarios.
    let dcs_params = params
        .training_ctx
        .dcs
        .filter(|p| p.is_active(params.iteration));

    // Rationale: `t` indexes several parallel per-stage collections
    // (params.frozen, params.ctx.cumulative_discount_factors) beyond
    // `per_stage_stats`, so an iterator over `per_stage_stats` alone would not
    // eliminate the range index.
    #[allow(clippy::needless_range_loop)]
    for t in (0..num_stages).map(StageIdx) {
        let cum_d = params
            .ctx
            .cumulative_discount_factors
            .get(t.0)
            .copied()
            .unwrap_or(1.0);

        for (local_m, m) in (start_m..end_m).enumerate() {
            // Each trajectory resolves its OWN visited node — the sampled walk;
            // below a recombination join, or on a declared fan, distinct
            // trajectories may sit at distinct nodes at the same stage `t`.
            let node = current_node_buf[local_m];
            let node_graph = params.training_ctx.node_graph;
            let pool_id = node_graph.nodes[node].pool_id;

            // Reset the solver's simplex state at the per-scenario boundary so
            // this scenario's landed vertex cannot depend on which scenarios the
            // worker solved before it (determinism across thread/rank counts).
            // No-op for HiGHS; for CLP recreates the model (`Clp_loadProblem`
            // leaves rim/pricing state stale). Must precede the per-scenario load
            // so the fresh CLP handle is the one repopulated — both the frozen
            // `load_model` below and the DCS-path load in `run_forward_stage`.
            ws.solver.reset_solver_state();

            // Reload model per scenario to ensure deterministic LP state across
            // thread assignments. The frozen all-cuts template is loaded here for
            // the frozen path; on the DCS path `run_forward_stage` instead loads
            // the cut-free base template (loading frozen would double-append the
            // embedded cut rows), so the frozen load is skipped.
            if dcs_params.is_none() {
                ws.solver.load_model(&params.frozen[pool_id]);
            }
            ws.current_state.clear();
            // Each trajectory's incoming state is its OWN previous-stage record —
            // a per-trajectory walk needs no parent-node lookup: every edge is
            // t -> t+1, so the state that fed this visit is always this same
            // trajectory's own `t - 1` solve, on a chain or a declared graph alike.
            let src: &[f64] = if t.0 == 0 {
                params.training_ctx.initial_state
            } else {
                &worker_records[local_m * num_stages + (t.0 - 1)].state
            };
            ws.current_state.extend_from_slice(src);

            if t.0 == 0 {
                if params.training_ctx.lag_accum_seed.is_empty() {
                    ws.scratch.lag_accumulator.fill(0.0);
                    ws.scratch.lag_weight_accum.fill(0.0);
                } else {
                    ws.scratch.lag_accumulator[..params.training_ctx.lag_accum_seed.len()]
                        .copy_from_slice(params.training_ctx.lag_accum_seed);
                    ws.scratch.lag_weight_accum[..params.training_ctx.lag_weight_seed.len()]
                        .copy_from_slice(params.training_ctx.lag_weight_seed);
                }
                ws.scratch.downstream_accumulator.fill(0.0);
                ws.scratch.downstream_weight_accum = 0.0;
                ws.scratch.downstream_completed_lags.fill(0.0);
                ws.scratch.downstream_n_completed = 0;
            }

            let global_scenario = params.fwd_offset + m;
            #[allow(clippy::cast_possible_truncation)]
            let (i32, s32, t32) = (params.iteration as u32, global_scenario as u32, t.0 as u32);

            let (node_opening_offset, node_opening_len) = node_graph.node_opening_range(node);

            if t.0 == 0 {
                let class_req = ClassSampleRequest {
                    iteration: i32,
                    scenario: s32,
                    stage: 0,
                    stage_idx: 0,
                    total_scenarios: total_scenarios_u32,
                    noise_group_id: 0,
                    node_opening_offset,
                    node_opening_len,
                    pinned_scenario: None,
                };
                params.sampler.apply_initial_state(
                    &class_req,
                    &mut ws.current_state,
                    params.training_ctx.state.inflow_lags.start,
                );
            }
            let noise = params.sampler.sample(SampleRequest {
                iteration: i32,
                scenario: s32,
                stage: t32,
                stage_idx: t.0,
                noise_buf: &mut raw_noise_buf,
                corr_scratch: &mut corr_scratch,
                total_scenarios: total_scenarios_u32,
                noise_group_id: params.ctx.noise_group_id_at(t),
                node_opening_offset,
                node_opening_len,
                pinned_scenario: None,
                tables: params.noise_tables,
            })?;
            let raw_noise = noise.as_slice();

            let key = StageKey {
                t,
                m,
                local_m,
                iteration: params.iteration,
                raw_noise,
                basis_row_capacity: params.frozen[pool_id].num_rows,
                pool: &params.fcf.pools[pool_id],
                dcs: dcs_params,
                node,
            };
            let stats_before_stage = ws.solver.statistics();
            let stage_cost = run_forward_stage(
                ws,
                basis_slice,
                params.ctx,
                params.training_ctx,
                &key,
                worker_records,
            )?;
            let stage_delta =
                SolverStatsDelta::from_snapshots(&stats_before_stage, &ws.solver.statistics());
            SolverStatsDelta::accumulate_into(&mut per_stage_stats[t.0], &stage_delta);
            ws.scratch.trajectory_costs_buf[local_m] += cum_d * stage_cost;

            // Advance this trajectory to the node it will visit at t + 1
            // (chain-parity contract stated once at `advance_sampled_node`).
            if t.next().0 < num_stages {
                current_node_buf[local_m] = advance_sampled_node(node_graph, node, i32, s32, t32);
            }
        }
    }

    ws.scratch.raw_noise_buf = raw_noise_buf;
    ws.scratch.corr_scratch = corr_scratch;
    ws.scratch.current_node_buf = current_node_buf;

    let local_solves = ws.solver.statistics().solve_count - local_solve_count_before;
    ws.worker_timing_buf.forward_wall_ms += worker_wall_start.elapsed().as_secs_f64() * 1_000.0;
    // Fold the forward-region lazy-scoring delta into the timing buffer (ms),
    // mirroring the `forward_wall_ms` fold above. Zero on the frozen path.
    ws.worker_timing_buf.scoring_ms +=
        (ws.backward_accum.dcs_solve.scoring_time_seconds - scoring_seconds_before) * 1_000.0;
    Ok(ForwardWorkerResult {
        // `mem::take` both buffers out of ws; post_process_worker_results returns
        // them so the allocations persist across training iterations.
        trajectory_costs: std::mem::take(&mut ws.scratch.trajectory_costs_buf),
        local_solves,
        per_stage_stats: std::mem::take(per_stage_stats),
    })
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use chrono::NaiveDate;
    use cobre_core::entities::hydro::{Hydro, HydroGenerationModel, HydroPenalties};
    use cobre_core::scenario::{
        CorrelationEntity, CorrelationGroup, CorrelationModel, CorrelationProfile, InflowModel,
        SamplingScheme,
    };
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    };
    use cobre_core::{Bus, DeficitSegment, EntityId, SystemBuilder, WorkerPhaseTimings};
    use cobre_solver::{
        Basis, LpSolution, ProfiledSolver, RowBatch, SolverError, SolverInterface,
        SolverStatistics, StageTemplate,
    };
    use cobre_stochastic::StochasticContext;
    use cobre_stochastic::context::{ClassSchemes, OpeningTreeInputs, build_stochastic_context};

    use super::*;
    use crate::{
        context::TrainingContext,
        cut::FutureCostFunction,
        horizon_mode::HorizonMode,
        inflow_method::InflowNonNegativityMethod,
        lp::builder::PatchBuffer,
        lp::indexer::{StateSpace, StudyDimensions},
        test_support::{
            StageContextFixture, equipment_free_geometry, permissive_state_boxes, state_layout,
            study_dims,
        },
        trajectory::TrajectoryRecord,
        workspace::{BackwardAccumulators, BasisStore, ScratchBuffers, SolverWorkspace},
    };

    // ── Minimal mock solver ────────────────────────────────────────────────

    struct MockSolver {
        solution: LpSolution,
        buf_primal: Vec<f64>,
        buf_dual: Vec<f64>,
        buf_reduced_costs: Vec<f64>,
        stats: SolverStatistics,
    }

    impl MockSolver {
        fn always_ok(solution: LpSolution) -> Self {
            let buf_primal = solution.primal.clone();
            let buf_dual = solution.dual.clone();
            let buf_reduced_costs = solution.reduced_costs.clone();
            Self {
                solution,
                buf_primal,
                buf_dual,
                buf_reduced_costs,
                stats: SolverStatistics::default(),
            }
        }
    }

    impl SolverInterface for MockSolver {
        type Profile = cobre_solver::ActiveProfile;

        fn apply_profile(&mut self, _profile: &cobre_solver::ActiveProfile) {}

        fn load_model(&mut self, _template: &StageTemplate) {}
        fn add_rows(&mut self, _rows: &RowBatch) {}
        fn set_row_bounds(&mut self, _i: &[usize], _lo: &[f64], _hi: &[f64]) {}
        fn set_col_bounds(&mut self, _i: &[usize], _lo: &[f64], _hi: &[f64]) {}
        fn solve(
            &mut self,
            _basis: Option<&Basis>,
        ) -> Result<cobre_solver::SolutionView<'_>, SolverError> {
            self.stats.solve_count += 1;
            self.buf_primal.copy_from_slice(&self.solution.primal);
            self.buf_dual.copy_from_slice(&self.solution.dual);
            self.buf_reduced_costs
                .copy_from_slice(&self.solution.reduced_costs);
            Ok(cobre_solver::SolutionView {
                objective: self.solution.objective,
                primal: &self.buf_primal,
                dual: &self.buf_dual,
                reduced_costs: &self.buf_reduced_costs,
                iterations: 0,
                solve_time_seconds: 0.0,
            })
        }
        fn get_basis(&mut self, out: &mut Basis) {
            crate::test_support::fill_consistent_basis(out);
        }
        fn statistics(&self) -> SolverStatistics {
            self.stats.clone()
        }
        fn statistics_into(&self, out: &mut SolverStatistics) {
            out.copy_from(&self.stats);
        }
        fn name(&self) -> &'static str {
            "MockSolver"
        }
        fn solver_name_version(&self) -> String {
            "MockSolver 0.0.0".to_string()
        }
    }

    // ── Fixture helpers ────────────────────────────────────────────────────

    fn minimal_template_1_0() -> StageTemplate {
        StageTemplate {
            num_cols: 4,
            num_rows: 1,
            num_nz: 1,
            col_starts: vec![0_i32, 0, 0, 1, 1],
            row_indices: vec![0_i32],
            values: vec![1.0],
            col_lower: vec![0.0, f64::NEG_INFINITY, 0.0, 0.0],
            col_upper: vec![f64::INFINITY; 4],
            objective: vec![0.0, 0.0, 0.0, 1.0],
            row_lower: vec![0.0],
            row_upper: vec![0.0],
            n_state: 1,
            col_scale: Vec::new(),
            row_scale: Vec::new(),
        }
    }

    fn fixed_solution_1_0() -> LpSolution {
        LpSolution {
            objective: 0.0,
            primal: vec![0.0; 4],
            dual: vec![0.0; 1],
            reduced_costs: vec![0.0; 4],
            iterations: 0,
            solve_time_seconds: 0.0,
        }
    }

    fn single_workspace(solver: MockSolver, state: &StateSpace) -> SolverWorkspace<MockSolver> {
        SolverWorkspace {
            rank: 0,
            worker_id: 0,
            solver: ProfiledSolver::new(solver),
            patch_buf: PatchBuffer::new(state, &[], &[]),
            current_state: Vec::with_capacity(state.n_state),
            scratch: ScratchBuffers {
                inflow_m3s_buf: Vec::with_capacity(state.hydro_count),
                lag_matrix_buf: Vec::with_capacity(state.max_par_order * state.hydro_count),
                par_inflow_buf: Vec::with_capacity(state.hydro_count),
                eta_floor_buf: Vec::with_capacity(state.hydro_count),
                zero_targets_buf: vec![0.0_f64; state.hydro_count],
                ncs_col_upper_buf: Vec::new(),
                ncs_col_lower_buf: Vec::new(),
                ncs_col_indices_buf: Vec::new(),
                ncs_col_lower_active_buf: Vec::new(),
                ncs_col_upper_active_buf: Vec::new(),
                last_ncs_col_start: usize::MAX,
                ncs_col_upper_extract_buf: Vec::new(),
                load_rhs_buf: Vec::new(),
                row_lower_buf: Vec::new(),
                z_inflow_rhs_buf: Vec::new(),
                effective_eta_buf: Vec::new(),
                unscaled_primal: Vec::new(),
                unscaled_dual: Vec::new(),
                lag_accumulator: vec![0.0_f64; state.hydro_count],
                lag_weight_accum: vec![0.0_f64; state.hydro_count],
                downstream_accumulator: Vec::new(),
                downstream_weight_accum: 0.0,
                downstream_completed_lags: Vec::new(),
                downstream_n_completed: 0,
                recon_slot_lookup: Vec::new(),
                trajectory_costs_buf: Vec::new(),
                raw_noise_buf: Vec::new(),
                corr_scratch: Vec::new(),
                current_node_buf: Vec::new(),
            },
            scratch_basis: Basis::new(0, 0),
            backward_accum: BackwardAccumulators::default(),
            worker_timing_buf: WorkerPhaseTimings::default(),
        }
    }

    #[allow(
        clippy::too_many_lines,
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap
    )]
    fn make_stochastic_context_2_stages() -> StochasticContext {
        let bus = Bus {
            id: EntityId(0),
            name: "B0".to_string(),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 1000.0,
            }],
            excess_cost: 0.0,
        };
        let mut hydro = Hydro {
            unit_groups: Vec::new(),
            id: EntityId(1),
            name: "H1".to_string(),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            downstream_id: None,
            travel_time_hours: None,
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: 0.0,
            max_storage_hm3: 100.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: HydroGenerationModel::ConstantProductivity,
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 100.0,
            specific_productivity_mw_per_m3s_per_m: None,
            min_generation_mw: 0.0,
            max_generation_mw: 100.0,
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: None,
            evaporation_reference_volumes_hm3: None,
            diversion: None,
            filling: None,
            penalties: HydroPenalties {
                spillage_cost: 0.0,
                diversion_cost: 0.0,
                turbined_cost: 0.0,
                storage_violation_below_cost: 0.0,
                filling_target_violation_cost: 0.0,
                turbined_violation_below_cost: 0.0,
                outflow_violation_below_cost: 0.0,
                outflow_violation_above_cost: 0.0,
                generation_violation_below_cost: 0.0,
                evaporation_violation_cost: 0.0,
                water_withdrawal_violation_cost: 0.0,
                water_withdrawal_violation_pos_cost: 0.0,
                water_withdrawal_violation_neg_cost: 0.0,
                evaporation_violation_pos_cost: 0.0,
                evaporation_violation_neg_cost: 0.0,
                inflow_nonnegativity_cost: 1000.0,
            },
        };
        hydro.declare_mirror_unit_group(EntityId(0));
        let stages: Vec<Stage> = make_stages_2();
        let inflow_models: Vec<InflowModel> = (0_i32..2)
            .map(|idx| InflowModel {
                hydro_id: EntityId(1),
                stage_id: idx,
                mean_m3s: 100.0,
                std_m3s: 30.0,
                ar_coefficients: vec![],
                residual_std_ratio: 1.0,
                annual: None,
            })
            .collect();
        let mut profiles = BTreeMap::new();
        profiles.insert(
            "default".to_string(),
            CorrelationProfile {
                groups: vec![CorrelationGroup {
                    name: "g1".to_string(),
                    entities: vec![CorrelationEntity {
                        entity_type: "inflow".to_string(),
                        id: EntityId(1),
                    }],
                    matrix: vec![vec![1.0]],
                }],
            },
        );
        let correlation = CorrelationModel {
            method: "spectral".to_string(),
            profiles,
            schedule: vec![],
        };
        let system = SystemBuilder::new()
            .buses(vec![bus])
            .hydros(vec![hydro])
            .stages(stages)
            .inflow_models(inflow_models)
            .correlation(correlation)
            .build()
            .unwrap();
        build_stochastic_context(
            &system,
            42,
            None,
            &[],
            &[],
            OpeningTreeInputs::default(),
            ClassSchemes {
                inflow: Some(SamplingScheme::InSample),
                load: Some(SamplingScheme::InSample),
                ncs: Some(SamplingScheme::InSample),
            },
        )
        .unwrap()
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    fn make_stages_2() -> Vec<Stage> {
        let make_stage = |idx: usize| Stage {
            index: idx,
            id: idx as i32,
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            season_id: Some(0),
            blocks: vec![Block {
                index: 0,
                name: "S".to_string(),
                duration_hours: 744.0,
            }],
            block_mode: BlockMode::Parallel,
            state_config: StageStateConfig {
                storage: true,
                inflow_lags: false,
            },
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor: 2,
                noise_method: NoiseMethod::Saa,
            },
        };
        (0..2).map(make_stage).collect()
    }

    // ── Fixture helper ────────────────────────────────────────────────────

    /// Owned data bundle for the 2-stage, 1-hydro, 2-scenario forward-pass fixture.
    ///
    /// Holds all owned values whose borrows must live long enough to create a
    /// `ForwardPassInputs`. Callers build `ForwardPassInputs` inline using the
    /// fields of this struct, which avoids lifetime issues that arise from
    /// returning borrows.
    struct ForwardFixture {
        n_stages: usize,
        n_scenarios: usize,
        state: StateSpace,
        templates: Vec<StageTemplate>,
        initial_state: Vec<f64>,
        fcf: FutureCostFunction,
        horizon: HorizonMode,
        stochastic: cobre_stochastic::StochasticContext,
        stages: Vec<Stage>,
        workspaces: Vec<SolverWorkspace<MockSolver>>,
        basis_store: BasisStore,
        records: Vec<TrajectoryRecord>,
    }

    impl ForwardFixture {
        fn new() -> Self {
            let n_stages = 2_usize;
            let n_scenarios = 2_usize;
            let state = state_layout(1, 0);
            let stochastic = make_stochastic_context_2_stages();
            let stages = make_stages_2();
            let solution = fixed_solution_1_0();
            let solver = MockSolver::always_ok(solution);
            let templates = vec![minimal_template_1_0(); n_stages];
            let initial_state = vec![0.0_f64; state.n_state];
            let fcf = FutureCostFunction::new(n_stages, state.n_state, 2, 10, &vec![0; n_stages]);
            let horizon = HorizonMode::Finite {
                num_stages: n_stages,
            };
            let workspaces = vec![single_workspace(solver, &state)];
            let basis_store = BasisStore::new(n_scenarios, n_stages);
            let records = (0..n_scenarios * n_stages)
                .map(|_| TrajectoryRecord {
                    primal: Vec::new(),
                    dual: Vec::new(),
                    stage_cost: 0.0,
                    node_id: NodeId(0),
                    state: Vec::new(),
                })
                .collect();
            Self {
                n_stages,
                n_scenarios,
                state,
                templates,
                initial_state,
                fcf,
                horizon,
                stochastic,
                stages,
                workspaces,
                basis_store,
                records,
            }
        }
    }

    // ── Tests ──────────────────────────────────────────────────────────────

    #[test]
    fn forward_pass_state_new_preallocates_per_worker_buffers() {
        let state = ForwardPassState::new(3, 5, 8);
        assert_eq!(state.worker_stage_stats.len(), 3);
        for inner in &state.worker_stage_stats {
            assert_eq!(inner.len(), 5);
        }
        assert_eq!(state.worker_stats_before.capacity(), 3);
        assert_eq!(state.worker_stats_after.capacity(), 3);
        assert_eq!(state.worker_deltas.capacity(), 3);
        assert_eq!(state.worker_totals.capacity(), 3);
    }

    /// Minimal 2-stage, 1-hydro, 2-scenario fixture driven through
    /// `ForwardPassState::run`. Asserts that the result carries exactly 2
    /// scenario costs (one per forward pass).
    #[test]
    fn forward_pass_state_run_produces_expected_scenario_count() {
        let mut fx = ForwardFixture::new();
        let state_boxes = permissive_state_boxes(fx.state.n_state, fx.n_stages);
        let geometry = equipment_free_geometry(&vec![0; fx.templates.len()]);
        let fixture = StageContextFixture::new(&fx.templates, &state_boxes, &geometry);
        let ctx = fixture.ctx();
        let study_dims = study_dims();
        let training_ctx = TrainingContext {
            node_graph: &crate::test_support::chain_node_graph(&fx.stochastic),
            horizon: &fx.horizon,
            state: &fx.state,
            cut_state_layouts: &[],
            study_dims: &study_dims,
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &fx.stochastic,
            initial_state: &fx.initial_state,
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            stages: &fx.stages,
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            lag_accum_seed: &[],
            lag_weight_seed: &[],
            dcs: None,
        };

        let mut state = ForwardPassState::new(1, fx.n_stages, fx.n_scenarios);
        let mut inputs = ForwardPassInputs {
            workspaces: &mut fx.workspaces,
            basis_store: &mut fx.basis_store,
            ctx: &ctx,
            frozen: &fx.templates,
            fcf: &fx.fcf,
            training_ctx: &training_ctx,
            records: &mut fx.records,
            local_forward_passes: fx.n_scenarios,
            total_forward_passes: fx.n_scenarios,
            iteration: 1,
            fwd_offset: 0,
            event_sender: None,
        };

        let result = state.run(&mut inputs).expect("forward pass must not error");

        assert_eq!(
            result.scenario_costs.len(),
            fx.n_scenarios,
            "result must carry one cost per forward-pass scenario"
        );
    }

    /// A profile installed via `set_profile` before `run()` is the one
    /// `ProfiledSolver::current_profile()` reports afterwards.
    #[test]
    fn forward_pass_state_set_profile_reaches_current_profile_after_run() {
        let mut fx = ForwardFixture::new();
        let state_boxes = permissive_state_boxes(fx.state.n_state, fx.n_stages);
        let geometry = equipment_free_geometry(&vec![0; fx.templates.len()]);
        let fixture = StageContextFixture::new(&fx.templates, &state_boxes, &geometry);
        let ctx = fixture.ctx();
        let study_dims = study_dims();
        let training_ctx = TrainingContext {
            node_graph: &crate::test_support::chain_node_graph(&fx.stochastic),
            horizon: &fx.horizon,
            state: &fx.state,
            cut_state_layouts: &[],
            study_dims: &study_dims,
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &fx.stochastic,
            initial_state: &fx.initial_state,
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            stages: &fx.stages,
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            lag_accum_seed: &[],
            lag_weight_seed: &[],
            dcs: None,
        };

        let mut state = ForwardPassState::new(1, fx.n_stages, fx.n_scenarios);
        let resolved =
            Phase::Forward.resolve_profile(Some(&cobre_io::config::PhaseSolverProfileConfig {
                dual_edge_weight: Some(cobre_io::config::DualEdgeWeight::Dantzig),
                scale: None,
                price: None,
                primal_feasibility_tolerance: None,
                dual_feasibility_tolerance: None,
                presolve: None,
                simplex_update_limit: None,
                cost_perturbation: None,
                refactor_error_tolerance: None,
                factor_pivot_threshold: None,
                use_warm_start: None,
                steepest_edge_devex_fallback_threshold: None,
            }));
        state.set_profile(resolved);
        let mut inputs = ForwardPassInputs {
            workspaces: &mut fx.workspaces,
            basis_store: &mut fx.basis_store,
            ctx: &ctx,
            frozen: &fx.templates,
            fcf: &fx.fcf,
            training_ctx: &training_ctx,
            records: &mut fx.records,
            local_forward_passes: fx.n_scenarios,
            total_forward_passes: fx.n_scenarios,
            iteration: 1,
            fwd_offset: 0,
            event_sender: None,
        };

        let _ = state.run(&mut inputs).expect("forward pass must not error");

        assert_eq!(
            inputs.workspaces[0].solver.current_profile(),
            &resolved,
            "the profile installed via set_profile must be the one stored on \
             current_profile after run()"
        );
    }

    /// Verify that `run_forward_worker` produces exactly `n_local` trajectory
    /// costs for the worker's scenario partition.
    #[test]
    fn run_forward_worker_produces_expected_trajectory_costs() {
        let fx = ForwardFixture::new();
        let state_boxes = permissive_state_boxes(fx.state.n_state, fx.n_stages);
        let geometry = equipment_free_geometry(&vec![0; fx.templates.len()]);
        let fixture = StageContextFixture::new(&fx.templates, &state_boxes, &geometry);
        let ctx = fixture.ctx();
        let study_dims = study_dims();
        let training_ctx = TrainingContext {
            node_graph: &crate::test_support::chain_node_graph(&fx.stochastic),
            horizon: &fx.horizon,
            state: &fx.state,
            cut_state_layouts: &[],
            study_dims: &study_dims,
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &fx.stochastic,
            initial_state: &fx.initial_state,
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            stages: &fx.stages,
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            lag_accum_seed: &[],
            lag_weight_seed: &[],
            dcs: None,
        };

        let sampler = build_forward_sampler(ForwardSamplerConfig {
            class_schemes: ClassSchemes {
                inflow: Some(SamplingScheme::InSample),
                load: Some(SamplingScheme::InSample),
                ncs: Some(SamplingScheme::InSample),
            },
            ctx: &fx.stochastic,
            forward_seed: fx.stochastic.forward_seed(),
            stages: &fx.stages,
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
        })
        .expect("sampler build must not error");
        let mut noise_tables = ForwardNoiseTables::default();
        sampler
            .rebuild_noise_tables(
                1,
                u32::try_from(fx.n_scenarios).expect("fits u32"),
                &[],
                &mut noise_tables,
            )
            .expect("test fixture never exceeds the Sobol dimension cap");

        let params = ForwardWorkerParams {
            forward_passes: fx.n_scenarios,
            total_forward_passes: fx.n_scenarios,
            n_workers: 1,
            iteration: 1,
            fwd_offset: 0,
            root_node: NodePos(0),
            ctx: &ctx,
            frozen: &fx.templates,
            fcf: &fx.fcf,
            training_ctx: &training_ctx,
            sampler: &sampler,
            noise_tables: &noise_tables,
        };

        // Mutable per-call state: independent allocations, not borrows of fx.
        let mut ws = single_workspace(MockSolver::always_ok(fixed_solution_1_0()), &fx.state);
        let mut basis_store = BasisStore::new(fx.n_scenarios, fx.n_stages);
        let mut basis_slices = basis_store.split_workers_mut(1);
        let mut basis_slice = basis_slices.remove(0);
        let mut records: Vec<TrajectoryRecord> = (0..fx.n_scenarios * fx.n_stages)
            .map(|_| TrajectoryRecord {
                primal: Vec::new(),
                dual: Vec::new(),
                stage_cost: 0.0,
                node_id: NodeId(0),
                state: Vec::new(),
            })
            .collect();
        let mut per_stage_stats: Vec<SolverStatsDelta> = (0..fx.n_stages)
            .map(|_| SolverStatsDelta::default())
            .collect();

        let result = run_forward_worker(
            0,
            &mut ws,
            &mut records,
            &mut basis_slice,
            &mut per_stage_stats,
            &params,
        )
        .expect("run_forward_worker must not error");

        assert_eq!(
            result.trajectory_costs.len(),
            fx.n_scenarios,
            "worker 0 owns all scenarios when n_workers=1; expected {} costs",
            fx.n_scenarios
        );
    }

    /// After two calls to `ForwardPassState::run` with the same dimensions,
    /// the shape is unchanged and the INNER per-stage stats buffers are
    /// recycled (reset in place, round-tripped through the workers, never
    /// reallocated). The OUTER Vec's buffer is deliberately not asserted:
    /// `run` takes it (`std::mem::take`) into the parallel region and rebuilds
    /// it in post-processing, so its address is an allocator coincidence, not
    /// a contract.
    #[test]
    fn forward_pass_state_run_preserves_worker_stage_stats_shape() {
        let mut fx = ForwardFixture::new();
        let state_boxes = permissive_state_boxes(fx.state.n_state, fx.n_stages);
        let geometry = equipment_free_geometry(&vec![0; fx.templates.len()]);
        let fixture = StageContextFixture::new(&fx.templates, &state_boxes, &geometry);
        let ctx = fixture.ctx();
        let study_dims = study_dims();
        let training_ctx = TrainingContext {
            node_graph: &crate::test_support::chain_node_graph(&fx.stochastic),
            horizon: &fx.horizon,
            state: &fx.state,
            cut_state_layouts: &[],
            study_dims: &study_dims,
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &fx.stochastic,
            initial_state: &fx.initial_state,
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            stages: &fx.stages,
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            lag_accum_seed: &[],
            lag_weight_seed: &[],
            dcs: None,
        };

        let mut state = ForwardPassState::new(1, fx.n_stages, fx.n_scenarios);

        {
            let mut inputs = ForwardPassInputs {
                workspaces: &mut fx.workspaces,
                basis_store: &mut fx.basis_store,
                ctx: &ctx,
                frozen: &fx.templates,
                fcf: &fx.fcf,
                training_ctx: &training_ctx,
                records: &mut fx.records,
                local_forward_passes: fx.n_scenarios,
                total_forward_passes: fx.n_scenarios,
                iteration: 1,
                fwd_offset: 0,
                event_sender: None,
            };
            let _ = state.run(&mut inputs).expect("first run must not error");
        }

        // Capture the INNER buffer's heap address after the first run; moves of
        // the Vec header (worker round-trip) do not move the heap buffer.
        let inner_ptr_after_first = state.worker_stage_stats[0].as_ptr();
        let inner_cap_after_first = state.worker_stage_stats[0].capacity();
        let len_after_first = state.worker_stage_stats.len();
        let inner_len_after_first = state.worker_stage_stats[0].len();

        {
            let mut inputs = ForwardPassInputs {
                workspaces: &mut fx.workspaces,
                basis_store: &mut fx.basis_store,
                ctx: &ctx,
                frozen: &fx.templates,
                fcf: &fx.fcf,
                training_ctx: &training_ctx,
                records: &mut fx.records,
                local_forward_passes: fx.n_scenarios,
                total_forward_passes: fx.n_scenarios,
                iteration: 2,
                fwd_offset: 0,
                event_sender: None,
            };
            let _ = state.run(&mut inputs).expect("second run must not error");
        }

        assert_eq!(
            state.worker_stage_stats[0].as_ptr(),
            inner_ptr_after_first,
            "inner per-stage stats buffer must be recycled, not reallocated"
        );
        assert_eq!(
            state.worker_stage_stats[0].capacity(),
            inner_cap_after_first,
            "inner per-stage stats buffer must not grow between runs"
        );
        assert_eq!(
            state.worker_stage_stats.len(),
            len_after_first,
            "outer Vec length must be unchanged"
        );
        assert_eq!(
            state.worker_stage_stats[0].len(),
            inner_len_after_first,
            "inner Vec length must be unchanged"
        );
    }

    /// `ForwardPassState::new` pre-allocates `scenario_costs` with capacity
    /// >= `max_local_fwd` so the first iteration never needs to grow the buffer.
    #[test]
    fn forward_pass_state_scenario_costs_are_preallocated() {
        let state = ForwardPassState::new(4, 12, 8);
        assert!(
            state.scenario_costs.capacity() >= 8,
            "scenario_costs must be pre-allocated with capacity >= max_local_fwd (8), got {}",
            state.scenario_costs.capacity()
        );
    }

    /// After two `run()` calls the returned `ForwardResult.scenario_costs` must
    /// retain capacity >= `n_scenarios` on both runs, proving that the
    /// pre-allocated buffer from `new()` is used on run 1 and that run 2 does
    /// not allocate a smaller buffer (`Vec::extend` preserves existing capacity).
    #[test]
    fn forward_pass_state_run_reuses_scenario_costs_allocation() {
        let mut fx = ForwardFixture::new();
        let state_boxes = permissive_state_boxes(fx.state.n_state, fx.n_stages);
        let geometry = equipment_free_geometry(&vec![0; fx.templates.len()]);
        let fixture = StageContextFixture::new(&fx.templates, &state_boxes, &geometry);
        let ctx = fixture.ctx();
        let study_dims = study_dims();
        let training_ctx = TrainingContext {
            node_graph: &crate::test_support::chain_node_graph(&fx.stochastic),
            horizon: &fx.horizon,
            state: &fx.state,
            cut_state_layouts: &[],
            study_dims: &study_dims,
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &fx.stochastic,
            initial_state: &fx.initial_state,
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            stages: &fx.stages,
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            lag_accum_seed: &[],
            lag_weight_seed: &[],
            dcs: None,
        };

        let mut state = ForwardPassState::new(1, fx.n_stages, fx.n_scenarios);

        let result1 = {
            let mut inputs = ForwardPassInputs {
                workspaces: &mut fx.workspaces,
                basis_store: &mut fx.basis_store,
                ctx: &ctx,
                frozen: &fx.templates,
                fcf: &fx.fcf,
                training_ctx: &training_ctx,
                records: &mut fx.records,
                local_forward_passes: fx.n_scenarios,
                total_forward_passes: fx.n_scenarios,
                iteration: 1,
                fwd_offset: 0,
                event_sender: None,
            };
            state.run(&mut inputs).expect("run 1 must not error")
        };

        assert!(
            result1.scenario_costs.capacity() >= fx.n_scenarios,
            "run 1: scenario_costs capacity {} must be >= n_scenarios {}",
            result1.scenario_costs.capacity(),
            fx.n_scenarios
        );

        let result2 = {
            let mut inputs = ForwardPassInputs {
                workspaces: &mut fx.workspaces,
                basis_store: &mut fx.basis_store,
                ctx: &ctx,
                frozen: &fx.templates,
                fcf: &fx.fcf,
                training_ctx: &training_ctx,
                records: &mut fx.records,
                local_forward_passes: fx.n_scenarios,
                total_forward_passes: fx.n_scenarios,
                iteration: 2,
                fwd_offset: 0,
                event_sender: None,
            };
            state.run(&mut inputs).expect("run 2 must not error")
        };

        assert!(
            result2.scenario_costs.capacity() >= fx.n_scenarios,
            "run 2: scenario_costs capacity {} must be >= n_scenarios {}",
            result2.scenario_costs.capacity(),
            fx.n_scenarios
        );
    }

    // ── Per-entity weight-seed reset fixture (2 hydros, 1 stage) ────────────

    fn minimal_template_2_hydros() -> StageTemplate {
        StageTemplate {
            num_cols: 7,
            num_rows: 2,
            num_nz: 2,
            col_starts: vec![0, 0, 0, 0, 0, 1, 2, 2],
            row_indices: vec![0, 1],
            values: vec![1.0, 1.0],
            col_lower: vec![
                0.0,
                0.0,
                f64::NEG_INFINITY,
                f64::NEG_INFINITY,
                0.0,
                0.0,
                0.0,
            ],
            col_upper: vec![f64::INFINITY; 7],
            objective: vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            row_lower: vec![0.0, 0.0],
            row_upper: vec![0.0, 0.0],
            n_state: 2,
            col_scale: Vec::new(),
            row_scale: Vec::new(),
        }
    }

    fn make_stage_1_2_hydros() -> Vec<Stage> {
        vec![Stage {
            index: 0,
            id: 0,
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            season_id: Some(0),
            blocks: vec![Block {
                index: 0,
                name: "S".to_string(),
                duration_hours: 744.0,
            }],
            block_mode: BlockMode::Parallel,
            state_config: StageStateConfig {
                storage: true,
                inflow_lags: false,
            },
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor: 1,
                noise_method: NoiseMethod::Saa,
            },
        }]
    }

    #[allow(clippy::too_many_lines)]
    fn make_stochastic_context_2_hydros_1_stage(stages: &[Stage]) -> StochasticContext {
        let bus = Bus {
            id: EntityId(0),
            name: "B0".to_string(),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 1000.0,
            }],
            excess_cost: 0.0,
        };
        let make_hydro = |id: i32, name: &str| {
            let mut hydro = Hydro {
                unit_groups: Vec::new(),
                id: EntityId(id),
                name: name.to_string(),
                operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                downstream_id: None,
                travel_time_hours: None,
                entry_stage_id: None,
                exit_stage_id: None,
                min_storage_hm3: 0.0,
                max_storage_hm3: 100.0,
                min_outflow_m3s: 0.0,
                max_outflow_m3s: None,
                generation_model: HydroGenerationModel::ConstantProductivity,
                min_turbined_m3s: 0.0,
                max_turbined_m3s: 100.0,
                specific_productivity_mw_per_m3s_per_m: None,
                min_generation_mw: 0.0,
                max_generation_mw: 100.0,
                tailrace: None,
                hydraulic_losses: None,
                efficiency: None,
                evaporation_coefficients_mm: None,
                evaporation_reference_volumes_hm3: None,
                diversion: None,
                filling: None,
                penalties: HydroPenalties {
                    spillage_cost: 0.0,
                    diversion_cost: 0.0,
                    turbined_cost: 0.0,
                    storage_violation_below_cost: 0.0,
                    filling_target_violation_cost: 0.0,
                    turbined_violation_below_cost: 0.0,
                    outflow_violation_below_cost: 0.0,
                    outflow_violation_above_cost: 0.0,
                    generation_violation_below_cost: 0.0,
                    evaporation_violation_cost: 0.0,
                    water_withdrawal_violation_cost: 0.0,
                    water_withdrawal_violation_pos_cost: 0.0,
                    water_withdrawal_violation_neg_cost: 0.0,
                    evaporation_violation_pos_cost: 0.0,
                    evaporation_violation_neg_cost: 0.0,
                    inflow_nonnegativity_cost: 1000.0,
                },
            };
            hydro.declare_mirror_unit_group(EntityId(0));
            hydro
        };
        let inflow_models: Vec<InflowModel> = [1_i32, 2]
            .into_iter()
            .map(|hid| InflowModel {
                hydro_id: EntityId(hid),
                stage_id: 0,
                mean_m3s: 100.0,
                std_m3s: 30.0,
                ar_coefficients: vec![],
                residual_std_ratio: 1.0,
                annual: None,
            })
            .collect();
        let mut profiles = BTreeMap::new();
        profiles.insert(
            "default".to_string(),
            CorrelationProfile {
                groups: vec![CorrelationGroup {
                    name: "g1".to_string(),
                    entities: vec![
                        CorrelationEntity {
                            entity_type: "inflow".to_string(),
                            id: EntityId(1),
                        },
                        CorrelationEntity {
                            entity_type: "inflow".to_string(),
                            id: EntityId(2),
                        },
                    ],
                    matrix: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
                }],
            },
        );
        let correlation = CorrelationModel {
            method: "spectral".to_string(),
            profiles,
            schedule: vec![],
        };
        let system = SystemBuilder::new()
            .buses(vec![bus])
            .hydros(vec![make_hydro(1, "H1"), make_hydro(2, "H2")])
            .stages(stages.to_vec())
            .inflow_models(inflow_models)
            .correlation(correlation)
            .build()
            .unwrap();
        build_stochastic_context(
            &system,
            42,
            None,
            &[],
            &[],
            OpeningTreeInputs::default(),
            ClassSchemes {
                inflow: Some(SamplingScheme::InSample),
                load: Some(SamplingScheme::InSample),
                ncs: Some(SamplingScheme::InSample),
            },
        )
        .unwrap()
    }

    /// Two hydros, one stage: hydro A carries a full-coverage weight seed
    /// (1.0), hydro B a half-coverage weight seed (0.5). After the
    /// trajectory-start reset, `ws.scratch.lag_weight_accum` must carry each
    /// hydro's OWN weight — a shared scalar (the pre-widening behavior) would
    /// instead broadcast one value to both entities.
    #[test]
    fn forward_reset_copies_per_entity_weight_seed() {
        let state = state_layout(2, 0);
        let stages = make_stage_1_2_hydros();
        let stochastic = make_stochastic_context_2_hydros_1_stage(&stages);
        let templates = vec![minimal_template_2_hydros()];
        let initial_state = vec![0.0_f64; state.n_state];
        let fcf = FutureCostFunction::new(1, state.n_state, 1, 10, &[0_u32]);
        let horizon = HorizonMode::Finite { num_stages: 1 };

        let state_boxes = permissive_state_boxes(state.n_state, 1);
        let geometry = equipment_free_geometry(&vec![0; templates.len()]);
        let fixture = StageContextFixture::new(&templates, &state_boxes, &geometry);
        let ctx = fixture.ctx();
        let study_dims = study_dims();
        let lag_accum_seed = [0.0_f64, 0.0_f64];
        let lag_weight_seed = [1.0_f64, 0.5_f64];
        let training_ctx = TrainingContext {
            node_graph: &crate::test_support::chain_node_graph(&stochastic),
            horizon: &horizon,
            state: &state,
            cut_state_layouts: &[],
            study_dims: &study_dims,
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &stochastic,
            initial_state: &initial_state,
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            stages: &stages,
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            lag_accum_seed: &lag_accum_seed,
            lag_weight_seed: &lag_weight_seed,
            dcs: None,
        };

        let sampler = build_forward_sampler(ForwardSamplerConfig {
            class_schemes: ClassSchemes {
                inflow: Some(SamplingScheme::InSample),
                load: Some(SamplingScheme::InSample),
                ncs: Some(SamplingScheme::InSample),
            },
            ctx: &stochastic,
            forward_seed: stochastic.forward_seed(),
            stages: &stages,
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
        })
        .expect("sampler build must not error");
        let mut noise_tables = ForwardNoiseTables::default();
        sampler
            .rebuild_noise_tables(1, 1, &[], &mut noise_tables)
            .expect("test fixture never exceeds the Sobol dimension cap");

        let params = ForwardWorkerParams {
            forward_passes: 1,
            total_forward_passes: 1,
            n_workers: 1,
            iteration: 1,
            fwd_offset: 0,
            root_node: NodePos(0),
            ctx: &ctx,
            frozen: &templates,
            fcf: &fcf,
            training_ctx: &training_ctx,
            sampler: &sampler,
            noise_tables: &noise_tables,
        };

        let solution = LpSolution {
            objective: 0.0,
            primal: vec![0.0; 7],
            dual: vec![0.0; 2],
            reduced_costs: vec![0.0; 7],
            iterations: 0,
            solve_time_seconds: 0.0,
        };
        let mut ws = single_workspace(MockSolver::always_ok(solution), &state);
        let mut basis_store = BasisStore::new(1, 1);
        let mut basis_slices = basis_store.split_workers_mut(1);
        let mut basis_slice = basis_slices.remove(0);
        let mut records: Vec<TrajectoryRecord> = vec![TrajectoryRecord {
            primal: Vec::new(),
            dual: Vec::new(),
            stage_cost: 0.0,
            node_id: NodeId(0),
            state: Vec::new(),
        }];
        let mut per_stage_stats = vec![SolverStatsDelta::default()];

        // Reset happens at trajectory start (t == 0), before any LP solve;
        // the mock solver's outcome is irrelevant to this assertion.
        let _ = run_forward_worker(
            0,
            &mut ws,
            &mut records,
            &mut basis_slice,
            &mut per_stage_stats,
            &params,
        );

        assert_eq!(
            ws.scratch.lag_weight_accum[0], 1.0,
            "hydro A (full coverage) weight should be 1.0, got {}",
            ws.scratch.lag_weight_accum[0]
        );
        assert_eq!(
            ws.scratch.lag_weight_accum[1], 0.5,
            "hydro B (half coverage) weight should be 0.5, got {}",
            ws.scratch.lag_weight_accum[1]
        );
    }

    // ── Declared-graph frontier resolution (branching coverage) ─────────────

    /// A root fanning into 4 leaves at stage 1. Each leaf's frontier visit
    /// must read its OWN pool id (the shared leaf pool) and node id — never a
    /// positional `nodes[t]` read, which on this declared graph would land on
    /// whichever leaf happens to occupy canonical position 1 — and its
    /// incoming state must resolve to the root's own outgoing state.
    #[test]
    fn declared_k_fan_frontier_resolves_each_leaf_own_pool_and_node_id_with_root_incoming_state() {
        use cobre_core::HorizonGraph;
        use cobre_core::temporal::{Node, PolicyGraphType, Transition};
        use cobre_io::StageIdResolver;

        use crate::setup::node_graph::build_node_graph;

        fn node(id: i32, stage_id: i32) -> Node {
            Node {
                id,
                stage_id,
                scenario_id: None,
                label: None,
            }
        }
        fn transition(source_id: i32, target_id: i32, probability: f64) -> Transition {
            Transition {
                source_id,
                target_id,
                probability,
                annual_discount_rate_override: None,
            }
        }

        let stochastic = make_stochastic_context_2_stages();
        let study_stage_ids = [0_i32, 1_i32];
        let resolver = StageIdResolver::from_study_stage_ids(&study_stage_ids);
        let graph = HorizonGraph {
            graph_type: PolicyGraphType::FiniteHorizon,
            annual_discount_rate: 0.0,
            nodes: vec![node(0, 0), node(1, 1), node(2, 1), node(3, 1), node(4, 1)],
            transitions: vec![
                transition(0, 1, 0.25),
                transition(0, 2, 0.25),
                transition(0, 3, 0.25),
                transition(0, 4, 0.25),
            ],
            stage_discount_rate_overrides: std::collections::BTreeMap::new(),
            season_map: None,
        };
        let node_graph = build_node_graph(&graph, 2, &resolver, &stochastic)
            .expect("declared K-fan graph must build");

        let root = NodePos(0);
        let leaves: Vec<NodePos> = node_graph.stage_frontier(StageIdx(1)).collect();
        assert_eq!(leaves.len(), 4, "all 4 leaves must be alive at stage 1");

        let expected_leaf_pool = node_graph.nodes[leaves[0]].pool_id;
        assert_ne!(
            expected_leaf_pool, node_graph.nodes[root].pool_id,
            "the shared leaf pool must differ from the root's own pool"
        );
        let leaf_node_ids: Vec<NodeId> = leaves.iter().map(|&l| node_graph.node_ids[l]).collect();
        assert_eq!(
            leaf_node_ids,
            vec![NodeId(1), NodeId(2), NodeId(3), NodeId(4)],
            "each leaf must resolve its OWN declared node id"
        );

        let root_state = vec![7.0_f64, 8.0_f64];
        let num_stages = 2_usize;
        let mut worker_records: Vec<TrajectoryRecord> = (0..num_stages)
            .map(|_| TrajectoryRecord {
                primal: Vec::new(),
                dual: Vec::new(),
                stage_cost: 0.0,
                node_id: NodeId(0),
                state: Vec::new(),
            })
            .collect();
        worker_records[0].state = root_state.clone();
        worker_records[0].node_id = node_graph.node_ids[root];

        for &leaf in &leaves {
            assert_eq!(
                node_graph.nodes[leaf].pool_id, expected_leaf_pool,
                "leaf {leaf} must read its own pool_id (the shared leaf pool)"
            );

            let parent = node_graph.node_parent(leaf);
            assert_eq!(
                parent,
                Some(root),
                "leaf {leaf}'s parent must resolve to the root"
            );
            let parent_stage = node_graph.nodes[parent.unwrap()].stage;
            assert_eq!(
                worker_records[parent_stage.0].state, root_state,
                "leaf {leaf}'s incoming state must equal the root's own outgoing state"
            );
        }
    }

    /// A full `run_forward_worker` pass over a declared K-fan: each
    /// trajectory's recorded leaf must equal `select_transition_child`'s own
    /// pinned-seed choice for that trajectory's `(iteration, global_scenario)`
    /// — never a stage-uniform node — and the pinned range must resolve more
    /// than one leaf, proving the walk is not collapsed to one node per
    /// stage (the exact restriction the singleton-frontier resolver used to
    /// impose).
    // Rationale: a real `run_forward_worker` call needs the full fixture
    // (graph, LP template, sampler, workspace) the sibling declared-K-fan
    // test above builds by hand; splitting the assembly out would scatter a
    // single-use fixture across helpers with no other caller.
    #[allow(clippy::too_many_lines)]
    #[test]
    fn run_forward_worker_k_fan_pinned_trajectories_match_selected_transitions() {
        use cobre_core::HorizonGraph;
        use cobre_core::temporal::{Node, PolicyGraphType, Transition};
        use cobre_io::StageIdResolver;

        use crate::setup::node_graph::build_node_graph;

        fn node(id: i32, stage_id: i32) -> Node {
            Node {
                id,
                stage_id,
                scenario_id: None,
                label: None,
            }
        }
        fn transition(source_id: i32, target_id: i32, probability: f64) -> Transition {
            Transition {
                source_id,
                target_id,
                probability,
                annual_discount_rate_override: None,
            }
        }

        let stochastic = make_stochastic_context_2_stages();
        let study_stage_ids = [0_i32, 1_i32];
        let resolver = StageIdResolver::from_study_stage_ids(&study_stage_ids);
        let graph = HorizonGraph {
            graph_type: PolicyGraphType::FiniteHorizon,
            annual_discount_rate: 0.0,
            nodes: vec![node(0, 0), node(1, 1), node(2, 1), node(3, 1), node(4, 1)],
            transitions: vec![
                transition(0, 1, 0.25),
                transition(0, 2, 0.25),
                transition(0, 3, 0.25),
                transition(0, 4, 0.25),
            ],
            stage_discount_rate_overrides: std::collections::BTreeMap::new(),
            season_map: None,
        };
        let node_graph = build_node_graph(&graph, 2, &resolver, &stochastic)
            .expect("declared K-fan graph must build");
        let root = NodePos(0);

        let state = state_layout(1, 0);
        let templates = vec![minimal_template_1_0(); 2];
        let initial_state = vec![0.0_f64; state.n_state];
        let fcf = FutureCostFunction::new(
            node_graph.n_pools,
            state.n_state,
            2,
            10,
            &vec![0; node_graph.n_pools],
        );
        let horizon = HorizonMode::Finite { num_stages: 2 };
        let state_boxes = permissive_state_boxes(state.n_state, 2);
        let geometry = equipment_free_geometry(&vec![0; templates.len()]);
        let fixture = StageContextFixture::new(&templates, &state_boxes, &geometry);
        let ctx = fixture.ctx();
        let study_dims = study_dims();
        let stages = make_stages_2();
        let training_ctx = TrainingContext {
            node_graph: &node_graph,
            horizon: &horizon,
            state: &state,
            cut_state_layouts: &[],
            study_dims: &study_dims,
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &stochastic,
            initial_state: &initial_state,
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            stages: &stages,
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            lag_accum_seed: &[],
            lag_weight_seed: &[],
            dcs: None,
        };
        let sampler = build_forward_sampler(ForwardSamplerConfig {
            class_schemes: ClassSchemes {
                inflow: Some(SamplingScheme::InSample),
                load: Some(SamplingScheme::InSample),
                ncs: Some(SamplingScheme::InSample),
            },
            ctx: &stochastic,
            forward_seed: stochastic.forward_seed(),
            stages: &stages,
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
        })
        .expect("sampler build must not error");

        let forward_passes = 6_usize;
        let pinned_iteration = 3_u64;
        let mut noise_tables = ForwardNoiseTables::default();
        sampler
            .rebuild_noise_tables(
                u32::try_from(pinned_iteration).expect("fits u32"),
                u32::try_from(forward_passes).expect("fits u32"),
                &[],
                &mut noise_tables,
            )
            .expect("test fixture never exceeds the Sobol dimension cap");
        let params = ForwardWorkerParams {
            forward_passes,
            total_forward_passes: forward_passes,
            n_workers: 1,
            iteration: pinned_iteration,
            fwd_offset: 0,
            root_node: root,
            ctx: &ctx,
            frozen: &templates,
            fcf: &fcf,
            training_ctx: &training_ctx,
            sampler: &sampler,
            noise_tables: &noise_tables,
        };

        let mut ws = single_workspace(MockSolver::always_ok(fixed_solution_1_0()), &state);
        // Basis store axis is NODE: the K-fan visits leaf nodes at positions
        // beyond `num_stages`, so the store must span every node, not the stages.
        let mut basis_store = BasisStore::new(forward_passes, node_graph.nodes.len());
        let mut basis_slices = basis_store.split_workers_mut(1);
        let mut basis_slice = basis_slices.remove(0);
        let mut records: Vec<TrajectoryRecord> = (0..forward_passes * 2)
            .map(|_| TrajectoryRecord {
                primal: Vec::new(),
                dual: Vec::new(),
                stage_cost: 0.0,
                node_id: NodeId(0),
                state: Vec::new(),
            })
            .collect();
        let mut per_stage_stats = vec![SolverStatsDelta::default(); 2];

        run_forward_worker(
            0,
            &mut ws,
            &mut records,
            &mut basis_slice,
            &mut per_stage_stats,
            &params,
        )
        .expect("run_forward_worker must not error");

        let successors = &node_graph.successors[root];
        let weights: Vec<f64> = successors.iter().map(|s| s.probability).collect();
        let mut resolved_leaves: std::collections::BTreeSet<NodeId> =
            std::collections::BTreeSet::new();
        #[allow(clippy::cast_possible_truncation)]
        let pinned_iteration_u32 = pinned_iteration as u32;
        for m in 0..forward_passes {
            #[allow(clippy::cast_possible_truncation)]
            let global_scenario = m as u32;
            let expected_idx = select_transition_child(
                pinned_iteration_u32,
                global_scenario,
                0,
                weights.iter().copied(),
            );
            let expected_leaf_node_id = node_graph.node_ids[successors[expected_idx].child];

            assert_eq!(
                records[m * 2].node_id,
                node_graph.node_ids[root],
                "trajectory {m}'s stage-0 record must carry the root's own node id"
            );
            assert_eq!(
                records[m * 2 + 1].node_id,
                expected_leaf_node_id,
                "trajectory {m}'s recorded leaf must match select_transition_child's own \
                 pinned-seed choice"
            );
            resolved_leaves.insert(expected_leaf_node_id);
        }
        assert!(
            resolved_leaves.len() > 1,
            "the pinned iteration/scenario range must resolve more than one leaf, or this \
             test cannot distinguish a per-trajectory walk from a collapsed stage-uniform one"
        );
    }

    // ── No graph-shape dispatch in the forward path ──────────────────────────

    /// The forward path carries no graph-shape dispatch — chain parity is
    /// degeneracy (the node-native path running on the one-node-per-stage
    /// graph), never a fork to preserved legacy code. Banned tokens are
    /// assembled from char arrays so this check is not itself a false-positive
    /// hit for the same predicate it looks for.
    #[test]
    fn forward_path_has_no_shape_selected_layout_branch() {
        let banned: Vec<String> = vec![
            ['i', 's', '_', 'c', 'h', 'a', 'i', 'n'].iter().collect(),
            [
                'n', 'o', 'd', 'e', 's', '.', 'i', 's', '_', 'e', 'm', 'p', 't', 'y', '(', ')',
            ]
            .iter()
            .collect(),
            [
                'g', 'r', 'a', 'p', 'h', '.', 'i', 's', '_', 'n', 'o', 'n', 'e', '(', ')',
            ]
            .iter()
            .collect(),
        ];
        let sources: [(&str, &str); 3] = [
            (
                "forward_pass_state.rs",
                include_str!("forward_pass_state.rs"),
            ),
            ("forward/mod.rs", include_str!("forward/mod.rs")),
            (
                "forward/stage_solve.rs",
                include_str!("forward/stage_solve.rs"),
            ),
        ];
        for (name, src) in sources {
            for token in &banned {
                assert!(
                    !src.contains(token.as_str()),
                    "{name}: forward path must not branch on a graph-shape predicate ({token})"
                );
            }
        }
    }

    // ── `require_terminal_node` validates the terminal stage ────────────────

    /// Build a minimal 2-stage chain `TrainingContext` (pool id == stage) for
    /// the `require_terminal_node` fixture below; only the `node_graph`/
    /// `horizon` fields it reads matter.
    fn terminal_boundary_test_ctx<'a>(
        stochastic: &'a StochasticContext,
        node_graph: &'a crate::setup::node_graph::NodeGraph,
        state: &'a StateSpace,
        stages: &'a [Stage],
        study_dims: &'a StudyDimensions,
        initial_state: &'a [f64],
    ) -> TrainingContext<'a> {
        TrainingContext {
            node_graph,
            horizon: &HorizonMode::Finite { num_stages: 2 },
            state,
            cut_state_layouts: &[],
            study_dims,
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic,
            initial_state,
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            stages,
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            lag_accum_seed: &[],
            lag_weight_seed: &[],
            dcs: None,
        }
    }

    /// `require_terminal_node` accepts a zero-stage horizon — there is no
    /// terminal stage to require.
    #[test]
    fn require_terminal_node_accepts_a_zero_stage_horizon() {
        let stochastic = make_stochastic_context_2_stages();
        let node_graph = crate::test_support::chain_node_graph(&stochastic);
        let state = state_layout(1, 0);
        let stages = make_stages_2();
        let study_dims = study_dims();
        let initial_state = vec![0.0; state.n_state];
        let training_ctx = TrainingContext {
            horizon: &HorizonMode::Finite { num_stages: 0 },
            ..terminal_boundary_test_ctx(
                &stochastic,
                &node_graph,
                &state,
                &stages,
                &study_dims,
                &initial_state,
            )
        };

        assert!(
            ForwardPassState::require_terminal_node(&training_ctx).is_ok(),
            "a zero-stage horizon never reaches the alive-node check"
        );
    }
}
