//! Owned scratch state and per-call inputs for the forward pass.
//!
//! [`ForwardPassState`] owns the per-worker, per-stage accumulator `Vec`s that
//! are allocated once at training-session setup and reused across every
//! iteration. It exposes a single entry point, [`ForwardPassState::run`], which
//! is the renamed body of the former free function `run_forward_pass`.
//!
//! [`ForwardPassInputs`] is a plain argument-bundle that groups all per-iteration
//! borrowed references (workspaces, basis store, contexts, records, etc.) that
//! change between calls. Splitting ownership from per-call inputs removes all
//! lifetime parameters from `ForwardPassState` and avoids variance gymnastics.
//!
//! [`ForwardWorkerParams`] bundles the read-only captures that every rayon
//! worker thread reads from: sampler, contexts, scalars, and seeds. Passing
//! `&ForwardWorkerParams` by shared reference allows rayon to fan out the
//! struct across workers without cloning.
//!
//! [`ForwardWorkerResult`] is the named return bundle from
//! [`run_forward_worker`], replacing the former anonymous 3-tuple and
//! eliminating the `clippy::type_complexity` suppression.
//!
//! ## Hot-path allocation discipline
//!
//! All `Vec` fields in `ForwardPassState` are pre-sized in
//! [`ForwardPassState::new`] and reused via `clear()` / `resize()` /
//! `extend()` inside `run`. No `Vec::new()` or `Vec::with_capacity` appears in
//! `run` or its helpers.

use std::sync::mpsc::Sender;
use std::time::Instant;

use cobre_core::{
    TrainingEvent, WORKER_TIMING_SLOT_FWD_SETUP, WORKER_TIMING_SLOT_FWD_WALL, WorkerTimingPhase,
};
use cobre_solver::{SolverInterface, SolverStatistics, StageTemplate};
use cobre_stochastic::context::ClassSchemes;
use cobre_stochastic::{
    ClassDimensions, ClassSampleRequest, ForwardSampler, ForwardSamplerConfig, SampleRequest,
    build_forward_sampler,
};
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator,
};

use crate::{
    FutureCostFunction, SddpError, StageIndexer, TrajectoryRecord,
    context::{StageContext, TrainingContext},
    forward::{ForwardResult, StageKey, partition, run_forward_stage},
    solver_stats::SolverStatsDelta,
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
    /// Baked LP templates including pre-appended prior-iteration cuts.
    pub baked: &'a [StageTemplate],
    /// Future-cost function — read-only for the forward pass.
    pub fcf: &'a FutureCostFunction,
    /// Study-level training context (horizon, indexer, stochastic model).
    pub training_ctx: &'a TrainingContext<'a>,
    /// Trajectory output records; pre-allocated by the caller.
    ///
    /// Length must equal `local_forward_passes * num_stages`.
    pub records: &'a mut [TrajectoryRecord],

    // ── Per-iteration batch scalars (formerly in `ForwardPassBatch`) ──────
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
    /// Activity-window size for the basis-reconstruction classifier (1..=31).
    pub basis_activity_window: u32,
}

impl<'a, S: SolverInterface + Send> ForwardPassInputs<'a, S> {
    /// Construct inputs from the fields of a `TrainingSession`, minus `fwd_state`.
    ///
    /// The caller is responsible for taking `&mut session.fwd_state` separately
    /// (which the borrow checker treats as a disjoint field borrow), then passing
    /// the remaining session fields here.  This collapses the multi-field struct
    /// literal into a single compact call:
    ///
    /// ```text
    /// let fwd = &mut self.fwd_state;
    /// let mut inputs = ForwardPassInputs::from_session_fields(
    ///     &mut self.fwd_pool, &mut self.basis_store, self.stage_ctx,
    ///     &mut self.scratch, self.fcf, self.training_ctx,
    ///     &self.config.cut_management, &self.ranks, &self.runtime, iteration,
    /// );
    /// let forward_result = fwd.run(&mut inputs)?;
    /// ```
    // RATIONALE: 10 args are disjoint borrows of `TrainingSession` fields required because
    // Rust NLL cannot split a single `&mut TrainingSession` borrow when `fwd_state` is also
    // borrowed mutably. Each arg maps to a distinct session field; no grouping is possible
    // without adding indirection or invalidating the disjoint-borrow design.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_session_fields(
        fwd_pool: &'a mut crate::workspace::WorkspacePool<S>,
        basis_store: &'a mut BasisStore,
        ctx: &'a StageContext<'a>,
        scratch: &'a mut crate::training_session::iteration_scratch::IterationScratch,
        fcf: &'a FutureCostFunction,
        training_ctx: &'a TrainingContext<'a>,
        cut_mgmt: &'a crate::CutManagementConfig,
        ranks: &crate::training_session::rank_distribution::RankDistribution,
        runtime: &'a crate::training_session::runtime::RuntimeHandles,
        iteration: u64,
    ) -> Self {
        let fwd_record_len = ranks.my_actual_fwd * training_ctx.horizon.num_stages();
        Self {
            workspaces: &mut fwd_pool.workspaces,
            basis_store,
            ctx,
            baked: &scratch.baked_templates,
            fcf,
            training_ctx,
            records: &mut scratch.records[..fwd_record_len],
            local_forward_passes: ranks.my_actual_fwd,
            total_forward_passes: ranks.num_total_forward_passes,
            iteration,
            fwd_offset: ranks.my_fwd_offset,
            event_sender: runtime.event_sender(),
            basis_activity_window: cut_mgmt.basis_activity_window,
        }
    }
}

/// Read-only captures shared across all rayon workers in the forward pass.
///
/// Built once by [`ForwardPassState::run`] before the parallel region and
/// passed by shared reference (`&ForwardWorkerParams`) to every
/// [`run_forward_worker`] invocation. All fields are either scalars or
/// immutable borrows; no field is mutated inside the worker.
///
/// The struct is not generic over the solver type `S` because no field holds an
/// `S`-typed value. The `SolverWorkspace<S>` is passed as a separate `ws`
/// argument to [`run_forward_worker`].
pub(crate) struct ForwardWorkerParams<'a> {
    /// Number of forward passes assigned to this rank (local partition size).
    pub forward_passes: usize,
    /// Total forward passes across all MPI ranks (for seed derivation).
    pub total_forward_passes: usize,
    /// Number of stages in the study horizon.
    pub num_stages: usize,
    /// Number of rayon worker threads on this rank.
    pub n_workers: usize,
    /// Current training iteration index (1-based).
    pub iteration: u64,
    /// Global index of this rank's first forward pass (for seed derivation).
    pub fwd_offset: usize,
    /// Activity-window size for the basis-reconstruction classifier (1..=31).
    pub basis_activity_window: u32,
    /// True when the last stage has warm-start (boundary) cuts.
    pub terminal_has_boundary_cuts: bool,
    /// Noise dimension for worker-local sampling buffers (`OutOfSample` path).
    pub noise_dim: usize,
    /// Initial reservoir state shared across all workers.
    pub initial_state: &'a [f64],
    /// Lag-accumulator seed values at trajectory start (empty → zero-init).
    pub recent_accum_seed: &'a [f64],
    /// Lag-accumulator weight seed at trajectory start.
    pub recent_weight_seed: f64,
    /// Stage-dimension indexer (state, hydro, lag counts).
    pub indexer: &'a StageIndexer,
    /// Stage-level LP context (templates, row counts, noise scales).
    pub ctx: &'a StageContext<'a>,
    /// Baked LP templates including pre-appended prior-iteration cuts.
    pub baked: &'a [StageTemplate],
    /// Future-cost function — read-only for the forward pass.
    pub fcf: &'a FutureCostFunction,
    /// Study-level training context (horizon, indexer, stochastic model).
    pub training_ctx: &'a TrainingContext<'a>,
    /// Forward sampler that drives per-scenario-per-stage noise generation.
    pub sampler: &'a ForwardSampler<'a>,
}

/// Return bundle from [`run_forward_worker`].
///
/// Replaces the anonymous 3-tuple `(Vec<f64>, u64, Vec<SolverStatsDelta>)`
/// that previously required a `#[allow(clippy::type_complexity)]` annotation.
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
    /// Number of forward-pass scenarios on this rank.
    forward_passes: usize,
    /// Number of stages in the study horizon.
    num_stages: usize,
    /// Wall-clock duration of the parallel region in milliseconds.
    parallel_wall_ms: u64,
    /// `Instant` captured at the start of the entire `run()` call.
    start: Instant,
}

/// Owned scratch buffers for the forward pass, allocated once and reused.
///
/// `ForwardPassState` is constructed once by `TrainingSession::new` and stored
/// as a field on `TrainingSession`. The buffers are pre-sized from the study
/// dimensions and reused across every iteration via `clear()` / `resize()` /
/// `extend()`. No allocation occurs on the hot path.
///
/// Per-iteration inputs (workspaces, basis store, stage context, records, etc.)
/// are passed via [`ForwardPassInputs`] at each `run()` call.
// The `worker_` prefix is intentional: all fields are per-worker scratch buffers.
#[allow(clippy::struct_field_names)]
pub(crate) struct ForwardPassState {
    /// Per-worker, per-stage solver-stats accumulators.
    ///
    /// Reused across iterations: cleared at the start of each `run()`.
    /// Shape: `n_workers × num_stages`.
    worker_stage_stats: Vec<Vec<SolverStatsDelta>>,

    /// Per-worker solver statistics snapshot taken **before** the parallel region.
    ///
    /// Cleared and repopulated at the start of each `run()`.
    worker_stats_before: Vec<SolverStatistics>,

    /// Per-worker solver statistics snapshot taken **after** the parallel region.
    ///
    /// Cleared and repopulated after the parallel region each `run()`.
    worker_stats_after: Vec<SolverStatistics>,

    /// Per-worker solver-statistics delta (after − before).
    ///
    /// Cleared and repopulated after the parallel region each `run()`.
    worker_deltas: Vec<SolverStatsDelta>,

    /// Per-worker wall-time total for load-imbalance decomposition.
    ///
    /// Cleared and repopulated after the parallel region each `run()`.
    worker_totals: Vec<f64>,
}

impl ForwardPassState {
    /// Allocate all scratch buffers sized for the given study dimensions.
    ///
    /// # Parameters
    ///
    /// - `n_workers`: number of rayon worker threads on this rank.
    /// - `num_stages`: total number of stages in the study horizon.
    pub(crate) fn new(n_workers: usize, num_stages: usize) -> Self {
        let worker_stage_stats = (0..n_workers)
            .map(|_| {
                (0..num_stages)
                    .map(|_| SolverStatsDelta::default())
                    .collect()
            })
            .collect();
        Self {
            worker_stage_stats,
            worker_stats_before: Vec::with_capacity(n_workers),
            worker_stats_after: Vec::with_capacity(n_workers),
            worker_deltas: Vec::with_capacity(n_workers),
            worker_totals: Vec::with_capacity(n_workers),
        }
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
    /// `build_forward_sampler` fails.
    ///
    /// # Panics (debug builds only)
    ///
    /// Panics if any of the following debug preconditions are violated:
    ///
    /// - `inputs.records.len() != inputs.local_forward_passes * num_stages`
    /// - `inputs.training_ctx.initial_state.len() != indexer.n_state`
    /// - `inputs.baked.len() != num_stages`
    pub(crate) fn run<S: SolverInterface + Send>(
        &mut self,
        inputs: &mut ForwardPassInputs<'_, S>,
    ) -> Result<ForwardResult, SddpError> {
        let training_ctx = inputs.training_ctx;
        let TrainingContext {
            horizon,
            indexer,
            stochastic,
            initial_state,
            recent_accum_seed,
            recent_weight_seed,
            ..
        } = training_ctx;
        let recent_weight_seed = *recent_weight_seed;

        let num_stages = horizon.num_stages();
        let forward_passes = inputs.local_forward_passes;

        debug_assert_eq!(inputs.records.len(), forward_passes * num_stages);
        debug_assert_eq!(initial_state.len(), indexer.n_state);
        debug_assert_eq!(
            inputs.baked.len(),
            num_stages,
            "baked templates length mismatch: expected {num_stages}, got {}",
            inputs.baked.len()
        );

        let sampler = build_forward_sampler(ForwardSamplerConfig {
            class_schemes: ClassSchemes {
                inflow: Some(training_ctx.inflow_scheme),
                load: Some(training_ctx.load_scheme),
                ncs: Some(training_ctx.ncs_scheme),
            },
            ctx: stochastic,
            stages: training_ctx.stages,
            dims: ClassDimensions {
                n_hydros: stochastic.n_hydros(),
                n_load_buses: stochastic.n_load_buses(),
                n_ncs: stochastic.n_stochastic_ncs(),
            },
            historical_library: training_ctx.historical_library,
            external_inflow_library: training_ctx.external_inflow_library,
            external_load_library: training_ctx.external_load_library,
            external_ncs_library: training_ctx.external_ncs_library,
        })?;

        let n_workers = inputs.workspaces.len().max(1);
        let start = Instant::now();

        // Partition record slice into per-worker sub-slices.
        let mut remaining: &mut [TrajectoryRecord] = inputs.records;
        let mut record_slices: Vec<&mut [TrajectoryRecord]> = Vec::with_capacity(n_workers);
        for w in 0..n_workers {
            let (start_m, end_m) = partition(forward_passes, n_workers, w);
            let (slice, rest) = remaining.split_at_mut((end_m - start_m) * num_stages);
            record_slices.push(slice);
            remaining = rest;
        }
        let basis_slices = inputs.basis_store.split_workers_mut(n_workers);

        // Noise dimension for worker-local sampling buffers (`OutOfSample` path).
        let noise_dim = stochastic.dim();

        // True when the last study stage has warm-start (boundary) cuts.
        let terminal_has_boundary_cuts =
            num_stages > 0 && inputs.fcf.pools[num_stages - 1].warm_start_count > 0;

        // Re-size per-worker per-stage stat accumulators to match the current
        // worker count (may differ from the count at new() if the pool shrank).
        // Each entry is reset to default.
        self.worker_stage_stats.clear();
        for _ in 0..n_workers {
            let stage_vec: Vec<SolverStatsDelta> = (0..num_stages)
                .map(|_| SolverStatsDelta::default())
                .collect();
            self.worker_stage_stats.push(stage_vec);
        }

        // Collect per-worker snapshots before the parallel region.
        self.worker_stats_before.clear();
        self.worker_stats_before
            .extend(inputs.workspaces.iter().map(|ws| ws.solver.statistics()));

        // Reset per-worker timing accumulators at the iteration boundary.
        for ws in inputs.workspaces.iter_mut() {
            ws.worker_timing_buf.fill(0.0);
        }

        // Parallel region: each worker processes its scenario partition.
        let parallel_start = Instant::now();
        // Temporarily drain `worker_stage_stats` into a Vec consumed by the
        // parallel closure. After the closure completes we receive the updated
        // stats back via the ForwardWorkerResult.
        let worker_stage_stats_for_par: Vec<Vec<SolverStatsDelta>> =
            std::mem::take(&mut self.worker_stage_stats);

        let params = ForwardWorkerParams {
            forward_passes,
            total_forward_passes: inputs.total_forward_passes,
            num_stages,
            n_workers,
            iteration: inputs.iteration,
            fwd_offset: inputs.fwd_offset,
            basis_activity_window: inputs.basis_activity_window,
            terminal_has_boundary_cuts,
            noise_dim,
            initial_state,
            recent_accum_seed,
            recent_weight_seed,
            indexer,
            ctx: inputs.ctx,
            baked: inputs.baked,
            fcf: inputs.fcf,
            training_ctx,
            sampler: &sampler,
        };
        // `params` is shared across all rayon workers via `&ForwardWorkerParams`.
        // All fields are either `Copy` scalars or `&'a T` shared references, so
        // `ForwardWorkerParams` is `Sync` by Rust's automatic derivation rules.

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

        // Capture parallel region wall-clock before sequential post-processing.
        #[allow(clippy::cast_possible_truncation)]
        let parallel_wall_ms = parallel_start.elapsed().as_millis() as u64;

        let ppc = PostProcessContext {
            n_workers,
            forward_passes,
            num_stages,
            parallel_wall_ms,
            start,
        };
        self.post_process_worker_results(inputs, worker_results, &ppc)
    }

    /// Sequential post-processing after the rayon parallel region.
    ///
    /// Collects per-worker solver-statistic snapshots, decomposes timing
    /// overhead, emits [`TrainingEvent::WorkerTiming`] events, and merges
    /// per-worker cost vectors and stage stats into the final [`ForwardResult`].
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
            forward_passes,
            num_stages,
            parallel_wall_ms,
            start,
        } = *ppc;

        // Collect per-worker snapshots after the parallel region and decompose overhead.
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

        // Per-worker elapsed: solve + setup phases.
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

        // Accumulate per-worker forward-setup time into timing buf slot FWD_SETUP.
        for (ws, delta) in inputs.workspaces.iter_mut().zip(&self.worker_deltas) {
            ws.worker_timing_buf[WORKER_TIMING_SLOT_FWD_SETUP] +=
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

        // Merge per-worker cost vectors in global scenario index order (canonical).
        // Simultaneously merge per-stage stats by summing element-wise across workers.
        let mut scenario_costs = Vec::with_capacity(forward_passes);
        let mut lp_solves = 0u64;
        let mut stage_stats: Vec<SolverStatsDelta> = (0..num_stages)
            .map(|_| SolverStatsDelta::default())
            .collect();
        for result in worker_results {
            let ForwardWorkerResult {
                trajectory_costs: worker_costs,
                local_solves: w_solves,
                per_stage_stats: worker_stage_stats,
            } = result?;
            scenario_costs.extend(worker_costs);
            lp_solves += w_solves;
            for (dst, src) in stage_stats.iter_mut().zip(&worker_stage_stats) {
                SolverStatsDelta::accumulate_into(dst, src);
            }
        }

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

/// Execute the forward pass for one rayon worker's scenario partition.
///
/// This function is the extracted body of the anonymous closure that previously
/// lived inside [`ForwardPassState::run`]. It processes `n_local` scenarios
/// (the worker's slice of the global forward-pass batch) through every stage,
/// accumulating trajectory costs and per-stage solver statistics.
///
/// # Parameters
///
/// - `w`: rayon worker index (0-based), used to derive the local scenario range
///   via [`partition`].
/// - `ws`: mutable reference to this worker's [`SolverWorkspace`].
/// - `worker_records`: mutable slice of [`TrajectoryRecord`] for this worker's
///   scenarios. Length is `n_local * num_stages`.
/// - `basis_slice`: mutable view into the basis warm-start store for this worker.
/// - `per_stage_stats`: mutable slice of per-stage [`SolverStatsDelta`]
///   accumulators. Length is `num_stages`. Values are accumulated in-place.
/// - `params`: shared read-only bundle of all captures for this parallel region.
///
/// # Errors
///
/// Propagates `Err(SddpError::Stochastic(_))` from `sampler.sample(...)` and
/// `Err(SddpError::Infeasible/Solver(_))` from [`run_forward_stage`].
pub(crate) fn run_forward_worker<S: SolverInterface + Send>(
    w: usize,
    ws: &mut SolverWorkspace<S>,
    worker_records: &mut [TrajectoryRecord],
    basis_slice: &mut BasisStoreSliceMut<'_>,
    per_stage_stats: &mut [SolverStatsDelta],
    params: &ForwardWorkerParams<'_>,
) -> Result<ForwardWorkerResult, SddpError> {
    let worker_wall_start = Instant::now();
    let (start_m, end_m) = partition(params.forward_passes, params.n_workers, w);
    let n_local = end_m - start_m;
    let mut trajectory_costs = vec![0.0_f64; n_local];
    let local_solve_count_before = ws.solver.statistics().solve_count;
    // Sampling scratch: lives here (not in ws) to avoid borrow conflicts
    // when run_forward_stage borrows ws while raw_noise is still live.
    let mut raw_noise_buf = vec![0.0_f64; params.noise_dim];
    #[allow(clippy::cast_possible_truncation)]
    let mut perm_scratch = vec![0_usize; (params.total_forward_passes).max(1)];
    #[allow(clippy::cast_possible_truncation)]
    let total_scenarios_u32 = params.total_forward_passes as u32;

    for t in 0..params.num_stages {
        let cum_d = params
            .ctx
            .cumulative_discount_factors
            .get(t)
            .copied()
            .unwrap_or(1.0);

        for (local_m, m) in (start_m..end_m).enumerate() {
            // Reload model per scenario to ensure deterministic LP state across
            // thread assignments.
            ws.solver.load_model(&params.baked[t]);
            ws.current_state.clear();
            let src: &[f64] = if t == 0 {
                params.initial_state
            } else {
                &worker_records[local_m * params.num_stages + (t - 1)].state
            };
            ws.current_state.extend_from_slice(src);

            // Seed (or zero) the lag accumulator at trajectory start.
            if t == 0 {
                if params.recent_accum_seed.is_empty() {
                    ws.scratch.lag_accumulator.iter_mut().for_each(|v| *v = 0.0);
                    ws.scratch.lag_weight_accum = 0.0;
                } else {
                    ws.scratch.lag_accumulator[..params.recent_accum_seed.len()]
                        .copy_from_slice(params.recent_accum_seed);
                    ws.scratch.lag_weight_accum = params.recent_weight_seed;
                }
                // Reset downstream accumulator at trajectory start.
                ws.scratch
                    .downstream_accumulator
                    .iter_mut()
                    .for_each(|v| *v = 0.0);
                ws.scratch.downstream_weight_accum = 0.0;
                ws.scratch
                    .downstream_completed_lags
                    .iter_mut()
                    .for_each(|v| *v = 0.0);
                ws.scratch.downstream_n_completed = 0;
            }

            let global_scenario = params.fwd_offset + m;
            #[allow(clippy::cast_possible_truncation)]
            let (i32, s32, t32) = (params.iteration as u32, global_scenario as u32, t as u32);

            if t == 0 {
                let class_req = ClassSampleRequest {
                    iteration: i32,
                    scenario: s32,
                    stage: 0,
                    stage_idx: 0,
                    total_scenarios: total_scenarios_u32,
                    noise_group_id: 0,
                };
                params.sampler.apply_initial_state(
                    &class_req,
                    &mut ws.current_state,
                    params.indexer.inflow_lags.start,
                );
            }
            let noise = params.sampler.sample(SampleRequest {
                iteration: i32,
                scenario: s32,
                stage: t32,
                stage_idx: t,
                noise_buf: &mut raw_noise_buf,
                perm_scratch: &mut perm_scratch,
                total_scenarios: total_scenarios_u32,
                noise_group_id: params.ctx.noise_group_id_at(t),
            })?;
            let raw_noise = noise.as_slice();
            let key = StageKey {
                t,
                m,
                local_m,
                num_stages: params.num_stages,
                iteration: params.iteration,
                raw_noise,
                basis_row_capacity: params.baked[t].num_rows,
                terminal_has_boundary_cuts: params.terminal_has_boundary_cuts,
                pool: &params.fcf.pools[t],
                baked_template: &params.baked[t],
                basis_activity_window: params.basis_activity_window,
            };
            // Snapshot solver statistics before the stage solve so the
            // per-stage delta can be accumulated without hot-path allocation.
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
            SolverStatsDelta::accumulate_into(&mut per_stage_stats[t], &stage_delta);
            trajectory_costs[local_m] += cum_d * stage_cost;
        }
    }

    let local_solves = ws.solver.statistics().solve_count - local_solve_count_before;
    ws.worker_timing_buf[WORKER_TIMING_SLOT_FWD_WALL] +=
        worker_wall_start.elapsed().as_secs_f64() * 1_000.0;
    Ok(ForwardWorkerResult {
        trajectory_costs,
        local_solves,
        per_stage_stats: per_stage_stats.to_vec(),
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
    use cobre_core::{Bus, DeficitSegment, EntityId, SystemBuilder};
    use cobre_solver::{
        Basis, LpSolution, RowBatch, SolverError, SolverInterface, SolverStatistics, StageTemplate,
    };
    use cobre_stochastic::StochasticContext;
    use cobre_stochastic::context::{ClassSchemes, OpeningTreeInputs, build_stochastic_context};

    use super::*;
    use crate::{
        FutureCostFunction, HorizonMode, InflowNonNegativityMethod, StageIndexer, TrajectoryRecord,
        context::{StageContext, TrainingContext},
        workspace::{BackwardAccumulators, BasisStore, SolverWorkspace},
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
        fn get_basis(&mut self, _out: &mut Basis) {}
        fn statistics(&self) -> SolverStatistics {
            self.stats.clone()
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
            n_transfer: 0,
            n_dual_relevant: 1,
            n_hydro: 1,
            max_par_order: 0,
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

    fn single_workspace(solver: MockSolver, indexer: &StageIndexer) -> SolverWorkspace<MockSolver> {
        SolverWorkspace {
            rank: 0,
            worker_id: 0,
            solver,
            patch_buf: crate::lp_builder::PatchBuffer::new(
                indexer.hydro_count,
                indexer.max_par_order,
                0,
                0,
            ),
            current_state: Vec::with_capacity(indexer.n_state),
            scratch: crate::workspace::ScratchBuffers {
                noise_buf: Vec::with_capacity(indexer.hydro_count),
                inflow_m3s_buf: Vec::with_capacity(indexer.hydro_count),
                lag_matrix_buf: Vec::with_capacity(indexer.max_par_order * indexer.hydro_count),
                par_inflow_buf: Vec::with_capacity(indexer.hydro_count),
                eta_floor_buf: Vec::with_capacity(indexer.hydro_count),
                zero_targets_buf: vec![0.0_f64; indexer.hydro_count],
                ncs_col_upper_buf: Vec::new(),
                ncs_col_lower_buf: Vec::new(),
                ncs_col_indices_buf: Vec::new(),
                load_rhs_buf: Vec::new(),
                row_lower_buf: Vec::new(),
                z_inflow_rhs_buf: Vec::new(),
                effective_eta_buf: Vec::new(),
                unscaled_primal: Vec::new(),
                unscaled_dual: Vec::new(),
                lag_accumulator: vec![],
                lag_weight_accum: 0.0,
                downstream_accumulator: Vec::new(),
                downstream_weight_accum: 0.0,
                downstream_completed_lags: Vec::new(),
                downstream_n_completed: 0,
                recon_slot_lookup: Vec::new(),
                promotion_scratch: crate::basis_reconstruct::PromotionScratch::default(),
            },
            scratch_basis: Basis::new(0, 0),
            backward_accum: BackwardAccumulators::default(),
            worker_timing_buf: [0.0_f64; 16],
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
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 1000.0,
            }],
            excess_cost: 0.0,
        };
        let hydro = Hydro {
            id: EntityId(1),
            name: "H1".to_string(),
            bus_id: EntityId(0),
            downstream_id: None,
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: 0.0,
            max_storage_hm3: 100.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: HydroGenerationModel::ConstantProductivity {
                productivity_mw_per_m3s: 1.0,
            },
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 100.0,
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
                fpha_turbined_cost: 0.0,
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
        let stages: Vec<Stage> = (0..2).map(make_stage).collect();
        let inflow_models: Vec<InflowModel> = (0_i32..2)
            .map(|idx| InflowModel {
                hydro_id: EntityId(1),
                stage_id: idx,
                mean_m3s: 100.0,
                std_m3s: 30.0,
                ar_coefficients: vec![],
                residual_std_ratio: 1.0,
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

    // ── Tests ──────────────────────────────────────────────────────────────

    #[test]
    fn forward_pass_state_new_preallocates_per_worker_buffers() {
        let state = ForwardPassState::new(3, 5);
        assert_eq!(state.worker_stage_stats.len(), 3);
        for inner in &state.worker_stage_stats {
            assert_eq!(inner.len(), 5);
        }
        // Per-worker stat Vecs are pre-allocated with the given capacity.
        assert_eq!(state.worker_stats_before.capacity(), 3);
        assert_eq!(state.worker_stats_after.capacity(), 3);
        assert_eq!(state.worker_deltas.capacity(), 3);
        assert_eq!(state.worker_totals.capacity(), 3);
    }

    /// Minimal 2-stage, 1-hydro, 2-scenario fixture driven through
    /// `ForwardPassState::run`. Asserts that the result carries exactly 2
    /// scenario costs (one per forward pass).
    #[test]
    #[allow(clippy::too_many_lines)]
    fn forward_pass_state_run_produces_expected_scenario_count() {
        let n_stages = 2_usize;
        let n_scenarios = 2_usize;

        let indexer = StageIndexer::new(1, 0); // 1 hydro, 0 lag order
        let stochastic = make_stochastic_context_2_stages();
        let stages = make_stages_2();

        let solution = fixed_solution_1_0();
        let solver = MockSolver::always_ok(solution);

        let templates = vec![minimal_template_1_0(); n_stages];
        // base_rows[t] is the index within template.row_lower where the inflow
        // constraint starts. The minimal template has num_rows=1, so base_row=0.
        let base_rows = vec![0_usize; n_stages];
        let initial_state = vec![0.0_f64; indexer.n_state];
        // noise_scale layout: [stage * n_hydros + hydro]. 2 stages × 1 hydro → 2 entries.
        let noise_scale = vec![0.0_f64; n_stages * indexer.hydro_count];

        let fcf = FutureCostFunction::new(n_stages, indexer.n_state, 2, 10, &vec![0; n_stages]);
        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };

        let ctx = StageContext {
            templates: &templates,
            base_rows: &base_rows,
            noise_scale: &noise_scale,
            n_hydros: 1,
            n_load_buses: 0,
            load_balance_row_starts: &[],
            load_bus_indices: &[],
            block_counts_per_stage: &[],
            ncs_max_gen: &[],
            discount_factors: &[],
            cumulative_discount_factors: &[],
            stage_lag_transitions: &[],
            noise_group_ids: &[],
            downstream_par_order: 0,
        };
        let training_ctx = TrainingContext {
            horizon: &horizon,
            indexer: &indexer,
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
            recent_accum_seed: &[],
            recent_weight_seed: 0.0,
        };

        let mut workspaces = vec![single_workspace(solver, &indexer)];
        let mut basis_store = BasisStore::new(n_scenarios, n_stages);
        let mut records: Vec<TrajectoryRecord> = (0..n_scenarios * n_stages)
            .map(|_| TrajectoryRecord {
                primal: Vec::new(),
                dual: Vec::new(),
                stage_cost: 0.0,
                state: Vec::new(),
            })
            .collect();

        let mut state = ForwardPassState::new(1, n_stages);
        let mut inputs = ForwardPassInputs {
            workspaces: &mut workspaces,
            basis_store: &mut basis_store,
            ctx: &ctx,
            baked: &templates,
            fcf: &fcf,
            training_ctx: &training_ctx,
            records: &mut records,
            local_forward_passes: n_scenarios,
            total_forward_passes: n_scenarios,
            iteration: 1,
            fwd_offset: 0,
            event_sender: None,
            basis_activity_window: crate::basis_reconstruct::DEFAULT_BASIS_ACTIVITY_WINDOW,
        };

        let result = state.run(&mut inputs).expect("forward pass must not error");

        assert_eq!(
            result.scenario_costs.len(),
            n_scenarios,
            "result must carry one cost per forward-pass scenario"
        );
    }

    /// Verify that `run_forward_worker` produces exactly `n_local` trajectory
    /// costs for the worker's scenario partition.
    #[test]
    #[allow(clippy::too_many_lines)]
    fn run_forward_worker_produces_expected_trajectory_costs() {
        let n_stages = 2_usize;
        let n_scenarios = 2_usize;

        let indexer = StageIndexer::new(1, 0);
        let stochastic = make_stochastic_context_2_stages();
        let stages = make_stages_2();

        let solution = fixed_solution_1_0();
        let solver = MockSolver::always_ok(solution);

        let templates = vec![minimal_template_1_0(); n_stages];
        let base_rows = vec![0_usize; n_stages];
        let initial_state = vec![0.0_f64; indexer.n_state];
        let noise_scale = vec![0.0_f64; n_stages * indexer.hydro_count];

        let fcf = FutureCostFunction::new(n_stages, indexer.n_state, 2, 10, &vec![0; n_stages]);
        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };

        let ctx = StageContext {
            templates: &templates,
            base_rows: &base_rows,
            noise_scale: &noise_scale,
            n_hydros: 1,
            n_load_buses: 0,
            load_balance_row_starts: &[],
            load_bus_indices: &[],
            block_counts_per_stage: &[],
            ncs_max_gen: &[],
            discount_factors: &[],
            cumulative_discount_factors: &[],
            stage_lag_transitions: &[],
            noise_group_ids: &[],
            downstream_par_order: 0,
        };
        let training_ctx = TrainingContext {
            horizon: &horizon,
            indexer: &indexer,
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
            recent_accum_seed: &[],
            recent_weight_seed: 0.0,
        };

        let sampler = build_forward_sampler(ForwardSamplerConfig {
            class_schemes: ClassSchemes {
                inflow: Some(SamplingScheme::InSample),
                load: Some(SamplingScheme::InSample),
                ncs: Some(SamplingScheme::InSample),
            },
            ctx: &stochastic,
            stages: &stages,
            dims: ClassDimensions {
                n_hydros: stochastic.n_hydros(),
                n_load_buses: stochastic.n_load_buses(),
                n_ncs: stochastic.n_stochastic_ncs(),
            },
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
        })
        .expect("sampler build must not error");

        let params = ForwardWorkerParams {
            forward_passes: n_scenarios,
            total_forward_passes: n_scenarios,
            num_stages: n_stages,
            n_workers: 1,
            iteration: 1,
            fwd_offset: 0,
            basis_activity_window: crate::basis_reconstruct::DEFAULT_BASIS_ACTIVITY_WINDOW,
            terminal_has_boundary_cuts: false,
            noise_dim: stochastic.dim(),
            initial_state: &initial_state,
            recent_accum_seed: &[],
            recent_weight_seed: 0.0,
            indexer: &indexer,
            ctx: &ctx,
            baked: &templates,
            fcf: &fcf,
            training_ctx: &training_ctx,
            sampler: &sampler,
        };

        let mut ws = single_workspace(solver, &indexer);
        let mut basis_store = BasisStore::new(n_scenarios, n_stages);
        let mut basis_slices = basis_store.split_workers_mut(1);
        let mut basis_slice = basis_slices.remove(0);
        let mut records: Vec<TrajectoryRecord> = (0..n_scenarios * n_stages)
            .map(|_| TrajectoryRecord {
                primal: Vec::new(),
                dual: Vec::new(),
                stage_cost: 0.0,
                state: Vec::new(),
            })
            .collect();
        let mut per_stage_stats: Vec<SolverStatsDelta> =
            (0..n_stages).map(|_| SolverStatsDelta::default()).collect();

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
            n_scenarios,
            "worker 0 owns all scenarios when n_workers=1; expected {n_scenarios} costs"
        );
    }
}
