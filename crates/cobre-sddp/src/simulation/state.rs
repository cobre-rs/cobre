//! Simulation state management and entry point.
//!
//! [`SimulationState`] owns re-freeze scratch buffers allocated once per run.
//! [`SimulationInputs`] bundles per-call borrowed inputs (no per-scenario allocation).
//!
//! The training path passes `frozen_templates: Some(..)`; the checkpoint path
//! passes `None`, because checkpoints do not store frozen templates, so
//! [`refreeze_templates_if_needed`] rebuilds them once at startup.
//!
//! ## Hot-path allocation discipline
//!
//! No allocations occur per scenario or per stage during the inner loops.
//! The re-freeze allocation in [`refreeze_templates_if_needed`] is a one-time
//! setup amortised across the simulation's per-scenario LP solves.

use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Instant;

use cobre_comm::Communicator;
use cobre_core::TrainingEvent;
use cobre_solver::ActiveProfile;
use cobre_solver::FreezeScratch;
use cobre_solver::freeze_rows_into_template;
use cobre_solver::{RowBatch, SolverInterface, StageTemplate};
use cobre_stochastic::context::ClassSchemes;
use cobre_stochastic::{
    ForwardNoiseTables, ForwardSampler, ForwardSamplerConfig, build_forward_sampler,
};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};

use crate::simulation::pipeline::SimScenarioLoadSpec;
use crate::solver_phase::Phase::Simulation;
use crate::{
    FutureCostFunction,
    context::{StageContext, TrainingContext},
    cut::row::build_cut_row_batch_into,
    setup::node_graph::{NodePos, StageIdx, Traversal},
    simulation::{
        config::SimulationConfig,
        enumerated::run_enumerated_simulation,
        error::SimulationError,
        extraction::assign_scenarios,
        pipeline::{
            SIMULATION_ITERATION, ScenarioIds, SimLookups, SimulationOutputSpec,
            SimulationRunResult, WorkerCosts, WorkerStats, dispatch_scenario_result,
            emit_sim_progress, process_scenario_stages,
        },
    },
    solve::partition,
    solver_stats::SolverStatsDelta,
    workspace::{CapturedBasis, SolverWorkspace},
};

/// Per-call argument bundle for [`SimulationState::run`].
///
/// Groups the borrowed inputs that vary between calls; owned scratch buffers
/// (re-freeze intermediates) live on [`SimulationState`].
pub(crate) struct SimulationInputs<'a, S: SolverInterface + Send, C> {
    /// Solver workspaces (one per rayon worker thread).
    pub workspaces: &'a mut [SolverWorkspace<S>],
    /// Stage-level LP context (templates, row counts, noise scales).
    pub ctx: &'a StageContext<'a>,
    /// Future-cost function — read-only for the simulation pass.
    pub fcf: &'a FutureCostFunction,
    /// Study-level training context (horizon, indexer, stochastic model).
    pub training_ctx: &'a TrainingContext<'a>,
    /// Simulation configuration (scenario count, I/O channel capacity).
    pub config: &'a SimulationConfig,
    /// Output-channel and extraction metadata for streaming scenario results.
    pub output: SimulationOutputSpec<'a>,
    /// Pre-frozen LP templates from training. `None` triggers local re-freeze.
    pub frozen_templates: Option<&'a [StageTemplate]>,
    /// Warm-start basis captured from the training checkpoint, one entry per
    /// canonical `NodeGraph` position — never per stage, so a branching
    /// simulation warm-starts from the visited node's own basis instead of
    /// whichever node's basis happens to land at that stage index.
    pub node_bases: &'a [Option<CapturedBasis>],
    /// MPI communicator.
    pub comm: &'a C,
    /// The resolved simulation-traversal axis: [`SimulationState::run`] forks
    /// on this directly — `Sampled` keeps the existing per-scenario loop,
    /// `Enumerated` dispatches to the node-native census driver.
    pub traversal: &'a Traversal,
}

/// Read-only captures shared by reference across all rayon workers; the mutable
/// per-worker [`SolverWorkspace`] rides on a separate `ws` argument.
///
/// Lifetime `'w` is [`SimulationState::run`]'s body scope, not `inputs`:
/// `scenarios_complete` and `sampler` are run-local, sound because `par_iter_mut`
/// joins before `run` returns.
pub(crate) struct SimWorkerParams<'w> {
    /// Stage-level LP context (templates, row counts, noise scales).
    ctx: &'w StageContext<'w>,
    /// Future-cost function — read-only for the simulation pass.
    fcf: &'w FutureCostFunction,
    /// Study-level training context (horizon, indexer, stochastic model).
    training_ctx: &'w TrainingContext<'w>,
    /// Output-channel and extraction metadata for streaming scenario results.
    output: &'w SimulationOutputSpec<'w>,
    /// Simulation configuration (scenario count, I/O channel capacity).
    config: &'w SimulationConfig,
    /// Warm-start basis cache, one entry per canonical `NodeGraph` position
    /// (see [`SimulationInputs::node_bases`]).
    node_bases: &'w [Option<CapturedBasis>],
    /// Resolved frozen LP templates (caller-supplied or locally re-frozen).
    frozen_templates: &'w [StageTemplate],
    /// Shared completion counter scaled to a global progress estimate.
    scenarios_complete: &'w AtomicU32,
    /// Wall-clock start of the simulation, for progress elapsed-time reporting.
    sim_start: Instant,
    /// Rank-local scenario count to partition across workers.
    local_count: usize,
    /// Number of rayon worker threads on this rank.
    n_workers: usize,
    /// Global index of this rank's first scenario (for `scenario_id` derivation).
    scenario_start: usize,
    /// Forward sampler that drives per-scenario-per-stage noise generation.
    sampler: &'w ForwardSampler<'w>,
    /// Per-run scenario-invariant tables backing `sampler`'s `OutOfSample` draws.
    noise_tables: &'w ForwardNoiseTables,
    /// Number of MPI ranks, scaling rank-local progress to a global estimate.
    world_size: u32,
    /// The stage-0 root's canonical `NodeGraph` position — every scenario's
    /// sampled walk starts here.
    root_node: NodePos,
}

/// Owned scratch state for one simulation run.
///
/// Typically created immediately before calling [`SimulationState::run`] and
/// discarded afterwards. The re-freeze buffers stay empty when the caller
/// always provides `frozen_templates`.
pub(crate) struct SimulationState {
    /// Owned frozen templates built by the lazy re-freeze branch (when
    /// `frozen_templates` is `None`).
    owned_frozen: Option<Vec<StageTemplate>>,
    /// Row-batch scratch for the lazy re-freeze loop.
    freeze_batch: RowBatch,
    /// Reusable scratch for `freeze_rows_into_template`, reused across re-freeze stages.
    freeze_scratch: FreezeScratch,
    /// Resolved simulation solver profile applied at [`Self::run`] entry.
    /// Defaults to `Phase::Simulation.profile()`; override with
    /// [`Self::set_profile`] before the first `run()` call.
    profile: ActiveProfile,

    /// Scenario-invariant per-class noise tables, rebuilt once per run in
    /// [`Self::run`] and shared by reference across every worker's draws.
    noise_tables: ForwardNoiseTables,
}

impl SimulationState {
    /// Construct a new `SimulationState` with empty re-freeze buffers.
    #[must_use]
    pub(crate) fn new(_num_stages: usize) -> Self {
        Self {
            owned_frozen: None,
            freeze_batch: RowBatch {
                num_rows: 0,
                row_starts: Vec::new(),
                col_indices: Vec::new(),
                values: Vec::new(),
                row_lower: Vec::new(),
                row_upper: Vec::new(),
            },
            freeze_scratch: FreezeScratch::new(),
            profile: Simulation.profile(),
            noise_tables: ForwardNoiseTables::default(),
        }
    }

    /// Overrides the simulation solver profile applied at [`Self::run`] entry
    /// (default: `Phase::Simulation.profile()`). Call before `run()`.
    pub(crate) fn set_profile(&mut self, profile: ActiveProfile) {
        self.profile = profile;
    }

    /// Execute a simulation run, evaluating the trained SDDP policy on
    /// `inputs.config.n_scenarios` scenarios and returning a compact cost buffer
    /// for MPI aggregation.
    ///
    /// # Errors
    ///
    /// Returns `Err(SimulationError::InvalidConfiguration { .. })` when
    /// `frozen_templates` is `Some` but has the wrong length.
    /// Returns `Err(SimulationError::LpInfeasible { .. })` when a stage LP has no
    /// feasible solution. Returns `Err(SimulationError::SolverError { .. })` for
    /// other terminal LP solver failures. Returns
    /// `Err(SimulationError::ChannelClosed)` when the channel receiver has been
    /// dropped.
    ///
    /// # Panics (debug builds only)
    ///
    /// Panics if any of the following debug preconditions are violated:
    ///
    /// - `inputs.ctx.templates.len() != num_stages`
    /// - `inputs.training_ctx.initial_state.len() != state.n_state`
    pub(crate) fn run<S, C: Communicator>(
        &mut self,
        inputs: &mut SimulationInputs<'_, S, C>,
    ) -> Result<SimulationRunResult, SimulationError>
    where
        S: SolverInterface<Profile = ActiveProfile> + Send,
    {
        let training_ctx = inputs.training_ctx;
        let TrainingContext {
            horizon,
            state,
            initial_state,
            ..
        } = training_ctx;
        let num_stages = horizon.num_stages();

        debug_assert_inputs(inputs.ctx, num_stages, initial_state.len(), state.n_state);

        let n_pools = training_ctx.node_graph.n_pools;
        if let Some(frozen) = inputs.frozen_templates
            && frozen.len() != n_pools
        {
            return Err(SimulationError::InvalidConfiguration(format!(
                "frozen_templates length {} != n_pools {}",
                frozen.len(),
                n_pools
            )));
        }

        refreeze_templates_if_needed(
            inputs.fcf,
            inputs.ctx,
            training_ctx,
            inputs.frozen_templates,
            &mut self.freeze_batch,
            &mut self.owned_frozen,
            &mut self.freeze_scratch,
        );

        let frozen_templates: &[StageTemplate] =
            match (inputs.frozen_templates, self.owned_frozen.as_deref()) {
                (Some(b), _) | (None, Some(b)) => b,
                (None, None) => unreachable!("owned_frozen is Some when frozen_templates is None"),
            };

        let n_workers = inputs.workspaces.len().max(1);
        let world_size = u32::try_from(inputs.comm.size()).unwrap_or(1).max(1);
        let sim_start = Instant::now();

        // Before the parallel region so rank 0's progress thread can render a
        // banner before any scenario completes (no-op on non-root ranks: sender None).
        if let Some(sender) = inputs.output.event_sender.as_ref() {
            #[allow(clippy::cast_possible_truncation)]
            let _ = sender.send(TrainingEvent::SimulationStarted {
                case_name: String::new(),
                n_scenarios: inputs.config.n_scenarios,
                n_stages: num_stages as u32,
                ranks: world_size,
                threads_per_rank: n_workers as u32,
                timestamp: String::new(),
            });
        }

        let sampler = build_sim_sampler(training_ctx, inputs.config.forward_seed)?;
        sampler.rebuild_noise_tables(
            SIMULATION_ITERATION,
            inputs.config.n_scenarios,
            inputs.ctx.noise_group_ids,
            &mut self.noise_tables,
        )?;

        // Apply the simulation solver profile before the parallel region. For CLP
        // this selects the primal simplex, which eliminates the dual simplex's
        // false-infeasibility on the warm-started, fully-frozen cut-laden LPs.
        for ws in inputs.workspaces.iter_mut() {
            ws.solver.set_profile(&self.profile);
        }

        let (all_costs, all_stats): (WorkerCosts, WorkerStats) = match inputs.traversal {
            Traversal::Sampled { .. } => run_sampled_simulation(
                inputs,
                frozen_templates,
                &sampler,
                &self.noise_tables,
                sim_start,
                n_workers,
                world_size,
            )?,
            Traversal::Enumerated(plan) => {
                let k = plan.paths.leaf.len();
                #[allow(clippy::cast_possible_truncation)]
                if k != inputs.config.n_scenarios as usize {
                    return Err(SimulationError::InvalidConfiguration(format!(
                        "enumerated plan carries {k} leaf paths but the simulation config \
                         resolved n_scenarios = {}",
                        inputs.config.n_scenarios
                    )));
                }
                run_enumerated_simulation(
                    plan,
                    inputs,
                    frozen_templates,
                    &sampler,
                    &self.noise_tables,
                )?
            }
        };

        if let Some(sender) = inputs.output.event_sender.take() {
            #[allow(clippy::cast_possible_truncation)]
            let _ = sender.send(TrainingEvent::SimulationFinished {
                scenarios: inputs.config.n_scenarios,
                output_dir: String::new(),
                elapsed_ms: sim_start.elapsed().as_millis() as u64,
            });
        }
        Ok(SimulationRunResult {
            costs: all_costs,
            solver_stats: all_stats,
            census_weights: inputs.traversal.path_weights().map(<[f64]>::to_vec),
        })
    }
}

/// Assert template/base-row slice lengths match `num_stages` and the initial
/// state length matches `n_state`; debug builds only.
fn debug_assert_inputs(
    ctx: &StageContext<'_>,
    num_stages: usize,
    n_initial: usize,
    n_state: usize,
) {
    debug_assert_eq!(
        ctx.templates.len(),
        num_stages,
        "templates.len()={} != num_stages={num_stages}",
        ctx.templates.len()
    );
    debug_assert_eq!(
        n_initial, n_state,
        "initial_state.len()={n_initial} != n_state={n_state}"
    );
}

/// Monte-Carlo simulation for one call: dispatched from [`SimulationState::run`]
/// under `Traversal::Sampled`. Partitions `inputs.config.n_scenarios` across
/// rayon workers, each running its own scenario range via
/// [`run_worker_scenarios`].
///
/// # Errors
///
/// Returns `Err(SimulationError::InvalidConfiguration { .. })` if stage 0
/// carries no alive node. Returns `Err(SimulationError::LpInfeasible { .. })`
/// when a stage LP has no feasible solution, `Err(SimulationError::SolverError
/// { .. })` for other terminal LP solver failures, and
/// `Err(SimulationError::ChannelClosed)` when the channel receiver has been
/// dropped.
fn run_sampled_simulation<S: SolverInterface + Send, C: Communicator>(
    inputs: &mut SimulationInputs<'_, S, C>,
    frozen_templates: &[StageTemplate],
    sampler: &ForwardSampler<'_>,
    noise_tables: &ForwardNoiseTables,
    sim_start: Instant,
    n_workers: usize,
    world_size: u32,
) -> Result<(WorkerCosts, WorkerStats), SimulationError> {
    let training_ctx = inputs.training_ctx;
    let rank = inputs.comm.rank();
    let scenarios_complete = AtomicU32::new(0);

    let scenario_range = assign_scenarios(inputs.config.n_scenarios, rank, inputs.comm.size());
    #[allow(clippy::cast_possible_truncation)]
    let local_count = (scenario_range.end - scenario_range.start) as usize;
    let scenario_start = scenario_range.start as usize;

    // Every scenario's sampled walk starts at the same stage-0 root —
    // resolved once, mirroring the training forward pass's own root_node.
    let root_node = training_ctx
        .node_graph
        .frontier_node(StageIdx(0))
        .ok_or_else(|| {
            SimulationError::InvalidConfiguration(
                "node graph: stage 0 carries no alive node".to_string(),
            )
        })?;

    let params = SimWorkerParams {
        ctx: inputs.ctx,
        fcf: inputs.fcf,
        training_ctx,
        output: &inputs.output,
        config: inputs.config,
        node_bases: inputs.node_bases,
        frozen_templates,
        scenarios_complete: &scenarios_complete,
        sim_start,
        local_count,
        n_workers,
        scenario_start,
        sampler,
        noise_tables,
        world_size,
        root_node,
    };

    let worker_results: Vec<Result<(WorkerCosts, WorkerStats), SimulationError>> = inputs
        .workspaces
        .par_iter_mut()
        .enumerate()
        .map(|(w, ws)| run_worker_scenarios(w, ws, &params))
        .collect();

    let mut all_costs = Vec::with_capacity(local_count);
    let mut all_stats = Vec::with_capacity(local_count);
    for result in worker_results {
        let (costs, stats) = result?;
        all_costs.extend(costs);
        all_stats.extend(stats);
    }
    // Each worker emits a contiguous ascending scenario_id range, so the
    // sequential `extend` leaves `all_costs`/`all_stats` sorted — no sort needed.
    debug_assert!(
        all_costs.windows(2).all(|w| w[0].0 <= w[1].0),
        "all_costs not pre-sorted: workers must emit ascending scenario_id"
    );
    debug_assert!(
        all_stats.windows(2).all(|w| w[0].0 <= w[1].0),
        "all_stats not pre-sorted: workers must emit ascending scenario_id"
    );
    Ok((all_costs, all_stats))
}

/// Execute one worker's share of scenarios in the rayon parallel region.
///
/// Returns `(worker_costs, worker_stats)` for the worker's assigned scenarios,
/// or a [`SimulationError`] if any scenario LP fails or the channel is closed.
fn run_worker_scenarios<S: SolverInterface + Send>(
    w: usize,
    ws: &mut SolverWorkspace<S>,
    params: &SimWorkerParams<'_>,
) -> Result<(WorkerCosts, WorkerStats), SimulationError> {
    let (start_local, end_local) = partition(params.local_count, params.n_workers, w);
    let n_scenarios = end_local - start_local;
    let mut worker_costs = Vec::with_capacity(n_scenarios);
    let mut worker_stats = Vec::with_capacity(n_scenarios);
    // Resize once per worker, reuse across scenarios: no per-scenario allocation.
    let noise_dim = params.training_ctx.stochastic.dim();
    ws.scratch.raw_noise_buf.resize(noise_dim, 0.0_f64);
    ws.scratch.corr_scratch.resize(2 * noise_dim, 0.0_f64);

    // Build once per worker: eliminates the per-(scenario, stage) allocation that
    // would otherwise occur inside extract_thermals / extract_hydros.
    let lookups = SimLookups::build(
        params.training_ctx.study_dims,
        params.ctx.geometry_per_stage,
        params.output.hydro_cell_index,
        params.output.entity_counts.hydro_ids.len(),
    );

    for local_idx in start_local..end_local {
        #[allow(clippy::cast_possible_truncation)]
        let scenario_id = (params.scenario_start + local_idx) as u32;

        let stats_before = ws.solver.statistics();
        let load_spec = SimScenarioLoadSpec {
            frozen_templates: params.frozen_templates,
            node_bases: params.node_bases,
        };
        // mem::take (capacity retained) so the immutable ScenarioIds borrows of
        // these slices do not conflict with the `&mut ws` passed below.
        let mut raw_noise_buf = std::mem::take(&mut ws.scratch.raw_noise_buf);
        let mut corr_scratch = std::mem::take(&mut ws.scratch.corr_scratch);
        let result = process_scenario_stages(
            ws,
            params.ctx,
            params.fcf,
            params.training_ctx,
            &load_spec,
            params.output,
            &mut ScenarioIds {
                scenario_id,
                global_scenario: scenario_id,
                total_scenarios: params.config.n_scenarios,
                raw_noise_buf: &mut raw_noise_buf,
                corr_scratch: &mut corr_scratch,
                sampler: params.sampler,
                noise_tables: params.noise_tables,
                root_node: params.root_node,
            },
            &lookups,
        );
        ws.scratch.raw_noise_buf = raw_noise_buf;
        ws.scratch.corr_scratch = corr_scratch;
        let (total_cost, stage_results) = result?;
        let stats_after = ws.solver.statistics();
        let scenario_delta = SolverStatsDelta::from_snapshots(&stats_before, &stats_after);
        let scenario_solve_time_ms = scenario_delta.solve_time_ms;
        let scenario_lp_solves = scenario_delta.lp_solves;
        // opening = -1: no opening loop; the sentinel maps to NULL in parquet.
        worker_stats.push((scenario_id, -1_i32, scenario_delta));

        worker_costs.push(dispatch_scenario_result(
            params.output,
            scenario_id,
            total_cost,
            stage_results,
        )?);
        let completed = params.scenarios_complete.fetch_add(1, Ordering::Relaxed) + 1;
        // Scale rank-local count to a global estimate (balanced-workload
        // assumption), clamped at the total so the last scenario lands on 100%.
        let completed_global = completed
            .saturating_mul(params.world_size)
            .min(params.config.n_scenarios);
        #[allow(clippy::cast_possible_truncation)]
        emit_sim_progress(
            params.output.event_sender.as_ref(),
            total_cost,
            scenario_solve_time_ms,
            scenario_lp_solves,
            completed_global,
            params.config.n_scenarios,
            params.sim_start.elapsed().as_millis() as u64,
        );
    }
    Ok((worker_costs, worker_stats))
}

/// Build the [`ForwardSampler`] for a simulation run from the training context.
pub(crate) fn build_sim_sampler<'a>(
    training_ctx: &'a TrainingContext<'a>,
    forward_seed: Option<u64>,
) -> Result<ForwardSampler<'a>, SimulationError> {
    Ok(build_forward_sampler(ForwardSamplerConfig {
        class_schemes: ClassSchemes {
            inflow: Some(training_ctx.inflow_scheme),
            load: Some(training_ctx.load_scheme),
            ncs: Some(training_ctx.ncs_scheme),
        },
        ctx: training_ctx.stochastic,
        forward_seed,
        stages: training_ctx.stages,
        historical_library: training_ctx.historical_library,
        external_inflow_library: training_ctx.external_inflow_library,
        external_load_library: training_ctx.external_load_library,
        external_ncs_library: training_ctx.external_ncs_library,
    })?)
}

/// Populate `owned_frozen` when the caller did not provide pre-frozen templates.
///
/// No-op if `caller_frozen` is `Some`. Otherwise rebuilds `owned_frozen` — one
/// frozen template per POOL — from the FCF, context templates, and state layout,
/// using `freeze_batch` as scratch (its post-call contents are unspecified). Cost
/// `O(n_pools * num_active_cuts)`.
fn refreeze_templates_if_needed(
    fcf: &FutureCostFunction,
    ctx: &StageContext<'_>,
    training_ctx: &TrainingContext<'_>,
    caller_frozen: Option<&[StageTemplate]>,
    freeze_batch: &mut RowBatch,
    owned_frozen: &mut Option<Vec<StageTemplate>>,
    freeze_scratch: &mut FreezeScratch,
) {
    if caller_frozen.is_some() {
        *owned_frozen = None;
        return;
    }

    let state = training_ctx.state;
    let cut_state_layouts = training_ctx.cut_state_layouts;
    let node_graph = training_ctx.node_graph;

    // Per-POOL frozen overlay, mirroring the training freeze: pool `p`'s cuts on
    // pool `p`'s base stage template `templates[pool_stage[p]]`. A per-stage build
    // would bake one node's cuts into a sibling's LP on a branching graph.
    let mut owned = Vec::with_capacity(node_graph.n_pools);
    // Rationale: `p` is the pool passed by value to `build_cut_row_batch_into` and
    // mapped through `pool_stage[p]` to the base stage template; an `enumerate`
    // over one pool-keyed slice would not carry those other uses.
    #[allow(clippy::needless_range_loop)]
    for p in 0..node_graph.n_pools {
        let t = node_graph.pool_stage[p];
        build_cut_row_batch_into(
            freeze_batch,
            fcf,
            p,
            state,
            &cut_state_layouts[p],
            &ctx.template(t).col_scale,
        );
        let mut frozen = StageTemplate::empty();
        freeze_rows_into_template(ctx.template(t), freeze_batch, &mut frozen, freeze_scratch);
        owned.push(frozen);
    }
    *owned_frozen = Some(owned);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simulation_state_new_allocates_empty_freeze_batch() {
        let state = SimulationState::new(3);
        assert!(state.owned_frozen.is_none(), "owned_frozen must be None");
        assert_eq!(
            state.freeze_batch.num_rows, 0,
            "freeze_batch.num_rows must be 0"
        );
    }

    /// Reproduces the scaling logic used inside `run_worker_scenarios` for
    /// `SimulationProgress.scenarios_complete`. Kept as a free helper so the
    /// invariants are testable without a full `SimulationInputs` fixture.
    fn scaled_global_count(local_completed: u32, world_size: u32, total: u32) -> u32 {
        local_completed.saturating_mul(world_size).min(total)
    }

    #[test]
    fn scaled_global_count_single_rank_is_identity() {
        assert_eq!(scaled_global_count(0, 1, 100), 0);
        assert_eq!(scaled_global_count(50, 1, 100), 50);
        assert_eq!(scaled_global_count(100, 1, 100), 100);
    }

    #[test]
    fn scaled_global_count_balanced_two_ranks_tracks_global() {
        // Rank 0 has 50 local scenarios out of 100 global; each local
        // completion advances the global estimate by world_size = 2.
        assert_eq!(scaled_global_count(1, 2, 100), 2);
        assert_eq!(scaled_global_count(25, 2, 100), 50);
        assert_eq!(scaled_global_count(50, 2, 100), 100);
    }

    #[test]
    fn scaled_global_count_clamps_at_total_when_unevenly_divided() {
        // N=100 across K=3 ranks: rank 0 has 34, ranks 1-2 have 33 each.
        // At local 33: estimate = 99. At local 34: estimate = 102 clamped to 100.
        assert_eq!(scaled_global_count(33, 3, 100), 99);
        assert_eq!(scaled_global_count(34, 3, 100), 100);
    }

    /// Assert that `all_costs` is ascending by `scenario_id` after a sequential
    /// `extend` from four workers covering 3 scenarios each (1-rank, 4-worker,
    /// 12-scenario layout): the ordering invariant is structural, not algorithmic.
    #[test]
    fn aggregate_costs_is_ascending_post_extend() {
        use crate::simulation::types::ScenarioCategoryCosts;

        let zero_cat = ScenarioCategoryCosts {
            resource_cost: 0.0,
            recourse_cost: 0.0,
            violation_cost: 0.0,
            regularization_cost: 0.0,
            imputed_cost: 0.0,
        };

        let worker_outputs: Vec<Vec<(u32, f64, ScenarioCategoryCosts)>> = (0u32..4)
            .map(|w| {
                (0..3u32)
                    .map(|i| (w * 3 + i, 0.0_f64, zero_cat.clone()))
                    .collect()
            })
            .collect();

        let mut all_costs: Vec<(u32, f64, ScenarioCategoryCosts)> = Vec::with_capacity(12);
        for costs in worker_outputs {
            all_costs.extend(costs);
        }

        assert!(
            all_costs.windows(2).all(|w| w[0].0 <= w[1].0),
            "all_costs must be ascending by scenario_id after sequential extend"
        );
        let ids: Vec<u32> = all_costs.iter().map(|e| e.0).collect();
        assert_eq!(ids, (0u32..12).collect::<Vec<_>>());
    }

    /// Assert that the `debug_assert!` invariant check catches an out-of-order
    /// `all_costs` sequence.
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "all_costs not pre-sorted")]
    fn debug_assert_fires_for_out_of_order_costs() {
        use crate::simulation::types::ScenarioCategoryCosts;

        let zero_cat = ScenarioCategoryCosts {
            resource_cost: 0.0,
            recourse_cost: 0.0,
            violation_cost: 0.0,
            regularization_cost: 0.0,
            imputed_cost: 0.0,
        };

        // Intentionally out-of-order: scenario 2 appears before scenario 1.
        let all_costs: Vec<(u32, f64, ScenarioCategoryCosts)> = vec![
            (0, 0.0, zero_cat.clone()),
            (2, 0.0, zero_cat.clone()),
            (1, 0.0, zero_cat.clone()),
        ];

        debug_assert!(
            all_costs.windows(2).all(|w| w[0].0 <= w[1].0),
            "all_costs not pre-sorted: workers must emit ascending scenario_id"
        );
    }
}
