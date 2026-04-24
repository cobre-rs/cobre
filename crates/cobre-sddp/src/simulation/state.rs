//! Simulation state management and entry point.
//!
//! [`SimulationState`] owns re-bake scratch buffers allocated once per run.
//! [`SimulationInputs`] bundles per-call borrowed inputs (no per-scenario allocation).
//!
//! ## Callers
//!
//! Production call sites of [`crate::simulate`] (via [`super::StudySetup::simulate`]):
//!
//! | Caller | File | `baked_templates` |
//! |--------|------|-------------------|
//! | `run_simulation_phase` — training path | `crates/cobre-cli/src/commands/run.rs` | `Some(...)` from `TrainingResult` produced by `TrainingSession::finalize` |
//! | `run_simulation_phase` — checkpoint path | `crates/cobre-cli/src/commands/run.rs` | `None` — baked templates are not stored in policy checkpoints; `rebake_templates_if_needed` rebuilds them once at startup |
//! | `run_simulation_phase_py` — training path | `crates/cobre-python/src/run.rs` | `Some(...)` from `TrainingResult` produced by `TrainingSession::finalize` |
//! | `run_simulation_phase_py` — checkpoint path | `crates/cobre-python/src/run.rs` | `None` — same checkpoint constraint as the CLI path |
//!
//! Tests in `crates/cobre-sddp/src/setup/mod.rs` pass `None` directly; this is
//! intentional — tests use minimal fixtures and are not on the hot path.
//!
//! ## Hot-path allocation discipline
//!
//! No allocations occur per scenario or per stage during the inner loops.
//! The re-bake allocation in [`rebake_templates_if_needed`] is a one-time
//! setup amortised across the simulation's per-scenario LP solves.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::mpsc::Sender;
use std::time::Instant;

use cobre_comm::Communicator;
use cobre_core::TrainingEvent;
use cobre_solver::{RowBatch, SolverInterface, StageTemplate};
use cobre_stochastic::context::ClassSchemes;
use cobre_stochastic::{
    build_forward_sampler, ClassDimensions, ForwardSampler, ForwardSamplerConfig,
};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};

use crate::{
    context::{StageContext, TrainingContext},
    forward::{build_cut_row_batch_into, partition},
    simulation::{
        config::SimulationConfig,
        error::SimulationError,
        extraction::assign_scenarios,
        pipeline::{
            dispatch_scenario_result, emit_sim_progress, process_scenario_stages, ScenarioIds,
            SimulationOutputSpec, SimulationRunResult, WorkerCosts, WorkerStats,
            SIMULATION_SEED_OFFSET,
        },
    },
    solver_stats::SolverStatsDelta,
    workspace::{CapturedBasis, SolverWorkspace},
    FutureCostFunction,
};

/// Per-call argument bundle for [`SimulationState::run`].
///
/// Groups all borrowed inputs that vary between calls: solver workspaces, stage
/// context, future cost function, training context, simulation configuration,
/// output spec, optional pre-baked templates, stage bases, and the communicator.
/// Owned scratch buffers (re-bake intermediates) live on [`SimulationState`].
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
    /// Pre-baked LP templates from training. `None` triggers local re-bake.
    pub baked_templates: Option<&'a [StageTemplate]>,
    /// Per-stage warm-start basis captured from the training checkpoint.
    pub stage_bases: &'a [Option<CapturedBasis>],
    /// MPI communicator.
    pub comm: &'a C,
}

impl<'a, S: SolverInterface + Send, C> SimulationInputs<'a, S, C> {
    /// Construct a `SimulationInputs` bundle from positional arguments.
    ///
    /// Equivalent to constructing the struct literal directly, but usable from
    /// call sites that prefer a constructor to avoid a struct-literal import.
    pub(crate) fn new(
        workspaces: &'a mut [SolverWorkspace<S>],
        ctx: &'a StageContext<'a>,
        fcf: &'a FutureCostFunction,
        training_ctx: &'a TrainingContext<'a>,
        config: &'a SimulationConfig,
        output: SimulationOutputSpec<'a>,
        baked_templates: Option<&'a [StageTemplate]>,
        stage_bases: &'a [Option<CapturedBasis>],
        comm: &'a C,
    ) -> Self {
        Self {
            workspaces,
            ctx,
            fcf,
            training_ctx,
            config,
            output,
            baked_templates,
            stage_bases,
            comm,
        }
    }
}

/// Owned scratch state for one simulation run.
///
/// `SimulationState` is typically created immediately before calling
/// [`SimulationState::run`] and discarded afterwards. The owned fields hold the
/// re-bake intermediates for the lazy re-bake branch: an optional
/// `Vec<StageTemplate>` built when the caller does not supply pre-baked
/// templates, and a [`RowBatch`] scratch used during that build.
///
/// If the caller always provides `baked_templates`, neither buffer is populated.
pub(crate) struct SimulationState {
    /// Owned baked templates built by the lazy re-bake branch.
    ///
    /// Populated by [`rebake_templates_if_needed`] when
    /// `SimulationInputs::baked_templates` is `None`. Cleared and
    /// repopulated on each `run()` call.
    owned_baked: Option<Vec<StageTemplate>>,
    /// Row-batch scratch used by the lazy re-bake loop.
    ///
    /// Initialized to an empty batch in [`SimulationState::new`].
    bake_batch: RowBatch,
}

impl SimulationState {
    /// Construct a new `SimulationState` with empty re-bake buffers.
    ///
    /// `num_stages` is used only to pre-allocate the `owned_baked` capacity;
    /// the field is initialized to `None` and populated lazily on the first
    /// run that requires re-baking.
    #[must_use]
    pub(crate) fn new(_num_stages: usize) -> Self {
        Self {
            owned_baked: None,
            bake_batch: RowBatch {
                num_rows: 0,
                row_starts: Vec::new(),
                col_indices: Vec::new(),
                values: Vec::new(),
                row_lower: Vec::new(),
                row_upper: Vec::new(),
            },
        }
    }

    /// Execute a simulation run.
    ///
    /// Evaluates the trained SDDP policy on `inputs.config.n_scenarios` scenarios
    /// by running a forward-only pass through all stages, extracting per-entity
    /// results at each stage, streaming completed scenario results through a
    /// bounded channel, and returning a compact cost buffer for MPI aggregation.
    ///
    /// # Errors
    ///
    /// Returns `Err(SimulationError::InvalidConfiguration { .. })` when
    /// `baked_templates` is `Some` but has the wrong length.
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
    /// - `inputs.ctx.base_rows.len() != num_stages`
    /// - `inputs.training_ctx.initial_state.len() != indexer.n_state`
    pub(crate) fn run<S: SolverInterface + Send, C: Communicator>(
        &mut self,
        inputs: &mut SimulationInputs<'_, S, C>,
    ) -> Result<SimulationRunResult, SimulationError> {
        let training_ctx = inputs.training_ctx;
        let TrainingContext {
            horizon,
            indexer,
            initial_state,
            ..
        } = training_ctx;
        let num_stages = horizon.num_stages();
        let rank = inputs.comm.rank();

        debug_assert_inputs(inputs.ctx, num_stages, initial_state.len(), indexer.n_state);

        // Validate baked-template slice length if provided.
        if let Some(baked) = inputs.baked_templates {
            if baked.len() != num_stages {
                return Err(SimulationError::InvalidConfiguration(format!(
                    "baked_templates length {} != num_stages {}",
                    baked.len(),
                    num_stages
                )));
            }
        }

        // Populate `self.owned_baked` when the caller did not provide templates.
        rebake_templates_if_needed(
            inputs.fcf,
            inputs.ctx,
            indexer,
            num_stages,
            inputs.baked_templates,
            &mut self.bake_batch,
            &mut self.owned_baked,
        );

        let baked_templates: &[StageTemplate] =
            match (inputs.baked_templates, self.owned_baked.as_deref()) {
                (Some(b), _) | (None, Some(b)) => b,
                (None, None) => unreachable!("owned_baked is Some when baked_templates is None"),
            };

        let scenario_range = assign_scenarios(inputs.config.n_scenarios, rank, inputs.comm.size());
        #[allow(clippy::cast_possible_truncation)]
        let local_count = (scenario_range.end - scenario_range.start) as usize;
        let scenario_start = scenario_range.start as usize;
        let n_workers = inputs.workspaces.len().max(1);
        let world_size = u32::try_from(inputs.comm.size()).unwrap_or(1).max(1);
        let sim_start = Instant::now();
        let scenarios_complete = AtomicU32::new(0);

        // Emit SimulationStarted once per rank before the parallel region so
        // rank 0's progress thread can render a banner before any scenario
        // completes. Non-root ranks' senders are None; this is a no-op on
        // those ranks.
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

        let sampler = build_sim_sampler(training_ctx)?;

        let worker_results: Vec<Result<(WorkerCosts, WorkerStats), SimulationError>> = inputs
            .workspaces
            .par_iter_mut()
            .enumerate()
            .map(|(w, ws)| {
                run_worker_scenarios(
                    w,
                    ws,
                    inputs.ctx,
                    inputs.fcf,
                    training_ctx,
                    &inputs.output,
                    inputs.config,
                    inputs.stage_bases,
                    baked_templates,
                    &scenarios_complete,
                    sim_start,
                    local_count,
                    n_workers,
                    scenario_start,
                    &sampler,
                    num_stages,
                    world_size,
                )
            })
            .collect();

        let mut all_costs = Vec::with_capacity(local_count);
        let mut all_stats = Vec::with_capacity(local_count);
        for result in worker_results {
            let (costs, stats) = result?;
            all_costs.extend(costs);
            all_stats.extend(stats);
        }
        all_costs.sort_by_key(|&(id, _, _)| id);
        all_stats.sort_by_key(|&(id, _, _)| id);

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
        })
    }
}

/// Assert simulation preconditions in debug builds.
///
/// Panics in debug builds if template or base-row slice lengths do not match
/// `num_stages`, or if the initial state length does not match `n_state`.
/// Compiled away in release builds (all assertions are `debug_assert`).
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
        ctx.base_rows.len(),
        num_stages,
        "base_rows.len()={} != num_stages={num_stages}",
        ctx.base_rows.len()
    );
    debug_assert_eq!(
        n_initial, n_state,
        "initial_state.len()={n_initial} != n_state={n_state}"
    );
}

/// Execute one worker's share of scenarios in the rayon parallel region.
///
/// Called once per worker thread by [`SimulationState::run`] via `par_iter_mut`.
/// Partitions `local_count` scenarios across `n_workers` workers using
/// [`partition`], drives per-scenario LP solves, accumulates costs and solver
/// statistics, and streams progress events through the output channel.
///
/// Returns `(worker_costs, worker_stats)` for the worker's assigned scenarios,
/// or a [`SimulationError`] if any scenario LP fails or the channel is closed.
// RATIONALE: 16 args are passed through the rayon parallel closure boundary.
// Each represents a distinct read-only data source (ctx, fcf, training_ctx, config,
// stage_bases, baked_templates, sampler) or mutable accumulator (ws), or progress
// tracking primitive (scenarios_complete, sim_start). Bundling into a struct would
// require either cloning or wrapping in Arc — both incompatible with the zero-allocation
// HPC constraint for the simulation hot path.
#[allow(clippy::too_many_arguments)]
fn run_worker_scenarios<S: SolverInterface + Send>(
    w: usize,
    ws: &mut SolverWorkspace<S>,
    ctx: &StageContext<'_>,
    fcf: &FutureCostFunction,
    training_ctx: &TrainingContext<'_>,
    output: &SimulationOutputSpec<'_>,
    config: &SimulationConfig,
    stage_bases: &[Option<CapturedBasis>],
    baked_templates: &[StageTemplate],
    scenarios_complete: &AtomicU32,
    sim_start: Instant,
    local_count: usize,
    n_workers: usize,
    scenario_start: usize,
    sampler: &ForwardSampler,
    num_stages: usize,
    world_size: u32,
) -> Result<(WorkerCosts, WorkerStats), SimulationError> {
    let (start_local, end_local) = partition(local_count, n_workers, w);
    let worker_sender: Option<Sender<TrainingEvent>> = output.event_sender.clone();
    let n_scenarios = end_local - start_local;
    let mut worker_costs = Vec::with_capacity(n_scenarios);
    let mut worker_stats = Vec::with_capacity(n_scenarios);
    // Sampling scratch: resize ws.scratch buffers once per worker, reuse across
    // scenarios. Avoids per-worker vec![...] allocations on the hot path.
    let noise_dim = training_ctx.stochastic.dim();
    ws.scratch.raw_noise_buf.resize(noise_dim, 0.0_f64);
    #[allow(clippy::cast_possible_truncation)]
    ws.scratch
        .perm_scratch
        .resize(config.n_scenarios.max(1) as usize, 0_usize);

    for local_idx in start_local..end_local {
        #[allow(clippy::cast_possible_truncation)]
        let scenario_id = (scenario_start + local_idx) as u32;
        let global_scenario = SIMULATION_SEED_OFFSET.saturating_add(scenario_id);

        let stats_before = ws.solver.statistics();
        let load_spec = crate::simulation::pipeline::SimScenarioLoadSpec {
            baked_templates,
            stage_bases,
            basis_activity_window: config.basis_activity_window,
        };
        // Split raw_noise_buf and perm_scratch out of ws.scratch so that the
        // immutable borrows of those slices in ScenarioIds do not conflict with
        // the &mut ws passed to process_scenario_stages.  Capacity is retained
        // across calls via mem::take + swap-back.
        let mut raw_noise_buf = std::mem::take(&mut ws.scratch.raw_noise_buf);
        let mut perm_scratch = std::mem::take(&mut ws.scratch.perm_scratch);
        let result = process_scenario_stages(
            ws,
            ctx,
            fcf,
            training_ctx,
            &load_spec,
            output,
            &mut ScenarioIds {
                scenario_id,
                global_scenario,
                num_stages,
                total_scenarios: config.n_scenarios,
                raw_noise_buf: &mut raw_noise_buf,
                perm_scratch: &mut perm_scratch,
                sampler,
            },
        );
        ws.scratch.raw_noise_buf = raw_noise_buf;
        ws.scratch.perm_scratch = perm_scratch;
        let (total_cost, stage_results) = result?;
        let stats_after = ws.solver.statistics();
        let scenario_delta = SolverStatsDelta::from_snapshots(&stats_before, &stats_after);
        let scenario_solve_time_ms = scenario_delta.solve_time_ms;
        let scenario_lp_solves = scenario_delta.lp_solves;
        // opening = -1: simulation has no opening loop (one solve per
        // scenario×stage). The sentinel maps to NULL in parquet.
        worker_stats.push((scenario_id, -1_i32, scenario_delta));

        worker_costs.push(dispatch_scenario_result(
            output,
            scenario_id,
            total_cost,
            stage_results,
        )?);
        let completed = scenarios_complete.fetch_add(1, Ordering::Relaxed) + 1;
        // Scale rank-local count to a global estimate assuming balanced
        // workload across ranks (the assign_scenarios invariant). Clamp at
        // the global total so the final scenario lands exactly on 100%.
        let completed_global = completed.saturating_mul(world_size).min(config.n_scenarios);
        #[allow(clippy::cast_possible_truncation)]
        emit_sim_progress(
            worker_sender.as_ref(),
            total_cost,
            scenario_solve_time_ms,
            scenario_lp_solves,
            completed_global,
            config.n_scenarios,
            sim_start.elapsed().as_millis() as u64,
        );
    }
    Ok((worker_costs, worker_stats))
}

/// Build the forward sampler for a simulation run from the training context.
///
/// Constructs a [`ForwardSampler`] using the inflow, load, and NCS class schemes
/// from `training_ctx`, together with the stochastic dimension metadata.
/// Called once per [`SimulationState::run`] invocation before the rayon parallel region.
fn build_sim_sampler<'a>(
    training_ctx: &'a TrainingContext<'a>,
) -> Result<ForwardSampler<'a>, SimulationError> {
    Ok(build_forward_sampler(ForwardSamplerConfig {
        class_schemes: ClassSchemes {
            inflow: Some(training_ctx.inflow_scheme),
            load: Some(training_ctx.load_scheme),
            ncs: Some(training_ctx.ncs_scheme),
        },
        ctx: training_ctx.stochastic,
        stages: training_ctx.stages,
        dims: ClassDimensions {
            n_hydros: training_ctx.stochastic.n_hydros(),
            n_load_buses: training_ctx.stochastic.n_load_buses(),
            n_ncs: training_ctx.stochastic.n_stochastic_ncs(),
        },
        historical_library: training_ctx.historical_library,
        external_inflow_library: training_ctx.external_inflow_library,
        external_load_library: training_ctx.external_load_library,
        external_ncs_library: training_ctx.external_ncs_library,
    })?)
}

/// Populate `owned_baked` when the caller did not provide pre-baked templates.
///
/// If `caller_baked` is `Some`, this function is a no-op: the caller's slice is
/// used directly and `owned_baked` is not touched.
///
/// If `caller_baked` is `None`, the function clears `owned_baked`, then rebuilds
/// it from the FCF, context templates, and indexer. `bake_batch` is used as
/// scratch and its contents after the call are unspecified.
///
/// Cost is `O(num_stages * num_active_cuts)` — a one-time setup amortised across
/// the simulation's per-scenario LP solves.
fn rebake_templates_if_needed(
    fcf: &FutureCostFunction,
    ctx: &StageContext<'_>,
    indexer: &crate::StageIndexer,
    num_stages: usize,
    caller_baked: Option<&[StageTemplate]>,
    bake_batch: &mut RowBatch,
    owned_baked: &mut Option<Vec<StageTemplate>>,
) {
    if caller_baked.is_some() {
        // Caller provided templates — skip re-bake entirely.
        *owned_baked = None;
        return;
    }

    let mut owned = Vec::with_capacity(num_stages);
    for t in 0..num_stages {
        build_cut_row_batch_into(bake_batch, fcf, t, indexer, &ctx.templates[t].col_scale);
        let mut baked = StageTemplate::empty();
        cobre_solver::bake_rows_into_template(&ctx.templates[t], bake_batch, &mut baked);
        owned.push(baked);
    }
    *owned_baked = Some(owned);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simulation_state_new_allocates_empty_bake_batch() {
        let state = SimulationState::new(3);
        assert!(state.owned_baked.is_none(), "owned_baked must be None");
        assert_eq!(
            state.bake_batch.num_rows, 0,
            "bake_batch.num_rows must be 0"
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
}
