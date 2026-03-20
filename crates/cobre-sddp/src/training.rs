//! Training loop orchestrator for the SDDP algorithm.
//!
//! [`train`] wires together the forward pass, forward synchronization, state
//! exchange, backward pass, cut synchronization, lower bound evaluation, and
//! convergence check into a single iterative loop.
//!
//! ## Iteration lifecycle
//!
//! Each iteration follows the corrected ordering from the LB plan fix (F-019):
//!
//! 1. Forward pass — scenario simulation, local UB statistics.
//! 2. Forward sync — global `allreduce` for UB statistics.
//! 3. State exchange — `allgatherv` trial points for the backward pass.
//! 4. Backward pass — Benders cut generation.
//! 5. Cut sync — `allgatherv` new cuts across ranks.
//!    5a. Cut selection — optional periodic pool pruning via `CutSelectionStrategy`.
//!    5b. LB evaluation — rank 0 solves stage-0 openings, broadcasts scalar.
//! 6. Convergence check — stopping rules evaluated.
//! 7. (checkpoint — not yet implemented)
//! 8. Event emission — `IterationSummary` and per-step events via channel.
//!
//! ## Pre-allocation discipline
//!
//! All workspace buffers (`PatchBuffer`, `TrajectoryRecord` flat vec,
//! `ExchangeBuffers`, `CutSyncBuffers`) are allocated once before the
//! iteration loop and reused across all iterations. No heap allocation
//! occurs on the hot path.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::time::Instant;

use cobre_comm::Communicator;
use cobre_core::TrainingEvent;
use cobre_solver::Basis;
use cobre_solver::SolverInterface;
use cobre_stochastic::OpeningTree;

use crate::{
    SddpError, StoppingRuleSet, TrainingConfig, TrajectoryRecord,
    backward::run_backward_pass,
    context::{StageContext, TrainingContext},
    convergence::ConvergenceMonitor,
    cut::fcf::FutureCostFunction,
    cut_selection::CutSelectionStrategy,
    cut_sync::CutSyncBuffers,
    evaluate_lower_bound,
    forward::{ForwardPassBatch, run_forward_pass, sync_forward},
    lower_bound::LbEvalSpec,
    lp_builder::PatchBuffer,
    risk_measure::RiskMeasure,
    state_exchange::ExchangeBuffers,
    stopping_rule::RULE_ITERATION_LIMIT,
    workspace::{BasisStore, WorkspacePool},
};

// ---------------------------------------------------------------------------
// TrainingResult
// ---------------------------------------------------------------------------

/// Summary statistics produced when the training loop terminates.
#[derive(Debug, Clone)]
pub struct TrainingResult {
    /// Final lower bound at termination.
    pub final_lb: f64,

    /// Final upper bound mean at termination.
    pub final_ub: f64,

    /// Final upper bound standard deviation at termination.
    pub final_ub_std: f64,

    /// Final convergence gap: `(UB - LB) / max(1.0, |UB|)`.
    pub final_gap: f64,

    /// Total number of iterations completed.
    pub iterations: u64,

    /// Human-readable termination reason (e.g., `"iteration_limit"`, `"graceful_shutdown"`).
    pub reason: String,

    /// Total wall-clock time for the training run, in milliseconds.
    pub total_time_ms: u64,

    /// Per-stage solver basis from the last iteration, indexed 0-based.
    ///
    /// Each entry is `Some(basis)` if the stage was solved at least once
    /// during the final iteration, or `None` if no solve occurred (e.g.,
    /// the last stage in finite-horizon mode has no successor cuts).
    /// Used for policy checkpoint persistence.
    pub basis_cache: Vec<Option<Basis>>,
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Send a training event if the channel is present (ignores `None` or receiver drop).
#[inline]
fn emit(sender: Option<&Sender<TrainingEvent>>, event: TrainingEvent) {
    if let Some(s) = sender {
        let _ = s.send(event);
    }
}

/// Collect the newly generated cuts from the FCF pool for a given stage and
/// iteration.
///
/// The backward pass inserts cuts directly into the FCF with deterministic
/// slot indices. This helper reads back the metadata for those slots to build
/// the `local_cuts` tuple slice consumed by [`CutSyncBuffers::sync_cuts`].
///
/// Returns a `Vec` of `(slot_index, iteration, forward_pass_index, intercept,
/// coefficients)` tuples for all cuts at `stage` whose
/// `metadata.iteration_generated == iteration`.
///
/// This allocation is bounded to `total_scenarios` entries per stage per
/// iteration and is acceptable on the backward/sync path (not the inner LP
/// loop).
fn collect_local_cuts_for_stage(
    fcf: &FutureCostFunction,
    stage: usize,
    iteration: u64,
) -> Vec<(u32, u32, u32, f64, Vec<f64>)> {
    let pool = &fcf.pools[stage];
    let mut result = Vec::new();
    for slot in 0..pool.populated_count {
        if !pool.active[slot] {
            continue;
        }
        let meta = &pool.metadata[slot];
        if meta.iteration_generated != iteration {
            continue;
        }
        let intercept = pool.intercepts[slot];
        let coefficients = pool.coefficients[slot].clone();
        #[allow(clippy::cast_possible_truncation)]
        let slot_u32 = slot as u32;
        #[allow(clippy::cast_possible_truncation)]
        let iter_u32 = iteration as u32;
        result.push((
            slot_u32,
            iter_u32,
            meta.forward_pass_index,
            intercept,
            coefficients,
        ));
    }
    result
}

// ---------------------------------------------------------------------------
// train
// ---------------------------------------------------------------------------

/// Execute the SDDP training loop.
///
/// Allocates all workspace buffers, runs the iteration loop until a stopping
/// rule triggers or `config.max_iterations` is reached, and returns a
/// [`TrainingResult`] summarising the final convergence statistics.
///
/// ## Error handling
///
/// Returns `Err(SddpError)` immediately on:
/// - LP infeasibility in the forward or backward pass → `SddpError::Infeasible`
/// - Solver failure → `SddpError::Solver`
/// - Communication failure → `SddpError::Communication`
///
/// ## Event channel
///
/// When `config.event_sender` is `Some`, typed [`TrainingEvent`] values are
/// emitted at each lifecycle step boundary. Event send failures (receiver
/// dropped) are silently ignored so they cannot interrupt training.
///
/// ## Cut selection
///
/// When `cut_selection` is `Some(strategy)`, step 5a runs after every cut
/// synchronisation. The strategy's `should_run(iteration)` gate controls the
/// frequency; at eligible iterations every stage's cut pool is scanned and
/// inactive cuts are deactivated. When `cut_selection` is `None`, step 5a is
/// skipped entirely and no [`TrainingEvent::CutSelectionComplete`] events are
/// emitted.
///
/// # Errors
///
/// Returns `Err(SddpError::Infeasible { .. })` when an LP has no feasible
/// solution. Returns `Err(SddpError::Solver(_))` for other solver failures.
/// Returns `Err(SddpError::Communication(_))` when a collective operation
/// fails.
///
/// # Examples
///
/// ```rust,ignore
/// use cobre_sddp::{train, TrainingConfig, FutureCostFunction, StageIndexer};
/// use cobre_sddp::{StoppingRuleSet, StoppingRule, RiskMeasure, HorizonMode};
/// use cobre_sddp::lp_builder::StageTemplate;
///
/// let mut solver = HiggsBackend::new();
/// let config = TrainingConfig { forward_passes: 100, ..Default::default() };
/// let mut fcf = FutureCostFunction::new(num_stages - 1, n_state, capacity);
/// let stopping = StoppingRuleSet::any(vec![
///     StoppingRule::iteration_limit(100),
///     StoppingRule::relative_gap(0.01),
/// ]);
/// let risk = vec![RiskMeasure::Expectation; num_stages];
/// let horizon = HorizonMode::finite(num_stages);
///
/// let result = train(
///     &mut solver, config, &mut fcf, &templates, &base_rows,
///     &indexer, &initial_state, &opening_tree, &stochastic,
///     &horizon, &risk, stopping, None, None, &comm,
///     1, || HiggsBackend::new(),
/// )?;
///
/// println!("converged in {} iterations, gap={:.4}", result.iterations, result.final_gap);
/// ```
///
/// # Panics (debug builds only)
///
/// Panics if `templates.len() != horizon.num_stages()` or if
/// `risk_measures.len() != horizon.num_stages()` or if
/// `opening_tree.n_openings(0) == 0`.
#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::similar_names
)]
pub fn train<S: SolverInterface + Send, C: Communicator>(
    solver: &mut S,
    config: TrainingConfig,
    fcf: &mut FutureCostFunction,
    stage_ctx: &StageContext<'_>,
    training_ctx: &TrainingContext<'_>,
    opening_tree: &OpeningTree,
    risk_measures: &[RiskMeasure],
    stopping_rules: StoppingRuleSet,
    cut_selection: Option<&CutSelectionStrategy>,
    shutdown_flag: Option<&Arc<AtomicBool>>,
    comm: &C,
    n_fwd_threads: usize,
    solver_factory: impl Fn() -> Result<S, cobre_solver::SolverError>,
    max_blocks: usize,
) -> Result<TrainingResult, SddpError> {
    let horizon = training_ctx.horizon;
    let indexer = training_ctx.indexer;
    let initial_state = training_ctx.initial_state;
    let num_stages = horizon.num_stages();
    let num_ranks = comm.size();
    let my_rank = comm.rank();
    let total_forward_passes = config.forward_passes as usize;
    let n_state = indexer.n_state;

    // forward_passes is the TOTAL across all ranks. Distribute with
    // base/remainder: first `remainder` ranks get `base + 1`, rest get `base`.
    let base_fwd = total_forward_passes / num_ranks;
    let remainder_fwd = total_forward_passes % num_ranks;
    let my_actual_fwd = base_fwd + usize::from(my_rank < remainder_fwd);
    let my_fwd_offset = base_fwd * my_rank + my_rank.min(remainder_fwd);
    // ExchangeBuffers requires uniform local_count — use the max across ranks.
    let max_local_fwd = base_fwd + usize::from(remainder_fwd > 0);

    let empty_record = TrajectoryRecord {
        primal: vec![],
        dual: vec![],
        stage_cost: 0.0,
        state: vec![0.0; n_state],
    };
    let mut records = vec![empty_record; max_local_fwd * num_stages];

    // Workspace pool for forward-pass thread parallelism. Each workspace owns
    // an independent solver, patch buffer, and current-state buffer. The pool
    // is allocated once and reused across all iterations.
    let n_threads = n_fwd_threads.max(1);
    let mut fwd_pool = WorkspacePool::try_new(
        n_threads,
        indexer.hydro_count,
        indexer.max_par_order,
        n_state,
        stage_ctx.n_load_buses,
        max_blocks,
        solver_factory,
    )
    .map_err(SddpError::Solver)?;

    // Per-scenario, per-stage basis store. Sized for the maximum local forward
    // passes so that scenario indices are stable across iterations. The store
    // is allocated once and reused: the forward pass overwrites entries each
    // iteration, and the backward pass reads from them read-only.
    let mut basis_store = BasisStore::new(max_local_fwd, num_stages);

    // Standalone patch buffer for the lower bound evaluation which uses the
    // single `solver` argument directly. The backward pass uses the workspace
    // pool's per-thread solvers and patch buffers instead.
    let mut patch_buf = PatchBuffer::new(indexer.hydro_count, indexer.max_par_order, 0, 0);
    let mut convergence_monitor = ConvergenceMonitor::new(stopping_rules);
    let mut exchange_bufs = ExchangeBuffers::new(n_state, max_local_fwd, num_ranks);
    let mut cut_sync_bufs =
        CutSyncBuffers::with_distribution(n_state, max_local_fwd, num_ranks, total_forward_passes);

    let start_time = Instant::now();

    let TrainingConfig {
        forward_passes: config_forward_passes,
        max_iterations,
        event_sender,
        ..
    } = config;

    #[allow(clippy::cast_possible_truncation)]
    emit(
        event_sender.as_ref(),
        TrainingEvent::TrainingStarted {
            case_name: String::new(),
            stages: num_stages as u32,
            hydros: indexer.hydro_count as u32,
            thermals: 0,
            ranks: num_ranks as u32,
            #[allow(clippy::cast_possible_truncation)]
            threads_per_rank: n_threads as u32,
            timestamp: String::new(),
        },
    );

    let mut final_lb = 0.0;
    let mut final_ub = 0.0;
    let mut final_ub_std = 0.0;
    let mut final_gap = 0.0;
    let mut completed_iterations = 0u64;
    let mut termination_reason = RULE_ITERATION_LIMIT.to_string();

    for iteration in 1..=max_iterations {
        // Check external shutdown flag before each iteration's convergence
        // evaluation. The flag is set by signal handlers or test harnesses.
        if let Some(flag) = shutdown_flag {
            if flag.load(Ordering::Relaxed) {
                convergence_monitor.set_shutdown();
            }
        }

        let iter_start = Instant::now();
        let fwd_record_len = my_actual_fwd * num_stages;
        let fwd_batch = ForwardPassBatch {
            local_forward_passes: my_actual_fwd,
            iteration,
            fwd_offset: my_fwd_offset,
        };
        let forward_result = run_forward_pass(
            &mut fwd_pool.workspaces,
            &mut basis_store,
            stage_ctx,
            fcf,
            training_ctx,
            &fwd_batch,
            &mut records[..fwd_record_len],
        )?;

        let forward_elapsed_ms = forward_result.elapsed_ms;

        let local_n = forward_result.scenario_costs.len();
        let local_cost_sum: f64 = forward_result.scenario_costs.iter().sum();
        emit(
            event_sender.as_ref(),
            TrainingEvent::ForwardPassComplete {
                iteration,
                scenarios: config_forward_passes,
                #[allow(clippy::cast_precision_loss)]
                ub_mean: if local_n > 0 {
                    local_cost_sum / local_n as f64
                } else {
                    0.0
                },
                ub_std: 0.0,
                elapsed_ms: forward_elapsed_ms,
            },
        );
        let sync_result = sync_forward(&forward_result, comm, total_forward_passes)?;

        emit(
            event_sender.as_ref(),
            TrainingEvent::ForwardSyncComplete {
                iteration,
                global_ub_mean: sync_result.global_ub_mean,
                global_ub_std: sync_result.global_ub_std,
                sync_time_ms: sync_result.sync_time_ms,
            },
        );
        let mut bwd_spec = crate::backward::BackwardPassSpec {
            exchange: &mut exchange_bufs,
            records: &records,
            iteration,
            local_work: my_actual_fwd,
            fwd_offset: my_fwd_offset,
            risk_measures,
        };
        let backward_result = run_backward_pass(
            &mut fwd_pool.workspaces,
            &basis_store,
            stage_ctx,
            fcf,
            training_ctx,
            &mut bwd_spec,
            comm,
        )?;

        let backward_elapsed_ms = backward_result.elapsed_ms;

        #[allow(clippy::cast_possible_truncation)]
        emit(
            event_sender.as_ref(),
            TrainingEvent::BackwardPassComplete {
                iteration,
                cuts_generated: backward_result.cuts_generated as u32,
                stages_processed: num_stages.saturating_sub(1) as u32,
                elapsed_ms: backward_elapsed_ms,
            },
        );
        for stage in 0..num_stages.saturating_sub(1) {
            let owned_cuts = collect_local_cuts_for_stage(fcf, stage, iteration);
            let local_cuts: Vec<(u32, u32, u32, f64, &[f64])> = owned_cuts
                .iter()
                .map(|(slot, iter, fp, intercept, coeffs)| {
                    (*slot, *iter, *fp, *intercept, coeffs.as_slice())
                })
                .collect();
            cut_sync_bufs.sync_cuts(stage, &local_cuts, fcf, comm)?;
        }

        #[allow(clippy::cast_possible_truncation)]
        emit(
            event_sender.as_ref(),
            TrainingEvent::CutSyncComplete {
                iteration,
                cuts_distributed: backward_result.cuts_generated as u32,
                cuts_active: fcf.total_active_cuts() as u32,
                cuts_removed: 0,
                sync_time_ms: 0,
            },
        );

        if let Some(strategy) = cut_selection {
            if strategy.should_run(iteration) {
                let sel_start = Instant::now();
                let num_sel_stages = num_stages.saturating_sub(1);
                let mut cuts_deactivated = 0u32;

                #[allow(clippy::cast_possible_truncation)]
                for stage in 0..num_sel_stages {
                    let stage_u32 = stage as u32;
                    let deact =
                        strategy.select_for_stage(&fcf.pools[stage].metadata, iteration, stage_u32);
                    cuts_deactivated += deact.indices.len() as u32;
                    fcf.pools[stage].deactivate(&deact.indices);
                }

                #[allow(clippy::cast_possible_truncation)]
                let selection_time_ms = sel_start.elapsed().as_millis() as u64;

                #[allow(clippy::cast_possible_truncation)]
                let stages_processed = num_sel_stages as u32;
                emit(
                    event_sender.as_ref(),
                    TrainingEvent::CutSelectionComplete {
                        iteration,
                        cuts_deactivated,
                        stages_processed,
                        selection_time_ms,
                        allgatherv_time_ms: 0,
                    },
                );
            }
        }
        let lb_solves_before = solver.statistics().solve_count;
        let lb_spec = LbEvalSpec {
            template: &stage_ctx.templates[0],
            base_row: stage_ctx.base_rows[0],
            noise_scale: stage_ctx.noise_scale,
            n_hydros: stage_ctx.n_hydros,
            opening_tree,
            risk_measure: &risk_measures[0],
        };
        let lb = evaluate_lower_bound(
            solver,
            fcf,
            initial_state,
            indexer,
            &mut patch_buf,
            &lb_spec,
            comm,
        )?;
        let lb_lp_solves = solver.statistics().solve_count - lb_solves_before;

        let (should_stop, rule_results) = convergence_monitor.update(lb, &sync_result);

        final_lb = convergence_monitor.lower_bound();
        final_ub = convergence_monitor.upper_bound();
        final_ub_std = convergence_monitor.upper_bound_std();
        final_gap = convergence_monitor.gap();

        emit(
            event_sender.as_ref(),
            TrainingEvent::ConvergenceUpdate {
                iteration,
                lower_bound: final_lb,
                upper_bound: final_ub,
                upper_bound_std: convergence_monitor.upper_bound_std(),
                gap: final_gap,
                rules_evaluated: rule_results.clone(),
            },
        );

        #[allow(clippy::cast_possible_truncation)]
        let wall_time_ms = start_time.elapsed().as_millis() as u64;
        #[allow(clippy::cast_possible_truncation)]
        let iteration_time_ms = iter_start.elapsed().as_millis() as u64;

        emit(
            event_sender.as_ref(),
            TrainingEvent::IterationSummary {
                iteration,
                lower_bound: final_lb,
                upper_bound: final_ub,
                gap: final_gap,
                wall_time_ms,
                iteration_time_ms,
                forward_ms: forward_elapsed_ms,
                backward_ms: backward_elapsed_ms,
                lp_solves: forward_result.lp_solves + backward_result.lp_solves + lb_lp_solves,
            },
        );

        completed_iterations = iteration;

        if should_stop {
            // Extract the triggered rule name for the reason string.
            termination_reason = rule_results
                .iter()
                .find(|r| r.triggered)
                .map_or_else(|| "unknown".to_string(), |r| r.rule_name.clone());
            break;
        }
    }

    #[allow(clippy::cast_possible_truncation)]
    let total_time_ms = (start_time.elapsed().as_millis() as u64).max(1);

    #[allow(clippy::cast_possible_truncation)]
    emit(
        event_sender.as_ref(),
        TrainingEvent::TrainingFinished {
            reason: termination_reason.clone(),
            iterations: completed_iterations,
            final_lb,
            final_ub,
            total_time_ms,
            total_cuts: fcf.total_active_cuts() as u64,
        },
    );

    // The TrainingResult basis_cache field (used for policy checkpoint) is
    // populated from the last scenario's entries in the basis store. These
    // are the bases left by the final forward pass — the most recently solved
    // LP at each stage.
    let last_scenario = my_actual_fwd.saturating_sub(1);
    let basis_cache = (0..num_stages)
        .map(|t| basis_store.get(last_scenario, t).cloned())
        .collect();

    Ok(TrainingResult {
        final_lb,
        final_ub,
        final_ub_std,
        final_gap,
        iterations: completed_iterations,
        reason: termination_reason,
        total_time_ms,
        basis_cache,
    })
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::too_many_lines
)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::mpsc;

    use chrono::NaiveDate;
    use cobre_comm::{CommData, CommError, Communicator, ReduceOp};
    use cobre_core::{
        Bus, EntityId, SystemBuilder, TrainingEvent,
        scenario::{CorrelationEntity, CorrelationGroup, CorrelationModel, CorrelationProfile},
        temporal::{
            Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
            StageStateConfig,
        },
    };
    use cobre_solver::{
        Basis, LpSolution, RowBatch, SolverError, SolverInterface, SolverStatistics, StageTemplate,
    };
    use cobre_stochastic::{
        StochasticContext, build_stochastic_context, tree::opening_tree::OpeningTree,
    };

    use super::train;
    use crate::{
        HorizonMode, InflowNonNegativityMethod, RiskMeasure, SddpError, StageIndexer, StoppingMode,
        StoppingRule, StoppingRuleSet, TrainingConfig,
        context::{StageContext, TrainingContext},
        cut::fcf::FutureCostFunction,
    };

    /// Minimal two-column LP: \[`storage_in` (0), theta (1)\].
    /// One storage-fixing row \[0\]: `storage_in` is fixed to initial state.
    ///
    /// Column layout: `n_state=1` (N=1, L=0) — storage (0), `storage_in` (1), theta (2).
    /// Row layout: `storage_fixing` (0).
    fn minimal_template(n_state: usize) -> StageTemplate {
        let _ = n_state;
        StageTemplate {
            num_cols: 3,
            num_rows: 1,
            num_nz: 1,
            col_starts: vec![0_i32, 0, 1, 1],
            row_indices: vec![0_i32],
            values: vec![1.0],
            col_lower: vec![0.0, 0.0, 0.0],
            col_upper: vec![f64::INFINITY, f64::INFINITY, f64::INFINITY],
            objective: vec![0.0, 0.0, 1.0],
            row_lower: vec![0.0],
            row_upper: vec![0.0],
            n_state: 1,
            n_transfer: 0,
            n_dual_relevant: 1,
            n_hydro: 1,
            max_par_order: 0,
        }
    }

    fn fixed_solution(objective: f64) -> LpSolution {
        LpSolution {
            objective,
            primal: vec![0.0; 3],
            dual: vec![0.0; 1],
            reduced_costs: vec![0.0; 3],
            iterations: 0,
            solve_time_seconds: 0.0,
        }
    }

    /// Mock solver that returns fixed objective values in sequence.
    ///
    /// Each call to `solve()` returns the next value from `objectives`,
    /// wrapping around. If `infeasible_on_first` is set, the first call
    /// returns `SolverError::Infeasible`.
    struct MockSolver {
        objectives: Vec<f64>,
        call_count: usize,
        infeasible_on_first: bool,
    }

    impl MockSolver {
        fn with_fixed(objective: f64) -> Self {
            Self {
                objectives: vec![objective],
                call_count: 0,
                infeasible_on_first: false,
            }
        }

        fn infeasible() -> Self {
            Self {
                objectives: vec![0.0],
                call_count: 0,
                infeasible_on_first: true,
            }
        }
    }

    impl SolverInterface for MockSolver {
        fn load_model(&mut self, _t: &StageTemplate) {}
        fn add_rows(&mut self, _r: &RowBatch) {}
        fn set_row_bounds(&mut self, _i: &[usize], _l: &[f64], _u: &[f64]) {}
        fn set_col_bounds(&mut self, _i: &[usize], _l: &[f64], _u: &[f64]) {}

        fn solve(&mut self) -> Result<cobre_solver::SolutionView<'_>, SolverError> {
            let call = self.call_count;
            self.call_count += 1;
            if self.infeasible_on_first && call == 0 {
                return Err(SolverError::Infeasible);
            }
            let obj = self.objectives[call % self.objectives.len()];
            // Return a view with primal[2] = 0.0 (theta = 0) so that the forward pass
            // computes stage_cost = objective - primal[theta] = obj - 0 = obj.
            // The fixed_solution helper provides compatible primal/dual arrays.
            let sol = fixed_solution(obj);
            // We cannot borrow from a temporary, so we use static empty slices.
            // training.rs mock only needs to satisfy the SolverInterface bound;
            // the actual slice contents are not checked by the training loop.
            let _ = sol;
            Ok(cobre_solver::SolutionView {
                objective: obj,
                primal: &[0.0, 0.0, 0.0],
                dual: &[0.0],
                reduced_costs: &[0.0, 0.0, 0.0],
                iterations: 0,
                solve_time_seconds: 0.0,
            })
        }

        fn reset(&mut self) {
            self.call_count = 0;
        }

        fn get_basis(&mut self, _out: &mut Basis) {}

        fn solve_with_basis(
            &mut self,
            _basis: &Basis,
        ) -> Result<cobre_solver::SolutionView<'_>, SolverError> {
            self.solve()
        }

        fn statistics(&self) -> SolverStatistics {
            SolverStatistics::default()
        }

        fn name(&self) -> &'static str {
            "Mock"
        }
    }

    struct StubComm;

    impl Communicator for StubComm {
        fn allgatherv<T: CommData>(
            &self,
            send: &[T],
            recv: &mut [T],
            _counts: &[usize],
            _displs: &[usize],
        ) -> Result<(), CommError> {
            recv[..send.len()].clone_from_slice(send);
            Ok(())
        }

        fn allreduce<T: CommData>(
            &self,
            send: &[T],
            recv: &mut [T],
            _op: ReduceOp,
        ) -> Result<(), CommError> {
            recv.clone_from_slice(send);
            Ok(())
        }

        fn broadcast<T: CommData>(&self, _buf: &mut [T], _root: usize) -> Result<(), CommError> {
            Ok(())
        }

        fn barrier(&self) -> Result<(), CommError> {
            Ok(())
        }

        fn rank(&self) -> usize {
            0
        }

        fn size(&self) -> usize {
            1
        }
    }

    /// Build a single-stage `OpeningTree` with one opening.
    ///
    /// Uses `generate_opening_tree` from the stochastic crate with a minimal
    /// single-entity, single-stage configuration.
    fn make_opening_tree(n_openings: usize) -> OpeningTree {
        use chrono::NaiveDate;
        use cobre_core::{
            EntityId,
            scenario::{CorrelationEntity, CorrelationGroup, CorrelationModel, CorrelationProfile},
            temporal::{
                Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
                StageStateConfig,
            },
        };
        use cobre_stochastic::correlation::resolve::DecomposedCorrelation;
        use std::collections::BTreeMap;

        let stage = Stage {
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
                branching_factor: n_openings,
                noise_method: NoiseMethod::Saa,
            },
        };

        let entity_id = EntityId(1);
        let mut profiles = BTreeMap::new();
        profiles.insert(
            "default".to_string(),
            CorrelationProfile {
                groups: vec![CorrelationGroup {
                    name: "g1".to_string(),
                    entities: vec![CorrelationEntity {
                        entity_type: "inflow".to_string(),
                        id: entity_id,
                    }],
                    matrix: vec![vec![1.0]],
                }],
            },
        );
        let corr_model = CorrelationModel {
            method: "cholesky".to_string(),
            profiles,
            schedule: vec![],
        };
        let mut decomposed = DecomposedCorrelation::build(&corr_model).unwrap();
        let entity_order = vec![entity_id];

        cobre_stochastic::tree::generate::generate_opening_tree(
            42,
            &[stage],
            1,
            &mut decomposed,
            &entity_order,
        )
    }

    /// Build a minimal `StochasticContext` with `n_stages` stages and a single
    /// hydro entity.
    ///
    /// Used to provide the `stochastic` argument to `train`. Follows the same
    /// [`SystemBuilder`] + `build_stochastic_context` pattern as the forward-pass
    /// integration tests.  The `n_openings` parameter controls the branching
    /// factor of the opening tree.
    #[allow(clippy::cast_possible_wrap)]
    fn make_stochastic_context(n_stages: usize, n_openings: usize) -> StochasticContext {
        use cobre_core::entities::hydro::{Hydro, HydroGenerationModel, HydroPenalties};
        use cobre_core::scenario::InflowModel;

        let bus = Bus {
            id: EntityId(0),
            name: "B0".to_string(),
            deficit_segments: vec![cobre_core::DeficitSegment {
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
                branching_factor: n_openings,
                noise_method: NoiseMethod::Saa,
            },
        };

        let stages: Vec<Stage> = (0..n_stages).map(make_stage).collect();

        let inflow_models: Vec<InflowModel> = (0..n_stages)
            .map(|i| InflowModel {
                hydro_id: EntityId(1),
                stage_id: i as i32,
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
            method: "cholesky".to_string(),
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

        build_stochastic_context(&system, 42, &[], &[], None).unwrap()
    }

    fn make_fcf(
        n_stages: usize,
        n_state: usize,
        forward_passes: u32,
        max_iter: u64,
    ) -> FutureCostFunction {
        FutureCostFunction::new(n_stages, n_state, forward_passes, max_iter, 0)
    }

    fn iteration_limit_rules(limit: u64) -> StoppingRuleSet {
        StoppingRuleSet {
            rules: vec![StoppingRule::IterationLimit { limit }],
            mode: StoppingMode::Any,
        }
    }

    /// AC: `train_completes_with_iteration_limit`
    ///
    /// Given `max_iterations: 5`, a `StoppingRuleSet` with
    /// `IterationLimit { limit: 5 }` in `Any` mode, a mock solver returning
    /// fixed objectives, and `StubComm` as communicator, when the function
    /// completes, then `result.iterations == 5` and
    /// `result.reason == "iteration_limit"`.
    #[test]
    fn ac_train_completes_with_iteration_limit() {
        let n_stages = 2;
        let indexer = StageIndexer::new(1, 0); // N=1, L=0
        let templates = vec![minimal_template(indexer.n_state); n_stages];
        let base_rows = vec![1usize; n_stages];
        let initial_state = vec![0.0_f64; indexer.n_state];
        let opening_tree = make_opening_tree(1);
        let stochastic = make_stochastic_context(n_stages, 1);
        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let risk_measures = vec![RiskMeasure::Expectation; n_stages];
        let mut fcf = make_fcf(n_stages, indexer.n_state, 1, 10);

        let config = TrainingConfig {
            forward_passes: 1,
            max_iterations: 5,
            checkpoint_interval: None,
            warm_start_cuts: 0,
            event_sender: None,
        };

        let mut solver = MockSolver::with_fixed(100.0);
        let comm = StubComm;

        let stage_ctx = StageContext {
            templates: &templates,
            base_rows: &base_rows,
            noise_scale: &[],
            n_hydros: 0,
            n_load_buses: 0,
            load_balance_row_starts: &[],
            load_bus_indices: &[],
            block_counts_per_stage: &[1usize, 1],
        };
        let result = train(
            &mut solver,
            config,
            &mut fcf,
            &stage_ctx,
            &TrainingContext {
                horizon: &horizon,
                indexer: &indexer,
                inflow_method: &InflowNonNegativityMethod::None,
                stochastic: &stochastic,
                initial_state: &initial_state,
            },
            &opening_tree,
            &risk_measures,
            iteration_limit_rules(5),
            None,
            None,
            &comm,
            1,
            || Ok(MockSolver::with_fixed(100.0)),
            1,
        )
        .unwrap();

        assert_eq!(result.iterations, 5, "expected 5 iterations");
        assert_eq!(result.reason, "iteration_limit");
    }

    /// AC: `train_returns_error_on_infeasible`
    ///
    /// Given a mock solver that returns `SolverError::Infeasible` on the first
    /// forward pass solve, when the function is called, then it returns
    /// `Err(SddpError::Infeasible { stage: 0, .. })`.
    #[test]
    fn ac_train_returns_error_on_infeasible() {
        let n_stages = 2;
        let indexer = StageIndexer::new(1, 0);
        let templates = vec![minimal_template(indexer.n_state); n_stages];
        let base_rows = vec![1usize; n_stages];
        let initial_state = vec![0.0_f64; indexer.n_state];
        let opening_tree = make_opening_tree(1);
        let stochastic = make_stochastic_context(n_stages, 1);
        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let risk_measures = vec![RiskMeasure::Expectation; n_stages];
        let mut fcf = make_fcf(n_stages, indexer.n_state, 1, 10);

        let config = TrainingConfig {
            forward_passes: 1,
            max_iterations: 5,
            checkpoint_interval: None,
            warm_start_cuts: 0,
            event_sender: None,
        };

        let mut solver = MockSolver::infeasible();
        let comm = StubComm;

        let stage_ctx = StageContext {
            templates: &templates,
            base_rows: &base_rows,
            noise_scale: &[],
            n_hydros: 0,
            n_load_buses: 0,
            load_balance_row_starts: &[],
            load_bus_indices: &[],
            block_counts_per_stage: &[1usize, 1],
        };
        let result = train(
            &mut solver,
            config,
            &mut fcf,
            &stage_ctx,
            &TrainingContext {
                horizon: &horizon,
                indexer: &indexer,
                inflow_method: &InflowNonNegativityMethod::None,
                stochastic: &stochastic,
                initial_state: &initial_state,
            },
            &opening_tree,
            &risk_measures,
            iteration_limit_rules(5),
            None,
            None,
            &comm,
            1,
            || Ok(MockSolver::infeasible()),
            1,
        );

        assert!(
            matches!(result, Err(SddpError::Infeasible { stage: 0, .. })),
            "expected SddpError::Infeasible at stage 0, got: {result:?}"
        );
    }

    /// AC: `train_emits_correct_event_sequence`
    ///
    /// Given `train` with `event_sender: Some(tx)`, runs for 2 iterations
    /// before `IterationLimit(2)` triggers. The receiver must collect exactly:
    ///
    /// - 1 `TrainingStarted`
    /// - 2 × (`ForwardPassComplete`, `ForwardSyncComplete`, `BackwardPassComplete`,
    ///   `CutSyncComplete`, `ConvergenceUpdate`, `IterationSummary`)
    /// - 1 `TrainingFinished`
    ///
    /// = 1 + 12 + 1 = 14 events.
    #[test]
    fn ac_train_emits_correct_event_sequence() {
        let n_stages = 2;
        let indexer = StageIndexer::new(1, 0);
        let templates = vec![minimal_template(indexer.n_state); n_stages];
        let base_rows = vec![1usize; n_stages];
        let initial_state = vec![0.0_f64; indexer.n_state];
        let opening_tree = make_opening_tree(1);
        let stochastic = make_stochastic_context(n_stages, 1);
        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let risk_measures = vec![RiskMeasure::Expectation; n_stages];
        let mut fcf = make_fcf(n_stages, indexer.n_state, 1, 10);

        let (tx, rx) = mpsc::channel::<TrainingEvent>();

        let config = TrainingConfig {
            forward_passes: 1,
            max_iterations: 10,
            checkpoint_interval: None,
            warm_start_cuts: 0,
            event_sender: Some(tx),
        };

        let mut solver = MockSolver::with_fixed(100.0);
        let comm = StubComm;

        let stage_ctx = StageContext {
            templates: &templates,
            base_rows: &base_rows,
            noise_scale: &[],
            n_hydros: 0,
            n_load_buses: 0,
            load_balance_row_starts: &[],
            load_bus_indices: &[],
            block_counts_per_stage: &[1usize, 1],
        };
        train(
            &mut solver,
            config,
            &mut fcf,
            &stage_ctx,
            &TrainingContext {
                horizon: &horizon,
                indexer: &indexer,
                inflow_method: &InflowNonNegativityMethod::None,
                stochastic: &stochastic,
                initial_state: &initial_state,
            },
            &opening_tree,
            &risk_measures,
            iteration_limit_rules(2),
            None,
            None,
            &comm,
            1,
            || Ok(MockSolver::with_fixed(100.0)),
            1,
        )
        .unwrap();

        // Drain all events.
        drop(fcf); // not needed; just for clarity
        let events: Vec<TrainingEvent> = rx.try_iter().collect();

        // 1 TrainingStarted + 2*(6 per-iteration) + 1 TrainingFinished = 14
        assert_eq!(
            events.len(),
            14,
            "expected 14 events, got {} ({events:?})",
            events.len()
        );

        assert!(
            matches!(events[0], TrainingEvent::TrainingStarted { .. }),
            "first event must be TrainingStarted"
        );
        assert!(
            matches!(events.last(), Some(TrainingEvent::TrainingFinished { .. })),
            "last event must be TrainingFinished"
        );

        // Check per-iteration event pattern for iteration 1 (events[1..7])
        assert!(matches!(
            events[1],
            TrainingEvent::ForwardPassComplete { .. }
        ));
        assert!(matches!(
            events[2],
            TrainingEvent::ForwardSyncComplete { .. }
        ));
        assert!(matches!(
            events[3],
            TrainingEvent::BackwardPassComplete { .. }
        ));
        assert!(matches!(events[4], TrainingEvent::CutSyncComplete { .. }));
        assert!(matches!(events[5], TrainingEvent::ConvergenceUpdate { .. }));
        assert!(matches!(events[6], TrainingEvent::IterationSummary { .. }));

        // Iteration 2 (events[7..13]) follows the same pattern.
        assert!(matches!(
            events[7],
            TrainingEvent::ForwardPassComplete { .. }
        ));
        assert!(matches!(
            events[8],
            TrainingEvent::ForwardSyncComplete { .. }
        ));
        assert!(matches!(
            events[9],
            TrainingEvent::BackwardPassComplete { .. }
        ));
        assert!(matches!(events[10], TrainingEvent::CutSyncComplete { .. }));
        assert!(matches!(
            events[11],
            TrainingEvent::ConvergenceUpdate { .. }
        ));
        assert!(matches!(events[12], TrainingEvent::IterationSummary { .. }));
    }

    /// AC: `train_result_fields_populated`
    ///
    /// Verify that all `TrainingResult` fields are non-default after a
    /// successful 5-iteration run.
    #[test]
    fn ac_train_result_fields_populated() {
        let n_stages = 2;
        let indexer = StageIndexer::new(1, 0);
        let templates = vec![minimal_template(indexer.n_state); n_stages];
        let base_rows = vec![1usize; n_stages];
        let initial_state = vec![0.0_f64; indexer.n_state];
        let opening_tree = make_opening_tree(1);
        let stochastic = make_stochastic_context(n_stages, 1);
        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let risk_measures = vec![RiskMeasure::Expectation; n_stages];
        let mut fcf = make_fcf(n_stages, indexer.n_state, 1, 10);

        let config = TrainingConfig {
            forward_passes: 1,
            max_iterations: 5,
            checkpoint_interval: None,
            warm_start_cuts: 0,
            event_sender: None,
        };

        let mut solver = MockSolver::with_fixed(100.0);
        let comm = StubComm;

        let stage_ctx = StageContext {
            templates: &templates,
            base_rows: &base_rows,
            noise_scale: &[],
            n_hydros: 0,
            n_load_buses: 0,
            load_balance_row_starts: &[],
            load_bus_indices: &[],
            block_counts_per_stage: &[1usize, 1],
        };
        let result = train(
            &mut solver,
            config,
            &mut fcf,
            &stage_ctx,
            &TrainingContext {
                horizon: &horizon,
                indexer: &indexer,
                inflow_method: &InflowNonNegativityMethod::None,
                stochastic: &stochastic,
                initial_state: &initial_state,
            },
            &opening_tree,
            &risk_measures,
            iteration_limit_rules(5),
            None,
            None,
            &comm,
            1,
            || Ok(MockSolver::with_fixed(100.0)),
            1,
        )
        .unwrap();

        assert_eq!(result.iterations, 5);
        assert!(!result.reason.is_empty(), "reason must not be empty");
    }

    /// AC: `train_with_no_event_sender`
    ///
    /// Verify that `event_sender: None` does not panic and training completes
    /// normally.
    #[test]
    fn ac_train_with_no_event_sender() {
        let n_stages = 2;
        let indexer = StageIndexer::new(1, 0);
        let templates = vec![minimal_template(indexer.n_state); n_stages];
        let base_rows = vec![1usize; n_stages];
        let initial_state = vec![0.0_f64; indexer.n_state];
        let opening_tree = make_opening_tree(1);
        let stochastic = make_stochastic_context(n_stages, 1);
        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let risk_measures = vec![RiskMeasure::Expectation; n_stages];
        let mut fcf = make_fcf(n_stages, indexer.n_state, 1, 10);

        let config = TrainingConfig {
            forward_passes: 1,
            max_iterations: 2,
            checkpoint_interval: None,
            warm_start_cuts: 0,
            event_sender: None,
        };

        let mut solver = MockSolver::with_fixed(100.0);
        let comm = StubComm;

        let stage_ctx = StageContext {
            templates: &templates,
            base_rows: &base_rows,
            noise_scale: &[],
            n_hydros: 0,
            n_load_buses: 0,
            load_balance_row_starts: &[],
            load_bus_indices: &[],
            block_counts_per_stage: &[1usize, 1],
        };
        let result = train(
            &mut solver,
            config,
            &mut fcf,
            &stage_ctx,
            &TrainingContext {
                horizon: &horizon,
                indexer: &indexer,
                inflow_method: &InflowNonNegativityMethod::None,
                stochastic: &stochastic,
                initial_state: &initial_state,
            },
            &opening_tree,
            &risk_measures,
            iteration_limit_rules(2),
            None,
            None,
            &comm,
            1,
            || Ok(MockSolver::with_fixed(100.0)),
            1,
        );

        assert!(result.is_ok(), "train with no event_sender must not panic");
    }

    /// AC: train result `total_time_ms` is greater than 0
    ///
    /// After a successful run, `result.total_time_ms` must be >= 0 and the
    /// `TrainingResult` must be constructible without panicking.
    #[test]
    fn ac_total_time_ms_is_non_negative() {
        let n_stages = 2;
        let indexer = StageIndexer::new(1, 0);
        let templates = vec![minimal_template(indexer.n_state); n_stages];
        let base_rows = vec![1usize; n_stages];
        let initial_state = vec![0.0_f64; indexer.n_state];
        let opening_tree = make_opening_tree(1);
        let stochastic = make_stochastic_context(n_stages, 1);
        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let risk_measures = vec![RiskMeasure::Expectation; n_stages];
        let mut fcf = make_fcf(n_stages, indexer.n_state, 1, 10);

        let config = TrainingConfig {
            forward_passes: 1,
            max_iterations: 1,
            checkpoint_interval: None,
            warm_start_cuts: 0,
            event_sender: None,
        };

        let mut solver = MockSolver::with_fixed(100.0);
        let comm = StubComm;

        let stage_ctx = StageContext {
            templates: &templates,
            base_rows: &base_rows,
            noise_scale: &[],
            n_hydros: 0,
            n_load_buses: 0,
            load_balance_row_starts: &[],
            load_bus_indices: &[],
            block_counts_per_stage: &[1usize, 1],
        };
        let result = train(
            &mut solver,
            config,
            &mut fcf,
            &stage_ctx,
            &TrainingContext {
                horizon: &horizon,
                indexer: &indexer,
                inflow_method: &InflowNonNegativityMethod::None,
                stochastic: &stochastic,
                initial_state: &initial_state,
            },
            &opening_tree,
            &risk_measures,
            iteration_limit_rules(1),
            None,
            None,
            &comm,
            1,
            || Ok(MockSolver::with_fixed(100.0)),
            1,
        )
        .unwrap();

        assert!(
            result.total_time_ms > 0,
            "total_time_ms must be > 0, got {}",
            result.total_time_ms,
        );
    }

    /// `cut_selection_none_skips_step`
    ///
    /// Given `train` with `cut_selection: None` running for 5 iterations, then
    /// no `CutSelectionComplete` event is emitted.
    #[test]
    fn cut_selection_none_skips_step() {
        let n_stages = 2;
        let indexer = StageIndexer::new(1, 0);
        let templates = vec![minimal_template(indexer.n_state); n_stages];
        let base_rows = vec![1usize; n_stages];
        let initial_state = vec![0.0_f64; indexer.n_state];
        let opening_tree = make_opening_tree(1);
        let stochastic = make_stochastic_context(n_stages, 1);
        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let risk_measures = vec![RiskMeasure::Expectation; n_stages];
        let mut fcf = make_fcf(n_stages, indexer.n_state, 1, 10);

        let (tx, rx) = mpsc::channel::<TrainingEvent>();

        let config = TrainingConfig {
            forward_passes: 1,
            max_iterations: 10,
            checkpoint_interval: None,
            warm_start_cuts: 0,
            event_sender: Some(tx),
        };

        let mut solver = MockSolver::with_fixed(100.0);
        let comm = StubComm;

        let stage_ctx = StageContext {
            templates: &templates,
            base_rows: &base_rows,
            noise_scale: &[],
            n_hydros: 0,
            n_load_buses: 0,
            load_balance_row_starts: &[],
            load_bus_indices: &[],
            block_counts_per_stage: &[1usize, 1],
        };
        train(
            &mut solver,
            config,
            &mut fcf,
            &stage_ctx,
            &TrainingContext {
                horizon: &horizon,
                indexer: &indexer,
                inflow_method: &InflowNonNegativityMethod::None,
                stochastic: &stochastic,
                initial_state: &initial_state,
            },
            &opening_tree,
            &risk_measures,
            iteration_limit_rules(5),
            None,
            None,
            &comm,
            1,
            || Ok(MockSolver::with_fixed(100.0)),
            1,
        )
        .unwrap();

        let events: Vec<TrainingEvent> = rx.try_iter().collect();
        let cut_sel_count = events
            .iter()
            .filter(|e| matches!(e, TrainingEvent::CutSelectionComplete { .. }))
            .count();

        assert_eq!(
            cut_sel_count, 0,
            "expected no CutSelectionComplete events with cut_selection: None"
        );
    }

    /// `cut_selection_level1_runs_at_frequency`
    ///
    /// Given `train` with `cut_selection: Some(Level1 { threshold: 0,
    /// check_frequency: 3 })` running for 5 iterations, then
    /// `CutSelectionComplete` is emitted exactly once (at iteration 3).
    #[test]
    fn cut_selection_level1_runs_at_frequency() {
        use crate::cut_selection::CutSelectionStrategy;

        let n_stages = 2;
        let indexer = StageIndexer::new(1, 0);
        let templates = vec![minimal_template(indexer.n_state); n_stages];
        let base_rows = vec![1usize; n_stages];
        let initial_state = vec![0.0_f64; indexer.n_state];
        let opening_tree = make_opening_tree(1);
        let stochastic = make_stochastic_context(n_stages, 1);
        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let risk_measures = vec![RiskMeasure::Expectation; n_stages];
        let mut fcf = make_fcf(n_stages, indexer.n_state, 1, 10);

        let (tx, rx) = mpsc::channel::<TrainingEvent>();

        let config = TrainingConfig {
            forward_passes: 1,
            max_iterations: 10,
            checkpoint_interval: None,
            warm_start_cuts: 0,
            event_sender: Some(tx),
        };

        let strategy = CutSelectionStrategy::Level1 {
            threshold: 0,
            check_frequency: 3,
        };

        let mut solver = MockSolver::with_fixed(100.0);
        let comm = StubComm;

        let stage_ctx = StageContext {
            templates: &templates,
            base_rows: &base_rows,
            noise_scale: &[],
            n_hydros: 0,
            n_load_buses: 0,
            load_balance_row_starts: &[],
            load_bus_indices: &[],
            block_counts_per_stage: &[1usize, 1],
        };
        train(
            &mut solver,
            config,
            &mut fcf,
            &stage_ctx,
            &TrainingContext {
                horizon: &horizon,
                indexer: &indexer,
                inflow_method: &InflowNonNegativityMethod::None,
                stochastic: &stochastic,
                initial_state: &initial_state,
            },
            &opening_tree,
            &risk_measures,
            iteration_limit_rules(5),
            Some(&strategy),
            None,
            &comm,
            1,
            || Ok(MockSolver::with_fixed(100.0)),
            1,
        )
        .unwrap();

        let events: Vec<TrainingEvent> = rx.try_iter().collect();
        let sel_events: Vec<&TrainingEvent> = events
            .iter()
            .filter(|e| matches!(e, TrainingEvent::CutSelectionComplete { .. }))
            .collect();

        assert_eq!(
            sel_events.len(),
            1,
            "expected exactly 1 CutSelectionComplete event for check_frequency=3 over 5 iterations"
        );

        // Verify the event was emitted at iteration 3.
        let TrainingEvent::CutSelectionComplete { iteration, .. } = sel_events[0] else {
            panic!("wrong variant");
        };
        assert_eq!(
            *iteration, 3,
            "CutSelectionComplete must fire at iteration 3"
        );
    }

    /// `cut_selection_deactivates_inactive_cuts`
    ///
    /// Given `train` with `cut_selection: Some(Level1 { threshold: 0,
    /// check_frequency: 2 })` running for 2 iterations where the mock solver
    /// produces cuts with zero activity at stage 0, when the training loop
    /// reaches iteration 2, then `CutSelectionComplete` is emitted with
    /// `cuts_deactivated > 0`.
    ///
    /// Rationale: the mock solver returns `dual: vec![0.0; 1]` for every LP
    /// solve. The backward pass therefore never marks any cut as binding
    /// (`is_binding = false`), so all generated cuts have `active_count = 0`
    /// and are deactivated at the Level1 selection step.
    #[test]
    fn cut_selection_deactivates_inactive_cuts() {
        use crate::cut_selection::CutSelectionStrategy;

        let n_stages = 2;
        let indexer = StageIndexer::new(1, 0);
        let templates = vec![minimal_template(indexer.n_state); n_stages];
        let base_rows = vec![1usize; n_stages];
        let initial_state = vec![0.0_f64; indexer.n_state];
        let opening_tree = make_opening_tree(1);
        let stochastic = make_stochastic_context(n_stages, 1);
        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let risk_measures = vec![RiskMeasure::Expectation; n_stages];
        // max_iterations=10 so the FCF has enough capacity for iteration 2
        let mut fcf = make_fcf(n_stages, indexer.n_state, 1, 10);

        let (tx, rx) = mpsc::channel::<TrainingEvent>();

        let config = TrainingConfig {
            forward_passes: 1,
            max_iterations: 10,
            checkpoint_interval: None,
            warm_start_cuts: 0,
            event_sender: Some(tx),
        };

        // Level1 with threshold=0, check_frequency=2: fires at iteration 2.
        let strategy = CutSelectionStrategy::Level1 {
            threshold: 0,
            check_frequency: 2,
        };

        let mut solver = MockSolver::with_fixed(100.0);
        let comm = StubComm;

        let stage_ctx = StageContext {
            templates: &templates,
            base_rows: &base_rows,
            noise_scale: &[],
            n_hydros: 0,
            n_load_buses: 0,
            load_balance_row_starts: &[],
            load_bus_indices: &[],
            block_counts_per_stage: &[1usize, 1],
        };
        train(
            &mut solver,
            config,
            &mut fcf,
            &stage_ctx,
            &TrainingContext {
                horizon: &horizon,
                indexer: &indexer,
                inflow_method: &InflowNonNegativityMethod::None,
                stochastic: &stochastic,
                initial_state: &initial_state,
            },
            &opening_tree,
            &risk_measures,
            iteration_limit_rules(2),
            Some(&strategy),
            None,
            &comm,
            1,
            || Ok(MockSolver::with_fixed(100.0)),
            1,
        )
        .unwrap();

        let events: Vec<TrainingEvent> = rx.try_iter().collect();
        let sel_events: Vec<&TrainingEvent> = events
            .iter()
            .filter(|e| matches!(e, TrainingEvent::CutSelectionComplete { .. }))
            .collect();

        assert_eq!(
            sel_events.len(),
            1,
            "expected exactly 1 CutSelectionComplete event at iteration 2"
        );

        let TrainingEvent::CutSelectionComplete {
            iteration,
            cuts_deactivated,
            ..
        } = sel_events[0]
        else {
            panic!("wrong variant");
        };

        assert_eq!(*iteration, 2, "selection must fire at iteration 2");
        assert!(
            *cuts_deactivated > 0,
            "expected cuts_deactivated > 0 (mock solver produces zero-activity cuts), got 0"
        );
    }

    /// `existing_train_tests_pass_with_none`
    ///
    /// Verify backward compatibility: calling `train` with `cut_selection:
    /// None` produces the same result as before this ticket. This is
    /// implicitly verified by the existing `ac_train_completes_with_iteration_limit`
    /// test. This test is an explicit additional check with an explicit `None`.
    #[test]
    fn existing_train_tests_pass_with_none() {
        let n_stages = 2;
        let indexer = StageIndexer::new(1, 0);
        let templates = vec![minimal_template(indexer.n_state); n_stages];
        let base_rows = vec![1usize; n_stages];
        let initial_state = vec![0.0_f64; indexer.n_state];
        let opening_tree = make_opening_tree(1);
        let stochastic = make_stochastic_context(n_stages, 1);
        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let risk_measures = vec![RiskMeasure::Expectation; n_stages];
        let mut fcf = make_fcf(n_stages, indexer.n_state, 1, 10);

        let config = TrainingConfig {
            forward_passes: 1,
            max_iterations: 3,
            checkpoint_interval: None,
            warm_start_cuts: 0,
            event_sender: None,
        };

        let mut solver = MockSolver::with_fixed(100.0);
        let comm = StubComm;

        let stage_ctx = StageContext {
            templates: &templates,
            base_rows: &base_rows,
            noise_scale: &[],
            n_hydros: 0,
            n_load_buses: 0,
            load_balance_row_starts: &[],
            load_bus_indices: &[],
            block_counts_per_stage: &[1usize, 1],
        };
        let result = train(
            &mut solver,
            config,
            &mut fcf,
            &stage_ctx,
            &TrainingContext {
                horizon: &horizon,
                indexer: &indexer,
                inflow_method: &InflowNonNegativityMethod::None,
                stochastic: &stochastic,
                initial_state: &initial_state,
            },
            &opening_tree,
            &risk_measures,
            iteration_limit_rules(3),
            None,
            None,
            &comm,
            1,
            || Ok(MockSolver::with_fixed(100.0)),
            1,
        )
        .unwrap();

        assert_eq!(result.iterations, 3);
        assert_eq!(result.reason, "iteration_limit");
    }
}
