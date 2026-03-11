//! Simulation forward pass loop for SDDP policy evaluation.
//!
//! [`simulate`] evaluates the trained SDDP policy on a set of scenarios by
//! running a forward-only pass through all stages, extracting per-entity
//! results at each stage, streaming completed scenario results through a
//! bounded channel, and returning a compact cost buffer for MPI aggregation.
//!
//! ## LP rebuild sequence
//!
//! Identical to the training forward pass (`forward.rs`):
//!
//! 1. `solver.load_model(template)` — reset to the structural LP.
//! 2. `solver.add_rows(cut_batch)` — append Benders cuts from the trained FCF.
//! 3. `solver.set_row_bounds(...)` — patch scenario-specific row bounds.
//!
//! ## Work distribution
//!
//! Scenarios are distributed across MPI ranks via [`assign_scenarios`] using a
//! two-level distribution (fat/lean). Each rank processes its assigned range
//! independently; MPI aggregation is performed by the caller.
//!
//! Within a rank, scenarios are further distributed across [`SolverWorkspace`]
//! instances using the same static partitioning as the training forward pass.
//! Each workspace owns its solver, patch buffer, and current-state buffer
//! exclusively. A per-worker `Vec<Option<Basis>>` is used for warm-starting
//! across consecutive scenarios within the same worker.
//! Rayon's `par_iter_mut` drives the scenario loop; results are sorted by
//! `scenario_id` after the parallel region to ensure deterministic MPI
//! aggregation regardless of thread
//! scheduling order.
//!
//! ## Seed domain separation
//!
//! To avoid seed collisions with training forward pass seeds (which use
//! `global_scenario = rank * forward_passes + m`), the simulation domain adds
//! an offset of `u32::MAX / 2` to the scenario ID before passing it to
//! [`sample_forward`]. This places simulation seeds in a disjoint region of
//! the SipHash-1-3 seed space (deterministic SipHash-1-3 seeds for communication-free parallel noise).
//!
//! ## Hot-path allocation discipline
//!
//! No allocations occur per scenario or per stage during the inner loops.
//! Each [`SolverWorkspace`] pre-allocates its [`crate::PatchBuffer`] and
//! `current_state`. The per-worker `basis_cache` (`Vec<Option<Basis>>`) is
//! allocated once per parallel worker before the scenario loop. The
//! [`RowBatch`] per stage is built once before the scenario loop — not once
//! per scenario.

use std::sync::mpsc::{Sender, SyncSender};
use std::time::Instant;

use cobre_comm::Communicator;
use cobre_core::TrainingEvent;
use cobre_solver::{Basis, RowBatch, SolverError, SolverInterface, StageTemplate};
use cobre_stochastic::{StochasticContext, sample_forward};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};

use crate::{
    FutureCostFunction, HorizonMode, InflowNonNegativityMethod, StageIndexer,
    forward::{build_cut_row_batch, partition},
    simulation::{
        config::SimulationConfig,
        error::SimulationError,
        extraction::EntityCounts,
        extraction::{accumulate_category_costs, assign_scenarios, extract_stage_result},
        types::{ScenarioCategoryCosts, SimulationScenarioResult},
    },
    workspace::SolverWorkspace,
};

/// Offset added to the simulation scenario ID before passing to [`sample_forward`].
///
/// Separates the simulation seed domain from the training forward pass domain.
/// Training uses `global_scenario = rank * forward_passes + m`, while
/// simulation uses `global_scenario = SIMULATION_SEED_OFFSET + scenario_id`.
/// Both fit in `u32`; the offset guarantees no overlap for practical scenario counts.
const SIMULATION_SEED_OFFSET: u32 = u32::MAX / 2;

/// Per-worker scenario cost accumulation type.
///
/// Each parallel worker returns `Ok(WorkerCosts)` for its assigned scenarios.
/// The outer function flattens and sorts the results.
type WorkerCosts = Vec<(u32, f64, ScenarioCategoryCosts)>;

// ---------------------------------------------------------------------------
// Welford online accumulator for running mean and variance
// ---------------------------------------------------------------------------

/// Online accumulator for mean and variance using Welford's algorithm.
///
/// Accumulates one value at a time with O(1) updates and O(1) statistics
/// queries, with no re-scanning of previous data. Safe for use on the main
/// thread after the parallel region completes — not intended for concurrent
/// access.
struct WelfordAccumulator {
    count: u64,
    mean: f64,
    /// Sum of squared deviations from the running mean.
    m2: f64,
}

impl WelfordAccumulator {
    /// Create a new accumulator with no observations.
    fn new() -> Self {
        Self {
            count: 0,
            mean: 0.0,
            m2: 0.0,
        }
    }

    /// Incorporate a new observation into the running statistics.
    fn update(&mut self, value: f64) {
        self.count += 1;
        let delta = value - self.mean;
        #[allow(clippy::cast_precision_loss)] // count is bounded by n_scenarios (u32-range)
        let count_f64 = self.count as f64;
        self.mean += delta / count_f64;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;
    }

    /// Running mean of all observed values, or `0.0` if no observations.
    fn mean(&self) -> f64 {
        self.mean
    }

    /// Population variance (`m2 / n`), or `0.0` if fewer than 2 observations.
    ///
    /// Returns `0.0` for zero or one observation since variance is undefined
    /// with insufficient data.
    fn variance(&self) -> f64 {
        if self.count < 2 {
            0.0
        } else {
            #[allow(clippy::cast_precision_loss)] // count is bounded by n_scenarios (u32-range)
            let count_f64 = self.count as f64;
            self.m2 / count_f64
        }
    }

    /// Population standard deviation, or `0.0` if fewer than 2 observations.
    fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Half-width of the 95% confidence interval (`1.96 * std / sqrt(n)`).
    ///
    /// Returns `0.0` when fewer than 2 observations are available.
    fn ci_95_half_width(&self) -> f64 {
        if self.count < 2 {
            0.0
        } else {
            #[allow(clippy::cast_precision_loss)] // count is bounded by n_scenarios (u32-range)
            let count_f64 = self.count as f64;
            1.96 * self.std_dev() / count_f64.sqrt()
        }
    }
}

/// Emit a [`TrainingEvent::SimulationProgress`] event if a sender is present.
///
/// Channel send failures are silently ignored: progress reporting is
/// best-effort and must never interfere with the simulation result.
fn emit_simulation_progress(
    sender: Option<&Sender<TrainingEvent>>,
    scenarios_complete: u32,
    scenarios_total: u32,
    elapsed_ms: u64,
    acc: &WelfordAccumulator,
) {
    if let Some(s) = sender {
        let _ = s.send(TrainingEvent::SimulationProgress {
            scenarios_complete,
            scenarios_total,
            elapsed_ms,
            mean_cost: acc.mean(),
            std_cost: acc.std_dev(),
            ci_95_half_width: acc.ci_95_half_width(),
        });
    }
}

/// Evaluate the trained SDDP policy on this rank's assigned scenarios.
///
/// Distributes locally assigned scenarios across worker threads using the same
/// static partitioning as the training forward pass. Each [`SolverWorkspace`]
/// owns its solver, patch buffer, and current-state buffer exclusively — there
/// is no shared mutable state between workers. A per-worker basis cache
/// (`Vec<Option<Basis>>`) is created locally for warm-starting across scenarios.
///
/// `SyncSender::send()` is thread-safe; each worker sends its
/// [`SimulationScenarioResult`] through `result_tx` as it completes each
/// scenario. Channel send order may differ from scenario order, but the
/// returned cost buffer is sorted by `scenario_id` for deterministic MPI
/// aggregation.
///
/// Returns a compact cost buffer — one `(scenario_id, total_cost, category_costs)`
/// entry per locally solved scenario, sorted by `scenario_id` in ascending
/// order — for MPI aggregation by the caller.
///
/// ## Error handling
///
/// On `SolverError::Infeasible`, returns
/// `SimulationError::LpInfeasible { scenario_id, stage_id, solver_message }`.
/// On any other `SolverError`, returns
/// `SimulationError::SolverError { scenario_id, stage_id, solver_message }`.
/// On channel send failure (receiver dropped), returns
/// `SimulationError::ChannelClosed`.
///
/// Partial results already sent through the channel before an error are valid
/// and may be consumed by the receiver.
///
/// # Errors
///
/// Returns `Err(SimulationError::LpInfeasible { .. })` when a stage LP has no
/// feasible solution, `Err(SimulationError::SolverError { .. })` for other
/// terminal LP solver failures, and `Err(SimulationError::ChannelClosed)` when
/// the channel receiver has been dropped.
///
/// # Panics (debug builds only)
///
/// Panics if any of the following debug preconditions are violated:
///
/// - `templates.len() != num_stages`
/// - `base_rows.len() != num_stages`
/// - `initial_state.len() != indexer.n_state`
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_lines)]
pub fn simulate<S: SolverInterface + Send, C: Communicator>(
    workspaces: &mut [SolverWorkspace<S>],
    templates: &[StageTemplate],
    base_rows: &[usize],
    fcf: &FutureCostFunction,
    stochastic: &StochasticContext,
    config: &SimulationConfig,
    horizon: &HorizonMode,
    initial_state: &[f64],
    indexer: &StageIndexer,
    entity_counts: &EntityCounts,
    comm: &C,
    result_tx: &SyncSender<SimulationScenarioResult>,
    _inflow_method: &InflowNonNegativityMethod,
    noise_scale: &[f64],
    n_hydros: usize,
    zeta_per_stage: &[f64],
    block_hours_per_stage: &[Vec<f64>],
    event_sender: Option<&Sender<TrainingEvent>>,
) -> Result<Vec<(u32, f64, ScenarioCategoryCosts)>, SimulationError> {
    let num_stages = horizon.num_stages();
    let rank = comm.rank();
    let world_size = comm.size();

    debug_assert_eq!(
        templates.len(),
        num_stages,
        "templates.len() {got} != num_stages {expected}",
        got = templates.len(),
        expected = num_stages,
    );
    debug_assert_eq!(
        base_rows.len(),
        num_stages,
        "base_rows.len() {got} != num_stages {expected}",
        got = base_rows.len(),
        expected = num_stages,
    );
    debug_assert_eq!(
        initial_state.len(),
        indexer.n_state,
        "initial_state.len() {got} != indexer.n_state {expected}",
        got = initial_state.len(),
        expected = indexer.n_state,
    );

    // Build one cut RowBatch per stage before the scenario loop.
    // Cuts are the same for all scenarios — build once, reuse many times.
    let cut_batches: Vec<RowBatch> = (0..num_stages)
        .map(|t| build_cut_row_batch(fcf, t, indexer))
        .collect();

    let tree_view = stochastic.tree_view();
    let base_seed = stochastic.base_seed();

    // Determine this rank's scenario range.
    let scenario_range = assign_scenarios(config.n_scenarios, rank, world_size);
    #[allow(clippy::cast_possible_truncation)]
    let local_count = (scenario_range.end - scenario_range.start) as usize;
    let scenario_start = scenario_range.start as usize;

    let n_workers = workspaces.len().max(1);

    // Start the wall-clock timer for simulation progress events.
    // Only instantiated here; the Instant is cheaply created regardless of
    // whether event_sender is Some or None.
    let sim_start = Instant::now();

    // Execute the scenario loop in parallel over workspaces using static
    // partitioning. Each worker processes a contiguous sub-range of the
    // locally assigned scenarios and accumulates its own cost buffer entries.
    // Worker-local WorkerCosts are returned as Ok values; the first error from
    // any worker short-circuits the collect.
    let worker_results: Vec<Result<WorkerCosts, SimulationError>> = workspaces
        .par_iter_mut()
        .enumerate()
        .map(|(w, ws)| {
            let (start_local, end_local) = partition(local_count, n_workers, w);

            // Per-worker, per-stage basis cache for warm-starting across scenarios.
            // Simulation has no cross-scenario backward pass, so a simple
            // local Vec suffices: basis[t] is reused from one scenario to the
            // next within this worker.
            let mut basis_cache: Vec<Option<Basis>> = vec![None; num_stages];

            let mut worker_costs: Vec<(u32, f64, ScenarioCategoryCosts)> =
                Vec::with_capacity(end_local - start_local);

            for local_idx in start_local..end_local {
                #[allow(clippy::cast_possible_truncation)]
                let scenario_id = (scenario_start + local_idx) as u32;

                // Simulation seed domain separation: place simulation seeds in a
                // disjoint region from training seeds (SipHash-1-3 domain).
                let global_scenario = SIMULATION_SEED_OFFSET.saturating_add(scenario_id);

                // Initialize current state for this scenario.
                ws.current_state.clear();
                ws.current_state.extend_from_slice(initial_state);

                let mut total_cost = 0.0_f64;
                let mut category_costs = ScenarioCategoryCosts {
                    resource_cost: 0.0,
                    recourse_cost: 0.0,
                    violation_cost: 0.0,
                    regularization_cost: 0.0,
                    imputed_cost: 0.0,
                };

                // Collect per-stage results for the scenario result payload.
                let mut stage_results = Vec::with_capacity(num_stages);

                // Inner loop: one LP solve per stage.
                for t in 0..num_stages {
                    // Cast indices to u32 for the sampling API (SipHash-1-3 seed
                    // derivation uses u32 fields). Bounded by u32::MAX; truncation safe.
                    #[allow(clippy::cast_possible_truncation)]
                    let stage_id_u32 = t as u32;

                    let (_opening_idx, raw_noise) =
                        sample_forward(&tree_view, base_seed, 0, global_scenario, stage_id_u32, t);

                    // Transform raw η → ζ*base + ζ*σ*η for the water-balance
                    // RHS patch (same transformation as the training forward pass).
                    ws.noise_buf.clear();
                    let stage_offset = t * n_hydros;
                    for (h, &eta) in raw_noise.iter().enumerate().take(n_hydros) {
                        let base_rhs = templates[t].row_lower[base_rows[t] + h];
                        ws.noise_buf
                            .push(base_rhs + noise_scale[stage_offset + h] * eta);
                    }

                    // LP rebuild sequence: template → cuts → scenario-specific row bounds.
                    ws.solver.load_model(&templates[t]);
                    ws.solver.add_rows(&cut_batches[t]);

                    ws.patch_buf.fill_forward_patches(
                        indexer,
                        &ws.current_state,
                        &ws.noise_buf,
                        base_rows[t],
                    );
                    let patch_count = ws.patch_buf.forward_patch_count();
                    ws.solver.set_row_bounds(
                        &ws.patch_buf.indices[..patch_count],
                        &ws.patch_buf.lower[..patch_count],
                        &ws.patch_buf.upper[..patch_count],
                    );

                    let solve_result = match basis_cache[t].as_ref() {
                        Some(rb) => ws.solver.solve_with_basis(rb),
                        None => ws.solver.solve(),
                    };
                    let view = solve_result.map_err(|e| {
                        // Invalidate the basis on error before returning.
                        basis_cache[t] = None;
                        match e {
                            SolverError::Infeasible => SimulationError::LpInfeasible {
                                scenario_id,
                                stage_id: stage_id_u32,
                                solver_message: "LP infeasible".to_string(),
                            },
                            other => SimulationError::SolverError {
                                scenario_id,
                                stage_id: stage_id_u32,
                                solver_message: other.to_string(),
                            },
                        }
                    })?;

                    // Stage cost = LP objective minus theta (future cost variable).
                    let stage_cost = view.objective - view.primal[indexer.theta];
                    total_cost += stage_cost;

                    // Convert water-balance RHS (hm³) back to inflow (m³/s)
                    // for output reporting.
                    ws.inflow_m3s_buf.clear();
                    if let Some(&zeta) = zeta_per_stage.get(t) {
                        if zeta > 0.0 {
                            for &rhs_hm3 in &ws.noise_buf {
                                ws.inflow_m3s_buf.push(rhs_hm3 / zeta);
                            }
                        }
                    }

                    // Extract per-entity typed result for this stage.
                    let blk_hrs = block_hours_per_stage
                        .get(t)
                        .map_or(&[][..], |v| v.as_slice());
                    let stage_result = extract_stage_result(
                        view.primal,
                        view.dual,
                        view.objective,
                        &templates[t].objective,
                        &templates[t].row_lower,
                        indexer,
                        stage_id_u32,
                        entity_counts,
                        &ws.inflow_m3s_buf,
                        blk_hrs,
                    );

                    // Accumulate per-category costs for this stage.
                    for cost_entry in &stage_result.costs {
                        accumulate_category_costs(cost_entry, &mut category_costs);
                    }

                    stage_results.push(stage_result);

                    // Advance state to the outgoing storage + lags from this stage.
                    ws.current_state.clear();
                    ws.current_state
                        .extend_from_slice(&view.primal[..indexer.n_state]);

                    // Update local basis cache for warm-starting the next scenario.
                    if let Some(rb) = &mut basis_cache[t] {
                        ws.solver.get_basis(rb);
                    } else {
                        let mut rb = Basis::new(templates[t].num_cols, templates[t].num_rows);
                        ws.solver.get_basis(&mut rb);
                        basis_cache[t] = Some(rb);
                    }
                }

                // Build the scenario result and send through the bounded channel.
                // SyncSender is thread-safe: all workers share the same sender.
                let scenario_result = SimulationScenarioResult {
                    scenario_id,
                    total_cost,
                    per_category_costs: ScenarioCategoryCosts {
                        resource_cost: category_costs.resource_cost,
                        recourse_cost: category_costs.recourse_cost,
                        violation_cost: category_costs.violation_cost,
                        regularization_cost: category_costs.regularization_cost,
                        imputed_cost: category_costs.imputed_cost,
                    },
                    stages: stage_results,
                };

                // Retain the compact entry for MPI aggregation before consuming
                // `scenario_result` through the channel send.
                let compact_category = ScenarioCategoryCosts {
                    resource_cost: scenario_result.per_category_costs.resource_cost,
                    recourse_cost: scenario_result.per_category_costs.recourse_cost,
                    violation_cost: scenario_result.per_category_costs.violation_cost,
                    regularization_cost: scenario_result.per_category_costs.regularization_cost,
                    imputed_cost: scenario_result.per_category_costs.imputed_cost,
                };

                result_tx
                    .send(scenario_result)
                    .map_err(|_| SimulationError::ChannelClosed)?;

                worker_costs.push((scenario_id, total_cost, compact_category));
            }

            Ok(worker_costs)
        })
        .collect();

    // Flatten worker cost buffers and sort by scenario_id for deterministic
    // MPI aggregation regardless of thread completion order.
    //
    // While flattening, accumulate per-scenario total costs into the Welford
    // accumulator and emit one progress event per scenario. The parallel
    // region is already complete at this point, so the accumulator runs
    // single-threaded on the main thread after `collect()` returns. Emitting
    // per-scenario (rather than per-worker-batch) ensures the progress bar
    // animates in real-time regardless of thread count, including the common
    // single-threaded case where there is only one worker batch.
    let mut all_costs: Vec<(u32, f64, ScenarioCategoryCosts)> = Vec::with_capacity(local_count);
    let mut acc = WelfordAccumulator::new();
    for result in worker_results {
        let batch = result?;
        for &(_, total_cost, _) in &batch {
            acc.update(total_cost);
            // Emit a progress event after each scenario.
            // `scenarios_complete` reflects this rank's locally accumulated count.
            #[allow(clippy::cast_possible_truncation)]
            let scenarios_complete = acc.count as u32;
            let elapsed_ms = sim_start.elapsed().as_millis();
            #[allow(clippy::cast_possible_truncation)]
            let elapsed_ms_u64 = elapsed_ms as u64;
            emit_simulation_progress(
                event_sender,
                scenarios_complete,
                config.n_scenarios,
                elapsed_ms_u64,
                &acc,
            );
        }
        all_costs.extend(batch);
    }
    all_costs.sort_by_key(|&(id, _, _)| id);

    Ok(all_costs)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic, clippy::too_many_lines)]
mod tests {
    use std::sync::mpsc;

    use cobre_comm::{CommData, CommError, Communicator, ReduceOp};
    use cobre_solver::{
        Basis, LpSolution, RowBatch, SolverError, SolverInterface, SolverStatistics, StageTemplate,
    };
    use cobre_stochastic::StochasticContext;

    use super::simulate;
    use crate::{
        FutureCostFunction, HorizonMode, InflowNonNegativityMethod, PatchBuffer, StageIndexer,
        simulation::{config::SimulationConfig, error::SimulationError, extraction::EntityCounts},
        workspace::SolverWorkspace,
    };

    // ── Stub communicator ────────────────────────────────────────────────────

    /// Single-rank stub communicator for unit tests.
    struct StubComm {
        rank: usize,
        size: usize,
    }

    impl Communicator for StubComm {
        fn allgatherv<T: CommData>(
            &self,
            _send: &[T],
            _recv: &mut [T],
            _counts: &[usize],
            _displs: &[usize],
        ) -> Result<(), CommError> {
            unreachable!("StubComm allgatherv not used in simulate tests")
        }

        fn allreduce<T: CommData>(
            &self,
            _send: &[T],
            _recv: &mut [T],
            _op: ReduceOp,
        ) -> Result<(), CommError> {
            unreachable!("StubComm allreduce not used in simulate tests")
        }

        fn broadcast<T: CommData>(&self, _buf: &mut [T], _root: usize) -> Result<(), CommError> {
            unreachable!("StubComm broadcast not used in simulate tests")
        }

        fn barrier(&self) -> Result<(), CommError> {
            Ok(())
        }

        fn rank(&self) -> usize {
            self.rank
        }

        fn size(&self) -> usize {
            self.size
        }
    }

    // ── Mock solver ──────────────────────────────────────────────────────────

    /// Mock solver that returns a configurable fixed `LpSolution` on every solve.
    ///
    /// Optionally returns `SolverError::Infeasible` at a specific (0-based) solve
    /// call index (counting across both cold-start and warm-start calls).
    struct MockSolver {
        solution: LpSolution,
        infeasible_at: Option<usize>,
        call_count: usize,
        buf_primal: Vec<f64>,
        buf_dual: Vec<f64>,
        buf_reduced_costs: Vec<f64>,
    }

    impl MockSolver {
        fn always_ok(solution: LpSolution) -> Self {
            let buf_primal = solution.primal.clone();
            let buf_dual = solution.dual.clone();
            let buf_reduced_costs = solution.reduced_costs.clone();
            Self {
                solution,
                infeasible_at: None,
                call_count: 0,
                buf_primal,
                buf_dual,
                buf_reduced_costs,
            }
        }

        fn infeasible_on(solution: LpSolution, n: usize) -> Self {
            let buf_primal = solution.primal.clone();
            let buf_dual = solution.dual.clone();
            let buf_reduced_costs = solution.reduced_costs.clone();
            Self {
                solution,
                infeasible_at: Some(n),
                call_count: 0,
                buf_primal,
                buf_dual,
                buf_reduced_costs,
            }
        }

        fn do_solve(&mut self) -> Result<cobre_solver::SolutionView<'_>, SolverError> {
            let call = self.call_count;
            self.call_count += 1;
            if self.infeasible_at == Some(call) {
                return Err(SolverError::Infeasible);
            }
            self.buf_primal.clone_from(&self.solution.primal);
            self.buf_dual.clone_from(&self.solution.dual);
            self.buf_reduced_costs
                .clone_from(&self.solution.reduced_costs);
            Ok(cobre_solver::SolutionView {
                objective: self.solution.objective,
                primal: &self.buf_primal,
                dual: &self.buf_dual,
                reduced_costs: &self.buf_reduced_costs,
                iterations: self.solution.iterations,
                solve_time_seconds: self.solution.solve_time_seconds,
            })
        }
    }

    impl SolverInterface for MockSolver {
        fn load_model(&mut self, _template: &StageTemplate) {}
        fn add_rows(&mut self, _cuts: &RowBatch) {}
        fn set_row_bounds(&mut self, _indices: &[usize], _lower: &[f64], _upper: &[f64]) {}
        fn set_col_bounds(&mut self, _indices: &[usize], _lower: &[f64], _upper: &[f64]) {}
        fn solve(&mut self) -> Result<cobre_solver::SolutionView<'_>, SolverError> {
            self.do_solve()
        }
        fn reset(&mut self) {}
        fn get_basis(&mut self, _out: &mut Basis) {}
        fn solve_with_basis(
            &mut self,
            _basis: &Basis,
        ) -> Result<cobre_solver::SolutionView<'_>, SolverError> {
            self.do_solve()
        }
        fn statistics(&self) -> SolverStatistics {
            SolverStatistics::default()
        }
        fn name(&self) -> &'static str {
            "Mock"
        }
    }

    // ── Helpers ──────────────────────────────────────────────────────────────

    /// Minimal valid stage template for N=1 hydro, L=0 PAR order.
    ///
    /// Column layout: `[storage (0), storage_in (1), theta (2)]`
    fn minimal_template_1_0() -> StageTemplate {
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

    /// Build a fixed `LpSolution` for the minimal N=1 L=0 template.
    ///
    /// `theta_col=2`, `primal[2]=theta_val`, `objective=objective`.
    fn fixed_solution(objective: f64, theta_val: f64) -> LpSolution {
        let num_cols = 3; // storage(0), storage_in(1), theta(2)
        let mut primal = vec![0.0_f64; num_cols];
        primal[2] = theta_val; // theta col
        LpSolution {
            objective,
            primal,
            dual: vec![0.0_f64; 1],
            reduced_costs: vec![0.0_f64; num_cols],
            iterations: 0,
            solve_time_seconds: 0.0,
        }
    }

    /// Build a minimal `EntityCounts` for 1 hydro, no other entities.
    fn entity_counts_1_hydro() -> EntityCounts {
        EntityCounts {
            hydro_ids: vec![1],
            hydro_productivities: vec![1.0],
            thermal_ids: vec![],
            line_ids: vec![],
            bus_ids: vec![],
            pumping_station_ids: vec![],
            contract_ids: vec![],
            non_controllable_ids: vec![],
        }
    }

    /// Build a minimal stochastic context for 1 hydro, `n_stages` stages.
    fn make_stochastic_context(n_stages: usize) -> StochasticContext {
        use std::collections::BTreeMap;

        use chrono::NaiveDate;
        use cobre_core::entities::hydro::{Hydro, HydroGenerationModel, HydroPenalties};
        use cobre_core::scenario::{
            CorrelationEntity, CorrelationGroup, CorrelationModel, CorrelationProfile, InflowModel,
        };
        use cobre_core::temporal::{
            Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
            StageStateConfig,
        };
        use cobre_core::{Bus, DeficitSegment, EntityId, SystemBuilder};
        use cobre_stochastic::context::build_stochastic_context;

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
        let make_stage = |idx: usize, id: i32| Stage {
            index: idx,
            id,
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
                branching_factor: 3,
                noise_method: NoiseMethod::Saa,
            },
        };
        let stages: Vec<Stage> = (0..n_stages)
            .map(|i| make_stage(i, i32::try_from(i).unwrap()))
            .collect();
        let inflow = |stage_id: i32| InflowModel {
            hydro_id: EntityId(1),
            stage_id,
            mean_m3s: 100.0,
            std_m3s: 30.0,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
        };
        let inflow_models: Vec<InflowModel> = (0..n_stages)
            .map(|i| inflow(i32::try_from(i).unwrap()))
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
        build_stochastic_context(&system, 42).unwrap()
    }

    // ── Helpers ──────────────────────────────────────────────────────────────

    /// Wrap a `MockSolver` in a single-workspace slice for `simulate()` calls.
    ///
    /// All tests use a single workspace (serial execution) so that existing
    /// assertions about scenario ordering and call counts remain valid.
    fn single_workspace(solver: MockSolver) -> Vec<SolverWorkspace<MockSolver>> {
        vec![SolverWorkspace {
            solver,
            patch_buf: PatchBuffer::new(1, 0), // N=1, L=0
            current_state: Vec::with_capacity(1),
            noise_buf: Vec::new(),
            inflow_m3s_buf: Vec::new(),
        }]
    }

    // ── Tests ────────────────────────────────────────────────────────────────

    /// Acceptance criterion: `n_scenarios=4`, single rank → exactly 4 results in
    /// channel and cost buffer has length 4.
    #[test]
    fn simulate_single_rank_4_scenarios_produces_4_results() {
        let n_stages = 2;
        let templates: Vec<StageTemplate> = (0..n_stages).map(|_| minimal_template_1_0()).collect();
        let base_rows: Vec<usize> = vec![0; n_stages];

        let indexer = StageIndexer::new(1, 0); // N=1, L=0; theta=2
        let fcf = FutureCostFunction::new(n_stages, 1, 1, 10, 0);
        let stochastic = make_stochastic_context(n_stages);
        let config = SimulationConfig {
            n_scenarios: 4,
            io_channel_capacity: 16,
        };
        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let initial_state = vec![50.0_f64]; // n_state=1

        let solution = fixed_solution(100.0, 30.0); // objective=100, theta=30
        let solver = MockSolver::always_ok(solution);
        let comm = StubComm { rank: 0, size: 1 };
        let entity_counts = entity_counts_1_hydro();

        let (tx, rx) = mpsc::sync_channel(16);

        let mut workspaces = single_workspace(solver);
        let result = simulate(
            &mut workspaces,
            &templates,
            &base_rows,
            &fcf,
            &stochastic,
            &config,
            &horizon,
            &initial_state,
            &indexer,
            &entity_counts,
            &comm,
            &tx,
            &InflowNonNegativityMethod::None,
            &[],
            0,
            &[],
            &[],
            None,
        );

        assert!(result.is_ok(), "simulate returned error: {result:?}");
        let cost_buffer = result.unwrap();
        assert_eq!(cost_buffer.len(), 4, "cost buffer should have 4 entries");

        // Drain the channel and count results.
        let mut received = 0;
        while rx.try_recv().is_ok() {
            received += 1;
        }
        assert_eq!(received, 4, "channel should have received 4 results");
    }

    /// Acceptance criterion: solver infeasible at scenario 2, stage 1 (0-based)
    /// → `SimulationError::LpInfeasible` with correct `scenario_id` and `stage_id`.
    ///
    /// With 4 scenarios and 2 stages, the solve calls are numbered 0..7 in
    /// scenario-outer, stage-inner order:
    ///   scenario 0: solves 0, 1
    ///   scenario 1: solves 2, 3
    ///   scenario 2: solves 4 (stage 0), 5 (stage 1)  ← infeasible at call 5
    ///   scenario 3: solves 6, 7
    ///
    /// Infeasible at call 5 = `scenario_id=2`, `stage_id=1`.
    #[test]
    fn simulate_infeasible_returns_lp_infeasible_error() {
        let n_stages = 2;
        let templates: Vec<StageTemplate> = (0..n_stages).map(|_| minimal_template_1_0()).collect();
        let base_rows: Vec<usize> = vec![0; n_stages];

        let indexer = StageIndexer::new(1, 0);
        let fcf = FutureCostFunction::new(n_stages, 1, 1, 10, 0);
        let stochastic = make_stochastic_context(n_stages);
        let config = SimulationConfig {
            n_scenarios: 4,
            io_channel_capacity: 16,
        };
        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let initial_state = vec![50.0_f64];

        let solution = fixed_solution(100.0, 30.0);
        // Call 5 = scenario_id=2 (0-indexed), stage=1 (0-indexed)
        let solver = MockSolver::infeasible_on(solution, 5);
        let comm = StubComm { rank: 0, size: 1 };
        let entity_counts = entity_counts_1_hydro();

        let (tx, _rx) = mpsc::sync_channel(16);

        let mut workspaces = single_workspace(solver);
        let result = simulate(
            &mut workspaces,
            &templates,
            &base_rows,
            &fcf,
            &stochastic,
            &config,
            &horizon,
            &initial_state,
            &indexer,
            &entity_counts,
            &comm,
            &tx,
            &InflowNonNegativityMethod::None,
            &[],
            0,
            &[],
            &[],
            None,
        );

        match result {
            Err(SimulationError::LpInfeasible {
                scenario_id,
                stage_id,
                ..
            }) => {
                assert_eq!(scenario_id, 2, "expected scenario_id=2, got {scenario_id}");
                assert_eq!(stage_id, 1, "expected stage_id=1, got {stage_id}");
            }
            other => panic!("expected LpInfeasible, got {other:?}"),
        }
    }

    /// Acceptance criterion (exact ticket spec): solver infeasible at scenario 2, stage 3
    /// with 4 scenarios and 4 stages → `SimulationError::LpInfeasible { scenario_id: 2, stage_id: 3 }`.
    ///
    /// Solve call index for (scenario=2, stage=3) = 2*4 + 3 = 11 (0-based).
    #[test]
    fn simulate_infeasible_at_scenario2_stage3() {
        let n_stages = 4;
        let templates: Vec<StageTemplate> = (0..n_stages).map(|_| minimal_template_1_0()).collect();
        let base_rows: Vec<usize> = vec![0; n_stages];

        let indexer = StageIndexer::new(1, 0);
        let fcf = FutureCostFunction::new(n_stages, 1, 1, 10, 0);
        let stochastic = make_stochastic_context(n_stages);
        let config = SimulationConfig {
            n_scenarios: 4,
            io_channel_capacity: 16,
        };
        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let initial_state = vec![50.0_f64];

        let solution = fixed_solution(100.0, 30.0);
        // Call 11 = scenario 2 (0-based), stage 3 (0-based): 2*4 + 3 = 11.
        let solver = MockSolver::infeasible_on(solution, 11);
        let comm = StubComm { rank: 0, size: 1 };
        let entity_counts = entity_counts_1_hydro();

        let (tx, _rx) = mpsc::sync_channel(16);

        let mut workspaces = single_workspace(solver);
        let result = simulate(
            &mut workspaces,
            &templates,
            &base_rows,
            &fcf,
            &stochastic,
            &config,
            &horizon,
            &initial_state,
            &indexer,
            &entity_counts,
            &comm,
            &tx,
            &InflowNonNegativityMethod::None,
            &[],
            0,
            &[],
            &[],
            None,
        );

        match result {
            Err(SimulationError::LpInfeasible {
                scenario_id,
                stage_id,
                ..
            }) => {
                assert_eq!(scenario_id, 2, "expected scenario_id=2, got {scenario_id}");
                assert_eq!(stage_id, 3, "expected stage_id=3, got {stage_id}");
            }
            other => panic!("expected LpInfeasible, got {other:?}"),
        }
    }

    /// Acceptance criterion: drop receiver before calling simulate → `ChannelClosed`.
    #[test]
    fn simulate_channel_closed_returns_error() {
        let n_stages = 2;
        let templates: Vec<StageTemplate> = (0..n_stages).map(|_| minimal_template_1_0()).collect();
        let base_rows: Vec<usize> = vec![0; n_stages];

        let indexer = StageIndexer::new(1, 0);
        let fcf = FutureCostFunction::new(n_stages, 1, 1, 10, 0);
        let stochastic = make_stochastic_context(n_stages);
        let config = SimulationConfig {
            n_scenarios: 2,
            io_channel_capacity: 1,
        };
        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let initial_state = vec![50.0_f64];

        let solution = fixed_solution(100.0, 30.0);
        let solver = MockSolver::always_ok(solution);
        let comm = StubComm { rank: 0, size: 1 };
        let entity_counts = entity_counts_1_hydro();

        let (tx, rx) = mpsc::sync_channel(1);
        // Drop the receiver immediately so send() will fail.
        drop(rx);

        let mut workspaces = single_workspace(solver);
        let result = simulate(
            &mut workspaces,
            &templates,
            &base_rows,
            &fcf,
            &stochastic,
            &config,
            &horizon,
            &initial_state,
            &indexer,
            &entity_counts,
            &comm,
            &tx,
            &InflowNonNegativityMethod::None,
            &[],
            0,
            &[],
            &[],
            None,
        );

        assert!(
            matches!(result, Err(SimulationError::ChannelClosed)),
            "expected ChannelClosed, got {result:?}"
        );
    }

    /// Acceptance criterion: `total_cost` in cost buffer equals sum of
    /// `(objective - primal[theta])` across all stages for each scenario.
    ///
    /// With objective=100.0 and theta=30.0: `stage_cost` = 100 - 30 = 70 per stage.
    /// For 3 stages: `total_cost` = 3 * 70 = 210.
    #[test]
    fn simulate_total_cost_equals_sum_of_stage_costs() {
        let n_stages = 3;
        let templates: Vec<StageTemplate> = (0..n_stages).map(|_| minimal_template_1_0()).collect();
        let base_rows: Vec<usize> = vec![0; n_stages];

        let indexer = StageIndexer::new(1, 0); // theta=2
        let fcf = FutureCostFunction::new(n_stages, 1, 1, 10, 0);
        let stochastic = make_stochastic_context(n_stages);
        let config = SimulationConfig {
            n_scenarios: 2,
            io_channel_capacity: 16,
        };
        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let initial_state = vec![50.0_f64];

        let objective = 100.0_f64;
        let theta_val = 30.0_f64;
        let expected_stage_cost = objective - theta_val; // 70.0
        #[allow(clippy::cast_precision_loss)]
        let expected_total_cost = expected_stage_cost * n_stages as f64; // 210.0

        let solution = fixed_solution(objective, theta_val);
        let solver = MockSolver::always_ok(solution);
        let comm = StubComm { rank: 0, size: 1 };
        let entity_counts = entity_counts_1_hydro();

        let (tx, _rx) = mpsc::sync_channel(16);

        let mut workspaces = single_workspace(solver);
        let cost_buffer = simulate(
            &mut workspaces,
            &templates,
            &base_rows,
            &fcf,
            &stochastic,
            &config,
            &horizon,
            &initial_state,
            &indexer,
            &entity_counts,
            &comm,
            &tx,
            &InflowNonNegativityMethod::None,
            &[],
            0,
            &[],
            &[],
            None,
        )
        .unwrap();

        assert_eq!(cost_buffer.len(), 2);
        for (scenario_id, total_cost, _) in &cost_buffer {
            assert!(
                (total_cost - expected_total_cost).abs() < 1e-9,
                "scenario {scenario_id}: expected total_cost={expected_total_cost}, got {total_cost}"
            );
        }
    }

    /// Verify that the `scenario_ids` in the cost buffer match the assigned range.
    ///
    /// With `n_scenarios=6`, `world_size=2`, rank=0: `assign_scenarios(6, 0, 2) = 0..3`.
    /// The cost buffer must contain `scenario_ids` 0, 1, 2 in that order.
    #[test]
    fn simulate_cost_buffer_scenario_ids_match_assigned_range() {
        let n_stages = 1;
        let templates: Vec<StageTemplate> = (0..n_stages).map(|_| minimal_template_1_0()).collect();
        let base_rows: Vec<usize> = vec![0; n_stages];

        let indexer = StageIndexer::new(1, 0);
        let fcf = FutureCostFunction::new(n_stages, 1, 1, 10, 0);
        let stochastic = make_stochastic_context(n_stages);
        let config = SimulationConfig {
            n_scenarios: 6,
            io_channel_capacity: 16,
        };
        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let initial_state = vec![50.0_f64];

        let solution = fixed_solution(50.0, 10.0);
        let solver = MockSolver::always_ok(solution);
        // rank=0 of 2: assign_scenarios(6, 0, 2) = 0..3
        let comm = StubComm { rank: 0, size: 2 };
        let entity_counts = entity_counts_1_hydro();

        let (tx, _rx) = mpsc::sync_channel(16);

        let mut workspaces = single_workspace(solver);
        let cost_buffer = simulate(
            &mut workspaces,
            &templates,
            &base_rows,
            &fcf,
            &stochastic,
            &config,
            &horizon,
            &initial_state,
            &indexer,
            &entity_counts,
            &comm,
            &tx,
            &InflowNonNegativityMethod::None,
            &[],
            0,
            &[],
            &[],
            None,
        )
        .unwrap();

        assert_eq!(cost_buffer.len(), 3, "rank 0 should process 3 scenarios");
        let ids: Vec<u32> = cost_buffer.iter().map(|(id, _, _)| *id).collect();
        assert_eq!(
            ids,
            vec![0, 1, 2],
            "scenario IDs must match assigned range 0..3"
        );
    }

    /// Verify channel receives results in scenario order for single rank.
    #[test]
    fn simulate_channel_receives_results_in_scenario_order() {
        let n_stages = 1;
        let templates: Vec<StageTemplate> = (0..n_stages).map(|_| minimal_template_1_0()).collect();
        let base_rows: Vec<usize> = vec![0; n_stages];

        let indexer = StageIndexer::new(1, 0);
        let fcf = FutureCostFunction::new(n_stages, 1, 1, 10, 0);
        let stochastic = make_stochastic_context(n_stages);
        let config = SimulationConfig {
            n_scenarios: 3,
            io_channel_capacity: 16,
        };
        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let initial_state = vec![50.0_f64];

        let solution = fixed_solution(100.0, 20.0);
        let solver = MockSolver::always_ok(solution);
        let comm = StubComm { rank: 0, size: 1 };
        let entity_counts = entity_counts_1_hydro();

        let (tx, rx) = mpsc::sync_channel(16);

        let mut workspaces = single_workspace(solver);
        simulate(
            &mut workspaces,
            &templates,
            &base_rows,
            &fcf,
            &stochastic,
            &config,
            &horizon,
            &initial_state,
            &indexer,
            &entity_counts,
            &comm,
            &tx,
            &InflowNonNegativityMethod::None,
            &[],
            0,
            &[],
            &[],
            None,
        )
        .unwrap();

        let received: Vec<u32> = (0..3).map(|_| rx.recv().unwrap().scenario_id).collect();
        assert_eq!(received, vec![0, 1, 2]);
    }

    /// New acceptance criterion: cost buffer from 1 workspace equals cost buffer from 4 workspaces.
    ///
    /// Both runs must produce identical `(scenario_id, total_cost, category_costs)` tuples for all
    /// 20 scenarios. The cost buffer must be sorted by `scenario_id` in both cases.
    #[test]
    fn test_simulation_parallel_cost_determinism() {
        let n_stages = 2;
        let n_scenarios = 20u32;
        let templates: Vec<StageTemplate> = (0..n_stages).map(|_| minimal_template_1_0()).collect();
        let base_rows: Vec<usize> = vec![0; n_stages];

        let indexer = StageIndexer::new(1, 0);
        let fcf = FutureCostFunction::new(n_stages, 1, 1, 10, 0);
        let stochastic = make_stochastic_context(n_stages);
        let config = SimulationConfig {
            n_scenarios,
            io_channel_capacity: 64,
        };
        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let initial_state = vec![50.0_f64];

        let objective = 100.0_f64;
        let theta_val = 30.0_f64;
        let solution = fixed_solution(objective, theta_val);
        let comm = StubComm { rank: 0, size: 1 };
        let entity_counts = entity_counts_1_hydro();

        // Run with 1 workspace.
        let (tx1, _rx1) = mpsc::sync_channel(64);
        let mut workspaces_1 = single_workspace(MockSolver::always_ok(solution.clone()));
        let costs_1 = simulate(
            &mut workspaces_1,
            &templates,
            &base_rows,
            &fcf,
            &stochastic,
            &config,
            &horizon,
            &initial_state,
            &indexer,
            &entity_counts,
            &comm,
            &tx1,
            &InflowNonNegativityMethod::None,
            &[],
            0,
            &[],
            &[],
            None,
        )
        .unwrap();

        // Run with 4 workspaces.
        let (tx4, _rx4) = mpsc::sync_channel(64);
        let mut workspaces_4: Vec<SolverWorkspace<MockSolver>> = (0..4)
            .map(|_| SolverWorkspace {
                solver: MockSolver::always_ok(solution.clone()),
                patch_buf: PatchBuffer::new(1, 0),
                current_state: Vec::with_capacity(1),
                noise_buf: Vec::new(),
                inflow_m3s_buf: Vec::new(),
            })
            .collect();
        let costs_4 = simulate(
            &mut workspaces_4,
            &templates,
            &base_rows,
            &fcf,
            &stochastic,
            &config,
            &horizon,
            &initial_state,
            &indexer,
            &entity_counts,
            &comm,
            &tx4,
            &InflowNonNegativityMethod::None,
            &[],
            0,
            &[],
            &[],
            None,
        )
        .unwrap();

        // Both cost buffers must have exactly 20 entries sorted by scenario_id.
        assert_eq!(
            costs_1.len(),
            n_scenarios as usize,
            "1-workspace: 20 entries"
        );
        assert_eq!(
            costs_4.len(),
            n_scenarios as usize,
            "4-workspace: 20 entries"
        );

        let ids_1: Vec<u32> = costs_1.iter().map(|(id, _, _)| *id).collect();
        let ids_4: Vec<u32> = costs_4.iter().map(|(id, _, _)| *id).collect();
        let expected_ids: Vec<u32> = (0..n_scenarios).collect();
        assert_eq!(ids_1, expected_ids, "1-workspace: sorted scenario IDs");
        assert_eq!(ids_4, expected_ids, "4-workspace: sorted scenario IDs");

        // Cost values must be identical.
        for i in 0..n_scenarios as usize {
            let (id1, cost1, _) = &costs_1[i];
            let (id4, cost4, _) = &costs_4[i];
            assert_eq!(id1, id4, "scenario_id mismatch at index {i}");
            assert!(
                (cost1 - cost4).abs() < 1e-9,
                "cost mismatch for scenario {id1}: 1-ws={cost1}, 4-ws={cost4}"
            );
        }
    }

    // ── Welford accumulator unit tests ───────────────────────────────────────

    /// Known dataset: `[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]`
    /// Expected: mean=5.0, variance=4.0, `std_dev`=2.0.
    #[test]
    fn welford_known_dataset_mean_variance_std() {
        let values = [2.0_f64, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let mut acc = super::WelfordAccumulator::new();
        for &v in &values {
            acc.update(v);
        }
        assert!(
            (acc.mean() - 5.0).abs() < 1e-10,
            "mean: expected 5.0, got {}",
            acc.mean()
        );
        assert!(
            (acc.variance() - 4.0).abs() < 1e-10,
            "variance: expected 4.0, got {}",
            acc.variance()
        );
        assert!(
            (acc.std_dev() - 2.0).abs() < 1e-10,
            "std_dev: expected 2.0, got {}",
            acc.std_dev()
        );
    }

    /// Single value: mean equals that value, `std_dev`=0.0, CI half-width=0.0.
    #[test]
    fn welford_single_value_no_variance() {
        let mut acc = super::WelfordAccumulator::new();
        acc.update(42.0);
        assert!(
            (acc.mean() - 42.0).abs() < 1e-10,
            "mean: expected 42.0, got {}",
            acc.mean()
        );
        assert_eq!(
            acc.std_dev(),
            0.0,
            "std_dev must be 0.0 with one observation"
        );
        assert_eq!(
            acc.ci_95_half_width(),
            0.0,
            "ci_95_half_width must be 0.0 with one observation"
        );
    }

    /// Zero updates: mean=0.0, `std_dev`=0.0.
    #[test]
    fn welford_zero_updates() {
        let acc = super::WelfordAccumulator::new();
        assert_eq!(acc.mean(), 0.0, "mean must be 0.0 with no observations");
        assert_eq!(
            acc.std_dev(),
            0.0,
            "std_dev must be 0.0 with no observations"
        );
    }

    // ── Integration tests for event emission ─────────────────────────────────

    /// Acceptance criterion: with `event_sender: Some(&tx)` and 10 scenarios,
    /// at least 1 `SimulationProgress` event is received with `scenarios_complete > 0`,
    /// finite non-NaN `mean_cost`, and `ci_95_half_width >= 0.0`.
    #[test]
    fn simulate_emits_progress_events() {
        use cobre_core::TrainingEvent;

        let n_stages = 2;
        let templates: Vec<StageTemplate> = (0..n_stages).map(|_| minimal_template_1_0()).collect();
        let base_rows: Vec<usize> = vec![0; n_stages];

        let indexer = StageIndexer::new(1, 0);
        let fcf = FutureCostFunction::new(n_stages, 1, 1, 10, 0);
        let stochastic = make_stochastic_context(n_stages);
        let config = SimulationConfig {
            n_scenarios: 10,
            io_channel_capacity: 32,
        };
        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let initial_state = vec![50.0_f64];

        let solution = fixed_solution(100.0, 30.0);
        let solver = MockSolver::always_ok(solution);
        let comm = StubComm { rank: 0, size: 1 };
        let entity_counts = entity_counts_1_hydro();

        let (result_tx, _result_rx) = mpsc::sync_channel(32);
        let (event_tx, event_rx) = mpsc::channel::<TrainingEvent>();

        let mut workspaces = single_workspace(solver);
        let result = simulate(
            &mut workspaces,
            &templates,
            &base_rows,
            &fcf,
            &stochastic,
            &config,
            &horizon,
            &initial_state,
            &indexer,
            &entity_counts,
            &comm,
            &result_tx,
            &InflowNonNegativityMethod::None,
            &[],
            0,
            &[],
            &[],
            Some(&event_tx),
        );
        assert!(result.is_ok(), "simulate returned error: {result:?}");

        // Drop the sender so the channel drains cleanly.
        drop(event_tx);

        let events: Vec<TrainingEvent> = event_rx.iter().collect();

        let progress_events: Vec<_> = events
            .iter()
            .filter(|e| matches!(e, TrainingEvent::SimulationProgress { .. }))
            .collect();

        assert!(
            !progress_events.is_empty(),
            "at least 1 SimulationProgress event expected"
        );

        for event in &progress_events {
            let TrainingEvent::SimulationProgress {
                scenarios_complete,
                mean_cost,
                ci_95_half_width,
                ..
            } = event
            else {
                continue;
            };

            assert!(
                *scenarios_complete > 0,
                "scenarios_complete must be > 0, got {scenarios_complete}"
            );
            assert!(
                mean_cost.is_finite() && !mean_cost.is_nan(),
                "mean_cost must be finite and non-NaN, got {mean_cost}"
            );
            assert!(
                *ci_95_half_width >= 0.0,
                "ci_95_half_width must be >= 0.0, got {ci_95_half_width}"
            );
        }
    }

    /// Acceptance criterion: with `event_sender: None`, no events are sent and
    /// the function returns the same cost buffer as before.
    #[test]
    fn simulate_no_events_when_sender_is_none() {
        let n_stages = 2;
        let templates: Vec<StageTemplate> = (0..n_stages).map(|_| minimal_template_1_0()).collect();
        let base_rows: Vec<usize> = vec![0; n_stages];

        let indexer = StageIndexer::new(1, 0);
        let fcf = FutureCostFunction::new(n_stages, 1, 1, 10, 0);
        let stochastic = make_stochastic_context(n_stages);
        let config = SimulationConfig {
            n_scenarios: 4,
            io_channel_capacity: 16,
        };
        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let initial_state = vec![50.0_f64];

        let solution = fixed_solution(100.0, 30.0);
        let solver = MockSolver::always_ok(solution);
        let comm = StubComm { rank: 0, size: 1 };
        let entity_counts = entity_counts_1_hydro();

        let (result_tx, _result_rx) = mpsc::sync_channel(16);

        let mut workspaces = single_workspace(solver);
        let result = simulate(
            &mut workspaces,
            &templates,
            &base_rows,
            &fcf,
            &stochastic,
            &config,
            &horizon,
            &initial_state,
            &indexer,
            &entity_counts,
            &comm,
            &result_tx,
            &InflowNonNegativityMethod::None,
            &[],
            0,
            &[],
            &[],
            None,
        );

        assert!(result.is_ok(), "simulate returned error: {result:?}");
        let cost_buffer = result.unwrap();
        assert_eq!(
            cost_buffer.len(),
            4,
            "cost buffer must have 4 entries when event_sender is None"
        );
    }
}
