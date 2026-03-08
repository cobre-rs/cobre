//! Simulation forward pass loop for SDDP policy evaluation.
//!
//! [`simulate`] evaluates the trained SDDP policy on a set of scenarios by
//! running a forward-only pass through all stages, extracting per-entity
//! results at each stage, streaming completed scenario results through a
//! bounded channel, and returning a compact cost buffer for MPI aggregation
//! (ticket-008).
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
//! independently; MPI aggregation is deferred to ticket-008.
//!
//! ## Seed domain separation
//!
//! To avoid seed collisions with training forward pass seeds (which use
//! `global_scenario = rank * forward_passes + m`), the simulation domain adds
//! an offset of `u32::MAX / 2` to the scenario ID before passing it to
//! [`sample_forward`]. This places simulation seeds in a disjoint region of
//! the SipHash-1-3 seed space (DEC-017).
//!
//! ## Hot-path allocation discipline
//!
//! No allocations occur per scenario or per stage during the inner loops.
//! The [`PatchBuffer`], `current_state`, and `basis_cache` are pre-allocated
//! before the scenario loop. The [`RowBatch`] per stage is built once before
//! the scenario loop — not once per scenario.

use std::sync::mpsc::SyncSender;

use cobre_comm::Communicator;
use cobre_solver::{Basis, RowBatch, SolverError, SolverInterface, StageTemplate};
use cobre_stochastic::{StochasticContext, sample_forward};

use crate::{
    FutureCostFunction, HorizonMode, PatchBuffer, StageIndexer,
    forward::build_cut_row_batch,
    simulation::{
        config::SimulationConfig,
        error::SimulationError,
        extraction::EntityCounts,
        extraction::{accumulate_category_costs, assign_scenarios, extract_stage_result},
        types::{ScenarioCategoryCosts, SimulationScenarioResult},
    },
};

/// Offset added to the simulation scenario ID before passing to [`sample_forward`].
///
/// Separates the simulation seed domain from the training forward pass domain.
/// Training uses `global_scenario = rank * forward_passes + m`, while
/// simulation uses `global_scenario = SIMULATION_SEED_OFFSET + scenario_id`.
/// Both fit in `u32`; the offset guarantees no overlap for practical scenario counts.
const SIMULATION_SEED_OFFSET: u32 = u32::MAX / 2;

/// Evaluate the trained SDDP policy on this rank's assigned scenarios.
///
/// Iterates over all scenarios assigned to this rank via [`assign_scenarios`].
/// For each scenario, solves the LP at every stage of the horizon using the
/// trained cut pools from `fcf`, extracts per-entity results via
/// [`extract_stage_result`], accumulates per-category costs via
/// [`accumulate_category_costs`], and sends the completed
/// [`SimulationScenarioResult`] through `result_tx`.
///
/// Returns a compact cost buffer — one `(scenario_id, total_cost, category_costs)`
/// entry per locally solved scenario — for MPI aggregation in ticket-008.
///
/// ## Pre-allocation
///
/// All workspace buffers ([`PatchBuffer`], `current_state`, `basis_cache`) are
/// allocated once before the scenario loop. No heap allocation occurs on the
/// hot path.
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
pub fn simulate<S: SolverInterface, C: Communicator>(
    solver: &mut S,
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

    // Pre-allocate the output cost buffer.
    let mut cost_buffer: Vec<(u32, f64, ScenarioCategoryCosts)> = Vec::with_capacity(local_count);

    // Pre-allocate workspace buffers — reused across all scenarios and stages.
    let mut patch_buf = PatchBuffer::new(indexer.hydro_count, indexer.max_par_order);
    let mut current_state: Vec<f64> = Vec::with_capacity(indexer.n_state);

    // Per-stage basis cache: reused for warm-starting across scenarios.
    // Initialized to None; populated after the first solve at each stage.
    let mut basis_cache: Vec<Option<Basis>> = vec![None; num_stages];

    // Outer loop: one iteration per locally assigned scenario.
    for scenario_id in scenario_range {
        // Simulation seed domain separation from training (DEC-017):
        // Use SIMULATION_SEED_OFFSET + scenario_id to place simulation seeds
        // in a disjoint region of the SipHash-1-3 seed space.
        // Iteration is fixed at 0 for simulation (one-shot evaluation).
        let global_scenario = SIMULATION_SEED_OFFSET.saturating_add(scenario_id);

        // Initialize current state for this scenario.
        current_state.clear();
        current_state.extend_from_slice(initial_state);

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
            // Cast indices to u32 for the sampling API (DEC-017).
            // Bounded by u32::MAX in practice; truncation is safe.
            #[allow(clippy::cast_possible_truncation)]
            let stage_id_u32 = t as u32;

            let (_opening_idx, noise) =
                sample_forward(&tree_view, base_seed, 0, global_scenario, stage_id_u32, t);

            // LP rebuild sequence: template → cuts → scenario-specific row bounds.
            solver.load_model(&templates[t]);
            solver.add_rows(&cut_batches[t]);

            patch_buf.fill_forward_patches(indexer, &current_state, noise, base_rows[t]);
            let patch_count = patch_buf.forward_patch_count();
            solver.set_row_bounds(
                &patch_buf.indices[..patch_count],
                &patch_buf.lower[..patch_count],
                &patch_buf.upper[..patch_count],
            );

            let view = (if let Some(rb) = basis_cache[t].as_ref() {
                solver.solve_with_basis(rb)
            } else {
                solver.solve()
            })
            .map_err(|e| {
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

            // Extract per-entity typed result for this stage.
            let stage_result = extract_stage_result(
                view.primal,
                view.dual,
                view.objective,
                indexer,
                stage_id_u32,
                entity_counts,
            );

            // Accumulate per-category costs for this stage.
            for cost_entry in &stage_result.costs {
                accumulate_category_costs(cost_entry, &mut category_costs);
            }

            stage_results.push(stage_result);

            // Advance state to the outgoing storage + lags from this stage.
            current_state.clear();
            current_state.extend_from_slice(&view.primal[..indexer.n_state]);

            // Update basis cache for warm-starting the next scenario.
            if let Some(rb) = &mut basis_cache[t] {
                solver.get_basis(rb);
            } else {
                let mut rb = Basis::new(templates[t].num_cols, templates[t].num_rows);
                solver.get_basis(&mut rb);
                basis_cache[t] = Some(rb);
            }
        }

        // Build the scenario result and send through the bounded channel.
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

        // Retain the compact (scenario_id, total_cost, category_costs) for MPI
        // aggregation in ticket-008 before consuming `scenario_result`.
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

        cost_buffer.push((scenario_id, total_cost, compact_category));
    }

    Ok(cost_buffer)
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
        FutureCostFunction, HorizonMode, StageIndexer,
        simulation::{config::SimulationConfig, error::SimulationError, extraction::EntityCounts},
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
        let mut solver = MockSolver::always_ok(solution);
        let comm = StubComm { rank: 0, size: 1 };
        let entity_counts = entity_counts_1_hydro();

        let (tx, rx) = mpsc::sync_channel(16);

        let result = simulate(
            &mut solver,
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
        let mut solver = MockSolver::infeasible_on(solution, 5);
        let comm = StubComm { rank: 0, size: 1 };
        let entity_counts = entity_counts_1_hydro();

        let (tx, _rx) = mpsc::sync_channel(16);

        let result = simulate(
            &mut solver,
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
        let mut solver = MockSolver::infeasible_on(solution, 11);
        let comm = StubComm { rank: 0, size: 1 };
        let entity_counts = entity_counts_1_hydro();

        let (tx, _rx) = mpsc::sync_channel(16);

        let result = simulate(
            &mut solver,
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
        let mut solver = MockSolver::always_ok(solution);
        let comm = StubComm { rank: 0, size: 1 };
        let entity_counts = entity_counts_1_hydro();

        let (tx, rx) = mpsc::sync_channel(1);
        // Drop the receiver immediately so send() will fail.
        drop(rx);

        let result = simulate(
            &mut solver,
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
        let mut solver = MockSolver::always_ok(solution);
        let comm = StubComm { rank: 0, size: 1 };
        let entity_counts = entity_counts_1_hydro();

        let (tx, _rx) = mpsc::sync_channel(16);

        let cost_buffer = simulate(
            &mut solver,
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
        let mut solver = MockSolver::always_ok(solution);
        // rank=0 of 2: assign_scenarios(6, 0, 2) = 0..3
        let comm = StubComm { rank: 0, size: 2 };
        let entity_counts = entity_counts_1_hydro();

        let (tx, _rx) = mpsc::sync_channel(16);

        let cost_buffer = simulate(
            &mut solver,
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
        let mut solver = MockSolver::always_ok(solution);
        let comm = StubComm { rank: 0, size: 1 };
        let entity_counts = entity_counts_1_hydro();

        let (tx, rx) = mpsc::sync_channel(16);

        simulate(
            &mut solver,
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
        )
        .unwrap();

        let received: Vec<u32> = (0..3).map(|_| rx.recv().unwrap().scenario_id).collect();
        assert_eq!(received, vec![0, 1, 2]);
    }
}
