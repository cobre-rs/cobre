//! Integration tests for [`cobre_sddp::simulate`] (simulation pipeline).
//!
//! Uses a [`MockSolver`] and [`StubComm`] to exercise the simulation pipeline
//! end-to-end without a real LP solver or MPI communicator. Covers scenario
//! count, error propagation, cost accumulation, event emission, load patching,
//! inflow truncation, frozen-template acceptance, and warm-start basis handling.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::too_many_lines,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::float_cmp,
    clippy::cast_precision_loss
)]
// `..Default::default()` in the make_* Spec calls is the intentional future-field
// seam from `common::builders` — a no-op today, not dead code.
#![allow(clippy::needless_update)]

use std::collections::HashMap;
use std::sync::mpsc;

use cobre_comm::{CommData, CommError, Communicator, ReduceOp};
use cobre_core::scenario::SamplingScheme;
use cobre_solver::{
    Basis, BasisStatus, LpSolution, RowBatch, SolverError, SolverInterface, SolverStatistics,
    StageTemplate,
};
use cobre_stochastic::{StochasticContext, select_transition_child};

use cobre_sddp::{
    CapturedBasis, EnergyConversionSet, Phase, SimulationError,
    context::TrainingContext,
    cut::FutureCostFunction,
    horizon_mode::HorizonMode,
    indexer::{StateSpace, StudyDimensions},
    inflow_method::InflowNonNegativityMethod,
    lead_time::AnticipatedResolution,
    lp::builder::PatchBuffer,
    setup::node_graph::{
        NodeGraph, NodeId, NodeOpenings, NodePos, NodeRuntime, NodeSuccessor, OpeningSource,
        StageIdx, Traversal,
    },
    simulation::{EntityCounts, SimulationConfig, SimulationOutputSpec},
    test_support::{
        StageContextFixture, all_enabled_cut_state_layouts, hydro_only_bus_geometry,
        hydro_only_bus_solution, hydro_only_bus_template, permissive_state_boxes,
    },
    workspace::{SolverWorkspace, WorkspaceSizing},
};

mod common;
use common::builders::{BusSpec, HydroSpec, StageSpec, make_bus, make_hydro, make_stage};

// ── Stub communicator ────────────────────────────────────────────────────────

/// Mirrors the gated `test_support::state_layout_for` via the public
/// [`StateSpace::new`] constructor: this external test crate cannot see the
/// parent crate's `#[cfg(test)]` surface, so it rebuilds byte-identical patch
/// columns on the default feature set.
fn state_layout_for(hydro_count: usize, max_par_order: usize) -> StateSpace {
    StateSpace::new(
        hydro_count,
        max_par_order,
        Vec::new(),
        vec![],
        AnticipatedResolution::default(),
        &vec![max_par_order; hydro_count],
    )
}

fn study_dims() -> StudyDimensions {
    StudyDimensions::default()
}

/// Single-rank stub communicator for pipeline tests.
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

    fn abort(&self, error_code: i32) -> ! {
        std::process::exit(error_code)
    }
}

// ── Mock solver ──────────────────────────────────────────────────────────

/// Mock solver returning a configurable fixed `LpSolution` on every solve, with
/// optional `SolverError::Infeasible` at a given solve index. The injection
/// index counts `call_count` (cold-start + warm-start combined, 0-based); the
/// split `solve_count` / `solve_with_basis_count` distinguish the two paths.
struct MockSolver {
    solution: LpSolution,
    infeasible_at: Option<usize>,
    call_count: usize,
    buf_primal: Vec<f64>,
    buf_dual: Vec<f64>,
    buf_reduced_costs: Vec<f64>,
    load_count: usize,
    add_rows_count: usize,
    solve_count: usize,
    solve_with_basis_count: usize,
    recorded_basis: Option<Basis>,
}

impl MockSolver {
    fn new(solution: LpSolution, infeasible_at: Option<usize>) -> Self {
        let buf_primal = solution.primal.clone();
        let buf_dual = solution.dual.clone();
        let buf_reduced_costs = solution.reduced_costs.clone();
        Self {
            solution,
            infeasible_at,
            call_count: 0,
            buf_primal,
            buf_dual,
            buf_reduced_costs,
            load_count: 0,
            add_rows_count: 0,
            solve_count: 0,
            solve_with_basis_count: 0,
            recorded_basis: None,
        }
    }

    fn always_ok(solution: LpSolution) -> Self {
        Self::new(solution, None)
    }

    fn infeasible_on(solution: LpSolution, n: usize) -> Self {
        Self::new(solution, Some(n))
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
    type Profile = cobre_solver::ActiveProfile;

    fn apply_profile(&mut self, _profile: &cobre_solver::ActiveProfile) {}
    fn solver_name_version(&self) -> String {
        "MockSolver 0.0.0".to_string()
    }
    fn load_model(&mut self, _template: &StageTemplate) {
        self.load_count += 1;
    }
    fn add_rows(&mut self, _cuts: &RowBatch) {
        self.add_rows_count += 1;
    }
    fn set_row_bounds(&mut self, _indices: &[usize], _lower: &[f64], _upper: &[f64]) {}
    fn set_col_bounds(&mut self, _indices: &[usize], _lower: &[f64], _upper: &[f64]) {}
    fn solve(
        &mut self,
        basis: Option<&Basis>,
    ) -> Result<cobre_solver::SolutionView<'_>, SolverError> {
        if let Some(b) = basis {
            self.solve_with_basis_count += 1;
            self.recorded_basis = Some(b.clone());
        } else {
            self.solve_count += 1;
        }
        self.do_solve()
    }
    fn get_basis(&mut self, out: &mut Basis) {
        cobre_sddp::test_support::fill_consistent_basis(out);
    }
    fn record_reconstruction_stats(&mut self) {}
    fn statistics(&self) -> SolverStatistics {
        SolverStatistics::default()
    }

    fn statistics_into(&self, out: &mut SolverStatistics) {
        *out = self.statistics();
    }

    fn name(&self) -> &'static str {
        "Mock"
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────

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
    use cobre_core::entities::hydro::{HydroGenerationModel, HydroPenalties};
    use cobre_core::scenario::{
        CorrelationEntity, CorrelationGroup, CorrelationModel, CorrelationProfile, InflowModel,
    };
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    };
    use cobre_core::{DeficitSegment, EntityId, SystemBuilder};
    use cobre_stochastic::context::{ClassSchemes, OpeningTreeInputs, build_stochastic_context};

    let bus = make_bus(
        EntityId(0),
        BusSpec {
            name: "B0".to_string(),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 1000.0,
            }],
            excess_cost: 0.0,
            ..Default::default()
        },
    );
    let hydro = make_hydro(
        EntityId(1),
        HydroSpec {
            name: "H1".to_string(),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId(0),
            downstream_id: None,
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
            ..Default::default()
        },
    );
    let stages: Vec<Stage> = (0..n_stages)
        .map(|i| {
            make_stage(
                i,
                StageSpec {
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
                    ..Default::default()
                },
            )
        })
        .collect();
    let inflow = |stage_id: i32| InflowModel {
        hydro_id: EntityId(1),
        stage_id,
        mean_m3s: 100.0,
        std_m3s: 30.0,
        ar_coefficients: vec![],
        residual_std_ratio: 1.0,
        annual: None,
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

// ── Helpers ──────────────────────────────────────────────────────────────

/// Per-stage hydro productivities matching `entity_counts_1_hydro` (one hydro, 1.0).
fn hydro_productivities_1hydro(n_stages: usize) -> Vec<Vec<f64>> {
    vec![vec![1.0]; n_stages]
}

/// Build a zero-valued [`EnergyConversionSet`] for tests
/// that do not assert on energy fields.
fn zero_energy_conversion(n_hydros: usize, n_stages: usize) -> EnergyConversionSet {
    use cobre_sddp::energy_conversion::EnergyConversion;
    let zero_ec = EnergyConversion {
        equivalent_productivity_mw_per_m3s: 0.0,
        reference_volume_hm3: 0.0,
        reference_outflow_m3s: 0.0,
    };
    EnergyConversionSet::new(
        vec![vec![zero_ec; n_stages]; n_hydros],
        vec![vec![0.0_f64; n_stages]; n_hydros],
        &cobre_sddp::test_support::minimal_hydros(n_hydros),
        n_stages,
    )
}

/// Wrap a `MockSolver` in a single-workspace slice for `simulate()` calls.
///
/// All tests use a single workspace (serial execution) so that existing
/// assertions about scenario ordering and call counts remain valid.
fn single_workspace(solver: MockSolver) -> Vec<SolverWorkspace<MockSolver>> {
    let state = state_layout_for(1, 0);
    let stochastic = cobre_sddp::test_support::hydro_free_stochastic_context(1, 1);
    let node_graph = cobre_sddp::test_support::chain_node_graph(&stochastic);
    let sd = study_dims();
    let horizon = HorizonMode::Finite { num_stages: 1 };
    let cut_state_layouts = all_enabled_cut_state_layouts(&state, 1);
    let initial_state: Vec<f64> = Vec::new();
    let training_ctx = TrainingContext {
        node_graph: &node_graph,
        horizon: &horizon,
        state: &state,
        cut_state_layouts: &cut_state_layouts,
        study_dims: &sd,
        inflow_method: &InflowNonNegativityMethod::None,
        stochastic: &stochastic,
        initial_state: &initial_state,
        inflow_scheme: SamplingScheme::InSample,
        load_scheme: SamplingScheme::InSample,
        ncs_scheme: SamplingScheme::InSample,
        stages: &[],
        historical_library: None,
        external_inflow_library: None,
        external_load_library: None,
        external_ncs_library: None,
        lag_accum_seed: &[],
        lag_weight_seed: &[],
        dcs: None,
    };
    vec![SolverWorkspace::new(
        0,
        0,
        solver,
        PatchBuffer::new(&state, &[], &[]),
        &training_ctx,
        &StageContextFixture::new(&[], &[], &[]).ctx(),
        WorkspaceSizing::default(),
    )]
}

// ── Tests ────────────────────────────────────────────────────────────────

/// Acceptance criterion: `n_scenarios=4`, single rank → exactly 4 results in
/// channel and cost buffer has length 4.
#[test]
fn simulate_single_rank_4_scenarios_produces_4_results() {
    let n_stages = 2;
    let templates: Vec<StageTemplate> = (0..n_stages).map(|_| hydro_only_bus_template()).collect();

    let fcf = FutureCostFunction::new(n_stages, 1, 1, 10, &vec![0; n_stages]);
    let stochastic = make_stochastic_context(n_stages);
    let config = SimulationConfig {
        n_scenarios: 4,
        io_channel_capacity: 16,
        profile: Phase::Simulation.profile(),
    };
    let state = state_layout_for(1, 0);
    let horizon = HorizonMode::Finite {
        num_stages: n_stages,
    };
    let initial_state = vec![50.0_f64];

    let solution = hydro_only_bus_solution(100.0, 30.0);
    let solver = MockSolver::always_ok(solution);
    let comm = StubComm { rank: 0, size: 1 };
    let entity_counts = entity_counts_1_hydro();

    let (tx, rx) = mpsc::sync_channel(16);

    let hprod = hydro_productivities_1hydro(n_stages);
    let ec = zero_energy_conversion(1, n_stages);
    let mut workspaces = single_workspace(solver);
    let state_boxes = permissive_state_boxes(state.n_state, n_stages);
    let geometry = vec![hydro_only_bus_geometry(); n_stages];
    let stage_ctx_fixture = StageContextFixture::new(&templates, &state_boxes, &geometry);
    let result = cobre_sddp::simulate(
        &mut workspaces,
        &stage_ctx_fixture.ctx(),
        &fcf,
        &TrainingContext {
            node_graph: &cobre_sddp::test_support::chain_node_graph(&stochastic),
            horizon: &horizon,
            state: &state,
            cut_state_layouts: &all_enabled_cut_state_layouts(&state, n_stages),
            study_dims: &study_dims(),
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &stochastic,
            initial_state: &initial_state,
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            stages: &[],
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            lag_accum_seed: &[],
            lag_weight_seed: &[],
            dcs: None,
        },
        &config,
        SimulationOutputSpec {
            result_tx: &tx,
            hydro_cell_index: &cobre_sddp::test_support::identity_hydro_cell_index(256),
            block_hours_per_stage: &vec![vec![744.0]; 2],
            entity_counts: &entity_counts,
            generic_constraint_row_entries: &vec![Vec::new(); 2],
            pumping_consumption_mw_per_m3s: &[],
            contract_prices_per_stage: &vec![Vec::new(); 2],
            contract_slots: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities_per_stage: &hprod,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[0.0],
            event_sender: None,
            extended_delivery_anchors: &[],
            transit_seed_arcs: &[],
            past_defluences: &[],
            study_stage_dates: &[],
        },
        None,
        &[],
        &comm,
        &Traversal::default(),
    );

    assert!(result.is_ok(), "simulate returned error: {result:?}");
    let run_result = result.unwrap();
    assert_eq!(
        run_result.costs.len(),
        4,
        "cost buffer should have 4 entries"
    );

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
    let templates: Vec<StageTemplate> = (0..n_stages).map(|_| hydro_only_bus_template()).collect();

    let state = state_layout_for(1, 0);
    let fcf = FutureCostFunction::new(n_stages, 1, 1, 10, &vec![0; n_stages]);
    let stochastic = make_stochastic_context(n_stages);
    let config = SimulationConfig {
        n_scenarios: 4,
        io_channel_capacity: 16,
        profile: Phase::Simulation.profile(),
    };
    let horizon = HorizonMode::Finite {
        num_stages: n_stages,
    };
    let initial_state = vec![50.0_f64];

    let solution = hydro_only_bus_solution(100.0, 30.0);
    let solver = MockSolver::infeasible_on(solution, 5);
    let comm = StubComm { rank: 0, size: 1 };
    let entity_counts = entity_counts_1_hydro();

    let (tx, _rx) = mpsc::sync_channel(16);

    let hprod = hydro_productivities_1hydro(n_stages);
    let ec = zero_energy_conversion(1, n_stages);
    let mut workspaces = single_workspace(solver);
    let state_boxes = permissive_state_boxes(state.n_state, n_stages);
    let geometry = vec![hydro_only_bus_geometry(); n_stages];
    let stage_ctx_fixture = StageContextFixture::new(&templates, &state_boxes, &geometry);
    let result = cobre_sddp::simulate(
        &mut workspaces,
        &stage_ctx_fixture.ctx(),
        &fcf,
        &TrainingContext {
            node_graph: &cobre_sddp::test_support::chain_node_graph(&stochastic),
            horizon: &horizon,
            state: &state,
            cut_state_layouts: &all_enabled_cut_state_layouts(&state, n_stages),
            study_dims: &study_dims(),
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &stochastic,
            initial_state: &initial_state,
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            stages: &[],
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            lag_accum_seed: &[],
            lag_weight_seed: &[],
            dcs: None,
        },
        &config,
        SimulationOutputSpec {
            result_tx: &tx,
            hydro_cell_index: &cobre_sddp::test_support::identity_hydro_cell_index(256),
            block_hours_per_stage: &vec![vec![744.0]; 2],
            entity_counts: &entity_counts,
            generic_constraint_row_entries: &vec![Vec::new(); 2],
            pumping_consumption_mw_per_m3s: &[],
            contract_prices_per_stage: &vec![Vec::new(); 2],
            contract_slots: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities_per_stage: &hprod,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[0.0],
            event_sender: None,
            extended_delivery_anchors: &[],
            transit_seed_arcs: &[],
            past_defluences: &[],
            study_stage_dates: &[],
        },
        None,
        &[],
        &comm,
        &Traversal::default(),
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

/// solver infeasible at scenario 2, stage 3
/// with 4 scenarios and 4 stages → `SimulationError::LpInfeasible { scenario_id: 2, stage_id: 3 }`.
///
/// Solve call index for (scenario=2, stage=3) = 2*4 + 3 = 11 (0-based).
#[test]
fn simulate_infeasible_at_scenario2_stage3() {
    let n_stages = 4;
    let templates: Vec<StageTemplate> = (0..n_stages).map(|_| hydro_only_bus_template()).collect();

    let state = state_layout_for(1, 0);
    let fcf = FutureCostFunction::new(n_stages, 1, 1, 10, &vec![0; n_stages]);
    let stochastic = make_stochastic_context(n_stages);
    let config = SimulationConfig {
        n_scenarios: 4,
        io_channel_capacity: 16,
        profile: Phase::Simulation.profile(),
    };
    let horizon = HorizonMode::Finite {
        num_stages: n_stages,
    };
    let initial_state = vec![50.0_f64];

    let solution = hydro_only_bus_solution(100.0, 30.0);
    let solver = MockSolver::infeasible_on(solution, 11);
    let comm = StubComm { rank: 0, size: 1 };
    let entity_counts = entity_counts_1_hydro();

    let (tx, _rx) = mpsc::sync_channel(16);

    let hprod = hydro_productivities_1hydro(n_stages);
    let ec = zero_energy_conversion(1, n_stages);
    let mut workspaces = single_workspace(solver);
    let state_boxes = permissive_state_boxes(state.n_state, n_stages);
    let geometry = vec![hydro_only_bus_geometry(); n_stages];
    let stage_ctx_fixture = StageContextFixture::new(&templates, &state_boxes, &geometry);
    let result = cobre_sddp::simulate(
        &mut workspaces,
        &stage_ctx_fixture.ctx(),
        &fcf,
        &TrainingContext {
            node_graph: &cobre_sddp::test_support::chain_node_graph(&stochastic),
            horizon: &horizon,
            state: &state,
            cut_state_layouts: &all_enabled_cut_state_layouts(&state, n_stages),
            study_dims: &study_dims(),
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &stochastic,
            initial_state: &initial_state,
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            stages: &[],
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            lag_accum_seed: &[],
            lag_weight_seed: &[],
            dcs: None,
        },
        &config,
        SimulationOutputSpec {
            result_tx: &tx,
            hydro_cell_index: &cobre_sddp::test_support::identity_hydro_cell_index(256),
            block_hours_per_stage: &vec![vec![744.0]; 4],
            entity_counts: &entity_counts,
            generic_constraint_row_entries: &vec![Vec::new(); 4],
            pumping_consumption_mw_per_m3s: &[],
            contract_prices_per_stage: &vec![Vec::new(); 4],
            contract_slots: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities_per_stage: &hprod,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[0.0],
            event_sender: None,
            extended_delivery_anchors: &[],
            transit_seed_arcs: &[],
            past_defluences: &[],
            study_stage_dates: &[],
        },
        None,
        &[],
        &comm,
        &Traversal::default(),
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
    let templates: Vec<StageTemplate> = (0..n_stages).map(|_| hydro_only_bus_template()).collect();

    let state = state_layout_for(1, 0);
    let fcf = FutureCostFunction::new(n_stages, 1, 1, 10, &vec![0; n_stages]);
    let stochastic = make_stochastic_context(n_stages);
    let config = SimulationConfig {
        n_scenarios: 2,
        io_channel_capacity: 1,
        profile: Phase::Simulation.profile(),
    };
    let horizon = HorizonMode::Finite {
        num_stages: n_stages,
    };
    let initial_state = vec![50.0_f64];

    let solution = hydro_only_bus_solution(100.0, 30.0);
    let solver = MockSolver::always_ok(solution);
    let comm = StubComm { rank: 0, size: 1 };
    let entity_counts = entity_counts_1_hydro();

    let (tx, rx) = mpsc::sync_channel(1);
    // Drop the receiver immediately so send() will fail.
    drop(rx);

    let hprod = hydro_productivities_1hydro(n_stages);
    let ec = zero_energy_conversion(1, n_stages);
    let mut workspaces = single_workspace(solver);
    let state_boxes = permissive_state_boxes(state.n_state, n_stages);
    let geometry = vec![hydro_only_bus_geometry(); n_stages];
    let stage_ctx_fixture = StageContextFixture::new(&templates, &state_boxes, &geometry);
    let result = cobre_sddp::simulate(
        &mut workspaces,
        &stage_ctx_fixture.ctx(),
        &fcf,
        &TrainingContext {
            node_graph: &cobre_sddp::test_support::chain_node_graph(&stochastic),
            horizon: &horizon,
            state: &state,
            cut_state_layouts: &all_enabled_cut_state_layouts(&state, n_stages),
            study_dims: &study_dims(),
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &stochastic,
            initial_state: &initial_state,
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            stages: &[],
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            lag_accum_seed: &[],
            lag_weight_seed: &[],
            dcs: None,
        },
        &config,
        SimulationOutputSpec {
            result_tx: &tx,
            hydro_cell_index: &cobre_sddp::test_support::identity_hydro_cell_index(256),
            block_hours_per_stage: &vec![vec![744.0]; 2],
            entity_counts: &entity_counts,
            generic_constraint_row_entries: &vec![Vec::new(); 2],
            pumping_consumption_mw_per_m3s: &[],
            contract_prices_per_stage: &vec![Vec::new(); 2],
            contract_slots: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities_per_stage: &hprod,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[0.0],
            event_sender: None,
            extended_delivery_anchors: &[],
            transit_seed_arcs: &[],
            past_defluences: &[],
            study_stage_dates: &[],
        },
        None,
        &[],
        &comm,
        &Traversal::default(),
    );

    assert!(
        matches!(result, Err(SimulationError::ChannelClosed)),
        "expected ChannelClosed, got {result:?}"
    );
}

/// Acceptance criterion: `total_cost` in cost buffer equals sum of
/// `(objective - primal[theta])` across all stages for each scenario.
///
/// With objective=100.0 and theta=30.0: `stage_cost` = (100 - 30) * `COST_SCALE_FACTOR` = `70_000_000` per stage.
/// For 3 stages: `total_cost` = 3 \* `70_000_000` = `210_000_000`.
#[test]
fn simulate_total_cost_equals_sum_of_stage_costs() {
    let n_stages = 3;
    let templates: Vec<StageTemplate> = (0..n_stages).map(|_| hydro_only_bus_template()).collect();

    let fcf = FutureCostFunction::new(n_stages, 1, 1, 10, &vec![0; n_stages]);
    let stochastic = make_stochastic_context(n_stages);
    let config = SimulationConfig {
        n_scenarios: 2,
        io_channel_capacity: 16,
        profile: Phase::Simulation.profile(),
    };
    let state = state_layout_for(1, 0);
    let horizon = HorizonMode::Finite {
        num_stages: n_stages,
    };
    let initial_state = vec![50.0_f64];

    let objective = 100.0_f64;
    let theta_val = 30.0_f64;
    let expected_stage_cost = (objective - theta_val) * 1_000_000.0;
    let expected_total_cost = expected_stage_cost * n_stages as f64;

    let solution = hydro_only_bus_solution(objective, theta_val);
    let solver = MockSolver::always_ok(solution);
    let comm = StubComm { rank: 0, size: 1 };
    let entity_counts = entity_counts_1_hydro();

    let (tx, _rx) = mpsc::sync_channel(16);

    let hprod = hydro_productivities_1hydro(n_stages);
    let ec = zero_energy_conversion(1, n_stages);
    let mut workspaces = single_workspace(solver);
    let state_boxes = permissive_state_boxes(state.n_state, n_stages);
    let geometry = vec![hydro_only_bus_geometry(); n_stages];
    let stage_ctx_fixture = StageContextFixture::new(&templates, &state_boxes, &geometry);
    let run_result = cobre_sddp::simulate(
        &mut workspaces,
        &stage_ctx_fixture.ctx(),
        &fcf,
        &TrainingContext {
            node_graph: &cobre_sddp::test_support::chain_node_graph(&stochastic),
            horizon: &horizon,
            state: &state,
            cut_state_layouts: &all_enabled_cut_state_layouts(&state, n_stages),
            study_dims: &study_dims(),
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &stochastic,
            initial_state: &initial_state,
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            stages: &[],
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            lag_accum_seed: &[],
            lag_weight_seed: &[],
            dcs: None,
        },
        &config,
        SimulationOutputSpec {
            result_tx: &tx,
            hydro_cell_index: &cobre_sddp::test_support::identity_hydro_cell_index(256),
            block_hours_per_stage: &vec![vec![744.0]; 3],
            entity_counts: &entity_counts,
            generic_constraint_row_entries: &vec![Vec::new(); 3],
            pumping_consumption_mw_per_m3s: &[],
            contract_prices_per_stage: &vec![Vec::new(); 3],
            contract_slots: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities_per_stage: &hprod,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[0.0],
            event_sender: None,
            extended_delivery_anchors: &[],
            transit_seed_arcs: &[],
            past_defluences: &[],
            study_stage_dates: &[],
        },
        None,
        &[],
        &comm,
        &Traversal::default(),
    )
    .unwrap();

    assert_eq!(run_result.costs.len(), 2);
    for (scenario_id, total_cost, _) in &run_result.costs {
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
    let templates: Vec<StageTemplate> = (0..n_stages).map(|_| hydro_only_bus_template()).collect();

    let state = state_layout_for(1, 0);
    let fcf = FutureCostFunction::new(n_stages, 1, 1, 10, &vec![0; n_stages]);
    let stochastic = make_stochastic_context(n_stages);
    let config = SimulationConfig {
        n_scenarios: 6,
        io_channel_capacity: 16,
        profile: Phase::Simulation.profile(),
    };
    let horizon = HorizonMode::Finite {
        num_stages: n_stages,
    };
    let initial_state = vec![50.0_f64];

    let solution = hydro_only_bus_solution(50.0, 10.0);
    let solver = MockSolver::always_ok(solution);
    let comm = StubComm { rank: 0, size: 2 };
    let entity_counts = entity_counts_1_hydro();

    let (tx, _rx) = mpsc::sync_channel(16);

    let hprod = hydro_productivities_1hydro(n_stages);
    let ec = zero_energy_conversion(1, n_stages);
    let mut workspaces = single_workspace(solver);
    let state_boxes = permissive_state_boxes(state.n_state, n_stages);
    let geometry = vec![hydro_only_bus_geometry(); n_stages];
    let stage_ctx_fixture = StageContextFixture::new(&templates, &state_boxes, &geometry);
    let run_result = cobre_sddp::simulate(
        &mut workspaces,
        &stage_ctx_fixture.ctx(),
        &fcf,
        &TrainingContext {
            node_graph: &cobre_sddp::test_support::chain_node_graph(&stochastic),
            horizon: &horizon,
            state: &state,
            cut_state_layouts: &all_enabled_cut_state_layouts(&state, n_stages),
            study_dims: &study_dims(),
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &stochastic,
            initial_state: &initial_state,
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            stages: &[],
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            lag_accum_seed: &[],
            lag_weight_seed: &[],
            dcs: None,
        },
        &config,
        SimulationOutputSpec {
            result_tx: &tx,
            hydro_cell_index: &cobre_sddp::test_support::identity_hydro_cell_index(256),
            block_hours_per_stage: &vec![vec![744.0]; 1],
            entity_counts: &entity_counts,
            generic_constraint_row_entries: &vec![Vec::new(); 1],
            pumping_consumption_mw_per_m3s: &[],
            contract_prices_per_stage: &vec![Vec::new(); 1],
            contract_slots: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities_per_stage: &hprod,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[0.0],
            event_sender: None,
            extended_delivery_anchors: &[],
            transit_seed_arcs: &[],
            past_defluences: &[],
            study_stage_dates: &[],
        },
        None,
        &[],
        &comm,
        &Traversal::default(),
    )
    .unwrap();

    assert_eq!(
        run_result.costs.len(),
        3,
        "rank 0 should process 3 scenarios"
    );
    let ids: Vec<u32> = run_result.costs.iter().map(|(id, _, _)| *id).collect();
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
    let templates: Vec<StageTemplate> = (0..n_stages).map(|_| hydro_only_bus_template()).collect();

    let state = state_layout_for(1, 0);
    let fcf = FutureCostFunction::new(n_stages, 1, 1, 10, &vec![0; n_stages]);
    let stochastic = make_stochastic_context(n_stages);
    let config = SimulationConfig {
        n_scenarios: 3,
        io_channel_capacity: 16,
        profile: Phase::Simulation.profile(),
    };
    let horizon = HorizonMode::Finite {
        num_stages: n_stages,
    };
    let initial_state = vec![50.0_f64];

    let solution = hydro_only_bus_solution(100.0, 20.0);
    let solver = MockSolver::always_ok(solution);
    let comm = StubComm { rank: 0, size: 1 };
    let entity_counts = entity_counts_1_hydro();

    let (tx, rx) = mpsc::sync_channel(16);

    let hprod = hydro_productivities_1hydro(n_stages);
    let ec = zero_energy_conversion(1, n_stages);
    let mut workspaces = single_workspace(solver);
    let state_boxes = permissive_state_boxes(state.n_state, n_stages);
    let geometry = vec![hydro_only_bus_geometry(); n_stages];
    let stage_ctx_fixture = StageContextFixture::new(&templates, &state_boxes, &geometry);
    cobre_sddp::simulate(
        &mut workspaces,
        &stage_ctx_fixture.ctx(),
        &fcf,
        &TrainingContext {
            node_graph: &cobre_sddp::test_support::chain_node_graph(&stochastic),
            horizon: &horizon,
            state: &state,
            cut_state_layouts: &all_enabled_cut_state_layouts(&state, n_stages),
            study_dims: &study_dims(),
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &stochastic,
            initial_state: &initial_state,
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            stages: &[],
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            lag_accum_seed: &[],
            lag_weight_seed: &[],
            dcs: None,
        },
        &config,
        SimulationOutputSpec {
            result_tx: &tx,
            hydro_cell_index: &cobre_sddp::test_support::identity_hydro_cell_index(256),
            block_hours_per_stage: &vec![vec![744.0]; 1],
            entity_counts: &entity_counts,
            generic_constraint_row_entries: &vec![Vec::new(); 1],
            pumping_consumption_mw_per_m3s: &[],
            contract_prices_per_stage: &vec![Vec::new(); 1],
            contract_slots: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities_per_stage: &hprod,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[0.0],
            event_sender: None,
            extended_delivery_anchors: &[],
            transit_seed_arcs: &[],
            past_defluences: &[],
            study_stage_dates: &[],
        },
        None,
        &[],
        &comm,
        &Traversal::default(),
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
    let templates: Vec<StageTemplate> = (0..n_stages).map(|_| hydro_only_bus_template()).collect();

    let state = state_layout_for(1, 0);
    let fcf = FutureCostFunction::new(n_stages, 1, 1, 10, &vec![0; n_stages]);
    let stochastic = make_stochastic_context(n_stages);
    let config = SimulationConfig {
        n_scenarios,
        io_channel_capacity: 64,
        profile: Phase::Simulation.profile(),
    };
    let horizon = HorizonMode::Finite {
        num_stages: n_stages,
    };
    let initial_state = vec![50.0_f64];

    let objective = 100.0_f64;
    let theta_val = 30.0_f64;
    let solution = hydro_only_bus_solution(objective, theta_val);
    let comm = StubComm { rank: 0, size: 1 };
    let entity_counts = entity_counts_1_hydro();

    let hprod = hydro_productivities_1hydro(n_stages);
    let ec = zero_energy_conversion(1, n_stages);

    let (tx1, _rx1) = mpsc::sync_channel(64);
    let mut workspaces_1 = single_workspace(MockSolver::always_ok(solution.clone()));
    let state_boxes = permissive_state_boxes(state.n_state, n_stages);
    let geometry = vec![hydro_only_bus_geometry(); n_stages];
    let stage_ctx_fixture = StageContextFixture::new(&templates, &state_boxes, &geometry);
    let result_1 = cobre_sddp::simulate(
        &mut workspaces_1,
        &stage_ctx_fixture.ctx(),
        &fcf,
        &TrainingContext {
            node_graph: &cobre_sddp::test_support::chain_node_graph(&stochastic),
            horizon: &horizon,
            state: &state,
            cut_state_layouts: &all_enabled_cut_state_layouts(&state, n_stages),
            study_dims: &study_dims(),
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &stochastic,
            initial_state: &initial_state,
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            stages: &[],
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            lag_accum_seed: &[],
            lag_weight_seed: &[],
            dcs: None,
        },
        &config,
        SimulationOutputSpec {
            result_tx: &tx1,
            hydro_cell_index: &cobre_sddp::test_support::identity_hydro_cell_index(256),
            block_hours_per_stage: &vec![vec![744.0]; 2],
            entity_counts: &entity_counts,
            generic_constraint_row_entries: &vec![Vec::new(); 2],
            pumping_consumption_mw_per_m3s: &[],
            contract_prices_per_stage: &vec![Vec::new(); 2],
            contract_slots: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities_per_stage: &hprod,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[0.0],
            event_sender: None,
            extended_delivery_anchors: &[],
            transit_seed_arcs: &[],
            past_defluences: &[],
            study_stage_dates: &[],
        },
        None,
        &[],
        &comm,
        &Traversal::default(),
    )
    .unwrap();

    let (tx4, _rx4) = mpsc::sync_channel(64);
    let workspace_4_training_ctx = TrainingContext {
        node_graph: &cobre_sddp::test_support::chain_node_graph(&stochastic),
        horizon: &horizon,
        state: &state,
        cut_state_layouts: &all_enabled_cut_state_layouts(&state, n_stages),
        study_dims: &study_dims(),
        inflow_method: &InflowNonNegativityMethod::None,
        stochastic: &stochastic,
        initial_state: &initial_state,
        inflow_scheme: SamplingScheme::InSample,
        load_scheme: SamplingScheme::InSample,
        ncs_scheme: SamplingScheme::InSample,
        stages: &[],
        historical_library: None,
        external_inflow_library: None,
        external_load_library: None,
        external_ncs_library: None,
        lag_accum_seed: &[],
        lag_weight_seed: &[],
        dcs: None,
    };
    let mut workspaces_4: Vec<SolverWorkspace<MockSolver>> = (0..4_i32)
        .map(|idx| {
            SolverWorkspace::new(
                0,
                idx,
                MockSolver::always_ok(solution.clone()),
                PatchBuffer::new(&state, &[], &[]),
                &workspace_4_training_ctx,
                &stage_ctx_fixture.ctx(),
                WorkspaceSizing::default(),
            )
        })
        .collect();
    let state_boxes = permissive_state_boxes(state.n_state, n_stages);
    let geometry = vec![hydro_only_bus_geometry(); n_stages];
    let stage_ctx_fixture = StageContextFixture::new(&templates, &state_boxes, &geometry);
    let result_4 = cobre_sddp::simulate(
        &mut workspaces_4,
        &stage_ctx_fixture.ctx(),
        &fcf,
        &TrainingContext {
            node_graph: &cobre_sddp::test_support::chain_node_graph(&stochastic),
            horizon: &horizon,
            state: &state,
            cut_state_layouts: &all_enabled_cut_state_layouts(&state, n_stages),
            study_dims: &study_dims(),
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &stochastic,
            initial_state: &initial_state,
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            stages: &[],
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            lag_accum_seed: &[],
            lag_weight_seed: &[],
            dcs: None,
        },
        &config,
        SimulationOutputSpec {
            result_tx: &tx4,
            hydro_cell_index: &cobre_sddp::test_support::identity_hydro_cell_index(256),
            block_hours_per_stage: &vec![vec![744.0]; 2],
            entity_counts: &entity_counts,
            generic_constraint_row_entries: &vec![Vec::new(); 2],
            pumping_consumption_mw_per_m3s: &[],
            contract_prices_per_stage: &vec![Vec::new(); 2],
            contract_slots: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities_per_stage: &hprod,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[0.0],
            event_sender: None,
            extended_delivery_anchors: &[],
            transit_seed_arcs: &[],
            past_defluences: &[],
            study_stage_dates: &[],
        },
        None,
        &[],
        &comm,
        &Traversal::default(),
    )
    .unwrap();

    let costs_1 = &result_1.costs;
    let costs_4 = &result_4.costs;

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

// ── Integration tests for event emission ─────────────────────────────────

/// Acceptance criterion: with `event_sender: Some(&tx)` and 10 scenarios,
/// at least 1 `SimulationProgress` event is received with `scenarios_complete > 0`
/// and a finite non-NaN `scenario_cost`.
#[test]
fn simulate_emits_progress_events() {
    use cobre_core::TrainingEvent;

    let n_stages = 2;
    let templates: Vec<StageTemplate> = (0..n_stages).map(|_| hydro_only_bus_template()).collect();

    let state = state_layout_for(1, 0);
    let fcf = FutureCostFunction::new(n_stages, 1, 1, 10, &vec![0; n_stages]);
    let stochastic = make_stochastic_context(n_stages);
    let config = SimulationConfig {
        n_scenarios: 10,
        io_channel_capacity: 32,
        profile: Phase::Simulation.profile(),
    };
    let horizon = HorizonMode::Finite {
        num_stages: n_stages,
    };
    let initial_state = vec![50.0_f64];

    let solution = hydro_only_bus_solution(100.0, 30.0);
    let solver = MockSolver::always_ok(solution);
    let comm = StubComm { rank: 0, size: 1 };
    let entity_counts = entity_counts_1_hydro();

    let (result_tx, _result_rx) = mpsc::sync_channel(32);
    let (event_tx, event_rx) = mpsc::channel::<TrainingEvent>();

    let hprod = hydro_productivities_1hydro(n_stages);
    let ec = zero_energy_conversion(1, n_stages);
    let mut workspaces = single_workspace(solver);
    let state_boxes = permissive_state_boxes(state.n_state, n_stages);
    let geometry = vec![hydro_only_bus_geometry(); n_stages];
    let stage_ctx_fixture = StageContextFixture::new(&templates, &state_boxes, &geometry);
    let result = cobre_sddp::simulate(
        &mut workspaces,
        &stage_ctx_fixture.ctx(),
        &fcf,
        &TrainingContext {
            node_graph: &cobre_sddp::test_support::chain_node_graph(&stochastic),
            horizon: &horizon,
            state: &state,
            cut_state_layouts: &all_enabled_cut_state_layouts(&state, n_stages),
            study_dims: &study_dims(),
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &stochastic,
            initial_state: &initial_state,
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            stages: &[],
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            lag_accum_seed: &[],
            lag_weight_seed: &[],
            dcs: None,
        },
        &config,
        SimulationOutputSpec {
            result_tx: &result_tx,
            hydro_cell_index: &cobre_sddp::test_support::identity_hydro_cell_index(256),
            block_hours_per_stage: &vec![vec![744.0]; 2],
            entity_counts: &entity_counts,
            generic_constraint_row_entries: &vec![Vec::new(); 2],
            pumping_consumption_mw_per_m3s: &[],
            contract_prices_per_stage: &vec![Vec::new(); 2],
            contract_slots: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities_per_stage: &hprod,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[0.0],
            event_sender: Some(event_tx),
            extended_delivery_anchors: &[],
            transit_seed_arcs: &[],
            past_defluences: &[],
            study_stage_dates: &[],
        },
        None,
        &[],
        &comm,
        &Traversal::default(),
    );
    assert!(result.is_ok(), "simulate returned error: {result:?}");

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
            scenario_cost,
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
            scenario_cost.is_finite() && !scenario_cost.is_nan(),
            "scenario_cost must be finite and non-NaN, got {scenario_cost}"
        );
    }
}

/// Acceptance criterion: with `event_sender: None`, no events are sent and
/// the function returns the same cost buffer as before.
#[test]
fn simulate_no_events_when_sender_is_none() {
    let n_stages = 2;
    let templates: Vec<StageTemplate> = (0..n_stages).map(|_| hydro_only_bus_template()).collect();

    let state = state_layout_for(1, 0);
    let fcf = FutureCostFunction::new(n_stages, 1, 1, 10, &vec![0; n_stages]);
    let stochastic = make_stochastic_context(n_stages);
    let config = SimulationConfig {
        n_scenarios: 4,
        io_channel_capacity: 16,
        profile: Phase::Simulation.profile(),
    };
    let horizon = HorizonMode::Finite {
        num_stages: n_stages,
    };
    let initial_state = vec![50.0_f64];

    let solution = hydro_only_bus_solution(100.0, 30.0);
    let solver = MockSolver::always_ok(solution);
    let comm = StubComm { rank: 0, size: 1 };
    let entity_counts = entity_counts_1_hydro();

    let (result_tx, _result_rx) = mpsc::sync_channel(16);

    let hprod = hydro_productivities_1hydro(n_stages);
    let ec = zero_energy_conversion(1, n_stages);
    let mut workspaces = single_workspace(solver);
    let state_boxes = permissive_state_boxes(state.n_state, n_stages);
    let geometry = vec![hydro_only_bus_geometry(); n_stages];
    let stage_ctx_fixture = StageContextFixture::new(&templates, &state_boxes, &geometry);
    let result = cobre_sddp::simulate(
        &mut workspaces,
        &stage_ctx_fixture.ctx(),
        &fcf,
        &TrainingContext {
            node_graph: &cobre_sddp::test_support::chain_node_graph(&stochastic),
            horizon: &horizon,
            state: &state,
            cut_state_layouts: &all_enabled_cut_state_layouts(&state, n_stages),
            study_dims: &study_dims(),
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &stochastic,
            initial_state: &initial_state,
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            stages: &[],
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            lag_accum_seed: &[],
            lag_weight_seed: &[],
            dcs: None,
        },
        &config,
        SimulationOutputSpec {
            result_tx: &result_tx,
            hydro_cell_index: &cobre_sddp::test_support::identity_hydro_cell_index(256),
            block_hours_per_stage: &vec![vec![744.0]; 2],
            entity_counts: &entity_counts,
            generic_constraint_row_entries: &vec![Vec::new(); 2],
            pumping_consumption_mw_per_m3s: &[],
            contract_prices_per_stage: &vec![Vec::new(); 2],
            contract_slots: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities_per_stage: &hprod,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[0.0],
            event_sender: None,
            extended_delivery_anchors: &[],
            transit_seed_arcs: &[],
            past_defluences: &[],
            study_stage_dates: &[],
        },
        None,
        &[],
        &comm,
        &Traversal::default(),
    );

    assert!(result.is_ok(), "simulate returned error: {result:?}");
    let run_result = result.unwrap();
    assert_eq!(
        run_result.costs.len(),
        4,
        "cost buffer must have 4 entries when event_sender is None"
    );
}

/// `SimulationProgress` events are
/// received in the channel BEFORE `simulate()` returns (during the
/// parallel region).
///
/// With a single workspace (serial rayon execution), the worker emits
/// progress events as each scenario completes. Because events are sent
/// from the closure rather than the post-collect loop, the receiver
/// contains events by the time `simulate()` returns.
#[test]
fn simulate_progress_events_received_before_return() {
    use cobre_core::TrainingEvent;

    let n_stages = 1;
    let n_scenarios = 10;
    let templates: Vec<StageTemplate> = (0..n_stages).map(|_| hydro_only_bus_template()).collect();

    let state = state_layout_for(1, 0);
    let fcf = FutureCostFunction::new(n_stages, 1, 1, 10, &vec![0; n_stages]);
    let stochastic = make_stochastic_context(n_stages);
    let config = SimulationConfig {
        n_scenarios,
        io_channel_capacity: 32,
        profile: Phase::Simulation.profile(),
    };
    let horizon = HorizonMode::Finite {
        num_stages: n_stages,
    };
    let initial_state = vec![50.0_f64];

    let solution = hydro_only_bus_solution(100.0, 30.0);
    let solver = MockSolver::always_ok(solution);
    let comm = StubComm { rank: 0, size: 1 };
    let entity_counts = entity_counts_1_hydro();

    let (result_tx, _result_rx) = mpsc::sync_channel(32);
    let (event_tx, event_rx) = mpsc::channel::<TrainingEvent>();

    let hprod = hydro_productivities_1hydro(n_stages);
    let ec = zero_energy_conversion(1, n_stages);
    let mut workspaces = single_workspace(solver);
    let state_boxes = permissive_state_boxes(state.n_state, n_stages);
    let geometry = vec![hydro_only_bus_geometry(); n_stages];
    let stage_ctx_fixture = StageContextFixture::new(&templates, &state_boxes, &geometry);
    cobre_sddp::simulate(
        &mut workspaces,
        &stage_ctx_fixture.ctx(),
        &fcf,
        &TrainingContext {
            node_graph: &cobre_sddp::test_support::chain_node_graph(&stochastic),
            horizon: &horizon,
            state: &state,
            cut_state_layouts: &all_enabled_cut_state_layouts(&state, n_stages),
            study_dims: &study_dims(),
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &stochastic,
            initial_state: &initial_state,
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            stages: &[],
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            lag_accum_seed: &[],
            lag_weight_seed: &[],
            dcs: None,
        },
        &config,
        SimulationOutputSpec {
            result_tx: &result_tx,
            hydro_cell_index: &cobre_sddp::test_support::identity_hydro_cell_index(256),
            block_hours_per_stage: &vec![vec![744.0]; 1],
            entity_counts: &entity_counts,
            generic_constraint_row_entries: &vec![Vec::new(); 1],
            pumping_consumption_mw_per_m3s: &[],
            contract_prices_per_stage: &vec![Vec::new(); 1],
            contract_slots: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities_per_stage: &hprod,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[0.0],
            event_sender: Some(event_tx),
            extended_delivery_anchors: &[],
            transit_seed_arcs: &[],
            past_defluences: &[],
            study_stage_dates: &[],
        },
        None,
        &[],
        &comm,
        &Traversal::default(),
    )
    .unwrap();

    // simulate() moved and dropped the sender, so the channel is closed and
    // event_rx.iter() terminates.
    let events: Vec<TrainingEvent> = event_rx.iter().collect();
    let progress_count = events
        .iter()
        .filter(|e| matches!(e, TrainingEvent::SimulationProgress { .. }))
        .count();

    assert!(
        progress_count > 0,
        "expected SimulationProgress events in channel after simulate() returns, got 0"
    );
    assert_eq!(
        progress_count, n_scenarios as usize,
        "expected {n_scenarios} SimulationProgress events (one per scenario), got {progress_count}"
    );
}

/// Acceptance criterion: each `SimulationProgress` event carries the raw
/// `scenario_cost` of the completed scenario.
///
/// With `MockSolver` returning a fixed solution, all scenarios have the same
/// `total_cost`. Validates that every `SimulationProgress.scenario_cost`
/// equals the expected per-scenario cost.
#[test]
fn simulate_progress_scenario_cost_equals_total_cost() {
    use cobre_core::TrainingEvent;

    let n_stages = 1;
    let n_scenarios = 5_u32;
    let templates: Vec<StageTemplate> = (0..n_stages).map(|_| hydro_only_bus_template()).collect();

    let state = state_layout_for(1, 0);
    let fcf = FutureCostFunction::new(n_stages, 1, 1, 10, &vec![0; n_stages]);
    let stochastic = make_stochastic_context(n_stages);
    let config = SimulationConfig {
        n_scenarios,
        io_channel_capacity: 32,
        profile: Phase::Simulation.profile(),
    };
    let horizon = HorizonMode::Finite {
        num_stages: n_stages,
    };
    let initial_state = vec![50.0_f64];

    // objective=100, theta=30 → stage_cost = (100-30)*COST_SCALE_FACTOR = 70_000_000.0 every scenario.
    let solution = hydro_only_bus_solution(100.0, 30.0);
    let expected_stage_cost = 70_000_000.0_f64;

    let solver = MockSolver::always_ok(solution);
    let comm = StubComm { rank: 0, size: 1 };
    let entity_counts = entity_counts_1_hydro();

    let (result_tx, _result_rx) = mpsc::sync_channel(32);
    let (event_tx, event_rx) = mpsc::channel::<TrainingEvent>();

    let hprod = hydro_productivities_1hydro(n_stages);
    let ec = zero_energy_conversion(1, n_stages);
    let mut workspaces = single_workspace(solver);
    let state_boxes = permissive_state_boxes(state.n_state, n_stages);
    let geometry = vec![hydro_only_bus_geometry(); n_stages];
    let stage_ctx_fixture = StageContextFixture::new(&templates, &state_boxes, &geometry);
    cobre_sddp::simulate(
        &mut workspaces,
        &stage_ctx_fixture.ctx(),
        &fcf,
        &TrainingContext {
            node_graph: &cobre_sddp::test_support::chain_node_graph(&stochastic),
            horizon: &horizon,
            state: &state,
            cut_state_layouts: &all_enabled_cut_state_layouts(&state, n_stages),
            study_dims: &study_dims(),
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &stochastic,
            initial_state: &initial_state,
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            stages: &[],
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            lag_accum_seed: &[],
            lag_weight_seed: &[],
            dcs: None,
        },
        &config,
        SimulationOutputSpec {
            result_tx: &result_tx,
            hydro_cell_index: &cobre_sddp::test_support::identity_hydro_cell_index(256),
            block_hours_per_stage: &vec![vec![744.0]; 1],
            entity_counts: &entity_counts,
            generic_constraint_row_entries: &vec![Vec::new(); 1],
            pumping_consumption_mw_per_m3s: &[],
            contract_prices_per_stage: &vec![Vec::new(); 1],
            contract_slots: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities_per_stage: &hprod,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[0.0],
            event_sender: Some(event_tx),
            extended_delivery_anchors: &[],
            transit_seed_arcs: &[],
            past_defluences: &[],
            study_stage_dates: &[],
        },
        None,
        &[],
        &comm,
        &Traversal::default(),
    )
    .unwrap();

    let events: Vec<TrainingEvent> = event_rx.iter().collect();
    let progress_events: Vec<_> = events
        .iter()
        .filter(|e| matches!(e, TrainingEvent::SimulationProgress { .. }))
        .collect();

    assert_eq!(
        progress_events.len(),
        n_scenarios as usize,
        "expected {n_scenarios} progress events"
    );

    for event in &progress_events {
        let TrainingEvent::SimulationProgress { scenario_cost, .. } = event else {
            continue;
        };
        assert!(
            (scenario_cost - expected_stage_cost).abs() < 1e-9,
            "scenario_cost must equal expected cost {expected_stage_cost}, got {scenario_cost}"
        );
    }
}

/// Acceptance criterion: `SimulationFinished` event is the last event
/// emitted after all `SimulationProgress` events.
#[test]
fn simulate_emits_simulation_finished_as_last_event() {
    use cobre_core::TrainingEvent;

    let n_stages = 1;
    let n_scenarios = 6_u32;
    let templates: Vec<StageTemplate> = (0..n_stages).map(|_| hydro_only_bus_template()).collect();

    let state = state_layout_for(1, 0);
    let fcf = FutureCostFunction::new(n_stages, 1, 1, 10, &vec![0; n_stages]);
    let stochastic = make_stochastic_context(n_stages);
    let config = SimulationConfig {
        n_scenarios,
        io_channel_capacity: 32,
        profile: Phase::Simulation.profile(),
    };
    let horizon = HorizonMode::Finite {
        num_stages: n_stages,
    };
    let initial_state = vec![50.0_f64];

    let solution = hydro_only_bus_solution(100.0, 30.0);
    let solver = MockSolver::always_ok(solution);
    let comm = StubComm { rank: 0, size: 1 };
    let entity_counts = entity_counts_1_hydro();

    let (result_tx, _result_rx) = mpsc::sync_channel(32);
    let (event_tx, event_rx) = mpsc::channel::<TrainingEvent>();

    let hprod = hydro_productivities_1hydro(n_stages);
    let ec = zero_energy_conversion(1, n_stages);
    let mut workspaces = single_workspace(solver);
    let state_boxes = permissive_state_boxes(state.n_state, n_stages);
    let geometry = vec![hydro_only_bus_geometry(); n_stages];
    let stage_ctx_fixture = StageContextFixture::new(&templates, &state_boxes, &geometry);
    cobre_sddp::simulate(
        &mut workspaces,
        &stage_ctx_fixture.ctx(),
        &fcf,
        &TrainingContext {
            node_graph: &cobre_sddp::test_support::chain_node_graph(&stochastic),
            horizon: &horizon,
            state: &state,
            cut_state_layouts: &all_enabled_cut_state_layouts(&state, n_stages),
            study_dims: &study_dims(),
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &stochastic,
            initial_state: &initial_state,
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            stages: &[],
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            lag_accum_seed: &[],
            lag_weight_seed: &[],
            dcs: None,
        },
        &config,
        SimulationOutputSpec {
            result_tx: &result_tx,
            hydro_cell_index: &cobre_sddp::test_support::identity_hydro_cell_index(256),
            block_hours_per_stage: &vec![vec![744.0]; 1],
            entity_counts: &entity_counts,
            generic_constraint_row_entries: &vec![Vec::new(); 1],
            pumping_consumption_mw_per_m3s: &[],
            contract_prices_per_stage: &vec![Vec::new(); 1],
            contract_slots: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities_per_stage: &hprod,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[0.0],
            event_sender: Some(event_tx),
            extended_delivery_anchors: &[],
            transit_seed_arcs: &[],
            past_defluences: &[],
            study_stage_dates: &[],
        },
        None,
        &[],
        &comm,
        &Traversal::default(),
    )
    .unwrap();

    let events: Vec<TrainingEvent> = event_rx.iter().collect();

    assert!(
        events.len() > n_scenarios as usize,
        "expected at least {} events, got {}",
        n_scenarios + 1,
        events.len()
    );

    let last = events.last().unwrap();
    assert!(
        matches!(last, TrainingEvent::SimulationFinished { .. }),
        "last event must be SimulationFinished, got {last:?}"
    );

    let TrainingEvent::SimulationFinished { scenarios, .. } = last else {
        panic!("last event is not SimulationFinished");
    };
    assert_eq!(
        *scenarios, n_scenarios,
        "SimulationFinished.scenarios must equal n_scenarios={n_scenarios}, got {scenarios}"
    );

    let progress_count = events
        .iter()
        .filter(|e| matches!(e, TrainingEvent::SimulationProgress { .. }))
        .count();
    assert_eq!(
        progress_count, n_scenarios as usize,
        "expected {n_scenarios} SimulationProgress events before SimulationFinished"
    );
}

/// Acceptance criterion: each `SimulationProgress` event carries a finite,
/// non-NaN `scenario_cost`. Statistics accumulation is deferred to the
/// progress thread; this test verifies the per-scenario cost
/// field is always valid.
#[test]
fn simulate_progress_scenario_cost_is_finite() {
    use cobre_core::TrainingEvent;

    let n_stages = 1;
    let templates: Vec<StageTemplate> = (0..n_stages).map(|_| hydro_only_bus_template()).collect();

    let state = state_layout_for(1, 0);
    let fcf = FutureCostFunction::new(n_stages, 1, 1, 10, &vec![0; n_stages]);
    let stochastic = make_stochastic_context(n_stages);
    let config = SimulationConfig {
        n_scenarios: 5,
        io_channel_capacity: 16,
        profile: Phase::Simulation.profile(),
    };
    let horizon = HorizonMode::Finite {
        num_stages: n_stages,
    };
    let initial_state = vec![50.0_f64];

    let solution = hydro_only_bus_solution(100.0, 30.0);
    let solver = MockSolver::always_ok(solution);
    let comm = StubComm { rank: 0, size: 1 };
    let entity_counts = entity_counts_1_hydro();

    let (result_tx, _result_rx) = mpsc::sync_channel(16);
    let (event_tx, event_rx) = mpsc::channel::<TrainingEvent>();

    let hprod = hydro_productivities_1hydro(n_stages);
    let ec = zero_energy_conversion(1, n_stages);
    let mut workspaces = single_workspace(solver);
    let state_boxes = permissive_state_boxes(state.n_state, n_stages);
    let geometry = vec![hydro_only_bus_geometry(); n_stages];
    let stage_ctx_fixture = StageContextFixture::new(&templates, &state_boxes, &geometry);
    cobre_sddp::simulate(
        &mut workspaces,
        &stage_ctx_fixture.ctx(),
        &fcf,
        &TrainingContext {
            node_graph: &cobre_sddp::test_support::chain_node_graph(&stochastic),
            horizon: &horizon,
            state: &state,
            cut_state_layouts: &all_enabled_cut_state_layouts(&state, n_stages),
            study_dims: &study_dims(),
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &stochastic,
            initial_state: &initial_state,
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            stages: &[],
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            lag_accum_seed: &[],
            lag_weight_seed: &[],
            dcs: None,
        },
        &config,
        SimulationOutputSpec {
            result_tx: &result_tx,
            hydro_cell_index: &cobre_sddp::test_support::identity_hydro_cell_index(256),
            block_hours_per_stage: &vec![vec![744.0]; 1],
            entity_counts: &entity_counts,
            generic_constraint_row_entries: &vec![Vec::new(); 1],
            pumping_consumption_mw_per_m3s: &[],
            contract_prices_per_stage: &vec![Vec::new(); 1],
            contract_slots: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities_per_stage: &hprod,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[0.0],
            event_sender: Some(event_tx),
            extended_delivery_anchors: &[],
            transit_seed_arcs: &[],
            past_defluences: &[],
            study_stage_dates: &[],
        },
        None,
        &[],
        &comm,
        &Traversal::default(),
    )
    .unwrap();

    let events: Vec<TrainingEvent> = event_rx.iter().collect();
    let progress_events: Vec<_> = events
        .iter()
        .filter(|e| matches!(e, TrainingEvent::SimulationProgress { .. }))
        .collect();

    for event in &progress_events {
        let TrainingEvent::SimulationProgress { scenario_cost, .. } = event else {
            continue;
        };
        assert!(
            scenario_cost.is_finite() && !scenario_cost.is_nan(),
            "scenario_cost must be finite and non-NaN, got {scenario_cost}"
        );
    }
}

// ── frozen-template acceptance tests ────────────────────────────

/// When `frozen_templates` is `Some`,
/// `add_rows` is never called (zero `add_rows_count`) and `load_model` is
/// called exactly `n_scenarios * n_stages` times.
#[test]
fn simulate_frozen_path_issues_zero_add_rows() {
    let n_stages = 2;
    let n_scenarios = 3u32;
    let templates: Vec<StageTemplate> = (0..n_stages).map(|_| hydro_only_bus_template()).collect();
    // For MockSolver the frozen content is irrelevant; reuse the minimal template.
    let frozen: Vec<StageTemplate> = (0..n_stages).map(|_| hydro_only_bus_template()).collect();

    let state = state_layout_for(1, 0);
    let fcf = FutureCostFunction::new(n_stages, 1, 1, 10, &vec![0; n_stages]);
    let stochastic = make_stochastic_context(n_stages);
    let config = SimulationConfig {
        n_scenarios,
        io_channel_capacity: 16,
        profile: Phase::Simulation.profile(),
    };
    let horizon = HorizonMode::Finite {
        num_stages: n_stages,
    };
    let initial_state = vec![50.0_f64];

    let solution = hydro_only_bus_solution(100.0, 30.0);
    let solver = MockSolver::always_ok(solution);
    let comm = StubComm { rank: 0, size: 1 };
    let entity_counts = entity_counts_1_hydro();
    let (tx, _rx) = mpsc::sync_channel(32);
    let hprod = hydro_productivities_1hydro(n_stages);
    let ec = zero_energy_conversion(1, n_stages);

    let mut workspaces = single_workspace(solver);
    let state_boxes = permissive_state_boxes(state.n_state, n_stages);
    let geometry = vec![hydro_only_bus_geometry(); n_stages];
    let stage_ctx_fixture = StageContextFixture::new(&templates, &state_boxes, &geometry);
    let result = cobre_sddp::simulate(
        &mut workspaces,
        &stage_ctx_fixture.ctx(),
        &fcf,
        &TrainingContext {
            node_graph: &cobre_sddp::test_support::chain_node_graph(&stochastic),
            horizon: &horizon,
            state: &state,
            cut_state_layouts: &all_enabled_cut_state_layouts(&state, n_stages),
            study_dims: &study_dims(),
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &stochastic,
            initial_state: &initial_state,
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            stages: &[],
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            lag_accum_seed: &[],
            lag_weight_seed: &[],
            dcs: None,
        },
        &config,
        SimulationOutputSpec {
            result_tx: &tx,
            hydro_cell_index: &cobre_sddp::test_support::identity_hydro_cell_index(256),
            block_hours_per_stage: &vec![vec![744.0]; 2],
            entity_counts: &entity_counts,
            generic_constraint_row_entries: &vec![Vec::new(); 2],
            pumping_consumption_mw_per_m3s: &[],
            contract_prices_per_stage: &vec![Vec::new(); 2],
            contract_slots: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities_per_stage: &hprod,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[0.0],
            event_sender: None,
            extended_delivery_anchors: &[],
            transit_seed_arcs: &[],
            past_defluences: &[],
            study_stage_dates: &[],
        },
        Some(frozen.as_slice()),
        &[],
        &comm,
        &Traversal::default(),
    );

    assert!(result.is_ok(), "frozen path must succeed: {result:?}");
    let expected_load_count = n_scenarios as usize * n_stages;
    let solver = workspaces[0].solver.inner();
    assert_eq!(
        solver.add_rows_count, 0,
        "frozen path must call add_rows 0 times; got {}",
        solver.add_rows_count
    );
    assert_eq!(
        solver.load_count, expected_load_count,
        "frozen path must call load_model {} times; got {}",
        expected_load_count, solver.load_count
    );
}

/// Fallback path (`frozen_templates: None`): `add_rows` is gated by
/// `if cut_batch.num_rows > 0`, so with a 0-cut FCF `add_rows_count == 0` while
/// `load_count == n_scenarios * n_stages` (same `load_model` count as the frozen path).
#[test]
fn simulate_fallback_path_issues_expected_add_rows() {
    let n_stages = 2;
    let n_scenarios = 3u32;
    let templates: Vec<StageTemplate> = (0..n_stages).map(|_| hydro_only_bus_template()).collect();

    let state = state_layout_for(1, 0);
    let fcf = FutureCostFunction::new(n_stages, 1, 1, 10, &vec![0; n_stages]);
    let stochastic = make_stochastic_context(n_stages);
    let config = SimulationConfig {
        n_scenarios,
        io_channel_capacity: 16,
        profile: Phase::Simulation.profile(),
    };
    let horizon = HorizonMode::Finite {
        num_stages: n_stages,
    };
    let initial_state = vec![50.0_f64];

    let solution = hydro_only_bus_solution(100.0, 30.0);
    let solver = MockSolver::always_ok(solution);
    let comm = StubComm { rank: 0, size: 1 };
    let entity_counts = entity_counts_1_hydro();
    let (tx, _rx) = mpsc::sync_channel(32);
    let hprod = hydro_productivities_1hydro(n_stages);
    let ec = zero_energy_conversion(1, n_stages);

    let mut workspaces = single_workspace(solver);
    let state_boxes = permissive_state_boxes(state.n_state, n_stages);
    let geometry = vec![hydro_only_bus_geometry(); n_stages];
    let stage_ctx_fixture = StageContextFixture::new(&templates, &state_boxes, &geometry);
    let result = cobre_sddp::simulate(
        &mut workspaces,
        &stage_ctx_fixture.ctx(),
        &fcf,
        &TrainingContext {
            node_graph: &cobre_sddp::test_support::chain_node_graph(&stochastic),
            horizon: &horizon,
            state: &state,
            cut_state_layouts: &all_enabled_cut_state_layouts(&state, n_stages),
            study_dims: &study_dims(),
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &stochastic,
            initial_state: &initial_state,
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            stages: &[],
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            lag_accum_seed: &[],
            lag_weight_seed: &[],
            dcs: None,
        },
        &config,
        SimulationOutputSpec {
            result_tx: &tx,
            hydro_cell_index: &cobre_sddp::test_support::identity_hydro_cell_index(256),
            block_hours_per_stage: &vec![vec![744.0]; 2],
            entity_counts: &entity_counts,
            generic_constraint_row_entries: &vec![Vec::new(); 2],
            pumping_consumption_mw_per_m3s: &[],
            contract_prices_per_stage: &vec![Vec::new(); 2],
            contract_slots: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities_per_stage: &hprod,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[0.0],
            event_sender: None,
            extended_delivery_anchors: &[],
            transit_seed_arcs: &[],
            past_defluences: &[],
            study_stage_dates: &[],
        },
        None,
        &[],
        &comm,
        &Traversal::default(),
    );

    assert!(result.is_ok(), "fallback path must succeed: {result:?}");
    let expected_load_count = n_scenarios as usize * n_stages;
    let solver = workspaces[0].solver.inner();
    assert_eq!(
        solver.add_rows_count, 0,
        "fallback path with zero cuts must call add_rows 0 times; got {}",
        solver.add_rows_count
    );
    assert_eq!(
        solver.load_count, expected_load_count,
        "fallback path must call load_model {} times; got {}",
        expected_load_count, solver.load_count
    );
}

/// When `frozen_templates` is `Some`
/// but the slice length differs from `num_stages`, `simulate` returns
/// `SimulationError::InvalidConfiguration` whose message contains both lengths.
#[test]
fn simulate_frozen_length_mismatch_returns_error() {
    let n_stages = 3;
    let templates: Vec<StageTemplate> = (0..n_stages).map(|_| hydro_only_bus_template()).collect();

    let state = state_layout_for(1, 0);
    let fcf = FutureCostFunction::new(n_stages, 1, 1, 10, &vec![0; n_stages]);
    let stochastic = make_stochastic_context(n_stages);
    let config = SimulationConfig {
        n_scenarios: 2,
        io_channel_capacity: 8,
        profile: Phase::Simulation.profile(),
    };
    let horizon = HorizonMode::Finite {
        num_stages: n_stages,
    };
    let initial_state = vec![50.0_f64];

    let solution = hydro_only_bus_solution(100.0, 30.0);
    let solver = MockSolver::always_ok(solution);
    let comm = StubComm { rank: 0, size: 1 };
    let entity_counts = entity_counts_1_hydro();
    let (tx, _rx) = mpsc::sync_channel(8);
    let hprod = hydro_productivities_1hydro(n_stages);
    let ec = zero_energy_conversion(1, n_stages);

    let wrong_frozen: Vec<StageTemplate> = (0..n_stages - 1)
        .map(|_| hydro_only_bus_template())
        .collect();

    let mut workspaces = single_workspace(solver);
    let state_boxes = permissive_state_boxes(state.n_state, n_stages);
    let geometry = vec![hydro_only_bus_geometry(); n_stages];
    let stage_ctx_fixture = StageContextFixture::new(&templates, &state_boxes, &geometry);
    let result = cobre_sddp::simulate(
        &mut workspaces,
        &stage_ctx_fixture.ctx(),
        &fcf,
        &TrainingContext {
            node_graph: &cobre_sddp::test_support::chain_node_graph(&stochastic),
            horizon: &horizon,
            state: &state,
            cut_state_layouts: &all_enabled_cut_state_layouts(&state, n_stages),
            study_dims: &study_dims(),
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &stochastic,
            initial_state: &initial_state,
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            stages: &[],
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            lag_accum_seed: &[],
            lag_weight_seed: &[],
            dcs: None,
        },
        &config,
        SimulationOutputSpec {
            result_tx: &tx,
            hydro_cell_index: &cobre_sddp::test_support::identity_hydro_cell_index(256),
            block_hours_per_stage: &vec![vec![744.0]; 3],
            entity_counts: &entity_counts,
            generic_constraint_row_entries: &vec![Vec::new(); 3],
            pumping_consumption_mw_per_m3s: &[],
            contract_prices_per_stage: &vec![Vec::new(); 3],
            contract_slots: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities_per_stage: &hprod,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[0.0],
            event_sender: None,
            extended_delivery_anchors: &[],
            transit_seed_arcs: &[],
            past_defluences: &[],
            study_stage_dates: &[],
        },
        Some(wrong_frozen.as_slice()),
        &[],
        &comm,
        &Traversal::default(),
    );

    match &result {
        Err(SimulationError::InvalidConfiguration(msg)) => {
            assert!(
                msg.contains('2') && msg.contains('3'),
                "error message must contain both lengths (2 and 3), got: {msg}"
            );
        }
        other => panic!("expected InvalidConfiguration error, got: {other:?}"),
    }
}

// ── Warm-start CapturedBasis acceptance tests ─────────────────

/// Slot-identity preservation: with a `CapturedBasis` whose cut rows match the
/// FCF pool's active slots (10, 11, 12), the basis handed to `solve(Some(&basis))`
/// has `row_status.len() == base_row_count + active_cuts_count` and its tail
/// reproduces the stored cut statuses verbatim.
#[test]
fn simulate_with_captured_basis_preserves_row_statuses() {
    // Arbitrary distinct statuses; the test only checks they pass through unchanged.
    const CUT_STATUS_0: BasisStatus = BasisStatus::Superbasic;
    const CUT_STATUS_1: BasisStatus = BasisStatus::Zero;
    const CUT_STATUS_2: BasisStatus = BasisStatus::Fixed;
    const BASE_STATUS: BasisStatus = BasisStatus::Basic;

    let n_stages = 1;
    let n_scenarios = 1u32;
    let templates: Vec<StageTemplate> = vec![hydro_only_bus_template()];

    let state = state_layout_for(1, 0);

    // Build an FCF with 3 active cuts at slots 10, 11, 12 for stage 0.
    // warm_start_count=10, forward_passes=1 →
    //   add_cut(iter=0, fwd=0) → slot 10
    //   add_cut(iter=1, fwd=0) → slot 11
    //   add_cut(iter=2, fwd=0) → slot 12
    let mut fcf = FutureCostFunction::new(n_stages, 1, 1, 5, &[10]);
    fcf.pools[0].add_cut(NodeId(0), 0, 0, 50.0, &[1.0]);
    fcf.pools[0].add_cut(NodeId(0), 1, 0, 60.0, &[1.0]);
    fcf.pools[0].add_cut(NodeId(0), 2, 0, 70.0, &[1.0]);
    assert_eq!(
        fcf.pools[0].active_count(),
        3,
        "pool must have exactly 3 active cuts at slots 10, 11, 12"
    );
    assert_eq!(
        fcf.pools[0].populated(),
        13,
        "populated_count must be 13 (slot 12 + 1)"
    );

    // node_id must match the chain's node_ids[0] == 0 (n_stages = 1 below) or the
    // new node-tag check drops this basis to cold, breaking the warm-start
    // assertions this test exists to pin.
    let mut cb = CapturedBasis::new(15, 10, 7, 3, 1, NodeId(0));
    cb.basis.row_status = vec![
        BASE_STATUS,
        BASE_STATUS,
        BASE_STATUS,
        BASE_STATUS,
        BASE_STATUS,
        BASE_STATUS,
        BASE_STATUS,
        CUT_STATUS_0,
        CUT_STATUS_1,
        CUT_STATUS_2,
    ];
    cb.basis.col_status = vec![BasisStatus::Basic; 15];
    cb.cut_row_slots.extend_from_slice(&[10u32, 11, 12]);
    cb.state_at_capture.push(1.0);

    let stage_bases: Vec<Option<CapturedBasis>> = vec![Some(cb)];

    let stochastic = make_stochastic_context(n_stages);
    let config = SimulationConfig {
        n_scenarios,
        io_channel_capacity: 8,
        profile: Phase::Simulation.profile(),
    };
    let horizon = HorizonMode::Finite {
        num_stages: n_stages,
    };
    let initial_state = vec![50.0_f64];

    let solution = hydro_only_bus_solution(100.0, 30.0);
    let solver = MockSolver::always_ok(solution);
    let comm = StubComm { rank: 0, size: 1 };
    let entity_counts = entity_counts_1_hydro();
    let (tx, _rx) = mpsc::sync_channel(16);
    let hprod = hydro_productivities_1hydro(n_stages);
    let ec = zero_energy_conversion(1, n_stages);

    let mut workspaces = single_workspace(solver);
    let state_boxes = permissive_state_boxes(state.n_state, n_stages);
    let geometry = vec![hydro_only_bus_geometry(); n_stages];
    let stage_ctx_fixture = StageContextFixture::new(&templates, &state_boxes, &geometry);
    let result = cobre_sddp::simulate(
        &mut workspaces,
        &stage_ctx_fixture.ctx(),
        &fcf,
        &TrainingContext {
            node_graph: &cobre_sddp::test_support::chain_node_graph(&stochastic),
            horizon: &horizon,
            state: &state,
            cut_state_layouts: &all_enabled_cut_state_layouts(&state, n_stages),
            study_dims: &study_dims(),
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &stochastic,
            initial_state: &initial_state,
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            stages: &[],
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            lag_accum_seed: &[],
            lag_weight_seed: &[],
            dcs: None,
        },
        &config,
        SimulationOutputSpec {
            result_tx: &tx,
            hydro_cell_index: &cobre_sddp::test_support::identity_hydro_cell_index(256),
            block_hours_per_stage: &vec![vec![744.0]; 1],
            entity_counts: &entity_counts,
            generic_constraint_row_entries: &vec![Vec::new(); 1],
            pumping_consumption_mw_per_m3s: &[],
            contract_prices_per_stage: &vec![Vec::new(); 1],
            contract_slots: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities_per_stage: &hprod,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[0.0],
            event_sender: None,
            extended_delivery_anchors: &[],
            transit_seed_arcs: &[],
            past_defluences: &[],
            study_stage_dates: &[],
        },
        // fallback path (no frozen templates); reconstruction uses pool.active_cuts()
        None,
        &stage_bases,
        &comm,
        &Traversal::default(),
    );

    assert!(
        result.is_ok(),
        "simulate must succeed with CapturedBasis warm-start: {result:?}"
    );

    let solver = workspaces[0].solver.inner();
    assert_eq!(
        solver.solve_with_basis_count, 1,
        "warm-start solve must be called exactly once (1 scenario × 1 stage)"
    );
    assert_eq!(
        solver.solve_count, 0,
        "cold-start solve must not be called when a CapturedBasis is provided"
    );

    let recorded = solver
        .recorded_basis
        .as_ref()
        .expect("recorded_basis must be Some after a warm-start solve");

    // Under the active-only freeze model the LP carries one row per active cut;
    // inactive populated slots are absent, so the basis length is base_rows +
    // active_count.
    let active_count = fcf.pools[0].active_count();
    assert_eq!(
        recorded.row_status.len(),
        7 + active_count,
        "reconstructed basis row_status must have length base_row_count(7) + \
         active_count({active_count}) = {}, got {}",
        7 + active_count,
        recorded.row_status.len()
    );

    // Active cuts are iterated in slot order (10, 11, 12), so slot 10 lands at
    // active-cuts position 0 — LP row 7 (after the 7 base rows) — and the stored
    // statuses must reappear there verbatim.
    let preserved_offset = 7;
    assert_eq!(
        recorded.row_status[preserved_offset], CUT_STATUS_0,
        "slot 10 must preserve its stored cut status"
    );
    assert_eq!(
        recorded.row_status[preserved_offset + 1],
        CUT_STATUS_1,
        "slot 11 must preserve its stored cut status"
    );
    assert_eq!(
        recorded.row_status[preserved_offset + 2],
        CUT_STATUS_2,
        "slot 12 must preserve its stored cut status"
    );
}

/// When `stage_bases` is `&[]`
/// (cold-start), every LP solve must go through `solver.solve(None)` and
/// `solve(Some(&basis))` must never be called.
///
/// Uses `solve_count` and `solve_with_basis_count` split on `MockSolver`.
#[test]
fn simulate_with_empty_stage_bases_cold_starts() {
    let n_stages = 2;
    let n_scenarios = 3u32;
    let templates: Vec<StageTemplate> = (0..n_stages).map(|_| hydro_only_bus_template()).collect();

    let state = state_layout_for(1, 0);
    let fcf = FutureCostFunction::new(n_stages, 1, 1, 10, &vec![0; n_stages]);
    let stochastic = make_stochastic_context(n_stages);
    let config = SimulationConfig {
        n_scenarios,
        io_channel_capacity: 16,
        profile: Phase::Simulation.profile(),
    };
    let horizon = HorizonMode::Finite {
        num_stages: n_stages,
    };
    let initial_state = vec![50.0_f64];

    let solution = hydro_only_bus_solution(100.0, 30.0);
    let solver = MockSolver::always_ok(solution);
    let comm = StubComm { rank: 0, size: 1 };
    let entity_counts = entity_counts_1_hydro();
    let (tx, _rx) = mpsc::sync_channel(32);
    let hprod = hydro_productivities_1hydro(n_stages);
    let ec = zero_energy_conversion(1, n_stages);

    let mut workspaces = single_workspace(solver);
    let state_boxes = permissive_state_boxes(state.n_state, n_stages);
    let geometry = vec![hydro_only_bus_geometry(); n_stages];
    let stage_ctx_fixture = StageContextFixture::new(&templates, &state_boxes, &geometry);
    let result = cobre_sddp::simulate(
        &mut workspaces,
        &stage_ctx_fixture.ctx(),
        &fcf,
        &TrainingContext {
            node_graph: &cobre_sddp::test_support::chain_node_graph(&stochastic),
            horizon: &horizon,
            state: &state,
            cut_state_layouts: &all_enabled_cut_state_layouts(&state, n_stages),
            study_dims: &study_dims(),
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &stochastic,
            initial_state: &initial_state,
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            stages: &[],
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            lag_accum_seed: &[],
            lag_weight_seed: &[],
            dcs: None,
        },
        &config,
        SimulationOutputSpec {
            result_tx: &tx,
            hydro_cell_index: &cobre_sddp::test_support::identity_hydro_cell_index(256),
            block_hours_per_stage: &vec![vec![744.0]; 2],
            entity_counts: &entity_counts,
            generic_constraint_row_entries: &vec![Vec::new(); 2],
            pumping_consumption_mw_per_m3s: &[],
            contract_prices_per_stage: &vec![Vec::new(); 2],
            contract_slots: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities_per_stage: &hprod,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[0.0],
            event_sender: None,
            extended_delivery_anchors: &[],
            transit_seed_arcs: &[],
            past_defluences: &[],
            study_stage_dates: &[],
        },
        None,
        &[],
        &comm,
        &Traversal::default(),
    );

    assert!(
        result.is_ok(),
        "cold-start simulate must succeed: {result:?}"
    );

    let solver = workspaces[0].solver.inner();
    let expected_solves = n_scenarios as usize * n_stages;

    assert_eq!(
        solver.solve_with_basis_count, 0,
        "warm-start solve must not be called when stage_bases is empty; \
         got solve_with_basis_count={}",
        solver.solve_with_basis_count
    );
    assert_eq!(
        solver.solve_count, expected_solves,
        "cold-start solve must be called exactly n_scenarios({n_scenarios}) × \
         n_stages({n_stages}) = {expected_solves} times; got {}",
        solver.solve_count
    );
}

// ── Node-keyed simulation warm-basis cache (branching K-fan) ───────────

/// A 2-stage K-fan: a stage-0 root (node 0) branching 50/50 into two stage-1
/// leaves (node 1, node 2) — the minimal shape where node position and stage
/// diverge (node 2 sits at position 2 while its stage is 1), the exact
/// condition a stage-keyed warm-basis lookup gets wrong. Leaves share pool 1
/// (this module's leaf-sharing rule); the warm-basis cache is keyed by node
/// position regardless.
fn two_leaf_fan_node_graph() -> NodeGraph {
    let root_openings = NodeOpenings {
        source: OpeningSource::Generated,
        offset: 0,
        len: 1,
        q: 1.0,
    };
    NodeGraph {
        node_ids: vec![NodeId(0), NodeId(1), NodeId(2)].into(),
        nodes: vec![
            NodeRuntime {
                stage: StageIdx(0),
                pool_id: 0,
                openings: root_openings,
            },
            NodeRuntime {
                stage: StageIdx(1),
                pool_id: 1,
                openings: root_openings,
            },
            NodeRuntime {
                stage: StageIdx(1),
                pool_id: 1,
                openings: root_openings,
            },
        ]
        .into(),
        successors: vec![
            vec![
                NodeSuccessor {
                    child: NodePos(1),
                    probability: 0.5,
                },
                NodeSuccessor {
                    child: NodePos(2),
                    probability: 0.5,
                },
            ],
            Vec::new(),
            Vec::new(),
        ]
        .into(),
        n_pools: 2,
        pool_stage: vec![StageIdx(0), StageIdx(1)],
    }
}

/// A `CapturedBasis` warm-startable against a template shaped `num_cols=4`,
/// `num_rows=2`, 0 cut rows: 2 BASIC columns + 0 BASIC rows
/// satisfies `enforce_basic_count_invariant`'s `total_basic == num_row`
/// requirement, mirroring `run_stage_solve_warm_start_frozen_path_succeeds`'s
/// fixture.
fn warm_basis_for_node(node_id: NodeId) -> CapturedBasis {
    let mut cb = CapturedBasis::new(4, 2, 2, 0, 1, node_id);
    cb.basis.col_status[0] = BasisStatus::Basic;
    cb.basis.col_status[1] = BasisStatus::Basic;
    cb.state_at_capture.push(0.0);
    cb
}

/// Acceptance: a branching simulation warm-starts from the VISITED node's
/// own basis, not whichever node's basis happens to land at that stage index.
/// The pre-fix, stage-keyed lookup (`stage_bases.get(t)`) would resolve node
/// 2's stage-1 solve to `node_bases[1]` — leaf A's basis, whose `node_id`
/// mismatches — and `run_stage_solve`'s node-tag guard would silently reject
/// it as cold. A node-keyed lookup warm-starts every stage-1 solve regardless
/// of which leaf it visits.
#[test]
fn simulate_branching_k_fan_warm_starts_from_visited_node_basis() {
    let node_graph = two_leaf_fan_node_graph();
    let n_stages = 2;
    let n_scenarios = 8u32;

    // Self-checked precondition (testing.md): the pinned scenario range must
    // resolve visits to BOTH leaves, or this test cannot distinguish a
    // per-node lookup from the stage-keyed bug it exists to catch.
    let weights = [0.5_f64, 0.5];
    let leaf_for_scenario: Vec<usize> = (0..n_scenarios)
        .map(|s| select_transition_child(0, s, 0, weights.iter().copied()))
        .collect();
    assert!(
        leaf_for_scenario.contains(&0),
        "pinned scenario range must resolve at least one visit to leaf A (node 1)"
    );
    assert!(
        leaf_for_scenario.contains(&1),
        "pinned scenario range must resolve at least one visit to leaf B (node 2) — \
         otherwise this test cannot exercise the node-vs-stage divergence"
    );

    let templates: Vec<StageTemplate> = (0..n_stages).map(|_| hydro_only_bus_template()).collect();
    let state = state_layout_for(1, 0);
    let fcf = FutureCostFunction::new(node_graph.n_pools, 1, 1, 10, &vec![0; node_graph.n_pools]);
    let stochastic = make_stochastic_context(n_stages);
    let config = SimulationConfig {
        n_scenarios,
        io_channel_capacity: 32,
        profile: Phase::Simulation.profile(),
    };
    let horizon = HorizonMode::Finite {
        num_stages: n_stages,
    };
    let initial_state = vec![50.0_f64];

    let solution = hydro_only_bus_solution(100.0, 30.0);
    let solver = MockSolver::always_ok(solution);
    let comm = StubComm { rank: 0, size: 1 };
    let entity_counts = entity_counts_1_hydro();
    let (tx, _rx) = mpsc::sync_channel(32);
    let hprod = hydro_productivities_1hydro(n_stages);
    let ec = zero_energy_conversion(1, n_stages);

    // node_bases[0] (root) carries no basis; nodes 1 and 2 each carry their
    // OWN basis, tagged with their OWN declared node_id.
    let node_bases: Vec<Option<CapturedBasis>> = vec![
        None,
        Some(warm_basis_for_node(NodeId(1))),
        Some(warm_basis_for_node(NodeId(2))),
    ];

    let mut workspaces = single_workspace(solver);
    let state_boxes = permissive_state_boxes(state.n_state, n_stages);
    let geometry = vec![hydro_only_bus_geometry(); n_stages];
    let stage_ctx_fixture = StageContextFixture::new(&templates, &state_boxes, &geometry);
    let result = cobre_sddp::simulate(
        &mut workspaces,
        &stage_ctx_fixture.ctx(),
        &fcf,
        &TrainingContext {
            node_graph: &node_graph,
            horizon: &horizon,
            state: &state,
            cut_state_layouts: &all_enabled_cut_state_layouts(&state, node_graph.n_pools),
            study_dims: &study_dims(),
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &stochastic,
            initial_state: &initial_state,
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            stages: &[],
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            lag_accum_seed: &[],
            lag_weight_seed: &[],
            dcs: None,
        },
        &config,
        SimulationOutputSpec {
            result_tx: &tx,
            hydro_cell_index: &cobre_sddp::test_support::identity_hydro_cell_index(256),
            block_hours_per_stage: &vec![vec![744.0]; 2],
            entity_counts: &entity_counts,
            generic_constraint_row_entries: &vec![Vec::new(); 2],
            pumping_consumption_mw_per_m3s: &[],
            contract_prices_per_stage: &vec![Vec::new(); 2],
            contract_slots: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities_per_stage: &hprod,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[0.0],
            event_sender: None,
            extended_delivery_anchors: &[],
            transit_seed_arcs: &[],
            past_defluences: &[],
            study_stage_dates: &[],
        },
        None,
        &node_bases,
        &comm,
        &Traversal::default(),
    );

    assert!(
        result.is_ok(),
        "branching K-fan simulate must succeed: {result:?}"
    );

    let solver = workspaces[0].solver.inner();
    let n_scenarios = n_scenarios as usize;
    assert_eq!(
        solver.solve_with_basis_count, n_scenarios,
        "every stage-1 solve must warm-start from its OWN visited node's basis, \
         regardless of which leaf it visits; got solve_with_basis_count={} \
         (expected {n_scenarios} — one per scenario)",
        solver.solve_with_basis_count
    );
    assert_eq!(
        solver.solve_count, n_scenarios,
        "every stage-0 (root) solve must be cold (no basis provided); got \
         solve_count={}",
        solver.solve_count
    );
}
