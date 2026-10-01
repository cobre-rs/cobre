#![allow(clippy::unwrap_used, clippy::panic, clippy::too_many_lines)]

use std::collections::HashMap;
use std::sync::mpsc;

use cobre_comm::{CommData, CommError, Communicator, ReduceOp};
use cobre_core::WorkerPhaseTimings;
use cobre_core::scenario::SamplingScheme;
use cobre_solver::{
    Basis, LpSolution, ProfiledSolver, RowBatch, SolverError, SolverInterface, SolverStatistics,
    StageTemplate,
};
use cobre_stochastic::StochasticContext;

use super::SimulationOutputSpec;
use crate::{
    context::{StageContext, TrainingContext},
    cut::FutureCostFunction,
    energy_conversion::EnergyConversionSet,
    horizon_mode::HorizonMode,
    inflow_method::InflowNonNegativityMethod,
    lp::builder::PatchBuffer,
    setup::node_graph::Traversal,
    simulation::{
        config::SimulationConfig,
        error::SimulationError,
        extraction::EntityCounts,
        state::{SimulationInputs, SimulationState},
    },
    solve::solver_phase::Phase,
    test_support::{self, StageContextFixture, permissive_state_boxes},
    workspace::{BackwardAccumulators, CapturedBasis, ScratchBuffers, SolverWorkspace},
};

// A params struct would churn every call site; the wide arity is deliberate.
#[allow(clippy::too_many_arguments)]
fn run_simulate<S, C: cobre_comm::Communicator>(
    workspaces: &mut [SolverWorkspace<S>],
    ctx: &StageContext<'_>,
    fcf: &FutureCostFunction,
    training_ctx: &TrainingContext<'_>,
    config: &SimulationConfig,
    output: SimulationOutputSpec<'_>,
    frozen_templates: Option<&[cobre_solver::StageTemplate]>,
    node_bases: &[Option<CapturedBasis>],
    comm: &C,
    traversal: &Traversal,
) -> Result<super::SimulationRunResult, SimulationError>
where
    S: cobre_solver::SolverInterface<Profile = cobre_solver::ActiveProfile> + Send,
{
    let num_stages = training_ctx.horizon.num_stages();
    let mut state = SimulationState::new(num_stages);
    let mut inputs = SimulationInputs {
        workspaces,
        ctx,
        fcf,
        training_ctx,
        config,
        output,
        frozen_templates,
        node_bases,
        comm,
        traversal,
    };
    state.run(&mut inputs)
}

/// Identical to [`run_simulate`], but installs `profile` via `set_profile`
/// before `run()` — exercises the resolved-profile threading mechanism.
// A params struct would churn every call site; the wide arity is deliberate.
#[allow(clippy::too_many_arguments)]
fn run_simulate_with_profile<S, C: cobre_comm::Communicator>(
    workspaces: &mut [SolverWorkspace<S>],
    ctx: &StageContext<'_>,
    fcf: &FutureCostFunction,
    training_ctx: &TrainingContext<'_>,
    config: &SimulationConfig,
    output: SimulationOutputSpec<'_>,
    frozen_templates: Option<&[cobre_solver::StageTemplate]>,
    node_bases: &[Option<CapturedBasis>],
    comm: &C,
    profile: cobre_solver::ActiveProfile,
    traversal: &Traversal,
) -> Result<super::SimulationRunResult, SimulationError>
where
    S: cobre_solver::SolverInterface<Profile = cobre_solver::ActiveProfile> + Send,
{
    let num_stages = training_ctx.horizon.num_stages();
    let mut state = SimulationState::new(num_stages);
    state.set_profile(profile);
    let mut inputs = SimulationInputs {
        workspaces,
        ctx,
        fcf,
        training_ctx,
        config,
        output,
        frozen_templates,
        node_bases,
        comm,
        traversal,
    };
    state.run(&mut inputs)
}

// ── Stub communicator ────────────────────────────────────────────────────

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

/// Mock solver returning a fixed `LpSolution` on every solve; `infeasible_at`
/// injects `SolverError::Infeasible` at that 0-based solve index, counted across
/// both cold- and warm-start calls.
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
    reconstruction_counter: u32,
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
            load_count: 0,
            add_rows_count: 0,
            solve_count: 0,
            solve_with_basis_count: 0,
            recorded_basis: None,
            reconstruction_counter: 0,
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
        crate::test_support::fill_consistent_basis(out);
    }
    fn record_reconstruction_stats(&mut self) {
        self.reconstruction_counter += 1;
    }
    fn statistics(&self) -> SolverStatistics {
        SolverStatistics::default()
    }
    fn statistics_into(&self, out: &mut SolverStatistics) {
        out.copy_from(&SolverStatistics::default());
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
    use cobre_core::entities::hydro::{Hydro, HydroGenerationModel, HydroPenalties};
    use cobre_core::scenario::{
        CorrelationEntity, CorrelationGroup, CorrelationModel, CorrelationProfile, InflowModel,
    };
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    };
    use cobre_core::{Bus, DeficitSegment, EntityId, SystemBuilder};
    use cobre_stochastic::context::{ClassSchemes, OpeningTreeInputs, build_stochastic_context};

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

/// Per-stage hydro productivities: `n_stages` vecs of a single `1.0`.
fn hydro_productivities_1hydro(n_stages: usize) -> Vec<Vec<f64>> {
    vec![vec![1.0]; n_stages]
}

/// Build a zero-valued [`EnergyConversionSet`] for tests
/// that do not assert on energy fields.
fn zero_energy_conversion(n_hydros: usize, n_stages: usize) -> EnergyConversionSet {
    use crate::energy_conversion::{EnergyConversion, EnergyConversionSet};
    let zero_ec = EnergyConversion {
        equivalent_productivity_mw_per_m3s: 0.0,
        reference_volume_hm3: 0.0,
        reference_outflow_m3s: 0.0,
    };
    EnergyConversionSet::new(
        vec![vec![zero_ec; n_stages]; n_hydros],
        vec![vec![0.0_f64; n_stages]; n_hydros],
        &crate::test_support::minimal_hydros(n_hydros),
        n_stages,
    )
}

/// Like `single_workspace`, but sizes the patch buffer and reserves
/// `load_rhs_buf` for `n_load_buses` stochastic load buses.
fn single_workspace_with_load_buses(
    solver: MockSolver,
    n_load_buses: usize,
) -> Vec<SolverWorkspace<MockSolver>> {
    let load_bus_indices: Vec<usize> = (0..n_load_buses).collect();
    let geometry = test_support::equipment_free_geometry(&[1]);
    vec![SolverWorkspace {
        rank: 0,
        worker_id: 0,
        solver: ProfiledSolver::new(solver),
        patch_buf: PatchBuffer::new(
            &test_support::state_layout(1, 0),
            &load_bus_indices,
            &geometry,
        ),
        current_state: Vec::with_capacity(1),
        scratch: ScratchBuffers {
            inflow_m3s_buf: Vec::new(),
            lag_matrix_buf: Vec::new(),
            par_inflow_buf: Vec::new(),
            eta_floor_buf: Vec::new(),
            zero_targets_buf: Vec::new(),
            ncs_col_upper_buf: Vec::new(),
            ncs_col_lower_buf: Vec::new(),
            ncs_col_indices_buf: Vec::new(),
            ncs_col_lower_active_buf: Vec::new(),
            ncs_col_upper_active_buf: Vec::new(),
            last_ncs_col_start: usize::MAX,
            ncs_col_upper_extract_buf: Vec::new(),
            load_rhs_buf: Vec::with_capacity(n_load_buses),
            row_lower_buf: Vec::new(),
            z_inflow_rhs_buf: Vec::new(),
            effective_eta_buf: Vec::new(),
            unscaled_primal: Vec::new(),
            unscaled_dual: Vec::new(),
            lag_accumulator: vec![],
            lag_weight_accum: vec![],
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
    }]
}

/// Wrap a `MockSolver` in a single-workspace slice for `simulate()` calls.
///
/// All tests use a single workspace (serial execution) so that existing
/// assertions about scenario ordering and call counts remain valid.
fn single_workspace(solver: MockSolver) -> Vec<SolverWorkspace<MockSolver>> {
    vec![SolverWorkspace {
        rank: 0,
        worker_id: 0,
        solver: ProfiledSolver::new(solver),
        patch_buf: PatchBuffer::new(&test_support::state_layout(1, 0), &[], &[]), // N=1, L=0
        current_state: Vec::with_capacity(1),
        scratch: ScratchBuffers {
            inflow_m3s_buf: Vec::new(),
            lag_matrix_buf: Vec::new(),
            par_inflow_buf: Vec::new(),
            eta_floor_buf: Vec::new(),
            zero_targets_buf: Vec::new(),
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
            lag_accumulator: vec![],
            lag_weight_accum: vec![],
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
    }]
}

// ── Load noise simulation tests ───────────────────────────────────────────

/// Build a stochastic context with 1 hydro and 1 stochastic load bus for
/// simulation load noise tests.
///
/// The context has 1 stage with branching factor 3.  The load model uses
/// `bus_id=1` (distinct from the hydro bus at `bus_id=0`), so the noise
/// vector has dimension 2: `[inflow_eta, load_eta]`.
fn make_stochastic_context_1_hydro_1_load_bus_sim(mean_mw: f64, std_mw: f64) -> StochasticContext {
    use std::collections::BTreeMap;

    use chrono::NaiveDate;
    use cobre_core::entities::hydro::{Hydro, HydroGenerationModel, HydroPenalties};
    use cobre_core::scenario::{CorrelationModel, InflowModel, LoadModel};
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    };
    use cobre_core::{Bus, DeficitSegment, EntityId, SystemBuilder};
    use cobre_stochastic::context::{ClassSchemes, OpeningTreeInputs, build_stochastic_context};

    let bus0 = Bus {
        id: EntityId(0),
        name: "B0".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 1000.0,
        }],
        excess_cost: 0.0,
    };
    let bus1 = Bus {
        id: EntityId(1),
        name: "B1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 1000.0,
        }],
        excess_cost: 0.0,
    };
    let mut hydro = Hydro {
        unit_groups: Vec::new(),
        id: EntityId(10),
        name: "H10".to_string(),
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
            branching_factor: 3,
            noise_method: NoiseMethod::Saa,
        },
    };
    let inflow_model = InflowModel {
        hydro_id: EntityId(10),
        stage_id: 0,
        mean_m3s: 100.0,
        std_m3s: 20.0,
        ar_coefficients: vec![],
        residual_std_ratio: 1.0,
        annual: None,
    };
    let load_model = LoadModel {
        bus_id: EntityId(1),
        stage_id: 0,
        mean_mw,
        std_mw,
    };
    let correlation = CorrelationModel {
        method: "spectral".to_string(),
        profiles: BTreeMap::new(),
        schedule: vec![],
    };
    let system = SystemBuilder::new()
        .buses(vec![bus0, bus1])
        .hydros(vec![hydro])
        .stages(vec![stage])
        .inflow_models(vec![inflow_model])
        .load_models(vec![load_model])
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

/// When a simulation has 1 stochastic load bus (mean=300, std=30),
/// verify that `load_rhs_buf` is populated with a positive value.
#[test]
fn simulation_load_patches_applied() {
    let n_stages = 1;
    let templates = vec![test_support::hydro_only_bus_template()];

    let n_load_buses = 1usize;
    let stochastic = make_stochastic_context_1_hydro_1_load_bus_sim(300.0, 30.0);
    let fcf = FutureCostFunction::new(n_stages, 1, 1, 10, &vec![0; n_stages]);
    let config = SimulationConfig {
        n_scenarios: 1,
        io_channel_capacity: 4,
        profile: Phase::Simulation.profile(),
    };
    let state = test_support::state_layout(1, 0);
    let horizon = HorizonMode::Finite {
        num_stages: n_stages,
    };
    let initial_state = vec![50.0_f64];

    let solution = test_support::hydro_only_bus_solution(100.0, 30.0);
    let solver = MockSolver::always_ok(solution);
    let comm = StubComm { rank: 0, size: 1 };
    let entity_counts = entity_counts_1_hydro();

    let (tx, _rx) = mpsc::sync_channel(4);

    let mut workspaces = single_workspace_with_load_buses(solver, n_load_buses);

    // load_bus_indices=[0] (bus position 0 in the block layout).
    let load_bus_indices = vec![0usize];
    let geometry_per_stage = vec![test_support::hydro_only_bus_geometry()];

    let hprod = hydro_productivities_1hydro(n_stages);
    let ec = zero_energy_conversion(1, n_stages);
    let state_boxes = permissive_state_boxes(state.n_state, templates.len());
    run_simulate(
        &mut workspaces,
        &StageContextFixture::new(&templates, &state_boxes, &geometry_per_stage)
            .load_bus_indices(&load_bus_indices)
            .ctx(),
        &fcf,
        &TrainingContext {
            node_graph: &crate::test_support::chain_node_graph(&stochastic),
            horizon: &horizon,
            state: &state,
            cut_state_layouts: &test_support::all_enabled_cut_state_layouts(&state, n_stages),
            study_dims: &test_support::study_dims(),
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
            hydro_cell_index: &test_support::identity_hydro_cell_index(256),
            block_hours_per_stage: &[vec![1.0]],
            entity_counts: &entity_counts,
            generic_constraint_row_entries: &[Vec::new()],
            pumping_consumption_mw_per_m3s: &[],
            contract_prices_per_stage: &[Vec::new()],
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
        workspaces[0].scratch.load_rhs_buf.len(),
        n_load_buses,
        "load_rhs_buf must have 1 entry (1 load bus x 1 block)"
    );
    assert!(
        workspaces[0].scratch.load_rhs_buf[0] > 0.0,
        "realization must be positive with mean=300, std=30: got {}",
        workspaces[0].scratch.load_rhs_buf[0]
    );

    // Verify the formula: d = max(0, mean + std * eta) * factor.
    // The exact eta drawn from the opening tree depends on the seed, but
    // we can verify formula consistency by back-computing eta from the
    // observed realization (d > 0 implies eta = (d - mean) / std).
    let d_observed = workspaces[0].scratch.load_rhs_buf[0];
    let mean_mw_val = 300.0_f64;
    let std_mw_val = 30.0_f64;
    assert!(
        d_observed != mean_mw_val,
        "realization must differ from template mean (noise was applied)"
    );
    let eta_back = (d_observed - mean_mw_val) / std_mw_val;
    let recomputed = (mean_mw_val + std_mw_val * eta_back).max(0.0);
    assert!(
        (d_observed - recomputed).abs() < 1e-10,
        "formula consistency: d={d_observed}, eta_back={eta_back}, recomputed={recomputed}"
    );

    let load_start = 0; // load patches start at slot 0
    assert_eq!(
        workspaces[0].patch_buf.lower[load_start], workspaces[0].scratch.load_rhs_buf[0],
        "patch_buf lower at load slot must equal load_rhs_buf[0]"
    );
    assert_eq!(
        workspaces[0].patch_buf.upper[load_start], workspaces[0].scratch.load_rhs_buf[0],
        "patch_buf upper at load slot must equal load_rhs_buf[0] (equality constraint)"
    );

    // row_lower_buf[2] is the load balance row (row index 2).
    assert!(
        !workspaces[0].scratch.row_lower_buf.is_empty(),
        "row_lower_buf must be populated for extraction"
    );
    assert_eq!(
        workspaces[0].scratch.row_lower_buf[2], d_observed,
        "extraction row_lower_buf must contain stochastic load, not template mean"
    );
    assert!(
        (workspaces[0].scratch.row_lower_buf[2] - mean_mw_val).abs() > 1e-6,
        "extracted load_mw must differ from template mean {mean_mw_val}: got {}",
        workspaces[0].scratch.row_lower_buf[2]
    );
}

/// when `n_load_buses == 0`,
/// `load_rhs_buf` remains empty and `forward_patch_count` equals the number of
/// z-inflow patches filled.
///
/// With one hydro, `fill_z_inflow_patches` fills exactly one z-inflow patch, so
/// `forward_patch_count` equals `state.hydro_count`.
#[test]
fn simulation_no_load_buses_unchanged() {
    let n_stages = 1;
    let templates = vec![test_support::hydro_only_bus_template()];

    let stochastic = make_stochastic_context(n_stages);
    let state = test_support::state_layout(1, 0);
    let fcf = FutureCostFunction::new(n_stages, 1, 1, 10, &vec![0; n_stages]);
    let config = SimulationConfig {
        n_scenarios: 1,
        io_channel_capacity: 4,
        profile: Phase::Simulation.profile(),
    };
    let horizon = HorizonMode::Finite {
        num_stages: n_stages,
    };
    let initial_state = vec![50.0_f64];

    let solution = test_support::hydro_only_bus_solution(100.0, 30.0);
    let solver = MockSolver::always_ok(solution);
    let comm = StubComm { rank: 0, size: 1 };
    let entity_counts = entity_counts_1_hydro();

    let (tx, _rx) = mpsc::sync_channel(4);

    let hprod = hydro_productivities_1hydro(n_stages);
    let ec = zero_energy_conversion(1, n_stages);
    let mut workspaces = single_workspace(solver);

    let state_boxes = permissive_state_boxes(state.n_state, templates.len());
    run_simulate(
        &mut workspaces,
        &StageContextFixture::new(
            &templates,
            &state_boxes,
            &[test_support::hydro_only_bus_geometry()],
        )
        .ctx(),
        &fcf,
        &TrainingContext {
            node_graph: &crate::test_support::chain_node_graph(&stochastic),
            horizon: &horizon,
            state: &state,
            cut_state_layouts: &test_support::all_enabled_cut_state_layouts(&state, n_stages),
            study_dims: &test_support::study_dims(),
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
            hydro_cell_index: &test_support::identity_hydro_cell_index(256),
            block_hours_per_stage: &[vec![1.0]],
            entity_counts: &entity_counts,
            generic_constraint_row_entries: &[Vec::new()],
            pumping_consumption_mw_per_m3s: &[],
            contract_prices_per_stage: &[Vec::new()],
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

    assert!(
        workspaces[0].scratch.load_rhs_buf.is_empty(),
        "load_rhs_buf must be empty when n_load_buses=0"
    );
    assert_eq!(
        workspaces[0].patch_buf.forward_patch_count(),
        state.hydro_count,
        "forward_patch_count must equal the z-inflow patches filled, got {}",
        workspaces[0].patch_buf.forward_patch_count()
    );
}

/// A profile installed via `set_profile` before `run()` is the one
/// `ProfiledSolver::current_profile()` reports afterwards.
#[test]
fn simulation_state_set_profile_reaches_current_profile_after_run() {
    let n_stages = 1;
    let templates = vec![test_support::hydro_only_bus_template()];

    let stochastic = make_stochastic_context(n_stages);
    let state = test_support::state_layout(1, 0);
    let fcf = FutureCostFunction::new(n_stages, 1, 1, 10, &vec![0; n_stages]);
    let config = SimulationConfig {
        n_scenarios: 1,
        io_channel_capacity: 4,
        profile: Phase::Simulation.profile(),
    };
    let horizon = HorizonMode::Finite {
        num_stages: n_stages,
    };
    let initial_state = vec![50.0_f64];

    let solution = test_support::hydro_only_bus_solution(100.0, 30.0);
    let solver = MockSolver::always_ok(solution);
    let comm = StubComm { rank: 0, size: 1 };
    let entity_counts = entity_counts_1_hydro();

    let (tx, _rx) = mpsc::sync_channel(4);

    let hprod = hydro_productivities_1hydro(n_stages);
    let ec = zero_energy_conversion(1, n_stages);
    let mut workspaces = single_workspace(solver);

    let resolved =
        Phase::Simulation.resolve_profile(Some(&cobre_io::config::PhaseSolverProfileConfig {
            dual_edge_weight: None,
            scale: Some(cobre_io::config::ScaleStrategy::SolverScaling),
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

    let state_boxes = permissive_state_boxes(state.n_state, templates.len());
    run_simulate_with_profile(
        &mut workspaces,
        &StageContextFixture::new(
            &templates,
            &state_boxes,
            &[test_support::hydro_only_bus_geometry()],
        )
        .ctx(),
        &fcf,
        &TrainingContext {
            node_graph: &crate::test_support::chain_node_graph(&stochastic),
            horizon: &horizon,
            state: &state,
            cut_state_layouts: &test_support::all_enabled_cut_state_layouts(&state, n_stages),
            study_dims: &test_support::study_dims(),
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
            hydro_cell_index: &test_support::identity_hydro_cell_index(256),
            block_hours_per_stage: &[vec![1.0]],
            entity_counts: &entity_counts,
            generic_constraint_row_entries: &[Vec::new()],
            pumping_consumption_mw_per_m3s: &[],
            contract_prices_per_stage: &[Vec::new()],
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
        resolved,
        &Traversal::default(),
    )
    .unwrap();

    assert_eq!(
        workspaces[0].solver.current_profile(),
        &resolved,
        "the profile installed via set_profile must be the one stored on \
         current_profile after run()"
    );
}

/// When load noise is present,
/// `z_inflow_rhs_buf` still contains only inflow values (not contaminated by
/// load noise).
///
/// `z_inflow_rhs_buf` contains inflow realizations for the `n_hydros` hydros.
/// After simulate runs with `n_hydros=1` and `n_load_buses=1`,
/// `z_inflow_rhs_buf` must have exactly 1 entry (inflow), while
/// `load_rhs_buf` has 1 entry (load).  The two buffers must not overlap.
#[test]
fn simulation_inflow_extraction_unaffected() {
    let n_stages = 1;
    let templates = vec![test_support::hydro_only_bus_template()];

    let n_load_buses = 1usize;
    let stochastic = make_stochastic_context_1_hydro_1_load_bus_sim(300.0, 30.0);
    let state = test_support::state_layout(1, 0);
    let fcf = FutureCostFunction::new(n_stages, 1, 1, 10, &vec![0; n_stages]);
    let config = SimulationConfig {
        n_scenarios: 1,
        io_channel_capacity: 4,
        profile: Phase::Simulation.profile(),
    };
    let horizon = HorizonMode::Finite {
        num_stages: n_stages,
    };
    let initial_state = vec![50.0_f64];

    let solution = test_support::hydro_only_bus_solution(100.0, 30.0);
    let solver = MockSolver::always_ok(solution);
    let comm = StubComm { rank: 0, size: 1 };
    let entity_counts = entity_counts_1_hydro();

    let (tx, _rx) = mpsc::sync_channel(4);

    let mut workspaces = single_workspace_with_load_buses(solver, n_load_buses);

    let load_bus_indices = vec![0usize];
    let geometry_per_stage = vec![test_support::hydro_only_bus_geometry()];

    let hprod = hydro_productivities_1hydro(n_stages);
    let ec = zero_energy_conversion(1, n_stages);
    let state_boxes = permissive_state_boxes(state.n_state, templates.len());
    run_simulate(
        &mut workspaces,
        &StageContextFixture::new(&templates, &state_boxes, &geometry_per_stage)
            .load_bus_indices(&load_bus_indices)
            .ctx(),
        &fcf,
        &TrainingContext {
            node_graph: &crate::test_support::chain_node_graph(&stochastic),
            horizon: &horizon,
            state: &state,
            cut_state_layouts: &test_support::all_enabled_cut_state_layouts(&state, n_stages),
            study_dims: &test_support::study_dims(),
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
            hydro_cell_index: &test_support::identity_hydro_cell_index(256),
            block_hours_per_stage: &[vec![1.0]],
            entity_counts: &entity_counts,
            generic_constraint_row_entries: &[Vec::new()],
            pumping_consumption_mw_per_m3s: &[],
            contract_prices_per_stage: &[Vec::new()],
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
        workspaces[0].scratch.z_inflow_rhs_buf.len(),
        1,
        "z_inflow_rhs_buf must have 1 entry (1 hydro), not contaminated by load noise: len={}",
        workspaces[0].scratch.z_inflow_rhs_buf.len()
    );
    // The inflow noise must be a reasonable value near the model's mean_m3s=100.0,
    // sigma=20.0: z_inflow_rhs_buf[0] = 100.0 + 20.0 * eta_inflow.
    assert!(
        workspaces[0].scratch.z_inflow_rhs_buf[0] > 0.0
            && workspaces[0].scratch.z_inflow_rhs_buf[0] < 200.0,
        "z_inflow_rhs_buf[0] must be a reasonable inflow value near 100.0, got {}",
        workspaces[0].scratch.z_inflow_rhs_buf[0]
    );
    assert_eq!(
        workspaces[0].scratch.load_rhs_buf.len(),
        n_load_buses,
        "load_rhs_buf must have 1 entry alongside z_inflow_rhs_buf"
    );
}

// ── Inflow truncation unit tests ──────────────────────────────────────────

/// Build a `StochasticContext` for 1 hydro, 1 stage with configurable
/// `mean_m3s` and `std_m3s`.  Used by the truncation tests below.
fn make_stochastic_1h_1s(mean_m3s: f64, std_m3s: f64) -> StochasticContext {
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
    use cobre_stochastic::context::{ClassSchemes, OpeningTreeInputs, build_stochastic_context};

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
            branching_factor: 3,
            noise_method: NoiseMethod::Saa,
        },
    };
    let inflow_model = InflowModel {
        hydro_id: EntityId(1),
        stage_id: 0,
        mean_m3s,
        std_m3s,
        ar_coefficients: vec![],
        residual_std_ratio: 1.0,
        annual: None,
    };
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
        .stages(vec![stage])
        .inflow_models(vec![inflow_model])
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

/// Build a workspace with `zero_targets_buf` pre-populated to `hydro_count` zeros.
///
/// The standard `single_workspace` helper leaves `zero_targets_buf` empty because
/// existing tests do not reach the truncation branch.  The truncation tests use
/// 1 hydro and must have `zero_targets_buf[..1]` accessible, so they use this
/// helper instead.
fn single_workspace_with_hydros(
    solver: MockSolver,
    hydro_count: usize,
) -> Vec<SolverWorkspace<MockSolver>> {
    vec![SolverWorkspace {
        rank: 0,
        worker_id: 0,
        solver: ProfiledSolver::new(solver),
        patch_buf: PatchBuffer::new(&test_support::state_layout(hydro_count, 0), &[], &[]),
        current_state: Vec::with_capacity(hydro_count),
        scratch: ScratchBuffers {
            inflow_m3s_buf: Vec::new(),
            lag_matrix_buf: Vec::new(),
            par_inflow_buf: Vec::new(),
            eta_floor_buf: Vec::new(),
            zero_targets_buf: vec![0.0_f64; hydro_count],
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
            lag_accumulator: vec![],
            lag_weight_accum: vec![],
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
    }]
}

/// Truncation clamps negative inflow noise in the simulation pipeline.
///
/// Set `mean_m3s = -1000.0` and `std_m3s = 1.0` so that the deterministic
/// PAR base alone would produce a hugely negative inflow for any sample.
///
/// With `InflowNonNegativityMethod::Truncation` active, the simulation pipeline
/// must clamp `eta` to the floor that produces zero inflow.  As a result,
/// `z_inflow_rhs_buf[0] = mean + sigma * eta_clamped >= 0.0` for all
/// scenarios processed.
///
/// `eta_floor` = (0 - mean) / sigma = 1000. So `z_inflow_rhs_buf`\[0\] =
/// -1000 + 1\*1000 = 0.
#[test]
fn simulation_truncation_clamps_negative_inflow_noise() {
    let mean_m3s = -1000.0_f64;
    let sigma = 1.0_f64;

    let n_stages = 1;
    let stochastic = make_stochastic_1h_1s(mean_m3s, sigma);
    let templates = vec![test_support::hydro_only_bus_template()];

    let state = test_support::state_layout(1, 0);
    let fcf = FutureCostFunction::new(n_stages, state.n_state, 1, 10, &vec![0; n_stages]);
    let config = SimulationConfig {
        n_scenarios: 4,
        io_channel_capacity: 16,
        profile: Phase::Simulation.profile(),
    };
    let horizon = HorizonMode::Finite {
        num_stages: n_stages,
    };
    let initial_state = vec![0.0_f64];

    let solution = test_support::hydro_only_bus_solution(0.0, 0.0);
    let solver = MockSolver::always_ok(solution);
    let comm = StubComm { rank: 0, size: 1 };
    let entity_counts = entity_counts_1_hydro();

    let (tx, _rx) = mpsc::sync_channel(16);

    let hprod = hydro_productivities_1hydro(n_stages);
    let ec = zero_energy_conversion(1, n_stages);
    let mut workspaces = single_workspace_with_hydros(solver, 1);
    let state_boxes = permissive_state_boxes(state.n_state, templates.len());
    run_simulate(
        &mut workspaces,
        &StageContextFixture::new(
            &templates,
            &state_boxes,
            &[test_support::hydro_only_bus_geometry()],
        )
        .ctx(),
        &fcf,
        &TrainingContext {
            node_graph: &crate::test_support::chain_node_graph(&stochastic),
            horizon: &horizon,
            state: &state,
            cut_state_layouts: &test_support::all_enabled_cut_state_layouts(&state, n_stages),
            study_dims: &test_support::study_dims(),
            inflow_method: &InflowNonNegativityMethod::Truncation,
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
            hydro_cell_index: &test_support::identity_hydro_cell_index(256),
            block_hours_per_stage: &[vec![1.0]],
            entity_counts: &entity_counts,
            generic_constraint_row_entries: &[Vec::new()],
            pumping_consumption_mw_per_m3s: &[],
            contract_prices_per_stage: &[Vec::new()],
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

    // z_inflow_rhs_buf holds the last scenario-stage's value; truncation clamps
    // it >= 0.
    assert_eq!(
        workspaces[0].scratch.z_inflow_rhs_buf.len(),
        1,
        "z_inflow_rhs_buf must have exactly 1 entry for 1 hydro"
    );
    assert!(
        workspaces[0].scratch.z_inflow_rhs_buf[0] >= 0.0,
        "after truncation, z_inflow_rhs_buf[0] must be >= 0 (inflow cannot be negative), got {}",
        workspaces[0].scratch.z_inflow_rhs_buf[0]
    );
}

/// `InflowNonNegativityMethod::None` in the simulation pipeline produces
/// raw (potentially negative) noise values.
///
/// With `mean_m3s = -1000.0` and `std_m3s = 1.0`, the PAR inflow is always
/// deeply negative.  The `None` path must NOT clamp eta, so the z-inflow
/// buffer value must be negative (`mean` + `sigma` \* `raw_eta` << 0).
#[test]
fn simulation_none_method_produces_raw_negative_noise() {
    let mean_m3s = -1000.0_f64;
    let sigma = 1.0_f64;

    let n_stages = 1;
    let stochastic = make_stochastic_1h_1s(mean_m3s, sigma);
    let templates = vec![test_support::hydro_only_bus_template()];

    let state = test_support::state_layout(1, 0);
    let fcf = FutureCostFunction::new(n_stages, state.n_state, 1, 10, &vec![0; n_stages]);
    let config = SimulationConfig {
        n_scenarios: 4,
        io_channel_capacity: 16,
        profile: Phase::Simulation.profile(),
    };
    let horizon = HorizonMode::Finite {
        num_stages: n_stages,
    };
    let initial_state = vec![0.0_f64];

    let solution = test_support::hydro_only_bus_solution(0.0, 0.0);
    let solver = MockSolver::always_ok(solution);
    let comm = StubComm { rank: 0, size: 1 };
    let entity_counts = entity_counts_1_hydro();

    let (tx, _rx) = mpsc::sync_channel(16);

    let hprod = hydro_productivities_1hydro(n_stages);
    let ec = zero_energy_conversion(1, n_stages);
    let mut workspaces = single_workspace_with_hydros(solver, 1);
    let state_boxes = permissive_state_boxes(state.n_state, templates.len());
    run_simulate(
        &mut workspaces,
        &StageContextFixture::new(
            &templates,
            &state_boxes,
            &[test_support::hydro_only_bus_geometry()],
        )
        .ctx(),
        &fcf,
        &TrainingContext {
            node_graph: &crate::test_support::chain_node_graph(&stochastic),
            horizon: &horizon,
            state: &state,
            cut_state_layouts: &test_support::all_enabled_cut_state_layouts(&state, n_stages),
            study_dims: &test_support::study_dims(),
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
            hydro_cell_index: &test_support::identity_hydro_cell_index(256),
            block_hours_per_stage: &[vec![1.0]],
            entity_counts: &entity_counts,
            generic_constraint_row_entries: &[Vec::new()],
            pumping_consumption_mw_per_m3s: &[],
            contract_prices_per_stage: &[Vec::new()],
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
        workspaces[0].scratch.z_inflow_rhs_buf.len(),
        1,
        "z_inflow_rhs_buf must have exactly 1 entry for 1 hydro"
    );
    // With None, no clamping occurs.  mean=-1000 and sigma=1, so
    // z_inflow_rhs_buf[0] = -1000 + 1 * eta.  For |eta| < 5 this remains << 0.
    assert!(
        workspaces[0].scratch.z_inflow_rhs_buf[0] < 0.0,
        "with None method, z_inflow_rhs_buf[0] must be negative (raw eta applied), got {}",
        workspaces[0].scratch.z_inflow_rhs_buf[0]
    );
}

// -----------------------------------------------------------------------
// Simulation DCS integration tests (real ActiveSolver)
// -----------------------------------------------------------------------
//
// Exercise the simulation DCS branch in `solve_simulation_stage`: load the
// cut-free base, patch the pinned incoming state, solve the cut pool lazily,
// and extract the primal. The LP shapes mirror the forward/backward DCS
// fixtures (a coupling row ties storage_out col0 to the pinned storage_in
// col2, and cuts constrain theta against col0). The theta-sensitive
// observable is `SimulationStageResult.future_cost = primal[theta] * SCALE`,
// which distinguishes the converged theta and so catches a wrong (frozen)
// template load.
mod dcs_simulation {
    use std::collections::HashMap;
    use std::sync::mpsc;

    use cobre_core::scenario::SamplingScheme;
    use cobre_solver::{ActiveSolver, StageTemplate};

    use super::super::{
        SimLookups, SimStageIds, SimStageLoadSpec, SimulationOutputSpec, solve_simulation_stage,
    };
    use crate::context::TrainingContext;
    use crate::cut::FutureCostFunction;
    use crate::cut_selection::CutMetadata;
    use crate::dcs::DcsParams;
    use crate::energy_conversion::{EnergyConversion, EnergyConversionSet};
    use crate::horizon_mode::HorizonMode;
    use crate::setup::NodePos;
    use crate::test_support::StageContextFixture;

    use crate::inflow_method::InflowNonNegativityMethod;
    use crate::lp::builder::{PatchBuffer, StageGeometry, StateBox};
    use crate::lp::indexer::BlockRowFamily;
    use crate::setup::NodeId;
    use crate::setup::node_graph::StageIdx;
    use crate::simulation::types::{SimulationCostResult, SimulationStageResult};
    use crate::test_support;
    use crate::workspace::{NoisePreallocation, SolverWorkspace, WorkspaceSizing};

    const X_HAT: f64 = 2.0;

    /// Cut-free base: cols `[storage_out=0, z_inflow=1, storage_in=2,
    /// theta=3]`, row 0 the z-inflow definition `z_inflow = rhs` (mirrors
    /// production's `fill_z_inflow_patches` row, keeping `z_inflow` a defined
    /// column rather than a free one, and matching every built template's own
    /// invariant that the z-inflow rows lead the row space), row 1 the
    /// coupling row `storage_out - storage_in = 0`, doubling as the water-balance
    /// row `run_one_sim_stage`'s geometry addresses, minimise `theta`.
    /// `storage_in` is pinned to `x_hat`; cuts constrain `theta` against
    /// `storage_out` (col 0). Cols 4-11 (turbine, spillage, withdrawal-slack,
    /// and the four operational-violation slack families) are decoupled
    /// padding — zero cost, zero NZ — so per-block hydro extraction addresses
    /// real columns without perturbing the solved LP.
    fn sim_core_template() -> StageTemplate {
        StageTemplate {
            num_cols: 12,
            num_rows: 2,
            num_nz: 3,
            col_starts: vec![0_i32, 1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
            row_indices: vec![1_i32, 0, 1],
            values: vec![1.0, 1.0, -1.0],
            col_lower: vec![
                0.0, 0.0, 0.0, -1.0e6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            col_upper: vec![
                f64::INFINITY,
                f64::INFINITY,
                f64::INFINITY,
                1.0e6,
                f64::INFINITY,
                f64::INFINITY,
                f64::INFINITY,
                f64::INFINITY,
                f64::INFINITY,
                f64::INFINITY,
                f64::INFINITY,
                f64::INFINITY,
            ],
            objective: vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            row_lower: vec![0.0, 0.0],
            row_upper: vec![0.0, 0.0],
            n_state: 1,
            col_scale: Vec::new(),
            row_scale: Vec::new(),
        }
    }

    /// All-cuts frozen template: cut-free base + the three pool cuts frozen as
    /// structural rows 2..5 (slot order), with the z-inflow definition row
    /// shifted to row 0 like every other fixture here. `num_rows = 5`. Cols
    /// 4-11 are the same decoupled padding as `sim_core_template`.
    fn sim_all_cuts_frozen() -> StageTemplate {
        StageTemplate {
            num_cols: 12,
            num_rows: 5,
            num_nz: 7,
            col_starts: vec![0_i32, 2, 3, 4, 7, 7, 7, 7, 7, 7, 7, 7, 7],
            row_indices: vec![1_i32, 3, 0, 1, 2, 3, 4],
            values: vec![1.0, -2.0, 1.0, -1.0, 1.0, 1.0, 1.0],
            col_lower: vec![
                0.0, 0.0, 0.0, -1.0e6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            col_upper: vec![
                f64::INFINITY,
                f64::INFINITY,
                f64::INFINITY,
                1.0e6,
                f64::INFINITY,
                f64::INFINITY,
                f64::INFINITY,
                f64::INFINITY,
                f64::INFINITY,
                f64::INFINITY,
                f64::INFINITY,
                f64::INFINITY,
            ],
            objective: vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            row_lower: vec![0.0, 0.0, 1.0, 0.0, 3.0],
            row_upper: vec![0.0, 0.0, f64::INFINITY, f64::INFINITY, f64::INFINITY],
            n_state: 1,
            col_scale: Vec::new(),
            row_scale: Vec::new(),
        }
    }

    /// Frozen template carrying a single DOMINATING spurious cut
    /// (`-5*col0 + theta >= 0`, floor 10 at `x_hat = 2`, NOT in the pool), plus
    /// the same leading z-inflow definition row as the other fixtures. Cols
    /// 4-11 are the same decoupled padding as `sim_core_template`.
    fn sim_frozen_dominating_cut() -> StageTemplate {
        StageTemplate {
            num_cols: 12,
            num_rows: 3,
            num_nz: 5,
            col_starts: vec![0_i32, 2, 3, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            row_indices: vec![1_i32, 2, 0, 1, 2],
            values: vec![1.0, -5.0, 1.0, -1.0, 1.0],
            col_lower: vec![
                0.0, 0.0, 0.0, -1.0e6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            col_upper: vec![
                f64::INFINITY,
                f64::INFINITY,
                f64::INFINITY,
                1.0e6,
                f64::INFINITY,
                f64::INFINITY,
                f64::INFINITY,
                f64::INFINITY,
                f64::INFINITY,
                f64::INFINITY,
                f64::INFINITY,
                f64::INFINITY,
            ],
            objective: vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            row_lower: vec![0.0, 0.0, 0.0],
            row_upper: vec![0.0, 0.0, f64::INFINITY],
            n_state: 1,
            col_scale: Vec::new(),
            row_scale: Vec::new(),
        }
    }

    /// Pool of three cuts on incoming storage; seeded so the DCS initial set
    /// omits the binding slot 1 (stale `last_active_iter`).
    fn sim_pool() -> FutureCostFunction {
        let mut fcf = FutureCostFunction::new(1, 1, 8, 10, &[0]);
        fcf.add_cut(NodeId(0), 0, 0, 0, 1.0, &[0.0]);
        fcf.add_cut(NodeId(0), 0, 0, 1, 0.0, &[2.0]); // binding: floor 2*x_hat = 4
        fcf.add_cut(NodeId(0), 0, 0, 2, 3.0, &[0.0]);
        let meta = |generated: u64, last: u64| CutMetadata {
            iteration_generated: generated,
            forward_pass_index: 0,
            node: NodeId(0),
            active_count: 0,
            last_active_iter: last,
        };
        fcf.pools[0].set_metadata_for_test(0, meta(1, 5));
        fcf.pools[0].set_metadata_for_test(1, meta(1, 1));
        fcf.pools[0].set_metadata_for_test(2, meta(1, 5));
        fcf
    }

    fn sim_active_workspace() -> SolverWorkspace<ActiveSolver> {
        let sizing = WorkspaceSizing {
            max_openings: 1,
            initial_pool_capacity: 16,
            max_local_fwd: 1,
            noise: NoisePreallocation::StochasticDim,
        };
        let solver = ActiveSolver::new().expect("ActiveSolver::new()");
        let state = test_support::state_layout(1, 0);
        let stochastic = test_support::hydro_free_stochastic_context(1, 1);
        let node_graph = crate::test_support::chain_node_graph(&stochastic);
        let study_dims = test_support::study_dims();
        let horizon = HorizonMode::Finite { num_stages: 1 };
        let cut_state_layouts = test_support::all_enabled_cut_state_layouts(&state, 1);
        let initial_state: Vec<f64> = Vec::new();
        let training_ctx = TrainingContext {
            node_graph: &node_graph,
            horizon: &horizon,
            state: &state,
            cut_state_layouts: &cut_state_layouts,
            study_dims: &study_dims,
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
        let stage_ctx_fixture = StageContextFixture::new(&[], &[], &[]);
        SolverWorkspace::new(
            0,
            0,
            solver,
            PatchBuffer::new(&state, &[], &[]),
            &training_ctx,
            &stage_ctx_fixture.ctx(),
            sizing,
        )
    }

    fn dcs_params() -> DcsParams {
        DcsParams {
            k1: None,
            k2: 2,
            nadic: 10,
            epsilon_viol: 1e-10,
            start_iteration: 2,
            max_inner_iterations: 50,
        }
    }

    /// Solve one simulation stage with the given `dcs` option and `frozen`
    /// template, returning `(immediate_cost, SimulationStageResult)`.
    // Builds the full two-branch (DCS vs frozen) stage-solve fixture inline;
    // extracting pieces would scatter the setup the branch comparison reads.
    fn run_one_sim_stage(
        dcs: Option<DcsParams>,
        frozen: &StageTemplate,
    ) -> (f64, SimulationStageResult) {
        let state = test_support::state_layout(1, 0);
        let core = sim_core_template();
        // turbine/spillage/withdrawal-slack/the four operational-violation
        // families address the decoupled padding cols the templates above
        // append; water_balance reuses the coupling row every fixture already
        // carries at row 1 (see `sim_core_template`).
        let geometry_per_stage = vec![StageGeometry {
            turbine: 4..5,
            spillage: 5..6,
            withdrawal_slack_neg: 6..7,
            withdrawal_slack_pos: 7..8,
            outflow_below_slack: 8..9,
            outflow_above_slack: 9..10,
            turbine_below_slack: 10..11,
            generation_below_slack: 11..12,
            water_balance: BlockRowFamily::one_per_entity(1..2),
            ..test_support::equipment_free_geometry(&[1])[0].clone()
        }];
        let templates = vec![core];
        let stochastic = super::make_stochastic_context(1);
        let horizon = HorizonMode::Finite { num_stages: 1 };
        let fcf = sim_pool();
        let entity_counts = super::entity_counts_1_hydro();
        let hprod = vec![vec![1.0]];
        let zero_ec = EnergyConversion {
            equivalent_productivity_mw_per_m3s: 0.0,
            reference_volume_hm3: 0.0,
            reference_outflow_m3s: 0.0,
        };
        let ec = EnergyConversionSet::new(
            vec![vec![zero_ec; 1]],
            vec![vec![0.0_f64; 1]],
            &test_support::minimal_hydros(1),
            1,
        );

        let mut ws = sim_active_workspace();
        ws.current_state.clear();
        ws.current_state.push(X_HAT);
        ws.scratch.load_rhs_buf.clear();
        ws.scratch.z_inflow_rhs_buf.clear();
        ws.scratch.ncs_col_upper_buf.clear();
        ws.scratch.inflow_m3s_buf.clear();
        ws.scratch.inflow_m3s_buf.push(0.0);

        let state_boxes = vec![StateBox {
            lower: vec![f64::NEG_INFINITY; state.n_state],
            upper: vec![f64::INFINITY; state.n_state],
        }];
        let fixture = StageContextFixture::new(&templates, &state_boxes, &geometry_per_stage);
        let ctx = fixture.ctx();
        let study_dims = test_support::study_dims();
        let training_ctx = TrainingContext {
            node_graph: &crate::test_support::chain_node_graph(&stochastic),
            horizon: &horizon,
            state: &state,
            cut_state_layouts: &test_support::all_enabled_cut_state_layouts(&state, 1),
            study_dims: &study_dims,
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &stochastic,
            initial_state: &[],
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
            dcs,
        };

        let (tx, _rx) = mpsc::sync_channel(4);
        let diversion: HashMap<cobre_core::EntityId, Vec<usize>> = HashMap::new();
        let output = SimulationOutputSpec {
            result_tx: &tx,
            hydro_cell_index: &test_support::identity_hydro_cell_index(256),
            block_hours_per_stage: &[vec![1.0]],
            entity_counts: &entity_counts,
            generic_constraint_row_entries: &[Vec::new()],
            pumping_consumption_mw_per_m3s: &[],
            contract_prices_per_stage: &[Vec::new()],
            contract_slots: &[],
            diversion_upstream: &diversion,
            hydro_productivities_per_stage: &hprod,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[0.0],
            event_sender: None,
            extended_delivery_anchors: &[],
            transit_seed_arcs: &[],
            past_defluences: &[],
            study_stage_dates: &[],
        };
        let ids = SimStageIds {
            t: StageIdx(0),
            stage_id_u32: 0,
            scenario_id: 0,
            node: NodePos(0),
            node_id: NodeId(0),
        };
        let load_spec = SimStageLoadSpec {
            frozen_template: frozen,
            warm_basis: None,
        };
        let lookups = SimLookups::build(
            &study_dims,
            ctx.geometry_per_stage,
            &test_support::identity_hydro_cell_index(256),
            1,
        );

        solve_simulation_stage(
            &mut ws,
            &ctx,
            &fcf,
            &training_ctx,
            &load_spec,
            &output,
            &ids,
            &lookups,
            &[0.0],
        )
        .expect("simulation stage solve must succeed")
    }

    /// The stage-level cost record (block 0) carrying total/immediate/future.
    fn stage_cost(result: &SimulationStageResult) -> &SimulationCostResult {
        result
            .costs
            .first()
            .expect("simulation stage result must carry a cost record")
    }

    /// The DCS branch (binding cut omitted from the seed) yields the
    /// same immediate cost and primal-derived fields as the frozen all-cuts
    /// path within 1e-9.
    #[test]
    fn sim_dcs_exact_matches_all_cuts() {
        let all_cuts = sim_all_cuts_frozen();
        let (frozen_imm, frozen) = run_one_sim_stage(None, &all_cuts);
        let (dcs_imm, dcs) = run_one_sim_stage(Some(dcs_params()), &all_cuts);

        assert!(
            (frozen_imm - dcs_imm).abs() < 1e-9,
            "immediate cost: frozen {frozen_imm} vs DCS {dcs_imm}"
        );
        let bc = stage_cost(&frozen);
        let dc = stage_cost(&dcs);
        assert!(
            (bc.immediate_cost - dc.immediate_cost).abs() < 1e-9,
            "record immediate_cost: frozen {} vs DCS {}",
            bc.immediate_cost,
            dc.immediate_cost
        );
        // future_cost = theta * SCALE; the binding cut floor is 4 at x_hat=2.
        assert!(
            (bc.future_cost - dc.future_cost).abs() < 1e-9,
            "future_cost: frozen {} vs DCS {}",
            bc.future_cost,
            dc.future_cost
        );
        assert!(
            (bc.total_cost - dc.total_cost).abs() < 1e-9,
            "total_cost: frozen {} vs DCS {}",
            bc.total_cost,
            dc.total_cost
        );
        assert!(
            (dc.future_cost - 4.0 * crate::DEFAULT_COST_SCALE_FACTOR).abs() < 1e-3,
            "DCS future_cost must reflect the binding cut theta=4, got {}",
            dc.future_cost
        );
    }

    /// A frozen template embedding a DOMINATING cut (floor 10, NOT in the
    /// pool) must NOT change the DCS result — proving the cut-free
    /// `ctx.templates[t]` is loaded, not `load_spec.frozen_template`. A wrong
    /// load surfaces as `future_cost` reflecting `theta = 10`.
    #[test]
    fn sim_dcs_frozen_cuts_present_uses_cut_free_core() {
        let all_cuts_template = sim_all_cuts_frozen();
        let dominating = sim_frozen_dominating_cut();
        let (_, frozen) = run_one_sim_stage(None, &all_cuts_template);
        let (_, dcs) = run_one_sim_stage(Some(dcs_params()), &dominating);

        let ac = stage_cost(&frozen);
        let dc = stage_cost(&dcs);
        assert!(
            (ac.future_cost - dc.future_cost).abs() < 1e-9,
            "future_cost: all-cuts {} vs DCS {} (DCS must ignore the dominating \
                 frozen cut and load the cut-free base)",
            ac.future_cost,
            dc.future_cost
        );
        assert!(
            (ac.immediate_cost - dc.immediate_cost).abs() < 1e-9,
            "immediate_cost: all-cuts {} vs DCS {}",
            ac.immediate_cost,
            dc.immediate_cost
        );
    }

    /// Complementary unit check of the `DcsParams::from_strategy` mapping
    /// that `simulation_ctx()` applies (the end-to-end `simulation_ctx()`
    /// wiring is covered by `simulation_ctx_propagates_dynamic_dcs_from_setup`
    /// in `setup/mod.rs`): a dynamic strategy yields `Some`, any other
    /// variant yields `None`.
    #[test]
    fn from_strategy_gates_dynamic_dcs() {
        use crate::cut_selection::CutSelectionStrategy;
        let dynamic = CutSelectionStrategy::Dynamic {
            k1: None,
            k2: 5,
            nadic: 10,
            epsilon_viol: 1e-10,
            start_iteration: 2,
        };
        let params = DcsParams::from_strategy(&dynamic)
            .expect("dynamic strategy must map to Some(DcsParams)");
        assert_eq!(params.k2, 5);
        let dominated = CutSelectionStrategy::Dominated {
            threshold: 1e-6,
            check_frequency: 10,
        };
        assert!(DcsParams::from_strategy(&dominated).is_none());
    }
}

/// Cross-path regression: the simulation pipeline's anticipated ring advances
/// identically to the training forward pass for the same solved-LP sequence.
///
/// `run_forward_stage` and `solve_simulation_stage` each drive their own
/// workspace through the identical three-stage solution sequence; both share
/// `debug_assert_bucket_copy_gap_intact` and `accumulate_and_shift_lag_state`,
/// so a residual shift reintroduced in either path diverges the captured
/// trajectories or trips that assert.
mod anticipated_ring_matches_forward_propagation {
    use std::collections::HashMap;
    use std::sync::mpsc;

    use cobre_core::scenario::SamplingScheme;
    use cobre_solver::{
        Basis, LpSolution, RowBatch, SolverError, SolverInterface, SolverStatistics, StageTemplate,
    };

    use super::super::{
        SimLookups, SimStageIds, SimStageLoadSpec, SimulationOutputSpec, solve_simulation_stage,
    };
    use crate::context::{StageContext, TrainingContext};
    use crate::cut::FutureCostFunction;
    use crate::energy_conversion::EnergyConversionSet;
    use crate::horizon_mode::HorizonMode;
    use crate::inflow_method::InflowNonNegativityMethod;
    use crate::lp::builder::{PatchBuffer, StateBox};
    use crate::lp::indexer::StateSpace;
    use crate::setup::NodeId;
    use crate::setup::NodePos;
    use crate::setup::node_graph::StageIdx;
    use crate::simulation::extraction::EntityCounts;
    use crate::test_support;
    use crate::test_support::{StageContextFixture, equipment_free_geometry};
    use crate::training::forward::{StageKey, run_forward_stage};
    use crate::trajectory::TrajectoryRecord;
    use crate::workspace::{BasisStore, NoisePreallocation, SolverWorkspace, WorkspaceSizing};

    const N_STAGES: usize = 3;

    /// Mock solver returning the `n`-th configured [`LpSolution`] on its `n`-th
    /// `solve()` call — one per stage, in order (unlike the file's shared
    /// [`MockSolver`], which always returns the same fixed solution).
    struct SequencedSolver {
        solutions: Vec<LpSolution>,
        call_count: usize,
        buf_primal: Vec<f64>,
        buf_dual: Vec<f64>,
        buf_reduced_costs: Vec<f64>,
    }

    impl SequencedSolver {
        fn new(solutions: Vec<LpSolution>) -> Self {
            Self {
                solutions,
                call_count: 0,
                buf_primal: Vec::new(),
                buf_dual: Vec::new(),
                buf_reduced_costs: Vec::new(),
            }
        }
    }

    impl SolverInterface for SequencedSolver {
        type Profile = cobre_solver::ActiveProfile;

        fn apply_profile(&mut self, _profile: &cobre_solver::ActiveProfile) {}
        fn solver_name_version(&self) -> String {
            "SequencedSolver 0.0.0".to_string()
        }
        fn load_model(&mut self, _template: &StageTemplate) {}
        fn add_rows(&mut self, _cuts: &RowBatch) {}
        fn set_row_bounds(&mut self, _indices: &[usize], _lower: &[f64], _upper: &[f64]) {}
        fn set_col_bounds(&mut self, _indices: &[usize], _lower: &[f64], _upper: &[f64]) {}
        fn solve(
            &mut self,
            _basis: Option<&Basis>,
        ) -> Result<cobre_solver::SolutionView<'_>, SolverError> {
            let sol = &self.solutions[self.call_count];
            self.call_count += 1;
            self.buf_primal.clone_from(&sol.primal);
            self.buf_dual.clone_from(&sol.dual);
            self.buf_reduced_costs.clone_from(&sol.reduced_costs);
            Ok(cobre_solver::SolutionView {
                objective: sol.objective,
                primal: &self.buf_primal,
                dual: &self.buf_dual,
                reduced_costs: &self.buf_reduced_costs,
                iterations: sol.iterations,
                solve_time_seconds: sol.solve_time_seconds,
            })
        }
        fn get_basis(&mut self, out: &mut Basis) {
            crate::test_support::fill_consistent_basis(out);
        }
        fn record_reconstruction_stats(&mut self) {}
        fn statistics(&self) -> SolverStatistics {
            SolverStatistics::default()
        }
        fn statistics_into(&self, out: &mut SolverStatistics) {
            out.copy_from(&SolverStatistics::default());
        }
        fn name(&self) -> &'static str {
            "Sequenced"
        }
    }

    fn ring_solution(slot0: f64, slot1: f64, num_cols: usize) -> LpSolution {
        let mut primal = vec![0.0_f64; num_cols];
        primal[0] = slot0;
        primal[1] = slot1;
        LpSolution {
            objective: 0.0,
            primal,
            dual: Vec::new(),
            reduced_costs: vec![0.0; num_cols],
            iterations: 0,
            solve_time_seconds: 0.0,
        }
    }

    /// Three-stage `k_max=2` ring sequence: an IC seed `[10, 20]`, then a
    /// stage-0 decision `100`, a stage-1 decision `200`, a stage-2 decision
    /// `300` — each stage's outgoing slots are `[prior slot1, this stage's
    /// decision]`, the ring-shift narrative the in-LP definition rows encode
    /// (irrelevant to this fixture: the mock solver returns these values
    /// unconditionally, so the test isolates state ADVANCE, not the ring's own
    /// LP-solved shift).
    fn ring_sequence(num_cols: usize) -> Vec<LpSolution> {
        vec![
            ring_solution(20.0, 100.0, num_cols),
            ring_solution(100.0, 200.0, num_cols),
            ring_solution(200.0, 300.0, num_cols),
        ]
    }

    fn ring_template(num_cols: usize, n_state: usize) -> StageTemplate {
        StageTemplate {
            num_cols,
            num_rows: 0,
            num_nz: 0,
            col_starts: vec![0_i32; num_cols + 1],
            row_indices: Vec::new(),
            values: Vec::new(),
            col_lower: vec![f64::NEG_INFINITY; num_cols],
            col_upper: vec![f64::INFINITY; num_cols],
            objective: vec![0.0; num_cols],
            row_lower: Vec::new(),
            row_upper: Vec::new(),
            n_state,
            col_scale: Vec::new(),
            row_scale: Vec::new(),
        }
    }

    fn ring_sizing() -> WorkspaceSizing {
        WorkspaceSizing {
            max_openings: 1,
            initial_pool_capacity: 16,
            max_local_fwd: 1,
            noise: NoisePreallocation::OnDemand,
        }
    }

    /// Drive `run_forward_stage` over `N_STAGES` stages, returning the captured
    /// `current_state` after each stage.
    fn run_forward_trajectory(
        state: &StateSpace,
        templates: &[StageTemplate],
        training_ctx: &TrainingContext<'_>,
        ctx: &StageContext<'_>,
        fcf: &FutureCostFunction,
        num_cols: usize,
    ) -> Vec<Vec<f64>> {
        let mut ws = SolverWorkspace::new(
            0,
            0,
            SequencedSolver::new(ring_sequence(num_cols)),
            PatchBuffer::new(state, &[], &[]),
            training_ctx,
            ctx,
            ring_sizing(),
        );
        ws.current_state.clear();
        ws.current_state.extend_from_slice(&[10.0, 20.0]);

        let mut basis_store = BasisStore::new(1, N_STAGES);
        let mut slices = basis_store.split_workers_mut(1);
        let mut records: Vec<TrajectoryRecord> = (0..N_STAGES)
            .map(|_| TrajectoryRecord {
                primal: Vec::new(),
                dual: Vec::new(),
                stage_cost: 0.0,
                node_id: NodeId(0),
                state: Vec::new(),
            })
            .collect();

        let mut trajectory = Vec::with_capacity(N_STAGES);
        for (t, template) in templates.iter().enumerate() {
            ws.solver.load_model(template);
            let key = StageKey {
                t: StageIdx(t),
                m: 0,
                local_m: 0,
                iteration: 1,
                raw_noise: &[],
                basis_row_capacity: template.num_rows,
                pool: &fcf.pools[t],
                dcs: None,
                node: NodePos(t),
            };
            run_forward_stage(
                &mut ws,
                &mut slices[0],
                ctx,
                training_ctx,
                &key,
                &mut records,
            )
            .expect("forward stage solve must succeed");
            trajectory.push(ws.current_state.clone());
        }
        trajectory
    }

    /// Drive `solve_simulation_stage` over `N_STAGES` stages, returning the
    /// captured `current_state` after each stage.
    fn run_simulation_trajectory(
        state: &StateSpace,
        templates: &[StageTemplate],
        training_ctx: &TrainingContext<'_>,
        ctx: &StageContext<'_>,
        fcf: &FutureCostFunction,
        num_cols: usize,
    ) -> Vec<Vec<f64>> {
        let mut ws = SolverWorkspace::new(
            0,
            0,
            SequencedSolver::new(ring_sequence(num_cols)),
            PatchBuffer::new(state, &[], &[]),
            training_ctx,
            ctx,
            ring_sizing(),
        );
        ws.current_state.clear();
        ws.current_state.extend_from_slice(&[10.0, 20.0]);
        ws.scratch.load_rhs_buf.clear();
        ws.scratch.z_inflow_rhs_buf.clear();
        ws.scratch.ncs_col_upper_buf.clear();
        ws.scratch.inflow_m3s_buf.clear();

        let entity_counts = EntityCounts {
            hydro_ids: Vec::new(),
            hydro_productivities: Vec::new(),
            thermal_ids: Vec::new(),
            line_ids: Vec::new(),
            bus_ids: Vec::new(),
            pumping_station_ids: Vec::new(),
            contract_ids: Vec::new(),
            non_controllable_ids: Vec::new(),
        };
        let hprod: Vec<Vec<f64>> = vec![Vec::new(); N_STAGES];
        let ec = EnergyConversionSet::new(Vec::new(), Vec::new(), &[], N_STAGES);
        let diversion: HashMap<cobre_core::EntityId, Vec<usize>> = HashMap::new();
        let (tx, _rx) = mpsc::sync_channel(N_STAGES.max(1));
        let output = SimulationOutputSpec {
            result_tx: &tx,
            hydro_cell_index: &test_support::identity_hydro_cell_index(256),
            block_hours_per_stage: &[vec![1.0], vec![1.0], vec![1.0]],
            entity_counts: &entity_counts,
            generic_constraint_row_entries: &[Vec::new(), Vec::new(), Vec::new()],
            pumping_consumption_mw_per_m3s: &[],
            contract_prices_per_stage: &[Vec::new(), Vec::new(), Vec::new()],
            contract_slots: &[],
            diversion_upstream: &diversion,
            hydro_productivities_per_stage: &hprod,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[],
            event_sender: None,
            extended_delivery_anchors: &[],
            transit_seed_arcs: &[],
            past_defluences: &[],
            study_stage_dates: &[],
        };
        let lookups = SimLookups::build(
            training_ctx.study_dims,
            ctx.geometry_per_stage,
            &test_support::identity_hydro_cell_index(256),
            0,
        );

        let mut trajectory = Vec::with_capacity(N_STAGES);
        for (t, template) in templates.iter().enumerate() {
            #[allow(clippy::cast_possible_truncation)]
            let ids = SimStageIds {
                t: StageIdx(t),
                stage_id_u32: t as u32,
                scenario_id: 0,
                node: NodePos(t),
                node_id: NodeId(t as i32),
            };
            let load_spec = SimStageLoadSpec {
                frozen_template: template,
                warm_basis: None,
            };
            let (_immediate, _result) = solve_simulation_stage(
                &mut ws,
                ctx,
                fcf,
                training_ctx,
                &load_spec,
                &output,
                &ids,
                &lookups,
                &[],
            )
            .expect("simulation stage solve must succeed");
            trajectory.push(ws.current_state.clone());
        }
        trajectory
    }

    /// The anticipated ring's per-stage outgoing state in simulation equals the
    /// training/forward propagation for the identical solved-LP sequence — the
    /// copy-outgoing convention, not a residual shift.
    #[test]
    fn simulation_ring_matches_forward_pass_for_identical_solves() {
        // Matches this fixture's own N_STAGES, not state_layout_full's wider
        // internal margin.
        let resolution = test_support::constant_lead_resolution(&[2], N_STAGES);
        let state = test_support::state_layout_with_transit_buckets_and_resolution(
            0,
            0,
            Vec::new(),
            vec![2],
            resolution,
        );
        let num_cols = state.theta + 1;
        let template = ring_template(num_cols, state.n_state);
        let templates = vec![template.clone(), template.clone(), template];
        let stochastic = test_support::hydro_free_stochastic_context(N_STAGES, 1);
        let horizon = HorizonMode::Finite {
            num_stages: N_STAGES,
        };
        let study_dims = test_support::study_dims();
        let fcf = FutureCostFunction::new(N_STAGES, state.n_state, 1, 1, &[0, 0, 0]);
        let cut_state_layouts = test_support::all_enabled_cut_state_layouts(&state, N_STAGES);

        let state_boxes = vec![
            StateBox {
                lower: vec![f64::NEG_INFINITY; state.n_state],
                upper: vec![f64::INFINITY; state.n_state],
            };
            N_STAGES
        ];
        let geometry = equipment_free_geometry(&[1, 1, 1]);
        let fixture = StageContextFixture::new(&templates, &state_boxes, &geometry);
        let ctx = fixture.ctx();
        let training_ctx = TrainingContext {
            node_graph: &crate::test_support::chain_node_graph(&stochastic),
            horizon: &horizon,
            state: &state,
            cut_state_layouts: &cut_state_layouts,
            study_dims: &study_dims,
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &stochastic,
            initial_state: &[],
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

        let forward_trajectory =
            run_forward_trajectory(&state, &templates, &training_ctx, &ctx, &fcf, num_cols);
        let sim_trajectory =
            run_simulation_trajectory(&state, &templates, &training_ctx, &ctx, &fcf, num_cols);

        let expected = [
            vec![20.0_f64, 100.0],
            vec![100.0, 200.0],
            vec![200.0, 300.0],
        ];
        for t in 0..N_STAGES {
            assert_eq!(
                forward_trajectory[t], sim_trajectory[t],
                "stage {t}: simulation's anticipated-ring state must match the \
                 forward pass's for the identical solved-LP sequence"
            );
            for (i, (&got, &want)) in forward_trajectory[t]
                .iter()
                .zip(expected[t].iter())
                .enumerate()
            {
                assert!(
                    (got - want).abs() < 1e-9,
                    "stage {t} slot {i}: expected {want}, got {got}"
                );
            }
        }
    }
}

/// The pin the two simulation node-visit sites set is the shared
/// `node_pinned_scenario`: `None` at a `Generated` node (the sampled/generated
/// byte-parity guarantee) and the declared column `k` at an `External` node.
#[test]
fn simulation_node_pin_is_none_for_generated_and_declared_column_for_external() {
    use crate::setup::node_graph::{
        NodeGraph, NodeId, NodeOpenings, NodePos, NodeRuntime, NodeSuccessor, OpeningSource,
        StageIdx,
    };

    let external = NodeRuntime {
        stage: StageIdx(0),
        pool_id: 0,
        openings: NodeOpenings {
            source: OpeningSource::External,
            offset: 5,
            len: 1,
            q: 1.0,
        },
    };
    let generated = NodeRuntime {
        stage: StageIdx(1),
        pool_id: 1,
        openings: NodeOpenings {
            source: OpeningSource::Generated,
            offset: 0,
            len: 3,
            q: 1.0 / 3.0,
        },
    };
    let ng = NodeGraph {
        node_ids: vec![NodeId(0), NodeId(1)].into(),
        nodes: vec![external, generated].into(),
        successors: vec![
            vec![NodeSuccessor {
                child: NodePos(1),
                probability: 1.0,
            }],
            Vec::new(),
        ]
        .into(),
        n_pools: 2,
        pool_stage: vec![StageIdx(0), StageIdx(1)],
    };

    assert_eq!(ng.node_pinned_scenario(NodePos(0)), Some(5));
    assert_eq!(ng.node_pinned_scenario(NodePos(1)), None);
}
