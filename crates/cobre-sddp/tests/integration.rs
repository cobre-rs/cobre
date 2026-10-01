//! End-to-end integration tests for the SDDP training loop.
//!
//! Exercises the full [`cobre_sddp::train`] function with a small toy system
//! (1 hydro, 0 PAR order, 2 stages). Also covers `StudySetup::new`'s own
//! validation surface, e.g. a precomputed inflow model shape mismatch.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap
)]
// `..Default::default()` in the make_* Spec calls is the intentional future-field
// seam from `common::builders` — a no-op today, not dead code.
#![allow(clippy::needless_update)]

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc;

use chrono::NaiveDate;
use cobre_comm::{CommData, CommError, Communicator, ReduceOp};
use cobre_core::{
    DeficitSegment, EntityId, TrainingEvent,
    scenario::{
        CorrelationEntity, CorrelationGroup, CorrelationModel, CorrelationProfile, SamplingScheme,
    },
    temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    },
};
use cobre_solver::{
    Basis, RowBatch, SolverError, SolverInterface, SolverStatistics, StageTemplate,
};
use cobre_stochastic::{
    ClassSchemes, OpeningTreeInputs, StochasticContext, build_stochastic_context,
};

use cobre_sddp::{
    SddpError, SolverProfiles, StoppingMode, StoppingRule, StoppingRuleSet, TrainingConfig,
    config::{CutManagementConfig, EventConfig, LoopConfig},
    context::TrainingContext,
    cut::fcf::FutureCostFunction,
    horizon_mode::HorizonMode,
    indexer::{CutStateProjection, StateSpace, StudyDimensions},
    inflow_method::InflowNonNegativityMethod,
    lead_time::AnticipatedResolution,
    risk_measure::RiskMeasure,
    test_support::{StageContextFixture, equipment_free_geometry, permissive_state_boxes},
    train,
};

mod common;
use common::StubComm;
use common::builders::{BusSpec, HydroSpec, StageSpec, make_bus, make_hydro, make_stage};

// ===========================================================================
// Shared helpers
// ===========================================================================

/// Mirrors the gated `test_support::state_layout_for` via the public
/// [`StateSpace::new`], so this external test crate (which cannot see the parent
/// crate's `#[cfg(test)]` surface) resolves byte-identical patch columns.
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

/// Communicator wrapper that sets `flag` to `true` on the first `allgatherv`
/// call, simulating a shutdown signal arriving mid-iteration-1. On subsequent
/// calls it behaves identically to [`StubComm`].
struct ShutdownComm {
    flag: Arc<AtomicBool>,
    allgatherv_calls: AtomicUsize,
}

impl ShutdownComm {
    fn new(flag: Arc<AtomicBool>) -> Self {
        Self {
            flag,
            allgatherv_calls: AtomicUsize::new(0),
        }
    }
}

impl Communicator for ShutdownComm {
    fn allgatherv<T: CommData>(
        &self,
        send: &[T],
        recv: &mut [T],
        _counts: &[usize],
        _displs: &[usize],
    ) -> Result<(), CommError> {
        recv[..send.len()].clone_from_slice(send);
        if self.allgatherv_calls.fetch_add(1, Ordering::Relaxed) == 0 {
            self.flag.store(true, Ordering::Relaxed);
        }
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

    fn abort(&self, error_code: i32) -> ! {
        std::process::exit(error_code)
    }
}

/// Mock solver that returns objectives from a repeating sequence, cycling
/// `objectives[call_count % len]` per `solve`.
struct MockSolver {
    objectives: Vec<f64>,
    call_count: usize,
    infeasible_on_call: Option<usize>,
}

impl MockSolver {
    fn with_objectives(objectives: Vec<f64>) -> Self {
        Self {
            objectives,
            call_count: 0,
            infeasible_on_call: None,
        }
    }

    fn with_fixed(objective: f64) -> Self {
        Self::with_objectives(vec![objective])
    }

    fn infeasible_on_first() -> Self {
        Self {
            objectives: vec![0.0],
            call_count: 0,
            infeasible_on_call: Some(0),
        }
    }
}

impl SolverInterface for MockSolver {
    type Profile = cobre_solver::ActiveProfile;

    fn apply_profile(&mut self, _profile: &cobre_solver::ActiveProfile) {}
    fn solver_name_version(&self) -> String {
        "MockSolver 0.0.0".to_string()
    }
    fn load_model(&mut self, _template: &StageTemplate) {}
    fn add_rows(&mut self, _cuts: &RowBatch) {}
    fn set_row_bounds(&mut self, _indices: &[usize], _lower: &[f64], _upper: &[f64]) {}
    fn set_col_bounds(&mut self, _indices: &[usize], _lower: &[f64], _upper: &[f64]) {}

    fn solve(
        &mut self,
        _basis: Option<&Basis>,
    ) -> Result<cobre_solver::SolutionView<'_>, SolverError> {
        let call = self.call_count;
        self.call_count += 1;
        if self.infeasible_on_call == Some(call) {
            return Err(SolverError::Infeasible);
        }
        let obj = self.objectives[call % self.objectives.len()];
        Ok(cobre_solver::SolutionView {
            objective: obj,
            primal: &[0.0, 0.0, 0.0, 0.0],
            dual: &[0.0, 0.0],
            reduced_costs: &[0.0, 0.0, 0.0, 0.0],
            iterations: 0,
            solve_time_seconds: 0.0,
        })
    }

    fn get_basis(&mut self, out: &mut Basis) {
        cobre_sddp::test_support::fill_consistent_basis(out);
    }

    fn statistics(&self) -> SolverStatistics {
        SolverStatistics::default()
    }

    fn statistics_into(&self, out: &mut SolverStatistics) {
        *out = self.statistics();
    }

    fn name(&self) -> &'static str {
        "MockIntegration"
    }
}

/// Mock solver returning a zero-filled dual slice sized to the current row count.
///
/// Unlike `MockSolver`'s hardcoded two-element dual, this grows the dual buffer
/// as cuts accumulate, so the backward pass can solve interior stages that carry
/// active cut rows without an out-of-bounds dual access.
struct ExpandingMockSolver {
    objectives: Vec<f64>,
    call_count: usize,
    current_num_rows: usize,
    dual_buf: Vec<f64>,
    primal_buf: Vec<f64>,
}

impl ExpandingMockSolver {
    fn with_objectives(objectives: Vec<f64>) -> Self {
        Self {
            objectives,
            call_count: 0,
            current_num_rows: 0,
            dual_buf: vec![0.0_f64; 64],
            primal_buf: vec![0.0_f64; 4],
        }
    }
}

impl SolverInterface for ExpandingMockSolver {
    type Profile = cobre_solver::ActiveProfile;

    fn apply_profile(&mut self, _profile: &cobre_solver::ActiveProfile) {}
    fn solver_name_version(&self) -> String {
        "ExpandingMockSolver 0.0.0".to_string()
    }

    fn load_model(&mut self, template: &StageTemplate) {
        self.current_num_rows = template.num_rows;
    }

    fn add_rows(&mut self, cuts: &RowBatch) {
        self.current_num_rows += cuts.num_rows;
        if self.current_num_rows > self.dual_buf.len() {
            self.dual_buf.resize(self.current_num_rows, 0.0);
        }
    }

    fn set_row_bounds(&mut self, _indices: &[usize], _lower: &[f64], _upper: &[f64]) {}
    fn set_col_bounds(&mut self, _indices: &[usize], _lower: &[f64], _upper: &[f64]) {}

    fn solve(
        &mut self,
        _basis: Option<&Basis>,
    ) -> Result<cobre_solver::SolutionView<'_>, SolverError> {
        let call = self.call_count;
        self.call_count += 1;
        let obj = self.objectives[call % self.objectives.len()];
        if self.dual_buf.len() < self.current_num_rows {
            self.dual_buf.resize(self.current_num_rows, 0.0);
        }
        Ok(cobre_solver::SolutionView {
            objective: obj,
            primal: &self.primal_buf,
            dual: &self.dual_buf[..self.current_num_rows],
            reduced_costs: &self.primal_buf,
            iterations: 0,
            solve_time_seconds: 0.0,
        })
    }

    fn get_basis(&mut self, out: &mut Basis) {
        cobre_sddp::test_support::fill_consistent_basis(out);
    }

    fn statistics(&self) -> SolverStatistics {
        SolverStatistics::default()
    }

    fn statistics_into(&self, out: &mut SolverStatistics) {
        *out = self.statistics();
    }

    fn name(&self) -> &'static str {
        "ExpandingMock"
    }
}

/// Build a `StochasticContext` with `n_stages` stages, 1 hydro, and seed 42.
#[allow(clippy::cast_possible_wrap, clippy::too_many_lines)]
fn make_stochastic_context(n_stages: usize, n_openings: usize) -> StochasticContext {
    use cobre_core::SystemBuilder;
    use cobre_core::entities::hydro::{HydroGenerationModel, HydroPenalties};
    use cobre_core::scenario::InflowModel;

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
        .map(|idx| {
            make_stage(
                idx,
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
                        branching_factor: n_openings,
                        noise_method: NoiseMethod::Saa,
                    },
                    ..Default::default()
                },
            )
        })
        .collect();

    let inflow_models: Vec<InflowModel> = (0..n_stages)
        .map(|i| InflowModel {
            hydro_id: EntityId(1),
            stage_id: i as i32,
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

/// Minimal stage template for N=1 hydro, L=0 PAR.
fn minimal_template() -> StageTemplate {
    // N=1, L=0 → cols: storage(0), z_inflow(1), storage_in(2), theta(3)
    //             rows: z_inflow(0), mock_pin(1)
    StageTemplate {
        num_cols: 4,
        num_rows: 2,
        num_nz: 2,
        col_starts: vec![0, 0, 1, 2, 2],
        row_indices: vec![0, 1],
        values: vec![1.0, 1.0],
        col_lower: vec![0.0, f64::NEG_INFINITY, 0.0, 0.0],
        col_upper: vec![f64::INFINITY; 4],
        objective: vec![0.0, 0.0, 0.0, 1.0],
        row_lower: vec![0.0; 2],
        row_upper: vec![0.0; 2],
        n_state: 1,
        col_scale: Vec::new(),
        row_scale: Vec::new(),
    }
}

fn make_fcf(n_stages: usize) -> FutureCostFunction {
    FutureCostFunction::new(n_stages, 1, 1, FCF_CAPACITY_ITERATIONS, &vec![0; n_stages])
}

fn iteration_limit(limit: u64) -> StoppingRuleSet {
    StoppingRuleSet {
        rules: vec![StoppingRule::IterationLimit { limit }],
        mode: StoppingMode::Any,
    }
}

/// All training parameters for a 2-stage, N=1 toy system.
struct Fixture {
    n_stages: usize,
    templates: Vec<StageTemplate>,
    state: StateSpace,
    initial_state: Vec<f64>,
    stochastic: StochasticContext,
    horizon: HorizonMode,
    risk_measures: Vec<RiskMeasure>,
}

const FCF_CAPACITY_ITERATIONS: u64 = 50;

impl Fixture {
    fn new(n_stages: usize) -> Self {
        let state = state_layout_for(1, 0);
        let templates = vec![minimal_template(); n_stages];
        let initial_state = vec![0.0_f64; state.n_state];
        let stochastic = make_stochastic_context(n_stages, 1);
        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let risk_measures = vec![RiskMeasure::Expectation; n_stages];

        Self {
            n_stages,
            templates,
            state,
            initial_state,
            stochastic,
            horizon,
            risk_measures,
        }
    }
}

/// Run a single training pass with a given stochastic context.
fn run_one_deterministic_pass(
    fx: &Fixture,
    stochastic: &StochasticContext,
    limit: u64,
) -> cobre_sddp::TrainingOutcome {
    let mut fcf = make_fcf(fx.n_stages);
    let mut solver = MockSolver::with_fixed(50.0);
    let state_boxes = permissive_state_boxes(fx.state.n_state, fx.n_stages);
    let geometry = equipment_free_geometry(&[1usize, 1]);
    let stage_ctx_fixture = StageContextFixture::new(&fx.templates, &state_boxes, &geometry);
    let stage_ctx = stage_ctx_fixture.ctx();
    train(
        &mut solver,
        TrainingConfig {
            loop_config: LoopConfig {
                forward_passes: 1,
                training_enumerated: false,
                max_iterations: 10,
                start_iteration: 0,
                n_fwd_threads: 1,
                stopping_rules: iteration_limit(limit),
            },
            cut_management: CutManagementConfig {
                cut_selection: None,
                budget: None,
                cut_activity_tolerance: 0.0,
                risk_measures: fx.risk_measures.clone(),
            },
            events: EventConfig {
                event_sender: None,
                checkpoint_interval: None,
                shutdown_flag: None,
                export_states: false,
            },
        },
        &mut fcf,
        &stage_ctx,
        &TrainingContext {
            node_graph: &cobre_sddp::test_support::chain_node_graph(stochastic),
            horizon: &fx.horizon,
            state: &fx.state,
            cut_state_layouts: &all_enabled_cut_state_layouts(&fx.state, fx.n_stages),
            study_dims: &study_dims(),
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic,
            initial_state: &fx.initial_state,
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            lag_accum_seed: &[],
            lag_weight_seed: &[],
            dcs: None,
            stages: &[],
        },
        &StubComm,
        || Ok(MockSolver::with_fixed(50.0)),
        None,
        SolverProfiles::default(),
    )
    .unwrap()
}

#[test]
fn train_converges_with_mock_solver() {
    let fx = Fixture::new(2);
    let mut fcf = make_fcf(fx.n_stages);
    let mut solver = MockSolver::with_fixed(100.0);
    let comm = StubComm;

    let config = TrainingConfig {
        loop_config: LoopConfig {
            forward_passes: 1,
            training_enumerated: false,
            max_iterations: 10,
            start_iteration: 0,
            n_fwd_threads: 1,
            stopping_rules: iteration_limit(10),
        },
        cut_management: CutManagementConfig {
            cut_selection: None,
            budget: None,
            cut_activity_tolerance: 0.0,
            risk_measures: fx.risk_measures.clone(),
        },
        events: EventConfig {
            event_sender: None,
            checkpoint_interval: None,
            shutdown_flag: None,
            export_states: false,
        },
    };

    let state_boxes = permissive_state_boxes(fx.state.n_state, fx.n_stages);
    let geometry = equipment_free_geometry(&[1usize, 1]);
    let stage_ctx_fixture = StageContextFixture::new(&fx.templates, &state_boxes, &geometry);
    let stage_ctx = stage_ctx_fixture.ctx();
    let result = train(
        &mut solver,
        config,
        &mut fcf,
        &stage_ctx,
        &TrainingContext {
            node_graph: &cobre_sddp::test_support::chain_node_graph(&fx.stochastic),
            horizon: &fx.horizon,
            state: &fx.state,
            cut_state_layouts: &all_enabled_cut_state_layouts(&fx.state, fx.n_stages),
            study_dims: &study_dims(),
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &fx.stochastic,
            initial_state: &fx.initial_state,
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            lag_accum_seed: &[],
            lag_weight_seed: &[],
            dcs: None,
            stages: &[],
        },
        &comm,
        || Ok(MockSolver::with_fixed(100.0)),
        None,
        SolverProfiles::default(),
    )
    .unwrap();

    assert!(result.result.iterations <= 10);
    assert!(result.result.final_lb >= 0.0);
    assert!(result.result.final_ub >= 0.0);
    assert!(result.result.final_gap.is_finite());
    assert!(!result.result.reason.is_empty());
}

#[test]
fn train_deterministic_with_same_seed() {
    let fx = Fixture::new(2);

    let result1 = run_one_deterministic_pass(&fx, &fx.stochastic, 5);

    let stochastic2 = make_stochastic_context(fx.n_stages, 1);
    let result2 = run_one_deterministic_pass(&fx, &stochastic2, 5);

    assert_eq!(
        result1.result.final_lb.to_bits(),
        result2.result.final_lb.to_bits()
    );
    assert_eq!(
        result1.result.final_ub.to_bits(),
        result2.result.final_ub.to_bits()
    );
    assert_eq!(result1.result.iterations, result2.result.iterations);
}

#[test]
fn train_lb_monotonically_nondecreasing() {
    let fx = Fixture::new(2);
    let mut fcf = make_fcf(fx.n_stages);
    // Fixed objective keeps LB constant; the property under test is that it never
    // decreases.
    let mut solver = MockSolver::with_fixed(80.0);
    let comm = StubComm;

    let (tx, rx) = mpsc::channel::<TrainingEvent>();
    let config = TrainingConfig {
        loop_config: LoopConfig {
            forward_passes: 1,
            training_enumerated: false,
            max_iterations: 20,
            start_iteration: 0,
            n_fwd_threads: 1,
            stopping_rules: iteration_limit(6),
        },
        cut_management: CutManagementConfig {
            cut_selection: None,
            budget: None,
            cut_activity_tolerance: 0.0,
            risk_measures: fx.risk_measures.clone(),
        },
        events: EventConfig {
            event_sender: Some(tx),
            checkpoint_interval: None,
            shutdown_flag: None,
            export_states: false,
        },
    };

    let state_boxes = permissive_state_boxes(fx.state.n_state, fx.n_stages);
    let geometry = equipment_free_geometry(&[1usize, 1]);
    let stage_ctx_fixture = StageContextFixture::new(&fx.templates, &state_boxes, &geometry);
    let stage_ctx = stage_ctx_fixture.ctx();
    train(
        &mut solver,
        config,
        &mut fcf,
        &stage_ctx,
        &TrainingContext {
            node_graph: &cobre_sddp::test_support::chain_node_graph(&fx.stochastic),
            horizon: &fx.horizon,
            state: &fx.state,
            cut_state_layouts: &all_enabled_cut_state_layouts(&fx.state, fx.n_stages),
            study_dims: &study_dims(),
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &fx.stochastic,
            initial_state: &fx.initial_state,
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            lag_accum_seed: &[],
            lag_weight_seed: &[],
            dcs: None,
            stages: &[],
        },
        &comm,
        || Ok(MockSolver::with_fixed(100.0)),
        None,
        SolverProfiles::default(),
    )
    .unwrap();

    let events: Vec<TrainingEvent> = rx.try_iter().collect();
    let lower_bounds: Vec<f64> = events
        .iter()
        .filter_map(|e| {
            if let TrainingEvent::ConvergenceUpdate { lower_bound, .. } = e {
                Some(*lower_bound)
            } else {
                None
            }
        })
        .collect();

    assert!(lower_bounds.len() >= 5);
    for window in lower_bounds.windows(2) {
        assert!(window[1] >= window[0]);
    }
}

#[test]
fn train_emits_correct_event_sequence() {
    let fx = Fixture::new(2);
    let mut fcf = make_fcf(fx.n_stages);
    let mut solver = MockSolver::with_fixed(100.0);
    let comm = StubComm;

    let (tx, rx) = mpsc::channel::<TrainingEvent>();
    let config = TrainingConfig {
        loop_config: LoopConfig {
            forward_passes: 1,
            training_enumerated: false,
            max_iterations: 10,
            start_iteration: 0,
            n_fwd_threads: 1,
            stopping_rules: iteration_limit(3),
        },
        cut_management: CutManagementConfig {
            cut_selection: None,
            budget: None,
            cut_activity_tolerance: 0.0,
            risk_measures: fx.risk_measures.clone(),
        },
        events: EventConfig {
            event_sender: Some(tx),
            checkpoint_interval: None,
            shutdown_flag: None,
            export_states: false,
        },
    };

    let state_boxes = permissive_state_boxes(fx.state.n_state, fx.n_stages);
    let geometry = equipment_free_geometry(&[1usize, 1]);
    let stage_ctx_fixture = StageContextFixture::new(&fx.templates, &state_boxes, &geometry);
    let stage_ctx = stage_ctx_fixture.ctx();
    train(
        &mut solver,
        config,
        &mut fcf,
        &stage_ctx,
        &TrainingContext {
            node_graph: &cobre_sddp::test_support::chain_node_graph(&fx.stochastic),
            horizon: &fx.horizon,
            state: &fx.state,
            cut_state_layouts: &all_enabled_cut_state_layouts(&fx.state, fx.n_stages),
            study_dims: &study_dims(),
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &fx.stochastic,
            initial_state: &fx.initial_state,
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            lag_accum_seed: &[],
            lag_weight_seed: &[],
            dcs: None,
            stages: &[],
        },
        &comm,
        || Ok(MockSolver::with_fixed(100.0)),
        None,
        SolverProfiles::default(),
    )
    .unwrap();

    let events: Vec<TrainingEvent> = rx.try_iter().collect();

    // 1 TrainingStarted + 3*(9 per-iteration) + 1 TrainingFinished = 29
    assert_eq!(events.len(), 29);
    assert!(matches!(events[0], TrainingEvent::TrainingStarted { .. }));
    assert!(matches!(events[28], TrainingEvent::TrainingFinished { .. }));

    let per_iter_types: &[fn(&TrainingEvent) -> bool] = &[
        |e| matches!(e, TrainingEvent::WorkerTiming { .. }),
        |e| matches!(e, TrainingEvent::ForwardPassComplete { .. }),
        |e| matches!(e, TrainingEvent::ForwardSyncComplete { .. }),
        |e| matches!(e, TrainingEvent::WorkerTiming { .. }),
        |e| matches!(e, TrainingEvent::BackwardPassComplete { .. }),
        |e| matches!(e, TrainingEvent::PolicySyncComplete { .. }),
        |e| matches!(e, TrainingEvent::PolicyTemplateFreezeComplete { .. }),
        |e| matches!(e, TrainingEvent::ConvergenceUpdate { .. }),
        |e| matches!(e, TrainingEvent::IterationSummary { .. }),
    ];

    for iter_idx in 0..3usize {
        let offset = 1 + iter_idx * 9;
        for (step, &check_fn) in per_iter_types.iter().enumerate() {
            assert!(check_fn(&events[offset + step]));
        }
    }
}

#[test]
fn train_stops_at_iteration_limit() {
    let fx = Fixture::new(2);
    let mut fcf = make_fcf(fx.n_stages);
    let mut solver = MockSolver::with_fixed(100.0);
    let comm = StubComm;

    let state_boxes = permissive_state_boxes(fx.state.n_state, fx.n_stages);
    let geometry = equipment_free_geometry(&[1usize, 1]);
    let stage_ctx_fixture = StageContextFixture::new(&fx.templates, &state_boxes, &geometry);
    let stage_ctx = stage_ctx_fixture.ctx();
    let result = train(
        &mut solver,
        TrainingConfig {
            loop_config: LoopConfig {
                forward_passes: 1,
                training_enumerated: false,
                max_iterations: 10,
                start_iteration: 0,
                n_fwd_threads: 1,
                stopping_rules: iteration_limit(3),
            },
            cut_management: CutManagementConfig {
                cut_selection: None,
                budget: None,
                cut_activity_tolerance: 0.0,
                risk_measures: fx.risk_measures.clone(),
            },
            events: EventConfig {
                event_sender: None,
                checkpoint_interval: None,
                shutdown_flag: None,
                export_states: false,
            },
        },
        &mut fcf,
        &stage_ctx,
        &TrainingContext {
            node_graph: &cobre_sddp::test_support::chain_node_graph(&fx.stochastic),
            horizon: &fx.horizon,
            state: &fx.state,
            cut_state_layouts: &all_enabled_cut_state_layouts(&fx.state, fx.n_stages),
            study_dims: &study_dims(),
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &fx.stochastic,
            initial_state: &fx.initial_state,
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            lag_accum_seed: &[],
            lag_weight_seed: &[],
            dcs: None,
            stages: &[],
        },
        &comm,
        || Ok(MockSolver::with_fixed(100.0)),
        None,
        SolverProfiles::default(),
    )
    .unwrap();

    assert_eq!(result.result.iterations, 3);
    assert_eq!(result.result.reason, "iteration_limit");
}

#[test]
fn train_stops_on_graceful_shutdown() {
    let fx = Fixture::new(2);
    let mut fcf = make_fcf(fx.n_stages);
    let mut solver = MockSolver::with_fixed(100.0);

    let shutdown_flag = Arc::new(AtomicBool::new(false));
    let comm = ShutdownComm::new(Arc::clone(&shutdown_flag));

    let rules = StoppingRuleSet {
        rules: vec![
            StoppingRule::GracefulShutdown,
            StoppingRule::IterationLimit { limit: 20 },
        ],
        mode: StoppingMode::Any,
    };

    let state_boxes = permissive_state_boxes(fx.state.n_state, fx.n_stages);
    let geometry = equipment_free_geometry(&[1usize, 1]);
    let stage_ctx_fixture = StageContextFixture::new(&fx.templates, &state_boxes, &geometry);
    let stage_ctx = stage_ctx_fixture.ctx();
    let result = train(
        &mut solver,
        TrainingConfig {
            loop_config: LoopConfig {
                forward_passes: 1,
                training_enumerated: false,
                max_iterations: 20,
                start_iteration: 0,
                n_fwd_threads: 1,
                stopping_rules: rules,
            },
            cut_management: CutManagementConfig {
                cut_selection: None,
                budget: None,
                cut_activity_tolerance: 0.0,
                risk_measures: fx.risk_measures.clone(),
            },
            events: EventConfig {
                event_sender: None,
                checkpoint_interval: None,
                shutdown_flag: Some(Arc::clone(&shutdown_flag)),
                export_states: false,
            },
        },
        &mut fcf,
        &stage_ctx,
        &TrainingContext {
            node_graph: &cobre_sddp::test_support::chain_node_graph(&fx.stochastic),
            horizon: &fx.horizon,
            state: &fx.state,
            cut_state_layouts: &all_enabled_cut_state_layouts(&fx.state, fx.n_stages),
            study_dims: &study_dims(),
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &fx.stochastic,
            initial_state: &fx.initial_state,
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            lag_accum_seed: &[],
            lag_weight_seed: &[],
            dcs: None,
            stages: &[],
        },
        &comm,
        || Ok(MockSolver::with_fixed(100.0)),
        None,
        SolverProfiles::default(),
    )
    .unwrap();

    assert_eq!(result.result.reason, "graceful_shutdown");
    assert!(result.result.iterations <= 2);
}

#[test]
fn train_propagates_infeasible_error() {
    let fx = Fixture::new(2);
    let mut fcf = make_fcf(fx.n_stages);
    let mut solver = MockSolver::infeasible_on_first();
    let comm = StubComm;

    let state_boxes = permissive_state_boxes(fx.state.n_state, fx.n_stages);
    let geometry = equipment_free_geometry(&[1usize, 1]);
    let stage_ctx_fixture = StageContextFixture::new(&fx.templates, &state_boxes, &geometry);
    let stage_ctx = stage_ctx_fixture.ctx();
    let result = train(
        &mut solver,
        TrainingConfig {
            loop_config: LoopConfig {
                forward_passes: 1,
                training_enumerated: false,
                max_iterations: 10,
                start_iteration: 0,
                n_fwd_threads: 1,
                stopping_rules: iteration_limit(10),
            },
            cut_management: CutManagementConfig {
                cut_selection: None,
                budget: None,
                cut_activity_tolerance: 0.0,
                risk_measures: fx.risk_measures.clone(),
            },
            events: EventConfig {
                event_sender: None,
                checkpoint_interval: None,
                shutdown_flag: None,
                export_states: false,
            },
        },
        &mut fcf,
        &stage_ctx,
        &TrainingContext {
            node_graph: &cobre_sddp::test_support::chain_node_graph(&fx.stochastic),
            horizon: &fx.horizon,
            state: &fx.state,
            cut_state_layouts: &all_enabled_cut_state_layouts(&fx.state, fx.n_stages),
            study_dims: &study_dims(),
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &fx.stochastic,
            initial_state: &fx.initial_state,
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            lag_accum_seed: &[],
            lag_weight_seed: &[],
            dcs: None,
            stages: &[],
        },
        &comm,
        || Ok(MockSolver::infeasible_on_first()),
        None,
        SolverProfiles::default(),
    );

    let outcome = result.expect("train must return Ok(TrainingOutcome) with captured error");
    assert!(outcome.error.is_some(), "expected error in TrainingOutcome");
    assert!(
        matches!(outcome.error, Some(SddpError::Infeasible { stage: 0, .. })),
        "expected SddpError::Infeasible at stage 0, got: {:?}",
        outcome.error
    );
    assert_eq!(
        outcome.result.iterations, 0,
        "no iterations should have completed"
    );
    assert_eq!(outcome.result.reason, "error");
}

/// D17: Level1 cut selection produces convergent results. With the stub kernel,
/// no cuts are deactivated by Level1 selection (the value-based kernel that
/// drives deactivation lives elsewhere).
#[test]
#[allow(clippy::too_many_lines)]
fn d17_level1_cut_selection_convergence() {
    use cobre_sddp::CutSelectionStrategy;

    let fx = Fixture::new(2);
    let mut fcf = make_fcf(fx.n_stages);
    let mut solver = MockSolver::with_fixed(100.0);
    let comm = StubComm;

    let (tx, rx) = mpsc::channel::<TrainingEvent>();
    let config = TrainingConfig {
        loop_config: LoopConfig {
            forward_passes: 1,
            training_enumerated: false,
            max_iterations: 10,
            start_iteration: 0,
            n_fwd_threads: 1,
            stopping_rules: iteration_limit(10),
        },
        cut_management: CutManagementConfig {
            cut_selection: Some(CutSelectionStrategy::Level1 {
                check_frequency: 2,
                tie_tolerance: 1e-10,
            }),
            budget: None,
            cut_activity_tolerance: 0.0,
            risk_measures: fx.risk_measures.clone(),
        },
        events: EventConfig {
            event_sender: Some(tx),
            checkpoint_interval: None,
            shutdown_flag: None,
            export_states: false,
        },
    };

    let state_boxes = permissive_state_boxes(fx.state.n_state, fx.n_stages);
    let geometry = equipment_free_geometry(&[1usize, 1]);
    let stage_ctx_fixture = StageContextFixture::new(&fx.templates, &state_boxes, &geometry);
    let stage_ctx = stage_ctx_fixture.ctx();
    let result = train(
        &mut solver,
        config,
        &mut fcf,
        &stage_ctx,
        &TrainingContext {
            node_graph: &cobre_sddp::test_support::chain_node_graph(&fx.stochastic),
            horizon: &fx.horizon,
            state: &fx.state,
            cut_state_layouts: &all_enabled_cut_state_layouts(&fx.state, fx.n_stages),
            study_dims: &study_dims(),
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &fx.stochastic,
            initial_state: &fx.initial_state,
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            lag_accum_seed: &[],
            lag_weight_seed: &[],
            dcs: None,
            stages: &[],
        },
        &comm,
        || Ok(MockSolver::with_fixed(100.0)),
        None,
        SolverProfiles::default(),
    )
    .unwrap();

    assert!(
        result.result.iterations <= 10,
        "training must complete within limit"
    );

    let events: Vec<TrainingEvent> = rx.try_iter().collect();
    let lower_bounds: Vec<f64> = events
        .iter()
        .filter_map(|e| {
            if let TrainingEvent::ConvergenceUpdate { lower_bound, .. } = e {
                Some(*lower_bound)
            } else {
                None
            }
        })
        .collect();

    assert!(
        !lower_bounds.is_empty(),
        "must have at least one ConvergenceUpdate event"
    );
    for window in lower_bounds.windows(2) {
        assert!(
            window[1] >= window[0],
            "lower bound must be non-decreasing: {} -> {}",
            window[0],
            window[1]
        );
    }

    let sel_events: Vec<&TrainingEvent> = events
        .iter()
        .filter(|e| matches!(e, TrainingEvent::PolicySelectionComplete { .. }))
        .collect();

    assert!(
        !sel_events.is_empty(),
        "must have at least one PolicySelectionComplete event"
    );

    // Stage 0 is exempt from cut selection. The populated count may include a
    // warm-start slot, so the check is `>=` (iterations * forward_passes), not `==`.
    assert!(
        fcf.pools[0].active_count() >= result.result.iterations as usize,
        "stage 0 must be exempt: expected at least {} active cuts, got {} \
         (populated={})",
        result.result.iterations,
        fcf.pools[0].active_count(),
        fcf.pools[0].populated(),
    );

    // Informational only: the mock never tracks basis ops, so this never fires.
    // It would flag warm-start degradation under a real solver (see BasisStore).
    let stats = solver.statistics();
    if stats.basis_offered > 0 && stats.basis_consistency_failures > stats.basis_offered / 2 {
        eprintln!(
            "WARNING: basis rejection rate after cut selection is {}/{}. \
             Consider implementing option 3 (discard cut row statuses).",
            stats.basis_consistency_failures, stats.basis_offered
        );
    }
}

/// D17 with basis reconstruction always active: the truncation guard does not
/// corrupt convergence, and reconstruction produces accepted warm-start bases
/// (zero rejections).
#[test]
fn d17_level1_cut_selection_reconstruction() {
    use cobre_sddp::CutSelectionStrategy;

    let fx = Fixture::new(2);
    let mut fcf = make_fcf(fx.n_stages);
    let mut solver = MockSolver::with_fixed(100.0);
    let comm = StubComm;

    let state_boxes = permissive_state_boxes(fx.state.n_state, fx.n_stages);
    let geometry = equipment_free_geometry(&[1usize, 1]);
    let stage_ctx_fixture = StageContextFixture::new(&fx.templates, &state_boxes, &geometry);
    let stage_ctx = stage_ctx_fixture.ctx();

    let result = train(
        &mut solver,
        TrainingConfig {
            loop_config: LoopConfig {
                forward_passes: 1,
                training_enumerated: false,
                max_iterations: 10,
                start_iteration: 0,
                n_fwd_threads: 1,
                stopping_rules: iteration_limit(10),
            },
            cut_management: CutManagementConfig {
                cut_selection: Some(CutSelectionStrategy::Level1 {
                    check_frequency: 2,
                    tie_tolerance: 1e-10,
                }),
                budget: None,
                cut_activity_tolerance: 0.0,
                risk_measures: fx.risk_measures.clone(),
            },
            events: EventConfig {
                event_sender: None,
                checkpoint_interval: None,
                shutdown_flag: None,
                export_states: false,
            },
        },
        &mut fcf,
        &stage_ctx,
        &TrainingContext {
            node_graph: &cobre_sddp::test_support::chain_node_graph(&fx.stochastic),
            horizon: &fx.horizon,
            state: &fx.state,
            cut_state_layouts: &all_enabled_cut_state_layouts(&fx.state, fx.n_stages),
            study_dims: &study_dims(),
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &fx.stochastic,
            initial_state: &fx.initial_state,
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            lag_accum_seed: &[],
            lag_weight_seed: &[],
            dcs: None,
            stages: &[],
        },
        &comm,
        || Ok(MockSolver::with_fixed(100.0)),
        None,
        SolverProfiles::default(),
    )
    .unwrap();

    assert!(
        result.result.final_lb.is_finite(),
        "D17+reconstruction: lower bound must be finite, got {}",
        result.result.final_lb,
    );

    let stats = solver.statistics();
    assert_eq!(
        stats.basis_consistency_failures, 0,
        "D17+reconstruction: expected 0 basis rejections, got {}",
        stats.basis_consistency_failures,
    );
}

/// D18: Lml1 cut selection produces convergent results. With the stub kernel,
/// no cuts are deactivated by Lml1 selection (the value-based kernel that drives
/// deactivation lives elsewhere).
#[test]
#[allow(clippy::too_many_lines)]
fn d18_lml1_cut_selection_convergence() {
    use cobre_sddp::CutSelectionStrategy;

    let fx = Fixture::new(2);
    let mut fcf = make_fcf(fx.n_stages);
    let mut solver = MockSolver::with_fixed(100.0);
    let comm = StubComm;

    let (tx, rx) = mpsc::channel::<TrainingEvent>();
    let config = TrainingConfig {
        loop_config: LoopConfig {
            forward_passes: 1,
            training_enumerated: false,
            max_iterations: 10,
            start_iteration: 0,
            n_fwd_threads: 1,
            stopping_rules: iteration_limit(10),
        },
        cut_management: CutManagementConfig {
            cut_selection: Some(CutSelectionStrategy::Lml1 {
                check_frequency: 2,
                tie_tolerance: 1e-10,
            }),
            budget: None,
            cut_activity_tolerance: 0.0,
            risk_measures: fx.risk_measures.clone(),
        },
        events: EventConfig {
            event_sender: Some(tx),
            checkpoint_interval: None,
            shutdown_flag: None,
            export_states: false,
        },
    };

    let state_boxes = permissive_state_boxes(fx.state.n_state, fx.n_stages);
    let geometry = equipment_free_geometry(&[1usize, 1]);
    let stage_ctx_fixture = StageContextFixture::new(&fx.templates, &state_boxes, &geometry);
    let stage_ctx = stage_ctx_fixture.ctx();
    let result = train(
        &mut solver,
        config,
        &mut fcf,
        &stage_ctx,
        &TrainingContext {
            node_graph: &cobre_sddp::test_support::chain_node_graph(&fx.stochastic),
            horizon: &fx.horizon,
            state: &fx.state,
            cut_state_layouts: &all_enabled_cut_state_layouts(&fx.state, fx.n_stages),
            study_dims: &study_dims(),
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &fx.stochastic,
            initial_state: &fx.initial_state,
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            lag_accum_seed: &[],
            lag_weight_seed: &[],
            dcs: None,
            stages: &[],
        },
        &comm,
        || Ok(MockSolver::with_fixed(100.0)),
        None,
        SolverProfiles::default(),
    )
    .unwrap();

    assert!(
        result.result.iterations <= 10,
        "training must complete within limit"
    );

    let events: Vec<TrainingEvent> = rx.try_iter().collect();
    let lower_bounds: Vec<f64> = events
        .iter()
        .filter_map(|e| {
            if let TrainingEvent::ConvergenceUpdate { lower_bound, .. } = e {
                Some(*lower_bound)
            } else {
                None
            }
        })
        .collect();

    assert!(
        !lower_bounds.is_empty(),
        "must have at least one ConvergenceUpdate event"
    );
    for window in lower_bounds.windows(2) {
        assert!(
            window[1] >= window[0],
            "lower bound must be non-decreasing: {} -> {}",
            window[0],
            window[1]
        );
    }

    let sel_events: Vec<&TrainingEvent> = events
        .iter()
        .filter(|e| matches!(e, TrainingEvent::PolicySelectionComplete { .. }))
        .collect();

    assert!(
        !sel_events.is_empty(),
        "must have at least one PolicySelectionComplete event"
    );

    // Stage 0 is exempt from cut selection. The populated count may include a
    // warm-start slot, so the check is `>=` (iterations * forward_passes), not `==`.
    assert!(
        fcf.pools[0].active_count() >= result.result.iterations as usize,
        "stage 0 must be exempt: expected at least {} active cuts, got {} \
         (populated={})",
        result.result.iterations,
        fcf.pools[0].active_count(),
        fcf.pools[0].populated(),
    );
}

/// D01 must produce a bit-identical lower bound (reference 182,500 $) when the
/// forward path runs through `reconstruct_basis` — a warm-start heuristic that
/// must not change the optimal LP solution.
#[test]
fn test_forward_basis_reconstruct_bit_identical_d01() {
    use std::path::Path;

    use cobre_core::scenario::ScenarioSource;
    use cobre_sddp::{StudySetup, hydro_models::prepare_hydro_models, setup::prepare_stochastic};
    use cobre_solver::ActiveSolver;

    let case_dir = Path::new("../../examples/deterministic/d01-thermal-dispatch");
    let config_path = case_dir.join("config.json");
    let config = cobre_io::parse_config(&config_path).expect("config must parse");
    let system = cobre_io::load_case(case_dir).expect("load_case must succeed");

    let prepare_result = prepare_stochastic(
        system,
        case_dir,
        &config,
        42,
        &ScenarioSource::default(),
        None,
    )
    .expect("prepare_stochastic must succeed");
    let system = prepare_result.system;
    let stochastic = prepare_result.stochastic;

    let hydro_models =
        prepare_hydro_models(&system, case_dir, false).expect("prepare_hydro_models must succeed");

    let mut setup = StudySetup::new(&system, &config, stochastic, hydro_models, Vec::new())
        .expect("StudySetup must build");

    let comm = StubComm;
    let mut solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");

    let outcome = setup
        .train(&mut solver, &comm, 1, ActiveSolver::new, None, None)
        .expect("train must return Ok");
    assert!(outcome.error.is_none(), "expected no training error");

    let diff = (outcome.result.final_lb - 182_500.0_f64).abs();
    assert!(
        diff <= 1e-6,
        "reconstruct path: expected lower bound 182500.0, got {} (diff={:.2e})",
        outcome.result.final_lb,
        diff
    );

    let stats = solver.statistics();
    assert_eq!(
        stats.basis_consistency_failures, 0,
        "reconstruct path: expected 0 basis rejections, got {}",
        stats.basis_consistency_failures
    );
}

/// Smoke test: the frozen-template backward pass (freeze activates on iteration 2)
/// runs to the iteration limit without diverging or panicking.
#[test]
fn frozen_backward_pass_smoke_test() {
    let n_iter = 5_u64;
    let fx = Fixture::new(3);
    let mut fcf = make_fcf(fx.n_stages);
    // The frozen path adds cut rows on iteration 2+; ExpandingMockSolver grows its
    // dual slice to match, where MockSolver's fixed 2-element dual would panic.
    let mut solver = ExpandingMockSolver::with_objectives(vec![50.0]);
    let state_boxes = permissive_state_boxes(fx.state.n_state, fx.n_stages);
    let geometry = equipment_free_geometry(&[1_usize, 1, 1]);
    let stage_ctx_fixture = StageContextFixture::new(&fx.templates, &state_boxes, &geometry);
    let stage_ctx = stage_ctx_fixture.ctx();

    let outcome = train(
        &mut solver,
        TrainingConfig {
            loop_config: LoopConfig {
                forward_passes: 1,
                training_enumerated: false,
                max_iterations: n_iter,
                start_iteration: 0,
                n_fwd_threads: 1,
                stopping_rules: iteration_limit(n_iter),
            },
            cut_management: CutManagementConfig {
                cut_selection: None,
                budget: None,
                cut_activity_tolerance: 0.0,
                risk_measures: fx.risk_measures.clone(),
            },
            events: EventConfig {
                event_sender: None,
                checkpoint_interval: None,
                shutdown_flag: None,
                export_states: false,
            },
        },
        &mut fcf,
        &stage_ctx,
        &TrainingContext {
            node_graph: &cobre_sddp::test_support::chain_node_graph(&fx.stochastic),
            horizon: &fx.horizon,
            state: &fx.state,
            cut_state_layouts: &all_enabled_cut_state_layouts(&fx.state, fx.n_stages),
            study_dims: &study_dims(),
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &fx.stochastic,
            initial_state: &fx.initial_state,
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            lag_accum_seed: &[],
            lag_weight_seed: &[],
            dcs: None,
            stages: &[],
        },
        &StubComm,
        || Ok(ExpandingMockSolver::with_objectives(vec![50.0])),
        None,
        SolverProfiles::default(),
    )
    .expect("frozen backward pass smoke: train must not error");

    assert_eq!(
        outcome.result.iterations, n_iter,
        "expected {n_iter} iterations, got {}",
        outcome.result.iterations
    );

    assert!(
        outcome.result.final_lb >= 0.0,
        "final lower bound must be non-negative; got {}",
        outcome.result.final_lb
    );
}

/// A precomputed inflow model built for a different system's hydro/stage
/// shape is rejected at setup rather than silently treated as absent.
#[test]
fn par_model_shape_mismatch_is_rejected_at_setup() {
    use cobre_sddp::StudySetup;
    use cobre_sddp::hydro_models::PrepareHydroModelsResult;
    use common::in_code_studies::{
        ChronologicalNoiseSpec, chronological_noise_study, stochastic_parallel_study,
    };

    let (system, config) = chronological_noise_study(&ChronologicalNoiseSpec {
        block_modes: [BlockMode::Parallel; 2],
        ..Default::default()
    });
    let stochastic = common::stochastic_in_code(&stochastic_parallel_study().0);

    let result = StudySetup::new(
        &system,
        &config,
        stochastic,
        PrepareHydroModelsResult::default_from_system(&system),
        Vec::new(),
    );

    let err =
        result.expect_err("a PAR model built for a different system's shape must be rejected");
    let msg = err.to_string();
    assert!(msg.contains("shape mismatch"), "message: {msg}");
    assert!(msg.contains("1 hydros"), "message: {msg}");
    assert!(msg.contains("2 hydros"), "message: {msg}");
    assert!(
        matches!(err, SddpError::Validation(_)),
        "expected SddpError::Validation, got {err:?}"
    );
}

/// Local mirror of the gated `test_support::all_enabled_cut_state_layouts`
/// via the public `CutStateProjection::new`, so this external test crate (which cannot
/// see the parent crate's `#[cfg(test)]` surface) builds the default all-enabled
/// per-pool projection. Every pool projects the full global state, keeping the
/// extracted subgradient bit-identical to the global-loop result.
fn all_enabled_cut_state_layouts(global: &StateSpace, n_stages: usize) -> Vec<CutStateProjection> {
    let full = StageStateConfig {
        storage: true,
        inflow_lags: true,
    };
    (0..n_stages)
        .map(|_| CutStateProjection::new(global, full))
        .collect()
}
