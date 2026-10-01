//! End-to-end integration tests for the stochastic load pipeline: [`System`]
//! construction with [`LoadModel`] entries through [`build_stochastic_context`]
//! and a mock-solver training run.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::too_many_lines
)]
// `..Default::default()` in the make_* Spec calls is the intentional future-field
// seam from `common::builders` — a no-op today, not dead code.
#![allow(clippy::needless_update)]

use std::collections::BTreeMap;
use std::sync::mpsc;

use chrono::NaiveDate;
use cobre_core::{
    DeficitSegment, EntityId, SystemBuilder, TrainingEvent,
    entities::hydro::{HydroGenerationModel, HydroPenalties},
    scenario::{CorrelationModel, InflowModel, LoadModel, SamplingScheme},
    temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    },
};
use cobre_sddp::{
    SolverProfiles, StoppingMode, StoppingRule, StoppingRuleSet, TrainingConfig,
    config::{CutManagementConfig, EventConfig, LoopConfig},
    context::TrainingContext,
    cut::fcf::FutureCostFunction,
    horizon_mode::HorizonMode,
    indexer::{CutStateProjection, StateSpace, StudyDimensions},
    inflow_method::InflowNonNegativityMethod,
    lead_time::AnticipatedResolution,
    risk_measure::RiskMeasure,
    test_support::{
        StageContextFixture, equipment_free_geometry, geometry_with_load_balance,
        permissive_state_boxes,
    },
    train,
};
use cobre_solver::{
    Basis, RowBatch, SolverError, SolverInterface, SolverStatistics, StageTemplate,
};
use cobre_stochastic::{
    ClassSchemes, OpeningTreeInputs, StochasticContext, build_stochastic_context,
};

mod common;
use common::StubComm;
use common::builders::{BusSpec, HydroSpec, StageSpec, make_bus, make_hydro, make_stage};

// ===========================================================================
// Shared helpers
// ===========================================================================

/// Mirror the gated `test_support::state_layout_for` via the public
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

/// Mock solver that returns a fixed objective on every `solve` call.
struct MockSolver {
    objective: f64,
}

impl MockSolver {
    fn with_fixed(objective: f64) -> Self {
        Self { objective }
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
        Ok(cobre_solver::SolutionView {
            objective: self.objective,
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
        "MockLoadIntegration"
    }
}

/// Build a `System` with 1 bus, 1 hydro, `n_stages` stages, and optionally
/// stochastic load. `load_std_mw == 0.0` yields deterministic load, so the
/// returned context reports `n_load_buses() == 0`. The correlation model is left
/// empty on purpose: `build_stochastic_context` treats that as independent
/// (identity) correlation.
fn build_system_with_load(
    n_stages: usize,
    n_openings: usize,
    load_mean_mw: f64,
    load_std_mw: f64,
) -> cobre_core::System {
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

    let load_models: Vec<LoadModel> = (0..n_stages)
        .map(|i| LoadModel {
            bus_id: EntityId(0),
            stage_id: i as i32,
            mean_mw: load_mean_mw,
            std_mw: load_std_mw,
        })
        .collect();

    let correlation = CorrelationModel {
        method: "spectral".to_string(),
        profiles: BTreeMap::new(),
        schedule: vec![],
    };

    SystemBuilder::new()
        .buses(vec![bus])
        .hydros(vec![hydro])
        .stages(stages)
        .inflow_models(inflow_models)
        .load_models(load_models)
        .correlation(correlation)
        .build()
        .unwrap()
}

/// Build a `StochasticContext` from a system with load models.
fn build_context_with_load(
    n_stages: usize,
    load_mean_mw: f64,
    load_std_mw: f64,
) -> StochasticContext {
    let system = build_system_with_load(n_stages, 1, load_mean_mw, load_std_mw);
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
    FutureCostFunction::new(n_stages, 1, 1, 50, &vec![0; n_stages])
}

fn iteration_limit(limit: u64) -> StoppingRuleSet {
    StoppingRuleSet {
        rules: vec![StoppingRule::IterationLimit { limit }],
        mode: StoppingMode::Any,
    }
}

// ===========================================================================
// Tests
// ===========================================================================

/// Verify that `build_stochastic_context` reports `n_load_buses() == 1` when
/// the system has a `LoadModel` with `std_mw > 0`, and `n_load_buses() == 0`
/// when `std_mw == 0.0` (deterministic load).
#[test]
fn test_stochastic_load_context_construction() {
    let stochastic_ctx = build_context_with_load(2, 500.0, 50.0);
    assert_eq!(
        stochastic_ctx.n_load_buses(),
        1,
        "n_load_buses must be 1 when std_mw=50.0 > 0"
    );

    let deterministic_ctx = build_context_with_load(2, 500.0, 0.0);
    assert_eq!(
        deterministic_ctx.n_load_buses(),
        0,
        "n_load_buses must be 0 when std_mw=0.0"
    );
}

/// Verify that a 3-iteration training run with stochastic load (`std_mw=50.0`)
/// completes successfully and produces exactly 3 lower-bound entries.
#[test]
fn test_stochastic_load_training_completes() {
    let n_stages = 2usize;
    let stochastic = build_context_with_load(n_stages, 500.0, 50.0);

    assert_eq!(
        stochastic.n_load_buses(),
        1,
        "pre-condition: n_load_buses must be 1"
    );

    let state = state_layout_for(1, 0);
    let templates = vec![minimal_template(); n_stages];
    let initial_state = vec![0.0_f64; state.n_state];
    let horizon = HorizonMode::Finite {
        num_stages: n_stages,
    };
    let risk_measures = vec![RiskMeasure::Expectation; n_stages];
    let mut fcf = make_fcf(n_stages);
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
            risk_measures,
        },
        events: EventConfig {
            event_sender: Some(tx),
            checkpoint_interval: None,
            shutdown_flag: None,
            export_states: false,
        },
    };

    // The mock solver ignores set_row_bounds, so only the slice length (n_stages)
    // matters here, not geometry_per_stage's load-balance row range.
    let load_bus_indices = vec![0usize];
    let geometry_per_stage = vec![geometry_with_load_balance(1, 1, 1); n_stages];

    let state_boxes = permissive_state_boxes(state.n_state, n_stages);
    let stage_ctx_fixture = StageContextFixture::new(&templates, &state_boxes, &geometry_per_stage)
        .load_bus_indices(&load_bus_indices);
    let stage_ctx = stage_ctx_fixture.ctx();
    let result = train(
        &mut solver,
        config,
        &mut fcf,
        &stage_ctx,
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
    .expect("train must succeed with stochastic load");

    assert_eq!(
        result.result.iterations, 3,
        "expected exactly 3 iterations, got {}",
        result.result.iterations
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

    assert_eq!(
        lower_bounds.len(),
        3,
        "expected 3 lower-bound entries (one per iteration), got {}",
        lower_bounds.len()
    );
}

/// Verify that a training run with deterministic load (`std_mw=0.0`) produces
/// `n_load_buses() == 0` and completes successfully.
///
/// With no stochastic load buses the system behaves identically to the
/// no-load-model baseline from `integration.rs`.
#[test]
fn test_deterministic_load_training_matches_baseline() {
    let n_stages = 2usize;
    let stochastic = build_context_with_load(n_stages, 500.0, 0.0);

    assert_eq!(
        stochastic.n_load_buses(),
        0,
        "pre-condition: deterministic load must yield n_load_buses=0"
    );

    let state = state_layout_for(1, 0);
    let templates = vec![minimal_template(); n_stages];
    let initial_state = vec![0.0_f64; state.n_state];
    let horizon = HorizonMode::Finite {
        num_stages: n_stages,
    };
    let risk_measures = vec![RiskMeasure::Expectation; n_stages];
    let mut fcf = make_fcf(n_stages);
    let mut solver = MockSolver::with_fixed(100.0);
    let comm = StubComm;

    let geometry = equipment_free_geometry(&vec![1usize; n_stages]);

    let state_boxes = permissive_state_boxes(state.n_state, n_stages);
    let stage_ctx_fixture = StageContextFixture::new(&templates, &state_boxes, &geometry);
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
                risk_measures,
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
    .expect("train must succeed with deterministic load");

    assert_eq!(
        result.result.iterations, 3,
        "expected exactly 3 iterations, got {}",
        result.result.iterations
    );
    assert!(
        result.result.final_lb >= 0.0,
        "final_lb must be non-negative"
    );
}

/// Verify that two training runs with identical seed=42 and stochastic load
/// configuration produce bit-for-bit identical lower-bound sequences.
#[test]
fn test_stochastic_load_seed_determinism() {
    let n_stages = 2usize;

    let run_training = || {
        let stochastic = build_context_with_load(n_stages, 500.0, 50.0);
        let state = state_layout_for(1, 0);
        let templates = vec![minimal_template(); n_stages];
        let initial_state = vec![0.0_f64; state.n_state];
        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let risk_measures = vec![RiskMeasure::Expectation; n_stages];
        let mut fcf = make_fcf(n_stages);
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
                risk_measures,
            },
            events: EventConfig {
                event_sender: Some(tx),
                checkpoint_interval: None,
                shutdown_flag: None,
                export_states: false,
            },
        };

        let load_bus_indices = vec![0usize];
        let geometry_per_stage = vec![geometry_with_load_balance(1, 1, 1); n_stages];

        let state_boxes = permissive_state_boxes(state.n_state, n_stages);
        let stage_ctx_fixture =
            StageContextFixture::new(&templates, &state_boxes, &geometry_per_stage)
                .load_bus_indices(&load_bus_indices);
        let stage_ctx = stage_ctx_fixture.ctx();
        let result = train(
            &mut solver,
            config,
            &mut fcf,
            &stage_ctx,
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
        .expect("train must succeed");

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

        (result, lower_bounds)
    };

    let (result1, lbs1) = run_training();
    let (result2, lbs2) = run_training();

    assert_eq!(
        result1.result.iterations, result2.result.iterations,
        "iteration counts must be identical: {} vs {}",
        result1.result.iterations, result2.result.iterations
    );

    assert_eq!(
        lbs1.len(),
        lbs2.len(),
        "lower-bound sequence lengths must match: {} vs {}",
        lbs1.len(),
        lbs2.len()
    );

    for (k, (lb1, lb2)) in lbs1.iter().zip(lbs2.iter()).enumerate() {
        assert_eq!(
            lb1.to_bits(),
            lb2.to_bits(),
            "lower bound at iteration {} must be bit-for-bit identical: {} vs {}",
            k + 1,
            lb1,
            lb2
        );
    }

    assert_eq!(
        result1.result.final_lb.to_bits(),
        result2.result.final_lb.to_bits(),
        "final_lb must be bit-for-bit identical: {} vs {}",
        result1.result.final_lb,
        result2.result.final_lb
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
