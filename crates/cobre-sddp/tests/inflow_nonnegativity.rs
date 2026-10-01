//! Integration tests for inflow non-negativity enforcement via the penalty method.
//!
//! All tests share a 2-hydro, 1-bus, 3-stage fixture whose PAR(0) inflow model
//! (`mean_m3s = 0.0`, `std_m3s = 30.0`, 10 openings/stage from seed 42) makes
//! roughly half of sampled noise values produce negative effective inflows —
//! the precondition the slack-value assertions rely on. `ActiveSolver` is used
//! throughout so the LP is solved and slack columns receive real primal values.

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

use std::collections::{BTreeMap, HashMap};
use std::sync::mpsc;

use chrono::NaiveDate;
use cobre_core::{
    BoundsCountsSpec, BoundsDefaults, BusStagePenalties, ContractBlockBounds, DeficitSegment,
    EntityId, HydroBlockBounds, HydroPenalties, HydroStageBounds, LineBlockBounds,
    LineStagePenalties, NcsStagePenalties, PenaltiesCountsSpec, PenaltiesDefaults,
    PumpingBlockBounds, ResolvedBounds, ResolvedPenalties, SystemBuilder, ThermalBlockBounds,
    ThermalStageBounds,
    scenario::{
        CorrelationEntity, CorrelationGroup, CorrelationModel, CorrelationProfile, SamplingScheme,
    },
    temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    },
};
use cobre_sddp::{
    Phase, ResolvedParameters, SolverProfiles, StoppingMode, StoppingRule, StoppingRuleSet,
    TrainingConfig, build_stage_templates_resolving_layout,
    config::{CutManagementConfig, EventConfig, LoopConfig},
    context::TrainingContext,
    cut::FutureCostFunction,
    energy_conversion::{EnergyConversion, EnergyConversionSet},
    horizon_mode::HorizonMode,
    hydro_models::PrepareHydroModelsResult,
    indexer::{AnticipatedPlants, CutStateProjection, StateSpace, StudyDimensions},
    inflow_method::InflowNonNegativityMethod,
    lead_time::AnticipatedResolution,
    lp::builder::{PatchBuffer, StageGeometry, StateBox},
    risk_measure::RiskMeasure,
    setup::node_graph::Traversal,
    simulate,
    simulation::{EntityCounts, SimulationConfig, SimulationOutputSpec},
    test_support::permissive_state_boxes,
    train,
    workspace::{SolverWorkspace, WorkspaceSizing},
};
use cobre_solver::ActiveSolver;
use cobre_stochastic::{
    ClassSchemes, OpeningTreeInputs, PrecomputedNormal, PrecomputedPar, StochasticContext,
    build_stochastic_context,
};

mod common;
use common::StubComm;
use common::builders::{BusSpec, HydroSpec, StageSpec, make_bus, make_hydro, make_stage};

/// External test crate cannot see parent's `test-support` surface; constructs
/// from explicit dimensions.
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

/// External test crate cannot see `test-support`; sets fields directly.
fn study_dims_for(has_inflow_penalty: bool) -> StudyDimensions {
    StudyDimensions {
        max_deficit_segments: 1,
        inflow_method: if has_inflow_penalty {
            InflowNonNegativityMethod::Penalty
        } else {
            InflowNonNegativityMethod::None
        },
        anticipated_plants: AnticipatedPlants::default(),
        downstream_par_order: 0,
    }
}

// ===========================================================================
// System and stochastic context fixture
// ===========================================================================

const N_STAGES: usize = 3;
const N_HYDROS: usize = 2;

fn build_system() -> cobre_core::System {
    use cobre_core::entities::hydro::{HydroGenerationModel, HydroPenalties};
    use cobre_core::scenario::InflowModel;

    let zero_entity_penalties = HydroPenalties {
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
    };

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

    let build_hydro = |id_val: i32, name: &str| {
        make_hydro(
            EntityId(id_val),
            HydroSpec {
                name: name.to_string(),
                operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                bus_id: EntityId(0),
                downstream_id: None,
                entry_stage_id: None,
                exit_stage_id: None,
                min_storage_hm3: 0.0,
                max_storage_hm3: 50.0,
                min_outflow_m3s: 0.0,
                max_outflow_m3s: None,
                generation_model: HydroGenerationModel::ConstantProductivity,
                min_turbined_m3s: 0.0,
                max_turbined_m3s: 50.0,
                specific_productivity_mw_per_m3s_per_m: None,
                min_generation_mw: 0.0,
                max_generation_mw: 50.0,
                tailrace: None,
                hydraulic_losses: None,
                efficiency: None,
                evaporation_coefficients_mm: None,
                evaporation_reference_volumes_hm3: None,
                diversion: None,
                filling: None,
                penalties: zero_entity_penalties,
                ..Default::default()
            },
        )
    };

    let hydros = vec![build_hydro(1, "H1"), build_hydro(2, "H2")];

    let stages: Vec<Stage> = (0..N_STAGES)
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
                        branching_factor: 10,
                        noise_method: NoiseMethod::Saa,
                    },
                    ..Default::default()
                },
            )
        })
        .collect();

    let inflow_models: Vec<InflowModel> = [EntityId(1), EntityId(2)]
        .iter()
        .flat_map(|&hydro_id| {
            (0..N_STAGES).map(move |stage_idx| InflowModel {
                hydro_id,
                stage_id: stage_idx as i32,
                mean_m3s: 0.0,
                std_m3s: 30.0,
                ar_coefficients: vec![],
                residual_std_ratio: 1.0,
                annual: None,
            })
        })
        .collect();

    let mut profiles = BTreeMap::new();
    profiles.insert(
        "default".to_string(),
        CorrelationProfile {
            groups: vec![CorrelationGroup {
                name: "g1".to_string(),
                entities: vec![
                    CorrelationEntity {
                        entity_type: "inflow".to_string(),
                        id: EntityId(1),
                    },
                    CorrelationEntity {
                        entity_type: "inflow".to_string(),
                        id: EntityId(2),
                    },
                ],
                matrix: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            }],
        },
    );
    let correlation = CorrelationModel {
        method: "spectral".to_string(),
        profiles,
        schedule: vec![],
    };

    let hydro_bounds_default = HydroStageBounds {
        min_storage_hm3: 0.0,
        max_storage_hm3: 50.0,
        filling_min_rate_m3s: 0.0,
        water_withdrawal_m3s: 0.0,
    };
    let hydro_bounds_default_block = HydroBlockBounds {
        max_turbined_m3s: 50.0,
        max_generation_mw: 50.0,
        ..Default::default()
    };
    let resolved_bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: N_HYDROS,
            n_thermals: 0,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages: N_STAGES,
            k_max: 0,
        },
        &BoundsDefaults {
            hydro: hydro_bounds_default,
            hydro_block: hydro_bounds_default_block,
            thermal: ThermalStageBounds { cost_per_mwh: 0.0 },
            thermal_block: ThermalBlockBounds {
                min_generation_mw: 0.0,
                max_generation_mw: 0.0,
            },
            line_block: LineBlockBounds {
                direct_mw: 0.0,
                reverse_mw: 0.0,
            },
            pumping_block: PumpingBlockBounds {
                min_flow_m3s: 0.0,
                max_flow_m3s: 0.0,
            },
            contract_block: ContractBlockBounds {
                min_mw: 0.0,
                max_mw: 0.0,
                price_per_mwh: 0.0,
            },
        },
    );

    let hydro_penalties_default = HydroPenalties {
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
    };
    let resolved_penalties = ResolvedPenalties::new(
        &PenaltiesCountsSpec {
            n_hydros: N_HYDROS,
            n_buses: 1,
            n_lines: 0,
            n_ncs: 0,
            n_stages: N_STAGES,
        },
        &PenaltiesDefaults {
            hydro: hydro_penalties_default,
            bus: BusStagePenalties { excess_cost: 0.0 },
            line: LineStagePenalties { exchange_cost: 0.0 },
            ncs: NcsStagePenalties {
                curtailment_cost: 0.0,
            },
        },
    );

    SystemBuilder::new()
        .buses(vec![bus])
        .hydros(hydros)
        .stages(stages)
        .inflow_models(inflow_models)
        .correlation(correlation)
        .bounds(resolved_bounds)
        .penalties(resolved_penalties)
        .build()
        .unwrap()
}

fn build_stochastic() -> StochasticContext {
    let system = build_system();
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

// ===========================================================================
// Shared fixture builder
// ===========================================================================

/// All resources needed to run training and simulation.
struct Fixture {
    stage_templates: cobre_sddp::StageTemplates,
    stochastic: StochasticContext,
    /// Production stage-0 geometry (the role-(b) equipment/slack column ranges),
    /// cloned from `stage_templates.geometry_per_stage[0]`.
    geometry: StageGeometry,
    study_dims: StudyDimensions,
    state: StateSpace,
    initial_state: Vec<f64>,
    horizon: HorizonMode,
    risk_measures: Vec<RiskMeasure>,
    entity_counts: EntityCounts,
    inflow_method: InflowNonNegativityMethod,
}

fn build_fixture() -> Fixture {
    build_fixture_with_method(InflowNonNegativityMethod::Penalty)
}

fn build_fixture_with_method(inflow_method: InflowNonNegativityMethod) -> Fixture {
    let system = build_system();

    let par_lp = PrecomputedPar::build(
        system.inflow_models(),
        &system
            .stages()
            .iter()
            .filter(|s| s.id >= 0)
            .cloned()
            .collect::<Vec<_>>(),
        &system.hydros().iter().map(|h| h.id).collect::<Vec<_>>(),
        None,
    )
    .unwrap();

    let hydro_models = PrepareHydroModelsResult::default_from_system(&system);
    let stage_templates = build_stage_templates_resolving_layout(
        &system,
        inflow_method,
        &par_lp,
        &PrecomputedNormal::default(),
        &hydro_models.production,
        &hydro_models.evaporation,
        &ResolvedParameters::default(),
    )
    .expect("no FPHA plants in integration test fixture");
    let stochastic = build_stochastic();

    let n_stages = stage_templates.templates.len();
    let n_h = system.hydros().len();
    let max_par_order = par_lp.max_order();
    let has_inflow_penalty = inflow_method.has_slack_columns() && n_h > 0;
    let study_dims = study_dims_for(has_inflow_penalty);
    let geometry = stage_templates
        .geometry_per_stage
        .first()
        .expect("at least 1 stage geometry")
        .clone();

    let state = state_layout_for(n_h, max_par_order);
    let initial_state = vec![0.0_f64; state.n_state];
    let horizon = HorizonMode::Finite {
        num_stages: n_stages,
    };
    let risk_measures = vec![RiskMeasure::Expectation; n_stages];

    let entity_counts = EntityCounts {
        hydro_ids: system.hydros().iter().map(|h| h.id.0).collect(),
        hydro_productivities: vec![1.0; system.hydros().len()],
        thermal_ids: vec![],
        line_ids: vec![],
        bus_ids: system.buses().iter().map(|b| b.id.0).collect(),
        pumping_station_ids: vec![],
        contract_ids: vec![],
        non_controllable_ids: vec![],
    };

    Fixture {
        stage_templates,
        stochastic,
        geometry,
        study_dims,
        state,
        initial_state,
        horizon,
        risk_measures,
        entity_counts,
        inflow_method,
    }
}

// ===========================================================================
// Shared test helpers
// ===========================================================================

fn base_stage_context<'a>(
    fx: &'a Fixture,
    state_boxes: &'a [StateBox],
) -> cobre_sddp::test_support::StageContextFixture<'a> {
    cobre_sddp::test_support::StageContextFixture::from_stage_templates(
        &fx.stage_templates,
        state_boxes,
    )
}

fn train_fixture(
    fx: &Fixture,
    iterations: u64,
) -> Result<cobre_sddp::TrainingOutcome, cobre_sddp::SddpError> {
    let n_stages = fx.stage_templates.templates.len();
    let mut fcf = FutureCostFunction::new(n_stages, fx.state.n_state, 1, 20, &vec![0; n_stages]);
    let mut solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");
    let comm = StubComm;

    let state_boxes = permissive_state_boxes(fx.state.n_state, n_stages);
    let stage_ctx_fixture = base_stage_context(fx, &state_boxes);
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
                stopping_rules: StoppingRuleSet {
                    rules: vec![StoppingRule::IterationLimit { limit: iterations }],
                    mode: StoppingMode::Any,
                },
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
            cut_state_layouts: &all_enabled_cut_state_layouts(&fx.state, n_stages),
            study_dims: &fx.study_dims,
            inflow_method: &fx.inflow_method,
            stochastic: &fx.stochastic,
            initial_state: &fx.initial_state,
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            stages: &[],
            lag_accum_seed: &[],
            lag_weight_seed: &[],
            dcs: None,
        },
        &comm,
        ActiveSolver::new,
        None,
        SolverProfiles::default(),
    )
}

fn simulate_fixture(
    fx: &Fixture,
    fcf: &FutureCostFunction,
) -> Result<Vec<cobre_sddp::SimulationScenarioResult>, cobre_sddp::SimulationError> {
    let (result_tx, result_rx) = mpsc::sync_channel(32);

    let collector_thread = std::thread::spawn(move || {
        let mut all_results = Vec::new();
        while let Ok(r) = result_rx.recv() {
            all_results.push(r);
        }
        all_results
    });

    let sim_training_ctx = TrainingContext {
        node_graph: &cobre_sddp::test_support::chain_node_graph(&fx.stochastic),
        horizon: &fx.horizon,
        state: &fx.state,
        cut_state_layouts: &all_enabled_cut_state_layouts(&fx.state, N_STAGES),
        study_dims: &fx.study_dims,
        inflow_method: &fx.inflow_method,
        stochastic: &fx.stochastic,
        initial_state: &fx.initial_state,
        inflow_scheme: SamplingScheme::InSample,
        load_scheme: SamplingScheme::InSample,
        ncs_scheme: SamplingScheme::InSample,
        historical_library: None,
        external_inflow_library: None,
        external_load_library: None,
        external_ncs_library: None,
        stages: &[],
        lag_accum_seed: &[],
        lag_weight_seed: &[],
        dcs: None,
    };
    let state_boxes_sim = permissive_state_boxes(fx.state.n_state, N_STAGES);
    let stage_ctx_fixture_sim = base_stage_context(fx, &state_boxes_sim);
    let stage_ctx_sim = stage_ctx_fixture_sim.ctx();
    let mut sim_workspaces = vec![SolverWorkspace::new(
        0,
        0,
        ActiveSolver::new().expect("ActiveSolver::new must succeed"),
        PatchBuffer::new(&fx.state, &[], &[]),
        &sim_training_ctx,
        &stage_ctx_sim,
        WorkspaceSizing::default(),
    )];
    let comm = StubComm;

    let zero_ec = EnergyConversion {
        equivalent_productivity_mw_per_m3s: 0.0,
        reference_volume_hm3: 0.0,
        reference_outflow_m3s: 0.0,
    };
    let ec = EnergyConversionSet::new(
        vec![vec![zero_ec; N_STAGES]; N_HYDROS],
        vec![vec![0.0_f64; N_STAGES]; N_HYDROS],
        &cobre_sddp::test_support::minimal_hydros(N_HYDROS),
        N_STAGES,
    );

    simulate(
        &mut sim_workspaces,
        &stage_ctx_sim,
        fcf,
        &sim_training_ctx,
        &SimulationConfig {
            n_scenarios: 20,
            io_channel_capacity: 32,
            profile: Phase::Simulation.profile(),
        },
        SimulationOutputSpec {
            result_tx: &result_tx,
            hydro_cell_index: &cobre_sddp::test_support::identity_hydro_cell_index(256),
            block_hours_per_stage: &fx.stage_templates.block_hours_per_stage,
            entity_counts: &fx.entity_counts,
            generic_constraint_row_entries: &fx.stage_templates.generic_constraint_row_entries,
            pumping_consumption_mw_per_m3s: &[],
            contract_prices_per_stage: &vec![Vec::new(); N_STAGES],
            contract_slots: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities_per_stage: &fx.stage_templates.hydro_productivities_per_stage,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[0.0; N_HYDROS],
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
    )?;

    drop(result_tx);
    Ok(collector_thread
        .join()
        .expect("collector thread must not panic"))
}

fn has_nonzero_slack(scenario_results: &[cobre_sddp::SimulationScenarioResult]) -> bool {
    scenario_results.iter().any(|scenario| {
        scenario.stages.iter().any(|stage| {
            stage
                .hydros
                .iter()
                .any(|h| h.inflow_nonnegativity_slack_m3s > 0.0)
        })
    })
}

// ===========================================================================
// Test 1: Penalty method prevents LP infeasibility
// ===========================================================================

/// Training with `Penalty` completes without `SddpError::Infeasible` despite the
/// fixture's negative effective inflows; the slack columns keep the LP feasible.
#[test]
fn test_penalty_method_prevents_infeasibility() {
    let fx = build_fixture();
    let result = train_fixture(&fx, 5);
    assert!(
        result.is_ok(),
        "training must succeed without SddpError::Infeasible with penalty method, got: {result:?}"
    );
}

// ===========================================================================
// Test 2: Penalty slack absorbs negative inflow in simulation
// ===========================================================================

/// With the penalty method active, simulation produces at least one
/// `SimulationHydroResult` with `inflow_nonnegativity_slack_m3s > 0.0`.
#[test]
fn test_penalty_slack_value_matches_negative_inflow() {
    let fx = build_fixture();
    let n_stages = fx.stage_templates.templates.len();
    let fcf = FutureCostFunction::new(n_stages, fx.state.n_state, 1, 20, &vec![0; n_stages]);

    train_fixture(&fx, 3).expect("training must succeed before simulation");
    let scenario_results = simulate_fixture(&fx, &fcf).expect("simulate must succeed");

    assert!(
        has_nonzero_slack(&scenario_results),
        "at least one hydro must have inflow_nonnegativity_slack_m3s > 0.0 across 20 scenarios \
         with mean_m3s=0 and std_m3s=30; none found"
    );
}

// ===========================================================================
// Test 3: Simulation slack output field is populated
// ===========================================================================

/// `SimulationHydroResult.inflow_nonnegativity_slack_m3s` is populated in
/// simulation output (non-zero in at least one hydro stage) under the penalty method.
#[test]
fn test_simulation_slack_output_populated() {
    let fx = build_fixture();
    let n_stages = fx.stage_templates.templates.len();
    let fcf = FutureCostFunction::new(n_stages, fx.state.n_state, 1, 20, &vec![0; n_stages]);

    train_fixture(&fx, 3).expect("training must succeed");
    let scenario_results = simulate_fixture(&fx, &fcf).expect("simulate must succeed");

    assert_eq!(
        scenario_results.len(),
        20,
        "expected 20 simulation results, got {}",
        scenario_results.len()
    );

    assert!(
        has_nonzero_slack(&scenario_results),
        "inflow_nonnegativity_slack_m3s must be > 0.0 in at least one hydro stage result"
    );
}

/// `TruncationWithPenalty` (clamping + penalty slack) trains end-to-end and the
/// LP template carries inflow slack columns.
#[test]
fn truncation_with_penalty_training_completes() {
    let fx = build_fixture_with_method(InflowNonNegativityMethod::TruncationWithPenalty);

    assert!(
        fx.inflow_method.has_slack_columns(),
        "TruncationWithPenalty must have slack columns"
    );

    assert!(
        !fx.geometry.inflow_slack.is_empty(),
        "geometry.inflow_slack must be non-empty for TruncationWithPenalty"
    );

    let outcome = train_fixture(&fx, 5).expect("training must succeed");
    assert!(
        outcome.error.is_none(),
        "TruncationWithPenalty: training error: {:?}",
        outcome.error
    );
    assert!(
        outcome.result.iterations <= 10,
        "TruncationWithPenalty: iterations={} (expected <= 10)",
        outcome.result.iterations
    );
}

/// Per-plant `inflow_nonnegativity_cost` (H1 = 100, H2 = 5000 R$/MWh) produces
/// distinct inflow-slack objective coefficients in the LP template: H1's equals
/// `100 * block_hours`, H2's `5000 * block_hours` (justifies the magic asserts).
#[test]
fn per_plant_inflow_penalty_differentiates_objective_coefficients() {
    let hydro_penalties_default = HydroPenalties {
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
        inflow_nonnegativity_cost: 100.0,
    };
    let mut resolved_penalties = ResolvedPenalties::new(
        &PenaltiesCountsSpec {
            n_hydros: N_HYDROS,
            n_buses: 1,
            n_lines: 0,
            n_ncs: 0,
            n_stages: N_STAGES,
        },
        &PenaltiesDefaults {
            hydro: hydro_penalties_default,
            bus: BusStagePenalties { excess_cost: 0.0 },
            line: LineStagePenalties { exchange_cost: 0.0 },
            ncs: NcsStagePenalties {
                curtailment_cost: 0.0,
            },
        },
    );
    for stage_idx in 0..N_STAGES {
        resolved_penalties
            .hydro_penalties_mut(1, stage_idx)
            .inflow_nonnegativity_cost = 5000.0;
    }

    let base_system = build_system();
    let system = SystemBuilder::new()
        .buses(base_system.buses().to_vec())
        .hydros(base_system.hydros().to_vec())
        .stages(base_system.stages().to_vec())
        .inflow_models(base_system.inflow_models().to_vec())
        .correlation(base_system.correlation().clone())
        .bounds(base_system.bounds().clone())
        .penalties(resolved_penalties)
        .build()
        .unwrap();

    let inflow_method = InflowNonNegativityMethod::Penalty;
    let par_lp = PrecomputedPar::build(
        system.inflow_models(),
        &system
            .stages()
            .iter()
            .filter(|s| s.id >= 0)
            .cloned()
            .collect::<Vec<_>>(),
        &system.hydros().iter().map(|h| h.id).collect::<Vec<_>>(),
        None,
    )
    .unwrap();
    let hydro_models = PrepareHydroModelsResult::default_from_system(&system);
    let templates = build_stage_templates_resolving_layout(
        &system,
        inflow_method,
        &par_lp,
        &PrecomputedNormal::default(),
        &hydro_models.production,
        &hydro_models.evaporation,
        &ResolvedParameters::default(),
    )
    .expect("build_stage_templates_resolving_layout must succeed");

    let tmpl0 = &templates.templates[0];
    let block_hours = 744.0_f64;

    let geometry = &templates.geometry_per_stage[0];

    assert_eq!(geometry.inflow_slack.len(), N_HYDROS);
    let h1_col = geometry.inflow_slack.start;
    let h2_col = geometry.inflow_slack.start + 1;

    let h1_obj = tmpl0.objective[h1_col];
    let h2_obj = tmpl0.objective[h2_col];

    // LP builder divides by COST_SCALE_FACTOR for conditioning.
    let cost_scale = 1_000_000.0_f64;
    let expected_h1 = 100.0 * block_hours / cost_scale;
    let expected_h2 = 5000.0 * block_hours / cost_scale;

    assert!(
        (h1_obj - expected_h1).abs() < 1e-6,
        "H1 inflow slack objective: expected {expected_h1}, got {h1_obj}"
    );
    assert!(
        (h2_obj - expected_h2).abs() < 1e-6,
        "H2 inflow slack objective: expected {expected_h2}, got {h2_obj}"
    );
    assert!(
        (h2_obj / h1_obj - 50.0).abs() < 1e-6,
        "H2/H1 objective ratio must be 50 (5000/100), got {}",
        h2_obj / h1_obj
    );
}

/// External test crate cannot see parent's `#[cfg(test)]` surface; mirrors
/// `test_support::all_enabled_cut_state_layouts` via public API.
fn all_enabled_cut_state_layouts(global: &StateSpace, n_stages: usize) -> Vec<CutStateProjection> {
    let full = StageStateConfig {
        storage: true,
        inflow_lags: true,
    };
    (0..n_stages)
        .map(|_| CutStateProjection::new(global, full))
        .collect()
}
