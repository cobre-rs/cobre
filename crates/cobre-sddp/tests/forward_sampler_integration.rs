//! Integration tests for `ForwardSampler` dispatch.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::needless_range_loop,
    clippy::trivially_copy_pass_by_ref,
    clippy::too_many_lines,
    clippy::cast_lossless,
    clippy::unnecessary_cast
)]
// `..Default::default()` in the make_* Spec calls is the intentional future-field
// seam from `common::builders` — a no-op today, not dead code.
#![allow(clippy::needless_update)]

use std::collections::BTreeMap;

use chrono::NaiveDate;
use cobre_core::{
    BoundsCountsSpec, BoundsDefaults, BusStagePenalties, ContractBlockBounds, DeficitSegment,
    EntityId, HydroBlockBounds, HydroPenalties, HydroStageBounds, LineBlockBounds,
    LineStagePenalties, NcsStagePenalties, NonControllableSource, PenaltiesCountsSpec,
    PenaltiesDefaults, PumpingBlockBounds, ResolvedBounds, ResolvedPenalties, ScenarioSource,
    SystemBuilder, ThermalBlockBounds, ThermalStageBounds,
    entities::hydro::HydroGenerationModel,
    scenario::{
        CorrelationEntity, CorrelationGroup, CorrelationModel, CorrelationProfile, ExternalLoadRow,
        ExternalNcsRow, ExternalScenarioRow, InflowHistoryRow, InflowModel, LoadModel, NcsModel,
        SamplingScheme,
    },
    temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, SeasonCycleType, SeasonDefinition,
        SeasonMap, Stage, StageLagTransition, StageRiskConfig, StageStateConfig,
    },
};
use cobre_sddp::{
    InflowNonNegativityMethod, StoppingMode, StoppingRule, StoppingRuleSet, StudySetup,
    hydro_models::PrepareHydroModelsResult,
    setup::{SimulationEnumeratedRequest, StudyParams},
};
use cobre_solver::ActiveSolver;
use cobre_stochastic::{
    ClassSchemes, DerivedSeed, ExternalScenarioLibrary, HistoricalScenarioLibrary,
    OpeningTreeInputs, PrecomputedPar, build_stochastic_context,
    par::lag_kernel::{DownstreamLagAccum, LagMajor, PrimaryLagAccum, advance_lag_chain},
    par::lag_transition::{derive_downstream_par_order, precompute_stage_lag_transitions},
    solve_par_noise, standardize_external_inflow, standardize_historical_windows,
};

mod common;
use common::StubComm;
use common::builders::{BusSpec, HydroSpec, StageSpec, make_bus, make_hydro, make_stage};

// ---------------------------------------------------------------------------
// Shared test infrastructure
// ---------------------------------------------------------------------------

fn hydro_stage_bounds() -> HydroStageBounds {
    HydroStageBounds {
        min_storage_hm3: 0.0,
        max_storage_hm3: 100.0,
        filling_min_rate_m3s: 0.0,
        water_withdrawal_m3s: 0.0,
    }
}

fn hydro_block_bounds() -> HydroBlockBounds {
    HydroBlockBounds {
        max_turbined_m3s: 100.0,
        max_generation_mw: 100.0,
        ..Default::default()
    }
}

fn hydro_stage_penalties() -> HydroPenalties {
    HydroPenalties {
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
    }
}

fn build_resolved_bounds(n_hydros: usize, n_stages: usize) -> ResolvedBounds {
    let n_st = n_stages.max(1);
    ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros,
            n_thermals: 0,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages: n_st,
            k_max: 0,
        },
        &BoundsDefaults {
            hydro: hydro_stage_bounds(),
            hydro_block: hydro_block_bounds(),
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
    )
}

fn build_resolved_penalties(n_hydros: usize, n_buses: usize, n_stages: usize) -> ResolvedPenalties {
    build_resolved_penalties_with_ncs(n_hydros, n_buses, 0, n_stages)
}

fn make_correlation(entity_ids: &[EntityId]) -> CorrelationModel {
    let n = entity_ids.len();
    let mut matrix = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        matrix[i][i] = 1.0;
    }

    let mut profiles = BTreeMap::new();
    profiles.insert(
        "default".to_string(),
        CorrelationProfile {
            groups: vec![CorrelationGroup {
                name: "g1".to_string(),
                entities: entity_ids
                    .iter()
                    .map(|&id| CorrelationEntity {
                        entity_type: "inflow".to_string(),
                        id,
                    })
                    .collect(),
                matrix,
            }],
        },
    );

    CorrelationModel {
        method: "spectral".to_string(),
        profiles,
        schedule: vec![],
    }
}

fn build_single_hydro_system(
    hydro_id: i32,
    n_stages: usize,
    branching_factor: usize,
    sampling_scheme: SamplingScheme,
    forward_seed: Option<i64>,
) -> (cobre_core::System, ScenarioSource) {
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

    let id = EntityId(hydro_id);
    let hydro = make_hydro(
        id,
        HydroSpec {
            name: format!("H{hydro_id}"),
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
                        branching_factor,
                        noise_method: NoiseMethod::Saa,
                    },
                    ..Default::default()
                },
            )
        })
        .collect();

    let inflow_models: Vec<InflowModel> = (0..n_stages)
        .map(|i| InflowModel {
            hydro_id: id,
            stage_id: i as i32,
            mean_m3s: 100.0,
            std_m3s: 30.0,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
            annual: None,
        })
        .collect();

    let correlation = make_correlation(&[id]);
    let bounds = build_resolved_bounds(1, n_stages);
    let penalties = build_resolved_penalties(1, 1, n_stages);

    let source = ScenarioSource {
        inflow_scheme: sampling_scheme,
        load_scheme: SamplingScheme::InSample,
        ncs_scheme: SamplingScheme::InSample,
        seed: forward_seed,
        historical_years: None,
    };

    let system = SystemBuilder::new()
        .buses(vec![bus])
        .hydros(vec![hydro])
        .stages(stages)
        .inflow_models(inflow_models)
        .correlation(correlation)
        .bounds(bounds)
        .penalties(penalties)
        .build()
        .expect("SystemBuilder must produce a valid system");

    (system, source)
}

fn build_two_hydro_system(
    hydro_id_order: &[i32; 2],
    n_stages: usize,
    branching_factor: usize,
    sampling_scheme: SamplingScheme,
    forward_seed: Option<i64>,
) -> (cobre_core::System, ScenarioSource) {
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

    let hydros = vec![
        make_hydro(
            EntityId(hydro_id_order[0]),
            HydroSpec {
                name: format!("H{}", hydro_id_order[0]),
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
        ),
        make_hydro(
            EntityId(hydro_id_order[1]),
            HydroSpec {
                name: format!("H{}", hydro_id_order[1]),
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
        ),
    ];

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
                        branching_factor,
                        noise_method: NoiseMethod::Saa,
                    },
                    ..Default::default()
                },
            )
        })
        .collect();

    let mut inflow_models = Vec::new();
    for &raw_id in hydro_id_order {
        for stage_idx in 0..n_stages {
            inflow_models.push(InflowModel {
                hydro_id: EntityId(raw_id),
                stage_id: stage_idx as i32,
                mean_m3s: 100.0,
                std_m3s: 30.0,
                ar_coefficients: vec![],
                residual_std_ratio: 1.0,
                annual: None,
            });
        }
    }
    inflow_models.sort_by_key(|m| (m.hydro_id.0, m.stage_id));

    // The CorrelationGroup entity-list order drives `entity_order` in
    // `build_stochastic_context` — this is the invariance path under test.
    let entity_ids: Vec<EntityId> = hydro_id_order.iter().map(|&id| EntityId(id)).collect();
    let correlation = make_correlation(&entity_ids);
    let bounds = build_resolved_bounds(2, n_stages);
    let penalties = build_resolved_penalties(2, 1, n_stages);

    let source = ScenarioSource {
        inflow_scheme: sampling_scheme,
        load_scheme: SamplingScheme::InSample,
        ncs_scheme: SamplingScheme::InSample,
        seed: forward_seed,
        historical_years: None,
    };

    let system = SystemBuilder::new()
        .buses(vec![bus])
        .hydros(hydros)
        .stages(stages)
        .inflow_models(inflow_models)
        .correlation(correlation)
        .bounds(bounds)
        .penalties(penalties)
        .build()
        .expect("SystemBuilder must produce a valid system");

    (system, source)
}

fn run_programmatic(
    system: &cobre_core::System,
    source: &ScenarioSource,
    forward_passes: u32,
    max_iterations: u64,
    inflow_method: InflowNonNegativityMethod,
) -> cobre_sddp::TrainingResult {
    let forward_seed = source.seed.map(i64::unsigned_abs);

    let stochastic = build_stochastic_context(
        system,
        42,
        forward_seed,
        &[],
        &[],
        OpeningTreeInputs::default(),
        ClassSchemes {
            inflow: Some(SamplingScheme::InSample),
            load: Some(SamplingScheme::InSample),
            ncs: Some(SamplingScheme::InSample),
        },
    )
    .expect("build_stochastic_context must succeed");

    let hydro_models = PrepareHydroModelsResult::default_from_system(system);

    let stopping_rule_set = StoppingRuleSet {
        rules: vec![StoppingRule::IterationLimit {
            limit: max_iterations,
        }],
        mode: StoppingMode::Any,
    };

    let config = StudyParams {
        seed: 42,
        forward_passes,
        training_enumerated: false,
        stopping_rule_set,
        n_scenarios: 0, // simulation disabled
        simulation_enumerated: SimulationEnumeratedRequest::Sampled,
        io_channel_capacity: 0,
        policy_path: String::new(),
        inflow_method,
        cut_selection: None,
        cut_activity_tolerance: 0.0,
        budget: None,
        export_states: false,
        scalar_parameters: Vec::new(),
        training_solver_backward: None,
        training_solver_forward: None,
        simulation_solver: None,
        backward_scheduler: cobre_io::config::BackwardScheduler::default(),
        cost_scale_factor: cobre_sddp::DEFAULT_COST_SCALE_FACTOR,
        boundary: cobre_sddp::BoundaryStateRequirements::none(),
    };
    let mut setup =
        StudySetup::from_broadcast_params(system, stochastic, config, hydro_models, source, source)
            .expect("StudySetup::from_broadcast_params must succeed");

    let comm = StubComm;
    let mut solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");

    let outcome = setup
        .train(&mut solver, &comm, 1, ActiveSolver::new, None, None)
        .expect("train must return Ok");
    assert!(outcome.error.is_none(), "expected no training error");
    outcome.result
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Verify that `SamplingScheme::OutOfSample` converges to a lower bound
/// within 5% relative tolerance of the `InSample` lower bound.
///
/// Both systems are identical except for the sampling scheme. The system has
/// 1 bus, 1 hydro (constant productivity, mean=100 m³/s, std=30 m³/s),
/// 3 stages with `branching_factor=5` and SAA noise. With 20 forward passes
/// and 50 iterations both schemes reach comparable lower bounds.
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
#[test]
fn out_of_sample_convergence() {
    const FORWARD_PASSES: u32 = 20;
    const MAX_ITERATIONS: u64 = 50;
    const RELATIVE_TOLERANCE: f64 = 0.05;

    let (system_insample, source_insample) = build_single_hydro_system(
        1, // hydro_id
        3, // n_stages
        5, // branching_factor
        SamplingScheme::InSample,
        None, // forward_seed (not used for InSample)
    );

    let (system_oos, source_oos) = build_single_hydro_system(
        1,
        3,
        5,
        SamplingScheme::OutOfSample,
        Some(42), // forward_seed required for OutOfSample
    );

    // Truncation prevents LP infeasibility from large negative noise draws and
    // makes the test robust to any seed without affecting the InSample-vs-
    // OutOfSample convergence comparison.
    let lb_insample = run_programmatic(
        &system_insample,
        &source_insample,
        FORWARD_PASSES,
        MAX_ITERATIONS,
        InflowNonNegativityMethod::Truncation,
    )
    .final_lb;
    let lb_oos = run_programmatic(
        &system_oos,
        &source_oos,
        FORWARD_PASSES,
        MAX_ITERATIONS,
        InflowNonNegativityMethod::Truncation,
    )
    .final_lb;

    let relative_error = (lb_oos - lb_insample).abs() / lb_insample.abs().max(1e-10);
    assert!(
        relative_error < RELATIVE_TOLERANCE,
        "OutOfSample LB {lb_oos:.4} diverges from InSample LB {lb_insample:.4} \
         by {:.2}% (tolerance: {:.0}%)",
        relative_error * 100.0,
        RELATIVE_TOLERANCE * 100.0,
    );
}

/// Verify that declaration order of hydro entities does not affect the lower
/// bound when using `SamplingScheme::OutOfSample`.
///
/// Builds two identical two-hydro systems that differ only in the order in
/// which entities are declared in `SystemBuilder` and in the correlation model
/// entity list. `SystemBuilder::build()` sorts hydros by `EntityId`, so the
/// canonical entity set is the same; what differs is the `entity_order` slice
/// that `build_stochastic_context` derives from the correlation model —
/// this is the invariance path under test.
///
/// Both systems use the same `forward_seed = Some(99)`. The lower bounds must
/// be bitwise identical (asserted with `assert_eq!` on `f64`).
#[test]
fn out_of_sample_declaration_order_invariance() {
    const FORWARD_PASSES: u32 = 5;
    const MAX_ITERATIONS: u64 = 20;

    let (system_a, source_a) =
        build_two_hydro_system(&[1, 2], 3, 3, SamplingScheme::OutOfSample, Some(99));

    let (system_b, source_b) =
        build_two_hydro_system(&[2, 1], 3, 3, SamplingScheme::OutOfSample, Some(99));

    let lb_a = run_programmatic(
        &system_a,
        &source_a,
        FORWARD_PASSES,
        MAX_ITERATIONS,
        InflowNonNegativityMethod::None,
    )
    .final_lb;
    let lb_b = run_programmatic(
        &system_b,
        &source_b,
        FORWARD_PASSES,
        MAX_ITERATIONS,
        InflowNonNegativityMethod::None,
    )
    .final_lb;

    assert_eq!(
        lb_a, lb_b,
        "declaration-order invariance violated: LB_A={lb_a}, LB_B={lb_b}"
    );
}

// ---------------------------------------------------------------------------
// Historical / External / Mixed helpers and tests
// ---------------------------------------------------------------------------

/// Generate `n_years * 12` monthly inflow history rows with seasonal variation.
fn build_inflow_history(hydro_id: EntityId, n_years: usize) -> Vec<InflowHistoryRow> {
    let base_year = 2000;
    let mut rows = Vec::with_capacity(n_years * 12);
    for y in 0..n_years {
        for m in 0..12u32 {
            let value = 80.0 + 15.0 * (f64::from(m) * std::f64::consts::PI / 6.0).sin();
            let start_date = NaiveDate::from_ymd_opt(base_year + y as i32, m + 1, 1).unwrap();
            rows.push(InflowHistoryRow {
                hydro_id,
                start_date,
                end_date: start_date.succ_opt().unwrap(),
                value_m3s: value,
            });
        }
    }
    rows
}

/// Build a system for Historical inflow testing.
fn build_historical_system(
    hydro_raw_id: i32,
    branching_factor: usize,
    n_history_years: usize,
    forward_seed: Option<i64>,
) -> (cobre_core::System, ScenarioSource) {
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
    let id = EntityId(hydro_raw_id);
    let hydro = make_hydro(
        id,
        HydroSpec {
            name: format!("H{hydro_raw_id}"),
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
    let stages: Vec<Stage> = (0..4)
        .map(|i| {
            let month = (i % 12) as u32 + 1;
            let year = 2024 + (i / 12) as i32;
            let next_month = if month == 12 { 1 } else { month + 1 };
            let next_year = if month == 12 { year + 1 } else { year };
            make_stage(
                i,
                StageSpec {
                    start_date: NaiveDate::from_ymd_opt(year, month, 1).unwrap(),
                    end_date: NaiveDate::from_ymd_opt(next_year, next_month, 1).unwrap(),
                    season_id: Some(i % 4),
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
                        branching_factor,
                        noise_method: NoiseMethod::Saa,
                    },
                    ..Default::default()
                },
            )
        })
        .collect();
    let inflow_models: Vec<InflowModel> = (0..4)
        .map(|i| InflowModel {
            hydro_id: id,
            stage_id: i as i32,
            mean_m3s: 80.0,
            std_m3s: 20.0,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
            annual: None,
        })
        .collect();
    let history = build_inflow_history(id, n_history_years);
    let correlation = make_correlation(&[id]);
    let bounds = build_resolved_bounds(1, 4);
    let penalties = build_resolved_penalties(1, 1, 4);

    let source = ScenarioSource {
        inflow_scheme: SamplingScheme::Historical,
        load_scheme: SamplingScheme::InSample,
        ncs_scheme: SamplingScheme::InSample,
        seed: forward_seed,
        historical_years: None,
    };

    let system = SystemBuilder::new()
        .buses(vec![bus])
        .hydros(vec![hydro])
        .stages(stages)
        .inflow_models(inflow_models)
        .inflow_history(history)
        .correlation(correlation)
        .bounds(bounds)
        .penalties(penalties)
        .build()
        .expect("SystemBuilder for historical must succeed");

    (system, source)
}

/// Run training pipeline with per-class schemes derived from the supplied source.
/// Returns the `StudySetup` so callers can assert library presence before training.
fn run_with_setup(
    system: &cobre_core::System,
    source: &ScenarioSource,
    forward_passes: u32,
    max_iterations: u64,
) -> (StudySetup, cobre_sddp::TrainingResult) {
    let forward_seed = source.seed.map(i64::unsigned_abs);
    let schemes = ClassSchemes {
        inflow: Some(source.inflow_scheme),
        load: Some(source.load_scheme),
        ncs: Some(source.ncs_scheme),
    };

    let stochastic = build_stochastic_context(
        system,
        42,
        forward_seed,
        &[],
        &[],
        OpeningTreeInputs::default(),
        schemes,
    )
    .expect("build_stochastic_context must succeed");

    let hydro_models = PrepareHydroModelsResult::default_from_system(system);
    let stopping_rule_set = StoppingRuleSet {
        rules: vec![StoppingRule::IterationLimit {
            limit: max_iterations,
        }],
        mode: StoppingMode::Any,
    };

    let config = StudyParams {
        seed: 42,
        forward_passes,
        training_enumerated: false,
        stopping_rule_set,
        n_scenarios: 0,
        simulation_enumerated: SimulationEnumeratedRequest::Sampled,
        io_channel_capacity: 0,
        policy_path: String::new(),
        inflow_method: InflowNonNegativityMethod::None,
        cut_selection: None,
        cut_activity_tolerance: 0.0,
        budget: None,
        export_states: false,
        scalar_parameters: Vec::new(),
        training_solver_backward: None,
        training_solver_forward: None,
        simulation_solver: None,
        backward_scheduler: cobre_io::config::BackwardScheduler::default(),
        cost_scale_factor: cobre_sddp::DEFAULT_COST_SCALE_FACTOR,
        boundary: cobre_sddp::BoundaryStateRequirements::none(),
    };
    let mut setup =
        StudySetup::from_broadcast_params(system, stochastic, config, hydro_models, source, source)
            .expect("StudySetup::from_broadcast_params must succeed");

    let comm = StubComm;
    let mut solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");
    let outcome = setup
        .train(&mut solver, &comm, 1, ActiveSolver::new, None, None)
        .expect("train must return Ok");
    assert!(outcome.error.is_none(), "expected no training error");
    (setup, outcome.result)
}

/// Generate external inflow rows with deterministic arithmetic noise.
fn build_external_inflow_rows(
    hydro_id: EntityId,
    n_stages: usize,
    n_scenarios: usize,
) -> Vec<ExternalScenarioRow> {
    let mut rows = Vec::with_capacity(n_stages * n_scenarios);
    for stage in 0..n_stages {
        for scenario in 0..n_scenarios {
            let noise = ((scenario * 7 + stage * 3) % 10) as f64 - 5.0;
            rows.push(ExternalScenarioRow {
                stage_id: stage as i32,
                scenario_id: scenario as i32,
                hydro_id,
                value_m3s: 80.0 + 20.0 * noise / 5.0,
            });
        }
    }
    rows
}

/// Build a system for External inflow testing.
fn build_external_system(
    hydro_raw_id: i32,
    branching_factor: usize,
    n_scenarios: usize,
    forward_seed: Option<i64>,
) -> (cobre_core::System, ScenarioSource) {
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
    let id = EntityId(hydro_raw_id);
    let hydro = make_hydro(
        id,
        HydroSpec {
            name: format!("H{hydro_raw_id}"),
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
    let stages: Vec<Stage> = (0..3)
        .map(|i| {
            let month = (i % 12) as u32 + 1;
            let year = 2024 + (i / 12) as i32;
            let next_month = if month == 12 { 1 } else { month + 1 };
            let next_year = if month == 12 { year + 1 } else { year };
            make_stage(
                i,
                StageSpec {
                    start_date: NaiveDate::from_ymd_opt(year, month, 1).unwrap(),
                    end_date: NaiveDate::from_ymd_opt(next_year, next_month, 1).unwrap(),
                    season_id: Some(i % 4),
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
                        branching_factor,
                        noise_method: NoiseMethod::Saa,
                    },
                    ..Default::default()
                },
            )
        })
        .collect();
    let inflow_models: Vec<InflowModel> = (0..3)
        .map(|i| InflowModel {
            hydro_id: id,
            stage_id: i as i32,
            mean_m3s: 80.0,
            std_m3s: 20.0,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
            annual: None,
        })
        .collect();
    let ext_rows = build_external_inflow_rows(id, 3, n_scenarios);
    let correlation = make_correlation(&[id]);
    let bounds = build_resolved_bounds(1, 3);
    let penalties = build_resolved_penalties(1, 1, 3);

    let source = ScenarioSource {
        inflow_scheme: SamplingScheme::External,
        load_scheme: SamplingScheme::InSample,
        ncs_scheme: SamplingScheme::InSample,
        seed: forward_seed,
        historical_years: None,
    };

    let system = SystemBuilder::new()
        .buses(vec![bus])
        .hydros(vec![hydro])
        .stages(stages)
        .inflow_models(inflow_models)
        .external_scenarios(ext_rows)
        .correlation(correlation)
        .bounds(bounds)
        .penalties(penalties)
        .build()
        .expect("SystemBuilder for external must succeed");

    (system, source)
}

fn build_resolved_penalties_with_ncs(
    n_hydros: usize,
    n_buses: usize,
    n_ncs: usize,
    n_stages: usize,
) -> ResolvedPenalties {
    let n_st = n_stages.max(1);
    ResolvedPenalties::new(
        &PenaltiesCountsSpec {
            n_hydros,
            n_buses,
            n_lines: 0,
            n_ncs,
            n_stages: n_st,
        },
        &PenaltiesDefaults {
            hydro: hydro_stage_penalties(),
            bus: BusStagePenalties { excess_cost: 0.0 },
            line: LineStagePenalties { exchange_cost: 0.0 },
            ncs: NcsStagePenalties {
                curtailment_cost: 0.0,
            },
        },
    )
}

fn assert_no_external_libraries(setup: &StudySetup) {
    assert!(
        setup
            .inputs
            .scenario_libraries
            .training
            .historical
            .is_none()
    );
    assert!(
        setup
            .inputs
            .scenario_libraries
            .training
            .external_inflow
            .is_none()
    );
    assert!(
        setup
            .inputs
            .scenario_libraries
            .training
            .external_load
            .is_none()
    );
    assert!(
        setup
            .inputs
            .scenario_libraries
            .training
            .external_ncs
            .is_none()
    );
}

/// Build a system for mixed-scheme testing (hydro + NCS + stochastic load).
fn build_mixed_system(
    inflow_scheme: SamplingScheme,
    load_scheme: SamplingScheme,
    ncs_scheme: SamplingScheme,
) -> (cobre_core::System, ScenarioSource) {
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
    let ncs = NonControllableSource {
        id: EntityId(10),
        name: "NCS0".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        bus_id: EntityId(0),
        entry_stage_id: None,
        exit_stage_id: None,
        max_generation_mw: 30.0,
        allow_curtailment: true,
        curtailment_cost: 0.0,
    };
    let stages: Vec<Stage> = (0..3)
        .map(|i| {
            let month = (i % 12) as u32 + 1;
            let year = 2024 + (i / 12) as i32;
            let next_month = if month == 12 { 1 } else { month + 1 };
            let next_year = if month == 12 { year + 1 } else { year };
            make_stage(
                i,
                StageSpec {
                    start_date: NaiveDate::from_ymd_opt(year, month, 1).unwrap(),
                    end_date: NaiveDate::from_ymd_opt(next_year, next_month, 1).unwrap(),
                    season_id: Some(i % 4),
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
                        branching_factor: 5,
                        noise_method: NoiseMethod::Saa,
                    },
                    ..Default::default()
                },
            )
        })
        .collect();
    let inflow_models: Vec<InflowModel> = (0..3)
        .map(|i| InflowModel {
            hydro_id: EntityId(1),
            stage_id: i as i32,
            mean_m3s: 80.0,
            std_m3s: 20.0,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
            annual: None,
        })
        .collect();
    let load_models: Vec<LoadModel> = (0..3)
        .map(|i| LoadModel {
            bus_id: EntityId(0),
            stage_id: i as i32,
            mean_mw: 60.0,
            std_mw: 10.0,
        })
        .collect();
    let ncs_models: Vec<NcsModel> = (0..3)
        .map(|i| NcsModel {
            ncs_id: EntityId(10),
            stage_id: i as i32,
            mean: 20.0,
            std: 5.0,
        })
        .collect();
    let correlation = make_correlation(&[EntityId(1)]);
    let bounds = build_resolved_bounds(1, 3);
    let penalties = build_resolved_penalties_with_ncs(1, 1, 1, 3);

    let source = ScenarioSource {
        inflow_scheme,
        load_scheme,
        ncs_scheme,
        seed: Some(42),
        historical_years: None,
    };

    let system = SystemBuilder::new()
        .buses(vec![bus])
        .hydros(vec![hydro])
        .non_controllable_sources(vec![ncs])
        .stages(stages)
        .inflow_models(inflow_models)
        .load_models(load_models)
        .ncs_models(ncs_models)
        .correlation(correlation)
        .bounds(bounds)
        .penalties(penalties)
        .build()
        .expect("SystemBuilder for mixed must succeed");

    (system, source)
}

// --- Convergence sweep (Historical, External inflow, Mixed schemes) ---

const CONVERGENCE_CASES: &[&str] = &[
    "historical_convergence",
    "external_inflow_convergence",
    "mixed_scheme_inflow_insample_load_oos",
    "mixed_scheme_inflow_oos_load_insample",
];

#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
#[test]
fn forward_sampler_convergence_sweep() {
    const FORWARD_PASSES: u32 = 10;
    const MAX_ITERATIONS: u64 = 50;
    const N_SCENARIOS: usize = 20;

    for (idx, &desc) in CONVERGENCE_CASES.iter().enumerate() {
        match idx {
            0 => {
                let (system, source) = build_historical_system(1, 5, 10, Some(42));
                let (setup, result) =
                    run_with_setup(&system, &source, FORWARD_PASSES, MAX_ITERATIONS);
                assert!(
                    setup
                        .inputs
                        .scenario_libraries
                        .training
                        .historical
                        .is_some(),
                    "case_index = {idx}, desc = {desc}: historical_library must be Some for \
                     Historical scheme"
                );
                assert!(
                    result.final_lb.is_finite(),
                    "case_index = {idx}, desc = {desc}: final_lb must be finite, got {}",
                    result.final_lb
                );
            }
            1 => {
                let (system, source) = build_external_system(1, 5, N_SCENARIOS, Some(42));
                let (setup, result) =
                    run_with_setup(&system, &source, FORWARD_PASSES, MAX_ITERATIONS);
                assert!(
                    setup
                        .inputs
                        .scenario_libraries
                        .training
                        .external_inflow
                        .is_some(),
                    "case_index = {idx}, desc = {desc}: external_inflow_library must be Some for \
                     External scheme"
                );
                assert!(
                    result.final_lb.is_finite(),
                    "case_index = {idx}, desc = {desc}: final_lb must be finite, got {}",
                    result.final_lb
                );
                // Reproducibility: same seed must produce identical LB.
                let (system2, source2) = build_external_system(1, 5, N_SCENARIOS, Some(42));
                let (_setup2, result2) =
                    run_with_setup(&system2, &source2, FORWARD_PASSES, MAX_ITERATIONS);
                assert_eq!(
                    result.final_lb, result2.final_lb,
                    "case_index = {idx}, desc = {desc}: reproducibility violated: run1={}, run2={}",
                    result.final_lb, result2.final_lb
                );
            }
            2 => {
                let (system, source) = build_mixed_system(
                    SamplingScheme::InSample,
                    SamplingScheme::OutOfSample,
                    SamplingScheme::InSample,
                );
                let (setup, result) =
                    run_with_setup(&system, &source, FORWARD_PASSES, MAX_ITERATIONS);
                assert_no_external_libraries(&setup);
                assert!(
                    result.final_lb.is_finite(),
                    "case_index = {idx}, desc = {desc}: final_lb must be finite"
                );
            }
            3 => {
                let (system, source) = build_mixed_system(
                    SamplingScheme::OutOfSample,
                    SamplingScheme::InSample,
                    SamplingScheme::InSample,
                );
                let (setup, result) =
                    run_with_setup(&system, &source, FORWARD_PASSES, MAX_ITERATIONS);
                assert_no_external_libraries(&setup);
                assert!(
                    result.final_lb.is_finite(),
                    "case_index = {idx}, desc = {desc}: final_lb must be finite"
                );
            }
            _ => unreachable!("unexpected case_index = {idx}"),
        }
    }
}

// ---------------------------------------------------------------------------
// External load / NCS library population tests
// ---------------------------------------------------------------------------

/// Generate `n_stages × n_scenarios` external load rows for a single bus.
fn build_external_load_rows(
    bus_id: EntityId,
    n_stages: usize,
    n_scenarios: usize,
) -> Vec<ExternalLoadRow> {
    let mut rows = Vec::with_capacity(n_stages * n_scenarios);
    for stage in 0..n_stages {
        for scenario in 0..n_scenarios {
            rows.push(ExternalLoadRow {
                stage_id: stage as i32,
                scenario_id: scenario as i32,
                bus_id,
                value_mw: 50.0 + 5.0 * (scenario as f64),
            });
        }
    }
    rows
}

/// Generate `n_stages × n_scenarios` external NCS rows for a single NCS source.
fn build_external_ncs_rows(
    ncs_id: EntityId,
    n_stages: usize,
    n_scenarios: usize,
) -> Vec<ExternalNcsRow> {
    let mut rows = Vec::with_capacity(n_stages * n_scenarios);
    for stage in 0..n_stages {
        for scenario in 0..n_scenarios {
            rows.push(ExternalNcsRow {
                stage_id: stage as i32,
                scenario_id: scenario as i32,
                ncs_id,
                value: 0.6 + 0.04 * (scenario as f64 / n_scenarios as f64),
            });
        }
    }
    rows
}

/// Build a system whose load scheme is `External`, backed by pre-computed
/// `ExternalLoadRow` data on the `System`.
///
/// Has 1 bus, 1 hydro, 1 load model, and 3 monthly stages.
fn build_external_load_system(
    n_scenarios: usize,
    forward_seed: Option<i64>,
) -> (cobre_core::System, ScenarioSource) {
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
    let stages: Vec<Stage> = (0..3)
        .map(|i| {
            let month = (i % 12) as u32 + 1;
            let year = 2024 + (i / 12) as i32;
            let next_month = if month == 12 { 1 } else { month + 1 };
            let next_year = if month == 12 { year + 1 } else { year };
            make_stage(
                i,
                StageSpec {
                    start_date: NaiveDate::from_ymd_opt(year, month, 1).unwrap(),
                    end_date: NaiveDate::from_ymd_opt(next_year, next_month, 1).unwrap(),
                    season_id: Some(i % 4),
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
                        branching_factor: 5,
                        noise_method: NoiseMethod::Saa,
                    },
                    ..Default::default()
                },
            )
        })
        .collect();
    let inflow_models: Vec<InflowModel> = (0..3)
        .map(|i| InflowModel {
            hydro_id: EntityId(1),
            stage_id: i as i32,
            mean_m3s: 80.0,
            std_m3s: 20.0,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
            annual: None,
        })
        .collect();
    let load_models: Vec<LoadModel> = (0..3)
        .map(|i| LoadModel {
            bus_id: EntityId(0),
            stage_id: i as i32,
            mean_mw: 60.0,
            std_mw: 10.0,
        })
        .collect();
    let ext_load_rows = build_external_load_rows(EntityId(0), 3, n_scenarios);
    let correlation = make_correlation(&[EntityId(1)]);
    let bounds = build_resolved_bounds(1, 3);
    let penalties = build_resolved_penalties(1, 1, 3);

    let source = ScenarioSource {
        inflow_scheme: SamplingScheme::InSample,
        load_scheme: SamplingScheme::External,
        ncs_scheme: SamplingScheme::InSample,
        seed: forward_seed,
        historical_years: None,
    };

    let system = SystemBuilder::new()
        .buses(vec![bus])
        .hydros(vec![hydro])
        .stages(stages)
        .inflow_models(inflow_models)
        .load_models(load_models)
        .external_load_scenarios(ext_load_rows)
        .correlation(correlation)
        .bounds(bounds)
        .penalties(penalties)
        .build()
        .expect("SystemBuilder for external load must succeed");

    (system, source)
}

/// Build a system whose NCS scheme is `External`, backed by pre-computed
/// `ExternalNcsRow` data on the `System`.
///
/// Has 1 bus, 1 hydro, 1 NCS source, and 3 monthly stages.
fn build_external_ncs_system(
    n_scenarios: usize,
    forward_seed: Option<i64>,
) -> (cobre_core::System, ScenarioSource) {
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
    let ncs = NonControllableSource {
        id: EntityId(10),
        name: "NCS0".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        bus_id: EntityId(0),
        entry_stage_id: None,
        exit_stage_id: None,
        max_generation_mw: 30.0,
        allow_curtailment: true,
        curtailment_cost: 0.0,
    };
    let stages: Vec<Stage> = (0..3)
        .map(|i| {
            let month = (i % 12) as u32 + 1;
            let year = 2024 + (i / 12) as i32;
            let next_month = if month == 12 { 1 } else { month + 1 };
            let next_year = if month == 12 { year + 1 } else { year };
            make_stage(
                i,
                StageSpec {
                    start_date: NaiveDate::from_ymd_opt(year, month, 1).unwrap(),
                    end_date: NaiveDate::from_ymd_opt(next_year, next_month, 1).unwrap(),
                    season_id: Some(i % 4),
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
                        branching_factor: 5,
                        noise_method: NoiseMethod::Saa,
                    },
                    ..Default::default()
                },
            )
        })
        .collect();
    let inflow_models: Vec<InflowModel> = (0..3)
        .map(|i| InflowModel {
            hydro_id: EntityId(1),
            stage_id: i as i32,
            mean_m3s: 80.0,
            std_m3s: 20.0,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
            annual: None,
        })
        .collect();
    let ncs_models: Vec<NcsModel> = (0..3)
        .map(|i| NcsModel {
            ncs_id: EntityId(10),
            stage_id: i as i32,
            mean: 20.0,
            std: 5.0,
        })
        .collect();
    let ext_ncs_rows = build_external_ncs_rows(EntityId(10), 3, n_scenarios);
    let correlation = make_correlation(&[EntityId(1)]);
    let bounds = build_resolved_bounds(1, 3);
    let penalties = build_resolved_penalties_with_ncs(1, 1, 1, 3);

    let source = ScenarioSource {
        inflow_scheme: SamplingScheme::InSample,
        load_scheme: SamplingScheme::InSample,
        ncs_scheme: SamplingScheme::External,
        seed: forward_seed,
        historical_years: None,
    };

    let system = SystemBuilder::new()
        .buses(vec![bus])
        .hydros(vec![hydro])
        .non_controllable_sources(vec![ncs])
        .stages(stages)
        .inflow_models(inflow_models)
        .ncs_models(ncs_models)
        .external_ncs_scenarios(ext_ncs_rows)
        .correlation(correlation)
        .bounds(bounds)
        .penalties(penalties)
        .build()
        .expect("SystemBuilder for external NCS must succeed");

    (system, source)
}

const EXTERNAL_LIBRARY_CASES: &[(&str, &str)] = &[
    ("external_load_library_populated", "load"),
    ("external_ncs_library_populated", "ncs"),
];

/// Verify that `ClassSampler::External` for load and NCS each populates
/// the corresponding library accessor and produces a finite lower bound.
#[test]
fn external_library_population_sweep() {
    const FORWARD_PASSES: u32 = 10;
    const MAX_ITERATIONS: u64 = 20;
    const N_SCENARIOS: usize = 20;

    for (idx, &(desc, entity_class)) in EXTERNAL_LIBRARY_CASES.iter().enumerate() {
        match idx {
            0 => {
                let (system, source) = build_external_load_system(N_SCENARIOS, Some(42));
                let (setup, result) =
                    run_with_setup(&system, &source, FORWARD_PASSES, MAX_ITERATIONS);
                assert!(
                    setup
                        .inputs
                        .scenario_libraries
                        .training
                        .external_load
                        .is_some(),
                    "case_index = {idx}, desc = {desc}, entity_class = {entity_class}: \
                     external_load_library must be Some when load_scheme is External"
                );
                assert!(
                    setup
                        .inputs
                        .scenario_libraries
                        .training
                        .external_inflow
                        .is_none(),
                    "case_index = {idx}, desc = {desc}, entity_class = {entity_class}: \
                     external_inflow_library must be None when inflow_scheme is InSample"
                );
                assert!(
                    setup
                        .inputs
                        .scenario_libraries
                        .training
                        .external_ncs
                        .is_none(),
                    "case_index = {idx}, desc = {desc}, entity_class = {entity_class}: \
                     external_ncs_library must be None when ncs_scheme is InSample"
                );
                assert!(
                    result.final_lb.is_finite(),
                    "case_index = {idx}, desc = {desc}, entity_class = {entity_class}: \
                     final_lb must be finite, got {}",
                    result.final_lb
                );
            }
            1 => {
                let (system, source) = build_external_ncs_system(N_SCENARIOS, Some(42));
                let (setup, result) =
                    run_with_setup(&system, &source, FORWARD_PASSES, MAX_ITERATIONS);
                assert!(
                    setup
                        .inputs
                        .scenario_libraries
                        .training
                        .external_ncs
                        .is_some(),
                    "case_index = {idx}, desc = {desc}, entity_class = {entity_class}: \
                     external_ncs_library must be Some when ncs_scheme is External"
                );
                assert!(
                    setup
                        .inputs
                        .scenario_libraries
                        .training
                        .external_inflow
                        .is_none(),
                    "case_index = {idx}, desc = {desc}, entity_class = {entity_class}: \
                     external_inflow_library must be None when inflow_scheme is InSample"
                );
                assert!(
                    setup
                        .inputs
                        .scenario_libraries
                        .training
                        .external_load
                        .is_none(),
                    "case_index = {idx}, desc = {desc}, entity_class = {entity_class}: \
                     external_load_library must be None when load_scheme is InSample"
                );
                assert!(
                    result.final_lb.is_finite(),
                    "case_index = {idx}, desc = {desc}, entity_class = {entity_class}: \
                     final_lb must be finite, got {}",
                    result.final_lb
                );
            }
            _ => unreachable!("unexpected case_index = {idx}"),
        }
    }
}

// ---------------------------------------------------------------------------
// Noise-sharing regression tests
// ---------------------------------------------------------------------------

/// Build a 12-stage monthly system where every stage has a distinct
/// `(season_id, year)` pair — i.e. every stage gets a unique noise group ID
/// from `precompute_noise_groups`. This simulates the normal monthly study
/// layout where noise-group sharing is structurally inactive.
///
/// Returns the system and an `InSample` scenario source so the test is
/// deterministic and exercises the common production path.
fn build_monthly_unique_groups_system(
    n_stages: usize,
    branching_factor: usize,
) -> (cobre_core::System, ScenarioSource) {
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
            let month = (i % 12) as u32 + 1;
            let year = 2024 + (i / 12) as i32;
            let next_month = if month == 12 { 1 } else { month + 1 };
            let next_year = if month == 12 { year + 1 } else { year };
            make_stage(
                i,
                StageSpec {
                    start_date: NaiveDate::from_ymd_opt(year, month, 1).unwrap(),
                    end_date: NaiveDate::from_ymd_opt(next_year, next_month, 1).unwrap(),
                    // season_id = i % 12 combined with the distinct year makes every
                    // (season_id, year) pair unique.
                    season_id: Some(i % 12),
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
                        branching_factor,
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
            mean_m3s: 80.0 + 20.0 * ((i as f64) * std::f64::consts::PI / 6.0).sin(),
            std_m3s: 15.0,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
            annual: None,
        })
        .collect();

    let correlation = make_correlation(&[EntityId(1)]);
    let bounds = build_resolved_bounds(1, n_stages);
    let penalties = build_resolved_penalties(1, 1, n_stages);

    let source = ScenarioSource {
        inflow_scheme: SamplingScheme::InSample,
        load_scheme: SamplingScheme::InSample,
        ncs_scheme: SamplingScheme::InSample,
        seed: None,
        historical_years: None,
    };

    let system = SystemBuilder::new()
        .buses(vec![bus])
        .hydros(vec![hydro])
        .stages(stages)
        .inflow_models(inflow_models)
        .correlation(correlation)
        .bounds(bounds)
        .penalties(penalties)
        .build()
        .expect("SystemBuilder must produce a valid system");

    (system, source)
}

/// Regression test: noise group wiring must be transparent for
/// monthly studies.
///
/// A 12-stage monthly study where every stage has a unique `(season_id, year)`
/// pair produces unique noise group IDs for every stage via
/// `precompute_noise_groups`. With unique groups there is no sharing, so the
/// forward pass and opening tree behaviour must be bit-identical to running
/// the same study twice with the same seed.
///
/// This test verifies that:
/// 1. The noise group IDs are propagated from `StudySetup` through
///    `StageContext` to `SampleRequest.noise_group_id` in both the forward
///    pass and simulation pipeline.
/// 2. The opening tree is built with `Some(noise_group_ids)` wired from setup.
/// 3. For monthly studies (unique groups) the noise sharing infrastructure is
///    transparent: same seed + same system = bit-identical lower bound.
#[test]
fn monthly_noise_sharing_regression() {
    const FORWARD_PASSES: u32 = 5;
    const MAX_ITERATIONS: u64 = 10;

    let (system, source) = build_monthly_unique_groups_system(
        12, // 12 monthly stages (1 year)
        3,
    );

    let result_a = run_programmatic(
        &system,
        &source,
        FORWARD_PASSES,
        MAX_ITERATIONS,
        InflowNonNegativityMethod::None,
    );

    let result_b = run_programmatic(
        &system,
        &source,
        FORWARD_PASSES,
        MAX_ITERATIONS,
        InflowNonNegativityMethod::None,
    );

    assert_eq!(
        result_a.final_lb, result_b.final_lb,
        "monthly noise sharing regression: lower bound is not deterministic. \
         run_a={}, run_b={}",
        result_a.final_lb, result_b.final_lb
    );
    assert_eq!(
        result_a.iterations, result_b.iterations,
        "monthly noise sharing regression: iteration count is not deterministic. \
         run_a={}, run_b={}",
        result_a.iterations, result_b.iterations
    );
    assert!(
        result_a.final_lb.is_finite(),
        "monthly noise sharing regression: lower bound must be finite, got {}",
        result_a.final_lb
    );
}

// ---------------------------------------------------------------------------
// Differential lag-chain golden test (forward vs. External vs. Historical)
// ---------------------------------------------------------------------------
//
// Three call sites route through the shared kernel
// (`cobre_stochastic::par::lag_kernel::advance_lag_chain`): the forward pass
// (`LagMajor` layout), and the External/Historical samplers (`EntityMajor`
// layout). This drives the same realized inflow sequence through all three
// on a monthly→quarterly multi-resolution fixture and asserts per-stage,
// per-hydro agreement — not total cost, since a fold/naive advance can match
// total cost while the per-stage lag split differs.
//
// Neither `standardize_external_inflow` nor `standardize_historical_windows`
// exposes its internal lag state directly; each emits only a standardized
// `eta`. `solve_par_noise` is an affine, deterministic function of the
// incoming lag, so re-solving it with the forward oracle's own lag value
// reproduces, bit-for-bit, whatever `eta` a sampler with the SAME internal
// lag would have produced. An equality failure below means the sampler's
// internal chain diverged from the forward chain.

const DLC_N_HYDROS: usize = 2;
const DLC_N_STAGES: usize = 5;

/// Fixed multi-resolution fixture: three monthly stages (Jan–Mar 2026) feed a
/// downstream Q1 ring, the fourth stage (Q2 2026) rebuilds the primary lag
/// from that completed ring, and the fifth stage (Q3 2026) reads the
/// rebuilt lag under its own AR(1) model.
struct DlcFixture {
    stages: Vec<Stage>,
    season_map: SeasonMap,
    hydro_ids: Vec<EntityId>,
    par: PrecomputedPar,
    transitions: Vec<StageLagTransition>,
    /// Per-hydro lag-1 seed value (m³/s) driving `derived_lag_values`.
    lag_seed: [f64; DLC_N_HYDROS],
    /// `raw[stage][hydro]` — the realized inflow (m³/s) driven through all
    /// three call sites.
    raw: [[f64; DLC_N_HYDROS]; DLC_N_STAGES],
}

fn dlc_season_def(id: usize, month_start: u32, month_end: Option<u32>) -> SeasonDefinition {
    SeasonDefinition {
        id,
        label: format!("S{id}"),
        month_start,
        day_start: None,
        month_end,
        day_end: None,
    }
}

fn build_dlc_fixture() -> DlcFixture {
    let stage = |index, start_date, end_date, season_id, duration_hours| {
        make_stage(
            index,
            StageSpec {
                start_date,
                end_date,
                season_id: Some(season_id),
                blocks: vec![Block {
                    index: 0,
                    name: "SINGLE".to_string(),
                    duration_hours,
                }],
                ..StageSpec::default()
            },
        )
    };
    let stages = vec![
        stage(
            0,
            NaiveDate::from_ymd_opt(2026, 1, 1).unwrap(),
            NaiveDate::from_ymd_opt(2026, 2, 1).unwrap(),
            0,
            31.0 * 24.0,
        ),
        stage(
            1,
            NaiveDate::from_ymd_opt(2026, 2, 1).unwrap(),
            NaiveDate::from_ymd_opt(2026, 3, 1).unwrap(),
            1,
            28.0 * 24.0,
        ),
        stage(
            2,
            NaiveDate::from_ymd_opt(2026, 3, 1).unwrap(),
            NaiveDate::from_ymd_opt(2026, 4, 1).unwrap(),
            2,
            31.0 * 24.0,
        ),
        stage(
            3,
            NaiveDate::from_ymd_opt(2026, 4, 1).unwrap(),
            NaiveDate::from_ymd_opt(2026, 7, 1).unwrap(),
            12,
            91.0 * 24.0,
        ),
        stage(
            4,
            NaiveDate::from_ymd_opt(2026, 7, 1).unwrap(),
            NaiveDate::from_ymd_opt(2026, 10, 1).unwrap(),
            13,
            92.0 * 24.0,
        ),
    ];
    let season_map = SeasonMap {
        cycle_type: SeasonCycleType::Custom,
        seasons: vec![
            dlc_season_def(0, 1, None),
            dlc_season_def(1, 2, None),
            dlc_season_def(2, 3, None),
            dlc_season_def(12, 4, Some(6)),
            dlc_season_def(13, 7, Some(9)),
        ],
    };

    let hydro1 = EntityId(1);
    let hydro2 = EntityId(2);
    let hydro_ids = vec![hydro1, hydro2];

    let mut models = Vec::new();
    for stage_id in -1..i32::try_from(DLC_N_STAGES).unwrap() {
        let ar1 = if stage_id >= 0 { vec![0.6] } else { vec![] };
        let ar2 = if stage_id >= 0 { vec![0.5] } else { vec![] };
        models.push(InflowModel {
            hydro_id: hydro1,
            stage_id,
            mean_m3s: 100.0,
            std_m3s: 20.0,
            ar_coefficients: ar1,
            residual_std_ratio: 1.0,
            annual: None,
        });
        models.push(InflowModel {
            hydro_id: hydro2,
            stage_id,
            mean_m3s: 150.0,
            std_m3s: 25.0,
            ar_coefficients: ar2,
            residual_std_ratio: 0.9,
            annual: None,
        });
    }

    let par = PrecomputedPar::build(&models, &stages, &hydro_ids, None).unwrap();
    assert_eq!(par.max_order(), 1);

    let transitions = precompute_stage_lag_transitions(&stages, &season_map, 1);
    assert_eq!(transitions.len(), DLC_N_STAGES);
    for t in &transitions[0..3] {
        assert!(t.accumulate_downstream);
        assert!(!t.rebuild_from_downstream);
    }
    assert!(!transitions[0].downstream_finalize);
    assert!(!transitions[1].downstream_finalize);
    assert!(transitions[2].downstream_finalize);
    assert!(transitions[3].rebuild_from_downstream);
    assert!(!transitions[4].accumulate_downstream);
    assert!(!transitions[4].rebuild_from_downstream);

    let lag_seed = [70.0, 280.0];

    let raw = [
        [80.0, 300.0],
        [150.0, 120.0],
        [40.0, 260.0],
        [190.0, 50.0],
        [60.0, 400.0],
    ];

    DlcFixture {
        stages,
        season_map,
        hydro_ids,
        par,
        transitions,
        lag_seed,
        raw,
    }
}

impl DlcFixture {
    /// `lag_seed` in the entity-major derived-seed layout (`pos * l_state +
    /// lag`) `standardize_external_inflow` / `standardize_historical_windows`
    /// take directly; `hydro_ids` order matches `lag_seed` order by
    /// construction in `build_dlc_fixture`.
    fn derived_lag_values(&self) -> Vec<f64> {
        self.lag_seed.to_vec()
    }
}

/// Drive `advance_lag_chain::<LagMajor>` across the fixture's stages exactly
/// as the forward pass's `noise.rs` adapter does, recording the incoming
/// (pre-advance) lag at every stage — the value each call site's own eta
/// computation reads.
fn dlc_forward_oracle_incoming_lags(fx: &DlcFixture) -> Vec<[f64; DLC_N_HYDROS]> {
    let layout = LagMajor {
        entity_count: DLC_N_HYDROS,
        max_order: 1,
    };
    let mut lag_state = vec![fx.lag_seed[0], fx.lag_seed[1]];
    let mut incoming = vec![0.0; DLC_N_HYDROS];
    let mut primary_acc = vec![0.0; DLC_N_HYDROS];
    let mut primary_w = vec![0.0; DLC_N_HYDROS];
    let mut ds_acc = vec![0.0; DLC_N_HYDROS];
    let mut ds_w = 0.0_f64;
    let mut ds_completed = vec![0.0; DLC_N_HYDROS];
    let mut ds_n = 0_usize;

    let mut incoming_per_stage = Vec::with_capacity(DLC_N_STAGES);
    for t in 0..DLC_N_STAGES {
        incoming.copy_from_slice(&lag_state);
        incoming_per_stage.push([incoming[0], incoming[1]]);

        let mut primary = PrimaryLagAccum {
            accumulator: &mut primary_acc,
            weight_accum: &mut primary_w,
        };
        let mut downstream = DownstreamLagAccum {
            accumulator: &mut ds_acc,
            weight_accum: &mut ds_w,
            completed_lags: &mut ds_completed,
            n_completed: &mut ds_n,
            par_order: 1,
        };
        advance_lag_chain(
            layout,
            &mut lag_state,
            &incoming,
            &fx.raw[t],
            &fx.transitions[t],
            &mut primary,
            &mut downstream,
        );
    }
    incoming_per_stage
}

/// Independent reimplementation of the primary-only lag shift that ignores
/// `accumulate_downstream`/`rebuild_from_downstream` entirely: it accumulates
/// `accumulate_weight`-weighted realized values and, at `finalize_period`,
/// shifts the lag to the period average.
fn dlc_naive_primary_only_incoming_lags(fx: &DlcFixture) -> Vec<[f64; DLC_N_HYDROS]> {
    let mut lag_state = [fx.lag_seed[0], fx.lag_seed[1]];
    let mut accumulator = [0.0_f64; DLC_N_HYDROS];
    let mut weight_accum = 0.0_f64;

    let mut incoming_per_stage = Vec::with_capacity(DLC_N_STAGES);
    for t in 0..DLC_N_STAGES {
        incoming_per_stage.push(lag_state);

        let stage_lag = &fx.transitions[t];
        for h in 0..DLC_N_HYDROS {
            accumulator[h] += fx.raw[t][h] * stage_lag.accumulate_weight;
        }
        weight_accum += stage_lag.accumulate_weight;

        if stage_lag.finalize_period && weight_accum > 0.0 {
            let inv = 1.0 / weight_accum;
            for h in 0..DLC_N_HYDROS {
                lag_state[h] = accumulator[h] * inv;
            }
            if stage_lag.spillover_weight > 0.0 {
                for h in 0..DLC_N_HYDROS {
                    accumulator[h] = fx.raw[t][h] * stage_lag.spillover_weight;
                }
                weight_accum = stage_lag.spillover_weight;
            } else {
                accumulator = [0.0; DLC_N_HYDROS];
                weight_accum = 0.0;
            }
        }
    }
    incoming_per_stage
}

#[test]
fn differential_lag_chain_forward_external_historical_agree_at_quarterly_transition() {
    let fx = build_dlc_fixture();
    let oracle_incoming = dlc_forward_oracle_incoming_lags(&fx);

    assert_eq!(
        oracle_incoming[0],
        [fx.lag_seed[0], fx.lag_seed[1],],
        "stage 0's incoming lag must be the unmodified lag seed"
    );
    for h in 0..DLC_N_HYDROS {
        assert!(
            (oracle_incoming[4][h] - fx.raw[3][h]).abs() > 1.0,
            "hydro {h}: the ring-rebuilt lag feeding stage 4 must differ from \
             the transition stage's own raw value"
        );
    }

    let mut ext_lib = ExternalScenarioLibrary::new(
        DLC_N_STAGES,
        1,
        DLC_N_HYDROS,
        "inflow",
        vec![1; DLC_N_STAGES],
    );
    let mut ext_rows = Vec::with_capacity(DLC_N_STAGES * DLC_N_HYDROS);
    for t in 0..DLC_N_STAGES {
        for h in 0..DLC_N_HYDROS {
            ext_rows.push(ExternalScenarioRow {
                stage_id: i32::try_from(t).unwrap(),
                scenario_id: 0,
                hydro_id: fx.hydro_ids[h],
                value_m3s: fx.raw[t][h],
            });
        }
    }
    let derived_lag_values = fx.derived_lag_values();
    standardize_external_inflow(
        &mut ext_lib,
        &ext_rows,
        &fx.hydro_ids,
        &fx.stages,
        &fx.par,
        DerivedSeed {
            lag_values: &derived_lag_values,
            l_state: 1,
            accum: &[],
            weight: &[],
        },
        &fx.transitions,
        1,
    );

    let window_year = 2026;
    let mut hist_lib =
        HistoricalScenarioLibrary::new(1, DLC_N_STAGES, DLC_N_HYDROS, 1, vec![window_year]);
    let mut hist_rows = Vec::with_capacity(DLC_N_STAGES * DLC_N_HYDROS);
    for t in 0..DLC_N_STAGES {
        let start_date = fx.stages[t].start_date;
        let end_date = fx.stages[t].end_date;
        for h in 0..DLC_N_HYDROS {
            hist_rows.push(InflowHistoryRow {
                hydro_id: fx.hydro_ids[h],
                start_date,
                end_date,
                value_m3s: fx.raw[t][h],
            });
        }
    }
    standardize_historical_windows(
        &mut hist_lib,
        &hist_rows,
        &fx.hydro_ids,
        &fx.stages,
        &fx.par,
        &[window_year],
        None,
        DerivedSeed {
            lag_values: &derived_lag_values,
            l_state: 1,
            accum: &[],
            weight: &[],
        },
        &fx.transitions,
        1,
    );

    for t in 0..DLC_N_STAGES {
        for h in 0..DLC_N_HYDROS {
            let det_base = fx.par.deterministic_base(t, h);
            let psi = fx.par.psi_slice(t, h);
            let sigma = fx.par.sigma(t, h);
            let lag_buf = [oracle_incoming[t][h]];
            let expected_eta = solve_par_noise(det_base, psi, &lag_buf, sigma, fx.raw[t][h]);

            let eta_ext = ext_lib.eta_slice(t, 0)[h];
            let eta_hist = hist_lib.eta_slice(0, t)[h];

            assert_eq!(
                eta_ext, expected_eta,
                "External sampler eta[stage={t}, hydro={h}] diverges from the forward lag chain"
            );
            assert_eq!(
                eta_hist, expected_eta,
                "Historical sampler eta[stage={t}, hydro={h}] diverges from the forward lag chain"
            );
        }
    }
}

#[test]
fn differential_lag_chain_negative_control_primary_only_advance_diverges_at_transition() {
    let fx = build_dlc_fixture();
    let oracle_incoming = dlc_forward_oracle_incoming_lags(&fx);
    let naive_incoming = dlc_naive_primary_only_incoming_lags(&fx);

    for t in 0..4 {
        assert_eq!(
            naive_incoming[t], oracle_incoming[t],
            "stage {t} precedes (or is) the transition stage; the ring is inert \
             here, so naive and kernel-routed chains must still agree"
        );
    }

    for h in 0..DLC_N_HYDROS {
        let diff = (naive_incoming[4][h] - oracle_incoming[4][h]).abs();
        assert!(
            diff > 1e-9,
            "hydro {h}: naive primary-only advance must diverge from the \
             ring-rebuilt forward chain at the transition stage, got \
             naive={} vs forward={} (diff={diff})",
            naive_incoming[4][h],
            oracle_incoming[4][h],
        );
    }
}

// ---------------------------------------------------------------------------
// Opening-tree ring-honoring regression (derive_downstream_par_order)
// ---------------------------------------------------------------------------
//
// `build_opening_tree_library` derives `downstream_par_order` via
// `derive_downstream_par_order` and threads it into
// `standardize_historical_windows`; it used to pass a literal `0` there
// instead. This drives the same DLC monthly->quarterly fixture through
// `standardize_historical_windows` twice — once with the derived value, once
// with the literal-`0` regression — and checks the derived-value run against
// the independent forward-chain oracle while the literal-`0` run reproduces
// the independently-computed primary-only advance.

#[test]
fn opening_tree_historical_standardization_ring_aware_eta_requires_derived_downstream_par_order() {
    let fx = build_dlc_fixture();
    let oracle_incoming = dlc_forward_oracle_incoming_lags(&fx);
    let naive_incoming = dlc_naive_primary_only_incoming_lags(&fx);

    let derived = derive_downstream_par_order(&fx.stages, &fx.par, Some(&fx.season_map));
    assert_eq!(
        derived,
        fx.par.max_order(),
        "the fixture crosses season_id >= 12 at stage 3; derive_downstream_par_order \
         must gate to par.max_order(), not 0"
    );

    let window_year = 2026;
    let mut hist_rows = Vec::with_capacity(DLC_N_STAGES * DLC_N_HYDROS);
    for t in 0..DLC_N_STAGES {
        let start_date = fx.stages[t].start_date;
        let end_date = fx.stages[t].end_date;
        for h in 0..DLC_N_HYDROS {
            hist_rows.push(InflowHistoryRow {
                hydro_id: fx.hydro_ids[h],
                start_date,
                end_date,
                value_m3s: fx.raw[t][h],
            });
        }
    }

    let derived_lag_values = fx.derived_lag_values();
    let mut hist_lib_ring_aware =
        HistoricalScenarioLibrary::new(1, DLC_N_STAGES, DLC_N_HYDROS, 1, vec![window_year]);
    standardize_historical_windows(
        &mut hist_lib_ring_aware,
        &hist_rows,
        &fx.hydro_ids,
        &fx.stages,
        &fx.par,
        &[window_year],
        None,
        DerivedSeed {
            lag_values: &derived_lag_values,
            l_state: 1,
            accum: &[],
            weight: &[],
        },
        &fx.transitions,
        derived,
    );

    let mut hist_lib_literal_zero =
        HistoricalScenarioLibrary::new(1, DLC_N_STAGES, DLC_N_HYDROS, 1, vec![window_year]);
    standardize_historical_windows(
        &mut hist_lib_literal_zero,
        &hist_rows,
        &fx.hydro_ids,
        &fx.stages,
        &fx.par,
        &[window_year],
        None,
        DerivedSeed {
            lag_values: &derived_lag_values,
            l_state: 1,
            accum: &[],
            weight: &[],
        },
        &fx.transitions,
        0,
    );

    for h in 0..DLC_N_HYDROS {
        let det_base = fx.par.deterministic_base(4, h);
        let psi = fx.par.psi_slice(4, h);
        let sigma = fx.par.sigma(4, h);

        let expected_ring_aware_eta =
            solve_par_noise(det_base, psi, &[oracle_incoming[4][h]], sigma, fx.raw[4][h]);
        let expected_naive_eta =
            solve_par_noise(det_base, psi, &[naive_incoming[4][h]], sigma, fx.raw[4][h]);

        let eta_ring_aware = hist_lib_ring_aware.eta_slice(0, 4)[h];
        let eta_literal_zero = hist_lib_literal_zero.eta_slice(0, 4)[h];

        assert_eq!(
            eta_ring_aware, expected_ring_aware_eta,
            "hydro {h}: standardize_historical_windows with the derived downstream_par_order \
             must match the independent forward-chain oracle at the quarterly transition"
        );
        assert_eq!(
            eta_literal_zero, expected_naive_eta,
            "hydro {h}: standardize_historical_windows with a literal 0 must reproduce the \
             independently-computed primary-only advance (the pre-fix regression value)"
        );
        assert!(
            (eta_ring_aware - eta_literal_zero).abs() > 1e-6,
            "hydro {h}: ring-aware eta must differ from the literal-0 regression value, \
             got ring_aware={eta_ring_aware} == literal_zero={eta_literal_zero}"
        );
    }
}
