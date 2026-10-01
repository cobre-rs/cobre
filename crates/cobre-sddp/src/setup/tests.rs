use super::{
    BoundaryStateRequirements, NodeId, NodePos, PhaseLibraries, ScenarioLibraries, StudySetup,
    assert_external_library_widths, build_contract_prices_per_stage, study_horizon_end,
    validate_par_shape,
};
use crate::SddpError;
use crate::block_clock::M3S_TO_HM3;
use crate::hydro_models::{PrepareHydroModelsResult, ProductionModelSet, ResolvedProductionModel};
use crate::lp::builder::StageGeometry;
use crate::lp::indexer::{AnticipatedPlants, StateSpace, ThermalSys};
use crate::test_support;
use crate::time_value::DeliveryCalendar;
use cobre_stochastic::ExternalScenarioLibrary;
use cobre_stochastic::par::precompute::PrecomputedPar;
use cobre_stochastic::season_cast::StageCalendar;

use chrono::{Duration, NaiveDate};
use cobre_core::{
    BlockBoundsCountsSpec, BoundsCountsSpec, BoundsDefaults, BusStagePenalties,
    ContractBlockBounds, ContractBlockOverride, HydroBlockBounds, HydroPenalties, HydroStageBounds,
    LineBlockBounds, LineStagePenalties, NcsStagePenalties, PenaltiesCountsSpec, PenaltiesDefaults,
    PumpingBlockBounds, ResolvedBlockBounds, ResolvedBounds, ResolvedPenalties, ThermalBlockBounds,
    ThermalStageBounds,
};
use cobre_core::{
    ContractType, EnergyContract, EntityId, HorizonGraph, HydroPastDefluence, InitialConditions,
    PostStudyStage, PostStudyStages, System, SystemBuilder,
    entities::{
        bus::{Bus, DeficitSegment},
        hydro::{Hydro, HydroGenerationModel},
        thermal::{AnticipatedConfig, Thermal},
    },
    scenario::{InflowHistoryRow, InflowModel, LoadModel, SamplingScheme},
    temporal::{
        Block, BlockMode, NoiseMethod, PolicyGraphType, ScenarioSourceConfig, SeasonMap, Stage,
        StageRiskConfig, StageStateConfig,
    },
};
use cobre_io::config::{
    Config, EstimationConfig, ExportsConfig, InflowNonNegativityConfig,
    InflowNonNegativityMethod as CfgInflowMethod, ModelingConfig, PolicyConfig,
    RawClassConfigEntry, RawSamplingScheme, RawScenarioSourceConfig, RowSelectionConfig,
    SimulationConfig as IoSimulationConfig, SimulationSelection, StoppingMode, StoppingRuleConfig,
    TrainingConfig, TrainingSelection, TrainingSolverConfig, UpperBoundEvaluationConfig,
};
use cobre_stochastic::{ClassSchemes, OpeningTreeInputs, build_stochastic_context};

/// Bounds and penalties are non-zero so the training tests below (e.g.
/// `train_generates_cuts_in_fcf`) solve a genuinely optimized LP with real
/// duals, not a degenerate all-zero one.
fn minimal_system(n_stages: usize) -> cobre_core::System {
    minimal_system_with_policy_graph(n_stages, HorizonGraph::default())
}

/// [`minimal_system`]'s body, generalized to accept a caller-supplied
/// `policy_graph` (a node-native or discount-override fixture, e.g.) instead
/// of always defaulting to a plain chain.
#[expect(
    clippy::too_many_lines,
    clippy::items_after_statements,
    reason = "the fixture spells out one complete study inline so each assertion traces to a literal, and fixture-local helper items sit beside the entities that use them"
)]
fn minimal_system_with_policy_graph(
    n_stages: usize,
    policy_graph: HorizonGraph,
) -> cobre_core::System {
    use chrono::NaiveDate;

    let bus = Bus {
        id: EntityId(1),
        name: "B1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 500.0,
        }],
        excess_cost: 0.0,
    };

    let thermal = Thermal {
        id: EntityId(2),
        name: "T1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        bus_id: EntityId(1),
        min_generation_mw: 0.0,
        max_generation_mw: 100.0,
        cost_per_mwh: 50.0,
        anticipated_config: None,
        entry_stage_id: None,
        exit_stage_id: None,
    };

    let mut hydro = Hydro {
        unit_groups: Vec::new(),
        id: EntityId(3),
        name: "H1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        downstream_id: None,
        travel_time_hours: None,
        entry_stage_id: None,
        exit_stage_id: None,
        min_storage_hm3: 0.0,
        max_storage_hm3: 200.0,
        min_outflow_m3s: 0.0,
        max_outflow_m3s: None,
        generation_model: HydroGenerationModel::ConstantProductivity,
        min_turbined_m3s: 0.0,
        max_turbined_m3s: 100.0,
        specific_productivity_mw_per_m3s_per_m: None,
        min_generation_mw: 0.0,
        max_generation_mw: 250.0,
        tailrace: None,
        hydraulic_losses: None,
        efficiency: None,
        evaporation_coefficients_mm: None,
        evaporation_reference_volumes_hm3: None,
        diversion: None,
        filling: None,
        penalties: HydroPenalties {
            spillage_cost: 0.01,
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
    hydro.declare_mirror_unit_group(EntityId(1));

    let stages: Vec<Stage> = (0..n_stages)
        .map(|i| Stage {
            index: i,
            id: i as i32,
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            season_id: None,
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
                branching_factor: 1,
                noise_method: NoiseMethod::Saa,
            },
        })
        .collect();

    let inflow_models: Vec<InflowModel> = (0..n_stages)
        .map(|i| InflowModel {
            hydro_id: EntityId(3),
            stage_id: i as i32,
            mean_m3s: 80.0,
            std_m3s: 20.0,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
            annual: None,
        })
        .collect();

    let load_models: Vec<LoadModel> = (0..n_stages)
        .map(|i| LoadModel {
            bus_id: EntityId(1),
            stage_id: i as i32,
            mean_mw: 100.0,
            std_mw: 0.0,
        })
        .collect();

    let n_st = n_stages.max(1);

    fn default_hydro_bounds() -> HydroStageBounds {
        HydroStageBounds {
            min_storage_hm3: 0.0,
            max_storage_hm3: 200.0,
            filling_min_rate_m3s: 0.0,
            water_withdrawal_m3s: 0.0,
        }
    }

    fn default_hydro_block_bounds() -> HydroBlockBounds {
        HydroBlockBounds {
            max_turbined_m3s: 100.0,
            max_generation_mw: 250.0,
            ..Default::default()
        }
    }

    fn default_hydro_penalties() -> HydroPenalties {
        HydroPenalties {
            spillage_cost: 0.01,
            diversion_cost: 0.0,
            turbined_cost: 0.0,
            storage_violation_below_cost: 500.0,
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

    let bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 1,
            n_thermals: 1,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages: n_st,
            k_max: 0,
        },
        &BoundsDefaults {
            hydro: default_hydro_bounds(),
            hydro_block: default_hydro_block_bounds(),
            thermal: ThermalStageBounds { cost_per_mwh: 0.0 },
            thermal_block: ThermalBlockBounds {
                min_generation_mw: 0.0,
                max_generation_mw: 100.0,
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

    let penalties = ResolvedPenalties::new(
        &PenaltiesCountsSpec {
            n_hydros: 1,
            n_buses: 1,
            n_lines: 0,
            n_ncs: 0,
            n_stages: n_st,
        },
        &PenaltiesDefaults {
            hydro: default_hydro_penalties(),
            bus: BusStagePenalties { excess_cost: 0.0 },
            line: LineStagePenalties { exchange_cost: 0.0 },
            ncs: NcsStagePenalties {
                curtailment_cost: 0.0,
            },
        },
    );

    SystemBuilder::new()
        .buses(vec![bus])
        .thermals(vec![thermal])
        .hydros(vec![hydro])
        .stages(stages)
        .inflow_models(inflow_models)
        .load_models(load_models)
        .bounds(bounds)
        .penalties(penalties)
        .policy_graph(policy_graph)
        .build()
        .expect("minimal_system: valid")
}

/// FPHA hydro with no VHA rows or `specific_productivity_mw_per_m3s_per_m`, so
/// the energy-conversion gate must reject it.
#[expect(
    clippy::too_many_lines,
    reason = "the fixture spells out one complete study inline so each assertion traces to a literal"
)]
fn minimal_fpha_misconfigured_system(n_stages: usize) -> cobre_core::System {
    use chrono::NaiveDate;

    let bus = Bus {
        id: EntityId(1),
        name: "B1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 500.0,
        }],
        excess_cost: 0.0,
    };

    let thermal = Thermal {
        id: EntityId(2),
        name: "T1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        bus_id: EntityId(1),
        min_generation_mw: 0.0,
        max_generation_mw: 100.0,
        cost_per_mwh: 50.0,
        anticipated_config: None,
        entry_stage_id: None,
        exit_stage_id: None,
    };

    let mut hydro = Hydro {
        unit_groups: Vec::new(),
        id: EntityId(3),
        name: "H_FPHA_BAD".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        downstream_id: None,
        travel_time_hours: None,
        entry_stage_id: None,
        exit_stage_id: None,
        min_storage_hm3: 0.0,
        max_storage_hm3: 200.0,
        min_outflow_m3s: 0.0,
        max_outflow_m3s: None,
        generation_model: HydroGenerationModel::Fpha,
        min_turbined_m3s: 0.0,
        max_turbined_m3s: 100.0,
        specific_productivity_mw_per_m3s_per_m: None,
        min_generation_mw: 0.0,
        max_generation_mw: 250.0,
        tailrace: None,
        hydraulic_losses: None,
        efficiency: None,
        evaporation_coefficients_mm: None,
        evaporation_reference_volumes_hm3: None,
        diversion: None,
        filling: None,
        penalties: HydroPenalties {
            spillage_cost: 0.01,
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
    hydro.declare_mirror_unit_group(EntityId(1));

    let stages: Vec<Stage> = (0..n_stages)
        .map(|i| Stage {
            index: i,
            id: i as i32,
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            season_id: None,
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
                branching_factor: 1,
                noise_method: NoiseMethod::Saa,
            },
        })
        .collect();

    let inflow_models: Vec<InflowModel> = (0..n_stages)
        .map(|i| InflowModel {
            hydro_id: EntityId(3),
            stage_id: i as i32,
            mean_m3s: 80.0,
            std_m3s: 20.0,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
            annual: None,
        })
        .collect();

    let load_models: Vec<LoadModel> = (0..n_stages)
        .map(|i| LoadModel {
            bus_id: EntityId(1),
            stage_id: i as i32,
            mean_mw: 100.0,
            std_mw: 0.0,
        })
        .collect();

    let n_st = n_stages.max(1);

    let bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 1,
            n_thermals: 1,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages: n_st,
            k_max: 0,
        },
        &BoundsDefaults {
            hydro: HydroStageBounds {
                min_storage_hm3: 0.0,
                max_storage_hm3: 200.0,
                filling_min_rate_m3s: 0.0,
                water_withdrawal_m3s: 0.0,
            },
            hydro_block: HydroBlockBounds {
                max_turbined_m3s: 100.0,
                max_generation_mw: 250.0,
                ..Default::default()
            },
            thermal: ThermalStageBounds { cost_per_mwh: 0.0 },
            thermal_block: ThermalBlockBounds {
                min_generation_mw: 0.0,
                max_generation_mw: 100.0,
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

    let penalties = ResolvedPenalties::new(
        &PenaltiesCountsSpec {
            n_hydros: 1,
            n_buses: 1,
            n_lines: 0,
            n_ncs: 0,
            n_stages: n_st,
        },
        &PenaltiesDefaults {
            hydro: HydroPenalties {
                spillage_cost: 0.01,
                diversion_cost: 0.0,
                turbined_cost: 0.0,
                storage_violation_below_cost: 500.0,
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
            bus: BusStagePenalties { excess_cost: 0.0 },
            line: LineStagePenalties { exchange_cost: 0.0 },
            ncs: NcsStagePenalties {
                curtailment_cost: 0.0,
            },
        },
    );

    SystemBuilder::new()
        .buses(vec![bus])
        .thermals(vec![thermal])
        .hydros(vec![hydro])
        .stages(stages)
        .inflow_models(inflow_models)
        .load_models(load_models)
        .bounds(bounds)
        .penalties(penalties)
        .build()
        .expect("minimal_fpha_misconfigured_system: valid")
}

fn minimal_config(forward_passes: u32, max_iterations: u32) -> Config {
    Config {
        schema: None,
        modeling: ModelingConfig {
            inflow_non_negativity: InflowNonNegativityConfig {
                method: CfgInflowMethod::Penalty,
            },
            cost_scale_factor: None,
        },
        training: TrainingConfig {
            enabled: true,
            tree_seed: Some(42),
            stopping_rules: Some(vec![StoppingRuleConfig::IterationLimit {
                limit: max_iterations,
            }]),
            stopping_mode: StoppingMode::Any,
            cut_selection: RowSelectionConfig::default(),
            solver: TrainingSolverConfig::default(),
            parallelism: cobre_io::config::ParallelismConfig::default(),
            scenario_source: None,
            selection: Some(TrainingSelection::Sampled { forward_passes }),
        },
        upper_bound_evaluation: UpperBoundEvaluationConfig::default(),
        policy: PolicyConfig::default(),
        simulation: IoSimulationConfig::default(),
        exports: ExportsConfig::default(),
        estimation: EstimationConfig::default(),
    }
}

/// `inflow_scheme`/`load_scheme`/`ncs_scheme` select each class's scheme; `None`
/// leaves the class at `in_sample`.
fn minimal_config_with_schemes(
    forward_passes: u32,
    max_iterations: u32,
    inflow_scheme: Option<RawSamplingScheme>,
    load_scheme: Option<RawSamplingScheme>,
    ncs_scheme: Option<RawSamplingScheme>,
) -> Config {
    // A seed is required when any class uses a non-in-sample scheme.
    let needs_seed = inflow_scheme.is_some_and(|s| s != RawSamplingScheme::InSample)
        || load_scheme.is_some_and(|s| s != RawSamplingScheme::InSample)
        || ncs_scheme.is_some_and(|s| s != RawSamplingScheme::InSample);
    let scenario_source = RawScenarioSourceConfig {
        seed: if needs_seed { Some(42) } else { None },
        historical_years: None,
        inflow: inflow_scheme.map(|scheme| RawClassConfigEntry { scheme }),
        load: load_scheme.map(|scheme| RawClassConfigEntry { scheme }),
        ncs: ncs_scheme.map(|scheme| RawClassConfigEntry { scheme }),
        openings: None,
    };
    let mut config = minimal_config(forward_passes, max_iterations);
    config.training.scenario_source = Some(scenario_source);
    config
}

#[test]
fn new_minimal_valid_system_returns_ok() {
    let system = minimal_system(2);
    let config = minimal_config(1, 10);
    let stochastic = build_stochastic_context(
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
    .expect("stochastic context");

    let result = StudySetup::new(
        &system,
        &config,
        stochastic,
        PrepareHydroModelsResult::default_from_system(&system),
        Vec::new(),
    );
    assert!(result.is_ok(), "expected Ok, got {result:?}");
    let setup = result.unwrap();
    assert!(!setup.inputs.stage_data.stage_templates.templates.is_empty());
}

#[test]
fn new_zero_stages_returns_validation_error() {
    let system = minimal_system(0);
    let config = minimal_config(1, 10);
    let stochastic = build_stochastic_context(
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
    .expect("stochastic context");

    let result = StudySetup::new(
        &system,
        &config,
        stochastic,
        PrepareHydroModelsResult::default_from_system(&system),
        Vec::new(),
    );
    assert!(result.is_err(), "expected Err, got Ok");
    let err = result.unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("no study stages"),
        "error message should contain 'no study stages': {msg}"
    );
}

#[test]
fn accessor_methods_return_expected_values() {
    let n_stages = 3;
    let system = minimal_system(n_stages);
    let config = minimal_config(2, 50);
    let stochastic = build_stochastic_context(
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
    .expect("stochastic context");

    let setup = StudySetup::new(
        &system,
        &config,
        stochastic,
        PrepareHydroModelsResult::default_from_system(&system),
        Vec::new(),
    )
    .expect("setup");

    assert_eq!(
        setup.inputs.stage_data.stage_templates.templates.len(),
        n_stages
    );

    assert_eq!(setup.loop_params.seed, 42);
    assert_eq!(setup.loop_params.forward_passes, 2);
    assert_eq!(setup.loop_params.max_iterations, 50);
    assert_eq!(setup.simulation_config.n_scenarios, 0); // simulation disabled by default
    assert_eq!(setup.policy_path, "./policy");

    assert_eq!(
        setup
            .inputs
            .stage_data
            .stage_templates
            .geometry_per_stage
            .len(),
        n_stages
    );
    assert!(
        StageGeometry::max_blocks(&setup.inputs.stage_data.stage_templates.geometry_per_stage) > 0
    );

    assert_eq!(setup.inputs.horizon.num_stages(), n_stages);

    assert_eq!(setup.inputs.cut_management.risk_measures.len(), n_stages);

    assert_eq!(setup.fcf.pools.len(), n_stages);

    assert_eq!(setup.inputs.stage_data.entity_counts.hydro_ids.len(), 1);
    assert_eq!(setup.inputs.stage_data.entity_counts.thermal_ids.len(), 1);
}

#[test]
fn fcf_mut_allows_cut_insertion() {
    let system = minimal_system(2);
    let config = minimal_config(1, 10);
    let stochastic = build_stochastic_context(
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
    .expect("stochastic context");

    let mut setup = StudySetup::new(
        &system,
        &config,
        stochastic,
        PrepareHydroModelsResult::default_from_system(&system),
        Vec::new(),
    )
    .expect("setup");

    let n_state = setup.inputs.stage_data.state.n_state;
    let coefficients = vec![1.0_f64; n_state];
    setup.fcf.add_cut(NodeId(0), 0, 0, 0, 42.0, &coefficients);
    assert_eq!(setup.fcf.total_active_cuts(), 1);
}

#[test]
fn inflow_method_reflects_config() {
    use crate::InflowNonNegativityMethod;

    let system = minimal_system(2);
    let config = minimal_config(1, 10);
    let stochastic = build_stochastic_context(
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
    .expect("stochastic context");

    let setup = StudySetup::new(
        &system,
        &config,
        stochastic,
        PrepareHydroModelsResult::default_from_system(&system),
        Vec::new(),
    )
    .expect("setup");

    assert!(
        !matches!(
            setup.inputs.stage_data.study_dims.inflow_method,
            InflowNonNegativityMethod::None
        ),
        "expected penalty or truncation method"
    );
}

#[test]
fn cut_selection_none_when_disabled() {
    let system = minimal_system(2);
    let config = minimal_config(1, 10);
    let stochastic = build_stochastic_context(
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
    .expect("stochastic context");

    let setup = StudySetup::new(
        &system,
        &config,
        stochastic,
        PrepareHydroModelsResult::default_from_system(&system),
        Vec::new(),
    )
    .expect("setup");

    assert!(
        setup.inputs.cut_management.cut_selection.is_none(),
        "cut_selection should be None when disabled"
    );
}

#[test]
fn stage_ctx_fields_match_study_setup() {
    let n_stages = 3;
    let system = minimal_system(n_stages);
    let config = minimal_config(2, 10);
    let stochastic = build_stochastic_context(
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
    .expect("stochastic context");

    let setup = StudySetup::new(
        &system,
        &config,
        stochastic,
        PrepareHydroModelsResult::default_from_system(&system),
        Vec::new(),
    )
    .expect("setup");
    let ctx = setup.stage_ctx();

    assert_eq!(
        ctx.templates.len(),
        setup.inputs.stage_data.stage_templates.templates.len(),
        "templates length mismatch"
    );
    assert_eq!(
        setup.inputs.stage_data.state.hydro_count,
        setup.inputs.stage_data.entity_counts.hydro_ids.len(),
        "n_hydros mismatch"
    );
    assert_eq!(
        ctx.geometry_per_stage.len(),
        n_stages,
        "geometry_per_stage length mismatch"
    );
}

#[test]
fn training_ctx_fields_match_study_setup() {
    let n_stages = 3;
    let system = minimal_system(n_stages);
    let config = minimal_config(2, 10);
    let stochastic = build_stochastic_context(
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
    .expect("stochastic context");

    let setup = StudySetup::new(
        &system,
        &config,
        stochastic,
        PrepareHydroModelsResult::default_from_system(&system),
        Vec::new(),
    )
    .expect("setup");
    let ctx = setup.training_ctx();

    assert_eq!(
        ctx.horizon.num_stages(),
        setup.inputs.horizon.num_stages(),
        "horizon num_stages mismatch"
    );
    assert_eq!(
        ctx.state.n_state, setup.inputs.stage_data.state.n_state,
        "indexer n_state mismatch"
    );
    assert_eq!(
        ctx.initial_state.len(),
        setup.inputs.initial.state.len(),
        "initial_state length mismatch"
    );
}

#[test]
fn simulation_ctx_propagates_dynamic_dcs_from_setup() {
    use crate::dcs::DcsParams;
    use cobre_io::config::SelectionMethod;

    let n_stages = 3;
    let system = minimal_system(n_stages);
    let mut config = minimal_config(2, 10);
    // A Dynamic strategy here is what makes `simulation_ctx()` populate `dcs`.
    config.training.cut_selection = RowSelectionConfig {
        selection: Some(SelectionMethod::Dynamic {
            start_iteration: 2,
            seed_window: 5,
            candidate_recency: None,
            max_added_per_round: 10,
            violation_tolerance: 1e-10,
        }),
        ..RowSelectionConfig::default()
    };
    let stochastic = build_stochastic_context(
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
    .expect("stochastic context");

    let setup = StudySetup::new(
        &system,
        &config,
        stochastic,
        PrepareHydroModelsResult::default_from_system(&system),
        Vec::new(),
    )
    .expect("setup");
    let ctx = setup.simulation_ctx();

    // The dynamic method with default fields maps to the spec defaults
    let expected = DcsParams {
        k1: None,
        k2: 5,
        nadic: 10,
        epsilon_viol: 1e-10,
        start_iteration: 2,
        max_inner_iterations: DcsParams::default().max_inner_iterations,
    };
    assert_eq!(
        ctx.dcs,
        Some(expected),
        "simulation_ctx().dcs must carry the configured dynamic DcsParams, got {:?}",
        ctx.dcs
    );
}

#[test]
fn train_completes_within_iteration_limit() {
    use cobre_comm::LocalBackend;
    use cobre_solver::ActiveSolver;

    let system = minimal_system(2);
    let config = minimal_config(1, 3);
    let stochastic = build_stochastic_context(
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
    .expect("stochastic context");

    let mut setup = StudySetup::new(
        &system,
        &config,
        stochastic,
        PrepareHydroModelsResult::default_from_system(&system),
        Vec::new(),
    )
    .expect("setup");
    let comm = LocalBackend;
    let mut solver = ActiveSolver::new().expect("solver");

    let result = setup
        .train(&mut solver, &comm, 1, ActiveSolver::new, None, None)
        .expect("train");

    assert!(
        result.result.iterations <= 3,
        "expected iterations <= 3, got {}",
        result.result.iterations
    );
    assert!(
        result.result.iterations >= 1,
        "expected at least 1 iteration, got {}",
        result.result.iterations
    );
}

#[test]
fn train_generates_cuts_in_fcf() {
    use cobre_comm::LocalBackend;
    use cobre_solver::ActiveSolver;

    let system = minimal_system(2);
    let config = minimal_config(1, 3);
    let stochastic = build_stochastic_context(
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
    .expect("stochastic context");

    let mut setup = StudySetup::new(
        &system,
        &config,
        stochastic,
        PrepareHydroModelsResult::default_from_system(&system),
        Vec::new(),
    )
    .expect("setup");
    let comm = LocalBackend;
    let mut solver = ActiveSolver::new().expect("solver");

    setup
        .train(&mut solver, &comm, 1, ActiveSolver::new, None, None)
        .expect("train");

    assert!(
        setup.fcf.pools[0].populated() > 0,
        "expected at least one cut in FCF pool[0] after training"
    );
}

/// A 3-stage binary tree (root fanning ×2 at stage 0, ×2 again at stage 1 —
/// the design's own headline node-native shape, DESIGN-node-native-engine.md's
/// fixture (b)) loads end-to-end through `StudySetup::new` and constructs
/// the runtime node graph correctly: node identity/count, the `node → pool`
/// map with leaf sharing, and canonical (ascending child id) successor lists.
/// All 7 nodes are generated (`scenario_id: None`).
///
/// TODO(per-node-backward-traversal): assert end-to-end branching training
/// here once per-node opening-to-child-node substrate routing and per-node
/// pools land. The backward pass still routes openings per STAGE
/// (`tree_view.n_openings(successor_stage)`), so training this graph panics
/// today: a node with more than one child duplicates or exceeds that stage's
/// raw opening count, violating `by_scenario.rs`'s "`solve_order(s)` must be a
/// permutation of `0..n_openings`" invariant. Chain topologies are unaffected
/// and stay bit-for-bit pinned by the golden parity suite.
#[test]
fn node_native_binary_tree_loads_and_constructs_node_graph() {
    use cobre_core::temporal::{Node, Transition};
    use std::collections::BTreeMap;

    let policy_graph = HorizonGraph {
        graph_type: PolicyGraphType::FiniteHorizon,
        annual_discount_rate: 0.0,
        nodes: vec![
            Node {
                id: 0,
                stage_id: 0,
                scenario_id: None,
                label: None,
            },
            Node {
                id: 1,
                stage_id: 1,
                scenario_id: None,
                label: None,
            },
            Node {
                id: 2,
                stage_id: 1,
                scenario_id: None,
                label: None,
            },
            Node {
                id: 3,
                stage_id: 2,
                scenario_id: None,
                label: None,
            },
            Node {
                id: 4,
                stage_id: 2,
                scenario_id: None,
                label: None,
            },
            Node {
                id: 5,
                stage_id: 2,
                scenario_id: None,
                label: None,
            },
            Node {
                id: 6,
                stage_id: 2,
                scenario_id: None,
                label: None,
            },
        ],
        transitions: vec![
            Transition {
                source_id: 0,
                target_id: 1,
                probability: 0.5,
                annual_discount_rate_override: None,
            },
            Transition {
                source_id: 0,
                target_id: 2,
                probability: 0.5,
                annual_discount_rate_override: None,
            },
            Transition {
                source_id: 1,
                target_id: 3,
                probability: 0.5,
                annual_discount_rate_override: None,
            },
            Transition {
                source_id: 1,
                target_id: 4,
                probability: 0.5,
                annual_discount_rate_override: None,
            },
            Transition {
                source_id: 2,
                target_id: 5,
                probability: 0.5,
                annual_discount_rate_override: None,
            },
            Transition {
                source_id: 2,
                target_id: 6,
                probability: 0.5,
                annual_discount_rate_override: None,
            },
        ],
        stage_discount_rate_overrides: BTreeMap::new(),
        season_map: None,
    };

    let system = minimal_system_with_policy_graph(3, policy_graph);
    let config = minimal_config(1, 1);
    let stochastic = build_stochastic_context(
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
    .expect("stochastic context");

    let setup = StudySetup::new(
        &system,
        &config,
        stochastic,
        PrepareHydroModelsResult::default_from_system(&system),
        Vec::new(),
    )
    .expect("setup: node-native binary tree must load end-to-end");

    // The runtime node graph mirrors the declared 7-node binary tree: nodes
    // 0/1/2 have successors and own their own pool; nodes 3/4/5/6 are leaves
    // and share exactly one pool.
    assert_eq!(setup.inputs.node_graph.nodes.len(), 7);
    assert_eq!(
        setup.inputs.node_graph.n_pools, 4,
        "3 internal nodes each own a pool, 4 leaves share one"
    );
    // The FCF and its paired cut-state layouts are sized to the pool axis
    // (`n_pools`), NOT the node count or the stage count (3).
    assert_eq!(
        setup.fcf.pools.len(),
        setup.inputs.node_graph.n_pools,
        "FutureCostFunction.pools is sized to n_pools, not node count or stage count"
    );
    assert_eq!(
        setup.inputs.cut_state_layouts.len(),
        setup.inputs.node_graph.n_pools,
        "cut_state_layouts is sized to n_pools, paired 1:1 with fcf.pools"
    );

    // Canonical (ascending child node id) successor structure, matching the
    // declared transitions: 0->{1,2}, 1->{3,4}, 2->{5,6}; leaves have none.
    let child_ids = |pos: usize| -> Vec<NodeId> {
        setup.inputs.node_graph.successors[NodePos(pos)]
            .iter()
            .map(|s| setup.inputs.node_graph.node_ids[s.child])
            .collect()
    };
    assert_eq!(child_ids(0), vec![NodeId(1), NodeId(2)]);
    assert_eq!(child_ids(1), vec![NodeId(3), NodeId(4)]);
    assert_eq!(child_ids(2), vec![NodeId(5), NodeId(6)]);
    for leaf_pos in 3..7 {
        assert!(setup.inputs.node_graph.successors[NodePos(leaf_pos)].is_empty());
    }
}

/// Counterpart to [`node_native_binary_tree_loads_and_constructs_node_graph`]:
/// on the chain degeneracy (`nodes[]` absent), `StudySetup`'s FCF has exactly
/// `num_stages` pools and every node's `pool_id` is the identity `t`, so the
/// pool re-key reduces byte-for-byte to the pre-node-native per-stage FCF.
#[test]
fn chain_fcf_pools_len_equals_num_stages_with_pool_id_identity() {
    let n_stages = 4;
    let system = minimal_system(n_stages);
    let config = minimal_config(1, 1);
    let stochastic = build_stochastic_context(
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
    .expect("stochastic context");

    let setup = StudySetup::new(
        &system,
        &config,
        stochastic,
        PrepareHydroModelsResult::default_from_system(&system),
        Vec::new(),
    )
    .expect("setup: chain must load end-to-end");

    assert_eq!(setup.inputs.node_graph.n_pools, n_stages);
    assert_eq!(setup.fcf.pools.len(), n_stages);
    assert_eq!(setup.inputs.cut_state_layouts.len(), n_stages);
    for t in 0..n_stages {
        assert_eq!(
            setup.inputs.node_graph.nodes[NodePos(t)].pool_id,
            t,
            "chain degeneracy: pool_id must equal stage index {t}"
        );
    }
}

#[test]
fn simulation_config_reflects_setup_fields() {
    let mut config = minimal_config(1, 5);
    config.simulation = IoSimulationConfig {
        enabled: true,
        selection: Some(SimulationSelection::Sampled { num_scenarios: 50 }),
        io_channel_capacity: 16,
        ..IoSimulationConfig::default()
    };

    let system = minimal_system(2);
    let stochastic = build_stochastic_context(
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
    .expect("stochastic context");

    let setup = StudySetup::new(
        &system,
        &config,
        stochastic,
        PrepareHydroModelsResult::default_from_system(&system),
        Vec::new(),
    )
    .expect("setup");

    let sim_cfg = setup.simulation_config();
    assert_eq!(sim_cfg.n_scenarios, setup.simulation_config.n_scenarios);
    assert_eq!(
        sim_cfg.io_channel_capacity,
        setup.simulation_config.io_channel_capacity
    );
}

#[test]
fn create_workspace_pool_returns_correct_size() {
    use cobre_comm::LocalBackend;
    use cobre_solver::ActiveSolver;

    let system = minimal_system(2);
    let config = minimal_config(1, 3);
    let stochastic = build_stochastic_context(
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
    .expect("stochastic context");

    let setup = StudySetup::new(
        &system,
        &config,
        stochastic,
        PrepareHydroModelsResult::default_from_system(&system),
        Vec::new(),
    )
    .expect("setup");

    let comm = LocalBackend;
    let pool = setup
        .create_workspace_pool(&comm, 2, ActiveSolver::new)
        .expect("workspace pool");

    assert_eq!(pool.workspaces.len(), 2);
}

#[test]
fn simulation_pool_scratch_is_sized_from_the_study_owners() {
    use cobre_comm::LocalBackend;
    use cobre_solver::ActiveSolver;

    let system = minimal_system(2);
    let config = minimal_config(1, 3);
    let stochastic = build_stochastic_context(
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
    .expect("stochastic context");

    let setup = StudySetup::new(
        &system,
        &config,
        stochastic,
        PrepareHydroModelsResult::default_from_system(&system),
        Vec::new(),
    )
    .expect("setup");

    let comm = LocalBackend;
    let pool = setup
        .create_workspace_pool(&comm, 1, ActiveSolver::new)
        .expect("workspace pool");

    let state = &setup.inputs.stage_data.state;
    let max_n_blks = setup
        .inputs
        .stage_data
        .stage_templates
        .geometry_per_stage
        .iter()
        .map(|g| g.n_blks)
        .max()
        .unwrap_or(0);
    let n_load_buses = setup.inputs.stage_data.stage_templates.n_load_buses();

    let scratch = &pool.workspaces[0].scratch;
    assert_eq!(scratch.lag_accumulator.len(), state.hydro_count);
    assert_eq!(
        scratch.downstream_completed_lags.len(),
        state.hydro_count * setup.inputs.stage_data.study_dims.downstream_par_order
    );
    assert!(scratch.load_rhs_buf.capacity() >= n_load_buses * max_n_blks);
    assert_eq!(scratch.raw_noise_buf.capacity(), 0);
}

#[test]
fn build_training_output_non_empty() {
    use cobre_comm::LocalBackend;
    use cobre_solver::ActiveSolver;

    let system = minimal_system(2);
    let config = minimal_config(1, 2);
    let stochastic = build_stochastic_context(
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
    .expect("stochastic context");

    let mut setup = StudySetup::new(
        &system,
        &config,
        stochastic,
        PrepareHydroModelsResult::default_from_system(&system),
        Vec::new(),
    )
    .expect("setup");
    let comm = LocalBackend;
    let mut solver = ActiveSolver::new().expect("solver");

    let (event_tx, event_rx) = std::sync::mpsc::channel();
    let result = setup
        .train(
            &mut solver,
            &comm,
            1,
            ActiveSolver::new,
            Some(event_tx),
            None,
        )
        .expect("train");

    let events: Vec<cobre_core::TrainingEvent> = event_rx.try_iter().collect();

    let output = setup.build_training_output(&result.result, &events);
    assert!(
        !output.convergence_records.is_empty(),
        "convergence_records must be non-empty after training"
    );
}

#[test]
fn simulate_after_train_returns_nonempty_costs() {
    use cobre_comm::LocalBackend;
    use cobre_solver::ActiveSolver;

    let mut config = minimal_config(1, 3);
    config.simulation = IoSimulationConfig {
        enabled: true,
        selection: Some(SimulationSelection::Sampled { num_scenarios: 3 }),
        io_channel_capacity: 8,
        ..IoSimulationConfig::default()
    };

    let system = minimal_system(2);
    let stochastic = build_stochastic_context(
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
    .expect("stochastic context");

    let mut setup = StudySetup::new(
        &system,
        &config,
        stochastic,
        PrepareHydroModelsResult::default_from_system(&system),
        Vec::new(),
    )
    .expect("setup");

    let comm = LocalBackend;
    let mut solver = ActiveSolver::new().expect("solver");
    setup
        .train(&mut solver, &comm, 1, ActiveSolver::new, None, None)
        .expect("train");

    let mut pool = setup
        .create_workspace_pool(&comm, 1, ActiveSolver::new)
        .expect("sim pool");

    let io_capacity = setup.simulation_config.io_channel_capacity.max(1);
    let (result_tx, result_rx) = std::sync::mpsc::sync_channel(io_capacity);
    let drain_handle = std::thread::spawn(move || result_rx.into_iter().collect::<Vec<_>>());

    let sim_result = setup
        .simulate(&mut pool.workspaces, &comm, &result_tx, None, None, &[])
        .expect("simulate");

    // Drop the sender so the drain thread terminates.
    drop(result_tx);
    let _results = drain_handle.join().expect("drain thread");

    assert!(
        !sim_result.costs.is_empty(),
        "simulate must return at least one cost entry"
    );
    assert_eq!(
        sim_result.solver_stats.len(),
        sim_result.costs.len(),
        "one solver stats entry per scenario"
    );
}

#[test]
fn study_params_from_config_defaults() {
    use super::{DEFAULT_FORWARD_PASSES, DEFAULT_SEED, StudyParams};
    use crate::stopping_rule::{StoppingMode, StoppingRule};

    let config = Config {
        schema: None,
        modeling: ModelingConfig {
            inflow_non_negativity: InflowNonNegativityConfig {
                method: CfgInflowMethod::None,
            },
            cost_scale_factor: None,
        },
        training: TrainingConfig {
            enabled: true,
            tree_seed: None,
            stopping_rules: None,
            stopping_mode: cobre_io::config::StoppingMode::Any,
            cut_selection: RowSelectionConfig::default(),
            solver: TrainingSolverConfig::default(),
            parallelism: cobre_io::config::ParallelismConfig::default(),
            scenario_source: None,
            selection: None,
        },
        upper_bound_evaluation: UpperBoundEvaluationConfig::default(),
        policy: PolicyConfig::default(),
        simulation: IoSimulationConfig::default(),
        exports: ExportsConfig::default(),
        estimation: EstimationConfig::default(),
    };

    let params = StudyParams::from_config(&config, Vec::new()).expect("from_config");

    assert_eq!(
        params.seed, DEFAULT_SEED,
        "seed should default to DEFAULT_SEED"
    );
    assert_eq!(
        params.forward_passes, DEFAULT_FORWARD_PASSES,
        "forward_passes should default to DEFAULT_FORWARD_PASSES"
    );
    assert_eq!(
        params.stopping_rule_set.rules.len(),
        1,
        "expected exactly 1 default stopping rule"
    );
    assert!(
        matches!(
            params.stopping_rule_set.rules[0],
            StoppingRule::IterationLimit { .. }
        ),
        "default rule should be IterationLimit"
    );
    assert!(
        matches!(params.stopping_rule_set.mode, StoppingMode::Any),
        "default stopping mode should be Any"
    );
    assert_eq!(
        params.n_scenarios, 0,
        "n_scenarios should be 0 when simulation disabled"
    );
    assert!(
        params.cut_selection.is_none(),
        "cut_selection should be None by default"
    );
}

#[test]
fn study_params_from_config_explicit() {
    use super::StudyParams;
    use crate::stopping_rule::{StoppingMode, StoppingRule};

    let config = Config {
        schema: None,
        modeling: ModelingConfig {
            inflow_non_negativity: InflowNonNegativityConfig {
                method: CfgInflowMethod::Penalty,
            },
            cost_scale_factor: None,
        },
        training: TrainingConfig {
            enabled: true,
            tree_seed: Some(1234),
            stopping_rules: Some(vec![
                StoppingRuleConfig::IterationLimit { limit: 50 },
                StoppingRuleConfig::TimeLimit { seconds: 60.0 },
            ]),
            stopping_mode: cobre_io::config::StoppingMode::All,
            cut_selection: RowSelectionConfig::default(),
            solver: TrainingSolverConfig::default(),
            parallelism: cobre_io::config::ParallelismConfig::default(),
            scenario_source: None,
            selection: Some(TrainingSelection::Sampled { forward_passes: 5 }),
        },
        upper_bound_evaluation: UpperBoundEvaluationConfig::default(),
        policy: PolicyConfig {
            path: "./my_policy".to_string(),
            ..PolicyConfig::default()
        },
        simulation: IoSimulationConfig {
            enabled: true,
            selection: Some(SimulationSelection::Sampled { num_scenarios: 200 }),
            ..IoSimulationConfig::default()
        },
        exports: ExportsConfig::default(),
        estimation: EstimationConfig::default(),
    };

    let params = StudyParams::from_config(&config, Vec::new()).expect("from_config");

    // Seed: i64::unsigned_abs(1234) == 1234
    assert_eq!(params.seed, 1234, "seed mismatch");
    assert_eq!(params.forward_passes, 5, "forward_passes mismatch");
    assert_eq!(
        params.stopping_rule_set.rules.len(),
        2,
        "stopping rule count mismatch"
    );
    assert!(
        matches!(
            params.stopping_rule_set.rules[0],
            StoppingRule::IterationLimit { limit: 50 }
        ),
        "first rule should be IterationLimit(50)"
    );
    assert!(
        matches!(
            params.stopping_rule_set.rules[1],
            StoppingRule::TimeLimit { seconds } if (seconds - 60.0).abs() < 1e-9
        ),
        "second rule should be TimeLimit(60.0)"
    );
    assert!(
        matches!(params.stopping_rule_set.mode, StoppingMode::All),
        "stopping mode should be All"
    );
    assert_eq!(params.n_scenarios, 200, "n_scenarios mismatch");
    assert_eq!(params.policy_path, "./my_policy", "policy_path mismatch");
}

/// Writes the structural files `validate_structure` requires. The optional
/// estimation and opening-tree files are left out; tests add them as needed.
fn write_minimal_case_dir(root: &std::path::Path) {
    use std::fs;

    fs::create_dir_all(root.join("system")).unwrap();
    fs::write(root.join("config.json"), b"{}").unwrap();
    fs::write(root.join("penalties.json"), b"{}").unwrap();
    fs::write(root.join("stages.json"), b"{}").unwrap();
    fs::write(root.join("initial_conditions.json"), b"{}").unwrap();
    fs::write(root.join("system/buses.json"), b"{}").unwrap();
    fs::write(root.join("system/lines.json"), b"{}").unwrap();
    fs::write(root.join("system/hydros.json"), b"{}").unwrap();
    fs::write(root.join("system/thermals.json"), b"{}").unwrap();
}

fn minimal_prepare_config() -> cobre_io::Config {
    Config {
        schema: None,
        modeling: ModelingConfig {
            inflow_non_negativity: InflowNonNegativityConfig {
                method: CfgInflowMethod::None,
            },
            cost_scale_factor: None,
        },
        training: TrainingConfig {
            enabled: true,
            tree_seed: None,
            stopping_rules: None,
            stopping_mode: StoppingMode::Any,
            cut_selection: RowSelectionConfig::default(),
            solver: TrainingSolverConfig::default(),
            parallelism: cobre_io::config::ParallelismConfig::default(),
            scenario_source: None,
            selection: None,
        },
        upper_bound_evaluation: UpperBoundEvaluationConfig::default(),
        policy: PolicyConfig::default(),
        simulation: IoSimulationConfig::default(),
        exports: ExportsConfig::default(),
        estimation: EstimationConfig::default(),
    }
}

#[test]
fn prepare_stochastic_no_history_no_tree_returns_none_report_and_generated_provenance() {
    use super::prepare_stochastic;
    use cobre_core::scenario::ScenarioSource;
    use cobre_stochastic::provenance::ComponentProvenance;
    use tempfile::TempDir;

    let dir = TempDir::new().unwrap();
    let root = dir.path();
    write_minimal_case_dir(root);

    let system = minimal_system(2);
    let config = minimal_prepare_config();
    let seed = 42_u64;

    let source = ScenarioSource {
        inflow_scheme: SamplingScheme::InSample,
        load_scheme: SamplingScheme::InSample,
        ncs_scheme: SamplingScheme::InSample,
        seed: None,
        historical_years: None,
    };
    let result = prepare_stochastic(system, root, &config, seed, &source, None)
        .expect("prepare_stochastic should succeed with no optional files");

    assert!(
        result.estimation_report.is_none(),
        "estimation_report must be None when no inflow_history.parquet is present"
    );
    assert_eq!(
        result.stochastic.provenance().opening_tree,
        ComponentProvenance::Generated,
        "opening_tree provenance must be Generated when no user tree is supplied"
    );
}

#[test]
fn prepare_stochastic_with_stats_file_present_skips_estimation() {
    use super::prepare_stochastic;
    use cobre_core::scenario::ScenarioSource;
    use std::fs;
    use tempfile::TempDir;

    let dir = TempDir::new().unwrap();
    let root = dir.path();
    write_minimal_case_dir(root);

    // No `inflow_history.parquet`: the estimation-skip path does not need it, and
    // its absence avoids parse errors (`validate_structure` checks existence,
    // not content).
    fs::create_dir_all(root.join("scenarios")).unwrap();
    fs::write(root.join("scenarios/inflow_seasonal_stats.parquet"), b"").unwrap();
    fs::write(root.join("scenarios/inflow_ar_coefficients.parquet"), b"").unwrap();

    let system = minimal_system(2);
    let config = minimal_prepare_config();
    let seed = 42_u64;

    let source = ScenarioSource {
        inflow_scheme: SamplingScheme::InSample,
        load_scheme: SamplingScheme::InSample,
        ncs_scheme: SamplingScheme::InSample,
        seed: None,
        historical_years: None,
    };
    let result = prepare_stochastic(system, root, &config, seed, &source, None)
        .expect("prepare_stochastic should succeed when stats file is present");

    assert!(
        result.estimation_report.is_none(),
        "estimation_report must be None when inflow_seasonal_stats.parquet is present"
    );
}

/// `load_user_opening_tree_inner` is exercised indirectly through
/// `prepare_stochastic`: with no `scenarios/noise_openings.parquet`, the
/// resulting context must not claim `UserSupplied` provenance.
#[test]
fn prepare_stochastic_no_opening_tree_gives_non_user_supplied_provenance() {
    use super::prepare_stochastic;
    use cobre_core::scenario::ScenarioSource;
    use cobre_stochastic::provenance::ComponentProvenance;
    use tempfile::TempDir;

    let dir = TempDir::new().unwrap();
    let root = dir.path();
    write_minimal_case_dir(root);

    let system = minimal_system(2);
    let config = minimal_prepare_config();

    let source = ScenarioSource {
        inflow_scheme: SamplingScheme::InSample,
        load_scheme: SamplingScheme::InSample,
        ncs_scheme: SamplingScheme::InSample,
        seed: None,
        historical_years: None,
    };
    let result = prepare_stochastic(system, root, &config, 0, &source, None)
        .expect("prepare_stochastic must succeed with no opening tree file");

    assert_ne!(
        result.stochastic.provenance().opening_tree,
        ComponentProvenance::UserSupplied,
        "opening_tree provenance must not be UserSupplied when file is absent"
    );
}

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "the fixture spells out one complete study inline so each assertion traces to a literal"
)]
fn test_prepare_stochastic_historical_residuals_noise_method() {
    use super::prepare_stochastic;
    use chrono::NaiveDate;
    use cobre_core::scenario::ScenarioSource;
    use tempfile::TempDir;

    let n_stages = 2usize;

    let bus = Bus {
        id: EntityId(1),
        name: "B1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 500.0,
        }],
        excess_cost: 0.0,
    };
    let thermal = Thermal {
        id: EntityId(2),
        name: "T1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        bus_id: EntityId(1),
        min_generation_mw: 0.0,
        max_generation_mw: 100.0,
        cost_per_mwh: 50.0,
        anticipated_config: None,
        entry_stage_id: None,
        exit_stage_id: None,
    };
    let mut hydro = Hydro {
        unit_groups: Vec::new(),
        id: EntityId(3),
        name: "H1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        downstream_id: None,
        travel_time_hours: None,
        entry_stage_id: None,
        exit_stage_id: None,
        min_storage_hm3: 0.0,
        max_storage_hm3: 200.0,
        min_outflow_m3s: 0.0,
        max_outflow_m3s: None,
        generation_model: HydroGenerationModel::ConstantProductivity,
        min_turbined_m3s: 0.0,
        max_turbined_m3s: 100.0,
        specific_productivity_mw_per_m3s_per_m: None,
        min_generation_mw: 0.0,
        max_generation_mw: 250.0,
        tailrace: None,
        hydraulic_losses: None,
        efficiency: None,
        evaporation_coefficients_mm: None,
        evaporation_reference_volumes_hm3: None,
        diversion: None,
        filling: None,
        penalties: HydroPenalties {
            spillage_cost: 0.01,
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
    hydro.declare_mirror_unit_group(EntityId(1));

    // branching_factor=2 → each stage selects 2 historical windows as openings.
    let stages: Vec<Stage> = (0..n_stages)
        .map(|i| Stage {
            index: i,
            id: i as i32,
            start_date: NaiveDate::from_ymd_opt(2024, (i as u32 % 12) + 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, (i as u32 % 12) + 1, 28).unwrap(),
            season_id: Some(i % 12),
            blocks: vec![Block {
                index: 0,
                name: "S".to_string(),
                duration_hours: 720.0,
            }],
            block_mode: BlockMode::Parallel,
            state_config: StageStateConfig {
                storage: true,
                inflow_lags: false,
            },
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor: 2,
                noise_method: NoiseMethod::HistoricalResiduals,
            },
        })
        .collect();

    let inflow_models: Vec<InflowModel> = (0..n_stages)
        .map(|i| InflowModel {
            hydro_id: EntityId(3),
            stage_id: i as i32,
            mean_m3s: 80.0,
            std_m3s: 20.0,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
            annual: None,
        })
        .collect();

    let load_models: Vec<LoadModel> = (0..n_stages)
        .map(|i| LoadModel {
            bus_id: EntityId(1),
            stage_id: i as i32,
            mean_mw: 100.0,
            std_mw: 0.0,
        })
        .collect();

    // Historical inflow data: 1990 and 1991 cover 12 months each — 2 valid windows.
    let inflow_history: Vec<InflowHistoryRow> = (1990_i32..=1991)
        .flat_map(|year| {
            (1u32..=12).map(move |month| {
                let start_date = NaiveDate::from_ymd_opt(year, month, 1).unwrap();
                InflowHistoryRow {
                    hydro_id: EntityId(3),
                    start_date,
                    end_date: start_date.succ_opt().unwrap(),
                    value_m3s: 80.0 + f64::from(year - 1990) * 5.0,
                }
            })
        })
        .collect();

    let n_st = n_stages.max(1);
    let bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 1,
            n_thermals: 1,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages: n_st,
            k_max: 0,
        },
        &BoundsDefaults {
            hydro: HydroStageBounds {
                min_storage_hm3: 0.0,
                max_storage_hm3: 200.0,
                filling_min_rate_m3s: 0.0,
                water_withdrawal_m3s: 0.0,
            },
            hydro_block: HydroBlockBounds {
                max_turbined_m3s: 100.0,
                max_generation_mw: 250.0,
                ..Default::default()
            },
            thermal: ThermalStageBounds { cost_per_mwh: 0.0 },
            thermal_block: ThermalBlockBounds {
                min_generation_mw: 0.0,
                max_generation_mw: 100.0,
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
    let penalties = ResolvedPenalties::new(
        &PenaltiesCountsSpec {
            n_hydros: 1,
            n_buses: 1,
            n_lines: 0,
            n_ncs: 0,
            n_stages: n_st,
        },
        &PenaltiesDefaults {
            hydro: HydroPenalties {
                spillage_cost: 0.01,
                diversion_cost: 0.0,
                turbined_cost: 0.0,
                storage_violation_below_cost: 500.0,
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
            bus: BusStagePenalties { excess_cost: 0.0 },
            line: LineStagePenalties { exchange_cost: 0.0 },
            ncs: NcsStagePenalties {
                curtailment_cost: 0.0,
            },
        },
    );

    let system = SystemBuilder::new()
        .buses(vec![bus])
        .thermals(vec![thermal])
        .hydros(vec![hydro])
        .stages(stages)
        .inflow_models(inflow_models)
        .load_models(load_models)
        .inflow_history(inflow_history)
        .bounds(bounds)
        .penalties(penalties)
        .build()
        .expect("test system: valid");

    let dir = TempDir::new().unwrap();
    let root = dir.path();
    write_minimal_case_dir(root);

    let config = minimal_prepare_config();
    let source = ScenarioSource {
        inflow_scheme: SamplingScheme::InSample,
        load_scheme: SamplingScheme::InSample,
        ncs_scheme: SamplingScheme::InSample,
        seed: None,
        historical_years: None,
    };
    let result = prepare_stochastic(system, root, &config, 42, &source, None)
        .expect("prepare_stochastic must succeed with HistoricalResiduals noise method");

    assert_eq!(
        result.stochastic.opening_tree().n_stages(),
        n_stages,
        "opening_tree must have n_stages == {n_stages}"
    );
}

#[test]
fn default_from_system_gives_constant_and_no_evaporation() {
    use crate::hydro_models::{EvaporationModel, ProductionModelSource, ResolvedProductionModel};

    let system = minimal_system(2);
    let result = PrepareHydroModelsResult::default_from_system(&system);

    assert_eq!(
        result.provenance.production_sources.len(),
        system.hydros().len(),
        "production_sources length must equal n_hydros"
    );
    for (_, source) in &result.provenance.production_sources {
        assert_eq!(
            *source,
            ProductionModelSource::DefaultConstant,
            "all hydros must use DefaultConstant"
        );
    }

    assert_eq!(
        result.provenance.evaporation_sources.len(),
        system.hydros().len(),
        "evaporation_sources length must equal n_hydros"
    );
    assert!(
        !result.evaporation.has_evaporation(),
        "default result must have no evaporation"
    );

    let model = result.production.model(0, 0);
    assert!(
        matches!(model, ResolvedProductionModel::ConstantProductivity { .. }),
        "default production model must be ConstantProductivity"
    );

    let evap = result.evaporation.model(0);
    assert!(
        matches!(evap, EvaporationModel::None),
        "default evaporation model must be None"
    );
}

#[test]
fn hydro_models_accessor_returns_stored_result() {
    use crate::hydro_models::ProductionModelSource;

    let system = minimal_system(2);
    let config = minimal_config(1, 5);
    let stochastic = build_stochastic_context(
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
    .expect("stochastic context");
    let hydro_result = PrepareHydroModelsResult::default_from_system(&system);

    let setup =
        StudySetup::new(&system, &config, stochastic, hydro_result, Vec::new()).expect("setup");

    let models = &setup.hydro_models;
    assert_eq!(
        models.provenance.production_sources.len(),
        system.hydros().len(),
        "hydro_models() must return the stored result (provenance length mismatch)"
    );
    for (_, source) in &models.provenance.production_sources {
        assert_eq!(
            *source,
            ProductionModelSource::DefaultConstant,
            "stored result must preserve DefaultConstant provenance"
        );
    }
}

/// The hydro has `ρ_eq=2.5` and no downstream, so `ρ_acum=2.5` at every stage.
#[test]
fn energy_conversion_accessor_returns_built_set() {
    let system = minimal_system(2);
    let config = minimal_config(1, 5);
    let stochastic = build_stochastic_context(
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
    .expect("stochastic context");

    // default_from_system seeds productivity 0.0; supply 2.5 so the ρ_acum
    // assertion has a non-zero expected value.
    let n_study_stages = system.stages().iter().filter(|s| s.id >= 0).count();
    let hydro_models_result = {
        let mut result = PrepareHydroModelsResult::default_from_system(&system);
        result.production = ProductionModelSet::new(
            vec![vec![
                ResolvedProductionModel::ConstantProductivity {
                    productivity: 2.5
                };
                n_study_stages
            ]],
            system.hydros(),
            n_study_stages,
        );
        result
    };

    let setup = StudySetup::new(
        &system,
        &config,
        stochastic,
        hydro_models_result,
        Vec::new(),
    )
    .expect("setup");

    let ec = setup.energy_conversion();
    assert_eq!(ec.n_hydros(), system.hydros().len());
    for s in 0..ec.n_stages() {
        assert!(
            (ec.accumulated_productivity(0, s) - 2.5).abs() < f64::EPSILON,
            "stage {s}: expected ρ_acum=2.5, got {}",
            ec.accumulated_productivity(0, s)
        );
    }
}

#[test]
fn study_setup_propagates_fpha_missing_equivalent_productivity() {
    let system = minimal_fpha_misconfigured_system(2);
    let config = minimal_config(1, 5);
    let stochastic = build_stochastic_context(
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
    .expect("stochastic context");

    let err = StudySetup::new(
        &system,
        &config,
        stochastic,
        PrepareHydroModelsResult::default_from_system(&system),
        Vec::new(),
    )
    .expect_err("setup must reject misconfigured FPHA hydro");

    let msg = err.to_string();
    assert!(
        msg.contains("cannot derive ρ_eq"),
        "error must come from FphaMissingEquivalentProductivity Display; got: {msg}"
    );
    assert!(
        msg.contains("H_FPHA_BAD"),
        "error must mention the offending hydro by name; got: {msg}"
    );
}

fn layout_for_lag_test(hydro_count: usize, max_par_order: usize) -> StateSpace {
    test_support::state_layout(hydro_count, max_par_order)
}

/// Must match [`counts_with_anticipated`]'s `n_anticipated` (`k_values.len()`):
/// 1 hydro, 0 lags, one plant per `k_values` entry.
fn layout_with_anticipated(k_values: &[usize]) -> StateSpace {
    test_support::state_layout_full(1, 0, k_values.to_vec())
}

/// 2-hydro PAR(2) system with `inflow_lags`, `season_map`, and
/// `inflow_history` threaded through (empty/`None` when a caller does not
/// need them) so a caller can exercise the derived-seed path end-to-end.
#[expect(
    clippy::too_many_lines,
    clippy::items_after_statements,
    reason = "the fixture spells out one complete study inline so each assertion traces to a literal, and fixture-local helper items sit beside the entities that use them"
)]
fn minimal_system_2_hydros_with_history(
    n_stages: usize,
    season_map: Option<SeasonMap>,
    inflow_history: Vec<InflowHistoryRow>,
) -> cobre_core::System {
    use chrono::NaiveDate;

    let bus = Bus {
        id: EntityId(1),
        name: "B1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 500.0,
        }],
        excess_cost: 0.0,
    };

    let make_hydro = |id: i32, name: &str| {
        let mut hydro = Hydro {
            unit_groups: Vec::new(),
            id: EntityId(id),
            name: name.to_string(),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            downstream_id: None,
            travel_time_hours: None,
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: 0.0,
            max_storage_hm3: 200.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: HydroGenerationModel::ConstantProductivity,
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 100.0,
            specific_productivity_mw_per_m3s_per_m: None,
            min_generation_mw: 0.0,
            max_generation_mw: 250.0,
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: None,
            evaporation_reference_volumes_hm3: None,
            diversion: None,
            filling: None,
            penalties: HydroPenalties {
                spillage_cost: 0.01,
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
        hydro.declare_mirror_unit_group(EntityId(1));
        hydro
    };

    let n_st = n_stages.max(1);
    let stages: Vec<Stage> = (0..n_stages)
        .map(|i| Stage {
            index: i,
            id: i as i32,
            start_date: NaiveDate::from_ymd_opt(2020, (i % 12 + 1) as u32, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(
                if (i % 12 + 1) == 12 { 2021 } else { 2020 },
                ((i % 12 + 1) % 12 + 1) as u32,
                1,
            )
            .unwrap(),
            season_id: Some(i),
            blocks: vec![Block {
                index: 0,
                name: "S".to_string(),
                duration_hours: 744.0,
            }],
            block_mode: BlockMode::Parallel,
            state_config: StageStateConfig {
                storage: true,
                inflow_lags: true,
            },
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor: 1,
                noise_method: NoiseMethod::Saa,
            },
        })
        .collect();

    let inflow_models: Vec<InflowModel> = [1_i32, 2]
        .into_iter()
        .flat_map(|hid| {
            (0..n_stages).map(move |i| InflowModel {
                hydro_id: EntityId(hid),
                stage_id: i as i32,
                mean_m3s: 80.0,
                std_m3s: 20.0,
                ar_coefficients: vec![0.5, 0.3],
                residual_std_ratio: 0.8,
                annual: None,
            })
        })
        .collect();

    let load_models: Vec<LoadModel> = (0..n_stages)
        .map(|i| LoadModel {
            bus_id: EntityId(1),
            stage_id: i as i32,
            mean_mw: 100.0,
            std_mw: 0.0,
        })
        .collect();

    fn default_hydro_bounds() -> HydroStageBounds {
        HydroStageBounds {
            min_storage_hm3: 0.0,
            max_storage_hm3: 200.0,
            filling_min_rate_m3s: 0.0,
            water_withdrawal_m3s: 0.0,
        }
    }

    fn default_hydro_block_bounds() -> HydroBlockBounds {
        HydroBlockBounds {
            max_turbined_m3s: 100.0,
            max_generation_mw: 250.0,
            ..Default::default()
        }
    }

    fn default_hydro_penalties() -> HydroPenalties {
        HydroPenalties {
            spillage_cost: 0.01,
            diversion_cost: 0.0,
            turbined_cost: 0.0,
            storage_violation_below_cost: 500.0,
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

    let bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 2,
            n_thermals: 0,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages: n_st,
            k_max: 0,
        },
        &BoundsDefaults {
            hydro: default_hydro_bounds(),
            hydro_block: default_hydro_block_bounds(),
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

    let penalties = ResolvedPenalties::new(
        &PenaltiesCountsSpec {
            n_hydros: 2,
            n_buses: 1,
            n_lines: 0,
            n_ncs: 0,
            n_stages: n_st,
        },
        &PenaltiesDefaults {
            hydro: default_hydro_penalties(),
            bus: BusStagePenalties { excess_cost: 0.0 },
            line: LineStagePenalties { exchange_cost: 0.0 },
            ncs: NcsStagePenalties {
                curtailment_cost: 0.0,
            },
        },
    );

    SystemBuilder::new()
        .buses(vec![bus])
        .thermals(vec![])
        .hydros(vec![make_hydro(1, "H1"), make_hydro(2, "H2")])
        .stages(stages)
        .inflow_models(inflow_models)
        .load_models(load_models)
        .inflow_history(inflow_history)
        .bounds(bounds)
        .penalties(penalties)
        .policy_graph(HorizonGraph {
            stage_discount_rate_overrides: std::collections::BTreeMap::new(),
            graph_type: PolicyGraphType::FiniteHorizon,
            annual_discount_rate: 0.0,
            transitions: vec![],
            nodes: Vec::new(),
            season_map,
        })
        .initial_conditions(cobre_core::InitialConditions {
            storage: vec![],
            filling_storage: vec![],
            past_anticipated_commitments: vec![],
            recent_observations: vec![],
            past_defluences: vec![],
        })
        .build()
        .expect("minimal_system_2_hydros_with_history: valid")
}

/// A default (no PAR model) is `Ok` regardless of the system's shape.
#[test]
fn validate_par_shape_default_is_ok() {
    let system = minimal_system_2_hydros_with_history(2, None, vec![]);
    assert!(validate_par_shape(&system, &PrecomputedPar::default()).is_ok());
}

/// A PAR built over the system's own hydros and study stages is `Ok`.
#[test]
fn validate_par_shape_matching_shape_is_ok() {
    let system = minimal_system_2_hydros_with_history(2, None, vec![]);
    let hydro_ids: Vec<_> = system.hydros().iter().map(|h| h.id).collect();
    let par = PrecomputedPar::build(system.inflow_models(), system.stages(), &hydro_ids, None)
        .expect("par build ok");
    assert!(validate_par_shape(&system, &par).is_ok());
}

/// A PAR built over only one of the system's two hydros is rejected, naming
/// both shapes.
#[test]
fn validate_par_shape_hydro_count_mismatch_is_err() {
    let system = minimal_system_2_hydros_with_history(2, None, vec![]);
    let hydro_ids: Vec<_> = system.hydros().iter().take(1).map(|h| h.id).collect();
    let par = PrecomputedPar::build(system.inflow_models(), system.stages(), &hydro_ids, None)
        .expect("par build ok");
    let err = validate_par_shape(&system, &par).expect_err("hydro count mismatch must be rejected");
    let SddpError::Validation(msg) = err else {
        panic!("expected SddpError::Validation, got {err:?}");
    };
    assert!(msg.contains("shape mismatch"), "message: {msg}");
    assert!(msg.contains("1 hydros"), "message: {msg}");
    assert!(msg.contains("2 hydros"), "message: {msg}");
}

/// A PAR built over one fewer stage than the system declares is rejected.
#[test]
fn validate_par_shape_stage_count_mismatch_is_err() {
    let system = minimal_system_2_hydros_with_history(2, None, vec![]);
    let hydro_ids: Vec<_> = system.hydros().iter().map(|h| h.id).collect();
    let short_stages = &system.stages()[..system.stages().len() - 1];
    let par = PrecomputedPar::build(system.inflow_models(), short_stages, &hydro_ids, None)
        .expect("par build ok");
    let err = validate_par_shape(&system, &par).expect_err("stage count mismatch must be rejected");
    assert!(matches!(err, SddpError::Validation(_)));
}

/// 12-month `Monthly` season map (`id == month_start - 1`), matching
/// [`minimal_system_2_hydros_with_history`]'s `season_id: Some(i)` stage
/// convention.
fn monthly_season_map_for_lag_seed_test() -> SeasonMap {
    let seasons = (0..12u32)
        .map(|i| cobre_core::temporal::SeasonDefinition {
            id: i as usize,
            label: format!("Month{}", i + 1),
            month_start: i + 1,
            day_start: None,
            month_end: None,
            day_end: None,
        })
        .collect();
    SeasonMap {
        cycle_type: cobre_core::temporal::SeasonCycleType::Monthly,
        seasons,
    }
}

/// Expected lag seeds — hydro 0 (id=1): lag0=600, lag1=500;
/// hydro 1 (id=2): lag0=200, lag1=100.
#[test]
fn build_initial_state_populates_lags_from_derived_values() {
    use super::build_initial_state;

    let system = minimal_system_2_hydros_with_history(1, None, vec![]);
    let layout = layout_for_lag_test(2, 2);
    // Entity-major: hydro 0 (id=1) at [0]=lag0, [1]=lag1; hydro 1 (id=2) at
    // [2]=lag0, [3]=lag1.
    let derived_lag_values = [600.0, 500.0, 200.0, 100.0];

    let state = build_initial_state(
        &system,
        &test_support::study_dims(),
        &layout,
        &derived_lag_values,
    );

    // State layout: storage(0..2), lags(2..6) in lag-major order.
    // Lag-major: slot = s + lag * N + h, where N = 2.
    let s = layout.inflow_lags.start;
    assert!(
        (state[s] - 600.0).abs() < 1e-10,
        "lag0 hydro 0: expected 600.0, got {}",
        state[s]
    );
    assert!(
        (state[s + 1] - 200.0).abs() < 1e-10,
        "lag0 hydro 1: expected 200.0, got {}",
        state[s + 1]
    );
    assert!(
        (state[s + 2] - 500.0).abs() < 1e-10,
        "lag1 hydro 0: expected 500.0, got {}",
        state[s + 2]
    );
    assert!(
        (state[s + 3] - 100.0).abs() < 1e-10,
        "lag1 hydro 1: expected 100.0, got {}",
        state[s + 3]
    );
    assert_eq!(
        state.len(),
        layout.n_state,
        "state length must equal n_state"
    );
}

#[test]
fn build_initial_state_zero_derived_lag_values_leaves_zero_lags() {
    use super::build_initial_state;

    let system = minimal_system(2);
    let layout = layout_for_lag_test(1, 3);
    let derived_lag_values = [0.0; 3];

    let state = build_initial_state(
        &system,
        &test_support::study_dims(),
        &layout,
        &derived_lag_values,
    );

    let s = layout.inflow_lags.start;
    for l in 0..3 {
        assert!(
            state[s + l].abs() < 1e-10,
            "lag slot {l} should be 0.0 when the derived lag values are zero, got {}",
            state[s + l]
        );
    }
}

/// No `recent_observations`, a monthly full-coverage `inflow_history` record:
/// the derived lag block orders December 2019 as lag0 and November 2019 as
/// lag1, per hydro.
#[test]
fn build_initial_state_derived_lags_match_positional_seed() {
    use super::build_initial_state;
    use cobre_stochastic::derive_inflow_seeds;

    let inflow_history = vec![
        InflowHistoryRow {
            hydro_id: EntityId(1),
            start_date: chrono::NaiveDate::from_ymd_opt(2019, 12, 1).unwrap(),
            end_date: chrono::NaiveDate::from_ymd_opt(2020, 1, 1).unwrap(),
            value_m3s: 600.0,
        },
        InflowHistoryRow {
            hydro_id: EntityId(1),
            start_date: chrono::NaiveDate::from_ymd_opt(2019, 11, 1).unwrap(),
            end_date: chrono::NaiveDate::from_ymd_opt(2019, 12, 1).unwrap(),
            value_m3s: 500.0,
        },
        InflowHistoryRow {
            hydro_id: EntityId(2),
            start_date: chrono::NaiveDate::from_ymd_opt(2019, 12, 1).unwrap(),
            end_date: chrono::NaiveDate::from_ymd_opt(2020, 1, 1).unwrap(),
            value_m3s: 200.0,
        },
        InflowHistoryRow {
            hydro_id: EntityId(2),
            start_date: chrono::NaiveDate::from_ymd_opt(2019, 11, 1).unwrap(),
            end_date: chrono::NaiveDate::from_ymd_opt(2019, 12, 1).unwrap(),
            value_m3s: 100.0,
        },
    ];
    let system = minimal_system_2_hydros_with_history(
        3,
        Some(monthly_season_map_for_lag_seed_test()),
        inflow_history,
    );
    let layout = layout_for_lag_test(2, 2);
    let first_stage = system
        .stages()
        .iter()
        .find(|s| s.id >= 0)
        .expect("study has a stage");
    let season_map = system
        .policy_graph()
        .season_map
        .as_ref()
        .expect("season map is set");

    let seeds = derive_inflow_seeds(
        system.inflow_history(),
        &system.initial_conditions().recent_observations,
        system.hydros(),
        first_stage,
        season_map,
        layout.max_par_order,
    );

    let state = build_initial_state(
        &system,
        &test_support::study_dims(),
        &layout,
        &seeds.lag_values,
    );

    // Lag-major: slot = s + lag * N + h, with h1=[600, 500] and h2=[200, 100].
    let s = layout.inflow_lags.start;
    assert!(
        (state[s] - 600.0).abs() < 1e-10,
        "lag0 hydro 0: expected 600.0, got {}",
        state[s]
    );
    assert!(
        (state[s + 1] - 200.0).abs() < 1e-10,
        "lag0 hydro 1: expected 200.0, got {}",
        state[s + 1]
    );
    assert!(
        (state[s + 2] - 500.0).abs() < 1e-10,
        "lag1 hydro 0: expected 500.0, got {}",
        state[s + 2]
    );
    assert!(
        (state[s + 3] - 100.0).abs() < 1e-10,
        "lag1 hydro 1: expected 100.0, got {}",
        state[s + 3]
    );
}

/// Build a 2-hydro system where hydro id=1 (the SMALLER id) has a LATER
/// `operational_start_date` than hydro id=2 (the LARGER id), so the canonical
/// `(operational_start_date, id)` order (`System::hydros()`) is
/// `[id=2, id=1]` — id-DESCENDING, not id-ascending. Accepts a caller-supplied
/// `InitialConditions` so a test can seed `storage` per hydro and check each
/// lands on its OWN coordinate.
#[expect(
    clippy::too_many_lines,
    clippy::items_after_statements,
    reason = "the fixture spells out one complete study inline so each assertion traces to a literal, and fixture-local helper items sit beside the entities that use them"
)]
fn staggered_dates_system_2_hydros(
    n_stages: usize,
    initial_conditions: cobre_core::InitialConditions,
) -> cobre_core::System {
    use chrono::NaiveDate;

    let bus = Bus {
        id: EntityId(1),
        name: "B1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 500.0,
        }],
        excess_cost: 0.0,
    };

    let make_hydro = |id: i32, name: &str, start: NaiveDate| {
        let mut hydro = Hydro {
            unit_groups: Vec::new(),
            id: EntityId(id),
            name: name.to_string(),
            operational_start_date: start,
            downstream_id: None,
            travel_time_hours: None,
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: 0.0,
            max_storage_hm3: 200.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: HydroGenerationModel::ConstantProductivity,
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 100.0,
            specific_productivity_mw_per_m3s_per_m: None,
            min_generation_mw: 0.0,
            max_generation_mw: 250.0,
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: None,
            evaporation_reference_volumes_hm3: None,
            diversion: None,
            filling: None,
            penalties: HydroPenalties {
                spillage_cost: 0.01,
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
        hydro.declare_mirror_unit_group(EntityId(1));
        hydro
    };

    let earlier = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();
    let later = NaiveDate::from_ymd_opt(2025, 6, 1).unwrap();

    let n_st = n_stages.max(1);
    let stages: Vec<Stage> = (0..n_stages)
        .map(|i| Stage {
            index: i,
            id: i as i32,
            start_date: NaiveDate::from_ymd_opt(2020, (i % 12 + 1) as u32, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(
                if (i % 12 + 1) == 12 { 2021 } else { 2020 },
                ((i % 12 + 1) % 12 + 1) as u32,
                1,
            )
            .unwrap(),
            season_id: Some(i),
            blocks: vec![Block {
                index: 0,
                name: "S".to_string(),
                duration_hours: 744.0,
            }],
            block_mode: BlockMode::Parallel,
            state_config: StageStateConfig {
                storage: true,
                inflow_lags: true,
            },
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor: 1,
                noise_method: NoiseMethod::Saa,
            },
        })
        .collect();

    let inflow_models: Vec<InflowModel> = (0..n_stages)
        .flat_map(|i| {
            [1_i32, 2].map(|hid| InflowModel {
                hydro_id: EntityId(hid),
                stage_id: i as i32,
                mean_m3s: 80.0,
                std_m3s: 20.0,
                ar_coefficients: vec![0.5, 0.3],
                residual_std_ratio: 0.8,
                annual: None,
            })
        })
        .collect();

    let load_models: Vec<LoadModel> = (0..n_stages)
        .map(|i| LoadModel {
            bus_id: EntityId(1),
            stage_id: i as i32,
            mean_mw: 100.0,
            std_mw: 0.0,
        })
        .collect();

    fn default_hydro_bounds() -> HydroStageBounds {
        HydroStageBounds {
            min_storage_hm3: 0.0,
            max_storage_hm3: 200.0,
            filling_min_rate_m3s: 0.0,
            water_withdrawal_m3s: 0.0,
        }
    }

    fn default_hydro_block_bounds() -> HydroBlockBounds {
        HydroBlockBounds {
            max_turbined_m3s: 100.0,
            max_generation_mw: 250.0,
            ..Default::default()
        }
    }

    fn default_hydro_penalties() -> HydroPenalties {
        HydroPenalties {
            spillage_cost: 0.01,
            diversion_cost: 0.0,
            turbined_cost: 0.0,
            storage_violation_below_cost: 500.0,
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

    let bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 2,
            n_thermals: 0,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages: n_st,
            k_max: 0,
        },
        &BoundsDefaults {
            hydro: default_hydro_bounds(),
            hydro_block: default_hydro_block_bounds(),
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

    let penalties = ResolvedPenalties::new(
        &PenaltiesCountsSpec {
            n_hydros: 2,
            n_buses: 1,
            n_lines: 0,
            n_ncs: 0,
            n_stages: n_st,
        },
        &PenaltiesDefaults {
            hydro: default_hydro_penalties(),
            bus: BusStagePenalties { excess_cost: 0.0 },
            line: LineStagePenalties { exchange_cost: 0.0 },
            ncs: NcsStagePenalties {
                curtailment_cost: 0.0,
            },
        },
    );

    let system = SystemBuilder::new()
        .buses(vec![bus])
        .thermals(vec![])
        .hydros(vec![
            make_hydro(1, "H1_later", later),
            make_hydro(2, "H2_earlier", earlier),
        ])
        .stages(stages)
        .inflow_models(inflow_models)
        .load_models(load_models)
        .bounds(bounds)
        .penalties(penalties)
        .initial_conditions(initial_conditions)
        .build()
        .expect("staggered_dates_system_2_hydros: valid");

    // Canonical order sorts by (operational_start_date, id): the earlier-date
    // hydro (id=2) must land at position 0, the later-date hydro (id=1) at
    // position 1 — id-descending. A test relying on this fixture to trigger
    // the staggered-order bug depends on this invariant holding.
    assert_eq!(system.hydros()[0].id, EntityId(2));
    assert_eq!(system.hydros()[1].id, EntityId(1));

    system
}

/// Regression: under a STAGGERED-commissioning system where the canonical
/// `(operational_start_date, id)` hydro order is id-DESCENDING (hydro id=1's
/// later commissioning date sorts it after hydro id=2), `build_initial_state`
/// must seed each hydro's OWN declared storage and lag values —
/// `binary_search_by_key` over `hydros()` requires id-ascending order and
/// silently drops the out-of-order record to the default `0.0`. The lag block
/// is fed positionally via `derived_lag_values` (entity-major, position 0 =
/// hydro id=2, position 1 = hydro id=1 — the canonical order), so this also
/// exercises that `build_initial_state` trusts the caller's pre-ordering with
/// no id lookup of its own.
#[test]
fn test_initial_state_seeds_correctly_under_staggered_commissioning_dates() {
    use super::build_initial_state;

    let h1_storage = 111.0_f64;
    let h2_storage = 222.0_f64;
    let h1_past = [11.0_f64, 12.0_f64];
    let h2_past = [21.0_f64, 22.0_f64];

    let ic = cobre_core::InitialConditions {
        storage: vec![
            cobre_core::HydroStorage {
                hydro_id: EntityId(1),
                value_hm3: h1_storage,
            },
            cobre_core::HydroStorage {
                hydro_id: EntityId(2),
                value_hm3: h2_storage,
            },
        ],
        filling_storage: vec![],
        past_anticipated_commitments: vec![],
        recent_observations: vec![],
        past_defluences: vec![],
    };
    let system = staggered_dates_system_2_hydros(1, ic);
    let layout = layout_for_lag_test(2, 2);
    // Entity-major, canonical position order [id=2, id=1]: position 0 (h2)
    // carries h2_past, position 1 (h1) carries h1_past.
    let derived_lag_values = [h2_past[0], h2_past[1], h1_past[0], h1_past[1]];

    let state = build_initial_state(
        &system,
        &test_support::study_dims(),
        &layout,
        &derived_lag_values,
    );

    // Storage: state[0] is hydro id=2's own coordinate (canonical position 0),
    // state[1] is hydro id=1's own coordinate (canonical position 1).
    assert!(
        (state[0] - h2_storage).abs() < 1e-10,
        "hydro id=2 (canonical position 0) storage should be its own IC value {h2_storage}, got {}",
        state[0]
    );
    assert!(
        (state[1] - h1_storage).abs() < 1e-10,
        "hydro id=1 (canonical position 1) storage should be its own IC value {h1_storage}, got {}",
        state[1]
    );

    // Lag block: lag-major layout, slot = lag_start + lag * N + idx.
    let s = layout.inflow_lags.start;
    assert!(
        (state[s] - h2_past[0]).abs() < 1e-10,
        "hydro id=2 lag0 should be its own derived value {}, got {}",
        h2_past[0],
        state[s]
    );
    assert!(
        (state[s + 1] - h1_past[0]).abs() < 1e-10,
        "hydro id=1 lag0 should be its own derived value {}, got {}",
        h1_past[0],
        state[s + 1]
    );
    assert!(
        (state[s + 2] - h2_past[1]).abs() < 1e-10,
        "hydro id=2 lag1 should be its own derived value {}, got {}",
        h2_past[1],
        state[s + 2]
    );
    assert!(
        (state[s + 3] - h1_past[1]).abs() < 1e-10,
        "hydro id=1 lag1 should be its own derived value {}, got {}",
        h1_past[1],
        state[s + 3]
    );
}

/// 2-hydro fixture: hydro 2 is a filling reservoir, hydro 1 operating, with
/// caller-supplied `initial_conditions`. `start_stage_id` sets hydro 2's
/// filling start stage (0 = mid-filling seed; >0 = empty pit).
#[expect(
    clippy::too_many_lines,
    clippy::items_after_statements,
    reason = "the fixture spells out one complete study inline so each assertion traces to a literal, and fixture-local helper items sit beside the entities that use them"
)]
fn filling_system_2_hydros(
    n_stages: usize,
    start_stage_id: i32,
    initial_conditions: cobre_core::InitialConditions,
) -> cobre_core::System {
    use chrono::NaiveDate;

    let bus = Bus {
        id: EntityId(1),
        name: "B1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 500.0,
        }],
        excess_cost: 0.0,
    };

    let make_hydro = |id: i32, name: &str, filling: Option<cobre_core::FillingConfig>| {
        let mut hydro = Hydro {
            unit_groups: Vec::new(),
            id: EntityId(id),
            name: name.to_string(),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            downstream_id: None,
            travel_time_hours: None,
            // A filling hydro requires `entry_stage_id` (the operating-handoff
            // stage) to be `Some`; the system builder rejects `filling` without
            // it. Operating hydros leave it `None`.
            entry_stage_id: filling.as_ref().map(|f| f.start_stage_id + 1),
            exit_stage_id: None,
            min_storage_hm3: 0.0,
            max_storage_hm3: 200.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: HydroGenerationModel::ConstantProductivity,
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 100.0,
            specific_productivity_mw_per_m3s_per_m: None,
            min_generation_mw: 0.0,
            max_generation_mw: 250.0,
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: None,
            evaporation_reference_volumes_hm3: None,
            diversion: None,
            filling,
            penalties: HydroPenalties {
                spillage_cost: 0.01,
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
        hydro.declare_mirror_unit_group(EntityId(1));
        hydro
    };

    let n_st = n_stages.max(1);
    let stages: Vec<Stage> = (0..n_stages)
        .map(|i| Stage {
            index: i,
            id: i as i32,
            start_date: NaiveDate::from_ymd_opt(2020, (i % 12 + 1) as u32, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(
                if (i % 12 + 1) == 12 { 2021 } else { 2020 },
                ((i % 12 + 1) % 12 + 1) as u32,
                1,
            )
            .unwrap(),
            season_id: Some(i),
            blocks: vec![Block {
                index: 0,
                name: "S".to_string(),
                duration_hours: 744.0,
            }],
            block_mode: BlockMode::Parallel,
            state_config: StageStateConfig {
                storage: true,
                inflow_lags: true,
            },
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor: 1,
                noise_method: NoiseMethod::Saa,
            },
        })
        .collect();

    let inflow_models: Vec<InflowModel> = (0..n_stages)
        .flat_map(|i| {
            [1_i32, 2].map(|hid| InflowModel {
                hydro_id: EntityId(hid),
                stage_id: i as i32,
                mean_m3s: 80.0,
                std_m3s: 20.0,
                ar_coefficients: vec![0.5, 0.3],
                residual_std_ratio: 0.8,
                annual: None,
            })
        })
        .collect();

    let load_models: Vec<LoadModel> = (0..n_stages)
        .map(|i| LoadModel {
            bus_id: EntityId(1),
            stage_id: i as i32,
            mean_mw: 100.0,
            std_mw: 0.0,
        })
        .collect();

    fn default_hydro_bounds() -> HydroStageBounds {
        HydroStageBounds {
            min_storage_hm3: 0.0,
            max_storage_hm3: 200.0,
            filling_min_rate_m3s: 0.0,
            water_withdrawal_m3s: 0.0,
        }
    }

    fn default_hydro_block_bounds() -> HydroBlockBounds {
        HydroBlockBounds {
            max_turbined_m3s: 100.0,
            max_generation_mw: 250.0,
            ..Default::default()
        }
    }

    fn default_hydro_penalties() -> HydroPenalties {
        HydroPenalties {
            spillage_cost: 0.01,
            diversion_cost: 0.0,
            turbined_cost: 0.0,
            storage_violation_below_cost: 500.0,
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

    let bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 2,
            n_thermals: 0,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages: n_st,
            k_max: 0,
        },
        &BoundsDefaults {
            hydro: default_hydro_bounds(),
            hydro_block: default_hydro_block_bounds(),
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

    let penalties = ResolvedPenalties::new(
        &PenaltiesCountsSpec {
            n_hydros: 2,
            n_buses: 1,
            n_lines: 0,
            n_ncs: 0,
            n_stages: n_st,
        },
        &PenaltiesDefaults {
            hydro: default_hydro_penalties(),
            bus: BusStagePenalties { excess_cost: 0.0 },
            line: LineStagePenalties { exchange_cost: 0.0 },
            ncs: NcsStagePenalties {
                curtailment_cost: 0.0,
            },
        },
    );

    let filling = Some(cobre_core::FillingConfig {
        start_stage_id,
        filling_min_rate_m3s: 50.0,
    });

    SystemBuilder::new()
        .buses(vec![bus])
        .thermals(vec![])
        .hydros(vec![
            make_hydro(1, "H1", None),
            make_hydro(2, "H2_FILL", filling),
        ])
        .stages(stages)
        .inflow_models(inflow_models)
        .load_models(load_models)
        .bounds(bounds)
        .penalties(penalties)
        .initial_conditions(initial_conditions)
        .build()
        .expect("filling_system_2_hydros: valid")
}

#[test]
fn build_initial_state_seeds_filling_storage() {
    use super::build_initial_state;

    let seed = 120.0_f64;
    let ic = cobre_core::InitialConditions {
        storage: vec![],
        filling_storage: vec![cobre_core::HydroStorage {
            hydro_id: EntityId(2),
            value_hm3: seed,
        }],
        past_anticipated_commitments: vec![],
        recent_observations: vec![],
        past_defluences: vec![],
    };
    let system = filling_system_2_hydros(1, 0, ic);
    let layout = layout_for_lag_test(2, 2);

    let state = build_initial_state(&system, &test_support::study_dims(), &layout, &[0.0; 4]);

    // Hydro id=2 is at system index 1; its storage coordinate is state[1].
    assert!(
        (state[1] - seed).abs() < 1e-10,
        "filling hydro storage coordinate should carry the seed {seed}, got {}",
        state[1]
    );
    // The operating hydro (id=1) had no storage IC, so its coordinate is 0.
    assert!(
        state[0].abs() < 1e-10,
        "non-seeded operating hydro storage should be 0.0, got {}",
        state[0]
    );
}

#[test]
fn build_initial_state_filling_empty_pit_is_zero() {
    use super::build_initial_state;

    let ic = cobre_core::InitialConditions {
        storage: vec![],
        filling_storage: vec![cobre_core::HydroStorage {
            hydro_id: EntityId(2),
            value_hm3: 0.0,
        }],
        past_anticipated_commitments: vec![],
        recent_observations: vec![],
        past_defluences: vec![],
    };
    let system = filling_system_2_hydros(1, 1, ic);
    let layout = layout_for_lag_test(2, 2);

    let state = build_initial_state(&system, &test_support::study_dims(), &layout, &[0.0; 4]);

    assert!(
        state[1].abs() < 1e-10,
        "empty-pit filling hydro storage should be 0.0, got {}",
        state[1]
    );
}

#[test]
fn build_initial_state_unknown_filling_hydro_skipped() {
    use super::build_initial_state;

    let layout = layout_for_lag_test(2, 2);
    let study_dims = test_support::study_dims();

    let baseline_ic = cobre_core::InitialConditions {
        storage: vec![],
        filling_storage: vec![],
        past_anticipated_commitments: vec![],
        recent_observations: vec![],
        past_defluences: vec![],
    };
    let baseline_system = filling_system_2_hydros(1, 0, baseline_ic);
    let baseline = build_initial_state(&baseline_system, &study_dims, &layout, &[0.0; 4]);

    let ic = cobre_core::InitialConditions {
        storage: vec![],
        filling_storage: vec![cobre_core::HydroStorage {
            hydro_id: EntityId(99),
            value_hm3: 150.0,
        }],
        past_anticipated_commitments: vec![],
        recent_observations: vec![],
        past_defluences: vec![],
    };
    let system = filling_system_2_hydros(1, 0, ic);
    let state = build_initial_state(&system, &study_dims, &layout, &[0.0; 4]);

    assert_eq!(
        state, baseline,
        "an unknown filling hydro_id must be silently skipped, leaving the \
             no-filling baseline state unchanged"
    );
}

#[test]
fn build_initial_state_mixed_operating_and_filling_seeds() {
    use super::build_initial_state;

    let operating_seed = 175.0_f64;
    let filling_seed = 90.0_f64;
    let ic = cobre_core::InitialConditions {
        storage: vec![cobre_core::HydroStorage {
            hydro_id: EntityId(1),
            value_hm3: operating_seed,
        }],
        filling_storage: vec![cobre_core::HydroStorage {
            hydro_id: EntityId(2),
            value_hm3: filling_seed,
        }],
        past_anticipated_commitments: vec![],
        recent_observations: vec![],
        past_defluences: vec![],
    };
    let system = filling_system_2_hydros(1, 0, ic);
    let layout = layout_for_lag_test(2, 2);
    // Entity-major: hydro 0 (id=1) at [0]=lag0, [1]=lag1; hydro 1 (id=2) at
    // [2]=lag0, [3]=lag1 — identical to the operating-only case.
    let derived_lag_values = [600.0, 500.0, 200.0, 100.0];

    let state = build_initial_state(
        &system,
        &test_support::study_dims(),
        &layout,
        &derived_lag_values,
    );

    assert!(
        (state[0] - operating_seed).abs() < 1e-10,
        "operating hydro storage should be {operating_seed}, got {}",
        state[0]
    );
    assert!(
        (state[1] - filling_seed).abs() < 1e-10,
        "filling hydro storage should be {filling_seed}, got {}",
        state[1]
    );

    // Lag block, identical to the operating-only case (lag-major: slot = s +
    // lag * N + h, N = 2).
    let s = layout.inflow_lags.start;
    assert!(
        (state[s] - 600.0).abs() < 1e-10,
        "lag0 hydro 0: expected 600.0, got {}",
        state[s]
    );
    assert!(
        (state[s + 1] - 200.0).abs() < 1e-10,
        "lag0 hydro 1: expected 200.0, got {}",
        state[s + 1]
    );
    assert!(
        (state[s + 2] - 500.0).abs() < 1e-10,
        "lag1 hydro 0: expected 500.0, got {}",
        state[s + 2]
    );
    assert!(
        (state[s + 3] - 100.0).abs() < 1e-10,
        "lag1 hydro 1: expected 100.0, got {}",
        state[s + 3]
    );
}

/// End-to-end: `StudySetup::new`'s hoisted `derive_inflow_seeds` call feeds
/// `build_initial_state`'s lag block from `inflow_history`. Stage 0 is
/// January 2020 (`season_id: Some(0)`, a full-coverage month), so each
/// hydro's k=1/k=2 previous-occurrence windows (December/November 2019)
/// resolve to their record's `value_m3s` verbatim.
#[test]
fn study_setup_initial_state_has_nonzero_lags_from_derived_inflow_history() {
    let inflow_history = vec![
        InflowHistoryRow {
            hydro_id: EntityId(1),
            start_date: chrono::NaiveDate::from_ymd_opt(2019, 12, 1).unwrap(),
            end_date: chrono::NaiveDate::from_ymd_opt(2020, 1, 1).unwrap(),
            value_m3s: 600.0,
        },
        InflowHistoryRow {
            hydro_id: EntityId(1),
            start_date: chrono::NaiveDate::from_ymd_opt(2019, 11, 1).unwrap(),
            end_date: chrono::NaiveDate::from_ymd_opt(2019, 12, 1).unwrap(),
            value_m3s: 500.0,
        },
        InflowHistoryRow {
            hydro_id: EntityId(2),
            start_date: chrono::NaiveDate::from_ymd_opt(2019, 12, 1).unwrap(),
            end_date: chrono::NaiveDate::from_ymd_opt(2020, 1, 1).unwrap(),
            value_m3s: 200.0,
        },
        InflowHistoryRow {
            hydro_id: EntityId(2),
            start_date: chrono::NaiveDate::from_ymd_opt(2019, 11, 1).unwrap(),
            end_date: chrono::NaiveDate::from_ymd_opt(2019, 12, 1).unwrap(),
            value_m3s: 100.0,
        },
    ];
    let system = minimal_system_2_hydros_with_history(
        3,
        Some(monthly_season_map_for_lag_seed_test()),
        inflow_history,
    );
    let config = minimal_config(1, 10);
    let stochastic = build_stochastic_context(
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
    .expect("stochastic context");

    let setup = StudySetup::new(
        &system,
        &config,
        stochastic,
        PrepareHydroModelsResult::default_from_system(&system),
        Vec::new(),
    )
    .expect("setup with inflow_history");

    let state = &setup.inputs.initial.state;

    // With 2 hydros (N=2) and max_par_order=2 (L=2), lag slots start at N=2.
    // Lag-major layout: slot = lag_start + lag * N + h.
    let n_hydros = 2;
    let lag_start = n_hydros;
    assert!(
        (state[lag_start] - 600.0).abs() < 1e-10,
        "lag0 hydro 0 should be 600.0 via StudySetup, got {}",
        state[lag_start]
    );
    assert!(
        (state[lag_start + 1] - 200.0).abs() < 1e-10,
        "lag0 hydro 1 should be 200.0 via StudySetup, got {}",
        state[lag_start + 1]
    );
    assert!(
        (state[lag_start + 2] - 500.0).abs() < 1e-10,
        "lag1 hydro 0 should be 500.0 via StudySetup, got {}",
        state[lag_start + 2]
    );
    assert!(
        (state[lag_start + 3] - 100.0).abs() < 1e-10,
        "lag1 hydro 1 should be 100.0 via StudySetup, got {}",
        state[lag_start + 3]
    );
}

#[test]
fn build_initial_state_no_lags_state_is_storage_only() {
    use super::build_initial_state;

    let system = minimal_system(2);
    let layout = layout_for_lag_test(1, 0);

    // n_state = N*(1+L) = 1*(1+0) = 1
    assert_eq!(layout.n_state, 1);
    assert!(
        layout.inflow_lags.is_empty(),
        "inflow_lags range should be empty for L=0"
    );

    let state = build_initial_state(&system, &test_support::study_dims(), &layout, &[]);

    assert_eq!(state.len(), 1, "state length must equal n_state=1");
}

// -----------------------------------------------------------------------
// build_initial_state — commit_in seed
// -----------------------------------------------------------------------

/// `GeometryDims` for 1 hydro, 0 lags, and the given anticipated metadata. The
/// anticipated `build_initial_state` tests derive their `StudyDimensions` from
/// these dims (`study_dims_for`), so geometry and study shape stay aligned.
fn counts_with_anticipated(
    n_anticipated: usize,
    k_values: &[usize],
    anticipated_positions: &[usize],
) -> test_support::GeometryDims {
    let lead_stages = k_values.iter().copied().max().unwrap_or(0);
    test_support::GeometryDims {
        hydro_count: 1,
        n_thermals: n_anticipated, // at least cover the anticipated plants
        n_buses: 1,
        n_blks: 1,
        n_anticipated,
        lead_stages,
        anticipated_plants: test_support::anticipated_plants_at(anticipated_positions),
        ..Default::default()
    }
}

/// The `i`-th stage's `[start_date, end_date)` window shared by
/// `system_with_anticipated_thermals` and
/// `system_with_two_anticipated_thermals_staggered_dates`: a fixed 31-day
/// block chained from `2024-01-01`. A window's real calendar span must equal
/// its stage's declared `duration_hours` (744.0) for `StageCalendar::coverage`
/// to resolve a whole-stage window at fraction 1.0 — a true calendar month
/// (28-31 days) would drift against the fixed 744-hour declaration.
fn anticipated_stage_window(i: usize) -> (chrono::NaiveDate, chrono::NaiveDate) {
    use chrono::{NaiveDate, TimeDelta};
    let start = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap() + TimeDelta::days(31 * i as i64);
    let end = start + TimeDelta::days(31);
    (start, end)
}

/// N anticipated thermals with the given `lead_stages`; IDs are `10 + i` (kept
/// clear of the bus id 1 and hydro id 3). `past_commits` must be pre-sorted by
/// `thermal_id`.
#[expect(
    clippy::too_many_lines,
    clippy::items_after_statements,
    reason = "the fixture spells out one complete study inline so each assertion traces to a literal, and fixture-local helper items sit beside the entities that use them"
)]
fn system_with_anticipated_thermals(
    k_values: &[u32],
    past_commits: Vec<cobre_core::AnticipatedCommitmentHistory>,
) -> cobre_core::System {
    use chrono::NaiveDate;

    let bus = Bus {
        id: EntityId(1),
        name: "B1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 500.0,
        }],
        excess_cost: 0.0,
    };

    let thermals: Vec<Thermal> = k_values
        .iter()
        .enumerate()
        .map(|(i, &k)| Thermal {
            id: EntityId(10 + i as i32),
            name: format!("AT{i}"),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId(1),
            min_generation_mw: 0.0,
            max_generation_mw: 500.0,
            cost_per_mwh: 50.0,
            anticipated_config: Some(AnticipatedConfig::LeadStages(k)),
            entry_stage_id: None,
            exit_stage_id: None,
        })
        .collect();

    let mut hydro = Hydro {
        unit_groups: Vec::new(),
        id: EntityId(3),
        name: "H1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        downstream_id: None,
        travel_time_hours: None,
        entry_stage_id: None,
        exit_stage_id: None,
        min_storage_hm3: 0.0,
        max_storage_hm3: 200.0,
        min_outflow_m3s: 0.0,
        max_outflow_m3s: None,
        generation_model: HydroGenerationModel::ConstantProductivity,
        min_turbined_m3s: 0.0,
        max_turbined_m3s: 100.0,
        specific_productivity_mw_per_m3s_per_m: None,
        min_generation_mw: 0.0,
        max_generation_mw: 250.0,
        tailrace: None,
        hydraulic_losses: None,
        efficiency: None,
        evaporation_coefficients_mm: None,
        evaporation_reference_volumes_hm3: None,
        diversion: None,
        filling: None,
        penalties: HydroPenalties {
            spillage_cost: 0.01,
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
    hydro.declare_mirror_unit_group(EntityId(1));

    let n_stages = 3_usize;
    let stages: Vec<Stage> = (0..n_stages)
        .map(|i| {
            let (start_date, end_date) = anticipated_stage_window(i);
            Stage {
                index: i,
                id: i as i32,
                start_date,
                end_date,
                season_id: None,
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
                    branching_factor: 1,
                    noise_method: NoiseMethod::Saa,
                },
            }
        })
        .collect();

    let inflow_models: Vec<InflowModel> = (0..n_stages)
        .map(|i| InflowModel {
            hydro_id: EntityId(3),
            stage_id: i as i32,
            mean_m3s: 80.0,
            std_m3s: 20.0,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
            annual: None,
        })
        .collect();

    let load_models: Vec<LoadModel> = (0..n_stages)
        .map(|i| LoadModel {
            bus_id: EntityId(1),
            stage_id: i as i32,
            mean_mw: 100.0,
            std_mw: 0.0,
        })
        .collect();

    let k_max_bounds = k_values.iter().copied().max().unwrap_or(0) as usize;
    let n_thermals = k_values.len();

    fn default_hydro_bounds() -> HydroStageBounds {
        HydroStageBounds {
            min_storage_hm3: 0.0,
            max_storage_hm3: 200.0,
            filling_min_rate_m3s: 0.0,
            water_withdrawal_m3s: 0.0,
        }
    }

    fn default_hydro_block_bounds() -> HydroBlockBounds {
        HydroBlockBounds {
            max_turbined_m3s: 100.0,
            max_generation_mw: 250.0,
            ..Default::default()
        }
    }

    fn default_hydro_penalties() -> HydroPenalties {
        HydroPenalties {
            spillage_cost: 0.01,
            diversion_cost: 0.0,
            turbined_cost: 0.0,
            storage_violation_below_cost: 500.0,
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

    let bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 1,
            n_thermals,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages,
            k_max: k_max_bounds,
        },
        &BoundsDefaults {
            hydro: default_hydro_bounds(),
            hydro_block: default_hydro_block_bounds(),
            thermal: ThermalStageBounds { cost_per_mwh: 0.0 },
            thermal_block: ThermalBlockBounds {
                min_generation_mw: 0.0,
                max_generation_mw: 500.0,
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

    let penalties = ResolvedPenalties::new(
        &PenaltiesCountsSpec {
            n_hydros: 1,
            n_buses: 1,
            n_lines: 0,
            n_ncs: 0,
            n_stages,
        },
        &PenaltiesDefaults {
            hydro: default_hydro_penalties(),
            bus: BusStagePenalties { excess_cost: 0.0 },
            line: LineStagePenalties { exchange_cost: 0.0 },
            ncs: NcsStagePenalties {
                curtailment_cost: 0.0,
            },
        },
    );

    SystemBuilder::new()
        .buses(vec![bus])
        .thermals(thermals)
        .hydros(vec![hydro])
        .stages(stages)
        .inflow_models(inflow_models)
        .load_models(load_models)
        .bounds(bounds)
        .penalties(penalties)
        .initial_conditions(cobre_core::InitialConditions {
            storage: vec![],
            filling_storage: vec![],
            past_anticipated_commitments: past_commits,
            recent_observations: vec![],
            past_defluences: vec![],
        })
        .build()
        .expect("system_with_anticipated_thermals: valid")
}

/// Build a 1-bus / 1-hydro / 2-thermal system where thermal id=10 (K=2, the
/// SMALLER id) has a LATER `operational_start_date` than thermal id=11 (K=3,
/// the LARGER id), so the canonical `(operational_start_date, id)` order
/// (`System::thermals()`) is `[id=11, id=10]` — id-DESCENDING, not
/// id-ascending. Mirrors [`system_with_anticipated_thermals`] but with
/// staggered dates, exercising the thermal id->position lookup under the
/// same bug trigger as the hydro-side fixtures.
#[expect(
    clippy::too_many_lines,
    clippy::items_after_statements,
    reason = "the fixture spells out one complete study inline so each assertion traces to a literal, and fixture-local helper items sit beside the entities that use them"
)]
fn system_with_two_anticipated_thermals_staggered_dates(
    past_commits: Vec<cobre_core::AnticipatedCommitmentHistory>,
) -> cobre_core::System {
    use chrono::NaiveDate;

    let bus = Bus {
        id: EntityId(1),
        name: "B1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 500.0,
        }],
        excess_cost: 0.0,
    };

    let earlier = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();
    let later = NaiveDate::from_ymd_opt(2025, 6, 1).unwrap();

    let thermals = vec![
        Thermal {
            id: EntityId(10),
            name: "AT10_later".to_string(),
            operational_start_date: later,
            bus_id: EntityId(1),
            min_generation_mw: 0.0,
            max_generation_mw: 500.0,
            cost_per_mwh: 50.0,
            anticipated_config: Some(AnticipatedConfig::LeadStages(2)),
            entry_stage_id: None,
            exit_stage_id: None,
        },
        Thermal {
            id: EntityId(11),
            name: "AT11_earlier".to_string(),
            operational_start_date: earlier,
            bus_id: EntityId(1),
            min_generation_mw: 0.0,
            max_generation_mw: 500.0,
            cost_per_mwh: 50.0,
            anticipated_config: Some(AnticipatedConfig::LeadStages(3)),
            entry_stage_id: None,
            exit_stage_id: None,
        },
    ];

    let mut hydro = Hydro {
        unit_groups: Vec::new(),
        id: EntityId(3),
        name: "H1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        downstream_id: None,
        travel_time_hours: None,
        entry_stage_id: None,
        exit_stage_id: None,
        min_storage_hm3: 0.0,
        max_storage_hm3: 200.0,
        min_outflow_m3s: 0.0,
        max_outflow_m3s: None,
        generation_model: HydroGenerationModel::ConstantProductivity,
        min_turbined_m3s: 0.0,
        max_turbined_m3s: 100.0,
        specific_productivity_mw_per_m3s_per_m: None,
        min_generation_mw: 0.0,
        max_generation_mw: 250.0,
        tailrace: None,
        hydraulic_losses: None,
        efficiency: None,
        evaporation_coefficients_mm: None,
        evaporation_reference_volumes_hm3: None,
        diversion: None,
        filling: None,
        penalties: HydroPenalties {
            spillage_cost: 0.01,
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
    hydro.declare_mirror_unit_group(EntityId(1));

    let n_stages = 3_usize;
    let stages: Vec<Stage> = (0..n_stages)
        .map(|i| {
            let (start_date, end_date) = anticipated_stage_window(i);
            Stage {
                index: i,
                id: i as i32,
                start_date,
                end_date,
                season_id: None,
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
                    branching_factor: 1,
                    noise_method: NoiseMethod::Saa,
                },
            }
        })
        .collect();

    let inflow_models: Vec<InflowModel> = (0..n_stages)
        .map(|i| InflowModel {
            hydro_id: EntityId(3),
            stage_id: i as i32,
            mean_m3s: 80.0,
            std_m3s: 20.0,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
            annual: None,
        })
        .collect();

    let load_models: Vec<LoadModel> = (0..n_stages)
        .map(|i| LoadModel {
            bus_id: EntityId(1),
            stage_id: i as i32,
            mean_mw: 100.0,
            std_mw: 0.0,
        })
        .collect();

    fn default_hydro_bounds() -> HydroStageBounds {
        HydroStageBounds {
            min_storage_hm3: 0.0,
            max_storage_hm3: 200.0,
            filling_min_rate_m3s: 0.0,
            water_withdrawal_m3s: 0.0,
        }
    }

    fn default_hydro_block_bounds() -> HydroBlockBounds {
        HydroBlockBounds {
            max_turbined_m3s: 100.0,
            max_generation_mw: 250.0,
            ..Default::default()
        }
    }

    fn default_hydro_penalties() -> HydroPenalties {
        HydroPenalties {
            spillage_cost: 0.01,
            diversion_cost: 0.0,
            turbined_cost: 0.0,
            storage_violation_below_cost: 500.0,
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

    let bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 1,
            n_thermals: 2,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages,
            k_max: 3,
        },
        &BoundsDefaults {
            hydro: default_hydro_bounds(),
            hydro_block: default_hydro_block_bounds(),
            thermal: ThermalStageBounds { cost_per_mwh: 0.0 },
            thermal_block: ThermalBlockBounds {
                min_generation_mw: 0.0,
                max_generation_mw: 500.0,
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

    let penalties = ResolvedPenalties::new(
        &PenaltiesCountsSpec {
            n_hydros: 1,
            n_buses: 1,
            n_lines: 0,
            n_ncs: 0,
            n_stages,
        },
        &PenaltiesDefaults {
            hydro: default_hydro_penalties(),
            bus: BusStagePenalties { excess_cost: 0.0 },
            line: LineStagePenalties { exchange_cost: 0.0 },
            ncs: NcsStagePenalties {
                curtailment_cost: 0.0,
            },
        },
    );

    let system = SystemBuilder::new()
        .buses(vec![bus])
        .thermals(thermals)
        .hydros(vec![hydro])
        .stages(stages)
        .inflow_models(inflow_models)
        .load_models(load_models)
        .bounds(bounds)
        .penalties(penalties)
        .initial_conditions(cobre_core::InitialConditions {
            storage: vec![],
            filling_storage: vec![],
            past_anticipated_commitments: past_commits,
            recent_observations: vec![],
            past_defluences: vec![],
        })
        .build()
        .expect("system_with_two_anticipated_thermals_staggered_dates: valid");

    // Canonical order sorts by (operational_start_date, id): the earlier-date
    // thermal (id=11) must land at position 0, the later-date thermal (id=10)
    // at position 1 — id-descending. A test relying on this fixture to
    // trigger the staggered-order bug depends on this invariant holding.
    assert_eq!(system.thermals()[0].id, EntityId(11));
    assert_eq!(system.thermals()[1].id, EntityId(10));

    system
}

/// Regression: under a STAGGERED-commissioning system where the canonical
/// `(operational_start_date, id)` thermal order is id-DESCENDING (thermal
/// id=10's later commissioning date sorts it after thermal id=11),
/// `build_initial_state` must seed each anticipated thermal's OWN declared
/// `past_anticipated_commitments` — `binary_search_by_key` over `thermals()`
/// requires id-ascending order and silently drops the out-of-order thermal's
/// entire commitment history.
#[test]
fn build_initial_state_anticipated_seed_correct_under_staggered_commissioning_dates() {
    use super::build_initial_state;
    use cobre_core::AnticipatedCommitmentHistory;

    let (s0_start, s0_end) = anticipated_stage_window(0);
    let (s1_start, s1_end) = anticipated_stage_window(1);
    let (s2_start, s2_end) = anticipated_stage_window(2);
    let past_commits = vec![
        AnticipatedCommitmentHistory {
            thermal_id: EntityId(10),
            start_date: s0_start,
            end_date: s0_end,
            value_mw: 10.0,
        },
        AnticipatedCommitmentHistory {
            thermal_id: EntityId(10),
            start_date: s1_start,
            end_date: s1_end,
            value_mw: 20.0,
        },
        AnticipatedCommitmentHistory {
            thermal_id: EntityId(11),
            start_date: s0_start,
            end_date: s0_end,
            value_mw: 100.0,
        },
        AnticipatedCommitmentHistory {
            thermal_id: EntityId(11),
            start_date: s1_start,
            end_date: s1_end,
            value_mw: 200.0,
        },
        AnticipatedCommitmentHistory {
            thermal_id: EntityId(11),
            start_date: s2_start,
            end_date: s2_end,
            value_mw: 300.0,
        },
    ];
    let system = system_with_two_anticipated_thermals_staggered_dates(past_commits);

    // Canonical (global) order is [id=11 (K=3), id=10 (K=2)]: k_values and
    // thermal_indices follow that order, not declaration order.
    let layout = layout_with_anticipated(&[3, 2]);

    let state = build_initial_state(
        &system,
        &test_support::study_dims_for(&counts_with_anticipated(2, &[3, 2], &[0, 1])),
        &layout,
        &[],
    );

    let s = layout.commit_out.start;
    // plant 0 = thermal id=11 (canonical position 0, K=3, own values [100,200,300])
    // plant 1 = thermal id=10 (canonical position 1, K=2, own values [10,20])
    assert!(
        (state[s] - 100.0).abs() < 1e-10,
        "thermal id=11 slot 0 should be its own IC value 100.0, got {}",
        state[s]
    );
    assert!(
        (state[s + 1] - 10.0).abs() < 1e-10,
        "thermal id=10 slot 0 should be its own IC value 10.0, got {}",
        state[s + 1]
    );
    assert!(
        (state[s + 2] - 200.0).abs() < 1e-10,
        "thermal id=11 slot 1 should be its own IC value 200.0, got {}",
        state[s + 2]
    );
    assert!(
        (state[s + 3] - 20.0).abs() < 1e-10,
        "thermal id=10 slot 1 should be its own IC value 20.0, got {}",
        state[s + 3]
    );
    assert!(
        (state[s + 4] - 300.0).abs() < 1e-10,
        "thermal id=11 slot 2 should be its own IC value 300.0, got {}",
        state[s + 4]
    );
    assert!(
        state[s + 5].abs() < 1e-10,
        "thermal id=10 slot 2 (K=2 padding) should be 0.0, got {}",
        state[s + 5]
    );

    // Numeric neutrality: resolving through StageCalendar::coverage lands
    // bit-identical (`==`, not an epsilon compare) to a positional per-stage
    // splice for the same values.
    assert_eq!(
        state[s..s + 6],
        [100.0, 10.0, 200.0, 20.0, 300.0, 0.0],
        "anticipated seed must be bit-identical to a positional splice"
    );
}

#[test]
fn build_initial_state_no_anticipated_state_unchanged() {
    use super::build_initial_state;

    let system = minimal_system(2);
    let layout = layout_for_lag_test(1, 0);

    assert_eq!(layout.n_anticipated, 0);
    assert!(layout.commit_out.is_empty());

    let state = build_initial_state(&system, &test_support::study_dims(), &layout, &[]);

    assert_eq!(
        state.len(),
        layout.n_state,
        "state length must equal n_state"
    );
    // All slots are 0.0 — storage IC is empty in minimal_system.
    assert!(
        state.iter().all(|&v| v == 0.0),
        "all state slots must be 0.0 when no anticipated thermals and no ICs set"
    );
}

/// Slot-major layout (`n_ant=1`), per-stage windowed values `[50.0, 75.0]`:
///   slot 0 (`ant_start`)   → 50.0
///   slot 1 (`ant_start+1`) → 75.0
#[test]
fn build_initial_state_single_anticipated_thermal_k2() {
    use super::build_initial_state;
    use cobre_core::AnticipatedCommitmentHistory;

    // Thermal ID 10 is the first (and only) anticipated plant.
    // The system thermals() sorts by ID, so global_idx == 0 for ID 10.
    let (s0_start, s0_end) = anticipated_stage_window(0);
    let (s1_start, s1_end) = anticipated_stage_window(1);
    let past_commits = vec![
        AnticipatedCommitmentHistory {
            thermal_id: EntityId(10),
            start_date: s0_start,
            end_date: s0_end,
            value_mw: 50.0,
        },
        AnticipatedCommitmentHistory {
            thermal_id: EntityId(10),
            start_date: s1_start,
            end_date: s1_end,
            value_mw: 75.0,
        },
    ];
    let system = system_with_anticipated_thermals(&[2], past_commits);

    let layout = layout_with_anticipated(&[2]);

    let state = build_initial_state(
        &system,
        &test_support::study_dims_for(&counts_with_anticipated(1, &[2], &[0])),
        &layout,
        &[],
    );

    assert_eq!(
        state.len(),
        layout.n_state,
        "state length must equal n_state"
    );
    let ant_start = layout.commit_out.start;
    assert!(
        (state[ant_start] - 50.0).abs() < 1e-10,
        "slot 0 expected 50.0, got {}",
        state[ant_start]
    );
    assert!(
        (state[ant_start + 1] - 75.0).abs() < 1e-10,
        "slot 1 expected 75.0, got {}",
        state[ant_start + 1]
    );
}

/// Slot-major layout (`n_ant=2`, `k_max=3`):
/// - (slot 0, plant 0) `ant_start+0*2+0` → 10.0
/// - (slot 0, plant 1) `ant_start+0*2+1` → 100.0
/// - (slot 1, plant 0) `ant_start+1*2+0` → 20.0
/// - (slot 1, plant 1) `ant_start+1*2+1` → 200.0
/// - (slot 2, plant 0) `ant_start+2*2+0` → 0.0  (padding: `K_0=2` < `k_max=3`)
/// - (slot 2, plant 1) `ant_start+2*2+1` → 300.0
#[test]
fn build_initial_state_two_anticipated_thermals_mixed_k() {
    use super::build_initial_state;
    use cobre_core::AnticipatedCommitmentHistory;

    // Thermal IDs 10 (K=2) and 11 (K=3); sorted ascending so global order
    // in system.thermals() is idx 0 → ID 10, idx 1 → ID 11.
    let (s0_start, s0_end) = anticipated_stage_window(0);
    let (s1_start, s1_end) = anticipated_stage_window(1);
    let (s2_start, s2_end) = anticipated_stage_window(2);
    let past_commits = vec![
        AnticipatedCommitmentHistory {
            thermal_id: EntityId(10),
            start_date: s0_start,
            end_date: s0_end,
            value_mw: 10.0,
        },
        AnticipatedCommitmentHistory {
            thermal_id: EntityId(10),
            start_date: s1_start,
            end_date: s1_end,
            value_mw: 20.0,
        },
        AnticipatedCommitmentHistory {
            thermal_id: EntityId(11),
            start_date: s0_start,
            end_date: s0_end,
            value_mw: 100.0,
        },
        AnticipatedCommitmentHistory {
            thermal_id: EntityId(11),
            start_date: s1_start,
            end_date: s1_end,
            value_mw: 200.0,
        },
        AnticipatedCommitmentHistory {
            thermal_id: EntityId(11),
            start_date: s2_start,
            end_date: s2_end,
            value_mw: 300.0,
        },
    ];
    let system = system_with_anticipated_thermals(&[2, 3], past_commits);

    let layout = layout_with_anticipated(&[2, 3]);

    let state = build_initial_state(
        &system,
        &test_support::study_dims_for(&counts_with_anticipated(2, &[2, 3], &[0, 1])),
        &layout,
        &[],
    );

    assert_eq!(
        state.len(),
        layout.n_state,
        "state length must equal n_state"
    );
    // offset from ant_start = slot * n_ant + plant.
    let s = layout.commit_out.start;

    assert!(
        (state[s] - 10.0).abs() < 1e-10,
        "slot 0 plant 0: expected 10.0, got {}",
        state[s]
    );
    assert!(
        (state[s + 1] - 100.0).abs() < 1e-10,
        "slot 0 plant 1: expected 100.0, got {}",
        state[s + 1]
    );
    assert!(
        (state[s + 2] - 20.0).abs() < 1e-10,
        "slot 1 plant 0: expected 20.0, got {}",
        state[s + 2]
    );
    assert!(
        (state[s + 3] - 200.0).abs() < 1e-10,
        "slot 1 plant 1: expected 200.0, got {}",
        state[s + 3]
    );
    assert!(
        state[s + 4].abs() < 1e-10,
        "slot 2 plant 0 (K_0=2 padding): expected 0.0, got {}",
        state[s + 4]
    );
    assert!(
        (state[s + 5] - 300.0).abs() < 1e-10,
        "slot 2 plant 1: expected 300.0, got {}",
        state[s + 5]
    );
}

#[test]
fn build_initial_state_empty_past_commitments_leaves_zeros() {
    use super::build_initial_state;

    let system = system_with_anticipated_thermals(&[2], vec![]);

    let layout = layout_with_anticipated(&[2]);

    let state = build_initial_state(
        &system,
        &test_support::study_dims_for(&counts_with_anticipated(1, &[2], &[0])),
        &layout,
        &[],
    );

    assert_eq!(
        state.len(),
        layout.n_state,
        "state length must equal n_state"
    );
    let ant_start = layout.commit_out.start;
    let ant_end = layout.commit_out.end;
    for (i, &v) in state[ant_start..ant_end].iter().enumerate() {
        assert!(v.abs() < 1e-10, "commit_in slot {i} expected 0.0, got {v}");
    }
}

#[test]
fn build_initial_state_unknown_thermal_id_silently_skipped() {
    use super::build_initial_state;
    use cobre_core::AnticipatedCommitmentHistory;

    let (s0_start, s0_end) = anticipated_stage_window(0);
    let past_commits = vec![AnticipatedCommitmentHistory {
        thermal_id: EntityId(99999),
        start_date: s0_start,
        end_date: s0_end,
        value_mw: 42.0,
    }];
    let system = system_with_anticipated_thermals(&[2], past_commits);

    let layout = layout_with_anticipated(&[2]);

    let state = build_initial_state(
        &system,
        &test_support::study_dims_for(&counts_with_anticipated(1, &[2], &[0])),
        &layout,
        &[],
    );

    assert_eq!(
        state.len(),
        layout.n_state,
        "state length must equal n_state"
    );
    let ant_start = layout.commit_out.start;
    let ant_end = layout.commit_out.end;
    for (i, &v) in state[ant_start..ant_end].iter().enumerate() {
        assert!(
            v.abs() < 1e-10,
            "commit_in slot {i} expected 0.0 for unknown ID, got {v}"
        );
    }
}

/// `past_anticipated_commitments` carries one window `100.0` for plant 0
/// (`K_0=1`, tiling its single leading stage) and two windows `[50.0, 75.0]`
/// for plant 1 (`K_1=2`, one per leading stage) — each plant's windows tile
/// its leading `K_i` stages exactly (the contract cobre-io's validator
/// enforces in production).
///
/// Expected layout (`n_ant = 2`, slot-major):
///   - `ant_start + 0*2 + 0` (slot 0, plant 0) -> 100.0  (seed)
///   - `ant_start + 0*2 + 1` (slot 0, plant 1) ->  50.0  (seed)
///   - `ant_start + 1*2 + 0` (slot 1, plant 0) ->   0.0  (padding; `K_0=1` < `k_max=2`)
///   - `ant_start + 1*2 + 1` (slot 1, plant 1) ->  75.0  (seed)
///
/// The padding-slot `debug_assert!` must not fire — the `.take(k_i)` clamp on
/// the resolved coverage iterator prevents writing past slot `K_0=1` on plant 0.
#[test]
fn build_initial_state_anticipated_seed_padding_slot_stays_zero() {
    use super::build_initial_state;
    use cobre_core::AnticipatedCommitmentHistory;

    let (s0_start, s0_end) = anticipated_stage_window(0);
    let (s1_start, s1_end) = anticipated_stage_window(1);
    let past_commits = vec![
        AnticipatedCommitmentHistory {
            thermal_id: EntityId(10),
            start_date: s0_start,
            end_date: s0_end,
            value_mw: 100.0,
        },
        AnticipatedCommitmentHistory {
            thermal_id: EntityId(11),
            start_date: s0_start,
            end_date: s0_end,
            value_mw: 50.0,
        },
        AnticipatedCommitmentHistory {
            thermal_id: EntityId(11),
            start_date: s1_start,
            end_date: s1_end,
            value_mw: 75.0,
        },
    ];
    let system = system_with_anticipated_thermals(&[1, 2], past_commits);
    let layout = layout_with_anticipated(&[1, 2]);

    let state = build_initial_state(
        &system,
        &test_support::study_dims_for(&counts_with_anticipated(2, &[1, 2], &[0, 1])),
        &layout,
        &[],
    );

    assert_eq!(
        state.len(),
        layout.n_state,
        "state length must equal n_state"
    );
    let s = layout.commit_out.start;
    let n_ant = layout.n_anticipated;
    assert_eq!(n_ant, 2);
    assert_eq!(layout.k_max, 2);

    assert!(
        (state[s] - 100.0).abs() < 1e-10,
        "slot 0 plant 0 expected 100.0, got {}",
        state[s]
    );
    assert!(
        (state[s + 1] - 50.0).abs() < 1e-10,
        "slot 0 plant 1 expected 50.0, got {}",
        state[s + 1]
    );
    // Padding slot: the invariant the debug_assert! protects.
    assert!(
        state[s + 2].abs() < 1e-10,
        "padding slot 1 plant 0 expected 0.0, got {}",
        state[s + 2]
    );
    assert!(
        (state[s + 3] - 75.0).abs() < 1e-10,
        "slot 1 plant 1 expected 75.0, got {}",
        state[s + 3]
    );
}

/// The ring-depth under-sizing reproduction's seeds (the sizing contract lives
/// in `.claude/rules/sddp.md`, "Ring depth sizing"): four
/// `AnticipatedCommitmentHistory` windows tiling
/// [`minimal_system_with_anticipated_and_commitments`]'s `[168.0, 168.0, 168.0,
/// 648.0]`-hour calendar (`2024-01-01` cursor) at distinct values
/// `100/200/300/400`.
fn bug_doc_reproduction_past_commits() -> Vec<cobre_core::AnticipatedCommitmentHistory> {
    use cobre_core::AnticipatedCommitmentHistory;

    vec![
        AnticipatedCommitmentHistory {
            thermal_id: EntityId(2),
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, 1, 8).unwrap(),
            value_mw: 100.0,
        },
        AnticipatedCommitmentHistory {
            thermal_id: EntityId(2),
            start_date: NaiveDate::from_ymd_opt(2024, 1, 8).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, 1, 15).unwrap(),
            value_mw: 200.0,
        },
        AnticipatedCommitmentHistory {
            thermal_id: EntityId(2),
            start_date: NaiveDate::from_ymd_opt(2024, 1, 15).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, 1, 22).unwrap(),
            value_mw: 300.0,
        },
        AnticipatedCommitmentHistory {
            thermal_id: EntityId(2),
            start_date: NaiveDate::from_ymd_opt(2024, 1, 22).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 18).unwrap(),
            value_mw: 400.0,
        },
    ]
}

/// Bug doc's reproduction shape: a `LeadTime(1160.0)` thermal whose lead
/// resolves all four study deliveries pre-study, plus one post-study month —
/// [`resolve_anticipated_commitments_widens_lead_time_plant_lead_to_the_ring_depth`]
/// pins `resolution.anchored_depth() == 4` and `lead_stages == [4]` for this exact shape.
fn bug_doc_reproduction_system() -> cobre_core::System {
    let post_study = PostStudyStages {
        stages: vec![PostStudyStage {
            start_date: NaiveDate::from_ymd_opt(2024, 2, 18).unwrap(),
            duration_hours: 648.0,
        }],
        thermal_bounds: Vec::new(),
    };
    minimal_system_with_anticipated_and_commitments(
        &[168.0, 168.0, 168.0, 648.0],
        AnticipatedConfig::LeadTime(1160.0),
        4,
        Some(post_study),
        bug_doc_reproduction_past_commits(),
    )
}

/// Reproduces the ring-depth under-sizing defect (the sizing contract in
/// `.claude/rules/sddp.md`, "Ring depth sizing"): when `K_i` is
/// undersized to the occupancy max `3`, the `.take(k_i)` clamp drops the stage-3
/// window and modular aliasing fishes stage 0's `100.0` for stage 3. With `K_i`
/// sized to the ring depth `4`, all four leading stages seed at four distinct
/// offsets, stage 3 holding its own `400.0`.
#[test]
fn initial_state_seeds_every_leading_commitment_under_the_widened_ring_depth() {
    use super::build_initial_state;

    let system = bug_doc_reproduction_system();
    let layout = layout_with_anticipated(&[4]);

    let state = build_initial_state(
        &system,
        &test_support::study_dims_for(&counts_with_anticipated(1, &[4], &[0])),
        &layout,
        &[],
    );

    let s = layout.commit_out.start;
    assert_eq!(
        state[s..s + 4],
        [100.0, 200.0, 300.0, 400.0],
        "all four leading stages must land at four distinct offsets"
    );
    assert!(
        (state[s + 3] - 400.0).abs() < 1e-10,
        "stage-3 slot must hold its own seed 400.0, not stage 0's aliased 100.0, got {}",
        state[s + 3]
    );
}

/// A short-lead plant's padding range `[K_i, k_max)` stays zero even when a
/// sibling plant's own lead widens the shared ring past it: plant 0 tiles
/// every ring slot with its own seed while plant 1 covers only its own
/// leading stage, leaving plant 1's remaining slots as genuine, non-empty
/// padding for the encoder-driven loop to protect.
#[test]
fn initial_state_leaves_padding_slots_zero() {
    use super::build_initial_state;
    use cobre_core::AnticipatedCommitmentHistory;

    let (s0_start, s0_end) = anticipated_stage_window(0);
    let (s1_start, s1_end) = anticipated_stage_window(1);
    let (s2_start, s2_end) = anticipated_stage_window(2);
    let past_commits = vec![
        AnticipatedCommitmentHistory {
            thermal_id: EntityId(10),
            start_date: s0_start,
            end_date: s0_end,
            value_mw: 111.0,
        },
        AnticipatedCommitmentHistory {
            thermal_id: EntityId(10),
            start_date: s1_start,
            end_date: s1_end,
            value_mw: 222.0,
        },
        AnticipatedCommitmentHistory {
            thermal_id: EntityId(10),
            start_date: s2_start,
            end_date: s2_end,
            value_mw: 333.0,
        },
        AnticipatedCommitmentHistory {
            thermal_id: EntityId(11),
            start_date: s0_start,
            end_date: s0_end,
            value_mw: 444.0,
        },
    ];
    let system = system_with_anticipated_thermals(&[3, 1], past_commits);
    let layout = layout_with_anticipated(&[3, 1]);
    let k_i_short = layout.anticipated_lead_stages[1];
    assert!(
        k_i_short < layout.k_max,
        "fixture must leave a non-empty padding range to exercise"
    );

    let state = build_initial_state(
        &system,
        &test_support::study_dims_for(&counts_with_anticipated(2, &[3, 1], &[0, 1])),
        &layout,
        &[],
    );

    let s = layout.commit_out.start;
    let plant0_seeds = [111.0, 222.0, 333.0];
    for (slot, &expected) in plant0_seeds.iter().enumerate() {
        let off = s + layout.commitment_hold_in_study_offset(0, slot);
        assert!(
            (state[off] - expected).abs() < 1e-10,
            "plant 0 slot {slot} expected {expected}, got {}",
            state[off]
        );
    }
    let off_plant1_seed = s + layout.commitment_hold_in_study_offset(1, 0);
    assert!(
        (state[off_plant1_seed] - 444.0).abs() < 1e-10,
        "plant 1 slot 0 expected 444.0, got {}",
        state[off_plant1_seed]
    );

    for slot in k_i_short..layout.k_max {
        let off = s + layout.commitment_hold_in_study_offset(1, slot);
        assert_eq!(state[off], 0.0, "padding slot {slot} must be 0.0");
    }
}

/// The seed walk's covered-stage cross-check: a layout whose `anticipated_lead_stages`
/// is deliberately narrower than a covered stage in `past_anticipated_commitments`
/// simulates a resolver/validator desync — never reachable through cobre-io's own
/// coverage rule — and the `debug_assert!` guarding the silent-drop replacement
/// fires in a debug build.
#[test]
#[should_panic(expected = "covered stage beyond plant's own lead")]
fn initial_state_rejects_a_covered_stage_beyond_the_plants_own_lead() {
    use super::build_initial_state;
    use cobre_core::AnticipatedCommitmentHistory;

    let (s0_start, s0_end) = anticipated_stage_window(0);
    let (s1_start, s1_end) = anticipated_stage_window(1);
    let past_commits = vec![
        AnticipatedCommitmentHistory {
            thermal_id: EntityId(10),
            start_date: s0_start,
            end_date: s0_end,
            value_mw: 100.0,
        },
        AnticipatedCommitmentHistory {
            thermal_id: EntityId(10),
            start_date: s1_start,
            end_date: s1_end,
            value_mw: 200.0,
        },
    ];
    let system = system_with_anticipated_thermals(&[2], past_commits);
    // Deliberately narrower than a resolved K_i would ever be for this
    // fully-covered history: simulates the desync the cross-check guards.
    let layout = layout_with_anticipated(&[1]);

    let _ = build_initial_state(
        &system,
        &test_support::study_dims_for(&counts_with_anticipated(1, &[1], &[0])),
        &layout,
        &[],
    );
}

#[test]
fn historical_library_none_for_insample() {
    let system = minimal_system(2);
    let config = minimal_config(1, 5);
    let stochastic = build_stochastic_context(
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
    .expect("stochastic context");

    let setup = StudySetup::new(
        &system,
        &config,
        stochastic,
        PrepareHydroModelsResult::default_from_system(&system),
        Vec::new(),
    )
    .expect("setup");

    assert!(
        setup
            .inputs
            .scenario_libraries
            .training
            .historical
            .is_none(),
        "historical_library must be None for InSample scheme"
    );
    assert!(
        setup
            .inputs
            .scenario_libraries
            .training
            .external_inflow
            .is_none(),
        "external_inflow_library must be None for InSample scheme"
    );
    assert!(
        setup
            .inputs
            .scenario_libraries
            .training
            .external_load
            .is_none(),
        "external_load_library must be None for InSample load scheme"
    );
    assert!(
        setup
            .inputs
            .scenario_libraries
            .training
            .external_ncs
            .is_none(),
        "external_ncs_library must be None for InSample ncs scheme"
    );
}

/// `Historical`-scheme fixture with the inflow history needed to discover at
/// least one window: 2 monthly stages (seasons 0-1) and data covering
/// 1990-1991. With `max_par_order = 0`, a window is valid when both study
/// months are observed — year 1990 covers months 0-1, so it qualifies.
#[expect(
    clippy::too_many_lines,
    reason = "the fixture spells out one complete study inline so each assertion traces to a literal"
)]
fn system_with_historical_inflow(n_stages: usize) -> cobre_core::System {
    use chrono::NaiveDate;

    fn default_hydro_bounds() -> HydroStageBounds {
        HydroStageBounds {
            min_storage_hm3: 0.0,
            max_storage_hm3: 200.0,
            filling_min_rate_m3s: 0.0,
            water_withdrawal_m3s: 0.0,
        }
    }

    fn default_hydro_block_bounds() -> HydroBlockBounds {
        HydroBlockBounds {
            max_turbined_m3s: 100.0,
            max_generation_mw: 250.0,
            ..Default::default()
        }
    }

    fn default_hydro_penalties() -> HydroPenalties {
        HydroPenalties {
            spillage_cost: 0.01,
            diversion_cost: 0.0,
            turbined_cost: 0.0,
            storage_violation_below_cost: 500.0,
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

    let bus = Bus {
        id: EntityId(1),
        name: "B1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 500.0,
        }],
        excess_cost: 0.0,
    };

    let thermal = Thermal {
        id: EntityId(2),
        name: "T1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        bus_id: EntityId(1),
        min_generation_mw: 0.0,
        max_generation_mw: 100.0,
        cost_per_mwh: 50.0,
        anticipated_config: None,
        entry_stage_id: None,
        exit_stage_id: None,
    };

    let mut hydro = Hydro {
        unit_groups: Vec::new(),
        id: EntityId(3),
        name: "H1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        downstream_id: None,
        travel_time_hours: None,
        entry_stage_id: None,
        exit_stage_id: None,
        min_storage_hm3: 0.0,
        max_storage_hm3: 200.0,
        min_outflow_m3s: 0.0,
        max_outflow_m3s: None,
        generation_model: HydroGenerationModel::ConstantProductivity,
        min_turbined_m3s: 0.0,
        max_turbined_m3s: 100.0,
        specific_productivity_mw_per_m3s_per_m: None,
        min_generation_mw: 0.0,
        max_generation_mw: 250.0,
        tailrace: None,
        hydraulic_losses: None,
        efficiency: None,
        evaporation_coefficients_mm: None,
        evaporation_reference_volumes_hm3: None,
        diversion: None,
        filling: None,
        penalties: HydroPenalties {
            spillage_cost: 0.01,
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
    hydro.declare_mirror_unit_group(EntityId(1));

    let stages: Vec<Stage> = (0..n_stages)
        .map(|i| Stage {
            index: i,
            id: i as i32,
            start_date: NaiveDate::from_ymd_opt(2024, (i as u32 % 12) + 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, (i as u32 % 12) + 1, 28).unwrap(),
            season_id: Some(i % 12),
            blocks: vec![Block {
                index: 0,
                name: "S".to_string(),
                duration_hours: 720.0,
            }],
            block_mode: BlockMode::Parallel,
            state_config: StageStateConfig {
                storage: true,
                inflow_lags: false,
            },
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor: 1,
                noise_method: NoiseMethod::Saa,
            },
        })
        .collect();

    let inflow_models: Vec<InflowModel> = (0..n_stages)
        .map(|i| InflowModel {
            hydro_id: EntityId(3),
            stage_id: i as i32,
            mean_m3s: 80.0,
            std_m3s: 20.0,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
            annual: None,
        })
        .collect();

    let load_models: Vec<LoadModel> = (0..n_stages)
        .map(|i| LoadModel {
            bus_id: EntityId(1),
            stage_id: i as i32,
            mean_mw: 100.0,
            std_mw: 0.0,
        })
        .collect();

    let inflow_history: Vec<InflowHistoryRow> = (1990_i32..=1991)
        .flat_map(|year| {
            (1u32..=12).map(move |month| {
                let start_date = NaiveDate::from_ymd_opt(year, month, 1).unwrap();
                InflowHistoryRow {
                    hydro_id: EntityId(3),
                    start_date,
                    end_date: start_date.succ_opt().unwrap(),
                    value_m3s: 80.0 + f64::from(year - 1990) * 5.0,
                }
            })
        })
        .collect();

    let n_st = n_stages.max(1);

    let bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 1,
            n_thermals: 1,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages: n_st,
            k_max: 0,
        },
        &BoundsDefaults {
            hydro: default_hydro_bounds(),
            hydro_block: default_hydro_block_bounds(),
            thermal: ThermalStageBounds { cost_per_mwh: 0.0 },
            thermal_block: ThermalBlockBounds {
                min_generation_mw: 0.0,
                max_generation_mw: 100.0,
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

    let penalties = ResolvedPenalties::new(
        &PenaltiesCountsSpec {
            n_hydros: 1,
            n_buses: 1,
            n_lines: 0,
            n_ncs: 0,
            n_stages: n_st,
        },
        &PenaltiesDefaults {
            hydro: default_hydro_penalties(),
            bus: BusStagePenalties { excess_cost: 0.0 },
            line: LineStagePenalties { exchange_cost: 0.0 },
            ncs: NcsStagePenalties {
                curtailment_cost: 0.0,
            },
        },
    );

    SystemBuilder::new()
        .buses(vec![bus])
        .thermals(vec![thermal])
        .hydros(vec![hydro])
        .stages(stages)
        .inflow_models(inflow_models)
        .load_models(load_models)
        .inflow_history(inflow_history)
        .bounds(bounds)
        .penalties(penalties)
        .build()
        .expect("system_with_historical_inflow: valid")
}

#[test]
fn historical_library_built_when_scheme_is_historical() {
    let system = system_with_historical_inflow(2);
    let config = minimal_config_with_schemes(1, 5, Some(RawSamplingScheme::Historical), None, None);
    let stochastic = build_stochastic_context(
        &system,
        42,
        None,
        &[],
        &[],
        OpeningTreeInputs::default(),
        ClassSchemes {
            inflow: Some(SamplingScheme::Historical),
            load: Some(SamplingScheme::InSample),
            ncs: Some(SamplingScheme::InSample),
        },
    )
    .expect("stochastic context");

    let setup = StudySetup::new(
        &system,
        &config,
        stochastic,
        PrepareHydroModelsResult::default_from_system(&system),
        Vec::new(),
    )
    .expect("setup");

    let lib = setup
        .inputs
        .scenario_libraries
        .training
        .historical
        .as_ref()
        .expect("expected Some(HistoricalScenarioLibrary) for Historical scheme");
    assert!(
        lib.n_windows() > 0,
        "expected at least one historical window, got 0"
    );
    assert_eq!(
        lib.n_stages(),
        2,
        "expected n_stages == 2 matching the system's study stages"
    );
    assert_eq!(lib.n_hydros(), 1, "expected n_hydros == 1");
}

#[test]
#[expect(
    clippy::too_many_lines,
    clippy::cast_lossless,
    reason = "the fixture spells out one complete study inline so each assertion traces to a literal, and fixture values widen with as to match the literal tables beside them"
)]
fn external_inflow_library_built_when_scheme_is_external() {
    use chrono::NaiveDate;
    use cobre_core::scenario::ExternalScenarioRow;
    use cobre_core::scenario::InflowModel as CoreInflowModel;

    let hydro_id = EntityId(3);
    let mut external_rows: Vec<ExternalScenarioRow> = Vec::new();
    for stage_id in 0i32..2 {
        for scenario_id in 0i32..3 {
            external_rows.push(ExternalScenarioRow {
                stage_id,
                scenario_id,
                hydro_id,
                value_m3s: 80.0 + scenario_id as f64 * 5.0,
            });
        }
    }

    // minimal_system has no seam to inject external rows, so rebuild the system
    // directly to carry them.
    let bus = Bus {
        id: EntityId(1),
        name: "B1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 500.0,
        }],
        excess_cost: 0.0,
    };
    let thermal = Thermal {
        id: EntityId(2),
        name: "T1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        bus_id: EntityId(1),
        min_generation_mw: 0.0,
        max_generation_mw: 100.0,
        cost_per_mwh: 50.0,
        anticipated_config: None,
        entry_stage_id: None,
        exit_stage_id: None,
    };
    let mut hydro = Hydro {
        unit_groups: Vec::new(),
        id: EntityId(3),
        name: "H1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        downstream_id: None,
        travel_time_hours: None,
        entry_stage_id: None,
        exit_stage_id: None,
        min_storage_hm3: 0.0,
        max_storage_hm3: 200.0,
        min_outflow_m3s: 0.0,
        max_outflow_m3s: None,
        generation_model: HydroGenerationModel::ConstantProductivity,
        min_turbined_m3s: 0.0,
        max_turbined_m3s: 100.0,
        specific_productivity_mw_per_m3s_per_m: None,
        min_generation_mw: 0.0,
        max_generation_mw: 250.0,
        tailrace: None,
        hydraulic_losses: None,
        efficiency: None,
        evaporation_coefficients_mm: None,
        evaporation_reference_volumes_hm3: None,
        diversion: None,
        filling: None,
        penalties: HydroPenalties {
            spillage_cost: 0.01,
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
    hydro.declare_mirror_unit_group(EntityId(1));
    let stages: Vec<Stage> = (0..2usize)
        .map(|i| Stage {
            index: i,
            id: i as i32,
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            season_id: None,
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
                branching_factor: 1,
                noise_method: NoiseMethod::Saa,
            },
        })
        .collect();

    let inflow_models: Vec<CoreInflowModel> = (0..2usize)
        .map(|i| CoreInflowModel {
            hydro_id: EntityId(3),
            stage_id: i as i32,
            mean_m3s: 80.0,
            std_m3s: 20.0,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
            annual: None,
        })
        .collect();

    let load_models: Vec<LoadModel> = (0..2usize)
        .map(|i| LoadModel {
            bus_id: EntityId(1),
            stage_id: i as i32,
            mean_mw: 100.0,
            std_mw: 0.0,
        })
        .collect();

    let bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 1,
            n_thermals: 1,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages: 2,
            k_max: 0,
        },
        &BoundsDefaults {
            hydro: HydroStageBounds {
                min_storage_hm3: 0.0,
                max_storage_hm3: 200.0,
                filling_min_rate_m3s: 0.0,
                water_withdrawal_m3s: 0.0,
            },
            hydro_block: HydroBlockBounds {
                max_turbined_m3s: 100.0,
                max_generation_mw: 250.0,
                ..Default::default()
            },
            thermal: ThermalStageBounds { cost_per_mwh: 0.0 },
            thermal_block: ThermalBlockBounds {
                min_generation_mw: 0.0,
                max_generation_mw: 100.0,
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
    let penalties = ResolvedPenalties::new(
        &PenaltiesCountsSpec {
            n_hydros: 1,
            n_buses: 1,
            n_lines: 0,
            n_ncs: 0,
            n_stages: 2,
        },
        &PenaltiesDefaults {
            hydro: HydroPenalties {
                spillage_cost: 0.01,
                diversion_cost: 0.0,
                turbined_cost: 0.0,
                storage_violation_below_cost: 500.0,
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
            bus: BusStagePenalties { excess_cost: 0.0 },
            line: LineStagePenalties { exchange_cost: 0.0 },
            ncs: NcsStagePenalties {
                curtailment_cost: 0.0,
            },
        },
    );

    let system = SystemBuilder::new()
        .buses(vec![bus])
        .thermals(vec![thermal])
        .hydros(vec![hydro])
        .stages(stages)
        .inflow_models(inflow_models)
        .load_models(load_models)
        .external_scenarios(external_rows)
        .bounds(bounds)
        .penalties(penalties)
        .build()
        .expect("system with external inflow: valid");

    let config = minimal_config_with_schemes(1, 5, Some(RawSamplingScheme::External), None, None);
    let stochastic = build_stochastic_context(
        &system,
        42,
        None,
        &[],
        &[],
        OpeningTreeInputs::default(),
        ClassSchemes {
            inflow: Some(SamplingScheme::External),
            load: Some(SamplingScheme::InSample),
            ncs: Some(SamplingScheme::InSample),
        },
    )
    .expect("stochastic context");

    let setup = StudySetup::new(
        &system,
        &config,
        stochastic,
        PrepareHydroModelsResult::default_from_system(&system),
        Vec::new(),
    )
    .expect("setup");

    let lib = setup
        .inputs
        .scenario_libraries
        .training
        .external_inflow
        .as_ref()
        .expect("expected Some(ExternalScenarioLibrary) for External inflow scheme");
    assert!(
        lib.n_entities() > 0,
        "expected n_entities > 0 in external inflow library"
    );
    assert_eq!(lib.n_stages(), 2);
    assert_eq!(lib.n_scenarios(), 3);
    assert_eq!(lib.entity_class(), "inflow");
}

#[test]
#[expect(
    clippy::too_many_lines,
    clippy::cast_lossless,
    reason = "the fixture spells out one complete study inline so each assertion traces to a literal, and fixture values widen with as to match the literal tables beside them"
)]
fn external_load_library_built_when_scheme_is_external() {
    use chrono::NaiveDate;
    use cobre_core::scenario::ExternalLoadRow;
    use cobre_core::scenario::InflowModel as CoreInflowModel;

    let bus = Bus {
        id: EntityId(1),
        name: "B1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 500.0,
        }],
        excess_cost: 0.0,
    };
    let thermal = Thermal {
        id: EntityId(2),
        name: "T1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        bus_id: EntityId(1),
        min_generation_mw: 0.0,
        max_generation_mw: 100.0,
        cost_per_mwh: 50.0,
        anticipated_config: None,
        entry_stage_id: None,
        exit_stage_id: None,
    };
    let mut hydro = Hydro {
        unit_groups: Vec::new(),
        id: EntityId(3),
        name: "H1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        downstream_id: None,
        travel_time_hours: None,
        entry_stage_id: None,
        exit_stage_id: None,
        min_storage_hm3: 0.0,
        max_storage_hm3: 200.0,
        min_outflow_m3s: 0.0,
        max_outflow_m3s: None,
        generation_model: HydroGenerationModel::ConstantProductivity,
        min_turbined_m3s: 0.0,
        max_turbined_m3s: 100.0,
        specific_productivity_mw_per_m3s_per_m: None,
        min_generation_mw: 0.0,
        max_generation_mw: 250.0,
        tailrace: None,
        hydraulic_losses: None,
        efficiency: None,
        evaporation_coefficients_mm: None,
        evaporation_reference_volumes_hm3: None,
        diversion: None,
        filling: None,
        penalties: HydroPenalties {
            spillage_cost: 0.01,
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
    hydro.declare_mirror_unit_group(EntityId(1));

    let stages: Vec<Stage> = (0..2usize)
        .map(|i| Stage {
            index: i,
            id: i as i32,
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            season_id: None,
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
                branching_factor: 1,
                noise_method: NoiseMethod::Saa,
            },
        })
        .collect();

    let inflow_models: Vec<CoreInflowModel> = (0..2usize)
        .map(|i| CoreInflowModel {
            hydro_id: EntityId(3),
            stage_id: i as i32,
            mean_m3s: 80.0,
            std_m3s: 20.0,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
            annual: None,
        })
        .collect();

    let load_models: Vec<LoadModel> = (0..2usize)
        .map(|i| LoadModel {
            bus_id: EntityId(1),
            stage_id: i as i32,
            mean_mw: 100.0,
            std_mw: 10.0,
        })
        .collect();

    let mut external_load_rows: Vec<ExternalLoadRow> = Vec::new();
    for stage_id in 0i32..2 {
        for scenario_id in 0i32..3 {
            external_load_rows.push(ExternalLoadRow {
                stage_id,
                scenario_id,
                bus_id: EntityId(1),
                value_mw: 90.0 + scenario_id as f64 * 10.0,
            });
        }
    }

    let bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 1,
            n_thermals: 1,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages: 2,
            k_max: 0,
        },
        &BoundsDefaults {
            hydro: HydroStageBounds {
                min_storage_hm3: 0.0,
                max_storage_hm3: 200.0,
                filling_min_rate_m3s: 0.0,
                water_withdrawal_m3s: 0.0,
            },
            hydro_block: HydroBlockBounds {
                max_turbined_m3s: 100.0,
                max_generation_mw: 250.0,
                ..Default::default()
            },
            thermal: ThermalStageBounds { cost_per_mwh: 0.0 },
            thermal_block: ThermalBlockBounds {
                min_generation_mw: 0.0,
                max_generation_mw: 100.0,
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
    let penalties = ResolvedPenalties::new(
        &PenaltiesCountsSpec {
            n_hydros: 1,
            n_buses: 1,
            n_lines: 0,
            n_ncs: 0,
            n_stages: 2,
        },
        &PenaltiesDefaults {
            hydro: HydroPenalties {
                spillage_cost: 0.01,
                diversion_cost: 0.0,
                turbined_cost: 0.0,
                storage_violation_below_cost: 500.0,
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
            bus: BusStagePenalties { excess_cost: 0.0 },
            line: LineStagePenalties { exchange_cost: 0.0 },
            ncs: NcsStagePenalties {
                curtailment_cost: 0.0,
            },
        },
    );

    let system = SystemBuilder::new()
        .buses(vec![bus])
        .thermals(vec![thermal])
        .hydros(vec![hydro])
        .stages(stages)
        .inflow_models(inflow_models)
        .load_models(load_models)
        .external_load_scenarios(external_load_rows)
        .bounds(bounds)
        .penalties(penalties)
        .build()
        .expect("system with external load: valid");

    let config = minimal_config_with_schemes(1, 5, None, Some(RawSamplingScheme::External), None);
    let stochastic = build_stochastic_context(
        &system,
        42,
        None,
        &[],
        &[],
        OpeningTreeInputs::default(),
        ClassSchemes {
            inflow: Some(SamplingScheme::InSample),
            load: Some(SamplingScheme::External),
            ncs: Some(SamplingScheme::InSample),
        },
    )
    .expect("stochastic context");

    let setup = StudySetup::new(
        &system,
        &config,
        stochastic,
        PrepareHydroModelsResult::default_from_system(&system),
        Vec::new(),
    )
    .expect("setup");

    let lib = setup
        .inputs
        .scenario_libraries
        .training
        .external_load
        .as_ref()
        .expect("expected Some(ExternalScenarioLibrary) for External load scheme");
    assert!(
        lib.n_entities() > 0,
        "expected n_entities > 0 in external load library"
    );
    assert_eq!(lib.n_stages(), 2);
    assert_eq!(lib.n_scenarios(), 3);
    assert_eq!(lib.entity_class(), "load");
}

/// A two-stage, one-hydro/one-thermal system with two load buses — `B1`
/// (`std_mw = 10.0`) and `B2` (`std_mw = 0.0`, deterministic) — and
/// three-scenario external load rows for both, shared by
/// [`external_load_library_includes_zero_sigma_bus_when_scheme_is_external`]
/// and [`sim_external_load_with_sigma_zero_bus_rejects_width_mismatch`].
#[expect(
    clippy::too_many_lines,
    clippy::cast_lossless,
    reason = "the fixture spells out one complete study inline so each assertion traces to a literal, and fixture values widen with as to match the literal tables beside them"
)]
fn zero_sigma_external_load_system() -> System {
    use chrono::NaiveDate;
    use cobre_core::scenario::ExternalLoadRow;
    use cobre_core::scenario::InflowModel as CoreInflowModel;

    let bus = Bus {
        id: EntityId(1),
        name: "B1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 500.0,
        }],
        excess_cost: 0.0,
    };
    let deterministic_bus = Bus {
        id: EntityId(4),
        name: "B2".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 500.0,
        }],
        excess_cost: 0.0,
    };
    let thermal = Thermal {
        id: EntityId(2),
        name: "T1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        bus_id: EntityId(1),
        min_generation_mw: 0.0,
        max_generation_mw: 100.0,
        cost_per_mwh: 50.0,
        anticipated_config: None,
        entry_stage_id: None,
        exit_stage_id: None,
    };
    let mut hydro = Hydro {
        unit_groups: Vec::new(),
        id: EntityId(3),
        name: "H1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        downstream_id: None,
        travel_time_hours: None,
        entry_stage_id: None,
        exit_stage_id: None,
        min_storage_hm3: 0.0,
        max_storage_hm3: 200.0,
        min_outflow_m3s: 0.0,
        max_outflow_m3s: None,
        generation_model: HydroGenerationModel::ConstantProductivity,
        min_turbined_m3s: 0.0,
        max_turbined_m3s: 100.0,
        specific_productivity_mw_per_m3s_per_m: None,
        min_generation_mw: 0.0,
        max_generation_mw: 250.0,
        tailrace: None,
        hydraulic_losses: None,
        efficiency: None,
        evaporation_coefficients_mm: None,
        evaporation_reference_volumes_hm3: None,
        diversion: None,
        filling: None,
        penalties: HydroPenalties {
            spillage_cost: 0.01,
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
    hydro.declare_mirror_unit_group(EntityId(1));

    let stages: Vec<Stage> = (0..2usize)
        .map(|i| Stage {
            index: i,
            id: i as i32,
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            season_id: None,
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
                branching_factor: 1,
                noise_method: NoiseMethod::Saa,
            },
        })
        .collect();

    let inflow_models: Vec<CoreInflowModel> = (0..2usize)
        .map(|i| CoreInflowModel {
            hydro_id: EntityId(3),
            stage_id: i as i32,
            mean_m3s: 80.0,
            std_m3s: 20.0,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
            annual: None,
        })
        .collect();

    let mut load_models: Vec<LoadModel> = Vec::new();
    for i in 0i32..2 {
        load_models.push(LoadModel {
            bus_id: EntityId(1),
            stage_id: i,
            mean_mw: 100.0,
            std_mw: 10.0,
        });
    }
    for i in 0i32..2 {
        load_models.push(LoadModel {
            bus_id: EntityId(4),
            stage_id: i,
            mean_mw: 50.0,
            std_mw: 0.0,
        });
    }

    let mut external_load_rows: Vec<ExternalLoadRow> = Vec::new();
    for stage_id in 0i32..2 {
        for scenario_id in 0i32..3 {
            external_load_rows.push(ExternalLoadRow {
                stage_id,
                scenario_id,
                bus_id: EntityId(1),
                value_mw: 90.0 + scenario_id as f64 * 10.0,
            });
            external_load_rows.push(ExternalLoadRow {
                stage_id,
                scenario_id,
                bus_id: EntityId(4),
                value_mw: 50.0,
            });
        }
    }

    let bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 1,
            n_thermals: 1,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages: 2,
            k_max: 0,
        },
        &BoundsDefaults {
            hydro: HydroStageBounds {
                min_storage_hm3: 0.0,
                max_storage_hm3: 200.0,
                filling_min_rate_m3s: 0.0,
                water_withdrawal_m3s: 0.0,
            },
            hydro_block: HydroBlockBounds {
                max_turbined_m3s: 100.0,
                max_generation_mw: 250.0,
                ..Default::default()
            },
            thermal: ThermalStageBounds { cost_per_mwh: 0.0 },
            thermal_block: ThermalBlockBounds {
                min_generation_mw: 0.0,
                max_generation_mw: 100.0,
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
    let penalties = ResolvedPenalties::new(
        &PenaltiesCountsSpec {
            n_hydros: 1,
            n_buses: 2,
            n_lines: 0,
            n_ncs: 0,
            n_stages: 2,
        },
        &PenaltiesDefaults {
            hydro: HydroPenalties {
                spillage_cost: 0.01,
                diversion_cost: 0.0,
                turbined_cost: 0.0,
                storage_violation_below_cost: 500.0,
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
            bus: BusStagePenalties { excess_cost: 0.0 },
            line: LineStagePenalties { exchange_cost: 0.0 },
            ncs: NcsStagePenalties {
                curtailment_cost: 0.0,
            },
        },
    );

    SystemBuilder::new()
        .buses(vec![bus, deterministic_bus])
        .thermals(vec![thermal])
        .hydros(vec![hydro])
        .stages(stages)
        .inflow_models(inflow_models)
        .load_models(load_models)
        .external_load_scenarios(external_load_rows)
        .bounds(bounds)
        .penalties(penalties)
        .build()
        .expect("system with external load incl. a zero-sigma bus: valid")
}

/// A `std_mw = 0.0` (deterministic) load bus under the `External` scheme keeps
/// a noise-vector slot (`noise_entity_order`'s `std_mw > 0.0 || scheme ==
/// External` membership rule) — `build_external_load_library` must include it
/// too, or setup rejects the study (`V3.5`/width mismatch) the moment its
/// external file carries a row for that bus. End-to-end proof: two buses, one
/// with `std_mw > 0.0` and one with `std_mw == 0.0`, both present in the
/// external load rows; setup must succeed and the library must carry both.
#[test]
fn external_load_library_includes_zero_sigma_bus_when_scheme_is_external() {
    use cobre_comm::LocalBackend;
    use cobre_solver::ActiveSolver;

    let system = zero_sigma_external_load_system();
    let config = minimal_config_with_schemes(1, 5, None, Some(RawSamplingScheme::External), None);
    let stochastic = build_stochastic_context(
        &system,
        42,
        None,
        &[],
        &[],
        OpeningTreeInputs::default(),
        ClassSchemes {
            inflow: Some(SamplingScheme::InSample),
            load: Some(SamplingScheme::External),
            ncs: Some(SamplingScheme::InSample),
        },
    )
    .expect("stochastic context");

    let mut setup = StudySetup::new(
        &system,
        &config,
        stochastic,
        PrepareHydroModelsResult::default_from_system(&system),
        Vec::new(),
    )
    .expect("setup must accept a sigma=0 External-scheme load bus");

    let lib = setup
        .inputs
        .scenario_libraries
        .training
        .external_load
        .as_ref()
        .expect("expected Some(ExternalScenarioLibrary) for External load scheme");
    assert_eq!(
        lib.n_entities(),
        2,
        "the zero-sigma bus must occupy a noise-vector slot alongside the nonzero-sigma bus"
    );
    assert_eq!(lib.n_stages(), 2);
    assert_eq!(lib.n_scenarios(), 3);
    assert_eq!(lib.entity_class(), "load");

    // Train-through smoke: the same setup, bounded by minimal_config_with_schemes's
    // 5-iteration limit. The thermal (100 MW @ 50/MWh) and deficit segment
    // (500/MWh) keep every stage trivially feasible.
    let comm = LocalBackend;
    let mut solver = ActiveSolver::new().expect("solver");
    setup
        .train(&mut solver, &comm, 1, ActiveSolver::new, None, None)
        .expect("train: a sigma=0 External-scheme load bus must not block training");
}

/// A σ=0 load bus is a training-scheme noise member only under `External`
/// (`LoadModel::is_noise_member`): training here stays `InSample`, so its
/// `normal_load_bus_ids` excludes the σ=0 bus while a divergent simulation
/// `External` scheme includes it via `external_load_scenarios`.
/// `assert_external_library_widths` checks the SIMULATION library's width
/// against the TRAINING-scheme-gated block, so the resulting narrower-vs-wider
/// mismatch must reject setup rather than pass silently.
#[test]
fn sim_external_load_with_sigma_zero_bus_rejects_width_mismatch() {
    let system = zero_sigma_external_load_system();
    let mut config = minimal_config_with_schemes(1, 5, None, None, None);
    config.simulation.scenario_source = Some(RawScenarioSourceConfig {
        seed: Some(42),
        historical_years: None,
        inflow: None,
        load: Some(RawClassConfigEntry {
            scheme: RawSamplingScheme::External,
        }),
        ncs: None,
        openings: None,
    });

    let stochastic = build_stochastic_context(
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
    .expect("stochastic context");

    let result = StudySetup::new(
        &system,
        &config,
        stochastic,
        PrepareHydroModelsResult::default_from_system(&system),
        Vec::new(),
    );

    match result {
        Err(SddpError::Validation(msg)) => {
            assert!(
                msg.contains("load") && msg.contains("width mismatch"),
                "expected a load width-mismatch Validation error, got: {msg}"
            );
        }
        other => panic!("expected a load width-mismatch Validation error, got {other:?}"),
    }
}

#[test]
#[expect(
    clippy::too_many_lines,
    clippy::cast_lossless,
    reason = "the fixture spells out one complete study inline so each assertion traces to a literal, and fixture values widen with as to match the literal tables beside them"
)]
fn external_ncs_library_built_when_scheme_is_external() {
    use chrono::NaiveDate;
    use cobre_core::scenario::InflowModel as CoreInflowModel;
    use cobre_core::{
        NonControllableSource,
        scenario::{ExternalNcsRow, NcsModel},
    };

    let bus = Bus {
        id: EntityId(1),
        name: "B1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 500.0,
        }],
        excess_cost: 0.0,
    };
    let thermal = Thermal {
        id: EntityId(2),
        name: "T1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        bus_id: EntityId(1),
        min_generation_mw: 0.0,
        max_generation_mw: 100.0,
        cost_per_mwh: 50.0,
        anticipated_config: None,
        entry_stage_id: None,
        exit_stage_id: None,
    };
    let mut hydro = Hydro {
        unit_groups: Vec::new(),
        id: EntityId(3),
        name: "H1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        downstream_id: None,
        travel_time_hours: None,
        entry_stage_id: None,
        exit_stage_id: None,
        min_storage_hm3: 0.0,
        max_storage_hm3: 200.0,
        min_outflow_m3s: 0.0,
        max_outflow_m3s: None,
        generation_model: HydroGenerationModel::ConstantProductivity,
        min_turbined_m3s: 0.0,
        max_turbined_m3s: 100.0,
        specific_productivity_mw_per_m3s_per_m: None,
        min_generation_mw: 0.0,
        max_generation_mw: 250.0,
        tailrace: None,
        hydraulic_losses: None,
        efficiency: None,
        evaporation_coefficients_mm: None,
        evaporation_reference_volumes_hm3: None,
        diversion: None,
        filling: None,
        penalties: HydroPenalties {
            spillage_cost: 0.01,
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
    hydro.declare_mirror_unit_group(EntityId(1));

    let ncs_id = EntityId(4);
    let ncs_source = NonControllableSource {
        id: ncs_id,
        name: "Wind1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        bus_id: EntityId(1),
        entry_stage_id: None,
        exit_stage_id: None,
        max_generation_mw: 100.0,
        allow_curtailment: true,
        curtailment_cost: 0.01,
    };

    let stages: Vec<Stage> = (0..2usize)
        .map(|i| Stage {
            index: i,
            id: i as i32,
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            season_id: None,
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
                branching_factor: 1,
                noise_method: NoiseMethod::Saa,
            },
        })
        .collect();

    let inflow_models: Vec<CoreInflowModel> = (0..2usize)
        .map(|i| CoreInflowModel {
            hydro_id: EntityId(3),
            stage_id: i as i32,
            mean_m3s: 80.0,
            std_m3s: 20.0,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
            annual: None,
        })
        .collect();

    let load_models: Vec<LoadModel> = (0..2usize)
        .map(|i| LoadModel {
            bus_id: EntityId(1),
            stage_id: i as i32,
            mean_mw: 100.0,
            std_mw: 0.0,
        })
        .collect();

    let ncs_models: Vec<NcsModel> = (0..2usize)
        .map(|i| NcsModel {
            ncs_id,
            stage_id: i as i32,
            mean: 0.8,
            std: 0.1,
        })
        .collect();

    let mut external_ncs_rows: Vec<ExternalNcsRow> = Vec::new();
    for stage_id in 0i32..2 {
        for scenario_id in 0i32..3 {
            external_ncs_rows.push(ExternalNcsRow {
                stage_id,
                scenario_id,
                ncs_id,
                value: 0.7 + scenario_id as f64 * 0.1,
            });
        }
    }

    let bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 1,
            n_thermals: 1,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages: 2,
            k_max: 0,
        },
        &BoundsDefaults {
            hydro: HydroStageBounds {
                min_storage_hm3: 0.0,
                max_storage_hm3: 200.0,
                filling_min_rate_m3s: 0.0,
                water_withdrawal_m3s: 0.0,
            },
            hydro_block: HydroBlockBounds {
                max_turbined_m3s: 100.0,
                max_generation_mw: 250.0,
                ..Default::default()
            },
            thermal: ThermalStageBounds { cost_per_mwh: 0.0 },
            thermal_block: ThermalBlockBounds {
                min_generation_mw: 0.0,
                max_generation_mw: 100.0,
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
    let penalties = ResolvedPenalties::new(
        &PenaltiesCountsSpec {
            n_hydros: 1,
            n_buses: 1,
            n_lines: 0,
            n_ncs: 1,
            n_stages: 2,
        },
        &PenaltiesDefaults {
            hydro: HydroPenalties {
                spillage_cost: 0.01,
                diversion_cost: 0.0,
                turbined_cost: 0.0,
                storage_violation_below_cost: 500.0,
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
            bus: BusStagePenalties { excess_cost: 0.0 },
            line: LineStagePenalties { exchange_cost: 0.0 },
            ncs: NcsStagePenalties {
                curtailment_cost: 0.0,
            },
        },
    );

    let system = SystemBuilder::new()
        .buses(vec![bus])
        .thermals(vec![thermal])
        .hydros(vec![hydro])
        .non_controllable_sources(vec![ncs_source])
        .stages(stages)
        .inflow_models(inflow_models)
        .load_models(load_models)
        .ncs_models(ncs_models)
        .external_ncs_scenarios(external_ncs_rows)
        .bounds(bounds)
        .penalties(penalties)
        .build()
        .expect("system with external NCS: valid");

    let config = minimal_config_with_schemes(1, 5, None, None, Some(RawSamplingScheme::External));
    let stochastic = build_stochastic_context(
        &system,
        42,
        None,
        &[],
        &[],
        OpeningTreeInputs::default(),
        ClassSchemes {
            inflow: Some(SamplingScheme::InSample),
            load: Some(SamplingScheme::InSample),
            ncs: Some(SamplingScheme::External),
        },
    )
    .expect("stochastic context");

    let setup = StudySetup::new(
        &system,
        &config,
        stochastic,
        PrepareHydroModelsResult::default_from_system(&system),
        Vec::new(),
    )
    .expect("setup");

    let lib = setup
        .inputs
        .scenario_libraries
        .training
        .external_ncs
        .as_ref()
        .expect("expected Some(ExternalScenarioLibrary) for External NCS scheme");
    assert!(
        lib.n_entities() > 0,
        "expected n_entities > 0 in external NCS library"
    );
    assert_eq!(lib.n_stages(), 2);
    assert_eq!(lib.n_scenarios(), 3);
    assert_eq!(lib.entity_class(), "ncs");
}

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "the fixture spells out one complete study inline so each assertion traces to a literal"
)]
fn historical_library_fails_when_no_valid_windows() {
    use chrono::NaiveDate;

    // Historical scheme with empty inflow_history guarantees zero candidate
    // years in discovery.
    let bus = Bus {
        id: EntityId(1),
        name: "B1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 500.0,
        }],
        excess_cost: 0.0,
    };
    let thermal = Thermal {
        id: EntityId(2),
        name: "T1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        bus_id: EntityId(1),
        min_generation_mw: 0.0,
        max_generation_mw: 100.0,
        cost_per_mwh: 50.0,
        anticipated_config: None,
        entry_stage_id: None,
        exit_stage_id: None,
    };
    let mut hydro = Hydro {
        unit_groups: Vec::new(),
        id: EntityId(3),
        name: "H1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        downstream_id: None,
        travel_time_hours: None,
        entry_stage_id: None,
        exit_stage_id: None,
        min_storage_hm3: 0.0,
        max_storage_hm3: 200.0,
        min_outflow_m3s: 0.0,
        max_outflow_m3s: None,
        generation_model: HydroGenerationModel::ConstantProductivity,
        min_turbined_m3s: 0.0,
        max_turbined_m3s: 100.0,
        specific_productivity_mw_per_m3s_per_m: None,
        min_generation_mw: 0.0,
        max_generation_mw: 250.0,
        tailrace: None,
        hydraulic_losses: None,
        efficiency: None,
        evaporation_coefficients_mm: None,
        evaporation_reference_volumes_hm3: None,
        diversion: None,
        filling: None,
        penalties: HydroPenalties {
            spillage_cost: 0.01,
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
    hydro.declare_mirror_unit_group(EntityId(1));

    let stages: Vec<Stage> = (0..2usize)
        .map(|i| Stage {
            index: i,
            id: i as i32,
            start_date: NaiveDate::from_ymd_opt(2024, (i as u32 % 12) + 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, (i as u32 % 12) + 1, 28).unwrap(),
            season_id: Some(i % 12),
            blocks: vec![Block {
                index: 0,
                name: "S".to_string(),
                duration_hours: 720.0,
            }],
            block_mode: BlockMode::Parallel,
            state_config: StageStateConfig {
                storage: true,
                inflow_lags: false,
            },
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor: 1,
                noise_method: NoiseMethod::Saa,
            },
        })
        .collect();

    let inflow_models: Vec<InflowModel> = (0..2usize)
        .map(|i| InflowModel {
            hydro_id: EntityId(3),
            stage_id: i as i32,
            mean_m3s: 80.0,
            std_m3s: 20.0,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
            annual: None,
        })
        .collect();

    let load_models: Vec<LoadModel> = (0..2usize)
        .map(|i| LoadModel {
            bus_id: EntityId(1),
            stage_id: i as i32,
            mean_mw: 100.0,
            std_mw: 0.0,
        })
        .collect();

    let bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 1,
            n_thermals: 1,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages: 2,
            k_max: 0,
        },
        &BoundsDefaults {
            hydro: HydroStageBounds {
                min_storage_hm3: 0.0,
                max_storage_hm3: 200.0,
                filling_min_rate_m3s: 0.0,
                water_withdrawal_m3s: 0.0,
            },
            hydro_block: HydroBlockBounds {
                max_turbined_m3s: 100.0,
                max_generation_mw: 250.0,
                ..Default::default()
            },
            thermal: ThermalStageBounds { cost_per_mwh: 0.0 },
            thermal_block: ThermalBlockBounds {
                min_generation_mw: 0.0,
                max_generation_mw: 100.0,
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
    let penalties = ResolvedPenalties::new(
        &PenaltiesCountsSpec {
            n_hydros: 1,
            n_buses: 1,
            n_lines: 0,
            n_ncs: 0,
            n_stages: 2,
        },
        &PenaltiesDefaults {
            hydro: HydroPenalties {
                spillage_cost: 0.01,
                diversion_cost: 0.0,
                turbined_cost: 0.0,
                storage_violation_below_cost: 500.0,
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
            bus: BusStagePenalties { excess_cost: 0.0 },
            line: LineStagePenalties { exchange_cost: 0.0 },
            ncs: NcsStagePenalties {
                curtailment_cost: 0.0,
            },
        },
    );

    let system = SystemBuilder::new()
        .buses(vec![bus])
        .thermals(vec![thermal])
        .hydros(vec![hydro])
        .stages(stages)
        .inflow_models(inflow_models)
        .load_models(load_models)
        .bounds(bounds)
        .penalties(penalties)
        .build()
        .expect("system: valid");

    let config = minimal_config_with_schemes(1, 5, Some(RawSamplingScheme::Historical), None, None);
    let stochastic = build_stochastic_context(
        &system,
        42,
        None,
        &[],
        &[],
        OpeningTreeInputs::default(),
        ClassSchemes {
            inflow: Some(SamplingScheme::Historical),
            load: Some(SamplingScheme::InSample),
            ncs: Some(SamplingScheme::InSample),
        },
    )
    .expect("stochastic context");

    let result = StudySetup::new(
        &system,
        &config,
        stochastic,
        PrepareHydroModelsResult::default_from_system(&system),
        Vec::new(),
    );

    assert!(result.is_err(), "expected Err when no historical data");
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("window") || err_msg.contains("historical"),
        "error should mention windows or historical, got: {err_msg}"
    );
}

#[test]
fn test_simulate_uses_simulation_scheme() {
    let system = minimal_system(2);

    let mut config = minimal_config(1, 5);
    config.simulation.scenario_source = Some(RawScenarioSourceConfig {
        seed: Some(99),
        historical_years: None,
        inflow: Some(RawClassConfigEntry {
            scheme: RawSamplingScheme::OutOfSample,
        }),
        load: None,
        ncs: None,
        openings: None,
    });

    let stochastic = build_stochastic_context(
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
    .expect("stochastic context");

    let setup = StudySetup::new(
        &system,
        &config,
        stochastic,
        PrepareHydroModelsResult::default_from_system(&system),
        Vec::new(),
    )
    .expect("setup");

    let train_ctx = setup.training_ctx();
    assert_eq!(
        train_ctx.inflow_scheme,
        SamplingScheme::InSample,
        "training context must use InSample inflow scheme"
    );

    let sim_ctx = setup.simulation_ctx();
    assert_eq!(
        sim_ctx.inflow_scheme,
        SamplingScheme::OutOfSample,
        "simulation context must use OutOfSample inflow scheme"
    );
}

#[test]
fn test_sim_historical_library_built_when_sim_scheme_is_historical() {
    let system = system_with_historical_inflow(2);

    let mut config = minimal_config(1, 5);
    config.simulation.scenario_source = Some(RawScenarioSourceConfig {
        seed: Some(42),
        historical_years: None,
        inflow: Some(RawClassConfigEntry {
            scheme: RawSamplingScheme::Historical,
        }),
        load: None,
        ncs: None,
        openings: None,
    });

    // The stochastic context is built for the training scheme (InSample).
    let stochastic = build_stochastic_context(
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
    .expect("stochastic context");

    let setup = StudySetup::new(
        &system,
        &config,
        stochastic,
        PrepareHydroModelsResult::default_from_system(&system),
        Vec::new(),
    )
    .expect("setup");

    assert!(
        setup.training_ctx().historical_library.is_none(),
        "training context must NOT have a historical library when scheme is InSample"
    );
    assert!(
        setup.simulation_ctx().historical_library.is_some(),
        "simulation context must have a historical library when sim scheme is Historical"
    );
}

/// [`minimal_system_with_anticipated_and_commitments`] with no
/// `past_anticipated_commitments` — the pre-existing thermal-only system every
/// caller but the seed-walk fixtures wants.
fn minimal_system_with_anticipated(
    stage_hours: &[f64],
    anticipated_config: AnticipatedConfig,
    k_max_bounds: usize,
    post_study_stages: Option<PostStudyStages>,
) -> cobre_core::System {
    minimal_system_with_anticipated_and_commitments(
        stage_hours,
        anticipated_config,
        k_max_bounds,
        post_study_stages,
        Vec::new(),
    )
}

/// Like [`minimal_system`] but the thermal carries `anticipated_config` and each
/// stage's block runs for the matching `stage_hours` entry. `k_max_bounds`
/// widens the thermal stage-bounds axis for delivery-stage padding.
/// `post_study_stages` threads straight onto the built system for post-horizon
/// commitment-window fixtures. `past_commits` seeds
/// `initial_conditions.past_anticipated_commitments` directly.
#[expect(
    clippy::too_many_lines,
    clippy::items_after_statements,
    reason = "the fixture spells out one complete study inline so each assertion traces to a literal, and fixture-local helper items sit beside the entities that use them"
)]
fn minimal_system_with_anticipated_and_commitments(
    stage_hours: &[f64],
    anticipated_config: AnticipatedConfig,
    k_max_bounds: usize,
    post_study_stages: Option<PostStudyStages>,
    past_commits: Vec<cobre_core::AnticipatedCommitmentHistory>,
) -> cobre_core::System {
    use chrono::NaiveDate;

    let n_stages = stage_hours.len();

    let bus = Bus {
        id: EntityId(1),
        name: "B1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 500.0,
        }],
        excess_cost: 0.0,
    };

    let thermal = Thermal {
        id: EntityId(2),
        name: "T1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        bus_id: EntityId(1),
        min_generation_mw: 0.0,
        max_generation_mw: 500.0,
        cost_per_mwh: 50.0,
        anticipated_config: Some(anticipated_config),
        entry_stage_id: None,
        exit_stage_id: None,
    };

    let mut hydro = Hydro {
        unit_groups: Vec::new(),
        id: EntityId(3),
        name: "H1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        downstream_id: None,
        travel_time_hours: None,
        entry_stage_id: None,
        exit_stage_id: None,
        min_storage_hm3: 0.0,
        max_storage_hm3: 200.0,
        min_outflow_m3s: 0.0,
        max_outflow_m3s: None,
        generation_model: HydroGenerationModel::ConstantProductivity,
        min_turbined_m3s: 0.0,
        max_turbined_m3s: 100.0,
        specific_productivity_mw_per_m3s_per_m: None,
        min_generation_mw: 0.0,
        max_generation_mw: 250.0,
        tailrace: None,
        hydraulic_losses: None,
        efficiency: None,
        evaporation_coefficients_mm: None,
        evaporation_reference_volumes_hm3: None,
        diversion: None,
        filling: None,
        penalties: HydroPenalties {
            spillage_cost: 0.01,
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
    hydro.declare_mirror_unit_group(EntityId(1));

    // Each stage's real calendar span (whole days, `stage_hours[i] / 24`) must
    // match its own declared `duration_hours` for `StageCalendar::coverage` to
    // resolve a whole-stage window at fraction 1.0 — a fixed date range shared
    // by every stage (the pre-StageCalendar fixture) leaves the calendar
    // overlapping and violates `StageCalendar::new`'s ordering precondition.
    let mut stage_cursor = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();
    let stages: Vec<Stage> = (0..n_stages)
        .map(|i| {
            let width_days = (stage_hours[i] / 24.0).round() as i64;
            let start_date = stage_cursor;
            let end_date = start_date + chrono::TimeDelta::days(width_days);
            stage_cursor = end_date;
            Stage {
                index: i,
                id: i as i32,
                start_date,
                end_date,
                season_id: None,
                blocks: vec![Block {
                    index: 0,
                    name: "S".to_string(),
                    duration_hours: stage_hours[i],
                }],
                block_mode: BlockMode::Parallel,
                state_config: StageStateConfig {
                    storage: true,
                    inflow_lags: false,
                },
                risk_config: StageRiskConfig::Expectation,
                scenario_config: ScenarioSourceConfig {
                    branching_factor: 1,
                    noise_method: NoiseMethod::Saa,
                },
            }
        })
        .collect();

    let inflow_models: Vec<InflowModel> = (0..n_stages)
        .map(|i| InflowModel {
            hydro_id: EntityId(3),
            stage_id: i as i32,
            mean_m3s: 80.0,
            std_m3s: 20.0,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
            annual: None,
        })
        .collect();

    let load_models: Vec<LoadModel> = (0..n_stages)
        .map(|i| LoadModel {
            bus_id: EntityId(1),
            stage_id: i as i32,
            mean_mw: 100.0,
            std_mw: 0.0,
        })
        .collect();

    let n_st = n_stages.max(1);

    fn default_hydro_bounds() -> HydroStageBounds {
        HydroStageBounds {
            min_storage_hm3: 0.0,
            max_storage_hm3: 200.0,
            filling_min_rate_m3s: 0.0,
            water_withdrawal_m3s: 0.0,
        }
    }

    fn default_hydro_block_bounds() -> HydroBlockBounds {
        HydroBlockBounds {
            max_turbined_m3s: 100.0,
            max_generation_mw: 250.0,
            ..Default::default()
        }
    }

    fn default_hydro_penalties() -> HydroPenalties {
        HydroPenalties {
            spillage_cost: 0.01,
            diversion_cost: 0.0,
            turbined_cost: 0.0,
            storage_violation_below_cost: 500.0,
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

    let bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 1,
            n_thermals: 1,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages: n_st,
            k_max: k_max_bounds,
        },
        &BoundsDefaults {
            hydro: default_hydro_bounds(),
            hydro_block: default_hydro_block_bounds(),
            thermal: ThermalStageBounds { cost_per_mwh: 0.0 },
            thermal_block: ThermalBlockBounds {
                min_generation_mw: 0.0,
                max_generation_mw: 500.0,
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

    let penalties = ResolvedPenalties::new(
        &PenaltiesCountsSpec {
            n_hydros: 1,
            n_buses: 1,
            n_lines: 0,
            n_ncs: 0,
            n_stages: n_st,
        },
        &PenaltiesDefaults {
            hydro: default_hydro_penalties(),
            bus: BusStagePenalties { excess_cost: 0.0 },
            line: LineStagePenalties { exchange_cost: 0.0 },
            ncs: NcsStagePenalties {
                curtailment_cost: 0.0,
            },
        },
    );

    SystemBuilder::new()
        .buses(vec![bus])
        .thermals(vec![thermal])
        .hydros(vec![hydro])
        .stages(stages)
        .inflow_models(inflow_models)
        .load_models(load_models)
        .bounds(bounds)
        .penalties(penalties)
        .initial_conditions(InitialConditions {
            past_anticipated_commitments: past_commits,
            ..InitialConditions::default()
        })
        .post_study_stages(post_study_stages)
        .build()
        .expect("minimal_system_with_anticipated_and_commitments: valid")
}

fn minimal_system_with_anticipated_lead_stages(
    n_stages: usize,
    lead_stages: u32,
) -> cobre_core::System {
    minimal_system_with_anticipated(
        &vec![744.0; n_stages],
        AnticipatedConfig::LeadStages(lead_stages),
        lead_stages as usize,
        None,
    )
}

#[test]
fn setup_wires_anticipated_metadata_into_indexer() {
    let system = minimal_system_with_anticipated_lead_stages(2, 2);
    let config = minimal_config(1, 10);
    let stochastic = build_stochastic_context(
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
    .expect("stochastic context");

    let setup = StudySetup::new(
        &system,
        &config,
        stochastic,
        PrepareHydroModelsResult::default_from_system(&system),
        Vec::new(),
    )
    .expect("setup");

    assert_eq!(
        setup.inputs.stage_data.state.n_anticipated, 1,
        "expected n_anticipated == 1"
    );
    assert_eq!(
        setup.inputs.stage_data.state.k_max, 2,
        "expected k_max == 2"
    );
    assert_eq!(
        setup.inputs.stage_data.state.anticipated_lead_stages,
        vec![2],
        "expected anticipated_lead_stages == [2]"
    );
}

/// A `LeadStages(2)` plant on a 5-stage uniform study resolves to per-plant
/// depth max 2 with singleton decision sets `{t+2}`; `k_max == 2` and
/// `n_state == 3` (the delivery-anchored values).
#[test]
fn setup_leadstages_resolution_preserves_k_max_and_state_dimension() {
    let system = minimal_system_with_anticipated_lead_stages(5, 2);
    let (resolution, lead_stages) = super::resolve_anticipated_commitments(
        &system,
        &DeliveryCalendar::from_system(&system),
        &AnticipatedPlants::build(system.thermals()),
    );

    assert_eq!(lead_stages, vec![2], "LeadStages keeps the constant ℓ == 2");
    let point = &resolution.per_plant[0];
    assert_eq!(
        point.depth.iter().copied().max(),
        Some(2),
        "per-plant depth max == ℓ == 2"
    );
    assert_eq!(point.decision_sets[0], vec![2], "C(0) == {{2}}");
    assert_eq!(point.decision_sets[1], vec![3], "C(1) == {{3}}");
    assert_eq!(point.decision_sets[2], vec![4], "C(2) == {{4}}");
    assert_eq!(
        resolution.anchored_depth(),
        2,
        "delivery-anchored ring depth == 2"
    );

    let config = minimal_config(1, 10);
    let stochastic = build_stochastic_context(
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
    .expect("stochastic context");
    let setup = StudySetup::new(
        &system,
        &config,
        stochastic,
        PrepareHydroModelsResult::default_from_system(&system),
        Vec::new(),
    )
    .expect("setup");

    assert_eq!(setup.inputs.stage_data.state.k_max, 2, "k_max unchanged");
    // n_state = N*(1+L) + A*k_max = 1*(1+0) + 1*2 = 3 (no PAR lags, one hydro).
    assert_eq!(
        setup.inputs.stage_data.state.n_state, 3,
        "state_dimension unchanged"
    );
    assert_eq!(
        setup
            .inputs
            .stage_data
            .state
            .anticipated_resolution
            .per_plant
            .len(),
        1,
        "resolution threaded onto StageData.state"
    );
}

/// Hand-derived: a `LeadTime(720.0)` plant on the weekly-then-monthly PMO
/// calendar `[168,168,168,168,720,720]` resolves (end-anchored `resolve_point`)
/// to `decider == [None,None,None,None,Some(3),Some(4)]`, `C(3) == {4}`,
/// `C(4) == {5}`, `depth == [0,0,0,1,1,0]`, `occupancy == [3,2,1,1,1,0]`. Its
/// leading None-run (4) exceeds both its max depth (1) and its max occupancy
/// (3, which subtracts the seed maturing at stage 0), so `ring_depth` sizes the
/// ring at 4 to hold every simultaneous pre-study seed.
#[test]
fn test_anticipated_resolve_point_pmo_calendar() {
    let system = minimal_system_with_anticipated(
        &[168.0, 168.0, 168.0, 168.0, 720.0, 720.0],
        AnticipatedConfig::LeadTime(720.0),
        6,
        None,
    );
    let (resolution, _) = super::resolve_anticipated_commitments(
        &system,
        &DeliveryCalendar::from_system(&system),
        &AnticipatedPlants::build(system.thermals()),
    );
    let point = &resolution.per_plant[0];

    assert_eq!(
        point.decider,
        vec![None, None, None, None, Some(3), Some(4)]
    );
    assert_eq!(point.decision_sets[3], vec![4]);
    assert_eq!(point.decision_sets[4], vec![5]);
    assert_eq!(point.depth, vec![0, 0, 0, 1, 1, 0]);
    assert_eq!(point.occupancy, vec![3, 2, 1, 1, 1, 0]);
    assert_eq!(resolution.anchored_depth(), 4);
}

/// Anchor: a `LeadTime(350.0)` plant on the uniform `[100.0; 4]` calendar
/// resolves cumulative boundaries `[0, 100, 200, 300, 400]`; the per-delivery
/// targets `end_m - 350` are `-250, -150, -50, +50`, so only delivery 3 has a
/// non-negative target and decides at stage 0 — `decider == [None, None,
/// None, Some(0)]`.
#[test]
fn lead_time_three_stage_lead_resolves_a_pre_study_prefix() {
    let system =
        minimal_system_with_anticipated(&[100.0; 4], AnticipatedConfig::LeadTime(350.0), 1, None);
    let (resolution, _) = super::resolve_anticipated_commitments_core(
        &system,
        &DeliveryCalendar::from_system(&system),
        &AnticipatedPlants::build(system.thermals()),
    );
    let point = &resolution.per_plant[0];

    assert_eq!(point.decider, vec![None, None, None, Some(0)]);
}

/// Ring-undersizing reproduction: the same four-stage `LeadTime` shape's true
/// stage-0 in-flight set is `{1, 2, 3}` (delivery 1 and 2 are pre-study,
/// `None`-decided; delivery 3 is decided at stage 0) — three items, so the
/// ring must be at least `k_max == 3` deep to hold them. `AnticipatedResolution::anchored_depth`
/// derives from `PointResolution::depth`, which structurally excludes
/// pre-study (`None`-decider) occupancy from its running count, so it
/// resolves a ring too narrow to hold the true in-flight set.
#[test]
fn ring_depth_counts_pre_study_occupancy() {
    let system =
        minimal_system_with_anticipated(&[100.0; 4], AnticipatedConfig::LeadTime(350.0), 1, None);
    let (resolution, _) = super::resolve_anticipated_commitments_core(
        &system,
        &DeliveryCalendar::from_system(&system),
        &AnticipatedPlants::build(system.thermals()),
    );

    assert_eq!(resolution.anchored_depth(), 3);
}

/// `LeadStages` at `ℓ == n_stages` with no post-study calendar: every delivery
/// is pre-study, so the leading None-run (`n_stages`) exceeds the occupancy max
/// (`n_stages − 1`, which subtracts the seed maturing at stage 0). `ring_depth`
/// sizes `k_max` at `ℓ` directly, and the per-plant lead stays `ℓ`, so the two
/// agree without leaning on `resolve_state_layout`'s
/// `.max(anticipated_lead_stages)` clamp.
#[test]
fn leadstages_ring_depth_covers_full_lead_when_lead_equals_horizon() {
    let n_stages = 4;
    let system =
        minimal_system_with_anticipated_lead_stages(n_stages, u32::try_from(n_stages).unwrap());
    let (resolution, lead_stages) = super::resolve_anticipated_commitments_core(
        &system,
        &DeliveryCalendar::from_system(&system),
        &AnticipatedPlants::build(system.thermals()),
    );

    assert_eq!(resolution.anchored_depth(), n_stages);
    assert_eq!(lead_stages, vec![n_stages]);
}

/// Hand-derived: a `LeadTime(720.0)` plant on the monthly-then-weekly fan-out
/// calendar `[720,168,168,168,168,168]` resolves to coarse decider 0 committing
/// four fine delivery stages — `C(0) == {1,2,3,4}` — `depth == [4,4,3,2,1,0]`,
/// ring depth 4.
#[test]
fn test_anticipated_resolve_point_fanout_calendar() {
    let system = minimal_system_with_anticipated(
        &[720.0, 168.0, 168.0, 168.0, 168.0, 168.0],
        AnticipatedConfig::LeadTime(720.0),
        6,
        None,
    );
    let (resolution, _) = super::resolve_anticipated_commitments(
        &system,
        &DeliveryCalendar::from_system(&system),
        &AnticipatedPlants::build(system.thermals()),
    );
    let point = &resolution.per_plant[0];

    assert_eq!(point.decision_sets[0], vec![1, 2, 3, 4]);
    assert_eq!(point.decision_sets[0].len(), 4);
    assert_eq!(point.depth, vec![4, 4, 3, 2, 1, 0]);
    assert_eq!(resolution.anchored_depth(), 4);
}

/// The bug doc's reproduction shape as a `System`: a `LeadTime(1160)` thermal
/// whose lead resolves all four study deliveries pre-study, plus one post-study
/// month. `resolve_anticipated_commitments_core` widens the per-plant lead to
/// the ring depth `4` — every simultaneous pre-study seed — matching
/// `resolution.anchored_depth()`, not the occupancy max `3` the pre-fix sizing reported.
#[test]
fn resolve_anticipated_commitments_widens_lead_time_plant_lead_to_the_ring_depth() {
    let post_study = PostStudyStages {
        stages: vec![PostStudyStage {
            start_date: NaiveDate::from_ymd_opt(2024, 2, 18).unwrap(),
            duration_hours: 648.0,
        }],
        thermal_bounds: Vec::new(),
    };
    let system = minimal_system_with_anticipated(
        &[168.0, 168.0, 168.0, 648.0],
        AnticipatedConfig::LeadTime(1160.0),
        4,
        Some(post_study),
    );
    let (resolution, lead_stages) = super::resolve_anticipated_commitments_core(
        &system,
        &DeliveryCalendar::from_system(&system),
        &AnticipatedPlants::build(system.thermals()),
    );

    assert_eq!(
        resolution.per_plant[0].occupancy.iter().copied().max(),
        Some(3),
        "occupancy max stays 3; ring_depth restores the fourth simultaneous seed"
    );
    assert_eq!(resolution.anchored_depth(), 4);
    assert_eq!(
        lead_stages,
        vec![4],
        "per-plant lead widened to ring_depth == k_max"
    );
}

/// Assert `state` is finalized and byte-for-byte reproduces a fresh
/// `StateSpace::new` over the same `(N, L, A, k_max, leads)` — the single-owner
/// property `resolve_state_layout` guarantees. Role-(b) equipment geometry lives
/// on `StageGeometry`, so there is no state half to compare here.
fn assert_state_layout_finalized(state: &StateSpace) {
    assert_eq!(
        state.state_to_lp_column_map.len(),
        state.n_state,
        "state_to_lp_column_map must be finalized to n_state length"
    );
    let reference = StateSpace::new(
        state.hydro_count,
        state.max_par_order,
        state.transit_bucket_column_order.clone(),
        state.anticipated_lead_stages.clone(),
        state.anticipated_resolution.clone(),
        &vec![state.max_par_order; state.hydro_count],
    );
    assert_eq!(
        state.state_to_lp_column_map, reference.state_to_lp_column_map,
        "state_to_lp_column_map must match a fresh StateSpace::new"
    );
    assert_eq!(state.n_state, reference.n_state, "n_state must match");
    assert_eq!(state.theta, reference.theta, "theta column must match");
    assert_eq!(state.storage, reference.storage, "storage range must match");
    assert_eq!(
        state.inflow_lags, reference.inflow_lags,
        "inflow_lags range must match"
    );
    assert_eq!(
        state.transit_buckets_out, reference.transit_buckets_out,
        "transit_buckets_out range must match"
    );
    assert_eq!(
        state.commit_in, reference.commit_in,
        "commit_in range must match"
    );
    assert_eq!(
        state.commit_out, reference.commit_out,
        "commit_out range must match"
    );
    assert_eq!(
        state.z_inflow, reference.z_inflow,
        "z_inflow range must match"
    );
    assert_eq!(
        state.storage_in, reference.storage_in,
        "storage_in range must match"
    );
    assert_eq!(
        state.transit_buckets_in, reference.transit_buckets_in,
        "transit_buckets_in range must match"
    );
}

/// `StateSpace::build`, resolved from a system's hydros/topology, must be
/// byte-for-byte identical to a fresh `StateSpace::new` fed the same six
/// values — the parity the constructor split (owner + loose) must preserve.
/// The fixture declares a real travel-time arc (`B > 0`) and a real
/// anticipated thermal, whose count sizes the independently-built leads
/// (`A > 0`), at a non-zero lag depth (`L > 0`).
#[test]
fn state_space_build_matches_the_loose_constructor() {
    let downstream = bucket_seed_hydro(1, None, None);
    let upstream = bucket_seed_hydro(2, Some(1), Some(24.0));
    let bus = Bus {
        id: EntityId(1),
        name: "B1".to_string(),
        operational_start_date: bucket_seed_date(2024, 1, 1),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 500.0,
        }],
        excess_cost: 0.0,
    };
    let thermal = Thermal {
        id: EntityId(3),
        name: "T1".to_string(),
        operational_start_date: bucket_seed_date(2024, 1, 1),
        bus_id: EntityId(1),
        min_generation_mw: 0.0,
        max_generation_mw: 100.0,
        cost_per_mwh: 50.0,
        anticipated_config: Some(AnticipatedConfig::LeadStages(2)),
        entry_stage_id: None,
        exit_stage_id: None,
    };
    let system = SystemBuilder::new()
        .buses(vec![bus])
        .thermals(vec![thermal])
        .hydros(vec![downstream, upstream])
        .stages(bucket_seed_study_stages(4, 24.0))
        .build()
        .expect("valid system");

    let calendar = DeliveryCalendar::from_system(&system);
    let topology = crate::bucket_topology::build_transit_bucket_topology(&system, &calendar, false);
    let anticipated_plants = AnticipatedPlants::build(system.thermals());
    let leads = vec![2; anticipated_plants.len()];
    let n_stages = leads.iter().copied().max().unwrap_or(0) + 2;
    let resolution = test_support::constant_lead_resolution(&leads, n_stages);
    let max_par_order = 1;
    let effective_lag_counts = vec![max_par_order; system.hydros().len()];

    assert!(topology.n_buckets() > 0, "fixture sanity: B > 0");
    assert!(!leads.is_empty(), "fixture sanity: A > 0");
    assert!(max_par_order > 0, "fixture sanity: L > 0");

    let built = StateSpace::build(
        system.hydros(),
        max_par_order,
        &effective_lag_counts,
        &topology,
        leads.clone(),
        resolution.clone(),
    );
    let loose = StateSpace::new(
        system.hydros().len(),
        max_par_order,
        topology.column_order.clone(),
        leads,
        resolution,
        &effective_lag_counts,
    );

    assert_eq!(format!("{built:?}"), format!("{loose:?}"));
}

#[test]
fn stage_data_state_matches_indexer_role_a_uniform() {
    let system = minimal_system(3);
    let config = minimal_config(2, 10);
    let stochastic = build_stochastic_context(
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
    .expect("stochastic context");

    let setup = StudySetup::new(
        &system,
        &config,
        stochastic,
        PrepareHydroModelsResult::default_from_system(&system),
        Vec::new(),
    )
    .expect("setup");

    assert_state_layout_finalized(&setup.inputs.stage_data.state);
}

/// Given a boundary-inferred depth (24) greater than the fitted AR order
/// (2, from `minimal_system_2_hydros_with_history`'s `ar_coefficients: vec![0.5,
/// 0.3]`), `resolve_state_layout` must widen BOTH the dense stride
/// (`max_par_order`) AND the per-hydro activeness mask in lockstep: the
/// `StateRegion::Lag` mask must contain `24 * hydro_count` active entries, not
/// `2 * hydro_count` — a regression that fails if the mask is left un-widened
/// while the dense stride grows (the exact wrong-but-compiling outcome this
/// lockstep widening exists to prevent).
#[test]
fn resolve_state_layout_widens_dense_stride_and_mask_to_declared_depth() {
    let system = minimal_system_2_hydros_with_history(3, None, vec![]);
    let config = minimal_config(1, 5);

    let stochastic = build_stochastic_context(
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
    .expect("stochastic context");

    let setup = StudySetup::new_with_boundary_requirements(
        &system,
        &config,
        stochastic,
        PrepareHydroModelsResult::default_from_system(&system),
        BoundaryStateRequirements::present(24),
        Vec::new(),
    )
    .expect("setup with a boundary depth exceeding the AR order");

    let state = &setup.inputs.stage_data.state;
    assert_eq!(
        state.max_par_order, 24,
        "dense stride must widen to the boundary-inferred depth (AR order is 2)"
    );

    let lag_active_count = state
        .nonzero_state_indices
        .iter()
        .filter(|d| state.inflow_lags.contains(&d.get()))
        .count();
    assert_eq!(
        lag_active_count,
        24 * system.hydros().len(),
        "lag mask must mark all 24 declared slots active per hydro, not just the AR order"
    );
}

/// Given a boundary-inferred depth (1) LESS than the fitted AR order (2),
/// `resolve_state_layout` must never shrink below the AR order: `max(AR order,
/// declared depth)` floors at AR order, both for the dense stride and for every
/// hydro's activeness-mask entry.
#[test]
fn resolve_state_layout_floors_declared_depth_at_ar_order() {
    let system = minimal_system_2_hydros_with_history(3, None, vec![]);
    let config = minimal_config(1, 5);

    let stochastic = build_stochastic_context(
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
    .expect("stochastic context");

    let setup = StudySetup::new_with_boundary_requirements(
        &system,
        &config,
        stochastic,
        PrepareHydroModelsResult::default_from_system(&system),
        BoundaryStateRequirements::present(1),
        Vec::new(),
    )
    .expect("setup with a boundary depth below the AR order");

    let state = &setup.inputs.stage_data.state;
    assert_eq!(
        state.max_par_order, 2,
        "a boundary depth below the AR order must not shrink the dense stride"
    );

    let lag_active_count = state
        .nonzero_state_indices
        .iter()
        .filter(|d| state.inflow_lags.contains(&d.get()))
        .count();
    assert_eq!(
        lag_active_count,
        2 * system.hydros().len(),
        "a boundary depth below the AR order must leave the mask at the unwidened AR order"
    );
}

/// Cross-crate coherence: cobre-io's seed lag depth
/// (`cobre_io::seed_lag_state_depth`, the formula `max_seed_lag_depth` uses) and
/// cobre-sddp's `resolve_state_layout` dense stride (`state.max_par_order`) must
/// return the identical `L_state` for a study with no loaded boundary — both are
/// the PAR-derived depth. A drift desyncs the load-time seed derivation from the
/// runtime state layout. A loaded boundary may widen the runtime deeper (its own
/// lags are supplied by the boundary cuts, not seeded), so this coherence is the
/// no-boundary case. The fixture's fitted AR order is 2 with no annual component.
#[test]
fn cobre_io_seed_depth_matches_resolve_state_layout_depth() {
    const FIXTURE_AR_ORDER: usize = 2;

    let system = minimal_system_2_hydros_with_history(3, None, vec![]);
    let config = minimal_config(1, 5);

    let stochastic = build_stochastic_context(
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
    .expect("stochastic context");

    let setup = StudySetup::new(
        &system,
        &config,
        stochastic,
        PrepareHydroModelsResult::default_from_system(&system),
        Vec::new(),
    )
    .expect("setup");

    let io_depth = cobre_io::seed_lag_state_depth(FIXTURE_AR_ORDER, false);
    assert_eq!(
        setup.inputs.stage_data.state.max_par_order, io_depth,
        "cobre-io seed depth and resolve_state_layout dense stride must agree \
         (both the PAR-derived depth) without a loaded boundary"
    );
}

/// Cross-crate coherence: a `StageIdResolver` built (via the cobre-io
/// constructor) from a `System`'s study stages agrees, in both directions, with
/// the canonical `study_stage_ids` slice `StudySetup` carries. Fails if the
/// resolver's index semantics ever diverge from that slice.
#[test]
fn stage_id_resolver_agrees_with_study_stage_ids() {
    let system = minimal_system_2_hydros_with_history(3, None, vec![]);
    let config = minimal_config(1, 5);
    let stochastic = build_stochastic_context(
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
    .expect("stochastic context");

    let setup = StudySetup::new(
        &system,
        &config,
        stochastic,
        PrepareHydroModelsResult::default_from_system(&system),
        Vec::new(),
    )
    .expect("setup");

    // Resolver built independently from the System's study stages.
    let ids: Vec<i32> = system
        .stages()
        .iter()
        .filter(|s| s.id >= 0)
        .map(|s| s.id)
        .collect();
    let resolver = cobre_io::StageIdResolver::from_study_stage_ids(&ids);

    assert_eq!(
        resolver.study_stage_ids(),
        setup.inputs.study_stage_ids.as_slice()
    );
    for (i, &id) in setup.inputs.study_stage_ids.iter().enumerate() {
        assert_eq!(resolver.resolve(id), Some(i));
        assert_eq!(resolver.id_at(i), Some(id));
    }
}

/// 2-hydro cascade: hydro 2 (upstream) declares a travel-time arc into hydro 1
/// (downstream), so `bucket_topology.n_buckets() > 0`.
#[expect(
    clippy::too_many_lines,
    clippy::items_after_statements,
    reason = "the fixture spells out one complete study inline so each assertion traces to a literal, and fixture-local helper items sit beside the entities that use them"
)]
fn system_with_travel_time_arc(n_stages: usize) -> cobre_core::System {
    use chrono::NaiveDate;

    let bus = Bus {
        id: EntityId(1),
        name: "B1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 500.0,
        }],
        excess_cost: 0.0,
    };

    let thermal = Thermal {
        id: EntityId(3),
        name: "T1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        bus_id: EntityId(1),
        min_generation_mw: 0.0,
        max_generation_mw: 100.0,
        cost_per_mwh: 50.0,
        anticipated_config: None,
        entry_stage_id: None,
        exit_stage_id: None,
    };

    let make_hydro =
        |id: i32, name: &str, downstream_id: Option<i32>, travel_time_hours: Option<f64>| {
            let mut hydro = Hydro {
                unit_groups: Vec::new(),
                id: EntityId(id),
                name: name.to_string(),
                operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                downstream_id: downstream_id.map(EntityId),
                travel_time_hours,
                entry_stage_id: None,
                exit_stage_id: None,
                min_storage_hm3: 0.0,
                max_storage_hm3: 200.0,
                min_outflow_m3s: 0.0,
                max_outflow_m3s: None,
                generation_model: HydroGenerationModel::ConstantProductivity,
                min_turbined_m3s: 0.0,
                max_turbined_m3s: 100.0,
                specific_productivity_mw_per_m3s_per_m: None,
                min_generation_mw: 0.0,
                max_generation_mw: 250.0,
                tailrace: None,
                hydraulic_losses: None,
                efficiency: None,
                evaporation_coefficients_mm: None,
                evaporation_reference_volumes_hm3: None,
                diversion: None,
                filling: None,
                penalties: HydroPenalties {
                    spillage_cost: 0.01,
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
            hydro.declare_mirror_unit_group(EntityId(1));
            hydro
        };

    // One 31-day calendar month per stage: `StageCalendar::new` (the
    // travel-time seed's resolver) requires chronologically ordered,
    // non-overlapping stage dates — a fixed date range shared by every stage
    // (the pre-StageCalendar fixture) violates that precondition.
    let mut stage_cursor = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();
    let stages: Vec<Stage> = (0..n_stages)
        .map(|i| {
            let start_date = stage_cursor;
            let end_date = start_date + chrono::Duration::days(31);
            stage_cursor = end_date;
            Stage {
                index: i,
                id: i as i32,
                start_date,
                end_date,
                season_id: None,
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
                    branching_factor: 1,
                    noise_method: NoiseMethod::Saa,
                },
            }
        })
        .collect();

    let inflow_models: Vec<InflowModel> = [1_i32, 2]
        .into_iter()
        .flat_map(|hid| {
            (0..n_stages).map(move |i| InflowModel {
                hydro_id: EntityId(hid),
                stage_id: i as i32,
                mean_m3s: 80.0,
                std_m3s: 20.0,
                ar_coefficients: vec![],
                residual_std_ratio: 1.0,
                annual: None,
            })
        })
        .collect();

    let load_models: Vec<LoadModel> = (0..n_stages)
        .map(|i| LoadModel {
            bus_id: EntityId(1),
            stage_id: i as i32,
            mean_mw: 100.0,
            std_mw: 0.0,
        })
        .collect();

    let n_st = n_stages.max(1);

    fn default_hydro_bounds() -> HydroStageBounds {
        HydroStageBounds {
            min_storage_hm3: 0.0,
            max_storage_hm3: 200.0,
            filling_min_rate_m3s: 0.0,
            water_withdrawal_m3s: 0.0,
        }
    }

    fn default_hydro_block_bounds() -> HydroBlockBounds {
        HydroBlockBounds {
            max_turbined_m3s: 100.0,
            max_generation_mw: 250.0,
            ..Default::default()
        }
    }

    fn default_hydro_penalties() -> HydroPenalties {
        HydroPenalties {
            spillage_cost: 0.01,
            diversion_cost: 0.0,
            turbined_cost: 0.0,
            storage_violation_below_cost: 500.0,
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

    let bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 2,
            n_thermals: 1,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages: n_st,
            k_max: 0,
        },
        &BoundsDefaults {
            hydro: default_hydro_bounds(),
            hydro_block: default_hydro_block_bounds(),
            thermal: ThermalStageBounds { cost_per_mwh: 0.0 },
            thermal_block: ThermalBlockBounds {
                min_generation_mw: 0.0,
                max_generation_mw: 100.0,
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

    let penalties = ResolvedPenalties::new(
        &PenaltiesCountsSpec {
            n_hydros: 2,
            n_buses: 1,
            n_lines: 0,
            n_ncs: 0,
            n_stages: n_st,
        },
        &PenaltiesDefaults {
            hydro: default_hydro_penalties(),
            bus: BusStagePenalties { excess_cost: 0.0 },
            line: LineStagePenalties { exchange_cost: 0.0 },
            ncs: NcsStagePenalties {
                curtailment_cost: 0.0,
            },
        },
    );

    SystemBuilder::new()
        .buses(vec![bus])
        .thermals(vec![thermal])
        .hydros(vec![
            make_hydro(1, "H1_downstream", None, None),
            make_hydro(2, "H2_upstream", Some(1), Some(2000.0)),
        ])
        .stages(stages)
        .inflow_models(inflow_models)
        .load_models(load_models)
        .bounds(bounds)
        .penalties(penalties)
        .initial_conditions(InitialConditions {
            storage: vec![],
            filling_storage: vec![],
            past_anticipated_commitments: vec![],
            recent_observations: vec![],
            past_defluences: vec![],
        })
        .build()
        .expect("system_with_travel_time_arc: valid")
}

/// A declared arc (`n_buckets > 0`) sizes `StageData.state` and the LP
/// template's `n_state` to the same value — both read the one threaded instance.
#[test]
fn setup_state_and_stage_template_agree_on_n_state_with_declared_arc() {
    let system = system_with_travel_time_arc(6);
    let setup = setup_from_system(&system);

    assert!(
        setup.inputs.stage_data.state.n_buckets > 0,
        "fixture must declare a real travel-time arc"
    );
    assert_eq!(
        setup.inputs.stage_data.stage_templates.templates[0].n_state,
        setup.inputs.stage_data.state.n_state,
        "the template's n_state must agree with StageData.state.n_state"
    );
}

/// Geometry byte-identity (the role-(b) analogue of
/// `assert_state_layout_finalized`): the production stage-0 `StageGeometry` is
/// byte-identical to an independent `test_support::geometry` build; a divergence
/// means the per-stage geometry drifted from the fixture's column/row arithmetic.
#[test]
fn stage_data_geometry_role_b_matches_reference_build() {
    let system = minimal_system(3);
    let config = minimal_config(2, 10);
    let stochastic = build_stochastic_context(
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
    .expect("stochastic context");

    let setup = StudySetup::new(
        &system,
        &config,
        stochastic,
        PrepareHydroModelsResult::default_from_system(&system),
        Vec::new(),
    )
    .expect("setup");

    let geometry = &setup.inputs.stage_data.stage_templates.geometry_per_stage[0];
    let study_dims = &setup.inputs.stage_data.study_dims;
    let dims = test_support::GeometryDims {
        hydro_count: geometry.water_balance.range().len(),
        max_par_order: 0, // role-(b) ranges do not depend on L
        n_thermals: system.thermals().len(),
        n_lines: system.lines().len(),
        n_buses: system.buses().len(),
        n_blks: geometry.n_blks,
        has_inflow_penalty: study_dims.inflow_method.has_slack_columns(),
        max_deficit_segments: study_dims.max_deficit_segments,
        n_anticipated: study_dims.anticipated_plants.len(),
        lead_stages: 0,
        anticipated_plants: study_dims.anticipated_plants.clone(),
    };
    let reference = test_support::geometry(
        &dims,
        geometry
            .fpha_hydro_indices
            .iter()
            .map(|h| h.get())
            .collect(),
        // minimal_system has no FPHA planes; mirror the built geometry's
        // FPHA-hydro count with a placeholder plane count per hydro.
        &vec![1usize; geometry.fpha_hydro_indices.len()],
        geometry
            .evap_hydro_indices
            .iter()
            .map(|h| h.get())
            .collect(),
    );

    assert_eq!(reference.turbine, geometry.turbine, "turbine range");
    assert_eq!(reference.spillage, geometry.spillage, "spillage range");
    assert_eq!(reference.diversion, geometry.diversion, "diversion range");
    assert_eq!(reference.thermal, geometry.thermal, "thermal range");
    assert_eq!(
        reference.anticipated_decision, geometry.anticipated_decision,
        "anticipated_decision range"
    );
    assert_eq!(reference.line_fwd, geometry.line_fwd, "line_fwd range");
    assert_eq!(reference.line_rev, geometry.line_rev, "line_rev range");
    assert_eq!(reference.deficit, geometry.deficit, "deficit range");
    assert_eq!(reference.excess, geometry.excess, "excess range");
    assert_eq!(
        reference.withdrawal_slack_neg, geometry.withdrawal_slack_neg,
        "withdrawal_slack_neg range"
    );
    assert_eq!(
        reference.withdrawal_slack_pos, geometry.withdrawal_slack_pos,
        "withdrawal_slack_pos range"
    );
    assert_eq!(
        reference.water_balance, geometry.water_balance,
        "water_balance"
    );
    assert_eq!(
        reference.load_balance, geometry.load_balance,
        "load_balance"
    );
    assert_eq!(reference.n_blks, geometry.n_blks, "n_blks");
}

/// `StageData.state` byte-identity with anticipated thermals present (`K_i = 2`),
/// exercising the `commit_out`/`commit_in` ranges.
#[test]
fn stage_data_state_matches_indexer_role_a_anticipated() {
    let system = minimal_system_with_anticipated_lead_stages(2, 2);
    let config = minimal_config(1, 10);
    let stochastic = build_stochastic_context(
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
    .expect("stochastic context");

    let setup = StudySetup::new(
        &system,
        &config,
        stochastic,
        PrepareHydroModelsResult::default_from_system(&system),
        Vec::new(),
    )
    .expect("setup");

    assert_eq!(
        setup.inputs.stage_data.state.n_anticipated, 1,
        "fixture sanity"
    );
    assert_state_layout_finalized(&setup.inputs.stage_data.state);
}

/// Cut-row byte-identity: the production `build_cut_row_batch` reading role-(a)
/// from `StageData.state` matches an independent reference loop over the same
/// `StateSpace` (mask, `theta`, `lp_column_for_state`) — the substitutability
/// guarantee the cut-path repoint relies on (same LP columns, same
/// negated-scaled coefficients).
#[test]
fn cut_row_from_state_matches_reference_loop() {
    use crate::cut::FutureCostFunction;
    use crate::cut::row::build_cut_row_batch;

    let system = minimal_system(3);
    let config = minimal_config(2, 10);
    let stochastic = build_stochastic_context(
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
    .expect("stochastic context");

    let setup = StudySetup::new(
        &system,
        &config,
        stochastic,
        PrepareHydroModelsResult::default_from_system(&system),
        Vec::new(),
    )
    .expect("setup");

    let state = &setup.inputs.stage_data.state;
    let n_state = state.n_state;
    assert!(n_state > 0, "fixture must have a non-empty state vector");

    // `u32::try_from` + `f64::from` keeps the indices lossless without a
    // cast-precision lint relaxation.
    let mut fcf = FutureCostFunction::new(2, n_state, 1, 4, &[0; 2]);
    let coefficients: Vec<f64> = (0..n_state)
        .map(|j| 1.0 + f64::from(u32::try_from(j).expect("state index fits u32")))
        .collect();
    fcf.add_cut(NodeId(0), 0, 0, 0, 7.5, &coefficients);

    let from_production = build_cut_row_batch(
        &fcf,
        0,
        state,
        &test_support::cut_state_projection(state),
        &[],
    );

    // Mirror of `build_cut_row_batch_into`'s mask-driven body; a disagreement
    // means the cut-path repoint changed the emitted row.
    let mut from_state = cobre_solver::RowBatch {
        num_rows: 0,
        row_starts: Vec::new(),
        col_indices: Vec::new(),
        values: Vec::new(),
        row_lower: Vec::new(),
        row_upper: Vec::new(),
    };
    let theta_col = state.theta;
    let mask = &state.nonzero_state_indices;
    for (_slot, intercept, coeffs) in fcf.active_cuts(0) {
        from_state.row_starts.push(0);
        for &j in mask {
            let lp_col = state.lp_column_for_state(j).get();
            from_state
                .col_indices
                .push(i32::try_from(lp_col).expect("col fits i32"));
            from_state.values.push(-coeffs[j.get()]);
        }
        from_state
            .col_indices
            .push(i32::try_from(theta_col).expect("theta fits i32"));
        from_state.values.push(1.0);
        from_state.row_lower.push(intercept);
        from_state.row_upper.push(f64::INFINITY);
    }
    from_state
        .row_starts
        .push(i32::try_from(mask.len() + 1).expect("nnz fits i32"));
    from_state.num_rows = 1;

    assert_eq!(
        from_production.row_starts, from_state.row_starts,
        "row_starts must be byte-identical"
    );
    assert_eq!(
        from_production.col_indices, from_state.col_indices,
        "col_indices must be byte-identical"
    );
    assert_eq!(
        from_production.values, from_state.values,
        "values must be byte-identical"
    );
    assert_eq!(
        from_production.row_lower, from_state.row_lower,
        "row_lower must be byte-identical"
    );
    assert_eq!(
        from_production.row_upper, from_state.row_upper,
        "row_upper must be byte-identical"
    );
    assert_eq!(
        from_production.num_rows, from_state.num_rows,
        "num_rows must match"
    );
}

// ── per-stage cut-pool sizing (build_cut_state_layouts) ───────────────────

/// PAR(2) study (one stage per `state_configs` entry): AR(2) coefficients plus
/// pre-study inflow models at stage ids -1/-2 give the PAR builder its lag
/// statistics, so the global `StateSpace` has `n_state = N*(1 + 2)`.
#[expect(
    clippy::too_many_lines,
    reason = "the fixture spells out one complete study inline so each assertion traces to a literal"
)]
fn par2_system_with_state_configs(state_configs: &[StageStateConfig]) -> cobre_core::System {
    use chrono::NaiveDate;

    const PHI_1: f64 = 0.5;
    const PHI_2: f64 = 0.2;
    const N_SEASONS: usize = 12;
    let hydro_id = EntityId(3);

    let bus = Bus {
        id: EntityId(1),
        name: "B1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 500.0,
        }],
        excess_cost: 0.0,
    };

    let thermal = Thermal {
        id: EntityId(2),
        name: "T1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        bus_id: EntityId(1),
        min_generation_mw: 0.0,
        max_generation_mw: 100.0,
        cost_per_mwh: 50.0,
        anticipated_config: None,
        entry_stage_id: None,
        exit_stage_id: None,
    };

    let mut hydro = Hydro {
        unit_groups: Vec::new(),
        id: hydro_id,
        name: "H1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        downstream_id: None,
        travel_time_hours: None,
        entry_stage_id: None,
        exit_stage_id: None,
        min_storage_hm3: 0.0,
        max_storage_hm3: 500.0,
        min_outflow_m3s: 0.0,
        max_outflow_m3s: None,
        generation_model: HydroGenerationModel::ConstantProductivity,
        min_turbined_m3s: 0.0,
        max_turbined_m3s: 200.0,
        specific_productivity_mw_per_m3s_per_m: None,
        min_generation_mw: 0.0,
        max_generation_mw: 200.0,
        tailrace: None,
        hydraulic_losses: None,
        efficiency: None,
        evaporation_coefficients_mm: None,
        evaporation_reference_volumes_hm3: None,
        diversion: None,
        filling: None,
        penalties: HydroPenalties {
            spillage_cost: 0.01,
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
    hydro.declare_mirror_unit_group(EntityId(1));

    let n_stages = state_configs.len();
    let stages: Vec<Stage> = state_configs
        .iter()
        .enumerate()
        .map(|(i, &state_config)| Stage {
            index: i,
            id: i as i32,
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            season_id: Some(i % N_SEASONS),
            blocks: vec![Block {
                index: 0,
                name: "S".to_string(),
                duration_hours: 744.0,
            }],
            block_mode: BlockMode::Parallel,
            state_config,
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor: 1,
                noise_method: NoiseMethod::Saa,
            },
        })
        .collect();

    let mut inflow_models: Vec<InflowModel> = Vec::new();
    for pre_id in [-2_i32, -1_i32] {
        inflow_models.push(InflowModel {
            hydro_id,
            stage_id: pre_id,
            mean_m3s: 1000.0,
            std_m3s: 200.0,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
            annual: None,
        });
    }
    for i in 0..n_stages {
        inflow_models.push(InflowModel {
            hydro_id,
            stage_id: i as i32,
            mean_m3s: 1000.0,
            std_m3s: 200.0,
            ar_coefficients: vec![PHI_1, PHI_2],
            residual_std_ratio: 0.7,
            annual: None,
        });
    }

    let load_models: Vec<LoadModel> = (0..n_stages)
        .map(|i| LoadModel {
            bus_id: EntityId(1),
            stage_id: i as i32,
            mean_mw: 100.0,
            std_mw: 0.0,
        })
        .collect();

    let bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 1,
            n_thermals: 1,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages: n_stages.max(1),
            k_max: 0,
        },
        &BoundsDefaults {
            hydro: HydroStageBounds {
                min_storage_hm3: 0.0,
                max_storage_hm3: 500.0,
                filling_min_rate_m3s: 0.0,
                water_withdrawal_m3s: 0.0,
            },
            hydro_block: HydroBlockBounds {
                max_turbined_m3s: 200.0,
                max_generation_mw: 200.0,
                ..Default::default()
            },
            thermal: ThermalStageBounds { cost_per_mwh: 0.0 },
            thermal_block: ThermalBlockBounds {
                min_generation_mw: 0.0,
                max_generation_mw: 100.0,
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

    let penalties = ResolvedPenalties::new(
        &PenaltiesCountsSpec {
            n_hydros: 1,
            n_buses: 1,
            n_lines: 0,
            n_ncs: 0,
            n_stages: n_stages.max(1),
        },
        &PenaltiesDefaults {
            hydro: HydroPenalties {
                spillage_cost: 0.01,
                diversion_cost: 0.0,
                turbined_cost: 0.0,
                storage_violation_below_cost: 500.0,
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
            bus: BusStagePenalties { excess_cost: 0.0 },
            line: LineStagePenalties { exchange_cost: 0.0 },
            ncs: NcsStagePenalties {
                curtailment_cost: 0.0,
            },
        },
    );

    SystemBuilder::new()
        .buses(vec![bus])
        .thermals(vec![thermal])
        .hydros(vec![hydro])
        .stages(stages)
        .inflow_models(inflow_models)
        .load_models(load_models)
        .bounds(bounds)
        .penalties(penalties)
        .initial_conditions(InitialConditions::default())
        .build()
        .expect("par2_system_with_state_configs: valid")
}

fn setup_from_system(system: &cobre_core::System) -> StudySetup {
    let config = minimal_config(1, 10);
    let stochastic = build_stochastic_context(
        system,
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
    .expect("stochastic context");
    StudySetup::new(
        system,
        &config,
        stochastic,
        PrepareHydroModelsResult::default_from_system(system),
        Vec::new(),
    )
    .expect("setup")
}

/// The `t + 1` governance rule: with `stages[1].state_config.inflow_lags =
/// false` and all other stages enabling lags, only `pools[0]` (governed by
/// stage 1) drops the lag dimensions; every other pool keeps the full global
/// `n_state`. With N=1 hydro, L=2, A=0: storage-only is `N = 1`,
/// `n_state = N*(1 + L) = 3`.
#[test]
fn cut_pool_sizing_t_plus_1_reduces_pool_zero_for_lagless_successor() {
    let lags = StageStateConfig {
        storage: true,
        inflow_lags: true,
    };
    let storage_only = StageStateConfig {
        storage: true,
        inflow_lags: false,
    };
    // Stage 1 is lag-less; it governs pool 0 (its predecessor).
    let system = par2_system_with_state_configs(&[lags, storage_only, lags, lags]);
    let setup = setup_from_system(&system);

    let global_n_state = setup.inputs.stage_data.state.n_state;
    assert_eq!(global_n_state, 3, "N=1, L=2 → n_state = N*(1+L) = 3");
    assert_eq!(setup.fcf.pools.len(), 4);

    // pool 0 ← stages[1] (inflow_lags=false) → storage-only = N + A*k_max = 1.
    assert_eq!(
        setup.fcf.pools[0].state_dimension, 1,
        "pool 0 is governed by stage 1's lag-less config (the t+1 rule)"
    );
    // pools 1, 2 ← stages[2], stages[3] (lags) → global n_state.
    assert_eq!(setup.fcf.pools[1].state_dimension, global_n_state);
    assert_eq!(setup.fcf.pools[2].state_dimension, global_n_state);
    // Terminal pool 3 ← full global n_state (no successor stage).
    assert_eq!(setup.fcf.pools[3].state_dimension, global_n_state);
}

/// Result-neutrality: when every stage enables all dimensions (the default
/// for every shipped case), every pool's `state_dimension` equals the global
/// `StateSpace::n_state`. This is the bit-identical-to-today guarantee.
#[test]
fn cut_pool_sizing_all_enabled_matches_global_n_state() {
    let lags = StageStateConfig {
        storage: true,
        inflow_lags: true,
    };
    let system = par2_system_with_state_configs(&[lags, lags, lags, lags]);
    let setup = setup_from_system(&system);

    let global_n_state = setup.inputs.stage_data.state.n_state;
    assert_eq!(global_n_state, 3);
    assert_eq!(setup.fcf.pools.len(), 4);
    for (t, pool) in setup.fcf.pools.iter().enumerate() {
        assert_eq!(
            pool.state_dimension, global_n_state,
            "all-enabled pool {t} must equal the global n_state"
        );
    }
    assert_eq!(setup.fcf.state_dimension, global_n_state);
}

/// Each pool's `CutStateProjection` has `n_slots()` equal to the pool's
/// `state_dimension` — the pairing the backward pass relies on to extract duals
/// at the right dimension.
#[test]
fn cut_state_layouts_stored_one_per_pool_and_reachable() {
    let lags = StageStateConfig {
        storage: true,
        inflow_lags: true,
    };
    let storage_only = StageStateConfig {
        storage: true,
        inflow_lags: false,
    };
    let system = par2_system_with_state_configs(&[lags, storage_only, lags, lags]);
    let setup = setup_from_system(&system);

    assert_eq!(
        setup.inputs.cut_state_layouts.len(),
        setup.fcf.pools.len(),
        "exactly one CutStateProjection per pool",
    );
    for (t, layout) in setup.inputs.cut_state_layouts.iter().enumerate() {
        assert_eq!(
            layout.n_slots(),
            setup.fcf.pools[t].state_dimension,
            "cut_state_layouts[{t}].n_slots() must match pool {t} dimension",
        );
    }

    // Pool `t` is sized by stage `t+1`'s config. The `storage_only` config sits
    // at stage index 1, so it sizes pool 0: its cut dimension drops the AR lags
    // to exactly the hydro (storage) count N, while pool 1 (sized by stage 2's
    // lag-enabled config) carries storage + lags (N*(1+L) > N).
    let n_hydros = setup.inputs.stage_data.state.hydro_count;
    assert_eq!(
        setup.fcf.pools[0].state_dimension, n_hydros,
        "storage-only pool 0 must have cut dimension N (lags dropped)",
    );
    assert!(
        setup.fcf.pools[1].state_dimension > n_hydros,
        "lag-enabled pool 1 must carry storage + lags (dimension > N)",
    );
}

// ---------------------------------------------------------------------------
// K = 0 sub-stage lead (`c(m) = m`) — exclude-with-advisory
// ---------------------------------------------------------------------------

/// WARN-capturing `tracing::Subscriber`, mirroring `params::tests::WarnRecorder`.
struct WarnRecorder {
    messages: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
}

impl WarnRecorder {
    fn new() -> (Self, std::sync::Arc<std::sync::Mutex<Vec<String>>>) {
        let messages = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        (
            Self {
                messages: std::sync::Arc::clone(&messages),
            },
            messages,
        )
    }
}

impl tracing::Subscriber for WarnRecorder {
    fn enabled(&self, metadata: &tracing::Metadata<'_>) -> bool {
        *metadata.level() <= tracing::Level::WARN
    }

    fn new_span(&self, _attrs: &tracing::span::Attributes<'_>) -> tracing::span::Id {
        tracing::span::Id::from_u64(1)
    }

    fn record(&self, _span: &tracing::span::Id, _values: &tracing::span::Record<'_>) {}

    fn record_follows_from(&self, _span: &tracing::span::Id, _follows: &tracing::span::Id) {}

    fn event(&self, event: &tracing::Event<'_>) {
        if *event.metadata().level() == tracing::Level::WARN {
            struct MessageVisitor(String);
            impl tracing::field::Visit for MessageVisitor {
                fn record_debug(
                    &mut self,
                    field: &tracing::field::Field,
                    value: &dyn std::fmt::Debug,
                ) {
                    if field.name() == "message" {
                        self.0 = format!("{value:?}");
                    }
                }

                fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
                    if field.name() == "message" {
                        self.0 = value.to_string();
                    }
                }
            }
            let mut visitor = MessageVisitor(String::new());
            event.record(&mut visitor);
            self.messages.lock().unwrap().push(visitor.0);
        }
    }

    fn enter(&self, _span: &tracing::span::Id) {}

    fn exit(&self, _span: &tracing::span::Id) {}
}

/// Hand-derived: a `LeadTime(720.0)` plant on the uniform 31-day-month
/// calendar `[744, 744, 744, 744]` h resolves `c(m) = m` at every delivery
/// stage (the 720h lead is shorter than each 744h stage, so `end_m - 720`
/// always lands inside stage `m`'s own window) — the `K = 0` sub-stage-lead
/// degeneracy, `depth == [0, 0, 0, 0]` and `resolution.anchored_depth() == 0` (never an
/// underflow).
#[test]
fn test_anticipated_resolve_point_k0_uniform_calendar() {
    let system = minimal_system_with_anticipated(
        &[744.0, 744.0, 744.0, 744.0],
        AnticipatedConfig::LeadTime(720.0),
        0,
        None,
    );
    let (resolution, lead_stages) = super::resolve_anticipated_commitments(
        &system,
        &DeliveryCalendar::from_system(&system),
        &AnticipatedPlants::build(system.thermals()),
    );
    let point = &resolution.per_plant[0];

    assert_eq!(
        point.decider,
        vec![Some(0), Some(1), Some(2), Some(3)],
        "every delivery stage self-delivers (K=0)"
    );
    assert_eq!(point.depth, vec![0, 0, 0, 0]);
    assert_eq!(resolution.anchored_depth(), 0, "ring depth collapses to 0");
    assert_eq!(resolution.max_fanout(), 0, "no genuine fan-out either");
    assert_eq!(
        point.self_delivered_stages().collect::<Vec<_>>(),
        vec![0, 1, 2, 3],
        "every stage is a self-delivery"
    );
    // The still-live constant-lead machinery's placeholder is the per-plant
    // max depth — 0 here, never underflowed downstream.
    assert_eq!(lead_stages, vec![0]);
}

/// `resolve_anticipated_commitments` emits one `tracing::WARN` event per
/// `K = 0` self-delivered stage (exclude-with-advisory, never a hard
/// error), naming the thermal, the stage, and the `lead_stages == 0`
/// stage-count alternative. Setup/load-time only (this is a direct,
/// non-per-trajectory call).
#[test]
fn resolve_anticipated_commitments_warns_on_k0_sub_stage_lead() {
    let system = minimal_system_with_anticipated(
        &[744.0, 744.0, 744.0, 744.0],
        AnticipatedConfig::LeadTime(720.0),
        0,
        None,
    );

    let (subscriber, messages) = WarnRecorder::new();
    tracing::subscriber::with_default(subscriber, || {
        let _ = super::resolve_anticipated_commitments(
            &system,
            &DeliveryCalendar::from_system(&system),
            &AnticipatedPlants::build(system.thermals()),
        );
    });
    let recorded = messages.lock().unwrap();
    let relevant: Vec<&str> = recorded
        .iter()
        .filter(|msg| msg.contains("lead_stages == 0"))
        .map(std::string::String::as_str)
        .collect();
    assert_eq!(
        relevant.len(),
        4,
        "expected one advisory per self-delivered stage (4 stages), got: {recorded:?}"
    );
    for (stage, msg) in relevant.iter().enumerate() {
        assert!(
            msg.contains(&format!("stage {stage}")),
            "advisory {stage} must name its stage; got: {msg}"
        );
        assert!(
            msg.contains("T1"),
            "advisory {stage} must name the plant; got: {msg}"
        );
    }
}

/// A `LeadStages` plant never triggers the `K = 0` advisory: a positive
/// stage-count lead never resolves `c(m) = m`.
#[test]
fn resolve_anticipated_commitments_leadstages_never_warns() {
    let system = minimal_system_with_anticipated_lead_stages(5, 2);

    let (subscriber, messages) = WarnRecorder::new();
    tracing::subscriber::with_default(subscriber, || {
        let _ = super::resolve_anticipated_commitments(
            &system,
            &DeliveryCalendar::from_system(&system),
            &AnticipatedPlants::build(system.thermals()),
        );
    });
    let recorded = messages.lock().unwrap();
    assert!(
        recorded.iter().all(|msg| !msg.contains("lead_stages == 0")),
        "LeadStages must never trigger the K=0 advisory; got: {recorded:?}"
    );
}

/// A mixed-cadence calendar where exactly ONE of several delivery stages
/// self-delivers (`c(m) = m`): three weekly (168h) stages decided ahead of
/// time, then one monthly (744h) stage whose 200h lead is shorter than its
/// own duration. Discriminates the fixture above (every stage self-delivers,
/// so it cannot tell "one advisory per self-delivered stage" apart from "one
/// advisory per stage"): exactly one advisory fires, naming that stage and
/// the plant.
#[test]
fn warn_on_sub_stage_lead_emits_once_per_self_delivered_stage() {
    let system = minimal_system_with_anticipated(
        &[168.0, 168.0, 168.0, 744.0],
        AnticipatedConfig::LeadTime(200.0),
        0,
        None,
    );

    let (subscriber, messages) = WarnRecorder::new();
    tracing::subscriber::with_default(subscriber, || {
        let _ = super::resolve_anticipated_commitments(
            &system,
            &DeliveryCalendar::from_system(&system),
            &AnticipatedPlants::build(system.thermals()),
        );
    });
    let recorded = messages.lock().unwrap();
    let relevant: Vec<&str> = recorded
        .iter()
        .filter(|msg| msg.contains("lead_stages == 0"))
        .map(std::string::String::as_str)
        .collect();
    assert_eq!(
        relevant.len(),
        1,
        "expected exactly one advisory for the single self-delivered stage, got: {recorded:?}"
    );
    assert!(
        relevant[0].contains("stage 3"),
        "advisory must name stage 3, got: {}",
        relevant[0]
    );
    assert!(
        relevant[0].contains("T1"),
        "advisory must name the plant, got: {}",
        relevant[0]
    );
}

// ---------------------------------------------------------------------------
// Extended delivery axis — n_delivery spans n_stages + n_post
// ---------------------------------------------------------------------------

/// The no-half-switch invariant: `resolve_anticipated_commitments_core`'s
/// primary axis site resolves a plant's `decider` vector to the EXTENDED
/// delivery width `n_stages + n_post`, not the study-only `n_stages` a
/// half-switched site would still report. Two study stages plus two declared
/// post-study stages resolve `decider.len() == 4`.
#[test]
fn resolve_anticipated_commitments_core_reports_the_extended_delivery_width() {
    let post_study = PostStudyStages {
        stages: vec![
            PostStudyStage {
                start_date: NaiveDate::from_ymd_opt(2024, 3, 1).unwrap(),
                duration_hours: 744.0,
            },
            PostStudyStage {
                start_date: NaiveDate::from_ymd_opt(2024, 4, 1).unwrap(),
                duration_hours: 720.0,
            },
        ],
        thermal_bounds: Vec::new(),
    };
    let system = minimal_system_with_anticipated(
        &[744.0, 744.0],
        AnticipatedConfig::LeadStages(1),
        1,
        Some(post_study),
    );

    let (resolution, _) = super::resolve_anticipated_commitments_core(
        &system,
        &DeliveryCalendar::from_system(&system),
        &AnticipatedPlants::build(system.thermals()),
    );

    assert_eq!(
        resolution.per_plant[0].decider.len(),
        4,
        "n_stages (2) + n_post (2) must resolve to 4, the extended delivery axis"
    );
}

/// A study declaring no post-study stages leaves the primary axis site
/// byte-identical to the pre-widening study-only axis: `decider.len() ==
/// n_stages`.
#[test]
fn resolve_anticipated_commitments_core_matches_study_only_width_without_post_study() {
    let system = minimal_system_with_anticipated(
        &[744.0, 744.0, 744.0],
        AnticipatedConfig::LeadStages(1),
        1,
        None,
    );

    let (resolution, _) = super::resolve_anticipated_commitments_core(
        &system,
        &DeliveryCalendar::from_system(&system),
        &AnticipatedPlants::build(system.thermals()),
    );

    assert_eq!(resolution.per_plant[0].decider.len(), 3);
}

// ---------------------------------------------------------------------------
// Boundary-absent post-study delivery advisory
// ---------------------------------------------------------------------------

/// A `LeadStages(1)` plant with one declared post-study stage decides its
/// post-study-targeted delivery in-study (`m = n_stages`, `c(m) = n_stages -
/// 1 < n_stages`). With no `config.policy.boundary` declared, exactly one
/// advisory fires naming the plant, once at setup.
#[test]
fn warn_on_boundary_absent_post_study_delivery_fires_once_when_boundary_absent() {
    let post_study = PostStudyStages {
        stages: vec![PostStudyStage {
            start_date: NaiveDate::from_ymd_opt(2024, 3, 1).unwrap(),
            duration_hours: 744.0,
        }],
        thermal_bounds: Vec::new(),
    };
    let system = minimal_system_with_anticipated(
        &[744.0, 744.0],
        AnticipatedConfig::LeadStages(1),
        1,
        Some(post_study),
    );
    let (resolution, _) = super::resolve_anticipated_commitments_core(
        &system,
        &DeliveryCalendar::from_system(&system),
        &AnticipatedPlants::build(system.thermals()),
    );

    let (subscriber, messages) = WarnRecorder::new();
    tracing::subscriber::with_default(subscriber, || {
        super::warn_on_boundary_absent_post_study_delivery(
            &system,
            &DeliveryCalendar::from_system(&system),
            &AnticipatedPlants::build(system.thermals()),
            &resolution,
            false,
        );
    });
    let recorded = messages.lock().unwrap();
    let relevant: Vec<&str> = recorded
        .iter()
        .filter(|msg| msg.contains("post-study-targeted delivery"))
        .map(std::string::String::as_str)
        .collect();
    assert_eq!(
        relevant.len(),
        1,
        "expected exactly one boundary-absent advisory, got: {recorded:?}"
    );
    assert!(
        relevant[0].contains("T1"),
        "advisory must name the plant, got: {}",
        relevant[0]
    );
}

/// The same fixture with `boundary_present == true` emits no advisory — the
/// post-study delivery has a real terminal FCF to price against.
#[test]
fn warn_on_boundary_absent_post_study_delivery_silent_when_boundary_present() {
    let post_study = PostStudyStages {
        stages: vec![PostStudyStage {
            start_date: NaiveDate::from_ymd_opt(2024, 3, 1).unwrap(),
            duration_hours: 744.0,
        }],
        thermal_bounds: Vec::new(),
    };
    let system = minimal_system_with_anticipated(
        &[744.0, 744.0],
        AnticipatedConfig::LeadStages(1),
        1,
        Some(post_study),
    );
    let (resolution, _) = super::resolve_anticipated_commitments_core(
        &system,
        &DeliveryCalendar::from_system(&system),
        &AnticipatedPlants::build(system.thermals()),
    );

    let (subscriber, messages) = WarnRecorder::new();
    tracing::subscriber::with_default(subscriber, || {
        super::warn_on_boundary_absent_post_study_delivery(
            &system,
            &DeliveryCalendar::from_system(&system),
            &AnticipatedPlants::build(system.thermals()),
            &resolution,
            true,
        );
    });
    let recorded = messages.lock().unwrap();
    assert!(
        recorded
            .iter()
            .all(|msg| !msg.contains("post-study-targeted delivery")),
        "a declared boundary must suppress the advisory; got: {recorded:?}"
    );
}

/// A plant with NO class-3 reach (no post-study stages declared, so
/// `decider.get(n_stages..)` is empty) that declares one non-zero class-4
/// fixed post-horizon window (`start_date >= horizon_end`) and no boundary:
/// exactly one advisory fires and names it.
#[test]
fn warn_on_boundary_absent_fires_for_nonzero_fixed_value_without_boundary() {
    use cobre_core::AnticipatedCommitmentHistory;

    let system = minimal_system_with_anticipated_and_commitments(
        &[744.0, 744.0],
        AnticipatedConfig::LeadStages(1),
        1,
        None,
        vec![AnticipatedCommitmentHistory {
            thermal_id: EntityId(2),
            start_date: NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2025, 2, 1).unwrap(),
            value_mw: 42.0,
        }],
    );
    let (resolution, _) = super::resolve_anticipated_commitments_core(
        &system,
        &DeliveryCalendar::from_system(&system),
        &AnticipatedPlants::build(system.thermals()),
    );

    let (subscriber, messages) = WarnRecorder::new();
    tracing::subscriber::with_default(subscriber, || {
        super::warn_on_boundary_absent_post_study_delivery(
            &system,
            &DeliveryCalendar::from_system(&system),
            &AnticipatedPlants::build(system.thermals()),
            &resolution,
            false,
        );
    });
    let recorded = messages.lock().unwrap();
    let relevant: Vec<&str> = recorded
        .iter()
        .filter(|msg| msg.contains("post-study-targeted delivery"))
        .map(std::string::String::as_str)
        .collect();
    assert_eq!(
        relevant.len(),
        1,
        "expected exactly one boundary-absent advisory for the non-zero fixed value, got: {recorded:?}"
    );
    assert!(
        relevant[0].contains("T1"),
        "advisory must name the plant, got: {}",
        relevant[0]
    );
}

/// The same no-class-3-reach fixture, but the plant's only post-horizon
/// declaration is an all-zero tiling (the mandatory horizon-end 0 MW stub):
/// a zero value is provably inert, so no advisory fires.
#[test]
fn warn_on_boundary_absent_silent_for_all_zero_stub_without_boundary() {
    use cobre_core::AnticipatedCommitmentHistory;

    let system = minimal_system_with_anticipated_and_commitments(
        &[744.0, 744.0],
        AnticipatedConfig::LeadStages(1),
        1,
        None,
        vec![AnticipatedCommitmentHistory {
            thermal_id: EntityId(2),
            start_date: NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2025, 2, 1).unwrap(),
            value_mw: 0.0,
        }],
    );
    let (resolution, _) = super::resolve_anticipated_commitments_core(
        &system,
        &DeliveryCalendar::from_system(&system),
        &AnticipatedPlants::build(system.thermals()),
    );

    let (subscriber, messages) = WarnRecorder::new();
    tracing::subscriber::with_default(subscriber, || {
        super::warn_on_boundary_absent_post_study_delivery(
            &system,
            &DeliveryCalendar::from_system(&system),
            &AnticipatedPlants::build(system.thermals()),
            &resolution,
            false,
        );
    });
    let recorded = messages.lock().unwrap();
    assert!(
        recorded
            .iter()
            .all(|msg| !msg.contains("post-study-targeted delivery")),
        "an all-zero fixed post-horizon window must not trigger the advisory; got: {recorded:?}"
    );
}

/// A plant hitting BOTH the class-3 condition (the `_fires_once_when_boundary_absent`
/// fixture) AND a non-zero class-4 fixed window: the merged predicate still
/// names it exactly once in the one emitted event, never a duplicate.
#[test]
fn warn_on_boundary_absent_names_dual_cause_plant_once() {
    use cobre_core::AnticipatedCommitmentHistory;

    let post_study = PostStudyStages {
        stages: vec![PostStudyStage {
            start_date: NaiveDate::from_ymd_opt(2024, 3, 1).unwrap(),
            duration_hours: 744.0,
        }],
        thermal_bounds: Vec::new(),
    };
    let system = minimal_system_with_anticipated_and_commitments(
        &[744.0, 744.0],
        AnticipatedConfig::LeadStages(1),
        1,
        Some(post_study),
        vec![AnticipatedCommitmentHistory {
            thermal_id: EntityId(2),
            start_date: NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2025, 2, 1).unwrap(),
            value_mw: 42.0,
        }],
    );
    let (resolution, _) = super::resolve_anticipated_commitments_core(
        &system,
        &DeliveryCalendar::from_system(&system),
        &AnticipatedPlants::build(system.thermals()),
    );

    let (subscriber, messages) = WarnRecorder::new();
    tracing::subscriber::with_default(subscriber, || {
        super::warn_on_boundary_absent_post_study_delivery(
            &system,
            &DeliveryCalendar::from_system(&system),
            &AnticipatedPlants::build(system.thermals()),
            &resolution,
            false,
        );
    });
    let recorded = messages.lock().unwrap();
    let relevant: Vec<&str> = recorded
        .iter()
        .filter(|msg| msg.contains("post-study-targeted delivery"))
        .map(std::string::String::as_str)
        .collect();
    assert_eq!(
        relevant.len(),
        1,
        "both causes on one plant must still emit a single merged event, got: {recorded:?}"
    );
    assert!(
        relevant[0].contains("1 anticipated thermal(s)"),
        "the plant must be counted exactly once, not once per cause; got: {}",
        relevant[0]
    );
    assert_eq!(
        relevant[0].matches("T1").count(),
        1,
        "the plant must be named exactly once in the merged list, got: {}",
        relevant[0]
    );
}

// ---------------------------------------------------------------------------
// LeadTime fan-out — rejected at setup, not silently dropped, no panic
// ---------------------------------------------------------------------------

/// A fan-out `LeadTime` calendar (`max_fanout > 1`) is rejected at
/// `StudySetup::new` with `SddpError::Validation` naming the fanned plant —
/// no panic, no silently dropped fan member.
#[test]
fn lead_time_fanout_rejected_at_setup() {
    use crate::error::SddpError;

    let system = minimal_system_with_anticipated(
        &[744.0, 168.0, 168.0],
        AnticipatedConfig::LeadTime(900.0),
        2,
        None,
    );

    // Sanity: the fixture genuinely fans out (guards the guard's own fixture).
    let (resolution, _) = super::resolve_anticipated_commitments(
        &system,
        &DeliveryCalendar::from_system(&system),
        &AnticipatedPlants::build(system.thermals()),
    );
    assert_eq!(
        resolution.max_fanout(),
        2,
        "fixture must fan out with width 2 at decision stage 0"
    );

    let config = minimal_config(1, 10);
    let stochastic = build_stochastic_context(
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
    .expect("stochastic context");

    let result = StudySetup::new(
        &system,
        &config,
        stochastic,
        PrepareHydroModelsResult::default_from_system(&system),
        Vec::new(),
    );

    let err = result.expect_err("a fan-out LeadTime study must be rejected at setup, not panic");
    assert!(
        matches!(err, SddpError::Validation(_)),
        "expected SddpError::Validation, got: {err:?}"
    );
    let msg = err.to_string();
    assert!(
        msg.contains("anticipated thermal 2"),
        "message should name the fanned plant (thermal id 2), got: {msg}"
    );
    assert!(
        msg.contains("LeadTime fan-out"),
        "message should state LeadTime fan-out, got: {msg}"
    );
    assert!(
        msg.contains("not yet supported"),
        "message should state fan-out output is not yet supported, got: {msg}"
    );
}

/// Two anticipated thermals: id=20 non-fanning `LeadStages(1)`, id=21
/// `LeadTime(900.0)` fanning (`max_fanout == 2`) on `[744,168,168]` h. Declared
/// `[fanning, non_fanning]` (`SystemBuilder` re-sorts by `EntityId` ascending) so
/// the declaration-order-invariance test proves rejection is input-order-independent.
#[expect(
    clippy::too_many_lines,
    reason = "the fixture spells out one complete study inline so each assertion traces to a literal"
)]
fn system_with_two_thermals_one_fanning() -> cobre_core::System {
    use chrono::NaiveDate;

    let bus = Bus {
        id: EntityId(1),
        name: "B1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 500.0,
        }],
        excess_cost: 0.0,
    };

    let durations = [744.0_f64, 168.0, 168.0];
    let n_stages = durations.len();

    let non_fanning = Thermal {
        id: EntityId(20),
        name: "T_non_fanning".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        bus_id: EntityId(1),
        min_generation_mw: 0.0,
        max_generation_mw: 100.0,
        cost_per_mwh: 50.0,
        anticipated_config: Some(AnticipatedConfig::LeadStages(1)),
        entry_stage_id: None,
        exit_stage_id: None,
    };
    let fanning = Thermal {
        id: EntityId(21),
        name: "T_fanning".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        bus_id: EntityId(1),
        min_generation_mw: 0.0,
        max_generation_mw: 100.0,
        cost_per_mwh: 50.0,
        anticipated_config: Some(AnticipatedConfig::LeadTime(900.0)),
        entry_stage_id: None,
        exit_stage_id: None,
    };

    let stages: Vec<Stage> = durations
        .iter()
        .enumerate()
        .map(|(i, &duration)| Stage {
            index: i,
            id: i as i32,
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            season_id: None,
            blocks: vec![Block {
                index: 0,
                name: "S".to_string(),
                duration_hours: duration,
            }],
            block_mode: BlockMode::Parallel,
            state_config: StageStateConfig {
                storage: false,
                inflow_lags: false,
            },
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor: 1,
                noise_method: NoiseMethod::Saa,
            },
        })
        .collect();

    let load_models: Vec<LoadModel> = (0..n_stages)
        .map(|i| LoadModel {
            bus_id: EntityId(1),
            stage_id: i as i32,
            mean_mw: 100.0,
            std_mw: 0.0,
        })
        .collect();

    let k_max_bounds = 2usize;
    let mut bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 0,
            n_thermals: 2,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages,
            k_max: k_max_bounds,
        },
        &BoundsDefaults {
            hydro: HydroStageBounds {
                min_storage_hm3: 0.0,
                max_storage_hm3: 0.0,
                filling_min_rate_m3s: 0.0,
                water_withdrawal_m3s: 0.0,
            },
            hydro_block: HydroBlockBounds::default(),
            thermal: ThermalStageBounds { cost_per_mwh: 50.0 },
            thermal_block: ThermalBlockBounds {
                min_generation_mw: 0.0,
                max_generation_mw: 100.0,
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
    for thermal_idx in 0..2 {
        for s in 0..(n_stages + k_max_bounds) {
            *bounds.thermal_bounds_mut(thermal_idx, s) = ThermalStageBounds { cost_per_mwh: 50.0 };
            *bounds.thermal_block_base_mut(thermal_idx, s) = ThermalBlockBounds {
                min_generation_mw: 0.0,
                max_generation_mw: 100.0,
            };
        }
    }

    let penalties = ResolvedPenalties::new(
        &PenaltiesCountsSpec {
            n_hydros: 0,
            n_buses: 1,
            n_lines: 0,
            n_ncs: 0,
            n_stages,
        },
        &PenaltiesDefaults {
            hydro: HydroPenalties {
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
                inflow_nonnegativity_cost: 0.0,
            },
            bus: BusStagePenalties { excess_cost: 0.0 },
            line: LineStagePenalties { exchange_cost: 0.0 },
            ncs: NcsStagePenalties {
                curtailment_cost: 0.0,
            },
        },
    );

    SystemBuilder::new()
        .buses(vec![bus])
        .thermals(vec![fanning, non_fanning])
        .stages(stages)
        .load_models(load_models)
        .bounds(bounds)
        .penalties(penalties)
        .initial_conditions(InitialConditions {
            storage: vec![],
            filling_storage: vec![],
            past_anticipated_commitments: vec![],
            recent_observations: vec![],
            past_defluences: vec![],
        })
        .build()
        .expect("two-thermal fan-out system: valid")
}

/// Only the second declared thermal (canonical order: `non_fanning`=20 before
/// `fanning`=21) fans out; the rejection still fires and names it, because
/// `system.thermals()` is canonical (ID-sorted) — order-invariant by construction.
#[test]
fn lead_time_fanout_rejection_is_declaration_order_invariant() {
    use crate::error::SddpError;

    let system = system_with_two_thermals_one_fanning();

    let config = minimal_config(1, 10);
    let stochastic = build_stochastic_context(
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
    .expect("stochastic context");

    let result = StudySetup::new(
        &system,
        &config,
        stochastic,
        PrepareHydroModelsResult::default_from_system(&system),
        Vec::new(),
    );

    let err = result.expect_err(
        "the fanning plant must still be rejected regardless of its declaration position",
    );
    assert!(matches!(err, SddpError::Validation(_)));
    let msg = err.to_string();
    assert!(
        msg.contains("anticipated thermal 21"),
        "message should name the fanning plant (thermal id 21), not the non-fanning one, got: {msg}"
    );
}

/// Anticipated (`T1`=20, `T3`=22) and non-anticipated (`T2`=21) thermals
/// interleaved in declaration order, sharing one `operational_start_date` so
/// canonical order matches declaration order.
#[expect(
    clippy::too_many_lines,
    reason = "the fixture spells out one complete study inline so each assertion traces to a literal"
)]
fn system_with_interleaved_anticipated_thermals() -> cobre_core::System {
    use chrono::NaiveDate;

    let bus = Bus {
        id: EntityId(1),
        name: "B1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 500.0,
        }],
        excess_cost: 0.0,
    };

    let durations = [744.0_f64, 168.0];
    let n_stages = durations.len();
    let start_date = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();

    let make_thermal = |id: i32, anticipated_config: Option<AnticipatedConfig>| Thermal {
        id: EntityId(id),
        name: format!("T{id}"),
        operational_start_date: start_date,
        bus_id: EntityId(1),
        min_generation_mw: 0.0,
        max_generation_mw: 100.0,
        cost_per_mwh: 50.0,
        anticipated_config,
        entry_stage_id: None,
        exit_stage_id: None,
    };
    let t1 = make_thermal(20, Some(AnticipatedConfig::LeadStages(1)));
    let t2 = make_thermal(21, None);
    let t3 = make_thermal(22, Some(AnticipatedConfig::LeadStages(1)));

    let stages: Vec<Stage> = durations
        .iter()
        .enumerate()
        .map(|(i, &duration)| Stage {
            index: i,
            id: i as i32,
            start_date,
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            season_id: None,
            blocks: vec![Block {
                index: 0,
                name: "S".to_string(),
                duration_hours: duration,
            }],
            block_mode: BlockMode::Parallel,
            state_config: StageStateConfig {
                storage: false,
                inflow_lags: false,
            },
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor: 1,
                noise_method: NoiseMethod::Saa,
            },
        })
        .collect();

    let load_models: Vec<LoadModel> = (0..n_stages)
        .map(|i| LoadModel {
            bus_id: EntityId(1),
            stage_id: i as i32,
            mean_mw: 100.0,
            std_mw: 0.0,
        })
        .collect();

    let k_max_bounds = 1usize;
    let mut bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 0,
            n_thermals: 3,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages,
            k_max: k_max_bounds,
        },
        &BoundsDefaults {
            hydro: HydroStageBounds {
                min_storage_hm3: 0.0,
                max_storage_hm3: 0.0,
                filling_min_rate_m3s: 0.0,
                water_withdrawal_m3s: 0.0,
            },
            hydro_block: HydroBlockBounds::default(),
            thermal: ThermalStageBounds { cost_per_mwh: 50.0 },
            thermal_block: ThermalBlockBounds {
                min_generation_mw: 0.0,
                max_generation_mw: 100.0,
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
    for thermal_idx in 0..3 {
        for s in 0..(n_stages + k_max_bounds) {
            *bounds.thermal_bounds_mut(thermal_idx, s) = ThermalStageBounds { cost_per_mwh: 50.0 };
            *bounds.thermal_block_base_mut(thermal_idx, s) = ThermalBlockBounds {
                min_generation_mw: 0.0,
                max_generation_mw: 100.0,
            };
        }
    }

    let penalties = ResolvedPenalties::new(
        &PenaltiesCountsSpec {
            n_hydros: 0,
            n_buses: 1,
            n_lines: 0,
            n_ncs: 0,
            n_stages,
        },
        &PenaltiesDefaults {
            hydro: HydroPenalties {
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
                inflow_nonnegativity_cost: 0.0,
            },
            bus: BusStagePenalties { excess_cost: 0.0 },
            line: LineStagePenalties { exchange_cost: 0.0 },
            ncs: NcsStagePenalties {
                curtailment_cost: 0.0,
            },
        },
    );

    SystemBuilder::new()
        .buses(vec![bus])
        .thermals(vec![t1, t2, t3])
        .stages(stages)
        .load_models(load_models)
        .bounds(bounds)
        .penalties(penalties)
        .initial_conditions(InitialConditions {
            storage: vec![],
            filling_storage: vec![],
            past_anticipated_commitments: vec![],
            recent_observations: vec![],
            past_defluences: vec![],
        })
        .build()
        .expect("interleaved anticipated thermals system: valid")
}

#[test]
fn anticipated_plants_build_returns_canonical_order_of_anticipated_thermals() {
    let system = system_with_interleaved_anticipated_thermals();
    let plants = AnticipatedPlants::build(system.thermals());
    assert_eq!(
        plants.thermals().collect::<Vec<_>>(),
        vec![ThermalSys::new(0), ThermalSys::new(2)],
        "must skip the interleaved non-anticipated thermal and return canonical ascending positions"
    );
}

// -------------------------------------------------------------------------
// build_contract_prices_per_stage
// -------------------------------------------------------------------------

/// Two-contract, no-hydro/thermal system for exercising
/// `build_contract_prices_per_stage` directly, with a caller-supplied `bounds`
/// table (its contract/stage counts must match `blocks_per_stage`).
fn system_with_contracts(blocks_per_stage: &[usize], bounds: ResolvedBounds) -> cobre_core::System {
    use chrono::NaiveDate;

    let date = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();
    let bus = Bus {
        id: EntityId(1),
        name: "B1".to_string(),
        operational_start_date: date,
        deficit_segments: vec![],
        excess_cost: 0.0,
    };

    let stages: Vec<Stage> = blocks_per_stage
        .iter()
        .enumerate()
        .map(|(i, &n_blk)| Stage {
            index: i,
            id: i as i32,
            start_date: date,
            end_date: date,
            season_id: None,
            blocks: (0..n_blk)
                .map(|b| Block {
                    index: b,
                    name: format!("blk{b}"),
                    duration_hours: 100.0,
                })
                .collect(),
            block_mode: BlockMode::Parallel,
            state_config: StageStateConfig {
                storage: false,
                inflow_lags: false,
            },
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor: 1,
                noise_method: NoiseMethod::Saa,
            },
        })
        .collect();

    let contract_a = EnergyContract {
        id: EntityId(10),
        name: "CA".to_string(),
        operational_start_date: date,
        bus_id: EntityId(1),
        contract_type: ContractType::Import,
        entry_stage_id: None,
        exit_stage_id: None,
        price_per_mwh: 80.0,
        min_mw: 0.0,
        max_mw: 100.0,
    };
    let contract_b = EnergyContract {
        id: EntityId(11),
        name: "CB".to_string(),
        ..contract_a.clone()
    };

    SystemBuilder::new()
        .buses(vec![bus])
        .contracts(vec![contract_a, contract_b])
        .stages(stages)
        .bounds(bounds)
        .build()
        .expect("contract-price fixture system: valid")
}

/// Zero-sized hydro/thermal/line/pumping defaults plus the given uniform
/// contract price, for a `ResolvedBounds` covering only contracts.
fn zero_bounds_defaults(contract_price: f64) -> BoundsDefaults {
    BoundsDefaults {
        hydro: HydroStageBounds {
            min_storage_hm3: 0.0,
            max_storage_hm3: 0.0,
            filling_min_rate_m3s: 0.0,
            water_withdrawal_m3s: 0.0,
        },
        hydro_block: HydroBlockBounds::default(),
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
            max_mw: 100.0,
            price_per_mwh: contract_price,
        },
    }
}

/// A two-contract study with per-stage block counts `[3, 2]` (differing,
/// so a stride bug reading a global max-blocks count instead of the
/// per-stage geometry would misreport stage 1's length), a price that
/// varies per contract AND per stage (so a stage-axis bug — reading stage 0's
/// price for every `t` — cannot hide behind a uniform fixture), and no
/// per-block price row — every one of the `n_contracts * n_blks` cells per
/// stage compares equal under `f64::to_bits` to that contract's OWN stage-level
/// `price_per_mwh`.
#[test]
fn test_contract_prices_per_block_are_uniform_without_overlay() {
    let blocks_per_stage = [3_usize, 2_usize];
    let n_contracts = 2;
    let mut bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 0,
            n_thermals: 0,
            n_lines: 0,
            n_pumping: 0,
            n_contracts,
            n_stages: blocks_per_stage.len(),
            k_max: 0,
        },
        &zero_bounds_defaults(80.0),
    );
    bounds.contract_bounds_mut(0, 0).price_per_mwh = 80.0;
    bounds.contract_bounds_mut(0, 1).price_per_mwh = 130.0;
    bounds.contract_bounds_mut(1, 0).price_per_mwh = 95.0;
    bounds.contract_bounds_mut(1, 1).price_per_mwh = 150.0;
    let system = system_with_contracts(&blocks_per_stage, bounds);

    let prices = build_contract_prices_per_stage(
        &system,
        &test_support::equipment_free_geometry(&blocks_per_stage),
    );

    assert_eq!(prices.len(), blocks_per_stage.len());
    for (t, &n_blks) in blocks_per_stage.iter().enumerate() {
        assert_eq!(
            prices[t].len(),
            n_contracts * n_blks,
            "stage {t}: price table length must be n_contracts * n_blks"
        );
        for c in 0..n_contracts {
            let stage_price = system.bounds().contract_block_base(c, t).price_per_mwh;
            for blk in 0..n_blks {
                assert_eq!(
                    prices[t][c * n_blks + blk].to_bits(),
                    stage_price.to_bits(),
                    "stage {t} contract {c} block {blk} must equal the stage-level price"
                );
            }
        }
    }
}

/// Contract 0's stage-0 (three-block) inner slice carries a `120.0`
/// override at block 1 over the stage-wide `80.0`; contract 1's three cells
/// are unaffected by contract 0's override.
#[test]
fn test_contract_price_table_carries_per_block_override() {
    let blocks_per_stage = [3_usize];
    let mut bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 0,
            n_thermals: 0,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 2,
            n_stages: 1,
            k_max: 0,
        },
        &zero_bounds_defaults(80.0),
    );
    let mut overlay = ResolvedBlockBounds::new(&BlockBoundsCountsSpec {
        n_hydros: 0,
        n_thermals: 0,
        n_lines: 0,
        n_pumping: 0,
        n_contracts: 2,
        n_stages: 1,
        max_blocks: 3,
    });
    *overlay
        .contract_override_mut(0, 0, 1)
        .expect("in-range override cell") = ContractBlockOverride {
        min_mw: None,
        max_mw: None,
        price_per_mwh: Some(120.0),
    };
    bounds.set_block_overlay(overlay);

    let system = system_with_contracts(&blocks_per_stage, bounds);

    let prices = build_contract_prices_per_stage(
        &system,
        &test_support::equipment_free_geometry(&blocks_per_stage),
    );

    assert_eq!(
        prices[0],
        vec![80.0, 120.0, 80.0, 80.0, 80.0, 80.0],
        "contract 0 carries the block-1 override ([80, 120, 80]); contract 1's three cells are unaffected"
    );
}

// ── G2 (rule 49): external-library width assertion ───────────────────────────

/// Build a `ScenarioLibraries` whose training phase carries `inflow` as its
/// external inflow library and no other external libraries.
fn scenario_libraries_with_inflow(inflow: Option<ExternalScenarioLibrary>) -> ScenarioLibraries {
    let empty = || PhaseLibraries {
        inflow_scheme: SamplingScheme::InSample,
        load_scheme: SamplingScheme::InSample,
        ncs_scheme: SamplingScheme::InSample,
        historical: None,
        external_inflow: None,
        external_load: None,
        external_ncs: None,
    };
    let mut training = empty();
    if inflow.is_some() {
        training.inflow_scheme = SamplingScheme::External;
    }
    training.external_inflow = inflow;
    ScenarioLibraries {
        training,
        simulation: empty(),
    }
}

/// A standardized external library whose `n_entities()` disagrees with its
/// `noise_entity_order` block width is rejected naming the class and both widths.
#[test]
fn g2_rejects_external_library_width_mismatch() {
    use cobre_core::scenario::ScenarioSource;

    // minimal_system has one hydro, so the inflow block width is 1.
    let system = minimal_system(1);
    let libs = scenario_libraries_with_inflow(Some(ExternalScenarioLibrary::new(
        1,
        2,
        2,
        "inflow",
        vec![2],
    )));
    match assert_external_library_widths(&system, &libs, &ScenarioSource::default()) {
        Err(SddpError::Validation(msg)) => assert!(
            msg.contains("inflow") && msg.contains('2') && msg.contains('1'),
            "the error names the class and both widths, got: {msg}"
        ),
        other => panic!("expected a width-mismatch Validation error, got {other:?}"),
    }
}

/// A library whose width matches the `noise_entity_order` block width passes —
/// the entity order used is `noise_entity_order`'s, not a re-derivation.
#[test]
fn g2_accepts_matching_external_library_width() {
    use cobre_core::scenario::ScenarioSource;

    let system = minimal_system(1);
    let libs = scenario_libraries_with_inflow(Some(ExternalScenarioLibrary::new(
        1,
        2,
        1,
        "inflow",
        vec![2],
    )));
    assert!(
        assert_external_library_widths(&system, &libs, &ScenarioSource::default()).is_ok(),
        "a library width matching noise_entity_order's block width must pass"
    );
}

// ---------------------------------------------------------------------------
// Admission gate: risk-measure arm, census substrate, enumeration cross-check
// ---------------------------------------------------------------------------

/// A generated-source runtime node with an `openings.len` of `len`. `pool_id`,
/// `offset`, and `q` are irrelevant to `enumerated_scenario_count`, which reads
/// only `stage`, `successors`, and `openings.len`.
fn ng_node(stage: usize, len: usize) -> super::NodeRuntime {
    super::NodeRuntime {
        stage: super::node_graph::StageIdx(stage),
        pool_id: 0,
        openings: super::NodeOpenings {
            source: super::OpeningSource::Generated,
            offset: 0,
            len,
            q: 0.0,
        },
    }
}

fn ng_edge(child: usize) -> super::NodeSuccessor {
    super::NodeSuccessor {
        child: NodePos(child),
        probability: 0.5,
    }
}

/// Root (stage 0, |Ω|=2) fans to child A (stage 1, |Ω|=3) → leaf grandchild
/// (stage 2, |Ω|=5), and directly to leaf B (stage 1, |Ω|=7): the root→leaf
/// path-product-sum is 2·3·5 + 2·7 = 44. Asymmetric so the test cannot pass a
/// uniform-fan tautology.
fn asymmetric_fan_node_graph() -> super::NodeGraph {
    super::NodeGraph {
        node_ids: vec![NodeId(0), NodeId(1), NodeId(2), NodeId(3)].into(),
        nodes: vec![ng_node(0, 2), ng_node(1, 3), ng_node(2, 5), ng_node(1, 7)].into(),
        successors: vec![
            vec![ng_edge(1), ng_edge(3)],
            vec![ng_edge(2)],
            vec![],
            vec![],
        ]
        .into(),
        n_pools: 1,
        pool_stage: vec![super::node_graph::StageIdx(0)],
    }
}

/// A 3-node chain whose per-node |Ω| = 6·10⁹ makes the path product overflow
/// `u64` (`u64::MAX ≈ 1.8·10¹⁹`, so the first multiply already overflows).
fn overflowing_chain_node_graph() -> super::NodeGraph {
    super::NodeGraph {
        node_ids: vec![NodeId(0), NodeId(1), NodeId(2)].into(),
        nodes: vec![
            ng_node(0, 6_000_000_000),
            ng_node(1, 6_000_000_000),
            ng_node(2, 6_000_000_000),
        ]
        .into(),
        successors: vec![vec![ng_edge(1)], vec![ng_edge(2)], vec![]].into(),
        n_pools: 1,
        pool_stage: vec![super::node_graph::StageIdx(0)],
    }
}

fn rules_with_gap() -> crate::stopping_rule::StoppingRuleSet {
    use crate::stopping_rule::{StoppingMode, StoppingRule, StoppingRuleSet};
    StoppingRuleSet {
        rules: vec![
            StoppingRule::IterationLimit { limit: 100 },
            StoppingRule::Gap {
                tolerance: Some(1000.0),
                relative_tolerance: None,
            },
        ],
        mode: StoppingMode::Any,
    }
}

fn rules_without_gap() -> crate::stopping_rule::StoppingRuleSet {
    use crate::stopping_rule::{StoppingMode, StoppingRule, StoppingRuleSet};
    StoppingRuleSet {
        rules: vec![StoppingRule::IterationLimit { limit: 100 }],
        mode: StoppingMode::Any,
    }
}

/// A `gap` rule under enumerated forwards whose per-stage measures are NOT
/// uniform (here `Expectation` at stage 0, effective `CVaR` at stage 1) is
/// rejected, the message naming the rule, the offending stage, the measure, and
/// the admitting (uniform-measure) condition.
#[test]
fn admission_gate_rejects_gap_under_nonuniform_risk_enumerated() {
    use crate::risk_measure::RiskMeasure;
    let measures = vec![
        RiskMeasure::Expectation,
        RiskMeasure::CVaR {
            alpha: 0.1,
            lambda: 0.5,
        },
    ];
    match super::admission_gate(&measures, &rules_with_gap(), true, None) {
        Err(SddpError::Validation(msg)) => {
            assert!(msg.contains("gap"), "names the rule: {msg}");
            assert!(msg.contains("CVaR"), "names the measure: {msg}");
            assert!(msg.contains("stage 1"), "names the offending stage: {msg}");
            assert!(
                msg.contains("uniform"),
                "names the admitting condition: {msg}"
            );
        }
        other => panic!("expected a Validation reject, got {other:?}"),
    }
}

/// A `gap` rule under enumerated forwards with a UNIFORM effective `CVaR` at
/// every stage is admitted: the enumerated engine makes the exact risk-adjusted
/// upper bound computable, so the gap brackets the risk-adjusted lower bound.
#[test]
fn admission_gate_accepts_gap_under_enumerated_uniform_cvar() {
    use crate::risk_measure::RiskMeasure;
    let cvar = RiskMeasure::CVaR {
        alpha: 0.15,
        lambda: 0.4,
    };
    let measures = vec![cvar, cvar, cvar];
    assert!(
        super::admission_gate(&measures, &rules_with_gap(), true, None).is_ok(),
        "a uniform CVaR under enumerated forwards must admit a gap rule"
    );
}

/// A `gap` rule under a uniform `CVaR` but SAMPLED forwards stays rejected: the
/// upper bound is then a statistical estimate, so there is no exact risk-adjusted
/// bound to gap against.
#[test]
fn admission_gate_rejects_gap_under_uniform_cvar_sampled() {
    use crate::risk_measure::RiskMeasure;
    let cvar = RiskMeasure::CVaR {
        alpha: 0.15,
        lambda: 0.4,
    };
    let measures = vec![cvar, cvar];
    match super::admission_gate(&measures, &rules_with_gap(), false, None) {
        Err(SddpError::Validation(msg)) => {
            assert!(msg.contains("gap"), "names the rule: {msg}");
            assert!(
                msg.contains("sampled"),
                "names the offending selection: {msg}"
            );
        }
        other => panic!("expected a Validation reject, got {other:?}"),
    }
}

/// `CVaR { lambda: 0 }` is documented-equivalent to `Expectation`, so a `gap`
/// rule under it is admitted — the effective-measure predicate must not trip.
#[test]
fn admission_gate_accepts_gap_under_cvar_lambda_zero() {
    use crate::risk_measure::RiskMeasure;
    let measures = vec![RiskMeasure::CVaR {
        alpha: 0.1,
        lambda: 0.0,
    }];
    assert!(
        super::admission_gate(&measures, &rules_with_gap(), true, None).is_ok(),
        "CVaR with lambda == 0 is effectively expectation and must admit a gap rule"
    );
}

/// A `gap` rule with an expectation measure at every stage is admitted.
#[test]
fn admission_gate_accepts_gap_under_all_expectation() {
    use crate::risk_measure::RiskMeasure;
    let measures = vec![RiskMeasure::Expectation, RiskMeasure::Expectation];
    assert!(super::admission_gate(&measures, &rules_with_gap(), true, None).is_ok());
}

/// An effective `CVaR` measure with no `gap` rule present is admitted — the arm
/// gates the pairing, not risk aversion alone.
#[test]
fn admission_gate_accepts_cvar_without_gap() {
    use crate::risk_measure::RiskMeasure;
    let measures = vec![RiskMeasure::CVaR {
        alpha: 0.1,
        lambda: 0.9,
    }];
    assert!(super::admission_gate(&measures, &rules_without_gap(), true, None).is_ok());
}

/// The default study shape (expectation everywhere, an iteration-limit rule)
/// returns `Ok(())` unconditionally — the byte-neutral path.
#[test]
fn admission_gate_accepts_default_shape() {
    use crate::risk_measure::RiskMeasure;
    let measures = vec![RiskMeasure::Expectation; 4];
    assert!(super::admission_gate(&measures, &rules_without_gap(), true, None).is_ok());
}

/// A `gap` rule under sampled forward selection (`training_enumerated == false`)
/// is rejected even with an expectation measure at every stage — the exact upper
/// bound the gap needs is produced only by the enumerated engine.
#[test]
fn admission_gate_rejects_gap_under_sampled_selection() {
    use crate::risk_measure::RiskMeasure;
    let measures = vec![RiskMeasure::Expectation, RiskMeasure::Expectation];
    match super::admission_gate(&measures, &rules_with_gap(), false, None) {
        Err(SddpError::Validation(msg)) => {
            assert!(msg.contains("gap"), "names the rule: {msg}");
            assert!(
                msg.contains("sampled"),
                "names the offending selection: {msg}"
            );
            assert!(
                msg.contains("enumerated"),
                "names the admitting condition: {msg}"
            );
        }
        other => panic!("expected a Validation reject, got {other:?}"),
    }
}

/// A `gap` rule under enumerated selection with an expectation measure at every
/// stage is admitted — the sampled arm gates selection, not the presence of a
/// gap rule alone.
#[test]
fn admission_gate_accepts_gap_under_enumerated_expectation() {
    use crate::risk_measure::RiskMeasure;
    let measures = vec![RiskMeasure::Expectation, RiskMeasure::Expectation];
    assert!(super::admission_gate(&measures, &rules_with_gap(), true, None).is_ok());
}

/// `enumerated_scenario_count` returns Σ over root→leaf paths of Π |Ω|: for the
/// asymmetric fan that is 2·3·5 + 2·7 = 44.
#[test]
fn enumerated_scenario_count_returns_path_product_sum() {
    let ng = asymmetric_fan_node_graph();
    assert_eq!(
        super::node_graph::enumerated_scenario_count(&ng).unwrap(),
        44
    );
}

/// `enumerated_scenario_count` returns an overflow `Err` (never a wrapped
/// count) when the path product exceeds `u64`.
#[test]
fn enumerated_scenario_count_errors_on_overflow() {
    let ng = overflowing_chain_node_graph();
    match super::node_graph::enumerated_scenario_count(&ng) {
        Err(SddpError::Validation(msg)) => {
            assert!(msg.contains("overflow"), "names the overflow: {msg}");
        }
        other => panic!("expected an overflow Validation error, got {other:?}"),
    }
}

/// Exactly one phase declaring `enumerated` emits an advisory naming both
/// phases and the specific unavailable capability — never a generic message.
/// Both asymmetric directions are exercised.
#[test]
fn enumeration_asymmetry_warns_naming_both_phases_and_capability() {
    // Training enumerated, simulation sampled: the weighted simulation
    // statistics are the missing capability.
    {
        let (subscriber, messages) = WarnRecorder::new();
        tracing::subscriber::with_default(subscriber, || {
            super::warn_on_enumeration_asymmetry(true, false);
        });
        let recorded = messages.lock().unwrap();
        assert_eq!(recorded.len(), 1, "exactly one advisory, got: {recorded:?}");
        let msg = &recorded[0];
        assert!(
            msg.contains("training") && msg.contains("simulation"),
            "names both phases: {msg}"
        );
        assert!(
            msg.contains("simulation statistics"),
            "names the weighted simulation statistics as unavailable: {msg}"
        );
    }
    // Simulation enumerated, training sampled: the exact lower bound is the
    // missing capability.
    {
        let (subscriber, messages) = WarnRecorder::new();
        tracing::subscriber::with_default(subscriber, || {
            super::warn_on_enumeration_asymmetry(false, true);
        });
        let recorded = messages.lock().unwrap();
        assert_eq!(recorded.len(), 1, "exactly one advisory, got: {recorded:?}");
        let msg = &recorded[0];
        assert!(
            msg.contains("training") && msg.contains("simulation"),
            "names both phases: {msg}"
        );
        assert!(
            msg.contains("exact lower bound"),
            "names the exact lower bound as unavailable: {msg}"
        );
    }
}

/// Symmetric declarations (both enumerated, or neither) emit no advisory.
#[test]
fn enumeration_asymmetry_symmetric_declarations_do_not_warn() {
    for (t, s) in [(true, true), (false, false)] {
        let (subscriber, messages) = WarnRecorder::new();
        tracing::subscriber::with_default(subscriber, || {
            super::warn_on_enumeration_asymmetry(t, s);
        });
        let recorded = messages.lock().unwrap();
        assert!(
            recorded.is_empty(),
            "symmetric declaration ({t}, {s}) must not warn, got: {recorded:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// enumerated_admissible_count: the shared guard both enumerated axes call
// ---------------------------------------------------------------------------

/// Root (stage 0, `|Ω| = 1`) branches structurally to three stage-1 leaves
/// (each `|Ω| = 1`): a single-predecessor tree with no within-node opening and
/// no recombination, whose derived scenario count is the leaf count,
/// `K = 3` — the shape both `resolve_enumerated_training_count` and
/// `resolve_enumerated_simulation_count` must admit now that the derived-≥-1
/// census gate is open.
fn terminal_fan_tree_node_graph() -> super::NodeGraph {
    super::NodeGraph {
        node_ids: vec![NodeId(0), NodeId(1), NodeId(2), NodeId(3)].into(),
        nodes: vec![ng_node(0, 1), ng_node(1, 1), ng_node(1, 1), ng_node(1, 1)].into(),
        successors: vec![
            vec![ng_edge(1), ng_edge(2), ng_edge(3)],
            vec![],
            vec![],
            vec![],
        ]
        .into(),
        n_pools: 1,
        pool_stage: vec![super::node_graph::StageIdx(0)],
    }
}

/// A `K = 3` branching tree is admitted for TRAINING through the shared
/// guard — unchanged behavior, pinning the refactor byte-neutral.
#[test]
fn resolve_enumerated_training_count_admits_branching_tree() {
    let ng = terminal_fan_tree_node_graph();
    assert_eq!(super::resolve_enumerated_training_count(&ng).unwrap(), 3);
}

/// The SAME `K = 3` branching tree is admitted for SIMULATION — the
/// derived-≥-1 census gate is open, so a non-degenerate leaf-path count
/// resolves rather than rejects.
#[test]
fn resolve_enumerated_simulation_count_admits_branching_tree() {
    let ng = terminal_fan_tree_node_graph();
    assert_eq!(super::resolve_enumerated_simulation_count(&ng).unwrap(), 3);
}

/// A `K^T` overflow propagates unchanged from `enumerated_scenario_count`
/// through the shared guard — the derived count is unrepresentable, never
/// silently admitted.
#[test]
fn resolve_enumerated_simulation_count_propagates_kt_overflow_guard() {
    let ng = overflowing_chain_node_graph();
    match super::resolve_enumerated_simulation_count(&ng) {
        Err(SddpError::Validation(msg)) => {
            assert!(msg.contains("overflow"), "names the overflow: {msg}");
        }
        other => panic!("expected an overflow Validation error, got {other:?}"),
    }
}

/// A 2-node chain whose root carries `|Ω| = 2` (a within-node opening set) and
/// a singleton-opening leaf: no recombination (the root is the sole
/// predecessor of its one child), so `reject_within_node_opening_enumeration`
/// is the guard that fires, not `reject_recombining_node_enumeration`.
fn within_node_multi_opening_node_graph() -> super::NodeGraph {
    super::NodeGraph {
        node_ids: vec![NodeId(0), NodeId(1)].into(),
        nodes: vec![ng_node(0, 2), ng_node(1, 1)].into(),
        successors: vec![vec![ng_edge(1)], vec![]].into(),
        n_pools: 1,
        pool_stage: vec![super::node_graph::StageIdx(0)],
    }
}

/// A within-node multi-opening node (`|Ω| > 1`) under enumerated SIMULATION is
/// a named `Validation` rejection — the sibling requirement to the
/// recombination guard tested below, both preconditions the exact node-dedup
/// traversal needs.
#[test]
fn resolve_enumerated_simulation_count_rejects_within_node_multi_opening() {
    let ng = within_node_multi_opening_node_graph();
    match super::resolve_enumerated_simulation_count(&ng) {
        Err(SddpError::Validation(msg)) => {
            assert!(msg.contains("node id 0"), "names the offending node: {msg}");
            assert!(msg.contains("stage 0"), "names its stage: {msg}");
            assert!(msg.contains("2 openings"), "names the opening count: {msg}");
        }
        other => panic!("expected a within-node-opening Validation error, got {other:?}"),
    }
}

/// Root (stage 0) fans to two stage-1 nodes that both point at one stage-2 leaf
/// (node id 3): that leaf is reached from two distinct parents, so its
/// in-degree is 2 — a recombination join. Every node carries a singleton
/// opening set, so the within-node-opening rejection passes and the
/// recombination guard is what fires.
fn recombining_tree_node_graph() -> super::NodeGraph {
    super::NodeGraph {
        node_ids: vec![NodeId(0), NodeId(1), NodeId(2), NodeId(3)].into(),
        nodes: vec![ng_node(0, 1), ng_node(1, 1), ng_node(1, 1), ng_node(2, 1)].into(),
        successors: vec![
            vec![ng_edge(1), ng_edge(2)],
            vec![ng_edge(3)],
            vec![ng_edge(3)],
            vec![],
        ]
        .into(),
        n_pools: 1,
        pool_stage: vec![super::node_graph::StageIdx(0)],
    }
}

/// A recombining node (in-degree ≥ 2) under enumerated TRAINING is a named
/// `Validation` rejection — the release-active guard that stands in for
/// `build_parent_map`'s single-predecessor `debug_assert`, so the enumerated
/// engine never reconstructs a multi-parent node's incoming state from one
/// arbitrary parent.
#[test]
fn resolve_enumerated_training_count_rejects_recombining_node() {
    let ng = recombining_tree_node_graph();

    // Confirm the fixture genuinely recombines before asserting: node id 3 must
    // be reached by two distinct parents, or the rejection assertion is vacuous.
    let leaf_in_degree = ng
        .successors
        .iter()
        .flatten()
        .filter(|s| s.child == NodePos(3))
        .count();
    assert_eq!(leaf_in_degree, 2, "fixture must recombine at node id 3");

    match super::resolve_enumerated_training_count(&ng) {
        Err(SddpError::Validation(msg)) => {
            assert!(msg.contains("node id 3"), "names the offending node: {msg}");
            assert!(msg.contains("stage 2"), "names its stage: {msg}");
            assert!(
                msg.contains("recombination"),
                "names the recombination cause: {msg}"
            );
        }
        other => panic!("expected a recombination Validation error, got {other:?}"),
    }
}

/// The SAME recombining graph is rejected under enumerated SIMULATION too —
/// the guard is shared (`enumerated_admissible_count`), so the two axes
/// cannot admit different graph shapes.
#[test]
fn resolve_enumerated_simulation_count_rejects_recombining_node() {
    let ng = recombining_tree_node_graph();
    match super::resolve_enumerated_simulation_count(&ng) {
        Err(SddpError::Validation(msg)) => {
            assert!(msg.contains("node id 3"), "names the offending node: {msg}");
            assert!(msg.contains("stage 2"), "names its stage: {msg}");
            assert!(
                msg.contains("recombination"),
                "names the recombination cause: {msg}"
            );
        }
        other => panic!("expected a recombination Validation error, got {other:?}"),
    }
}

/// The SAME recombining graph is admitted by the sampled forward's setup-time
/// consumer (`pool_cut_stride`): sampled carries per-trajectory state and
/// needs no single-predecessor assumption, so the recombination guard is
/// enumerated-only.
#[test]
fn sampled_admits_the_recombining_node_graph() {
    let ng = recombining_tree_node_graph();
    let bound = ng.pool_cut_stride(8);
    assert_eq!(
        bound.len(),
        ng.n_pools,
        "the sampled visit-bound path consumes the recombining graph without rejection"
    );
}

// ---------------------------------------------------------------------------
// reject_scenario_id_under_sampled_selection: the footgun-close guard
// ---------------------------------------------------------------------------

/// A runtime node whose `openings.source` is `External` with scenario column
/// `offset` — the shape a declared node `scenario_id` produces.
fn ng_external_node(stage: usize, offset: usize) -> super::NodeRuntime {
    super::NodeRuntime {
        stage: super::node_graph::StageIdx(stage),
        pool_id: 0,
        openings: super::NodeOpenings {
            source: super::OpeningSource::External,
            offset,
            len: 1,
            q: 1.0,
        },
    }
}

/// Root (stage 0, Generated, node id 7) → child (stage 1, External scenario
/// column 2, node id 9): the child's `scenario_id` surfaces as an `External`
/// opening, the shape the sampled-selection guard rejects.
fn external_pointer_node_graph() -> super::NodeGraph {
    super::NodeGraph {
        node_ids: vec![NodeId(7), NodeId(9)].into(),
        nodes: vec![ng_node(0, 3), ng_external_node(1, 2)].into(),
        successors: vec![vec![ng_edge(1)], vec![]].into(),
        n_pools: 2,
        pool_stage: vec![
            super::node_graph::StageIdx(0),
            super::node_graph::StageIdx(1),
        ],
    }
}

/// A node carrying a scenario pointer (`External` opening) under sampled forward
/// selection is a named `Validation` rejection — closing the footgun where a
/// `scenario_id` under sampled would be validated at load and silently ignored
/// at solve.
#[test]
fn scenario_id_under_sampled_selection_is_rejected_naming_node_and_stage() {
    let ng = external_pointer_node_graph();

    // Confirm the fixture genuinely carries an external pointer before asserting.
    assert!(
        ng.nodes
            .iter()
            .any(|n| n.openings.source == super::OpeningSource::External),
        "fixture must carry an external-bound node"
    );

    match super::reject_scenario_id_under_sampled_selection(&ng, false) {
        Err(SddpError::Validation(msg)) => {
            assert!(msg.contains("node 9"), "names the offending node: {msg}");
            assert!(msg.contains("stage 1"), "names its stage: {msg}");
            assert!(
                msg.contains("enumerated selection"),
                "names the admitting condition: {msg}"
            );
        }
        other => panic!("expected a sampled-selection Validation error, got {other:?}"),
    }
}

/// The SAME external-pointer graph is admitted under enumerated selection — the
/// pointer is legal there (its count resolution is a separate concern).
#[test]
fn scenario_id_under_enumerated_selection_is_admitted() {
    let ng = external_pointer_node_graph();
    super::reject_scenario_id_under_sampled_selection(&ng, true)
        .expect("enumerated selection admits a node scenario_id");
}

// ---------------------------------------------------------------------------
// reject_insample_class_under_external_nodes: the unsupported-mixed-config guard
// ---------------------------------------------------------------------------

/// A non-empty in-sample class alongside an external-column node graph is a named
/// `Validation` rejection: the in-sample class would sample a wrong opening at the
/// external column offset, so the mixed config is refused at setup rather than
/// silently mis-sampled.
#[test]
fn insample_class_under_external_nodes_is_rejected_naming_class() {
    let ng = external_pointer_node_graph();

    // Confirm the fixture genuinely carries an external-column node before asserting.
    assert!(
        ng.nodes
            .iter()
            .any(|n| n.openings.source == super::OpeningSource::External),
        "fixture must carry an external-column node"
    );

    match super::reject_insample_class_under_external_nodes(
        &ng,
        (Some(SamplingScheme::External), 1),
        (Some(SamplingScheme::InSample), 1),
        (Some(SamplingScheme::InSample), 0),
    ) {
        Err(SddpError::Validation(msg)) => {
            assert!(msg.contains("load"), "names the offending class: {msg}");
            assert!(
                msg.contains("external"),
                "names the admitting condition: {msg}"
            );
        }
        other => panic!("expected a mixed-config Validation error, got {other:?}"),
    }
}

/// The all-external config whose only in-sample class is zero-entity (the
/// degenerate NCS an all-external study still carries) is admitted — a class that
/// draws nothing never trips the guard.
#[test]
fn all_external_with_empty_insample_ncs_is_admitted() {
    let ng = external_pointer_node_graph();
    super::reject_insample_class_under_external_nodes(
        &ng,
        (Some(SamplingScheme::External), 1),
        (Some(SamplingScheme::External), 1),
        (Some(SamplingScheme::InSample), 0),
    )
    .expect("all-external with an empty in-sample NCS class is admitted");
}

/// A graph with no external-column node admits non-empty in-sample classes: the
/// guard is gated on external-node presence, so an ordinary generated graph is
/// never touched.
#[test]
fn insample_classes_without_external_nodes_are_admitted() {
    let ng = asymmetric_fan_node_graph();
    assert!(
        ng.nodes
            .iter()
            .all(|n| n.openings.source == super::OpeningSource::Generated),
        "fixture must be all-generated"
    );
    super::reject_insample_class_under_external_nodes(
        &ng,
        (Some(SamplingScheme::InSample), 1),
        (Some(SamplingScheme::InSample), 1),
        (Some(SamplingScheme::InSample), 0),
    )
    .expect("a generated-only graph admits in-sample classes");
}

/// Documented exhaustive-destructure guard (a compile-fail proxy; trybuild is
/// not wired in this workspace). `is_effective_non_expectation` destructures
/// `RiskMeasure` and `rule_is_gap` destructures `StoppingRule` with every field
/// and variant named — no `..` on the gated variants, no `_ =>` arm — so adding
/// a field to `RiskMeasure::CVaR` / `StoppingRule::Gap`, or a new variant to
/// either enum, fails to compile until it is dispositioned in those matches.
/// This test exercises every current arm so the totality is executed, not only
/// asserted in prose.
#[test]
fn admission_gate_predicates_destructure_exhaustively() {
    use crate::risk_measure::RiskMeasure;
    use crate::stopping_rule::StoppingRule;

    assert!(!super::is_effective_non_expectation(
        &RiskMeasure::Expectation
    ));
    assert!(!super::is_effective_non_expectation(&RiskMeasure::CVaR {
        alpha: 0.1,
        lambda: 0.0,
    }));
    assert!(super::is_effective_non_expectation(&RiskMeasure::CVaR {
        alpha: 0.1,
        lambda: 0.5,
    }));

    assert!(super::rule_is_gap(&StoppingRule::Gap {
        tolerance: Some(1.0),
        relative_tolerance: None,
    }));
    assert!(!super::rule_is_gap(&StoppingRule::IterationLimit {
        limit: 1
    }));
    assert!(!super::rule_is_gap(&StoppingRule::TimeLimit {
        seconds: 1.0
    }));
    assert!(!super::rule_is_gap(&StoppingRule::BoundStalling {
        tolerance: 0.1,
        iterations: 1,
    }));
    assert!(!super::rule_is_gap(&StoppingRule::GracefulShutdown));
}

// ---------------------------------------------------------------------------
// Travel-time bucket seed (`build_initial_transit_bucket_state`)
// ---------------------------------------------------------------------------

fn bucket_seed_zero_penalties() -> HydroPenalties {
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
        inflow_nonnegativity_cost: 0.0,
    }
}

fn bucket_seed_date(y: i32, m: u32, d: u32) -> NaiveDate {
    NaiveDate::from_ymd_opt(y, m, d).unwrap()
}

fn bucket_seed_hydro(id: i32, downstream_id: Option<i32>, travel_time_hours: Option<f64>) -> Hydro {
    let mut hydro = Hydro {
        unit_groups: Vec::new(),
        id: EntityId(id),
        name: format!("H{id}"),
        operational_start_date: bucket_seed_date(2024, 1, 1),
        downstream_id: downstream_id.map(EntityId),
        travel_time_hours,
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
        penalties: bucket_seed_zero_penalties(),
    };
    hydro.declare_mirror_unit_group(EntityId(1));
    hydro
}

/// `n` study stages (`id = 0..n`), each carrying a single `hours`-long block,
/// all anchored at `start_0 = 2024-01-01`.
/// One whole calendar day per stage, independent of `hours`: `NaiveDate` has
/// no sub-day resolution, and `StageCalendar::hour_window_shares` (the seed's
/// resolver) reads only each `Stage`'s `duration_hours`, never its calendar
/// dates — the nominal one-day-per-stage span exists solely to satisfy
/// `StageCalendar::new`'s chronological-ordering precondition.
fn bucket_seed_study_stages(n: i32, hours: f64) -> Vec<Stage> {
    (0..n)
        .map(|id| {
            let start_date = bucket_seed_date(2024, 1, 1) + Duration::days(i64::from(id));
            let end_date = start_date + Duration::days(1);
            Stage {
                index: usize::try_from(id).unwrap_or(0),
                id,
                start_date,
                end_date,
                season_id: None,
                blocks: vec![Block {
                    index: 0,
                    name: "FLAT".to_string(),
                    duration_hours: hours,
                }],
                block_mode: BlockMode::Parallel,
                state_config: StageStateConfig {
                    storage: true,
                    inflow_lags: false,
                },
                risk_config: StageRiskConfig::Expectation,
                scenario_config: ScenarioSourceConfig {
                    branching_factor: 1,
                    noise_method: NoiseMethod::Saa,
                },
            }
        })
        .collect()
}

fn bucket_seed_build_system(
    hydros: Vec<Hydro>,
    stages: Vec<Stage>,
    past_defluences: Vec<HydroPastDefluence>,
) -> cobre_core::System {
    let bus = Bus {
        id: EntityId(1),
        name: "B1".to_string(),
        operational_start_date: bucket_seed_date(2024, 1, 1),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 500.0,
        }],
        excess_cost: 0.0,
    };
    SystemBuilder::new()
        .buses(vec![bus])
        .hydros(hydros)
        .stages(stages)
        .initial_conditions(InitialConditions {
            past_defluences,
            ..InitialConditions::default()
        })
        .build()
        .expect("valid system")
}

/// `start_0 = 2024-01-01`. A single `past_defluences` window ending
/// `start_0_minus_hours` before `start_0` and spanning `width_hours`, at rate
/// `value` m³/s.
fn bucket_seed_defluence_window(
    hydro_id: i32,
    start_0_minus_hours: f64,
    width_hours: f64,
    value: f64,
) -> HydroPastDefluence {
    let start_0 = bucket_seed_date(2024, 1, 1);
    let end_date = start_0 - Duration::hours(start_0_minus_hours as i64);
    let start_date = end_date - Duration::hours(width_hours as i64);
    HydroPastDefluence {
        hydro_id: EntityId(hydro_id),
        start_date,
        end_date,
        value_m3s: value,
    }
}

/// Single arc, `k = [1/2, 1/2]`, one window `[start_0 − 24h, start_0)` at
/// 100 m³/s ⇒ `b_1 = k_1 · D = 1/2 · D` (`D` the width-scaled volume,
/// mirroring how an in-study release is already volume-scaled by `τ`).
#[test]
fn test_single_arc_unroll_matches_ac1() {
    let downstream = bucket_seed_hydro(1, None, None);
    let upstream = bucket_seed_hydro(2, Some(1), Some(24.0));
    let system = bucket_seed_build_system(
        vec![downstream, upstream],
        bucket_seed_study_stages(4, 12.0),
        vec![bucket_seed_defluence_window(2, 0.0, 24.0, 100.0)],
    );

    let calendar = DeliveryCalendar::from_system(&system);
    let topology = crate::bucket_topology::build_transit_bucket_topology(&system, &calendar, false);
    let state = crate::test_support::bucket_seed_state(&system, &topology);
    assert_eq!(
        state
            .transit_bucket_plants()
            .map(|(_, r)| r.len())
            .collect::<Vec<_>>(),
        vec![2],
        "sanity: 2-bucket depth"
    );

    let seed = super::build_initial_transit_bucket_state(&system, &topology, &state);
    assert_eq!(seed.len(), topology.n_buckets());

    let volume = 24.0 * M3S_TO_HM3 * 100.0;
    assert!(
        (seed[0] - 0.5 * volume).abs() < 1e-9,
        "b_1 must equal 1/2 * volume, got {} vs expected {}",
        seed[0],
        0.5 * volume
    );
}

/// A mid-horizon upstream entrant (`entry_stage_id`
/// mid-study) supplies a zero-valued `past_defluences` window -- the
/// physically correct value, since the plant did not exist pre-study --
/// and every stage-0 bucket the arc feeds comes out zero.
/// [`super::build_initial_transit_bucket_state`] never reads `entry_stage_id`;
/// conservation is forced by the input data, not a code branch.
#[test]
fn test_mid_horizon_entrant_zero_history_zero_seeds_stage_0_transit_buckets() {
    let downstream = bucket_seed_hydro(1, None, None);
    let mut upstream = bucket_seed_hydro(2, Some(1), Some(24.0));
    upstream.entry_stage_id = Some(2);
    let system = bucket_seed_build_system(
        vec![downstream, upstream],
        bucket_seed_study_stages(4, 12.0),
        vec![bucket_seed_defluence_window(2, 0.0, 24.0, 0.0)],
    );

    let calendar = DeliveryCalendar::from_system(&system);
    let topology = crate::bucket_topology::build_transit_bucket_topology(&system, &calendar, false);
    let state = crate::test_support::bucket_seed_state(&system, &topology);
    assert_eq!(
        state
            .transit_bucket_plants()
            .map(|(_, r)| r.len())
            .collect::<Vec<_>>(),
        vec![2],
        "sanity: 2-bucket depth"
    );

    let seed = super::build_initial_transit_bucket_state(&system, &topology, &state);

    assert!(
        seed.iter().all(|&v| v.abs() < 1e-9),
        "a mid-horizon entrant's zero-valued pre-study history must zero-seed \
         every stage-0 bucket, got {seed:?}"
    );
}

/// Confluence: two upstreams with different `t_v` feeding one downstream
/// plant sum their unrolled shares into the SAME per-plant bucket block.
#[test]
fn test_confluence_aggregates_two_upstreams_into_shared_transit_buckets() {
    let downstream = bucket_seed_hydro(1, None, None);
    let upstream_a = bucket_seed_hydro(2, Some(1), Some(24.0));
    let upstream_b = bucket_seed_hydro(3, Some(1), Some(12.0));
    let system = bucket_seed_build_system(
        vec![downstream, upstream_a, upstream_b],
        bucket_seed_study_stages(4, 12.0),
        vec![
            bucket_seed_defluence_window(2, 0.0, 24.0, 100.0),
            bucket_seed_defluence_window(3, 0.0, 24.0, 50.0),
        ],
    );

    let calendar = DeliveryCalendar::from_system(&system);
    let topology = crate::bucket_topology::build_transit_bucket_topology(&system, &calendar, false);
    let state = crate::test_support::bucket_seed_state(&system, &topology);
    assert_eq!(
        state
            .transit_bucket_plants()
            .map(|(_, r)| r.len())
            .collect::<Vec<_>>(),
        vec![2],
        "sanity: 2-bucket depth"
    );

    let seed = super::build_initial_transit_bucket_state(&system, &topology, &state);

    let vol_a = 24.0 * M3S_TO_HM3 * 100.0;
    let vol_b = 24.0 * M3S_TO_HM3 * 50.0;
    let expected_b1 = 0.5 * vol_a + 0.5 * vol_b;
    let expected_b2 = 0.5 * vol_a;

    assert!(
        (seed[0] - expected_b1).abs() < 1e-9,
        "b_1 must sum both arcs' shares, got {} vs expected {expected_b1}",
        seed[0]
    );
    assert!(
        (seed[1] - expected_b2).abs() < 1e-9,
        "b_2 must carry only the deeper arc's share, got {} vs expected {expected_b2}",
        seed[1]
    );
}

/// Declaration-order invariance: swapping the hydro input order must not
/// change the seed (canonical sort in `SystemBuilder::build` plus the
/// canonical-index-driven aggregation loop).
#[test]
fn test_seed_is_declaration_order_invariant() {
    let downstream = bucket_seed_hydro(1, None, None);
    let upstream_a = bucket_seed_hydro(2, Some(1), Some(24.0));
    let upstream_b = bucket_seed_hydro(3, Some(1), Some(12.0));
    let defluences = vec![
        bucket_seed_defluence_window(2, 0.0, 24.0, 100.0),
        bucket_seed_defluence_window(3, 0.0, 24.0, 50.0),
    ];

    let system_a = bucket_seed_build_system(
        vec![downstream.clone(), upstream_a.clone(), upstream_b.clone()],
        bucket_seed_study_stages(4, 12.0),
        defluences.clone(),
    );
    let system_b = bucket_seed_build_system(
        vec![upstream_b, upstream_a, downstream],
        bucket_seed_study_stages(4, 12.0),
        defluences,
    );

    let calendar_a = DeliveryCalendar::from_system(&system_a);
    let calendar_b = DeliveryCalendar::from_system(&system_b);
    let topology_a =
        crate::bucket_topology::build_transit_bucket_topology(&system_a, &calendar_a, false);
    let topology_b =
        crate::bucket_topology::build_transit_bucket_topology(&system_b, &calendar_b, false);
    let state_a = crate::test_support::bucket_seed_state(&system_a, &topology_a);
    let state_b = crate::test_support::bucket_seed_state(&system_b, &topology_b);
    let seed_a = super::build_initial_transit_bucket_state(&system_a, &topology_a, &state_a);
    let seed_b = super::build_initial_transit_bucket_state(&system_b, &topology_b, &state_b);

    assert_eq!(
        seed_a, seed_b,
        "seed must be bit-identical across input order"
    );
}

/// `seed.len() == B` for every declared topology, including when no arc
/// is declared at all (`B == 0`).
#[test]
fn test_seed_len_matches_n_buckets() {
    let downstream = bucket_seed_hydro(1, None, None);
    let upstream = bucket_seed_hydro(2, Some(1), Some(24.0));
    let system = bucket_seed_build_system(
        vec![downstream, upstream],
        bucket_seed_study_stages(4, 12.0),
        vec![bucket_seed_defluence_window(2, 0.0, 24.0, 100.0)],
    );
    let calendar = DeliveryCalendar::from_system(&system);
    let topology = crate::bucket_topology::build_transit_bucket_topology(&system, &calendar, false);
    let state = crate::test_support::bucket_seed_state(&system, &topology);
    let seed = super::build_initial_transit_bucket_state(&system, &topology, &state);
    assert_eq!(seed.len(), topology.n_buckets());

    let no_arc_downstream = bucket_seed_hydro(1, None, None);
    let no_arc_system = bucket_seed_build_system(
        vec![no_arc_downstream],
        bucket_seed_study_stages(3, 24.0),
        vec![],
    );
    let no_arc_calendar = DeliveryCalendar::from_system(&no_arc_system);
    let no_arc_topology = crate::bucket_topology::build_transit_bucket_topology(
        &no_arc_system,
        &no_arc_calendar,
        false,
    );
    assert_eq!(no_arc_topology.n_buckets(), 0);
    let no_arc_state = crate::test_support::bucket_seed_state(&no_arc_system, &no_arc_topology);
    let no_arc_seed =
        super::build_initial_transit_bucket_state(&no_arc_system, &no_arc_topology, &no_arc_state);
    assert_eq!(no_arc_seed.len(), 0);
}

/// Two gapped (non-contiguous) windows for the same 72h arc land in
/// DISJOINT bucket pairs: the recent window `[start_0 − 24h, start_0)`
/// arrives at buckets 4-5 (`k = [0, 0, 0, 0, 1/2, 1/2]`), the older
/// window `[start_0 − 72h, start_0 − 48h)` arrives at buckets 0-1
/// (`k = [1/2, 1/2]`) -- a genuine 24h gap (`[start_0 − 48h,
/// start_0 − 24h)`) separates the two release windows. Because the
/// windows land in disjoint buckets, dropping the older one (the bug a
/// `.find()` in place of `.filter()` would introduce) zeroes buckets 0-1
/// and fails the assertion below.
#[test]
fn test_gapped_windows_contribute_additively() {
    let downstream = bucket_seed_hydro(1, None, None);
    let upstream = bucket_seed_hydro(2, Some(1), Some(72.0));
    let system = bucket_seed_build_system(
        vec![downstream, upstream],
        bucket_seed_study_stages(6, 12.0),
        vec![
            bucket_seed_defluence_window(2, 0.0, 24.0, 100.0),
            bucket_seed_defluence_window(2, 48.0, 24.0, 40.0),
        ],
    );

    let delivery_calendar = DeliveryCalendar::from_system(&system);
    let topology =
        crate::bucket_topology::build_transit_bucket_topology(&system, &delivery_calendar, false);
    let state = crate::test_support::bucket_seed_state(&system, &topology);
    let seed = super::build_initial_transit_bucket_state(&system, &topology, &state);

    let vol_recent = 24.0 * M3S_TO_HM3 * 100.0;
    let vol_older = 24.0 * M3S_TO_HM3 * 40.0;
    let study_stages: Vec<Stage> = system
        .stages()
        .iter()
        .filter(|s| s.id >= 0)
        .cloned()
        .collect();
    let calendar = StageCalendar::new(&study_stages);
    let k_recent = calendar.hour_window_shares(72.0, 0.0, 24.0);
    let k_older = calendar.hour_window_shares(72.0, 48.0, 24.0);

    let mut expected = vec![0.0_f64; topology.n_buckets()];
    for (d, &k_val) in k_recent.iter().enumerate() {
        expected[d] += k_val * vol_recent;
    }
    for (d, &k_val) in k_older.iter().enumerate() {
        expected[d] += k_val * vol_older;
    }

    assert_eq!(seed.len(), expected.len());
    for (idx, (&got, &want)) in seed.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-9,
            "bucket {idx}: gapped windows must contribute additively, got {got} vs expected {want}"
        );
    }
}

// ── study_horizon_end ───────────────────────────────────────────────────────

/// Given a system whose stages are `[-2, -1, 0, 1, 2]` (pre-study stages carry
/// negative ids) with stage `2` ending `2031-12-01`, `study_horizon_end`
/// returns `Some(2031-12-01)`, ignoring the pre-study stages entirely.
#[test]
fn study_horizon_end_ignores_pre_study_stages() {
    let day = |y: i32, m: u32, d: u32| NaiveDate::from_ymd_opt(y, m, d).unwrap();
    let stage = |id: i32, start: NaiveDate, end: NaiveDate| Stage {
        index: 0,
        id,
        start_date: start,
        end_date: end,
        season_id: None,
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
            branching_factor: 1,
            noise_method: NoiseMethod::Saa,
        },
    };
    let stages = vec![
        stage(-2, day(2030, 9, 1), day(2030, 10, 1)),
        stage(-1, day(2030, 10, 1), day(2030, 11, 1)),
        stage(0, day(2030, 11, 1), day(2030, 12, 1)),
        stage(1, day(2030, 12, 1), day(2031, 1, 1)),
        stage(2, day(2031, 11, 1), day(2031, 12, 1)),
    ];

    let system = SystemBuilder::new()
        .stages(stages)
        .build()
        .expect("a stages-only system with no cross-referenced entities is valid");

    assert_eq!(study_horizon_end(&system), Some(day(2031, 12, 1)));
}

// ── Terminal boundary flag: S1 vs S2/S3 formula agreement ───────────────────

/// The bake's active-cut-count classification (S1, `training/session/mod.rs`)
/// and the per-solve warm-start-count read (S2/S3, the forward pass and the
/// simulation pipeline) must classify the terminal pool identically, on a
/// plain and on an injected terminal pool, on a chain and on a terminal fan.
#[test]
fn terminal_boundary_flag_formulas_agree_on_chain_and_terminal_fan() {
    fn boundary_record(state_dimension: usize) -> cobre_io::OwnedPolicyCutRecord {
        cobre_io::OwnedPolicyCutRecord {
            cut_id: 0,
            slot_index: 0,
            coefficients: vec![0.0; state_dimension],
            intercept: 100.0,
            is_active: true,
            iteration: 0,
            forward_pass_index: 0,
        }
    }

    fn check(setup: &StudySetup, min_terminal_nodes: usize, injected: bool) {
        let node_graph = &setup.inputs.node_graph;
        let num_stages = setup.training_ctx().horizon.num_stages();
        let last = super::node_graph::StageIdx(num_stages - 1);
        let terminal_pool_id = node_graph
            .terminal_pool(num_stages)
            .expect("terminal stage carries an alive node");

        let terminal_nodes: Vec<_> = node_graph
            .nodes
            .iter()
            .filter(|n| n.stage == last)
            .collect();
        assert!(
            terminal_nodes.len() >= min_terminal_nodes,
            "expected at least {min_terminal_nodes} terminal-stage node(s), got {}",
            terminal_nodes.len()
        );
        assert!(
            terminal_nodes.iter().all(|n| n.pool_id == terminal_pool_id),
            "every terminal-stage node must share the highest-id pool"
        );

        let s1_pool = (0..node_graph.n_pools)
            .rfind(|&p| node_graph.pool_stage[p] == last)
            .expect("a terminal stage owns at least one pool");
        let s1 = setup.fcf.pools[s1_pool].active_count() > 0;

        let s2s3_node = node_graph
            .any_stage_node(last)
            .expect("terminal stage carries an alive node");
        let s2s3_pool = node_graph.nodes[s2s3_node].pool_id;
        let s2s3 = setup.fcf.pools[s2s3_pool].has_warm_start_cuts();

        assert_eq!(s1, s2s3, "the S1 and S2/S3 formulas must agree");
        assert_eq!(
            s1, injected,
            "both formulas must equal whether a boundary record was injected"
        );
    }

    let chain_plain = test_support::oracle_chain_setup(1);
    check(&chain_plain, 1, false);

    let mut chain_injected = test_support::oracle_chain_setup(1);
    let record = boundary_record(chain_injected.fcf.state_dimension);
    crate::inject_boundary_cuts(
        &mut chain_injected,
        &crate::ValidatedBoundaryCuts::from_broadcast_records(vec![record]),
    )
    .unwrap();
    check(&chain_injected, 1, true);

    let fan_plain = test_support::terminal_generated_fan_setup(2, 1);
    check(&fan_plain, 2, false);

    let mut fan_injected = test_support::terminal_generated_fan_setup(2, 1);
    let record = boundary_record(fan_injected.fcf.state_dimension);
    crate::inject_boundary_cuts(
        &mut fan_injected,
        &crate::ValidatedBoundaryCuts::from_broadcast_records(vec![record]),
    )
    .unwrap();
    check(&fan_injected, 2, true);
}
