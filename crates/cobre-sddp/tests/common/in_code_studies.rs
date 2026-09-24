//! In-code `System`/`Config` fixtures for the template snapshot manifest
//! (`tests/template_snapshot.rs`): each isolates a stage-LP builder axis no
//! committed deck combines.

#![allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]

use chrono::NaiveDate;
use cobre_core::entities::hydro::{HydroGenerationModel, HydroPenalties};
use cobre_core::scenario::{InflowModel, LoadModel};
use cobre_core::temporal::{
    Block, BlockMode, NoiseMethod, PolicyGraphType, ScenarioSourceConfig, Stage, StageRiskConfig,
    StageStateConfig,
};
use cobre_core::{
    AnticipatedCommitmentHistory, AnticipatedConfig, BoundsCountsSpec, BoundsDefaults,
    BusStagePenalties, ContractBlockBounds, DeficitSegment, EntityId, HorizonGraph,
    HydroBlockBounds, HydroStageBounds, HydroStorage, InitialConditions, LineBlockBounds,
    LineStagePenalties, NcsStagePenalties, PenaltiesCountsSpec, PenaltiesDefaults,
    PumpingBlockBounds, ResolvedBounds, ResolvedPenalties, SystemBuilder, ThermalBlockBounds,
    ThermalStageBounds,
};
use cobre_io::config::{
    Config, EstimationConfig, ExportsConfig, InflowNonNegativityConfig,
    InflowNonNegativityMethod as CfgInflowMethod, ModelingConfig, PolicyConfig, RowSelectionConfig,
    SimulationConfig as IoSimulationConfig, StoppingRuleConfig, TrainingConfig, TrainingSelection,
    TrainingSolverConfig, UpperBoundEvaluationConfig,
};
use cobre_sddp::hydro_models::{
    EvaporationModel, EvaporationModelSet, LinearizedEvaporation, PrepareHydroModelsResult,
};

use super::builders::{
    BusSpec, HydroSpec, StageSpec, ThermalSpec, make_bus, make_hydro, make_stage, make_thermal,
};

const N_STAGES: usize = 4;
const LEAD_STAGES: u32 = 2;
const BUS_ID: EntityId = EntityId(1);
const HYDRO_ID: EntityId = EntityId(2);
const THERMAL_ID: EntityId = EntityId(3);

fn stage_date(index: usize) -> NaiveDate {
    NaiveDate::from_ymd_opt(2024, 1 + index as u32, 1).expect("stage_date: valid calendar month")
}

fn hydro_penalties() -> HydroPenalties {
    HydroPenalties {
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
    }
}

// Rationale: the entity/bounds/penalties construction is one sequential
// fixture; splitting it into helper fns would fragment the declared shape
// across call sites with no reuse benefit.
#[allow(clippy::too_many_lines)]
fn build_system() -> cobre_core::System {
    let bus = make_bus(
        BUS_ID,
        BusSpec {
            name: "B1".to_string(),
            operational_start_date: stage_date(0),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 500.0,
            }],
            excess_cost: 0.0,
        },
    );

    let hydro = make_hydro(
        HYDRO_ID,
        HydroSpec {
            name: "H1".to_string(),
            operational_start_date: stage_date(0),
            bus_id: BUS_ID,
            min_storage_hm3: 0.0,
            max_storage_hm3: 200.0,
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 100.0,
            min_generation_mw: 0.0,
            max_generation_mw: 250.0,
            generation_model: HydroGenerationModel::ConstantProductivity,
            penalties: hydro_penalties(),
            ..Default::default()
        },
    );

    let thermal = make_thermal(
        THERMAL_ID,
        ThermalSpec {
            name: "T_ant".to_string(),
            operational_start_date: stage_date(0),
            bus_id: BUS_ID,
            min_generation_mw: 0.0,
            max_generation_mw: 100.0,
            cost_per_mwh: 50.0,
            anticipated_config: Some(AnticipatedConfig::LeadStages(LEAD_STAGES)),
            ..Default::default()
        },
    );

    // 2 blocks of 360 h each (total 720 h/stage), mirroring
    // `build_hydro_one_ant_system`'s NPV-tractable calendar.
    let blocks = vec![
        Block {
            index: 0,
            name: "BLK0".to_string(),
            duration_hours: 360.0,
        },
        Block {
            index: 1,
            name: "BLK1".to_string(),
            duration_hours: 360.0,
        },
    ];

    let stages: Vec<Stage> = (0..N_STAGES)
        .map(|i| {
            make_stage(
                i,
                StageSpec {
                    start_date: stage_date(i),
                    end_date: stage_date(i + 1),
                    season_id: None,
                    blocks: blocks.clone(),
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
                },
            )
        })
        .collect();

    let inflow_models: Vec<InflowModel> = (0..N_STAGES)
        .map(|i| InflowModel {
            hydro_id: HYDRO_ID,
            stage_id: i as i32,
            mean_m3s: 80.0,
            std_m3s: 0.0,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
            annual: None,
        })
        .collect();

    let load_models: Vec<LoadModel> = (0..N_STAGES)
        .map(|i| LoadModel {
            bus_id: BUS_ID,
            stage_id: i as i32,
            mean_mw: 100.0,
            std_mw: 0.0,
        })
        .collect();

    let k_max = LEAD_STAGES as usize;
    let thermal_axis = N_STAGES + k_max;
    let mut bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 1,
            n_thermals: 1,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages: N_STAGES,
            k_max,
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
    // The padding region [n_stages, n_stages + k) is the delivery-stage axis
    // `fill_anticipated_columns` reads; it must carry the thermal's own cost
    // and capacity so the decision column's objective coefficient is non-zero.
    for s in 0..thermal_axis {
        *bounds.thermal_bounds_mut(0, s) = ThermalStageBounds { cost_per_mwh: 50.0 };
        *bounds.thermal_block_base_mut(0, s) = ThermalBlockBounds {
            min_generation_mw: 0.0,
            max_generation_mw: 100.0,
        };
    }

    let penalties = ResolvedPenalties::new(
        &PenaltiesCountsSpec {
            n_hydros: 1,
            n_buses: 1,
            n_lines: 0,
            n_ncs: 0,
            n_stages: N_STAGES,
        },
        &PenaltiesDefaults {
            hydro: hydro_penalties(),
            bus: BusStagePenalties { excess_cost: 0.0 },
            line: LineStagePenalties { exchange_cost: 0.0 },
            ncs: NcsStagePenalties {
                curtailment_cost: 0.0,
            },
        },
    );

    // Zero seeds: the K=2 ring's two pre-study deliveries (stage 0 and stage 1)
    // are decided before the study, at zero MW, mirroring the K=2 reconciliation
    // fixture in `tests/anticipated_core.rs`.
    let past_anticipated_commitments = (0..k_max)
        .map(|i| AnticipatedCommitmentHistory {
            thermal_id: THERMAL_ID,
            start_date: stage_date(i),
            end_date: stage_date(i + 1),
            value_mw: 0.0,
        })
        .collect();

    let initial_conditions = InitialConditions {
        storage: vec![HydroStorage {
            hydro_id: HYDRO_ID,
            value_hm3: 100.0,
        }],
        filling_storage: vec![],
        past_anticipated_commitments,
        recent_observations: vec![],
        past_defluences: vec![],
    };

    let policy_graph = HorizonGraph {
        stage_discount_rate_overrides: std::collections::BTreeMap::new(),
        graph_type: PolicyGraphType::FiniteHorizon,
        annual_discount_rate: 0.06,
        transitions: vec![],
        nodes: Vec::new(),
        season_map: None,
    };

    SystemBuilder::new()
        .buses(vec![bus])
        .hydros(vec![hydro])
        .thermals(vec![thermal])
        .stages(stages)
        .inflow_models(inflow_models)
        .load_models(load_models)
        .bounds(bounds)
        .penalties(penalties)
        .initial_conditions(initial_conditions)
        .policy_graph(policy_graph)
        .build()
        .expect("discounted_anticipated_study: valid system")
}

fn build_config() -> Config {
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
            tree_seed: Some(42),
            stopping_rules: Some(vec![StoppingRuleConfig::IterationLimit { limit: 1 }]),
            stopping_mode: cobre_io::config::StoppingMode::Any,
            cut_selection: RowSelectionConfig::default(),
            solver: TrainingSolverConfig::default(),
            parallelism: cobre_io::config::ParallelismConfig::default(),
            scenario_source: None,
            selection: Some(TrainingSelection::Sampled { forward_passes: 1 }),
        },
        upper_bound_evaluation: UpperBoundEvaluationConfig::default(),
        policy: PolicyConfig::default(),
        simulation: IoSimulationConfig::default(),
        exports: ExportsConfig::default(),
        estimation: EstimationConfig::default(),
    }
}

/// Discounted (6%/yr), 4-stage study with a `LeadStages(2)` anticipated
/// thermal: isolates an anticipated decision priced after stage 0 under a
/// nonzero discount rate, a combination no committed deck exercises.
#[must_use]
pub fn discounted_anticipated_study() -> (cobre_core::System, Config) {
    (build_system(), build_config())
}

const EVAP_N_STAGES: usize = 2;
const EVAP_BUS_ID: EntityId = EntityId(1);
const EVAP_HYDRO_ID: EntityId = EntityId(2);
const EVAP_THERMAL_ID: EntityId = EntityId(3);

fn evap_hydro_penalties() -> HydroPenalties {
    HydroPenalties {
        evaporation_violation_pos_cost: 11.0,
        evaporation_violation_neg_cost: 7.0,
        ..hydro_penalties()
    }
}

// Rationale: the entity/bounds/penalties construction is one sequential
// fixture; splitting it into helper fns would fragment the declared shape
// across call sites with no reuse benefit.
#[allow(clippy::too_many_lines)]
fn build_parallel_evap_system() -> cobre_core::System {
    let bus = make_bus(
        EVAP_BUS_ID,
        BusSpec {
            name: "B1".to_string(),
            operational_start_date: stage_date(0),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 500.0,
            }],
            excess_cost: 0.0,
        },
    );

    let hydro = make_hydro(
        EVAP_HYDRO_ID,
        HydroSpec {
            name: "H1".to_string(),
            operational_start_date: stage_date(0),
            bus_id: EVAP_BUS_ID,
            min_storage_hm3: 0.0,
            max_storage_hm3: 200.0,
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 100.0,
            min_generation_mw: 0.0,
            max_generation_mw: 250.0,
            generation_model: HydroGenerationModel::ConstantProductivity,
            penalties: evap_hydro_penalties(),
            ..Default::default()
        },
    );

    let thermal = make_thermal(
        EVAP_THERMAL_ID,
        ThermalSpec {
            name: "T1".to_string(),
            operational_start_date: stage_date(0),
            bus_id: EVAP_BUS_ID,
            min_generation_mw: 0.0,
            max_generation_mw: 100.0,
            cost_per_mwh: 50.0,
            ..Default::default()
        },
    );

    // 3 blocks of 200h/244h/300h (744 h/stage total): three distinct block
    // durations, so a slack priced at one block's hours is distinguishable
    // from one priced at the stage total.
    let blocks = vec![
        Block {
            index: 0,
            name: "BLK0".to_string(),
            duration_hours: 200.0,
        },
        Block {
            index: 1,
            name: "BLK1".to_string(),
            duration_hours: 244.0,
        },
        Block {
            index: 2,
            name: "BLK2".to_string(),
            duration_hours: 300.0,
        },
    ];

    let stages: Vec<Stage> = (0..EVAP_N_STAGES)
        .map(|i| {
            make_stage(
                i,
                StageSpec {
                    start_date: stage_date(i),
                    end_date: stage_date(i + 1),
                    season_id: None,
                    blocks: blocks.clone(),
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
                },
            )
        })
        .collect();

    let inflow_models: Vec<InflowModel> = (0..EVAP_N_STAGES)
        .map(|i| InflowModel {
            hydro_id: EVAP_HYDRO_ID,
            stage_id: i as i32,
            mean_m3s: 80.0,
            std_m3s: 0.0,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
            annual: None,
        })
        .collect();

    let load_models: Vec<LoadModel> = (0..EVAP_N_STAGES)
        .map(|i| LoadModel {
            bus_id: EVAP_BUS_ID,
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
            n_stages: EVAP_N_STAGES,
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

    let penalties = ResolvedPenalties::new(
        &PenaltiesCountsSpec {
            n_hydros: 1,
            n_buses: 1,
            n_lines: 0,
            n_ncs: 0,
            n_stages: EVAP_N_STAGES,
        },
        &PenaltiesDefaults {
            hydro: evap_hydro_penalties(),
            bus: BusStagePenalties { excess_cost: 0.0 },
            line: LineStagePenalties { exchange_cost: 0.0 },
            ncs: NcsStagePenalties {
                curtailment_cost: 0.0,
            },
        },
    );

    let initial_conditions = InitialConditions {
        storage: vec![HydroStorage {
            hydro_id: EVAP_HYDRO_ID,
            value_hm3: 100.0,
        }],
        filling_storage: vec![],
        past_anticipated_commitments: vec![],
        recent_observations: vec![],
        past_defluences: vec![],
    };

    let policy_graph = HorizonGraph {
        stage_discount_rate_overrides: std::collections::BTreeMap::new(),
        graph_type: PolicyGraphType::FiniteHorizon,
        annual_discount_rate: 0.0,
        transitions: vec![],
        nodes: Vec::new(),
        season_map: None,
    };

    SystemBuilder::new()
        .buses(vec![bus])
        .hydros(vec![hydro])
        .thermals(vec![thermal])
        .stages(stages)
        .inflow_models(inflow_models)
        .load_models(load_models)
        .bounds(bounds)
        .penalties(penalties)
        .initial_conditions(initial_conditions)
        .policy_graph(policy_graph)
        .build()
        .expect("parallel_multiblock_evaporation_study: valid system")
}

fn parallel_evap_hydro_models(system: &cobre_core::System) -> PrepareHydroModelsResult {
    let mut hydro_models = PrepareHydroModelsResult::default_from_system(system);
    hydro_models.evaporation = EvaporationModelSet::new(vec![EvaporationModel::Linearized {
        coefficients: vec![
            LinearizedEvaporation {
                intercept_m3s: 1.0,
                volume_slope_m3s_per_hm3: 0.01,
            },
            LinearizedEvaporation {
                intercept_m3s: 1.0,
                volume_slope_m3s_per_hm3: 0.01,
            },
        ],
        reference_volumes_hm3: vec![100.0, 100.0],
    }]);
    hydro_models
}

/// Parallel, 2-stage, 3-block-per-stage study with an active linearized
/// evaporation model on its one hydro: isolates the parallel multi-block
/// evaporation slot (R7), a combination no committed deck exercises. The
/// three distinct block durations (200, 244, 300 h) make a slack priced at
/// one block's hours distinguishable from one priced at the stage's 744 h.
#[must_use]
pub fn parallel_multiblock_evaporation_study()
-> (cobre_core::System, Config, PrepareHydroModelsResult) {
    let system = build_parallel_evap_system();
    let hydro_models = parallel_evap_hydro_models(&system);
    (system, build_config(), hydro_models)
}
