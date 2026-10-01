//! In-code `System`/`Config` fixtures for the stage-LP builder test suite.
//! `discounted_anticipated_study` and `parallel_multiblock_evaporation_study`
//! each isolate a stage-LP builder axis no committed deck combines.
//! `stochastic_parallel_study` backs the one-hot patch-ownership sweep
//! (`tests/patch_ownership_sweep.rs`), supplying the stochastic load and NCS
//! noise no committed deck exercises.
//! `chronological_noise_study` backs the chronological inflow-noise-ownership
//! test in `tests/chronological_inflow_noise.rs` and the `(Chronological,
//! inflow)` cell of the patch-ownership sweep.
//! `mixed_lead_anticipated_study` exercises two anticipated thermals with
//! different lead depths on one study, a combination no committed deck
//! combines.
//! `chronological_noise_study`'s `pumping_station` field adds a pumping
//! station whose source hydro sits at a nonzero canonical position, a
//! combination no committed deck combines.
//! `discounted_delivery_oracle_study` backs `discounted_delivery_closed_form_lb`
//! in `tests/anticipated_core.rs`.

#![allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]

use chrono::{NaiveDate, TimeDelta};
use cobre_core::entities::hydro::{HydroGenerationModel, HydroPenalties};
use cobre_core::scenario::{InflowModel, LoadModel, NcsModel};
use cobre_core::temporal::{
    Block, BlockMode, NoiseMethod, PolicyGraphType, ScenarioSourceConfig, Stage, StageRiskConfig,
    StageStateConfig,
};
use cobre_core::{
    AnticipatedCommitmentHistory, AnticipatedConfig, BoundsCountsSpec, BoundsDefaults,
    BusStagePenalties, ContractBlockBounds, DeficitSegment, EntityId, HorizonGraph,
    HydroBlockBounds, HydroStageBounds, HydroStorage, InitialConditions, LineBlockBounds,
    LineStagePenalties, NcsStagePenalties, NonControllableSource, PenaltiesCountsSpec,
    PenaltiesDefaults, PostStudyStage, PostStudyStages, PostStudyThermalBound, PumpingBlockBounds,
    PumpingStation, ResolvedBounds, ResolvedPenalties, SystemBuilder, ThermalBlockBounds,
    ThermalStageBounds,
};
use cobre_io::config::{
    Config, EstimationConfig, ExportsConfig, InflowNonNegativityConfig,
    InflowNonNegativityMethod as CfgInflowMethod, ModelingConfig, PolicyConfig, RowSelectionConfig,
    SimulationConfig as IoSimulationConfig, StoppingRuleConfig, TrainingConfig, TrainingSelection,
    TrainingSolverConfig, UpperBoundEvaluationConfig,
};
use cobre_sddp::StudySetup;
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
    // The loop fills the whole thermal axis [0, n_stages + k), so the study's
    // resolved bounds are total. The decision columns read the study stages'
    // cost and capacity; a delivery past the horizon is priced from the
    // post-study calendar, not from this axis.
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

fn build_config_with_inflow_penalty() -> Config {
    let mut config = build_config();
    config.modeling.inflow_non_negativity.method = CfgInflowMethod::Penalty;
    config
}

/// [`discounted_anticipated_study`]'s parallel, two-block-per-stage system
/// with `InflowNonNegativityMethod::Penalty` active: the smallest in-code
/// study whose inflow-slack column is live on a multi-block parallel
/// stage — no committed deck combines a `Penalty`/`TruncationWithPenalty`
/// inflow method with more than one parallel block.
#[must_use]
pub fn parallel_inflow_slack_study() -> (cobre_core::System, Config) {
    (build_system(), build_config_with_inflow_penalty())
}

const MIXED_LEAD_N_STAGES: usize = 5;
const MIXED_LEAD_BUS_ID: EntityId = EntityId(1);
const MIXED_LEAD_HYDRO_ID: EntityId = EntityId(2);
const MIXED_LEAD_SHORT_THERMAL_ID: EntityId = EntityId(10);
const MIXED_LEAD_LONG_THERMAL_ID: EntityId = EntityId(20);
const MIXED_LEAD_SHORT_LEAD: u32 = 1;
const MIXED_LEAD_LONG_LEAD: u32 = 3;
/// Actual calendar hours of 2025's Jan-May, one block per stage.
const MIXED_LEAD_MONTH_HOURS: [f64; MIXED_LEAD_N_STAGES] = [744.0, 672.0, 744.0, 720.0, 744.0];
/// Actual calendar hours of 2025's Jun-Aug, one per post-study stage.
const MIXED_LEAD_POST_STUDY_MONTH_HOURS: [f64; 3] = [720.0, 744.0, 744.0];

fn mixed_lead_stage_date(index: usize) -> NaiveDate {
    NaiveDate::from_ymd_opt(2025, 1 + index as u32, 1)
        .expect("mixed_lead_anticipated_study: valid date")
}

// Rationale: the entity/bounds/penalties construction is one sequential
// fixture; splitting it into helper fns would fragment the declared shape
// across call sites with no reuse benefit.
#[allow(clippy::too_many_lines)]
fn build_mixed_lead_system(reversed: bool) -> cobre_core::System {
    let bus = make_bus(
        MIXED_LEAD_BUS_ID,
        BusSpec {
            name: "B1".to_string(),
            operational_start_date: mixed_lead_stage_date(0),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 500.0,
            }],
            excess_cost: 0.0,
        },
    );

    let hydro = make_hydro(
        MIXED_LEAD_HYDRO_ID,
        HydroSpec {
            name: "H1".to_string(),
            operational_start_date: mixed_lead_stage_date(0),
            bus_id: MIXED_LEAD_BUS_ID,
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

    let thermal_short = make_thermal(
        MIXED_LEAD_SHORT_THERMAL_ID,
        ThermalSpec {
            name: "T_short".to_string(),
            operational_start_date: mixed_lead_stage_date(0),
            bus_id: MIXED_LEAD_BUS_ID,
            min_generation_mw: 0.0,
            max_generation_mw: 100.0,
            cost_per_mwh: 50.0,
            anticipated_config: Some(AnticipatedConfig::LeadStages(MIXED_LEAD_SHORT_LEAD)),
            ..Default::default()
        },
    );
    let thermal_long = make_thermal(
        MIXED_LEAD_LONG_THERMAL_ID,
        ThermalSpec {
            name: "T_long".to_string(),
            operational_start_date: mixed_lead_stage_date(0),
            bus_id: MIXED_LEAD_BUS_ID,
            min_generation_mw: 0.0,
            max_generation_mw: 100.0,
            cost_per_mwh: 50.0,
            anticipated_config: Some(AnticipatedConfig::LeadStages(MIXED_LEAD_LONG_LEAD)),
            ..Default::default()
        },
    );

    let stages: Vec<Stage> = (0..MIXED_LEAD_N_STAGES)
        .map(|i| {
            make_stage(
                i,
                StageSpec {
                    start_date: mixed_lead_stage_date(i),
                    end_date: mixed_lead_stage_date(i + 1),
                    season_id: None,
                    blocks: vec![Block {
                        index: 0,
                        name: "BLK0".to_string(),
                        duration_hours: MIXED_LEAD_MONTH_HOURS[i],
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
                },
            )
        })
        .collect();

    let inflow_models: Vec<InflowModel> = (0..MIXED_LEAD_N_STAGES)
        .map(|i| InflowModel {
            hydro_id: MIXED_LEAD_HYDRO_ID,
            stage_id: i as i32,
            mean_m3s: 80.0,
            std_m3s: 10.0,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
            annual: None,
        })
        .collect();

    let load_models: Vec<LoadModel> = (0..MIXED_LEAD_N_STAGES)
        .map(|i| LoadModel {
            bus_id: MIXED_LEAD_BUS_ID,
            stage_id: i as i32,
            mean_mw: 100.0,
            std_mw: 0.0,
        })
        .collect();

    let k_max = MIXED_LEAD_LONG_LEAD as usize;
    let thermal_axis = MIXED_LEAD_N_STAGES + k_max;
    let mut bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 1,
            n_thermals: 2,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages: MIXED_LEAD_N_STAGES,
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
    // The loop fills the whole thermal axis [0, n_stages + k_max) for both
    // thermals regardless of their own (shallower) lead, so the study's
    // resolved bounds are total. The decision columns read the study stages'
    // cost and capacity; a delivery past the horizon is priced from the
    // post-study calendar below, not from this axis.
    for thermal_idx in 0..2 {
        for s in 0..thermal_axis {
            *bounds.thermal_bounds_mut(thermal_idx, s) = ThermalStageBounds { cost_per_mwh: 50.0 };
            *bounds.thermal_block_base_mut(thermal_idx, s) = ThermalBlockBounds {
                min_generation_mw: 0.0,
                max_generation_mw: 100.0,
            };
        }
    }

    let penalties = ResolvedPenalties::new(
        &PenaltiesCountsSpec {
            n_hydros: 1,
            n_buses: 1,
            n_lines: 0,
            n_ncs: 0,
            n_stages: MIXED_LEAD_N_STAGES,
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

    // Each thermal's own pre-study commitment history, built the way
    // `build_system` builds its single thermal's: one zero-MW window per
    // pre-study stage its own lead decides.
    let past_anticipated_commitments = [
        (MIXED_LEAD_SHORT_THERMAL_ID, MIXED_LEAD_SHORT_LEAD),
        (MIXED_LEAD_LONG_THERMAL_ID, MIXED_LEAD_LONG_LEAD),
    ]
    .into_iter()
    .flat_map(|(thermal_id, lead)| {
        (0..lead as usize).map(move |i| AnticipatedCommitmentHistory {
            thermal_id,
            start_date: mixed_lead_stage_date(i),
            end_date: mixed_lead_stage_date(i + 1),
            value_mw: 0.0,
        })
    })
    .collect();

    let initial_conditions = InitialConditions {
        storage: vec![HydroStorage {
            hydro_id: MIXED_LEAD_HYDRO_ID,
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

    // Post-study calendar covering the long lead's post-horizon deliveries
    // (decision stages 2-4 deliver at stages 5-7, i.e. post-study indices
    // 0-2) and the short lead's own single one (decision stage 4 delivers
    // at stage 5, post-study index 0).
    let post_study_stages: Vec<PostStudyStage> = (MIXED_LEAD_N_STAGES..thermal_axis)
        .map(|i| PostStudyStage {
            start_date: mixed_lead_stage_date(i),
            duration_hours: MIXED_LEAD_POST_STUDY_MONTH_HOURS[i - MIXED_LEAD_N_STAGES],
        })
        .collect();
    let post_study = PostStudyStages {
        stages: post_study_stages,
        thermal_bounds: vec![
            PostStudyThermalBound {
                thermal_id: MIXED_LEAD_SHORT_THERMAL_ID,
                post_study_stage_index: 0,
                cost_per_mwh: 50.0,
                min_mw: 0.0,
                max_mw: 100.0,
            },
            PostStudyThermalBound {
                thermal_id: MIXED_LEAD_LONG_THERMAL_ID,
                post_study_stage_index: 0,
                cost_per_mwh: 50.0,
                min_mw: 0.0,
                max_mw: 100.0,
            },
            PostStudyThermalBound {
                thermal_id: MIXED_LEAD_LONG_THERMAL_ID,
                post_study_stage_index: 1,
                cost_per_mwh: 50.0,
                min_mw: 0.0,
                max_mw: 100.0,
            },
            PostStudyThermalBound {
                thermal_id: MIXED_LEAD_LONG_THERMAL_ID,
                post_study_stage_index: 2,
                cost_per_mwh: 50.0,
                min_mw: 0.0,
                max_mw: 100.0,
            },
        ],
    };

    let mut buses = vec![bus];
    let mut hydros = vec![hydro];
    let mut thermals = vec![thermal_short, thermal_long];
    if reversed {
        buses.reverse();
        hydros.reverse();
        thermals.reverse();
    }

    SystemBuilder::new()
        .buses(buses)
        .hydros(hydros)
        .thermals(thermals)
        .stages(stages)
        .inflow_models(inflow_models)
        .load_models(load_models)
        .bounds(bounds)
        .penalties(penalties)
        .initial_conditions(initial_conditions)
        .policy_graph(policy_graph)
        .post_study_stages(Some(post_study))
        .build()
        .expect("mixed_lead_anticipated_study: valid system")
}

/// 5 monthly stages (2025-01 to 2025-05, one block each) discounted at 6%/yr,
/// with two anticipated thermals of different lead depths (`LeadStages(1)`
/// and `LeadStages(3)`) over a declared 3-month post-study calendar: the
/// long lead's decisions at stages 2-4 (and the short lead's at stage 4)
/// have no in-study delivery target left and instead target a post-study
/// delivery — a mixed-lead combination no committed deck exercises.
/// `reversed == true` reverses every entity vector before `SystemBuilder::build`.
#[must_use]
pub fn mixed_lead_anticipated_study(reversed: bool) -> (cobre_core::System, Config) {
    (build_mixed_lead_system(reversed), build_config())
}

pub const DELIVERY_ORACLE_ANNUAL_RATE: f64 = 0.06;
pub const DELIVERY_ORACLE_STAGE_DAYS: [i64; 2] = [31, 28];
pub const DELIVERY_ORACLE_STAGE0_HOURS: f64 = 744.0;
pub const DELIVERY_ORACLE_STAGE1_BLOCK_HOURS: [f64; 2] = [400.0, 272.0];
pub const DELIVERY_ORACLE_POST_STUDY_HOURS: f64 = 744.0;
pub const DELIVERY_ORACLE_LOAD_MW: f64 = 50.0;
pub const DELIVERY_ORACLE_ANTICIPATED_CAP_MW: f64 = 40.0;
pub const DELIVERY_ORACLE_POST_STUDY_MIN_MW: f64 = 30.0;
pub const DELIVERY_ORACLE_BACKUP_COST: f64 = 100.0;
pub const DELIVERY_ORACLE_ANTICIPATED_COST: [f64; 2] = [15.0, 10.0];
pub const DELIVERY_ORACLE_POST_STUDY_COST: f64 = 20.0;

const DELIVERY_ORACLE_N_STAGES: usize = DELIVERY_ORACLE_STAGE_DAYS.len();
const DELIVERY_ORACLE_BACKUP_CAP_MW: f64 = 200.0;
const DELIVERY_ORACLE_BUS_ID: EntityId = EntityId(1);
const DELIVERY_ORACLE_ANTICIPATED_ID: EntityId = EntityId(2);
const DELIVERY_ORACLE_BACKUP_ID: EntityId = EntityId(3);

fn delivery_oracle_boundaries() -> [NaiveDate; DELIVERY_ORACLE_N_STAGES + 1] {
    let start = NaiveDate::from_ymd_opt(2025, 1, 1)
        .expect("discounted_delivery_oracle_study: valid start date");
    let mid = start + TimeDelta::days(DELIVERY_ORACLE_STAGE_DAYS[0]);
    [
        start,
        mid,
        mid + TimeDelta::days(DELIVERY_ORACLE_STAGE_DAYS[1]),
    ]
}

fn delivery_oracle_stages() -> Vec<Stage> {
    let boundaries = delivery_oracle_boundaries();
    let block_hours: [&[f64]; DELIVERY_ORACLE_N_STAGES] = [
        &[DELIVERY_ORACLE_STAGE0_HOURS],
        &DELIVERY_ORACLE_STAGE1_BLOCK_HOURS,
    ];
    (0..DELIVERY_ORACLE_N_STAGES)
        .map(|i| {
            make_stage(
                i,
                StageSpec {
                    start_date: boundaries[i],
                    end_date: boundaries[i + 1],
                    season_id: None,
                    blocks: block_hours[i]
                        .iter()
                        .enumerate()
                        .map(|(index, &duration_hours)| Block {
                            index,
                            name: format!("BLK{index}"),
                            duration_hours,
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
                },
            )
        })
        .collect()
}

fn delivery_oracle_bounds() -> ResolvedBounds {
    let backup_capacity = ThermalBlockBounds {
        min_generation_mw: 0.0,
        max_generation_mw: DELIVERY_ORACLE_BACKUP_CAP_MW,
    };
    let mut bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 0,
            n_thermals: 2,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages: DELIVERY_ORACLE_N_STAGES,
            k_max: 1,
        },
        &BoundsDefaults {
            hydro: HydroStageBounds {
                min_storage_hm3: 0.0,
                max_storage_hm3: 0.0,
                filling_min_rate_m3s: 0.0,
                water_withdrawal_m3s: 0.0,
            },
            hydro_block: HydroBlockBounds::default(),
            thermal: ThermalStageBounds {
                cost_per_mwh: DELIVERY_ORACLE_BACKUP_COST,
            },
            thermal_block: backup_capacity,
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
    for axis in 0..=DELIVERY_ORACLE_N_STAGES {
        *bounds.thermal_bounds_mut(0, axis) = ThermalStageBounds {
            cost_per_mwh: DELIVERY_ORACLE_ANTICIPATED_COST[axis.min(DELIVERY_ORACLE_N_STAGES - 1)],
        };
        *bounds.thermal_block_base_mut(0, axis) = ThermalBlockBounds {
            min_generation_mw: 0.0,
            max_generation_mw: DELIVERY_ORACLE_ANTICIPATED_CAP_MW,
        };
    }
    bounds
}

fn delivery_oracle_post_study(study_end: NaiveDate) -> PostStudyStages {
    PostStudyStages {
        stages: vec![PostStudyStage {
            start_date: study_end,
            duration_hours: DELIVERY_ORACLE_POST_STUDY_HOURS,
        }],
        thermal_bounds: vec![PostStudyThermalBound {
            thermal_id: DELIVERY_ORACLE_ANTICIPATED_ID,
            post_study_stage_index: 0,
            cost_per_mwh: DELIVERY_ORACLE_POST_STUDY_COST,
            min_mw: DELIVERY_ORACLE_POST_STUDY_MIN_MW,
            max_mw: DELIVERY_ORACLE_ANTICIPATED_CAP_MW,
        }],
    }
}

fn build_delivery_oracle_system(annual_discount_rate: f64) -> cobre_core::System {
    let boundaries = delivery_oracle_boundaries();
    let bus = make_bus(
        DELIVERY_ORACLE_BUS_ID,
        BusSpec {
            name: "B1".to_string(),
            operational_start_date: boundaries[0],
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 1000.0,
            }],
            excess_cost: 0.0,
        },
    );
    let thermal_anticipated = make_thermal(
        DELIVERY_ORACLE_ANTICIPATED_ID,
        ThermalSpec {
            name: "T_ant".to_string(),
            operational_start_date: boundaries[0],
            bus_id: DELIVERY_ORACLE_BUS_ID,
            min_generation_mw: 0.0,
            max_generation_mw: DELIVERY_ORACLE_ANTICIPATED_CAP_MW,
            cost_per_mwh: DELIVERY_ORACLE_ANTICIPATED_COST[0],
            anticipated_config: Some(AnticipatedConfig::LeadStages(1)),
            ..Default::default()
        },
    );
    let thermal_backup = make_thermal(
        DELIVERY_ORACLE_BACKUP_ID,
        ThermalSpec {
            name: "T_backup".to_string(),
            operational_start_date: boundaries[0],
            bus_id: DELIVERY_ORACLE_BUS_ID,
            min_generation_mw: 0.0,
            max_generation_mw: DELIVERY_ORACLE_BACKUP_CAP_MW,
            cost_per_mwh: DELIVERY_ORACLE_BACKUP_COST,
            ..Default::default()
        },
    );
    let load_models: Vec<LoadModel> = (0..DELIVERY_ORACLE_N_STAGES)
        .map(|i| LoadModel {
            bus_id: DELIVERY_ORACLE_BUS_ID,
            stage_id: i as i32,
            mean_mw: DELIVERY_ORACLE_LOAD_MW,
            std_mw: 0.0,
        })
        .collect();
    let penalties = ResolvedPenalties::new(
        &PenaltiesCountsSpec {
            n_hydros: 0,
            n_buses: 1,
            n_lines: 0,
            n_ncs: 0,
            n_stages: DELIVERY_ORACLE_N_STAGES,
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
    let initial_conditions = InitialConditions {
        storage: vec![],
        filling_storage: vec![],
        past_anticipated_commitments: vec![AnticipatedCommitmentHistory {
            thermal_id: DELIVERY_ORACLE_ANTICIPATED_ID,
            start_date: boundaries[0],
            end_date: boundaries[1],
            value_mw: 0.0,
        }],
        recent_observations: vec![],
        past_defluences: vec![],
    };
    let policy_graph = HorizonGraph {
        stage_discount_rate_overrides: std::collections::BTreeMap::new(),
        graph_type: PolicyGraphType::FiniteHorizon,
        annual_discount_rate,
        transitions: vec![],
        nodes: Vec::new(),
        season_map: None,
    };

    SystemBuilder::new()
        .buses(vec![bus])
        .thermals(vec![thermal_anticipated, thermal_backup])
        .stages(delivery_oracle_stages())
        .load_models(load_models)
        .bounds(delivery_oracle_bounds())
        .penalties(penalties)
        .initial_conditions(initial_conditions)
        .policy_graph(policy_graph)
        .post_study_stages(Some(delivery_oracle_post_study(boundaries[2])))
        .build()
        .expect("discounted_delivery_oracle_study: valid system")
}

/// Two stages (Jan and Feb 2025; the second has two unequal blocks) with a
/// `LeadStages(1)` anticipated thermal delivering at the second stage and one
/// post-study delivery, discounted at `annual_discount_rate`: unequal delivery
/// hours, a positive rate and a post-study delivery in one no-hydro study.
#[must_use]
pub fn discounted_delivery_oracle_study(annual_discount_rate: f64) -> (cobre_core::System, Config) {
    let mut config = build_config();
    config.training.stopping_rules = Some(vec![StoppingRuleConfig::IterationLimit { limit: 3 }]);
    (build_delivery_oracle_system(annual_discount_rate), config)
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
/// evaporation slot, a combination no committed deck exercises. The
/// three distinct block durations (200, 244, 300 h) make a slack priced at
/// one block's hours distinguishable from one priced at the stage's 744 h.
#[must_use]
pub fn parallel_multiblock_evaporation_study()
-> (cobre_core::System, Config, PrepareHydroModelsResult) {
    let system = build_parallel_evap_system();
    let hydro_models = parallel_evap_hydro_models(&system);
    (system, build_config(), hydro_models)
}

const STOCHASTIC_N_STAGES: usize = 2;
const STOCHASTIC_BUS_ID: EntityId = EntityId(1);
const STOCHASTIC_HYDRO_ID: EntityId = EntityId(2);
const STOCHASTIC_NCS_ID: EntityId = EntityId(3);

// Rationale: the entity/bounds/penalties construction is one sequential
// fixture; splitting it into helper fns would fragment the declared shape
// across call sites with no reuse benefit.
#[allow(clippy::too_many_lines)]
fn build_stochastic_parallel_system() -> cobre_core::System {
    let bus = make_bus(
        STOCHASTIC_BUS_ID,
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
        STOCHASTIC_HYDRO_ID,
        HydroSpec {
            name: "H1".to_string(),
            operational_start_date: stage_date(0),
            bus_id: STOCHASTIC_BUS_ID,
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

    let ncs = NonControllableSource {
        id: STOCHASTIC_NCS_ID,
        name: "NCS0".to_string(),
        operational_start_date: stage_date(0),
        bus_id: STOCHASTIC_BUS_ID,
        entry_stage_id: None,
        exit_stage_id: None,
        max_generation_mw: 30.0,
        allow_curtailment: true,
        curtailment_cost: 0.0,
    };

    // Two distinct block durations (300, 444 h) so a slack priced at one
    // block's hours is distinguishable from one priced at the stage total.
    let blocks = vec![
        Block {
            index: 0,
            name: "BLK0".to_string(),
            duration_hours: 300.0,
        },
        Block {
            index: 1,
            name: "BLK1".to_string(),
            duration_hours: 444.0,
        },
    ];

    let stages: Vec<Stage> = (0..STOCHASTIC_N_STAGES)
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

    let inflow_models: Vec<InflowModel> = (0..STOCHASTIC_N_STAGES)
        .map(|i| InflowModel {
            hydro_id: STOCHASTIC_HYDRO_ID,
            stage_id: i as i32,
            mean_m3s: 80.0,
            std_m3s: 20.0,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
            annual: None,
        })
        .collect();

    let load_models: Vec<LoadModel> = (0..STOCHASTIC_N_STAGES)
        .map(|i| LoadModel {
            bus_id: STOCHASTIC_BUS_ID,
            stage_id: i as i32,
            mean_mw: 100.0,
            std_mw: 10.0,
        })
        .collect();

    let ncs_models: Vec<NcsModel> = (0..STOCHASTIC_N_STAGES)
        .map(|i| NcsModel {
            ncs_id: STOCHASTIC_NCS_ID,
            stage_id: i as i32,
            mean: 0.5,
            std: 0.2,
        })
        .collect();

    let bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 1,
            n_thermals: 0,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages: STOCHASTIC_N_STAGES,
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
            n_hydros: 1,
            n_buses: 1,
            n_lines: 0,
            n_ncs: 1,
            n_stages: STOCHASTIC_N_STAGES,
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

    let initial_conditions = InitialConditions {
        storage: vec![HydroStorage {
            hydro_id: STOCHASTIC_HYDRO_ID,
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
        .non_controllable_sources(vec![ncs])
        .stages(stages)
        .inflow_models(inflow_models)
        .load_models(load_models)
        .ncs_models(ncs_models)
        .bounds(bounds)
        .penalties(penalties)
        .initial_conditions(initial_conditions)
        .policy_graph(policy_graph)
        .build()
        .expect("stochastic_parallel_study: valid system")
}

/// Two parallel stages, two blocks each (300 h, 444 h): stochastic inflow
/// noise on its one hydro, stochastic load noise on its one bus, and
/// stochastic availability noise on its one non-controllable source —
/// isolates the load- and NCS-noise patch paths no committed deck exercises.
#[must_use]
pub fn stochastic_parallel_study() -> (cobre_core::System, Config) {
    (build_stochastic_parallel_system(), build_config())
}

/// Block durations (h) of the chronological-noise study's two blocks.
pub const CHRONOLOGICAL_NOISE_BLOCK_HOURS: [f64; 2] = [300.0, 444.0];
/// Mean inflow (m3/s), shared by both hydros and stages.
pub const CHRONOLOGICAL_NOISE_INFLOW_MEAN_M3S: f64 = 60.0;
/// Per-hydro inflow standard deviation (m3/s), shared across both stages.
pub const CHRONOLOGICAL_NOISE_INFLOW_STD_M3S: [f64; 2] = [10.0, 20.0];
/// Initial storage (hm3), shared by both hydros.
pub const CHRONOLOGICAL_NOISE_INITIAL_STORAGE_HM3: f64 = 1000.0;
/// Spillage and turbined unit cost, shared by both hydros.
pub const CHRONOLOGICAL_NOISE_RELEASE_COST: f64 = 0.01;

const CHRONOLOGICAL_NOISE_N_STAGES: usize = 2;
const CHRONOLOGICAL_NOISE_BUS_ID: i32 = 10;
const CHRONOLOGICAL_NOISE_HYDRO_IDS: [i32; 2] = [1, 2];

/// [`chronological_noise_study`]'s varying axes: the consumer-set fields only
/// (`block_modes` for a future parallel variant, `max_storage_hm3` and
/// `branching_factor` for the lower-bound behavioural test).
pub struct ChronologicalNoiseSpec {
    /// Per-stage block mode.
    pub block_modes: [BlockMode; 2],
    /// Reservoir capacity, shared by both hydros.
    pub max_storage_hm3: f64,
    /// Stage scenario branching factor (root opening count).
    pub branching_factor: usize,
    /// Adds one pumping station from hydro id 2 (canonical position 1) to
    /// hydro id 1 (canonical position 0) -- a nonzero source position no
    /// committed deck exercises, pinning `fill_pumping_water_entries`'s
    /// per-hydro block-row addressing.
    pub pumping_station: bool,
    /// Reverses every entity vector before `SystemBuilder::build`.
    pub reverse_declaration_order: bool,
    /// Withdrawal \[m³/s\] on hydro id 1's stage-0 water-balance row only.
    pub water_withdrawal_m3s: f64,
}

impl Default for ChronologicalNoiseSpec {
    fn default() -> Self {
        Self {
            block_modes: [BlockMode::Chronological; 2],
            max_storage_hm3: 10_000.0,
            branching_factor: 1,
            pumping_station: false,
            reverse_declaration_order: false,
            water_withdrawal_m3s: 0.0,
        }
    }
}

fn chronological_noise_hydro_penalties() -> HydroPenalties {
    HydroPenalties {
        spillage_cost: CHRONOLOGICAL_NOISE_RELEASE_COST,
        turbined_cost: CHRONOLOGICAL_NOISE_RELEASE_COST,
        inflow_nonnegativity_cost: 0.0,
        ..hydro_penalties()
    }
}

// Rationale: the entity/bounds/penalties construction is one sequential
// fixture; splitting it into helper fns would fragment the declared shape
// across call sites with no reuse benefit.
#[allow(clippy::too_many_lines)]
fn build_chronological_noise_system(spec: &ChronologicalNoiseSpec) -> cobre_core::System {
    let start = NaiveDate::from_ymd_opt(2024, 1, 1).expect("chronological_noise_study: valid date");

    let bus = make_bus(
        EntityId(CHRONOLOGICAL_NOISE_BUS_ID),
        BusSpec {
            name: "B".to_string(),
            operational_start_date: start,
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 500.0,
            }],
            excess_cost: 0.0,
        },
    );

    let mut hydros: Vec<_> = CHRONOLOGICAL_NOISE_HYDRO_IDS
        .iter()
        .map(|&id| {
            make_hydro(
                EntityId(id),
                HydroSpec {
                    name: format!("H{id}"),
                    operational_start_date: start,
                    bus_id: EntityId(CHRONOLOGICAL_NOISE_BUS_ID),
                    min_storage_hm3: 0.0,
                    max_storage_hm3: spec.max_storage_hm3,
                    max_turbined_m3s: 100.0,
                    generation_model: HydroGenerationModel::ConstantProductivity,
                    specific_productivity_mw_per_m3s_per_m: Some(0.5),
                    max_generation_mw: 250.0,
                    penalties: chronological_noise_hydro_penalties(),
                    ..Default::default()
                },
            )
        })
        .collect();

    let blocks: Vec<Block> = CHRONOLOGICAL_NOISE_BLOCK_HOURS
        .iter()
        .enumerate()
        .map(|(index, &duration_hours)| Block {
            index,
            name: format!("B{index}"),
            duration_hours,
        })
        .collect();

    let stages: Vec<Stage> = (0..CHRONOLOGICAL_NOISE_N_STAGES)
        .map(|i| {
            make_stage(
                i,
                StageSpec {
                    start_date: stage_date(i),
                    end_date: stage_date(i + 1),
                    season_id: Some(0),
                    blocks: blocks.clone(),
                    block_mode: spec.block_modes[i],
                    state_config: StageStateConfig {
                        storage: true,
                        inflow_lags: false,
                    },
                    risk_config: StageRiskConfig::Expectation,
                    scenario_config: ScenarioSourceConfig {
                        branching_factor: spec.branching_factor,
                        noise_method: NoiseMethod::Saa,
                    },
                },
            )
        })
        .collect();

    let inflow_models = CHRONOLOGICAL_NOISE_HYDRO_IDS
        .iter()
        .enumerate()
        .flat_map(|(h, &id)| {
            (0..CHRONOLOGICAL_NOISE_N_STAGES).map(move |i| InflowModel {
                hydro_id: EntityId(id),
                stage_id: i32::try_from(i).expect("stage index fits i32"),
                mean_m3s: CHRONOLOGICAL_NOISE_INFLOW_MEAN_M3S,
                std_m3s: CHRONOLOGICAL_NOISE_INFLOW_STD_M3S[h],
                ar_coefficients: vec![],
                residual_std_ratio: 1.0,
                annual: None,
            })
        })
        .collect();

    let load_models = (0..CHRONOLOGICAL_NOISE_N_STAGES)
        .map(|i| LoadModel {
            bus_id: EntityId(CHRONOLOGICAL_NOISE_BUS_ID),
            stage_id: i32::try_from(i).expect("stage index fits i32"),
            mean_mw: 0.0,
            std_mw: 0.0,
        })
        .collect();

    let mut bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: CHRONOLOGICAL_NOISE_HYDRO_IDS.len(),
            n_thermals: 1,
            n_lines: 0,
            n_pumping: usize::from(spec.pumping_station),
            n_contracts: 0,
            n_stages: CHRONOLOGICAL_NOISE_N_STAGES,
            k_max: 0,
        },
        &BoundsDefaults {
            hydro: HydroStageBounds {
                min_storage_hm3: 0.0,
                max_storage_hm3: spec.max_storage_hm3,
                filling_min_rate_m3s: 0.0,
                water_withdrawal_m3s: 0.0,
            },
            hydro_block: HydroBlockBounds {
                max_turbined_m3s: 100.0,
                max_generation_mw: 250.0,
                ..Default::default()
            },
            thermal: ThermalStageBounds {
                cost_per_mwh: 100.0,
            },
            thermal_block: ThermalBlockBounds {
                min_generation_mw: 0.0,
                max_generation_mw: 400.0,
            },
            line_block: LineBlockBounds {
                direct_mw: 0.0,
                reverse_mw: 0.0,
            },
            pumping_block: PumpingBlockBounds {
                min_flow_m3s: 0.0,
                max_flow_m3s: 50.0,
            },
            contract_block: ContractBlockBounds {
                min_mw: 0.0,
                max_mw: 0.0,
                price_per_mwh: 0.0,
            },
        },
    );
    bounds.hydro_bounds_mut(0, 0).water_withdrawal_m3s = spec.water_withdrawal_m3s;

    let penalties = ResolvedPenalties::new(
        &PenaltiesCountsSpec {
            n_hydros: CHRONOLOGICAL_NOISE_HYDRO_IDS.len(),
            n_buses: 1,
            n_lines: 0,
            n_ncs: 0,
            n_stages: CHRONOLOGICAL_NOISE_N_STAGES,
        },
        &PenaltiesDefaults {
            hydro: chronological_noise_hydro_penalties(),
            bus: BusStagePenalties { excess_cost: 0.0 },
            line: LineStagePenalties { exchange_cost: 0.0 },
            ncs: NcsStagePenalties {
                curtailment_cost: 0.0,
            },
        },
    );

    let initial_conditions = InitialConditions {
        storage: CHRONOLOGICAL_NOISE_HYDRO_IDS
            .iter()
            .map(|&id| HydroStorage {
                hydro_id: EntityId(id),
                value_hm3: CHRONOLOGICAL_NOISE_INITIAL_STORAGE_HM3,
            })
            .collect(),
        filling_storage: vec![],
        past_anticipated_commitments: vec![],
        recent_observations: vec![],
        past_defluences: vec![],
    };

    let mut buses = vec![bus];
    let mut thermals = vec![make_thermal(
        EntityId(20),
        ThermalSpec {
            name: "T".to_string(),
            operational_start_date: start,
            bus_id: EntityId(CHRONOLOGICAL_NOISE_BUS_ID),
            cost_per_mwh: 100.0,
            min_generation_mw: 0.0,
            max_generation_mw: 400.0,
            anticipated_config: None,
            ..Default::default()
        },
    )];
    // Source position 1 (hydro id 2) != 0 pins fill_pumping_water_entries
    // addressing each hydro's own block row rather than a fixed offset.
    let mut pumping_stations = if spec.pumping_station {
        vec![PumpingStation {
            id: EntityId(10),
            name: "P1".to_string(),
            operational_start_date: start,
            bus_id: EntityId(CHRONOLOGICAL_NOISE_BUS_ID),
            source_hydro_id: EntityId(CHRONOLOGICAL_NOISE_HYDRO_IDS[1]),
            destination_hydro_id: EntityId(CHRONOLOGICAL_NOISE_HYDRO_IDS[0]),
            entry_stage_id: None,
            exit_stage_id: None,
            consumption_mw_per_m3s: 0.5,
            min_flow_m3s: 0.0,
            max_flow_m3s: 50.0,
        }]
    } else {
        vec![]
    };
    if spec.reverse_declaration_order {
        buses.reverse();
        hydros.reverse();
        thermals.reverse();
        pumping_stations.reverse();
    }

    SystemBuilder::new()
        .buses(buses)
        .thermals(thermals)
        .pumping_stations(pumping_stations)
        .hydros(hydros)
        .stages(stages)
        .inflow_models(inflow_models)
        .load_models(load_models)
        .bounds(bounds)
        .penalties(penalties)
        .initial_conditions(initial_conditions)
        .build()
        .expect("chronological_noise_study: two independent hydros on a two-stage system")
}

/// Two chronological (default) stages, two blocks each (300 h, 444 h), on two
/// independent hydros with per-hydro inflow standard deviations `[10, 20]`
/// m3/s: backs the chronological inflow-noise-ownership test
/// (`tests/chronological_inflow_noise.rs`) and the `(Chronological, inflow)`
/// cell of the patch-ownership sweep (`tests/patch_ownership_sweep.rs`).
#[must_use]
pub fn chronological_noise_study(spec: &ChronologicalNoiseSpec) -> (cobre_core::System, Config) {
    (build_chronological_noise_system(spec), build_config())
}

/// [`chronological_noise_study`] with its pumping station enabled, built
/// once with every stage [`BlockMode::Chronological`] and once
/// [`BlockMode::Parallel`] through the spec's own per-stage `block_modes`:
/// the chronological-vs-parallel sum identity's pumping non-vacuity case, no
/// committed deck combining a pumping station with a multi-block stage. Also
/// carries a small nonzero `water_withdrawal_m3s` (no committed deck or other
/// in-code study sets one), so the withdrawal term in the chronological
/// water-row RHS is no longer multiplied by zero here.
///
/// Each build's builder is called twice — see [`structural_studies`]'s doc
/// comment for why.
#[must_use]
pub fn chronological_pumping_pair() -> (
    (cobre_core::System, StudySetup),
    (cobre_core::System, StudySetup),
) {
    let chrono_spec = ChronologicalNoiseSpec {
        pumping_station: true,
        water_withdrawal_m3s: 2.0,
        ..Default::default()
    };
    let parallel_spec = ChronologicalNoiseSpec {
        block_modes: [BlockMode::Parallel; 2],
        pumping_station: true,
        water_withdrawal_m3s: 2.0,
        ..Default::default()
    };

    let (chrono_system, chrono_config) = chronological_noise_study(&chrono_spec);
    let (chrono_system_for_setup, _) = chronological_noise_study(&chrono_spec);
    let chrono_setup = super::build_setup_in_code(chrono_system_for_setup, &chrono_config);

    let (parallel_system, parallel_config) = chronological_noise_study(&parallel_spec);
    let (parallel_system_for_setup, _) = chronological_noise_study(&parallel_spec);
    let parallel_setup = super::build_setup_in_code(parallel_system_for_setup, &parallel_config);

    (
        (chrono_system, chrono_setup),
        (parallel_system, parallel_setup),
    )
}

const TWO_HYDRO_EVAP_N_STAGES: usize = 2;
const TWO_HYDRO_EVAP_BUS_ID: EntityId = EntityId(1);
const TWO_HYDRO_EVAP_HYDRO0_ID: EntityId = EntityId(2);
const TWO_HYDRO_EVAP_HYDRO1_ID: EntityId = EntityId(3);
const TWO_HYDRO_EVAP_THERMAL_ID: EntityId = EntityId(4);

// Rationale: the entity/bounds/penalties construction is one sequential
// fixture; splitting it into helper fns would fragment the declared shape
// across call sites with no reuse benefit.
#[allow(clippy::too_many_lines)]
fn build_two_hydro_evap_system() -> cobre_core::System {
    let bus = make_bus(
        TWO_HYDRO_EVAP_BUS_ID,
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

    let hydro0 = make_hydro(
        TWO_HYDRO_EVAP_HYDRO0_ID,
        HydroSpec {
            name: "H0".to_string(),
            operational_start_date: stage_date(0),
            bus_id: TWO_HYDRO_EVAP_BUS_ID,
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
    let hydro1 = make_hydro(
        TWO_HYDRO_EVAP_HYDRO1_ID,
        HydroSpec {
            name: "H1".to_string(),
            operational_start_date: stage_date(0),
            bus_id: TWO_HYDRO_EVAP_BUS_ID,
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
        TWO_HYDRO_EVAP_THERMAL_ID,
        ThermalSpec {
            name: "T0".to_string(),
            operational_start_date: stage_date(0),
            bus_id: TWO_HYDRO_EVAP_BUS_ID,
            min_generation_mw: 0.0,
            max_generation_mw: 100.0,
            cost_per_mwh: 50.0,
            ..Default::default()
        },
    );

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

    let stages: Vec<Stage> = (0..TWO_HYDRO_EVAP_N_STAGES)
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

    let inflow_models: Vec<InflowModel> = [TWO_HYDRO_EVAP_HYDRO0_ID, TWO_HYDRO_EVAP_HYDRO1_ID]
        .into_iter()
        .flat_map(|hydro_id| {
            (0..TWO_HYDRO_EVAP_N_STAGES).map(move |i| InflowModel {
                hydro_id,
                stage_id: i as i32,
                mean_m3s: 80.0,
                std_m3s: 0.0,
                ar_coefficients: vec![],
                residual_std_ratio: 1.0,
                annual: None,
            })
        })
        .collect();

    let load_models: Vec<LoadModel> = (0..TWO_HYDRO_EVAP_N_STAGES)
        .map(|i| LoadModel {
            bus_id: TWO_HYDRO_EVAP_BUS_ID,
            stage_id: i as i32,
            mean_mw: 100.0,
            std_mw: 0.0,
        })
        .collect();

    let bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 2,
            n_thermals: 1,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages: TWO_HYDRO_EVAP_N_STAGES,
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
            n_hydros: 2,
            n_buses: 1,
            n_lines: 0,
            n_ncs: 0,
            n_stages: TWO_HYDRO_EVAP_N_STAGES,
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

    let initial_conditions = InitialConditions {
        storage: vec![
            HydroStorage {
                hydro_id: TWO_HYDRO_EVAP_HYDRO0_ID,
                value_hm3: 100.0,
            },
            HydroStorage {
                hydro_id: TWO_HYDRO_EVAP_HYDRO1_ID,
                value_hm3: 100.0,
            },
        ],
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
        .hydros(vec![hydro0, hydro1])
        .thermals(vec![thermal])
        .stages(stages)
        .inflow_models(inflow_models)
        .load_models(load_models)
        .bounds(bounds)
        .penalties(penalties)
        .initial_conditions(initial_conditions)
        .policy_graph(policy_graph)
        .build()
        .expect("two_hydro_evaporation_study: valid system")
}

fn two_hydro_evap_hydro_models(system: &cobre_core::System) -> PrepareHydroModelsResult {
    let mut hydro_models = PrepareHydroModelsResult::default_from_system(system);
    hydro_models.evaporation = EvaporationModelSet::new(vec![
        EvaporationModel::None,
        EvaporationModel::Linearized {
            coefficients: vec![
                LinearizedEvaporation {
                    intercept_m3s: 1.0,
                    volume_slope_m3s_per_hm3: 0.01,
                };
                TWO_HYDRO_EVAP_N_STAGES
            ],
            reference_volumes_hm3: vec![100.0; TWO_HYDRO_EVAP_N_STAGES],
        },
    ]);
    hydro_models
}

/// Two hydros (`H0`, `H1`); only `H1` — canonical position **1**, not `0` —
/// evaporates, so a local-index-vs-system-index mix-up keying the evaporation
/// row's storage columns on the evaporating hydro's position within
/// `evap_hydro_indices` (`0`) instead of its system index (`1`) is
/// distinguishable here, unlike on any single-hydro evaporating fixture.
#[must_use]
pub fn two_hydro_evaporation_study() -> (cobre_core::System, Config, PrepareHydroModelsResult) {
    let system = build_two_hydro_evap_system();
    let hydro_models = two_hydro_evap_hydro_models(&system);
    (system, build_config(), hydro_models)
}

/// The in-code studies the structural sweep (`for_each_study`) visits after the
/// committed decks, in visit order: [`discounted_anticipated_study`] and
/// [`parallel_multiblock_evaporation_study`], each isolating a stage-LP builder
/// axis no committed deck combines; [`mixed_lead_anticipated_study`]'s two
/// anticipated lanes; [`two_hydro_evaporation_study`]'s nonzero-position
/// evaporating hydro; and [`parallel_inflow_slack_study`]'s multi-block
/// parallel inflow slack: the smallest in-code study that reaches the
/// inflow-slack-per-block mutation check, unreachable on every committed deck
/// and every other in-code study.
///
/// Each study's builder is called twice — once for the returned `System`,
/// once for the `System` `build_setup_in_code*` consumes by value — since
/// `cobre_core::System` has no `Clone` impl and every builder here is a pure,
/// deterministic function of its literal inputs, so the two calls yield
/// equal systems.
#[must_use]
pub fn structural_studies() -> Vec<(String, cobre_core::System, StudySetup)> {
    let (discounted_system, discounted_config) = discounted_anticipated_study();
    let (discounted_system_for_setup, _) = discounted_anticipated_study();
    let (parallel_evap_system, parallel_evap_config, _) = parallel_multiblock_evaporation_study();
    let (parallel_evap_system_for_setup, _, parallel_evap_hydro_models_for_setup) =
        parallel_multiblock_evaporation_study();
    let (mixed_lead_system, mixed_lead_config) = mixed_lead_anticipated_study(false);
    let (mixed_lead_system_for_setup, _) = mixed_lead_anticipated_study(false);
    let (evap_system, evap_config, _) = two_hydro_evaporation_study();
    let (evap_system_for_setup, _, evap_hydro_models_for_setup) = two_hydro_evaporation_study();
    let (slack_system, slack_config) = parallel_inflow_slack_study();
    let (slack_system_for_setup, _) = parallel_inflow_slack_study();
    vec![
        (
            "in-code/discounted-anticipated".to_string(),
            discounted_system,
            super::build_setup_in_code(discounted_system_for_setup, &discounted_config),
        ),
        (
            "in-code/parallel-multiblock-evaporation".to_string(),
            parallel_evap_system,
            super::build_setup_in_code_with_models(
                parallel_evap_system_for_setup,
                &parallel_evap_config,
                parallel_evap_hydro_models_for_setup,
            ),
        ),
        (
            "structural/mixed-lead-anticipated".to_string(),
            mixed_lead_system,
            super::build_setup_in_code(mixed_lead_system_for_setup, &mixed_lead_config),
        ),
        (
            "structural/two-hydro-evaporation".to_string(),
            evap_system,
            super::build_setup_in_code_with_models(
                evap_system_for_setup,
                &evap_config,
                evap_hydro_models_for_setup,
            ),
        ),
        (
            "structural/parallel-inflow-slack".to_string(),
            slack_system,
            super::build_setup_in_code(slack_system_for_setup, &slack_config),
        ),
    ]
}
