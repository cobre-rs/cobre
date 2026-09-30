//! Integration tests for [`cobre_sddp::build_stage_templates_resolving_layout`].
//!
//! Covers structural (column/row counts, CSC validity), objective coefficient
//! wiring, and constraint-matrix entries for hydro / FPHA / evaporation /
//! water-withdrawal / multi-segment deficit / generic constraints / operational
//! violation slacks / inflow non-negativity / stochastic load balance / PAR
//! max-order derivation / water-balance z-inflow coupling.

#![allow(
    clippy::doc_markdown,
    clippy::too_many_lines,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::cast_precision_loss
)]
// `..Default::default()` in the make_* Spec calls is the intentional future-field
// seam from `common::builders` — a no-op today, not dead code.
#![allow(clippy::needless_update)]

use cobre_core::{
    AnticipatedConfig, BoundsCountsSpec, BoundsDefaults, Bus, BusStagePenalties,
    ContractBlockBounds, DeficitSegment, EntityId, HydroBlockBounds, HydroPenalties,
    HydroStageBounds, LineBlockBounds, LineStagePenalties, NcsStagePenalties, PenaltiesCountsSpec,
    PenaltiesDefaults, PumpingBlockBounds, ResolvedBounds, ResolvedPenalties, SystemBuilder,
    ThermalBlockBounds, ThermalStageBounds, scenario::InflowModel,
};
use cobre_stochastic::normal::precompute::PrecomputedNormal;
use cobre_stochastic::par::precompute::PrecomputedPar;

use cobre_sddp::{
    build_stage_templates_resolving_layout,
    hydro_models::{
        EvaporationModel, EvaporationModelSet, FphaPlane, LinearizedEvaporation,
        PrepareHydroModelsResult, ProductionModelSet, ResolvedProductionModel,
    },
    indexer::{BlockGrid, BlockRowFamily, StateSpace},
    inflow_method::InflowNonNegativityMethod,
    lead_time::AnticipatedResolution,
    lp::builder::PatchBuffer,
    resolved_parameters::ResolvedParameters,
};

mod common;
use common::builders::{
    BusSpec, HydroSpec, StageSpec, ThermalSpec, make_bus, make_hydro, make_stage, make_thermal,
};

/// LP objective cost scale factor. Matches `cobre_sddp::setup::params::DEFAULT_COST_SCALE_FACTOR`.
const COST_SCALE_FACTOR: f64 = 1_000_000.0;

/// Evaporation flow safety margin multiplier. Matches `cobre_sddp::lp::builder::EVAPORATION_FLOW_SAFETY_MARGIN`.
const EVAPORATION_FLOW_SAFETY_MARGIN: f64 = 2.0;

fn default_production(system: &cobre_core::System) -> ProductionModelSet {
    PrepareHydroModelsResult::default_from_system(system).production
}

fn default_evaporation(system: &cobre_core::System) -> EvaporationModelSet {
    PrepareHydroModelsResult::default_from_system(system).evaporation
}

fn production_set(productivities: &[f64], n_stages: usize) -> ProductionModelSet {
    let n_hydros = productivities.len();
    let models = productivities
        .iter()
        .map(|&p| vec![ResolvedProductionModel::ConstantProductivity { productivity: p }; n_stages])
        .collect();
    ProductionModelSet::new(
        models,
        &cobre_sddp::test_support::minimal_hydros(n_hydros),
        n_stages,
    )
}

fn no_penalty_config() -> InflowNonNegativityMethod {
    InflowNonNegativityMethod::None
}

fn penalty_config(_cost: f64) -> InflowNonNegativityMethod {
    InflowNonNegativityMethod::Penalty
}

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

/// One-bus, no-entity system with `n_stages` study stages.
#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
fn one_bus_system(n_stages: usize) -> cobre_core::System {
    use chrono::NaiveDate;
    use cobre_core::scenario::LoadModel;
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    };

    let bus = make_bus(
        EntityId(1),
        BusSpec {
            name: "B1".to_string(),
            operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 500.0,
            }],
            excess_cost: 0.0,
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
                    season_id: None,
                    blocks: vec![Block {
                        index: 0,
                        name: "S".to_string(),
                        duration_hours: 744.0,
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
                    ..Default::default()
                },
            )
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
            n_hydros: 0,
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
            n_hydros: 0,
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
        .stages(stages)
        .load_models(load_models)
        .bounds(bounds)
        .penalties(penalties)
        .build()
        .expect("one_bus_system: valid")
}

/// N=1 hydro, B=1 bus, no thermals/lines, K=1 block, L=`lag_order`.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::too_many_lines
)]
fn one_hydro_system(n_stages: usize, lag_order: usize) -> cobre_core::System {
    use chrono::NaiveDate;
    use cobre_core::entities::hydro::{HydroGenerationModel, HydroPenalties};
    use cobre_core::scenario::{InflowModel, LoadModel};
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    };

    let bus = make_bus(
        EntityId(1),
        BusSpec {
            name: "B1".to_string(),
            operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 500.0,
            }],
            excess_cost: 0.0,
            ..Default::default()
        },
    );

    let hydro = make_hydro(
        EntityId(2),
        HydroSpec {
            name: "H1".to_string(),
            operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId(1),
            downstream_id: None,
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
                    season_id: None,
                    blocks: vec![Block {
                        index: 0,
                        name: "S".to_string(),
                        duration_hours: 744.0,
                    }],
                    block_mode: BlockMode::Parallel,
                    state_config: StageStateConfig {
                        storage: true,
                        inflow_lags: lag_order > 0,
                    },
                    risk_config: StageRiskConfig::Expectation,
                    scenario_config: ScenarioSourceConfig {
                        branching_factor: 1,
                        noise_method: NoiseMethod::Saa,
                    },
                    ..Default::default()
                },
            )
        })
        .collect();

    let ar_coefficients: Vec<f64> = (0..lag_order).map(|_| 0.5).collect();
    let inflow_models: Vec<InflowModel> = (0..n_stages)
        .map(|i| InflowModel {
            hydro_id: EntityId(2),
            stage_id: i as i32,
            mean_m3s: 80.0,
            std_m3s: 20.0,
            ar_coefficients: ar_coefficients.clone(),
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
        .hydros(vec![hydro])
        .stages(stages)
        .inflow_models(inflow_models)
        .load_models(load_models)
        .bounds(bounds)
        .penalties(penalties)
        .build()
        .expect("one_hydro_system: valid")
}

/// 1-bus, 1-FPHA-hydro, 1-stage system with `n_planes` planes, parameterized
/// `turbined_cost`, and explicit block durations.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::too_many_lines
)]
fn fpha_system_with_turbined_cost(
    n_planes: usize,
    turbined_cost: f64,
    block_durations_hours: &[f64],
) -> (cobre_core::System, ProductionModelSet) {
    use chrono::NaiveDate;
    use cobre_core::entities::hydro::{HydroGenerationModel, HydroPenalties};
    use cobre_core::scenario::{InflowModel, LoadModel};
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    };

    let n_blks = block_durations_hours.len();
    assert!(n_blks > 0, "must have at least one block");

    let bus = make_bus(
        EntityId(1),
        BusSpec {
            name: "B1".to_string(),
            operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 500.0,
            }],
            excess_cost: 0.0,
            ..Default::default()
        },
    );
    let hydro = make_hydro(
        EntityId(2),
        HydroSpec {
            name: "FPHA1".to_string(),
            operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId(1),
            downstream_id: None,
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: 0.0,
            max_storage_hm3: 500.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: HydroGenerationModel::Fpha,
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 150.0,
            specific_productivity_mw_per_m3s_per_m: None,
            min_generation_mw: 0.0,
            max_generation_mw: 300.0,
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
                turbined_cost,
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

    let blocks: Vec<Block> = block_durations_hours
        .iter()
        .enumerate()
        .map(|(i, &hours)| Block {
            index: i,
            name: format!("BLK{i}"),
            duration_hours: hours,
        })
        .collect();

    let stages: Vec<Stage> = vec![make_stage(
        0,
        StageSpec {
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            season_id: None,
            blocks,
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
            ..Default::default()
        },
    )];

    let inflow_models: Vec<InflowModel> = vec![InflowModel {
        hydro_id: EntityId(2),
        stage_id: 0,
        mean_m3s: 80.0,
        std_m3s: 20.0,
        ar_coefficients: vec![],
        residual_std_ratio: 1.0,
        annual: None,
    }];

    let load_models: Vec<LoadModel> = vec![LoadModel {
        bus_id: EntityId(1),
        stage_id: 0,
        mean_mw: 200.0,
        std_mw: 0.0,
    }];

    let bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 1,
            n_thermals: 0,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages: 1,
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
                max_turbined_m3s: 150.0,
                max_generation_mw: 300.0,
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
            n_ncs: 0,
            n_stages: 1,
        },
        &PenaltiesDefaults {
            hydro: HydroPenalties {
                spillage_cost: 0.01,
                diversion_cost: 0.0,
                turbined_cost,
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
            bus: BusStagePenalties { excess_cost: 0.0 },
            line: LineStagePenalties { exchange_cost: 0.0 },
            ncs: NcsStagePenalties {
                curtailment_cost: 0.0,
            },
        },
    );

    let system = SystemBuilder::new()
        .buses(vec![bus])
        .hydros(vec![hydro])
        .stages(stages)
        .inflow_models(inflow_models)
        .load_models(load_models)
        .bounds(bounds)
        .penalties(penalties)
        .build()
        .expect("fpha_system_with_turbined_cost: valid");

    let plane = FphaPlane {
        intercept: 10.0,
        gamma_v: 0.5,
        gamma_q: 2.0,
        gamma_s: 0.1,
    };
    let planes = vec![plane; n_planes];
    let models = vec![vec![ResolvedProductionModel::Fpha { planes }]];
    let production = ProductionModelSet::new(models, system.hydros(), 1);

    (system, production)
}

/// Build a two-bus system with N hydros and K blocks per stage.
/// Bus B1 (EntityId=10) has `std_mw` = 0 (no load noise).
/// Bus B2 (EntityId=20) has `std_mw` > 0 (stochastic load).
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::too_many_lines
)]
fn two_bus_system_with_stochastic_load(
    n_stages: usize,
    n_hydros_in_system: usize,
    n_blocks: usize,
) -> cobre_core::System {
    use chrono::NaiveDate;
    use cobre_core::entities::hydro::{Hydro, HydroGenerationModel, HydroPenalties};
    use cobre_core::scenario::{InflowModel, LoadModel};
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    };

    let bus1 = make_bus(
        EntityId(10),
        BusSpec {
            name: "B1".to_string(),
            operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 500.0,
            }],
            excess_cost: 0.0,
            ..Default::default()
        },
    );
    let bus2 = make_bus(
        EntityId(20),
        BusSpec {
            name: "B2".to_string(),
            operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 500.0,
            }],
            excess_cost: 0.0,
            ..Default::default()
        },
    );

    let blocks: Vec<_> = (0..n_blocks)
        .map(|b| Block {
            index: b,
            name: format!("B{b}"),
            duration_hours: 240.0,
        })
        .collect();

    let stages: Vec<Stage> = (0..n_stages)
        .map(|i| {
            make_stage(
                i,
                StageSpec {
                    start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                    end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
                    season_id: None,
                    blocks: blocks.clone(),
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
                    ..Default::default()
                },
            )
        })
        .collect();

    let hydros: Vec<Hydro> = (0..n_hydros_in_system)
        .map(|h| {
            make_hydro(
                EntityId((h + 100) as i32),
                HydroSpec {
                    name: format!("H{h}"),
                    operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                    bus_id: EntityId(10),
                    downstream_id: None,
                    entry_stage_id: None,
                    exit_stage_id: None,
                    min_storage_hm3: 0.0,
                    max_storage_hm3: 200.0,
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
            )
        })
        .collect();

    let inflow_models: Vec<InflowModel> = hydros
        .iter()
        .flat_map(|h| {
            (0..n_stages).map(move |s| InflowModel {
                hydro_id: h.id,
                stage_id: s as i32,
                mean_m3s: 50.0,
                std_m3s: 10.0,
                ar_coefficients: vec![],
                residual_std_ratio: 1.0,
                annual: None,
            })
        })
        .collect();

    let load_models: Vec<LoadModel> = (0..n_stages)
        .map(|s| LoadModel {
            bus_id: EntityId(10),
            stage_id: s as i32,
            mean_mw: 80.0,
            std_mw: 0.0, // B1: no noise
        })
        .chain((0..n_stages).map(|s| LoadModel {
            bus_id: EntityId(20),
            stage_id: s as i32,
            mean_mw: 120.0,
            std_mw: 15.0, // B2: stochastic
        }))
        .collect();

    let n_st = n_stages.max(1);
    let bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: n_hydros_in_system,
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
            n_hydros: n_hydros_in_system,
            n_buses: 2,
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

    let mut builder = SystemBuilder::new()
        .buses(vec![bus1, bus2])
        .stages(stages)
        .load_models(load_models)
        .bounds(bounds)
        .penalties(penalties);
    if !hydros.is_empty() {
        builder = builder.hydros(hydros).inflow_models(inflow_models);
    }
    builder.build().expect("two_bus_system: valid")
}

/// CSC coefficient at (`col`, `row`); `None` if the column has no entry in that row.
#[allow(clippy::cast_sign_loss)] // col_starts and row_indices are non-negative by construction
fn csc_entry(tmpl: &cobre_solver::StageTemplate, col: usize, row: usize) -> Option<f64> {
    let start = tmpl.col_starts[col] as usize;
    let end = tmpl.col_starts[col + 1] as usize;
    for pos in start..end {
        if tmpl.row_indices[pos] as usize == row {
            return Some(tmpl.values[pos]);
        }
    }
    None
}

/// 1-bus, 1-FPHA-hydro, 1-stage, 1-block system with `n_planes` planes, each
/// `intercept=10.0, gamma_v=0.5, gamma_q=2.0, gamma_s=0.1`.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::too_many_lines
)]
fn one_fpha_hydro_system(n_planes: usize) -> (cobre_core::System, ProductionModelSet) {
    use chrono::NaiveDate;
    use cobre_core::entities::hydro::{HydroGenerationModel, HydroPenalties};
    use cobre_core::scenario::{InflowModel, LoadModel};
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    };

    let bus = make_bus(
        EntityId(1),
        BusSpec {
            name: "B1".to_string(),
            operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 500.0,
            }],
            excess_cost: 0.0,
            ..Default::default()
        },
    );
    let hydro = make_hydro(
        EntityId(2),
        HydroSpec {
            name: "FPHA1".to_string(),
            operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId(1),
            downstream_id: None,
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: 0.0,
            max_storage_hm3: 500.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: HydroGenerationModel::Fpha,
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 150.0,
            specific_productivity_mw_per_m3s_per_m: None,
            min_generation_mw: 0.0,
            max_generation_mw: 300.0,
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
            ..Default::default()
        },
    );

    let stages: Vec<Stage> = vec![make_stage(
        0,
        StageSpec {
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
            ..Default::default()
        },
    )];

    let inflow_models: Vec<InflowModel> = vec![InflowModel {
        hydro_id: EntityId(2),
        stage_id: 0,
        mean_m3s: 80.0,
        std_m3s: 20.0,
        ar_coefficients: vec![],
        residual_std_ratio: 1.0,
        annual: None,
    }];

    let load_models: Vec<LoadModel> = vec![LoadModel {
        bus_id: EntityId(1),
        stage_id: 0,
        mean_mw: 200.0,
        std_mw: 0.0,
    }];

    let bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 1,
            n_thermals: 0,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages: 1,
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
                max_turbined_m3s: 150.0,
                max_generation_mw: 300.0,
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
            n_ncs: 0,
            n_stages: 1,
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
        .hydros(vec![hydro])
        .stages(stages)
        .inflow_models(inflow_models)
        .load_models(load_models)
        .bounds(bounds)
        .penalties(penalties)
        .build()
        .expect("one_fpha_hydro_system: valid");

    let plane = FphaPlane {
        intercept: 10.0,
        gamma_v: 0.5,
        gamma_q: 2.0,
        gamma_s: 0.1,
    };
    let planes = vec![plane; n_planes];
    let models = vec![vec![ResolvedProductionModel::Fpha { planes }]];
    let production = ProductionModelSet::new(models, system.hydros(), 1);

    (system, production)
}

/// 1-bus, 4-hydro, 1-stage, 1-block system. EntityId-sorted: 100 (const),
/// 101 (const), 102 (fpha), 103 (fpha) → hydro indices 0,1 are constant
/// productivity; indices 2,3 are FPHA (3 planes each).
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::too_many_lines
)]
fn four_hydro_mixed_system() -> (cobre_core::System, ProductionModelSet) {
    use chrono::NaiveDate;
    use cobre_core::entities::hydro::{HydroGenerationModel, HydroPenalties};
    use cobre_core::scenario::{InflowModel, LoadModel};
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    };

    let bus = make_bus(
        EntityId(1),
        BusSpec {
            name: "B1".to_string(),
            operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 500.0,
            }],
            excess_cost: 0.0,
            ..Default::default()
        },
    );

    let hydro_penalties = HydroPenalties {
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
    };

    let hydros = vec![
        make_hydro(
            EntityId(100),
            HydroSpec {
                name: "H100".to_string(),
                operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                bus_id: EntityId(1),
                downstream_id: None,
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
                penalties: hydro_penalties,
                ..Default::default()
            },
        ),
        make_hydro(
            EntityId(101),
            HydroSpec {
                name: "H101".to_string(),
                operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                bus_id: EntityId(1),
                downstream_id: None,
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
                penalties: hydro_penalties,
                ..Default::default()
            },
        ),
        make_hydro(
            EntityId(102),
            HydroSpec {
                name: "H102".to_string(),
                operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                bus_id: EntityId(1),
                downstream_id: None,
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
                penalties: hydro_penalties,
                ..Default::default()
            },
        ),
        make_hydro(
            EntityId(103),
            HydroSpec {
                name: "H103".to_string(),
                operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                bus_id: EntityId(1),
                downstream_id: None,
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
                penalties: hydro_penalties,
                ..Default::default()
            },
        ),
    ];

    let stages: Vec<Stage> = vec![make_stage(
        0,
        StageSpec {
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
            ..Default::default()
        },
    )];

    let inflow_models: Vec<InflowModel> = hydros
        .iter()
        .map(|h| InflowModel {
            hydro_id: h.id,
            stage_id: 0,
            mean_m3s: 80.0,
            std_m3s: 20.0,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
            annual: None,
        })
        .collect();

    let load_models: Vec<LoadModel> = vec![LoadModel {
        bus_id: EntityId(1),
        stage_id: 0,
        mean_mw: 400.0,
        std_mw: 0.0,
    }];

    let bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 4,
            n_thermals: 0,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages: 1,
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
            n_hydros: 4,
            n_buses: 1,
            n_lines: 0,
            n_ncs: 0,
            n_stages: 1,
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
        .hydros(hydros)
        .stages(stages)
        .inflow_models(inflow_models)
        .load_models(load_models)
        .bounds(bounds)
        .penalties(penalties)
        .build()
        .expect("four_hydro_mixed_system: valid");

    let plane = FphaPlane {
        intercept: 10.0,
        gamma_v: 0.5,
        gamma_q: 2.0,
        gamma_s: 0.1,
    };
    let fpha_planes = vec![plane; 3];
    let models = vec![
        vec![ResolvedProductionModel::ConstantProductivity { productivity: 2.5 }],
        vec![ResolvedProductionModel::ConstantProductivity { productivity: 3.0 }],
        vec![ResolvedProductionModel::Fpha {
            planes: fpha_planes.clone(),
        }],
        vec![ResolvedProductionModel::Fpha {
            planes: fpha_planes,
        }],
    ];
    let production = ProductionModelSet::new(models, system.hydros(), 1);

    (system, production)
}

// -------------------------------------------------------------------------
// FPHA LP integration tests (HiGHS end-to-end solve)
// -------------------------------------------------------------------------
//
// Planes (intercept=300.0, gamma_v=1.0, gamma_q=3.0, gamma_s=0.0) keep the LP
// feasible: at v_in=100 hm³, load=200 MW, g=200, q=50 m³/s the FPHA RHS is
// 300 + 1.0*v_avg + 3.0*50 ≈ 550 >> 200.

/// 1-FPHA-hydro system reusing `one_fpha_hydro_system` but with 3 planes whose
/// large intercept keeps the solve feasible for any `v_in` in `[0, 500]` hm³:
/// `intercept=300.0, gamma_v=1.0, gamma_q=3.0, gamma_s=0.0`.
fn fpha_solve_system() -> (cobre_core::System, ProductionModelSet) {
    let planes = vec![
        FphaPlane {
            intercept: 300.0,
            gamma_v: 1.0,
            gamma_q: 3.0,
            gamma_s: 0.0,
        };
        3
    ];
    let models = vec![vec![ResolvedProductionModel::Fpha { planes }]];
    let (system, _) = one_fpha_hydro_system(3);
    let production = ProductionModelSet::new(models, system.hydros(), 1);
    (system, production)
}

// =========================================================================
// Evaporation variable tests
// =========================================================================

use cobre_solver::StageTemplate;

/// `EvaporationModelSet` giving `None` to every hydro except those in
/// `evap_indices`, which get `Linearized` from `intercept_per_stage`.
fn evap_set_for_system(
    system: &cobre_core::System,
    evap_indices: &[usize],
    intercept_per_stage: &[f64],
) -> EvaporationModelSet {
    let n_hydros = system.hydros().len();
    let n_stages = system.stages().iter().filter(|s| s.id >= 0).count();
    let models = (0..n_hydros)
        .map(|h| {
            if evap_indices.contains(&h) {
                let coefficients = (0..n_stages)
                    .map(|s| LinearizedEvaporation {
                        intercept_m3s: intercept_per_stage
                            .get(s)
                            .copied()
                            .unwrap_or(intercept_per_stage.first().copied().unwrap_or(0.0)),
                        volume_slope_m3s_per_hm3: 0.0,
                    })
                    .collect();
                EvaporationModel::Linearized {
                    coefficients,
                    reference_volumes_hm3: vec![100.0; n_stages],
                }
            } else {
                EvaporationModel::None
            }
        })
        .collect();
    EvaporationModelSet::new(models)
}

// =========================================================================
// fill_evaporation_entries — CSC matrix entries
// =========================================================================

/// Like `evap_set_for_system` but with an explicit `volume_slope_m3s_per_hm3`
/// alongside `intercept_m3s`.
fn evap_set_with_volume_slope(
    system: &cobre_core::System,
    evap_indices: &[usize],
    intercept_m3s: f64,
    volume_slope_m3s_per_hm3: f64,
) -> EvaporationModelSet {
    let n_hydros = system.hydros().len();
    let n_stages = system.stages().iter().filter(|s| s.id >= 0).count();
    let models = (0..n_hydros)
        .map(|h| {
            if evap_indices.contains(&h) {
                let coefficients = (0..n_stages)
                    .map(|_| LinearizedEvaporation {
                        intercept_m3s,
                        volume_slope_m3s_per_hm3,
                    })
                    .collect();
                EvaporationModel::Linearized {
                    coefficients,
                    reference_volumes_hm3: vec![100.0; n_stages],
                }
            } else {
                EvaporationModel::None
            }
        })
        .collect();
    EvaporationModelSet::new(models)
}

/// All `(row, value)` CSC entries for a given column.
#[allow(clippy::cast_sign_loss)] // col_starts and row_indices are non-negative by construction
fn entries_for_col(t: &StageTemplate, col: usize) -> Vec<(usize, f64)> {
    let start = t.col_starts[col] as usize;
    let end = t.col_starts[col + 1] as usize;
    (start..end)
        .map(|i| (t.row_indices[i] as usize, t.values[i]))
        .collect()
}

// =========================================================================
// Evaporation violation cost tests
// =========================================================================

/// 1-bus, 1-hydro (constant-productivity), 1-stage system with a parameterized
/// `evaporation_violation_cost` and block duration.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::too_many_lines
)]
fn evap_hydro_system_with_violation_cost(
    block_hours: f64,
    evaporation_violation_cost: f64,
) -> cobre_core::System {
    use chrono::NaiveDate;
    use cobre_core::entities::hydro::{HydroGenerationModel, HydroPenalties};
    use cobre_core::scenario::{InflowModel, LoadModel};
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, StageRiskConfig, StageStateConfig,
    };

    let bus = make_bus(
        EntityId(1),
        BusSpec {
            name: "B1".to_string(),
            operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 500.0,
            }],
            excess_cost: 0.0,
            ..Default::default()
        },
    );

    let hydro = make_hydro(
        EntityId(2),
        HydroSpec {
            name: "H1".to_string(),
            operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId(1),
            downstream_id: None,
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: 0.0,
            max_storage_hm3: 2_000.0,
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
                evaporation_violation_cost,
                water_withdrawal_violation_cost: 0.0,
                water_withdrawal_violation_pos_cost: 0.0,
                water_withdrawal_violation_neg_cost: 0.0,
                evaporation_violation_pos_cost: evaporation_violation_cost,
                evaporation_violation_neg_cost: evaporation_violation_cost,
                inflow_nonnegativity_cost: 1000.0,
            },
            ..Default::default()
        },
    );

    let stages = vec![make_stage(
        0,
        StageSpec {
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            season_id: None,
            blocks: vec![Block {
                index: 0,
                name: "S".to_string(),
                duration_hours: block_hours,
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
            ..Default::default()
        },
    )];

    let inflow_models = vec![InflowModel {
        hydro_id: EntityId(2),
        stage_id: 0,
        mean_m3s: 50.0,
        std_m3s: 10.0,
        ar_coefficients: vec![],
        residual_std_ratio: 1.0,
        annual: None,
    }];

    let load_models = vec![LoadModel {
        bus_id: EntityId(1),
        stage_id: 0,
        mean_mw: 100.0,
        std_mw: 0.0,
    }];

    let bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 1,
            n_thermals: 0,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages: 1,
            k_max: 0,
        },
        &BoundsDefaults {
            hydro: HydroStageBounds {
                min_storage_hm3: 0.0,
                max_storage_hm3: 2_000.0,
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
            n_ncs: 0,
            n_stages: 1,
        },
        &PenaltiesDefaults {
            hydro: HydroPenalties {
                spillage_cost: 0.01,
                diversion_cost: 0.0,
                turbined_cost: 0.0,
                storage_violation_below_cost: 0.0,
                filling_target_violation_cost: 0.0,
                turbined_violation_below_cost: 0.0,
                outflow_violation_below_cost: 0.0,
                outflow_violation_above_cost: 0.0,
                generation_violation_below_cost: 0.0,
                evaporation_violation_cost,
                water_withdrawal_violation_cost: 0.0,
                water_withdrawal_violation_pos_cost: 0.0,
                water_withdrawal_violation_neg_cost: 0.0,
                evaporation_violation_pos_cost: evaporation_violation_cost,
                evaporation_violation_neg_cost: evaporation_violation_cost,
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
        .hydros(vec![hydro])
        .stages(stages)
        .inflow_models(inflow_models)
        .load_models(load_models)
        .bounds(bounds)
        .penalties(penalties)
        .build()
        .expect("evap_hydro_system_with_violation_cost: valid")
}

// ─── Multi-segment deficit tests ──────────────────────────────────────────

/// No-hydro, no-thermal, no-line system over the given buses, 1 stage / 1 block.
#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
fn multi_segment_system(buses: Vec<Bus>, block_hours: f64) -> cobre_core::System {
    use chrono::NaiveDate;
    use cobre_core::scenario::LoadModel;
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, StageRiskConfig, StageStateConfig,
    };

    let n_buses = buses.len();
    let load_models: Vec<LoadModel> = buses
        .iter()
        .map(|b| LoadModel {
            bus_id: b.id,
            stage_id: 0,
            mean_mw: 0.0,
            std_mw: 0.0,
        })
        .collect();

    let stage = make_stage(
        0,
        StageSpec {
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            season_id: None,
            blocks: vec![Block {
                index: 0,
                name: "S".to_string(),
                duration_hours: block_hours,
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
            ..Default::default()
        },
    );

    let bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 0,
            n_thermals: 0,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages: 1,
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
            n_hydros: 0,
            n_buses,
            n_lines: 0,
            n_ncs: 0,
            n_stages: 1,
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
        .buses(buses)
        .stages(vec![stage])
        .load_models(load_models)
        .bounds(bounds)
        .penalties(penalties)
        .build()
        .expect("multi_segment_system: valid")
}

/// `one_hydro_system` variant injecting `water_withdrawal_m3s` and
/// `water_withdrawal_violation_cost`. One 744h block; `lag_order` adds AR lag columns.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::too_many_lines
)]
fn one_hydro_system_with_withdrawal(
    n_stages: usize,
    lag_order: usize,
    water_withdrawal_m3s: f64,
    water_withdrawal_violation_cost: f64,
) -> cobre_core::System {
    use chrono::NaiveDate;
    use cobre_core::entities::hydro::{HydroGenerationModel, HydroPenalties};
    use cobre_core::scenario::{InflowModel, LoadModel};
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    };

    let bus = make_bus(
        EntityId(1),
        BusSpec {
            name: "B1".to_string(),
            operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 500.0,
            }],
            excess_cost: 0.0,
            ..Default::default()
        },
    );

    let hydro = make_hydro(
        EntityId(2),
        HydroSpec {
            name: "H1".to_string(),
            operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId(1),
            downstream_id: None,
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
                water_withdrawal_violation_cost,
                water_withdrawal_violation_pos_cost: water_withdrawal_violation_cost,
                water_withdrawal_violation_neg_cost: water_withdrawal_violation_cost,
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
                    season_id: None,
                    blocks: vec![Block {
                        index: 0,
                        name: "S".to_string(),
                        duration_hours: 744.0,
                    }],
                    block_mode: BlockMode::Parallel,
                    state_config: StageStateConfig {
                        storage: true,
                        inflow_lags: lag_order > 0,
                    },
                    risk_config: StageRiskConfig::Expectation,
                    scenario_config: ScenarioSourceConfig {
                        branching_factor: 1,
                        noise_method: NoiseMethod::Saa,
                    },
                    ..Default::default()
                },
            )
        })
        .collect();

    let ar_coefficients: Vec<f64> = (0..lag_order).map(|_| 0.5).collect();
    let inflow_models: Vec<InflowModel> = (0..n_stages)
        .map(|i| InflowModel {
            hydro_id: EntityId(2),
            stage_id: i as i32,
            mean_m3s: 80.0,
            std_m3s: 20.0,
            ar_coefficients: ar_coefficients.clone(),
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

    // Build bounds with the specified withdrawal rate for every (hydro, stage) cell.
    let mut bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 1,
            n_thermals: 0,
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
                water_withdrawal_m3s,
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
    for s in 0..n_st {
        bounds.hydro_bounds_mut(0, s).water_withdrawal_m3s = water_withdrawal_m3s;
    }

    let mut penalties = ResolvedPenalties::new(
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
                storage_violation_below_cost: 0.0,
                filling_target_violation_cost: 0.0,
                turbined_violation_below_cost: 0.0,
                outflow_violation_below_cost: 0.0,
                outflow_violation_above_cost: 0.0,
                generation_violation_below_cost: 0.0,
                evaporation_violation_cost: 0.0,
                water_withdrawal_violation_cost,
                water_withdrawal_violation_pos_cost: water_withdrawal_violation_cost,
                water_withdrawal_violation_neg_cost: water_withdrawal_violation_cost,
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
    for s in 0..n_st {
        penalties
            .hydro_penalties_mut(0, s)
            .water_withdrawal_violation_cost = water_withdrawal_violation_cost;
    }

    SystemBuilder::new()
        .buses(vec![bus])
        .hydros(vec![hydro])
        .stages(stages)
        .inflow_models(inflow_models)
        .load_models(load_models)
        .bounds(bounds)
        .penalties(penalties)
        .build()
        .expect("one_hydro_system_with_withdrawal: valid")
}

/// One-bus, one-stage system with `n_blks` operating blocks.
#[allow(clippy::cast_possible_wrap)]
fn one_bus_system_n_blks(n_blks: usize) -> cobre_core::System {
    use chrono::NaiveDate;
    use cobre_core::scenario::LoadModel;
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, StageRiskConfig, StageStateConfig,
    };

    let bus = make_bus(
        EntityId(1),
        BusSpec {
            name: "B1".to_string(),
            operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 500.0,
            }],
            excess_cost: 0.0,
            ..Default::default()
        },
    );

    let blocks: Vec<Block> = (0..n_blks)
        .map(|i| Block {
            index: i,
            name: format!("BLK{i}"),
            duration_hours: 720.0,
        })
        .collect();

    let stage = make_stage(
        0,
        StageSpec {
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            season_id: None,
            blocks,
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
            ..Default::default()
        },
    );

    let load_models: Vec<LoadModel> = vec![LoadModel {
        bus_id: EntityId(1),
        stage_id: 0,
        mean_mw: 100.0,
        std_mw: 0.0,
    }];

    let bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 0,
            n_thermals: 0,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages: 1,
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
            n_hydros: 0,
            n_buses: 1,
            n_lines: 0,
            n_ncs: 0,
            n_stages: 1,
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
        .stages(vec![stage])
        .load_models(load_models)
        .bounds(bounds)
        .penalties(penalties)
        .build()
        .expect("one_bus_system_n_blks: valid")
}

/// `GenericConstraint` with a trivial (no-term) expression. A no-term expression
/// is vacuously block-independent, so a `block_id = None` bound on it collapses to
/// a single stage-level row.
fn make_constraint(id: i32, slack_enabled: bool) -> cobre_core::GenericConstraint {
    use cobre_core::ConstraintExpression;
    make_constraint_with_expr(id, slack_enabled, ConstraintExpression { terms: vec![] })
}

/// Make a `GenericConstraint` carrying the given expression.
fn make_constraint_with_expr(
    id: i32,
    slack_enabled: bool,
    expression: cobre_core::ConstraintExpression,
) -> cobre_core::GenericConstraint {
    use cobre_core::{GenericConstraint, SlackConfig};
    GenericConstraint {
        id: EntityId(id),
        name: format!("gc_{id}"),
        description: None,
        expression,
        slack: SlackConfig {
            enabled: slack_enabled,
            penalty: if slack_enabled { Some(5000.0) } else { None },
        },
        bound_lower_affine: None,
        bound_upper_affine: None,
    }
}

/// A block-level expression: one `BusExcess` term with variable-level
/// `block_id = None`. `BusExcess` resolves to a per-block column
/// (`excess.start + bus_pos * n_blks + block_idx`), so a `block_id = None` bound
/// on this expression is **not** collapsible — distinct blocks yield distinct rows.
fn block_level_excess_expr(bus_id: i32) -> cobre_core::ConstraintExpression {
    use cobre_core::{ConstraintExpression, LinearTerm, VariableRef};
    ConstraintExpression {
        terms: vec![LinearTerm::literal(
            1.0,
            VariableRef::BusExcess {
                bus_id: EntityId(bus_id),
                block_id: None,
            },
        )],
    }
}

/// Build templates for `system` using the no-penalty method and default PAR/Normal.
fn build_templates_for(system: &cobre_core::System) -> Vec<cobre_solver::StageTemplate> {
    let production = default_production(system);
    let evaporation = default_evaporation(system);
    build_stage_templates_resolving_layout(
        system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &production,
        &evaporation,
        &ResolvedParameters::default(),
    )
    .expect("build_templates_for: valid")
    .templates
}

/// Helper: build a one-bus system with `n_blks` operating blocks and
/// the given generic constraints + resolved bounds.
#[allow(clippy::cast_possible_wrap)]
fn one_bus_system_n_blks_with_generic(
    n_blks: usize,
    constraints: Vec<cobre_core::GenericConstraint>,
    bounds: cobre_core::ResolvedGenericConstraintBounds,
) -> cobre_core::System {
    use chrono::NaiveDate;
    use cobre_core::scenario::LoadModel;
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, StageRiskConfig, StageStateConfig,
    };

    let bus = make_bus(
        EntityId(1),
        BusSpec {
            name: "B1".to_string(),
            operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 500.0,
            }],
            excess_cost: 0.0,
            ..Default::default()
        },
    );

    let blks: Vec<Block> = (0..n_blks)
        .map(|i| Block {
            index: i,
            name: format!("BLK{i}"),
            duration_hours: 720.0,
        })
        .collect();

    let stage = make_stage(
        0,
        StageSpec {
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            season_id: None,
            blocks: blks,
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
            ..Default::default()
        },
    );

    let load_models: Vec<LoadModel> = vec![LoadModel {
        bus_id: EntityId(1),
        stage_id: 0,
        mean_mw: 100.0,
        std_mw: 0.0,
    }];

    let resolved_bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 0,
            n_thermals: 0,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages: 1,
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
            n_hydros: 0,
            n_buses: 1,
            n_lines: 0,
            n_ncs: 0,
            n_stages: 1,
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
        .stages(vec![stage])
        .load_models(load_models)
        .bounds(resolved_bounds)
        .penalties(penalties)
        .generic_constraints(constraints)
        .resolved_generic_bounds(bounds)
        .build()
        .expect("one_bus_system_n_blks_with_generic: valid")
}

// ── Helper: scan CSC for entries in a specific (column, row) pair ──────────

/// All values stored at `(col, row)` in the CSC template. Returns a `Vec` (not
/// `Option`) because two fill helpers can add to the same position; tests check
/// the total or assert uniqueness.
fn csc_entries_at(t: &cobre_solver::StageTemplate, col: usize, row: usize) -> Vec<f64> {
    let start = t.col_starts[col] as usize;
    let end = t.col_starts[col + 1] as usize;
    t.row_indices[start..end]
        .iter()
        .zip(t.values[start..end].iter())
        .filter_map(|(&r, &v)| if r as usize == row { Some(v) } else { None })
        .collect()
}

// ── Helper: build a one-bus, one-thermal system with generic constraints ───

/// 1-bus, 1-thermal (constant productivity), 1-block, 1-stage system with the
/// given generic constraints. Column layout (N=0, T=1, B=1, K=1, no penalty/FPHA/evap):
///   theta=0, thermal=[1,2), deficit=[2,3), excess=[3,4),
///   withdrawal_slack=[] (n_h=0), col_generic_slack_start=4.
/// Row layout: load_balance=[0,1), generic_start=1.
#[allow(clippy::cast_possible_wrap)]
fn one_bus_one_thermal_system(
    thermal_entity_id: EntityId,
    constraints: Vec<cobre_core::GenericConstraint>,
    bounds: cobre_core::ResolvedGenericConstraintBounds,
) -> cobre_core::System {
    use chrono::NaiveDate;
    use cobre_core::scenario::LoadModel;
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, StageRiskConfig, StageStateConfig,
    };

    let bus = make_bus(
        EntityId(1),
        BusSpec {
            name: "B1".to_string(),
            operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 500.0,
            }],
            excess_cost: 0.0,
            ..Default::default()
        },
    );

    let thermal = make_thermal(
        thermal_entity_id,
        ThermalSpec {
            name: "T1".to_string(),
            operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId(1),
            min_generation_mw: 0.0,
            max_generation_mw: 100.0,
            cost_per_mwh: 50.0,
            anticipated_config: None,
            entry_stage_id: None,
            exit_stage_id: None,
            ..Default::default()
        },
    );

    let stage = make_stage(
        0,
        StageSpec {
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            season_id: None,
            blocks: vec![Block {
                index: 0,
                name: "BLK0".to_string(),
                duration_hours: 744.0,
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
            ..Default::default()
        },
    );

    let load_models = vec![LoadModel {
        bus_id: EntityId(1),
        stage_id: 0,
        mean_mw: 100.0,
        std_mw: 0.0,
    }];

    let resolved_bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 0,
            n_thermals: 1,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages: 1,
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
            n_hydros: 0,
            n_buses: 1,
            n_lines: 0,
            n_ncs: 0,
            n_stages: 1,
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
        .stages(vec![stage])
        .load_models(load_models)
        .bounds(resolved_bounds)
        .penalties(penalties)
        .generic_constraints(constraints)
        .resolved_generic_bounds(bounds)
        .build()
        .expect("one_bus_one_thermal_system: valid")
}

// ─────────────────────────────────────────────────────────────────────
// Operational violation slack structural tests
// ─────────────────────────────────────────────────────────────────────

/// 1-hydro, 1-bus, 2-block system with active operational bounds.
/// Blocks: 720.0h (heavy), 48.0h (light).
/// Hydro: min_outflow=50.0, max_outflow=Some(800.0), min_turbined=10.0,
///        min_generation=5.0, productivity=0.5.
/// Penalties: outflow_below/above=1000, turbined_below=1000, generation_below=1000.
#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
fn one_hydro_active_violations(n_stages: usize) -> cobre_core::System {
    use chrono::NaiveDate;
    use cobre_core::entities::hydro::{HydroGenerationModel, HydroPenalties};
    use cobre_core::scenario::{InflowModel, LoadModel};
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    };

    let bus = make_bus(
        EntityId(1),
        BusSpec {
            name: "B1".to_string(),
            operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 500.0,
            }],
            excess_cost: 0.0,
            ..Default::default()
        },
    );

    let hydro = make_hydro(
        EntityId(2),
        HydroSpec {
            name: "H1".to_string(),
            operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId(1),
            downstream_id: None,
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: 0.0,
            max_storage_hm3: 200.0,
            min_outflow_m3s: 50.0,
            max_outflow_m3s: Some(800.0),
            generation_model: HydroGenerationModel::ConstantProductivity,
            min_turbined_m3s: 10.0,
            max_turbined_m3s: 100.0,
            specific_productivity_mw_per_m3s_per_m: None,
            min_generation_mw: 5.0,
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
                turbined_violation_below_cost: 1000.0,
                outflow_violation_below_cost: 1000.0,
                outflow_violation_above_cost: 1000.0,
                generation_violation_below_cost: 1000.0,
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
                    season_id: None,
                    blocks: vec![
                        Block {
                            index: 0,
                            name: "Heavy".to_string(),
                            duration_hours: 720.0,
                        },
                        Block {
                            index: 1,
                            name: "Light".to_string(),
                            duration_hours: 48.0,
                        },
                    ],
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
                    ..Default::default()
                },
            )
        })
        .collect();

    let inflow_models: Vec<InflowModel> = (0..n_stages)
        .map(|i| InflowModel {
            hydro_id: EntityId(2),
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
            n_thermals: 0,
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
                min_turbined_m3s: 10.0,
                max_turbined_m3s: 100.0,
                min_outflow_m3s: 50.0,
                max_outflow_m3s: Some(800.0),
                min_generation_mw: 5.0,
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
            n_ncs: 0,
            n_stages: n_st,
        },
        &PenaltiesDefaults {
            hydro: HydroPenalties {
                spillage_cost: 0.01,
                diversion_cost: 0.0,
                turbined_cost: 0.0,
                storage_violation_below_cost: 0.0,
                filling_target_violation_cost: 0.0,
                turbined_violation_below_cost: 1000.0,
                outflow_violation_below_cost: 1000.0,
                outflow_violation_above_cost: 1000.0,
                generation_violation_below_cost: 1000.0,
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
        .hydros(vec![hydro])
        .stages(stages)
        .inflow_models(inflow_models)
        .load_models(load_models)
        .bounds(bounds)
        .penalties(penalties)
        .build()
        .expect("one_hydro_active_violations: valid")
}
fn build_active_violations_template() -> cobre_sddp::StageTemplates {
    let system = one_hydro_active_violations(1);
    // Productivity = 0.5 to match the coefficient expected by
    // `min_generation_constant_productivity_coefficients`.
    let pm = production_set(&[0.5], 1);
    build_stage_templates_resolving_layout(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &pm,
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("active violations ok")
}

// -------------------------------------------------------------------------
// max_par_order derivation tests (annual component path)
// -------------------------------------------------------------------------

/// 1-stage, 2-hydro system with AR order `ar_order` per hydro. `season_id: Some(0)`
/// lets `PrecomputedPar::build` resolve lag-stage statistics via the season fallback
/// even with no pre-study inflow models.
#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
fn two_hydro_par_system(ar_order: usize, inflow_models: Vec<InflowModel>) -> cobre_core::System {
    use chrono::NaiveDate;
    use cobre_core::entities::hydro::{HydroGenerationModel, HydroPenalties};
    use cobre_core::scenario::LoadModel;
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, StageRiskConfig, StageStateConfig,
    };

    let bus = make_bus(
        EntityId(1),
        BusSpec {
            name: "B1".to_string(),
            operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 500.0,
            }],
            excess_cost: 0.0,
            ..Default::default()
        },
    );

    let stages = vec![make_stage(
        0,
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
                inflow_lags: ar_order > 0,
            },
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor: 1,
                noise_method: NoiseMethod::Saa,
            },
            ..Default::default()
        },
    )];

    let load_models = vec![LoadModel {
        bus_id: EntityId(1),
        stage_id: 0,
        mean_mw: 100.0,
        std_mw: 0.0,
    }];

    let bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 2,
            n_thermals: 0,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages: 1,
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
            n_stages: 1,
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
        .hydros(vec![
            make_hydro(
                EntityId(2),
                HydroSpec {
                    name: "H1".to_string(),
                    operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                    bus_id: EntityId(1),
                    downstream_id: None,
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
                    ..Default::default()
                },
            ),
            make_hydro(
                EntityId(3),
                HydroSpec {
                    name: "H2".to_string(),
                    operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                    bus_id: EntityId(1),
                    downstream_id: None,
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
                    ..Default::default()
                },
            ),
        ])
        .stages(stages)
        .inflow_models(inflow_models)
        .load_models(load_models)
        .bounds(bounds)
        .penalties(penalties)
        .build()
        .expect("two_hydro_par_system: valid")
}

// ─────────────────────────────────────────────────────────────────────────────
// Anticipated-decision column bounds tests
// ─────────────────────────────────────────────────────────────────────────────

/// Build a minimal system with one anticipated thermal (`K_i = lead_stages`,
/// `min_generation_mw`, `max_generation_mw`) and `n_stages` study stages.
///
/// System geometry: 0 hydros, 1 thermal (anticipated), 1 bus, 1 block per
/// stage, 1 deficit segment. `ResolvedBounds` is constructed with `k_max =
/// lead_stages` so delivery-stage lookups (`t + K_i`) never exceed the thermal
/// stage axis.
///
/// Column layout for this geometry (0 hydros, 1 thermal, 1 anticipated, 1 blk):
/// - `n_ant_state  = n_anticipated * k_max = 1 * lead_stages`
/// - `theta        = n_ant_state` (no hydro z_inflow or storage_in columns)
/// - `decision_start = theta + 1`
/// - `col_thermal_start = decision_start` (0 turbine/spillage/diversion cols)
/// - `col_anticipated_decision_start = col_thermal_start + n_thermals * n_blks`
///   = `theta + 1 + 1` = `theta + 2`
///
/// For `lead_stages = 2`: `n_ant_state = 2`, `theta = 2`,
/// `col_anticipated_decision_start = 4`.
#[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
fn one_anticipated_thermal_system(
    n_stages: usize,
    lead_stages: u32,
    min_generation_mw: f64,
    max_generation_mw: f64,
) -> cobre_core::System {
    use chrono::NaiveDate;
    use cobre_core::scenario::LoadModel;
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    };

    let bus = make_bus(
        EntityId(1),
        BusSpec {
            name: "B1".to_string(),
            operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 500.0,
            }],
            excess_cost: 0.0,
            ..Default::default()
        },
    );

    let thermal = make_thermal(
        EntityId(2),
        ThermalSpec {
            name: "T_ant".to_string(),
            operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId(1),
            min_generation_mw,
            max_generation_mw,
            cost_per_mwh: 50.0,
            anticipated_config: Some(AnticipatedConfig::LeadStages(lead_stages)),
            entry_stage_id: None,
            exit_stage_id: None,
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
                    season_id: None,
                    blocks: vec![Block {
                        index: 0,
                        name: "BLK0".to_string(),
                        duration_hours: 744.0,
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
                    ..Default::default()
                },
            )
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

    let k_max = lead_stages as usize;
    let n_st = n_stages.max(1);
    let bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 0,
            n_thermals: 1,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages: n_st,
            k_max,
        },
        &BoundsDefaults {
            hydro: default_hydro_bounds(),
            hydro_block: default_hydro_block_bounds(),
            thermal: ThermalStageBounds { cost_per_mwh: 50.0 },
            thermal_block: ThermalBlockBounds {
                min_generation_mw,
                max_generation_mw,
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
            n_hydros: 0,
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
        .stages(stages)
        .load_models(load_models)
        .bounds(bounds)
        .penalties(penalties)
        .build()
        .expect("one_anticipated_thermal_system: valid")
}

/// Compute `col_anticipated_decision_start` for the minimal geometry used by
/// the anticipated-decision tests (0 hydros, 1 thermal, 1 anticipated, 1 blk).
///
/// Layout derivation (in-LP anticipated ring, `StateSpace::commit_out`
/// / `commit_in`):
/// - `n_ant_state = n_anticipated * k_max = 1 * lead_stages`
/// - the ring contributes `n_ant_state` outgoing columns AND `n_ant_state`
///   incoming columns (doubled, unlike the water/lag blocks which are
///   outgoing-only): `theta = n_ant_state + n_ant_state + 0 (z_inflow)
///   + 0 (storage_in) = 2 * n_ant_state`
/// - `decision_start = theta + 1`
/// - `col_thermal_start = decision_start` (0 turbine/spillage/diversion cols)
/// - `col_anticipated_decision_start = col_thermal_start + n_thermals * n_blks`
fn anticipated_decision_col(lead_stages: usize) -> usize {
    let n_ant_state = lead_stages; // n_anticipated=1, k_max=lead_stages
    let theta = 2 * n_ant_state; // outgoing + incoming ring blocks
    let decision_start = theta + 1;
    let col_thermal_start = decision_start; // 0 hydro turbine/spillage/diversion cols
    col_thermal_start + 1 // n_thermals=1, n_blks=1
}

// ── Helper: two-thermal system (one anticipated, one not) ────────────────────

/// Build a system with two thermals: `thermal 0` anticipated (K=`lead_stages`),
/// `thermal 1` non-anticipated. Both have `cost_per_mwh = 50.0`.
///
/// Column layout (0 hydros, 2 thermals, 1 anticipated, 1 blk per stage):
/// - `n_ant_state = n_anticipated * k_max = 1 * lead_stages`
/// - `theta = 2 * n_ant_state` (the ring's outgoing AND incoming blocks)
/// - `col_thermal_start = theta + 1` (0 turbine/spillage/diversion cols)
/// - `col_anticipated_decision_start = col_thermal_start + 2 * n_blks`
///   = `theta + 1 + 2`
#[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
fn two_thermal_one_anticipated_system(n_stages: usize, lead_stages: u32) -> cobre_core::System {
    use chrono::NaiveDate;
    use cobre_core::scenario::LoadModel;
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    };

    let bus = make_bus(
        EntityId(1),
        BusSpec {
            name: "B1".to_string(),
            operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 500.0,
            }],
            excess_cost: 0.0,
            ..Default::default()
        },
    );

    let thermal_ant = make_thermal(
        EntityId(2),
        ThermalSpec {
            name: "T_ant".to_string(),
            operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId(1),
            min_generation_mw: 0.0,
            max_generation_mw: 100.0,
            cost_per_mwh: 50.0,
            anticipated_config: Some(AnticipatedConfig::LeadStages(lead_stages)),
            entry_stage_id: None,
            exit_stage_id: None,
            ..Default::default()
        },
    );
    let thermal_non = make_thermal(
        EntityId(3),
        ThermalSpec {
            name: "T_non".to_string(),
            operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId(1),
            min_generation_mw: 0.0,
            max_generation_mw: 100.0,
            cost_per_mwh: 50.0,
            anticipated_config: None,
            entry_stage_id: None,
            exit_stage_id: None,
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
                    season_id: None,
                    blocks: vec![Block {
                        index: 0,
                        name: "BLK0".to_string(),
                        duration_hours: 744.0,
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
                    ..Default::default()
                },
            )
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

    let k_max = lead_stages as usize;
    let n_st = n_stages.max(1);
    let bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 0,
            n_thermals: 2,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages: n_st,
            k_max,
        },
        &BoundsDefaults {
            hydro: default_hydro_bounds(),
            hydro_block: default_hydro_block_bounds(),
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
            n_hydros: 0,
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
        .thermals(vec![thermal_ant, thermal_non])
        .stages(stages)
        .load_models(load_models)
        .bounds(bounds)
        .penalties(penalties)
        .build()
        .expect("two_thermal_one_anticipated_system: valid")
}

/// Compute column offsets for the two-thermal geometry.
///
/// Layout: 0 hydros, 2 thermals, 1 anticipated (thermal 0), 1 blk, K=lead_stages.
/// The anticipated ring's outgoing AND incoming blocks (each width
/// `n_ant_state`) both precede `theta`.
/// - `n_ant_state = lead_stages`
/// - `theta = 2 * n_ant_state`
/// - `col_thermal_start = theta + 1`
/// - `col_thermal_0_blk0 = col_thermal_start`           (thermal 0, block 0)
/// - `col_thermal_1_blk0 = col_thermal_start + 1`       (thermal 1, block 0)
/// - `col_anticipated_start = col_thermal_start + 2`    (2 thermals * 1 blk)
fn two_thermal_col_thermal_start(lead_stages: usize) -> usize {
    let n_ant_state = lead_stages;
    let theta = 2 * n_ant_state; // outgoing + incoming ring blocks
    theta + 1
}

// ── anticipated-fishing row tests ────────────────────────────────────────────

/// Build a system with two anticipated thermals (K_0=1, K_1=2) and one bus.
///
/// Both thermals are anticipated; no non-anticipated thermals in this fixture.
/// Geometry (0 hydros, 2 thermals, n_anticipated=2, k_max=2, 1 bus, 1 blk/stage):
/// - `n_ant_state = 2 * 2 = 4`
/// - `theta = 8` (N=0 → N*(3+L) = 0; theta = 2 * n_ant_state, outgoing + incoming)
/// - `col_thermal_start = 9` (decision_start = theta+1 = 9; 0 turbine/spillage/diversion)
/// - `StateSpace::commit_out.start = 0` (N*(1+L)=0, outgoing ring)
#[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
fn two_anticipated_thermal_system(n_stages: usize) -> cobre_core::System {
    use chrono::NaiveDate;
    use cobre_core::scenario::LoadModel;
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    };

    let bus = make_bus(
        EntityId(1),
        BusSpec {
            name: "B1".to_string(),
            operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 500.0,
            }],
            excess_cost: 0.0,
            ..Default::default()
        },
    );

    let thermal_0 = make_thermal(
        EntityId(2),
        ThermalSpec {
            name: "T_ant0".to_string(),
            operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId(1),
            min_generation_mw: 0.0,
            max_generation_mw: 100.0,
            cost_per_mwh: 50.0,
            anticipated_config: Some(AnticipatedConfig::LeadStages(1)),
            entry_stage_id: None,
            exit_stage_id: None,
            ..Default::default()
        },
    );
    let thermal_1 = make_thermal(
        EntityId(3),
        ThermalSpec {
            name: "T_ant1".to_string(),
            operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId(1),
            min_generation_mw: 0.0,
            max_generation_mw: 100.0,
            cost_per_mwh: 50.0,
            anticipated_config: Some(AnticipatedConfig::LeadStages(2)),
            entry_stage_id: None,
            exit_stage_id: None,
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
                    season_id: None,
                    blocks: vec![Block {
                        index: 0,
                        name: "BLK0".to_string(),
                        duration_hours: 744.0,
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
                    ..Default::default()
                },
            )
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

    // k_max = max(K_0, K_1) = 2.
    let k_max = 2_usize;
    let n_st = n_stages.max(1);
    let bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 0,
            n_thermals: 2,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages: n_st,
            k_max,
        },
        &BoundsDefaults {
            hydro: default_hydro_bounds(),
            hydro_block: default_hydro_block_bounds(),
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
            n_hydros: 0,
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
        .thermals(vec![thermal_0, thermal_1])
        .stages(stages)
        .load_models(load_models)
        .bounds(bounds)
        .penalties(penalties)
        .build()
        .expect("two_anticipated_thermal_system: valid")
}

/// 1 hydro (max_par_order=1) + 1 anticipated thermal (K=2), exercising the
/// `n_state = N*(1+L) + n_ant_state` formula with non-zero terms:
/// `1*(1+1) + 2 = 4`.
///
/// `PrecomputedPar::default()` is safe here: the matrix builder guards par_lp
/// accesses with `par_lp.n_stages() > 0` (false for the default), so PAR
/// coefficients are treated as zero — acceptable for structural tests.
#[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
fn one_hydro_one_ant_system(n_stages: usize) -> cobre_core::System {
    use chrono::NaiveDate;
    use cobre_core::entities::hydro::{HydroGenerationModel, HydroPenalties};
    use cobre_core::scenario::{InflowModel, LoadModel};
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    };

    let bus = make_bus(
        EntityId(1),
        BusSpec {
            name: "B1".to_string(),
            operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 500.0,
            }],
            excess_cost: 0.0,
            ..Default::default()
        },
    );

    let hydro = make_hydro(
        EntityId(2),
        HydroSpec {
            name: "H1".to_string(),
            operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId(1),
            downstream_id: None,
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
            ..Default::default()
        },
    );

    // One anticipated thermal with K=2 → n_anticipated=1, k_max=2, n_ant_state=2.
    let thermal = make_thermal(
        EntityId(3),
        ThermalSpec {
            name: "T_ant".to_string(),
            operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId(1),
            min_generation_mw: 0.0,
            max_generation_mw: 100.0,
            cost_per_mwh: 50.0,
            anticipated_config: Some(AnticipatedConfig::LeadStages(2)),
            entry_stage_id: None,
            exit_stage_id: None,
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
                    season_id: None,
                    blocks: vec![Block {
                        index: 0,
                        name: "S".to_string(),
                        duration_hours: 744.0,
                    }],
                    block_mode: BlockMode::Parallel,
                    state_config: StageStateConfig {
                        storage: true,
                        inflow_lags: true, // AR(1) → contributes to max_par_order
                    },
                    risk_config: StageRiskConfig::Expectation,
                    scenario_config: ScenarioSourceConfig {
                        branching_factor: 1,
                        noise_method: NoiseMethod::Saa,
                    },
                    ..Default::default()
                },
            )
        })
        .collect();

    // AR(1) inflow model per stage: ar_coefficients.len()=1 drives max_par_order=1.
    // Note: season_id=None is safe here because PrecomputedPar::default() is used
    // (n_stages()==0 bypasses all par_lp branches in the matrix builder).
    let inflow_models: Vec<InflowModel> = (0..n_stages)
        .map(|i| InflowModel {
            hydro_id: EntityId(2),
            stage_id: i as i32,
            mean_m3s: 80.0,
            std_m3s: 20.0,
            ar_coefficients: vec![0.5],
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

    // k_max = K = 2 (single anticipated thermal with lead_stages=2).
    let k_max = 2_usize;
    let n_st = n_stages.max(1);
    let bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 1,
            n_thermals: 1,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages: n_st,
            k_max,
        },
        &BoundsDefaults {
            hydro: default_hydro_bounds(),
            hydro_block: default_hydro_block_bounds(),
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
        .hydros(vec![hydro])
        .thermals(vec![thermal])
        .stages(stages)
        .inflow_models(inflow_models)
        .load_models(load_models)
        .bounds(bounds)
        .penalties(penalties)
        .build()
        .expect("one_hydro_one_ant_system: valid")
}

// ── Anticipated-ring geometry fixtures ───────────────────────────────────────
//
// Two-anticipated-thermal system geometry:
//   n_hydros=0, n_anticipated=2 (K_0=1, K_1=2), k_max=2, n_ant_state=4
//   StateSpace::commit_out.start = 0   (N*(1+L) = 0, outgoing ring)
//   theta = 2 * n_ant_state = 8  (outgoing + incoming ring blocks)
//   decision_start = 9
//   col_thermal_start = 9  (0 turbine/spillage/diversion cols)
//   col_anticipated_decision_start = 9 + 2*1 = 11  (2 thermals, 1 blk)
//
// One-anticipated-thermal system geometry (K=2):
//   n_hydros=0, n_anticipated=1, k_max=2, n_ant_state=2
//   StateSpace::commit_out.start = 0
//   col_anticipated_decision_start = anticipated_decision_col(2) = 6

// ─── Anticipated Thermals K=1/2/3 Roundtrip ────────────────────────────────
//
// These integration tests exercise the full LP construction
// for synthetic systems with one hydro + one anticipated thermal at K=1, K=2,
// and K=3.  They verify all cross-cutting structural invariants simultaneously:
// column count, row count, n_state, anticipated_decision bounds, NPV objective
// coefficient, decision-write CSC, and fishing-row
// CSC pattern.
//
// System geometry (shared across K=1/2/3):
//   N=1 hydro (constant productivity, L=0, max_par_order=0)
//   T=1 thermal (anticipated, K_i=K, min=0, max=100, cost=50)
//   B=1 bus, n_blks=2, block_hours=360h
//   n_stages=4, no FPHA, no evaporation, no generic constraints
//
// Columns are addressed through `rt_col_thermal_start` and its siblings below.
//
// Row layout derivation (K arbitrary, stage t; state is pinned via column
// bounds, so the row layout carries no K-dependent state row block): the
// K-independent z_inflow/water_balance/load_balance/min_outflow/max_outflow/
// min_turbine/min_generation rows total 12, so `anticipated_fishing` (always
// active) starts at row 12 (`rt_row_ant_fishing_start`). The remaining rows —
// `anticipated_state_out_def` and the interior ring-shift definition rows —
// are counted by `rt_expected_num_rows`.
//
// ─────────────────────────────────────────────────────────────────────────────

/// Build a system with 1 hydro (constant productivity, L=0) and 1 anticipated
/// thermal (K_i = `lead_stages`), 1 bus, 2 blocks of 360 h per stage.
///
/// Used by the K=1, K=2, K=3 roundtrip tests.
///
/// Block durations are 360h so that total_stage_hours = 720h (consistent
/// across stages), which keeps the NPV objective computation tractable for
/// the discount-rate verification below.
#[allow(
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::too_many_lines
)]
fn build_hydro_one_ant_system(
    n_stages: usize,
    lead_stages: u32,
    annual_discount_rate: f64,
) -> cobre_core::System {
    use chrono::NaiveDate;
    use cobre_core::HorizonGraph;
    use cobre_core::entities::hydro::{HydroGenerationModel, HydroPenalties};
    use cobre_core::scenario::{InflowModel, LoadModel};
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, PolicyGraphType, ScenarioSourceConfig, Stage,
        StageRiskConfig, StageStateConfig,
    };

    let bus = make_bus(
        EntityId(1),
        BusSpec {
            name: "B1".to_string(),
            operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 500.0,
            }],
            excess_cost: 0.0,
            ..Default::default()
        },
    );

    let hydro = make_hydro(
        EntityId(2),
        HydroSpec {
            name: "H1".to_string(),
            operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId(1),
            downstream_id: None,
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
            ..Default::default()
        },
    );

    let thermal = make_thermal(
        EntityId(3),
        ThermalSpec {
            name: "T_ant".to_string(),
            operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId(1),
            min_generation_mw: 0.0,
            max_generation_mw: 100.0,
            cost_per_mwh: 50.0,
            anticipated_config: Some(AnticipatedConfig::LeadStages(lead_stages)),
            entry_stage_id: None,
            exit_stage_id: None,
            ..Default::default()
        },
    );

    // 2 blocks of 360 h each (total 720 h/stage).
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

    let stages: Vec<Stage> = (0..n_stages)
        .map(|i| {
            make_stage(
                i,
                StageSpec {
                    start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                    end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
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
                    ..Default::default()
                },
            )
        })
        .collect();

    // AR(0) inflow model (no lags → max_par_order=0).
    let inflow_models: Vec<InflowModel> = (0..n_stages)
        .map(|i| InflowModel {
            hydro_id: EntityId(2),
            stage_id: i as i32,
            mean_m3s: 80.0,
            std_m3s: 0.0,
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

    let k_max = lead_stages as usize;
    let n_st = n_stages.max(1);
    let bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 1,
            n_thermals: 1,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages: n_st,
            k_max,
        },
        &BoundsDefaults {
            hydro: default_hydro_bounds(),
            hydro_block: default_hydro_block_bounds(),
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
        .hydros(vec![hydro])
        .thermals(vec![thermal])
        .stages(stages)
        .inflow_models(inflow_models)
        .load_models(load_models)
        .bounds(bounds)
        .penalties(penalties)
        .policy_graph(policy_graph)
        .build()
        .expect("build_hydro_one_ant_system: valid")
}

/// Build the K=1 roundtrip system (lead_stages=1, no discounting).
fn build_k1_system() -> cobre_core::System {
    build_hydro_one_ant_system(4, 1, 0.0)
}

/// Build the K=2 roundtrip system (lead_stages=2, no discounting).
fn build_k2_system() -> cobre_core::System {
    build_hydro_one_ant_system(4, 2, 0.0)
}

/// Build the K=3 roundtrip system (lead_stages=3, no discounting).
fn build_k3_system() -> cobre_core::System {
    build_hydro_one_ant_system(4, 3, 0.0)
}

/// K=0 baseline: 1 hydro + 1 NON-anticipated thermal (`anticipated_config: None`
/// → `n_anticipated=0`), so the LP layout matches the pre-anticipated baseline.
fn build_k0_baseline_system() -> cobre_core::System {
    use chrono::NaiveDate;
    use cobre_core::entities::hydro::{HydroGenerationModel, HydroPenalties};
    use cobre_core::scenario::{InflowModel, LoadModel};
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    };

    let bus = make_bus(
        EntityId(1),
        BusSpec {
            name: "B1".to_string(),
            operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 500.0,
            }],
            excess_cost: 0.0,
            ..Default::default()
        },
    );

    let hydro = make_hydro(
        EntityId(2),
        HydroSpec {
            name: "H1".to_string(),
            operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId(1),
            downstream_id: None,
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
            ..Default::default()
        },
    );

    // Non-anticipated thermal — same bounds as the anticipated thermal in K-cases.
    let thermal = make_thermal(
        EntityId(3),
        ThermalSpec {
            name: "T_non".to_string(),
            operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId(1),
            min_generation_mw: 0.0,
            max_generation_mw: 100.0,
            cost_per_mwh: 50.0,
            anticipated_config: None,
            entry_stage_id: None,
            exit_stage_id: None,
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

    let stages: Vec<Stage> = (0..4)
        .map(|i| {
            make_stage(
                i,
                StageSpec {
                    start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                    end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
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
                    ..Default::default()
                },
            )
        })
        .collect();

    let inflow_models: Vec<InflowModel> = (0_i32..4)
        .map(|i| InflowModel {
            hydro_id: EntityId(2),
            stage_id: i,
            mean_m3s: 80.0,
            std_m3s: 0.0,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
            annual: None,
        })
        .collect();

    let load_models: Vec<LoadModel> = (0_i32..4)
        .map(|i| LoadModel {
            bus_id: EntityId(1),
            stage_id: i,
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
            n_stages: 4,
            k_max: 0,
        },
        &BoundsDefaults {
            hydro: default_hydro_bounds(),
            hydro_block: default_hydro_block_bounds(),
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
            n_stages: 4,
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
        .hydros(vec![hydro])
        .thermals(vec![thermal])
        .stages(stages)
        .inflow_models(inflow_models)
        .load_models(load_models)
        .bounds(bounds)
        .penalties(penalties)
        .build()
        .expect("build_k0_baseline_system: valid")
}

// ── Column layout helpers for the roundtrip geometry ─────────────────────────

/// `commit_in.start` (incoming ring, the column the always-active
/// fishing row couples) for the roundtrip geometry (N=1, L=0, K=k).
///
/// = N*(3+L) + k = 3 + k: the incoming block sits after
/// `z_inflow`/`storage_in`, K stages after `theta`'s outgoing-side start.
fn rt_col_ant_state_incoming_start(k: usize) -> usize {
    3 + k
}

/// `col_thermal_start` for the roundtrip geometry (N=1, L=0, K=k).
///
/// The anticipated ring contributes `2*k` columns (outgoing `commit_out`
/// AND incoming `commit_in`, each width `k`) before `theta`.
/// theta = N*(3+L) + 2*k = 3 + 2*k, decision_start = theta+1 = 4+2k, so
/// col_thermal_start = decision_start + 3*N*n_blks = decision_start + 6 = 10+2K.
fn rt_col_thermal_start(k: usize) -> usize {
    10 + 2 * k
}

/// `col_anticipated_decision_start` for the roundtrip geometry.
///
/// = col_thermal_start + T*n_blks = (10+2K) + 2 = 12+2K.
fn rt_col_ant_dec_start(k: usize) -> usize {
    12 + 2 * k
}

/// `row_anticipated_fishing_start` for the roundtrip geometry. Incoming state
/// is pinned by column bounds, so no row pins it: = min_generation_start +
/// n_op_rows = 11 + 1 = 12 (K-independent: row layout does not depend on the
/// anticipated ring's column width).
fn rt_row_ant_fishing_start(_k: usize) -> usize {
    12
}

/// Expected `num_cols` for the roundtrip geometry with anticipation K=k.
///
/// = 27+2K: the anticipated ring contributes `2*n_ant_state = 2*k` columns
/// (outgoing `commit_out` + incoming `commit_in`) plus the
/// stage-level `anticipated_decision` column, one more than the pre-ring
/// no-anticipated baseline's single combined block.
fn rt_expected_num_cols(k: usize) -> usize {
    27 + 2 * k
}

/// Expected `num_rows` for the roundtrip geometry with anticipation K=k and stage
/// `stage_idx` (`n_stages=4`, single anticipated plant). Incoming state is
/// pinned by column bounds; no row pins it.
/// Fishing row always-active (one per anticipated plant); the newest-slot
/// `anticipated_state_out_def` row is active iff `stage_idx + k < 4` (strict
/// gate); each of the `k - 1` interior ring slots gets its own ring-shift
/// definition row iff it is within the horizon-reachable cap
/// `slot < n_stages - stage_idx - 1`.
fn rt_expected_num_rows(k: usize, stage_idx: usize) -> usize {
    // base = 12 (incoming state is pinned by column bounds; no row pins it)
    let fishing = 1_usize; // always-active: 1 fishing row per anticipated plant
    let state_out_def = usize::from(stage_idx + k < 4);
    let horizon_cap = 4_usize.saturating_sub(stage_idx + 1);
    let interior_active = (0..k.saturating_sub(1))
        .filter(|&slot| slot < horizon_cap)
        .count();
    12 + fishing + state_out_def + interior_active
}

#[path = "template_integration/deficit_and_withdrawal.rs"]
mod deficit_and_withdrawal;
#[path = "template_integration/evaporation.rs"]
mod evaporation;
#[path = "template_integration/fpha_model.rs"]
mod fpha_model;
#[path = "template_integration/fpha_structure.rs"]
mod fpha_structure;
#[path = "template_integration/generic_constraints.rs"]
mod generic_constraints;
#[path = "template_integration/penalty.rs"]
mod penalty;
#[path = "template_integration/shape.rs"]
mod shape;
#[path = "template_integration/split_plant_bounds.rs"]
mod split_plant_bounds;
#[path = "template_integration/stochastic_load.rs"]
mod stochastic_load;
#[path = "template_integration/turbined_cost.rs"]
mod turbined_cost;
#[path = "template_integration/violations_par_anticipated.rs"]
mod violations_par_anticipated;
#[path = "template_integration/water_balance_coupling.rs"]
mod water_balance_coupling;
