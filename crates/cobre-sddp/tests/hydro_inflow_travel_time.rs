//! `hydro_inflow` equals the water balance's inflow side: each upstream
//! release is weighted by the share the downstream balance row credits to
//! this block, and the maturing transit bucket enters as a rate. Checked
//! without travel time (every share is `1.0`), with travel time on a
//! parallel stage, and with travel time on a chronological stage. It also
//! covers the water an upstream plant that is not yet built passes straight
//! through, a travel time longer than one stage, and a plant with several
//! cells.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation
)]

use std::collections::HashMap;

use chrono::{Duration, NaiveDate};
use cobre_core::entities::hydro::HydroGenerationModel;
use cobre_core::scenario::InflowModel;
use cobre_core::temporal::{
    Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig, StageStateConfig,
};
use cobre_core::{
    BoundsCountsSpec, BoundsDefaults, Bus, BusStagePenalties, ConstraintExpression,
    ContractBlockBounds, DeficitSegment, DiversionChannel, EntityId, GenericConstraint,
    HydroBlockBounds, HydroPenalties, HydroStageBounds, HydroStorage, HydroUnitGroup,
    InitialConditions, LineBlockBounds, LineStagePenalties, LinearTerm, NcsStagePenalties,
    PenaltiesCountsSpec, PenaltiesDefaults, PumpingBlockBounds, ResolvedBounds,
    ResolvedGenericConstraintBounds, ResolvedPenalties, SlackConfig, System, SystemBuilder,
    ThermalBlockBounds, ThermalStageBounds, VariableRef,
};
use cobre_io::config::{
    Config, EstimationConfig, ExportsConfig, InflowNonNegativityConfig, InflowNonNegativityMethod,
    ModelingConfig, PolicyConfig, RowSelectionConfig, SimulationConfig as IoSimulationConfig,
    SimulationSelection, StoppingMode, StoppingRuleConfig, TrainingConfig, TrainingSelection,
    TrainingSolverConfig, UpperBoundEvaluationConfig,
};
use cobre_sddp::indexer::{BlockGrid, BlockIdx, HydroCell, HydroCellIndex, HydroSys, StateSpace};
use cobre_sddp::lp::StageGeometry;
use cobre_sddp::{StageTemplates, StudySetup};
use cobre_solver::StageTemplate;

mod common;

use common::build_setup_in_code;
use common::builders::{BusSpec, HydroSpec, StageSpec, make_bus, make_hydro, make_stage};

const BUS_ID: i32 = 1;
const UPSTREAM_ID: i32 = 1;
const DOWNSTREAM_ID: i32 = 2;
const UPSTREAM_POS: usize = 0;
const DOWNSTREAM_POS: usize = 1;
const GENERIC_CONSTRAINT_ID: i32 = 1;
const N_STAGES: usize = 2;
const BLOCK_HOURS: [f64; 2] = [300.0, 444.0];
const TRAVEL_TIME_HOURS: f64 = 372.0;
/// 1.5 stages (`744.0 * 1.5`): a travel time deep enough to widen the bucket
/// ring past depth 1.
const DEEP_TRAVEL_TIME_HOURS: f64 = 1_116.0;
const FORCED_RELEASE_M3S: f64 = 100.0;
const NON_BINDING_BOUND: f64 = 1.0e6;
const PREFILLING_UPSTREAM_A_ID: i32 = 3;
const PREFILLING_UPSTREAM_A_POS: usize = 2;
const PREFILLING_UPSTREAM_B_ID: i32 = 4;
const PREFILLING_UPSTREAM_B_POS: usize = 3;
const DIVERSION_SOURCE_ID: i32 = 5;
const DIVERSION_SOURCE_MAX_FLOW_M3S: f64 = 50.0;
const DIVERSION_SOURCE_POS_NO_CHAIN: usize = 3;
const DIVERSION_SOURCE_POS_CHAIN: usize = 4;
const PREFILLING_ENTRY_STAGE_ID: i32 = 2;

fn stages(block_mode: BlockMode) -> Vec<Stage> {
    let base = NaiveDate::from_ymd_opt(2024, 1, 1).expect("2024-01-01 is a valid date");
    (0..N_STAGES)
        .map(|i| {
            let start = base + Duration::days(31 * i64::try_from(i).unwrap_or(0));
            make_stage(
                i,
                StageSpec {
                    start_date: start,
                    end_date: start + Duration::days(31),
                    blocks: vec![
                        Block {
                            index: 0,
                            name: "B0".to_string(),
                            duration_hours: BLOCK_HOURS[0],
                        },
                        Block {
                            index: 1,
                            name: "B1".to_string(),
                            duration_hours: BLOCK_HOURS[1],
                        },
                    ],
                    block_mode,
                    state_config: StageStateConfig {
                        storage: true,
                        inflow_lags: false,
                    },
                    risk_config: StageRiskConfig::Expectation,
                    scenario_config: ScenarioSourceConfig {
                        branching_factor: 1,
                        noise_method: NoiseMethod::Saa,
                    },
                    ..StageSpec::default()
                },
            )
        })
        .collect()
}

fn zero_hydro_penalties() -> HydroPenalties {
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

fn resolved_bounds(n_stages: usize, n_hydros: usize) -> ResolvedBounds {
    ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros,
            n_thermals: 0,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages,
            k_max: 0,
        },
        &BoundsDefaults {
            hydro: HydroStageBounds {
                min_storage_hm3: 0.0,
                max_storage_hm3: 10_000.0,
                filling_min_rate_m3s: 0.0,
                water_withdrawal_m3s: 0.0,
            },
            hydro_block: HydroBlockBounds {
                max_turbined_m3s: 500.0,
                max_generation_mw: 1_000.0,
                ..HydroBlockBounds::default()
            },
            thermal: ThermalStageBounds {
                cost_per_mwh: 500.0,
            },
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

fn resolved_penalties(n_stages: usize, n_hydros: usize, n_buses: usize) -> ResolvedPenalties {
    ResolvedPenalties::new(
        &PenaltiesCountsSpec {
            n_hydros,
            n_buses,
            n_lines: 0,
            n_ncs: 0,
            n_stages,
        },
        &PenaltiesDefaults {
            hydro: zero_hydro_penalties(),
            bus: BusStagePenalties { excess_cost: 0.0 },
            line: LineStagePenalties { exchange_cost: 0.0 },
            ncs: NcsStagePenalties {
                curtailment_cost: 0.0,
            },
        },
    )
}

/// The one `GenericConstraint` on `VariableRef::HydroInflow { hydro_id:
/// DOWNSTREAM_ID, block_id: None }`, non-binding, active at stage 1.
fn downstream_inflow_constraint() -> (GenericConstraint, ResolvedGenericConstraintBounds) {
    let generic_constraint = GenericConstraint {
        id: EntityId(GENERIC_CONSTRAINT_ID),
        name: "downstream_inflow_cap".to_string(),
        description: None,
        expression: ConstraintExpression {
            terms: vec![LinearTerm::literal(
                1.0,
                VariableRef::HydroInflow {
                    hydro_id: EntityId(DOWNSTREAM_ID),
                    block_id: None,
                },
            )],
        },
        slack: SlackConfig {
            enabled: false,
            penalty: None,
        },
        bound_lower_affine: None,
        bound_upper_affine: None,
    };
    let id_map: HashMap<i32, usize> = [(GENERIC_CONSTRAINT_ID, 0)].into_iter().collect();
    let rows = vec![(
        GENERIC_CONSTRAINT_ID,
        1_i32,
        None::<i32>,
        None::<f64>,
        Some(NON_BINDING_BOUND),
    )];
    let resolved_generic_bounds = ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter());
    (generic_constraint, resolved_generic_bounds)
}

fn standard_bus() -> Bus {
    make_bus(
        EntityId(BUS_ID),
        BusSpec {
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 500.0,
            }],
            excess_cost: 0.0,
            ..BusSpec::default()
        },
    )
}

/// A minimal upstream (`hydro 1`) -> downstream (`hydro 2`) cascade, each
/// stage two parallel blocks (`BLOCK_HOURS`), with one `GenericConstraint` on
/// `VariableRef::HydroInflow { hydro_id: DOWNSTREAM_ID, block_id: None }` at a
/// non-binding upper bound active at stage 1. `travel_time_hours` toggles the
/// arc's travel-time bucket between present and absent.
fn build_system(travel_time_hours: Option<f64>, block_mode: BlockMode) -> System {
    let bus = standard_bus();

    let downstream = make_hydro(
        EntityId(DOWNSTREAM_ID),
        HydroSpec {
            bus_id: EntityId(BUS_ID),
            min_storage_hm3: 0.0,
            max_storage_hm3: 10_000.0,
            max_turbined_m3s: 500.0,
            max_generation_mw: 1_000.0,
            generation_model: HydroGenerationModel::ConstantProductivity,
            ..HydroSpec::default()
        },
    );

    let upstream = make_hydro(
        EntityId(UPSTREAM_ID),
        HydroSpec {
            bus_id: EntityId(BUS_ID),
            downstream_id: Some(EntityId(DOWNSTREAM_ID)),
            travel_time_hours,
            min_storage_hm3: 0.0,
            max_storage_hm3: 10_000.0,
            max_turbined_m3s: 500.0,
            max_generation_mw: 1_000.0,
            generation_model: HydroGenerationModel::ConstantProductivity,
            ..HydroSpec::default()
        },
    );

    let stages = stages(block_mode);
    let n_stages = stages.len();

    let inflow_models: Vec<InflowModel> = (0..n_stages)
        .map(|i| InflowModel {
            hydro_id: EntityId(UPSTREAM_ID),
            stage_id: i32::try_from(i).unwrap_or(0),
            mean_m3s: FORCED_RELEASE_M3S,
            std_m3s: 0.0,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
            annual: None,
        })
        .collect();

    let (generic_constraint, resolved_generic_bounds) = downstream_inflow_constraint();

    let system = SystemBuilder::new()
        .buses(vec![bus])
        .hydros(vec![downstream, upstream])
        .stages(stages)
        .inflow_models(inflow_models)
        .bounds(resolved_bounds(n_stages, 2))
        .penalties(resolved_penalties(n_stages, 2, 1))
        .generic_constraints(vec![generic_constraint])
        .resolved_generic_bounds(resolved_generic_bounds)
        .initial_conditions(InitialConditions {
            storage: vec![
                HydroStorage {
                    hydro_id: EntityId(DOWNSTREAM_ID),
                    value_hm3: 0.0,
                },
                HydroStorage {
                    hydro_id: EntityId(UPSTREAM_ID),
                    value_hm3: 0.0,
                },
            ],
            ..InitialConditions::default()
        })
        .build()
        .expect("hydro_inflow_travel_time: valid two-hydro cascade");

    assert_eq!(
        system.hydros()[UPSTREAM_POS].id,
        EntityId(UPSTREAM_ID),
        "the upstream plant must occupy canonical position {UPSTREAM_POS}"
    );
    assert_eq!(
        system.hydros()[DOWNSTREAM_POS].id,
        EntityId(DOWNSTREAM_ID),
        "the downstream plant must occupy canonical position {DOWNSTREAM_POS}"
    );
    system
}

/// Like [`build_system`] (`TRAVEL_TIME_HOURS`, fixed), but the upstream plant
/// declares two unit groups on two buses — the way
/// `examples/deterministic/d51-split-plant-two-bus` does — so it occupies two
/// `HydroCellIndex` cells instead of one.
fn build_two_cell_system(block_mode: BlockMode) -> System {
    let split_bus_id = EntityId(BUS_ID + 1);
    let bus = standard_bus();
    let split_bus = make_bus(split_bus_id, BusSpec::default());

    let downstream = make_hydro(
        EntityId(DOWNSTREAM_ID),
        HydroSpec {
            bus_id: EntityId(BUS_ID),
            min_storage_hm3: 0.0,
            max_storage_hm3: 10_000.0,
            max_turbined_m3s: 500.0,
            max_generation_mw: 1_000.0,
            generation_model: HydroGenerationModel::ConstantProductivity,
            ..HydroSpec::default()
        },
    );

    let upstream = make_hydro(
        EntityId(UPSTREAM_ID),
        HydroSpec {
            downstream_id: Some(EntityId(DOWNSTREAM_ID)),
            travel_time_hours: Some(TRAVEL_TIME_HOURS),
            min_storage_hm3: 0.0,
            max_storage_hm3: 10_000.0,
            max_turbined_m3s: 500.0,
            max_generation_mw: 1_000.0,
            generation_model: HydroGenerationModel::ConstantProductivity,
            unit_groups: vec![
                HydroUnitGroup {
                    id: EntityId(0),
                    name: "U1-A".to_string(),
                    bus_id: EntityId(BUS_ID),
                    min_generation_mw: 0.0,
                    max_generation_mw: 500.0,
                    min_turbined_m3s: 0.0,
                    max_turbined_m3s: 250.0,
                },
                HydroUnitGroup {
                    id: EntityId(1),
                    name: "U1-B".to_string(),
                    bus_id: split_bus_id,
                    min_generation_mw: 0.0,
                    max_generation_mw: 500.0,
                    min_turbined_m3s: 0.0,
                    max_turbined_m3s: 250.0,
                },
            ],
            ..HydroSpec::default()
        },
    );

    let stages = stages(block_mode);
    let n_stages = stages.len();

    let inflow_models: Vec<InflowModel> = (0..n_stages)
        .map(|i| InflowModel {
            hydro_id: EntityId(UPSTREAM_ID),
            stage_id: i32::try_from(i).unwrap_or(0),
            mean_m3s: FORCED_RELEASE_M3S,
            std_m3s: 0.0,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
            annual: None,
        })
        .collect();

    let (generic_constraint, resolved_generic_bounds) = downstream_inflow_constraint();

    let system = SystemBuilder::new()
        .buses(vec![bus, split_bus])
        .hydros(vec![downstream, upstream])
        .stages(stages)
        .inflow_models(inflow_models)
        .bounds(resolved_bounds(n_stages, 2))
        .penalties(resolved_penalties(n_stages, 2, 2))
        .generic_constraints(vec![generic_constraint])
        .resolved_generic_bounds(resolved_generic_bounds)
        .initial_conditions(InitialConditions {
            storage: vec![
                HydroStorage {
                    hydro_id: EntityId(DOWNSTREAM_ID),
                    value_hm3: 0.0,
                },
                HydroStorage {
                    hydro_id: EntityId(UPSTREAM_ID),
                    value_hm3: 0.0,
                },
            ],
            ..InitialConditions::default()
        })
        .build()
        .expect("hydro_inflow_travel_time: valid two-cell upstream cascade");

    assert_eq!(
        system.hydros()[UPSTREAM_POS].id,
        EntityId(UPSTREAM_ID),
        "the upstream plant must occupy canonical position {UPSTREAM_POS}"
    );
    assert_eq!(
        system.hydros()[DOWNSTREAM_POS].id,
        EntityId(DOWNSTREAM_ID),
        "the downstream plant must occupy canonical position {DOWNSTREAM_POS}"
    );
    system
}

fn prefilling_hydro_defaults() -> HydroSpec {
    HydroSpec {
        bus_id: EntityId(BUS_ID),
        min_storage_hm3: 0.0,
        max_storage_hm3: 10_000.0,
        max_turbined_m3s: 500.0,
        max_generation_mw: 1_000.0,
        generation_model: HydroGenerationModel::ConstantProductivity,
        ..HydroSpec::default()
    }
}

/// A source `W` -> `H` cascade routed through one `PreFilling` plant `U_a`
/// (`chain: false`, `W -> U_a -> H`) or two (`chain: true`,
/// `W -> U_a -> U_b -> H`), plus a diversion source `S` feeding `U_a`. Every
/// `U` has `entry_stage_id: Some(PREFILLING_ENTRY_STAGE_ID)`, so it is
/// `PreFilling` at both study stages. `travel_time_hours` sits on the one arc
/// into `H` (`U_a`'s without `chain`, `U_b`'s with it); every other arc is
/// lag-free, because `cobre-io` travel-time rule 12 rejects a lagged arc into
/// a plant that is not yet operating.
fn build_prefilling_system(
    chain: bool,
    travel_time_hours: Option<f64>,
    block_mode: BlockMode,
) -> System {
    let bus = standard_bus();

    let w = make_hydro(
        EntityId(UPSTREAM_ID),
        HydroSpec {
            downstream_id: Some(EntityId(PREFILLING_UPSTREAM_A_ID)),
            ..prefilling_hydro_defaults()
        },
    );
    let h = make_hydro(EntityId(DOWNSTREAM_ID), prefilling_hydro_defaults());
    let upstream_a = make_hydro(
        EntityId(PREFILLING_UPSTREAM_A_ID),
        HydroSpec {
            downstream_id: Some(if chain {
                EntityId(PREFILLING_UPSTREAM_B_ID)
            } else {
                EntityId(DOWNSTREAM_ID)
            }),
            travel_time_hours: if chain { None } else { travel_time_hours },
            entry_stage_id: Some(PREFILLING_ENTRY_STAGE_ID),
            ..prefilling_hydro_defaults()
        },
    );
    let diversion_source = make_hydro(
        EntityId(DIVERSION_SOURCE_ID),
        HydroSpec {
            diversion: Some(DiversionChannel {
                downstream_id: EntityId(PREFILLING_UPSTREAM_A_ID),
                max_flow_m3s: DIVERSION_SOURCE_MAX_FLOW_M3S,
            }),
            ..prefilling_hydro_defaults()
        },
    );

    let mut hydros = vec![w, h, upstream_a];
    if chain {
        hydros.push(make_hydro(
            EntityId(PREFILLING_UPSTREAM_B_ID),
            HydroSpec {
                downstream_id: Some(EntityId(DOWNSTREAM_ID)),
                travel_time_hours,
                entry_stage_id: Some(PREFILLING_ENTRY_STAGE_ID),
                ..prefilling_hydro_defaults()
            },
        ));
    }
    hydros.push(diversion_source);
    let n_hydros = hydros.len();

    let stages = stages(block_mode);
    let n_stages = stages.len();

    let mut inflow_models = Vec::new();
    for hydro_id in [UPSTREAM_ID, PREFILLING_UPSTREAM_A_ID] {
        for i in 0..n_stages {
            inflow_models.push(InflowModel {
                hydro_id: EntityId(hydro_id),
                stage_id: i32::try_from(i).unwrap_or(0),
                mean_m3s: FORCED_RELEASE_M3S,
                std_m3s: 0.0,
                ar_coefficients: vec![],
                residual_std_ratio: 1.0,
                annual: None,
            });
        }
    }

    let (generic_constraint, resolved_generic_bounds) = downstream_inflow_constraint();
    let storage = hydros
        .iter()
        .map(|h| HydroStorage {
            hydro_id: h.id,
            value_hm3: 0.0,
        })
        .collect();

    let system = SystemBuilder::new()
        .buses(vec![bus])
        .hydros(hydros)
        .stages(stages)
        .inflow_models(inflow_models)
        .bounds(resolved_bounds(n_stages, n_hydros))
        .penalties(resolved_penalties(n_stages, n_hydros, 1))
        .generic_constraints(vec![generic_constraint])
        .resolved_generic_bounds(resolved_generic_bounds)
        .initial_conditions(InitialConditions {
            storage,
            ..InitialConditions::default()
        })
        .build()
        .expect("hydro_inflow_travel_time: valid pre-filling cascade");

    assert_eq!(
        system.hydros()[UPSTREAM_POS].id,
        EntityId(UPSTREAM_ID),
        "the source plant must occupy canonical position {UPSTREAM_POS}"
    );
    assert_eq!(
        system.hydros()[DOWNSTREAM_POS].id,
        EntityId(DOWNSTREAM_ID),
        "the downstream plant must occupy canonical position {DOWNSTREAM_POS}"
    );
    assert_eq!(
        system.hydros()[PREFILLING_UPSTREAM_A_POS].id,
        EntityId(PREFILLING_UPSTREAM_A_ID),
        "the first pre-filling upstream plant must occupy canonical position \
         {PREFILLING_UPSTREAM_A_POS}"
    );
    if chain {
        assert_eq!(
            system.hydros()[PREFILLING_UPSTREAM_B_POS].id,
            EntityId(PREFILLING_UPSTREAM_B_ID),
            "the second pre-filling upstream plant must occupy canonical position \
             {PREFILLING_UPSTREAM_B_POS}"
        );
    }
    let diversion_pos = if chain {
        DIVERSION_SOURCE_POS_CHAIN
    } else {
        DIVERSION_SOURCE_POS_NO_CHAIN
    };
    assert_eq!(
        system.hydros()[diversion_pos].id,
        EntityId(DIVERSION_SOURCE_ID),
        "the diversion source must occupy canonical position {diversion_pos}"
    );
    system
}

fn config() -> Config {
    Config {
        schema: None,
        modeling: ModelingConfig {
            inflow_non_negativity: InflowNonNegativityConfig {
                method: InflowNonNegativityMethod::Penalty,
            },
            cost_scale_factor: Some(1.0),
        },
        training: TrainingConfig {
            enabled: true,
            tree_seed: Some(42),
            stopping_rules: Some(vec![StoppingRuleConfig::IterationLimit { limit: 1 }]),
            stopping_mode: StoppingMode::Any,
            cut_selection: RowSelectionConfig::default(),
            solver: TrainingSolverConfig::default(),
            parallelism: cobre_io::config::ParallelismConfig::default(),
            scenario_source: None,
            selection: Some(TrainingSelection::Sampled { forward_passes: 1 }),
        },
        upper_bound_evaluation: UpperBoundEvaluationConfig::default(),
        policy: PolicyConfig::default(),
        simulation: IoSimulationConfig {
            enabled: true,
            io_channel_capacity: 16,
            selection: Some(SimulationSelection::Sampled { num_scenarios: 1 }),
            ..IoSimulationConfig::default()
        },
        exports: ExportsConfig::default(),
        estimation: EstimationConfig::default(),
    }
}

/// The physical (unscaled) coefficient at `[row, col]`: `postprocess_templates`
/// prescales every stored value by `col_scale[col] * row_scale[row]`
/// (`D_r * A * D_c`), so a raw CSC read compares apples to oranges across
/// columns with different scale factors.
fn matrix_entry(tpl: &StageTemplate, row: usize, col: usize) -> f64 {
    let start = tpl.col_starts[col] as usize;
    let end = tpl.col_starts[col + 1] as usize;
    let stored = tpl.row_indices[start..end]
        .iter()
        .zip(&tpl.values[start..end])
        .find(|&(&r, _)| r as usize == row)
        .map_or(0.0, |(_, &v)| v);
    let col_scale = tpl.col_scale.get(col).copied().unwrap_or(1.0);
    let row_scale = tpl.row_scale.get(row).copied().unwrap_or(1.0);
    stored / (col_scale * row_scale)
}

/// The first `hydro_inflow` row for `stage`: generic-constraint rows are the
/// last row family the builder allocates (`layout.rows.row_generic_start =
/// row.pos()` immediately before `enumerate_generic_constraint_rows`, and no
/// row family is allocated after it), so they occupy the trailing
/// `generic_constraint_row_entries[stage].len()` rows of `[0, num_rows)`.
fn generic_row_start(templates: &StageTemplates, stage: usize) -> usize {
    let tpl = &templates.templates[stage];
    let n_generic = templates.generic_constraint_row_entries[stage].len();
    tpl.num_rows - n_generic
}

/// Every one of the plant's cells' turbine columns for `block`, `cells` in
/// declaration order (never derived from `geom.turbine.start` arithmetic).
fn turbine_cols(geom: &StageGeometry, cells: &[usize], block: usize) -> Vec<usize> {
    cells
        .iter()
        .map(|&c| geom.turbine_col(HydroCell::new(c), BlockIdx::new(block)))
        .collect()
}

fn spillage_col(geom: &StageGeometry, hydro_pos: usize, block: usize) -> usize {
    BlockGrid::new(geom.n_blks, 0).flat(geom.spillage.start, hydro_pos, BlockIdx::new(block))
}

fn diversion_col(geom: &StageGeometry, hydro_pos: usize, block: usize) -> usize {
    BlockGrid::new(geom.n_blks, 0).flat(geom.diversion.start, hydro_pos, BlockIdx::new(block))
}

fn inflow_columns(
    geom: &StageGeometry,
    state: &StateSpace,
    z_plants: &[usize],
    turbine_cells: &[usize],
    spillage_plants: &[usize],
    diversion_plants: &[usize],
) -> Vec<usize> {
    let mut columns: Vec<usize> = z_plants.iter().map(|&p| state.z_inflow.start + p).collect();
    for b in 0..geom.n_blks {
        columns.extend(turbine_cols(geom, turbine_cells, b));
        for &p in spillage_plants {
            columns.push(spillage_col(geom, p, b));
        }
        for &p in diversion_plants {
            columns.push(diversion_col(geom, p, b));
        }
    }
    columns
}

/// Every nonzero column of generic constraint row `row` must be one of
/// `allowed` — a stray pair breaks the closed-column-set contract this test
/// pins.
fn assert_generic_row_columns_are_closed(tpl: &StageTemplate, row: usize, allowed: &[usize]) {
    for col in 0..tpl.num_cols {
        let entry = matrix_entry(tpl, row, col);
        assert!(
            entry == 0.0 || allowed.contains(&col),
            "row {row}: unexpected nonzero column {col} (value {entry}) outside the checked set"
        );
    }
}

/// For downstream hydro 2 at `stage`, checks that the per-block `k`-weighted
/// `hydro_inflow` rate matches the water-balance row's own inflow-side volume,
/// for every column in `inflow_columns` and every `transit_buckets_in` column
/// the balance row actually reads. Every other column of each generic row
/// must be zero (the closed-set contract).
fn assert_hydro_inflow_matches_water_balance(
    setup: &StudySetup,
    stage: usize,
    downstream_cell: usize,
    inflow_columns: &[usize],
) {
    let templates = &setup.inputs.stage_data.stage_templates;
    let tpl = &templates.templates[stage];
    let geom = &templates.geometry_per_stage[stage];
    let state_space = setup.stage_state();
    let n_blks = geom.n_blks;

    let w_row = geom.water_balance_row(HydroSys::new(DOWNSTREAM_POS), BlockIdx::new(0));
    let g_row_start = generic_row_start(templates, stage);
    let g_row = |b: usize| g_row_start + b;

    let tau: Vec<f64> = (0..n_blks)
        .map(|b| matrix_entry(tpl, w_row, turbine_cols(geom, &[downstream_cell], b)[0]))
        .collect();

    let mut columns: Vec<usize> = inflow_columns.to_vec();
    columns.extend(
        state_space
            .transit_buckets_in
            .clone()
            .filter(|&c| matrix_entry(tpl, w_row, c) != 0.0),
    );

    for &c in &columns {
        let lhs: f64 = (0..n_blks)
            .map(|b| tau[b] * matrix_entry(tpl, g_row(b), c))
            .sum();
        let w_entry = matrix_entry(tpl, w_row, c);
        let rhs = -w_entry;
        let scale = 1.0_f64.max(w_entry.abs());
        assert!(
            (lhs - rhs).abs() <= 1e-9 * scale,
            "stage {stage} column {c}: hydro_inflow sum={lhs} does not match \
             -water_balance[{w_row}, {c}]={rhs} (tol {})",
            1e-9 * scale
        );
    }

    for b in 0..n_blks {
        assert_generic_row_columns_are_closed(tpl, g_row(b), &columns);
    }
}

/// For downstream hydro 2 at `stage`, checks each block's own chronological
/// water-balance row against `hydro_inflow`'s per-block generic row: for
/// every column in `inflow_columns`, `τ_b · g[b, c] == -w_b[c]`. Every other
/// column of each generic row must be zero (the closed-set contract).
fn assert_hydro_inflow_matches_each_chronological_water_balance_row(
    setup: &StudySetup,
    stage: usize,
    downstream_cell: usize,
    inflow_columns: &[usize],
) {
    let templates = &setup.inputs.stage_data.stage_templates;
    let tpl = &templates.templates[stage];
    let geom = &templates.geometry_per_stage[stage];
    let state_space = setup.stage_state();
    let n_blks = geom.n_blks;

    let g_row_start = generic_row_start(templates, stage);
    let g_row = |b: usize| g_row_start + b;
    let w_row = |b: usize| geom.water_balance_row(HydroSys::new(DOWNSTREAM_POS), BlockIdx::new(b));

    let mut columns: Vec<usize> = inflow_columns.to_vec();
    columns.extend(
        state_space
            .transit_buckets_in
            .clone()
            .filter(|&c| (0..n_blks).any(|b| matrix_entry(tpl, w_row(b), c) != 0.0)),
    );

    for b in 0..n_blks {
        let row_w = w_row(b);
        let row_g = g_row(b);
        let tau_b = matrix_entry(tpl, row_w, turbine_cols(geom, &[downstream_cell], b)[0]);

        for &c in &columns {
            let lhs = tau_b * matrix_entry(tpl, row_g, c);
            let w_entry = matrix_entry(tpl, row_w, c);
            let rhs = -w_entry;
            let scale = 1.0_f64.max(w_entry.abs());
            assert!(
                (lhs - rhs).abs() <= 1e-9 * scale,
                "stage {stage} block {b} column {c}: hydro_inflow={lhs} does not match \
                 -water_balance[{row_w}, {c}]={rhs} (tol {})",
                1e-9 * scale
            );
        }

        assert_generic_row_columns_are_closed(tpl, row_g, &columns);
    }
}

#[test]
fn hydro_inflow_rows_match_the_water_balance_inflow_side_without_travel_time() {
    let setup = build_setup_in_code(build_system(None, BlockMode::Parallel), &config());
    let geom = &setup.inputs.stage_data.stage_templates.geometry_per_stage[1];
    let columns = inflow_columns(
        geom,
        setup.stage_state(),
        &[DOWNSTREAM_POS],
        &[UPSTREAM_POS],
        &[UPSTREAM_POS],
        &[],
    );
    assert_hydro_inflow_matches_water_balance(&setup, 1, DOWNSTREAM_POS, &columns);
}

#[test]
fn hydro_inflow_rows_match_the_water_balance_inflow_side_with_travel_time() {
    let setup = build_setup_in_code(
        build_system(Some(TRAVEL_TIME_HOURS), BlockMode::Parallel),
        &config(),
    );
    let templates = &setup.inputs.stage_data.stage_templates;
    let tpl = &templates.templates[1];
    let geom = &templates.geometry_per_stage[1];
    let state = setup.stage_state();
    let w_row = geom.water_balance_row(HydroSys::new(DOWNSTREAM_POS), BlockIdx::new(0));

    let has_bucket_contribution = state
        .transit_buckets_in
        .clone()
        .any(|c| matrix_entry(tpl, w_row, c) != 0.0);
    assert!(
        has_bucket_contribution,
        "power guard: the water-balance row must carry a nonzero transit_buckets_in \
         entry, or the arc never routes water through a bucket"
    );

    let tau_0 = matrix_entry(tpl, w_row, turbine_cols(geom, &[DOWNSTREAM_POS], 0)[0]);
    let same_stage_share =
        -matrix_entry(tpl, w_row, turbine_cols(geom, &[UPSTREAM_POS], 0)[0]) / tau_0;
    assert!(
        same_stage_share > 0.0 && same_stage_share < 1.0,
        "power guard: hydro 1's block-0 same-stage share must be a genuine split \
         (0, 1), got {same_stage_share}"
    );

    let columns = inflow_columns(
        geom,
        state,
        &[DOWNSTREAM_POS],
        &[UPSTREAM_POS],
        &[UPSTREAM_POS],
        &[],
    );
    assert_hydro_inflow_matches_water_balance(&setup, 1, DOWNSTREAM_POS, &columns);
}

/// A release in block 0 (0-300h) arrives at 372-672h, all of it within the
/// same stage's block 1. A block-1 release (300-744h local) arrives at
/// 672-1116h, so part of it lands past the stage and matures into stage 1 —
/// exercising the crossing deposit into the bucket, then the bucket's
/// per-block arrival density on the receiving side.
#[test]
fn hydro_inflow_rows_match_each_chronological_water_balance_row_with_travel_time() {
    let setup = build_setup_in_code(
        build_system(Some(TRAVEL_TIME_HOURS), BlockMode::Chronological),
        &config(),
    );
    let templates = &setup.inputs.stage_data.stage_templates;
    let tpl = &templates.templates[1];
    let geom = &templates.geometry_per_stage[1];
    let state = setup.stage_state();

    let w_row_1 = geom.water_balance_row(HydroSys::new(DOWNSTREAM_POS), BlockIdx::new(1));
    assert!(
        matrix_entry(tpl, w_row_1, turbine_cols(geom, &[UPSTREAM_POS], 0)[0]) != 0.0,
        "power guard: hydro 1's block-0 turbine column must have a nonzero entry \
         on downstream balance row 1"
    );

    let has_bucket_contribution = (0..geom.n_blks).any(|b| {
        let w_row = geom.water_balance_row(HydroSys::new(DOWNSTREAM_POS), BlockIdx::new(b));
        state
            .transit_buckets_in
            .clone()
            .any(|c| matrix_entry(tpl, w_row, c) != 0.0)
    });
    assert!(
        has_bucket_contribution,
        "power guard: some transit_buckets_in column must be nonzero on some \
         downstream balance row"
    );

    let columns = inflow_columns(
        geom,
        state,
        &[DOWNSTREAM_POS],
        &[UPSTREAM_POS],
        &[UPSTREAM_POS],
        &[],
    );
    assert_hydro_inflow_matches_each_chronological_water_balance_row(
        &setup,
        1,
        DOWNSTREAM_POS,
        &columns,
    );
}

#[test]
fn hydro_inflow_rows_count_a_prefilling_upstream_plants_water_on_a_parallel_stage() {
    let setup = build_setup_in_code(
        build_prefilling_system(false, Some(TRAVEL_TIME_HOURS), BlockMode::Parallel),
        &config(),
    );
    let templates = &setup.inputs.stage_data.stage_templates;
    let tpl = &templates.templates[1];
    let geom = &templates.geometry_per_stage[1];
    let state = setup.stage_state();
    let w_row = geom.water_balance_row(HydroSys::new(DOWNSTREAM_POS), BlockIdx::new(0));

    assert!(
        matrix_entry(tpl, w_row, state.z_inflow.start + PREFILLING_UPSTREAM_A_POS) != 0.0,
        "power guard: the pre-filling upstream plant's short-circuit route to hydro 2 \
         must be live"
    );
    let tau_0 = matrix_entry(tpl, w_row, turbine_cols(geom, &[DOWNSTREAM_POS], 0)[0]);
    assert!(
        (matrix_entry(tpl, w_row, turbine_cols(geom, &[UPSTREAM_POS], 0)[0]) + tau_0).abs() < 1e-9,
        "power guard: the source plant's block-0 release must reach hydro 2 whole, in its \
         own block"
    );

    let columns = inflow_columns(
        geom,
        state,
        &[DOWNSTREAM_POS, PREFILLING_UPSTREAM_A_POS],
        &[PREFILLING_UPSTREAM_A_POS, UPSTREAM_POS],
        &[PREFILLING_UPSTREAM_A_POS, UPSTREAM_POS],
        &[DIVERSION_SOURCE_POS_NO_CHAIN],
    );
    assert_hydro_inflow_matches_water_balance(&setup, 1, DOWNSTREAM_POS, &columns);
}

#[test]
fn hydro_inflow_rows_count_a_prefilling_upstream_plants_water_on_each_chronological_block() {
    let setup = build_setup_in_code(
        build_prefilling_system(false, Some(TRAVEL_TIME_HOURS), BlockMode::Chronological),
        &config(),
    );
    let templates = &setup.inputs.stage_data.stage_templates;
    let tpl = &templates.templates[1];
    let geom = &templates.geometry_per_stage[1];
    let state = setup.stage_state();
    let w_row = geom.water_balance_row(HydroSys::new(DOWNSTREAM_POS), BlockIdx::new(0));

    assert!(
        matrix_entry(tpl, w_row, state.z_inflow.start + PREFILLING_UPSTREAM_A_POS) != 0.0,
        "power guard: the pre-filling upstream plant's short-circuit route to hydro 2 \
         must be live"
    );
    let tau_0 = matrix_entry(tpl, w_row, turbine_cols(geom, &[DOWNSTREAM_POS], 0)[0]);
    assert!(
        (matrix_entry(tpl, w_row, turbine_cols(geom, &[UPSTREAM_POS], 0)[0]) + tau_0).abs() < 1e-9,
        "power guard: the source plant's block-0 release must reach hydro 2 whole, in its \
         own block"
    );

    let columns = inflow_columns(
        geom,
        state,
        &[DOWNSTREAM_POS, PREFILLING_UPSTREAM_A_POS],
        &[PREFILLING_UPSTREAM_A_POS, UPSTREAM_POS],
        &[PREFILLING_UPSTREAM_A_POS, UPSTREAM_POS],
        &[DIVERSION_SOURCE_POS_NO_CHAIN],
    );
    assert_hydro_inflow_matches_each_chronological_water_balance_row(
        &setup,
        1,
        DOWNSTREAM_POS,
        &columns,
    );
}

#[test]
fn hydro_inflow_rows_count_a_two_plant_prefilling_chain_on_a_parallel_stage() {
    let setup = build_setup_in_code(
        build_prefilling_system(true, None, BlockMode::Parallel),
        &config(),
    );
    let templates = &setup.inputs.stage_data.stage_templates;
    let tpl = &templates.templates[1];
    let geom = &templates.geometry_per_stage[1];
    let state = setup.stage_state();
    let w_row = geom.water_balance_row(HydroSys::new(DOWNSTREAM_POS), BlockIdx::new(0));

    assert!(
        matrix_entry(tpl, w_row, state.z_inflow.start + PREFILLING_UPSTREAM_A_POS) != 0.0,
        "power guard: the pre-filling upstream plant's short-circuit route to hydro 2 \
         must be live"
    );
    let tau_0 = matrix_entry(tpl, w_row, turbine_cols(geom, &[DOWNSTREAM_POS], 0)[0]);
    assert!(
        (matrix_entry(tpl, w_row, turbine_cols(geom, &[UPSTREAM_POS], 0)[0]) + tau_0).abs() < 1e-9,
        "power guard: the source plant's block-0 release must reach hydro 2 whole, in its \
         own block"
    );

    let columns = inflow_columns(
        geom,
        state,
        &[
            DOWNSTREAM_POS,
            PREFILLING_UPSTREAM_A_POS,
            PREFILLING_UPSTREAM_B_POS,
        ],
        &[
            PREFILLING_UPSTREAM_B_POS,
            PREFILLING_UPSTREAM_A_POS,
            UPSTREAM_POS,
        ],
        &[
            PREFILLING_UPSTREAM_B_POS,
            PREFILLING_UPSTREAM_A_POS,
            UPSTREAM_POS,
        ],
        &[DIVERSION_SOURCE_POS_CHAIN],
    );
    assert_hydro_inflow_matches_water_balance(&setup, 1, DOWNSTREAM_POS, &columns);
}

/// A travel time deeper than one stage (`DEEP_TRAVEL_TIME_HOURS`, 1.5
/// stages): the bucket ring widens past depth 1, and the same per-block
/// `k`-weighted identity still holds over every bucket slot.
#[test]
fn hydro_inflow_rows_match_the_water_balance_inflow_side_with_a_two_stage_travel_time() {
    let setup = build_setup_in_code(
        build_system(Some(DEEP_TRAVEL_TIME_HOURS), BlockMode::Parallel),
        &config(),
    );
    let state = setup.stage_state();
    assert_eq!(
        state.transit_bucket_column_order.len(),
        2,
        "power guard: a travel time of 1.5 stages must widen the bucket ring to depth 2"
    );

    let templates = &setup.inputs.stage_data.stage_templates;
    let tpl = &templates.templates[1];
    let geom = &templates.geometry_per_stage[1];
    let w_row = geom.water_balance_row(HydroSys::new(DOWNSTREAM_POS), BlockIdx::new(0));
    let bucket_cols: Vec<usize> = state.transit_buckets_in.clone().collect();
    assert_eq!(
        bucket_cols.len(),
        2,
        "power guard: one arc at depth 2 must yield exactly 2 bucket columns"
    );
    assert!(
        matrix_entry(tpl, w_row, bucket_cols[0]) != 0.0,
        "power guard: the maturing (slot-0) bucket column must carry the water-balance \
         entry, not a deeper, not-yet-mature slot"
    );
    assert!(
        matrix_entry(tpl, w_row, bucket_cols[1]) == 0.0,
        "power guard: the not-yet-mature (slot-1) bucket column must carry no \
         water-balance entry"
    );

    let columns = inflow_columns(
        geom,
        state,
        &[DOWNSTREAM_POS],
        &[UPSTREAM_POS],
        &[UPSTREAM_POS],
        &[],
    );
    assert_hydro_inflow_matches_water_balance(&setup, 1, DOWNSTREAM_POS, &columns);
}

/// A multi-cell upstream plant (`build_two_cell_system`): each of its two
/// bus-split cells carries its own turbine column, and the closed-column-set
/// identity holds independently at each cell's column (an arc's release
/// weight is replicated onto every cell, never apportioned).
#[test]
fn hydro_inflow_rows_match_the_water_balance_inflow_side_with_a_two_cell_upstream_plant() {
    let system = build_two_cell_system(BlockMode::Parallel);
    let cell_index = HydroCellIndex::build(system.hydros());
    let upstream_cells: Vec<usize> = cell_index.cells_of(HydroSys::new(UPSTREAM_POS)).collect();
    let downstream_cell = cell_index
        .cells_of(HydroSys::new(DOWNSTREAM_POS))
        .next()
        .expect("power guard: the downstream plant must have a cell");
    assert_eq!(
        upstream_cells.len(),
        2,
        "power guard: the upstream plant must have two turbine columns per block \
         (split across two buses)"
    );

    let setup = build_setup_in_code(system, &config());
    let geom = &setup.inputs.stage_data.stage_templates.geometry_per_stage[1];
    let state = setup.stage_state();

    let columns = inflow_columns(
        geom,
        state,
        &[DOWNSTREAM_POS],
        &upstream_cells,
        &[UPSTREAM_POS],
        &[],
    );
    assert_hydro_inflow_matches_water_balance(&setup, 1, downstream_cell, &columns);
}
