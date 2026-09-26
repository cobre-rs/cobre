//! `hydro_inflow` mislays a traveling upstream release: it counts the full
//! upstream turbine/spillage rate as this stage's inflow and carries no term
//! for the maturing travel-time bucket, while the water-balance row it
//! mirrors defers part of that release to a later stage. Pins the defect as
//! an ignored red test against the case with no travel time, which the
//! identity holds for exactly.

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
    BoundsCountsSpec, BoundsDefaults, BusStagePenalties, ConstraintExpression, ContractBlockBounds,
    DeficitSegment, EntityId, GenericConstraint, HydroBlockBounds, HydroPenalties,
    HydroStageBounds, HydroStorage, InitialConditions, LineBlockBounds, LineStagePenalties,
    LinearTerm, NcsStagePenalties, PenaltiesCountsSpec, PenaltiesDefaults, PumpingBlockBounds,
    ResolvedBounds, ResolvedGenericConstraintBounds, ResolvedPenalties, SlackConfig, System,
    SystemBuilder, ThermalBlockBounds, ThermalStageBounds, VariableRef,
};
use cobre_io::config::{
    Config, EstimationConfig, ExportsConfig, InflowNonNegativityConfig, InflowNonNegativityMethod,
    ModelingConfig, PolicyConfig, RowSelectionConfig, SimulationConfig as IoSimulationConfig,
    SimulationSelection, StoppingMode, StoppingRuleConfig, TrainingConfig, TrainingSelection,
    TrainingSolverConfig, UpperBoundEvaluationConfig,
};
use cobre_sddp::indexer::{BlockGrid, BlockIdx};
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
const FORCED_RELEASE_M3S: f64 = 100.0;
const NON_BINDING_BOUND: f64 = 1.0e6;

fn stages() -> Vec<Stage> {
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

fn resolved_bounds(n_stages: usize) -> ResolvedBounds {
    ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 2,
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

fn resolved_penalties(n_stages: usize) -> ResolvedPenalties {
    ResolvedPenalties::new(
        &PenaltiesCountsSpec {
            n_hydros: 2,
            n_buses: 1,
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

/// A minimal upstream (`hydro 1`) -> downstream (`hydro 2`) cascade, each
/// stage two parallel blocks (`BLOCK_HOURS`), with one `GenericConstraint` on
/// `VariableRef::HydroInflow { hydro_id: DOWNSTREAM_ID, block_id: None }` at a
/// non-binding upper bound active at stage 1. `travel_time_hours` toggles the
/// arc's travel-time bucket between present and absent.
fn build_system(travel_time_hours: Option<f64>) -> System {
    let bus = make_bus(
        EntityId(BUS_ID),
        BusSpec {
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 500.0,
            }],
            excess_cost: 0.0,
            ..BusSpec::default()
        },
    );

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

    let stages = stages();
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
        .bounds(resolved_bounds(n_stages))
        .penalties(resolved_penalties(n_stages))
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

fn turbine_col(geom: &StageGeometry, hydro_pos: usize, block: usize) -> usize {
    BlockGrid::new(geom.n_blks, 0).flat(geom.turbine.start, hydro_pos, BlockIdx::new(block))
}

fn spillage_col(geom: &StageGeometry, hydro_pos: usize, block: usize) -> usize {
    BlockGrid::new(geom.n_blks, 0).flat(geom.spillage.start, hydro_pos, BlockIdx::new(block))
}

/// For downstream hydro 2 at `stage`, checks that the per-block `k`-weighted
/// `hydro_inflow` rate matches the water-balance row's own inflow-side volume,
/// for every column the balance row's inflow side reads: hydro 2's own
/// `z_inflow`, hydro 1's turbine/spillage columns in every block, and every
/// `transit_buckets_in` column the balance row actually reads.
fn assert_hydro_inflow_matches_water_balance(setup: &StudySetup, stage: usize) {
    let templates = &setup.stage_data.stage_templates;
    let tpl = &templates.templates[stage];
    let geom = &templates.geometry_per_stage[stage];
    let state_space = setup.stage_state();
    let n_blks = geom.n_blks;

    let w_row = geom.water_balance.start + DOWNSTREAM_POS;
    let g_row_start = generic_row_start(templates, stage);
    let g_row = |b: usize| g_row_start + b;

    let tau: Vec<f64> = (0..n_blks)
        .map(|b| matrix_entry(tpl, w_row, turbine_col(geom, DOWNSTREAM_POS, b)))
        .collect();

    let mut columns: Vec<usize> = vec![state_space.z_inflow.start + DOWNSTREAM_POS];
    for b in 0..n_blks {
        columns.push(turbine_col(geom, UPSTREAM_POS, b));
        columns.push(spillage_col(geom, UPSTREAM_POS, b));
    }
    let bucket_columns: Vec<usize> = state_space
        .transit_buckets_in
        .clone()
        .filter(|&c| matrix_entry(tpl, w_row, c) != 0.0)
        .collect();
    columns.extend(&bucket_columns);

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
}

#[test]
fn hydro_inflow_rows_match_the_water_balance_inflow_side_without_travel_time() {
    let setup = build_setup_in_code(build_system(None), &config());
    assert_hydro_inflow_matches_water_balance(&setup, 1);
}

#[test]
#[ignore = "hydro_inflow counts the full upstream release in the same stage and omits the maturing transit water"]
fn hydro_inflow_rows_match_the_water_balance_inflow_side_with_travel_time() {
    let setup = build_setup_in_code(build_system(Some(TRAVEL_TIME_HOURS)), &config());
    let templates = &setup.stage_data.stage_templates;
    let tpl = &templates.templates[1];
    let geom = &templates.geometry_per_stage[1];
    let state = setup.stage_state();
    let w_row = geom.water_balance.start + DOWNSTREAM_POS;

    let has_bucket_contribution = state
        .transit_buckets_in
        .clone()
        .any(|c| matrix_entry(tpl, w_row, c) != 0.0);
    assert!(
        has_bucket_contribution,
        "power guard: the water-balance row must carry a nonzero transit_buckets_in \
         entry, or the arc never routes water through a bucket"
    );

    let tau_0 = matrix_entry(tpl, w_row, turbine_col(geom, DOWNSTREAM_POS, 0));
    let same_stage_share = -matrix_entry(tpl, w_row, turbine_col(geom, UPSTREAM_POS, 0)) / tau_0;
    assert!(
        same_stage_share > 0.0 && same_stage_share < 1.0,
        "power guard: hydro 1's block-0 same-stage share must be a genuine split \
         (0, 1), got {same_stage_share}"
    );

    assert_hydro_inflow_matches_water_balance(&setup, 1);
}
