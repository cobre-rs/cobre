//! `deficit_and_withdrawal` section tests.

use super::*;

#[test]
#[allow(clippy::too_many_lines)]
fn test_multi_segment_deficit_column_count() {
    use chrono::NaiveDate;
    use cobre_core::scenario::LoadModel;
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, StageRiskConfig, StageStateConfig,
    };

    let bus0 = make_bus(
        EntityId(1),
        BusSpec {
            name: "Bus0".to_string(),
            deficit_segments: vec![
                DeficitSegment {
                    depth_mw: Some(10.0),
                    cost_per_mwh: 100.0,
                },
                DeficitSegment {
                    depth_mw: Some(20.0),
                    cost_per_mwh: 200.0,
                },
                DeficitSegment {
                    depth_mw: None,
                    cost_per_mwh: 5000.0,
                },
            ],
            ..Default::default()
        },
    );
    let bus1 = make_bus(
        EntityId(2),
        BusSpec {
            name: "Bus1".to_string(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 1000.0,
            }],
            ..Default::default()
        },
    );

    let load_models = vec![
        LoadModel {
            bus_id: EntityId(1),
            stage_id: 0,
            mean_mw: 0.0,
            std_mw: 0.0,
        },
        LoadModel {
            bus_id: EntityId(2),
            stage_id: 0,
            mean_mw: 0.0,
            std_mw: 0.0,
        },
    ];

    let stage = make_stage(
        0,
        StageSpec {
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            season_id: None,
            blocks: vec![
                Block {
                    index: 0,
                    name: "B0".to_string(),
                    duration_hours: 360.0,
                },
                Block {
                    index: 1,
                    name: "B1".to_string(),
                    duration_hours: 360.0,
                },
            ],
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
            n_buses: 2,
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
        .buses(vec![bus0, bus1])
        .stages(vec![stage])
        .load_models(load_models)
        .bounds(bounds)
        .penalties(penalties)
        .build()
        .expect("2-bus 2-block system: valid");

    let result = build_stage_templates_resolving_layout(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("build ok");

    let t = &result.templates[0];

    // N=0, L=0 → theta=0, decision_start=1
    // No thermals, no lines → col_deficit_start = 1
    // B=2, S_max=3, K=2 → deficit region = 2*3*2 = 12 columns → col_excess_start = 13
    // excess region = B*K = 2*2 = 4 → num_cols = 13 + 4 = 17
    let col_deficit_start = 1_usize;
    let max_deficit_segments = 3_usize;
    let n_blks = 2_usize;
    let n_buses = 2_usize;
    let deficit_region = n_buses * max_deficit_segments * n_blks;
    assert_eq!(
        deficit_region, 12,
        "deficit region must be B*S_max*K = 2*3*2 = 12"
    );
    let col_excess_start = col_deficit_start + deficit_region;
    let excess_region = n_buses * n_blks;
    let expected_num_cols = col_excess_start + excess_region;
    assert_eq!(
        t.num_cols, expected_num_cols,
        "num_cols must include expanded deficit region"
    );
}

#[test]
fn test_multi_segment_deficit_bounds_and_objective() {
    let bus = make_bus(
        EntityId(1),
        BusSpec {
            name: "Bus0".to_string(),
            deficit_segments: vec![
                DeficitSegment {
                    depth_mw: Some(10.0),
                    cost_per_mwh: 500.0,
                },
                DeficitSegment {
                    depth_mw: None,
                    cost_per_mwh: 5000.0,
                },
            ],
            ..Default::default()
        },
    );

    let block_hours = 730.0_f64;
    let system = multi_segment_system(vec![bus], block_hours);

    let result = build_stage_templates_resolving_layout(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("build ok");

    let t = &result.templates[0];

    // N=0 → theta=0, decision_start=1, no thermals/lines
    // col_deficit_start = 1
    // B=1, S_max=2, K=1 → deficit region = 1*2*1 = 2
    // seg0 col = 1 + 0*2*1 + 0*1 + 0 = 1
    // seg1 col = 1 + 0*2*1 + 1*1 + 0 = 2
    let col_seg0 = 1_usize;
    let col_seg1 = 2_usize;

    assert_eq!(
        t.col_upper[col_seg0], 10.0,
        "segment 0 upper bound must equal depth_mw = 10.0"
    );
    assert!(
        t.col_upper[col_seg1].is_infinite() && t.col_upper[col_seg1] > 0.0,
        "segment 1 upper bound must be +infinity (unbounded final segment)"
    );
    assert!(
        (t.objective[col_seg0] - 500.0 * block_hours / COST_SCALE_FACTOR).abs() < 1e-12,
        "segment 0 objective must be cost * block_hours / COST_SCALE_FACTOR = {} but got {}",
        500.0 * block_hours / COST_SCALE_FACTOR,
        t.objective[col_seg0]
    );
    assert!(
        (t.objective[col_seg1] - 5000.0 * block_hours / COST_SCALE_FACTOR).abs() < 1e-12,
        "segment 1 objective must be cost * block_hours / COST_SCALE_FACTOR = {} but got {}",
        5000.0 * block_hours / COST_SCALE_FACTOR,
        t.objective[col_seg1]
    );
}

#[test]
fn test_single_segment_backward_compat() {
    let cost = 1000.0_f64;
    let block_hours = 744.0_f64;

    let bus = make_bus(
        EntityId(1),
        BusSpec {
            name: "Bus0".to_string(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: cost,
            }],
            ..Default::default()
        },
    );

    let system = multi_segment_system(vec![bus], block_hours);

    let result = build_stage_templates_resolving_layout(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("build ok");

    let t = &result.templates[0];

    // N=0 → theta=0, decision_start=1, col_deficit_start=1
    // B=1, S_max=1, K=1 → 1 deficit column at index 1
    let col_def = 1_usize;

    assert!(
        t.col_upper[col_def].is_infinite() && t.col_upper[col_def] > 0.0,
        "single segment must be unbounded (None depth_mw)"
    );
    assert!(
        (t.objective[col_def] - cost * block_hours / COST_SCALE_FACTOR).abs() < 1e-12,
        "single-segment objective must be cost * block_hours / COST_SCALE_FACTOR"
    );

    // Excess column immediately follows deficit (S_max=1, B=1, K=1 → excess at col 2)
    let col_exc = 2_usize;
    assert!(
        t.col_upper[col_exc].is_infinite() && t.col_upper[col_exc] > 0.0,
        "excess column must be unbounded"
    );
}

#[test]
fn test_multi_segment_deficit_load_balance_coefficients() {
    let bus = make_bus(
        EntityId(1),
        BusSpec {
            name: "Bus0".to_string(),
            deficit_segments: vec![
                DeficitSegment {
                    depth_mw: Some(10.0),
                    cost_per_mwh: 500.0,
                },
                DeficitSegment {
                    depth_mw: None,
                    cost_per_mwh: 5000.0,
                },
            ],
            ..Default::default()
        },
    );

    let system = multi_segment_system(vec![bus], 730.0);

    let result = build_stage_templates_resolving_layout(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("build ok");

    let t = &result.templates[0];

    // N=0, no thermals/lines → col_deficit_start = 1
    // B=1, S_max=2, K=1 → seg0 at col 1, seg1 at col 2
    let col_seg0 = 1_usize;
    let col_seg1 = 2_usize;

    // N=0 hydros → row_load_balance_start = 0; bus 0, block 0 → row 0
    let load_balance_row = 0_usize;

    let entries_seg0 = entries_for_col(t, col_seg0);
    assert_eq!(
        entries_seg0.len(),
        1,
        "deficit segment 0 column must have exactly 1 CSC entry (load-balance row), got {entries_seg0:?}"
    );
    assert_eq!(
        entries_seg0[0].0, load_balance_row,
        "deficit segment 0 entry must be at the load-balance row {load_balance_row}, got row {}",
        entries_seg0[0].0
    );
    assert!(
        (entries_seg0[0].1 - 1.0).abs() < 1e-12,
        "deficit segment 0 load-balance coefficient must be +1.0, got {}",
        entries_seg0[0].1
    );

    let entries_seg1 = entries_for_col(t, col_seg1);
    assert_eq!(
        entries_seg1.len(),
        1,
        "deficit segment 1 column must have exactly 1 CSC entry (load-balance row), got {entries_seg1:?}"
    );
    assert_eq!(
        entries_seg1[0].0, load_balance_row,
        "deficit segment 1 entry must be at the load-balance row {load_balance_row}, got row {}",
        entries_seg1[0].0
    );
    assert!(
        (entries_seg1[0].1 - 1.0).abs() < 1e-12,
        "deficit segment 1 load-balance coefficient must be +1.0, got {}",
        entries_seg1[0].1
    );
}

/// Water balance RHS = -(ζ * `water_withdrawal_m3s`); the realized inflow
/// couples through the z-inflow column rather than a template RHS term. For
/// withdrawal=10 the RHS is `-(744 * 3600/1_000_000 * 10) = -2.6784`.
#[test]
fn withdrawal_rhs_subtracted_from_water_balance() {
    let withdrawal = 10.0_f64;
    let system = one_hydro_system_with_withdrawal(1, 0, withdrawal, 0.0);
    let result = build_stage_templates_resolving_layout(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("build ok");

    let t = &result.templates[0];
    let row_water = 1_usize; // row_water_balance_start = N = 1
    let total_hours = 744.0_f64;
    let zeta = total_hours * 3_600.0 / 1_000_000.0;
    let expected_rhs = -(zeta * withdrawal);
    assert!(
        (t.row_lower[row_water] - expected_rhs).abs() < 1e-12,
        "water balance row_lower: expected {expected_rhs}, got {}",
        t.row_lower[row_water]
    );
    assert!(
        (t.row_upper[row_water] - expected_rhs).abs() < 1e-12,
        "water balance row_upper: expected {expected_rhs}, got {}",
        t.row_upper[row_water]
    );
}

/// `PrecomputedPar::default()` has `n_stages`=0, so `deterministic_base`=0 cannot
/// be set to a non-zero value here; this test only confirms withdrawal=0 leaves
/// the RHS identical to the no-withdrawal case.
#[test]
fn withdrawal_zero_leaves_rhs_unchanged_from_base() {
    let system_zero = one_hydro_system_with_withdrawal(1, 0, 0.0, 0.0);
    let system_base = one_hydro_system(1, 0);

    let result_zero = build_stage_templates_resolving_layout(
        &system_zero,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system_zero),
        &default_evaporation(&system_zero),
        &ResolvedParameters::default(),
    )
    .expect("zero-withdrawal build ok");

    let result_base = build_stage_templates_resolving_layout(
        &system_base,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system_base),
        &default_evaporation(&system_base),
        &ResolvedParameters::default(),
    )
    .expect("base build ok");

    let t_zero = &result_zero.templates[0];
    let t_base = &result_base.templates[0];

    // N=1, L=0: row_water_balance_start = 1.
    let row_water = 1_usize;
    assert!(
        (t_zero.row_lower[row_water] - t_base.row_lower[row_water]).abs() < 1e-15,
        "zero-withdrawal row_lower must equal base: {} vs {}",
        t_zero.row_lower[row_water],
        t_base.row_lower[row_water]
    );
    assert!(
        (t_zero.row_upper[row_water] - t_base.row_upper[row_water]).abs() < 1e-15,
        "zero-withdrawal row_upper must equal base: {} vs {}",
        t_zero.row_upper[row_water],
        t_base.row_upper[row_water]
    );
}

#[test]
fn withdrawal_slack_matrix_entry_coefficient_is_minus_zeta() {
    let system = one_hydro_system_with_withdrawal(1, 0, 5.0, 1000.0);
    let result = build_stage_templates_resolving_layout(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("build ok");

    let t = &result.templates[0];
    let n_h = system.hydros().len();
    // Withdrawal slack neg column: followed by N pos cols and 4*N operational violation slack columns.
    let col_neg = t.num_cols - 1 - 4 * n_h - n_h;
    let row_water = 1_usize; // water balance for hydro 0 = N + h = 1 + 0

    let total_hours = 744.0_f64;
    let zeta = total_hours * 3_600.0 / 1_000_000.0;

    let coeff = csc_entry(t, col_neg, row_water).unwrap_or_else(|| {
        panic!(
            "withdrawal slack neg column {col_neg} has no entry at water balance row {row_water}"
        )
    });
    assert!(
        (coeff - (-zeta)).abs() < 1e-12,
        "withdrawal slack neg coefficient: expected {}, got {coeff}",
        -zeta
    );

    let all_entries = entries_for_col(t, col_neg);
    assert_eq!(
        all_entries.len(),
        1,
        "withdrawal slack neg column must have exactly 1 CSC entry, got {all_entries:?}"
    );
}

#[test]
fn withdrawal_slack_objective_equals_cost_times_hours() {
    let violation_cost = 1_000.0_f64;
    let total_hours = 744.0_f64;
    let system = one_hydro_system_with_withdrawal(1, 0, 5.0, violation_cost);
    let result = build_stage_templates_resolving_layout(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("build ok");

    let t = &result.templates[0];
    let n_h = system.hydros().len();
    // Withdrawal slack neg: followed by N pos cols and 4*N operational violation slack columns.
    let col_w = t.num_cols - 1 - 4 * n_h - n_h;
    let expected_obj = violation_cost * total_hours / COST_SCALE_FACTOR;
    assert!(
        (t.objective[col_w] - expected_obj).abs() < 1e-12,
        "withdrawal slack objective: expected {expected_obj}, got {}",
        t.objective[col_w]
    );
}

#[test]
fn withdrawal_slack_objective_zero_when_cost_is_zero() {
    let system = one_hydro_system_with_withdrawal(1, 0, 0.0, 0.0);
    let result = build_stage_templates_resolving_layout(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("build ok");

    let t = &result.templates[0];
    let n_h = system.hydros().len();
    // Withdrawal slack neg: followed by N pos + 4*N operational violation slack columns.
    let col_w = t.num_cols - 1 - 5 * n_h;
    assert!(
        t.objective[col_w].abs() < 1e-15,
        "withdrawal slack neg objective must be 0 when cost=0, got {}",
        t.objective[col_w]
    );
}

#[test]
fn withdrawal_slack_bounds_are_sign_aware_positive_target() {
    let system = one_hydro_system_with_withdrawal(1, 0, 10.0, 5_000.0);
    let result = build_stage_templates_resolving_layout(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("build ok");

    let t = &result.templates[0];
    let n_h = system.hydros().len();
    // Trailing slack layout per hydro: neg, pos, then 4 operational violation
    // slacks (5 families after neg). col_neg is the first of the 6 trailing cols.
    let col_neg = t.num_cols - 1 - 5 * n_h;
    let col_pos = col_neg + n_h;
    assert!(
        (t.col_lower[col_neg] - 0.0).abs() < 1e-15,
        "neg slack lower bound must be 0.0, got {}",
        t.col_lower[col_neg]
    );
    assert!(
        (t.col_upper[col_neg] - 10.0).abs() < 1e-15,
        "neg slack upper bound must be |T| = 10.0 for T > 0, got {}",
        t.col_upper[col_neg]
    );
    assert!(
        (t.col_lower[col_pos] - 0.0).abs() < 1e-15,
        "pos slack lower bound must be 0.0, got {}",
        t.col_lower[col_pos]
    );
    assert!(
        t.col_upper[col_pos].is_infinite() && t.col_upper[col_pos] > 0.0,
        "pos slack upper bound must be +inf for T > 0 (over-withdrawal latitude), got {}",
        t.col_upper[col_pos]
    );
}

#[test]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::too_many_lines
)]
fn two_hydro_withdrawal_slack_entries_per_hydro() {
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
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 500.0,
            }],
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

    let inflow_models = vec![
        InflowModel {
            hydro_id: EntityId(2),
            stage_id: 0,
            mean_m3s: 80.0,
            std_m3s: 20.0,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
            annual: None,
        },
        InflowModel {
            hydro_id: EntityId(3),
            stage_id: 0,
            mean_m3s: 50.0,
            std_m3s: 10.0,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
            annual: None,
        },
    ];

    let load_models = vec![LoadModel {
        bus_id: EntityId(1),
        stage_id: 0,
        mean_mw: 100.0,
        std_mw: 0.0,
    }];

    let hydro_bounds_default = HydroStageBounds {
        min_storage_hm3: 0.0,
        max_storage_hm3: 200.0,
        filling_min_rate_m3s: 0.0,
        water_withdrawal_m3s: 0.0,
    };
    let hydro_bounds_default_block = HydroBlockBounds {
        max_turbined_m3s: 100.0,
        max_generation_mw: 250.0,
        ..Default::default()
    };
    let mut bounds = ResolvedBounds::new(
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
    bounds.hydro_bounds_mut(0, 0).water_withdrawal_m3s = 8.0;
    bounds.hydro_bounds_mut(1, 0).water_withdrawal_m3s = 12.0;

    let hydro_penalties_default = HydroPenalties {
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
        water_withdrawal_violation_cost: 2_000.0,
        water_withdrawal_violation_pos_cost: 2_000.0,
        water_withdrawal_violation_neg_cost: 2_000.0,
        evaporation_violation_pos_cost: 0.0,
        evaporation_violation_neg_cost: 0.0,
        inflow_nonnegativity_cost: 1000.0,
    };
    let penalties = ResolvedPenalties::new(
        &PenaltiesCountsSpec {
            n_hydros: 2,
            n_buses: 1,
            n_lines: 0,
            n_ncs: 0,
            n_stages: 1,
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

    let system = SystemBuilder::new()
        .buses(vec![bus])
        .hydros(vec![
            make_hydro(
                EntityId(2),
                HydroSpec {
                    name: "H2".to_string(),
                    bus_id: EntityId(1),
                    max_storage_hm3: 200.0,
                    generation_model: HydroGenerationModel::ConstantProductivity,
                    max_turbined_m3s: 100.0,
                    max_generation_mw: 250.0,
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
                        water_withdrawal_violation_cost: 2_000.0,
                        water_withdrawal_violation_pos_cost: 2_000.0,
                        water_withdrawal_violation_neg_cost: 2_000.0,
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
                    name: "H3".to_string(),
                    bus_id: EntityId(1),
                    max_storage_hm3: 200.0,
                    generation_model: HydroGenerationModel::ConstantProductivity,
                    max_turbined_m3s: 100.0,
                    max_generation_mw: 250.0,
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
                        water_withdrawal_violation_cost: 2_000.0,
                        water_withdrawal_violation_pos_cost: 2_000.0,
                        water_withdrawal_violation_neg_cost: 2_000.0,
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
        .expect("two_hydro_with_withdrawal: valid");

    let result = build_stage_templates_resolving_layout(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("build ok");

    let t = &result.templates[0];
    let n_h = system.hydros().len();

    // N=2, L=0: state+aux = N*(3+L)+1 = 7.
    // col_withdrawal_neg_start = 15, col_withdrawal_pos_start = 17,
    // operational violation slacks = 4*2 = 8, num_cols = 27.
    // Withdrawal neg columns: num_cols - 6*N = 27 - 12 = 15.
    let col_w0 = t.num_cols - 6 * n_h;
    let col_w1 = t.num_cols - 6 * n_h + 1;

    let row_w0 = 2_usize; // water balance for hydro 0 (row_water_balance_start = N = 2)
    let row_w1 = 3_usize; // water balance for hydro 1

    let total_hours = 744.0_f64;
    let zeta = total_hours * 3_600.0 / 1_000_000.0;

    let coeff_w0 = csc_entry(t, col_w0, row_w0).unwrap_or_else(|| {
        panic!("withdrawal slack col {col_w0} has no entry at water balance row {row_w0}")
    });
    assert!(
        (coeff_w0 - (-zeta)).abs() < 1e-12,
        "hydro-0 withdrawal slack coeff: expected {}, got {coeff_w0}",
        -zeta
    );

    let coeff_w1 = csc_entry(t, col_w1, row_w1).unwrap_or_else(|| {
        panic!("withdrawal slack col {col_w1} has no entry at water balance row {row_w1}")
    });
    assert!(
        (coeff_w1 - (-zeta)).abs() < 1e-12,
        "hydro-1 withdrawal slack coeff: expected {}, got {coeff_w1}",
        -zeta
    );

    assert!(
        csc_entry(t, col_w0, row_w1).is_none(),
        "hydro-0 withdrawal slack must not appear in hydro-1 water balance row"
    );
    assert!(
        csc_entry(t, col_w1, row_w0).is_none(),
        "hydro-1 withdrawal slack must not appear in hydro-0 water balance row"
    );
}

#[test]
#[allow(clippy::too_many_lines)]
fn three_hydro_num_cols_includes_three_withdrawal_slacks() {
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
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 500.0,
            }],
            ..Default::default()
        },
    );

    #[allow(clippy::cast_possible_wrap)]
    let stages = vec![make_stage(
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

    let inflow_models: Vec<InflowModel> = [1, 2, 3]
        .iter()
        .map(|&hid| InflowModel {
            hydro_id: EntityId(hid),
            stage_id: 0,
            mean_m3s: 50.0,
            std_m3s: 10.0,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
            annual: None,
        })
        .collect();

    let load_models = vec![LoadModel {
        bus_id: EntityId(1),
        stage_id: 0,
        mean_mw: 100.0,
        std_mw: 0.0,
    }];

    let bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 3,
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
                max_storage_hm3: 200.0,
                filling_min_rate_m3s: 0.0,
                water_withdrawal_m3s: 5.0,
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
            n_hydros: 3,
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
                evaporation_violation_cost: 0.0,
                water_withdrawal_violation_cost: 1_000.0,
                water_withdrawal_violation_pos_cost: 1_000.0,
                water_withdrawal_violation_neg_cost: 1_000.0,
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
        .hydros(vec![
            make_hydro(
                EntityId(1),
                HydroSpec {
                    name: "H1".to_string(),
                    bus_id: EntityId(1),
                    max_storage_hm3: 200.0,
                    generation_model: HydroGenerationModel::ConstantProductivity,
                    max_turbined_m3s: 100.0,
                    max_generation_mw: 250.0,
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
                        water_withdrawal_violation_cost: 1_000.0,
                        water_withdrawal_violation_pos_cost: 1_000.0,
                        water_withdrawal_violation_neg_cost: 1_000.0,
                        evaporation_violation_pos_cost: 0.0,
                        evaporation_violation_neg_cost: 0.0,
                        inflow_nonnegativity_cost: 1000.0,
                    },
                    ..Default::default()
                },
            ),
            make_hydro(
                EntityId(2),
                HydroSpec {
                    name: "H2".to_string(),
                    bus_id: EntityId(1),
                    max_storage_hm3: 200.0,
                    generation_model: HydroGenerationModel::ConstantProductivity,
                    max_turbined_m3s: 100.0,
                    max_generation_mw: 250.0,
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
                        water_withdrawal_violation_cost: 1_000.0,
                        water_withdrawal_violation_pos_cost: 1_000.0,
                        water_withdrawal_violation_neg_cost: 1_000.0,
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
                    name: "H3".to_string(),
                    bus_id: EntityId(1),
                    max_storage_hm3: 200.0,
                    generation_model: HydroGenerationModel::ConstantProductivity,
                    max_turbined_m3s: 100.0,
                    max_generation_mw: 250.0,
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
                        water_withdrawal_violation_cost: 1_000.0,
                        water_withdrawal_violation_pos_cost: 1_000.0,
                        water_withdrawal_violation_neg_cost: 1_000.0,
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
        .expect("three_hydro_system: valid");

    let result = build_stage_templates_resolving_layout(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("build ok");

    let t = &result.templates[0];

    // N=3, L=0, 1 block, no thermals/lines/evap/inflow-penalty:
    // state+aux: 3 storage + 3 z_inflow + 3 storage_in + 1 theta = 10
    // turbine: 3, spillage: 3, diversion: 3
    // deficit: 1, excess: 1
    // inflow slack: 0 (no penalty config)
    // evap: 0
    // withdrawal slacks: 3 neg + 3 pos = 6
    // operational violation slacks: 4*3 = 12
    // Total = 10 + 3 + 3 + 3 + 1 + 1 + 6 + 12 = 39
    let expected_cols = 39_usize;
    assert_eq!(
        t.num_cols, expected_cols,
        "3-hydro system: num_cols should be {expected_cols}, got {}",
        t.num_cols
    );

    // Withdrawal slacks: 2*N withdrawal + 4*N operational = 6*N trailing columns;
    // withdrawal_neg starts at num_cols - 6*N.
    let n_h = system.hydros().len();
    let neg_start = t.num_cols - 6 * n_h;
    let pos_start = neg_start + n_h;
    for h in 0..n_h {
        assert!(
            (t.col_upper[neg_start + h] - 5.0).abs() < 1e-15,
            "withdrawal slack neg column for hydro {h} must be capped at |T| = 5.0, got {}",
            t.col_upper[neg_start + h]
        );
        assert_eq!(
            t.col_upper[pos_start + h],
            f64::INFINITY,
            "withdrawal slack pos column for hydro {h} should be unbounded above (T > 0)"
        );
    }
}
