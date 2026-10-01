//! `evaporation` section tests.

use super::*;

use super::common::build_setup_in_code_with_models;
use super::common::in_code_studies::parallel_multiblock_evaporation_study;

fn load_template_with_no_cuts(
    template: &cobre_solver::StageTemplate,
) -> cobre_solver::ActiveSolver {
    use cobre_solver::SolverInterface;

    let mut solver = cobre_solver::ActiveSolver::new().expect("ActiveSolver::new must succeed");
    solver.load_model(template);
    let empty_cuts = cobre_solver::RowBatch {
        num_rows: 0,
        row_starts: vec![0_i32],
        col_indices: vec![],
        values: vec![],
        row_lower: vec![],
        row_upper: vec![],
    };
    solver.add_rows(&empty_cuts);
    solver
}

#[test]
fn evap_zero_hydros_layout_unchanged() {
    let system = one_hydro_system(1, 0);
    let no_evap = default_evaporation(&system);
    let with_evap = build_stage_templates_resolving_layout(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &no_evap,
        &ResolvedParameters::default(),
    )
    .expect("no evaporation ok");

    let baseline = build_stage_templates_resolving_layout(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &EvaporationModelSet::new(vec![EvaporationModel::None]),
        &ResolvedParameters::default(),
    )
    .expect("none evaporation ok");

    assert_eq!(
        with_evap.templates[0].num_cols, baseline.templates[0].num_cols,
        "num_cols must match with zero evaporation hydros"
    );
    assert_eq!(
        with_evap.templates[0].num_rows, baseline.templates[0].num_rows,
        "num_rows must match with zero evaporation hydros"
    );
}

#[test]
fn evap_two_hydros_increases_cols_and_rows() {
    let system1 = one_hydro_system(1, 0);

    let baseline = build_stage_templates_resolving_layout(
        &system1,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system1),
        &EvaporationModelSet::new(vec![EvaporationModel::None]),
        &ResolvedParameters::default(),
    )
    .expect("no evaporation baseline ok");

    let evap = evap_set_for_system(&system1, &[0], &[1.5]);
    let with_evap = build_stage_templates_resolving_layout(
        &system1,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system1),
        &evap,
        &ResolvedParameters::default(),
    )
    .expect("1 evaporation hydro ok");

    let base_cols = baseline.templates[0].num_cols;
    let base_rows = baseline.templates[0].num_rows;
    let evap_cols = with_evap.templates[0].num_cols;
    let evap_rows = with_evap.templates[0].num_rows;

    assert_eq!(
        evap_cols,
        base_cols + 3,
        "1 evap hydro must add exactly 3 columns (evaporation outflow, f_evap_plus, f_evap_minus)"
    );
    assert_eq!(
        evap_rows,
        base_rows + 1,
        "1 evap hydro must add exactly 1 row (evaporation equality constraint)"
    );
}

#[test]
fn evap_row_bounds_equality_at_intercept() {
    let system = one_hydro_system(1, 0);
    let intercept_m3s = 1.5_f64;
    let evap = evap_set_for_system(&system, &[0], &[intercept_m3s]);

    let result = build_stage_templates_resolving_layout(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &evap,
        &ResolvedParameters::default(),
    )
    .expect("evaporation system ok");

    let t = &result.templates[0];
    let n_h = system.hydros().len();

    // Evaporation row: followed by 4*N operational violation rows.
    let evap_row = t.num_rows - 1 - 4 * n_h;
    assert_eq!(
        t.row_lower[evap_row], intercept_m3s,
        "evaporation row_lower must equal intercept_m3s = {intercept_m3s}, got {}",
        t.row_lower[evap_row]
    );
    assert_eq!(
        t.row_upper[evap_row], intercept_m3s,
        "evaporation row_upper must equal intercept_m3s = {intercept_m3s}, got {}",
        t.row_upper[evap_row]
    );
}

#[test]
fn evap_col_bounds_and_objective() {
    let system = one_hydro_system(1, 0);
    let evap = evap_set_for_system(&system, &[0], &[1.5]);

    let result = build_stage_templates_resolving_layout(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &evap,
        &ResolvedParameters::default(),
    )
    .expect("evaporation system ok");

    let t = &result.templates[0];
    let n_h = system.hydros().len();

    // The 3 evaporation columns are followed by 1 withdrawal slack + 4 operational
    // violation slack columns (5*N=5 total for N=1).
    let col_evaporation_flow = t.num_cols - 4 - 5 * n_h;
    let col_f_plus = t.num_cols - 3 - 5 * n_h;
    let col_f_minus = t.num_cols - 2 - 5 * n_h;

    // The evaporation-outflow column is free-signed: [-q_max, +q_max] where
    // q_max = |intercept_m3s + volume_slope_m3s_per_hm3 * v_max| * margin.
    // intercept_m3s = 1.5, volume_slope_m3s_per_hm3 = 0.0, v_max = 200.0 → q_max = 1.5 * 2.0 = 3.0.
    let expected_evaporation_flow_bound = 1.5 * EVAPORATION_FLOW_SAFETY_MARGIN;
    assert!(
        (t.col_lower[col_evaporation_flow] - (-expected_evaporation_flow_bound)).abs() < 1e-12,
        "evaporation-outflow lower bound must be {}, got {}",
        -expected_evaporation_flow_bound,
        t.col_lower[col_evaporation_flow]
    );
    assert!(
        (t.col_upper[col_evaporation_flow] - expected_evaporation_flow_bound).abs() < 1e-12,
        "evaporation-outflow upper bound must be {expected_evaporation_flow_bound}, got {}",
        t.col_upper[col_evaporation_flow]
    );
    assert_eq!(
        t.objective[col_evaporation_flow], 0.0,
        "evaporation-outflow objective must be 0.0, got {}",
        t.objective[col_evaporation_flow]
    );

    for &col in &[col_f_plus, col_f_minus] {
        assert_eq!(
            t.col_lower[col], 0.0,
            "evap slack column {col} lower bound must be 0.0, got {}",
            t.col_lower[col]
        );
        assert!(
            t.col_upper[col].is_infinite() && t.col_upper[col] > 0.0,
            "evap slack column {col} upper bound must be +inf, got {}",
            t.col_upper[col]
        );
        assert_eq!(
            t.objective[col], 0.0,
            "evap slack column {col} objective must be 0.0, got {}",
            t.objective[col]
        );
    }
}

/// With `volume_slope_m3s_per_hm3 = 0.02`, the evaporation constraint row carries
/// `(evap-outflow, +1.0)`, `(v_col, -0.01)`, `(v_in_col, -0.01)` (= -slope/2),
/// `(f_plus_col, +1.0)`, `(f_minus_col, -1.0)`; the evap-outflow column also enters
/// the water balance row with `+zeta`.
#[test]
fn evap_csc_entries_one_hydro_correct_coefficients() {
    let system = one_hydro_system(1, 0);
    let volume_slope_m3s_per_hm3 = 0.02_f64;
    let evap = evap_set_with_volume_slope(&system, &[0], 1.5, volume_slope_m3s_per_hm3);

    let result = build_stage_templates_resolving_layout(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &evap,
        &ResolvedParameters::default(),
    )
    .expect("evaporation system ok");

    let t = &result.templates[0];
    let n_h = system.hydros().len();

    // Column layout for 1-hydro system (N=1, L=0, T=0, B=1, K=1):
    //   col 0 = v (storage_out)  col 1 = z_inflow  col 2 = v_in  col 3 = theta
    //   col 4 = turbine  col 5 = spillage  col 6 = diversion
    //   col 7 = deficit  col 8 = excess
    // Row layout for N=1, L=0, B=1, K=1, no FPHA (no state-fixing rows):
    //   row 0: z_inflow definition
    //   row 1: water balance (row_water_balance_start = N = 1)
    //   row 2: load balance
    //   row 3: evaporation constraint
    //   rows 4-7: operational violation rows
    // Evaporation columns come before withdrawal slack + 4*N operational slacks.
    let col_evaporation_flow = t.num_cols - 4 - 5 * n_h;
    let col_f_plus = t.num_cols - 3 - 5 * n_h;
    let col_f_minus = t.num_cols - 2 - 5 * n_h;
    let evap_row = t.num_rows - 1 - 4 * n_h;
    let water_balance_row = 1_usize; // row_water_balance_start = N = 1

    // Entries are sorted by row ascending: [0] = water balance, [1] = evap constraint.
    let zeta = 744.0 * (3_600.0 / 1_000_000.0);
    let entries_evaporation_flow = entries_for_col(t, col_evaporation_flow);
    assert_eq!(
        entries_evaporation_flow.len(),
        2,
        "evaporation outflow column must have exactly 2 entries (water balance + evap constraint), got {entries_evaporation_flow:?}"
    );
    assert_eq!(
        entries_evaporation_flow[0].0, water_balance_row,
        "evaporation outflow first entry must be at water balance row"
    );
    assert!(
        (entries_evaporation_flow[0].1 - zeta).abs() < 1e-12,
        "evaporation outflow water balance coefficient must be +zeta={zeta}, got {}",
        entries_evaporation_flow[0].1
    );
    assert_eq!(
        entries_evaporation_flow[1].0, evap_row,
        "evaporation outflow second entry must be at evap_row"
    );
    assert!(
        (entries_evaporation_flow[1].1 - 1.0).abs() < 1e-12,
        "evaporation outflow evap constraint coefficient must be +1.0, got {}",
        entries_evaporation_flow[1].1
    );

    let entries_f_plus = entries_for_col(t, col_f_plus);
    assert_eq!(
        entries_f_plus.len(),
        1,
        "f_plus column must have exactly 1 entry, got {entries_f_plus:?}"
    );
    assert_eq!(
        entries_f_plus[0].0, evap_row,
        "f_plus entry must be at evap_row"
    );
    assert!(
        (entries_f_plus[0].1 - 1.0).abs() < 1e-12,
        "f_plus coefficient must be +1.0, got {}",
        entries_f_plus[0].1
    );

    let entries_f_minus = entries_for_col(t, col_f_minus);
    assert_eq!(
        entries_f_minus.len(),
        1,
        "f_minus column must have exactly 1 entry, got {entries_f_minus:?}"
    );
    assert_eq!(
        entries_f_minus[0].0, evap_row,
        "f_minus entry must be at evap_row"
    );
    assert!(
        (entries_f_minus[0].1 - (-1.0)).abs() < 1e-12,
        "f_minus coefficient must be -1.0, got {}",
        entries_f_minus[0].1
    );

    // v and v_in carry -volume_slope/2 at evap_row (average-storage split).
    let expected_coeff = -volume_slope_m3s_per_hm3 / 2.0;
    let entry_v = entries_for_col(t, 0)
        .into_iter()
        .find(|&(r, _)| r == evap_row)
        .expect("v column must have an entry at evap_row");
    assert!(
        (entry_v.1 - expected_coeff).abs() < 1e-12,
        "v coefficient must be {expected_coeff}, got {}",
        entry_v.1
    );

    // v_in column: storage_in.start for 1-hydro (L=0) = N*(2+L) = 2; col_v_in = 2 + h_idx = 2.
    let col_v_in = 2;
    let entry_v_in = entries_for_col(t, col_v_in)
        .into_iter()
        .find(|&(r, _)| r == evap_row)
        .expect("v_in column must have an entry at evap_row");
    assert!(
        (entry_v_in.1 - expected_coeff).abs() < 1e-12,
        "v_in coefficient must be {expected_coeff}, got {}",
        entry_v_in.1
    );
}

#[test]
fn evap_csc_entries_coefficient_scaling() {
    let system = one_hydro_system(1, 0);
    let volume_slope_m3s_per_hm3 = 0.04_f64;
    let evap = evap_set_with_volume_slope(&system, &[0], 0.0, volume_slope_m3s_per_hm3);

    let result = build_stage_templates_resolving_layout(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &evap,
        &ResolvedParameters::default(),
    )
    .expect("evaporation system ok");

    let t = &result.templates[0];
    let n_h = system.hydros().len();
    let evap_row = t.num_rows - 1 - 4 * n_h;
    let expected_coeff = -volume_slope_m3s_per_hm3 / 2.0; // -0.02

    let entry_v = entries_for_col(t, 0)
        .into_iter()
        .find(|&(r, _)| r == evap_row)
        .expect("v column must have evap_row entry");
    assert!(
        (entry_v.1 - expected_coeff).abs() < 1e-12,
        "v coefficient: expected {expected_coeff}, got {}",
        entry_v.1
    );

    // storage_in.start for 1-hydro (L=0): N*(2+L) = 2; col_v_in = 2 + h_idx = 2.
    let col_v_in = 2;
    let entry_v_in = entries_for_col(t, col_v_in)
        .into_iter()
        .find(|&(r, _)| r == evap_row)
        .expect("v_in column must have evap_row entry");
    assert!(
        (entry_v_in.1 - expected_coeff).abs() < 1e-12,
        "v_in coefficient: expected {expected_coeff}, got {}",
        entry_v_in.1
    );
}

#[test]
fn evap_csc_entries_zero_hydros_no_op() {
    let system = one_hydro_system(1, 0);
    let no_evap = default_evaporation(&system);

    let result = build_stage_templates_resolving_layout(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &no_evap,
        &ResolvedParameters::default(),
    )
    .expect("no evaporation ok");

    let baseline = build_stage_templates_resolving_layout(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &EvaporationModelSet::new(vec![EvaporationModel::None]),
        &ResolvedParameters::default(),
    )
    .expect("none evaporation ok");

    assert_eq!(
        result.templates[0].num_nz, baseline.templates[0].num_nz,
        "num_nz must be identical with zero evaporation hydros"
    );
}

#[test]
fn evap_csc_entries_two_hydros_independent_rows() {
    let (system, production) = four_hydro_mixed_system();
    let n_stages = system.stages().iter().filter(|s| s.id >= 0).count();

    let models = vec![
        EvaporationModel::Linearized {
            coefficients: vec![
                LinearizedEvaporation {
                    intercept_m3s: 1.0,
                    volume_slope_m3s_per_hm3: 0.02,
                };
                n_stages
            ],
            reference_volumes_hm3: vec![100.0; n_stages],
        },
        EvaporationModel::Linearized {
            coefficients: vec![
                LinearizedEvaporation {
                    intercept_m3s: 2.0,
                    volume_slope_m3s_per_hm3: 0.06,
                };
                n_stages
            ],
            reference_volumes_hm3: vec![100.0; n_stages],
        },
        EvaporationModel::None,
        EvaporationModel::None,
    ];
    let evap = EvaporationModelSet::new(models);

    let result = build_stage_templates_resolving_layout(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &production,
        &evap,
        &ResolvedParameters::default(),
    )
    .expect("2-evap-hydro system ok");

    let t = &result.templates[0];
    let n_h = system.hydros().len();
    // 2 evap hydros: evap rows are followed by 4*N operational violation rows.
    let evap_row_0 = t.num_rows - 2 - 4 * n_h;
    let evap_row_1 = t.num_rows - 1 - 4 * n_h;

    // Hydro 0 (volume_slope_m3s_per_hm3=0.02): v coefficient = -0.01.
    let entry_v_h0 = entries_for_col(t, 0)
        .into_iter()
        .find(|&(r, _)| r == evap_row_0)
        .expect("hydro 0 v col entry");
    assert!(
        (entry_v_h0.1 - (-0.01)).abs() < 1e-12,
        "hydro 0 v: expected -0.01, got {}",
        entry_v_h0.1
    );

    // Hydro 1 (volume_slope_m3s_per_hm3=0.06): v coefficient = -0.03.
    let entry_v_h1 = entries_for_col(t, 1)
        .into_iter()
        .find(|&(r, _)| r == evap_row_1)
        .expect("hydro 1 v col entry");
    assert!(
        (entry_v_h1.1 - (-0.03)).abs() < 1e-12,
        "hydro 1 v: expected -0.03, got {}",
        entry_v_h1.1
    );

    // Row bounds: hydro 0 → intercept_m3s=1.0, hydro 1 → intercept_m3s=2.0.
    assert!((t.row_lower[evap_row_0] - 1.0).abs() < 1e-12);
    assert!((t.row_lower[evap_row_1] - 2.0).abs() < 1e-12);
}

#[test]
fn evap_csc_entries_zero_volume_slope_produces_zero_volume_coefficients() {
    let system = one_hydro_system(1, 0);
    let evap = evap_set_with_volume_slope(&system, &[0], 2.0, 0.0);

    let result = build_stage_templates_resolving_layout(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &evap,
        &ResolvedParameters::default(),
    )
    .expect("evaporation system ok");

    let t = &result.templates[0];
    let n_h = system.hydros().len();
    let evap_row = t.num_rows - 1 - 4 * n_h;

    let entry_v = entries_for_col(t, 0)
        .into_iter()
        .find(|&(r, _)| r == evap_row)
        .expect("v column must have evap_row entry");
    assert!(
        entry_v.1.abs() < 1e-12,
        "v coefficient must be 0.0 when volume_slope_m3s_per_hm3=0, got {}",
        entry_v.1
    );

    // storage_in.start for 1-hydro (L=0): N*(2+L) = 2; col_v_in = 2 + h_idx = 2.
    let col_v_in = 2;
    let entry_v_in = entries_for_col(t, col_v_in)
        .into_iter()
        .find(|&(r, _)| r == evap_row)
        .expect("v_in column must have evap_row entry");
    assert!(
        entry_v_in.1.abs() < 1e-12,
        "v_in coefficient must be 0.0 when volume_slope_m3s_per_hm3=0, got {}",
        entry_v_in.1
    );
}

#[test]
#[allow(clippy::cast_sign_loss)]
fn evap_water_balance_one_hydro_coefficient_is_zeta() {
    let system = one_hydro_system(1, 0);
    let evap = evap_set_with_volume_slope(&system, &[0], 0.0, 0.0);

    let result = build_stage_templates_resolving_layout(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &evap,
        &ResolvedParameters::default(),
    )
    .expect("evaporation system ok");

    let t = &result.templates[0];
    let n_h = system.hydros().len();

    let water_balance_row = 1_usize; // row_water_balance_start = N = 1

    // evap outflow is the first of 3 evaporation columns; before withdrawal + 4*N op slacks.
    let col_evaporation_flow = t.num_cols - 4 - 5 * n_h;

    let entries = entries_for_col(t, col_evaporation_flow);
    let entry = entries
        .iter()
        .find(|&&(r, _)| r == water_balance_row)
        .copied()
        .expect("evaporation outflow column must have an entry in the water balance row");

    let zeta = 744.0_f64 * (3_600.0 / 1_000_000.0);
    assert!(
        (entry.1 - zeta).abs() < 1e-12,
        "evaporation outflow water balance coefficient must be +zeta={zeta}, got {}",
        entry.1
    );
}

#[test]
#[allow(clippy::cast_sign_loss, clippy::too_many_lines)]
fn evap_water_balance_only_second_hydro_has_evap() {
    use chrono::NaiveDate;
    use cobre_core::entities::hydro::{HydroGenerationModel, HydroPenalties};
    use cobre_core::scenario::{InflowModel, LoadModel};
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage as CStage, StageRiskConfig,
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
    let hp = HydroPenalties {
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
            EntityId(2),
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
                penalties: hp,
                ..Default::default()
            },
        ),
        make_hydro(
            EntityId(3),
            HydroSpec {
                name: "H3".to_string(),
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
                penalties: hp,
                ..Default::default()
            },
        ),
    ];
    let stages = vec![CStage {
        index: 0,
        id: 0,
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
    }];
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
    let load_models = vec![LoadModel {
        bus_id: EntityId(1),
        stage_id: 0,
        mean_mw: 200.0,
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
    let system = SystemBuilder::new()
        .buses(vec![bus])
        .hydros(hydros)
        .stages(stages)
        .inflow_models(inflow_models)
        .load_models(load_models)
        .bounds(bounds)
        .penalties(penalties)
        .build()
        .expect("2-hydro system ok");

    let evap = evap_set_with_volume_slope(&system, &[1], 0.0, 0.0);

    let result = build_stage_templates_resolving_layout(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &evap,
        &ResolvedParameters::default(),
    )
    .expect("2-hydro evap system ok");

    let t = &result.templates[0];
    let n_h = system.hydros().len();

    // row_water_balance_start = N = 2 (z_inflow rows [0,2)); hydro 0 row 2, hydro 1 row 3.
    let water_balance_row_h0 = 2_usize;
    let water_balance_row_h1 = 3_usize;

    // evaporation outflow for hydro 1 (local_idx=0, since only hydro 1 is evap): col_evap_start + 0*3.
    // N=2 withdrawal + 4*N operational slack columns follow evap.
    let col_evaporation_flow_h1 = t.num_cols - 5 - 5 * n_h;

    let entries_h1 = entries_for_col(t, col_evaporation_flow_h1);
    let found_h1 = entries_h1
        .iter()
        .find(|&&(r, _)| r == water_balance_row_h1)
        .copied();
    assert!(
        found_h1.is_some(),
        "evaporation outflow for hydro 1 must have an entry in water balance row {water_balance_row_h1}"
    );
    let zeta = 744.0_f64 * (3_600.0 / 1_000_000.0);
    assert!(
        (found_h1.unwrap().1 - zeta).abs() < 1e-12,
        "evaporation outflow (h1) water balance coefficient must be +zeta={zeta}, got {}",
        found_h1.unwrap().1
    );

    let found_h0 = entries_h1.iter().any(|&(r, _)| r == water_balance_row_h0);
    assert!(
        !found_h0,
        "evaporation outflow for hydro 1 must not appear in hydro 0's water balance row"
    );
}

#[test]
fn evap_water_balance_zero_hydros_no_op() {
    let system = one_hydro_system(1, 0);
    let no_evap = EvaporationModelSet::new(vec![EvaporationModel::None]);

    let result = build_stage_templates_resolving_layout(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &no_evap,
        &ResolvedParameters::default(),
    )
    .expect("no evaporation ok");

    let baseline = build_stage_templates_resolving_layout(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("default evaporation ok");

    assert_eq!(
        result.templates[0].num_nz, baseline.templates[0].num_nz,
        "num_nz must be identical with zero evaporation hydros (no water balance entries added)"
    );
}

#[test]
fn evap_violation_cost_applied_to_slack_columns() {
    let system = evap_hydro_system_with_violation_cost(730.0, 500.0);
    let evap = evap_set_with_volume_slope(&system, &[0], 1.0, 0.02);

    let result = build_stage_templates_resolving_layout(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &evap,
        &ResolvedParameters::default(),
    )
    .expect("evap violation cost system builds ok");

    let t = &result.templates[0];
    let n_h = system.hydros().len();

    // Evaporation columns (evaporation outflow, f_plus, f_minus) are followed by
    // 1 withdrawal slack + 4*N operational slacks.
    let col_evaporation_flow = t.num_cols - 4 - 5 * n_h;
    let col_f_plus = t.num_cols - 3 - 5 * n_h;
    let col_f_minus = t.num_cols - 2 - 5 * n_h;

    let expected_base = 500.0 * 730.0 / COST_SCALE_FACTOR;

    assert!(
        t.objective[col_evaporation_flow].abs() < 1e-12,
        "evaporation outflow column objective must be 0.0 (evaporation flow itself has no cost), got {}",
        t.objective[col_evaporation_flow]
    );
    assert!(
        (t.objective[col_f_plus] - expected_base).abs() < 1e-12,
        "f_evap_plus objective: expected {expected_base}, got {}",
        t.objective[col_f_plus]
    );
    // f_evap_minus uses evaporation_violation_pos_cost; the test sets pos_cost == base_cost.
    assert!(
        (t.objective[col_f_minus] - expected_base).abs() < 1e-6,
        "f_evap_minus objective: expected {expected_base} (pos_cost == base_cost in test), got {}",
        t.objective[col_f_minus]
    );
}

#[test]
fn evap_outflow_objective_is_zero() {
    let system = evap_hydro_system_with_violation_cost(730.0, 500.0);
    let evap = evap_set_with_volume_slope(&system, &[0], 0.0, 0.0);

    let result = build_stage_templates_resolving_layout(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &evap,
        &ResolvedParameters::default(),
    )
    .expect("evap system with zero k_evap builds ok");

    let t = &result.templates[0];
    let n_h = system.hydros().len();
    // N=1 withdrawal + 4*N operational slacks follow the 3 evap columns.
    let col_evaporation_flow = t.num_cols - 4 - 5 * n_h;

    assert!(
        t.objective[col_evaporation_flow].abs() < 1e-12,
        "evaporation outflow objective must be 0.0, got {}",
        t.objective[col_evaporation_flow]
    );
}

#[test]
fn evap_lp_solvable_and_outflow_positive_coefficients() {
    use cobre_solver::SolverInterface;

    let system = evap_hydro_system_with_violation_cost(730.0, 500.0);
    let evap = evap_set_with_volume_slope(&system, &[0], 1.0, 0.02);

    let result = build_stage_templates_resolving_layout(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &evap,
        &ResolvedParameters::default(),
    )
    .expect("evap system template build must succeed");

    let template = &result.templates[0];
    let n_h = system.hydros().len();
    let mut solver = load_template_with_no_cuts(template);

    // Fix v_in = 1000 hm3 via column bounds on storage_in.
    let col_storage_in = 2_usize; // col 0 = storage_out, col 1 = z_inflow, col 2 = storage_in
    let v_in = 1_000.0_f64;
    solver.set_col_bounds(&[col_storage_in], &[v_in], &[v_in]);

    let view = solver
        .solve(None)
        .expect("evaporation LP must be feasible and optimal");

    // evaporation outflow is the first evaporation column (before withdrawal + 4*N operational slacks).
    let col_evaporation_flow = template.num_cols - 4 - 5 * n_h;
    let evaporation_flow = view.primal[col_evaporation_flow];

    // Tight lower bound: evaporation outflow >= intercept_m3s + (volume_slope_m3s_per_hm3 / 2) · v_min + (volume_slope_m3s_per_hm3 / 2) · v_in
    //                         >= 1.0   + 0.0                       + 0.01 · 1000 = 11.0.
    // A loose threshold (`evaporation_flow > -1e-8`) would silently pass a sign-convention
    // regression that flipped the bound; assert the structurally-forced minimum.
    assert!(
        evaporation_flow > 10.0,
        "evaporation outflow must reflect the positive linearised target (>= 11.0), got {evaporation_flow}"
    );
}

#[test]
fn evap_violation_slacks_near_zero_feasible_constraint() {
    use cobre_solver::SolverInterface;

    let system = evap_hydro_system_with_violation_cost(730.0, 500.0);
    let evap = evap_set_with_volume_slope(&system, &[0], 1.0, 0.02);

    let result = build_stage_templates_resolving_layout(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &evap,
        &ResolvedParameters::default(),
    )
    .expect("evap system template build must succeed");

    let template = &result.templates[0];
    let n_h = system.hydros().len();
    let mut solver = load_template_with_no_cuts(template);

    let v_in = 1_000.0_f64;
    solver.set_row_bounds(&[0], &[v_in], &[v_in]);

    let view = solver
        .solve(None)
        .expect("evaporation LP must be feasible and optimal");

    // Evaporation violation slack columns are before withdrawal + 4*N operational slacks.
    let col_f_plus = template.num_cols - 3 - 5 * n_h;
    let col_f_minus = template.num_cols - 2 - 5 * n_h;
    let f_plus = view.primal[col_f_plus];
    let f_minus = view.primal[col_f_minus];

    assert!(
        f_plus.abs() < 1e-6,
        "f_evap_plus slack must be near zero (constraint satisfied without violation), got {f_plus}"
    );
    assert!(
        f_minus.abs() < 1e-6,
        "f_evap_minus slack must be near zero (constraint satisfied without violation), got {f_minus}"
    );
}

#[test]
fn evap_incoming_storage_reduced_cost_differs_from_no_evaporation() {
    use cobre_solver::SolverInterface;

    // System with evaporation violation cost (so slacks are penalised).
    let system_evap = evap_hydro_system_with_violation_cost(730.0, 500.0);
    let evap = evap_set_with_volume_slope(&system_evap, &[0], 1.0, 0.02);
    let evap_result = build_stage_templates_resolving_layout(
        &system_evap,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system_evap),
        &evap,
        &ResolvedParameters::default(),
    )
    .expect("evap system template build must succeed");

    // Baseline system without evaporation (same structure, EvaporationModel::None).
    let system_base = one_hydro_system(1, 0);
    let no_evap = EvaporationModelSet::new(vec![EvaporationModel::None]);
    let base_result = build_stage_templates_resolving_layout(
        &system_base,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system_base),
        &no_evap,
        &ResolvedParameters::default(),
    )
    .expect("baseline system template build must succeed");

    // Storage is pinned via column bounds: col 0 = storage_out, 1 = z_inflow, 2 = storage_in.
    let col_storage_in = 2_usize;
    let solve_and_get_storage_reduced_cost = |template: &cobre_solver::StageTemplate| -> f64 {
        let mut solver = load_template_with_no_cuts(template);
        let v_in = 1_000.0_f64;
        solver.set_col_bounds(&[col_storage_in], &[v_in], &[v_in]);
        let view = solver.solve(None).expect("LP must solve to optimal");
        view.reduced_costs[col_storage_in]
    };

    let evap_rc = solve_and_get_storage_reduced_cost(&evap_result.templates[0]);
    let base_rc = solve_and_get_storage_reduced_cost(&base_result.templates[0]);

    // The evaporation constraint couples evaporation outflow to v and v_in via volume_slope_m3s_per_hm3,
    // so the marginal value of initial storage differs from the no-evaporation case.
    assert_ne!(
        (evap_rc * 1e6).round(),
        (base_rc * 1e6).round(),
        "incoming-storage reduced cost must differ between evaporation ({evap_rc}) and \
         no-evaporation ({base_rc}) configurations"
    );
}

#[test]
fn evap_bound_prevents_dump_valve() {
    use cobre_solver::SolverInterface;

    let system = evap_hydro_system_with_violation_cost(730.0, 500.0);
    let evap = evap_set_with_volume_slope(&system, &[0], 2.0, 0.0001);

    let result = build_stage_templates_resolving_layout(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &evap,
        &ResolvedParameters::default(),
    )
    .expect("evap dump valve test: template build must succeed");

    let template = &result.templates[0];
    let n_h = system.hydros().len();
    let mut solver = load_template_with_no_cuts(template);

    // col 0 = storage_out, col 1 = z_inflow, col 2 = storage_in (N=1, L=0).
    let col_storage_in = 2_usize;
    let v_in = 2_000.0_f64;
    solver.set_col_bounds(&[col_storage_in], &[v_in], &[v_in]);

    // Inject large inflow via water-balance RHS (row 1; row 0 = z_inflow[0]).
    // Water balance: v + zeta*(turbine + spill + div) - v_in + zeta*evaporation outflow = RHS.
    // The template RHS = zeta * base = 2.628 * 50 = 131.4.
    // Set RHS to zeta * 500 = 1314 to simulate a 500 m3/s inflow.
    // The LP must then satisfy: v + zeta*(turbine+spill+...) = v_in + 1314 = 3314.
    // With v <= 2000 and max turbine = 262.8 hm3, surplus > 1000 hm3 must spill.
    let zeta = 730.0 * 3600.0 / 1e6;
    let high_inflow_rhs = zeta * 500.0;
    // Water balance row index: N + h = 1 + 0 = 1 for N=1, L=0, h=0.
    let water_balance_row = 1_usize;
    solver.set_row_bounds(&[water_balance_row], &[high_inflow_rhs], &[high_inflow_rhs]);

    let view = solver
        .solve(None)
        .expect("evap dump valve LP must be feasible and optimal");

    // Column layout: N=1, L=0, K=1.
    // col 0: v, col 1: z_inflow, col 2: v_in, col 3: theta,
    // col 4: turbine, col 5: spillage, col 6: diversion,
    // col 7: deficit, col 8: excess.
    // Evaporation columns: evaporation outflow, f_plus, f_minus, then withdrawal + 4*N operational slacks.
    let col_spillage = 5;
    let col_evaporation_flow = template.num_cols - 4 - 5 * n_h;
    let col_f_minus = template.num_cols - 2 - 5 * n_h;

    let evaporation_flow = view.primal[col_evaporation_flow];
    let f_minus = view.primal[col_f_minus];
    let spillage = view.primal[col_spillage];

    // evaporation outflow must respect the symmetric magnitude bound.
    // intercept_m3s=2.0, volume_slope_m3s_per_hm3=0.0001, max_storage_hm3=2000.0
    // evaporation_flow_max = |2.0 + 0.0001*2000| * 2.0 = 2.2 * 2.0 = 4.4
    let evaporation_flow_max = (2.0 + 0.0001 * 2_000.0_f64).abs() * EVAPORATION_FLOW_SAFETY_MARGIN;
    assert!(
        evaporation_flow <= evaporation_flow_max + 1e-8,
        "evaporation outflow must be bounded by physical limit {evaporation_flow_max}, got {evaporation_flow}"
    );

    assert!(
        f_minus < 1e-6,
        "f_minus (over-evaporation) must be near zero, got {f_minus}"
    );

    assert!(
        spillage > 1e-6,
        "spillage must be positive when excess water needs dumping, got {spillage}"
    );
}

/// On every stage of a parallel multi-block study, the StudySetup-built LP
/// reserves exactly one evaporation slot per evaporating hydro, coupled with
/// the stage's `ζ` on the water row, its violation slacks priced at the
/// violation cost times the stage's total hours (744 h = 200 + 244 + 300), and
/// its evaporation row's storage entries confined to the incoming/outgoing
/// storage columns.
#[test]
fn parallel_multiblock_evaporation_study_has_one_priced_stage_slot() {
    const M3S_TO_HM3: f64 = 3_600.0 / 1_000_000.0;
    use cobre_sddp::indexer::Boundary;
    use cobre_sddp::indexer::HydroSys;

    let (system, config, hydro_models) = parallel_multiblock_evaporation_study();
    let setup = build_setup_in_code_with_models(system, &config, hydro_models);
    let state = cobre_sddp::test_support::state_space(&setup);
    let templates = &setup.inputs.stage_data.stage_templates;
    let total_stage_hours = 744.0_f64;

    let unscale = |t: &StageTemplate, r: usize, c: usize, v: f64| -> f64 {
        let rs = t.row_scale.get(r).copied().unwrap_or(1.0);
        let cs = t.col_scale.get(c).copied().unwrap_or(1.0);
        v / (rs * cs)
    };
    let unscale_objective = |t: &StageTemplate, c: usize| -> f64 {
        let cs = t.col_scale.get(c).copied().unwrap_or(1.0);
        t.objective[c] * templates.cost_scale_factor / cs
    };

    for (s, t) in templates.templates.iter().enumerate() {
        let g = &templates.geometry_per_stage[s];
        assert_eq!(
            g.evap_indices.len(),
            1,
            "stage {s}: a parallel stage must reserve exactly one evaporation slot"
        );
        let ei = g.evap_indices[0];
        let zeta = templates.block_hours_per_stage[s].iter().sum::<f64>() * M3S_TO_HM3;

        let water_value = csc_entry(t, ei.evaporation_flow_col, g.water_balance.start())
            .expect("evaporation flow column must have an entry on the water row");
        let unscaled_water = unscale(
            t,
            g.water_balance.start(),
            ei.evaporation_flow_col,
            water_value,
        );
        assert!(
            (unscaled_water - zeta).abs() < 1e-12 * zeta.abs(),
            "stage {s}: unscaled water-row coefficient must equal zeta = {zeta}, got {unscaled_water}"
        );

        let unscaled_f_plus = unscale_objective(t, ei.f_evap_plus_col);
        let expected_f_plus = 7.0 * total_stage_hours;
        assert!(
            (unscaled_f_plus - expected_f_plus).abs() < 1e-12 * expected_f_plus,
            "stage {s}: unscaled f_evap_plus objective must equal {expected_f_plus}, got {unscaled_f_plus}"
        );

        let unscaled_f_minus = unscale_objective(t, ei.f_evap_minus_col);
        let expected_f_minus = 11.0 * total_stage_hours;
        assert!(
            (unscaled_f_minus - expected_f_minus).abs() < 1e-12 * expected_f_minus,
            "stage {s}: unscaled f_evap_minus objective must equal {expected_f_minus}, got {unscaled_f_minus}"
        );

        let cols_at_evap_row: Vec<usize> = (0..t.num_cols)
            .filter(|&c| csc_entry(t, c, ei.evap_row).is_some())
            .collect();
        let mut expected_cols = vec![
            g.storage_boundary_grid()
                .col(state, HydroSys::new(0), Boundary::Incoming),
            g.storage_boundary_grid()
                .col(state, HydroSys::new(0), Boundary::Outgoing),
            ei.evaporation_flow_col,
            ei.f_evap_plus_col,
            ei.f_evap_minus_col,
        ];
        expected_cols.sort_unstable();
        assert_eq!(
            cols_at_evap_row, expected_cols,
            "stage {s}: evaporation row's storage entries must be confined to the incoming/outgoing storage columns"
        );
    }
}

/// The parallel-multiblock-evaporation fixture's stage 0 is a 3-block
/// parallel stage with active evaporation.
#[test]
fn parallel_evaporation_fixture_evaporates_on_a_multiblock_parallel_stage() {
    let (system, config, hydro_models) = parallel_multiblock_evaporation_study();
    let setup = build_setup_in_code_with_models(system, &config, hydro_models);

    let geometry = &setup.inputs.stage_data.stage_templates.geometry_per_stage[0];
    assert_eq!(geometry.block_mode, cobre_core::BlockMode::Parallel);
    assert_eq!(geometry.n_blks, 3);
    assert!(
        !geometry.evap_hydro_indices.is_empty(),
        "stage 0 must have an active evaporation slot"
    );
}
