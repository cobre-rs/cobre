//! `penalty` section tests.

use super::*;

#[test]
fn test_penalty_columns_added() {
    let system = one_hydro_system(1, 0);
    let without = build_stage_templates_resolving_layout(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("constant productivity ok");
    let with_p = build_stage_templates_resolving_layout(
        &system,
        penalty_config(1000.0),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("constant productivity ok");
    assert_eq!(
        with_p.templates[0].num_cols,
        without.templates[0].num_cols + 1,
        "penalty method must add exactly n_hydros extra columns"
    );
}

#[test]
fn test_penalty_columns_added_3_hydros() {
    // Despite the name, this checks the n_hydros == 0 edge of
    // num_cols(penalty) = num_cols(none) + n_hydros: zero hydros add zero slacks
    // regardless of config (the N=1 column count is covered above).
    let system = one_bus_system(1);
    let with_p = build_stage_templates_resolving_layout(
        &system,
        penalty_config(1000.0),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("constant productivity ok");
    let without = build_stage_templates_resolving_layout(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("constant productivity ok");
    assert_eq!(
        with_p.templates[0].num_cols, without.templates[0].num_cols,
        "no slack columns when n_hydros == 0, even with penalty config"
    );
}

#[test]
fn test_penalty_objective_coefficient() {
    let system = one_hydro_system(1, 0);
    let config = penalty_config(1000.0);
    let result = build_stage_templates_resolving_layout(
        &system,
        config,
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("constant productivity ok");
    let t = &result.templates[0];
    let n_h = system.hydros().len();
    // N=1, L=0: theta=3, decision_start=4, turbine=4, spillage=5, diversion=6,
    // deficit=7, excess=8, inflow_slack=9, withdrawal_neg=10, withdrawal_pos=11,
    // outflow_below=12, outflow_above=13, turbine_below=14, generation_below=15.
    // inflow_slack sits before the 2 withdrawal + 4 op-violation slacks (6 per hydro).
    let slack_col = t.num_cols - 1 - 6 * n_h;
    let expected_obj = 1000.0 * 744.0 / COST_SCALE_FACTOR;
    assert!(
        (t.objective[slack_col] - expected_obj).abs() < 1e-12,
        "expected objective {expected_obj}, got {}",
        t.objective[slack_col]
    );
}

#[test]
fn test_no_penalty_columns_when_none() {
    let system = one_hydro_system(1, 2);
    let result = build_stage_templates_resolving_layout(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("constant productivity ok");
    let t = &result.templates[0];
    // N=1, L=2: state+aux = N*(3+L)+1 = 6 (storage, lags, z_inflow, storage_in, theta);
    // decisions = turb+spill+diversion+def+exc = 5; withdrawal = 2 (neg+pos);
    // operational violation slacks = 4; total = 17.
    assert_eq!(
        t.num_cols, 17,
        "method=none must not add extra penalty columns"
    );
    // num_rows = N z_inflow + N water_balance + B*K load_balance (1+1+1)
    // + 4 op-violation rows = 7; no state-fixing rows (state pinned via column bounds).
    assert_eq!(t.num_rows, 7, "method=none must not add extra penalty rows");
}

#[test]
#[allow(clippy::cast_sign_loss)]
fn test_penalty_slack_in_water_balance() {
    let system = one_hydro_system(1, 0);
    let config = penalty_config(1000.0);
    let result = build_stage_templates_resolving_layout(
        &system,
        config,
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("constant productivity ok");
    let t = &result.templates[0];
    let n_h = system.hydros().len();

    // inflow_slack sits before the withdrawal + 4 op-violation slacks (6 per hydro).
    let slack_col = t.num_cols - 1 - 6 * n_h;

    let water_balance_row = 1_usize; // N + h = 1 + 0

    let col_start = t.col_starts[slack_col] as usize;
    let col_end = t.col_starts[slack_col + 1] as usize;
    let found = t.row_indices[col_start..col_end]
        .iter()
        .zip(&t.values[col_start..col_end])
        .any(|(&r, &v)| r as usize == water_balance_row && v.abs() > 1e-12);

    assert!(
        found,
        "slack column must have a non-zero entry in the water balance row"
    );
}

#[test]
fn test_penalty_slack_bounds() {
    let system = one_hydro_system(1, 0);
    let config = penalty_config(1000.0);
    let result = build_stage_templates_resolving_layout(
        &system,
        config,
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("constant productivity ok");
    let t = &result.templates[0];
    let n_h = system.hydros().len();
    // inflow_slack sits before the withdrawal + 4 op-violation slacks (6 per hydro).
    let slack_col = t.num_cols - 1 - 6 * n_h;
    assert_eq!(t.col_lower[slack_col], 0.0, "slack lower bound must be 0.0");
    assert!(
        t.col_upper[slack_col].is_infinite() && t.col_upper[slack_col] > 0.0,
        "slack upper bound must be +infinity"
    );
}

// The penalty slack is virtual inflow, so it enters the LHS of the water-balance
// constraint (outflows - inflows = RHS) with coefficient -ζ, ζ = tau_total * M3S_TO_HM3.
// For 1 block of 744h: ζ = 744.0 * (3600.0 / 1_000_000.0) = 2.6784 → coefficient -2.6784.
#[test]
#[allow(clippy::cast_sign_loss)]
fn test_penalty_water_balance_coefficient_value() {
    let system = one_hydro_system(1, 0);
    let config = penalty_config(1000.0);
    let result = build_stage_templates_resolving_layout(
        &system,
        config,
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("constant productivity ok");
    let t = &result.templates[0];
    let n_h = system.hydros().len();

    // inflow_slack sits before the withdrawal + 4 op-violation slacks (6 per hydro).
    let slack_col = t.num_cols - 1 - 6 * n_h;
    let water_balance_row = 1_usize; // N + h = 1 + 0
    let zeta = 744.0 * (3_600.0 / 1_000_000.0);
    let expected_coeff = -zeta;

    let col_start = t.col_starts[slack_col] as usize;
    let col_end = t.col_starts[slack_col + 1] as usize;
    let coeff = t.row_indices[col_start..col_end]
        .iter()
        .zip(&t.values[col_start..col_end])
        .find(|&(&r, _)| r as usize == water_balance_row)
        .map(|(_, &v)| v);

    assert!(
        coeff.is_some(),
        "slack column must have an entry in the water balance row"
    );
    let coeff = coeff.unwrap();
    assert!(
        (coeff - expected_coeff).abs() < 1e-9,
        "expected coefficient {expected_coeff:.9}, got {coeff:.9}"
    );
}

#[test]
fn test_penalty_multi_stage_consistent() {
    let system = one_hydro_system(3, 1);
    let config = penalty_config(2000.0);
    let result = build_stage_templates_resolving_layout(
        &system,
        config,
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("constant productivity ok");
    assert_eq!(result.templates.len(), 3);
    let base_cols = result.templates[0].num_cols;
    for t in &result.templates {
        assert_eq!(
            t.num_cols, base_cols,
            "all stages must have the same column count"
        );
    }
}

// A large negative noise forces the inflow slack on: the deficit exceeds the
// available storage drawdown. Water balance is
// v_out - v_in + ζ*(turbine + spillage - inflow_slack) = RHS; with v_in = 100 hm³
// and RHS = -110 hm³ it reduces to ζ*inflow_slack ≥ 10 > 0 (v_out, turbine,
// spillage ≥ 0), so the slack is mandatory regardless of turbine level.
#[test]
fn test_penalty_slack_absorbs_negative_inflow() {
    use cobre_solver::{ActiveSolver, RowBatch, SolverInterface};

    let system = one_hydro_system(1, 0);
    let config = penalty_config(1000.0);
    let pm = production_set(&[0.9], 1);
    let result = build_stage_templates_resolving_layout(
        &system,
        config,
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &pm,
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("constant productivity ok");
    let template = &result.templates[0];
    let n_h = system.hydros().len();

    // inflow_slack sits before the withdrawal + 4 op-violation slacks (6 per hydro).
    let col_inflow_slack_start = template.num_cols - 1 - 6 * n_h;

    let col_storage_in = 2_usize; // storage_in for hydro 0 when N=1, L=0
    let water_balance_row = 1_usize; // N + h = 1 + 0

    let mut solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");
    solver.load_model(template);

    let empty_cuts = RowBatch {
        num_rows: 0,
        row_starts: vec![0_i32],
        col_indices: vec![],
        values: vec![],
        row_lower: vec![],
        row_upper: vec![],
    };
    solver.add_rows(&empty_cuts);

    // Incoming storage is pinned via column bounds, not a row-0 equality.
    let initial_storage = 100.0_f64;
    let negative_noise = -110.0_f64;
    solver.set_col_bounds(&[col_storage_in], &[initial_storage], &[initial_storage]);
    solver.set_row_bounds(&[water_balance_row], &[negative_noise], &[negative_noise]);

    let view = solver
        .solve(None)
        .expect("LP must be feasible with inflow slack active");

    let primal = view.primal;

    assert!(
        primal[col_inflow_slack_start] > 0.0,
        "inflow slack must be positive when noise is negative, got {}",
        primal[col_inflow_slack_start]
    );

    assert!(
        view.objective > 0.0,
        "objective must include a positive penalty contribution, got {}",
        view.objective
    );
}
