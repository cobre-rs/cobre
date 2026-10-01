//! `shape` section tests.

use super::*;

#[test]
fn empty_stages_returns_empty() {
    let system = one_bus_system(0);
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
    assert!(result.templates.is_empty());
    assert!(result.block_hours_per_stage.is_empty());
}

#[test]
fn one_stage_one_template() {
    let system = one_bus_system(1);
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
    assert_eq!(result.templates.len(), 1);
    assert_eq!(result.block_hours_per_stage.len(), 1);
}

#[test]
fn num_cols_formula_no_hydro_no_thermal_no_line() {
    // N=0, T=0, Lines=0, B=1, K=1, L=0
    // num_cols = N*(2+L)+1 + N*K*2 + T*K + Lines*K*2 + B*K*2 = 1 + 1*1*2 = 3
    let system = one_bus_system(1);
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
    // theta + deficit + excess = 1 + 1 + 1 = 3
    assert_eq!(t.num_cols, 3, "num_cols mismatch for no-entity system");
}

#[test]
fn num_cols_formula_one_hydro_lag_zero() {
    // N=1, L=0, T=0, Lines=0, B=1, K=1
    // State cols: N*(2+L)+1 = 1*2+1 = 3  (v_out, v_in, theta)
    // Decision: turbine[1] + spillage[1] + deficit[1] + excess[1] = 4
    // Total: 7
    let system = one_hydro_system(1, 0);
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
    // N=1 withdrawal slacks add 2 columns (neg + pos): 7 + 2 = 9.
    // N=1 operational violation slacks add 4 columns: 9 + 4 = 13.
    // N=1 z-inflow column adds 1: 13 + 1 = 14.
    // N=1 diversion column adds 1: 14 + 1 = 15.
    assert_eq!(t.num_cols, 15, "num_cols mismatch for N=1 L=0");
}

#[test]
fn num_cols_formula_one_hydro_lag_two() {
    // N=1, L=2, T=0, Lines=0, B=1, K=1
    // State cols: N*(2+L)+1 = 1*4+1 = 5  (v_out, lag0, lag1, v_in, theta)
    // Decision: turbine[1] + spillage[1] + deficit[1] + excess[1] = 4
    // Total: 9
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
    // N=1 withdrawal slacks add 2 columns (neg + pos): 9 + 2 = 11.
    // N=1 operational violation slacks add 4 columns: 11 + 4 = 15.
    // N=1 z-inflow column adds 1: 15 + 1 = 16.
    // N=1 diversion column adds 1: 16 + 1 = 17.
    assert_eq!(t.num_cols, 17, "num_cols mismatch for N=1 L=2");
}

#[test]
fn num_rows_formula_no_hydro() {
    // N=0, B=1, K=1, L=0 → n_state = 0*(1+0) = 0
    // fixing rows: 0, water balance: 0, load balance: 1*1 = 1
    // num_rows = 0 + 0 + 1 = 1
    let system = one_bus_system(1);
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
    assert_eq!(t.num_rows, 1, "num_rows mismatch for no-hydro system");
}

#[test]
fn num_rows_formula_one_hydro_lag_zero() {
    // N=1, L=0, B=1, K=1; no state-fixing rows (incoming state pinned via column bounds).
    // num_rows = N z_inflow(1) + N water_balance(1) + B*K load_balance(1) + 4 op-violation = 7
    let system = one_hydro_system(1, 0);
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
    assert_eq!(t.num_rows, 7, "num_rows mismatch for N=1 L=0");
}

#[test]
fn num_rows_formula_one_hydro_lag_two() {
    // N=1, L=2, B=1, K=1; lags do not add rows, no state-fixing rows. Same 7 as L=0:
    // num_rows = N z_inflow(1) + N water_balance(1) + B*K load_balance(1) + 4 op-violation = 7
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
    assert_eq!(t.num_rows, 7, "num_rows mismatch for N=1 L=2");
}

#[test]
fn n_state_matches_indexer() {
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
    let expected = StateSpace::new(
        1,
        2,
        Vec::new(),
        vec![],
        AnticipatedResolution::default(),
        &[2; 1],
    )
    .n_state;
    assert_eq!(t.n_state, expected, "n_state must match StateSpace");
}

#[test]
fn csc_col_starts_monotone_nondecreasing() {
    let system = one_hydro_system(1, 1);
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
    for w in t.col_starts.windows(2) {
        assert!(w[0] <= w[1], "col_starts not monotone: {} > {}", w[0], w[1]);
    }
    assert_eq!(t.col_starts.len(), t.num_cols + 1);
}

#[test]
#[allow(clippy::cast_sign_loss)]
fn csc_row_indices_in_range() {
    let system = one_hydro_system(1, 1);
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
    for &r in &t.row_indices {
        assert!(
            r >= 0 && (r as usize) < t.num_rows,
            "row index {r} out of range [0, {})",
            t.num_rows
        );
    }
}

#[test]
#[allow(clippy::cast_sign_loss)]
fn csc_nz_count_matches_col_starts() {
    let system = one_hydro_system(1, 1);
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
    assert_eq!(
        t.num_nz,
        *t.col_starts.last().unwrap() as usize,
        "num_nz must equal col_starts[num_cols]"
    );
    assert_eq!(
        t.row_indices.len(),
        t.num_nz,
        "row_indices.len() must equal num_nz"
    );
    assert_eq!(t.values.len(), t.num_nz, "values.len() must equal num_nz");
}

#[test]
fn theta_column_has_unit_objective() {
    let lag_order = 2;
    let system = one_hydro_system(1, lag_order);
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
    let theta_col = StateSpace::new(
        1,
        lag_order,
        Vec::new(),
        vec![],
        AnticipatedResolution::default(),
        &[lag_order; 1],
    )
    .theta;
    assert_eq!(
        t.objective[theta_col], 1.0,
        "theta column objective must be the stage's one-step discount factor \
         (1.0 at rate 0), never divided by the cost scale factor"
    );
}

#[test]
fn spillage_objective_nonzero_for_nonzero_penalty() {
    // Hydro fixture has spillage_cost = 0.01 over a 744h block, so the spillage
    // objective is strictly positive.
    let system = one_hydro_system(1, 0);
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
    // spillage col for h=0, blk=0: col_spillage_start + 0 = N*(3+L)+1 + N*K
    // With N=1, L=0, K=1: theta=3, decision_start=4, turbine_start=4, spill_start=5
    let spill_col = 5;
    assert!(
        t.objective[spill_col] > 0.0,
        "spillage objective must be > 0 when spillage_cost > 0"
    );
}
