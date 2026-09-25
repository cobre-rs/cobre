//! `stochastic_load` section tests.

use super::*;

#[test]
fn stage_templates_load_balance_row_starts_correct() {
    let system = two_bus_system_with_stochastic_load(2, 2, 3);
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

    assert_eq!(
        result.load_balance_row_starts.len(),
        result.templates.len(),
        "load_balance_row_starts length must match templates length"
    );

    // N=2 hydros, L=0: row_water_balance_start = n_hydros (z_inflow occupies rows
    // [0, n_hydros)); row_load_balance_start = row_water_balance_start + n_hydros.
    let expected_row_start = result.n_hydros + result.n_hydros;
    assert_eq!(
        result.load_balance_row_starts[0], expected_row_start,
        "load_balance_row_starts[0] must equal row_water_balance_start + n_hydros"
    );
    assert_eq!(
        result.load_balance_row_starts[0], result.load_balance_row_starts[1],
        "identical stages share the same load balance row start"
    );
}

#[test]
fn stage_templates_n_load_buses_matches_stochastic_buses() {
    let system = two_bus_system_with_stochastic_load(1, 0, 1);
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

    assert_eq!(
        result.n_load_buses, 1,
        "only B2 has std_mw > 0 → n_load_buses must be 1"
    );
    assert_eq!(
        result.load_bus_indices.len(),
        1,
        "load_bus_indices must have exactly one entry"
    );
    assert_eq!(
        result.load_bus_indices[0], 1,
        "B2 is at buses slice index 1 (buses are [B1(10), B2(20)])"
    );
}

#[test]
fn stage_templates_no_load_buses_gives_zero() {
    // one_bus_system uses std_mw = 0 for all load models.
    let system = one_bus_system(2);
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

    assert_eq!(
        result.n_load_buses, 0,
        "system with std_mw = 0 everywhere must give n_load_buses = 0"
    );
    assert!(
        result.load_bus_indices.is_empty(),
        "load_bus_indices must be empty when n_load_buses = 0"
    );
    assert_eq!(
        result.load_balance_row_starts.len(),
        result.templates.len(),
        "load_balance_row_starts length must always match templates length"
    );
}
