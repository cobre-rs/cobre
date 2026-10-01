//! `stochastic_load` section tests.

use super::*;
use cobre_core::scenario::SamplingScheme;
use cobre_core::temporal::Stage;

/// A [`PrecomputedNormal`] built over `system`'s own noise-member load buses,
/// so it satisfies the strict `n_entities() == n_load_buses` equality
/// [`build_stage_templates_resolving_layout`] now asserts.
fn noise_member_load_normal(system: &cobre_core::System) -> PrecomputedNormal {
    let stages: Vec<Stage> = system
        .stages()
        .iter()
        .filter(|s| s.id >= 0)
        .cloned()
        .collect();
    let max_blocks = stages.iter().map(|s| s.blocks.len()).max().unwrap_or(0);
    PrecomputedNormal::build(
        system.load_models(),
        &[],
        &stages,
        &system.load_noise_member_bus_ids(SamplingScheme::InSample),
        max_blocks,
    )
    .expect("load normal builds over the noise-member buses")
}

#[test]
fn stage_templates_load_balance_family_starts_after_the_water_rows() {
    let system = two_bus_system_with_stochastic_load(2, 2, 3);
    let result = build_stage_templates_resolving_layout(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &noise_member_load_normal(&system),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("constant productivity ok");

    assert_eq!(
        result.geometry_per_stage.len(),
        result.templates.len(),
        "geometry_per_stage length must match templates length"
    );

    // N=2 hydros, L=0: row_water_balance_start = n_hydros (z_inflow occupies rows
    // [0, n_hydros)); row_load_balance_start = row_water_balance_start + n_hydros.
    let n_hydros = system.hydros().len();
    let expected_row_start = n_hydros + n_hydros;
    assert_eq!(
        result.geometry_per_stage[0].load_balance.start(),
        expected_row_start,
        "geometry_per_stage[0].load_balance must start at row_water_balance_start + n_hydros"
    );
    assert_eq!(
        result.geometry_per_stage[0].load_balance.start(),
        result.geometry_per_stage[1].load_balance.start(),
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
        &noise_member_load_normal(&system),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("constant productivity ok");

    assert_eq!(
        result.n_load_buses(),
        1,
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
        result.n_load_buses(),
        0,
        "system with std_mw = 0 everywhere must give n_load_buses = 0"
    );
    assert!(
        result.load_bus_indices.is_empty(),
        "load_bus_indices must be empty when n_load_buses = 0"
    );
    assert_eq!(
        result.geometry_per_stage.len(),
        result.templates.len(),
        "geometry_per_stage length must always match templates length"
    );
}
