//! Layer 5 — Semantic validation: hydro, thermal, stage, penalty, and scenario rules.
//!
//! Validates all domain-specific business rules after Layers 2-4 have
//! ensured schema correctness, referential integrity, and dimensional
//! consistency.
//!
//! ## Bound-precedence law
//!
//! Every block-eligible bound column resolves via a four-layer precedence
//! law. See [`resolve_bounds`](crate::resolution::resolve_bounds) for the full
//! law and [the per-column applicability table](crate::constraints::bounds).
//!
//! ## Layer 5a rules (hydro and thermal domain) — `validate_semantic_hydro_thermal`
//!
//! The Layer 5a rules are the `semantic.5a.*` and `travel_time.*` entries of [`RULES`](crate::validation::rules::RULES).
//!
//! A hydro unit group bounds row's `block_id` range and duplicate-row keying
//! are covered by `semantic.5a.35` and `semantic.5a.36`; a row referencing a non-existent
//! unit group id is still checked by `check_bounds_references` (Layer 3), not
//! here. Hydro unit group turbined-bound sign (`min_turbined_m3s >= 0` and
//! `max_turbined_m3s >= 0`, the retired `semantic.5a.42`) is validated at the PARSE
//! layer (`system/hydros.rs`'s `validate_unit_groups`, `LoadError::SchemaError`),
//! not the semantic layer.
//!
//! ## Layer 5b rules (stages, penalties, and scenario domain) — `validate_semantic_stages_penalties_scenarios`
//!
//! The Layer 5b rules are the `semantic.5b.*` entries of
//! [`RULES`](crate::validation::rules::RULES). Rule 49 is checked at study setup,
//! once the standardized external libraries exist, and has no entry there.

use super::{ValidationContext, schema::ParsedData};

mod block_bounds;
mod constraints;
mod correlation;
mod hydro;
mod inflow_seeding;
mod pumping;
mod scenarios;
mod season;
mod sobol;
mod stages;
mod thermal;
mod travel_time;

pub use inflow_seeding::seed_lag_state_depth;

pub(crate) fn validate_semantic_hydro_thermal(data: &ParsedData, ctx: &mut ValidationContext) {
    hydro::check_cascade_acyclic(data, ctx);
    hydro::check_hydro_bounds(data, ctx);
    hydro::check_diversion_floor_requires_channel(data, ctx);
    hydro::check_lifecycle_consistency(data, ctx);
    hydro::check_lifecycle_consistency_remaining(data, ctx);
    hydro::check_filling_config(data, ctx);
    hydro::check_filling_guards(data, ctx);
    hydro::check_geometry_monotonicity(data, ctx);
    hydro::check_evaporation_geometry_coverage(data, ctx);
    hydro::check_fpha_constraints(data, ctx);
    hydro::check_hydro_unit_groups(data, ctx);
    thermal::check_thermal_generation_bounds(data, ctx);
    thermal::check_anticipated_thermals(data, ctx);
    thermal::check_anticipated_cadence_transition(data, ctx);
    thermal::check_post_study_stages(data, ctx);
    thermal::check_anticipated_decision_target_is_anticipated(data, ctx);
    thermal::warn_thermal_generation_on_anticipated_thermal(data, ctx);
    constraints::check_per_block_storage_interior_reference(data, ctx);
    constraints::check_productivity_tag_pairing(data, ctx);
    block_bounds::check_bound_block_id_range(data, ctx);
    block_bounds::check_bound_stage_id_range(data, ctx);
    block_bounds::check_duplicate_bound_rows(data, ctx);
    block_bounds::check_block_id_on_ineligible_column(data, ctx);
    block_bounds::check_block_id_on_anticipated_thermal(data, ctx);
    block_bounds::check_bound_raises_declared_capacity(data, ctx);
    block_bounds::check_group_bound_raises_declared_capacity(data, ctx);
    pumping::check_pumping_semantics(data, ctx);
    pumping::check_pumping_operating_window(data, ctx);
    travel_time::validate_travel_time(data, ctx);
    inflow_seeding::validate_inflow_seeding(data, ctx);
}

/// Layer 5b. Every violation is collected into `ctx` before returning — no rule
/// short-circuits another.
pub(crate) fn validate_semantic_stages_penalties_scenarios(
    data: &ParsedData,
    ctx: &mut ValidationContext,
) {
    stages::check_stage_structure(data, ctx);
    stages::check_node_graph(data, ctx);
    stages::check_num_openings_declaration(data, ctx);
    stages::check_edge_discount_override_under_nodes(data, ctx);
    stages::check_nodes_and_noise_openings(data, ctx);
    stages::check_sampling_method_meaningfulness(data, ctx);
    stages::check_inflow_lags_vs_par_order(data, ctx);
    stages::check_study_stage_blocks(data, ctx);
    sobol::check_sobol_power_of_2(data, ctx);
    scenarios::check_penalty_ordering(data, ctx);
    scenarios::check_filling_sufficiency(data, ctx);
    scenarios::check_fpha_penalty_rule(data, ctx);
    scenarios::check_scenario_models(data, ctx);
    scenarios::check_par_stationarity(data, ctx);
    correlation::check_correlation_matrices(data, ctx);
    correlation::check_correlation_same_type(data, ctx);
    scenarios::check_external_scheme_has_files(data, ctx);
    scenarios::check_external_library_coherence(data, ctx);
    scenarios::check_load_factor_consistency(data, ctx);
    scenarios::check_estimation_prerequisites(data, ctx);
    season::check_season_id_consistency(data, ctx);
    season::check_observation_season_alignment(data, ctx);
}

// ── Tolerances ────────────────────────────────────────────────────────────────

const PROB_TOLERANCE: f64 = 1e-6;

const CORR_TOLERANCE: f64 = 1e-9;

/// Absorbs binary rounding when declared group maxima sum to the plant's value in
/// decimal but not in binary (0.1 + 0.2 > 0.3); a plant declaring no groups is
/// already exact and is admitted by the strict `>` in `check_hydro_unit_groups`.
const ENVELOPE_TOLERANCE: f64 = 1e-9;

/// `ENVELOPE_TOLERANCE` scaled to `value`'s own magnitude, floored at `1.0` so a
/// near-zero declared/required value doesn't collapse the tolerance to zero.
fn envelope_tolerance(value: f64) -> f64 {
    ENVELOPE_TOLERANCE * value.abs().max(1.0)
}
