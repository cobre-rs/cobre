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
//! | #  | Rule                                                                    | Source file                                    | `ErrorKind`              |
//! |----|-------------------------------------------------------------------------|------------------------------------------------|--------------------------|
//! | 1  | Every transition `source_id`/`target_id` must refer to an existing stage| `stages.json`                                  | `InvalidValue`           |
//! | 2  | Outgoing transition probabilities sum to 1.0 (±1e-6) per source stage  | `stages.json`                                  | `InvalidValue`           |
//! | 3  | Cyclic graph: `annual_discount_rate > 0.0`                              | `stages.json`                                  | `InvalidValue`           |
//! | 4  | *(retired — enforced at the parse layer by `stages.rs`'s `validate_block_hours` / `validate_risk_measure`; number never reused)* | — | — |
//! | 5  | *(retired — enforced at the parse layer by `stages.rs`'s `validate_block_hours` / `validate_risk_measure`; number never reused)* | — | — |
//! | 6  | *(retired — compared a $/hm³ cost with a $/`MWh` cost, which needs plant productivity; number never reused)* | — | — |
//! | 7  | *(retired — compared a $/hm³ cost with a $/`MWh` cost, which needs plant productivity; number never reused)* | — | — |
//! | 8  | `max(deficit_segment_costs) > generation_violation_below_cost` (both $/`MWh`) | `penalties.json`                               | `ModelQuality` (warning) |
//! | 9  | `min(flow_violation_costs) > max(resource_costs)` (both $/(m³/s·h))     | `penalties.json`                               | `ModelQuality` (warning) |
//! |10  | `min(resource_costs) > 0`                                               | `penalties.json`                               | `ModelQuality` (warning) |
//! |11  | FPHA hydros: `turbined_cost >= 0`                                  | `penalties.json`                               | `BusinessRuleViolation`  |
//! |12  | `std_m3s >= 0.0`; warn when `== 0.0` (deterministic inflow) — suppressed for a class whose resolved scheme is External | `scenarios/inflow_seasonal_stats.parquet` | `ModelQuality` (warning) |
//! |13  | *(retired — number never reused)* | — | — |
//! |14  | Correlation matrix symmetry (`matrix[i][j] == matrix[j][i]` ±1e-9)     | `scenarios/correlation.json`                   | `BusinessRuleViolation`  |
//! |15  | Correlation matrix diagonal entries equal 1.0 (±1e-9)                  | `scenarios/correlation.json`                   | `BusinessRuleViolation`  |
//! |16  | Correlation off-diagonal entries in [-1.0, 1.0]                        | `scenarios/correlation.json`                   | `BusinessRuleViolation`  |
//! |16a | All entities within a correlation group share the same `entity_type`  | `scenarios/correlation.json`                   | `BusinessRuleViolation`  |
//! |17  | Each `block_factors[j].block_id` matches a `Block.index` in its stage  | `scenarios/load_factors.json`                  | `BusinessRuleViolation`  |
//! |18  | *(retired — number never reused)* | — | — |
//! |19  | `season_definitions` required in `stages.json` when estimating          | `scenarios/inflow_history.parquet`             | `BusinessRuleViolation`  |
//! |20  | Minimum observations per `(hydro, season)` group for estimation         | `scenarios/inflow_history.parquet`             | `ModelQuality` (warning) |
//! |21  | All hydros in `hydros.json` must have observations in history           | `scenarios/inflow_history.parquet`             | `BusinessRuleViolation`  |
//! |22  | *(retired — number never reused)* | — | — |
//! |23  | *(retired — number never reused)* | — | — |
//! |24  | *(retired — number never reused)* | — | — |
//! |25  | Sobol stages: `branching_factor` should be a power of 2                 | `stages.json`                                  | `ModelQuality` (warning) |
//! |26  | *(retired — `simulation.selection.method` is parse-layer enforced by a `#[serde(deny_unknown_fields)]`-tagged enum, not the semantic layer; number never reused)* | — | — |
//! |27  | Every stage `season_id` must reference a season defined in `season_definitions` | `stages.json`                        | `BusinessRuleViolation`  |
//! |28  | Season with zero observations when inflow scheme is not External         | `stages.json`                                  | `ModelQuality` (warning) |
//! |29  | All stages sharing a `season_id` must have compatible durations (within 7d) | `stages.json`                        | `BusinessRuleViolation`  |
//! |30  | Season defined in `season_definitions` but not referenced by any stage   | `stages.json`                                  | `ModelQuality` (warning) |
//! |31  | Observation-to-season alignment: finer-than-season observations are aggregated during PAR estimation (warning); an interior hydro-year missing a season under coarser-than-season observations cannot be disaggregated (error) | `scenarios/inflow_history.parquet` | `BusinessRuleViolation` |
//! |32  | *(retired — number never reused)* | — | — |
//! |33  | Filling schedule reaches the dead volume, within a relative tolerance: `Σ ζ_s·rate_s >= min_storage − seed` | `system/hydros.json` | `BusinessRuleViolation`  |
//! |34  | PAR order > 0 but every study stage has `inflow_lags == false` (inflow-lag state omitted) | `stages.json`        | `ModelQuality` (warning) |
//! |35  | User-supplied `inflow_ar_coefficients.parquet` must pass the periodic-ACF closure stationarity gate (external-input path only; annual-aware; season resolved via `resolve_stage_seasons`'s `season_map`-or-fallback) | `scenarios/inflow_ar_coefficients.parquet` | `InvalidValue` (or `BusinessRuleViolation` when a stage's season is genuinely unresolvable) |
//! |36  | Node `scenario_id` required at a stage carrying a slot-occupying external class, rejected as meaningless where none (declared `nodes[]`, enumerated forward selection only) | `stages.json` | `InvalidValue` |
//! |37  | Node `scenario_id` in `[0, raw_c(t))` for every slot-occupying external class (declared `nodes[]`, enumerated forward selection only) | `stages.json` | `InvalidValue` |
//! |38  | Node graph well-formedness: unique/known node ids, resolvable stage, no empty stage, no unreachable node, acyclic, no mid-horizon leaf (declared `nodes[]` only) | `stages.json` | `InvalidValue` / `DuplicateId` / `CycleDetected` |
//! |39  | Every graph edge advances exactly one stage (`t → t+1`, no stage-skipping) (declared `nodes[]` only) | `stages.json` | `InvalidValue` |
//! |40  | A stage carrying multiple nodes with structurally identical subtrees (recombinable signature) (declared `nodes[]` only) | `stages.json` | `ModelQuality` (warning) |
//! |41  | `num_openings` required at a stage carrying generated openings, rejected as meaningless where a stage carries only external openings (declared `nodes[]` only; chain-dialect requiredness is a parse-layer check) | `stages.json` | `InvalidValue` |
//! |42  | Per-edge `annual_discount_rate_override` rejected under `nodes[]` — the override is a per-stage quantity on `stages[]` (legal in the chain dialect) | `stages.json` | `InvalidValue` |
//! |43  | `scenarios/noise_openings.parquet` present under enumerated forward selection — the generated backward opening tree is not consumed there | `stages.json` | `InvalidValue` |
//! |44  | `sampling_method` inert under external openings / ill-defined at a multi-node stage (declared `nodes[]` only) | `stages.json` | `ModelQuality` (warning) |
//! |44a | A class resolved to the `External` scheme has non-empty `external_*_scenarios.parquet` data | `config.json` | `BusinessRuleViolation` |
//! |45  | All slot-occupying external classes agree on the per-stage raw column-count vector `raw_c(t)` — no element-wise-minimum reconciliation, fires with or without `nodes[]` (P-B1) | `scenarios/external_*_scenarios.parquet` | `BusinessRuleViolation` |
//! |46  | Every (slot-occupying external class, stage) carries the exact `scenario_id` set `{0..raw_c(t)-1}` per entity — a set check (rejects 1-based deck, gap, duplicate, out-of-range), not a bound check (A1) | `scenarios/external_*_scenarios.parquet` | `BusinessRuleViolation` |
//! |47  | Every external scenario row's `stage_id` resolves to a declared study stage via the [`crate::StageIdResolver`], never silently dropped (A2) | `scenarios/external_*_scenarios.parquet` | `InvalidValue` |
//! |48  | Per edge `n → m` and slot-occupying external class, the raw cells of columns `scenario_id(n)`/`scenario_id(m)` agree bitwise over the shared prefix `s <= t(n)` (declared `nodes[]` only) | `scenarios/external_*_scenarios.parquet` | `ModelQuality` (warning) |
//! |50  | Under External, load/NCS get no σ check at all (their μ is defined by the external file itself, so there is no seasonal μ left to disagree with); inflow's remaining σ = 0 case is decided from the same external cells' own sample σ ([`cobre_stochastic::derive_external_sample_moments`], the reduction the engine also derives its `(μ, σ)` from) — accepted for an AR(0) hydro (no declared lag coefficient or annual component: its deterministic base is exactly μ), rejected for an AR(p > 0) hydro, naming the entity and stage, since a deterministic value there would have to equal that model's own deterministic PAR output, which this loader does not compute upstream | `scenarios/external_*_scenarios.parquet` | `BusinessRuleViolation` |
//! |52  | Every study stage declares at least one block and every block's `hours` is finite and `> 0` | `stages.json` | `InvalidValue` |
//!
//! Rule 49 (G2 — each standardized external library's `n_entities()` matches its
//! `noise_entity_order` block width) is enforced downstream at study setup
//! (`build_scenario_libraries`), where the standardized libraries exist; it is
//! not a pre-build load-time semantic rule.

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
