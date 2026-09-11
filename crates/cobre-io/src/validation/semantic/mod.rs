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
//! | # | Rule                                              | Source file                           | `ErrorKind`            |
//! |---|---------------------------------------------------|---------------------------------------|------------------------|
//! | 1 | Hydro cascade graph must be acyclic               | `system/hydros.json`                  | `CycleDetected`        |
//! | 2 | `min_storage_hm3 <= max_storage_hm3`              | `system/hydros.json`                  | `InvalidValue`         |
//! | 3 | `min_turbined_m3s <= max_turbined_m3s`            | `system/hydros.json`                  | `InvalidValue`         |
//! | 4 | `min_outflow_m3s <= max_outflow_m3s` (when Some)  | `system/hydros.json`                  | `InvalidValue`         |
//! | 5 | `min_generation_mw <= max_generation_mw` (hydro)  | `system/hydros.json`                  | `InvalidValue`         |
//! | 6 | `entry_stage_id < exit_stage_id` (when both Some) | all six entity types                  | `InvalidValue`         |
//! | 7 | Filling `start_stage_id` in study stage set       | `system/hydros.json`                  | `InvalidValue`         |
//! | 7a| Filling guards (hard): `filling ⟹ entry_stage_id` (a bare window without filling is valid), `start_stage_id < entry_stage_id`, seed in `[0, min_storage_hm3)`, no `exit_stage_id` on a filling hydro, seed `== 0` when `start_stage_id > 0` | `system/hydros.json` | `InvalidValue` |
//! | 7b| `entry_stage_id >= horizon` on a filling hydro (fills throughout, never operates within this study) | `system/hydros.json` | `ModelQuality` (warning) |
//! | 8 | Geometry `volume_hm3` strictly increasing         | `system/hydro_geometry.parquet`       | `BusinessRuleViolation`|
//! | 9 | Geometry `height_m` non-decreasing (within a relative tolerance) | `system/hydro_geometry.parquet` | `BusinessRuleViolation`|
//! |10 | Geometry `area_km2` non-decreasing (within a relative tolerance) | `system/hydro_geometry.parquet` | `BusinessRuleViolation`|
//! |11 | FPHA: at least 1 plane per (hydro, stage)         | `system/fpha_hyperplanes.parquet`     | `BusinessRuleViolation`|
//! |12 | FPHA: `gamma_v >= 0`, `gamma_s <= 0`              | `system/fpha_hyperplanes.parquet`     | `BusinessRuleViolation`|
//! |13 | `min_generation_mw <= max_generation_mw` (thermal)| `system/thermals.json`                | `InvalidValue`         |
//! |14 | Anticipated thermal `lead_stages` within study horizon and lifecycle bounds | `system/thermals.json` | `BusinessRuleViolation` |
//! |15 | Anticipated thermals bijection with `past_anticipated_commitments` entries  | `initial_conditions.json` | `BusinessRuleViolation` |
//! |16 | Thermal `thermal_bounds.parquet` override `stage_id` within `[0, n_stages)` | `constraints/thermal_bounds.parquet` | `BusinessRuleViolation` |
//! |17 | `anticipated_decision(N)` in generic constraint targets an anticipated thermal | `constraints/generic_constraints.json` | `BusinessRuleViolation` |
//! |18 | `thermal_generation(N)` in generic constraint when `N` is anticipated (warn) | `constraints/generic_constraints.json` | `SemanticAmbiguity` (warning) |
//! |19 | Pumping `source_hydro_id != destination_hydro_id`  | `system/pumping_stations.json`        | `InvalidValue`         |
//! |20 | Per-block storage reference resolves to a real boundary (parallel `K>1` interior / out-of-range block rejected) | `constraints/generic_constraints.json` | `BusinessRuleViolation` |
//! |21 | `travel_time_hours` negative or non-finite                             | `system/hydros.json`                  | `InvalidValue`         |
//! |22 | `travel_time_hours == 0.0` — treated as undeclared, no arc created     | `system/hydros.json`                  | `ModelQuality` (warning) |
//! |23 | Declared arc: `max_t(t_v/h_t)` below a smallness threshold             | `system/hydros.json`                  | `ModelQuality` (warning) |
//! |24 | Declared arc: `t_v` exceeds the remaining study horizon at some stage  | `system/hydros.json`                  | `ModelQuality` (warning) |
//! |25 | Declared arc: `past_defluences` windows do not cover the arc's required pre-study depth | `initial_conditions.json` | `BusinessRuleViolation` (or `ModelQuality` warning) |
//! |26 | 2+ declared arcs into one downstream plant with differing `travel_time_hours`, while any study stage is `Chronological` | `system/hydros.json` | `NotImplemented` |
//! |27 | *(retired — number never reused)* | — | — |
//! |28 | `lead_stages` anticipated active window spans a stage-cadence transition (adjacent unequal stage durations); `lead_time` is the physically-anchored alternative | `system/thermals.json` | `ModelQuality` (warning) |
//! |29 | Study supplies an inflow annual component (`inflow_annual_components` non-empty) while `season_map.cycle_type` is not `Monthly` — PAR(p)-A is monthly-exclusive by design | `scenarios/inflow_annual_component.parquet` | `BusinessRuleViolation` |
//! |30 | PAR lag slots `1..=max_AR_order` (read by the PAR equation at every stage) must have full record/conditioning coverage (`coverage == 1.0`) | `scenarios/inflow_history.parquet` | `BusinessRuleViolation` |
//! |31 | PAR lag slots `max_AR_order < s <= L_state − n_fin` require the same full coverage (still terminal-reachable); slots `s > L_state − n_fin` are provably never read, so a gap there is advisory only | `scenarios/inflow_history.parquet` | `BusinessRuleViolation` / `ModelQuality` (warning) |
//! |32 | A `recent_observations` conditioning window extends past the study start, into the solved study itself | `initial_conditions.json` | `InvalidValue` |
//! |33 | The in-progress period `[period_start, study_start)` is covered strictly between 0 and 1 | `scenarios/inflow_history.parquet` | `ModelQuality` (warning) |
//! |34 | The first study stage's season is unresolvable (no `season_map`, no `season_id`, or an unmatched id) while PAR seeding is active | `initial_conditions.json` | `ModelQuality` (warning) |
//! |35 | Bound-override row `block_id` within `[0, n_blocks)` for its stage, across all six bound families (thermal, hydro, line, pumping, contract, hydro unit group) | `constraints/*_bounds.parquet` | `BusinessRuleViolation` |
//! |36 | Bound-override row uniqueness per `(entity_id, stage_id, block_id, column)` — widened to `(hydro_id, hydro_unit_group_id, stage_id, block_id, column)` for the hydro unit group family — across all six bound families (thermal, hydro, line, pumping, contract, hydro unit group); a `None` `block_id` is a distinct key from `Some(b)` | `constraints/*_bounds.parquet` | `DuplicateId` |
//! |37 | `block_id` on a hydro/thermal bound column with no per-block LP variable (hydro storage/filling-rate/withdrawal, thermal cost) | `constraints/{hydro,thermal}_bounds.parquet` | `BusinessRuleViolation` |
//! |38 | `block_id` on a `thermal_bounds` row targeting an anticipated thermal (commitment decision is stage-level; delivery-stage reconciliation compares per-block bounds) | `constraints/thermal_bounds.parquet` | `BusinessRuleViolation` |
//! |39 | Hydro unit group `id` unique within its own plant (ids are plant-scoped, not global) | `system/hydros.json` | `DuplicateId` |
//! |40 | Hydro unit group bounds internally consistent: `min_turbined_m3s <= max_turbined_m3s` and `min_generation_mw <= max_generation_mw`, checked independently | `system/hydros.json` | `InvalidValue` |
//! |41 | Sum of unit group maxima (`max_turbined_m3s`, `max_generation_mw`, checked independently) must not exceed the plant's own value; checked against the entity declaration only, never against per-stage resolved bounds | `system/hydros.json` | `InvalidValue` |
//! |42 | *(retired — turbined-bound sign is a parse-layer check, not semantic; see note below)* | — | — |
//! |43 | `hydro_bounds` row `max_turbined_m3s`/`max_generation_mw` must not exceed the hydro's own declared value (checked independently); scope is these two columns only — lowering, `min_*`/storage/filling/withdrawal, and the other four bound families are untouched, each a separate decision with its own back-compat surface | `constraints/hydro_bounds.parquet` | `InvalidValue` |
//! |44 | Sum of unit group minima (`min_turbined_m3s`, `min_generation_mw`, checked independently) must reach the plant's own declared value — the flipped direction of rule 41: rule 41 caps `Σ group max ≤ plant max`, this floors `Σ group min ≥ plant min`; checked against the entity declaration only, never against per-stage resolved bounds | `system/hydros.json` | `InvalidValue` |
//! |45 | `hydro_unit_group_bounds` row `max_turbined_m3s`/`max_generation_mw` must not exceed that GROUP's own declared value (checked independently) — the group-axis mirror of rule 43, which checks the plant's own declared value instead | `constraints/hydro_unit_group_bounds.parquet` | `InvalidValue` |
//! |46 | `hydro_bounds` row `min_diversion_m3s` set for a hydro declaring no `diversion` channel (the channel is pinned `[0, 0]` with none declared, making a positive floor infeasible); cross-row/cross-source min/max inversion is deliberately out of scope for this rule | `constraints/hydro_bounds.parquet` | `InvalidValue` |
//! |47 | Post-study boundary (`post_study_stages.json`), the sole post-horizon surface: stages date-contiguous with first `start_date` at the study horizon end (a); a `PostStudyThermalBound` for every post-study stage an anticipated thermal's extended lead reaches from an in-study, commissioning-active decision (Rule 1); the plant's pre-study-decided post-study stages tiled by `past_anticipated_commitments` at coverage `1.0`, an explicit `0 MW` window included where required (V2); no commitment window covering a study-decided or beyond-reach post-study stage (V3); a non-zero fixed value only inside the plant's commissioning window at its delivery stage (V5) | `post_study_stages.json` | `BusinessRuleViolation` |
//! |48 | *(retired — number never reused)* | — | — |
//! |49 | Bound-override row `stage_id` a member of the study stage id set, across five of the six bound families (hydro, line, pumping, contract, hydro unit group); thermal (rule 16) and NCS (the Layer-3 referential check) already own that stage-axis defect for their own families | `constraints/*_bounds.parquet` | `BusinessRuleViolation` |
//!
//! A hydro unit group bounds row's `block_id` range and duplicate-row keying
//! are covered by rules 35 and 36 above; a row referencing a non-existent
//! unit group id is still checked by `check_bounds_references` (Layer 3), not
//! here. Hydro unit group turbined-bound sign (`min_turbined_m3s >= 0` and
//! `max_turbined_m3s >= 0`, retired rule 42's slot) is validated at the PARSE
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
//! | 4  | Every `Block.duration_hours > 0.0`                                      | `stages.json`                                  | `InvalidValue`           |
//! | 5  | `CVaR`: `alpha` in (0, 1], `lambda` in [0, 1]                             | `stages.json`                                  | `InvalidValue`           |
//! | 6  | `max(deficit_segment_costs) > filling_target_violation_cost`            | `penalties.json`                               | `ModelQuality` (warning) |
//! | 7  | `storage_violation_below_cost > max(deficit_segment_costs)`             | `penalties.json`                               | `ModelQuality` (warning) |
//! | 8  | `max(deficit_segment_costs) > max(constraint_violation_costs)`          | `penalties.json`                               | `ModelQuality` (warning) |
//! | 9  | `min(constraint_violation_costs) > max(resource_costs)`                 | `penalties.json`                               | `ModelQuality` (warning) |
//! |10  | `min(resource_costs) > 0`                                               | `penalties.json`                               | `ModelQuality` (warning) |
//! |11  | FPHA hydros: `turbined_cost >= 0`                                  | `penalties.json`                               | `BusinessRuleViolation`  |
//! |12  | `std_m3s >= 0.0`; warn when `== 0.0` (deterministic inflow) — suppressed for a class whose resolved scheme is External | `scenarios/inflow_seasonal_stats.parquet` | `ModelQuality` (warning) |
//! |13  | *(retired — number never reused)* | — | — |
//! |14  | Correlation matrix symmetry (`matrix[i][j] == matrix[j][i]` ±1e-9)     | `scenarios/correlation.json`                   | `BusinessRuleViolation`  |
//! |15  | Correlation matrix diagonal entries equal 1.0 (±1e-9)                  | `scenarios/correlation.json`                   | `BusinessRuleViolation`  |
//! |16  | Correlation off-diagonal entries in [-1.0, 1.0]                        | `scenarios/correlation.json`                   | `BusinessRuleViolation`  |
//! |17  | Each `block_factors[j].block_id` matches a `Block.index` in its stage  | `scenarios/load_factors.json`                  | `BusinessRuleViolation`  |
//! |18  | *(retired — number never reused)* | — | — |
//! |19  | `season_definitions` required in `stages.json` when estimating          | `scenarios/inflow_history.parquet`             | `BusinessRuleViolation`  |
//! |20  | Minimum observations per `(hydro, season)` group for estimation         | `scenarios/inflow_history.parquet`             | `ModelQuality` (warning) |
//! |21  | All hydros in `hydros.json` must have observations in history           | `scenarios/inflow_history.parquet`             | `BusinessRuleViolation`  |
//! |22  | *(retired — number never reused)* | — | — |
//! |23  | *(retired — number never reused)* | — | — |
//! |24  | *(retired — number never reused)* | — | — |
//! |25  | Sobol stages: `branching_factor` should be a power of 2                 | `stages.json`                                  | `ModelQuality` (warning) |
//! |26  | `simulation.sampling_scheme.type` must be a known scheme string          | `config.json`                                  | `InvalidValue`           |
//! |27  | Every stage `season_id` must reference a season defined in `season_definitions` | `stages.json`                        | `BusinessRuleViolation`  |
//! |28  | Season with zero observations when inflow scheme is not External         | `stages.json`                                  | `ModelQuality` (warning) |
//! |29  | All stages sharing a `season_id` must have compatible durations (within 7d) | `stages.json`                        | `BusinessRuleViolation`  |
//! |30  | Season defined in `season_definitions` but not referenced by any stage   | `stages.json`                                  | `ModelQuality` (warning) |
//! |31  | Observation resolution must not be finer than season resolution          | `scenarios/inflow_history.parquet`             | `BusinessRuleViolation`  |
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
//! |45  | All slot-occupying external classes agree on the per-stage raw column-count vector `raw_c(t)` — no element-wise-minimum reconciliation, fires with or without `nodes[]` (P-B1) | `scenarios/external_*_scenarios.parquet` | `BusinessRuleViolation` |
//! |46  | Every (slot-occupying external class, stage) carries the exact `scenario_id` set `{0..raw_c(t)-1}` per entity — a set check (rejects 1-based deck, gap, duplicate, out-of-range), not a bound check (A1) | `scenarios/external_*_scenarios.parquet` | `BusinessRuleViolation` |
//! |47  | Every external scenario row's `stage_id` resolves to a declared study stage via the [`crate::StageIdResolver`], never silently dropped (A2) | `scenarios/external_*_scenarios.parquet` | `InvalidValue` |
//! |48  | Per edge `n → m` and slot-occupying external class, the raw cells of columns `scenario_id(n)`/`scenario_id(m)` agree bitwise over the shared prefix `s <= t(n)` (declared `nodes[]` only) | `scenarios/external_*_scenarios.parquet` | `ModelQuality` (warning) |
//! |50  | Under External, load/NCS get no σ check at all (their μ is defined by the external file itself, so there is no seasonal μ left to disagree with); inflow's remaining σ = 0 case is decided from the same external cells' own sample σ ([`cobre_stochastic::derive_external_sample_moments`], the reduction the engine also derives its `(μ, σ)` from) — accepted for an AR(0) hydro (no declared lag coefficient or annual component: its deterministic base is exactly μ), rejected for an AR(p > 0) hydro, naming the entity and stage, since a deterministic value there would have to equal that model's own deterministic PAR output, which this loader does not compute upstream | `scenarios/external_*_scenarios.parquet` | `BusinessRuleViolation` |
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

#[cfg(test)]
mod test_support;

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
    thermal::check_thermal_bounds_override_stage_range(data, ctx);
    thermal::check_post_study_stages(data, ctx);
    thermal::check_anticipated_decision_target_is_anticipated(data, ctx);
    thermal::warn_thermal_generation_on_anticipated_thermal(data, ctx);
    constraints::check_per_block_storage_interior_reference(data, ctx);
    block_bounds::check_bound_block_id_range(data, ctx);
    block_bounds::check_bound_stage_id_range(data, ctx);
    block_bounds::check_duplicate_bound_rows(data, ctx);
    block_bounds::check_block_id_on_ineligible_column(data, ctx);
    block_bounds::check_block_id_on_anticipated_thermal(data, ctx);
    block_bounds::check_bound_raises_declared_capacity(data, ctx);
    block_bounds::check_group_bound_raises_declared_capacity(data, ctx);
    pumping::check_pumping_semantics(data, ctx);
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
