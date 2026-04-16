//! Layer 5 — Semantic validation: hydro, thermal, stage, penalty, and scenario rules.
//!
//! Validates all domain-specific business rules after Layers 2-4 have
//! ensured schema correctness, referential integrity, and dimensional
//! consistency.
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
//! | 6 | `entry_stage_id < exit_stage_id` (when both Some) | hydros/lines/thermals                 | `InvalidValue`         |
//! | 7 | Filling `start_stage_id` in study stage set       | `system/hydros.json`                  | `InvalidValue`         |
//! | 8 | Geometry `volume_hm3` strictly increasing         | `system/hydro_geometry.parquet`       | `BusinessRuleViolation`|
//! | 9 | Geometry `height_m` non-decreasing                | `system/hydro_geometry.parquet`       | `BusinessRuleViolation`|
//! |10 | Geometry `area_km2` non-decreasing                | `system/hydro_geometry.parquet`       | `BusinessRuleViolation`|
//! |11 | FPHA: at least 1 plane per (hydro, stage)         | `system/fpha_hyperplanes.parquet`     | `BusinessRuleViolation`|
//! |12 | FPHA: `gamma_v >= 0`, `gamma_s <= 0`              | `system/fpha_hyperplanes.parquet`     | `BusinessRuleViolation`|
//! |13 | `min_generation_mw <= max_generation_mw` (thermal)| `system/thermals.json`                | `InvalidValue`         |
//!
//! ## Layer 5b rules (stages, penalties, and scenario domain) — `validate_semantic_stages_penalties_scenarios`
//!
//! | #  | Rule                                                                    | Source file                                    | `ErrorKind`              |
//! |----|-------------------------------------------------------------------------|------------------------------------------------|--------------------------|
//! | 1  | Every transition `source_id`/`target_id` must refer to an existing stage| `stages.json`                                  | `InvalidValue`           |
//! | 2  | Outgoing transition probabilities sum to 1.0 (±1e-6) per source stage  | `stages.json`                                  | `InvalidValue`           |
//! | 3  | Cyclic graph: `annual_discount_rate > 0.0`                              | `stages.json`                                  | `InvalidValue`           |
//! | 4  | Every `Block.duration_hours > 0.0`                                      | `stages.json`                                  | `InvalidValue`           |
//! | 5  | CVaR: `alpha` in (0, 1], `lambda` in [0, 1]                             | `stages.json`                                  | `InvalidValue`           |
//! | 6  | `filling_target_violation_cost > storage_violation_below_cost`          | `penalties.json`                               | `ModelQuality` (warning) |
//! | 7  | `storage_violation_below_cost > max(deficit_segment_costs)`             | `penalties.json`                               | `ModelQuality` (warning) |
//! | 8  | `max(deficit_segment_costs) > max(constraint_violation_costs)`          | `penalties.json`                               | `ModelQuality` (warning) |
//! | 9  | `min(constraint_violation_costs) > max(resource_costs)`                 | `penalties.json`                               | `ModelQuality` (warning) |
//! |10  | `min(resource_costs) > 0`                                               | `penalties.json`                               | `ModelQuality` (warning) |
//! |11  | FPHA hydros: `fpha_turbined_cost >= 0`                                  | `penalties.json`                               | `BusinessRuleViolation`  |
//! |12  | `std_m3s >= 0.0`; warn when `== 0.0` (deterministic inflow)            | `scenarios/inflow_seasonal_stats.parquet`      | `ModelQuality` (warning) |
//! |13  | `residual_std_ratio` consistent across all lag rows of same group       | `scenarios/inflow_ar_coefficients.parquet`     | `InvalidValue`           |
//! |14  | Correlation matrix symmetry (`matrix[i][j] == matrix[j][i]` ±1e-9)     | `scenarios/correlation.json`                   | `BusinessRuleViolation`  |
//! |15  | Correlation matrix diagonal entries equal 1.0 (±1e-9)                  | `scenarios/correlation.json`                   | `BusinessRuleViolation`  |
//! |16  | Correlation off-diagonal entries in [-1.0, 1.0]                        | `scenarios/correlation.json`                   | `BusinessRuleViolation`  |
//! |17  | Each `block_factors[j].block_id` matches a `Block.index` in its stage  | `scenarios/load_factors.json`                  | `BusinessRuleViolation`  |
//! |18  | Load-factors entry for `(bus_id, stage_id)` with `std_mw == 0.0`       | `scenarios/load_factors.json`                  | `ModelQuality` (warning) |
//! |19  | `season_definitions` required in `stages.json` when estimating          | `scenarios/inflow_history.parquet`             | `BusinessRuleViolation`  |
//! |20  | Minimum observations per `(hydro, season)` group for estimation         | `scenarios/inflow_history.parquet`             | `ModelQuality` (warning) |
//! |21  | All hydros in `hydros.json` must have observations in history           | `scenarios/inflow_history.parquet`             | `BusinessRuleViolation`  |
//! |22  | `inflow_lags: true` with PAR order > 0 requires non-empty `past_inflows` | `initial_conditions.json`                      | `BusinessRuleViolation`  |
//! |23  | Each hydro with PAR order `p` must have a `past_inflows` entry with `values_m3s.len() >= p` | `initial_conditions.json` | `BusinessRuleViolation`  |
//! |24  | All hydro IDs in `past_inflows` must exist in the hydro registry        | `initial_conditions.json`                      | `BusinessRuleViolation`  |
//! |25  | Sobol stages: `branching_factor` should be a power of 2                 | `stages.json`                                  | `ModelQuality` (warning) |
//! |26  | `simulation.sampling_scheme.type` must be a known scheme string          | `config.json`                                  | `InvalidValue`           |
//! |27  | Every stage `season_id` must reference a season defined in `season_definitions` | `stages.json`                        | `BusinessRuleViolation`  |
//! |28  | Season with zero observations when inflow scheme is not External         | `stages.json`                                  | `ModelQuality` (warning) |
//! |29  | All stages sharing a `season_id` must have compatible durations (within 7d) | `stages.json`                        | `BusinessRuleViolation`  |
//! |30  | Season defined in `season_definitions` but not referenced by any stage   | `stages.json`                                  | `ModelQuality` (warning) |
//! |31  | Observation resolution must not be finer than season resolution          | `scenarios/inflow_history.parquet`             | `BusinessRuleViolation`  |
//! |32  | Each `season_id` in `past_inflows[i].season_ids` must exist in `SeasonMap` | `initial_conditions.json`                    | `BusinessRuleViolation`  |

use std::collections::{HashMap, HashSet};

use super::{ErrorKind, ValidationContext, schema::ParsedData};

pub(crate) fn validate_semantic_hydro_thermal(data: &ParsedData, ctx: &mut ValidationContext) {
    check_cascade_acyclic(data, ctx);
    check_hydro_bounds(data, ctx);
    check_lifecycle_consistency(data, ctx);
    check_filling_config(data, ctx);
    check_geometry_monotonicity(data, ctx);
    check_evaporation_geometry_coverage(data, ctx);
    check_fpha_constraints(data, ctx);
    check_thermal_generation_bounds(data, ctx);
}

fn check_cascade_acyclic(data: &ParsedData, ctx: &mut ValidationContext) {
    if data.hydros.is_empty() {
        return;
    }

    let all_ids: Vec<i32> = data.hydros.iter().map(|h| h.id.0).collect();
    let downstream_set: HashSet<i32> = all_ids.iter().copied().collect();

    let mut adjacency: HashMap<i32, Vec<i32>> =
        all_ids.iter().copied().map(|id| (id, Vec::new())).collect();
    for hydro in &data.hydros {
        if let Some(ds) = hydro.downstream_id {
            if downstream_set.contains(&ds.0) {
                adjacency.entry(hydro.id.0).or_default().push(ds.0);
            }
        }
    }

    let mut in_degree: HashMap<i32, usize> = all_ids.iter().copied().map(|id| (id, 0)).collect();
    for hydro in &data.hydros {
        if let Some(ds) = hydro.downstream_id {
            if downstream_set.contains(&ds.0) {
                *in_degree.entry(ds.0).or_insert(0) += 1;
            }
        }
    }

    let mut queue: std::collections::VecDeque<i32> = in_degree
        .iter()
        .filter(|&(_, deg)| *deg == 0)
        .map(|(&id, _)| id)
        .collect();

    let mut visited_count: usize = 0;

    while let Some(node) = queue.pop_front() {
        visited_count += 1;
        if let Some(neighbors) = adjacency.get(&node) {
            for &neighbor in neighbors {
                let deg = in_degree.entry(neighbor).or_insert(0);
                if *deg > 0 {
                    *deg -= 1;
                }
                if *deg == 0 {
                    queue.push_back(neighbor);
                }
            }
        }
    }

    if visited_count < all_ids.len() {
        let mut cycle_participants: Vec<i32> = in_degree
            .iter()
            .filter(|&(_, deg)| *deg > 0)
            .map(|(&id, _)| id)
            .collect();
        cycle_participants.sort_unstable();

        ctx.add_error(
            ErrorKind::CycleDetected,
            "system/hydros.json",
            None::<&str>,
            format!(
                "hydro cascade contains a cycle involving hydro IDs: [{}]",
                cycle_participants
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        );
    }
}

fn check_hydro_bounds(data: &ParsedData, ctx: &mut ValidationContext) {
    for hydro in &data.hydros {
        let entity_str = format!("Hydro {}", hydro.id.0);

        if hydro.min_storage_hm3 > hydro.max_storage_hm3 {
            ctx.add_error(
                ErrorKind::InvalidValue,
                "system/hydros.json",
                Some(&entity_str),
                format!(
                    "{entity_str}: min_storage_hm3 ({}) > max_storage_hm3 ({}); storage bounds are inconsistent",
                    hydro.min_storage_hm3, hydro.max_storage_hm3
                ),
            );
        }

        if hydro.min_turbined_m3s > hydro.max_turbined_m3s {
            ctx.add_error(
                ErrorKind::InvalidValue,
                "system/hydros.json",
                Some(&entity_str),
                format!(
                    "{entity_str}: min_turbined_m3s ({}) > max_turbined_m3s ({}); turbine bounds are inconsistent",
                    hydro.min_turbined_m3s, hydro.max_turbined_m3s
                ),
            );
        }

        if let Some(max_outflow) = hydro.max_outflow_m3s {
            if hydro.min_outflow_m3s > max_outflow {
                ctx.add_error(
                    ErrorKind::InvalidValue,
                    "system/hydros.json",
                    Some(&entity_str),
                    format!(
                        "{entity_str}: min_outflow_m3s ({}) > max_outflow_m3s ({}); outflow bounds are inconsistent",
                        hydro.min_outflow_m3s, max_outflow
                    ),
                );
            }
        }

        if hydro.min_generation_mw > hydro.max_generation_mw {
            ctx.add_error(
                ErrorKind::InvalidValue,
                "system/hydros.json",
                Some(&entity_str),
                format!(
                    "{entity_str}: min_generation_mw ({}) > max_generation_mw ({}); generation bounds are inconsistent",
                    hydro.min_generation_mw, hydro.max_generation_mw
                ),
            );
        }
    }
}

fn check_lifecycle_consistency(data: &ParsedData, ctx: &mut ValidationContext) {
    for hydro in &data.hydros {
        if let (Some(entry), Some(exit)) = (hydro.entry_stage_id, hydro.exit_stage_id) {
            if entry >= exit {
                let entity_str = format!("Hydro {}", hydro.id.0);
                ctx.add_error(
                    ErrorKind::InvalidValue,
                    "system/hydros.json",
                    Some(&entity_str),
                    format!(
                        "{entity_str}: entry_stage_id ({entry}) >= exit_stage_id ({exit}); entry must precede exit"
                    ),
                );
            }
        }
    }

    for line in &data.lines {
        if let (Some(entry), Some(exit)) = (line.entry_stage_id, line.exit_stage_id) {
            if entry >= exit {
                let entity_str = format!("Line {}", line.id.0);
                ctx.add_error(
                    ErrorKind::InvalidValue,
                    "system/lines.json",
                    Some(&entity_str),
                    format!(
                        "{entity_str}: entry_stage_id ({entry}) >= exit_stage_id ({exit}); entry must precede exit"
                    ),
                );
            }
        }
    }

    for thermal in &data.thermals {
        if let (Some(entry), Some(exit)) = (thermal.entry_stage_id, thermal.exit_stage_id) {
            if entry >= exit {
                let entity_str = format!("Thermal {}", thermal.id.0);
                ctx.add_error(
                    ErrorKind::InvalidValue,
                    "system/thermals.json",
                    Some(&entity_str),
                    format!(
                        "{entity_str}: entry_stage_id ({entry}) >= exit_stage_id ({exit}); entry must precede exit"
                    ),
                );
            }
        }
    }
}

fn check_filling_config(data: &ParsedData, ctx: &mut ValidationContext) {
    let study_stage_ids: HashSet<i32> = data
        .stages
        .stages
        .iter()
        .filter(|s| s.id >= 0)
        .map(|s| s.id)
        .collect();

    for hydro in &data.hydros {
        if let Some(filling) = &hydro.filling {
            if !study_stage_ids.contains(&filling.start_stage_id) {
                let entity_str = format!("Hydro {}", hydro.id.0);
                ctx.add_error(
                    ErrorKind::InvalidValue,
                    "system/hydros.json",
                    Some(&entity_str),
                    format!(
                        "{entity_str}: filling.start_stage_id ({}) is not a valid study stage ID",
                        filling.start_stage_id
                    ),
                );
            }
        }
    }
}

fn check_geometry_monotonicity(data: &ParsedData, ctx: &mut ValidationContext) {
    if data.hydro_geometry.is_empty() {
        return;
    }

    let mut i = 0;
    let rows = &data.hydro_geometry;

    while i < rows.len() {
        let current_hydro_id = rows[i].hydro_id.0;
        let group_start = i;

        // Find end of this hydro's group (rows are sorted by hydro_id then volume_hm3).
        while i < rows.len() && rows[i].hydro_id.0 == current_hydro_id {
            i += 1;
        }
        let group = &rows[group_start..i];

        for pair in group.windows(2) {
            let prev = &pair[0];
            let curr = &pair[1];
            let entity_str = format!("Hydro {current_hydro_id}");

            if curr.volume_hm3 <= prev.volume_hm3 {
                ctx.add_error(
                    ErrorKind::BusinessRuleViolation,
                    "system/hydro_geometry.parquet",
                    Some(&entity_str),
                    format!(
                        "{entity_str}: volume_hm3 values are not strictly increasing ({} then {}); geometry curve must have strictly increasing volume",
                        prev.volume_hm3, curr.volume_hm3
                    ),
                );
            }

            if curr.height_m < prev.height_m {
                ctx.add_error(
                    ErrorKind::BusinessRuleViolation,
                    "system/hydro_geometry.parquet",
                    Some(&entity_str),
                    format!(
                        "{entity_str}: height_m values are not non-decreasing ({} then {}); geometry curve must have non-decreasing height with volume",
                        prev.height_m, curr.height_m
                    ),
                );
            }

            if curr.area_km2 < prev.area_km2 {
                ctx.add_error(
                    ErrorKind::BusinessRuleViolation,
                    "system/hydro_geometry.parquet",
                    Some(&entity_str),
                    format!(
                        "{entity_str}: area_km2 values are not non-decreasing ({} then {}); geometry curve must have non-decreasing area with volume",
                        prev.area_km2, curr.area_km2
                    ),
                );
            }
        }
    }
}

/// Hydros with `evaporation_coefficients_mm` require geometry rows in
/// `hydro_geometry.parquet` (area-volume curve for linearization).
fn check_evaporation_geometry_coverage(data: &ParsedData, ctx: &mut ValidationContext) {
    let geometry_hydro_ids: HashSet<i32> =
        data.hydro_geometry.iter().map(|r| r.hydro_id.0).collect();

    for hydro in &data.hydros {
        if hydro.evaporation_coefficients_mm.is_some() && !geometry_hydro_ids.contains(&hydro.id.0)
        {
            ctx.add_error(
                ErrorKind::BusinessRuleViolation,
                "system/hydros.json",
                Some(format!("Hydro {} (id={})", hydro.name, hydro.id.0)),
                format!(
                    "hydro {} (id={}) has evaporation_coefficients_mm but no geometry data \
                     in hydro_geometry.parquet; evaporation linearization requires \
                     area-volume curve data",
                    hydro.name, hydro.id.0
                ),
            );
        }
    }
}

fn check_fpha_constraints(data: &ParsedData, ctx: &mut ValidationContext) {
    if data.fpha_hyperplanes.is_empty() {
        return;
    }

    for row in &data.fpha_hyperplanes {
        let entity_str = format!("Hydro {}", row.hydro_id.0);

        if row.gamma_v < 0.0 {
            ctx.add_error(
                ErrorKind::BusinessRuleViolation,
                "system/fpha_hyperplanes.parquet",
                Some(&entity_str),
                format!(
                    "{entity_str} (stage={}, plane={}): gamma_v ({}) must be non-negative (>= 0); \
                     power must not decrease with volume/head (zero is valid for constant-head plants)",
                    row.stage_id.map_or_else(|| "all".to_string(), |s| s.to_string()),
                    row.plane_id,
                    row.gamma_v
                ),
            );
        }

        if row.gamma_s > 0.0 {
            ctx.add_error(
                ErrorKind::BusinessRuleViolation,
                "system/fpha_hyperplanes.parquet",
                Some(&entity_str),
                format!(
                    "{entity_str} (stage={}, plane={}): gamma_s ({}) must be non-positive (<= 0); power must not increase with spillage",
                    row.stage_id.map_or_else(|| "all".to_string(), |s| s.to_string()),
                    row.plane_id,
                    row.gamma_s
                ),
            );
        }
    }

    let rows = &data.fpha_hyperplanes;
    let mut i = 0;

    while i < rows.len() {
        let current_hydro_id = rows[i].hydro_id.0;
        let current_stage_id = rows[i].stage_id;
        let group_start = i;

        while i < rows.len()
            && rows[i].hydro_id.0 == current_hydro_id
            && rows[i].stage_id == current_stage_id
        {
            i += 1;
        }

        let plane_count = i - group_start;

        if plane_count < 1 {
            let entity_str = format!("Hydro {current_hydro_id}");
            let stage_label = current_stage_id.map_or_else(|| "all".to_string(), |s| s.to_string());
            ctx.add_error(
                ErrorKind::BusinessRuleViolation,
                "system/fpha_hyperplanes.parquet",
                Some(&entity_str),
                format!(
                    "{entity_str} (stage={stage_label}): no FPHA planes defined; \
                     at least 1 plane is required"
                ),
            );
        }
    }
}

fn check_thermal_generation_bounds(data: &ParsedData, ctx: &mut ValidationContext) {
    for thermal in &data.thermals {
        if thermal.min_generation_mw > thermal.max_generation_mw {
            let entity_str = format!("Thermal {}", thermal.id.0);
            ctx.add_error(
                ErrorKind::InvalidValue,
                "system/thermals.json",
                Some(&entity_str),
                format!(
                    "{entity_str}: min_generation_mw ({}) > max_generation_mw ({}); generation bounds are inconsistent",
                    thermal.min_generation_mw, thermal.max_generation_mw
                ),
            );
        }
    }
}

// ── validate_semantic_stages_penalties_scenarios ──────────────────────────────

/// Performs Layer 5b semantic validation: stage structure, penalty ordering,
/// and scenario model rules.
///
/// All 21 rules are checked regardless of failures in earlier rules — every
/// violation is collected before returning.  This function is infallible; it
/// never returns a `Result`.  Errors are pushed to `ctx` as
/// [`ErrorKind::InvalidValue`] or [`ErrorKind::BusinessRuleViolation`] entries;
/// penalty ordering warnings use [`ErrorKind::ModelQuality`].
///
/// # Arguments
///
/// * `data` — fully parsed case data produced by [`super::schema::validate_schema`].
/// * `ctx`  — mutable validation context that accumulates diagnostics.
///
/// # Conditional checks
///
/// Rules 12-13 are only checked when `data.inflow_seasonal_stats` is non-empty.
/// Rules 14-16 are only checked when `data.correlation` is `Some`.
/// Rules 17-18 are only checked when `data.load_factors` is non-empty.
pub(crate) fn validate_semantic_stages_penalties_scenarios(
    data: &ParsedData,
    ctx: &mut ValidationContext,
) {
    check_stage_structure(data, ctx);
    check_sobol_power_of_2(data, ctx);
    check_penalty_ordering(data, ctx);
    check_fpha_penalty_rule(data, ctx);
    check_scenario_models(data, ctx);
    check_correlation_matrices(data, ctx);
    check_correlation_same_type(data, ctx);
    check_external_scheme_has_files(data, ctx);
    check_load_factor_consistency(data, ctx);
    check_estimation_prerequisites(data, ctx);
    check_past_inflows_coverage(data, ctx);
    check_past_inflows_season_ids(data, ctx);
    check_season_id_consistency(data, ctx);
    check_observation_season_alignment(data, ctx);
}

// ── Tolerances ────────────────────────────────────────────────────────────────

/// Tolerance for floating-point probability sum comparisons.
const PROB_TOLERANCE: f64 = 1e-6;

/// Tolerance for floating-point correlation matrix comparisons.
const CORR_TOLERANCE: f64 = 1e-9;

// ── Rules 1-5: Stage structure ────────────────────────────────────────────────

/// Validates policy graph transitions, block durations, and `CVaR` parameters.
#[allow(clippy::too_many_lines)]
fn check_stage_structure(data: &ParsedData, ctx: &mut ValidationContext) {
    use cobre_core::temporal::{PolicyGraphType, StageRiskConfig};

    let graph = &data.stages.policy_graph;
    let stages = &data.stages.stages;

    // Build a set of all valid stage IDs for fast membership tests.
    let stage_ids: HashSet<i32> = stages.iter().map(|s| s.id).collect();

    // Rule 1: Every source_id and target_id in transitions must be a valid stage ID.
    for transition in &graph.transitions {
        if !stage_ids.contains(&transition.source_id) {
            ctx.add_error(
                ErrorKind::InvalidValue,
                "stages.json",
                None::<&str>,
                format!(
                    "transition source_id {} does not refer to a valid stage ID",
                    transition.source_id
                ),
            );
        }
        if !stage_ids.contains(&transition.target_id) {
            ctx.add_error(
                ErrorKind::InvalidValue,
                "stages.json",
                None::<&str>,
                format!(
                    "transition target_id {} does not refer to a valid stage ID",
                    transition.target_id
                ),
            );
        }
    }

    // Rule 2: For each unique source_id, outgoing probability sum must be ≈ 1.0.
    // Group transitions by source_id and sum probabilities.
    let mut prob_sums: HashMap<i32, f64> = HashMap::new();
    for transition in &graph.transitions {
        *prob_sums.entry(transition.source_id).or_insert(0.0) += transition.probability;
    }
    let mut sorted_sources: Vec<i32> = prob_sums.keys().copied().collect();
    sorted_sources.sort_unstable();
    for source_id in sorted_sources {
        let total = prob_sums[&source_id];
        if (total - 1.0).abs() > PROB_TOLERANCE {
            ctx.add_error(
                ErrorKind::InvalidValue,
                "stages.json",
                None::<&str>,
                format!(
                    "outgoing transition probabilities from stage {source_id} sum to {total:.8} \
                     (expected 1.0 ±{PROB_TOLERANCE}); probability must sum to 1.0"
                ),
            );
        }
    }

    // Rule 3: Cyclic graphs require annual_discount_rate > 0.0.
    if graph.graph_type == PolicyGraphType::Cyclic && graph.annual_discount_rate <= 0.0 {
        ctx.add_error(
            ErrorKind::InvalidValue,
            "stages.json",
            None::<&str>,
            format!(
                "cyclic policy graph requires annual_discount_rate > 0.0 for convergence, \
                 got {}",
                graph.annual_discount_rate
            ),
        );
    }

    // Rule 4: Every Block.duration_hours must be > 0.0.
    for stage in stages {
        for block in &stage.blocks {
            if block.duration_hours <= 0.0 {
                ctx.add_error(
                    ErrorKind::InvalidValue,
                    "stages.json",
                    Some(format!("Stage {}", stage.id)),
                    format!(
                        "Stage {}: block has duration_hours {} which is not > 0.0; \
                         block duration must be positive",
                        stage.id, block.duration_hours
                    ),
                );
            }
        }
    }

    // Rule 5: CVaR alpha must be in (0, 1] and lambda must be in [0, 1].
    for stage in stages {
        if let StageRiskConfig::CVaR { alpha, lambda } = stage.risk_config {
            if alpha <= 0.0 || alpha > 1.0 {
                ctx.add_error(
                    ErrorKind::InvalidValue,
                    "stages.json",
                    Some(format!("Stage {}", stage.id)),
                    format!(
                        "Stage {}: CVaR alpha ({alpha}) must be in (0, 1]; \
                         alpha must be a valid tail probability",
                        stage.id
                    ),
                );
            }
            if !(0.0..=1.0).contains(&lambda) {
                ctx.add_error(
                    ErrorKind::InvalidValue,
                    "stages.json",
                    Some(format!("Stage {}", stage.id)),
                    format!(
                        "Stage {}: CVaR lambda ({lambda}) must be in [0, 1]; \
                         lambda is the CVaR mixing weight",
                        stage.id
                    ),
                );
            }
        }
    }
}

// ── Rule 25: Sobol power-of-2 branching factor ───────────────────────────────

/// Warns when a stage uses `QmcSobol` with a non-power-of-2 `branching_factor`.
///
/// Sobol sequences achieve optimal low-discrepancy uniformity only when the
/// number of sample points is a power of 2. A non-power-of-2 value produces
/// valid noise but loses the stratification guarantee of the Gray-code
/// recurrence. This emits a `ModelQuality` warning (not an error) because the
/// configuration is valid but suboptimal.
fn check_sobol_power_of_2(data: &ParsedData, ctx: &mut ValidationContext) {
    use cobre_core::temporal::NoiseMethod;

    for stage in &data.stages.stages {
        if stage.id < 0 {
            continue; // skip pre-study stages
        }
        let bf = stage.scenario_config.branching_factor;
        if stage.scenario_config.noise_method == NoiseMethod::QmcSobol && !bf.is_power_of_two() {
            // bf == 0 is unreachable after parsing validation, but guard
            // defensively to prevent overflow in leading_zeros arithmetic.
            let suggestion = if bf > 0 {
                let lower = 1usize << (usize::BITS - bf.leading_zeros() - 1);
                let upper = lower << 1;
                format!("consider {lower} or {upper}")
            } else {
                "consider a positive power of 2".to_string()
            };
            ctx.add_warning(
                ErrorKind::ModelQuality,
                "stages.json",
                Some(format!("Stage {}", stage.id)),
                format!(
                    "Stage {}: qmc_sobol with num_scenarios={bf} which is not a \
                     power of 2; Sobol sequences have optimal uniformity at powers \
                     of 2 ({suggestion})",
                    stage.id,
                ),
            );
        }
    }
}

// ── Rules 6-10: Penalty ordering ──────────────────────────────────────────────

/// Checks the penalty hierarchy ordering across all hydros and buses.
///
/// Emits one `ModelQuality` warning per violated ordering check, aggregating
/// all violating entities into a single warning with the count and worst-case ID.
#[allow(clippy::too_many_lines)]
fn check_penalty_ordering(data: &ParsedData, ctx: &mut ValidationContext) {
    // Collect max deficit cost across all buses (combining all deficit segments).
    // When no deficit segments exist on any bus, the max is 0.0.
    let max_deficit_cost: f64 = data
        .buses
        .iter()
        .flat_map(|b| b.deficit_segments.iter().map(|s| s.cost_per_mwh))
        .fold(f64::NEG_INFINITY, f64::max)
        .max(0.0);

    // For each hydro, collect the relevant cost groups.
    // We aggregate violations per check across all hydros.

    // Check 6: filling_target_violation_cost > storage_violation_below_cost
    {
        let mut violations: Vec<(i32, f64, f64)> = Vec::new(); // (id, higher, lower)
        for hydro in &data.hydros {
            let higher = hydro.penalties.filling_target_violation_cost;
            let lower = hydro.penalties.storage_violation_below_cost;
            if higher <= lower {
                violations.push((hydro.id.0, higher, lower));
            }
        }
        // Worst case: the hydro with the largest (lower - higher) gap.
        if let Some(worst) = violations.iter().max_by(|a, b| {
            (b.2 - b.1)
                .partial_cmp(&(a.2 - a.1))
                .unwrap_or(std::cmp::Ordering::Equal)
        }) {
            let count = violations.len();
            ctx.add_warning(
                ErrorKind::ModelQuality,
                "penalties.json",
                None::<&str>,
                format!(
                    "Penalty ordering violation: filling_target_violation_cost ({}) should be > \
                     storage_violation_below_cost ({}) -- {count} hydro(s) affected, \
                     worst case: Hydro {}",
                    worst.1, worst.2, worst.0
                ),
            );
        }
    }

    // Check 7: storage_violation_below_cost > max(deficit_segment_costs)
    {
        let mut violations: Vec<(i32, f64)> = Vec::new(); // (id, storage_violation_cost)
        for hydro in &data.hydros {
            let higher = hydro.penalties.storage_violation_below_cost;
            if higher <= max_deficit_cost {
                violations.push((hydro.id.0, higher));
            }
        }
        if let Some(worst) = violations
            .iter()
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        {
            let count = violations.len();
            ctx.add_warning(
                ErrorKind::ModelQuality,
                "penalties.json",
                None::<&str>,
                format!(
                    "Penalty ordering violation: storage_violation_below_cost ({}) should be > \
                     max(deficit_segment_costs) ({max_deficit_cost}) -- {count} hydro(s) affected, \
                     worst case: Hydro {}",
                    worst.1, worst.0
                ),
            );
        }
    }

    // Check 8: max(deficit_segment_costs) > max(constraint_violation_costs)
    // Constraint violation costs: turbined_violation_below_cost, outflow_violation_below_cost,
    // outflow_violation_above_cost, generation_violation_below_cost, evaporation_violation_cost,
    // water_withdrawal_violation_cost.
    {
        // Helper: compute max constraint violation cost for a hydro.
        let max_cv = |h: &cobre_core::entities::Hydro| {
            let p = &h.penalties;
            p.turbined_violation_below_cost
                .max(p.outflow_violation_below_cost)
                .max(p.outflow_violation_above_cost)
                .max(p.generation_violation_below_cost)
                .max(p.evaporation_violation_cost)
                .max(p.water_withdrawal_violation_cost)
        };

        let max_constraint_cost: f64 = data
            .hydros
            .iter()
            .map(max_cv)
            .fold(f64::NEG_INFINITY, f64::max)
            .max(0.0);

        if !data.hydros.is_empty() && max_deficit_cost <= max_constraint_cost {
            // Find the hydro with the highest constraint_violation_cost.
            if let Some(worst_hydro) = data.hydros.iter().max_by(|a, b| {
                max_cv(a)
                    .partial_cmp(&max_cv(b))
                    .unwrap_or(std::cmp::Ordering::Equal)
            }) {
                ctx.add_warning(
                    ErrorKind::ModelQuality,
                    "penalties.json",
                    None::<&str>,
                    format!(
                        "Penalty ordering violation: max(deficit_segment_costs) \
                         ({max_deficit_cost}) should be > max(constraint_violation_costs) \
                         ({max_constraint_cost}) -- 1 hydro(s) affected, worst case: Hydro {}",
                        worst_hydro.id.0
                    ),
                );
            }
        }
    }

    // Check 9: min(constraint_violation_costs) > max(resource_costs)
    // Resource costs: spillage_cost, diversion_cost.
    {
        if !data.hydros.is_empty() {
            // Helper: compute min constraint violation cost for a hydro.
            let min_cv = |h: &cobre_core::entities::Hydro| {
                let p = &h.penalties;
                p.turbined_violation_below_cost
                    .min(p.outflow_violation_below_cost)
                    .min(p.outflow_violation_above_cost)
                    .min(p.generation_violation_below_cost)
                    .min(p.evaporation_violation_cost)
                    .min(p.water_withdrawal_violation_cost)
            };

            let min_constraint_cost: f64 =
                data.hydros.iter().map(min_cv).fold(f64::INFINITY, f64::min);

            let max_resource_cost: f64 = data
                .hydros
                .iter()
                .map(|h| h.penalties.spillage_cost.max(h.penalties.diversion_cost))
                .fold(f64::NEG_INFINITY, f64::max)
                .max(0.0);

            if min_constraint_cost <= max_resource_cost {
                // Find the hydro with the lowest constraint cost (the worst offender).
                if let Some(worst_hydro) = data.hydros.iter().min_by(|a, b| {
                    min_cv(a)
                        .partial_cmp(&min_cv(b))
                        .unwrap_or(std::cmp::Ordering::Equal)
                }) {
                    ctx.add_warning(
                        ErrorKind::ModelQuality,
                        "penalties.json",
                        None::<&str>,
                        format!(
                            "Penalty ordering violation: min(constraint_violation_costs) \
                             ({min_constraint_cost}) should be > max(resource_costs) \
                             ({max_resource_cost}) -- 1 hydro(s) affected, worst case: Hydro {}",
                            worst_hydro.id.0
                        ),
                    );
                }
            }
        }
    }

    // Check 10: min(resource_costs) > 0 (regularization costs must be positive)
    {
        let mut violations: Vec<(i32, f64)> = Vec::new(); // (id, min_resource_cost)
        for hydro in &data.hydros {
            let min_resource = hydro
                .penalties
                .spillage_cost
                .min(hydro.penalties.diversion_cost);
            if min_resource <= 0.0 {
                violations.push((hydro.id.0, min_resource));
            }
        }
        if let Some(worst) = violations
            .iter()
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        {
            let count = violations.len();
            ctx.add_warning(
                ErrorKind::ModelQuality,
                "penalties.json",
                None::<&str>,
                format!(
                    "Penalty ordering violation: min(resource_costs) ({}) should be > 0 \
                     (regularization costs must be positive to prevent LP degeneracy) -- \
                     {count} hydro(s) affected, worst case: Hydro {}",
                    worst.1, worst.0
                ),
            );
        }
    }
}

// ── Rule 11: FPHA penalty rule ─────────────────────────────────────────────────

/// Checks that FPHA hydros have `fpha_turbined_cost >= 0`.
///
/// A zero cost is valid for constant-head plants (e.g., `gamma_v = 0`) where the
/// LP has no incentive to spill rather than turbine. Negative values are rejected
/// because they would make turbining artificially profitable and distort dispatch.
fn check_fpha_penalty_rule(data: &ParsedData, ctx: &mut ValidationContext) {
    use cobre_core::entities::HydroGenerationModel;
    for hydro in &data.hydros {
        if hydro.generation_model == HydroGenerationModel::Fpha {
            let fpha_cost = hydro.penalties.fpha_turbined_cost;
            if fpha_cost < 0.0 {
                let entity_str = format!("Hydro {}", hydro.id.0);
                ctx.add_error(
                    ErrorKind::BusinessRuleViolation,
                    "penalties.json",
                    Some(&entity_str),
                    format!(
                        "{entity_str}: fpha_turbined_cost ({fpha_cost}) must be non-negative (>= 0) \
                         for FPHA hydros; negative values distort LP dispatch"
                    ),
                );
            }
        }
    }
}

// ── Rules 12-13: Scenario model rules ─────────────────────────────────────────

/// Validates inflow model standard deviation and AR coefficient count consistency.
fn check_scenario_models(data: &ParsedData, ctx: &mut ValidationContext) {
    // Rule 12: std_m3s >= 0.0; warn when == 0.0 (deterministic inflow).
    // Note: std_m3s < 0 is already caught by the schema parser. However, the
    // schema parser only produces a SchemaError; here we emit a ModelQuality
    // warning for std_m3s == 0.0 (valid but unusual deterministic inflow).
    for row in &data.inflow_seasonal_stats {
        if row.std_m3s == 0.0 {
            ctx.add_warning(
                ErrorKind::ModelQuality,
                "scenarios/inflow_seasonal_stats.parquet",
                Some(format!("Hydro {}", row.hydro_id.0)),
                format!(
                    "Hydro {} stage {}: std_m3s is 0.0, indicating deterministic inflow \
                     (no stochastic component); verify this is intentional",
                    row.hydro_id.0, row.stage_id
                ),
            );
        }
    }

    // Rule 13: residual_std_ratio consistency across lag rows (V-AR-4).
    // For each (hydro_id, stage_id) group, all lag rows must share the same
    // residual_std_ratio value. Range validation is already done by the parser
    // range validation is done by the parser; this rule only checks cross-row consistency within a group.
    {
        let mut ratio_by_group: HashMap<(i32, i32), f64> = HashMap::new();
        for row in &data.inflow_ar_coefficients {
            let key = (row.hydro_id.0, row.stage_id);
            match ratio_by_group.entry(key) {
                std::collections::hash_map::Entry::Vacant(e) => {
                    e.insert(row.residual_std_ratio);
                }
                std::collections::hash_map::Entry::Occupied(e) => {
                    if (*e.get() - row.residual_std_ratio).abs() > f64::EPSILON {
                        ctx.add_error(
                            ErrorKind::InvalidValue,
                            "scenarios/inflow_ar_coefficients.parquet",
                            Some(format!("Hydro {}", row.hydro_id.0)),
                            format!(
                                "Hydro {} stage {}: inconsistent residual_std_ratio across \
                                 lag rows (first={}, current={}); all lags must share the \
                                 same ratio",
                                row.hydro_id.0,
                                row.stage_id,
                                e.get(),
                                row.residual_std_ratio,
                            ),
                        );
                    }
                }
            }
        }
    }
}

// ── Rules 14-16: Correlation matrix validation ────────────────────────────────

/// Validates correlation matrix symmetry, diagonal, and off-diagonal range for
/// all groups in all profiles of the correlation model.
///
/// Only runs when `data.correlation` is `Some`.
fn check_correlation_matrices(data: &ParsedData, ctx: &mut ValidationContext) {
    let Some(correlation) = &data.correlation else {
        return;
    };

    for profile in correlation.profiles.values() {
        for group in &profile.groups {
            let n = group.entities.len();
            let group_name = &group.name;

            // Rules 14-16 require a square matrix; the matrix row count is guaranteed
            // to match entity count by Layer 4 (dimensional check 4). Be defensive.
            if group.matrix.len() != n {
                continue;
            }

            for i in 0..n {
                if group.matrix[i].len() != n {
                    continue;
                }
                for j in 0..n {
                    let val = group.matrix[i][j];

                    // Rule 15: Diagonal entries must be 1.0 (±CORR_TOLERANCE).
                    if i == j && (val - 1.0).abs() > CORR_TOLERANCE {
                        ctx.add_error(
                            ErrorKind::BusinessRuleViolation,
                            "scenarios/correlation.json",
                            Some(format!("CorrelationGroup {group_name}")),
                            format!(
                                "CorrelationGroup '{group_name}': diagonal entry matrix[{i}][{i}] \
                                 is {val}, expected 1.0 (±{CORR_TOLERANCE}); \
                                 correlation matrix diagonal must be 1.0"
                            ),
                        );
                    }

                    // Rule 16: Off-diagonal entries must be in [-1.0, 1.0].
                    if i != j && !((-1.0_f64)..=1.0).contains(&val) {
                        ctx.add_error(
                            ErrorKind::BusinessRuleViolation,
                            "scenarios/correlation.json",
                            Some(format!("CorrelationGroup {group_name}")),
                            format!(
                                "CorrelationGroup '{group_name}': off-diagonal entry \
                                 matrix[{i}][{j}] is {val}, outside valid range [-1.0, 1.0]; \
                                 correlation coefficients must be in [-1.0, 1.0]"
                            ),
                        );
                    }

                    // Rule 14: Symmetry check (only check upper triangle to avoid duplicates).
                    if i < j {
                        let symmetric = group.matrix[j][i];
                        if (val - symmetric).abs() > CORR_TOLERANCE {
                            ctx.add_error(
                                ErrorKind::BusinessRuleViolation,
                                "scenarios/correlation.json",
                                Some(format!("CorrelationGroup {group_name}")),
                                format!(
                                    "CorrelationGroup '{group_name}': correlation matrix is not \
                                     symmetric at ({i},{j}): matrix[{i}][{j}]={val} but \
                                     matrix[{j}][{i}]={symmetric}; tolerance is {CORR_TOLERANCE}"
                                ),
                            );
                        }
                    }
                }
            }
        }
    }
}

// ── M4: Same-type enforcement within correlation groups ──────────────────────

/// Validates that all entities within each correlation group share the same
/// `entity_type` value. Mixed groups produce incorrect covariance matrices.
fn check_correlation_same_type(data: &ParsedData, ctx: &mut ValidationContext) {
    let Some(correlation) = &data.correlation else {
        return;
    };

    for profile in correlation.profiles.values() {
        for group in &profile.groups {
            if group.entities.is_empty() {
                continue;
            }
            let first_type = &group.entities[0].entity_type;
            for entity in &group.entities[1..] {
                if entity.entity_type != *first_type {
                    ctx.add_error(
                        ErrorKind::BusinessRuleViolation,
                        "scenarios/correlation.json",
                        Some(format!("CorrelationGroup '{}'", group.name)),
                        format!(
                            "CorrelationGroup '{}': entity {} has type '{}' but entity {} has \
                             type '{}'; all entities in a group must share the same entity_type",
                            group.name,
                            group.entities[0].id.0,
                            first_type,
                            entity.id.0,
                            entity.entity_type,
                        ),
                    );
                    break;
                }
            }
        }
    }
}

// ── F2-002: External scheme requires external scenario files ─────────────────

/// Validates that when a class uses the `External` sampling scheme, the
/// corresponding external scenario file data is non-empty.
fn check_external_scheme_has_files(data: &ParsedData, ctx: &mut ValidationContext) {
    use cobre_core::scenario::SamplingScheme;
    use std::path::Path;

    // scenario_source is read from config.json (training and simulation sections).
    // Config has already been validated by Layer 2, so these calls will not fail.
    let Ok(training_source) = data
        .config
        .training_scenario_source(Path::new("config.json"))
    else {
        return;
    };
    let Ok(simulation_source) = data
        .config
        .simulation_scenario_source(Path::new("config.json"))
    else {
        return;
    };

    // Only check simulation independently when it explicitly defines its own
    // scenario_source; otherwise simulation falls back to training, which is
    // already checked, and checking again would produce duplicate errors.
    let sources: &[(&str, &_)] = if data.config.simulation.scenario_source.is_some() {
        &[
            ("training", &training_source),
            ("simulation", &simulation_source),
        ]
    } else {
        &[("training", &training_source)]
    };

    let mut check_external =
        |section: &str, scheme: SamplingScheme, class_name: &str, is_empty: bool| {
            if scheme == SamplingScheme::External && is_empty {
                ctx.add_error(
                    ErrorKind::BusinessRuleViolation,
                    "config.json",
                    Some(format!("{section}.scenario_source.{class_name}")),
                    format!(
                        "{class_name} class uses 'external' scheme but no \
                     external_{class_name}_scenarios.parquet data was found; \
                     external scheme requires corresponding scenario file"
                    ),
                );
            }
        };

    for (section, source) in sources {
        check_external(
            section,
            source.inflow_scheme,
            "inflow",
            data.external_scenarios.is_empty(),
        );
        check_external(
            section,
            source.load_scheme,
            "load",
            data.external_load_scenarios.is_empty(),
        );
        check_external(
            section,
            source.ncs_scheme,
            "ncs",
            data.external_ncs_scenarios.is_empty(),
        );
    }
}

// ── Rules 17-18: Load factor consistency ─────────────────────────────────────

/// Validates cross-file consistency between `load_factors.json` and
/// `load_seasonal_stats.parquet`.
///
/// Rule 17: For every `LoadFactorEntry`, each `block_factors[j].block_id` must
/// match a `Block.index` in the corresponding stage's `blocks` array.
///
/// Rule 18: A `LoadFactorEntry` for a `(bus_id, stage_id)` pair where
/// `load_seasonal_stats` has `std_mw == 0.0` (deterministic load) produces a
/// `ModelQuality` warning because block factors have no effect on deterministic
/// loads.
///
/// Silently skips when `data.load_factors` is empty.
fn check_load_factor_consistency(data: &ParsedData, ctx: &mut ValidationContext) {
    if data.load_factors.is_empty() {
        return;
    }

    // Build a map from stage_id to the set of valid block indices for that stage.
    let stage_block_indices: HashMap<i32, HashSet<usize>> = data
        .stages
        .stages
        .iter()
        .filter(|s| s.id >= 0)
        .map(|s| {
            let indices: HashSet<usize> = s.blocks.iter().map(|b| b.index).collect();
            (s.id, indices)
        })
        .collect();

    // Build a map from (bus_id, stage_id) to std_mw for deterministic-load detection.
    let load_std: HashMap<(i32, i32), f64> = data
        .load_seasonal_stats
        .iter()
        .map(|row| ((row.bus_id.0, row.stage_id), row.std_mw))
        .collect();

    for (i, entry) in data.load_factors.iter().enumerate() {
        // Rule 17: each block_id must match a Block.index in the entry's stage.
        if let Some(valid_indices) = stage_block_indices.get(&entry.stage_id) {
            for bf in &entry.block_factors {
                let block_idx = usize::try_from(bf.block_id).unwrap_or(usize::MAX);
                if !valid_indices.contains(&block_idx) {
                    let sorted: Vec<usize> = {
                        let mut v: Vec<usize> = valid_indices.iter().copied().collect();
                        v.sort_unstable();
                        v
                    };
                    ctx.add_error(
                        ErrorKind::BusinessRuleViolation,
                        "scenarios/load_factors.json",
                        Some(format!("LoadFactorEntry[{i}]")),
                        format!(
                            "LoadFactorEntry[{i}] has block_id {} which is not in the block set \
                             {sorted:?} for stage {}",
                            bf.block_id, entry.stage_id
                        ),
                    );
                }
            }
        }

        // Rule 18: warn when the (bus_id, stage_id) pair has std_mw == 0.0.
        let key = (entry.bus_id.0, entry.stage_id);
        if let Some(&std_mw) = load_std.get(&key) {
            if std_mw == 0.0 {
                ctx.add_warning(
                    ErrorKind::ModelQuality,
                    "scenarios/load_factors.json",
                    Some(format!("LoadFactorEntry[{i}]")),
                    format!(
                        "LoadFactorEntry[{i}] (bus {}, stage {}) references a deterministic load \
                         (std_mw == 0.0); block factors have no effect on deterministic loads",
                        entry.bus_id.0, entry.stage_id
                    ),
                );
            }
        }
    }
}

// ── Rules 19-21: Estimation prerequisites ─────────────────────────────────────

/// Validates prerequisites for the history-based PAR(p) estimation path.
///
/// Runs only when `inflow_history.parquet` is present and
/// `inflow_seasonal_stats.parquet` is absent — i.e., when the estimation path
/// will be triggered (same condition used by the estimation pipeline).
///
/// Rule 19: `season_definitions` must be present in `stages.json` so that
/// observations can be grouped by season.
///
/// Rule 20: Each `(hydro_id, season_id)` group must have at least
/// `config.estimation.min_observations_per_season` observations.
///
/// Rule 21: Every hydro in the system must have at least one observation in
/// `inflow_history.parquet`; missing hydros cannot be estimated.
fn check_estimation_prerequisites(data: &ParsedData, ctx: &mut ValidationContext) {
    // Detect the estimation path: history present AND (stats absent OR AR coefficients absent).
    // Estimation runs whenever the system needs to derive AR coefficients from history.
    // The runtime in estimation.rs skips only when BOTH stats AND coefficients are present.
    let has_history = !data.inflow_history.is_empty();
    let has_stats = !data.inflow_seasonal_stats.is_empty();
    let has_ar_coefficients = !data.inflow_ar_coefficients.is_empty();
    let estimation_active = has_history && !(has_stats && has_ar_coefficients);

    if !estimation_active {
        return;
    }

    // Rule 19: season_definitions (season_map) must be present.
    if data.stages.policy_graph.season_map.is_none() {
        ctx.add_error(
            ErrorKind::BusinessRuleViolation,
            "scenarios/inflow_history.parquet",
            None::<&str>,
            "season_definitions is required in stages.json when estimating from \
             inflow_history.parquet; add a season_definitions section to stages.json",
        );
    }

    // Rule 21: every hydro must have at least one observation.
    let hydro_ids_in_history: HashSet<i32> =
        data.inflow_history.iter().map(|r| r.hydro_id.0).collect();
    let mut missing_hydros: Vec<i32> = data
        .hydros
        .iter()
        .filter(|h| !hydro_ids_in_history.contains(&h.id.0))
        .map(|h| h.id.0)
        .collect();
    missing_hydros.sort_unstable();
    for id in missing_hydros {
        ctx.add_error(
            ErrorKind::BusinessRuleViolation,
            "scenarios/inflow_history.parquet",
            Some(format!("Hydro {id}")),
            format!(
                "hydro {id} has no observations in inflow_history.parquet but estimation \
                 is required; add historical inflow data for this hydro"
            ),
        );
    }

    // Rule 20: warn for (hydro, season) groups with fewer than the minimum
    // observations.  Only possible when season_map is Some; if it is None,
    // Rule 19 already emitted an error — skip to avoid a confusing cascade.
    if let Some(_season_map) = &data.stages.policy_graph.season_map {
        let min_obs = data.config.estimation.min_observations_per_season as usize;

        // Build a stage index: (start_date, end_date, season_id) in canonical order.
        // Stages are already sorted by id (canonical order), which matches date order.
        let stage_index: Vec<(chrono::NaiveDate, chrono::NaiveDate, usize)> = data
            .stages
            .stages
            .iter()
            .filter_map(|s| s.season_id.map(|sid| (s.start_date, s.end_date, sid)))
            .collect();

        // Count observations per (hydro_id, season_id).
        let mut counts: HashMap<(i32, usize), usize> = HashMap::new();
        for row in &data.inflow_history {
            // Find the season for this observation's date.
            let pos = stage_index.partition_point(|(start, _, _)| *start <= row.date);
            let season_id = if pos > 0 {
                let (_, end_date, sid) = stage_index[pos - 1];
                if row.date < end_date { Some(sid) } else { None }
            } else {
                None
            };

            if let Some(sid) = season_id {
                *counts.entry((row.hydro_id.0, sid)).or_insert(0) += 1;
            }
        }

        // Emit a warning for each (hydro, season) below the minimum.
        let mut violations: Vec<(i32, usize, usize)> = counts
            .iter()
            .filter(|&(_, n)| *n < min_obs)
            .map(|(&(hid, sid), &n)| (hid, sid, n))
            .collect();
        // Sort for deterministic output order.
        violations.sort_unstable_by_key(|&(hid, sid, _)| (hid, sid));
        for (hid, sid, n) in violations {
            ctx.add_warning(
                ErrorKind::ModelQuality,
                "scenarios/inflow_history.parquet",
                Some(format!("Hydro {hid}")),
                format!(
                    "hydro {hid} season {sid} has {n} observations \
                     (minimum recommended: {min_obs}); estimation accuracy may be \
                     insufficient with so few observations"
                ),
            );
        }
    }
}

// ── Rules 22-24: Past inflows coverage ────────────────────────────────────────

/// Validates that `initial_conditions.json` provides sufficient `past_inflows`
/// entries for lag initialization when `inflow_lags: true` and PAR order > 0.
///
/// Runs only when at least one study stage has `state_config.inflow_lags: true`
/// AND `inflow_ar_coefficients` is non-empty with maximum PAR order > 0.
///
/// Rule 22: `past_inflows` must be non-empty when lag initialization is needed.
///
/// Rule 23: For each hydro with per-hydro PAR order `p` (max lag across all its
/// `(hydro_id, stage_id)` groups), `past_inflows` must contain an entry for that
/// hydro with `values_m3s.len() >= p`.
///
/// Rule 24: Every hydro ID present in `past_inflows` must exist in the hydro
/// registry.
fn check_past_inflows_coverage(data: &ParsedData, ctx: &mut ValidationContext) {
    // Precondition: at least one study stage (id >= 0) has inflow_lags: true.
    let lags_enabled = data
        .stages
        .stages
        .iter()
        .filter(|s| s.id >= 0)
        .any(|s| s.state_config.inflow_lags);
    if !lags_enabled {
        return;
    }

    // Precondition: AR coefficients present and max lag order > 0.
    // The per-hydro PAR order is the maximum `lag` value across all rows for
    // that hydro (lags are 1-based, so max lag == PAR order p).
    let max_order_overall: i32 = data
        .inflow_ar_coefficients
        .iter()
        .map(|c| c.lag)
        .max()
        .unwrap_or(0);
    if max_order_overall == 0 {
        return;
    }

    let past_inflows = &data.initial_conditions.past_inflows;

    // Rule 22: past_inflows must be non-empty.
    if past_inflows.is_empty() {
        ctx.add_error(
            ErrorKind::BusinessRuleViolation,
            "initial_conditions.json",
            None::<&str>,
            "inflow_lags is enabled with PAR order > 0 but              initial_conditions.json has no past_inflows entries;              lag initialization requires past inflow values",
        );
        return; // rules 23-24 require non-empty past_inflows
    }

    // Build per-hydro maximum PAR order from inflow_ar_coefficients.
    // Key: hydro_id; value: max lag seen for that hydro across all stages.
    let mut max_order_per_hydro: HashMap<i32, i32> = HashMap::new();
    for row in &data.inflow_ar_coefficients {
        let entry = max_order_per_hydro.entry(row.hydro_id.0).or_insert(0);
        if row.lag > *entry {
            *entry = row.lag;
        }
    }

    // Build a lookup from hydro_id -> number of past_inflows values provided.
    let past_inflows_len: HashMap<i32, usize> = past_inflows
        .iter()
        .map(|pi| (pi.hydro_id.0, pi.values_m3s.len()))
        .collect();

    // Rule 23: for each hydro with PAR order p, verify that past_inflows
    // contains an entry for that hydro with at least p values.
    {
        let mut coverage_violations: Vec<(i32, i32, usize)> = Vec::new(); // (hydro_id, order, provided)
        for (&hydro_id, &order) in &max_order_per_hydro {
            if order == 0 {
                continue;
            }
            let required = usize::try_from(order).unwrap_or(usize::MAX);
            let provided = past_inflows_len.get(&hydro_id).copied().unwrap_or(0);
            if provided < required {
                coverage_violations.push((hydro_id, order, provided));
            }
        }

        // Sort for deterministic output order.
        coverage_violations.sort_unstable_by_key(|&(hid, _, _)| hid);
        for (hydro_id, order, provided) in coverage_violations {
            let entity_str = format!("Hydro {hydro_id}");
            ctx.add_error(
                ErrorKind::BusinessRuleViolation,
                "initial_conditions.json",
                Some(&entity_str),
                format!(
                    "Hydro {hydro_id}: insufficient past_inflows for lag initialization; \
                     PAR order is {order} but initial_conditions.json provides only \
                     {provided} value(s) in past_inflows (need at least {order})"
                ),
            );
        }
    }

    // Rule 24: every hydro ID in past_inflows must exist in the hydro registry.
    {
        let hydro_registry: HashSet<i32> = data.hydros.iter().map(|h| h.id.0).collect();
        let past_inflow_ids: HashSet<i32> = past_inflows.iter().map(|pi| pi.hydro_id.0).collect();
        let mut unknown_ids: Vec<i32> = past_inflow_ids
            .difference(&hydro_registry)
            .copied()
            .collect();
        unknown_ids.sort_unstable();
        for id in unknown_ids {
            let entity_str = format!("Hydro {id}");
            ctx.add_error(
                ErrorKind::BusinessRuleViolation,
                "initial_conditions.json",
                Some(&entity_str),
                format!(
                    "Hydro {id} appears in past_inflows but does not exist \
                     in the hydro registry (system/hydros.json); \
                     remove the unknown hydro or add it to the registry"
                ),
            );
        }
    }
}

// ── Rule 32: past_inflows season_ids against SeasonMap ───────────────────────

/// Rule 32: when `past_inflows[i].season_ids` is `Some` and the hydro has
/// PAR order > 0, each `season_id` value must exist in the `SeasonMap`.
///
/// Skips the check when `season_map` is `None` — the semantic layer cannot
/// validate season IDs without a `SeasonMap`. Schema-layer length validation
/// (matching `season_ids.len() == values_m3s.len()`) is handled in
/// `cobre-io/src/initial_conditions.rs`.
fn check_past_inflows_season_ids(data: &ParsedData, ctx: &mut ValidationContext) {
    let Some(season_map) = &data.stages.policy_graph.season_map else {
        return;
    };

    // Build per-hydro maximum PAR order from inflow_ar_coefficients.
    let mut max_order_per_hydro: HashMap<i32, i32> = HashMap::new();
    for row in &data.inflow_ar_coefficients {
        let entry = max_order_per_hydro.entry(row.hydro_id.0).or_insert(0);
        if row.lag > *entry {
            *entry = row.lag;
        }
    }

    let valid_ids: HashSet<usize> = season_map.seasons.iter().map(|s| s.id).collect();
    let mut sorted_valid_ids: Vec<usize> = valid_ids.iter().copied().collect();
    sorted_valid_ids.sort_unstable();

    for pi in &data.initial_conditions.past_inflows {
        let par_order = max_order_per_hydro
            .get(&pi.hydro_id.0)
            .copied()
            .unwrap_or(0);
        if par_order == 0 {
            continue;
        }

        let Some(season_ids) = &pi.season_ids else {
            continue;
        };

        for &sid in season_ids {
            let sid_usize = sid as usize;
            if !valid_ids.contains(&sid_usize) {
                let entity_str = format!("Hydro {}", pi.hydro_id.0);
                ctx.add_error(
                    ErrorKind::BusinessRuleViolation,
                    "initial_conditions.json",
                    Some(&entity_str),
                    format!(
                        "Hydro {}: past_inflows.season_ids contains season_id {} which is \
                         not defined in season_definitions; valid season IDs are {:?}",
                        pi.hydro_id.0, sid, sorted_valid_ids,
                    ),
                );
            }
        }
    }
}

// ── Rules 27+29: Season ID range coverage and resolution consistency ──────────

/// Validates that every stage `season_id` references a season defined in
/// `season_definitions` (Rule 27), and that all stages sharing a `season_id`
/// have compatible temporal durations (Rule 29).
///
/// Skips the check entirely when `season_map` is `None` — Rule 19 already
/// handles the missing `season_definitions` case.
///
/// Rule 27: Each stage with a `season_id` must reference a season ID that
/// exists in `season_definitions.seasons[].id`.
///
/// Rule 29: All stages in the same `season_id` group must have durations
/// within 7 days of each other.  A spread greater than 7 days indicates
/// mixed temporal resolutions (e.g., monthly 30d alongside quarterly 91d)
/// which leads to conflicting PAR model parameterisations.
fn check_season_id_consistency(data: &ParsedData, ctx: &mut ValidationContext) {
    let Some(season_map) = &data.stages.policy_graph.season_map else {
        return;
    };

    let valid_ids: HashSet<usize> = season_map.seasons.iter().map(|s| s.id).collect();
    let mut sorted_valid_ids: Vec<usize> = valid_ids.iter().copied().collect();
    sorted_valid_ids.sort_unstable();

    for stage in &data.stages.stages {
        let Some(sid) = stage.season_id else {
            continue;
        };
        if !valid_ids.contains(&sid) {
            ctx.add_error(
                ErrorKind::BusinessRuleViolation,
                "stages.json",
                Some(format!("Stage {}", stage.id)),
                format!(
                    "stage {} has season_id {} which is not defined in \
                     season_definitions; valid season IDs are {:?}",
                    stage.id, sid, sorted_valid_ids,
                ),
            );
        }
    }

    let mut season_groups: HashMap<usize, Vec<(i32, i64)>> = HashMap::new();
    for stage in &data.stages.stages {
        if let Some(sid) = stage.season_id {
            let duration_days = (stage.end_date - stage.start_date).num_days();
            season_groups
                .entry(sid)
                .or_default()
                .push((stage.id, duration_days));
        }
    }

    let mut sorted_season_ids: Vec<usize> = season_groups.keys().copied().collect();
    sorted_season_ids.sort_unstable();

    for sid in sorted_season_ids {
        let members = &season_groups[&sid];
        if members.len() < 2 {
            continue;
        }
        debug_assert!(!members.is_empty(), "guarded by len() >= 2 above");
        let min_d = members.iter().map(|&(_, d)| d).min().unwrap_or(0);
        let max_d = members.iter().map(|&(_, d)| d).max().unwrap_or(0);
        if max_d - min_d > 7 {
            let mut details_parts: Vec<String> = members
                .iter()
                .map(|&(id, d)| format!("stage {id} ({d}d)"))
                .collect();
            details_parts.sort_unstable();
            let details = details_parts.join(", ");
            ctx.add_error(
                ErrorKind::BusinessRuleViolation,
                "stages.json",
                Some(format!("Season {sid}")),
                format!(
                    "stages sharing season_id {sid} have incompatible durations: {details}; \
                     stages within the same season must have the same temporal resolution \
                     (e.g., all monthly or all weekly)",
                ),
            );
        }
    }

    check_season_observation_coverage(data, season_map, ctx);
    check_season_contiguity(data, season_map, ctx);
}

// ── Rule 31: Observation-to-season alignment ──────────────────────────────────

/// Rule 31: Observation-to-season alignment check.
///
/// Detects two cases:
///
/// **Finer-than-season (warning)**: If any `(hydro_id, season_id, year)` triple
/// has more than one observation, the observation data has finer temporal
/// resolution than the season definitions (e.g., monthly observations with
/// quarterly seasons). The PAR estimation pipeline will automatically aggregate
/// these observations using duration-weighted averaging. A warning is emitted.
///
/// **Coarser-than-season (error)**: If a hydro has at least one observation in
/// a given year but has no observation for some `(season_id, year)` group that
/// the season map covers, the observation data is coarser than the season
/// resolution (e.g., quarterly observations with monthly seasons). Aggregation
/// cannot disaggregate observations, so this is an unrecoverable data error.
///
/// Only runs when estimation is active (history present, not both stats and AR
/// coefficients pre-computed) and `season_map` is `Some`.
fn check_observation_season_alignment(data: &ParsedData, ctx: &mut ValidationContext) {
    use chrono::Datelike;

    let has_history = !data.inflow_history.is_empty();
    let has_stats = !data.inflow_seasonal_stats.is_empty();
    let has_ar_coefficients = !data.inflow_ar_coefficients.is_empty();
    let estimation_active = has_history && !(has_stats && has_ar_coefficients);

    if !estimation_active {
        return;
    }

    let Some(season_map) = &data.stages.policy_graph.season_map else {
        return;
    };

    // Stages are sorted by id (canonical order), which matches date order.
    let stage_index: Vec<(chrono::NaiveDate, chrono::NaiveDate, usize)> = data
        .stages
        .stages
        .iter()
        .filter_map(|s| s.season_id.map(|sid| (s.start_date, s.end_date, sid)))
        .collect();

    // Build counts: (hydro_id, season_id, year) -> observation count.
    let mut counts: HashMap<(i32, usize, i32), usize> = HashMap::new();
    for row in &data.inflow_history {
        let pos = stage_index.partition_point(|(start, _, _)| *start <= row.date);
        let season_id = if pos > 0 {
            let (_, end_date, sid) = stage_index[pos - 1];
            if row.date < end_date { Some(sid) } else { None }
        } else {
            None
        }
        .or_else(|| season_map.season_for_date(row.date));

        if let Some(sid) = season_id {
            let year = row.date.year();
            *counts.entry((row.hydro_id.0, sid, year)).or_insert(0) += 1;
        }
    }

    // ── Finer-than-season: warn when count > 1 (observations will be aggregated) ──
    let mut finer_violations: Vec<(i32, usize, i32, usize)> = counts
        .iter()
        .filter(|&(_, &n)| n > 1)
        .map(|(&(hid, sid, yr), &n)| (hid, sid, yr, n))
        .collect();
    finer_violations.sort_unstable();
    for (hid, sid, yr, count) in finer_violations {
        ctx.add_warning(
            ErrorKind::BusinessRuleViolation,
            "scenarios/inflow_history.parquet",
            Some(format!("Hydro {hid}")),
            format!(
                "hydro {hid} has {count} observations for season {sid} year {yr} \
                 in inflow_history.parquet; these will be aggregated to season \
                 resolution during PAR estimation",
            ),
        );
    }

    // ── Coarser-than-season: error when a hydro has observations in a year but
    // is missing an entire (season_id, year) group that the season map defines ──
    //
    // Collect the distinct season IDs defined in the season map.
    let defined_season_ids: Vec<usize> = season_map.seasons.iter().map(|s| s.id).collect();

    // For each hydro, collect the set of years where it has at least one observation.
    let mut hydro_years: HashMap<i32, HashSet<i32>> = HashMap::new();
    for &(hid, _sid, yr) in counts.keys() {
        hydro_years.entry(hid).or_default().insert(yr);
    }

    // Detect missing (hydro, season, year) groups: the hydro has history in
    // that year, but the counts map has no entry for that (hydro, season, year).
    let mut coarser_violations: Vec<(i32, usize, i32)> = Vec::new();
    for (&hid, years) in &hydro_years {
        for &yr in years {
            for &sid in &defined_season_ids {
                if !counts.contains_key(&(hid, sid, yr)) {
                    coarser_violations.push((hid, sid, yr));
                }
            }
        }
    }
    coarser_violations.sort_unstable();
    for (hid, sid, yr) in coarser_violations {
        ctx.add_error(
            ErrorKind::BusinessRuleViolation,
            "scenarios/inflow_history.parquet",
            Some(format!("Hydro {hid}")),
            format!(
                "hydro {hid} has no observations for season {sid} year {yr} \
                 in inflow_history.parquet, suggesting coarser-than-season \
                 observation resolution; coarser-than-season observations \
                 cannot be disaggregated and are not supported",
            ),
        );
    }
}

/// V4.2 — Observation coverage (Rule 28).
///
/// Warns when a season has zero inflow observations across all hydros and
/// the training inflow scheme is not External (External scenarios do not need
/// PAR fitting so missing observations are expected).
///
/// Only runs when estimation is active (history present, not both stats and AR
/// coefficients pre-computed).
fn check_season_observation_coverage(
    data: &ParsedData,
    season_map: &cobre_core::temporal::SeasonMap,
    ctx: &mut ValidationContext,
) {
    use cobre_core::scenario::SamplingScheme;
    use std::path::Path;

    let has_history = !data.inflow_history.is_empty();
    let has_stats = !data.inflow_seasonal_stats.is_empty();
    let has_ar = !data.inflow_ar_coefficients.is_empty();
    if !has_history || (has_stats && has_ar) {
        return;
    }

    let Ok(training_source) = data
        .config
        .training_scenario_source(Path::new("config.json"))
    else {
        return;
    };
    if training_source.inflow_scheme == SamplingScheme::External {
        return;
    }

    // Stages are sorted by id (canonical order), which matches date order.
    let stage_index: Vec<(chrono::NaiveDate, chrono::NaiveDate, usize)> = data
        .stages
        .stages
        .iter()
        .filter_map(|s| s.season_id.map(|sid| (s.start_date, s.end_date, sid)))
        .collect();

    let mut season_obs_count: HashMap<usize, usize> = HashMap::new();
    for row in &data.inflow_history {
        let pos = stage_index.partition_point(|(start, _, _)| *start <= row.date);
        let season_id = if pos > 0 {
            let (_, end_date, sid) = stage_index[pos - 1];
            if row.date < end_date { Some(sid) } else { None }
        } else {
            None
        };
        if let Some(sid) = season_id {
            *season_obs_count.entry(sid).or_insert(0) += 1;
        }
    }

    for season in season_map
        .seasons
        .iter()
        .filter(|s| season_obs_count.get(&s.id).copied().unwrap_or(0) == 0)
    {
        ctx.add_warning(
            ErrorKind::ModelQuality,
            "stages.json",
            Some(format!("Season {}", season.id)),
            format!(
                "season {} ('{}') has no inflow observations in \
                 inflow_history.parquet; PAR estimation for this season will have \
                 no data unless all stages use External scenarios",
                season.id, season.label,
            ),
        );
    }
}

/// V4.4 — Contiguity within resolution bands (Rule 30).
///
/// Warns when seasons defined in `season_definitions` are not referenced by
/// any stage, helping users detect accidental gaps.
fn check_season_contiguity(
    data: &ParsedData,
    season_map: &cobre_core::temporal::SeasonMap,
    ctx: &mut ValidationContext,
) {
    let referenced_ids: HashSet<usize> = data
        .stages
        .stages
        .iter()
        .filter_map(|s| s.season_id)
        .collect();
    let defined_ids: HashSet<usize> = season_map.seasons.iter().map(|s| s.id).collect();
    let mut unreferenced: Vec<usize> = defined_ids.difference(&referenced_ids).copied().collect();
    unreferenced.sort_unstable();
    for sid in unreferenced {
        ctx.add_warning(
            ErrorKind::ModelQuality,
            "stages.json",
            Some(format!("Season {sid}")),
            format!(
                "season {sid} is defined in season_definitions but not referenced by any \
                 stage; this season will have no PAR parameters",
            ),
        );
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::panic,
    clippy::too_many_lines,
    clippy::doc_markdown,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]
mod tests {
    use super::*;
    use cobre_core::{
        EntityId,
        entities::{Bus, Hydro, HydroGenerationModel, HydroPenalties, Line, Thermal},
        initial_conditions::InitialConditions,
        penalty::GlobalPenaltyDefaults,
        temporal::{
            BlockMode, NoiseMethod, PolicyGraph, PolicyGraphType, ScenarioSourceConfig, Stage,
            StageRiskConfig, StageStateConfig,
        },
    };

    use crate::{
        config::Config,
        extensions::{FphaHyperplaneRow, HydroGeometryRow},
        stages::StagesData,
        validation::{ErrorKind, ValidationContext, schema::ParsedData},
    };

    // ── Test helpers ──────────────────────────────────────────────────────────

    /// Build a minimal valid `HydroPenalties` with all fields set to `v`.
    fn penalties_all(v: f64) -> HydroPenalties {
        HydroPenalties {
            spillage_cost: v,
            diversion_cost: v,
            fpha_turbined_cost: v,
            storage_violation_below_cost: v,
            filling_target_violation_cost: v,
            turbined_violation_below_cost: v,
            outflow_violation_below_cost: v,
            outflow_violation_above_cost: v,
            generation_violation_below_cost: v,
            evaporation_violation_cost: v,
            water_withdrawal_violation_cost: v,
            water_withdrawal_violation_pos_cost: v,
            water_withdrawal_violation_neg_cost: v,
            evaporation_violation_pos_cost: v,
            evaporation_violation_neg_cost: v,
            inflow_nonnegativity_cost: 1000.0,
        }
    }

    /// Build a minimal valid `Hydro` using default sensible values.
    fn make_hydro(id: i32, downstream_id: Option<i32>) -> Hydro {
        Hydro {
            id: EntityId::from(id),
            name: format!("Hydro {id}"),
            bus_id: EntityId::from(1),
            downstream_id: downstream_id.map(EntityId::from),
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: 0.0,
            max_storage_hm3: 1000.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: HydroGenerationModel::ConstantProductivity {
                productivity_mw_per_m3s: 1.0,
            },
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 1000.0,
            min_generation_mw: 0.0,
            max_generation_mw: 1000.0,
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: None,
            evaporation_reference_volumes_hm3: None,
            diversion: None,
            filling: None,
            penalties: penalties_all(1.0),
        }
    }

    /// Build a minimal valid `Thermal`.
    fn make_thermal(id: i32, min_mw: f64, max_mw: f64) -> Thermal {
        Thermal {
            id: EntityId::from(id),
            name: format!("Thermal {id}"),
            bus_id: EntityId::from(1),
            entry_stage_id: None,
            exit_stage_id: None,
            cost_per_mwh: 100.0,
            min_generation_mw: min_mw,
            max_generation_mw: max_mw,
            gnl_config: None,
        }
    }

    /// Build one study stage with the given `id`.
    fn make_stage(id: i32) -> Stage {
        Stage {
            id,
            index: 0,
            start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: chrono::NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            season_id: None,
            blocks: vec![],
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
        }
    }

    /// Build a minimal valid `StagesData` with the given stage IDs.
    fn make_stages(ids: Vec<i32>) -> StagesData {
        StagesData {
            stages: ids.into_iter().map(make_stage).collect(),
            policy_graph: PolicyGraph {
                graph_type: PolicyGraphType::FiniteHorizon,
                annual_discount_rate: 0.06,
                transitions: vec![],
                season_map: None,
            },
        }
    }

    /// Build a minimal `ParsedData` with the provided hydros, thermals, stages,
    /// geometry, and FPHA rows.  All other fields are empty/minimal.
    #[allow(clippy::too_many_arguments)]
    fn make_data(
        hydros: Vec<Hydro>,
        thermals: Vec<Thermal>,
        lines: Vec<Line>,
        stages: StagesData,
        hydro_geometry: Vec<HydroGeometryRow>,
        fpha_hyperplanes: Vec<FphaHyperplaneRow>,
    ) -> ParsedData {
        ParsedData {
            config: minimal_config(),
            penalties: minimal_global_penalties(),
            stages,
            initial_conditions: InitialConditions {
                storage: vec![],
                filling_storage: vec![],
                past_inflows: vec![],
                recent_observations: vec![],
            },
            buses: vec![Bus {
                id: EntityId::from(1),
                name: "BUS_1".to_string(),
                deficit_segments: vec![],
                excess_cost: 100.0,
            }],
            thermals,
            hydros,
            lines,
            non_controllable_sources: vec![],
            pumping_stations: vec![],
            energy_contracts: vec![],
            hydro_geometry,
            production_models: vec![],
            fpha_hyperplanes,
            inflow_history: vec![],
            inflow_seasonal_stats: vec![],
            inflow_ar_coefficients: vec![],
            external_scenarios: vec![],
            external_load_scenarios: vec![],
            external_ncs_scenarios: vec![],
            load_seasonal_stats: vec![],
            load_factors: vec![],
            correlation: None,
            non_controllable_factors: vec![],
            ncs_models: vec![],
            thermal_bounds: vec![],
            hydro_bounds: vec![],
            line_bounds: vec![],
            pumping_bounds: vec![],
            contract_bounds: vec![],
            exchange_factors: vec![],
            generic_constraints: vec![],
            generic_constraint_bounds: vec![],
            penalty_overrides_bus: vec![],
            penalty_overrides_line: vec![],
            penalty_overrides_hydro: vec![],
            penalty_overrides_ncs: vec![],
            ncs_bounds: vec![],
        }
    }

    /// Minimal `Config` required to fill `ParsedData`.
    fn minimal_config() -> Config {
        // Use the same JSON fragment that schema.rs tests use for config.json.
        let json = r#"{
            "training": {
                "forward_passes": 10,
                "stopping_rules": [
                    { "type": "iteration_limit", "limit": 100 }
                ]
            }
        }"#;
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), json).unwrap();
        crate::config::parse_config(tmp.path()).unwrap()
    }

    /// Minimal `GlobalPenaltyDefaults` required to fill `ParsedData`.
    fn minimal_global_penalties() -> GlobalPenaltyDefaults {
        use cobre_core::entities::DeficitSegment;
        GlobalPenaltyDefaults {
            bus_deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 1.0,
            }],
            bus_excess_cost: 1.0,
            line_exchange_cost: 1.0,
            hydro: HydroPenalties {
                spillage_cost: 1.0,
                fpha_turbined_cost: 1.0,
                diversion_cost: 1.0,
                storage_violation_below_cost: 1.0,
                filling_target_violation_cost: 1.0,
                turbined_violation_below_cost: 1.0,
                outflow_violation_below_cost: 1.0,
                outflow_violation_above_cost: 1.0,
                generation_violation_below_cost: 1.0,
                evaporation_violation_cost: 1.0,
                water_withdrawal_violation_cost: 1.0,
                water_withdrawal_violation_pos_cost: 1.0,
                water_withdrawal_violation_neg_cost: 1.0,
                evaporation_violation_pos_cost: 1.0,
                evaporation_violation_neg_cost: 1.0,
                inflow_nonnegativity_cost: 1000.0,
            },
            ncs_curtailment_cost: 1.0,
        }
    }

    /// Build a minimal `FphaHyperplaneRow` with the given parameters.
    fn make_fpha_row(hydro_id: i32, stage_id: Option<i32>, plane_id: i32) -> FphaHyperplaneRow {
        FphaHyperplaneRow {
            hydro_id: EntityId::from(hydro_id),
            stage_id,
            plane_id,
            gamma_0: 100.0,
            gamma_v: 0.5, // valid: > 0
            gamma_q: 0.8,
            gamma_s: -0.02, // valid: <= 0
            kappa: 1.0,
            valid_v_min_hm3: None,
            valid_v_max_hm3: None,
            valid_q_max_m3s: None,
        }
    }

    /// Build a minimal `HydroGeometryRow`.
    fn make_geom_row(
        hydro_id: i32,
        volume_hm3: f64,
        height_m: f64,
        area_km2: f64,
    ) -> HydroGeometryRow {
        HydroGeometryRow {
            hydro_id: EntityId::from(hydro_id),
            volume_hm3,
            height_m,
            area_km2,
        }
    }

    // ── Cascade acyclicity tests ───────────────────────────────────────────────

    /// Given an acyclic cascade A -> B -> C (all have downstream_id pointing to next),
    /// no errors are produced.
    #[test]
    fn test_cascade_acyclic_valid() {
        let hydros = vec![
            make_hydro(1, Some(2)), // 1 -> 2
            make_hydro(2, Some(3)), // 2 -> 3
            make_hydro(3, None),    // root (no downstream)
        ];
        let data = make_data(hydros, vec![], vec![], make_stages(vec![0]), vec![], vec![]);
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(
            !ctx.has_errors(),
            "valid acyclic cascade should produce no errors, got: {:?}",
            ctx.errors()
        );
    }

    /// Given a cycle A -> B -> C -> A, exactly one CycleDetected error is produced.
    #[test]
    fn test_cascade_cycle_detected() {
        let hydros = vec![
            make_hydro(1, Some(2)), // 1 -> 2
            make_hydro(2, Some(3)), // 2 -> 3
            make_hydro(3, Some(1)), // 3 -> 1 (cycle!)
        ];
        let data = make_data(hydros, vec![], vec![], make_stages(vec![0]), vec![], vec![]);
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(ctx.has_errors(), "cycle should produce errors");
        let cycle_errors: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::CycleDetected)
            .collect();
        assert!(
            !cycle_errors.is_empty(),
            "should have at least one CycleDetected error"
        );
    }

    /// Empty hydro list produces no cascade errors.
    #[test]
    fn test_cascade_empty_hydros() {
        let data = make_data(vec![], vec![], vec![], make_stages(vec![0]), vec![], vec![]);
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(!ctx.has_errors());
    }

    // ── Hydro storage bounds tests ────────────────────────────────────────────

    /// min_storage > max_storage produces one InvalidValue error with "Hydro 5"
    /// and "storage" in the message.
    #[test]
    fn test_hydro_storage_min_greater_than_max() {
        let mut hydro = make_hydro(5, None);
        hydro.min_storage_hm3 = 200.0;
        hydro.max_storage_hm3 = 100.0;
        let data = make_data(
            vec![hydro],
            vec![],
            vec![],
            make_stages(vec![0]),
            vec![],
            vec![],
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(ctx.has_errors());
        let errors = ctx.errors();
        let relevant: Vec<_> = errors
            .iter()
            .filter(|e| e.kind == ErrorKind::InvalidValue)
            .collect();
        assert_eq!(relevant.len(), 1, "exactly 1 InvalidValue error expected");
        let msg = &relevant[0].message;
        assert!(
            msg.contains("Hydro 5"),
            "message should contain 'Hydro 5', got: {msg}"
        );
        assert!(
            msg.contains("storage"),
            "message should contain 'storage', got: {msg}"
        );
    }

    /// min_storage == max_storage (run-of-river) produces no error.
    #[test]
    fn test_hydro_storage_equal_bounds_valid() {
        let mut hydro = make_hydro(1, None);
        hydro.min_storage_hm3 = 500.0;
        hydro.max_storage_hm3 = 500.0;
        let data = make_data(
            vec![hydro],
            vec![],
            vec![],
            make_stages(vec![0]),
            vec![],
            vec![],
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(
            !ctx.has_errors(),
            "equal storage bounds should be valid, got: {:?}",
            ctx.errors()
        );
    }

    // ── Hydro turbine bounds tests ────────────────────────────────────────────

    /// min_turbined > max_turbined produces one InvalidValue error.
    #[test]
    fn test_hydro_turbine_min_greater_than_max() {
        let mut hydro = make_hydro(2, None);
        hydro.min_turbined_m3s = 500.0;
        hydro.max_turbined_m3s = 100.0;
        let data = make_data(
            vec![hydro],
            vec![],
            vec![],
            make_stages(vec![0]),
            vec![],
            vec![],
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(ctx.has_errors());
        let turbine_errors: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::InvalidValue)
            .collect();
        assert!(!turbine_errors.is_empty());
    }

    // ── Hydro outflow bounds tests ────────────────────────────────────────────

    /// When max_outflow_m3s is None, no outflow bound error is produced even if
    /// min_outflow_m3s has any value.
    #[test]
    fn test_hydro_outflow_no_max_no_error() {
        let mut hydro = make_hydro(3, None);
        hydro.min_outflow_m3s = 999.0;
        hydro.max_outflow_m3s = None;
        let data = make_data(
            vec![hydro],
            vec![],
            vec![],
            make_stages(vec![0]),
            vec![],
            vec![],
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(!ctx.has_errors());
    }

    /// When max_outflow_m3s is Some but min > max, one InvalidValue error is produced.
    #[test]
    fn test_hydro_outflow_min_greater_than_max() {
        let mut hydro = make_hydro(4, None);
        hydro.min_outflow_m3s = 500.0;
        hydro.max_outflow_m3s = Some(300.0);
        let data = make_data(
            vec![hydro],
            vec![],
            vec![],
            make_stages(vec![0]),
            vec![],
            vec![],
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(ctx.has_errors());
    }

    // ── Lifecycle consistency tests ───────────────────────────────────────────

    /// Hydro with entry >= exit produces one InvalidValue error.
    #[test]
    fn test_hydro_lifecycle_entry_gte_exit() {
        let mut hydro = make_hydro(7, None);
        hydro.entry_stage_id = Some(10);
        hydro.exit_stage_id = Some(5);
        let data = make_data(
            vec![hydro],
            vec![],
            vec![],
            make_stages(vec![0]),
            vec![],
            vec![],
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(ctx.has_errors());
        let errors = ctx.errors();
        assert!(
            errors.iter().any(|e| e.kind == ErrorKind::InvalidValue),
            "should have InvalidValue error for lifecycle"
        );
    }

    /// Hydro with only entry_stage_id set (no exit) produces no lifecycle error.
    #[test]
    fn test_hydro_lifecycle_only_entry_no_error() {
        let mut hydro = make_hydro(8, None);
        hydro.entry_stage_id = Some(5);
        hydro.exit_stage_id = None;
        let data = make_data(
            vec![hydro],
            vec![],
            vec![],
            make_stages(vec![0]),
            vec![],
            vec![],
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(
            !ctx.has_errors(),
            "only entry_stage_id set should produce no error, got: {:?}",
            ctx.errors()
        );
    }

    /// Hydro with valid entry < exit produces no lifecycle error.
    #[test]
    fn test_hydro_lifecycle_valid() {
        let mut hydro = make_hydro(9, None);
        hydro.entry_stage_id = Some(0);
        hydro.exit_stage_id = Some(10);
        let data = make_data(
            vec![hydro],
            vec![],
            vec![],
            make_stages(vec![0]),
            vec![],
            vec![],
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(!ctx.has_errors());
    }

    // ── Geometry monotonicity tests ───────────────────────────────────────────

    /// Empty geometry slice produces no errors.
    #[test]
    fn test_geometry_empty_no_error() {
        let data = make_data(
            vec![make_hydro(1, None)],
            vec![],
            vec![],
            make_stages(vec![0]),
            vec![],
            vec![],
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(!ctx.has_errors());
    }

    /// Strictly increasing volume, non-decreasing height and area produces no error.
    #[test]
    fn test_geometry_valid_monotonic() {
        let geometry = vec![
            make_geom_row(1, 10.0, 100.0, 1.0),
            make_geom_row(1, 20.0, 110.0, 1.5),
            make_geom_row(1, 30.0, 120.0, 2.0),
        ];
        let data = make_data(
            vec![make_hydro(1, None)],
            vec![],
            vec![],
            make_stages(vec![0]),
            geometry,
            vec![],
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(
            !ctx.has_errors(),
            "valid monotonic geometry should produce no errors, got: {:?}",
            ctx.errors()
        );
    }

    /// Non-monotonic volume produces BusinessRuleViolation with "Hydro 3" and "volume".
    #[test]
    fn test_geometry_non_monotonic_volume() {
        // Volume sequence [10.0, 20.0, 15.0] has a decrease at index 2.
        // Note: rows pre-sorted by (hydro_id, volume_hm3), but we construct
        // the violation by using the same volume values — the parser would have
        // sorted them, so [10, 15, 20] after sort. To test the actual validation,
        // we craft a case where sorted order still violates (equal volumes).
        // Use equal volumes to trigger the "not strictly increasing" check.
        let geometry = vec![
            make_geom_row(3, 10.0, 100.0, 1.0),
            make_geom_row(3, 20.0, 110.0, 1.5),
            make_geom_row(3, 20.0, 115.0, 1.6), // duplicate volume — not strictly increasing
        ];
        let data = make_data(
            vec![make_hydro(3, None)],
            vec![],
            vec![],
            make_stages(vec![0]),
            geometry,
            vec![],
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(ctx.has_errors());
        let errors = ctx.errors();
        let relevant: Vec<_> = errors
            .iter()
            .filter(|e| e.kind == ErrorKind::BusinessRuleViolation)
            .collect();
        assert!(!relevant.is_empty(), "should have BusinessRuleViolation");
        let msg = &relevant[0].message;
        assert!(
            msg.contains("Hydro 3"),
            "message should contain 'Hydro 3', got: {msg}"
        );
        assert!(
            msg.contains("volume"),
            "message should contain 'volume', got: {msg}"
        );
    }

    /// Non-monotonic height produces BusinessRuleViolation with "height" in message.
    #[test]
    fn test_geometry_non_monotonic_height() {
        let geometry = vec![
            make_geom_row(2, 10.0, 100.0, 1.0),
            make_geom_row(2, 20.0, 90.0, 1.5), // height decreased — violation
            make_geom_row(2, 30.0, 110.0, 2.0),
        ];
        let data = make_data(
            vec![make_hydro(2, None)],
            vec![],
            vec![],
            make_stages(vec![0]),
            geometry,
            vec![],
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(ctx.has_errors());
        let errors = ctx.errors();
        let relevant: Vec<_> = errors
            .iter()
            .filter(|e| e.kind == ErrorKind::BusinessRuleViolation)
            .collect();
        assert!(!relevant.is_empty());
        let msg = &relevant[0].message;
        assert!(
            msg.contains("height"),
            "message should mention 'height', got: {msg}"
        );
    }

    // ── FPHA minimum planes tests ─────────────────────────────────────────────

    /// 1 plane for (hydro, stage) is valid — minimum count is 1.
    #[test]
    fn test_fpha_one_plane_valid() {
        let rows = vec![make_fpha_row(1, Some(0), 0)];
        let data = make_data(
            vec![make_hydro(1, None)],
            vec![],
            vec![],
            make_stages(vec![0]),
            vec![],
            rows,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(
            !ctx.has_errors(),
            "1 plane should be valid (minimum is 1), got: {:?}",
            ctx.errors()
        );
    }

    /// 2 planes for (hydro, stage) is valid — minimum count is 1.
    #[test]
    fn test_fpha_two_planes_valid() {
        let rows = vec![make_fpha_row(1, Some(0), 0), make_fpha_row(1, Some(0), 1)];
        let data = make_data(
            vec![make_hydro(1, None)],
            vec![],
            vec![],
            make_stages(vec![0]),
            vec![],
            rows,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(
            !ctx.has_errors(),
            "2 planes should be valid (minimum is 1), got: {:?}",
            ctx.errors()
        );
    }

    /// 3 planes for (hydro, stage) produces no minimum-count error.
    #[test]
    fn test_fpha_minimum_planes_valid() {
        let rows = vec![
            make_fpha_row(1, Some(0), 0),
            make_fpha_row(1, Some(0), 1),
            make_fpha_row(1, Some(0), 2),
        ];
        let data = make_data(
            vec![make_hydro(1, None)],
            vec![],
            vec![],
            make_stages(vec![0]),
            vec![],
            rows,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(
            !ctx.has_errors(),
            "3 planes should be valid, got: {:?}",
            ctx.errors()
        );
    }

    // ── FPHA gamma sign tests ─────────────────────────────────────────────────

    /// Negative gamma_v produces BusinessRuleViolation.
    #[test]
    fn test_fpha_negative_gamma_v() {
        let mut row = make_fpha_row(1, None, 0);
        row.gamma_v = -0.5; // invalid: must be >= 0
        let rows = vec![row, make_fpha_row(1, None, 1), make_fpha_row(1, None, 2)];
        let data = make_data(
            vec![make_hydro(1, None)],
            vec![],
            vec![],
            make_stages(vec![0]),
            vec![],
            rows,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(ctx.has_errors());
        let errors = ctx.errors();
        assert!(
            errors
                .iter()
                .any(|e| e.kind == ErrorKind::BusinessRuleViolation),
            "negative gamma_v should produce BusinessRuleViolation"
        );
    }

    /// Positive gamma_s produces BusinessRuleViolation.
    #[test]
    fn test_fpha_positive_gamma_s() {
        let mut row = make_fpha_row(1, None, 0);
        row.gamma_s = 0.1; // invalid: must be <= 0
        let rows = vec![row, make_fpha_row(1, None, 1), make_fpha_row(1, None, 2)];
        let data = make_data(
            vec![make_hydro(1, None)],
            vec![],
            vec![],
            make_stages(vec![0]),
            vec![],
            rows,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(ctx.has_errors());
        let errors = ctx.errors();
        assert!(
            errors
                .iter()
                .any(|e| e.kind == ErrorKind::BusinessRuleViolation),
            "positive gamma_s should produce BusinessRuleViolation"
        );
    }

    /// gamma_s == 0.0 is valid (non-positive).
    #[test]
    fn test_fpha_gamma_s_zero_valid() {
        let rows: Vec<FphaHyperplaneRow> = (0..3)
            .map(|i| {
                let mut r = make_fpha_row(1, None, i);
                r.gamma_s = 0.0;
                r
            })
            .collect();
        let data = make_data(
            vec![make_hydro(1, None)],
            vec![],
            vec![],
            make_stages(vec![0]),
            vec![],
            rows,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(
            !ctx.has_errors(),
            "gamma_s == 0 should be valid, got: {:?}",
            ctx.errors()
        );
    }

    /// gamma_v == 0.0 is valid (constant-head plant: zero storage coefficient).
    #[test]
    fn test_fpha_gamma_v_zero_valid() {
        let mut row = make_fpha_row(1, None, 0);
        row.gamma_v = 0.0; // valid: >= 0 (constant-head)
        let rows = vec![row];
        let data = make_data(
            vec![make_hydro(1, None)],
            vec![],
            vec![],
            make_stages(vec![0]),
            vec![],
            rows,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(
            !ctx.has_errors(),
            "gamma_v == 0 should be valid for constant-head plants, got: {:?}",
            ctx.errors()
        );
    }

    /// Empty FPHA slice produces no errors (rules 11-12 are skipped).
    #[test]
    fn test_fpha_empty_no_error() {
        let data = make_data(
            vec![make_hydro(1, None)],
            vec![],
            vec![],
            make_stages(vec![0]),
            vec![],
            vec![],
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(!ctx.has_errors());
    }

    // ── Thermal generation bounds tests ───────────────────────────────────────

    /// min_generation_mw > max_generation_mw produces InvalidValue with "Thermal <id>".
    #[test]
    fn test_thermal_generation_min_greater_than_max() {
        let thermal = make_thermal(10, 500.0, 100.0); // min > max — violation
        let data = make_data(
            vec![],
            vec![thermal],
            vec![],
            make_stages(vec![0]),
            vec![],
            vec![],
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(ctx.has_errors());
        let errors = ctx.errors();
        let relevant: Vec<_> = errors
            .iter()
            .filter(|e| e.kind == ErrorKind::InvalidValue)
            .collect();
        assert_eq!(relevant.len(), 1, "exactly 1 InvalidValue error expected");
        let msg = &relevant[0].message;
        assert!(
            msg.contains("Thermal 10"),
            "message should contain 'Thermal 10', got: {msg}"
        );
    }

    /// min_generation_mw == max_generation_mw produces no error.
    #[test]
    fn test_thermal_generation_equal_bounds_valid() {
        let thermal = make_thermal(11, 200.0, 200.0);
        let data = make_data(
            vec![],
            vec![thermal],
            vec![],
            make_stages(vec![0]),
            vec![],
            vec![],
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(!ctx.has_errors());
    }

    // ── All-rules-checked test (no short-circuit) ─────────────────────────────

    /// Given two hydros each with bound violations, both errors are collected
    /// (all rules checked, no early exit).
    #[test]
    fn test_all_rules_checked_no_short_circuit() {
        let mut h1 = make_hydro(1, None);
        h1.min_storage_hm3 = 200.0;
        h1.max_storage_hm3 = 100.0; // violation

        let mut h2 = make_hydro(2, None);
        h2.min_generation_mw = 500.0;
        h2.max_generation_mw = 100.0; // violation

        let data = make_data(
            vec![h1, h2],
            vec![],
            vec![],
            make_stages(vec![0]),
            vec![],
            vec![],
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(
            ctx.errors().len() >= 2,
            "both violations should be collected; got {} errors",
            ctx.errors().len()
        );
    }

    // ── Acceptance criteria tests ─────────────────────────────────────────────

    /// AC 1: Valid data produces no errors.
    #[test]
    fn test_ac1_valid_data_no_errors() {
        let geometry = vec![
            make_geom_row(1, 10.0, 100.0, 1.0),
            make_geom_row(1, 20.0, 110.0, 2.0),
            make_geom_row(1, 30.0, 120.0, 3.0),
        ];
        let fpha: Vec<FphaHyperplaneRow> = (0..3).map(|i| make_fpha_row(1, Some(0), i)).collect();
        let data = make_data(
            vec![make_hydro(1, None)],
            vec![make_thermal(1, 0.0, 500.0)],
            vec![],
            make_stages(vec![0]),
            geometry,
            fpha,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(
            !ctx.has_errors(),
            "valid data should produce no errors, got: {:?}",
            ctx.errors()
        );
    }

    /// AC 2: Hydro id=5 with inverted storage bounds produces exactly 1 InvalidValue
    /// entry whose message contains "Hydro 5" and "storage".
    #[test]
    fn test_ac2_hydro_storage_bounds_error() {
        let mut hydro = make_hydro(5, None);
        hydro.min_storage_hm3 = 200.0;
        hydro.max_storage_hm3 = 100.0;
        let data = make_data(
            vec![hydro],
            vec![],
            vec![],
            make_stages(vec![0]),
            vec![],
            vec![],
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(ctx.has_errors());
        let errors = ctx.errors();
        let relevant: Vec<_> = errors
            .iter()
            .filter(|e| e.kind == ErrorKind::InvalidValue)
            .collect();
        assert_eq!(relevant.len(), 1);
        let msg = &relevant[0].message;
        assert!(msg.contains("Hydro 5"), "message must contain 'Hydro 5'");
        assert!(msg.contains("storage"), "message must contain 'storage'");
    }

    /// AC 3: Cycle A->B->C->A produces at least 1 CycleDetected error.
    #[test]
    fn test_ac3_cycle_detected() {
        let hydros = vec![
            make_hydro(1, Some(2)),
            make_hydro(2, Some(3)),
            make_hydro(3, Some(1)),
        ];
        let data = make_data(hydros, vec![], vec![], make_stages(vec![0]), vec![], vec![]);
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(
            ctx.errors()
                .iter()
                .any(|e| e.kind == ErrorKind::CycleDetected),
            "should have CycleDetected error"
        );
    }

    /// AC 4: Non-monotonic volume for hydro id=3 produces BusinessRuleViolation
    /// with "Hydro 3" and "volume" in the message.
    #[test]
    fn test_ac4_geometry_non_monotonic_volume_error() {
        // Use equal volumes (10.0, 20.0, 20.0) to trigger the strict-increase check.
        let geometry = vec![
            make_geom_row(3, 10.0, 100.0, 1.0),
            make_geom_row(3, 20.0, 110.0, 1.5),
            make_geom_row(3, 20.0, 115.0, 1.6),
        ];
        let data = make_data(
            vec![make_hydro(3, None)],
            vec![],
            vec![],
            make_stages(vec![0]),
            geometry,
            vec![],
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(ctx.has_errors());
        let errors = ctx.errors();
        let relevant: Vec<_> = errors
            .iter()
            .filter(|e| e.kind == ErrorKind::BusinessRuleViolation)
            .collect();
        assert!(!relevant.is_empty(), "should have BusinessRuleViolation");
        let msg = &relevant[0].message;
        assert!(msg.contains("Hydro 3"), "must contain 'Hydro 3': {msg}");
        assert!(msg.contains("volume"), "must contain 'volume': {msg}");
    }

    /// AC 5: Empty geometry and FPHA produce no errors from rules 8-12.
    #[test]
    fn test_ac5_empty_geometry_and_fpha_no_false_positives() {
        let data = make_data(
            vec![make_hydro(1, None)],
            vec![],
            vec![],
            make_stages(vec![0]),
            vec![], // empty geometry
            vec![], // empty FPHA
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(
            !ctx.has_errors(),
            "empty geometry and FPHA should produce no errors, got: {:?}",
            ctx.errors()
        );
    }

    // ── Tests for validate_semantic_stages_penalties_scenarios ────────────────

    use crate::scenarios::{
        BlockFactor, InflowArCoefficientRow, InflowSeasonalStatsRow, LoadFactorEntry,
        LoadSeasonalStatsRow,
    };
    use cobre_core::{
        entities::DeficitSegment,
        scenario::{CorrelationEntity, CorrelationGroup, CorrelationModel, CorrelationProfile},
        temporal::{Block, Transition},
    };
    use std::collections::BTreeMap;

    /// Build a minimal valid `ParsedData` for Layer 5b tests.
    /// All hydro penalties satisfy the ordering hierarchy by default.
    fn make_data_5b(
        hydros: Vec<Hydro>,
        stages: StagesData,
        buses: Vec<Bus>,
        inflow_stats: Vec<InflowSeasonalStatsRow>,
        inflow_ar: Vec<InflowArCoefficientRow>,
        correlation: Option<CorrelationModel>,
    ) -> ParsedData {
        ParsedData {
            config: minimal_config(),
            penalties: minimal_global_penalties(),
            stages,
            initial_conditions: InitialConditions {
                storage: vec![],
                filling_storage: vec![],
                past_inflows: vec![],
                recent_observations: vec![],
            },
            buses,
            thermals: vec![],
            hydros,
            lines: vec![],
            non_controllable_sources: vec![],
            pumping_stations: vec![],
            energy_contracts: vec![],
            hydro_geometry: vec![],
            production_models: vec![],
            fpha_hyperplanes: vec![],
            inflow_history: vec![],
            inflow_seasonal_stats: inflow_stats,
            inflow_ar_coefficients: inflow_ar,
            external_scenarios: vec![],
            external_load_scenarios: vec![],
            external_ncs_scenarios: vec![],
            load_seasonal_stats: vec![],
            load_factors: vec![],
            correlation,
            non_controllable_factors: vec![],
            ncs_models: vec![],
            thermal_bounds: vec![],
            hydro_bounds: vec![],
            line_bounds: vec![],
            pumping_bounds: vec![],
            contract_bounds: vec![],
            exchange_factors: vec![],
            generic_constraints: vec![],
            generic_constraint_bounds: vec![],
            penalty_overrides_bus: vec![],
            penalty_overrides_line: vec![],
            penalty_overrides_hydro: vec![],
            penalty_overrides_ncs: vec![],
            ncs_bounds: vec![],
        }
    }

    /// Build a hydro with penalties satisfying the ordering hierarchy.
    /// filling (1000) > storage_viol (500) > constraint_viol (50) > resource (1)
    fn make_hydro_ordered_penalties(id: i32) -> Hydro {
        let mut h = make_hydro(id, None);
        h.penalties = HydroPenalties {
            filling_target_violation_cost: 1000.0,
            storage_violation_below_cost: 500.0,
            turbined_violation_below_cost: 50.0,
            outflow_violation_below_cost: 50.0,
            outflow_violation_above_cost: 50.0,
            generation_violation_below_cost: 50.0,
            evaporation_violation_cost: 50.0,
            water_withdrawal_violation_cost: 50.0,
            water_withdrawal_violation_pos_cost: 50.0,
            water_withdrawal_violation_neg_cost: 50.0,
            evaporation_violation_pos_cost: 50.0,
            evaporation_violation_neg_cost: 50.0,
            spillage_cost: 1.0,
            diversion_cost: 1.0,
            fpha_turbined_cost: 2.0,
            inflow_nonnegativity_cost: 1000.0,
        };
        h
    }

    /// Build a minimal valid `StagesData` with the given stage IDs and a
    /// `FiniteHorizon` policy graph with valid transitions.
    fn make_stages_5b(ids: Vec<i32>) -> StagesData {
        StagesData {
            stages: ids.into_iter().map(make_stage).collect(),
            policy_graph: PolicyGraph {
                graph_type: PolicyGraphType::FiniteHorizon,
                annual_discount_rate: 0.06,
                transitions: vec![],
                season_map: None,
            },
        }
    }

    /// Build a bus with a single deficit segment at the given cost.
    fn make_bus_with_deficit(id: i32, cost_per_mwh: f64) -> Bus {
        Bus {
            id: EntityId::from(id),
            name: format!("Bus {id}"),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh,
            }],
            excess_cost: 100.0,
        }
    }

    /// Build a valid 2x2 symmetric correlation group.
    fn make_corr_group(name: &str, matrix: Vec<Vec<f64>>) -> CorrelationGroup {
        CorrelationGroup {
            name: name.to_string(),
            entities: vec![
                CorrelationEntity {
                    entity_type: "inflow".to_string(),
                    id: EntityId::from(1),
                },
                CorrelationEntity {
                    entity_type: "inflow".to_string(),
                    id: EntityId::from(2),
                },
            ],
            matrix,
        }
    }

    /// Build a `CorrelationModel` with a single "default" profile containing the given group.
    fn make_correlation(group: CorrelationGroup) -> CorrelationModel {
        let mut profiles = BTreeMap::new();
        profiles.insert(
            "default".to_string(),
            CorrelationProfile {
                groups: vec![group],
            },
        );
        CorrelationModel {
            method: "spectral".to_string(),
            profiles,
            schedule: vec![],
        }
    }

    // ── Valid case: no errors ─────────────────────────────────────────────────

    /// Given valid stages, ordered penalties, and valid correlation, no errors
    /// or warnings are produced.
    #[test]
    fn test_5b_all_valid_no_errors() {
        let hydro = make_hydro_ordered_penalties(1);
        // Penalty hierarchy: filling=1000 > storage_viol=500 > deficit=75 > constraint=50 > resource=1
        // deficit=75 satisfies: storage_viol(500) > deficit(75) > constraint(50) > resource(1)
        let bus = make_bus_with_deficit(1, 75.0);
        let group = make_corr_group("All", vec![vec![1.0, 0.8], vec![0.8, 1.0]]);
        let corr = make_correlation(group);
        let data = make_data_5b(
            vec![hydro],
            make_stages_5b(vec![0, 1]),
            vec![bus],
            vec![],
            vec![],
            Some(corr),
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);
        assert!(
            !ctx.has_errors(),
            "valid data should produce no errors, got: {:?}",
            ctx.errors()
        );
        assert!(
            ctx.warnings().is_empty(),
            "valid data should produce no warnings, got: {:?}",
            ctx.warnings()
        );
    }

    // ── Rule 1: Transition stage validity ─────────────────────────────────────

    /// Transition referencing a non-existent source_id produces InvalidValue error.
    #[test]
    fn test_5b_transition_invalid_source_id() {
        let mut stages = make_stages_5b(vec![0, 1]);
        stages.policy_graph.transitions = vec![Transition {
            source_id: 99, // does not exist
            target_id: 1,
            probability: 1.0,
            annual_discount_rate_override: None,
        }];
        let data = make_data_5b(
            vec![make_hydro_ordered_penalties(1)],
            stages,
            vec![make_bus_with_deficit(1, 10.0)],
            vec![],
            vec![],
            None,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);
        assert!(ctx.has_errors());
        let errors = ctx.errors();
        assert!(
            errors.iter().any(|e| e.kind == ErrorKind::InvalidValue),
            "should have InvalidValue for invalid source_id"
        );
    }

    /// Transition referencing a non-existent target_id produces InvalidValue error.
    #[test]
    fn test_5b_transition_invalid_target_id() {
        let mut stages = make_stages_5b(vec![0, 1]);
        stages.policy_graph.transitions = vec![Transition {
            source_id: 0,
            target_id: 99, // does not exist
            probability: 1.0,
            annual_discount_rate_override: None,
        }];
        let data = make_data_5b(
            vec![make_hydro_ordered_penalties(1)],
            stages,
            vec![make_bus_with_deficit(1, 10.0)],
            vec![],
            vec![],
            None,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);
        assert!(ctx.has_errors());
        assert!(
            ctx.errors()
                .iter()
                .any(|e| e.kind == ErrorKind::InvalidValue),
            "should have InvalidValue for invalid target_id"
        );
    }

    // ── Rule 2: Transition probability sums ───────────────────────────────────

    /// Transitions from stage 0 with probability sum 0.5 produce one InvalidValue
    /// error with "probability" and "stage 0" in the message.
    #[test]
    fn test_5b_transition_probability_sum_wrong() {
        let mut stages = make_stages_5b(vec![0, 1]);
        stages.policy_graph.transitions = vec![Transition {
            source_id: 0,
            target_id: 1,
            probability: 0.5, // should sum to 1.0
            annual_discount_rate_override: None,
        }];
        let data = make_data_5b(
            vec![make_hydro_ordered_penalties(1)],
            stages,
            vec![make_bus_with_deficit(1, 10.0)],
            vec![],
            vec![],
            None,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);
        assert!(ctx.has_errors());
        let errors = ctx.errors();
        let relevant: Vec<_> = errors
            .iter()
            .filter(|e| e.kind == ErrorKind::InvalidValue)
            .collect();
        assert_eq!(relevant.len(), 1, "exactly 1 InvalidValue error expected");
        let msg = &relevant[0].message;
        assert!(
            msg.contains("probability"),
            "message should contain 'probability', got: {msg}"
        );
        assert!(
            msg.contains("stage 0"),
            "message should contain 'stage 0', got: {msg}"
        );
    }

    /// Transitions from stage 0 summing exactly 1.0 produce no probability error.
    #[test]
    fn test_5b_transition_probability_sum_valid() {
        let mut stages = make_stages_5b(vec![0, 1, 2]);
        stages.policy_graph.transitions = vec![
            Transition {
                source_id: 0,
                target_id: 1,
                probability: 0.6,
                annual_discount_rate_override: None,
            },
            Transition {
                source_id: 0,
                target_id: 2,
                probability: 0.4,
                annual_discount_rate_override: None,
            },
        ];
        let data = make_data_5b(
            vec![make_hydro_ordered_penalties(1)],
            stages,
            vec![make_bus_with_deficit(1, 10.0)],
            vec![],
            vec![],
            None,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);
        let prob_errors: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::InvalidValue)
            .collect();
        assert!(
            prob_errors.is_empty(),
            "valid probability sum should produce no InvalidValue errors, got: {prob_errors:?}"
        );
    }

    // ── Rule 3: Cyclic discount rate ──────────────────────────────────────────

    /// Cyclic graph with annual_discount_rate = 0.0 produces InvalidValue error.
    #[test]
    fn test_5b_cyclic_zero_discount_rate() {
        let mut stages = make_stages_5b(vec![0]);
        stages.policy_graph.graph_type = PolicyGraphType::Cyclic;
        stages.policy_graph.annual_discount_rate = 0.0;
        let data = make_data_5b(
            vec![make_hydro_ordered_penalties(1)],
            stages,
            vec![make_bus_with_deficit(1, 10.0)],
            vec![],
            vec![],
            None,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);
        assert!(ctx.has_errors());
        assert!(
            ctx.errors()
                .iter()
                .any(|e| e.kind == ErrorKind::InvalidValue),
            "cyclic with 0 discount rate should produce InvalidValue"
        );
    }

    /// Cyclic graph with annual_discount_rate > 0.0 produces no discount rate error.
    #[test]
    fn test_5b_cyclic_positive_discount_rate_valid() {
        let mut stages = make_stages_5b(vec![0]);
        stages.policy_graph.graph_type = PolicyGraphType::Cyclic;
        stages.policy_graph.annual_discount_rate = 0.06;
        let data = make_data_5b(
            vec![make_hydro_ordered_penalties(1)],
            stages,
            vec![make_bus_with_deficit(1, 10.0)],
            vec![],
            vec![],
            None,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);
        let discount_errors: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::InvalidValue)
            .collect();
        assert!(
            discount_errors.is_empty(),
            "cyclic with positive discount rate should produce no error, got: {discount_errors:?}"
        );
    }

    // ── Rule 4: Block duration positivity ─────────────────────────────────────

    /// A block with duration_hours = 0.0 produces an InvalidValue error.
    #[test]
    fn test_5b_block_zero_duration() {
        let mut stages = make_stages_5b(vec![0]);
        stages.stages[0].blocks = vec![Block {
            index: 0,
            name: "Peak".to_string(),
            duration_hours: 0.0, // invalid
        }];
        let data = make_data_5b(
            vec![make_hydro_ordered_penalties(1)],
            stages,
            vec![make_bus_with_deficit(1, 10.0)],
            vec![],
            vec![],
            None,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);
        assert!(ctx.has_errors());
        assert!(
            ctx.errors()
                .iter()
                .any(|e| e.kind == ErrorKind::InvalidValue),
            "zero duration block should produce InvalidValue"
        );
    }

    /// A block with positive duration_hours produces no block duration error.
    #[test]
    fn test_5b_block_positive_duration_valid() {
        let mut stages = make_stages_5b(vec![0]);
        stages.stages[0].blocks = vec![Block {
            index: 0,
            name: "Peak".to_string(),
            duration_hours: 168.0,
        }];
        let data = make_data_5b(
            vec![make_hydro_ordered_penalties(1)],
            stages,
            vec![make_bus_with_deficit(1, 10.0)],
            vec![],
            vec![],
            None,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);
        let errors: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::InvalidValue)
            .collect();
        assert!(
            errors.is_empty(),
            "positive block duration should produce no error, got: {errors:?}"
        );
    }

    // ── Rule 5: CVaR parameter validity ───────────────────────────────────────

    /// CVaR alpha = 0.0 (invalid, must be in (0, 1]) produces InvalidValue.
    #[test]
    fn test_5b_cvar_alpha_zero_invalid() {
        let mut stages = make_stages_5b(vec![0]);
        stages.stages[0].risk_config = StageRiskConfig::CVaR {
            alpha: 0.0, // invalid: must be in (0, 1]
            lambda: 0.5,
        };
        let data = make_data_5b(
            vec![make_hydro_ordered_penalties(1)],
            stages,
            vec![make_bus_with_deficit(1, 10.0)],
            vec![],
            vec![],
            None,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);
        assert!(ctx.has_errors());
        assert!(
            ctx.errors()
                .iter()
                .any(|e| e.kind == ErrorKind::InvalidValue),
            "CVaR alpha=0.0 should produce InvalidValue"
        );
    }

    /// CVaR lambda = -0.1 (invalid, must be in [0, 1]) produces InvalidValue.
    #[test]
    fn test_5b_cvar_lambda_out_of_range() {
        let mut stages = make_stages_5b(vec![0]);
        stages.stages[0].risk_config = StageRiskConfig::CVaR {
            alpha: 0.95,
            lambda: -0.1, // invalid: must be in [0, 1]
        };
        let data = make_data_5b(
            vec![make_hydro_ordered_penalties(1)],
            stages,
            vec![make_bus_with_deficit(1, 10.0)],
            vec![],
            vec![],
            None,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);
        assert!(ctx.has_errors());
        assert!(
            ctx.errors()
                .iter()
                .any(|e| e.kind == ErrorKind::InvalidValue),
            "CVaR lambda=-0.1 should produce InvalidValue"
        );
    }

    // ── Rule 6: Penalty ordering — filling > storage_viol ────────────────────

    /// Hydro 7 with filling=100, storage_viol=200 (ordering violation) produces
    /// a ModelQuality warning with "filling" and "storage" in the message.
    #[test]
    fn test_5b_penalty_ordering_filling_less_than_storage_violation() {
        let mut hydro = make_hydro_ordered_penalties(7);
        hydro.penalties.filling_target_violation_cost = 100.0;
        hydro.penalties.storage_violation_below_cost = 200.0;
        let data = make_data_5b(
            vec![hydro],
            make_stages_5b(vec![0]),
            vec![make_bus_with_deficit(1, 10.0)],
            vec![],
            vec![],
            None,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);
        let warnings = ctx.warnings();
        assert!(
            !warnings.is_empty(),
            "ordering violation should produce at least 1 warning"
        );
        let relevant: Vec<_> = warnings
            .iter()
            .filter(|w| w.kind == ErrorKind::ModelQuality)
            .collect();
        assert!(
            !relevant.is_empty(),
            "should have ModelQuality warning for penalty ordering"
        );
        let msg = &relevant[0].message;
        assert!(
            msg.contains("filling"),
            "message should contain 'filling', got: {msg}"
        );
        assert!(
            msg.contains("storage"),
            "message should contain 'storage', got: {msg}"
        );
    }

    // ── Rule 11: FPHA penalty rule ────────────────────────────────────────────

    /// Hydro 3 with Fpha model, fpha_turbined_cost=-0.01
    /// produces a BusinessRuleViolation error with "Hydro 3" and "fpha_turbined_cost".
    #[test]
    fn test_5b_fpha_penalty_violated() {
        let mut hydro = make_hydro_ordered_penalties(3);
        hydro.generation_model = HydroGenerationModel::Fpha;
        hydro.penalties.fpha_turbined_cost = -0.01; // invalid: must be >= 0
        let data = make_data_5b(
            vec![hydro],
            make_stages_5b(vec![0]),
            vec![make_bus_with_deficit(1, 10.0)],
            vec![],
            vec![],
            None,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);
        assert!(ctx.has_errors());
        let errors = ctx.errors();
        let relevant: Vec<_> = errors
            .iter()
            .filter(|e| e.kind == ErrorKind::BusinessRuleViolation)
            .collect();
        assert_eq!(
            relevant.len(),
            1,
            "exactly 1 BusinessRuleViolation expected"
        );
        let msg = &relevant[0].message;
        assert!(
            msg.contains("Hydro 3"),
            "message should contain 'Hydro 3', got: {msg}"
        );
        assert!(
            msg.contains("fpha_turbined_cost"),
            "message should contain 'fpha_turbined_cost', got: {msg}"
        );
    }

    /// FPHA hydro with fpha_turbined_cost == 0.0 (constant-head) produces no error.
    #[test]
    fn test_5b_fpha_penalty_zero_valid() {
        let mut hydro = make_hydro_ordered_penalties(3);
        hydro.generation_model = HydroGenerationModel::Fpha;
        hydro.penalties.fpha_turbined_cost = 0.0; // valid: constant-head plant
        let data = make_data_5b(
            vec![hydro],
            make_stages_5b(vec![0]),
            vec![make_bus_with_deficit(1, 10.0)],
            vec![],
            vec![],
            None,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);
        let errors: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::BusinessRuleViolation)
            .collect();
        assert!(
            errors.is_empty(),
            "fpha_turbined_cost == 0.0 should be valid for constant-head plants, got: {errors:?}"
        );
    }

    /// FPHA hydro with fpha_turbined_cost == spillage_cost produces no error.
    #[test]
    fn test_5b_fpha_penalty_equal_spillage_valid() {
        let mut hydro = make_hydro_ordered_penalties(3);
        hydro.generation_model = HydroGenerationModel::Fpha;
        hydro.penalties.fpha_turbined_cost = 1.0;
        hydro.penalties.spillage_cost = 1.0; // equality is now valid
        let data = make_data_5b(
            vec![hydro],
            make_stages_5b(vec![0]),
            vec![make_bus_with_deficit(1, 10.0)],
            vec![],
            vec![],
            None,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);
        let errors: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::BusinessRuleViolation)
            .collect();
        assert!(
            errors.is_empty(),
            "fpha_turbined_cost == spillage_cost should be valid, got: {errors:?}"
        );
    }

    /// FPHA hydro with fpha_turbined_cost > spillage_cost produces no error.
    #[test]
    fn test_5b_fpha_penalty_valid() {
        let mut hydro = make_hydro_ordered_penalties(4);
        hydro.generation_model = HydroGenerationModel::Fpha;
        hydro.penalties.fpha_turbined_cost = 2.0;
        hydro.penalties.spillage_cost = 1.0;
        let data = make_data_5b(
            vec![hydro],
            make_stages_5b(vec![0]),
            vec![make_bus_with_deficit(1, 10.0)],
            vec![],
            vec![],
            None,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);
        let errors: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::BusinessRuleViolation)
            .collect();
        assert!(
            errors.is_empty(),
            "valid FPHA penalty ordering should produce no BusinessRuleViolation, got: {errors:?}"
        );
    }

    // ── Rule 12: Inflow std_m3s = 0.0 warning ────────────────────────────────

    /// std_m3s = 0.0 produces a ModelQuality warning (deterministic inflow).
    #[test]
    fn test_5b_inflow_std_zero_warning() {
        let stats = vec![InflowSeasonalStatsRow {
            hydro_id: EntityId::from(1),
            stage_id: 0,
            mean_m3s: 100.0,
            std_m3s: 0.0, // triggers ModelQuality warning
        }];
        let data = make_data_5b(
            vec![make_hydro_ordered_penalties(1)],
            make_stages_5b(vec![0]),
            vec![make_bus_with_deficit(1, 10.0)],
            stats,
            vec![],
            None,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);
        assert!(
            !ctx.has_errors(),
            "std_m3s=0.0 should produce warning, not error, got: {:?}",
            ctx.errors()
        );
        let warnings = ctx.warnings();
        assert!(
            !warnings.is_empty(),
            "std_m3s=0.0 should produce at least 1 ModelQuality warning"
        );
        assert!(
            warnings.iter().any(|w| w.kind == ErrorKind::ModelQuality),
            "should have ModelQuality warning"
        );
    }

    // ── Rule 13: residual_std_ratio consistency ───────────────────────────────

    /// Two AR coefficient rows for the same (hydro, stage) with identical
    /// `residual_std_ratio` values produce no `InvalidValue` error.
    #[test]
    fn test_5b_residual_std_ratio_consistent_no_error() {
        let ar_rows = vec![
            InflowArCoefficientRow {
                hydro_id: EntityId::from(1),
                stage_id: 0,
                lag: 1,
                coefficient: 0.5,
                residual_std_ratio: 0.85,
            },
            InflowArCoefficientRow {
                hydro_id: EntityId::from(1),
                stage_id: 0,
                lag: 2,
                coefficient: 0.3,
                residual_std_ratio: 0.85, // same ratio as lag 1
            },
        ];
        let data = make_data_5b(
            vec![make_hydro_ordered_penalties(1)],
            make_stages_5b(vec![0]),
            vec![make_bus_with_deficit(1, 10.0)],
            vec![],
            ar_rows,
            None,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);
        let errors = ctx.errors();
        let invalid_value_errors: Vec<_> = errors
            .iter()
            .filter(|e| {
                e.kind == ErrorKind::InvalidValue && e.message.contains("residual_std_ratio")
            })
            .collect();
        assert!(
            invalid_value_errors.is_empty(),
            "consistent residual_std_ratio should produce no InvalidValue errors, got: \
             {invalid_value_errors:?}"
        );
    }

    /// Two AR coefficient rows for the same (hydro, stage) with different
    /// `residual_std_ratio` values produce an `InvalidValue` error whose message
    /// contains "residual_std_ratio" and "inconsistent".
    #[test]
    fn test_5b_residual_std_ratio_inconsistent_error() {
        let ar_rows = vec![
            InflowArCoefficientRow {
                hydro_id: EntityId::from(1),
                stage_id: 0,
                lag: 1,
                coefficient: 0.5,
                residual_std_ratio: 0.85,
            },
            InflowArCoefficientRow {
                hydro_id: EntityId::from(1),
                stage_id: 0,
                lag: 2,
                coefficient: 0.3,
                residual_std_ratio: 0.90, // different ratio — triggers V-AR-4
            },
        ];
        let data = make_data_5b(
            vec![make_hydro_ordered_penalties(1)],
            make_stages_5b(vec![0]),
            vec![make_bus_with_deficit(1, 10.0)],
            vec![],
            ar_rows,
            None,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);
        let errors = ctx.errors();
        let invalid_value_errors: Vec<_> = errors
            .iter()
            .filter(|e| e.kind == ErrorKind::InvalidValue)
            .collect();
        assert!(
            !invalid_value_errors.is_empty(),
            "inconsistent residual_std_ratio should produce at least one InvalidValue error"
        );
        let ratio_error = invalid_value_errors.iter().find(|e| {
            e.message.contains("residual_std_ratio") && e.message.contains("inconsistent")
        });
        assert!(
            ratio_error.is_some(),
            "InvalidValue error message should contain 'residual_std_ratio' and 'inconsistent', \
             got: {invalid_value_errors:?}"
        );
    }

    // ── Rule 14: Correlation matrix symmetry ──────────────────────────────────

    /// Asymmetric correlation matrix (matrix[0][1]=0.8, matrix[1][0]=0.5) produces
    /// a BusinessRuleViolation error with "symmetric" in the message.
    #[test]
    fn test_5b_correlation_asymmetric() {
        let group = make_corr_group(
            "Asymmetric",
            vec![
                vec![1.0, 0.8],
                vec![0.5, 1.0], // asymmetric: should be 0.8
            ],
        );
        let corr = make_correlation(group);
        let data = make_data_5b(
            vec![make_hydro_ordered_penalties(1)],
            make_stages_5b(vec![0]),
            vec![make_bus_with_deficit(1, 10.0)],
            vec![],
            vec![],
            Some(corr),
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);
        assert!(ctx.has_errors());
        let errors = ctx.errors();
        let relevant: Vec<_> = errors
            .iter()
            .filter(|e| e.kind == ErrorKind::BusinessRuleViolation)
            .collect();
        assert!(
            !relevant.is_empty(),
            "asymmetric matrix should produce BusinessRuleViolation"
        );
        let msg = &relevant[0].message;
        assert!(
            msg.contains("symmetric"),
            "message should contain 'symmetric', got: {msg}"
        );
    }

    // ── Rule 15: Correlation matrix diagonal ──────────────────────────────────

    /// Diagonal entry not equal to 1.0 produces a BusinessRuleViolation.
    #[test]
    fn test_5b_correlation_diagonal_not_one() {
        let group = make_corr_group(
            "BadDiag",
            vec![
                vec![0.9, 0.0], // diagonal entry 0.9 != 1.0
                vec![0.0, 1.0],
            ],
        );
        let corr = make_correlation(group);
        let data = make_data_5b(
            vec![make_hydro_ordered_penalties(1)],
            make_stages_5b(vec![0]),
            vec![make_bus_with_deficit(1, 10.0)],
            vec![],
            vec![],
            Some(corr),
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);
        assert!(ctx.has_errors());
        assert!(
            ctx.errors()
                .iter()
                .any(|e| e.kind == ErrorKind::BusinessRuleViolation),
            "diagonal != 1.0 should produce BusinessRuleViolation"
        );
    }

    // ── Rule 16: Correlation coefficient range ────────────────────────────────

    /// Off-diagonal entry > 1.0 produces a BusinessRuleViolation.
    #[test]
    fn test_5b_correlation_off_diagonal_out_of_range() {
        let group = make_corr_group(
            "BadRange",
            vec![
                vec![1.0, 1.5], // 1.5 > 1.0 — out of range
                vec![1.5, 1.0],
            ],
        );
        let corr = make_correlation(group);
        let data = make_data_5b(
            vec![make_hydro_ordered_penalties(1)],
            make_stages_5b(vec![0]),
            vec![make_bus_with_deficit(1, 10.0)],
            vec![],
            vec![],
            Some(corr),
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);
        assert!(ctx.has_errors());
        assert!(
            ctx.errors()
                .iter()
                .any(|e| e.kind == ErrorKind::BusinessRuleViolation),
            "off-diagonal > 1.0 should produce BusinessRuleViolation"
        );
    }

    /// Valid symmetric correlation matrix produces no errors.
    #[test]
    fn test_5b_correlation_valid_symmetric() {
        let group = make_corr_group("Valid", vec![vec![1.0, 0.6], vec![0.6, 1.0]]);
        let corr = make_correlation(group);
        let data = make_data_5b(
            vec![make_hydro_ordered_penalties(1)],
            make_stages_5b(vec![0]),
            vec![make_bus_with_deficit(1, 10.0)],
            vec![],
            vec![],
            Some(corr),
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);
        assert!(
            !ctx.has_errors(),
            "valid symmetric matrix should produce no errors, got: {:?}",
            ctx.errors()
        );
    }

    // ── Edge cases: empty correlation and inflow data ─────────────────────────

    /// Empty correlation (None) and empty inflow data produce no false positives.
    #[test]
    fn test_5b_no_correlation_no_inflow_no_false_positives() {
        let data = make_data_5b(
            vec![make_hydro_ordered_penalties(1)],
            make_stages_5b(vec![0]),
            vec![make_bus_with_deficit(1, 10.0)],
            vec![],
            vec![],
            None, // no correlation
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);
        assert!(
            !ctx.has_errors(),
            "empty correlation and inflow should produce no errors, got: {:?}",
            ctx.errors()
        );
    }

    // ── Rules 17-18: Load factor consistency ─────────────────────────────────

    /// Build a `StagesData` with one stage that has one block (index = 0).
    fn make_stages_with_block(stage_id: i32) -> StagesData {
        let mut stage = make_stage(stage_id);
        stage.blocks = vec![Block {
            index: 0,
            name: "FLAT".to_string(),
            duration_hours: 744.0,
        }];
        StagesData {
            stages: vec![stage],
            policy_graph: PolicyGraph {
                graph_type: PolicyGraphType::FiniteHorizon,
                annual_discount_rate: 0.06,
                transitions: vec![],
                season_map: None,
            },
        }
    }

    /// `LoadFactorEntry` with a `block_id` not present in the stage's blocks
    /// produces 1 `BusinessRuleViolation` error.
    #[test]
    fn test_5b_load_factors_invalid_block_id() {
        let mut data = make_data_5b(
            vec![],
            make_stages_with_block(0),
            vec![make_bus_with_deficit(1, 10.0)],
            vec![],
            vec![],
            None,
        );
        // Stage 0 has block index 0 only; block_id=99 is invalid.
        data.load_factors = vec![LoadFactorEntry {
            bus_id: EntityId::from(1),
            stage_id: 0,
            block_factors: vec![BlockFactor {
                block_id: 99,
                factor: 1.0,
            }],
        }];
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);
        assert!(ctx.has_errors());
        let errors: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::BusinessRuleViolation)
            .collect();
        assert_eq!(
            errors.len(),
            1,
            "expected 1 BusinessRuleViolation, got: {errors:?}"
        );
        assert!(
            errors[0].file.to_string_lossy().contains("load_factors"),
            "error should reference load_factors.json"
        );
        assert!(
            errors[0].message.contains("99"),
            "message should mention invalid block_id 99"
        );
    }

    /// `LoadFactorEntry` for a `(bus_id, stage_id)` where `load_seasonal_stats`
    /// has `std_mw == 0.0` produces 1 `ModelQuality` warning.
    #[test]
    fn test_5b_load_factors_deterministic_bus_warning() {
        let mut data = make_data_5b(
            vec![],
            make_stages_with_block(0),
            vec![make_bus_with_deficit(1, 10.0)],
            vec![],
            vec![],
            None,
        );
        // Bus 1, stage 0 with std_mw == 0.0 (deterministic load).
        data.load_seasonal_stats = vec![LoadSeasonalStatsRow {
            bus_id: EntityId::from(1),
            stage_id: 0,
            mean_mw: 100.0,
            std_mw: 0.0,
        }];
        data.load_factors = vec![LoadFactorEntry {
            bus_id: EntityId::from(1),
            stage_id: 0,
            block_factors: vec![BlockFactor {
                block_id: 0,
                factor: 1.0,
            }],
        }];
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);
        assert!(
            !ctx.has_errors(),
            "deterministic load warning should not produce an error, got: {:?}",
            ctx.errors()
        );
        let warnings = ctx.warnings();
        let relevant: Vec<_> = warnings
            .iter()
            .filter(|w| w.kind == ErrorKind::ModelQuality)
            .filter(|w| w.file.to_string_lossy().contains("load_factors"))
            .collect();
        assert_eq!(
            relevant.len(),
            1,
            "expected 1 ModelQuality warning for load_factors.json, got: {warnings:?}"
        );
    }

    /// Empty `load_factors` produces zero load-related diagnostics.
    #[test]
    fn test_5b_load_factors_empty_no_errors() {
        let data = make_data_5b(
            vec![],
            make_stages_5b(vec![0]),
            vec![make_bus_with_deficit(1, 10.0)],
            vec![],
            vec![],
            None,
        );
        // load_factors is already empty in make_data_5b
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);
        let load_factor_errors: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.file.to_string_lossy().contains("load_factors"))
            .collect();
        let load_factor_warnings: Vec<_> = ctx
            .warnings()
            .into_iter()
            .filter(|w| w.file.to_string_lossy().contains("load_factors"))
            .collect();
        assert!(
            load_factor_errors.is_empty() && load_factor_warnings.is_empty(),
            "empty load_factors should produce no load-related diagnostics; \
             errors: {load_factor_errors:?}, warnings: {load_factor_warnings:?}"
        );
    }

    // ── Estimation prerequisites tests (Rules 19-21) ──────────────────────────

    use cobre_core::temporal::{SeasonCycleType, SeasonDefinition, SeasonMap};

    use crate::scenarios::InflowHistoryRow;

    /// Build a monthly `SeasonMap` with 12 seasons (January=0 .. December=11).
    fn make_monthly_season_map() -> SeasonMap {
        let seasons = (0..12u32)
            .map(|m| SeasonDefinition {
                id: m as usize,
                label: format!("Month{m}"),
                month_start: m + 1,
                day_start: None,
                month_end: None,
                day_end: None,
            })
            .collect();
        SeasonMap {
            cycle_type: SeasonCycleType::Monthly,
            seasons,
        }
    }

    /// Build `n_obs` `InflowHistoryRow` records for `hydro_id`, one per calendar
    /// month starting from January 2000, dated the 15th so they fall within a
    /// monthly stage's `[1st, 1st-of-next-month)` window.
    fn make_history_rows(hydro_id: i32, n_obs: usize) -> Vec<InflowHistoryRow> {
        let mut rows = Vec::with_capacity(n_obs);
        for i in 0..n_obs {
            let year = 2000 + (i / 12) as i32;
            let month = (i % 12) as u32 + 1;
            let date = chrono::NaiveDate::from_ymd_opt(year, month, 15).unwrap();
            rows.push(InflowHistoryRow {
                hydro_id: EntityId::from(hydro_id),
                date,
                value_m3s: 100.0,
            });
        }
        rows
    }

    /// Build a `StagesData` whose stages cover `n_months` monthly periods
    /// starting from January 2000, each with `season_id = month_index % 12`.
    /// The policy graph includes a `SeasonMap` when `with_season_map` is `true`.
    fn make_stages_with_seasons(n_months: usize, with_season_map: bool) -> StagesData {
        let mut stages = Vec::with_capacity(n_months);
        for i in 0..n_months {
            let year = 2000 + (i / 12) as i32;
            let month = (i % 12) as u32 + 1;
            let start_date = chrono::NaiveDate::from_ymd_opt(year, month, 1).unwrap();
            let (end_year, end_month) = if month == 12 {
                (year + 1, 1u32)
            } else {
                (year, month + 1)
            };
            let end_date = chrono::NaiveDate::from_ymd_opt(end_year, end_month, 1).unwrap();
            let season_id = i % 12;
            stages.push(Stage {
                index: i,
                id: i as i32,
                start_date,
                end_date,
                season_id: Some(season_id),
                blocks: vec![],
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
            });
        }
        let season_map = if with_season_map {
            Some(make_monthly_season_map())
        } else {
            None
        };
        StagesData {
            stages,
            policy_graph: PolicyGraph {
                graph_type: PolicyGraphType::FiniteHorizon,
                annual_discount_rate: 0.06,
                transitions: vec![],
                season_map,
            },
        }
    }

    /// Build `ParsedData` for estimation prerequisite tests.
    ///
    /// `inflow_history` rows are provided directly; `inflow_seasonal_stats` is
    /// empty (triggering the estimation path when history is non-empty).
    fn make_data_estimation(
        hydros: Vec<Hydro>,
        stages: StagesData,
        inflow_history: Vec<InflowHistoryRow>,
    ) -> ParsedData {
        ParsedData {
            config: minimal_config(),
            penalties: minimal_global_penalties(),
            stages,
            initial_conditions: cobre_core::initial_conditions::InitialConditions {
                storage: vec![],
                filling_storage: vec![],
                past_inflows: vec![],
                recent_observations: vec![],
            },
            buses: vec![Bus {
                id: EntityId::from(1),
                name: "BUS_1".to_string(),
                deficit_segments: vec![],
                excess_cost: 100.0,
            }],
            thermals: vec![],
            hydros,
            lines: vec![],
            non_controllable_sources: vec![],
            pumping_stations: vec![],
            energy_contracts: vec![],
            hydro_geometry: vec![],
            production_models: vec![],
            fpha_hyperplanes: vec![],
            inflow_history,
            inflow_seasonal_stats: vec![], // empty → estimation path active
            inflow_ar_coefficients: vec![],
            external_scenarios: vec![],
            external_load_scenarios: vec![],
            external_ncs_scenarios: vec![],
            load_seasonal_stats: vec![],
            load_factors: vec![],
            correlation: None,
            non_controllable_factors: vec![],
            ncs_models: vec![],
            thermal_bounds: vec![],
            hydro_bounds: vec![],
            line_bounds: vec![],
            pumping_bounds: vec![],
            contract_bounds: vec![],
            exchange_factors: vec![],
            generic_constraints: vec![],
            generic_constraint_bounds: vec![],
            penalty_overrides_bus: vec![],
            penalty_overrides_line: vec![],
            penalty_overrides_hydro: vec![],
            penalty_overrides_ncs: vec![],
            ncs_bounds: vec![],
        }
    }

    /// Given `inflow_history` present, `inflow_seasonal_stats` absent, and
    /// `stages.json` WITHOUT `season_definitions`, validation produces a
    /// `BusinessRuleViolation` mentioning "season_definitions is required".
    #[test]
    fn test_estimation_requires_season_definitions() {
        let history = make_history_rows(1, 12);
        let stages = make_stages_with_seasons(12, /*with_season_map=*/ false);
        let data = make_data_estimation(vec![make_hydro(1, None)], stages, history);

        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);

        let matching: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| {
                e.kind == ErrorKind::BusinessRuleViolation
                    && e.message.contains("season_definitions is required")
            })
            .collect();
        assert!(
            !matching.is_empty(),
            "expected a BusinessRuleViolation about season_definitions, got errors: {:?}",
            ctx.errors()
        );
    }

    /// Given `inflow_history` with only 3 observations for one (hydro, season),
    /// validation produces a `ModelQuality` warning containing "has 3 observations".
    #[test]
    fn test_estimation_warns_low_observations() {
        // 3 observations for hydro 1: one per January (season 0) over 3 years.
        let history: Vec<InflowHistoryRow> = (0..3)
            .map(|y| InflowHistoryRow {
                hydro_id: EntityId::from(1),
                date: chrono::NaiveDate::from_ymd_opt(2000 + y, 1, 15).unwrap(),
                value_m3s: 100.0,
            })
            .collect();

        // 3 years × 12 months = 36 stages, with season_map present.
        let stages = make_stages_with_seasons(36, true);
        let data = make_data_estimation(vec![make_hydro(1, None)], stages, history);

        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);

        let matching: Vec<_> = ctx
            .warnings()
            .into_iter()
            .filter(|w| {
                w.kind == ErrorKind::ModelQuality && w.message.contains("has 3 observations")
            })
            .collect();
        assert!(
            !matching.is_empty(),
            "expected a ModelQuality warning about 3 observations, got warnings: {:?}",
            ctx.warnings()
        );
    }

    /// Given `inflow_history` with observations for hydro 1 only, but `hydros`
    /// containing hydro 1 and hydro 2, validation produces a
    /// `BusinessRuleViolation` for hydro 2.
    #[test]
    fn test_estimation_error_missing_hydro() {
        let history = make_history_rows(1, 36); // only hydro 1
        let stages = make_stages_with_seasons(36, true);
        let hydros = vec![make_hydro(1, None), make_hydro(2, None)];
        let data = make_data_estimation(hydros, stages, history);

        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);

        let matching: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| {
                e.kind == ErrorKind::BusinessRuleViolation
                    && e.message.contains("hydro 2 has no observations")
            })
            .collect();
        assert!(
            !matching.is_empty(),
            "expected a BusinessRuleViolation for hydro 2, got errors: {:?}",
            ctx.errors()
        );
    }

    /// When BOTH `inflow_seasonal_stats` and `inflow_ar_coefficients` are
    /// non-empty, no estimation-related errors or warnings are produced.
    #[test]
    fn test_no_estimation_when_stats_and_coefficients_present() {
        use crate::scenarios::InflowSeasonalStatsRow;

        let history = make_history_rows(1, 12);
        let stages = make_stages_with_seasons(12, false); // no season_map

        // Provide both stats AND AR coefficients to fully deactivate estimation.
        let stats = vec![InflowSeasonalStatsRow {
            hydro_id: EntityId::from(1),
            stage_id: 0,
            mean_m3s: 500.0,
            std_m3s: 50.0,
        }];
        let ar_coefficients = vec![make_ar_row(1, 0, 1)];

        let mut data = make_data_estimation(vec![make_hydro(1, None)], stages, history);
        data.inflow_seasonal_stats = stats;
        data.inflow_ar_coefficients = ar_coefficients;

        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);

        // No estimation errors — the estimation path is not active.
        let estimation_errors: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.file.to_string_lossy().contains("inflow_history.parquet"))
            .collect();
        let estimation_warnings: Vec<_> = ctx
            .warnings()
            .into_iter()
            .filter(|w| w.file.to_string_lossy().contains("inflow_history.parquet"))
            .collect();
        assert!(
            estimation_errors.is_empty() && estimation_warnings.is_empty(),
            "stats+coefficients present should disable estimation checks; \
             errors: {estimation_errors:?}, warnings: {estimation_warnings:?}"
        );
    }

    /// When `inflow_seasonal_stats` is present but `inflow_ar_coefficients`
    /// is absent, estimation IS active and season_definitions is required.
    #[test]
    fn test_estimation_active_when_stats_present_but_coefficients_absent() {
        use crate::scenarios::InflowSeasonalStatsRow;

        let history = make_history_rows(1, 12);
        let stages = make_stages_with_seasons(12, false); // no season_map

        let stats = vec![InflowSeasonalStatsRow {
            hydro_id: EntityId::from(1),
            stage_id: 0,
            mean_m3s: 500.0,
            std_m3s: 50.0,
        }];

        let mut data = make_data_estimation(vec![make_hydro(1, None)], stages, history);
        data.inflow_seasonal_stats = stats;
        // AR coefficients NOT provided — estimation should be active.

        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);

        // Should produce a season_definitions error (Rule 19).
        let estimation_errors: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| {
                e.file.to_string_lossy().contains("inflow_history.parquet")
                    && e.message.contains("season_definitions")
            })
            .collect();
        assert!(
            !estimation_errors.is_empty(),
            "stats present without coefficients should trigger estimation checks; \
             got errors: {:?}",
            ctx.errors()
        );
    }

    // ── Rules 22-24: Past inflows coverage tests ─────────────────────────────

    /// Build an `InflowArCoefficientRow` with the given hydro_id, stage_id, and lag.
    fn make_ar_row(hydro_id: i32, stage_id: i32, lag: i32) -> InflowArCoefficientRow {
        InflowArCoefficientRow {
            hydro_id: EntityId::from(hydro_id),
            stage_id,
            lag,
            coefficient: 0.5,
            residual_std_ratio: 0.9,
        }
    }

    /// Build a `ParsedData` suitable for rules 22-24 tests.
    ///
    /// `inflow_lags_enabled` controls `state_config.inflow_lags` on stage 0.
    /// `past_inflows` is placed directly in `initial_conditions`.
    fn make_data_past_inflows(
        hydros: Vec<Hydro>,
        inflow_lags_enabled: bool,
        past_inflows: Vec<cobre_core::HydroPastInflows>,
        inflow_ar_coefficients: Vec<InflowArCoefficientRow>,
    ) -> ParsedData {
        use cobre_core::EntityId as EId;
        let stage_0_start = chrono::NaiveDate::from_ymd_opt(2020, 1, 1).unwrap();
        let stage_0 = Stage {
            id: 0,
            index: 0,
            start_date: stage_0_start,
            end_date: stage_0_start
                .checked_add_months(chrono::Months::new(1))
                .unwrap_or(stage_0_start),
            season_id: None,
            blocks: vec![],
            block_mode: BlockMode::Parallel,
            state_config: StageStateConfig {
                storage: true,
                inflow_lags: inflow_lags_enabled,
            },
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor: 1,
                noise_method: NoiseMethod::Saa,
            },
        };
        ParsedData {
            config: minimal_config(),
            penalties: minimal_global_penalties(),
            stages: StagesData {
                stages: vec![stage_0],
                policy_graph: PolicyGraph {
                    graph_type: PolicyGraphType::FiniteHorizon,
                    annual_discount_rate: 0.06,
                    transitions: vec![],
                    season_map: None,
                },
            },
            initial_conditions: cobre_core::InitialConditions {
                storage: vec![],
                filling_storage: vec![],
                past_inflows,
                recent_observations: vec![],
            },
            buses: vec![Bus {
                id: EId::from(1),
                name: "BUS_1".to_string(),
                deficit_segments: vec![],
                excess_cost: 100.0,
            }],
            thermals: vec![],
            hydros,
            lines: vec![],
            non_controllable_sources: vec![],
            pumping_stations: vec![],
            energy_contracts: vec![],
            hydro_geometry: vec![],
            production_models: vec![],
            fpha_hyperplanes: vec![],
            inflow_history: vec![],
            // Populate a sentinel entry so the estimation path (rules 19-21) is
            // inactive. Rules 22-24 are independent of the estimation path.
            inflow_seasonal_stats: vec![crate::scenarios::InflowSeasonalStatsRow {
                hydro_id: EId::from(1),
                stage_id: 0,
                mean_m3s: 500.0,
                std_m3s: 50.0,
            }],
            inflow_ar_coefficients,
            external_scenarios: vec![],
            external_load_scenarios: vec![],
            external_ncs_scenarios: vec![],
            load_seasonal_stats: vec![],
            load_factors: vec![],
            correlation: None,
            non_controllable_factors: vec![],
            ncs_models: vec![],
            thermal_bounds: vec![],
            hydro_bounds: vec![],
            line_bounds: vec![],
            pumping_bounds: vec![],
            contract_bounds: vec![],
            exchange_factors: vec![],
            generic_constraints: vec![],
            generic_constraint_bounds: vec![],
            penalty_overrides_bus: vec![],
            penalty_overrides_line: vec![],
            penalty_overrides_hydro: vec![],
            penalty_overrides_ncs: vec![],
            ncs_bounds: vec![],
        }
    }

    /// Rule 22: inflow_lags true, PAR order 3, empty past_inflows -> one
    /// `BusinessRuleViolation` mentioning "inflow_lags is enabled" and
    /// "initial_conditions.json".
    #[test]
    fn test_rule22_lags_enabled_no_past_inflows_errors() {
        let ar_rows = vec![
            make_ar_row(1, 0, 1),
            make_ar_row(1, 0, 2),
            make_ar_row(1, 0, 3),
        ];
        let data = make_data_past_inflows(
            vec![make_hydro(1, None)],
            true,
            vec![], // empty past_inflows
            ar_rows,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);

        let matching: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| {
                e.kind == ErrorKind::BusinessRuleViolation
                    && e.message.contains("inflow_lags is enabled")
            })
            .collect();
        assert_eq!(
            matching.len(),
            1,
            "expected exactly one rule-22 BusinessRuleViolation, got: {:?}",
            ctx.errors()
        );
        assert!(
            matching[0]
                .file
                .to_string_lossy()
                .contains("initial_conditions.json"),
            "error file should reference initial_conditions.json"
        );
    }

    /// Rule 23: inflow_lags true, hydro 1 PAR order 3, past_inflows has 3 values
    /// -> no rule-22/23/24 violations.
    #[test]
    fn test_rule23_sufficient_past_inflows_no_error() {
        let ar_rows = vec![
            make_ar_row(1, 0, 1),
            make_ar_row(1, 0, 2),
            make_ar_row(1, 0, 3),
        ];
        let past = vec![cobre_core::HydroPastInflows {
            hydro_id: EntityId::from(1),
            values_m3s: vec![300.0, 200.0, 100.0], // 3 values >= PAR order 3
            season_ids: None,
        }];
        let data = make_data_past_inflows(vec![make_hydro(1, None)], true, past, ar_rows);
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);

        let lag_errors: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| {
                e.kind == ErrorKind::BusinessRuleViolation
                    && e.file.to_string_lossy().contains("initial_conditions.json")
            })
            .collect();
        assert!(
            lag_errors.is_empty(),
            "sufficient past_inflows should produce no errors, got: {lag_errors:?}"
        );
    }

    /// Rule 23: inflow_lags true, hydro 1 PAR order 3, past_inflows has only 2
    /// values -> one `BusinessRuleViolation` for hydro 1 mentioning insufficient
    /// past_inflows.
    #[test]
    fn test_rule23_insufficient_past_inflows_errors() {
        let ar_rows = vec![
            make_ar_row(1, 0, 1),
            make_ar_row(1, 0, 2),
            make_ar_row(1, 0, 3),
        ];
        let past = vec![cobre_core::HydroPastInflows {
            hydro_id: EntityId::from(1),
            values_m3s: vec![200.0, 100.0], // only 2 values, need 3
            season_ids: None,
        }];
        let data = make_data_past_inflows(vec![make_hydro(1, None)], true, past, ar_rows);
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);

        let coverage_errors: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| {
                e.kind == ErrorKind::BusinessRuleViolation
                    && e.message.contains("Hydro 1")
                    && e.message.contains("insufficient past_inflows")
            })
            .collect();
        assert!(
            !coverage_errors.is_empty(),
            "insufficient past_inflows should produce a BusinessRuleViolation for Hydro 1;              got errors: {:?}",
            ctx.errors()
        );
    }

    /// Rules 22-24 are skipped when no stage has `inflow_lags: true` —
    /// no rule-22/23/24 errors regardless of past_inflows content.
    #[test]
    fn test_rules_skip_when_lags_disabled() {
        let ar_rows = vec![make_ar_row(1, 0, 1), make_ar_row(1, 0, 2)];
        let data = make_data_past_inflows(
            vec![make_hydro(1, None)],
            false,  // lags disabled
            vec![], // empty past_inflows — would trigger rule 22 if lags enabled
            ar_rows,
        );
        let mut ctx = ValidationContext::new();
        // Rules 22-24 fire only from check_past_inflows_coverage; call directly.
        check_past_inflows_coverage(&data, &mut ctx);

        assert!(
            !ctx.has_errors(),
            "lags disabled should produce no rule-22/23/24 errors; got: {:?}",
            ctx.errors()
        );
    }

    /// Rules 22-24 are skipped when `inflow_ar_coefficients` is empty —
    /// no PAR model means no lags needed regardless of the `inflow_lags` flag.
    #[test]
    fn test_rules_skip_when_par_order_zero() {
        let data = make_data_past_inflows(
            vec![make_hydro(1, None)],
            true,   // lags enabled
            vec![], // empty past_inflows — would trigger rule 22 if AR coefficients present
            vec![], // no AR coefficients -> max_order == 0, early return
        );
        let mut ctx = ValidationContext::new();
        check_past_inflows_coverage(&data, &mut ctx);

        assert!(
            !ctx.has_errors(),
            "no AR coefficients should produce no rule-22/23/24 errors; got: {:?}",
            ctx.errors()
        );
    }

    /// Rule 24: hydro ID in past_inflows that does not exist in the hydro registry
    /// produces a `BusinessRuleViolation` mentioning the unknown hydro ID.
    #[test]
    fn test_rule24_unknown_hydro_in_past_inflows_errors() {
        // past_inflows contains hydro 99, which is not in the registry.
        let past = vec![
            cobre_core::HydroPastInflows {
                hydro_id: EntityId::from(1),
                values_m3s: vec![100.0],
                season_ids: None,
            },
            cobre_core::HydroPastInflows {
                hydro_id: EntityId::from(99), // unknown
                values_m3s: vec![50.0],
                season_ids: None,
            },
        ];
        // Provide enough AR rows so rule 22 and 23 are satisfied for hydro 1.
        let ar_rows = vec![make_ar_row(1, 0, 1)];
        let data = make_data_past_inflows(
            vec![make_hydro(1, None)], // only hydro 1 in registry
            true,
            past,
            ar_rows,
        );
        let mut ctx = ValidationContext::new();
        check_past_inflows_coverage(&data, &mut ctx);

        let rule24_errors: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| {
                e.kind == ErrorKind::BusinessRuleViolation && e.message.contains("Hydro 99")
            })
            .collect();
        assert!(
            !rule24_errors.is_empty(),
            "unknown hydro 99 in past_inflows should produce a BusinessRuleViolation; \
             got errors: {:?}",
            ctx.errors()
        );
    }

    // ── Rule 32: past_inflows season_ids against SeasonMap ──────────────────

    /// Build a `ParsedData` like `make_data_past_inflows` but with a SeasonMap
    /// containing seasons with IDs `0..num_seasons`.
    fn make_data_past_inflows_with_season_map(
        hydros: Vec<Hydro>,
        past_inflows: Vec<cobre_core::HydroPastInflows>,
        inflow_ar_coefficients: Vec<InflowArCoefficientRow>,
        num_seasons: usize,
    ) -> ParsedData {
        use cobre_core::temporal::{SeasonCycleType, SeasonDefinition, SeasonMap};

        let seasons = (0..num_seasons)
            .map(|i| SeasonDefinition {
                id: i,
                label: format!("Season{i}"),
                month_start: (i % 12 + 1) as u32,
                day_start: None,
                month_end: None,
                day_end: None,
            })
            .collect();
        let season_map = SeasonMap {
            cycle_type: SeasonCycleType::Monthly,
            seasons,
        };

        let mut data = make_data_past_inflows(hydros, true, past_inflows, inflow_ar_coefficients);
        data.stages.policy_graph.season_map = Some(season_map);
        data
    }

    /// Rule 32: `past_inflows[i].season_ids` contains an ID not in the `SeasonMap`
    /// -> `BusinessRuleViolation` mentioning the invalid season ID.
    #[test]
    fn test_past_inflows_season_ids_invalid_season() {
        let past = vec![cobre_core::HydroPastInflows {
            hydro_id: EntityId::from(1),
            values_m3s: vec![300.0, 200.0],
            season_ids: Some(vec![0, 99]), // season_id 99 is invalid (only 0..4 exist)
        }];
        let ar_rows = vec![make_ar_row(1, 0, 1), make_ar_row(1, 0, 2)];
        let data = make_data_past_inflows_with_season_map(
            vec![make_hydro(1, None)],
            past,
            ar_rows,
            5, // seasons 0..4 exist
        );
        let mut ctx = ValidationContext::new();
        check_past_inflows_season_ids(&data, &mut ctx);

        let rule32_errors: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| {
                e.kind == ErrorKind::BusinessRuleViolation
                    && e.message.contains("season_id")
                    && e.message.contains("99")
            })
            .collect();
        assert!(
            !rule32_errors.is_empty(),
            "invalid season_id 99 should produce a BusinessRuleViolation; \
             got errors: {:?}",
            ctx.errors()
        );
    }

    /// Rule 32: all `season_ids` are valid -> no `BusinessRuleViolation`.
    #[test]
    fn test_past_inflows_season_ids_valid() {
        let past = vec![cobre_core::HydroPastInflows {
            hydro_id: EntityId::from(1),
            values_m3s: vec![300.0, 200.0],
            season_ids: Some(vec![3, 2]), // both exist in seasons 0..4
        }];
        let ar_rows = vec![make_ar_row(1, 0, 1), make_ar_row(1, 0, 2)];
        let data = make_data_past_inflows_with_season_map(
            vec![make_hydro(1, None)],
            past,
            ar_rows,
            5, // seasons 0..4 exist
        );
        let mut ctx = ValidationContext::new();
        check_past_inflows_season_ids(&data, &mut ctx);

        let rule32_errors: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| {
                e.kind == ErrorKind::BusinessRuleViolation
                    && e.file.to_string_lossy().contains("initial_conditions.json")
                    && e.message.contains("season_id")
            })
            .collect();
        assert!(
            rule32_errors.is_empty(),
            "valid season_ids should produce no rule-32 errors; got: {:?}",
            ctx.errors()
        );
    }

    /// Rule 32 is skipped when `season_map` is `None`.
    #[test]
    fn test_past_inflows_season_ids_no_season_map_skipped() {
        let past = vec![cobre_core::HydroPastInflows {
            hydro_id: EntityId::from(1),
            values_m3s: vec![300.0],
            season_ids: Some(vec![999]), // would be invalid if season_map were present
        }];
        let ar_rows = vec![make_ar_row(1, 0, 1)];
        // make_data_past_inflows uses season_map: None
        let data = make_data_past_inflows(vec![make_hydro(1, None)], true, past, ar_rows);
        let mut ctx = ValidationContext::new();
        check_past_inflows_season_ids(&data, &mut ctx);

        assert!(
            !ctx.has_errors(),
            "no season_map means rule 32 should be skipped; got: {:?}",
            ctx.errors()
        );
    }

    // ── Rule 25: Sobol non-power-of-2 branching factor ───────────────────────

    /// Stage with `QmcSobol` and `branching_factor: 50` (not a power of 2)
    /// produces exactly 1 `ModelQuality` warning containing the stage ID and
    /// the actual branching factor value.
    #[test]
    fn test_sobol_non_power_of_2_emits_warning() {
        let mut stages = make_stages_5b(vec![0]);
        stages.stages[0].scenario_config = ScenarioSourceConfig {
            branching_factor: 50,
            noise_method: NoiseMethod::QmcSobol,
        };
        let data = make_data_5b(
            vec![make_hydro_ordered_penalties(1)],
            stages,
            vec![make_bus_with_deficit(1, 10.0)],
            vec![],
            vec![],
            None,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);

        let all_warnings = ctx.warnings();
        let quality_warnings: Vec<_> = all_warnings
            .iter()
            .filter(|w| w.kind == ErrorKind::ModelQuality && w.message.contains("qmc_sobol"))
            .collect();
        assert_eq!(
            quality_warnings.len(),
            1,
            "expected exactly 1 ModelQuality warning, got: {:?}",
            ctx.warnings()
        );
        let msg = &quality_warnings[0].message;
        assert!(
            msg.contains("50"),
            "warning message should contain the branching factor '50', got: {msg}"
        );
        assert!(
            msg.contains("Stage "),
            "warning message should contain 'Stage ', got: {msg}"
        );
    }

    /// Stage with `QmcSobol` and `branching_factor: 64` (a power of 2)
    /// produces no warnings.
    #[test]
    fn test_sobol_power_of_2_no_warning() {
        let mut stages = make_stages_5b(vec![0]);
        stages.stages[0].scenario_config = ScenarioSourceConfig {
            branching_factor: 64,
            noise_method: NoiseMethod::QmcSobol,
        };
        let data = make_data_5b(
            vec![make_hydro_ordered_penalties(1)],
            stages,
            vec![make_bus_with_deficit(1, 10.0)],
            vec![],
            vec![],
            None,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);

        let all_warnings = ctx.warnings();
        let quality_warnings: Vec<_> = all_warnings
            .iter()
            .filter(|w| w.kind == ErrorKind::ModelQuality && w.message.contains("qmc_sobol"))
            .collect();
        assert!(
            quality_warnings.is_empty(),
            "branching_factor=64 (power of 2) should produce no ModelQuality warnings, \
             got: {quality_warnings:?}"
        );
    }

    /// Stage with `Saa` and `branching_factor: 50` (not a power of 2)
    /// produces no warnings — the check only applies to `QmcSobol`.
    #[test]
    fn test_saa_non_power_of_2_no_warning() {
        let mut stages = make_stages_5b(vec![0]);
        stages.stages[0].scenario_config = ScenarioSourceConfig {
            branching_factor: 50,
            noise_method: NoiseMethod::Saa,
        };
        let data = make_data_5b(
            vec![make_hydro_ordered_penalties(1)],
            stages,
            vec![make_bus_with_deficit(1, 10.0)],
            vec![],
            vec![],
            None,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);

        let all_warnings = ctx.warnings();
        let quality_warnings: Vec<_> = all_warnings
            .iter()
            .filter(|w| w.kind == ErrorKind::ModelQuality && w.message.contains("qmc_sobol"))
            .collect();
        assert!(
            quality_warnings.is_empty(),
            "SAA with non-power-of-2 branching factor should produce no ModelQuality warnings, \
             got: {quality_warnings:?}"
        );
    }

    /// Two stages: stage 0 uses `QmcSobol` with `branching_factor: 100` (not a
    /// power of 2), stage 1 uses `QmcSobol` with `branching_factor: 128` (power
    /// of 2). Exactly 1 `ModelQuality` warning should be emitted, for stage 0.
    #[test]
    fn test_sobol_mixed_stages_only_warns_non_power() {
        let mut stages = make_stages_5b(vec![0, 1]);
        stages.stages[0].scenario_config = ScenarioSourceConfig {
            branching_factor: 100,
            noise_method: NoiseMethod::QmcSobol,
        };
        stages.stages[1].scenario_config = ScenarioSourceConfig {
            branching_factor: 128,
            noise_method: NoiseMethod::QmcSobol,
        };
        let data = make_data_5b(
            vec![make_hydro_ordered_penalties(1)],
            stages,
            vec![make_bus_with_deficit(1, 10.0)],
            vec![],
            vec![],
            None,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);

        let all_warnings = ctx.warnings();
        let quality_warnings: Vec<_> = all_warnings
            .iter()
            .filter(|w| w.kind == ErrorKind::ModelQuality && w.message.contains("qmc_sobol"))
            .collect();
        assert_eq!(
            quality_warnings.len(),
            1,
            "expected exactly 1 ModelQuality warning (for stage 0 only), got: {:?}",
            ctx.warnings()
        );
        let msg = &quality_warnings[0].message;
        assert!(
            msg.contains("Stage 0"),
            "warning should be for stage 0, got: {msg}"
        );
        assert!(
            msg.contains("100"),
            "warning should mention branching_factor 100, got: {msg}"
        );
    }

    // ── F2-002: External scheme requires external scenario files ──────────────

    /// Build a `Config` with `training.scenario_source.inflow.scheme = "external"`.
    fn config_with_training_external_inflow() -> Config {
        let json = r#"{
            "training": {
                "forward_passes": 10,
                "stopping_rules": [
                    { "type": "iteration_limit", "limit": 100 }
                ],
                "scenario_source": {
                    "seed": 42,
                    "inflow": { "scheme": "external" }
                }
            }
        }"#;
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), json).unwrap();
        crate::config::parse_config(tmp.path()).unwrap()
    }

    /// Build a `Config` with `simulation.scenario_source.load.scheme = "external"`.
    fn config_with_simulation_external_load() -> Config {
        let json = r#"{
            "training": {
                "forward_passes": 10,
                "stopping_rules": [
                    { "type": "iteration_limit", "limit": 100 }
                ]
            },
            "simulation": {
                "scenario_source": {
                    "seed": 7,
                    "load": { "scheme": "external" }
                }
            }
        }"#;
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), json).unwrap();
        crate::config::parse_config(tmp.path()).unwrap()
    }

    /// AC1: `config.training.scenario_source.inflow.scheme = "external"` with no
    /// `external_scenarios` data produces an error referencing `"config.json"` and
    /// field `"training.scenario_source.inflow"`.
    #[test]
    fn test_training_external_inflow_without_file_is_error() {
        let mut data = make_data_5b(
            vec![make_hydro_ordered_penalties(1)],
            make_stages_5b(vec![0]),
            vec![make_bus_with_deficit(1, 75.0)],
            vec![],
            vec![],
            None,
        );
        data.config = config_with_training_external_inflow();
        data.external_scenarios = vec![]; // no external inflow file

        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);

        let errors = ctx.errors();
        let matching: Vec<_> = errors
            .iter()
            .filter(|e| {
                e.kind == ErrorKind::BusinessRuleViolation
                    && e.file == std::path::Path::new("config.json")
                    && e.entity
                        .as_deref()
                        .is_some_and(|f| f.contains("training.scenario_source.inflow"))
            })
            .collect();
        assert_eq!(
            matching.len(),
            1,
            "expected 1 error for missing external inflow file (training), got: {errors:?}"
        );
    }

    /// AC2: `config.simulation.scenario_source.load.scheme = "external"` with no
    /// `external_load_scenarios` data produces an error referencing `"config.json"`
    /// and field `"simulation.scenario_source.load"`.
    #[test]
    fn test_simulation_external_load_without_file_is_error() {
        let mut data = make_data_5b(
            vec![make_hydro_ordered_penalties(1)],
            make_stages_5b(vec![0]),
            vec![make_bus_with_deficit(1, 75.0)],
            vec![],
            vec![],
            None,
        );
        data.config = config_with_simulation_external_load();
        data.external_load_scenarios = vec![]; // no external load file

        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);

        let errors = ctx.errors();
        let matching: Vec<_> = errors
            .iter()
            .filter(|e| {
                e.kind == ErrorKind::BusinessRuleViolation
                    && e.file == std::path::Path::new("config.json")
                    && e.entity
                        .as_deref()
                        .is_some_and(|f| f.contains("simulation.scenario_source.load"))
            })
            .collect();
        assert_eq!(
            matching.len(),
            1,
            "expected 1 error for missing external load file (simulation), got: {errors:?}"
        );
    }

    /// Training uses External inflow but the external file is present: no error.
    #[test]
    fn test_training_external_inflow_with_file_is_ok() {
        use cobre_core::scenario::ExternalScenarioRow;
        let mut data = make_data_5b(
            vec![make_hydro_ordered_penalties(1)],
            make_stages_5b(vec![0]),
            vec![make_bus_with_deficit(1, 75.0)],
            vec![],
            vec![],
            None,
        );
        data.config = config_with_training_external_inflow();
        // Provide at least one row so the file is considered non-empty.
        data.external_scenarios = vec![ExternalScenarioRow {
            hydro_id: EntityId::from(1),
            stage_id: 0,
            scenario_id: 1,
            value_m3s: 10.0,
        }];

        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);

        let errors = ctx.errors();
        let external_errors: Vec<_> = errors
            .iter()
            .filter(|e| {
                e.kind == ErrorKind::BusinessRuleViolation
                    && e.file == std::path::Path::new("config.json")
            })
            .collect();
        assert!(
            external_errors.is_empty(),
            "no external-file errors expected when file is present, got: {external_errors:?}"
        );
    }

    // ── Rule 27: Season ID range coverage tests ───────────────────────────────

    /// Build a `StagesData` where each stage has `season_id = Some(i % num_seasons)`.
    /// The policy graph contains a `SeasonMap` with `num_seasons` seasons (IDs 0..num_seasons).
    fn make_stages_with_explicit_season_map(num_stages: usize, num_seasons: usize) -> StagesData {
        let seasons = (0..num_seasons)
            .map(|i| SeasonDefinition {
                id: i,
                label: format!("Season{i}"),
                month_start: (i % 12 + 1) as u32,
                day_start: None,
                month_end: None,
                day_end: None,
            })
            .collect();
        let season_map = SeasonMap {
            cycle_type: SeasonCycleType::Monthly,
            seasons,
        };
        let stages = (0..num_stages)
            .map(|i| Stage {
                id: i as i32,
                index: i,
                start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                end_date: chrono::NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
                season_id: Some(i % num_seasons),
                blocks: vec![],
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
            })
            .collect();
        StagesData {
            stages,
            policy_graph: PolicyGraph {
                graph_type: PolicyGraphType::FiniteHorizon,
                annual_discount_rate: 0.06,
                transitions: vec![],
                season_map: Some(season_map),
            },
        }
    }

    /// Given a monthly study with 12 seasons (IDs 0-11) and 12 stages each
    /// referencing seasons 0-11, no errors are emitted by rule 27.
    #[test]
    fn test_season_id_range_coverage_valid_monthly() {
        let stages = make_stages_with_explicit_season_map(12, 12);
        let data = make_data_5b(
            vec![make_hydro_ordered_penalties(1)],
            stages,
            vec![make_bus_with_deficit(1, 10.0)],
            vec![],
            vec![],
            None,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);

        let rule27_errors: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| {
                e.kind == ErrorKind::BusinessRuleViolation
                    && e.file == std::path::Path::new("stages.json")
                    && e.message.contains("season_definitions")
            })
            .collect();
        assert!(
            rule27_errors.is_empty(),
            "all valid season_ids should produce no rule-27 errors; got: {:?}",
            ctx.errors()
        );
    }

    /// Given a study where stage 5 has `season_id = 15` but `season_definitions`
    /// only defines seasons 0-11, one `BusinessRuleViolation` is emitted
    /// mentioning stage 5, season_id 15, and the valid range.
    #[test]
    fn test_season_id_range_coverage_undefined_season() {
        let mut stages = make_stages_with_explicit_season_map(12, 12);
        // Overwrite stage 5's season_id with an invalid value.
        stages.stages[5].season_id = Some(15);

        let data = make_data_5b(
            vec![make_hydro_ordered_penalties(1)],
            stages,
            vec![make_bus_with_deficit(1, 10.0)],
            vec![],
            vec![],
            None,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);

        let rule27_errors: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| {
                e.kind == ErrorKind::BusinessRuleViolation
                    && e.message.contains("stage 5")
                    && e.message.contains("season_id 15")
            })
            .collect();
        assert_eq!(
            rule27_errors.len(),
            1,
            "expected exactly one rule-27 error for stage 5 / season_id 15; got: {:?}",
            ctx.errors()
        );
        // The error message must include the valid range.
        assert!(
            rule27_errors[0].message.contains("season_definitions"),
            "error message should mention season_definitions; got: {}",
            rule27_errors[0].message
        );
    }

    /// Given a study with `season_map = None` (no `season_definitions`),
    /// the season_id consistency check is skipped — no errors from rule 27.
    #[test]
    fn test_season_id_range_coverage_no_season_map() {
        // Build stages with season_ids but no SeasonMap.
        let mut stages = make_stages_5b(vec![0, 1, 2]);
        stages.stages[0].season_id = Some(0);
        stages.stages[1].season_id = Some(1);
        stages.stages[2].season_id = Some(99); // would be invalid if season_map were present

        let data = make_data_5b(
            vec![make_hydro_ordered_penalties(1)],
            stages,
            vec![make_bus_with_deficit(1, 10.0)],
            vec![],
            vec![],
            None,
        );
        let mut ctx = ValidationContext::new();
        // Call the function directly to isolate rule 27 from other rules.
        check_season_id_consistency(&data, &mut ctx);

        assert!(
            !ctx.has_errors(),
            "no season_map means rule 27 should be skipped entirely; got: {:?}",
            ctx.errors()
        );
    }

    /// Given a study where two stages reference undefined season IDs, two
    /// `BusinessRuleViolation` errors are emitted — one per offending stage.
    #[test]
    fn test_season_id_range_coverage_multiple_violations() {
        let mut stages = make_stages_with_explicit_season_map(12, 12);
        // Stages 3 and 7 both reference invalid season IDs.
        stages.stages[3].season_id = Some(20);
        stages.stages[7].season_id = Some(55);

        let data = make_data_5b(
            vec![make_hydro_ordered_penalties(1)],
            stages,
            vec![make_bus_with_deficit(1, 10.0)],
            vec![],
            vec![],
            None,
        );
        let mut ctx = ValidationContext::new();
        check_season_id_consistency(&data, &mut ctx);

        let rule27_errors: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| {
                e.kind == ErrorKind::BusinessRuleViolation
                    && e.message.contains("season_definitions")
            })
            .collect();
        assert_eq!(
            rule27_errors.len(),
            2,
            "expected two rule-27 errors (one per offending stage); got: {:?}",
            ctx.errors()
        );
        let has_stage3 = rule27_errors
            .iter()
            .any(|e| e.message.contains("stage 3") && e.message.contains("season_id 20"));
        let has_stage7 = rule27_errors
            .iter()
            .any(|e| e.message.contains("stage 7") && e.message.contains("season_id 55"));
        assert!(has_stage3, "expected an error for stage 3 / season_id 20");
        assert!(has_stage7, "expected an error for stage 7 / season_id 55");
    }

    // ── Rule 29: Resolution consistency tests ─────────────────────────────────

    /// Build a `StagesData` for Rule 29 tests.  Each `Stage` is given an
    /// explicit `start_date`, `end_date`, and `season_id`.  The `SeasonMap`
    /// is constructed from the union of all supplied `season_id` values.
    fn make_stages_for_resolution_check(
        stage_specs: Vec<(i32, chrono::NaiveDate, chrono::NaiveDate, usize)>,
    ) -> StagesData {
        let season_ids: std::collections::BTreeSet<usize> =
            stage_specs.iter().map(|&(_, _, _, sid)| sid).collect();
        let seasons: Vec<SeasonDefinition> = season_ids
            .iter()
            .enumerate()
            .map(|(pos, &id)| SeasonDefinition {
                id,
                label: format!("Season{id}"),
                month_start: (pos % 12 + 1) as u32,
                day_start: None,
                month_end: None,
                day_end: None,
            })
            .collect();
        let season_map = SeasonMap {
            cycle_type: SeasonCycleType::Monthly,
            seasons,
        };
        let stages = stage_specs
            .into_iter()
            .enumerate()
            .map(|(index, (id, start_date, end_date, season_id))| Stage {
                id,
                index,
                start_date,
                end_date,
                season_id: Some(season_id),
                blocks: vec![],
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
            })
            .collect();
        StagesData {
            stages,
            policy_graph: PolicyGraph {
                graph_type: PolicyGraphType::FiniteHorizon,
                annual_discount_rate: 0.06,
                transitions: vec![],
                season_map: Some(season_map),
            },
        }
    }

    /// Given a monthly study where all stages for each season_id have
    /// durations between 28 and 31 days, no errors are emitted by rule 29.
    #[test]
    fn test_resolution_consistency_monthly_valid() {
        use chrono::NaiveDate;
        // 12 monthly stages; each stage's season_id matches its month index.
        // Durations vary between 28 and 31 days — well within the 7-day band.
        let specs = vec![
            (
                0,
                NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
                0,
            ), // 31d
            (
                1,
                NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
                NaiveDate::from_ymd_opt(2024, 3, 1).unwrap(),
                1,
            ), // 29d (leap)
            (
                2,
                NaiveDate::from_ymd_opt(2024, 3, 1).unwrap(),
                NaiveDate::from_ymd_opt(2024, 4, 1).unwrap(),
                2,
            ), // 31d
            (
                3,
                NaiveDate::from_ymd_opt(2024, 4, 1).unwrap(),
                NaiveDate::from_ymd_opt(2024, 5, 1).unwrap(),
                3,
            ), // 30d
            (
                4,
                NaiveDate::from_ymd_opt(2024, 5, 1).unwrap(),
                NaiveDate::from_ymd_opt(2024, 6, 1).unwrap(),
                4,
            ), // 31d
            (
                5,
                NaiveDate::from_ymd_opt(2024, 6, 1).unwrap(),
                NaiveDate::from_ymd_opt(2024, 7, 1).unwrap(),
                5,
            ), // 30d
            (
                6,
                NaiveDate::from_ymd_opt(2024, 7, 1).unwrap(),
                NaiveDate::from_ymd_opt(2024, 8, 1).unwrap(),
                6,
            ), // 31d
            (
                7,
                NaiveDate::from_ymd_opt(2024, 8, 1).unwrap(),
                NaiveDate::from_ymd_opt(2024, 9, 1).unwrap(),
                7,
            ), // 31d
            (
                8,
                NaiveDate::from_ymd_opt(2024, 9, 1).unwrap(),
                NaiveDate::from_ymd_opt(2024, 10, 1).unwrap(),
                8,
            ), // 30d
            (
                9,
                NaiveDate::from_ymd_opt(2024, 10, 1).unwrap(),
                NaiveDate::from_ymd_opt(2024, 11, 1).unwrap(),
                9,
            ), // 31d
            (
                10,
                NaiveDate::from_ymd_opt(2024, 11, 1).unwrap(),
                NaiveDate::from_ymd_opt(2024, 12, 1).unwrap(),
                10,
            ), // 30d
            (
                11,
                NaiveDate::from_ymd_opt(2024, 12, 1).unwrap(),
                NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(),
                11,
            ), // 31d
        ];
        let stages = make_stages_for_resolution_check(specs);
        let data = make_data_5b(
            vec![make_hydro_ordered_penalties(1)],
            stages,
            vec![make_bus_with_deficit(1, 10.0)],
            vec![],
            vec![],
            None,
        );
        let mut ctx = ValidationContext::new();
        check_season_id_consistency(&data, &mut ctx);

        let rule29_errors: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| {
                e.kind == ErrorKind::BusinessRuleViolation
                    && e.message.contains("incompatible durations")
            })
            .collect();
        assert!(
            rule29_errors.is_empty(),
            "monthly study with 28-31d stages should produce no rule-29 errors; got: {rule29_errors:?}"
        );
    }

    /// Given a study where season_id 0 is shared by a 30-day stage (monthly)
    /// and a 91-day stage (quarterly), one `BusinessRuleViolation` is emitted
    /// mentioning season_id 0 and listing both stages with their durations.
    #[test]
    fn test_resolution_consistency_mixed_monthly_quarterly() {
        use chrono::NaiveDate;
        // season_id 0: one monthly stage (30d) and one quarterly stage (91d).
        let specs = vec![
            (
                0,
                NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                NaiveDate::from_ymd_opt(2024, 1, 31).unwrap(),
                0,
            ), // 30d monthly
            (
                1,
                NaiveDate::from_ymd_opt(2024, 4, 1).unwrap(),
                NaiveDate::from_ymd_opt(2024, 7, 1).unwrap(),
                0,
            ), // 91d quarterly
        ];
        let stages = make_stages_for_resolution_check(specs);
        let data = make_data_5b(
            vec![make_hydro_ordered_penalties(1)],
            stages,
            vec![make_bus_with_deficit(1, 10.0)],
            vec![],
            vec![],
            None,
        );
        let mut ctx = ValidationContext::new();
        check_season_id_consistency(&data, &mut ctx);

        let rule29_errors: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| {
                e.kind == ErrorKind::BusinessRuleViolation
                    && e.message.contains("incompatible durations")
            })
            .collect();
        assert_eq!(
            rule29_errors.len(),
            1,
            "expected exactly one rule-29 error for season_id 0; got: {rule29_errors:?}"
        );
        let msg = &rule29_errors[0].message;
        assert!(
            msg.contains("season_id 0"),
            "error message must mention season_id 0; got: {msg}"
        );
        assert!(
            msg.contains("stage 0") && msg.contains("stage 1"),
            "error message must list both conflicting stage IDs; got: {msg}"
        );
        assert!(
            msg.contains("30d") && msg.contains("91d"),
            "error message must include durations; got: {msg}"
        );
    }

    /// Given a Custom SeasonMap study with monthly stages (season_ids 0-11,
    /// durations 28-31d) and quarterly stages (season_ids 12-15, durations
    /// 89-92d), where all stages reference disjoint season_id ranges, no
    /// errors are emitted by rule 29.
    #[test]
    fn test_resolution_consistency_disjoint_resolutions() {
        use chrono::NaiveDate;
        // Monthly stages: season_ids 0-11, one stage each.
        let monthly: Vec<(i32, chrono::NaiveDate, chrono::NaiveDate, usize)> = vec![
            (
                0,
                NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
                0,
            ),
            (
                1,
                NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
                NaiveDate::from_ymd_opt(2024, 3, 1).unwrap(),
                1,
            ),
            (
                2,
                NaiveDate::from_ymd_opt(2024, 3, 1).unwrap(),
                NaiveDate::from_ymd_opt(2024, 4, 1).unwrap(),
                2,
            ),
            (
                3,
                NaiveDate::from_ymd_opt(2024, 4, 1).unwrap(),
                NaiveDate::from_ymd_opt(2024, 5, 1).unwrap(),
                3,
            ),
            (
                4,
                NaiveDate::from_ymd_opt(2024, 5, 1).unwrap(),
                NaiveDate::from_ymd_opt(2024, 6, 1).unwrap(),
                4,
            ),
            (
                5,
                NaiveDate::from_ymd_opt(2024, 6, 1).unwrap(),
                NaiveDate::from_ymd_opt(2024, 7, 1).unwrap(),
                5,
            ),
            (
                6,
                NaiveDate::from_ymd_opt(2024, 7, 1).unwrap(),
                NaiveDate::from_ymd_opt(2024, 8, 1).unwrap(),
                6,
            ),
            (
                7,
                NaiveDate::from_ymd_opt(2024, 8, 1).unwrap(),
                NaiveDate::from_ymd_opt(2024, 9, 1).unwrap(),
                7,
            ),
            (
                8,
                NaiveDate::from_ymd_opt(2024, 9, 1).unwrap(),
                NaiveDate::from_ymd_opt(2024, 10, 1).unwrap(),
                8,
            ),
            (
                9,
                NaiveDate::from_ymd_opt(2024, 10, 1).unwrap(),
                NaiveDate::from_ymd_opt(2024, 11, 1).unwrap(),
                9,
            ),
            (
                10,
                NaiveDate::from_ymd_opt(2024, 11, 1).unwrap(),
                NaiveDate::from_ymd_opt(2024, 12, 1).unwrap(),
                10,
            ),
            (
                11,
                NaiveDate::from_ymd_opt(2024, 12, 1).unwrap(),
                NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(),
                11,
            ),
        ];
        // Quarterly stages: season_ids 12-15, one stage each.
        let quarterly: Vec<(i32, chrono::NaiveDate, chrono::NaiveDate, usize)> = vec![
            (
                12,
                NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                NaiveDate::from_ymd_opt(2024, 4, 1).unwrap(),
                12,
            ), // 91d
            (
                13,
                NaiveDate::from_ymd_opt(2024, 4, 1).unwrap(),
                NaiveDate::from_ymd_opt(2024, 7, 1).unwrap(),
                13,
            ), // 91d
            (
                14,
                NaiveDate::from_ymd_opt(2024, 7, 1).unwrap(),
                NaiveDate::from_ymd_opt(2024, 10, 1).unwrap(),
                14,
            ), // 92d
            (
                15,
                NaiveDate::from_ymd_opt(2024, 10, 1).unwrap(),
                NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(),
                15,
            ), // 92d
        ];
        let specs: Vec<_> = monthly.into_iter().chain(quarterly).collect();
        let stages = make_stages_for_resolution_check(specs);
        let data = make_data_5b(
            vec![make_hydro_ordered_penalties(1)],
            stages,
            vec![make_bus_with_deficit(1, 10.0)],
            vec![],
            vec![],
            None,
        );
        let mut ctx = ValidationContext::new();
        check_season_id_consistency(&data, &mut ctx);

        let rule29_errors: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| {
                e.kind == ErrorKind::BusinessRuleViolation
                    && e.message.contains("incompatible durations")
            })
            .collect();
        assert!(
            rule29_errors.is_empty(),
            "disjoint monthly (0-11) and quarterly (12-15) season_ids should produce no rule-29 errors; got: {rule29_errors:?}"
        );
    }

    /// Given a study where season_id 3 is shared by a 7-day stage (weekly)
    /// and a 30-day stage (monthly), one `BusinessRuleViolation` is emitted
    /// for season_id 3.
    #[test]
    fn test_resolution_consistency_weekly_vs_monthly() {
        use chrono::NaiveDate;
        // season_id 3: one weekly stage (7d) and one monthly stage (30d).
        let specs = vec![
            (
                0,
                NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                NaiveDate::from_ymd_opt(2024, 1, 8).unwrap(),
                3,
            ), // 7d weekly
            (
                1,
                NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
                NaiveDate::from_ymd_opt(2024, 3, 2).unwrap(),
                3,
            ), // 30d monthly
        ];
        let stages = make_stages_for_resolution_check(specs);
        let data = make_data_5b(
            vec![make_hydro_ordered_penalties(1)],
            stages,
            vec![make_bus_with_deficit(1, 10.0)],
            vec![],
            vec![],
            None,
        );
        let mut ctx = ValidationContext::new();
        check_season_id_consistency(&data, &mut ctx);

        let rule29_errors: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| {
                e.kind == ErrorKind::BusinessRuleViolation
                    && e.message.contains("incompatible durations")
            })
            .collect();
        assert_eq!(
            rule29_errors.len(),
            1,
            "expected exactly one rule-29 error for season_id 3; got: {rule29_errors:?}"
        );
        let msg = &rule29_errors[0].message;
        assert!(
            msg.contains("season_id 3"),
            "error message must mention season_id 3; got: {msg}"
        );
        assert!(
            msg.contains("7d") && msg.contains("30d"),
            "error message must include both stage durations; got: {msg}"
        );
    }

    // ── Rule 28: Observation coverage tests ───────────────────────────────────

    /// Given a monthly study with 12 seasons and observations present for all
    /// 12 seasons (one per month over 3 years), no warnings are emitted by
    /// rule 28.
    #[test]
    fn test_observation_coverage_all_seasons_have_obs() {
        // 3 years × 12 months = 36 stages; 3 observations per season (one per year).
        let stages = make_stages_with_seasons(36, /*with_season_map=*/ true);
        let history = make_history_rows(1, 36);
        let data = make_data_estimation(vec![make_hydro(1, None)], stages, history);

        let mut ctx = ValidationContext::new();
        check_season_id_consistency(&data, &mut ctx);

        let rule28_warnings: Vec<_> = ctx
            .warnings()
            .into_iter()
            .filter(|w| {
                w.kind == ErrorKind::ModelQuality
                    && w.message.contains("has no inflow observations")
            })
            .collect();
        assert!(
            rule28_warnings.is_empty(),
            "all seasons with observations should produce no rule-28 warnings; got: {rule28_warnings:?}"
        );
    }

    /// Given a study where season 5 has zero observations in inflow_history and
    /// the inflow scheme is InSample (not External), a `ModelQuality` warning is
    /// emitted for season 5 mentioning zero observations.
    #[test]
    fn test_observation_coverage_season_missing_obs_non_external() {
        // Build 3 years of monthly stages with season_map (seasons 0..11).
        let stages = make_stages_with_seasons(36, /*with_season_map=*/ true);
        // Provide observations for all seasons EXCEPT season 5 (month index 5 = June).
        // make_history_rows produces one row per month starting January 2000.
        // Season 5 corresponds to months where month_index % 12 == 5, i.e., June.
        // We build rows manually, skipping every June.
        let mut history = Vec::new();
        for i in 0..36usize {
            let month_index = i % 12;
            if month_index == 5 {
                continue; // skip June — season 5 gets no observations
            }
            let year = 2000 + (i / 12) as i32;
            let month = month_index as u32 + 1;
            history.push(InflowHistoryRow {
                hydro_id: EntityId::from(1),
                date: chrono::NaiveDate::from_ymd_opt(year, month, 15).unwrap(),
                value_m3s: 100.0,
            });
        }
        // config defaults to InSample (minimal_config has no scenario_source).
        let data = make_data_estimation(vec![make_hydro(1, None)], stages, history);

        let mut ctx = ValidationContext::new();
        check_season_id_consistency(&data, &mut ctx);

        let rule28_warnings: Vec<_> = ctx
            .warnings()
            .into_iter()
            .filter(|w| {
                w.kind == ErrorKind::ModelQuality
                    && w.message.contains("has no inflow observations")
            })
            .collect();
        assert_eq!(
            rule28_warnings.len(),
            1,
            "expected exactly one rule-28 warning for season 5; got: {rule28_warnings:?}"
        );
        let msg = &rule28_warnings[0].message;
        assert!(
            msg.contains("season 5"),
            "warning message must mention season 5; got: {msg}"
        );
    }

    /// Given a study where season 5 has zero observations but the inflow scheme
    /// is External, no warning is emitted by rule 28.
    #[test]
    fn test_observation_coverage_season_missing_obs_external() {
        let stages = make_stages_with_seasons(36, /*with_season_map=*/ true);
        // Same history as above: no observations for season 5 (June).
        let mut history = Vec::new();
        for i in 0..36usize {
            let month_index = i % 12;
            if month_index == 5 {
                continue;
            }
            let year = 2000 + (i / 12) as i32;
            let month = month_index as u32 + 1;
            history.push(InflowHistoryRow {
                hydro_id: EntityId::from(1),
                date: chrono::NaiveDate::from_ymd_opt(year, month, 15).unwrap(),
                value_m3s: 100.0,
            });
        }
        let mut data = make_data_estimation(vec![make_hydro(1, None)], stages, history);
        // Override config to use External inflow scheme.
        data.config = config_with_training_external_inflow();

        let mut ctx = ValidationContext::new();
        check_season_id_consistency(&data, &mut ctx);

        let rule28_warnings: Vec<_> = ctx
            .warnings()
            .into_iter()
            .filter(|w| {
                w.kind == ErrorKind::ModelQuality
                    && w.message.contains("has no inflow observations")
            })
            .collect();
        assert!(
            rule28_warnings.is_empty(),
            "External inflow scheme should suppress rule-28 warnings; got: {rule28_warnings:?}"
        );
    }

    // ── Rule 30: Contiguity within resolution bands tests ─────────────────────

    /// Given a study where all defined seasons (0-11) are referenced by at
    /// least one stage, no warnings are emitted by rule 30.
    #[test]
    fn test_contiguity_no_gaps() {
        // 12 stages each referencing a distinct season 0..11.
        let stages = make_stages_with_explicit_season_map(12, 12);
        let data = make_data_5b(
            vec![make_hydro_ordered_penalties(1)],
            stages,
            vec![make_bus_with_deficit(1, 10.0)],
            vec![],
            vec![],
            None,
        );
        let mut ctx = ValidationContext::new();
        check_season_id_consistency(&data, &mut ctx);

        let rule30_warnings: Vec<_> = ctx
            .warnings()
            .into_iter()
            .filter(|w| {
                w.kind == ErrorKind::ModelQuality
                    && w.message.contains("not referenced by any stage")
            })
            .collect();
        assert!(
            rule30_warnings.is_empty(),
            "no gaps should produce no rule-30 warnings; got: {rule30_warnings:?}"
        );
    }

    /// Given a study where stages reference season_ids {0, 1, 2, 4, 5} and
    /// season 3 is defined in `season_definitions` but unreferenced, a
    /// `ModelQuality` warning is emitted for season 3.
    #[test]
    fn test_contiguity_gap_detected() {
        // Build a SeasonMap with 6 seasons (0..5) and 5 stages referencing
        // seasons {0, 1, 2, 4, 5} — leaving season 3 unreferenced.
        let seasons: Vec<SeasonDefinition> = (0..6)
            .map(|i| SeasonDefinition {
                id: i,
                label: format!("Season{i}"),
                month_start: (i % 12 + 1) as u32,
                day_start: None,
                month_end: None,
                day_end: None,
            })
            .collect();
        let season_map = SeasonMap {
            cycle_type: SeasonCycleType::Monthly,
            seasons,
        };
        // 5 stages: reference seasons 0, 1, 2, 4, 5 (skipping 3).
        let referenced = [0usize, 1, 2, 4, 5];
        let stages_vec: Vec<Stage> = referenced
            .iter()
            .enumerate()
            .map(|(idx, &sid)| Stage {
                id: idx as i32,
                index: idx,
                start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                end_date: chrono::NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
                season_id: Some(sid),
                blocks: vec![],
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
            })
            .collect();
        let stages = StagesData {
            stages: stages_vec,
            policy_graph: PolicyGraph {
                graph_type: PolicyGraphType::FiniteHorizon,
                annual_discount_rate: 0.06,
                transitions: vec![],
                season_map: Some(season_map),
            },
        };
        let data = make_data_5b(
            vec![make_hydro_ordered_penalties(1)],
            stages,
            vec![make_bus_with_deficit(1, 10.0)],
            vec![],
            vec![],
            None,
        );
        let mut ctx = ValidationContext::new();
        check_season_id_consistency(&data, &mut ctx);

        let rule30_warnings: Vec<_> = ctx
            .warnings()
            .into_iter()
            .filter(|w| {
                w.kind == ErrorKind::ModelQuality
                    && w.message.contains("not referenced by any stage")
            })
            .collect();
        assert_eq!(
            rule30_warnings.len(),
            1,
            "expected exactly one rule-30 warning for season 3; got: {rule30_warnings:?}"
        );
        let msg = &rule30_warnings[0].message;
        assert!(
            msg.contains("season 3"),
            "warning message must mention season 3; got: {msg}"
        );
    }

    // ── Rule 31: Observation-to-season alignment tests ────────────────────────

    /// Given a monthly study with one observation per hydro per month per year
    /// (one obs per (hydro, season, year)), no errors and no rule-31 warnings
    /// are emitted.
    #[test]
    fn test_observation_alignment_valid_monthly() {
        // 36 stages = 3 years × 12 months, season_map present.
        let stages = make_stages_with_seasons(36, /*with_season_map=*/ true);
        // Exactly one obs per (hydro 1, season, year): 36 observations total.
        let history = make_history_rows(1, 36);
        let data = make_data_estimation(vec![make_hydro(1, None)], stages, history);

        let mut ctx = ValidationContext::new();
        check_observation_season_alignment(&data, &mut ctx);

        let rule31_errors: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::BusinessRuleViolation)
            .collect();
        assert!(
            rule31_errors.is_empty(),
            "valid monthly observations should produce no rule-31 errors; got: {rule31_errors:?}"
        );
        let rule31_warnings: Vec<_> = ctx
            .warnings()
            .into_iter()
            .filter(|w| {
                w.kind == ErrorKind::BusinessRuleViolation && w.message.contains("aggregated")
            })
            .collect();
        assert!(
            rule31_warnings.is_empty(),
            "valid monthly observations should produce no aggregation warnings; got: {rule31_warnings:?}"
        );
    }

    /// Given a monthly study where hydro 1 has 2 observations for season 0
    /// (January) year 2020 and exactly 1 observation for every other season,
    /// a warning (not an error) is emitted for the duplicated season. No
    /// `BusinessRuleViolation` error is produced.
    #[test]
    fn test_observation_alignment_duplicate_obs() {
        // Season 0 (January 2020) has two observations — finer-than-season.
        // Seasons 1-11 have exactly one observation each — no coarser-than-season
        // check fires because no season is missing for the year.
        let mut history = vec![
            InflowHistoryRow {
                hydro_id: EntityId::from(1),
                date: chrono::NaiveDate::from_ymd_opt(2020, 1, 5).unwrap(),
                value_m3s: 100.0,
            },
            InflowHistoryRow {
                hydro_id: EntityId::from(1),
                date: chrono::NaiveDate::from_ymd_opt(2020, 1, 20).unwrap(),
                value_m3s: 200.0,
            },
        ];
        // Add one observation per month for February–December 2020.
        for month in 2u32..=12 {
            history.push(InflowHistoryRow {
                hydro_id: EntityId::from(1),
                date: chrono::NaiveDate::from_ymd_opt(2020, month, 15).unwrap(),
                value_m3s: f64::from(month) * 10.0,
            });
        }
        // Build 12 stages covering January 2020 – December 2020, season_map present.
        let mut stages_2020 = make_stages_with_seasons(12, /*with_season_map=*/ true);
        // Override stage dates to cover year 2020.
        for (i, stage) in stages_2020.stages.iter_mut().enumerate() {
            let month = (i % 12) as u32 + 1;
            stage.start_date = chrono::NaiveDate::from_ymd_opt(2020, month, 1).unwrap();
            let (end_year, end_month) = if month == 12 {
                (2021, 1u32)
            } else {
                (2020, month + 1)
            };
            stage.end_date = chrono::NaiveDate::from_ymd_opt(end_year, end_month, 1).unwrap();
        }
        let data = make_data_estimation(vec![make_hydro(1, None)], stages_2020, history);

        let mut ctx = ValidationContext::new();
        check_observation_season_alignment(&data, &mut ctx);

        // Must produce no errors — finer-than-season is now a warning only.
        assert!(
            ctx.errors().is_empty(),
            "finer-than-season observations must not produce errors; got: {:?}",
            ctx.errors()
        );

        // Must produce exactly one warning stating observations will be aggregated.
        let rule31_warnings: Vec<_> = ctx
            .warnings()
            .into_iter()
            .filter(|w| {
                w.kind == ErrorKind::BusinessRuleViolation
                    && w.message.contains("will be aggregated")
            })
            .collect();
        assert_eq!(
            rule31_warnings.len(),
            1,
            "expected exactly one rule-31 aggregation warning; got: {rule31_warnings:?}"
        );
        let msg = &rule31_warnings[0].message;
        assert!(
            msg.contains("hydro 1"),
            "warning must mention hydro 1; got: {msg}"
        );
        assert!(
            msg.contains("season 0"),
            "warning must mention season 0; got: {msg}"
        );
        assert!(
            msg.contains("year 2020"),
            "warning must mention year 2020; got: {msg}"
        );
        assert!(
            msg.contains(" 2 ") || msg.contains("has 2 observations"),
            "warning must mention count 2; got: {msg}"
        );
        // Verify entity context mentions hydro 1.
        assert_eq!(
            rule31_warnings[0].entity,
            Some("Hydro 1".to_string()),
            "entity context must be 'Hydro 1'; got: {:?}",
            rule31_warnings[0].entity
        );
    }

    /// Given quarterly observations (1 obs per quarter) with a monthly SeasonMap
    /// (12 seasons), a `BusinessRuleViolation` error is emitted stating that
    /// coarser-than-season observations cannot be disaggregated.
    #[test]
    fn test_observation_alignment_coarser_than_season() {
        // Build 12 monthly stages covering 2020, season_map present (12 seasons).
        let mut stages_2020 = make_stages_with_seasons(12, /*with_season_map=*/ true);
        for (i, stage) in stages_2020.stages.iter_mut().enumerate() {
            let month = (i % 12) as u32 + 1;
            stage.start_date = chrono::NaiveDate::from_ymd_opt(2020, month, 1).unwrap();
            let (end_year, end_month) = if month == 12 {
                (2021, 1u32)
            } else {
                (2020, month + 1)
            };
            stage.end_date = chrono::NaiveDate::from_ymd_opt(end_year, end_month, 1).unwrap();
        }

        // Only 4 quarterly observations (one per quarter mid-point), not 12.
        // This simulates coarser-than-season data: the hydro has observations
        // in 2020 but many (season, year) groups have no observations.
        let history = vec![
            InflowHistoryRow {
                hydro_id: EntityId::from(1),
                date: chrono::NaiveDate::from_ymd_opt(2020, 2, 15).unwrap(), // Feb → season 1
                value_m3s: 100.0,
            },
            InflowHistoryRow {
                hydro_id: EntityId::from(1),
                date: chrono::NaiveDate::from_ymd_opt(2020, 5, 15).unwrap(), // May → season 4
                value_m3s: 200.0,
            },
            InflowHistoryRow {
                hydro_id: EntityId::from(1),
                date: chrono::NaiveDate::from_ymd_opt(2020, 8, 15).unwrap(), // Aug → season 7
                value_m3s: 300.0,
            },
            InflowHistoryRow {
                hydro_id: EntityId::from(1),
                date: chrono::NaiveDate::from_ymd_opt(2020, 11, 15).unwrap(), // Nov → season 10
                value_m3s: 400.0,
            },
        ];
        let data = make_data_estimation(vec![make_hydro(1, None)], stages_2020, history);

        let mut ctx = ValidationContext::new();
        check_observation_season_alignment(&data, &mut ctx);

        // Must produce at least one coarser-than-season error.
        let coarser_errors: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| {
                e.kind == ErrorKind::BusinessRuleViolation
                    && e.message.contains("coarser-than-season")
            })
            .collect();
        assert!(
            !coarser_errors.is_empty(),
            "coarser-than-season observations must produce at least one BusinessRuleViolation error; got none"
        );
        // All errors must mention "cannot be disaggregated".
        for e in &coarser_errors {
            assert!(
                e.message.contains("cannot be disaggregated"),
                "error message must mention 'cannot be disaggregated'; got: {}",
                e.message
            );
        }
    }

    /// Given a study where `season_map` is `None`, rule 31 is skipped entirely
    /// (no errors produced even with observations present).
    #[test]
    fn test_observation_alignment_no_season_map() {
        // No season_map → rule 31 must not run.
        let stages = make_stages_with_seasons(12, /*with_season_map=*/ false);
        let history = make_history_rows(1, 12);
        let data = make_data_estimation(vec![make_hydro(1, None)], stages, history);

        let mut ctx = ValidationContext::new();
        check_observation_season_alignment(&data, &mut ctx);

        assert!(
            ctx.errors().is_empty(),
            "rule 31 must be skipped when season_map is None; got errors: {:?}",
            ctx.errors()
        );
    }

    /// Given a study where estimation is not active (both `inflow_seasonal_stats`
    /// and `inflow_ar_coefficients` are present), rule 31 is skipped.
    #[test]
    fn test_observation_alignment_estimation_inactive() {
        let stages = make_stages_with_seasons(12, /*with_season_map=*/ true);
        let history = make_history_rows(1, 12);
        let mut data = make_data_estimation(vec![make_hydro(1, None)], stages, history);
        // Supply both stats and AR coefficients to deactivate estimation.
        data.inflow_seasonal_stats = vec![crate::scenarios::InflowSeasonalStatsRow {
            hydro_id: EntityId::from(1),
            stage_id: 0,
            mean_m3s: 100.0,
            std_m3s: 10.0,
        }];
        data.inflow_ar_coefficients = vec![crate::scenarios::InflowArCoefficientRow {
            hydro_id: EntityId::from(1),
            stage_id: 0,
            lag: 1,
            coefficient: 0.5,
            residual_std_ratio: 0.9,
        }];

        let mut ctx = ValidationContext::new();
        check_observation_season_alignment(&data, &mut ctx);

        assert!(
            ctx.errors().is_empty(),
            "rule 31 must be skipped when estimation is inactive; got errors: {:?}",
            ctx.errors()
        );
    }
}
