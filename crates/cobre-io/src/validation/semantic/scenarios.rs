//! Layer 5b — scenario, penalty, and probability-data validation.
//!
//! Scenario model existence, load-factor consistency, AR
//! estimation prerequisites, external-scheme file existence,
//! past-inflows coverage and season alignment, penalty cost
//! ordering, and FPHA penalty-rule shape.

use std::collections::{HashMap, HashSet};

use super::super::{ErrorKind, ValidationContext, schema::ParsedData};

// ── Rules 6-10: Penalty ordering ──────────────────────────────────────────────

/// Checks the penalty hierarchy ordering across all hydros and buses.
///
/// Emits one `ModelQuality` warning per violated ordering check, aggregating
/// all violating entities into a single warning with the count and worst-case ID.
#[allow(clippy::too_many_lines)]
pub(super) fn check_penalty_ordering(data: &ParsedData, ctx: &mut ValidationContext) {
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
pub(super) fn check_fpha_penalty_rule(data: &ParsedData, ctx: &mut ValidationContext) {
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
pub(super) fn check_scenario_models(data: &ParsedData, ctx: &mut ValidationContext) {
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

// ── F2-002: External scheme requires external scenario files ─────────────────

/// Validates that when a class uses the `External` sampling scheme, the
/// corresponding external scenario file data is non-empty.
pub(super) fn check_external_scheme_has_files(data: &ParsedData, ctx: &mut ValidationContext) {
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
pub(super) fn check_load_factor_consistency(data: &ParsedData, ctx: &mut ValidationContext) {
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
pub(super) fn check_estimation_prerequisites(data: &ParsedData, ctx: &mut ValidationContext) {
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
pub(super) fn check_past_inflows_coverage(data: &ParsedData, ctx: &mut ValidationContext) {
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
pub(super) fn check_past_inflows_season_ids(data: &ParsedData, ctx: &mut ValidationContext) {
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
