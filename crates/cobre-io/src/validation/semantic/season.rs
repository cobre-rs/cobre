//! Layer 5b — season-domain semantic validation.
//!
//! Season-id consistency, observation-season alignment, season
//! observation coverage, and season-definition contiguity.

use std::collections::{HashMap, HashSet};

use super::super::{schema::ParsedData, ErrorKind, ValidationContext};

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
pub(super) fn check_season_id_consistency(data: &ParsedData, ctx: &mut ValidationContext) {
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
pub(super) fn check_observation_season_alignment(data: &ParsedData, ctx: &mut ValidationContext) {
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
            if row.date < end_date {
                Some(sid)
            } else {
                None
            }
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
    //
    // Boundary years (the first and last calendar year each hydro has any
    // observation) are excluded from this check. Partial coverage in a boundary
    // year is expected — real-world history commonly starts in April and ends in
    // September, giving incomplete first/last calendar years.  Interior years
    // that are missing a season ARE genuine coarser-than-season indicators and
    // are still reported.
    //
    // Edge case: a hydro with only 1 or 2 years of data has no interior years at
    // all, so the check is skipped entirely for that hydro.  Other validation
    // rules (minimum observation count) will surface insufficient data.
    let mut coarser_violations: Vec<(i32, usize, i32)> = Vec::new();
    for (&hid, years) in &hydro_years {
        let min_yr = years.iter().copied().min().unwrap_or(0);
        let max_yr = years.iter().copied().max().unwrap_or(0);
        for &yr in years {
            // Skip boundary years — partial coverage is expected there.
            if yr == min_yr || yr == max_yr {
                continue;
            }
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
pub(super) fn check_season_observation_coverage(
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
            if row.date < end_date {
                Some(sid)
            } else {
                None
            }
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
pub(super) fn check_season_contiguity(
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
