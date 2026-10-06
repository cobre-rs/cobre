//! Layer 5b — season-domain semantic validation.

use std::collections::{HashMap, HashSet};

use super::super::{ValidationContext, rules, schema::ParsedData};

use cobre_core::SeasonMap;
use cobre_core::temporal::SUB_PERIOD_TOLERANCE_DAYS;
use cobre_stochastic::season_cast::observation_occurrence_year;

// ── Rules 27+29: Season ID range coverage and resolution consistency ──────────

/// Validates stage `season_id` references and duration consistency (Rules 27, 29).
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
            ctx.emit(
                &rules::SEMANTIC_STAGE_SEASON_UNDEFINED,
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
        let min_d = members.iter().map(|&(_, d)| d).min().unwrap_or(0);
        let max_d = members.iter().map(|&(_, d)| d).max().unwrap_or(0);
        if max_d - min_d > SUB_PERIOD_TOLERANCE_DAYS {
            let mut details_parts: Vec<String> = members
                .iter()
                .map(|&(id, d)| format!("stage {id} ({d}d)"))
                .collect();
            details_parts.sort_unstable();
            let details = details_parts.join(", ");
            let level_note = if season_map.is_multi_resolution() {
                " (season_definitions layers multiple resolution levels; verify \
                  each stage's declared season_id matches its intended level)"
            } else {
                ""
            };
            ctx.emit(
                &rules::SEMANTIC_SEASON_DURATION_SPREAD,
                "stages.json",
                Some(format!("Season {sid}")),
                format!(
                    "stages sharing season_id {sid} have incompatible durations: {details}; \
                     stages within the same season must have the same temporal resolution \
                     (e.g., all monthly or all weekly){level_note}",
                ),
            );
        }
    }

    check_season_observation_coverage(data, season_map, ctx);
    check_season_contiguity(data, season_map, ctx);
}

// ── Shared stage-index lookup (Rules 28, 31) ──────────────────────────────────

/// Stages sorted by id (matches date order; `partition_point` relies on it).
pub(super) fn build_stage_index(
    data: &ParsedData,
) -> Vec<(chrono::NaiveDate, chrono::NaiveDate, usize)> {
    data.stages
        .stages
        .iter()
        .filter_map(|s| s.season_id.map(|sid| (s.start_date, s.end_date, sid)))
        .collect()
}

/// Resolves the stage-index position containing `date`, or `None` if outside every window.
pub(super) fn resolve_stage_position(
    stage_index: &[(chrono::NaiveDate, chrono::NaiveDate, usize)],
    date: chrono::NaiveDate,
) -> Option<usize> {
    let pos = stage_index.partition_point(|(start, _, _)| *start <= date);
    if pos == 0 {
        return None;
    }
    let (_, end_date, _) = stage_index[pos - 1];
    (date < end_date).then_some(pos - 1)
}

/// Looks up the season whose stage window contains `date`.
fn season_id_for_date(
    stage_index: &[(chrono::NaiveDate, chrono::NaiveDate, usize)],
    date: chrono::NaiveDate,
) -> Option<usize> {
    resolve_stage_position(stage_index, date).map(|p| stage_index[p].2)
}

/// Whether PAR estimation runs (must agree with [`crate::scenarios::estimation::EstimationPath::resolve`]).
pub(super) fn estimation_active(data: &ParsedData) -> bool {
    let has_history = !data.inflow_history.is_empty();
    let has_stats = !data.inflow_seasonal_stats.is_empty();
    let has_ar = !data.inflow_ar_coefficients.is_empty();
    has_history && !(has_stats && has_ar)
}

// ── Rule 31: Observation-to-season alignment ──────────────────────────────────

/// Rule 31: warns on finer-than-season observations (auto-aggregated), errors on coarser-than-season (cannot disaggregate).
pub(super) fn check_observation_season_alignment(data: &ParsedData, ctx: &mut ValidationContext) {
    if !estimation_active(data) {
        return;
    }

    let Some(season_map) = &data.stages.policy_graph.season_map else {
        return;
    };

    let stage_index = build_stage_index(data);

    let mut counts: HashMap<(i32, usize, i32), usize> = HashMap::new();
    for row in &data.inflow_history {
        let season_id = season_id_for_date(&stage_index, row.start_date)
            .or_else(|| season_map.season_for_date(row.start_date));

        if let Some(sid) = season_id {
            let year = observation_occurrence_year(season_map, sid, row.start_date);
            *counts.entry((row.hydro_id.0, sid, year)).or_insert(0) += 1;
        }
    }

    let mut finer_violations: Vec<(i32, usize, i32, usize)> = counts
        .iter()
        .filter(|&(_, &n)| n > 1)
        .map(|(&(hid, sid, yr), &n)| (hid, sid, yr, n))
        .collect();
    finer_violations.sort_unstable();
    for (hid, sid, yr, count) in finer_violations {
        ctx.emit(
            &rules::SEMANTIC_HISTORY_FINER_THAN_SEASON,
            "scenarios/inflow_history.parquet",
            Some(format!("Hydro {hid}")),
            format!(
                "hydro {hid} has {count} observations for season {sid} year {yr} \
                 in inflow_history.parquet; these will be aggregated to season \
                 resolution during PAR estimation",
            ),
        );
    }

    let defined_season_ids: Vec<usize> = season_map.seasons.iter().map(|s| s.id).collect();

    let mut hydro_years: HashMap<i32, HashSet<i32>> = HashMap::new();
    for &(hid, _sid, yr) in counts.keys() {
        hydro_years.entry(hid).or_default().insert(yr);
    }

    // Boundary years (each hydro's first and last with any observation) are
    // skipped: partial coverage there is expected (history often starts mid-year),
    // so a missing season is not a coarser-than-season signal. Only interior years
    // are checked; a hydro with <3 years has none.
    let mut coarser_violations: Vec<(i32, usize, i32)> = Vec::new();
    for (&hid, years) in &hydro_years {
        let min_yr = years.iter().copied().min().unwrap_or(0);
        let max_yr = years.iter().copied().max().unwrap_or(0);
        for &yr in years {
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
        ctx.emit(
            &rules::SEMANTIC_HISTORY_COARSER_THAN_SEASON,
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

/// Rule 28: warns when a season has zero observations and inflow scheme is not External.
pub(super) fn check_season_observation_coverage(
    data: &ParsedData,
    season_map: &SeasonMap,
    ctx: &mut ValidationContext,
) {
    use cobre_core::scenario::SamplingScheme;
    use std::path::Path;

    if !estimation_active(data) {
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

    let stage_index = build_stage_index(data);

    let mut season_obs_count: HashMap<usize, usize> = HashMap::new();
    for row in &data.inflow_history {
        if let Some(sid) = season_id_for_date(&stage_index, row.start_date) {
            *season_obs_count.entry(sid).or_insert(0) += 1;
        }
    }

    for season in season_map
        .seasons
        .iter()
        .filter(|s| season_obs_count.get(&s.id).copied().unwrap_or(0) == 0)
    {
        ctx.emit(
            &rules::SEMANTIC_SEASON_WITHOUT_OBSERVATIONS,
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

/// Rule 30: warns when a defined season is unreferenced by any stage.
pub(super) fn check_season_contiguity(
    data: &ParsedData,
    season_map: &SeasonMap,
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
        ctx.emit(
            &rules::SEMANTIC_SEASON_UNREFERENCED,
            "stages.json",
            Some(format!("Season {sid}")),
            format!(
                "season {sid} is defined in season_definitions but not referenced by any \
                 stage; this season will have no PAR parameters",
            ),
        );
    }
}

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
    use super::super::validate_semantic_stages_penalties_scenarios;
    use super::*;
    use crate::test_support::*;
    use crate::{
        scenarios::{InflowArCoefficientRow, InflowHistoryRow, InflowSeasonalStatsRow},
        stages::StagesData,
        validation::{ErrorKind, ValidationContext},
    };
    use cobre_core::{
        EntityId, HorizonGraph,
        temporal::{
            BlockMode, NoiseMethod, PolicyGraphType, ScenarioSourceConfig, SeasonCycleType,
            SeasonDefinition, SeasonMap, Stage, StageRiskConfig, StageStateConfig,
        },
    };

    // ── Local helpers ─────────────────────────────────────────────────────────

    /// Independent oracle for [`resolve_stage_position`]; kept duplicate so a regression cannot corrupt the reference.
    fn resolve_stage_position_reference(
        stage_index: &[(chrono::NaiveDate, chrono::NaiveDate, usize)],
        date: chrono::NaiveDate,
    ) -> Option<usize> {
        let pos = stage_index.partition_point(|(start, _, _)| *start <= date);
        if pos == 0 {
            return None;
        }
        let (_, end_date, _) = stage_index[pos - 1];
        (date < end_date).then_some(pos - 1)
    }

    /// Build a windowed `InflowHistoryRow` covering `[date, date + 1 day)`.
    fn history_row(
        hydro_id: EntityId,
        date: chrono::NaiveDate,
        value_m3s: f64,
    ) -> InflowHistoryRow {
        InflowHistoryRow {
            hydro_id,
            start_date: date,
            end_date: date.succ_opt().unwrap(),
            value_m3s,
        }
    }

    /// Build a `StagesData` for Rule 29 tests.
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
            openings_declared: std::collections::HashSet::new(),
            stages,
            policy_graph: HorizonGraph {
                stage_discount_rate_overrides: std::collections::BTreeMap::new(),
                graph_type: PolicyGraphType::FiniteHorizon,
                annual_discount_rate: 0.06,
                transitions: vec![],
                nodes: Vec::new(),
                season_map: Some(season_map),
            },
        }
    }

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
            openings_declared: std::collections::HashSet::new(),
            stages,
            policy_graph: HorizonGraph {
                stage_discount_rate_overrides: std::collections::BTreeMap::new(),
                graph_type: PolicyGraphType::FiniteHorizon,
                annual_discount_rate: 0.06,
                transitions: vec![],
                nodes: Vec::new(),
                season_map: Some(season_map),
            },
        }
    }

    // ── Rule 27: Season ID range coverage ────────────────────────────────────

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

    // ── Rule 29: Resolution consistency ──────────────────────────────────────

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
            "monthly study with 28-31d stages should produce no rule-29 errors; \
             got: {rule29_errors:?}"
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

    /// Given disjoint monthly (0-11) and quarterly (12-15) season_ids with no
    /// mixing, no errors are emitted by rule 29.
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
            "disjoint monthly (0-11) and quarterly (12-15) season_ids should produce no \
             rule-29 errors; got: {rule29_errors:?}"
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

    /// Given a `Custom` multi-resolution `season_map` (monthly + quarterly
    /// definitions, `d30-multi-resolution-monthly-quarterly`-shaped) where two
    /// stages mistakenly share quarterly `season_id` 12 despite incompatible
    /// durations, the Rule 29 error names the mismatch AND notes the
    /// season_map layers multiple resolution levels.
    #[test]
    fn test_resolution_consistency_multi_resolution_map_notes_layering() {
        use cobre_core::temporal::SeasonCycleType;

        let mut seasons: Vec<SeasonDefinition> = (0..12)
            .map(|i| SeasonDefinition {
                id: i,
                label: format!("Month{}", i + 1),
                month_start: (i as u32) + 1,
                day_start: Some(1),
                month_end: Some((i as u32) + 1),
                day_end: Some(31),
            })
            .collect();
        seasons.push(SeasonDefinition {
            id: 12,
            label: "Q3".to_string(),
            month_start: 7,
            day_start: Some(1),
            month_end: Some(9),
            day_end: Some(30),
        });
        let season_map = SeasonMap {
            cycle_type: SeasonCycleType::Custom,
            seasons,
        };
        assert!(
            season_map.is_multi_resolution(),
            "precondition: Q3 overlaps the monthly definitions"
        );

        let specs = vec![
            (
                0,
                chrono::NaiveDate::from_ymd_opt(2024, 7, 1).unwrap(),
                chrono::NaiveDate::from_ymd_opt(2024, 10, 1).unwrap(),
                12,
            ), // 92d quarterly
            (
                1,
                chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                chrono::NaiveDate::from_ymd_opt(2024, 1, 31).unwrap(),
                12,
            ), // 30d, mistakenly labeled season_id 12
        ];
        let stages: Vec<Stage> = specs
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
        let stages_data = StagesData {
            openings_declared: std::collections::HashSet::new(),
            stages,
            policy_graph: HorizonGraph {
                stage_discount_rate_overrides: std::collections::BTreeMap::new(),
                graph_type: PolicyGraphType::FiniteHorizon,
                annual_discount_rate: 0.06,
                transitions: vec![],
                nodes: Vec::new(),
                season_map: Some(season_map),
            },
        };
        let data = make_data_5b(
            vec![make_hydro_ordered_penalties(1)],
            stages_data,
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
            "expected exactly one rule-29 error for season_id 12; got: {rule29_errors:?}"
        );
        let msg = &rule29_errors[0].message;
        assert!(
            msg.contains("layers multiple resolution levels"),
            "a multi-resolution season_map must be noted in the message; got: {msg}"
        );
    }

    // ── Rule 28: Observation coverage ────────────────────────────────────────

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
            "all seasons with observations should produce no rule-28 warnings; \
             got: {rule28_warnings:?}"
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
        let mut history = Vec::new();
        for i in 0..36usize {
            let month_index = i % 12;
            if month_index == 5 {
                continue; // skip June — season 5 gets no observations
            }
            let year = 2000 + (i / 12) as i32;
            let month = month_index as u32 + 1;
            history.push(history_row(
                EntityId::from(1),
                chrono::NaiveDate::from_ymd_opt(year, month, 15).unwrap(),
                100.0,
            ));
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
            history.push(history_row(
                EntityId::from(1),
                chrono::NaiveDate::from_ymd_opt(year, month, 15).unwrap(),
                100.0,
            ));
        }
        let mut data = make_data_estimation(vec![make_hydro(1, None)], stages, history);
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
            "External inflow scheme should suppress rule-28 warnings; \
             got: {rule28_warnings:?}"
        );
    }

    // ── Rule 30: Contiguity ───────────────────────────────────────────────────

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

    /// Given stages referencing seasons {0, 1, 2, 4, 5} with season 3 defined
    /// but unreferenced, a `ModelQuality` warning is emitted for season 3.
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
            openings_declared: std::collections::HashSet::new(),
            stages: stages_vec,
            policy_graph: HorizonGraph {
                stage_discount_rate_overrides: std::collections::BTreeMap::new(),
                graph_type: PolicyGraphType::FiniteHorizon,
                annual_discount_rate: 0.06,
                transitions: vec![],
                nodes: Vec::new(),
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

    // ── Rule 31: Observation-to-season alignment ──────────────────────────────

    /// Given a monthly study with one observation per hydro per month per year,
    /// no errors and no rule-31 warnings are emitted.
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
            "valid monthly observations should produce no rule-31 errors; \
             got: {rule31_errors:?}"
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
            "valid monthly observations should produce no aggregation warnings; \
             got: {rule31_warnings:?}"
        );
    }

    /// Given a monthly study where hydro 1 has 2 observations for season 0
    /// (January) year 2020, a warning (not an error) is emitted for the
    /// duplicated season.
    #[test]
    fn test_observation_alignment_duplicate_obs() {
        // Season 0 (January 2020) has two observations — finer-than-season.
        let mut history = vec![
            history_row(
                EntityId::from(1),
                chrono::NaiveDate::from_ymd_opt(2020, 1, 5).unwrap(),
                100.0,
            ),
            history_row(
                EntityId::from(1),
                chrono::NaiveDate::from_ymd_opt(2020, 1, 20).unwrap(),
                200.0,
            ),
        ];
        // Add one observation per month for February–December 2020.
        for month in 2u32..=12 {
            history.push(history_row(
                EntityId::from(1),
                chrono::NaiveDate::from_ymd_opt(2020, month, 15).unwrap(),
                f64::from(month) * 10.0,
            ));
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

        // Must produce no errors — finer-than-season is a warning, not an error.
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
        assert_eq!(
            rule31_warnings[0].entity,
            Some("Hydro 1".to_string()),
            "entity context must be 'Hydro 1'; got: {:?}",
            rule31_warnings[0].entity
        );
    }

    /// Given quarterly observations (1 obs per quarter) with a monthly SeasonMap,
    /// a `BusinessRuleViolation` error is emitted for coarser-than-season
    /// observations that cannot be disaggregated.
    ///
    /// The coarser-than-season check only applies to interior years (not the
    /// first or last calendar year a hydro has observations). Three calendar
    /// years are used so 2019 is an interior year. Only 2019 gets the quarterly
    /// (coarse) pattern; boundary years 2018 and 2020 get one full-month
    /// observation each.
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

        // History spans 2018-2020 so that 2019 is an interior year.
        let history = vec![
            // Boundary year 2018 — one observation, not checked.
            history_row(
                EntityId::from(1),
                chrono::NaiveDate::from_ymd_opt(2018, 1, 15).unwrap(),
                50.0,
            ),
            // Interior year 2019 — quarterly (coarse), should trigger error.
            history_row(
                EntityId::from(1),
                chrono::NaiveDate::from_ymd_opt(2019, 2, 15).unwrap(),
                100.0,
            ),
            history_row(
                EntityId::from(1),
                chrono::NaiveDate::from_ymd_opt(2019, 5, 15).unwrap(),
                200.0,
            ),
            history_row(
                EntityId::from(1),
                chrono::NaiveDate::from_ymd_opt(2019, 8, 15).unwrap(),
                300.0,
            ),
            history_row(
                EntityId::from(1),
                chrono::NaiveDate::from_ymd_opt(2019, 11, 15).unwrap(),
                400.0,
            ),
            // Boundary year 2020 — one observation, not checked.
            history_row(
                EntityId::from(1),
                chrono::NaiveDate::from_ymd_opt(2020, 1, 15).unwrap(),
                50.0,
            ),
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
            "coarser-than-season observations must produce at least one \
             BusinessRuleViolation error; got none"
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

    /// Given a study where `season_map` is `None`, rule 31 is skipped entirely.
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

    /// Given estimation inactive (both stats and AR coefficients present),
    /// rule 31 is skipped.
    #[test]
    fn test_observation_alignment_estimation_inactive() {
        let stages = make_stages_with_seasons(12, /*with_season_map=*/ true);
        let history = make_history_rows(1, 12);
        let mut data = make_data_estimation(vec![make_hydro(1, None)], stages, history);
        // Supply both stats and AR coefficients to deactivate estimation.
        data.inflow_seasonal_stats = vec![InflowSeasonalStatsRow {
            hydro_id: EntityId::from(1),
            stage_id: 0,
            mean_m3s: 100.0,
            std_m3s: 10.0,
        }];
        data.inflow_ar_coefficients = vec![InflowArCoefficientRow {
            hydro_id: EntityId::from(1),
            stage_id: 0,
            lag: 1,
            coefficient: 0.5,
        }];

        let mut ctx = ValidationContext::new();
        check_observation_season_alignment(&data, &mut ctx);

        assert!(
            ctx.errors().is_empty(),
            "rule 31 must be skipped when estimation is inactive; got errors: {:?}",
            ctx.errors()
        );
    }

    /// `resolve_stage_position` (the shared helper behind `season_id_for_date`,
    /// `check_season_observation_coverage`, and
    /// `scenarios::check_estimation_prerequisites`) must resolve every history
    /// row's stage position identically to the independent
    /// `resolve_stage_position_reference` oracle; `estimation_active` must
    /// match the inline `has_history && !(has_stats && has_ar)` predicate.
    #[test]
    fn shared_stage_index_helper_matches_pre_share_inline_builds() {
        let stages = make_stages_with_seasons(12, /*with_season_map=*/ true);
        let history = make_history_rows(1, 12);
        let data = make_data_estimation(vec![make_hydro(1, None)], stages, history);

        let has_history = !data.inflow_history.is_empty();
        let has_stats = !data.inflow_seasonal_stats.is_empty();
        let has_ar = !data.inflow_ar_coefficients.is_empty();
        assert_eq!(
            estimation_active(&data),
            has_history && !(has_stats && has_ar),
            "estimation_active must match the inline predicate all three sites computed"
        );
        assert!(
            estimation_active(&data),
            "fixture must have estimation active"
        );

        let stage_index = build_stage_index(&data);
        for row in &data.inflow_history {
            let resolved = resolve_stage_position(&stage_index, row.start_date);
            assert_eq!(
                resolved,
                resolve_stage_position_reference(&stage_index, row.start_date),
                "resolved stage position for {:?} must match the pre-share inline build",
                row.start_date
            );
            assert_eq!(
                season_id_for_date(&stage_index, row.start_date),
                resolved.map(|p| stage_index[p].2),
                "season_id_for_date must derive its result from the shared position"
            );
        }
    }

    /// Given a monthly study where inflow history runs from April 1990 through
    /// September 2020, the boundary years 1990 and 2020 are partial (missing
    /// Jan-Mar 1990 and Oct-Dec 2020 respectively). The coarser-than-season
    /// check must produce no errors for those years.
    #[test]
    fn test_observation_alignment_partial_boundary_years_no_error() {
        let stages = make_stages_with_seasons(12, /*with_season_map=*/ true);

        // Build history: April 1990 – September 2020 (inclusive).
        let mut history: Vec<InflowHistoryRow> = Vec::new();

        // Partial first year: April–December 1990.
        for month in 4u32..=12 {
            history.push(history_row(
                EntityId::from(1),
                chrono::NaiveDate::from_ymd_opt(1990, month, 15).unwrap(),
                100.0,
            ));
        }
        // Complete interior years: 1991 through 2019.
        for year in 1991i32..=2019 {
            for month in 1u32..=12 {
                history.push(history_row(
                    EntityId::from(1),
                    chrono::NaiveDate::from_ymd_opt(year, month, 15).unwrap(),
                    100.0,
                ));
            }
        }
        // Partial last year: January–September 2020.
        for month in 1u32..=9 {
            history.push(history_row(
                EntityId::from(1),
                chrono::NaiveDate::from_ymd_opt(2020, month, 15).unwrap(),
                100.0,
            ));
        }

        let data = make_data_estimation(vec![make_hydro(1, None)], stages, history);

        let mut ctx = ValidationContext::new();
        check_observation_season_alignment(&data, &mut ctx);

        let coarser_errors: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| {
                e.kind == ErrorKind::BusinessRuleViolation
                    && e.message.contains("coarser-than-season")
            })
            .collect();
        assert!(
            coarser_errors.is_empty(),
            "partial boundary years must not produce coarser-than-season errors; \
             got: {coarser_errors:?}"
        );
    }

    /// Given a monthly study (12 seasons) where inflow history spans complete
    /// calendar years (1991-2019) but season 6 (July) is missing for the
    /// interior year 2005, a coarser-than-season error IS produced for that
    /// missing `(hydro, season, year)` group.
    #[test]
    fn test_observation_alignment_missing_interior_season_produces_error() {
        let stages = make_stages_with_seasons(12, /*with_season_map=*/ true);

        // Build history: 1991-2019, all months, except July 2005 (season 6).
        let mut history: Vec<InflowHistoryRow> = Vec::new();
        for year in 1991i32..=2019 {
            for month in 1u32..=12 {
                // Skip July 2005 to simulate a missing interior-year season.
                if year == 2005 && month == 7 {
                    continue;
                }
                history.push(history_row(
                    EntityId::from(1),
                    chrono::NaiveDate::from_ymd_opt(year, month, 15).unwrap(),
                    100.0,
                ));
            }
        }

        let data = make_data_estimation(vec![make_hydro(1, None)], stages, history);

        let mut ctx = ValidationContext::new();
        check_observation_season_alignment(&data, &mut ctx);

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
            "a missing season in an interior year must produce a coarser-than-season error"
        );
        // The error must specifically reference hydro 1, season 6, year 2005.
        let target = coarser_errors.iter().find(|e| {
            e.message.contains("hydro 1")
                && e.message.contains("season 6")
                && e.message.contains("year 2005")
        });
        assert!(
            target.is_some(),
            "expected an error for hydro 1 season 6 year 2005; got: {coarser_errors:?}"
        );
    }

    #[test]
    fn rule_31_counts_weekly_history_by_iso_week_numbering_year() {
        use cobre_stochastic::test_support::weekly_season_map;

        let mut stages = make_stages_with_seasons(0, /*with_season_map=*/ false);
        stages.policy_graph.season_map = Some(weekly_season_map());
        let last_monday = chrono::NaiveDate::from_ymd_opt(2017, 12, 25).unwrap();
        let history: Vec<InflowHistoryRow> = chrono::NaiveDate::from_ymd_opt(2013, 12, 30)
            .unwrap()
            .iter_weeks()
            .take_while(|monday| *monday <= last_monday)
            .map(|monday| history_row(EntityId::from(1), monday, 100.0))
            .collect();
        let data = make_data_estimation(vec![make_hydro(1, None)], stages, history);

        let mut ctx = ValidationContext::new();
        check_observation_season_alignment(&data, &mut ctx);

        let missing_week_1: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.message.contains("season 0 year 2015"))
            .collect();
        assert!(
            missing_week_1.is_empty(),
            "ISO 2015-W01 starts on 2014-12-29 and must count toward 2015; got: {missing_week_1:?}"
        );
        let week_53_fold: Vec<_> = ctx
            .warnings()
            .into_iter()
            .filter(|w| {
                w.message
                    .contains("has 2 observations for season 51 year 2015")
            })
            .collect();
        assert_eq!(
            week_53_fold.len(),
            1,
            "ISO 2015-W52 and W53 must share one season-51 bucket; got: {:?}",
            ctx.warnings()
        );
    }
}
