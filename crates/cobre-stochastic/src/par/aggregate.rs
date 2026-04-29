//! Observation aggregation from fine-grained to coarser season resolution.
//!
//! When the season resolution of a study is coarser than the resolution of
//! the available observation history (e.g., quarterly seasons with monthly
//! observations), each `(entity, season, year)` group contains multiple
//! observations that must be collapsed into a single representative value
//! before passing them to the PAR fitting pipeline.
//!
//! This module provides [`aggregate_observations_to_season`], which performs
//! duration-weighted averaging over each group. The formula preserves the
//! physical meaning of volumetric average flow rate across unequal month
//! lengths within a coarser period:
//!
//! ```text
//! agg_value = sum(value_i * days_i) / sum(days_i)
//! ```
//!
//! where `days_i` is the number of calendar days in the observation's month.
//!
//! Groups with exactly one observation pass through unchanged (identity case).

use std::collections::HashMap;

use chrono::{Datelike, Months, NaiveDate};
use cobre_core::{
    temporal::{SeasonMap, Stage},
    EntityId,
};

use crate::par::fitting::find_season_for_date;
use crate::StochasticError;

/// Aggregate fine-grained observations into one observation per
/// `(entity, season, year)` group using duration-weighted averaging.
///
/// # Purpose
///
/// When the season cycle of a study is coarser than the observation
/// frequency (e.g., quarterly seasons derived from monthly inflow history),
/// calling the PAR fitting functions directly would result in multiple
/// observations per group, distorting the estimated parameters. This
/// function collapses each group into a single value before fitting.
///
/// # Date-to-season resolution
///
/// For each observation date the function first attempts to locate a
/// matching stage in `stages` via binary search (`find_season_for_date`).
/// If no stage covers the date, it falls back to a calendar-based lookup
/// via `season_map.season_for_date`. This two-tier strategy handles both
/// in-study and pre-study historical observations.
///
/// # Duration weights
///
/// The weight for each observation is the number of calendar days in its
/// month (e.g., January = 31, February = 28 or 29 in a leap year). This
/// preserves volumetric correctness when averaging mean flow rates measured
/// over periods of different lengths.
///
/// # Representative date
///
/// The aggregated observation inherits the earliest (chronologically first)
/// date within each group.
///
/// # Errors
///
/// Returns [`StochasticError::InsufficientData`] if any observation date
/// cannot be resolved to a season via either the stage index or the
/// `SeasonMap`.
///
/// # Examples
///
/// ```
/// use chrono::NaiveDate;
/// use cobre_core::{EntityId, temporal::{SeasonMap, SeasonCycleType, SeasonDefinition, Stage}};
/// use cobre_stochastic::par::aggregate_observations_to_season;
///
/// // Quarterly SeasonMap: season 0 spans Jan–Mar.
/// let season_map = SeasonMap {
///     cycle_type: SeasonCycleType::Custom,
///     seasons: vec![SeasonDefinition {
///         id: 0,
///         label: "Q1".to_string(),
///         month_start: 1,
///         day_start: Some(1),
///         month_end: Some(3),
///         day_end: Some(31),
///     }],
/// };
///
/// let entity = EntityId::from(1);
/// let observations = vec![
///     (entity, NaiveDate::from_ymd_opt(2020, 1, 15).unwrap(), 100.0),
///     (entity, NaiveDate::from_ymd_opt(2020, 2, 15).unwrap(), 200.0),
///     (entity, NaiveDate::from_ymd_opt(2020, 3, 15).unwrap(), 300.0),
/// ];
///
/// let result = aggregate_observations_to_season(&observations, &[], &season_map).unwrap();
/// assert_eq!(result.len(), 1);
/// ```
pub fn aggregate_observations_to_season(
    observations: &[(EntityId, NaiveDate, f64)],
    stages: &[Stage],
    season_map: &SeasonMap,
) -> Result<Vec<(EntityId, NaiveDate, f64)>, StochasticError> {
    if observations.is_empty() {
        return Ok(Vec::new());
    }

    // Build stage index sorted by start_date for binary-search based range
    // lookup. Only stages with a season_id contribute.
    let mut stage_index: Vec<(NaiveDate, NaiveDate, i32, usize)> = stages
        .iter()
        .filter_map(|s| s.season_id.map(|sid| (s.start_date, s.end_date, s.id, sid)))
        .collect();
    stage_index.sort_unstable_by_key(|(start, _, _, _)| *start);

    // Group observations by (entity_id, season_id, year).
    // Value: Vec<(date, value)> — dates kept for choosing the representative.
    let mut group_map: HashMap<(EntityId, usize, i32), Vec<(NaiveDate, f64)>> =
        HashMap::with_capacity(observations.len());

    for &(entity_id, date, value) in observations {
        // Two-tier date-to-season resolution: stage index first, then
        // SeasonMap calendar mapping for out-of-range historical observations.
        let season_id = find_season_for_date(&stage_index, date)
            .or_else(|| season_map.season_for_date(date))
            .ok_or_else(|| StochasticError::InsufficientData {
                context: format!(
                    "observation date {date} for entity {entity_id} \
                     does not match any stage date range or season definition"
                ),
            })?;

        let year = date.year();
        group_map
            .entry((entity_id, season_id, year))
            .or_default()
            .push((date, value));
    }

    // For each group compute the duration-weighted average and pick the
    // earliest date as the representative.
    let mut result: Vec<(EntityId, NaiveDate, f64)> = Vec::with_capacity(group_map.len());

    for ((entity_id, _season_id, _year), mut entries) in group_map {
        // Sort entries by date so the minimum date is entries[0].
        entries.sort_unstable_by_key(|(d, _)| *d);

        if entries.len() == 1 {
            // Identity case: pass through unchanged.
            let (date, value) = entries[0];
            result.push((entity_id, date, value));
        } else {
            // Duration-weighted average.
            // Weight for each observation = number of calendar days in its month.
            let mut weighted_sum = 0.0_f64;
            let mut total_days = 0_u32;

            for (date, value) in &entries {
                let days = days_in_month(*date);
                weighted_sum += value * f64::from(days);
                total_days += days;
            }

            // total_days is always >= 28 per entry, so > 0. The cast to f64
            // is safe: total_days fits well within the exact-integer range of
            // f64 (max ~12 months * 31 days = 372, far below 2^53).
            #[allow(clippy::cast_precision_loss)]
            let agg_value = weighted_sum / f64::from(total_days);

            // Representative date is the earliest (entries[0] after sort).
            let rep_date = entries[0].0;
            result.push((entity_id, rep_date, agg_value));
        }
    }

    // Sort by (entity_id, date) ascending to match parser convention.
    result.sort_unstable_by_key(|(eid, date, _)| (eid.0, *date));

    Ok(result)
}

/// Return the number of calendar days in the month containing `date`.
///
/// Uses `chrono::Months` arithmetic to compute the first day of the next
/// month and then takes the difference. This handles February in both leap
/// years (29 days) and non-leap years (28 days) correctly.
fn days_in_month(date: NaiveDate) -> u32 {
    let first_of_month = NaiveDate::from_ymd_opt(date.year(), date.month(), 1).unwrap_or(date);
    let first_of_next = first_of_month
        .checked_add_months(Months::new(1))
        .unwrap_or(first_of_month);
    let diff = first_of_next.signed_duration_since(first_of_month);
    // Duration is always in [28, 31] days — cast to u32 is safe.
    u32::try_from(diff.num_days()).unwrap_or(30)
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::float_cmp, clippy::panic)]
mod tests {
    use chrono::{Datelike, NaiveDate};
    use cobre_core::{
        temporal::{
            Block, BlockMode, NoiseMethod, ScenarioSourceConfig, SeasonCycleType, SeasonDefinition,
            SeasonMap, Stage, StageRiskConfig, StageStateConfig,
        },
        EntityId,
    };

    use super::aggregate_observations_to_season;
    use crate::StochasticError;

    // -----------------------------------------------------------------------
    // Helper constructors
    // -----------------------------------------------------------------------

    fn make_stage(
        id: i32,
        index: usize,
        year_start: i32,
        month_start: u32,
        year_end: i32,
        month_end: u32,
        season_id: Option<usize>,
    ) -> Stage {
        Stage {
            index,
            id,
            start_date: NaiveDate::from_ymd_opt(year_start, month_start, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(year_end, month_end, 1).unwrap(),
            season_id,
            blocks: vec![Block {
                index: 0,
                name: "SINGLE".to_string(),
                duration_hours: 720.0,
            }],
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

    /// Build quarterly stages for `n_years` starting at `base_year`.
    /// Season IDs: 0 = Q1 (Jan–Mar), 1 = Q2 (Apr–Jun), 2 = Q3 (Jul–Sep), 3 = Q4 (Oct–Dec).
    fn make_quarterly_stages(base_year: i32, n_years: u32) -> Vec<Stage> {
        // (start_month, end_month_exclusive)
        let quarters = [(1u32, 4u32), (4, 7), (7, 10), (10, 1)];
        let mut stages: Vec<Stage> = Vec::new();
        for year_offset in 0..n_years {
            let year = base_year + i32::try_from(year_offset).unwrap_or(0);
            for (qidx, &(m_start, m_end_excl)) in quarters.iter().enumerate() {
                let (end_year, end_month) = if m_end_excl == 1 {
                    (year + 1, 1u32)
                } else {
                    (year, m_end_excl)
                };
                let stage_id =
                    i32::try_from(year_offset * 4 + u32::try_from(qidx).unwrap_or(0) + 1)
                        .unwrap_or(0);
                stages.push(make_stage(
                    stage_id,
                    stages.len(),
                    year,
                    m_start,
                    end_year,
                    end_month,
                    Some(qidx),
                ));
            }
        }
        stages
    }

    /// Build a quarterly `SeasonMap` with 4 Custom seasons (Q1–Q4).
    fn make_quarterly_season_map() -> SeasonMap {
        SeasonMap {
            cycle_type: SeasonCycleType::Custom,
            seasons: vec![
                SeasonDefinition {
                    id: 0,
                    label: "Q1".to_string(),
                    month_start: 1,
                    day_start: Some(1),
                    month_end: Some(3),
                    day_end: Some(31),
                },
                SeasonDefinition {
                    id: 1,
                    label: "Q2".to_string(),
                    month_start: 4,
                    day_start: Some(1),
                    month_end: Some(6),
                    day_end: Some(30),
                },
                SeasonDefinition {
                    id: 2,
                    label: "Q3".to_string(),
                    month_start: 7,
                    day_start: Some(1),
                    month_end: Some(9),
                    day_end: Some(30),
                },
                SeasonDefinition {
                    id: 3,
                    label: "Q4".to_string(),
                    month_start: 10,
                    day_start: Some(1),
                    month_end: Some(12),
                    day_end: Some(31),
                },
            ],
        }
    }

    /// Build a monthly `SeasonMap` (12 seasons, `Monthly` cycle).
    fn make_monthly_season_map() -> SeasonMap {
        let seasons = (1u32..=12)
            .map(|m| SeasonDefinition {
                id: (m - 1) as usize,
                label: format!("Month{m:02}"),
                month_start: m,
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

    fn obs(entity_id: i32, year: i32, month: u32, value: f64) -> (EntityId, NaiveDate, f64) {
        (
            EntityId::from(entity_id),
            NaiveDate::from_ymd_opt(year, month, 15).unwrap(),
            value,
        )
    }

    // -----------------------------------------------------------------------
    // Test 1: quarterly aggregation for a single entity (Jan + Feb + Mar 2020)
    // -----------------------------------------------------------------------

    #[test]
    fn test_quarterly_aggregation_single_entity() {
        // 2020 is a leap year, so February has 29 days.
        let v_jan = 100.0_f64;
        let v_feb = 200.0_f64;
        let v_mar = 300.0_f64;

        let observations = vec![
            obs(1, 2020, 1, v_jan),
            obs(1, 2020, 2, v_feb),
            obs(1, 2020, 3, v_mar),
        ];

        let stages = make_quarterly_stages(2020, 1);
        let season_map = make_quarterly_season_map();

        let result = aggregate_observations_to_season(&observations, &stages, &season_map).unwrap();

        assert_eq!(
            result.len(),
            1,
            "expected 1 aggregated observation for Q1 2020"
        );

        let (entity_id, date, value) = result[0];
        assert_eq!(entity_id, EntityId::from(1));
        // Representative date = Jan 15 (earliest in the group).
        assert_eq!(date, NaiveDate::from_ymd_opt(2020, 1, 15).unwrap());

        // Duration-weighted average: (100*31 + 200*29 + 300*31) / (31+29+31)
        let expected = (v_jan * 31.0 + v_feb * 29.0 + v_mar * 31.0) / (31.0 + 29.0 + 31.0);
        assert!(
            (value - expected).abs() < 1e-10,
            "expected {expected}, got {value}"
        );
    }

    // -----------------------------------------------------------------------
    // Test 2: identity case — monthly observations with monthly SeasonMap
    // -----------------------------------------------------------------------

    #[test]
    fn test_identity_case_monthly_obs_monthly_seasons() {
        // 12 monthly observations, one per month in 2020.
        let season_map = make_monthly_season_map();
        // For the monthly identity case we only need the SeasonMap (no stages).
        let observations: Vec<(EntityId, NaiveDate, f64)> = (1u32..=12)
            .map(|m| obs(1, 2020, m, f64::from(m) * 10.0))
            .collect();

        let result = aggregate_observations_to_season(&observations, &[], &season_map).unwrap();

        assert_eq!(
            result.len(),
            12,
            "identity case must produce same number of observations"
        );

        // Values must be identical (ordering by date may differ so build a map).
        let result_map: std::collections::HashMap<NaiveDate, f64> =
            result.iter().map(|&(_, d, v)| (d, v)).collect();

        for &(_, date, value) in &observations {
            let got = *result_map.get(&date).expect("date present in result");
            assert!(
                (got - value).abs() < 1e-10,
                "value mismatch for date {date}: expected {value}, got {got}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Test 3: multi-entity — 2 entities x 4 quarters x 3 months = 24 obs -> 8
    // -----------------------------------------------------------------------

    #[test]
    fn test_multi_entity_two_entities_four_quarters() {
        let stages = make_quarterly_stages(2020, 1);
        let season_map = make_quarterly_season_map();

        let mut observations: Vec<(EntityId, NaiveDate, f64)> = Vec::new();
        // Entity 1 and entity 2, all 12 months of 2020.
        for entity_id in [1, 2] {
            for month in 1u32..=12 {
                observations.push(obs(entity_id, 2020, month, f64::from(month)));
            }
        }

        let result = aggregate_observations_to_season(&observations, &stages, &season_map).unwrap();

        assert_eq!(
            result.len(),
            8,
            "expected 2 entities x 4 quarters = 8 observations"
        );

        // Verify sorted by (entity_id, date).
        let mut prev_key: Option<(i32, NaiveDate)> = None;
        for &(eid, date, _) in &result {
            let key = (eid.0, date);
            if let Some(pk) = prev_key {
                assert!(key >= pk, "result not sorted: {pk:?} followed by {key:?}");
            }
            prev_key = Some(key);
        }

        // Entity 1 and entity 2 must each appear exactly 4 times.
        let count_e1 = result.iter().filter(|&&(eid, _, _)| eid.0 == 1).count();
        let count_e2 = result.iter().filter(|&&(eid, _, _)| eid.0 == 2).count();
        assert_eq!(count_e1, 4);
        assert_eq!(count_e2, 4);
    }

    // -----------------------------------------------------------------------
    // Test 4: multi-year — 1 entity x 2 years x 4 quarters -> 8 observations
    // -----------------------------------------------------------------------

    #[test]
    fn test_multi_year_two_years_four_quarters() {
        let stages = make_quarterly_stages(2020, 2);
        let season_map = make_quarterly_season_map();

        let mut observations: Vec<(EntityId, NaiveDate, f64)> = Vec::new();
        for year in [2020, 2021] {
            for month in 1u32..=12 {
                observations.push(obs(1, year, month, f64::from(month)));
            }
        }

        let result = aggregate_observations_to_season(&observations, &stages, &season_map).unwrap();

        assert_eq!(
            result.len(),
            8,
            "expected 1 entity x 2 years x 4 quarters = 8 observations"
        );

        // Each year must contribute exactly 4 observations.
        let count_2020 = result.iter().filter(|&&(_, d, _)| d.year() == 2020).count();
        let count_2021 = result.iter().filter(|&&(_, d, _)| d.year() == 2021).count();
        assert_eq!(count_2020, 4, "expected 4 observations for 2020");
        assert_eq!(count_2021, 4, "expected 4 observations for 2021");
    }

    // -----------------------------------------------------------------------
    // Test 5: SeasonMap fallback for out-of-stage observation
    // -----------------------------------------------------------------------

    #[test]
    fn test_season_map_fallback_for_out_of_range_date() {
        // Stages cover 2020 only; observation is from 1990 (out of range).
        let stages = make_quarterly_stages(2020, 1);
        let season_map = make_quarterly_season_map();

        // 1990-06-15 is in Q2 (April–June) per the SeasonMap calendar definition.
        let observations = vec![obs(1, 1990, 6, 42.0)];

        let result = aggregate_observations_to_season(&observations, &stages, &season_map).unwrap();

        assert_eq!(
            result.len(),
            1,
            "out-of-range observation must be processed via SeasonMap fallback"
        );
        let (_, _, value) = result[0];
        assert!(
            (value - 42.0).abs() < 1e-10,
            "value must pass through unchanged for single-observation group"
        );
    }

    // -----------------------------------------------------------------------
    // Test 6: unresolvable date returns InsufficientData error
    // -----------------------------------------------------------------------

    #[test]
    fn test_unresolvable_date_returns_error() {
        // Custom SeasonMap with only Q1 (Jan–Mar); 1800-06-15 maps to nothing.
        let season_map = SeasonMap {
            cycle_type: SeasonCycleType::Custom,
            seasons: vec![SeasonDefinition {
                id: 0,
                label: "Q1".to_string(),
                month_start: 1,
                day_start: Some(1),
                month_end: Some(3),
                day_end: Some(31),
            }],
        };

        // No stages; the SeasonMap only covers Jan–Mar.
        let observations = vec![(
            EntityId::from(1),
            NaiveDate::from_ymd_opt(1800, 6, 15).unwrap(),
            99.0,
        )];

        let err = aggregate_observations_to_season(&observations, &[], &season_map)
            .expect_err("expected InsufficientData for unresolvable date");

        assert!(
            matches!(err, StochasticError::InsufficientData { .. }),
            "error variant must be InsufficientData, got: {err:?}"
        );
        // The error message must contain the unresolvable date.
        let msg = err.to_string();
        assert!(
            msg.contains("1800-06-15"),
            "error message must contain the unresolvable date; got: {msg}"
        );
    }

    // -----------------------------------------------------------------------
    // Test 7: leap year February weight differs from non-leap year
    // -----------------------------------------------------------------------

    #[test]
    fn test_leap_year_february_weight() {
        // Q1 of 2020 (leap year: Feb has 29 days) vs Q1 of 2021 (non-leap: 28 days).
        let stages = make_quarterly_stages(2020, 2);
        let season_map = make_quarterly_season_map();

        // Same values for all three months, same for both years.
        // If weights differ, the weighted averages differ due to Feb days.
        let v = 60.0_f64;
        let observations = vec![
            // 2020 Q1
            obs(1, 2020, 1, v),
            obs(1, 2020, 2, v),
            obs(1, 2020, 3, v),
            // 2021 Q1
            obs(1, 2021, 1, v),
            obs(1, 2021, 2, v),
            obs(1, 2021, 3, v),
        ];

        let result = aggregate_observations_to_season(&observations, &stages, &season_map).unwrap();

        assert_eq!(
            result.len(),
            2,
            "expected 2 aggregated observations (Q1 2020 and Q1 2021)"
        );

        // When all values are equal, the weighted average equals v regardless
        // of the weights. This test verifies correctness without explicit weight
        // assertion, and separately tests that the days_in_month function
        // handles leap years correctly by invoking it directly.
        for &(_, _, value) in &result {
            assert!(
                (value - v).abs() < 1e-10,
                "expected {v}, got {value} (constant values must survive duration-weighting)"
            );
        }

        // Directly verify the internal days_in_month calculation.
        // Feb 2020 (leap year) = 29 days; Feb 2021 (non-leap) = 28 days.
        let feb_2020 = NaiveDate::from_ymd_opt(2020, 2, 15).unwrap();
        let feb_2021 = NaiveDate::from_ymd_opt(2021, 2, 15).unwrap();
        assert_eq!(
            super::days_in_month(feb_2020),
            29,
            "Feb 2020 must have 29 days (leap year)"
        );
        assert_eq!(
            super::days_in_month(feb_2021),
            28,
            "Feb 2021 must have 28 days (non-leap year)"
        );
    }
}
