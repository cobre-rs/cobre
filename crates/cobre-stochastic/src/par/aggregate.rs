//! Observation aggregation from fine-grained to coarser season resolution.
//!
//! When the season cycle is coarser than the observation history (e.g.
//! quarterly seasons with monthly observations), each `(entity, season, year)`
//! group is collapsed by duration-weighted averaging before PAR fitting. The
//! day-count weights preserve volumetric average flow rate across unequal month
//! lengths:
//!
//! ```text
//! agg_value = sum(value_i * days_i) / sum(days_i)
//! ```

use std::collections::HashMap;

use chrono::{Datelike, Months, NaiveDate};
use cobre_core::{
    EntityId,
    temporal::{SeasonMap, Stage},
};

use crate::StochasticError;
use crate::par::fitting::find_season_for_date;
use crate::season_cast::observation_occurrence_year;

/// Aggregate fine-grained observations into one duration-weighted observation
/// per `(entity, season, year)` group; each group inherits its earliest date.
///
/// The year is the season occurrence's ([`observation_occurrence_year`]), so a
/// group is one occurrence: an ISO week-numbering year under a `Weekly` map, and
/// one occurrence for a `Custom` season that spans 1 January. Under a `Weekly`
/// map ISO week 53 carries season 51, so a 53-week year contributes one season-51
/// value averaged over weeks 52 and 53 (week 53 is folded for fitting).
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

    let mut stage_index: Vec<(NaiveDate, NaiveDate, i32, usize)> = stages
        .iter()
        .filter_map(|s| s.season_id.map(|sid| (s.start_date, s.end_date, s.id, sid)))
        .collect();
    stage_index.sort_unstable_by_key(|(start, _, _, _)| *start);

    let mut group_map: HashMap<(EntityId, usize, i32), Vec<(NaiveDate, f64)>> =
        HashMap::with_capacity(observations.len());

    for &(entity_id, date, value) in observations {
        // Stage index first, then SeasonMap calendar fallback for out-of-study
        // historical dates.
        let season_id = find_season_for_date(&stage_index, date)
            .or_else(|| season_map.season_for_date(date))
            .ok_or_else(|| StochasticError::InsufficientData {
                context: format!(
                    "observation date {date} for entity {entity_id} \
                     does not match any stage date range or season definition"
                ),
            })?;

        let year = observation_occurrence_year(season_map, season_id, date);
        group_map
            .entry((entity_id, season_id, year))
            .or_default()
            .push((date, value));
    }

    let mut result: Vec<(EntityId, NaiveDate, f64)> = Vec::with_capacity(group_map.len());

    for ((entity_id, _season_id, _year), mut entries) in group_map {
        // Sort by date so entries[0] is the representative (earliest) date.
        entries.sort_unstable_by_key(|(d, _)| *d);

        if entries.len() == 1 {
            let (date, value) = entries[0];
            result.push((entity_id, date, value));
        } else {
            let mut weighted_sum = 0.0_f64;
            let mut total_days = 0_u32;

            for (date, value) in &entries {
                let days = days_in_month(*date);
                weighted_sum += value * f64::from(days);
                total_days += days;
            }

            // total_days ≤ ~372 fits exactly in f64; cast cannot lose precision.
            #[allow(clippy::cast_precision_loss)]
            let agg_value = weighted_sum / f64::from(total_days);

            let rep_date = entries[0].0;
            result.push((entity_id, rep_date, agg_value));
        }
    }

    // Sort by (entity_id, date) to match parser convention (declaration-order
    // invariance: output must not depend on group_map iteration order).
    result.sort_unstable_by_key(|(eid, date, _)| (eid.0, *date));

    Ok(result)
}

/// Return the number of calendar days in the month containing `date`
/// (leap-year aware).
fn days_in_month(date: NaiveDate) -> u32 {
    let first_of_month = NaiveDate::from_ymd_opt(date.year(), date.month(), 1).unwrap_or(date);
    let first_of_next = first_of_month
        .checked_add_months(Months::new(1))
        .unwrap_or(first_of_month);
    let diff = first_of_next.signed_duration_since(first_of_month);
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
        EntityId,
        temporal::{SeasonCycleType, SeasonDefinition, SeasonMap, Stage},
        test_support::{StageSpec, date, f64_bits_eq, single_block},
    };

    use super::aggregate_observations_to_season;
    use crate::StochasticError;
    use crate::test_support::{MonthlyLabels, monthly_season_map, weekly_season_map};

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
        cobre_core::test_support::make_stage(StageSpec {
            id,
            index: Some(index),
            start_date: date(year_start, month_start, 1),
            end_date: date(year_end, month_end, 1),
            season_id,
            blocks: single_block("SINGLE", 720.0),
            ..Default::default()
        })
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
        let season_map = monthly_season_map(MonthlyLabels::ZeroPadded);
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

    fn daily_observations(first: NaiveDate, last: NaiveDate) -> Vec<(EntityId, NaiveDate, f64)> {
        first
            .iter_days()
            .take_while(|day| *day <= last)
            .zip(0_u32..)
            .map(|(day, i)| (EntityId::from(1), day, f64::from(i)))
            .collect()
    }

    fn mean_over_days(
        observations: &[(EntityId, NaiveDate, f64)],
        first: NaiveDate,
        last: NaiveDate,
    ) -> f64 {
        let values: Vec<f64> = observations
            .iter()
            .filter(|(_, day, _)| (first..=last).contains(day))
            .map(|&(_, _, value)| value)
            .collect();
        values.iter().sum::<f64>() / f64::from(u32::try_from(values.len()).unwrap())
    }

    #[test]
    fn weekly_history_is_grouped_by_iso_week_numbering_year() {
        let season_map = weekly_season_map();
        let observations = daily_observations(date(2013, 12, 30), date(2015, 1, 4));

        let result = aggregate_observations_to_season(&observations, &[], &season_map).unwrap();

        let season_0: Vec<(NaiveDate, f64)> = result
            .iter()
            .filter(|(_, day, _)| season_map.season_for_date(*day) == Some(0))
            .map(|&(_, day, value)| (day, value))
            .collect();
        assert_eq!(
            season_0.iter().map(|&(day, _)| day).collect::<Vec<_>>(),
            vec![date(2013, 12, 30), date(2014, 12, 29)],
            "one season-0 value per ISO week-numbering year: {season_0:?}"
        );
        let expected = [
            mean_over_days(&observations, date(2013, 12, 30), date(2014, 1, 5)),
            mean_over_days(&observations, date(2014, 12, 29), date(2015, 1, 4)),
        ];
        for (&(day, got), want) in season_0.iter().zip(expected) {
            assert!(f64_bits_eq(got, want), "{day}: expected {want}, got {got}");
        }
    }

    #[test]
    fn iso_week_53_is_folded_into_the_last_weekly_season() {
        let season_map = weekly_season_map();
        let observations = daily_observations(date(2015, 12, 21), date(2016, 1, 3));

        let result = aggregate_observations_to_season(&observations, &[], &season_map).unwrap();

        assert_eq!(
            result.len(),
            1,
            "weeks 52 and 53 of ISO 2015 form one value: {result:?}"
        );
        let (_, day, got) = result[0];
        assert_eq!(day, date(2015, 12, 21));
        let want = mean_over_days(&observations, date(2015, 12, 21), date(2016, 1, 3));
        assert!(f64_bits_eq(got, want), "expected {want}, got {got}");
    }

    #[test]
    fn a_custom_season_spanning_new_year_is_grouped_as_one_occurrence() {
        let season = |id: usize, month_start: u32, month_end: u32| SeasonDefinition {
            id,
            label: format!("S{id}"),
            month_start,
            day_start: None,
            month_end: Some(month_end),
            day_end: None,
        };
        let season_map = SeasonMap {
            cycle_type: SeasonCycleType::Custom,
            seasons: vec![
                season(0, 3, 5),
                season(1, 6, 8),
                season(2, 9, 11),
                season(3, 12, 2),
            ],
        };
        let (a, b, c) = (110.3, 230.7, 170.9);
        let entity = EntityId::from(1);
        let observations = vec![
            (entity, date(2020, 12, 1), a),
            (entity, date(2021, 1, 1), b),
            (entity, date(2021, 2, 1), c),
        ];

        let result = aggregate_observations_to_season(&observations, &[], &season_map).unwrap();

        assert_eq!(
            result.len(),
            1,
            "the Dec 2020 - Feb 2021 occurrence forms one value: {result:?}"
        );
        let (_, day, got) = result[0];
        assert_eq!(day, date(2020, 12, 1));
        let want = (a * 31.0 + b * 31.0 + c * 28.0) / 90.0;
        assert!(f64_bits_eq(got, want), "expected {want}, got {got}");
    }
}
