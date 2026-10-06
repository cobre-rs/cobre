//! Historical window discovery algorithm.
//!
//! A "window" is a starting year `y` such that every hydro in the study has a
//! historical observation for every study season, the observations
//! standardization reads. Observations align to study stages by `season_id`
//! matching, not by raw calendar arithmetic.
//!
//! `build_observation_sequence` owns the `(year_offset, season_id)` layout the
//! window year is resolved against. `y` is the first study observation's year.

use std::collections::HashSet;

use chrono::{Datelike, NaiveDate};
use cobre_core::{
    EntityId,
    scenario::{HistoricalYears, InflowHistoryRow},
    temporal::{SeasonMap, Stage},
};

use crate::{
    StochasticError, par::fitting::find_season_for_date, season_cast::observation_occurrence_year,
};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Discover the set of valid historical window starting years.
///
/// A window starting year `y` is **valid** when every hydro in `hydro_ids`
/// has a historical observation for every study entry
/// `(y + year_offset, season_id)` that `build_observation_sequence` emits.
///
/// A `Some` `user_pool` restricts the result to that pool's expanded years.
/// The `month0()` season fallback applies only when `season_map` is `None`; a
/// `Some` map that cannot resolve a date drops the row from the lookup.
///
/// # Errors
///
/// Returns [`StochasticError::InsufficientData`] when no valid windows are
/// found after applying the user pool filter.
///
/// # Examples
///
/// ```
/// use chrono::NaiveDate;
/// use cobre_core::{EntityId, scenario::InflowHistoryRow, temporal::Stage};
/// use cobre_stochastic::sampling::discover_historical_windows;
///
/// // Build a minimal monthly history for one hydro, 1990-01 through 1991-12.
/// let hydro_id = EntityId(1);
/// let history: Vec<InflowHistoryRow> = (1990_i32..=1991)
///     .flat_map(|y| {
///         (1u32..=12).map(move |m| {
///             let start_date = NaiveDate::from_ymd_opt(y, m, 1).unwrap();
///             InflowHistoryRow {
///                 hydro_id,
///                 start_date,
///                 end_date: start_date.checked_add_months(chrono::Months::new(1)).unwrap(),
///                 value_m3s: 100.0,
///             }
///         })
///     })
///     .collect();
///
/// let stages: Vec<Stage> = (0_usize..12)
///     .map(|i| {
///         use cobre_core::temporal::{Block, BlockMode, NoiseMethod, ScenarioSourceConfig,
///             StageRiskConfig, StageStateConfig};
///         Stage {
///             index: i,
///             id: i as i32,
///             start_date: NaiveDate::from_ymd_opt(1990, (i as u32 % 12) + 1, 1).unwrap(),
///             end_date: NaiveDate::from_ymd_opt(1990, (i as u32 % 12) + 1, 28).unwrap(),
///             season_id: Some(i),
///             blocks: vec![Block { index: 0, name: "SINGLE".into(), duration_hours: 720.0 }],
///             block_mode: BlockMode::Parallel,
///             state_config: StageStateConfig { storage: true, inflow_lags: false },
///             risk_config: StageRiskConfig::Expectation,
///             scenario_config: ScenarioSourceConfig {
///                 branching_factor: 1,
///                 noise_method: NoiseMethod::Saa,
///             },
///         }
///     })
///     .collect();
///
/// let windows = discover_historical_windows(&history, &[hydro_id], &stages, None, None, 10)
///     .unwrap();
///
/// // Each year holds all twelve study seasons; no pre-window lag is read, so
/// // 1990 qualifies without any 1989 observation.
/// assert_eq!(windows, vec![1990, 1991]);
/// ```
pub fn discover_historical_windows(
    inflow_history: &[InflowHistoryRow],
    hydro_ids: &[EntityId],
    stages: &[Stage],
    user_pool: Option<&HistoricalYears>,
    season_map: Option<&SeasonMap>,
    forward_passes: u32,
) -> Result<Vec<i32>, StochasticError> {
    let mut stage_index: Vec<(NaiveDate, NaiveDate, i32, usize)> = stages
        .iter()
        .filter_map(|s| s.season_id.map(|sid| (s.start_date, s.end_date, s.id, sid)))
        .collect();
    stage_index.sort_unstable_by_key(|(start, _, _, _)| *start);

    let lookup: HashSet<(EntityId, i32, usize)> = inflow_history
        .iter()
        .filter_map(|r| {
            let (season_id, year) = history_row_key(&stage_index, season_map, r.start_date)?;
            Some((r.hydro_id, year, season_id))
        })
        .collect();

    let required_sequence: Vec<(i32, usize)> =
        super::build_observation_sequence(stages, season_map);

    let mut candidate_years: Vec<i32> = match user_pool {
        Some(pool) => pool.to_years(),
        None => lookup
            .iter()
            .map(|&(_, year, _)| year)
            .collect::<HashSet<i32>>()
            .into_iter()
            .collect(),
    };
    candidate_years.sort_unstable();

    let valid_windows: Vec<i32> = candidate_years
        .into_iter()
        .filter(|&y| is_window_complete(y, &required_sequence, hydro_ids, &lookup))
        .collect();

    if valid_windows.is_empty() {
        return Err(StochasticError::InsufficientData {
            context: "no valid historical windows found: ensure that inflow history covers \
                      the required seasons for at least one starting year"
                .to_string(),
        });
    }

    if valid_windows.len() < forward_passes as usize {
        tracing::warn!(
            n_windows = valid_windows.len(),
            forward_passes,
            "fewer windows ({}) than forward passes ({forward_passes}): \
             historical sampling will repeat windows across forward passes",
            valid_windows.len()
        );
    }

    Ok(valid_windows)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// The `(season_id, year)` key of a history row dated `date`. Under a season
/// map the year is [`observation_occurrence_year`], so it equals the year
/// `build_observation_sequence` dates the matching study entry with.
pub(super) fn history_row_key(
    stage_index: &[(NaiveDate, NaiveDate, i32, usize)],
    season_map: Option<&SeasonMap>,
    date: NaiveDate,
) -> Option<(usize, i32)> {
    let stage_season = find_season_for_date(stage_index, date);
    let Some(sm) = season_map else {
        return Some((stage_season.unwrap_or(date.month0() as usize), date.year()));
    };
    let season_id = stage_season.or_else(|| sm.season_for_date(date))?;
    Some((season_id, observation_occurrence_year(sm, season_id, date)))
}

fn is_window_complete(
    y: i32,
    required_sequence: &[(i32, usize)],
    hydro_ids: &[EntityId],
    lookup: &HashSet<(EntityId, i32, usize)>,
) -> bool {
    for &hydro_id in hydro_ids {
        for &(year_offset, season_id) in required_sequence {
            if !lookup.contains(&(hydro_id, y + year_offset, season_id)) {
                return false;
            }
        }
    }
    true
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp
)]
mod tests {
    use chrono::{Datelike, NaiveDate};
    use cobre_core::{
        EntityId,
        scenario::{HistoricalYears, InflowHistoryRow},
        temporal::{
            Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
            StageStateConfig,
        },
    };

    use super::discover_historical_windows;
    use crate::test_support::{
        MonthlyLabels, monthly_season_map, quarterly_season_map, sparse_ring_season_map,
    };

    fn monthly_history(hydro_id: EntityId, from_year: i32, to_year: i32) -> Vec<InflowHistoryRow> {
        (from_year..=to_year)
            .flat_map(|y| {
                (1u32..=12).map(move |m| {
                    let start_date = NaiveDate::from_ymd_opt(y, m, 1).unwrap();
                    InflowHistoryRow {
                        hydro_id,
                        start_date,
                        end_date: start_date
                            .checked_add_months(chrono::Months::new(1))
                            .unwrap(),
                        value_m3s: 100.0,
                    }
                })
            })
            .collect()
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    fn twelve_monthly_stages() -> Vec<Stage> {
        (0_usize..12)
            .map(|i| Stage {
                index: i,
                id: i as i32,
                start_date: NaiveDate::from_ymd_opt(2024, (i as u32 % 12) + 1, 1).unwrap(),
                end_date: NaiveDate::from_ymd_opt(2024, (i as u32 % 12) + 1, 28).unwrap(),
                season_id: Some(i),
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
                    branching_factor: 5,
                    noise_method: NoiseMethod::Saa,
                },
            })
            .collect()
    }

    #[test]
    fn test_auto_discovery_all_valid() {
        let hydro1 = EntityId(1);
        let hydro2 = EntityId(2);
        let mut history = monthly_history(hydro1, 1990, 2010);
        history.extend(monthly_history(hydro2, 1990, 2010));

        let stages = twelve_monthly_stages();
        let windows =
            discover_historical_windows(&history, &[hydro1, hydro2], &stages, None, None, 10)
                .unwrap();

        let expected: Vec<i32> = (1990..=2010).collect();
        assert_eq!(windows, expected, "expected exactly years 1990–2010");
    }

    #[test]
    fn test_user_pool_list_filters() {
        let hydro1 = EntityId(1);
        let hydro2 = EntityId(2);
        let mut history = monthly_history(hydro1, 1990, 2010);
        history.extend(monthly_history(hydro2, 1990, 2010));

        let stages = twelve_monthly_stages();
        let pool = HistoricalYears::List(vec![1995, 2000]);
        let windows =
            discover_historical_windows(&history, &[hydro1, hydro2], &stages, Some(&pool), None, 5)
                .unwrap();

        assert_eq!(windows, vec![1995, 2000]);
    }

    #[test]
    fn test_user_pool_range_expands() {
        let hydro1 = EntityId(1);
        let hydro2 = EntityId(2);
        let mut history = monthly_history(hydro1, 1990, 2010);
        history.extend(monthly_history(hydro2, 1990, 2010));

        let stages = twelve_monthly_stages();
        let pool = HistoricalYears::Range {
            from: 2000,
            to: 2002,
        };
        let windows =
            discover_historical_windows(&history, &[hydro1, hydro2], &stages, Some(&pool), None, 5)
                .unwrap();

        assert_eq!(windows, vec![2000, 2001, 2002]);
    }

    #[test]
    fn test_no_valid_windows_returns_error() {
        let hydro1 = EntityId(1);
        let hydro2 = EntityId(2);
        let mut history = monthly_history(hydro1, 1990, 2010);
        history.extend(monthly_history(hydro2, 1990, 2010));

        let stages = twelve_monthly_stages();
        let pool = HistoricalYears::List(vec![2020]);
        let result =
            discover_historical_windows(&history, &[hydro1, hydro2], &stages, Some(&pool), None, 1);

        assert!(result.is_err(), "expected Err when no valid windows found");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("no valid historical windows"),
            "error message should mention 'no valid historical windows', got: {msg}"
        );
    }

    #[test]
    fn test_incomplete_hydro_excludes_window() {
        let hydro1 = EntityId(1);
        let hydro2 = EntityId(2);
        let mut history = monthly_history(hydro1, 1990, 2010);

        history.extend(monthly_history(hydro2, 1990, 2005));
        history.extend(monthly_history(hydro2, 2007, 2010));

        let stages = twelve_monthly_stages();
        let windows =
            discover_historical_windows(&history, &[hydro1, hydro2], &stages, None, None, 5)
                .unwrap();

        assert!(
            !windows.contains(&2006),
            "window 2006 should be excluded because hydro2 lacks 2006 data"
        );
        assert!(windows.contains(&2005), "window 2005 should still be valid");
        assert!(
            windows.contains(&2007),
            "window 2007 should be valid although hydro2 lacks the 2006 data before it"
        );
    }

    #[test]
    fn test_to_years_list() {
        let years = HistoricalYears::List(vec![1, 3, 5]);
        assert_eq!(years.to_years(), vec![1, 3, 5]);
    }

    #[test]
    fn test_to_years_range() {
        let years = HistoricalYears::Range {
            from: 2000,
            to: 2003,
        };
        assert_eq!(years.to_years(), vec![2000, 2001, 2002, 2003]);
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    fn four_quarterly_stages() -> Vec<Stage> {
        let quarter_starts = [(1u32, 1u32), (4, 1), (7, 1), (10, 1)];
        let quarter_ends = [(4u32, 1u32), (7, 1), (10, 1), (12, 31)];
        (0_usize..4)
            .map(|i| {
                let (sm, sd) = quarter_starts[i];
                let (em, ed) = quarter_ends[i];
                Stage {
                    index: i,
                    id: i as i32,
                    start_date: NaiveDate::from_ymd_opt(2024, sm, sd).unwrap(),
                    end_date: NaiveDate::from_ymd_opt(2024, em, ed).unwrap(),
                    season_id: Some(i),
                    blocks: vec![Block {
                        index: 0,
                        name: "SINGLE".to_string(),
                        duration_hours: 2160.0,
                    }],
                    block_mode: BlockMode::Parallel,
                    state_config: StageStateConfig {
                        storage: true,
                        inflow_lags: false,
                    },
                    risk_config: StageRiskConfig::Expectation,
                    scenario_config: ScenarioSourceConfig {
                        branching_factor: 5,
                        noise_method: NoiseMethod::Saa,
                    },
                }
            })
            .collect()
    }

    fn quarterly_history(
        hydro_id: EntityId,
        from_year: i32,
        to_year: i32,
    ) -> Vec<InflowHistoryRow> {
        let quarter_months = [1u32, 4, 7, 10];
        (from_year..=to_year)
            .flat_map(|y| {
                quarter_months.iter().map(move |&m| {
                    let start_date = NaiveDate::from_ymd_opt(y, m, 1).unwrap();
                    InflowHistoryRow {
                        hydro_id,
                        start_date,
                        end_date: start_date
                            .checked_add_months(chrono::Months::new(3))
                            .unwrap(),
                        value_m3s: 100.0,
                    }
                })
            })
            .collect()
    }

    #[test]
    fn test_monthly_season_map_identical_to_month0() {
        let hydro1 = EntityId(1);
        let hydro2 = EntityId(2);
        let mut history = monthly_history(hydro1, 1990, 2010);
        history.extend(monthly_history(hydro2, 1990, 2010));
        let stages = twelve_monthly_stages();

        let sm = monthly_season_map(MonthlyLabels::ZeroBased);

        let windows_none =
            discover_historical_windows(&history, &[hydro1, hydro2], &stages, None, None, 10)
                .unwrap();
        let windows_with_sm =
            discover_historical_windows(&history, &[hydro1, hydro2], &stages, None, Some(&sm), 10)
                .unwrap();

        assert_eq!(
            windows_none, windows_with_sm,
            "monthly SeasonMap must produce identical results to month0() fallback"
        );
    }

    #[test]
    fn test_quarterly_season_map_window_discovery() {
        let hydro1 = EntityId(1);
        let history = quarterly_history(hydro1, 1990, 2010);
        let stages = four_quarterly_stages();
        let sm = quarterly_season_map();

        let windows =
            discover_historical_windows(&history, &[hydro1], &stages, None, Some(&sm), 10).unwrap();

        let expected: Vec<i32> = (1990..=2010).collect();
        assert_eq!(
            windows, expected,
            "expected windows 1990–2010 for quarterly study"
        );
    }

    #[test]
    fn test_none_season_map_backward_compat() {
        let hydro1 = EntityId(1);
        let mut history = monthly_history(hydro1, 1990, 2010);
        history.extend(monthly_history(EntityId(2), 1990, 2010));
        let stages = twelve_monthly_stages();

        let windows =
            discover_historical_windows(&history, &[hydro1, EntityId(2)], &stages, None, None, 10)
                .unwrap();

        let expected: Vec<i32> = (1990..=2010).collect();
        assert_eq!(
            windows, expected,
            "None season_map must reproduce the month0()-based result (1990–2010)"
        );
    }

    #[test]
    fn test_month0_fallback_matches_monthly_season_map() {
        let hydro1 = EntityId(1);
        let hydro2 = EntityId(2);
        let mut history = monthly_history(hydro1, 1990, 2010);
        history.extend(monthly_history(hydro2, 1990, 2010));
        let stages = twelve_monthly_stages();
        let sm = monthly_season_map(MonthlyLabels::ZeroBased);

        let windows_none =
            discover_historical_windows(&history, &[hydro1, hydro2], &stages, None, None, 10)
                .unwrap();
        let windows_with_sm =
            discover_historical_windows(&history, &[hydro1, hydro2], &stages, None, Some(&sm), 10)
                .unwrap();

        assert_eq!(
            windows_none, windows_with_sm,
            "month0() fallback (season_map = None) must produce identical window years \
             to the monthly SeasonMap path"
        );

        // Pin the absolute window set too: the comparison above passes if both
        // paths are broken identically.
        let expected: Vec<i32> = (1990..=2010).collect();
        assert_eq!(
            windows_none, expected,
            "monthly study must discover windows 1990–2010"
        );
    }

    fn three_monthly_stages(year: i32) -> Vec<Stage> {
        [(0_usize, 0_i32, 1_u32), (1, 1, 2), (2, 2, 3)]
            .into_iter()
            .map(|(index, id, month)| Stage {
                index,
                id,
                start_date: NaiveDate::from_ymd_opt(year, month, 1).unwrap(),
                end_date: NaiveDate::from_ymd_opt(year, month, 28).unwrap(),
                season_id: Some(index),
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
                    branching_factor: 5,
                    noise_method: NoiseMethod::Saa,
                },
            })
            .collect()
    }

    fn history_row(hydro_id: EntityId, year: i32, month: u32, value: f64) -> InflowHistoryRow {
        let start_date = NaiveDate::from_ymd_opt(year, month, 1).unwrap();
        InflowHistoryRow {
            hydro_id,
            start_date,
            end_date: start_date
                .checked_add_months(chrono::Months::new(1))
                .unwrap(),
            value_m3s: value,
        }
    }

    #[test]
    fn discover_admits_the_first_record_year_when_no_consumer_reads_its_lags() {
        let hydro = EntityId(1);
        let history = monthly_history(hydro, 1990, 1991);

        let windows = discover_historical_windows(
            &history,
            &[hydro],
            &twelve_monthly_stages(),
            None,
            None,
            10,
        )
        .unwrap();

        assert_eq!(windows, vec![1990, 1991]);
    }

    #[test]
    fn discover_rejects_a_year_missing_a_study_entry_observation() {
        let hydro = EntityId(1);
        let history: Vec<InflowHistoryRow> = monthly_history(hydro, 1990, 1991)
            .into_iter()
            .filter(|r| !(r.start_date.year() == 1990 && r.start_date.month() == 6))
            .collect();

        let windows = discover_historical_windows(
            &history,
            &[hydro],
            &twelve_monthly_stages(),
            None,
            None,
            10,
        )
        .unwrap();

        assert_eq!(windows, vec![1991]);
    }

    #[test]
    fn discover_admits_a_year_without_its_ring_predecessor_on_a_sparse_id_map() {
        let hydro = EntityId(1);
        let stages = three_monthly_stages(2026);
        let sm = sparse_ring_season_map();

        let mut history: Vec<InflowHistoryRow> = Vec::new();
        for &year in &[2024, 2025] {
            for month in 1..=3u32 {
                history.push(history_row(hydro, year, month, 100.0));
            }
        }

        let windows =
            discover_historical_windows(&history, &[hydro], &stages, None, Some(&sm), 10).unwrap();

        assert_eq!(windows, vec![2024, 2025]);
    }
}
