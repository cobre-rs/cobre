//! Historical window discovery algorithm.
//!
//! A "window" is a starting year `y` such that every hydro in the study has a
//! historical observation for every study season and every pre-study lag
//! season the season-map walk resolves. Observations align to study stages by
//! `season_id` matching, not by raw calendar arithmetic.
//!
//! `build_observation_sequence` owns the `(year_offset, season_id)` layout the
//! window year is resolved against; its lag entries come from
//! [`StageCalendar::season_occurrences`](crate::season_cast::StageCalendar::season_occurrences),
//! never from arithmetic on declared season ids. `y` is the first study
//! observation's year.

use std::collections::HashSet;

use chrono::{Datelike, NaiveDate};
use cobre_core::{
    EntityId,
    scenario::{HistoricalYears, InflowHistoryRow},
    temporal::{SeasonMap, Stage},
};

use crate::{StochasticError, par::fitting::find_season_for_date};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Discover the set of valid historical window starting years.
///
/// A window starting year `y` is **valid** when every hydro in `hydro_ids`
/// has a historical observation for every `(y + year_offset, season_id)` pair
/// `build_observation_sequence` emits.
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
/// let windows = discover_historical_windows(
///     &history,
///     &[hydro_id],
///     &stages,
///     2,
///     None,
///     None,
///     10,
/// )
/// .unwrap();
///
/// // window_year=1991: study at 1991, lags at 1990 (season 10/11) — all present.
/// // window_year=1990: lags would be at 1989 — not in history.
/// assert_eq!(windows, vec![1991]);
/// ```
pub fn discover_historical_windows(
    inflow_history: &[InflowHistoryRow],
    hydro_ids: &[EntityId],
    stages: &[Stage],
    max_par_order: usize,
    user_pool: Option<&HistoricalYears>,
    season_map: Option<&SeasonMap>,
    forward_passes: u32,
) -> Result<Vec<i32>, StochasticError> {
    let all_years: HashSet<i32> = inflow_history.iter().map(|r| r.start_date.year()).collect();

    let mut stage_index: Vec<(NaiveDate, NaiveDate, i32, usize)> = stages
        .iter()
        .filter_map(|s| s.season_id.map(|sid| (s.start_date, s.end_date, s.id, sid)))
        .collect();
    stage_index.sort_unstable_by_key(|(start, _, _, _)| *start);

    let lookup: HashSet<(EntityId, i32, usize)> = inflow_history
        .iter()
        .filter_map(|r| {
            let season_id = find_season_for_date(&stage_index, r.start_date)
                .or_else(|| season_map.and_then(|sm| sm.season_for_date(r.start_date)))
                .or_else(|| season_map.is_none().then(|| r.start_date.month0() as usize))?;
            Some((r.hydro_id, r.start_date.year(), season_id))
        })
        .collect();

    let required_sequence: Vec<(i32, usize)> =
        super::build_observation_sequence(stages, max_par_order, season_map);

    let mut candidate_years: Vec<i32> = match user_pool {
        Some(pool) => pool.to_years(),
        None => all_years.into_iter().collect(),
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
            Block, BlockMode, NoiseMethod, ScenarioSourceConfig, SeasonCycleType, SeasonDefinition,
            SeasonMap, Stage, StageRiskConfig, StageStateConfig,
        },
    };

    use super::discover_historical_windows;
    use crate::season_cast::{nth_previous_occurrence, season_period_window};
    use crate::test_support::{MonthlyLabels, monthly_season_map, quarterly_season_map};

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
        // Window y needs (y-1, seasons 10/11) + (y, seasons 0..11), so 1990
        // (lags in 1989) and 2011 (study in 2011) fall outside a 1990–2010 history.
        let hydro1 = EntityId(1);
        let hydro2 = EntityId(2);
        let mut history = monthly_history(hydro1, 1990, 2010);
        history.extend(monthly_history(hydro2, 1990, 2010));

        let stages = twelve_monthly_stages();
        let windows =
            discover_historical_windows(&history, &[hydro1, hydro2], &stages, 2, None, None, 10)
                .unwrap();

        let expected: Vec<i32> = (1991..=2010).collect();
        assert_eq!(windows, expected, "expected exactly years 1991–2010");
    }

    #[test]
    fn test_user_pool_list_filters() {
        let hydro1 = EntityId(1);
        let hydro2 = EntityId(2);
        let mut history = monthly_history(hydro1, 1990, 2010);
        history.extend(monthly_history(hydro2, 1990, 2010));

        let stages = twelve_monthly_stages();
        let pool = HistoricalYears::List(vec![1995, 2000]);
        let windows = discover_historical_windows(
            &history,
            &[hydro1, hydro2],
            &stages,
            2,
            Some(&pool),
            None,
            5,
        )
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
        let windows = discover_historical_windows(
            &history,
            &[hydro1, hydro2],
            &stages,
            2,
            Some(&pool),
            None,
            5,
        )
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
        let result = discover_historical_windows(
            &history,
            &[hydro1, hydro2],
            &stages,
            2,
            Some(&pool),
            None,
            1,
        );

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
            discover_historical_windows(&history, &[hydro1, hydro2], &stages, 2, None, None, 5)
                .unwrap();

        assert!(
            !windows.contains(&2006),
            "window 2006 should be excluded because hydro2 lacks 2006 data"
        );
        // 2005 needs only (2004, 10/11) + (2005, 0..11), all present.
        assert!(windows.contains(&2005), "window 2005 should still be valid");
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
            discover_historical_windows(&history, &[hydro1, hydro2], &stages, 2, None, None, 10)
                .unwrap();
        let windows_with_sm = discover_historical_windows(
            &history,
            &[hydro1, hydro2],
            &stages,
            2,
            None,
            Some(&sm),
            10,
        )
        .unwrap();

        assert_eq!(
            windows_none, windows_with_sm,
            "monthly SeasonMap must produce identical results to month0() fallback"
        );
    }

    #[test]
    fn test_quarterly_season_map_window_discovery() {
        // Window y needs (y-1, Q4) + (y, Q1–Q4), so 1990 and 2011 fall outside
        // a 1990–2010 history.
        let hydro1 = EntityId(1);
        let history = quarterly_history(hydro1, 1990, 2010);
        let stages = four_quarterly_stages();
        let sm = quarterly_season_map();

        let windows =
            discover_historical_windows(&history, &[hydro1], &stages, 1, None, Some(&sm), 10)
                .unwrap();

        let expected: Vec<i32> = (1991..=2010).collect();
        assert_eq!(
            windows, expected,
            "expected windows 1991–2010 for quarterly study"
        );
    }

    #[test]
    fn test_none_season_map_backward_compat() {
        let hydro1 = EntityId(1);
        let mut history = monthly_history(hydro1, 1990, 2010);
        history.extend(monthly_history(EntityId(2), 1990, 2010));
        let stages = twelve_monthly_stages();

        let windows = discover_historical_windows(
            &history,
            &[hydro1, EntityId(2)],
            &stages,
            2,
            None,
            None,
            10,
        )
        .unwrap();

        let expected: Vec<i32> = (1991..=2010).collect();
        assert_eq!(
            windows, expected,
            "None season_map must reproduce the month0()-based result (1991–2010)"
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
            discover_historical_windows(&history, &[hydro1, hydro2], &stages, 2, None, None, 10)
                .unwrap();
        let windows_with_sm = discover_historical_windows(
            &history,
            &[hydro1, hydro2],
            &stages,
            2,
            None,
            Some(&sm),
            10,
        )
        .unwrap();

        assert_eq!(
            windows_none, windows_with_sm,
            "month0() fallback (season_map = None) must produce identical window years \
             to the monthly SeasonMap path"
        );

        // Pin the absolute window set too: the comparison above passes if both
        // paths are broken identically.
        let expected: Vec<i32> = (1991..=2010).collect();
        assert_eq!(
            windows_none, expected,
            "monthly study must discover windows 1991–2010"
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

    /// A sparse `Custom` map: `[0 Jan, 1 Feb, 2 Mar, 12 Apr-Jun, 13 Jul-Sep]`,
    /// the same shape as `stochastic_pipeline.rs`'s `ring_season_map`.
    fn sparse_ring_season_map() -> SeasonMap {
        let def = |id: usize, month_start: u32, month_end: Option<u32>| SeasonDefinition {
            id,
            label: format!("S{id}"),
            month_start,
            day_start: None,
            month_end,
            day_end: None,
        };
        SeasonMap {
            cycle_type: SeasonCycleType::Custom,
            seasons: vec![
                def(0, 1, None),
                def(1, 2, None),
                def(2, 3, None),
                def(12, 4, Some(6)),
                def(13, 7, Some(9)),
            ],
        }
    }

    #[test]
    fn discover_walks_the_calendar_predecessor_on_a_partial_year_study() {
        let hydro = EntityId(1);
        let stages = three_monthly_stages(2024);
        let sm = monthly_season_map(MonthlyLabels::ZeroBased);

        let mut history: Vec<InflowHistoryRow> = Vec::new();
        for &year in &[1991, 1992] {
            for month in 1..=3u32 {
                history.push(history_row(hydro, year, month, 100.0));
            }
        }
        history.push(history_row(hydro, 1990, 12, 50.0));
        history.push(history_row(hydro, 1991, 12, 50.0));

        let windows =
            discover_historical_windows(&history, &[hydro], &stages, 1, None, Some(&sm), 10)
                .unwrap();

        assert_eq!(windows, vec![1991, 1992]);
    }

    #[test]
    fn discover_covers_lags_beyond_the_declared_span() {
        let hydro = EntityId(1);
        let stages = three_monthly_stages(2024);
        let sm = monthly_season_map(MonthlyLabels::ZeroBased);
        let history = monthly_history(hydro, 1990, 1992);

        let windows =
            discover_historical_windows(&history, &[hydro], &stages, 6, None, Some(&sm), 10)
                .unwrap();

        assert_eq!(windows, vec![1991, 1992]);
    }

    #[test]
    fn discover_requires_the_ring_predecessor_on_a_sparse_id_map() {
        let hydro = EntityId(1);
        let stages = three_monthly_stages(2026);
        let sm = sparse_ring_season_map();

        let mut history: Vec<InflowHistoryRow> = Vec::new();
        for &year in &[2024, 2025] {
            for month in 1..=3u32 {
                history.push(history_row(hydro, year, month, 100.0));
            }
        }
        history.push(history_row(hydro, 2023, 8, 999.0));
        history.push(history_row(hydro, 2024, 8, 999.0));

        let windows =
            discover_historical_windows(&history, &[hydro], &stages, 1, None, Some(&sm), 10)
                .unwrap();

        assert_eq!(windows, vec![2024, 2025]);
    }

    #[test]
    fn discover_lag_entries_follow_nth_previous_occurrence() {
        let hydro = EntityId(1);
        let stages = three_monthly_stages(2024);
        let sm = monthly_season_map(MonthlyLabels::ZeroBased);
        let jan = sm.seasons.iter().find(|d| d.id == 0).unwrap();
        let anchor = season_period_window(&sm, jan, &stages[0]);
        let history = monthly_history(hydro, 1990, 1993);

        for k in 1..=3 {
            let occ = nth_previous_occurrence(&sm, jan, &anchor, k).unwrap();
            let target_year = 1992 + (occ.start.year() - stages[0].start_date.year());
            let target_month = occ.start.month();

            let probe_history: Vec<InflowHistoryRow> = history
                .iter()
                .filter(|r| {
                    !(r.start_date.year() == target_year && r.start_date.month() == target_month)
                })
                .cloned()
                .collect();

            let windows = discover_historical_windows(
                &probe_history,
                &[hydro],
                &stages,
                3,
                None,
                Some(&sm),
                10,
            )
            .unwrap();
            assert!(
                !windows.contains(&1992),
                "removing the walk's k={k} lag row must exclude window 1992"
            );
        }

        let probe_history: Vec<InflowHistoryRow> = history
            .iter()
            .filter(|r| !(r.start_date.year() == 1991 && r.start_date.month() == 3))
            .cloned()
            .collect();
        let windows =
            discover_historical_windows(&probe_history, &[hydro], &stages, 3, None, Some(&sm), 10)
                .unwrap();
        assert!(
            windows.contains(&1992),
            "removing Mar 1991, a row the walk does not name, must not exclude window 1992"
        );
    }
}
