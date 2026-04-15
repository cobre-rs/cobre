//! Historical window discovery algorithm.
//!
//! A "window" is a starting year `y` such that every hydro in the study has a
//! contiguous sequence of historical observations covering `max_par_order +
//! n_study_stages` seasons beginning in year `y`. The algorithm aligns
//! observations to study stages via `season_id` matching (not raw calendar
//! arithmetic), supports optional user-specified year pools
//! ([`HistoricalYears`]), and emits a [`tracing::warn!`] when the discovered
//! pool is smaller than the number of forward passes.
//!
//! ## Season-to-year mapping
//!
//! The window starting year `y` is the year of the **first study observation**.
//! Lag seasons are derived by stepping backwards `max_par_order` steps from
//! the first study stage's `season_id` using modular arithmetic on
//! `n_seasons`. Lag entries receive negative year offsets relative to `y`,
//! and the year decrements whenever the season sequence wraps from `0` back
//! to `n_seasons - 1` going backwards.
//!
//! ### Example (monthly, `max_par_order` = 2, `season_ids` 0–11)
//!
//! Full season sequence (offsets after normalization): `[(-1,10), (-1,11), (0,0), (0,1), …, (0,11)]`.
//!
//! For window year `y = 1990`:
//! - Lag observations: `(1989, season 10)`, `(1989, season 11)`
//! - Study observations: `(1990, season 0)` … `(1990, season 11)`
//!
//! [`HistoricalYears`]: cobre_core::scenario::HistoricalYears

use std::collections::HashSet;

use chrono::{Datelike, NaiveDate};
use cobre_core::{
    scenario::{HistoricalYears, InflowHistoryRow},
    temporal::{SeasonMap, Stage},
    EntityId,
};

use crate::{par::fitting::find_season_for_date, StochasticError};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Discover the set of valid historical window starting years.
///
/// A window starting year `y` is **valid** when every hydro in `hydro_ids`
/// has a historical observation for every required `(year, season_id)` pair
/// in the window's observation sequence.
///
/// The observation sequence for a window of year `y` consists of:
/// 1. `max_par_order` lag observations (seasons immediately before the first
///    study stage, in time order), located in year `y`.
/// 2. `stages.len()` study observations (one per study stage), located in
///    year `y + 1` (or further, depending on how many seasonal cycles the
///    study spans).
///
/// When `user_pool` is `Some`, only windows whose starting year appears in
/// the expanded pool are returned. When `None`, all valid auto-discovered
/// windows are returned.
///
/// The `season_map` parameter controls how observation dates are mapped to
/// season IDs when building the observation lookup:
///
/// 1. Dates that fall within a study stage's `[start_date, end_date)` range
///    are mapped via binary search on the stage index (exact match).
/// 2. Dates outside the study range are mapped via
///    `season_map.season_for_date(date)` when `season_map` is `Some`.
/// 3. When `season_map` is `None`, dates outside the study range fall back
///    to `month0()` (0 = January … 11 = December) for backward compatibility.
///
/// # Errors
///
/// Returns [`StochasticError::InsufficientData`] when no valid windows are
/// found after applying the user pool filter.
///
/// # Warnings
///
/// Emits [`tracing::warn!`] when the number of valid windows is less than
/// `forward_passes`.
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
///         (1u32..=12).map(move |m| InflowHistoryRow {
///             hydro_id,
///             date: NaiveDate::from_ymd_opt(y, m, 1).unwrap(),
///             value_m3s: 100.0,
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
    let all_years: HashSet<i32> = inflow_history.iter().map(|r| r.date.year()).collect();

    let mut stage_index: Vec<(NaiveDate, NaiveDate, i32, usize)> = stages
        .iter()
        .filter_map(|s| s.season_id.map(|sid| (s.start_date, s.end_date, s.id, sid)))
        .collect();
    stage_index.sort_unstable_by_key(|(start, _, _, _)| *start);

    let lookup: HashSet<(EntityId, i32, usize)> = inflow_history
        .iter()
        .filter_map(|r| {
            let season_id = find_season_for_date(&stage_index, r.date)
                .or_else(|| season_map.and_then(|sm| sm.season_for_date(r.date)))
                .or_else(|| {
                    if season_map.is_none() {
                        Some(r.date.month0() as usize)
                    } else {
                        None
                    }
                })?;
            Some((r.hydro_id, r.date.year(), season_id))
        })
        .collect();

    let n_seasons = stages
        .iter()
        .filter_map(|s| s.season_id)
        .max()
        .map_or(1, |m| m + 1);

    let required_sequence: Vec<(i32, usize)> =
        super::build_observation_sequence(stages, max_par_order, n_seasons);

    let candidate_years: Vec<i32> = if let Some(pool) = user_pool {
        let mut years: Vec<i32> = pool.to_years().into_iter().collect();
        years.sort_unstable();
        years
    } else {
        let mut years: Vec<i32> = all_years.into_iter().collect();
        years.sort_unstable();
        years
    };

    let mut valid_windows: Vec<i32> = candidate_years
        .into_iter()
        .filter(|&y| is_window_complete(y, &required_sequence, hydro_ids, &lookup))
        .collect();

    valid_windows.sort_unstable();

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

/// Return `true` if all required observations exist in the lookup for every hydro.
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
    use chrono::NaiveDate;
    use cobre_core::{
        scenario::{HistoricalYears, InflowHistoryRow},
        temporal::{
            Block, BlockMode, NoiseMethod, ScenarioSourceConfig, SeasonCycleType, SeasonDefinition,
            SeasonMap, Stage, StageRiskConfig, StageStateConfig,
        },
        EntityId,
    };

    use super::discover_historical_windows;

    // -----------------------------------------------------------------------
    // Test helpers
    // -----------------------------------------------------------------------

    /// Build a monthly history for `hydro_id` spanning years [`from_year`, `to_year`].
    /// Each month has one row on the 1st of the month.
    fn monthly_history(hydro_id: EntityId, from_year: i32, to_year: i32) -> Vec<InflowHistoryRow> {
        (from_year..=to_year)
            .flat_map(|y| {
                (1u32..=12).map(move |m| InflowHistoryRow {
                    hydro_id,
                    date: NaiveDate::from_ymd_opt(y, m, 1).unwrap(),
                    value_m3s: 100.0,
                })
            })
            .collect()
    }

    /// Build 12 monthly study stages with `season_ids` 0–11.
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

    // -----------------------------------------------------------------------
    // Test 1: auto-discovery returns all valid years
    // -----------------------------------------------------------------------

    #[test]
    fn test_auto_discovery_all_valid() {
        // 2 hydros, monthly history 1990–2010 (252 rows each), 12 study stages,
        // max_par_order = 2. Expected: years 1991–2010 (20 windows).
        //
        // Under the new convention, window_year Y means study starts at year Y.
        // For window y, the algorithm needs:
        //   lags  → (y-1, season 10), (y-1, season 11)
        //   study → (y,   season 0) … (y,   season 11)
        //
        // y = 1991: needs (1990, 10/11) + (1991, 0..11) → all in [1990, 2010] ✓
        // y = 2010: needs (2009, 10/11) + (2010, 0..11) → all in [1990, 2010] ✓
        // y = 1990: needs (1989, 10/11) → 1989 not in data ✗
        // y = 2011: needs (2010, 10/11) + (2011, 0..11) → 2011 not in data ✗
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

    // -----------------------------------------------------------------------
    // Test 2: user pool (List) filters correctly
    // -----------------------------------------------------------------------

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

    // -----------------------------------------------------------------------
    // Test 3: user pool (Range) expands correctly
    // -----------------------------------------------------------------------

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

    // -----------------------------------------------------------------------
    // Test 4: user pool with no valid windows returns Err
    // -----------------------------------------------------------------------

    #[test]
    fn test_no_valid_windows_returns_error() {
        let hydro1 = EntityId(1);
        let hydro2 = EntityId(2);
        let mut history = monthly_history(hydro1, 1990, 2010);
        history.extend(monthly_history(hydro2, 1990, 2010));

        let stages = twelve_monthly_stages();
        // Year 2020 has no data → no valid windows
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

    // -----------------------------------------------------------------------
    // Test 5: incomplete hydro excludes window from auto-discovery
    // -----------------------------------------------------------------------

    #[test]
    fn test_incomplete_hydro_excludes_window() {
        // hydro1 has full data 1990–2010.
        // hydro2 is missing all of 2006.
        //
        // Under the new convention, window_year Y means study starts at year Y.
        // Window y=2006 requires (2005, seasons 10/11) and (2006, seasons 0..11).
        // hydro2 has no 2006 data → window 2006 must be excluded.
        let hydro1 = EntityId(1);
        let hydro2 = EntityId(2);
        let mut history = monthly_history(hydro1, 1990, 2010);

        // hydro2: full range except 2006
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
        // 2005 should still be valid: needs (2004, 10/11) + (2005, 0..11).
        // hydro2 has data through 2005 → 2005 is valid.
        assert!(windows.contains(&2005), "window 2005 should still be valid");
    }

    // -----------------------------------------------------------------------
    // Test 6: HistoricalYears::List to_years returns list as-is
    // -----------------------------------------------------------------------

    #[test]
    fn test_to_years_list() {
        let years = HistoricalYears::List(vec![1, 3, 5]);
        assert_eq!(years.to_years(), vec![1, 3, 5]);
    }

    // -----------------------------------------------------------------------
    // Test 7: HistoricalYears::Range to_years expands correctly
    // -----------------------------------------------------------------------

    #[test]
    fn test_to_years_range() {
        let years = HistoricalYears::Range {
            from: 2000,
            to: 2003,
        };
        assert_eq!(years.to_years(), vec![2000, 2001, 2002, 2003]);
    }

    // -----------------------------------------------------------------------
    // Test helpers for SeasonMap construction
    // -----------------------------------------------------------------------

    /// Build a standard monthly `SeasonMap` (12 seasons, IDs 0–11).
    fn monthly_season_map() -> SeasonMap {
        let seasons = (0_usize..12)
            .map(|i| SeasonDefinition {
                id: i,
                label: format!("Month{i}"),
                #[allow(clippy::cast_possible_truncation)]
                month_start: (i as u32) + 1,
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

    /// Build quarterly stages (4 stages, each 3 months, `season_ids` 0–3).
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    fn four_quarterly_stages() -> Vec<Stage> {
        // Q1: Jan–Mar (start Jan 1, end Apr 1)
        // Q2: Apr–Jun (start Apr 1, end Jul 1)
        // Q3: Jul–Sep (start Jul 1, end Oct 1)
        // Q4: Oct–Dec (start Oct 1, end Jan 1 next year)
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

    /// Build a quarterly history: one row per quarter per year for `hydro_id`.
    fn quarterly_history(
        hydro_id: EntityId,
        from_year: i32,
        to_year: i32,
    ) -> Vec<InflowHistoryRow> {
        let quarter_months = [1u32, 4, 7, 10];
        (from_year..=to_year)
            .flat_map(|y| {
                quarter_months.iter().map(move |&m| InflowHistoryRow {
                    hydro_id,
                    date: NaiveDate::from_ymd_opt(y, m, 1).unwrap(),
                    value_m3s: 100.0,
                })
            })
            .collect()
    }

    /// Build a quarterly `SeasonMap` (4 seasons, IDs 0–3).
    fn quarterly_season_map() -> SeasonMap {
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

    // -----------------------------------------------------------------------
    // Test 8: monthly SeasonMap produces identical results to month0() (None)
    // -----------------------------------------------------------------------

    #[test]
    fn test_monthly_season_map_identical_to_month0() {
        // A standard Monthly SeasonMap with IDs 0-11 must produce the same
        // window set as passing None (which falls back to month0()).
        let hydro1 = EntityId(1);
        let hydro2 = EntityId(2);
        let mut history = monthly_history(hydro1, 1990, 2010);
        history.extend(monthly_history(hydro2, 1990, 2010));
        let stages = twelve_monthly_stages();

        let sm = monthly_season_map();

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

    // -----------------------------------------------------------------------
    // Test 9: quarterly SeasonMap discovers correct windows
    // -----------------------------------------------------------------------

    #[test]
    fn test_quarterly_season_map_window_discovery() {
        // One hydro, quarterly data from 1990-2010, 4 quarterly stages,
        // max_par_order = 1.
        //
        // Under the new convention, window_year Y means study starts at year Y.
        // Window sequence for y (1 lag + 4 study quarters):
        //   lag  → (y-1, season 3) [Q4=Oct]
        //   study → (y,   season 0..3) [Q1-Q4]
        //
        // y = 1991: needs (1990, Q4) and (1991, Q1–Q4) → all present ✓
        // y = 2010: needs (2009, Q4) and (2010, Q1–Q4) → all present ✓
        // y = 1990: needs (1989, Q4) → 1989 missing ✗
        // y = 2011: needs (2010, Q4) + (2011, Q1–Q4) → 2011 missing ✗
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

    // -----------------------------------------------------------------------
    // Test 10: None season_map backward compatibility
    // -----------------------------------------------------------------------

    #[test]
    fn test_none_season_map_backward_compat() {
        // Passing None must fall back to month0() for season ID resolution.
        // Under the new convention, window_year Y means study starts at year Y,
        // so lags are sought at year Y-1.
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

    // -----------------------------------------------------------------------
    // Test 11: month0() fallback is identical to monthly SeasonMap path
    // -----------------------------------------------------------------------

    /// Given 12 monthly stages and monthly history for 2 hydros from 1990–2010,
    /// `discover_historical_windows` with `season_map = None` (which falls back
    /// to `month0()`) must return exactly the same window years as calling it
    /// with `season_map = Some(&monthly_sm)`.
    ///
    /// This directly verifies that the `month0()` fallback is correct for
    /// monthly studies: the fallback path (`month0()`) and the calendar-based
    /// `SeasonMap` path both produce `season_id = date.month0()` for dates on
    /// the 1st of each month, so the discovered window sets must be identical.
    #[test]
    fn test_month0_fallback_matches_monthly_season_map() {
        let hydro1 = EntityId(1);
        let hydro2 = EntityId(2);
        let mut history = monthly_history(hydro1, 1990, 2010);
        history.extend(monthly_history(hydro2, 1990, 2010));
        let stages = twelve_monthly_stages();
        let sm = monthly_season_map();

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

        // Additionally verify the expected window set (1991–2010) so this test
        // is self-contained and not merely a reflexive comparison.
        let expected: Vec<i32> = (1991..=2010).collect();
        assert_eq!(
            windows_none, expected,
            "monthly study must discover windows 1991–2010"
        );
    }
}
