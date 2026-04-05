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
//! The window starting year `y` is the year of the **first lag observation**.
//! Lag seasons are derived by stepping backwards `max_par_order` steps from
//! the first study stage's `season_id` using modular arithmetic on
//! `n_seasons`. The year increments whenever the season sequence wraps from
//! `n_seasons - 1` back to `0`.
//!
//! ### Example (monthly, `max_par_order` = 2, `season_ids` 0–11)
//!
//! Full season sequence: `[10, 11, 0, 1, …, 11]` (14 elements).
//!
//! For window year `y = 1990`:
//! - Lag observations: `(1990, season 10)`, `(1990, season 11)`
//! - Study observations: `(1991, season 0)` … `(1991, season 11)`
//!
//! [`HistoricalYears`]: cobre_core::scenario::HistoricalYears

use std::collections::HashSet;

use chrono::Datelike;
use cobre_core::{
    EntityId,
    scenario::{HistoricalYears, InflowHistoryRow},
    temporal::Stage,
};

use crate::StochasticError;

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
///     10,
/// )
/// .unwrap();
///
/// assert_eq!(windows, vec![1990]);
/// ```
pub fn discover_historical_windows(
    inflow_history: &[InflowHistoryRow],
    hydro_ids: &[EntityId],
    stages: &[Stage],
    max_par_order: usize,
    user_pool: Option<&HistoricalYears>,
    forward_passes: u32,
) -> Result<Vec<i32>, StochasticError> {
    // -----------------------------------------------------------------------
    // BR1: collect all unique years present in the history (candidate years)
    // -----------------------------------------------------------------------
    let all_years: HashSet<i32> = inflow_history.iter().map(|r| r.date.year()).collect();

    // -----------------------------------------------------------------------
    // Build observation lookup: (hydro_id, year, season_id) -> present
    //
    // Season is derived from the observation date via month0() (0 = January),
    // matching the resolve_season_id convention used by PrecomputedPar.
    // -----------------------------------------------------------------------
    let lookup: HashSet<(EntityId, i32, usize)> = inflow_history
        .iter()
        .map(|r| {
            let season_id = r.date.month0() as usize;
            (r.hydro_id, r.date.year(), season_id)
        })
        .collect();

    // -----------------------------------------------------------------------
    // Determine n_seasons from the stage season_ids.
    // n_seasons = max(season_id) + 1 across all study stages.
    // For a 12-month cycle with ids 0–11, n_seasons = 12.
    // -----------------------------------------------------------------------
    let n_seasons = stages
        .iter()
        .filter_map(|s| s.season_id)
        .max()
        .map_or(1, |m| m + 1);

    // -----------------------------------------------------------------------
    // Build the full observation-sequence template as a Vec<(year_offset, season_id)>.
    //
    // The template describes, for each required observation, the year offset
    // relative to the window starting year y and the required season_id.
    //
    // 1. Study seasons come from stages in order.
    // 2. Lag seasons are prepended: step backwards max_par_order times from
    //    study_seasons[0] using modular arithmetic.
    //
    // Year offset increments whenever the season_id sequence wraps from
    // (n_seasons - 1) to 0, indicating a calendar year boundary.
    // -----------------------------------------------------------------------
    let required_sequence: Vec<(i32, usize)> =
        build_required_sequence(stages, max_par_order, n_seasons);

    // -----------------------------------------------------------------------
    // Determine the candidate year set to evaluate.
    //
    // When a user pool is provided (BR4), expand it and intersect with
    // years present in the history. When absent (BR5), use all years found
    // in the history.
    // -----------------------------------------------------------------------
    let candidate_years: Vec<i32> = if let Some(pool) = user_pool {
        let pool_years: HashSet<i32> = pool.to_years().into_iter().collect();
        // Keep only years that appear in the pool (they may or may not be
        // present in the history — validity is checked below).
        let mut years: Vec<i32> = pool_years.into_iter().collect();
        years.sort_unstable();
        years
    } else {
        let mut years: Vec<i32> = all_years.into_iter().collect();
        years.sort_unstable();
        years
    };

    // -----------------------------------------------------------------------
    // BR2/BR3: for each candidate year, check that every required
    // (hydro_id, y + year_offset, season_id) triple is in the lookup.
    // -----------------------------------------------------------------------
    let mut valid_windows: Vec<i32> = candidate_years
        .into_iter()
        .filter(|&y| is_window_complete(y, &required_sequence, hydro_ids, &lookup))
        .collect();

    // -----------------------------------------------------------------------
    // BR6: sort ascending for deterministic output
    // -----------------------------------------------------------------------
    valid_windows.sort_unstable();

    // -----------------------------------------------------------------------
    // BR7: error when no valid windows remain
    // -----------------------------------------------------------------------
    if valid_windows.is_empty() {
        return Err(StochasticError::InsufficientData {
            context: "no valid historical windows found: ensure that inflow history covers \
                      the required seasons for at least one starting year"
                .to_string(),
        });
    }

    // -----------------------------------------------------------------------
    // BR8: warn when the pool is smaller than forward_passes
    // -----------------------------------------------------------------------
    let n_windows = valid_windows.len();
    if n_windows < forward_passes as usize {
        tracing::warn!(
            n_windows,
            forward_passes,
            "fewer windows ({n_windows}) than forward passes ({forward_passes}): \
             historical sampling will repeat windows across forward passes"
        );
    }

    Ok(valid_windows)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build the full required observation sequence as `(year_offset, season_id)` pairs.
///
/// The sequence starts with `max_par_order` lag observations (derived by
/// stepping backwards from `study_seasons[0]`), followed by the study stage
/// observations in order.
///
/// The year offset relative to the window starting year `y` increments
/// whenever the season index wraps from `n_seasons - 1` to `0`.
fn build_required_sequence(
    stages: &[Stage],
    max_par_order: usize,
    n_seasons: usize,
) -> Vec<(i32, usize)> {
    if stages.is_empty() {
        return Vec::new();
    }

    // Collect study season_ids in stage order.
    let study_seasons: Vec<usize> = stages.iter().filter_map(|s| s.season_id).collect();

    if study_seasons.is_empty() {
        return Vec::new();
    }

    // Build lag seasons by stepping backwards from study_seasons[0].
    // Lag seasons are in chronological order (oldest lag first).
    let first_study_season = study_seasons[0];
    let lag_seasons: Vec<usize> = (1..=max_par_order)
        .rev()
        .map(|k| {
            // Step k seasons before first_study_season, wrapping modularly.
            // n_seasons is always small (12 for monthly, 52 for weekly) so
            // truncation from usize to i32 is safe in practice.
            #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
            let n = n_seasons as i32;
            #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
            let s = first_study_season as i32;
            #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
            let k_i32 = k as i32;
            #[allow(clippy::cast_sign_loss)]
            let season = ((s - k_i32 % n + n) % n) as usize;
            season
        })
        .collect();

    // Concatenate: lag seasons (oldest first) then study seasons.
    let full_seasons: Vec<usize> = lag_seasons.into_iter().chain(study_seasons).collect();

    // Assign year offsets: year increments whenever season wraps 0..n_seasons-1 -> 0.
    let mut result = Vec::with_capacity(full_seasons.len());
    let mut year_offset: i32 = 0;
    let mut prev_season = full_seasons[0];

    for (i, &season) in full_seasons.iter().enumerate() {
        if i > 0 && season < prev_season {
            // Season wrapped around: we crossed a year boundary.
            year_offset += 1;
        }
        result.push((year_offset, season));
        prev_season = season;
    }

    result
}

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
        EntityId,
        scenario::{HistoricalYears, InflowHistoryRow},
        temporal::{
            Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
            StageStateConfig,
        },
    };

    use super::discover_historical_windows;

    // -----------------------------------------------------------------------
    // Test helpers
    // -----------------------------------------------------------------------

    /// Build a monthly history for `hydro_id` spanning years [from_year, to_year].
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

    /// Build 12 monthly study stages with season_ids 0–11.
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
        // max_par_order = 2. Expected: years 1990–2009 (20 windows).
        //
        // For window y, the algorithm needs:
        //   lags  → (y, season 10), (y, season 11)
        //   study → (y+1, season 0) … (y+1, season 11)
        //
        // y = 1990: needs (1990, 10/11) + (1991, 0..11) → all in [1990, 2010] ✓
        // y = 2009: needs (2009, 10/11) + (2010, 0..11) → all in [1990, 2010] ✓
        // y = 2010: needs (2010, 10/11) + (2011, 0..11) → 2011 not in data ✗
        let hydro1 = EntityId(1);
        let hydro2 = EntityId(2);
        let mut history = monthly_history(hydro1, 1990, 2010);
        history.extend(monthly_history(hydro2, 1990, 2010));

        let stages = twelve_monthly_stages();
        let windows =
            discover_historical_windows(&history, &[hydro1, hydro2], &stages, 2, None, 10).unwrap();

        let expected: Vec<i32> = (1990..=2009).collect();
        assert_eq!(windows, expected, "expected exactly years 1990–2009");
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
        let windows =
            discover_historical_windows(&history, &[hydro1, hydro2], &stages, 2, Some(&pool), 5)
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
        let windows =
            discover_historical_windows(&history, &[hydro1, hydro2], &stages, 2, Some(&pool), 5)
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
        let result =
            discover_historical_windows(&history, &[hydro1, hydro2], &stages, 2, Some(&pool), 1);

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
        // hydro2 is missing all of 2006 (year_offset+1 for window y=2005).
        // Window y=2005 requires (2005, seasons 10/11) and (2006, seasons 0..11).
        // hydro2 has no 2006 data → window 2005 must be excluded.
        let hydro1 = EntityId(1);
        let hydro2 = EntityId(2);
        let mut history = monthly_history(hydro1, 1990, 2010);

        // hydro2: full range except 2006
        history.extend(monthly_history(hydro2, 1990, 2005));
        history.extend(monthly_history(hydro2, 2007, 2010));

        let stages = twelve_monthly_stages();
        let windows =
            discover_historical_windows(&history, &[hydro1, hydro2], &stages, 2, None, 5).unwrap();

        assert!(
            !windows.contains(&2005),
            "window 2005 should be excluded because hydro2 lacks 2006 data"
        );
        // 2004 should still be valid: needs (2004, 10/11) + (2005, 0..11).
        // hydro2 has data through 2005 → 2004 is valid.
        assert!(windows.contains(&2004), "window 2004 should still be valid");
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
}
