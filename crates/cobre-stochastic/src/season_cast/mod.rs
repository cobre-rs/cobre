//! Cycle-generic season-period-window arithmetic (`Monthly`/`Weekly`/`Custom`)
//! and the [`StageCalendar`] Layer-2 resolver built on top of it.
//!
//! Resolves the concrete calendar `[start, end)` window and total duration for
//! a stage's season-period occurrence, and the next chronological occurrence.
//! The backward walk over those occurrences also feeds the stitched lag-season
//! lookup, [`StitchedSeasonMap`].

mod stitched;

pub use stitched::StitchedSeasonMap;

use chrono::{Datelike, NaiveDate, TimeDelta, Weekday};
use cobre_core::PostStudyStage;
use cobre_core::temporal::{
    Block, BlockMode, NoiseMethod, ScenarioSourceConfig, SeasonCycleType, SeasonCycles,
    SeasonDefinition, SeasonMap, Stage, StageRiskConfig, StageStateConfig, window_period_overlaps,
};

pub(crate) fn month_exclusive_end(year: i32, month: u32) -> NaiveDate {
    let (next_year, next_month) = if month == 12 {
        (year + 1, 1u32)
    } else {
        (year, month + 1)
    };
    NaiveDate::from_ymd_opt(next_year, next_month, 1)
        .unwrap_or_else(|| unreachable!("next-month date is always valid"))
}

/// Total hours in calendar month `month` (1–12) of `year`. Each day is exactly
/// 24 hours (timezone-free calendar dates, no DST).
pub(crate) fn month_total_hours(year: i32, month: u32) -> f64 {
    f64::from(days_in_month(year, month)) * 24.0
}

/// Determine the calendar year whose occurrence of `season_month` overlaps the
/// stage interval `[start_date, end_date)`, in a `Monthly` cycle.
///
/// Candidates are checked in order: `start_date.year()`, then the previous year
/// (a December-season stage starting in January), then the next year as a
/// fallback against unexpected gaps.
pub(crate) fn find_season_year_monthly(
    start_date: NaiveDate,
    end_date: NaiveDate,
    season_month: u32,
) -> i32 {
    let candidate_year = start_date.year();
    let period_start = NaiveDate::from_ymd_opt(candidate_year, season_month, 1)
        .unwrap_or_else(|| unreachable!("season month is always valid"));
    let period_end = month_exclusive_end(candidate_year, season_month);

    if start_date < period_end && end_date > period_start {
        return candidate_year;
    }

    let prev_year = candidate_year - 1;
    let period_start_prev = NaiveDate::from_ymd_opt(prev_year, season_month, 1)
        .unwrap_or_else(|| unreachable!("season month with previous year is always valid"));
    let period_end_prev = month_exclusive_end(prev_year, season_month);

    if start_date < period_end_prev && end_date > period_start_prev {
        return prev_year;
    }

    candidate_year + 1
}

/// Concrete `[start, end)` calendar window for one occurrence of a season
/// period, plus its total duration in hours.
#[derive(Clone)]
pub struct SeasonPeriodWindow {
    /// Inclusive window start.
    pub start: NaiveDate,
    /// Exclusive window end.
    pub end: NaiveDate,
    /// Total window duration in hours.
    pub hours: f64,
}

pub(crate) fn days_in_month(year: i32, month: u32) -> u32 {
    let first = NaiveDate::from_ymd_opt(year, month, 1)
        .unwrap_or_else(|| unreachable!("month-start date is always valid"));
    let next = month_exclusive_end(year, month);
    u32::try_from((next - first).num_days())
        .unwrap_or_else(|_| unreachable!("days in a month always fit in u32"))
}

/// Resolve a `Custom` `season_def`'s `[start, end)` range anchored in `year`.
///
/// Mirrors `season_for_date`'s `Custom` arm (`day_start`/`day_end` defaults,
/// `start <= end` wrap-around); `day_end` is clamped to the real month length
/// here because a concrete date must be constructed (the tuple comparison in
/// `season_for_date` needs no such clamp).
pub(crate) fn custom_period_bounds(
    year: i32,
    season_def: &SeasonDefinition,
) -> (NaiveDate, NaiveDate) {
    let month_start = season_def.month_start;
    let day_start = season_def.day_start.unwrap_or(1);
    let month_end = season_def.month_end.unwrap_or(month_start);
    let day_end = season_def.day_end.unwrap_or(31);

    let start_day = day_start.min(days_in_month(year, month_start));
    let start = NaiveDate::from_ymd_opt(year, month_start, start_day)
        .unwrap_or_else(|| unreachable!("clamped custom start date is always valid"));

    let wraps = (month_start, day_start) > (month_end, day_end);
    let end_year = if wraps { year + 1 } else { year };
    let end_day = day_end.min(days_in_month(end_year, month_end));
    let end_inclusive = NaiveDate::from_ymd_opt(end_year, month_end, end_day)
        .unwrap_or_else(|| unreachable!("clamped custom end date is always valid"));

    (start, end_inclusive + TimeDelta::days(1))
}

fn latest_custom_occurrence_ending_by(
    season_def: &SeasonDefinition,
    bound: NaiveDate,
) -> Option<SeasonPeriodWindow> {
    let year = bound.year();
    let (start, end) = [year, year - 1, year - 2]
        .into_iter()
        .map(|candidate| custom_period_bounds(candidate, season_def))
        .filter(|&(_, end)| end <= bound)
        .max_by_key(|&(_, end)| end)?;
    let days = u32::try_from((end - start).num_days()).ok()?;
    Some(SeasonPeriodWindow {
        start,
        end,
        hours: f64::from(days) * 24.0,
    })
}

/// Determine the calendar year whose occurrence of a `Custom` `season_def`
/// overlaps `[start_date, end_date)`. Generalizes
/// `find_season_year_monthly`'s candidate/previous-year/fallback search to a
/// day-level range.
pub(crate) fn find_season_year_custom(
    start_date: NaiveDate,
    end_date: NaiveDate,
    season_def: &SeasonDefinition,
) -> i32 {
    let candidate_year = start_date.year();
    let (period_start, period_end) = custom_period_bounds(candidate_year, season_def);
    if start_date < period_end && end_date > period_start {
        return candidate_year;
    }

    let prev_year = candidate_year - 1;
    let (period_start_prev, period_end_prev) = custom_period_bounds(prev_year, season_def);
    if start_date < period_end_prev && end_date > period_start_prev {
        return prev_year;
    }

    candidate_year + 1
}

/// Resolve `season_def`'s concrete calendar window for `stage`'s occurrence.
///
/// `season_for_date`'s week-53→52 fold is a season-id label fold, not a window
/// fold, so a `Weekly` window stays 7 real days regardless.
#[must_use]
pub fn season_period_window(
    season_map: &SeasonMap,
    season_def: &SeasonDefinition,
    stage: &Stage,
) -> SeasonPeriodWindow {
    match season_map.cycle_type {
        SeasonCycleType::Monthly => {
            let season_month = season_def.month_start;
            let year = find_season_year_monthly(stage.start_date, stage.end_date, season_month);
            let start = NaiveDate::from_ymd_opt(year, season_month, 1)
                .unwrap_or_else(|| unreachable!("season month is always valid"));
            let end = month_exclusive_end(year, season_month);
            let hours = month_total_hours(year, season_month);
            SeasonPeriodWindow { start, end, hours }
        }
        SeasonCycleType::Weekly => {
            let iso_week = stage.start_date.iso_week();
            let start = NaiveDate::from_isoywd_opt(iso_week.year(), iso_week.week(), Weekday::Mon)
                .unwrap_or_else(|| unreachable!("iso week start date is always valid"));
            let end = start + TimeDelta::days(7);
            SeasonPeriodWindow {
                start,
                end,
                hours: 7.0 * 24.0,
            }
        }
        SeasonCycleType::Custom => {
            let year = find_season_year_custom(stage.start_date, stage.end_date, season_def);
            let (start, end) = custom_period_bounds(year, season_def);
            let days = u32::try_from((end - start).num_days())
                .unwrap_or_else(|_| unreachable!("custom period day count always fits in u32"));
            SeasonPeriodWindow {
                start,
                end,
                hours: f64::from(days) * 24.0,
            }
        }
    }
}

/// Resolve the period window immediately following `current`, for forward
/// spillover accounting.
///
/// The `Custom` arm advances to the next `season_def` in id order, wrapping the
/// list: `season_map.seasons` is sorted by id, so that is the next entry.
#[must_use]
pub fn next_season_period_window(
    season_map: &SeasonMap,
    season_def: &SeasonDefinition,
    current: &SeasonPeriodWindow,
) -> Option<SeasonPeriodWindow> {
    match season_map.cycle_type {
        SeasonCycleType::Monthly => {
            let season_month = season_def.month_start;
            let year = current.start.year();
            let (next_year, next_month) = if season_month == 12 {
                (year + 1, 1u32)
            } else {
                (year, season_month + 1)
            };
            let start = current.end;
            let end = month_exclusive_end(next_year, next_month);
            let hours = month_total_hours(next_year, next_month);
            Some(SeasonPeriodWindow { start, end, hours })
        }
        SeasonCycleType::Weekly => {
            let start = current.end;
            let end = start + TimeDelta::days(7);
            Some(SeasonPeriodWindow {
                start,
                end,
                hours: 7.0 * 24.0,
            })
        }
        SeasonCycleType::Custom => {
            let pos = season_map
                .seasons
                .iter()
                .position(|s| s.id == season_def.id)?;
            let next_def = &season_map.seasons[(pos + 1) % season_map.seasons.len()];
            let probe_end = current.end + TimeDelta::days(1);
            let year = find_season_year_custom(current.end, probe_end, next_def);
            let (start, end) = custom_period_bounds(year, next_def);
            let days = u32::try_from((end - start).num_days())
                .unwrap_or_else(|_| unreachable!("custom period day count always fits in u32"));
            Some(SeasonPeriodWindow {
                start,
                end,
                hours: f64::from(days) * 24.0,
            })
        }
    }
}

/// Resolve the period window immediately preceding `current`, with the season
/// id the backward walk records for it. This is the walk's single step.
///
/// On `Monthly` and `Weekly` maps the step is the exact inverse of
/// [`next_season_period_window`]. On a `Custom` map it steps to the previous
/// season of the current season's resolution-level cycle in `cycles` and takes
/// the year from the dates: the window is that season's latest occurrence
/// ending at or before `current.start`.
#[must_use]
pub(crate) fn previous_occurrence(
    season_map: &SeasonMap,
    cycles: &SeasonCycles,
    current_id: usize,
    current: &SeasonPeriodWindow,
) -> Option<(usize, SeasonPeriodWindow)> {
    let season_def = season_map.seasons.iter().find(|s| s.id == current_id)?;
    let previous = match season_map.cycle_type {
        SeasonCycleType::Monthly => {
            let season_month = season_def.month_start;
            let year = current.start.year();
            let (prev_year, prev_month) = if season_month == 1 {
                (year - 1, 12u32)
            } else {
                (year, season_month - 1)
            };
            let start = NaiveDate::from_ymd_opt(prev_year, prev_month, 1)
                .unwrap_or_else(|| unreachable!("previous month is always valid"));
            let end = current.start;
            let hours = month_total_hours(prev_year, prev_month);
            SeasonPeriodWindow { start, end, hours }
        }
        SeasonCycleType::Weekly => {
            let end = current.start;
            let start = end - TimeDelta::days(7);
            SeasonPeriodWindow {
                start,
                end,
                hours: 7.0 * 24.0,
            }
        }
        SeasonCycleType::Custom => {
            let predecessor = cycles.predecessor(current_id)?;
            let predecessor_def = season_map.seasons.iter().find(|s| s.id == predecessor)?;
            let window = latest_custom_occurrence_ending_by(predecessor_def, current.start)?;
            return Some((predecessor, window));
        }
    };
    Some((season_map.season_for_date(previous.start)?, previous))
}

/// Walk [`previous_occurrence`] `k` times from `anchor` (`k == 0` returns `anchor`).
#[must_use]
pub fn nth_previous_occurrence(
    season_map: &SeasonMap,
    season_def: &SeasonDefinition,
    anchor: &SeasonPeriodWindow,
    k: usize,
) -> Option<SeasonPeriodWindow> {
    let cycles = SeasonCycles::new(season_map);
    let mut window = anchor.clone();
    let mut current_id = season_def.id;
    for _ in 0..k {
        (current_id, window) = previous_occurrence(season_map, &cycles, current_id, &window)?;
    }
    Some(window)
}

/// The calendar year identifying which occurrence of `season_def` overlaps
/// `[start, end)`, disambiguating repeated season ids across years.
///
/// `Weekly` uses the ISO week-numbering year (`iso_week().year()`), not the
/// calendar year of `start`: a week's Monday can fall in the prior December
/// (`from_isoywd_opt(2004, 1, Mon)` is 2003-12-29), so `start`'s calendar year
/// would misclassify that week as the prior year's.
pub(crate) fn occurrence_year(
    season_map: &SeasonMap,
    season_def: &SeasonDefinition,
    start: NaiveDate,
    end: NaiveDate,
) -> i32 {
    match season_map.cycle_type {
        SeasonCycleType::Monthly => find_season_year_monthly(start, end, season_def.month_start),
        SeasonCycleType::Weekly => start.iso_week().year(),
        SeasonCycleType::Custom => find_season_year_custom(start, end, season_def),
    }
}

/// The year identifying which occurrence of season `season_id` an observation
/// dated `date` belongs to: the ISO week-numbering year under a `Weekly` map,
/// and under a `Monthly` or `Custom` map the year the occurrence starts in, so
/// the days after 1 January of a season spanning it belong to the previous
/// year. Falls back to `date.year()` when `season_id` is unknown or, under a
/// `Monthly` or `Custom` map, `date` lies outside that season's own window.
#[must_use]
pub fn observation_occurrence_year(
    season_map: &SeasonMap,
    season_id: usize,
    date: NaiveDate,
) -> i32 {
    let Some(season_def) = season_map.seasons.iter().find(|s| s.id == season_id) else {
        return date.year();
    };
    let Some(next_day) = date.succ_opt() else {
        return date.year();
    };
    let in_own_window = match season_map.cycle_type {
        SeasonCycleType::Weekly => true,
        SeasonCycleType::Monthly | SeasonCycleType::Custom => {
            season_map.season_for_date(date) == Some(season_id)
        }
    };
    if in_own_window {
        occurrence_year(season_map, season_def, date, next_day)
    } else {
        date.year()
    }
}

/// The calendar year identifying which occurrence of `season_def` `stage`
/// belongs to. See [`occurrence_year`].
pub(crate) fn resolved_year(
    season_map: &SeasonMap,
    season_def: &SeasonDefinition,
    stage: &Stage,
) -> i32 {
    occurrence_year(season_map, season_def, stage.start_date, stage.end_date)
}

/// One hydro's realized-inflow window over `[start_date, end_date)`.
pub struct RealizedWindow {
    /// Inclusive window start.
    pub start_date: NaiveDate,
    /// Exclusive window end (`end_date > start_date`).
    pub end_date: NaiveDate,
    /// Realized inflow value in m3/s over the window.
    pub value_m3s: f64,
}

/// Day-weighted mean value and period-coverage fraction produced by [`cast`].
pub struct Projection {
    /// Coverage-normalized day-weighted mean of the overlapping windows' values.
    pub value: f64,
    /// Fraction of the period's hours covered by the input windows.
    pub coverage: f64,
}

fn overlap_hours(
    window_start: NaiveDate,
    window_end: NaiveDate,
    period_start: NaiveDate,
    period_end: NaiveDate,
) -> f64 {
    let overlap_start = window_start.max(period_start);
    let overlap_end = window_end.min(period_end);
    let days = u32::try_from((overlap_end - overlap_start).num_days().max(0))
        .unwrap_or_else(|_| unreachable!("overlap day count always fits in u32"));
    f64::from(days) * 24.0
}

/// Day-weighted projection of `windows` onto one `period` occurrence: the
/// coverage-normalized mean value over the overlap, and the covered fraction
/// of `period`.
///
/// `windows` must be non-overlapping — overlapping input double-counts shared
/// days into both `value` and `coverage`.
#[must_use]
pub fn cast(windows: &[RealizedWindow], period: &SeasonPeriodWindow) -> Projection {
    let mut overlap = 0.0_f64;
    let mut weighted_value = 0.0_f64;
    for window in windows {
        let hours = overlap_hours(window.start_date, window.end_date, period.start, period.end);
        overlap += hours;
        weighted_value += window.value_m3s * hours;
    }

    let value = if overlap > 0.0 {
        weighted_value / overlap
    } else {
        0.0
    };
    let coverage = overlap / period.hours;

    Projection { value, coverage }
}

/// Sorted, deduplicated day boundaries from `record` and `conditioning`
/// window edges — the maximal segments over which the covering window (if
/// any) in either layer cannot change.
fn window_boundaries(record: &[RealizedWindow], conditioning: &[RealizedWindow]) -> Vec<NaiveDate> {
    let mut points: Vec<NaiveDate> = record
        .iter()
        .chain(conditioning)
        .flat_map(|w| [w.start_date, w.end_date])
        .collect();
    points.sort_unstable();
    points.dedup();
    points
}

/// Value of the window in `windows` (assumed non-overlapping within the
/// slice) that covers `day` under the `[start_date, end_date)` convention.
fn covering_value(windows: &[RealizedWindow], day: NaiveDate) -> Option<f64> {
    windows
        .iter()
        .find(|w| w.start_date <= day && day < w.end_date)
        .map(|w| w.value_m3s)
}

/// Day-resolved, non-overlapping merge of `record` and `conditioning`:
/// `conditioning` shadows `record` on every day both cover (the conditioning
/// value replaces the record value; the two never blend), `record` alone
/// covers a day neither shadows, and a day covered by neither is absent from
/// the output. The result is sorted by `start_date` and satisfies `cast`'s
/// non-overlapping precondition.
#[must_use]
pub fn merge_layered_windows(
    record: &[RealizedWindow],
    conditioning: &[RealizedWindow],
) -> Vec<RealizedWindow> {
    let cut_points = window_boundaries(record, conditioning);
    let mut merged = Vec::with_capacity(cut_points.len().saturating_sub(1));

    for segment in cut_points.windows(2) {
        let (segment_start, segment_end) = (segment[0], segment[1]);
        let value = covering_value(conditioning, segment_start)
            .or_else(|| covering_value(record, segment_start));
        if let Some(value_m3s) = value {
            merged.push(RealizedWindow {
                start_date: segment_start,
                end_date: segment_end,
                value_m3s,
            });
        }
    }

    merged
}

/// A dated window's `[start_date, end_date)` extent, with no attached value —
/// the forward-coverage counterpart to [`RealizedWindow`].
pub struct DatedWindow {
    /// Inclusive window start.
    pub start_date: NaiveDate,
    /// Exclusive window end (`end_date > start_date`).
    pub end_date: NaiveDate,
}

// Rationale: calendar spans are on the order of centuries at most, far under
// f64's exact-integer range; a checked conversion buys nothing.
#[allow(clippy::cast_precision_loss)]
fn hours_between(later: NaiveDate, earlier: NaiveDate) -> f64 {
    (later - earlier).num_hours() as f64
}

/// Convert a post-study calendar segment (`PostStudyStages::stages`) into
/// [`Stage`]s a [`StageCalendar`] can resolve against.
///
/// `end_date = start_date + round(duration_hours / 24)` days — the whole-day
/// rounding the `cobre-io` semantic validator applies when it builds the same
/// segment, so this resolver and that validator agree on exactly which
/// post-study stage a window covers. Only `start_date`/`end_date`/`blocks` are
/// load-bearing (the fields [`StageCalendar`] reads); the rest are inert
/// placeholders.
#[must_use]
#[allow(clippy::cast_possible_truncation)] // a validated finite positive duration yields a small non-negative day count
pub fn post_study_calendar_stages(stages: &[PostStudyStage]) -> Vec<Stage> {
    stages
        .iter()
        .enumerate()
        .map(|(index, stage)| {
            let days = (stage.duration_hours / 24.0).round() as i64;
            let end_date = stage.start_date + TimeDelta::days(days);
            Stage {
                index,
                id: i32::try_from(index).unwrap_or(i32::MAX),
                start_date: stage.start_date,
                end_date,
                season_id: None,
                blocks: vec![Block {
                    index: 0,
                    name: "post_study".to_string(),
                    duration_hours: stage.duration_hours,
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
        })
        .collect()
}

/// Layer-2 resolver over the study's ordered, non-overlapping
/// `[start_date, end_date)` stage calendar: forward date-window coverage for
/// windowed inputs, and the two backward pre-study projections that season-
/// anchored and hour-anchored seeding each need.
pub struct StageCalendar<'a> {
    stages: &'a [Stage],
    stage_hours: Vec<f64>,
}

impl<'a> StageCalendar<'a> {
    /// Build a resolver over `stages`, the study's own chronologically
    /// ordered, non-overlapping calendar.
    #[must_use]
    pub fn new(stages: &'a [Stage]) -> Self {
        debug_assert!(
            stages.iter().all(|s| s.start_date < s.end_date),
            "every StageCalendar stage must have end_date after start_date"
        );
        debug_assert!(
            stages
                .windows(2)
                .all(|pair| pair[0].end_date <= pair[1].start_date),
            "StageCalendar stages must be chronologically ordered and non-overlapping"
        );
        let stage_hours = stages.iter().map(Stage::total_hours).collect();
        Self {
            stages,
            stage_hours,
        }
    }

    /// Forward per-study-stage overlap fraction of `window`: index `i` is the
    /// fraction of stage `i`'s own real `[start_date, end_date)` calendar span
    /// that `window` covers, `0.0` where `window` does not reach. Length
    /// always equals the calendar's stage count, padded with trailing zeros
    /// past `window`'s last overlapped stage. The basis is each stage's real
    /// date span, not its declared `duration_hours` — the two diverge on a
    /// nominal-month (730 h) calendar, where a whole-day window can never
    /// exactly tile a stage on the declared-hours clock.
    #[must_use]
    pub fn coverage(&self, window: &DatedWindow) -> Vec<f64> {
        debug_assert!(
            window.end_date > window.start_date,
            "DatedWindow end_date must be strictly after start_date"
        );
        let window_width_hours = hours_between(window.end_date, window.start_date);
        self.stages
            .iter()
            .map(|stage| {
                let stage_span_hours = hours_between(stage.end_date, stage.start_date);
                let window_start_hours = hours_between(window.start_date, stage.start_date);
                let overlap = window_period_overlaps(
                    window_start_hours,
                    window_width_hours,
                    &[stage_span_hours],
                )
                .first()
                .copied()
                .unwrap_or(0.0);
                overlap / stage_span_hours
            })
            .collect()
    }

    /// Whether `windows`, summed per stage, cover the leading `k` study
    /// stages at exactly `1.0` — no gap, no double-covered overlap. `k` must
    /// not exceed the calendar's stage count.
    #[must_use]
    #[allow(clippy::float_cmp)] // coverage's whole-day-hours arithmetic keeps a full-coverage ratio bit-exact
    pub fn covers_exactly(&self, windows: &[DatedWindow], k: usize) -> bool {
        debug_assert!(
            k <= self.stages.len(),
            "covers_exactly: k must not exceed the calendar's stage count"
        );
        let mut total = vec![0.0_f64; self.stages.len()];
        for window in windows {
            for (stage_total, fraction) in total.iter_mut().zip(self.coverage(window)) {
                *stage_total += fraction;
            }
        }
        total.iter().take(k).all(|&fraction| fraction == 1.0)
    }

    /// Backward season-occurrence projection: [`cast`]-projects `windows`
    /// onto the `k`-th previous occurrence of `season_def` before this
    /// calendar's own first stage (`k == 0` is that stage's in-progress
    /// occurrence). `None` when the calendar has no stages or the walk cannot
    /// resolve occurrence `k`.
    #[must_use]
    pub fn season_occurrence(
        &self,
        season_map: &SeasonMap,
        season_def: &SeasonDefinition,
        windows: &[RealizedWindow],
        k: usize,
    ) -> Option<Projection> {
        let anchor_stage = self.stages.first()?;
        let in_progress = season_period_window(season_map, season_def, anchor_stage);
        let occurrence = nth_previous_occurrence(season_map, season_def, &in_progress, k)?;
        Some(cast(windows, &occurrence))
    }

    /// Backward occurrence-window sequence: entry `0` is this calendar's own
    /// in-progress occurrence of `season_def`, entries `1..=max_k` are the
    /// consecutive results of walking [`previous_occurrence`], each
    /// computed exactly once — equivalent to calling [`Self::season_occurrence`]
    /// independently for every `k` in `0..=max_k`, but without restarting the
    /// walk from the anchor each time.
    ///
    /// `None` only when the calendar has no stages. A `Some` result may hold
    /// fewer than `max_k + 1` entries: [`nth_previous_occurrence`]'s per-step
    /// season lookup can fail partway through the chain, and every later step
    /// depends on it, so the walk stops there — the same `k`s an independent
    /// per-call [`Self::season_occurrence`] would have returned `None` for.
    #[must_use]
    pub fn season_occurrences(
        &self,
        season_map: &SeasonMap,
        season_def: &SeasonDefinition,
        max_k: usize,
    ) -> Option<Vec<SeasonPeriodWindow>> {
        let anchor_stage = self.stages.first()?;
        let cycles = SeasonCycles::new(season_map);
        let mut window = season_period_window(season_map, season_def, anchor_stage);
        let mut current_id = season_def.id;
        let mut occurrences = Vec::with_capacity(max_k + 1);
        occurrences.push(window.clone());

        for _ in 0..max_k {
            let Some((next_id, previous)) =
                previous_occurrence(season_map, &cycles, current_id, &window)
            else {
                break;
            };
            current_id = next_id;
            window = previous;
            occurrences.push(window.clone());
        }

        Some(occurrences)
    }

    /// Backward hour-window projection: the per-study-stage share of a
    /// `period_duration`-hour, fixed-rate pre-study window landing in each
    /// stage, where the window ends `cumulative_before` hours before an
    /// anchor `t_v` hours in the future — the window is
    /// `[t_v − cumulative_before − period_duration, t_v − cumulative_before)`.
    /// Shares are normalized by the WINDOW's own duration, not each stage's:
    /// they sum to `1.0` when the window falls entirely inside the calendar,
    /// less otherwise.
    #[must_use]
    pub fn hour_window_shares(
        &self,
        t_v: f64,
        cumulative_before: f64,
        period_duration: f64,
    ) -> Vec<f64> {
        let window_start = t_v - cumulative_before - period_duration;
        window_period_overlaps(window_start, period_duration, &self.stage_hours)
            .into_iter()
            .map(|overlap| overlap / period_duration)
            .collect()
    }

    /// Resolve `window`'s covering stage index: the one stage [`Self::coverage`]
    /// reports at fraction `1.0`, with every other stage at `0.0` — i.e.
    /// `window` aligns exactly to one calendar stage's boundaries. `None` when
    /// no stage covers it exactly (straddles stages, a partial-day
    /// misalignment, or falls off the segment).
    #[must_use]
    #[allow(clippy::float_cmp)] // coverage's whole-day-hours arithmetic keeps a full-coverage fraction bit-exact
    pub fn resolve_window(&self, window: &DatedWindow) -> Option<usize> {
        let mut resolved = None;
        for (idx, fraction) in self.coverage(window).into_iter().enumerate() {
            if fraction == 1.0 {
                if resolved.is_some() {
                    return None;
                }
                resolved = Some(idx);
            } else if fraction != 0.0 {
                return None;
            }
        }
        resolved
    }
}

#[cfg(test)]
mod tests {
    use super::{
        DatedWindow, RealizedWindow, SeasonCycleType, SeasonDefinition, SeasonMap,
        SeasonPeriodWindow, Stage, StageCalendar, cast, merge_layered_windows, month_exclusive_end,
        nth_previous_occurrence, observation_occurrence_year, post_study_calendar_stages,
        season_period_window,
    };
    use crate::test_support::{MonthlyLabels, monthly_season_map, weekly_season_map};
    use chrono::NaiveDate;
    use cobre_core::PostStudyStage;
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, StageRiskConfig, StageStateConfig,
    };

    fn twelve_month_season_map() -> SeasonMap {
        let seasons: Vec<SeasonDefinition> = (0..12u32)
            .map(|i| SeasonDefinition {
                id: i as usize,
                label: format!("Month{}", i + 1),
                month_start: i + 1,
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

    #[test]
    fn test_season_period_window_monthly_april_stage() {
        let season_map = twelve_month_season_map();
        let season_def = &season_map.seasons[3];

        let start = NaiveDate::from_ymd_opt(2026, 4, 4).unwrap();
        let end = NaiveDate::from_ymd_opt(2026, 5, 2).unwrap();
        let stage = Stage {
            index: 0,
            id: 0,
            start_date: start,
            end_date: end,
            season_id: Some(3),
            blocks: vec![Block {
                index: 0,
                name: "SINGLE".to_string(),
                duration_hours: 28.0 * 24.0,
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
        };

        let window = season_period_window(&season_map, season_def, &stage);

        assert_eq!(window.hours, 30.0 * 24.0);
        assert_eq!(window.start, NaiveDate::from_ymd_opt(2026, 4, 1).unwrap());
        assert_eq!(window.end, NaiveDate::from_ymd_opt(2026, 5, 1).unwrap());
    }

    #[test]
    fn test_previous_occurrence_monthly_walks_back() {
        let season_map = twelve_month_season_map();
        let season_def = &season_map.seasons[3];

        let start = NaiveDate::from_ymd_opt(2026, 4, 4).unwrap();
        let end = NaiveDate::from_ymd_opt(2026, 5, 2).unwrap();
        let stage = Stage {
            index: 0,
            id: 0,
            start_date: start,
            end_date: end,
            season_id: Some(3),
            blocks: vec![Block {
                index: 0,
                name: "SINGLE".to_string(),
                duration_hours: 28.0 * 24.0,
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
        };
        let anchor = season_period_window(&season_map, season_def, &stage);

        let march = nth_previous_occurrence(&season_map, season_def, &anchor, 1)
            .expect("k=1 must resolve to March 2026");
        assert_eq!(march.start, NaiveDate::from_ymd_opt(2026, 3, 1).unwrap());
        assert_eq!(march.end, NaiveDate::from_ymd_opt(2026, 4, 1).unwrap());
        assert_eq!(march.hours, 31.0 * 24.0);

        let february = nth_previous_occurrence(&season_map, season_def, &anchor, 2)
            .expect("k=2 must resolve to February 2026");
        assert_eq!(february.start, NaiveDate::from_ymd_opt(2026, 2, 1).unwrap());
        assert_eq!(february.end, NaiveDate::from_ymd_opt(2026, 3, 1).unwrap());
        assert_eq!(february.hours, 28.0 * 24.0);

        let january = nth_previous_occurrence(&season_map, season_def, &anchor, 3)
            .expect("k=3 must resolve to January 2026");
        assert_eq!(january.start, NaiveDate::from_ymd_opt(2026, 1, 1).unwrap());
        assert_eq!(january.end, NaiveDate::from_ymd_opt(2026, 2, 1).unwrap());
        assert_eq!(january.hours, 31.0 * 24.0);
    }

    #[test]
    fn test_nth_previous_occurrence_custom_year_wrap() {
        let seasons = vec![
            SeasonDefinition {
                id: 5,
                label: "June".to_string(),
                month_start: 6,
                day_start: None,
                month_end: None,
                day_end: None,
            },
            SeasonDefinition {
                id: 12,
                label: "Q3".to_string(),
                month_start: 7,
                day_start: None,
                month_end: Some(9),
                day_end: None,
            },
        ];
        let season_map = SeasonMap {
            cycle_type: SeasonCycleType::Custom,
            seasons,
        };
        let season_def = &season_map.seasons[1];

        let anchor = SeasonPeriodWindow {
            start: NaiveDate::from_ymd_opt(2024, 7, 1).unwrap(),
            end: NaiveDate::from_ymd_opt(2024, 10, 1).unwrap(),
            hours: 92.0 * 24.0,
        };

        let june_2024 = nth_previous_occurrence(&season_map, season_def, &anchor, 1)
            .expect("k=1 must resolve to June 2024");
        assert_eq!(
            june_2024.start,
            NaiveDate::from_ymd_opt(2024, 6, 1).unwrap()
        );
        assert_eq!(june_2024.end, NaiveDate::from_ymd_opt(2024, 7, 1).unwrap());

        let q3_2023 = nth_previous_occurrence(&season_map, season_def, &anchor, 2)
            .expect("k=2 must resolve to Q3 2023, not Q3 2024");
        assert_eq!(q3_2023.start, NaiveDate::from_ymd_opt(2023, 7, 1).unwrap());
        assert_eq!(q3_2023.end, NaiveDate::from_ymd_opt(2023, 10, 1).unwrap());
    }

    #[test]
    fn previous_occurrence_steps_a_year_spanning_season_back_in_time() {
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
            seasons: vec![season(0, 11, 2), season(1, 3, 10)],
        };
        let season_def = &season_map.seasons[1];

        let anchor = SeasonPeriodWindow {
            start: NaiveDate::from_ymd_opt(2024, 3, 1).unwrap(),
            end: NaiveDate::from_ymd_opt(2024, 11, 1).unwrap(),
            hours: 245.0 * 24.0,
        };

        let nov_feb = nth_previous_occurrence(&season_map, season_def, &anchor, 1)
            .expect("k=1 must resolve to Nov 2023-Feb 2024");
        assert_eq!(nov_feb.start, NaiveDate::from_ymd_opt(2023, 11, 1).unwrap());
        assert_eq!(nov_feb.end, NaiveDate::from_ymd_opt(2024, 3, 1).unwrap());
        assert_eq!(nov_feb.hours, 121.0 * 24.0);

        let mar_oct = nth_previous_occurrence(&season_map, season_def, &anchor, 2)
            .expect("k=2 must resolve to Mar-Oct 2023");
        assert_eq!(mar_oct.start, NaiveDate::from_ymd_opt(2023, 3, 1).unwrap());
        assert_eq!(mar_oct.end, NaiveDate::from_ymd_opt(2023, 11, 1).unwrap());
        assert_eq!(mar_oct.hours, 245.0 * 24.0);
    }

    #[test]
    fn test_nth_previous_occurrence_k_zero_is_identity() {
        let season_map = twelve_month_season_map();
        let season_def = &season_map.seasons[3];
        let anchor = SeasonPeriodWindow {
            start: NaiveDate::from_ymd_opt(2026, 4, 1).unwrap(),
            end: NaiveDate::from_ymd_opt(2026, 5, 1).unwrap(),
            hours: 30.0 * 24.0,
        };

        let result = nth_previous_occurrence(&season_map, season_def, &anchor, 0)
            .expect("k=0 must return anchor");

        assert_eq!(result.start, anchor.start);
        assert_eq!(result.end, anchor.end);
        assert_eq!(result.hours, anchor.hours);
    }

    #[test]
    fn observation_occurrence_year_names_the_occurrence_a_dated_observation_belongs_to() {
        let day = |y: i32, m: u32, d: u32| NaiveDate::from_ymd_opt(y, m, d).unwrap();
        let season = |id: usize, month_start: u32, month_end: u32| SeasonDefinition {
            id,
            label: format!("S{id}"),
            month_start,
            day_start: None,
            month_end: Some(month_end),
            day_end: None,
        };
        let weekly = weekly_season_map();
        let custom = SeasonMap {
            cycle_type: SeasonCycleType::Custom,
            seasons: vec![
                season(0, 3, 5),
                season(1, 6, 8),
                season(2, 9, 11),
                season(3, 12, 2),
            ],
        };
        let monthly = monthly_season_map(MonthlyLabels::ZeroBased);

        let cases = [
            (&weekly, 0, day(2013, 12, 30), 2014),
            (&weekly, 0, day(2014, 12, 29), 2015),
            (&weekly, 51, day(2016, 1, 3), 2015),
            (&custom, 3, day(2020, 12, 1), 2020),
            (&custom, 3, day(2021, 2, 15), 2020),
            (&custom, 0, day(2021, 4, 1), 2021),
            (&custom, 3, day(2021, 4, 1), 2021),
            (&custom, 99, day(2021, 4, 1), 2021),
            (&monthly, 11, day(2020, 12, 15), 2020),
            (&monthly, 0, day(2021, 2, 1), 2021),
        ];
        for (season_map, season_id, date, want) in cases {
            assert_eq!(
                observation_occurrence_year(season_map, season_id, date),
                want,
                "{:?} season {season_id} on {date}",
                season_map.cycle_type
            );
        }
    }

    fn april_2026() -> SeasonPeriodWindow {
        SeasonPeriodWindow {
            start: NaiveDate::from_ymd_opt(2026, 4, 1).unwrap(),
            end: NaiveDate::from_ymd_opt(2026, 5, 1).unwrap(),
            hours: 30.0 * 24.0,
        }
    }

    #[test]
    fn test_cast_full_coverage_returns_window_value() {
        let period = april_2026();
        let windows = [RealizedWindow {
            start_date: period.start,
            end_date: period.end,
            value_m3s: 480.0,
        }];

        let projection = cast(&windows, &period);

        assert_eq!(projection.value, 480.0);
        assert_eq!(projection.coverage, 1.0);
    }

    #[test]
    fn test_cast_partition_day_weighted_mean() {
        let period = april_2026();
        let windows = [
            RealizedWindow {
                start_date: NaiveDate::from_ymd_opt(2026, 4, 1).unwrap(),
                end_date: NaiveDate::from_ymd_opt(2026, 4, 4).unwrap(),
                value_m3s: 500.0,
            },
            RealizedWindow {
                start_date: NaiveDate::from_ymd_opt(2026, 4, 4).unwrap(),
                end_date: NaiveDate::from_ymd_opt(2026, 5, 1).unwrap(),
                value_m3s: 480.0,
            },
        ];

        let projection = cast(&windows, &period);

        assert_eq!(projection.value, (500.0 * 72.0 + 480.0 * 648.0) / 720.0);
        assert_eq!(projection.coverage, 1.0);
    }

    #[test]
    fn test_cast_straddling_window_split_once() {
        let april = april_2026();
        let may = SeasonPeriodWindow {
            start: NaiveDate::from_ymd_opt(2026, 5, 1).unwrap(),
            end: NaiveDate::from_ymd_opt(2026, 6, 1).unwrap(),
            hours: 31.0 * 24.0,
        };
        let straddling = [RealizedWindow {
            start_date: NaiveDate::from_ymd_opt(2026, 4, 25).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2026, 5, 8).unwrap(),
            value_m3s: 350.0,
        }];

        let april_projection = cast(&straddling, &april);
        let may_projection = cast(&straddling, &may);

        assert_eq!(april_projection.coverage, 144.0 / 720.0);
        assert_eq!(may_projection.coverage, 168.0 / 744.0);
        assert_eq!(april_projection.value, 350.0);
        assert_eq!(may_projection.value, 350.0);
    }

    #[test]
    fn test_straddling_window_contributes_each_day_once() {
        let april = april_2026();
        let may = SeasonPeriodWindow {
            start: NaiveDate::from_ymd_opt(2026, 5, 1).unwrap(),
            end: NaiveDate::from_ymd_opt(2026, 6, 1).unwrap(),
            hours: 31.0 * 24.0,
        };
        let window_start = NaiveDate::from_ymd_opt(2026, 4, 25).unwrap();
        let window_end = NaiveDate::from_ymd_opt(2026, 5, 8).unwrap();
        let value = 350.0;
        let straddling = [RealizedWindow {
            start_date: window_start,
            end_date: window_end,
            value_m3s: value,
        }];
        let window_total_hours =
            f64::from(u32::try_from((window_end - window_start).num_days()).unwrap()) * 24.0;

        let april_projection = cast(&straddling, &april);
        let may_projection = cast(&straddling, &may);

        let april_overlap_hours = april_projection.coverage * april.hours;
        let may_overlap_hours = may_projection.coverage * may.hours;
        assert_eq!(
            april_overlap_hours + may_overlap_hours,
            window_total_hours,
            "each day of the straddling window must contribute to exactly one \
             period: overlap hours must sum to the window's total, no double-count"
        );

        let april_partial = april_overlap_hours * april_projection.value;
        let may_partial = may_overlap_hours * may_projection.value;
        assert_eq!(
            april_partial + may_partial,
            window_total_hours * value,
            "the two per-period partial values must sum to the window's \
             day-weighted total"
        );
    }

    #[test]
    fn test_cast_out_of_period_window_contributes_zero() {
        let period = april_2026();
        let march_window = [RealizedWindow {
            start_date: NaiveDate::from_ymd_opt(2026, 3, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2026, 4, 1).unwrap(),
            value_m3s: 999.0,
        }];

        let projection = cast(&march_window, &period);

        assert_eq!(projection.value, 0.0);
        assert_eq!(projection.coverage, 0.0);
    }

    #[test]
    fn test_merge_no_conditioning_preserves_record_cast() {
        let period = april_2026();
        let record = [RealizedWindow {
            start_date: period.start,
            end_date: period.end,
            value_m3s: 480.0,
        }];

        let merged = merge_layered_windows(&record, &[]);
        let projection = cast(&merged, &period);

        assert_eq!(projection.value, 480.0);
        assert_eq!(projection.coverage, 1.0);
    }

    #[test]
    fn test_merge_partial_shadow_blends_by_day() {
        let period = april_2026();
        let record = [RealizedWindow {
            start_date: period.start,
            end_date: period.end,
            value_m3s: 480.0,
        }];
        let conditioning = [RealizedWindow {
            start_date: NaiveDate::from_ymd_opt(2026, 4, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2026, 4, 4).unwrap(),
            value_m3s: 500.0,
        }];

        let merged = merge_layered_windows(&record, &conditioning);
        let projection = cast(&merged, &period);

        assert_eq!(projection.value, (500.0 * 72.0 + 480.0 * 648.0) / 720.0);
        assert_eq!(projection.coverage, 1.0);
    }

    fn stage_with_width(index: usize, start: NaiveDate, width_days: i64) -> Stage {
        let end = start + chrono::TimeDelta::days(width_days);
        Stage {
            index,
            id: i32::try_from(index).unwrap(),
            start_date: start,
            end_date: end,
            season_id: None,
            blocks: vec![Block {
                index: 0,
                name: "SINGLE".to_string(),
                duration_hours: f64::from(u32::try_from(width_days).unwrap()) * 24.0,
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

    fn six_stage_calendar() -> Vec<Stage> {
        let widths_days = [10, 20, 5, 15, 30, 10];
        let mut cursor = NaiveDate::from_ymd_opt(2026, 1, 1).unwrap();
        widths_days
            .iter()
            .enumerate()
            .map(|(i, &width_days)| {
                let stage = stage_with_width(i, cursor, width_days);
                cursor = stage.end_date;
                stage
            })
            .collect()
    }

    #[test]
    fn test_coverage_full_leading_two_stages() {
        let stages = six_stage_calendar();
        let calendar = StageCalendar::new(&stages);
        let window = DatedWindow {
            start_date: stages[0].start_date,
            end_date: stages[1].end_date,
        };

        let fractions = calendar.coverage(&window);

        assert_eq!(fractions, vec![1.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
        assert!(calendar.covers_exactly(&[window], 2));
    }

    #[test]
    fn test_covers_exactly_false_on_gap() {
        let stages = six_stage_calendar();
        let calendar = StageCalendar::new(&stages);
        let stage_0_window = DatedWindow {
            start_date: stages[0].start_date,
            end_date: stages[0].end_date,
        };
        let stage_2_window = DatedWindow {
            start_date: stages[2].start_date,
            end_date: stages[2].end_date,
        };

        assert!(!calendar.covers_exactly(&[stage_0_window, stage_2_window], 3));
    }

    /// A 730 h nominal-month `Stage`: the real span is the month's actual
    /// calendar length, which no real month makes equal to the flat 730 h
    /// declared block.
    fn nominal_month_stage(index: usize, year: i32, month: u32) -> Stage {
        let start = NaiveDate::from_ymd_opt(year, month, 1).unwrap();
        let end = month_exclusive_end(year, month);
        Stage {
            index,
            id: i32::try_from(index).unwrap(),
            start_date: start,
            end_date: end,
            season_id: None,
            blocks: vec![Block {
                index: 0,
                name: "SINGLE".to_string(),
                duration_hours: 730.0,
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

    /// Jan/Feb/Mar 2024: real spans 744 h / 696 h / 744 h, none equal to 730 h.
    fn nominal_month_calendar() -> Vec<Stage> {
        vec![
            nominal_month_stage(0, 2024, 1),
            nominal_month_stage(1, 2024, 2),
            nominal_month_stage(2, 2024, 3),
        ]
    }

    #[test]
    fn test_coverage_nominal_month_tiles_by_real_date_span() {
        let stages = nominal_month_calendar();
        let calendar = StageCalendar::new(&stages);
        let jan_window = || DatedWindow {
            start_date: stages[0].start_date,
            end_date: stages[0].end_date,
        };
        let feb_window = || DatedWindow {
            start_date: stages[1].start_date,
            end_date: stages[1].end_date,
        };
        let mar_window = || DatedWindow {
            start_date: stages[2].start_date,
            end_date: stages[2].end_date,
        };

        assert_eq!(
            calendar.coverage(&jan_window()),
            vec![1.0, 0.0, 0.0],
            "a 730h-declared stage tiled by its own real date range resolves \
             to 1.0, not the declared-hours fraction"
        );
        assert!(calendar.covers_exactly(&[jan_window(), feb_window()], 2));

        assert!(
            !calendar.covers_exactly(&[jan_window()], 2),
            "gap: February left uncovered"
        );
        assert!(
            !calendar.covers_exactly(&[jan_window(), jan_window()], 2),
            "overlap: January double-covered, February left uncovered"
        );
        assert!(
            !calendar.covers_exactly(&[jan_window(), mar_window()], 2),
            "beyond-K_i: the second window targets March (index 2) instead of \
             February, leaving the leading stage uncovered"
        );
    }

    fn two_post_study_stages() -> Vec<PostStudyStage> {
        vec![
            PostStudyStage {
                start_date: NaiveDate::from_ymd_opt(2026, 11, 1).unwrap(),
                duration_hours: 720.0,
            },
            PostStudyStage {
                start_date: NaiveDate::from_ymd_opt(2026, 12, 1).unwrap(),
                duration_hours: 744.0,
            },
        ]
    }

    #[test]
    fn test_post_study_calendar_stages_end_date_rounds_duration_to_days() {
        let stages = post_study_calendar_stages(&two_post_study_stages());

        assert_eq!(stages.len(), 2);
        assert_eq!(
            stages[0].start_date,
            NaiveDate::from_ymd_opt(2026, 11, 1).unwrap()
        );
        assert_eq!(
            stages[0].end_date,
            NaiveDate::from_ymd_opt(2026, 12, 1).unwrap()
        );
        assert_eq!(
            stages[1].start_date,
            NaiveDate::from_ymd_opt(2026, 12, 1).unwrap()
        );
        assert_eq!(
            stages[1].end_date,
            NaiveDate::from_ymd_opt(2027, 1, 1).unwrap()
        );

        // A duration that is not an exact day multiple (700h = 29.1(6) days)
        // rounds to the nearest whole day, mirroring the `cobre-io` validator's
        // own `post_study_end_date` convention.
        let odd = vec![PostStudyStage {
            start_date: NaiveDate::from_ymd_opt(2026, 11, 1).unwrap(),
            duration_hours: 700.0,
        }];
        let odd_stages = post_study_calendar_stages(&odd);
        assert_eq!(
            odd_stages[0].end_date,
            NaiveDate::from_ymd_opt(2026, 11, 30).unwrap()
        );
    }

    #[test]
    fn test_resolve_window_spanning_a_stage_resolves_to_it() {
        let stages = post_study_calendar_stages(&two_post_study_stages());
        let calendar = StageCalendar::new(&stages);

        let stage_1_window = DatedWindow {
            start_date: NaiveDate::from_ymd_opt(2026, 12, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2027, 1, 1).unwrap(),
        };
        assert_eq!(calendar.resolve_window(&stage_1_window), Some(1));

        let straddling = DatedWindow {
            start_date: NaiveDate::from_ymd_opt(2026, 11, 15).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2026, 12, 15).unwrap(),
        };
        assert_eq!(calendar.resolve_window(&straddling), None);

        let partial = DatedWindow {
            start_date: NaiveDate::from_ymd_opt(2026, 11, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2026, 11, 15).unwrap(),
        };
        assert_eq!(calendar.resolve_window(&partial), None);
    }

    #[test]
    #[allow(clippy::float_cmp)] // both paths run the identical arithmetic; the comparison must be bit-exact
    fn test_season_occurrence_matches_pre_refactor_helper_sequence() {
        let season_map = twelve_month_season_map();
        let season_def = &season_map.seasons[3];
        let first_stage = stage_with_width(0, NaiveDate::from_ymd_opt(2026, 4, 4).unwrap(), 28);
        let stages = [first_stage];
        let windows = [RealizedWindow {
            start_date: NaiveDate::from_ymd_opt(2026, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2026, 5, 1).unwrap(),
            value_m3s: 250.0,
        }];
        let calendar = StageCalendar::new(&stages);

        for k in 0..=3 {
            let expected_in_progress = season_period_window(&season_map, season_def, &stages[0]);
            let expected_occurrence =
                nth_previous_occurrence(&season_map, season_def, &expected_in_progress, k).unwrap();
            let expected = cast(&windows, &expected_occurrence);

            let actual = calendar
                .season_occurrence(&season_map, season_def, &windows, k)
                .unwrap();

            assert_eq!(actual.value, expected.value);
            assert_eq!(actual.coverage, expected.coverage);
        }
    }

    #[test]
    #[allow(clippy::float_cmp)] // both paths run the identical arithmetic; the comparison must be bit-exact
    fn test_season_occurrences_matches_per_k_restart_sequence() {
        let season_map = twelve_month_season_map();
        let season_def = &season_map.seasons[3];
        let first_stage = stage_with_width(0, NaiveDate::from_ymd_opt(2026, 4, 4).unwrap(), 28);
        let stages = [first_stage];
        let calendar = StageCalendar::new(&stages);
        let max_k = 5;

        let occurrences = calendar
            .season_occurrences(&season_map, season_def, max_k)
            .unwrap();
        assert_eq!(occurrences.len(), max_k + 1);

        let anchor = season_period_window(&season_map, season_def, &stages[0]);
        for (k, occurrence) in occurrences.iter().enumerate() {
            let expected = nth_previous_occurrence(&season_map, season_def, &anchor, k).unwrap();
            assert_eq!(occurrence.start, expected.start);
            assert_eq!(occurrence.end, expected.end);
            assert_eq!(occurrence.hours, expected.hours);
        }
    }

    mod cast_proptests {
        use proptest::prelude::*;

        use super::super::{RealizedWindow, SeasonPeriodWindow, cast};
        use chrono::{NaiveDate, TimeDelta};

        proptest! {
            #[test]
            fn cast_partition_matches_day_weighted_mean(
                segments in prop::collection::vec((1_u32..=15, -1000.0_f64..1000.0_f64), 1..=12),
            ) {
                let period_start = NaiveDate::from_ymd_opt(2026, 1, 1).unwrap();
                let mut windows = Vec::with_capacity(segments.len());
                let mut cursor = period_start;
                let mut weighted_hours_sum = 0.0_f64;
                let mut total_hours = 0.0_f64;
                for &(days, value) in &segments {
                    let end = cursor + TimeDelta::days(i64::from(days));
                    let hours = f64::from(days) * 24.0;
                    windows.push(RealizedWindow {
                        start_date: cursor,
                        end_date: end,
                        value_m3s: value,
                    });
                    weighted_hours_sum += value * hours;
                    total_hours += hours;
                    cursor = end;
                }
                let period = SeasonPeriodWindow {
                    start: period_start,
                    end: cursor,
                    hours: total_hours,
                };
                let expected_value = weighted_hours_sum / total_hours;

                let projection = cast(&windows, &period);

                prop_assert!((projection.value - expected_value).abs() < 1e-9);
                prop_assert!((projection.coverage - 1.0).abs() < 1e-9);
            }
        }
    }

    mod merge_proptests {
        use proptest::prelude::*;

        use super::super::{RealizedWindow, SeasonPeriodWindow, cast, merge_layered_windows};
        use chrono::{NaiveDate, TimeDelta};

        proptest! {
            #[test]
            fn merge_disjoint_windows_matches_concatenated_cast(
                segments in prop::collection::vec((any::<bool>(), 1_u32..=10, -500.0_f64..500.0_f64), 1..=16),
            ) {
                let period_start = NaiveDate::from_ymd_opt(2026, 1, 1).unwrap();
                let mut record = Vec::new();
                let mut conditioning = Vec::new();
                let mut cursor = period_start;
                let mut total_hours = 0.0_f64;
                for &(is_conditioning, days, value) in &segments {
                    let end = cursor + TimeDelta::days(i64::from(days));
                    let window = RealizedWindow {
                        start_date: cursor,
                        end_date: end,
                        value_m3s: value,
                    };
                    if is_conditioning {
                        conditioning.push(window);
                    } else {
                        record.push(window);
                    }
                    total_hours += f64::from(days) * 24.0;
                    cursor = end;
                }
                let period = SeasonPeriodWindow {
                    start: period_start,
                    end: cursor,
                    hours: total_hours,
                };

                let merged = merge_layered_windows(&record, &conditioning);
                let mut concatenated = Vec::with_capacity(record.len() + conditioning.len());
                concatenated.extend(record);
                concatenated.extend(conditioning);

                let merged_projection = cast(&merged, &period);
                let concatenated_projection = cast(&concatenated, &period);

                prop_assert!((merged_projection.value - concatenated_projection.value).abs() < 1e-9);
                prop_assert!(
                    (merged_projection.coverage - concatenated_projection.coverage).abs() < 1e-9
                );
            }
        }
    }
}
