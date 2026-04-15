//! Precomputation of per-stage lag accumulation weights and period
//! finalization flags from stage date boundaries and season definitions.
//!
//! The [`precompute_stage_lag_transitions`] function runs once at setup time
//! and produces a [`Vec<StageLagTransition>`] indexed by stage. The resulting
//! slice is consumed read-only on the hot path, eliminating calendar arithmetic
//! from inner solver loops.
//!
//! See [Design Doc — Temporal Resolution Debts §6](../../docs/design/temporal-resolution-debts.md).

// All public items in this module are used starting in Epic 2 (ticket-003).
// Until then, suppress dead_code so the crate compiles cleanly with -D warnings.
#![allow(dead_code)]

use std::collections::HashMap;

use chrono::{Datelike, NaiveDate};
use cobre_core::{
    entities::hydro::Hydro,
    initial_conditions::RecentObservation,
    temporal::{SeasonCycleType, SeasonDefinition, SeasonMap, Stage, StageLagTransition},
};

/// Pre-computed seed values for the lag accumulator, derived from
/// [`RecentObservation`] data in [`cobre_core::InitialConditions`].
///
/// Computed once at setup time by [`compute_recent_observation_seed`] and
/// stored in [`crate::setup::StudySetup`]. Applied at every trajectory start
/// (forward pass and simulation pipeline) instead of zero-filling the
/// accumulator.
///
/// When `weight_seed == 0.0` (no observations or non-Monthly season cycle),
/// the behavior is identical to the previous zero-reset.
#[derive(Debug, Clone)]
pub struct RecentObservationSeed {
    /// Per-hydro accumulated `value_m3s * observation_hours` values.
    ///
    /// Length equals `hydro_count`. Zero for hydros without observations.
    pub accum_seed: Vec<f64>,
    /// Fraction of the lag period covered by pre-study observations.
    ///
    /// Computed as `total_observation_hours / total_period_hours`. A single
    /// scalar because all observations share the same calendar period.
    pub weight_seed: f64,
}

impl RecentObservationSeed {
    /// Construct an all-zero seed for `hydro_count` hydros.
    #[must_use]
    pub fn zero(hydro_count: usize) -> Self {
        Self {
            accum_seed: vec![0.0_f64; hydro_count],
            weight_seed: 0.0,
        }
    }
}

/// Compute the lag accumulator seed from pre-study [`RecentObservation`] data.
///
/// Runs once at setup time. Returns a [`RecentObservationSeed`] whose
/// [`accum_seed`](RecentObservationSeed::accum_seed) and
/// [`weight_seed`](RecentObservationSeed::weight_seed) values are applied at
/// every trajectory start.
///
/// # Behavior by cycle type
///
/// - **`Monthly`**: lag-period boundaries are calendar month boundaries derived
///   from the first study stage's `season_id` and `start_date`.
/// - **`Weekly`** and **`Custom`**: not yet implemented; returns a zero seed.
///
/// Returns a zero seed when:
/// - `recent_obs` is empty.
/// - `first_stage.season_id` is `None`.
/// - The season cycle type is not `Monthly`.
/// - `hydros` is empty.
///
/// Unknown `hydro_id` values (not found in the `hydros` registry) are silently
/// skipped, matching the pattern in `build_initial_state`.
pub(crate) fn compute_recent_observation_seed(
    recent_obs: &[RecentObservation],
    first_stage: &Stage,
    season_map: &SeasonMap,
    hydros: &[Hydro],
) -> RecentObservationSeed {
    let hydro_count = hydros.len();
    if recent_obs.is_empty() || hydro_count == 0 {
        return RecentObservationSeed::zero(hydro_count);
    }

    let Some(season_id) = first_stage.season_id else {
        return RecentObservationSeed::zero(hydro_count);
    };

    if !matches!(season_map.cycle_type, SeasonCycleType::Monthly) {
        return RecentObservationSeed::zero(hydro_count);
    }

    let Some(season_def) = season_map.seasons.iter().find(|s| s.id == season_id) else {
        return RecentObservationSeed::zero(hydro_count);
    };

    let season_month = season_def.month_start;
    let year = find_season_year_monthly(first_stage.start_date, first_stage.end_date, season_month);
    let total_period_hours = month_total_hours(year, season_month);

    let mut accum_seed = vec![0.0_f64; hydro_count];
    let mut total_obs_hours = 0.0_f64;

    for obs in recent_obs {
        // Silently skip unknown hydro IDs, same as build_initial_state.
        let Ok(idx) = hydros.binary_search_by_key(&obs.hydro_id.0, |h| h.id.0) else {
            continue;
        };
        let obs_days = (obs.end_date - obs.start_date).num_days();
        let obs_hours = f64::from(
            u32::try_from(obs_days)
                .unwrap_or_else(|_| unreachable!("observation days always fit in u32")),
        ) * 24.0;
        accum_seed[idx] += obs.value_m3s * obs_hours;
        total_obs_hours += obs_hours;
    }

    let weight_seed = total_obs_hours / total_period_hours;

    RecentObservationSeed {
        accum_seed,
        weight_seed,
    }
}

/// Compute the exclusive end date of the calendar month identified by
/// `month` (1–12) and `year`.
pub(crate) fn month_exclusive_end(year: i32, month: u32) -> NaiveDate {
    let (next_year, next_month) = if month == 12 {
        (year + 1, 1u32)
    } else {
        (year, month + 1)
    };
    // next_month is always in 1..=12 and day 1 always exists.
    NaiveDate::from_ymd_opt(next_year, next_month, 1)
        .unwrap_or_else(|| unreachable!("next-month date is always valid"))
}

/// Returns the total hours in the calendar month identified by `year` and
/// `month` (1–12). Each day is exactly 24 hours (timezone-free calendar dates, no DST).
pub(crate) fn month_total_hours(year: i32, month: u32) -> f64 {
    let first = NaiveDate::from_ymd_opt(year, month, 1)
        .unwrap_or_else(|| unreachable!("month-start date is always valid"));
    let next = month_exclusive_end(year, month);
    // num_days() returns an i64; days in a month always fit in u32.
    let days = u32::try_from((next - first).num_days())
        .unwrap_or_else(|_| unreachable!("days in a month always fit in u32"));
    f64::from(days) * 24.0
}

/// Determine the calendar year for the lag period of a stage in a `Monthly`
/// cycle.
///
/// The stage's `season_id` maps to a calendar month via `season_def.month_start`.
/// We find which year's occurrence of that month overlaps the stage interval
/// `[start_date, end_date)`.
///
/// Two candidates are checked in order: `start_date.year()` (the common case
/// and the pre-study case where the stage starts one month before its season),
/// then `start_date.year() - 1` (for a December-season stage starting in
/// January of the next year).
pub(crate) fn find_season_year_monthly(
    start_date: NaiveDate,
    end_date: NaiveDate,
    season_month: u32,
) -> i32 {
    let candidate_year = start_date.year();
    let period_start = NaiveDate::from_ymd_opt(candidate_year, season_month, 1)
        .unwrap_or_else(|| unreachable!("season month is always valid"));
    let period_end = month_exclusive_end(candidate_year, season_month);

    // Overlap condition: stage_start < period_end AND stage_end > period_start.
    if start_date < period_end && end_date > period_start {
        return candidate_year;
    }

    // Try previous year (December-season stage starting in January).
    let prev_year = candidate_year - 1;
    let period_start_prev = NaiveDate::from_ymd_opt(prev_year, season_month, 1)
        .unwrap_or_else(|| unreachable!("season month with previous year is always valid"));
    let period_end_prev = month_exclusive_end(prev_year, season_month);

    if start_date < period_end_prev && end_date > period_start_prev {
        return prev_year;
    }

    // Fallback: try next year (guards against unexpected gaps).
    candidate_year + 1
}

/// Count the number of days in `[stage_start, stage_end)` that fall within
/// `[period_start, period_end)`. Returns 0 if there is no overlap.
pub(crate) fn days_in_period(
    stage_start: NaiveDate,
    stage_end: NaiveDate,
    period_start: NaiveDate,
    period_end: NaiveDate,
) -> u32 {
    let overlap_start = stage_start.max(period_start);
    let overlap_end = stage_end.min(period_end);
    if overlap_end > overlap_start {
        u32::try_from((overlap_end - overlap_start).num_days())
            .unwrap_or_else(|_| unreachable!("overlap days always fit in u32"))
    } else {
        0
    }
}

/// Compute the [`StageLagTransition`] for a single stage in a `Monthly`
/// season cycle.
pub(crate) fn compute_monthly_transition(
    stage: &Stage,
    season_def: &SeasonDefinition,
    all_stages: &[Stage],
) -> StageLagTransition {
    let season_month = season_def.month_start;
    let year = find_season_year_monthly(stage.start_date, stage.end_date, season_month);

    let period_start = NaiveDate::from_ymd_opt(year, season_month, 1)
        .unwrap_or_else(|| unreachable!("season month is always valid"));
    let period_end = month_exclusive_end(year, season_month);
    let period_hours = month_total_hours(year, season_month);

    let days_current = days_in_period(stage.start_date, stage.end_date, period_start, period_end);
    let accumulate_weight = f64::from(days_current) * 24.0 / period_hours;

    // Spillover: days that fall within the next calendar month.
    let next_period_start = period_end;
    let (next_year, next_month) = if season_month == 12 {
        (year + 1, 1u32)
    } else {
        (year, season_month + 1)
    };
    let next_period_end = month_exclusive_end(next_year, next_month);
    let next_period_hours = month_total_hours(next_year, next_month);

    let days_next = days_in_period(
        stage.start_date,
        stage.end_date,
        next_period_start,
        next_period_end,
    );
    let spillover_weight = if days_next > 0 {
        f64::from(days_next) * 24.0 / next_period_hours
    } else {
        0.0
    };

    // finalize_period: true when no later stage has the same (season_id, year).
    let season_id = season_def.id;
    let is_last_in_period = all_stages
        .iter()
        .skip(stage.index + 1)
        .filter(|s| s.season_id == Some(season_id))
        .all(|s| find_season_year_monthly(s.start_date, s.end_date, season_month) != year);

    StageLagTransition {
        accumulate_weight,
        spillover_weight,
        finalize_period: is_last_in_period,
        accumulate_downstream: false,
        downstream_accumulate_weight: 0.0,
        downstream_spillover_weight: 0.0,
        downstream_finalize: false,
        rebuild_from_downstream: false,
    }
}

/// Precompute one [`StageLagTransition`] per stage from stage date boundaries
/// and season definitions.
///
/// This function runs once at setup time. The resulting `Vec<StageLagTransition>`
/// is indexed by stage index and consumed read-only on the forward-pass hot path,
/// eliminating all calendar arithmetic from inner solver loops.
///
/// # Behavior by cycle type
///
/// - **`Monthly`**: lag period boundaries are calendar month boundaries. Each
///   `SeasonDefinition.month_start` identifies the month.
/// - **`Weekly`** and **`Custom`**: not yet implemented; returns zero-weight
///   no-op transitions for all stages.
///
/// # No-op stages
///
/// Stages with `season_id = None` produce a fully zeroed/false
/// `StageLagTransition` including all downstream fields.
///
/// # Downstream accumulation
///
/// When `downstream_par_order > 0`, the function detects a resolution
/// transition (stages whose `season_id` crosses from the monthly range into the
/// quarterly range, i.e. `season_id >= 12`) and computes downstream fields for
/// the `downstream_par_order * 3` monthly stages immediately before the
/// transition. Passing `0` disables downstream computation entirely (all
/// downstream fields are default).
///
/// # Infallible
///
/// Invalid inputs (stages outside any season, empty season maps) produce
/// zero-weight entries. Upstream validation in `cobre-io` rejects structurally
/// invalid inputs before this function is called.
pub fn precompute_stage_lag_transitions(
    stages: &[Stage],
    season_map: &SeasonMap,
    downstream_par_order: usize,
) -> Vec<StageLagTransition> {
    let noop = StageLagTransition {
        accumulate_weight: 0.0,
        spillover_weight: 0.0,
        finalize_period: false,
        accumulate_downstream: false,
        downstream_accumulate_weight: 0.0,
        downstream_spillover_weight: 0.0,
        downstream_finalize: false,
        rebuild_from_downstream: false,
    };

    let mut result: Vec<StageLagTransition> = stages
        .iter()
        .map(|stage| {
            let Some(season_id) = stage.season_id else {
                return noop;
            };

            let Some(season_def) = season_map.seasons.iter().find(|s| s.id == season_id) else {
                return noop;
            };

            match season_map.cycle_type {
                SeasonCycleType::Monthly => compute_monthly_transition(stage, season_def, stages),
                // Weekly and Custom cycle types will be implemented when the
                // corresponding solver support is added.
                SeasonCycleType::Weekly | SeasonCycleType::Custom => noop,
            }
        })
        .collect();

    if downstream_par_order > 0 {
        compute_downstream_transitions(stages, &mut result, downstream_par_order);
    }

    result
}

/// Detect a resolution transition in `stages` and populate downstream
/// accumulation fields on the pre-transition window entries in `transitions`.
///
/// A transition is detected as the first stage whose `season_id` is `>= 12`
/// (quarterly range). The pre-transition window covers the
/// `downstream_par_order * 3` monthly stages immediately before that point.
///
/// For each stage in the window the downstream weights are computed using
/// quarterly calendar boundaries: months 1–3 → Q1, 4–6 → Q2, 7–9 → Q3,
/// 10–12 → Q4. `downstream_finalize` is set on the last monthly stage of
/// each calendar quarter within the window.
///
/// No-ops (no transition found, window is empty, or `downstream_par_order`
/// is 0) leave `transitions` unchanged.
fn compute_downstream_transitions(
    stages: &[Stage],
    transitions: &mut [StageLagTransition],
    downstream_par_order: usize,
) {
    // Find the index of the first quarterly stage (season_id >= 12).
    let Some(transition_idx) = stages
        .iter()
        .position(|s| s.season_id.is_some_and(|id| id >= 12))
    else {
        // No quarterly stage found — nothing to do.
        return;
    };

    let window_len = downstream_par_order * 3;
    let window_start = transition_idx.saturating_sub(window_len);

    for stage_idx in window_start..transition_idx {
        let stage = &stages[stage_idx];
        let Some(season_id) = stage.season_id else {
            continue;
        };

        // Map the monthly season_id (0-based: 0=Jan … 11=Dec) to a
        // 1-based calendar month for date arithmetic.
        let month = u32::try_from(season_id % 12 + 1)
            .unwrap_or_else(|_| unreachable!("season_id % 12 always fits in u32"));

        // Determine which calendar quarter this month belongs to and
        // compute its start/end boundaries.
        let quarter_start_month: u32 = ((month - 1) / 3) * 3 + 1; // 1, 4, 7, or 10
        let quarter_end_month: u32 = quarter_start_month + 2; // last month of quarter

        let year = find_season_year_monthly(stage.start_date, stage.end_date, month);

        // Compute the quarter's total hours (sum of the 3 constituent months).
        let quarter_total_hours: f64 = (quarter_start_month..=quarter_end_month)
            .map(|m| {
                let (y, mo) = if m > 12 {
                    (year + 1, m - 12)
                } else {
                    (year, m)
                };
                month_total_hours(y, mo)
            })
            .sum();

        // Period boundaries for the entire quarter.
        let quarter_period_start = NaiveDate::from_ymd_opt(year, quarter_start_month, 1)
            .unwrap_or_else(|| unreachable!("quarter start date is always valid"));
        let last_quarter_month_end = month_exclusive_end(year, quarter_end_month);

        let days_current = days_in_period(
            stage.start_date,
            stage.end_date,
            quarter_period_start,
            last_quarter_month_end,
        );
        let downstream_accumulate_weight = f64::from(days_current) * 24.0 / quarter_total_hours;

        // Spillover into the next quarter.
        let next_quarter_start_month = quarter_end_month + 1; // may be 13 → wrap to next year
        let (next_q_year, next_q_start_month) = if next_quarter_start_month > 12 {
            (year + 1, next_quarter_start_month - 12)
        } else {
            (year, next_quarter_start_month)
        };
        let next_quarter_end_month = next_q_start_month + 2;
        let next_quarter_start = NaiveDate::from_ymd_opt(next_q_year, next_q_start_month, 1)
            .unwrap_or_else(|| unreachable!("next quarter start date is always valid"));
        let (next_q_end_year, next_q_end_month_adj) = if next_quarter_end_month > 12 {
            (next_q_year + 1, next_quarter_end_month - 12)
        } else {
            (next_q_year, next_quarter_end_month)
        };
        let next_quarter_end = month_exclusive_end(next_q_end_year, next_q_end_month_adj);
        let next_quarter_total_hours: f64 = (next_q_start_month..=next_quarter_end_month)
            .map(|m| {
                let (y, mo) = if m > 12 {
                    (next_q_year + 1, m - 12)
                } else {
                    (next_q_year, m)
                };
                month_total_hours(y, mo)
            })
            .sum();
        let days_next = days_in_period(
            stage.start_date,
            stage.end_date,
            next_quarter_start,
            next_quarter_end,
        );
        let downstream_spillover_weight = if days_next > 0 {
            f64::from(days_next) * 24.0 / next_quarter_total_hours
        } else {
            0.0
        };

        // downstream_finalize: true when this is the last monthly stage of
        // its calendar quarter within the pre-transition window.
        let is_last_of_quarter = stages[stage_idx + 1..transition_idx].iter().all(|later| {
            let later_month = later.season_id.map_or(u32::MAX, |id| {
                u32::try_from(id % 12 + 1).unwrap_or(u32::MAX)
            });
            let later_quarter_start = ((later_month.saturating_sub(1)) / 3) * 3 + 1;
            later_quarter_start != quarter_start_month
        });

        transitions[stage_idx].accumulate_downstream = true;
        transitions[stage_idx].downstream_accumulate_weight = downstream_accumulate_weight;
        transitions[stage_idx].downstream_spillover_weight = downstream_spillover_weight;
        transitions[stage_idx].downstream_finalize = is_last_of_quarter;
    }

    // Mark the transition stage (first quarterly stage) for lag-state rebuild.
    // At this stage, the primary lag state is discarded and rebuilt from the
    // completed quarterly lags in the downstream ring buffer.
    if transition_idx < transitions.len() {
        transitions[transition_idx].rebuild_from_downstream = true;
    }
}

/// Precompute a noise group ID for each study stage.
///
/// Stages sharing the same `(season_id, year)` pair are assigned the same
/// group ID, where `year` is derived from `stage.start_date.year()`. This
/// allows the forward sampler to draw a single noise sample per group and
/// broadcast it to all stages in that group (Pattern C: weekly stages with
/// monthly PAR noise).
///
/// # Assignment rules
///
/// - Stages with `season_id = Some(id)` are grouped by the key
///   `(id, start_date.year())`. The first stage encountered for a new key
///   defines that key's group ID.
/// - Stages with `season_id = None` each receive a unique group ID. No
///   sharing occurs for unassigned stages.
/// - Group IDs are consecutive integers starting from 0. The first distinct
///   key encountered in stage-index order receives group 0.
///
/// # Backward compatibility
///
/// For uniform monthly studies — where every stage has a unique
/// `(season_id, year)` pair — the returned vector is `[0, 1, 2, ..., n-1]`.
/// This is equivalent to the existing per-stage indexing used by
/// `derive_forward_seed`, so no behavioural change is triggered until the
/// caller switches to `derive_forward_seed_grouped`.
///
/// # Infallible
///
/// Every stage receives exactly one group ID. The returned `Vec<u32>` has the
/// same length as `stages`.
pub fn precompute_noise_groups(stages: &[Stage]) -> Vec<u32> {
    let mut group_map: HashMap<(usize, i32), u32> = HashMap::new();
    let mut next_group_id: u32 = 0;
    let mut result = Vec::with_capacity(stages.len());
    for stage in stages {
        if let Some(season_id) = stage.season_id {
            let key = (season_id, stage.start_date.year());
            let gid = *group_map.entry(key).or_insert_with(|| {
                let id = next_group_id;
                next_group_id += 1;
                id
            });
            result.push(gid);
        } else {
            result.push(next_group_id);
            next_group_id += 1;
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, SeasonCycleType, SeasonDefinition,
        SeasonMap, Stage, StageRiskConfig, StageStateConfig,
    };

    fn monthly_season_map() -> SeasonMap {
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

    fn make_stage(
        index: usize,
        start: NaiveDate,
        end: NaiveDate,
        season_id: Option<usize>,
    ) -> Stage {
        let days = u32::try_from((end - start).num_days()).unwrap();
        Stage {
            index,
            id: i32::try_from(index).unwrap(),
            start_date: start,
            end_date: end,
            season_id,
            blocks: vec![Block {
                index: 0,
                name: "SINGLE".to_string(),
                duration_hours: f64::from(days) * 24.0,
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

    fn d(y: i32, m: u32, day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, day).unwrap()
    }

    #[test]
    fn test_uniform_monthly_identity() {
        let season_map = monthly_season_map();
        let stages: Vec<Stage> = (0..12usize)
            .map(|i| {
                let month = u32::try_from(i + 1).unwrap();
                let start = d(2026, month, 1);
                let (ny, nm) = if month == 12 {
                    (2027, 1u32)
                } else {
                    (2026, month + 1)
                };
                let end = d(ny, nm, 1);
                make_stage(i, start, end, Some(i))
            })
            .collect();

        let transitions = precompute_stage_lag_transitions(&stages, &season_map, 0);

        assert_eq!(transitions.len(), 12);
        for (i, t) in transitions.iter().enumerate() {
            assert!(
                (t.accumulate_weight - 1.0).abs() < 1e-10,
                "stage {i}: accumulate_weight expected 1.0, got {}",
                t.accumulate_weight
            );
            assert!(
                t.spillover_weight.abs() < 1e-10,
                "stage {i}: spillover_weight expected 0.0, got {}",
                t.spillover_weight
            );
            assert!(
                t.finalize_period,
                "stage {i}: finalize_period expected true"
            );
        }
    }

    /// Six-stage mixed weekly+monthly layout from the design doc.
    ///
    /// Stage dates use exclusive-end (`[start, end)`) convention:
    /// - W1: `[2026-03-28, 2026-04-04)` — 3 April days (pre-study March days excluded)
    /// - W2: `[2026-04-04, 2026-04-11)` — 7 April days
    /// - W3: `[2026-04-11, 2026-04-18)` — 7 April days
    /// - W4: `[2026-04-18, 2026-04-25)` — 7 April days
    /// - W5: `[2026-04-25, 2026-05-02)` — 6 April days + 1 May day (spillover)
    /// - M2: `[2026-05-02, 2026-06-01)` — 30 May days
    ///
    /// April = 720 h; May = 744 h.
    #[test]
    fn test_pmo_apr_2026_rv0_trace() {
        let season_map = monthly_season_map();

        let stages = vec![
            make_stage(0, d(2026, 3, 28), d(2026, 4, 4), Some(3)),
            make_stage(1, d(2026, 4, 4), d(2026, 4, 11), Some(3)),
            make_stage(2, d(2026, 4, 11), d(2026, 4, 18), Some(3)),
            make_stage(3, d(2026, 4, 18), d(2026, 4, 25), Some(3)),
            make_stage(4, d(2026, 4, 25), d(2026, 5, 2), Some(3)),
            make_stage(5, d(2026, 5, 2), d(2026, 6, 1), Some(4)),
        ];

        let transitions = precompute_stage_lag_transitions(&stages, &season_map, 0);
        assert_eq!(transitions.len(), 6);

        let april_hours = 30.0 * 24.0;
        let may_hours = 31.0 * 24.0;
        let tol = 1e-6;

        let w1 = transitions[0];
        assert!(
            (w1.accumulate_weight - 3.0 * 24.0 / april_hours).abs() < tol,
            "W1 accumulate_weight: expected {}, got {}",
            3.0 * 24.0 / april_hours,
            w1.accumulate_weight
        );
        assert!(
            w1.spillover_weight.abs() < tol,
            "W1 spillover_weight must be 0"
        );
        assert!(!w1.finalize_period, "W1 must not finalize");

        let w2 = transitions[1];
        assert!(
            (w2.accumulate_weight - 7.0 * 24.0 / april_hours).abs() < tol,
            "W2 accumulate_weight: expected {}, got {}",
            7.0 * 24.0 / april_hours,
            w2.accumulate_weight
        );
        assert!(
            w2.spillover_weight.abs() < tol,
            "W2 spillover_weight must be 0"
        );
        assert!(!w2.finalize_period, "W2 must not finalize");

        let w3 = transitions[2];
        assert!(
            (w3.accumulate_weight - 7.0 * 24.0 / april_hours).abs() < tol,
            "W3 accumulate_weight: expected {}, got {}",
            7.0 * 24.0 / april_hours,
            w3.accumulate_weight
        );
        assert!(
            w3.spillover_weight.abs() < tol,
            "W3 spillover_weight must be 0"
        );
        assert!(!w3.finalize_period, "W3 must not finalize");

        let w4 = transitions[3];
        assert!(
            (w4.accumulate_weight - 7.0 * 24.0 / april_hours).abs() < tol,
            "W4 accumulate_weight: expected {}, got {}",
            7.0 * 24.0 / april_hours,
            w4.accumulate_weight
        );
        assert!(
            w4.spillover_weight.abs() < tol,
            "W4 spillover_weight must be 0"
        );
        assert!(!w4.finalize_period, "W4 must not finalize");

        let w5 = transitions[4];
        assert!(
            (w5.accumulate_weight - 6.0 * 24.0 / april_hours).abs() < tol,
            "W5 accumulate_weight: expected {}, got {}",
            6.0 * 24.0 / april_hours,
            w5.accumulate_weight
        );
        assert!(
            (w5.spillover_weight - 1.0 * 24.0 / may_hours).abs() < tol,
            "W5 spillover_weight: expected {}, got {}",
            1.0 * 24.0 / may_hours,
            w5.spillover_weight
        );
        assert!(w5.finalize_period, "W5 must finalize");

        let m2 = transitions[5];
        assert!(
            (m2.accumulate_weight - 30.0 * 24.0 / may_hours).abs() < tol,
            "M2 accumulate_weight: expected {}, got {}",
            30.0 * 24.0 / may_hours,
            m2.accumulate_weight
        );
        assert!(
            m2.spillover_weight.abs() < tol,
            "M2 spillover_weight must be 0"
        );
        assert!(m2.finalize_period, "M2 must finalize");
    }

    // -----------------------------------------------------------------------
    // Test 3: single stage straddling a month boundary
    // -----------------------------------------------------------------------

    /// Stage `[2026-01-28, 2026-02-04)` with `season_id=0` (January).
    ///
    /// "Jan 28 to Feb 3" in inclusive notation equals `[Jan 28, Feb 04)` in
    /// Cobre exclusive-end convention.  That gives 4 January days (28–31) and
    /// 3 February days (01–03).
    ///
    /// January 2026: 31 days = 744 h.
    /// February 2026: 28 days = 672 h (not a leap year).
    #[test]
    fn test_boundary_straddling_week() {
        let season_map = monthly_season_map();
        let stage = make_stage(0, d(2026, 1, 28), d(2026, 2, 4), Some(0));
        let stages = vec![stage];

        let transitions = precompute_stage_lag_transitions(&stages, &season_map, 0);
        assert_eq!(transitions.len(), 1);

        let t = transitions[0];
        let jan_hours = 31.0 * 24.0;
        let feb_hours = 28.0 * 24.0;
        let tol = 1e-10;

        assert!(
            (t.accumulate_weight - 4.0 * 24.0 / jan_hours).abs() < tol,
            "accumulate_weight: expected {}, got {}",
            4.0 * 24.0 / jan_hours,
            t.accumulate_weight
        );
        assert!(
            (t.spillover_weight - 3.0 * 24.0 / feb_hours).abs() < tol,
            "spillover_weight: expected {}, got {}",
            3.0 * 24.0 / feb_hours,
            t.spillover_weight
        );
        assert!(t.finalize_period, "single stage must finalize its period");
    }

    // -----------------------------------------------------------------------
    // Test 4: stage with season_id = None produces no-op
    // -----------------------------------------------------------------------

    #[test]
    fn test_no_season_id_produces_noop() {
        let season_map = monthly_season_map();
        let stage = make_stage(0, d(2026, 1, 1), d(2026, 2, 1), None);
        let stages = vec![stage];

        let transitions = precompute_stage_lag_transitions(&stages, &season_map, 0);
        assert_eq!(transitions.len(), 1);

        let t = transitions[0];
        assert_eq!(t.accumulate_weight, 0.0);
        assert_eq!(t.spillover_weight, 0.0);
        assert!(!t.finalize_period);
    }

    // -----------------------------------------------------------------------
    // Test 5: two consecutive monthly stages each finalise their own period
    // -----------------------------------------------------------------------

    #[test]
    fn test_single_stage_per_month_finalizes() {
        let season_map = monthly_season_map();
        let stages = vec![
            make_stage(0, d(2026, 1, 1), d(2026, 2, 1), Some(0)),
            make_stage(1, d(2026, 2, 1), d(2026, 3, 1), Some(1)),
        ];

        let transitions = precompute_stage_lag_transitions(&stages, &season_map, 0);
        assert_eq!(transitions.len(), 2);
        assert!(
            transitions[0].finalize_period,
            "January stage must finalize"
        );
        assert!(
            transitions[1].finalize_period,
            "February stage must finalize"
        );
    }

    // -----------------------------------------------------------------------
    // Test 6: four weekly stages in January — only the last finalises
    // -----------------------------------------------------------------------

    #[test]
    fn test_multiple_weekly_stages_only_last_finalizes() {
        let season_map = monthly_season_map();
        let stages = vec![
            make_stage(0, d(2026, 1, 1), d(2026, 1, 8), Some(0)),
            make_stage(1, d(2026, 1, 8), d(2026, 1, 15), Some(0)),
            make_stage(2, d(2026, 1, 15), d(2026, 1, 22), Some(0)),
            make_stage(3, d(2026, 1, 22), d(2026, 1, 29), Some(0)),
        ];

        let transitions = precompute_stage_lag_transitions(&stages, &season_map, 0);
        assert_eq!(transitions.len(), 4);

        let jan_hours = 31.0 * 24.0;
        let tol = 1e-10;

        for (i, t) in transitions.iter().enumerate().take(3) {
            assert!(
                !t.finalize_period,
                "stage {i}: finalize_period must be false"
            );
            assert!(
                (t.accumulate_weight - 7.0 * 24.0 / jan_hours).abs() < tol,
                "stage {i}: accumulate_weight wrong: {}",
                t.accumulate_weight
            );
            assert!(
                t.spillover_weight.abs() < tol,
                "stage {i}: spillover_weight must be 0"
            );
        }

        let w4 = transitions[3];
        assert!(w4.finalize_period, "W4 must be the finalising stage");
        assert!(
            (w4.accumulate_weight - 7.0 * 24.0 / jan_hours).abs() < tol,
            "W4 accumulate_weight wrong: {}",
            w4.accumulate_weight
        );
    }

    // -----------------------------------------------------------------------
    // Tests for compute_recent_observation_seed
    // -----------------------------------------------------------------------

    use cobre_core::{
        entities::hydro::{HydroGenerationModel, HydroPenalties},
        initial_conditions::RecentObservation,
        EntityId,
    };

    fn make_hydro(id: i32) -> Hydro {
        Hydro {
            id: EntityId(id),
            name: format!("H{id}"),
            bus_id: EntityId(1),
            downstream_id: None,
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: 0.0,
            max_storage_hm3: 100.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: HydroGenerationModel::ConstantProductivity {
                productivity_mw_per_m3s: 0.95,
            },
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 100.0,
            min_generation_mw: 0.0,
            max_generation_mw: 100.0,
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: None,
            evaporation_reference_volumes_hm3: None,
            diversion: None,
            filling: None,
            penalties: HydroPenalties {
                spillage_cost: 0.0,
                diversion_cost: 0.0,
                fpha_turbined_cost: 0.0,
                storage_violation_below_cost: 0.0,
                filling_target_violation_cost: 0.0,
                turbined_violation_below_cost: 0.0,
                outflow_violation_below_cost: 0.0,
                outflow_violation_above_cost: 0.0,
                generation_violation_below_cost: 0.0,
                evaporation_violation_cost: 0.0,
                water_withdrawal_violation_cost: 0.0,
                water_withdrawal_violation_pos_cost: 0.0,
                water_withdrawal_violation_neg_cost: 0.0,
                evaporation_violation_pos_cost: 0.0,
                evaporation_violation_neg_cost: 0.0,
                inflow_nonnegativity_cost: 1000.0,
            },
        }
    }

    fn make_observation(
        hydro_id: i32,
        y: i32,
        m1: u32,
        d1: u32,
        m2: u32,
        d2: u32,
        val: f64,
    ) -> RecentObservation {
        RecentObservation {
            hydro_id: EntityId(hydro_id),
            start_date: d(y, m1, d1),
            end_date: d(y, m2, d2),
            value_m3s: val,
        }
    }

    // April 2026: 30 days = 720 h.
    const APRIL_2026_HOURS: f64 = 720.0;

    /// Test 7: empty `recent_observations` — zero seed.
    #[test]
    fn test_seed_empty_observations_returns_zero() {
        let season_map = monthly_season_map();
        // First study stage: April 4 → May 2 (season_id = 3 → April).
        let stage = make_stage(0, d(2026, 4, 4), d(2026, 5, 2), Some(3));
        let hydros = vec![make_hydro(0)];

        let seed = compute_recent_observation_seed(&[], &stage, &season_map, &hydros);

        assert_eq!(seed.accum_seed.len(), 1);
        assert_eq!(seed.accum_seed[0], 0.0);
        assert_eq!(seed.weight_seed, 0.0);
    }

    /// Test 8: one observation for one hydro, 3 days (April 1–4) at 500.0 m3/s.
    ///
    /// Expected: `accum_seed[0] == 500.0 * 72.0`, `weight_seed == 72.0 / 720.0`.
    #[test]
    fn test_seed_one_observation_one_hydro() {
        let season_map = monthly_season_map();
        let stage = make_stage(0, d(2026, 4, 4), d(2026, 5, 2), Some(3));
        let hydros = vec![make_hydro(0)];
        let obs = vec![make_observation(0, 2026, 4, 1, 4, 4, 500.0)];

        let seed = compute_recent_observation_seed(&obs, &stage, &season_map, &hydros);

        let expected_accum = 500.0 * 72.0;
        let expected_weight = 72.0 / APRIL_2026_HOURS;
        let tol = 1e-10;
        assert!(
            (seed.accum_seed[0] - expected_accum).abs() < tol,
            "accum_seed[0]: expected {expected_accum}, got {}",
            seed.accum_seed[0]
        );
        assert!(
            (seed.weight_seed - expected_weight).abs() < tol,
            "weight_seed: expected {expected_weight}, got {}",
            seed.weight_seed
        );
    }

    /// Test 9: two observations for the same hydro (rv2 pattern: Apr 1–4 at 500.0 and
    /// Apr 4–11 at 480.0) → additive accumulation.
    ///
    /// `accum_seed[0] == 500.0 * 72.0 + 480.0 * 168.0`
    /// `weight_seed == (72.0 + 168.0) / 720.0`
    #[test]
    fn test_seed_two_observations_same_hydro_additive() {
        let season_map = monthly_season_map();
        let stage = make_stage(0, d(2026, 4, 11), d(2026, 5, 2), Some(3));
        let hydros = vec![make_hydro(0)];
        let obs = vec![
            make_observation(0, 2026, 4, 1, 4, 4, 500.0),
            make_observation(0, 2026, 4, 4, 4, 11, 480.0),
        ];

        let seed = compute_recent_observation_seed(&obs, &stage, &season_map, &hydros);

        let expected_accum = 500.0 * 72.0 + 480.0 * 168.0;
        let expected_weight = (72.0 + 168.0) / APRIL_2026_HOURS;
        let tol = 1e-10;
        assert!(
            (seed.accum_seed[0] - expected_accum).abs() < tol,
            "accum_seed[0]: expected {expected_accum}, got {}",
            seed.accum_seed[0]
        );
        assert!(
            (seed.weight_seed - expected_weight).abs() < tol,
            "weight_seed: expected {expected_weight}, got {}",
            seed.weight_seed
        );
    }

    /// Test 10: observations for two different hydros → each slot is independent.
    #[test]
    fn test_seed_two_observations_different_hydros_independent() {
        let season_map = monthly_season_map();
        let stage = make_stage(0, d(2026, 4, 4), d(2026, 5, 2), Some(3));
        let hydros = vec![make_hydro(0), make_hydro(1)];
        let obs = vec![
            make_observation(0, 2026, 4, 1, 4, 4, 500.0), // hydro 0: 3 days
            make_observation(1, 2026, 4, 1, 4, 4, 300.0), // hydro 1: 3 days
        ];

        let seed = compute_recent_observation_seed(&obs, &stage, &season_map, &hydros);

        let tol = 1e-10;
        assert!(
            (seed.accum_seed[0] - 500.0 * 72.0).abs() < tol,
            "accum_seed[0]: expected {}, got {}",
            500.0 * 72.0,
            seed.accum_seed[0]
        );
        assert!(
            (seed.accum_seed[1] - 300.0 * 72.0).abs() < tol,
            "accum_seed[1]: expected {}, got {}",
            300.0 * 72.0,
            seed.accum_seed[1]
        );
        // weight counts both hydros' observation hours (same calendar period, so
        // summing gives 2 * 72h, but the weight formula only divides by total_period_hours once
        // per observation — two observations with the same date range double the weight).
        let expected_weight = (72.0 + 72.0) / APRIL_2026_HOURS;
        assert!(
            (seed.weight_seed - expected_weight).abs() < tol,
            "weight_seed: expected {expected_weight}, got {}",
            seed.weight_seed
        );
    }

    /// Test 11: observation for unknown `hydro_id` — silently skipped, zero seed.
    #[test]
    fn test_seed_unknown_hydro_id_silently_skipped() {
        let season_map = monthly_season_map();
        let stage = make_stage(0, d(2026, 4, 4), d(2026, 5, 2), Some(3));
        let hydros = vec![make_hydro(0)];
        // hydro_id = 99 is not in the registry.
        let obs = vec![make_observation(99, 2026, 4, 1, 4, 4, 500.0)];

        let seed = compute_recent_observation_seed(&obs, &stage, &season_map, &hydros);

        assert_eq!(seed.accum_seed.len(), 1);
        assert_eq!(seed.accum_seed[0], 0.0, "unknown hydro_id must be skipped");
        assert_eq!(
            seed.weight_seed, 0.0,
            "weight must be 0 when all hydros unknown"
        );
    }

    /// Test 12: first stage has `season_id` = None — zero seed returned.
    #[test]
    fn test_seed_no_season_id_returns_zero() {
        let season_map = monthly_season_map();
        let stage = make_stage(0, d(2026, 4, 1), d(2026, 5, 1), None);
        let hydros = vec![make_hydro(0)];
        let obs = vec![make_observation(0, 2026, 4, 1, 4, 4, 500.0)];

        let seed = compute_recent_observation_seed(&obs, &stage, &season_map, &hydros);

        assert_eq!(seed.accum_seed[0], 0.0);
        assert_eq!(seed.weight_seed, 0.0);
    }

    #[test]
    fn test_noise_groups_monthly_unique() {
        let stages: Vec<Stage> = (0..12usize)
            .map(|i| {
                let month = u32::try_from(i + 1).unwrap();
                let start = d(2024, month, 1);
                let (ny, nm) = if month == 12 {
                    (2025, 1u32)
                } else {
                    (2024, month + 1)
                };
                let end = d(ny, nm, 1);
                make_stage(i, start, end, Some(i))
            })
            .collect();

        let groups = precompute_noise_groups(&stages);

        assert_eq!(groups.len(), 12);
        let expected: Vec<u32> = (0..12u32).collect();
        assert_eq!(groups, expected);
    }

    #[test]
    fn test_noise_groups_weekly_shared() {
        let stages_s0: Vec<Stage> = (0..4usize)
            .map(|i| {
                let day_start = u32::try_from(i * 7 + 1).unwrap();
                let day_end = u32::try_from(i * 7 + 8).unwrap();
                let start = d(2024, 1, day_start);
                let end = d(2024, 1, day_end);
                make_stage(i, start, end, Some(0))
            })
            .collect();
        let stages_s1: Vec<Stage> = (0..4usize)
            .map(|i| {
                let day_start = u32::try_from(i * 7 + 1).unwrap();
                let day_end = u32::try_from(i * 7 + 8).unwrap();
                let start = d(2024, 2, day_start);
                let end = d(2024, 2, day_end);
                make_stage(i + 4, start, end, Some(1))
            })
            .collect();

        let mut all_stages = stages_s0;
        all_stages.extend(stages_s1);

        let groups = precompute_noise_groups(&all_stages);

        assert_eq!(groups.len(), 8);
        assert!(groups[0..4].iter().all(|&g| g == 0));
        assert!(groups[4..8].iter().all(|&g| g == 1));
    }

    #[test]
    fn test_noise_groups_mixed_weekly_monthly() {
        let weekly: Vec<Stage> = (0..4usize)
            .map(|i| {
                let day_start = u32::try_from(i * 7 + 1).unwrap();
                let day_end = u32::try_from(i * 7 + 8).unwrap();
                let start = d(2024, 1, day_start);
                let end = d(2024, 1, day_end);
                make_stage(i, start, end, Some(0))
            })
            .collect();
        let monthly = make_stage(4, d(2024, 1, 1), d(2024, 2, 1), Some(0));

        let mut stages = weekly;
        stages.push(monthly);

        let groups = precompute_noise_groups(&stages);

        assert_eq!(groups.len(), 5);
        assert!(
            groups.iter().all(|&g| g == 0),
            "all stages must share group 0"
        );
    }

    #[test]
    fn test_noise_groups_none_season_id() {
        let stages: Vec<Stage> = (0..3usize)
            .map(|i| {
                let start = d(2024, 1, u32::try_from(i + 1).unwrap());
                let end = d(2024, 1, u32::try_from(i + 2).unwrap());
                make_stage(i, start, end, None)
            })
            .collect();

        let groups = precompute_noise_groups(&stages);

        assert_eq!(groups.len(), 3);
        assert_eq!(groups[0], 0);
        assert_eq!(groups[1], 1);
        assert_eq!(groups[2], 2);
    }

    /// Test 5: same `season_id` but different years must produce different groups.
    #[test]
    fn test_noise_groups_cross_year() {
        // Two weekly stages: season_id=0, year 2024 and year 2025.
        let stage_2024 = make_stage(0, d(2024, 1, 1), d(2024, 1, 8), Some(0));
        let stage_2025 = make_stage(1, d(2025, 1, 1), d(2025, 1, 8), Some(0));

        let stages = vec![stage_2024, stage_2025];
        let groups = precompute_noise_groups(&stages);

        assert_eq!(groups.len(), 2);
        assert_ne!(
            groups[0], groups[1],
            "different years must yield different groups"
        );
        assert_eq!(groups[0], 0);
        assert_eq!(groups[1], 1);
    }
}
