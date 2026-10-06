//! Precomputation of per-stage lag accumulation weights and period
//! finalization flags from stage date boundaries and season definitions.
//!
//! [`precompute_stage_lag_transitions`] runs once at setup; the resulting
//! per-stage slice is consumed read-only on the hot path, keeping calendar
//! arithmetic out of inner solver loops.

use std::collections::HashMap;

use chrono::{Datelike, NaiveDate};
use cobre_core::{
    temporal::{SUB_PERIOD_TOLERANCE_DAYS, SeasonDefinition, SeasonMap, Stage, StageLagTransition},
    window_period_overlaps,
};

use super::precompute::PrecomputedPar;
use crate::season_cast::{
    find_season_year_monthly, month_total_hours, next_season_period_window, resolved_year,
    season_period_window,
};

/// Overlap hours between `stage`'s calendar span and a single period of
/// `period_hours` duration starting at `period_start`, via
/// [`window_period_overlaps`] framed with the origin at `period_start`.
///
/// A negative offset (the stage starts before `period_start`) is the intended
/// pre-period straddle case — `window_period_overlaps` clamps it, so this
/// does not guard or reject it.
fn single_period_overlap_hours(stage: &Stage, period_start: NaiveDate, period_hours: f64) -> f64 {
    let start_days = i32::try_from((stage.start_date - period_start).num_days())
        .unwrap_or_else(|_| unreachable!("stage-to-period day offset always fits in i32"));
    let window_start_hours = f64::from(start_days) * 24.0;

    let width_days = u32::try_from((stage.end_date - stage.start_date).num_days())
        .unwrap_or_else(|_| unreachable!("stage width in days always fits in u32"));
    let window_width_hours = f64::from(width_days) * 24.0;

    window_period_overlaps(window_start_hours, window_width_hours, &[period_hours])
        .first()
        .copied()
        .unwrap_or(0.0)
}

/// An all-zero, non-finalizing [`StageLagTransition`] — the shared absent-case
/// value for a stage with no season or an unresolvable season.
fn noop_transition() -> StageLagTransition {
    StageLagTransition {
        accumulate_weight: 0.0,
        spillover_weight: 0.0,
        finalize_period: false,
        accumulate_downstream: false,
        downstream_accumulate_weight: 0.0,
        downstream_spillover_weight: 0.0,
        downstream_finalize: false,
        rebuild_from_downstream: false,
    }
}

/// The full-weight, finalizing [`StageLagTransition`] — one stage folded
/// entirely into one lag bucket. The out-of-bounds fallback for
/// [`resolve_stage_lag_transition`], shared with the per-stage evaluation
/// path's own `unwrap_or` default.
const UNIFORM_MONTHLY_TRANSITION: StageLagTransition = StageLagTransition {
    accumulate_weight: 1.0,
    spillover_weight: 0.0,
    finalize_period: true,
    accumulate_downstream: false,
    downstream_accumulate_weight: 0.0,
    downstream_spillover_weight: 0.0,
    downstream_finalize: false,
    rebuild_from_downstream: false,
};

/// Resolve stage `t`'s transition from `transitions`: a present entry —
/// including a `noop_transition` one — is consumed as-is; the full-weight,
/// finalizing identity transition is the fallback ONLY when `t` is out of
/// bounds. Every η-inversion and lag-accumulation call site shares this
/// one convention; swapping a present noop entry for the fallback would
/// desync that site's lag chain from every other reader of the same
/// `transitions` slice.
#[must_use]
#[inline]
pub fn resolve_stage_lag_transition(
    transitions: &[StageLagTransition],
    t: usize,
) -> StageLagTransition {
    transitions
        .get(t)
        .copied()
        .unwrap_or(UNIFORM_MONTHLY_TRANSITION)
}

/// Compute the [`StageLagTransition`] for a single stage from its resolved
/// `season_def`'s period window — the day-weighted accumulate/spillover/
/// finalize arithmetic generalized across `Monthly`/`Weekly`/`Custom` cycles.
pub(crate) fn compute_period_transition(
    stage: &Stage,
    position: usize,
    season_map: &SeasonMap,
    season_def: &SeasonDefinition,
    all_stages: &[Stage],
) -> StageLagTransition {
    let current = season_period_window(season_map, season_def, stage);

    let accumulate_weight =
        single_period_overlap_hours(stage, current.start, current.hours) / current.hours;

    let spillover_weight = next_season_period_window(season_map, season_def, &current)
        .map_or(0.0, |next| {
            single_period_overlap_hours(stage, next.start, next.hours) / next.hours
        });

    let year = resolved_year(season_map, season_def, stage);
    let finalize_period = !all_stages
        .iter()
        .skip(position + 1)
        .filter(|s| s.season_id == Some(season_def.id))
        .any(|s| resolved_year(season_map, season_def, s) == year);

    StageLagTransition {
        accumulate_weight,
        spillover_weight,
        finalize_period,
        accumulate_downstream: false,
        downstream_accumulate_weight: 0.0,
        downstream_spillover_weight: 0.0,
        downstream_finalize: false,
        rebuild_from_downstream: false,
    }
}

/// Derives the `downstream_par_order` gate consumed by
/// [`precompute_stage_lag_transitions`] and by η-inversion
/// (`standardize_historical_windows`). The order belongs to the PAR model
/// whose ψ the downstream ring feeds — never to a lag-state depth, which may
/// be wider than the model's own order; `par`'s global
/// [`max_order`](PrecomputedPar::max_order) stands in for a quarterly order
/// until a separate quarterly PAR model exists. The gate reads
/// `par.max_order()` when some stage's season spans a calendar quarter right
/// after a stage whose season spans a calendar month, each within
/// [`SUB_PERIOD_TOLERANCE_DAYS`] of the calendar length; otherwise, and for a
/// `None` `season_map`, the ring stays inert (`0`).
#[must_use]
pub fn derive_downstream_par_order(
    stages: &[Stage],
    par: &PrecomputedPar,
    season_map: Option<&SeasonMap>,
) -> usize {
    if season_map.is_some_and(|sm| month_to_quarter_step(stages, sm).is_some()) {
        par.max_order()
    } else {
        0
    }
}

const CALENDAR_MONTH_DAYS: (i64, i64) = (28, 31);
const CALENDAR_QUARTER_DAYS: (i64, i64) = (90, 92);

fn season_spans(
    season_map: &SeasonMap,
    season_id: Option<usize>,
    (shortest, longest): (i64, i64),
) -> bool {
    season_id
        .and_then(|id| season_map.resolution_level_of(id))
        .and_then(|days| i64::try_from(days).ok())
        .is_some_and(|days| {
            (shortest - SUB_PERIOD_TOLERANCE_DAYS..=longest + SUB_PERIOD_TOLERANCE_DAYS)
                .contains(&days)
        })
}

fn month_to_quarter_step(stages: &[Stage], season_map: &SeasonMap) -> Option<usize> {
    stages
        .windows(2)
        .position(|pair| {
            season_spans(season_map, pair[0].season_id, CALENDAR_MONTH_DAYS)
                && season_spans(season_map, pair[1].season_id, CALENDAR_QUARTER_DAYS)
        })
        .map(|index| index + 1)
}

/// Precompute one [`StageLagTransition`] per stage from stage date boundaries
/// and season definitions; consumed read-only on the per-stage evaluation hot path.
///
/// A `season_id = None` stage, or any input outside a season, produces a
/// fully zeroed no-op transition.
///
/// # Downstream accumulation
///
/// `downstream_par_order > 0` places the resolution transition at the
/// month-to-quarter step [`derive_downstream_par_order`] detects and fills
/// downstream fields for the `downstream_par_order * 3` monthly stages before
/// it. Passing `0`, or stages without that step, leaves every downstream field
/// at its default — the downstream fields are inert unless populated here.
#[must_use]
pub fn precompute_stage_lag_transitions(
    stages: &[Stage],
    season_map: &SeasonMap,
    downstream_par_order: usize,
) -> Vec<StageLagTransition> {
    let mut result: Vec<StageLagTransition> = stages
        .iter()
        .enumerate()
        .map(|(position, stage)| {
            let Some(season_id) = stage.season_id else {
                return noop_transition();
            };

            let Some(season_def) = season_map.seasons.iter().find(|s| s.id == season_id) else {
                return noop_transition();
            };

            compute_period_transition(stage, position, season_map, season_def, stages)
        })
        .collect();

    if downstream_par_order > 0 {
        compute_downstream_transitions(stages, season_map, &mut result, downstream_par_order);
    }

    result
}

fn calendar_month_of(season_map: &SeasonMap, season_id: usize) -> Option<u32> {
    season_map
        .seasons
        .iter()
        .find(|def| def.id == season_id)
        .map(|def| def.month_start)
        .filter(|month| (1..=12).contains(month))
}

/// Sum of hours across the 3 months from `start_month`, wrapping into
/// `start_year + 1` for a month index that spills past December.
fn quarter_hours(start_year: i32, start_month: u32) -> f64 {
    (start_month..=start_month + 2)
        .map(|m| {
            let (y, mo) = if m > 12 {
                (start_year + 1, m - 12)
            } else {
                (start_year, m)
            };
            month_total_hours(y, mo)
        })
        .sum()
}

/// Populate downstream accumulation fields on the pre-transition window entries
/// in `transitions`.
///
/// The transition is the month-to-quarter step of
/// [`derive_downstream_par_order`]; the window is the `downstream_par_order * 3`
/// monthly stages before it. Weights use the calendar quarter of each window
/// season's `month_start` (months 1–3 → Q1, 4–6 → Q2, 7–9 → Q3, 10–12 → Q4);
/// `downstream_finalize` is set on the last monthly stage of each calendar
/// quarter within the window. No transition / empty window leaves
/// `transitions` unchanged.
fn compute_downstream_transitions(
    stages: &[Stage],
    season_map: &SeasonMap,
    transitions: &mut [StageLagTransition],
    downstream_par_order: usize,
) {
    let Some(transition_idx) = month_to_quarter_step(stages, season_map) else {
        return;
    };

    let window_len = downstream_par_order * 3;
    let window_start = transition_idx.saturating_sub(window_len);

    for stage_idx in window_start..transition_idx {
        let stage = &stages[stage_idx];
        let Some(month) = stage
            .season_id
            .and_then(|id| calendar_month_of(season_map, id))
        else {
            continue;
        };

        let quarter_start_month: u32 = ((month - 1) / 3) * 3 + 1;
        let quarter_end_month: u32 = quarter_start_month + 2;

        let year = find_season_year_monthly(stage.start_date, stage.end_date, month);

        let quarter_total_hours = quarter_hours(year, quarter_start_month);

        let quarter_period_start = NaiveDate::from_ymd_opt(year, quarter_start_month, 1)
            .unwrap_or_else(|| unreachable!("quarter start date is always valid"));

        let downstream_accumulate_weight =
            single_period_overlap_hours(stage, quarter_period_start, quarter_total_hours)
                / quarter_total_hours;

        let next_quarter_start_month = quarter_end_month + 1;
        let (next_q_year, next_q_start_month) = if next_quarter_start_month > 12 {
            (year + 1, next_quarter_start_month - 12)
        } else {
            (year, next_quarter_start_month)
        };
        let next_quarter_start = NaiveDate::from_ymd_opt(next_q_year, next_q_start_month, 1)
            .unwrap_or_else(|| unreachable!("next quarter start date is always valid"));
        let next_quarter_total_hours = quarter_hours(next_q_year, next_q_start_month);

        let downstream_spillover_weight =
            single_period_overlap_hours(stage, next_quarter_start, next_quarter_total_hours)
                / next_quarter_total_hours;

        let is_last_of_quarter = stages[stage_idx + 1..transition_idx].iter().all(|later| {
            later
                .season_id
                .and_then(|id| calendar_month_of(season_map, id))
                .is_none_or(|later_month| ((later_month - 1) / 3) * 3 + 1 != quarter_start_month)
        });

        transitions[stage_idx].accumulate_downstream = true;
        transitions[stage_idx].downstream_accumulate_weight = downstream_accumulate_weight;
        transitions[stage_idx].downstream_spillover_weight = downstream_spillover_weight;
        transitions[stage_idx].downstream_finalize = is_last_of_quarter;
    }

    // rebuild_from_downstream: at the transition the primary lag state is
    // discarded and rebuilt from the completed quarterly lags in the downstream
    // ring buffer.
    if transition_idx < transitions.len() {
        transitions[transition_idx].rebuild_from_downstream = true;
    }
}

/// Precompute a noise group ID for each study stage, so [`ForwardSampler`] can
/// draw one noise sample per group and broadcast it (weekly stages sharing
/// monthly PAR noise).
///
/// [`ForwardSampler`]: crate::ForwardSampler
///
/// Stages with `season_id = Some(id)` group by `(id, start_date.year())`,
/// consecutive IDs from 0 in slice order of first occurrence; a
/// `season_id = None` stage each receives its own unique ID (no sharing). For a
/// uniform monthly study the result is `[0, 1, …, n-1]`.
#[must_use]
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
    use cobre_core::temporal::{SeasonCycleType, SeasonCycles, SeasonDefinition, SeasonMap, Stage};
    use cobre_core::{
        EntityId, Hydro, InflowHistoryRow, RecentObservation,
        test_support::{HydroSpec, MirrorUnitGroup, StageSpec, date, single_block},
    };

    use crate::par::lag_kernel::{
        DownstreamLagAccum, EntityMajor, PrimaryLagAccum, advance_lag_chain,
    };
    use crate::seeds::{DerivedInflowSeeds, derive_inflow_seeds};
    use crate::test_support::{
        InflowModelSpec, MonthlyLabels, make_inflow_model, monthly_quarterly_season_map,
        monthly_season_map, sparse_ring_season_map, weekly_season_map,
    };

    /// A one-hydro [`PrecomputedPar`] with `max_order() == 1`: an order-1
    /// model anchored at `stages[0]`'s id, plus a `stage_id = -1` pre-study
    /// model with no coefficients (lag initialization).
    fn par_of_order_one(stages: &[Stage]) -> PrecomputedPar {
        let models = vec![
            make_inflow_model(InflowModelSpec {
                hydro_id: 1,
                stage_id: stages[0].id,
                ar_coefficients: vec![0.5],
                ..Default::default()
            }),
            make_inflow_model(InflowModelSpec {
                hydro_id: 1,
                stage_id: -1,
                ar_coefficients: vec![],
                ..Default::default()
            }),
        ];
        let par = PrecomputedPar::build(&models, stages, &[EntityId(1)], None)
            .expect("par_of_order_one: valid PAR build");
        assert_eq!(par.max_order(), 1);
        par
    }

    fn make_stage(
        index: usize,
        start: NaiveDate,
        end: NaiveDate,
        season_id: Option<usize>,
    ) -> Stage {
        let days = u32::try_from((end - start).num_days()).unwrap();
        cobre_core::test_support::make_stage(StageSpec {
            id: i32::try_from(index).unwrap(),
            index: Some(index),
            start_date: start,
            end_date: end,
            season_id,
            blocks: single_block("SINGLE", f64::from(days) * 24.0),
            ..Default::default()
        })
    }

    fn d(y: i32, m: u32, day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, day).unwrap()
    }

    // -----------------------------------------------------------------------
    // derive_downstream_par_order gate (month-to-quarter span step)
    // -----------------------------------------------------------------------

    #[test]
    fn test_derive_downstream_par_order_weekly_returns_zero() {
        let season_map = weekly_season_map();
        let stages = vec![
            make_stage(0, d(2026, 1, 1), d(2026, 1, 8), Some(0)),
            make_stage(1, d(2026, 1, 8), d(2026, 1, 15), Some(1)),
            make_stage(2, d(2026, 1, 15), d(2026, 1, 22), Some(2)),
            make_stage(3, d(2026, 1, 22), d(2026, 1, 29), Some(12)),
        ];

        let derived =
            derive_downstream_par_order(&stages, &par_of_order_one(&stages), Some(&season_map));
        assert_eq!(
            derived, 0,
            "a Weekly season cycle must never activate the quarterly ring, even \
             when a stage crosses season_id >= 12"
        );

        let transitions = precompute_stage_lag_transitions(&stages, &season_map, derived);
        assert!(
            transitions.iter().all(|t| !t.rebuild_from_downstream),
            "the ring must stay provably inert: no stage rebuilds from downstream"
        );
    }

    #[test]
    fn test_derive_downstream_par_order_monthly_map_stays_inert() {
        let season_map = monthly_season_map(MonthlyLabels::OneBased);
        let stages = vec![
            make_stage(0, d(2026, 1, 1), d(2026, 2, 1), Some(0)),
            make_stage(1, d(2026, 2, 1), d(2026, 3, 1), Some(1)),
            make_stage(2, d(2026, 3, 1), d(2026, 4, 1), Some(2)),
            make_stage(3, d(2026, 4, 1), d(2026, 7, 1), Some(12)),
        ];

        let derived =
            derive_downstream_par_order(&stages, &par_of_order_one(&stages), Some(&season_map));
        assert_eq!(
            derived, 0,
            "every season of a Monthly map spans one month, so no stage spans a \
             quarter and the quarterly ring stays inert"
        );
    }

    #[test]
    fn test_derive_downstream_par_order_no_season_map_returns_zero() {
        let stages = vec![
            make_stage(0, d(2026, 1, 1), d(2026, 2, 1), Some(0)),
            make_stage(1, d(2026, 2, 1), d(2026, 3, 1), Some(1)),
            make_stage(2, d(2026, 3, 1), d(2026, 4, 1), Some(2)),
            make_stage(3, d(2026, 4, 1), d(2026, 7, 1), Some(12)),
        ];

        let derived = derive_downstream_par_order(&stages, &par_of_order_one(&stages), None);
        assert_eq!(derived, 0, "a None season_map must leave the ring inert");
    }

    #[test]
    fn test_derive_downstream_par_order_custom_quarterly_stays_active() {
        let season_map = custom_multi_resolution_season_map();
        let june_stage = make_stage(0, d(2024, 6, 1), d(2024, 7, 1), Some(5));
        let q3_stage = make_stage(1, d(2024, 7, 1), d(2024, 10, 1), Some(12));
        let stages = vec![june_stage, q3_stage];

        let derived =
            derive_downstream_par_order(&stages, &par_of_order_one(&stages), Some(&season_map));
        assert_eq!(
            derived, 1,
            "a Custom season cycle crossing season_id >= 12 must keep the \
             quarterly ring active at par.max_order() — only Weekly is gated off"
        );
    }

    fn ring_study_stages(season_ids: [usize; 5]) -> Vec<Stage> {
        let bounds = [
            (d(2026, 1, 1), d(2026, 2, 1)),
            (d(2026, 2, 1), d(2026, 3, 1)),
            (d(2026, 3, 1), d(2026, 4, 1)),
            (d(2026, 4, 1), d(2026, 7, 1)),
            (d(2026, 7, 1), d(2026, 10, 1)),
        ];
        bounds
            .into_iter()
            .zip(season_ids)
            .enumerate()
            .map(|(index, ((start, end), season_id))| {
                make_stage(index, start, end, Some(season_id))
            })
            .collect()
    }

    fn renumbered(season_map: &SeasonMap, renumber: fn(usize) -> usize) -> SeasonMap {
        SeasonMap {
            cycle_type: season_map.cycle_type,
            seasons: season_map
                .seasons
                .iter()
                .map(|def| SeasonDefinition {
                    id: renumber(def.id),
                    ..def.clone()
                })
                .collect(),
        }
    }

    fn flagged(
        transitions: &[StageLagTransition],
        flag: fn(&StageLagTransition) -> bool,
    ) -> Vec<usize> {
        transitions
            .iter()
            .enumerate()
            .filter(|(_, transition)| flag(transition))
            .map(|(index, _)| index)
            .collect()
    }

    #[test]
    fn cascade_step_ignores_season_id_numbering() {
        let ring_map = sparse_ring_season_map();
        let renumbered_map = renumbered(&ring_map, |id| match id {
            12 => 3,
            13 => 4,
            id => id,
        });
        let ring_stages = ring_study_stages([0, 1, 2, 12, 13]);
        let renumbered_stages = ring_study_stages([0, 1, 2, 3, 4]);

        let derived = derive_downstream_par_order(
            &renumbered_stages,
            &par_of_order_one(&renumbered_stages),
            Some(&renumbered_map),
        );
        assert_eq!(
            derived, 1,
            "quarters numbered 3 and 4 right after month-long seasons must activate the ring"
        );
        assert_eq!(
            precompute_stage_lag_transitions(&renumbered_stages, &renumbered_map, 1),
            precompute_stage_lag_transitions(&ring_stages, &ring_map, 1),
            "renumbering the quarterly seasons must not change any transition"
        );
    }

    #[test]
    fn cascade_window_ignores_month_season_numbering() {
        let ring_map = sparse_ring_season_map();
        let misaligned_map = renumbered(&ring_map, |id| match id {
            0 => 2,
            1 => 3,
            2 => 4,
            12 => 0,
            13 => 1,
            id => id,
        });

        assert_eq!(
            precompute_stage_lag_transitions(
                &ring_study_stages([2, 3, 4, 0, 1]),
                &misaligned_map,
                1
            ),
            precompute_stage_lag_transitions(&ring_study_stages([0, 1, 2, 12, 13]), &ring_map, 1),
            "each window month must come from its season's definition, not its id"
        );
    }

    #[test]
    fn cascade_needs_a_month_long_season_before_the_quarter() {
        let season_map = SeasonMap {
            cycle_type: SeasonCycleType::Custom,
            seasons: monthly_quarterly_season_map()
                .seasons
                .into_iter()
                .filter(|def| (12..=15).contains(&def.id))
                .collect(),
        };
        let stages = vec![
            make_stage(0, d(2024, 7, 1), d(2024, 10, 1), Some(12)),
            make_stage(1, d(2024, 10, 1), d(2025, 1, 1), Some(13)),
            make_stage(2, d(2025, 1, 1), d(2025, 4, 1), Some(14)),
        ];

        let derived =
            derive_downstream_par_order(&stages, &par_of_order_one(&stages), Some(&season_map));
        assert_eq!(
            derived, 0,
            "a study of quarters alone has no month-long season to step from"
        );

        let transitions = precompute_stage_lag_transitions(&stages, &season_map, 1);
        assert!(
            transitions
                .iter()
                .all(|t| !t.accumulate_downstream && !t.rebuild_from_downstream),
            "without a month-to-quarter step no stage may accumulate or rebuild \
             downstream: {transitions:?}"
        );
    }

    #[test]
    fn cascade_step_matches_the_resolution_groups_on_a_layered_map() {
        let season_map = monthly_quarterly_season_map();
        let stages = vec![
            make_stage(0, d(2024, 1, 1), d(2024, 2, 1), Some(0)),
            make_stage(1, d(2024, 2, 1), d(2024, 3, 1), Some(1)),
            make_stage(2, d(2024, 3, 1), d(2024, 4, 1), Some(2)),
            make_stage(3, d(2024, 4, 1), d(2024, 5, 1), Some(3)),
            make_stage(4, d(2024, 5, 1), d(2024, 6, 1), Some(4)),
            make_stage(5, d(2024, 6, 1), d(2024, 7, 1), Some(5)),
            make_stage(6, d(2024, 7, 1), d(2024, 10, 1), Some(12)),
            make_stage(7, d(2024, 10, 1), d(2025, 1, 1), Some(13)),
            make_stage(8, d(2025, 1, 1), d(2025, 4, 1), Some(14)),
            make_stage(9, d(2025, 4, 1), d(2025, 7, 1), Some(15)),
        ];

        let derived =
            derive_downstream_par_order(&stages, &par_of_order_one(&stages), Some(&season_map));
        assert_eq!(derived, 1);

        let transitions = precompute_stage_lag_transitions(&stages, &season_map, 1);
        assert_eq!(flagged(&transitions, |t| t.rebuild_from_downstream), [6]);
        assert_eq!(
            flagged(&transitions, |t| t.accumulate_downstream),
            [3, 4, 5]
        );

        let cycles = SeasonCycles::new(&season_map);
        let groups: Vec<Option<usize>> = stages
            .iter()
            .map(|s| s.season_id.and_then(|id| cycles.group_of(id)))
            .collect();
        let first_group_change = groups
            .windows(2)
            .position(|pair| pair[0] != pair[1])
            .map(|index| index + 1);
        assert_eq!(
            first_group_change,
            Some(6),
            "the month-to-quarter step must sit where the resolution group changes"
        );
    }

    #[test]
    fn test_uniform_monthly_identity() {
        let season_map = monthly_season_map(MonthlyLabels::OneBased);
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
        let season_map = monthly_season_map(MonthlyLabels::OneBased);

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
        let season_map = monthly_season_map(MonthlyLabels::OneBased);
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
        let season_map = monthly_season_map(MonthlyLabels::OneBased);
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
        let season_map = monthly_season_map(MonthlyLabels::OneBased);
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
        let season_map = monthly_season_map(MonthlyLabels::OneBased);
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
    // Weekly / Custom period-window generalization (compute_period_transition)
    // -----------------------------------------------------------------------

    /// Four consecutive real ISO weeks of January 2024 (2024-01-01 is a Monday,
    /// ISO week 1), each stage exactly spanning its own week.
    #[test]
    fn test_weekly_cycle_per_week_finalizes() {
        let season_map = weekly_season_map();
        let stages = vec![
            make_stage(0, d(2024, 1, 1), d(2024, 1, 8), Some(0)),
            make_stage(1, d(2024, 1, 8), d(2024, 1, 15), Some(1)),
            make_stage(2, d(2024, 1, 15), d(2024, 1, 22), Some(2)),
            make_stage(3, d(2024, 1, 22), d(2024, 1, 29), Some(3)),
        ];

        let transitions = precompute_stage_lag_transitions(&stages, &season_map, 0);
        assert_eq!(transitions.len(), 4);

        let tol = 1e-10;
        for (i, t) in transitions.iter().enumerate() {
            assert!(
                (t.accumulate_weight - 1.0).abs() < tol,
                "stage {i}: accumulate_weight expected 1.0, got {}",
                t.accumulate_weight
            );
            assert!(
                t.spillover_weight.abs() < tol,
                "stage {i}: spillover_weight expected 0.0, got {}",
                t.spillover_weight
            );
            assert!(
                t.finalize_period,
                "stage {i}: finalize_period expected true (weekly PAR finalizes every stage)"
            );
        }
    }

    /// ISO week 53 of 2026 spans `[2026-12-28, 2027-01-04)`; `season_for_date`
    /// folds it to season id 51 (the same id as ISO week 52), but the physical
    /// window stays the real 7-day week-53 span.
    #[test]
    fn test_weekly_iso_week_53_folds() {
        let season_map = weekly_season_map();
        assert_eq!(season_map.season_for_date(d(2026, 12, 28)), Some(51));

        let stage = make_stage(0, d(2026, 12, 28), d(2027, 1, 4), Some(51));
        let transitions = precompute_stage_lag_transitions(&[stage], &season_map, 0);
        assert_eq!(transitions.len(), 1);

        let t = transitions[0];
        let tol = 1e-10;
        assert!(
            (t.accumulate_weight - 1.0).abs() < tol,
            "accumulate_weight expected 1.0 (the full physical week-53 span), got {}",
            t.accumulate_weight
        );
        assert!(
            t.spillover_weight.abs() < tol,
            "spillover_weight expected 0.0, got {}",
            t.spillover_weight
        );
        assert!(t.finalize_period, "single stage must finalize its period");
    }

    /// d30-style multi-resolution `Custom` map: a monthly definition (June) and
    /// a quarterly definition (Q3) in the SAME `season_map`. Each stage must
    /// weight against its OWN level's period hours, never a flattened cycle.
    fn custom_multi_resolution_season_map() -> SeasonMap {
        SeasonMap {
            cycle_type: SeasonCycleType::Custom,
            seasons: vec![
                SeasonDefinition {
                    id: 5,
                    label: "June".to_string(),
                    month_start: 6,
                    day_start: Some(1),
                    month_end: Some(6),
                    day_end: Some(30),
                },
                SeasonDefinition {
                    id: 12,
                    label: "Q3".to_string(),
                    month_start: 7,
                    day_start: Some(1),
                    month_end: Some(9),
                    day_end: Some(30),
                },
            ],
        }
    }

    #[test]
    fn test_custom_multi_resolution_each_stage_in_own_level() {
        let season_map = custom_multi_resolution_season_map();

        // June stage spans its whole month (30 days); the Q3 stage spans only
        // the first 30 days of the 92-day quarter — a sub-window, so its
        // weight can only match if it used the QUARTER's hours, not July's.
        let june_stage = make_stage(0, d(2024, 6, 1), d(2024, 7, 1), Some(5));
        let q3_partial_stage = make_stage(1, d(2024, 7, 1), d(2024, 7, 31), Some(12));
        let stages = vec![june_stage, q3_partial_stage];

        let transitions = precompute_stage_lag_transitions(&stages, &season_map, 0);
        assert_eq!(transitions.len(), 2);

        let tol = 1e-10;

        let june_hours = 30.0 * 24.0;
        let expected_june_weight = 30.0 * 24.0 / june_hours;
        assert!(
            (transitions[0].accumulate_weight - expected_june_weight).abs() < tol,
            "June stage must weight against its own month's hours ({june_hours}): expected {expected_june_weight}, got {}",
            transitions[0].accumulate_weight
        );
        assert!(transitions[0].finalize_period, "June stage must finalize");

        let q3_hours = 92.0 * 24.0;
        let expected_q3_weight = 30.0 * 24.0 / q3_hours;
        assert!(
            (transitions[1].accumulate_weight - expected_q3_weight).abs() < tol,
            "Q3 stage must weight against its own quarter's hours ({q3_hours}): expected {expected_q3_weight}, got {}",
            transitions[1].accumulate_weight
        );
        assert!(transitions[1].finalize_period, "Q3 stage must finalize");

        let flattened_to_july_weight = 30.0 * 24.0 / (31.0 * 24.0);
        assert!(
            (transitions[1].accumulate_weight - flattened_to_july_weight).abs() > 1e-3,
            "Q3 stage must not collapse to a monthly-level weight"
        );
    }

    // -----------------------------------------------------------------------
    // Migration-equivalence: window_period_overlaps route vs the
    // pre-migration days_in_period * 24.0 route
    // -----------------------------------------------------------------------

    /// Private copy of the pre-migration `days_in_period` formula, kept only
    /// to assert bit-equality against the `window_period_overlaps` route the
    /// production code now uses exclusively.
    fn reference_days_in_period(
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

    fn reference_weight(
        stage_start: NaiveDate,
        stage_end: NaiveDate,
        period_start: NaiveDate,
        period_end: NaiveDate,
        period_hours: f64,
    ) -> f64 {
        let days = reference_days_in_period(stage_start, stage_end, period_start, period_end);
        f64::from(days) * 24.0 / period_hours
    }

    #[test]
    fn test_migration_equivalence_window_period_overlaps_bit_identical_to_days_in_period() {
        // (a) stage straddling a monthly boundary: accumulate + nonzero spillover.
        let stage_monthly = make_stage(0, d(2026, 1, 28), d(2026, 2, 4), Some(0));
        let jan_start = d(2026, 1, 1);
        let jan_end = d(2026, 2, 1);
        let jan_hours = 31.0 * 24.0;
        let feb_start = d(2026, 2, 1);
        let feb_end = d(2026, 3, 1);
        let feb_hours = 28.0 * 24.0;

        let accumulate_new =
            single_period_overlap_hours(&stage_monthly, jan_start, jan_hours) / jan_hours;
        let accumulate_ref = reference_weight(
            stage_monthly.start_date,
            stage_monthly.end_date,
            jan_start,
            jan_end,
            jan_hours,
        );
        assert_eq!(
            accumulate_new, accumulate_ref,
            "monthly accumulate weight must be bit-identical"
        );

        let spillover_new =
            single_period_overlap_hours(&stage_monthly, feb_start, feb_hours) / feb_hours;
        let spillover_ref = reference_weight(
            stage_monthly.start_date,
            stage_monthly.end_date,
            feb_start,
            feb_end,
            feb_hours,
        );
        assert!(
            spillover_ref > 0.0,
            "fixture must exercise nonzero spillover"
        );
        assert_eq!(
            spillover_new, spillover_ref,
            "monthly spillover weight must be bit-identical"
        );

        // (b) Weekly full-week stage.
        let stage_weekly = make_stage(0, d(2024, 1, 1), d(2024, 1, 8), Some(0));
        let week_start = d(2024, 1, 1);
        let week_end = d(2024, 1, 8);
        let week_hours = 7.0 * 24.0;
        let weekly_new =
            single_period_overlap_hours(&stage_weekly, week_start, week_hours) / week_hours;
        let weekly_ref = reference_weight(
            stage_weekly.start_date,
            stage_weekly.end_date,
            week_start,
            week_end,
            week_hours,
        );
        assert_eq!(
            weekly_new, weekly_ref,
            "weekly full-week weight must be bit-identical"
        );

        // (c) Custom sub-window stage (the D30 shape): a Q3 stage spanning
        // only the first 30 days of a 92-day quarter.
        let stage_custom = make_stage(1, d(2024, 7, 1), d(2024, 7, 31), Some(12));
        let q3_start = d(2024, 7, 1);
        let q3_end = d(2024, 10, 1);
        let q3_hours = 92.0 * 24.0;
        let custom_new = single_period_overlap_hours(&stage_custom, q3_start, q3_hours) / q3_hours;
        let custom_ref = reference_weight(
            stage_custom.start_date,
            stage_custom.end_date,
            q3_start,
            q3_end,
            q3_hours,
        );
        assert_eq!(
            custom_new, custom_ref,
            "custom sub-window weight must be bit-identical"
        );

        // (d) quarterly downstream window: a stage straddling the Q3/Q4
        // boundary, mirroring compute_downstream_transitions's own weighting.
        let stage_quarterly = make_stage(0, d(2026, 9, 25), d(2026, 10, 4), Some(8));
        let q3_2026_start = d(2026, 7, 1);
        let q3_2026_end = d(2026, 10, 1);
        let q3_2026_hours = 92.0 * 24.0;
        let q4_2026_start = d(2026, 10, 1);
        let q4_2026_end = d(2027, 1, 1);
        let q4_2026_hours = 92.0 * 24.0;

        let downstream_accumulate_new =
            single_period_overlap_hours(&stage_quarterly, q3_2026_start, q3_2026_hours)
                / q3_2026_hours;
        let downstream_accumulate_ref = reference_weight(
            stage_quarterly.start_date,
            stage_quarterly.end_date,
            q3_2026_start,
            q3_2026_end,
            q3_2026_hours,
        );
        assert_eq!(
            downstream_accumulate_new, downstream_accumulate_ref,
            "quarterly downstream accumulate weight must be bit-identical"
        );

        let downstream_spillover_new =
            single_period_overlap_hours(&stage_quarterly, q4_2026_start, q4_2026_hours)
                / q4_2026_hours;
        let downstream_spillover_ref = reference_weight(
            stage_quarterly.start_date,
            stage_quarterly.end_date,
            q4_2026_start,
            q4_2026_end,
            q4_2026_hours,
        );
        assert!(
            downstream_spillover_ref > 0.0,
            "fixture must exercise nonzero downstream spillover"
        );
        assert_eq!(
            downstream_spillover_new, downstream_spillover_ref,
            "quarterly downstream spillover weight must be bit-identical"
        );
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

    // -----------------------------------------------------------------------
    // Seeded weekly-to-monthly finalize (item 9, RV0/RV1 fixtures)
    // -----------------------------------------------------------------------

    fn make_hydro(id: i32) -> Hydro {
        cobre_core::test_support::make_hydro(HydroSpec {
            id,
            name: format!("H{id}"),
            max_storage_hm3: 100.0,
            max_turbined_m3s: 100.0,
            max_generation_mw: 100.0,
            operational_start_date: date(2020, 1, 1),
            mirror_unit_group: MirrorUnitGroup::None,
            ..Default::default()
        })
    }

    /// Finalizes `derived`'s `accum`/`weight` seed through `stage_lag_transitions`
    /// for a single entity, accumulating `realized_per_stage[i]` at stage `i`;
    /// returns the resulting lag-1 value.
    fn finalize_seeded_single_entity(
        derived: &DerivedInflowSeeds,
        stage_lag_transitions: &[StageLagTransition],
        realized_per_stage: &[f64],
    ) -> f64 {
        let mut lag_state = vec![0.0_f64; 1];
        let mut accumulator = derived.accum.clone();
        let mut weight_accum = derived.weight.clone();
        let incoming_lags = vec![0.0_f64; 1];
        let mut downstream_accumulator: Vec<f64> = Vec::new();
        let mut downstream_weight_accum = 0.0_f64;
        let mut completed_lags: Vec<f64> = Vec::new();
        let mut n_completed = 0_usize;

        for (t, &realized) in realized_per_stage.iter().enumerate() {
            let mut primary = PrimaryLagAccum {
                accumulator: &mut accumulator,
                weight_accum: &mut weight_accum,
            };
            let mut downstream = DownstreamLagAccum {
                accumulator: &mut downstream_accumulator,
                weight_accum: &mut downstream_weight_accum,
                completed_lags: &mut completed_lags,
                n_completed: &mut n_completed,
                par_order: 0,
            };
            advance_lag_chain(
                EntityMajor {
                    entity_count: 1,
                    max_order: 1,
                },
                &mut lag_state,
                &incoming_lags,
                &[realized],
                &stage_lag_transitions[t],
                &mut primary,
                &mut downstream,
            );
        }

        lag_state[0]
    }

    /// Four weekly-shaped stages fully covering April 2026, study starting
    /// April 4 (a 3-day pre-study record seeds the in-progress accumulator).
    #[test]
    fn test_seeded_weekly_to_monthly_finalize_values_exact() {
        let season_map = monthly_season_map(MonthlyLabels::OneBased);
        let hydro_id = EntityId(1);
        let hydros = vec![make_hydro(1)];

        let stages = vec![
            make_stage(0, d(2026, 4, 4), d(2026, 4, 11), Some(3)),
            make_stage(1, d(2026, 4, 11), d(2026, 4, 18), Some(3)),
            make_stage(2, d(2026, 4, 18), d(2026, 4, 25), Some(3)),
            make_stage(3, d(2026, 4, 25), d(2026, 5, 1), Some(3)),
        ];
        let first_stage = stages[0].clone();

        let record = vec![InflowHistoryRow {
            hydro_id,
            start_date: d(2026, 4, 1),
            end_date: d(2026, 4, 4),
            value_m3s: 210.0,
        }];

        let derived = derive_inflow_seeds(&record, &[], &hydros, &first_stage, &season_map, 0);
        assert!(
            derived.weight[0] > 0.0 && derived.weight[0] < 1.0,
            "the pre-study seed must be a genuine partial-coverage fraction, \
             got weight={}",
            derived.weight[0]
        );

        let stage_lag_transitions = precompute_stage_lag_transitions(&stages, &season_map, 0);
        assert!(
            stage_lag_transitions[..3]
                .iter()
                .all(|t| !t.finalize_period),
            "only the last weekly stage may finalize April's monthly lag"
        );
        assert!(
            stage_lag_transitions[3].finalize_period,
            "the last weekly stage must finalize April's monthly lag"
        );

        let realized = [400.0, 420.0, 440.0, 460.0];
        let finalized = finalize_seeded_single_entity(&derived, &stage_lag_transitions, &realized);

        let expected = (210.0 * 3.0 + 400.0 * 7.0 + 420.0 * 7.0 + 440.0 * 7.0 + 460.0 * 6.0) / 30.0;
        assert_eq!(
            finalized, expected,
            "seeded weekly-to-monthly finalize must equal the exact \
             day-weighted average across the pre-study seed and the four \
             weekly stages"
        );
    }

    /// RV1: the same four weekly-shaped stages, but the seed is a single
    /// elapsed-week conditioning window straddling the March/April boundary
    /// (4 March days + 3 April days) instead of a pre-cut record row —
    /// exercising item 3's straddling-window contract (only the
    /// April-overlapping days seed the accumulator) through seed derivation.
    #[test]
    fn test_rv1_elapsed_week_conditioning_with_straddle() {
        let season_map = monthly_season_map(MonthlyLabels::OneBased);
        let hydro_id = EntityId(1);
        let hydros = vec![make_hydro(1)];

        let stages = vec![
            make_stage(0, d(2026, 4, 4), d(2026, 4, 11), Some(3)),
            make_stage(1, d(2026, 4, 11), d(2026, 4, 18), Some(3)),
            make_stage(2, d(2026, 4, 18), d(2026, 4, 25), Some(3)),
            make_stage(3, d(2026, 4, 25), d(2026, 5, 1), Some(3)),
        ];
        let first_stage = stages[0].clone();

        let conditioning = vec![RecentObservation {
            hydro_id,
            start_date: d(2026, 3, 28),
            end_date: d(2026, 4, 4),
            value_m3s: 220.0,
        }];

        let derived =
            derive_inflow_seeds(&[], &conditioning, &hydros, &first_stage, &season_map, 0);
        let expected_seed_weight = 3.0 * 24.0 / (30.0 * 24.0);
        assert_eq!(
            derived.weight[0], expected_seed_weight,
            "the straddling conditioning window must seed only its 3 \
             April-overlapping days, not the full 7-day observation"
        );

        let stage_lag_transitions = precompute_stage_lag_transitions(&stages, &season_map, 0);
        let realized = [400.0, 420.0, 440.0, 460.0];
        let finalized = finalize_seeded_single_entity(&derived, &stage_lag_transitions, &realized);

        let expected = (220.0 * 3.0 + 400.0 * 7.0 + 420.0 * 7.0 + 440.0 * 7.0 + 460.0 * 6.0) / 30.0;
        assert_eq!(
            finalized, expected,
            "the elapsed-week straddling conditioning must finalize to the \
             same exact day-weighted average as an equivalent non-straddling \
             seed"
        );
    }
}
