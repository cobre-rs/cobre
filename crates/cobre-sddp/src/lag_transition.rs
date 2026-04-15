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
/// Stages with `season_id = None` produce
/// `StageLagTransition { accumulate_weight: 0.0, spillover_weight: 0.0, finalize_period: false }`.
///
/// # Infallible
///
/// Invalid inputs (stages outside any season, empty season maps) produce
/// zero-weight entries. Upstream validation in `cobre-io` rejects structurally
/// invalid inputs before this function is called.
pub fn precompute_stage_lag_transitions(
    stages: &[Stage],
    season_map: &SeasonMap,
) -> Vec<StageLagTransition> {
    let noop = StageLagTransition {
        accumulate_weight: 0.0,
        spillover_weight: 0.0,
        finalize_period: false,
    };

    stages
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
        .collect()
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

    // -----------------------------------------------------------------------
    // Test 1: uniform monthly identity
    // -----------------------------------------------------------------------

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

        let transitions = precompute_stage_lag_transitions(&stages, &season_map);

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

    // -----------------------------------------------------------------------
    // Test 2: PMO_APR_2026_rv0 production DECOMP trace
    // -----------------------------------------------------------------------

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

        let transitions = precompute_stage_lag_transitions(&stages, &season_map);
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

        let transitions = precompute_stage_lag_transitions(&stages, &season_map);
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

        let transitions = precompute_stage_lag_transitions(&stages, &season_map);
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

        let transitions = precompute_stage_lag_transitions(&stages, &season_map);
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

        let transitions = precompute_stage_lag_transitions(&stages, &season_map);
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
}
