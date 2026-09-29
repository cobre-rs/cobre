//! Layer 5a — water travel-time arc validation (config-time gates only).
//!
//! Validates `Hydro::travel_time_hours` and its `InitialConditions::past_defluences`
//! history before the arc is sized into any solver-side state. Runtime
//! conservation checks and recourse-feasibility rows live downstream, not here.
//!
//! | # | Rule                                                                    | `ErrorKind`               |
//! |---|--------------------------------------------------------------------------|---------------------------|
//! | 1 | `travel_time_hours` negative or non-finite                              | `InvalidValue`            |
//! | 2 | `travel_time_hours == 0.0` — treated as undeclared, no arc created      | `ModelQuality` (warning)  |
//! | 3 | `max_t(t_v / h_t)` below [`NEGLIGIBLE_RATIO_THRESHOLD`]                  | `ModelQuality` (warning)  |
//! | 4 | `t_v` exceeds the remaining study horizon at some stage                 | `ModelQuality` (warning)  |
//! | 5 | `past_defluences` windows do not cover the arc's in-transit span `[start_0 − t_v, start_0)` (gap or no windows) | `BusinessRuleViolation` |
//! | 5b | A `past_defluences` window ends after `start_0` (future-dated) | `InvalidValue` |
//! | 6 | Chronological confluence: 2+ declared arcs into one downstream plant with differing `travel_time_hours`, while any study stage is chronological | `NotImplemented` |
//! | 11 | A declared arc's downstream `exit_stage_id` falls inside the arrival window of a release whose own stage is still Operating | `ModelQuality` (warning) |
//! | 12 | A declared arc releases at a stage where its downstream has not yet reached Operating status (`PreFilling`/`Filling`, or before `entry_stage_id`) | `BusinessRuleViolation` |
//! | 13 | *(retired — number never reused)* | |

use chrono::NaiveDate;
use cobre_core::{
    BlockMode, EntityId, Hydro, Stage, window_period_reach_depth, window_reaches_any_period,
};

use super::super::{ErrorKind, ValidationContext, schema::ParsedData};

/// Below this `max_t(t_v/h_t)` ratio, cross-stage transport carries a
/// mass-fraction small enough to treat as negligible.
const NEGLIGIBLE_RATIO_THRESHOLD: f64 = 0.01;

pub(super) fn validate_travel_time(data: &ParsedData, ctx: &mut ValidationContext) {
    let study_durations = study_stage_durations(data);
    let start_0 = study_start_date(data);

    for hydro in &data.hydros {
        let Some(t) = hydro.travel_time_hours else {
            continue;
        };
        let hydro_id = hydro.id.0;

        if !t.is_finite() || t < 0.0 {
            ctx.add_error(
                ErrorKind::InvalidValue,
                "system/hydros.json",
                Some(format!("Hydro {hydro_id}")),
                format!("Hydro {hydro_id}: travel_time_hours must be finite and >= 0.0, got {t}"),
            );
            continue;
        }

        if t == 0.0 {
            ctx.add_warning(
                ErrorKind::ModelQuality,
                "system/hydros.json",
                Some(format!("Hydro {hydro_id}")),
                format!(
                    "Hydro {hydro_id}: travel_time_hours == 0.0 is treated as undeclared \
                     (instantaneous transfer); no cross-stage arc is created"
                ),
            );
            continue;
        }

        check_negligible_ratio(hydro_id, t, &study_durations, ctx);
        check_horizon_inertness(hydro_id, t, &study_durations, ctx);
        if let Some(start_0) = start_0 {
            check_defluence_coverage(hydro, t, start_0, data, ctx);
        }
    }

    check_chronological_confluence_heterogeneous_travel_time(data, ctx);
    check_recourse_downstream_not_operating(data, &study_durations, ctx);
    check_recourse_downstream_exit_within_window(data, &study_durations, ctx);
}

/// Whether `hydro` has not reached full commissioning at `stage_id` for a
/// reason OTHER than having exited: still in its own pre-`start_stage_id`
/// sub-phase, still `Filling` (short of `entry_stage_id`), or (a non-filling
/// hydro) still short of `entry_stage_id`. A hydro past its `exit_stage_id`
/// (which always exceeds `entry_stage_id`, `check_lifecycle_consistency`)
/// returns `false` here — that reversion is row 11's territory, never row
/// 12's, since a filling hydro never carries `exit_stage_id`
/// (`check_filling_guards` guard 5).
fn hydro_not_yet_entered(hydro: &Hydro, stage_id: i32) -> bool {
    if let Some(filling) = &hydro.filling
        && filling.start_stage_id > 0
        && stage_id < filling.start_stage_id
    {
        return true;
    }
    hydro.entry_stage_id.is_some_and(|entry| stage_id < entry)
}

/// Deepest future stage a release anchored at `anchor` reaches on the study
/// stage clock: the same [`window_period_reach_depth`] sweep
/// [`check_horizon_inertness`] already reuses, restricted to `anchor`'s own
/// remaining calendar so a horizon-truncated arrival never overstates depth.
fn arrival_depth(t: f64, anchor: usize, study_durations: &[f64]) -> usize {
    let future = &study_durations[anchor..];
    window_period_reach_depth(t, future[0], future)
}

/// Row 12: rejects a declared arc that releases while its downstream has not
/// yet entered — the `PreFilling` short-circuit is same-stage and cannot
/// carry a delayed delivery into an absent balance row, and a `Filling`
/// downstream's sufficiency budget does not model bucket-borne arrivals.
/// Checking only the release (anchor) stage suffices: a
/// downstream's phase never reverts to non-Operating except via
/// `exit_stage_id`, which [`hydro_not_yet_entered`] excludes and row 11 owns.
fn check_recourse_downstream_not_operating(
    data: &ParsedData,
    study_durations: &[f64],
    ctx: &mut ValidationContext,
) {
    for hydro in &data.hydros {
        let Some(t) = hydro.travel_time_hours.filter(|&t| t > 0.0) else {
            continue;
        };
        let Some(downstream_id) = hydro.downstream_id else {
            continue;
        };
        let Some(downstream) = data.hydros.iter().find(|h| h.id == downstream_id) else {
            continue;
        };

        for anchor in 0..study_durations.len() {
            let anchor_id = i32::try_from(anchor).unwrap_or(i32::MAX);
            if !hydro_not_yet_entered(downstream, anchor_id) {
                continue;
            }
            let window_end = anchor + arrival_depth(t, anchor, study_durations);
            ctx.add_error(
                ErrorKind::BusinessRuleViolation,
                "system/hydros.json",
                Some(format!("Hydro {}", downstream_id.0)),
                format!(
                    "Hydro {}: declared arc from hydro {} (travel_time_hours={t}) releases at \
                     stage {anchor} (arrival window [stage {anchor}, stage {window_end}]) while \
                     hydro {} has not reached Operating status there (PreFilling/Filling, or \
                     before entry_stage_id); a same-stage short-circuit cannot carry a delayed \
                     delivery into an absent balance row",
                    downstream_id.0, hydro.id.0, downstream_id.0
                ),
            );
            break;
        }
    }
}

/// Row 11: a declared arc whose release stage is Operating, but whose
/// downstream's `exit_stage_id` falls at or before the arrival window's far
/// stage, delivers into a plant that has since reverted to `PreFilling` —
/// the maturing delivery past `exit_stage_id` is silently dropped (no
/// consuming term on the frozen balance row), mirroring the terminal-drop
/// convention: an advisory, never an error.
fn check_recourse_downstream_exit_within_window(
    data: &ParsedData,
    study_durations: &[f64],
    ctx: &mut ValidationContext,
) {
    for hydro in &data.hydros {
        let Some(t) = hydro.travel_time_hours.filter(|&t| t > 0.0) else {
            continue;
        };
        let Some(downstream_id) = hydro.downstream_id else {
            continue;
        };
        let Some(downstream) = data.hydros.iter().find(|h| h.id == downstream_id) else {
            continue;
        };
        let Some(exit) = downstream.exit_stage_id else {
            continue;
        };

        for anchor in 0..study_durations.len() {
            let anchor_id = i32::try_from(anchor).unwrap_or(i32::MAX);
            if hydro_not_yet_entered(downstream, anchor_id) {
                continue;
            }
            let window_end = anchor + arrival_depth(t, anchor, study_durations);
            let window_end_id = i32::try_from(window_end).unwrap_or(i32::MAX);
            if exit > window_end_id {
                continue;
            }
            ctx.add_warning(
                ErrorKind::ModelQuality,
                "system/hydros.json",
                Some(format!("Hydro {}", downstream_id.0)),
                format!(
                    "Hydro {}: declared arc from hydro {} (travel_time_hours={t}) released at \
                     stage {anchor} carries a delivery into arrival window [stage {anchor}, \
                     stage {window_end}] that reaches or crosses hydro {}'s exit_stage_id \
                     ({exit}); the maturing delivery past exit is dropped, not consumed",
                    downstream_id.0, hydro.id.0, downstream_id.0
                ),
            );
            break;
        }
    }
}

/// Row 6: rejects a superset of the true heterogeneous-confluence cases (any
/// chronological study stage, not only the ones whose per-stage-pair spread
/// resolution actually disagrees) — this infrastructure crate has no access
/// to that downstream, per-arc computation and must not reproduce it.
fn check_chronological_confluence_heterogeneous_travel_time(
    data: &ParsedData,
    ctx: &mut ValidationContext,
) {
    let any_chronological = data
        .stages
        .stages
        .iter()
        .any(|s| s.id >= 0 && s.block_mode == BlockMode::Chronological);
    if !any_chronological {
        return;
    }

    let mut by_downstream: Vec<(EntityId, Vec<(i32, f64)>)> = Vec::new();
    for hydro in &data.hydros {
        let Some(t) = hydro
            .travel_time_hours
            .filter(|&t| t.is_finite() && t > 0.0)
        else {
            continue;
        };
        let Some(downstream_id) = hydro.downstream_id else {
            continue;
        };
        match by_downstream
            .iter_mut()
            .find(|(id, _)| *id == downstream_id)
        {
            Some((_, arcs)) => arcs.push((hydro.id.0, t)),
            None => by_downstream.push((downstream_id, vec![(hydro.id.0, t)])),
        }
    }

    for (downstream_id, arcs) in &by_downstream {
        if arcs.len() < 2 {
            continue;
        }
        let first_t = arcs[0].1;
        if arcs.iter().all(|&(_, t)| (t - first_t).abs() < 1e-9) {
            continue;
        }
        let arc_desc = arcs
            .iter()
            .map(|(id, t)| format!("hydro {id} (travel_time_hours={t})"))
            .collect::<Vec<_>>()
            .join(", ");
        ctx.add_error(
            ErrorKind::NotImplemented,
            "system/hydros.json",
            Some(format!("Hydro {downstream_id}")),
            format!(
                "Hydro {downstream_id}: chronological confluence with heterogeneous travel \
                 times is unsupported in v1 — {} declared arcs feed this downstream plant with \
                 differing travel_time_hours ({arc_desc}); align the arcs' travel_time_hours or \
                 keep every study stage in Parallel mode",
                arcs.len()
            ),
        );
    }
}

/// Study-stage (`id >= 0`) durations in canonical (ascending `id`) order,
/// each summed from its blocks (blocks sum to the stage duration).
pub(super) fn study_stage_durations(data: &ParsedData) -> Vec<f64> {
    data.stages
        .stages
        .iter()
        .filter(|s| s.id >= 0)
        .map(Stage::total_hours)
        .collect()
}

/// The first study stage's start date (`id >= 0`, lowest `id`; stages are
/// canonical-sorted ascending). `None` when the study declares no study stages.
fn study_start_date(data: &ParsedData) -> Option<NaiveDate> {
    data.stages
        .stages
        .iter()
        .filter(|s| s.id >= 0)
        .min_by_key(|s| s.id)
        .map(|s| s.start_date)
}

/// Row 3: `max_t(t_v/h_t)` below [`NEGLIGIBLE_RATIO_THRESHOLD`] is an advisory
/// ("consider not declaring"), never a silent fold.
fn check_negligible_ratio(
    hydro_id: i32,
    t: f64,
    study_durations: &[f64],
    ctx: &mut ValidationContext,
) {
    if study_durations.is_empty() {
        return;
    }
    let max_ratio = study_durations
        .iter()
        .fold(f64::NEG_INFINITY, |acc, &h| acc.max(t / h));
    if max_ratio >= NEGLIGIBLE_RATIO_THRESHOLD {
        return;
    }
    ctx.add_warning(
        ErrorKind::ModelQuality,
        "system/hydros.json",
        Some(format!("Hydro {hydro_id}")),
        format!(
            "Hydro {hydro_id}: travel_time_hours ({t}) is negligible relative to every study \
             stage length (max t_v/h_t = {max_ratio:.4} < {NEGLIGIBLE_RATIO_THRESHOLD}); \
             consider not declaring this arc"
        ),
    );
}

/// Row 4: `t_v` exceeding the remaining study horizon at some stage — the
/// arc's release never arrives before the horizon ends from that stage
/// onward. Routed through [`window_reaches_any_period`] (sharing the same
/// overlap sweep [`arrival_depth`] reuses via [`window_period_reach_depth`])
/// rather than a hand-rolled remaining-hours sum, so a future change to the
/// arrival window's definition cannot silently diverge this check from the
/// rest of the feature. Sizing stays safe (depth is capped by `n_stages -
/// t`), so this is an advisory, never an error.
fn check_horizon_inertness(
    hydro_id: i32,
    t: f64,
    study_durations: &[f64],
    ctx: &mut ValidationContext,
) {
    for stage_t in 0..study_durations.len() {
        let future = &study_durations[stage_t..];
        if window_reaches_any_period(t, future[0], future) {
            continue;
        }
        ctx.add_warning(
            ErrorKind::ModelQuality,
            "system/hydros.json",
            Some(format!("Hydro {hydro_id}")),
            format!(
                "Hydro {hydro_id}: travel_time_hours ({t}) exceeds the remaining study horizon \
                 from stage {stage_t}; the arc is economically inert from stage {stage_t} onward"
            ),
        );
        return;
    }
}

/// Tolerance for the coverage sweep. Window offsets are whole-day multiples of
/// 24 hours (day-resolution dates), so contiguity and reach compare exactly;
/// the slack only absorbs a fractional `travel_time_hours`.
const COVERAGE_EPS: f64 = 1e-6;

/// Hours of wall clock between `date` and `start_0`, positive when `date`
/// precedes `start_0`.
// Rationale: pre-study spans are on the order of years (<1e6 hours), far under
// f64's exact-integer range (2^52); a checked conversion buys nothing.
#[allow(clippy::cast_precision_loss)]
fn hours_before(start_0: NaiveDate, date: NaiveDate) -> f64 {
    (start_0 - date).num_hours() as f64
}

/// Row 5: a declared arc's `past_defluences` windows must cover the in-transit
/// span `(0, t_v]` hours before the first study stage's start (`start_0`).
///
/// Each window `[start_date, end_date)` maps to the hours-before-`start_0`
/// interval `[e_off, s_off)` with `e_off = start_0 − end_date` and
/// `s_off = start_0 − start_date`. Sweeping the windows most-recent-first, the
/// newest must end at `start_0` (`e_off == 0`) and the run must stay contiguous
/// (each next `e_off` at the prior `s_off`, windows being non-overlapping per
/// the loader) until it reaches `s_off ≥ t`. A gap or missing history leaves
/// part of `(0, t_v]` uncovered — a hard error naming the uncovered span, never
/// a silent zero-seed. A window ending after `start_0` is future-dated: it
/// cannot seed pre-study transit and is rejected outright.
fn check_defluence_coverage(
    hydro: &Hydro,
    t: f64,
    start_0: NaiveDate,
    data: &ParsedData,
    ctx: &mut ValidationContext,
) {
    if hydro.downstream_id.is_none() {
        return;
    }
    let hydro_id = hydro.id.0;

    let mut windows: Vec<(f64, f64)> = Vec::new();
    for w in data
        .initial_conditions
        .past_defluences
        .iter()
        .filter(|e| e.hydro_id.0 == hydro_id)
    {
        if w.end_date > start_0 {
            ctx.add_error(
                ErrorKind::InvalidValue,
                "initial_conditions.json",
                Some(format!("Hydro {hydro_id}")),
                format!(
                    "Hydro {hydro_id}: past_defluences window [{}, {}) ends after the study \
                     start {start_0}; a defluence window must be entirely pre-study",
                    w.start_date, w.end_date
                ),
            );
            return;
        }
        windows.push((
            hours_before(start_0, w.end_date),
            hours_before(start_0, w.start_date),
        ));
    }

    windows.sort_by(|a, b| a.0.total_cmp(&b.0));

    let mut covered = 0.0_f64;
    for (e_off, s_off) in windows {
        if e_off > covered + COVERAGE_EPS {
            break;
        }
        covered = covered.max(s_off);
    }

    if covered + COVERAGE_EPS >= t {
        return;
    }

    ctx.add_error(
        ErrorKind::BusinessRuleViolation,
        "initial_conditions.json",
        Some(format!("Hydro {hydro_id}")),
        format!(
            "Hydro {hydro_id}: declared arc (travel_time_hours={t}) requires past_defluences \
             windows covering (0, {t}] hours before the study start {start_0}; coverage reaches \
             only {covered} hour(s), leaving ({covered}, {t}] uncovered — supply contiguous \
             pre-study release windows ending at {start_0} and reaching {t} hours back"
        ),
    );
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::panic,
    clippy::too_many_lines,
    clippy::doc_markdown,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]
mod tests {
    use super::*;
    use crate::stages::StagesData;
    use crate::test_support::*;
    use cobre_core::entities::Hydro;
    use cobre_core::temporal::{Block, PolicyGraphType, Stage};
    use cobre_core::{EntityId, HorizonGraph, HydroPastDefluence};

    fn make_hydro_with_travel_time(id: i32, downstream_id: i32, t: Option<f64>) -> Hydro {
        let mut h = make_hydro(id, Some(downstream_id));
        h.travel_time_hours = t;
        h
    }

    /// One study stage (`id`) carrying a single block of `duration_hours`.
    fn make_study_stage(id: i32, duration_hours: f64) -> Stage {
        let mut stage = make_stage(id);
        stage.blocks = vec![Block {
            index: 0,
            name: "FLAT".to_string(),
            duration_hours,
        }];
        stage
    }

    /// `n_study` study stages of `study_duration_hours` each, preceded by
    /// `n_pre_study` pre-study stages of `pre_study_duration_hours` each
    /// (`id = -1` most recent through `id = -n_pre_study` oldest).
    fn make_stages_with_pre_study(
        n_study: i32,
        study_duration_hours: f64,
        n_pre_study: i32,
        pre_study_duration_hours_per_period: f64,
    ) -> StagesData {
        let mut stages: Vec<Stage> = Vec::new();
        for id in (1..=n_pre_study).rev() {
            let mut stage = make_stage(-id);
            let days = (pre_study_duration_hours_per_period / 24.0).round() as i64;
            stage.start_date = chrono::NaiveDate::from_ymd_opt(2023, 1, 1).unwrap();
            stage.end_date = stage.start_date + chrono::Duration::days(days);
            stages.push(stage);
        }
        for id in 0..n_study {
            stages.push(make_study_stage(id, study_duration_hours));
        }
        StagesData {
            openings_declared: std::collections::HashSet::new(),
            stages,
            policy_graph: HorizonGraph {
                stage_discount_rate_overrides: std::collections::BTreeMap::new(),
                graph_type: PolicyGraphType::FiniteHorizon,
                annual_discount_rate: 0.06,
                transitions: vec![],
                nodes: Vec::new(),
                season_map: None,
            },
        }
    }

    /// The default study start date every `make_study_stage` carries (via
    /// `make_stage`): the anchor `check_defluence_coverage` measures against.
    fn study_start() -> NaiveDate {
        chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap()
    }

    /// A single past-defluence window `[start, end)` at 100 m³/s.
    fn defluence_window(hydro_id: i32, start: NaiveDate, end: NaiveDate) -> HydroPastDefluence {
        HydroPastDefluence {
            hydro_id: EntityId::from(hydro_id),
            start_date: start,
            end_date: end,
            value_m3s: 100.0,
        }
    }

    /// One window ending at [`study_start`] and reaching `t` hours back — enough
    /// to cover `(0, t]` so the non-row-5 tests never trip the coverage gate.
    fn covering_defluences(hydro_id: i32, t: f64) -> Vec<HydroPastDefluence> {
        let days = (t / 24.0).ceil().max(1.0) as i64;
        vec![defluence_window(
            hydro_id,
            study_start() - chrono::Duration::days(days),
            study_start(),
        )]
    }

    // ── Row 1: negative / non-finite ──────────────────────────────────────────

    #[test]
    fn test_negative_travel_time_hard_errors_naming_hydro() {
        let hydro = make_hydro_with_travel_time(1, 2, Some(-5.0));
        let stages = make_stages_with_pre_study(3, 720.0, 2, 720.0);
        let data = make_data(vec![hydro], vec![], vec![], stages, vec![], vec![]);

        let mut ctx = ValidationContext::new();
        validate_travel_time(&data, &mut ctx);

        assert!(ctx.has_errors(), "negative travel_time_hours must error");
        let errors = ctx.errors();
        assert!(
            errors.iter().any(|e| e.kind == ErrorKind::InvalidValue
                && e.message.contains("Hydro 1")
                && e.message.contains("travel_time_hours")),
            "error must name the offending hydro, got: {errors:?}"
        );
    }

    #[test]
    fn test_nan_travel_time_hard_errors() {
        let hydro = make_hydro_with_travel_time(3, 4, Some(f64::NAN));
        let stages = make_stages_with_pre_study(3, 720.0, 2, 720.0);
        let data = make_data(vec![hydro], vec![], vec![], stages, vec![], vec![]);

        let mut ctx = ValidationContext::new();
        validate_travel_time(&data, &mut ctx);

        assert!(ctx.has_errors(), "NaN travel_time_hours must error");
        assert!(
            ctx.errors()
                .iter()
                .any(|e| e.kind == ErrorKind::InvalidValue && e.message.contains("Hydro 3")),
        );
    }

    #[test]
    fn test_positive_travel_time_no_row1_error() {
        let hydro = make_hydro_with_travel_time(1, 2, Some(48.0));
        let stages = make_stages_with_pre_study(3, 720.0, 2, 720.0);
        let mut data = make_data(vec![hydro], vec![], vec![], stages, vec![], vec![]);
        data.initial_conditions.past_defluences = covering_defluences(1, 48.0);

        let mut ctx = ValidationContext::new();
        validate_travel_time(&data, &mut ctx);

        assert!(
            !ctx.has_errors(),
            "a well-formed declared arc must not hard-error, got: {:?}",
            ctx.errors()
        );
    }

    // ── Row 2: zero == undeclared, advisory only ────────────────────────────────

    #[test]
    fn test_zero_travel_time_is_advisory_not_error() {
        let hydro = make_hydro_with_travel_time(1, 2, Some(0.0));
        let stages = make_stages_with_pre_study(3, 720.0, 0, 720.0);
        let data = make_data(vec![hydro], vec![], vec![], stages, vec![], vec![]);

        let mut ctx = ValidationContext::new();
        validate_travel_time(&data, &mut ctx);

        assert!(
            !ctx.has_errors(),
            "travel_time_hours == 0.0 is not an error"
        );
        assert!(
            ctx.warnings()
                .iter()
                .any(|w| w.kind == ErrorKind::ModelQuality
                    && w.message.contains("Hydro 1")
                    && w.message.contains("undeclared")),
            "expected an advisory naming the hydro, got: {:?}",
            ctx.warnings()
        );
    }

    #[test]
    fn test_none_travel_time_no_diagnostics() {
        let hydro = make_hydro_with_travel_time(1, 2, None);
        let stages = make_stages_with_pre_study(3, 720.0, 0, 720.0);
        let data = make_data(vec![hydro], vec![], vec![], stages, vec![], vec![]);

        let mut ctx = ValidationContext::new();
        validate_travel_time(&data, &mut ctx);

        assert!(!ctx.has_errors());
        assert!(
            ctx.warnings().is_empty(),
            "None travel_time_hours (today's instantaneous model) must be silent"
        );
    }

    // ── Row 3: negligible ratio advisory ─────────────────────────────────────

    #[test]
    fn test_negligible_ratio_emits_advisory() {
        // t_v/h_t = 6/720 ~= 0.0083 < 0.01 threshold.
        let hydro = make_hydro_with_travel_time(1, 2, Some(6.0));
        let stages = make_stages_with_pre_study(3, 720.0, 1, 720.0);
        let mut data = make_data(vec![hydro], vec![], vec![], stages, vec![], vec![]);
        data.initial_conditions.past_defluences = covering_defluences(1, 6.0);

        let mut ctx = ValidationContext::new();
        validate_travel_time(&data, &mut ctx);

        assert!(!ctx.has_errors());
        assert!(
            ctx.warnings()
                .iter()
                .any(|w| w.message.contains("negligible") && w.message.contains("Hydro 1")),
            "expected a negligible-ratio advisory, got: {:?}",
            ctx.warnings()
        );
    }

    #[test]
    fn test_non_negligible_ratio_no_advisory() {
        // t_v/h_t = 360/720 = 0.5, well above the threshold.
        let hydro = make_hydro_with_travel_time(1, 2, Some(360.0));
        let stages = make_stages_with_pre_study(3, 720.0, 1, 720.0);
        let mut data = make_data(vec![hydro], vec![], vec![], stages, vec![], vec![]);
        data.initial_conditions.past_defluences = covering_defluences(1, 360.0);

        let mut ctx = ValidationContext::new();
        validate_travel_time(&data, &mut ctx);

        assert!(
            !ctx.warnings()
                .iter()
                .any(|w| w.message.contains("negligible")),
            "a non-negligible ratio must not advise, got: {:?}",
            ctx.warnings()
        );
    }

    // ── Row 4: horizon-inertness advisory ─────────────────────────────────────

    #[test]
    fn test_travel_time_exceeds_tail_horizon_emits_advisory() {
        // 3 monthly (720h) study stages; t_v = 1500h exceeds the last stage's
        // remaining horizon (720h at stage 2, 1440h at stage 1) but not stage
        // 0's (2160h) — inert from stage 1 onward.
        let hydro = make_hydro_with_travel_time(1, 2, Some(1500.0));
        let stages = make_stages_with_pre_study(3, 720.0, 2, 720.0);
        let mut data = make_data(vec![hydro], vec![], vec![], stages, vec![], vec![]);
        data.initial_conditions.past_defluences = covering_defluences(1, 1500.0);

        let mut ctx = ValidationContext::new();
        validate_travel_time(&data, &mut ctx);

        assert!(!ctx.has_errors());
        assert!(
            ctx.warnings()
                .iter()
                .any(|w| w.message.contains("economically inert") && w.message.contains("stage 1")),
            "expected the first-onset inert-stage advisory, got: {:?}",
            ctx.warnings()
        );
    }

    #[test]
    fn test_travel_time_within_horizon_no_inertness_advisory() {
        let hydro = make_hydro_with_travel_time(1, 2, Some(48.0));
        let stages = make_stages_with_pre_study(3, 720.0, 1, 720.0);
        let mut data = make_data(vec![hydro], vec![], vec![], stages, vec![], vec![]);
        data.initial_conditions.past_defluences = covering_defluences(1, 48.0);

        let mut ctx = ValidationContext::new();
        validate_travel_time(&data, &mut ctx);

        assert!(
            !ctx.warnings()
                .iter()
                .any(|w| w.message.contains("economically inert")),
            "a short travel time within every stage's reach must not advise, got: {:?}",
            ctx.warnings()
        );
    }

    // ── Row 5: past_defluences windowed coverage of (0, t_v] ──────────────────

    #[test]
    fn test_sufficient_coverage_no_row5_diagnostics() {
        // t_v = 48h; one window [2023-12-30, 2024-01-01) ending at start_0 covers
        // the full 48h span back from start_0.
        let hydro = make_hydro_with_travel_time(1, 2, Some(48.0));
        let stages = make_stages_with_pre_study(3, 720.0, 2, 720.0);
        let mut data = make_data(vec![hydro], vec![], vec![], stages, vec![], vec![]);
        data.initial_conditions.past_defluences = vec![defluence_window(
            1,
            chrono::NaiveDate::from_ymd_opt(2023, 12, 30).unwrap(),
            study_start(),
        )];

        let mut ctx = ValidationContext::new();
        validate_travel_time(&data, &mut ctx);

        assert!(
            !ctx.has_errors(),
            "full coverage must not error: {:?}",
            ctx.errors()
        );
        assert!(
            !ctx.warnings()
                .iter()
                .any(|w| w.message.contains("past_defluences")),
            "full coverage must not advise, got: {:?}",
            ctx.warnings()
        );
    }

    #[test]
    fn test_undercoverage_hard_errors_naming_span() {
        // t_v = 48h but the only window [2023-12-31, 2024-01-01) covers just 24h,
        // leaving (24, 48] uncovered.
        let hydro = make_hydro_with_travel_time(1, 2, Some(48.0));
        let stages = make_stages_with_pre_study(3, 720.0, 2, 720.0);
        let mut data = make_data(vec![hydro], vec![], vec![], stages, vec![], vec![]);
        data.initial_conditions.past_defluences = vec![defluence_window(
            1,
            chrono::NaiveDate::from_ymd_opt(2023, 12, 31).unwrap(),
            study_start(),
        )];

        let mut ctx = ValidationContext::new();
        validate_travel_time(&data, &mut ctx);

        assert!(ctx.has_errors(), "under-coverage must hard-error");
        let errors = ctx.errors();
        assert!(
            errors
                .iter()
                .any(|e| e.kind == ErrorKind::BusinessRuleViolation
                    && e.message.contains("Hydro 1")
                    && e.message.contains("48")
                    && e.message.contains("uncovered")),
            "error must name the hydro, t_v, and the uncovered span, got: {errors:?}"
        );
    }

    #[test]
    fn test_no_windows_hard_errors() {
        // A declared arc with an empty past_defluences leaves (0, 48] fully
        // uncovered — a hard error, never a silent zero-seed (no fallback).
        let hydro = make_hydro_with_travel_time(1, 2, Some(48.0));
        let stages = make_stages_with_pre_study(3, 720.0, 2, 720.0);
        let data = make_data(vec![hydro], vec![], vec![], stages, vec![], vec![]);

        let mut ctx = ValidationContext::new();
        validate_travel_time(&data, &mut ctx);

        assert!(ctx.has_errors(), "no windows must hard-error");
        let errors = ctx.errors();
        assert!(
            errors
                .iter()
                .any(|e| e.kind == ErrorKind::BusinessRuleViolation
                    && e.message.contains("Hydro 1")
                    && e.message.contains("uncovered")),
            "error must name the hydro and the uncovered span, got: {errors:?}"
        );
    }

    #[test]
    fn test_future_dated_window_invalid_value() {
        // A window whose end_date (2024-01-05) is after start_0 (2024-01-01)
        // cannot seed pre-study transit.
        let hydro = make_hydro_with_travel_time(1, 2, Some(48.0));
        let stages = make_stages_with_pre_study(3, 720.0, 2, 720.0);
        let mut data = make_data(vec![hydro], vec![], vec![], stages, vec![], vec![]);
        data.initial_conditions.past_defluences = vec![defluence_window(
            1,
            chrono::NaiveDate::from_ymd_opt(2023, 12, 30).unwrap(),
            chrono::NaiveDate::from_ymd_opt(2024, 1, 5).unwrap(),
        )];

        let mut ctx = ValidationContext::new();
        validate_travel_time(&data, &mut ctx);

        assert!(ctx.has_errors(), "a future-dated window must hard-error");
        let errors = ctx.errors();
        assert!(
            errors.iter().any(|e| e.kind == ErrorKind::InvalidValue
                && e.message.contains("Hydro 1")
                && e.message.contains("after the study start")),
            "error must flag the future-dated window naming the hydro, got: {errors:?}"
        );
    }

    // ── Row 6: chronological confluence with heterogeneous travel time ───────

    /// Hydro 1 and hydro 2 both feed downstream hydro 3 with `travel_time_hours`
    /// `t1`/`t2`; the middle study stage runs `Chronological` when
    /// `chronological` is `true` (else every stage stays the default `Parallel`).
    fn confluence_data(t1: f64, t2: f64, chronological: bool) -> ParsedData {
        let hydro1 = make_hydro_with_travel_time(1, 3, Some(t1));
        let hydro2 = make_hydro_with_travel_time(2, 3, Some(t2));
        let hydro3 = make_hydro(3, None);
        let mut stages = make_stages_with_pre_study(3, 720.0, 1, 720.0);
        if chronological {
            stages
                .stages
                .iter_mut()
                .find(|s| s.id == 1)
                .unwrap()
                .block_mode = BlockMode::Chronological;
        }
        let mut data = make_data(
            vec![hydro1, hydro2, hydro3],
            vec![],
            vec![],
            stages,
            vec![],
            vec![],
        );
        data.initial_conditions.past_defluences = vec![
            covering_defluences(1, t1).remove(0),
            covering_defluences(2, t2).remove(0),
        ];
        data
    }

    #[test]
    fn test_chronological_confluence_heterogeneous_travel_time_hard_errors() {
        let data = confluence_data(48.0, 96.0, true);

        let mut ctx = ValidationContext::new();
        validate_travel_time(&data, &mut ctx);

        assert!(
            ctx.has_errors(),
            "chronological confluence with differing travel_time_hours must hard-error"
        );
        let errors = ctx.errors();
        assert!(
            errors.iter().any(|e| e.kind == ErrorKind::NotImplemented
                && e.message.contains("Hydro 3")
                && e.message.contains("chronological confluence")),
            "error must name the downstream plant and the condition, got: {errors:?}"
        );
    }

    #[test]
    fn test_chronological_confluence_equal_travel_time_no_row6_error() {
        let data = confluence_data(48.0, 48.0, true);

        let mut ctx = ValidationContext::new();
        validate_travel_time(&data, &mut ctx);

        assert!(
            !ctx.errors()
                .iter()
                .any(|e| e.kind == ErrorKind::NotImplemented),
            "equal travel_time_hours confluence must not be rejected, got: {:?}",
            ctx.errors()
        );
    }

    #[test]
    fn test_parallel_confluence_heterogeneous_travel_time_no_row6_error() {
        let data = confluence_data(48.0, 96.0, false);

        let mut ctx = ValidationContext::new();
        validate_travel_time(&data, &mut ctx);

        assert!(
            !ctx.errors()
                .iter()
                .any(|e| e.kind == ErrorKind::NotImplemented),
            "a parallel-mode confluence must not be rejected, got: {:?}",
            ctx.errors()
        );
    }

    // ── Row 12: downstream not yet Operating at release -- reject ────────────

    /// A declared arc into a downstream hydro that has not yet reached
    /// `entry_stage_id` at the release (anchor) stage hard-errors, naming the
    /// downstream plant and the offending stage.
    #[test]
    fn test_row12_downstream_not_yet_entered_hard_errors_naming_plant_and_stage() {
        let up = make_hydro_with_travel_time(1, 2, Some(48.0));
        let mut down = make_hydro(2, None);
        down.entry_stage_id = Some(3);
        let stages = make_stages_with_pre_study(4, 720.0, 1, 720.0);
        let mut data = make_data(vec![up, down], vec![], vec![], stages, vec![], vec![]);
        data.initial_conditions.past_defluences = covering_defluences(1, 48.0);

        let mut ctx = ValidationContext::new();
        validate_travel_time(&data, &mut ctx);

        assert!(
            ctx.has_errors(),
            "a release into a not-yet-entered downstream must hard-error"
        );
        let errors = ctx.errors();
        assert!(
            errors
                .iter()
                .any(|e| e.kind == ErrorKind::BusinessRuleViolation
                    && e.message.contains("Hydro 2")
                    && e.message.contains("stage 0")),
            "error must name the downstream plant and the offending stage, got: {errors:?}"
        );
    }

    /// The same arc into a downstream hydro that is Operating for the whole
    /// horizon (no `entry_stage_id`) must never trigger row 12.
    #[test]
    fn test_row12_operating_downstream_no_error() {
        let up = make_hydro_with_travel_time(1, 2, Some(48.0));
        let down = make_hydro(2, None);
        let stages = make_stages_with_pre_study(4, 720.0, 1, 720.0);
        let mut data = make_data(vec![up, down], vec![], vec![], stages, vec![], vec![]);
        data.initial_conditions.past_defluences = covering_defluences(1, 48.0);

        let mut ctx = ValidationContext::new();
        validate_travel_time(&data, &mut ctx);

        assert!(
            !ctx.has_errors(),
            "an always-Operating downstream must never hard-error, got: {:?}",
            ctx.errors()
        );
    }

    // ── Row 11: downstream exit inside the arrival window -- advisory ────────

    /// A declared arc whose downstream exits inside the arrival window drops
    /// the maturing delivery with an advisory; setup proceeds (no error).
    #[test]
    fn test_row11_downstream_exit_inside_window_is_advisory_setup_proceeds() {
        let up = make_hydro_with_travel_time(1, 2, Some(800.0));
        let mut down = make_hydro(2, None);
        down.exit_stage_id = Some(2);
        let stages = make_stages_with_pre_study(4, 720.0, 1, 720.0);
        let mut data = make_data(vec![up, down], vec![], vec![], stages, vec![], vec![]);
        data.initial_conditions.past_defluences = covering_defluences(1, 800.0);

        let mut ctx = ValidationContext::new();
        validate_travel_time(&data, &mut ctx);

        assert!(
            !ctx.has_errors(),
            "downstream exit inside the arrival window must not hard-error, got: {:?}",
            ctx.errors()
        );
        assert!(
            ctx.warnings()
                .iter()
                .any(|w| w.kind == ErrorKind::ModelQuality
                    && w.message.contains("Hydro 2")
                    && w.message.contains("exit_stage_id")),
            "expected an advisory naming the downstream plant and exit_stage_id, got: {:?}",
            ctx.warnings()
        );
    }

    /// The same arc into a downstream with no `exit_stage_id` must never
    /// trigger row 11.
    #[test]
    fn test_row11_no_exit_stage_id_no_advisory() {
        let up = make_hydro_with_travel_time(1, 2, Some(800.0));
        let down = make_hydro(2, None);
        let stages = make_stages_with_pre_study(4, 720.0, 1, 720.0);
        let mut data = make_data(vec![up, down], vec![], vec![], stages, vec![], vec![]);
        data.initial_conditions.past_defluences = covering_defluences(1, 800.0);

        let mut ctx = ValidationContext::new();
        validate_travel_time(&data, &mut ctx);

        assert!(
            !ctx.warnings()
                .iter()
                .any(|w| w.message.contains("exit_stage_id")),
            "no exit_stage_id must never emit a row-11 advisory, got: {:?}",
            ctx.warnings()
        );
    }
}
