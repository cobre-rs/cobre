//! Layer 5a — thermal-domain semantic validation.
//!
//! Thermal generation bounds (`min_generation_mw <= max_generation_mw`) and
//! anticipated-thermal cross-field invariants.

use std::collections::{HashMap, HashSet};

use chrono::{NaiveDate, TimeDelta};
use cobre_core::commissioning::commissioning_active;
use cobre_core::temporal::{
    Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig, StageStateConfig,
};
use cobre_core::{
    AnticipatedCommitmentHistory, AnticipatedConfig, EntityId, PostStudyStages, ResolvedBounds,
    Thermal, VariableRef,
};
use cobre_stochastic::season_cast::{DatedWindow, StageCalendar};

use super::super::{ErrorKind, ValidationContext, schema::ParsedData};
use super::envelope_tolerance;
use super::travel_time::study_stage_durations;
use crate::StageIdResolver;
use crate::resolution::{BoundsEntitySlices, BoundsOverrides, resolve_bounds};

/// Rule 13: `min_generation_mw <= max_generation_mw` for a thermal.
pub(super) fn check_thermal_generation_bounds(data: &ParsedData, ctx: &mut ValidationContext) {
    for thermal in &data.thermals {
        if thermal.min_generation_mw > thermal.max_generation_mw {
            let entity_str = format!("Thermal {}", thermal.id.0);
            ctx.add_error(
                ErrorKind::InvalidValue,
                "system/thermals.json",
                Some(&entity_str),
                format!(
                    "{entity_str}: min_generation_mw ({}) > max_generation_mw ({}); generation bounds are inconsistent",
                    thermal.min_generation_mw, thermal.max_generation_mw
                ),
            );
        }
    }
}

/// Rules 14-15: cross-field invariants for anticipated thermal plants.
///
/// 1. **Per-plant lead horizon** — `LeadStages` rejects `K == 0` (defence in
///    depth; parse-time also rejects it). Both `LeadStages`'s `K > n_stages`
///    and `LeadTime`'s `delta_hours` exceeding the summed study-stage
///    durations (strict `>`, so a delivery landing exactly on the final stage
///    is accepted) reject only a thermal whose lead reaches NO declared
///    post-study stage: a thermal whose extended lead reaches a post-study
///    stage — decided in-study (`carried`) or decided pre-study
///    (`fixed_post_study`, the full-anticipation regime) — legitimately
///    delivers post-horizon, so it is exempt, identically for either lead
///    mode. Whether the lead reaches a post-study stage is resolved once per
///    thermal by `classify_deliveries`'s `carried` and `fixed_post_study`
///    sets over the concatenated study + post-study calendar — computed
///    independently of the solver crate's resolver (`cobre-io` is upstream
///    and cannot depend on it) — and shared by both guards. A commissioning
///    window IS supported and composes with the lookahead; these checks
///    validate the LEAD itself, independent of any window.
/// 2. **Horizon-straddle guard** — see `check_no_straddling_commitment_window`:
///    a `past_anticipated_commitments` window whose `[start_date, end_date)`
///    span crosses the study horizon end (`start_date < horizon_end &&
///    end_date > horizon_end`) is rejected, since it is unrepresentable by
///    either (3)'s study-side bijection or `check_post_study_stages`'s
///    post-study tiling and today silently prices its post-horizon tail at
///    zero. A window ending exactly at, or starting exactly at, the horizon is
///    accepted (purely in-study / purely post-study — two semantically
///    distinct coverages, so a straddling declaration must be split in two).
///    Gates (3)-(5) for the offending plant, since a straddling window's
///    coverage numbers are meaningless.
/// 3. **Past-commitments registry bijection** with
///    `ic.past_anticipated_commitments`: each anticipated thermal has at least
///    one commitment window, each window references an anticipated thermal, and
///    the plant's windows tile its leading `lead_delivery_stage_count` delivery
///    stages exactly (coverage `1.0`, no gap, no overlap, none beyond the
///    horizon) via the shared [`StageCalendar`] resolver — a hard gate, no
///    fallback (the "Pre-study anticipated commitments: calendar-derived
///    coverage" contract).
/// 4. **Committed-value generation bounds** — see `check_committed_value_bounds`.
/// 5. **Seed-vs-window consistency** — see `check_seed_within_window`.
pub(super) fn check_anticipated_thermals(data: &ParsedData, ctx: &mut ValidationContext) {
    // Study stages (id >= 0) are the contiguous suffix of the id-sorted stage
    // list; pre-study stages (negative IDs) are never delivery targets.
    let study_stages: &[Stage] = match data.stages.stages.iter().position(|s| s.id >= 0) {
        Some(idx) => &data.stages.stages[idx..],
        None => &[],
    };
    let study_stage_ids: Vec<i32> = study_stages.iter().map(|s| s.id).collect();
    let n_stages = study_stage_ids.len();
    let study_durations = study_stage_durations(data);
    let horizon_end = study_stages.last().map(|s| s.end_date);

    let extended_axis = build_extended_delivery_axis(data);

    for thermal in &data.thermals {
        let Some(cfg) = thermal.anticipated_config else {
            continue;
        };
        let thermal_id = thermal.id.0;

        let classes = classify_deliveries(
            cfg,
            extended_axis.as_ref(),
            &study_durations,
            n_stages,
            (thermal.entry_stage_id, thermal.exit_stage_id),
        );
        // A plant reaching only fixed_post_study stages (empty carried) is
        // the full-anticipation regime — every in-study delivery decided
        // pre-study — and is fully representable; only beyond_reach (never
        // included here) means the plant truly cannot deliver. Shared by
        // both lead-mode guards below so `LeadStages` and `LeadTime` exempt
        // an equally-reachable plant identically.
        let reaches_post_study =
            !classes.carried.is_empty() || !classes.fixed_post_study.is_empty();

        if let AnticipatedConfig::LeadTime(delta_hours) = cfg {
            let total_horizon_hours: f64 = study_durations.iter().sum();
            if delta_hours > total_horizon_hours && !reaches_post_study {
                let entity_str = format!("thermals[id={thermal_id}].anticipated_config.lead_time");
                ctx.add_error(
                    ErrorKind::BusinessRuleViolation,
                    "system/thermals.json",
                    Some(&entity_str),
                    format!(
                        "Thermal {thermal_id}: lead_time exceeds study horizon \
                         (lead_time={delta_hours}, total_horizon_hours={total_horizon_hours}); \
                         the plant can never deliver within the study horizon"
                    ),
                );
            }
        }

        let Some(k) = cfg.lead_stages() else {
            continue;
        };
        let entity_str = format!("thermals[id={thermal_id}].anticipated_config.lead_stages");

        if k == 0 {
            ctx.add_error(
                ErrorKind::BusinessRuleViolation,
                "system/thermals.json",
                Some(&entity_str),
                format!("Thermal {thermal_id}: anticipated_config.lead_stages must be >= 1, got 0"),
            );
            continue;
        }

        let k_u = k as usize;

        if k_u > n_stages && !reaches_post_study {
            ctx.add_error(
                ErrorKind::BusinessRuleViolation,
                "system/thermals.json",
                Some(&entity_str),
                format!(
                    "Thermal {thermal_id}: lead_stages exceeds study horizon \
                     (lead_stages={k}, n_stages={n_stages}); \
                     the plant can never deliver within the study horizon"
                ),
            );
        }
    }

    let ic = &data.initial_conditions;
    let windows_by_id = group_commitments_by_thermal(&ic.past_anticipated_commitments);

    // The resolver is only consulted for a thermal that carries commitment
    // windows; skip building it (and its ordered-calendar precondition) when
    // there are none to validate.
    let calendar = (!windows_by_id.is_empty()).then(|| StageCalendar::new(study_stages));

    // Thermal-only resolved bounds table, built once under the same guard as
    // `calendar` so `check_committed_value_bounds` reads each covered
    // delivery stage's box from the same source `fill_anticipated_columns`
    // reads (`ResolvedBounds::thermal_block_base`), never a re-derived copy.
    let bounds = (!windows_by_id.is_empty())
        .then(|| build_thermal_delivery_bounds(data, study_stages, n_stages, &study_stage_ids));

    let anticipated_thermal_ids: HashSet<EntityId> = collect_anticipated_thermal_ids(data);

    for (thermal_pos, thermal) in data.thermals.iter().enumerate() {
        let Some(cfg) = thermal.anticipated_config else {
            continue;
        };
        let thermal_id = thermal.id;

        match windows_by_id.get(&thermal_id) {
            None => {
                ctx.add_error(
                    ErrorKind::BusinessRuleViolation,
                    "initial_conditions.json",
                    Some("initial_conditions.past_anticipated_commitments"),
                    format!(
                        "Thermal {}: missing entry in initial_conditions.past_anticipated_commitments; \
                         every anticipated thermal must have at least one commitment window",
                        thermal_id.0
                    ),
                );
            }
            Some(records) => {
                if let Some(horizon_end) = horizon_end
                    && !check_no_straddling_commitment_window(thermal_id, records, horizon_end, ctx)
                {
                    continue;
                }
                let Some(calendar) = calendar.as_ref() else {
                    continue;
                };
                let Some(bounds) = bounds.as_ref() else {
                    continue;
                };
                let k_i = lead_delivery_stage_count(cfg, &study_durations, n_stages);
                if check_commitment_coverage(
                    thermal_id,
                    records,
                    calendar,
                    k_i,
                    &study_stage_ids,
                    ctx,
                ) {
                    let delivery_box = DeliveryBoxContext {
                        bounds,
                        thermal_pos,
                        calendar,
                        study_stage_ids: &study_stage_ids,
                    };
                    check_committed_value_bounds(thermal, thermal_id, records, &delivery_box, ctx);
                    check_seed_within_window(
                        thermal,
                        thermal_id,
                        records,
                        calendar,
                        &study_stage_ids,
                        ctx,
                    );
                }
            }
        }
    }

    let mut reported: HashSet<EntityId> = HashSet::new();
    for history in &ic.past_anticipated_commitments {
        if !anticipated_thermal_ids.contains(&history.thermal_id)
            && reported.insert(history.thermal_id)
        {
            ctx.add_error(
                ErrorKind::BusinessRuleViolation,
                "initial_conditions.json",
                Some(format!(
                    "initial_conditions.past_anticipated_commitments[thermal_id={}]",
                    history.thermal_id.0
                )),
                format!(
                    "Thermal {}: referenced in past_anticipated_commitments \
                     but is not an anticipated thermal (anticipated_config is None or thermal does not exist)",
                    history.thermal_id.0
                ),
            );
        }
    }
}

/// The thermal-only resolved bounds table `check_committed_value_bounds` reads
/// each covered delivery stage's box from, via `stage_resolver.index_map()`
/// (`n_stages`, `k_max`, and `blocks_per_stage` mirror `pipeline.rs`'s own
/// `resolve_bounds` call).
fn build_thermal_delivery_bounds(
    data: &ParsedData,
    study_stages: &[Stage],
    n_stages: usize,
    study_stage_ids: &[i32],
) -> ResolvedBounds {
    let stage_resolver = StageIdResolver::from_study_stage_ids(study_stage_ids);
    let k_max: usize = data
        .thermals
        .iter()
        .filter_map(|t| t.anticipated_config.as_ref())
        .map(|c| usize::try_from(c.lead_stages().unwrap_or(0)).unwrap_or(usize::MAX))
        .max()
        .unwrap_or(0);
    let blocks_per_stage: Vec<usize> = study_stages.iter().map(|s| s.blocks.len()).collect();
    resolve_bounds(
        &BoundsEntitySlices {
            hydros: &[],
            thermals: &data.thermals,
            lines: &[],
            pumping_stations: &[],
            contracts: &[],
        },
        n_stages,
        k_max,
        stage_resolver.index_map(),
        &BoundsOverrides {
            hydro: &[],
            thermal: &data.thermal_bounds,
            line: &[],
            pumping: &[],
            contract: &[],
        },
        &blocks_per_stage,
    )
}

/// Rule 28. Advisory (`ModelQuality`): a `lead_stages`-configured thermal whose active
/// window — decision stage `t` through delivery `t + lead_stages`, `t` ranging
/// over the plant's commissioning window and the delivery side clamped to the
/// study horizon — spans a pair of adjacent study stages with differing
/// durations. A fixed stage-count lead delivers a different physical lead on
/// each side of such a cadence change; `anticipated_config.lead_time` anchors
/// the lead to physical hours instead and is immune to this. Never a hard error.
pub(super) fn check_anticipated_cadence_transition(data: &ParsedData, ctx: &mut ValidationContext) {
    let study_durations = study_stage_durations(data);
    let n_stages = study_durations.len();
    if n_stages < 2 {
        return;
    }

    for thermal in &data.thermals {
        let Some(cfg) = thermal.anticipated_config else {
            continue;
        };
        let Some(k) = cfg.lead_stages() else {
            continue;
        };
        let thermal_id = thermal.id.0;
        let k_u = usize::try_from(k).unwrap_or(usize::MAX);

        let decision_start = usize::try_from(thermal.entry_stage_id.unwrap_or(0).max(0))
            .unwrap_or(0)
            .min(n_stages - 1);
        let decision_end_id = thermal.exit_stage_id.map_or_else(
            || i32::try_from(n_stages - 1).unwrap_or(i32::MAX),
            |exit| exit - 1,
        );
        if decision_end_id < 0 {
            continue;
        }
        let decision_end = usize::try_from(decision_end_id)
            .unwrap_or(0)
            .min(n_stages - 1);
        if decision_start > decision_end {
            continue;
        }
        let window_end = decision_end.saturating_add(k_u).min(n_stages - 1);

        let entity_str = format!("thermals[id={thermal_id}].anticipated_config.lead_stages");
        for i in decision_start..window_end {
            let (prev, next) = (study_durations[i], study_durations[i + 1]);
            if (prev - next).abs() > 1e-9 {
                ctx.add_warning(
                    ErrorKind::ModelQuality,
                    "system/thermals.json",
                    Some(&entity_str),
                    format!(
                        "Thermal {thermal_id}: anticipated_config.lead_stages={k} active window \
                         spans a stage-cadence transition between stage {i} ({prev}h) and stage \
                         {} ({next}h); a fixed stage-count lead delivers a different physical \
                         lead on each side of the transition. Consider anticipated_config.lead_time, \
                         which anchors the lead to physical hours instead of a stage count.",
                        i + 1
                    ),
                );
                break;
            }
        }
    }
}

/// Group `histories` by `thermal_id`, preserving each plant's windows in
/// declared order. Shared by [`check_anticipated_thermals`] (study-side
/// coverage) and [`check_post_study_stages`] (post-study-side coverage) so
/// the two never drift on how a plant's windows are gathered.
fn group_commitments_by_thermal(
    histories: &[AnticipatedCommitmentHistory],
) -> HashMap<EntityId, Vec<&AnticipatedCommitmentHistory>> {
    let mut windows_by_id: HashMap<EntityId, Vec<&AnticipatedCommitmentHistory>> = HashMap::new();
    for history in histories {
        windows_by_id
            .entry(history.thermal_id)
            .or_default()
            .push(history);
    }
    windows_by_id
}

/// The set of thermal ids with `anticipated_config: Some(_)`. Shared by
/// [`check_anticipated_thermals`], [`check_anticipated_decision_target_is_anticipated`],
/// and [`warn_thermal_generation_on_anticipated_thermal`] so the three never
/// drift on which thermals count as anticipated.
fn collect_anticipated_thermal_ids(data: &ParsedData) -> HashSet<EntityId> {
    data.thermals
        .iter()
        .filter(|t| t.anticipated_config.is_some())
        .map(|t| t.id)
        .collect()
}

/// Calendar-derived count of leading pre-study-committed delivery stages (the
/// study-only prefix `past_anticipated_commitments` must tile): `LeadStages(l)`
/// clamps `l` to `n_stages`; `LeadTime(delta)` counts the leading study stages
/// whose stage-end cumulative hours are `<= delta` (tie-inclusive). Both
/// deciders are monotonic in stage-end cumulative hours, so the pre-study run is
/// always the leading prefix `0..k`. The delivery-anchored reach INTO the
/// post-study calendar is resolved by `classify_deliveries`/`extended_deciders`
/// over the concatenated study + post-study calendar; this count stays
/// study-only. Computed independently of the solver crate's point-commitment
/// resolver (cobre-io is upstream and cannot depend on it), mirroring
/// `check_defluence_coverage`'s own calendar walk
/// (`validation/semantic/travel_time.rs`). Do not shortcut `LeadTime` to a stage
/// count from a window length — on a non-uniform calendar the cumulative-hours
/// walk and a bare length diverge.
fn lead_delivery_stage_count(
    mode: AnticipatedConfig,
    study_durations: &[f64],
    n_stages: usize,
) -> usize {
    match mode {
        AnticipatedConfig::LeadStages(lead_stages) => (lead_stages as usize).min(n_stages),
        AnticipatedConfig::LeadTime(delta_hours) => {
            let mut cumulative_hours = 0.0_f64;
            let mut count = 0usize;
            for &duration in study_durations.iter().take(n_stages) {
                cumulative_hours += duration;
                if cumulative_hours > delta_hours {
                    break;
                }
                count += 1;
            }
            count
        }
    }
}

/// The concatenated study + post-study delivery axis: extended per-stage
/// durations (hours) and the continued stage-id sequence, both length
/// `n_stages + n_post`. `None` when the study declares no `post_study_stages`,
/// so the delivery axis is the study axis and no lead can reach post-study.
///
/// Durations use each post-study stage's raw `duration_hours` and the ids
/// continue the study sequence (`max_study_id + 1 ..`), matching the solver's
/// own extended-axis concatenation and delivery-stage id sequence so the reach
/// resolved here and the ring the solver builds agree on which post-study stage
/// each delivery lands in.
struct ExtendedDeliveryAxis {
    /// Per-delivery-stage total hours, study durations then post-study durations.
    durations: Vec<f64>,
    /// Per-delivery-stage id, study ids then the continued post-study sequence.
    stage_ids: Vec<i32>,
    /// In-study (decision) stage count — the `[0, n_stages)` decision prefix.
    n_stages: usize,
}

/// Build the concatenated delivery axis, or `None` when `post_study_stages` is
/// absent. See `ExtendedDeliveryAxis` for the concatenation contract.
fn build_extended_delivery_axis(data: &ParsedData) -> Option<ExtendedDeliveryAxis> {
    let post_study = data.post_study_stages.as_ref()?;
    let study_durations = study_stage_durations(data);
    let n_stages = study_durations.len();
    let study_ids: Vec<i32> = data
        .stages
        .stages
        .iter()
        .filter(|s| s.id >= 0)
        .map(|s| s.id)
        .collect();
    let max_study_id = study_ids.last().copied().unwrap_or(-1);
    let durations: Vec<f64> = study_durations
        .iter()
        .copied()
        .chain(post_study.stages.iter().map(|s| s.duration_hours))
        .collect();
    let stage_ids: Vec<i32> = study_ids
        .into_iter()
        .chain(
            (0..post_study.stages.len())
                .map(|j| max_study_id + 1 + i32::try_from(j).unwrap_or(i32::MAX)),
        )
        .collect();
    Some(ExtendedDeliveryAxis {
        durations,
        stage_ids,
        n_stages,
    })
}

/// Delivery-anchored decider `c(m)` for each delivery stage `m` over the
/// concatenated calendar, mirroring the solver's `resolve_decider_*`:
/// `LeadStages(l)` shifts by a bare stage count (`c(m) = m − l`); `LeadTime(δ)`
/// anchors at the delivery stage's cumulative end (`c(m)` = the stage containing
/// `end_m − δ`, boundary ties resolving to the earlier stage). `None` is a
/// pre-study (initial-conditions) decider. Computed independently of the solver
/// crate's `resolve_point` (cobre-io is upstream and cannot depend on it). Do
/// not shortcut `LeadTime` to a bare window length — on a non-uniform calendar
/// the cumulative-hours walk and a bare length diverge.
fn extended_deciders(mode: AnticipatedConfig, durations: &[f64]) -> Vec<Option<usize>> {
    let n_delivery = durations.len();
    match mode {
        AnticipatedConfig::LeadStages(lead_stages) => {
            let lead = lead_stages as usize;
            (0..n_delivery).map(|m| m.checked_sub(lead)).collect()
        }
        AnticipatedConfig::LeadTime(delta_hours) => {
            let mut boundaries = Vec::with_capacity(n_delivery + 1);
            let mut cumulative = 0.0_f64;
            boundaries.push(cumulative);
            for &duration in durations {
                cumulative += duration;
                boundaries.push(cumulative);
            }
            (0..n_delivery)
                .map(|m| {
                    let target = boundaries[m + 1] - delta_hours;
                    boundaries
                        .partition_point(|&boundary| boundary < target)
                        .checked_sub(1)
                })
                .collect()
        }
    }
}

/// Post-study-relative classification of an anticipated plant's extended
/// delivery axis. Every post-study index `j` lands in exactly one of
/// `fixed_post_study`, `carried`, `beyond_reach`, and
/// `commissioning_inactive`; `leading_in_study` counts a disjoint, in-study
/// range.
struct DeliveryClasses {
    /// In-study deliveries decided pre-study — always the leading prefix
    /// `[0, k)`, since both deciders are monotonic in stage-end cumulative
    /// hours. Not yet read by a validator; reserved for the rule that will
    /// require a fixed commitment for this prefix.
    // RATIONALE: reserved seam (see field doc) plus a live debug-only consumer —
    // `classify_deliveries`'s `debug_assert_eq!` cross-checks it against the `None`
    // run of `extended_deciders`, and that read compiles out in release, so the
    // field reads as dead there. Removing it drops both the reserved seam and the
    // consistency check.
    #[allow(dead_code)]
    leading_in_study: usize,
    /// Post-study deliveries decided at a pre-study stage — each requires a
    /// tiled commitment window (V2).
    fixed_post_study: Vec<usize>,
    /// Post-study deliveries reached from an in-study, commissioning-active
    /// decision (a carrier — each needs a
    /// `PostStudyThermalBound(thermal_id, j)`).
    carried: Vec<usize>,
    /// Post-study deliveries decided post-study: past the plant's decision
    /// reach, unrepresentable by any in-study decision variable.
    beyond_reach: Vec<usize>,
    /// Post-study stages the plant's commissioning window excludes — where
    /// today's walk silently skips the index. Read by
    /// [`check_fixed_commitment_within_window`] to reject a non-zero fixed
    /// value there.
    commissioning_inactive: Vec<usize>,
}

/// Classify `window`'s deliveries over `axis` (`None` when the study declares
/// no post-study stages): `leading_in_study` resolves from
/// `lead_delivery_stage_count` regardless of `axis`; the four post-study
/// vectors resolve from `mode`'s extended decider and are empty when `axis`
/// is `None`. The commissioning gate keys on the DELIVERY stage's continued
/// id (`axis.stage_ids[m]`), uniformly at every delivery stage — a plant
/// whose `exit_stage_id` lies inside the study is inactive for every later
/// delivery, study or post-study — mirroring the solver's own
/// delivery-anchored gate. Do not shortcut `LeadTime` to a stage count from a
/// window length — on a non-uniform calendar the cumulative-hours walk and a
/// bare length diverge (the classifier reuses `lead_delivery_stage_count`'s
/// and `extended_deciders`' arithmetic, so the same trap applies here).
fn classify_deliveries(
    mode: AnticipatedConfig,
    axis: Option<&ExtendedDeliveryAxis>,
    study_durations: &[f64],
    n_stages: usize,
    window: (Option<i32>, Option<i32>),
) -> DeliveryClasses {
    let leading_in_study = lead_delivery_stage_count(mode, study_durations, n_stages);

    let Some(axis) = axis else {
        return DeliveryClasses {
            leading_in_study,
            fixed_post_study: Vec::new(),
            carried: Vec::new(),
            beyond_reach: Vec::new(),
            commissioning_inactive: Vec::new(),
        };
    };

    let (entry, exit) = window;
    let deciders = extended_deciders(mode, &axis.durations);
    // leading_in_study (a cumulative-hours count) and the None run over
    // deciders[..axis.n_stages] (a boundary partition_point) are two
    // independent walks that must agree; a drift here leaves the post-study
    // classes below keying on the wrong study/post-study boundary.
    debug_assert_eq!(
        leading_in_study,
        deciders[..axis.n_stages]
            .iter()
            .filter(|decider| decider.is_none())
            .count(),
        "leading_in_study must match the None run of extended_deciders over the study prefix"
    );

    let mut fixed_post_study = Vec::new();
    let mut carried = Vec::new();
    let mut beyond_reach = Vec::new();
    let mut commissioning_inactive = Vec::new();
    // `j` (the enumerate index over the post-study suffix) IS the post-study stage
    // index; `stage_ids[n_stages..]` and `deciders[n_stages..]` are the same length.
    for (j, (&stage_id, decider)) in axis.stage_ids[axis.n_stages..]
        .iter()
        .zip(&deciders[axis.n_stages..])
        .enumerate()
    {
        if !commissioning_active(entry, exit, stage_id) {
            commissioning_inactive.push(j);
            continue;
        }
        match decider {
            Some(c) if *c < axis.n_stages => carried.push(j),
            None => fixed_post_study.push(j),
            Some(_) => beyond_reach.push(j),
        }
    }

    debug_assert_eq!(
        fixed_post_study.len() + carried.len() + beyond_reach.len() + commissioning_inactive.len(),
        axis.stage_ids.len() - axis.n_stages,
        "the four post-study classes must partition every post-study index exactly once"
    );

    DeliveryClasses {
        leading_in_study,
        fixed_post_study,
        carried,
        beyond_reach,
        commissioning_inactive,
    }
}

/// Horizon-straddle guard (item 2 of [`check_anticipated_thermals`]): rejects
/// every window in `records` whose `[start_date, end_date)` span crosses
/// `horizon_end` (`start_date < horizon_end && end_date > horizon_end`) — a
/// single window cannot deliver both in-study and post-study coverage, and
/// left unchecked the post-study tail is silently priced at zero rather than
/// rejected. A window ending exactly at `horizon_end` (purely in-study) or
/// starting exactly at it (purely post-study) is accepted; the two are
/// semantically distinct coverages and must be declared as separate windows
/// split at the horizon. Returns whether every window passed, gating the
/// bijection/bounds/seed checks the caller runs next for this plant.
fn check_no_straddling_commitment_window(
    thermal_id: EntityId,
    records: &[&AnticipatedCommitmentHistory],
    horizon_end: NaiveDate,
    ctx: &mut ValidationContext,
) -> bool {
    let mut valid = true;
    let entity_str = format!("thermals[id={}].anticipated_config", thermal_id.0);
    for record in records {
        if record.start_date < horizon_end && record.end_date > horizon_end {
            ctx.add_error(
                ErrorKind::BusinessRuleViolation,
                "initial_conditions.json",
                Some(&entity_str),
                format!(
                    "Thermal {}: past_anticipated_commitments window [{}, {}) straddles the \
                     study horizon end ({horizon_end}); a single window may not cover both \
                     in-study and post-study delivery. Declare two windows split at the \
                     horizon instead: one ending at {horizon_end} for the in-study coverage, \
                     one starting at {horizon_end} for the post-study coverage.",
                    thermal_id.0, record.start_date, record.end_date
                ),
            );
            valid = false;
        }
    }
    valid
}

/// A thermal's commitment windows must tile its leading `k_i` delivery stages
/// exactly: every leading stage covered at fraction `1.0` (via
/// [`StageCalendar::covers_exactly`]), and no window reaching any stage at or
/// beyond `k_i`. Emits a named `BusinessRuleViolation` for an uncovered leading
/// stage (gap) or a stage covered beyond the horizon (over-coverage); overlap
/// is rejected earlier by the shared windowed-record validator. Returns whether
/// coverage is exact, gating the per-window bounds/commissioning checks.
///
/// This is the study-side half of the coverage rule, checked against the study
/// [`StageCalendar`] only; the post-study half lives in
/// [`check_post_study_stages`]. Do not widen the range checked here onto a
/// post-horizon gap — this function has no post-study calendar to check it
/// against.
#[allow(clippy::float_cmp)] // whole-day-hours coverage keeps the per-stage tiling fraction bit-exact (mirrors covers_exactly)
fn check_commitment_coverage(
    thermal_id: EntityId,
    records: &[&AnticipatedCommitmentHistory],
    calendar: &StageCalendar,
    k_i: usize,
    study_stage_ids: &[i32],
    ctx: &mut ValidationContext,
) -> bool {
    let windows: Vec<DatedWindow> = records
        .iter()
        .map(|r| DatedWindow {
            start_date: r.start_date,
            end_date: r.end_date,
        })
        .collect();

    let mut per_stage = vec![0.0_f64; study_stage_ids.len()];
    for window in &windows {
        for (total, fraction) in per_stage.iter_mut().zip(calendar.coverage(window)) {
            *total += fraction;
        }
    }

    let entity_str = format!("thermals[id={}].anticipated_config", thermal_id.0);
    let mut valid = true;

    if !calendar.covers_exactly(&windows, k_i) {
        let uncovered: Vec<i32> = (0..k_i)
            .filter(|&i| per_stage[i] != 1.0)
            .map(|i| study_stage_ids[i])
            .collect();
        ctx.add_error(
            ErrorKind::BusinessRuleViolation,
            "initial_conditions.json",
            Some(&entity_str),
            format!(
                "Thermal {}: past_anticipated_commitments do not tile the leading {k_i} \
                 delivery stage(s) at coverage 1.0; study stage id(s) {uncovered:?} are not \
                 covered exactly once. Write a commitment window (a committed 0 MW is explicit) \
                 for every leading delivery stage.",
                thermal_id.0
            ),
        );
        valid = false;
    }

    let over_covered: Vec<i32> = (k_i..study_stage_ids.len())
        .filter(|&i| per_stage[i] != 0.0)
        .map(|i| study_stage_ids[i])
        .collect();
    if !over_covered.is_empty() {
        ctx.add_error(
            ErrorKind::BusinessRuleViolation,
            "initial_conditions.json",
            Some(&entity_str),
            format!(
                "Thermal {}: past_anticipated_commitments cover study stage id(s) {over_covered:?} \
                 beyond the leading {k_i} calendar-derived delivery stage(s); a commitment window \
                 may not cover a study stage the study itself decides. Shorten the plant's \
                 anticipated_config lead so its window stays within the leading {k_i} delivery \
                 stage(s).",
                thermal_id.0
            ),
        );
        valid = false;
    }

    valid
}

/// Read-only inputs shared by every record of one thermal's
/// [`check_committed_value_bounds`] call: the thermal-only resolved bounds
/// table, this thermal's position into it, and the calendar/stage-id axis
/// used to resolve each window's covered study stages.
struct DeliveryBoxContext<'a, 'b> {
    bounds: &'a ResolvedBounds,
    thermal_pos: usize,
    calendar: &'a StageCalendar<'b>,
    study_stage_ids: &'a [i32],
}

/// A window's `value_mw` must be deliverable at every study stage it covers,
/// within a relative-with-floor tolerance at each boundary (a value set
/// exactly at a bound may drift a hair outside it in whatever pipeline
/// generated it). For a window with nonzero in-study coverage, each covered,
/// commissioning-active study stage is checked against that stage's resolved
/// generation box (`ResolvedBounds::thermal_block_base`) — the same source
/// `fill_anticipated_columns` reads for the decision column, so a value
/// accepted here is a value the runtime admissible box will not have to
/// clamp as a genuine defect; a commissioning-dormant covered stage is
/// skipped here, since a nonzero value maturing there is
/// `check_seed_within_window`'s report, never double-reported. A window with
/// zero in-study coverage is purely post-horizon (a straddling window is
/// already rejected upstream by `check_no_straddling_commitment_window`);
/// there is no resolved box to check it against, so it keeps the static
/// `[min_generation_mw, max_generation_mw]` comparison — post-horizon there
/// is no LP to reject an out-of-tolerance value, so it instead reaches the
/// terminal-boundary valuation and the reported outputs as generation the
/// plant cannot produce. `records` carries every window of the plant,
/// in-study and post-horizon alike; this check is gated on the study-side
/// coverage rule (`check_commitment_coverage`) passing, never on the
/// post-study coverage rules. `value_mw` finiteness is the parse layer's
/// contract (`initial_conditions.rs`'s call into
/// `crate::windowed_history::validate_windowed_records`), not re-checked
/// here.
#[allow(clippy::float_cmp)] // day-aligned coverage is exactly 0 or nonzero per stage; the in-study/post-horizon split is a bit-exact test
fn check_committed_value_bounds(
    thermal: &Thermal,
    thermal_id: EntityId,
    records: &[&AnticipatedCommitmentHistory],
    delivery_box: &DeliveryBoxContext<'_, '_>,
    ctx: &mut ValidationContext,
) {
    let min_mw = thermal.min_generation_mw;
    let max_mw = thermal.max_generation_mw;
    let min_tolerance = envelope_tolerance(min_mw);
    let max_tolerance = envelope_tolerance(max_mw);
    let entry = thermal.entry_stage_id;
    let exit = thermal.exit_stage_id;
    let entity_str = format!("thermals[id={}].anticipated_config", thermal_id.0);

    for record in records {
        let v = record.value_mw;
        let window = DatedWindow {
            start_date: record.start_date,
            end_date: record.end_date,
        };
        let coverage = delivery_box.calendar.coverage(&window);

        if coverage.iter().all(|&fraction| fraction == 0.0) {
            if v < min_mw - min_tolerance || v > max_mw + max_tolerance {
                ctx.add_error(
                    ErrorKind::BusinessRuleViolation,
                    "initial_conditions.json",
                    Some(&entity_str),
                    format!(
                        "Thermal {}: past_anticipated_commitments window [{}, {}) value_mw = {v} \
                         is outside the plant's generation bounds [{min_mw}, {max_mw}]; \
                         the LP delivery equality on the covered stage(s) cannot be \
                         satisfied and the LP will be infeasible",
                        thermal_id.0, record.start_date, record.end_date
                    ),
                );
            }
            continue;
        }

        for (i, &fraction) in coverage.iter().enumerate() {
            if fraction == 0.0 {
                continue;
            }
            let stage_id = delivery_box.study_stage_ids[i];
            if !commissioning_active(entry, exit, stage_id) {
                continue;
            }
            let bx = delivery_box
                .bounds
                .thermal_block_base(delivery_box.thermal_pos, i);
            let lb = bx.min_generation_mw;
            let ub = bx.max_generation_mw;
            if lb > ub {
                ctx.add_error(
                    ErrorKind::BusinessRuleViolation,
                    "initial_conditions.json",
                    Some(&entity_str),
                    format!(
                        "Thermal {}: past_anticipated_commitments window [{}, {}) — \
                         delivery stage id {stage_id} has an empty resolved generation box \
                         [{lb}, {ub}] (lower > upper); the committed value cannot be delivered.",
                        thermal_id.0, record.start_date, record.end_date
                    ),
                );
                continue;
            }
            let lb_tolerance = envelope_tolerance(lb);
            let ub_tolerance = envelope_tolerance(ub);
            if v < lb - lb_tolerance || v > ub + ub_tolerance {
                ctx.add_error(
                    ErrorKind::BusinessRuleViolation,
                    "initial_conditions.json",
                    Some(&entity_str),
                    format!(
                        "Thermal {}: past_anticipated_commitments window [{}, {}) \
                         value_mw = {v} is outside the resolved generation box [{lb}, {ub}] at \
                         delivery stage id {stage_id}; the LP delivery equality on that stage \
                         cannot be satisfied and the LP will be infeasible.",
                        thermal_id.0, record.start_date, record.end_date
                    ),
                );
            }
        }
    }
}

/// For a windowed anticipated thermal, reject any nonzero `value_mw` maturing at
/// a covered study stage outside the operation window: the matured generation
/// column is pinned to `[0, 0]` there, so the always-active fishing equality
/// reads `0 == seed` — an infeasible LP. A zero rate is consistent at any stage
/// and allowed.
///
/// The window predicate is the LP builder's own
/// `cobre_core::commissioning::commissioning_active`, so validation and the
/// builder cannot drift on what an infeasible seed is. Each window's covered
/// study stages are resolved through the shared [`StageCalendar`], so a window
/// straddling the commissioning boundary is checked stage-by-stage.
#[allow(clippy::float_cmp)] // day-aligned coverage is exactly 0 or nonzero per stage; the zero-rate skip is a bit-exact test
fn check_seed_within_window(
    thermal: &Thermal,
    thermal_id: EntityId,
    records: &[&AnticipatedCommitmentHistory],
    calendar: &StageCalendar,
    study_stage_ids: &[i32],
    ctx: &mut ValidationContext,
) {
    let entry = thermal.entry_stage_id;
    let exit = thermal.exit_stage_id;
    if entry.is_none() && exit.is_none() {
        return;
    }
    let entity_str = format!("thermals[id={}].anticipated_config", thermal_id.0);
    for record in records {
        let v = record.value_mw;
        if v == 0.0 {
            continue;
        }
        let window = DatedWindow {
            start_date: record.start_date,
            end_date: record.end_date,
        };
        for (i, fraction) in calendar.coverage(&window).into_iter().enumerate() {
            if fraction == 0.0 {
                continue;
            }
            let stage_id = study_stage_ids[i];
            if !commissioning_active(entry, exit, stage_id) {
                ctx.add_error(
                    ErrorKind::BusinessRuleViolation,
                    "initial_conditions.json",
                    Some(&entity_str),
                    format!(
                        "Thermal {}: past_anticipated_commitments window [{}, {}) value_mw = {v} \
                         matures at study stage id {stage_id}, which is outside the \
                         plant's commissioning window [entry={entry:?}, exit={exit:?}); \
                         the matured generation column is pinned to [0, 0] there, so the \
                         fishing equality reads 0 == {v} and the LP is infeasible. \
                         Commit a zero rate at this stage or widen the window.",
                        thermal_id.0, record.start_date, record.end_date
                    ),
                );
            }
        }
    }
}

/// The post-study counterpart of [`check_seed_within_window`]: a non-zero
/// fixed value maturing at a `commissioning_inactive` post-study stage is a
/// modelling contradiction — the plant is not in service for that stage, so
/// the value can never be delivered. Unlike the in-study rule, there is no LP
/// column here to make infeasible; left unchecked, the value would instead be
/// folded into the terminal-boundary valuation and reported as a delivery
/// from a plant that is not in service. A zero-valued window over an
/// inactive stage stays legitimate (a blanket zero across the plant's
/// post-study span).
///
/// `commissioning_inactive` is read from [`DeliveryClasses`], never
/// re-derived: it is already keyed on the delivery stage's continued id, the
/// same predicate the solver's own commissioning gate uses. Coverage is
/// computed per record, not read from the accumulated per-stage vector V2/V3
/// share — naming the offending window's dates needs to know which window
/// covered the stage, which the accumulated vector cannot say.
#[allow(clippy::float_cmp)] // whole-day-hours coverage keeps the per-stage fraction bit-exact (mirrors check_seed_within_window)
fn check_fixed_commitment_within_window(
    thermal: &Thermal,
    thermal_id: EntityId,
    records: &[&AnticipatedCommitmentHistory],
    post_calendar: &StageCalendar,
    commissioning_inactive: &[usize],
    ctx: &mut ValidationContext,
) {
    let entry = thermal.entry_stage_id;
    let exit = thermal.exit_stage_id;
    if entry.is_none() && exit.is_none() {
        return;
    }
    let entity_str = format!("thermals[id={}].anticipated_config", thermal_id.0);
    for record in records {
        let v = record.value_mw;
        if v == 0.0 {
            continue;
        }
        let window = DatedWindow {
            start_date: record.start_date,
            end_date: record.end_date,
        };
        let coverage = post_calendar.coverage(&window);
        for &j in commissioning_inactive {
            if coverage[j] == 0.0 {
                continue;
            }
            ctx.add_error(
                ErrorKind::BusinessRuleViolation,
                "post_study_stages.json",
                Some(&entity_str),
                format!(
                    "Thermal {}: past_anticipated_commitments window [{}, {}) value_mw = {v} \
                     covers post-study stage index {j}, which is outside the plant's \
                     commissioning window [entry={entry:?}, exit={exit:?}); the plant is not \
                     in service for this stage and the fixed commitment cannot be delivered. \
                     Declare a zero commitment at this stage, or widen the commissioning window.",
                    thermal_id.0, record.start_date, record.end_date
                ),
            );
        }
    }
}

/// Rule 17. Layer 5a — rejects use of `anticipated_decision(N)` in a generic constraint
/// when thermal `N` does not have `anticipated_config: Some(_)`.
///
/// `anticipated_decision` is an LP column that only exists for plants committed
/// in advance. Referencing a non-anticipated thermal via this variant is always
/// a model error: the column does not appear in the LP and the constraint would
/// silently become an equality 0 = bound, which is either trivially satisfied
/// or immediately infeasible.
pub(super) fn check_anticipated_decision_target_is_anticipated(
    data: &ParsedData,
    ctx: &mut ValidationContext,
) {
    let anticipated_ids: HashSet<EntityId> = collect_anticipated_thermal_ids(data);

    for constraint in &data.generic_constraints {
        for term in &constraint.expression.terms {
            if let VariableRef::AnticipatedDecision { thermal_id } = term.variable
                && !anticipated_ids.contains(&thermal_id)
            {
                let entity_str = format!("constraint[id={}]", constraint.id.0);
                ctx.add_error(
                    ErrorKind::BusinessRuleViolation,
                    "constraints/generic_constraints.json",
                    Some(&entity_str),
                    format!(
                        "Constraint \"{}\": anticipated_decision({}) references Thermal {} \
                             which is not an anticipated thermal (anticipated_config is None). \
                             The anticipated_decision column only exists for plants with \
                             anticipated_config set.",
                        constraint.name, thermal_id.0, thermal_id.0,
                    ),
                );
            }
        }
    }
}

/// Rule 18. Layer 5a — warns when `thermal_generation(N)` is used in a generic
/// constraint and thermal `N` is anticipated.
///
/// `thermal_generation` for an anticipated thermal references the per-block
/// generation at the *delivery* stage, not the *commitment* made at the current
/// stage. This is valid but surprising; to constrain the commitment use
/// `anticipated_decision(N)` instead. Emits a `SemanticAmbiguity` warning.
pub(super) fn warn_thermal_generation_on_anticipated_thermal(
    data: &ParsedData,
    ctx: &mut ValidationContext,
) {
    let anticipated_ids: HashSet<EntityId> = collect_anticipated_thermal_ids(data);

    if anticipated_ids.is_empty() {
        return;
    }

    for constraint in &data.generic_constraints {
        for term in &constraint.expression.terms {
            if let VariableRef::ThermalGeneration { thermal_id, .. } = term.variable
                && anticipated_ids.contains(&thermal_id)
            {
                let entity_str = format!("constraint[id={}]", constraint.id.0);
                ctx.add_warning(
                    ErrorKind::SemanticAmbiguity,
                    "constraints/generic_constraints.json",
                    Some(&entity_str),
                    format!(
                        "Constraint \"{}\": thermal_generation({id}) references an \
                             anticipated thermal. thermal_generation refers to the \
                             per-block generation at the delivery stage, not the \
                             forward commitment. If you intend to constrain the \
                             commitment itself, use anticipated_decision({id}) instead.",
                        constraint.name,
                        id = thermal_id.0,
                    ),
                );
            }
        }
    }
}

/// Rule 47. Layer 5a — validates the standalone `post_study_stages.json` boundary input,
/// the sole post-horizon surface, against the study calendar.
///
/// Rejects unless:
/// - (a) the post-study stages are date-contiguous and the first `start_date`
///   equals the study horizon end (`study_stages.last().end_date`);
/// - **Rule 1 (missing bound cell)** — every post-study stage index `j` an
///   anticipated thermal's extended lead reaches (from an in-study,
///   commissioning-active decision, resolved by `classify_deliveries`) has a
///   `PostStudyThermalBound(thermal_id, j)`. A missing cell is the sole surface's
///   analogue of the retired lane subsystem's silent `[0, 0]` degradation
///   (finiteness / `min_mw <= max_mw` remain the reader's parse-layer contract).
/// - **V2 (fixed post-horizon tiling)** — every post-study stage index in the
///   plant's `fixed_post_study` class (deliveries decided at a pre-study
///   stage) is tiled by `initial_conditions.past_anticipated_commitments` at
///   coverage `1.0` over the post-study calendar, an explicit `0 MW` window
///   included — the same commitment-window convention
///   [`check_anticipated_thermals`] already applies to the study side. A
///   `commissioning_inactive` post-study stage is never a member of
///   `fixed_post_study`, so V2 never demands a window there; do not read V2 as
///   requiring tiling over every post-study stage — only class 4 is covered.
/// - **V3 (no window on an unreachable stage)** — no commitment window covers a
///   post-study stage in the plant's `carried` class (the study itself decides
///   that delivery) or its `beyond_reach` class (past the plant's decision
///   reach, unrepresentable). The two causes are reported as distinct errors —
///   their remedies are opposite (lengthen vs. shorten the lead). A
///   `commissioning_inactive` post-study stage is exempt from V3: a declared
///   window there is inert, not contradictory (a later rule rejects a
///   *nonzero* value there instead). Together, V2 and V3 complete the
///   coverage rule: a plant's covered post-study stages must be *exactly* its
///   `fixed_post_study` class — V2 supplies "at least", V3 supplies "at most".
/// - **V5 (fixed value outside the commissioning window)** — a non-zero fixed
///   value covering a `commissioning_inactive` post-study stage is rejected: the
///   plant is not in service for that stage, so the value can never be
///   delivered. Mirrors the in-study seed rule
///   ([`check_seed_within_window`]) without its infeasibility claim — there is
///   no LP column post-horizon. An explicit zero-valued window over an
///   inactive stage stays legitimate.
///
/// Each failure is a `BusinessRuleViolation` naming the offending plant (and the
/// post-study stage(s), for Rule 1 / V2 / V3 / V5 / for (a)). No rule short-circuits
/// another; a study without `post_study_stages.json` is validated in
/// [`check_anticipated_thermals`] (a `lead > horizon` plant with no post-study
/// stages is a hard reject there). A `past_anticipated_commitments` window
/// straddling the study horizon is also rejected there, before this
/// function's coverage rules run, so V2/V3 may assume no window straddles the
/// horizon end.
pub(super) fn check_post_study_stages(data: &ParsedData, ctx: &mut ValidationContext) {
    let Some(post_study) = data.post_study_stages.as_ref() else {
        return;
    };

    let study_end = data
        .stages
        .stages
        .iter()
        .rfind(|s| s.id >= 0)
        .map(|s| s.end_date);

    let (calendar_stages, well_formed) = build_post_study_calendar(post_study, study_end, ctx);
    if !well_formed || calendar_stages.is_empty() {
        return;
    }

    let Some(axis) = build_extended_delivery_axis(data) else {
        return;
    };
    let study_durations = study_stage_durations(data);

    let bound_cells: HashSet<(EntityId, usize)> = post_study
        .thermal_bounds
        .iter()
        .map(|b| (b.thermal_id, b.post_study_stage_index))
        .collect();

    let windows_by_id =
        group_commitments_by_thermal(&data.initial_conditions.past_anticipated_commitments);
    let post_calendar = StageCalendar::new(&calendar_stages);

    for thermal in &data.thermals {
        let Some(cfg) = thermal.anticipated_config else {
            continue;
        };
        let classes = classify_deliveries(
            cfg,
            Some(&axis),
            &study_durations,
            axis.n_stages,
            (thermal.entry_stage_id, thermal.exit_stage_id),
        );
        let entity_str = format!("thermals[id={}].anticipated_config", thermal.id.0);

        for &j in &classes.carried {
            if !bound_cells.contains(&(thermal.id, j)) {
                ctx.add_error(
                    ErrorKind::BusinessRuleViolation,
                    "post_study_stages.json",
                    Some(&entity_str),
                    format!(
                        "Thermal {}: anticipated lead reaches post-study stage index {j}, but \
                         post_study_stages.json has no thermal_bounds entry for (thermal_id {}, \
                         post_study_stage_index {j}).",
                        thermal.id.0, thermal.id.0
                    ),
                );
            }
        }

        let windows = windows_by_id
            .get(&thermal.id)
            .map(Vec::as_slice)
            .unwrap_or_default();
        let per_stage =
            post_study_coverage_per_stage(windows, &post_calendar, calendar_stages.len());
        check_fixed_post_study_tiling(thermal.id, &per_stage, &classes.fixed_post_study, ctx);
        check_post_study_window_excludes_unreachable_stages(
            thermal.id,
            &per_stage,
            &classes.carried,
            &classes.beyond_reach,
            ctx,
        );
        check_fixed_commitment_within_window(
            thermal,
            thermal.id,
            windows,
            &post_calendar,
            &classes.commissioning_inactive,
            ctx,
        );
    }
}

/// Per-post-study-stage coverage fraction from `windows`, accumulated once
/// over `post_calendar`. V2's tiling check and V3's carried/beyond-reach
/// rejection both read this same vector rather than each re-walking the
/// windows, so the two rules can never disagree about which calendar is in
/// play.
fn post_study_coverage_per_stage(
    windows: &[&AnticipatedCommitmentHistory],
    post_calendar: &StageCalendar,
    n_post_study_stages: usize,
) -> Vec<f64> {
    let dated: Vec<DatedWindow> = windows
        .iter()
        .map(|r| DatedWindow {
            start_date: r.start_date,
            end_date: r.end_date,
        })
        .collect();

    let mut per_stage = vec![0.0_f64; n_post_study_stages];
    for window in &dated {
        for (total, fraction) in per_stage.iter_mut().zip(post_calendar.coverage(window)) {
            *total += fraction;
        }
    }
    per_stage
}

/// V2: every index in `fixed_post_study` must be tiled by `per_stage` at
/// coverage `1.0`, mirroring [`check_commitment_coverage`]'s study-side walk
/// on the other calendar. Emits one `BusinessRuleViolation` naming every
/// uncovered index; a `commissioning_inactive` post-study stage is never a
/// member of `fixed_post_study`, so it is never demanded here. No-op when
/// `fixed_post_study` is empty — the common case, every existing deck.
#[allow(clippy::float_cmp)] // whole-day-hours coverage keeps the per-stage tiling fraction bit-exact (mirrors check_commitment_coverage)
fn check_fixed_post_study_tiling(
    thermal_id: EntityId,
    per_stage: &[f64],
    fixed_post_study: &[usize],
    ctx: &mut ValidationContext,
) {
    if fixed_post_study.is_empty() {
        return;
    }

    let uncovered: Vec<usize> = fixed_post_study
        .iter()
        .copied()
        .filter(|&j| per_stage[j] != 1.0)
        .collect();
    if uncovered.is_empty() {
        return;
    }

    let entity_str = format!("thermals[id={}].anticipated_config", thermal_id.0);
    ctx.add_error(
        ErrorKind::BusinessRuleViolation,
        "post_study_stages.json",
        Some(&entity_str),
        format!(
            "Thermal {}: past_anticipated_commitments do not tile post-study stage \
             index(es) {uncovered:?} at coverage 1.0; declare the fixed commitment for \
             each (a committed 0 MW is explicit).",
            thermal_id.0
        ),
    );
}

/// V3: a commitment window may not cover a post-study stage the study itself
/// decides (`carried`) or one past the plant's decision reach
/// (`beyond_reach`); the remedies are opposite — lengthen the lead so a
/// carried stage becomes pre-study-decided, or shorten it so a beyond-reach
/// stage falls back inside reach — so each cause is reported as its own
/// error, never merged. A `commissioning_inactive` stage is exempt: a
/// declared window there is inert, not contradictory. No-op when neither
/// class has a covered index — the common case, every existing deck.
#[allow(clippy::float_cmp)] // whole-day-hours coverage keeps the per-stage tiling fraction bit-exact (mirrors check_commitment_coverage)
fn check_post_study_window_excludes_unreachable_stages(
    thermal_id: EntityId,
    per_stage: &[f64],
    carried: &[usize],
    beyond_reach: &[usize],
    ctx: &mut ValidationContext,
) {
    let entity_str = format!("thermals[id={}].anticipated_config", thermal_id.0);

    let covered_carried: Vec<usize> = carried
        .iter()
        .copied()
        .filter(|&j| per_stage[j] != 0.0)
        .collect();
    if !covered_carried.is_empty() {
        ctx.add_error(
            ErrorKind::BusinessRuleViolation,
            "post_study_stages.json",
            Some(&entity_str),
            format!(
                "Thermal {}: past_anticipated_commitments cover post-study stage \
                 index(es) {covered_carried:?}, which the study itself decides; a declared \
                 fixed value there contradicts the study's own decision. Remove the window, \
                 or lengthen anticipated_config's lead so the delivery becomes \
                 pre-study-decided.",
                thermal_id.0
            ),
        );
    }

    let covered_beyond_reach: Vec<usize> = beyond_reach
        .iter()
        .copied()
        .filter(|&j| per_stage[j] != 0.0)
        .collect();
    if !covered_beyond_reach.is_empty() {
        ctx.add_error(
            ErrorKind::BusinessRuleViolation,
            "post_study_stages.json",
            Some(&entity_str),
            format!(
                "Thermal {}: past_anticipated_commitments cover post-study stage \
                 index(es) {covered_beyond_reach:?}, which are past the plant's decision \
                 reach and cannot be represented. Remove the window, or shorten \
                 anticipated_config's lead so the delivery falls inside the reach.",
                thermal_id.0
            ),
        );
    }
}

/// Build the post-study calendar as `Stage`s (each `end_date` = `start_date`
/// advanced by `duration_hours` rounded to whole days — the whole-day alignment
/// [`StageCalendar`] coverage needs, mirroring the pre-study duration→days round
/// in `travel_time`), enforcing invariant (a). Returns the stages and whether
/// they are well-formed; a non-well-formed calendar suppresses the coverage pass
/// so its errors are not noise on a broken calendar.
fn build_post_study_calendar(
    post_study: &PostStudyStages,
    study_end: Option<NaiveDate>,
    ctx: &mut ValidationContext,
) -> (Vec<Stage>, bool) {
    let mut calendar_stages: Vec<Stage> = Vec::with_capacity(post_study.stages.len());
    let mut well_formed = true;

    for (i, stage) in post_study.stages.iter().enumerate() {
        match post_study_end_date(stage.start_date, stage.duration_hours) {
            Some(end_date) if end_date > stage.start_date => {
                calendar_stages.push(make_calendar_stage(
                    i,
                    stage.start_date,
                    end_date,
                    stage.duration_hours,
                ));
            }
            _ => {
                ctx.add_error(
                    ErrorKind::BusinessRuleViolation,
                    "post_study_stages.json",
                    Some(format!("stages[{i}]")),
                    format!(
                        "post-study stage starting {} has duration_hours {} that rounds to a \
                         non-positive whole-day span; declare a duration of at least a day.",
                        stage.start_date, stage.duration_hours
                    ),
                );
                well_formed = false;
            }
        }
    }

    match (calendar_stages.first(), study_end) {
        (Some(first), Some(end)) if first.start_date != end => {
            ctx.add_error(
                ErrorKind::BusinessRuleViolation,
                "post_study_stages.json",
                Some("stages[0].start_date"),
                format!(
                    "first post-study stage starts {} but the study horizon ends {}; the \
                     post-study calendar must begin exactly at the study horizon end.",
                    first.start_date, end
                ),
            );
            well_formed = false;
        }
        (Some(_), None) => {
            ctx.add_error(
                ErrorKind::BusinessRuleViolation,
                "post_study_stages.json",
                Some("stages"),
                "post_study_stages.json is declared but the study has no study stages to anchor \
                 the post-study calendar against."
                    .to_string(),
            );
            well_formed = false;
        }
        _ => {}
    }

    for pair in calendar_stages.windows(2) {
        if pair[0].end_date != pair[1].start_date {
            ctx.add_error(
                ErrorKind::BusinessRuleViolation,
                "post_study_stages.json",
                Some(format!("stages start_date={}", pair[1].start_date)),
                format!(
                    "post-study stages are not date-contiguous: the stage starting {} ends {} \
                     but the next stage starts {}; each stage must end exactly where the next \
                     begins.",
                    pair[0].start_date, pair[0].end_date, pair[1].start_date
                ),
            );
            well_formed = false;
        }
    }

    (calendar_stages, well_formed)
}

/// Post-study stage end date: `start_date` advanced by `duration_hours` rounded
/// to whole days. `None` only on calendar overflow.
#[allow(clippy::cast_possible_truncation)] // a validated finite positive duration yields a small non-negative day count
fn post_study_end_date(start_date: NaiveDate, duration_hours: f64) -> Option<NaiveDate> {
    let days = (duration_hours / 24.0).round() as i64;
    start_date.checked_add_signed(TimeDelta::days(days))
}

/// Synthesize a `Stage` carrying only the fields [`StageCalendar`] reads
/// (`start_date`, `end_date`, `blocks`); the rest are inert placeholders.
fn make_calendar_stage(
    index: usize,
    start_date: NaiveDate,
    end_date: NaiveDate,
    duration_hours: f64,
) -> Stage {
    Stage {
        index,
        id: i32::try_from(index).unwrap_or(i32::MAX),
        start_date,
        end_date,
        season_id: None,
        blocks: vec![Block {
            index: 0,
            name: "post_study".to_string(),
            duration_hours,
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
    use cobre_core::temporal::{Block, PolicyGraphType, Stage};
    use cobre_core::{
        AnticipatedCommitmentHistory, AnticipatedConfig, EntityId, HorizonGraph, PostStudyStage,
        PostStudyStages, PostStudyThermalBound, Thermal,
    };

    use chrono::{NaiveDate, TimeDelta};

    use super::super::validate_semantic_hydro_thermal;
    use super::{
        ExtendedDeliveryAxis, build_extended_delivery_axis, check_anticipated_thermals,
        check_post_study_stages, classify_deliveries, extended_deciders, lead_delivery_stage_count,
    };
    use crate::constraints::ThermalBoundsRow;
    use crate::stages::StagesData;
    use crate::test_support::*;
    use crate::validation::schema::ParsedData;
    use crate::validation::{ErrorKind, ValidationContext};

    // ── Helper: build a thermal with anticipated_config ───────────────────────

    fn make_anticipated_thermal(
        id: i32,
        lead_stages: u32,
        entry_stage_id: Option<i32>,
        exit_stage_id: Option<i32>,
    ) -> Thermal {
        Thermal {
            anticipated_config: Some(AnticipatedConfig::LeadStages(lead_stages)),
            entry_stage_id,
            exit_stage_id,
            ..make_thermal(id, 0.0, 500.0)
        }
    }

    /// Build a `LeadTime`-configured thermal.
    fn make_lead_time_anticipated_thermal(
        id: i32,
        lead_time_hours: f64,
        entry_stage_id: Option<i32>,
        exit_stage_id: Option<i32>,
    ) -> Thermal {
        Thermal {
            anticipated_config: Some(AnticipatedConfig::LeadTime(lead_time_hours)),
            entry_stage_id,
            exit_stage_id,
            ..make_thermal(id, 0.0, 500.0)
        }
    }

    /// Anchor of every anticipated-thermal test calendar; stage 0 starts here.
    fn calendar_anchor() -> NaiveDate {
        NaiveDate::from_ymd_opt(2024, 1, 1).unwrap()
    }

    /// Contiguous stage boundary dates for a calendar whose stage `i` spans
    /// `durations_hours[i] / 24` whole days from [`calendar_anchor`]; length is
    /// `durations_hours.len() + 1`. Every fixture duration is a whole-day
    /// multiple so the date span and block hours agree, the alignment
    /// [`StageCalendar`] coverage needs.
    fn stage_boundaries(durations_hours: &[f64]) -> Vec<NaiveDate> {
        let mut cursor = calendar_anchor();
        let mut dates = vec![cursor];
        for &hours in durations_hours {
            cursor += TimeDelta::days((hours / 24.0) as i64);
            dates.push(cursor);
        }
        dates
    }

    /// Contiguous study stages (ids `0..durations_hours.len()`), stage `i` a
    /// single block of `durations_hours[i]` over `[boundary[i], boundary[i+1])`.
    fn contiguous_stages(durations_hours: &[f64]) -> Vec<Stage> {
        let boundaries = stage_boundaries(durations_hours);
        durations_hours
            .iter()
            .enumerate()
            .map(|(i, &hours)| {
                let mut stage = make_stage(i as i32);
                stage.index = i;
                stage.start_date = boundaries[i];
                stage.end_date = boundaries[i + 1];
                stage.blocks = vec![Block {
                    index: 0,
                    name: "S".to_string(),
                    duration_hours: hours,
                }];
                stage
            })
            .collect()
    }

    /// One commitment window per value, tiling the leading `values.len()` stages
    /// of the contiguous calendar defined by `durations_hours`. `values.len()`
    /// may differ from `durations_hours.len()` (fewer → gap; more → beyond the
    /// horizon) to build the coverage-failure fixtures.
    fn commitments_on(
        thermal_id: i32,
        values: &[f64],
        durations_hours: &[f64],
    ) -> Vec<AnticipatedCommitmentHistory> {
        let boundaries = stage_boundaries(durations_hours);
        values
            .iter()
            .enumerate()
            .map(|(i, &value_mw)| AnticipatedCommitmentHistory {
                thermal_id: EntityId::from(thermal_id),
                start_date: boundaries[i],
                end_date: boundaries[i + 1],
                value_mw,
            })
            .collect()
    }

    /// One commitment window per value on the uniform 30-day calendar
    /// [`make_data_anticipated`] builds — value `i` delivers over study stage
    /// `i`.
    fn commitments(thermal_id: i32, values: &[f64]) -> Vec<AnticipatedCommitmentHistory> {
        commitments_on(thermal_id, values, &vec![720.0; values.len().max(1)])
    }

    /// Assemble `ParsedData` from a contiguous stage calendar and commitments.
    fn anticipated_data(
        thermals: Vec<Thermal>,
        durations_hours: &[f64],
        past_anticipated_commitments: Vec<AnticipatedCommitmentHistory>,
    ) -> ParsedData {
        let stages_data = StagesData {
            openings_declared: std::collections::HashSet::new(),
            stages: contiguous_stages(durations_hours),
            policy_graph: HorizonGraph {
                stage_discount_rate_overrides: std::collections::BTreeMap::new(),
                graph_type: PolicyGraphType::FiniteHorizon,
                annual_discount_rate: 0.06,
                transitions: vec![],
                nodes: Vec::new(),
                season_map: None,
            },
        };
        let mut data = make_data(vec![], thermals, vec![], stages_data, vec![], vec![]);
        data.initial_conditions.past_anticipated_commitments = past_anticipated_commitments;
        data
    }

    /// Build `ParsedData` for anticipated-thermal tests on a non-uniform
    /// per-stage calendar (`durations_hours`, one entry per study stage,
    /// `id = 0..durations_hours.len()`).
    fn make_data_anticipated_with_durations(
        thermals: Vec<Thermal>,
        durations_hours: &[f64],
        past_anticipated_commitments: Vec<AnticipatedCommitmentHistory>,
    ) -> ParsedData {
        anticipated_data(thermals, durations_hours, past_anticipated_commitments)
    }

    /// Build `ParsedData` for anticipated-thermal tests on a uniform 30-day
    /// (720 h) calendar of `n_stages` stages with IDs `0..n_stages`.
    fn make_data_anticipated(
        thermals: Vec<Thermal>,
        n_stages: usize,
        past_anticipated_commitments: Vec<AnticipatedCommitmentHistory>,
    ) -> ParsedData {
        anticipated_data(
            thermals,
            &vec![720.0; n_stages],
            past_anticipated_commitments,
        )
    }

    /// Build a `PostStudyStages` whose calendar is date-contiguous from the study
    /// horizon end (`stage_boundaries(study_durations)` last date), one stage per
    /// `post_durations` entry; `bounds` is a slice of
    /// `(thermal_id, post_study_stage_index, min_mw, max_mw)`.
    fn post_study_from(
        study_durations: &[f64],
        post_durations: &[f64],
        bounds: &[(i32, usize, f64, f64)],
    ) -> PostStudyStages {
        let boundaries = stage_boundaries(study_durations);
        let mut cursor = boundaries[study_durations.len()];
        let mut stages = Vec::with_capacity(post_durations.len());
        for &hours in post_durations {
            stages.push(PostStudyStage {
                start_date: cursor,
                duration_hours: hours,
            });
            cursor += TimeDelta::days((hours / 24.0) as i64);
        }
        let thermal_bounds = bounds
            .iter()
            .map(|&(thermal_id, j, min_mw, max_mw)| PostStudyThermalBound {
                thermal_id: EntityId::from(thermal_id),
                post_study_stage_index: j,
                cost_per_mwh: 100.0,
                min_mw,
                max_mw,
            })
            .collect();
        PostStudyStages {
            stages,
            thermal_bounds,
        }
    }

    /// [`make_data_anticipated_with_durations`] with a `post_study_stages`
    /// attached.
    fn data_with_post_study(
        thermals: Vec<Thermal>,
        durations_hours: &[f64],
        past_anticipated_commitments: Vec<AnticipatedCommitmentHistory>,
        post_study: PostStudyStages,
    ) -> ParsedData {
        let mut data = make_data_anticipated_with_durations(
            thermals,
            durations_hours,
            past_anticipated_commitments,
        );
        data.post_study_stages = Some(post_study);
        data
    }

    // ── AC-1: valid anticipated thermal — no errors ──────────────────────────

    /// Given a valid anticipated thermal (lead_stages=2, n_stages=5, no entry/exit,
    /// two all-zero commitment windows tiling stages 0..2), semantic validation
    /// returns no errors.
    #[test]
    fn test_valid_anticipated_thermal_ok() {
        let thermal = make_anticipated_thermal(1, 2, None, None);
        let data = make_data_anticipated(vec![thermal], 5, commitments(1, &[0.0, 0.0]));
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(
            !ctx.has_errors(),
            "expected no errors, got: {:?}",
            ctx.errors()
        );
    }

    // ── LeadTime coverage ─────────────────────────────────────────────────────

    /// Given an otherwise-valid `LeadTime`-configured thermal on the
    /// weekly-then-monthly PMO calendar (calendar-covered commitments), when
    /// semantic validation runs, then no `NotImplemented` error is produced and
    /// the study passes validation entirely: `LeadTime` falls through to the
    /// shared coverage/bounds/window checks unchanged.
    #[test]
    fn test_lead_time_thermal_accepted() {
        let thermal = make_lead_time_anticipated_thermal(1, 720.0, None, None);
        let durations = [168.0, 168.0, 168.0, 168.0, 720.0, 720.0];
        let data = make_data_anticipated_with_durations(
            vec![thermal],
            &durations,
            commitments_on(1, &[0.0, 0.0, 0.0, 0.0], &durations),
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(
            !ctx.errors()
                .iter()
                .any(|e| e.kind == ErrorKind::NotImplemented),
            "LeadTime must no longer be rejected as NotImplemented, got: {:?}",
            ctx.errors()
        );
        assert!(
            !ctx.has_errors(),
            "a valid LeadTime thermal with calendar-covered commitments must pass \
             validation entirely, got: {:?}",
            ctx.errors()
        );
    }

    /// Given a valid single-decider `LeadTime(744.0)` thermal on a uniform
    /// 3×744h calendar with a matching one-entry `past_anticipated_commitments`
    /// history, when semantic validation runs, then the study passes validation
    /// with no errors — the first `LeadTime` config the gate admits through to
    /// setup.
    #[test]
    fn lead_time_single_decider_passes_validation() {
        let thermal = make_lead_time_anticipated_thermal(1, 744.0, None, None);
        let durations = [744.0, 744.0, 744.0];
        let data = make_data_anticipated_with_durations(
            vec![thermal],
            &durations,
            commitments_on(1, &[0.0], &durations),
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(
            !ctx.has_errors(),
            "a valid single-decider LeadTime thermal must pass validation, got: {:?}",
            ctx.errors()
        );
    }

    // ── AC-2: missing history entry ───────────────────────────────────────────

    /// Given an anticipated thermal with no matching entry in
    /// past_anticipated_commitments, a `BusinessRuleViolation` names the
    /// thermal id and "missing".
    #[test]
    fn test_missing_history_entry_error() {
        let thermal = make_anticipated_thermal(1, 2, None, None);
        let data = make_data_anticipated(vec![thermal], 5, vec![]);
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(ctx.has_errors());
        let errors = ctx.errors();
        let relevant: Vec<_> = errors
            .iter()
            .filter(|e| e.kind == ErrorKind::BusinessRuleViolation)
            .collect();
        assert!(
            !relevant.is_empty(),
            "expected BusinessRuleViolation, got: {errors:?}"
        );
        let msg = &relevant[0].message;
        assert!(
            msg.contains("Thermal 1"),
            "message should contain 'Thermal 1', got: {msg}"
        );
        assert!(
            msg.contains("missing"),
            "message should contain 'missing', got: {msg}"
        );
        let file = relevant[0].file.to_string_lossy();
        assert!(
            file.contains("initial_conditions"),
            "file path should reference initial_conditions, got: {file}"
        );
        let entity = relevant[0].entity.as_deref().unwrap_or("");
        assert!(
            entity.contains("initial_conditions.past_anticipated_commitments"),
            "entity should contain 'initial_conditions.past_anticipated_commitments', got: {entity}"
        );
    }

    // ── AC-3: history entry for non-anticipated thermal ───────────────────────

    /// Given a history entry whose thermal_id references a thermal with
    /// anticipated_config == None, a `BusinessRuleViolation` names "not an
    /// anticipated thermal".
    #[test]
    fn test_history_entry_for_non_anticipated_thermal_error() {
        // thermal 1 is NOT anticipated
        let thermal = make_thermal(1, 0.0, 500.0);
        let data = make_data_anticipated(vec![thermal], 5, commitments(1, &[100.0, 200.0]));
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(ctx.has_errors());
        let errors = ctx.errors();
        let relevant: Vec<_> = errors
            .iter()
            .filter(|e| e.kind == ErrorKind::BusinessRuleViolation)
            .collect();
        assert!(
            !relevant.is_empty(),
            "expected BusinessRuleViolation, got: {errors:?}"
        );
        let msg = &relevant[0].message;
        assert!(
            msg.contains("not an anticipated thermal"),
            "message should contain 'not an anticipated thermal', got: {msg}"
        );
    }

    // ── AC-4: windows beyond the lead horizon (over-coverage) ────────────────

    /// Given lead_stages=2 but three tiling windows (covering stages 0..3), the
    /// window on stage 2 lies beyond the leading K=2 delivery horizon and is
    /// rejected with a named over-coverage error.
    #[test]
    fn test_over_coverage_beyond_lead_horizon_rejected() {
        let thermal = make_anticipated_thermal(1, 2, None, None);
        let data = make_data_anticipated(vec![thermal], 5, commitments(1, &[100.0, 200.0, 300.0]));
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(ctx.has_errors());
        let errors = ctx.errors();
        let relevant: Vec<_> = errors
            .iter()
            .filter(|e| {
                e.kind == ErrorKind::BusinessRuleViolation
                    && e.message.contains("beyond the leading")
            })
            .collect();
        assert_eq!(
            relevant.len(),
            1,
            "expected exactly one over-coverage error, got: {errors:?}"
        );
        let msg = &relevant[0].message;
        assert!(
            msg.contains('2'),
            "message should name over-covered study stage id 2, got: {msg}"
        );
    }

    // ── AC-5: gap in the lead horizon (under-coverage) ───────────────────────

    /// Given lead_stages=2 but only one tiling window (covering stage 0), study
    /// stage 1 is uncovered and rejected with a named under-coverage error.
    #[test]
    fn test_under_coverage_gap_in_lead_horizon_rejected() {
        let thermal = make_anticipated_thermal(1, 2, None, None);
        let data = make_data_anticipated(vec![thermal], 5, commitments(1, &[100.0]));
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(ctx.has_errors());
        let errors = ctx.errors();
        let relevant: Vec<_> = errors
            .iter()
            .filter(|e| {
                e.kind == ErrorKind::BusinessRuleViolation
                    && e.message.contains("do not tile the leading")
            })
            .collect();
        assert_eq!(
            relevant.len(),
            1,
            "expected exactly one under-coverage error, got: {errors:?}"
        );
        let msg = &relevant[0].message;
        assert!(
            msg.contains('1'),
            "message should name the uncovered study stage id 1, got: {msg}"
        );
    }

    // ── study-side coverage boundary: post-horizon windows are legal ─────────

    /// Given a plant with `k_i == 2` over four study stages, windows tiling
    /// study stages `0` and `1`, and one additional window covering only
    /// dates on the post-study calendar, `check_anticipated_thermals` emits no
    /// error: a window entirely past the study horizon contributes nothing to
    /// the study-side coverage sums and is not reported as a gap or as
    /// over-coverage.
    #[test]
    fn commitment_coverage_accepts_a_window_past_the_study_horizon() {
        let thermal = make_anticipated_thermal(1, 2, None, None);
        let study_durations = vec![720.0; 4];
        let post_study = post_study_from(&study_durations, &[720.0], &[]);
        let mut commitments = commitments_on(1, &[0.0, 0.0], &study_durations);
        commitments.push(AnticipatedCommitmentHistory {
            thermal_id: EntityId::from(1),
            start_date: post_study.stages[0].start_date,
            end_date: post_study.stages[0].start_date + TimeDelta::days(30),
            value_mw: 0.0,
        });
        let data = data_with_post_study(vec![thermal], &study_durations, commitments, post_study);
        let mut ctx = ValidationContext::new();
        check_anticipated_thermals(&data, &mut ctx);
        assert!(
            !ctx.has_errors(),
            "a window past the study horizon must not be reported as a gap or as \
             over-coverage, got: {:?}",
            ctx.errors()
        );
    }

    /// Given a plant with `k_i == 2` over four study stages and a window
    /// covering study stage `2`, `check_anticipated_thermals` emits a
    /// `BusinessRuleViolation` naming study stage id `2`, whose message
    /// contains neither "past the pre-study delivery horizon" nor any claim
    /// that a window may not extend past the horizon.
    #[test]
    fn commitment_coverage_rejects_a_window_on_a_study_decided_stage() {
        let thermal = make_anticipated_thermal(1, 2, None, None);
        let data = make_data_anticipated(vec![thermal], 4, commitments(1, &[0.0, 0.0, 0.0]));
        let mut ctx = ValidationContext::new();
        check_anticipated_thermals(&data, &mut ctx);
        assert!(ctx.has_errors());
        let errors = ctx.errors();
        let relevant: Vec<_> = errors
            .iter()
            .filter(|e| {
                e.kind == ErrorKind::BusinessRuleViolation
                    && e.message.contains("beyond the leading")
            })
            .collect();
        assert_eq!(
            relevant.len(),
            1,
            "expected exactly one over-coverage error, got: {errors:?}"
        );
        let msg = &relevant[0].message;
        assert!(
            msg.contains('2'),
            "message should name over-covered study stage id 2, got: {msg}"
        );
        assert!(
            !msg.contains("past the pre-study delivery horizon"),
            "the retired horizon phrasing must not appear, got: {msg}"
        );
        assert!(
            !msg.contains("extend past the horizon"),
            "the message must not claim a window may not extend past the horizon, got: {msg}"
        );
    }

    /// The gap-error path is untouched by the over-coverage message rewrite:
    /// given `k_i == 3` and windows tiling only study stages `0` and `1`,
    /// `check_anticipated_thermals` emits the gap error naming uncovered study
    /// stage id `2`, byte-identical to its pre-existing text.
    #[test]
    fn commitment_coverage_gap_message_is_unchanged() {
        let thermal = make_anticipated_thermal(1, 3, None, None);
        let data = make_data_anticipated(vec![thermal], 5, commitments(1, &[0.0, 0.0]));
        let mut ctx = ValidationContext::new();
        check_anticipated_thermals(&data, &mut ctx);
        assert!(ctx.has_errors());
        let errors = ctx.errors();
        let relevant: Vec<_> = errors
            .iter()
            .filter(|e| {
                e.kind == ErrorKind::BusinessRuleViolation
                    && e.message.contains("do not tile the leading")
            })
            .collect();
        assert_eq!(
            relevant.len(),
            1,
            "expected exactly one gap error, got: {errors:?}"
        );
        assert_eq!(
            relevant[0].message,
            "Thermal 1: past_anticipated_commitments do not tile the leading 3 delivery \
             stage(s) at coverage 1.0; study stage id(s) [2] are not covered exactly once. \
             Write a commitment window (a committed 0 MW is explicit) for every leading \
             delivery stage."
        );
    }

    // ── AC-6: lead_stages exceeds study horizon ───────────────────────────────

    /// Given lead_stages=10 and n_stages=5, a `BusinessRuleViolation` names
    /// "lead_stages exceeds study horizon".
    #[test]
    fn test_lead_stages_exceeds_study_horizon_error() {
        let thermal = make_anticipated_thermal(1, 10, None, None);
        let data = make_data_anticipated(vec![thermal], 5, vec![]);
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(ctx.has_errors());
        let errors = ctx.errors();
        let relevant: Vec<_> = errors
            .iter()
            .filter(|e| {
                e.kind == ErrorKind::BusinessRuleViolation
                    && e.message.contains("lead_stages exceeds study horizon")
            })
            .collect();
        assert!(
            !relevant.is_empty(),
            "expected BusinessRuleViolation with 'lead_stages exceeds study horizon', got: {errors:?}"
        );
    }

    // ── Boundary: lead_stages == n_stages is accepted (strict-greater check) ──

    /// Boundary case for AC-6: lead_stages == n_stages must NOT error.
    /// The horizon check is `lead_stages > n_stages` (strict). A plant whose
    /// first delivery lands on the final study stage is a valid configuration.
    #[test]
    fn test_lead_stages_equal_n_stages_ok() {
        let thermal = make_anticipated_thermal(1, 5, None, None);
        let data =
            make_data_anticipated(vec![thermal], 5, commitments(1, &[0.0, 0.0, 0.0, 0.0, 0.0]));
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(
            !ctx.has_errors(),
            "lead_stages == n_stages must be accepted, got: {:?}",
            ctx.errors()
        );
    }

    // ── lead_time exceeds study horizon ───────────────────────────────────────

    /// Given a `LeadTime(delta_hours)` thermal whose `delta_hours` strictly
    /// exceeds the summed study-stage durations, when semantic validation runs,
    /// then exactly one `BusinessRuleViolation` is appended naming the thermal
    /// id, `delta_hours`, and the total horizon hours.
    #[test]
    fn test_lead_time_exceeds_study_horizon_error() {
        let thermal = make_lead_time_anticipated_thermal(1, 3000.0, None, None);
        let data = make_data_anticipated_with_durations(
            vec![thermal],
            &[168.0, 168.0, 168.0, 168.0, 720.0, 720.0],
            vec![],
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        let errors = ctx.errors();
        let relevant: Vec<_> = errors
            .iter()
            .filter(|e| {
                e.kind == ErrorKind::BusinessRuleViolation
                    && e.message.contains("lead_time exceeds study horizon")
            })
            .collect();
        assert_eq!(
            relevant.len(),
            1,
            "expected exactly one BusinessRuleViolation with 'lead_time exceeds study horizon', got: {errors:?}"
        );
        let msg = &relevant[0].message;
        assert!(
            msg.contains("Thermal 1"),
            "message should contain 'Thermal 1', got: {msg}"
        );
        assert!(
            msg.contains("3000"),
            "message should contain delta_hours 3000, got: {msg}"
        );
        assert!(
            msg.contains("2112"),
            "message should contain the total horizon hours 2112, got: {msg}"
        );
    }

    // ── Boundary: lead_time == total horizon is accepted (strict-greater check) ──

    /// Boundary case: `delta_hours == total_horizon_hours` must NOT error. The
    /// horizon check is `delta_hours > total_horizon_hours` (strict), mirroring
    /// the `LeadStages` boundary — a delivery landing exactly on the final study
    /// stage is a valid configuration.
    #[test]
    fn test_lead_time_equal_total_horizon_ok() {
        let thermal = make_lead_time_anticipated_thermal(1, 2112.0, None, None);
        let durations = [168.0, 168.0, 168.0, 168.0, 720.0, 720.0];
        let data = make_data_anticipated_with_durations(
            vec![thermal],
            &durations,
            commitments_on(1, &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], &durations),
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(
            !ctx.has_errors(),
            "delta_hours == total_horizon_hours must be accepted, got: {:?}",
            ctx.errors()
        );
    }

    // ── Post-horizon delivery exemption from the lead-horizon cap ────────────

    /// Given a `LeadTime(delta_hours)` thermal whose `delta_hours` strictly
    /// exceeds the summed study-stage durations, when the study declares a
    /// post-study stage its extended lead reaches (decided in-study, delivered
    /// post-horizon), then no `BusinessRuleViolation` naming "lead_time exceeds
    /// study horizon" is produced — the re-keyed lead-cap exemption.
    #[test]
    fn test_lead_time_exceeding_horizon_with_reachable_post_study_accepted() {
        let thermal = make_lead_time_anticipated_thermal(1, 3000.0, None, None);
        let durations = [168.0, 168.0, 168.0, 168.0, 720.0, 720.0];
        // A 60-day post-study stage: the 3000h lead, decided at study stage 3,
        // delivers into post-study stage 0.
        let post_study = post_study_from(&durations, &[1440.0], &[(1, 0, 0.0, 500.0)]);
        let data = data_with_post_study(vec![thermal], &durations, vec![], post_study);
        let mut ctx = ValidationContext::new();
        check_anticipated_thermals(&data, &mut ctx);
        let errors = ctx.errors();
        let lead_cap_errors: Vec<_> = errors
            .iter()
            .filter(|e| {
                e.kind == ErrorKind::BusinessRuleViolation
                    && e.message.contains("lead_time exceeds study horizon")
            })
            .collect();
        assert!(
            lead_cap_errors.is_empty(),
            "a thermal whose lead reaches a declared post-study stage must be exempt from the \
             lead-horizon cap, got: {errors:?}"
        );
    }

    /// Given the same over-horizon `LeadTime` thermal but with NO post-study
    /// stages declared, the lead-horizon cap still rejects it: the exemption is
    /// scoped to a study that declares a reachable post-study stage, never a
    /// blanket relaxation of the cap.
    #[test]
    fn test_lead_time_exceeding_horizon_without_post_study_still_rejected() {
        let thermal = make_lead_time_anticipated_thermal(1, 3000.0, None, None);
        let durations = [168.0, 168.0, 168.0, 168.0, 720.0, 720.0];
        let data = make_data_anticipated_with_durations(vec![thermal], &durations, vec![]);
        assert!(
            data.post_study_stages.is_none(),
            "fixture must declare no post-study stages"
        );
        let mut ctx = ValidationContext::new();
        check_anticipated_thermals(&data, &mut ctx);
        let errors = ctx.errors();
        let relevant: Vec<_> = errors
            .iter()
            .filter(|e| {
                e.kind == ErrorKind::BusinessRuleViolation
                    && e.message.contains("lead_time exceeds study horizon")
            })
            .collect();
        assert_eq!(
            relevant.len(),
            1,
            "expected exactly one BusinessRuleViolation with 'lead_time exceeds study horizon', got: {errors:?}"
        );
    }

    /// Given a `LeadTime` lead whose reach is entirely class-4 (empty
    /// `carried`, non-empty `fixed_post_study` — the full-anticipation regime),
    /// when `check_anticipated_thermals` runs, then the widened
    /// `reaches_post_study` predicate exempts it from the lead-horizon cap: no
    /// "lead_time exceeds study horizon" error is emitted. A single 720h study
    /// stage with `LeadTime(2000)` decides its only post-study stage pre-study.
    #[test]
    fn lead_exceeding_the_horizon_is_accepted_when_it_reaches_a_fixed_post_study_stage() {
        let thermal = make_lead_time_anticipated_thermal(1, 2000.0, None, None);
        let durations = [720.0];
        let post_study = post_study_from(&durations, &[720.0], &[(1, 0, 0.0, 500.0)]);
        let data = data_with_post_study(vec![thermal], &durations, vec![], post_study);
        let mut ctx = ValidationContext::new();
        check_anticipated_thermals(&data, &mut ctx);
        let errors = ctx.errors();
        let lead_cap_errors: Vec<_> = errors
            .iter()
            .filter(|e| {
                e.kind == ErrorKind::BusinessRuleViolation
                    && e.message.contains("lead_time exceeds study horizon")
            })
            .collect();
        assert!(
            lead_cap_errors.is_empty(),
            "a thermal whose lead reaches only a class-4 (pre-study-decided) post-study stage \
             must be exempt from the lead-horizon cap, got: {errors:?}"
        );
    }

    /// Given a `LeadTime` lead exceeding the study horizon with NO post-study
    /// stages declared at all, when `check_anticipated_thermals` runs, then the
    /// lead-horizon cap still rejects it — the pre-existing guard for a plant
    /// that genuinely can never deliver.
    #[test]
    fn lead_exceeding_the_horizon_without_post_study_stages_is_still_rejected() {
        let thermal = make_lead_time_anticipated_thermal(1, 2000.0, None, None);
        let durations = [720.0];
        let data = make_data_anticipated_with_durations(vec![thermal], &durations, vec![]);
        assert!(
            data.post_study_stages.is_none(),
            "fixture must declare no post-study stages"
        );
        let mut ctx = ValidationContext::new();
        check_anticipated_thermals(&data, &mut ctx);
        let errors = ctx.errors();
        let relevant: Vec<_> = errors
            .iter()
            .filter(|e| {
                e.kind == ErrorKind::BusinessRuleViolation
                    && e.message.contains("lead_time exceeds study horizon")
            })
            .collect();
        assert_eq!(
            relevant.len(),
            1,
            "expected exactly one BusinessRuleViolation with 'lead_time exceeds study horizon', got: {errors:?}"
        );
    }

    /// Given a `LeadTime` thermal exceeding the horizon whose extended lead
    /// reaches a declared post-study stage AND `past_anticipated_commitments`
    /// tiling every leading study stage at 0 MW (`lead_delivery_stage_count`
    /// resolves to `n_stages` whenever `delta_hours` exceeds the horizon),
    /// `check_anticipated_thermals` validates end-to-end with no errors: the
    /// re-keyed lead-cap exemption composes cleanly with the full-tiling coverage
    /// contract.
    #[test]
    fn test_post_horizon_delivery_thermal_tiling_all_study_stages_validates() {
        let thermal = make_lead_time_anticipated_thermal(1, 3000.0, None, None);
        let durations = [168.0, 168.0, 168.0, 168.0, 720.0, 720.0];
        let post_study = post_study_from(&durations, &[1440.0], &[(1, 0, 0.0, 500.0)]);
        let data = data_with_post_study(
            vec![thermal],
            &durations,
            commitments_on(1, &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], &durations),
            post_study,
        );
        let mut ctx = ValidationContext::new();
        check_anticipated_thermals(&data, &mut ctx);
        assert!(
            !ctx.has_errors(),
            "a post-horizon-delivery thermal tiling every leading study stage must validate \
             cleanly through check_anticipated_thermals, got: {:?}",
            ctx.errors()
        );
    }

    // ── AC: the widened `LeadStages` guard matches `LeadTime` (full anticipation) ──

    /// Given a `LeadStages(7)` thermal on a uniform 4×720h study horizon
    /// (`n_stages = 4`) whose extended lead reaches only post-study stages —
    /// three pre-study-decided (`fixed_post_study`, tiled by
    /// `past_anticipated_commitments`) and one study-decided (`carried`,
    /// bound by a declared `PostStudyThermalBound`) —
    /// `validate_semantic_hydro_thermal` returns no errors: the `lead_stages`
    /// guard now shares the `lead_time` guard's `reaches_post_study`
    /// exemption instead of unconditionally rejecting `lead_stages >
    /// n_stages`.
    #[test]
    fn full_anticipation_lead_stages_reaching_only_post_study_stages_validates() {
        let thermal = make_anticipated_thermal(1, 7, None, None);
        let study_durations = [720.0; 4];
        let post_durations = [720.0; 4];
        let post_study = post_study_from(&study_durations, &post_durations, &[(1, 3, 0.0, 500.0)]);
        let concatenated: Vec<f64> = study_durations
            .iter()
            .copied()
            .chain(post_durations[..3].iter().copied())
            .collect();
        let commitments = commitments_on(1, &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 500.0], &concatenated);
        let data = data_with_post_study(vec![thermal], &study_durations, commitments, post_study);
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(
            !ctx.has_errors(),
            "a full-anticipation LeadStages thermal reaching only post-study stages must \
             validate cleanly end-to-end, got: {:?}",
            ctx.errors()
        );
    }

    /// Given a `LeadStages(5)` thermal on the same 4×720h study horizon whose
    /// extended lead exceeds `n_stages` but reaches a `carried` (study-decided)
    /// post-study stage with no declared `PostStudyThermalBound` cell, the
    /// widened guard does not reject it (the plant IS reachable), but
    /// `check_post_study_stages`'s Rule 1 still reports the missing bound cell:
    /// the guard widening exempts a reachable plant from the horizon cap
    /// without masking the genuine missing-bound defect.
    #[test]
    fn lead_stages_short_lead_reaching_carried_stage_without_bound_still_rejected() {
        let thermal = make_anticipated_thermal(1, 5, None, None);
        let study_durations = [720.0; 4];
        let post_durations = [720.0, 720.0];
        let post_study = post_study_from(&study_durations, &post_durations, &[]);
        let concatenated: Vec<f64> = study_durations
            .iter()
            .copied()
            .chain(post_durations[..1].iter().copied())
            .collect();
        let commitments = commitments_on(1, &[0.0, 0.0, 0.0, 0.0, 0.0], &concatenated);
        let data = data_with_post_study(vec![thermal], &study_durations, commitments, post_study);
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        let errors = ctx.errors();
        assert!(
            !errors
                .iter()
                .any(|e| e.message.contains("lead_stages exceeds study horizon")),
            "the widened guard must not fire when the plant reaches a carried post-study \
             stage, got: {errors:?}"
        );
        assert!(
            errors
                .iter()
                .any(|e| e.message.contains("no thermal_bounds entry")
                    && e.message.contains("post_study_stage_index 1")),
            "check_post_study_stages must still reject the missing bound cell on the \
             reached carried stage, got: {errors:?}"
        );
    }

    /// `LeadStages(7)` and the equivalent `LeadTime(5040.0)` (`7 × 720h`) on the
    /// identical uniform 4×720h study + 4×720h post-study calendar resolve to
    /// the same delivery classification and both validate with no errors: the
    /// two lead modes hit the widened horizon guard identically (AC4).
    #[test]
    fn lead_stages_and_lead_time_resolve_the_horizon_guard_identically() {
        let study_durations = [720.0; 4];
        let post_durations = [720.0; 4];
        let values = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 500.0];
        let concatenated: Vec<f64> = study_durations
            .iter()
            .copied()
            .chain(post_durations[..3].iter().copied())
            .collect();

        let lead_stages_thermal = make_anticipated_thermal(1, 7, None, None);
        let post_study_a =
            post_study_from(&study_durations, &post_durations, &[(1, 3, 0.0, 500.0)]);
        let data_a = data_with_post_study(
            vec![lead_stages_thermal],
            &study_durations,
            commitments_on(1, &values, &concatenated),
            post_study_a,
        );
        let mut ctx_a = ValidationContext::new();
        validate_semantic_hydro_thermal(&data_a, &mut ctx_a);
        assert!(
            !ctx_a.has_errors(),
            "LeadStages(7) must validate cleanly, got: {:?}",
            ctx_a.errors()
        );

        let lead_time_thermal = make_lead_time_anticipated_thermal(1, 5040.0, None, None);
        let post_study_b =
            post_study_from(&study_durations, &post_durations, &[(1, 3, 0.0, 500.0)]);
        let data_b = data_with_post_study(
            vec![lead_time_thermal],
            &study_durations,
            commitments_on(1, &values, &concatenated),
            post_study_b,
        );
        let mut ctx_b = ValidationContext::new();
        validate_semantic_hydro_thermal(&data_b, &mut ctx_b);
        assert!(
            !ctx_b.has_errors(),
            "the equivalent LeadTime(5040.0) must validate cleanly identically, got: {:?}",
            ctx_b.errors()
        );
    }

    // ── The concatenated-calendar walk: extended_deciders / classify_deliveries ──

    /// `extended_deciders` mirrors the solver's `resolve_point` on the PMO
    /// calendar (weekly-then-monthly), producing the same delivery-anchored
    /// decider shape `[None×4, Some(3), Some(4), Some(5)]` — the fixture pins
    /// cobre-io's independent walk to the solver's resolution shape.
    #[test]
    fn test_extended_deciders_lead_time_pmo_matches_solver_resolution() {
        let durations = [168.0, 168.0, 168.0, 168.0, 720.0, 720.0, 720.0];
        let deciders = extended_deciders(AnticipatedConfig::LeadTime(720.0), &durations);
        assert_eq!(
            deciders,
            vec![None, None, None, None, Some(3), Some(4), Some(5)]
        );
    }

    /// `extended_deciders` for `LeadStages(l)` is a bare stage shift `c(m) = m − l`
    /// (calendar never consulted), independent of the (non-uniform) durations.
    #[test]
    fn test_extended_deciders_lead_stages_is_bare_shift() {
        let durations = [168.0, 720.0, 168.0, 720.0, 168.0, 720.0];
        let deciders = extended_deciders(AnticipatedConfig::LeadStages(2), &durations);
        assert_eq!(
            deciders,
            vec![None, None, Some(0), Some(1), Some(2), Some(3)]
        );
    }

    /// Uniform post-study durations: an over-horizon lead splits the post-study
    /// deliveries into a pre-study-decided prefix (`fixed_post_study`) and an
    /// in-study-decided suffix (`carried`). Two study stages (ids 0,1) plus two
    /// continued post-study stages (ids 2,3); `LeadTime(2500)` on a `720`h
    /// calendar decides post-study stage 0 pre-study and post-study stage 1 at
    /// study stage 0.
    #[test]
    fn test_classify_deliveries_uniform_splits_carried_and_fixed_post_study() {
        let axis = ExtendedDeliveryAxis {
            durations: vec![720.0, 720.0, 720.0, 720.0],
            stage_ids: vec![0, 1, 2, 3],
            n_stages: 2,
        };
        let classes = classify_deliveries(
            AnticipatedConfig::LeadTime(2500.0),
            Some(&axis),
            &[720.0, 720.0],
            2,
            (None, None),
        );
        assert_eq!(classes.carried, vec![1]);
        assert_eq!(classes.fixed_post_study, vec![0]);
    }

    /// Non-uniform post-study durations: `LeadTime(1000)` on a `[168,720|720,720]`
    /// extended calendar reaches post-study stage 0 from an in-study decision
    /// (study stage 1) and no pre-study-decided prefix.
    #[test]
    fn test_classify_deliveries_non_uniform_reaches_first_post_study_stage() {
        let axis = ExtendedDeliveryAxis {
            durations: vec![168.0, 720.0, 720.0, 720.0],
            stage_ids: vec![0, 1, 2, 3],
            n_stages: 2,
        };
        let classes = classify_deliveries(
            AnticipatedConfig::LeadTime(1000.0),
            Some(&axis),
            &[168.0, 720.0],
            2,
            (None, None),
        );
        assert_eq!(classes.carried, vec![0]);
        assert!(classes.fixed_post_study.is_empty());
    }

    /// The commissioning gate keys on the DELIVERY stage's continued id: an
    /// `exit_stage_id` that decommissions the plant at a post-study id drops the
    /// later post-study delivery from `carried`, even though its in-study decider
    /// exists (mirrors the solver's delivery-anchored gate).
    #[test]
    fn test_classify_deliveries_commissioning_gate_drops_decommissioned_delivery() {
        let axis = ExtendedDeliveryAxis {
            durations: vec![720.0, 720.0, 720.0, 720.0],
            stage_ids: vec![0, 1, 2, 3],
            n_stages: 2,
        };
        // exit=3 decommissions the plant at continued post-study id 3, so the
        // delivery at delivery stage 3 (post-study index 1) is inactive.
        let classes = classify_deliveries(
            AnticipatedConfig::LeadTime(2500.0),
            Some(&axis),
            &[720.0, 720.0],
            2,
            (None, Some(3)),
        );
        assert!(
            classes.carried.is_empty(),
            "the post-study-index-1 delivery is commissioning-inactive, got: {:?}",
            classes.carried
        );
        assert_eq!(classes.fixed_post_study, vec![0]);
        assert_eq!(classes.commissioning_inactive, vec![1]);
    }

    // ── classify_deliveries: the five-class split and the leading-count mirror ──

    /// A four-study/seven-post-study `LeadTime` decider (`None` for `m in 0..7`,
    /// `Some(t)` with `t < 4` for `m in 7..11`) splits the post-study range into
    /// a pre-study-decided prefix (`fixed_post_study`) and an in-study-decided
    /// suffix (`carried`), with no reach beyond the horizon and no commissioning
    /// exclusion on a windowless plant.
    #[test]
    fn classify_deliveries_splits_fixed_and_carried_post_study_stages() {
        let axis = ExtendedDeliveryAxis {
            durations: vec![720.0; 11],
            stage_ids: (0..11).collect(),
            n_stages: 4,
        };
        let classes = classify_deliveries(
            AnticipatedConfig::LeadTime(5040.0),
            Some(&axis),
            &[720.0; 4],
            4,
            (None, None),
        );
        assert_eq!(classes.leading_in_study, 4);
        assert_eq!(classes.fixed_post_study, vec![0, 1, 2]);
        assert_eq!(classes.carried, vec![3, 4, 5, 6]);
        assert!(classes.beyond_reach.is_empty());
        assert!(classes.commissioning_inactive.is_empty());
    }

    /// A post-study index whose decider resolves in-study but at or beyond
    /// `n_stages` (a post-study decision for a post-study delivery) lands in
    /// `beyond_reach` and none of the other three vectors.
    #[test]
    fn classify_deliveries_records_a_beyond_reach_post_study_stage() {
        let axis = ExtendedDeliveryAxis {
            durations: vec![100.0, 100.0, 100.0],
            stage_ids: vec![0, 1, 2],
            n_stages: 1,
        };
        let classes = classify_deliveries(
            AnticipatedConfig::LeadStages(1),
            Some(&axis),
            &[100.0],
            1,
            (None, None),
        );
        assert_eq!(classes.beyond_reach, vec![1]);
        assert!(!classes.carried.contains(&1));
        assert!(!classes.fixed_post_study.contains(&1));
        assert!(!classes.commissioning_inactive.contains(&1));
    }

    /// An `exit_stage_id` inside the study decommissions the plant for every
    /// later delivery; all three post-study indices land in
    /// `commissioning_inactive` and the other three vectors stay empty.
    #[test]
    fn classify_deliveries_records_a_commissioning_inactive_post_study_stage() {
        let axis = ExtendedDeliveryAxis {
            durations: vec![100.0, 100.0, 100.0, 100.0, 100.0],
            stage_ids: vec![0, 1, 2, 3, 4],
            n_stages: 2,
        };
        let classes = classify_deliveries(
            AnticipatedConfig::LeadStages(1),
            Some(&axis),
            &[100.0, 100.0],
            2,
            (None, Some(1)),
        );
        assert_eq!(classes.commissioning_inactive, vec![0, 1, 2]);
        assert!(classes.fixed_post_study.is_empty());
        assert!(classes.carried.is_empty());
        assert!(classes.beyond_reach.is_empty());
    }

    /// A study with no `post_study_stages` (`axis = None`) reports every
    /// post-study vector empty and resolves `leading_in_study` exactly as
    /// `lead_delivery_stage_count` does for the same plant.
    #[test]
    fn classify_deliveries_without_a_post_study_axis_reports_only_the_leading_count() {
        let study_durations = vec![100.0; 5];
        let classes = classify_deliveries(
            AnticipatedConfig::LeadStages(2),
            None,
            &study_durations,
            5,
            (None, None),
        );
        assert_eq!(
            classes.leading_in_study,
            lead_delivery_stage_count(AnticipatedConfig::LeadStages(2), &study_durations, 5)
        );
        assert!(classes.fixed_post_study.is_empty());
        assert!(classes.carried.is_empty());
        assert!(classes.beyond_reach.is_empty());
        assert!(classes.commissioning_inactive.is_empty());
    }

    /// `leading_in_study` (a cumulative-hours count) matches the `None` run of
    /// `extended_deciders` over the study prefix (a boundary `partition_point`)
    /// — the mirror `classify_deliveries`'s internal `debug_assert_eq!` guards —
    /// for both `LeadStages` and `LeadTime`.
    #[test]
    fn classify_deliveries_leading_count_matches_the_extended_decider_none_run() {
        let axis = ExtendedDeliveryAxis {
            durations: vec![720.0; 5],
            stage_ids: (0..5).collect(),
            n_stages: 3,
        };
        let study_durations = [720.0; 3];

        for mode in [
            AnticipatedConfig::LeadStages(2),
            AnticipatedConfig::LeadTime(1000.0),
        ] {
            let classes = classify_deliveries(mode, Some(&axis), &study_durations, 3, (None, None));
            let none_run = extended_deciders(mode, &axis.durations)[..axis.n_stages]
                .iter()
                .filter(|decider| decider.is_none())
                .count();
            assert_eq!(
                classes.leading_in_study, none_run,
                "leading_in_study must match the extended_deciders None run for {mode:?}"
            );
        }
    }

    /// A study with no `post_study_stages` yields no delivery axis, so nothing
    /// reaches post-study — the `None` return the lead-cap reject keys on.
    #[test]
    fn test_extended_delivery_axis_absent_without_post_study() {
        let data =
            make_data_anticipated(vec![make_anticipated_thermal(1, 2, None, None)], 3, vec![]);
        assert!(build_extended_delivery_axis(&data).is_none());
    }

    /// `build_extended_delivery_axis` concatenates the study durations/ids with
    /// the post-study durations and the continued id sequence
    /// (`max_study_id + 1 ..`).
    #[test]
    fn test_extended_delivery_axis_concatenates_and_continues_ids() {
        let durations = [720.0, 720.0];
        let post_study = post_study_from(&durations, &[696.0, 744.0], &[]);
        let data = data_with_post_study(
            vec![make_anticipated_thermal(1, 1, None, None)],
            &durations,
            commitments_on(1, &[0.0], &durations),
            post_study,
        );
        let Some(axis) = build_extended_delivery_axis(&data) else {
            panic!("post-study declared, so the axis must build");
        };
        assert_eq!(axis.n_stages, 2);
        assert_eq!(axis.durations, vec![720.0, 720.0, 696.0, 744.0]);
        assert_eq!(axis.stage_ids, vec![0, 1, 2, 3]);
    }

    // ── Rule 1 (missing bound cell) and V2 (fixed post-horizon tiling) ────────

    /// Rule 1 failing fixture: an anticipated thermal whose extended lead reaches
    /// post-study stage 0 with no `PostStudyThermalBound(id, 0)` is rejected,
    /// naming the plant and the stage.
    #[test]
    fn test_missing_post_study_bound_cell_rejected() {
        let thermal = make_lead_time_anticipated_thermal(1, 3000.0, None, None);
        let durations = [168.0, 168.0, 168.0, 168.0, 720.0, 720.0];
        // Post-study stage 0 is reached (see the exemption test) but no bound cell
        // is declared for it.
        let post_study = post_study_from(&durations, &[1440.0], &[]);
        let data = data_with_post_study(
            vec![thermal],
            &durations,
            commitments_on(1, &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], &durations),
            post_study,
        );
        let mut ctx = ValidationContext::new();
        check_post_study_stages(&data, &mut ctx);
        let errors = ctx.errors();
        let relevant: Vec<_> = errors
            .iter()
            .filter(|e| {
                e.kind == ErrorKind::BusinessRuleViolation
                    && e.message.contains("no thermal_bounds entry")
            })
            .collect();
        assert_eq!(
            relevant.len(),
            1,
            "expected exactly one missing-bound-cell rejection, got: {errors:?}"
        );
        let msg = &relevant[0].message;
        assert!(
            msg.contains("Thermal 1") && msg.contains("post_study_stage_index 0"),
            "message should name the plant and post-study stage 0, got: {msg}"
        );
    }

    /// Rule 1 passing sibling: adding the `PostStudyThermalBound(1, 0)` cell makes
    /// the reached delivery valid — no missing-bound-cell rejection.
    #[test]
    fn test_present_post_study_bound_cell_accepted() {
        let thermal = make_lead_time_anticipated_thermal(1, 3000.0, None, None);
        let durations = [168.0, 168.0, 168.0, 168.0, 720.0, 720.0];
        let post_study = post_study_from(&durations, &[1440.0], &[(1, 0, 0.0, 500.0)]);
        let data = data_with_post_study(
            vec![thermal],
            &durations,
            commitments_on(1, &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], &durations),
            post_study,
        );
        let mut ctx = ValidationContext::new();
        check_post_study_stages(&data, &mut ctx);
        assert!(
            !ctx.has_errors(),
            "a reached delivery with its bound cell present must not be rejected, got: {:?}",
            ctx.errors()
        );
    }

    /// V2 passing fixture — the retired no-carrier reject's positive inverse: a
    /// `LeadTime` lead so long its FIRST post-study delivery is decided at a
    /// pre-study stage loads cleanly once a commitment window tiles that stage.
    /// `[720,720|720,720]` with `LeadTime(2500)` decides post-study stage 0
    /// pre-study; the third window (reusing `commitments_on` over a 3-stage
    /// duration list) lands exactly on post-study stage 0's dates.
    #[test]
    fn test_pre_study_decided_post_study_delivery_loads_when_tiled() {
        let thermal = make_lead_time_anticipated_thermal(1, 2500.0, None, None);
        let durations = [720.0, 720.0];
        let post_study = post_study_from(
            &durations,
            &[720.0, 720.0],
            &[(1, 0, 0.0, 500.0), (1, 1, 0.0, 500.0)],
        );
        let data = data_with_post_study(
            vec![thermal],
            &durations,
            commitments_on(1, &[0.0, 0.0, 0.0], &[720.0, 720.0, 720.0]),
            post_study,
        );
        let mut ctx = ValidationContext::new();
        check_post_study_stages(&data, &mut ctx);
        assert!(
            !ctx.has_errors(),
            "a pre-study-decided post-study delivery tiled by a commitment window must not be \
             rejected, got: {:?}",
            ctx.errors()
        );
    }

    /// V2 stays silent when `fixed_post_study` is empty: shortening the lead so
    /// every post-study delivery is decided in-study needs no tiling window.
    /// `LeadTime(800)` on the same calendar decides post-study stage 0 at study
    /// stage 0.
    #[test]
    fn test_in_study_decided_post_study_delivery_accepted() {
        let thermal = make_lead_time_anticipated_thermal(1, 800.0, None, None);
        let durations = [720.0, 720.0];
        let post_study = post_study_from(
            &durations,
            &[720.0, 720.0],
            &[(1, 0, 0.0, 500.0), (1, 1, 0.0, 500.0)],
        );
        let data = data_with_post_study(
            vec![thermal],
            &durations,
            commitments_on(1, &[0.0], &durations),
            post_study,
        );
        let mut ctx = ValidationContext::new();
        check_post_study_stages(&data, &mut ctx);
        assert!(
            !ctx.has_errors(),
            "an in-study-decided post-study delivery must not be rejected, got: {:?}",
            ctx.errors()
        );
    }

    /// Invariant (a) is retained: a post-study calendar whose first stage does not
    /// start at the study horizon end is still rejected.
    #[test]
    fn test_post_study_calendar_anchor_still_enforced() {
        let thermal = make_lead_time_anticipated_thermal(1, 3000.0, None, None);
        let durations = [168.0, 168.0, 168.0, 168.0, 720.0, 720.0];
        let mut post_study = post_study_from(&durations, &[1440.0], &[(1, 0, 0.0, 500.0)]);
        // Shift the anchor a day past the study horizon end.
        post_study.stages[0].start_date += TimeDelta::days(1);
        let data = data_with_post_study(
            vec![thermal],
            &durations,
            commitments_on(1, &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], &durations),
            post_study,
        );
        let mut ctx = ValidationContext::new();
        check_post_study_stages(&data, &mut ctx);
        assert!(
            ctx.errors()
                .iter()
                .any(|e| e.message.contains("post-study calendar must begin exactly")),
            "invariant (a) anchor check must still fire, got: {:?}",
            ctx.errors()
        );
    }

    // ── Cadence-transition advisory (check_anticipated_cadence_transition) ────

    /// Given a `lead_stages=2` thermal whose active window spans a
    /// weekly (168h) -> monthly (744h) stage-cadence transition, when
    /// semantic validation runs, then exactly one advisory is produced naming
    /// the thermal and citing `lead_time`, and validation still succeeds.
    #[test]
    fn lead_stages_cadence_transition_emits_advisory() {
        let thermal = make_anticipated_thermal(1, 2, None, None);
        let durations = [168.0, 168.0, 168.0, 744.0, 744.0];
        let data = make_data_anticipated_with_durations(
            vec![thermal],
            &durations,
            commitments_on(1, &[0.0, 0.0], &durations),
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);

        assert!(
            !ctx.has_errors(),
            "a cadence transition is advisory-only, never an error; got: {:?}",
            ctx.errors()
        );
        let relevant: Vec<_> = ctx
            .warnings()
            .into_iter()
            .filter(|w| w.kind == ErrorKind::ModelQuality && w.message.contains("Thermal 1"))
            .collect();
        assert_eq!(
            relevant.len(),
            1,
            "expected exactly one cadence-transition advisory, got: {:?}",
            ctx.warnings()
        );
        let msg = &relevant[0].message;
        assert!(
            msg.contains("lead_time"),
            "advisory must cite lead_time as the physically-anchored alternative, got: {msg}"
        );
        assert!(
            msg.contains("stage 2") && msg.contains("stage 3"),
            "advisory must cite the transition stage pair, got: {msg}"
        );
    }

    /// Given a `lead_stages=2` thermal whose active window spans only
    /// equal-duration (uniform) stages, when semantic validation runs, then no
    /// cadence-transition advisory is produced.
    #[test]
    fn lead_stages_uniform_calendar_no_advisory() {
        let thermal = make_anticipated_thermal(1, 2, None, None);
        let durations = [744.0, 744.0, 744.0, 744.0, 744.0];
        let data = make_data_anticipated_with_durations(
            vec![thermal],
            &durations,
            commitments_on(1, &[0.0, 0.0], &durations),
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);

        assert!(
            !ctx.has_errors(),
            "a uniform calendar is valid, got: {:?}",
            ctx.errors()
        );
        assert!(
            !ctx.warnings()
                .iter()
                .any(|w| w.kind == ErrorKind::ModelQuality && w.message.contains("cadence")),
            "a uniform calendar must not trigger a cadence-transition advisory, got: {:?}",
            ctx.warnings()
        );
    }

    // ── lead_delivery_stage_count: direct unit tests ─────────────────────────

    /// Weekly-then-monthly PMO calendar `[168,168,168,168,720,720]` h with
    /// `LeadTime(720.0)`: the leading count is 4 (`S_{m+1} <= 720` holds for
    /// `m = 0..=3` only; `S_5 = 1392 > 720`).
    #[test]
    fn test_lead_delivery_stage_count_lead_time_pmo_calendar() {
        let durations = [168.0, 168.0, 168.0, 168.0, 720.0, 720.0];
        let count = lead_delivery_stage_count(
            AnticipatedConfig::LeadTime(720.0),
            &durations,
            durations.len(),
        );
        assert_eq!(count, 4);
    }

    /// Uniform 5x720h calendar with `LeadTime(1440.0)`: the count is 2
    /// (`S_{m+1} <= 1440` holds for `m = 0, 1` only; tie-inclusive at `m = 1`).
    #[test]
    fn test_lead_delivery_stage_count_lead_time_uniform_calendar() {
        let durations = [720.0; 5];
        let count = lead_delivery_stage_count(
            AnticipatedConfig::LeadTime(1440.0),
            &durations,
            durations.len(),
        );
        assert_eq!(count, 2);
    }

    /// `LeadStages` ignores durations entirely and clamps to `n_stages`.
    #[test]
    fn test_lead_delivery_stage_count_lead_stages_clamped() {
        let count = lead_delivery_stage_count(AnticipatedConfig::LeadStages(2), &[720.0; 5], 5);
        assert_eq!(count, 2);
    }

    // ── Calendar-derived coverage: LeadTime on a non-uniform calendar ────────

    /// Given a `LeadTime(720.0)` plant on the weekly-then-monthly PMO calendar
    /// `[168,168,168,168,720,720]` h whose four windows tile the leading four
    /// delivery stages exactly, no coverage error fires.
    #[test]
    fn test_anticipated_lead_time_coverage_pmo_calendar() {
        let thermal = make_lead_time_anticipated_thermal(1, 720.0, None, None);
        let durations = [168.0, 168.0, 168.0, 168.0, 720.0, 720.0];
        let data = make_data_anticipated_with_durations(
            vec![thermal],
            &durations,
            commitments_on(1, &[0.0, 0.0, 0.0, 0.0], &durations),
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        let errors = ctx.errors();
        let coverage_errors: Vec<_> = errors
            .iter()
            .filter(|e| {
                e.kind == ErrorKind::BusinessRuleViolation
                    && (e.message.contains("do not tile the leading")
                        || e.message.contains("beyond the leading"))
            })
            .collect();
        assert!(
            coverage_errors.is_empty(),
            "expected no coverage error, got: {errors:?}"
        );
    }

    /// Given the same PMO calendar and `LeadTime(720.0)` plant but windows
    /// tiling only the leading two of the four calendar-derived delivery stages,
    /// exactly one `BusinessRuleViolation` fires naming the uncovered stages.
    #[test]
    fn test_anticipated_lead_time_coverage_pmo_calendar_under_coverage_rejected() {
        let thermal = make_lead_time_anticipated_thermal(1, 720.0, None, None);
        let durations = [168.0, 168.0, 168.0, 168.0, 720.0, 720.0];
        let data = make_data_anticipated_with_durations(
            vec![thermal],
            &durations,
            commitments_on(1, &[0.0, 0.0], &durations),
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        let errors = ctx.errors();
        let coverage_errors: Vec<_> = errors
            .iter()
            .filter(|e| {
                e.kind == ErrorKind::BusinessRuleViolation
                    && e.message.contains("do not tile the leading")
            })
            .collect();
        assert_eq!(
            coverage_errors.len(),
            1,
            "expected exactly one under-coverage error, got: {errors:?}"
        );
        let msg = &coverage_errors[0].message;
        assert!(
            msg.contains('2') && msg.contains('3'),
            "message should name the uncovered study stage ids 2 and 3, got: {msg}"
        );
    }

    // ── AC-7: entry_stage_id window on an anticipated thermal is ACCEPTED ─────

    /// An anticipated thermal that sets `entry_stage_id` is accepted: the window
    /// composes with the K-stage delivery lookahead via the shifted decision gate.
    /// With all-zero seeds the seed-vs-window rule does not fire (a zero seed is
    /// consistent with the `[0, 0]` pin at a dormant stage).
    #[test]
    fn test_entry_window_on_anticipated_thermal_accepted() {
        let thermal = make_anticipated_thermal(1, 3, Some(4), None);
        let data = make_data_anticipated(vec![thermal], 5, commitments(1, &[0.0, 0.0, 0.0]));
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(
            !ctx.has_errors(),
            "an entry window on an anticipated thermal must be accepted, got: {:?}",
            ctx.errors()
        );
        assert!(
            !ctx.errors()
                .iter()
                .any(|e| e.message.contains("not supported on anticipated thermals")),
            "the removed window rejection must not fire"
        );
    }

    // ── AC-8: exit_stage_id window on an anticipated thermal is ACCEPTED ──────

    /// An anticipated thermal that sets `exit_stage_id` is accepted. With all-zero
    /// seeds the seed-vs-window rule does not fire.
    #[test]
    fn test_exit_window_on_anticipated_thermal_accepted() {
        let thermal = make_anticipated_thermal(1, 3, None, Some(2));
        let data = make_data_anticipated(vec![thermal], 5, commitments(1, &[0.0, 0.0, 0.0]));
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(
            !ctx.has_errors(),
            "an exit window on an anticipated thermal must be accepted, got: {:?}",
            ctx.errors()
        );
        assert!(
            !ctx.errors()
                .iter()
                .any(|e| e.message.contains("not supported on anticipated thermals")),
            "the removed window rejection must not fire"
        );
    }

    // ── Seed-vs-window: nonzero seed maturing outside the window is rejected ──

    /// A windowed anticipated thermal whose nonzero seed matures OUTSIDE the
    /// operation window is rejected. K=2, n_stages=5, commissioning window
    /// `[entry=2, exit=4)`, commitment windows `[100.0, 0.0]` over study stages
    /// 0 and 1: the 100 MW window delivers at study stage id 0, before
    /// `entry=2`, so `commissioning_active(2, 4, 0) == false` and the nonzero
    /// rate forces an infeasible `0 == 100` fishing equality. The 0 MW window is
    /// consistent at any stage and emits no error.
    #[test]
    fn test_nonzero_seed_outside_window_rejected() {
        let thermal = make_anticipated_thermal(1, 2, Some(2), Some(4));
        let data = make_data_anticipated(vec![thermal], 5, commitments(1, &[100.0, 0.0]));
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        let errors = ctx.errors();
        let relevant: Vec<_> = errors
            .iter()
            .filter(|e| {
                e.kind == ErrorKind::BusinessRuleViolation
                    && e.message
                        .contains("outside the plant's commissioning window")
            })
            .collect();
        assert_eq!(
            relevant.len(),
            1,
            "expected exactly one seed-vs-window error (stage 0 only), got: {errors:?}"
        );
        let msg = &relevant[0].message;
        assert!(
            msg.contains("study stage id 0") && msg.contains("Thermal 1"),
            "error must identify study stage 0 of Thermal 1, got: {msg}"
        );
    }

    /// A windowed anticipated thermal whose nonzero seeds all mature INSIDE the
    /// window is accepted. K=1, n_stages=5, commissioning window
    /// `[entry=0, exit=5)`, a single 50 MW window over study stage id 0 ∈
    /// `[0, 5)`, so `commissioning_active(0, 5, 0) == true` and no seed-vs-window
    /// error fires.
    #[test]
    fn test_nonzero_seed_inside_window_accepted() {
        let thermal = make_anticipated_thermal(1, 1, Some(0), Some(5));
        let data = make_data_anticipated(vec![thermal], 5, commitments(1, &[50.0]));
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(
            !ctx.errors().iter().any(|e| e
                .message
                .contains("outside the plant's commissioning window")),
            "an in-window nonzero seed must not be rejected, got: {:?}",
            ctx.errors()
        );
    }

    // ── Commissioning window on a PLAIN thermal is accepted (applied, not warned) ─

    /// A non-anticipated thermal that sets a commissioning window is accepted:
    /// the window is applied at the LP fill site (dormant column pinned to
    /// `[0, 0]`), so the anticipated-only rejection must NOT fire and no error
    /// is emitted.
    #[test]
    fn test_window_on_plain_thermal_accepted() {
        let thermal = Thermal {
            entry_stage_id: Some(1),
            exit_stage_id: Some(2),
            ..make_thermal(1, 0.0, 100.0)
        };
        let data = make_data(
            vec![],
            vec![thermal],
            vec![],
            make_stages(vec![0, 1, 2]),
            vec![],
            vec![],
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(
            !ctx.has_errors(),
            "a window on a plain (non-anticipated) thermal must not error, got: {:?}",
            ctx.errors()
        );
        assert!(
            !ctx.errors()
                .iter()
                .any(|e| e.message.contains("not supported on anticipated thermals")),
            "the anticipated-only window rejection must not fire for a plain thermal"
        );
    }

    /// An anticipated thermal with NO commissioning window is accepted: the
    /// rejection is gated on a window being set, so a windowless anticipated
    /// plant passes (regression guard against a blanket reject).
    #[test]
    fn test_anticipated_thermal_without_window_accepted() {
        let thermal = make_anticipated_thermal(1, 2, None, None);
        let data = make_data_anticipated(vec![thermal], 5, commitments(1, &[0.0, 0.0]));
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(
            !ctx.errors()
                .iter()
                .any(|e| e.message.contains("not supported on anticipated thermals")),
            "a windowless anticipated thermal must not be rejected, got: {:?}",
            ctx.errors()
        );
    }

    // ── AC-9: a window value above bounds → bounds error, in-bounds window passes ─

    /// Given windows `[600.0, 200.0]` against a plant with max_generation_mw =
    /// 500.0, the validator emits exactly one bounds-violation error for the
    /// 600 MW window (600.0 > 500.0). The 200 MW window is within [0.0, 500.0]
    /// and passes silently.
    ///
    /// Expected: one BusinessRuleViolation whose message contains
    /// "outside the resolved generation box" (the window is in-study), names
    /// "Thermal 3", and the offending window.
    #[test]
    fn test_committed_value_out_of_bounds_error() {
        let thermal = make_anticipated_thermal(3, 2, None, None); // min=0.0, max=500.0
        let data = make_data_anticipated(vec![thermal], 5, commitments(3, &[600.0, 200.0]));
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(ctx.has_errors());
        let errors = ctx.errors();
        let relevant: Vec<_> = errors
            .iter()
            .filter(|e| {
                e.kind == ErrorKind::BusinessRuleViolation
                    && e.message.contains("outside the resolved generation box")
            })
            .collect();
        assert_eq!(
            relevant.len(),
            1,
            "expected exactly one bounds-violation error (600 MW window only), got: {errors:?}"
        );
        let msg0 = &relevant[0].message;
        assert!(
            msg0.contains("Thermal 3"),
            "message should contain 'Thermal 3', got: {msg0}"
        );
        assert!(
            msg0.contains("2024-01-01"),
            "message should identify the offending window by its start date, got: {msg0}"
        );
        assert!(
            msg0.contains("600"),
            "message should contain the offending value 600, got: {msg0}"
        );
        let file = relevant[0].file.to_string_lossy();
        assert!(
            file.contains("initial_conditions"),
            "file path should reference initial_conditions, got: {file}"
        );
        let entity = relevant[0].entity.as_deref().unwrap_or("");
        assert!(
            entity.contains("thermals[id=3].anticipated_config"),
            "entity should reference the thermal anticipated_config, got: {entity}"
        );
    }

    // ── AC-10: a window value below min_gen → bounds error for that window ────

    /// Given windows `[200.0, 50.0]` against a plant with min_generation_mw =
    /// 100.0 and max_generation_mw = 500.0, the validator emits exactly one
    /// bounds-violation error for the 50 MW window (50.0 < 100.0). The 200 MW
    /// window is within [100.0, 500.0] and passes silently.
    #[test]
    fn test_committed_value_below_min_gen_bounds_error() {
        // Build a thermal with min_mw=100.0.
        let thermal = Thermal {
            anticipated_config: Some(AnticipatedConfig::LeadStages(2)),
            ..make_thermal(5, 100.0, 500.0)
        };
        let data = make_data_anticipated(vec![thermal], 5, commitments(5, &[200.0, 50.0]));
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(ctx.has_errors());
        let errors = ctx.errors();
        let relevant: Vec<_> = errors
            .iter()
            .filter(|e| {
                e.kind == ErrorKind::BusinessRuleViolation
                    && e.message.contains("outside the resolved generation box")
            })
            .collect();
        assert_eq!(
            relevant.len(),
            1,
            "expected exactly one bounds-violation error (50 MW window only), got: {errors:?}"
        );
        let msg1 = &relevant[0].message;
        assert!(
            msg1.contains("Thermal 5"),
            "message should contain 'Thermal 5', got: {msg1}"
        );
        assert!(
            msg1.contains("50"),
            "message should contain the offending value 50, got: {msg1}"
        );
    }

    // ── AC-11: all values zero — passes (within [min_gen, max_gen]) ──────────

    /// Given all-zero window values and min_gen = 0.0, no bounds error is
    /// emitted (0.0 is within [0.0, 400.0]).
    #[test]
    fn test_committed_values_all_zero_ok() {
        let thermal = Thermal {
            anticipated_config: Some(AnticipatedConfig::LeadStages(3)),
            ..make_thermal(7, 0.0, 400.0)
        };
        let data = make_data_anticipated(vec![thermal], 5, commitments(7, &[0.0, 0.0, 0.0]));
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(
            !ctx.has_errors(),
            "expected no errors for all-zero window values, got: {:?}",
            ctx.errors()
        );
    }

    // ── AC-12: boundary — value == min_gen (0.0) is accepted ─────────────────

    /// Given a single window value == 0.0 and min_gen == 0.0, no bounds error is
    /// emitted (the minimum is inclusive).
    #[test]
    fn test_committed_value_zero_accepted() {
        let thermal = Thermal {
            anticipated_config: Some(AnticipatedConfig::LeadStages(1)),
            ..make_thermal(9, 0.0, 400.0)
        };
        let data = make_data_anticipated(vec![thermal], 5, commitments(9, &[0.0]));
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(
            !ctx.has_errors(),
            "zero value must be accepted, got: {:?}",
            ctx.errors()
        );
    }

    // ── AC-13: single value above max_generation_mw → bounds error ───────────

    /// Given a single window value 400.0 against max_generation_mw = 350.0, the
    /// validator emits exactly one bounds-violation error because
    /// 400.0 > 350.0 is outside [100.0, 350.0].
    #[test]
    fn test_committed_value_above_max_bounds_error() {
        let thermal = Thermal {
            anticipated_config: Some(AnticipatedConfig::LeadStages(1)),
            ..make_thermal(11, 100.0, 350.0)
        };
        let data = make_data_anticipated(vec![thermal], 5, commitments(11, &[400.0]));
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(ctx.has_errors());
        let errors = ctx.errors();
        let relevant: Vec<_> = errors
            .iter()
            .filter(|e| {
                e.kind == ErrorKind::BusinessRuleViolation
                    && e.message.contains("outside the resolved generation box")
            })
            .collect();
        assert_eq!(
            relevant.len(),
            1,
            "expected exactly one bounds-violation error, got: {errors:?}"
        );
        let msg = &relevant[0].message;
        assert!(
            msg.contains("Thermal 11"),
            "message should contain 'Thermal 11', got: {msg}"
        );
        assert!(
            msg.contains("400"),
            "message should contain the offending value 400, got: {msg}"
        );
    }

    /// A window value a hair above `max_generation_mw` — within the
    /// relative-with-floor tolerance — must not be rejected (a value generated by
    /// some upstream pipeline may drift a hair past a bound it was meant to sit on).
    #[test]
    fn test_committed_value_within_tolerance_above_max_accepted() {
        let thermal = Thermal {
            anticipated_config: Some(AnticipatedConfig::LeadStages(1)),
            ..make_thermal(13, 100.0, 350.0)
        };
        // 1e-8 above max_generation_mw: an order of magnitude below the 3.5e-7 tolerance.
        let data = make_data_anticipated(vec![thermal], 5, commitments(13, &[350.0 + 1e-8]));
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(
            !ctx.has_errors(),
            "a value within relative tolerance of max_generation_mw must not be rejected, \
             got: {:?}",
            ctx.errors()
        );
    }

    /// A window value a hair below `min_generation_mw` — within the
    /// relative-with-floor tolerance — must not be rejected.
    #[test]
    fn test_committed_value_within_tolerance_below_min_accepted() {
        let thermal = Thermal {
            anticipated_config: Some(AnticipatedConfig::LeadStages(1)),
            ..make_thermal(14, 100.0, 350.0)
        };
        // 1e-8 below min_generation_mw: an order of magnitude below the 1e-7 tolerance.
        let data = make_data_anticipated(vec![thermal], 5, commitments(14, &[100.0 - 1e-8]));
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(
            !ctx.has_errors(),
            "a value within relative tolerance of min_generation_mw must not be rejected, \
             got: {:?}",
            ctx.errors()
        );
    }

    /// A window value beyond the tolerance above `max_generation_mw` (an order of
    /// magnitude larger than the tolerance) is still rejected — the tolerance
    /// must have power, not just admit a hairline drift.
    #[test]
    fn test_committed_value_beyond_tolerance_above_max_still_rejected() {
        let thermal = Thermal {
            anticipated_config: Some(AnticipatedConfig::LeadStages(1)),
            ..make_thermal(15, 100.0, 350.0)
        };
        // 3.5e-6 above max_generation_mw: an order of magnitude above the 3.5e-7 tolerance.
        let data = make_data_anticipated(vec![thermal], 5, commitments(15, &[350.0 + 3.5e-6]));
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        let relevant: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| {
                e.kind == ErrorKind::BusinessRuleViolation
                    && e.message.contains("outside the resolved generation box")
            })
            .collect();
        assert_eq!(
            relevant.len(),
            1,
            "a value beyond tolerance above max_generation_mw must still be rejected, \
             got: {:?}",
            ctx.errors()
        );
    }

    // ── The envelope check is window-agnostic: it applies identically to a ───
    // ── window dated past the study horizon, with no post-study calendar ─────
    // ── declared at all.  ──────────────────────────────────────────────────

    /// A window tiling leading study stage 0 (in bounds) plus a second window
    /// dated a full year past the study horizon — no `post_study_stages` is
    /// declared, so `check_commitment_coverage` never sees it either — carries
    /// an out-of-envelope `value_mw`. The envelope check does not filter by
    /// date: it is rejected exactly as an in-study window would be.
    #[test]
    fn committed_value_bounds_reject_an_out_of_envelope_post_horizon_window() {
        let thermal = Thermal {
            anticipated_config: Some(AnticipatedConfig::LeadStages(1)),
            ..make_thermal(21, 0.0, 200.0)
        };
        let mut records = commitments(21, &[0.0]);
        records.push(AnticipatedCommitmentHistory {
            thermal_id: EntityId::from(21),
            start_date: NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2025, 2, 1).unwrap(),
            value_mw: 350.0,
        });
        let data = make_data_anticipated(vec![thermal], 2, records);
        let mut ctx = ValidationContext::new();
        check_anticipated_thermals(&data, &mut ctx);
        let relevant: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| {
                e.kind == ErrorKind::BusinessRuleViolation
                    && e.message.contains("outside the plant's generation bounds")
            })
            .collect();
        assert_eq!(
            relevant.len(),
            1,
            "expected exactly one envelope violation for the post-horizon window, got: {:?}",
            ctx.errors()
        );
        assert!(
            relevant[0].message.contains("350"),
            "message should contain the offending value 350, got: {}",
            relevant[0].message
        );
    }

    /// The same fixture with the post-horizon window's `value_mw` inside
    /// `[0.0, 200.0]`: no envelope error is emitted.
    #[test]
    fn committed_value_bounds_accept_an_in_envelope_post_horizon_window() {
        let thermal = Thermal {
            anticipated_config: Some(AnticipatedConfig::LeadStages(1)),
            ..make_thermal(22, 0.0, 200.0)
        };
        let mut records = commitments(22, &[0.0]);
        records.push(AnticipatedCommitmentHistory {
            thermal_id: EntityId::from(22),
            start_date: NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2025, 2, 1).unwrap(),
            value_mw: 150.0,
        });
        let data = make_data_anticipated(vec![thermal], 2, records);
        let mut ctx = ValidationContext::new();
        check_anticipated_thermals(&data, &mut ctx);
        assert!(
            !ctx.errors()
                .iter()
                .any(|e| e.message.contains("outside the plant's generation bounds")),
            "an in-envelope post-horizon window must not be rejected, got: {:?}",
            ctx.errors()
        );
    }

    /// The post-horizon window declares an explicit `0.0` while
    /// `min_generation_mw` is also `0.0`: the boundary value is accepted, not
    /// mistaken for a missing declaration.
    #[test]
    fn committed_value_bounds_accept_an_explicit_zero_at_the_lower_bound() {
        let thermal = Thermal {
            anticipated_config: Some(AnticipatedConfig::LeadStages(1)),
            ..make_thermal(23, 0.0, 200.0)
        };
        let mut records = commitments(23, &[100.0]);
        records.push(AnticipatedCommitmentHistory {
            thermal_id: EntityId::from(23),
            start_date: NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2025, 2, 1).unwrap(),
            value_mw: 0.0,
        });
        let data = make_data_anticipated(vec![thermal], 2, records);
        let mut ctx = ValidationContext::new();
        check_anticipated_thermals(&data, &mut ctx);
        assert!(
            !ctx.errors()
                .iter()
                .any(|e| e.message.contains("outside the plant's generation bounds")),
            "an explicit zero at the lower bound on a post-horizon window must not be \
             rejected, got: {:?}",
            ctx.errors()
        );
    }

    // ── delivery_box: in-study values validate against the resolved box ──────

    /// A per-stage `thermal_bounds` override tightens `max_generation_mw` below
    /// the static bound at the covered delivery stage: a value inside the
    /// static bound but above the resolved box is rejected.
    #[test]
    fn delivery_box_override_tightens_max_rejects_over_commitment() {
        let thermal = make_anticipated_thermal(31, 1, None, None); // min=0.0, max=500.0
        let mut data = make_data_anticipated(vec![thermal], 5, commitments(31, &[450.0]));
        data.thermal_bounds = vec![ThermalBoundsRow {
            thermal_id: EntityId::from(31),
            stage_id: 0,
            min_generation_mw: None,
            max_generation_mw: Some(400.0),
            cost_per_mwh: None,
            block_id: None,
        }];
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        let errors = ctx.errors();
        let relevant: Vec<_> = errors
            .iter()
            .filter(|e| {
                e.kind == ErrorKind::BusinessRuleViolation
                    && e.message.contains("outside the resolved generation box")
            })
            .collect();
        assert_eq!(
            relevant.len(),
            1,
            "expected exactly one resolved-box violation, got: {errors:?}"
        );
        let msg = &relevant[0].message;
        assert!(
            msg.contains("delivery stage id 0") && msg.contains("[0, 400]"),
            "message should name delivery stage 0 and the tightened box [0, 400], got: {msg}"
        );
    }

    /// A per-stage override makes the resolved box empty (`min > max`) at a
    /// reachable, commissioning-active delivery stage: rejected regardless of
    /// the committed value.
    #[test]
    fn delivery_box_empty_box_rejected() {
        let thermal = make_anticipated_thermal(32, 1, None, None); // min=0.0, max=500.0
        let mut data = make_data_anticipated(vec![thermal], 5, commitments(32, &[0.0]));
        data.thermal_bounds = vec![ThermalBoundsRow {
            thermal_id: EntityId::from(32),
            stage_id: 0,
            min_generation_mw: Some(600.0),
            max_generation_mw: None,
            cost_per_mwh: None,
            block_id: None,
        }];
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        let errors = ctx.errors();
        let relevant: Vec<_> = errors
            .iter()
            .filter(|e| {
                e.kind == ErrorKind::BusinessRuleViolation
                    && e.message.contains("empty resolved generation box")
            })
            .collect();
        assert_eq!(
            relevant.len(),
            1,
            "expected exactly one empty-box violation, got: {errors:?}"
        );
        assert!(
            relevant[0].message.contains("delivery stage id 0"),
            "message should name delivery stage 0, got: {}",
            relevant[0].message
        );
    }

    /// A nonzero value maturing at a commissioning-dormant covered delivery
    /// stage is reported exactly once — by `check_seed_within_window` — even
    /// though it is also large enough to violate the resolved box; the box
    /// check must skip a dormant stage, never double-report it.
    #[test]
    fn delivery_box_dormant_stage_reported_once_by_seed_window_check() {
        let thermal = make_anticipated_thermal(33, 1, Some(2), None); // dormant at stage 0
        let data = make_data_anticipated(vec![thermal], 5, commitments(33, &[600.0]));
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        let errors = ctx.errors();
        let relevant: Vec<_> = errors
            .iter()
            .filter(|e| e.kind == ErrorKind::BusinessRuleViolation)
            .collect();
        assert_eq!(
            relevant.len(),
            1,
            "a dormant covered stage must be reported exactly once, got: {errors:?}"
        );
        assert!(
            relevant[0]
                .message
                .contains("outside the plant's commissioning window"),
            "the single report must be check_seed_within_window's, got: {}",
            relevant[0].message
        );
        assert!(
            !relevant[0].message.contains("resolved generation box"),
            "the box check must not also report the dormant stage, got: {}",
            relevant[0].message
        );
    }

    /// A value a hair outside a resolved-box boundary — within
    /// `envelope_tolerance` — is accepted, the same as at the static bound.
    #[test]
    fn delivery_box_sub_tolerance_drift_accepted() {
        let thermal = make_anticipated_thermal(34, 1, None, None); // min=0.0, max=500.0
        // 1e-8 above the tightened max: an order of magnitude below the 3e-7 tolerance.
        let mut data = make_data_anticipated(vec![thermal], 5, commitments(34, &[300.0 + 1e-8]));
        data.thermal_bounds = vec![ThermalBoundsRow {
            thermal_id: EntityId::from(34),
            stage_id: 0,
            min_generation_mw: None,
            max_generation_mw: Some(300.0),
            cost_per_mwh: None,
            block_id: None,
        }];
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(
            !ctx.errors()
                .iter()
                .any(|e| e.message.contains("resolved generation box")),
            "a sub-tolerance drift above the resolved box must not be rejected, got: {:?}",
            ctx.errors()
        );
    }

    /// With no per-stage `thermal_bounds` override, the resolved box at every
    /// in-study stage equals the plant's static bounds exactly: an
    /// over-commitment message names the same numbers the static bound would
    /// have, proving the rewrite is byte-identical to the static bound for an
    /// unaffected study.
    #[test]
    fn delivery_box_no_override_matches_static_bounds() {
        let thermal = make_anticipated_thermal(35, 1, None, None); // min=0.0, max=500.0
        let data = make_data_anticipated(vec![thermal], 5, commitments(35, &[600.0]));
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        let errors = ctx.errors();
        let relevant: Vec<_> = errors
            .iter()
            .filter(|e| {
                e.kind == ErrorKind::BusinessRuleViolation
                    && e.message.contains("outside the resolved generation box")
            })
            .collect();
        assert_eq!(
            relevant.len(),
            1,
            "expected exactly one resolved-box violation, got: {errors:?}"
        );
        assert!(
            relevant[0].message.contains("[0, 500]"),
            "with no override the resolved box must equal the static bounds [0, 500], got: {}",
            relevant[0].message
        );
    }

    // ── AC-14: non-zero in-bounds window value K=1 — accepted ────────────────

    /// K=1 case: a single window value 204.5647 against `max_generation_mw =
    /// 350.0`. The value lies within [0.0, 350.0], so the bounds check passes
    /// and the validator emits no errors.
    #[test]
    fn test_nonzero_value_in_bounds_accepted_k1() {
        let thermal = Thermal {
            anticipated_config: Some(AnticipatedConfig::LeadStages(1)),
            ..make_thermal(2, 0.0, 350.0)
        };
        let data = make_data_anticipated(vec![thermal], 5, commitments(2, &[204.5647]));
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(
            ctx.errors().is_empty(),
            "expected no errors for an in-bounds window value, got: {:?}",
            ctx.errors()
        );
    }

    /// K=2 acceptance: two non-zero windows, both within
    /// `[min_generation_mw, max_generation_mw]`, are accepted with zero errors.
    /// The pre-horizon-seeding contract delivers each window's rate over its
    /// covered study stage as a sunk-cost commitment; in-bounds seeds carry no
    /// validator errors.
    #[test]
    fn test_k2_two_in_bounds_nonzero_values_accepted() {
        let thermal = make_anticipated_thermal(5, 2, None, None); // min=0.0, max=500.0
        let data = make_data_anticipated(vec![thermal], 5, commitments(5, &[50.0, 30.0]));
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(
            ctx.errors().is_empty(),
            "expected no errors for in-bounds K=2 windows, got: {:?}",
            ctx.errors()
        );
    }

    /// Regression lock: `check_committed_value_bounds` must not emit any
    /// `SemanticAmbiguity` warning for an in-bounds non-zero seed — guards against
    /// reintroducing a same-dispatch-as-zero-seed warning.
    #[test]
    fn test_nonzero_in_bounds_seed_emits_no_semantic_ambiguity_warning() {
        let thermal = Thermal {
            anticipated_config: Some(AnticipatedConfig::LeadStages(2)),
            ..make_thermal(3, 0.0, 350.0)
        };
        let data = make_data_anticipated(vec![thermal], 5, commitments(3, &[100.0, 200.0]));
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        let all_warnings = ctx.warnings();
        let ambiguity_warnings: Vec<_> = all_warnings
            .iter()
            .filter(|w| {
                w.kind == ErrorKind::SemanticAmbiguity
                    && w.file.to_string_lossy().contains("initial_conditions.json")
            })
            .collect();
        assert!(
            ambiguity_warnings.is_empty(),
            "expected no SemanticAmbiguity warning from initial_conditions.json \
             for an in-bounds non-zero seed, got: {ambiguity_warnings:?}"
        );
    }

    /// K=2 acceptance: mixed in-bounds and out-of-bounds windows. Only the
    /// out-of-bounds window produces an error; the in-bounds window (a zero
    /// value) passes silently. Verifies that the bounds check is per-window, not
    /// per-history.
    #[test]
    fn test_k2_mixed_in_bounds_and_out_of_bounds_only_oob_reported() {
        let thermal = make_anticipated_thermal(7, 2, None, None); // min=0.0, max=500.0
        let data = make_data_anticipated(vec![thermal], 5, commitments(7, &[0.0, 600.0]));
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        let errors = ctx.errors();
        let relevant: Vec<_> = errors
            .iter()
            .filter(|e| {
                e.kind == ErrorKind::BusinessRuleViolation
                    && e.message.contains("outside the resolved generation box")
            })
            .collect();
        assert_eq!(
            relevant.len(),
            1,
            "expected exactly one bounds error (600 MW window only), got: {errors:?}"
        );
        assert!(
            relevant[0].message.contains("600"),
            "error must contain value 600, got: {}",
            relevant[0].message
        );
    }

    // ── Bounds test ───────────────────────────────────────────────────────────

    /// min_generation_mw > max_generation_mw produces InvalidValue with "Thermal <id>".
    #[test]
    fn test_thermal_generation_min_greater_than_max() {
        let thermal = make_thermal(10, 500.0, 100.0); // min > max — violation
        let data = make_data(
            vec![],
            vec![thermal],
            vec![],
            make_stages(vec![0]),
            vec![],
            vec![],
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(ctx.has_errors());
        let errors = ctx.errors();
        let relevant: Vec<_> = errors
            .iter()
            .filter(|e| e.kind == ErrorKind::InvalidValue)
            .collect();
        assert_eq!(relevant.len(), 1, "exactly 1 InvalidValue error expected");
        let msg = &relevant[0].message;
        assert!(
            msg.contains("Thermal 10"),
            "message should contain 'Thermal 10', got: {msg}"
        );
    }

    /// min_generation_mw == max_generation_mw produces no error.
    #[test]
    fn test_thermal_generation_equal_bounds_valid() {
        let thermal = make_thermal(11, 200.0, 200.0);
        let data = make_data(
            vec![],
            vec![thermal],
            vec![],
            make_stages(vec![0]),
            vec![],
            vec![],
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(!ctx.has_errors());
    }

    // ── AD-1: anticipated_decision on non-anticipated thermal → hard error ─────

    /// AD-1: A generic constraint that references `anticipated_decision(N)` where
    /// thermal `N` is NOT anticipated produces a `BusinessRuleViolation` naming
    /// the constraint and the thermal ID.
    #[test]
    fn test_anticipated_decision_on_non_anticipated_thermal_error() {
        use cobre_core::{
            ConstraintExpression, GenericConstraint, LinearTerm, SlackConfig, VariableRef,
        };

        let thermal = make_thermal(7, 0.0, 500.0); // NOT anticipated
        let constraint = GenericConstraint {
            id: EntityId::from(1),
            name: "bad_constraint".to_string(),
            description: None,
            expression: ConstraintExpression {
                terms: vec![LinearTerm::literal(
                    1.0,
                    VariableRef::AnticipatedDecision {
                        thermal_id: EntityId::from(7),
                    },
                )],
            },
            slack: SlackConfig {
                enabled: false,
                penalty: None,
            },
            bound_lower_affine: None,
            bound_upper_affine: None,
        };
        let stage_ids: Vec<i32> = (0..5).collect();
        let mut data = make_data(
            vec![],
            vec![thermal],
            vec![],
            make_stages(stage_ids),
            vec![],
            vec![],
        );
        data.generic_constraints = vec![constraint];

        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(
            ctx.has_errors(),
            "expected error for non-anticipated thermal"
        );
        let errors = ctx.errors();
        let relevant: Vec<_> = errors
            .iter()
            .filter(|e| e.kind == ErrorKind::BusinessRuleViolation)
            .collect();
        assert!(
            !relevant.is_empty(),
            "expected BusinessRuleViolation, got: {errors:?}"
        );
        let msg = &relevant[0].message;
        assert!(
            msg.contains("bad_constraint"),
            "message should contain constraint name, got: {msg}"
        );
        assert!(
            msg.contains('7'),
            "message should contain thermal id 7, got: {msg}"
        );
        assert!(
            msg.contains("not an anticipated thermal"),
            "message should explain the rule, got: {msg}"
        );
        let file = relevant[0].file.to_string_lossy();
        assert!(
            file.contains("generic_constraints.json"),
            "file should reference generic_constraints.json, got: {file}"
        );
    }

    // ── AD-2: anticipated_decision on anticipated thermal → no error ───────────

    /// AD-2: A generic constraint that references `anticipated_decision(N)` where
    /// thermal `N` IS anticipated produces no `BusinessRuleViolation` from the
    /// new validator.
    #[test]
    fn test_anticipated_decision_on_anticipated_thermal_ok() {
        use cobre_core::{
            ConstraintExpression, GenericConstraint, LinearTerm, SlackConfig, VariableRef,
            entities::AnticipatedConfig,
        };

        let thermal = Thermal {
            anticipated_config: Some(AnticipatedConfig::LeadStages(2)),
            ..make_thermal(3, 0.0, 500.0)
        };
        let constraint = GenericConstraint {
            id: EntityId::from(10),
            name: "valid_anticipated_constraint".to_string(),
            description: None,
            expression: ConstraintExpression {
                terms: vec![LinearTerm::literal(
                    1.0,
                    VariableRef::AnticipatedDecision {
                        thermal_id: EntityId::from(3),
                    },
                )],
            },
            slack: SlackConfig {
                enabled: false,
                penalty: None,
            },
            bound_lower_affine: None,
            bound_upper_affine: None,
        };
        let mut data = make_data_anticipated(vec![thermal], 5, commitments(3, &[0.0, 0.0]));
        data.generic_constraints = vec![constraint];

        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        let errors = ctx.errors();
        let relevant: Vec<_> = errors
            .iter()
            .filter(|e| {
                e.kind == ErrorKind::BusinessRuleViolation
                    && e.file
                        .to_string_lossy()
                        .contains("generic_constraints.json")
            })
            .collect();
        assert!(
            relevant.is_empty(),
            "anticipated_decision on an anticipated thermal must not produce a BusinessRuleViolation, got: {relevant:?}"
        );
    }

    // ── TG-1: thermal_generation on anticipated thermal → SemanticAmbiguity ───

    /// TG-1: A generic constraint that references `thermal_generation(N)` where
    /// thermal `N` IS anticipated produces a `SemanticAmbiguity` warning naming
    /// the constraint and the thermal ID with a hint to use `anticipated_decision`.
    #[test]
    fn test_thermal_generation_on_anticipated_thermal_warns() {
        use cobre_core::{
            ConstraintExpression, GenericConstraint, LinearTerm, SlackConfig, VariableRef,
            entities::AnticipatedConfig,
        };

        let thermal = Thermal {
            anticipated_config: Some(AnticipatedConfig::LeadStages(1)),
            ..make_thermal(5, 0.0, 300.0)
        };
        let constraint = GenericConstraint {
            id: EntityId::from(20),
            name: "ambiguous_constraint".to_string(),
            description: None,
            expression: ConstraintExpression {
                terms: vec![LinearTerm::literal(
                    1.0,
                    VariableRef::ThermalGeneration {
                        thermal_id: EntityId::from(5),
                        block_id: None,
                    },
                )],
            },
            slack: SlackConfig {
                enabled: false,
                penalty: None,
            },
            bound_lower_affine: None,
            bound_upper_affine: None,
        };
        let mut data = make_data_anticipated(vec![thermal], 5, commitments(5, &[0.0]));
        data.generic_constraints = vec![constraint];

        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);

        // No hard errors from the new validator.
        let errors = ctx.errors();
        let hard: Vec<_> = errors
            .iter()
            .filter(|e| {
                e.file
                    .to_string_lossy()
                    .contains("generic_constraints.json")
            })
            .collect();
        assert!(
            hard.is_empty(),
            "thermal_generation on anticipated thermal must not produce a hard error, got: {hard:?}"
        );

        // Exactly one SemanticAmbiguity warning.
        let warnings = ctx.warnings();
        let relevant: Vec<_> = warnings
            .iter()
            .filter(|w| w.kind == ErrorKind::SemanticAmbiguity)
            .collect();
        assert_eq!(
            relevant.len(),
            1,
            "expected exactly one SemanticAmbiguity warning, got: {warnings:?}"
        );
        let msg = &relevant[0].message;
        assert!(
            msg.contains("ambiguous_constraint"),
            "warning should name the constraint, got: {msg}"
        );
        assert!(
            msg.contains('5'),
            "warning should mention thermal id 5, got: {msg}"
        );
        assert!(
            msg.contains("anticipated_decision"),
            "warning should suggest anticipated_decision, got: {msg}"
        );
        let file = relevant[0].file.to_string_lossy();
        assert!(
            file.contains("generic_constraints.json"),
            "file should reference generic_constraints.json, got: {file}"
        );
    }

    // ── TG-2: thermal_generation on non-anticipated thermal → no warning ──────

    /// TG-2: A generic constraint that references `thermal_generation(N)` where
    /// thermal `N` is NOT anticipated produces no `SemanticAmbiguity` warning.
    #[test]
    fn test_thermal_generation_on_non_anticipated_thermal_no_warn() {
        use cobre_core::{
            ConstraintExpression, GenericConstraint, LinearTerm, SlackConfig, VariableRef,
        };

        let thermal = make_thermal(9, 0.0, 200.0); // NOT anticipated
        let constraint = GenericConstraint {
            id: EntityId::from(30),
            name: "plain_thermal_constraint".to_string(),
            description: None,
            expression: ConstraintExpression {
                terms: vec![LinearTerm::literal(
                    1.0,
                    VariableRef::ThermalGeneration {
                        thermal_id: EntityId::from(9),
                        block_id: None,
                    },
                )],
            },
            slack: SlackConfig {
                enabled: false,
                penalty: None,
            },
            bound_lower_affine: None,
            bound_upper_affine: None,
        };
        let stage_ids: Vec<i32> = (0..5).collect();
        let mut data = make_data(
            vec![],
            vec![thermal],
            vec![],
            make_stages(stage_ids),
            vec![],
            vec![],
        );
        data.generic_constraints = vec![constraint];

        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);

        let warnings = ctx.warnings();
        let relevant: Vec<_> = warnings
            .iter()
            .filter(|w| w.kind == ErrorKind::SemanticAmbiguity)
            .collect();
        assert!(
            relevant.is_empty(),
            "thermal_generation on a non-anticipated thermal must not emit SemanticAmbiguity, got: {relevant:?}"
        );
    }
}
