//! Layer 5a — hydro-domain semantic validation.

use std::collections::{HashMap, HashSet};

use cobre_core::{EntityId, Hydro};

use super::super::{ValidationContext, rules, schema::ParsedData};
use super::envelope_tolerance;

/// Rule 1: the hydro cascade graph must be acyclic.
pub(super) fn check_cascade_acyclic(data: &ParsedData, ctx: &mut ValidationContext) {
    if data.hydros.is_empty() {
        return;
    }

    let all_ids: Vec<i32> = data.hydros.iter().map(|h| h.id.0).collect();
    let downstream_set: HashSet<i32> = all_ids.iter().copied().collect();

    let mut adjacency: HashMap<i32, Vec<i32>> =
        all_ids.iter().copied().map(|id| (id, Vec::new())).collect();
    let mut in_degree: HashMap<i32, usize> = all_ids.iter().copied().map(|id| (id, 0)).collect();
    for hydro in &data.hydros {
        if let Some(ds) = hydro.downstream_id
            && downstream_set.contains(&ds.0)
        {
            adjacency.entry(hydro.id.0).or_default().push(ds.0);
            *in_degree.entry(ds.0).or_insert(0) += 1;
        }
    }

    let mut queue: std::collections::VecDeque<i32> = in_degree
        .iter()
        .filter(|&(_, deg)| *deg == 0)
        .map(|(&id, _)| id)
        .collect();

    let mut visited_count: usize = 0;

    while let Some(node) = queue.pop_front() {
        visited_count += 1;
        if let Some(neighbors) = adjacency.get(&node) {
            for &neighbor in neighbors {
                let deg = in_degree.entry(neighbor).or_insert(0);
                if *deg > 0 {
                    *deg -= 1;
                }
                if *deg == 0 {
                    queue.push_back(neighbor);
                }
            }
        }
    }

    if visited_count < all_ids.len() {
        let mut cycle_participants: Vec<i32> = in_degree
            .iter()
            .filter(|&(_, deg)| *deg > 0)
            .map(|(&id, _)| id)
            .collect();
        cycle_participants.sort_unstable();

        ctx.emit(
            &rules::SEMANTIC_HYDRO_CASCADE_CYCLE,
            "system/hydros.json",
            None::<&str>,
            format!(
                "hydro cascade contains a cycle involving hydro IDs: [{}]",
                cycle_participants
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        );
    }
}

/// Rules 2-5: `min <= max` for a hydro's storage, turbined, outflow (when
/// `Some`), and generation bounds.
pub(super) fn check_hydro_bounds(data: &ParsedData, ctx: &mut ValidationContext) {
    for hydro in &data.hydros {
        let entity_str = format!("Hydro {}", hydro.id.0);

        if hydro.min_storage_hm3 > hydro.max_storage_hm3 {
            ctx.emit(
                &rules::SEMANTIC_HYDRO_STORAGE_BOUNDS_INVERTED,
                "system/hydros.json",
                Some(&entity_str),
                format!(
                    "{entity_str}: min_storage_hm3 ({}) > max_storage_hm3 ({}); storage bounds are inconsistent",
                    hydro.min_storage_hm3, hydro.max_storage_hm3
                ),
            );
        }

        if hydro.min_turbined_m3s > hydro.max_turbined_m3s {
            ctx.emit(
                &rules::SEMANTIC_HYDRO_TURBINED_BOUNDS_INVERTED,
                "system/hydros.json",
                Some(&entity_str),
                format!(
                    "{entity_str}: min_turbined_m3s ({}) > max_turbined_m3s ({}); turbine bounds are inconsistent",
                    hydro.min_turbined_m3s, hydro.max_turbined_m3s
                ),
            );
        }

        if let Some(max_outflow) = hydro.max_outflow_m3s
            && hydro.min_outflow_m3s > max_outflow
        {
            ctx.emit(
                &rules::SEMANTIC_HYDRO_OUTFLOW_BOUNDS_INVERTED,
                "system/hydros.json",
                Some(&entity_str),
                format!(
                    "{entity_str}: min_outflow_m3s ({}) > max_outflow_m3s ({}); outflow bounds are inconsistent",
                    hydro.min_outflow_m3s, max_outflow
                ),
            );
        }

        if hydro.min_generation_mw > hydro.max_generation_mw {
            ctx.emit(
                &rules::SEMANTIC_HYDRO_GENERATION_BOUNDS_INVERTED,
                "system/hydros.json",
                Some(&entity_str),
                format!(
                    "{entity_str}: min_generation_mw ({}) > max_generation_mw ({}); generation bounds are inconsistent",
                    hydro.min_generation_mw, hydro.max_generation_mw
                ),
            );
        }
    }
}

/// Rule 46: rejects a `min_diversion_m3s` override on a hydro that declares no `diversion`
/// channel: with no channel, [`resolve_bounds`](crate::resolution::resolve_bounds)
/// pins the diversion column to `[0, 0]`, so a positive floor is the infeasible
/// `[min > 0, 0]`.
///
/// This is a same-column, override-vs-declaration check only. Cross-row and
/// cross-source min/max inversion — an override floor exceeding a maximum
/// declared elsewhere, whether from a differently-keyed override row or from the
/// entity's own `DiversionChannel.max_flow_m3s` — is deliberately not validated
/// here: a combinatorial checker re-implementing the resolver's precedence would
/// duplicate it and drift. That residual surfaces as LP infeasibility instead.
pub(super) fn check_diversion_floor_requires_channel(
    data: &ParsedData,
    ctx: &mut ValidationContext,
) {
    let declared: HashMap<EntityId, &Hydro> = data.hydros.iter().map(|h| (h.id, h)).collect();

    for row in &data.hydro_bounds {
        let Some(min_diversion) = row.min_diversion_m3s.filter(|&m| m > 0.0) else {
            continue;
        };
        let Some(&hydro) = declared.get(&row.hydro_id) else {
            continue;
        };
        if hydro.diversion.is_some() {
            continue;
        }

        let entity_str = format!("Hydro {}", hydro.id.0);
        ctx.emit(
            &rules::SEMANTIC_HYDRO_DIVERSION_FLOOR_WITHOUT_CHANNEL,
            "constraints/hydro_bounds.parquet",
            Some(&entity_str),
            format!(
                "{entity_str}: hydro_bounds row at stage_id={} sets min_diversion_m3s=\
                 {min_diversion}, but the hydro declares no diversion channel; diversion is \
                 pinned [0, 0] with no channel, making a positive floor infeasible",
                row.stage_id
            ),
        );
    }
}

fn check_entry_precedes_exit(
    file: &str,
    entity_kind: &str,
    id: i32,
    entry: Option<i32>,
    exit: Option<i32>,
    ctx: &mut ValidationContext,
) {
    if let (Some(entry), Some(exit)) = (entry, exit)
        && entry >= exit
    {
        let entity_str = format!("{entity_kind} {id}");
        ctx.emit(
            &rules::LIFECYCLE_ENTRY_NOT_BEFORE_EXIT,
            file,
            Some(&entity_str),
            format!(
                "{entity_str}: entry_stage_id ({entry}) >= exit_stage_id ({exit}); entry must precede exit"
            ),
        );
    }
}

/// Rule 6: `entry_stage_id < exit_stage_id` (when both `Some`), for hydros,
/// lines, and thermals.
pub(super) fn check_lifecycle_consistency(data: &ParsedData, ctx: &mut ValidationContext) {
    for hydro in &data.hydros {
        check_entry_precedes_exit(
            "system/hydros.json",
            "Hydro",
            hydro.id.0,
            hydro.entry_stage_id,
            hydro.exit_stage_id,
            ctx,
        );
    }

    for line in &data.lines {
        check_entry_precedes_exit(
            "system/lines.json",
            "Line",
            line.id.0,
            line.entry_stage_id,
            line.exit_stage_id,
            ctx,
        );
    }

    for thermal in &data.thermals {
        check_entry_precedes_exit(
            "system/thermals.json",
            "Thermal",
            thermal.id.0,
            thermal.entry_stage_id,
            thermal.exit_stage_id,
            ctx,
        );
    }
}

/// Rule 6 (remaining entity types): extends the `entry < exit` ordering check
/// in [`check_lifecycle_consistency`] to the entity types it does not cover:
/// pumping stations, non-controllable sources, and energy contracts. An
/// unchecked window-bearing entity would pass validation while the others
/// reject `entry >= exit` — the parity this closes.
pub(super) fn check_lifecycle_consistency_remaining(
    data: &ParsedData,
    ctx: &mut ValidationContext,
) {
    for station in &data.pumping_stations {
        check_entry_precedes_exit(
            "system/pumping_stations.json",
            "PumpingStation",
            station.id.0,
            station.entry_stage_id,
            station.exit_stage_id,
            ctx,
        );
    }

    for source in &data.non_controllable_sources {
        check_entry_precedes_exit(
            "system/non_controllable_sources.json",
            "NonControllableSource",
            source.id.0,
            source.entry_stage_id,
            source.exit_stage_id,
            ctx,
        );
    }

    for contract in &data.energy_contracts {
        check_entry_precedes_exit(
            "system/energy_contracts.json",
            "EnergyContract",
            contract.id.0,
            contract.entry_stage_id,
            contract.exit_stage_id,
            ctx,
        );
    }
}

/// Rule 7: a filling hydro's `filling.start_stage_id` must be a declared
/// study stage.
pub(super) fn check_filling_config(data: &ParsedData, ctx: &mut ValidationContext) {
    let study_stage_ids: HashSet<i32> = data
        .stages
        .stages
        .iter()
        .filter(|s| s.id >= 0)
        .map(|s| s.id)
        .collect();

    for hydro in &data.hydros {
        if let Some(filling) = &hydro.filling
            && !study_stage_ids.contains(&filling.start_stage_id)
        {
            let entity_str = format!("Hydro {}", hydro.id.0);
            ctx.emit(
                &rules::SEMANTIC_FILLING_START_STAGE_UNKNOWN,
                "system/hydros.json",
                Some(&entity_str),
                format!(
                    "{entity_str}: filling.start_stage_id ({}) is not a valid study stage ID",
                    filling.start_stage_id
                ),
            );
        }
    }
}

/// Rules 7a-7b: enforces the structural guards a filling hydro must satisfy beyond the
/// start-stage-validity check in [`check_filling_config`]. Each rejects an
/// ill-formed combination that would otherwise yield a meaningless or infeasible
/// `PreFilling`/`Filling`/`Operating` lifecycle, except guard 3, which warns:
///
/// 1. `filling.is_some()` ⟹ `entry_stage_id.is_some()` — a filling config needs an
///    entry to fill toward. The converse does NOT hold: a bare `entry_stage_id`
///    with no `filling` is a valid non-filling commissioning window. The forbidden
///    alternative — `filling` without an entry — leaves the reservoir filling toward
///    nothing.
/// 2. `start_stage_id < entry_stage_id` — else the `Filling` phase is empty.
/// 3. `entry_stage_id >= horizon` (study stage count) warns rather than
///    rejects: the plant fills throughout and never operates within this study
///    (a longer study reuses the same system file). It must still load.
/// 4. the `filling_storage` seed lies in `[0, min_storage_hm3)` — strictly below
///    the dead volume. Equality with `min_storage_hm3` belongs to neither the
///    filling range nor the operating `.storage` range `[min_storage,
///    max_storage]`, so it is rejected.
/// 5. no `exit_stage_id` — a filling hydro is entry-only; an exit is ill-posed
///    for a state-carrying reservoir.
/// 6. `start_stage_id > 0` (a `PreFilling` phase exists) requires the seed `== 0`
///    (empty pit): `PreFilling` freezes storage at the seed before the dam
///    exists, so a nonzero seed asserts impounded water in a reservoir not yet
///    built. A nonzero seed is valid only mid-filling (`start_stage_id == 0`).
pub(super) fn check_filling_guards(data: &ParsedData, ctx: &mut ValidationContext) {
    let horizon =
        i32::try_from(data.stages.stages.iter().filter(|s| s.id >= 0).count()).unwrap_or(i32::MAX);

    for hydro in &data.hydros {
        let entity_str = format!("Hydro {}", hydro.id.0);

        if hydro.filling.is_some() && hydro.entry_stage_id.is_none() {
            ctx.emit(
                &rules::SEMANTIC_FILLING_GUARD_VIOLATED,
                "system/hydros.json",
                Some(&entity_str),
                format!(
                    "{entity_str}: filling is set but entry_stage_id is absent; a filling config \
                     requires an entry_stage_id to fill toward (a bare entry_stage_id without \
                     filling is a valid non-filling commissioning window)"
                ),
            );
        }

        if let Some(filling) = &hydro.filling {
            let seed = data
                .initial_conditions
                .filling_storage
                .iter()
                .find(|s| s.hydro_id == hydro.id);

            if let Some(entry) = hydro.entry_stage_id
                && filling.start_stage_id >= entry
            {
                ctx.emit(
                    &rules::SEMANTIC_FILLING_GUARD_VIOLATED,
                    "system/hydros.json",
                    Some(&entity_str),
                    format!(
                        "{entity_str}: filling.start_stage_id ({}) must be less than \
                         entry_stage_id ({entry}); the filling phase must precede operation",
                        filling.start_stage_id
                    ),
                );
            }

            if let Some(entry) = hydro.entry_stage_id
                && entry >= horizon
            {
                ctx.emit(
                    &rules::SEMANTIC_FILLING_NEVER_OPERATES,
                    "system/hydros.json",
                    Some(&entity_str),
                    format!(
                        "{entity_str}: entry_stage_id ({entry}) is at or beyond the study \
                         horizon ({horizon}); the hydro fills throughout and never operates \
                         within this study"
                    ),
                );
            }

            if let Some(seed) = seed
                && !(seed.value_hm3 >= 0.0 && seed.value_hm3 < hydro.min_storage_hm3)
            {
                ctx.emit(
                    &rules::SEMANTIC_FILLING_GUARD_VIOLATED,
                    "system/hydros.json",
                    Some(&entity_str),
                    format!(
                        "{entity_str}: filling_storage seed ({}) must lie in \
                         [0, min_storage_hm3) = [0, {}); the seed must be strictly below \
                         the dead volume",
                        seed.value_hm3, hydro.min_storage_hm3
                    ),
                );
            }

            if let Some(exit) = hydro.exit_stage_id {
                ctx.emit(
                    &rules::SEMANTIC_FILLING_GUARD_VIOLATED,
                    "system/hydros.json",
                    Some(&entity_str),
                    format!(
                        "{entity_str}: exit_stage_id ({exit}) is set on a filling hydro; \
                         a filling hydro is entry-only and exit is ill-posed for a \
                         state-carrying reservoir"
                    ),
                );
            }

            if filling.start_stage_id > 0
                && let Some(seed) = seed
                && seed.value_hm3 != 0.0
            {
                ctx.emit(
                    &rules::SEMANTIC_FILLING_GUARD_VIOLATED,
                    "system/hydros.json",
                    Some(&entity_str),
                    format!(
                        "{entity_str}: filling_storage seed ({}) must be 0 (empty pit) \
                         when start_stage_id ({}) > 0; a PreFilling phase freezes storage \
                         at the seed before the dam exists",
                        seed.value_hm3, filling.start_stage_id
                    ),
                );
            }
        }
    }
}

/// Rules 8-10: `hydro_geometry.parquet`'s `volume_hm3` is strictly increasing,
/// and `height_m`/`area_km2` are non-decreasing (within a relative tolerance),
/// per hydro.
pub(super) fn check_geometry_monotonicity(data: &ParsedData, ctx: &mut ValidationContext) {
    if data.hydro_geometry.is_empty() {
        return;
    }

    let mut i = 0;
    let rows = &data.hydro_geometry;

    while i < rows.len() {
        let current_hydro_id = rows[i].hydro_id.0;
        let group_start = i;

        // Rows are sorted by hydro_id then volume_hm3 — group detection relies on it.
        while i < rows.len() && rows[i].hydro_id.0 == current_hydro_id {
            i += 1;
        }
        let group = &rows[group_start..i];
        let entity_str = format!("Hydro {current_hydro_id}");

        for pair in group.windows(2) {
            let prev = &pair[0];
            let curr = &pair[1];

            // `volume_hm3` stays a raw strict comparison, deliberately not
            // tolerance-shifted like `height_m`/`area_km2` below: a duplicate
            // volume between two distinct rows is a real data error (a
            // vertical, non-function segment of the V-H curve), never a
            // round-off artifact — these are direct parquet field reads with
            // no arithmetic upstream, so there is no computed-vs-computed
            // residual for a tolerance to absorb.
            if curr.volume_hm3 <= prev.volume_hm3 {
                ctx.emit(
                    &rules::GEOMETRY_VOLUME_NOT_INCREASING,
                    "system/hydro_geometry.parquet",
                    Some(&entity_str),
                    format!(
                        "{entity_str}: volume_hm3 values are not strictly increasing ({} then {}); geometry curve must have strictly increasing volume",
                        prev.volume_hm3, curr.volume_hm3
                    ),
                );
            }

            let height_tolerance = envelope_tolerance(prev.height_m);
            if curr.height_m < prev.height_m - height_tolerance {
                ctx.emit(
                    &rules::GEOMETRY_HEIGHT_DECREASING,
                    "system/hydro_geometry.parquet",
                    Some(&entity_str),
                    format!(
                        "{entity_str}: height_m values are not non-decreasing ({} then {}); geometry curve must have non-decreasing height with volume",
                        prev.height_m, curr.height_m
                    ),
                );
            }

            let area_tolerance = envelope_tolerance(prev.area_km2);
            if curr.area_km2 < prev.area_km2 - area_tolerance {
                ctx.emit(
                    &rules::GEOMETRY_AREA_DECREASING,
                    "system/hydro_geometry.parquet",
                    Some(&entity_str),
                    format!(
                        "{entity_str}: area_km2 values are not non-decreasing ({} then {}); geometry curve must have non-decreasing area with volume",
                        prev.area_km2, curr.area_km2
                    ),
                );
            }
        }
    }
}

/// Rule 50: hydros with `evaporation_coefficients_mm` require geometry rows in
/// `hydro_geometry.parquet` (area-volume curve for linearization).
pub(super) fn check_evaporation_geometry_coverage(data: &ParsedData, ctx: &mut ValidationContext) {
    let geometry_hydro_ids: HashSet<i32> =
        data.hydro_geometry.iter().map(|r| r.hydro_id.0).collect();

    for hydro in &data.hydros {
        if hydro.evaporation_coefficients_mm.is_some() && !geometry_hydro_ids.contains(&hydro.id.0)
        {
            ctx.emit(
                &rules::SEMANTIC_HYDRO_EVAPORATION_WITHOUT_GEOMETRY,
                "system/hydros.json",
                Some(format!("Hydro {} (id={})", hydro.name, hydro.id.0)),
                format!(
                    "hydro {} (id={}) has evaporation_coefficients_mm but no geometry data \
                     in hydro_geometry.parquet; evaporation linearization requires \
                     area-volume curve data",
                    hydro.name, hydro.id.0
                ),
            );
        }
    }
}

/// Rules 11-12: every (hydro, stage) has at least 1 FPHA plane, and each
/// plane's `gamma_v >= 0` / `gamma_s <= 0`.
pub(super) fn check_fpha_constraints(data: &ParsedData, ctx: &mut ValidationContext) {
    if data.fpha_hyperplanes.is_empty() {
        return;
    }

    for row in &data.fpha_hyperplanes {
        let entity_str = format!("Hydro {}", row.hydro_id.0);

        if row.gamma_v < 0.0 {
            ctx.emit(
                &rules::SEMANTIC_FPHA_PLANE_COEFFICIENT_SIGN,
                "system/fpha_hyperplanes.parquet",
                Some(&entity_str),
                format!(
                    "{entity_str} (stage={}, plane={}): gamma_v ({}) must be non-negative (>= 0); \
                     power must not decrease with volume/head (zero is valid for constant-head plants)",
                    row.stage_id.map_or_else(|| "all".to_string(), |s| s.to_string()),
                    row.plane_id,
                    row.gamma_v
                ),
            );
        }

        if row.gamma_s > 0.0 {
            ctx.emit(
                &rules::SEMANTIC_FPHA_PLANE_COEFFICIENT_SIGN,
                "system/fpha_hyperplanes.parquet",
                Some(&entity_str),
                format!(
                    "{entity_str} (stage={}, plane={}): gamma_s ({}) must be non-positive (<= 0); power must not increase with spillage",
                    row.stage_id.map_or_else(|| "all".to_string(), |s| s.to_string()),
                    row.plane_id,
                    row.gamma_s
                ),
            );
        }
    }

    let rows = &data.fpha_hyperplanes;
    let mut i = 0;

    while i < rows.len() {
        let current_hydro_id = rows[i].hydro_id.0;
        let current_stage_id = rows[i].stage_id;
        let group_start = i;

        while i < rows.len()
            && rows[i].hydro_id.0 == current_hydro_id
            && rows[i].stage_id == current_stage_id
        {
            i += 1;
        }

        let plane_count = i - group_start;

        if plane_count < 1 {
            let entity_str = format!("Hydro {current_hydro_id}");
            let stage_label = current_stage_id.map_or_else(|| "all".to_string(), |s| s.to_string());
            ctx.emit(
                &rules::SEMANTIC_FPHA_STAGE_WITHOUT_PLANES,
                "system/fpha_hyperplanes.parquet",
                Some(&entity_str),
                format!(
                    "{entity_str} (stage={stage_label}): no FPHA planes defined; \
                     at least 1 plane is required"
                ),
            );
        }
    }
}

/// Rules 39-41, 44: unit group `id` uniqueness within its own plant (rule 39,
/// ids are plant-scoped — a `HashSet` rebuilt per hydro, never hoisted above
/// the loop); per-group turbined/generation bound consistency (rule 40); the
/// sum of group maxima against the plant's own declared value (rule 41,
/// entity declaration only — a per-stage `hydro_bounds` override is not
/// checked here); and, in the OPPOSITE direction, the sum of group minima
/// against the plant's own declared value (rule 44 — the plant's declared
/// floor must be reachable by summing its groups' own floors, `Σ ≥ declared`,
/// never `Σ ≤ declared` as rule 41 checks for the ceiling). Per-group
/// turbined bound sign is rejected earlier, at parse time, by
/// `hydros.rs::validate_unit_groups` — `validate_schema` aborts
/// all-or-nothing before a negative value ever reaches this Layer-5 check, so
/// it is not re-checked here. Mirrors the plant's turbined-only sign guard:
/// `min_generation_mw`/`max_generation_mw` are unchecked for sign at both
/// plant and group level, inherited, not introduced here. A group's `bus_id`
/// reference is Layer 3's `referential::check_hydro_references`, not this
/// function — duplicating it here would split the hydro bus check across two
/// message shapes.
pub(super) fn check_hydro_unit_groups(data: &ParsedData, ctx: &mut ValidationContext) {
    for hydro in &data.hydros {
        let entity_str = format!("Hydro {}", hydro.id.0);

        let mut seen_group_ids: HashSet<i32> = HashSet::new();
        for group in &hydro.unit_groups {
            if !seen_group_ids.insert(group.id.0) {
                ctx.emit(
                    &rules::UNIT_GROUP_DUPLICATE_ID,
                    "system/hydros.json",
                    Some(&entity_str),
                    format!(
                        "{entity_str}: unit group id {} is declared more than once; unit \
                         group ids must be unique within a plant",
                        group.id.0
                    ),
                );
            }
        }

        for group in &hydro.unit_groups {
            let group_str = format!("{entity_str} unit group {}", group.id.0);

            if group.min_turbined_m3s > group.max_turbined_m3s {
                ctx.emit(
                    &rules::UNIT_GROUP_BOUNDS_INVERTED,
                    "system/hydros.json",
                    Some(&group_str),
                    format!(
                        "{group_str}: min_turbined_m3s ({}) > max_turbined_m3s ({}); unit \
                         group turbine bounds are inconsistent",
                        group.min_turbined_m3s, group.max_turbined_m3s
                    ),
                );
            }

            if group.min_generation_mw > group.max_generation_mw {
                ctx.emit(
                    &rules::UNIT_GROUP_BOUNDS_INVERTED,
                    "system/hydros.json",
                    Some(&group_str),
                    format!(
                        "{group_str}: min_generation_mw ({}) > max_generation_mw ({}); unit \
                         group generation bounds are inconsistent",
                        group.min_generation_mw, group.max_generation_mw
                    ),
                );
            }
        }

        let turbined_sum: f64 = hydro.unit_groups.iter().map(|g| g.max_turbined_m3s).sum();
        let turbined_tolerance = envelope_tolerance(hydro.max_turbined_m3s);
        if turbined_sum > hydro.max_turbined_m3s + turbined_tolerance {
            ctx.emit(
                &rules::UNIT_GROUP_MAXIMA_EXCEED_PLANT,
                "system/hydros.json",
                Some(&entity_str),
                format!(
                    "{entity_str}: unit group max_turbined_m3s sums to {turbined_sum} across \
                     {} unit groups, exceeding the plant's own max_turbined_m3s ({}); the \
                     plant value is the envelope, so declaring unit groups cannot increase a \
                     plant's capacity",
                    hydro.unit_groups.len(),
                    hydro.max_turbined_m3s
                ),
            );
        }

        let generation_sum: f64 = hydro.unit_groups.iter().map(|g| g.max_generation_mw).sum();
        let generation_tolerance = envelope_tolerance(hydro.max_generation_mw);
        if generation_sum > hydro.max_generation_mw + generation_tolerance {
            ctx.emit(
                &rules::UNIT_GROUP_MAXIMA_EXCEED_PLANT,
                "system/hydros.json",
                Some(&entity_str),
                format!(
                    "{entity_str}: unit group max_generation_mw sums to {generation_sum} \
                     across {} unit groups, exceeding the plant's own max_generation_mw ({}); \
                     the plant value is the envelope, so declaring unit groups cannot increase \
                     a plant's capacity",
                    hydro.unit_groups.len(),
                    hydro.max_generation_mw
                ),
            );
        }

        // Rule 44 (flipped direction vs rule 41 above): the plant's declared
        // MINIMUM is a floor its unit groups must be able to reach, `Σ ≥
        // declared`, so the violation condition is `sum < declared - tolerance`
        // — the mirror of rule 41's `sum > declared + tolerance` ceiling check.
        let min_turbined_sum: f64 = hydro.unit_groups.iter().map(|g| g.min_turbined_m3s).sum();
        let min_turbined_tolerance = envelope_tolerance(hydro.min_turbined_m3s);
        if min_turbined_sum < hydro.min_turbined_m3s - min_turbined_tolerance {
            ctx.emit(
                &rules::UNIT_GROUP_MINIMA_BELOW_PLANT,
                "system/hydros.json",
                Some(&entity_str),
                format!(
                    "{entity_str}: unit group min_turbined_m3s sums to {min_turbined_sum} \
                     across {} unit groups, below the plant's own min_turbined_m3s ({}); the \
                     plant's declared minimum is a floor its unit groups must be able to cover",
                    hydro.unit_groups.len(),
                    hydro.min_turbined_m3s
                ),
            );
        }

        let min_generation_sum: f64 = hydro.unit_groups.iter().map(|g| g.min_generation_mw).sum();
        let min_generation_tolerance = envelope_tolerance(hydro.min_generation_mw);
        if min_generation_sum < hydro.min_generation_mw - min_generation_tolerance {
            ctx.emit(
                &rules::UNIT_GROUP_MINIMA_BELOW_PLANT,
                "system/hydros.json",
                Some(&entity_str),
                format!(
                    "{entity_str}: unit group min_generation_mw sums to {min_generation_sum} \
                     across {} unit groups, below the plant's own min_generation_mw ({}); the \
                     plant's declared minimum is a floor its unit groups must be able to cover",
                    hydro.unit_groups.len(),
                    hydro.min_generation_mw
                ),
            );
        }
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
    use super::super::validate_semantic_hydro_thermal;
    use crate::FphaHyperplaneRow;
    use crate::constraints::HydroBoundsRow;
    use crate::test_support::*;
    use crate::validation::{ErrorKind, ValidationContext};
    use chrono::NaiveDate;
    use cobre_core::DiversionChannel;

    // ── Cascade acyclicity tests ───────────────────────────────────────────────

    /// Given an acyclic cascade A -> B -> C (all have downstream_id pointing to next),
    /// no errors are produced.
    #[test]
    fn test_cascade_acyclic_valid() {
        let hydros = vec![
            make_hydro(1, Some(2)), // 1 -> 2
            make_hydro(2, Some(3)), // 2 -> 3
            make_hydro(3, None),    // root (no downstream)
        ];
        let data = make_data(hydros, vec![], vec![], make_stages(vec![0]), vec![], vec![]);
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(
            !ctx.has_errors(),
            "valid acyclic cascade should produce no errors, got: {:?}",
            ctx.errors()
        );
    }

    /// Given a cycle A -> B -> C -> A, exactly one CycleDetected error is produced.
    #[test]
    fn test_cascade_cycle_detected() {
        let hydros = vec![
            make_hydro(1, Some(2)), // 1 -> 2
            make_hydro(2, Some(3)), // 2 -> 3
            make_hydro(3, Some(1)), // 3 -> 1 (cycle!)
        ];
        let data = make_data(hydros, vec![], vec![], make_stages(vec![0]), vec![], vec![]);
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(ctx.has_errors(), "cycle should produce errors");
        let cycle_errors: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::CycleDetected)
            .collect();
        assert!(
            !cycle_errors.is_empty(),
            "should have at least one CycleDetected error"
        );
    }

    /// Empty hydro list produces no cascade errors.
    #[test]
    fn test_cascade_empty_hydros() {
        let data = make_data(vec![], vec![], vec![], make_stages(vec![0]), vec![], vec![]);
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(!ctx.has_errors());
    }

    // ── Hydro storage bounds tests ────────────────────────────────────────────

    /// min_storage > max_storage produces one InvalidValue error with "Hydro 5"
    /// and "storage" in the message.
    #[test]
    fn test_hydro_storage_min_greater_than_max() {
        let mut hydro = make_hydro(5, None);
        hydro.min_storage_hm3 = 200.0;
        hydro.max_storage_hm3 = 100.0;
        let data = make_data(
            vec![hydro],
            vec![],
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
            msg.contains("Hydro 5"),
            "message should contain 'Hydro 5', got: {msg}"
        );
        assert!(
            msg.contains("storage"),
            "message should contain 'storage', got: {msg}"
        );
    }

    /// min_storage == max_storage (run-of-river) produces no error.
    #[test]
    fn test_hydro_storage_equal_bounds_valid() {
        let mut hydro = make_hydro(1, None);
        hydro.min_storage_hm3 = 500.0;
        hydro.max_storage_hm3 = 500.0;
        let data = make_data(
            vec![hydro],
            vec![],
            vec![],
            make_stages(vec![0]),
            vec![],
            vec![],
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(
            !ctx.has_errors(),
            "equal storage bounds should be valid, got: {:?}",
            ctx.errors()
        );
    }

    // ── Hydro turbine bounds tests ────────────────────────────────────────────

    /// min_turbined > max_turbined produces exactly one PLANT-LEVEL InvalidValue
    /// error. Declares an explicit unit group with bounds that satisfy rules
    /// 40/41 on their own and carries no `hydro_unit_group_bounds` override row,
    /// so rule 45 (the only rule that checks the GROUP's own bounds against an
    /// override) has nothing to inspect and does not fire. Rule 44 (Σ group min
    /// ≥ plant min) still fires as a mathematical consequence, not a fixture
    /// bug: rule 41 caps the group's own max at the plant's max (100), rule 40
    /// caps the group's own min at its own max, so the group's min can never
    /// reach the plant's inconsistent min (500) — both findings are
    /// legitimately true of this deliberately-broken plant.
    #[test]
    fn test_hydro_turbine_min_greater_than_max() {
        let mut hydro = make_hydro(2, None);
        hydro.min_turbined_m3s = 500.0;
        hydro.max_turbined_m3s = 100.0;
        hydro.unit_groups = vec![make_unit_group(1, 1, 0.0, 1000.0, 0.0, 100.0)];
        let data = make_data(
            vec![hydro],
            vec![],
            vec![],
            make_stages(vec![0]),
            vec![],
            vec![],
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(ctx.has_errors());
        let turbine_errors: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::InvalidValue)
            .collect();
        let plant_level: Vec<_> = turbine_errors
            .iter()
            .filter(|e| !e.message.contains("unit group"))
            .collect();
        assert_eq!(
            plant_level.len(),
            1,
            "expected exactly 1 plant-level InvalidValue error, got: {:?}",
            turbine_errors
                .iter()
                .map(|e| &e.message)
                .collect::<Vec<_>>()
        );
        assert!(
            plant_level[0].message.contains("Hydro 2"),
            "message should contain 'Hydro 2', got: {}",
            plant_level[0].message
        );
        let group_level: Vec<_> = turbine_errors
            .iter()
            .filter(|e| e.message.contains("unit group"))
            .collect();
        assert_eq!(
            group_level.len(),
            1,
            "rule 44 must also fire — the group's own bounds provably cannot \
             reach the plant's inconsistent minimum, got: {:?}",
            turbine_errors
                .iter()
                .map(|e| &e.message)
                .collect::<Vec<_>>()
        );
        assert!(
            group_level[0].message.contains("min_turbined_m3s"),
            "the companion group-level finding must be the turbined floor, got: {}",
            group_level[0].message
        );
    }

    // ── Hydro outflow bounds tests ────────────────────────────────────────────

    /// When max_outflow_m3s is None, no outflow bound error is produced even if
    /// min_outflow_m3s has any value.
    #[test]
    fn test_hydro_outflow_no_max_no_error() {
        let mut hydro = make_hydro(3, None);
        hydro.min_outflow_m3s = 999.0;
        hydro.max_outflow_m3s = None;
        let data = make_data(
            vec![hydro],
            vec![],
            vec![],
            make_stages(vec![0]),
            vec![],
            vec![],
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(!ctx.has_errors());
    }

    /// When max_outflow_m3s is Some but min > max, one InvalidValue error is produced.
    #[test]
    fn test_hydro_outflow_min_greater_than_max() {
        let mut hydro = make_hydro(4, None);
        hydro.min_outflow_m3s = 500.0;
        hydro.max_outflow_m3s = Some(300.0);
        let data = make_data(
            vec![hydro],
            vec![],
            vec![],
            make_stages(vec![0]),
            vec![],
            vec![],
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(ctx.has_errors());
    }

    // ── Diversion floor requires channel tests ────────────────────────────────

    /// A `min_diversion_m3s` override on a hydro declaring no `diversion` channel
    /// emits exactly one InvalidValue finding naming the hydro and the column.
    #[test]
    fn test_min_diversion_without_channel_emits_one_finding() {
        let hydro = make_hydro(7, None);
        let mut data = make_data(
            vec![hydro],
            vec![],
            vec![],
            make_stages(vec![0]),
            vec![],
            vec![],
        );
        data.hydro_bounds = vec![HydroBoundsRow {
            hydro_id: EntityId::from(7),
            stage_id: 0,
            min_diversion_m3s: Some(5.0),
            ..Default::default()
        }];

        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);

        let findings: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| {
                e.kind == ErrorKind::InvalidValue && e.message.contains("min_diversion_m3s")
            })
            .collect();
        assert_eq!(
            findings.len(),
            1,
            "expected exactly one min_diversion_m3s finding, got: {:?}",
            ctx.errors()
        );
        assert!(
            findings[0].message.contains("Hydro 7"),
            "message should name Hydro 7, got: {}",
            findings[0].message
        );
    }

    #[test]
    fn test_min_diversion_zero_without_channel_emits_no_finding() {
        let hydro = make_hydro(7, None);
        let mut data = make_data(
            vec![hydro],
            vec![],
            vec![],
            make_stages(vec![0]),
            vec![],
            vec![],
        );
        data.hydro_bounds = vec![HydroBoundsRow {
            hydro_id: EntityId::from(7),
            stage_id: 0,
            min_diversion_m3s: Some(0.0),
            ..Default::default()
        }];

        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);

        assert!(
            !ctx.errors()
                .iter()
                .any(|e| e.message.contains("min_diversion_m3s")),
            "a zero floor resolves the channel-less diversion column to [0, 0] (feasible), \
             so it must not be rejected, got: {:?}",
            ctx.errors()
        );
    }

    /// The same override on a hydro that DOES declare a diversion channel emits
    /// no finding.
    #[test]
    fn test_min_diversion_with_channel_emits_no_finding() {
        let mut hydro = make_hydro(7, None);
        hydro.diversion = Some(DiversionChannel {
            downstream_id: EntityId::from(9),
            max_flow_m3s: 10.0,
        });
        let mut data = make_data(
            vec![hydro],
            vec![],
            vec![],
            make_stages(vec![0]),
            vec![],
            vec![],
        );
        data.hydro_bounds = vec![HydroBoundsRow {
            hydro_id: EntityId::from(7),
            stage_id: 0,
            min_diversion_m3s: Some(5.0),
            ..Default::default()
        }];

        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);

        assert!(
            !ctx.errors()
                .iter()
                .any(|e| e.message.contains("min_diversion_m3s")),
            "a hydro with a declared diversion channel should not trigger the \
             no-channel finding, got: {:?}",
            ctx.errors()
        );
    }

    // ── Lifecycle consistency tests ───────────────────────────────────────────

    /// Hydro with entry >= exit produces one InvalidValue error.
    #[test]
    fn test_hydro_lifecycle_entry_gte_exit() {
        let mut hydro = make_hydro(7, None);
        hydro.entry_stage_id = Some(10);
        hydro.exit_stage_id = Some(5);
        let data = make_data(
            vec![hydro],
            vec![],
            vec![],
            make_stages(vec![0]),
            vec![],
            vec![],
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(ctx.has_errors());
        let errors = ctx.errors();
        assert!(
            errors.iter().any(|e| e.kind == ErrorKind::InvalidValue),
            "should have InvalidValue error for lifecycle"
        );
    }

    /// An entity with only `entry_stage_id` set (no exit) produces no lifecycle
    /// ordering error. A line carries the generic `entry/exit` ordering check
    /// without the hydro-specific filling guards, isolating the ordering-only
    /// assertion from them.
    #[test]
    fn test_lifecycle_only_entry_no_error() {
        let line = make_windowed_line(8, Some(5), None);
        let data = make_data(
            vec![make_hydro(1, None), make_hydro(2, None)],
            vec![],
            vec![line],
            make_stages(vec![0]),
            vec![],
            vec![],
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(
            !ctx.has_errors(),
            "only entry_stage_id set should produce no error, got: {:?}",
            ctx.errors()
        );
    }

    /// An entity with valid `entry < exit` produces no lifecycle ordering error.
    /// A line is the carrier (see [`test_lifecycle_only_entry_no_error`]): the
    /// generic ordering check accepts a window with `entry < exit`, while a hydro
    /// with both fields would be rejected as entry-only.
    #[test]
    fn test_lifecycle_valid() {
        let line = make_windowed_line(9, Some(0), Some(10));
        let data = make_data(
            vec![make_hydro(1, None), make_hydro(2, None)],
            vec![],
            vec![line],
            make_stages(vec![0]),
            vec![],
            vec![],
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(!ctx.has_errors());
    }

    // ── Commissioning hygiene: ordering parity + applied-window silence ────────

    use cobre_core::entities::{
        ContractType, EnergyContract, NonControllableSource, PumpingStation,
    };
    use cobre_core::{EntityId, Line, Thermal};

    /// Build a `PumpingStation` with the given entry/exit commissioning window.
    fn make_pumping_lc(id: i32, entry: Option<i32>, exit: Option<i32>) -> PumpingStation {
        PumpingStation {
            id: EntityId::from(id),
            name: format!("Pump_{id}"),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId::from(1),
            source_hydro_id: EntityId::from(1),
            destination_hydro_id: EntityId::from(2),
            entry_stage_id: entry,
            exit_stage_id: exit,
            consumption_mw_per_m3s: 0.5,
            min_flow_m3s: 0.0,
            max_flow_m3s: 100.0,
        }
    }

    /// Build a `NonControllableSource` with the given entry/exit window.
    fn make_ncs_lc(id: i32, entry: Option<i32>, exit: Option<i32>) -> NonControllableSource {
        NonControllableSource {
            id: EntityId::from(id),
            name: format!("NCS_{id}"),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId::from(1),
            entry_stage_id: entry,
            exit_stage_id: exit,
            max_generation_mw: 300.0,
            allow_curtailment: true,
            curtailment_cost: 0.01,
        }
    }

    /// Build an `EnergyContract` with the given entry/exit window.
    fn make_contract_lc(id: i32, entry: Option<i32>, exit: Option<i32>) -> EnergyContract {
        EnergyContract {
            id: EntityId::from(id),
            name: format!("Contract_{id}"),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId::from(1),
            contract_type: ContractType::Import,
            entry_stage_id: entry,
            exit_stage_id: exit,
            price_per_mwh: 200.0,
            min_mw: 0.0,
            max_mw: 1000.0,
        }
    }

    /// A pumping station with entry >= exit produces an InvalidValue error citing
    /// `system/pumping_stations.json`.
    #[test]
    fn test_pumping_lifecycle_entry_gte_exit() {
        let mut data = make_data(
            vec![make_hydro(1, None), make_hydro(2, None)],
            vec![],
            vec![],
            make_stages(vec![0]),
            vec![],
            vec![],
        );
        data.pumping_stations = vec![make_pumping_lc(5, Some(5), Some(3))];
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        let errs: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| {
                e.kind == ErrorKind::InvalidValue
                    && e.file.to_string_lossy() == "system/pumping_stations.json"
            })
            .collect();
        assert_eq!(
            errs.len(),
            1,
            "expected 1 InvalidValue for pumping ordering, got: {:?}",
            errs.iter().map(|e| &e.message).collect::<Vec<_>>()
        );
        assert!(errs[0].message.contains("PumpingStation 5"));
    }

    /// An NCS with entry >= exit produces an InvalidValue error citing
    /// `system/non_controllable_sources.json`.
    #[test]
    fn test_ncs_lifecycle_entry_gte_exit() {
        let mut data = make_data(
            vec![make_hydro(1, None)],
            vec![],
            vec![],
            make_stages(vec![0]),
            vec![],
            vec![],
        );
        data.non_controllable_sources = vec![make_ncs_lc(7, Some(8), Some(2))];
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        let errs: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| {
                e.kind == ErrorKind::InvalidValue
                    && e.file.to_string_lossy() == "system/non_controllable_sources.json"
            })
            .collect();
        assert_eq!(errs.len(), 1, "expected 1 InvalidValue for NCS ordering");
        assert!(errs[0].message.contains("NonControllableSource 7"));
    }

    /// An energy contract with entry >= exit produces an InvalidValue error citing
    /// `system/energy_contracts.json`.
    #[test]
    fn test_energy_contract_lifecycle_entry_gte_exit() {
        let mut data = make_data(
            vec![make_hydro(1, None)],
            vec![],
            vec![],
            make_stages(vec![0]),
            vec![],
            vec![],
        );
        data.energy_contracts = vec![make_contract_lc(9, Some(4), Some(4))];
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        let errs: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| {
                e.kind == ErrorKind::InvalidValue
                    && e.file.to_string_lossy() == "system/energy_contracts.json"
            })
            .collect();
        assert_eq!(
            errs.len(),
            1,
            "expected 1 InvalidValue for contract ordering (entry == exit)"
        );
        assert!(errs[0].message.contains("EnergyContract 9"));
    }

    /// A filling hydro's window IS applied (the `FillingConfig` drives its
    /// lifecycle), so it emits NO `ModelQuality` warning, and the case still loads
    /// (`has_errors()` is false). The filling pairing keeps the hydro guard-clean
    /// (a bare entry without filling would be rejected), isolating the
    /// commissioning behavior under test.
    #[test]
    fn test_filling_hydro_emits_no_warning() {
        // start (1) < entry (2) < horizon (3); no exit; seed left empty (no
        // filling_storage entry ⇒ guard 4 does not fire).
        let hydro = make_filling_hydro(3, 1, 2, 10.0);
        let data = make_data(
            vec![hydro],
            vec![],
            vec![],
            make_stages(vec![0, 1, 2]),
            vec![],
            vec![],
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(
            !ctx.has_errors(),
            "a well-formed filling hydro must not produce an error, got: {:?}",
            ctx.errors()
        );
        let warnings: Vec<_> = ctx
            .warnings()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::ModelQuality)
            .collect();
        assert!(
            warnings.is_empty(),
            "a filling hydro's window is applied; no ModelQuality warning \
             is expected, got: {:?}",
            warnings.iter().map(|e| &e.message).collect::<Vec<_>>()
        );
    }

    /// A non-filling hydro with `exit_stage_id` set (and `filling = None`) has its
    /// commissioning window applied at the LP fill site, so it emits NO
    /// `ModelQuality` warning. Exit alone (no entry) clears both the
    /// `entry >= exit` ordering check and the filling guards, isolating the
    /// commissioning behavior under test.
    #[test]
    fn test_non_filling_windowed_hydro_emits_no_warning() {
        let mut hydro = make_hydro(7, None);
        hydro.exit_stage_id = Some(10);
        let data = make_data(
            vec![hydro],
            vec![],
            vec![],
            make_stages(vec![0]),
            vec![],
            vec![],
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(
            !ctx.has_errors(),
            "a non-filling exit-only hydro must not produce an error, got: {:?}",
            ctx.errors()
        );
        let warnings: Vec<_> = ctx
            .warnings()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::ModelQuality)
            .collect();
        assert!(
            warnings.is_empty(),
            "a non-filling hydro's window is applied; no ModelQuality warning is \
             expected, got: {:?}",
            warnings.iter().map(|e| &e.message).collect::<Vec<_>>()
        );
    }

    /// With no entity setting entry/exit (the inert default), no ModelQuality
    /// commissioning warning is emitted.
    #[test]
    fn test_commissioning_unset_no_warning() {
        let data = make_data(
            vec![make_hydro(1, None)],
            vec![make_thermal(1, 0.0, 500.0)],
            vec![],
            make_stages(vec![0]),
            vec![],
            vec![],
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(
            !ctx.warnings()
                .iter()
                .any(|e| e.kind == ErrorKind::ModelQuality),
            "no entity sets entry/exit, so no ModelQuality warning is expected, got: {:?}",
            ctx.warnings()
        );
    }

    /// Build a windowed `Line` (entry < exit) for the commissioning tests.
    fn make_windowed_line(id: i32, entry: Option<i32>, exit: Option<i32>) -> Line {
        Line {
            id: EntityId::from(id),
            name: format!("Line_{id}"),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            source_bus_id: EntityId::from(1),
            target_bus_id: EntityId::from(2),
            entry_stage_id: entry,
            exit_stage_id: exit,
            direct_capacity_mw: 100.0,
            reverse_capacity_mw: 100.0,
            losses_percent: 0.0,
            exchange_cost: 0.01,
        }
    }

    /// A windowed thermal, line, NCS, and pumping station each have their
    /// commissioning window APPLIED at the LP fill site, so none of them emits a
    /// `ModelQuality` warning.
    #[test]
    fn test_applied_window_entities_emit_no_warning() {
        let thermal = Thermal {
            entry_stage_id: Some(1),
            exit_stage_id: Some(2),
            ..make_thermal(1, 0.0, 100.0)
        };
        let line = make_windowed_line(1, Some(1), Some(2));
        let mut data = make_data(
            vec![make_hydro(1, None), make_hydro(2, None)],
            vec![thermal],
            vec![line],
            make_stages(vec![0, 1, 2]),
            vec![],
            vec![],
        );
        data.non_controllable_sources = vec![make_ncs_lc(3, Some(1), Some(2))];
        data.pumping_stations = vec![make_pumping_lc(4, Some(1), Some(2))];

        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        let model_quality: Vec<_> = ctx
            .warnings()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::ModelQuality)
            .collect();
        assert!(
            model_quality.is_empty(),
            "thermal/line/NCS/pumping windows are applied; no ModelQuality \
             warning is expected, got: {:?}",
            model_quality.iter().map(|e| &e.message).collect::<Vec<_>>()
        );
    }

    /// A windowed energy contract's commissioning window IS applied at the LP fill
    /// site (dormant stages zero-pinned), so it emits no `ModelQuality` warning.
    #[test]
    fn test_windowed_contract_emits_no_warning() {
        let mut data = make_data(
            vec![make_hydro(1, None)],
            vec![],
            vec![],
            make_stages(vec![0, 1, 2]),
            vec![],
            vec![],
        );
        data.energy_contracts = vec![make_contract_lc(5, Some(2), Some(10))];
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        let model_quality: Vec<_> = ctx
            .warnings()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::ModelQuality)
            .collect();
        assert!(
            model_quality.is_empty(),
            "a windowed energy contract's window is applied; no ModelQuality \
             warning is expected, got: {:?}",
            model_quality.iter().map(|e| &e.message).collect::<Vec<_>>()
        );
    }

    // ── Filling guard tests ───────────────────────────────────────────────────

    use cobre_core::HydroStorage;
    use cobre_core::entities::{FillingConfig, Hydro};

    /// Build a well-formed filling hydro: `start < entry < horizon`, an inflow
    /// cap of `inflow`, and an entry stage paired with the filling config.
    fn make_filling_hydro(id: i32, start_stage_id: i32, entry_stage_id: i32, inflow: f64) -> Hydro {
        let mut h = make_hydro(id, None);
        h.entry_stage_id = Some(entry_stage_id);
        h.filling = Some(FillingConfig {
            start_stage_id,
            filling_min_rate_m3s: inflow,
        });
        h
    }

    /// Pull only the `system/hydros.json` `InvalidValue` errors out of `ctx`.
    fn hydro_invalid_value_messages(ctx: &ValidationContext) -> Vec<String> {
        ctx.errors()
            .into_iter()
            .filter(|e| {
                e.kind == ErrorKind::InvalidValue
                    && e.file.to_string_lossy() == "system/hydros.json"
            })
            .map(|e| e.message.clone())
            .collect()
    }

    /// Guard 1 (well-formed): `entry_stage_id = Some` with `filling = None` is a
    /// valid non-filling commissioning window and produces NO error — the converse
    /// of the filling⟹entry implication does not hold.
    #[test]
    fn test_filling_guard_entry_without_filling_no_error() {
        let mut hydro = make_hydro(1, None);
        hydro.entry_stage_id = Some(4);
        hydro.filling = None;
        let data = make_data(
            vec![hydro],
            vec![],
            vec![],
            make_stages(vec![0, 1, 2, 3, 4, 5]),
            vec![],
            vec![],
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(
            hydro_invalid_value_messages(&ctx).is_empty(),
            "a non-filling hydro with a commissioning window must validate clean, got: {:?}",
            ctx.errors()
        );
    }

    /// Guard 1 (violating): `filling = Some` with `entry_stage_id = None` still
    /// errors — a filling config needs an entry to fill toward.
    #[test]
    fn test_filling_guard_filling_without_entry_errors() {
        let mut hydro = make_hydro(1, None);
        hydro.entry_stage_id = None;
        hydro.filling = Some(FillingConfig {
            start_stage_id: 0,
            filling_min_rate_m3s: 10.0,
        });
        let data = make_data(
            vec![hydro],
            vec![],
            vec![],
            make_stages(vec![0, 1, 2, 3, 4, 5]),
            vec![],
            vec![],
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        let msgs = hydro_invalid_value_messages(&ctx);
        assert!(
            msgs.iter()
                .any(|m| m.contains("Hydro 1") && m.contains("requires an entry_stage_id")),
            "expected filling-requires-entry error, got: {msgs:?}"
        );
    }

    /// Guard 1 (well-formed): neither `entry_stage_id` nor `filling` set produces
    /// no filling-attributable error.
    #[test]
    fn test_filling_guard_neither_set_no_error() {
        let data = make_data(
            vec![make_hydro(1, None)],
            vec![],
            vec![],
            make_stages(vec![0, 1, 2, 3]),
            vec![],
            vec![],
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(
            hydro_invalid_value_messages(&ctx).is_empty(),
            "no filling and no entry should produce no error, got: {:?}",
            ctx.errors()
        );
    }

    /// Guard 2 (violating): `start_stage_id (4) >= entry_stage_id (2)` yields an
    /// `InvalidValue` stating `start_stage_id` must be less than `entry_stage_id`.
    #[test]
    fn test_filling_guard_start_not_before_entry_errors() {
        let hydro = make_filling_hydro(1, 4, 2, 10.0);
        let data = make_data(
            vec![hydro],
            vec![],
            vec![],
            make_stages(vec![0, 1, 2, 3, 4, 5]),
            vec![],
            vec![],
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        let msgs = hydro_invalid_value_messages(&ctx);
        assert!(
            msgs.iter().any(|m| m.contains("Hydro 1")
                && m.contains("must be less than")
                && m.contains("entry_stage_id")),
            "expected start<entry error, got: {msgs:?}"
        );
    }

    /// Guard 2 (well-formed): `start_stage_id (1) < entry_stage_id (3)` produces no
    /// error.
    #[test]
    fn test_filling_guard_start_before_entry_no_error() {
        let hydro = make_filling_hydro(1, 1, 3, 10.0);
        let data = make_data(
            vec![hydro],
            vec![],
            vec![],
            make_stages(vec![0, 1, 2, 3, 4, 5]),
            vec![],
            vec![],
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(
            hydro_invalid_value_messages(&ctx).is_empty(),
            "start < entry should produce no error, got: {:?}",
            ctx.errors()
        );
    }

    /// Guard 3: `entry_stage_id (6)` at the 6-stage horizon leaves no operating
    /// stage. It is a tolerated, warned condition (filling-throughout), not a
    /// rejection: exactly one `ModelQuality` warning and no error, so the case
    /// still loads.
    #[test]
    fn test_filling_guard_entry_at_horizon_warns() {
        // Six stages (ids 0..=5) ⇒ horizon = 6; entry at 6 has no operating stage.
        let hydro = make_filling_hydro(1, 1, 6, 10.0);
        let data = make_data(
            vec![hydro],
            vec![],
            vec![],
            make_stages(vec![0, 1, 2, 3, 4, 5]),
            vec![],
            vec![],
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(
            !ctx.has_errors(),
            "entry at horizon on a filling hydro must not produce an error, got: {:?}",
            ctx.errors()
        );
        let warnings: Vec<_> = ctx
            .warnings()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::ModelQuality)
            .collect();
        assert_eq!(
            warnings.len(),
            1,
            "expected exactly 1 ModelQuality warning, got: {:?}",
            warnings.iter().map(|e| &e.message).collect::<Vec<_>>()
        );
        assert!(
            warnings[0].message.contains("Hydro 1")
                && warnings[0].message.contains("never operates"),
            "warning should name the entity and state it never operates within the study, got: {}",
            warnings[0].message
        );
    }

    /// Guard 3 (well-formed): `entry_stage_id (4)` strictly below the 6-stage
    /// horizon leaves an operating stage and produces no error.
    #[test]
    fn test_filling_guard_entry_below_horizon_no_error() {
        let hydro = make_filling_hydro(1, 1, 4, 10.0);
        let data = make_data(
            vec![hydro],
            vec![],
            vec![],
            make_stages(vec![0, 1, 2, 3, 4, 5]),
            vec![],
            vec![],
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(
            hydro_invalid_value_messages(&ctx).is_empty(),
            "entry below horizon should produce no error, got: {:?}",
            ctx.errors()
        );
    }

    /// Guard 4 (violating): a seed equal to `min_storage_hm3` is rejected; the
    /// upper bound of the filling range is strict.
    #[test]
    fn test_filling_guard_seed_at_min_storage_errors() {
        let mut hydro = make_filling_hydro(1, 1, 4, 10.0);
        hydro.min_storage_hm3 = 200.0;
        let mut data = make_data(
            vec![hydro],
            vec![],
            vec![],
            make_stages(vec![0, 1, 2, 3, 4, 5]),
            vec![],
            vec![],
        );
        data.initial_conditions.filling_storage = vec![HydroStorage {
            hydro_id: EntityId::from(1),
            value_hm3: 200.0, // == min_storage_hm3 ⇒ rejected (strict upper bound)
        }];
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        let msgs = hydro_invalid_value_messages(&ctx);
        assert!(
            msgs.iter()
                .any(|m| m.contains("Hydro 1") && m.contains("filling_storage seed")),
            "expected seed-range error at min_storage, got: {msgs:?}"
        );
    }

    /// Guard 4 (well-formed): a seed strictly inside `[0, min_storage_hm3)`
    /// produces no error. `start_stage_id == 0` (study starts mid-filling) is
    /// the only setting where a nonzero seed is valid; under `start_stage_id > 0`
    /// guard 6 would require the empty-pit seed `0`.
    #[test]
    fn test_filling_guard_seed_in_range_no_error() {
        let mut hydro = make_filling_hydro(1, 0, 4, 10.0);
        hydro.min_storage_hm3 = 200.0;
        let mut data = make_data(
            vec![hydro],
            vec![],
            vec![],
            make_stages(vec![0, 1, 2, 3, 4, 5]),
            vec![],
            vec![],
        );
        data.initial_conditions.filling_storage = vec![HydroStorage {
            hydro_id: EntityId::from(1),
            value_hm3: 50.0, // strictly in [0, 200)
        }];
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(
            hydro_invalid_value_messages(&ctx).is_empty(),
            "seed in range should produce no error, got: {:?}",
            ctx.errors()
        );
    }

    /// Guard 5 (violating): `exit_stage_id` set on a filling hydro yields an
    /// `InvalidValue` stating exit is rejected for filling hydros.
    #[test]
    fn test_filling_guard_exit_on_filling_errors() {
        let mut hydro = make_filling_hydro(1, 1, 4, 10.0);
        hydro.exit_stage_id = Some(10);
        let data = make_data(
            vec![hydro],
            vec![],
            vec![],
            make_stages(vec![0, 1, 2, 3, 4, 5]),
            vec![],
            vec![],
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        let msgs = hydro_invalid_value_messages(&ctx);
        assert!(
            msgs.iter()
                .any(|m| m.contains("Hydro 1") && m.contains("entry-only")),
            "expected exit-rejected error, got: {msgs:?}"
        );
    }

    /// Guard 5 (well-formed): a filling hydro with no `exit_stage_id` produces no
    /// error.
    #[test]
    fn test_filling_guard_no_exit_no_error() {
        let hydro = make_filling_hydro(1, 1, 4, 10.0);
        let data = make_data(
            vec![hydro],
            vec![],
            vec![],
            make_stages(vec![0, 1, 2, 3, 4, 5]),
            vec![],
            vec![],
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(
            hydro_invalid_value_messages(&ctx).is_empty(),
            "no exit on a filling hydro should produce no error, got: {:?}",
            ctx.errors()
        );
    }

    /// A fully well-formed filling hydro (`start < entry < horizon`, seed in
    /// range, no exit, zero inflow cap) produces zero errors AND zero warnings.
    /// `start_stage_id == 0` (study starts mid-filling) lets the nonzero in-range
    /// seed coexist with guard 6's empty-pit rule. A filling hydro's window IS
    /// applied via the `FillingConfig`, so no `ModelQuality` warning is emitted
    /// either.
    #[test]
    fn test_filling_guard_well_formed_no_error() {
        let mut hydro = make_filling_hydro(1, 0, 4, 0.0); // inflow cap 0.0 is valid
        hydro.min_storage_hm3 = 200.0;
        let mut data = make_data(
            vec![hydro],
            vec![],
            vec![],
            make_stages(vec![0, 1, 2, 3, 4, 5]),
            vec![],
            vec![],
        );
        data.initial_conditions.filling_storage = vec![HydroStorage {
            hydro_id: EntityId::from(1),
            value_hm3: 50.0,
        }];
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(
            !ctx.has_errors(),
            "a well-formed filling hydro should produce no errors, got: {:?}",
            ctx.errors()
        );
        assert!(
            ctx.warnings().is_empty(),
            "a filling hydro's window is applied, so no warning is expected, got: {:?}",
            ctx.warnings()
        );
    }

    /// Guard 6 (violating): `start_stage_id > 0` (a `PreFilling` phase exists)
    /// with a nonzero `filling_storage` seed yields an `InvalidValue` stating the
    /// seed must be the empty-pit `0`.
    #[test]
    fn test_filling_guard_start_above_zero_nonzero_seed_errors() {
        let mut hydro = make_filling_hydro(1, 2, 4, 10.0); // start (2) > 0 ⇒ PreFilling
        hydro.min_storage_hm3 = 200.0;
        let mut data = make_data(
            vec![hydro],
            vec![],
            vec![],
            make_stages(vec![0, 1, 2, 3, 4, 5]),
            vec![],
            vec![],
        );
        data.initial_conditions.filling_storage = vec![HydroStorage {
            hydro_id: EntityId::from(1),
            value_hm3: 50.0, // nonzero ⇒ rejected when a PreFilling phase exists
        }];
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        let msgs = hydro_invalid_value_messages(&ctx);
        assert!(
            msgs.iter()
                .any(|m| m.contains("Hydro 1") && m.contains("must be 0 (empty pit)")),
            "expected empty-pit seed error, got: {msgs:?}"
        );
    }

    /// Guard 6 (well-formed): `start_stage_id > 0` with the empty-pit seed `0`
    /// produces no error.
    #[test]
    fn test_filling_guard_start_above_zero_empty_pit_no_error() {
        let mut hydro = make_filling_hydro(1, 2, 4, 10.0); // start (2) > 0 ⇒ PreFilling
        hydro.min_storage_hm3 = 200.0;
        let mut data = make_data(
            vec![hydro],
            vec![],
            vec![],
            make_stages(vec![0, 1, 2, 3, 4, 5]),
            vec![],
            vec![],
        );
        data.initial_conditions.filling_storage = vec![HydroStorage {
            hydro_id: EntityId::from(1),
            value_hm3: 0.0, // empty pit ⇒ valid when a PreFilling phase exists
        }];
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(
            hydro_invalid_value_messages(&ctx).is_empty(),
            "empty-pit seed with a PreFilling phase should produce no error, got: {:?}",
            ctx.errors()
        );
    }

    // ── Geometry monotonicity tests ───────────────────────────────────────────

    /// Empty geometry slice produces no errors.
    #[test]
    fn test_geometry_empty_no_error() {
        let data = make_data(
            vec![make_hydro(1, None)],
            vec![],
            vec![],
            make_stages(vec![0]),
            vec![],
            vec![],
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(!ctx.has_errors());
    }

    /// Strictly increasing volume, non-decreasing height and area produces no error.
    #[test]
    fn test_geometry_valid_monotonic() {
        let geometry = vec![
            make_geom_row(1, 10.0, 100.0, 1.0),
            make_geom_row(1, 20.0, 110.0, 1.5),
            make_geom_row(1, 30.0, 120.0, 2.0),
        ];
        let data = make_data(
            vec![make_hydro(1, None)],
            vec![],
            vec![],
            make_stages(vec![0]),
            geometry,
            vec![],
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(
            !ctx.has_errors(),
            "valid monotonic geometry should produce no errors, got: {:?}",
            ctx.errors()
        );
    }

    /// Non-monotonic volume produces BusinessRuleViolation with "Hydro 3" and "volume".
    #[test]
    fn test_geometry_non_monotonic_volume() {
        // Equal (not decreasing) volumes: the parser pre-sorts by volume, so only
        // a duplicate survives sorting to trigger the strict-increase check.
        let geometry = vec![
            make_geom_row(3, 10.0, 100.0, 1.0),
            make_geom_row(3, 20.0, 110.0, 1.5),
            make_geom_row(3, 20.0, 115.0, 1.6), // duplicate volume — not strictly increasing
        ];
        let data = make_data(
            vec![make_hydro(3, None)],
            vec![],
            vec![],
            make_stages(vec![0]),
            geometry,
            vec![],
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(ctx.has_errors());
        let errors = ctx.errors();
        let relevant: Vec<_> = errors
            .iter()
            .filter(|e| e.kind == ErrorKind::BusinessRuleViolation)
            .collect();
        assert!(!relevant.is_empty(), "should have BusinessRuleViolation");
        let msg = &relevant[0].message;
        assert!(
            msg.contains("Hydro 3"),
            "message should contain 'Hydro 3', got: {msg}"
        );
        assert!(
            msg.contains("volume"),
            "message should contain 'volume', got: {msg}"
        );
    }

    /// Non-monotonic height produces BusinessRuleViolation with "height" in message.
    #[test]
    fn test_geometry_non_monotonic_height() {
        let geometry = vec![
            make_geom_row(2, 10.0, 100.0, 1.0),
            make_geom_row(2, 20.0, 90.0, 1.5), // height decreased — violation
            make_geom_row(2, 30.0, 110.0, 2.0),
        ];
        let data = make_data(
            vec![make_hydro(2, None)],
            vec![],
            vec![],
            make_stages(vec![0]),
            geometry,
            vec![],
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(ctx.has_errors());
        let errors = ctx.errors();
        let relevant: Vec<_> = errors
            .iter()
            .filter(|e| e.kind == ErrorKind::BusinessRuleViolation)
            .collect();
        assert!(!relevant.is_empty());
        let msg = &relevant[0].message;
        assert!(
            msg.contains("height"),
            "message should mention 'height', got: {msg}"
        );
    }

    /// A `height_m` decrease within the relative-with-floor tolerance (a hair
    /// below the previous row, on a ~100 m curve) must not be rejected.
    #[test]
    fn test_geometry_height_within_relative_tolerance_no_error() {
        let geometry = vec![
            make_geom_row(2, 10.0, 100.0, 1.0),
            // Decrease of 1e-8, an order of magnitude below the 1e-7 tolerance.
            make_geom_row(2, 20.0, 100.0 - 1e-8, 1.5),
        ];
        let data = make_data(
            vec![make_hydro(2, None)],
            vec![],
            vec![],
            make_stages(vec![0]),
            geometry,
            vec![],
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(
            !ctx.has_errors(),
            "a height decrease within relative tolerance must not be rejected, got: {:?}",
            ctx.errors()
        );
    }

    /// A `height_m` decrease beyond the tolerance (an order of magnitude
    /// larger) is still rejected — the tolerance must have power.
    #[test]
    fn test_geometry_height_beyond_tolerance_still_errors() {
        let geometry = vec![
            make_geom_row(2, 10.0, 100.0, 1.0),
            // Decrease of 1e-6, an order of magnitude above the 1e-7 tolerance.
            make_geom_row(2, 20.0, 100.0 - 1e-6, 1.5),
        ];
        let data = make_data(
            vec![make_hydro(2, None)],
            vec![],
            vec![],
            make_stages(vec![0]),
            geometry,
            vec![],
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        let relevant: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::BusinessRuleViolation && e.message.contains("height"))
            .collect();
        assert_eq!(
            relevant.len(),
            1,
            "a height decrease beyond tolerance must still be rejected, got: {:?}",
            ctx.errors()
        );
    }

    /// An `area_km2` decrease within the relative-with-floor tolerance must
    /// not be rejected.
    #[test]
    fn test_geometry_area_within_relative_tolerance_no_error() {
        let geometry = vec![
            make_geom_row(2, 10.0, 100.0, 1.0),
            // Decrease of 1e-10, an order of magnitude below the 1e-9 floor
            // tolerance (area_km2 magnitude is below the envelope_tolerance floor).
            make_geom_row(2, 20.0, 110.0, 1.0 - 1e-10),
        ];
        let data = make_data(
            vec![make_hydro(2, None)],
            vec![],
            vec![],
            make_stages(vec![0]),
            geometry,
            vec![],
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(
            !ctx.has_errors(),
            "an area decrease within relative tolerance must not be rejected, got: {:?}",
            ctx.errors()
        );
    }

    /// An `area_km2` decrease beyond the tolerance (an order of magnitude
    /// larger) is still rejected — the tolerance must have power.
    #[test]
    fn test_geometry_area_beyond_tolerance_still_errors() {
        let geometry = vec![
            make_geom_row(2, 10.0, 100.0, 1.0),
            // Decrease of 1e-8, an order of magnitude above the 1e-9 floor tolerance.
            make_geom_row(2, 20.0, 110.0, 1.0 - 1e-8),
        ];
        let data = make_data(
            vec![make_hydro(2, None)],
            vec![],
            vec![],
            make_stages(vec![0]),
            geometry,
            vec![],
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        let relevant: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::BusinessRuleViolation && e.message.contains("area"))
            .collect();
        assert_eq!(
            relevant.len(),
            1,
            "an area decrease beyond tolerance must still be rejected, got: {:?}",
            ctx.errors()
        );
    }

    // ── FPHA minimum planes tests ─────────────────────────────────────────────

    /// 1 plane for (hydro, stage) is valid — minimum count is 1.
    #[test]
    fn test_fpha_one_plane_valid() {
        let rows = vec![make_fpha_row(1, Some(0), 0)];
        let data = make_data(
            vec![make_hydro(1, None)],
            vec![],
            vec![],
            make_stages(vec![0]),
            vec![],
            rows,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(
            !ctx.has_errors(),
            "1 plane should be valid (minimum is 1), got: {:?}",
            ctx.errors()
        );
    }

    /// 2 planes for (hydro, stage) is valid — minimum count is 1.
    #[test]
    fn test_fpha_two_planes_valid() {
        let rows = vec![make_fpha_row(1, Some(0), 0), make_fpha_row(1, Some(0), 1)];
        let data = make_data(
            vec![make_hydro(1, None)],
            vec![],
            vec![],
            make_stages(vec![0]),
            vec![],
            rows,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(
            !ctx.has_errors(),
            "2 planes should be valid (minimum is 1), got: {:?}",
            ctx.errors()
        );
    }

    /// 3 planes for (hydro, stage) produces no minimum-count error.
    #[test]
    fn test_fpha_minimum_planes_valid() {
        let rows = vec![
            make_fpha_row(1, Some(0), 0),
            make_fpha_row(1, Some(0), 1),
            make_fpha_row(1, Some(0), 2),
        ];
        let data = make_data(
            vec![make_hydro(1, None)],
            vec![],
            vec![],
            make_stages(vec![0]),
            vec![],
            rows,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(
            !ctx.has_errors(),
            "3 planes should be valid, got: {:?}",
            ctx.errors()
        );
    }

    // ── FPHA gamma sign tests ─────────────────────────────────────────────────

    /// Negative gamma_v produces BusinessRuleViolation.
    #[test]
    fn test_fpha_negative_gamma_v() {
        let mut row = make_fpha_row(1, None, 0);
        row.gamma_v = -0.5; // invalid: must be >= 0
        let rows = vec![row, make_fpha_row(1, None, 1), make_fpha_row(1, None, 2)];
        let data = make_data(
            vec![make_hydro(1, None)],
            vec![],
            vec![],
            make_stages(vec![0]),
            vec![],
            rows,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(ctx.has_errors());
        let errors = ctx.errors();
        assert!(
            errors
                .iter()
                .any(|e| e.kind == ErrorKind::BusinessRuleViolation),
            "negative gamma_v should produce BusinessRuleViolation"
        );
    }

    /// Positive gamma_s produces BusinessRuleViolation.
    #[test]
    fn test_fpha_positive_gamma_s() {
        let mut row = make_fpha_row(1, None, 0);
        row.gamma_s = 0.1; // invalid: must be <= 0
        let rows = vec![row, make_fpha_row(1, None, 1), make_fpha_row(1, None, 2)];
        let data = make_data(
            vec![make_hydro(1, None)],
            vec![],
            vec![],
            make_stages(vec![0]),
            vec![],
            rows,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(ctx.has_errors());
        let errors = ctx.errors();
        assert!(
            errors
                .iter()
                .any(|e| e.kind == ErrorKind::BusinessRuleViolation),
            "positive gamma_s should produce BusinessRuleViolation"
        );
    }

    /// gamma_s == 0.0 is valid (non-positive).
    #[test]
    fn test_fpha_gamma_s_zero_valid() {
        let rows: Vec<FphaHyperplaneRow> = (0..3)
            .map(|i| {
                let mut r = make_fpha_row(1, None, i);
                r.gamma_s = 0.0;
                r
            })
            .collect();
        let data = make_data(
            vec![make_hydro(1, None)],
            vec![],
            vec![],
            make_stages(vec![0]),
            vec![],
            rows,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(
            !ctx.has_errors(),
            "gamma_s == 0 should be valid, got: {:?}",
            ctx.errors()
        );
    }

    /// gamma_v == 0.0 is valid (constant-head plant: zero storage coefficient).
    #[test]
    fn test_fpha_gamma_v_zero_valid() {
        let mut row = make_fpha_row(1, None, 0);
        row.gamma_v = 0.0; // valid: >= 0 (constant-head)
        let rows = vec![row];
        let data = make_data(
            vec![make_hydro(1, None)],
            vec![],
            vec![],
            make_stages(vec![0]),
            vec![],
            rows,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(
            !ctx.has_errors(),
            "gamma_v == 0 should be valid for constant-head plants, got: {:?}",
            ctx.errors()
        );
    }

    /// Empty FPHA slice produces no errors (rules 11-12 are skipped).
    #[test]
    fn test_fpha_empty_no_error() {
        let data = make_data(
            vec![make_hydro(1, None)],
            vec![],
            vec![],
            make_stages(vec![0]),
            vec![],
            vec![],
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(!ctx.has_errors());
    }

    // ── All-rules-checked test (no short-circuit) ─────────────────────────────

    /// Given two hydros each with bound violations, both errors are collected
    /// (all rules checked, no early exit).
    #[test]
    fn test_all_rules_checked_no_short_circuit() {
        let mut h1 = make_hydro(1, None);
        h1.min_storage_hm3 = 200.0;
        h1.max_storage_hm3 = 100.0; // violation

        let mut h2 = make_hydro(2, None);
        h2.min_generation_mw = 500.0;
        h2.max_generation_mw = 100.0; // violation

        let data = make_data(
            vec![h1, h2],
            vec![],
            vec![],
            make_stages(vec![0]),
            vec![],
            vec![],
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(
            ctx.errors().len() >= 2,
            "both violations should be collected; got {} errors",
            ctx.errors().len()
        );
    }

    // ── Acceptance criteria tests ─────────────────────────────────────────────

    /// AC 1: Valid data produces no errors.
    #[test]
    fn test_ac1_valid_data_no_errors() {
        let geometry = vec![
            make_geom_row(1, 10.0, 100.0, 1.0),
            make_geom_row(1, 20.0, 110.0, 2.0),
            make_geom_row(1, 30.0, 120.0, 3.0),
        ];
        let fpha: Vec<FphaHyperplaneRow> = (0..3).map(|i| make_fpha_row(1, Some(0), i)).collect();
        let data = make_data(
            vec![make_hydro(1, None)],
            vec![make_thermal(1, 0.0, 500.0)],
            vec![],
            make_stages(vec![0]),
            geometry,
            fpha,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(
            !ctx.has_errors(),
            "valid data should produce no errors, got: {:?}",
            ctx.errors()
        );
    }

    /// AC 2: Hydro id=5 with inverted storage bounds produces exactly 1 InvalidValue
    /// entry whose message contains "Hydro 5" and "storage".
    #[test]
    fn test_ac2_hydro_storage_bounds_error() {
        let mut hydro = make_hydro(5, None);
        hydro.min_storage_hm3 = 200.0;
        hydro.max_storage_hm3 = 100.0;
        let data = make_data(
            vec![hydro],
            vec![],
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
        assert_eq!(relevant.len(), 1);
        let msg = &relevant[0].message;
        assert!(msg.contains("Hydro 5"), "message must contain 'Hydro 5'");
        assert!(msg.contains("storage"), "message must contain 'storage'");
    }

    /// AC 3: Cycle A->B->C->A produces at least 1 CycleDetected error.
    #[test]
    fn test_ac3_cycle_detected() {
        let hydros = vec![
            make_hydro(1, Some(2)),
            make_hydro(2, Some(3)),
            make_hydro(3, Some(1)),
        ];
        let data = make_data(hydros, vec![], vec![], make_stages(vec![0]), vec![], vec![]);
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(
            ctx.errors()
                .iter()
                .any(|e| e.kind == ErrorKind::CycleDetected),
            "should have CycleDetected error"
        );
    }

    /// AC 4: Non-monotonic volume for hydro id=3 produces BusinessRuleViolation
    /// with "Hydro 3" and "volume" in the message.
    #[test]
    fn test_ac4_geometry_non_monotonic_volume_error() {
        // Use equal volumes (10.0, 20.0, 20.0) to trigger the strict-increase check.
        let geometry = vec![
            make_geom_row(3, 10.0, 100.0, 1.0),
            make_geom_row(3, 20.0, 110.0, 1.5),
            make_geom_row(3, 20.0, 115.0, 1.6),
        ];
        let data = make_data(
            vec![make_hydro(3, None)],
            vec![],
            vec![],
            make_stages(vec![0]),
            geometry,
            vec![],
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(ctx.has_errors());
        let errors = ctx.errors();
        let relevant: Vec<_> = errors
            .iter()
            .filter(|e| e.kind == ErrorKind::BusinessRuleViolation)
            .collect();
        assert!(!relevant.is_empty(), "should have BusinessRuleViolation");
        let msg = &relevant[0].message;
        assert!(msg.contains("Hydro 3"), "must contain 'Hydro 3': {msg}");
        assert!(msg.contains("volume"), "must contain 'volume': {msg}");
    }

    /// AC 5: Empty geometry and FPHA produce no errors from rules 8-12.
    #[test]
    fn test_ac5_empty_geometry_and_fpha_no_false_positives() {
        let data = make_data(
            vec![make_hydro(1, None)],
            vec![],
            vec![],
            make_stages(vec![0]),
            vec![], // empty geometry
            vec![], // empty FPHA
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(
            !ctx.has_errors(),
            "empty geometry and FPHA should produce no errors, got: {:?}",
            ctx.errors()
        );
    }

    // ── Unit group rules (39-42) ──────────────────────────────────────────────

    /// Rule 39: hydro 1 declares unit group ids `[4, 4]` (a duplicate) and hydro
    /// 2 declares `[4, 9]` (reusing hydro 1's id 4, but locally unique). Exactly
    /// one `DuplicateId` is emitted, naming Hydro 1 and group id 4; no finding
    /// names Hydro 2 — a per-plant `HashSet` accepts the same id recurring on a
    /// different plant, which a global set would wrongly reject.
    #[test]
    fn test_duplicate_unit_group_id_is_rejected_per_plant() {
        let mut hydro1 = make_hydro(1, None);
        hydro1.unit_groups = vec![
            make_unit_group(4, 1, 0.0, 400.0, 0.0, 400.0),
            make_unit_group(4, 1, 0.0, 400.0, 0.0, 400.0),
        ];

        let mut hydro2 = make_hydro(2, None);
        hydro2.unit_groups = vec![
            make_unit_group(4, 1, 0.0, 400.0, 0.0, 400.0),
            make_unit_group(9, 1, 0.0, 400.0, 0.0, 400.0),
        ];

        let data = make_data(
            vec![hydro1, hydro2],
            vec![],
            vec![],
            make_stages(vec![0]),
            vec![],
            vec![],
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);

        let duplicates: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::DuplicateId)
            .collect();
        assert_eq!(
            duplicates.len(),
            1,
            "expected exactly 1 DuplicateId, got: {:?}",
            duplicates.iter().map(|e| &e.message).collect::<Vec<_>>()
        );
        assert!(
            duplicates[0].message.contains("Hydro 1"),
            "message should name Hydro 1, got: {}",
            duplicates[0].message
        );
        assert!(
            duplicates[0].message.contains('4'),
            "message should name group id 4, got: {}",
            duplicates[0].message
        );
        assert!(
            !ctx.errors().iter().any(|e| e.message.contains("Hydro 2")),
            "hydro 2 reusing hydro 1's group id must produce no finding, got: {:?}",
            ctx.errors()
        );
    }

    /// Rule 40: one hydro declares three unit groups — group 3 violates only
    /// `min_turbined_m3s <= max_turbined_m3s`, group 5 violates only
    /// `min_generation_mw <= max_generation_mw`, group 13 is consistent on both
    /// columns. Exactly 2 `InvalidValue` findings are emitted (each column
    /// checked independently), naming groups 3 and 5 respectively, and none
    /// names group 13.
    #[test]
    fn test_unit_group_min_exceeds_max_is_rejected_per_column() {
        let mut hydro = make_hydro(7, None);
        hydro.max_turbined_m3s = 1300.0; // == sum of group maxima below
        hydro.max_generation_mw = 900.0; // == sum of group maxima below
        hydro.unit_groups = vec![
            make_unit_group(3, 1, 0.0, 400.0, 900.0, 600.0), // turbined min > max only
            make_unit_group(5, 1, 900.0, 200.0, 0.0, 400.0), // generation min > max only
            make_unit_group(13, 1, 0.0, 300.0, 0.0, 300.0),  // consistent
        ];

        let data = make_data(
            vec![hydro],
            vec![],
            vec![],
            make_stages(vec![0]),
            vec![],
            vec![],
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);

        let invalid_values: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::InvalidValue)
            .collect();
        assert_eq!(
            invalid_values.len(),
            2,
            "expected exactly 2 InvalidValue findings, got: {:?}",
            invalid_values
                .iter()
                .map(|e| &e.message)
                .collect::<Vec<_>>()
        );
        assert!(
            invalid_values
                .iter()
                .any(|e| e.message.contains("unit group 3")
                    && e.message.contains("min_turbined_m3s")),
            "expected a min_turbined_m3s violation naming group 3, got: {:?}",
            invalid_values
                .iter()
                .map(|e| &e.message)
                .collect::<Vec<_>>()
        );
        assert!(
            invalid_values
                .iter()
                .any(|e| e.message.contains("unit group 5")
                    && e.message.contains("min_generation_mw")),
            "expected a min_generation_mw violation naming group 5, got: {:?}",
            invalid_values
                .iter()
                .map(|e| &e.message)
                .collect::<Vec<_>>()
        );
        assert!(
            !invalid_values
                .iter()
                .any(|e| e.message.contains("unit group 13")),
            "the consistent group 13 must produce no finding, got: {:?}",
            invalid_values
                .iter()
                .map(|e| &e.message)
                .collect::<Vec<_>>()
        );
    }

    /// Rule 41: hydro 9 (`max_generation_mw = 7000`, `max_turbined_m3s = 5250`)
    /// declares two groups whose `max_generation_mw` sums to 8000 (exceeds) and
    /// whose `max_turbined_m3s` sums to exactly 5250 (equals — must be
    /// accepted, since containment is "must not exceed"). Hydro 11 declares no
    /// groups at all, so the empty-group sum is `0.0` on both columns — never
    /// exceeding the plant's own maxima — the positive-path pin every plant
    /// with no declared groups exercises.
    #[test]
    fn test_envelope_containment_rejects_excess_and_accepts_equality() {
        let mut excess_hydro = make_hydro(9, None);
        excess_hydro.max_generation_mw = 7000.0;
        excess_hydro.max_turbined_m3s = 5250.0;
        excess_hydro.unit_groups = vec![
            make_unit_group(3, 1, 0.0, 3000.0, 0.0, 2000.0),
            make_unit_group(5, 1, 0.0, 5000.0, 0.0, 3250.0),
        ];

        let no_groups_hydro = make_hydro(11, None);

        let data = make_data(
            vec![excess_hydro, no_groups_hydro],
            vec![],
            vec![],
            make_stages(vec![0]),
            vec![],
            vec![],
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);

        let invalid_values: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::InvalidValue)
            .collect();
        assert_eq!(
            invalid_values.len(),
            1,
            "expected exactly 1 InvalidValue (generation envelope only), got: {:?}",
            invalid_values
                .iter()
                .map(|e| &e.message)
                .collect::<Vec<_>>()
        );
        let msg = &invalid_values[0].message;
        assert!(
            msg.contains("Hydro 9"),
            "message should name Hydro 9, got: {msg}"
        );
        assert!(
            msg.contains("max_generation_mw"),
            "message should name max_generation_mw, got: {msg}"
        );
        assert!(
            msg.contains("8000"),
            "message should name the sum 8000, got: {msg}"
        );
        assert!(
            msg.contains("7000"),
            "message should name the plant value 7000, got: {msg}"
        );
        assert!(
            !msg.contains("max_turbined_m3s"),
            "the exactly-equal turbined column must not be named, got: {msg}"
        );
        assert!(
            !ctx.errors().iter().any(|e| e.message.contains("Hydro 11")),
            "the no-declared-groups plant must produce zero findings, got: {:?}",
            ctx.errors()
        );
    }

    /// Rule 44 (the flipped-inequality mirror of rule 41 above): hydro 9
    /// (`min_generation_mw = 100`, `min_turbined_m3s = 50`) declares two groups
    /// whose `min_generation_mw` sums to 80 (below — rejected, the floor is
    /// unreachable) and whose `min_turbined_m3s` sums to exactly 50 (equals —
    /// must be accepted, since containment here is "must reach", not "must
    /// exceed"). Hydro 11 declares no groups at all, so its declared minima
    /// stay at `make_hydro`'s default `0.0`, never falling short of themselves
    /// — the positive-path pin every plant with no declared groups exercises.
    #[test]
    fn test_min_envelope_containment_rejects_shortfall_and_accepts_equality() {
        let mut shortfall_hydro = make_hydro(9, None);
        shortfall_hydro.min_generation_mw = 100.0;
        shortfall_hydro.min_turbined_m3s = 50.0;
        shortfall_hydro.unit_groups = vec![
            make_unit_group(3, 1, 30.0, 500.0, 20.0, 500.0),
            make_unit_group(5, 1, 50.0, 500.0, 30.0, 500.0),
        ];

        let no_groups_hydro = make_hydro(11, None);

        let data = make_data(
            vec![shortfall_hydro, no_groups_hydro],
            vec![],
            vec![],
            make_stages(vec![0]),
            vec![],
            vec![],
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);

        let invalid_values: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::InvalidValue)
            .collect();
        assert_eq!(
            invalid_values.len(),
            1,
            "expected exactly 1 InvalidValue (generation floor only), got: {:?}",
            invalid_values
                .iter()
                .map(|e| &e.message)
                .collect::<Vec<_>>()
        );
        let msg = &invalid_values[0].message;
        assert!(
            msg.contains("Hydro 9"),
            "message should name Hydro 9, got: {msg}"
        );
        assert!(
            msg.contains("min_generation_mw"),
            "message should name min_generation_mw, got: {msg}"
        );
        assert!(
            msg.contains("80"),
            "message should name the sum 80, got: {msg}"
        );
        assert!(
            msg.contains("100"),
            "message should name the plant value 100, got: {msg}"
        );
        assert!(
            !msg.contains("min_turbined_m3s"),
            "the exactly-equal turbined column must not be named, got: {msg}"
        );
        assert!(
            !ctx.errors().iter().any(|e| e.message.contains("Hydro 11")),
            "the no-declared-groups plant must produce zero findings, got: {:?}",
            ctx.errors()
        );
    }
}
