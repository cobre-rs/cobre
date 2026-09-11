//! Layer 5a — bound-override-family validation: block-axis rules for the six
//! Parquet families, plus block_id-agnostic stage-wide envelope rules (e.g.
//! `check_bound_raises_declared_capacity`, rule 43, and its group-axis mirror
//! `check_group_bound_raises_declared_capacity`, rule 45).
//!
//! [`resolve_bounds`](crate::resolution::resolve_bounds) resolves each family's
//! `block_id` through the four-layer bound-precedence law described alongside
//! the [applicability table](crate::constraints::bounds) and is deliberately
//! infallible: its `block_slot` helper silently skips a row whose `block_id` is
//! negative or outside `[0, n_blocks)` for the row's stage, and other callers of
//! the resolver depend on that tolerance. Rejecting the same row loudly is
//! therefore this module's job, run before `ctx.into_result()?` aborts the load
//! and the resolver ever sees it.

use std::collections::{HashMap, HashSet};

use cobre_core::{EntityId, Hydro, HydroUnitGroup};

use super::super::{ErrorKind, ValidationContext, schema::ParsedData};
use super::envelope_tolerance;

/// Per-family constants the rejection messages need: the capitalized family
/// name, the row's id-field name, the row-kind label, and the source Parquet
/// path.
struct FamilyMeta {
    family: &'static str,
    entity_label: &'static str,
    row_label: &'static str,
    file: &'static str,
}

const THERMAL: FamilyMeta = FamilyMeta {
    family: "Thermal",
    entity_label: "thermal_id",
    row_label: "thermal_bounds",
    file: "constraints/thermal_bounds.parquet",
};

const HYDRO: FamilyMeta = FamilyMeta {
    family: "Hydro",
    entity_label: "hydro_id",
    row_label: "hydro_bounds",
    file: "constraints/hydro_bounds.parquet",
};

const LINE: FamilyMeta = FamilyMeta {
    family: "Line",
    entity_label: "line_id",
    row_label: "line_bounds",
    file: "constraints/line_bounds.parquet",
};

const PUMPING: FamilyMeta = FamilyMeta {
    family: "Pumping",
    entity_label: "pumping_station_id",
    row_label: "pumping_bounds",
    file: "constraints/pumping_bounds.parquet",
};

const CONTRACT: FamilyMeta = FamilyMeta {
    family: "Contract",
    entity_label: "contract_id",
    row_label: "contract_bounds",
    file: "constraints/contract_bounds.parquet",
};

const HYDRO_UNIT_GROUP: FamilyMeta = FamilyMeta {
    family: "Hydro unit group",
    entity_label: "hydro_id",
    row_label: "hydro_unit_group_bounds",
    file: "constraints/hydro_unit_group_bounds.parquet",
};

/// `", hydro_unit_group_id={g}"` for `Some(g)`, else empty — the extra key
/// axis a group row's `entity` field carries that the five plant-level
/// families never set.
fn group_id_entity_clause(group_id: Option<i32>) -> String {
    match group_id {
        Some(g) => format!(", hydro_unit_group_id={g}"),
        None => String::new(),
    }
}

/// `", unit group {g}"` for `Some(g)`, else empty — the prose counterpart of
/// [`group_id_entity_clause`] for rejection messages.
fn group_id_message_clause(group_id: Option<i32>) -> String {
    match group_id {
        Some(g) => format!(", unit group {g}"),
        None => String::new(),
    }
}

/// Rejects a bound-override row whose `block_id` is negative or outside
/// `[0, n_blocks)` for the stage it names, across all six block-eligible
/// bound families in the fixed order thermal, hydro, line, pumping, contract,
/// hydro unit group.
///
/// A row whose `stage_id` is not a study stage is skipped without a finding —
/// that is a stage-axis defect this rule does not own.
pub(super) fn check_bound_block_id_range(data: &ParsedData, ctx: &mut ValidationContext) {
    let stage_block_counts: HashMap<i32, usize> = data
        .stages
        .stages
        .iter()
        .filter(|s| s.id >= 0)
        .map(|s| (s.id, s.blocks.len()))
        .collect();

    for row in &data.thermal_bounds {
        check_row(
            &THERMAL,
            row.thermal_id.0,
            None,
            row.stage_id,
            row.block_id,
            &stage_block_counts,
            ctx,
        );
    }
    for row in &data.hydro_bounds {
        check_row(
            &HYDRO,
            row.hydro_id.0,
            None,
            row.stage_id,
            row.block_id,
            &stage_block_counts,
            ctx,
        );
    }
    for row in &data.line_bounds {
        check_row(
            &LINE,
            row.line_id.0,
            None,
            row.stage_id,
            row.block_id,
            &stage_block_counts,
            ctx,
        );
    }
    for row in &data.pumping_bounds {
        check_row(
            &PUMPING,
            row.station_id.0,
            None,
            row.stage_id,
            row.block_id,
            &stage_block_counts,
            ctx,
        );
    }
    for row in &data.contract_bounds {
        check_row(
            &CONTRACT,
            row.contract_id.0,
            None,
            row.stage_id,
            row.block_id,
            &stage_block_counts,
            ctx,
        );
    }
    for row in &data.hydro_unit_group_bounds {
        check_row(
            &HYDRO_UNIT_GROUP,
            row.hydro_id.0,
            Some(row.hydro_unit_group_id.0),
            row.stage_id,
            row.block_id,
            &stage_block_counts,
            ctx,
        );
    }
}

fn check_row(
    meta: &FamilyMeta,
    entity_id: i32,
    group_id: Option<i32>,
    stage_id: i32,
    block_id: Option<i32>,
    counts: &HashMap<i32, usize>,
    ctx: &mut ValidationContext,
) {
    let Some(b) = block_id else { return };
    let Some(&k) = counts.get(&stage_id) else {
        return;
    };

    let b_i64 = i64::from(b);
    let k_i64 = i64::try_from(k).unwrap_or(i64::MAX);
    if b_i64 < 0 || b_i64 >= k_i64 {
        let family = meta.family;
        let entity_label = meta.entity_label;
        let row_label = meta.row_label;
        let group_entity = group_id_entity_clause(group_id);
        let group_message = group_id_message_clause(group_id);
        let entity_str = format!("{entity_label}={entity_id}{group_entity}, stage_id={stage_id}");
        ctx.add_error(
            ErrorKind::BusinessRuleViolation,
            meta.file,
            Some(entity_str),
            format!(
                "{family} {entity_id}{group_message}: {row_label} row at stage_id={stage_id} has \
                 block_id={b} but stage {stage_id} declares only {k} block(s) \
                 (valid range: 0..{k})"
            ),
        );
    }
}

/// Thermal (rule 16) and NCS (the Layer-3 referential check) already own the
/// stage-axis defect for their own families, so both are excluded here to avoid
/// double-reporting.
pub(super) fn check_bound_stage_id_range(data: &ParsedData, ctx: &mut ValidationContext) {
    let study_stage_ids: HashSet<i32> = data
        .stages
        .stages
        .iter()
        .filter(|s| s.id >= 0)
        .map(|s| s.id)
        .collect();
    let n_stages = study_stage_ids.len();

    for row in &data.hydro_bounds {
        check_row_stage_range(
            &HYDRO,
            row.hydro_id.0,
            None,
            row.stage_id,
            &study_stage_ids,
            n_stages,
            ctx,
        );
    }
    for row in &data.line_bounds {
        check_row_stage_range(
            &LINE,
            row.line_id.0,
            None,
            row.stage_id,
            &study_stage_ids,
            n_stages,
            ctx,
        );
    }
    for row in &data.pumping_bounds {
        check_row_stage_range(
            &PUMPING,
            row.station_id.0,
            None,
            row.stage_id,
            &study_stage_ids,
            n_stages,
            ctx,
        );
    }
    for row in &data.contract_bounds {
        check_row_stage_range(
            &CONTRACT,
            row.contract_id.0,
            None,
            row.stage_id,
            &study_stage_ids,
            n_stages,
            ctx,
        );
    }
    for row in &data.hydro_unit_group_bounds {
        check_row_stage_range(
            &HYDRO_UNIT_GROUP,
            row.hydro_id.0,
            Some(row.hydro_unit_group_id.0),
            row.stage_id,
            &study_stage_ids,
            n_stages,
            ctx,
        );
    }
}

fn check_row_stage_range(
    meta: &FamilyMeta,
    entity_id: i32,
    group_id: Option<i32>,
    stage_id: i32,
    study_stage_ids: &HashSet<i32>,
    n_stages: usize,
    ctx: &mut ValidationContext,
) {
    if study_stage_ids.contains(&stage_id) {
        return;
    }

    let family = meta.family;
    let entity_label = meta.entity_label;
    let row_label = meta.row_label;
    let group_entity = group_id_entity_clause(group_id);
    let group_message = group_id_message_clause(group_id);
    let entity_str = format!("{entity_label}={entity_id}{group_entity}, stage_id={stage_id}");
    ctx.add_error(
        ErrorKind::BusinessRuleViolation,
        meta.file,
        Some(entity_str),
        format!(
            "{family} {entity_id}{group_message}: {row_label} override at stage_id={stage_id} \
             is outside the study horizon [0, {n_stages})"
        ),
    );
}

/// Rejects two bound rows in the same family that set the same column for
/// the same `(entity_id, stage_id, block_id)` — [`resolve_bounds`] resolves
/// the second such row over the first (last write wins) with no diagnostic,
/// so this rule is what makes that collision loud. The key's `block_id` is
/// `None`-vs-`Some(b)` sensitive: a stage-wide row and a block row sharing an
/// `(entity, stage)` is designed usage, not a duplicate, and two rows setting
/// disjoint columns for the same `(entity, stage, block)` are legitimate
/// sparse input, never a collision — the key is scoped per column, not per
/// row.
///
/// [`resolve_bounds`]: crate::resolution::resolve_bounds
pub(super) fn check_duplicate_bound_rows(data: &ParsedData, ctx: &mut ValidationContext) {
    let mut thermal_seen = HashSet::new();
    for row in &data.thermal_bounds {
        check_row_columns(
            &THERMAL,
            row.thermal_id.0,
            None,
            row.stage_id,
            row.block_id,
            &[
                ("min_generation_mw", row.min_generation_mw),
                ("max_generation_mw", row.max_generation_mw),
                ("cost_per_mwh", row.cost_per_mwh),
            ],
            &mut thermal_seen,
            ctx,
        );
    }

    let mut hydro_seen = HashSet::new();
    for row in &data.hydro_bounds {
        check_row_columns(
            &HYDRO,
            row.hydro_id.0,
            None,
            row.stage_id,
            row.block_id,
            &[
                ("min_turbined_m3s", row.min_turbined_m3s),
                ("max_turbined_m3s", row.max_turbined_m3s),
                ("min_storage_hm3", row.min_storage_hm3),
                ("max_storage_hm3", row.max_storage_hm3),
                ("min_outflow_m3s", row.min_outflow_m3s),
                ("max_outflow_m3s", row.max_outflow_m3s),
                ("min_generation_mw", row.min_generation_mw),
                ("max_generation_mw", row.max_generation_mw),
                ("min_diversion_m3s", row.min_diversion_m3s),
                ("max_diversion_m3s", row.max_diversion_m3s),
                ("min_spillage_m3s", row.min_spillage_m3s),
                ("max_spillage_m3s", row.max_spillage_m3s),
                ("filling_min_rate_m3s", row.filling_min_rate_m3s),
                ("water_withdrawal_m3s", row.water_withdrawal_m3s),
            ],
            &mut hydro_seen,
            ctx,
        );
    }

    let mut line_seen = HashSet::new();
    for row in &data.line_bounds {
        check_row_columns(
            &LINE,
            row.line_id.0,
            None,
            row.stage_id,
            row.block_id,
            &[("direct_mw", row.direct_mw), ("reverse_mw", row.reverse_mw)],
            &mut line_seen,
            ctx,
        );
    }

    let mut pumping_seen = HashSet::new();
    for row in &data.pumping_bounds {
        check_row_columns(
            &PUMPING,
            row.station_id.0,
            None,
            row.stage_id,
            row.block_id,
            &[("min_m3s", row.min_m3s), ("max_m3s", row.max_m3s)],
            &mut pumping_seen,
            ctx,
        );
    }

    let mut contract_seen = HashSet::new();
    for row in &data.contract_bounds {
        check_row_columns(
            &CONTRACT,
            row.contract_id.0,
            None,
            row.stage_id,
            row.block_id,
            &[
                ("min_mw", row.min_mw),
                ("max_mw", row.max_mw),
                ("price_per_mwh", row.price_per_mwh),
            ],
            &mut contract_seen,
            ctx,
        );
    }

    let mut hydro_unit_group_seen = HashSet::new();
    for row in &data.hydro_unit_group_bounds {
        check_row_columns(
            &HYDRO_UNIT_GROUP,
            row.hydro_id.0,
            Some(row.hydro_unit_group_id.0),
            row.stage_id,
            row.block_id,
            &[
                ("min_turbined_m3s", row.min_turbined_m3s),
                ("max_turbined_m3s", row.max_turbined_m3s),
                ("min_generation_mw", row.min_generation_mw),
                ("max_generation_mw", row.max_generation_mw),
            ],
            &mut hydro_unit_group_seen,
            ctx,
        );
    }
}

/// `(entity_id, group_id, stage_id, block_id, column)` — `group_id` is `None`
/// for the five plant-level families, `Some(g)` for the hydro unit group
/// family, so two groups of one plant never collide on this key.
type DuplicateRowKey = (i32, Option<i32>, i32, Option<i32>, &'static str);

fn check_row_columns(
    meta: &FamilyMeta,
    entity_id: i32,
    group_id: Option<i32>,
    stage_id: i32,
    block_id: Option<i32>,
    columns: &[(&'static str, Option<f64>)],
    seen: &mut HashSet<DuplicateRowKey>,
    ctx: &mut ValidationContext,
) {
    for &(column, value) in columns {
        if value.is_none() {
            continue;
        }
        if seen.insert((entity_id, group_id, stage_id, block_id, column)) {
            continue;
        }
        let family = meta.family;
        let entity_label = meta.entity_label;
        let row_label = meta.row_label;
        let group_entity = group_id_entity_clause(group_id);
        let group_message = group_id_message_clause(group_id);
        let block_str = match block_id {
            Some(b) => format!("block_id={b}"),
            None => "stage-wide".to_string(),
        };
        ctx.add_error(
            ErrorKind::DuplicateId,
            meta.file,
            Some(format!(
                "{entity_label}={entity_id}{group_entity}, stage_id={stage_id}"
            )),
            format!(
                "{family} {entity_id}{group_message}: {row_label} sets {column} twice for \
                 stage_id={stage_id}, {block_str}"
            ),
        );
    }
}

/// One bound column with no per-block LP variable, and the reason to surface
/// in its rejection message.
///
/// Hand-synced against `cobre_core::resolved::HydroBlockOverride` /
/// `ThermalBlockOverride`'s excluded fields, which mirror
/// [`crate::constraints::bounds`]'s `## Block eligibility` table — the struct
/// field sets are the actual eligibility check; this table is not derived
/// from them at compile time. Never add `price_per_mwh` here: contract price
/// is deliberately block-eligible (`ContractBlockOverride::price_per_mwh`),
/// asymmetric with thermal `cost_per_mwh` below.
struct IneligibleColumn {
    name: &'static str,
    reason: &'static str,
}

const INELIGIBLE_HYDRO_COLUMNS: [IneligibleColumn; 4] = [
    IneligibleColumn {
        name: "min_storage_hm3",
        reason: "storage bounds are stage-level and have no per-block variable",
    },
    IneligibleColumn {
        name: "max_storage_hm3",
        reason: "storage bounds are stage-level and have no per-block variable",
    },
    IneligibleColumn {
        name: "filling_min_rate_m3s",
        reason: "the filling schedule is stage-level and has no per-block variable",
    },
    IneligibleColumn {
        name: "water_withdrawal_m3s",
        reason: "water withdrawal is stage-level and has no per-block variable",
    },
];

const INELIGIBLE_THERMAL_COLUMNS: [IneligibleColumn; 1] = [IneligibleColumn {
    name: "cost_per_mwh",
    reason: "per-block thermal cost is out of scope; thermal cost is stage-level, \
             unlike contract price, which is block-eligible",
}];

/// Rejects a `block_id` on a hydro or thermal bound column with no per-block
/// LP variable: [`resolve_bounds`] never reads the column for a block row at
/// all (`HydroBlockOverride`/`ThermalBlockOverride` carry no field for it), so
/// today the value is dropped without a diagnostic; this rule turns that into
/// a hard error. A row mixing an ineligible column with an eligible one under
/// the same `block_id` is rejected — one finding per ineligible column
/// present, never a partial application.
///
/// [`resolve_bounds`]: crate::resolution::resolve_bounds
pub(super) fn check_block_id_on_ineligible_column(data: &ParsedData, ctx: &mut ValidationContext) {
    for row in &data.hydro_bounds {
        let Some(b) = row.block_id else { continue };
        let values = [
            row.min_storage_hm3,
            row.max_storage_hm3,
            row.filling_min_rate_m3s,
            row.water_withdrawal_m3s,
        ];
        for (column, value) in INELIGIBLE_HYDRO_COLUMNS.iter().zip(values) {
            if value.is_some() {
                emit_ineligible_column_error(&HYDRO, row.hydro_id.0, row.stage_id, b, column, ctx);
            }
        }
    }

    for row in &data.thermal_bounds {
        let Some(b) = row.block_id else { continue };
        let values = [row.cost_per_mwh];
        for (column, value) in INELIGIBLE_THERMAL_COLUMNS.iter().zip(values) {
            if value.is_some() {
                emit_ineligible_column_error(
                    &THERMAL,
                    row.thermal_id.0,
                    row.stage_id,
                    b,
                    column,
                    ctx,
                );
            }
        }
    }
}

fn emit_ineligible_column_error(
    meta: &FamilyMeta,
    entity_id: i32,
    stage_id: i32,
    block_id: i32,
    column: &IneligibleColumn,
    ctx: &mut ValidationContext,
) {
    let family = meta.family;
    let entity_label = meta.entity_label;
    let row_label = meta.row_label;
    let name = column.name;
    let reason = column.reason;
    let entity_str = format!("{entity_label}={entity_id}, stage_id={stage_id}");
    ctx.add_error(
        ErrorKind::BusinessRuleViolation,
        meta.file,
        Some(entity_str),
        format!(
            "{family} {entity_id}: {row_label} row at stage_id={stage_id} sets \
             {name} with block_id={block_id}, but {reason}; remove the block_id \
             or move the value to a stage-wide row"
        ),
    );
}

/// Rejects any non-null `block_id` on a `thermal_bounds` row targeting a
/// thermal with `anticipated_config` set, whatever columns the row carries.
///
/// The anticipated commitment decision is a stage-level column, but
/// delivery-stage reconciliation compares the pinned commitment against
/// **each block's** resolved bounds; heterogeneous per-block bounds can
/// therefore make a physically valid commitment unsatisfiable at training
/// time, with an error naming neither the block axis nor this row. Rejecting
/// the row at load surfaces that failure here instead.
///
/// A row whose `thermal_id` has no matching entry in `data.thermals` is
/// skipped without a finding — a dangling reference is already reported by
/// referential validation.
pub(super) fn check_block_id_on_anticipated_thermal(
    data: &ParsedData,
    ctx: &mut ValidationContext,
) {
    let anticipated: HashMap<EntityId, bool> = data
        .thermals
        .iter()
        .map(|t| (t.id, t.anticipated_config.is_some()))
        .collect();

    for row in &data.thermal_bounds {
        let Some(b) = row.block_id else { continue };
        let Some(&is_anticipated) = anticipated.get(&row.thermal_id) else {
            continue;
        };
        if !is_anticipated {
            continue;
        }
        let thermal_id = row.thermal_id.0;
        let stage_id = row.stage_id;
        let entity_str = format!("thermal_id={thermal_id}, stage_id={stage_id}");
        ctx.add_error(
            ErrorKind::BusinessRuleViolation,
            "constraints/thermal_bounds.parquet",
            Some(entity_str),
            format!(
                "Thermal {thermal_id}: thermal_bounds row at stage_id={stage_id} \
                 carries block_id={b}, but thermal {thermal_id} declares \
                 anticipated_config; its commitment decision is stage-level while \
                 delivery-stage reconciliation compares each block's bounds, so \
                 per-block bounds can make a valid commitment unsatisfiable at \
                 training time"
            ),
        );
    }
}

/// Rejects a `hydro_bounds` row that raises `max_turbined_m3s` or
/// `max_generation_mw` above the hydro's own declared value, each column
/// checked independently. A row on an unknown `hydro_id` is skipped — Layer 3
/// referential validation already owns that rejection. The comparison is
/// strict `>`, not `>=`: restating a plant's declared maximum on an override
/// row is legitimate.
///
/// A mid-horizon uprate is expressible without raising: declare the plant at
/// its final capacity and tighten the earlier stages with `hydro_bounds` rows
/// instead (the rejection message states this remedy) — but nothing enforces
/// that the tightening rows cover every pre-uprate stage, so a missed stage
/// silently sits at full final capacity.
pub(super) fn check_bound_raises_declared_capacity(data: &ParsedData, ctx: &mut ValidationContext) {
    let declared: HashMap<EntityId, &Hydro> = data.hydros.iter().map(|h| (h.id, h)).collect();

    for row in &data.hydro_bounds {
        let Some(&hydro) = declared.get(&row.hydro_id) else {
            continue;
        };

        let columns = [
            (
                "max_turbined_m3s",
                row.max_turbined_m3s,
                hydro.max_turbined_m3s,
            ),
            (
                "max_generation_mw",
                row.max_generation_mw,
                hydro.max_generation_mw,
            ),
        ];

        for (column, value, declared_value) in columns {
            let Some(value) = value else { continue };
            let tolerance = envelope_tolerance(declared_value);
            if value > declared_value + tolerance {
                emit_raises_declared_capacity_error(
                    row.hydro_id.0,
                    row.stage_id,
                    row.block_id,
                    column,
                    declared_value,
                    value,
                    ctx,
                );
            }
        }
    }
}

fn emit_raises_declared_capacity_error(
    entity_id: i32,
    stage_id: i32,
    block_id: Option<i32>,
    column: &'static str,
    declared: f64,
    value: f64,
    ctx: &mut ValidationContext,
) {
    let family = HYDRO.family;
    let row_label = HYDRO.row_label;
    let entity_label = HYDRO.entity_label;
    let entity_str = format!("{entity_label}={entity_id}, stage_id={stage_id}");
    let block_str = match block_id {
        Some(b) => format!(", block_id={b}"),
        None => String::new(),
    };
    ctx.add_error(
        ErrorKind::InvalidValue,
        HYDRO.file,
        Some(entity_str),
        format!(
            "{family} {entity_id}: {row_label} row at stage_id={stage_id}{block_str} sets \
             {column}={value}, raising it above the plant's declared {column} ({declared}) in \
             system/hydros.json; declare the plant at its final (post-uprate) {column} instead \
             and add hydro_bounds rows tightening the earlier stages down to their true value"
        ),
    );
}

/// Rejects a `hydro_unit_group_bounds` row that raises `max_turbined_m3s` or
/// `max_generation_mw` above that GROUP's own declared value — not the
/// plant's — each column checked independently; the group-axis mirror of
/// [`check_bound_raises_declared_capacity`] (rule 43). A row on an unknown
/// `(hydro_id, hydro_unit_group_id)` pair is skipped — Layer 3 referential
/// validation already owns that rejection. The comparison is strict `>`, not
/// `>=`: restating a group's declared maximum on an override row is legitimate.
pub(super) fn check_group_bound_raises_declared_capacity(
    data: &ParsedData,
    ctx: &mut ValidationContext,
) {
    let declared: HashMap<(EntityId, EntityId), &HydroUnitGroup> = data
        .hydros
        .iter()
        .flat_map(|h| h.unit_groups.iter().map(move |g| ((h.id, g.id), g)))
        .collect();

    for row in &data.hydro_unit_group_bounds {
        let Some(&group) = declared.get(&(row.hydro_id, row.hydro_unit_group_id)) else {
            continue;
        };

        let columns = [
            (
                "max_turbined_m3s",
                row.max_turbined_m3s,
                group.max_turbined_m3s,
            ),
            (
                "max_generation_mw",
                row.max_generation_mw,
                group.max_generation_mw,
            ),
        ];

        for (column, value, declared_value) in columns {
            let Some(value) = value else { continue };
            let tolerance = envelope_tolerance(declared_value);
            if value > declared_value + tolerance {
                emit_group_raises_declared_capacity_error(
                    row.hydro_id.0,
                    row.hydro_unit_group_id.0,
                    row.stage_id,
                    row.block_id,
                    column,
                    declared_value,
                    value,
                    ctx,
                );
            }
        }
    }
}

fn emit_group_raises_declared_capacity_error(
    hydro_id: i32,
    group_id: i32,
    stage_id: i32,
    block_id: Option<i32>,
    column: &'static str,
    declared: f64,
    value: f64,
    ctx: &mut ValidationContext,
) {
    let family = HYDRO_UNIT_GROUP.family;
    let row_label = HYDRO_UNIT_GROUP.row_label;
    let entity_label = HYDRO_UNIT_GROUP.entity_label;
    let group_entity = group_id_entity_clause(Some(group_id));
    let group_message = group_id_message_clause(Some(group_id));
    let entity_str = format!("{entity_label}={hydro_id}{group_entity}, stage_id={stage_id}");
    let block_str = match block_id {
        Some(b) => format!(", block_id={b}"),
        None => String::new(),
    };
    ctx.add_error(
        ErrorKind::InvalidValue,
        HYDRO_UNIT_GROUP.file,
        Some(entity_str),
        format!(
            "{family} {hydro_id}{group_message}: {row_label} row at stage_id={stage_id}{block_str} \
             sets {column}={value}, raising it above the group's declared {column} ({declared}) in \
             system/hydros.json; declare the group at its final (post-uprate) {column} instead and \
             add hydro_unit_group_bounds rows tightening the earlier stages down to their true value"
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
    use std::collections::HashSet;

    use cobre_core::temporal::PolicyGraphType;
    use cobre_core::{
        AnticipatedCommitmentHistory, AnticipatedConfig, EntityId, HorizonGraph, Hydro, Thermal,
    };

    use chrono::NaiveDate;

    use super::super::test_support::*;
    use super::super::validate_semantic_hydro_thermal;
    use crate::ValidationEntry;
    use crate::constraints::{
        ContractBoundsRow, HydroBoundsRow, HydroUnitGroupBoundsRow, LineBoundsRow,
        PumpingBoundsRow, ThermalBoundsRow,
    };
    use crate::stages::StagesData;
    use crate::validation::schema::ParsedData;
    use crate::validation::{ErrorKind, ValidationContext};

    const BOUND_FILES: [&str; 5] = [
        "constraints/thermal_bounds.parquet",
        "constraints/hydro_bounds.parquet",
        "constraints/line_bounds.parquet",
        "constraints/pumping_bounds.parquet",
        "constraints/contract_bounds.parquet",
    ];

    /// Stage 0 declares 3 blocks, stage 1 declares 2 — the two counts must
    /// stay distinct so a global-maximum bug and a per-stage lookup diverge.
    fn two_stage_study_stages() -> StagesData {
        // Contiguous 30-day (720 h) stages so the block-hour totals and the
        // calendar-date span agree — the alignment StageCalendar coverage needs.
        let mut stage_0 = make_stage_with_blocks(0, 3);
        stage_0.start_date = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();
        stage_0.end_date = NaiveDate::from_ymd_opt(2024, 1, 31).unwrap();
        let mut stage_1 = make_stage_with_blocks(1, 2);
        stage_1.start_date = NaiveDate::from_ymd_opt(2024, 1, 31).unwrap();
        stage_1.end_date = NaiveDate::from_ymd_opt(2024, 3, 1).unwrap();
        StagesData {
            openings_declared: std::collections::HashSet::new(),
            stages: vec![stage_0, stage_1],
            policy_graph: HorizonGraph {
                stage_discount_rate_overrides: std::collections::HashMap::new(),
                graph_type: PolicyGraphType::FiniteHorizon,
                annual_discount_rate: 0.06,
                transitions: vec![],
                nodes: Vec::new(),
                season_map: None,
            },
        }
    }

    /// Two hydros whose declared `max_turbined_m3s`/`max_generation_mw` differ
    /// from each other and from every hydro/stage id: hydro 1 (600/400) is the
    /// entity under test; hydro 2 (1000/700) exists so a positional
    /// `data.hydros[id]` lookup (ids are 1-based, positions 0-based) would
    /// compare a hydro-1 row against hydro 2's higher declared value instead
    /// and miss a violation.
    fn two_hydro_capacity_study() -> Vec<Hydro> {
        vec![
            Hydro {
                max_turbined_m3s: 600.0,
                max_generation_mw: 400.0,
                ..make_hydro(1, None)
            },
            Hydro {
                max_turbined_m3s: 1000.0,
                max_generation_mw: 700.0,
                ..make_hydro(2, None)
            },
        ]
    }

    fn thermal_row(id: i32, stage_id: i32, block_id: Option<i32>) -> ThermalBoundsRow {
        ThermalBoundsRow {
            thermal_id: EntityId::from(id),
            stage_id,
            min_generation_mw: None,
            max_generation_mw: None,
            cost_per_mwh: None,
            block_id,
        }
    }

    fn hydro_row(id: i32, stage_id: i32, block_id: Option<i32>) -> HydroBoundsRow {
        HydroBoundsRow {
            hydro_id: EntityId::from(id),
            stage_id,
            block_id,
            ..Default::default()
        }
    }

    fn line_row(id: i32, stage_id: i32, block_id: Option<i32>) -> LineBoundsRow {
        LineBoundsRow {
            line_id: EntityId::from(id),
            stage_id,
            direct_mw: None,
            reverse_mw: None,
            block_id,
        }
    }

    fn pumping_row(id: i32, stage_id: i32, block_id: Option<i32>) -> PumpingBoundsRow {
        PumpingBoundsRow {
            station_id: EntityId::from(id),
            stage_id,
            min_m3s: None,
            max_m3s: None,
            block_id,
        }
    }

    fn contract_row(id: i32, stage_id: i32, block_id: Option<i32>) -> ContractBoundsRow {
        ContractBoundsRow {
            contract_id: EntityId::from(id),
            stage_id,
            min_mw: None,
            max_mw: None,
            price_per_mwh: None,
            block_id,
        }
    }

    fn group_bounds_row(
        hydro_id: i32,
        group_id: i32,
        stage_id: i32,
        block_id: Option<i32>,
    ) -> HydroUnitGroupBoundsRow {
        HydroUnitGroupBoundsRow {
            hydro_id: EntityId::from(hydro_id),
            hydro_unit_group_id: EntityId::from(group_id),
            stage_id,
            min_turbined_m3s: None,
            max_turbined_m3s: None,
            min_generation_mw: None,
            max_generation_mw: None,
            block_id,
        }
    }

    /// Runs the full Layer 5a dispatch and returns only the findings of `kind`
    /// raised against `constraints/hydro_unit_group_bounds.parquet` — the
    /// group family sits outside [`BOUND_FILES`], so [`bound_errors_of_kind`]
    /// cannot see it.
    fn group_bound_errors_of_kind(data: &ParsedData, kind: ErrorKind) -> Vec<ValidationEntry> {
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(data, &mut ctx);
        ctx.errors()
            .iter()
            .filter(|e| {
                e.kind == kind
                    && e.file == std::path::Path::new("constraints/hydro_unit_group_bounds.parquet")
            })
            .map(|e| (*e).clone())
            .collect()
    }

    /// Runs the full Layer 5a dispatch and returns only the findings of `kind`
    /// raised against one of the five bound parquets.
    fn bound_errors_of_kind(data: &ParsedData, kind: ErrorKind) -> Vec<ValidationEntry> {
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(data, &mut ctx);
        ctx.errors()
            .iter()
            .filter(|e| e.kind == kind && BOUND_FILES.contains(&e.file.to_string_lossy().as_ref()))
            .map(|e| (*e).clone())
            .collect()
    }

    fn bound_range_errors(data: &ParsedData) -> Vec<ValidationEntry> {
        bound_errors_of_kind(data, ErrorKind::BusinessRuleViolation)
    }

    fn duplicate_errors(data: &ParsedData) -> Vec<ValidationEntry> {
        bound_errors_of_kind(data, ErrorKind::DuplicateId)
    }

    /// No other Layer 5a rule emits `InvalidValue` against one of the five
    /// bound parquets, so this isolates `check_bound_raises_declared_capacity`
    /// from the plant-level `InvalidValue` rules that target
    /// `system/hydros.json` instead (e.g. rule 3's turbine-bound ordering).
    fn capacity_raise_errors(data: &ParsedData) -> Vec<ValidationEntry> {
        bound_errors_of_kind(data, ErrorKind::InvalidValue)
    }

    /// The group-family counterpart of [`capacity_raise_errors`]: isolates
    /// `check_group_bound_raises_declared_capacity` (rule 45) from the
    /// `BusinessRuleViolation`/`DuplicateId` findings the other group-family
    /// rules raise against the same file.
    fn group_capacity_raise_errors(data: &ParsedData) -> Vec<ValidationEntry> {
        group_bound_errors_of_kind(data, ErrorKind::InvalidValue)
    }

    #[test]
    fn test_block_id_within_declared_block_count_accepted() {
        let mut data = make_data(
            vec![],
            vec![],
            vec![],
            two_stage_study_stages(),
            vec![],
            vec![],
        );
        // Two rows per family, one at each stage, so both declared block
        // counts (3 and 2) and two distinct entities are exercised.
        data.thermal_bounds = vec![thermal_row(1, 0, Some(2)), thermal_row(2, 1, Some(1))];
        data.hydro_bounds = vec![hydro_row(1, 0, Some(2)), hydro_row(2, 1, Some(1))];
        data.line_bounds = vec![line_row(1, 0, Some(2)), line_row(2, 1, Some(1))];
        data.pumping_bounds = vec![pumping_row(1, 0, Some(2)), pumping_row(2, 1, Some(1))];
        data.contract_bounds = vec![contract_row(1, 0, Some(2)), contract_row(2, 1, Some(1))];

        assert!(
            bound_range_errors(&data).is_empty(),
            "in-range block rows must not be rejected"
        );
    }

    #[test]
    fn test_block_id_uses_per_stage_not_global_block_count() {
        let mut data = make_data(
            vec![],
            vec![],
            vec![],
            two_stage_study_stages(),
            vec![],
            vec![],
        );
        data.hydro_bounds = vec![
            hydro_row(1, 0, Some(1)), // valid: stage 0 declares 3 blocks (0..3)
            hydro_row(2, 1, Some(2)), // invalid: stage 1 declares 2 blocks (0..2);
                                      // a global-maximum reader (max = 3) would wrongly accept this
        ];

        let errors = bound_range_errors(&data);
        assert_eq!(
            errors.len(),
            1,
            "expected exactly one error, got: {errors:?}"
        );
        let err = &errors[0];
        assert!(
            err.message.contains("block_id=2"),
            "message: {}",
            err.message
        );
        assert!(err.message.contains("stage 1"), "message: {}", err.message);
        assert!(err.message.contains("0..2"), "message: {}", err.message);
        assert_eq!(
            err.entity.as_deref(),
            Some("hydro_id=2, stage_id=1"),
            "entity string mismatch: {:?}",
            err.entity
        );
    }

    #[test]
    fn test_all_five_bound_families_are_checked() {
        let mut data = make_data(
            vec![],
            vec![],
            vec![],
            StagesData {
                openings_declared: std::collections::HashSet::new(),
                stages: vec![make_stage_with_blocks(0, 1), make_stage_with_blocks(1, 3)],
                policy_graph: HorizonGraph {
                    stage_discount_rate_overrides: std::collections::HashMap::new(),
                    graph_type: PolicyGraphType::FiniteHorizon,
                    annual_discount_rate: 0.06,
                    transitions: vec![],
                    nodes: Vec::new(),
                    season_map: None,
                },
            },
            vec![],
            vec![],
        );
        // Stage 0 declares 1 block (valid range 0..1); block_id = 1 is out of
        // range in every family. Stage 1 (3 blocks) exists only to vary the
        // fixture's per-stage block counts; no row references it.
        data.thermal_bounds = vec![thermal_row(1, 0, Some(1))];
        data.hydro_bounds = vec![hydro_row(1, 0, Some(1))];
        data.line_bounds = vec![line_row(1, 0, Some(1))];
        data.pumping_bounds = vec![pumping_row(1, 0, Some(1))];
        data.contract_bounds = vec![contract_row(1, 0, Some(1))];

        let errors = bound_range_errors(&data);
        assert_eq!(
            errors.len(),
            5,
            "expected one error per family, got: {errors:?}"
        );
        let files: HashSet<String> = errors
            .iter()
            .map(|e| e.file.to_string_lossy().into_owned())
            .collect();
        let expected: HashSet<String> = BOUND_FILES.iter().map(|s| (*s).to_string()).collect();
        assert_eq!(
            files, expected,
            "expected all five family files to be present"
        );
    }

    #[test]
    fn test_negative_block_id_and_out_of_horizon_stage_both_rejected() {
        let mut data = make_data(
            vec![],
            vec![],
            vec![],
            two_stage_study_stages(),
            vec![],
            vec![],
        );
        data.line_bounds = vec![
            line_row(1, 0, Some(-1)), // invalid: negative block_id
            line_row(2, 1, Some(1)),  // valid: stage 1 declares 2 blocks (0..2)
        ];
        data.contract_bounds = vec![
            contract_row(1, 99, Some(0)), // invalid: stage_id 99 is outside the study horizon
            contract_row(2, 0, Some(2)),  // valid: stage 0 declares 3 blocks (0..3)
        ];

        let errors = bound_range_errors(&data);
        assert_eq!(
            errors.len(),
            2,
            "expected exactly two errors, got: {errors:?}"
        );
        let block_error = errors
            .iter()
            .find(|e| e.message.contains("block_id=-1"))
            .unwrap_or_else(|| panic!("expected a block_id range finding: {errors:?}"));
        assert!(
            block_error.message.contains("valid range: 0.."),
            "message: {}",
            block_error.message
        );
        let stage_error = errors
            .iter()
            .find(|e| e.message.contains("stage_id=99"))
            .unwrap_or_else(|| panic!("expected a stage_id range finding: {errors:?}"));
        assert!(
            stage_error.message.contains("outside the study horizon"),
            "message: {}",
            stage_error.message
        );
    }

    // ── stage-axis rule (check_bound_stage_id_range, rule 49) ────────────────

    #[test]
    fn test_all_five_bound_families_in_horizon_stage_accepted() {
        let mut data = make_data(
            vec![],
            vec![],
            vec![],
            two_stage_study_stages(),
            vec![],
            vec![],
        );
        data.hydro_bounds = vec![hydro_row(1, 0, None)];
        data.line_bounds = vec![line_row(1, 1, None)];
        data.pumping_bounds = vec![pumping_row(1, 0, None)];
        data.contract_bounds = vec![contract_row(1, 1, None)];
        data.hydro_unit_group_bounds = vec![group_bounds_row(1, 2, 0, None)];

        assert!(
            bound_range_errors(&data).is_empty(),
            "in-horizon stage rows must not be rejected"
        );
        assert!(
            group_bound_errors_of_kind(&data, ErrorKind::BusinessRuleViolation).is_empty(),
            "in-horizon stage group-family row must not be rejected"
        );
    }

    #[test]
    fn test_hydro_bounds_out_of_horizon_and_negative_stage_rejected() {
        let mut data = make_data(
            vec![],
            vec![],
            vec![],
            two_stage_study_stages(),
            vec![],
            vec![],
        );
        data.hydro_bounds = vec![hydro_row(1, 2, None), hydro_row(1, -1, None)];

        let errors = bound_range_errors(&data);
        assert_eq!(
            errors.len(),
            2,
            "expected exactly two errors, got: {errors:?}"
        );
        assert!(
            errors.iter().any(|e| e.message.contains("Hydro 1")
                && e.message.contains("stage_id=2")
                && e.message.contains("outside the study horizon")),
            "expected a stage_id=2 finding: {errors:?}"
        );
        assert!(
            errors.iter().any(|e| e.message.contains("Hydro 1")
                && e.message.contains("stage_id=-1")
                && e.message.contains("outside the study horizon")),
            "expected a stage_id=-1 finding: {errors:?}"
        );
    }

    #[test]
    fn test_line_bounds_out_of_horizon_and_negative_stage_rejected() {
        let mut data = make_data(
            vec![],
            vec![],
            vec![],
            two_stage_study_stages(),
            vec![],
            vec![],
        );
        data.line_bounds = vec![line_row(1, 2, None), line_row(1, -1, None)];

        let errors = bound_range_errors(&data);
        assert_eq!(
            errors.len(),
            2,
            "expected exactly two errors, got: {errors:?}"
        );
        assert!(
            errors.iter().any(|e| e.message.contains("Line 1")
                && e.message.contains("stage_id=2")
                && e.message.contains("outside the study horizon")),
            "expected a stage_id=2 finding: {errors:?}"
        );
        assert!(
            errors.iter().any(|e| e.message.contains("Line 1")
                && e.message.contains("stage_id=-1")
                && e.message.contains("outside the study horizon")),
            "expected a stage_id=-1 finding: {errors:?}"
        );
    }

    #[test]
    fn test_pumping_bounds_out_of_horizon_and_negative_stage_rejected() {
        let mut data = make_data(
            vec![],
            vec![],
            vec![],
            two_stage_study_stages(),
            vec![],
            vec![],
        );
        data.pumping_bounds = vec![pumping_row(1, 2, None), pumping_row(1, -1, None)];

        let errors = bound_range_errors(&data);
        assert_eq!(
            errors.len(),
            2,
            "expected exactly two errors, got: {errors:?}"
        );
        assert!(
            errors.iter().any(|e| e.message.contains("Pumping 1")
                && e.message.contains("stage_id=2")
                && e.message.contains("outside the study horizon")),
            "expected a stage_id=2 finding: {errors:?}"
        );
        assert!(
            errors.iter().any(|e| e.message.contains("Pumping 1")
                && e.message.contains("stage_id=-1")
                && e.message.contains("outside the study horizon")),
            "expected a stage_id=-1 finding: {errors:?}"
        );
    }

    #[test]
    fn test_contract_bounds_out_of_horizon_and_negative_stage_rejected() {
        let mut data = make_data(
            vec![],
            vec![],
            vec![],
            two_stage_study_stages(),
            vec![],
            vec![],
        );
        data.contract_bounds = vec![contract_row(1, 2, None), contract_row(1, -1, None)];

        let errors = bound_range_errors(&data);
        assert_eq!(
            errors.len(),
            2,
            "expected exactly two errors, got: {errors:?}"
        );
        assert!(
            errors.iter().any(|e| e.message.contains("Contract 1")
                && e.message.contains("stage_id=2")
                && e.message.contains("outside the study horizon")),
            "expected a stage_id=2 finding: {errors:?}"
        );
        assert!(
            errors.iter().any(|e| e.message.contains("Contract 1")
                && e.message.contains("stage_id=-1")
                && e.message.contains("outside the study horizon")),
            "expected a stage_id=-1 finding: {errors:?}"
        );
    }

    #[test]
    fn test_out_of_horizon_stage_with_block_id_produces_only_stage_finding() {
        let mut data = make_data(
            vec![],
            vec![],
            vec![],
            two_stage_study_stages(),
            vec![],
            vec![],
        );
        data.contract_bounds = vec![contract_row(1, 99, Some(0))];

        let errors = bound_range_errors(&data);
        assert_eq!(
            errors.len(),
            1,
            "expected exactly one error, got: {errors:?}"
        );
        let err = &errors[0];
        assert!(
            err.file.to_string_lossy().as_ref() == "constraints/contract_bounds.parquet",
            "file: {:?}",
            err.file
        );
        assert!(
            err.message.contains("stage_id=99"),
            "message: {}",
            err.message
        );
        assert!(
            !err.message.contains("block_id"),
            "message must not mention block_id: {}",
            err.message
        );
    }

    #[test]
    fn test_disjoint_columns_on_same_entity_stage_block_accepted() {
        let mut data = make_data(
            vec![],
            vec![],
            vec![],
            two_stage_study_stages(),
            vec![],
            vec![],
        );
        // Same (hydro_id, stage_id, block_id); each row sets a different
        // column, so the two must not collide.
        data.hydro_bounds = vec![
            HydroBoundsRow {
                min_turbined_m3s: Some(10.0),
                ..hydro_row(1, 0, Some(1))
            },
            HydroBoundsRow {
                max_turbined_m3s: Some(20.0),
                ..hydro_row(1, 0, Some(1))
            },
        ];

        assert!(
            duplicate_errors(&data).is_empty(),
            "disjoint columns on the same (entity, stage, block) must not collide"
        );
    }

    #[test]
    fn test_stage_wide_and_block_row_on_same_column_accepted() {
        let mut data = make_data(
            vec![],
            vec![],
            vec![],
            two_stage_study_stages(),
            vec![],
            vec![],
        );
        // Same (hydro_id, stage_id), same column; one row is stage-wide
        // (block_id = None), the other targets block 1 — the precedence
        // law's designed usage, not a duplicate.
        data.hydro_bounds = vec![
            HydroBoundsRow {
                max_turbined_m3s: Some(10.0),
                ..hydro_row(1, 0, None)
            },
            HydroBoundsRow {
                max_turbined_m3s: Some(20.0),
                ..hydro_row(1, 0, Some(1))
            },
        ];

        assert!(
            duplicate_errors(&data).is_empty(),
            "a stage-wide row and a block row on the same column must not collide"
        );
    }

    #[test]
    fn test_duplicate_column_rejected_for_block_and_stage_wide_rows() {
        let mut data = make_data(
            vec![],
            vec![],
            vec![],
            two_stage_study_stages(),
            vec![],
            vec![],
        );
        // Two entities, two families, two stages: a block-row collision
        // (hydro, stage 0, block 1) and a stage-wide collision (contract,
        // stage 1) must each surface independently.
        data.hydro_bounds = vec![
            HydroBoundsRow {
                max_turbined_m3s: Some(10.0),
                ..hydro_row(1, 0, Some(1))
            },
            HydroBoundsRow {
                max_turbined_m3s: Some(20.0),
                ..hydro_row(1, 0, Some(1))
            },
        ];
        data.contract_bounds = vec![
            ContractBoundsRow {
                price_per_mwh: Some(30.0),
                ..contract_row(4, 1, None)
            },
            ContractBoundsRow {
                price_per_mwh: Some(40.0),
                ..contract_row(4, 1, None)
            },
        ];

        let errors = duplicate_errors(&data);
        assert_eq!(
            errors.len(),
            2,
            "expected exactly two duplicate errors, got: {errors:?}"
        );
        assert!(
            errors
                .iter()
                .any(|e| e.message.contains("max_turbined_m3s") && e.message.contains("block_id=1")),
            "expected a block-row collision naming max_turbined_m3s and block_id=1: {errors:?}"
        );
        assert!(
            errors
                .iter()
                .any(|e| e.message.contains("price_per_mwh") && e.message.contains("stage-wide")),
            "expected a stage-wide collision naming price_per_mwh: {errors:?}"
        );
    }

    #[test]
    fn test_key_discriminates_entity_stage_block_and_column() {
        let mut data = make_data(
            vec![],
            vec![],
            vec![],
            two_stage_study_stages(),
            vec![],
            vec![],
        );
        // Baseline (1, 0, Some(0)) plus one row differing in exactly one of
        // stage_id, entity_id, and block_id — a key missing any one of
        // entity_id, stage_id, or block_id collapses two of these onto the
        // same slot and false-reports.
        data.hydro_bounds = vec![
            HydroBoundsRow {
                max_turbined_m3s: Some(1.0),
                ..hydro_row(1, 0, Some(0))
            },
            HydroBoundsRow {
                max_turbined_m3s: Some(2.0),
                ..hydro_row(1, 1, Some(0))
            },
            HydroBoundsRow {
                max_turbined_m3s: Some(3.0),
                ..hydro_row(2, 0, Some(0))
            },
            HydroBoundsRow {
                max_turbined_m3s: Some(4.0),
                ..hydro_row(1, 0, Some(1))
            },
        ];

        let errors = duplicate_errors(&data);
        assert!(
            errors.is_empty(),
            "a key discriminating entity, stage, and block must not false-report: {errors:?}"
        );
    }

    /// Two rows for the same `(hydro_id, stage_id, block_id)` both setting
    /// `min_spillage_m3s` collide — pins that the widened duplicate-row column
    /// list registers the new spillage/diversion axes, not just the
    /// pre-existing seven.
    #[test]
    fn test_duplicate_min_spillage_m3s_rejected() {
        let mut data = make_data(
            vec![],
            vec![],
            vec![],
            two_stage_study_stages(),
            vec![],
            vec![],
        );
        data.hydro_bounds = vec![
            HydroBoundsRow {
                min_spillage_m3s: Some(1.0),
                ..hydro_row(1, 0, Some(0))
            },
            HydroBoundsRow {
                min_spillage_m3s: Some(2.0),
                ..hydro_row(1, 0, Some(0))
            },
        ];

        let errors = duplicate_errors(&data);
        assert_eq!(
            errors.len(),
            1,
            "expected exactly one error, got: {errors:?}"
        );
        assert!(
            errors[0].message.contains("min_spillage_m3s"),
            "message: {}",
            errors[0].message
        );
    }

    // ── group family (hydro_unit_group_bounds) ───────────────────────────────

    #[test]
    fn test_group_bounds_block_id_out_of_range() {
        let mut data = make_data(
            vec![],
            vec![],
            vec![],
            StagesData {
                openings_declared: std::collections::HashSet::new(),
                stages: vec![make_stage_with_blocks(0, 2)],
                policy_graph: HorizonGraph {
                    stage_discount_rate_overrides: std::collections::HashMap::new(),
                    graph_type: PolicyGraphType::FiniteHorizon,
                    annual_discount_rate: 0.06,
                    transitions: vec![],
                    nodes: Vec::new(),
                    season_map: None,
                },
            },
            vec![],
            vec![],
        );
        data.hydro_unit_group_bounds = vec![group_bounds_row(7, 3, 0, Some(5))];

        let errors = group_bound_errors_of_kind(&data, ErrorKind::BusinessRuleViolation);
        assert_eq!(
            errors.len(),
            1,
            "expected exactly one error, got: {errors:?}"
        );
        let msg = &errors[0].message;
        assert!(msg.contains("block_id=5"), "message: {msg}");
        assert!(msg.contains("2 block(s)"), "message: {msg}");
        assert!(msg.contains("unit group 3"), "message: {msg}");
    }

    /// Pins the whole point of the widened duplicate key: two DIFFERENT
    /// groups of one plant overriding the same column at the same
    /// `(stage_id, block_id)` is the file's designed usage, not a collision.
    #[test]
    fn test_group_bounds_two_groups_one_plant_is_not_a_duplicate() {
        let mut data = make_data(
            vec![],
            vec![],
            vec![],
            two_stage_study_stages(),
            vec![],
            vec![],
        );
        data.hydro_unit_group_bounds = vec![
            HydroUnitGroupBoundsRow {
                max_turbined_m3s: Some(10.0),
                ..group_bounds_row(1, 2, 0, Some(1))
            },
            HydroUnitGroupBoundsRow {
                max_turbined_m3s: Some(20.0),
                ..group_bounds_row(1, 5, 0, Some(1))
            },
        ];

        let errors = group_bound_errors_of_kind(&data, ErrorKind::DuplicateId);
        assert!(
            errors.is_empty(),
            "two different groups of one plant setting the same column at the \
             same stage/block must not collide: {errors:?}"
        );
    }

    #[test]
    fn test_group_bounds_duplicate_column_rejected_disjoint_columns_allowed() {
        let mut data = make_data(
            vec![],
            vec![],
            vec![],
            two_stage_study_stages(),
            vec![],
            vec![],
        );
        // Two rows on the same (hydro, group, stage, block) both set
        // min_generation_mw (must collide exactly once); a third row on that
        // same key sets only the disjoint max_generation_mw (must not add a
        // second finding).
        data.hydro_unit_group_bounds = vec![
            HydroUnitGroupBoundsRow {
                min_generation_mw: Some(1.0),
                ..group_bounds_row(1, 2, 0, Some(1))
            },
            HydroUnitGroupBoundsRow {
                min_generation_mw: Some(2.0),
                ..group_bounds_row(1, 2, 0, Some(1))
            },
            HydroUnitGroupBoundsRow {
                max_generation_mw: Some(3.0),
                ..group_bounds_row(1, 2, 0, Some(1))
            },
        ];

        let errors = group_bound_errors_of_kind(&data, ErrorKind::DuplicateId);
        assert_eq!(
            errors.len(),
            1,
            "expected exactly one error, got: {errors:?}"
        );
        assert!(
            errors[0].message.contains("min_generation_mw"),
            "message: {}",
            errors[0].message
        );
    }

    #[test]
    fn test_group_bounds_out_of_horizon_and_negative_stage_rejected() {
        let mut data = make_data(
            vec![],
            vec![],
            vec![],
            two_stage_study_stages(),
            vec![],
            vec![],
        );
        data.hydro_unit_group_bounds = vec![
            group_bounds_row(1, 3, 2, None),
            group_bounds_row(1, 3, -1, None),
        ];

        let errors = group_bound_errors_of_kind(&data, ErrorKind::BusinessRuleViolation);
        assert_eq!(
            errors.len(),
            2,
            "expected exactly two errors, got: {errors:?}"
        );
        assert!(
            errors
                .iter()
                .any(|e| e.message.contains("Hydro unit group 1")
                    && e.message.contains("unit group 3")
                    && e.message.contains("stage_id=2")
                    && e.message.contains("outside the study horizon")),
            "expected a stage_id=2 finding: {errors:?}"
        );
        assert!(
            errors
                .iter()
                .any(|e| e.message.contains("Hydro unit group 1")
                    && e.message.contains("unit group 3")
                    && e.message.contains("stage_id=-1")
                    && e.message.contains("outside the study horizon")),
            "expected a stage_id=-1 finding: {errors:?}"
        );
    }

    // ── check_block_id_on_ineligible_column ──────────────────────────────────

    #[test]
    fn test_stage_level_hydro_columns_reject_block_id() {
        let mut data = make_data(
            vec![],
            vec![],
            vec![],
            two_stage_study_stages(),
            vec![],
            vec![],
        );
        // Each of the four ineligible hydro columns fires exactly once, spread
        // across both hydros (1, 2) and both stages (0, 1) — a fixture on one
        // entity/stage could not distinguish this rule from one that reports
        // unconditionally.
        data.hydro_bounds = vec![
            HydroBoundsRow {
                min_storage_hm3: Some(10.0),
                ..hydro_row(1, 0, Some(0))
            },
            HydroBoundsRow {
                max_storage_hm3: Some(20.0),
                ..hydro_row(2, 1, Some(0))
            },
            HydroBoundsRow {
                filling_min_rate_m3s: Some(1.0),
                ..hydro_row(1, 1, Some(0))
            },
            HydroBoundsRow {
                water_withdrawal_m3s: Some(2.0),
                ..hydro_row(2, 0, Some(0))
            },
        ];

        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        let errors = ctx.errors();
        assert_eq!(
            errors.len(),
            4,
            "expected exactly four errors, got: {errors:?}"
        );
        for (column, hydro_id, stage_id) in [
            ("min_storage_hm3", 1, 0),
            ("max_storage_hm3", 2, 1),
            ("filling_min_rate_m3s", 1, 1),
            ("water_withdrawal_m3s", 2, 0),
        ] {
            assert!(
                errors.iter().any(|e| e.message.contains(column)
                    && e.message.contains(&format!("Hydro {hydro_id}"))
                    && e.message.contains(&format!("stage_id={stage_id}"))),
                "expected an error naming {column} on hydro {hydro_id} stage {stage_id}: {errors:?}"
            );
        }
    }

    #[test]
    fn test_thermal_cost_rejected_while_contract_price_accepted() {
        let mut data = make_data(
            vec![],
            vec![],
            vec![],
            two_stage_study_stages(),
            vec![],
            vec![],
        );
        data.thermal_bounds = vec![ThermalBoundsRow {
            cost_per_mwh: Some(50.0),
            ..thermal_row(3, 0, Some(2))
        }];
        data.contract_bounds = vec![ContractBoundsRow {
            price_per_mwh: Some(30.0),
            ..contract_row(4, 1, Some(1))
        }];

        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        let errors = ctx.errors();

        let cost_errors: Vec<_> = errors
            .iter()
            .filter(|e| e.message.contains("cost_per_mwh"))
            .collect();
        assert_eq!(
            cost_errors.len(),
            1,
            "expected exactly one error naming cost_per_mwh, got: {errors:?}"
        );
        let msg = &cost_errors[0].message;
        assert!(
            msg.contains("thermal") || msg.contains("Thermal"),
            "message must name thermal specifically: {msg}"
        );
        assert!(
            !errors.iter().any(|e| e.message.contains("price_per_mwh")),
            "no error may name price_per_mwh (the block-eligible contract column): {errors:?}"
        );
    }

    #[test]
    fn test_block_eligible_columns_accept_block_id() {
        let mut data = make_data(
            vec![],
            vec![],
            vec![],
            two_stage_study_stages(),
            vec![],
            vec![],
        );
        // One row per block-eligible column across all five families —
        // every column NOT on the ineligible list must be accepted.
        data.hydro_bounds = vec![
            HydroBoundsRow {
                min_turbined_m3s: Some(1.0),
                ..hydro_row(1, 0, Some(0))
            },
            HydroBoundsRow {
                max_turbined_m3s: Some(2.0),
                ..hydro_row(1, 0, Some(1))
            },
            HydroBoundsRow {
                min_outflow_m3s: Some(3.0),
                ..hydro_row(1, 0, Some(2))
            },
            HydroBoundsRow {
                max_outflow_m3s: Some(4.0),
                ..hydro_row(2, 1, Some(0))
            },
            HydroBoundsRow {
                min_generation_mw: Some(5.0),
                ..hydro_row(2, 1, Some(1))
            },
            HydroBoundsRow {
                max_generation_mw: Some(6.0),
                ..hydro_row(2, 0, Some(0))
            },
            HydroBoundsRow {
                max_diversion_m3s: Some(7.0),
                ..hydro_row(2, 0, Some(1))
            },
        ];
        data.thermal_bounds = vec![
            ThermalBoundsRow {
                min_generation_mw: Some(1.0),
                ..thermal_row(1, 0, Some(0))
            },
            ThermalBoundsRow {
                max_generation_mw: Some(2.0),
                ..thermal_row(1, 1, Some(0))
            },
        ];
        data.line_bounds = vec![
            LineBoundsRow {
                direct_mw: Some(1.0),
                ..line_row(1, 0, Some(0))
            },
            LineBoundsRow {
                reverse_mw: Some(2.0),
                ..line_row(1, 1, Some(0))
            },
        ];
        data.pumping_bounds = vec![
            PumpingBoundsRow {
                min_m3s: Some(1.0),
                ..pumping_row(1, 0, Some(0))
            },
            PumpingBoundsRow {
                max_m3s: Some(2.0),
                ..pumping_row(1, 1, Some(0))
            },
        ];
        data.contract_bounds = vec![
            ContractBoundsRow {
                min_mw: Some(1.0),
                ..contract_row(1, 0, Some(0))
            },
            ContractBoundsRow {
                max_mw: Some(2.0),
                ..contract_row(1, 0, Some(1))
            },
            ContractBoundsRow {
                price_per_mwh: Some(3.0),
                ..contract_row(1, 1, Some(0))
            },
        ];

        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(
            !ctx.has_errors(),
            "block-eligible columns must not be rejected, got: {:?}",
            ctx.errors()
        );
    }

    // ── check_block_id_on_anticipated_thermal ────────────────────────────────

    #[test]
    fn test_block_row_rejected_on_anticipated_thermal_only() {
        let anticipated_thermal = Thermal {
            anticipated_config: Some(AnticipatedConfig::LeadStages(1)),
            ..make_thermal(1, 0.0, 500.0)
        };
        let plain_thermal = make_thermal(2, 0.0, 500.0);
        let mut data = make_data(
            vec![],
            vec![anticipated_thermal, plain_thermal],
            vec![],
            two_stage_study_stages(),
            vec![],
            vec![],
        );
        // Full-coverage, in-bounds, zero-rate window over the single leading
        // delivery stage (stage 0) so no rule other than the one under test
        // finds anything in this fixture.
        data.initial_conditions.past_anticipated_commitments = vec![AnticipatedCommitmentHistory {
            thermal_id: EntityId::from(1),
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, 1, 31).unwrap(),
            value_mw: 0.0,
        }];
        // The anticipated thermal (1) appears at both stages and the plain
        // thermal (2) appears at stage 0 too, so `stage_id` cannot stand in
        // for `anticipated_config` in the predicate under test.
        data.thermal_bounds = vec![
            ThermalBoundsRow {
                max_generation_mw: Some(100.0),
                ..thermal_row(1, 0, Some(0))
            },
            ThermalBoundsRow {
                max_generation_mw: Some(150.0),
                ..thermal_row(1, 1, Some(0))
            },
            ThermalBoundsRow {
                max_generation_mw: Some(200.0),
                ..thermal_row(2, 0, Some(0))
            },
        ];

        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        let errors = ctx.errors();
        assert_eq!(
            errors.len(),
            2,
            "expected exactly two errors, got: {errors:?}"
        );
        assert!(
            errors
                .iter()
                .any(|e| e.message.contains("Thermal 1") && e.message.contains("stage_id=0")),
            "expected an error naming Thermal 1 at stage_id=0: {errors:?}"
        );
        assert!(
            errors
                .iter()
                .any(|e| e.message.contains("Thermal 1") && e.message.contains("stage_id=1")),
            "expected an error naming Thermal 1 at stage_id=1: {errors:?}"
        );
        assert!(
            !errors.iter().any(|e| e.message.contains("Thermal 2")),
            "no error may name Thermal 2 (the plain thermal): {errors:?}"
        );
    }

    // ── check_bound_raises_declared_capacity ─────────────────────────────────

    #[test]
    fn test_bounds_row_raising_max_turbined_is_rejected() {
        let mut data = make_data(
            two_hydro_capacity_study(),
            vec![],
            vec![],
            two_stage_study_stages(),
            vec![],
            vec![],
        );
        data.hydro_bounds = vec![HydroBoundsRow {
            max_turbined_m3s: Some(900.0),
            max_generation_mw: Some(400.0),
            ..hydro_row(1, 0, None)
        }];

        let errors = capacity_raise_errors(&data);
        assert_eq!(
            errors.len(),
            1,
            "expected exactly one error, got: {errors:?}"
        );
        let err = &errors[0];
        assert_eq!(
            err.file.to_string_lossy(),
            "constraints/hydro_bounds.parquet"
        );
        assert!(err.message.contains("Hydro 1"), "message: {}", err.message);
        assert!(
            err.message.contains("stage_id=0"),
            "message: {}",
            err.message
        );
        assert!(err.message.contains("600"), "message: {}", err.message);
        assert!(err.message.contains("900"), "message: {}", err.message);
        assert!(
            !err.message.contains("max_generation_mw"),
            "no finding may name max_generation_mw (it is equal to its \
             declared value): {}",
            err.message
        );
    }

    /// Extends [`test_bounds_row_raising_max_turbined_is_rejected`]'s study
    /// with a lowering row, an exact-match row, and a stage-wide-plus-per-block
    /// pair (200 then 500, both <= hydro 2's declared 1000) — the last pair is
    /// what would trip an implementation comparing against the row-to-row
    /// resolved bound instead of the plant's own declared value, since 500
    /// exceeds the stage-wide override (200) while still not exceeding the
    /// plant's declared capacity.
    #[test]
    fn test_bounds_row_lowering_or_matching_declared_capacity_is_accepted() {
        let mut data = make_data(
            two_hydro_capacity_study(),
            vec![],
            vec![],
            two_stage_study_stages(),
            vec![],
            vec![],
        );
        data.hydro_bounds = vec![
            HydroBoundsRow {
                max_turbined_m3s: Some(100.0),
                ..hydro_row(2, 1, None)
            },
            HydroBoundsRow {
                max_turbined_m3s: Some(600.0),
                ..hydro_row(1, 1, None)
            },
            HydroBoundsRow {
                max_turbined_m3s: Some(200.0),
                ..hydro_row(2, 0, None)
            },
            HydroBoundsRow {
                max_turbined_m3s: Some(500.0),
                ..hydro_row(2, 0, Some(1))
            },
        ];

        let errors = capacity_raise_errors(&data);
        assert!(
            errors.is_empty(),
            "lowering, exact-match, and a per-block row between a stage-wide \
             override and the declared value must all be accepted: {errors:?}"
        );
    }

    #[test]
    fn test_bounds_row_for_unknown_hydro_is_skipped_by_the_capacity_rule() {
        let mut data = make_data(
            two_hydro_capacity_study(),
            vec![],
            vec![],
            two_stage_study_stages(),
            vec![],
            vec![],
        );
        data.hydro_bounds = vec![HydroBoundsRow {
            max_turbined_m3s: Some(9999.0),
            max_generation_mw: Some(9999.0),
            ..hydro_row(99, 0, None)
        }];

        assert!(
            capacity_raise_errors(&data).is_empty(),
            "a row on an unknown hydro_id must produce no finding from this \
             rule: {:?}",
            capacity_raise_errors(&data)
        );
    }

    /// The `max_generation_mw` counterpart of
    /// [`test_bounds_row_raising_max_turbined_is_rejected`]: raises generation
    /// only, turbined equal to declared — pins that deleting the
    /// `max_generation_mw` arm would go undetected by the first test alone
    /// (there, generation is already equal to its declared value).
    #[test]
    fn test_bounds_row_raising_max_generation_only_is_rejected() {
        let mut data = make_data(
            two_hydro_capacity_study(),
            vec![],
            vec![],
            two_stage_study_stages(),
            vec![],
            vec![],
        );
        data.hydro_bounds = vec![HydroBoundsRow {
            max_turbined_m3s: Some(600.0),
            max_generation_mw: Some(900.0),
            ..hydro_row(1, 0, None)
        }];

        let errors = capacity_raise_errors(&data);
        assert_eq!(
            errors.len(),
            1,
            "expected exactly one error, got: {errors:?}"
        );
        let msg = &errors[0].message;
        assert!(msg.contains("Hydro 1"), "message: {msg}");
        assert!(msg.contains("max_generation_mw"), "message: {msg}");
        assert!(msg.contains("400"), "message: {msg}");
        assert!(msg.contains("900"), "message: {msg}");
        assert!(
            !msg.contains("max_turbined_m3s"),
            "no finding may name max_turbined_m3s (it is equal to its \
             declared value): {msg}"
        );
    }

    /// A second, later row for the same hydro is the one that violates —
    /// pins that every row is checked, not only the first row seen per hydro.
    #[test]
    fn test_capacity_rule_checks_every_row_not_only_the_first_per_hydro() {
        let mut data = make_data(
            two_hydro_capacity_study(),
            vec![],
            vec![],
            two_stage_study_stages(),
            vec![],
            vec![],
        );
        data.hydro_bounds = vec![
            HydroBoundsRow {
                max_turbined_m3s: Some(500.0),
                ..hydro_row(1, 1, None)
            },
            HydroBoundsRow {
                max_turbined_m3s: Some(900.0),
                ..hydro_row(1, 0, None)
            },
        ];

        let errors = capacity_raise_errors(&data);
        assert_eq!(
            errors.len(),
            1,
            "expected exactly one error (the second row), got: {errors:?}"
        );
        assert!(
            errors[0].message.contains("stage_id=0"),
            "message: {}",
            errors[0].message
        );
    }

    // ── check_group_bound_raises_declared_capacity (rule 45) ────────────────

    /// Two hydros; hydro 1 carries two unit groups (2 and 5) whose declared
    /// `max_turbined_m3s`/`max_generation_mw` differ from each other, from
    /// hydro 1's own declared value, and from hydro 2's single group (3) —
    /// a lookup bug comparing against the wrong group, the sibling group, or
    /// the plant's own declared value instead of the target group's would
    /// pass or fail these tests for the wrong reason. Each hydro's group
    /// maxima sum to exactly its own declared value so rule 41 never fires
    /// alongside rule 45 in these fixtures; each hydro's `min_*` is 0 and
    /// matched by its groups' `min_*`, so rule 44 stays silent too.
    fn two_group_capacity_study() -> Vec<Hydro> {
        vec![
            Hydro {
                min_turbined_m3s: 0.0,
                max_turbined_m3s: 900.0,
                min_generation_mw: 0.0,
                max_generation_mw: 700.0,
                unit_groups: vec![
                    make_unit_group(2, 1, 0.0, 400.0, 0.0, 600.0),
                    make_unit_group(5, 1, 0.0, 300.0, 0.0, 300.0),
                ],
                ..make_hydro(1, None)
            },
            Hydro {
                min_turbined_m3s: 0.0,
                max_turbined_m3s: 1200.0,
                min_generation_mw: 0.0,
                max_generation_mw: 1000.0,
                unit_groups: vec![make_unit_group(3, 2, 0.0, 1000.0, 0.0, 1200.0)],
                ..make_hydro(2, None)
            },
        ]
    }

    #[test]
    fn test_group_bounds_row_raising_max_turbined_is_rejected() {
        let mut data = make_data(
            two_group_capacity_study(),
            vec![],
            vec![],
            two_stage_study_stages(),
            vec![],
            vec![],
        );
        data.hydro_unit_group_bounds = vec![HydroUnitGroupBoundsRow {
            max_turbined_m3s: Some(850.0),
            max_generation_mw: Some(400.0),
            ..group_bounds_row(1, 2, 0, None)
        }];

        let errors = group_capacity_raise_errors(&data);
        assert_eq!(
            errors.len(),
            1,
            "expected exactly one error, got: {errors:?}"
        );
        let err = &errors[0];
        assert_eq!(
            err.file.to_string_lossy(),
            "constraints/hydro_unit_group_bounds.parquet"
        );
        assert_eq!(
            err.entity.as_deref(),
            Some("hydro_id=1, hydro_unit_group_id=2, stage_id=0"),
            "entity string mismatch: {:?}",
            err.entity
        );
        assert!(
            err.message.contains("unit group 2"),
            "message: {}",
            err.message
        );
        assert!(
            err.message.contains("stage_id=0"),
            "message: {}",
            err.message
        );
        assert!(err.message.contains("600"), "message: {}", err.message);
        assert!(err.message.contains("850"), "message: {}", err.message);
        assert!(
            !err.message.contains("max_generation_mw"),
            "no finding may name max_generation_mw (it is equal to its \
             declared value): {}",
            err.message
        );
    }

    /// The `max_generation_mw` counterpart: turbined equal to the group's
    /// declared value, generation raised — pins that deleting the
    /// `max_generation_mw` arm would go undetected by the turbined test alone.
    #[test]
    fn test_group_bounds_row_raising_max_generation_only_is_rejected() {
        let mut data = make_data(
            two_group_capacity_study(),
            vec![],
            vec![],
            two_stage_study_stages(),
            vec![],
            vec![],
        );
        data.hydro_unit_group_bounds = vec![HydroUnitGroupBoundsRow {
            max_turbined_m3s: Some(600.0),
            max_generation_mw: Some(900.0),
            ..group_bounds_row(1, 2, 0, None)
        }];

        let errors = group_capacity_raise_errors(&data);
        assert_eq!(
            errors.len(),
            1,
            "expected exactly one error, got: {errors:?}"
        );
        let err = &errors[0];
        assert_eq!(
            err.entity.as_deref(),
            Some("hydro_id=1, hydro_unit_group_id=2, stage_id=0"),
            "entity string mismatch: {:?}",
            err.entity
        );
        let msg = &err.message;
        assert!(msg.contains("unit group 2"), "message: {msg}");
        assert!(msg.contains("max_generation_mw"), "message: {msg}");
        assert!(msg.contains("400"), "message: {msg}");
        assert!(msg.contains("900"), "message: {msg}");
        assert!(
            !msg.contains("max_turbined_m3s"),
            "no finding may name max_turbined_m3s (it is equal to its \
             declared value): {msg}"
        );
    }

    /// Raises group 5's `max_turbined_m3s` above ITS OWN declared value (300)
    /// while staying below group 2's declared value (600) and hydro 1's own
    /// declared value (900) — the case that would slip through a bug that
    /// compares against the sibling group or the plant instead of group 5.
    #[test]
    fn test_group_bounds_row_compares_against_its_own_group_not_a_sibling_or_the_plant() {
        let mut data = make_data(
            two_group_capacity_study(),
            vec![],
            vec![],
            two_stage_study_stages(),
            vec![],
            vec![],
        );
        data.hydro_unit_group_bounds = vec![HydroUnitGroupBoundsRow {
            max_turbined_m3s: Some(350.0),
            ..group_bounds_row(1, 5, 0, None)
        }];

        let errors = group_capacity_raise_errors(&data);
        assert_eq!(
            errors.len(),
            1,
            "expected exactly one error, got: {errors:?}"
        );
        let msg = &errors[0].message;
        assert!(msg.contains("unit group 5"), "message: {msg}");
        assert!(msg.contains("300"), "message: {msg}");
        assert!(msg.contains("350"), "message: {msg}");
    }

    /// Extends the turbined-raise fixture with a lowering row, an exact-match
    /// row, and a stage-wide-plus-per-block pair (100 then 250, both <= group
    /// 5's declared 300) on hydro 2's group 3 — the pair is what would trip an
    /// implementation comparing against the resolved bound instead of the
    /// group's own declared value.
    #[test]
    fn test_group_bounds_row_lowering_or_matching_declared_capacity_is_accepted() {
        let mut data = make_data(
            two_group_capacity_study(),
            vec![],
            vec![],
            two_stage_study_stages(),
            vec![],
            vec![],
        );
        data.hydro_unit_group_bounds = vec![
            HydroUnitGroupBoundsRow {
                max_turbined_m3s: Some(600.0),
                ..group_bounds_row(1, 2, 1, None)
            },
            HydroUnitGroupBoundsRow {
                max_generation_mw: Some(250.0),
                ..group_bounds_row(1, 5, 0, None)
            },
            HydroUnitGroupBoundsRow {
                max_turbined_m3s: Some(100.0),
                ..group_bounds_row(2, 3, 0, None)
            },
            HydroUnitGroupBoundsRow {
                max_turbined_m3s: Some(250.0),
                ..group_bounds_row(2, 3, 0, Some(1))
            },
        ];

        let errors = group_capacity_raise_errors(&data);
        assert!(
            errors.is_empty(),
            "lowering, exact-match, and a per-block row between a stage-wide \
             override and the declared value must all be accepted: {errors:?}"
        );
    }

    #[test]
    fn test_group_bounds_row_for_unknown_group_is_skipped_by_the_capacity_rule() {
        let mut data = make_data(
            two_group_capacity_study(),
            vec![],
            vec![],
            two_stage_study_stages(),
            vec![],
            vec![],
        );
        data.hydro_unit_group_bounds = vec![
            HydroUnitGroupBoundsRow {
                max_turbined_m3s: Some(9999.0),
                max_generation_mw: Some(9999.0),
                ..group_bounds_row(1, 99, 0, None)
            },
            HydroUnitGroupBoundsRow {
                max_turbined_m3s: Some(9999.0),
                ..group_bounds_row(99, 2, 0, None)
            },
        ];

        let errors = group_capacity_raise_errors(&data);
        assert!(
            errors.is_empty(),
            "a row on an unknown (hydro_id, hydro_unit_group_id) pair must \
             produce no finding from this rule: {errors:?}"
        );
    }

    /// A second, later row for the same group is the one that violates —
    /// pins that every row is checked, not only the first row seen per group.
    #[test]
    fn test_group_capacity_rule_checks_every_row_not_only_the_first_per_group() {
        let mut data = make_data(
            two_group_capacity_study(),
            vec![],
            vec![],
            two_stage_study_stages(),
            vec![],
            vec![],
        );
        data.hydro_unit_group_bounds = vec![
            HydroUnitGroupBoundsRow {
                max_turbined_m3s: Some(500.0),
                ..group_bounds_row(1, 2, 1, None)
            },
            HydroUnitGroupBoundsRow {
                max_turbined_m3s: Some(850.0),
                ..group_bounds_row(1, 2, 0, None)
            },
        ];

        let errors = group_capacity_raise_errors(&data);
        assert_eq!(
            errors.len(),
            1,
            "expected exactly one error (the second row), got: {errors:?}"
        );
        assert!(
            errors[0].message.contains("stage_id=0"),
            "message: {}",
            errors[0].message
        );
    }
}
