//! Layer 3 — Referential integrity validation.
//!
//! Verifies that every cross-entity reference in `ParsedData` resolves to an
//! existing entity in the corresponding registry.  All 32 rules are checked
//! regardless of errors found in earlier rules — every dangling reference is
//! collected before returning.
//!
//! The primary entry point is `validate_referential_integrity`.

use std::collections::HashSet;

use super::{ErrorKind, ValidationContext, schema::ParsedData};

// ── validate_referential_integrity ───────────────────────────────────────────

/// Performs Layer 3 referential integrity validation on the parsed data.
///
/// For each of the 32 cross-reference rules, checks that the referenced entity
/// ID exists in the target registry.  Any dangling reference adds one
/// [`ErrorKind::InvalidReference`] entry to `ctx` with the message:
///
/// ```text
/// "<source_type> <source_id> references non-existent <target_type> <target_id> via field '<field_name>'"
/// ```
///
/// This function is infallible — it never returns a `Result`.  All errors are
/// collected in `ctx`.  Optional data collections (empty `Vec` or `None`) are
/// silently skipped.
///
/// # Arguments
///
/// * `data` — fully parsed case data produced by [`super::schema::validate_schema`].
/// * `ctx`  — mutable validation context that accumulates diagnostics.
pub(crate) fn validate_referential_integrity(data: &ParsedData, ctx: &mut ValidationContext) {
    let ids = LookupSets {
        bus: data.buses.iter().map(|b| b.id.0).collect(),
        hydro: data.hydros.iter().map(|h| h.id.0).collect(),
        thermal: data.thermals.iter().map(|t| t.id.0).collect(),
        line: data.lines.iter().map(|l| l.id.0).collect(),
        pumping: data.pumping_stations.iter().map(|p| p.id.0).collect(),
        contract: data.energy_contracts.iter().map(|c| c.id.0).collect(),
        ncs: data
            .non_controllable_sources
            .iter()
            .map(|n| n.id.0)
            .collect(),
        generic_constraint: data.generic_constraints.iter().map(|g| g.id.0).collect(),
    };

    check_line_references(data, ctx, &ids.bus);
    check_hydro_references(data, ctx, &ids.bus, &ids.hydro);
    check_thermal_references(data, ctx, &ids.bus);
    check_ncs_references(data, ctx, &ids.bus, &ids.ncs);
    check_pumping_references(data, ctx, &ids.bus, &ids.hydro);
    check_contract_references(data, ctx, &ids.bus);
    check_extension_references(data, ctx, &ids.hydro);
    check_scenario_references(data, ctx, &ids.bus, &ids.hydro, &ids.ncs);
    check_bounds_references(data, ctx, &ids);
    check_penalty_override_references(data, ctx, &ids.bus, &ids.hydro, &ids.line, &ids.ncs);
    check_load_factor_references(data, ctx, &ids.bus);
    check_generic_constraint_expression_references(data, ctx, &ids);
    check_generic_constraint_bounds_validity(data, ctx);
    check_ncs_bounds_and_factors(data, ctx, &ids.ncs);
}

/// O(1) lookup sets for all entity registries, built once and shared across helpers.
struct LookupSets {
    bus: HashSet<i32>,
    hydro: HashSet<i32>,
    thermal: HashSet<i32>,
    line: HashSet<i32>,
    pumping: HashSet<i32>,
    contract: HashSet<i32>,
    ncs: HashSet<i32>,
    generic_constraint: HashSet<i32>,
}

// ── Per-entity-group helper functions ─────────────────────────────────────────

/// Rules 1-2: Line -> bus references (`source_bus_id`, `target_bus_id`).
fn check_line_references(data: &ParsedData, ctx: &mut ValidationContext, bus_ids: &HashSet<i32>) {
    for line in &data.lines {
        let entity_str = format!("Line {}", line.id.0);

        if !bus_ids.contains(&line.source_bus_id.0) {
            ctx.add_error(
                ErrorKind::InvalidReference,
                "system/lines.json",
                Some(&entity_str),
                format!(
                    "{entity_str} references non-existent Bus {} via field 'source_bus_id'",
                    line.source_bus_id.0
                ),
            );
        }

        if !bus_ids.contains(&line.target_bus_id.0) {
            ctx.add_error(
                ErrorKind::InvalidReference,
                "system/lines.json",
                Some(&entity_str),
                format!(
                    "{entity_str} references non-existent Bus {} via field 'target_bus_id'",
                    line.target_bus_id.0
                ),
            );
        }
    }
}

/// Rules 3, 7-8: Hydro -> bus, downstream hydro, and diversion references.
fn check_hydro_references(
    data: &ParsedData,
    ctx: &mut ValidationContext,
    bus_ids: &HashSet<i32>,
    hydro_ids: &HashSet<i32>,
) {
    for hydro in &data.hydros {
        let entity_str = format!("Hydro {}", hydro.id.0);

        // Rule 3: bus reference.
        if !bus_ids.contains(&hydro.bus_id.0) {
            ctx.add_error(
                ErrorKind::InvalidReference,
                "system/hydros.json",
                Some(&entity_str),
                format!(
                    "{entity_str} references non-existent Bus {} via field 'bus_id'",
                    hydro.bus_id.0
                ),
            );
        }

        // Rule 7: downstream hydro reference (optional).
        if let Some(downstream_id) = hydro.downstream_id {
            if !hydro_ids.contains(&downstream_id.0) {
                ctx.add_error(
                    ErrorKind::InvalidReference,
                    "system/hydros.json",
                    Some(&entity_str),
                    format!(
                        "{entity_str} references non-existent Hydro {} via field 'downstream_id'",
                        downstream_id.0
                    ),
                );
            }
        }

        // Rule 8: diversion downstream hydro reference.
        if let Some(ref diversion) = hydro.diversion {
            if !hydro_ids.contains(&diversion.downstream_id.0) {
                ctx.add_error(
                    ErrorKind::InvalidReference,
                    "system/hydros.json",
                    Some(&entity_str),
                    format!(
                        "{entity_str} references non-existent Hydro {} via field 'diversion.downstream_id'",
                        diversion.downstream_id.0
                    ),
                );
            }
        }
    }
}

/// Rule 4: Thermal -> bus reference.
fn check_thermal_references(
    data: &ParsedData,
    ctx: &mut ValidationContext,
    bus_ids: &HashSet<i32>,
) {
    for thermal in &data.thermals {
        let entity_str = format!("Thermal {}", thermal.id.0);

        if !bus_ids.contains(&thermal.bus_id.0) {
            ctx.add_error(
                ErrorKind::InvalidReference,
                "system/thermals.json",
                Some(&entity_str),
                format!(
                    "{entity_str} references non-existent Bus {} via field 'bus_id'",
                    thermal.bus_id.0
                ),
            );
        }
    }
}

/// Rules 5, 19b: NCS -> bus reference and NCS model references.
fn check_ncs_references(
    data: &ParsedData,
    ctx: &mut ValidationContext,
    bus_ids: &HashSet<i32>,
    ncs_ids: &HashSet<i32>,
) {
    for ncs in &data.non_controllable_sources {
        let entity_str = format!("NonControllableSource {}", ncs.id.0);

        if !bus_ids.contains(&ncs.bus_id.0) {
            ctx.add_error(
                ErrorKind::InvalidReference,
                "system/non_controllable_sources.json",
                Some(&entity_str),
                format!(
                    "{entity_str} references non-existent Bus {} via field 'bus_id'",
                    ncs.bus_id.0
                ),
            );
        }
    }

    for (i, model) in data.ncs_models.iter().enumerate() {
        if !ncs_ids.contains(&model.ncs_id.0) {
            ctx.add_error(
                ErrorKind::InvalidReference,
                "scenarios/non_controllable_stats.parquet",
                Some(format!("NcsModel[{i}]")),
                format!(
                    "NcsModel[{i}] references non-existent NonControllableSource {} via field 'ncs_id'",
                    model.ncs_id.0
                ),
            );
        }
    }
}

/// Rules 6, 9-10: `PumpingStation` -> bus and hydro references.
fn check_pumping_references(
    data: &ParsedData,
    ctx: &mut ValidationContext,
    bus_ids: &HashSet<i32>,
    hydro_ids: &HashSet<i32>,
) {
    for station in &data.pumping_stations {
        let entity_str = format!("PumpingStation {}", station.id.0);

        if !bus_ids.contains(&station.bus_id.0) {
            ctx.add_error(
                ErrorKind::InvalidReference,
                "system/pumping_stations.json",
                Some(&entity_str),
                format!(
                    "{entity_str} references non-existent Bus {} via field 'bus_id'",
                    station.bus_id.0
                ),
            );
        }

        if !hydro_ids.contains(&station.source_hydro_id.0) {
            ctx.add_error(
                ErrorKind::InvalidReference,
                "system/pumping_stations.json",
                Some(&entity_str),
                format!(
                    "{entity_str} references non-existent Hydro {} via field 'source_hydro_id'",
                    station.source_hydro_id.0
                ),
            );
        }

        if !hydro_ids.contains(&station.destination_hydro_id.0) {
            ctx.add_error(
                ErrorKind::InvalidReference,
                "system/pumping_stations.json",
                Some(&entity_str),
                format!(
                    "{entity_str} references non-existent Hydro {} via field 'destination_hydro_id'",
                    station.destination_hydro_id.0
                ),
            );
        }
    }
}

/// Rule 11: `EnergyContract` -> bus reference.
fn check_contract_references(
    data: &ParsedData,
    ctx: &mut ValidationContext,
    bus_ids: &HashSet<i32>,
) {
    for contract in &data.energy_contracts {
        let entity_str = format!("EnergyContract {}", contract.id.0);

        if !bus_ids.contains(&contract.bus_id.0) {
            ctx.add_error(
                ErrorKind::InvalidReference,
                "system/energy_contracts.json",
                Some(&entity_str),
                format!(
                    "{entity_str} references non-existent Bus {} via field 'bus_id'",
                    contract.bus_id.0
                ),
            );
        }
    }
}

/// Rules 12-14: Extension data -> hydro references (geometry, production models, FPHA).
fn check_extension_references(
    data: &ParsedData,
    ctx: &mut ValidationContext,
    hydro_ids: &HashSet<i32>,
) {
    for (i, row) in data.hydro_geometry.iter().enumerate() {
        if !hydro_ids.contains(&row.hydro_id.0) {
            ctx.add_error(
                ErrorKind::InvalidReference,
                "system/hydro_geometry.parquet",
                Some(format!("HydroGeometryRow[{i}]")),
                format!(
                    "HydroGeometryRow[{i}] references non-existent Hydro {} via field 'hydro_id'",
                    row.hydro_id.0
                ),
            );
        }
    }

    for (i, model) in data.production_models.iter().enumerate() {
        if !hydro_ids.contains(&model.hydro_id.0) {
            ctx.add_error(
                ErrorKind::InvalidReference,
                "system/hydro_production_models.json",
                Some(format!("ProductionModelConfig[{i}]")),
                format!(
                    "ProductionModelConfig[{i}] references non-existent Hydro {} via field 'hydro_id'",
                    model.hydro_id.0
                ),
            );
        }
    }

    for (i, row) in data.fpha_hyperplanes.iter().enumerate() {
        if !hydro_ids.contains(&row.hydro_id.0) {
            ctx.add_error(
                ErrorKind::InvalidReference,
                "system/fpha_hyperplanes.parquet",
                Some(format!("FphaHyperplaneRow[{i}]")),
                format!(
                    "FphaHyperplaneRow[{i}] references non-existent Hydro {} via field 'hydro_id'",
                    row.hydro_id.0
                ),
            );
        }
    }
}

/// Rules 15-20: Scenario data references.
#[allow(clippy::too_many_lines)]
fn check_scenario_references(
    data: &ParsedData,
    ctx: &mut ValidationContext,
    bus_ids: &HashSet<i32>,
    hydro_ids: &HashSet<i32>,
    ncs_ids: &HashSet<i32>,
) {
    for (i, row) in data.inflow_seasonal_stats.iter().enumerate() {
        if !hydro_ids.contains(&row.hydro_id.0) {
            ctx.add_error(
                ErrorKind::InvalidReference,
                "scenarios/inflow_seasonal_stats.parquet",
                Some(format!("InflowSeasonalStatsRow[{i}]")),
                format!(
                    "InflowSeasonalStatsRow[{i}] references non-existent Hydro {} via field 'hydro_id'",
                    row.hydro_id.0
                ),
            );
        }
    }

    for (i, row) in data.inflow_ar_coefficients.iter().enumerate() {
        if !hydro_ids.contains(&row.hydro_id.0) {
            ctx.add_error(
                ErrorKind::InvalidReference,
                "scenarios/inflow_ar_coefficients.parquet",
                Some(format!("InflowArCoefficientRow[{i}]")),
                format!(
                    "InflowArCoefficientRow[{i}] references non-existent Hydro {} via field 'hydro_id'",
                    row.hydro_id.0
                ),
            );
        }
    }

    for (i, row) in data.inflow_history.iter().enumerate() {
        if !hydro_ids.contains(&row.hydro_id.0) {
            ctx.add_error(
                ErrorKind::InvalidReference,
                "scenarios/inflow_history.parquet",
                Some(format!("InflowHistoryRow[{i}]")),
                format!(
                    "InflowHistoryRow[{i}] references non-existent Hydro {} via field 'hydro_id'",
                    row.hydro_id.0
                ),
            );
        }
    }

    for (i, row) in data.load_seasonal_stats.iter().enumerate() {
        if !bus_ids.contains(&row.bus_id.0) {
            ctx.add_error(
                ErrorKind::InvalidReference,
                "scenarios/load_seasonal_stats.parquet",
                Some(format!("LoadSeasonalStatsRow[{i}]")),
                format!(
                    "LoadSeasonalStatsRow[{i}] references non-existent Bus {} via field 'bus_id'",
                    row.bus_id.0
                ),
            );
        }
    }

    if let Some(ref correlation) = data.correlation {
        for profile in correlation.profiles.values() {
            for group in &profile.groups {
                for entity in &group.entities {
                    let (valid, type_label, registry_label) = match entity.entity_type.as_str() {
                        "inflow" => (hydro_ids.contains(&entity.id.0), "inflow", "Hydro"),
                        "load" => (bus_ids.contains(&entity.id.0), "load", "Bus"),
                        "ncs" => (
                            ncs_ids.contains(&entity.id.0),
                            "ncs",
                            "NonControllableSource",
                        ),
                        other => {
                            ctx.add_error(
                                ErrorKind::InvalidReference,
                                "scenarios/correlation.json",
                                Some(format!("CorrelationEntity({other}, {})", entity.id.0)),
                                format!(
                                    "unknown entity_type '{other}'; valid types are: inflow, load, ncs"
                                ),
                            );
                            continue;
                        }
                    };
                    if !valid {
                        let entity_str =
                            format!("CorrelationEntity({type_label}, {})", entity.id.0);
                        ctx.add_error(
                            ErrorKind::InvalidReference,
                            "scenarios/correlation.json",
                            Some(&entity_str),
                            format!(
                                "{entity_str} references non-existent {registry_label} {} via field 'id'",
                                entity.id.0
                            ),
                        );
                    }
                }
            }
        }
    }

    for (i, row) in data.external_scenarios.iter().enumerate() {
        if !hydro_ids.contains(&row.hydro_id.0) {
            ctx.add_error(
                ErrorKind::InvalidReference,
                "scenarios/external_inflow_scenarios.parquet",
                Some(format!("ExternalScenarioRow[{i}]")),
                format!(
                    "ExternalScenarioRow[{i}] references non-existent Hydro {} via field 'hydro_id'",
                    row.hydro_id.0
                ),
            );
        }
    }

    for (i, row) in data.external_load_scenarios.iter().enumerate() {
        if !bus_ids.contains(&row.bus_id.0) {
            ctx.add_error(
                ErrorKind::InvalidReference,
                "scenarios/external_load_scenarios.parquet",
                Some(format!("ExternalLoadRow[{i}]")),
                format!(
                    "ExternalLoadRow[{i}] references non-existent Bus {} via field 'bus_id'",
                    row.bus_id.0
                ),
            );
        }
    }

    for (i, row) in data.external_ncs_scenarios.iter().enumerate() {
        if !ncs_ids.contains(&row.ncs_id.0) {
            ctx.add_error(
                ErrorKind::InvalidReference,
                "scenarios/external_ncs_scenarios.parquet",
                Some(format!("ExternalNcsRow[{i}]")),
                format!(
                    "ExternalNcsRow[{i}] references non-existent NonControllableSource {} via field 'ncs_id'",
                    row.ncs_id.0
                ),
            );
        }
    }
}

/// Rules 21-26: Bounds rows -> entity references.
fn check_bounds_references(data: &ParsedData, ctx: &mut ValidationContext, ids: &LookupSets) {
    for (i, row) in data.thermal_bounds.iter().enumerate() {
        if !ids.thermal.contains(&row.thermal_id.0) {
            ctx.add_error(
                ErrorKind::InvalidReference,
                "constraints/thermal_bounds.parquet",
                Some(format!("ThermalBoundsRow[{i}]")),
                format!(
                    "ThermalBoundsRow[{i}] references non-existent Thermal {} via field 'thermal_id'",
                    row.thermal_id.0
                ),
            );
        }
    }

    for (i, row) in data.hydro_bounds.iter().enumerate() {
        if !ids.hydro.contains(&row.hydro_id.0) {
            ctx.add_error(
                ErrorKind::InvalidReference,
                "constraints/hydro_bounds.parquet",
                Some(format!("HydroBoundsRow[{i}]")),
                format!(
                    "HydroBoundsRow[{i}] references non-existent Hydro {} via field 'hydro_id'",
                    row.hydro_id.0
                ),
            );
        }
    }

    for (i, row) in data.line_bounds.iter().enumerate() {
        if !ids.line.contains(&row.line_id.0) {
            ctx.add_error(
                ErrorKind::InvalidReference,
                "constraints/line_bounds.parquet",
                Some(format!("LineBoundsRow[{i}]")),
                format!(
                    "LineBoundsRow[{i}] references non-existent Line {} via field 'line_id'",
                    row.line_id.0
                ),
            );
        }
    }

    for (i, row) in data.pumping_bounds.iter().enumerate() {
        if !ids.pumping.contains(&row.station_id.0) {
            ctx.add_error(
                ErrorKind::InvalidReference,
                "constraints/pumping_bounds.parquet",
                Some(format!("PumpingBoundsRow[{i}]")),
                format!(
                    "PumpingBoundsRow[{i}] references non-existent PumpingStation {} via field 'station_id'",
                    row.station_id.0
                ),
            );
        }
    }

    for (i, row) in data.contract_bounds.iter().enumerate() {
        if !ids.contract.contains(&row.contract_id.0) {
            ctx.add_error(
                ErrorKind::InvalidReference,
                "constraints/contract_bounds.parquet",
                Some(format!("ContractBoundsRow[{i}]")),
                format!(
                    "ContractBoundsRow[{i}] references non-existent EnergyContract {} via field 'contract_id'",
                    row.contract_id.0
                ),
            );
        }
    }

    for (i, row) in data.generic_constraint_bounds.iter().enumerate() {
        if !ids.generic_constraint.contains(&row.constraint_id) {
            ctx.add_error(
                ErrorKind::InvalidReference,
                "constraints/generic_constraint_bounds.parquet",
                Some(format!("GenericConstraintBoundsRow[{i}]")),
                format!(
                    "GenericConstraintBoundsRow[{i}] references non-existent GenericConstraint {} via field 'constraint_id'",
                    row.constraint_id
                ),
            );
        }
    }
}

/// Rules 27-30: Penalty override rows -> entity references.
fn check_penalty_override_references(
    data: &ParsedData,
    ctx: &mut ValidationContext,
    bus_ids: &HashSet<i32>,
    hydro_ids: &HashSet<i32>,
    line_ids: &HashSet<i32>,
    ncs_ids: &HashSet<i32>,
) {
    for (i, row) in data.penalty_overrides_bus.iter().enumerate() {
        if !bus_ids.contains(&row.bus_id.0) {
            ctx.add_error(
                ErrorKind::InvalidReference,
                "constraints/penalty_overrides_bus.parquet",
                Some(format!("BusPenaltyOverrideRow[{i}]")),
                format!(
                    "BusPenaltyOverrideRow[{i}] references non-existent Bus {} via field 'bus_id'",
                    row.bus_id.0
                ),
            );
        }
    }

    for (i, row) in data.penalty_overrides_line.iter().enumerate() {
        if !line_ids.contains(&row.line_id.0) {
            ctx.add_error(
                ErrorKind::InvalidReference,
                "constraints/penalty_overrides_line.parquet",
                Some(format!("LinePenaltyOverrideRow[{i}]")),
                format!(
                    "LinePenaltyOverrideRow[{i}] references non-existent Line {} via field 'line_id'",
                    row.line_id.0
                ),
            );
        }
    }

    for (i, row) in data.penalty_overrides_hydro.iter().enumerate() {
        if !hydro_ids.contains(&row.hydro_id.0) {
            ctx.add_error(
                ErrorKind::InvalidReference,
                "constraints/penalty_overrides_hydro.parquet",
                Some(format!("HydroPenaltyOverrideRow[{i}]")),
                format!(
                    "HydroPenaltyOverrideRow[{i}] references non-existent Hydro {} via field 'hydro_id'",
                    row.hydro_id.0
                ),
            );
        }
    }

    for (i, row) in data.penalty_overrides_ncs.iter().enumerate() {
        if !ncs_ids.contains(&row.source_id.0) {
            ctx.add_error(
                ErrorKind::InvalidReference,
                "constraints/penalty_overrides_ncs.parquet",
                Some(format!("NcsPenaltyOverrideRow[{i}]")),
                format!(
                    "NcsPenaltyOverrideRow[{i}] references non-existent NonControllableSource {} via field 'source_id'",
                    row.source_id.0
                ),
            );
        }
    }
}

/// Rules 31-32: `LoadFactorEntry` -> bus and stage references.
fn check_load_factor_references(
    data: &ParsedData,
    ctx: &mut ValidationContext,
    bus_ids: &HashSet<i32>,
) {
    let study_stage_ids: HashSet<i32> = data
        .stages
        .stages
        .iter()
        .filter(|s| s.id >= 0)
        .map(|s| s.id)
        .collect();

    for (i, entry) in data.load_factors.iter().enumerate() {
        if !bus_ids.contains(&entry.bus_id.0) {
            ctx.add_error(
                ErrorKind::InvalidReference,
                "scenarios/load_factors.json",
                Some(format!("LoadFactorEntry[{i}]")),
                format!(
                    "LoadFactorEntry[{i}] references non-existent Bus {} via field 'bus_id'",
                    entry.bus_id.0
                ),
            );
        }

        if !study_stage_ids.contains(&entry.stage_id) {
            ctx.add_error(
                ErrorKind::InvalidReference,
                "scenarios/load_factors.json",
                Some(format!("LoadFactorEntry[{i}]")),
                format!(
                    "LoadFactorEntry[{i}] references non-existent Stage {} via field 'stage_id'",
                    entry.stage_id
                ),
            );
        }
    }
}

/// Rule 33: `GenericConstraint` expression entity ID existence.
fn check_generic_constraint_expression_references(
    data: &ParsedData,
    ctx: &mut ValidationContext,
    ids: &LookupSets,
) {
    let entity_ids = EntityIdSets {
        hydro: &ids.hydro,
        thermal: &ids.thermal,
        line: &ids.line,
        bus: &ids.bus,
        pumping: &ids.pumping,
        contract: &ids.contract,
        ncs: &ids.ncs,
    };
    for constraint in &data.generic_constraints {
        let gc_label = format!("GenericConstraint {}", constraint.id.0);
        for (term_idx, term) in constraint.expression.terms.iter().enumerate() {
            let label = format!("{gc_label} term[{term_idx}]");
            validate_variable_ref_entity(&term.variable, &label, &entity_ids, ctx);
        }
    }
}

/// Rules 34-35: `GenericConstraintBoundsRow` `block_id` validity and duplicate key detection.
fn check_generic_constraint_bounds_validity(data: &ParsedData, ctx: &mut ValidationContext) {
    let stage_block_counts: std::collections::HashMap<i32, usize> = data
        .stages
        .stages
        .iter()
        .filter(|s| s.id >= 0)
        .map(|s| (s.id, s.blocks.len()))
        .collect();

    for (i, row) in data.generic_constraint_bounds.iter().enumerate() {
        if let Some(blk) = row.block_id {
            if let Some(&n_blocks) = stage_block_counts.get(&row.stage_id) {
                #[allow(clippy::cast_sign_loss)]
                let blk_usize = blk as usize;
                if blk < 0 || blk_usize >= n_blocks {
                    ctx.add_error(
                        ErrorKind::InvalidValue,
                        "constraints/generic_constraint_bounds.parquet",
                        Some(format!("GenericConstraintBoundsRow[{i}]")),
                        format!(
                            "GenericConstraintBoundsRow[{i}] has block_id={blk} but Stage {} has only {n_blocks} block(s) (valid range: 0..{n_blocks})",
                            row.stage_id
                        ),
                    );
                }
            }
        }
    }

    {
        let mut seen_keys: HashSet<(i32, i32, Option<i32>)> = HashSet::new();
        for (i, row) in data.generic_constraint_bounds.iter().enumerate() {
            let key = (row.constraint_id, row.stage_id, row.block_id);
            if !seen_keys.insert(key) {
                ctx.add_error(
                    ErrorKind::DuplicateId,
                    "constraints/generic_constraint_bounds.parquet",
                    Some(format!("GenericConstraintBoundsRow[{i}]")),
                    format!(
                        "Duplicate key (constraint_id={}, stage_id={}, block_id={:?}) in generic constraint bounds",
                        row.constraint_id, row.stage_id, row.block_id
                    ),
                );
            }
        }
    }
}

/// Rules 36-41: NCS bounds and NCS factor entry checks.
fn check_ncs_bounds_and_factors(
    data: &ParsedData,
    ctx: &mut ValidationContext,
    ncs_ids: &HashSet<i32>,
) {
    let study_stage_ids: HashSet<i32> = data
        .stages
        .stages
        .iter()
        .filter(|s| s.id >= 0)
        .map(|s| s.id)
        .collect();

    for (i, row) in data.ncs_bounds.iter().enumerate() {
        if !ncs_ids.contains(&row.ncs_id.0) {
            ctx.add_error(
                ErrorKind::InvalidReference,
                "constraints/ncs_bounds.parquet",
                Some(format!("NcsBoundsRow[{i}]")),
                format!(
                    "NcsBoundsRow[{i}] references non-existent NonControllableSource {} via field 'ncs_id'",
                    row.ncs_id.0
                ),
            );
        }
        if !study_stage_ids.contains(&row.stage_id) {
            ctx.add_error(
                ErrorKind::InvalidReference,
                "constraints/ncs_bounds.parquet",
                Some(format!("NcsBoundsRow[{i}]")),
                format!(
                    "NcsBoundsRow[{i}] has invalid stage_id {} (not a valid study stage)",
                    row.stage_id
                ),
            );
        }
        if row.available_generation_mw < 0.0 {
            ctx.add_error(
                ErrorKind::InvalidValue,
                "constraints/ncs_bounds.parquet",
                Some(format!("NcsBoundsRow[{i}]")),
                format!(
                    "NcsBoundsRow[{i}] has negative available_generation_mw: {}",
                    row.available_generation_mw
                ),
            );
        }
    }

    for (i, entry) in data.non_controllable_factors.iter().enumerate() {
        if !ncs_ids.contains(&entry.ncs_id.0) {
            ctx.add_error(
                ErrorKind::InvalidReference,
                "scenarios/non_controllable_factors.json",
                Some(format!("NcsFactorEntry[{i}]")),
                format!(
                    "NcsFactorEntry[{i}] references non-existent NonControllableSource {} via field 'ncs_id'",
                    entry.ncs_id.0
                ),
            );
        }
        if !study_stage_ids.contains(&entry.stage_id) {
            ctx.add_error(
                ErrorKind::InvalidReference,
                "scenarios/non_controllable_factors.json",
                Some(format!("NcsFactorEntry[{i}]")),
                format!(
                    "NcsFactorEntry[{i}] has invalid stage_id {} (not a valid study stage)",
                    entry.stage_id
                ),
            );
        }
        for (j, bf) in entry.block_factors.iter().enumerate() {
            if bf.factor < 0.0 {
                ctx.add_error(
                    ErrorKind::InvalidValue,
                    "scenarios/non_controllable_factors.json",
                    Some(format!("NcsFactorEntry[{i}].block_factors[{j}]")),
                    format!(
                        "NcsFactorEntry[{i}] block_factors[{j}] has negative factor: {}",
                        bf.factor
                    ),
                );
            }
        }
    }
}

/// Entity ID sets used by Rule 33 to check [`cobre_core::VariableRef`] existence.
struct EntityIdSets<'a> {
    hydro: &'a HashSet<i32>,
    thermal: &'a HashSet<i32>,
    line: &'a HashSet<i32>,
    bus: &'a HashSet<i32>,
    pumping: &'a HashSet<i32>,
    contract: &'a HashSet<i32>,
    ncs: &'a HashSet<i32>,
}

/// Helper for Rule 33: validate that a [`VariableRef`] references an existing entity.
///
/// Emits an error if the entity ID does not exist in the corresponding registry.
/// Emits a warning for stub entity types (pumping, contracts, non-controllable)
/// that have no LP effect.
fn validate_variable_ref_entity(
    var: &cobre_core::VariableRef,
    label: &str,
    ids: &EntityIdSets<'_>,
    ctx: &mut ValidationContext,
) {
    use cobre_core::VariableRef;

    let file = "system/generic_constraints.json";
    match var {
        VariableRef::HydroStorage { hydro_id, .. }
        | VariableRef::HydroEvaporation { hydro_id, .. }
        | VariableRef::HydroWithdrawal { hydro_id, .. }
        | VariableRef::HydroTurbined { hydro_id, .. }
        | VariableRef::HydroSpillage { hydro_id, .. }
        | VariableRef::HydroDiversion { hydro_id, .. }
        | VariableRef::HydroOutflow { hydro_id, .. }
        | VariableRef::HydroGeneration { hydro_id, .. } => {
            if !ids.hydro.contains(&hydro_id.0) {
                ctx.add_error(
                    ErrorKind::InvalidReference,
                    file,
                    Some(label.to_string()),
                    format!("{label} references non-existent Hydro {}", hydro_id.0),
                );
            }
        }
        VariableRef::ThermalGeneration { thermal_id, .. } => {
            if !ids.thermal.contains(&thermal_id.0) {
                ctx.add_error(
                    ErrorKind::InvalidReference,
                    file,
                    Some(label.to_string()),
                    format!("{label} references non-existent Thermal {}", thermal_id.0),
                );
            }
        }
        VariableRef::LineDirect { line_id, .. }
        | VariableRef::LineReverse { line_id, .. }
        | VariableRef::LineExchange { line_id, .. } => {
            if !ids.line.contains(&line_id.0) {
                ctx.add_error(
                    ErrorKind::InvalidReference,
                    file,
                    Some(label.to_string()),
                    format!("{label} references non-existent Line {}", line_id.0),
                );
            }
        }
        VariableRef::BusDeficit { bus_id, .. } | VariableRef::BusExcess { bus_id, .. } => {
            if !ids.bus.contains(&bus_id.0) {
                ctx.add_error(
                    ErrorKind::InvalidReference,
                    file,
                    Some(label.to_string()),
                    format!("{label} references non-existent Bus {}", bus_id.0),
                );
            }
        }
        VariableRef::PumpingFlow { station_id, .. }
        | VariableRef::PumpingPower { station_id, .. } => {
            if !ids.pumping.contains(&station_id.0) {
                ctx.add_warning(
                    ErrorKind::UnusedEntity,
                    file,
                    Some(label.to_string()),
                    format!(
                        "{label} references PumpingStation {} which is a stub entity with no LP effect",
                        station_id.0
                    ),
                );
            }
        }
        VariableRef::ContractImport { contract_id, .. }
        | VariableRef::ContractExport { contract_id, .. } => {
            if !ids.contract.contains(&contract_id.0) {
                ctx.add_warning(
                    ErrorKind::UnusedEntity,
                    file,
                    Some(label.to_string()),
                    format!(
                        "{label} references Contract {} which is a stub entity with no LP effect",
                        contract_id.0
                    ),
                );
            }
        }
        VariableRef::NonControllableGeneration { source_id, .. }
        | VariableRef::NonControllableCurtailment { source_id, .. } => {
            if !ids.ncs.contains(&source_id.0) {
                ctx.add_warning(
                    ErrorKind::UnusedEntity,
                    file,
                    Some(label.to_string()),
                    format!(
                        "{label} references NonControllableSource {} which is a stub entity with no LP effect",
                        source_id.0
                    ),
                );
            }
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::too_many_lines,
    clippy::doc_markdown
)]
mod tests {
    use super::*;
    use cobre_core::{
        EntityId,
        entities::{
            DiversionChannel, Hydro, HydroGenerationModel, HydroPenalties, Line,
            NonControllableSource, PumpingStation, Thermal, ThermalCostSegment,
        },
        scenario::{CorrelationEntity, CorrelationGroup, CorrelationModel, CorrelationProfile},
    };
    use std::collections::BTreeMap;
    use std::fs;
    use tempfile::TempDir;

    use crate::{
        constraints::{
            BusPenaltyOverrideRow, GenericConstraintBoundsRow, HydroBoundsRow, LineBoundsRow,
            NcsBoundsRow, NcsPenaltyOverrideRow, ThermalBoundsRow,
        },
        extensions::HydroGeometryRow,
        scenarios::{
            BlockFactor, InflowSeasonalStatsRow, LoadFactorEntry, LoadSeasonalStatsRow,
            NcsFactorEntry,
        },
        validation::{
            schema::{ParsedData, validate_schema},
            structural::validate_structure,
        },
    };

    const VALID_CONFIG_JSON: &str = r#"{
        "training": {
            "forward_passes": 10,
            "stopping_rules": [
                { "type": "iteration_limit", "limit": 100 }
            ]
        }
    }"#;

    const VALID_PENALTIES_JSON: &str = r#"{
        "bus": {
            "deficit_segments": [
                { "depth_mw": 500.0, "cost": 1000.0 },
                { "depth_mw": null,  "cost": 5000.0 }
            ],
            "excess_cost": 100.0
        },
        "line": { "exchange_cost": 2.0 },
        "hydro": {
            "spillage_cost": 0.01,
            "fpha_turbined_cost": 0.05,
            "diversion_cost": 0.1,
            "storage_violation_below_cost": 10000.0,
            "filling_target_violation_cost": 50000.0,
            "turbined_violation_below_cost": 500.0,
            "outflow_violation_below_cost": 500.0,
            "outflow_violation_above_cost": 500.0,
            "generation_violation_below_cost": 1000.0,
            "evaporation_violation_cost": 5000.0,
            "water_withdrawal_violation_cost": 1000.0
        },
        "non_controllable_source": { "curtailment_cost": 0.005 }
    }"#;

    const VALID_STAGES_JSON: &str = r#"{
        "policy_graph": {
            "type": "finite_horizon",
            "annual_discount_rate": 0.06,
            "transitions": []
        },
        "scenario_source": { "seed": 42 },
        "stages": [
            {
                "id": 0,
                "start_date": "2024-01-01",
                "end_date": "2024-02-01",
                "blocks": [{ "id": 0, "name": "FLAT", "hours": 744.0 }],
                "num_scenarios": 50
            }
        ]
    }"#;

    const VALID_INITIAL_CONDITIONS_JSON: &str = r#"{
        "storage": [],
        "filling_storage": []
    }"#;

    /// Write a string to a relative path under `root`, creating parent dirs.
    fn write_file(root: &std::path::Path, relative: &str, content: &str) {
        let full = root.join(relative);
        if let Some(parent) = full.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(&full, content).unwrap();
    }

    /// Build a minimal case directory with buses=[1], hydros=[], thermals=[], lines=[].
    fn make_minimal_case(dir: &TempDir) {
        let root = dir.path();
        write_file(root, "config.json", VALID_CONFIG_JSON);
        write_file(root, "penalties.json", VALID_PENALTIES_JSON);
        write_file(root, "stages.json", VALID_STAGES_JSON);
        write_file(
            root,
            "initial_conditions.json",
            VALID_INITIAL_CONDITIONS_JSON,
        );
        write_file(
            root,
            "system/buses.json",
            r#"{ "buses": [{ "id": 1, "name": "BUS_1" }] }"#,
        );
        write_file(root, "system/lines.json", r#"{ "lines": [] }"#);
        write_file(root, "system/hydros.json", r#"{ "hydros": [] }"#);
        write_file(root, "system/thermals.json", r#"{ "thermals": [] }"#);
    }

    /// Parse the case directory at `dir` and return `ParsedData`.
    /// Panics if validation fails — all test cases start from valid data.
    fn parse_case(dir: &TempDir) -> ParsedData {
        let mut ctx = ValidationContext::new();
        let manifest = validate_structure(dir.path(), &mut ctx);
        assert!(
            !ctx.has_errors(),
            "structural validation failed: {:?}",
            ctx.errors()
        );
        let data = validate_schema(dir.path(), &manifest, &mut ctx)
            .expect("schema validation should succeed for valid case");
        assert!(
            !ctx.has_errors(),
            "schema validation failed: {:?}",
            ctx.errors()
        );
        data
    }

    fn hydro_penalties() -> HydroPenalties {
        HydroPenalties {
            spillage_cost: 1.0,
            diversion_cost: 1.0,
            fpha_turbined_cost: 1.0,
            storage_violation_below_cost: 1.0,
            filling_target_violation_cost: 1.0,
            turbined_violation_below_cost: 1.0,
            outflow_violation_below_cost: 1.0,
            outflow_violation_above_cost: 1.0,
            generation_violation_below_cost: 1.0,
            evaporation_violation_cost: 1.0,
            water_withdrawal_violation_cost: 1.0,
            water_withdrawal_violation_pos_cost: 1.0,
            water_withdrawal_violation_neg_cost: 1.0,
            evaporation_violation_pos_cost: 1.0,
            evaporation_violation_neg_cost: 1.0,
            inflow_nonnegativity_cost: 1000.0,
        }
    }

    fn make_hydro(id: i32, bus_id: i32) -> Hydro {
        Hydro {
            id: EntityId::from(id),
            name: format!("Hydro_{id}"),
            bus_id: EntityId::from(bus_id),
            downstream_id: None,
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: 0.0,
            max_storage_hm3: 1000.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: HydroGenerationModel::ConstantProductivity {
                productivity_mw_per_m3s: 1.0,
            },
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 1000.0,
            min_generation_mw: 0.0,
            max_generation_mw: 1000.0,
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: None,
            evaporation_reference_volumes_hm3: None,
            diversion: None,
            filling: None,
            penalties: hydro_penalties(),
        }
    }

    fn make_line(id: i32, source_bus: i32, target_bus: i32) -> Line {
        Line {
            id: EntityId::from(id),
            name: format!("Line_{id}"),
            source_bus_id: EntityId::from(source_bus),
            target_bus_id: EntityId::from(target_bus),
            entry_stage_id: None,
            exit_stage_id: None,
            direct_capacity_mw: 100.0,
            reverse_capacity_mw: 100.0,
            losses_percent: 0.0,
            exchange_cost: 0.0,
        }
    }

    fn make_ncs(id: i32, bus_id: i32) -> NonControllableSource {
        NonControllableSource {
            id: EntityId::from(id),
            name: format!("Ncs_{id}"),
            bus_id: EntityId::from(bus_id),
            entry_stage_id: None,
            exit_stage_id: None,
            max_generation_mw: 50.0,
            curtailment_cost: 1.0,
        }
    }

    fn make_pumping(id: i32, bus_id: i32, src_hydro: i32, dst_hydro: i32) -> PumpingStation {
        PumpingStation {
            id: EntityId::from(id),
            name: format!("Pump_{id}"),
            bus_id: EntityId::from(bus_id),
            source_hydro_id: EntityId::from(src_hydro),
            destination_hydro_id: EntityId::from(dst_hydro),
            entry_stage_id: None,
            exit_stage_id: None,
            consumption_mw_per_m3s: 0.5,
            min_flow_m3s: 0.0,
            max_flow_m3s: 100.0,
        }
    }

    /// Given a `ParsedData` where all entity cross-references are valid,
    /// `validate_referential_integrity` adds no errors to `ctx`.
    #[test]
    fn test_all_valid_references_no_errors() {
        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let data = parse_case(&dir);
        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);
        assert!(
            !ctx.has_errors(),
            "expected no errors for valid data, got: {:?}",
            ctx.errors()
        );
    }

    /// Given a `ParsedData` where Line id=5 has `source_bus_id` referencing
    /// non-existent bus id=999, `validate_referential_integrity` adds exactly 1
    /// `InvalidReference` error mentioning `"Line 5"` and `"999"`.
    #[test]
    fn test_line_invalid_source_bus() {
        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = parse_case(&dir);
        // bus 999 does not exist — only bus 1 was loaded
        data.lines = vec![make_line(5, 999, 1)];
        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);
        assert!(ctx.has_errors(), "expected errors for invalid line ref");
        let errors = ctx.errors();
        let inv_ref: Vec<_> = errors
            .iter()
            .filter(|e| e.kind == ErrorKind::InvalidReference)
            .collect();
        assert_eq!(
            inv_ref.len(),
            1,
            "expected exactly 1 InvalidReference error"
        );
        let msg = &inv_ref[0].message;
        assert!(
            msg.contains("Line 5"),
            "message should contain 'Line 5', got: {msg}"
        );
        assert!(
            msg.contains("999"),
            "message should contain '999', got: {msg}"
        );
    }

    /// Given a `ParsedData` where `Hydro` id=3 has `downstream_id = Some(EntityId(100))`
    /// and hydro 100 does not exist, `validate_referential_integrity` adds an
    /// `InvalidReference` error mentioning `"Hydro 3"` and `"downstream_id"`.
    #[test]
    fn test_hydro_invalid_downstream_id() {
        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = parse_case(&dir);
        let mut hydro = make_hydro(3, 1);
        hydro.downstream_id = Some(EntityId::from(100)); // hydro 100 does not exist
        data.hydros = vec![hydro];
        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);
        assert!(
            ctx.has_errors(),
            "expected error for dangling downstream_id"
        );
        let errors = ctx.errors();
        let inv_ref: Vec<_> = errors
            .iter()
            .filter(|e| e.kind == ErrorKind::InvalidReference)
            .collect();
        assert!(
            !inv_ref.is_empty(),
            "expected at least 1 InvalidReference error"
        );
        let msg = &inv_ref[0].message;
        assert!(
            msg.contains("Hydro 3"),
            "message should contain 'Hydro 3', got: {msg}"
        );
        assert!(
            msg.contains("downstream_id"),
            "message should contain 'downstream_id', got: {msg}"
        );
    }

    /// Given a `ParsedData` with empty `pumping_stations` and `energy_contracts`,
    /// `validate_referential_integrity` produces no errors for those rules.
    #[test]
    fn test_empty_optional_collections_no_errors() {
        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = parse_case(&dir);
        data.pumping_stations = vec![];
        data.energy_contracts = vec![];
        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);
        assert!(
            !ctx.has_errors(),
            "empty optional collections should not produce errors, got: {:?}",
            ctx.errors()
        );
    }

    /// Given a `ParsedData` with 2 invalid bus references (Line, Thermal)
    /// and 1 invalid hydro reference (HydroGeometryRow), all 3 are collected.
    #[test]
    fn test_multiple_invalid_references_all_collected() {
        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = parse_case(&dir);
        // Line with bad target_bus_id (bus 999 does not exist)
        data.lines = vec![make_line(5, 1, 999)];
        // Thermal with bad bus_id (bus 777 does not exist)
        data.thermals = vec![Thermal {
            id: EntityId::from(20),
            name: "T20".to_string(),
            bus_id: EntityId::from(777), // bad
            entry_stage_id: None,
            exit_stage_id: None,
            cost_segments: vec![ThermalCostSegment {
                capacity_mw: 100.0,
                cost_per_mwh: 50.0,
            }],
            min_generation_mw: 0.0,
            max_generation_mw: 100.0,
            gnl_config: None,
        }];
        // HydroGeometryRow referencing non-existent hydro (888)
        data.hydro_geometry = vec![HydroGeometryRow {
            hydro_id: EntityId::from(888),
            volume_hm3: 0.0,
            area_km2: 0.0,
            height_m: 0.0,
        }];

        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);
        assert!(
            ctx.has_errors(),
            "expected errors for multiple invalid refs"
        );
        let errors = ctx.errors();
        let inv_ref: Vec<_> = errors
            .iter()
            .filter(|e| e.kind == ErrorKind::InvalidReference)
            .collect();
        assert_eq!(
            inv_ref.len(),
            3,
            "expected exactly 3 InvalidReference errors, got {}: {:?}",
            inv_ref.len(),
            inv_ref.iter().map(|e| &e.message).collect::<Vec<_>>()
        );
    }

    /// Hydro with a valid bus_id produces no error.
    #[test]
    fn test_hydro_valid_bus_ref() {
        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = parse_case(&dir);
        data.hydros = vec![make_hydro(10, 1)]; // bus 1 exists
        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);
        assert!(!ctx.has_errors());
    }

    /// Hydro with a missing bus_id produces one `InvalidReference` error.
    #[test]
    fn test_hydro_invalid_bus_ref() {
        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = parse_case(&dir);
        data.hydros = vec![make_hydro(10, 42)]; // bus 42 does not exist
        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);
        assert!(ctx.has_errors());
        let errors = ctx.errors();
        let inv_ref: Vec<_> = errors
            .iter()
            .filter(|e| e.kind == ErrorKind::InvalidReference)
            .collect();
        assert_eq!(inv_ref.len(), 1);
        assert!(inv_ref[0].message.contains("Hydro 10"));
        assert!(inv_ref[0].message.contains("42"));
        assert!(inv_ref[0].message.contains("bus_id"));
    }

    /// Hydro with `downstream_id = None` must not produce any error for rule 7.
    #[test]
    fn test_hydro_downstream_id_none_no_error() {
        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = parse_case(&dir);
        let mut hydro = make_hydro(10, 1);
        hydro.downstream_id = None;
        data.hydros = vec![hydro];
        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);
        assert!(
            !ctx.has_errors(),
            "downstream_id = None should not produce errors"
        );
    }

    /// Hydro with `diversion = None` must not produce any error for rule 8.
    #[test]
    fn test_hydro_diversion_none_no_error() {
        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = parse_case(&dir);
        let mut hydro = make_hydro(10, 1);
        hydro.diversion = None;
        data.hydros = vec![hydro];
        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);
        assert!(
            !ctx.has_errors(),
            "diversion = None should not produce errors"
        );
    }

    /// Hydro with a diversion referencing a non-existent downstream produces 1 error.
    #[test]
    fn test_hydro_diversion_invalid_downstream() {
        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = parse_case(&dir);
        let mut hydro = make_hydro(10, 1);
        hydro.diversion = Some(DiversionChannel {
            downstream_id: EntityId::from(999), // does not exist
            max_flow_m3s: 100.0,
        });
        data.hydros = vec![hydro];
        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);
        assert!(ctx.has_errors());
        let inv: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::InvalidReference)
            .collect();
        assert_eq!(inv.len(), 1);
        assert!(inv[0].message.contains("diversion.downstream_id"));
        assert!(inv[0].message.contains("999"));
    }

    /// PumpingStation with valid bus and hydro references produces no error.
    #[test]
    fn test_pumping_valid_refs() {
        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = parse_case(&dir);
        data.hydros = vec![make_hydro(10, 1)];
        data.pumping_stations = vec![make_pumping(1, 1, 10, 10)]; // bus 1, hydros 10,10 all exist
        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);
        assert!(!ctx.has_errors());
    }

    /// PumpingStation referencing a non-existent source hydro produces 1 error.
    #[test]
    fn test_pumping_invalid_source_hydro() {
        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = parse_case(&dir);
        data.hydros = vec![make_hydro(10, 1)];
        // source hydro 999 missing, destination hydro 10 exists
        data.pumping_stations = vec![make_pumping(1, 1, 999, 10)];
        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);
        assert!(ctx.has_errors());
        let inv: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::InvalidReference)
            .collect();
        assert_eq!(inv.len(), 1);
        assert!(inv[0].message.contains("source_hydro_id"));
        assert!(inv[0].message.contains("999"));
    }

    /// `InflowSeasonalStatsRow` referencing non-existent hydro produces 1 error.
    #[test]
    fn test_inflow_seasonal_stats_invalid_hydro_ref() {
        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = parse_case(&dir);
        // hydro 999 does not exist
        data.inflow_seasonal_stats = vec![InflowSeasonalStatsRow {
            hydro_id: EntityId::from(999),
            stage_id: 0,
            mean_m3s: 100.0,
            std_m3s: 10.0,
        }];
        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);
        assert!(ctx.has_errors());
        let inv: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::InvalidReference)
            .collect();
        assert_eq!(inv.len(), 1);
        assert!(inv[0].message.contains("999"));
        assert!(inv[0].message.contains("hydro_id"));
    }

    /// `LoadSeasonalStatsRow` referencing non-existent bus produces 1 error.
    #[test]
    fn test_load_seasonal_stats_invalid_bus_ref() {
        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = parse_case(&dir);
        // bus 777 does not exist
        data.load_seasonal_stats = vec![LoadSeasonalStatsRow {
            bus_id: EntityId::from(777),
            stage_id: 0,
            mean_mw: 100.0,
            std_mw: 10.0,
        }];
        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);
        assert!(ctx.has_errors());
        let inv: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::InvalidReference)
            .collect();
        assert_eq!(inv.len(), 1);
        assert!(inv[0].message.contains("777"));
        assert!(inv[0].message.contains("bus_id"));
    }

    /// `CorrelationEntity` with invalid inflow, load, and ncs entity references
    /// produces one `InvalidReference` error per invalid reference.
    #[test]
    fn test_correlation_entity_inflow_invalid_hydro() {
        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = parse_case(&dir);
        let mut profiles = BTreeMap::new();
        profiles.insert(
            "profile1".to_string(),
            CorrelationProfile {
                groups: vec![CorrelationGroup {
                    name: "group1".to_string(),
                    entities: vec![
                        CorrelationEntity {
                            entity_type: "inflow".to_string(),
                            id: EntityId::from(999), // does not exist
                        },
                        CorrelationEntity {
                            entity_type: "unknown".to_string(), // unknown type: not checked
                            id: EntityId::from(9999),
                        },
                    ],
                    matrix: vec![],
                }],
            },
        );
        data.correlation = Some(CorrelationModel {
            method: "pearson".to_string(),
            profiles,
            schedule: vec![],
        });

        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);
        assert!(ctx.has_errors());
        let inv: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::InvalidReference)
            .collect();
        // The "inflow" entity with non-existent hydro produces an error,
        // and the "unknown" entity_type also produces an error (M3 fix).
        assert_eq!(
            inv.len(),
            2,
            "expected errors for invalid hydro and unknown entity_type"
        );
        assert!(inv.iter().any(|e| e.message.contains("999")));
        assert!(
            inv.iter()
                .any(|e| e.message.contains("unknown entity_type"))
        );
    }

    /// `CorrelationEntity` with `entity_type == "inflow"` and a valid hydro id
    /// produces no error.
    #[test]
    fn test_correlation_entity_inflow_valid_hydro() {
        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = parse_case(&dir);
        data.hydros = vec![make_hydro(10, 1)];
        let mut profiles = BTreeMap::new();
        profiles.insert(
            "profile1".to_string(),
            CorrelationProfile {
                groups: vec![CorrelationGroup {
                    name: "group1".to_string(),
                    entities: vec![CorrelationEntity {
                        entity_type: "inflow".to_string(),
                        id: EntityId::from(10), // hydro 10 exists
                    }],
                    matrix: vec![],
                }],
            },
        );
        data.correlation = Some(CorrelationModel {
            method: "pearson".to_string(),
            profiles,
            schedule: vec![],
        });

        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);
        assert!(
            !ctx.has_errors(),
            "valid inflow ref should not produce errors"
        );
    }

    /// `ThermalBoundsRow` referencing a non-existent thermal produces 1 error.
    #[test]
    fn test_thermal_bounds_invalid_thermal_ref() {
        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = parse_case(&dir);
        // thermal 999 does not exist
        data.thermal_bounds = vec![ThermalBoundsRow {
            thermal_id: EntityId::from(999),
            stage_id: 0,
            min_generation_mw: None,
            max_generation_mw: None,
        }];
        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);
        assert!(ctx.has_errors());
        let inv: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::InvalidReference)
            .collect();
        assert_eq!(inv.len(), 1);
        assert!(inv[0].message.contains("999"));
        assert!(inv[0].message.contains("thermal_id"));
    }

    /// `HydroBoundsRow` referencing a non-existent hydro produces 1 error.
    #[test]
    fn test_hydro_bounds_invalid_hydro_ref() {
        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = parse_case(&dir);
        // hydro 555 does not exist
        data.hydro_bounds = vec![HydroBoundsRow {
            hydro_id: EntityId::from(555),
            stage_id: 0,
            min_turbined_m3s: None,
            max_turbined_m3s: None,
            min_storage_hm3: None,
            max_storage_hm3: None,
            min_outflow_m3s: None,
            max_outflow_m3s: None,
            min_generation_mw: None,
            max_generation_mw: None,
            max_diversion_m3s: None,
            filling_inflow_m3s: None,
            water_withdrawal_m3s: None,
        }];
        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);
        assert!(ctx.has_errors());
        let inv: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::InvalidReference)
            .collect();
        assert_eq!(inv.len(), 1);
        assert!(inv[0].message.contains("555"));
        assert!(inv[0].message.contains("hydro_id"));
    }

    /// `LineBoundsRow` referencing a non-existent line produces 1 error.
    #[test]
    fn test_line_bounds_invalid_line_ref() {
        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = parse_case(&dir);
        // line 333 does not exist
        data.line_bounds = vec![LineBoundsRow {
            line_id: EntityId::from(333),
            stage_id: 0,
            direct_mw: None,
            reverse_mw: None,
        }];
        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);
        assert!(ctx.has_errors());
        let inv: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::InvalidReference)
            .collect();
        assert_eq!(inv.len(), 1);
        assert!(inv[0].message.contains("333"));
        assert!(inv[0].message.contains("line_id"));
    }

    /// `GenericConstraintBoundsRow` referencing a non-existent constraint produces 1 error.
    #[test]
    fn test_generic_constraint_bounds_invalid_constraint_ref() {
        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = parse_case(&dir);
        // constraint 888 does not exist
        data.generic_constraint_bounds = vec![GenericConstraintBoundsRow {
            constraint_id: 888,
            stage_id: 0,
            block_id: None,
            bound: 1000.0,
        }];
        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);
        assert!(ctx.has_errors());
        let inv: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::InvalidReference)
            .collect();
        assert_eq!(inv.len(), 1);
        assert!(inv[0].message.contains("888"));
        assert!(inv[0].message.contains("constraint_id"));
    }

    /// `BusPenaltyOverrideRow` referencing a non-existent bus produces 1 error.
    #[test]
    fn test_bus_penalty_override_invalid_bus_ref() {
        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = parse_case(&dir);
        // bus 777 does not exist
        data.penalty_overrides_bus = vec![BusPenaltyOverrideRow {
            bus_id: EntityId::from(777),
            stage_id: 0,
            excess_cost: None,
        }];
        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);
        assert!(ctx.has_errors());
        let inv: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::InvalidReference)
            .collect();
        assert_eq!(inv.len(), 1);
        assert!(inv[0].message.contains("777"));
        assert!(inv[0].message.contains("bus_id"));
    }

    /// `NcsPenaltyOverrideRow` referencing a non-existent NCS source produces 1 error.
    #[test]
    fn test_ncs_penalty_override_invalid_ncs_ref() {
        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = parse_case(&dir);
        // NCS source 444 does not exist
        data.penalty_overrides_ncs = vec![NcsPenaltyOverrideRow {
            source_id: EntityId::from(444),
            stage_id: 0,
            curtailment_cost: None,
        }];
        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);
        assert!(ctx.has_errors());
        let inv: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::InvalidReference)
            .collect();
        assert_eq!(inv.len(), 1);
        assert!(inv[0].message.contains("444"));
        assert!(inv[0].message.contains("source_id"));
    }

    /// `NcsPenaltyOverrideRow` with a valid NCS source produces no error.
    #[test]
    fn test_ncs_penalty_override_valid_ncs_ref() {
        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = parse_case(&dir);
        data.non_controllable_sources = vec![make_ncs(1, 1)];
        data.penalty_overrides_ncs = vec![NcsPenaltyOverrideRow {
            source_id: EntityId::from(1),
            stage_id: 0,
            curtailment_cost: None,
        }];
        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);
        assert!(!ctx.has_errors(), "valid NCS ref should not produce errors");
    }

    /// `LoadFactorEntry` with a non-existent `bus_id` produces 1
    /// `InvalidReference` error for `scenarios/load_factors.json`.
    #[test]
    fn test_load_factors_invalid_bus_ref() {
        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = parse_case(&dir);
        // bus 999 does not exist
        data.load_factors = vec![LoadFactorEntry {
            bus_id: EntityId::from(999),
            stage_id: 0,
            block_factors: vec![BlockFactor {
                block_id: 0,
                factor: 1.0,
            }],
        }];
        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);
        assert!(ctx.has_errors());
        let inv: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::InvalidReference)
            .collect();
        assert_eq!(inv.len(), 1);
        assert!(inv[0].message.contains("999"));
        assert!(inv[0].message.contains("bus_id"));
        assert!(
            inv[0]
                .entity
                .as_deref()
                .unwrap_or("")
                .contains("LoadFactorEntry")
        );
    }

    /// `LoadFactorEntry` with a non-existent `stage_id` produces 1
    /// `InvalidReference` error for `scenarios/load_factors.json`.
    #[test]
    fn test_load_factors_invalid_stage_ref() {
        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = parse_case(&dir);
        // stage 999 does not exist; bus 1 does exist (added by make_minimal_case)
        data.load_factors = vec![LoadFactorEntry {
            bus_id: EntityId::from(1),
            stage_id: 999,
            block_factors: vec![BlockFactor {
                block_id: 0,
                factor: 1.0,
            }],
        }];
        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);
        assert!(ctx.has_errors());
        let inv: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::InvalidReference)
            .collect();
        assert_eq!(inv.len(), 1);
        assert!(inv[0].message.contains("999"));
        assert!(inv[0].message.contains("stage_id"));
        assert!(
            inv[0]
                .entity
                .as_deref()
                .unwrap_or("")
                .contains("LoadFactorEntry")
        );
    }

    /// `LoadFactorEntry` with valid `bus_id` and `stage_id` produces no
    /// `InvalidReference` errors from the load-factors rules.
    #[test]
    fn test_load_factors_valid_refs_no_error() {
        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = parse_case(&dir);
        // bus 1 and stage 0 both exist in the minimal case
        data.load_factors = vec![LoadFactorEntry {
            bus_id: EntityId::from(1),
            stage_id: 0,
            block_factors: vec![BlockFactor {
                block_id: 0,
                factor: 1.0,
            }],
        }];
        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);
        assert!(
            !ctx.has_errors(),
            "valid load_factors refs should produce no errors"
        );
    }

    /// Valid `NcsBoundsRow` with an existing NCS ID and valid stage produces no errors.
    #[test]
    fn test_ncs_bounds_valid_refs_no_error() {
        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = parse_case(&dir);
        data.non_controllable_sources = vec![make_ncs(1, 1)];
        data.ncs_bounds = vec![NcsBoundsRow {
            ncs_id: EntityId::from(1),
            stage_id: 0,
            available_generation_mw: 50.0,
        }];
        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);
        assert!(
            !ctx.has_errors(),
            "valid NCS bounds should produce no errors"
        );
    }

    /// `NcsBoundsRow` with a non-existent NCS ID produces `InvalidReference`.
    #[test]
    fn test_ncs_bounds_invalid_ncs_ref() {
        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = parse_case(&dir);
        data.ncs_bounds = vec![NcsBoundsRow {
            ncs_id: EntityId::from(999),
            stage_id: 0,
            available_generation_mw: 50.0,
        }];
        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);
        assert!(ctx.has_errors());
        let inv: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::InvalidReference)
            .filter(|e| e.file.to_str().unwrap_or("").contains("ncs_bounds"))
            .collect();
        assert_eq!(inv.len(), 1);
        assert!(inv[0].message.contains("999"));
    }

    /// `NcsBoundsRow` with negative `available_generation_mw` produces `InvalidValue`.
    #[test]
    fn test_ncs_bounds_negative_available_generation() {
        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = parse_case(&dir);
        data.non_controllable_sources = vec![make_ncs(1, 1)];
        data.ncs_bounds = vec![NcsBoundsRow {
            ncs_id: EntityId::from(1),
            stage_id: 0,
            available_generation_mw: -10.0,
        }];
        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);
        assert!(ctx.has_errors());
        let inv: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::InvalidValue)
            .collect();
        assert_eq!(inv.len(), 1);
        assert!(inv[0].message.contains("negative"));
    }

    /// Valid `NcsFactorEntry` with an existing NCS ID and valid stage produces no errors.
    #[test]
    fn test_ncs_factors_valid_refs_no_error() {
        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = parse_case(&dir);
        data.non_controllable_sources = vec![make_ncs(1, 1)];
        data.non_controllable_factors = vec![NcsFactorEntry {
            ncs_id: EntityId::from(1),
            stage_id: 0,
            block_factors: vec![BlockFactor {
                block_id: 0,
                factor: 1.0,
            }],
        }];
        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);
        assert!(
            !ctx.has_errors(),
            "valid NCS factors should produce no errors"
        );
    }

    /// `NcsFactorEntry` with a non-existent NCS ID produces `InvalidReference`.
    #[test]
    fn test_ncs_factors_invalid_ncs_ref() {
        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = parse_case(&dir);
        data.non_controllable_factors = vec![NcsFactorEntry {
            ncs_id: EntityId::from(999),
            stage_id: 0,
            block_factors: vec![BlockFactor {
                block_id: 0,
                factor: 1.0,
            }],
        }];
        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);
        assert!(ctx.has_errors());
        let inv: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::InvalidReference)
            .filter(|e| {
                e.file
                    .to_str()
                    .unwrap_or("")
                    .contains("non_controllable_factors")
            })
            .collect();
        assert_eq!(inv.len(), 1);
        assert!(inv[0].message.contains("999"));
    }

    /// `NcsFactorEntry` with an invalid `stage_id` produces `InvalidReference`.
    #[test]
    fn test_ncs_factors_invalid_stage_ref() {
        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = parse_case(&dir);
        data.non_controllable_sources = vec![make_ncs(1, 1)];
        data.non_controllable_factors = vec![NcsFactorEntry {
            ncs_id: EntityId::from(1),
            stage_id: 999,
            block_factors: vec![BlockFactor {
                block_id: 0,
                factor: 1.0,
            }],
        }];
        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);
        assert!(ctx.has_errors());
        let inv: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::InvalidReference)
            .filter(|e| {
                e.file
                    .to_str()
                    .unwrap_or("")
                    .contains("non_controllable_factors")
            })
            .collect();
        assert_eq!(inv.len(), 1);
        assert!(inv[0].message.contains("999"));
    }

    /// `NcsFactorEntry` with a negative block factor produces `InvalidValue`.
    #[test]
    fn test_ncs_factors_negative_factor() {
        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = parse_case(&dir);
        data.non_controllable_sources = vec![make_ncs(1, 1)];
        data.non_controllable_factors = vec![NcsFactorEntry {
            ncs_id: EntityId::from(1),
            stage_id: 0,
            block_factors: vec![BlockFactor {
                block_id: 0,
                factor: -0.5,
            }],
        }];
        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);
        assert!(ctx.has_errors());
        let inv: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::InvalidValue)
            .collect();
        assert_eq!(inv.len(), 1);
        assert!(inv[0].message.contains("negative"));
    }
}
