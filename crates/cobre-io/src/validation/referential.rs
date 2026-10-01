//! Layer 3 — Referential integrity validation.
//!
//! Verifies that every cross-entity reference in `ParsedData` resolves to an
//! existing entity in the corresponding registry. Every check runs regardless of
//! errors found in earlier checks — every dangling reference is collected before
//! returning.
//!
//! The primary entry point is `validate_referential_integrity`.

use std::collections::{HashMap, HashSet};

use cobre_core::AffineBound;

use super::{ErrorKind, ValidationContext, schema::ParsedData};

// ── validate_referential_integrity ───────────────────────────────────────────

/// Performs Layer 3 referential integrity validation on the parsed data.
///
/// Checks that every referenced entity ID exists in its target registry. Most
/// dangling-reference findings share one message shape, owned by
/// [`emit_dangling_ref`] / [`emit_dangling_ref_at`] — see either for the
/// exact wording.
///
/// Infallible — all errors are collected in `ctx`; optional data collections
/// (empty `Vec` or `None`) are silently skipped.
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
        hydro_unit_group: data
            .hydros
            .iter()
            .map(|h| (h.id.0, h.unit_groups.iter().map(|g| g.id.0).collect()))
            .collect(),
        hydro_group_bus: data
            .hydros
            .iter()
            .map(|h| (h.id.0, h.unit_groups.iter().map(|g| g.bus_id.0).collect()))
            .collect(),
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
    // Plant id -> its own group ids; group ids are plant-scoped, not global, so
    // this stays keyed by plant rather than flattened into one set.
    hydro_unit_group: HashMap<i32, HashSet<i32>>,
    // Plant id -> the bus ids its unit groups sit on. A bus selector names a
    // cell, which is plant-scoped, even though a bus is a global entity — the
    // global `bus` set would wrongly accept a bus that exists but that this
    // plant has no group on, so membership is checked per plant here instead.
    hydro_group_bus: HashMap<i32, HashSet<i32>>,
}

// ── Dangling-reference descriptor and emit helper ────────────────────────────
//
// The single owner of the "<location> references non-existent <target entity>
// <id> via field '<field>'" message shape (see the module doc). A row-type
// descriptor names the source file, the referenced entity type, and the
// field; `emit_dangling_ref` (indexed rows) and `emit_dangling_ref_at` (a
// caller-supplied location, e.g. an already-built `entity_str`) are the only
// two call sites that format the message. A block whose location or message
// does not fit this shape (a nested per-plant lookup, an extra clause, a
// runtime-dispatched field) stays open-coded.

/// One dangling-reference row-type: the source file, the referenced entity
/// type, and the field name that failed to resolve.
struct DanglingRefDescriptor {
    file: &'static str,
    target_entity: &'static str,
    field: &'static str,
}

/// Emits one `InvalidReference` finding for an indexed row: location
/// `"<row_label>[<row_index>]"`.
fn emit_dangling_ref(
    descriptor: &DanglingRefDescriptor,
    row_label: &str,
    row_index: usize,
    id: i32,
    ctx: &mut ValidationContext,
) {
    emit_dangling_ref_at(descriptor, &format!("{row_label}[{row_index}]"), id, ctx);
}

/// Emits one `InvalidReference` finding at a caller-supplied `location`
/// (an indexed row label, or an entity-named location such as `"Line 5"`).
fn emit_dangling_ref_at(
    descriptor: &DanglingRefDescriptor,
    location: &str,
    id: i32,
    ctx: &mut ValidationContext,
) {
    ctx.add_error(
        ErrorKind::InvalidReference,
        descriptor.file,
        Some(location),
        format!(
            "{location} references non-existent {} {id} via field '{}'",
            descriptor.target_entity, descriptor.field
        ),
    );
}

const HYDRO_GEOMETRY_ROW_HYDRO: DanglingRefDescriptor = DanglingRefDescriptor {
    file: "system/hydro_geometry.parquet",
    target_entity: "Hydro",
    field: "hydro_id",
};
const PRODUCTION_MODEL_CONFIG_HYDRO: DanglingRefDescriptor = DanglingRefDescriptor {
    file: "system/hydro_production_models.json",
    target_entity: "Hydro",
    field: "hydro_id",
};
const FPHA_HYPERPLANE_ROW_HYDRO: DanglingRefDescriptor = DanglingRefDescriptor {
    file: "system/fpha_hyperplanes.parquet",
    target_entity: "Hydro",
    field: "hydro_id",
};
const INFLOW_SEASONAL_STATS_ROW_HYDRO: DanglingRefDescriptor = DanglingRefDescriptor {
    file: "scenarios/inflow_seasonal_stats.parquet",
    target_entity: "Hydro",
    field: "hydro_id",
};
const INFLOW_AR_COEFFICIENT_ROW_HYDRO: DanglingRefDescriptor = DanglingRefDescriptor {
    file: "scenarios/inflow_ar_coefficients.parquet",
    target_entity: "Hydro",
    field: "hydro_id",
};
const INFLOW_ANNUAL_COMPONENT_ROW_HYDRO: DanglingRefDescriptor = DanglingRefDescriptor {
    file: "scenarios/inflow_annual_component.parquet",
    target_entity: "Hydro",
    field: "hydro_id",
};
const INFLOW_HISTORY_ROW_HYDRO: DanglingRefDescriptor = DanglingRefDescriptor {
    file: "scenarios/inflow_history.parquet",
    target_entity: "Hydro",
    field: "hydro_id",
};
const LOAD_SEASONAL_STATS_ROW_BUS: DanglingRefDescriptor = DanglingRefDescriptor {
    file: "scenarios/load_seasonal_stats.parquet",
    target_entity: "Bus",
    field: "bus_id",
};
const EXTERNAL_SCENARIO_ROW_HYDRO: DanglingRefDescriptor = DanglingRefDescriptor {
    file: "scenarios/external_inflow_scenarios.parquet",
    target_entity: "Hydro",
    field: "hydro_id",
};
const EXTERNAL_LOAD_ROW_BUS: DanglingRefDescriptor = DanglingRefDescriptor {
    file: "scenarios/external_load_scenarios.parquet",
    target_entity: "Bus",
    field: "bus_id",
};
const EXTERNAL_NCS_ROW_NCS: DanglingRefDescriptor = DanglingRefDescriptor {
    file: "scenarios/external_ncs_scenarios.parquet",
    target_entity: "NonControllableSource",
    field: "ncs_id",
};
const NCS_MODEL_NCS: DanglingRefDescriptor = DanglingRefDescriptor {
    file: "scenarios/non_controllable_stats.parquet",
    target_entity: "NonControllableSource",
    field: "ncs_id",
};
const THERMAL_BOUNDS_ROW_THERMAL: DanglingRefDescriptor = DanglingRefDescriptor {
    file: "constraints/thermal_bounds.parquet",
    target_entity: "Thermal",
    field: "thermal_id",
};
const HYDRO_BOUNDS_ROW_HYDRO: DanglingRefDescriptor = DanglingRefDescriptor {
    file: "constraints/hydro_bounds.parquet",
    target_entity: "Hydro",
    field: "hydro_id",
};
const HYDRO_UNIT_GROUP_BOUNDS_ROW_HYDRO: DanglingRefDescriptor = DanglingRefDescriptor {
    file: "constraints/hydro_unit_group_bounds.parquet",
    target_entity: "Hydro",
    field: "hydro_id",
};
const LINE_BOUNDS_ROW_LINE: DanglingRefDescriptor = DanglingRefDescriptor {
    file: "constraints/line_bounds.parquet",
    target_entity: "Line",
    field: "line_id",
};
const PUMPING_BOUNDS_ROW_PUMPING: DanglingRefDescriptor = DanglingRefDescriptor {
    file: "constraints/pumping_bounds.parquet",
    target_entity: "PumpingStation",
    field: "station_id",
};
const CONTRACT_BOUNDS_ROW_CONTRACT: DanglingRefDescriptor = DanglingRefDescriptor {
    file: "constraints/contract_bounds.parquet",
    target_entity: "EnergyContract",
    field: "contract_id",
};
const GENERIC_CONSTRAINT_BOUNDS_ROW_CONSTRAINT: DanglingRefDescriptor = DanglingRefDescriptor {
    file: "constraints/generic_constraint_bounds.parquet",
    target_entity: "GenericConstraint",
    field: "constraint_id",
};
const BUS_PENALTY_OVERRIDE_ROW_BUS: DanglingRefDescriptor = DanglingRefDescriptor {
    file: "constraints/penalty_overrides_bus.parquet",
    target_entity: "Bus",
    field: "bus_id",
};
const LINE_PENALTY_OVERRIDE_ROW_LINE: DanglingRefDescriptor = DanglingRefDescriptor {
    file: "constraints/penalty_overrides_line.parquet",
    target_entity: "Line",
    field: "line_id",
};
const HYDRO_PENALTY_OVERRIDE_ROW_HYDRO: DanglingRefDescriptor = DanglingRefDescriptor {
    file: "constraints/penalty_overrides_hydro.parquet",
    target_entity: "Hydro",
    field: "hydro_id",
};
const NCS_PENALTY_OVERRIDE_ROW_NCS: DanglingRefDescriptor = DanglingRefDescriptor {
    file: "constraints/penalty_overrides_ncs.parquet",
    target_entity: "NonControllableSource",
    field: "source_id",
};
const LOAD_FACTOR_ENTRY_BUS: DanglingRefDescriptor = DanglingRefDescriptor {
    file: "scenarios/load_factors.json",
    target_entity: "Bus",
    field: "bus_id",
};
const LOAD_FACTOR_ENTRY_STAGE: DanglingRefDescriptor = DanglingRefDescriptor {
    file: "scenarios/load_factors.json",
    target_entity: "Stage",
    field: "stage_id",
};
const NCS_BOUNDS_ROW_NCS: DanglingRefDescriptor = DanglingRefDescriptor {
    file: "constraints/ncs_bounds.parquet",
    target_entity: "NonControllableSource",
    field: "ncs_id",
};
const NCS_FACTOR_ENTRY_NCS: DanglingRefDescriptor = DanglingRefDescriptor {
    file: "scenarios/non_controllable_factors.json",
    target_entity: "NonControllableSource",
    field: "ncs_id",
};

const LINE_SOURCE_BUS: DanglingRefDescriptor = DanglingRefDescriptor {
    file: "system/lines.json",
    target_entity: "Bus",
    field: "source_bus_id",
};
const LINE_TARGET_BUS: DanglingRefDescriptor = DanglingRefDescriptor {
    file: "system/lines.json",
    target_entity: "Bus",
    field: "target_bus_id",
};
const HYDRO_DOWNSTREAM_HYDRO: DanglingRefDescriptor = DanglingRefDescriptor {
    file: "system/hydros.json",
    target_entity: "Hydro",
    field: "downstream_id",
};
const HYDRO_DIVERSION_DOWNSTREAM_HYDRO: DanglingRefDescriptor = DanglingRefDescriptor {
    file: "system/hydros.json",
    target_entity: "Hydro",
    field: "diversion.downstream_id",
};
const HYDRO_UNIT_GROUP_BUS: DanglingRefDescriptor = DanglingRefDescriptor {
    file: "system/hydros.json",
    target_entity: "Bus",
    field: "bus_id",
};
const THERMAL_BUS: DanglingRefDescriptor = DanglingRefDescriptor {
    file: "system/thermals.json",
    target_entity: "Bus",
    field: "bus_id",
};
const NCS_BUS: DanglingRefDescriptor = DanglingRefDescriptor {
    file: "system/non_controllable_sources.json",
    target_entity: "Bus",
    field: "bus_id",
};
const PUMPING_BUS: DanglingRefDescriptor = DanglingRefDescriptor {
    file: "system/pumping_stations.json",
    target_entity: "Bus",
    field: "bus_id",
};
const PUMPING_SOURCE_HYDRO: DanglingRefDescriptor = DanglingRefDescriptor {
    file: "system/pumping_stations.json",
    target_entity: "Hydro",
    field: "source_hydro_id",
};
const PUMPING_DESTINATION_HYDRO: DanglingRefDescriptor = DanglingRefDescriptor {
    file: "system/pumping_stations.json",
    target_entity: "Hydro",
    field: "destination_hydro_id",
};
const CONTRACT_BUS: DanglingRefDescriptor = DanglingRefDescriptor {
    file: "system/energy_contracts.json",
    target_entity: "Bus",
    field: "bus_id",
};

// ── Per-entity-group helper functions ─────────────────────────────────────────

/// Line -> bus references (`source_bus_id`, `target_bus_id`).
fn check_line_references(data: &ParsedData, ctx: &mut ValidationContext, bus_ids: &HashSet<i32>) {
    for line in &data.lines {
        let entity_str = format!("Line {}", line.id.0);

        if !bus_ids.contains(&line.source_bus_id.0) {
            emit_dangling_ref_at(&LINE_SOURCE_BUS, &entity_str, line.source_bus_id.0, ctx);
        }

        if !bus_ids.contains(&line.target_bus_id.0) {
            emit_dangling_ref_at(&LINE_TARGET_BUS, &entity_str, line.target_bus_id.0, ctx);
        }
    }
}

/// Hydro -> downstream hydro, diversion, and unit group bus references.
fn check_hydro_references(
    data: &ParsedData,
    ctx: &mut ValidationContext,
    bus_ids: &HashSet<i32>,
    hydro_ids: &HashSet<i32>,
) {
    for hydro in &data.hydros {
        let entity_str = format!("Hydro {}", hydro.id.0);

        if let Some(downstream_id) = hydro.downstream_id
            && !hydro_ids.contains(&downstream_id.0)
        {
            emit_dangling_ref_at(&HYDRO_DOWNSTREAM_HYDRO, &entity_str, downstream_id.0, ctx);
        }

        if let Some(ref diversion) = hydro.diversion
            && !hydro_ids.contains(&diversion.downstream_id.0)
        {
            emit_dangling_ref_at(
                &HYDRO_DIVERSION_DOWNSTREAM_HYDRO,
                &entity_str,
                diversion.downstream_id.0,
                ctx,
            );
        }

        for group in &hydro.unit_groups {
            if !bus_ids.contains(&group.bus_id.0) {
                let group_str = format!("{entity_str} unit group {}", group.id.0);
                emit_dangling_ref_at(&HYDRO_UNIT_GROUP_BUS, &group_str, group.bus_id.0, ctx);
            }
        }
    }
}

/// Thermal -> bus reference.
fn check_thermal_references(
    data: &ParsedData,
    ctx: &mut ValidationContext,
    bus_ids: &HashSet<i32>,
) {
    for thermal in &data.thermals {
        let entity_str = format!("Thermal {}", thermal.id.0);

        if !bus_ids.contains(&thermal.bus_id.0) {
            emit_dangling_ref_at(&THERMAL_BUS, &entity_str, thermal.bus_id.0, ctx);
        }
    }
}

/// NCS -> bus reference and NCS model references.
fn check_ncs_references(
    data: &ParsedData,
    ctx: &mut ValidationContext,
    bus_ids: &HashSet<i32>,
    ncs_ids: &HashSet<i32>,
) {
    for ncs in &data.non_controllable_sources {
        let entity_str = format!("NonControllableSource {}", ncs.id.0);

        if !bus_ids.contains(&ncs.bus_id.0) {
            emit_dangling_ref_at(&NCS_BUS, &entity_str, ncs.bus_id.0, ctx);
        }
    }

    for (i, model) in data.ncs_models.iter().enumerate() {
        if !ncs_ids.contains(&model.ncs_id.0) {
            emit_dangling_ref(&NCS_MODEL_NCS, "NcsModel", i, model.ncs_id.0, ctx);
        }
    }
}

/// `PumpingStation` -> bus and hydro references.
fn check_pumping_references(
    data: &ParsedData,
    ctx: &mut ValidationContext,
    bus_ids: &HashSet<i32>,
    hydro_ids: &HashSet<i32>,
) {
    for station in &data.pumping_stations {
        let entity_str = format!("PumpingStation {}", station.id.0);

        if !bus_ids.contains(&station.bus_id.0) {
            emit_dangling_ref_at(&PUMPING_BUS, &entity_str, station.bus_id.0, ctx);
        }

        if !hydro_ids.contains(&station.source_hydro_id.0) {
            emit_dangling_ref_at(
                &PUMPING_SOURCE_HYDRO,
                &entity_str,
                station.source_hydro_id.0,
                ctx,
            );
        }

        if !hydro_ids.contains(&station.destination_hydro_id.0) {
            emit_dangling_ref_at(
                &PUMPING_DESTINATION_HYDRO,
                &entity_str,
                station.destination_hydro_id.0,
                ctx,
            );
        }
    }
}

/// `EnergyContract` -> bus reference.
fn check_contract_references(
    data: &ParsedData,
    ctx: &mut ValidationContext,
    bus_ids: &HashSet<i32>,
) {
    for contract in &data.energy_contracts {
        let entity_str = format!("EnergyContract {}", contract.id.0);

        if !bus_ids.contains(&contract.bus_id.0) {
            emit_dangling_ref_at(&CONTRACT_BUS, &entity_str, contract.bus_id.0, ctx);
        }
    }
}

/// Extension data -> hydro references (geometry, production models, FPHA).
fn check_extension_references(
    data: &ParsedData,
    ctx: &mut ValidationContext,
    hydro_ids: &HashSet<i32>,
) {
    for (i, row) in data.hydro_geometry.iter().enumerate() {
        if !hydro_ids.contains(&row.hydro_id.0) {
            emit_dangling_ref(
                &HYDRO_GEOMETRY_ROW_HYDRO,
                "HydroGeometryRow",
                i,
                row.hydro_id.0,
                ctx,
            );
        }
    }

    for (i, model) in data.production_models.iter().enumerate() {
        if !hydro_ids.contains(&model.hydro_id.0) {
            emit_dangling_ref(
                &PRODUCTION_MODEL_CONFIG_HYDRO,
                "ProductionModelConfig",
                i,
                model.hydro_id.0,
                ctx,
            );
        }
    }

    for (i, row) in data.fpha_hyperplanes.iter().enumerate() {
        if !hydro_ids.contains(&row.hydro_id.0) {
            emit_dangling_ref(
                &FPHA_HYPERPLANE_ROW_HYDRO,
                "FphaHyperplaneRow",
                i,
                row.hydro_id.0,
                ctx,
            );
        }
    }
}

/// Scenario data references.
// Rationale: the scenario data sources are checked in one error-accumulating pass;
// splitting would force multiple passes over `ParsedData` or thread sub-results between
// helpers, obscuring that all checks share one accumulator and one return point.
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
            emit_dangling_ref(
                &INFLOW_SEASONAL_STATS_ROW_HYDRO,
                "InflowSeasonalStatsRow",
                i,
                row.hydro_id.0,
                ctx,
            );
        }
    }

    for (i, row) in data.inflow_ar_coefficients.iter().enumerate() {
        if !hydro_ids.contains(&row.hydro_id.0) {
            emit_dangling_ref(
                &INFLOW_AR_COEFFICIENT_ROW_HYDRO,
                "InflowArCoefficientRow",
                i,
                row.hydro_id.0,
                ctx,
            );
        }
    }

    for (i, row) in data.inflow_annual_components.iter().enumerate() {
        if !hydro_ids.contains(&row.hydro_id.0) {
            emit_dangling_ref(
                &INFLOW_ANNUAL_COMPONENT_ROW_HYDRO,
                "InflowAnnualComponentRow",
                i,
                row.hydro_id.0,
                ctx,
            );
        }
    }

    for (i, row) in data.inflow_history.iter().enumerate() {
        if !hydro_ids.contains(&row.hydro_id.0) {
            emit_dangling_ref(
                &INFLOW_HISTORY_ROW_HYDRO,
                "InflowHistoryRow",
                i,
                row.hydro_id.0,
                ctx,
            );
        }
    }

    for (i, row) in data.load_seasonal_stats.iter().enumerate() {
        if !bus_ids.contains(&row.bus_id.0) {
            emit_dangling_ref(
                &LOAD_SEASONAL_STATS_ROW_BUS,
                "LoadSeasonalStatsRow",
                i,
                row.bus_id.0,
                ctx,
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
                        let descriptor = DanglingRefDescriptor {
                            file: "scenarios/correlation.json",
                            target_entity: registry_label,
                            field: "id",
                        };
                        emit_dangling_ref_at(&descriptor, &entity_str, entity.id.0, ctx);
                    }
                }
            }
        }
    }

    for (i, row) in data.external_scenarios.iter().enumerate() {
        if !hydro_ids.contains(&row.hydro_id.0) {
            emit_dangling_ref(
                &EXTERNAL_SCENARIO_ROW_HYDRO,
                "ExternalScenarioRow",
                i,
                row.hydro_id.0,
                ctx,
            );
        }
    }

    for (i, row) in data.external_load_scenarios.iter().enumerate() {
        if !bus_ids.contains(&row.bus_id.0) {
            emit_dangling_ref(
                &EXTERNAL_LOAD_ROW_BUS,
                "ExternalLoadRow",
                i,
                row.bus_id.0,
                ctx,
            );
        }
    }

    for (i, row) in data.external_ncs_scenarios.iter().enumerate() {
        if !ncs_ids.contains(&row.ncs_id.0) {
            emit_dangling_ref(
                &EXTERNAL_NCS_ROW_NCS,
                "ExternalNcsRow",
                i,
                row.ncs_id.0,
                ctx,
            );
        }
    }
}

/// Bounds rows -> entity references.
fn check_bounds_references(data: &ParsedData, ctx: &mut ValidationContext, ids: &LookupSets) {
    for (i, row) in data.thermal_bounds.iter().enumerate() {
        if !ids.thermal.contains(&row.thermal_id.0) {
            emit_dangling_ref(
                &THERMAL_BOUNDS_ROW_THERMAL,
                "ThermalBoundsRow",
                i,
                row.thermal_id.0,
                ctx,
            );
        }
    }

    for (i, row) in data.hydro_bounds.iter().enumerate() {
        if !ids.hydro.contains(&row.hydro_id.0) {
            emit_dangling_ref(
                &HYDRO_BOUNDS_ROW_HYDRO,
                "HydroBoundsRow",
                i,
                row.hydro_id.0,
                ctx,
            );
        }
    }

    for (i, row) in data.hydro_unit_group_bounds.iter().enumerate() {
        // An unknown hydro_id makes the group check unanswerable, so it is an
        // else-if, not two independent ifs — one finding per row, never both.
        if !ids.hydro.contains(&row.hydro_id.0) {
            emit_dangling_ref(
                &HYDRO_UNIT_GROUP_BOUNDS_ROW_HYDRO,
                "HydroUnitGroupBoundsRow",
                i,
                row.hydro_id.0,
                ctx,
            );
        } else if !ids
            .hydro_unit_group
            .get(&row.hydro_id.0)
            .is_some_and(|groups| groups.contains(&row.hydro_unit_group_id.0))
        {
            ctx.add_error(
                ErrorKind::InvalidReference,
                "constraints/hydro_unit_group_bounds.parquet",
                Some(format!("HydroUnitGroupBoundsRow[{i}]")),
                format!(
                    "HydroUnitGroupBoundsRow[{i}] references non-existent unit group {} of Hydro {} via field 'hydro_unit_group_id'; unit group ids are unique within a plant, not globally",
                    row.hydro_unit_group_id.0, row.hydro_id.0
                ),
            );
        }
    }

    for (i, row) in data.line_bounds.iter().enumerate() {
        if !ids.line.contains(&row.line_id.0) {
            emit_dangling_ref(
                &LINE_BOUNDS_ROW_LINE,
                "LineBoundsRow",
                i,
                row.line_id.0,
                ctx,
            );
        }
    }

    for (i, row) in data.pumping_bounds.iter().enumerate() {
        if !ids.pumping.contains(&row.station_id.0) {
            emit_dangling_ref(
                &PUMPING_BOUNDS_ROW_PUMPING,
                "PumpingBoundsRow",
                i,
                row.station_id.0,
                ctx,
            );
        }
    }

    for (i, row) in data.contract_bounds.iter().enumerate() {
        if !ids.contract.contains(&row.contract_id.0) {
            emit_dangling_ref(
                &CONTRACT_BOUNDS_ROW_CONTRACT,
                "ContractBoundsRow",
                i,
                row.contract_id.0,
                ctx,
            );
        }
    }

    for (i, row) in data.generic_constraint_bounds.iter().enumerate() {
        if !ids.generic_constraint.contains(&row.constraint_id) {
            emit_dangling_ref(
                &GENERIC_CONSTRAINT_BOUNDS_ROW_CONSTRAINT,
                "GenericConstraintBoundsRow",
                i,
                row.constraint_id,
                ctx,
            );
        }
    }
}

/// Penalty override rows -> entity references.
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
            emit_dangling_ref(
                &BUS_PENALTY_OVERRIDE_ROW_BUS,
                "BusPenaltyOverrideRow",
                i,
                row.bus_id.0,
                ctx,
            );
        }
    }

    for (i, row) in data.penalty_overrides_line.iter().enumerate() {
        if !line_ids.contains(&row.line_id.0) {
            emit_dangling_ref(
                &LINE_PENALTY_OVERRIDE_ROW_LINE,
                "LinePenaltyOverrideRow",
                i,
                row.line_id.0,
                ctx,
            );
        }
    }

    for (i, row) in data.penalty_overrides_hydro.iter().enumerate() {
        if !hydro_ids.contains(&row.hydro_id.0) {
            emit_dangling_ref(
                &HYDRO_PENALTY_OVERRIDE_ROW_HYDRO,
                "HydroPenaltyOverrideRow",
                i,
                row.hydro_id.0,
                ctx,
            );
        }
    }

    for (i, row) in data.penalty_overrides_ncs.iter().enumerate() {
        if !ncs_ids.contains(&row.source_id.0) {
            emit_dangling_ref(
                &NCS_PENALTY_OVERRIDE_ROW_NCS,
                "NcsPenaltyOverrideRow",
                i,
                row.source_id.0,
                ctx,
            );
        }
    }
}

/// Study stage IDs; the negative-id pre-study stages are excluded.
fn collect_study_stage_ids(data: &ParsedData) -> HashSet<i32> {
    data.stages
        .stages
        .iter()
        .filter(|s| s.id >= 0)
        .map(|s| s.id)
        .collect()
}

/// `LoadFactorEntry` -> bus and stage references.
fn check_load_factor_references(
    data: &ParsedData,
    ctx: &mut ValidationContext,
    bus_ids: &HashSet<i32>,
) {
    let study_stage_ids = collect_study_stage_ids(data);

    for (i, entry) in data.load_factors.iter().enumerate() {
        if !bus_ids.contains(&entry.bus_id.0) {
            emit_dangling_ref(
                &LOAD_FACTOR_ENTRY_BUS,
                "LoadFactorEntry",
                i,
                entry.bus_id.0,
                ctx,
            );
        }

        if !study_stage_ids.contains(&entry.stage_id) {
            emit_dangling_ref(
                &LOAD_FACTOR_ENTRY_STAGE,
                "LoadFactorEntry",
                i,
                entry.stage_id,
                ctx,
            );
        }
    }
}

/// `GenericConstraint` expression entity ID existence.
fn check_generic_constraint_expression_references(
    data: &ParsedData,
    ctx: &mut ValidationContext,
    ids: &LookupSets,
) {
    for constraint in &data.generic_constraints {
        let gc_label = format!("GenericConstraint {}", constraint.id.0);
        for (term_idx, term) in constraint.expression.terms.iter().enumerate() {
            let label = format!("{gc_label} term[{term_idx}]");
            validate_variable_ref_entity(&term.variable, &label, ids, ctx);
        }
    }
}

/// A generic-constraint endpoint's fold, resolved to one scalar only when it is
/// static: `literal` alone with no affine remainder, or
/// `literal.unwrap_or(0.0) + affine.constant` when the remainder carries no
/// `@param` terms. A `@param`-bearing remainder makes the endpoint
/// stage-varying — `None` here, deferring its value (and any inversion
/// against the other endpoint) to the LP builder's `fold_endpoint`.
fn static_fold(literal: Option<f64>, affine: Option<&AffineBound>) -> Option<f64> {
    match affine {
        None => literal,
        Some(bound) if bound.terms.is_empty() => Some(literal.unwrap_or(0.0) + bound.constant),
        Some(_) => None,
    }
}

/// `GenericConstraintBoundsRow` per-row endpoint-interval checks: at least one
/// endpoint present, and same-row static-fold inversion. `block_id` range and
/// duplicate-key detection are owned by the Layer 5a bound-override family
/// (`validation::semantic::block_bounds`), alongside the other six families.
fn check_generic_constraint_bounds_validity(data: &ParsedData, ctx: &mut ValidationContext) {
    // Which affine remainder (if any) each constraint assigns to each endpoint,
    // keyed by constraint id. A parquet numeric endpoint and an affine remainder
    // on the same endpoint compose (`base + R`, `fold_endpoint` in the LP
    // builder) rather than conflict — any remainder shape, constant-only or
    // `@param`-bearing, is a legal fold.
    let constraint_affines: HashMap<i32, (Option<&AffineBound>, Option<&AffineBound>)> = data
        .generic_constraints
        .iter()
        .map(|gc| {
            (
                gc.id.0,
                (
                    gc.bound_lower_affine.as_ref(),
                    gc.bound_upper_affine.as_ref(),
                ),
            )
        })
        .collect();

    // The interval IS the constraint: shape derives from which endpoints a row
    // carries — a numeric column here or an affine remainder on the constraint.
    // A dangling `constraint_id` is reported separately by `check_bounds_references`.
    for (i, row) in data.generic_constraint_bounds.iter().enumerate() {
        let (lower_affine, upper_affine) = constraint_affines
            .get(&row.constraint_id)
            .copied()
            .unwrap_or((None, None));

        if row.bound_lower.is_none()
            && row.bound_upper.is_none()
            && lower_affine.is_none()
            && upper_affine.is_none()
        {
            ctx.add_error(
                ErrorKind::InvalidValue,
                "constraints/generic_constraint_bounds.parquet",
                Some(format!("GenericConstraintBoundsRow[{i}]")),
                format!(
                    "GenericConstraintBoundsRow[{i}] on constraint {} has neither bound_lower nor bound_upper: at least one endpoint is required",
                    row.constraint_id
                ),
            );
        }

        // Only a same-row STATIC fold is checked here (an affine remainder with
        // no `@param` terms resolves to one scalar regardless of stage/block); a
        // `@param`-bearing remainder makes the endpoint stage-varying, so its
        // inversion is left to LP infeasibility rather than a combinatorial
        // pre-solve check — matching how a cross-source bound is validated only
        // for its same-row static interval elsewhere in this module.
        let static_lower = static_fold(row.bound_lower, lower_affine);
        let static_upper = static_fold(row.bound_upper, upper_affine);
        if let (Some(bound_lower), Some(bound_upper)) = (static_lower, static_upper)
            && bound_upper < bound_lower
        {
            ctx.add_error(
                ErrorKind::InvalidValue,
                "constraints/generic_constraint_bounds.parquet",
                Some(format!("GenericConstraintBoundsRow[{i}]")),
                format!(
                    "GenericConstraintBoundsRow[{i}] on constraint {} has bound_upper={bound_upper} less than bound_lower={bound_lower}: an inverted interval is not allowed",
                    row.constraint_id
                ),
            );
        }
    }

    // The parquet is the activation grid: a symbolic bound supplies a value, but the
    // constraint's applicable (stage, block) cells still come from its parquet rows.
    // A reference with no rows would apply to nothing and be silently inert.
    let constraints_with_rows: HashSet<i32> = data
        .generic_constraint_bounds
        .iter()
        .map(|row| row.constraint_id)
        .collect();
    for gc in &data.generic_constraints {
        if (gc.bound_lower_affine.is_some() || gc.bound_upper_affine.is_some())
            && !constraints_with_rows.contains(&gc.id.0)
        {
            ctx.add_error(
                ErrorKind::InvalidReference,
                "constraints/generic_constraints.json",
                Some(format!("GenericConstraint {}", gc.id.0)),
                format!(
                    "GenericConstraint {} declares a bound reference but has no activation rows in generic_constraint_bounds.parquet: the reference would apply to nothing",
                    gc.id.0
                ),
            );
        }
    }
}

/// NCS bounds and NCS factor entry checks: dangling `ncs_id`/`stage_id`
/// references only. `available_generation_mw`/`factor` sign is a parse-layer
/// contract (`constraints/ncs_bounds.rs`'s and
/// `scenarios/non_controllable_factors.rs`'s parsers reject a negative value
/// with `LoadError::SchemaError` before this layer runs), not re-checked here.
fn check_ncs_bounds_and_factors(
    data: &ParsedData,
    ctx: &mut ValidationContext,
    ncs_ids: &HashSet<i32>,
) {
    let study_stage_ids = collect_study_stage_ids(data);

    for (i, row) in data.ncs_bounds.iter().enumerate() {
        if !ncs_ids.contains(&row.ncs_id.0) {
            emit_dangling_ref(&NCS_BOUNDS_ROW_NCS, "NcsBoundsRow", i, row.ncs_id.0, ctx);
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
    }

    for (i, entry) in data.non_controllable_factors.iter().enumerate() {
        if !ncs_ids.contains(&entry.ncs_id.0) {
            emit_dangling_ref(
                &NCS_FACTOR_ENTRY_NCS,
                "NcsFactorEntry",
                i,
                entry.ncs_id.0,
                ctx,
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
    }
}

/// Validate that a [`VariableRef`](cobre_core::VariableRef) references an existing entity.
///
/// A dangling reference is an [`ErrorKind::InvalidReference`] error for every
/// modeled entity type. `Contract` is the sole remaining stub (data-complete but
/// contributing no LP variables), so a dangling `Contract` reference is downgraded
/// to an [`ErrorKind::UnusedEntity`] warning, not an error.
fn validate_variable_ref_entity(
    var: &cobre_core::VariableRef,
    label: &str,
    ids: &LookupSets,
    ctx: &mut ValidationContext,
) {
    use cobre_core::VariableRef;

    let file = "system/generic_constraints.json";
    match var {
        VariableRef::HydroStorage { hydro_id, .. }
        | VariableRef::HydroEvaporation { hydro_id, .. }
        | VariableRef::HydroWithdrawal { hydro_id, .. }
        | VariableRef::HydroSpillage { hydro_id, .. }
        | VariableRef::HydroDiversion { hydro_id, .. }
        | VariableRef::HydroOutflow { hydro_id, .. }
        | VariableRef::HydroInflow { hydro_id, .. }
        | VariableRef::HydroStorageInitial { hydro_id, .. }
        | VariableRef::HydroStorageFinal { hydro_id, .. }
        | VariableRef::HydroUsefulVolumeInitial { hydro_id, .. }
        | VariableRef::HydroUsefulVolumeFinal { hydro_id, .. } => {
            if !ids.hydro.contains(&hydro_id.0) {
                ctx.add_error(
                    ErrorKind::InvalidReference,
                    file,
                    Some(label.to_string()),
                    format!("{label} references non-existent Hydro {}", hydro_id.0),
                );
            }
        }
        VariableRef::HydroTurbined {
            hydro_id, bus_id, ..
        }
        | VariableRef::HydroGeneration {
            hydro_id, bus_id, ..
        } => {
            if !ids.hydro.contains(&hydro_id.0) {
                ctx.add_error(
                    ErrorKind::InvalidReference,
                    file,
                    Some(label.to_string()),
                    format!("{label} references non-existent Hydro {}", hydro_id.0),
                );
            } else if let Some(b) = bus_id
                && !ids
                    .hydro_group_bus
                    .get(&hydro_id.0)
                    .is_some_and(|buses| buses.contains(&b.0))
            {
                ctx.add_error(
                    ErrorKind::InvalidReference,
                    file,
                    Some(label.to_string()),
                    format!(
                        "{label} references bus {}, on which Hydro {} has no unit group, via field 'bus_id'; a bus selector names one side of a split plant, not any bus in the system",
                        b.0, hydro_id.0
                    ),
                );
            }
        }
        VariableRef::ThermalGeneration { thermal_id, .. }
        | VariableRef::AnticipatedDecision { thermal_id, .. } => {
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
                ctx.add_error(
                    ErrorKind::InvalidReference,
                    file,
                    Some(label.to_string()),
                    format!(
                        "{label} references non-existent PumpingStation {}",
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
                ctx.add_error(
                    ErrorKind::InvalidReference,
                    file,
                    Some(label.to_string()),
                    format!(
                        "{label} references non-existent NonControllableSource {}",
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
    use chrono::NaiveDate;
    use cobre_core::{
        EntityId,
        entities::{
            Bus, DiversionChannel, HydroUnitGroup, Line, NonControllableSource, PumpingStation,
            Thermal,
        },
        scenario::{CorrelationEntity, CorrelationGroup, CorrelationModel, CorrelationProfile},
    };
    use std::collections::BTreeMap;
    use tempfile::TempDir;

    use crate::{
        constraints::{
            BusPenaltyOverrideRow, GenericConstraintBoundsRow, HydroBoundsRow,
            HydroUnitGroupBoundsRow, LineBoundsRow, NcsBoundsRow, NcsPenaltyOverrideRow,
            ThermalBoundsRow,
        },
        extensions::HydroGeometryRow,
        scenarios::{
            BlockFactor, InflowSeasonalStatsRow, LoadFactorEntry, LoadSeasonalStatsRow,
            NcsFactorEntry,
        },
        test_support::{make_hydro, make_minimal_case, make_unit_group},
        validation::{
            schema::{ParsedData, validate_schema},
            structural::validate_structure,
        },
    };

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

    fn make_line(id: i32, source_bus: i32, target_bus: i32) -> Line {
        Line {
            id: EntityId::from(id),
            name: format!("Line_{id}"),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
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
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId::from(bus_id),
            entry_stage_id: None,
            exit_stage_id: None,
            max_generation_mw: 50.0,
            allow_curtailment: true,
            curtailment_cost: 1.0,
        }
    }

    fn make_pumping(id: i32, bus_id: i32, src_hydro: i32, dst_hydro: i32) -> PumpingStation {
        PumpingStation {
            id: EntityId::from(id),
            name: format!("Pump_{id}"),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
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
        let mut hydro = make_hydro(3, None);
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
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId::from(777), // bad
            entry_stage_id: None,
            exit_stage_id: None,
            cost_per_mwh: 50.0,
            min_generation_mw: 0.0,
            max_generation_mw: 100.0,
            anticipated_config: None,
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

    /// A hydro with no declared unit groups produces no referential error —
    /// the group-bus check has nothing to iterate.
    #[test]
    fn test_hydro_valid_bus_ref() {
        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = parse_case(&dir);
        data.hydros = vec![make_hydro(10, None)];
        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);
        assert!(!ctx.has_errors());
    }

    /// A hydro whose plant `bus_id` names a nonexistent bus is no longer
    /// checked at plant level: with a group on a valid bus, Layer 3 reports
    /// no error at all.
    #[test]
    fn test_hydro_invalid_plant_bus_with_valid_group_bus_produces_no_error() {
        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = parse_case(&dir);
        let mut hydro = make_hydro(10, None);
        hydro.unit_groups = vec![make_unit_group(0, 1, 0.0, 100.0, 0.0, 100.0)]; // group bus 1 exists
        data.hydros = vec![hydro];
        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);
        assert!(!ctx.has_errors());
    }

    /// A hydro whose unit group's `bus_id` equals its own (also nonexistent)
    /// plant `bus_id` is rejected. Exactly one `InvalidReference` is
    /// reported and it names the unit group, not the plant.
    #[test]
    fn test_hydro_group_bus_equals_invalid_plant_bus_is_rejected() {
        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = parse_case(&dir);
        let mut hydro = make_hydro(10, None);
        hydro.unit_groups = vec![make_unit_group(0, 999, 0.0, 100.0, 0.0, 100.0)]; // group bus also 999
        data.hydros = vec![hydro];
        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);
        let inv: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::InvalidReference)
            .collect();
        assert_eq!(inv.len(), 1);
        assert!(inv[0].message.contains("Hydro 10 unit group 0"));
        assert!(!inv[0].message.contains("Hydro 10 references"));
        assert!(inv[0].message.contains("Bus 999"));
    }

    /// Hydro with `downstream_id = None` must not produce any error.
    #[test]
    fn test_hydro_downstream_id_none_no_error() {
        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = parse_case(&dir);
        let mut hydro = make_hydro(10, None);
        hydro.downstream_id = None;
        data.hydros = vec![hydro];
        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);
        assert!(
            !ctx.has_errors(),
            "downstream_id = None should not produce errors"
        );
    }

    /// Hydro with `diversion = None` must not produce any error.
    #[test]
    fn test_hydro_diversion_none_no_error() {
        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = parse_case(&dir);
        let mut hydro = make_hydro(10, None);
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
        let mut hydro = make_hydro(10, None);
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

    /// Given a two-hydro study with buses `{0, 1}` declared, where hydro 1's
    /// unit group sits on bus 0 (valid, distinct from its own bus 1) and hydro
    /// 2's unit group sits on bus 42 (nonexistent), exactly one
    /// `InvalidReference` is emitted naming hydro 2 and its group, and hydro 1
    /// produces no finding.
    #[test]
    fn test_unit_group_on_nonexistent_bus_is_rejected() {
        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = parse_case(&dir);
        data.buses.push(Bus {
            id: EntityId::from(0),
            name: "BUS_0".to_string(),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            deficit_segments: vec![],
            excess_cost: 100.0,
        });

        let mut hydro1 = make_hydro(1, None);
        hydro1.unit_groups = vec![HydroUnitGroup {
            id: EntityId::from(4),
            name: "Group A".to_string(),
            bus_id: EntityId::from(0), // bus 0 exists, distinct from hydro1's own bus 1
            min_generation_mw: 0.0,
            max_generation_mw: 100.0,
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 100.0,
        }];

        let mut hydro2 = make_hydro(2, None);
        hydro2.unit_groups = vec![HydroUnitGroup {
            id: EntityId::from(7),
            name: "Group B".to_string(),
            bus_id: EntityId::from(42), // does not exist
            min_generation_mw: 0.0,
            max_generation_mw: 100.0,
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 100.0,
        }];

        data.hydros = vec![hydro1, hydro2];

        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);
        let inv: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::InvalidReference)
            .collect();
        assert_eq!(
            inv.len(),
            1,
            "expected exactly 1 InvalidReference, got: {:?}",
            inv.iter().map(|e| &e.message).collect::<Vec<_>>()
        );
        assert!(
            inv[0].message.contains("Hydro 2 unit group 7"),
            "message should name Hydro 2 unit group 7, got: {}",
            inv[0].message
        );
        assert!(inv[0].message.contains("Bus 42"));
        assert!(inv[0].message.contains("bus_id"));
        assert!(
            !inv.iter().any(|e| e.message.contains("Hydro 1")),
            "hydro 1's valid-bus group must produce no finding, got: {:?}",
            inv.iter().map(|e| &e.message).collect::<Vec<_>>()
        );
    }

    /// PumpingStation with valid bus and hydro references produces no error.
    #[test]
    fn test_pumping_valid_refs() {
        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = parse_case(&dir);
        data.hydros = vec![make_hydro(10, None)];
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
        data.hydros = vec![make_hydro(10, None)];
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

    /// PumpingStation referencing a non-existent bus produces 1 error.
    #[test]
    fn test_pumping_invalid_bus() {
        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = parse_case(&dir);
        data.hydros = vec![make_hydro(10, None)];
        // bus 777 missing; source/destination hydro 10 exists
        data.pumping_stations = vec![make_pumping(1, 777, 10, 10)];
        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);
        assert!(ctx.has_errors());
        let inv: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::InvalidReference)
            .collect();
        assert_eq!(inv.len(), 1);
        assert!(inv[0].message.contains("bus_id"));
        assert!(inv[0].message.contains("777"));
    }

    /// PumpingStation referencing a non-existent destination hydro produces 1 error.
    #[test]
    fn test_pumping_invalid_destination_hydro() {
        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = parse_case(&dir);
        data.hydros = vec![make_hydro(10, None)];
        // source hydro 10 exists; destination hydro 999 missing
        data.pumping_stations = vec![make_pumping(1, 1, 10, 999)];
        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);
        assert!(ctx.has_errors());
        let inv: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::InvalidReference)
            .collect();
        assert_eq!(inv.len(), 1);
        assert!(inv[0].message.contains("destination_hydro_id"));
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

    /// `CorrelationEntity` with a dangling inflow reference and one with an
    /// unknown `entity_type` each produce one `InvalidReference` error.
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
                            entity_type: "unknown".to_string(),
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
        data.hydros = vec![make_hydro(10, None)];
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
            cost_per_mwh: None,
            block_id: None,
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
            ..Default::default()
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

    /// A `hydro_unit_group_bounds` row naming unit group 4 on Hydro 7 (which
    /// declares groups `{0, 3}`) is rejected even though group 4 exists on a
    /// different plant (Hydro 2) — group ids are plant-scoped, not global.
    #[test]
    fn test_hydro_unit_group_bounds_unknown_group_ref() {
        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = parse_case(&dir);

        let mut hydro7 = make_hydro(7, None);
        hydro7.unit_groups = vec![
            make_unit_group(0, 1, 0.0, 100.0, 0.0, 100.0),
            make_unit_group(3, 1, 0.0, 100.0, 0.0, 100.0),
        ];
        let mut hydro2 = make_hydro(2, None);
        hydro2.unit_groups = vec![make_unit_group(4, 1, 0.0, 100.0, 0.0, 100.0)];
        data.hydros = vec![hydro7, hydro2];

        data.hydro_unit_group_bounds = vec![HydroUnitGroupBoundsRow {
            hydro_id: EntityId::from(7),
            hydro_unit_group_id: EntityId::from(4),
            stage_id: 9,
            min_turbined_m3s: None,
            max_turbined_m3s: None,
            min_generation_mw: None,
            max_generation_mw: Some(50.0),
            block_id: Some(1),
        }];

        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);
        let inv: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::InvalidReference)
            .collect();
        assert_eq!(
            inv.len(),
            1,
            "expected exactly 1 InvalidReference, got: {:?}",
            inv.iter().map(|e| &e.message).collect::<Vec<_>>()
        );
        assert!(inv[0].file == std::path::Path::new("constraints/hydro_unit_group_bounds.parquet"));
        assert!(inv[0].message.contains("unit group 4"));
        assert!(inv[0].message.contains("Hydro 7"));
    }

    /// A `hydro_unit_group_bounds` row with `hydro_id = 99` (no such plant)
    /// emits exactly one finding — the dangling `hydro_id` — and no
    /// `hydro_unit_group_id` finding, even though group 4 does not exist on
    /// plant 99 either: the plant reference is unanswerable first.
    #[test]
    fn test_hydro_unit_group_bounds_unknown_hydro_ref_emits_one_finding() {
        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = parse_case(&dir);

        let mut hydro7 = make_hydro(7, None);
        hydro7.unit_groups = vec![
            make_unit_group(0, 1, 0.0, 100.0, 0.0, 100.0),
            make_unit_group(3, 1, 0.0, 100.0, 0.0, 100.0),
        ];
        data.hydros = vec![hydro7];

        data.hydro_unit_group_bounds = vec![HydroUnitGroupBoundsRow {
            hydro_id: EntityId::from(99),
            hydro_unit_group_id: EntityId::from(4),
            stage_id: 9,
            min_turbined_m3s: None,
            max_turbined_m3s: None,
            min_generation_mw: None,
            max_generation_mw: Some(50.0),
            block_id: Some(1),
        }];

        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);
        let inv: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::InvalidReference)
            .collect();
        assert_eq!(
            inv.len(),
            1,
            "expected exactly 1 InvalidReference (hydro_id only), got: {:?}",
            inv.iter().map(|e| &e.message).collect::<Vec<_>>()
        );
        assert!(inv[0].message.contains("Hydro 99"));
        assert!(inv[0].message.contains("hydro_id"));
        assert!(!inv[0].message.contains("hydro_unit_group_id"));
    }

    /// Every `hydro_unit_group_bounds` row names a declared `(plant, group)`
    /// pair on plants whose group ids are neither `0` nor equal to their own
    /// position in `unit_groups` — no finding is produced against
    /// `constraints/hydro_unit_group_bounds.parquet`.
    #[test]
    fn test_hydro_unit_group_bounds_valid_refs_no_error() {
        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = parse_case(&dir);

        let mut hydro5 = make_hydro(5, None);
        hydro5.unit_groups = vec![
            make_unit_group(7, 1, 0.0, 100.0, 0.0, 100.0),
            make_unit_group(2, 1, 0.0, 100.0, 0.0, 100.0),
        ];
        let mut hydro6 = make_hydro(6, None);
        hydro6.unit_groups = vec![
            make_unit_group(10, 1, 0.0, 100.0, 0.0, 100.0),
            make_unit_group(20, 1, 0.0, 100.0, 0.0, 100.0),
        ];
        data.hydros = vec![hydro5, hydro6];

        data.hydro_unit_group_bounds = vec![
            HydroUnitGroupBoundsRow {
                hydro_id: EntityId::from(5),
                hydro_unit_group_id: EntityId::from(2),
                stage_id: 8,
                min_turbined_m3s: None,
                max_turbined_m3s: None,
                min_generation_mw: None,
                max_generation_mw: Some(50.0),
                block_id: Some(3),
            },
            HydroUnitGroupBoundsRow {
                hydro_id: EntityId::from(6),
                hydro_unit_group_id: EntityId::from(10),
                stage_id: 4,
                min_turbined_m3s: Some(1.0),
                max_turbined_m3s: None,
                min_generation_mw: None,
                max_generation_mw: None,
                block_id: None,
            },
        ];

        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);
        let inv: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::InvalidReference)
            .filter(|e| {
                e.file == std::path::Path::new("constraints/hydro_unit_group_bounds.parquet")
            })
            .collect();
        assert!(
            inv.is_empty(),
            "expected no hydro_unit_group_bounds errors, got: {:?}",
            inv.iter().map(|e| &e.message).collect::<Vec<_>>()
        );
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
            block_id: None,
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
            bound_lower: Some(1000.0),
            bound_upper: None,
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

    /// A bounds row with neither endpoint present emits exactly one finding
    /// naming the constraint id — no constraint-registry lookup is needed, the
    /// check is per-row.
    #[test]
    fn test_generic_constraint_bounds_rejects_both_absent() {
        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = parse_case(&dir);

        data.generic_constraint_bounds = vec![GenericConstraintBoundsRow {
            constraint_id: 1,
            stage_id: 0,
            block_id: None,
            bound_lower: None,
            bound_upper: None,
        }];

        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);

        let inv: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::InvalidValue)
            .collect();
        assert_eq!(
            inv.len(),
            1,
            "expected exactly 1 InvalidValue, got: {inv:?}"
        );
        assert!(inv[0].message.contains("constraint 1"));
    }

    fn symbolic_constraint(
        id: i32,
        lower_ref: Option<i32>,
        upper_ref: Option<i32>,
    ) -> cobre_core::GenericConstraint {
        use cobre_core::{AffineBound, ConstraintExpression, GenericConstraint, SlackConfig};

        GenericConstraint {
            id: EntityId(id),
            name: format!("sym{id}"),
            description: None,
            expression: ConstraintExpression { terms: vec![] },
            slack: SlackConfig {
                enabled: false,
                penalty: None,
            },
            bound_lower_affine: lower_ref.map(|id| AffineBound::single(EntityId(id))),
            bound_upper_affine: upper_ref.map(|id| AffineBound::single(EntityId(id))),
        }
    }

    /// A numeric parquet endpoint composed with a constant-only affine remainder
    /// on the same endpoint folds (`base + R`) rather than conflicts: zero errors.
    #[test]
    fn test_generic_constraint_bounds_literal_and_constant_affine_fold_adds_no_error() {
        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = parse_case(&dir);

        let mut constraint = symbolic_constraint(5, None, None);
        constraint.bound_upper_affine = Some(AffineBound {
            constant: -5.0,
            terms: vec![],
        });
        data.generic_constraints = vec![constraint];
        data.generic_constraint_bounds = vec![GenericConstraintBoundsRow {
            constraint_id: 5,
            stage_id: 0,
            block_id: None,
            bound_lower: None,
            bound_upper: Some(100.0),
        }];

        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);

        assert!(
            !ctx.has_errors(),
            "a literal base composed with a constant-only remainder is a legal fold, got: {:?}",
            ctx.errors()
        );
    }

    /// A numeric parquet endpoint composed with a `@param`-bearing affine
    /// remainder on the same endpoint also folds — the fold does not
    /// distinguish a constant remainder from a symbolic one: zero errors.
    #[test]
    fn test_generic_constraint_bounds_literal_and_param_affine_fold_adds_no_error() {
        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = parse_case(&dir);

        data.generic_constraints = vec![symbolic_constraint(5, None, Some(99))];
        data.generic_constraint_bounds = vec![GenericConstraintBoundsRow {
            constraint_id: 5,
            stage_id: 0,
            block_id: None,
            bound_lower: None,
            bound_upper: Some(100.0),
        }];

        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);

        assert!(
            !ctx.has_errors(),
            "a literal base composed with a @param-bearing remainder is a legal fold, got: {:?}",
            ctx.errors()
        );
    }

    /// A constant-only affine remainder can fold an endpoint below the other
    /// side even when the raw parquet columns alone are not inverted: here
    /// `bound_lower=50`, `bound_upper=60` compare fine on their own, but the
    /// upper endpoint's `-30` remainder folds it to `30`, which IS below the
    /// lower endpoint — the inversion check must catch this static fold.
    #[test]
    fn test_generic_constraint_bounds_constant_fold_reveals_inversion() {
        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = parse_case(&dir);

        let mut constraint = symbolic_constraint(5, None, None);
        constraint.bound_upper_affine = Some(AffineBound {
            constant: -30.0,
            terms: vec![],
        });
        data.generic_constraints = vec![constraint];
        data.generic_constraint_bounds = vec![GenericConstraintBoundsRow {
            constraint_id: 5,
            stage_id: 0,
            block_id: None,
            bound_lower: Some(50.0),
            bound_upper: Some(60.0),
        }];

        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);

        let inv: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::InvalidValue)
            .collect();
        assert_eq!(
            inv.len(),
            1,
            "the folded upper (60 + -30 = 30) is below the lower (50); expected exactly 1 InvalidValue, got: {inv:?}"
        );
        assert!(inv[0].message.contains("constraint 5"));
        assert!(inv[0].message.contains("bound_upper"));
    }

    /// A `@param`-bearing remainder that COULD invert its endpoint is not
    /// statically resolvable, so the inversion check must defer to the LP
    /// rather than false-reject: same shape as the constant-fold case above,
    /// but the upper remainder carries a parameter term.
    #[test]
    fn test_generic_constraint_bounds_param_fold_defers_inversion_check() {
        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = parse_case(&dir);

        data.generic_constraints = vec![symbolic_constraint(5, None, Some(99))];
        data.generic_constraint_bounds = vec![GenericConstraintBoundsRow {
            constraint_id: 5,
            stage_id: 0,
            block_id: None,
            bound_lower: Some(50.0),
            bound_upper: Some(60.0),
        }];

        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);

        assert!(
            !ctx.has_errors(),
            "a @param-bearing remainder is stage-varying; its inversion is left to LP infeasibility, got: {:?}",
            ctx.errors()
        );
    }

    /// A both-numeric-null row is valid when the constraint supplies a reference for
    /// a side (the reference fills it): no InvalidValue.
    #[test]
    fn test_generic_constraint_bounds_ref_fills_side_allows_both_null() {
        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = parse_case(&dir);

        data.generic_constraints = vec![symbolic_constraint(5, None, Some(99))];
        data.generic_constraint_bounds = vec![GenericConstraintBoundsRow {
            constraint_id: 5,
            stage_id: 0,
            block_id: None,
            bound_lower: None,
            bound_upper: None,
        }];

        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);

        assert!(
            !ctx.has_errors(),
            "a reference fills the endpoint, so a both-null row is valid, got: {:?}",
            ctx.errors()
        );
    }

    /// A constraint declaring a reference but with no rows in the bounds parquet is an
    /// InvalidReference naming the constraint id — the reference would apply to nothing.
    #[test]
    fn test_generic_constraint_bounds_ref_with_no_rows_is_invalid_reference() {
        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = parse_case(&dir);

        data.generic_constraints = vec![symbolic_constraint(5, None, Some(99))];
        data.generic_constraint_bounds = vec![];

        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);

        let inv: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::InvalidReference)
            .collect();
        assert_eq!(
            inv.len(),
            1,
            "expected exactly 1 InvalidReference, got: {inv:?}"
        );
        assert!(inv[0].message.contains("GenericConstraint 5"));
    }

    /// A bounds row with `bound_upper < bound_lower` (an inverted interval)
    /// emits exactly one finding naming the constraint id.
    #[test]
    fn test_generic_constraint_bounds_rejects_inverted_interval() {
        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = parse_case(&dir);

        data.generic_constraint_bounds = vec![GenericConstraintBoundsRow {
            constraint_id: 1,
            stage_id: 0,
            block_id: None,
            bound_lower: Some(20.0),
            bound_upper: Some(5.0),
        }];

        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);

        let inv: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::InvalidValue)
            .collect();
        assert_eq!(
            inv.len(),
            1,
            "expected exactly 1 InvalidValue, got: {inv:?}"
        );
        assert!(inv[0].message.contains("constraint 1"));
        assert!(inv[0].message.contains("bound_upper"));
    }

    /// A bounds row with `bound_upper == bound_lower` (a degenerate equal band)
    /// is accepted — it is an equality, not an authoring error.
    #[test]
    fn test_generic_constraint_bounds_accepts_degenerate_equal_band() {
        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = parse_case(&dir);

        data.generic_constraint_bounds = vec![GenericConstraintBoundsRow {
            constraint_id: 1,
            stage_id: 0,
            block_id: None,
            bound_lower: Some(10.0),
            bound_upper: Some(10.0),
        }];

        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);

        let inv: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::InvalidValue)
            .collect();
        assert!(
            inv.is_empty(),
            "a degenerate equal band must not be rejected, got: {inv:?}"
        );
    }

    /// A lower-only row (one-sided) is accepted.
    #[test]
    fn test_generic_constraint_bounds_accepts_lower_only_row() {
        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = parse_case(&dir);

        data.generic_constraint_bounds = vec![GenericConstraintBoundsRow {
            constraint_id: 1,
            stage_id: 0,
            block_id: None,
            bound_lower: Some(5.0),
            bound_upper: None,
        }];

        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);

        let inv: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::InvalidValue)
            .collect();
        assert!(
            inv.is_empty(),
            "a lower-only row must not be rejected, got: {inv:?}"
        );
    }

    /// An upper-only row (one-sided) is accepted.
    #[test]
    fn test_generic_constraint_bounds_accepts_upper_only_row() {
        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = parse_case(&dir);

        data.generic_constraint_bounds = vec![GenericConstraintBoundsRow {
            constraint_id: 1,
            stage_id: 0,
            block_id: None,
            bound_lower: None,
            bound_upper: Some(20.0),
        }];

        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);

        let inv: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::InvalidValue)
            .collect();
        assert!(
            inv.is_empty(),
            "an upper-only row must not be rejected, got: {inv:?}"
        );
    }

    /// A well-formed two-sided band (`bound_upper > bound_lower`) produces no
    /// `InvalidValue` finding.
    #[test]
    fn test_generic_constraint_bounds_accepts_two_sided_band() {
        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = parse_case(&dir);

        data.generic_constraint_bounds = vec![GenericConstraintBoundsRow {
            constraint_id: 1,
            stage_id: 0,
            block_id: None,
            bound_lower: Some(5.0),
            bound_upper: Some(20.0),
        }];

        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);

        let inv: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::InvalidValue)
            .collect();
        assert!(
            inv.is_empty(),
            "a well-formed two-sided band must not be rejected, got: {inv:?}"
        );
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

    // ── AnticipatedDecision referential validation ─────────────────────

    /// A constraint with `anticipated_decision(99)` where Thermal 99 does
    /// not exist produces `ErrorKind::InvalidReference` naming Thermal 99 and
    /// including the constraint id in the context.
    #[test]
    fn test_anticipated_decision_unknown_thermal_ref() {
        use cobre_core::{
            ConstraintExpression, GenericConstraint, LinearTerm, SlackConfig, VariableRef,
        };

        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = parse_case(&dir);

        // Build a generic constraint referencing Thermal 99, which does not exist.
        let gc = GenericConstraint {
            id: EntityId::from(1),
            name: "test_constraint".to_string(),
            description: None,
            expression: ConstraintExpression {
                terms: vec![LinearTerm::literal(
                    1.0,
                    VariableRef::AnticipatedDecision {
                        thermal_id: EntityId::from(99),
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
        data.generic_constraints = vec![gc];

        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);
        assert!(ctx.has_errors(), "expected referential errors");

        let inv: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::InvalidReference)
            .collect();
        assert_eq!(
            inv.len(),
            1,
            "expected exactly 1 InvalidReference, got: {inv:?}"
        );
        assert!(
            inv[0].message.contains("99"),
            "error message must name Thermal 99, got: {}",
            inv[0].message
        );
        assert!(
            inv[0].message.contains("Thermal"),
            "error message must include 'Thermal', got: {}",
            inv[0].message
        );
    }

    #[test]
    fn test_hydro_inflow_unknown_hydro_ref() {
        use cobre_core::{
            ConstraintExpression, GenericConstraint, LinearTerm, SlackConfig, VariableRef,
        };

        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = parse_case(&dir);

        // Build a generic constraint referencing Hydro 99, which does not exist.
        let gc = GenericConstraint {
            id: EntityId::from(1),
            name: "test_constraint".to_string(),
            description: None,
            expression: ConstraintExpression {
                terms: vec![LinearTerm::literal(
                    1.0,
                    VariableRef::HydroInflow {
                        hydro_id: EntityId::from(99),
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
        data.generic_constraints = vec![gc];

        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);
        assert!(ctx.has_errors(), "expected referential errors");

        let inv: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::InvalidReference)
            .collect();
        assert_eq!(
            inv.len(),
            1,
            "expected exactly 1 InvalidReference, got: {inv:?}"
        );
        assert!(
            inv[0].message.contains("non-existent Hydro 99"),
            "error message must name non-existent Hydro 99, got: {}",
            inv[0].message
        );
    }

    /// A block-qualified `hydro_inflow(99, 0)` referencing a non-existent hydro is also
    /// flagged: the hydro-bearing arm's `..` pattern absorbs `block_id`, so the
    /// `Some(_)` form validates identically to the `None` form.
    #[test]
    fn test_hydro_inflow_with_block_unknown_hydro_ref() {
        use cobre_core::{
            ConstraintExpression, GenericConstraint, LinearTerm, SlackConfig, VariableRef,
        };

        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = parse_case(&dir);

        let gc = GenericConstraint {
            id: EntityId::from(1),
            name: "test_constraint".to_string(),
            description: None,
            expression: ConstraintExpression {
                terms: vec![LinearTerm::literal(
                    1.0,
                    VariableRef::HydroInflow {
                        hydro_id: EntityId::from(99),
                        block_id: Some(0),
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
        data.generic_constraints = vec![gc];

        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);
        assert!(ctx.has_errors(), "expected referential errors");

        let inv: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::InvalidReference)
            .collect();
        assert_eq!(
            inv.len(),
            1,
            "expected exactly 1 InvalidReference, got: {inv:?}"
        );
        assert!(
            inv[0].message.contains("non-existent Hydro 99"),
            "error message must name non-existent Hydro 99, got: {}",
            inv[0].message
        );
    }

    /// A constraint referencing `hydro_storage_initial(99, 0)` where Hydro 99 does
    /// not exist produces exactly one `InvalidReference` naming Hydro 99. The
    /// hydro arm's `..` pattern absorbs `block_id`, matching `HydroStorage`.
    #[test]
    fn test_hydro_storage_initial_unknown_hydro_ref() {
        use cobre_core::{
            ConstraintExpression, GenericConstraint, LinearTerm, SlackConfig, VariableRef,
        };

        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = parse_case(&dir);

        let gc = GenericConstraint {
            id: EntityId::from(1),
            name: "test_constraint".to_string(),
            description: None,
            expression: ConstraintExpression {
                terms: vec![LinearTerm::literal(
                    1.0,
                    VariableRef::HydroStorageInitial {
                        hydro_id: EntityId::from(99),
                        block_id: Some(0),
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
        data.generic_constraints = vec![gc];

        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);
        assert!(ctx.has_errors(), "expected referential errors");

        let inv: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::InvalidReference)
            .collect();
        assert_eq!(
            inv.len(),
            1,
            "expected exactly 1 InvalidReference, got: {inv:?}"
        );
        assert!(
            inv[0].message.contains("non-existent Hydro 99"),
            "error message must name non-existent Hydro 99, got: {}",
            inv[0].message
        );
    }

    /// A constraint referencing `hydro_useful_volume_final(99, 0)` where Hydro 99
    /// does not exist produces exactly one `InvalidReference` naming Hydro 99 —
    /// the same check `hydro_storage_final` gets, via the shared hydro arm.
    #[test]
    fn test_hydro_useful_volume_final_unknown_hydro_ref() {
        use cobre_core::{
            ConstraintExpression, GenericConstraint, LinearTerm, SlackConfig, VariableRef,
        };

        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = parse_case(&dir);

        let gc = GenericConstraint {
            id: EntityId::from(1),
            name: "test_constraint".to_string(),
            description: None,
            expression: ConstraintExpression {
                terms: vec![LinearTerm::literal(
                    1.0,
                    VariableRef::HydroUsefulVolumeFinal {
                        hydro_id: EntityId::from(99),
                        block_id: Some(0),
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
        data.generic_constraints = vec![gc];

        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);
        assert!(ctx.has_errors(), "expected referential errors");

        let inv: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::InvalidReference)
            .collect();
        assert_eq!(
            inv.len(),
            1,
            "expected exactly 1 InvalidReference, got: {inv:?}"
        );
        assert!(
            inv[0].message.contains("non-existent Hydro 99"),
            "error message must name non-existent Hydro 99, got: {}",
            inv[0].message
        );
    }

    // ── `bus_id` selector referential validation ───────────────────────

    /// Hydro 7's two unit groups (ids 20, 21 at positions 0, 1) sit on buses 1
    /// and 4; Hydro 8 has one group (id 30) on bus 9. Buses 1, 4, and 9 are all
    /// declared; bus 777 is declared nowhere. Group ids, positions, and bus ids
    /// are mutually disjoint so a group-id or position lookup cannot pass as a
    /// bus lookup.
    fn make_split_plant_bus_selector_fixture(dir: &TempDir) -> ParsedData {
        let mut data = parse_case(dir);
        data.buses.push(Bus {
            id: EntityId::from(4),
            name: "BUS_4".to_string(),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            deficit_segments: vec![],
            excess_cost: 100.0,
        });
        data.buses.push(Bus {
            id: EntityId::from(9),
            name: "BUS_9".to_string(),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            deficit_segments: vec![],
            excess_cost: 100.0,
        });

        let mut hydro7 = make_hydro(7, None);
        hydro7.unit_groups = vec![
            make_unit_group(20, 1, 0.0, 100.0, 0.0, 100.0),
            make_unit_group(21, 4, 0.0, 100.0, 0.0, 100.0),
        ];

        let mut hydro8 = make_hydro(8, None);
        hydro8.unit_groups = vec![make_unit_group(30, 9, 0.0, 100.0, 0.0, 100.0)];

        data.hydros = vec![hydro7, hydro8];
        data
    }

    /// A `hydro_turbined(7, bus=9)` term names bus 9, which genuinely exists in
    /// the system (Hydro 8 has a group there) but on which Hydro 7 has no unit
    /// group. The per-plant check rejects it even though bus 9 is a declared bus.
    #[test]
    fn test_generic_constraint_unknown_bus_selector_rejected() {
        use cobre_core::{
            ConstraintExpression, GenericConstraint, LinearTerm, SlackConfig, VariableRef,
        };

        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = make_split_plant_bus_selector_fixture(&dir);

        let gc = GenericConstraint {
            id: EntityId::from(1),
            name: "test_constraint".to_string(),
            description: None,
            expression: ConstraintExpression {
                terms: vec![LinearTerm::literal(
                    1.0,
                    VariableRef::HydroTurbined {
                        hydro_id: EntityId::from(7),
                        block_id: None,
                        bus_id: Some(EntityId::from(9)),
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
        data.generic_constraints = vec![gc];

        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);

        let inv: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::InvalidReference)
            .filter(|e| e.file == std::path::Path::new("system/generic_constraints.json"))
            .collect();
        assert_eq!(
            inv.len(),
            1,
            "expected exactly 1 InvalidReference, got: {:?}",
            inv.iter().map(|e| &e.message).collect::<Vec<_>>()
        );
        assert!(
            inv[0].message.contains("bus 9"),
            "message must name bus 9, got: {}",
            inv[0].message
        );
        assert!(
            inv[0].message.contains("Hydro 7"),
            "message must name Hydro 7, got: {}",
            inv[0].message
        );
        assert!(
            inv[0].message.contains("GenericConstraint 1 term[0]"),
            "message must name the constraint's term label, got: {}",
            inv[0].message
        );
    }

    /// The same fixture with the term's `bus_id` changed to `Some(EntityId(4))`
    /// (a real bus of Hydro 7's second group) and a second constraint whose
    /// `HydroGeneration` term carries `bus_id: None` (the plant-wide reference)
    /// produce no finding: `None` stays silent, and resolving membership by
    /// bus rather than group id accepts the valid selector.
    #[test]
    fn test_generic_constraint_valid_bus_selector_and_none_accepted() {
        use cobre_core::{
            ConstraintExpression, GenericConstraint, LinearTerm, SlackConfig, VariableRef,
        };

        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = make_split_plant_bus_selector_fixture(&dir);

        let gc_turbined = GenericConstraint {
            id: EntityId::from(1),
            name: "test_constraint_turbined".to_string(),
            description: None,
            expression: ConstraintExpression {
                terms: vec![LinearTerm::literal(
                    1.0,
                    VariableRef::HydroTurbined {
                        hydro_id: EntityId::from(7),
                        block_id: None,
                        bus_id: Some(EntityId::from(4)),
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
        let gc_generation = GenericConstraint {
            id: EntityId::from(2),
            name: "test_constraint_generation".to_string(),
            description: None,
            expression: ConstraintExpression {
                terms: vec![LinearTerm::literal(
                    1.0,
                    VariableRef::HydroGeneration {
                        hydro_id: EntityId::from(7),
                        block_id: None,
                        bus_id: None,
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
        data.generic_constraints = vec![gc_turbined, gc_generation];

        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);

        let inv: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::InvalidReference)
            .filter(|e| e.file == std::path::Path::new("system/generic_constraints.json"))
            .collect();
        assert!(
            inv.is_empty(),
            "expected no InvalidReference against generic_constraints.json, got: {:?}",
            inv.iter().map(|e| &e.message).collect::<Vec<_>>()
        );
    }

    /// A `hydro_generation(99, bus=1)` term where Hydro 99 does not exist emits
    /// only the `hydro_id` finding — the bus half is unanswerable without a
    /// plant and must not also fire.
    #[test]
    fn test_generic_constraint_unknown_hydro_with_selector_emits_one_finding() {
        use cobre_core::{
            ConstraintExpression, GenericConstraint, LinearTerm, SlackConfig, VariableRef,
        };

        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = parse_case(&dir);

        let gc = GenericConstraint {
            id: EntityId::from(1),
            name: "test_constraint".to_string(),
            description: None,
            expression: ConstraintExpression {
                terms: vec![LinearTerm::literal(
                    1.0,
                    VariableRef::HydroGeneration {
                        hydro_id: EntityId::from(99),
                        block_id: None,
                        bus_id: Some(EntityId::from(1)),
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
        data.generic_constraints = vec![gc];

        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);

        let inv: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::InvalidReference)
            .filter(|e| e.file == std::path::Path::new("system/generic_constraints.json"))
            .collect();
        assert_eq!(
            inv.len(),
            1,
            "expected exactly 1 InvalidReference, got: {:?}",
            inv.iter().map(|e| &e.message).collect::<Vec<_>>()
        );
        assert!(
            inv[0].message.contains("Hydro 99"),
            "message must name Hydro 99, got: {}",
            inv[0].message
        );
        assert!(
            !inv[0].message.contains("bus"),
            "message must not carry a bus finding, got: {}",
            inv[0].message
        );
    }

    /// A bus that exists in no plant's group set and is not a declared bus at
    /// all is rejected with exactly one finding: the per-plant check subsumes
    /// the "bus does not exist" case, so no second finding is added.
    #[test]
    fn test_generic_constraint_nonexistent_bus_selector_emits_one_finding() {
        use cobre_core::{
            ConstraintExpression, GenericConstraint, LinearTerm, SlackConfig, VariableRef,
        };

        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        let mut data = make_split_plant_bus_selector_fixture(&dir);

        let gc = GenericConstraint {
            id: EntityId::from(1),
            name: "test_constraint".to_string(),
            description: None,
            expression: ConstraintExpression {
                terms: vec![LinearTerm::literal(
                    1.0,
                    VariableRef::HydroTurbined {
                        hydro_id: EntityId::from(7),
                        block_id: None,
                        bus_id: Some(EntityId::from(777)),
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
        data.generic_constraints = vec![gc];

        let mut ctx = ValidationContext::new();
        validate_referential_integrity(&data, &mut ctx);

        let inv: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::InvalidReference)
            .filter(|e| e.file == std::path::Path::new("system/generic_constraints.json"))
            .collect();
        assert_eq!(
            inv.len(),
            1,
            "expected exactly 1 InvalidReference, got: {:?}",
            inv.iter().map(|e| &e.message).collect::<Vec<_>>()
        );
        assert!(
            inv[0].message.contains("bus 777"),
            "message must name bus 777, got: {}",
            inv[0].message
        );
        assert!(
            inv[0].message.contains("Hydro 7"),
            "message must name Hydro 7, got: {}",
            inv[0].message
        );
    }
}
