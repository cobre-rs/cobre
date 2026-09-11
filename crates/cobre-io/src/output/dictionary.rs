//! [`write_dictionaries`] writes the self-documenting files to
//! `training/dictionaries/`:
//!
//! - `codes.json` — categorical code mappings (operative state, bound type, etc.)
//! - `entities.csv` — one row per entity with id, name, bus, and system columns
//! - `variables.csv` — one row per column across all output Parquet schemas
//! - `bounds.parquet` — per-entity, per-stage bound values

use std::path::Path;
use std::sync::Arc;

use arrow::array::{Float64Builder, Int8Builder, Int32Builder, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use cobre_core::System;

use crate::Config;
use crate::output::atomic::{write_bytes_atomic, write_parquet_atomic};
use crate::output::error::OutputError;
use crate::output::parquet_config::ParquetWriterConfig;
use crate::output::schemas::{
    buses_schema, contracts_schema, convergence_schema, costs_schema, exchanges_schema,
    generic_violations_schema, hydro_bus_generation_schema, hydro_energy_productivity_schema,
    hydros_schema, in_transit_schema, inflow_lags_schema, iteration_timing_schema,
    non_controllables_schema, paths_schema, pumping_stations_schema, rank_timing_schema,
    retry_histogram_schema, row_selection_schema, scenario_summary_schema,
    solver_iterations_schema, thermals_schema, transit_seed_schema,
};

// ─── Entity type codes (SS3) ─────────────────────────────────────────────────

const OUTPUT_ENTITY_HYDRO: i8 = 0;
const OUTPUT_ENTITY_THERMAL: i8 = 1;
const OUTPUT_ENTITY_BUS: i8 = 2;
const OUTPUT_ENTITY_LINE: i8 = 3;
const OUTPUT_ENTITY_PUMPING_STATION: i8 = 4;
const OUTPUT_ENTITY_CONTRACT: i8 = 5;
const OUTPUT_ENTITY_NON_CONTROLLABLE: i8 = 7;
const OUTPUT_ENTITY_HYDRO_UNIT_GROUP: i8 = 8;
// 9 is reserved for a future thermal unit group; do not reuse it.

// ─── Bound type codes (SS3) ──────────────────────────────────────────────────

const BOUND_STORAGE_MIN: i8 = 0;
const BOUND_STORAGE_MAX: i8 = 1;
const BOUND_TURBINED_MIN: i8 = 2;
const BOUND_TURBINED_MAX: i8 = 3;
const BOUND_OUTFLOW_MIN: i8 = 4;
const BOUND_OUTFLOW_MAX: i8 = 5;
const BOUND_GENERATION_MIN: i8 = 6;
const BOUND_GENERATION_MAX: i8 = 7;
const BOUND_FLOW_MIN: i8 = 8;
const BOUND_FLOW_MAX: i8 = 9;

// ─── Public entry point ───────────────────────────────────────────────────────

/// Write the dictionary files — `codes.json`, `entities.csv`, `variables.csv`,
/// `bounds.parquet` (see module doc) — to `path`.
///
/// `path` must point to an already-created dictionaries directory (typically
/// `output_dir/training/dictionaries/`).
///
/// # Errors
///
/// - [`OutputError::IoError`] for file write failures.
/// - [`OutputError::SerializationError`] for Arrow `RecordBatch` construction
///   failures in `bounds.parquet`.
/// - [`OutputError::ManifestError`] for JSON serialization failures.
pub fn write_dictionaries(
    path: &Path,
    system: &System,
    _config: &Config,
) -> Result<(), OutputError> {
    write_codes_json(path)?;
    write_entities_csv(path, system)?;
    write_variables_csv(path)?;
    write_bounds_parquet(path, system, &ParquetWriterConfig::default())?;
    Ok(())
}

// ─── codes.json ──────────────────────────────────────────────────────────────

/// Write `codes.json`: code-to-label mappings for every categorical integer
/// code used in the Parquet output files.
fn write_codes_json(path: &Path) -> Result<(), OutputError> {
    let generated_at = chrono::Utc::now().to_rfc3339();

    let content = serde_json::json!({
        "version": "1.0",
        "generated_at": generated_at,
        "operative_state": {
            "0": "deactivated",
            "1": "maintenance",
            "2": "operating",
            "3": "saturated"
        },
        "storage_binding": {
            "0": "none",
            "1": "below_minimum",
            "2": "above_maximum",
            "3": "both"
        },
        "contract_type": {
            "0": "import",
            "1": "export"
        },
        "entity_type": {
            "0": "hydro",
            "1": "thermal",
            "2": "bus",
            "3": "line",
            "4": "pumping_station",
            "5": "contract",
            "7": "non_controllable",
            "8": "hydro_unit_group"
        },
        "bound_type": {
            "0": "storage_min",
            "1": "storage_max",
            "2": "turbined_min",
            "3": "turbined_max",
            "4": "outflow_min",
            "5": "outflow_max",
            "6": "generation_min",
            "7": "generation_max",
            "8": "flow_min",
            "9": "flow_max"
        }
    });

    let json_str =
        serde_json::to_string_pretty(&content).map_err(|e| OutputError::ManifestError {
            manifest_type: "codes.json".to_string(),
            message: e.to_string(),
        })?;

    write_bytes_atomic(path.join("codes.json").as_path(), json_str.as_bytes())
}

// ─── entities.csv ────────────────────────────────────────────────────────────

fn finish_csv_atomic(mut wtr: csv::Writer<Vec<u8>>, file_path: &Path) -> Result<(), OutputError> {
    wtr.flush().map_err(|e| OutputError::io(file_path, e))?;
    let bytes = wtr
        .into_inner()
        .map_err(|e| OutputError::io(file_path, std::io::Error::other(e)))?;
    write_bytes_atomic(file_path, &bytes)
}

/// Write `entities.csv`, one row per entity, ordered by `entity_type_code`
/// ascending then by entity ID (canonical accessor order). The
/// `entity_type_code` 8 block breaks that second axis: a group's `entity_id`
/// is plant-scoped, so it orders plant-major (canonical `System::hydros()`
/// order) then group-minor (each plant's own id-sorted `unit_groups` order).
fn write_entities_csv(path: &Path, system: &System) -> Result<(), OutputError> {
    let file_path = path.join("entities.csv");
    let mut wtr = csv::Writer::from_writer(Vec::new());

    wtr.write_record([
        "entity_type_code",
        "entity_id",
        "name",
        "bus_id",
        "system_id",
    ])
    .map_err(|e| OutputError::io(&file_path, std::io::Error::other(e)))?;

    let mut write_row = |entity_type: i8, id: i32, name: &str, bus_id: i32| {
        wtr.write_record([
            entity_type.to_string(),
            id.to_string(),
            name.to_string(),
            bus_id.to_string(),
            "0".to_string(),
        ])
        .map_err(|e| OutputError::io(&file_path, std::io::Error::other(e)))
    };

    for h in system.hydros() {
        // The plant's hydro_unit_group rows own the bus association; a split
        // plant has no single owning bus.
        write_row(OUTPUT_ENTITY_HYDRO, h.id.0, &h.name, -1)?;
    }

    for t in system.thermals() {
        write_row(OUTPUT_ENTITY_THERMAL, t.id.0, &t.name, t.bus_id.0)?;
    }

    for b in system.buses() {
        write_row(OUTPUT_ENTITY_BUS, b.id.0, &b.name, b.id.0)?;
    }

    for l in system.lines() {
        // A line connects two buses, so it has no single owning bus.
        write_row(OUTPUT_ENTITY_LINE, l.id.0, &l.name, -1)?;
    }

    for p in system.pumping_stations() {
        write_row(OUTPUT_ENTITY_PUMPING_STATION, p.id.0, &p.name, p.bus_id.0)?;
    }

    for c in system.contracts() {
        write_row(OUTPUT_ENTITY_CONTRACT, c.id.0, &c.name, c.bus_id.0)?;
    }

    for n in system.non_controllable_sources() {
        write_row(OUTPUT_ENTITY_NON_CONTROLLABLE, n.id.0, &n.name, n.bus_id.0)?;
    }

    for h in system.hydros() {
        for g in &h.unit_groups {
            write_row(
                OUTPUT_ENTITY_HYDRO_UNIT_GROUP,
                g.id.0,
                &format!("{}/{}", h.id.0, g.name),
                g.bus_id.0,
            )?;
        }
    }

    finish_csv_atomic(wtr, &file_path)
}

// ─── variables.csv ───────────────────────────────────────────────────────────

/// Every `(file, schema)` pair `variables.csv` documents — the single owner of
/// the list, shared with the no-empty-description test that guards it.
fn variables_csv_schemas() -> Vec<(&'static str, Schema)> {
    vec![
        ("costs", costs_schema()),
        ("hydros", hydros_schema()),
        ("hydro_bus_generation", hydro_bus_generation_schema()),
        ("thermals", thermals_schema()),
        ("exchanges", exchanges_schema()),
        ("buses", buses_schema()),
        ("pumping_stations", pumping_stations_schema()),
        ("contracts", contracts_schema()),
        ("non_controllables", non_controllables_schema()),
        ("inflow_lags", inflow_lags_schema()),
        ("in_transit", in_transit_schema()),
        ("transit_seed", transit_seed_schema()),
        ("generic_violations", generic_violations_schema()),
        ("paths", paths_schema()),
        ("scenario_summary", scenario_summary_schema()),
        ("convergence", convergence_schema()),
        ("iteration_timing", iteration_timing_schema()),
        ("rank_timing", rank_timing_schema()),
        ("cut_selection", row_selection_schema()),
        ("solver_iterations", solver_iterations_schema()),
        ("retry_histogram", retry_histogram_schema()),
        (
            "hydro_energy_productivity",
            hydro_energy_productivity_schema(),
        ),
    ]
}

/// Write `variables.csv`, one row per column across every output schema, grouped
/// by file and ordered by column position within each schema.
fn write_variables_csv(path: &Path) -> Result<(), OutputError> {
    let file_path = path.join("variables.csv");
    let mut wtr = csv::Writer::from_writer(Vec::new());

    wtr.write_record(["file", "column", "type", "unit", "description", "nullable"])
        .map_err(|e| OutputError::io(&file_path, std::io::Error::other(e)))?;

    for (schema_name, schema) in &variables_csv_schemas() {
        for field in schema.fields() {
            let type_str = arrow_type_str(field.data_type());
            let unit = unit_for(schema_name, field.name());
            let description = description_for(schema_name, field.name());
            let nullable = if field.is_nullable() { "true" } else { "false" };

            wtr.write_record([
                *schema_name,
                field.name().as_str(),
                type_str,
                unit,
                description,
                nullable,
            ])
            .map_err(|e| OutputError::io(&file_path, std::io::Error::other(e)))?;
        }
    }

    finish_csv_atomic(wtr, &file_path)
}

/// Map an Arrow `DataType` to the string representation used in `variables.csv`.
fn arrow_type_str(dt: &DataType) -> &'static str {
    match dt {
        DataType::Int8 => "i8",
        DataType::Int32 => "i32",
        DataType::Int64 => "i64",
        DataType::UInt32 => "u32",
        DataType::UInt64 => "u64",
        DataType::Float64 => "f64",
        DataType::Boolean => "bool",
        DataType::Utf8 => "string",
        DataType::Date32 => "date32",
        _ => "unknown",
    }
}

/// Return the physical unit string for a given (file, column) pair.
///
/// Returns `""` for dimensionless columns or columns without a defined unit.
// Rationale: one authoritative (file, column) → unit lookup table; identical
// arms are intentional (same unit recurs across schemas), so splitting or
// collapsing arms would degrade it as a catalog.
#[allow(clippy::too_many_lines, clippy::match_same_arms)]
fn unit_for(file: &str, column: &str) -> &'static str {
    match column {
        "scenario_id"
        | "stage_id"
        | "node_id"
        | "block_id"
        | "lag"
        | "iteration"
        | "rank"
        | "forward_passes"
        | "scenarios_processed" => return "",
        "generation_mw"
        | "available_mw"
        | "curtailment_mw"
        | "direct_flow_mw"
        | "reverse_flow_mw"
        | "net_flow_mw"
        | "losses_mw"
        | "load_mw"
        | "deficit_mw"
        | "excess_mw"
        | "spot_price"
        | "pumped_flow_m3s"
        | "power_consumption_mw"
        | "anticipated_committed_mw"
        | "anticipated_decision_mw"
        | "power_mw" => return "MW",
        "generation_mwh"
        | "curtailment_mwh"
        | "net_flow_mwh"
        | "losses_mwh"
        | "load_mwh"
        | "deficit_mwh"
        | "excess_mwh"
        | "energy_consumption_mwh"
        | "energy_mwh" => return "MWh",
        "turbined_m3s"
        | "spillage_m3s"
        | "outflow_m3s"
        | "evaporation_m3s"
        | "diverted_inflow_m3s"
        | "diverted_outflow_m3s"
        | "incremental_inflow_m3s"
        | "inflow_m3s"
        | "turbined_slack_m3s"
        | "outflow_slack_below_m3s"
        | "outflow_slack_above_m3s"
        | "evaporation_violation_pos_m3s"
        | "evaporation_violation_neg_m3s"
        | "inflow_nonnegativity_slack_m3s"
        | "water_withdrawal_violation_pos_m3s"
        | "water_withdrawal_violation_neg_m3s"
        | "pumped_volume_hm3"
        | "value_m3s" => return "m3/s",
        "storage_initial_hm3"
        | "storage_final_hm3"
        | "storage_violation_below_hm3"
        | "filling_target_violation_hm3"
        | "in_transit_volume_hm3"
        | "delayed_arrival_hm3" => return "hm3",
        "total_cost"
        | "immediate_cost"
        | "discounted_immediate_cost"
        | "future_cost"
        | "thermal_cost"
        | "anticipated_thermal_cost"
        | "contract_cost"
        | "deficit_cost"
        | "excess_cost"
        | "storage_violation_cost"
        | "filling_target_cost"
        | "hydro_violation_cost"
        | "outflow_violation_below_cost"
        | "outflow_violation_above_cost"
        | "turbined_violation_cost"
        | "generation_violation_cost"
        | "evaporation_violation_cost"
        | "withdrawal_violation_cost"
        | "inflow_penalty_cost"
        | "generic_violation_cost"
        | "spillage_cost"
        | "turbined_cost"
        | "curtailment_cost"
        | "exchange_cost"
        | "pumping_cost"
        | "generation_cost"
        | "total_cost_convergence"
        | "pumping_cost_csv"
        | "price_per_mwh"
        | "slack_cost" => return "$",
        "time_forward_ms"
        | "time_backward_ms"
        | "time_total_ms"
        | "forward_solve_ms"
        | "forward_sample_ms"
        | "backward_solve_ms"
        | "backward_cut_ms"
        | "cut_selection_ms"
        | "mpi_allreduce_ms"
        | "mpi_broadcast_ms"
        | "io_write_ms"
        | "state_exchange_ms"
        | "cut_batch_build_ms"
        | "bwd_setup_ms"
        | "bwd_load_imbalance_ms"
        | "bwd_scheduling_overhead_ms"
        | "fwd_setup_ms"
        | "fwd_load_imbalance_ms"
        | "fwd_scheduling_overhead_ms"
        | "overhead_ms"
        | "lazy_scoring_ms"
        | "forward_time_ms"
        | "backward_time_ms"
        | "communication_time_ms"
        | "idle_time_ms"
        | "solve_time_ms"
        | "load_model_time_ms"
        | "set_bounds_time_ms"
        | "basis_set_time_ms"
        | "selection_time_ms" => return "ms",
        _ => {}
    }
    match (file, column) {
        ("hydros", "water_value_per_hm3") => "$/hm3",
        ("hydros", "equivalent_productivity_mw_per_m3s") => "MW/(m3/s)",
        ("hydros", "accumulated_productivity_mw_per_m3s") => "MW/(m3/s)",
        ("hydros", "incremental_inflow_energy_mw") => "MW",
        ("hydros", "stored_energy_initial_mwh") => "MWh",
        ("hydros", "stored_energy_final_mwh") => "MWh",
        ("hydros", "generation_slack_mw") => "MW",
        _ => "",
    }
}

/// Return a short description for a given (file, column) pair.
///
/// Returns `""` for columns without a registered description.
// Rationale: one authoritative (file, column) → description lookup table;
// identical arms are intentional, so collapsing them would hide additions and
// per-schema divergence.
#[allow(clippy::too_many_lines, clippy::match_same_arms)]
fn description_for(file: &str, column: &str) -> &'static str {
    // scenario_id/node_id return here regardless of file; a (file, column) arm
    // for either name below is unreachable.
    match column {
        "scenario_id" => return "0-based scenario identifier",
        "node_id" => return "Declared node id visited at this stage",
        _ => {}
    }
    match (file, column) {
        ("paths", "stage_id") => "Stage index",
        ("costs", "stage_id") => "Stage index",
        ("costs", "block_id") => "Block index within stage (nullable)",
        ("costs", "total_cost") => "Total stage cost",
        ("costs", "immediate_cost") => "Immediate (operation) cost",
        ("costs", "future_cost") => "Expected future cost (envelope value)",
        ("costs", "discount_factor") => "Discount factor applied to this stage",
        ("costs", "thermal_cost") => "Total thermal generation cost",
        ("costs", "anticipated_thermal_cost") => {
            "Total anticipated (forward-committed) thermal generation cost"
        }
        ("costs", "contract_cost") => "Total contract cost",
        ("costs", "deficit_cost") => "Total load-deficit penalty cost",
        ("costs", "excess_cost") => "Total excess-generation cost",
        ("costs", "storage_violation_cost") => "Total storage violation penalty",
        ("costs", "filling_target_cost") => "Total filling-target violation cost",
        ("costs", "hydro_violation_cost") => "Total hydro constraint violation cost",
        ("costs", "outflow_violation_below_cost") => "Cost of minimum outflow violations",
        ("costs", "outflow_violation_above_cost") => "Cost of maximum outflow violations",
        ("costs", "turbined_violation_cost") => "Cost of minimum turbining violations",
        ("costs", "generation_violation_cost") => "Cost of minimum generation violations",
        ("costs", "evaporation_violation_cost") => "Cost of evaporation constraint violations",
        ("costs", "withdrawal_violation_cost") => "Cost of water withdrawal constraint violations",
        ("costs", "inflow_penalty_cost") => "Total inflow non-negativity penalty",
        ("costs", "generic_violation_cost") => "Total generic constraint violation cost",
        ("costs", "spillage_cost") => "Total spillage regularization cost",
        ("costs", "turbined_cost") => "Total turbined regularization cost",
        ("costs", "curtailment_cost") => "Total curtailment cost",
        ("costs", "exchange_cost") => "Total exchange regularization cost",
        ("costs", "pumping_cost") => "Total pumping cost",
        ("hydros", "stage_id") => "Stage index",
        ("hydros", "block_id") => "Block index within stage (nullable)",
        ("hydros", "hydro_id") => "Hydro plant identifier",
        ("hydros", "turbined_m3s") => "Turbined flow",
        ("hydros", "spillage_m3s") => "Spilled flow",
        ("hydros", "outflow_m3s") => "Total outflow (turbined + spilled)",
        ("hydros", "evaporation_m3s") => "Evaporation loss (nullable)",
        ("hydros", "diverted_inflow_m3s") => "Diverted inflow received (nullable)",
        ("hydros", "diverted_outflow_m3s") => "Diverted outflow sent (nullable)",
        ("hydros", "incremental_inflow_m3s") => "Incremental (local) inflow",
        ("hydros", "inflow_m3s") => "Total inflow including upstream contributions",
        ("hydros", "storage_initial_hm3") => "Reservoir storage at start of stage",
        ("hydros", "storage_final_hm3") => "Reservoir storage at end of stage",
        ("hydros", "generation_mw") => "Hydro generation",
        ("hydros", "generation_mwh") => "Hydro energy generated",
        ("hydros", "equivalent_productivity_mw_per_m3s") => {
            "Equivalent productivity `ρ_eq` (always populated)"
        }
        ("hydros", "accumulated_productivity_mw_per_m3s") => {
            "Accumulated productivity `ρ_acum` along downstream cascade"
        }
        ("hydros", "incremental_inflow_energy_mw") => {
            "Incremental natural energy inflow (`ρ_acum` · incremental inflow)"
        }
        ("hydros", "stored_energy_initial_mwh") => "Stored energy at start of block",
        ("hydros", "stored_energy_final_mwh") => "Stored energy at end of block",
        ("hydros", "spillage_cost") => "Spillage regularization cost",
        ("hydros", "water_value_per_hm3") => "Marginal water value",
        ("hydros", "storage_binding_code") => "Storage bound binding code",
        ("hydros", "operative_state_code") => "Operative state code",
        ("hydros", "turbined_slack_m3s") => "Turbined minimum slack",
        ("hydros", "outflow_slack_below_m3s") => "Outflow below-minimum slack",
        ("hydros", "outflow_slack_above_m3s") => "Outflow above-maximum slack",
        ("hydros", "generation_slack_mw") => "Generation minimum slack",
        ("hydros", "storage_violation_below_hm3") => "Storage below dead-volume violation",
        ("hydros", "filling_target_violation_hm3") => "Filling target violation",
        ("hydros", "evaporation_violation_pos_m3s") => "Over-evaporation constraint violation",
        ("hydros", "evaporation_violation_neg_m3s") => "Under-evaporation constraint violation",
        ("hydros", "inflow_nonnegativity_slack_m3s") => "Inflow non-negativity slack",
        ("hydros", "water_withdrawal_violation_pos_m3s") => "Over-withdrawal constraint violation",
        ("hydros", "water_withdrawal_violation_neg_m3s") => "Under-withdrawal constraint violation",
        ("hydro_bus_generation", "stage_id") => "Stage index",
        ("hydro_bus_generation", "block_id") => "Block index within stage (nullable)",
        ("hydro_bus_generation", "hydro_id") => "Hydro plant identifier",
        ("hydro_bus_generation", "bus_id") => "Bus identifier",
        ("hydro_bus_generation", "turbined_m3s") => "Turbined flow",
        ("hydro_bus_generation", "generation_mw") => "Hydro generation",
        ("hydro_bus_generation", "generation_mwh") => "Hydro energy generated",
        ("thermals", "stage_id") => "Stage index",
        ("thermals", "block_id") => "Block index within stage (nullable)",
        ("thermals", "thermal_id") => "Thermal plant identifier",
        ("thermals", "generation_mw") => "Thermal generation",
        ("thermals", "generation_mwh") => "Thermal energy generated",
        ("thermals", "generation_cost") => "Thermal generation cost",
        ("thermals", "is_anticipated") => "Whether plant uses anticipated dispatch",
        ("thermals", "anticipated_committed_mw") => "Anticipated committed capacity (nullable)",
        ("thermals", "anticipated_decision_mw") => "Anticipated dispatch decision (nullable)",
        ("thermals", "operative_state_code") => "Operative state code",
        ("exchanges", "stage_id") => "Stage index",
        ("exchanges", "block_id") => "Block index within stage (nullable)",
        ("exchanges", "line_id") => "Transmission line identifier",
        ("exchanges", "direct_flow_mw") => "Flow in direct direction",
        ("exchanges", "reverse_flow_mw") => "Flow in reverse direction",
        ("exchanges", "net_flow_mw") => "Net flow (direct minus reverse)",
        ("exchanges", "net_flow_mwh") => "Net energy exchanged",
        ("exchanges", "losses_mw") => "Transmission losses",
        ("exchanges", "losses_mwh") => "Transmission energy losses",
        ("exchanges", "exchange_cost") => "Exchange regularization cost",
        ("exchanges", "operative_state_code") => "Operative state code",
        ("buses", "stage_id") => "Stage index",
        ("buses", "block_id") => "Block index within stage (nullable)",
        ("buses", "bus_id") => "Bus identifier",
        ("buses", "load_mw") => "Load demand",
        ("buses", "load_mwh") => "Load energy demand",
        ("buses", "deficit_mw") => "Unmet demand (deficit)",
        ("buses", "deficit_mwh") => "Unmet energy demand",
        ("buses", "excess_mw") => "Excess generation absorbed",
        ("buses", "excess_mwh") => "Excess energy absorbed",
        ("buses", "spot_price") => "Bus spot price (dual of balance constraint)",
        ("pumping_stations", "stage_id") => "Stage index",
        ("pumping_stations", "block_id") => "Block index within stage (nullable)",
        ("pumping_stations", "pumping_station_id") => "Pumping station identifier",
        ("pumping_stations", "pumped_flow_m3s") => "Pumped water flow",
        ("pumping_stations", "pumped_volume_hm3") => "Pumped water volume",
        ("pumping_stations", "power_consumption_mw") => "Electrical power consumed",
        ("pumping_stations", "energy_consumption_mwh") => "Electrical energy consumed",
        ("pumping_stations", "pumping_cost") => "Pumping operation cost",
        ("pumping_stations", "operative_state_code") => "Operative state code",
        ("contracts", "stage_id") => "Stage index",
        ("contracts", "block_id") => "Block index within stage (nullable)",
        ("contracts", "contract_id") => "Contract identifier",
        ("contracts", "power_mw") => "Contracted power",
        ("contracts", "energy_mwh") => "Contracted energy",
        ("contracts", "price_per_mwh") => "Effective contract price",
        ("contracts", "total_cost") => "Total contract cost",
        ("contracts", "operative_state_code") => "Operative state code",
        ("non_controllables", "stage_id") => "Stage index",
        ("non_controllables", "block_id") => "Block index within stage (nullable)",
        ("non_controllables", "non_controllable_id") => "Non-controllable source identifier",
        ("non_controllables", "generation_mw") => "Non-controllable generation dispatched",
        ("non_controllables", "generation_mwh") => "Non-controllable energy generated",
        ("non_controllables", "available_mw") => "Available generation capacity",
        ("non_controllables", "curtailment_mw") => "Curtailed generation",
        ("non_controllables", "curtailment_mwh") => "Curtailed energy",
        ("non_controllables", "curtailment_cost") => "Curtailment cost",
        ("non_controllables", "operative_state_code") => "Operative state code",
        ("inflow_lags", "stage_id") => "Stage index",
        ("inflow_lags", "hydro_id") => "Hydro plant identifier",
        ("inflow_lags", "lag_index") => "AR lag index (1-based)",
        ("inflow_lags", "inflow_m3s") => "Historical inflow for this lag",
        ("in_transit", "stage_id") => "Stage index",
        ("in_transit", "hydro_id") => "Downstream hydro plant identifier",
        ("in_transit", "lag") => "Maturity bucket index (1-based)",
        ("in_transit", "in_transit_volume_hm3") => {
            "Outgoing in-transit water volume at this maturity"
        }
        ("in_transit", "delayed_arrival_hm3") => {
            "Water delivered this stage (non-zero only at lag 1)"
        }
        ("transit_seed", "hydro_id") => {
            "Upstream entity identifier whose release the window covers"
        }
        ("transit_seed", "start_date") => "Start of the release window (inclusive)",
        ("transit_seed", "end_date") => "End of the release window (exclusive)",
        ("transit_seed", "value_m3s") => "Mean release rate over the window",
        ("generic_violations", "stage_id") => "Stage index",
        ("generic_violations", "block_id") => "Block index within stage (nullable)",
        ("generic_violations", "constraint_id") => "Generic constraint identifier",
        ("generic_violations", "slack_value") => "Constraint slack value",
        ("generic_violations", "slack_cost") => "Constraint slack penalty cost",
        ("scenario_summary", "probability") => {
            "Per-scenario leaf-path probability under a declared census; NULL under sampled selection"
        }
        ("scenario_summary", "discounted_immediate_cost") => {
            "Per-scenario discounted immediate cost; excludes the future-cost term that \
             costs.parquet total_cost includes"
        }
        ("convergence", "iteration") => "Iteration number (1-based)",
        ("convergence", "lower_bound") => "Lower bound on the optimal value",
        ("convergence", "upper_bound") => {
            "Upper bound estimate (sample mean under a sampled forward, exact \
             probability-weighted bound under an enumerated forward)"
        }
        ("convergence", "upper_bound_std") => {
            "Std deviation of upper bound (NULL under an exact bound)"
        }
        ("convergence", "upper_bound_kind") => {
            "Upper bound regime: statistical (sampled forward) or exact (enumerated forward)"
        }
        ("convergence", "gap_percent") => "Relative optimality gap in percent (nullable)",
        ("convergence", "cuts_added") => "Cuts added in this iteration",
        ("convergence", "cuts_removed") => "Cuts removed in this iteration",
        ("convergence", "cuts_active") => "Total active cuts after iteration",
        ("convergence", "time_forward_ms") => "Forward-pass wall-clock time",
        ("convergence", "time_backward_ms") => "Backward-pass wall-clock time",
        ("convergence", "time_total_ms") => "Total iteration wall-clock time",
        ("convergence", "forward_passes") => "Number of forward-pass scenarios",
        ("convergence", "lp_solves") => "Total LP solves in iteration",
        ("convergence", "mean_rows_in_lp") => {
            "Mean resident rows loaded per lazy-selection LP solve this iteration \
             (0 when no lazy selection ran)"
        }
        ("iteration_timing", "iteration") => "Iteration number (1-based)",
        ("iteration_timing", "rank") => {
            "MPI rank that produced this row. Always set; \
             single-rank runs use 0."
        }
        ("iteration_timing", "worker_id") => {
            "Worker thread index within the rank's pool. NULL on \
             rank-aggregated rows that carry rank-only timings (cut_selection, \
             mpi_allreduce, cut_sync, lower_bound, state_exchange, cut_batch_build, \
             load_imbalance / scheduling_overhead, overhead). Set on per-worker \
             rows that carry parallel-region timings (forward_wall, backward_wall, \
             fwd_setup, bwd_setup, lazy_scoring)."
        }
        ("iteration_timing", "forward_wall_ms") => "Forward pass wall-clock time",
        ("iteration_timing", "backward_wall_ms") => "Backward pass wall-clock time",
        ("iteration_timing", "cut_selection_ms") => "Row-selection time",
        ("iteration_timing", "mpi_allreduce_ms") => "MPI allreduce time",
        ("iteration_timing", "cut_sync_ms") => "Per-stage row-sync allgatherv time",
        ("iteration_timing", "lower_bound_ms") => "Lower bound evaluation time",
        ("iteration_timing", "state_exchange_ms") => "State exchange allgatherv time",
        ("iteration_timing", "cut_batch_build_ms") => "Row-batch assembly time",
        ("iteration_timing", "bwd_setup_ms") => "Thread-pool setup time before backward pass",
        ("iteration_timing", "bwd_load_imbalance_ms") => {
            "Estimated load imbalance across backward pass worker threads"
        }
        ("iteration_timing", "bwd_scheduling_overhead_ms") => {
            "Scheduling and synchronisation overhead in the backward pass"
        }
        ("iteration_timing", "fwd_setup_ms") => "Thread-pool setup time before forward pass",
        ("iteration_timing", "fwd_load_imbalance_ms") => {
            "Estimated load imbalance across forward pass worker threads"
        }
        ("iteration_timing", "fwd_scheduling_overhead_ms") => {
            "Scheduling and synchronisation overhead in the forward pass"
        }
        ("iteration_timing", "overhead_ms") => {
            "Residual iteration time not attributed to any phase"
        }
        ("iteration_timing", "lazy_scoring_ms") => {
            "Per-worker time spent in lazy candidate scoring inside the \
             lazy-selection solve; 0 when that solve path is not used. A \
             sub-component of the forward/backward phases."
        }
        ("rank_timing", "iteration") => "Iteration number (1-based)",
        ("rank_timing", "rank") => "MPI rank",
        ("rank_timing", "forward_time_ms") => "Forward-pass time for this rank",
        ("rank_timing", "backward_time_ms") => "Backward-pass time for this rank",
        ("rank_timing", "communication_time_ms") => "Communication time for this rank",
        ("rank_timing", "idle_time_ms") => "Idle time for this rank",
        ("rank_timing", "lp_solves") => "LP solves on this rank",
        ("rank_timing", "scenarios_processed") => "Scenarios processed by this rank",
        ("cut_selection", "iteration") => "Iteration number (1-based)",
        ("cut_selection", "stage_id") => "Declared study stage id",
        ("cut_selection", "cuts_populated") => "Total cuts ever generated at this stage",
        ("cut_selection", "cuts_active_before") => "Active cuts before selection ran",
        ("cut_selection", "cuts_deactivated") => "Cuts deactivated by selection",
        ("cut_selection", "cuts_reactivated") => {
            "Cuts reactivated (previously deactivated, re-entered LP)"
        }
        ("cut_selection", "cuts_active_after") => "Active cuts after selection",
        ("cut_selection", "selection_time_ms") => "Wall-clock time for selection at this stage",
        ("cut_selection", "budget_evicted") => {
            "Cuts evicted by budget enforcement (null when budget disabled)"
        }
        ("cut_selection", "active_after_budget") => {
            "Active cuts after budget enforcement (null when budget disabled)"
        }
        ("solver_iterations", "iteration") => {
            "Training iteration number (1-based); NULL on a simulation row"
        }
        ("solver_iterations", "scenario_id") => {
            "Simulation trajectory id (0-based); NULL on a training row"
        }
        ("solver_iterations", "phase") => {
            "Solver phase (forward, backward, lower_bound, simulation)"
        }
        ("solver_iterations", "stage_id") => {
            "Declared study stage id (NULL for the lower_bound and simulation phases)"
        }
        ("solver_iterations", "lp_solves") => "Number of LP solves",
        ("solver_iterations", "lp_successes") => "Solves that returned optimal",
        ("solver_iterations", "lp_retries") => "Solves requiring retry escalation",
        ("solver_iterations", "lp_failures") => "Solves that exhausted all retry levels",
        ("solver_iterations", "retry_attempts") => "Total retry attempts across all solves",
        ("solver_iterations", "basis_offered") => {
            "Number of warm-start solve calls (basis-offered)"
        }
        ("solver_iterations", "basis_consistency_failures") => {
            "Number of warm-start solve calls rejected because isBasisConsistent returned false"
        }
        ("solver_iterations", "simplex_iterations") => "Total simplex iterations",
        ("solver_iterations", "solve_time_ms") => "Cumulative solve wall-clock time",
        ("solver_iterations", "load_model_time_ms") => "Cumulative load_model call time",
        ("solver_iterations", "set_bounds_time_ms") => "Cumulative set_bounds call time",
        ("solver_iterations", "basis_set_time_ms") => "Cumulative set_basis call time",
        ("solver_iterations", "opening_index") => {
            "Opening (noise realization) index within the stage, for backward-pass \
             rows. NULL for forward, lower_bound, and simulation rows — these phases \
             do not have an opening dimension. Backward rows range 0..n_openings."
        }
        ("solver_iterations", "rank") => {
            "MPI rank that produced this row. NULL for rank-aggregated rows."
        }
        ("solver_iterations", "worker_id") => {
            "Worker thread index within the rank's pool that produced this row. \
             NULL for rank-aggregated rows."
        }
        ("retry_histogram", "iteration") => "Iteration number (1-based) or scenario ID (0-based)",
        ("retry_histogram", "phase") => "Solver phase (forward, backward, lower_bound, simulation)",
        ("retry_histogram", "stage_id") => {
            "Declared study stage id (NULL for the forward, lower_bound, and simulation phases)"
        }
        ("retry_histogram", "retry_level") => "Retry escalation level (0-based)",
        ("retry_histogram", "count") => "Number of solves recovered at this level",
        ("bounds", "hydro_id") => {
            "Owning plant id for a hydro-unit-group row (entity_type_code 8). \
             NULL for the five plant-level entity families."
        }
        ("hydro_energy_productivity", "hydro_id") => "Hydro plant identifier",
        ("hydro_energy_productivity", "stage_id") => {
            "Stage index; null for a per-hydro default across all stages"
        }
        ("hydro_energy_productivity", "equivalent_productivity_mw_per_m3s") => {
            "Equivalent productivity override"
        }
        ("hydro_energy_productivity", "reference_volume_hm3") => {
            "Reference storage volume for the productivity override"
        }
        ("hydro_energy_productivity", "reference_outflow_m3s") => {
            "Reference outflow for the productivity override"
        }
        ("hydro_energy_productivity", "specific_productivity_mw_per_m3s_per_m") => {
            "Specific productivity (per metre of head) override"
        }
        _ => "",
    }
}

// ─── bounds.parquet ──────────────────────────────────────────────────────────

/// Write `bounds.parquet` with per-entity, per-stage resolved bounds: one
/// null-`block_id` row per (entity, stage, `bound_type`), plus one non-null-
/// `block_id` row for each (entity, stage, block, `bound_type`) a per-block
/// override resolved. Only the ten coded `bound_type`s (`write_codes_json`) are
/// reported at either row class; `LineBlockOverride::reverse_mw`,
/// `ContractBlockOverride::price_per_mwh`, and
/// `HydroBlockOverride::max_diversion_m3s` resolve into the LP but have no
/// `bound_type` code and are not emitted here.
///
/// `hydro_id` is populated only for a hydro-unit-group row
/// (`entity_type_code == 8`, the owning plant's id) and `NULL` for the five
/// plant-level families, since a group's `entity_id` is plant-scoped (two
/// different plants may declare a group with the same id). A group's
/// resolved turbined/generation bound follows the same block-precedence law
/// as every other bound-override family: the group's own `stage_override`
/// falling back to its declared value for the null-`block_id` row, then its
/// raw `block_override` (no fallback) for each block a group-axis override
/// resolved — the same two-row-class shape the five plant families already
/// use, so the dictionary reports what the resolved bound actually is rather
/// than a value no code path ever resolves.
// Rationale: all six entity classes share the pre-allocated Arrow column
// builders; per-entity helpers would force passing seven builders through each
// or rebuilding per class, losing the single pre-estimated capacity allocation.
#[allow(
    clippy::too_many_lines,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap
)]
fn write_bounds_parquet(
    path: &Path,
    system: &System,
    config: &ParquetWriterConfig,
) -> Result<(), OutputError> {
    let schema = Arc::new(bounds_schema());
    let n_stages = system.bounds().n_stages();

    let n_groups: usize = system.hydros().iter().map(|h| h.unit_groups.len()).sum();

    // Over-estimate: the optional max-outflow bound may be skipped per hydro.
    let capacity = (system.n_hydros() * 8
        + system.n_thermals() * 2
        + system.n_lines() * 2
        + system.n_pumping_stations() * 2
        + system.n_contracts() * 2
        + n_groups * 4)
        * n_stages;

    let mut entity_type_codes = Int8Builder::with_capacity(capacity);
    let mut entity_ids = Int32Builder::with_capacity(capacity);
    let mut hydro_ids = Int32Builder::with_capacity(capacity);
    let mut stage_ids = Int32Builder::with_capacity(capacity);
    let mut block_ids = Int32Builder::with_capacity(capacity);
    let mut bound_type_codes = Int8Builder::with_capacity(capacity);
    let mut bound_values = Float64Builder::with_capacity(capacity);

    macro_rules! append_bound {
        ($entity_type:expr, $entity_id:expr, $stage_id:expr, $bound_type:expr, $value:expr) => {
            entity_type_codes.append_value($entity_type);
            entity_ids.append_value($entity_id);
            hydro_ids.append_null();
            stage_ids.append_value($stage_id);
            block_ids.append_null();
            bound_type_codes.append_value($bound_type);
            bound_values.append_value($value);
        };
    }

    macro_rules! append_block_bound {
        ($entity_type:expr, $entity_id:expr, $stage_id:expr, $block_id:expr, $bound_type:expr, $value:expr) => {
            entity_type_codes.append_value($entity_type);
            entity_ids.append_value($entity_id);
            hydro_ids.append_null();
            stage_ids.append_value($stage_id);
            block_ids.append_value($block_id);
            bound_type_codes.append_value($bound_type);
            bound_values.append_value($value);
        };
    }

    macro_rules! append_group_bound {
        ($entity_id:expr, $hydro_id:expr, $stage_id:expr, $bound_type:expr, $value:expr) => {
            entity_type_codes.append_value(OUTPUT_ENTITY_HYDRO_UNIT_GROUP);
            entity_ids.append_value($entity_id);
            hydro_ids.append_value($hydro_id);
            stage_ids.append_value($stage_id);
            block_ids.append_null();
            bound_type_codes.append_value($bound_type);
            bound_values.append_value($value);
        };
    }

    macro_rules! append_group_block_bound {
        ($entity_id:expr, $hydro_id:expr, $stage_id:expr, $block_id:expr, $bound_type:expr, $value:expr) => {
            entity_type_codes.append_value(OUTPUT_ENTITY_HYDRO_UNIT_GROUP);
            entity_ids.append_value($entity_id);
            hydro_ids.append_value($hydro_id);
            stage_ids.append_value($stage_id);
            block_ids.append_value($block_id);
            bound_type_codes.append_value($bound_type);
            bound_values.append_value($value);
        };
    }

    let overlay = system.bounds().block_overlay();
    let has_overlay = !overlay.is_empty();

    for (hydro_idx, hydro) in system.hydros().iter().enumerate() {
        let entity_id = hydro.id.0;
        for stage_idx in 0..n_stages {
            let stage_id = system.stages()[stage_idx].id;
            let b = system.bounds().hydro_bounds(hydro_idx, stage_idx);
            let bb = system.bounds().hydro_block_base(hydro_idx, stage_idx);

            append_bound!(
                OUTPUT_ENTITY_HYDRO,
                entity_id,
                stage_id,
                BOUND_STORAGE_MIN,
                b.min_storage_hm3
            );
            append_bound!(
                OUTPUT_ENTITY_HYDRO,
                entity_id,
                stage_id,
                BOUND_STORAGE_MAX,
                b.max_storage_hm3
            );
            append_bound!(
                OUTPUT_ENTITY_HYDRO,
                entity_id,
                stage_id,
                BOUND_TURBINED_MIN,
                bb.min_turbined_m3s
            );
            append_bound!(
                OUTPUT_ENTITY_HYDRO,
                entity_id,
                stage_id,
                BOUND_TURBINED_MAX,
                bb.max_turbined_m3s
            );
            append_bound!(
                OUTPUT_ENTITY_HYDRO,
                entity_id,
                stage_id,
                BOUND_OUTFLOW_MIN,
                bb.min_outflow_m3s
            );
            if let Some(max_outflow) = bb.max_outflow_m3s {
                append_bound!(
                    OUTPUT_ENTITY_HYDRO,
                    entity_id,
                    stage_id,
                    BOUND_OUTFLOW_MAX,
                    max_outflow
                );
            }
            append_bound!(
                OUTPUT_ENTITY_HYDRO,
                entity_id,
                stage_id,
                BOUND_GENERATION_MIN,
                bb.min_generation_mw
            );
            append_bound!(
                OUTPUT_ENTITY_HYDRO,
                entity_id,
                stage_id,
                BOUND_GENERATION_MAX,
                bb.max_generation_mw
            );

            if has_overlay {
                for blk in 0..system.stages()[stage_idx].blocks.len() {
                    let block_id = blk as i32;
                    let o = overlay.hydro_override(hydro_idx, stage_idx, blk);
                    if let Some(v) = o.min_turbined_m3s {
                        append_block_bound!(
                            OUTPUT_ENTITY_HYDRO,
                            entity_id,
                            stage_id,
                            block_id,
                            BOUND_TURBINED_MIN,
                            v
                        );
                    }
                    if let Some(v) = o.max_turbined_m3s {
                        append_block_bound!(
                            OUTPUT_ENTITY_HYDRO,
                            entity_id,
                            stage_id,
                            block_id,
                            BOUND_TURBINED_MAX,
                            v
                        );
                    }
                    if let Some(v) = o.min_outflow_m3s {
                        append_block_bound!(
                            OUTPUT_ENTITY_HYDRO,
                            entity_id,
                            stage_id,
                            block_id,
                            BOUND_OUTFLOW_MIN,
                            v
                        );
                    }
                    if let Some(v) = o.max_outflow_m3s {
                        append_block_bound!(
                            OUTPUT_ENTITY_HYDRO,
                            entity_id,
                            stage_id,
                            block_id,
                            BOUND_OUTFLOW_MAX,
                            v
                        );
                    }
                    if let Some(v) = o.min_generation_mw {
                        append_block_bound!(
                            OUTPUT_ENTITY_HYDRO,
                            entity_id,
                            stage_id,
                            block_id,
                            BOUND_GENERATION_MIN,
                            v
                        );
                    }
                    if let Some(v) = o.max_generation_mw {
                        append_block_bound!(
                            OUTPUT_ENTITY_HYDRO,
                            entity_id,
                            stage_id,
                            block_id,
                            BOUND_GENERATION_MAX,
                            v
                        );
                    }
                }
            }
        }
    }

    for (thermal_idx, thermal) in system.thermals().iter().enumerate() {
        let entity_id = thermal.id.0;
        for stage_idx in 0..n_stages {
            let stage_id = system.stages()[stage_idx].id;
            let b = system.bounds().thermal_block_base(thermal_idx, stage_idx);

            append_bound!(
                OUTPUT_ENTITY_THERMAL,
                entity_id,
                stage_id,
                BOUND_GENERATION_MIN,
                b.min_generation_mw
            );
            append_bound!(
                OUTPUT_ENTITY_THERMAL,
                entity_id,
                stage_id,
                BOUND_GENERATION_MAX,
                b.max_generation_mw
            );

            if has_overlay {
                for blk in 0..system.stages()[stage_idx].blocks.len() {
                    let block_id = blk as i32;
                    let o = overlay.thermal_override(thermal_idx, stage_idx, blk);
                    if let Some(v) = o.min_generation_mw {
                        append_block_bound!(
                            OUTPUT_ENTITY_THERMAL,
                            entity_id,
                            stage_id,
                            block_id,
                            BOUND_GENERATION_MIN,
                            v
                        );
                    }
                    if let Some(v) = o.max_generation_mw {
                        append_block_bound!(
                            OUTPUT_ENTITY_THERMAL,
                            entity_id,
                            stage_id,
                            block_id,
                            BOUND_GENERATION_MAX,
                            v
                        );
                    }
                }
            }
        }
    }

    for (line_idx, line) in system.lines().iter().enumerate() {
        let entity_id = line.id.0;
        for stage_idx in 0..n_stages {
            let stage_id = system.stages()[stage_idx].id;
            let b = system.bounds().line_block_base(line_idx, stage_idx);

            append_bound!(
                OUTPUT_ENTITY_LINE,
                entity_id,
                stage_id,
                BOUND_FLOW_MIN,
                0.0_f64
            );
            append_bound!(
                OUTPUT_ENTITY_LINE,
                entity_id,
                stage_id,
                BOUND_FLOW_MAX,
                b.direct_mw
            );

            if has_overlay {
                for blk in 0..system.stages()[stage_idx].blocks.len() {
                    let block_id = blk as i32;
                    let o = overlay.line_override(line_idx, stage_idx, blk);
                    if let Some(v) = o.direct_mw {
                        append_block_bound!(
                            OUTPUT_ENTITY_LINE,
                            entity_id,
                            stage_id,
                            block_id,
                            BOUND_FLOW_MAX,
                            v
                        );
                    }
                }
            }
        }
    }

    for (pumping_idx, pumping) in system.pumping_stations().iter().enumerate() {
        let entity_id = pumping.id.0;
        for stage_idx in 0..n_stages {
            let stage_id = system.stages()[stage_idx].id;
            let b = system.bounds().pumping_block_base(pumping_idx, stage_idx);

            append_bound!(
                OUTPUT_ENTITY_PUMPING_STATION,
                entity_id,
                stage_id,
                BOUND_FLOW_MIN,
                b.min_flow_m3s
            );
            append_bound!(
                OUTPUT_ENTITY_PUMPING_STATION,
                entity_id,
                stage_id,
                BOUND_FLOW_MAX,
                b.max_flow_m3s
            );

            if has_overlay {
                for blk in 0..system.stages()[stage_idx].blocks.len() {
                    let block_id = blk as i32;
                    let o = overlay.pumping_override(pumping_idx, stage_idx, blk);
                    if let Some(v) = o.min_flow_m3s {
                        append_block_bound!(
                            OUTPUT_ENTITY_PUMPING_STATION,
                            entity_id,
                            stage_id,
                            block_id,
                            BOUND_FLOW_MIN,
                            v
                        );
                    }
                    if let Some(v) = o.max_flow_m3s {
                        append_block_bound!(
                            OUTPUT_ENTITY_PUMPING_STATION,
                            entity_id,
                            stage_id,
                            block_id,
                            BOUND_FLOW_MAX,
                            v
                        );
                    }
                }
            }
        }
    }

    for (contract_idx, contract) in system.contracts().iter().enumerate() {
        let entity_id = contract.id.0;
        for stage_idx in 0..n_stages {
            let stage_id = system.stages()[stage_idx].id;
            let b = system.bounds().contract_block_base(contract_idx, stage_idx);

            append_bound!(
                OUTPUT_ENTITY_CONTRACT,
                entity_id,
                stage_id,
                BOUND_FLOW_MIN,
                b.min_mw
            );
            append_bound!(
                OUTPUT_ENTITY_CONTRACT,
                entity_id,
                stage_id,
                BOUND_FLOW_MAX,
                b.max_mw
            );

            if has_overlay {
                for blk in 0..system.stages()[stage_idx].blocks.len() {
                    let block_id = blk as i32;
                    let o = overlay.contract_override(contract_idx, stage_idx, blk);
                    if let Some(v) = o.min_mw {
                        append_block_bound!(
                            OUTPUT_ENTITY_CONTRACT,
                            entity_id,
                            stage_id,
                            block_id,
                            BOUND_FLOW_MIN,
                            v
                        );
                    }
                    if let Some(v) = o.max_mw {
                        append_block_bound!(
                            OUTPUT_ENTITY_CONTRACT,
                            entity_id,
                            stage_id,
                            block_id,
                            BOUND_FLOW_MAX,
                            v
                        );
                    }
                }
            }
        }
    }

    let group_overlay = system.bounds().group_overlay();
    let has_group_overlay = !group_overlay.is_empty();

    for (hydro_idx, hydro) in system.hydros().iter().enumerate() {
        let plant_id = hydro.id.0;
        for (group_pos, group) in hydro.unit_groups.iter().enumerate() {
            let entity_id = group.id.0;
            for stage_idx in 0..n_stages {
                let stage_id = system.stages()[stage_idx].id;
                let stage_override = group_overlay.stage_override(hydro_idx, group_pos, stage_idx);

                append_group_bound!(
                    entity_id,
                    plant_id,
                    stage_id,
                    BOUND_TURBINED_MIN,
                    stage_override
                        .min_turbined_m3s
                        .unwrap_or(group.min_turbined_m3s)
                );
                append_group_bound!(
                    entity_id,
                    plant_id,
                    stage_id,
                    BOUND_TURBINED_MAX,
                    stage_override
                        .max_turbined_m3s
                        .unwrap_or(group.max_turbined_m3s)
                );
                append_group_bound!(
                    entity_id,
                    plant_id,
                    stage_id,
                    BOUND_GENERATION_MIN,
                    stage_override
                        .min_generation_mw
                        .unwrap_or(group.min_generation_mw)
                );
                append_group_bound!(
                    entity_id,
                    plant_id,
                    stage_id,
                    BOUND_GENERATION_MAX,
                    stage_override
                        .max_generation_mw
                        .unwrap_or(group.max_generation_mw)
                );

                if has_group_overlay {
                    for blk in 0..system.stages()[stage_idx].blocks.len() {
                        let block_id = blk as i32;
                        let block_override =
                            group_overlay.block_override(hydro_idx, group_pos, stage_idx, blk);
                        if let Some(v) = block_override.min_turbined_m3s {
                            append_group_block_bound!(
                                entity_id,
                                plant_id,
                                stage_id,
                                block_id,
                                BOUND_TURBINED_MIN,
                                v
                            );
                        }
                        if let Some(v) = block_override.max_turbined_m3s {
                            append_group_block_bound!(
                                entity_id,
                                plant_id,
                                stage_id,
                                block_id,
                                BOUND_TURBINED_MAX,
                                v
                            );
                        }
                        if let Some(v) = block_override.min_generation_mw {
                            append_group_block_bound!(
                                entity_id,
                                plant_id,
                                stage_id,
                                block_id,
                                BOUND_GENERATION_MIN,
                                v
                            );
                        }
                        if let Some(v) = block_override.max_generation_mw {
                            append_group_block_bound!(
                                entity_id,
                                plant_id,
                                stage_id,
                                block_id,
                                BOUND_GENERATION_MAX,
                                v
                            );
                        }
                    }
                }
            }
        }
    }

    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(entity_type_codes.finish()),
            Arc::new(entity_ids.finish()),
            Arc::new(hydro_ids.finish()),
            Arc::new(stage_ids.finish()),
            Arc::new(block_ids.finish()),
            Arc::new(bound_type_codes.finish()),
            Arc::new(bound_values.finish()),
        ],
    )
    .map_err(|e| OutputError::serialization("bounds", e.to_string()))?;

    let parquet_path = path.join("bounds.parquet");
    write_parquet_atomic(&parquet_path, &batch, config)
}

/// Build the Arrow schema for `bounds.parquet`.
fn bounds_schema() -> Schema {
    Schema::new(vec![
        Field::new("entity_type_code", DataType::Int8, false),
        Field::new("entity_id", DataType::Int32, false),
        Field::new("hydro_id", DataType::Int32, true),
        Field::new("stage_id", DataType::Int32, false),
        Field::new("block_id", DataType::Int32, true),
        Field::new("bound_type_code", DataType::Int8, false),
        Field::new("bound_value", DataType::Float64, false),
    ])
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
mod tests {
    use super::*;
    use arrow::array::Array;
    use chrono::NaiveDate;
    use cobre_core::{
        Block, BlockMode, Bus, ContractType, DeficitSegment, EnergyContract, EntityId, Hydro,
        HydroGenerationModel, HydroPenalties, HydroUnitGroup, Line, NoiseMethod, PumpingStation,
        ScenarioSourceConfig, Stage, StageRiskConfig, StageStateConfig, SystemBuilder, Thermal,
        resolved::{
            BlockBoundsCountsSpec, BoundsCountsSpec, BoundsDefaults, ContractBlockBounds,
            HydroBlockBounds, HydroStageBounds, HydroUnitGroupBoundsCountsSpec, LineBlockBounds,
            PumpingBlockBounds, ResolvedBlockBounds, ResolvedBounds, ResolvedHydroUnitGroupBounds,
            ThermalBlockBounds, ThermalStageBounds,
        },
    };

    #[test]
    fn every_listed_schema_column_has_a_nonempty_description() {
        for (file, schema) in &variables_csv_schemas() {
            for field in schema.fields() {
                let description = description_for(file, field.name());
                assert!(
                    !description.is_empty(),
                    "variables.csv column '{file}.{}' has an empty description",
                    field.name()
                );
            }
        }
    }

    // ── Fixtures ─────────────────────────────────────────────────────────────

    fn hydro_penalties_zero() -> HydroPenalties {
        HydroPenalties {
            spillage_cost: 0.0,
            diversion_cost: 0.0,
            turbined_cost: 0.0,
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
        }
    }

    fn make_hydro(id: i32, name: &str, bus_id: i32) -> Hydro {
        let mut hydro = Hydro {
            unit_groups: Vec::new(),
            id: EntityId(id),
            name: name.to_string(),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            downstream_id: None,
            travel_time_hours: None,
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: 0.0,
            max_storage_hm3: 100.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: HydroGenerationModel::ConstantProductivity,
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 50.0,
            specific_productivity_mw_per_m3s_per_m: None,
            min_generation_mw: 0.0,
            max_generation_mw: 45.0,
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: None,
            evaporation_reference_volumes_hm3: None,
            diversion: None,
            filling: None,
            penalties: hydro_penalties_zero(),
        };
        hydro.declare_mirror_unit_group(EntityId(bus_id));
        hydro
    }

    fn make_thermal(id: i32, name: &str, bus_id: i32) -> Thermal {
        Thermal {
            id: EntityId(id),
            name: name.to_string(),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId(bus_id),
            entry_stage_id: None,
            exit_stage_id: None,
            cost_per_mwh: 50.0,
            min_generation_mw: 0.0,
            max_generation_mw: 100.0,
            anticipated_config: None,
        }
    }

    fn make_bus(id: i32) -> Bus {
        Bus {
            id: EntityId(id),
            name: format!("Bus{id}"),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 1000.0,
            }],
            excess_cost: 0.0,
        }
    }

    fn make_stage(id: i32) -> Stage {
        Stage {
            index: id.max(0) as usize,
            id,
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            season_id: Some(0),
            blocks: vec![Block {
                index: 0,
                name: "SINGLE".to_string(),
                duration_hours: 720.0,
            }],
            block_mode: BlockMode::Parallel,
            state_config: StageStateConfig {
                storage: true,
                inflow_lags: false,
            },
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor: 10,
                noise_method: NoiseMethod::Saa,
            },
        }
    }

    fn hydro_stage_bounds(min_storage_hm3: f64, max_storage_hm3: f64) -> HydroStageBounds {
        HydroStageBounds {
            min_storage_hm3,
            max_storage_hm3,
            filling_min_rate_m3s: 0.0,
            water_withdrawal_m3s: 0.0,
        }
    }

    fn hydro_block_bounds_default() -> HydroBlockBounds {
        HydroBlockBounds {
            max_turbined_m3s: 50.0,
            max_generation_mw: 45.0,
            ..Default::default()
        }
    }

    /// Build a `System` with 2 hydros and 1 thermal for standard tests.
    fn make_system_2h_1t() -> System {
        let bus = make_bus(1);
        let h1 = make_hydro(1, "Hydro1", 1);
        let h2 = make_hydro(2, "Hydro2", 1);
        let t1 = make_thermal(1, "Thermal1", 1);
        let stage = make_stage(0);

        let hydro_bounds_default = hydro_stage_bounds(0.0, 100.0);
        let thermal_bounds_default = ThermalStageBounds { cost_per_mwh: 0.0 };
        let thermal_block_default = ThermalBlockBounds {
            min_generation_mw: 0.0,
            max_generation_mw: 100.0,
        };
        let line_default = LineBlockBounds {
            direct_mw: 500.0,
            reverse_mw: 500.0,
        };
        let pumping_default = PumpingBlockBounds {
            min_flow_m3s: 0.0,
            max_flow_m3s: 0.0,
        };
        let contract_default = ContractBlockBounds {
            min_mw: 0.0,
            max_mw: 0.0,
            price_per_mwh: 0.0,
        };
        let bounds = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 2,
                n_thermals: 1,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages: 1,
                k_max: 0,
            },
            &BoundsDefaults {
                hydro: hydro_bounds_default,
                hydro_block: hydro_block_bounds_default(),
                thermal: thermal_bounds_default,
                thermal_block: thermal_block_default,
                line_block: line_default,
                pumping_block: pumping_default,
                contract_block: contract_default,
            },
        );

        SystemBuilder::new()
            .buses(vec![bus])
            .hydros(vec![h1, h2])
            .thermals(vec![t1])
            .stages(vec![stage])
            .bounds(bounds)
            .build()
            .expect("valid system")
    }

    /// Build a `System` with 1 hydro, 2 stages, and custom bounds for bounds tests.
    fn make_system_1h_2stages(min_storage: f64, max_storage: f64) -> System {
        let bus = make_bus(1);
        let h1 = make_hydro(1, "Hydro1", 1);
        let stage0 = make_stage(0);
        let stage1 = make_stage(1);

        let hydro_bounds_default = hydro_stage_bounds(min_storage, max_storage);
        let thermal_default = ThermalStageBounds { cost_per_mwh: 0.0 };
        let thermal_block_default = ThermalBlockBounds {
            min_generation_mw: 0.0,
            max_generation_mw: 100.0,
        };
        let line_default = LineBlockBounds {
            direct_mw: 500.0,
            reverse_mw: 500.0,
        };
        let pumping_default = PumpingBlockBounds {
            min_flow_m3s: 0.0,
            max_flow_m3s: 0.0,
        };
        let contract_default = ContractBlockBounds {
            min_mw: 0.0,
            max_mw: 0.0,
            price_per_mwh: 0.0,
        };
        let bounds = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 1,
                n_thermals: 0,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages: 2,
                k_max: 0,
            },
            &BoundsDefaults {
                hydro: hydro_bounds_default,
                hydro_block: hydro_block_bounds_default(),
                thermal: thermal_default,
                thermal_block: thermal_block_default,
                line_block: line_default,
                pumping_block: pumping_default,
                contract_block: contract_default,
            },
        );

        SystemBuilder::new()
            .buses(vec![bus])
            .hydros(vec![h1])
            .stages(vec![stage0, stage1])
            .bounds(bounds)
            .build()
            .expect("valid system")
    }

    /// Build a stage with `n` equal-duration blocks, indices `0..n`.
    fn make_stage_with_blocks(id: i32, n: usize) -> Stage {
        Stage {
            index: id.max(0) as usize,
            id,
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            season_id: Some(0),
            blocks: (0..n)
                .map(|i| Block {
                    index: i,
                    name: format!("BLK{i}"),
                    duration_hours: 1.0,
                })
                .collect(),
            block_mode: BlockMode::Parallel,
            state_config: StageStateConfig {
                storage: true,
                inflow_lags: false,
            },
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor: 10,
                noise_method: NoiseMethod::Saa,
            },
        }
    }

    /// Build a 2-hydro, 2-thermal, 2-stage system (stage 0 has 3 blocks, stage
    /// 1 has 2) installing `overlay` as the bounds' per-block override table.
    fn make_system_2h_2t_blocks(overlay: ResolvedBlockBounds) -> System {
        let bus = make_bus(1);
        let h1 = make_hydro(1, "Hydro1", 1);
        let h2 = make_hydro(2, "Hydro2", 1);
        let t1 = make_thermal(1, "Thermal1", 1);
        let t2 = make_thermal(2, "Thermal2", 1);
        let stage0 = make_stage_with_blocks(0, 3);
        let stage1 = make_stage_with_blocks(1, 2);

        let hydro_bounds_default = hydro_stage_bounds(0.0, 100.0);
        let thermal_bounds_default = ThermalStageBounds { cost_per_mwh: 0.0 };
        let thermal_block_default = ThermalBlockBounds {
            min_generation_mw: 0.0,
            max_generation_mw: 100.0,
        };
        let line_default = LineBlockBounds {
            direct_mw: 500.0,
            reverse_mw: 500.0,
        };
        let pumping_default = PumpingBlockBounds {
            min_flow_m3s: 0.0,
            max_flow_m3s: 0.0,
        };
        let contract_default = ContractBlockBounds {
            min_mw: 0.0,
            max_mw: 0.0,
            price_per_mwh: 0.0,
        };
        let mut bounds = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 2,
                n_thermals: 2,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages: 2,
                k_max: 0,
            },
            &BoundsDefaults {
                hydro: hydro_bounds_default,
                hydro_block: hydro_block_bounds_default(),
                thermal: thermal_bounds_default,
                thermal_block: thermal_block_default,
                line_block: line_default,
                pumping_block: pumping_default,
                contract_block: contract_default,
            },
        );
        bounds.set_block_overlay(overlay);

        SystemBuilder::new()
            .buses(vec![bus])
            .hydros(vec![h1, h2])
            .thermals(vec![t1, t2])
            .stages(vec![stage0, stage1])
            .bounds(bounds)
            .build()
            .expect("valid system")
    }

    /// Build a per-block override table sized for [`make_system_2h_2t_blocks`]
    /// with `max_turbined_m3s = 42.0` on `(hydro_idx=1, stage_idx=1,
    /// block_idx=1)` and `max_generation_mw = 7.0` on `(thermal_idx=0,
    /// stage_idx=0, block_idx=2)` — different entity, stage, and block on each.
    fn make_two_block_overrides() -> ResolvedBlockBounds {
        let mut overlay = ResolvedBlockBounds::new(&BlockBoundsCountsSpec {
            n_hydros: 2,
            n_thermals: 2,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages: 2,
            max_blocks: 3,
        });
        overlay
            .hydro_override_mut(1, 1, 1)
            .expect("cell must exist for a fixture-sized overlay")
            .max_turbined_m3s = Some(42.0);
        overlay
            .thermal_override_mut(0, 0, 2)
            .expect("cell must exist for a fixture-sized overlay")
            .max_generation_mw = Some(7.0);
        overlay
    }

    fn make_line(id: i32, bus_id: i32) -> Line {
        Line {
            id: EntityId(id),
            name: format!("Line{id}"),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            source_bus_id: EntityId(bus_id),
            target_bus_id: EntityId(bus_id),
            entry_stage_id: None,
            exit_stage_id: None,
            direct_capacity_mw: 500.0,
            reverse_capacity_mw: 500.0,
            losses_percent: 0.0,
            exchange_cost: 0.0,
        }
    }

    fn make_pumping_station(
        id: i32,
        bus_id: i32,
        source_hydro_id: i32,
        destination_hydro_id: i32,
    ) -> PumpingStation {
        PumpingStation {
            id: EntityId(id),
            name: format!("Pumping{id}"),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId(bus_id),
            source_hydro_id: EntityId(source_hydro_id),
            destination_hydro_id: EntityId(destination_hydro_id),
            entry_stage_id: None,
            exit_stage_id: None,
            consumption_mw_per_m3s: 0.5,
            min_flow_m3s: 0.0,
            max_flow_m3s: 100.0,
        }
    }

    fn make_contract(id: i32, bus_id: i32) -> EnergyContract {
        EnergyContract {
            id: EntityId(id),
            name: format!("Contract{id}"),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId(bus_id),
            contract_type: ContractType::Import,
            entry_stage_id: None,
            exit_stage_id: None,
            price_per_mwh: 50.0,
            min_mw: 0.0,
            max_mw: 200.0,
        }
    }

    /// Build a system with 2 hydros (referenced only as the pumping station's
    /// distinct source/destination reservoirs), 2 lines, 1 pumping station,
    /// and 2 contracts across 2 stages (stage 0 has 3 blocks, stage 1 has 2),
    /// installing `overlay` as the bounds' per-block override table.
    ///
    /// A companion to [`make_system_2h_2t_blocks`] rather than an extension of
    /// it: `bounds_parquet_emits_no_block_rows_when_overlay_empty` hard-codes
    /// its expected row count from that fixture's hydro/thermal counts alone,
    /// so adding line/pumping/contract entities there would require editing
    /// that count instead of only adding coverage.
    fn make_system_lines_pumping_contracts_blocks(overlay: ResolvedBlockBounds) -> System {
        let bus = make_bus(1);
        let h1 = make_hydro(1, "Hydro1", 1);
        let h2 = make_hydro(2, "Hydro2", 1);
        let l1 = make_line(1, 1);
        let l2 = make_line(2, 1);
        let p1 = make_pumping_station(1, 1, 1, 2);
        let c1 = make_contract(1, 1);
        let c2 = make_contract(2, 1);
        let stage0 = make_stage_with_blocks(0, 3);
        let stage1 = make_stage_with_blocks(1, 2);

        let hydro_bounds_default = hydro_stage_bounds(0.0, 100.0);
        let thermal_bounds_default = ThermalStageBounds { cost_per_mwh: 0.0 };
        let thermal_block_default = ThermalBlockBounds {
            min_generation_mw: 0.0,
            max_generation_mw: 0.0,
        };
        let line_default = LineBlockBounds {
            direct_mw: 500.0,
            reverse_mw: 500.0,
        };
        let pumping_default = PumpingBlockBounds {
            min_flow_m3s: 0.0,
            max_flow_m3s: 80.0,
        };
        let contract_default = ContractBlockBounds {
            min_mw: 0.0,
            max_mw: 200.0,
            price_per_mwh: 50.0,
        };
        let mut bounds = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 2,
                n_thermals: 0,
                n_lines: 2,
                n_pumping: 1,
                n_contracts: 2,
                n_stages: 2,
                k_max: 0,
            },
            &BoundsDefaults {
                hydro: hydro_bounds_default,
                hydro_block: hydro_block_bounds_default(),
                thermal: thermal_bounds_default,
                thermal_block: thermal_block_default,
                line_block: line_default,
                pumping_block: pumping_default,
                contract_block: contract_default,
            },
        );
        bounds.set_block_overlay(overlay);

        SystemBuilder::new()
            .buses(vec![bus])
            .hydros(vec![h1, h2])
            .lines(vec![l1, l2])
            .pumping_stations(vec![p1])
            .contracts(vec![c1, c2])
            .stages(vec![stage0, stage1])
            .bounds(bounds)
            .build()
            .expect("valid system")
    }

    /// Build a per-block override table sized for
    /// [`make_system_lines_pumping_contracts_blocks`]. Each of the three
    /// families sits at a cell where `stage_idx != block_idx` (guards against
    /// a `block_id`-from-`stage_idx` regression), and the entity index varies
    /// (`line_idx=1`, `contract_idx=1`) so a hard-coded-index regression is
    /// also caught:
    ///
    /// - line: `direct_mw = 123.0` at `(line_idx=1, stage_idx=0, block_idx=2)`.
    /// - pumping: `min_flow_m3s = 11.0` at `(pumping_idx=0, stage_idx=1,
    ///   block_idx=0)` — `max_flow_m3s` left unset, to catch a bug that
    ///   re-emits unset columns.
    /// - contract: `min_mw = 5.0`, `max_mw = 50.0` at `(contract_idx=1,
    ///   stage_idx=0, block_idx=1)`.
    fn make_line_pumping_contract_block_overrides() -> ResolvedBlockBounds {
        let mut overlay = ResolvedBlockBounds::new(&BlockBoundsCountsSpec {
            n_hydros: 2,
            n_thermals: 0,
            n_lines: 2,
            n_pumping: 1,
            n_contracts: 2,
            n_stages: 2,
            max_blocks: 3,
        });
        overlay
            .line_override_mut(1, 0, 2)
            .expect("cell must exist for a fixture-sized overlay")
            .direct_mw = Some(123.0);
        overlay
            .pumping_override_mut(0, 1, 0)
            .expect("cell must exist for a fixture-sized overlay")
            .min_flow_m3s = Some(11.0);
        let contract = overlay
            .contract_override_mut(1, 0, 1)
            .expect("cell must exist for a fixture-sized overlay");
        contract.min_mw = Some(5.0);
        contract.max_mw = Some(50.0);
        overlay
    }

    // ── codes.json ────────────────────────────────────────────────────────────

    #[test]
    fn codes_json_roundtrip() {
        let tmp = tempfile::tempdir().unwrap();
        write_codes_json(tmp.path()).expect("write_codes_json must succeed");

        let raw = std::fs::read_to_string(tmp.path().join("codes.json")).unwrap();
        let val: serde_json::Value = serde_json::from_str(&raw).unwrap();

        assert_eq!(
            val["operative_state"]["2"],
            serde_json::json!("operating"),
            "operative_state[\"2\"] must equal \"operating\""
        );
        assert_eq!(
            val["storage_binding"]["1"],
            serde_json::json!("below_minimum"),
            "storage_binding[\"1\"] must equal \"below_minimum\""
        );
        assert_eq!(
            val["entity_type"]["0"],
            serde_json::json!("hydro"),
            "entity_type[\"0\"] must equal \"hydro\""
        );
        assert_eq!(
            val["entity_type"]["8"],
            serde_json::json!("hydro_unit_group"),
            "entity_type[\"8\"] must equal \"hydro_unit_group\""
        );
        assert!(
            val["entity_type"].get("9").is_none(),
            "entity_type must have no \"9\" key: no study can emit code 9"
        );
        assert_eq!(
            val["bound_type"]["0"],
            serde_json::json!("storage_min"),
            "bound_type[\"0\"] must equal \"storage_min\""
        );
        assert!(
            val["generated_at"].is_string(),
            "generated_at must be a string"
        );
        assert_eq!(
            val["version"],
            serde_json::json!("1.0"),
            "version must be \"1.0\""
        );
    }

    // ── entities.csv ─────────────────────────────────────────────────────────

    #[test]
    fn entities_csv_correct_rows() {
        let system = make_system_2h_1t();
        let tmp = tempfile::tempdir().unwrap();
        write_entities_csv(tmp.path(), &system).expect("write_entities_csv must succeed");

        assert!(
            !tmp.path().join("entities.csv.tmp").exists(),
            "no .tmp file must remain beside entities.csv"
        );

        let content = std::fs::read_to_string(tmp.path().join("entities.csv")).unwrap();
        let mut rdr = csv::Reader::from_reader(content.as_bytes());

        let headers: Vec<String> = rdr
            .headers()
            .unwrap()
            .iter()
            .map(ToString::to_string)
            .collect();
        assert_eq!(
            headers,
            vec![
                "entity_type_code",
                "entity_id",
                "name",
                "bus_id",
                "system_id"
            ],
            "header row must stay exactly entity_type_code,entity_id,name,bus_id,system_id"
        );

        let rows: Vec<Vec<String>> = rdr
            .records()
            .map(|r| r.unwrap().iter().map(ToString::to_string).collect())
            .collect();

        assert_eq!(
            rows.len(),
            6,
            "expected 6 data rows (2 hydros + 1 thermal + 1 bus + 2 hydro-unit-group rows)"
        );

        assert_eq!(rows[0][0], "0", "row 0: entity_type_code must be 0 (hydro)");
        assert_eq!(rows[0][1], "1", "row 0: entity_id must be 1");
        assert_eq!(rows[0][2], "Hydro1", "row 0: name must be Hydro1");
        assert_eq!(
            rows[0][3], "-1",
            "row 0: bus_id must be the -1 sentinel; the group rows own the association"
        );

        assert_eq!(rows[1][0], "0", "row 1: entity_type_code must be 0 (hydro)");
        assert_eq!(rows[1][1], "2", "row 1: entity_id must be 2");
        assert_eq!(rows[1][2], "Hydro2", "row 1: name must be Hydro2");
        assert_eq!(
            rows[1][3], "-1",
            "row 1: bus_id must be the -1 sentinel; the group rows own the association"
        );

        assert_eq!(
            rows[2][0], "1",
            "row 2: entity_type_code must be 1 (thermal)"
        );
        assert_eq!(rows[2][1], "1", "row 2: entity_id must be 1");
        assert_eq!(rows[2][2], "Thermal1", "row 2: name must be Thermal1");
        assert_eq!(
            rows[2][3], "1",
            "row 2: bus_id must still be the thermal's own bus, unlike the hydro rows"
        );

        assert_eq!(
            rows[4][0], "8",
            "row 4: entity_type_code must be 8 (hydro_unit_group)"
        );
        assert_eq!(
            rows[4][1], "0",
            "row 4: entity_id must be the group's own id"
        );
        assert_eq!(rows[4][3], "1", "row 4: bus_id must equal Hydro1's own bus");
        assert_eq!(rows[4][4], "0", "row 4: system_id must be 0");

        assert_eq!(
            rows[5][0], "8",
            "row 5: entity_type_code must be 8 (hydro_unit_group)"
        );
        assert_eq!(
            rows[5][1], "0",
            "row 5: entity_id must be the group's own id"
        );
        assert_eq!(rows[5][3], "1", "row 5: bus_id must equal Hydro2's own bus");
        assert_eq!(rows[5][4], "0", "row 5: system_id must be 0");
    }

    /// Build a 2-hydro system where both plants declare a unit group sharing
    /// the same plant-local id and the same group name, so a rendering that
    /// dropped the plant-numeric prefix would collide.
    fn make_system_2h_same_group_id_and_name() -> System {
        let mut h1 = make_hydro(1, "PlantA", 1);
        h1.unit_groups = vec![HydroUnitGroup {
            id: EntityId(0),
            name: "GroupX".to_string(),
            bus_id: EntityId(1),
            min_generation_mw: 0.0,
            max_generation_mw: 45.0,
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 50.0,
        }];
        let mut h2 = make_hydro(2, "PlantB", 1);
        h2.unit_groups = vec![HydroUnitGroup {
            id: EntityId(0),
            name: "GroupX".to_string(),
            bus_id: EntityId(1),
            min_generation_mw: 0.0,
            max_generation_mw: 45.0,
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 50.0,
        }];

        let bus = make_bus(1);
        let stage = make_stage(0);
        let bounds = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 2,
                n_thermals: 0,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages: 1,
                k_max: 0,
            },
            &BoundsDefaults {
                hydro: hydro_stage_bounds(0.0, 100.0),
                hydro_block: hydro_block_bounds_default(),
                thermal: ThermalStageBounds { cost_per_mwh: 0.0 },
                thermal_block: ThermalBlockBounds {
                    min_generation_mw: 0.0,
                    max_generation_mw: 100.0,
                },
                line_block: LineBlockBounds {
                    direct_mw: 500.0,
                    reverse_mw: 500.0,
                },
                pumping_block: PumpingBlockBounds {
                    min_flow_m3s: 0.0,
                    max_flow_m3s: 0.0,
                },
                contract_block: ContractBlockBounds {
                    min_mw: 0.0,
                    max_mw: 0.0,
                    price_per_mwh: 0.0,
                },
            },
        );

        SystemBuilder::new()
            .buses(vec![bus])
            .hydros(vec![h1, h2])
            .stages(vec![stage])
            .bounds(bounds)
            .build()
            .expect("valid system")
    }

    #[test]
    fn entities_csv_group_rows_are_plant_qualified() {
        let system = make_system_2h_same_group_id_and_name();
        let tmp = tempfile::tempdir().unwrap();
        write_entities_csv(tmp.path(), &system).expect("write_entities_csv must succeed");

        let content = std::fs::read_to_string(tmp.path().join("entities.csv")).unwrap();
        let mut rdr = csv::Reader::from_reader(content.as_bytes());
        let rows: Vec<Vec<String>> = rdr
            .records()
            .map(|r| r.unwrap().iter().map(ToString::to_string).collect())
            .collect();

        let group_rows: Vec<&Vec<String>> = rows.iter().filter(|r| r[0] == "8").collect();
        assert_eq!(group_rows.len(), 2, "expected 2 hydro-unit-group rows");

        assert_eq!(
            group_rows[0][2], "1/GroupX",
            "Hydro 1's group row must be prefixed with its own hydro_id"
        );
        assert_eq!(
            group_rows[1][2], "2/GroupX",
            "Hydro 2's group row must be prefixed with its own hydro_id"
        );
        assert_ne!(
            group_rows[0][2], group_rows[1][2],
            "two plants sharing a group id and group name must still differ by numeric prefix"
        );
    }

    #[test]
    fn entities_csv_group_row_carries_the_groups_own_bus() {
        let mut h1 = make_hydro(1, "Plant1", 1);
        h1.unit_groups = vec![
            HydroUnitGroup {
                id: EntityId(5),
                name: "GroupHigh".to_string(),
                bus_id: EntityId(3),
                min_generation_mw: 0.0,
                max_generation_mw: 20.0,
                min_turbined_m3s: 0.0,
                max_turbined_m3s: 25.0,
            },
            HydroUnitGroup {
                id: EntityId(2),
                name: "GroupLow".to_string(),
                bus_id: EntityId(2),
                min_generation_mw: 0.0,
                max_generation_mw: 25.0,
                min_turbined_m3s: 0.0,
                max_turbined_m3s: 25.0,
            },
        ];

        let bus1 = make_bus(1);
        let bus2 = make_bus(2);
        let bus3 = make_bus(3);
        let stage = make_stage(0);
        let bounds = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 1,
                n_thermals: 0,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages: 1,
                k_max: 0,
            },
            &BoundsDefaults {
                hydro: hydro_stage_bounds(0.0, 100.0),
                hydro_block: hydro_block_bounds_default(),
                thermal: ThermalStageBounds { cost_per_mwh: 0.0 },
                thermal_block: ThermalBlockBounds {
                    min_generation_mw: 0.0,
                    max_generation_mw: 100.0,
                },
                line_block: LineBlockBounds {
                    direct_mw: 500.0,
                    reverse_mw: 500.0,
                },
                pumping_block: PumpingBlockBounds {
                    min_flow_m3s: 0.0,
                    max_flow_m3s: 0.0,
                },
                contract_block: ContractBlockBounds {
                    min_mw: 0.0,
                    max_mw: 0.0,
                    price_per_mwh: 0.0,
                },
            },
        );

        let system = SystemBuilder::new()
            .buses(vec![bus1, bus2, bus3])
            .hydros(vec![h1])
            .stages(vec![stage])
            .bounds(bounds)
            .build()
            .expect("valid system");

        let tmp = tempfile::tempdir().unwrap();
        write_entities_csv(tmp.path(), &system).expect("write_entities_csv must succeed");

        let content = std::fs::read_to_string(tmp.path().join("entities.csv")).unwrap();
        let mut rdr = csv::Reader::from_reader(content.as_bytes());
        let rows: Vec<Vec<String>> = rdr
            .records()
            .map(|r| r.unwrap().iter().map(ToString::to_string).collect())
            .collect();

        let group_rows: Vec<&Vec<String>> = rows.iter().filter(|r| r[0] == "8").collect();
        assert_eq!(group_rows.len(), 2, "expected 2 hydro-unit-group rows");

        assert_eq!(
            group_rows[0][1], "2",
            "first group row must be the lower group id (ascending group-id order)"
        );
        assert_eq!(
            group_rows[0][3], "2",
            "first group row's bus_id must be its own group's bus, not the plant's"
        );

        assert_eq!(
            group_rows[1][1], "5",
            "second group row must be the higher group id"
        );
        assert_eq!(
            group_rows[1][3], "3",
            "second group row's bus_id must be its own group's bus, not the plant's"
        );

        assert!(
            group_rows.iter().all(|r| r[3] != "1"),
            "neither group row may carry the plant's own bus_id"
        );
    }

    #[test]
    fn entities_csv_entity_type_order() {
        let system = make_system_2h_1t();
        let tmp = tempfile::tempdir().unwrap();
        write_entities_csv(tmp.path(), &system).expect("write_entities_csv must succeed");

        let content = std::fs::read_to_string(tmp.path().join("entities.csv")).unwrap();
        let mut rdr = csv::Reader::from_reader(content.as_bytes());

        let type_codes: Vec<i8> = rdr
            .records()
            .map(|r| r.unwrap().get(0).unwrap().parse::<i8>().unwrap())
            .collect();

        for window in type_codes.windows(2) {
            assert!(
                window[0] <= window[1],
                "entity_type_codes must be non-decreasing, found {} followed by {}",
                window[0],
                window[1]
            );
        }
    }

    #[test]
    fn entities_csv_system_id_is_zero() {
        let system = make_system_2h_1t();
        let tmp = tempfile::tempdir().unwrap();
        write_entities_csv(tmp.path(), &system).expect("write_entities_csv must succeed");

        let content = std::fs::read_to_string(tmp.path().join("entities.csv")).unwrap();
        let mut rdr = csv::Reader::from_reader(content.as_bytes());

        for rec in rdr.records() {
            let row = rec.unwrap();
            assert_eq!(row.get(4).unwrap(), "0", "system_id must be 0 for all rows");
        }
    }

    // ── variables.csv ─────────────────────────────────────────────────────────

    #[test]
    fn variables_csv_total_columns() {
        let tmp = tempfile::tempdir().unwrap();
        write_variables_csv(tmp.path()).expect("write_variables_csv must succeed");

        assert!(
            !tmp.path().join("variables.csv.tmp").exists(),
            "no .tmp file must remain beside variables.csv"
        );

        let content = std::fs::read_to_string(tmp.path().join("variables.csv")).unwrap();
        let mut rdr = csv::Reader::from_reader(content.as_bytes());

        let row_count = rdr.records().count();
        assert_eq!(
            row_count, 258,
            "variables.csv must have exactly 258 data rows (one per column across all schemas)"
        );
    }

    #[test]
    fn scenario_summary_cost_description_states_future_cost_exclusion() {
        let cost = description_for("scenario_summary", "discounted_immediate_cost");
        assert!(
            cost.contains("exclude") && cost.contains("total_cost"),
            "the discounted_immediate_cost description must state it excludes the future-cost \
             term that costs.parquet total_cost includes, got: {cost:?}"
        );
        assert!(
            !description_for("scenario_summary", "probability").is_empty(),
            "scenario_summary.probability must carry a description"
        );
    }

    #[test]
    fn variables_csv_has_required_columns_in_header() {
        let tmp = tempfile::tempdir().unwrap();
        write_variables_csv(tmp.path()).expect("write_variables_csv must succeed");

        let content = std::fs::read_to_string(tmp.path().join("variables.csv")).unwrap();
        let mut rdr = csv::Reader::from_reader(content.as_bytes());

        let headers: Vec<String> = rdr
            .headers()
            .unwrap()
            .iter()
            .map(ToString::to_string)
            .collect();

        assert!(
            headers.contains(&"file".to_string()),
            "header must contain 'file'"
        );
        assert!(
            headers.contains(&"column".to_string()),
            "header must contain 'column'"
        );
        assert!(
            headers.contains(&"type".to_string()),
            "header must contain 'type'"
        );
        assert!(
            headers.contains(&"unit".to_string()),
            "header must contain 'unit'"
        );
        assert!(
            headers.contains(&"description".to_string()),
            "header must contain 'description'"
        );
        assert!(
            headers.contains(&"nullable".to_string()),
            "header must contain 'nullable'"
        );
    }

    #[test]
    fn variables_csv_nullable_reflects_schema() {
        let tmp = tempfile::tempdir().unwrap();
        write_variables_csv(tmp.path()).expect("write_variables_csv must succeed");

        let content = std::fs::read_to_string(tmp.path().join("variables.csv")).unwrap();
        let mut rdr = csv::Reader::from_reader(content.as_bytes());

        let block_id_nullable = rdr
            .records()
            .find(|r| {
                let row = r.as_ref().unwrap();
                row.get(0).unwrap() == "costs" && row.get(1).unwrap() == "block_id"
            })
            .map(|r| r.unwrap().get(5).unwrap().to_string());

        assert_eq!(
            block_id_nullable,
            Some("true".to_string()),
            "costs.block_id must have nullable=true"
        );
    }

    // ── bounds.parquet ────────────────────────────────────────────────────────

    #[test]
    fn bounds_parquet_roundtrip() {
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

        let system = make_system_1h_2stages(100.0, 500.0);
        let tmp = tempfile::tempdir().unwrap();
        let config = ParquetWriterConfig::default();

        write_bounds_parquet(tmp.path(), &system, &config)
            .expect("write_bounds_parquet must succeed");

        let path = tmp.path().join("bounds.parquet");
        assert!(path.exists(), "bounds.parquet must exist");

        let file = std::fs::File::open(&path).unwrap();
        let builder = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
        let mut reader = builder.build().unwrap();
        let batch = reader.next().expect("must have rows").expect("batch Ok");

        // 1 hydro, 2 stages, 7 bound types per stage (no max_outflow) = 14
        // plant rows, plus `make_hydro`'s 1 mirror unit group × 2 stages × 4
        // group bound types = 8 group rows -- 22 total.
        assert_eq!(
            batch.num_rows(),
            22,
            "1 hydro × 2 stages × 7 bounds + 1 group × 2 stages × 4 bounds = 22 rows"
        );

        let entity_type_col = batch
            .column_by_name("entity_type_code")
            .unwrap()
            .as_any()
            .downcast_ref::<arrow::array::Int8Array>()
            .unwrap();
        let bound_type_col = batch
            .column_by_name("bound_type_code")
            .unwrap()
            .as_any()
            .downcast_ref::<arrow::array::Int8Array>()
            .unwrap();
        let bound_value_col = batch
            .column_by_name("bound_value")
            .unwrap()
            .as_any()
            .downcast_ref::<arrow::array::Float64Array>()
            .unwrap();
        let block_id_col = batch.column_by_name("block_id").unwrap();

        for row in 0..batch.num_rows() {
            assert!(
                block_id_col.is_null(row),
                "block_id must be null at row {row}"
            );
        }

        let storage_min_row = (0..batch.num_rows()).find(|&i| {
            entity_type_col.value(i) == OUTPUT_ENTITY_HYDRO
                && bound_type_col.value(i) == BOUND_STORAGE_MIN
        });
        assert!(
            storage_min_row.is_some(),
            "must have a storage_min row for hydro"
        );
        let row = storage_min_row.unwrap();
        assert!(
            (bound_value_col.value(row) - 100.0).abs() < f64::EPSILON,
            "storage_min must be 100.0, got {}",
            bound_value_col.value(row)
        );

        let storage_max_row = (0..batch.num_rows()).find(|&i| {
            entity_type_col.value(i) == OUTPUT_ENTITY_HYDRO
                && bound_type_col.value(i) == BOUND_STORAGE_MAX
        });
        assert!(
            storage_max_row.is_some(),
            "must have a storage_max row for hydro"
        );
        let row = storage_max_row.unwrap();
        assert!(
            (bound_value_col.value(row) - 500.0).abs() < f64::EPSILON,
            "storage_max must be 500.0, got {}",
            bound_value_col.value(row)
        );
    }

    #[test]
    fn bounds_parquet_emits_no_block_rows_when_overlay_empty() {
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

        let system = make_system_2h_2t_blocks(ResolvedBlockBounds::empty());
        let tmp = tempfile::tempdir().unwrap();
        let config = ParquetWriterConfig::default();

        write_bounds_parquet(tmp.path(), &system, &config)
            .expect("write_bounds_parquet must succeed");

        let file = std::fs::File::open(tmp.path().join("bounds.parquet")).unwrap();
        let builder = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
        let mut reader = builder.build().unwrap();
        let batch = reader.next().expect("must have rows").expect("batch Ok");

        // 2 hydros x 2 stages x 7 stage-level bounds (no max_outflow) = 28,
        // plus 2 thermals x 2 stages x 2 stage-level bounds = 8, plus each
        // hydro's 1 mirror unit group x 2 stages x 4 group bounds = 16.
        let expected_rows = 2 * 2 * 7 + 2 * 2 * 2 + 2 * 2 * 4;
        assert_eq!(
            batch.num_rows(),
            expected_rows,
            "row count must equal the stage-level mapping alone"
        );

        let block_id_col = batch.column_by_name("block_id").unwrap();
        for row in 0..batch.num_rows() {
            assert!(
                block_id_col.is_null(row),
                "block_id must be null at row {row} when the overlay is empty"
            );
        }
    }

    #[test]
    fn bounds_parquet_emits_per_block_override_rows() {
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

        let system = make_system_2h_2t_blocks(make_two_block_overrides());
        let tmp = tempfile::tempdir().unwrap();
        let config = ParquetWriterConfig::default();

        write_bounds_parquet(tmp.path(), &system, &config)
            .expect("write_bounds_parquet must succeed");

        let file = std::fs::File::open(tmp.path().join("bounds.parquet")).unwrap();
        let builder = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
        let mut reader = builder.build().unwrap();
        let batch = reader.next().expect("must have rows").expect("batch Ok");

        let entity_type_col = batch
            .column_by_name("entity_type_code")
            .unwrap()
            .as_any()
            .downcast_ref::<arrow::array::Int8Array>()
            .unwrap();
        let entity_id_col = batch
            .column_by_name("entity_id")
            .unwrap()
            .as_any()
            .downcast_ref::<arrow::array::Int32Array>()
            .unwrap();
        let stage_id_col = batch
            .column_by_name("stage_id")
            .unwrap()
            .as_any()
            .downcast_ref::<arrow::array::Int32Array>()
            .unwrap();
        let block_id_col = batch
            .column_by_name("block_id")
            .unwrap()
            .as_any()
            .downcast_ref::<arrow::array::Int32Array>()
            .unwrap();
        let bound_type_col = batch
            .column_by_name("bound_type_code")
            .unwrap()
            .as_any()
            .downcast_ref::<arrow::array::Int8Array>()
            .unwrap();
        let bound_value_col = batch
            .column_by_name("bound_value")
            .unwrap()
            .as_any()
            .downcast_ref::<arrow::array::Float64Array>()
            .unwrap();

        let block_rows: Vec<usize> = (0..batch.num_rows())
            .filter(|&i| !block_id_col.is_null(i))
            .collect();
        assert_eq!(
            block_rows.len(),
            2,
            "expected exactly two non-null-block_id rows"
        );

        let h2_id = system.hydros()[1].id.0;
        let t1_id = system.thermals()[0].id.0;
        let stage0_id = system.stages()[0].id;
        let stage1_id = system.stages()[1].id;

        let hydro_row = block_rows
            .iter()
            .copied()
            .find(|&i| entity_type_col.value(i) == OUTPUT_ENTITY_HYDRO)
            .expect("must have a hydro block row");
        assert_eq!(entity_id_col.value(hydro_row), h2_id, "hydro entity_id");
        assert_eq!(stage_id_col.value(hydro_row), stage1_id, "hydro stage_id");
        assert_eq!(block_id_col.value(hydro_row), 1, "hydro block_id");
        assert_eq!(
            bound_type_col.value(hydro_row),
            BOUND_TURBINED_MAX,
            "hydro bound_type_code"
        );
        assert_eq!(
            bound_value_col.value(hydro_row).to_bits(),
            42.0_f64.to_bits(),
            "hydro bound_value"
        );

        let thermal_row = block_rows
            .iter()
            .copied()
            .find(|&i| entity_type_col.value(i) == OUTPUT_ENTITY_THERMAL)
            .expect("must have a thermal block row");
        assert_eq!(entity_id_col.value(thermal_row), t1_id, "thermal entity_id");
        assert_eq!(
            stage_id_col.value(thermal_row),
            stage0_id,
            "thermal stage_id"
        );
        assert_eq!(block_id_col.value(thermal_row), 2, "thermal block_id");
        assert_eq!(
            bound_type_col.value(thermal_row),
            BOUND_GENERATION_MAX,
            "thermal bound_type_code"
        );
        assert_eq!(
            bound_value_col.value(thermal_row).to_bits(),
            7.0_f64.to_bits(),
            "thermal bound_value"
        );
    }

    #[test]
    fn bounds_parquet_stage_rows_unchanged_by_overlay() {
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

        fn read_null_block_rows(path: &std::path::Path) -> Vec<(i8, i32, i32, i8, u64)> {
            let file = std::fs::File::open(path).unwrap();
            let builder = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
            let mut reader = builder.build().unwrap();
            let batch = reader.next().expect("must have rows").expect("batch Ok");

            let entity_type_col = batch
                .column_by_name("entity_type_code")
                .unwrap()
                .as_any()
                .downcast_ref::<arrow::array::Int8Array>()
                .unwrap();
            let entity_id_col = batch
                .column_by_name("entity_id")
                .unwrap()
                .as_any()
                .downcast_ref::<arrow::array::Int32Array>()
                .unwrap();
            let stage_id_col = batch
                .column_by_name("stage_id")
                .unwrap()
                .as_any()
                .downcast_ref::<arrow::array::Int32Array>()
                .unwrap();
            let block_id_col = batch.column_by_name("block_id").unwrap();
            let bound_type_col = batch
                .column_by_name("bound_type_code")
                .unwrap()
                .as_any()
                .downcast_ref::<arrow::array::Int8Array>()
                .unwrap();
            let bound_value_col = batch
                .column_by_name("bound_value")
                .unwrap()
                .as_any()
                .downcast_ref::<arrow::array::Float64Array>()
                .unwrap();

            (0..batch.num_rows())
                .filter(|&i| block_id_col.is_null(i))
                .map(|i| {
                    (
                        entity_type_col.value(i),
                        entity_id_col.value(i),
                        stage_id_col.value(i),
                        bound_type_col.value(i),
                        bound_value_col.value(i).to_bits(),
                    )
                })
                .collect()
        }

        let config = ParquetWriterConfig::default();

        let empty_system = make_system_2h_2t_blocks(ResolvedBlockBounds::empty());
        let tmp_empty = tempfile::tempdir().unwrap();
        write_bounds_parquet(tmp_empty.path(), &empty_system, &config)
            .expect("write_bounds_parquet must succeed");
        let empty_rows = read_null_block_rows(&tmp_empty.path().join("bounds.parquet"));

        let overlaid_system = make_system_2h_2t_blocks(make_two_block_overrides());
        let tmp_overlaid = tempfile::tempdir().unwrap();
        write_bounds_parquet(tmp_overlaid.path(), &overlaid_system, &config)
            .expect("write_bounds_parquet must succeed");
        let overlaid_rows = read_null_block_rows(&tmp_overlaid.path().join("bounds.parquet"));

        assert_eq!(
            empty_rows, overlaid_rows,
            "the null-block_id row subsequence must be bit-identical regardless of the overlay"
        );
    }

    #[test]
    fn bounds_parquet_emits_per_block_override_rows_for_line_pumping_contract() {
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

        let system = make_system_lines_pumping_contracts_blocks(
            make_line_pumping_contract_block_overrides(),
        );
        let tmp = tempfile::tempdir().unwrap();
        let config = ParquetWriterConfig::default();

        write_bounds_parquet(tmp.path(), &system, &config)
            .expect("write_bounds_parquet must succeed");

        let file = std::fs::File::open(tmp.path().join("bounds.parquet")).unwrap();
        let builder = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
        let mut reader = builder.build().unwrap();
        let batch = reader.next().expect("must have rows").expect("batch Ok");

        let entity_type_col = batch
            .column_by_name("entity_type_code")
            .unwrap()
            .as_any()
            .downcast_ref::<arrow::array::Int8Array>()
            .unwrap();
        let entity_id_col = batch
            .column_by_name("entity_id")
            .unwrap()
            .as_any()
            .downcast_ref::<arrow::array::Int32Array>()
            .unwrap();
        let stage_id_col = batch
            .column_by_name("stage_id")
            .unwrap()
            .as_any()
            .downcast_ref::<arrow::array::Int32Array>()
            .unwrap();
        let block_id_col = batch
            .column_by_name("block_id")
            .unwrap()
            .as_any()
            .downcast_ref::<arrow::array::Int32Array>()
            .unwrap();
        let bound_type_col = batch
            .column_by_name("bound_type_code")
            .unwrap()
            .as_any()
            .downcast_ref::<arrow::array::Int8Array>()
            .unwrap();
        let bound_value_col = batch
            .column_by_name("bound_value")
            .unwrap()
            .as_any()
            .downcast_ref::<arrow::array::Float64Array>()
            .unwrap();

        let block_rows: Vec<usize> = (0..batch.num_rows())
            .filter(|&i| !block_id_col.is_null(i))
            .collect();
        assert_eq!(
            block_rows.len(),
            4,
            "expected exactly one line row, one pumping row, and two contract rows"
        );

        let line_id = system.lines()[1].id.0;
        let pumping_id = system.pumping_stations()[0].id.0;
        let contract_id = system.contracts()[1].id.0;
        let stage0_id = system.stages()[0].id;
        let stage1_id = system.stages()[1].id;

        let line_row = block_rows
            .iter()
            .copied()
            .find(|&i| entity_type_col.value(i) == OUTPUT_ENTITY_LINE)
            .expect("must have a line block row");
        assert_eq!(entity_id_col.value(line_row), line_id, "line entity_id");
        assert_eq!(stage_id_col.value(line_row), stage0_id, "line stage_id");
        assert_eq!(block_id_col.value(line_row), 2, "line block_id");
        assert_eq!(
            bound_type_col.value(line_row),
            BOUND_FLOW_MAX,
            "line bound_type_code"
        );
        assert_eq!(
            bound_value_col.value(line_row).to_bits(),
            123.0_f64.to_bits(),
            "line bound_value"
        );

        let pumping_row = block_rows
            .iter()
            .copied()
            .find(|&i| entity_type_col.value(i) == OUTPUT_ENTITY_PUMPING_STATION)
            .expect("must have a pumping block row");
        assert_eq!(
            entity_id_col.value(pumping_row),
            pumping_id,
            "pumping entity_id"
        );
        assert_eq!(
            stage_id_col.value(pumping_row),
            stage1_id,
            "pumping stage_id"
        );
        assert_eq!(block_id_col.value(pumping_row), 0, "pumping block_id");
        assert_eq!(
            bound_type_col.value(pumping_row),
            BOUND_FLOW_MIN,
            "pumping bound_type_code"
        );
        assert_eq!(
            bound_value_col.value(pumping_row).to_bits(),
            11.0_f64.to_bits(),
            "pumping bound_value"
        );

        let contract_rows: Vec<usize> = block_rows
            .iter()
            .copied()
            .filter(|&i| entity_type_col.value(i) == OUTPUT_ENTITY_CONTRACT)
            .collect();
        assert_eq!(
            contract_rows.len(),
            2,
            "expected exactly two contract block rows (min_mw and max_mw)"
        );
        for &i in &contract_rows {
            assert_eq!(entity_id_col.value(i), contract_id, "contract entity_id");
            assert_eq!(stage_id_col.value(i), stage0_id, "contract stage_id");
            assert_eq!(block_id_col.value(i), 1, "contract block_id");
        }
        let contract_min_row = contract_rows
            .iter()
            .copied()
            .find(|&i| bound_type_col.value(i) == BOUND_FLOW_MIN)
            .expect("must have a contract min_mw row");
        assert_eq!(
            bound_value_col.value(contract_min_row).to_bits(),
            5.0_f64.to_bits(),
            "contract min_mw value"
        );
        let contract_max_row = contract_rows
            .iter()
            .copied()
            .find(|&i| bound_type_col.value(i) == BOUND_FLOW_MAX)
            .expect("must have a contract max_mw row");
        assert_eq!(
            bound_value_col.value(contract_max_row).to_bits(),
            50.0_f64.to_bits(),
            "contract max_mw value"
        );
    }

    /// One hydro (id 7) declaring two unit groups with distinct declared
    /// bounds -- group id 2 (`GroupLow`, bus 2) and group id 5 (`GroupHigh`,
    /// bus 3) -- and `overlay` as the group-axis resolved override table.
    /// `SystemBuilder::build` canonically id-sorts `unit_groups`, so group id
    /// 2 always resolves to `group_pos` 0 and group id 5 to `group_pos` 1
    /// regardless of declaration order here.
    fn make_system_1h_2groups(overlay: ResolvedHydroUnitGroupBounds) -> System {
        let bus2 = make_bus(2);
        let bus3 = make_bus(3);
        let mut h1 = make_hydro(7, "Plant7", 2);
        h1.unit_groups = vec![
            HydroUnitGroup {
                id: EntityId(5),
                name: "GroupHigh".to_string(),
                bus_id: EntityId(3),
                min_generation_mw: 1.0,
                max_generation_mw: 20.0,
                min_turbined_m3s: 2.0,
                max_turbined_m3s: 25.0,
            },
            HydroUnitGroup {
                id: EntityId(2),
                name: "GroupLow".to_string(),
                bus_id: EntityId(2),
                min_generation_mw: 3.0,
                max_generation_mw: 15.0,
                min_turbined_m3s: 4.0,
                max_turbined_m3s: 18.0,
            },
        ];
        let stage0 = make_stage(0);

        let mut bounds = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 1,
                n_thermals: 0,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages: 1,
                k_max: 0,
            },
            &BoundsDefaults {
                hydro: hydro_stage_bounds(0.0, 100.0),
                hydro_block: hydro_block_bounds_default(),
                thermal: ThermalStageBounds { cost_per_mwh: 0.0 },
                thermal_block: ThermalBlockBounds {
                    min_generation_mw: 0.0,
                    max_generation_mw: 100.0,
                },
                line_block: LineBlockBounds {
                    direct_mw: 500.0,
                    reverse_mw: 500.0,
                },
                pumping_block: PumpingBlockBounds {
                    min_flow_m3s: 0.0,
                    max_flow_m3s: 0.0,
                },
                contract_block: ContractBlockBounds {
                    min_mw: 0.0,
                    max_mw: 0.0,
                    price_per_mwh: 0.0,
                },
            },
        );
        bounds.set_group_overlay(overlay);

        SystemBuilder::new()
            .buses(vec![bus2, bus3])
            .hydros(vec![h1])
            .stages(vec![stage0])
            .bounds(bounds)
            .build()
            .expect("valid system")
    }

    /// Reads `bounds.parquet` and returns the columns this section's group
    /// tests need, downcast once.
    fn read_bounds_parquet_columns(
        path: &std::path::Path,
    ) -> (
        arrow::array::RecordBatch,
        arrow::array::Int8Array,
        arrow::array::Int32Array,
    ) {
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

        let file = std::fs::File::open(path).unwrap();
        let builder = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
        let mut reader = builder.build().unwrap();
        let batch = reader.next().expect("must have rows").expect("batch Ok");

        let entity_type_col = batch
            .column_by_name("entity_type_code")
            .unwrap()
            .as_any()
            .downcast_ref::<arrow::array::Int8Array>()
            .unwrap()
            .clone();
        let entity_id_col = batch
            .column_by_name("entity_id")
            .unwrap()
            .as_any()
            .downcast_ref::<arrow::array::Int32Array>()
            .unwrap()
            .clone();
        (batch, entity_type_col, entity_id_col)
    }

    #[test]
    fn bounds_parquet_group_rows_report_resolved_values_with_hydro_id() {
        let system = make_system_1h_2groups(ResolvedHydroUnitGroupBounds::empty());
        let tmp = tempfile::tempdir().unwrap();
        let config = ParquetWriterConfig::default();

        write_bounds_parquet(tmp.path(), &system, &config)
            .expect("write_bounds_parquet must succeed");

        let (batch, entity_type_col, entity_id_col) =
            read_bounds_parquet_columns(&tmp.path().join("bounds.parquet"));
        let hydro_id_col = batch
            .column_by_name("hydro_id")
            .unwrap()
            .as_any()
            .downcast_ref::<arrow::array::Int32Array>()
            .unwrap();
        let bound_type_col = batch
            .column_by_name("bound_type_code")
            .unwrap()
            .as_any()
            .downcast_ref::<arrow::array::Int8Array>()
            .unwrap();
        let bound_value_col = batch
            .column_by_name("bound_value")
            .unwrap()
            .as_any()
            .downcast_ref::<arrow::array::Float64Array>()
            .unwrap();

        let plant_id = system.hydros()[0].id.0;

        for row in 0..batch.num_rows() {
            if entity_type_col.value(row) == OUTPUT_ENTITY_HYDRO_UNIT_GROUP {
                assert!(
                    !hydro_id_col.is_null(row),
                    "group row {row} must carry a non-null hydro_id"
                );
                assert_eq!(
                    hydro_id_col.value(row),
                    plant_id,
                    "group row {row}'s hydro_id must be the owning plant's id"
                );
            } else {
                assert!(
                    hydro_id_col.is_null(row),
                    "plant-family row {row} (entity_type_code {}) must carry a null hydro_id",
                    entity_type_col.value(row)
                );
            }
        }

        let group_rows: Vec<usize> = (0..batch.num_rows())
            .filter(|&i| entity_type_col.value(i) == OUTPUT_ENTITY_HYDRO_UNIT_GROUP)
            .collect();
        assert_eq!(
            group_rows.len(),
            8,
            "2 groups x 1 stage x 4 group bound types = 8 rows"
        );

        let bound_of = |group_id: i32, bound_type: i8| -> f64 {
            let row = group_rows
                .iter()
                .copied()
                .find(|&i| {
                    entity_id_col.value(i) == group_id && bound_type_col.value(i) == bound_type
                })
                .unwrap_or_else(|| panic!("missing group {group_id} bound_type {bound_type} row"));
            bound_value_col.value(row)
        };

        // Group id 2 ("GroupLow") -- no override, so every value is declared.
        assert_eq!(bound_of(2, BOUND_TURBINED_MIN).to_bits(), 4.0_f64.to_bits());
        assert_eq!(
            bound_of(2, BOUND_TURBINED_MAX).to_bits(),
            18.0_f64.to_bits()
        );
        assert_eq!(
            bound_of(2, BOUND_GENERATION_MIN).to_bits(),
            3.0_f64.to_bits()
        );
        assert_eq!(
            bound_of(2, BOUND_GENERATION_MAX).to_bits(),
            15.0_f64.to_bits()
        );

        // Group id 5 ("GroupHigh") -- its own declared values, never group 2's
        // or the plant's.
        assert_eq!(bound_of(5, BOUND_TURBINED_MIN).to_bits(), 2.0_f64.to_bits());
        assert_eq!(
            bound_of(5, BOUND_TURBINED_MAX).to_bits(),
            25.0_f64.to_bits()
        );
        assert_eq!(
            bound_of(5, BOUND_GENERATION_MIN).to_bits(),
            1.0_f64.to_bits()
        );
        assert_eq!(
            bound_of(5, BOUND_GENERATION_MAX).to_bits(),
            20.0_f64.to_bits()
        );
    }

    #[test]
    fn bounds_parquet_group_override_row_reports_resolved_not_declared_value() {
        let mut overlay = ResolvedHydroUnitGroupBounds::new(&HydroUnitGroupBoundsCountsSpec {
            groups_per_plant: &[2],
            n_stages: 1,
            max_blocks: 1,
        });
        // group_pos 0 is group id 2 (the lower id, canonical-sorted first);
        // override only its max_generation_mw.
        overlay
            .stage_override_mut(0, 0, 0)
            .expect("cell must exist for a fixture-sized overlay")
            .max_generation_mw = Some(12.0);

        let system = make_system_1h_2groups(overlay);
        let tmp = tempfile::tempdir().unwrap();
        let config = ParquetWriterConfig::default();

        write_bounds_parquet(tmp.path(), &system, &config)
            .expect("write_bounds_parquet must succeed");

        let (batch, entity_type_col, entity_id_col) =
            read_bounds_parquet_columns(&tmp.path().join("bounds.parquet"));
        let bound_type_col = batch
            .column_by_name("bound_type_code")
            .unwrap()
            .as_any()
            .downcast_ref::<arrow::array::Int8Array>()
            .unwrap();
        let bound_value_col = batch
            .column_by_name("bound_value")
            .unwrap()
            .as_any()
            .downcast_ref::<arrow::array::Float64Array>()
            .unwrap();
        let block_id_col = batch.column_by_name("block_id").unwrap();

        let group_rows: Vec<usize> = (0..batch.num_rows())
            .filter(|&i| entity_type_col.value(i) == OUTPUT_ENTITY_HYDRO_UNIT_GROUP)
            .collect();
        assert_eq!(
            group_rows.len(),
            8,
            "the override is stage-wide, so it still reports as a null-block_id row, \
             not an additional block row -- still 8 group rows total"
        );

        let bound_of = |group_id: i32, bound_type: i8| -> f64 {
            let row = group_rows
                .iter()
                .copied()
                .find(|&i| {
                    entity_id_col.value(i) == group_id && bound_type_col.value(i) == bound_type
                })
                .unwrap_or_else(|| panic!("missing group {group_id} bound_type {bound_type} row"));
            assert!(
                block_id_col.is_null(row),
                "a stage-wide group override must still report at the null-block_id row"
            );
            bound_value_col.value(row)
        };

        // Group 2's overridden column reports the OVERRIDE, not the declared 15.0.
        assert_eq!(
            bound_of(2, BOUND_GENERATION_MAX).to_bits(),
            12.0_f64.to_bits(),
            "group 2's max_generation_mw must report the resolved override, not its \
             declared value"
        );
        // Group 2's other three columns are untouched by the override (column
        // independence).
        assert_eq!(bound_of(2, BOUND_TURBINED_MIN).to_bits(), 4.0_f64.to_bits());
        assert_eq!(
            bound_of(2, BOUND_TURBINED_MAX).to_bits(),
            18.0_f64.to_bits()
        );
        assert_eq!(
            bound_of(2, BOUND_GENERATION_MIN).to_bits(),
            3.0_f64.to_bits()
        );
        // Group 5 (a sibling group, never touched by group 2's override) keeps
        // every declared value.
        assert_eq!(bound_of(5, BOUND_TURBINED_MIN).to_bits(), 2.0_f64.to_bits());
        assert_eq!(
            bound_of(5, BOUND_TURBINED_MAX).to_bits(),
            25.0_f64.to_bits()
        );
        assert_eq!(
            bound_of(5, BOUND_GENERATION_MIN).to_bits(),
            1.0_f64.to_bits()
        );
        assert_eq!(
            bound_of(5, BOUND_GENERATION_MAX).to_bits(),
            20.0_f64.to_bits()
        );
    }

    #[test]
    fn bounds_hydro_id_column_has_a_description() {
        assert!(
            !description_for("bounds", "hydro_id").is_empty(),
            "hydro_id must have a description"
        );
    }

    // ── dictionary unit/description tests ────────────────────────────────────

    #[test]
    fn new_energy_columns_have_units() {
        assert_eq!(
            unit_for("hydros", "equivalent_productivity_mw_per_m3s"),
            "MW/(m3/s)"
        );
        assert_eq!(
            unit_for("hydros", "accumulated_productivity_mw_per_m3s"),
            "MW/(m3/s)"
        );
        assert_eq!(unit_for("hydros", "incremental_inflow_energy_mw"), "MW");
        assert_eq!(unit_for("hydros", "stored_energy_initial_mwh"), "MWh");
        assert_eq!(unit_for("hydros", "stored_energy_final_mwh"), "MWh");
    }

    #[test]
    fn new_energy_columns_have_descriptions() {
        assert!(
            !description_for("hydros", "equivalent_productivity_mw_per_m3s").is_empty(),
            "equivalent_productivity_mw_per_m3s must have a description"
        );
        assert!(
            !description_for("hydros", "accumulated_productivity_mw_per_m3s").is_empty(),
            "accumulated_productivity_mw_per_m3s must have a description"
        );
        assert!(
            !description_for("hydros", "incremental_inflow_energy_mw").is_empty(),
            "incremental_inflow_energy_mw must have a description"
        );
        assert!(
            !description_for("hydros", "stored_energy_initial_mwh").is_empty(),
            "stored_energy_initial_mwh must have a description"
        );
        assert!(
            !description_for("hydros", "stored_energy_final_mwh").is_empty(),
            "stored_energy_final_mwh must have a description"
        );
    }

    #[test]
    fn old_productivity_field_returns_default_unit() {
        assert_eq!(
            unit_for("hydros", "productivity_mw_per_m3s"),
            "",
            "removed column must fall through to the default empty-string arm"
        );
    }

    #[test]
    fn every_hydros_schema_column_has_description() {
        let schema = hydros_schema();
        for field in schema.fields() {
            let desc = description_for("hydros", field.name());
            assert!(
                !desc.is_empty(),
                "hydros column '{}' has no description in description_for",
                field.name()
            );
        }
    }

    #[test]
    fn every_hydro_bus_generation_schema_column_has_description() {
        let schema = hydro_bus_generation_schema();
        for field in schema.fields() {
            let desc = description_for("hydro_bus_generation", field.name());
            assert!(
                !desc.is_empty(),
                "hydro_bus_generation column '{}' has no description in description_for",
                field.name()
            );
        }
    }
}
