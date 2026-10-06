//! Parquet parsers for entity penalty override files in the `constraints/` subdirectory.
//!
//! Each parser reads a sparse Parquet file containing stage-varying penalty cost overrides
//! for a specific entity type. Sparse storage means only `(entity_id, stage_id)` pairs
//! that differ from entity-level or global defaults need rows.
//!
//! ## Parquet schemas
//!
//! ### `penalty_overrides_bus`
//!
//! | Column         | Type   | Required | Description                       |
//! | -------------- | ------ | -------- | --------------------------------- |
//! | `bus_id`       | INT32  | Yes      | Bus ID                            |
//! | `stage_id`     | INT32  | Yes      | Stage ID                          |
//! | `excess_cost`  | DOUBLE | No       | Excess injection cost (USD/`MWh`) |
//!
//! Note: Bus deficit segments are NOT stage-varying (Penalty System spec SS3).
//! Only `excess_cost` is overrideable per stage for buses.
//!
//! ### `penalty_overrides_line`
//!
//! | Column          | Type   | Required | Description                       |
//! | --------------- | ------ | -------- | --------------------------------- |
//! | `line_id`       | INT32  | Yes      | Transmission line ID              |
//! | `stage_id`      | INT32  | Yes      | Stage ID                          |
//! | `exchange_cost` | DOUBLE | No       | Exchange flow cost (USD/`MWh`)    |
//!
//! ### `penalty_overrides_hydro`
//!
//! | Column                            | Type   | Required | Description                          |
//! | --------------------------------- | ------ | -------- | ------------------------------------ |
//! | `hydro_id`                        | INT32  | Yes      | Hydro plant ID                       |
//! | `stage_id`                        | INT32  | Yes      | Stage ID                             |
//! | `spillage_cost`                   | DOUBLE | No       | Spillage penalty (USD/(m³/s·h))          |
//! | `turbined_cost`                   | DOUBLE | No       | Turbined regularization cost (USD/(m³/s·h)) |
//! | `diversion_cost`                  | DOUBLE | No       | Diversion penalty (USD/(m³/s·h))         |
//! | `storage_violation_below_cost`    | DOUBLE | No       | Storage below-min violation (USD/hm³)|
//! | `filling_target_violation_cost`   | DOUBLE | No       | Filling target violation (USD/hm³)   |
//! | `turbined_violation_below_cost`   | DOUBLE | No       | Turbined below-min violation (USD/(m³/s·h))|
//! | `outflow_violation_below_cost`    | DOUBLE | No       | Outflow below-min violation (USD/(m³/s·h))|
//! | `outflow_violation_above_cost`    | DOUBLE | No       | Outflow above-max violation (USD/(m³/s·h))|
//! | `generation_violation_below_cost` | DOUBLE | No       | Generation below-min violation (USD/`MWh`)|
//! | `evaporation_violation_cost`                 | DOUBLE | No       | Evaporation violation (USD/(m³/s·h))             |
//! | `water_withdrawal_violation_cost`            | DOUBLE | No       | Water withdrawal violation (USD/(m³/s·h))        |
//! | `water_withdrawal_violation_pos_cost`        | DOUBLE | No       | Over-withdrawal violation (USD/(m³/s·h))         |
//! | `water_withdrawal_violation_neg_cost`        | DOUBLE | No       | Under-withdrawal violation (USD/(m³/s·h))        |
//! | `evaporation_violation_pos_cost`             | DOUBLE | No       | Over-evaporation violation (USD/(m³/s·h))        |
//! | `evaporation_violation_neg_cost`             | DOUBLE | No       | Under-evaporation violation (USD/(m³/s·h))       |
//! | `inflow_nonnegativity_cost`                  | DOUBLE | No       | Inflow non-negativity cost (USD/(m³/s·h))        |
//!
//! ### `penalty_overrides_ncs`
//!
//! | Column              | Type   | Required | Description                                       |
//! | ------------------- | ------ | -------- | ------------------------------------------------- |
//! | `ncs_id`            | INT32  | Yes      | Non-controllable source ID                        |
//! | `stage_id`          | INT32  | Yes      | Stage ID                                          |
//! | `curtailment_cost`  | DOUBLE | No       | Curtailment penalty (USD/`MWh`)                   |
//!
//! ## Output ordering
//!
//! All parsers return rows sorted by `(entity_id, stage_id)` ascending.
//!
//! ## Validation
//!
//! Per-row constraints enforced by these parsers:
//!
//! - Required key columns (`*_id`, `stage_id`) must be present with Int32 type.
//! - Any provided (non-null) optional Float64 value must be finite — NaN and ±Inf are rejected.
//! - Any provided (non-null) optional Float64 value must be > 0.0 — penalties must be positive.
//!
//! Deferred validations (not performed here):
//!
//! - Entity ID existence in registries — Layer 3.
//! - Duplicate `(entity_id, stage_id)` pairs — deferred.
//! - Semantic cross-validation (e.g., penalty ordering constraints) — deferred.

use arrow::array::{Array, Float64Array};
use cobre_core::EntityId;
use std::path::Path;

use crate::LoadError;
use crate::parquet_helpers::{
    extract_optional_float64, extract_required_int32, open_record_batch_reader,
};

fn optional_value(col: Option<&Float64Array>, row: usize) -> Option<f64> {
    col.filter(|c| !c.is_null(row)).map(|c| c.value(row))
}

// ── Row types ─────────────────────────────────────────────────────────────────

/// A single row from `constraints/penalty_overrides_bus.parquet`.
///
/// Carries a stage-varying penalty cost override for a bus. The `excess_cost`
/// field is `None` when the column is absent or null in the Parquet file
/// (sparse storage: absent means "use entity-level or global default").
///
/// Note: Bus deficit segment costs are NOT stage-varying per the Penalty System
/// spec (SS3). Only `excess_cost` is overrideable per stage.
///
/// # Examples
///
/// ```
/// use cobre_io::constraints::BusPenaltyOverrideRow;
/// use cobre_core::EntityId;
///
/// let row = BusPenaltyOverrideRow {
///     bus_id: EntityId::from(1),
///     stage_id: 3,
///     excess_cost: Some(500.0),
/// };
/// assert_eq!(row.bus_id, EntityId::from(1));
/// assert_eq!(row.excess_cost, Some(500.0));
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct BusPenaltyOverrideRow {
    /// Bus ID.
    pub bus_id: EntityId,
    /// Stage ID.
    pub stage_id: i32,
    /// Excess cost override.
    pub excess_cost: Option<f64>,
}

/// A single row from `constraints/penalty_overrides_line.parquet`.
///
/// Carries a stage-varying penalty cost override for a transmission line.
/// The `exchange_cost` field is `None` when the column is absent or null
/// (sparse storage: absent means "use entity-level or global default").
///
/// # Examples
///
/// ```
/// use cobre_io::constraints::LinePenaltyOverrideRow;
/// use cobre_core::EntityId;
///
/// let row = LinePenaltyOverrideRow {
///     line_id: EntityId::from(10),
///     stage_id: 0,
///     exchange_cost: Some(1000.0),
/// };
/// assert_eq!(row.line_id, EntityId::from(10));
/// assert_eq!(row.exchange_cost, Some(1000.0));
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct LinePenaltyOverrideRow {
    /// Line ID.
    pub line_id: EntityId,
    /// Stage ID.
    pub stage_id: i32,
    /// Exchange cost override.
    pub exchange_cost: Option<f64>,
}

/// A single row from `constraints/penalty_overrides_hydro.parquet`.
///
/// Carries stage-varying penalty cost overrides for a hydro plant. All sixteen
/// penalty columns are optional. An absent or null directional evaporation or
/// withdrawal column takes the row's symmetric cost when the row sets it (see
/// [`resolve_penalties`](crate::resolution::resolve_penalties)); any other absent or
/// null column means "use entity-level or global default".
///
/// # Examples
///
/// ```
/// use cobre_io::constraints::HydroPenaltyOverrideRow;
/// use cobre_core::EntityId;
///
/// let row = HydroPenaltyOverrideRow {
///     hydro_id: EntityId::from(5),
///     stage_id: 2,
///     spillage_cost: Some(0.01),
///     turbined_cost: None,
///     diversion_cost: None,
///     storage_violation_below_cost: Some(9999.0),
///     filling_target_violation_cost: None,
///     turbined_violation_below_cost: None,
///     outflow_violation_below_cost: None,
///     outflow_violation_above_cost: None,
///     generation_violation_below_cost: None,
///     evaporation_violation_cost: None,
///     water_withdrawal_violation_cost: None,
///     water_withdrawal_violation_pos_cost: None,
///     water_withdrawal_violation_neg_cost: None,
///     evaporation_violation_pos_cost: None,
///     evaporation_violation_neg_cost: None,
///     inflow_nonnegativity_cost: None,
/// };
/// assert_eq!(row.hydro_id, EntityId::from(5));
/// assert_eq!(row.spillage_cost, Some(0.01));
/// assert!(row.turbined_cost.is_none());
/// ```
#[allow(clippy::struct_field_names)]
#[derive(Debug, Clone, PartialEq)]
pub struct HydroPenaltyOverrideRow {
    /// Hydro plant ID.
    pub hydro_id: EntityId,
    /// Stage ID.
    pub stage_id: i32,
    /// Spillage penalty override.
    pub spillage_cost: Option<f64>,
    /// Turbined regularization cost override.
    pub turbined_cost: Option<f64>,
    /// Diversion penalty override.
    pub diversion_cost: Option<f64>,
    /// Storage violation override.
    pub storage_violation_below_cost: Option<f64>,
    /// Filling target violation override.
    pub filling_target_violation_cost: Option<f64>,
    /// Turbined violation override.
    pub turbined_violation_below_cost: Option<f64>,
    /// Outflow violation override.
    pub outflow_violation_below_cost: Option<f64>,
    /// Outflow above violation override.
    pub outflow_violation_above_cost: Option<f64>,
    /// Generation violation override.
    pub generation_violation_below_cost: Option<f64>,
    /// Evaporation violation override.
    pub evaporation_violation_cost: Option<f64>,
    /// Water withdrawal violation override.
    pub water_withdrawal_violation_cost: Option<f64>,
    /// Over-withdrawal violation override.
    pub water_withdrawal_violation_pos_cost: Option<f64>,
    /// Under-withdrawal violation override.
    pub water_withdrawal_violation_neg_cost: Option<f64>,
    /// Over-evaporation violation override.
    pub evaporation_violation_pos_cost: Option<f64>,
    /// Under-evaporation violation override.
    pub evaporation_violation_neg_cost: Option<f64>,
    /// Inflow non-negativity cost override.
    pub inflow_nonnegativity_cost: Option<f64>,
}

/// A single row from `constraints/penalty_overrides_ncs.parquet`.
///
/// Carries a stage-varying penalty cost override for a non-controllable source.
/// The `curtailment_cost` field is `None` when the column is absent or null
/// (sparse storage: absent means "use entity-level or global default").
///
/// # Examples
///
/// ```
/// use cobre_io::constraints::NcsPenaltyOverrideRow;
/// use cobre_core::EntityId;
///
/// let row = NcsPenaltyOverrideRow {
///     source_id: EntityId::from(3),
///     stage_id: 0,
///     curtailment_cost: Some(250.0),
/// };
/// assert_eq!(row.source_id, EntityId::from(3));
/// assert_eq!(row.curtailment_cost, Some(250.0));
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct NcsPenaltyOverrideRow {
    /// Source ID.
    pub source_id: EntityId,
    /// Stage ID.
    pub stage_id: i32,
    /// Curtailment cost override.
    pub curtailment_cost: Option<f64>,
}

// ── Validation helpers ────────────────────────────────────────────────────────

/// Validate that a present (non-null) optional float value is finite and positive.
///
/// Penalties are strictly positive: a value of exactly 0.0 is rejected (`<= 0.0`, not
/// `< 0.0`).
fn validate_optional_positive(
    value: Option<f64>,
    file_label: &str,
    row_idx: usize,
    column: &str,
    path: &Path,
) -> Result<(), LoadError> {
    if let Some(v) = value {
        if !v.is_finite() {
            return Err(LoadError::SchemaError {
                path: path.to_path_buf(),
                field: format!("{file_label}[{row_idx}].{column}"),
                message: format!("value must be finite, got {v}"),
            });
        }
        if v <= 0.0 {
            return Err(LoadError::SchemaError {
                path: path.to_path_buf(),
                field: format!("{file_label}[{row_idx}].{column}"),
                message: format!("value must be > 0.0, got {v}"),
            });
        }
    }
    Ok(())
}

// ── Parsers ───────────────────────────────────────────────────────────────────

/// Parse `constraints/penalty_overrides_bus.parquet`, returning rows sorted by
/// `(bus_id, stage_id)` ascending.
///
/// The `excess_cost` column is optional; absent from the file schema yields
/// `excess_cost: None` in all rows (the sparse-override design, not a schema error).
///
/// # Errors
///
/// | Condition                                            | Error variant              |
/// |----------------------------------------------------- |--------------------------- |
/// | File not found or permission denied                  | [`LoadError::IoError`]     |
/// | Malformed Parquet (corrupt header, etc.)             | [`LoadError::ParseError`]  |
/// | Required column missing or wrong type                | [`LoadError::SchemaError`] |
/// | Non-finite value in any present penalty column       | [`LoadError::SchemaError`] |
/// | Value <= 0.0 in any present penalty column           | [`LoadError::SchemaError`] |
///
/// # Examples
///
/// ```no_run
/// use cobre_io::constraints::parse_penalty_overrides_bus;
/// use std::path::Path;
///
/// let rows = parse_penalty_overrides_bus(Path::new("constraints/penalty_overrides_bus.parquet"))
///     .expect("valid bus penalty overrides file");
/// println!("loaded {} bus penalty override rows", rows.len());
/// ```
pub fn parse_penalty_overrides_bus(path: &Path) -> Result<Vec<BusPenaltyOverrideRow>, LoadError> {
    let reader = open_record_batch_reader(path)?;

    let mut rows: Vec<BusPenaltyOverrideRow> = Vec::new();

    for batch_result in reader {
        let batch = batch_result.map_err(|e| LoadError::parse(path, e.to_string()))?;

        let bus_id_col = extract_required_int32(&batch, "bus_id", path)?;
        let stage_id_col = extract_required_int32(&batch, "stage_id", path)?;

        let excess_cost_col = extract_optional_float64(&batch, "excess_cost", path)?;

        let n = batch.num_rows();
        let base_idx = rows.len();
        rows.reserve(n);

        for i in 0..n {
            let row_idx = base_idx + i;

            let bus_id = EntityId::from(bus_id_col.value(i));
            let stage_id = stage_id_col.value(i);

            let excess_cost = optional_value(excess_cost_col, i);

            validate_optional_positive(
                excess_cost,
                "penalty_overrides_bus",
                row_idx,
                "excess_cost",
                path,
            )?;

            rows.push(BusPenaltyOverrideRow {
                bus_id,
                stage_id,
                excess_cost,
            });
        }
    }

    rows.sort_by(|a, b| {
        a.bus_id
            .0
            .cmp(&b.bus_id.0)
            .then_with(|| a.stage_id.cmp(&b.stage_id))
    });

    Ok(rows)
}

/// Parse `constraints/penalty_overrides_line.parquet`, returning rows sorted by
/// `(line_id, stage_id)` ascending.
///
/// The `exchange_cost` column is optional; absent from the file schema yields
/// `exchange_cost: None` in all rows (the sparse-override design, not a schema error).
///
/// # Errors
///
/// | Condition                                            | Error variant              |
/// |----------------------------------------------------- |--------------------------- |
/// | File not found or permission denied                  | [`LoadError::IoError`]     |
/// | Malformed Parquet (corrupt header, etc.)             | [`LoadError::ParseError`]  |
/// | Required column missing or wrong type                | [`LoadError::SchemaError`] |
/// | Non-finite value in any present penalty column       | [`LoadError::SchemaError`] |
/// | Value <= 0.0 in any present penalty column           | [`LoadError::SchemaError`] |
///
/// # Examples
///
/// ```no_run
/// use cobre_io::constraints::parse_penalty_overrides_line;
/// use std::path::Path;
///
/// let rows = parse_penalty_overrides_line(Path::new("constraints/penalty_overrides_line.parquet"))
///     .expect("valid line penalty overrides file");
/// println!("loaded {} line penalty override rows", rows.len());
/// ```
pub fn parse_penalty_overrides_line(path: &Path) -> Result<Vec<LinePenaltyOverrideRow>, LoadError> {
    let reader = open_record_batch_reader(path)?;

    let mut rows: Vec<LinePenaltyOverrideRow> = Vec::new();

    for batch_result in reader {
        let batch = batch_result.map_err(|e| LoadError::parse(path, e.to_string()))?;

        let line_id_col = extract_required_int32(&batch, "line_id", path)?;
        let stage_id_col = extract_required_int32(&batch, "stage_id", path)?;

        let exchange_cost_col = extract_optional_float64(&batch, "exchange_cost", path)?;

        let n = batch.num_rows();
        let base_idx = rows.len();
        rows.reserve(n);

        for i in 0..n {
            let row_idx = base_idx + i;

            let line_id = EntityId::from(line_id_col.value(i));
            let stage_id = stage_id_col.value(i);

            let exchange_cost = optional_value(exchange_cost_col, i);

            validate_optional_positive(
                exchange_cost,
                "penalty_overrides_line",
                row_idx,
                "exchange_cost",
                path,
            )?;

            rows.push(LinePenaltyOverrideRow {
                line_id,
                stage_id,
                exchange_cost,
            });
        }
    }

    rows.sort_by(|a, b| {
        a.line_id
            .0
            .cmp(&b.line_id.0)
            .then_with(|| a.stage_id.cmp(&b.stage_id))
    });

    Ok(rows)
}

/// Parse `constraints/penalty_overrides_hydro.parquet`, returning rows sorted by
/// `(hydro_id, stage_id)` ascending.
///
/// The file may contain any subset of the sixteen optional penalty columns; columns
/// absent from the file schema produce `None` in all rows (the sparse-override design,
/// not a schema error).
///
/// # Errors
///
/// | Condition                                            | Error variant              |
/// |----------------------------------------------------- |--------------------------- |
/// | File not found or permission denied                  | [`LoadError::IoError`]     |
/// | Malformed Parquet (corrupt header, etc.)             | [`LoadError::ParseError`]  |
/// | Required column missing or wrong type                | [`LoadError::SchemaError`] |
/// | Non-finite value in any present penalty column       | [`LoadError::SchemaError`] |
/// | Value <= 0.0 in any present penalty column           | [`LoadError::SchemaError`] |
///
/// # Examples
///
/// ```no_run
/// use cobre_io::constraints::parse_penalty_overrides_hydro;
/// use std::path::Path;
///
/// let rows = parse_penalty_overrides_hydro(Path::new("constraints/penalty_overrides_hydro.parquet"))
///     .expect("valid hydro penalty overrides file");
/// println!("loaded {} hydro penalty override rows", rows.len());
/// ```
// Rationale: a single sequential pass validates every column for every row before returning;
// splitting would require multiple passes over the batch.
#[allow(clippy::too_many_lines)]
pub fn parse_penalty_overrides_hydro(
    path: &Path,
) -> Result<Vec<HydroPenaltyOverrideRow>, LoadError> {
    let reader = open_record_batch_reader(path)?;

    let mut rows: Vec<HydroPenaltyOverrideRow> = Vec::new();

    for batch_result in reader {
        let batch = batch_result.map_err(|e| LoadError::parse(path, e.to_string()))?;

        let hydro_id_col = extract_required_int32(&batch, "hydro_id", path)?;
        let stage_id_col = extract_required_int32(&batch, "stage_id", path)?;

        let spillage_cost_col = extract_optional_float64(&batch, "spillage_cost", path)?;
        let turbined_cost_col = extract_optional_float64(&batch, "turbined_cost", path)?;
        let diversion_cost_col = extract_optional_float64(&batch, "diversion_cost", path)?;
        let storage_violation_below_cost_col =
            extract_optional_float64(&batch, "storage_violation_below_cost", path)?;
        let filling_target_violation_cost_col =
            extract_optional_float64(&batch, "filling_target_violation_cost", path)?;
        let turbined_violation_below_cost_col =
            extract_optional_float64(&batch, "turbined_violation_below_cost", path)?;
        let outflow_violation_below_cost_col =
            extract_optional_float64(&batch, "outflow_violation_below_cost", path)?;
        let outflow_violation_above_cost_col =
            extract_optional_float64(&batch, "outflow_violation_above_cost", path)?;
        let generation_violation_below_cost_col =
            extract_optional_float64(&batch, "generation_violation_below_cost", path)?;
        let evaporation_violation_cost_col =
            extract_optional_float64(&batch, "evaporation_violation_cost", path)?;
        let water_withdrawal_violation_cost_col =
            extract_optional_float64(&batch, "water_withdrawal_violation_cost", path)?;
        let water_withdrawal_violation_pos_cost_col =
            extract_optional_float64(&batch, "water_withdrawal_violation_pos_cost", path)?;
        let water_withdrawal_violation_neg_cost_col =
            extract_optional_float64(&batch, "water_withdrawal_violation_neg_cost", path)?;
        let evaporation_violation_pos_cost_col =
            extract_optional_float64(&batch, "evaporation_violation_pos_cost", path)?;
        let evaporation_violation_neg_cost_col =
            extract_optional_float64(&batch, "evaporation_violation_neg_cost", path)?;
        let inflow_nonnegativity_cost_col =
            extract_optional_float64(&batch, "inflow_nonnegativity_cost", path)?;

        let n = batch.num_rows();
        let base_idx = rows.len();
        rows.reserve(n);

        for i in 0..n {
            let row_idx = base_idx + i;

            let hydro_id = EntityId::from(hydro_id_col.value(i));
            let stage_id = stage_id_col.value(i);

            let spillage_cost = optional_value(spillage_cost_col, i);
            let turbined_cost = optional_value(turbined_cost_col, i);
            let diversion_cost = optional_value(diversion_cost_col, i);
            let storage_violation_below_cost = optional_value(storage_violation_below_cost_col, i);
            let filling_target_violation_cost =
                optional_value(filling_target_violation_cost_col, i);
            let turbined_violation_below_cost =
                optional_value(turbined_violation_below_cost_col, i);
            let outflow_violation_below_cost = optional_value(outflow_violation_below_cost_col, i);
            let outflow_violation_above_cost = optional_value(outflow_violation_above_cost_col, i);
            let generation_violation_below_cost =
                optional_value(generation_violation_below_cost_col, i);
            let evaporation_violation_cost = optional_value(evaporation_violation_cost_col, i);
            let water_withdrawal_violation_cost =
                optional_value(water_withdrawal_violation_cost_col, i);
            let water_withdrawal_violation_pos_cost =
                optional_value(water_withdrawal_violation_pos_cost_col, i);
            let water_withdrawal_violation_neg_cost =
                optional_value(water_withdrawal_violation_neg_cost_col, i);
            let evaporation_violation_pos_cost =
                optional_value(evaporation_violation_pos_cost_col, i);
            let evaporation_violation_neg_cost =
                optional_value(evaporation_violation_neg_cost_col, i);
            let inflow_nonnegativity_cost = optional_value(inflow_nonnegativity_cost_col, i);

            validate_optional_positive(
                spillage_cost,
                "penalty_overrides_hydro",
                row_idx,
                "spillage_cost",
                path,
            )?;
            validate_optional_positive(
                turbined_cost,
                "penalty_overrides_hydro",
                row_idx,
                "turbined_cost",
                path,
            )?;
            validate_optional_positive(
                diversion_cost,
                "penalty_overrides_hydro",
                row_idx,
                "diversion_cost",
                path,
            )?;
            validate_optional_positive(
                storage_violation_below_cost,
                "penalty_overrides_hydro",
                row_idx,
                "storage_violation_below_cost",
                path,
            )?;
            validate_optional_positive(
                filling_target_violation_cost,
                "penalty_overrides_hydro",
                row_idx,
                "filling_target_violation_cost",
                path,
            )?;
            validate_optional_positive(
                turbined_violation_below_cost,
                "penalty_overrides_hydro",
                row_idx,
                "turbined_violation_below_cost",
                path,
            )?;
            validate_optional_positive(
                outflow_violation_below_cost,
                "penalty_overrides_hydro",
                row_idx,
                "outflow_violation_below_cost",
                path,
            )?;
            validate_optional_positive(
                outflow_violation_above_cost,
                "penalty_overrides_hydro",
                row_idx,
                "outflow_violation_above_cost",
                path,
            )?;
            validate_optional_positive(
                generation_violation_below_cost,
                "penalty_overrides_hydro",
                row_idx,
                "generation_violation_below_cost",
                path,
            )?;
            validate_optional_positive(
                evaporation_violation_cost,
                "penalty_overrides_hydro",
                row_idx,
                "evaporation_violation_cost",
                path,
            )?;
            validate_optional_positive(
                water_withdrawal_violation_cost,
                "penalty_overrides_hydro",
                row_idx,
                "water_withdrawal_violation_cost",
                path,
            )?;
            validate_optional_positive(
                water_withdrawal_violation_pos_cost,
                "penalty_overrides_hydro",
                row_idx,
                "water_withdrawal_violation_pos_cost",
                path,
            )?;
            validate_optional_positive(
                water_withdrawal_violation_neg_cost,
                "penalty_overrides_hydro",
                row_idx,
                "water_withdrawal_violation_neg_cost",
                path,
            )?;
            validate_optional_positive(
                evaporation_violation_pos_cost,
                "penalty_overrides_hydro",
                row_idx,
                "evaporation_violation_pos_cost",
                path,
            )?;
            validate_optional_positive(
                evaporation_violation_neg_cost,
                "penalty_overrides_hydro",
                row_idx,
                "evaporation_violation_neg_cost",
                path,
            )?;
            validate_optional_positive(
                inflow_nonnegativity_cost,
                "penalty_overrides_hydro",
                row_idx,
                "inflow_nonnegativity_cost",
                path,
            )?;

            rows.push(HydroPenaltyOverrideRow {
                hydro_id,
                stage_id,
                spillage_cost,
                turbined_cost,
                diversion_cost,
                storage_violation_below_cost,
                filling_target_violation_cost,
                turbined_violation_below_cost,
                outflow_violation_below_cost,
                outflow_violation_above_cost,
                generation_violation_below_cost,
                evaporation_violation_cost,
                water_withdrawal_violation_cost,
                water_withdrawal_violation_pos_cost,
                water_withdrawal_violation_neg_cost,
                evaporation_violation_pos_cost,
                evaporation_violation_neg_cost,
                inflow_nonnegativity_cost,
            });
        }
    }

    rows.sort_by(|a, b| {
        a.hydro_id
            .0
            .cmp(&b.hydro_id.0)
            .then_with(|| a.stage_id.cmp(&b.stage_id))
    });

    Ok(rows)
}

/// Parse `constraints/penalty_overrides_ncs.parquet`, returning rows sorted by
/// `(source_id, stage_id)` ascending.
///
/// The `curtailment_cost` column is optional; absent from the file schema yields
/// `curtailment_cost: None` in all rows (the sparse-override design, not a schema error).
///
/// # Errors
///
/// | Condition                                            | Error variant              |
/// |----------------------------------------------------- |--------------------------- |
/// | File not found or permission denied                  | [`LoadError::IoError`]     |
/// | Malformed Parquet (corrupt header, etc.)             | [`LoadError::ParseError`]  |
/// | Required column missing or wrong type                | [`LoadError::SchemaError`] |
/// | Non-finite value in any present penalty column       | [`LoadError::SchemaError`] |
/// | Value <= 0.0 in any present penalty column           | [`LoadError::SchemaError`] |
///
/// # Examples
///
/// ```no_run
/// use cobre_io::constraints::parse_penalty_overrides_ncs;
/// use std::path::Path;
///
/// let rows = parse_penalty_overrides_ncs(Path::new("constraints/penalty_overrides_ncs.parquet"))
///     .expect("valid NCS penalty overrides file");
/// println!("loaded {} NCS penalty override rows", rows.len());
/// ```
pub fn parse_penalty_overrides_ncs(path: &Path) -> Result<Vec<NcsPenaltyOverrideRow>, LoadError> {
    let reader = open_record_batch_reader(path)?;

    let mut rows: Vec<NcsPenaltyOverrideRow> = Vec::new();

    for batch_result in reader {
        let batch = batch_result.map_err(|e| LoadError::parse(path, e.to_string()))?;

        let source_id_col = extract_required_int32(&batch, "ncs_id", path)?;
        let stage_id_col = extract_required_int32(&batch, "stage_id", path)?;

        let curtailment_cost_col = extract_optional_float64(&batch, "curtailment_cost", path)?;

        let n = batch.num_rows();
        let base_idx = rows.len();
        rows.reserve(n);

        for i in 0..n {
            let row_idx = base_idx + i;

            let source_id = EntityId::from(source_id_col.value(i));
            let stage_id = stage_id_col.value(i);

            let curtailment_cost = optional_value(curtailment_cost_col, i);

            validate_optional_positive(
                curtailment_cost,
                "penalty_overrides_ncs",
                row_idx,
                "curtailment_cost",
                path,
            )?;

            rows.push(NcsPenaltyOverrideRow {
                source_id,
                stage_id,
                curtailment_cost,
            });
        }
    }

    rows.sort_by(|a, b| {
        a.source_id
            .0
            .cmp(&b.source_id.0)
            .then_with(|| a.stage_id.cmp(&b.stage_id))
    });

    Ok(rows)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(
    clippy::doc_markdown,
    clippy::expect_used,
    clippy::panic,
    clippy::too_many_lines,
    clippy::unwrap_used
)]
mod tests {
    use super::*;
    use crate::test_support::write_parquet;
    use arrow::array::{Float64Array, Int32Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use std::sync::Arc;

    // ── BusPenaltyOverrideRow tests ───────────────────────────────────────────

    fn bus_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("bus_id", DataType::Int32, false),
            Field::new("stage_id", DataType::Int32, false),
            Field::new("excess_cost", DataType::Float64, true),
        ]))
    }

    fn make_bus_batch(
        bus_ids: &[i32],
        stage_ids: &[i32],
        excess_cost: Vec<Option<f64>>,
    ) -> RecordBatch {
        RecordBatch::try_new(
            bus_schema(),
            vec![
                Arc::new(Int32Array::from(bus_ids.to_vec())),
                Arc::new(Int32Array::from(stage_ids.to_vec())),
                Arc::new(Float64Array::from(excess_cost)),
            ],
        )
        .expect("valid batch")
    }

    /// AC: 3 rows parsed correctly, sorted by (bus_id, stage_id).
    #[test]
    fn test_bus_valid_3_rows_sorted() {
        // Scrambled: (3,0), (1,1), (1,0) -> sorted: (1,0),(1,1),(3,0).
        let batch = make_bus_batch(
            &[3, 1, 1],
            &[0, 1, 0],
            vec![Some(500.0), Some(1000.0), None],
        );
        let tmp = write_parquet(&batch);
        let rows = parse_penalty_overrides_bus(tmp.path()).unwrap();

        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0].bus_id, EntityId::from(1));
        assert_eq!(rows[0].stage_id, 0);
        assert!(rows[0].excess_cost.is_none());
        assert_eq!(rows[1].bus_id, EntityId::from(1));
        assert_eq!(rows[1].stage_id, 1);
        assert!((rows[1].excess_cost.unwrap() - 1000.0).abs() < f64::EPSILON);
        assert_eq!(rows[2].bus_id, EntityId::from(3));
        assert_eq!(rows[2].stage_id, 0);
        assert!((rows[2].excess_cost.unwrap() - 500.0).abs() < f64::EPSILON);
    }

    /// AC: missing required `bus_id` column -> SchemaError with field "bus_id".
    #[test]
    fn test_bus_missing_bus_id() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("stage_id", DataType::Int32, false),
            Field::new("excess_cost", DataType::Float64, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![0_i32])),
                Arc::new(Float64Array::from(vec![Some(500.0_f64)])),
            ],
        )
        .unwrap();
        let tmp = write_parquet(&batch);
        let err = parse_penalty_overrides_bus(tmp.path()).unwrap_err();

        match &err {
            LoadError::SchemaError { field, .. } => {
                assert!(
                    field.contains("bus_id"),
                    "field should contain 'bus_id', got: {field}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// AC: negative penalty value -> SchemaError with "must be > 0.0".
    #[test]
    fn test_bus_negative_excess_cost() {
        let batch = make_bus_batch(&[1], &[0], vec![Some(-5.0)]);
        let tmp = write_parquet(&batch);
        let err = parse_penalty_overrides_bus(tmp.path()).unwrap_err();

        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("excess_cost"),
                    "field should contain 'excess_cost', got: {field}"
                );
                assert!(
                    message.contains("must be > 0.0"),
                    "message should contain 'must be > 0.0', got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// AC: NaN penalty value -> SchemaError mentioning "finite".
    #[test]
    fn test_bus_nan_excess_cost() {
        let batch = make_bus_batch(&[1], &[0], vec![Some(f64::NAN)]);
        let tmp = write_parquet(&batch);
        let err = parse_penalty_overrides_bus(tmp.path()).unwrap_err();

        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("excess_cost"),
                    "field should contain 'excess_cost', got: {field}"
                );
                assert!(
                    message.contains("finite"),
                    "message should contain 'finite', got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// AC: empty file (0 rows) -> Ok(Vec::new()).
    #[test]
    fn test_bus_empty_parquet() {
        let batch = make_bus_batch(&[], &[], vec![]);
        let tmp = write_parquet(&batch);
        let rows = parse_penalty_overrides_bus(tmp.path()).unwrap();
        assert!(rows.is_empty());
    }

    /// AC: load_penalty_overrides_bus(None) -> Ok(Vec::new()).
    #[test]
    fn test_load_bus_none() {
        let rows = super::super::load_penalty_overrides_bus(None).unwrap();
        assert!(rows.is_empty());
    }

    // ── LinePenaltyOverrideRow tests ──────────────────────────────────────────

    fn line_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("line_id", DataType::Int32, false),
            Field::new("stage_id", DataType::Int32, false),
            Field::new("exchange_cost", DataType::Float64, true),
        ]))
    }

    fn make_line_batch(
        line_ids: &[i32],
        stage_ids: &[i32],
        exchange_cost: Vec<Option<f64>>,
    ) -> RecordBatch {
        RecordBatch::try_new(
            line_schema(),
            vec![
                Arc::new(Int32Array::from(line_ids.to_vec())),
                Arc::new(Int32Array::from(stage_ids.to_vec())),
                Arc::new(Float64Array::from(exchange_cost)),
            ],
        )
        .expect("valid batch")
    }

    /// AC: 2 rows parsed correctly, sorted by (line_id, stage_id).
    #[test]
    fn test_line_valid_2_rows_sorted() {
        // Scrambled: (5,1),(5,0) -> sorted: (5,0),(5,1).
        let batch = make_line_batch(&[5, 5], &[1, 0], vec![Some(2000.0), Some(1500.0)]);
        let tmp = write_parquet(&batch);
        let rows = parse_penalty_overrides_line(tmp.path()).unwrap();

        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].line_id, EntityId::from(5));
        assert_eq!(rows[0].stage_id, 0);
        assert!((rows[0].exchange_cost.unwrap() - 1500.0).abs() < f64::EPSILON);
        assert_eq!(rows[1].line_id, EntityId::from(5));
        assert_eq!(rows[1].stage_id, 1);
        assert!((rows[1].exchange_cost.unwrap() - 2000.0).abs() < f64::EPSILON);
    }

    /// AC: missing required `line_id` column -> SchemaError with field "line_id".
    #[test]
    fn test_line_missing_line_id() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("stage_id", DataType::Int32, false),
            Field::new("exchange_cost", DataType::Float64, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![0_i32])),
                Arc::new(Float64Array::from(vec![Some(1000.0_f64)])),
            ],
        )
        .unwrap();
        let tmp = write_parquet(&batch);
        let err = parse_penalty_overrides_line(tmp.path()).unwrap_err();

        match &err {
            LoadError::SchemaError { field, .. } => {
                assert!(
                    field.contains("line_id"),
                    "field should contain 'line_id', got: {field}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// AC: negative penalty in exchange_cost -> SchemaError with "must be > 0.0".
    #[test]
    fn test_line_negative_exchange_cost() {
        let batch = make_line_batch(&[1], &[0], vec![Some(-5.0)]);
        let tmp = write_parquet(&batch);
        let err = parse_penalty_overrides_line(tmp.path()).unwrap_err();

        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("exchange_cost"),
                    "field should contain 'exchange_cost', got: {field}"
                );
                assert!(
                    message.contains("must be > 0.0"),
                    "message should contain 'must be > 0.0', got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// AC: NaN in exchange_cost -> SchemaError mentioning "finite".
    #[test]
    fn test_line_nan_exchange_cost() {
        let batch = make_line_batch(&[1], &[0], vec![Some(f64::NAN)]);
        let tmp = write_parquet(&batch);
        let err = parse_penalty_overrides_line(tmp.path()).unwrap_err();

        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("exchange_cost"),
                    "field should contain 'exchange_cost', got: {field}"
                );
                assert!(
                    message.contains("finite"),
                    "message should contain 'finite', got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// AC: empty file (0 rows) -> Ok(Vec::new()).
    #[test]
    fn test_line_empty_parquet() {
        let batch = make_line_batch(&[], &[], vec![]);
        let tmp = write_parquet(&batch);
        let rows = parse_penalty_overrides_line(tmp.path()).unwrap();
        assert!(rows.is_empty());
    }

    /// AC: load_penalty_overrides_line(None) -> Ok(Vec::new()).
    #[test]
    fn test_load_line_none() {
        let rows = super::super::load_penalty_overrides_line(None).unwrap();
        assert!(rows.is_empty());
    }

    // ── HydroPenaltyOverrideRow tests ─────────────────────────────────────────

    fn hydro_full_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("hydro_id", DataType::Int32, false),
            Field::new("stage_id", DataType::Int32, false),
            Field::new("spillage_cost", DataType::Float64, true),
            Field::new("turbined_cost", DataType::Float64, true),
            Field::new("diversion_cost", DataType::Float64, true),
            Field::new("storage_violation_below_cost", DataType::Float64, true),
            Field::new("filling_target_violation_cost", DataType::Float64, true),
            Field::new("turbined_violation_below_cost", DataType::Float64, true),
            Field::new("outflow_violation_below_cost", DataType::Float64, true),
            Field::new("outflow_violation_above_cost", DataType::Float64, true),
            Field::new("generation_violation_below_cost", DataType::Float64, true),
            Field::new("evaporation_violation_cost", DataType::Float64, true),
            Field::new("water_withdrawal_violation_cost", DataType::Float64, true),
        ]))
    }

    /// AC: 2 rows for 1 hydro at 2 stages, only spillage_cost and
    /// storage_violation_below_cost have non-null values; other 9 fields are None.
    #[test]
    fn test_hydro_two_columns_non_null_rest_none() {
        let batch = RecordBatch::try_new(
            hydro_full_schema(),
            vec![
                Arc::new(Int32Array::from(vec![7_i32, 7_i32])),
                Arc::new(Int32Array::from(vec![0_i32, 1_i32])),
                Arc::new(Float64Array::from(vec![Some(0.01_f64), Some(0.02_f64)])),
                Arc::new(Float64Array::from(vec![None::<f64>, None::<f64>])),
                Arc::new(Float64Array::from(vec![None::<f64>, None::<f64>])),
                Arc::new(Float64Array::from(vec![Some(9999.0_f64), Some(8888.0_f64)])),
                Arc::new(Float64Array::from(vec![None::<f64>, None::<f64>])),
                Arc::new(Float64Array::from(vec![None::<f64>, None::<f64>])),
                Arc::new(Float64Array::from(vec![None::<f64>, None::<f64>])),
                Arc::new(Float64Array::from(vec![None::<f64>, None::<f64>])),
                Arc::new(Float64Array::from(vec![None::<f64>, None::<f64>])),
                Arc::new(Float64Array::from(vec![None::<f64>, None::<f64>])),
                Arc::new(Float64Array::from(vec![None::<f64>, None::<f64>])),
            ],
        )
        .unwrap();
        let tmp = write_parquet(&batch);
        let rows = parse_penalty_overrides_hydro(tmp.path()).unwrap();

        assert_eq!(rows.len(), 2);

        // Stage 0
        assert_eq!(rows[0].hydro_id, EntityId::from(7));
        assert_eq!(rows[0].stage_id, 0);
        assert!((rows[0].spillage_cost.unwrap() - 0.01).abs() < f64::EPSILON);
        assert!(rows[0].turbined_cost.is_none());
        assert!(rows[0].diversion_cost.is_none());
        assert!((rows[0].storage_violation_below_cost.unwrap() - 9999.0).abs() < f64::EPSILON);
        assert!(rows[0].filling_target_violation_cost.is_none());
        assert!(rows[0].turbined_violation_below_cost.is_none());
        assert!(rows[0].outflow_violation_below_cost.is_none());
        assert!(rows[0].outflow_violation_above_cost.is_none());
        assert!(rows[0].generation_violation_below_cost.is_none());
        assert!(rows[0].evaporation_violation_cost.is_none());
        assert!(rows[0].water_withdrawal_violation_cost.is_none());

        // Stage 1
        assert_eq!(rows[1].hydro_id, EntityId::from(7));
        assert_eq!(rows[1].stage_id, 1);
        assert!((rows[1].spillage_cost.unwrap() - 0.02).abs() < f64::EPSILON);
        assert!((rows[1].storage_violation_below_cost.unwrap() - 8888.0).abs() < f64::EPSILON);
    }

    /// AC: valid file with all 11 optional columns present, mix of null and non-null.
    #[test]
    fn test_hydro_all_11_columns_mixed() {
        let batch = RecordBatch::try_new(
            hydro_full_schema(),
            vec![
                Arc::new(Int32Array::from(vec![2_i32])),
                Arc::new(Int32Array::from(vec![0_i32])),
                Arc::new(Float64Array::from(vec![Some(0.01_f64)])),
                Arc::new(Float64Array::from(vec![Some(100.0_f64)])),
                Arc::new(Float64Array::from(vec![None::<f64>])),
                Arc::new(Float64Array::from(vec![Some(9999.0_f64)])),
                Arc::new(Float64Array::from(vec![Some(5000.0_f64)])),
                Arc::new(Float64Array::from(vec![None::<f64>])),
                Arc::new(Float64Array::from(vec![Some(2000.0_f64)])),
                Arc::new(Float64Array::from(vec![Some(3000.0_f64)])),
                Arc::new(Float64Array::from(vec![None::<f64>])),
                Arc::new(Float64Array::from(vec![Some(50.0_f64)])),
                Arc::new(Float64Array::from(vec![None::<f64>])),
            ],
        )
        .unwrap();
        let tmp = write_parquet(&batch);
        let rows = parse_penalty_overrides_hydro(tmp.path()).unwrap();

        assert_eq!(rows.len(), 1);
        let r = &rows[0];
        assert_eq!(r.hydro_id, EntityId::from(2));
        assert_eq!(r.stage_id, 0);
        assert!(r.spillage_cost.is_some());
        assert!(r.turbined_cost.is_some());
        assert!(r.diversion_cost.is_none());
        assert!(r.storage_violation_below_cost.is_some());
        assert!(r.filling_target_violation_cost.is_some());
        assert!(r.turbined_violation_below_cost.is_none());
        assert!(r.outflow_violation_below_cost.is_some());
        assert!(r.outflow_violation_above_cost.is_some());
        assert!(r.generation_violation_below_cost.is_none());
        assert!(r.evaporation_violation_cost.is_some());
        assert!(r.water_withdrawal_violation_cost.is_none());
    }

    /// AC: only 2 of 11 optional columns present in the Parquet schema.
    #[test]
    fn test_hydro_only_2_columns_in_schema() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("hydro_id", DataType::Int32, false),
            Field::new("stage_id", DataType::Int32, false),
            Field::new("spillage_cost", DataType::Float64, true),
            Field::new("storage_violation_below_cost", DataType::Float64, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![3_i32, 3_i32])),
                Arc::new(Int32Array::from(vec![0_i32, 1_i32])),
                Arc::new(Float64Array::from(vec![Some(0.5_f64), None::<f64>])),
                Arc::new(Float64Array::from(vec![None::<f64>, Some(100.0_f64)])),
            ],
        )
        .unwrap();
        let tmp = write_parquet(&batch);
        let rows = parse_penalty_overrides_hydro(tmp.path()).unwrap();

        assert_eq!(rows.len(), 2);
        // The other 9 columns absent from schema -> all None.
        assert!(rows[0].turbined_cost.is_none());
        assert!(rows[0].diversion_cost.is_none());
        assert!(rows[0].filling_target_violation_cost.is_none());
        assert!(rows[0].turbined_violation_below_cost.is_none());
        assert!(rows[0].outflow_violation_below_cost.is_none());
        assert!(rows[0].outflow_violation_above_cost.is_none());
        assert!(rows[0].generation_violation_below_cost.is_none());
        assert!(rows[0].evaporation_violation_cost.is_none());
        assert!(rows[0].water_withdrawal_violation_cost.is_none());
        // spillage_cost present in schema
        assert!((rows[0].spillage_cost.unwrap() - 0.5).abs() < f64::EPSILON);
        assert!(rows[0].storage_violation_below_cost.is_none());
        // Stage 1
        assert!(rows[1].spillage_cost.is_none());
        assert!((rows[1].storage_violation_below_cost.unwrap() - 100.0).abs() < f64::EPSILON);
    }

    /// AC: missing required `hydro_id` column -> SchemaError with field "hydro_id".
    #[test]
    fn test_hydro_missing_hydro_id() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("stage_id", DataType::Int32, false),
            Field::new("spillage_cost", DataType::Float64, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![0_i32])),
                Arc::new(Float64Array::from(vec![Some(0.01_f64)])),
            ],
        )
        .unwrap();
        let tmp = write_parquet(&batch);
        let err = parse_penalty_overrides_hydro(tmp.path()).unwrap_err();

        match &err {
            LoadError::SchemaError { field, .. } => {
                assert!(
                    field.contains("hydro_id"),
                    "field should contain 'hydro_id', got: {field}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// AC: negative penalty in hydro spillage_cost -> SchemaError with "must be > 0.0".
    #[test]
    fn test_hydro_negative_spillage_cost() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("hydro_id", DataType::Int32, false),
            Field::new("stage_id", DataType::Int32, false),
            Field::new("spillage_cost", DataType::Float64, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![1_i32])),
                Arc::new(Int32Array::from(vec![0_i32])),
                Arc::new(Float64Array::from(vec![Some(-0.01_f64)])),
            ],
        )
        .unwrap();
        let tmp = write_parquet(&batch);
        let err = parse_penalty_overrides_hydro(tmp.path()).unwrap_err();

        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("spillage_cost"),
                    "field should contain 'spillage_cost', got: {field}"
                );
                assert!(
                    message.contains("must be > 0.0"),
                    "message should contain 'must be > 0.0', got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// AC: NaN in any hydro penalty column -> SchemaError mentioning "finite".
    #[test]
    fn test_hydro_nan_penalty() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("hydro_id", DataType::Int32, false),
            Field::new("stage_id", DataType::Int32, false),
            Field::new("evaporation_violation_cost", DataType::Float64, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![1_i32])),
                Arc::new(Int32Array::from(vec![0_i32])),
                Arc::new(Float64Array::from(vec![Some(f64::NAN)])),
            ],
        )
        .unwrap();
        let tmp = write_parquet(&batch);
        let err = parse_penalty_overrides_hydro(tmp.path()).unwrap_err();

        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("evaporation_violation_cost"),
                    "field should contain 'evaporation_violation_cost', got: {field}"
                );
                assert!(
                    message.contains("finite"),
                    "message should contain 'finite', got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// AC: empty file (0 rows) -> Ok(Vec::new()).
    #[test]
    fn test_hydro_empty_parquet() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("hydro_id", DataType::Int32, false),
            Field::new("stage_id", DataType::Int32, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![] as Vec<i32>)),
                Arc::new(Int32Array::from(vec![] as Vec<i32>)),
            ],
        )
        .unwrap();
        let tmp = write_parquet(&batch);
        let rows = parse_penalty_overrides_hydro(tmp.path()).unwrap();
        assert!(rows.is_empty());
    }

    /// AC: load_penalty_overrides_hydro(None) -> Ok(Vec::new()).
    #[test]
    fn test_load_hydro_none() {
        let rows = super::super::load_penalty_overrides_hydro(None).unwrap();
        assert!(rows.is_empty());
    }

    /// A Parquet file containing the 5 directional override columns parses each
    /// into its matching struct field.
    #[test]
    fn test_hydro_directional_columns_parsed() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("hydro_id", DataType::Int32, false),
            Field::new("stage_id", DataType::Int32, false),
            Field::new(
                "water_withdrawal_violation_pos_cost",
                DataType::Float64,
                true,
            ),
            Field::new(
                "water_withdrawal_violation_neg_cost",
                DataType::Float64,
                true,
            ),
            Field::new("evaporation_violation_pos_cost", DataType::Float64, true),
            Field::new("evaporation_violation_neg_cost", DataType::Float64, true),
            Field::new("inflow_nonnegativity_cost", DataType::Float64, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![4_i32])),
                Arc::new(Int32Array::from(vec![2_i32])),
                Arc::new(Float64Array::from(vec![Some(120.0_f64)])),
                Arc::new(Float64Array::from(vec![Some(80.0_f64)])),
                Arc::new(Float64Array::from(vec![Some(55.0_f64)])),
                Arc::new(Float64Array::from(vec![None::<f64>])),
                Arc::new(Float64Array::from(vec![Some(200.0_f64)])),
            ],
        )
        .unwrap();
        let tmp = write_parquet(&batch);
        let rows = parse_penalty_overrides_hydro(tmp.path()).unwrap();

        assert_eq!(rows.len(), 1);
        let r = &rows[0];
        assert_eq!(r.hydro_id, EntityId::from(4));
        assert_eq!(r.stage_id, 2);
        assert!(
            (r.water_withdrawal_violation_pos_cost.unwrap() - 120.0).abs() < f64::EPSILON,
            "water_withdrawal_violation_pos_cost must be 120.0"
        );
        assert!(
            (r.water_withdrawal_violation_neg_cost.unwrap() - 80.0).abs() < f64::EPSILON,
            "water_withdrawal_violation_neg_cost must be 80.0"
        );
        assert!(
            (r.evaporation_violation_pos_cost.unwrap() - 55.0).abs() < f64::EPSILON,
            "evaporation_violation_pos_cost must be 55.0"
        );
        assert!(
            r.evaporation_violation_neg_cost.is_none(),
            "evaporation_violation_neg_cost must be None (null in file)"
        );
        assert!(
            (r.inflow_nonnegativity_cost.unwrap() - 200.0).abs() < f64::EPSILON,
            "inflow_nonnegativity_cost must be 200.0"
        );
        // Columns absent from file schema -> None.
        assert!(r.spillage_cost.is_none());
        assert!(r.water_withdrawal_violation_cost.is_none());
    }

    /// AC: negative value in a directional column -> SchemaError mentioning the column name.
    #[test]
    fn test_hydro_directional_column_negative_rejected() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("hydro_id", DataType::Int32, false),
            Field::new("stage_id", DataType::Int32, false),
            Field::new("inflow_nonnegativity_cost", DataType::Float64, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![1_i32])),
                Arc::new(Int32Array::from(vec![0_i32])),
                Arc::new(Float64Array::from(vec![Some(-5.0_f64)])),
            ],
        )
        .unwrap();
        let tmp = write_parquet(&batch);
        let err = parse_penalty_overrides_hydro(tmp.path()).unwrap_err();

        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("inflow_nonnegativity_cost"),
                    "field should contain 'inflow_nonnegativity_cost', got: {field}"
                );
                assert!(
                    message.contains("must be > 0.0"),
                    "message should contain 'must be > 0.0', got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// AC: scrambled input order -> output sorted by (hydro_id, stage_id).
    #[test]
    fn test_hydro_declaration_order_invariance() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("hydro_id", DataType::Int32, false),
            Field::new("stage_id", DataType::Int32, false),
            Field::new("spillage_cost", DataType::Float64, true),
        ]));
        let batch_scrambled = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![5_i32, 1_i32, 5_i32, 1_i32])),
                Arc::new(Int32Array::from(vec![1_i32, 1_i32, 0_i32, 0_i32])),
                Arc::new(Float64Array::from(vec![
                    Some(5.1_f64),
                    Some(1.1_f64),
                    Some(5.0_f64),
                    Some(1.0_f64),
                ])),
            ],
        )
        .unwrap();
        let batch_asc = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![1_i32, 1_i32, 5_i32, 5_i32])),
                Arc::new(Int32Array::from(vec![0_i32, 1_i32, 0_i32, 1_i32])),
                Arc::new(Float64Array::from(vec![
                    Some(1.0_f64),
                    Some(1.1_f64),
                    Some(5.0_f64),
                    Some(5.1_f64),
                ])),
            ],
        )
        .unwrap();
        let tmp_scrambled = write_parquet(&batch_scrambled);
        let tmp_asc = write_parquet(&batch_asc);
        let rows_scrambled = parse_penalty_overrides_hydro(tmp_scrambled.path()).unwrap();
        let rows_asc = parse_penalty_overrides_hydro(tmp_asc.path()).unwrap();

        let keys_s: Vec<(i32, i32)> = rows_scrambled
            .iter()
            .map(|r| (r.hydro_id.0, r.stage_id))
            .collect();
        let keys_a: Vec<(i32, i32)> = rows_asc
            .iter()
            .map(|r| (r.hydro_id.0, r.stage_id))
            .collect();
        assert_eq!(keys_s, keys_a);
    }

    // ── NcsPenaltyOverrideRow tests ───────────────────────────────────────────

    fn ncs_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("ncs_id", DataType::Int32, false),
            Field::new("stage_id", DataType::Int32, false),
            Field::new("curtailment_cost", DataType::Float64, true),
        ]))
    }

    fn make_ncs_batch(
        source_ids: &[i32],
        stage_ids: &[i32],
        curtailment_cost: Vec<Option<f64>>,
    ) -> RecordBatch {
        RecordBatch::try_new(
            ncs_schema(),
            vec![
                Arc::new(Int32Array::from(source_ids.to_vec())),
                Arc::new(Int32Array::from(stage_ids.to_vec())),
                Arc::new(Float64Array::from(curtailment_cost)),
            ],
        )
        .expect("valid batch")
    }

    /// AC: 3 rows for 2 NCS entities, correctly sorted by (source_id, stage_id).
    #[test]
    fn test_ncs_valid_3_rows_sorted() {
        // Scrambled: (2,0),(1,1),(1,0) -> sorted: (1,0),(1,1),(2,0).
        let batch = make_ncs_batch(&[2, 1, 1], &[0, 1, 0], vec![Some(300.0), Some(250.0), None]);
        let tmp = write_parquet(&batch);
        let rows = parse_penalty_overrides_ncs(tmp.path()).unwrap();

        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0].source_id, EntityId::from(1));
        assert_eq!(rows[0].stage_id, 0);
        assert!(rows[0].curtailment_cost.is_none());
        assert_eq!(rows[1].source_id, EntityId::from(1));
        assert_eq!(rows[1].stage_id, 1);
        assert!((rows[1].curtailment_cost.unwrap() - 250.0).abs() < f64::EPSILON);
        assert_eq!(rows[2].source_id, EntityId::from(2));
        assert_eq!(rows[2].stage_id, 0);
        assert!((rows[2].curtailment_cost.unwrap() - 300.0).abs() < f64::EPSILON);
    }

    /// Missing id column -> error names the `ncs_id` spelling.
    #[test]
    fn test_ncs_missing_id_column_errors_new_spelling() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("stage_id", DataType::Int32, false),
            Field::new("curtailment_cost", DataType::Float64, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![0_i32])),
                Arc::new(Float64Array::from(vec![Some(250.0_f64)])),
            ],
        )
        .unwrap();
        let tmp = write_parquet(&batch);
        let err = parse_penalty_overrides_ncs(tmp.path()).unwrap_err();

        match &err {
            LoadError::SchemaError { field, .. } => {
                assert_eq!(field, "ncs_id", "field should be 'ncs_id', got: {field}");
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// AC: negative curtailment_cost -> SchemaError with "must be > 0.0".
    #[test]
    fn test_ncs_negative_curtailment_cost() {
        let batch = make_ncs_batch(&[1], &[0], vec![Some(-1.0)]);
        let tmp = write_parquet(&batch);
        let err = parse_penalty_overrides_ncs(tmp.path()).unwrap_err();

        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("curtailment_cost"),
                    "field should contain 'curtailment_cost', got: {field}"
                );
                assert!(
                    message.contains("must be > 0.0"),
                    "message should contain 'must be > 0.0', got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// AC: NaN in curtailment_cost -> SchemaError mentioning "finite".
    #[test]
    fn test_ncs_nan_curtailment_cost() {
        let batch = make_ncs_batch(&[1], &[0], vec![Some(f64::NAN)]);
        let tmp = write_parquet(&batch);
        let err = parse_penalty_overrides_ncs(tmp.path()).unwrap_err();

        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("curtailment_cost"),
                    "field should contain 'curtailment_cost', got: {field}"
                );
                assert!(
                    message.contains("finite"),
                    "message should contain 'finite', got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// AC: empty file (0 rows) -> Ok(Vec::new()).
    #[test]
    fn test_ncs_empty_parquet() {
        let batch = make_ncs_batch(&[], &[], vec![]);
        let tmp = write_parquet(&batch);
        let rows = parse_penalty_overrides_ncs(tmp.path()).unwrap();
        assert!(rows.is_empty());
    }

    /// AC: load_penalty_overrides_ncs(None) -> Ok(Vec::new()).
    #[test]
    fn test_load_ncs_none() {
        let rows = super::super::load_penalty_overrides_ncs(None).unwrap();
        assert!(rows.is_empty());
    }
}
