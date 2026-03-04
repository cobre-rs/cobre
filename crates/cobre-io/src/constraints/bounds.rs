//! Parquet parsers for entity bounds override files in the `constraints/` subdirectory.
//!
//! Each parser reads a sparse Parquet file containing stage-varying bound overrides
//! for a specific entity type. Sparse storage means only `(entity_id, stage_id)` pairs
//! that differ from base values need rows.
//!
//! ## Parquet schemas
//!
//! ### `thermal_bounds`
//!
//! | Column             | Type   | Required | Description                    |
//! | ------------------ | ------ | -------- | ------------------------------ |
//! | `thermal_id`       | INT32  | Yes      | Thermal plant ID               |
//! | `stage_id`         | INT32  | Yes      | Stage ID                       |
//! | `min_generation_mw`| DOUBLE | No       | Minimum generation (MW)        |
//! | `max_generation_mw`| DOUBLE | No       | Maximum generation (MW)        |
//!
//! ### `hydro_bounds`
//!
//! | Column                | Type   | Required | Description                        |
//! | --------------------- | ------ | -------- | ---------------------------------- |
//! | `hydro_id`            | INT32  | Yes      | Hydro plant ID                     |
//! | `stage_id`            | INT32  | Yes      | Stage ID                           |
//! | `min_turbined_m3s`    | DOUBLE | No       | Min turbined flow (m3/s)           |
//! | `max_turbined_m3s`    | DOUBLE | No       | Max turbined flow (m3/s)           |
//! | `min_storage_hm3`     | DOUBLE | No       | Min reservoir storage (hm3)        |
//! | `max_storage_hm3`     | DOUBLE | No       | Max reservoir storage (hm3)        |
//! | `min_outflow_m3s`     | DOUBLE | No       | Min total outflow (m3/s)           |
//! | `max_outflow_m3s`     | DOUBLE | No       | Max total outflow (m3/s)           |
//! | `min_generation_mw`   | DOUBLE | No       | Min generation (MW)                |
//! | `max_generation_mw`   | DOUBLE | No       | Max generation (MW)                |
//! | `max_diversion_m3s`   | DOUBLE | No       | Max diversion flow (m3/s)          |
//! | `filling_inflow_m3s`  | DOUBLE | No       | Filling inflow override (m3/s)     |
//! | `water_withdrawal_m3s`| DOUBLE | No       | Water withdrawal (m3/s)            |
//!
//! ### `line_bounds`
//!
//! | Column       | Type   | Required | Description                        |
//! | ------------ | ------ | -------- | ---------------------------------- |
//! | `line_id`    | INT32  | Yes      | Transmission line ID               |
//! | `stage_id`   | INT32  | Yes      | Stage ID                           |
//! | `direct_mw`  | DOUBLE | No       | Direct-flow capacity (MW)          |
//! | `reverse_mw` | DOUBLE | No       | Reverse-flow capacity (MW)         |
//!
//! ### `pumping_bounds`
//!
//! | Column       | Type   | Required | Description                        |
//! | ------------ | ------ | -------- | ---------------------------------- |
//! | `station_id` | INT32  | Yes      | Pumping station ID                 |
//! | `stage_id`   | INT32  | Yes      | Stage ID                           |
//! | `min_m3s`    | DOUBLE | No       | Minimum pumping flow (m3/s)        |
//! | `max_m3s`    | DOUBLE | No       | Maximum pumping flow (m3/s)        |
//!
//! ### `contract_bounds`
//!
//! | Column           | Type   | Required | Description                    |
//! | ---------------- | ------ | -------- | ------------------------------ |
//! | `contract_id`    | INT32  | Yes      | Energy contract ID             |
//! | `stage_id`       | INT32  | Yes      | Stage ID                       |
//! | `min_mw`         | DOUBLE | No       | Minimum power (MW)             |
//! | `max_mw`         | DOUBLE | No       | Maximum power (MW)             |
//! | `price_per_mwh`  | DOUBLE | No       | Price override ($/`MWh`)       |
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
//!
//! Deferred validations (not performed here):
//!
//! - Entity ID existence in registries — Layer 3, Epic 06.
//! - Duplicate `(entity_id, stage_id)` pairs — Epic 06.
//! - Semantic cross-validation (e.g., min < max) — Epic 06.

use arrow::array::Array;
use cobre_core::EntityId;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::fs::File;
use std::path::Path;

use crate::LoadError;
use crate::parquet_helpers::{extract_optional_float64, extract_required_int32};

// ── Row types ─────────────────────────────────────────────────────────────────

/// A single row from `constraints/thermal_bounds.parquet`.
///
/// Carries stage-varying bound overrides for a thermal generation plant.
/// Fields are `None` when the corresponding column is absent or null in the
/// Parquet file (sparse storage: absent means "use base value").
///
/// # Examples
///
/// ```
/// use cobre_io::constraints::ThermalBoundsRow;
/// use cobre_core::EntityId;
///
/// let row = ThermalBoundsRow {
///     thermal_id: EntityId::from(2),
///     stage_id: 5,
///     min_generation_mw: Some(10.0),
///     max_generation_mw: None,
/// };
/// assert_eq!(row.thermal_id, EntityId::from(2));
/// assert_eq!(row.min_generation_mw, Some(10.0));
/// assert!(row.max_generation_mw.is_none());
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct ThermalBoundsRow {
    /// Thermal plant ID.
    pub thermal_id: EntityId,
    /// Stage ID.
    pub stage_id: i32,
    /// Minimum generation override (MW).
    pub min_generation_mw: Option<f64>,
    /// Maximum generation override (MW).
    pub max_generation_mw: Option<f64>,
}

/// A single row from `constraints/hydro_bounds.parquet`.
///
/// Carries stage-varying bound overrides for a hydro plant. All eleven bound
/// columns are optional; absent or null means "use base value".
///
/// # Examples
///
/// ```
/// use cobre_io::constraints::HydroBoundsRow;
/// use cobre_core::EntityId;
///
/// let row = HydroBoundsRow {
///     hydro_id: EntityId::from(1),
///     stage_id: 3,
///     min_turbined_m3s: Some(50.0),
///     max_turbined_m3s: None,
///     min_storage_hm3: None,
///     max_storage_hm3: Some(200.0),
///     min_outflow_m3s: None,
///     max_outflow_m3s: None,
///     min_generation_mw: None,
///     max_generation_mw: None,
///     max_diversion_m3s: None,
///     filling_inflow_m3s: None,
///     water_withdrawal_m3s: None,
/// };
/// assert_eq!(row.min_turbined_m3s, Some(50.0));
/// assert!(row.max_turbined_m3s.is_none());
/// ```
#[allow(clippy::struct_field_names)]
#[derive(Debug, Clone, PartialEq)]
pub struct HydroBoundsRow {
    /// Hydro plant ID.
    pub hydro_id: EntityId,
    /// Stage ID.
    pub stage_id: i32,
    /// Minimum turbined flow override (m³/s).
    pub min_turbined_m3s: Option<f64>,
    /// Maximum turbined flow override (m³/s).
    pub max_turbined_m3s: Option<f64>,
    /// Minimum storage override (hm³).
    pub min_storage_hm3: Option<f64>,
    /// Maximum storage override (hm³).
    pub max_storage_hm3: Option<f64>,
    /// Minimum outflow override (m³/s).
    pub min_outflow_m3s: Option<f64>,
    /// Maximum outflow override (m³/s).
    pub max_outflow_m3s: Option<f64>,
    /// Minimum generation override (MW).
    pub min_generation_mw: Option<f64>,
    /// Maximum generation override (MW).
    pub max_generation_mw: Option<f64>,
    /// Maximum diversion override (m³/s).
    pub max_diversion_m3s: Option<f64>,
    /// Filling inflow override (m³/s).
    pub filling_inflow_m3s: Option<f64>,
    /// Water withdrawal override (m³/s).
    pub water_withdrawal_m3s: Option<f64>,
}

/// A single row from `constraints/line_bounds.parquet`.
///
/// Carries stage-varying capacity overrides for a transmission line.
/// Fields are `None` when absent or null (sparse storage).
///
/// # Examples
///
/// ```
/// use cobre_io::constraints::LineBoundsRow;
/// use cobre_core::EntityId;
///
/// let row = LineBoundsRow {
///     line_id: EntityId::from(10),
///     stage_id: 0,
///     direct_mw: Some(500.0),
///     reverse_mw: Some(500.0),
/// };
/// assert_eq!(row.line_id, EntityId::from(10));
/// assert_eq!(row.direct_mw, Some(500.0));
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct LineBoundsRow {
    /// Transmission line this override applies to.
    pub line_id: EntityId,
    /// Stage (0-based) this override applies to.
    pub stage_id: i32,
    /// Override for direct-flow capacity (MW). `None` means use base value.
    pub direct_mw: Option<f64>,
    /// Override for reverse-flow capacity (MW). `None` means use base value.
    pub reverse_mw: Option<f64>,
}

/// A single row from `constraints/pumping_bounds.parquet`.
///
/// Carries stage-varying flow bound overrides for a pumping station.
/// Fields are `None` when absent or null (sparse storage).
///
/// # Examples
///
/// ```
/// use cobre_io::constraints::PumpingBoundsRow;
/// use cobre_core::EntityId;
///
/// let row = PumpingBoundsRow {
///     station_id: EntityId::from(3),
///     stage_id: 2,
///     min_m3s: Some(0.0),
///     max_m3s: Some(100.0),
/// };
/// assert_eq!(row.station_id, EntityId::from(3));
/// assert_eq!(row.max_m3s, Some(100.0));
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct PumpingBoundsRow {
    /// Pumping station this override applies to.
    pub station_id: EntityId,
    /// Stage (0-based) this override applies to.
    pub stage_id: i32,
    /// Override for minimum pumping flow (m³/s). `None` means use base value.
    pub min_m3s: Option<f64>,
    /// Override for maximum pumping flow (m³/s). `None` means use base value.
    pub max_m3s: Option<f64>,
}

/// A single row from `constraints/contract_bounds.parquet`.
///
/// Carries stage-varying bound and price overrides for an energy contract.
/// Fields are `None` when absent or null (sparse storage).
///
/// # Examples
///
/// ```
/// use cobre_io::constraints::ContractBoundsRow;
/// use cobre_core::EntityId;
///
/// let row = ContractBoundsRow {
///     contract_id: EntityId::from(7),
///     stage_id: 1,
///     min_mw: Some(0.0),
///     max_mw: Some(200.0),
///     price_per_mwh: None,
/// };
/// assert_eq!(row.contract_id, EntityId::from(7));
/// assert!(row.price_per_mwh.is_none());
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct ContractBoundsRow {
    /// Energy contract this override applies to.
    pub contract_id: EntityId,
    /// Stage (0-based) this override applies to.
    pub stage_id: i32,
    /// Override for minimum power (MW). `None` means use base value.
    pub min_mw: Option<f64>,
    /// Override for maximum power (MW). `None` means use base value.
    pub max_mw: Option<f64>,
    /// Override for contract price (USD/`MWh`). `None` means use base value.
    pub price_per_mwh: Option<f64>,
}

// ── Parsers ───────────────────────────────────────────────────────────────────

/// Validate that a present (non-null) optional float value is finite.
///
/// Returns `SchemaError` when the value is NaN or infinite. The `file_label`
/// is used in the field path: `"<file_label>[{row_idx}].<column>"`.
fn validate_optional_finite(
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
    }
    Ok(())
}

/// Parse `constraints/thermal_bounds.parquet` and return a sorted row table.
///
/// Reads all record batches from the Parquet file at `path`, validates per-row
/// constraints, then returns all rows sorted by `(thermal_id, stage_id)` ascending.
///
/// # Errors
///
/// | Condition                                     | Error variant              |
/// |---------------------------------------------- |--------------------------- |
/// | File not found or permission denied           | [`LoadError::IoError`]     |
/// | Malformed Parquet (corrupt header, etc.)      | [`LoadError::ParseError`]  |
/// | Required column missing or wrong type         | [`LoadError::SchemaError`] |
/// | Non-finite value in any present bound column  | [`LoadError::SchemaError`] |
///
/// # Examples
///
/// ```no_run
/// use cobre_io::constraints::parse_thermal_bounds;
/// use std::path::Path;
///
/// let rows = parse_thermal_bounds(Path::new("constraints/thermal_bounds.parquet"))
///     .expect("valid thermal bounds file");
/// println!("loaded {} thermal bounds rows", rows.len());
/// ```
pub fn parse_thermal_bounds(path: &Path) -> Result<Vec<ThermalBoundsRow>, LoadError> {
    let file = File::open(path).map_err(|e| LoadError::io(path, e))?;

    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| LoadError::parse(path, e.to_string()))?;

    let reader = builder
        .build()
        .map_err(|e| LoadError::parse(path, e.to_string()))?;

    let mut rows: Vec<ThermalBoundsRow> = Vec::new();

    for batch_result in reader {
        let batch = batch_result.map_err(|e| LoadError::parse(path, e.to_string()))?;

        // ── Required key columns ──────────────────────────────────────────────
        let thermal_id_col = extract_required_int32(&batch, "thermal_id", path)?;
        let stage_id_col = extract_required_int32(&batch, "stage_id", path)?;

        // ── Optional bound columns ────────────────────────────────────────────
        let min_gen_col = extract_optional_float64(&batch, "min_generation_mw", path)?;
        let max_gen_col = extract_optional_float64(&batch, "max_generation_mw", path)?;

        let n = batch.num_rows();
        let base_idx = rows.len();
        rows.reserve(n);

        for i in 0..n {
            let row_idx = base_idx + i;

            let thermal_id = EntityId::from(thermal_id_col.value(i));
            let stage_id = stage_id_col.value(i);

            let min_generation_mw = min_gen_col
                .filter(|col| !col.is_null(i))
                .map(|col| col.value(i));
            let max_generation_mw = max_gen_col
                .filter(|col| !col.is_null(i))
                .map(|col| col.value(i));

            validate_optional_finite(
                min_generation_mw,
                "thermal_bounds",
                row_idx,
                "min_generation_mw",
                path,
            )?;
            validate_optional_finite(
                max_generation_mw,
                "thermal_bounds",
                row_idx,
                "max_generation_mw",
                path,
            )?;

            rows.push(ThermalBoundsRow {
                thermal_id,
                stage_id,
                min_generation_mw,
                max_generation_mw,
            });
        }
    }

    rows.sort_by(|a, b| {
        a.thermal_id
            .0
            .cmp(&b.thermal_id.0)
            .then_with(|| a.stage_id.cmp(&b.stage_id))
    });

    Ok(rows)
}

/// Parse `constraints/hydro_bounds.parquet` and return a sorted row table.
///
/// Reads all record batches from the Parquet file at `path`, validates per-row
/// constraints, then returns all rows sorted by `(hydro_id, stage_id)` ascending.
///
/// The Parquet file may contain any subset of the eleven optional bound columns.
/// Columns absent from the file schema produce `None` in all rows (not a schema
/// error — this is the intended sparse-override design).
///
/// # Errors
///
/// | Condition                                     | Error variant              |
/// |---------------------------------------------- |--------------------------- |
/// | File not found or permission denied           | [`LoadError::IoError`]     |
/// | Malformed Parquet (corrupt header, etc.)      | [`LoadError::ParseError`]  |
/// | Required column missing or wrong type         | [`LoadError::SchemaError`] |
/// | Non-finite value in any present bound column  | [`LoadError::SchemaError`] |
///
/// # Examples
///
/// ```no_run
/// use cobre_io::constraints::parse_hydro_bounds;
/// use std::path::Path;
///
/// let rows = parse_hydro_bounds(Path::new("constraints/hydro_bounds.parquet"))
///     .expect("valid hydro bounds file");
/// println!("loaded {} hydro bounds rows", rows.len());
/// ```
#[allow(clippy::too_many_lines)]
pub fn parse_hydro_bounds(path: &Path) -> Result<Vec<HydroBoundsRow>, LoadError> {
    let file = File::open(path).map_err(|e| LoadError::io(path, e))?;

    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| LoadError::parse(path, e.to_string()))?;

    let reader = builder
        .build()
        .map_err(|e| LoadError::parse(path, e.to_string()))?;

    let mut rows: Vec<HydroBoundsRow> = Vec::new();

    for batch_result in reader {
        let batch = batch_result.map_err(|e| LoadError::parse(path, e.to_string()))?;

        // ── Required key columns ──────────────────────────────────────────────
        let hydro_id_col = extract_required_int32(&batch, "hydro_id", path)?;
        let stage_id_col = extract_required_int32(&batch, "stage_id", path)?;

        // ── Optional bound columns (all 11) ───────────────────────────────────
        let min_turbined_col = extract_optional_float64(&batch, "min_turbined_m3s", path)?;
        let max_turbined_col = extract_optional_float64(&batch, "max_turbined_m3s", path)?;
        let min_storage_col = extract_optional_float64(&batch, "min_storage_hm3", path)?;
        let max_storage_col = extract_optional_float64(&batch, "max_storage_hm3", path)?;
        let min_outflow_col = extract_optional_float64(&batch, "min_outflow_m3s", path)?;
        let max_outflow_col = extract_optional_float64(&batch, "max_outflow_m3s", path)?;
        let min_gen_col = extract_optional_float64(&batch, "min_generation_mw", path)?;
        let max_gen_col = extract_optional_float64(&batch, "max_generation_mw", path)?;
        let max_diversion_col = extract_optional_float64(&batch, "max_diversion_m3s", path)?;
        let filling_inflow_col = extract_optional_float64(&batch, "filling_inflow_m3s", path)?;
        let water_withdrawal_col = extract_optional_float64(&batch, "water_withdrawal_m3s", path)?;

        let n = batch.num_rows();
        let base_idx = rows.len();
        rows.reserve(n);

        for i in 0..n {
            let row_idx = base_idx + i;

            let hydro_id = EntityId::from(hydro_id_col.value(i));
            let stage_id = stage_id_col.value(i);

            let min_turbined_m3s = min_turbined_col
                .filter(|col| !col.is_null(i))
                .map(|col| col.value(i));
            let max_turbined_m3s = max_turbined_col
                .filter(|col| !col.is_null(i))
                .map(|col| col.value(i));
            let min_storage_hm3 = min_storage_col
                .filter(|col| !col.is_null(i))
                .map(|col| col.value(i));
            let max_storage_hm3 = max_storage_col
                .filter(|col| !col.is_null(i))
                .map(|col| col.value(i));
            let min_outflow_m3s = min_outflow_col
                .filter(|col| !col.is_null(i))
                .map(|col| col.value(i));
            let max_outflow_m3s = max_outflow_col
                .filter(|col| !col.is_null(i))
                .map(|col| col.value(i));
            let min_generation_mw = min_gen_col
                .filter(|col| !col.is_null(i))
                .map(|col| col.value(i));
            let max_generation_mw = max_gen_col
                .filter(|col| !col.is_null(i))
                .map(|col| col.value(i));
            let max_diversion_m3s = max_diversion_col
                .filter(|col| !col.is_null(i))
                .map(|col| col.value(i));
            let filling_inflow_m3s = filling_inflow_col
                .filter(|col| !col.is_null(i))
                .map(|col| col.value(i));
            let water_withdrawal_m3s = water_withdrawal_col
                .filter(|col| !col.is_null(i))
                .map(|col| col.value(i));

            validate_optional_finite(
                min_turbined_m3s,
                "hydro_bounds",
                row_idx,
                "min_turbined_m3s",
                path,
            )?;
            validate_optional_finite(
                max_turbined_m3s,
                "hydro_bounds",
                row_idx,
                "max_turbined_m3s",
                path,
            )?;
            validate_optional_finite(
                min_storage_hm3,
                "hydro_bounds",
                row_idx,
                "min_storage_hm3",
                path,
            )?;
            validate_optional_finite(
                max_storage_hm3,
                "hydro_bounds",
                row_idx,
                "max_storage_hm3",
                path,
            )?;
            validate_optional_finite(
                min_outflow_m3s,
                "hydro_bounds",
                row_idx,
                "min_outflow_m3s",
                path,
            )?;
            validate_optional_finite(
                max_outflow_m3s,
                "hydro_bounds",
                row_idx,
                "max_outflow_m3s",
                path,
            )?;
            validate_optional_finite(
                min_generation_mw,
                "hydro_bounds",
                row_idx,
                "min_generation_mw",
                path,
            )?;
            validate_optional_finite(
                max_generation_mw,
                "hydro_bounds",
                row_idx,
                "max_generation_mw",
                path,
            )?;
            validate_optional_finite(
                max_diversion_m3s,
                "hydro_bounds",
                row_idx,
                "max_diversion_m3s",
                path,
            )?;
            validate_optional_finite(
                filling_inflow_m3s,
                "hydro_bounds",
                row_idx,
                "filling_inflow_m3s",
                path,
            )?;
            validate_optional_finite(
                water_withdrawal_m3s,
                "hydro_bounds",
                row_idx,
                "water_withdrawal_m3s",
                path,
            )?;

            rows.push(HydroBoundsRow {
                hydro_id,
                stage_id,
                min_turbined_m3s,
                max_turbined_m3s,
                min_storage_hm3,
                max_storage_hm3,
                min_outflow_m3s,
                max_outflow_m3s,
                min_generation_mw,
                max_generation_mw,
                max_diversion_m3s,
                filling_inflow_m3s,
                water_withdrawal_m3s,
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

/// Parse `constraints/line_bounds.parquet` and return a sorted row table.
///
/// Reads all record batches from the Parquet file at `path`, validates per-row
/// constraints, then returns all rows sorted by `(line_id, stage_id)` ascending.
///
/// # Errors
///
/// | Condition                                     | Error variant              |
/// |---------------------------------------------- |--------------------------- |
/// | File not found or permission denied           | [`LoadError::IoError`]     |
/// | Malformed Parquet (corrupt header, etc.)      | [`LoadError::ParseError`]  |
/// | Required column missing or wrong type         | [`LoadError::SchemaError`] |
/// | Non-finite value in any present bound column  | [`LoadError::SchemaError`] |
///
/// # Examples
///
/// ```no_run
/// use cobre_io::constraints::parse_line_bounds;
/// use std::path::Path;
///
/// let rows = parse_line_bounds(Path::new("constraints/line_bounds.parquet"))
///     .expect("valid line bounds file");
/// println!("loaded {} line bounds rows", rows.len());
/// ```
pub fn parse_line_bounds(path: &Path) -> Result<Vec<LineBoundsRow>, LoadError> {
    let file = File::open(path).map_err(|e| LoadError::io(path, e))?;

    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| LoadError::parse(path, e.to_string()))?;

    let reader = builder
        .build()
        .map_err(|e| LoadError::parse(path, e.to_string()))?;

    let mut rows: Vec<LineBoundsRow> = Vec::new();

    for batch_result in reader {
        let batch = batch_result.map_err(|e| LoadError::parse(path, e.to_string()))?;

        // ── Required key columns ──────────────────────────────────────────────
        let line_id_col = extract_required_int32(&batch, "line_id", path)?;
        let stage_id_col = extract_required_int32(&batch, "stage_id", path)?;

        // ── Optional bound columns ────────────────────────────────────────────
        let direct_col = extract_optional_float64(&batch, "direct_mw", path)?;
        let reverse_col = extract_optional_float64(&batch, "reverse_mw", path)?;

        let n = batch.num_rows();
        let base_idx = rows.len();
        rows.reserve(n);

        for i in 0..n {
            let row_idx = base_idx + i;

            let line_id = EntityId::from(line_id_col.value(i));
            let stage_id = stage_id_col.value(i);

            let direct_mw = direct_col
                .filter(|col| !col.is_null(i))
                .map(|col| col.value(i));
            let reverse_mw = reverse_col
                .filter(|col| !col.is_null(i))
                .map(|col| col.value(i));

            validate_optional_finite(direct_mw, "line_bounds", row_idx, "direct_mw", path)?;
            validate_optional_finite(reverse_mw, "line_bounds", row_idx, "reverse_mw", path)?;

            rows.push(LineBoundsRow {
                line_id,
                stage_id,
                direct_mw,
                reverse_mw,
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

/// Parse `constraints/pumping_bounds.parquet` and return a sorted row table.
///
/// Reads all record batches from the Parquet file at `path`, validates per-row
/// constraints, then returns all rows sorted by `(station_id, stage_id)` ascending.
///
/// # Errors
///
/// | Condition                                     | Error variant              |
/// |---------------------------------------------- |--------------------------- |
/// | File not found or permission denied           | [`LoadError::IoError`]     |
/// | Malformed Parquet (corrupt header, etc.)      | [`LoadError::ParseError`]  |
/// | Required column missing or wrong type         | [`LoadError::SchemaError`] |
/// | Non-finite value in any present bound column  | [`LoadError::SchemaError`] |
///
/// # Examples
///
/// ```no_run
/// use cobre_io::constraints::parse_pumping_bounds;
/// use std::path::Path;
///
/// let rows = parse_pumping_bounds(Path::new("constraints/pumping_bounds.parquet"))
///     .expect("valid pumping bounds file");
/// println!("loaded {} pumping bounds rows", rows.len());
/// ```
pub fn parse_pumping_bounds(path: &Path) -> Result<Vec<PumpingBoundsRow>, LoadError> {
    let file = File::open(path).map_err(|e| LoadError::io(path, e))?;

    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| LoadError::parse(path, e.to_string()))?;

    let reader = builder
        .build()
        .map_err(|e| LoadError::parse(path, e.to_string()))?;

    let mut rows: Vec<PumpingBoundsRow> = Vec::new();

    for batch_result in reader {
        let batch = batch_result.map_err(|e| LoadError::parse(path, e.to_string()))?;

        // ── Required key columns ──────────────────────────────────────────────
        let station_id_col = extract_required_int32(&batch, "station_id", path)?;
        let stage_id_col = extract_required_int32(&batch, "stage_id", path)?;

        // ── Optional bound columns ────────────────────────────────────────────
        let min_col = extract_optional_float64(&batch, "min_m3s", path)?;
        let max_col = extract_optional_float64(&batch, "max_m3s", path)?;

        let n = batch.num_rows();
        let base_idx = rows.len();
        rows.reserve(n);

        for i in 0..n {
            let row_idx = base_idx + i;

            let station_id = EntityId::from(station_id_col.value(i));
            let stage_id = stage_id_col.value(i);

            let min_m3s = min_col
                .filter(|col| !col.is_null(i))
                .map(|col| col.value(i));
            let max_m3s = max_col
                .filter(|col| !col.is_null(i))
                .map(|col| col.value(i));

            validate_optional_finite(min_m3s, "pumping_bounds", row_idx, "min_m3s", path)?;
            validate_optional_finite(max_m3s, "pumping_bounds", row_idx, "max_m3s", path)?;

            rows.push(PumpingBoundsRow {
                station_id,
                stage_id,
                min_m3s,
                max_m3s,
            });
        }
    }

    rows.sort_by(|a, b| {
        a.station_id
            .0
            .cmp(&b.station_id.0)
            .then_with(|| a.stage_id.cmp(&b.stage_id))
    });

    Ok(rows)
}

/// Parse `constraints/contract_bounds.parquet` and return a sorted row table.
///
/// Reads all record batches from the Parquet file at `path`, validates per-row
/// constraints, then returns all rows sorted by `(contract_id, stage_id)` ascending.
///
/// # Errors
///
/// | Condition                                     | Error variant              |
/// |---------------------------------------------- |--------------------------- |
/// | File not found or permission denied           | [`LoadError::IoError`]     |
/// | Malformed Parquet (corrupt header, etc.)      | [`LoadError::ParseError`]  |
/// | Required column missing or wrong type         | [`LoadError::SchemaError`] |
/// | Non-finite value in any present bound column  | [`LoadError::SchemaError`] |
///
/// # Examples
///
/// ```no_run
/// use cobre_io::constraints::parse_contract_bounds;
/// use std::path::Path;
///
/// let rows = parse_contract_bounds(Path::new("constraints/contract_bounds.parquet"))
///     .expect("valid contract bounds file");
/// println!("loaded {} contract bounds rows", rows.len());
/// ```
pub fn parse_contract_bounds(path: &Path) -> Result<Vec<ContractBoundsRow>, LoadError> {
    let file = File::open(path).map_err(|e| LoadError::io(path, e))?;

    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| LoadError::parse(path, e.to_string()))?;

    let reader = builder
        .build()
        .map_err(|e| LoadError::parse(path, e.to_string()))?;

    let mut rows: Vec<ContractBoundsRow> = Vec::new();

    for batch_result in reader {
        let batch = batch_result.map_err(|e| LoadError::parse(path, e.to_string()))?;

        // ── Required key columns ──────────────────────────────────────────────
        let contract_id_col = extract_required_int32(&batch, "contract_id", path)?;
        let stage_id_col = extract_required_int32(&batch, "stage_id", path)?;

        // ── Optional bound columns ────────────────────────────────────────────
        let min_col = extract_optional_float64(&batch, "min_mw", path)?;
        let max_col = extract_optional_float64(&batch, "max_mw", path)?;
        let price_col = extract_optional_float64(&batch, "price_per_mwh", path)?;

        let n = batch.num_rows();
        let base_idx = rows.len();
        rows.reserve(n);

        for i in 0..n {
            let row_idx = base_idx + i;

            let contract_id = EntityId::from(contract_id_col.value(i));
            let stage_id = stage_id_col.value(i);

            let min_mw = min_col
                .filter(|col| !col.is_null(i))
                .map(|col| col.value(i));
            let max_mw = max_col
                .filter(|col| !col.is_null(i))
                .map(|col| col.value(i));
            let price_per_mwh = price_col
                .filter(|col| !col.is_null(i))
                .map(|col| col.value(i));

            validate_optional_finite(min_mw, "contract_bounds", row_idx, "min_mw", path)?;
            validate_optional_finite(max_mw, "contract_bounds", row_idx, "max_mw", path)?;
            validate_optional_finite(
                price_per_mwh,
                "contract_bounds",
                row_idx,
                "price_per_mwh",
                path,
            )?;

            rows.push(ContractBoundsRow {
                contract_id,
                stage_id,
                min_mw,
                max_mw,
                price_per_mwh,
            });
        }
    }

    rows.sort_by(|a, b| {
        a.contract_id
            .0
            .cmp(&b.contract_id.0)
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
    use arrow::array::{Float64Array, Int32Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;
    use std::sync::Arc;
    use tempfile::NamedTempFile;

    // ── Shared test helpers ───────────────────────────────────────────────────

    fn write_parquet(batch: &RecordBatch) -> NamedTempFile {
        let tmp = NamedTempFile::new().expect("tempfile");
        let mut writer = ArrowWriter::try_new(tmp.reopen().expect("reopen"), batch.schema(), None)
            .expect("ArrowWriter");
        writer.write(batch).expect("write batch");
        writer.close().expect("close writer");
        tmp
    }

    // ── ThermalBoundsRow tests ────────────────────────────────────────────────

    fn thermal_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("thermal_id", DataType::Int32, false),
            Field::new("stage_id", DataType::Int32, false),
            Field::new("min_generation_mw", DataType::Float64, true),
            Field::new("max_generation_mw", DataType::Float64, true),
        ]))
    }

    fn make_thermal_batch(
        thermal_ids: &[i32],
        stage_ids: &[i32],
        min_gen: Vec<Option<f64>>,
        max_gen: Vec<Option<f64>>,
    ) -> RecordBatch {
        RecordBatch::try_new(
            thermal_schema(),
            vec![
                Arc::new(Int32Array::from(thermal_ids.to_vec())),
                Arc::new(Int32Array::from(stage_ids.to_vec())),
                Arc::new(Float64Array::from(min_gen)),
                Arc::new(Float64Array::from(max_gen)),
            ],
        )
        .expect("valid batch")
    }

    /// AC: 3 rows for 2 thermals across 2 stages, valid happy path, correct sort order.
    #[test]
    fn test_thermal_valid_3_rows_sorted() {
        // Scrambled: (3,0), (1,1), (1,0) — result must be sorted (1,0),(1,1),(3,0).
        let batch = make_thermal_batch(
            &[3, 1, 1],
            &[0, 1, 0],
            vec![Some(10.0), Some(5.0), Some(8.0)],
            vec![Some(100.0), Some(80.0), None],
        );
        let tmp = write_parquet(&batch);
        let rows = parse_thermal_bounds(tmp.path()).unwrap();

        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0].thermal_id, EntityId::from(1));
        assert_eq!(rows[0].stage_id, 0);
        assert!((rows[0].min_generation_mw.unwrap() - 8.0).abs() < f64::EPSILON);
        assert!(rows[0].max_generation_mw.is_none());
        assert_eq!(rows[1].thermal_id, EntityId::from(1));
        assert_eq!(rows[1].stage_id, 1);
        assert_eq!(rows[2].thermal_id, EntityId::from(3));
        assert_eq!(rows[2].stage_id, 0);
        assert!((rows[2].min_generation_mw.unwrap() - 10.0).abs() < f64::EPSILON);
        assert!((rows[2].max_generation_mw.unwrap() - 100.0).abs() < f64::EPSILON);
    }

    /// AC: missing `stage_id` column -> SchemaError with field "stage_id".
    #[test]
    fn test_thermal_missing_stage_id() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("thermal_id", DataType::Int32, false),
            Field::new("max_generation_mw", DataType::Float64, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![1_i32])),
                Arc::new(Float64Array::from(vec![Some(100.0_f64)])),
            ],
        )
        .unwrap();
        let tmp = write_parquet(&batch);
        let err = parse_thermal_bounds(tmp.path()).unwrap_err();

        match &err {
            LoadError::SchemaError { field, .. } => {
                assert!(
                    field.contains("stage_id"),
                    "field should contain 'stage_id', got: {field}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// AC: NaN in `max_generation_mw` -> SchemaError mentioning "max_generation_mw" and "finite".
    #[test]
    fn test_thermal_nan_max_generation() {
        let batch = make_thermal_batch(&[1], &[0], vec![Some(10.0)], vec![Some(f64::NAN)]);
        let tmp = write_parquet(&batch);
        let err = parse_thermal_bounds(tmp.path()).unwrap_err();

        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("max_generation_mw"),
                    "field should contain 'max_generation_mw', got: {field}"
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
    fn test_thermal_empty_parquet() {
        let batch = make_thermal_batch(&[], &[], vec![], vec![]);
        let tmp = write_parquet(&batch);
        let rows = parse_thermal_bounds(tmp.path()).unwrap();
        assert!(rows.is_empty());
    }

    /// AC: scrambled input order -> output sorted by (thermal_id, stage_id).
    #[test]
    fn test_thermal_declaration_order_invariance() {
        let batch_asc = make_thermal_batch(
            &[1, 1, 5, 5],
            &[0, 1, 0, 1],
            vec![Some(10.0), Some(11.0), Some(50.0), Some(51.0)],
            vec![Some(100.0), Some(110.0), Some(500.0), Some(510.0)],
        );
        let batch_desc = make_thermal_batch(
            &[5, 5, 1, 1],
            &[1, 0, 1, 0],
            vec![Some(51.0), Some(50.0), Some(11.0), Some(10.0)],
            vec![Some(510.0), Some(500.0), Some(110.0), Some(100.0)],
        );
        let tmp_asc = write_parquet(&batch_asc);
        let tmp_desc = write_parquet(&batch_desc);
        let rows_asc = parse_thermal_bounds(tmp_asc.path()).unwrap();
        let rows_desc = parse_thermal_bounds(tmp_desc.path()).unwrap();

        let keys_asc: Vec<(i32, i32)> = rows_asc
            .iter()
            .map(|r| (r.thermal_id.0, r.stage_id))
            .collect();
        let keys_desc: Vec<(i32, i32)> = rows_desc
            .iter()
            .map(|r| (r.thermal_id.0, r.stage_id))
            .collect();
        assert_eq!(keys_asc, keys_desc);
    }

    /// AC: load_thermal_bounds(None) returns Ok(Vec::new()).
    #[test]
    fn test_load_thermal_bounds_none() {
        let rows = super::super::load_thermal_bounds(None).unwrap();
        assert!(rows.is_empty());
    }

    // ── HydroBoundsRow tests ──────────────────────────────────────────────────

    fn hydro_full_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("hydro_id", DataType::Int32, false),
            Field::new("stage_id", DataType::Int32, false),
            Field::new("min_turbined_m3s", DataType::Float64, true),
            Field::new("max_turbined_m3s", DataType::Float64, true),
            Field::new("min_storage_hm3", DataType::Float64, true),
            Field::new("max_storage_hm3", DataType::Float64, true),
            Field::new("min_outflow_m3s", DataType::Float64, true),
            Field::new("max_outflow_m3s", DataType::Float64, true),
            Field::new("min_generation_mw", DataType::Float64, true),
            Field::new("max_generation_mw", DataType::Float64, true),
            Field::new("max_diversion_m3s", DataType::Float64, true),
            Field::new("filling_inflow_m3s", DataType::Float64, true),
            Field::new("water_withdrawal_m3s", DataType::Float64, true),
        ]))
    }

    /// AC: valid file with all 11 optional columns, some null and some filled.
    #[test]
    fn test_hydro_all_11_columns_mixed_null() {
        let batch = RecordBatch::try_new(
            hydro_full_schema(),
            vec![
                Arc::new(Int32Array::from(vec![1_i32])),
                Arc::new(Int32Array::from(vec![0_i32])),
                Arc::new(Float64Array::from(vec![Some(50.0_f64)])), // min_turbined
                Arc::new(Float64Array::from(vec![None::<f64>])),    // max_turbined: null
                Arc::new(Float64Array::from(vec![Some(100.0_f64)])), // min_storage
                Arc::new(Float64Array::from(vec![Some(500.0_f64)])), // max_storage
                Arc::new(Float64Array::from(vec![None::<f64>])),    // min_outflow: null
                Arc::new(Float64Array::from(vec![Some(200.0_f64)])), // max_outflow
                Arc::new(Float64Array::from(vec![None::<f64>])),    // min_gen: null
                Arc::new(Float64Array::from(vec![Some(80.0_f64)])), // max_gen
                Arc::new(Float64Array::from(vec![None::<f64>])),    // max_diversion: null
                Arc::new(Float64Array::from(vec![Some(10.0_f64)])), // filling_inflow
                Arc::new(Float64Array::from(vec![None::<f64>])),    // water_withdrawal: null
            ],
        )
        .unwrap();
        let tmp = write_parquet(&batch);
        let rows = parse_hydro_bounds(tmp.path()).unwrap();

        assert_eq!(rows.len(), 1);
        let r = &rows[0];
        assert_eq!(r.hydro_id, EntityId::from(1));
        assert_eq!(r.stage_id, 0);
        assert!((r.min_turbined_m3s.unwrap() - 50.0).abs() < f64::EPSILON);
        assert!(r.max_turbined_m3s.is_none());
        assert!((r.min_storage_hm3.unwrap() - 100.0).abs() < f64::EPSILON);
        assert!((r.max_storage_hm3.unwrap() - 500.0).abs() < f64::EPSILON);
        assert!(r.min_outflow_m3s.is_none());
        assert!((r.max_outflow_m3s.unwrap() - 200.0).abs() < f64::EPSILON);
        assert!(r.min_generation_mw.is_none());
        assert!((r.max_generation_mw.unwrap() - 80.0).abs() < f64::EPSILON);
        assert!(r.max_diversion_m3s.is_none());
        assert!((r.filling_inflow_m3s.unwrap() - 10.0).abs() < f64::EPSILON);
        assert!(r.water_withdrawal_m3s.is_none());
    }

    /// AC: only a subset of optional columns present in the schema.
    #[test]
    fn test_hydro_subset_of_optional_columns() {
        // Only min_turbined_m3s and max_generation_mw present; all others absent.
        let schema = Arc::new(Schema::new(vec![
            Field::new("hydro_id", DataType::Int32, false),
            Field::new("stage_id", DataType::Int32, false),
            Field::new("min_turbined_m3s", DataType::Float64, true),
            Field::new("max_generation_mw", DataType::Float64, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![2_i32])),
                Arc::new(Int32Array::from(vec![3_i32])),
                Arc::new(Float64Array::from(vec![Some(75.0_f64)])),
                Arc::new(Float64Array::from(vec![Some(150.0_f64)])),
            ],
        )
        .unwrap();
        let tmp = write_parquet(&batch);
        let rows = parse_hydro_bounds(tmp.path()).unwrap();

        assert_eq!(rows.len(), 1);
        let r = &rows[0];
        assert_eq!(r.hydro_id, EntityId::from(2));
        assert_eq!(r.stage_id, 3);
        assert!((r.min_turbined_m3s.unwrap() - 75.0).abs() < f64::EPSILON);
        assert!(r.max_turbined_m3s.is_none());
        assert!(r.min_storage_hm3.is_none());
        assert!(r.max_storage_hm3.is_none());
        assert!(r.min_outflow_m3s.is_none());
        assert!(r.max_outflow_m3s.is_none());
        assert!(r.min_generation_mw.is_none());
        assert!((r.max_generation_mw.unwrap() - 150.0).abs() < f64::EPSILON);
        assert!(r.max_diversion_m3s.is_none());
        assert!(r.filling_inflow_m3s.is_none());
        assert!(r.water_withdrawal_m3s.is_none());
    }

    /// AC: missing `stage_id` column -> SchemaError with field "stage_id".
    #[test]
    fn test_hydro_missing_stage_id() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("hydro_id", DataType::Int32, false),
            Field::new("min_turbined_m3s", DataType::Float64, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![1_i32])),
                Arc::new(Float64Array::from(vec![Some(50.0_f64)])),
            ],
        )
        .unwrap();
        let tmp = write_parquet(&batch);
        let err = parse_hydro_bounds(tmp.path()).unwrap_err();

        match &err {
            LoadError::SchemaError { field, .. } => {
                assert!(
                    field.contains("stage_id"),
                    "field should contain 'stage_id', got: {field}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// AC: NaN in a present optional column -> SchemaError mentioning column and "finite".
    #[test]
    fn test_hydro_nan_in_optional_column() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("hydro_id", DataType::Int32, false),
            Field::new("stage_id", DataType::Int32, false),
            Field::new("min_storage_hm3", DataType::Float64, true),
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
        let err = parse_hydro_bounds(tmp.path()).unwrap_err();

        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("min_storage_hm3"),
                    "field should contain 'min_storage_hm3', got: {field}"
                );
                assert!(
                    message.contains("finite"),
                    "message should contain 'finite', got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// AC: empty file -> Ok(Vec::new()).
    #[test]
    fn test_hydro_empty_parquet() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("hydro_id", DataType::Int32, false),
            Field::new("stage_id", DataType::Int32, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(Vec::<i32>::new())),
                Arc::new(Int32Array::from(Vec::<i32>::new())),
            ],
        )
        .unwrap();
        let tmp = write_parquet(&batch);
        let rows = parse_hydro_bounds(tmp.path()).unwrap();
        assert!(rows.is_empty());
    }

    /// AC: scrambled input order -> output sorted by (hydro_id, stage_id).
    #[test]
    fn test_hydro_declaration_order_invariance() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("hydro_id", DataType::Int32, false),
            Field::new("stage_id", DataType::Int32, false),
            Field::new("max_turbined_m3s", DataType::Float64, true),
        ]));

        let make = |hydro_ids: &[i32], stage_ids: &[i32]| -> RecordBatch {
            RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(Int32Array::from(hydro_ids.to_vec())),
                    Arc::new(Int32Array::from(stage_ids.to_vec())),
                    Arc::new(Float64Array::from(
                        hydro_ids
                            .iter()
                            .map(|_| Some(100.0_f64))
                            .collect::<Vec<_>>(),
                    )),
                ],
            )
            .unwrap()
        };

        let tmp_asc = write_parquet(&make(&[1, 1, 3, 3], &[0, 1, 0, 1]));
        let tmp_desc = write_parquet(&make(&[3, 3, 1, 1], &[1, 0, 1, 0]));
        let rows_asc = parse_hydro_bounds(tmp_asc.path()).unwrap();
        let rows_desc = parse_hydro_bounds(tmp_desc.path()).unwrap();

        let keys_asc: Vec<(i32, i32)> = rows_asc
            .iter()
            .map(|r| (r.hydro_id.0, r.stage_id))
            .collect();
        let keys_desc: Vec<(i32, i32)> = rows_desc
            .iter()
            .map(|r| (r.hydro_id.0, r.stage_id))
            .collect();
        assert_eq!(keys_asc, keys_desc);
    }

    /// AC: load_hydro_bounds(None) returns Ok(Vec::new()).
    #[test]
    fn test_load_hydro_bounds_none() {
        let rows = super::super::load_hydro_bounds(None).unwrap();
        assert!(rows.is_empty());
    }

    // ── LineBoundsRow tests ───────────────────────────────────────────────────

    fn line_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("line_id", DataType::Int32, false),
            Field::new("stage_id", DataType::Int32, false),
            Field::new("direct_mw", DataType::Float64, true),
            Field::new("reverse_mw", DataType::Float64, true),
        ]))
    }

    fn make_line_batch(
        line_ids: &[i32],
        stage_ids: &[i32],
        direct: Vec<Option<f64>>,
        reverse: Vec<Option<f64>>,
    ) -> RecordBatch {
        RecordBatch::try_new(
            line_schema(),
            vec![
                Arc::new(Int32Array::from(line_ids.to_vec())),
                Arc::new(Int32Array::from(stage_ids.to_vec())),
                Arc::new(Float64Array::from(direct)),
                Arc::new(Float64Array::from(reverse)),
            ],
        )
        .expect("valid batch")
    }

    /// AC: valid line bounds, correct sort order.
    #[test]
    fn test_line_valid_rows_sorted() {
        let batch = make_line_batch(
            &[5, 2, 2],
            &[0, 1, 0],
            vec![Some(400.0), Some(300.0), Some(320.0)],
            vec![Some(400.0), None, Some(320.0)],
        );
        let tmp = write_parquet(&batch);
        let rows = parse_line_bounds(tmp.path()).unwrap();

        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0].line_id, EntityId::from(2));
        assert_eq!(rows[0].stage_id, 0);
        assert!((rows[0].direct_mw.unwrap() - 320.0).abs() < f64::EPSILON);
        assert_eq!(rows[1].line_id, EntityId::from(2));
        assert_eq!(rows[1].stage_id, 1);
        assert!(rows[1].reverse_mw.is_none());
        assert_eq!(rows[2].line_id, EntityId::from(5));
    }

    /// AC: missing `stage_id` -> SchemaError.
    #[test]
    fn test_line_missing_stage_id() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("line_id", DataType::Int32, false),
            Field::new("direct_mw", DataType::Float64, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![1_i32])),
                Arc::new(Float64Array::from(vec![Some(100.0_f64)])),
            ],
        )
        .unwrap();
        let tmp = write_parquet(&batch);
        let err = parse_line_bounds(tmp.path()).unwrap_err();

        match &err {
            LoadError::SchemaError { field, .. } => {
                assert!(field.contains("stage_id"), "got: {field}");
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// AC: NaN in `direct_mw` -> SchemaError mentioning "direct_mw" and "finite".
    #[test]
    fn test_line_nan_direct_mw() {
        let batch = make_line_batch(&[1], &[0], vec![Some(f64::NAN)], vec![Some(100.0)]);
        let tmp = write_parquet(&batch);
        let err = parse_line_bounds(tmp.path()).unwrap_err();

        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(field.contains("direct_mw"), "got: {field}");
                assert!(message.contains("finite"), "got: {message}");
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// AC: empty file -> Ok(Vec::new()).
    #[test]
    fn test_line_empty_parquet() {
        let batch = make_line_batch(&[], &[], vec![], vec![]);
        let tmp = write_parquet(&batch);
        assert!(parse_line_bounds(tmp.path()).unwrap().is_empty());
    }

    /// AC: scrambled input -> sorted output.
    #[test]
    fn test_line_declaration_order_invariance() {
        let asc = make_line_batch(
            &[1, 1, 4, 4],
            &[0, 1, 0, 1],
            vec![Some(100.0), Some(110.0), Some(400.0), Some(410.0)],
            vec![Some(100.0), Some(110.0), Some(400.0), Some(410.0)],
        );
        let desc = make_line_batch(
            &[4, 4, 1, 1],
            &[1, 0, 1, 0],
            vec![Some(410.0), Some(400.0), Some(110.0), Some(100.0)],
            vec![Some(410.0), Some(400.0), Some(110.0), Some(100.0)],
        );
        let tmp_asc = write_parquet(&asc);
        let tmp_desc = write_parquet(&desc);
        let rows_asc = parse_line_bounds(tmp_asc.path()).unwrap();
        let rows_desc = parse_line_bounds(tmp_desc.path()).unwrap();

        let keys_asc: Vec<(i32, i32)> =
            rows_asc.iter().map(|r| (r.line_id.0, r.stage_id)).collect();
        let keys_desc: Vec<(i32, i32)> = rows_desc
            .iter()
            .map(|r| (r.line_id.0, r.stage_id))
            .collect();
        assert_eq!(keys_asc, keys_desc);
    }

    /// AC: load_line_bounds(None) returns Ok(Vec::new()).
    #[test]
    fn test_load_line_bounds_none() {
        let rows = super::super::load_line_bounds(None).unwrap();
        assert!(rows.is_empty());
    }

    // ── PumpingBoundsRow tests ────────────────────────────────────────────────

    fn pumping_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("station_id", DataType::Int32, false),
            Field::new("stage_id", DataType::Int32, false),
            Field::new("min_m3s", DataType::Float64, true),
            Field::new("max_m3s", DataType::Float64, true),
        ]))
    }

    fn make_pumping_batch(
        station_ids: &[i32],
        stage_ids: &[i32],
        min_m3s: Vec<Option<f64>>,
        max_m3s: Vec<Option<f64>>,
    ) -> RecordBatch {
        RecordBatch::try_new(
            pumping_schema(),
            vec![
                Arc::new(Int32Array::from(station_ids.to_vec())),
                Arc::new(Int32Array::from(stage_ids.to_vec())),
                Arc::new(Float64Array::from(min_m3s)),
                Arc::new(Float64Array::from(max_m3s)),
            ],
        )
        .expect("valid batch")
    }

    /// AC: valid pumping bounds, correct sort order.
    #[test]
    fn test_pumping_valid_rows_sorted() {
        let batch = make_pumping_batch(
            &[3, 1, 1],
            &[0, 1, 0],
            vec![Some(0.0), None, Some(5.0)],
            vec![Some(80.0), Some(90.0), Some(85.0)],
        );
        let tmp = write_parquet(&batch);
        let rows = parse_pumping_bounds(tmp.path()).unwrap();

        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0].station_id, EntityId::from(1));
        assert_eq!(rows[0].stage_id, 0);
        assert!((rows[0].min_m3s.unwrap() - 5.0).abs() < f64::EPSILON);
        assert_eq!(rows[1].station_id, EntityId::from(1));
        assert_eq!(rows[1].stage_id, 1);
        assert!(rows[1].min_m3s.is_none());
        assert_eq!(rows[2].station_id, EntityId::from(3));
    }

    /// AC: missing `stage_id` -> SchemaError.
    #[test]
    fn test_pumping_missing_stage_id() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("station_id", DataType::Int32, false),
            Field::new("min_m3s", DataType::Float64, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![1_i32])),
                Arc::new(Float64Array::from(vec![Some(5.0_f64)])),
            ],
        )
        .unwrap();
        let tmp = write_parquet(&batch);
        let err = parse_pumping_bounds(tmp.path()).unwrap_err();

        match &err {
            LoadError::SchemaError { field, .. } => {
                assert!(field.contains("stage_id"), "got: {field}");
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// AC: NaN in `max_m3s` -> SchemaError mentioning "max_m3s" and "finite".
    #[test]
    fn test_pumping_nan_max_m3s() {
        let batch = make_pumping_batch(&[1], &[0], vec![Some(0.0)], vec![Some(f64::NAN)]);
        let tmp = write_parquet(&batch);
        let err = parse_pumping_bounds(tmp.path()).unwrap_err();

        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(field.contains("max_m3s"), "got: {field}");
                assert!(message.contains("finite"), "got: {message}");
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// AC: empty file -> Ok(Vec::new()).
    #[test]
    fn test_pumping_empty_parquet() {
        let batch = make_pumping_batch(&[], &[], vec![], vec![]);
        let tmp = write_parquet(&batch);
        assert!(parse_pumping_bounds(tmp.path()).unwrap().is_empty());
    }

    /// AC: scrambled input -> sorted output.
    #[test]
    fn test_pumping_declaration_order_invariance() {
        let asc = make_pumping_batch(
            &[1, 1, 2, 2],
            &[0, 1, 0, 1],
            vec![Some(0.0), Some(0.0), Some(1.0), Some(1.0)],
            vec![Some(10.0), Some(10.0), Some(20.0), Some(20.0)],
        );
        let desc = make_pumping_batch(
            &[2, 2, 1, 1],
            &[1, 0, 1, 0],
            vec![Some(1.0), Some(1.0), Some(0.0), Some(0.0)],
            vec![Some(20.0), Some(20.0), Some(10.0), Some(10.0)],
        );
        let tmp_asc = write_parquet(&asc);
        let tmp_desc = write_parquet(&desc);
        let rows_asc = parse_pumping_bounds(tmp_asc.path()).unwrap();
        let rows_desc = parse_pumping_bounds(tmp_desc.path()).unwrap();

        let keys_asc: Vec<(i32, i32)> = rows_asc
            .iter()
            .map(|r| (r.station_id.0, r.stage_id))
            .collect();
        let keys_desc: Vec<(i32, i32)> = rows_desc
            .iter()
            .map(|r| (r.station_id.0, r.stage_id))
            .collect();
        assert_eq!(keys_asc, keys_desc);
    }

    /// AC: load_pumping_bounds(None) returns Ok(Vec::new()).
    #[test]
    fn test_load_pumping_bounds_none() {
        let rows = super::super::load_pumping_bounds(None).unwrap();
        assert!(rows.is_empty());
    }

    // ── ContractBoundsRow tests ───────────────────────────────────────────────

    fn contract_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("contract_id", DataType::Int32, false),
            Field::new("stage_id", DataType::Int32, false),
            Field::new("min_mw", DataType::Float64, true),
            Field::new("max_mw", DataType::Float64, true),
            Field::new("price_per_mwh", DataType::Float64, true),
        ]))
    }

    fn make_contract_batch(
        contract_ids: &[i32],
        stage_ids: &[i32],
        min_mw: Vec<Option<f64>>,
        max_mw: Vec<Option<f64>>,
        price: Vec<Option<f64>>,
    ) -> RecordBatch {
        RecordBatch::try_new(
            contract_schema(),
            vec![
                Arc::new(Int32Array::from(contract_ids.to_vec())),
                Arc::new(Int32Array::from(stage_ids.to_vec())),
                Arc::new(Float64Array::from(min_mw)),
                Arc::new(Float64Array::from(max_mw)),
                Arc::new(Float64Array::from(price)),
            ],
        )
        .expect("valid batch")
    }

    /// AC: valid contract bounds, correct sort order.
    #[test]
    fn test_contract_valid_rows_sorted() {
        let batch = make_contract_batch(
            &[7, 3, 3],
            &[0, 1, 0],
            vec![Some(0.0), None, Some(0.0)],
            vec![Some(200.0), Some(150.0), Some(180.0)],
            vec![Some(50.0), None, Some(45.0)],
        );
        let tmp = write_parquet(&batch);
        let rows = parse_contract_bounds(tmp.path()).unwrap();

        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0].contract_id, EntityId::from(3));
        assert_eq!(rows[0].stage_id, 0);
        assert!((rows[0].max_mw.unwrap() - 180.0).abs() < f64::EPSILON);
        assert!((rows[0].price_per_mwh.unwrap() - 45.0).abs() < f64::EPSILON);
        assert_eq!(rows[1].contract_id, EntityId::from(3));
        assert_eq!(rows[1].stage_id, 1);
        assert!(rows[1].min_mw.is_none());
        assert!(rows[1].price_per_mwh.is_none());
        assert_eq!(rows[2].contract_id, EntityId::from(7));
    }

    /// AC: missing `stage_id` -> SchemaError.
    #[test]
    fn test_contract_missing_stage_id() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("contract_id", DataType::Int32, false),
            Field::new("max_mw", DataType::Float64, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![1_i32])),
                Arc::new(Float64Array::from(vec![Some(100.0_f64)])),
            ],
        )
        .unwrap();
        let tmp = write_parquet(&batch);
        let err = parse_contract_bounds(tmp.path()).unwrap_err();

        match &err {
            LoadError::SchemaError { field, .. } => {
                assert!(field.contains("stage_id"), "got: {field}");
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// AC: NaN in `price_per_mwh` -> SchemaError mentioning "price_per_mwh" and "finite".
    #[test]
    fn test_contract_nan_price() {
        let batch = make_contract_batch(
            &[1],
            &[0],
            vec![Some(0.0)],
            vec![Some(100.0)],
            vec![Some(f64::NAN)],
        );
        let tmp = write_parquet(&batch);
        let err = parse_contract_bounds(tmp.path()).unwrap_err();

        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(field.contains("price_per_mwh"), "got: {field}");
                assert!(message.contains("finite"), "got: {message}");
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// AC: empty file -> Ok(Vec::new()).
    #[test]
    fn test_contract_empty_parquet() {
        let batch = make_contract_batch(&[], &[], vec![], vec![], vec![]);
        let tmp = write_parquet(&batch);
        assert!(parse_contract_bounds(tmp.path()).unwrap().is_empty());
    }

    /// AC: scrambled input -> sorted output.
    #[test]
    fn test_contract_declaration_order_invariance() {
        let asc = make_contract_batch(
            &[1, 1, 5, 5],
            &[0, 1, 0, 1],
            vec![Some(0.0), Some(0.0), Some(0.0), Some(0.0)],
            vec![Some(100.0), Some(100.0), Some(500.0), Some(500.0)],
            vec![Some(10.0), Some(10.0), Some(50.0), Some(50.0)],
        );
        let desc = make_contract_batch(
            &[5, 5, 1, 1],
            &[1, 0, 1, 0],
            vec![Some(0.0), Some(0.0), Some(0.0), Some(0.0)],
            vec![Some(500.0), Some(500.0), Some(100.0), Some(100.0)],
            vec![Some(50.0), Some(50.0), Some(10.0), Some(10.0)],
        );
        let tmp_asc = write_parquet(&asc);
        let tmp_desc = write_parquet(&desc);
        let rows_asc = parse_contract_bounds(tmp_asc.path()).unwrap();
        let rows_desc = parse_contract_bounds(tmp_desc.path()).unwrap();

        let keys_asc: Vec<(i32, i32)> = rows_asc
            .iter()
            .map(|r| (r.contract_id.0, r.stage_id))
            .collect();
        let keys_desc: Vec<(i32, i32)> = rows_desc
            .iter()
            .map(|r| (r.contract_id.0, r.stage_id))
            .collect();
        assert_eq!(keys_asc, keys_desc);
    }

    /// AC: load_contract_bounds(None) returns Ok(Vec::new()).
    #[test]
    fn test_load_contract_bounds_none() {
        let rows = super::super::load_contract_bounds(None).unwrap();
        assert!(rows.is_empty());
    }
}
