//! Parquet and JSON writers for stochastic artifact output files.
//!
//! [`write_noise_openings`] exports an [`OpeningTree`] to
//! `output/stochastic/noise_openings.parquet` using the 4-column schema
//! defined below:
//!
//! | Column          | Type    | Description                                  |
//! |-----------------|---------|----------------------------------------------|
//! | `stage_id`      | INT32   | Stage index (0-based)                        |
//! | `opening_index` | UINT32  | Opening index within the stage (0-based)     |
//! | `entity_index`  | UINT32  | Entity index within the noise vector (0-based)|
//! | `value`         | DOUBLE  | Noise realisation value                      |
//!
//! Rows are written in `(stage_id, opening_index, entity_index)` order,
//! matching the stage-major storage layout of the tree.
//!
//! [`write_inflow_seasonal_stats`] exports fitted seasonal statistics to
//! `output/stochastic/inflow_seasonal_stats.parquet` using the 4-column
//! schema matching the corresponding input file:
//!
//! | Column     | Type   | Description                         |
//! |------------|--------|-------------------------------------|
//! | `hydro_id` | INT32  | Hydro plant ID                      |
//! | `stage_id` | INT32  | Stage ID                            |
//! | `mean_m3s` | DOUBLE | Seasonal mean inflow (m³/s)         |
//! | `std_m3s`  | DOUBLE | Seasonal standard deviation (m³/s)  |
//!
//! [`write_inflow_ar_coefficients`] exports fitted AR lag coefficients to
//! `output/stochastic/inflow_ar_coefficients.parquet` using the 5-column
//! schema matching the corresponding input file:
//!
//! | Column               | Type   | Description                                  |
//! |----------------------|--------|----------------------------------------------|
//! | `hydro_id`           | INT32  | Hydro plant ID                               |
//! | `stage_id`           | INT32  | Stage ID                                     |
//! | `lag`                | INT32  | Lag index (1-based)                          |
//! | `coefficient`        | DOUBLE | AR coefficient (standardized, dimensionless) |
//! | `residual_std_ratio` | DOUBLE | Residual std ratio in (0, 1]                 |
//!
//! [`write_correlation_json`] exports a [`CorrelationModel`] to
//! `output/stochastic/correlation.json` using the same format as the input
//! `scenarios/correlation.json` so that copying the output file back into
//! `scenarios/` produces a round-trip-identical model.
//!
//! [`write_inflow_annual_component`] exports fitted annual component statistics to
//! `output/stochastic/inflow_annual_component.parquet` using the 5-column schema
//! matching the corresponding input file:
//!
//! | Column               | Type   | Description                                    |
//! |----------------------|--------|------------------------------------------------|
//! | `hydro_id`           | INT32  | Hydro plant ID                                 |
//! | `stage_id`           | INT32  | Stage ID                                       |
//! | `annual_coefficient` | DOUBLE | Annual component coefficient ψ (dimensionless) |
//! | `annual_mean_m3s`    | DOUBLE | Mean of rolling 12-month average (m³/s)        |
//! | `annual_std_m3s`     | DOUBLE | Std of rolling 12-month average (m³/s)         |
//!
//! [`write_load_seasonal_stats`] exports per-bus-per-stage load statistics to
//! `output/stochastic/load_seasonal_stats.parquet` using the 4-column schema
//! matching the corresponding input file:
//!
//! | Column     | Type   | Description                              |
//! |------------|--------|------------------------------------------|
//! | `bus_id`   | INT32  | Bus ID                                   |
//! | `stage_id` | INT32  | Stage ID                                 |
//! | `mean_mw`  | DOUBLE | Seasonal mean load demand (MW)           |
//! | `std_mw`   | DOUBLE | Seasonal standard deviation (MW)         |
//!
//! [`write_fitting_report`] exports a [`FittingReport`] to
//! `output/stochastic/fitting_report.json` using the structure:
//!
//! ```json
//! {
//!   "hydros": {
//!     "<hydro_id>": {
//!       "selected_order": 3,
//!       "coefficients": [[0.42, -0.11, 0.07]]
//!     }
//!   }
//! }
//! ```
//!
//! All writers use atomic file creation: data is first written to a `.tmp`
//! suffix, then renamed to the final path.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use arrow::array::{Float64Builder, Int32Builder, RecordBatch, UInt32Builder};
use arrow::datatypes::{DataType, Field, Schema};
use cobre_core::scenario::{CorrelationModel, CorrelationScheduleEntry};
use cobre_stochastic::OpeningTree;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use serde::Serialize;

use crate::output::error::OutputError;
use crate::output::parquet_config::ParquetWriterConfig;
use crate::scenarios::{
    InflowAnnualComponentRow, InflowArCoefficientRow, InflowSeasonalStatsRow, LoadSeasonalStatsRow,
};

/// Write an [`OpeningTree`] to a Parquet file at `path`.
///
/// The output schema is exactly 4 columns — `stage_id: Int32`,
/// `opening_index: UInt32`, `entity_index: UInt32`, `value: Float64` — in
/// that order, matching the noise openings schema and the input expected by
/// `parse_noise_openings` / `assemble_opening_tree`. Rows are written in
/// `(stage_id, opening_index, entity_index)` order.
///
/// The parent directory is created if it does not already exist. The write
/// is atomic: data goes to `{path}.tmp` first, then the file is renamed to
/// `path`. A partial `.tmp` file may remain on disk if the process is killed
/// mid-write, but the final path is never partially written.
///
/// # Errors
///
/// - [`OutputError::IoError`] — directory creation, file open, or rename fails.
/// - [`OutputError::SerializationError`] — Arrow/Parquet construction fails.
///
/// # Examples
///
/// ```no_run
/// use cobre_io::output::stochastic::write_noise_openings;
/// use cobre_stochastic::OpeningTree;
/// use std::path::Path;
///
/// # fn main() -> Result<(), cobre_io::OutputError> {
/// let tree = OpeningTree::from_parts(
///     vec![1.0, 2.0, 3.0, 4.0],
///     vec![1, 1],
///     2,
/// );
/// write_noise_openings(Path::new("/tmp/out/noise_openings.parquet"), &tree)?;
/// # Ok(())
/// # }
/// ```
pub fn write_noise_openings(path: &Path, tree: &OpeningTree) -> Result<(), OutputError> {
    ensure_parent_dir(path)?;
    let config = ParquetWriterConfig::default();
    let batch = build_noise_openings_batch(tree)?;
    write_parquet_atomic(path, &batch, &config)
}

/// Write a slice of [`InflowSeasonalStatsRow`] to a Parquet file at `path`.
///
/// The output schema is exactly 4 columns — `hydro_id: Int32`, `stage_id: Int32`,
/// `mean_m3s: Float64`, `std_m3s: Float64` — in that order, matching the schema
/// of `scenarios/inflow_seasonal_stats.parquet`. Rows are written in the order
/// given; the caller is responsible for sorting if canonical ordering is required.
///
/// The parent directory is created if it does not already exist. The write is
/// atomic: data goes to `{path}.tmp` first, then the file is renamed to `path`.
///
/// # Errors
///
/// - [`OutputError::IoError`] — directory creation, file open, or rename fails.
/// - [`OutputError::SerializationError`] — Arrow/Parquet construction fails.
///
/// # Examples
///
/// ```no_run
/// use cobre_io::output::stochastic::write_inflow_seasonal_stats;
/// use cobre_io::scenarios::InflowSeasonalStatsRow;
/// use cobre_core::EntityId;
/// use std::path::Path;
///
/// # fn main() -> Result<(), cobre_io::OutputError> {
/// let rows = vec![
///     InflowSeasonalStatsRow {
///         hydro_id: EntityId::from(1),
///         stage_id: 0,
///         mean_m3s: 150.0,
///         std_m3s: 30.0,
///     },
/// ];
/// write_inflow_seasonal_stats(
///     Path::new("/tmp/out/stochastic/inflow_seasonal_stats.parquet"),
///     &rows,
/// )?;
/// # Ok(())
/// # }
/// ```
pub fn write_inflow_seasonal_stats(
    path: &Path,
    rows: &[InflowSeasonalStatsRow],
) -> Result<(), OutputError> {
    ensure_parent_dir(path)?;
    let config = ParquetWriterConfig::default();
    let batch = build_inflow_seasonal_stats_batch(rows)?;
    write_parquet_atomic(path, &batch, &config)
}

/// Write a slice of [`InflowArCoefficientRow`] to a Parquet file at `path`.
///
/// The output schema is exactly 5 columns — `hydro_id: Int32`, `stage_id: Int32`,
/// `lag: Int32`, `coefficient: Float64`, `residual_std_ratio: Float64` — in that
/// order, matching the schema of `scenarios/inflow_ar_coefficients.parquet`.
/// Rows are written in the order given; the caller is responsible for sorting
/// if canonical ordering is required.
///
/// The parent directory is created if it does not already exist. The write is
/// atomic: data goes to `{path}.tmp` first, then the file is renamed to `path`.
///
/// # Errors
///
/// - [`OutputError::IoError`] — directory creation, file open, or rename fails.
/// - [`OutputError::SerializationError`] — Arrow/Parquet construction fails.
///
/// # Examples
///
/// ```no_run
/// use cobre_io::output::stochastic::write_inflow_ar_coefficients;
/// use cobre_io::scenarios::InflowArCoefficientRow;
/// use cobre_core::EntityId;
/// use std::path::Path;
///
/// # fn main() -> Result<(), cobre_io::OutputError> {
/// let rows = vec![
///     InflowArCoefficientRow {
///         hydro_id: EntityId::from(1),
///         stage_id: 0,
///         lag: 1,
///         coefficient: 0.45,
///         residual_std_ratio: 0.85,
///     },
/// ];
/// write_inflow_ar_coefficients(
///     Path::new("/tmp/out/stochastic/inflow_ar_coefficients.parquet"),
///     &rows,
/// )?;
/// # Ok(())
/// # }
/// ```
pub fn write_inflow_ar_coefficients(
    path: &Path,
    rows: &[InflowArCoefficientRow],
) -> Result<(), OutputError> {
    ensure_parent_dir(path)?;
    let config = ParquetWriterConfig::default();
    let batch = build_inflow_ar_coefficients_batch(rows)?;
    write_parquet_atomic(path, &batch, &config)
}

/// Write a slice of [`InflowAnnualComponentRow`] to a Parquet file at `path`.
///
/// The output schema is exactly 5 columns — `hydro_id: Int32`, `stage_id: Int32`,
/// `annual_coefficient: Float64`, `annual_mean_m3s: Float64`, `annual_std_m3s: Float64` —
/// in that order, matching the schema of `scenarios/inflow_annual_component.parquet`.
/// Rows are written in the order given; the caller is responsible for sorting by
/// `(hydro_id, stage_id)` if canonical ordering is required.
///
/// The parent directory is created if it does not already exist. The write is
/// atomic: data goes to `{path}.tmp` first, then the file is renamed to `path`.
///
/// # Errors
///
/// - [`OutputError::IoError`] — directory creation, file open, or rename fails.
/// - [`OutputError::SerializationError`] — Arrow/Parquet construction fails.
///
/// # Examples
///
/// ```no_run
/// use cobre_io::output::stochastic::write_inflow_annual_component;
/// use cobre_io::scenarios::InflowAnnualComponentRow;
/// use cobre_core::EntityId;
/// use std::path::Path;
///
/// # fn main() -> Result<(), cobre_io::OutputError> {
/// let rows = vec![
///     InflowAnnualComponentRow {
///         hydro_id: EntityId::from(1),
///         stage_id: 0,
///         annual_coefficient: -0.5,
///         annual_mean_m3s: 1500.0,
///         annual_std_m3s: 300.0,
///     },
/// ];
/// write_inflow_annual_component(
///     Path::new("/tmp/out/stochastic/inflow_annual_component.parquet"),
///     &rows,
/// )?;
/// # Ok(())
/// # }
/// ```
pub fn write_inflow_annual_component(
    path: &Path,
    rows: &[InflowAnnualComponentRow],
) -> Result<(), OutputError> {
    ensure_parent_dir(path)?;
    let config = ParquetWriterConfig::default();
    let batch = build_inflow_annual_component_batch(rows)?;
    write_parquet_atomic(path, &batch, &config)
}

// ── Intermediate serde types for correlation JSON output ──────────────────────

/// Top-level serialization type for `correlation.json` output.
///
/// Uses the same field names and structure as the input format so that
/// copying the output file back to `scenarios/` produces an identical model.
/// The `$schema` field is intentionally omitted — it is informational and not
/// required by [`crate::scenarios::parse_correlation`].
#[derive(Serialize)]
struct WriteCorrelationFile {
    method: String,
    profiles: BTreeMap<String, WriteProfile>,
    schedule: Vec<WriteScheduleEntry>,
}

/// Serialization type for a single named correlation profile.
///
/// Uses `correlation_groups` to match the input JSON field name. The
/// `cobre-core` type uses `groups` internally, so this intermediate type
/// performs the rename at the serialization boundary.
#[derive(Serialize)]
struct WriteProfile {
    correlation_groups: Vec<WriteCorrelationGroup>,
}

/// Serialization type for a single correlation group.
#[derive(Serialize)]
struct WriteCorrelationGroup {
    name: String,
    entities: Vec<WriteEntity>,
    matrix: Vec<Vec<f64>>,
}

/// Serialization type for a single entity reference.
///
/// Uses `#[serde(rename = "type")]` to match the input JSON field name.
/// The `cobre-core` type stores the field as `entity_type`, but the input
/// schema uses `"type"`.
#[derive(Serialize)]
struct WriteEntity {
    #[serde(rename = "type")]
    entity_type: String,
    id: i32,
}

/// Serialization type for a single schedule entry.
#[derive(Serialize)]
struct WriteScheduleEntry {
    stage_id: i32,
    profile_name: String,
}

// ── Correlation JSON writer ───────────────────────────────────────────────────

/// Convert a [`CorrelationModel`] to its intermediate write representation.
///
/// Profiles are iterated in [`BTreeMap`] order (alphabetical), which preserves
/// declaration-order invariance across all callers.
fn to_write_format(model: &CorrelationModel) -> WriteCorrelationFile {
    let profiles: BTreeMap<String, WriteProfile> = model
        .profiles
        .iter()
        .map(|(name, profile)| {
            let groups: Vec<WriteCorrelationGroup> = profile
                .groups
                .iter()
                .map(|group| {
                    let entities: Vec<WriteEntity> = group
                        .entities
                        .iter()
                        .map(|entity| WriteEntity {
                            entity_type: entity.entity_type.clone(),
                            id: entity.id.0,
                        })
                        .collect();
                    WriteCorrelationGroup {
                        name: group.name.clone(),
                        entities,
                        matrix: group.matrix.clone(),
                    }
                })
                .collect();
            (
                name.clone(),
                WriteProfile {
                    correlation_groups: groups,
                },
            )
        })
        .collect();

    let schedule: Vec<WriteScheduleEntry> = model
        .schedule
        .iter()
        .map(|entry: &CorrelationScheduleEntry| WriteScheduleEntry {
            stage_id: entry.stage_id,
            profile_name: entry.profile_name.clone(),
        })
        .collect();

    WriteCorrelationFile {
        method: model.method.clone(),
        profiles,
        schedule,
    }
}

/// Write a [`CorrelationModel`] to a pretty-printed JSON file at `path`.
///
/// The output format matches `scenarios/correlation.json` exactly, using
/// `"correlation_groups"` (not `"groups"`) and `"type"` (not `"entity_type"`)
/// as field names, so that copying the output back to `scenarios/` produces
/// a round-trip-identical model when parsed by [`crate::scenarios::parse_correlation`].
///
/// The `$schema` field is omitted — it is informational and not required by
/// the parser. Profiles appear in alphabetical order (`BTreeMap` iteration
/// order). Schedule entries preserve the order given in the model.
///
/// The parent directory is created if it does not already exist. The write is
/// atomic: data goes to `{path}.tmp` first, then the file is renamed to
/// `path`. A partial `.tmp` file may remain on disk if the process is killed
/// mid-write, but the final path is never partially written.
///
/// # Errors
///
/// - [`OutputError::IoError`] — directory creation, file open, or rename fails.
/// - [`OutputError::SerializationError`] — JSON serialization fails.
///
/// # Examples
///
/// ```no_run
/// use cobre_io::output::stochastic::write_correlation_json;
/// use cobre_core::scenario::CorrelationModel;
/// use std::path::Path;
///
/// # fn main() -> Result<(), cobre_io::OutputError> {
/// let model = CorrelationModel::default();
/// write_correlation_json(Path::new("/tmp/out/stochastic/correlation.json"), &model)?;
/// # Ok(())
/// # }
/// ```
pub fn write_correlation_json(path: &Path, model: &CorrelationModel) -> Result<(), OutputError> {
    ensure_parent_dir(path)?;
    let write_model = to_write_format(model);
    let bytes = serde_json::to_vec_pretty(&write_model)
        .map_err(|e| OutputError::serialization("correlation_json", e.to_string()))?;
    write_bytes_atomic(path, &bytes)
}

// ── Load seasonal stats Parquet writer ───────────────────────────────────────

/// Write a slice of [`LoadSeasonalStatsRow`] to a Parquet file at `path`.
///
/// The output schema is exactly 4 columns — `bus_id: Int32`, `stage_id: Int32`,
/// `mean_mw: Float64`, `std_mw: Float64` — in that order, matching the schema
/// of `scenarios/load_seasonal_stats.parquet`. Rows are written in the order
/// given; the caller is responsible for sorting if canonical ordering is required.
///
/// The parent directory is created if it does not already exist. The write is
/// atomic: data goes to `{path}.tmp` first, then the file is renamed to `path`.
///
/// # Errors
///
/// - [`OutputError::IoError`] — directory creation, file open, or rename fails.
/// - [`OutputError::SerializationError`] — Arrow/Parquet construction fails.
///
/// # Examples
///
/// ```no_run
/// use cobre_io::output::stochastic::write_load_seasonal_stats;
/// use cobre_io::scenarios::LoadSeasonalStatsRow;
/// use cobre_core::EntityId;
/// use std::path::Path;
///
/// # fn main() -> Result<(), cobre_io::OutputError> {
/// let rows = vec![
///     LoadSeasonalStatsRow {
///         bus_id: EntityId::from(1),
///         stage_id: 0,
///         mean_mw: 500.0,
///         std_mw: 50.0,
///     },
/// ];
/// write_load_seasonal_stats(
///     Path::new("/tmp/out/stochastic/load_seasonal_stats.parquet"),
///     &rows,
/// )?;
/// # Ok(())
/// # }
/// ```
pub fn write_load_seasonal_stats(
    path: &Path,
    rows: &[LoadSeasonalStatsRow],
) -> Result<(), OutputError> {
    ensure_parent_dir(path)?;
    let config = ParquetWriterConfig::default();
    let batch = build_load_seasonal_stats_batch(rows)?;
    write_parquet_atomic(path, &batch, &config)
}

// ── Shared helpers ────────────────────────────────────────────────────────────

/// Ensure the parent directory of `path` exists, creating it if necessary.
pub(crate) fn ensure_parent_dir(path: &Path) -> Result<(), OutputError> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| OutputError::io(parent, e))?;
    }
    Ok(())
}

// ── Schema builders ───────────────────────────────────────────────────────────

fn noise_openings_schema() -> Schema {
    Schema::new(vec![
        Field::new("stage_id", DataType::Int32, false),
        Field::new("opening_index", DataType::UInt32, false),
        Field::new("entity_index", DataType::UInt32, false),
        Field::new("value", DataType::Float64, false),
    ])
}

#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
fn build_noise_openings_batch(tree: &OpeningTree) -> Result<RecordBatch, OutputError> {
    let n_rows: usize = (0..tree.n_stages())
        .map(|s| tree.n_openings(s))
        .sum::<usize>()
        * tree.dim();

    let mut stage_id_col = Int32Builder::with_capacity(n_rows);
    let mut opening_index_col = UInt32Builder::with_capacity(n_rows);
    let mut entity_index_col = UInt32Builder::with_capacity(n_rows);
    let mut value_col = Float64Builder::with_capacity(n_rows);

    for stage in 0..tree.n_stages() {
        let stage_i32 = stage as i32;
        for opening_idx in 0..tree.n_openings(stage) {
            let opening_u32 = opening_idx as u32;
            let noise = tree.opening(stage, opening_idx);
            for (entity_idx, &v) in noise.iter().enumerate() {
                stage_id_col.append_value(stage_i32);
                opening_index_col.append_value(opening_u32);
                entity_index_col.append_value(entity_idx as u32);
                value_col.append_value(v);
            }
        }
    }

    RecordBatch::try_new(
        Arc::new(noise_openings_schema()),
        vec![
            Arc::new(stage_id_col.finish()),
            Arc::new(opening_index_col.finish()),
            Arc::new(entity_index_col.finish()),
            Arc::new(value_col.finish()),
        ],
    )
    .map_err(|e| OutputError::serialization("noise_openings", e.to_string()))
}

fn inflow_seasonal_stats_schema() -> Schema {
    Schema::new(vec![
        Field::new("hydro_id", DataType::Int32, false),
        Field::new("stage_id", DataType::Int32, false),
        Field::new("mean_m3s", DataType::Float64, false),
        Field::new("std_m3s", DataType::Float64, false),
    ])
}

#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
fn build_inflow_seasonal_stats_batch(
    rows: &[InflowSeasonalStatsRow],
) -> Result<RecordBatch, OutputError> {
    let n = rows.len();
    let mut hydro_id_col = Int32Builder::with_capacity(n);
    let mut stage_id_col = Int32Builder::with_capacity(n);
    let mut mean_m3s_col = Float64Builder::with_capacity(n);
    let mut std_m3s_col = Float64Builder::with_capacity(n);

    for row in rows {
        hydro_id_col.append_value(row.hydro_id.0);
        stage_id_col.append_value(row.stage_id);
        mean_m3s_col.append_value(row.mean_m3s);
        std_m3s_col.append_value(row.std_m3s);
    }

    RecordBatch::try_new(
        Arc::new(inflow_seasonal_stats_schema()),
        vec![
            Arc::new(hydro_id_col.finish()),
            Arc::new(stage_id_col.finish()),
            Arc::new(mean_m3s_col.finish()),
            Arc::new(std_m3s_col.finish()),
        ],
    )
    .map_err(|e| OutputError::serialization("inflow_seasonal_stats", e.to_string()))
}

fn inflow_ar_coefficients_schema() -> Schema {
    Schema::new(vec![
        Field::new("hydro_id", DataType::Int32, false),
        Field::new("stage_id", DataType::Int32, false),
        Field::new("lag", DataType::Int32, false),
        Field::new("coefficient", DataType::Float64, false),
        Field::new("residual_std_ratio", DataType::Float64, false),
    ])
}

#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
fn build_inflow_ar_coefficients_batch(
    rows: &[InflowArCoefficientRow],
) -> Result<RecordBatch, OutputError> {
    let n = rows.len();
    let mut hydro_id_col = Int32Builder::with_capacity(n);
    let mut stage_id_col = Int32Builder::with_capacity(n);
    let mut lag_col = Int32Builder::with_capacity(n);
    let mut coefficient_col = Float64Builder::with_capacity(n);
    let mut residual_std_ratio_col = Float64Builder::with_capacity(n);

    for row in rows {
        hydro_id_col.append_value(row.hydro_id.0);
        stage_id_col.append_value(row.stage_id);
        lag_col.append_value(row.lag);
        coefficient_col.append_value(row.coefficient);
        residual_std_ratio_col.append_value(row.residual_std_ratio);
    }

    RecordBatch::try_new(
        Arc::new(inflow_ar_coefficients_schema()),
        vec![
            Arc::new(hydro_id_col.finish()),
            Arc::new(stage_id_col.finish()),
            Arc::new(lag_col.finish()),
            Arc::new(coefficient_col.finish()),
            Arc::new(residual_std_ratio_col.finish()),
        ],
    )
    .map_err(|e| OutputError::serialization("inflow_ar_coefficients", e.to_string()))
}

fn inflow_annual_component_schema() -> Schema {
    Schema::new(vec![
        Field::new("hydro_id", DataType::Int32, false),
        Field::new("stage_id", DataType::Int32, false),
        Field::new("annual_coefficient", DataType::Float64, false),
        Field::new("annual_mean_m3s", DataType::Float64, false),
        Field::new("annual_std_m3s", DataType::Float64, false),
    ])
}

#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
fn build_inflow_annual_component_batch(
    rows: &[InflowAnnualComponentRow],
) -> Result<RecordBatch, OutputError> {
    let n = rows.len();
    let mut hydro_id_col = Int32Builder::with_capacity(n);
    let mut stage_id_col = Int32Builder::with_capacity(n);
    let mut annual_coefficient_col = Float64Builder::with_capacity(n);
    let mut annual_mean_m3s_col = Float64Builder::with_capacity(n);
    let mut annual_std_m3s_col = Float64Builder::with_capacity(n);

    for row in rows {
        hydro_id_col.append_value(row.hydro_id.0);
        stage_id_col.append_value(row.stage_id);
        annual_coefficient_col.append_value(row.annual_coefficient);
        annual_mean_m3s_col.append_value(row.annual_mean_m3s);
        annual_std_m3s_col.append_value(row.annual_std_m3s);
    }

    RecordBatch::try_new(
        Arc::new(inflow_annual_component_schema()),
        vec![
            Arc::new(hydro_id_col.finish()),
            Arc::new(stage_id_col.finish()),
            Arc::new(annual_coefficient_col.finish()),
            Arc::new(annual_mean_m3s_col.finish()),
            Arc::new(annual_std_m3s_col.finish()),
        ],
    )
    .map_err(|e| OutputError::serialization("inflow_annual_component", e.to_string()))
}

fn load_seasonal_stats_schema() -> Schema {
    Schema::new(vec![
        Field::new("bus_id", DataType::Int32, false),
        Field::new("stage_id", DataType::Int32, false),
        Field::new("mean_mw", DataType::Float64, false),
        Field::new("std_mw", DataType::Float64, false),
    ])
}

#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
fn build_load_seasonal_stats_batch(
    rows: &[LoadSeasonalStatsRow],
) -> Result<RecordBatch, OutputError> {
    let n = rows.len();
    let mut bus_id_col = Int32Builder::with_capacity(n);
    let mut stage_id_col = Int32Builder::with_capacity(n);
    let mut mean_mw_col = Float64Builder::with_capacity(n);
    let mut std_mw_col = Float64Builder::with_capacity(n);

    for row in rows {
        bus_id_col.append_value(row.bus_id.0);
        stage_id_col.append_value(row.stage_id);
        mean_mw_col.append_value(row.mean_mw);
        std_mw_col.append_value(row.std_mw);
    }

    RecordBatch::try_new(
        Arc::new(load_seasonal_stats_schema()),
        vec![
            Arc::new(bus_id_col.finish()),
            Arc::new(stage_id_col.finish()),
            Arc::new(mean_mw_col.finish()),
            Arc::new(std_mw_col.finish()),
        ],
    )
    .map_err(|e| OutputError::serialization("load_seasonal_stats", e.to_string()))
}

// ── Fitting report types and writer ──────────────────────────────────────────

/// Per-hydro entry in a diagnostic fitting report.
///
/// Captures the AIC-selected AR order, the full AIC score vector (one entry
/// per candidate order), and the per-season AR coefficients produced by the
/// fitting step.
///
/// The `coefficients` field is a nested array: one row per season, each row
/// containing the AR lag coefficients for that season in lag order.
#[derive(Debug, Clone, serde::Serialize)]
#[cfg_attr(test, derive(serde::Deserialize))]
pub struct HydroFittingEntry {
    /// Selected AR order for this hydro plant.
    pub selected_order: u32,
    /// Per-season AR coefficients.
    ///
    /// `coefficients[s]` contains the AR lag coefficients for season `s`,
    /// with `coefficients[s][k]` being the coefficient for lag `k + 1`.
    pub coefficients: Vec<Vec<f64>>,
    /// Contribution-based order reductions applied during fitting.
    ///
    /// Each entry documents a season where the initial order was reduced
    /// due to negative contributions from the recursive composition analysis.
    /// Empty when no reductions were needed.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub contribution_reductions: Vec<FittingReductionEntry>,
}

/// A single order reduction event in the fitting report.
///
/// Records that a season's AR order was reduced during the estimation
/// pipeline. The `reason` field identifies the mechanism that triggered
/// the reduction.
#[derive(Debug, Clone, serde::Serialize)]
#[cfg_attr(test, derive(serde::Deserialize))]
pub struct FittingReductionEntry {
    /// Season where the reduction occurred.
    pub season_id: usize,
    /// Order before reduction.
    pub original_order: usize,
    /// Order after reduction.
    pub reduced_order: usize,
    /// Contribution values at the original order that triggered the reduction.
    pub contributions: Vec<f64>,
    /// The mechanism that triggered this reduction.
    ///
    /// One of: `"magnitude_bound"`, `"phi1_negative"`, `"negative_contribution"`.
    #[serde(default)]
    pub reason: String,
}

/// Diagnostic fitting report produced after the AR order selection step.
///
/// Contains one [`HydroFittingEntry`] per hydro plant that was fitted.
/// Keys are hydro IDs serialized as strings (e.g., `"1"`, `"5"`) so that
/// JSON output is readable without a custom serializer. The `BTreeMap`
/// ensures hydro entries appear in ascending key order regardless of
/// insertion order.
///
/// This type is write-only: `fitting_report.json` is a diagnostic artifact
/// and is not consumed as input on subsequent runs.
#[derive(Debug, Clone, serde::Serialize)]
#[cfg_attr(test, derive(serde::Deserialize))]
pub struct FittingReport {
    /// Map from hydro ID string to per-hydro fitting diagnostics.
    pub hydros: BTreeMap<String, HydroFittingEntry>,
}

/// Write a [`FittingReport`] to a pretty-printed JSON file at `path`.
///
/// The output matches the fitting report schema. Hydro IDs appear as
/// string keys in ascending sort order (`BTreeMap` iteration order). An empty
/// report produces `{"hydros":{}}`.
///
/// The parent directory is created if it does not already exist. The write is
/// atomic: data goes to `{path}.tmp` first, then the file is renamed to
/// `path`. A partial `.tmp` file may remain on disk if the process is killed
/// mid-write, but the final path is never partially written.
///
/// # Errors
///
/// - [`OutputError::IoError`] — directory creation, file open, or rename fails.
/// - [`OutputError::SerializationError`] — JSON serialization fails.
///
/// # Examples
///
/// ```no_run
/// use cobre_io::output::stochastic::{write_fitting_report, FittingReport, HydroFittingEntry};
/// use std::collections::BTreeMap;
/// use std::path::Path;
///
/// # fn main() -> Result<(), cobre_io::OutputError> {
/// let mut hydros = BTreeMap::new();
/// hydros.insert("1".to_string(), HydroFittingEntry {
///     selected_order: 3,
///     coefficients: vec![vec![0.42, -0.11, 0.07]],
///     contribution_reductions: Vec::new(),
/// });
/// let report = FittingReport { hydros };
/// write_fitting_report(Path::new("/tmp/out/stochastic/fitting_report.json"), &report)?;
/// # Ok(())
/// # }
/// ```
pub fn write_fitting_report(path: &Path, report: &FittingReport) -> Result<(), OutputError> {
    ensure_parent_dir(path)?;
    let bytes = serde_json::to_vec_pretty(report)
        .map_err(|e| OutputError::serialization("fitting_report", e.to_string()))?;
    write_bytes_atomic(path, &bytes)
}

/// Write bytes to `path` atomically via a `.tmp` intermediate file.
///
/// Creates `{path}.tmp`, writes `bytes`, then renames to `path`.
/// Parent directory must already exist before calling.
fn write_bytes_atomic(path: &Path, bytes: &[u8]) -> Result<(), OutputError> {
    let tmp_path = match path.extension() {
        Some(ext) => path.with_extension(format!("{}.tmp", ext.to_string_lossy())),
        None => path.with_extension("tmp"),
    };

    std::fs::write(&tmp_path, bytes).map_err(|e| OutputError::io(&tmp_path, e))?;
    std::fs::rename(&tmp_path, path).map_err(|e| OutputError::io(path, e))?;

    Ok(())
}

pub(crate) fn write_parquet_atomic(
    path: &Path,
    batch: &RecordBatch,
    config: &ParquetWriterConfig,
) -> Result<(), OutputError> {
    let tmp_path = match path.extension() {
        Some(ext) => path.with_extension(format!("{}.tmp", ext.to_string_lossy())),
        None => path.with_extension("tmp"),
    };

    let props = WriterProperties::builder()
        .set_compression(config.compression)
        .set_max_row_group_row_count(Some(config.row_group_size))
        .set_dictionary_enabled(config.dictionary_encoding)
        .build();

    let file = std::fs::File::create(&tmp_path).map_err(|e| OutputError::io(&tmp_path, e))?;
    let mut writer = ArrowWriter::try_new(file, batch.schema(), Some(props))
        .map_err(|e| OutputError::serialization("parquet_writer", e.to_string()))?;

    writer
        .write(batch)
        .and_then(|()| writer.close())
        .map_err(|e| OutputError::serialization("parquet_writer", e.to_string()))?;

    std::fs::rename(&tmp_path, path).map_err(|e| OutputError::io(path, e))?;
    Ok(())
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::float_cmp,
    clippy::cast_possible_truncation
)]
mod tests {
    use super::*;
    use cobre_core::EntityId;
    use cobre_stochastic::OpeningTree;

    fn make_tree_2s_2d() -> OpeningTree {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        OpeningTree::from_parts(data, vec![2, 3], 2)
    }

    // -------------------------------------------------------------------------
    // write_then_read_round_trips
    // -------------------------------------------------------------------------

    #[test]
    fn write_then_read_round_trips() {
        use crate::scenarios::{NoiseOpeningRow, assemble_opening_tree, parse_noise_openings};

        let tree = make_tree_2s_2d();
        let tmp = tempfile::tempdir().expect("tempdir must succeed");
        let path = tmp.path().join("noise_openings.parquet");

        write_noise_openings(&path, &tree).expect("write must succeed");
        assert!(path.exists(), "file must exist after write");

        let rows: Vec<NoiseOpeningRow> = parse_noise_openings(&path).expect("parse must succeed");
        let recovered = assemble_opening_tree(rows, tree.dim());

        assert_eq!(
            recovered.n_stages(),
            tree.n_stages(),
            "n_stages must be identical"
        );
        assert_eq!(recovered.dim(), tree.dim(), "dim must be identical");
        assert_eq!(
            recovered.openings_per_stage_slice(),
            tree.openings_per_stage_slice(),
            "openings_per_stage_slice must be identical"
        );
        assert_eq!(
            recovered.data(),
            tree.data(),
            "data arrays must be bit-for-bit identical"
        );
    }

    // -------------------------------------------------------------------------
    // write_creates_parent_directory
    // -------------------------------------------------------------------------

    #[test]
    fn write_creates_parent_directory() {
        let tree = make_tree_2s_2d();
        let tmp = tempfile::tempdir().expect("tempdir must succeed");
        // Nested path — neither intermediate directory exists yet.
        let path = tmp.path().join("output/stochastic/noise_openings.parquet");

        assert!(
            !path.parent().unwrap().exists(),
            "parent must not exist yet"
        );
        write_noise_openings(&path, &tree).expect("write must succeed even with missing parent");
        assert!(path.exists(), "file must exist");
        assert!(
            path.parent().unwrap().is_dir(),
            "parent directory must have been created"
        );
    }

    // -------------------------------------------------------------------------
    // write_correct_schema
    // -------------------------------------------------------------------------

    #[test]
    fn write_correct_schema() {
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

        let tree = make_tree_2s_2d();
        let tmp = tempfile::tempdir().expect("tempdir must succeed");
        let path = tmp.path().join("noise_openings.parquet");

        write_noise_openings(&path, &tree).expect("write must succeed");

        let file = std::fs::File::open(&path).expect("file must open");
        let builder =
            ParquetRecordBatchReaderBuilder::try_new(file).expect("builder must be created");
        let schema = builder.schema().clone();

        assert_eq!(
            schema.fields().len(),
            4,
            "schema must have exactly 4 fields"
        );

        let fields: Vec<(&str, &DataType)> = schema
            .fields()
            .iter()
            .map(|f| (f.name().as_str(), f.data_type()))
            .collect();

        assert_eq!(fields[0], ("stage_id", &DataType::Int32));
        assert_eq!(fields[1], ("opening_index", &DataType::UInt32));
        assert_eq!(fields[2], ("entity_index", &DataType::UInt32));
        assert_eq!(fields[3], ("value", &DataType::Float64));
    }

    // -------------------------------------------------------------------------
    // write_row_count_matches_tree
    // -------------------------------------------------------------------------

    #[test]
    fn write_row_count_matches_tree() {
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

        let tree = make_tree_2s_2d();
        // Expected rows: (2 + 3) * 2 = 10
        let expected_rows: usize =
            tree.openings_per_stage_slice().iter().sum::<usize>() * tree.dim();

        let tmp = tempfile::tempdir().expect("tempdir must succeed");
        let path = tmp.path().join("noise_openings.parquet");

        write_noise_openings(&path, &tree).expect("write must succeed");

        let file = std::fs::File::open(&path).expect("file must open");
        let reader = ParquetRecordBatchReaderBuilder::try_new(file)
            .expect("builder")
            .build()
            .expect("reader");

        let total_rows: usize = reader
            .map(|b| b.expect("batch must be Ok").num_rows())
            .sum();

        assert_eq!(
            total_rows, expected_rows,
            "row count must equal sum(openings_per_stage) * dim"
        );
    }

    // -------------------------------------------------------------------------
    // Atomic write: .tmp file must not remain after success
    // -------------------------------------------------------------------------

    #[test]
    fn write_atomic_no_tmp_file_remains() {
        let tree = make_tree_2s_2d();
        let tmp = tempfile::tempdir().expect("tempdir must succeed");
        let path = tmp.path().join("noise_openings.parquet");

        write_noise_openings(&path, &tree).expect("write must succeed");

        let tmp_path = path.with_extension("parquet.tmp");
        assert!(
            !tmp_path.exists(),
            ".tmp file must not remain after successful atomic write"
        );
        assert!(path.exists(), "final file must exist");
    }

    // -------------------------------------------------------------------------
    // Correct (stage_id, opening_index, entity_index, value) tuples
    // -------------------------------------------------------------------------

    #[test]
    fn write_correct_row_tuples() {
        use arrow::array::{Float64Array, Int32Array, UInt32Array};
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

        let tree = make_tree_2s_2d();
        let tmp = tempfile::tempdir().expect("tempdir must succeed");
        let path = tmp.path().join("noise_openings.parquet");

        write_noise_openings(&path, &tree).expect("write must succeed");

        let file = std::fs::File::open(&path).expect("file must open");
        let mut reader = ParquetRecordBatchReaderBuilder::try_new(file)
            .expect("builder")
            .build()
            .expect("reader");
        let batch = reader.next().expect("must have a batch").expect("batch Ok");

        let stage_col = batch
            .column_by_name("stage_id")
            .unwrap()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        let opening_col = batch
            .column_by_name("opening_index")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap();
        let entity_col = batch
            .column_by_name("entity_index")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap();
        let value_col = batch
            .column_by_name("value")
            .unwrap()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();

        // Row 0: stage=0, opening=0, entity=0, value=1.0
        assert_eq!(stage_col.value(0), 0);
        assert_eq!(opening_col.value(0), 0);
        assert_eq!(entity_col.value(0), 0);
        assert_eq!(value_col.value(0), 1.0);

        // Row 1: stage=0, opening=0, entity=1, value=2.0
        assert_eq!(stage_col.value(1), 0);
        assert_eq!(opening_col.value(1), 0);
        assert_eq!(entity_col.value(1), 1);
        assert_eq!(value_col.value(1), 2.0);

        // Row 2: stage=0, opening=1, entity=0, value=3.0
        assert_eq!(stage_col.value(2), 0);
        assert_eq!(opening_col.value(2), 1);
        assert_eq!(entity_col.value(2), 0);
        assert_eq!(value_col.value(2), 3.0);

        // Row 4: stage=1, opening=0, entity=0, value=5.0
        assert_eq!(stage_col.value(4), 1);
        assert_eq!(opening_col.value(4), 0);
        assert_eq!(entity_col.value(4), 0);
        assert_eq!(value_col.value(4), 5.0);

        // Last row (9): stage=1, opening=2, entity=1, value=10.0
        assert_eq!(stage_col.value(9), 1);
        assert_eq!(opening_col.value(9), 2);
        assert_eq!(entity_col.value(9), 1);
        assert_eq!(value_col.value(9), 10.0);
    }

    // -------------------------------------------------------------------------
    // Empty tree (0 stages) writes a valid zero-row file
    // -------------------------------------------------------------------------

    #[test]
    fn write_empty_tree_zero_rows() {
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

        let tree = OpeningTree::from_parts(vec![], vec![], 2);
        let tmp = tempfile::tempdir().expect("tempdir must succeed");
        let path = tmp.path().join("noise_openings.parquet");

        write_noise_openings(&path, &tree).expect("write must succeed for empty tree");

        let file = std::fs::File::open(&path).expect("file must open");
        let reader = ParquetRecordBatchReaderBuilder::try_new(file)
            .expect("builder")
            .build()
            .expect("reader");

        let total_rows: usize = reader
            .map(|b| b.expect("batch must be Ok").num_rows())
            .sum();
        assert_eq!(total_rows, 0, "empty tree must produce 0-row file");
    }

    // =========================================================================
    // write_inflow_seasonal_stats tests
    // =========================================================================

    fn make_inflow_stats_rows() -> Vec<InflowSeasonalStatsRow> {
        vec![
            InflowSeasonalStatsRow {
                hydro_id: EntityId::from(1),
                stage_id: 0,
                mean_m3s: 150.0,
                std_m3s: 30.0,
            },
            InflowSeasonalStatsRow {
                hydro_id: EntityId::from(1),
                stage_id: 1,
                mean_m3s: 160.0,
                std_m3s: 32.0,
            },
            InflowSeasonalStatsRow {
                hydro_id: EntityId::from(3),
                stage_id: 0,
                mean_m3s: 180.0,
                std_m3s: 35.0,
            },
            InflowSeasonalStatsRow {
                hydro_id: EntityId::from(3),
                stage_id: 1,
                mean_m3s: 200.0,
                std_m3s: 40.0,
            },
        ]
    }

    // -------------------------------------------------------------------------
    // write_then_read_inflow_stats_round_trips
    // -------------------------------------------------------------------------

    #[test]
    fn write_then_read_inflow_stats_round_trips() {
        use crate::scenarios::parse_inflow_seasonal_stats;

        let rows = make_inflow_stats_rows();
        let tmp = tempfile::tempdir().expect("tempdir must succeed");
        let path = tmp.path().join("inflow_seasonal_stats.parquet");

        write_inflow_seasonal_stats(&path, &rows).expect("write must succeed");
        assert!(path.exists(), "file must exist after write");

        let recovered = parse_inflow_seasonal_stats(&path).expect("parse must succeed after write");

        assert_eq!(recovered.len(), rows.len(), "row count must match");
        for (original, parsed) in rows.iter().zip(recovered.iter()) {
            assert_eq!(parsed.hydro_id, original.hydro_id, "hydro_id must match");
            assert_eq!(parsed.stage_id, original.stage_id, "stage_id must match");
            assert!(
                (parsed.mean_m3s - original.mean_m3s).abs() < 1e-10,
                "mean_m3s must be bit-for-bit identical"
            );
            assert!(
                (parsed.std_m3s - original.std_m3s).abs() < 1e-10,
                "std_m3s must be bit-for-bit identical"
            );
        }
    }

    // -------------------------------------------------------------------------
    // write_inflow_stats_creates_parent_directory
    // -------------------------------------------------------------------------

    #[test]
    fn write_inflow_stats_creates_parent_directory() {
        let rows = make_inflow_stats_rows();
        let tmp = tempfile::tempdir().expect("tempdir must succeed");
        let path = tmp
            .path()
            .join("output/stochastic/inflow_seasonal_stats.parquet");

        assert!(
            !path.parent().unwrap().exists(),
            "parent must not exist yet"
        );
        write_inflow_seasonal_stats(&path, &rows)
            .expect("write must succeed even with missing parent");
        assert!(path.exists(), "file must exist");
        assert!(
            path.parent().unwrap().is_dir(),
            "parent directory must have been created"
        );
    }

    // -------------------------------------------------------------------------
    // write_inflow_stats_empty_rows_valid_schema
    // -------------------------------------------------------------------------

    #[test]
    fn write_inflow_stats_empty_rows_valid_schema() {
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

        let tmp = tempfile::tempdir().expect("tempdir must succeed");
        let path = tmp.path().join("inflow_seasonal_stats.parquet");

        write_inflow_seasonal_stats(&path, &[]).expect("write must succeed for empty rows");

        let file = std::fs::File::open(&path).expect("file must open");
        let builder =
            ParquetRecordBatchReaderBuilder::try_new(file).expect("builder must be created");
        let schema = builder.schema().clone();
        let reader = builder.build().expect("reader must be created");

        let total_rows: usize = reader
            .map(|b| b.expect("batch must be Ok").num_rows())
            .sum();
        assert_eq!(total_rows, 0, "empty rows must produce 0-row file");
        assert_eq!(
            schema.fields().len(),
            4,
            "schema must have exactly 4 columns"
        );

        let fields: Vec<(&str, &DataType)> = schema
            .fields()
            .iter()
            .map(|f| (f.name().as_str(), f.data_type()))
            .collect();
        assert_eq!(fields[0], ("hydro_id", &DataType::Int32));
        assert_eq!(fields[1], ("stage_id", &DataType::Int32));
        assert_eq!(fields[2], ("mean_m3s", &DataType::Float64));
        assert_eq!(fields[3], ("std_m3s", &DataType::Float64));
    }

    // -------------------------------------------------------------------------
    // write_inflow_stats_no_tmp_file_remains
    // -------------------------------------------------------------------------

    #[test]
    fn write_inflow_stats_no_tmp_file_remains() {
        let rows = make_inflow_stats_rows();
        let tmp = tempfile::tempdir().expect("tempdir must succeed");
        let path = tmp.path().join("inflow_seasonal_stats.parquet");

        write_inflow_seasonal_stats(&path, &rows).expect("write must succeed");

        let tmp_path = path.with_extension("parquet.tmp");
        assert!(
            !tmp_path.exists(),
            ".tmp file must not remain after successful atomic write"
        );
        assert!(path.exists(), "final file must exist");
    }

    // =========================================================================
    // write_inflow_ar_coefficients tests
    // =========================================================================

    fn make_ar_coefficient_rows() -> Vec<InflowArCoefficientRow> {
        vec![
            InflowArCoefficientRow {
                hydro_id: EntityId::from(1),
                stage_id: 0,
                lag: 1,
                coefficient: 0.45,
                residual_std_ratio: 0.85,
            },
            InflowArCoefficientRow {
                hydro_id: EntityId::from(1),
                stage_id: 0,
                lag: 2,
                coefficient: 0.20,
                residual_std_ratio: 0.85,
            },
            InflowArCoefficientRow {
                hydro_id: EntityId::from(1),
                stage_id: 0,
                lag: 3,
                coefficient: 0.10,
                residual_std_ratio: 0.85,
            },
            InflowArCoefficientRow {
                hydro_id: EntityId::from(2),
                stage_id: 0,
                lag: 1,
                coefficient: 0.30,
                residual_std_ratio: 0.75,
            },
            InflowArCoefficientRow {
                hydro_id: EntityId::from(2),
                stage_id: 0,
                lag: 2,
                coefficient: 0.15,
                residual_std_ratio: 0.75,
            },
            InflowArCoefficientRow {
                hydro_id: EntityId::from(2),
                stage_id: 0,
                lag: 3,
                coefficient: 0.05,
                residual_std_ratio: 0.75,
            },
        ]
    }

    // -------------------------------------------------------------------------
    // write_then_read_ar_coefficients_round_trips
    // -------------------------------------------------------------------------

    #[test]
    fn write_then_read_ar_coefficients_round_trips() {
        use crate::scenarios::parse_inflow_ar_coefficients;

        let rows = make_ar_coefficient_rows();
        let tmp = tempfile::tempdir().expect("tempdir must succeed");
        let path = tmp.path().join("inflow_ar_coefficients.parquet");

        write_inflow_ar_coefficients(&path, &rows).expect("write must succeed");
        assert!(path.exists(), "file must exist after write");

        let recovered =
            parse_inflow_ar_coefficients(&path).expect("parse must succeed after write");

        assert_eq!(recovered.len(), rows.len(), "row count must match");
        for (original, parsed) in rows.iter().zip(recovered.iter()) {
            assert_eq!(parsed.hydro_id, original.hydro_id, "hydro_id must match");
            assert_eq!(parsed.stage_id, original.stage_id, "stage_id must match");
            assert_eq!(parsed.lag, original.lag, "lag must match");
            assert!(
                (parsed.coefficient - original.coefficient).abs() < 1e-10,
                "coefficient must be bit-for-bit identical"
            );
            assert!(
                (parsed.residual_std_ratio - original.residual_std_ratio).abs() < 1e-10,
                "residual_std_ratio must be bit-for-bit identical"
            );
        }
    }

    // -------------------------------------------------------------------------
    // write_ar_coefficients_creates_parent_directory
    // -------------------------------------------------------------------------

    #[test]
    fn write_ar_coefficients_creates_parent_directory() {
        let rows = make_ar_coefficient_rows();
        let tmp = tempfile::tempdir().expect("tempdir must succeed");
        let path = tmp
            .path()
            .join("output/stochastic/inflow_ar_coefficients.parquet");

        assert!(
            !path.parent().unwrap().exists(),
            "parent must not exist yet"
        );
        write_inflow_ar_coefficients(&path, &rows)
            .expect("write must succeed even with missing parent");
        assert!(path.exists(), "file must exist");
        assert!(
            path.parent().unwrap().is_dir(),
            "parent directory must have been created"
        );
    }

    // -------------------------------------------------------------------------
    // write_ar_coefficients_empty_rows_valid_schema
    // -------------------------------------------------------------------------

    #[test]
    fn write_ar_coefficients_empty_rows_valid_schema() {
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

        let tmp = tempfile::tempdir().expect("tempdir must succeed");
        let path = tmp.path().join("inflow_ar_coefficients.parquet");

        write_inflow_ar_coefficients(&path, &[]).expect("write must succeed for empty rows");

        let file = std::fs::File::open(&path).expect("file must open");
        let builder =
            ParquetRecordBatchReaderBuilder::try_new(file).expect("builder must be created");
        let schema = builder.schema().clone();
        let reader = builder.build().expect("reader must be created");

        let total_rows: usize = reader
            .map(|b| b.expect("batch must be Ok").num_rows())
            .sum();
        assert_eq!(total_rows, 0, "empty rows must produce 0-row file");
        assert_eq!(
            schema.fields().len(),
            5,
            "schema must have exactly 5 columns"
        );

        let fields: Vec<(&str, &DataType)> = schema
            .fields()
            .iter()
            .map(|f| (f.name().as_str(), f.data_type()))
            .collect();
        assert_eq!(fields[0], ("hydro_id", &DataType::Int32));
        assert_eq!(fields[1], ("stage_id", &DataType::Int32));
        assert_eq!(fields[2], ("lag", &DataType::Int32));
        assert_eq!(fields[3], ("coefficient", &DataType::Float64));
        assert_eq!(fields[4], ("residual_std_ratio", &DataType::Float64));
    }

    // -------------------------------------------------------------------------
    // write_ar_coefficients_no_tmp_file_remains
    // -------------------------------------------------------------------------

    #[test]
    fn write_ar_coefficients_no_tmp_file_remains() {
        let rows = make_ar_coefficient_rows();
        let tmp = tempfile::tempdir().expect("tempdir must succeed");
        let path = tmp.path().join("inflow_ar_coefficients.parquet");

        write_inflow_ar_coefficients(&path, &rows).expect("write must succeed");

        let tmp_path = path.with_extension("parquet.tmp");
        assert!(
            !tmp_path.exists(),
            ".tmp file must not remain after successful atomic write"
        );
        assert!(path.exists(), "final file must exist");
    }

    // =========================================================================
    // write_correlation_json tests
    // =========================================================================

    use cobre_core::scenario::{
        CorrelationEntity, CorrelationGroup, CorrelationModel, CorrelationProfile,
        CorrelationScheduleEntry,
    };
    use std::collections::BTreeMap;

    fn make_simple_correlation_model() -> CorrelationModel {
        let mut profiles = BTreeMap::new();
        profiles.insert(
            "default".to_string(),
            CorrelationProfile {
                groups: vec![CorrelationGroup {
                    name: "group_a".to_string(),
                    entities: vec![
                        CorrelationEntity {
                            entity_type: "inflow".to_string(),
                            id: EntityId::from(1),
                        },
                        CorrelationEntity {
                            entity_type: "inflow".to_string(),
                            id: EntityId::from(2),
                        },
                    ],
                    matrix: vec![vec![1.0, 0.75], vec![0.75, 1.0]],
                }],
            },
        );
        CorrelationModel {
            method: "spectral".to_string(),
            profiles,
            schedule: vec![],
        }
    }

    fn make_two_profile_correlation_model() -> CorrelationModel {
        let mut profiles = BTreeMap::new();
        profiles.insert(
            "default".to_string(),
            CorrelationProfile {
                groups: vec![CorrelationGroup {
                    name: "all".to_string(),
                    entities: vec![
                        CorrelationEntity {
                            entity_type: "inflow".to_string(),
                            id: EntityId::from(1),
                        },
                        CorrelationEntity {
                            entity_type: "inflow".to_string(),
                            id: EntityId::from(2),
                        },
                    ],
                    matrix: vec![vec![1.0, 0.5], vec![0.5, 1.0]],
                }],
            },
        );
        profiles.insert(
            "wet_season".to_string(),
            CorrelationProfile {
                groups: vec![CorrelationGroup {
                    name: "southeast".to_string(),
                    entities: vec![
                        CorrelationEntity {
                            entity_type: "inflow".to_string(),
                            id: EntityId::from(1),
                        },
                        CorrelationEntity {
                            entity_type: "inflow".to_string(),
                            id: EntityId::from(2),
                        },
                    ],
                    matrix: vec![vec![1.0, 0.9], vec![0.9, 1.0]],
                }],
            },
        );
        CorrelationModel {
            method: "spectral".to_string(),
            profiles,
            schedule: vec![
                CorrelationScheduleEntry {
                    stage_id: 0,
                    profile_name: "wet_season".to_string(),
                },
                CorrelationScheduleEntry {
                    stage_id: 6,
                    profile_name: "default".to_string(),
                },
            ],
        }
    }

    // -------------------------------------------------------------------------
    // write_then_read_correlation_json_round_trips_simple
    // -------------------------------------------------------------------------

    #[test]
    fn write_then_read_correlation_json_round_trips_simple() {
        use crate::scenarios::parse_correlation;

        let model = make_simple_correlation_model();
        let tmp = tempfile::tempdir().expect("tempdir must succeed");
        let path = tmp.path().join("correlation.json");

        write_correlation_json(&path, &model).expect("write must succeed");
        assert!(path.exists(), "file must exist after write");

        let recovered = parse_correlation(&path).expect("parse must succeed after write");

        assert_eq!(recovered.method, "spectral", "method must match");
        assert_eq!(recovered.profiles.len(), 1, "profiles.len must be 1");
        assert!(recovered.schedule.is_empty(), "schedule must be empty");

        let profile = &recovered.profiles["default"];
        assert_eq!(profile.groups.len(), 1);
        let group = &profile.groups[0];
        assert_eq!(group.entities.len(), 2);
        assert_eq!(group.entities[0].entity_type, "inflow");
        assert_eq!(group.entities[0].id, EntityId::from(1));
        assert_eq!(group.entities[1].entity_type, "inflow");
        assert_eq!(group.entities[1].id, EntityId::from(2));
        assert!((group.matrix[0][1] - 0.75).abs() < 1e-10);
        assert!((group.matrix[1][0] - 0.75).abs() < 1e-10);
    }

    // -------------------------------------------------------------------------
    // write_then_read_correlation_json_round_trips_with_schedule
    // -------------------------------------------------------------------------

    #[test]
    fn write_then_read_correlation_json_round_trips_with_schedule() {
        use crate::scenarios::parse_correlation;

        let model = make_two_profile_correlation_model();
        let tmp = tempfile::tempdir().expect("tempdir must succeed");
        let path = tmp.path().join("correlation.json");

        write_correlation_json(&path, &model).expect("write must succeed");

        let recovered = parse_correlation(&path).expect("parse must succeed after write");

        assert_eq!(recovered.profiles.len(), 2, "profiles.len must be 2");
        assert_eq!(recovered.schedule.len(), 2, "schedule.len must be 2");
        assert_eq!(recovered.method, "spectral");

        let keys: Vec<&String> = recovered.profiles.keys().collect();
        assert_eq!(keys[0], "default");
        assert_eq!(keys[1], "wet_season");

        assert_eq!(recovered.schedule[0].stage_id, 0);
        assert_eq!(recovered.schedule[0].profile_name, "wet_season");
        assert_eq!(recovered.schedule[1].stage_id, 6);
        assert_eq!(recovered.schedule[1].profile_name, "default");

        let default_group = &recovered.profiles["default"].groups[0];
        assert_eq!(default_group.name, "all");
        assert!((default_group.matrix[0][1] - 0.5).abs() < 1e-10);

        let wet_group = &recovered.profiles["wet_season"].groups[0];
        assert_eq!(wet_group.name, "southeast");
        assert!((wet_group.matrix[0][1] - 0.9).abs() < 1e-10);
    }

    // -------------------------------------------------------------------------
    // write_correlation_json_field_names_match_input_format
    // -------------------------------------------------------------------------

    #[test]
    fn write_correlation_json_field_names_match_input_format() {
        let model = make_simple_correlation_model();
        let tmp = tempfile::tempdir().expect("tempdir must succeed");
        let path = tmp.path().join("correlation.json");

        write_correlation_json(&path, &model).expect("write must succeed");

        let content = std::fs::read_to_string(&path).expect("file must be readable");

        assert!(
            content.contains("\"correlation_groups\""),
            "JSON must use 'correlation_groups' key, not 'groups'"
        );
        assert!(
            content.contains("\"type\""),
            "JSON must use 'type' key for entity type"
        );
        assert!(
            !content.contains("\"entity_type\""),
            "JSON must NOT use 'entity_type' key"
        );
        assert!(
            !content.contains("\"groups\""),
            "JSON must NOT use 'groups' key at the profile level"
        );
    }

    // -------------------------------------------------------------------------
    // write_correlation_json_creates_parent_directory
    // -------------------------------------------------------------------------

    #[test]
    fn write_correlation_json_creates_parent_directory() {
        let model = make_simple_correlation_model();
        let tmp = tempfile::tempdir().expect("tempdir must succeed");
        let path = tmp.path().join("output/stochastic/correlation.json");

        assert!(
            !path.parent().unwrap().exists(),
            "parent must not exist yet"
        );
        write_correlation_json(&path, &model).expect("write must succeed even with missing parent");
        assert!(path.exists(), "file must exist");
        assert!(
            path.parent().unwrap().is_dir(),
            "parent directory must have been created"
        );
    }

    // -------------------------------------------------------------------------
    // write_correlation_json_no_tmp_file_remains
    // -------------------------------------------------------------------------

    #[test]
    fn write_correlation_json_no_tmp_file_remains() {
        let model = make_simple_correlation_model();
        let tmp = tempfile::tempdir().expect("tempdir must succeed");
        let path = tmp.path().join("correlation.json");

        write_correlation_json(&path, &model).expect("write must succeed");

        let tmp_path = path.with_extension("json.tmp");
        assert!(
            !tmp_path.exists(),
            ".tmp file must not remain after successful atomic write"
        );
        assert!(path.exists(), "final file must exist");
    }

    // =========================================================================
    // write_load_seasonal_stats tests
    // =========================================================================

    fn make_load_stats_rows() -> Vec<LoadSeasonalStatsRow> {
        vec![
            LoadSeasonalStatsRow {
                bus_id: EntityId::from(1),
                stage_id: 0,
                mean_mw: 500.0,
                std_mw: 50.0,
            },
            LoadSeasonalStatsRow {
                bus_id: EntityId::from(1),
                stage_id: 1,
                mean_mw: 520.0,
                std_mw: 52.0,
            },
            LoadSeasonalStatsRow {
                bus_id: EntityId::from(3),
                stage_id: 0,
                mean_mw: 700.0,
                std_mw: 70.0,
            },
            LoadSeasonalStatsRow {
                bus_id: EntityId::from(3),
                stage_id: 1,
                mean_mw: 720.0,
                std_mw: 72.0,
            },
        ]
    }

    // -------------------------------------------------------------------------
    // write_then_read_load_stats_round_trips
    // -------------------------------------------------------------------------

    #[test]
    fn write_then_read_load_stats_round_trips() {
        use crate::scenarios::parse_load_seasonal_stats;

        let rows = make_load_stats_rows();
        let tmp = tempfile::tempdir().expect("tempdir must succeed");
        let path = tmp.path().join("load_seasonal_stats.parquet");

        write_load_seasonal_stats(&path, &rows).expect("write must succeed");
        assert!(path.exists(), "file must exist after write");

        let recovered = parse_load_seasonal_stats(&path).expect("parse must succeed after write");

        assert_eq!(recovered.len(), rows.len(), "row count must match");
        for (original, parsed) in rows.iter().zip(recovered.iter()) {
            assert_eq!(parsed.bus_id, original.bus_id, "bus_id must match");
            assert_eq!(parsed.stage_id, original.stage_id, "stage_id must match");
            assert!(
                (parsed.mean_mw - original.mean_mw).abs() < 1e-10,
                "mean_mw must be bit-for-bit identical"
            );
            assert!(
                (parsed.std_mw - original.std_mw).abs() < 1e-10,
                "std_mw must be bit-for-bit identical"
            );
        }
    }

    // -------------------------------------------------------------------------
    // write_load_stats_empty_rows_valid_schema
    // -------------------------------------------------------------------------

    #[test]
    fn write_load_stats_empty_rows_valid_schema() {
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

        let tmp = tempfile::tempdir().expect("tempdir must succeed");
        let path = tmp.path().join("load_seasonal_stats.parquet");

        write_load_seasonal_stats(&path, &[]).expect("write must succeed for empty rows");

        let file = std::fs::File::open(&path).expect("file must open");
        let builder =
            ParquetRecordBatchReaderBuilder::try_new(file).expect("builder must be created");
        let schema = builder.schema().clone();
        let reader = builder.build().expect("reader must be created");

        let total_rows: usize = reader
            .map(|b| b.expect("batch must be Ok").num_rows())
            .sum();
        assert_eq!(total_rows, 0, "empty rows must produce 0-row file");
        assert_eq!(
            schema.fields().len(),
            4,
            "schema must have exactly 4 columns"
        );

        let fields: Vec<(&str, &DataType)> = schema
            .fields()
            .iter()
            .map(|f| (f.name().as_str(), f.data_type()))
            .collect();
        assert_eq!(fields[0], ("bus_id", &DataType::Int32));
        assert_eq!(fields[1], ("stage_id", &DataType::Int32));
        assert_eq!(fields[2], ("mean_mw", &DataType::Float64));
        assert_eq!(fields[3], ("std_mw", &DataType::Float64));
    }

    // -------------------------------------------------------------------------
    // write_load_stats_creates_parent_directory
    // -------------------------------------------------------------------------

    #[test]
    fn write_load_stats_creates_parent_directory() {
        let rows = make_load_stats_rows();
        let tmp = tempfile::tempdir().expect("tempdir must succeed");
        let path = tmp
            .path()
            .join("output/stochastic/load_seasonal_stats.parquet");

        assert!(
            !path.parent().unwrap().exists(),
            "parent must not exist yet"
        );
        write_load_seasonal_stats(&path, &rows)
            .expect("write must succeed even with missing parent");
        assert!(path.exists(), "file must exist");
        assert!(
            path.parent().unwrap().is_dir(),
            "parent directory must have been created"
        );
    }

    // =========================================================================
    // write_fitting_report tests
    // =========================================================================

    fn make_two_hydro_report() -> FittingReport {
        let mut hydros = BTreeMap::new();
        hydros.insert(
            "1".to_string(),
            HydroFittingEntry {
                selected_order: 3,
                coefficients: vec![vec![0.42, -0.11, 0.07], vec![0.35, -0.08, 0.05]],
                contribution_reductions: Vec::new(),
            },
        );
        hydros.insert(
            "5".to_string(),
            HydroFittingEntry {
                selected_order: 1,
                coefficients: vec![vec![0.60], vec![0.55]],
                contribution_reductions: Vec::new(),
            },
        );
        FittingReport { hydros }
    }

    // -------------------------------------------------------------------------
    // write_fitting_report_two_hydros
    // -------------------------------------------------------------------------

    #[test]
    fn write_fitting_report_two_hydros() {
        let report = make_two_hydro_report();
        let tmp = tempfile::tempdir().expect("tempdir must succeed");
        let path = tmp.path().join("fitting_report.json");

        write_fitting_report(&path, &report).expect("write must succeed");

        let content = std::fs::read_to_string(&path).expect("file must be readable");
        let value: serde_json::Value = serde_json::from_str(&content).expect("must be valid JSON");

        let hydros = value["hydros"]
            .as_object()
            .expect("hydros must be an object");
        assert!(hydros.contains_key("1"), "hydros must contain key \"1\"");
        assert!(hydros.contains_key("5"), "hydros must contain key \"5\"");
        assert_eq!(hydros.len(), 2, "hydros must have exactly 2 keys");

        assert_eq!(
            value["hydros"]["1"]["selected_order"].as_u64(),
            Some(3),
            "selected_order for hydro 1 must be 3"
        );
        let coefficients = value["hydros"]["1"]["coefficients"]
            .as_array()
            .expect("coefficients must be an array");
        assert_eq!(
            coefficients.len(),
            2,
            "coefficients for hydro 1 must have 2 rows"
        );
        assert_eq!(
            coefficients[0]
                .as_array()
                .expect("row 0 must be array")
                .len(),
            3,
            "each coefficient row must have 3 elements"
        );
    }

    // -------------------------------------------------------------------------
    // write_fitting_report_empty_hydros
    // -------------------------------------------------------------------------

    #[test]
    fn write_fitting_report_empty_hydros() {
        let report = FittingReport {
            hydros: BTreeMap::new(),
        };
        let tmp = tempfile::tempdir().expect("tempdir must succeed");
        let path = tmp.path().join("fitting_report.json");

        write_fitting_report(&path, &report).expect("write must succeed for empty report");

        let content = std::fs::read_to_string(&path).expect("file must be readable");
        let value: serde_json::Value = serde_json::from_str(&content).expect("must be valid JSON");

        let hydros = value["hydros"]
            .as_object()
            .expect("hydros must be an object");
        assert!(
            hydros.is_empty(),
            "empty report must produce empty hydros object"
        );
    }

    // -------------------------------------------------------------------------
    // write_fitting_report_creates_parent_directory
    // -------------------------------------------------------------------------

    #[test]
    fn write_fitting_report_creates_parent_directory() {
        let report = make_two_hydro_report();
        let tmp = tempfile::tempdir().expect("tempdir must succeed");
        let path = tmp.path().join("output/stochastic/fitting_report.json");

        assert!(
            !path.parent().unwrap().exists(),
            "parent must not exist yet"
        );
        write_fitting_report(&path, &report).expect("write must succeed even with missing parent");
        assert!(path.exists(), "file must exist");
        assert!(
            path.parent().unwrap().is_dir(),
            "parent directory must have been created"
        );
    }

    // -------------------------------------------------------------------------
    // write_fitting_report_no_tmp_file_remains
    // -------------------------------------------------------------------------

    #[test]
    fn write_fitting_report_no_tmp_file_remains() {
        let report = make_two_hydro_report();
        let tmp = tempfile::tempdir().expect("tempdir must succeed");
        let path = tmp.path().join("fitting_report.json");

        write_fitting_report(&path, &report).expect("write must succeed");

        let tmp_path = path.with_extension("json.tmp");
        assert!(
            !tmp_path.exists(),
            ".tmp file must not remain after successful atomic write"
        );
        assert!(path.exists(), "final file must exist");
    }

    // -------------------------------------------------------------------------
    // write_fitting_report_aic_scores_preserved
    // -------------------------------------------------------------------------

    // =========================================================================
    // write_correlation_json_multi_profile_round_trip
    // =========================================================================

    #[test]
    fn write_correlation_json_multi_profile_round_trip() {
        use crate::scenarios::parse_correlation;

        let entities = vec![
            CorrelationEntity {
                entity_type: "inflow".to_string(),
                id: EntityId::from(1),
            },
            CorrelationEntity {
                entity_type: "inflow".to_string(),
                id: EntityId::from(2),
            },
        ];

        let mut profiles = BTreeMap::new();
        profiles.insert(
            "default".to_string(),
            CorrelationProfile {
                groups: vec![CorrelationGroup {
                    name: "all".to_string(),
                    entities: entities.clone(),
                    matrix: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
                }],
            },
        );
        profiles.insert(
            "season_0".to_string(),
            CorrelationProfile {
                groups: vec![CorrelationGroup {
                    name: "all".to_string(),
                    entities: entities.clone(),
                    matrix: vec![vec![1.0, 0.8], vec![0.8, 1.0]],
                }],
            },
        );
        profiles.insert(
            "season_1".to_string(),
            CorrelationProfile {
                groups: vec![CorrelationGroup {
                    name: "all".to_string(),
                    entities: entities.clone(),
                    matrix: vec![vec![1.0, 0.3], vec![0.3, 1.0]],
                }],
            },
        );

        let original = CorrelationModel {
            method: "spectral".to_string(),
            profiles,
            schedule: vec![
                CorrelationScheduleEntry {
                    stage_id: 1,
                    profile_name: "season_0".to_string(),
                },
                CorrelationScheduleEntry {
                    stage_id: 2,
                    profile_name: "season_1".to_string(),
                },
            ],
        };

        let tmp = tempfile::tempdir().expect("tempdir must succeed");
        let path = tmp.path().join("correlation.json");

        write_correlation_json(&path, &original).expect("write must succeed");
        assert!(path.exists());

        let recovered = parse_correlation(&path).expect("parse must succeed");

        assert_eq!(recovered.profiles.len(), 3);
        assert!(recovered.profiles.contains_key("default"));
        assert!(recovered.profiles.contains_key("season_0"));
        assert!(recovered.profiles.contains_key("season_1"));

        assert_eq!(recovered.schedule.len(), 2);
        assert_eq!(recovered.schedule[0].stage_id, 1);
        assert_eq!(recovered.schedule[0].profile_name, "season_0");
        assert_eq!(recovered.schedule[1].stage_id, 2);
        assert_eq!(recovered.schedule[1].profile_name, "season_1");

        let default_mat = &recovered.profiles["default"].groups[0].matrix;
        assert!((default_mat[0][0] - 1.0).abs() < 1e-10);
        assert!(default_mat[0][1].abs() < 1e-10);

        let s0_mat = &recovered.profiles["season_0"].groups[0].matrix;
        assert!((s0_mat[0][1] - 0.8).abs() < 1e-10);
        assert!((s0_mat[1][0] - 0.8).abs() < 1e-10);

        let s1_mat = &recovered.profiles["season_1"].groups[0].matrix;
        assert!((s1_mat[0][1] - 0.3).abs() < 1e-10);
        assert!((s1_mat[1][0] - 0.3).abs() < 1e-10);
    }

    // =========================================================================
    // write_inflow_annual_component tests
    // =========================================================================

    fn make_annual_component_rows() -> Vec<InflowAnnualComponentRow> {
        vec![
            InflowAnnualComponentRow {
                hydro_id: EntityId::from(1),
                stage_id: 0,
                annual_coefficient: 0.35,
                annual_mean_m3s: 1500.0,
                annual_std_m3s: 300.0,
            },
            InflowAnnualComponentRow {
                hydro_id: EntityId::from(1),
                stage_id: 1,
                annual_coefficient: -0.12,
                annual_mean_m3s: 1200.0,
                annual_std_m3s: 250.0,
            },
            InflowAnnualComponentRow {
                hydro_id: EntityId::from(2),
                stage_id: 0,
                annual_coefficient: 0.48,
                annual_mean_m3s: 800.0,
                annual_std_m3s: 150.0,
            },
        ]
    }

    // -------------------------------------------------------------------------
    // test_write_inflow_annual_component_round_trip_three_rows
    // -------------------------------------------------------------------------

    #[test]
    fn test_write_inflow_annual_component_round_trip_three_rows() {
        use crate::scenarios::parse_inflow_annual_component;

        let rows = make_annual_component_rows();
        let tmp = tempfile::tempdir().expect("tempdir must succeed");
        let path = tmp.path().join("inflow_annual_component.parquet");

        write_inflow_annual_component(&path, &rows).expect("write must succeed");
        assert!(path.exists(), "file must exist after write");

        let recovered =
            parse_inflow_annual_component(&path).expect("parse must succeed after write");

        assert_eq!(recovered.len(), rows.len(), "row count must match");
        for (original, parsed) in rows.iter().zip(recovered.iter()) {
            assert_eq!(parsed.hydro_id, original.hydro_id, "hydro_id must match");
            assert_eq!(parsed.stage_id, original.stage_id, "stage_id must match");
            assert_eq!(
                parsed.annual_coefficient, original.annual_coefficient,
                "annual_coefficient must be bit-for-bit identical"
            );
            assert_eq!(
                parsed.annual_mean_m3s, original.annual_mean_m3s,
                "annual_mean_m3s must be bit-for-bit identical"
            );
            assert_eq!(
                parsed.annual_std_m3s, original.annual_std_m3s,
                "annual_std_m3s must be bit-for-bit identical"
            );
        }
    }

    // -------------------------------------------------------------------------
    // test_write_inflow_annual_component_empty_round_trip
    // -------------------------------------------------------------------------

    #[test]
    fn test_write_inflow_annual_component_empty_round_trip() {
        use crate::scenarios::parse_inflow_annual_component;

        let tmp = tempfile::tempdir().expect("tempdir must succeed");
        let path = tmp.path().join("inflow_annual_component.parquet");

        write_inflow_annual_component(&path, &[]).expect("write must succeed for empty rows");
        assert!(path.exists(), "file must exist after write");

        let recovered =
            parse_inflow_annual_component(&path).expect("parse must succeed after write");
        assert!(
            recovered.is_empty(),
            "empty write must produce an empty parsed result"
        );
    }

    // -------------------------------------------------------------------------
    // test_write_inflow_annual_component_byte_stable_two_writes
    // -------------------------------------------------------------------------

    #[test]
    fn test_write_inflow_annual_component_byte_stable_two_writes() {
        let rows = make_annual_component_rows();
        let tmp = tempfile::tempdir().expect("tempdir must succeed");
        let path = tmp.path().join("inflow_annual_component.parquet");

        write_inflow_annual_component(&path, &rows).expect("first write must succeed");
        let bytes1 = std::fs::read(&path).expect("first read must succeed");

        write_inflow_annual_component(&path, &rows).expect("second write must succeed");
        let bytes2 = std::fs::read(&path).expect("second read must succeed");

        assert_eq!(
            bytes1, bytes2,
            "two writes of the same data must produce byte-identical files"
        );
    }

    // -------------------------------------------------------------------------
    // test_write_inflow_annual_component_creates_parent_dir
    // -------------------------------------------------------------------------

    #[test]
    fn test_write_inflow_annual_component_creates_parent_dir() {
        let rows = make_annual_component_rows();
        let tmp = tempfile::tempdir().expect("tempdir must succeed");
        let path = tmp.path().join("a/b/inflow_annual_component.parquet");

        assert!(
            !path.parent().unwrap().exists(),
            "parent must not exist yet"
        );
        write_inflow_annual_component(&path, &rows)
            .expect("write must succeed even with missing parent");
        assert!(path.exists(), "file must exist");
        assert!(
            path.parent().unwrap().is_dir(),
            "parent directory must have been created"
        );
    }

    // -------------------------------------------------------------------------
    // test_write_inflow_annual_component_negative_psi_round_trip
    // -------------------------------------------------------------------------

    #[test]
    fn test_write_inflow_annual_component_negative_psi_round_trip() {
        use crate::scenarios::parse_inflow_annual_component;

        let rows = vec![InflowAnnualComponentRow {
            hydro_id: EntityId::from(7),
            stage_id: 3,
            annual_coefficient: -0.5,
            annual_mean_m3s: 87.5,
            annual_std_m3s: 11.25,
        }];

        let tmp = tempfile::tempdir().expect("tempdir must succeed");
        let path = tmp.path().join("inflow_annual_component.parquet");

        write_inflow_annual_component(&path, &rows).expect("write must succeed");
        let recovered =
            parse_inflow_annual_component(&path).expect("parse must succeed after write");

        assert_eq!(recovered.len(), 1, "must have exactly one row");
        let row = &recovered[0];
        assert_eq!(row.hydro_id, EntityId::from(7), "hydro_id must match");
        assert_eq!(row.stage_id, 3, "stage_id must match");
        assert_eq!(
            row.annual_coefficient, -0.5,
            "annual_coefficient must be bit-for-bit identical"
        );
        assert_eq!(
            row.annual_mean_m3s, 87.5,
            "annual_mean_m3s must be bit-for-bit identical"
        );
        assert_eq!(
            row.annual_std_m3s, 11.25,
            "annual_std_m3s must be bit-for-bit identical"
        );
    }
}
