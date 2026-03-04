//! Parsing for `scenarios/load_seasonal_stats.parquet` — per-bus-per-stage
//! mean and standard deviation of load demand.
//!
//! [`parse_load_seasonal_stats`] reads `scenarios/load_seasonal_stats.parquet`
//! and returns a sorted `Vec<LoadSeasonalStatsRow>`.
//!
//! ## Parquet schema (spec SS3.3)
//!
//! | Column     | Type   | Required | Description                          |
//! | ---------- | ------ | -------- | ------------------------------------ |
//! | `bus_id`   | INT32  | Yes      | Bus ID                               |
//! | `stage_id` | INT32  | Yes      | Stage ID                             |
//! | `mean_mw`  | DOUBLE | Yes      | Mean load demand (MW)                |
//! | `std_mw`   | DOUBLE | Yes      | Standard deviation (MW), 0 = deterministic |
//!
//! ## Output ordering
//!
//! Rows are sorted by `(bus_id, stage_id)` ascending.
//!
//! ## Validation
//!
//! Per-row constraints enforced by this parser:
//!
//! - All four columns must be present with the correct types.
//! - `mean_mw` must be finite (NaN and ±inf are rejected).
//! - `std_mw` must be non-negative and finite.
//!
//! Deferred validations (not performed here):
//!
//! - `bus_id` existence in the bus registry — Layer 3, Epic 06.
//! - `stage_id` existence in the stages registry — Layer 3, Epic 06.

use cobre_core::EntityId;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::fs::File;
use std::path::Path;

use crate::parquet_helpers::{extract_required_float64, extract_required_int32};
use crate::LoadError;

/// A single row from `scenarios/load_seasonal_stats.parquet`.
///
/// Carries the seasonal load statistics for a (bus, stage) pair. These rows
/// are later assembled by ticket-023 to produce [`cobre_core::scenario::LoadModel`]
/// entries.
///
/// # Examples
///
/// ```
/// use cobre_io::scenarios::LoadSeasonalStatsRow;
/// use cobre_core::EntityId;
///
/// let row = LoadSeasonalStatsRow {
///     bus_id: EntityId::from(1),
///     stage_id: 3,
///     mean_mw: 500.0,
///     std_mw: 50.0,
/// };
/// assert_eq!(row.bus_id, EntityId::from(1));
/// assert_eq!(row.stage_id, 3);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct LoadSeasonalStatsRow {
    /// Bus this load model belongs to.
    pub bus_id: EntityId,
    /// Stage (0-based index within `System::stages`) this model applies to.
    pub stage_id: i32,
    /// Seasonal mean load demand μ in MW. Must be finite.
    pub mean_mw: f64,
    /// Seasonal standard deviation σ in MW. Must be non-negative and finite.
    /// A value of 0.0 indicates deterministic load.
    pub std_mw: f64,
}

/// Parse `scenarios/load_seasonal_stats.parquet` and return a sorted row table.
///
/// Reads all record batches from the Parquet file at `path`, validates per-row
/// constraints, then returns all rows sorted by `(bus_id, stage_id)` ascending.
///
/// # Errors
///
/// | Condition                                     | Error variant              |
/// |---------------------------------------------- |--------------------------- |
/// | File not found or permission denied           | [`LoadError::IoError`]     |
/// | Malformed Parquet (corrupt header, etc.)      | [`LoadError::ParseError`]  |
/// | Required column missing or wrong type         | [`LoadError::SchemaError`] |
/// | `mean_mw` is NaN or infinite                 | [`LoadError::SchemaError`] |
/// | `std_mw` is negative or not finite            | [`LoadError::SchemaError`] |
///
/// # Examples
///
/// ```no_run
/// use cobre_io::scenarios::parse_load_seasonal_stats;
/// use std::path::Path;
///
/// let rows = parse_load_seasonal_stats(Path::new("scenarios/load_seasonal_stats.parquet"))
///     .expect("valid load seasonal stats file");
/// println!("loaded {} load seasonal stats rows", rows.len());
/// ```
pub fn parse_load_seasonal_stats(path: &Path) -> Result<Vec<LoadSeasonalStatsRow>, LoadError> {
    let file = File::open(path).map_err(|e| LoadError::io(path, e))?;

    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| LoadError::parse(path, e.to_string()))?;

    let reader = builder
        .build()
        .map_err(|e| LoadError::parse(path, e.to_string()))?;

    let mut rows: Vec<LoadSeasonalStatsRow> = Vec::new();

    for batch_result in reader {
        let batch = batch_result.map_err(|e| LoadError::parse(path, e.to_string()))?;

        // ── Required columns ──────────────────────────────────────────────────
        let bus_id_col = extract_required_int32(&batch, "bus_id", path)?;
        let stage_id_col = extract_required_int32(&batch, "stage_id", path)?;
        let mean_mw_col = extract_required_float64(&batch, "mean_mw", path)?;
        let std_mw_col = extract_required_float64(&batch, "std_mw", path)?;

        // ── Build rows with per-row validation ────────────────────────────────
        let n = batch.num_rows();
        let base_idx = rows.len();
        rows.reserve(n);

        for i in 0..n {
            let row_idx = base_idx + i;

            let bus_id = EntityId::from(bus_id_col.value(i));
            let stage_id = stage_id_col.value(i);
            let mean_mw = mean_mw_col.value(i);
            let std_mw = std_mw_col.value(i);

            // Validate mean_mw: must be finite (NaN and ±inf rejected).
            if !mean_mw.is_finite() {
                return Err(LoadError::SchemaError {
                    path: path.to_path_buf(),
                    field: format!("load_seasonal_stats[{row_idx}].mean_mw"),
                    message: format!("value must be finite, got {mean_mw}"),
                });
            }

            // Validate std_mw: must be non-negative and finite.
            if !std_mw.is_finite() || std_mw < 0.0 {
                return Err(LoadError::SchemaError {
                    path: path.to_path_buf(),
                    field: format!("load_seasonal_stats[{row_idx}].std_mw"),
                    message: format!("value must be non-negative and finite, got {std_mw}"),
                });
            }

            rows.push(LoadSeasonalStatsRow {
                bus_id,
                stage_id,
                mean_mw,
                std_mw,
            });
        }
    }

    // ── Sort by (bus_id, stage_id) ascending ─────────────────────────────────
    rows.sort_by(|a, b| {
        a.bus_id
            .0
            .cmp(&b.bus_id.0)
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

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("bus_id", DataType::Int32, false),
            Field::new("stage_id", DataType::Int32, false),
            Field::new("mean_mw", DataType::Float64, false),
            Field::new("std_mw", DataType::Float64, false),
        ]))
    }

    fn write_parquet(batch: &RecordBatch) -> NamedTempFile {
        let tmp = NamedTempFile::new().expect("tempfile");
        let mut writer = ArrowWriter::try_new(tmp.reopen().expect("reopen"), batch.schema(), None)
            .expect("ArrowWriter");
        writer.write(batch).expect("write batch");
        writer.close().expect("close writer");
        tmp
    }

    fn make_batch(bus_ids: &[i32], stage_ids: &[i32], means: &[f64], stds: &[f64]) -> RecordBatch {
        RecordBatch::try_new(
            schema(),
            vec![
                Arc::new(Int32Array::from(bus_ids.to_vec())),
                Arc::new(Int32Array::from(stage_ids.to_vec())),
                Arc::new(Float64Array::from(means.to_vec())),
                Arc::new(Float64Array::from(stds.to_vec())),
            ],
        )
        .expect("valid batch")
    }

    // ── AC: valid file with 4 rows, verify sort order and field values ─────────

    /// Valid 4-row file (2 buses x 2 stages) in scrambled order.
    /// Result: sorted by (bus_id, stage_id); first row bus_id = EntityId(1).
    #[test]
    fn test_valid_4_rows_sorted_by_bus_stage() {
        // Input order: (3,1), (1,0), (3,0), (1,1) — out of sort order.
        let batch = make_batch(
            &[3, 1, 3, 1],
            &[1, 0, 0, 1],
            &[700.0, 500.0, 650.0, 520.0],
            &[70.0, 50.0, 65.0, 52.0],
        );
        let tmp = write_parquet(&batch);
        let rows = parse_load_seasonal_stats(tmp.path()).unwrap();

        assert_eq!(rows.len(), 4);
        assert_eq!(rows[0].bus_id, EntityId::from(1));
        assert_eq!(rows[0].stage_id, 0);
        assert!((rows[0].mean_mw - 500.0).abs() < 1e-10);
        assert!((rows[0].std_mw - 50.0).abs() < 1e-10);
        assert_eq!(rows[1].bus_id, EntityId::from(1));
        assert_eq!(rows[1].stage_id, 1);
        assert_eq!(rows[2].bus_id, EntityId::from(3));
        assert_eq!(rows[2].stage_id, 0);
        assert_eq!(rows[3].bus_id, EntityId::from(3));
        assert_eq!(rows[3].stage_id, 1);
    }

    // ── AC: std_mw = 0.0 (deterministic) is accepted ─────────────────────────

    /// `std_mw = 0.0` is valid (deterministic load). Must not return an error.
    #[test]
    fn test_zero_std_mw_is_accepted() {
        let batch = make_batch(&[1], &[0], &[500.0], &[0.0]);
        let tmp = write_parquet(&batch);
        let rows = parse_load_seasonal_stats(tmp.path()).unwrap();

        assert_eq!(rows.len(), 1);
        assert!(rows[0].std_mw.abs() < f64::EPSILON);
    }

    // ── AC: std_mw negative -> SchemaError ───────────────────────────────────

    /// `std_mw = -5.0` -> SchemaError with field path containing "std_mw".
    #[test]
    fn test_negative_std_mw() {
        let batch = make_batch(&[1], &[0], &[500.0], &[-5.0]);
        let tmp = write_parquet(&batch);
        let err = parse_load_seasonal_stats(tmp.path()).unwrap_err();

        match &err {
            LoadError::SchemaError { field, .. } => {
                assert!(
                    field.contains("std_mw"),
                    "field should contain 'std_mw', got: {field}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    // ── AC: mean_mw NaN -> SchemaError ───────────────────────────────────────

    /// `mean_mw = NaN` -> SchemaError with field path containing "mean_mw".
    #[test]
    fn test_nan_mean_mw() {
        let batch = make_batch(&[1], &[0], &[f64::NAN], &[50.0]);
        let tmp = write_parquet(&batch);
        let err = parse_load_seasonal_stats(tmp.path()).unwrap_err();

        match &err {
            LoadError::SchemaError { field, .. } => {
                assert!(
                    field.contains("mean_mw"),
                    "field should contain 'mean_mw', got: {field}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    // ── AC: missing required column -> SchemaError ────────────────────────────

    /// File missing `mean_mw` column -> SchemaError.
    #[test]
    fn test_missing_mean_mw_column() {
        let schema_no_mean = Arc::new(Schema::new(vec![
            Field::new("bus_id", DataType::Int32, false),
            Field::new("stage_id", DataType::Int32, false),
            Field::new("std_mw", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema_no_mean,
            vec![
                Arc::new(Int32Array::from(vec![1_i32])),
                Arc::new(Int32Array::from(vec![0_i32])),
                Arc::new(Float64Array::from(vec![50.0])),
            ],
        )
        .unwrap();
        let tmp = write_parquet(&batch);
        let err = parse_load_seasonal_stats(tmp.path()).unwrap_err();

        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("mean_mw"),
                    "field should contain 'mean_mw', got: {field}"
                );
                assert!(
                    message.contains("missing required column"),
                    "message should mention missing column, got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    // ── AC: empty file -> Ok(vec![]) ──────────────────────────────────────────

    /// Empty Parquet (0 rows) -> Ok(Vec::new()).
    #[test]
    fn test_empty_parquet_returns_empty_vec() {
        let batch = make_batch(&[], &[], &[], &[]);
        let tmp = write_parquet(&batch);
        let rows = parse_load_seasonal_stats(tmp.path()).unwrap();
        assert!(rows.is_empty());
    }

    // ── AC: declaration-order invariance ─────────────────────────────────────

    /// Reordering the Parquet rows does not change the output ordering.
    #[test]
    fn test_declaration_order_invariance() {
        let batch_asc = make_batch(
            &[1, 1, 5, 5],
            &[0, 1, 0, 1],
            &[100.0, 110.0, 200.0, 210.0],
            &[10.0, 11.0, 20.0, 21.0],
        );
        let batch_desc = make_batch(
            &[5, 5, 1, 1],
            &[1, 0, 1, 0],
            &[210.0, 200.0, 110.0, 100.0],
            &[21.0, 20.0, 11.0, 10.0],
        );
        let tmp_asc = write_parquet(&batch_asc);
        let tmp_desc = write_parquet(&batch_desc);
        let rows_asc = parse_load_seasonal_stats(tmp_asc.path()).unwrap();
        let rows_desc = parse_load_seasonal_stats(tmp_desc.path()).unwrap();

        let keys_asc: Vec<(i32, i32)> = rows_asc.iter().map(|r| (r.bus_id.0, r.stage_id)).collect();
        let keys_desc: Vec<(i32, i32)> =
            rows_desc.iter().map(|r| (r.bus_id.0, r.stage_id)).collect();
        assert_eq!(keys_asc, keys_desc);
    }
}
