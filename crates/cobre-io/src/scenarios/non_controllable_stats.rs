//! Parsing for `scenarios/non_controllable_stats.parquet` — per-NCS-per-stage
//! mean and standard deviation of the stochastic availability factor.
//!
//! [`parse_ncs_stats`] reads `scenarios/non_controllable_stats.parquet`
//! and returns a sorted `Vec<NcsModel>`.
//!
//! ## Parquet schema
//!
//! | Column     | Type   | Required | Description                                            |
//! | ---------- | ------ | -------- | ------------------------------------------------------ |
//! | `ncs_id`   | INT32  | Yes      | Non-controllable source entity ID                      |
//! | `stage_id` | INT32  | Yes      | Stage ID                                               |
//! | `mean`     | DOUBLE | Yes      | Mean availability factor [0, 1]                        |
//! | `std`      | DOUBLE | Yes      | Std dev of availability factor (>= 0), 0 = deterministic |
//!
//! ## Output ordering
//!
//! Rows are sorted by `(ncs_id, stage_id)` ascending.
//!
//! ## Validation
//!
//! Per-row constraints enforced by this parser:
//!
//! - All four columns must be present with the correct types.
//! - `mean` must be finite and in `[0, 1]` (NaN, +/-inf, and out-of-range are rejected).
//! - `std` must be non-negative and finite.
//!
//! Deferred validations (not performed here):
//!
//! - `ncs_id` existence in the NCS registry — Layer 3 referential validation.
//! - `stage_id` existence in the stages registry — Layer 3 referential validation.

use cobre_core::EntityId;
use cobre_core::scenario::NcsModel;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::fs::File;
use std::path::Path;

use crate::LoadError;
use crate::parquet_helpers::{extract_required_float64, extract_required_int32};

/// Parse `scenarios/non_controllable_stats.parquet` and return a sorted `Vec<NcsModel>`.
///
/// Reads all record batches from the Parquet file at `path`, validates per-row
/// constraints, then returns all rows sorted by `(ncs_id, stage_id)` ascending.
///
/// # Errors
///
/// | Condition                                     | Error variant              |
/// |---------------------------------------------- |--------------------------- |
/// | File not found or permission denied           | [`LoadError::IoError`]     |
/// | Malformed Parquet (corrupt header, etc.)      | [`LoadError::ParseError`]  |
/// | Required column missing or wrong type         | [`LoadError::SchemaError`] |
/// | `mean` is NaN, infinite, or outside `[0, 1]` | [`LoadError::SchemaError`] |
/// | `std` is negative or not finite              | [`LoadError::SchemaError`] |
///
/// # Examples
///
/// ```no_run
/// use cobre_io::scenarios::parse_ncs_stats;
/// use std::path::Path;
///
/// let models = parse_ncs_stats(Path::new("scenarios/non_controllable_stats.parquet"))
///     .expect("valid NCS models file");
/// println!("loaded {} NCS model rows", models.len());
/// ```
pub fn parse_ncs_stats(path: &Path) -> Result<Vec<NcsModel>, LoadError> {
    let file = File::open(path).map_err(|e| LoadError::io(path, e))?;

    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| LoadError::parse(path, e.to_string()))?;

    let reader = builder
        .build()
        .map_err(|e| LoadError::parse(path, e.to_string()))?;

    let mut rows: Vec<NcsModel> = Vec::new();

    for batch_result in reader {
        let batch = batch_result.map_err(|e| LoadError::parse(path, e.to_string()))?;

        // ── Required columns ──────────────────────────────────────────────────
        let ncs_id_col = extract_required_int32(&batch, "ncs_id", path)?;
        let stage_id_col = extract_required_int32(&batch, "stage_id", path)?;
        let mean_col = extract_required_float64(&batch, "mean", path)?;
        let std_col = extract_required_float64(&batch, "std", path)?;

        // ── Build rows with per-row validation ────────────────────────────────
        let n = batch.num_rows();
        let base_idx = rows.len();
        rows.reserve(n);

        for i in 0..n {
            let row_idx = base_idx + i;

            let ncs_id = EntityId::from(ncs_id_col.value(i));
            let stage_id = stage_id_col.value(i);
            let mean = mean_col.value(i);
            let std = std_col.value(i);

            // Validate mean: must be finite and in [0, 1].
            if !mean.is_finite() || !(0.0..=1.0).contains(&mean) {
                return Err(LoadError::SchemaError {
                    path: path.to_path_buf(),
                    field: format!("non_controllable_stats[{row_idx}].mean"),
                    message: format!("value must be finite and in [0, 1], got {mean}"),
                });
            }

            // Validate std: must be non-negative and finite.
            if !std.is_finite() || std < 0.0 {
                return Err(LoadError::SchemaError {
                    path: path.to_path_buf(),
                    field: format!("non_controllable_stats[{row_idx}].std"),
                    message: format!("value must be non-negative and finite, got {std}"),
                });
            }

            rows.push(NcsModel {
                ncs_id,
                stage_id,
                mean,
                std,
            });
        }
    }

    // ── Sort by (ncs_id, stage_id) ascending ─────────────────────────────────
    rows.sort_by(|a, b| {
        a.ncs_id
            .0
            .cmp(&b.ncs_id.0)
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
            Field::new("ncs_id", DataType::Int32, false),
            Field::new("stage_id", DataType::Int32, false),
            Field::new("mean", DataType::Float64, false),
            Field::new("std", DataType::Float64, false),
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

    fn make_batch(ncs_ids: &[i32], stage_ids: &[i32], means: &[f64], stds: &[f64]) -> RecordBatch {
        RecordBatch::try_new(
            schema(),
            vec![
                Arc::new(Int32Array::from(ncs_ids.to_vec())),
                Arc::new(Int32Array::from(stage_ids.to_vec())),
                Arc::new(Float64Array::from(means.to_vec())),
                Arc::new(Float64Array::from(stds.to_vec())),
            ],
        )
        .expect("valid batch")
    }

    // ── AC: valid file with 4 rows, verify sort order and field values ───────

    #[test]
    fn test_valid_4_rows_sorted_by_ncs_stage() {
        // Input order: (3,1), (1,0), (3,0), (1,1) — out of sort order.
        let batch = make_batch(
            &[3, 1, 3, 1],
            &[1, 0, 0, 1],
            &[0.5, 0.3, 0.45, 0.35],
            &[0.05, 0.03, 0.045, 0.035],
        );
        let tmp = write_parquet(&batch);
        let rows = parse_ncs_stats(tmp.path()).unwrap();

        assert_eq!(rows.len(), 4);
        assert_eq!(rows[0].ncs_id, EntityId::from(1));
        assert_eq!(rows[0].stage_id, 0);
        assert!((rows[0].mean - 0.3).abs() < 1e-10);
        assert!((rows[0].std - 0.03).abs() < 1e-10);
        assert_eq!(rows[1].ncs_id, EntityId::from(1));
        assert_eq!(rows[1].stage_id, 1);
        assert_eq!(rows[2].ncs_id, EntityId::from(3));
        assert_eq!(rows[2].stage_id, 0);
        assert_eq!(rows[3].ncs_id, EntityId::from(3));
        assert_eq!(rows[3].stage_id, 1);
    }

    // ── AC: std = 0.0 (deterministic) is accepted ───────────────────────────

    #[test]
    fn test_zero_std_is_accepted() {
        let batch = make_batch(&[1], &[0], &[0.5], &[0.0]);
        let tmp = write_parquet(&batch);
        let rows = parse_ncs_stats(tmp.path()).unwrap();

        assert_eq!(rows.len(), 1);
        assert!(rows[0].std.abs() < f64::EPSILON);
    }

    // ── AC: std negative -> SchemaError ─────────────────────────────────────

    #[test]
    fn test_negative_std() {
        let batch = make_batch(&[1], &[0], &[0.5], &[-0.05]);
        let tmp = write_parquet(&batch);
        let err = parse_ncs_stats(tmp.path()).unwrap_err();

        match &err {
            LoadError::SchemaError { field, .. } => {
                assert!(
                    field.contains("std"),
                    "field should contain 'std', got: {field}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    // ── AC: mean NaN -> SchemaError ──────────────────────────────────────────

    #[test]
    fn test_nan_mean() {
        let batch = make_batch(&[1], &[0], &[f64::NAN], &[0.05]);
        let tmp = write_parquet(&batch);
        let err = parse_ncs_stats(tmp.path()).unwrap_err();

        match &err {
            LoadError::SchemaError { field, .. } => {
                assert!(
                    field.contains("mean"),
                    "field should contain 'mean', got: {field}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    // ── AC: mean > 1.0 -> SchemaError ───────────────────────────────────────

    #[test]
    fn test_mean_out_of_range() {
        let batch = make_batch(&[1], &[0], &[1.5], &[0.0]);
        let tmp = write_parquet(&batch);
        let err = parse_ncs_stats(tmp.path()).unwrap_err();

        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("mean"),
                    "field should contain 'mean', got: {field}"
                );
                assert!(
                    message.contains("[0, 1]"),
                    "message should mention [0, 1] range, got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    // ── AC: missing required column -> SchemaError ──────────────────────────

    #[test]
    fn test_missing_mean_column() {
        let schema_no_mean = Arc::new(Schema::new(vec![
            Field::new("ncs_id", DataType::Int32, false),
            Field::new("stage_id", DataType::Int32, false),
            Field::new("std", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema_no_mean,
            vec![
                Arc::new(Int32Array::from(vec![1_i32])),
                Arc::new(Int32Array::from(vec![0_i32])),
                Arc::new(Float64Array::from(vec![0.05])),
            ],
        )
        .unwrap();
        let tmp = write_parquet(&batch);
        let err = parse_ncs_stats(tmp.path()).unwrap_err();

        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("mean"),
                    "field should contain 'mean', got: {field}"
                );
                assert!(
                    message.contains("missing required column"),
                    "message should mention missing column, got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    // ── AC: empty file -> Ok(vec![]) ────────────────────────────────────────

    #[test]
    fn test_empty_parquet_returns_empty_vec() {
        let batch = make_batch(&[], &[], &[], &[]);
        let tmp = write_parquet(&batch);
        let rows = parse_ncs_stats(tmp.path()).unwrap();
        assert!(rows.is_empty());
    }

    // ── AC: declaration-order invariance ────────────────────────────────────

    #[test]
    fn test_declaration_order_invariance() {
        let batch_asc = make_batch(
            &[1, 1, 5, 5],
            &[0, 1, 0, 1],
            &[0.30, 0.35, 0.50, 0.55],
            &[0.03, 0.035, 0.05, 0.055],
        );
        let batch_desc = make_batch(
            &[5, 5, 1, 1],
            &[1, 0, 1, 0],
            &[0.55, 0.50, 0.35, 0.30],
            &[0.055, 0.05, 0.035, 0.03],
        );
        let tmp_asc = write_parquet(&batch_asc);
        let tmp_desc = write_parquet(&batch_desc);
        let rows_asc = parse_ncs_stats(tmp_asc.path()).unwrap();
        let rows_desc = parse_ncs_stats(tmp_desc.path()).unwrap();

        let keys_asc: Vec<(i32, i32)> = rows_asc.iter().map(|r| (r.ncs_id.0, r.stage_id)).collect();
        let keys_desc: Vec<(i32, i32)> =
            rows_desc.iter().map(|r| (r.ncs_id.0, r.stage_id)).collect();
        assert_eq!(keys_asc, keys_desc);
    }
}
