//! Parsing for `scenarios/inflow_seasonal_stats.parquet` — PAR(p) seasonal
//! mean and standard deviation per (hydro, stage).
//!
//! [`parse_inflow_seasonal_stats`] reads `scenarios/inflow_seasonal_stats.parquet`
//! and returns a sorted `Vec<InflowSeasonalStatsRow>`.
//!
//! ## Parquet schema (spec SS3.1)
//!
//! | Column     | Type   | Required | Description                          |
//! | ---------- | ------ | -------- | ------------------------------------ |
//! | `hydro_id` | INT32  | Yes      | Hydro plant ID                       |
//! | `stage_id` | INT32  | Yes      | Stage ID                             |
//! | `mean_m3s` | DOUBLE | Yes      | Seasonal mean inflow (m³/s)          |
//! | `std_m3s`  | DOUBLE | Yes      | Seasonal standard deviation (m³/s)   |
//!
//! ## Output ordering
//!
//! Rows are sorted by `(hydro_id, stage_id)` ascending.
//!
//! ## Validation
//!
//! Per-row constraints enforced by this parser:
//!
//! - All four columns must be present with the correct types.
//! - `mean_m3s` must be finite (NaN and ±inf are rejected).
//! - `std_m3s` must be non-negative and finite.
//!
//! Deferred validations (not performed here):
//!
//! - `hydro_id` existence in the hydro registry — Layer 3, Epic 06.
//! - `stage_id` existence in the stages registry — Layer 3, Epic 06.
//! - Coverage: every (hydro, stage) with AR coefficients has a stats row — Layer 4/5.

use cobre_core::EntityId;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::fs::File;
use std::path::Path;

use crate::parquet_helpers::{extract_required_float64, extract_required_int32};
use crate::LoadError;

/// A single row from `scenarios/inflow_seasonal_stats.parquet`.
///
/// Carries the PAR(p) seasonal statistics for a (hydro, stage) pair loaded
/// from the `inflow_seasonal_stats.parquet` file. These rows are later joined
/// with [`InflowArCoefficientRow`](super::InflowArCoefficientRow) by
/// [`super::assemble_inflow_models`] to produce [`cobre_core::scenario::InflowModel`] entries.
///
/// # Examples
///
/// ```
/// use cobre_io::scenarios::InflowSeasonalStatsRow;
/// use cobre_core::EntityId;
///
/// let row = InflowSeasonalStatsRow {
///     hydro_id: EntityId::from(1),
///     stage_id: 3,
///     mean_m3s: 150.0,
///     std_m3s: 30.0,
/// };
/// assert_eq!(row.hydro_id, EntityId::from(1));
/// assert_eq!(row.stage_id, 3);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct InflowSeasonalStatsRow {
    /// Hydro plant this model belongs to.
    pub hydro_id: EntityId,
    /// Stage (0-based index within `System::stages`) this model applies to.
    pub stage_id: i32,
    /// Seasonal mean inflow μ in m³/s. Must be finite.
    pub mean_m3s: f64,
    /// Seasonal sample standard deviation `s_m` in m³/s. Must be non-negative and finite.
    pub std_m3s: f64,
}

/// Parse `scenarios/inflow_seasonal_stats.parquet` and return a sorted row table.
///
/// Reads all record batches from the Parquet file at `path`, validates per-row
/// constraints, then returns all rows sorted by `(hydro_id, stage_id)` ascending.
///
/// # Errors
///
/// | Condition                                     | Error variant              |
/// |---------------------------------------------- |--------------------------- |
/// | File not found or permission denied           | [`LoadError::IoError`]     |
/// | Malformed Parquet (corrupt header, etc.)      | [`LoadError::ParseError`]  |
/// | Required column missing or wrong type         | [`LoadError::SchemaError`] |
/// | `mean_m3s` is NaN or infinite                 | [`LoadError::SchemaError`] |
/// | `std_m3s` is negative or not finite           | [`LoadError::SchemaError`] |
///
/// # Examples
///
/// ```no_run
/// use cobre_io::scenarios::parse_inflow_seasonal_stats;
/// use std::path::Path;
///
/// let rows = parse_inflow_seasonal_stats(Path::new("scenarios/inflow_seasonal_stats.parquet"))
///     .expect("valid inflow seasonal stats file");
/// println!("loaded {} inflow seasonal stats rows", rows.len());
/// ```
pub fn parse_inflow_seasonal_stats(path: &Path) -> Result<Vec<InflowSeasonalStatsRow>, LoadError> {
    let file = File::open(path).map_err(|e| LoadError::io(path, e))?;

    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| LoadError::parse(path, e.to_string()))?;

    let reader = builder
        .build()
        .map_err(|e| LoadError::parse(path, e.to_string()))?;

    let mut rows: Vec<InflowSeasonalStatsRow> = Vec::new();

    for batch_result in reader {
        let batch = batch_result.map_err(|e| LoadError::parse(path, e.to_string()))?;

        let hydro_id_col = extract_required_int32(&batch, "hydro_id", path)?;
        let stage_id_col = extract_required_int32(&batch, "stage_id", path)?;
        let mean_m3s_col = extract_required_float64(&batch, "mean_m3s", path)?;
        let std_m3s_col = extract_required_float64(&batch, "std_m3s", path)?;

        let n = batch.num_rows();
        let base_idx = rows.len();
        rows.reserve(n);

        for i in 0..n {
            let row_idx = base_idx + i;

            let hydro_id = EntityId::from(hydro_id_col.value(i));
            let stage_id = stage_id_col.value(i);
            let mean_m3s = mean_m3s_col.value(i);
            let std_m3s = std_m3s_col.value(i);

            // Validate mean_m3s: must be finite (NaN and ±inf rejected).
            if !mean_m3s.is_finite() {
                return Err(LoadError::SchemaError {
                    path: path.to_path_buf(),
                    field: format!("inflow_seasonal_stats[{row_idx}].mean_m3s"),
                    message: format!("value must be finite, got {mean_m3s}"),
                });
            }

            // Validate std_m3s: must be non-negative and finite.
            if !std_m3s.is_finite() || std_m3s < 0.0 {
                return Err(LoadError::SchemaError {
                    path: path.to_path_buf(),
                    field: format!("inflow_seasonal_stats[{row_idx}].std_m3s"),
                    message: format!("value must be non-negative and finite, got {std_m3s}"),
                });
            }

            rows.push(InflowSeasonalStatsRow {
                hydro_id,
                stage_id,
                mean_m3s,
                std_m3s,
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

    fn schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("hydro_id", DataType::Int32, false),
            Field::new("stage_id", DataType::Int32, false),
            Field::new("mean_m3s", DataType::Float64, false),
            Field::new("std_m3s", DataType::Float64, false),
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

    fn make_batch(
        hydro_ids: &[i32],
        stage_ids: &[i32],
        means: &[f64],
        stds: &[f64],
    ) -> RecordBatch {
        RecordBatch::try_new(
            schema(),
            vec![
                Arc::new(Int32Array::from(hydro_ids.to_vec())),
                Arc::new(Int32Array::from(stage_ids.to_vec())),
                Arc::new(Float64Array::from(means.to_vec())),
                Arc::new(Float64Array::from(stds.to_vec())),
            ],
        )
        .expect("valid batch")
    }

    #[test]
    fn test_valid_4_rows_sorted_by_hydro_stage() {
        let batch = make_batch(
            &[3, 1, 3, 1],
            &[1, 0, 0, 1],
            &[200.0, 150.0, 180.0, 160.0],
            &[40.0, 30.0, 35.0, 32.0],
        );
        let tmp = write_parquet(&batch);
        let rows = parse_inflow_seasonal_stats(tmp.path()).unwrap();

        assert_eq!(rows.len(), 4);
        assert_eq!(rows[0].hydro_id, EntityId::from(1));
        assert_eq!(rows[0].stage_id, 0);
        assert!((rows[0].mean_m3s - 150.0).abs() < 1e-10);
        assert!((rows[0].std_m3s - 30.0).abs() < 1e-10);
        assert_eq!(rows[1].hydro_id, EntityId::from(1));
        assert_eq!(rows[1].stage_id, 1);
        assert_eq!(rows[2].hydro_id, EntityId::from(3));
        assert_eq!(rows[2].stage_id, 0);
        assert_eq!(rows[3].hydro_id, EntityId::from(3));
        assert_eq!(rows[3].stage_id, 1);
    }

    #[test]
    fn test_missing_mean_m3s_column() {
        let schema_no_mean = Arc::new(Schema::new(vec![
            Field::new("hydro_id", DataType::Int32, false),
            Field::new("stage_id", DataType::Int32, false),
            Field::new("std_m3s", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema_no_mean,
            vec![
                Arc::new(Int32Array::from(vec![1_i32])),
                Arc::new(Int32Array::from(vec![0_i32])),
                Arc::new(Float64Array::from(vec![30.0])),
            ],
        )
        .unwrap();
        let tmp = write_parquet(&batch);
        let err = parse_inflow_seasonal_stats(tmp.path()).unwrap_err();

        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("mean_m3s"),
                    "field should contain 'mean_m3s', got: {field}"
                );
                assert!(
                    message.contains("missing required column"),
                    "message should mention missing column, got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    #[test]
    fn test_negative_std_m3s() {
        let batch = make_batch(&[1], &[0], &[150.0], &[-1.0]);
        let tmp = write_parquet(&batch);
        let err = parse_inflow_seasonal_stats(tmp.path()).unwrap_err();

        match &err {
            LoadError::SchemaError { field, .. } => {
                assert!(
                    field.contains("std_m3s"),
                    "field should contain 'std_m3s', got: {field}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    #[test]
    fn test_nan_mean_m3s() {
        let batch = make_batch(&[1], &[0], &[f64::NAN], &[30.0]);
        let tmp = write_parquet(&batch);
        let err = parse_inflow_seasonal_stats(tmp.path()).unwrap_err();

        match &err {
            LoadError::SchemaError { field, .. } => {
                assert!(
                    field.contains("mean_m3s"),
                    "field should contain 'mean_m3s', got: {field}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    #[test]
    fn test_empty_parquet_returns_empty_vec() {
        let batch = make_batch(&[], &[], &[], &[]);
        let tmp = write_parquet(&batch);
        let rows = parse_inflow_seasonal_stats(tmp.path()).unwrap();
        assert!(rows.is_empty());
    }

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
        let rows_asc = parse_inflow_seasonal_stats(tmp_asc.path()).unwrap();
        let rows_desc = parse_inflow_seasonal_stats(tmp_desc.path()).unwrap();

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
}
