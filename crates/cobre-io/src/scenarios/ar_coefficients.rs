//! Parsing for `scenarios/inflow_ar_coefficients.parquet` — AR lag coefficients
//! per (hydro, stage, lag).
//!
//! [`parse_inflow_ar_coefficients`] reads `scenarios/inflow_ar_coefficients.parquet`
//! and returns a sorted `Vec<InflowArCoefficientRow>`.
//!
//! ## Parquet schema (spec SS3.2)
//!
//! | Column               | Type   | Required | Description                                  |
//! | -------------------- | ------ | -------- | -------------------------------------------- |
//! | `hydro_id`           | INT32  | Yes      | Hydro plant ID                               |
//! | `stage_id`           | INT32  | Yes      | Stage ID                                     |
//! | `lag`                | INT32  | Yes      | Lag index (1-based)                          |
//! | `coefficient`        | DOUBLE | Yes      | AR coefficient (standardized, dimensionless) |
//! | `residual_std_ratio` | DOUBLE | Yes      | Residual std ratio in (0, 1]                 |
//!
//! ## Output ordering
//!
//! Rows are sorted by `(hydro_id, stage_id, lag)` ascending.
//!
//! ## Validation
//!
//! Per-row constraints enforced by this parser:
//!
//! - All five columns must be present with the correct types.
//! - `lag` must be ≥ 1 (lags are 1-based per spec).
//! - `residual_std_ratio` must be finite and in (0, 1].
//!
//! Deferred validations (not performed here):
//!
//! - `hydro_id` existence in the hydro registry — Layer 3, Epic 06.
//! - `stage_id` existence in the stages registry — Layer 3, Epic 06.
//! - Lag contiguity (1, 2, …, p for each (hydro, stage)) — Layer 3/5, Epic 06.
//! - Coefficient count matching `ar_order` from stats — Layer 3/5, Epic 06.
//! - `residual_std_ratio` consistency across lag rows of the same (hydro, stage) — Layer 2/5, Epic 06.

use cobre_core::EntityId;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::fs::File;
use std::path::Path;

use crate::parquet_helpers::{extract_required_float64, extract_required_int32};
use crate::LoadError;

/// A single row from `scenarios/inflow_ar_coefficients.parquet`.
///
/// Each row defines one lag coefficient for the PAR(p) model of a
/// (hydro, stage) pair. Multiple rows with the same `(hydro_id, stage_id)`
/// cover lags 1 through p, where p is the `ar_order` from
/// [`InflowSeasonalStatsRow`](super::InflowSeasonalStatsRow).
///
/// # Examples
///
/// ```
/// use cobre_io::scenarios::InflowArCoefficientRow;
/// use cobre_core::EntityId;
///
/// let row = InflowArCoefficientRow {
///     hydro_id: EntityId::from(1),
///     stage_id: 0,
///     lag: 1,
///     coefficient: 0.45,
///     residual_std_ratio: 0.85,
/// };
/// assert_eq!(row.lag, 1);
/// assert!((row.coefficient - 0.45).abs() < 1e-10);
/// assert!((row.residual_std_ratio - 0.85).abs() < 1e-10);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct InflowArCoefficientRow {
    /// Hydro plant this coefficient belongs to.
    pub hydro_id: EntityId,
    /// Stage (0-based index within `System::stages`) this coefficient applies to.
    pub stage_id: i32,
    /// Lag index, 1-based (ψ₁ = lag 1, ψ₂ = lag 2, …).
    pub lag: i32,
    /// AR coefficient `ψ*_lag`, standardized by seasonal std (dimensionless).
    pub coefficient: f64,
    /// Residual std ratio (`sigma_m` / `s_m`) for this (hydro, stage). Dimensionless, in (0, 1].
    /// Repeated across all lag rows of the same (`hydro_id`, `stage_id`) group.
    pub residual_std_ratio: f64,
}

/// Parse `scenarios/inflow_ar_coefficients.parquet` and return a sorted row table.
///
/// Reads all record batches from the Parquet file at `path`, validates per-row
/// constraints, then returns all rows sorted by `(hydro_id, stage_id, lag)` ascending.
///
/// # Errors
///
/// | Condition                                     | Error variant              |
/// |---------------------------------------------- |--------------------------- |
/// | File not found or permission denied           | [`LoadError::IoError`]     |
/// | Malformed Parquet (corrupt header, etc.)      | [`LoadError::ParseError`]  |
/// | Required column missing or wrong type         | [`LoadError::SchemaError`] |
/// | `lag` < 1                                     | [`LoadError::SchemaError`] |
/// | `residual_std_ratio` not finite or not in (0, 1] | [`LoadError::SchemaError`] |
///
/// # Examples
///
/// ```no_run
/// use cobre_io::scenarios::parse_inflow_ar_coefficients;
/// use std::path::Path;
///
/// let rows = parse_inflow_ar_coefficients(Path::new("scenarios/inflow_ar_coefficients.parquet"))
///     .expect("valid AR coefficients file");
/// println!("loaded {} AR coefficient rows", rows.len());
/// ```
pub fn parse_inflow_ar_coefficients(path: &Path) -> Result<Vec<InflowArCoefficientRow>, LoadError> {
    let file = File::open(path).map_err(|e| LoadError::io(path, e))?;

    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| LoadError::parse(path, e.to_string()))?;

    let reader = builder
        .build()
        .map_err(|e| LoadError::parse(path, e.to_string()))?;

    let mut rows: Vec<InflowArCoefficientRow> = Vec::new();

    for batch_result in reader {
        let batch = batch_result.map_err(|e| LoadError::parse(path, e.to_string()))?;

        let hydro_id_col = extract_required_int32(&batch, "hydro_id", path)?;
        let stage_id_col = extract_required_int32(&batch, "stage_id", path)?;
        let lag_col = extract_required_int32(&batch, "lag", path)?;
        let coefficient_col = extract_required_float64(&batch, "coefficient", path)?;
        let residual_std_ratio_col = extract_required_float64(&batch, "residual_std_ratio", path)?;

        let n = batch.num_rows();
        let base_idx = rows.len();
        rows.reserve(n);

        for i in 0..n {
            let row_idx = base_idx + i;

            let hydro_id = EntityId::from(hydro_id_col.value(i));
            let stage_id = stage_id_col.value(i);
            let lag = lag_col.value(i);
            let coefficient = coefficient_col.value(i);
            let residual_std_ratio = residual_std_ratio_col.value(i);

            // Validate lag: must be >= 1 (1-based per spec).
            if lag < 1 {
                return Err(LoadError::SchemaError {
                    path: path.to_path_buf(),
                    field: format!("inflow_ar_coefficients[{row_idx}].lag"),
                    message: format!("lag must be >= 1 (1-based), got {lag}"),
                });
            }

            // Validate residual_std_ratio: must be finite and in (0, 1].
            if !residual_std_ratio.is_finite()
                || residual_std_ratio <= 0.0
                || residual_std_ratio > 1.0
            {
                return Err(LoadError::SchemaError {
                    path: path.to_path_buf(),
                    field: format!("inflow_ar_coefficients[{row_idx}].residual_std_ratio"),
                    message: format!("value must be in (0, 1], got {residual_std_ratio}"),
                });
            }

            rows.push(InflowArCoefficientRow {
                hydro_id,
                stage_id,
                lag,
                coefficient,
                residual_std_ratio,
            });
        }
    }

    rows.sort_by(|a, b| {
        a.hydro_id
            .0
            .cmp(&b.hydro_id.0)
            .then_with(|| a.stage_id.cmp(&b.stage_id))
            .then_with(|| a.lag.cmp(&b.lag))
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
            Field::new("lag", DataType::Int32, false),
            Field::new("coefficient", DataType::Float64, false),
            Field::new("residual_std_ratio", DataType::Float64, false),
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
        lags: &[i32],
        coefficients: &[f64],
        residual_std_ratios: &[f64],
    ) -> RecordBatch {
        RecordBatch::try_new(
            schema(),
            vec![
                Arc::new(Int32Array::from(hydro_ids.to_vec())),
                Arc::new(Int32Array::from(stage_ids.to_vec())),
                Arc::new(Int32Array::from(lags.to_vec())),
                Arc::new(Float64Array::from(coefficients.to_vec())),
                Arc::new(Float64Array::from(residual_std_ratios.to_vec())),
            ],
        )
        .expect("valid batch")
    }

    #[test]
    fn test_valid_6_rows_sorted_by_hydro_stage_lag() {
        let batch = make_batch(
            &[2, 2, 2, 1, 1, 1],
            &[0, 0, 0, 0, 0, 0],
            &[3, 2, 1, 2, 3, 1],
            &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            &[0.85, 0.85, 0.85, 0.85, 0.85, 0.85],
        );
        let tmp = write_parquet(&batch);
        let rows = parse_inflow_ar_coefficients(tmp.path()).unwrap();

        assert_eq!(rows.len(), 6);
        assert_eq!(rows[0].hydro_id, EntityId::from(1));
        assert_eq!(rows[0].lag, 1);
        assert_eq!(rows[1].hydro_id, EntityId::from(1));
        assert_eq!(rows[1].lag, 2);
        assert_eq!(rows[2].hydro_id, EntityId::from(1));
        assert_eq!(rows[2].lag, 3);
        assert_eq!(rows[3].hydro_id, EntityId::from(2));
        assert_eq!(rows[3].lag, 1);
        assert_eq!(rows[4].hydro_id, EntityId::from(2));
        assert_eq!(rows[4].lag, 2);
        assert_eq!(rows[5].hydro_id, EntityId::from(2));
        assert_eq!(rows[5].lag, 3);
    }

    #[test]
    fn test_lag_zero_is_schema_error() {
        let batch = make_batch(&[1], &[0], &[0], &[0.45], &[0.85]);
        let tmp = write_parquet(&batch);
        let err = parse_inflow_ar_coefficients(tmp.path()).unwrap_err();

        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("lag"),
                    "field should contain 'lag', got: {field}"
                );
                assert!(
                    message.contains('1'),
                    "message should mention >= 1, got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    #[test]
    fn test_missing_coefficient_column() {
        let schema_no_coeff = Arc::new(Schema::new(vec![
            Field::new("hydro_id", DataType::Int32, false),
            Field::new("stage_id", DataType::Int32, false),
            Field::new("lag", DataType::Int32, false),
        ]));
        let batch = RecordBatch::try_new(
            schema_no_coeff,
            vec![
                Arc::new(Int32Array::from(vec![1_i32])),
                Arc::new(Int32Array::from(vec![0_i32])),
                Arc::new(Int32Array::from(vec![1_i32])),
            ],
        )
        .unwrap();
        let tmp = write_parquet(&batch);
        let err = parse_inflow_ar_coefficients(tmp.path()).unwrap_err();

        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("coefficient"),
                    "field should contain 'coefficient', got: {field}"
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
    fn test_empty_parquet_returns_empty_vec() {
        let batch = make_batch(&[], &[], &[], &[], &[]);
        let tmp = write_parquet(&batch);
        let rows = parse_inflow_ar_coefficients(tmp.path()).unwrap();
        assert!(rows.is_empty());
    }

    #[test]
    fn test_coefficient_values_preserved() {
        let batch = make_batch(&[42], &[3], &[1], &[0.12345], &[0.75]);
        let tmp = write_parquet(&batch);
        let rows = parse_inflow_ar_coefficients(tmp.path()).unwrap();

        assert_eq!(rows.len(), 1);
        let row = &rows[0];
        assert_eq!(row.hydro_id, EntityId::from(42));
        assert_eq!(row.stage_id, 3);
        assert_eq!(row.lag, 1);
        assert!((row.coefficient - 0.12345).abs() < 1e-10);
        assert!((row.residual_std_ratio - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_valid_residual_std_ratio_preserved() {
        let batch = make_batch(
            &[1, 1, 1, 2, 2, 2],
            &[0, 0, 0, 0, 0, 0],
            &[1, 2, 3, 1, 2, 3],
            &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            &[0.85, 0.85, 0.85, 0.85, 0.85, 0.85],
        );
        let tmp = write_parquet(&batch);
        let rows = parse_inflow_ar_coefficients(tmp.path()).unwrap();

        assert_eq!(rows.len(), 6);
        for row in &rows {
            assert!(
                (row.residual_std_ratio - 0.85).abs() < 1e-10,
                "expected 0.85, got {}",
                row.residual_std_ratio
            );
        }
    }

    #[test]
    fn test_residual_std_ratio_zero_is_schema_error() {
        let batch = make_batch(&[1], &[0], &[1], &[0.45], &[0.0]);
        let tmp = write_parquet(&batch);
        let err = parse_inflow_ar_coefficients(tmp.path()).unwrap_err();

        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("residual_std_ratio"),
                    "field should contain 'residual_std_ratio', got: {field}"
                );
                assert!(
                    message.contains("(0, 1]"),
                    "message should mention '(0, 1]', got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    #[test]
    fn test_residual_std_ratio_above_one_is_schema_error() {
        let batch = make_batch(&[1], &[0], &[1], &[0.45], &[1.5]);
        let tmp = write_parquet(&batch);
        let err = parse_inflow_ar_coefficients(tmp.path()).unwrap_err();

        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("residual_std_ratio"),
                    "field should contain 'residual_std_ratio', got: {field}"
                );
                assert!(
                    message.contains("(0, 1]"),
                    "message should mention '(0, 1]', got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    #[test]
    fn test_residual_std_ratio_nan_is_schema_error() {
        let batch = make_batch(&[1], &[0], &[1], &[0.45], &[f64::NAN]);
        let tmp = write_parquet(&batch);
        let err = parse_inflow_ar_coefficients(tmp.path()).unwrap_err();

        match &err {
            LoadError::SchemaError { field, .. } => {
                assert!(
                    field.contains("residual_std_ratio"),
                    "field should contain 'residual_std_ratio', got: {field}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    #[test]
    fn test_missing_residual_std_ratio_column() {
        let schema_no_ratio = Arc::new(Schema::new(vec![
            Field::new("hydro_id", DataType::Int32, false),
            Field::new("stage_id", DataType::Int32, false),
            Field::new("lag", DataType::Int32, false),
            Field::new("coefficient", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema_no_ratio,
            vec![
                Arc::new(Int32Array::from(vec![1_i32])),
                Arc::new(Int32Array::from(vec![0_i32])),
                Arc::new(Int32Array::from(vec![1_i32])),
                Arc::new(Float64Array::from(vec![0.45_f64])),
            ],
        )
        .unwrap();
        let tmp = write_parquet(&batch);
        let err = parse_inflow_ar_coefficients(tmp.path()).unwrap_err();

        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("residual_std_ratio"),
                    "field should contain 'residual_std_ratio', got: {field}"
                );
                assert!(
                    message.contains("missing required column"),
                    "message should mention missing column, got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }
}
