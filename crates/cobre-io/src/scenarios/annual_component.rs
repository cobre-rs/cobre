//! Parsing for `scenarios/inflow_annual_component.parquet` — annual component
//! statistics per (hydro, stage) for the PAR(p)-A extension.
//!
//! [`parse_inflow_annual_component`] reads
//! `scenarios/inflow_annual_component.parquet` and returns a sorted
//! `Vec<InflowAnnualComponentRow>`.
//!
//! ## Parquet schema
//!
//! | Column               | Type   | Required | Description                                  |
//! | -------------------- | ------ | -------- | -------------------------------------------- |
//! | `hydro_id`           | INT32  | Yes      | Hydro plant ID                               |
//! | `stage_id`           | INT32  | Yes      | Stage ID                                     |
//! | `annual_coefficient` | DOUBLE | Yes      | Annual component coefficient ψ (dimensionless, any finite value) |
//! | `annual_mean_m3s`    | DOUBLE | Yes      | Mean of rolling 12-month average (m³/s, any finite value) |
//! | `annual_std_m3s`     | DOUBLE | Yes      | Std of rolling 12-month average (m³/s, strictly positive) |
//!
//! ## Output ordering
//!
//! Rows are sorted by `(hydro_id, stage_id)` ascending.
//!
//! ## Validation
//!
//! Per-row constraints enforced by this parser:
//!
//! - All five columns must be present with the correct types and no null values.
//! - `annual_coefficient` must be finite (NaN and ±∞ are rejected). No range
//!   check: ψ may be negative, zero, or positive.
//! - `annual_mean_m3s` must be finite. No range check; inflow means are physical
//!   quantities and may be any finite value.
//! - `annual_std_m3s` must be finite and strictly greater than 0.0. The sample
//!   std must be positive because downstream code divides by it.
//!
//! Deferred validations (not performed here):
//!
//! - `hydro_id` existence in the hydro registry — Layer 3.
//! - `stage_id` existence in the stages registry — Layer 3.
//! - Uniqueness of `(hydro_id, stage_id)` pairs across rows — detected and
//!   rejected during assembly by [`crate::scenarios::assemble_inflow_models`].

use arrow::array::Array;
use cobre_core::EntityId;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::fs::File;
use std::path::Path;

use crate::LoadError;
use crate::parquet_helpers::{extract_required_float64, extract_required_int32};

/// A single row from `scenarios/inflow_annual_component.parquet`.
///
/// Each row defines the annual component statistics for the PAR(p)-A model
/// of a (hydro, stage) pair. The three floating-point fields correspond to
/// the annual coefficient ψ and the sample statistics (μ, σ) of the rolling
/// 12-month average inflow.
///
/// # Examples
///
/// ```
/// use cobre_io::scenarios::InflowAnnualComponentRow;
/// use cobre_core::EntityId;
///
/// let row = InflowAnnualComponentRow {
///     hydro_id: EntityId::from(1),
///     stage_id: 0,
///     annual_coefficient: -0.25,
///     annual_mean_m3s: 1500.0,
///     annual_std_m3s: 300.0,
/// };
/// assert_eq!(row.stage_id, 0);
/// assert!((row.annual_coefficient - (-0.25)).abs() < 1e-10);
/// assert!((row.annual_std_m3s - 300.0).abs() < 1e-10);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct InflowAnnualComponentRow {
    /// Hydro plant this annual component belongs to.
    pub hydro_id: EntityId,
    /// Stage (0-based index within `System::stages`) this annual component applies to.
    pub stage_id: i32,
    /// Annual component coefficient ψ (dimensionless). May be any finite value.
    pub annual_coefficient: f64,
    /// Mean of the rolling 12-month average inflow (m³/s). May be any finite value.
    pub annual_mean_m3s: f64,
    /// Standard deviation of the rolling 12-month average inflow (m³/s). Must be > 0.
    pub annual_std_m3s: f64,
}

/// Parse `scenarios/inflow_annual_component.parquet` and return a sorted row table.
///
/// Reads all record batches from the Parquet file at `path`, validates per-row
/// constraints, then returns all rows sorted by `(hydro_id, stage_id)` ascending.
///
/// # Errors
///
/// | Condition                                              | Error variant              |
/// |--------------------------------------------------------|--------------------------- |
/// | File not found or permission denied                    | [`LoadError::IoError`]     |
/// | Malformed Parquet (corrupt header, etc.)               | [`LoadError::ParseError`]  |
/// | Required column missing or wrong type                  | [`LoadError::SchemaError`] |
/// | Null value in any non-null column                      | [`LoadError::SchemaError`] |
/// | `annual_coefficient` not finite                        | [`LoadError::SchemaError`] |
/// | `annual_mean_m3s` not finite                           | [`LoadError::SchemaError`] |
/// | `annual_std_m3s` not finite or `<= 0.0`               | [`LoadError::SchemaError`] |
///
/// # Examples
///
/// ```no_run
/// use cobre_io::scenarios::parse_inflow_annual_component;
/// use std::path::Path;
///
/// let rows = parse_inflow_annual_component(Path::new("scenarios/inflow_annual_component.parquet"))
///     .expect("valid annual component file");
/// println!("loaded {} annual component rows", rows.len());
/// ```
pub fn parse_inflow_annual_component(
    path: &Path,
) -> Result<Vec<InflowAnnualComponentRow>, LoadError> {
    let file = File::open(path).map_err(|e| LoadError::io(path, e))?;

    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| LoadError::parse(path, e.to_string()))?;

    let reader = builder
        .build()
        .map_err(|e| LoadError::parse(path, e.to_string()))?;

    let mut rows: Vec<InflowAnnualComponentRow> = Vec::new();

    for batch_result in reader {
        let batch = batch_result.map_err(|e| LoadError::parse(path, e.to_string()))?;

        let hydro_id_col = extract_required_int32(&batch, "hydro_id", path)?;
        let stage_id_col = extract_required_int32(&batch, "stage_id", path)?;
        let annual_coefficient_col = extract_required_float64(&batch, "annual_coefficient", path)?;
        let annual_mean_m3s_col = extract_required_float64(&batch, "annual_mean_m3s", path)?;
        let annual_std_m3s_col = extract_required_float64(&batch, "annual_std_m3s", path)?;

        let n = batch.num_rows();
        let base_idx = rows.len();
        rows.reserve(n);

        for i in 0..n {
            let row_idx = base_idx + i;

            // Reject null values in any required column.
            if hydro_id_col.is_null(i) {
                return Err(LoadError::SchemaError {
                    path: path.to_path_buf(),
                    field: format!("inflow_annual_component[{row_idx}].hydro_id"),
                    message: "null value in non-null column hydro_id".to_string(),
                });
            }
            if stage_id_col.is_null(i) {
                return Err(LoadError::SchemaError {
                    path: path.to_path_buf(),
                    field: format!("inflow_annual_component[{row_idx}].stage_id"),
                    message: "null value in non-null column stage_id".to_string(),
                });
            }
            if annual_coefficient_col.is_null(i) {
                return Err(LoadError::SchemaError {
                    path: path.to_path_buf(),
                    field: format!("inflow_annual_component[{row_idx}].annual_coefficient"),
                    message: "null value in non-null column annual_coefficient".to_string(),
                });
            }
            if annual_mean_m3s_col.is_null(i) {
                return Err(LoadError::SchemaError {
                    path: path.to_path_buf(),
                    field: format!("inflow_annual_component[{row_idx}].annual_mean_m3s"),
                    message: "null value in non-null column annual_mean_m3s".to_string(),
                });
            }
            if annual_std_m3s_col.is_null(i) {
                return Err(LoadError::SchemaError {
                    path: path.to_path_buf(),
                    field: format!("inflow_annual_component[{row_idx}].annual_std_m3s"),
                    message: "null value in non-null column annual_std_m3s".to_string(),
                });
            }

            let hydro_id = EntityId::from(hydro_id_col.value(i));
            let stage_id = stage_id_col.value(i);
            let annual_coefficient = annual_coefficient_col.value(i);
            let annual_mean_m3s = annual_mean_m3s_col.value(i);
            let annual_std_m3s = annual_std_m3s_col.value(i);

            // Validate annual_coefficient: must be finite.
            if !annual_coefficient.is_finite() {
                return Err(LoadError::SchemaError {
                    path: path.to_path_buf(),
                    field: format!("inflow_annual_component[{row_idx}].annual_coefficient"),
                    message: format!(
                        "value must be finite (NaN and ±infinity are rejected), got {annual_coefficient}"
                    ),
                });
            }

            // Validate annual_mean_m3s: must be finite.
            if !annual_mean_m3s.is_finite() {
                return Err(LoadError::SchemaError {
                    path: path.to_path_buf(),
                    field: format!("inflow_annual_component[{row_idx}].annual_mean_m3s"),
                    message: format!(
                        "value must be finite (NaN and ±infinity are rejected), got {annual_mean_m3s}"
                    ),
                });
            }

            // Validate annual_std_m3s: must be finite and strictly positive.
            if !annual_std_m3s.is_finite() || annual_std_m3s <= 0.0 {
                return Err(LoadError::SchemaError {
                    path: path.to_path_buf(),
                    field: format!("inflow_annual_component[{row_idx}].annual_std_m3s"),
                    message: format!(
                        "value must be finite and > 0.0 (sample std must be positive), got {annual_std_m3s}"
                    ),
                });
            }

            rows.push(InflowAnnualComponentRow {
                hydro_id,
                stage_id,
                annual_coefficient,
                annual_mean_m3s,
                annual_std_m3s,
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
            Field::new("annual_coefficient", DataType::Float64, false),
            Field::new("annual_mean_m3s", DataType::Float64, false),
            Field::new("annual_std_m3s", DataType::Float64, false),
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
        annual_coefficients: &[f64],
        annual_means: &[f64],
        annual_stds: &[f64],
    ) -> RecordBatch {
        RecordBatch::try_new(
            schema(),
            vec![
                Arc::new(Int32Array::from(hydro_ids.to_vec())),
                Arc::new(Int32Array::from(stage_ids.to_vec())),
                Arc::new(Float64Array::from(annual_coefficients.to_vec())),
                Arc::new(Float64Array::from(annual_means.to_vec())),
                Arc::new(Float64Array::from(annual_stds.to_vec())),
            ],
        )
        .expect("valid batch")
    }

    /// AC #1: Three rows for two hydros across two stages are returned sorted by
    /// `(hydro_id, stage_id)` ascending.
    #[test]
    fn test_valid_three_rows_sorted_by_hydro_stage() {
        // Deliberately insert out-of-order to confirm sorting.
        let batch = make_batch(
            &[2, 1, 1],
            &[0, 1, 0],
            &[0.5, -0.1, 0.3],
            &[2000.0, 1200.0, 1500.0],
            &[400.0, 150.0, 300.0],
        );
        let tmp = write_parquet(&batch);
        let rows = parse_inflow_annual_component(tmp.path()).unwrap();

        assert_eq!(rows.len(), 3);

        // Sorted order: hydro 1 stage 0, hydro 1 stage 1, hydro 2 stage 0.
        assert_eq!(rows[0].hydro_id, EntityId::from(1));
        assert_eq!(rows[0].stage_id, 0);
        assert!((rows[0].annual_coefficient - 0.3).abs() < 1e-10);
        assert!((rows[0].annual_mean_m3s - 1500.0).abs() < 1e-10);
        assert!((rows[0].annual_std_m3s - 300.0).abs() < 1e-10);

        assert_eq!(rows[1].hydro_id, EntityId::from(1));
        assert_eq!(rows[1].stage_id, 1);
        assert!((rows[1].annual_coefficient - (-0.1)).abs() < 1e-10);

        assert_eq!(rows[2].hydro_id, EntityId::from(2));
        assert_eq!(rows[2].stage_id, 0);
        assert!((rows[2].annual_coefficient - 0.5).abs() < 1e-10);
    }

    /// AC #2: A row with `annual_coefficient = NaN` returns `SchemaError` where
    /// `field` contains `annual_coefficient` and `message` contains `finite`.
    #[test]
    fn test_annual_coefficient_nan_is_schema_error() {
        let batch = make_batch(&[1], &[0], &[f64::NAN], &[1500.0], &[300.0]);
        let tmp = write_parquet(&batch);
        let err = parse_inflow_annual_component(tmp.path()).unwrap_err();

        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("annual_coefficient"),
                    "field should contain 'annual_coefficient', got: {field}"
                );
                assert!(
                    message.contains("finite"),
                    "message should contain 'finite', got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// AC #3: A row with `annual_std_m3s = 0.0` returns `SchemaError` where
    /// `field` contains `annual_std_m3s` and `message` mentions positivity.
    #[test]
    fn test_annual_std_zero_is_schema_error() {
        let batch = make_batch(&[1], &[0], &[0.3], &[1500.0], &[0.0]);
        let tmp = write_parquet(&batch);
        let err = parse_inflow_annual_component(tmp.path()).unwrap_err();

        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("annual_std_m3s"),
                    "field should contain 'annual_std_m3s', got: {field}"
                );
                assert!(
                    message.contains("positive") || message.contains("> 0"),
                    "message should mention positivity violation, got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// AC #4: A row with `annual_std_m3s = f64::INFINITY` returns `SchemaError`
    /// where `field` contains `annual_std_m3s`.
    #[test]
    fn test_annual_std_infinity_is_schema_error() {
        let batch = make_batch(&[1], &[0], &[0.3], &[1500.0], &[f64::INFINITY]);
        let tmp = write_parquet(&batch);
        let err = parse_inflow_annual_component(tmp.path()).unwrap_err();

        match &err {
            LoadError::SchemaError { field, .. } => {
                assert!(
                    field.contains("annual_std_m3s"),
                    "field should contain 'annual_std_m3s', got: {field}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// AC #5: A parquet file missing the `annual_mean_m3s` column returns
    /// `SchemaError` where `field` contains `annual_mean_m3s` and `message`
    /// contains `missing required column`.
    #[test]
    fn test_missing_annual_mean_m3s_column() {
        let schema_no_mean = Arc::new(Schema::new(vec![
            Field::new("hydro_id", DataType::Int32, false),
            Field::new("stage_id", DataType::Int32, false),
            Field::new("annual_coefficient", DataType::Float64, false),
            Field::new("annual_std_m3s", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema_no_mean,
            vec![
                Arc::new(Int32Array::from(vec![1_i32])),
                Arc::new(Int32Array::from(vec![0_i32])),
                Arc::new(Float64Array::from(vec![0.3_f64])),
                Arc::new(Float64Array::from(vec![300.0_f64])),
            ],
        )
        .unwrap();
        let tmp = write_parquet(&batch);
        let err = parse_inflow_annual_component(tmp.path()).unwrap_err();

        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("annual_mean_m3s"),
                    "field should contain 'annual_mean_m3s', got: {field}"
                );
                assert!(
                    message.contains("missing required column"),
                    "message should mention missing column, got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// AC #6: A parquet file where `annual_coefficient` is declared nullable and
    /// contains a null at row 0 returns `SchemaError` with `field` referencing
    /// `annual_coefficient`.
    #[test]
    fn test_null_annual_coefficient_value_is_schema_error() {
        let schema_nullable_coeff = Arc::new(Schema::new(vec![
            Field::new("hydro_id", DataType::Int32, false),
            Field::new("stage_id", DataType::Int32, false),
            Field::new("annual_coefficient", DataType::Float64, true), // nullable
            Field::new("annual_mean_m3s", DataType::Float64, false),
            Field::new("annual_std_m3s", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema_nullable_coeff,
            vec![
                Arc::new(Int32Array::from(vec![1_i32])),
                Arc::new(Int32Array::from(vec![0_i32])),
                Arc::new(Float64Array::from(vec![None::<f64>])), // null
                Arc::new(Float64Array::from(vec![1500.0_f64])),
                Arc::new(Float64Array::from(vec![300.0_f64])),
            ],
        )
        .unwrap();
        let tmp = write_parquet(&batch);
        let err = parse_inflow_annual_component(tmp.path()).unwrap_err();

        match &err {
            LoadError::SchemaError { field, .. } => {
                assert!(
                    field.contains("annual_coefficient"),
                    "field should contain 'annual_coefficient', got: {field}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// A nullable `hydro_id` column with a null at row 0 returns `SchemaError`
    /// referencing `hydro_id`.
    #[test]
    fn test_null_hydro_id_value_is_schema_error() {
        let schema_nullable_hydro = Arc::new(Schema::new(vec![
            Field::new("hydro_id", DataType::Int32, true), // nullable
            Field::new("stage_id", DataType::Int32, false),
            Field::new("annual_coefficient", DataType::Float64, false),
            Field::new("annual_mean_m3s", DataType::Float64, false),
            Field::new("annual_std_m3s", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema_nullable_hydro,
            vec![
                Arc::new(Int32Array::from(vec![None::<i32>])), // null
                Arc::new(Int32Array::from(vec![0_i32])),
                Arc::new(Float64Array::from(vec![0.15_f64])),
                Arc::new(Float64Array::from(vec![1500.0_f64])),
                Arc::new(Float64Array::from(vec![300.0_f64])),
            ],
        )
        .unwrap();
        let tmp = write_parquet(&batch);
        let err = parse_inflow_annual_component(tmp.path()).unwrap_err();

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

    /// A nullable `stage_id` column with a null at row 0 returns `SchemaError`
    /// referencing `stage_id`.
    #[test]
    fn test_null_stage_id_value_is_schema_error() {
        let schema_nullable_stage = Arc::new(Schema::new(vec![
            Field::new("hydro_id", DataType::Int32, false),
            Field::new("stage_id", DataType::Int32, true), // nullable
            Field::new("annual_coefficient", DataType::Float64, false),
            Field::new("annual_mean_m3s", DataType::Float64, false),
            Field::new("annual_std_m3s", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema_nullable_stage,
            vec![
                Arc::new(Int32Array::from(vec![1_i32])),
                Arc::new(Int32Array::from(vec![None::<i32>])), // null
                Arc::new(Float64Array::from(vec![0.15_f64])),
                Arc::new(Float64Array::from(vec![1500.0_f64])),
                Arc::new(Float64Array::from(vec![300.0_f64])),
            ],
        )
        .unwrap();
        let tmp = write_parquet(&batch);
        let err = parse_inflow_annual_component(tmp.path()).unwrap_err();

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

    /// AC #7: A non-existent path returns `LoadError::IoError`.
    #[test]
    fn test_nonexistent_path_returns_io_error() {
        let err = parse_inflow_annual_component(std::path::Path::new(
            "/nonexistent/path/inflow_annual_component.parquet",
        ))
        .unwrap_err();

        match &err {
            LoadError::IoError { .. } => {}
            other => panic!("expected IoError, got: {other:?}"),
        }
    }

    /// AC #8: A zero-row parquet with the correct schema returns `Ok(vec![])`.
    #[test]
    fn test_empty_parquet_returns_empty_vec() {
        let batch = make_batch(&[], &[], &[], &[], &[]);
        let tmp = write_parquet(&batch);
        let rows = parse_inflow_annual_component(tmp.path()).unwrap();
        assert!(rows.is_empty());
    }

    /// Sanity: a negative ψ value (`-0.25`) round-trips without rejection.
    /// Confirms the "no range check" rule for `annual_coefficient`.
    #[test]
    fn test_negative_psi_value_preserved() {
        let batch = make_batch(&[42], &[3], &[-0.25], &[1200.0], &[250.0]);
        let tmp = write_parquet(&batch);
        let rows = parse_inflow_annual_component(tmp.path()).unwrap();

        assert_eq!(rows.len(), 1);
        let row = &rows[0];
        assert_eq!(row.hydro_id, EntityId::from(42));
        assert_eq!(row.stage_id, 3);
        assert!((row.annual_coefficient - (-0.25)).abs() < 1e-10);
        assert!((row.annual_mean_m3s - 1200.0).abs() < 1e-10);
        assert!((row.annual_std_m3s - 250.0).abs() < 1e-10);
    }
}
