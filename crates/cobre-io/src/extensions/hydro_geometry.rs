//! Parsing for `system/hydro_geometry.parquet` — Volume-Height-Area (VHA) curves.
//!
//! [`parse_hydro_geometry`] reads `system/hydro_geometry.parquet` from the case
//! directory and returns a flat, sorted `Vec<HydroGeometryRow>`.
//!
//! ## Parquet schema
//!
//! | Column       | Parquet type | Description                        |
//! |------------- |------------- |------------------------------------|
//! | `hydro_id`   | INT32        | Hydro plant identifier             |
//! | `volume_hm3` | DOUBLE       | Total reservoir volume (hm³)       |
//! | `height_m`   | DOUBLE       | Reservoir surface elevation (m)    |
//! | `area_km2`   | DOUBLE       | Water surface area (km²)           |
//!
//! ## Output ordering
//!
//! Rows are sorted by `(hydro_id, volume_hm3)` ascending. Multiple rows per
//! hydro constitute the VHA curve for that plant.
//!
//! ## Validation
//!
//! Per-row constraints enforced by this parser:
//!
//! - All four columns must be present with the correct types.
//! - `volume_hm3`, `height_m`, and `area_km2` must all be non-negative and finite
//!   (NaN and `+inf` / `-inf` are rejected).
//!
//! Deferred validations (not performed here):
//!
//! - Monotonicity (`volume_hm3` increasing within each hydro) — Layer 5, Epic 06.
//! - `hydro_id` existence in the hydro registry — Layer 3, Epic 06.

use arrow::array::{Array, Float64Array, Int32Array};
use cobre_core::EntityId;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::fs::File;
use std::path::Path;

use crate::LoadError;

/// A single row from `system/hydro_geometry.parquet`.
///
/// Each row is one point on the Volume-Height-Area (VHA) curve for the hydro
/// plant identified by `hydro_id`. A complete curve consists of multiple rows
/// with the same `hydro_id`, sorted by ascending `volume_hm3`.
///
/// # Examples
///
/// ```
/// use cobre_io::extensions::HydroGeometryRow;
/// use cobre_core::EntityId;
///
/// let row = HydroGeometryRow {
///     hydro_id: EntityId::from(42),
///     volume_hm3: 100.0,
///     height_m: 380.0,
///     area_km2: 4.2,
/// };
/// assert_eq!(row.hydro_id, EntityId::from(42));
/// assert_eq!(row.volume_hm3, 100.0);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct HydroGeometryRow {
    /// Hydro plant this curve point belongs to.
    pub hydro_id: EntityId,
    /// Total reservoir volume at this point (hm³). Non-negative.
    pub volume_hm3: f64,
    /// Reservoir surface elevation at this volume (m). Non-negative.
    pub height_m: f64,
    /// Water surface area at this volume (km²). Non-negative.
    pub area_km2: f64,
}

/// Parse `system/hydro_geometry.parquet` and return a sorted VHA curve table.
///
/// Reads all record batches from the Parquet file at `path`, validates per-row
/// constraints, then returns all rows sorted by `(hydro_id, volume_hm3)` ascending.
///
/// # Errors
///
/// | Condition                                       | Error variant              |
/// |------------------------------------------------ |--------------------------- |
/// | File not found or permission denied             | [`LoadError::IoError`]     |
/// | Malformed Parquet (corrupt header, etc.)        | [`LoadError::ParseError`]  |
/// | Required column missing or wrong type           | [`LoadError::SchemaError`] |
/// | Negative or non-finite `volume_hm3` / `height_m` / `area_km2` | [`LoadError::SchemaError`] |
///
/// # Examples
///
/// ```no_run
/// use cobre_io::extensions::parse_hydro_geometry;
/// use std::path::Path;
///
/// let rows = parse_hydro_geometry(Path::new("system/hydro_geometry.parquet"))
///     .expect("valid geometry file");
/// println!("loaded {} VHA curve points", rows.len());
/// ```
pub fn parse_hydro_geometry(path: &Path) -> Result<Vec<HydroGeometryRow>, LoadError> {
    let file = File::open(path).map_err(|e| LoadError::io(path, e))?;

    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| LoadError::parse(path, e.to_string()))?;

    let reader = builder
        .build()
        .map_err(|e| LoadError::parse(path, e.to_string()))?;

    let mut rows: Vec<HydroGeometryRow> = Vec::new();

    for batch_result in reader {
        let batch = batch_result.map_err(|e| LoadError::parse(path, e.to_string()))?;

        // ── Extract columns by name ───────────────────────────────────────────
        let hydro_id_col = extract_int32_column(&batch, "hydro_id", path)?;
        let volume_col = extract_float64_column(&batch, "volume_hm3", path)?;
        let height_col = extract_float64_column(&batch, "height_m", path)?;
        let area_col = extract_float64_column(&batch, "area_km2", path)?;

        // ── Build rows with per-row validation ───────────────────────────────
        let n = batch.num_rows();
        let base_idx = rows.len();
        rows.reserve(n);

        for i in 0..n {
            let row_idx = base_idx + i;

            let hydro_id = EntityId::from(hydro_id_col.value(i));

            let volume_hm3 =
                validate_non_negative(volume_col.value(i), row_idx, "volume_hm3", path)?;
            let height_m = validate_non_negative(height_col.value(i), row_idx, "height_m", path)?;
            let area_km2 = validate_non_negative(area_col.value(i), row_idx, "area_km2", path)?;

            rows.push(HydroGeometryRow {
                hydro_id,
                volume_hm3,
                height_m,
                area_km2,
            });
        }
    }

    // ── Sort by (hydro_id, volume_hm3) ascending ─────────────────────────────
    rows.sort_by(|a, b| {
        a.hydro_id.0.cmp(&b.hydro_id.0).then_with(|| {
            a.volume_hm3
                .partial_cmp(&b.volume_hm3)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    });

    Ok(rows)
}

/// Extract a column as [`Int32Array`] by name, returning [`LoadError::SchemaError`]
/// if the column is absent or has the wrong Arrow type.
fn extract_int32_column<'a>(
    batch: &'a arrow::record_batch::RecordBatch,
    name: &str,
    path: &Path,
) -> Result<&'a Int32Array, LoadError> {
    let col = batch
        .column_by_name(name)
        .ok_or_else(|| LoadError::SchemaError {
            path: path.to_path_buf(),
            field: name.to_string(),
            message: format!("missing column \"{name}\""),
        })?;
    col.as_any()
        .downcast_ref::<Int32Array>()
        .ok_or_else(|| LoadError::SchemaError {
            path: path.to_path_buf(),
            field: name.to_string(),
            message: format!(
                "column \"{name}\" has type {} but Int32 is required",
                col.data_type()
            ),
        })
}

/// Extract a column as [`Float64Array`] by name, returning [`LoadError::SchemaError`]
/// if the column is absent or has the wrong Arrow type.
fn extract_float64_column<'a>(
    batch: &'a arrow::record_batch::RecordBatch,
    name: &str,
    path: &Path,
) -> Result<&'a Float64Array, LoadError> {
    let col = batch
        .column_by_name(name)
        .ok_or_else(|| LoadError::SchemaError {
            path: path.to_path_buf(),
            field: name.to_string(),
            message: format!("missing column \"{name}\""),
        })?;
    col.as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| LoadError::SchemaError {
            path: path.to_path_buf(),
            field: name.to_string(),
            message: format!(
                "column \"{name}\" has type {} but Float64 is required",
                col.data_type()
            ),
        })
}

/// Validate that a `f64` value is non-negative and finite, returning a
/// [`LoadError::SchemaError`] with the field path `"hydro_geometry[N].column_name"`
/// if the check fails.
fn validate_non_negative(
    value: f64,
    row_idx: usize,
    column: &str,
    path: &Path,
) -> Result<f64, LoadError> {
    if value.is_finite() && value >= 0.0 {
        Ok(value)
    } else {
        Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: format!("hydro_geometry[{row_idx}].{column}"),
            message: format!("value must be non-negative and finite, got {value}"),
        })
    }
}

#[cfg(test)]
#[allow(
    clippy::doc_markdown,
    clippy::expect_used,
    clippy::float_cmp,
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

    /// Build a record batch with the canonical four-column schema.
    fn make_batch(
        hydro_ids: &[i32],
        volumes: &[f64],
        heights: &[f64],
        areas: &[f64],
    ) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("hydro_id", DataType::Int32, false),
            Field::new("volume_hm3", DataType::Float64, false),
            Field::new("height_m", DataType::Float64, false),
            Field::new("area_km2", DataType::Float64, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(hydro_ids.to_vec())),
                Arc::new(Float64Array::from(volumes.to_vec())),
                Arc::new(Float64Array::from(heights.to_vec())),
                Arc::new(Float64Array::from(areas.to_vec())),
            ],
        )
        .expect("valid batch construction")
    }

    /// Write a single record batch to a temporary Parquet file and return the
    /// temporary file handle (keeps the file alive until dropped).
    fn write_parquet(batch: &RecordBatch) -> NamedTempFile {
        let tmp = NamedTempFile::new().expect("tempfile");
        let mut writer = ArrowWriter::try_new(tmp.reopen().expect("reopen"), batch.schema(), None)
            .expect("ArrowWriter");
        writer.write(batch).expect("write batch");
        writer.close().expect("close writer");
        tmp
    }

    /// Write multiple record batches to a temporary Parquet file.
    fn write_parquet_batches(batches: &[RecordBatch]) -> NamedTempFile {
        assert!(!batches.is_empty(), "must provide at least one batch");
        let tmp = NamedTempFile::new().expect("tempfile");
        let mut writer =
            ArrowWriter::try_new(tmp.reopen().expect("reopen"), batches[0].schema(), None)
                .expect("ArrowWriter");
        for batch in batches {
            writer.write(batch).expect("write batch");
        }
        writer.close().expect("close writer");
        tmp
    }

    // ── AC: valid single-hydro file ───────────────────────────────────────────

    /// Valid Parquet with 5 rows for hydro 42 (Sobradinho-style VHA curve).
    /// Result: Ok with 5 rows, all hydro_id = EntityId(42), sorted by volume.
    #[test]
    fn test_valid_single_hydro_five_rows() {
        let batch = make_batch(
            &[42, 42, 42, 42, 42],
            &[0.0, 2_000.0, 10_000.0, 24_500.0, 34_116.0],
            &[386.5, 390.0, 396.0, 400.5, 401.3],
            &[2.5, 3.1, 5.2, 6.8, 7.0],
        );
        let tmp = write_parquet(&batch);
        let rows = parse_hydro_geometry(tmp.path()).unwrap();

        assert_eq!(rows.len(), 5, "expected 5 rows");
        for row in &rows {
            assert_eq!(row.hydro_id, EntityId::from(42));
        }
        // Verify sorted by volume_hm3
        let volumes: Vec<f64> = rows.iter().map(|r| r.volume_hm3).collect();
        assert_eq!(volumes, vec![0.0, 2_000.0, 10_000.0, 24_500.0, 34_116.0]);
    }

    // ── AC: sorted output for multiple hydros ─────────────────────────────────

    /// Rows for 2 hydros in arbitrary order -> result sorted by (hydro_id, volume_hm3).
    #[test]
    fn test_multiple_hydros_sorted() {
        // Intentionally interleaved and out-of-order
        let batch = make_batch(
            &[99, 10, 10, 99, 10],
            &[500.0, 200.0, 100.0, 100.0, 300.0],
            &[350.0, 310.0, 305.0, 345.0, 320.0],
            &[1.0, 0.5, 0.4, 0.9, 0.6],
        );
        let tmp = write_parquet(&batch);
        let rows = parse_hydro_geometry(tmp.path()).unwrap();

        assert_eq!(rows.len(), 5);

        // First three rows: hydro 10, sorted by volume
        assert_eq!(rows[0].hydro_id, EntityId::from(10));
        assert_eq!(rows[0].volume_hm3, 100.0);
        assert_eq!(rows[1].hydro_id, EntityId::from(10));
        assert_eq!(rows[1].volume_hm3, 200.0);
        assert_eq!(rows[2].hydro_id, EntityId::from(10));
        assert_eq!(rows[2].volume_hm3, 300.0);

        // Last two rows: hydro 99, sorted by volume
        assert_eq!(rows[3].hydro_id, EntityId::from(99));
        assert_eq!(rows[3].volume_hm3, 100.0);
        assert_eq!(rows[4].hydro_id, EntityId::from(99));
        assert_eq!(rows[4].volume_hm3, 500.0);
    }

    // ── AC: missing column ────────────────────────────────────────────────────

    /// Parquet file missing `area_km2` column -> SchemaError with field "area_km2"
    /// and message containing "missing column".
    #[test]
    fn test_missing_area_km2_column() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("hydro_id", DataType::Int32, false),
            Field::new("volume_hm3", DataType::Float64, false),
            Field::new("height_m", DataType::Float64, false),
            // area_km2 is deliberately omitted
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![42])),
                Arc::new(Float64Array::from(vec![100.0])),
                Arc::new(Float64Array::from(vec![380.0])),
            ],
        )
        .unwrap();
        let tmp = write_parquet(&batch);
        let err = parse_hydro_geometry(tmp.path()).unwrap_err();

        match err {
            LoadError::SchemaError { field, message, .. } => {
                assert_eq!(
                    field, "area_km2",
                    "field should be 'area_km2', got: {field}"
                );
                assert!(
                    message.contains("missing column"),
                    "message should contain 'missing column', got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// Parquet file missing `hydro_id` column -> SchemaError with field "hydro_id".
    #[test]
    fn test_missing_hydro_id_column() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("volume_hm3", DataType::Float64, false),
            Field::new("height_m", DataType::Float64, false),
            Field::new("area_km2", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![100.0])),
                Arc::new(Float64Array::from(vec![380.0])),
                Arc::new(Float64Array::from(vec![4.0])),
            ],
        )
        .unwrap();
        let tmp = write_parquet(&batch);
        let err = parse_hydro_geometry(tmp.path()).unwrap_err();

        match err {
            LoadError::SchemaError { field, message, .. } => {
                assert_eq!(field, "hydro_id");
                assert!(message.contains("missing column"));
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    // ── AC: wrong column type ─────────────────────────────────────────────────

    /// `hydro_id` provided as Float64 instead of Int32 -> SchemaError.
    #[test]
    fn test_wrong_type_hydro_id_as_float64() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("hydro_id", DataType::Float64, false), // wrong type
            Field::new("volume_hm3", DataType::Float64, false),
            Field::new("height_m", DataType::Float64, false),
            Field::new("area_km2", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![42.0_f64])),
                Arc::new(Float64Array::from(vec![100.0])),
                Arc::new(Float64Array::from(vec![380.0])),
                Arc::new(Float64Array::from(vec![4.0])),
            ],
        )
        .unwrap();
        let tmp = write_parquet(&batch);
        let err = parse_hydro_geometry(tmp.path()).unwrap_err();

        match err {
            LoadError::SchemaError { field, .. } => {
                assert_eq!(field, "hydro_id");
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    // ── AC: negative value ────────────────────────────────────────────────────

    /// Negative `volume_hm3` value -> SchemaError with field containing "volume_hm3".
    #[test]
    fn test_negative_volume_hm3() {
        let batch = make_batch(
            &[42, 42],
            &[100.0, -50.0], // second row is negative
            &[380.0, 375.0],
            &[4.0, 3.5],
        );
        let tmp = write_parquet(&batch);
        let err = parse_hydro_geometry(tmp.path()).unwrap_err();

        match err {
            LoadError::SchemaError { field, .. } => {
                assert!(
                    field.contains("volume_hm3"),
                    "field should contain 'volume_hm3', got: {field}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// Negative `height_m` value -> SchemaError with field containing "height_m".
    #[test]
    fn test_negative_height_m() {
        let batch = make_batch(&[42], &[100.0], &[-5.0], &[4.0]);
        let tmp = write_parquet(&batch);
        let err = parse_hydro_geometry(tmp.path()).unwrap_err();

        match err {
            LoadError::SchemaError { field, .. } => {
                assert!(field.contains("height_m"));
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// Negative `area_km2` value -> SchemaError with field containing "area_km2".
    #[test]
    fn test_negative_area_km2() {
        let batch = make_batch(&[42], &[100.0], &[380.0], &[-1.0]);
        let tmp = write_parquet(&batch);
        let err = parse_hydro_geometry(tmp.path()).unwrap_err();

        match err {
            LoadError::SchemaError { field, .. } => {
                assert!(field.contains("area_km2"));
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    // ── AC: file not found ────────────────────────────────────────────────────

    /// Non-existent path -> IoError with the matching path.
    #[test]
    fn test_file_not_found() {
        let path = Path::new("/nonexistent/path/hydro_geometry.parquet");
        let err = parse_hydro_geometry(path).unwrap_err();

        match err {
            LoadError::IoError { path: err_path, .. } => {
                assert_eq!(err_path, path);
            }
            other => panic!("expected IoError, got: {other:?}"),
        }
    }

    // ── AC: empty file ────────────────────────────────────────────────────────

    /// Empty Parquet (zero rows) -> Ok(Vec::new()).
    #[test]
    fn test_empty_parquet_returns_empty_vec() {
        let batch = make_batch(&[], &[], &[], &[]);
        let tmp = write_parquet(&batch);
        let rows = parse_hydro_geometry(tmp.path()).unwrap();
        assert!(rows.is_empty(), "expected empty vec for empty Parquet");
    }

    // ── Additional: NaN value rejected ───────────────────────────────────────

    /// NaN `volume_hm3` -> SchemaError (non-finite values are not allowed).
    #[test]
    fn test_nan_volume_rejected() {
        let batch = make_batch(&[42], &[f64::NAN], &[380.0], &[4.0]);
        let tmp = write_parquet(&batch);
        let err = parse_hydro_geometry(tmp.path()).unwrap_err();

        match err {
            LoadError::SchemaError { field, .. } => {
                assert!(field.contains("volume_hm3"));
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// Positive infinity `area_km2` -> SchemaError (non-finite values not allowed).
    #[test]
    fn test_infinite_area_rejected() {
        let batch = make_batch(&[42], &[100.0], &[380.0], &[f64::INFINITY]);
        let tmp = write_parquet(&batch);
        let err = parse_hydro_geometry(tmp.path()).unwrap_err();

        match err {
            LoadError::SchemaError { field, .. } => {
                assert!(field.contains("area_km2"));
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    // ── Additional: field values are preserved correctly ──────────────────────

    /// Verify that all four field values are correctly round-tripped through
    /// the Parquet read path.
    #[test]
    fn test_field_values_preserved() {
        let batch = make_batch(&[7], &[1234.5], &[399.8], &[6.123]);
        let tmp = write_parquet(&batch);
        let rows = parse_hydro_geometry(tmp.path()).unwrap();

        assert_eq!(rows.len(), 1);
        let row = &rows[0];
        assert_eq!(row.hydro_id, EntityId::from(7));
        assert!((row.volume_hm3 - 1234.5).abs() < f64::EPSILON);
        assert!((row.height_m - 399.8).abs() < f64::EPSILON);
        assert!((row.area_km2 - 6.123).abs() < f64::EPSILON);
    }

    // ── Additional: multi-batch file ──────────────────────────────────────────

    /// A Parquet file with multiple record batches is fully read and sorted.
    #[test]
    fn test_multiple_record_batches() {
        let batch1 = make_batch(&[5, 5], &[200.0, 100.0], &[310.0, 305.0], &[0.5, 0.4]);
        let batch2 = make_batch(&[5], &[300.0], &[320.0], &[0.6]);
        let tmp = write_parquet_batches(&[batch1, batch2]);
        let rows = parse_hydro_geometry(tmp.path()).unwrap();

        assert_eq!(rows.len(), 3);
        // Must be sorted by volume_hm3
        assert_eq!(rows[0].volume_hm3, 100.0);
        assert_eq!(rows[1].volume_hm3, 200.0);
        assert_eq!(rows[2].volume_hm3, 300.0);
    }

    // ── Additional: zero volume is valid ─────────────────────────────────────

    /// A row with volume_hm3 = 0.0 is valid (dead storage at minimum).
    #[test]
    fn test_zero_volume_is_valid() {
        let batch = make_batch(&[42], &[0.0], &[380.0], &[2.5]);
        let tmp = write_parquet(&batch);
        let rows = parse_hydro_geometry(tmp.path()).unwrap();

        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].volume_hm3, 0.0);
    }

    /// The `field` in `SchemaError` for a negative value uses the format
    /// `"hydro_geometry[N].column_name"` with the correct row index.
    #[test]
    fn test_schema_error_field_format_includes_row_index() {
        // Row 0 is valid; row 1 has a negative volume
        let batch = make_batch(&[42, 42], &[100.0, -10.0], &[380.0, 375.0], &[4.0, 3.5]);
        let tmp = write_parquet(&batch);
        let err = parse_hydro_geometry(tmp.path()).unwrap_err();

        match err {
            LoadError::SchemaError { field, .. } => {
                assert!(
                    field.contains('['),
                    "field should include row index, got: {field}"
                );
                assert!(
                    field.contains("volume_hm3"),
                    "field should name the column, got: {field}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }
}
