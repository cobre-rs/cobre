//! Parquet parser for `constraints/ncs_bounds.parquet` — per-stage NCS available generation.
//!
//! [`parse_ncs_bounds`] reads a Parquet file containing the available generation
//! schedule for non-controllable sources. Each row specifies the maximum generation
//! (MW) for a given NCS entity at a given stage.
//!
//! ## Parquet schema
//!
//! | Column                    | Type   | Required | Description                        |
//! | ------------------------- | ------ | -------- | ---------------------------------- |
//! | `ncs_id`                  | INT32  | Yes      | Non-controllable source ID         |
//! | `stage_id`                | INT32  | Yes      | Stage ID                           |
//! | `available_generation_mw` | DOUBLE | Yes      | Available generation (MW)          |
//!
//! ## Output ordering
//!
//! Rows are sorted by `(ncs_id, stage_id)` ascending.
//!
//! ## Validation
//!
//! - Required columns must be present with correct types.
//! - `available_generation_mw` must be finite and >= 0.0.

use cobre_core::EntityId;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::fs::File;
use std::path::Path;

use crate::parquet_helpers::{extract_required_float64, extract_required_int32};
use crate::LoadError;

// ── Row type ─────────────────────────────────────────────────────────────────

/// A single row from `constraints/ncs_bounds.parquet`.
///
/// Carries the per-stage available generation for a non-controllable source.
///
/// # Examples
///
/// ```
/// use cobre_io::constraints::NcsBoundsRow;
/// use cobre_core::EntityId;
///
/// let row = NcsBoundsRow {
///     ncs_id: EntityId::from(0),
///     stage_id: 3,
///     available_generation_mw: 150.0,
/// };
/// assert_eq!(row.ncs_id, EntityId::from(0));
/// assert!((row.available_generation_mw - 150.0).abs() < f64::EPSILON);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct NcsBoundsRow {
    /// Non-controllable source ID.
    pub ncs_id: EntityId,
    /// Stage ID.
    pub stage_id: i32,
    /// Available generation (MW). Must be >= 0.0.
    pub available_generation_mw: f64,
}

// ── Parser ───────────────────────────────────────────────────────────────────

/// Parse `constraints/ncs_bounds.parquet` and return a sorted row table.
///
/// Reads all record batches from the Parquet file at `path`, validates per-row
/// constraints, then returns all rows sorted by `(ncs_id, stage_id)` ascending.
///
/// # Errors
///
/// | Condition                                     | Error variant              |
/// |-----------------------------------------------|--------------------------- |
/// | File not found or permission denied           | [`LoadError::IoError`]     |
/// | Malformed Parquet (corrupt header, etc.)      | [`LoadError::ParseError`]  |
/// | Required column missing or wrong type         | [`LoadError::SchemaError`] |
/// | Negative or non-finite `available_generation_mw` | [`LoadError::SchemaError`] |
///
/// # Examples
///
/// ```no_run
/// use cobre_io::constraints::parse_ncs_bounds;
/// use std::path::Path;
///
/// let rows = parse_ncs_bounds(Path::new("constraints/ncs_bounds.parquet"))
///     .expect("valid NCS bounds file");
/// println!("loaded {} NCS bounds rows", rows.len());
/// ```
pub fn parse_ncs_bounds(path: &Path) -> Result<Vec<NcsBoundsRow>, LoadError> {
    let file = File::open(path).map_err(|e| LoadError::io(path, e))?;

    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| LoadError::parse(path, e.to_string()))?;

    let reader = builder
        .build()
        .map_err(|e| LoadError::parse(path, e.to_string()))?;

    let mut rows: Vec<NcsBoundsRow> = Vec::new();

    for batch_result in reader {
        let batch = batch_result.map_err(|e| LoadError::parse(path, e.to_string()))?;

        let ncs_id_col = extract_required_int32(&batch, "ncs_id", path)?;
        let stage_id_col = extract_required_int32(&batch, "stage_id", path)?;
        let avail_gen_col = extract_required_float64(&batch, "available_generation_mw", path)?;

        let n = batch.num_rows();
        let base_idx = rows.len();
        rows.reserve(n);

        for i in 0..n {
            let row_idx = base_idx + i;
            let ncs_id = EntityId::from(ncs_id_col.value(i));
            let stage_id = stage_id_col.value(i);
            let available_generation_mw = avail_gen_col.value(i);

            if !available_generation_mw.is_finite() || available_generation_mw < 0.0 {
                return Err(LoadError::SchemaError {
                    path: path.to_path_buf(),
                    field: format!("ncs_bounds[{row_idx}].available_generation_mw"),
                    message: format!(
                        "value must be finite and >= 0.0, got {available_generation_mw}"
                    ),
                });
            }

            rows.push(NcsBoundsRow {
                ncs_id,
                stage_id,
                available_generation_mw,
            });
        }
    }

    rows.sort_by(|a, b| {
        a.ncs_id
            .0
            .cmp(&b.ncs_id.0)
            .then_with(|| a.stage_id.cmp(&b.stage_id))
    });

    Ok(rows)
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::float_cmp, clippy::panic)]
mod tests {
    use super::*;
    use arrow::array::{Float64Array, Int32Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;
    use std::sync::Arc;
    use tempfile::NamedTempFile;

    /// Write a Parquet file with the given NCS bounds data.
    fn write_ncs_bounds_parquet(
        ncs_ids: &[i32],
        stage_ids: &[i32],
        avail_gen: &[f64],
    ) -> NamedTempFile {
        let schema = Arc::new(Schema::new(vec![
            Field::new("ncs_id", DataType::Int32, false),
            Field::new("stage_id", DataType::Int32, false),
            Field::new("available_generation_mw", DataType::Float64, false),
        ]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(ncs_ids.to_vec())),
                Arc::new(Int32Array::from(stage_ids.to_vec())),
                Arc::new(Float64Array::from(avail_gen.to_vec())),
            ],
        )
        .unwrap();

        let tmp = NamedTempFile::new().unwrap();
        let file = tmp.as_file().try_clone().unwrap();
        let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
        tmp
    }

    #[test]
    fn test_parse_valid_2_rows() {
        let tmp = write_ncs_bounds_parquet(&[0, 1], &[0, 0], &[100.0, 200.0]);
        let rows = parse_ncs_bounds(tmp.path()).unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].ncs_id, EntityId::from(0));
        assert_eq!(rows[0].stage_id, 0);
        assert!((rows[0].available_generation_mw - 100.0).abs() < f64::EPSILON);
        assert_eq!(rows[1].ncs_id, EntityId::from(1));
        assert!((rows[1].available_generation_mw - 200.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_parse_sorted_output() {
        // Out of order: ncs_id=1 before ncs_id=0.
        let tmp = write_ncs_bounds_parquet(&[1, 0], &[0, 0], &[200.0, 100.0]);
        let rows = parse_ncs_bounds(tmp.path()).unwrap();
        assert_eq!(rows[0].ncs_id, EntityId::from(0));
        assert_eq!(rows[1].ncs_id, EntityId::from(1));
    }

    #[test]
    fn test_parse_negative_value_rejected() {
        let tmp = write_ncs_bounds_parquet(&[0], &[0], &[-50.0]);
        let err = parse_ncs_bounds(tmp.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(field.contains("available_generation_mw"), "field: {field}");
                assert!(message.contains("-50"), "message: {message}");
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    #[test]
    fn test_parse_nan_rejected() {
        let tmp = write_ncs_bounds_parquet(&[0], &[0], &[f64::NAN]);
        let err = parse_ncs_bounds(tmp.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { field, .. } => {
                assert!(field.contains("available_generation_mw"));
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    #[test]
    fn test_parse_missing_column_rejected() {
        // Write a Parquet with only ncs_id and stage_id (missing available_generation_mw).
        let schema = Arc::new(Schema::new(vec![
            Field::new("ncs_id", DataType::Int32, false),
            Field::new("stage_id", DataType::Int32, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![0])),
                Arc::new(Int32Array::from(vec![0])),
            ],
        )
        .unwrap();
        let tmp = NamedTempFile::new().unwrap();
        let file = tmp.as_file().try_clone().unwrap();
        let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();

        let err = parse_ncs_bounds(tmp.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { .. } => {}
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    #[test]
    fn test_parse_empty_file() {
        let tmp = write_ncs_bounds_parquet(&[], &[], &[]);
        let rows = parse_ncs_bounds(tmp.path()).unwrap();
        assert!(rows.is_empty());
    }

    #[test]
    fn test_parse_zero_available_gen_accepted() {
        let tmp = write_ncs_bounds_parquet(&[0], &[0], &[0.0]);
        let rows = parse_ncs_bounds(tmp.path()).unwrap();
        assert_eq!(rows.len(), 1);
        assert!((rows[0].available_generation_mw).abs() < f64::EPSILON);
    }
}
