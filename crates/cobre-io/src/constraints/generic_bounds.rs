//! Parsing for `constraints/generic_constraint_bounds.parquet` — stage/block-varying
//! RHS values for user-defined generic constraints.
//!
//! [`parse_generic_constraint_bounds`] reads
//! `constraints/generic_constraint_bounds.parquet` and returns a sorted
//! `Vec<GenericConstraintBoundsRow>`.
//!
//! ## Parquet schema
//!
//! | Column          | Type          | Required | Description                       |
//! | --------------- | ------------- | -------- | --------------------------------- |
//! | `constraint_id` | INT32         | Yes      | References constraint definition  |
//! | `stage_id`      | INT32         | Yes      | Stage index                       |
//! | `block_id`      | INT32 (null)  | No       | Block index (`null` = all blocks) |
//! | `bound`         | DOUBLE        | Yes      | RHS value                         |
//!
//! ## Output ordering
//!
//! Rows are sorted by `(constraint_id, stage_id, block_id)` ascending.
//! `None` block sorts before `Some(i)`.
//!
//! ## Validation
//!
//! Per-row constraints enforced by this parser:
//!
//! - All required columns must be present with the correct types.
//! - `bound` must be finite (NaN and ±Inf are rejected).
//!
//! Deferred validations (not performed here, Epic 06):
//!
//! - `constraint_id` existence in the generic constraint registry.
//! - `block_id` validity for the referenced stage.
//! - Duplicate `(constraint_id, stage_id, block_id)` key detection.

use arrow::array::Array;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::fs::File;
use std::path::Path;

use crate::parquet_helpers::{
    extract_optional_int32, extract_required_float64, extract_required_int32,
};
use crate::LoadError;

/// A single row from `constraints/generic_constraint_bounds.parquet`.
///
/// Carries a stage/block-varying RHS value for a generic constraint. When
/// `block_id` is `None`, the bound applies to all blocks of the given stage.
///
/// # Examples
///
/// ```
/// use cobre_io::constraints::GenericConstraintBoundsRow;
///
/// let row = GenericConstraintBoundsRow {
///     constraint_id: 0,
///     stage_id: 5,
///     block_id: None,
///     bound: 1500.0,
/// };
/// assert_eq!(row.constraint_id, 0);
/// assert!(row.block_id.is_none());
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct GenericConstraintBoundsRow {
    /// Constraint ID.
    pub constraint_id: i32,
    /// Stage ID.
    pub stage_id: i32,
    /// Block ID (None = all blocks).
    pub block_id: Option<i32>,
    /// RHS bound value.
    pub bound: f64,
}

/// Parse `constraints/generic_constraint_bounds.parquet` and return a sorted row table.
///
/// Reads all record batches from the Parquet file at `path`, validates per-row
/// constraints, then returns all rows sorted by `(constraint_id, stage_id, block_id)`
/// ascending. `None` block sorts before `Some(i)`.
///
/// # Errors
///
/// | Condition                                 | Error variant              |
/// | ----------------------------------------- | -------------------------- |
/// | File not found or permission denied       | [`LoadError::IoError`]     |
/// | Malformed Parquet (corrupt header, etc.)  | [`LoadError::ParseError`]  |
/// | Required column missing or wrong type     | [`LoadError::SchemaError`] |
/// | `bound` is NaN or infinite                | [`LoadError::SchemaError`] |
///
/// # Examples
///
/// ```no_run
/// use cobre_io::constraints::parse_generic_constraint_bounds;
/// use std::path::Path;
///
/// let rows = parse_generic_constraint_bounds(
///     Path::new("case/constraints/generic_constraint_bounds.parquet")
/// ).expect("valid bounds file");
/// println!("loaded {} bound rows", rows.len());
/// ```
pub fn parse_generic_constraint_bounds(
    path: &Path,
) -> Result<Vec<GenericConstraintBoundsRow>, LoadError> {
    let file = File::open(path).map_err(|e| LoadError::io(path, e))?;

    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| LoadError::parse(path, e.to_string()))?;

    let reader = builder
        .build()
        .map_err(|e| LoadError::parse(path, e.to_string()))?;

    let mut rows: Vec<GenericConstraintBoundsRow> = Vec::new();

    for batch_result in reader {
        let batch = batch_result.map_err(|e| LoadError::parse(path, e.to_string()))?;

        // ── Required columns ──────────────────────────────────────────────────
        let constraint_id_col = extract_required_int32(&batch, "constraint_id", path)?;
        let stage_id_col = extract_required_int32(&batch, "stage_id", path)?;
        let bound_col = extract_required_float64(&batch, "bound", path)?;

        // ── Optional column ───────────────────────────────────────────────────
        let block_id_col = extract_optional_int32(&batch, "block_id", path)?;

        // ── Build rows with per-row validation ────────────────────────────────
        let n = batch.num_rows();
        let base_idx = rows.len();
        rows.reserve(n);

        for i in 0..n {
            let row_idx = base_idx + i;

            let constraint_id = constraint_id_col.value(i);
            let stage_id = stage_id_col.value(i);
            let block_id = block_id_col
                .filter(|col| !col.is_null(i))
                .map(|col| col.value(i));
            let bound = bound_col.value(i);

            // Validate bound: must be finite.
            if !bound.is_finite() {
                return Err(LoadError::SchemaError {
                    path: path.to_path_buf(),
                    field: format!("generic_constraint_bounds[{row_idx}].bound"),
                    message: format!("value must be finite, got {bound}"),
                });
            }

            rows.push(GenericConstraintBoundsRow {
                constraint_id,
                stage_id,
                block_id,
                bound,
            });
        }
    }

    // ── Sort by (constraint_id, stage_id, block_id) ascending ─────────────────
    // None sorts before Some (null block = applies to all blocks comes first).
    rows.sort_by(|a, b| {
        a.constraint_id
            .cmp(&b.constraint_id)
            .then_with(|| a.stage_id.cmp(&b.stage_id))
            .then_with(|| match (a.block_id, b.block_id) {
                (None, None) => std::cmp::Ordering::Equal,
                (None, Some(_)) => std::cmp::Ordering::Less,
                (Some(_), None) => std::cmp::Ordering::Greater,
                (Some(a), Some(b)) => a.cmp(&b),
            })
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

    fn schema_with_block() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("constraint_id", DataType::Int32, false),
            Field::new("stage_id", DataType::Int32, false),
            Field::new("block_id", DataType::Int32, true), // nullable
            Field::new("bound", DataType::Float64, false),
        ]))
    }

    fn schema_without_block() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("constraint_id", DataType::Int32, false),
            Field::new("stage_id", DataType::Int32, false),
            Field::new("bound", DataType::Float64, false),
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

    /// Build a batch with nullable block_id column.
    fn make_batch_with_block(
        constraint_ids: &[i32],
        stage_ids: &[i32],
        block_ids: Vec<Option<i32>>,
        bounds: &[f64],
    ) -> RecordBatch {
        let block_arr: Int32Array = block_ids.into_iter().collect();
        RecordBatch::try_new(
            schema_with_block(),
            vec![
                Arc::new(Int32Array::from(constraint_ids.to_vec())),
                Arc::new(Int32Array::from(stage_ids.to_vec())),
                Arc::new(block_arr),
                Arc::new(Float64Array::from(bounds.to_vec())),
            ],
        )
        .expect("valid batch")
    }

    /// Build a batch without block_id column.
    fn make_batch_without_block(
        constraint_ids: &[i32],
        stage_ids: &[i32],
        bounds: &[f64],
    ) -> RecordBatch {
        RecordBatch::try_new(
            schema_without_block(),
            vec![
                Arc::new(Int32Array::from(constraint_ids.to_vec())),
                Arc::new(Int32Array::from(stage_ids.to_vec())),
                Arc::new(Float64Array::from(bounds.to_vec())),
            ],
        )
        .expect("valid batch")
    }

    // ── Tests ─────────────────────────────────────────────────────────────────

    /// Valid rows with nullable block_id.
    #[test]
    fn test_parse_with_nullable_block_id() {
        let batch = make_batch_with_block(
            &[0, 0, 1],
            &[0, 0, 2],
            vec![Some(0), None, Some(1)],
            &[100.0, 200.0, 300.0],
        );
        let tmp = write_parquet(&batch);
        let rows = parse_generic_constraint_bounds(tmp.path()).unwrap();

        assert_eq!(rows.len(), 3);

        // After sorting by (constraint_id, stage_id, block_id):
        // (0, 0, None) < (0, 0, Some(0)) < (1, 2, Some(1))
        assert_eq!(rows[0].constraint_id, 0);
        assert_eq!(rows[0].stage_id, 0);
        assert_eq!(rows[0].block_id, None);
        assert!((rows[0].bound - 200.0).abs() < f64::EPSILON);

        assert_eq!(rows[1].constraint_id, 0);
        assert_eq!(rows[1].stage_id, 0);
        assert_eq!(rows[1].block_id, Some(0));
        assert!((rows[1].bound - 100.0).abs() < f64::EPSILON);

        assert_eq!(rows[2].constraint_id, 1);
        assert_eq!(rows[2].stage_id, 2);
        assert_eq!(rows[2].block_id, Some(1));
        assert!((rows[2].bound - 300.0).abs() < f64::EPSILON);
    }

    /// When block_id column is absent, all block_id values are None.
    #[test]
    fn test_parse_without_block_id_column() {
        let batch = make_batch_without_block(&[0, 1], &[3, 4], &[50.0, 75.0]);
        let tmp = write_parquet(&batch);
        let rows = parse_generic_constraint_bounds(tmp.path()).unwrap();

        assert_eq!(rows.len(), 2);
        assert!(rows[0].block_id.is_none());
        assert!(rows[1].block_id.is_none());
    }

    /// Non-finite bound → SchemaError.
    #[test]
    fn test_parse_non_finite_bound_returns_schema_error() {
        let batch = make_batch_without_block(&[0], &[0], &[f64::NAN]);
        let tmp = write_parquet(&batch);
        let err = parse_generic_constraint_bounds(tmp.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("bound"),
                    "field should contain 'bound', got: {field}"
                );
                assert!(
                    message.contains("finite"),
                    "message should mention 'finite', got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// Infinite bound → SchemaError.
    #[test]
    fn test_parse_infinite_bound_returns_schema_error() {
        let batch = make_batch_without_block(&[0], &[0], &[f64::INFINITY]);
        let tmp = write_parquet(&batch);
        let err = parse_generic_constraint_bounds(tmp.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { field, .. } => {
                assert!(
                    field.contains("bound"),
                    "field should contain 'bound', got: {field}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// Missing required column → SchemaError.
    #[test]
    fn test_parse_missing_bound_column_returns_schema_error() {
        // Build a batch without 'bound' column.
        let schema = Arc::new(Schema::new(vec![
            Field::new("constraint_id", DataType::Int32, false),
            Field::new("stage_id", DataType::Int32, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![0])),
                Arc::new(Int32Array::from(vec![0])),
            ],
        )
        .expect("valid batch");
        let tmp = write_parquet(&batch);
        let err = parse_generic_constraint_bounds(tmp.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("bound"),
                    "field should contain 'bound', got: {field}"
                );
                assert!(
                    message.contains("missing"),
                    "message should mention 'missing', got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// Empty Parquet file → Ok(Vec::new()).
    #[test]
    fn test_parse_empty_parquet() {
        let batch = make_batch_without_block(&[], &[], &[]);
        let tmp = write_parquet(&batch);
        let rows = parse_generic_constraint_bounds(tmp.path()).unwrap();
        assert!(rows.is_empty());
    }

    /// Declaration-order invariance: scrambled input → sorted output.
    #[test]
    fn test_parse_sort_order_invariance() {
        let batch = make_batch_with_block(
            &[2, 0, 1, 0],
            &[0, 1, 0, 0],
            vec![None, Some(0), None, None],
            &[10.0, 20.0, 30.0, 40.0],
        );
        let tmp = write_parquet(&batch);
        let rows = parse_generic_constraint_bounds(tmp.path()).unwrap();

        assert_eq!(rows.len(), 4);
        // Expected order: (0,0,None), (0,1,Some(0)), (1,0,None), (2,0,None)
        assert_eq!(
            (rows[0].constraint_id, rows[0].stage_id, rows[0].block_id),
            (0, 0, None)
        );
        assert_eq!(
            (rows[1].constraint_id, rows[1].stage_id, rows[1].block_id),
            (0, 1, Some(0))
        );
        assert_eq!(
            (rows[2].constraint_id, rows[2].stage_id, rows[2].block_id),
            (1, 0, None)
        );
        assert_eq!(
            (rows[3].constraint_id, rows[3].stage_id, rows[3].block_id),
            (2, 0, None)
        );
    }
}
