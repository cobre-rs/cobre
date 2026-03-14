//! Parsing for `scenarios/noise_openings.parquet` — user-supplied noise
//! realisations for the opening scenario tree.
//!
//! [`parse_noise_openings`] reads `scenarios/noise_openings.parquet` and returns a
//! sorted `Vec<NoiseOpeningRow>`. [`validate_noise_openings`] checks dimension and
//! stage coverage. [`assemble_opening_tree`] converts validated rows into an
//! [`OpeningTree`].
//!
//! ## Parquet schema (ADR-008)
//!
//! | Column           | Type    | Required | Description                                  |
//! | ---------------- | ------- | -------- | -------------------------------------------- |
//! | `stage_id`       | INT32   | Yes      | Stage index (0-based)                        |
//! | `opening_index`  | UINT32  | Yes      | Opening index within the stage (0-based)     |
//! | `entity_index`   | UINT32  | Yes      | Entity index within the noise vector (0-based)|
//! | `value`          | DOUBLE  | Yes      | Noise realisation value                      |
//!
//! ## Output ordering
//!
//! Rows are sorted by `(stage_id, opening_index, entity_index)` ascending, which
//! matches the stage-major, row-major layout required by [`OpeningTree::from_parts`].
//!
//! ## Validation
//!
//! Cross-dimensional constraints enforced by [`validate_noise_openings`]:
//!
//! - Distinct `entity_index` count must equal `expected_dim`.
//! - Distinct `stage_id` count must equal `expected_stages`.
//! - For every `(stage, entity)`, the set of `opening_index` values must be exactly
//!   `0..expected_openings_per_stage[stage]` (no gaps, no extras).

use std::path::PathBuf;

use cobre_stochastic::OpeningTree;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::path::Path;

use crate::parquet_helpers::{
    extract_required_float64, extract_required_int32, extract_required_uint32,
};
use crate::LoadError;

/// A single row from `scenarios/noise_openings.parquet`.
///
/// Each row carries a single noise realisation for one `(stage, opening, entity)`
/// triple. Rows are sorted by `(stage_id, opening_index, entity_index)` after
/// parsing to match the flat layout expected by [`OpeningTree::from_parts`].
///
/// # Examples
///
/// ```
/// use cobre_io::scenarios::NoiseOpeningRow;
///
/// let row = NoiseOpeningRow {
///     stage_id: 0,
///     opening_index: 1,
///     entity_index: 2,
///     value: -0.5,
/// };
/// assert_eq!(row.stage_id, 0);
/// assert_eq!(row.opening_index, 1);
/// assert_eq!(row.entity_index, 2);
/// assert!((row.value - (-0.5)).abs() < 1e-15);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct NoiseOpeningRow {
    /// Stage index (0-based within `System::stages`).
    pub stage_id: i32,
    /// Opening index within the stage (0-based).
    pub opening_index: u32,
    /// Entity index within the noise vector (0-based).
    pub entity_index: u32,
    /// Noise realisation value.
    pub value: f64,
}

/// Parse `scenarios/noise_openings.parquet` and return a sorted row table.
///
/// Reads all record batches from the Parquet file at `path` and returns all rows
/// sorted by `(stage_id, opening_index, entity_index)` ascending. No per-row
/// value validation is applied here; cross-dimensional validation is deferred to
/// [`validate_noise_openings`].
///
/// # Errors
///
/// | Condition                                     | Error variant              |
/// |---------------------------------------------- |--------------------------- |
/// | File not found or permission denied           | [`LoadError::IoError`]     |
/// | Malformed Parquet (corrupt header, etc.)      | [`LoadError::ParseError`]  |
/// | Required column missing or wrong type         | [`LoadError::SchemaError`] |
///
/// # Examples
///
/// ```no_run
/// use cobre_io::scenarios::parse_noise_openings;
/// use std::path::Path;
///
/// let rows = parse_noise_openings(Path::new("scenarios/noise_openings.parquet"))
///     .expect("valid noise openings file");
/// println!("loaded {} noise opening rows", rows.len());
/// ```
pub fn parse_noise_openings(path: &Path) -> Result<Vec<NoiseOpeningRow>, LoadError> {
    let file = File::open(path).map_err(|e| LoadError::io(path, e))?;

    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| LoadError::parse(path, e.to_string()))?;

    let reader = builder
        .build()
        .map_err(|e| LoadError::parse(path, e.to_string()))?;

    let mut rows: Vec<NoiseOpeningRow> = Vec::new();

    for batch_result in reader {
        let batch = batch_result.map_err(|e| LoadError::parse(path, e.to_string()))?;

        let stage_id_col = extract_required_int32(&batch, "stage_id", path)?;
        let opening_index_col = extract_required_uint32(&batch, "opening_index", path)?;
        let entity_index_col = extract_required_uint32(&batch, "entity_index", path)?;
        let value_col = extract_required_float64(&batch, "value", path)?;

        let n = batch.num_rows();
        rows.reserve(n);

        for i in 0..n {
            rows.push(NoiseOpeningRow {
                stage_id: stage_id_col.value(i),
                opening_index: opening_index_col.value(i),
                entity_index: entity_index_col.value(i),
                value: value_col.value(i),
            });
        }
    }

    rows.sort_by(|a, b| {
        a.stage_id
            .cmp(&b.stage_id)
            .then_with(|| a.opening_index.cmp(&b.opening_index))
            .then_with(|| a.entity_index.cmp(&b.entity_index))
    });

    Ok(rows)
}

/// Validate parsed noise opening rows against expected system dimensions.
///
/// Checks that:
/// - The number of distinct `entity_index` values equals `expected_dim`.
/// - The number of distinct `stage_id` values equals `expected_stages`.
/// - For each stage, the set of `opening_index` values across all entities is
///   exactly `0..expected_openings_per_stage[stage]` (no gaps, no extras checked
///   per entity).
///
/// This function assumes `rows` is already sorted by `(stage_id, opening_index,
/// entity_index)` as produced by [`parse_noise_openings`].
///
/// # Errors
///
/// | Condition                                                      | Error variant              |
/// |----------------------------------------------------------------|----------------------------|
/// | Distinct entity count != `expected_dim`                       | [`LoadError::SchemaError`] |
/// | Distinct stage count != `expected_stages`                     | [`LoadError::SchemaError`] |
/// | Opening indices for any stage are not `0..openings_per_stage` | [`LoadError::SchemaError`] |
///
/// Returns [`LoadError::SchemaError`] if any `expected_count` exceeds `u32::MAX`.
///
/// # Examples
///
/// ```
/// use cobre_io::scenarios::{NoiseOpeningRow, validate_noise_openings};
///
/// // 2 stages, 3 openings each, dim=2 → 12 rows
/// let rows: Vec<NoiseOpeningRow> = (0..2_i32)
///     .flat_map(|s| (0..3_u32).flat_map(move |o| (0..2_u32).map(move |e| NoiseOpeningRow {
///         stage_id: s, opening_index: o, entity_index: e, value: 0.0,
///     })))
///     .collect();
///
/// validate_noise_openings(&rows, 2, 2, &[3, 3]).expect("valid dimensions");
/// ```
pub fn validate_noise_openings(
    rows: &[NoiseOpeningRow],
    expected_dim: usize,
    expected_stages: usize,
    expected_openings_per_stage: &[usize],
) -> Result<(), LoadError> {
    // Collect distinct entity_index values across all rows.
    let distinct_entities: BTreeSet<u32> = rows.iter().map(|r| r.entity_index).collect();
    let actual_dim = distinct_entities.len();
    if actual_dim != expected_dim {
        return Err(LoadError::SchemaError {
            path: std::path::PathBuf::from("scenarios/noise_openings.parquet"),
            field: "entity_index".to_string(),
            message: format!(
                "dimension mismatch: expected {expected_dim} entities, found {actual_dim}"
            ),
        });
    }

    // Collect distinct stage_id values.
    let distinct_stages: BTreeSet<i32> = rows.iter().map(|r| r.stage_id).collect();
    let actual_stages = distinct_stages.len();
    if actual_stages != expected_stages {
        return Err(LoadError::SchemaError {
            path: std::path::PathBuf::from("scenarios/noise_openings.parquet"),
            field: "stage_id".to_string(),
            message: format!(
                "stage count mismatch: expected {expected_stages} stages, found {actual_stages}"
            ),
        });
    }

    let mut openings_by_stage: BTreeMap<i32, BTreeSet<u32>> = BTreeMap::new();
    for row in rows {
        openings_by_stage
            .entry(row.stage_id)
            .or_default()
            .insert(row.opening_index);
    }

    for (stage_pos, (&stage_id, opening_set)) in openings_by_stage.iter().enumerate() {
        let expected_count = expected_openings_per_stage[stage_pos];
        let expected_max = u32::try_from(expected_count).map_err(|_| LoadError::SchemaError {
            path: PathBuf::from("noise_openings.parquet"),
            field: String::new(),
            message: format!("opening count {expected_count} exceeds u32::MAX"),
        })?;
        let expected_set: BTreeSet<u32> = (0..expected_max).collect();
        if *opening_set != expected_set {
            return Err(LoadError::SchemaError {
                path: std::path::PathBuf::from("scenarios/noise_openings.parquet"),
                field: "opening_index".to_string(),
                message: format!("missing opening indices for stage {stage_id}"),
            });
        }
    }

    Ok(())
}

/// Assemble an [`OpeningTree`] from validated, sorted noise opening rows.
///
/// `rows` must be sorted by `(stage_id, opening_index, entity_index)` ascending —
/// the layout produced by [`parse_noise_openings`] — and must have already passed
/// [`validate_noise_openings`]. The sort order matches the stage-major, row-major
/// memory layout required by [`OpeningTree::from_parts`].
///
/// `dim` is the number of entities per opening vector (the noise dimension).
///
/// # Panics
///
/// Panics if `rows.len()` is not consistent with the implied
/// `sum(openings_per_stage) * dim` (delegated to [`OpeningTree::from_parts`]).
///
/// # Examples
///
/// ```
/// use cobre_io::scenarios::{NoiseOpeningRow, assemble_opening_tree};
///
/// // 2 stages, 3 openings each, dim=2 → 12 rows
/// let rows: Vec<NoiseOpeningRow> = (0..2_i32)
///     .flat_map(|s| (0..3_u32).flat_map(move |o| (0..2_u32).map(move |e| NoiseOpeningRow {
///         stage_id: s, opening_index: o, entity_index: e, value: f64::from(s * 6 + o as i32 * 2 + e as i32),
///     })))
///     .collect();
///
/// let tree = assemble_opening_tree(rows, 2);
/// assert_eq!(tree.n_stages(), 2);
/// assert_eq!(tree.n_openings(0), 3);
/// assert_eq!(tree.dim(), 2);
/// ```
#[must_use]
pub fn assemble_opening_tree(rows: Vec<NoiseOpeningRow>, dim: usize) -> OpeningTree {
    let mut openings_per_stage: Vec<usize> = Vec::new();
    let mut current_stage: Option<i32> = None;
    let mut current_opening_count: usize = 0;
    let mut last_opening: Option<u32> = None;

    for row in &rows {
        if current_stage != Some(row.stage_id) {
            if current_stage.is_some() {
                openings_per_stage.push(current_opening_count);
            }
            current_stage = Some(row.stage_id);
            current_opening_count = 1;
            last_opening = Some(row.opening_index);
        } else if Some(row.opening_index) != last_opening {
            current_opening_count += 1;
            last_opening = Some(row.opening_index);
        }
    }
    if current_stage.is_some() {
        openings_per_stage.push(current_opening_count);
    }

    let data: Vec<f64> = rows.into_iter().map(|r| r.value).collect();
    OpeningTree::from_parts(data, openings_per_stage, dim)
}

// ── Tests ──────────────────────────────────────────────────────────────────────

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
    use arrow::array::{Float64Array, Int32Array, UInt32Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;
    use std::sync::Arc;
    use tempfile::NamedTempFile;

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("stage_id", DataType::Int32, false),
            Field::new("opening_index", DataType::UInt32, false),
            Field::new("entity_index", DataType::UInt32, false),
            Field::new("value", DataType::Float64, false),
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
        stage_ids: &[i32],
        opening_indices: &[u32],
        entity_indices: &[u32],
        values: &[f64],
    ) -> RecordBatch {
        RecordBatch::try_new(
            schema(),
            vec![
                Arc::new(Int32Array::from(stage_ids.to_vec())),
                Arc::new(UInt32Array::from(opening_indices.to_vec())),
                Arc::new(UInt32Array::from(entity_indices.to_vec())),
                Arc::new(Float64Array::from(values.to_vec())),
            ],
        )
        .expect("valid batch")
    }

    /// Build a complete, sorted row set for `n_stages` stages each with
    /// `openings` openings and `dim` entities. Values are sequential floats.
    fn make_rows(n_stages: usize, openings: usize, dim: usize) -> Vec<NoiseOpeningRow> {
        let mut rows = Vec::new();
        let mut v = 0.0_f64;
        for s in 0..n_stages {
            for o in 0..openings {
                for e in 0..dim {
                    rows.push(NoiseOpeningRow {
                        stage_id: i32::try_from(s).unwrap(),
                        opening_index: u32::try_from(o).unwrap(),
                        entity_index: u32::try_from(e).unwrap(),
                        value: v,
                    });
                    v += 1.0;
                }
            }
        }
        rows
    }

    // ── parse_valid_file_returns_sorted_rows ──────────────────────────────────

    /// Valid file with 12 rows (2 stages × 3 openings × 2 entities) written in
    /// scrambled order. Parser must return exactly 12 rows sorted by
    /// (stage_id, opening_index, entity_index).
    #[test]
    fn parse_valid_file_returns_sorted_rows() {
        // Write rows in reversed order: stage 1 before stage 0, etc.
        let batch = make_batch(
            &[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            &[2, 2, 1, 1, 0, 0, 2, 2, 1, 1, 0, 0],
            &[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            &[11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0],
        );
        let tmp = write_parquet(&batch);
        let rows = parse_noise_openings(tmp.path()).unwrap();

        assert_eq!(rows.len(), 12, "expected 12 rows");

        // Verify sort order: (stage_id, opening_index, entity_index) ascending.
        for w in rows.windows(2) {
            let a = &w[0];
            let b = &w[1];
            let cmp = a
                .stage_id
                .cmp(&b.stage_id)
                .then_with(|| a.opening_index.cmp(&b.opening_index))
                .then_with(|| a.entity_index.cmp(&b.entity_index));
            assert!(
                cmp != std::cmp::Ordering::Greater,
                "rows not sorted: {a:?} > {b:?}"
            );
        }

        // First row must be (stage=0, opening=0, entity=0).
        assert_eq!(rows[0].stage_id, 0);
        assert_eq!(rows[0].opening_index, 0);
        assert_eq!(rows[0].entity_index, 0);
        assert!((rows[0].value - 0.0).abs() < 1e-15);
    }

    // ── parse_missing_column_returns_schema_error ─────────────────────────────

    /// File missing the `value` column must return `SchemaError` with field
    /// containing "value".
    #[test]
    fn parse_missing_column_returns_schema_error() {
        let schema_no_value = Arc::new(Schema::new(vec![
            Field::new("stage_id", DataType::Int32, false),
            Field::new("opening_index", DataType::UInt32, false),
            Field::new("entity_index", DataType::UInt32, false),
        ]));
        let batch = RecordBatch::try_new(
            schema_no_value,
            vec![
                Arc::new(Int32Array::from(vec![0_i32])),
                Arc::new(UInt32Array::from(vec![0_u32])),
                Arc::new(UInt32Array::from(vec![0_u32])),
            ],
        )
        .unwrap();
        let tmp = write_parquet(&batch);
        let err = parse_noise_openings(tmp.path()).unwrap_err();

        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("value"),
                    "field should contain 'value', got: {field}"
                );
                assert!(
                    message.contains("missing required column"),
                    "message should mention missing column, got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    // ── validate_correct_dimensions_returns_ok ────────────────────────────────

    /// 2 stages × 3 openings × dim=2 passes validation with matching expectations.
    #[test]
    fn validate_correct_dimensions_returns_ok() {
        let rows = make_rows(2, 3, 2);
        validate_noise_openings(&rows, 2, 2, &[3, 3]).unwrap();
    }

    // ── validate_dimension_mismatch_returns_error ─────────────────────────────

    /// Rows with 3 distinct entity_index values but expected_dim=2 must return
    /// SchemaError containing "dimension mismatch".
    #[test]
    fn validate_dimension_mismatch_returns_error() {
        // Build rows with dim=3 but validate against expected_dim=2.
        let rows = make_rows(2, 3, 3);
        let err = validate_noise_openings(&rows, 2, 2, &[3, 3]).unwrap_err();

        match &err {
            LoadError::SchemaError { message, .. } => {
                assert!(
                    message.contains("dimension mismatch"),
                    "message should contain 'dimension mismatch', got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    // ── validate_stage_count_mismatch_returns_error ───────────────────────────

    /// Rows with 2 distinct stage_id values but expected_stages=3 must return
    /// SchemaError containing "stage count mismatch".
    #[test]
    fn validate_stage_count_mismatch_returns_error() {
        let rows = make_rows(2, 3, 2);
        let err = validate_noise_openings(&rows, 2, 3, &[3, 3, 3]).unwrap_err();

        match &err {
            LoadError::SchemaError { message, .. } => {
                assert!(
                    message.contains("stage count mismatch"),
                    "message should contain 'stage count mismatch', got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    // ── validate_missing_openings_returns_error ───────────────────────────────

    /// Stage 0 has opening indices {0, 2} (missing index 1). Validation with
    /// expected_openings_per_stage=[3] must return SchemaError containing
    /// "missing opening indices".
    #[test]
    fn validate_missing_openings_returns_error() {
        // 1 stage, openings 0 and 2 only (index 1 missing), dim=2 → 4 rows.
        let rows: Vec<NoiseOpeningRow> = [0u32, 2u32]
            .iter()
            .flat_map(|&o| {
                [0u32, 1u32].iter().map(move |&e| NoiseOpeningRow {
                    stage_id: 0,
                    opening_index: o,
                    entity_index: e,
                    value: 0.0,
                })
            })
            .collect();

        let err = validate_noise_openings(&rows, 2, 1, &[3]).unwrap_err();

        match &err {
            LoadError::SchemaError { message, .. } => {
                assert!(
                    message.contains("missing opening indices"),
                    "message should contain 'missing opening indices', got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    // ── assemble_produces_correct_opening_tree ────────────────────────────────

    /// Sorted rows for 2 stages × 3 openings × dim=2 assemble into an
    /// OpeningTree whose opening() slices match the input values.
    #[test]
    fn assemble_produces_correct_opening_tree() {
        let rows = make_rows(2, 3, 2);
        // Capture expected values before moving rows.
        let expected: Vec<f64> = rows.iter().map(|r| r.value).collect();

        let tree = assemble_opening_tree(rows, 2);

        assert_eq!(tree.n_stages(), 2);
        assert_eq!(tree.n_openings(0), 3);
        assert_eq!(tree.n_openings(1), 3);
        assert_eq!(tree.dim(), 2);

        // Verify that opening(s, o) slices contain the expected values.
        // make_rows produces values in sequential order stage-major.
        assert_eq!(tree.data(), expected.as_slice());

        // Spot-check individual opening slices.
        // Stage 0, opening 0: values 0.0, 1.0
        assert_eq!(tree.opening(0, 0), &[0.0_f64, 1.0]);
        // Stage 0, opening 2: values 4.0, 5.0
        assert_eq!(tree.opening(0, 2), &[4.0_f64, 5.0]);
        // Stage 1, opening 0: values 6.0, 7.0
        assert_eq!(tree.opening(1, 0), &[6.0_f64, 7.0]);
        // Stage 1, opening 2: values 10.0, 11.0
        assert_eq!(tree.opening(1, 2), &[10.0_f64, 11.0]);
    }
}
