//! Parquet writer for the resolved generic-constraint echo.
//!
//! [`write_generic_constraint_echo`] exports a slice of
//! [`GenericConstraintEchoRow`] — each generic constraint's fully-resolved flat
//! form, one row per `(constraint, stage, block, term)`. The on-disk shape is
//! owned by `generic_constraint_echo_schema` in [`crate::output::schemas`].
//!
//! All writes are atomic: data is first written to a `.tmp` suffix, then renamed
//! to the final path.

use std::path::Path;
use std::sync::Arc;

use arrow::array::{BooleanBuilder, Float64Builder, Int32Builder, RecordBatch, StringBuilder};

use crate::output::atomic::write_batch_atomic;
use crate::output::error::OutputError;
use crate::output::schemas::generic_constraint_echo_schema;

/// The resolved generic-constraint echo file's path relative to the output directory.
pub const GENERIC_CONSTRAINT_ECHO_FILE: &str = "generic_constraints/resolved_echo.parquet";

/// One row of the resolved generic-constraint echo.
///
/// Each row is one resolved LHS term of one generic constraint at one
/// `(stage, block)`. Following the sense-free interval model, the constraint's
/// bound is carried as the nullable pair `bound_lower`/`bound_upper` (min before
/// max) plus a derived shape label — there is no authored sense. A term-less
/// (constant-only) constraint contributes a single placeholder row whose per-term
/// columns are all `None`.
#[derive(Debug, Clone)]
pub struct GenericConstraintEchoRow {
    /// Study stage id.
    pub stage_id: i32,
    /// Block id; `None` on a collapsed stage-level row.
    pub block_id: Option<i32>,
    /// Generic constraint id.
    pub constraint_id: i32,
    /// Generic constraint name.
    pub constraint_name: String,
    /// Term position in the resolved LHS.
    pub term_index: Option<i32>,
    /// `VariableRef` discriminant (e.g. `"thermal_generation"`).
    pub variable_kind: Option<String>,
    /// Rendered variable label.
    pub variable: Option<String>,
    /// Resolved numeric coefficient.
    pub coefficient: Option<f64>,
    /// Lower interval endpoint; `None` when unbounded below.
    pub bound_lower: Option<f64>,
    /// Upper interval endpoint; `None` when unbounded above.
    pub bound_upper: Option<f64>,
    /// Shape label derived from which endpoints are finite.
    pub derived_shape: String,
    /// Whether the constraint carries a slack term.
    pub slack_enabled: bool,
    /// Slack penalty; `None` when slack is disabled.
    pub slack_penalty: Option<f64>,
}

/// Write a slice of [`GenericConstraintEchoRow`] to a Parquet file at `path`.
///
/// Rows are written in the order given; the caller emits them in canonical
/// `(constraint, stage, block)` order. An empty slice produces a valid 0-row
/// file. The parent directory is created if absent; the write is atomic.
///
/// # Errors
///
/// - [`OutputError::IoError`] — directory creation, file open, or rename fails.
/// - [`OutputError::SerializationError`] — Arrow/Parquet construction fails.
///
/// # Examples
///
/// ```no_run
/// use cobre_io::{GenericConstraintEchoRow, write_generic_constraint_echo};
/// use std::path::Path;
///
/// # fn main() -> Result<(), cobre_io::OutputError> {
/// let rows = vec![GenericConstraintEchoRow {
///     stage_id: 1,
///     block_id: Some(0),
///     constraint_id: 7,
///     constraint_name: "reservoir_link".to_string(),
///     term_index: Some(0),
///     variable_kind: Some("thermal_generation".to_string()),
///     variable: Some("thermal[3]".to_string()),
///     coefficient: Some(1.5),
///     bound_lower: Some(10.0),
///     bound_upper: Some(50.0),
///     derived_shape: "band".to_string(),
///     slack_enabled: true,
///     slack_penalty: Some(1000.0),
/// }];
/// write_generic_constraint_echo(
///     Path::new("/tmp/out/generic_constraints_echo.parquet"),
///     &rows,
/// )?;
/// # Ok(())
/// # }
/// ```
pub fn write_generic_constraint_echo(
    path: &Path,
    rows: &[GenericConstraintEchoRow],
) -> Result<(), OutputError> {
    let batch = build_generic_constraint_echo_batch(rows)?;
    write_batch_atomic(path, &batch)
}

fn build_generic_constraint_echo_batch(
    rows: &[GenericConstraintEchoRow],
) -> Result<RecordBatch, OutputError> {
    let n = rows.len();

    let mut stage_id_col = Int32Builder::with_capacity(n);
    let mut block_id_col = Int32Builder::with_capacity(n);
    let mut constraint_id_col = Int32Builder::with_capacity(n);
    let mut constraint_name_col = StringBuilder::with_capacity(n, n * 16);
    let mut term_index_col = Int32Builder::with_capacity(n);
    let mut variable_kind_col = StringBuilder::with_capacity(n, n * 16);
    let mut variable_col = StringBuilder::with_capacity(n, n * 16);
    let mut coefficient_col = Float64Builder::with_capacity(n);
    let mut bound_lower_col = Float64Builder::with_capacity(n);
    let mut bound_upper_col = Float64Builder::with_capacity(n);
    let mut derived_shape_col = StringBuilder::with_capacity(n, n * 8);
    let mut slack_enabled_col = BooleanBuilder::with_capacity(n);
    let mut slack_penalty_col = Float64Builder::with_capacity(n);

    for row in rows {
        stage_id_col.append_value(row.stage_id);
        block_id_col.append_option(row.block_id);
        constraint_id_col.append_value(row.constraint_id);
        constraint_name_col.append_value(&row.constraint_name);
        term_index_col.append_option(row.term_index);
        variable_kind_col.append_option(row.variable_kind.as_deref());
        variable_col.append_option(row.variable.as_deref());
        coefficient_col.append_option(row.coefficient);
        bound_lower_col.append_option(row.bound_lower);
        bound_upper_col.append_option(row.bound_upper);
        derived_shape_col.append_value(&row.derived_shape);
        slack_enabled_col.append_value(row.slack_enabled);
        slack_penalty_col.append_option(row.slack_penalty);
    }

    RecordBatch::try_new(
        Arc::new(generic_constraint_echo_schema()),
        vec![
            Arc::new(stage_id_col.finish()),
            Arc::new(block_id_col.finish()),
            Arc::new(constraint_id_col.finish()),
            Arc::new(constraint_name_col.finish()),
            Arc::new(term_index_col.finish()),
            Arc::new(variable_kind_col.finish()),
            Arc::new(variable_col.finish()),
            Arc::new(coefficient_col.finish()),
            Arc::new(bound_lower_col.finish()),
            Arc::new(bound_upper_col.finish()),
            Arc::new(derived_shape_col.finish()),
            Arc::new(slack_enabled_col.finish()),
            Arc::new(slack_penalty_col.finish()),
        ],
    )
    .map_err(|e| OutputError::serialization("generic_constraint_echo", e.to_string()))
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::float_cmp, clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::test_support::output::read_first_batch;
    use arrow::array::{Array, BooleanArray, Float64Array, Int32Array, StringArray};
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    use tempfile::tempdir;

    fn sample_rows() -> Vec<GenericConstraintEchoRow> {
        vec![
            // Per-block, fully-resolved term, both bounds → "band"; slack on.
            GenericConstraintEchoRow {
                stage_id: 1,
                block_id: Some(0),
                constraint_id: 7,
                constraint_name: "reservoir_link".to_string(),
                term_index: Some(0),
                variable_kind: Some("thermal_generation".to_string()),
                variable: Some("thermal[3]".to_string()),
                coefficient: Some(1.5),
                bound_lower: Some(10.0),
                bound_upper: Some(50.0),
                derived_shape: "band".to_string(),
                slack_enabled: true,
                slack_penalty: Some(1000.0),
            },
            // Stage-level (block_id None), upper-only → "cap"; slack off.
            GenericConstraintEchoRow {
                stage_id: 1,
                block_id: None,
                constraint_id: 8,
                constraint_name: "import_cap".to_string(),
                term_index: Some(1),
                variable_kind: Some("exchange".to_string()),
                variable: Some("line[2]".to_string()),
                coefficient: Some(-1.0),
                bound_lower: None,
                bound_upper: Some(200.0),
                derived_shape: "cap".to_string(),
                slack_enabled: false,
                slack_penalty: None,
            },
            // Term-less placeholder (all per-term columns None), lower-only → "floor".
            GenericConstraintEchoRow {
                stage_id: 2,
                block_id: Some(1),
                constraint_id: 9,
                constraint_name: "const_only".to_string(),
                term_index: None,
                variable_kind: None,
                variable: None,
                coefficient: None,
                bound_lower: Some(5.0),
                bound_upper: None,
                derived_shape: "floor".to_string(),
                slack_enabled: false,
                slack_penalty: None,
            },
        ]
    }

    #[test]
    fn generic_constraint_echo_round_trip_band_cap_floor_rows() {
        let rows = sample_rows();
        let tmp = tempdir().expect("tempdir");
        let path = tmp.path().join("generic_constraints_echo.parquet");

        write_generic_constraint_echo(&path, &rows).expect("write must succeed");
        assert!(path.exists(), "file must exist after write");

        let batch = read_first_batch(&path);
        assert_eq!(batch.num_columns(), 13, "must have 13 columns");
        assert_eq!(batch.num_rows(), 3, "must have 3 rows");

        let stage_id = batch
            .column_by_name("stage_id")
            .unwrap()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        let block_id = batch
            .column_by_name("block_id")
            .unwrap()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        let constraint_id = batch
            .column_by_name("constraint_id")
            .unwrap()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        let constraint_name = batch
            .column_by_name("constraint_name")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let term_index = batch
            .column_by_name("term_index")
            .unwrap()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        let variable_kind = batch
            .column_by_name("variable_kind")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let variable = batch
            .column_by_name("variable")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let coefficient = batch
            .column_by_name("coefficient")
            .unwrap()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        let bound_lower = batch
            .column_by_name("bound_lower")
            .unwrap()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        let bound_upper = batch
            .column_by_name("bound_upper")
            .unwrap()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        let derived_shape = batch
            .column_by_name("derived_shape")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let slack_enabled = batch
            .column_by_name("slack_enabled")
            .unwrap()
            .as_any()
            .downcast_ref::<BooleanArray>()
            .unwrap();
        let slack_penalty = batch
            .column_by_name("slack_penalty")
            .unwrap()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();

        // Row 0 — band, full term, slack on.
        assert_eq!(stage_id.value(0), 1);
        assert!(!block_id.is_null(0) && block_id.value(0) == 0);
        assert_eq!(constraint_id.value(0), 7);
        assert_eq!(constraint_name.value(0), "reservoir_link");
        assert!(!term_index.is_null(0) && term_index.value(0) == 0);
        assert_eq!(variable_kind.value(0), "thermal_generation");
        assert_eq!(variable.value(0), "thermal[3]");
        assert!(coefficient.value(0) == 1.5);
        assert!(bound_lower.value(0) == 10.0);
        assert!(bound_upper.value(0) == 50.0);
        assert_eq!(derived_shape.value(0), "band");
        assert!(slack_enabled.value(0));
        assert!(!slack_penalty.is_null(0) && slack_penalty.value(0) == 1000.0);

        // Row 1 — cap, stage-level (block_id NULL), bound_lower NULL, slack off.
        assert_eq!(stage_id.value(1), 1);
        assert!(block_id.is_null(1), "stage-level row block_id must be NULL");
        assert_eq!(constraint_id.value(1), 8);
        assert_eq!(constraint_name.value(1), "import_cap");
        assert!(!term_index.is_null(1) && term_index.value(1) == 1);
        assert_eq!(variable_kind.value(1), "exchange");
        assert_eq!(variable.value(1), "line[2]");
        assert!(coefficient.value(1) == -1.0);
        assert!(bound_lower.is_null(1), "cap row bound_lower must be NULL");
        assert!(bound_upper.value(1) == 200.0);
        assert_eq!(derived_shape.value(1), "cap");
        assert!(!slack_enabled.value(1));
        assert!(
            slack_penalty.is_null(1),
            "slack-off row penalty must be NULL"
        );

        // Row 2 — floor, term-less placeholder (per-term columns NULL), bound_upper NULL.
        assert_eq!(stage_id.value(2), 2);
        assert!(!block_id.is_null(2) && block_id.value(2) == 1);
        assert_eq!(constraint_id.value(2), 9);
        assert_eq!(constraint_name.value(2), "const_only");
        assert!(
            term_index.is_null(2),
            "term-less row term_index must be NULL"
        );
        assert!(variable_kind.is_null(2), "term-less row variable_kind NULL");
        assert!(variable.is_null(2), "term-less row variable NULL");
        assert!(coefficient.is_null(2), "term-less row coefficient NULL");
        assert!(bound_lower.value(2) == 5.0);
        assert!(bound_upper.is_null(2), "floor row bound_upper must be NULL");
        assert_eq!(derived_shape.value(2), "floor");
        assert!(!slack_enabled.value(2));
        assert!(slack_penalty.is_null(2));
    }

    #[test]
    fn generic_constraint_echo_empty_slice_produces_valid_zero_row_file() {
        let tmp = tempdir().expect("tempdir");
        let path = tmp.path().join("generic_constraints_echo.parquet");

        write_generic_constraint_echo(&path, &[]).expect("write must succeed for empty slice");
        assert!(path.exists(), "file must exist after write");

        let file = std::fs::File::open(&path).unwrap();
        let builder = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
        assert_eq!(
            builder.schema().fields().len(),
            13,
            "empty file must still carry the full 13-column schema"
        );
        let total_rows: usize = builder
            .build()
            .unwrap()
            .flatten()
            .map(|b| b.num_rows())
            .sum();
        assert_eq!(total_rows, 0, "empty slice must produce 0 rows");
    }
}
