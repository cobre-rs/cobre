//! Parquet writer for stochastic artifact output files.
//!
//! [`write_noise_openings`] exports an [`OpeningTree`] to
//! `output/stochastic/noise_openings.parquet` using the 4-column schema
//! defined in ADR-008:
//!
//! | Column          | Type    | Description                                  |
//! |-----------------|---------|----------------------------------------------|
//! | `stage_id`      | INT32   | Stage index (0-based)                        |
//! | `opening_index` | UINT32  | Opening index within the stage (0-based)     |
//! | `entity_index`  | UINT32  | Entity index within the noise vector (0-based)|
//! | `value`         | DOUBLE  | Noise realisation value                      |
//!
//! Rows are written in `(stage_id, opening_index, entity_index)` order,
//! matching the stage-major storage layout of the tree. The file is written
//! atomically: data is first written to a `.tmp` suffix, then renamed to the
//! final path.

use std::path::Path;
use std::sync::Arc;

use arrow::array::{Float64Builder, Int32Builder, RecordBatch, UInt32Builder};
use arrow::datatypes::{DataType, Field, Schema};
use cobre_stochastic::OpeningTree;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;

use crate::output::error::OutputError;
use crate::output::parquet_config::ParquetWriterConfig;

/// Write an [`OpeningTree`] to a Parquet file at `path`.
///
/// The output schema is exactly 4 columns — `stage_id: Int32`,
/// `opening_index: UInt32`, `entity_index: UInt32`, `value: Float64` — in
/// that order, matching the ADR-008 schema and the input expected by
/// `parse_noise_openings` / `assemble_opening_tree`. Rows are written in
/// `(stage_id, opening_index, entity_index)` order.
///
/// The parent directory is created if it does not already exist. The write
/// is atomic: data goes to `{path}.tmp` first, then the file is renamed to
/// `path`. A partial `.tmp` file may remain on disk if the process is killed
/// mid-write, but the final path is never partially written.
///
/// # Errors
///
/// - [`OutputError::IoError`] — directory creation, file open, or rename fails.
/// - [`OutputError::SerializationError`] — Arrow/Parquet construction fails.
///
/// # Examples
///
/// ```no_run
/// use cobre_io::output::stochastic::write_noise_openings;
/// use cobre_stochastic::OpeningTree;
/// use std::path::Path;
///
/// # fn main() -> Result<(), cobre_io::OutputError> {
/// let tree = OpeningTree::from_parts(
///     vec![1.0, 2.0, 3.0, 4.0],
///     vec![1, 1],
///     2,
/// );
/// write_noise_openings(Path::new("/tmp/out/noise_openings.parquet"), &tree)?;
/// # Ok(())
/// # }
/// ```
pub fn write_noise_openings(path: &Path, tree: &OpeningTree) -> Result<(), OutputError> {
    // Create parent directory if absent.
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| OutputError::io(parent, e))?;
    }

    let config = ParquetWriterConfig::default();
    let batch = build_noise_openings_batch(tree)?;
    write_parquet_atomic(path, &batch, &config)
}

fn noise_openings_schema() -> Schema {
    Schema::new(vec![
        Field::new("stage_id", DataType::Int32, false),
        Field::new("opening_index", DataType::UInt32, false),
        Field::new("entity_index", DataType::UInt32, false),
        Field::new("value", DataType::Float64, false),
    ])
}

#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
fn build_noise_openings_batch(tree: &OpeningTree) -> Result<RecordBatch, OutputError> {
    let n_rows: usize = (0..tree.n_stages())
        .map(|s| tree.n_openings(s))
        .sum::<usize>()
        * tree.dim();

    let mut stage_id_col = Int32Builder::with_capacity(n_rows);
    let mut opening_index_col = UInt32Builder::with_capacity(n_rows);
    let mut entity_index_col = UInt32Builder::with_capacity(n_rows);
    let mut value_col = Float64Builder::with_capacity(n_rows);

    for stage in 0..tree.n_stages() {
        let stage_i32 = stage as i32;
        for opening_idx in 0..tree.n_openings(stage) {
            let opening_u32 = opening_idx as u32;
            let noise = tree.opening(stage, opening_idx);
            for (entity_idx, &v) in noise.iter().enumerate() {
                stage_id_col.append_value(stage_i32);
                opening_index_col.append_value(opening_u32);
                entity_index_col.append_value(entity_idx as u32);
                value_col.append_value(v);
            }
        }
    }

    RecordBatch::try_new(
        Arc::new(noise_openings_schema()),
        vec![
            Arc::new(stage_id_col.finish()),
            Arc::new(opening_index_col.finish()),
            Arc::new(entity_index_col.finish()),
            Arc::new(value_col.finish()),
        ],
    )
    .map_err(|e| OutputError::serialization("noise_openings", e.to_string()))
}

fn write_parquet_atomic(
    path: &Path,
    batch: &RecordBatch,
    config: &ParquetWriterConfig,
) -> Result<(), OutputError> {
    let tmp_path = path.with_extension(path.extension().map_or_else(
        || "tmp".to_string(),
        |ext| format!("{}.tmp", ext.to_string_lossy()),
    ));

    let props = WriterProperties::builder()
        .set_compression(config.compression)
        .set_max_row_group_row_count(Some(config.row_group_size))
        .set_dictionary_enabled(config.dictionary_encoding)
        .build();

    let file = std::fs::File::create(&tmp_path).map_err(|e| OutputError::io(&tmp_path, e))?;

    let mut writer = ArrowWriter::try_new(file, batch.schema(), Some(props))
        .map_err(|e| OutputError::serialization("parquet_writer", e.to_string()))?;

    writer
        .write(batch)
        .and_then(|()| writer.close())
        .map_err(|e| OutputError::serialization("parquet_writer", e.to_string()))?;

    std::fs::rename(&tmp_path, path).map_err(|e| OutputError::io(path, e))?;

    Ok(())
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::float_cmp,
    clippy::cast_possible_truncation
)]
mod tests {
    use super::*;
    use cobre_stochastic::OpeningTree;

    fn make_tree_2s_2d() -> OpeningTree {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        OpeningTree::from_parts(data, vec![2, 3], 2)
    }

    // -------------------------------------------------------------------------
    // write_then_read_round_trips
    // -------------------------------------------------------------------------

    #[test]
    fn write_then_read_round_trips() {
        use crate::scenarios::{assemble_opening_tree, parse_noise_openings, NoiseOpeningRow};

        let tree = make_tree_2s_2d();
        let tmp = tempfile::tempdir().expect("tempdir must succeed");
        let path = tmp.path().join("noise_openings.parquet");

        write_noise_openings(&path, &tree).expect("write must succeed");
        assert!(path.exists(), "file must exist after write");

        let rows: Vec<NoiseOpeningRow> = parse_noise_openings(&path).expect("parse must succeed");
        let recovered = assemble_opening_tree(rows, tree.dim());

        // Verify structural equality.
        assert_eq!(
            recovered.n_stages(),
            tree.n_stages(),
            "n_stages must be identical"
        );
        assert_eq!(recovered.dim(), tree.dim(), "dim must be identical");
        assert_eq!(
            recovered.openings_per_stage_slice(),
            tree.openings_per_stage_slice(),
            "openings_per_stage_slice must be identical"
        );
        // Verify data equality.
        assert_eq!(
            recovered.data(),
            tree.data(),
            "data arrays must be bit-for-bit identical"
        );
    }

    // -------------------------------------------------------------------------
    // write_creates_parent_directory
    // -------------------------------------------------------------------------

    #[test]
    fn write_creates_parent_directory() {
        let tree = make_tree_2s_2d();
        let tmp = tempfile::tempdir().expect("tempdir must succeed");
        // Nested path — neither intermediate directory exists yet.
        let path = tmp.path().join("output/stochastic/noise_openings.parquet");

        assert!(
            !path.parent().unwrap().exists(),
            "parent must not exist yet"
        );
        write_noise_openings(&path, &tree).expect("write must succeed even with missing parent");
        assert!(path.exists(), "file must exist");
        assert!(
            path.parent().unwrap().is_dir(),
            "parent directory must have been created"
        );
    }

    // -------------------------------------------------------------------------
    // write_correct_schema
    // -------------------------------------------------------------------------

    #[test]
    fn write_correct_schema() {
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

        let tree = make_tree_2s_2d();
        let tmp = tempfile::tempdir().expect("tempdir must succeed");
        let path = tmp.path().join("noise_openings.parquet");

        write_noise_openings(&path, &tree).expect("write must succeed");

        let file = std::fs::File::open(&path).expect("file must open");
        let builder =
            ParquetRecordBatchReaderBuilder::try_new(file).expect("builder must be created");
        let schema = builder.schema().clone();

        assert_eq!(
            schema.fields().len(),
            4,
            "schema must have exactly 4 fields"
        );

        // Verify field names and types.
        let fields: Vec<(&str, &DataType)> = schema
            .fields()
            .iter()
            .map(|f| (f.name().as_str(), f.data_type()))
            .collect();

        assert_eq!(fields[0], ("stage_id", &DataType::Int32));
        assert_eq!(fields[1], ("opening_index", &DataType::UInt32));
        assert_eq!(fields[2], ("entity_index", &DataType::UInt32));
        assert_eq!(fields[3], ("value", &DataType::Float64));
    }

    // -------------------------------------------------------------------------
    // write_row_count_matches_tree
    // -------------------------------------------------------------------------

    #[test]
    fn write_row_count_matches_tree() {
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

        let tree = make_tree_2s_2d();
        // Expected rows: (2 + 3) * 2 = 10
        let expected_rows: usize =
            tree.openings_per_stage_slice().iter().sum::<usize>() * tree.dim();

        let tmp = tempfile::tempdir().expect("tempdir must succeed");
        let path = tmp.path().join("noise_openings.parquet");

        write_noise_openings(&path, &tree).expect("write must succeed");

        let file = std::fs::File::open(&path).expect("file must open");
        let reader = ParquetRecordBatchReaderBuilder::try_new(file)
            .expect("builder")
            .build()
            .expect("reader");

        let total_rows: usize = reader
            .map(|b| b.expect("batch must be Ok").num_rows())
            .sum();

        assert_eq!(
            total_rows, expected_rows,
            "row count must equal sum(openings_per_stage) * dim"
        );
    }

    // -------------------------------------------------------------------------
    // Atomic write: .tmp file must not remain after success
    // -------------------------------------------------------------------------

    #[test]
    fn write_atomic_no_tmp_file_remains() {
        let tree = make_tree_2s_2d();
        let tmp = tempfile::tempdir().expect("tempdir must succeed");
        let path = tmp.path().join("noise_openings.parquet");

        write_noise_openings(&path, &tree).expect("write must succeed");

        let tmp_path = path.with_extension("parquet.tmp");
        assert!(
            !tmp_path.exists(),
            ".tmp file must not remain after successful atomic write"
        );
        assert!(path.exists(), "final file must exist");
    }

    // -------------------------------------------------------------------------
    // Correct (stage_id, opening_index, entity_index, value) tuples
    // -------------------------------------------------------------------------

    #[test]
    fn write_correct_row_tuples() {
        use arrow::array::{Float64Array, Int32Array, UInt32Array};
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

        let tree = make_tree_2s_2d();
        let tmp = tempfile::tempdir().expect("tempdir must succeed");
        let path = tmp.path().join("noise_openings.parquet");

        write_noise_openings(&path, &tree).expect("write must succeed");

        let file = std::fs::File::open(&path).expect("file must open");
        let mut reader = ParquetRecordBatchReaderBuilder::try_new(file)
            .expect("builder")
            .build()
            .expect("reader");
        let batch = reader.next().expect("must have a batch").expect("batch Ok");

        let stage_col = batch
            .column_by_name("stage_id")
            .unwrap()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        let opening_col = batch
            .column_by_name("opening_index")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap();
        let entity_col = batch
            .column_by_name("entity_index")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap();
        let value_col = batch
            .column_by_name("value")
            .unwrap()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();

        // Row 0: stage=0, opening=0, entity=0, value=1.0
        assert_eq!(stage_col.value(0), 0);
        assert_eq!(opening_col.value(0), 0);
        assert_eq!(entity_col.value(0), 0);
        assert_eq!(value_col.value(0), 1.0);

        // Row 1: stage=0, opening=0, entity=1, value=2.0
        assert_eq!(stage_col.value(1), 0);
        assert_eq!(opening_col.value(1), 0);
        assert_eq!(entity_col.value(1), 1);
        assert_eq!(value_col.value(1), 2.0);

        // Row 2: stage=0, opening=1, entity=0, value=3.0
        assert_eq!(stage_col.value(2), 0);
        assert_eq!(opening_col.value(2), 1);
        assert_eq!(entity_col.value(2), 0);
        assert_eq!(value_col.value(2), 3.0);

        // Row 4: stage=1, opening=0, entity=0, value=5.0
        assert_eq!(stage_col.value(4), 1);
        assert_eq!(opening_col.value(4), 0);
        assert_eq!(entity_col.value(4), 0);
        assert_eq!(value_col.value(4), 5.0);

        // Last row (9): stage=1, opening=2, entity=1, value=10.0
        assert_eq!(stage_col.value(9), 1);
        assert_eq!(opening_col.value(9), 2);
        assert_eq!(entity_col.value(9), 1);
        assert_eq!(value_col.value(9), 10.0);
    }

    // -------------------------------------------------------------------------
    // Empty tree (0 stages) writes a valid zero-row file
    // -------------------------------------------------------------------------

    #[test]
    fn write_empty_tree_zero_rows() {
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

        let tree = OpeningTree::from_parts(vec![], vec![], 2);
        let tmp = tempfile::tempdir().expect("tempdir must succeed");
        let path = tmp.path().join("noise_openings.parquet");

        write_noise_openings(&path, &tree).expect("write must succeed for empty tree");

        let file = std::fs::File::open(&path).expect("file must open");
        let reader = ParquetRecordBatchReaderBuilder::try_new(file)
            .expect("builder")
            .build()
            .expect("reader");

        let total_rows: usize = reader
            .map(|b| b.expect("batch must be Ok").num_rows())
            .sum();
        assert_eq!(total_rows, 0, "empty tree must produce 0-row file");
    }
}
