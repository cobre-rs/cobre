//! Parquet writer for training output files.
//!
//! [`TrainingParquetWriter`] produces two output files from a completed
//! training run:
//!
//! - `training/convergence.parquet` — one row per iteration, capturing
//!   bounds, gap, cut pool statistics, timing, and resource usage.
//! - `training/timing/iterations.parquet` — per-iteration timing breakdown
//!   (currently placeholder zeros — per-phase timing is not yet collected).
//!
//! The writer runs on rank 0 only, after training completes.  Both files
//! are written atomically: the data is written to a `.tmp` suffix first,
//! then renamed to the final path.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::array::{ArrayRef, Float64Builder, Int32Builder, Int64Builder, RecordBatch};
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;

use super::{IterationRecord, TrainingOutput};
use crate::output::error::OutputError;
use crate::output::parquet_config::ParquetWriterConfig;
use crate::output::schemas::{convergence_schema, iteration_timing_schema};

/// Writes training output to `training/convergence.parquet` and
/// `training/timing/iterations.parquet`.
///
/// Construct via [`TrainingParquetWriter::new`], then call [`write`][Self::write].
/// Both output files are written atomically using a `.tmp`-suffix temporary file.
///
/// # Examples
///
/// ```no_run
/// use cobre_io::{TrainingOutput, CutStatistics, ParquetWriterConfig};
/// use cobre_io::output::training_writer::TrainingParquetWriter;
/// use std::path::Path;
///
/// # fn main() -> Result<(), cobre_io::OutputError> {
/// let config = ParquetWriterConfig::default();
/// let writer = TrainingParquetWriter::new(Path::new("/tmp/out"), &config)?;
/// let training = TrainingOutput {
///     convergence_records: Vec::new(),
///     final_lower_bound: 42.0,
///     final_upper_bound: None,
///     final_gap_percent: None,
///     iterations_completed: 0,
///     converged: false,
///     termination_reason: "iteration limit".to_string(),
///     total_time_ms: 0,
///     cut_stats: CutStatistics {
///         total_generated: 0,
///         total_active: 0,
///         peak_active: 0,
///     },
///     cut_selection_records: Vec::new(),
/// };
/// writer.write(&training)?;
/// # Ok(())
/// # }
/// ```
pub struct TrainingParquetWriter {
    output_dir: PathBuf,
    config: ParquetWriterConfig,
}

impl TrainingParquetWriter {
    /// Create a new writer targeting `output_dir`.
    ///
    /// The training subdirectories (`training/` and `training/timing/`) must
    /// already exist — they are created by `write_results` before this
    /// constructor is called.
    ///
    /// # Errors
    ///
    /// - [`OutputError::IoError`] if the `training/` or `training/timing/`
    ///   directories do not exist or are not accessible.
    pub fn new(output_dir: &Path, config: &ParquetWriterConfig) -> Result<Self, OutputError> {
        let training_dir = output_dir.join("training");
        let timing_dir = output_dir.join("training/timing");

        if !training_dir.exists() {
            return Err(OutputError::io(
                &training_dir,
                std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    "training/ directory does not exist",
                ),
            ));
        }
        if !timing_dir.exists() {
            return Err(OutputError::io(
                &timing_dir,
                std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    "training/timing/ directory does not exist",
                ),
            ));
        }

        Ok(Self {
            output_dir: output_dir.to_path_buf(),
            config: config.clone(),
        })
    }

    /// Write `training/convergence.parquet` and `training/timing/iterations.parquet`.
    ///
    /// Both files are written atomically: each is first written to a `.tmp`
    /// suffix and then renamed to its final path. An empty
    /// `convergence_records` slice produces valid zero-row Parquet files
    /// with the correct schema.
    ///
    /// # Errors
    ///
    /// - [`OutputError::SerializationError`] if the Arrow `RecordBatch`
    ///   cannot be constructed (e.g., array length mismatch).
    /// - [`OutputError::IoError`] if any filesystem operation fails.
    pub fn write(&self, training_output: &TrainingOutput) -> Result<(), OutputError> {
        let records = &training_output.convergence_records;

        let convergence_batch = build_convergence_batch(records)?;
        let convergence_path = self.output_dir.join("training/convergence.parquet");
        write_parquet(&convergence_path, &convergence_batch, &self.config)?;

        let timing_batch = build_iteration_timing_batch(records)?;
        let timing_path = self.output_dir.join("training/timing/iterations.parquet");
        write_parquet(&timing_path, &timing_batch, &self.config)?;

        Ok(())
    }
}

/// Build a `RecordBatch` for `training/convergence.parquet` from iteration records.
///
/// Returns an error if the Arrow `RecordBatch::try_new` call fails (e.g.,
/// mismatched array lengths, which indicates a programming error).
#[allow(
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn build_convergence_batch(records: &[IterationRecord]) -> Result<RecordBatch, OutputError> {
    let schema = Arc::new(convergence_schema());
    let n = records.len();

    let mut iteration = Int32Builder::with_capacity(n);
    let mut lower_bound = Float64Builder::with_capacity(n);
    let mut upper_bound_mean = Float64Builder::with_capacity(n);
    let mut upper_bound_std = Float64Builder::with_capacity(n);
    let mut gap_percent = Float64Builder::with_capacity(n);
    let mut cuts_added = Int32Builder::with_capacity(n);
    let mut cuts_removed = Int32Builder::with_capacity(n);
    let mut cuts_active = Int64Builder::with_capacity(n);
    let mut time_forward_ms = Int64Builder::with_capacity(n);
    let mut time_backward_ms = Int64Builder::with_capacity(n);
    let mut time_total_ms = Int64Builder::with_capacity(n);
    let mut forward_passes = Int32Builder::with_capacity(n);
    let mut lp_solves = Int64Builder::with_capacity(n);

    for rec in records {
        iteration.append_value(rec.iteration as i32);
        lower_bound.append_value(rec.lower_bound);
        upper_bound_mean.append_value(rec.upper_bound_mean);
        upper_bound_std.append_value(rec.upper_bound_std);
        gap_percent.append_option(rec.gap_percent);
        cuts_added.append_value(rec.cuts_added as i32);
        cuts_removed.append_value(rec.cuts_removed as i32);
        cuts_active.append_value(i64::from(rec.cuts_active));
        time_forward_ms.append_value(rec.time_forward_ms as i64);
        time_backward_ms.append_value(rec.time_backward_ms as i64);
        time_total_ms.append_value(rec.time_total_ms as i64);
        forward_passes.append_value(rec.forward_passes as i32);
        lp_solves.append_value(i64::from(rec.lp_solves));
    }

    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(iteration.finish()),
            Arc::new(lower_bound.finish()),
            Arc::new(upper_bound_mean.finish()),
            Arc::new(upper_bound_std.finish()),
            Arc::new(gap_percent.finish()),
            Arc::new(cuts_added.finish()),
            Arc::new(cuts_removed.finish()),
            Arc::new(cuts_active.finish()),
            Arc::new(time_forward_ms.finish()),
            Arc::new(time_backward_ms.finish()),
            Arc::new(time_total_ms.finish()),
            Arc::new(forward_passes.finish()),
            Arc::new(lp_solves.finish()),
        ],
    )
    .map_err(|e| OutputError::serialization("convergence", e.to_string()))
}

/// Build a `RecordBatch` for `training/timing/iterations.parquet`.
///
/// Reads per-phase timing fields from each [`IterationRecord`] and writes them
/// as `i64` millisecond columns.
#[allow(clippy::cast_possible_wrap)]
fn build_iteration_timing_batch(records: &[IterationRecord]) -> Result<RecordBatch, OutputError> {
    let schema = Arc::new(iteration_timing_schema());
    let n = records.len();

    let mut iteration = Int32Builder::with_capacity(n);
    let mut forward_wall_ms = Int64Builder::with_capacity(n);
    let mut backward_wall_ms = Int64Builder::with_capacity(n);
    let mut cut_selection_ms = Int64Builder::with_capacity(n);
    let mut mpi_allreduce_ms = Int64Builder::with_capacity(n);
    let mut cut_sync_ms = Int64Builder::with_capacity(n);
    let mut lower_bound_ms = Int64Builder::with_capacity(n);
    let mut state_exchange_ms = Int64Builder::with_capacity(n);
    let mut cut_batch_build_ms = Int64Builder::with_capacity(n);
    let mut bwd_rayon_overhead_ms = Int64Builder::with_capacity(n);
    let mut fwd_rayon_overhead_ms = Int64Builder::with_capacity(n);
    let mut overhead_ms = Int64Builder::with_capacity(n);

    for rec in records {
        iteration.append_value(rec.iteration as i32);
        forward_wall_ms.append_value(rec.time_forward_wall_ms as i64);
        backward_wall_ms.append_value(rec.time_backward_wall_ms as i64);
        cut_selection_ms.append_value(rec.time_cut_selection_ms as i64);
        mpi_allreduce_ms.append_value(rec.time_mpi_allreduce_ms as i64);
        cut_sync_ms.append_value(rec.time_cut_sync_ms as i64);
        lower_bound_ms.append_value(rec.time_lower_bound_ms as i64);
        state_exchange_ms.append_value(rec.time_state_exchange_ms as i64);
        cut_batch_build_ms.append_value(rec.time_cut_batch_build_ms as i64);
        bwd_rayon_overhead_ms.append_value(rec.time_bwd_rayon_overhead_ms as i64);
        fwd_rayon_overhead_ms.append_value(rec.time_fwd_rayon_overhead_ms as i64);
        overhead_ms.append_value(rec.time_overhead_ms as i64);
    }

    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(iteration.finish()),
            Arc::new(forward_wall_ms.finish()),
            Arc::new(backward_wall_ms.finish()),
            Arc::new(cut_selection_ms.finish()),
            Arc::new(mpi_allreduce_ms.finish()),
            Arc::new(cut_sync_ms.finish()),
            Arc::new(lower_bound_ms.finish()),
            Arc::new(state_exchange_ms.finish()),
            Arc::new(cut_batch_build_ms.finish()),
            Arc::new(bwd_rayon_overhead_ms.finish()),
            Arc::new(fwd_rayon_overhead_ms.finish()),
            Arc::new(overhead_ms.finish()),
        ],
    )
    .map_err(|e| OutputError::serialization("iteration_timing", e.to_string()))
}

/// Write `training/cut_selection/iterations.parquet` from cut selection records.
///
/// Creates the `training/cut_selection/` directory if it does not exist.
/// Does nothing if `records` is empty.
///
/// # Errors
///
/// Returns [`OutputError`] on filesystem or serialization failures.
pub fn write_cut_selection_records(
    output_dir: &Path,
    records: &[super::CutSelectionRecord],
    config: &ParquetWriterConfig,
) -> Result<(), OutputError> {
    if records.is_empty() {
        return Ok(());
    }

    let dir = output_dir.join("training/cut_selection");
    std::fs::create_dir_all(&dir).map_err(|e| OutputError::io(&dir, e))?;

    let schema = Arc::new(super::schemas::cut_selection_schema());

    let n = records.len();
    let mut iteration_builder = Int32Builder::with_capacity(n);
    let mut stage_builder = Int32Builder::with_capacity(n);
    let mut populated_builder = Int32Builder::with_capacity(n);
    let mut active_before_builder = Int32Builder::with_capacity(n);
    let mut deactivated_builder = Int32Builder::with_capacity(n);
    let mut active_after_builder = Int32Builder::with_capacity(n);
    let mut selection_time_builder = Float64Builder::with_capacity(n);
    let mut budget_evicted_builder = Int32Builder::with_capacity(n);
    let mut active_after_angular_builder = Int32Builder::with_capacity(n);
    let mut active_after_budget_builder = Int32Builder::with_capacity(n);

    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    for r in records {
        iteration_builder.append_value(r.iteration as i32);
        stage_builder.append_value(r.stage as i32);
        populated_builder.append_value(r.cuts_populated as i32);
        active_before_builder.append_value(r.cuts_active_before as i32);
        deactivated_builder.append_value(r.cuts_deactivated as i32);
        active_after_builder.append_value(r.cuts_active_after as i32);
        selection_time_builder.append_value(r.selection_time_ms);
        budget_evicted_builder.append_option(r.budget_evicted.map(|v| v as i32));
        active_after_angular_builder.append_option(r.active_after_angular.map(|v| v as i32));
        active_after_budget_builder.append_option(r.active_after_budget.map(|v| v as i32));
    }

    let columns: Vec<ArrayRef> = vec![
        Arc::new(iteration_builder.finish()),
        Arc::new(stage_builder.finish()),
        Arc::new(populated_builder.finish()),
        Arc::new(active_before_builder.finish()),
        Arc::new(deactivated_builder.finish()),
        Arc::new(active_after_builder.finish()),
        Arc::new(selection_time_builder.finish()),
        Arc::new(budget_evicted_builder.finish()),
        Arc::new(active_after_angular_builder.finish()),
        Arc::new(active_after_budget_builder.finish()),
    ];

    let batch = RecordBatch::try_new(Arc::clone(&schema), columns)
        .map_err(|e| OutputError::serialization("cut_selection", e.to_string()))?;

    write_parquet(&dir.join("iterations.parquet"), &batch, config)
}

/// Write a `RecordBatch` to `path` as a Parquet file, atomically.
///
/// The batch is written to `{path}.tmp` first, then renamed to `path`.  If
/// any step fails the `.tmp` file may remain on disk but the final path is
/// never partially written.
fn write_parquet(
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
        .map_err(|e| OutputError::serialization("parquet_writer", e.to_string()))?;

    writer
        .close()
        .map_err(|e| OutputError::serialization("parquet_writer", e.to_string()))?;

    std::fs::rename(&tmp_path, path).map_err(|e| OutputError::io(path, e))?;

    Ok(())
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp
)]
mod tests {
    use super::*;
    use crate::output::{CutStatistics, TrainingOutput};

    fn make_record(iteration: u32, gap: Option<f64>) -> IterationRecord {
        IterationRecord {
            iteration,
            lower_bound: f64::from(iteration) * 10.0,
            upper_bound_mean: f64::from(iteration) * 11.0,
            upper_bound_std: 0.5,
            gap_percent: gap,
            cuts_added: 5,
            cuts_removed: 1,
            cuts_active: 4,
            time_forward_ms: 100,
            time_backward_ms: 200,
            time_total_ms: 300,
            time_forward_wall_ms: 100,
            time_backward_wall_ms: 200,
            time_cut_selection_ms: 0,
            time_mpi_allreduce_ms: 0,
            time_cut_sync_ms: 0,
            time_lower_bound_ms: 0,
            time_state_exchange_ms: 0,
            time_cut_batch_build_ms: 0,
            time_bwd_rayon_overhead_ms: 0,
            time_fwd_rayon_overhead_ms: 0,
            time_overhead_ms: 0,
            forward_passes: 4,
            lp_solves: 40,
            solve_time_ms: 0.0,
        }
    }

    fn make_training_output(records: Vec<IterationRecord>) -> TrainingOutput {
        TrainingOutput {
            convergence_records: records,
            final_lower_bound: 99.5,
            final_upper_bound: Some(101.0),
            final_gap_percent: Some(1.51),
            iterations_completed: 0,
            converged: true,
            termination_reason: "gap tolerance reached".to_string(),
            total_time_ms: 5_000,
            cut_stats: CutStatistics {
                total_generated: 200,
                total_active: 80,
                peak_active: 95,
            },
            cut_selection_records: vec![],
        }
    }

    // -------------------------------------------------------------------------
    // build_convergence_batch tests
    // -------------------------------------------------------------------------

    #[test]
    fn convergence_batch_from_empty_records() {
        let batch = build_convergence_batch(&[]).expect("empty batch must succeed");
        assert_eq!(batch.num_rows(), 0, "empty records yield 0 rows");
        assert_eq!(batch.num_columns(), 13, "convergence schema has 13 columns");
    }

    #[test]
    fn convergence_batch_field_count_and_types() {
        let records: Vec<IterationRecord> = (1..=3).map(|i| make_record(i, Some(5.0))).collect();
        let batch = build_convergence_batch(&records).expect("batch must be built");
        assert_eq!(batch.num_rows(), 3);
        assert_eq!(batch.num_columns(), 13);

        let expected_schema = convergence_schema();
        assert_eq!(
            batch.schema().fields(),
            expected_schema.fields(),
            "schema must match convergence_schema()"
        );
    }

    #[test]
    fn convergence_batch_nullable_columns() {
        let records = vec![
            make_record(1, Some(10.0)),
            make_record(2, Some(5.0)),
            make_record(3, None), // gap_percent is None for record 3 (index 2)
        ];
        let batch = build_convergence_batch(&records).expect("batch must be built");

        let gap_col = batch
            .column_by_name("gap_percent")
            .expect("gap_percent column must exist");

        // Arrow arrays track nulls via the validity bitmap.
        assert!(!gap_col.is_null(0), "row 0: Some(10.0) must not be null");
        assert!(!gap_col.is_null(1), "row 1: Some(5.0) must not be null");
        assert!(gap_col.is_null(2), "row 2: None must be null");
    }

    // -------------------------------------------------------------------------
    // build_iteration_timing_batch tests
    // -------------------------------------------------------------------------

    #[test]
    fn iteration_timing_batch_field_count() {
        let records: Vec<IterationRecord> = (1..=3).map(|i| make_record(i, Some(1.0))).collect();
        let batch = build_iteration_timing_batch(&records).expect("timing batch must be built");
        assert_eq!(batch.num_rows(), 3, "3 records yield 3 rows");
        assert_eq!(
            batch.num_columns(),
            12,
            "iteration_timing schema has 12 columns"
        );

        let expected_schema = iteration_timing_schema();
        assert_eq!(
            batch.schema().fields(),
            expected_schema.fields(),
            "schema must match iteration_timing_schema()"
        );
    }

    // -------------------------------------------------------------------------
    // write_parquet + roundtrip tests
    // -------------------------------------------------------------------------

    #[test]
    fn write_convergence_parquet_roundtrip() {
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

        let records: Vec<IterationRecord> = (1..=5).map(|i| make_record(i, Some(1.0))).collect();
        let batch = build_convergence_batch(&records).expect("batch must be built");

        let tmp = tempfile::tempdir().expect("tempdir must succeed");
        let path = tmp.path().join("convergence.parquet");
        let config = ParquetWriterConfig::default();

        write_parquet(&path, &batch, &config).expect("write must succeed");
        assert!(path.exists(), "convergence.parquet must exist after write");

        // Read back and verify row count + column values.
        let file = std::fs::File::open(&path).expect("file must open");
        let builder =
            ParquetRecordBatchReaderBuilder::try_new(file).expect("builder must be created");
        let mut reader = builder.build().expect("reader must be built");

        let read_batch = reader
            .next()
            .expect("must have at least one batch")
            .expect("batch must be Ok");
        assert_eq!(read_batch.num_rows(), 5, "must have 5 rows");

        let expected_schema = convergence_schema();
        assert_eq!(
            read_batch.schema().fields(),
            expected_schema.fields(),
            "schema must match convergence_schema()"
        );

        // Verify iteration column values [1, 2, 3, 4, 5] as Int32.
        let iteration_col = read_batch
            .column_by_name("iteration")
            .expect("iteration column must exist");
        let iteration_arr = iteration_col
            .as_any()
            .downcast_ref::<arrow::array::Int32Array>()
            .expect("iteration must be Int32Array");
        let iteration_values: Vec<i32> = (0..5).map(|i| iteration_arr.value(i)).collect();
        assert_eq!(iteration_values, vec![1, 2, 3, 4, 5]);

        // Verify lower_bound column.
        let lb_col = read_batch
            .column_by_name("lower_bound")
            .expect("lower_bound column must exist");
        let lb_arr = lb_col
            .as_any()
            .downcast_ref::<arrow::array::Float64Array>()
            .expect("lower_bound must be Float64Array");
        for (i, rec) in records.iter().enumerate() {
            assert_eq!(
                lb_arr.value(i),
                rec.lower_bound,
                "lower_bound mismatch at row {i}"
            );
        }
    }

    #[test]
    fn write_convergence_parquet_atomic_rename() {
        let records: Vec<IterationRecord> = (1..=2).map(|i| make_record(i, None)).collect();
        let batch = build_convergence_batch(&records).expect("batch must be built");

        let tmp = tempfile::tempdir().expect("tempdir must succeed");
        let path = tmp.path().join("convergence.parquet");
        let config = ParquetWriterConfig::default();

        write_parquet(&path, &batch, &config).expect("write must succeed");

        // The .tmp file must not remain after a successful write.
        let tmp_path = path.with_extension("parquet.tmp");
        assert!(
            !tmp_path.exists(),
            ".tmp file must not exist after successful atomic rename"
        );
        assert!(path.exists(), "final file must exist");
    }

    // -------------------------------------------------------------------------
    // TrainingParquetWriter integration tests
    // -------------------------------------------------------------------------

    #[test]
    fn writer_fails_if_training_dir_missing() {
        let tmp = tempfile::tempdir().expect("tempdir must succeed");
        let config = ParquetWriterConfig::default();

        // Do not create the training/ directory.
        let result = TrainingParquetWriter::new(tmp.path(), &config);
        assert!(result.is_err(), "new() must fail when training/ is missing");
    }

    #[test]
    fn writer_fails_if_timing_dir_missing() {
        let tmp = tempfile::tempdir().expect("tempdir must succeed");
        let config = ParquetWriterConfig::default();

        // Create training/ but not training/timing/.
        std::fs::create_dir_all(tmp.path().join("training")).unwrap();

        let result = TrainingParquetWriter::new(tmp.path(), &config);
        assert!(
            result.is_err(),
            "new() must fail when training/timing/ is missing"
        );
    }

    #[test]
    fn writer_writes_empty_training_output() {
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

        let tmp = tempfile::tempdir().expect("tempdir must succeed");
        std::fs::create_dir_all(tmp.path().join("training/timing")).unwrap();
        let config = ParquetWriterConfig::default();

        let writer = TrainingParquetWriter::new(tmp.path(), &config).expect("new must succeed");
        let training = make_training_output(vec![]);
        writer.write(&training).expect("write must succeed");

        let conv_path = tmp.path().join("training/convergence.parquet");
        assert!(conv_path.exists(), "convergence.parquet must exist");

        let timing_path = tmp.path().join("training/timing/iterations.parquet");
        assert!(timing_path.exists(), "iterations.parquet must exist");

        // Verify zero-row convergence file with correct schema.
        let file = std::fs::File::open(&conv_path).expect("file must open");
        let builder = ParquetRecordBatchReaderBuilder::try_new(file).expect("builder created");
        let schema = builder.schema().clone();
        let reader = builder.build().expect("reader built");

        // A zero-row file may yield no batches at all — that is correct.
        let total_rows: usize = reader
            .map(|b| b.expect("batch must be Ok").num_rows())
            .sum();
        assert_eq!(total_rows, 0, "empty training output must produce 0 rows");

        let expected_schema = convergence_schema();
        assert_eq!(
            schema.fields(),
            expected_schema.fields(),
            "schema must match convergence_schema()"
        );
    }

    #[test]
    fn writer_writes_five_records() {
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

        let tmp = tempfile::tempdir().expect("tempdir must succeed");
        std::fs::create_dir_all(tmp.path().join("training/timing")).unwrap();
        let config = ParquetWriterConfig::default();

        let records: Vec<IterationRecord> = (1..=5).map(|i| make_record(i, Some(1.0))).collect();
        let training = make_training_output(records);

        let writer = TrainingParquetWriter::new(tmp.path(), &config).expect("new must succeed");
        writer.write(&training).expect("write must succeed");

        let conv_path = tmp.path().join("training/convergence.parquet");
        let file = std::fs::File::open(&conv_path).expect("file must open");
        let mut reader = ParquetRecordBatchReaderBuilder::try_new(file)
            .expect("builder")
            .build()
            .expect("reader");
        let batch = reader.next().expect("must have rows").expect("batch Ok");
        assert_eq!(batch.num_rows(), 5);
        assert_eq!(batch.num_columns(), 13);

        let timing_path = tmp.path().join("training/timing/iterations.parquet");
        let file = std::fs::File::open(&timing_path).expect("file must open");
        let mut reader = ParquetRecordBatchReaderBuilder::try_new(file)
            .expect("builder")
            .build()
            .expect("reader");
        let batch = reader.next().expect("must have rows").expect("batch Ok");
        assert_eq!(batch.num_rows(), 5);
        assert_eq!(batch.num_columns(), 12);
    }

    #[test]
    fn writer_gap_percent_null_at_correct_row() {
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

        let tmp = tempfile::tempdir().expect("tempdir must succeed");
        std::fs::create_dir_all(tmp.path().join("training/timing")).unwrap();
        let config = ParquetWriterConfig::default();

        let records = vec![
            make_record(1, Some(10.0)),
            make_record(2, Some(5.0)),
            make_record(3, None), // record 3 (row index 2): gap_percent = None
            make_record(4, Some(2.0)),
            make_record(5, Some(1.0)),
        ];
        let training = make_training_output(records);

        let writer = TrainingParquetWriter::new(tmp.path(), &config).expect("new must succeed");
        writer.write(&training).expect("write must succeed");

        let conv_path = tmp.path().join("training/convergence.parquet");
        let file = std::fs::File::open(&conv_path).expect("file must open");
        let mut reader = ParquetRecordBatchReaderBuilder::try_new(file)
            .expect("builder")
            .build()
            .expect("reader");
        let batch = reader.next().expect("must have rows").expect("batch Ok");

        let gap_col = batch
            .column_by_name("gap_percent")
            .expect("gap_percent column must exist");

        assert!(!gap_col.is_null(0), "row 0: Some(10.0) must not be null");
        assert!(!gap_col.is_null(1), "row 1: Some(5.0) must not be null");
        assert!(gap_col.is_null(2), "row 2: None must be null");
        assert!(!gap_col.is_null(3), "row 3: Some(2.0) must not be null");
        assert!(!gap_col.is_null(4), "row 4: Some(1.0) must not be null");
    }

    // -------------------------------------------------------------------------
    // write_cut_selection_records tests
    // -------------------------------------------------------------------------

    #[test]
    fn write_cut_selection_empty_is_noop() {
        let tmp = tempfile::tempdir().unwrap();
        let config = ParquetWriterConfig::default();
        write_cut_selection_records(tmp.path(), &[], &config).unwrap();
        assert!(
            !tmp.path()
                .join("training/cut_selection/iterations.parquet")
                .exists()
        );
    }

    #[test]
    fn write_cut_selection_roundtrip() {
        use super::super::CutSelectionRecord;

        let tmp = tempfile::tempdir().unwrap();
        let config = ParquetWriterConfig::default();
        let records = vec![
            CutSelectionRecord {
                iteration: 3,
                stage: 0,
                cuts_populated: 10,
                cuts_active_before: 10,
                cuts_deactivated: 0,
                cuts_active_after: 10,
                selection_time_ms: 0.0,
                budget_evicted: None,
                active_after_angular: None,
                active_after_budget: None,
            },
            CutSelectionRecord {
                iteration: 3,
                stage: 1,
                cuts_populated: 8,
                cuts_active_before: 8,
                cuts_deactivated: 2,
                cuts_active_after: 6,
                selection_time_ms: 1.5,
                budget_evicted: None,
                active_after_angular: None,
                active_after_budget: None,
            },
        ];
        write_cut_selection_records(tmp.path(), &records, &config).unwrap();
        let path = tmp.path().join("training/cut_selection/iterations.parquet");
        assert!(path.exists());

        let file = std::fs::File::open(&path).unwrap();
        let reader = parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(file)
            .unwrap()
            .build()
            .unwrap();
        let batch: RecordBatch = reader.into_iter().next().unwrap().unwrap();
        assert_eq!(batch.num_rows(), 2);
        assert_eq!(batch.num_columns(), 10);
    }

    #[test]
    fn write_cut_selection_with_budget_columns_roundtrip() {
        use super::super::CutSelectionRecord;
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

        let tmp = tempfile::tempdir().unwrap();
        let config = ParquetWriterConfig::default();
        let records = vec![
            // Record with all budget columns populated (budget enabled).
            CutSelectionRecord {
                iteration: 5,
                stage: 0,
                cuts_populated: 20,
                cuts_active_before: 20,
                cuts_deactivated: 0,
                cuts_active_after: 20,
                selection_time_ms: 0.0,
                budget_evicted: Some(3),
                active_after_angular: Some(18),
                active_after_budget: Some(15),
            },
            // Record with all budget columns None (budget disabled).
            CutSelectionRecord {
                iteration: 5,
                stage: 1,
                cuts_populated: 15,
                cuts_active_before: 15,
                cuts_deactivated: 2,
                cuts_active_after: 13,
                selection_time_ms: 2.0,
                budget_evicted: None,
                active_after_angular: None,
                active_after_budget: None,
            },
        ];
        write_cut_selection_records(tmp.path(), &records, &config).unwrap();
        let path = tmp.path().join("training/cut_selection/iterations.parquet");
        assert!(path.exists());

        let file = std::fs::File::open(&path).unwrap();
        let mut reader = ParquetRecordBatchReaderBuilder::try_new(file)
            .unwrap()
            .build()
            .unwrap();
        let batch = reader.next().unwrap().unwrap();
        assert_eq!(batch.num_rows(), 2);
        assert_eq!(batch.num_columns(), 10);

        // Verify nullable columns: row 0 has Some values, row 1 has None.
        let budget_evicted_col = batch.column_by_name("budget_evicted").unwrap();
        assert!(
            !budget_evicted_col.is_null(0),
            "row 0: budget_evicted Some(3) must not be null"
        );
        assert!(
            budget_evicted_col.is_null(1),
            "row 1: budget_evicted None must be null"
        );

        let angular_col = batch.column_by_name("active_after_angular").unwrap();
        assert!(
            !angular_col.is_null(0),
            "row 0: active_after_angular Some(18) must not be null"
        );
        assert!(
            angular_col.is_null(1),
            "row 1: active_after_angular None must be null"
        );

        let budget_col = batch.column_by_name("active_after_budget").unwrap();
        assert!(
            !budget_col.is_null(0),
            "row 0: active_after_budget Some(15) must not be null"
        );
        assert!(
            budget_col.is_null(1),
            "row 1: active_after_budget None must be null"
        );

        // Verify the actual values for row 0.
        let budget_evicted_arr = budget_evicted_col
            .as_any()
            .downcast_ref::<arrow::array::Int32Array>()
            .unwrap();
        assert_eq!(budget_evicted_arr.value(0), 3);

        let angular_arr = angular_col
            .as_any()
            .downcast_ref::<arrow::array::Int32Array>()
            .unwrap();
        assert_eq!(angular_arr.value(0), 18);

        let budget_arr = budget_col
            .as_any()
            .downcast_ref::<arrow::array::Int32Array>()
            .unwrap();
        assert_eq!(budget_arr.value(0), 15);
    }
}
