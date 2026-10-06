//! Parquet writer for training output files.
//!
//! [`TrainingParquetWriter`] produces two output files from a completed
//! training run:
//!
//! - `training/convergence.parquet` — one row per iteration, capturing
//!   bounds, gap, row-pool statistics, timing, and resource usage.
//! - `training/timing/iterations.parquet` — per-iteration timing breakdown
//!   with measured wall-clock milliseconds for each solver phase.
//!
//! The writer runs on rank 0 only.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::array::{
    ArrayRef, Float64Builder, Int32Builder, Int64Builder, RecordBatch, StringBuilder,
};

use super::{IterationRecord, TrainingOutput, WorkerTimingRecord};
use crate::output::atomic::{write_batch_atomic, write_parquet_atomic};
use crate::output::error::OutputError;
use crate::output::fixed_delivery::FIXED_DELIVERIES_FILE;
use crate::output::generic_constraints_echo::GENERIC_CONSTRAINT_ECHO_FILE;
use crate::output::hydro_models::{
    EVAPORATION_MODELS_FILE, FPHA_DEVIATION_POINTS_FILE, FPHA_HYPERPLANES_FILE,
};
use crate::output::schemas::{convergence_schema, iteration_timing_schema};
use crate::output::simulation_writer::{remove_dir_if_empty, remove_file_if_present};
use crate::output::solver_stats_writer::{
    SOLVER_ITERATIONS_FILE, SOLVER_RETRY_HISTOGRAM_FILE, TRAINING_SOLVER_DIR,
};

pub(crate) const CUT_SELECTION_FILE: &str = "training/cut_selection/iterations.parquet";

/// Writes training output to `training/convergence.parquet` and
/// `training/timing/iterations.parquet`, each written atomically.
///
/// # Examples
///
/// ```no_run
/// use cobre_io::{TrainingOutput, RowPoolStatistics, RunStatus};
/// use cobre_io::MetadataTrainingSolveStats;
/// use cobre_io::output::training_writer::TrainingParquetWriter;
/// use std::path::Path;
///
/// # fn main() -> Result<(), cobre_io::OutputError> {
/// let writer = TrainingParquetWriter::new(Path::new("/tmp/out"))?;
/// let training = TrainingOutput {
///     convergence_records: Vec::new(),
///     final_lower_bound: 42.0,
///     final_upper_bound: None,
///     final_gap_percent: None,
///     final_upper_bound_std: None,
///     final_upper_bound_kind: "statistical".to_string(),
///     iterations_completed: 0,
///     converged: false,
///     termination_reason: "iteration limit".to_string(),
///     status: RunStatus::Complete,
///     total_time_ms: 0,
///     cut_stats: RowPoolStatistics {
///         total_generated: 0,
///         total_active: 0,
///         peak_active: 0,
///         cuts_active: 0,
///         rows_in_lp_total: 0,
///         rows_in_lp_solve_count: 0,
///         rows_in_lp_max: 0,
///         total_loaded: 0,
///     },
///     cut_selection_records: Vec::new(),
///     worker_timing_records: Vec::new(),
///     training_solve_stats: MetadataTrainingSolveStats::default(),
/// };
/// writer.write(&training)?;
/// # Ok(())
/// # }
/// ```
pub struct TrainingParquetWriter {
    output_dir: PathBuf,
}

impl TrainingParquetWriter {
    /// Create a new writer targeting `output_dir`.
    ///
    /// The `training/` and `training/timing/` subdirectories must already exist
    /// (created by `write_results`); the constructor errors otherwise.
    ///
    /// # Errors
    ///
    /// - [`OutputError::IoError`] if the `training/` or `training/timing/`
    ///   directories do not exist or are not accessible.
    pub fn new(output_dir: &Path) -> Result<Self, OutputError> {
        let training_dir = output_dir.join("training");
        let timing_dir = output_dir.join("training/timing");

        require_dir_exists(&training_dir, "training/")?;
        require_dir_exists(&timing_dir, "training/timing/")?;

        Ok(Self {
            output_dir: output_dir.to_path_buf(),
        })
    }

    /// Write `training/convergence.parquet` and `training/timing/iterations.parquet`.
    ///
    /// An empty `convergence_records` slice produces valid zero-row Parquet files
    /// with the correct schema.
    ///
    /// # Errors
    ///
    /// - [`OutputError::SerializationError`] if the Arrow `RecordBatch`
    ///   cannot be constructed (e.g., array length mismatch).
    /// - [`OutputError::IoError`] if any filesystem operation fails.
    pub fn write(&self, training_output: &TrainingOutput) -> Result<(), OutputError> {
        let convergence_batch = build_convergence_batch(
            &training_output.convergence_records,
            &training_output.final_upper_bound_kind,
        )?;
        let convergence_path = self.output_dir.join("training/convergence.parquet");
        write_parquet_atomic(&convergence_path, &convergence_batch)?;

        let timing_batch = build_iteration_timing_batch(&training_output.worker_timing_records)?;
        let timing_path = self.output_dir.join("training/timing/iterations.parquet");
        write_parquet_atomic(&timing_path, &timing_batch)?;

        Ok(())
    }
}

fn require_dir_exists(dir: &Path, label: &str) -> Result<(), OutputError> {
    if !dir.exists() {
        return Err(OutputError::io(
            dir,
            std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("{label} directory does not exist"),
            ),
        ));
    }
    Ok(())
}

/// Build a `RecordBatch` for `training/convergence.parquet` from iteration records.
#[allow(
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn build_convergence_batch(
    records: &[IterationRecord],
    upper_bound_kind: &str,
) -> Result<RecordBatch, OutputError> {
    let schema = Arc::new(convergence_schema());
    let n = records.len();
    // Exact bounds carry no sampling distribution; std is NULL for all rows when is_exact.
    let is_exact = upper_bound_kind == "exact";

    let mut iteration = Int32Builder::with_capacity(n);
    let mut lower_bound = Float64Builder::with_capacity(n);
    let mut upper_bound = Float64Builder::with_capacity(n);
    let mut upper_bound_std = Float64Builder::with_capacity(n);
    let mut upper_bound_kind_col = StringBuilder::with_capacity(n, n * 12);
    let mut gap_percent = Float64Builder::with_capacity(n);
    let mut cuts_added = Int32Builder::with_capacity(n);
    let mut cuts_removed = Int32Builder::with_capacity(n);
    let mut cuts_active = Int64Builder::with_capacity(n);
    let mut time_forward_ms = Int64Builder::with_capacity(n);
    let mut time_backward_ms = Int64Builder::with_capacity(n);
    let mut time_total_ms = Int64Builder::with_capacity(n);
    let mut forward_passes = Int32Builder::with_capacity(n);
    let mut lp_solves = Int64Builder::with_capacity(n);
    let mut mean_rows_in_lp = Float64Builder::with_capacity(n);

    for rec in records {
        iteration.append_value(rec.iteration as i32);
        lower_bound.append_value(rec.lower_bound);
        upper_bound.append_value(rec.upper_bound);
        if is_exact {
            upper_bound_std.append_null();
        } else {
            upper_bound_std.append_value(rec.upper_bound_std);
        }
        upper_bound_kind_col.append_value(upper_bound_kind);
        gap_percent.append_option(rec.gap_percent);
        cuts_added.append_value(rec.cuts_added as i32);
        cuts_removed.append_value(rec.cuts_removed as i32);
        cuts_active.append_value(i64::from(rec.cuts_active));
        time_forward_ms.append_value(rec.time_forward_ms as i64);
        time_backward_ms.append_value(rec.time_backward_ms as i64);
        time_total_ms.append_value(rec.time_total_ms as i64);
        forward_passes.append_value(rec.forward_passes as i32);
        lp_solves.append_value(i64::from(rec.lp_solves));
        mean_rows_in_lp.append_value(rec.mean_rows_in_lp);
    }

    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(iteration.finish()),
            Arc::new(lower_bound.finish()),
            Arc::new(upper_bound.finish()),
            Arc::new(upper_bound_std.finish()),
            Arc::new(upper_bound_kind_col.finish()),
            Arc::new(gap_percent.finish()),
            Arc::new(cuts_added.finish()),
            Arc::new(cuts_removed.finish()),
            Arc::new(cuts_active.finish()),
            Arc::new(time_forward_ms.finish()),
            Arc::new(time_backward_ms.finish()),
            Arc::new(time_total_ms.finish()),
            Arc::new(forward_passes.finish()),
            Arc::new(lp_solves.finish()),
            Arc::new(mean_rows_in_lp.finish()),
        ],
    )
    .map_err(|e| OutputError::serialization("convergence", e.to_string()))
}

/// Build a `RecordBatch` for `training/timing/iterations.parquet`, one row per
/// [`WorkerTimingRecord`]. `worker_id` is `NULL` for rank-aggregated rows.
#[allow(clippy::cast_possible_wrap)]
fn build_iteration_timing_batch(
    records: &[WorkerTimingRecord],
) -> Result<RecordBatch, OutputError> {
    let schema = Arc::new(iteration_timing_schema());
    let n = records.len();

    let mut iteration = Int32Builder::with_capacity(n);
    let mut rank = Int32Builder::with_capacity(n);
    let mut worker_id = Int32Builder::with_capacity(n);
    let mut forward_wall_ms = Int64Builder::with_capacity(n);
    let mut backward_wall_ms = Int64Builder::with_capacity(n);
    let mut cut_selection_ms = Int64Builder::with_capacity(n);
    let mut mpi_allreduce_ms = Int64Builder::with_capacity(n);
    let mut cut_sync_ms = Int64Builder::with_capacity(n);
    let mut lower_bound_ms = Int64Builder::with_capacity(n);
    let mut state_exchange_ms = Int64Builder::with_capacity(n);
    let mut cut_batch_build_ms = Int64Builder::with_capacity(n);
    let mut bwd_setup_ms = Int64Builder::with_capacity(n);
    let mut bwd_load_imbalance_ms = Int64Builder::with_capacity(n);
    let mut bwd_scheduling_overhead_ms = Int64Builder::with_capacity(n);
    let mut fwd_setup_ms = Int64Builder::with_capacity(n);
    let mut fwd_load_imbalance_ms = Int64Builder::with_capacity(n);
    let mut fwd_scheduling_overhead_ms = Int64Builder::with_capacity(n);
    let mut overhead_ms = Int64Builder::with_capacity(n);
    let mut lazy_scoring_ms = Int64Builder::with_capacity(n);

    for rec in records {
        iteration.append_value(rec.iteration as i32);
        rank.append_value(rec.rank);
        worker_id.append_option(rec.worker_id);
        // timings slot order must match the column order in iteration_timing_schema().
        forward_wall_ms.append_value(rec.timings[0] as i64);
        backward_wall_ms.append_value(rec.timings[1] as i64);
        cut_selection_ms.append_value(rec.timings[2] as i64);
        mpi_allreduce_ms.append_value(rec.timings[3] as i64);
        cut_sync_ms.append_value(rec.timings[4] as i64);
        lower_bound_ms.append_value(rec.timings[5] as i64);
        state_exchange_ms.append_value(rec.timings[6] as i64);
        cut_batch_build_ms.append_value(rec.timings[7] as i64);
        bwd_setup_ms.append_value(rec.timings[8] as i64);
        bwd_load_imbalance_ms.append_value(rec.timings[9] as i64);
        bwd_scheduling_overhead_ms.append_value(rec.timings[10] as i64);
        fwd_setup_ms.append_value(rec.timings[11] as i64);
        fwd_load_imbalance_ms.append_value(rec.timings[12] as i64);
        fwd_scheduling_overhead_ms.append_value(rec.timings[13] as i64);
        overhead_ms.append_value(rec.timings[14] as i64);
        lazy_scoring_ms.append_value(rec.timings[15] as i64);
    }

    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(iteration.finish()),
            Arc::new(rank.finish()),
            Arc::new(worker_id.finish()),
            Arc::new(forward_wall_ms.finish()),
            Arc::new(backward_wall_ms.finish()),
            Arc::new(cut_selection_ms.finish()),
            Arc::new(mpi_allreduce_ms.finish()),
            Arc::new(cut_sync_ms.finish()),
            Arc::new(lower_bound_ms.finish()),
            Arc::new(state_exchange_ms.finish()),
            Arc::new(cut_batch_build_ms.finish()),
            Arc::new(bwd_setup_ms.finish()),
            Arc::new(bwd_load_imbalance_ms.finish()),
            Arc::new(bwd_scheduling_overhead_ms.finish()),
            Arc::new(fwd_setup_ms.finish()),
            Arc::new(fwd_load_imbalance_ms.finish()),
            Arc::new(fwd_scheduling_overhead_ms.finish()),
            Arc::new(overhead_ms.finish()),
            Arc::new(lazy_scoring_ms.finish()),
        ],
    )
    .map_err(|e| OutputError::serialization("iteration_timing", e.to_string()))
}

/// Write `training/cut_selection/iterations.parquet`.
///
/// Does nothing (no directory, no file) if `records` is empty.
///
/// # Errors
///
/// Returns [`OutputError`] on filesystem or serialization failures.
pub fn write_row_selection_records(
    output_dir: &Path,
    records: &[super::RowSelectionRecord],
) -> Result<(), OutputError> {
    if records.is_empty() {
        return Ok(());
    }

    let schema = Arc::new(super::schemas::row_selection_schema());

    let n = records.len();
    let mut iteration_builder = Int32Builder::with_capacity(n);
    let mut stage_builder = Int32Builder::with_capacity(n);
    let mut populated_builder = Int32Builder::with_capacity(n);
    let mut active_before_builder = Int32Builder::with_capacity(n);
    let mut deactivated_builder = Int32Builder::with_capacity(n);
    let mut reactivated_builder = Int32Builder::with_capacity(n);
    let mut active_after_builder = Int32Builder::with_capacity(n);
    let mut selection_time_builder = Float64Builder::with_capacity(n);
    let mut budget_evicted_builder = Int32Builder::with_capacity(n);
    let mut active_after_budget_builder = Int32Builder::with_capacity(n);
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    for r in records {
        iteration_builder.append_value(r.iteration as i32);
        stage_builder.append_value(r.stage as i32);
        populated_builder.append_value(r.cuts_populated as i32);
        active_before_builder.append_value(r.cuts_active_before as i32);
        deactivated_builder.append_value(r.cuts_deactivated as i32);
        reactivated_builder.append_value(r.cuts_reactivated as i32);
        active_after_builder.append_value(r.cuts_active_after as i32);
        selection_time_builder.append_value(r.selection_time_ms);
        budget_evicted_builder.append_option(r.budget_evicted.map(|v| v as i32));
        active_after_budget_builder.append_option(r.active_after_budget.map(|v| v as i32));
    }

    let columns: Vec<ArrayRef> = vec![
        Arc::new(iteration_builder.finish()),
        Arc::new(stage_builder.finish()),
        Arc::new(populated_builder.finish()),
        Arc::new(active_before_builder.finish()),
        Arc::new(deactivated_builder.finish()),
        Arc::new(reactivated_builder.finish()),
        Arc::new(active_after_builder.finish()),
        Arc::new(selection_time_builder.finish()),
        Arc::new(budget_evicted_builder.finish()),
        Arc::new(active_after_budget_builder.finish()),
    ];

    let batch = RecordBatch::try_new(schema, columns)
        .map_err(|e| OutputError::serialization("cut_selection", e.to_string()))?;

    write_batch_atomic(&output_dir.join(CUT_SELECTION_FILE), &batch)
}

fn conditional_training_output_files(output_dir: &Path) -> [PathBuf; 8] {
    let solver_dir = output_dir.join(TRAINING_SOLVER_DIR);
    [
        output_dir.join(CUT_SELECTION_FILE),
        solver_dir.join(SOLVER_ITERATIONS_FILE),
        solver_dir.join(SOLVER_RETRY_HISTOGRAM_FILE),
        output_dir.join(FIXED_DELIVERIES_FILE),
        output_dir.join(FPHA_HYPERPLANES_FILE),
        output_dir.join(EVAPORATION_MODELS_FILE),
        output_dir.join(FPHA_DEVIATION_POINTS_FILE),
        output_dir.join(GENERIC_CONSTRAINT_ECHO_FILE),
    ]
}

/// Remove the training outputs that a run writes only when it has rows for
/// them from under `output_dir`, together with each of their directories this
/// leaves empty. Callers invoke it before the training phase's first write,
/// after removing the stale training `_SUCCESS` marker.
///
/// # Errors
///
/// Returns [`OutputError::IoError`] for the first entry that exists but cannot be
/// removed, such as a directory at a file path or a regular file where a parent
/// directory belongs: the entry is not one this writer wrote.
pub fn remove_conditional_training_outputs(output_dir: &Path) -> Result<(), OutputError> {
    let files = conditional_training_output_files(output_dir);
    for file in &files {
        remove_file_if_present(file)?;
    }
    for dir in files.iter().filter_map(|file| file.parent()) {
        remove_dir_if_empty(dir)?;
    }
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
    use crate::MetadataTrainingSolveStats;
    use crate::output::{RowPoolStatistics, RunStatus, TrainingOutput};
    use crate::test_support::output::read_first_batch;

    fn make_record(iteration: u32, gap: Option<f64>) -> IterationRecord {
        IterationRecord {
            iteration,
            lower_bound: f64::from(iteration) * 10.0,
            upper_bound: f64::from(iteration) * 11.0,
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
            time_bwd_load_imbalance_ms: 0,
            time_bwd_scheduling_overhead_ms: 0,
            time_fwd_load_imbalance_ms: 0,
            time_fwd_scheduling_overhead_ms: 0,
            time_overhead_ms: 0,
            forward_passes: 4,
            lp_solves: 40,
            solve_time_ms: 0.0,
            mean_rows_in_lp: 0.0,
        }
    }

    fn make_training_output(records: Vec<IterationRecord>) -> TrainingOutput {
        TrainingOutput {
            convergence_records: records,
            final_lower_bound: 99.5,
            final_upper_bound: Some(101.0),
            final_gap_percent: Some(1.51),
            final_upper_bound_std: Some(0.5),
            final_upper_bound_kind: "statistical".to_string(),
            iterations_completed: 0,
            converged: true,
            termination_reason: "gap tolerance reached".to_string(),
            status: RunStatus::Complete,
            total_time_ms: 5_000,
            cut_stats: RowPoolStatistics {
                total_generated: 200,
                total_active: 80,
                peak_active: 95,
                cuts_active: 80,
                rows_in_lp_total: 0,
                rows_in_lp_solve_count: 0,
                rows_in_lp_max: 0,
                total_loaded: 0,
            },
            cut_selection_records: vec![],
            worker_timing_records: vec![],
            training_solve_stats: MetadataTrainingSolveStats::default(),
        }
    }

    fn make_worker_timing_record(iteration: u32) -> WorkerTimingRecord {
        WorkerTimingRecord {
            iteration,
            rank: 0,
            worker_id: None,
            timings: [100, 200, 10, 5, 8, 3, 4, 7, 50, 60, 20, 40, 30, 15, 12, 0],
        }
    }

    // -------------------------------------------------------------------------
    // build_convergence_batch tests
    // -------------------------------------------------------------------------

    #[test]
    fn convergence_batch_from_empty_records() {
        let batch = build_convergence_batch(&[], "statistical").expect("empty batch must succeed");
        assert_eq!(batch.num_rows(), 0, "empty records yield 0 rows");
        assert_eq!(batch.num_columns(), 15, "convergence schema has 15 columns");
    }

    #[test]
    fn convergence_batch_field_count_and_types() {
        let records: Vec<IterationRecord> = (1..=3).map(|i| make_record(i, Some(5.0))).collect();
        let batch = build_convergence_batch(&records, "statistical").expect("batch must be built");
        assert_eq!(batch.num_rows(), 3);
        assert_eq!(batch.num_columns(), 15);

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
            make_record(3, None),
        ];
        let batch = build_convergence_batch(&records, "statistical").expect("batch must be built");

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
        let records: Vec<WorkerTimingRecord> = (1..=3).map(make_worker_timing_record).collect();
        let batch = build_iteration_timing_batch(&records).expect("timing batch must be built");
        assert_eq!(batch.num_rows(), 3, "3 records yield 3 rows");
        assert_eq!(
            batch.num_columns(),
            19,
            "iteration_timing schema has 19 columns (16 timings + iteration + rank + worker_id)"
        );

        let expected_schema = iteration_timing_schema();
        assert_eq!(
            batch.schema().fields(),
            expected_schema.fields(),
            "schema must match iteration_timing_schema()"
        );
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn iteration_timing_columns_six_decomposed_overhead() {
        use arrow::array::Int64Array;

        let tmp = tempfile::tempdir().expect("tempdir must succeed");
        std::fs::create_dir_all(tmp.path().join("training/timing")).unwrap();

        // Build 3 records with distinct non-zero overhead component values so we
        // can verify each column carries the right field.
        let records: Vec<IterationRecord> = (1u32..=3)
            .map(|i| IterationRecord {
                iteration: i,
                lower_bound: f64::from(i) * 10.0,
                upper_bound: f64::from(i) * 11.0,
                upper_bound_std: 0.5,
                gap_percent: Some(5.0),
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
                time_bwd_load_imbalance_ms: u64::from(i) * 20,
                time_bwd_scheduling_overhead_ms: u64::from(i) * 30,
                time_fwd_load_imbalance_ms: u64::from(i) * 50,
                time_fwd_scheduling_overhead_ms: u64::from(i) * 60,
                time_overhead_ms: 0,
                forward_passes: 4,
                lp_solves: 40,
                solve_time_ms: 0.0,
                mean_rows_in_lp: 0.0,
            })
            .collect();

        // Build matching WorkerTimingRecord rank-aggregated rows directly so the
        // writer has data to emit (the timing parquet reads from
        // worker_timing_records, not convergence_records).
        let worker_records: Vec<WorkerTimingRecord> = (1u32..=3)
            .map(|i| WorkerTimingRecord {
                iteration: i,
                rank: 0,
                worker_id: None,
                timings: [
                    100,
                    200,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    u64::from(i) * 10,
                    u64::from(i) * 20,
                    u64::from(i) * 30,
                    u64::from(i) * 40,
                    u64::from(i) * 50,
                    u64::from(i) * 60,
                    0,
                    0,
                ],
            })
            .collect();
        let mut training = make_training_output(records);
        training.worker_timing_records = worker_records;
        let writer = TrainingParquetWriter::new(tmp.path()).expect("new must succeed");
        writer.write(&training).expect("write must succeed");

        let timing_path = tmp.path().join("training/timing/iterations.parquet");
        assert!(timing_path.exists(), "iterations.parquet must exist");

        let batch = read_first_batch(&timing_path);

        assert_eq!(batch.num_rows(), 3);
        assert_eq!(batch.num_columns(), 19);

        // Old column names should not be present.
        assert!(batch.column_by_name("bwd_rayon_overhead_ms").is_none());
        assert!(batch.column_by_name("fwd_rayon_overhead_ms").is_none());

        let expected_schema = iteration_timing_schema();
        assert_eq!(
            batch.schema().fields(),
            expected_schema.fields(),
            "schema must match iteration_timing_schema()"
        );

        for (col_name, expected_row1_val) in &[
            ("bwd_setup_ms", 10_i64),
            ("bwd_load_imbalance_ms", 20_i64),
            ("bwd_scheduling_overhead_ms", 30_i64),
            ("fwd_setup_ms", 40_i64),
            ("fwd_load_imbalance_ms", 50_i64),
            ("fwd_scheduling_overhead_ms", 60_i64),
        ] {
            let col = batch
                .column_by_name(col_name)
                .unwrap_or_else(|| panic!("column {col_name} must exist"));
            let arr = col
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap_or_else(|| panic!("{col_name} must be Int64Array"));
            // Row 0 corresponds to iteration=1, multiplier=1.
            assert_eq!(
                arr.value(0),
                *expected_row1_val,
                "{col_name} row 0 must be {expected_row1_val}"
            );
            // Row 1: iteration=2, multiplier=2.
            assert_eq!(
                arr.value(1),
                expected_row1_val * 2,
                "{col_name} row 1 must be {}",
                expected_row1_val * 2
            );
        }
    }

    // -------------------------------------------------------------------------
    // write_parquet + roundtrip tests
    // -------------------------------------------------------------------------

    #[test]
    fn write_convergence_parquet_roundtrip() {
        let records: Vec<IterationRecord> = (1..=5).map(|i| make_record(i, Some(1.0))).collect();
        let batch = build_convergence_batch(&records, "statistical").expect("batch must be built");

        let tmp = tempfile::tempdir().expect("tempdir must succeed");
        let path = tmp.path().join("convergence.parquet");

        write_parquet_atomic(&path, &batch).expect("write must succeed");
        assert!(path.exists(), "convergence.parquet must exist after write");

        let read_batch = read_first_batch(&path);
        assert_eq!(read_batch.num_rows(), 5, "must have 5 rows");

        let expected_schema = convergence_schema();
        assert_eq!(
            read_batch.schema().fields(),
            expected_schema.fields(),
            "schema must match convergence_schema()"
        );

        let iteration_col = read_batch
            .column_by_name("iteration")
            .expect("iteration column must exist");
        let iteration_arr = iteration_col
            .as_any()
            .downcast_ref::<arrow::array::Int32Array>()
            .expect("iteration must be Int32Array");
        let iteration_values: Vec<i32> = (0..5).map(|i| iteration_arr.value(i)).collect();
        assert_eq!(iteration_values, vec![1, 2, 3, 4, 5]);

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
        let batch = build_convergence_batch(&records, "statistical").expect("batch must be built");

        let tmp = tempfile::tempdir().expect("tempdir must succeed");
        let path = tmp.path().join("convergence.parquet");

        write_parquet_atomic(&path, &batch).expect("write must succeed");

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

        // Do not create the training/ directory.
        let result = TrainingParquetWriter::new(tmp.path());
        assert!(result.is_err(), "new() must fail when training/ is missing");
    }

    #[test]
    fn writer_fails_if_timing_dir_missing() {
        let tmp = tempfile::tempdir().expect("tempdir must succeed");

        // Create training/ but not training/timing/.
        std::fs::create_dir_all(tmp.path().join("training")).unwrap();

        let result = TrainingParquetWriter::new(tmp.path());
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

        let writer = TrainingParquetWriter::new(tmp.path()).expect("new must succeed");
        let training = make_training_output(vec![]);
        writer.write(&training).expect("write must succeed");

        let conv_path = tmp.path().join("training/convergence.parquet");
        assert!(conv_path.exists(), "convergence.parquet must exist");

        let timing_path = tmp.path().join("training/timing/iterations.parquet");
        assert!(timing_path.exists(), "iterations.parquet must exist");

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
        let tmp = tempfile::tempdir().expect("tempdir must succeed");
        std::fs::create_dir_all(tmp.path().join("training/timing")).unwrap();

        let records: Vec<IterationRecord> = (1..=5).map(|i| make_record(i, Some(1.0))).collect();
        let mut training = make_training_output(records);
        // The timing parquet reads from worker_timing_records, not
        // convergence_records. Add 5 rank-aggregated rows for parity with the
        // convergence rows.
        training.worker_timing_records = (1u32..=5).map(make_worker_timing_record).collect();

        let writer = TrainingParquetWriter::new(tmp.path()).expect("new must succeed");
        writer.write(&training).expect("write must succeed");

        let conv_path = tmp.path().join("training/convergence.parquet");
        let batch = read_first_batch(&conv_path);
        assert_eq!(batch.num_rows(), 5);
        assert_eq!(batch.num_columns(), 15);

        let timing_path = tmp.path().join("training/timing/iterations.parquet");
        let batch = read_first_batch(&timing_path);
        assert_eq!(batch.num_rows(), 5);
        assert_eq!(batch.num_columns(), 19, "timing schema has 19 columns");
    }

    #[test]
    fn writer_gap_percent_null_at_correct_row() {
        let tmp = tempfile::tempdir().expect("tempdir must succeed");
        std::fs::create_dir_all(tmp.path().join("training/timing")).unwrap();

        let records = vec![
            make_record(1, Some(10.0)),
            make_record(2, Some(5.0)),
            make_record(3, None), // record 3 (row index 2): gap_percent = None
            make_record(4, Some(2.0)),
            make_record(5, Some(1.0)),
        ];
        let training = make_training_output(records);

        let writer = TrainingParquetWriter::new(tmp.path()).expect("new must succeed");
        writer.write(&training).expect("write must succeed");

        let conv_path = tmp.path().join("training/convergence.parquet");
        let batch = read_first_batch(&conv_path);

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
    // write_row_selection_records tests
    // -------------------------------------------------------------------------

    #[test]
    fn write_cut_selection_empty_is_noop() {
        let tmp = tempfile::tempdir().unwrap();
        write_row_selection_records(tmp.path(), &[]).unwrap();
        assert!(
            !tmp.path()
                .join("training/cut_selection/iterations.parquet")
                .exists()
        );
    }

    #[test]
    fn write_cut_selection_roundtrip() {
        use super::super::RowSelectionRecord;

        let tmp = tempfile::tempdir().unwrap();
        let records = vec![
            RowSelectionRecord {
                iteration: 3,
                stage: 0,
                cuts_populated: 10,
                cuts_active_before: 10,
                cuts_deactivated: 0,
                cuts_reactivated: 0,
                cuts_active_after: 10,
                selection_time_ms: 0.0,
                budget_evicted: None,
                active_after_budget: None,
            },
            RowSelectionRecord {
                iteration: 3,
                stage: 1,
                cuts_populated: 8,
                cuts_active_before: 8,
                cuts_deactivated: 2,
                cuts_reactivated: 0,
                cuts_active_after: 6,
                selection_time_ms: 1.5,
                budget_evicted: None,
                active_after_budget: None,
            },
        ];
        write_row_selection_records(tmp.path(), &records).unwrap();
        let path = tmp.path().join("training/cut_selection/iterations.parquet");
        assert!(path.exists());

        let batch = read_first_batch(&path);
        assert_eq!(batch.num_rows(), 2);
        assert_eq!(batch.num_columns(), 10);
    }

    #[test]
    fn write_cut_selection_with_budget_columns_roundtrip() {
        use super::super::RowSelectionRecord;

        let tmp = tempfile::tempdir().unwrap();
        let records = vec![
            // Record with all budget columns populated (budget enabled).
            RowSelectionRecord {
                iteration: 5,
                stage: 0,
                cuts_populated: 20,
                cuts_active_before: 20,
                cuts_deactivated: 0,
                cuts_reactivated: 0,
                cuts_active_after: 20,
                selection_time_ms: 0.0,
                budget_evicted: Some(3),
                active_after_budget: Some(15),
            },
            // Record with all budget columns None (budget disabled).
            RowSelectionRecord {
                iteration: 5,
                stage: 1,
                cuts_populated: 15,
                cuts_active_before: 15,
                cuts_deactivated: 2,
                cuts_reactivated: 1,
                cuts_active_after: 13,
                selection_time_ms: 2.0,
                budget_evicted: None,
                active_after_budget: None,
            },
        ];
        write_row_selection_records(tmp.path(), &records).unwrap();
        let path = tmp.path().join("training/cut_selection/iterations.parquet");
        assert!(path.exists());

        let batch = read_first_batch(&path);
        assert_eq!(batch.num_rows(), 2);
        assert_eq!(batch.num_columns(), 10);

        let budget_evicted_col = batch.column_by_name("budget_evicted").unwrap();
        assert!(
            !budget_evicted_col.is_null(0),
            "row 0: budget_evicted Some(3) must not be null"
        );
        assert!(
            budget_evicted_col.is_null(1),
            "row 1: budget_evicted None must be null"
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

        let budget_evicted_arr = budget_evicted_col
            .as_any()
            .downcast_ref::<arrow::array::Int32Array>()
            .unwrap();
        assert_eq!(budget_evicted_arr.value(0), 3);

        let budget_arr = budget_col
            .as_any()
            .downcast_ref::<arrow::array::Int32Array>()
            .unwrap();
        assert_eq!(budget_arr.value(0), 15);
    }

    #[test]
    fn parquet_schema_includes_cuts_active_column() {
        use arrow::datatypes::DataType;
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

        use super::super::RowSelectionRecord;

        let tmp = tempfile::tempdir().unwrap();
        let records = vec![RowSelectionRecord {
            iteration: 1,
            stage: 0,
            cuts_populated: 5,
            cuts_active_before: 5,
            cuts_deactivated: 0,
            cuts_reactivated: 0,
            cuts_active_after: 5,
            selection_time_ms: 0.0,
            budget_evicted: None,
            active_after_budget: None,
        }];
        write_row_selection_records(tmp.path(), &records).unwrap();
        let path = tmp.path().join("training/cut_selection/iterations.parquet");

        let file = std::fs::File::open(&path).unwrap();
        let builder = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
        let schema = builder.schema().clone();

        let cuts_reactivated = schema
            .field_with_name("cuts_reactivated")
            .expect("cuts_reactivated column must exist in schema");
        assert_eq!(
            cuts_reactivated.data_type(),
            &DataType::Int32,
            "cuts_reactivated must be Int32"
        );
        assert!(
            !cuts_reactivated.is_nullable(),
            "cuts_reactivated must not be nullable"
        );

        assert!(
            schema.field_with_name("cuts_in_lp").is_err(),
            "cuts_in_lp column must not be present in schema"
        );
    }

    // -------------------------------------------------------------------------
    // remove_conditional_training_outputs tests
    // -------------------------------------------------------------------------

    const CONDITIONAL_TRAINING_OUTPUTS: [&str; 8] = [
        "training/cut_selection/iterations.parquet",
        "training/solver/iterations.parquet",
        "training/solver/retry_histogram.parquet",
        "anticipated/fixed_deliveries.parquet",
        "hydro_models/fpha_hyperplanes.parquet",
        "hydro_models/evaporation_models.parquet",
        "hydro_models/fpha_deviation_points.parquet",
        "generic_constraints/resolved_echo.parquet",
    ];

    fn seed_stale_file(root: &Path, relative: &str) {
        let path = root.join(relative);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, b"stale").unwrap();
    }

    #[test]
    fn conditional_training_output_remover_clears_each_output_and_its_empty_directory() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let kept_files = [
            "hydro_models/notes.txt",
            "hydro_models/fpha_hyperplanes.parquet.tmp",
            "training/_SUCCESS",
            "training/metadata.json",
            "policy/manifest.json",
            "simulation/solver/iterations.parquet",
        ];
        for relative in CONDITIONAL_TRAINING_OUTPUTS.into_iter().chain(kept_files) {
            seed_stale_file(root, relative);
        }
        let leftover_tmp = root.join("training/solver/retry_histogram.parquet.tmp");
        std::fs::create_dir_all(&leftover_tmp).unwrap();

        remove_conditional_training_outputs(root).unwrap();

        for relative in CONDITIONAL_TRAINING_OUTPUTS.into_iter().chain([
            "training/cut_selection",
            "anticipated",
            "generic_constraints",
        ]) {
            assert!(!root.join(relative).exists(), "{relative} must be removed");
        }
        for relative in kept_files {
            assert!(root.join(relative).is_file(), "{relative} must be kept");
        }
        assert!(leftover_tmp.is_dir(), "a .tmp leftover must be kept");
        assert!(root.join("hydro_models").is_dir());
        assert!(root.join("training/solver").is_dir());
    }

    #[test]
    fn conditional_training_output_remover_clears_what_the_writers_wrote() {
        use chrono::NaiveDate;

        use super::super::RowSelectionRecord;
        use crate::output::{
            FixedDeliveryRow, write_evaporation_models, write_fixed_delivery,
            write_fpha_deviation_points, write_fpha_hyperplanes, write_generic_constraint_echo,
            write_solver_stats,
        };

        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let record = RowSelectionRecord {
            iteration: 3,
            stage: 0,
            cuts_populated: 10,
            cuts_active_before: 10,
            cuts_deactivated: 0,
            cuts_reactivated: 0,
            cuts_active_after: 10,
            selection_time_ms: 0.0,
            budget_evicted: None,
            active_after_budget: None,
        };
        write_row_selection_records(root, &[record]).unwrap();
        write_solver_stats(root, &[]).unwrap();
        let delivery = FixedDeliveryRow {
            thermal_id: 3,
            start_date: NaiveDate::from_ymd_opt(2030, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2030, 6, 30).unwrap(),
            value_mw: 120.5,
        };
        write_fixed_delivery(root, &[delivery]).unwrap();
        write_fpha_hyperplanes(&root.join(FPHA_HYPERPLANES_FILE), &[]).unwrap();
        write_evaporation_models(&root.join(EVAPORATION_MODELS_FILE), &[]).unwrap();
        write_fpha_deviation_points(&root.join(FPHA_DEVIATION_POINTS_FILE), &[]).unwrap();
        write_generic_constraint_echo(&root.join(GENERIC_CONSTRAINT_ECHO_FILE), &[]).unwrap();
        for relative in CONDITIONAL_TRAINING_OUTPUTS {
            assert!(root.join(relative).is_file(), "{relative} must be written");
        }

        remove_conditional_training_outputs(root).unwrap();

        let entries: Vec<String> = std::fs::read_dir(root)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().into_string().unwrap())
            .collect();
        assert_eq!(entries, ["training"]);
        assert_eq!(std::fs::read_dir(root.join("training")).unwrap().count(), 0);
    }

    #[test]
    fn conditional_training_output_remover_accepts_missing_outputs() {
        let tmp = tempfile::tempdir().unwrap();
        remove_conditional_training_outputs(tmp.path()).unwrap();
        remove_conditional_training_outputs(&tmp.path().join("absent")).unwrap();
    }

    #[test]
    fn conditional_training_output_remover_reports_an_entry_it_cannot_remove() {
        let file_at_parent = tempfile::tempdir().unwrap();
        seed_stale_file(file_at_parent.path(), "training/solver");
        let dir_at_file = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(
            dir_at_file
                .path()
                .join("anticipated/fixed_deliveries.parquet"),
        )
        .unwrap();

        for (root, blocker, reported) in [
            (
                file_at_parent.path(),
                "training/solver",
                "solver/iterations.parquet",
            ),
            (
                dir_at_file.path(),
                "anticipated/fixed_deliveries.parquet",
                "fixed_deliveries.parquet",
            ),
        ] {
            let err = remove_conditional_training_outputs(root).unwrap_err();
            assert!(
                matches!(&err, OutputError::IoError { path, .. } if path.ends_with(reported)),
                "expected an IoError naming {reported}, got {err:?}"
            );
            assert!(root.join(blocker).exists(), "{blocker} must be kept");
        }
    }

    #[test]
    fn conditional_training_outputs_are_registered_training_files() {
        use crate::output::file_registry::{FileLayout, OUTPUT_FILES, WritePhase};

        for file in conditional_training_output_files(Path::new("")) {
            let path = file.to_str().unwrap();
            assert!(
                OUTPUT_FILES.iter().any(|entry| entry.path == path
                    && entry.layout == FileLayout::File
                    && entry.phase == WritePhase::Training),
                "{path} must be a registered training-phase file"
            );
        }
    }
}
