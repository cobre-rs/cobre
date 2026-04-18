//! Parquet writer for per-iteration solver statistics.
//!
//! Writes `training/solver/iterations.parquet` (scalar metrics) and
//! `training/solver/retry_histogram.parquet` (normalized per-level retry counts).

use std::path::Path;
use std::sync::Arc;

use arrow::array::{Float64Array, Int32Array, StringBuilder, UInt32Array, UInt64Array};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::properties::WriterProperties;

use super::error::OutputError;
use super::schemas::{retry_histogram_schema, solver_iterations_schema};

/// A single row in the solver statistics Parquet file.
#[derive(Debug, Clone)]
pub struct SolverStatsRow {
    /// Row identifier: iteration number (1-based) for training phases,
    /// scenario ID (0-based) for the simulation phase.
    pub iteration: u32,
    /// Phase name: `"forward"`, `"backward"`, `"lower_bound"`, or `"simulation"`.
    pub phase: String,
    /// Stage index for backward phase, `-1` for forward/LB.
    pub stage: i32,
    /// Number of LP solves in this phase.
    pub lp_solves: u32,
    /// Solves that returned optimal.
    pub lp_successes: u32,
    /// Solves that required retry escalation.
    pub lp_retries: u32,
    /// Solves that exhausted all retry levels.
    pub lp_failures: u32,
    /// Total retry attempts across all retried solves.
    pub retry_attempts: u32,
    /// Number of `solve_with_basis` calls.
    pub basis_offered: u32,
    /// Times the basis was rejected.
    pub basis_rejections: u32,
    /// Times `solve_with_basis` fell back from the non-alien path to the alien path
    /// because `HiGHS` rejected the non-alien basis (`isBasisConsistent` failed).
    pub basis_non_alien_rejections: u32,
    /// Total `clear_solver_state` calls across all solvers in this phase.
    ///
    /// Non-zero under `CanonicalStateStrategy::ClearSolver`; zero under `Disabled`.
    pub clear_solver_count: u64,
    /// `clear_solver_state` calls that returned an FFI error in this phase.
    ///
    /// Should be zero in a healthy `HiGHS` build.
    pub clear_solver_failures: u64,
    /// Total simplex iterations.
    pub simplex_iterations: u64,
    /// Cumulative solve time in milliseconds.
    pub solve_time_ms: f64,
    /// Cumulative time in `load_model` calls, in milliseconds.
    pub load_model_time_ms: f64,
    /// Cumulative time in `add_rows` calls, in milliseconds.
    pub add_rows_time_ms: f64,
    /// Cumulative time in `set_row_bounds`/`set_col_bounds` calls, in milliseconds.
    pub set_bounds_time_ms: f64,
    /// Cumulative time in `set_basis` FFI calls, in milliseconds.
    pub basis_set_time_ms: f64,
    /// Number of cut rows whose status was preserved from a stored basis via slot
    /// reconciliation during reconstruction.
    pub basis_preserved: u64,
    /// Number of newly-added cut rows assigned `NONBASIC_LOWER` after evaluation
    /// at the padding state.
    pub basis_new_tight: u64,
    /// Number of newly-added cut rows assigned `BASIC` after evaluation at the
    /// padding state.
    pub basis_new_slack: u64,
    /// Number of BASIC row statuses demoted to LOWER by
    /// `enforce_basic_count_invariant` on the forward path (ticket-009).
    /// Zero on backward and simulation paths.
    pub basis_demotions: u64,
    /// Per-level retry success counts. Length depends on the solver backend
    /// (e.g. 12 for `HiGHS`).
    pub retry_level_histogram: Vec<u64>,
}

/// Write training solver statistics to `training/solver/iterations.parquet`.
///
/// Creates the `training/solver/` directory if it does not exist. Uses atomic
/// write (`.tmp` + rename).
///
/// # Errors
///
/// Returns [`OutputError`] on filesystem or serialization failures.
pub fn write_solver_stats(output_dir: &Path, rows: &[SolverStatsRow]) -> Result<(), OutputError> {
    write_solver_stats_to(&output_dir.join("training/solver"), rows)
}

/// Write simulation solver statistics to `simulation/solver/iterations.parquet`.
///
/// Creates the `simulation/solver/` directory if it does not exist. Uses atomic
/// write (`.tmp` + rename).
///
/// # Errors
///
/// Returns [`OutputError`] on filesystem or serialization failures.
pub fn write_simulation_solver_stats(
    output_dir: &Path,
    rows: &[SolverStatsRow],
) -> Result<(), OutputError> {
    write_solver_stats_to(&output_dir.join("simulation/solver"), rows)
}

/// Build Arrow column arrays for `iterations.parquet` (scalar metrics only).
fn build_iterations_columns(rows: &[SolverStatsRow]) -> Vec<Arc<dyn arrow::array::Array>> {
    let n = rows.len();
    let iteration_arr = UInt32Array::from(rows.iter().map(|r| r.iteration).collect::<Vec<_>>());
    let mut phase_builder = StringBuilder::with_capacity(n, n * 10);
    for r in rows {
        phase_builder.append_value(&r.phase);
    }
    let phase_arr = phase_builder.finish();
    let stage_arr = Int32Array::from(rows.iter().map(|r| r.stage).collect::<Vec<_>>());
    let lp_solves_arr = UInt32Array::from(rows.iter().map(|r| r.lp_solves).collect::<Vec<_>>());
    let lp_successes_arr =
        UInt32Array::from(rows.iter().map(|r| r.lp_successes).collect::<Vec<_>>());
    let lp_retries_arr = UInt32Array::from(rows.iter().map(|r| r.lp_retries).collect::<Vec<_>>());
    let lp_failures_arr = UInt32Array::from(rows.iter().map(|r| r.lp_failures).collect::<Vec<_>>());
    let retry_attempts_arr =
        UInt32Array::from(rows.iter().map(|r| r.retry_attempts).collect::<Vec<_>>());
    let basis_offered_arr =
        UInt32Array::from(rows.iter().map(|r| r.basis_offered).collect::<Vec<_>>());
    let basis_rejections_arr =
        UInt32Array::from(rows.iter().map(|r| r.basis_rejections).collect::<Vec<_>>());
    let basis_non_alien_rejections_arr = UInt32Array::from(
        rows.iter()
            .map(|r| r.basis_non_alien_rejections)
            .collect::<Vec<_>>(),
    );
    let clear_solver_count_arr = UInt64Array::from(
        rows.iter()
            .map(|r| r.clear_solver_count)
            .collect::<Vec<_>>(),
    );
    let clear_solver_failures_arr = UInt64Array::from(
        rows.iter()
            .map(|r| r.clear_solver_failures)
            .collect::<Vec<_>>(),
    );
    let simplex_iter_arr = UInt64Array::from(
        rows.iter()
            .map(|r| r.simplex_iterations)
            .collect::<Vec<_>>(),
    );
    let solve_time_arr =
        Float64Array::from(rows.iter().map(|r| r.solve_time_ms).collect::<Vec<_>>());
    let load_model_time_arr = Float64Array::from(
        rows.iter()
            .map(|r| r.load_model_time_ms)
            .collect::<Vec<_>>(),
    );
    let add_rows_time_arr =
        Float64Array::from(rows.iter().map(|r| r.add_rows_time_ms).collect::<Vec<_>>());
    let set_bounds_time_arr = Float64Array::from(
        rows.iter()
            .map(|r| r.set_bounds_time_ms)
            .collect::<Vec<_>>(),
    );
    let basis_set_time_arr =
        Float64Array::from(rows.iter().map(|r| r.basis_set_time_ms).collect::<Vec<_>>());
    let basis_preserved_arr =
        UInt64Array::from(rows.iter().map(|r| r.basis_preserved).collect::<Vec<_>>());
    let basis_new_tight_arr =
        UInt64Array::from(rows.iter().map(|r| r.basis_new_tight).collect::<Vec<_>>());
    let basis_new_slack_arr =
        UInt64Array::from(rows.iter().map(|r| r.basis_new_slack).collect::<Vec<_>>());
    let basis_demotions_arr =
        UInt64Array::from(rows.iter().map(|r| r.basis_demotions).collect::<Vec<_>>());

    vec![
        Arc::new(iteration_arr),
        Arc::new(phase_arr),
        Arc::new(stage_arr),
        Arc::new(lp_solves_arr),
        Arc::new(lp_successes_arr),
        Arc::new(lp_retries_arr),
        Arc::new(lp_failures_arr),
        Arc::new(retry_attempts_arr),
        Arc::new(basis_offered_arr),
        Arc::new(basis_rejections_arr),
        Arc::new(basis_non_alien_rejections_arr),
        Arc::new(clear_solver_count_arr),
        Arc::new(clear_solver_failures_arr),
        Arc::new(simplex_iter_arr),
        Arc::new(solve_time_arr),
        Arc::new(load_model_time_arr),
        Arc::new(add_rows_time_arr),
        Arc::new(set_bounds_time_arr),
        Arc::new(basis_set_time_arr),
        Arc::new(basis_preserved_arr),
        Arc::new(basis_new_tight_arr),
        Arc::new(basis_new_slack_arr),
        Arc::new(basis_demotions_arr),
    ]
}

/// Build a `RecordBatch` for `retry_histogram.parquet` from the histogram data
/// embedded in each `SolverStatsRow`. Only rows with `count > 0` are emitted.
fn build_retry_histogram_batch(rows: &[SolverStatsRow]) -> Result<RecordBatch, OutputError> {
    let mut iterations = Vec::new();
    let mut phases = Vec::new();
    let mut stages = Vec::new();
    let mut levels = Vec::new();
    let mut counts = Vec::new();

    for r in rows {
        #[allow(clippy::cast_possible_truncation)]
        for (level, &count) in r.retry_level_histogram.iter().enumerate() {
            if count > 0 {
                iterations.push(r.iteration);
                phases.push(r.phase.as_str());
                stages.push(r.stage);
                levels.push(level as u32);
                counts.push(count);
            }
        }
    }

    let n = iterations.len();
    let mut phase_builder = StringBuilder::with_capacity(n, n * 10);
    for &p in &phases {
        phase_builder.append_value(p);
    }

    let schema = Arc::new(retry_histogram_schema());
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(UInt32Array::from(iterations)),
            Arc::new(phase_builder.finish()),
            Arc::new(Int32Array::from(stages)),
            Arc::new(UInt32Array::from(levels)),
            Arc::new(UInt64Array::from(counts)),
        ],
    )
    .map_err(|e| OutputError::serialization("retry_histogram", format!("RecordBatch: {e}")))
}

/// Write a `RecordBatch` to a Parquet file using atomic `.tmp` + rename.
fn write_parquet(
    path: &Path,
    schema: &Arc<arrow::datatypes::Schema>,
    batch: &RecordBatch,
) -> Result<(), OutputError> {
    let tmp_path = path.with_extension("parquet.tmp");
    let file = std::fs::File::create(&tmp_path).map_err(|e| OutputError::io(&tmp_path, e))?;
    let props = WriterProperties::builder()
        .set_compression(Compression::ZSTD(ZstdLevel::default()))
        .build();
    let mut writer = ArrowWriter::try_new(file, Arc::clone(schema), Some(props))
        .map_err(|e| OutputError::serialization("solver_stats", format!("ArrowWriter: {e}")))?;
    writer
        .write(batch)
        .map_err(|e| OutputError::serialization("solver_stats", format!("write: {e}")))?;
    writer
        .close()
        .map_err(|e| OutputError::serialization("solver_stats", format!("close: {e}")))?;
    std::fs::rename(&tmp_path, path).map_err(|e| OutputError::io(path, e))?;
    Ok(())
}

/// Internal: write solver statistics to `{dir}/iterations.parquet` and
/// `{dir}/retry_histogram.parquet`.
fn write_solver_stats_to(dir: &Path, rows: &[SolverStatsRow]) -> Result<(), OutputError> {
    std::fs::create_dir_all(dir).map_err(|e| OutputError::io(dir, e))?;

    // iterations.parquet — scalar metrics
    let iter_schema = Arc::new(solver_iterations_schema());
    let columns = build_iterations_columns(rows);
    let iter_batch = RecordBatch::try_new(Arc::clone(&iter_schema), columns)
        .map_err(|e| OutputError::serialization("solver_stats", format!("RecordBatch: {e}")))?;
    write_parquet(&dir.join("iterations.parquet"), &iter_schema, &iter_batch)?;

    // retry_histogram.parquet — normalized per-level retry counts
    let hist_schema = Arc::new(retry_histogram_schema());
    let hist_batch = build_retry_histogram_batch(rows)?;
    write_parquet(
        &dir.join("retry_histogram.parquet"),
        &hist_schema,
        &hist_batch,
    )?;

    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::float_cmp)]
mod tests {
    use super::*;
    use arrow::array::{Float64Array, UInt32Array, UInt64Array};
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

    fn make_rows() -> Vec<SolverStatsRow> {
        vec![
            SolverStatsRow {
                iteration: 1,
                phase: "forward".to_string(),
                stage: -1,
                lp_solves: 100,
                lp_successes: 98,
                lp_retries: 2,
                lp_failures: 0,
                retry_attempts: 4,
                basis_offered: 90,
                basis_rejections: 3,
                basis_non_alien_rejections: 0,
                clear_solver_count: 0,
                clear_solver_failures: 0,
                simplex_iterations: 5000,
                solve_time_ms: 42.5,
                load_model_time_ms: 0.0,
                add_rows_time_ms: 0.0,
                set_bounds_time_ms: 0.0,
                basis_set_time_ms: 0.0,
                basis_preserved: 0,
                basis_new_tight: 0,
                basis_new_slack: 0,
                basis_demotions: 0,
                retry_level_histogram: vec![0; 12],
            },
            SolverStatsRow {
                iteration: 1,
                phase: "backward".to_string(),
                stage: 2,
                lp_solves: 200,
                lp_successes: 200,
                lp_retries: 0,
                lp_failures: 0,
                retry_attempts: 0,
                basis_offered: 180,
                basis_rejections: 1,
                basis_non_alien_rejections: 0,
                clear_solver_count: 0,
                clear_solver_failures: 0,
                simplex_iterations: 10000,
                solve_time_ms: 85.0,
                load_model_time_ms: 0.0,
                add_rows_time_ms: 0.0,
                set_bounds_time_ms: 0.0,
                basis_set_time_ms: 0.0,
                basis_preserved: 0,
                basis_new_tight: 0,
                basis_new_slack: 0,
                basis_demotions: 0,
                retry_level_histogram: vec![0; 12],
            },
        ]
    }

    fn read_parquet(path: &std::path::Path) -> RecordBatch {
        let file = std::fs::File::open(path).unwrap();
        let builder = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
        let mut reader = builder.build().unwrap();
        reader.next().unwrap().unwrap()
    }

    #[test]
    fn write_and_read_back() {
        let dir = tempfile::TempDir::new().unwrap();
        let rows = make_rows();

        write_solver_stats(dir.path(), &rows).unwrap();

        // iterations.parquet — 23 scalar columns
        let iter_path = dir.path().join("training/solver/iterations.parquet");
        assert!(iter_path.exists());
        let batch = read_parquet(&iter_path);

        assert_eq!(batch.num_rows(), 2);
        assert_eq!(batch.num_columns(), 23);

        let iteration_col = batch
            .column(0)
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap();
        assert_eq!(iteration_col.value(0), 1);
        assert_eq!(iteration_col.value(1), 1);

        // Column indices:
        // 9 = basis_rejections, 10 = basis_non_alien_rejections,
        // 11 = clear_solver_count, 12 = clear_solver_failures,
        // 13 = simplex_iterations, 14 = solve_time_ms
        let solve_time_col = batch
            .column(14)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        assert!((solve_time_col.value(0) - 42.5).abs() < 1e-10);
        assert!((solve_time_col.value(1) - 85.0).abs() < 1e-10);

        let simplex_col = batch
            .column(13)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        assert_eq!(simplex_col.value(0), 5000);

        // retry_histogram.parquet — empty (make_rows has all-zero histograms)
        let hist_path = dir.path().join("training/solver/retry_histogram.parquet");
        assert!(hist_path.exists());
        let file = std::fs::File::open(&hist_path).unwrap();
        let builder = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
        assert_eq!(builder.schema().fields().len(), 5);
        let total_rows: usize = builder
            .build()
            .unwrap()
            .flatten()
            .map(|b| b.num_rows())
            .sum();
        assert_eq!(total_rows, 0);
    }

    #[test]
    fn write_empty_rows() {
        let dir = tempfile::TempDir::new().unwrap();
        write_solver_stats(dir.path(), &[]).unwrap();

        let iter_path = dir.path().join("training/solver/iterations.parquet");
        assert!(iter_path.exists());
        let file = std::fs::File::open(&iter_path).unwrap();
        let builder = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
        assert_eq!(builder.schema().fields().len(), 23);

        let hist_path = dir.path().join("training/solver/retry_histogram.parquet");
        assert!(hist_path.exists());
        let file = std::fs::File::open(&hist_path).unwrap();
        let builder = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
        assert_eq!(builder.schema().fields().len(), 5);
    }

    #[test]
    fn retry_histogram_sparse_encoding() {
        let dir = tempfile::TempDir::new().unwrap();
        let rows = vec![
            SolverStatsRow {
                iteration: 1,
                phase: "forward".to_string(),
                stage: -1,
                lp_solves: 50,
                lp_successes: 48,
                lp_retries: 2,
                lp_failures: 0,
                retry_attempts: 3,
                basis_offered: 40,
                basis_rejections: 0,
                basis_non_alien_rejections: 0,
                clear_solver_count: 0,
                clear_solver_failures: 0,
                simplex_iterations: 2000,
                solve_time_ms: 10.0,
                load_model_time_ms: 0.0,
                add_rows_time_ms: 0.0,
                set_bounds_time_ms: 0.0,
                basis_set_time_ms: 0.0,
                basis_preserved: 0,
                basis_new_tight: 0,
                basis_new_slack: 0,
                basis_demotions: 0,
                // Level 0: 5 recoveries, level 2: 1 recovery
                retry_level_histogram: vec![5, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            },
            SolverStatsRow {
                iteration: 1,
                phase: "backward".to_string(),
                stage: 0,
                lp_solves: 100,
                lp_successes: 100,
                lp_retries: 0,
                lp_failures: 0,
                retry_attempts: 0,
                basis_offered: 80,
                basis_rejections: 0,
                basis_non_alien_rejections: 0,
                clear_solver_count: 0,
                clear_solver_failures: 0,
                simplex_iterations: 5000,
                solve_time_ms: 20.0,
                load_model_time_ms: 0.0,
                add_rows_time_ms: 0.0,
                set_bounds_time_ms: 0.0,
                basis_set_time_ms: 0.0,
                basis_preserved: 0,
                basis_new_tight: 0,
                basis_new_slack: 0,
                basis_demotions: 0,
                retry_level_histogram: vec![0; 12],
            },
        ];

        write_solver_stats(dir.path(), &rows).unwrap();

        let hist_path = dir.path().join("training/solver/retry_histogram.parquet");
        let batch = read_parquet(&hist_path);

        // Only 2 nonzero entries: (forward, level 0, 5) and (forward, level 2, 1)
        assert_eq!(batch.num_rows(), 2);

        let level_col = batch
            .column(3)
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap();
        assert_eq!(level_col.value(0), 0);
        assert_eq!(level_col.value(1), 2);

        let count_col = batch
            .column(4)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        assert_eq!(count_col.value(0), 5);
        assert_eq!(count_col.value(1), 1);
    }
}
