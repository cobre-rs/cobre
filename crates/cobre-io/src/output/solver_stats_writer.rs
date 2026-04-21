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
    /// Opening (noise realization) index within the stage. `Some(ω)` for
    /// backward rows, `None` for forward, `lower_bound`, and simulation
    /// rows (which have no opening dimension).
    pub opening: Option<i32>,
    /// MPI rank that produced this row. `None` for rank-aggregated rows (the
    /// current state in T002); populated with real values in T005.
    pub rank: Option<i32>,
    /// Worker thread ID within the rank. `None` for rank-aggregated rows (the
    /// current state in T002); populated with real values in T005.
    pub worker_id: Option<i32>,
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
    /// Number of warm-start `solve(Some(&basis))` calls.
    pub basis_offered: u32,
    /// Times the offered basis was rejected because `isBasisConsistent` returned false.
    pub basis_consistency_failures: u32,
    /// Total simplex iterations.
    pub simplex_iterations: u64,
    /// Cumulative solve time in milliseconds.
    pub solve_time_ms: f64,
    /// Cumulative time in `load_model` calls, in milliseconds.
    pub load_model_time_ms: f64,
    /// Cumulative time in `set_row_bounds`/`set_col_bounds` calls, in milliseconds.
    pub set_bounds_time_ms: f64,
    /// Cumulative time in `set_basis` FFI calls, in milliseconds.
    pub basis_set_time_ms: f64,
    /// Number of `reconstruct_basis` invocations with a non-empty stored basis
    /// during this phase (once per warm-start solve that applied a stored basis
    /// via slot reconciliation).
    pub basis_reconstructions: u64,
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
    let opening_arr =
        Int32Array::from(rows.iter().map(|r| r.opening).collect::<Vec<Option<i32>>>());
    let rank_arr = Int32Array::from(rows.iter().map(|r| r.rank).collect::<Vec<Option<i32>>>());
    let worker_id_arr = Int32Array::from(
        rows.iter()
            .map(|r| r.worker_id)
            .collect::<Vec<Option<i32>>>(),
    );
    let lp_solves_arr = UInt32Array::from(rows.iter().map(|r| r.lp_solves).collect::<Vec<_>>());
    let lp_successes_arr =
        UInt32Array::from(rows.iter().map(|r| r.lp_successes).collect::<Vec<_>>());
    let lp_retries_arr = UInt32Array::from(rows.iter().map(|r| r.lp_retries).collect::<Vec<_>>());
    let lp_failures_arr = UInt32Array::from(rows.iter().map(|r| r.lp_failures).collect::<Vec<_>>());
    let retry_attempts_arr =
        UInt32Array::from(rows.iter().map(|r| r.retry_attempts).collect::<Vec<_>>());
    let basis_offered_arr =
        UInt32Array::from(rows.iter().map(|r| r.basis_offered).collect::<Vec<_>>());
    let basis_consistency_failures_arr = UInt32Array::from(
        rows.iter()
            .map(|r| r.basis_consistency_failures)
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
    let set_bounds_time_arr = Float64Array::from(
        rows.iter()
            .map(|r| r.set_bounds_time_ms)
            .collect::<Vec<_>>(),
    );
    let basis_set_time_arr =
        Float64Array::from(rows.iter().map(|r| r.basis_set_time_ms).collect::<Vec<_>>());
    let basis_reconstructions_arr = UInt64Array::from(
        rows.iter()
            .map(|r| r.basis_reconstructions)
            .collect::<Vec<_>>(),
    );

    vec![
        Arc::new(iteration_arr),
        Arc::new(phase_arr),
        Arc::new(stage_arr),
        Arc::new(opening_arr),
        Arc::new(rank_arr),
        Arc::new(worker_id_arr),
        Arc::new(lp_solves_arr),
        Arc::new(lp_successes_arr),
        Arc::new(lp_retries_arr),
        Arc::new(lp_failures_arr),
        Arc::new(retry_attempts_arr),
        Arc::new(basis_offered_arr),
        Arc::new(basis_consistency_failures_arr),
        Arc::new(simplex_iter_arr),
        Arc::new(solve_time_arr),
        Arc::new(load_model_time_arr),
        Arc::new(set_bounds_time_arr),
        Arc::new(basis_set_time_arr),
        Arc::new(basis_reconstructions_arr),
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
    use arrow::array::{Array, Float64Array, Int32Array, UInt32Array, UInt64Array};
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

    fn make_rows() -> Vec<SolverStatsRow> {
        vec![
            SolverStatsRow {
                iteration: 1,
                phase: "forward".to_string(),
                stage: 0, // ticket-011a: forward rows use real stage index, not -1
                opening: None,
                rank: None,
                worker_id: None,
                lp_solves: 100,
                lp_successes: 98,
                lp_retries: 2,
                lp_failures: 0,
                retry_attempts: 4,
                basis_offered: 90,
                basis_consistency_failures: 3,
                simplex_iterations: 5000,
                solve_time_ms: 42.5,
                load_model_time_ms: 0.0,
                set_bounds_time_ms: 0.0,
                basis_set_time_ms: 0.0,
                basis_reconstructions: 0,
                retry_level_histogram: vec![0; 12],
            },
            SolverStatsRow {
                iteration: 1,
                phase: "backward".to_string(),
                stage: 2,
                opening: Some(0),
                rank: None,
                worker_id: None,
                lp_solves: 200,
                lp_successes: 200,
                lp_retries: 0,
                lp_failures: 0,
                retry_attempts: 0,
                basis_offered: 180,
                basis_consistency_failures: 1,
                simplex_iterations: 10000,
                solve_time_ms: 85.0,
                load_model_time_ms: 0.0,
                set_bounds_time_ms: 0.0,
                basis_set_time_ms: 0.0,
                basis_reconstructions: 0,
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

        // iterations.parquet — 19 scalar columns (opening + rank + worker_id: Int32, nullable)
        let iter_path = dir.path().join("training/solver/iterations.parquet");
        assert!(iter_path.exists());
        let batch = read_parquet(&iter_path);

        assert_eq!(batch.num_rows(), 2);
        assert_eq!(batch.num_columns(), 19);

        let iteration_col = batch
            .column(0)
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap();
        assert_eq!(iteration_col.value(0), 1);
        assert_eq!(iteration_col.value(1), 1);

        // Column indices:
        // 0 = iteration, 1 = phase, 2 = stage, 3 = opening, 4 = rank, 5 = worker_id,
        // 6 = lp_solves, ..., 12 = basis_consistency_failures,
        // 13 = simplex_iterations, 14 = solve_time_ms, ..., 18 = basis_reconstructions
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
        assert_eq!(builder.schema().fields().len(), 19);

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
                stage: 0, // ticket-011a: forward rows use real stage index, not -1
                opening: None,
                rank: None,
                worker_id: None,
                lp_solves: 50,
                lp_successes: 48,
                lp_retries: 2,
                lp_failures: 0,
                retry_attempts: 3,
                basis_offered: 40,
                basis_consistency_failures: 0,
                simplex_iterations: 2000,
                solve_time_ms: 10.0,
                load_model_time_ms: 0.0,
                set_bounds_time_ms: 0.0,
                basis_set_time_ms: 0.0,
                basis_reconstructions: 0,
                // Level 0: 5 recoveries, level 2: 1 recovery
                retry_level_histogram: vec![5, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            },
            SolverStatsRow {
                iteration: 1,
                phase: "backward".to_string(),
                stage: 0,
                opening: Some(0),
                rank: None,
                worker_id: None,
                lp_solves: 100,
                lp_successes: 100,
                lp_retries: 0,
                lp_failures: 0,
                retry_attempts: 0,
                basis_offered: 80,
                basis_consistency_failures: 0,
                simplex_iterations: 5000,
                solve_time_ms: 20.0,
                load_model_time_ms: 0.0,
                set_bounds_time_ms: 0.0,
                basis_set_time_ms: 0.0,
                basis_reconstructions: 0,
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

    #[test]
    fn test_solver_stats_row_builds_with_none_opening() {
        // Verify a row with opening=None can be written and read back without error.
        // Uses stage=0 (real stage index) consistent with ticket-011a per-stage shape.
        let dir = tempfile::TempDir::new().unwrap();
        let rows = vec![SolverStatsRow {
            iteration: 1,
            phase: "forward".to_string(),
            stage: 0,
            opening: None,
            rank: None,
            worker_id: None,
            lp_solves: 10,
            lp_successes: 10,
            lp_retries: 0,
            lp_failures: 0,
            retry_attempts: 0,
            basis_offered: 8,
            basis_consistency_failures: 0,
            simplex_iterations: 500,
            solve_time_ms: 1.0,
            load_model_time_ms: 0.0,
            set_bounds_time_ms: 0.0,
            basis_set_time_ms: 0.0,
            basis_reconstructions: 0,
            retry_level_histogram: vec![0; 12],
        }];

        write_solver_stats(dir.path(), &rows).unwrap();

        let iter_path = dir.path().join("training/solver/iterations.parquet");
        let batch = read_parquet(&iter_path);
        assert_eq!(batch.num_rows(), 1);

        // opening column is at index 3, must be null for forward rows.
        let opening_col = batch
            .column(3)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        assert!(opening_col.is_null(0), "forward row must have NULL opening");
    }

    #[test]
    fn test_solver_stats_row_builds_with_none_rank_and_worker_id() {
        // Verify a row with rank=None, worker_id=None, opening=Some(3) can be
        // written and read back without error. Both new nullable columns must
        // appear as NULL in the output parquet (T002 produces all-NULL values;
        // T005 will populate them with real values).
        let dir = tempfile::TempDir::new().unwrap();
        let rows = vec![SolverStatsRow {
            iteration: 1,
            phase: "backward".to_string(),
            stage: 0,
            opening: Some(3),
            rank: None,
            worker_id: None,
            lp_solves: 1,
            lp_successes: 1,
            lp_retries: 0,
            lp_failures: 0,
            retry_attempts: 0,
            basis_offered: 0,
            basis_consistency_failures: 0,
            simplex_iterations: 0,
            solve_time_ms: 0.0,
            load_model_time_ms: 0.0,
            set_bounds_time_ms: 0.0,
            basis_set_time_ms: 0.0,
            basis_reconstructions: 0,
            retry_level_histogram: vec![0; 12],
        }];

        write_solver_stats(dir.path(), &rows).unwrap();

        let iter_path = dir.path().join("training/solver/iterations.parquet");
        let batch = read_parquet(&iter_path);

        // Schema must have exactly 19 columns.
        assert_eq!(batch.num_columns(), 19);

        // rank column is at index 4, must be NULL.
        let rank_col = batch.column_by_name("rank").unwrap();
        assert_eq!(
            rank_col.null_count(),
            1,
            "rank must be NULL for every T002 row"
        );

        // worker_id column is at index 5, must be NULL.
        let worker_col = batch.column_by_name("worker_id").unwrap();
        assert_eq!(
            worker_col.null_count(),
            1,
            "worker_id must be NULL for every T002 row"
        );

        // opening column must carry the value we set (Some(3) → not NULL).
        let opening_col = batch
            .column(3)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        assert!(!opening_col.is_null(0), "opening must be non-NULL");
        assert_eq!(opening_col.value(0), 3, "opening value must be 3");
    }

    #[allow(clippy::too_many_lines)]
    #[test]
    fn test_opening_column_sum_invariant() {
        // Invariant: SUM(lp_solves) GROUP BY (iteration, phase, stage) in the
        // new per-opening schema equals what the old collapsed schema would have
        // reported (i.e., the sum of per-opening rows matches the old total).
        //
        // Fixture: iteration 1, backward phase, stage 0, 3 openings.
        let dir = tempfile::TempDir::new().unwrap();
        let rows = vec![
            // Forward row (opening=None, stage=0): 50 lp_solves
            // ticket-011a: forward rows use real stage index, not -1.
            SolverStatsRow {
                iteration: 1,
                phase: "forward".to_string(),
                stage: 0,
                opening: None,
                rank: None,
                worker_id: None,
                lp_solves: 50,
                lp_successes: 50,
                lp_retries: 0,
                lp_failures: 0,
                retry_attempts: 0,
                basis_offered: 40,
                basis_consistency_failures: 0,
                simplex_iterations: 1000,
                solve_time_ms: 5.0,
                load_model_time_ms: 0.0,
                set_bounds_time_ms: 0.0,
                basis_set_time_ms: 0.0,
                basis_reconstructions: 0,
                retry_level_histogram: vec![0; 12],
            },
            // Backward rows (opening=Some(0..2)): 10, 20, 30 lp_solves → sum=60
            SolverStatsRow {
                iteration: 1,
                phase: "backward".to_string(),
                stage: 0,
                opening: Some(0),
                rank: None,
                worker_id: None,
                lp_solves: 10,
                lp_successes: 10,
                lp_retries: 0,
                lp_failures: 0,
                retry_attempts: 0,
                basis_offered: 8,
                basis_consistency_failures: 0,
                simplex_iterations: 200,
                solve_time_ms: 2.0,
                load_model_time_ms: 0.0,
                set_bounds_time_ms: 0.0,
                basis_set_time_ms: 0.0,
                basis_reconstructions: 0,
                retry_level_histogram: vec![0; 12],
            },
            SolverStatsRow {
                iteration: 1,
                phase: "backward".to_string(),
                stage: 0,
                opening: Some(1),
                rank: None,
                worker_id: None,
                lp_solves: 20,
                lp_successes: 20,
                lp_retries: 0,
                lp_failures: 0,
                retry_attempts: 0,
                basis_offered: 18,
                basis_consistency_failures: 0,
                simplex_iterations: 400,
                solve_time_ms: 4.0,
                load_model_time_ms: 0.0,
                set_bounds_time_ms: 0.0,
                basis_set_time_ms: 0.0,
                basis_reconstructions: 0,
                retry_level_histogram: vec![0; 12],
            },
            SolverStatsRow {
                iteration: 1,
                phase: "backward".to_string(),
                stage: 0,
                opening: Some(2),
                rank: None,
                worker_id: None,
                lp_solves: 30,
                lp_successes: 30,
                lp_retries: 0,
                lp_failures: 0,
                retry_attempts: 0,
                basis_offered: 28,
                basis_consistency_failures: 0,
                simplex_iterations: 600,
                solve_time_ms: 6.0,
                load_model_time_ms: 0.0,
                set_bounds_time_ms: 0.0,
                basis_set_time_ms: 0.0,
                basis_reconstructions: 0,
                retry_level_histogram: vec![0; 12],
            },
        ];

        write_solver_stats(dir.path(), &rows).unwrap();

        let iter_path = dir.path().join("training/solver/iterations.parquet");
        let batch = read_parquet(&iter_path);

        // 4 rows: 1 forward + 3 backward-opening rows.
        assert_eq!(batch.num_rows(), 4);

        // lp_solves is at column index 6 (after: iteration, phase, stage, opening, rank, worker_id).
        let lp_col = batch
            .column(6)
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap();

        // Group by (iteration=1, phase="backward", stage=0): sum across openings.
        let backward_sum: u32 = (0..4)
            .filter(|&i| {
                batch
                    .column(1)
                    .as_any()
                    .downcast_ref::<arrow::array::StringArray>()
                    .unwrap()
                    .value(i)
                    == "backward"
            })
            .map(|i| lp_col.value(i))
            .sum();

        // The sum of per-opening lp_solves (10+20+30) must equal the old
        // collapsed per-stage total.
        assert_eq!(
            backward_sum, 60,
            "SUM(lp_solves) for backward stage 0 must equal 60"
        );

        // Forward row: lp_solves=50, opening=NULL.
        let opening_col = batch
            .column(3)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        assert!(opening_col.is_null(0), "forward row must have NULL opening");
        assert_eq!(opening_col.value(1), 0, "backward opening[0] must be 0");
        assert_eq!(opening_col.value(2), 1, "backward opening[1] must be 1");
        assert_eq!(opening_col.value(3), 2, "backward opening[2] must be 2");
    }

    /// ticket-011a: forward rows are per-stage (one row per stage, opening=NULL).
    ///
    /// Verifies that 3 forward rows for stages 0, 1, 2 produce a parquet with
    /// exactly 3 rows, each with opening=NULL and the correct per-stage stage index.
    #[test]
    fn test_forward_rows_are_per_stage_in_parquet() {
        // Helper defined first (before any statements) to satisfy
        // clippy::items_after_statements.
        fn make_forward_row(stage: i32, lp_solves: u32) -> SolverStatsRow {
            SolverStatsRow {
                iteration: 1,
                phase: "forward".to_string(),
                stage,
                opening: None, // forward has no opening dimension
                rank: None,
                worker_id: None,
                lp_solves,
                lp_successes: lp_solves,
                lp_retries: 0,
                lp_failures: 0,
                retry_attempts: 0,
                basis_offered: 0,
                basis_consistency_failures: 0,
                simplex_iterations: u64::from(lp_solves) * 5,
                solve_time_ms: f64::from(lp_solves) * 0.5,
                load_model_time_ms: 0.0,
                set_bounds_time_ms: 0.0,
                basis_set_time_ms: 0.0,
                basis_reconstructions: 0,
                retry_level_histogram: vec![0; 12],
            }
        }

        // Simulate one iteration with 3 stages — one forward row per stage.
        let dir = tempfile::TempDir::new().unwrap();
        let rows = vec![
            make_forward_row(0, 10), // stage 0: 10 lp_solves
            make_forward_row(1, 20), // stage 1: 20 lp_solves
            make_forward_row(2, 30), // stage 2: 30 lp_solves
        ];

        write_solver_stats(dir.path(), &rows).unwrap();

        let iter_path = dir.path().join("training/solver/iterations.parquet");
        let batch = read_parquet(&iter_path);

        // ticket-011a AC: parquet has exactly num_stages rows for the forward phase.
        assert_eq!(batch.num_rows(), 3, "one forward row per stage");

        let opening_col = batch
            .column(3)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        let stage_col = batch
            .column(2)
            .as_any()
            .downcast_ref::<arrow::array::Int32Array>()
            .unwrap();
        let lp_col = batch
            .column(6)
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap();

        for row in 0..3 {
            // AC: every forward row has opening = NULL.
            assert!(
                opening_col.is_null(row),
                "forward row {row} must have NULL opening"
            );
            // AC: stage index equals the loop variable (0, 1, 2).
            assert_eq!(
                stage_col.value(row),
                i32::try_from(row).unwrap(),
                "forward row {row} must have stage = {row}"
            );
        }

        // AC: no stage = -1 among forward rows.
        for row in 0..3 {
            assert_ne!(
                stage_col.value(row),
                -1,
                "forward rows must not use stage = -1 (ticket-011a)"
            );
        }

        // AC: per-stage lp_solves are preserved correctly.
        assert_eq!(lp_col.value(0), 10, "stage 0: 10 lp_solves");
        assert_eq!(lp_col.value(1), 20, "stage 1: 20 lp_solves");
        assert_eq!(lp_col.value(2), 30, "stage 2: 30 lp_solves");
    }
}
