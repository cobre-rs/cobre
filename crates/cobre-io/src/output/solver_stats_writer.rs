//! Parquet writer for per-iteration solver statistics.
//!
//! Writes `training/solver/iterations.parquet` with one row per
//! (iteration, phase, stage) triple.

use std::path::Path;
use std::sync::Arc;

use arrow::array::{Float64Array, Int32Array, StringBuilder, UInt32Array, UInt64Array};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;

use super::error::OutputError;
use super::schemas::solver_iterations_schema;

/// A single row in the solver statistics Parquet file.
#[derive(Debug, Clone)]
pub struct SolverStatsRow {
    /// Iteration number (1-based).
    pub iteration: u32,
    /// Phase name: `"forward"`, `"backward"`, or `"lower_bound"`.
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
    /// Total simplex iterations.
    pub simplex_iterations: u64,
    /// Cumulative solve time in milliseconds.
    pub solve_time_ms: f64,
}

/// Write solver statistics rows to `training/solver/iterations.parquet`.
///
/// Creates the `training/solver/` directory if it does not exist. Uses atomic
/// write (`.tmp` + rename).
///
/// # Errors
///
/// Returns [`OutputError`] on filesystem or serialization failures.
pub fn write_solver_stats(output_dir: &Path, rows: &[SolverStatsRow]) -> Result<(), OutputError> {
    let dir = output_dir.join("training/solver");
    std::fs::create_dir_all(&dir).map_err(|e| OutputError::io(&dir, e))?;

    let path = dir.join("iterations.parquet");
    let tmp_path = path.with_extension("parquet.tmp");

    let schema = Arc::new(solver_iterations_schema());

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
    let simplex_iter_arr = UInt64Array::from(
        rows.iter()
            .map(|r| r.simplex_iterations)
            .collect::<Vec<_>>(),
    );
    let solve_time_arr =
        Float64Array::from(rows.iter().map(|r| r.solve_time_ms).collect::<Vec<_>>());

    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
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
            Arc::new(simplex_iter_arr),
            Arc::new(solve_time_arr),
        ],
    )
    .map_err(|e| OutputError::serialization("solver_stats", format!("RecordBatch: {e}")))?;

    let file = std::fs::File::create(&tmp_path).map_err(|e| OutputError::io(&tmp_path, e))?;
    let props = WriterProperties::builder()
        .set_compression(Compression::ZSTD(Default::default()))
        .build();
    let mut writer = ArrowWriter::try_new(file, Arc::clone(&schema), Some(props))
        .map_err(|e| OutputError::serialization("solver_stats", format!("ArrowWriter: {e}")))?;
    writer
        .write(&batch)
        .map_err(|e| OutputError::serialization("solver_stats", format!("write: {e}")))?;
    writer
        .close()
        .map_err(|e| OutputError::serialization("solver_stats", format!("close: {e}")))?;
    std::fs::rename(&tmp_path, &path).map_err(|e| OutputError::io(&path, e))?;

    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::float_cmp)]
mod tests {
    use super::*;
    use arrow::array::{AsArray, Float64Array, UInt32Array, UInt64Array};
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
                simplex_iterations: 5000,
                solve_time_ms: 42.5,
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
                simplex_iterations: 10000,
                solve_time_ms: 85.0,
            },
        ]
    }

    #[test]
    fn write_and_read_back() {
        let dir = tempfile::TempDir::new().unwrap();
        let rows = make_rows();

        write_solver_stats(dir.path(), &rows).unwrap();

        let path = dir.path().join("training/solver/iterations.parquet");
        assert!(path.exists());

        let file = std::fs::File::open(&path).unwrap();
        let builder = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
        let mut reader = builder.build().unwrap();
        let batch = reader.next().unwrap().unwrap();

        assert_eq!(batch.num_rows(), 2);
        assert_eq!(batch.num_columns(), 12);

        let iteration_col = batch
            .column(0)
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap();
        assert_eq!(iteration_col.value(0), 1);
        assert_eq!(iteration_col.value(1), 1);

        let solve_time_col = batch
            .column(11)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        assert!((solve_time_col.value(0) - 42.5).abs() < 1e-10);
        assert!((solve_time_col.value(1) - 85.0).abs() < 1e-10);

        let simplex_col = batch
            .column(10)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        assert_eq!(simplex_col.value(0), 5000);
    }

    #[test]
    fn write_empty_rows() {
        let dir = tempfile::TempDir::new().unwrap();
        write_solver_stats(dir.path(), &[]).unwrap();

        let path = dir.path().join("training/solver/iterations.parquet");
        assert!(path.exists());

        let file = std::fs::File::open(&path).unwrap();
        let builder = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
        let schema = builder.schema();
        assert_eq!(schema.fields().len(), 12);
    }
}
