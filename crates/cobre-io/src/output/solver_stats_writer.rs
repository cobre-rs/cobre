//! Parquet writer for per-iteration solver statistics.
//!
//! Writes `training/solver/iterations.parquet` (scalar metrics) and
//! `training/solver/retry_histogram.parquet` (normalized per-level retry counts).

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use arrow::array::{
    Float64Builder, Int32Array, Int32Builder, StringBuilder, UInt32Array, UInt32Builder,
    UInt64Array, UInt64Builder,
};
use arrow::record_batch::RecordBatch;

use super::atomic::write_parquet_atomic;
use super::error::OutputError;
use super::schemas::{retry_histogram_schema, solver_iterations_schema};

pub(crate) const TRAINING_SOLVER_DIR: &str = "training/solver";
pub(crate) const SIMULATION_SOLVER_DIR: &str = "simulation/solver";
pub(crate) const SOLVER_ITERATIONS_FILE: &str = "iterations.parquet";
pub(crate) const SOLVER_RETRY_HISTOGRAM_FILE: &str = "retry_histogram.parquet";

/// A single row in the solver statistics Parquet file.
#[derive(Debug, Clone)]
pub struct SolverStatsRow {
    /// Training iteration number (1-based); `None` on a simulation row, which
    /// fills [`Self::scenario_id`] instead.
    pub iteration: Option<i32>,
    /// Simulation trajectory id (0-based); `None` on a training row, which fills
    /// [`Self::iteration`] instead.
    pub scenario_id: Option<i32>,
    /// Phase name: `"forward"`, `"backward"`, `"lower_bound"`, or `"simulation"`.
    pub phase: &'static str,
    /// Declared study `stage_id` for forward/backward rows; `None` for the
    /// lower-bound and simulation rows that carry no per-stage attribution.
    pub stage_id: Option<i32>,
    /// Opening (noise realization) index within the stage. `Some(ω)` for
    /// backward rows, `None` for forward, `lower_bound`, and simulation
    /// rows (which have no opening dimension).
    pub opening_index: Option<i32>,
    /// MPI rank that produced this row. `None` for rank-aggregated rows.
    pub rank: Option<i32>,
    /// Rayon worker index within the rank's thread pool. `None` for rank-aggregated rows.
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
    /// Per-level retry success counts. Length depends on the solver backend
    /// (e.g. 12 for `HiGHS`).
    pub retry_level_histogram: Vec<u64>,
}

/// Write training solver statistics to `training/solver/iterations.parquet`.
///
/// # Errors
///
/// Returns [`OutputError`] on filesystem or serialization failures.
pub fn write_solver_stats(output_dir: &Path, rows: &[SolverStatsRow]) -> Result<(), OutputError> {
    write_solver_stats_to(&output_dir.join(TRAINING_SOLVER_DIR), rows)
}

/// Write simulation solver statistics to `simulation/solver/iterations.parquet`.
///
/// # Errors
///
/// Returns [`OutputError`] on filesystem or serialization failures.
pub fn write_simulation_solver_stats(
    output_dir: &Path,
    rows: &[SolverStatsRow],
) -> Result<(), OutputError> {
    write_solver_stats_to(&output_dir.join(SIMULATION_SOLVER_DIR), rows)
}

/// Build Arrow column arrays for `iterations.parquet` (scalar metrics only).
fn build_iterations_columns(rows: &[SolverStatsRow]) -> Vec<Arc<dyn arrow::array::Array>> {
    let n = rows.len();
    let mut iteration = Int32Builder::with_capacity(n);
    let mut scenario_id = Int32Builder::with_capacity(n);
    let mut phase = StringBuilder::with_capacity(n, n * 10);
    let mut stage_id = Int32Builder::with_capacity(n);
    let mut opening_index = Int32Builder::with_capacity(n);
    let mut rank = Int32Builder::with_capacity(n);
    let mut worker_id = Int32Builder::with_capacity(n);
    let mut lp_solves = UInt32Builder::with_capacity(n);
    let mut lp_successes = UInt32Builder::with_capacity(n);
    let mut lp_retries = UInt32Builder::with_capacity(n);
    let mut lp_failures = UInt32Builder::with_capacity(n);
    let mut retry_attempts = UInt32Builder::with_capacity(n);
    let mut basis_offered = UInt32Builder::with_capacity(n);
    let mut basis_consistency_failures = UInt32Builder::with_capacity(n);
    let mut simplex_iterations = UInt64Builder::with_capacity(n);
    let mut solve_time_ms = Float64Builder::with_capacity(n);
    let mut load_model_time_ms = Float64Builder::with_capacity(n);
    let mut set_bounds_time_ms = Float64Builder::with_capacity(n);
    let mut basis_set_time_ms = Float64Builder::with_capacity(n);

    for r in rows {
        iteration.append_option(r.iteration);
        scenario_id.append_option(r.scenario_id);
        phase.append_value(r.phase);
        stage_id.append_option(r.stage_id);
        opening_index.append_option(r.opening_index);
        rank.append_option(r.rank);
        worker_id.append_option(r.worker_id);
        lp_solves.append_value(r.lp_solves);
        lp_successes.append_value(r.lp_successes);
        lp_retries.append_value(r.lp_retries);
        lp_failures.append_value(r.lp_failures);
        retry_attempts.append_value(r.retry_attempts);
        basis_offered.append_value(r.basis_offered);
        basis_consistency_failures.append_value(r.basis_consistency_failures);
        simplex_iterations.append_value(r.simplex_iterations);
        solve_time_ms.append_value(r.solve_time_ms);
        load_model_time_ms.append_value(r.load_model_time_ms);
        set_bounds_time_ms.append_value(r.set_bounds_time_ms);
        basis_set_time_ms.append_value(r.basis_set_time_ms);
    }

    vec![
        Arc::new(iteration.finish()),
        Arc::new(scenario_id.finish()),
        Arc::new(phase.finish()),
        Arc::new(stage_id.finish()),
        Arc::new(opening_index.finish()),
        Arc::new(rank.finish()),
        Arc::new(worker_id.finish()),
        Arc::new(lp_solves.finish()),
        Arc::new(lp_successes.finish()),
        Arc::new(lp_retries.finish()),
        Arc::new(lp_failures.finish()),
        Arc::new(retry_attempts.finish()),
        Arc::new(basis_offered.finish()),
        Arc::new(basis_consistency_failures.finish()),
        Arc::new(simplex_iterations.finish()),
        Arc::new(solve_time_ms.finish()),
        Arc::new(load_model_time_ms.finish()),
        Arc::new(set_bounds_time_ms.finish()),
        Arc::new(basis_set_time_ms.finish()),
    ]
}

/// Build a `RecordBatch` for `retry_histogram.parquet`: one row per
/// (iteration, phase, `stage_id`, `retry_level`) tuple whose summed `count > 0`.
///
/// The row identity is `iteration` on a training row and `scenario_id` on a
/// simulation row (exactly one is set); `stage_id` is `None` for the
/// forward/lower-bound/simulation rows that carry no per-stage attribution.
fn build_retry_histogram_batch(rows: &[SolverStatsRow]) -> Result<RecordBatch, OutputError> {
    // Sum per-level counts across every row sharing an (id, phase, stage_id)
    // key. BTreeMap (never HashMap) emits canonical (id, phase, stage_id,
    // level) order, making the file a pure function of the aggregate retry data
    // regardless of rank/worker/opening partitioning — the declaration-order rule.
    let mut aggregated: BTreeMap<(i32, &str, Option<i32>), Vec<u64>> = BTreeMap::new();
    for r in rows {
        let id = r.iteration.or(r.scenario_id).unwrap_or_default();
        let entry = aggregated.entry((id, r.phase, r.stage_id)).or_default();
        if entry.len() < r.retry_level_histogram.len() {
            entry.resize(r.retry_level_histogram.len(), 0);
        }
        for (acc, &count) in entry.iter_mut().zip(r.retry_level_histogram.iter()) {
            *acc += count;
        }
    }

    let mut iterations = Vec::new();
    let mut phases = Vec::new();
    let mut stages: Vec<Option<i32>> = Vec::new();
    let mut levels = Vec::new();
    let mut counts = Vec::new();

    for (&(id, phase, stage_id), histogram) in &aggregated {
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        for (level, &count) in histogram.iter().enumerate() {
            if count > 0 {
                iterations.push(id as u32);
                phases.push(phase);
                stages.push(stage_id);
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

/// Internal: write solver statistics to `{dir}/iterations.parquet` and
/// `{dir}/retry_histogram.parquet`.
fn write_solver_stats_to(dir: &Path, rows: &[SolverStatsRow]) -> Result<(), OutputError> {
    std::fs::create_dir_all(dir).map_err(|e| OutputError::io(dir, e))?;

    let iter_schema = Arc::new(solver_iterations_schema());
    let columns = build_iterations_columns(rows);
    let iter_batch = RecordBatch::try_new(Arc::clone(&iter_schema), columns)
        .map_err(|e| OutputError::serialization("solver_stats", format!("RecordBatch: {e}")))?;
    write_parquet_atomic(&dir.join(SOLVER_ITERATIONS_FILE), &iter_batch)?;

    let hist_batch = build_retry_histogram_batch(rows)?;
    write_parquet_atomic(&dir.join(SOLVER_RETRY_HISTOGRAM_FILE), &hist_batch)?;

    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::float_cmp)]
mod tests {
    use super::*;
    use crate::test_support::output::read_first_batch;
    use arrow::array::{Array, Float64Array, Int32Array, StringArray, UInt32Array, UInt64Array};
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

    /// A zeroed row with all axis columns absent; tests set the axes they exercise.
    fn base_row(phase: &'static str) -> SolverStatsRow {
        SolverStatsRow {
            iteration: None,
            scenario_id: None,
            phase,
            stage_id: None,
            opening_index: None,
            rank: None,
            worker_id: None,
            lp_solves: 0,
            lp_successes: 0,
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
            retry_level_histogram: vec![0; 12],
        }
    }

    fn make_rows() -> Vec<SolverStatsRow> {
        vec![
            SolverStatsRow {
                iteration: Some(1),
                stage_id: Some(0),
                lp_solves: 100,
                lp_successes: 98,
                lp_retries: 2,
                basis_offered: 90,
                basis_consistency_failures: 3,
                simplex_iterations: 5000,
                solve_time_ms: 42.5,
                ..base_row("forward")
            },
            SolverStatsRow {
                iteration: Some(1),
                stage_id: Some(2),
                opening_index: Some(0),
                lp_solves: 200,
                lp_successes: 200,
                basis_offered: 180,
                basis_consistency_failures: 1,
                simplex_iterations: 10000,
                solve_time_ms: 85.0,
                ..base_row("backward")
            },
        ]
    }

    #[test]
    fn write_and_read_back() {
        let dir = tempfile::TempDir::new().unwrap();
        let rows = make_rows();

        write_solver_stats(dir.path(), &rows).unwrap();

        let iter_path = dir.path().join("training/solver/iterations.parquet");
        assert!(iter_path.exists());
        let batch = read_first_batch(&iter_path);

        assert_eq!(batch.num_rows(), 2);
        assert_eq!(batch.num_columns(), 19);

        let iteration_col = batch
            .column_by_name("iteration")
            .unwrap()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        assert_eq!(iteration_col.value(0), 1);
        assert_eq!(iteration_col.value(1), 1);

        let solve_time_col = batch
            .column_by_name("solve_time_ms")
            .unwrap()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        assert!((solve_time_col.value(0) - 42.5).abs() < 1e-10);
        assert!((solve_time_col.value(1) - 85.0).abs() < 1e-10);

        let simplex_col = batch
            .column_by_name("simplex_iterations")
            .unwrap()
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
    fn stage_id_null_for_lower_bound_and_simulation_and_no_minus_one() {
        // forward/backward rows carry a real (domain) stage_id; lower_bound and
        // simulation rows carry NULL. No -1 sentinel appears in any output column.
        let dir = tempfile::TempDir::new().unwrap();
        let rows = vec![
            SolverStatsRow {
                iteration: Some(1),
                stage_id: Some(3),
                ..base_row("forward")
            },
            SolverStatsRow {
                iteration: Some(1),
                stage_id: Some(4),
                opening_index: Some(0),
                ..base_row("backward")
            },
            SolverStatsRow {
                iteration: Some(1),
                stage_id: None,
                ..base_row("lower_bound")
            },
        ];
        write_solver_stats(dir.path(), &rows).unwrap();

        let batch = read_first_batch(&dir.path().join("training/solver/iterations.parquet"));
        let stage = batch
            .column_by_name("stage_id")
            .unwrap()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        assert!(
            !stage.is_null(0) && stage.value(0) == 3,
            "forward keeps stage_id"
        );
        assert!(
            !stage.is_null(1) && stage.value(1) == 4,
            "backward keeps stage_id"
        );
        assert!(stage.is_null(2), "lower_bound stage_id must be NULL");

        for name in ["stage_id", "iteration", "scenario_id", "opening_index"] {
            let col = batch
                .column_by_name(name)
                .unwrap()
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap();
            for i in 0..col.len() {
                assert!(
                    col.is_null(i) || col.value(i) != -1,
                    "no -1 sentinel allowed in {name}"
                );
            }
        }
    }

    #[test]
    fn training_row_fills_iteration_simulation_row_fills_scenario_id() {
        let dir = tempfile::TempDir::new().unwrap();
        let rows = vec![
            SolverStatsRow {
                iteration: Some(7),
                stage_id: Some(0),
                ..base_row("forward")
            },
            SolverStatsRow {
                scenario_id: Some(12),
                ..base_row("simulation")
            },
        ];
        write_solver_stats(dir.path(), &rows).unwrap();

        let batch = read_first_batch(&dir.path().join("training/solver/iterations.parquet"));
        let iter_col = batch
            .column_by_name("iteration")
            .unwrap()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        let scen_col = batch
            .column_by_name("scenario_id")
            .unwrap()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        assert!(!iter_col.is_null(0) && iter_col.value(0) == 7);
        assert!(scen_col.is_null(0), "training row scenario_id must be NULL");
        assert!(iter_col.is_null(1), "simulation row iteration must be NULL");
        assert!(!scen_col.is_null(1) && scen_col.value(1) == 12);
    }

    #[test]
    fn retry_histogram_sparse_encoding() {
        let dir = tempfile::TempDir::new().unwrap();
        let rows = vec![
            SolverStatsRow {
                iteration: Some(1),
                stage_id: Some(0),
                lp_solves: 50,
                // Level 0: 5 recoveries, level 2: 1 recovery
                retry_level_histogram: vec![5, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ..base_row("forward")
            },
            SolverStatsRow {
                iteration: Some(1),
                stage_id: Some(0),
                opening_index: Some(0),
                lp_solves: 100,
                ..base_row("backward")
            },
        ];

        write_solver_stats(dir.path(), &rows).unwrap();

        let hist_path = dir.path().join("training/solver/retry_histogram.parquet");
        let batch = read_first_batch(&hist_path);

        // Only 2 nonzero entries: (forward, level 0, 5) and (forward, level 2, 1);
        // the all-zero backward row contributes nothing.
        assert_eq!(batch.num_rows(), 2);

        let phase_col = batch
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(phase_col.value(0), "forward");
        assert_eq!(phase_col.value(1), "forward");

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
    fn retry_histogram_aggregates_shared_keys_in_canonical_order() {
        // Rows sharing (iteration, phase, stage_id) but differing by opening/rank,
        // fed SHUFFLED, must aggregate to exactly one row per
        // (iteration, phase, stage_id, retry_level) with summed counts, emitted in
        // canonical order — proving the file is a pure function of the aggregate
        // retry data, independent of input partitioning/order.
        fn make(
            iteration: i32,
            phase: &'static str,
            stage_id: i32,
            opening_index: Option<i32>,
            rank: Option<i32>,
            hist: Vec<u64>,
        ) -> SolverStatsRow {
            SolverStatsRow {
                iteration: Some(iteration),
                stage_id: Some(stage_id),
                opening_index,
                rank,
                retry_level_histogram: hist,
                ..base_row(phase)
            }
        }

        let dir = tempfile::TempDir::new().unwrap();
        let rows = vec![
            make(
                1,
                "backward",
                0,
                Some(2),
                Some(1),
                vec![0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ),
            make(
                1,
                "forward",
                0,
                None,
                Some(0),
                vec![7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ),
            make(
                1,
                "backward",
                1,
                Some(0),
                Some(0),
                vec![0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ),
            make(
                1,
                "backward",
                0,
                Some(0),
                Some(0),
                vec![2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ),
            make(
                1,
                "backward",
                0,
                Some(1),
                Some(1),
                vec![1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ),
        ];

        write_solver_stats(dir.path(), &rows).unwrap();

        let hist_path = dir.path().join("training/solver/retry_histogram.parquet");
        let batch = read_first_batch(&hist_path);

        // Aggregated unique tuples, canonical order (phase sorts "backward" < "forward"):
        //   (1, backward, 0, level 0) = 2 + 1 = 3
        //   (1, backward, 0, level 1) = 3 + 1 = 4
        //   (1, backward, 1, level 2) = 4
        //   (1, forward,  0, level 0) = 7
        assert_eq!(batch.num_rows(), 4);

        let iter_col = batch
            .column(0)
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap();
        let phase_col = batch
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let stage_col = batch
            .column(2)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        let level_col = batch
            .column(3)
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap();
        let count_col = batch
            .column(4)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();

        let expected = [
            (1u32, "backward", 0i32, 0u32, 3u64),
            (1, "backward", 0, 1, 4),
            (1, "backward", 1, 2, 4),
            (1, "forward", 0, 0, 7),
        ];
        for (i, (it, ph, st, lv, ct)) in expected.iter().enumerate() {
            assert_eq!(iter_col.value(i), *it, "iteration row {i}");
            assert_eq!(phase_col.value(i), *ph, "phase row {i}");
            assert_eq!(stage_col.value(i), *st, "stage_id row {i}");
            assert_eq!(level_col.value(i), *lv, "retry_level row {i}");
            assert_eq!(count_col.value(i), *ct, "count row {i}");
        }
    }

    #[test]
    fn opening_index_none_writes_null() {
        let dir = tempfile::TempDir::new().unwrap();
        let rows = vec![SolverStatsRow {
            iteration: Some(1),
            stage_id: Some(0),
            ..base_row("forward")
        }];

        write_solver_stats(dir.path(), &rows).unwrap();

        let batch = read_first_batch(&dir.path().join("training/solver/iterations.parquet"));
        let opening_col = batch
            .column_by_name("opening_index")
            .unwrap()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        assert!(
            opening_col.is_null(0),
            "forward row must have NULL opening_index"
        );
    }

    #[test]
    fn rank_and_worker_null_opening_index_present() {
        let dir = tempfile::TempDir::new().unwrap();
        let rows = vec![SolverStatsRow {
            iteration: Some(1),
            stage_id: Some(0),
            opening_index: Some(3),
            ..base_row("backward")
        }];

        write_solver_stats(dir.path(), &rows).unwrap();

        let batch = read_first_batch(&dir.path().join("training/solver/iterations.parquet"));
        assert_eq!(batch.num_columns(), 19);

        assert_eq!(
            batch.column_by_name("rank").unwrap().null_count(),
            1,
            "rank must be NULL for rank-aggregated rows"
        );
        assert_eq!(
            batch.column_by_name("worker_id").unwrap().null_count(),
            1,
            "worker_id must be NULL for rank-aggregated rows"
        );

        let opening_col = batch
            .column_by_name("opening_index")
            .unwrap()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        assert!(!opening_col.is_null(0), "opening_index must be non-NULL");
        assert_eq!(opening_col.value(0), 3, "opening_index value must be 3");
    }

    #[test]
    fn opening_column_sum_invariant() {
        // SUM(lp_solves) GROUP BY (iteration, phase, stage_id) over the per-opening
        // schema equals what the old collapsed per-stage total reported.
        let dir = tempfile::TempDir::new().unwrap();
        let mut rows = vec![SolverStatsRow {
            iteration: Some(1),
            stage_id: Some(0),
            lp_solves: 50,
            ..base_row("forward")
        }];
        for (opening, lp) in [(0, 10), (1, 20), (2, 30)] {
            rows.push(SolverStatsRow {
                iteration: Some(1),
                stage_id: Some(0),
                opening_index: Some(opening),
                lp_solves: lp,
                ..base_row("backward")
            });
        }

        write_solver_stats(dir.path(), &rows).unwrap();

        let batch = read_first_batch(&dir.path().join("training/solver/iterations.parquet"));
        assert_eq!(batch.num_rows(), 4);

        let lp_col = batch
            .column_by_name("lp_solves")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap();
        let phase_col = batch
            .column_by_name("phase")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let backward_sum: u32 = (0..4)
            .filter(|&i| phase_col.value(i) == "backward")
            .map(|i| lp_col.value(i))
            .sum();
        assert_eq!(
            backward_sum, 60,
            "SUM(lp_solves) for backward stage 0 must equal 60"
        );

        let opening_col = batch
            .column_by_name("opening_index")
            .unwrap()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        assert!(
            opening_col.is_null(0),
            "forward row must have NULL opening_index"
        );
        assert_eq!(opening_col.value(1), 0);
        assert_eq!(opening_col.value(2), 1);
        assert_eq!(opening_col.value(3), 2);
    }

    #[test]
    fn forward_rows_are_per_stage_in_parquet() {
        // Three forward rows for stages 0,1,2 produce three parquet rows, each with
        // a real (non-NULL, non -1) stage_id and NULL opening_index.
        let dir = tempfile::TempDir::new().unwrap();
        let rows: Vec<SolverStatsRow> = [(0, 10), (1, 20), (2, 30)]
            .into_iter()
            .map(|(stage, lp)| SolverStatsRow {
                iteration: Some(1),
                stage_id: Some(stage),
                lp_solves: lp,
                ..base_row("forward")
            })
            .collect();

        write_solver_stats(dir.path(), &rows).unwrap();

        let batch = read_first_batch(&dir.path().join("training/solver/iterations.parquet"));
        assert_eq!(batch.num_rows(), 3, "one forward row per stage");

        let opening_col = batch
            .column_by_name("opening_index")
            .unwrap()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        let stage_col = batch
            .column_by_name("stage_id")
            .unwrap()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        let lp_col = batch
            .column_by_name("lp_solves")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap();

        for row in 0..3 {
            assert!(
                opening_col.is_null(row),
                "forward row {row} NULL opening_index"
            );
            assert!(!stage_col.is_null(row), "forward row {row} keeps stage_id");
            assert_eq!(stage_col.value(row), i32::try_from(row).unwrap());
            assert_ne!(stage_col.value(row), -1, "no -1 stage sentinel");
        }
        assert_eq!(lp_col.value(0), 10);
        assert_eq!(lp_col.value(1), 20);
        assert_eq!(lp_col.value(2), 30);
    }
}
