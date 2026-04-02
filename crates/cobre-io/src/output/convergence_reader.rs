//! Reader for `training/convergence.parquet`.
//!
//! This module provides [`read_convergence_summary`], which reads the
//! convergence log written by the training pipeline and returns an
//! aggregated [`ConvergenceSummary`] suitable for display in post-run
//! reporting commands.

use std::path::Path;

use arrow::array::{Array, AsArray, RecordBatch};
use arrow::datatypes::{Float64Type, Int64Type};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use super::error::OutputError;

/// Aggregated summary extracted from `training/convergence.parquet`.
///
/// Produced by [`read_convergence_summary`]. Fields are summed or sampled
/// from the last row of the convergence table, so this struct is suitable
/// for display without holding a full per-iteration record list.
#[derive(Debug, Clone)]
pub struct ConvergenceSummary {
    /// Total number of LP solves summed across all iterations.
    pub total_lp_solves: u64,
    /// Total wall-clock time summed across all iterations (milliseconds).
    pub total_time_ms: u64,
    /// Lower bound value from the final iteration (0.0 when no rows).
    pub final_lower_bound: f64,
    /// Mean upper bound estimate from the final iteration (0.0 when no rows).
    pub final_upper_bound_mean: f64,
    /// Standard deviation of the upper bound from the final iteration (0.0 when no rows).
    pub final_upper_bound_std: f64,
    /// Relative gap from the final iteration, or `None` when no rows or gap was undefined.
    pub final_gap_percent: Option<f64>,
}

/// Read `training/convergence.parquet` and return an aggregated summary.
///
/// Reads all record batches from `path`, sums `lp_solves` and `time_total_ms`
/// across every row, and takes the bound and gap fields from the last row.
///
/// When the file contains zero rows, all numeric fields are zero and
/// `final_gap_percent` is `None`.
///
/// # Errors
///
/// - [`OutputError::IoError`] when `path` does not exist or cannot be opened.
/// - [`OutputError::SerializationError`] when the Parquet file is malformed or
///   the reader fails to iterate over batches.
/// - [`OutputError::SchemaError`] when a required column is absent from the file.
pub fn read_convergence_summary(path: &Path) -> Result<ConvergenceSummary, OutputError> {
    let file = std::fs::File::open(path).map_err(|e| OutputError::io(path, e))?;

    let reader = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| OutputError::SerializationError {
            entity: "convergence".to_string(),
            message: e.to_string(),
        })?
        .build()
        .map_err(|e| OutputError::SerializationError {
            entity: "convergence".to_string(),
            message: e.to_string(),
        })?;

    let mut totals = BatchTotals::default();

    for batch_result in reader {
        let batch = batch_result.map_err(|e| OutputError::SerializationError {
            entity: "convergence".to_string(),
            message: e.to_string(),
        })?;
        if batch.num_rows() > 0 {
            accumulate_batch(&batch, &mut totals)?;
        }
    }

    Ok(totals.into_summary())
}

// ── Private helpers ───────────────────────────────────────────────────────────

/// Mutable accumulator updated once per non-empty record batch.
#[derive(Default)]
struct BatchTotals {
    total_lp_solves: i64,
    total_time_ms: i64,
    final_lower_bound: f64,
    final_upper_bound_mean: f64,
    final_upper_bound_std: f64,
    final_gap_percent: Option<f64>,
    has_rows: bool,
}

impl BatchTotals {
    fn into_summary(self) -> ConvergenceSummary {
        if !self.has_rows {
            return ConvergenceSummary {
                total_lp_solves: 0,
                total_time_ms: 0,
                final_lower_bound: 0.0,
                final_upper_bound_mean: 0.0,
                final_upper_bound_std: 0.0,
                final_gap_percent: None,
            };
        }
        #[allow(clippy::cast_sign_loss)]
        ConvergenceSummary {
            total_lp_solves: self.total_lp_solves.max(0) as u64,
            total_time_ms: self.total_time_ms.max(0) as u64,
            final_lower_bound: self.final_lower_bound,
            final_upper_bound_mean: self.final_upper_bound_mean,
            final_upper_bound_std: self.final_upper_bound_std,
            final_gap_percent: self.final_gap_percent,
        }
    }
}

/// Extract an `Int64` column from `batch` by name, returning a schema error on failure.
fn get_i64_column<'a>(
    batch: &'a RecordBatch,
    name: &str,
) -> Result<&'a arrow::array::PrimitiveArray<Int64Type>, OutputError> {
    let col = batch
        .column_by_name(name)
        .ok_or_else(|| OutputError::SchemaError {
            file: "convergence.parquet".to_string(),
            column: name.to_string(),
            message: "column not found".to_string(),
        })?;
    col.as_primitive_opt::<Int64Type>()
        .ok_or_else(|| OutputError::SchemaError {
            file: "convergence.parquet".to_string(),
            column: name.to_string(),
            message: "expected Int64 column".to_string(),
        })
}

/// Extract a `Float64` column from `batch` by name, returning a schema error on failure.
fn get_f64_column<'a>(
    batch: &'a RecordBatch,
    name: &str,
) -> Result<&'a arrow::array::PrimitiveArray<Float64Type>, OutputError> {
    let col = batch
        .column_by_name(name)
        .ok_or_else(|| OutputError::SchemaError {
            file: "convergence.parquet".to_string(),
            column: name.to_string(),
            message: "column not found".to_string(),
        })?;
    col.as_primitive_opt::<Float64Type>()
        .ok_or_else(|| OutputError::SchemaError {
            file: "convergence.parquet".to_string(),
            column: name.to_string(),
            message: "expected Float64 column".to_string(),
        })
}

/// Update `totals` with data from a single non-empty record batch.
fn accumulate_batch(batch: &RecordBatch, totals: &mut BatchTotals) -> Result<(), OutputError> {
    let lp_solves_arr = get_i64_column(batch, "lp_solves")?;
    for i in 0..lp_solves_arr.len() {
        totals.total_lp_solves = totals
            .total_lp_solves
            .saturating_add(lp_solves_arr.value(i));
    }

    let time_arr = get_i64_column(batch, "time_total_ms")?;
    for i in 0..time_arr.len() {
        totals.total_time_ms = totals.total_time_ms.saturating_add(time_arr.value(i));
    }

    let last = batch.num_rows() - 1;

    totals.final_lower_bound = get_f64_column(batch, "lower_bound")?.value(last);
    totals.final_upper_bound_mean = get_f64_column(batch, "upper_bound_mean")?.value(last);
    totals.final_upper_bound_std = get_f64_column(batch, "upper_bound_std")?.value(last);

    let gap_arr = get_f64_column(batch, "gap_percent")?;
    // gap_percent is nullable: distinguish null from 0.0 using is_valid.
    totals.final_gap_percent = if gap_arr.is_valid(last) {
        Some(gap_arr.value(last))
    } else {
        None
    };

    totals.has_rows = true;
    Ok(())
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap
)]
mod tests {
    use super::*;
    use crate::output::{
        CutStatistics, IterationRecord, SimulationOutput, TrainingOutput, write_results,
    };

    fn make_iteration_record(iteration: u32, lp_solves: u32) -> IterationRecord {
        IterationRecord {
            iteration,
            lower_bound: f64::from(iteration) * 10.0,
            upper_bound_mean: f64::from(iteration) * 10.0 + 2.0,
            upper_bound_std: 0.5,
            gap_percent: Some(1.0),
            cuts_added: 5,
            cuts_removed: 1,
            cuts_active: 4,
            time_forward_ms: 100,
            time_backward_ms: 200,
            time_total_ms: 300,
            forward_passes: 4,
            lp_solves,
            time_forward_solve_ms: 100,
            time_forward_sample_ms: 0,
            time_backward_solve_ms: 200,
            time_backward_cut_ms: 0,
            time_cut_selection_ms: 0,
            time_mpi_allreduce_ms: 0,
            time_mpi_broadcast_ms: 0,
            time_io_write_ms: 0,
            time_state_exchange_ms: 0,
            time_cut_batch_build_ms: 0,
            time_rayon_overhead_ms: 0,
            time_overhead_ms: 0,
            solve_time_ms: 0.0,
        }
    }

    fn make_training_output(records: Vec<IterationRecord>) -> TrainingOutput {
        let n = records.len() as u32;
        TrainingOutput {
            convergence_records: records,
            final_lower_bound: 99.5,
            final_upper_bound: Some(101.0),
            final_gap_percent: Some(1.51),
            iterations_completed: n,
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

    fn make_system() -> cobre_core::System {
        cobre_core::SystemBuilder::new()
            .build()
            .expect("empty system must be valid")
    }

    fn make_config() -> crate::Config {
        use crate::config::{
            CheckpointingConfig, CutSelectionConfig, EstimationConfig, ExportsConfig,
            InflowNonNegativityConfig, ModelingConfig, PolicyConfig, PolicyMode, SimulationConfig,
            SimulationSamplingConfig, StoppingRuleConfig, TrainingConfig, TrainingSolverConfig,
            UpperBoundEvaluationConfig,
        };
        crate::Config {
            schema: None,
            modeling: ModelingConfig {
                inflow_non_negativity: InflowNonNegativityConfig::default(),
            },
            training: TrainingConfig {
                enabled: true,
                seed: None,
                forward_passes: Some(4),
                stopping_rules: Some(vec![StoppingRuleConfig::IterationLimit { limit: 10 }]),
                stopping_mode: "any".to_string(),
                cut_formulation: None,
                forward_pass: None,
                cut_selection: CutSelectionConfig::default(),
                solver: TrainingSolverConfig::default(),
            },
            upper_bound_evaluation: UpperBoundEvaluationConfig::default(),
            policy: PolicyConfig {
                path: "./policy".to_string(),
                mode: PolicyMode::Fresh,
                validate_compatibility: true,
                checkpointing: CheckpointingConfig::default(),
            },
            simulation: SimulationConfig {
                enabled: false,
                num_scenarios: 0,
                policy_type: "outer".to_string(),
                output_path: None,
                output_mode: None,
                io_channel_capacity: 64,
                sampling_scheme: SimulationSamplingConfig::default(),
            },
            exports: ExportsConfig::default(),
            estimation: EstimationConfig::default(),
        }
    }

    fn write_convergence(
        tmp: &tempfile::TempDir,
        records: Vec<IterationRecord>,
    ) -> std::path::PathBuf {
        let training = make_training_output(records);
        write_results(
            tmp.path(),
            &training,
            None::<&SimulationOutput>,
            &make_system(),
            &make_config(),
        )
        .expect("write_results must succeed");
        tmp.path().join("training/convergence.parquet")
    }

    // ── Acceptance criteria ───────────────────────────────────────────────────

    #[test]
    fn read_convergence_summary_from_real_parquet() {
        let tmp = tempfile::tempdir().unwrap();
        // Three records with lp_solves = [40, 50, 60].
        let records = vec![
            make_iteration_record(1, 40),
            make_iteration_record(2, 50),
            make_iteration_record(3, 60),
        ];
        let path = write_convergence(&tmp, records);

        let summary = read_convergence_summary(&path).expect("read must succeed");

        assert_eq!(
            summary.total_lp_solves, 150,
            "total_lp_solves must equal sum of all records: 40+50+60=150"
        );
        // The last record has iteration=3, lower_bound = 3*10.0 = 30.0.
        assert_eq!(
            summary.final_lower_bound, 30.0,
            "final_lower_bound must come from the last row"
        );
        // total_time_ms: 3 records × 300 ms each = 900 ms.
        assert_eq!(
            summary.total_time_ms, 900,
            "total_time_ms must be sum across all rows"
        );
        // gap_percent is Some(1.0) for every record; last row should be Some(1.0).
        assert_eq!(
            summary.final_gap_percent,
            Some(1.0),
            "final_gap_percent must come from the last row"
        );
    }

    #[test]
    fn read_convergence_summary_empty_file() {
        let tmp = tempfile::tempdir().unwrap();
        let path = write_convergence(&tmp, vec![]);

        let summary = read_convergence_summary(&path).expect("read must succeed on empty file");

        assert_eq!(
            summary.total_lp_solves, 0,
            "total_lp_solves must be 0 for empty file"
        );
        assert_eq!(
            summary.total_time_ms, 0,
            "total_time_ms must be 0 for empty file"
        );
        assert_eq!(
            summary.final_lower_bound, 0.0,
            "final_lower_bound must be 0.0 for empty file"
        );
        assert_eq!(
            summary.final_upper_bound_mean, 0.0,
            "final_upper_bound_mean must be 0.0 for empty file"
        );
        assert_eq!(
            summary.final_upper_bound_std, 0.0,
            "final_upper_bound_std must be 0.0 for empty file"
        );
        assert!(
            summary.final_gap_percent.is_none(),
            "final_gap_percent must be None for empty file"
        );
    }

    #[test]
    fn read_convergence_summary_missing_file() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("nonexistent.parquet");

        let result = read_convergence_summary(&path);

        assert!(
            matches!(result, Err(OutputError::IoError { .. })),
            "missing file must return OutputError::IoError, got: {result:?}"
        );
    }

    #[test]
    fn read_convergence_summary_single_row() {
        let tmp = tempfile::tempdir().unwrap();
        let records = vec![make_iteration_record(1, 40)];
        let path = write_convergence(&tmp, records);

        let summary = read_convergence_summary(&path).expect("read must succeed");

        assert_eq!(summary.total_lp_solves, 40);
        assert_eq!(summary.total_time_ms, 300);
        assert_eq!(summary.final_lower_bound, 10.0);
        assert_eq!(summary.final_upper_bound_mean, 12.0);
        assert_eq!(summary.final_upper_bound_std, 0.5);
        assert_eq!(summary.final_gap_percent, Some(1.0));
    }
}
