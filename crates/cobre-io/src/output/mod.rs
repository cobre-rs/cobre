//! Output writers for simulation results and policy files.
//!
//! This module provides Hive-partitioned Parquet writers for simulation pipeline
//! output and `FlatBuffers` policy writers.
//!
//! The top-level entry point is [`write_results`], which mirrors [`crate::load_case`]:
//! it accepts aggregate result types and writes all output artifacts to the
//! specified directory.

pub mod convergence_reader;
pub mod dictionary;
pub mod error;
pub mod hydro_models;
pub mod manifest;
pub mod parquet_config;
pub mod policy;
pub mod provenance;
pub mod results_writer;
pub mod scaling_report;
pub(crate) mod schemas;
pub mod simulation_writer;
pub mod solver_stats_writer;
pub mod stochastic;
pub mod training_writer;

pub use convergence_reader::{ConvergenceSummary, read_convergence_summary};
pub use dictionary::write_dictionaries;
pub use error::OutputError;
pub use hydro_models::write_fpha_hyperplanes;
pub use manifest::{
    MetadataConfiguration, MetadataConvergence, MetadataCuts, MetadataIterations,
    MetadataProblemDimensions, MetadataScenarios, MpiInfo, OutputContext, SimulationMetadata,
    TrainingMetadata, get_hostname, now_iso8601, read_simulation_metadata, read_training_metadata,
    write_simulation_metadata, write_training_metadata,
};
pub use parquet_config::ParquetWriterConfig;
pub use provenance::write_provenance_report;
pub use results_writer::{write_results, write_simulation_results, write_training_results};
pub use scaling_report::write_scaling_report;
pub use simulation_writer::SimulationParquetWriter;
pub use solver_stats_writer::{SolverStatsRow, write_simulation_solver_stats, write_solver_stats};
pub use stochastic::{
    FittingReductionEntry, FittingReport, HydroFittingEntry, write_correlation_json,
    write_fitting_report, write_inflow_ar_coefficients, write_inflow_seasonal_stats,
    write_load_seasonal_stats, write_noise_openings,
};
pub use training_writer::{TrainingParquetWriter, write_cut_selection_records};

/// One row of convergence data corresponding to a single training iteration.
///
/// Instances are accumulated in [`TrainingOutput::convergence_records`] and are
/// written verbatim to `training/convergence.parquet` by the convergence writer.
#[derive(Debug, Clone)]
pub struct IterationRecord {
    /// Sequential iteration number (1-based).
    pub iteration: u32,

    /// Lower bound on the optimal value at the end of this iteration.
    pub lower_bound: f64,

    /// Mean upper bound estimate across all forward-pass scenarios.
    pub upper_bound_mean: f64,

    /// Standard deviation of the upper bound estimate across scenarios.
    pub upper_bound_std: f64,

    /// Relative gap between upper and lower bounds as a percentage, if defined.
    ///
    /// `None` when the lower bound is zero or negative (gap is ill-defined).
    pub gap_percent: Option<f64>,

    /// Number of cuts added to the cut pool during this iteration.
    pub cuts_added: u32,

    /// Number of cuts removed from the cut pool during this iteration.
    pub cuts_removed: u32,

    /// Total number of active cuts in the pool after this iteration.
    pub cuts_active: u32,

    /// Wall-clock time spent in the forward pass for this iteration (ms).
    pub time_forward_ms: u64,

    /// Wall-clock time spent in the backward pass for this iteration (ms).
    pub time_backward_ms: u64,

    /// Total wall-clock time for this iteration (ms).
    pub time_total_ms: u64,

    /// Wall-clock time for the forward solve phase (ms).
    ///
    /// Maps to `forward_solve_ms` in `training/timing/iterations.parquet`.
    pub time_forward_solve_ms: u64,

    /// Wall-clock time for forward scenario sampling (ms).
    ///
    /// Currently 0 — not measured separately from the forward solve.
    /// Maps to `forward_sample_ms` in `training/timing/iterations.parquet`.
    pub time_forward_sample_ms: u64,

    /// Wall-clock time for the backward solve phase (ms).
    ///
    /// Maps to `backward_solve_ms` in `training/timing/iterations.parquet`.
    pub time_backward_solve_ms: u64,

    /// Wall-clock time for cut generation within the backward pass (ms).
    ///
    /// Currently 0 — not separated from backward solve time.
    /// Maps to `backward_cut_ms` in `training/timing/iterations.parquet`.
    pub time_backward_cut_ms: u64,

    /// Wall-clock time for the cut selection phase (ms).
    ///
    /// Maps to `cut_selection_ms` in `training/timing/iterations.parquet`.
    pub time_cut_selection_ms: u64,

    /// Wall-clock time for MPI allreduce (forward bound synchronization) (ms).
    ///
    /// Maps to `mpi_allreduce_ms` in `training/timing/iterations.parquet`.
    pub time_mpi_allreduce_ms: u64,

    /// Wall-clock time for MPI broadcast (cut synchronization) (ms).
    ///
    /// Maps to `mpi_broadcast_ms` in `training/timing/iterations.parquet`.
    pub time_mpi_broadcast_ms: u64,

    /// Wall-clock time for I/O writes in this iteration (ms).
    ///
    /// Currently 0 — I/O timing is not tracked at the iteration level.
    /// Maps to `io_write_ms` in `training/timing/iterations.parquet`.
    pub time_io_write_ms: u64,

    /// Wall-clock time for state exchange (`allgatherv`) in the backward pass (ms).
    ///
    /// Maps to `state_exchange_ms` in `training/timing/iterations.parquet`.
    pub time_state_exchange_ms: u64,

    /// Wall-clock time for cut batch assembly in the backward pass (ms).
    ///
    /// Maps to `cut_batch_build_ms` in `training/timing/iterations.parquet`.
    pub time_cut_batch_build_ms: u64,

    /// Estimated rayon barrier + scheduling overhead in the backward pass (ms).
    ///
    /// Maps to `rayon_overhead_ms` in `training/timing/iterations.parquet`.
    pub time_rayon_overhead_ms: u64,

    /// Residual wall-clock time not attributed to any specific phase (ms).
    ///
    /// Computed as `time_total_ms - sum(all other time_* fields)`.
    /// Maps to `overhead_ms` in `training/timing/iterations.parquet`.
    pub time_overhead_ms: u64,

    /// Number of forward-pass scenarios solved in this iteration.
    pub forward_passes: u32,

    /// Total number of LP solves (across all stages and passes) in this iteration.
    pub lp_solves: u32,

    /// Cumulative LP solve wall-clock time for this iteration, in milliseconds.
    pub solve_time_ms: f64,
}

/// Summary statistics for the cut pool at the end of a training run.
///
/// Carried inside [`TrainingOutput`] and written to `training/timing/cut_stats.parquet`.
#[derive(Debug, Clone)]
pub struct CutStatistics {
    /// Total number of cuts generated over the entire training run.
    pub total_generated: u64,

    /// Number of cuts still active in the pool at the end of training.
    pub total_active: u64,

    /// Highest number of active cuts observed at any point during training.
    pub peak_active: u64,
}

/// One row in `training/cut_selection/iterations.parquet`.
///
/// Represents per-stage cut selection statistics for a single iteration.
/// Only populated when cut selection is enabled.
#[derive(Debug, Clone)]
pub struct CutSelectionRecord {
    /// Iteration number (1-based).
    pub iteration: u32,
    /// 0-based stage index.
    pub stage: u32,
    /// Total cuts ever generated at this stage.
    pub cuts_populated: u32,
    /// Active cuts before selection ran.
    pub cuts_active_before: u32,
    /// Cuts deactivated by selection at this stage.
    pub cuts_deactivated: u32,
    /// Active cuts after selection.
    pub cuts_active_after: u32,
    /// Wall-clock time for selection at this stage, in milliseconds.
    pub selection_time_ms: f64,
}

/// Aggregate type carrying all training data needed for output writing.
///
/// Constructed by the solver after training completes and passed to
/// [`write_results`]. All convergence records and summary statistics are
/// held here so the writer can read them without contacting the solver.
#[derive(Debug, Clone)]
pub struct TrainingOutput {
    /// Ordered convergence records — one entry per completed iteration.
    pub convergence_records: Vec<IterationRecord>,

    /// Lower bound value reported after the final iteration.
    pub final_lower_bound: f64,

    /// Upper bound value reported after the final iteration, if available.
    ///
    /// `None` when no upper-bound evaluation was performed.
    pub final_upper_bound: Option<f64>,

    /// Relative gap between final upper and lower bounds as a percentage.
    ///
    /// `None` when the lower bound is zero/negative or `final_upper_bound` is `None`.
    pub final_gap_percent: Option<f64>,

    /// Number of iterations completed before the stopping condition was triggered.
    pub iterations_completed: u32,

    /// `true` when training converged within the configured tolerance.
    pub converged: bool,

    /// Human-readable description of the rule that terminated training.
    pub termination_reason: String,

    /// Total elapsed wall-clock time for the entire training run (ms).
    pub total_time_ms: u64,

    /// Summary cut pool statistics for the run.
    pub cut_stats: CutStatistics,

    /// Per-stage cut selection records for Parquet output.
    ///
    /// Empty when cut selection is disabled. When non-empty, written to
    /// `training/cut_selection/iterations.parquet`.
    pub cut_selection_records: Vec<CutSelectionRecord>,
}

/// Aggregate type carrying simulation completion data for output writing.
///
/// Constructed by the simulation pipeline after it completes and optionally
/// passed to [`write_results`]. When `None` is supplied, the simulation
/// output directory is still created (ready for future use), but no
/// simulation artifacts are written.
#[derive(Debug, Clone)]
pub struct SimulationOutput {
    /// Total number of scenarios dispatched for simulation.
    pub n_scenarios: u32,

    /// Number of scenarios that completed without error.
    pub completed: u32,

    /// Number of scenarios that failed during simulation.
    pub failed: u32,

    /// Total elapsed wall-clock time for the simulation run (ms).
    pub total_time_ms: u64,

    /// Hive partition paths written by the simulation writer.
    ///
    /// Each element is a relative path string such as
    /// `"simulation/costs/year=2030/month=01/part-00.parquet"`.
    pub partitions_written: Vec<String>,
}

impl SimulationOutput {
    /// Combine multiple per-rank simulation outputs into a single aggregate.
    ///
    /// Merge rules:
    /// - `n_scenarios`: sum across all outputs.
    /// - `completed`: sum across all outputs.
    /// - `failed`: sum across all outputs.
    /// - `total_time_ms`: max across all outputs (wall-clock = slowest rank).
    /// - `partitions_written`: concatenation of all outputs' partitions, sorted
    ///   for deterministic ordering regardless of input order.
    ///
    /// Returns a zeroed [`SimulationOutput`] with empty partitions when the
    /// input slice is empty.
    #[must_use]
    pub fn merge(outputs: &[Self]) -> Self {
        if outputs.is_empty() {
            return Self {
                n_scenarios: 0,
                completed: 0,
                failed: 0,
                total_time_ms: 0,
                partitions_written: Vec::new(),
            };
        }

        let n_scenarios = outputs.iter().map(|o| o.n_scenarios).sum();
        let completed = outputs.iter().map(|o| o.completed).sum();
        let failed = outputs.iter().map(|o| o.failed).sum();
        let total_time_ms = outputs.iter().map(|o| o.total_time_ms).max().unwrap_or(0);

        let mut partitions_written: Vec<String> = outputs
            .iter()
            .flat_map(|o| o.partitions_written.iter().cloned())
            .collect();
        partitions_written.sort();

        Self {
            n_scenarios,
            completed,
            failed,
            total_time_ms,
            partitions_written,
        }
    }
}

/// Write all output artifacts to `output_dir`.
///
/// This function is the symmetric counterpart of [`crate::load_case`]: it
/// accepts the aggregate result types produced by the solver and writes
/// every output artifact — convergence tables, dictionaries, manifests,
/// metadata, and optionally simulation artifacts — to the specified root
/// directory.
///
/// The function creates the complete directory structure, delegates to
/// sub-writers in the order required by the output specification, and
/// writes `_SUCCESS` marker files after all artifacts are complete.
///
/// # Directory layout produced
///
/// ```text
/// output_dir/
///   training/
///     _SUCCESS
///     metadata.json
///     convergence.parquet
///     dictionaries/
///       codes.json
///       entities.csv
///       variables.csv
///       bounds.parquet
///       state_dictionary.json
///     timing/
///       iterations.parquet
///   simulation/
///     _SUCCESS              (only when simulation_output is Some)
///     metadata.json         (only when simulation_output is Some)
/// ```
///
/// # Parameters
///
/// - `output_dir` — root directory that will receive all output artifacts.
/// - `training_output` — convergence records and summary statistics from
///   the completed training run.
/// - `simulation_output` — optional simulation results; `None` is allowed
///   and still causes the `simulation/` directory to be created.
/// - `system` — the loaded system, used by dictionary writers to enumerate
///   entity identifiers.
/// - `config` — run configuration supplying writer settings and metadata.
///
/// # Errors
///
/// - [`OutputError::IoError`] — directory creation or file write failed.
/// - [`OutputError::SerializationError`] — Arrow batch construction failed.
/// - [`OutputError::ManifestError`] — JSON serialization failed.
///
/// # Examples
///
/// ```no_run
/// use cobre_io::{write_results, TrainingOutput, CutStatistics};
/// use std::path::Path;
///
/// # fn main() -> Result<(), cobre_io::OutputError> {
/// # let system = unimplemented!();
/// # let config = unimplemented!();
/// let training = TrainingOutput {
///     convergence_records: Vec::new(),
///     final_lower_bound: 42.0,
///     final_upper_bound: Some(44.0),
///     final_gap_percent: Some(4.76),
///     iterations_completed: 10,
///     converged: true,
///     termination_reason: "gap tolerance reached".to_string(),
///     total_time_ms: 3_000,
///     cut_stats: CutStatistics {
///         total_generated: 200,
///         total_active: 80,
///         peak_active: 95,
///     },
///     cut_selection_records: Vec::new(),
/// };
/// write_results(Path::new("/tmp/out"), &training, None, system, config)?;
/// # Ok(())
/// # }
/// ```
#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::float_cmp,
    clippy::cast_possible_truncation
)]
mod tests {
    use super::*;

    #[test]
    fn training_output_construction_and_field_access() {
        let records: Vec<IterationRecord> = (1..=5)
            .map(|i| IterationRecord {
                iteration: i,
                lower_bound: 1.0,
                upper_bound_mean: 2.0,
                upper_bound_std: 0.1,
                gap_percent: Some(50.0),
                cuts_added: 10,
                cuts_removed: 2,
                cuts_active: 8,
                time_forward_ms: 100,
                time_backward_ms: 200,
                time_total_ms: 300,
                forward_passes: 4,
                lp_solves: 40,
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
            })
            .collect();
        let output = TrainingOutput {
            convergence_records: records,
            final_lower_bound: 50.0,
            final_upper_bound: Some(52.0),
            final_gap_percent: Some(3.85),
            iterations_completed: 5,
            converged: true,
            termination_reason: "relative gap < 1%".to_string(),
            total_time_ms: 12_000,
            cut_stats: CutStatistics {
                total_generated: 300,
                total_active: 120,
                peak_active: 150,
            },
            cut_selection_records: vec![],
        };

        assert_eq!(output.convergence_records.len(), 5);
        assert_eq!(output.final_lower_bound, 50.0);
        assert_eq!(output.final_upper_bound, Some(52.0));
        assert_eq!(output.final_gap_percent, Some(3.85));
        assert_eq!(output.iterations_completed, 5);
        assert!(output.converged);
        assert_eq!(output.termination_reason, "relative gap < 1%");
        assert_eq!(output.total_time_ms, 12_000);
        assert_eq!(output.cut_stats.total_generated, 300);
        assert_eq!(output.cut_stats.total_active, 120);
        assert_eq!(output.cut_stats.peak_active, 150);
    }

    #[test]
    fn iteration_record_construction_and_field_access() {
        let record = IterationRecord {
            iteration: 7,
            lower_bound: 10.5,
            upper_bound_mean: 11.0,
            upper_bound_std: 0.25,
            gap_percent: Some(4.55),
            cuts_added: 15,
            cuts_removed: 3,
            cuts_active: 42,
            time_forward_ms: 150,
            time_backward_ms: 250,
            time_total_ms: 400,
            forward_passes: 8,
            lp_solves: 80,
            time_forward_solve_ms: 150,
            time_forward_sample_ms: 0,
            time_backward_solve_ms: 250,
            time_backward_cut_ms: 0,
            time_cut_selection_ms: 5,
            time_mpi_allreduce_ms: 3,
            time_mpi_broadcast_ms: 2,
            time_io_write_ms: 0,
            time_state_exchange_ms: 0,
            time_cut_batch_build_ms: 0,
            time_rayon_overhead_ms: 0,
            time_overhead_ms: 400u64.saturating_sub(150 + 250 + 5 + 3 + 2),
            solve_time_ms: 0.0,
        };

        assert_eq!(record.iteration, 7);
        assert_eq!(record.lower_bound, 10.5);
        assert_eq!(record.upper_bound_mean, 11.0);
        assert_eq!(record.upper_bound_std, 0.25);
        assert_eq!(record.gap_percent, Some(4.55));
        assert_eq!(record.cuts_added, 15);
        assert_eq!(record.cuts_removed, 3);
        assert_eq!(record.cuts_active, 42);
        assert_eq!(record.time_forward_ms, 150);
        assert_eq!(record.time_backward_ms, 250);
        assert_eq!(record.time_total_ms, 400);
        assert_eq!(record.forward_passes, 8);
        assert_eq!(record.lp_solves, 80);
        assert_eq!(record.time_forward_solve_ms, 150);
        assert_eq!(record.time_forward_sample_ms, 0);
        assert_eq!(record.time_backward_solve_ms, 250);
        assert_eq!(record.time_backward_cut_ms, 0);
        assert_eq!(record.time_cut_selection_ms, 5);
        assert_eq!(record.time_mpi_allreduce_ms, 3);
        assert_eq!(record.time_mpi_broadcast_ms, 2);
        assert_eq!(record.time_io_write_ms, 0);
    }

    #[test]
    fn simulation_output_construction_and_field_access() {
        let output = SimulationOutput {
            n_scenarios: 100,
            completed: 100,
            failed: 0,
            total_time_ms: 3_200,
            partitions_written: vec![
                "simulation/costs/year=2030/part-00.parquet".to_string(),
                "simulation/costs/year=2031/part-00.parquet".to_string(),
            ],
        };

        assert_eq!(output.n_scenarios, 100);
        assert_eq!(output.completed, 100);
        assert_eq!(output.failed, 0);
        assert_eq!(output.total_time_ms, 3_200);
        assert_eq!(output.partitions_written.len(), 2);
    }

    #[test]
    fn cut_statistics_construction() {
        let stats = CutStatistics {
            total_generated: 500,
            total_active: 200,
            peak_active: 250,
        };

        assert_eq!(stats.total_generated, 500);
        assert_eq!(stats.total_active, 200);
        assert_eq!(stats.peak_active, 250);
    }

    #[test]
    fn test_merge_empty_slice() {
        let merged = SimulationOutput::merge(&[]);
        assert_eq!(merged.n_scenarios, 0);
        assert_eq!(merged.completed, 0);
        assert_eq!(merged.failed, 0);
        assert_eq!(merged.total_time_ms, 0);
        assert!(merged.partitions_written.is_empty());
    }

    #[test]
    fn test_merge_single_output() {
        let output = SimulationOutput {
            n_scenarios: 5,
            completed: 4,
            failed: 1,
            total_time_ms: 1000,
            partitions_written: vec!["simulation/costs/scenario_id=0000/data.parquet".to_string()],
        };
        let merged = SimulationOutput::merge(std::slice::from_ref(&output));
        assert_eq!(merged.n_scenarios, 5);
        assert_eq!(merged.completed, 4);
        assert_eq!(merged.failed, 1);
        assert_eq!(merged.total_time_ms, 1000);
        assert_eq!(merged.partitions_written, output.partitions_written);
    }

    #[test]
    fn test_merge_two_outputs() {
        let a = SimulationOutput {
            n_scenarios: 3,
            completed: 3,
            failed: 0,
            total_time_ms: 500,
            partitions_written: vec![
                "simulation/costs/scenario_id=0000/data.parquet".to_string(),
                "simulation/costs/scenario_id=0001/data.parquet".to_string(),
            ],
        };
        let b = SimulationOutput {
            n_scenarios: 2,
            completed: 1,
            failed: 1,
            total_time_ms: 800,
            partitions_written: vec!["simulation/costs/scenario_id=0002/data.parquet".to_string()],
        };
        let merged = SimulationOutput::merge(&[a, b]);
        assert_eq!(merged.n_scenarios, 5);
        assert_eq!(merged.completed, 4);
        assert_eq!(merged.failed, 1);
        // total_time_ms uses max, not sum
        assert_eq!(merged.total_time_ms, 800);
        assert_eq!(merged.partitions_written.len(), 3);
    }

    #[test]
    fn test_merge_partitions_sorted() {
        let a = SimulationOutput {
            n_scenarios: 1,
            completed: 1,
            failed: 0,
            total_time_ms: 100,
            partitions_written: vec![
                "simulation/hydros/scenario_id=0002/data.parquet".to_string(),
                "simulation/costs/scenario_id=0002/data.parquet".to_string(),
            ],
        };
        let b = SimulationOutput {
            n_scenarios: 1,
            completed: 1,
            failed: 0,
            total_time_ms: 200,
            partitions_written: vec![
                "simulation/costs/scenario_id=0001/data.parquet".to_string(),
                "simulation/hydros/scenario_id=0001/data.parquet".to_string(),
            ],
        };
        let merged = SimulationOutput::merge(&[a, b]);
        // Partitions must be sorted regardless of input order
        let expected = vec![
            "simulation/costs/scenario_id=0001/data.parquet".to_string(),
            "simulation/costs/scenario_id=0002/data.parquet".to_string(),
            "simulation/hydros/scenario_id=0001/data.parquet".to_string(),
            "simulation/hydros/scenario_id=0002/data.parquet".to_string(),
        ];
        assert_eq!(merged.partitions_written, expected);
    }
}
