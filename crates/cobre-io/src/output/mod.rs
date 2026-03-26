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
    ManifestChecksum, ManifestConvergence, ManifestCuts, ManifestIterations, ManifestMpiInfo,
    ManifestScenarios, MetadataConfigSnapshot, MetadataDataIntegrity, MetadataEnvironment,
    MetadataPerformanceSummary, MetadataProblemDimensions, MetadataRunInfo, SimulationManifest,
    TrainingManifest, TrainingMetadata, read_simulation_manifest, read_training_manifest,
    write_metadata, write_simulation_manifest, write_training_manifest,
};
pub use parquet_config::ParquetWriterConfig;
pub use scaling_report::write_scaling_report;
pub use simulation_writer::SimulationParquetWriter;
pub use solver_stats_writer::{SolverStatsRow, write_simulation_solver_stats, write_solver_stats};
pub use stochastic::{
    FittingReductionEntry, FittingReport, HydroFittingEntry, write_correlation_json,
    write_fitting_report, write_inflow_ar_coefficients, write_inflow_seasonal_stats,
    write_load_seasonal_stats, write_noise_openings,
};
pub use training_writer::{TrainingParquetWriter, write_cut_selection_records};

use cobre_core::System;
use std::path::Path;

use crate::Config;

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
///     _manifest.json
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
///     _manifest.json        (only when simulation_output is Some)
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
/// Write training artifacts to the output directory.
///
/// Creates the `training/` subdirectory structure (dictionaries, convergence
/// Parquet, timing, manifest, metadata) and writes the `training/_SUCCESS`
/// marker on completion. Also creates an empty `simulation/` directory so
/// downstream code can unconditionally write into it.
///
/// This function is one half of the split from [`write_results`]; it can be
/// called independently to persist training outputs before simulation starts.
#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
pub fn write_training_results(
    output_dir: &Path,
    training_output: &TrainingOutput,
    system: &System,
    config: &Config,
) -> Result<(), OutputError> {
    std::fs::create_dir_all(output_dir.join("training/dictionaries"))
        .map_err(|e| OutputError::io(output_dir.join("training/dictionaries"), e))?;
    std::fs::create_dir_all(output_dir.join("training/timing"))
        .map_err(|e| OutputError::io(output_dir.join("training/timing"), e))?;
    std::fs::create_dir_all(output_dir.join("simulation"))
        .map_err(|e| OutputError::io(output_dir.join("simulation"), e))?;

    write_dictionaries(&output_dir.join("training/dictionaries"), system, config)?;

    let parquet_config = ParquetWriterConfig::default();
    let writer = TrainingParquetWriter::new(output_dir, &parquet_config)?;
    writer.write(training_output)?;

    let converged_at = training_output
        .converged
        .then_some(training_output.iterations_completed);
    let training_manifest = TrainingManifest {
        version: "2.0.0".to_string(),
        status: "complete".to_string(),
        started_at: None,
        completed_at: None,
        iterations: ManifestIterations {
            max_iterations: None,
            completed: training_output.iterations_completed,
            converged_at,
        },
        convergence: ManifestConvergence {
            achieved: training_output.converged,
            final_gap_percent: training_output.final_gap_percent,
            termination_reason: training_output.termination_reason.clone(),
        },
        cuts: ManifestCuts {
            total_generated: training_output.cut_stats.total_generated,
            total_active: training_output.cut_stats.total_active,
            peak_active: training_output.cut_stats.peak_active,
        },
        checksum: None,
        mpi_info: ManifestMpiInfo::default(),
    };
    write_training_manifest(
        &output_dir.join("training/_manifest.json"),
        &training_manifest,
    )?;

    let training_metadata = TrainingMetadata {
        version: "2.0.0".to_string(),
        run_info: MetadataRunInfo {
            run_id: "not-implemented".to_string(),
            started_at: None,
            completed_at: None,
            duration_seconds: Some(training_output.total_time_ms as f64 / 1_000.0),
            cobre_version: env!("CARGO_PKG_VERSION").to_string(),
            solver: None,
            solver_version: None,
            hostname: None,
            user: None,
        },
        configuration_snapshot: MetadataConfigSnapshot {
            seed: config.training.seed,
            forward_passes: config.training.forward_passes,
            stopping_mode: config.training.stopping_mode.clone(),
            policy_mode: config.policy.mode.clone(),
        },
        problem_dimensions: MetadataProblemDimensions {
            num_stages: system.n_stages() as u32,
            num_hydros: system.n_hydros() as u32,
            num_thermals: system.n_thermals() as u32,
            num_buses: system.n_buses() as u32,
            num_lines: system.n_lines() as u32,
        },
        performance_summary: None,
        data_integrity: None,
        environment: MetadataEnvironment {
            mpi_implementation: None,
            mpi_version: None,
            num_ranks: None,
            cpus_per_rank: None,
            memory_per_rank_gb: None,
        },
    };
    write_metadata(
        &output_dir.join("training/metadata.json"),
        &training_metadata,
    )?;

    std::fs::write(output_dir.join("training/_SUCCESS"), b"")
        .map_err(|e| OutputError::io(output_dir.join("training/_SUCCESS"), e))?;

    Ok(())
}

/// Write simulation artifacts to the output directory.
///
/// Writes `simulation/_manifest.json` and the `simulation/_SUCCESS` marker.
/// The `simulation/` directory must already exist (created by
/// [`write_training_results`]).
///
/// # Errors
///
/// Returns [`OutputError`] if manifest serialization or file I/O fails.
#[allow(clippy::cast_precision_loss)]
pub fn write_simulation_results(
    output_dir: &Path,
    simulation_output: &SimulationOutput,
) -> Result<(), OutputError> {
    let sim_manifest = SimulationManifest {
        version: "2.0.0".to_string(),
        status: "complete".to_string(),
        started_at: None,
        completed_at: None,
        duration_seconds: Some(simulation_output.total_time_ms as f64 / 1_000.0),
        scenarios: ManifestScenarios {
            total: simulation_output.n_scenarios,
            completed: simulation_output.completed,
            failed: simulation_output.failed,
        },
        partitions_written: simulation_output.partitions_written.clone(),
        checksum: None,
        mpi_info: ManifestMpiInfo::default(),
    };
    write_simulation_manifest(&output_dir.join("simulation/_manifest.json"), &sim_manifest)?;

    std::fs::write(output_dir.join("simulation/_SUCCESS"), b"")
        .map_err(|e| OutputError::io(output_dir.join("simulation/_SUCCESS"), e))?;

    Ok(())
}

/// Write all output artifacts (training + simulation) to the output directory.
///
/// Convenience wrapper that calls [`write_training_results`] followed by
/// [`write_simulation_results`] (when simulation output is present).
/// Retained for backward compatibility.
///
/// # Errors
///
/// Returns [`OutputError`] if any file I/O or serialization step fails.
pub fn write_results(
    output_dir: &Path,
    training_output: &TrainingOutput,
    simulation_output: Option<&SimulationOutput>,
    system: &System,
    config: &Config,
) -> Result<(), OutputError> {
    write_training_results(output_dir, training_output, system, config)?;
    if let Some(sim) = simulation_output {
        write_simulation_results(output_dir, sim)?;
    }
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

    fn make_iteration_record(iteration: u32) -> IterationRecord {
        IterationRecord {
            iteration,
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
        }
    }

    fn make_training_output(n_records: usize) -> TrainingOutput {
        let records = (1..=n_records as u32).map(make_iteration_record).collect();
        TrainingOutput {
            convergence_records: records,
            final_lower_bound: 99.5,
            final_upper_bound: Some(101.0),
            final_gap_percent: Some(1.51),
            iterations_completed: n_records as u32,
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

    #[test]
    fn write_results_creates_training_directories() {
        let tmp = tempfile::tempdir().unwrap();
        let training = make_training_output(0);

        write_results(tmp.path(), &training, None, &make_system(), &make_config())
            .expect("write_results must succeed");

        assert!(tmp.path().join("training").is_dir(), "training/ must exist");
        assert!(
            tmp.path().join("training/dictionaries").is_dir(),
            "training/dictionaries/ must exist"
        );
        assert!(
            tmp.path().join("training/timing").is_dir(),
            "training/timing/ must exist"
        );
    }

    #[test]
    fn write_results_creates_simulation_directory() {
        let tmp = tempfile::tempdir().unwrap();
        let training = make_training_output(0);

        // Passing None for simulation_output must still create simulation/.
        write_results(tmp.path(), &training, None, &make_system(), &make_config())
            .expect("write_results must succeed with simulation_output = None");

        assert!(
            tmp.path().join("simulation").is_dir(),
            "simulation/ must exist even when simulation_output is None"
        );
    }

    #[test]
    fn write_results_returns_ok_on_success() {
        let tmp = tempfile::tempdir().unwrap();
        let training = make_training_output(3);
        let simulation = SimulationOutput {
            n_scenarios: 10,
            completed: 10,
            failed: 0,
            total_time_ms: 1_500,
            partitions_written: vec!["simulation/costs/part-00.parquet".to_string()],
        };

        let result = write_results(
            tmp.path(),
            &training,
            Some(&simulation),
            &make_system(),
            &make_config(),
        );
        assert!(
            result.is_ok(),
            "write_results must return Ok(()) on success"
        );
    }

    #[test]
    fn training_output_construction_and_field_access() {
        let records: Vec<IterationRecord> = (1..=5).map(make_iteration_record).collect();
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
    fn write_results_creates_success_marker() {
        let tmp = tempfile::tempdir().unwrap();
        let training = make_training_output(0);

        write_results(tmp.path(), &training, None, &make_system(), &make_config())
            .expect("write_results must succeed");

        assert!(
            tmp.path().join("training/_SUCCESS").is_file(),
            "training/_SUCCESS must exist after write_results"
        );
    }

    #[test]
    fn write_results_creates_training_manifest() {
        let tmp = tempfile::tempdir().unwrap();
        let training = make_training_output(0);

        write_results(tmp.path(), &training, None, &make_system(), &make_config())
            .expect("write_results must succeed");

        let path = tmp.path().join("training/_manifest.json");
        assert!(path.is_file(), "training/_manifest.json must exist");

        let content = std::fs::read_to_string(&path).unwrap();
        let _parsed: serde_json::Value =
            serde_json::from_str(&content).expect("_manifest.json must contain valid JSON");
    }

    #[test]
    fn write_results_creates_metadata() {
        let tmp = tempfile::tempdir().unwrap();
        let training = make_training_output(0);

        write_results(tmp.path(), &training, None, &make_system(), &make_config())
            .expect("write_results must succeed");

        let path = tmp.path().join("training/metadata.json");
        assert!(path.is_file(), "training/metadata.json must exist");

        let content = std::fs::read_to_string(&path).unwrap();
        let _parsed: serde_json::Value =
            serde_json::from_str(&content).expect("metadata.json must contain valid JSON");
    }

    #[test]
    fn write_results_creates_convergence_parquet() {
        let tmp = tempfile::tempdir().unwrap();
        let training = make_training_output(3);

        write_results(tmp.path(), &training, None, &make_system(), &make_config())
            .expect("write_results must succeed");

        assert!(
            tmp.path().join("training/convergence.parquet").is_file(),
            "training/convergence.parquet must exist"
        );
    }

    #[test]
    fn write_results_convergence_parquet_row_count() {
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

        let tmp = tempfile::tempdir().unwrap();
        let training = make_training_output(3);

        write_results(tmp.path(), &training, None, &make_system(), &make_config())
            .expect("write_results must succeed");

        let path = tmp.path().join("training/convergence.parquet");
        let file = std::fs::File::open(&path).unwrap();
        let reader = ParquetRecordBatchReaderBuilder::try_new(file)
            .unwrap()
            .build()
            .unwrap();

        let total_rows: usize = reader
            .map(|b| b.expect("batch must be Ok").num_rows())
            .sum();
        assert_eq!(total_rows, 3, "convergence.parquet must have 3 rows");
    }

    #[test]
    fn write_results_empty_training_convergence_parquet_correct_schema() {
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

        let tmp = tempfile::tempdir().unwrap();
        let training = make_training_output(0);

        write_results(tmp.path(), &training, None, &make_system(), &make_config())
            .expect("write_results must succeed");

        let path = tmp.path().join("training/convergence.parquet");
        assert!(path.is_file(), "training/convergence.parquet must exist");

        let file = std::fs::File::open(&path).unwrap();
        let builder = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
        let schema = builder.schema().clone();
        let reader = builder.build().unwrap();

        let total_rows: usize = reader
            .map(|b| b.expect("batch must be Ok").num_rows())
            .sum();
        assert_eq!(total_rows, 0, "empty training must produce 0 rows");
        assert_eq!(
            schema.fields().len(),
            13,
            "convergence schema must have 13 columns"
        );

        assert!(
            tmp.path().join("training/_SUCCESS").is_file(),
            "training/_SUCCESS must exist even with 0 records"
        );
    }

    #[test]
    fn write_results_simulation_success_marker_conditional() {
        let tmp = tempfile::tempdir().unwrap();
        let training = make_training_output(0);
        let simulation = SimulationOutput {
            n_scenarios: 10,
            completed: 10,
            failed: 0,
            total_time_ms: 0,
            partitions_written: vec![],
        };

        // With simulation_output = Some: both markers must exist.
        write_results(
            tmp.path(),
            &training,
            Some(&simulation),
            &make_system(),
            &make_config(),
        )
        .expect("write_results must succeed");

        assert!(
            tmp.path().join("simulation/_SUCCESS").is_file(),
            "simulation/_SUCCESS must exist when simulation_output is Some"
        );
        assert!(
            tmp.path().join("training/_SUCCESS").is_file(),
            "training/_SUCCESS must exist"
        );

        // With simulation_output = None: simulation/_SUCCESS must NOT exist.
        let tmp2 = tempfile::tempdir().unwrap();
        write_results(tmp2.path(), &training, None, &make_system(), &make_config())
            .expect("write_results must succeed");

        assert!(
            !tmp2.path().join("simulation/_SUCCESS").exists(),
            "simulation/_SUCCESS must NOT exist when simulation_output is None"
        );
    }

    #[test]
    fn write_results_simulation_manifest_scenarios_total() {
        let tmp = tempfile::tempdir().unwrap();
        let training = make_training_output(3);
        let simulation = SimulationOutput {
            n_scenarios: 10,
            completed: 10,
            failed: 0,
            total_time_ms: 0,
            partitions_written: vec![],
        };

        write_results(
            tmp.path(),
            &training,
            Some(&simulation),
            &make_system(),
            &make_config(),
        )
        .expect("write_results must succeed");

        let path = tmp.path().join("simulation/_manifest.json");
        assert!(path.is_file(), "simulation/_manifest.json must exist");

        let content = std::fs::read_to_string(&path).unwrap();
        let value: serde_json::Value = serde_json::from_str(&content).unwrap();
        assert_eq!(
            value["scenarios"]["total"].as_u64(),
            Some(10),
            "$.scenarios.total must equal 10"
        );
    }

    #[test]
    fn write_results_creates_dictionaries() {
        let tmp = tempfile::tempdir().unwrap();
        let training = make_training_output(0);

        write_results(tmp.path(), &training, None, &make_system(), &make_config())
            .expect("write_results must succeed");

        assert!(
            tmp.path()
                .join("training/dictionaries/codes.json")
                .is_file(),
            "training/dictionaries/codes.json must exist"
        );
    }

    #[test]
    fn write_results_codes_json_contains_operative_state() {
        let tmp = tempfile::tempdir().unwrap();
        let training = make_training_output(0);

        write_results(tmp.path(), &training, None, &make_system(), &make_config())
            .expect("write_results must succeed");

        let path = tmp.path().join("training/dictionaries/codes.json");
        let content = std::fs::read_to_string(&path).unwrap();
        let value: serde_json::Value = serde_json::from_str(&content).unwrap();

        assert!(
            value["operative_state"].is_object(),
            "codes.json must contain an operative_state object"
        );
        assert_eq!(
            value["operative_state"]["2"].as_str(),
            Some("operating"),
            r#"codes.json operative_state["2"] must equal "operating""#
        );
    }

    fn make_system() -> System {
        cobre_core::SystemBuilder::new()
            .build()
            .expect("empty system must be valid")
    }

    fn make_config() -> Config {
        use crate::config::{
            CheckpointingConfig, CutSelectionConfig, EstimationConfig, ExportsConfig,
            InflowNonNegativityConfig, ModelingConfig, PolicyConfig, SimulationConfig,
            SimulationSamplingConfig, StoppingRuleConfig, TrainingConfig, TrainingSolverConfig,
            UpperBoundEvaluationConfig,
        };
        Config {
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
                mode: "fresh".to_string(),
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

    fn make_simulation_output() -> SimulationOutput {
        SimulationOutput {
            n_scenarios: 10,
            completed: 10,
            failed: 0,
            total_time_ms: 1_000,
            partitions_written: vec!["simulation/costs/part-00.parquet".to_string()],
        }
    }

    #[test]
    fn write_training_results_produces_complete_output() {
        let tmp = tempfile::tempdir().unwrap();
        let training = make_training_output(3);

        write_training_results(tmp.path(), &training, &make_system(), &make_config())
            .expect("write_training_results must succeed");

        assert!(tmp.path().join("training").is_dir());
        assert!(tmp.path().join("training/dictionaries").is_dir());
        assert!(tmp.path().join("training/timing").is_dir());
        assert!(tmp.path().join("training/_manifest.json").is_file());
        assert!(tmp.path().join("training/metadata.json").is_file());
        assert!(tmp.path().join("training/_SUCCESS").is_file());
        assert!(
            tmp.path().join("simulation").is_dir(),
            "simulation/ directory must be created by write_training_results"
        );
    }

    #[test]
    fn write_simulation_results_produces_manifest_and_success() {
        let tmp = tempfile::tempdir().unwrap();
        // Simulation dir must exist (normally created by write_training_results).
        std::fs::create_dir_all(tmp.path().join("simulation")).unwrap();
        let sim = make_simulation_output();

        write_simulation_results(tmp.path(), &sim).expect("write_simulation_results must succeed");

        assert!(tmp.path().join("simulation/_manifest.json").is_file());
        assert!(tmp.path().join("simulation/_SUCCESS").is_file());
    }

    #[test]
    fn split_functions_match_write_results_output() {
        let tmp_combined = tempfile::tempdir().unwrap();
        let tmp_split = tempfile::tempdir().unwrap();
        let training = make_training_output(2);
        let sim = make_simulation_output();

        // Write combined.
        write_results(
            tmp_combined.path(),
            &training,
            Some(&sim),
            &make_system(),
            &make_config(),
        )
        .expect("write_results must succeed");

        // Write split.
        write_training_results(tmp_split.path(), &training, &make_system(), &make_config())
            .expect("write_training_results must succeed");
        write_simulation_results(tmp_split.path(), &sim)
            .expect("write_simulation_results must succeed");

        // Both should produce the same file set.
        let combined_training_success = tmp_combined.path().join("training/_SUCCESS").is_file();
        let split_training_success = tmp_split.path().join("training/_SUCCESS").is_file();
        assert_eq!(combined_training_success, split_training_success);

        let combined_sim_success = tmp_combined.path().join("simulation/_SUCCESS").is_file();
        let split_sim_success = tmp_split.path().join("simulation/_SUCCESS").is_file();
        assert_eq!(combined_sim_success, split_sim_success);

        let combined_manifest = tmp_combined
            .path()
            .join("training/_manifest.json")
            .is_file();
        let split_manifest = tmp_split.path().join("training/_manifest.json").is_file();
        assert_eq!(combined_manifest, split_manifest);

        let combined_sim_manifest = tmp_combined
            .path()
            .join("simulation/_manifest.json")
            .is_file();
        let split_sim_manifest = tmp_split.path().join("simulation/_manifest.json").is_file();
        assert_eq!(combined_sim_manifest, split_sim_manifest);
    }
}
