//! `cobre summary <OUTPUT_DIR>` subcommand.
//!
//! Reads the training metadata and convergence log from a completed run's output
//! directory and prints the same human-readable summary as `cobre run` to stderr.
//! This lets users inspect a past run without re-executing the study.
//!
//! # Behavior
//!
//! - `training/metadata.json` — required; missing file returns [`CliError::Io`].
//! - `training/convergence.parquet` — optional; missing file falls back to
//!   zero-valued bounds (`lp_solves` and timing reported as 0).
//! - `simulation/metadata.json` — optional; missing file silently skips the
//!   simulation section in the output.
//!
//! All output goes to stderr, matching the `cobre run` convention. stdout is
//! reserved for machine-readable output (see `cobre report`).

use std::path::PathBuf;

use clap::Args;
use console::Term;

use cobre_io::{
    ConvergenceSummary, OutputError, SimulationMetadata, TrainingMetadata,
    read_convergence_summary, read_simulation_metadata, read_training_metadata,
};

use crate::{
    error::CliError,
    summary::{
        SimulationSummary, TrainingSummary, print_simulation_summary, print_training_summary,
    },
};

// ── Arguments ────────────────────────────────────────────────────────────────

/// Arguments for the `cobre summary` subcommand.
#[derive(Debug, Args)]
#[command(about = "Display the post-run summary from a completed output directory")]
pub struct SummaryArgs {
    /// Path to the output directory produced by `cobre run`.
    pub output_dir: PathBuf,
}

// ── Execute ──────────────────────────────────────────────────────────────────

/// Execute the `summary` subcommand.
///
/// Reads metadata and convergence data from `args.output_dir` and prints
/// the human-readable training (and optionally simulation) summary to stderr.
/// The format matches what `cobre run` prints at the end of a completed study.
///
/// # Errors
///
/// - [`CliError::Io`] when the output directory does not exist or
///   `training/metadata.json` cannot be read.
/// - [`CliError::Internal`] when a metadata file contains malformed JSON.
pub fn execute(args: SummaryArgs) -> Result<(), CliError> {
    let output_dir = args.output_dir;

    // Verify the output directory exists before attempting to read any files.
    if !output_dir.try_exists().map_err(|e| CliError::Io {
        source: e,
        context: "output directory".to_string(),
    })? {
        return Err(CliError::Io {
            source: std::io::Error::new(std::io::ErrorKind::NotFound, "output directory not found"),
            context: output_dir.display().to_string(),
        });
    }

    // training/metadata.json is required; absence is an error.
    let training_metadata_path = output_dir.join("training/metadata.json");
    let metadata: TrainingMetadata =
        read_training_metadata(&training_metadata_path).map_err(CliError::from)?;

    // training/convergence.parquet is optional; fall back to zero-valued summary on error.
    let convergence_path = output_dir.join("training/convergence.parquet");
    let convergence = read_convergence_summary(&convergence_path)
        .unwrap_or_else(|_| convergence_fallback(&metadata));

    // simulation/metadata.json is optional; missing file is silently skipped.
    let simulation_metadata_path = output_dir.join("simulation/metadata.json");
    let simulation: Option<SimulationMetadata> =
        read_optional_simulation_metadata(&simulation_metadata_path)?;

    // Build and print training summary.
    let training_summary = build_training_summary(&metadata, &convergence);
    let stderr = Term::stderr();
    print_training_summary(&stderr, &training_summary);

    // Build and print simulation summary if the metadata was present.
    if let Some(sim) = simulation {
        let simulation_summary = build_simulation_summary(&sim);
        let _ = stderr.write_line("");
        print_simulation_summary(&stderr, &simulation_summary);
    }

    Ok(())
}

// ── Private helpers ──────────────────────────────────────────────────────────

/// Construct a zero-valued [`ConvergenceSummary`] that derives bounds from the
/// metadata's `convergence.final_gap_percent` field.
///
/// Used when `convergence.parquet` is missing or unreadable.
fn convergence_fallback(metadata: &TrainingMetadata) -> ConvergenceSummary {
    ConvergenceSummary {
        total_lp_solves: 0,
        total_time_ms: 0,
        final_lower_bound: 0.0,
        final_upper_bound_mean: 0.0,
        final_upper_bound_std: 0.0,
        final_gap_percent: metadata.convergence.final_gap_percent,
    }
}

/// Build a [`TrainingSummary`] by mapping fields from the metadata and convergence data.
fn build_training_summary(
    metadata: &TrainingMetadata,
    convergence: &ConvergenceSummary,
) -> TrainingSummary {
    TrainingSummary {
        iterations: u64::from(metadata.iterations.completed),
        converged: metadata.convergence.achieved,
        converged_at: metadata.iterations.converged_at.map(u64::from),
        reason: metadata.convergence.termination_reason.clone(),
        lower_bound: convergence.final_lower_bound,
        upper_bound: convergence.final_upper_bound_mean,
        upper_bound_std: convergence.final_upper_bound_std,
        gap_percent: convergence.final_gap_percent.unwrap_or(0.0),
        total_cuts_active: metadata.cuts.total_active,
        total_cuts_generated: metadata.cuts.total_generated,
        total_lp_solves: convergence.total_lp_solves,
        total_time_ms: convergence.total_time_ms,
        total_first_try: None,
        total_retried: None,
        total_failed: None,
        total_solve_time_seconds: None,
        total_basis_offered: None,
        total_basis_rejections: None,
        total_simplex_iterations: None,
    }
}

/// Build a [`SimulationSummary`] from a [`SimulationMetadata`].
fn build_simulation_summary(metadata: &SimulationMetadata) -> SimulationSummary {
    SimulationSummary {
        n_scenarios: metadata.scenarios.total,
        completed: metadata.scenarios.completed,
        failed: metadata.scenarios.failed,
        total_time_ms: 0,
        mean_cost: None,
        std_cost: None,
        total_lp_solves: None,
        total_first_try: None,
        total_retried: None,
        total_failed_solves: None,
        total_solve_time_seconds: None,
        total_basis_offered: None,
        total_basis_rejections: None,
        total_simplex_iterations: None,
    }
}

/// Attempt to read an optional simulation metadata file.
///
/// Returns `Ok(None)` when the file does not exist.
fn read_optional_simulation_metadata(
    path: &std::path::Path,
) -> Result<Option<SimulationMetadata>, CliError> {
    match read_simulation_metadata(path) {
        Ok(metadata) => Ok(Some(metadata)),
        Err(OutputError::IoError { source, .. })
            if source.kind() == std::io::ErrorKind::NotFound =>
        {
            Ok(None)
        }
        Err(e) => Err(CliError::from(e)),
    }
}

// ── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::path::PathBuf;

    use cobre_io::{
        ConvergenceSummary, DistributionInfo, MetadataConfiguration, MetadataConvergence,
        MetadataCuts, MetadataIterations, MetadataProblemDimensions, TrainingMetadata,
    };

    use super::{SummaryArgs, build_training_summary, convergence_fallback};

    fn make_training_metadata() -> TrainingMetadata {
        TrainingMetadata {
            cobre_version: env!("CARGO_PKG_VERSION").to_string(),
            hostname: "test-host".to_string(),
            solver: "highs".to_string(),
            started_at: "2026-01-17T08:00:00Z".to_string(),
            completed_at: "2026-01-17T12:30:00Z".to_string(),
            duration_seconds: 16_200.0,
            status: "complete".to_string(),
            configuration: MetadataConfiguration {
                seed: Some(42),
                max_iterations: Some(100),
                forward_passes: Some(192),
                stopping_mode: "any".to_string(),
                policy_mode: "fresh".to_string(),
            },
            problem_dimensions: MetadataProblemDimensions {
                num_stages: 12,
                num_hydros: 160,
                num_thermals: 200,
                num_buses: 5,
                num_lines: 8,
            },
            iterations: MetadataIterations {
                completed: 42,
                converged_at: Some(42),
            },
            convergence: MetadataConvergence {
                achieved: true,
                final_gap_percent: Some(0.45),
                termination_reason: "gap_tolerance".to_string(),
            },
            cuts: MetadataCuts {
                total_generated: 1_250_000,
                total_active: 980_000,
                peak_active: 1_100_000,
            },
            distribution: DistributionInfo {
                backend: "local".to_string(),
                world_size: 1,
                ranks_participated: 1,
                num_nodes: 1,
                threads_per_rank: 1,
                mpi_library: None,
                mpi_standard: None,
                thread_level: None,
                slurm_job_id: None,
            },
        }
    }

    fn make_convergence_summary() -> ConvergenceSummary {
        ConvergenceSummary {
            total_lp_solves: 84_000,
            total_time_ms: 12_345,
            final_lower_bound: 48_500.0,
            final_upper_bound_mean: 49_000.0,
            final_upper_bound_std: 250.0,
            final_gap_percent: Some(1.03),
        }
    }

    #[test]
    fn summary_args_parses_output_dir() {
        let args = SummaryArgs {
            output_dir: PathBuf::from("/tmp/out"),
        };
        assert_eq!(args.output_dir, PathBuf::from("/tmp/out"));
    }

    #[test]
    fn construct_training_summary_from_metadata() {
        let metadata = make_training_metadata();
        let convergence = make_convergence_summary();

        let summary = build_training_summary(&metadata, &convergence);

        assert_eq!(summary.iterations, 42);
        assert!(summary.converged);
        assert_eq!(summary.converged_at, Some(42));
        assert_eq!(summary.reason, "gap_tolerance");
        assert!((summary.lower_bound - 48_500.0).abs() < f64::EPSILON);
        assert!((summary.upper_bound - 49_000.0).abs() < f64::EPSILON);
        assert!((summary.upper_bound_std - 250.0).abs() < f64::EPSILON);
        assert!((summary.gap_percent - 1.03).abs() < 1e-9);
        assert_eq!(summary.total_cuts_active, 980_000);
        assert_eq!(summary.total_cuts_generated, 1_250_000);
        assert_eq!(summary.total_lp_solves, 84_000);
        assert_eq!(summary.total_time_ms, 12_345);
    }

    #[test]
    fn convergence_fallback_uses_metadata_gap_percent() {
        let metadata = make_training_metadata();
        let fallback = convergence_fallback(&metadata);

        assert_eq!(fallback.total_lp_solves, 0);
        assert_eq!(fallback.total_time_ms, 0);
        assert_eq!(fallback.final_gap_percent, Some(0.45));
    }

    #[test]
    fn convergence_fallback_gap_none_when_metadata_has_no_gap() {
        let mut metadata = make_training_metadata();
        metadata.convergence.final_gap_percent = None;

        let fallback = convergence_fallback(&metadata);

        assert!(fallback.final_gap_percent.is_none());
    }

    #[test]
    fn build_training_summary_gap_defaults_to_zero_when_none() {
        let metadata = make_training_metadata();
        let convergence = ConvergenceSummary {
            final_gap_percent: None,
            ..make_convergence_summary()
        };

        let summary = build_training_summary(&metadata, &convergence);

        assert!(summary.gap_percent.abs() < f64::EPSILON);
    }

    #[test]
    fn build_training_summary_converged_at_none_when_metadata_has_none() {
        let mut metadata = make_training_metadata();
        metadata.iterations.converged_at = None;
        metadata.convergence.achieved = false;

        let convergence = make_convergence_summary();
        let summary = build_training_summary(&metadata, &convergence);

        assert!(summary.converged_at.is_none());
        assert!(!summary.converged);
    }
}
