//! `cobre summary <OUTPUT_DIR>` subcommand.
//!
//! Reads the training manifest and convergence log from a completed run's output
//! directory and prints the same human-readable summary as `cobre run` to stderr.
//! This lets users inspect a past run without re-executing the study.
//!
//! # Behavior
//!
//! - `training/_manifest.json` — required; missing file returns [`CliError::Io`].
//! - `training/convergence.parquet` — optional; missing file falls back to
//!   zero-valued bounds (`lp_solves` and timing reported as 0).
//! - `simulation/_manifest.json` — optional; missing file silently skips the
//!   simulation section in the output.
//!
//! All output goes to stderr, matching the `cobre run` convention. stdout is
//! reserved for machine-readable output (see `cobre report`).

use std::path::PathBuf;

use clap::Args;
use console::Term;

use cobre_io::{
    read_convergence_summary, read_simulation_manifest, read_training_manifest, ConvergenceSummary,
    OutputError, SimulationManifest, TrainingManifest,
};

use crate::{
    error::CliError,
    summary::{
        print_simulation_summary, print_training_summary, SimulationSummary, TrainingSummary,
    },
};

// ── Arguments ─────────────────────────────────────────────────────────────────

/// Arguments for the `cobre summary` subcommand.
#[derive(Debug, Args)]
#[command(about = "Display the post-run summary from a completed output directory")]
pub struct SummaryArgs {
    /// Path to the output directory produced by `cobre run`.
    pub output_dir: PathBuf,
}

// ── Execute ───────────────────────────────────────────────────────────────────

/// Execute the `summary` subcommand.
///
/// Reads manifests and convergence data from `args.output_dir` and prints
/// the human-readable training (and optionally simulation) summary to stderr.
/// The format matches what `cobre run` prints at the end of a completed study.
///
/// # Errors
///
/// - [`CliError::Io`] when the output directory does not exist or
///   `training/_manifest.json` cannot be read.
/// - [`CliError::Internal`] when a manifest file contains malformed JSON.
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

    // training/_manifest.json is required; absence is an error.
    let training_manifest_path = output_dir.join("training/_manifest.json");
    let manifest: TrainingManifest =
        read_training_manifest(&training_manifest_path).map_err(CliError::from)?;

    // training/convergence.parquet is optional; fall back to zero-valued summary on error.
    let convergence_path = output_dir.join("training/convergence.parquet");
    let convergence = read_convergence_summary(&convergence_path)
        .unwrap_or_else(|_| convergence_fallback(&manifest));

    // simulation/_manifest.json is optional; missing file is silently skipped.
    let simulation_manifest_path = output_dir.join("simulation/_manifest.json");
    let simulation: Option<SimulationManifest> =
        read_optional_simulation_manifest(&simulation_manifest_path)?;

    // Build and print training summary.
    let training_summary = build_training_summary(&manifest, &convergence);
    let stderr = Term::stderr();
    print_training_summary(&stderr, &training_summary);

    // Build and print simulation summary if the manifest was present.
    if let Some(sim) = simulation {
        let simulation_summary = build_simulation_summary(&sim);
        let _ = stderr.write_line("");
        print_simulation_summary(&stderr, &simulation_summary);
    }

    Ok(())
}

// ── Private helpers ───────────────────────────────────────────────────────────

/// Construct a zero-valued [`ConvergenceSummary`] that derives bounds from the
/// manifest's `convergence.final_gap_percent` field.
///
/// Used when `convergence.parquet` is missing or unreadable. The `lp_solves` and
/// timing fields are set to 0; bounds are 0.0 since the manifest does not carry
/// them separately.
fn convergence_fallback(manifest: &TrainingManifest) -> ConvergenceSummary {
    ConvergenceSummary {
        total_lp_solves: 0,
        total_time_ms: 0,
        final_lower_bound: 0.0,
        final_upper_bound_mean: 0.0,
        final_upper_bound_std: 0.0,
        final_gap_percent: manifest.convergence.final_gap_percent,
    }
}

/// Build a [`TrainingSummary`] by mapping fields from the manifest and convergence data.
fn build_training_summary(
    manifest: &TrainingManifest,
    convergence: &ConvergenceSummary,
) -> TrainingSummary {
    TrainingSummary {
        iterations: u64::from(manifest.iterations.completed),
        converged: manifest.convergence.achieved,
        converged_at: manifest.iterations.converged_at.map(u64::from),
        reason: manifest.convergence.termination_reason.clone(),
        lower_bound: convergence.final_lower_bound,
        upper_bound: convergence.final_upper_bound_mean,
        upper_bound_std: convergence.final_upper_bound_std,
        gap_percent: convergence.final_gap_percent.unwrap_or(0.0),
        total_cuts_active: manifest.cuts.total_active,
        total_cuts_generated: manifest.cuts.total_generated,
        total_lp_solves: convergence.total_lp_solves,
        total_time_ms: convergence.total_time_ms,
        // Solver detail fields are not available from the manifest; filled when
        // reading solver stats Parquet in a future version.
        total_first_try: 0,
        total_retried: 0,
        total_failed: 0,
        total_solve_time_seconds: 0.0,
        total_basis_offered: 0,
        total_basis_rejections: 0,
        total_simplex_iterations: 0,
    }
}

/// Build a [`SimulationSummary`] from a [`SimulationManifest`].
///
/// The simulation manifest does not carry timing data, so `total_time_ms` is
/// set to 0.
fn build_simulation_summary(manifest: &SimulationManifest) -> SimulationSummary {
    SimulationSummary {
        n_scenarios: manifest.scenarios.total,
        completed: manifest.scenarios.completed,
        failed: manifest.scenarios.failed,
        total_time_ms: 0,
        mean_cost: None,
        std_cost: None,
        // Solver stats are not stored in the manifest; zero-filled for `cobre summary`.
        total_lp_solves: 0,
        total_first_try: 0,
        total_retried: 0,
        total_failed_solves: 0,
        total_solve_time_seconds: 0.0,
        total_basis_offered: 0,
        total_basis_rejections: 0,
        total_simplex_iterations: 0,
    }
}

/// Attempt to read an optional simulation manifest.
///
/// Returns `Ok(None)` when the file does not exist (file-not-found I/O error).
/// Propagates all other [`OutputError`] variants as [`CliError`] via the
/// existing [`From<OutputError>`] implementation.
fn read_optional_simulation_manifest(
    path: &std::path::Path,
) -> Result<Option<SimulationManifest>, CliError> {
    match read_simulation_manifest(path) {
        Ok(manifest) => Ok(Some(manifest)),
        Err(OutputError::IoError { source, .. })
            if source.kind() == std::io::ErrorKind::NotFound =>
        {
            Ok(None)
        }
        Err(e) => Err(CliError::from(e)),
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::path::PathBuf;

    use cobre_io::{
        ConvergenceSummary, ManifestConvergence, ManifestCuts, ManifestIterations, ManifestMpiInfo,
        TrainingManifest,
    };

    use super::{build_training_summary, convergence_fallback, SummaryArgs};

    fn make_training_manifest() -> TrainingManifest {
        TrainingManifest {
            version: "2.0.0".to_string(),
            status: "complete".to_string(),
            started_at: Some("2026-01-17T08:00:00Z".to_string()),
            completed_at: Some("2026-01-17T12:30:00Z".to_string()),
            iterations: ManifestIterations {
                max_iterations: Some(100),
                completed: 42,
                converged_at: Some(42),
            },
            convergence: ManifestConvergence {
                achieved: true,
                final_gap_percent: Some(0.45),
                termination_reason: "gap_tolerance".to_string(),
            },
            cuts: ManifestCuts {
                total_generated: 1_250_000,
                total_active: 980_000,
                peak_active: 1_100_000,
            },
            checksum: None,
            mpi_info: ManifestMpiInfo::default(),
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

    // ── summary_args_parses_output_dir ─────────────────────────────────────

    #[test]
    fn summary_args_parses_output_dir() {
        let args = SummaryArgs {
            output_dir: PathBuf::from("/tmp/out"),
        };
        assert_eq!(args.output_dir, PathBuf::from("/tmp/out"));
    }

    // ── construct_training_summary_from_manifest ───────────────────────────

    #[test]
    fn construct_training_summary_from_manifest() {
        let manifest = make_training_manifest();
        let convergence = make_convergence_summary();

        let summary = build_training_summary(&manifest, &convergence);

        assert_eq!(
            summary.iterations, 42,
            "iterations must equal manifest.iterations.completed"
        );
        assert!(
            summary.converged,
            "converged must match manifest.convergence.achieved"
        );
        assert_eq!(
            summary.converged_at,
            Some(42),
            "converged_at must be mapped from manifest.iterations.converged_at"
        );
        assert_eq!(
            summary.reason, "gap_tolerance",
            "reason must equal manifest.convergence.termination_reason"
        );
        assert!(
            (summary.lower_bound - 48_500.0).abs() < f64::EPSILON,
            "lower_bound must come from convergence data"
        );
        assert!(
            (summary.upper_bound - 49_000.0).abs() < f64::EPSILON,
            "upper_bound must come from convergence data"
        );
        assert!(
            (summary.upper_bound_std - 250.0).abs() < f64::EPSILON,
            "upper_bound_std must come from convergence data"
        );
        assert!(
            (summary.gap_percent - 1.03).abs() < 1e-9,
            "gap_percent must come from convergence data"
        );
        assert_eq!(
            summary.total_cuts_active, 980_000,
            "total_cuts_active must come from manifest.cuts"
        );
        assert_eq!(
            summary.total_cuts_generated, 1_250_000,
            "total_cuts_generated must come from manifest.cuts"
        );
        assert_eq!(
            summary.total_lp_solves, 84_000,
            "total_lp_solves must come from convergence data"
        );
        assert_eq!(
            summary.total_time_ms, 12_345,
            "total_time_ms must come from convergence data"
        );
    }

    #[test]
    fn convergence_fallback_uses_manifest_gap_percent() {
        let manifest = make_training_manifest();
        let fallback = convergence_fallback(&manifest);

        assert_eq!(
            fallback.total_lp_solves, 0,
            "fallback total_lp_solves must be 0"
        );
        assert_eq!(
            fallback.total_time_ms, 0,
            "fallback total_time_ms must be 0"
        );
        assert_eq!(
            fallback.final_gap_percent,
            Some(0.45),
            "fallback gap_percent must come from manifest.convergence.final_gap_percent"
        );
    }

    #[test]
    fn convergence_fallback_gap_none_when_manifest_has_no_gap() {
        let mut manifest = make_training_manifest();
        manifest.convergence.final_gap_percent = None;

        let fallback = convergence_fallback(&manifest);

        assert!(
            fallback.final_gap_percent.is_none(),
            "fallback gap_percent must be None when manifest has no gap"
        );
    }

    #[test]
    fn build_training_summary_gap_defaults_to_zero_when_none() {
        let manifest = make_training_manifest();
        let convergence = ConvergenceSummary {
            final_gap_percent: None,
            ..make_convergence_summary()
        };

        let summary = build_training_summary(&manifest, &convergence);

        assert!(
            summary.gap_percent.abs() < f64::EPSILON,
            "gap_percent must default to 0.0 when convergence summary has None"
        );
    }

    #[test]
    fn build_training_summary_converged_at_none_when_manifest_has_none() {
        let mut manifest = make_training_manifest();
        manifest.iterations.converged_at = None;
        manifest.convergence.achieved = false;

        let convergence = make_convergence_summary();
        let summary = build_training_summary(&manifest, &convergence);

        assert!(
            summary.converged_at.is_none(),
            "converged_at must be None when manifest has no converged_at"
        );
        assert!(!summary.converged, "converged must be false");
    }
}
