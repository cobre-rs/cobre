//! `cobre report <RESULTS_DIR>` subcommand.
//!
//! Reads JSON metadata from an output directory produced by `cobre run` and
//! prints a machine-readable JSON summary to stdout. The output is designed to
//! be piped directly to `jq` or other JSON-processing tools.
//!
//! # Output format
//!
//! ```json
//! {
//!   "output_directory": "/abs/path/to/results",
//!   "status": "complete",
//!   "training": { "iterations": ..., "convergence": ..., "cuts": ..., ... },
//!   "simulation": { "scenarios": ..., ... } | null
//! }
//! ```

use std::path::PathBuf;

use clap::Args;
use serde::Serialize;

use cobre_io::{OutputError, SimulationMetadata, TrainingMetadata, read_training_metadata};

use crate::error::CliError;

// ── Output shape ──────────────────────────────────────────────────────────────

/// Top-level JSON output produced by `cobre report`.
#[derive(Debug, Serialize)]
pub struct ReportOutput {
    /// Absolute path to the results directory that was queried.
    pub output_directory: String,
    /// Run status extracted from `training/metadata.json`.
    pub status: String,
    /// Training metadata from `training/metadata.json`.
    pub training: TrainingMetadata,
    /// Simulation metadata from `simulation/metadata.json`, or `null` when
    /// simulation was skipped or not yet run.
    pub simulation: Option<SimulationMetadata>,
}

// ── Arguments ─────────────────────────────────────────────────────────────────

/// Arguments for the `cobre report` subcommand.
#[derive(Debug, Args)]
#[command(about = "Query results from a completed run and print them to stdout")]
pub struct ReportArgs {
    /// Path to the results directory produced by `cobre run`.
    pub results_dir: PathBuf,
}

// ── Execute ───────────────────────────────────────────────────────────────────

/// Execute the `report` subcommand.
///
/// Reads manifest and metadata files from `args.results_dir` and prints a
/// pretty-printed JSON summary to stdout. The output format is stable and
/// suitable for piping to `jq`.
///
/// # Errors
///
/// - [`CliError::Io`] when the results directory does not exist or a required
///   manifest file cannot be read.
/// - [`CliError::Internal`] when a manifest file contains malformed JSON.
pub fn execute(args: ReportArgs) -> Result<(), CliError> {
    let results_dir = args.results_dir;

    // Verify the results directory exists before attempting to read any files.
    if !results_dir.try_exists().map_err(|e| CliError::Io {
        source: e,
        context: results_dir.display().to_string(),
    })? {
        return Err(CliError::Io {
            source: std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "results directory not found",
            ),
            context: results_dir.display().to_string(),
        });
    }

    // training/metadata.json is required; absence is an error.
    let training_metadata_path = results_dir.join("training/metadata.json");
    let training: TrainingMetadata =
        read_training_metadata(&training_metadata_path).map_err(CliError::from)?;

    // simulation/metadata.json is optional (absent when simulation was skipped).
    let simulation_metadata_path = results_dir.join("simulation/metadata.json");
    let simulation: Option<SimulationMetadata> = read_optional_metadata(&simulation_metadata_path)?;

    let output_directory = results_dir.canonicalize().map_or_else(
        |_| results_dir.display().to_string(),
        |p| p.display().to_string(),
    );

    let status = training.status.clone();

    let report = ReportOutput {
        output_directory,
        status,
        training,
        simulation,
    };

    let json = serde_json::to_string_pretty(&report).map_err(|e| CliError::Internal {
        message: format!("failed to serialize report output: {e}"),
    })?;

    println!("{json}");
    Ok(())
}

// ── Private helpers ───────────────────────────────────────────────────────────

/// Read and deserialize an optional manifest file from `path`.
///
/// Returns `Ok(None)` when the file does not exist (file-not-found
/// [`OutputError::IoError`]). Propagates all other [`OutputError`] variants
/// as [`CliError`] via the existing [`From<OutputError>`] implementation.
fn read_optional_metadata<T>(path: &std::path::Path) -> Result<Option<T>, CliError>
where
    T: serde::de::DeserializeOwned,
{
    // Deserialize via serde_json directly, mirroring the pattern used by the
    // manifest readers, but treating file-not-found as an absent optional.
    match std::fs::read_to_string(path) {
        Ok(content) => {
            let value: T = serde_json::from_str(&content).map_err(|e| CliError::Internal {
                message: format!("malformed JSON in {}: {e}", path.display()),
            })?;
            Ok(Some(value))
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(CliError::from(OutputError::io(path, e))),
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn make_training_metadata_json() -> &'static str {
        r#"{
            "cobre_version": "0.3.2",
            "hostname": "test-host",
            "solver": "highs",
            "started_at": "2026-01-17T08:00:00Z",
            "completed_at": "2026-01-17T12:30:00Z",
            "duration_seconds": 16200.0,
            "status": "complete",
            "configuration": {
                "seed": 42,
                "max_iterations": 100,
                "forward_passes": 192,
                "stopping_mode": "any",
                "policy_mode": "fresh"
            },
            "problem_dimensions": {
                "num_stages": 12,
                "num_hydros": 160,
                "num_thermals": 200,
                "num_buses": 5,
                "num_lines": 8
            },
            "iterations": { "completed": 10, "converged_at": 10 },
            "convergence": {
                "achieved": true,
                "final_gap_percent": 0.45,
                "termination_reason": "bound_stalling"
            },
            "cuts": {
                "total_generated": 1250000,
                "total_active": 980000,
                "peak_active": 1100000
            },
            "mpi": { "world_size": 1, "ranks_participated": 1 }
        }"#
    }

    fn make_simulation_metadata_json() -> &'static str {
        r#"{
            "cobre_version": "0.3.2",
            "hostname": "test-host",
            "solver": "highs",
            "started_at": "2026-01-17T13:00:00Z",
            "completed_at": "2026-01-17T13:15:00Z",
            "duration_seconds": 900.0,
            "status": "complete",
            "scenarios": { "total": 100, "completed": 100, "failed": 0 },
            "mpi": { "world_size": 1, "ranks_participated": 1 }
        }"#
    }

    // ── ReportOutput serialization ────────────────────────────────────────

    #[test]
    fn report_output_serializes_to_valid_json() {
        let training: TrainingMetadata =
            serde_json::from_str(make_training_metadata_json()).unwrap();
        let report = ReportOutput {
            output_directory: "/tmp/results".to_string(),
            status: training.status.clone(),
            training,
            simulation: None,
        };

        let json = serde_json::to_string_pretty(&report).unwrap();
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();

        assert!(value["output_directory"].is_string());
        assert!(value["status"].is_string());
        assert!(value["training"].is_object());
        assert!(value["simulation"].is_null());
    }

    #[test]
    fn report_output_contains_iterations_field() {
        let training: TrainingMetadata =
            serde_json::from_str(make_training_metadata_json()).unwrap();
        let report = ReportOutput {
            output_directory: "/tmp/results".to_string(),
            status: training.status.clone(),
            training,
            simulation: None,
        };

        let json = serde_json::to_string_pretty(&report).unwrap();
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();

        assert!(value["training"]["iterations"].is_object());
        assert_eq!(
            value["training"]["iterations"]["completed"].as_u64(),
            Some(10),
        );
    }

    #[test]
    fn report_output_with_simulation_not_null() {
        let training: TrainingMetadata =
            serde_json::from_str(make_training_metadata_json()).unwrap();
        let simulation: SimulationMetadata =
            serde_json::from_str(make_simulation_metadata_json()).unwrap();
        let report = ReportOutput {
            output_directory: "/tmp/results".to_string(),
            status: training.status.clone(),
            training,
            simulation: Some(simulation),
        };

        let json = serde_json::to_string_pretty(&report).unwrap();
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();

        assert!(value["simulation"].is_object());
        assert_eq!(
            value["simulation"]["scenarios"]["total"].as_u64(),
            Some(100),
        );
    }

    #[test]
    fn report_output_status_from_training_metadata() {
        let training: TrainingMetadata =
            serde_json::from_str(make_training_metadata_json()).unwrap();
        let report = ReportOutput {
            output_directory: "/tmp/results".to_string(),
            status: training.status.clone(),
            training,
            simulation: None,
        };

        let json = serde_json::to_string_pretty(&report).unwrap();
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();

        assert_eq!(value["status"].as_str(), Some("complete"));
    }

    // ── read_optional_metadata helper ────────────────────────────────────

    #[test]
    fn read_optional_returns_none_for_nonexistent_file() {
        let result: Result<Option<serde_json::Value>, CliError> =
            read_optional_metadata(std::path::Path::new("/nonexistent/file.json"));
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
    }

    #[test]
    fn read_optional_returns_value_for_existing_file() {
        use std::io::Write;
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(file, r#"{{"key": "value"}}"#).unwrap();

        let result: Result<Option<serde_json::Value>, CliError> =
            read_optional_metadata(file.path());
        assert!(result.is_ok());
        let value = result.unwrap().unwrap();
        assert_eq!(value["key"].as_str(), Some("value"));
    }

    #[test]
    fn read_optional_returns_internal_error_for_malformed_json() {
        use std::io::Write;
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(file, "{{not valid json").unwrap();

        let result: Result<Option<serde_json::Value>, CliError> =
            read_optional_metadata(file.path());
        assert!(matches!(result, Err(CliError::Internal { .. })));
    }

    // ── cobre-io reader integration ──────────────────────────────────────

    #[test]
    fn read_training_metadata_returns_io_error_for_missing_file() {
        let result = read_training_metadata(std::path::Path::new("/nonexistent/metadata.json"))
            .map_err(CliError::from);
        assert!(
            matches!(result, Err(CliError::Io { .. })),
            "missing required file must map to CliError::Io"
        );
    }
}
