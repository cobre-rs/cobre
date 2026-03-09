//! `cobre report <RESULTS_DIR>` subcommand.
//!
//! Reads JSON manifests from an output directory produced by `cobre run` and
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
//!   "simulation": { "scenarios": ..., ... } | null,
//!   "metadata": { "run_info": ..., "configuration_snapshot": ..., ... } | null
//! }
//! ```

use std::path::PathBuf;

use clap::Args;
use serde::Serialize;

use cobre_io::{SimulationManifest, TrainingManifest, TrainingMetadata};

use crate::error::CliError;

// ── Output shape ──────────────────────────────────────────────────────────────

/// Top-level JSON output produced by `cobre report`.
#[derive(Debug, Serialize)]
pub struct ReportOutput {
    /// Absolute path to the results directory that was queried.
    pub output_directory: String,
    /// Run status extracted from `training/_manifest.json`.
    ///
    /// Typical values: `"complete"`, `"converged"`, `"running"`, `"failed"`,
    /// `"partial"`.
    pub status: String,
    /// Training summary from `training/_manifest.json`.
    pub training: TrainingManifest,
    /// Simulation summary from `simulation/_manifest.json`, or `null` when
    /// simulation was skipped or not yet run.
    pub simulation: Option<SimulationManifest>,
    /// Metadata from `training/metadata.json`, or `null` when the file is
    /// absent.
    pub metadata: Option<TrainingMetadata>,
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

    // training/_manifest.json is required; absence is an error.
    let training_manifest_path = results_dir.join("training/_manifest.json");
    let training: TrainingManifest =
        read_json_file(&training_manifest_path, "training/_manifest.json")?;

    // training/metadata.json is optional.
    let metadata_path = results_dir.join("training/metadata.json");
    let metadata: Option<TrainingMetadata> = read_optional_json_file(&metadata_path)?;

    // simulation/_manifest.json is optional (absent when --skip-simulation).
    let simulation_manifest_path = results_dir.join("simulation/_manifest.json");
    let simulation: Option<SimulationManifest> =
        read_optional_json_file(&simulation_manifest_path)?;

    // Canonicalize the output directory path to an absolute form. Fall back to
    // the display string when canonicalization fails (e.g., path is a symlink
    // whose target has been removed).
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
        metadata,
    };

    let json = serde_json::to_string_pretty(&report).map_err(|e| CliError::Internal {
        message: format!("failed to serialize report output: {e}"),
    })?;

    println!("{json}");
    Ok(())
}

// ── Private helpers ───────────────────────────────────────────────────────────

/// Read and deserialize a required JSON file.
///
/// Returns [`CliError::Io`] when the file cannot be read and
/// [`CliError::Internal`] when the JSON is malformed.
fn read_json_file<T>(path: &std::path::Path, description: &str) -> Result<T, CliError>
where
    T: serde::de::DeserializeOwned,
{
    let content = std::fs::read_to_string(path).map_err(|e| CliError::Io {
        source: e,
        context: description.to_string(),
    })?;

    serde_json::from_str(&content).map_err(|e| CliError::Internal {
        message: format!("malformed JSON in {description}: {e}"),
    })
}

/// Read and deserialize an optional JSON file.
///
/// Returns `Ok(None)` when the file does not exist. Returns
/// [`CliError::Internal`] when the file exists but contains malformed JSON.
fn read_optional_json_file<T>(path: &std::path::Path) -> Result<Option<T>, CliError>
where
    T: serde::de::DeserializeOwned,
{
    match std::fs::read_to_string(path) {
        Ok(content) => {
            let value: T = serde_json::from_str(&content).map_err(|e| CliError::Internal {
                message: format!("malformed JSON in {}: {e}", path.display()),
            })?;
            Ok(Some(value))
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(CliError::Io {
            source: e,
            context: path.display().to_string(),
        }),
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn make_training_manifest_json() -> &'static str {
        r#"{
            "version": "2.0.0",
            "status": "complete",
            "started_at": "2026-01-17T08:00:00Z",
            "completed_at": "2026-01-17T12:30:00Z",
            "iterations": {
                "max_iterations": 100,
                "completed": 10,
                "converged_at": 10
            },
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
            "checksum": null,
            "mpi_info": { "world_size": 1, "ranks_participated": 1 }
        }"#
    }

    fn make_simulation_manifest_json() -> &'static str {
        r#"{
            "version": "2.0.0",
            "status": "complete",
            "started_at": "2026-01-17T13:00:00Z",
            "completed_at": "2026-01-17T13:15:00Z",
            "scenarios": { "total": 100, "completed": 100, "failed": 0 },
            "partitions_written": [],
            "checksum": null,
            "mpi_info": { "world_size": 1, "ranks_participated": 1 }
        }"#
    }

    // ── ReportOutput serialization ─────────────────────────────────────────

    #[test]
    fn report_output_serializes_to_valid_json() {
        let training: TrainingManifest =
            serde_json::from_str(make_training_manifest_json()).unwrap();
        let report = ReportOutput {
            output_directory: "/tmp/results".to_string(),
            status: training.status.clone(),
            training,
            simulation: None,
            metadata: None,
        };

        let json = serde_json::to_string_pretty(&report).unwrap();
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();

        assert!(
            value["output_directory"].is_string(),
            "output_directory must be a string"
        );
        assert!(value["status"].is_string(), "status must be a string");
        assert!(value["training"].is_object(), "training must be an object");
        assert!(
            value["simulation"].is_null(),
            "simulation must be null when absent"
        );
        assert!(
            value["metadata"].is_null(),
            "metadata must be null when absent"
        );
    }

    #[test]
    fn report_output_contains_iterations_field() {
        let training: TrainingManifest =
            serde_json::from_str(make_training_manifest_json()).unwrap();
        let report = ReportOutput {
            output_directory: "/tmp/results".to_string(),
            status: training.status.clone(),
            training,
            simulation: None,
            metadata: None,
        };

        let json = serde_json::to_string_pretty(&report).unwrap();
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();

        assert!(
            value["training"]["iterations"].is_object(),
            "training.iterations must be an object"
        );
        assert_eq!(
            value["training"]["iterations"]["completed"].as_u64(),
            Some(10),
            "training.iterations.completed must equal 10"
        );
    }

    #[test]
    fn report_output_with_simulation_not_null() {
        let training: TrainingManifest =
            serde_json::from_str(make_training_manifest_json()).unwrap();
        let simulation: SimulationManifest =
            serde_json::from_str(make_simulation_manifest_json()).unwrap();
        let report = ReportOutput {
            output_directory: "/tmp/results".to_string(),
            status: training.status.clone(),
            training,
            simulation: Some(simulation),
            metadata: None,
        };

        let json = serde_json::to_string_pretty(&report).unwrap();
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();

        assert!(
            value["simulation"].is_object(),
            "simulation must be an object when present"
        );
        assert_eq!(
            value["simulation"]["scenarios"]["total"].as_u64(),
            Some(100),
            "simulation.scenarios.total must equal 100"
        );
    }

    #[test]
    fn report_output_status_from_training_manifest() {
        let training: TrainingManifest =
            serde_json::from_str(make_training_manifest_json()).unwrap();
        let report = ReportOutput {
            output_directory: "/tmp/results".to_string(),
            status: training.status.clone(),
            training,
            simulation: None,
            metadata: None,
        };

        let json = serde_json::to_string_pretty(&report).unwrap();
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();

        assert_eq!(
            value["status"].as_str(),
            Some("complete"),
            "status must equal the training manifest status"
        );
    }

    // ── read_optional_json_file helper ─────────────────────────────────────

    #[test]
    fn read_optional_returns_none_for_nonexistent_file() {
        let result: Result<Option<serde_json::Value>, CliError> =
            read_optional_json_file(std::path::Path::new("/nonexistent/file.json"));
        assert!(result.is_ok(), "nonexistent file must return Ok(None)");
        assert!(result.unwrap().is_none());
    }

    #[test]
    fn read_optional_returns_value_for_existing_file() {
        use std::io::Write;
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(file, r#"{{"key": "value"}}"#).unwrap();

        let result: Result<Option<serde_json::Value>, CliError> =
            read_optional_json_file(file.path());
        assert!(result.is_ok(), "existing file must return Ok(Some(...))");
        let value = result.unwrap().unwrap();
        assert_eq!(value["key"].as_str(), Some("value"));
    }

    #[test]
    fn read_optional_returns_internal_error_for_malformed_json() {
        use std::io::Write;
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(file, "{{not valid json").unwrap();

        let result: Result<Option<serde_json::Value>, CliError> =
            read_optional_json_file(file.path());
        assert!(
            matches!(result, Err(CliError::Internal { .. })),
            "malformed JSON must return CliError::Internal"
        );
    }

    // ── read_json_file helper ──────────────────────────────────────────────

    #[test]
    fn read_json_file_returns_io_error_for_missing_file() {
        let result: Result<serde_json::Value, CliError> =
            read_json_file(std::path::Path::new("/nonexistent/file.json"), "test file");
        assert!(
            matches!(result, Err(CliError::Io { .. })),
            "missing required file must return CliError::Io"
        );
    }

    #[test]
    fn read_json_file_returns_internal_error_for_malformed_json() {
        use std::io::Write;
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(file, "{{not valid json").unwrap();

        let result: Result<serde_json::Value, CliError> = read_json_file(file.path(), "test file");
        assert!(
            matches!(result, Err(CliError::Internal { .. })),
            "malformed JSON must return CliError::Internal"
        );
    }
}
