//! Metadata writers for the output pipeline.
//!
//! This module provides JSON writers for two metadata files:
//!
//! - [`write_training_metadata`] — writes `training/metadata.json` capturing
//!   run context, configuration, convergence, and cut statistics.
//! - [`write_simulation_metadata`] — writes `simulation/metadata.json` capturing
//!   run context and scenario completion counts.
//!
//! Both replace the previous split of `_manifest.json` + `metadata.json` with a
//! single merged file per output directory. The `_SUCCESS` marker still signals
//! completion; metadata files capture the run details.
//!
//! All writers use an atomic write pattern: data is serialized to a `.tmp` file
//! first, then atomically renamed to the target path. This prevents partial files
//! from being visible to readers.

use std::path::Path;

use serde::{Deserialize, Serialize};

use super::error::OutputError;

// ── OutputContext ─────────────────────────────────────────────────────────────

/// Runtime context for metadata output files.
///
/// Captures environment information not available from the solver output
/// or configuration alone: hostname, execution distribution, and wall-clock
/// timestamps. Built by the CLI or Python entry point and passed to the
/// output writers.
pub struct OutputContext {
    /// Hostname of the machine that produced this output.
    pub hostname: String,
    /// LP solver backend name (e.g. `"highs"`).
    pub solver: String,
    /// ISO 8601 timestamp when the phase started.
    pub started_at: String,
    /// ISO 8601 timestamp when the phase completed.
    pub completed_at: String,
    /// Execution distribution and environment information.
    pub distribution: DistributionInfo,
}

/// Read the system hostname.
///
/// Tries `/proc/sys/kernel/hostname` first (Linux), then the `HOSTNAME`
/// environment variable, falling back to `"unknown"`.
#[must_use]
pub fn get_hostname() -> String {
    std::fs::read_to_string("/proc/sys/kernel/hostname")
        .map(|s| s.trim().to_string())
        .or_else(|_| std::env::var("HOSTNAME"))
        .unwrap_or_else(|_| "unknown".to_string())
}

/// Return the current UTC time as an ISO 8601 string (e.g. `"2026-04-05T14:30:00Z"`).
#[must_use]
pub fn now_iso8601() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
}

// ── Shared nested structs ────────────────────────────────────────────────────

/// Execution distribution information embedded in metadata files.
///
/// Captures the communication backend, process topology, and optional
/// MPI/scheduler metadata for reproducibility. Replaces the previous
/// `MpiInfo` struct with richer environment context.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributionInfo {
    /// Communication backend: `"mpi"` or `"local"`.
    pub backend: String,
    /// Total number of processes in the communicator.
    pub world_size: u32,
    /// Number of processes that actually participated in computation.
    pub ranks_participated: u32,
    /// Number of distinct physical hosts.
    pub num_nodes: u32,
    /// Rayon threads per process.
    pub threads_per_rank: u32,
    /// MPI implementation version, e.g. `"Open MPI v4.1.6"`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mpi_library: Option<String>,
    /// MPI standard version, e.g. `"MPI 4.0"`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mpi_standard: Option<String>,
    /// Negotiated MPI thread safety level, e.g. `"Funneled"`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thread_level: Option<String>,
    /// SLURM job ID, if running under SLURM.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub slurm_job_id: Option<String>,
}

/// Selected training configuration fields captured for reproducibility.
///
/// This is an informational snapshot, not a normative schema. The canonical
/// configuration schema lives in `config.json` (see `cobre_io::config`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetadataConfiguration {
    /// Random seed used for scenario generation.
    pub seed: Option<i64>,
    /// Maximum iterations from the iteration-limit stopping rule.
    pub max_iterations: Option<u32>,
    /// Number of forward-pass scenario trajectories per iteration.
    pub forward_passes: Option<u32>,
    /// How multiple stopping rules combine: `"any"` or `"all"`.
    pub stopping_mode: String,
    /// Policy warm-start mode (e.g. `"fresh"`, `"resume"`).
    pub policy_mode: String,
}

/// Problem dimensionality embedded in [`TrainingMetadata`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetadataProblemDimensions {
    /// Number of stages in the planning horizon.
    pub num_stages: u32,
    /// Total number of hydro plants.
    pub num_hydros: u32,
    /// Total number of thermal plants.
    pub num_thermals: u32,
    /// Total number of buses.
    pub num_buses: u32,
    /// Total number of transmission lines.
    pub num_lines: u32,
}

/// Iteration counts embedded in [`TrainingMetadata`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetadataIterations {
    /// Number of iterations actually completed.
    pub completed: u32,
    /// Iteration at which convergence was achieved (`null` if not converged).
    pub converged_at: Option<u32>,
}

/// Convergence summary embedded in [`TrainingMetadata`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetadataConvergence {
    /// Whether a convergence-oriented stopping rule triggered termination.
    pub achieved: bool,
    /// Final optimality gap in percent (`null` when upper bound evaluation is disabled).
    pub final_gap_percent: Option<f64>,
    /// Human-readable description of the rule that terminated the run.
    pub termination_reason: String,
}

/// Cut pool summary embedded in [`TrainingMetadata`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetadataCuts {
    /// Total cuts generated over the entire run.
    pub total_generated: u64,
    /// Cuts still active in the pool at termination.
    pub total_active: u64,
    /// Highest number of simultaneously active cuts observed.
    pub peak_active: u64,
}

/// Scenario counts embedded in [`SimulationMetadata`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetadataScenarios {
    /// Total number of scenarios dispatched for simulation.
    pub total: u32,
    /// Number of scenarios that completed without error.
    pub completed: u32,
    /// Number of scenarios that encountered a terminal error.
    pub failed: u32,
}

// ── TrainingMetadata ─────────────────────────────────────────────────────────

/// Merged metadata for the training output directory (`training/metadata.json`).
///
/// Replaces the previous split of `training/_manifest.json` (convergence/cuts)
/// and `training/metadata.json` (configuration/environment) with a single file
/// containing all run information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingMetadata {
    /// Version of the cobre crate that produced this output.
    pub cobre_version: String,
    /// Hostname of the machine that ran training.
    pub hostname: String,
    /// LP solver backend name (e.g. `"highs"`).
    pub solver: String,
    /// ISO 8601 timestamp when training started.
    pub started_at: String,
    /// ISO 8601 timestamp when training completed.
    pub completed_at: String,
    /// Total training wall-clock duration in seconds.
    pub duration_seconds: f64,
    /// Run status: `"complete"` or `"partial"`.
    pub status: String,
    /// Snapshot of key configuration fields.
    pub configuration: MetadataConfiguration,
    /// Problem size dimensions.
    pub problem_dimensions: MetadataProblemDimensions,
    /// Iteration completion counts.
    pub iterations: MetadataIterations,
    /// Convergence outcome.
    pub convergence: MetadataConvergence,
    /// Cut pool summary.
    pub cuts: MetadataCuts,
    /// Execution distribution and environment information.
    pub distribution: DistributionInfo,
}

// ── SimulationMetadata ───────────────────────────────────────────────────────

/// Metadata for the simulation output directory (`simulation/metadata.json`).
///
/// Replaces the previous `simulation/_manifest.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationMetadata {
    /// Version of the cobre crate that produced this output.
    pub cobre_version: String,
    /// Hostname of the machine that ran simulation.
    pub hostname: String,
    /// LP solver backend name (e.g. `"highs"`).
    pub solver: String,
    /// ISO 8601 timestamp when simulation started.
    pub started_at: String,
    /// ISO 8601 timestamp when simulation completed.
    pub completed_at: String,
    /// Total simulation wall-clock duration in seconds.
    pub duration_seconds: f64,
    /// Run status: `"complete"` or `"partial"`.
    pub status: String,
    /// Scenario completion counts.
    pub scenarios: MetadataScenarios,
    /// Execution distribution and environment information.
    pub distribution: DistributionInfo,
}

// ── Writers ──────────────────────────────────────────────────────────────────

/// Write training metadata to `path` using the atomic write pattern.
///
/// # Errors
///
/// - [`OutputError::ManifestError`] if JSON serialization fails.
/// - [`OutputError::IoError`] if the file write or atomic rename fails.
pub fn write_training_metadata(
    path: &Path,
    metadata: &TrainingMetadata,
) -> Result<(), OutputError> {
    write_json_atomic(path, metadata, "training_metadata")
}

/// Write simulation metadata to `path` using the atomic write pattern.
///
/// # Errors
///
/// - [`OutputError::ManifestError`] if JSON serialization fails.
/// - [`OutputError::IoError`] if the file write or atomic rename fails.
pub fn write_simulation_metadata(
    path: &Path,
    metadata: &SimulationMetadata,
) -> Result<(), OutputError> {
    write_json_atomic(path, metadata, "simulation_metadata")
}

/// Read training metadata from `path`.
///
/// # Errors
///
/// - [`OutputError::IoError`] if the file cannot be read.
/// - [`OutputError::ManifestError`] if the file contains malformed JSON.
pub fn read_training_metadata(path: &Path) -> Result<TrainingMetadata, OutputError> {
    read_json(path, "training_metadata")
}

/// Read simulation metadata from `path`.
///
/// # Errors
///
/// - [`OutputError::IoError`] if the file cannot be read.
/// - [`OutputError::ManifestError`] if the file contains malformed JSON.
pub fn read_simulation_metadata(path: &Path) -> Result<SimulationMetadata, OutputError> {
    read_json(path, "simulation_metadata")
}

// ── Internal helpers ─────────────────────────────────────────────────────────

/// Read and deserialize a JSON file into `T`.
fn read_json<T>(path: &Path, manifest_type: &str) -> Result<T, OutputError>
where
    T: serde::de::DeserializeOwned,
{
    let content = std::fs::read_to_string(path).map_err(|e| OutputError::io(path, e))?;
    serde_json::from_str(&content).map_err(|e| OutputError::ManifestError {
        manifest_type: manifest_type.to_string(),
        message: e.to_string(),
    })
}

/// Serialize `value` to pretty-printed JSON and atomically write it to `path`.
fn write_json_atomic<T: Serialize>(
    path: &Path,
    value: &T,
    manifest_type: &str,
) -> Result<(), OutputError> {
    let json = serde_json::to_string_pretty(value).map_err(|e| OutputError::ManifestError {
        manifest_type: manifest_type.to_string(),
        message: e.to_string(),
    })?;

    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, &json).map_err(|e| OutputError::io(&tmp, e))?;
    std::fs::rename(&tmp, path).map_err(|e| OutputError::io(path, e))?;

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
    use tempfile::tempdir;

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn make_distribution_info() -> DistributionInfo {
        DistributionInfo {
            backend: "local".to_string(),
            world_size: 1,
            ranks_participated: 1,
            num_nodes: 1,
            threads_per_rank: 1,
            mpi_library: None,
            mpi_standard: None,
            thread_level: None,
            slurm_job_id: None,
        }
    }

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
                completed: 100,
                converged_at: Some(95),
            },
            convergence: MetadataConvergence {
                achieved: true,
                final_gap_percent: Some(0.45),
                termination_reason: "bound_stalling".to_string(),
            },
            cuts: MetadataCuts {
                total_generated: 1_250_000,
                total_active: 980_000,
                peak_active: 1_100_000,
            },
            distribution: make_distribution_info(),
        }
    }

    fn make_simulation_metadata() -> SimulationMetadata {
        SimulationMetadata {
            cobre_version: env!("CARGO_PKG_VERSION").to_string(),
            hostname: "test-host".to_string(),
            solver: "highs".to_string(),
            started_at: "2026-01-17T13:00:00Z".to_string(),
            completed_at: "2026-01-17T13:15:00Z".to_string(),
            duration_seconds: 900.0,
            status: "complete".to_string(),
            scenarios: MetadataScenarios {
                total: 100,
                completed: 100,
                failed: 0,
            },
            distribution: make_distribution_info(),
        }
    }

    // ── Roundtrip tests ──────────────────────────────────────────────────────

    #[test]
    fn training_metadata_roundtrip() {
        let original = make_training_metadata();
        let json = serde_json::to_string_pretty(&original).unwrap();
        let decoded: TrainingMetadata = serde_json::from_str(&json).unwrap();

        assert_eq!(decoded.cobre_version, original.cobre_version);
        assert_eq!(decoded.hostname, original.hostname);
        assert_eq!(decoded.solver, original.solver);
        assert_eq!(decoded.started_at, original.started_at);
        assert_eq!(decoded.completed_at, original.completed_at);
        assert_eq!(decoded.duration_seconds, original.duration_seconds);
        assert_eq!(decoded.status, original.status);
        assert_eq!(decoded.iterations.completed, original.iterations.completed);
        assert_eq!(
            decoded.iterations.converged_at,
            original.iterations.converged_at
        );
        assert_eq!(decoded.convergence.achieved, original.convergence.achieved);
        assert_eq!(
            decoded.convergence.final_gap_percent,
            original.convergence.final_gap_percent
        );
        assert_eq!(decoded.cuts.total_generated, original.cuts.total_generated);
        assert_eq!(decoded.cuts.total_active, original.cuts.total_active);
        assert_eq!(decoded.cuts.peak_active, original.cuts.peak_active);
        assert_eq!(
            decoded.distribution.world_size,
            original.distribution.world_size
        );
    }

    #[test]
    fn simulation_metadata_roundtrip() {
        let original = make_simulation_metadata();
        let json = serde_json::to_string_pretty(&original).unwrap();
        let decoded: SimulationMetadata = serde_json::from_str(&json).unwrap();

        assert_eq!(decoded.cobre_version, original.cobre_version);
        assert_eq!(decoded.status, original.status);
        assert_eq!(decoded.scenarios.total, original.scenarios.total);
        assert_eq!(decoded.scenarios.completed, original.scenarios.completed);
        assert_eq!(decoded.scenarios.failed, original.scenarios.failed);
        assert_eq!(
            decoded.distribution.world_size,
            original.distribution.world_size
        );
    }

    // ── Writer tests ─────────────────────────────────────────────────────────

    #[test]
    fn write_training_metadata_creates_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("metadata.json");
        let metadata = make_training_metadata();

        write_training_metadata(&path, &metadata).expect("write must succeed");

        assert!(path.exists(), "metadata file must exist after write");
        let content = std::fs::read_to_string(&path).unwrap();
        let _parsed: serde_json::Value =
            serde_json::from_str(&content).expect("file must contain valid JSON");
    }

    #[test]
    fn write_simulation_metadata_creates_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("metadata.json");
        let metadata = make_simulation_metadata();

        write_simulation_metadata(&path, &metadata).expect("write must succeed");

        assert!(path.exists(), "metadata file must exist after write");
        let content = std::fs::read_to_string(&path).unwrap();
        let _parsed: serde_json::Value =
            serde_json::from_str(&content).expect("file must contain valid JSON");
    }

    #[test]
    fn write_training_metadata_fields_survive_write_read_cycle() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("metadata.json");
        let original = make_training_metadata();

        write_training_metadata(&path, &original).expect("write must succeed");
        let decoded = read_training_metadata(&path).expect("read must succeed");

        assert_eq!(decoded.iterations.completed, 100);
        assert!(decoded.convergence.achieved);
        assert_eq!(decoded.cuts.total_generated, 1_250_000);
    }

    #[test]
    fn write_simulation_metadata_fields_survive_write_read_cycle() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("metadata.json");
        let original = make_simulation_metadata();

        write_simulation_metadata(&path, &original).expect("write must succeed");
        let decoded = read_simulation_metadata(&path).expect("read must succeed");

        assert_eq!(decoded.scenarios.total, 100);
        assert_eq!(decoded.scenarios.completed, 100);
    }

    // ── Error handling ───────────────────────────────────────────────────────

    #[test]
    fn write_training_metadata_missing_parent_returns_io_error() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("nonexistent_subdir").join("metadata.json");
        let metadata = make_training_metadata();

        let result = write_training_metadata(&path, &metadata);

        assert!(
            matches!(result, Err(OutputError::IoError { .. })),
            "error must be IoError when parent directory is missing, got: {result:?}"
        );
    }

    #[test]
    fn write_simulation_metadata_missing_parent_returns_io_error() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("nonexistent_subdir").join("metadata.json");
        let metadata = make_simulation_metadata();

        let result = write_simulation_metadata(&path, &metadata);

        assert!(
            matches!(result, Err(OutputError::IoError { .. })),
            "error must be IoError when parent directory is missing"
        );
    }

    #[test]
    fn read_training_metadata_missing_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("nonexistent.json");

        let result = read_training_metadata(&path);

        assert!(
            matches!(result, Err(OutputError::IoError { .. })),
            "missing file must return OutputError::IoError, got: {result:?}"
        );
    }

    #[test]
    fn read_training_metadata_malformed_json() {
        use std::io::Write;
        let dir = tempdir().unwrap();
        let path = dir.path().join("metadata.json");
        let mut file = std::fs::File::create(&path).unwrap();
        writeln!(file, "{{not valid json at all").unwrap();

        let result = read_training_metadata(&path);

        assert!(
            matches!(result, Err(OutputError::ManifestError { .. })),
            "malformed JSON must return OutputError::ManifestError, got: {result:?}"
        );
    }

    // ── Atomic write ─────────────────────────────────────────────────────────

    #[test]
    fn write_metadata_atomic_no_tmp_remains() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("metadata.json");
        let metadata = make_training_metadata();

        write_training_metadata(&path, &metadata).expect("write must succeed");

        let tmp = path.with_extension("json.tmp");
        assert!(
            !tmp.exists(),
            "no .tmp file must remain after a successful write"
        );
        assert!(path.exists(), "the target file must exist");
    }

    // ── cobre_version ────────────────────────────────────────────────────────

    #[test]
    fn training_metadata_cobre_version_matches_cargo_pkg_version() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("metadata.json");
        let metadata = make_training_metadata();

        write_training_metadata(&path, &metadata).expect("write must succeed");

        let content = std::fs::read_to_string(&path).unwrap();
        let value: serde_json::Value = serde_json::from_str(&content).unwrap();

        let version = value["cobre_version"]
            .as_str()
            .expect("cobre_version must be a string");
        assert_eq!(version, env!("CARGO_PKG_VERSION"));
    }

    // ── Helpers ──────────────────────────────────────────────────────────────

    #[test]
    fn now_iso8601_returns_valid_format() {
        let ts = now_iso8601();
        // Must match pattern like "2026-04-05T14:30:00Z"
        assert!(ts.ends_with('Z'), "timestamp must end with Z: {ts}");
        assert!(ts.contains('T'), "timestamp must contain T separator: {ts}");
    }
}
