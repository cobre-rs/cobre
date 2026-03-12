//! Manifest and metadata writers for the output pipeline.
//!
//! This module provides JSON writers for three files that accompany Parquet
//! output artifacts:
//!
//! - [`write_simulation_manifest`] — writes `simulation/_manifest.json` tracking
//!   scenario completion status for crash recovery.
//! - [`write_training_manifest`] — writes `training/_manifest.json` tracking
//!   iteration and convergence state.
//! - [`write_metadata`] — writes `training/metadata.json` capturing configuration
//!   snapshots and environment information for reproducibility.
//!
//! All writers use an atomic write pattern: data is serialized to a `.tmp` file
//! first, then atomically renamed to the target path. This prevents partial files
//! from being visible to readers.
//!
//! # Minimal viable behaviour
//!
//! Several fields in the spec are deferred for later implementation:
//!
//! - `checksum` fields in manifests are always `null` (checksum computation is deferred).
//! - `data_integrity` in metadata is always `null` (hash computation is deferred).
//! - `performance_summary` in metadata is always `null` (detailed LP timing is deferred).
//! - `environment` fields other than `cobre_version` are always `null`.
//! - MPI fields (`world_size`, `ranks_participated`) are set to `1` (single-rank default).

use std::path::Path;

use serde::{Deserialize, Serialize};

use super::error::OutputError;

// ── Shared nested structs ─────────────────────────────────────────────────────

/// Integrity checksum for a manifest file.
///
/// In the minimal viable implementation all instances are `None` (checksum
/// computation is deferred). The field serializes as `"checksum": null` in JSON.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestChecksum {
    /// Checksum algorithm identifier (e.g. `"xxhash64"`).
    pub algorithm: String,
    /// Hex-encoded checksum value.
    pub value: String,
}

/// MPI participation information embedded in both manifest types.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestMpiInfo {
    /// Total number of MPI ranks in the communicator.
    pub world_size: u32,
    /// Number of ranks that actually wrote data.
    pub ranks_participated: u32,
}

impl Default for ManifestMpiInfo {
    /// Returns single-rank defaults for the minimal viable implementation.
    fn default() -> Self {
        Self {
            world_size: 1,
            ranks_participated: 1,
        }
    }
}

// ── SimulationManifest ────────────────────────────────────────────────────────

/// Scenario counts embedded in [`SimulationManifest`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestScenarios {
    /// Total number of scenarios dispatched for simulation.
    pub total: u32,
    /// Number of scenarios that completed without error.
    pub completed: u32,
    /// Number of scenarios that encountered a terminal error.
    pub failed: u32,
}

/// Manifest for the simulation output directory (`simulation/_manifest.json`).
///
/// Enables crash recovery: consumers read `status` to decide whether to resume
/// or restart, and read `partitions_written` to identify already-completed work.
///
/// Corresponds to the schema defined in output-infrastructure.md §1.1.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationManifest {
    /// Manifest schema version (`"2.0.0"`).
    pub version: String,
    /// Run status: `"running"`, `"complete"`, `"failed"`, or `"partial"`.
    pub status: String,
    /// ISO 8601 timestamp when the run started.
    pub started_at: Option<String>,
    /// ISO 8601 timestamp when the run completed (null while running).
    pub completed_at: Option<String>,
    /// Scenario completion counts.
    pub scenarios: ManifestScenarios,
    /// Relative paths to Hive partition directories written.
    pub partitions_written: Vec<String>,
    /// Integrity checksum over all written partitions (`null` in minimal viable version).
    pub checksum: Option<ManifestChecksum>,
    /// MPI participation information.
    pub mpi_info: ManifestMpiInfo,
}

// ── TrainingManifest ──────────────────────────────────────────────────────────

/// Iteration counts embedded in [`TrainingManifest`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestIterations {
    /// Maximum iterations allowed by the iteration-limit stopping rule.
    pub max_iterations: Option<u32>,
    /// Number of iterations actually completed.
    pub completed: u32,
    /// Iteration at which a convergence-oriented rule triggered (`null` if terminated by safety limit).
    pub converged_at: Option<u32>,
}

/// Convergence summary embedded in [`TrainingManifest`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestConvergence {
    /// Whether a convergence-oriented stopping rule triggered termination.
    pub achieved: bool,
    /// Final optimality gap in percent (`null` when upper bound evaluation is disabled).
    pub final_gap_percent: Option<f64>,
    /// Human-readable description of the rule that terminated the run.
    pub termination_reason: String,
}

/// Cut pool summary embedded in [`TrainingManifest`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestCuts {
    /// Total Benders cuts generated over the entire run.
    pub total_generated: u64,
    /// Cuts still active in the pool at termination.
    pub total_active: u64,
    /// Highest number of simultaneously active cuts observed.
    pub peak_active: u64,
}

/// Manifest for the training output directory (`training/_manifest.json`).
///
/// Enables crash recovery and post-run inspection. The `status` field indicates
/// whether the run completed normally; `iterations.completed` and
/// `convergence.achieved` capture the essential outcome.
///
/// Corresponds to the schema defined in output-infrastructure.md §1.2.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingManifest {
    /// Manifest schema version (`"2.0.0"`).
    pub version: String,
    /// Run status: `"running"`, `"complete"`, `"failed"`, or `"converged"`.
    pub status: String,
    /// ISO 8601 timestamp when training started.
    pub started_at: Option<String>,
    /// ISO 8601 timestamp when training completed (null while running).
    pub completed_at: Option<String>,
    /// Iteration completion counts.
    pub iterations: ManifestIterations,
    /// Convergence outcome.
    pub convergence: ManifestConvergence,
    /// Cut pool summary.
    pub cuts: ManifestCuts,
    /// Integrity checksum over policy and convergence files (`null` in minimal viable version).
    pub checksum: Option<ManifestChecksum>,
    /// MPI participation information.
    pub mpi_info: ManifestMpiInfo,
}

// ── TrainingMetadata ──────────────────────────────────────────────────────────

/// Run identification and timing embedded in [`TrainingMetadata`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetadataRunInfo {
    /// Unique run identifier (placeholder until uuid support is added).
    pub run_id: String,
    /// ISO 8601 timestamp when the run started.
    pub started_at: Option<String>,
    /// ISO 8601 timestamp when the run completed.
    pub completed_at: Option<String>,
    /// Total run duration in seconds.
    pub duration_seconds: Option<f64>,
    /// Version of the cobre-io crate that produced this output.
    pub cobre_version: String,
    /// LP solver backend identifier (e.g. `"highs"`).
    pub solver: Option<String>,
    /// LP solver library version.
    pub solver_version: Option<String>,
    /// Hostname of the primary compute node (`null` in minimal viable version).
    pub hostname: Option<String>,
    /// Username that initiated the run (`null` in minimal viable version).
    pub user: Option<String>,
}

/// Selected training configuration fields captured for reproducibility.
///
/// This is an informational snapshot, not a normative schema. The canonical
/// configuration schema lives in `config.json` (see `cobre-io::config`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetadataConfigSnapshot {
    /// Random seed used for scenario generation.
    pub seed: Option<i64>,
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

/// LP timing performance summary embedded in [`TrainingMetadata`].
///
/// All fields are `null` in the minimal viable implementation (detailed LP
/// timing is deferred).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetadataPerformanceSummary {
    /// Total number of LP solves across all iterations.
    pub total_lp_solves: Option<u64>,
    /// Average LP solve time in microseconds.
    pub avg_lp_time_us: Option<f64>,
    /// Median LP solve time in microseconds.
    pub median_lp_time_us: Option<f64>,
    /// 99th-percentile LP solve time in microseconds.
    pub p99_lp_time_us: Option<f64>,
    /// Peak resident memory in megabytes.
    pub peak_memory_mb: Option<f64>,
}

/// Cryptographic hashes for reproducibility verification.
///
/// All fields are `null` in the minimal viable implementation (hash computation
/// is deferred).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetadataDataIntegrity {
    /// SHA-256 of concatenated input file hashes.
    pub input_hash: Option<String>,
    /// SHA-256 of normalized `config.json`.
    pub config_hash: Option<String>,
    /// SHA-256 computed over the policy `FlatBuffers` files.
    pub policy_hash: Option<String>,
    /// SHA-256 of `training/convergence.parquet` content.
    pub convergence_hash: Option<String>,
}

/// Runtime environment information embedded in [`TrainingMetadata`].
///
/// Only `cobre_version` is populated in the minimal viable implementation;
/// all other fields are `null` (environment introspection is deferred).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetadataEnvironment {
    /// MPI implementation name (e.g. `"OpenMPI"`) — `null` in minimal viable version.
    pub mpi_implementation: Option<String>,
    /// MPI library version string — `null` in minimal viable version.
    pub mpi_version: Option<String>,
    /// Number of MPI ranks — `null` in minimal viable version.
    pub num_ranks: Option<u32>,
    /// Number of CPU cores per rank — `null` in minimal viable version.
    pub cpus_per_rank: Option<u32>,
    /// Memory per rank in gigabytes — `null` in minimal viable version.
    pub memory_per_rank_gb: Option<f64>,
}

/// Comprehensive metadata file (`training/metadata.json`).
///
/// Captures the configuration snapshot, problem dimensions, performance
/// summary, data integrity hashes, and environment information for
/// reproducibility and audit trail purposes.
///
/// Corresponds to the schema defined in output-infrastructure.md §2.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingMetadata {
    /// Metadata schema version (`"2.0.0"`).
    pub version: String,
    /// Run identification and timing.
    pub run_info: MetadataRunInfo,
    /// Snapshot of key configuration fields.
    pub configuration_snapshot: MetadataConfigSnapshot,
    /// Problem size dimensions.
    pub problem_dimensions: MetadataProblemDimensions,
    /// LP performance statistics (`null` in minimal viable version).
    pub performance_summary: Option<MetadataPerformanceSummary>,
    /// Data integrity hashes (`null` in minimal viable version).
    pub data_integrity: Option<MetadataDataIntegrity>,
    /// Runtime environment (`null` fields in minimal viable version).
    pub environment: MetadataEnvironment,
}

/// Write a simulation manifest to `path` using the atomic write pattern.
///
/// Serializes `manifest` to pretty-printed JSON, writes to `path.json.tmp`,
/// then atomically renames to `path`. If the parent directory does not exist,
/// returns [`OutputError::IoError`].
///
/// # Errors
///
/// - [`OutputError::ManifestError`] if JSON serialization fails.
/// - [`OutputError::IoError`] if the file write or atomic rename fails.
pub fn write_simulation_manifest(
    path: &Path,
    manifest: &SimulationManifest,
) -> Result<(), OutputError> {
    write_json_atomic(path, manifest, "simulation")
}

/// Write a training manifest to `path` using the atomic write pattern.
///
/// Serializes `manifest` to pretty-printed JSON, writes to `path.json.tmp`,
/// then atomically renames to `path`. If the parent directory does not exist,
/// returns [`OutputError::IoError`].
///
/// # Errors
///
/// - [`OutputError::ManifestError`] if JSON serialization fails.
/// - [`OutputError::IoError`] if the file write or atomic rename fails.
pub fn write_training_manifest(
    path: &Path,
    manifest: &TrainingManifest,
) -> Result<(), OutputError> {
    write_json_atomic(path, manifest, "training")
}

/// Write a training metadata file to `path` using the atomic write pattern.
///
/// Serializes `metadata` to pretty-printed JSON, writes to `path.json.tmp`,
/// then atomically renames to `path`. If the parent directory does not exist,
/// returns [`OutputError::IoError`].
///
/// # Errors
///
/// - [`OutputError::ManifestError`] if JSON serialization fails.
/// - [`OutputError::IoError`] if the file write or atomic rename fails.
pub fn write_metadata(path: &Path, metadata: &TrainingMetadata) -> Result<(), OutputError> {
    write_json_atomic(path, metadata, "metadata")
}

/// Read a training manifest from `path`.
///
/// Reads the file at `path`, then deserializes the JSON content into a
/// [`TrainingManifest`].
///
/// # Errors
///
/// - [`OutputError::IoError`] if the file cannot be read (e.g., not found).
/// - [`OutputError::ManifestError`] if the file contains malformed JSON.
pub fn read_training_manifest(path: &Path) -> Result<TrainingManifest, OutputError> {
    read_json(path, "training")
}

/// Read a simulation manifest from `path`.
///
/// Reads the file at `path`, then deserializes the JSON content into a
/// [`SimulationManifest`].
///
/// # Errors
///
/// - [`OutputError::IoError`] if the file cannot be read (e.g., not found).
/// - [`OutputError::ManifestError`] if the file contains malformed JSON.
pub fn read_simulation_manifest(path: &Path) -> Result<SimulationManifest, OutputError> {
    read_json(path, "simulation")
}

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
///
/// The write sequence is:
/// 1. Serialize to a `String` via `serde_json::to_string_pretty`.
/// 2. Write the string to `{path}.tmp` (a sibling temp file).
/// 3. Atomically rename `{path}.tmp` to `path`.
///
/// If the parent directory does not exist, step 2 returns `IoError` immediately
/// before any rename is attempted.
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

    fn make_simulation_manifest() -> SimulationManifest {
        SimulationManifest {
            version: "2.0.0".to_string(),
            status: "complete".to_string(),
            started_at: Some("2026-01-17T10:00:00Z".to_string()),
            completed_at: Some("2026-01-17T10:15:00Z".to_string()),
            scenarios: ManifestScenarios {
                total: 100,
                completed: 100,
                failed: 0,
            },
            partitions_written: vec!["scenario_id=0/".to_string(), "scenario_id=1/".to_string()],
            checksum: None,
            mpi_info: ManifestMpiInfo::default(),
        }
    }

    fn make_training_manifest() -> TrainingManifest {
        TrainingManifest {
            version: "2.0.0".to_string(),
            status: "complete".to_string(),
            started_at: Some("2026-01-17T08:00:00Z".to_string()),
            completed_at: Some("2026-01-17T12:30:00Z".to_string()),
            iterations: ManifestIterations {
                max_iterations: Some(100),
                completed: 10,
                converged_at: Some(10),
            },
            convergence: ManifestConvergence {
                achieved: true,
                final_gap_percent: Some(0.45),
                termination_reason: "bound_stalling".to_string(),
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

    fn make_training_metadata() -> TrainingMetadata {
        TrainingMetadata {
            version: "2.0.0".to_string(),
            run_info: MetadataRunInfo {
                run_id: "not-implemented".to_string(),
                started_at: Some("2026-01-17T08:00:00Z".to_string()),
                completed_at: Some("2026-01-17T12:30:00Z".to_string()),
                duration_seconds: Some(16_200.0),
                cobre_version: env!("CARGO_PKG_VERSION").to_string(),
                solver: Some("highs".to_string()),
                solver_version: None,
                hostname: None,
                user: None,
            },
            configuration_snapshot: MetadataConfigSnapshot {
                seed: Some(42),
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
            performance_summary: None,
            data_integrity: None,
            environment: MetadataEnvironment {
                mpi_implementation: None,
                mpi_version: None,
                num_ranks: None,
                cpus_per_rank: None,
                memory_per_rank_gb: None,
            },
        }
    }

    // ── Roundtrip tests ───────────────────────────────────────────────────────

    #[test]
    fn simulation_manifest_roundtrip() {
        let original = make_simulation_manifest();
        let json = serde_json::to_string_pretty(&original).unwrap();
        let decoded: SimulationManifest = serde_json::from_str(&json).unwrap();

        assert_eq!(decoded.version, original.version);
        assert_eq!(decoded.status, original.status);
        assert_eq!(decoded.scenarios.total, original.scenarios.total);
        assert_eq!(decoded.scenarios.completed, original.scenarios.completed);
        assert_eq!(decoded.scenarios.failed, original.scenarios.failed);
        assert_eq!(
            decoded.partitions_written.len(),
            original.partitions_written.len()
        );
        assert!(decoded.checksum.is_none());
        assert_eq!(decoded.mpi_info.world_size, original.mpi_info.world_size);
        assert_eq!(
            decoded.mpi_info.ranks_participated,
            original.mpi_info.ranks_participated
        );
    }

    #[test]
    fn training_manifest_roundtrip() {
        let original = make_training_manifest();
        let json = serde_json::to_string_pretty(&original).unwrap();
        let decoded: TrainingManifest = serde_json::from_str(&json).unwrap();

        assert_eq!(decoded.version, original.version);
        assert_eq!(decoded.status, original.status);
        assert_eq!(decoded.iterations.completed, original.iterations.completed);
        assert_eq!(
            decoded.iterations.max_iterations,
            original.iterations.max_iterations
        );
        assert_eq!(
            decoded.iterations.converged_at,
            original.iterations.converged_at
        );
        assert_eq!(decoded.convergence.achieved, original.convergence.achieved);
        assert_eq!(
            decoded.convergence.final_gap_percent,
            original.convergence.final_gap_percent
        );
        assert_eq!(
            decoded.convergence.termination_reason,
            original.convergence.termination_reason
        );
        assert_eq!(decoded.cuts.total_generated, original.cuts.total_generated);
        assert_eq!(decoded.cuts.total_active, original.cuts.total_active);
        assert_eq!(decoded.cuts.peak_active, original.cuts.peak_active);
        assert!(decoded.checksum.is_none());
    }

    #[test]
    fn training_metadata_serialization() {
        let metadata = make_training_metadata();
        let json = serde_json::to_string_pretty(&metadata).unwrap();
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();

        // Verify key paths exist in the serialized JSON.
        assert!(
            value["run_info"].is_object(),
            "run_info must be a JSON object"
        );
        assert!(
            value["run_info"]["cobre_version"].is_string(),
            "run_info.cobre_version must be a string"
        );
        assert!(
            value["configuration_snapshot"].is_object(),
            "configuration_snapshot must be a JSON object"
        );
        assert!(
            value["problem_dimensions"].is_object(),
            "problem_dimensions must be a JSON object"
        );
        assert!(
            value["performance_summary"].is_null(),
            "performance_summary must be null in minimal viable version"
        );
        assert!(
            value["data_integrity"].is_null(),
            "data_integrity must be null in minimal viable version"
        );
        assert!(
            value["environment"].is_object(),
            "environment must be a JSON object"
        );
    }

    // ── Writer tests ──────────────────────────────────────────────────────────

    #[test]
    fn write_simulation_manifest_creates_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("_manifest.json");
        let manifest = make_simulation_manifest();

        write_simulation_manifest(&path, &manifest).expect("write must succeed");

        assert!(path.exists(), "manifest file must exist after write");
        let content = std::fs::read_to_string(&path).unwrap();
        let _parsed: serde_json::Value =
            serde_json::from_str(&content).expect("file must contain valid JSON");
    }

    #[test]
    fn write_training_manifest_creates_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("_manifest.json");
        let manifest = make_training_manifest();

        write_training_manifest(&path, &manifest).expect("write must succeed");

        assert!(path.exists(), "manifest file must exist after write");
        let content = std::fs::read_to_string(&path).unwrap();
        let _parsed: serde_json::Value =
            serde_json::from_str(&content).expect("file must contain valid JSON");
    }

    #[test]
    fn write_metadata_creates_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("metadata.json");
        let metadata = make_training_metadata();

        write_metadata(&path, &metadata).expect("write must succeed");

        assert!(path.exists(), "metadata file must exist after write");
        let content = std::fs::read_to_string(&path).unwrap();
        let _parsed: serde_json::Value =
            serde_json::from_str(&content).expect("file must contain valid JSON");
    }

    // ── Acceptance criterion: training manifest field values ──────────────────

    #[test]
    fn write_training_manifest_fields_survive_write_read_cycle() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("_manifest.json");

        let manifest = TrainingManifest {
            version: "2.0.0".to_string(),
            status: "complete".to_string(),
            started_at: None,
            completed_at: None,
            iterations: ManifestIterations {
                max_iterations: Some(100),
                completed: 10,
                converged_at: Some(10),
            },
            convergence: ManifestConvergence {
                achieved: true,
                final_gap_percent: None,
                termination_reason: "iteration_limit".to_string(),
            },
            cuts: ManifestCuts {
                total_generated: 200,
                total_active: 80,
                peak_active: 95,
            },
            checksum: None,
            mpi_info: ManifestMpiInfo::default(),
        };

        write_training_manifest(&path, &manifest).expect("write must succeed");

        let content = std::fs::read_to_string(&path).unwrap();
        let decoded: TrainingManifest = serde_json::from_str(&content).unwrap();

        assert_eq!(
            decoded.iterations.completed, 10,
            "iterations.completed must round-trip correctly"
        );
        assert!(
            decoded.convergence.achieved,
            "convergence.achieved must round-trip correctly"
        );
    }

    // ── Acceptance criterion: simulation manifest JSON values ─────────────────

    #[test]
    fn write_simulation_manifest_json_field_values() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("_manifest.json");

        let manifest = SimulationManifest {
            version: "2.0.0".to_string(),
            status: "complete".to_string(),
            started_at: None,
            completed_at: None,
            scenarios: ManifestScenarios {
                total: 100,
                completed: 100,
                failed: 0,
            },
            partitions_written: vec![],
            checksum: None,
            mpi_info: ManifestMpiInfo::default(),
        };

        write_simulation_manifest(&path, &manifest).expect("write must succeed");

        let content = std::fs::read_to_string(&path).unwrap();
        let value: serde_json::Value = serde_json::from_str(&content).unwrap();

        assert_eq!(
            value["scenarios"]["total"].as_u64(),
            Some(100),
            "$.scenarios.total must equal 100"
        );
        assert_eq!(
            value["scenarios"]["completed"].as_u64(),
            Some(100),
            "$.scenarios.completed must equal 100"
        );
    }

    // ── Acceptance criterion: cobre_version from CARGO_PKG_VERSION ───────────

    #[test]
    fn write_metadata_cobre_version_matches_cargo_pkg_version() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("metadata.json");
        let metadata = make_training_metadata();

        write_metadata(&path, &metadata).expect("write must succeed");

        let content = std::fs::read_to_string(&path).unwrap();
        let value: serde_json::Value = serde_json::from_str(&content).unwrap();

        let version = value["run_info"]["cobre_version"]
            .as_str()
            .expect("run_info.cobre_version must be a string");
        assert_eq!(
            version,
            env!("CARGO_PKG_VERSION"),
            "cobre_version must equal CARGO_PKG_VERSION"
        );
    }

    // ── Acceptance criterion: missing parent directory returns IoError ────────

    #[test]
    fn write_manifest_missing_parent_directory_returns_io_error() {
        let dir = tempdir().unwrap();
        // Reference a non-existent subdirectory.
        let path = dir.path().join("nonexistent_subdir").join("_manifest.json");
        let manifest = make_simulation_manifest();

        let result = write_simulation_manifest(&path, &manifest);

        assert!(
            result.is_err(),
            "write must fail when parent directory does not exist"
        );
        assert!(
            matches!(result, Err(OutputError::IoError { .. })),
            "error must be IoError when parent directory is missing, got: {result:?}"
        );
    }

    #[test]
    fn write_training_manifest_missing_parent_returns_io_error() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("nonexistent_subdir").join("_manifest.json");
        let manifest = make_training_manifest();

        let result = write_training_manifest(&path, &manifest);

        assert!(
            matches!(result, Err(OutputError::IoError { .. })),
            "error must be IoError when parent directory is missing"
        );
    }

    #[test]
    fn write_metadata_missing_parent_returns_io_error() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("nonexistent_subdir").join("metadata.json");
        let metadata = make_training_metadata();

        let result = write_metadata(&path, &metadata);

        assert!(
            matches!(result, Err(OutputError::IoError { .. })),
            "error must be IoError when parent directory is missing"
        );
    }

    // ── Acceptance criterion: no .tmp file remains after successful write ─────

    #[test]
    fn write_manifest_atomic_no_tmp_remains() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("_manifest.json");
        let manifest = make_simulation_manifest();

        write_simulation_manifest(&path, &manifest).expect("write must succeed");

        let tmp = path.with_extension("json.tmp");
        assert!(
            !tmp.exists(),
            "no .tmp file must remain after a successful write, but found: {}",
            tmp.display()
        );
        assert!(path.exists(), "the target file must exist");
    }

    // ── Acceptance criterion: checksum serializes as null ─────────────────────

    #[test]
    fn manifest_null_checksum_serializes() {
        let manifest = SimulationManifest {
            version: "2.0.0".to_string(),
            status: "complete".to_string(),
            started_at: None,
            completed_at: None,
            scenarios: ManifestScenarios {
                total: 0,
                completed: 0,
                failed: 0,
            },
            partitions_written: vec![],
            checksum: None,
            mpi_info: ManifestMpiInfo::default(),
        };

        let json = serde_json::to_string_pretty(&manifest).unwrap();
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();

        assert!(
            value["checksum"].is_null(),
            "checksum: None must serialize as null in JSON, got: {}",
            value["checksum"]
        );
    }

    #[test]
    fn training_manifest_null_checksum_serializes() {
        let manifest = make_training_manifest();
        let json = serde_json::to_string_pretty(&manifest).unwrap();
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();

        assert!(
            value["checksum"].is_null(),
            "checksum: None must serialize as null in training manifest"
        );
    }

    // ── Reader tests ──────────────────────────────────────────────────────────

    #[test]
    fn read_training_manifest_roundtrip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("_manifest.json");
        let original = make_training_manifest();

        write_training_manifest(&path, &original).expect("write must succeed");
        let decoded = read_training_manifest(&path).expect("read must succeed");

        assert_eq!(decoded.version, original.version);
        assert_eq!(decoded.status, original.status);
        assert_eq!(decoded.iterations.completed, original.iterations.completed);
        assert_eq!(
            decoded.iterations.max_iterations,
            original.iterations.max_iterations
        );
        assert_eq!(decoded.convergence.achieved, original.convergence.achieved);
        assert_eq!(
            decoded.convergence.final_gap_percent,
            original.convergence.final_gap_percent
        );
        assert_eq!(
            decoded.convergence.termination_reason,
            original.convergence.termination_reason
        );
        assert_eq!(decoded.cuts.total_generated, original.cuts.total_generated);
        assert_eq!(decoded.cuts.total_active, original.cuts.total_active);
        assert_eq!(decoded.cuts.peak_active, original.cuts.peak_active);
    }

    #[test]
    fn read_training_manifest_missing_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("nonexistent_manifest.json");

        let result = read_training_manifest(&path);

        assert!(
            matches!(result, Err(OutputError::IoError { .. })),
            "missing file must return OutputError::IoError, got: {result:?}"
        );
    }

    #[test]
    fn read_simulation_manifest_roundtrip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("_manifest.json");
        let original = make_simulation_manifest();

        write_simulation_manifest(&path, &original).expect("write must succeed");
        let decoded = read_simulation_manifest(&path).expect("read must succeed");

        assert_eq!(decoded.version, original.version);
        assert_eq!(decoded.status, original.status);
        assert_eq!(decoded.scenarios.total, original.scenarios.total);
        assert_eq!(decoded.scenarios.completed, original.scenarios.completed);
        assert_eq!(decoded.scenarios.failed, original.scenarios.failed);
        assert_eq!(
            decoded.partitions_written.len(),
            original.partitions_written.len()
        );
    }

    #[test]
    fn read_training_manifest_malformed_json_returns_manifest_error() {
        use std::io::Write;
        let dir = tempdir().unwrap();
        let path = dir.path().join("_manifest.json");
        let mut file = std::fs::File::create(&path).unwrap();
        writeln!(file, "{{not valid json at all").unwrap();

        let result = read_training_manifest(&path);

        assert!(
            matches!(result, Err(OutputError::ManifestError { .. })),
            "malformed JSON must return OutputError::ManifestError, got: {result:?}"
        );
    }
}
