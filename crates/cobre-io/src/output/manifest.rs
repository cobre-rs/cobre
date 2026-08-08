//! JSON metadata writers for the output pipeline:
//!
//! - [`write_training_metadata`] — `training/metadata.json` (run context,
//!   configuration, convergence, row-pool statistics).
//! - [`write_simulation_metadata`] — `simulation/metadata.json` (run context,
//!   scenario completion counts).
//!
//! Each output directory gets a single merged metadata file; the `_SUCCESS`
//! marker signals completion.

use std::path::Path;

use serde::{Deserialize, Serialize};

use super::atomic::write_bytes_atomic;
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
    /// LP solver backend name (e.g. `"highs"` or `"clp"`).
    pub solver: String,
    /// LP solver version string (e.g. `"1.8.0"`), if known.
    pub solver_version: Option<String>,
    /// ISO 8601 timestamp when the phase started.
    pub started_at: String,
    /// ISO 8601 timestamp when the phase completed.
    pub completed_at: String,
    /// Execution distribution and environment information.
    pub distribution: DistributionInfo,
    /// Per-phase setup wall time, collected on the rank that writes metadata.
    /// `None` when the producer did not collect setup timings (e.g. the
    /// simulation context, or a producer that has not wired collection); the
    /// metadata `setup` section is then omitted.
    pub setup: Option<SetupTimings>,
    /// Run-level rollup of the per-entity production-model fit deviation.
    /// `None` when the producer fitted no model whose deviation is measured
    /// (e.g. the simulation context, or a run with no computed production
    /// model); the metadata `production_fit_deviation` section is then omitted.
    pub production_fit_deviation: Option<DeviationSummary>,
}

/// Read the system hostname via the `gethostname` syscall.
///
/// Falls back to `"unknown"` when the syscall yields an empty name.
#[must_use]
pub fn get_hostname() -> String {
    let name = gethostname::gethostname().to_string_lossy().into_owned();
    if name.is_empty() {
        "unknown".to_string()
    } else {
        name
    }
}

/// Return the current UTC time as an ISO 8601 string (e.g. `"2026-04-05T14:30:00Z"`).
#[must_use]
pub fn now_iso8601() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
}

// ── Shared nested structs ────────────────────────────────────────────────────

/// Per-host rank assignment for a single physical host.
///
/// Captures which global ranks were placed on a given host, enabling
/// reconstruction of the multi-host process layout from persisted metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HostLayout {
    /// Hostname as reported by the backend.
    pub hostname: String,
    /// Sorted global ranks assigned to this host.
    pub ranks: Vec<u32>,
}

/// CPU topology and worker binding resolved for one process/rank.
///
/// These fields are operational telemetry: changing the execution shape may
/// change them without changing deterministic numerical results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankAffinity {
    /// Global communicator rank (`0` for the local backend and Python).
    pub rank: u32,
    /// Requested binding policy: `"none"`, `"core"`, or `"numa"`.
    pub policy: String,
    /// Logical CPUs online on the host, including CPUs outside this rank's set.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub online_processing_units: Option<u32>,
    /// Logical CPUs visible to this rank.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub visible_processing_units: Option<u32>,
    /// Physical cores visible to this rank.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub physical_cores: Option<u32>,
    /// NUMA nodes intersecting this rank's visible CPU set.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub numa_nodes: Option<u32>,
    /// Visible operating-system CPU identifiers.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub visible_cpus: Vec<u32>,
    /// Active memory policy, when readable.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory_policy: Option<String>,
    /// NUMA nodes selected by the active memory policy.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub memory_policy_nodes: Vec<u32>,
    /// NUMA nodes allowed for memory allocation.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub allowed_memory_nodes: Vec<u32>,
    /// Non-fatal memory-binding discovery failure.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory_discovery_error: Option<String>,
    /// Worker-index to operating-system CPU mapping; empty when unbound.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub worker_cpus: Vec<u32>,
    /// Discovery error retained when the unbound policy continues safely.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub discovery_error: Option<String>,
}

/// Execution distribution information embedded in metadata files.
///
/// Captures the communication backend, process topology, and optional
/// MPI/scheduler metadata for reproducibility.
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
    /// Per-host rank assignment for multi-node runs. Empty for single-host or
    /// local runs.
    #[serde(default)]
    pub hosts: Vec<HostLayout>,
    /// Per-rank CPU topology and resolved worker placement. Absent in metadata
    /// written by versions before NUMA-aware execution support.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub rank_affinity: Vec<RankAffinity>,
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

/// Row-pool summary embedded in [`TrainingMetadata`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetadataRowPool {
    /// Total rows generated over the entire run.
    pub total_generated: u64,
    /// Rows still active in the pool at termination.
    pub total_active: u64,
    /// Highest number of simultaneously active rows observed.
    pub peak_active: u64,
    /// Rows currently active in the LP at termination.
    #[serde(default)]
    pub cuts_active: u64,
    /// Sum of resident rows-in-LP over every lazy-selection solve in the run
    /// (reduced across ranks). With `rows_in_lp_solve_count`, the mean per solve.
    /// Zero when no lazy selection ran. `serde(default)` for old-metadata reads.
    #[serde(default)]
    pub rows_in_lp_total: u64,
    /// Number of lazy-selection solves in the run (reduced across ranks).
    #[serde(default)]
    pub rows_in_lp_solve_count: u64,
    /// Largest resident rows-in-LP over any single lazy-selection solve
    /// (reduced across ranks). Zero when no lazy selection ran.
    #[serde(default)]
    pub rows_in_lp_max: u64,
}

/// Final objective bounds embedded in [`TrainingMetadata`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetadataBounds {
    /// Final lower bound on the objective at termination.
    pub final_lower_bound: f64,
    /// Final upper bound estimate (`null` when upper-bound evaluation is disabled).
    pub final_upper_bound: Option<f64>,
    /// Standard deviation of the final upper-bound estimate (`null` when unavailable).
    pub final_upper_bound_std: Option<f64>,
}

/// Default bounds (`final_lower_bound` `0.0`, absent upper bounds) used when
/// metadata omits the `bounds` field.
#[must_use]
pub fn default_bounds() -> MetadataBounds {
    MetadataBounds {
        final_lower_bound: 0.0,
        final_upper_bound: None,
        final_upper_bound_std: None,
    }
}

/// Training solve statistics embedded in [`TrainingMetadata`].
///
/// Each field is optional so that absence is represented faithfully when the
/// producing run did not record the corresponding statistic.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MetadataTrainingSolveStats {
    /// Total number of LP solves performed during training.
    pub total_lp_solves: Option<u64>,
    /// Number of LP solves that succeeded on the first attempt.
    pub first_try: Option<u64>,
    /// Number of LP solves that succeeded after one or more retries.
    pub retried: Option<u64>,
    /// Number of LP solves that failed terminally.
    pub failed: Option<u64>,
    /// Cumulative wall-clock seconds spent in forward-phase LP solves.
    pub forward_solve_seconds: Option<f64>,
    /// Cumulative wall-clock seconds spent in backward-phase LP solves.
    pub backward_solve_seconds: Option<f64>,
    /// Degree of parallelism (e.g. worker count) used during training.
    pub parallelism: Option<u32>,
}

/// Per-phase setup wall time (wall-clock seconds) embedded in
/// [`TrainingMetadata`].
///
/// **Non-deterministic** — informational, never hashed, excluded from any parity
/// computation. All fields are `#[serde(default)]`, so pre-section metadata reads
/// back as zeros.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SetupTimings {
    /// Wall-clock seconds spent loading the input case.
    #[serde(default)]
    pub load_seconds: f64,
    /// Wall-clock seconds spent fitting the stochastic process.
    #[serde(default)]
    pub stochastic_fit_seconds: f64,
    /// Wall-clock seconds spent fitting the production model.
    #[serde(default)]
    pub production_fit_seconds: f64,
    /// Wall-clock seconds spent fitting the evaporation model.
    #[serde(default)]
    pub evaporation_fit_seconds: f64,
    /// Wall-clock seconds spent broadcasting setup data across ranks.
    #[serde(default)]
    pub broadcast_seconds: f64,
}

/// Run-level rollup of a per-entity model-fit deviation, embedded in
/// [`TrainingMetadata`].
///
/// Field names are deliberately generic — the producer maps its records into
/// this shape — so the metadata struct stays free of algorithm vocabulary. All
/// fields are `#[serde(default)]`, so pre-section metadata reads back as zeros /
/// `None`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DeviationSummary {
    /// Number of per-entity entries the rollup summarizes (`>= 1` whenever the
    /// summary is present; the producer omits the section for an empty set).
    #[serde(default)]
    pub n_entries: u32,
    /// Arithmetic mean of the per-entry mean absolute deviation magnitudes.
    #[serde(default)]
    pub mean_abs: f64,
    /// Maximum of the per-entry max absolute deviation magnitudes.
    #[serde(default)]
    pub max_abs: f64,
    /// Largest per-entry relative (dimensionless) deviation across all entries.
    #[serde(default)]
    pub worst_relative: f64,
    /// The entry with the largest relative deviation, when any entry exists.
    #[serde(default)]
    pub worst_entry: Option<DeviationWorstEntry>,
}

/// The single worst per-entity entry of a [`DeviationSummary`].
///
/// Field names are deliberately generic (no producer-specific vocabulary); all
/// fields are `#[serde(default)]`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DeviationWorstEntry {
    /// Integer identifier of the entity owning the worst entry.
    #[serde(default)]
    pub entity_id: i32,
    /// Integer identifier of the stage owning the worst entry.
    #[serde(default)]
    pub stage_id: i32,
    /// Relative (dimensionless) deviation of the worst entry.
    #[serde(default)]
    pub relative: f64,
    /// Mean absolute deviation magnitude of the worst entry.
    #[serde(default)]
    pub mean_abs: f64,
    /// Max absolute deviation magnitude of the worst entry.
    #[serde(default)]
    pub max_abs: f64,
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

/// Aggregate cost statistics embedded in [`SimulationMetadata`].
///
/// Captures the expected total cost across the simulated scenarios together
/// with its dispersion and a tail-risk summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetadataCost {
    /// Mean total cost across simulated scenarios.
    pub mean_cost: f64,
    /// Standard deviation of the total cost across simulated scenarios.
    pub std_cost: f64,
    /// Conditional Value-at-Risk at `cvar_alpha`.
    pub cvar: f64,
    /// Confidence level used for the `CVaR` computation, in `(0, 1)`.
    pub cvar_alpha: f64,
}

/// Simulation solve statistics embedded in [`SimulationMetadata`].
///
/// Each field is optional so that absence is represented faithfully when the
/// producing run did not record the corresponding statistic.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MetadataSimulationSolveStats {
    /// Total number of LP solves performed during simulation.
    pub total_lp_solves: Option<u64>,
    /// Number of LP solves that succeeded on the first attempt.
    pub first_try: Option<u64>,
    /// Number of LP solves that succeeded after one or more retries.
    pub retried: Option<u64>,
    /// Number of LP solves that failed terminally.
    pub failed: Option<u64>,
    /// Cumulative wall-clock seconds spent in simulation LP solves.
    pub solve_seconds: Option<f64>,
    /// Degree of parallelism (e.g. worker count) used during simulation.
    pub parallelism: Option<u32>,
}

// ── TrainingMetadata ─────────────────────────────────────────────────────────

/// Merged metadata for the training output directory (`training/metadata.json`).
///
/// A single file containing all run information: convergence, cuts,
/// configuration, and environment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingMetadata {
    /// Version of the cobre crate that produced this output.
    pub cobre_version: String,
    /// Hostname of the machine that ran training.
    pub hostname: String,
    /// LP solver backend name (e.g. `"highs"` or `"clp"`).
    pub solver: String,
    /// LP solver version string (e.g. `"1.8.0"`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub solver_version: Option<String>,
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
    /// Row-pool summary.
    pub row_pool: MetadataRowPool,
    /// Final objective bounds at termination.
    #[serde(default = "default_bounds")]
    pub bounds: MetadataBounds,
    /// Training solve statistics.
    #[serde(default)]
    pub solve_stats: MetadataTrainingSolveStats,
    /// Per-phase setup wall time (informational, non-deterministic). Absent
    /// when a run did not collect setup timings and from any legacy metadata.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub setup: Option<SetupTimings>,
    /// Run-level rollup of the per-entity production-model fit deviation
    /// (informational, never hashed). Absent when a run fitted no model whose
    /// deviation is measured and from any legacy metadata.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub production_fit_deviation: Option<DeviationSummary>,
    /// Execution distribution and environment information.
    pub distribution: DistributionInfo,
}

// ── SimulationMetadata ───────────────────────────────────────────────────────

/// Metadata for the simulation output directory (`simulation/metadata.json`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationMetadata {
    /// Version of the cobre crate that produced this output.
    pub cobre_version: String,
    /// Hostname of the machine that ran simulation.
    pub hostname: String,
    /// LP solver backend name (e.g. `"highs"` or `"clp"`).
    pub solver: String,
    /// LP solver version string (e.g. `"1.8.0"`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub solver_version: Option<String>,
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
    /// Aggregate cost statistics (`null` when cost was not persisted).
    #[serde(default)]
    pub cost: Option<MetadataCost>,
    /// Simulation solve statistics.
    #[serde(default)]
    pub solve_stats: MetadataSimulationSolveStats,
    /// Execution distribution and environment information.
    pub distribution: DistributionInfo,
}

// ── Writers ──────────────────────────────────────────────────────────────────

/// Write training metadata to `path` atomically.
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

/// Write simulation metadata to `path` atomically.
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
    // Serialize here, not via the shared JSON helper, to keep the `ManifestError`
    // failure variant callers expect (bytes are identical); the shared helper
    // then owns the crash-safe write.
    let json = serde_json::to_string_pretty(value).map_err(|e| OutputError::ManifestError {
        manifest_type: manifest_type.to_string(),
        message: e.to_string(),
    })?;

    write_bytes_atomic(path, json.as_bytes())
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
            hosts: Vec::new(),
            rank_affinity: Vec::new(),
        }
    }

    fn make_training_metadata() -> TrainingMetadata {
        TrainingMetadata {
            cobre_version: env!("CARGO_PKG_VERSION").to_string(),
            hostname: "test-host".to_string(),
            solver: "highs".to_string(),
            solver_version: Some("1.8.0".to_string()),
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
            row_pool: MetadataRowPool {
                total_generated: 1_250_000,
                total_active: 980_000,
                peak_active: 1_100_000,
                cuts_active: 980_000,
                rows_in_lp_total: 0,
                rows_in_lp_solve_count: 0,
                rows_in_lp_max: 0,
            },
            bounds: MetadataBounds {
                final_lower_bound: 48_500.0,
                final_upper_bound: Some(49_000.0),
                final_upper_bound_std: Some(250.0),
            },
            solve_stats: MetadataTrainingSolveStats {
                total_lp_solves: Some(84_000),
                first_try: Some(80_000),
                retried: Some(3_800),
                failed: Some(200),
                forward_solve_seconds: Some(123.5),
                backward_solve_seconds: Some(456.75),
                parallelism: Some(8),
            },
            setup: None,
            production_fit_deviation: None,
            distribution: make_distribution_info(),
        }
    }

    fn make_simulation_metadata() -> SimulationMetadata {
        SimulationMetadata {
            cobre_version: env!("CARGO_PKG_VERSION").to_string(),
            hostname: "test-host".to_string(),
            solver: "highs".to_string(),
            solver_version: Some("1.8.0".to_string()),
            started_at: "2026-01-17T13:00:00Z".to_string(),
            completed_at: "2026-01-17T13:15:00Z".to_string(),
            duration_seconds: 900.0,
            status: "complete".to_string(),
            scenarios: MetadataScenarios {
                total: 100,
                completed: 100,
                failed: 0,
            },
            cost: Some(MetadataCost {
                mean_cost: 12_345.6,
                std_cost: 200.0,
                cvar: 13_000.0,
                cvar_alpha: 0.95,
            }),
            solve_stats: MetadataSimulationSolveStats {
                total_lp_solves: Some(50_000),
                first_try: Some(48_000),
                retried: Some(1_900),
                failed: Some(100),
                solve_seconds: Some(321.0),
                parallelism: Some(8),
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
        assert_eq!(
            decoded.row_pool.total_generated,
            original.row_pool.total_generated
        );
        assert_eq!(
            decoded.row_pool.total_active,
            original.row_pool.total_active
        );
        assert_eq!(decoded.row_pool.peak_active, original.row_pool.peak_active);
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

    #[test]
    fn simulation_metadata_cost_round_trip() {
        let original = SimulationMetadata {
            cost: Some(MetadataCost {
                mean_cost: 12_345.6,
                std_cost: 200.0,
                cvar: 13_000.0,
                cvar_alpha: 0.95,
            }),
            ..make_simulation_metadata()
        };

        let json = serde_json::to_string(&original).unwrap();
        assert!(
            json.contains(r#""mean_cost":12345.6"#),
            "serialized JSON must contain the mean cost, got: {json}"
        );
        assert!(
            json.contains(r#""cvar_alpha":0.95"#),
            "serialized JSON must contain the CVaR alpha, got: {json}"
        );

        let decoded: SimulationMetadata = serde_json::from_str(&json).unwrap();
        let cost = decoded.cost.expect("cost must be present after round-trip");
        assert_eq!(cost.mean_cost, 12_345.6);
        assert_eq!(cost.std_cost, 200.0);
        assert_eq!(cost.cvar, 13_000.0);
        assert_eq!(cost.cvar_alpha, 0.95);
    }

    #[test]
    fn simulation_metadata_solve_stats_round_trip() {
        let original = SimulationMetadata {
            solve_stats: MetadataSimulationSolveStats {
                total_lp_solves: Some(50_000),
                first_try: Some(48_000),
                retried: Some(1_900),
                failed: Some(100),
                solve_seconds: Some(321.0),
                parallelism: Some(8),
            },
            ..make_simulation_metadata()
        };

        let dir = tempdir().unwrap();
        let path = dir.path().join("metadata.json");
        write_simulation_metadata(&path, &original).expect("write must succeed");
        let decoded = read_simulation_metadata(&path).expect("read must succeed");

        assert_eq!(decoded.solve_stats.total_lp_solves, Some(50_000));
        assert_eq!(decoded.solve_stats.first_try, Some(48_000));
        assert_eq!(decoded.solve_stats.retried, Some(1_900));
        assert_eq!(decoded.solve_stats.failed, Some(100));
        assert_eq!(decoded.solve_stats.solve_seconds, Some(321.0));
        assert_eq!(decoded.solve_stats.parallelism, Some(8));
    }

    #[test]
    fn simulation_metadata_back_compat_without_cost_or_solve_stats() {
        let legacy = r#"{
            "cobre_version": "0.0.0",
            "hostname": "legacy-host",
            "solver": "highs",
            "started_at": "2026-01-17T13:00:00Z",
            "completed_at": "2026-01-17T13:15:00Z",
            "duration_seconds": 900.0,
            "status": "complete",
            "scenarios": {
                "total": 100,
                "completed": 100,
                "failed": 0
            },
            "distribution": {
                "backend": "local",
                "world_size": 1,
                "ranks_participated": 1,
                "num_nodes": 1,
                "threads_per_rank": 1
            }
        }"#;

        let decoded: SimulationMetadata = serde_json::from_str(legacy).unwrap();
        assert!(decoded.cost.is_none());
        assert_eq!(decoded.solve_stats.total_lp_solves, None);
        assert_eq!(decoded.solve_stats.parallelism, None);
    }

    #[test]
    fn distribution_info_hosts_round_trip() {
        let original = DistributionInfo {
            hosts: vec![HostLayout {
                hostname: "node01".to_string(),
                ranks: vec![0, 1, 2, 3],
            }],
            ..make_distribution_info()
        };

        let json = serde_json::to_string(&original).unwrap();
        assert!(
            json.contains(r#""hosts":[{"hostname":"node01","ranks":[0,1,2,3]}]"#),
            "serialized JSON must contain the hosts array, got: {json}"
        );

        let decoded: DistributionInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.hosts.len(), 1);
        assert_eq!(decoded.hosts[0].hostname, "node01");
        assert_eq!(decoded.hosts[0].ranks, vec![0, 1, 2, 3]);
    }

    #[test]
    fn distribution_info_empty_hosts_serialize_as_array() {
        let info = make_distribution_info();
        let json = serde_json::to_string(&info).unwrap();
        assert!(
            json.contains(r#""hosts":[]"#),
            "empty hosts must serialize as [], got: {json}"
        );
    }

    #[test]
    fn distribution_info_rank_affinity_round_trip() {
        let mut original = make_distribution_info();
        original.rank_affinity.push(RankAffinity {
            rank: 0,
            policy: "core".to_string(),
            online_processing_units: Some(96),
            visible_processing_units: Some(48),
            physical_cores: Some(48),
            numa_nodes: Some(4),
            visible_cpus: vec![0, 2, 4, 6],
            memory_policy: Some("bind".to_string()),
            memory_policy_nodes: vec![0, 1],
            allowed_memory_nodes: vec![0, 1, 2, 3],
            memory_discovery_error: None,
            worker_cpus: vec![0, 2],
            discovery_error: None,
        });

        let json = serde_json::to_string(&original).unwrap();
        let decoded: DistributionInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.rank_affinity.len(), 1);
        let affinity = &decoded.rank_affinity[0];
        assert_eq!(affinity.policy, "core");
        assert_eq!(affinity.visible_processing_units, Some(48));
        assert_eq!(affinity.memory_policy.as_deref(), Some("bind"));
        assert_eq!(affinity.memory_policy_nodes, vec![0, 1]);
        assert_eq!(affinity.allowed_memory_nodes, vec![0, 1, 2, 3]);
        assert_eq!(affinity.worker_cpus, vec![0, 2]);
    }

    #[test]
    fn distribution_info_back_compat_without_hosts() {
        let legacy = r#"{
            "backend": "local",
            "world_size": 1,
            "ranks_participated": 1,
            "num_nodes": 1,
            "threads_per_rank": 1
        }"#;

        let decoded: DistributionInfo = serde_json::from_str(legacy).unwrap();
        assert!(
            decoded.hosts.is_empty(),
            "missing hosts key must deserialize to an empty vector"
        );
        assert!(decoded.rank_affinity.is_empty());
    }

    #[test]
    fn training_metadata_bounds_round_trip() {
        let original = TrainingMetadata {
            bounds: MetadataBounds {
                final_lower_bound: 48_500.0,
                final_upper_bound: Some(49_000.0),
                final_upper_bound_std: Some(250.0),
            },
            ..make_training_metadata()
        };

        let json = serde_json::to_string(&original).unwrap();
        assert!(
            json.contains(r#""final_lower_bound":48500.0"#),
            "serialized JSON must contain the lower bound, got: {json}"
        );
        assert!(
            json.contains(r#""final_upper_bound":49000.0"#),
            "serialized JSON must contain the upper bound, got: {json}"
        );

        let decoded: TrainingMetadata = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.bounds.final_lower_bound, 48_500.0);
        assert_eq!(decoded.bounds.final_upper_bound, Some(49_000.0));
        assert_eq!(decoded.bounds.final_upper_bound_std, Some(250.0));
    }

    #[test]
    fn training_metadata_solve_stats_round_trip() {
        let original = TrainingMetadata {
            solve_stats: MetadataTrainingSolveStats {
                total_lp_solves: Some(84_000),
                first_try: Some(80_000),
                retried: Some(3_800),
                failed: Some(200),
                forward_solve_seconds: Some(123.5),
                backward_solve_seconds: Some(456.75),
                parallelism: Some(8),
            },
            ..make_training_metadata()
        };

        let dir = tempdir().unwrap();
        let path = dir.path().join("metadata.json");
        write_training_metadata(&path, &original).expect("write must succeed");
        let decoded = read_training_metadata(&path).expect("read must succeed");

        assert_eq!(decoded.solve_stats.total_lp_solves, Some(84_000));
        assert_eq!(decoded.solve_stats.first_try, Some(80_000));
        assert_eq!(decoded.solve_stats.retried, Some(3_800));
        assert_eq!(decoded.solve_stats.failed, Some(200));
        assert_eq!(decoded.solve_stats.forward_solve_seconds, Some(123.5));
        assert_eq!(decoded.solve_stats.backward_solve_seconds, Some(456.75));
        assert_eq!(decoded.solve_stats.parallelism, Some(8));
    }

    #[test]
    fn training_metadata_back_compat_without_bounds_or_solve_stats() {
        let legacy = r#"{
            "cobre_version": "0.0.0",
            "hostname": "legacy-host",
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
            "iterations": {
                "completed": 100,
                "converged_at": 95
            },
            "convergence": {
                "achieved": true,
                "final_gap_percent": 0.45,
                "termination_reason": "bound_stalling"
            },
            "row_pool": {
                "total_generated": 1250000,
                "total_active": 980000,
                "peak_active": 1100000
            },
            "distribution": {
                "backend": "local",
                "world_size": 1,
                "ranks_participated": 1,
                "num_nodes": 1,
                "threads_per_rank": 1
            }
        }"#;

        let decoded: TrainingMetadata = serde_json::from_str(legacy).unwrap();
        assert_eq!(decoded.bounds.final_lower_bound, 0.0);
        assert_eq!(decoded.bounds.final_upper_bound, None);
        assert_eq!(decoded.bounds.final_upper_bound_std, None);
        assert_eq!(decoded.solve_stats.total_lp_solves, None);
        assert_eq!(decoded.solve_stats.parallelism, None);
    }

    #[test]
    fn setup_timings_round_trips() {
        let original = TrainingMetadata {
            setup: Some(SetupTimings {
                load_seconds: 1.5,
                stochastic_fit_seconds: 2.25,
                production_fit_seconds: 3.75,
                evaporation_fit_seconds: 0.5,
                broadcast_seconds: 0.125,
            }),
            ..make_training_metadata()
        };

        let json = serde_json::to_string(&original).unwrap();
        let decoded: TrainingMetadata = serde_json::from_str(&json).unwrap();

        let setup = decoded
            .setup
            .expect("setup must be present after round-trip");
        assert_eq!(setup.load_seconds, 1.5);
        assert_eq!(setup.stochastic_fit_seconds, 2.25);
        assert_eq!(setup.production_fit_seconds, 3.75);
        assert_eq!(setup.evaporation_fit_seconds, 0.5);
        assert_eq!(setup.broadcast_seconds, 0.125);
    }

    #[test]
    fn training_metadata_without_setup_reads_as_none() {
        let without_setup = r#"{
            "cobre_version": "0.0.0",
            "hostname": "legacy-host",
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
            "iterations": {
                "completed": 100,
                "converged_at": 95
            },
            "convergence": {
                "achieved": true,
                "final_gap_percent": 0.45,
                "termination_reason": "bound_stalling"
            },
            "row_pool": {
                "total_generated": 1250000,
                "total_active": 980000,
                "peak_active": 1100000
            },
            "distribution": {
                "backend": "local",
                "world_size": 1,
                "ranks_participated": 1,
                "num_nodes": 1,
                "threads_per_rank": 1
            }
        }"#;

        let decoded: TrainingMetadata = serde_json::from_str(without_setup).unwrap();
        assert!(decoded.setup.is_none());
    }

    // ── DeviationSummary ───────────────────────────────────────────────────────

    #[test]
    fn deviation_summary_with_worst_entry_round_trips() {
        let original = TrainingMetadata {
            production_fit_deviation: Some(DeviationSummary {
                n_entries: 3,
                mean_abs: 4.1,
                max_abs: 31.7,
                worst_relative: 0.062,
                worst_entry: Some(DeviationWorstEntry {
                    entity_id: 12,
                    stage_id: 4,
                    relative: 0.062,
                    mean_abs: 8.0,
                    max_abs: 31.7,
                }),
            }),
            ..make_training_metadata()
        };

        let json = serde_json::to_string(&original).unwrap();
        assert!(
            json.contains(r#""production_fit_deviation""#),
            "serialized JSON must contain the deviation section, got: {json}"
        );

        let decoded: TrainingMetadata = serde_json::from_str(&json).unwrap();
        let summary = decoded
            .production_fit_deviation
            .expect("deviation summary must survive round-trip");
        assert_eq!(summary.n_entries, 3);
        assert_eq!(summary.mean_abs, 4.1);
        assert_eq!(summary.max_abs, 31.7);
        assert_eq!(summary.worst_relative, 0.062);
        let worst = summary.worst_entry.expect("worst entry must be present");
        assert_eq!(worst.entity_id, 12);
        assert_eq!(worst.stage_id, 4);
        assert_eq!(worst.relative, 0.062);
        assert_eq!(worst.mean_abs, 8.0);
        assert_eq!(worst.max_abs, 31.7);
    }

    #[test]
    fn training_metadata_skips_deviation_when_none() {
        let metadata = TrainingMetadata {
            production_fit_deviation: None,
            ..make_training_metadata()
        };

        let json = serde_json::to_string(&metadata).unwrap();
        assert!(
            !json.contains("production_fit_deviation"),
            "the key must be omitted when None, got: {json}"
        );
    }

    #[test]
    fn training_metadata_without_deviation_reads_as_none() {
        let without_deviation = r#"{
            "cobre_version": "0.0.0",
            "hostname": "legacy-host",
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
            "iterations": {
                "completed": 100,
                "converged_at": 95
            },
            "convergence": {
                "achieved": true,
                "final_gap_percent": 0.45,
                "termination_reason": "bound_stalling"
            },
            "row_pool": {
                "total_generated": 1250000,
                "total_active": 980000,
                "peak_active": 1100000
            },
            "distribution": {
                "backend": "local",
                "world_size": 1,
                "ranks_participated": 1,
                "num_nodes": 1,
                "threads_per_rank": 1
            }
        }"#;

        let decoded: TrainingMetadata = serde_json::from_str(without_deviation).unwrap();
        assert!(decoded.production_fit_deviation.is_none());
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
        assert_eq!(decoded.row_pool.total_generated, 1_250_000);
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
        assert!(ts.ends_with('Z'), "timestamp must end with Z: {ts}");
        assert!(ts.contains('T'), "timestamp must contain T separator: {ts}");
    }
}
