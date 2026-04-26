# Output Infrastructure

## Purpose

This spec defines the infrastructure layer for Cobre output: metadata files for reproducibility, `_SUCCESS` marker files for crash recovery, MPI-native Hive partitioning for parallel writes, and validation/integrity checks.

For output Parquet schemas (simulation and training column definitions), see [Output Schemas](output-schemas.md).
For output configuration options within `config.json`, see [Configuration Reference](../configuration/configuration-reference.md).

## 1. Completion Marker and Metadata Files

Run completion is tracked by two separate mechanisms: a `metadata.json` file containing run metadata (timing, configuration snapshot, problem dimensions), and a `_SUCCESS` marker file whose presence indicates successful completion. The old `_manifest.json` pattern with its `status` field has been removed.

### 1.1 Completion Marker (`_SUCCESS`)

Each output phase writes a zero-byte `_SUCCESS` marker file upon successful completion:

- `training/_SUCCESS` — written by rank 0 after all training outputs (Parquet files, metadata, policy checkpoint) are flushed.
- `simulation/_SUCCESS` — written by rank 0 after the simulation manifest/metadata and all simulation Parquet partitions are confirmed complete.

**Crash Recovery Protocol:**

1. On startup, check whether `_SUCCESS` exists in the relevant output directory.
2. If absent, the previous run did not complete successfully. Examine existing partition directories to identify completed work.
3. Resume from incomplete scenarios/iterations.
4. Write `_SUCCESS` only after all output files are confirmed flushed.

The `_SUCCESS` marker is the last file written in each output phase. Its presence is both necessary and sufficient to consider the output directory complete.

### 1.2 Simulation Metadata (`simulation/metadata.json`)

```json
{
  "$schema": "https://cobre.dev/schemas/v2/simulation_metadata.schema.json",
  "version": "2.0.0",
  "started_at": "2026-01-17T10:00:00Z",
  "completed_at": "2026-01-17T10:15:00Z",
  "scenarios": {
    "total": 2000,
    "completed": 2000
  },
  "partitions_written": ["scenario_id=0/", "scenario_id=1/", "..."],
  "checksum": {
    "algorithm": "xxhash64",
    "value": "a1b2c3d4e5f6"
  },
  "mpi_info": {
    "world_size": 128,
    "ranks_participated": 128
  }
}
```

| Field                         | Type   | Description                                |
| ----------------------------- | ------ | ------------------------------------------ |
| `started_at`                  | string | ISO 8601 timestamp                         |
| `completed_at`                | string | ISO 8601 timestamp                         |
| `scenarios.total`             | i32    | Total scenarios to simulate                |
| `scenarios.completed`         | i32    | Successfully completed scenarios           |
| `partitions_written`          | array  | List of Hive partition directories written |
| `checksum`                    | object | Integrity checksum for validation          |
| `mpi_info.world_size`         | i32    | Number of MPI ranks                        |
| `mpi_info.ranks_participated` | i32    | Ranks that wrote data                      |

### 1.3 Training Metadata (`training/metadata.json`)

Training metadata captures convergence outcome, iteration counts, and cut statistics. The detailed run metadata (timing, configuration snapshot, problem dimensions) is documented in SS2.

```json
{
  "$schema": "https://cobre.dev/schemas/v2/training_metadata.schema.json",
  "version": "2.0.0",
  "started_at": "2026-01-17T08:00:00Z",
  "completed_at": "2026-01-17T12:30:00Z",
  "iterations": {
    "max_iterations": 100,
    "completed": 100,
    "converged_at": 87
  },
  "convergence": {
    "achieved": true,
    "final_gap_percent": 0.45,
    "termination_reason": "simulation"
  },
  "cuts": {
    "total_generated": 1250000,
    "total_active": 980000,
    "peak_active": 1100000
  },
  "checksum": {
    "algorithm": "xxhash64",
    "policy_value": "f1e2d3c4b5a6",
    "convergence_value": "1a2b3c4d5e6f"
  },
  "mpi_info": {
    "world_size": 128,
    "forward_passes_per_iteration": 8
  }
}
```

| Field                            | Type   | Description                                                                                                                                                                              |
| -------------------------------- | ------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `iterations.max_iterations`      | i32    | Maximum iterations from `iteration_limit` stopping rule                                                                                                                                  |
| `iterations.completed`           | i32    | Iterations actually run                                                                                                                                                                  |
| `iterations.converged_at`        | i32    | Iteration where convergence-oriented rule triggered (null if terminated by safety limit)                                                                                                 |
| `convergence.achieved`           | bool   | Whether a convergence-oriented rule (`bound_stalling` or `simulation`) triggered, as opposed to a safety limit (`iteration_limit`, `time_limit`)                                         |
| `convergence.final_gap_percent`  | f64    | Final optimality gap (null if upper bound evaluation is disabled). Under CVaR risk measures, this gap is not a valid optimality bound; see [Risk Measures](../math/risk-measures.md) §10 |
| `convergence.termination_reason` | string | One of: `"iteration_limit"`, `"time_limit"`, `"bound_stalling"`, `"simulation"`. See [Stopping Rules](../math/stopping-rules.md)                                                         |
| `cuts.total_generated`           | i64    | Total cuts generated during training                                                                                                                                                     |
| `cuts.total_active`              | i64    | Active cuts at termination                                                                                                                                                               |
| `cuts.peak_active`               | i64    | Peak active cuts during training                                                                                                                                                         |

### 1.4 CLI Report Access

Metadata files are accessible via the `report` subcommand, which reads them from disk and returns structured JSON. This enables agents and scripts to inspect training status, convergence outcome, and simulation progress without parsing the file contents directly.

```bash
# Query training metadata
cobre report /path/to/output --output-format json --section convergence

# Query simulation metadata
cobre report /path/to/output --output-format json --section simulation
```

The `report` subcommand is a read-only operation that does not require MPI. It reads the metadata files documented in SS1.2, SS1.3, and SS2, wraps them in the CLI response envelope (see [CLI and Lifecycle §8](../architecture/cli-and-lifecycle.md) and [Structured Output §4](../interfaces/structured-output.md)), and emits the result to stdout. The MCP tool `cobre/query-convergence` performs the same operation via the MCP protocol (see [MCP Server](../interfaces/mcp-server.md)).

## 2. Metadata File (`training/metadata.json`)

Comprehensive metadata for reproducibility, audit trails, and debugging.

```json
{
  "$schema": "https://cobre.dev/schemas/v2/training_metadata.schema.json",
  "version": "2.0.0",
  "run_info": {
    "run_id": "uuid-v4-here",
    "started_at": "2026-01-17T08:00:00Z",
    "completed_at": "2026-01-17T12:30:00Z",
    "duration_seconds": 16200,
    "cobre_version": "2.0.0",
    "solver": "highs",
    "solver_version": "1.7.2",
    "hostname": "compute-node-001",
    "user": "scheduler"
  },
  "configuration_snapshot": {
    "seed": 42,
    "forward_passes": 192,
    "stopping_rules": [
      { "type": "iteration_limit", "limit": 100 },
      {
        "type": "simulation",
        "replications": 100,
        "period": 20,
        "bound_window": 5,
        "distance_tol": 0.01,
        "bound_tol": 0.0001
      }
    ],
    "stopping_mode": "any",
    "cut_selection": {
      "enabled": true,
      "method": "level1"
    },
    "upper_bound_evaluation": {
      "enabled": true,
      "initial_iteration": 10,
      "interval_iterations": 5
    },
    "policy_mode": "fresh"
  },
  "problem_dimensions": {
    "num_stages": 12,
    "num_scenarios": 2000,
    "num_openings": 50,
    "num_plants": 360,
    "num_buses": 5
  }
}
```

**Notes on `configuration_snapshot`:**

- This is an informational record of the training configuration, not a normative schema. The canonical config schema is defined in [Configuration Reference](../configuration/configuration-reference.md).
- `stopping_rules` is recorded verbatim from `config.json` so that the termination behavior can be reconstructed from the output alone.
- `cut_selection.method` uses the values from [Cut Management §9](../math/cut-management.md): `"level1"`, `"lml1"`, or `"domination"`.
- `upper_bound_evaluation` mirrors the config section from [Input Directory Structure §2](input-directory-structure.md). Vertex-based (SIDP) upper bounds are enabled when the `upper_bound_evaluation` section is present with `enabled: true`; see [Upper Bound Evaluation](../math/upper-bound-evaluation.md).

**Notes on `problem_dimensions`:**

- The 5 fields (`num_stages`, `num_scenarios`, `num_openings`, `num_plants`, `num_buses`) reflect the actual code. `num_plants` is the combined count of hydro and thermal plants.
- Additional dimension fields (e.g., `num_blocks_per_stage`, `num_lines`, `state_dimension`, `lp_dimensions`) are planned but not yet implemented in the code.

## 3. MPI Direct Hive Partitioning

Each MPI rank writes directly to Hive partition directories without coordination. For Hive partitioning design principles, see [Output Schemas](output-schemas.md) §2.1.

### 3.1 Directory Layout

```
simulation/
├── costs/
│   ├── scenario_id=0/data.parquet      # Written by rank 0
│   ├── scenario_id=1/data.parquet      # Written by rank 0
│   ├── scenario_id=2/data.parquet      # Written by rank 1
│   └── ...
├── hydros/
│   ├── scenario_id=0/data.parquet
│   └── ...
├── metadata.json                        # Written by rank 0 only
└── _SUCCESS                             # Written by rank 0 after all partitions confirmed
```

### 3.2 Write Semantics

**Scenario assignment:** Round-robin — `rank = scenario_id % world_size`. Each rank writes only its assigned scenarios.

**Write protocol:**

1. Each rank writes its assigned partitions independently (embarrassingly parallel — no inter-rank coordination during writes).
2. All ranks synchronize at a barrier after writes complete.
3. Rank 0 writes the metadata file and `_SUCCESS` marker after the barrier.

**Atomic write pattern:** Each file is written to a temporary path (`data.parquet.tmp`), flushed to disk, then atomically renamed to `data.parquet`. This prevents partial files from appearing as valid output.

> **Note — Intra-rank thread parallelism (pending HPC specs):** The write protocol above describes rank-level granularity. However, Cobre uses hybrid MPI+OpenMP parallelism where multiple threads within each rank may independently process scenarios. If all thread-owned scenarios funnel through a single rank-level writer, this becomes a serialization bottleneck at scale. The actual write responsibility assignment (per-rank vs per-thread) and synchronization strategy will be defined in the HPC work distribution specs. The invariants that must be preserved regardless of the final design are: (1) each partition is written by exactly one writer, (2) writes use the atomic temp-file-then-rename pattern, and (3) the manifest is written only after all partitions are confirmed complete.

### 3.3 Failure Handling

| Failure Type         | Detection                 | Recovery                            |
| -------------------- | ------------------------- | ----------------------------------- |
| Rank crash mid-write | Missing `_SUCCESS` marker | Re-run failed scenarios only        |
| Partial file write   | Parquet read failure      | Delete and re-write partition       |
| Metadata corruption  | JSON parse error          | Rebuild from partition listing      |
| Disk full            | Write error               | Alert, do not corrupt existing data |

## 4. Output Size Estimates

Reference output sizes for production-scale SDDP runs. For problem dimension profiles (Small through Extra Large), see [Production Scale Reference](../overview/production-scale-reference.md).

| Output                         | Small  | Medium | Large   | Extra Large |
| ------------------------------ | ------ | ------ | ------- | ----------- |
| `simulation/costs/`            | 50 MB  | 800 MB | 4 GB    | 20 GB       |
| `simulation/hydros/`           | 200 MB | 5 GB   | 30 GB   | 150 GB      |
| `simulation/thermals/`         | 150 MB | 4 GB   | 25 GB   | 120 GB      |
| `training/convergence.parquet` | 10 KB  | 50 KB  | 100 KB  | 250 KB      |
| `training/timing/`             | 1 MB   | 15 MB  | 120 MB  | 1.2 GB      |
| `policy/` (cuts)               | 500 MB | 8 GB   | 40 GB   | 200 GB      |
| **Total**                      | ~1 GB  | ~20 GB | ~100 GB | ~500 GB     |

**Storage recommendations:**

- Use SSD/NVMe for training (frequent random writes)
- Network filesystem acceptable for simulation (sequential writes)
- Consider parallel filesystem (Lustre, GPFS) for >100 GB outputs
- Enable compression for network transfers

### 4.1 I/O Bandwidth Requirements

| Scale       | Write Throughput | Duration | Bottleneck  |
| ----------- | ---------------- | -------- | ----------- |
| Small       | 50 MB/s          | 20s      | None        |
| Medium      | 200 MB/s         | 100s     | Network     |
| Large       | 500 MB/s         | 200s     | Filesystem  |
| Extra Large | 1+ GB/s          | 500s     | Parallel FS |

## 5. Validation and Integrity

### 5.1 Schema Validation

Each output entity must conform to the Parquet schema defined in [Output Schemas](output-schemas.md). Validation verifies column names, types, and nullability against the schema definitions.

### 5.2 Data Integrity Checks

| Check                  | Method                       | Frequency |
| ---------------------- | ---------------------------- | --------- |
| Parquet file integrity | Footer checksum              | On read   |
| Partition completeness | `_SUCCESS` marker + metadata | Post-run  |
| Row count consistency  | Cross-entity validation      | Post-run  |
| Value range validation | Min/max from bounds.parquet  | Optional  |

**Cross-entity validation:** For each scenario, the number of rows in every entity output must be consistent with the stage and block counts for that scenario. For example, `costs/` has one row per (stage, block), while `hydros/` has one row per (stage, block, hydro). Missing or extra rows indicate a write failure.

### 5.3 Reproducibility Verification

Reproducibility can be verified by comparing output artifacts across runs:

- Given the same inputs, configuration, and random seed, the system must produce identical policy and convergence outputs.
- Two runs can be compared by computing checksums over policy files (`policy/cuts/stage_*.bin`) and `training/convergence.parquet`.
- If input data differs between runs, the policy outputs are not directly comparable.

## 6. Output Writer API

This section defines the Rust types and function signatures for writing all Cobre output: simulation Parquet files, training Parquet files, manifest files, metadata files, dictionary files, and policy checkpoints. These types live in **cobre-io** and are consumed by **cobre-sddp** (training loop and simulation phase) and **cobre-cli** (orchestration layer).

The API follows the same design pattern as the input loading API ([Input Loading Pipeline SS8.1](../architecture/input-loading-pipeline.md)): a top-level anchoring function, concrete writer types (not traits), and a dedicated error enum.

**Design decisions:**

1. **Separate concrete writers, not a single trait.** The four output chains (simulation Parquet, training Parquet, manifests/metadata, policy checkpoint) have different formats, lifecycles, and thread-safety requirements. A unified `OutputWriter` trait would impose artificial uniformity.
2. **Parquet library: `arrow-rs`.** The `arrow-rs` ecosystem (`arrow`, `parquet` crates) is the Rust standard for Apache Arrow and Parquet I/O, with active maintenance and broad ecosystem support.
3. **Synchronous API.** Writer methods are blocking. Async decoupling is provided at the architecture level by the bounded channel between simulation threads and the background I/O thread ([Simulation Architecture SS6.1](../architecture/simulation-architecture.md)).
4. **No column definitions here.** Parquet column schemas are defined in [Output Schemas SS5--6](output-schemas.md). The writers reference those schemas but do not duplicate them.
5. **serde derives.** Manifest structs (§1.1, §1.2) and the metadata struct (§2) derive `serde::Serialize` for JSON serialization. Parquet writers use Arrow `RecordBatch` arrays directly and do not require serde on per-row structs. FlatBuffers uses generated code from the `.fbs` schema ([Binary Formats SS3.1](binary-formats.md)).

### 6.1 `write_results` Anchoring Function

`write_results` is the top-level entry point from `cobre-cli` (rank 0) into `cobre-io` for writing all output artifacts after training and optional simulation complete. It orchestrates the individual writers defined in §6.2--§6.7.

**Function signature:**

```rust
/// Write all output artifacts for a completed Cobre run.
///
/// This is the primary entry point from cobre-cli (rank 0) into cobre-io
/// for output writing. It orchestrates writing of training results,
/// simulation results (when present), manifests, metadata, and
/// dictionary files.
///
/// # Parameters
///
/// - `output_dir` -- Root output directory. Training outputs are written
///   to `output_dir/training/` and simulation outputs to
///   `output_dir/simulation/`. The directory is created if it does not
///   exist.
///
/// - `training_output` -- Training results: convergence log, per-iteration
///   timing, per-rank timing. Always present (training always runs).
///
/// - `simulation_output` -- Simulation results. `None` when simulation is
///   disabled (`simulation.enabled = false` in config). When `Some`, the
///   simulation Parquet files have already been written by the streaming
///   I/O thread (§6.2); this function writes only the simulation manifest.
///
/// - `system` -- Shared reference to the loaded system. Used for
///   dictionary generation (entity names, IDs, bounds).
///
/// - `config` -- Run configuration. Used for metadata snapshot and
///   Parquet writer configuration (compression, row group size).
///
/// # Errors
///
/// Returns `OutputError` if any output file cannot be written.
/// Partial writes may leave some output files on disk; the manifest
/// `status` field will remain `"running"` (not `"complete"`),
/// enabling crash recovery on re-run (§1.1).
///
/// # Execution context
///
/// Called on rank 0 only, after the MPI barrier that confirms all
/// ranks have completed their partition writes. See
/// [Output Infrastructure §3.2](output-infrastructure.md) for the
/// write protocol.
pub fn write_results(
    output_dir: &Path,
    training_output: &TrainingOutput,
    simulation_output: Option<&SimulationOutput>,
    system: &System,
    config: &Config,
    ctx: &OutputContext,
) -> Result<(), OutputError>
```

`write_results` performs the following steps, in order:

1. Create `output_dir/training/` and `output_dir/simulation/` directories if they do not exist.
2. Write dictionary files via `write_dictionaries` (§6.5).
3. Write training Parquet files via `TrainingParquetWriter` (§6.3).
4. Write training metadata via `write_training_metadata` (§6.5).
5. If `simulation_output` is `Some`, write simulation metadata via `write_simulation_metadata` (§6.5).
6. Write `_SUCCESS` marker files.

`write_results` does NOT write simulation Parquet files. Those are written by the streaming I/O thread during simulation execution, using the `SimulationParquetWriter` (§6.2). By the time `write_results` is called, the simulation Parquet files are already on disk. `write_results` writes only the simulation manifest (which requires the final scenario counts and checksums).

**Input types:**

`TrainingOutput` and `SimulationOutput` are aggregate types defined in **cobre-sddp** that carry all data needed for output writing. Their exact field definitions are determined by the training and simulation return types ([Training Loop SS2.1](../architecture/training-loop.md), [Simulation Architecture SS3.4.4](../architecture/simulation-architecture.md)). The key fields consumed by `write_results` are:

| Type               | Key Fields                                                                     |
| ------------------ | ------------------------------------------------------------------------------ |
| `TrainingOutput`   | convergence log records, per-iteration timing, per-rank timing, cut statistics |
| `SimulationOutput` | scenario count, completion status, per-partition checksums, cost statistics    |

### 6.2 `SimulationParquetWriter`

`SimulationParquetWriter` is the concrete writer for simulation Parquet files. It is used by the background I/O thread that receives `SimulationScenarioResult` values through the bounded channel ([Simulation Architecture SS6.1](../architecture/simulation-architecture.md)).

```rust
/// Writer for simulation Parquet files. Receives per-scenario results
/// and writes them to Hive-partitioned Parquet files under
/// `output_dir/simulation/`.
///
/// # Thread safety
///
/// `SimulationParquetWriter` implements `Send` because it is created
/// on the main thread and moved to the dedicated background I/O
/// thread. It does NOT implement `Sync` -- only one thread (the I/O
/// thread) accesses it at a time.
///
/// # Lifecycle
///
/// 1. Created via `new()` before the simulation phase begins.
/// 2. `write_scenario()` called once per completed scenario, in
///    arrival order (not necessarily scenario ID order).
/// 3. `finalize()` called after the channel is closed (all senders
///    dropped), flushing any buffered data and computing checksums.
pub struct SimulationParquetWriter { /* ... */ }

impl SimulationParquetWriter {
    /// Create a new simulation Parquet writer.
    ///
    /// # Parameters
    ///
    /// - `output_dir` -- Root output directory. Parquet files are
    ///   written under `output_dir/simulation/{entity}/scenario_id=XXXX/`.
    ///
    /// - `system` -- Shared reference to the system for entity
    ///   metadata (entity counts, block counts per stage, line loss
    ///   factors, block durations). Used to compute derived columns
    ///   (energy = power x duration, losses, net flows) during
    ///   Parquet writing. See [Simulation Architecture SS3.4](../architecture/simulation-architecture.md)
    ///   for the list of excluded (derived) columns.
    ///
    /// - `config` -- Parquet writer configuration: compression codec
    ///   (Zstd level 3), row group size (~100,000 rows), dictionary
    ///   encoding for categorical columns. See [Binary Formats SS5](binary-formats.md).
    ///
    /// # Errors
    ///
    /// Returns `OutputError::IoError` if the output directory cannot
    /// be created.
    pub fn new(
        output_dir: &Path,
        system: &System,
        config: &ParquetWriterConfig,
    ) -> Result<Self, OutputError>

    /// Write one scenario's simulation results to Parquet files.
    ///
    /// Each call writes one Hive partition per entity type:
    /// `{entity}/scenario_id={id:04d}/data.parquet`. Files are written
    /// atomically (write to `.tmp`, then rename) per the protocol in §3.2.
    ///
    /// The writer converts the nested per-entity-type layout of
    /// `SimulationScenarioResult` into the columnar Arrow `RecordBatch`
    /// format, computing derived columns (MWh energy, net flow, losses)
    /// from system metadata. Column schemas are defined in
    /// [Output Schemas SS5.1--5.11](output-schemas.md).
    ///
    /// # Parameters
    ///
    /// - `result` -- Complete simulation result for one scenario, as
    ///   produced by the simulation forward pass. See
    ///   [Simulation Architecture SS3.4.3](../architecture/simulation-architecture.md).
    ///
    /// # Errors
    ///
    /// Returns `OutputError::IoError` on disk write failure or
    /// `OutputError::SerializationError` on Arrow/Parquet encoding failure.
    pub fn write_scenario(
        &mut self,
        result: SimulationScenarioResult,
    ) -> Result<(), OutputError>

    /// Finalize the writer: flush any buffered data, compute checksums
    /// over all written partitions, and return the manifest data.
    ///
    /// This is a consuming method -- the writer cannot be used after
    /// finalization. The returned `SimulationManifest` contains the
    /// scenario counts, partition list, and checksums needed by
    /// `write_simulation_metadata` (§6.5).
    ///
    /// # Errors
    ///
    /// Returns `OutputError::IoError` if final flush or checksum
    /// computation fails.
    pub fn finalize(self) -> Result<SimulationManifest, OutputError>
}
```

**Concurrency model:** The `SimulationParquetWriter` runs on a single dedicated I/O thread per MPI rank. Multiple simulation threads send `SimulationScenarioResult` values through the bounded channel; the I/O thread receives and writes them sequentially. There is no lock contention on the writer itself. Backpressure from the bounded channel (capacity configured via `simulation.io_channel_capacity`, default 64) throttles simulation threads when I/O falls behind.

### 6.3 `TrainingParquetWriter`

`TrainingParquetWriter` writes the three training Parquet files: convergence log, iteration timing, and MPI rank timing.

```rust
/// Writer for training Parquet files. Writes convergence log,
/// iteration timing, and MPI rank timing under
/// `output_dir/training/`.
///
/// # Thread safety
///
/// `TrainingParquetWriter` runs on the main thread (rank 0) after
/// training completes. It does not need `Send` or `Sync`.
///
/// # Lifecycle
///
/// 1. Created via `new()` after training completes.
/// 2. `write_iteration()` called once per training iteration.
/// 3. `write_rank_timing()` called once with all rank timing records.
/// 4. `finalize()` called to flush and close all files.
pub struct TrainingParquetWriter { /* ... */ }

impl TrainingParquetWriter {
    /// Create a new training Parquet writer.
    ///
    /// # Parameters
    ///
    /// - `output_dir` -- Root output directory. Files are written to
    ///   `output_dir/training/convergence.parquet`,
    ///   `output_dir/training/timing/iterations.parquet`, and
    ///   `output_dir/training/timing/mpi_ranks.parquet`.
    ///
    /// # Errors
    ///
    /// Returns `OutputError::IoError` if the output directories
    /// cannot be created.
    pub fn new(output_dir: &Path) -> Result<Self, OutputError>

    /// Write one iteration's convergence and timing data.
    ///
    /// Appends one row to `convergence.parquet` and one row to
    /// `timing/iterations.parquet`. Column schemas are defined in
    /// [Output Schemas SS6.1](output-schemas.md) and
    /// [Output Schemas SS6.2](output-schemas.md).
    ///
    /// # Parameters
    ///
    /// - `record` -- Convergence and timing data for one iteration.
    ///   Contains all fields from the convergence log schema (SS6.1)
    ///   and iteration timing schema (SS6.2).
    ///
    /// # Errors
    ///
    /// Returns `OutputError::SerializationError` on Arrow encoding
    /// failure.
    pub fn write_iteration(
        &mut self,
        record: &IterationRecord,
    ) -> Result<(), OutputError>

    /// Write MPI rank timing records for all iterations.
    ///
    /// Writes all rows to `timing/mpi_ranks.parquet`. Column schema
    /// is defined in [Output Schemas SS6.3](output-schemas.md).
    ///
    /// # Parameters
    ///
    /// - `records` -- Rank timing records, one per (iteration, rank)
    ///   pair.
    ///
    /// # Errors
    ///
    /// Returns `OutputError::SerializationError` on Arrow encoding
    /// failure.
    pub fn write_rank_timing(
        &mut self,
        records: &[RankTimingRecord],
    ) -> Result<(), OutputError>

    /// Finalize the writer: flush all buffered data and close files.
    ///
    /// # Errors
    ///
    /// Returns `OutputError::IoError` if final flush fails.
    pub fn finalize(self) -> Result<(), OutputError>
}
```

### 6.4 `OutputError`

`OutputError` is the error type for all output writing operations. It mirrors the structure of `LoadError` ([Input Loading Pipeline SS8.1](../architecture/input-loading-pipeline.md)) with variants ordered by the phase in which they typically occur.

```rust
/// Errors that can occur during output writing.
#[derive(Debug, thiserror::Error)]
pub enum OutputError {
    /// Filesystem write failure (permission denied, disk full, rename
    /// failure during atomic write).
    #[error("I/O error writing {path}: {source}")]
    IoError {
        path: PathBuf,
        source: std::io::Error,
    },

    /// Arrow or Parquet encoding failure (schema mismatch between
    /// constructed RecordBatch and expected schema, unsupported type
    /// conversion).
    #[error("serialization error for {entity}: {message}")]
    SerializationError {
        entity: String,
        message: String,
    },

    /// Parquet schema validation failure (column count mismatch,
    /// unexpected null in non-nullable column, data type mismatch
    /// against the schemas defined in Output Schemas SS5--6).
    #[error("schema error in {file}: column {column}: {message}")]
    SchemaError {
        file: String,
        column: String,
        message: String,
    },

    /// Manifest construction or serialization failure (missing
    /// required field, JSON serialization error, checksum computation
    /// failure).
    #[error("manifest error for {manifest_type}: {message}")]
    ManifestError {
        manifest_type: String,
        message: String,
    },
}
```

| Variant              | Typical Trigger                                               | Example                                                        |
| -------------------- | ------------------------------------------------------------- | -------------------------------------------------------------- |
| `IoError`            | Filesystem operations (create dir, write file, atomic rename) | Disk full writing `hydros/scenario_id=0042/data.parquet`       |
| `SerializationError` | Arrow RecordBatch construction or Parquet row group encoding  | Float-to-int conversion failure in Arrow array builder         |
| `SchemaError`        | Column count or type mismatch during Parquet write            | Expected 24 columns in hydros schema, RecordBatch has 23       |
| `ManifestError`      | Manifest JSON serialization or checksum computation           | xxhash64 checksum computation failed for simulation partitions |

### 6.5 Manifest, Metadata, and Dictionary Writers

These are standalone functions (not methods on a struct) because each writes a single file atomically.

```rust
/// Write the simulation metadata to `output_dir/simulation/metadata.json`.
///
/// The metadata schema is defined in §1.2. The metadata value is
/// produced by the simulation runner after all scenarios complete.
///
/// # Errors
///
/// Returns `OutputError::IoError` on write failure or
/// `OutputError::SerializationError` on JSON serialization failure.
pub fn write_simulation_metadata(
    path: &Path,
    metadata: &SimulationMetadata,
) -> Result<(), OutputError>

/// Write the training metadata to `output_dir/training/metadata.json`.
///
/// The metadata schema is defined in §2. The metadata struct
/// captures the run configuration snapshot, problem dimensions,
/// performance summary, data integrity hashes, and environment
/// information.
///
/// # Errors
///
/// Returns `OutputError::IoError` on write failure or
/// `OutputError::ManifestError` on JSON serialization failure.
pub fn write_training_metadata(
    path: &Path,
    metadata: &TrainingMetadata,
) -> Result<(), OutputError>

/// Write all dictionary files to `output_dir/training/dictionaries/`.
///
/// Produces the following files:
/// - `codes.json` -- categorical code mappings (§3)
/// - `bounds.parquet` -- entity bounds by stage/block (§4.1)
/// - `state_dictionary.json` -- state space definition (§4.2)
/// - `variables.csv` -- variable metadata (§4.3)
/// - `entities.csv` -- entity metadata (§4.4)
///
/// Dictionary schemas are defined in [Output Schemas SS3--4](output-schemas.md).
///
/// # Parameters
///
/// - `path` -- Dictionary directory path
///   (`output_dir/training/dictionaries/`).
/// - `system` -- System reference for entity names, IDs, bus
///   assignments, and bounds.
/// - `config` -- Configuration for stage/block structure and
///   state variable definitions.
///
/// # Errors
///
/// Returns `OutputError::IoError` on write failure or
/// `OutputError::SerializationError` on encoding failure.
pub fn write_dictionaries(
    path: &Path,
    system: &System,
    config: &Config,
) -> Result<(), OutputError>
```

**serde derives for manifest and metadata types:** The `SimulationManifest`, `TrainingManifest`, and `TrainingMetadata` structs derive `serde::Serialize` (and `serde::Deserialize` for manifest types, to support crash recovery reads). These structs map directly to the JSON schemas in §1.1, §1.2, and §2 respectively. The serde field names match the JSON field names using `#[serde(rename_all = "snake_case")]` (which is already the naming convention in the JSON schemas).

### 6.6 Policy Checkpoint Writer

```rust
/// Write a policy checkpoint to the policy directory.
///
/// Serializes the current cut pool, visited states, and solver basis
/// cache to FlatBuffers `.bin` files under `output_dir/policy/`.
/// The FlatBuffers schema is defined in [Binary Formats SS3.1](binary-formats.md).
/// The directory structure follows [Binary Formats SS3.2](binary-formats.md).
///
/// # Parameters
///
/// - `path` -- Policy directory path (`output_dir/policy/`).
/// - `stage_cuts` -- Per-stage cut collections. Each element is
///   serialized to `cuts/stage_{NNN}.bin`.
/// - `stage_bases` -- Per-stage cached solver bases. Each element is
///   serialized to `basis/stage_{NNN}.bin`.
/// - `metadata` -- Policy metadata (version, iteration count, bounds,
///   RNG state). Serialized to `metadata.json` (JSON, not FlatBuffers,
///   for human readability).
///
/// # Errors
///
/// Returns `OutputError::IoError` on write failure or
/// `OutputError::SerializationError` on FlatBuffers encoding failure.
pub fn write_policy_checkpoint(
    path: &Path,
    stage_cuts: &[StageCuts],
    stage_bases: &[StageBasis],
    metadata: &PolicyMetadata,
) -> Result<(), OutputError>
```

### 6.7 API Element Summary

The following table maps the 9 API elements identified in report-013 section 4.4 to their definitions in this section:

| #   | API Element                        | Definition  | Format                      |
| --- | ---------------------------------- | ----------- | --------------------------- |
| 1   | Simulation Parquet writer type     | §6.2        | Arrow RecordBatch + Parquet |
| 2   | Training Parquet writer type       | §6.3        | Arrow RecordBatch + Parquet |
| 3   | Manifest writer function           | §6.5        | JSON via serde              |
| 4   | Metadata writer function           | §6.5        | JSON via serde              |
| 5   | Dictionary writer functions        | §6.5        | JSON + Parquet + CSV        |
| 6   | FlatBuffers serialization function | §6.6        | FlatBuffers generated code  |
| 7   | Output error type (`OutputError`)  | §6.4        | `thiserror` enum            |
| 8   | serde derives on output types      | §6.5 (note) | `serde::Serialize`          |
| 9   | Parquet library selection          | §6 (intro)  | `arrow-rs` ecosystem        |

## Cross-References

- [Output Schemas](output-schemas.md) — Parquet column definitions for all entity types and Hive partitioning design
- [Binary Formats](binary-formats.md) — Policy file format (FlatBuffers `.bin` files) and directory structure
- [Input System Entities](input-system-entities.md) — Entity registries (entity IDs, names)
- [Penalty System](penalty-system.md) — Penalty costs affecting output values
- [Configuration Reference](../configuration/configuration-reference.md) — Full `config.json` reference including output configuration
- [Input Directory Structure](input-directory-structure.md) — `config.json` structure and `upper_bound_evaluation` section
- [Production Scale Reference](../overview/production-scale-reference.md) — Problem dimension profiles
- [Stopping Rules](../math/stopping-rules.md) — Termination criteria definitions
- [Cut Management](../math/cut-management.md) — Cut selection strategy names and parameters
- [Risk Measures](../math/risk-measures.md) — CVaR and lower bound validity
- [Upper Bound Evaluation](../math/upper-bound-evaluation.md) — Simulation-based and SIDP upper bound mechanisms
- [Block Formulations](../math/block-formulations.md) — Block mode definitions (per-stage)
- [Design Principles](../overview/design-principles.md) — Overall design philosophy
- [Input Loading Pipeline](../architecture/input-loading-pipeline.md) — `load_case` / `LoadError` pattern mirrored by `write_results` / `OutputError` (SS8.1)
- [Simulation Architecture](../architecture/simulation-architecture.md) — `SimulationScenarioResult` type (SS3.4.3), bounded channel streaming (SS6.1), `fn simulate()` signature (SS3.4.6)
- [Training Loop](../architecture/training-loop.md) — Training iteration lifecycle and event emission (SS2.1)
