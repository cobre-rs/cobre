# Design: Execution Topology Reporting

**Date**: 2026-04-10
**Scope**: cobre-comm, cobre-io, cobre-cli, cobre-python
**Related**: ferrompi v0.3.0 ([changelog](https://github.com/cobre-rs/ferrompi/blob/v0.3.0/CHANGELOG.md))

---

## Context

ferrompi v0.3.0 adds `Communicator::topology(&mpi) → TopologyInfo`, a collective
that gathers rank-to-host mapping, MPI library version, MPI standard version,
thread level, and (with the `numa` feature) SLURM job metadata across all ranks.

Cobre currently reports almost none of this. The user sees:

- **Banner**: `COBRE v{version}` — no environment info
- **`cobre version`**: `comm: mpi` — no library identity
- **Metadata JSON**: `mpi: { world_size, ranks_participated }` — no topology
- **Hostname**: Read from `/proc/sys/kernel/hostname` fallback, not MPI-native

For HPC users running `mpiexec -n 64 cobre run ...` across multiple nodes, there
is no way to verify at a glance that the job landed on the right nodes with the
right resource allocation. For reproducibility, the metadata JSON doesn't record
which MPI implementation or thread level was used.

### ferrompi v0.3.0 API Surface

```rust
// Collective — all ranks must call
comm.topology(&mpi) -> Result<TopologyInfo>

// Accessors on TopologyInfo
fn library_version() -> &str      // "Open MPI v4.1.6"
fn standard_version() -> &str     // "MPI 4.0"
fn thread_level() -> ThreadLevel  // Funneled
fn size() -> i32                  // 8
fn num_hosts() -> usize           // 2
fn hosts() -> &[HostEntry]        // [{hostname, ranks}]
fn slurm() -> Option<&SlurmInfo>  // {job_id, node_list, cpus_per_task} (numa feature)

// Display impl produces a formatted box report
```

## Design Goals

1. **Model in cobre-comm** — cobre-native types, no ferrompi re-exports
2. **Separate trait** — `TopologyProvider`, orthogonal to `Communicator` (same
   pattern as `SharedMemoryProvider`). Not on the hot path; keeps `Communicator`
   focused on collective operations.
3. **Backend-extensible** — each backend implements `TopologyProvider` with its
   own data sources. TCP and SHM backends can implement it in the future.
4. **Consumer-agnostic** — CLI, Python, MCP, TUI all receive the same
   `ExecutionTopology` struct. Display formatting is the consumer's responsibility.
5. **Gather once, query many** — topology is collected during backend initialization
   and cached. Subsequent calls are non-collective read-only accessors.

## 1. cobre-comm: Types and Trait

### 1.1 New types

New file `crates/cobre-comm/src/topology.rs`:

```rust
/// Execution topology gathered at communicator initialization.
///
/// Describes the process layout, communication backend metadata, and optional
/// scheduler information. Built once during backend creation and queryable
/// thereafter (no further collectives needed).
pub struct ExecutionTopology {
    /// Which backend is active.
    pub backend: BackendKind,
    /// Total number of processes.
    pub world_size: usize,
    /// Per-host rank assignments, ordered by first rank on each host.
    pub hosts: Vec<HostInfo>,
    /// MPI runtime metadata (None for non-MPI backends).
    pub mpi: Option<MpiRuntimeInfo>,
    /// SLURM job metadata (None when not under SLURM or feature not enabled).
    pub slurm: Option<SlurmJobInfo>,
}

/// A single host and its assigned ranks.
pub struct HostInfo {
    /// Hostname as reported by the backend (MPI_Get_processor_name or OS).
    pub hostname: String,
    /// Sorted list of global ranks on this host.
    pub ranks: Vec<usize>,
}

/// MPI runtime metadata.
pub struct MpiRuntimeInfo {
    /// Implementation version, e.g. "Open MPI v4.1.6".
    pub library_version: String,
    /// Standard version, e.g. "MPI 4.0".
    pub standard_version: String,
    /// Negotiated thread safety level, e.g. "Funneled".
    pub thread_level: String,
}

/// SLURM job metadata.
pub struct SlurmJobInfo {
    /// SLURM_JOB_ID.
    pub job_id: String,
    /// Compact node list, e.g. "compute-[01-04]".
    pub node_list: Option<String>,
    /// CPUs allocated per task.
    pub cpus_per_task: Option<u32>,
}
```

### 1.2 TopologyProvider trait

```rust
/// Companion trait to `Communicator` for topology introspection.
///
/// Orthogonal to `Communicator` (same pattern as `SharedMemoryProvider`).
/// Implementations gather topology once during initialization and cache
/// the result. The accessor is non-collective and allocation-free.
pub trait TopologyProvider: Send + Sync {
    /// Returns the cached execution topology.
    fn topology(&self) -> &ExecutionTopology;
}
```

Consumers that need both communication and topology use combined bounds:
`C: Communicator + TopologyProvider`.

### 1.3 Convenience methods on ExecutionTopology

```rust
impl ExecutionTopology {
    /// Number of distinct hosts.
    pub fn num_hosts(&self) -> usize { self.hosts.len() }

    /// True if all hosts have the same number of ranks.
    pub fn is_homogeneous(&self) -> bool { ... }

    /// Hostname of the first (or only) host. Useful for local/single-node display.
    pub fn leader_hostname(&self) -> &str { ... }
}
```

### 1.4 FerrompiBackend implementation

In `FerrompiBackend::new()`, after `world.split_shared()`:

```rust
// Collective: all ranks gather topology.
let ferrompi_topo = world.topology(&mpi)?;

// Convert ferrompi types → cobre-comm types.
let topology = ExecutionTopology {
    backend: BackendKind::Mpi,
    world_size: size,
    hosts: ferrompi_topo.hosts().iter().map(|h| HostInfo {
        hostname: h.hostname.clone(),
        ranks: h.ranks.iter().map(|&r| r as usize).collect(),
    }).collect(),
    mpi: Some(MpiRuntimeInfo {
        library_version: ferrompi_topo.library_version().to_string(),
        standard_version: ferrompi_topo.standard_version().to_string(),
        thread_level: format!("{:?}", ferrompi_topo.thread_level()),
    }),
    slurm: convert_slurm_info(&ferrompi_topo),  // None when numa feature disabled
};
```

The `topology` field is stored in `FerrompiBackend` alongside `rank`, `size`, etc.

For SLURM conversion:

```rust
#[cfg(feature = "numa")]
fn convert_slurm_info(topo: &ferrompi::TopologyInfo) -> Option<SlurmJobInfo> {
    topo.slurm().map(|s| SlurmJobInfo {
        job_id: s.job_id.clone(),
        node_list: s.node_list.clone(),
        cpus_per_task: s.cpus_per_task.map(|v| v as u32),
    })
}

#[cfg(not(feature = "numa"))]
fn convert_slurm_info(_topo: &ferrompi::TopologyInfo) -> Option<SlurmJobInfo> {
    None
}
```

### 1.5 LocalBackend implementation

```rust
impl TopologyProvider for LocalBackend {
    fn topology(&self) -> &ExecutionTopology {
        // Lazily initialized via OnceLock — LocalBackend is a ZST, so we use
        // a module-level static. The topology is always the same: single process
        // on the current host.
        static LOCAL_TOPOLOGY: OnceLock<ExecutionTopology> = OnceLock::new();
        LOCAL_TOPOLOGY.get_or_init(|| ExecutionTopology {
            backend: BackendKind::Local,
            world_size: 1,
            hosts: vec![HostInfo {
                hostname: hostname_from_system(),
                ranks: vec![0],
            }],
            mpi: None,
            slurm: None,
        })
    }
}
```

`hostname_from_system()` reuses the `/proc/sys/kernel/hostname` → `$HOSTNAME` →
`"unknown"` fallback chain currently in `cobre_io::get_hostname()`. This logic
moves to cobre-comm (where it belongs — it's a backend concern) and cobre-io's
`get_hostname()` becomes a thin re-export or is replaced by reading from the
topology.

### 1.6 CommBackend dispatch

```rust
impl TopologyProvider for CommBackend {
    fn topology(&self) -> &ExecutionTopology {
        match self {
            #[cfg(feature = "mpi")]
            Self::Mpi(backend) => backend.topology(),
            Self::Local(backend) => backend.topology(),
        }
    }
}
```

### 1.7 Feature gating

In `crates/cobre-comm/Cargo.toml`:

```toml
[features]
mpi = ["dep:ferrompi"]
numa = ["mpi", "ferrompi/numa"]
```

The `numa` feature activates ferrompi's SLURM helpers. Since the SLURM helpers are
zero-cost (env var reads, no FFI), `numa` should be enabled by default when `mpi`
is enabled. The workspace `Cargo.toml` or cobre-cli's feature forwarding should
wire this:

```toml
# cobre-cli/Cargo.toml
[features]
mpi = ["cobre-comm/mpi", "cobre-comm/numa"]
```

### 1.8 Public exports

```rust
// cobre-comm/src/lib.rs
pub use topology::{ExecutionTopology, HostInfo, MpiRuntimeInfo, SlurmJobInfo};
pub use traits::TopologyProvider;
```

## 2. cobre-io: Metadata Enrichment

### 2.1 Replace MpiInfo with DistributionInfo

The current `MpiInfo` in `manifest.rs` is too narrow. Replace it with a struct
that captures the full execution environment:

```rust
/// Execution distribution information embedded in metadata files.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributionInfo {
    /// Communication backend: "mpi" or "local".
    pub backend: String,
    /// Total number of processes.
    pub world_size: u32,
    /// Number of processes that participated in computation.
    pub ranks_participated: u32,
    /// Number of distinct physical hosts.
    pub num_nodes: u32,
    /// Rayon threads per process.
    pub threads_per_rank: u32,
    /// MPI implementation version, e.g. "Open MPI v4.1.6".
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mpi_library: Option<String>,
    /// MPI standard version, e.g. "MPI 4.0".
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mpi_standard: Option<String>,
    /// Negotiated MPI thread level, e.g. "Funneled".
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thread_level: Option<String>,
    /// SLURM job ID if running under SLURM.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub slurm_job_id: Option<String>,
}
```

### 2.2 Update TrainingMetadata and SimulationMetadata

Both structs replace `pub mpi: MpiInfo` with `pub distribution: DistributionInfo`.

### 2.3 Update OutputContext

Replace the individual MPI fields with a reference to the topology:

```rust
pub struct OutputContext {
    pub hostname: String,
    pub solver: String,
    pub started_at: String,
    pub completed_at: String,
    pub distribution: DistributionInfo,
}
```

Construction in `run.rs` becomes:

```rust
fn build_distribution_info(
    topology: &ExecutionTopology,
    n_threads: usize,
    ranks_participated: u32,
) -> DistributionInfo {
    DistributionInfo {
        backend: format!("{:?}", topology.backend).to_lowercase(),
        world_size: topology.world_size as u32,
        ranks_participated,
        num_nodes: topology.num_hosts() as u32,
        threads_per_rank: n_threads as u32,
        mpi_library: topology.mpi.as_ref().map(|m| m.library_version.clone()),
        mpi_standard: topology.mpi.as_ref().map(|m| m.standard_version.clone()),
        thread_level: topology.mpi.as_ref().map(|m| m.thread_level.clone()),
        slurm_job_id: topology.slurm.as_ref().map(|s| s.job_id.clone()),
    }
}
```

### 2.4 Hostname

`get_hostname()` in `manifest.rs` currently reads `/proc/sys/kernel/hostname`.
After this change, the canonical hostname comes from `topology.leader_hostname()`
(which uses `MPI_Get_processor_name` under MPI). The `get_hostname()` function
is preserved for non-topology contexts (e.g., Python bindings with no MPI) but
callers in `run.rs` should prefer the topology's hostname.

## 3. cobre-cli: Display

### 3.1 New print function

In `crates/cobre-cli/src/summary.rs`, add `print_execution_topology()` following
the existing section pattern (bold header, indented detail lines):

```rust
pub fn print_execution_topology(
    stderr: &Term,
    topology: &ExecutionTopology,
    n_threads: usize,
)
```

### 3.2 Format specification

**Local backend** (simplest case):

```
Execution
  Backend:   local
  Host:      rogerio-desktop
  Threads:   5 rayon threads
```

**MPI, single node** (common development scenario):

```
Execution
  Backend:   MPI (Open MPI v4.1.6, MPI 4.0)
  Threads:   Funneled, 5 rayon threads per rank
  Layout:    4 ranks on rogerio-desktop
```

**MPI, multi-node** (production HPC):

```
Execution
  Backend:   MPI (Open MPI v4.1.6, MPI 4.0)
  Threads:   Funneled, 5 rayon threads per rank
  Layout:    8 ranks across 2 nodes
    compute-01: ranks 0–3  (4 ranks)
    compute-02: ranks 4–7  (4 ranks)
```

**MPI, multi-node, SLURM** (cluster with scheduler):

```
Execution
  Backend:   MPI (Open MPI v4.1.6, MPI 4.0)
  Threads:   Funneled, 5 rayon threads per rank
  Layout:    8 ranks across 2 nodes
    compute-01: ranks 0–3  (4 ranks)
    compute-02: ranks 4–7  (4 ranks)
  SLURM:     job 123456, nodes compute-[01-02], 8 CPUs/task
```

**Formatting rules**:

- Rank lists use en-dash range notation when contiguous: `0–3` not `0, 1, 2, 3`
- Sparse ranks use comma-separated: `0, 2, 5`
- Per-host lines only shown when `num_hosts > 1`
- SLURM line only shown when `slurm.is_some()`
- Thread level only shown for MPI backend
- "rayon threads" vs "rayon thread" follows singular/plural

### 3.3 Calling point

In `run.rs::setup_communicator()`, immediately after the banner:

```rust
if !quiet {
    crate::banner::print_banner(&stderr);
    crate::summary::print_execution_topology(
        &stderr,
        comm.topology(),
        n_threads,
    );
}
```

This requires `setup_communicator`'s generic bound to become
`C: Communicator + TopologyProvider` (or the concrete `CommBackend` which
implements both).

### 3.4 Enrich `cobre version`

The `version` subcommand currently shows `comm: mpi`. It cannot show the MPI
library version because MPI is not initialized during `cobre version`. This is
acceptable — `cobre version` shows compile-time info, topology is runtime info
shown during `cobre run`. No changes to `version.rs`.

## 4. cobre-python: Exposure

`cobre-python` uses `LocalBackend` directly. `LocalBackend` implements
`TopologyProvider`, so the topology is available without changes. The metadata
JSON will correctly show `backend: "local"`, `world_size: 1`, `num_nodes: 1`.

If a future Python API exposes topology programmatically (e.g. via PyO3), the
`ExecutionTopology` struct is ready for it. No work needed now.

## 5. Metadata JSON Output Examples

### Before (current)

```json
{
  "cobre_version": "0.4.1",
  "hostname": "compute-01",
  "solver": "highs",
  "started_at": "2026-04-10T08:00:00Z",
  "completed_at": "2026-04-10T12:30:00Z",
  "duration_seconds": 16200.0,
  "status": "complete",
  "mpi": {
    "world_size": 8,
    "ranks_participated": 8
  }
}
```

### After

```json
{
  "cobre_version": "0.4.1",
  "hostname": "compute-01",
  "solver": "highs",
  "started_at": "2026-04-10T08:00:00Z",
  "completed_at": "2026-04-10T12:30:00Z",
  "duration_seconds": 16200.0,
  "status": "complete",
  "distribution": {
    "backend": "mpi",
    "world_size": 8,
    "ranks_participated": 8,
    "num_nodes": 2,
    "threads_per_rank": 5,
    "mpi_library": "Open MPI v4.1.6",
    "mpi_standard": "MPI 4.0",
    "thread_level": "Funneled",
    "slurm_job_id": "123456"
  }
}
```

For local runs, `mpi_library`, `mpi_standard`, `thread_level`, and
`slurm_job_id` are omitted from the JSON (via `skip_serializing_if`).

## 6. Implementation Tickets

| #   | Scope    | Crate        | Description                                                                                                                                                                                                             | Depends |
| --- | -------- | ------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| 1   | Model    | cobre-comm   | Add `topology.rs` with `ExecutionTopology`, `HostInfo`, `MpiRuntimeInfo`, `SlurmJobInfo`. Add `TopologyProvider` trait. Implement for `LocalBackend` (synthetic, `OnceLock`). Forward `numa` feature. Export new types. | —       |
| 2   | Backend  | cobre-comm   | Implement `TopologyProvider` for `FerrompiBackend`: call `world.topology(&mpi)` during `new()`, convert and cache. Implement dispatch for `CommBackend`.                                                                | 1       |
| 3   | Metadata | cobre-io     | Replace `MpiInfo` with `DistributionInfo`. Update `OutputContext`, `TrainingMetadata`, `SimulationMetadata`. Add `build_distribution_info()` helper. Update all tests.                                                  | 1       |
| 4   | Display  | cobre-cli    | Add `print_execution_topology()` to `summary.rs` with range formatting. Wire into `setup_communicator()` after banner. Update `RunContext` bounds. Update `OutputContext` construction to use topology.                 | 1, 3    |
| 5   | Python   | cobre-python | Update `OutputContext` construction in `run.rs` to use `LocalBackend.topology()` for `DistributionInfo` fields.                                                                                                         | 1, 3    |

Dependency graph: `1 → {2, 3} → {4, 5}`. Tickets 2 and 3 are independent of
each other but both depend on 1. Tickets 4 and 5 depend on both the trait (1)
and the metadata changes (3).
