# Backend Registration and Selection

## Purpose

This spec defines the two-level mechanism for selecting which `Communicator` backend implementation is used by Cobre: **compile-time feature flags** (primary) and **runtime environment variable selection** (secondary). The design follows the same architectural pattern established for LP solvers in [Solver Abstraction §10](../architecture/solver-abstraction.md): Cargo feature flags control which backends are compiled into the binary, enabling dead code elimination and full monomorphization when a single backend is selected. When multiple backends are compiled in, an optional `COBRE_COMM_BACKEND` environment variable selects among them at startup with a priority-based default. The factory function returns a concrete type via `cfg`-gated match arms -- never a `Box<dyn Communicator>` -- preserving zero-cost abstraction on the hot path.

## 1. Compile-Time Feature Flags

### 1.1 Architectural Precedent

The communicator backend selection mirrors the compile-time solver selection pattern from [Solver Abstraction §10](../architecture/solver-abstraction.md), with one key difference: the solver abstraction expects exactly one solver per build, while the communicator abstraction allows multiple backends to coexist in a single binary. When multiple backends are present, an enum dispatch wrapper provides a single branch point at initialization; the training loop remains generic over `C: Communicator`.

### 1.2 Feature Flag Matrix

Three Cargo features control which communication backends are compiled into the binary. The `local` backend is always available -- it requires no feature flag and has no external dependencies.

| Feature    | Backend             | External Dependencies                                    | Link-Time Requirements          | Binary Size Impact    | Description                                                            |
| ---------- | ------------------- | -------------------------------------------------------- | ------------------------------- | --------------------- | ---------------------------------------------------------------------- |
| `mpi`      | ferrompi (MPI)      | MPI development libraries (OpenMPI, MPICH, or Intel MPI) | Links `libmpi.so` or equivalent | ~200 KB + MPI runtime | Full distributed communication via MPI collectives                     |
| `tcp`      | TCP sockets         | None (`std::net`)                                        | None                            | ~50 KB                | Distributed communication over TCP/IP without MPI dependency           |
| `shm`      | POSIX shared memory | POSIX shared memory APIs (`shm_open`, `mmap`)            | Links `librt` on Linux          | ~30 KB                | Intra-node communication via shared memory segments                    |
| _(always)_ | Local (no-op)       | None                                                     | None                            | ~1 KB (inlined away)  | Single-process mode; all collectives are identity operations or no-ops |

**Feature flag rules:**

1. `local` is unconditional -- always compiled, no feature gate, no external dependencies.
2. Feature flags are additive -- enabling `mpi` does not disable `tcp` or `shm`.
3. The `mpi` feature gates the `ferrompi` crate dependency. When disabled, no MPI headers or libraries are required at build time.

### 1.3 Build Profiles

Recommended feature combinations for each deployment target:

| Build Profile    | Cargo Features | Target                                                     | Rationale                                                                                                                                                                                                                                     |
| ---------------- | -------------- | ---------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **CLI / HPC**    | `mpi`          | `cobre` binary for cluster deployment via `mpiexec`/`srun` | MPI is the production communication layer. `local` is implicitly available as a fallback for single-rank runs.                                                                                                                                |
| **Python wheel** | `tcp,shm`      | `cobre-python` (`cdylib`) for PyPI distribution            | No MPI dependency on user machines ([Python Bindings §1.2](../interfaces/python-bindings.md)). TCP enables multi-process coordination from Python. Shared memory enables efficient intra-node communication. `local` is implicitly available. |
| **Test / CI**    | _(none)_       | Unit tests, integration tests, CI pipelines                | Only `local` is available. Deterministic, no external dependencies, fast compilation.                                                                                                                                                         |
| **Development**  | `mpi,tcp,shm`  | Developer workstation with all backends for testing        | All backends compiled in. Runtime selection via `COBRE_COMM_BACKEND` env var.                                                                                                                                                                 |

**Example Cargo invocations:**

```bash
# CLI / HPC production build
cargo build --release --features mpi

# Python wheel build (maturin)
maturin build --release --features tcp,shm

# Test / CI (local-only, no features)
cargo test

# Development build (all backends)
cargo build --features mpi,tcp,shm
```

### 1.4 Monomorphization for Single-Feature Builds

When only one communication backend is compiled in (the common case for production builds), the compiler resolves the generic parameter `C: Communicator` to a single concrete type. This enables:

1. **Full inlining** -- The `local` backend's `allgatherv` (memcpy), `allreduce` (identity), `broadcast` (no-op), and `barrier` (no-op) are inlined directly into the training loop, compiling to zero instructions for no-op operations.
2. **Dead code elimination** -- All `cfg`-gated code paths for non-compiled backends are removed entirely from the binary.
3. **No enum dispatch overhead** -- The `CommBackend` enum (SS4) is not instantiated; the factory returns the concrete type directly.
4. **Identical codegen to hardcoded calls** -- The generated assembly for `train::<MpiCommunicator>(...)` is identical to code that directly calls ferrompi functions without any abstraction layer.

## 2. Runtime Backend Selection

When multiple backends are compiled into the same binary (e.g., the development build profile with `mpi,tcp,shm`), the `COBRE_COMM_BACKEND` environment variable selects which backend to use at startup. This is the **secondary** selection mechanism -- it only applies when compile-time feature flags have left the choice ambiguous.

### 2.1 Environment Variable

| Variable             | Values                               | Default | Description                                               |
| -------------------- | ------------------------------------ | ------- | --------------------------------------------------------- |
| `COBRE_COMM_BACKEND` | `auto`, `mpi`, `tcp`, `shm`, `local` | `auto`  | Selects the communication backend among those compiled in |

When `COBRE_COMM_BACKEND` is not set or is set to `auto`, the priority-based auto-detection algorithm (SS2.2) selects the backend.

### 2.2 Auto-Detection Algorithm

The `auto` mode selects a backend by testing compiled-in backends in priority order. Each backend has a runtime precondition that must be satisfied for it to be selected. The first backend whose precondition is met wins.

**Priority chain** (highest to lowest):

```
         COBRE_COMM_BACKEND set?
         /                     \
       yes                     no (or "auto")
        |                       |
   value compiled in? -----> Priority chain:
   /           \
  yes          no             1. mpi feature compiled?
   |            |                 yes --> MPI launch detected?
  use it    ERROR                         yes --> use mpi
   |        (SS6)                         no  --> fall through
   |                          2. tcp feature compiled? [NOT YET IMPLEMENTED]
   |                              yes --> COBRE_TCP_COORDINATOR set?
   |                                      yes --> use tcp
   |                                      no  --> fall through
   |                          3. shm feature compiled? [NOT YET IMPLEMENTED]
   |                              yes --> COBRE_SHM_NAME set?
   |                                      yes --> use shm
   |                                      no  --> fall through
   |                          4. local (always available)
   |                              use local
   v                              |
  [initialize selected backend]<--+
```

> **Note:** Steps 2 (TCP) and 3 (SHM) describe planned backends that are not yet implemented. The current `BackendKind` enum has only `Auto`, `Mpi`, and `Local` variants. See [Backend TCP](./backend-tcp.md) and [Backend SHM](./backend-shm.md) for the individual backend specifications and their implementation status.

**MPI launch detection**: The auto-detection algorithm probes for environment variables that MPI launchers inject into the process environment. The presence of any of the following variables indicates that the process was launched under an MPI runtime:

| Variable               | Set By                     |
| ---------------------- | -------------------------- |
| `PMI_RANK`             | MPICH, Intel MPI, Cray MPI |
| `PMI_SIZE`             | MPICH, Intel MPI, Cray MPI |
| `OMPI_COMM_WORLD_RANK` | OpenMPI                    |
| `OMPI_COMM_WORLD_SIZE` | OpenMPI                    |
| `MPI_LOCALRANKID`      | Intel MPI (alternative)    |
| `SLURM_PROCID`         | SLURM `srun` with MPI      |

If any of these variables is set and the `mpi` feature is compiled in, the MPI backend is selected. This is consistent with the existing MPI detection logic in [Hybrid Parallelism §6](./hybrid-parallelism.md) and the scheduler detection pattern in [CLI and Lifecycle §6.3](../architecture/cli-and-lifecycle.md).

### 2.3 Single-Feature Build Behavior

When only one backend is compiled in, the `COBRE_COMM_BACKEND` variable is accepted if it names the compiled-in backend or `auto`, and rejected with a clear error (SS6) if it names an absent backend. The runtime selection code is eliminated by the compiler -- all `cfg`-gated branches for absent backends are removed at compile time.

## 3. Backend Configuration

Each backend accepts additional configuration via environment variables. These follow the resource allocation pattern from [Hybrid Parallelism §4.1](./hybrid-parallelism.md): the program reads them from the environment and does not override them.

### 3.1 Per-Backend Environment Variables

**MPI backend** (`mpi` feature):

| Variable                     | Required              | Description                                                                                                                                                                                                                                        |
| ---------------------------- | --------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| _(MPI launcher environment)_ | Yes (set by launcher) | Rank count, rank ID, and MPI runtime configuration are provided by the MPI launcher (`mpiexec`, `srun`). The program reads them via `MPI_Comm_rank` / `MPI_Comm_size` after `MPI_Init_thread`. No Cobre-specific environment variables are needed. |

**TCP backend** (`tcp` feature):

| Variable                 | Required | Default   | Description                                                                                                                                                        |
| ------------------------ | -------- | --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `COBRE_TCP_COORDINATOR`  | Yes      | _(none)_  | Hostname or IP address of the coordinator (rank 0). All ranks connect to this address during initialization. See [TCP Backend §8.1](./backend-tcp.md) for details. |
| `COBRE_TCP_PORT`         | No       | `29500`   | TCP port on which the coordinator listens. All ranks must use the same port value.                                                                                 |
| `COBRE_TCP_RANK`         | Yes      | _(none)_  | Rank index of this process (`0..COBRE_TCP_SIZE`).                                                                                                                  |
| `COBRE_TCP_SIZE`         | Yes      | _(none)_  | Total number of ranks in the communicator.                                                                                                                         |
| `COBRE_TCP_BIND_ADDR`    | No       | `0.0.0.0` | Local address to bind for incoming peer connections.                                                                                                               |
| `COBRE_TCP_TIMEOUT_SECS` | No       | `60`      | Timeout in seconds for collective operations and connection attempts. See [Backend TCP SS5.1](./backend-tcp.md).                                                   |

**Shared memory backend** (`shm` feature):

| Variable         | Required | Default  | Description                                                                                 |
| ---------------- | -------- | -------- | ------------------------------------------------------------------------------------------- |
| `COBRE_SHM_NAME` | Yes      | _(none)_ | Name of the POSIX shared memory segment. All ranks on the same node must use the same name. |
| `COBRE_SHM_RANK` | Yes      | _(none)_ | Rank index of this process (`0..COBRE_SHM_SIZE`).                                           |
| `COBRE_SHM_SIZE` | Yes      | _(none)_ | Total number of ranks sharing the memory segment.                                           |

**Local backend** (always available):

| Variable | Required | Description                                                                   |
| -------- | -------- | ----------------------------------------------------------------------------- |
| _(none)_ | --       | The local backend requires no configuration. It operates as rank 0 of size 1. |

### 3.2 Environment Variable Naming Convention

All Cobre communication environment variables use the `COBRE_` prefix, followed by the backend name in uppercase, followed by the parameter name. This avoids collisions with MPI launcher variables, OpenMP variables, and solver variables, consistent with the variable namespace separation established in [Hybrid Parallelism §4](./hybrid-parallelism.md).

## 4. Factory Pattern

The `create_communicator` function instantiates the selected backend and returns it as a concrete type. The implementation uses `cfg`-gated code paths to ensure that only compiled-in backends generate code, and that single-feature builds achieve full monomorphization with zero dispatch overhead.

### 4.1 Single-Feature Build (Monomorphized)

When only one backend is compiled in, the factory returns the concrete type directly via `impl Communicator`:

```rust
#[cfg(all(feature = "mpi", not(feature = "tcp"), not(feature = "shm")))]
pub fn create_communicator() -> Result<impl Communicator, BackendError> {
    if mpi_launch_detected() {
        Ok(MpiCommunicator::init()?)
    } else {
        Ok(LocalCommunicator::new())
    }
}
```

When only `local` is available (test/CI builds):

```rust
/// Creates a communicator for local-only builds (no features enabled).
///
/// Always returns `LocalCommunicator`. All collective operations compile
/// to no-ops or identity operations after inlining.
#[cfg(not(any(feature = "mpi", feature = "tcp", feature = "shm")))]
pub fn create_communicator() -> Result<LocalCommunicator, BackendError> {
    Ok(LocalCommunicator::new())
}
```

### 4.2 Multi-Feature Build (Enum Dispatch)

When multiple backends are compiled in, the factory returns a `CommBackend` enum that wraps each possible concrete type. The enum provides a **single branch point at initialization** -- the `match` happens once when the communicator is created, and the training loop receives the variant through the generic `C: Communicator` parameter.

```rust
/// Enum wrapper for multi-feature builds.
///
/// Each variant wraps a concrete backend type. The enum implements
/// `Communicator` by delegating to the inner type via match arms.
/// This introduces one level of enum dispatch per collective call,
/// which is negligible compared to the cost of the collective itself.
#[cfg(any(
    all(feature = "mpi", any(feature = "tcp", feature = "shm")),
    all(feature = "tcp", feature = "shm"),
))]
pub enum CommBackend {
    #[cfg(feature = "mpi")]
    Mpi(MpiCommunicator),

    #[cfg(feature = "tcp")]
    Tcp(TcpCommunicator),

    #[cfg(feature = "shm")]
    Shm(ShmCommunicator),

    Local(LocalCommunicator),
}
```

The `CommBackend` enum implements `Communicator` by delegating to the contained backend:

```rust
impl Communicator for CommBackend {
    fn allgatherv<T: CommData>(
        &self,
        send: &[T],
        recv: &mut [T],
        counts: &[usize],
        displs: &[usize],
    ) -> Result<(), CommError> {
        match self {
            #[cfg(feature = "mpi")]
            CommBackend::Mpi(inner) => inner.allgatherv(send, recv, counts, displs),

            #[cfg(feature = "tcp")]
            CommBackend::Tcp(inner) => inner.allgatherv(send, recv, counts, displs),

            #[cfg(feature = "shm")]
            CommBackend::Shm(inner) => inner.allgatherv(send, recv, counts, displs),

            CommBackend::Local(inner) => inner.allgatherv(send, recv, counts, displs),
        }
    }

    // ... analogous implementations for allreduce, broadcast, barrier, rank, size
}
```

The factory for multi-feature builds reads `COBRE_COMM_BACKEND` and applies the selection logic from SS2:

```rust
/// Creates a communicator for multi-feature builds.
///
/// Reads `COBRE_COMM_BACKEND` to determine which backend to use.
/// If unset or "auto", applies the priority chain from SS2.2.
#[cfg(any(
    all(feature = "mpi", any(feature = "tcp", feature = "shm")),
    all(feature = "tcp", feature = "shm"),
))]
pub fn create_communicator() -> Result<CommBackend, BackendError> {
    let requested = std::env::var("COBRE_COMM_BACKEND")
        .unwrap_or_else(|_| "auto".to_string());

    match requested.as_str() {
        "auto" => auto_detect(),

        #[cfg(feature = "mpi")]
        "mpi" => Ok(CommBackend::Mpi(MpiCommunicator::init()?)),

        #[cfg(feature = "tcp")]
        "tcp" => Ok(CommBackend::Tcp(TcpCommunicator::connect()?)),

        #[cfg(feature = "shm")]
        "shm" => Ok(CommBackend::Shm(ShmCommunicator::attach()?)),

        "local" => Ok(CommBackend::Local(LocalCommunicator::new())),

        other => Err(BackendError::InvalidBackend {
            requested: other.to_string(),
            available: available_backends(),
        }),
    }
}
```

### 4.3 Dispatch Cost Analysis

The `CommBackend` enum dispatch adds one `match` per collective call. This cost is negligible in context:

| Operation                                  | Enum dispatch cost            | Operation cost (production scale)                                    | Overhead ratio |
| ------------------------------------------ | ----------------------------- | -------------------------------------------------------------------- | -------------- |
| `allgatherv` (cuts, per stage)             | ~1 ns (branch prediction hit) | ~2.7 ms ([Communication Patterns §3.1](./communication-patterns.md)) | < 0.0001%      |
| `allgatherv` (trial points, per iteration) | ~1 ns                         | ~17 ms                                                               | < 0.00006%     |
| `allreduce` (bounds, per iteration)        | ~1 ns                         | ~0.1 ms                                                              | < 0.001%       |
| `barrier` (checkpoint only)                | ~1 ns                         | ~0.5 ms                                                              | < 0.0002%      |

The enum dispatch cost is unmeasurable against the latency of any real communication operation. Multi-feature builds sacrifice nothing meaningful in performance.

## 5. Library-Mode API

Library-mode callers (`cobre-python`, `cobre-mcp`) bypass the environment variable mechanism and select a backend programmatically. This is necessary because:

1. Library callers control the process environment and should not rely on environment variable side effects.
2. The backend choice is a library initialization concern, not a user configuration concern.
3. Multiple independent Cobre sessions within the same process may need different backends (e.g., test harnesses).

### 5.1 Programmatic Backend Selection

Library-mode callers construct the communicator directly and pass it to the training function:

```rust
// Example: cobre-python, single-process mode
pub fn example_python_entry(case_path: &Path) -> Result<TrainingResult, CobreError> {
    let comm = LocalCommunicator::new();
    let config = load_and_validate(case_path)?;
    train(&comm, &config)
}

// Example: cobre-python, distributed mode
pub fn example_python_distributed(
    case_path: &Path,
    coordinator: &str,
    rank: usize,
    size: usize,
) -> Result<TrainingResult, CobreError> {
    let tcp_config = TcpConfig {
        coordinator: coordinator.parse()?,
        rank,
        size,
        bind_addr: "0.0.0.0".parse().unwrap(),
        connect_timeout: Duration::from_secs(30),
    };
    let comm = TcpCommunicator::connect_with_config(tcp_config)?;
    let config = load_and_validate(case_path)?;
    train(&comm, &config)
}
```

### 5.2 Backend Kind Enum

A `BackendKind` enum is provided for callers that want to use the factory function with a programmatic selection rather than environment variables:

```rust
pub enum BackendKind {
    /// Automatic detection using the priority chain (SS2.2).
    Auto,

    /// MPI backend (requires `mpi` feature).
    Mpi,

    /// TCP backend (requires `tcp` feature). Caller must provide
    /// configuration via `TcpConfig`.
    Tcp(TcpConfig),

    /// Shared memory backend (requires `shm` feature). Caller must
    /// provide configuration via `ShmConfig`.
    Shm(ShmConfig),

    /// Local (no-op) backend. Always available.
    Local,
}
```

### 5.3 Python Bindings Integration

The Python bindings ([Python Bindings §1.2](../interfaces/python-bindings.md)) operate in single-process mode by default, using `LocalCommunicator`. The binding layer constructs the communicator before releasing the GIL and entering the Rust computation:

```python
import cobre

# Default: local backend, single-process mode
result = cobre.train("/path/to/case")

# Explicit backend selection (future: distributed Python)
result = cobre.train(
    "/path/to/case",
    backend="tcp",
    coordinator="10.0.0.1:9000",
    rank=0,
    size=4,
)
```

The Python `train()` function maps the `backend` parameter to a `BackendKind` value, constructs the corresponding communicator, and passes it to the Rust training function.

### 5.4 MCP Server Integration

The MCP server ([MCP Server](../interfaces/mcp-server.md)) is a long-lived single-process server that always uses `LocalCommunicator`. It constructs the communicator once at server startup and reuses it for all training invocations:

```rust
fn start_mcp_server() -> Result<(), McpError> {
    let comm = LocalCommunicator::new();
    // Server loop: receive requests, run training with `comm`
    mcp_serve(&comm)
}
```

## 6. Error Handling

### 6.1 Compile-Time Enforcement

Non-compiled backend types do not exist in the binary. Any attempt to reference `MpiCommunicator` in a build without the `mpi` feature produces a Rust compilation error -- impossible configurations are rejected before the binary is produced.

For feature flag combinations that are structurally invalid (e.g., a hypothetical future constraint where two features conflict), `compile_error!` guards in the crate root enforce the constraint at compile time:

```rust
#[cfg(all(feature = "example_conflict_a", feature = "example_conflict_b"))]
compile_error!("Features 'example_conflict_a' and 'example_conflict_b' are mutually exclusive.");
```

Currently, all communication feature flags are additive (SS1.2), so no `compile_error!` guards are needed. This mechanism is reserved for future constraints.

### 6.2 Backend Not Compiled In (Runtime)

When a user requests a backend via `COBRE_COMM_BACKEND` that was not compiled into the binary, the error message lists the available backends:

```rust
/// Error type for backend selection failures.
#[derive(Debug)]
pub enum BackendError {
    /// The requested backend is not compiled into the binary.
    BackendNotAvailable {
        requested: String,
        available: Vec<String>,
    },

    /// The requested backend name is not recognized.
    InvalidBackend {
        requested: String,
        available: Vec<String>,
    },

    /// The backend initialization failed (e.g., MPI runtime not found,
    /// TCP coordinator unreachable, shared memory segment does not exist).
    InitializationFailed {
        backend: String,
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    /// Required environment variables for the selected backend are missing.
    MissingConfiguration {
        backend: String,
        missing_vars: Vec<String>,
    },
}
```

**Structured error output**: When backend selection fails, the error is reported using the `IncompatibleSettings` error kind from the error kind registry ([Structured Output §2.3](../interfaces/structured-output.md)):

```json
{
  "kind": "IncompatibleSettings",
  "message": "Communication backend 'mpi' is not available in this build",
  "context": {
    "setting": "COBRE_COMM_BACKEND",
    "requested_value": "mpi",
    "available_backends": ["tcp", "shm", "local"],
    "build_features": ["tcp", "shm"]
  },
  "suggestion": "Set COBRE_COMM_BACKEND to one of: tcp, shm, local. To use MPI, rebuild with --features mpi."
}
```

### 6.3 Initialization Failure

When a backend is correctly selected but fails to initialize (MPI runtime not installed, TCP coordinator unreachable, shared memory segment not created), the error is reported using the `MpiError` error kind (generalized to cover all communication backend failures):

```json
{
  "kind": "MpiError",
  "message": "TCP backend failed to connect to coordinator at 10.0.0.1:9000",
  "context": {
    "operation": "connect",
    "backend": "tcp",
    "coordinator": "10.0.0.1:9000",
    "error_code": 111,
    "detail": "Connection refused (os error 111)"
  },
  "suggestion": "Verify that the TCP coordinator is running at 10.0.0.1:9000 and that the port is accessible from this host."
}
```

### 6.4 Missing Configuration

When a backend requires environment variables that are not set:

```json
{
  "kind": "IncompatibleSettings",
  "message": "TCP backend requires COBRE_TCP_COORDINATOR, COBRE_TCP_RANK, and COBRE_TCP_SIZE to be set",
  "context": {
    "setting": "COBRE_COMM_BACKEND",
    "requested_value": "tcp",
    "missing_variables": [
      "COBRE_TCP_COORDINATOR",
      "COBRE_TCP_RANK",
      "COBRE_TCP_SIZE"
    ]
  },
  "suggestion": "Set the required environment variables. Example: COBRE_TCP_COORDINATOR=host COBRE_TCP_PORT=29500 COBRE_TCP_RANK=0 COBRE_TCP_SIZE=4 cobre run /path/to/case"
}
```

### 6.5 Error Precedence

Backend selection occurs during the Startup phase ([CLI and Lifecycle §5](../architecture/cli-and-lifecycle.md)), before any input validation or solver initialization. The initialization sequence is:

1. Parse CLI arguments
2. **Select and initialize communication backend** (this spec)
3. Detect scheduler environment ([CLI and Lifecycle §6.3](../architecture/cli-and-lifecycle.md))
4. Initialize OpenMP ([Hybrid Parallelism §6, Step 4](./hybrid-parallelism.md))
5. Proceed to Validation phase

A backend selection error terminates the program with exit code 4 (runtime error, per [CLI and Lifecycle §4](../architecture/cli-and-lifecycle.md)) before any further initialization.

## 7. Relationship to Initialization Sequence

This spec defines **what** is selected and how. The initialization of the selected backend follows the existing initialization sequences:

- **MPI backend**: Follows [Hybrid Parallelism §6](./hybrid-parallelism.md) Steps 1-3 (MPI init, topology detection, shared memory communicator).
- **TCP backend**: Connects to the coordinator, exchanges peer addresses, establishes point-to-point connections. The protocol details are specified in the TCP backend spec (Epic 02).
- **Shared memory backend**: Opens or creates the named shared memory segment, maps it into the process address space. The protocol details are specified in the shared memory backend spec (Epic 02).
- **Local backend**: No initialization needed. Returns immediately with rank=0, size=1. Follows [Hybrid Parallelism §6a](./hybrid-parallelism.md) (single-process mode, Steps 1-3 skipped).

After backend initialization, the common initialization sequence resumes at Step 4 (OpenMP configuration) regardless of which backend was selected.

## Cross-References

- [Communicator Trait §1](./communicator-trait.md) -- The `Communicator` trait, `CommData`, `ReduceOp`, and `CommError` type definitions that the factory function returns
- [Communicator Trait §3](./communicator-trait.md) -- Generic parameterization pattern (`train<C: Communicator>`) that the factory feeds into
- [Solver Abstraction §10](../architecture/solver-abstraction.md) -- Compile-time solver selection via Cargo feature flags; the architectural precedent for this spec's feature flag mechanism
- [Hybrid Parallelism §4.1](./hybrid-parallelism.md) -- Resource allocation environment variables (read-only from environment); the pattern followed by SS3 backend configuration variables
- [Hybrid Parallelism §6](./hybrid-parallelism.md) -- MPI initialization sequence (Steps 1-3) that the MPI backend follows
- [Hybrid Parallelism §6a](./hybrid-parallelism.md) -- Single-process mode initialization that the local backend follows
- [CLI and Lifecycle §5](../architecture/cli-and-lifecycle.md) -- Startup phase where backend selection occurs
- [CLI and Lifecycle §6.3](../architecture/cli-and-lifecycle.md) -- Scheduler detection pattern (environment variable probing) analogous to MPI launch detection in SS2.2
- [Design Principles §5](../overview/design-principles.md) -- Rust implementation strategy, feature flags, compile-time selection patterns
- [Structured Output §2.3](../interfaces/structured-output.md) -- Error kind registry (`IncompatibleSettings`, `MpiError`) used for structured error reporting in SS6
- [Python Bindings §1.2](../interfaces/python-bindings.md) -- Single-process execution mode and library-mode initialization
- [MCP Server](../interfaces/mcp-server.md) -- Single-process server that always uses the local backend
