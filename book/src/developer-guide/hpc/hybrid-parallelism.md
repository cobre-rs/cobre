# Hybrid Parallelism

## Purpose

This spec defines the hybrid parallelization strategy used by Cobre: a pluggable communication backend (selected at compile time via Cargo feature flags; see [Backend Selection](./backend-selection.md)) for inter-node communication, intra-node shared memory, and topology detection; Rayon as the threading layer for intra-rank parallelism; the design rationale for this split; configuration; initialization; and build integration.

## 1. Hybrid Architecture Overview

Cobre employs a hybrid parallelization strategy optimized for modern HPC architectures with multi-socket, many-core nodes. A **pluggable communication backend** implementing the `Communicator` and `SharedMemoryProvider` traits ([Communicator Trait §1](./communicator-trait.md), [Communicator Trait §4](./communicator-trait.md)) is the backbone for all process-level parallelism and shared memory. The production backend is **ferrompi** (safe Rust MPI bindings; see [Ferrompi Backend](./backend-ferrompi.md)), but alternative backends (local, TCP, shared-memory) can be selected at compile time via Cargo feature flags (see [Backend Selection §1](./backend-selection.md)). **Rayon** fills the one gap inter-process communication cannot cover: thread-level parallelism within a single rank.

| Level                        | Technology                                                                                                     | Scope                            | Responsibility                                                                         |
| ---------------------------- | -------------------------------------------------------------------------------------------------------------- | -------------------------------- | -------------------------------------------------------------------------------------- |
| **Inter-node / Inter-NUMA**  | `Communicator` trait ([Communicator Trait §1](./communicator-trait.md))                                        | Across NUMA domains and nodes    | Work distribution, cut synchronization (`allgatherv`), bound aggregation (`allreduce`) |
| **Intra-node shared memory** | `SharedMemoryProvider` trait ([Communicator Trait §4](./communicator-trait.md))                                | Ranks on the same physical node  | `SharedRegion<T>` for scenario storage and cut pool shared without replication         |
| **Topology detection**       | `SharedMemoryProvider::split_local()` ([Communicator Trait §4.1](./communicator-trait.md)) + SLURM integration | Node and NUMA discovery          | Identify co-located ranks, NUMA domain mapping                                         |
| **Intra-rank threading**     | Rayon                                                                                                          | Threads within a single MPI rank | Parallel LP solves across scenario trajectories                                        |

![Hybrid parallelism architecture — MPI ranks map to NUMA domains, Rayon thread pools within each rank, SharedRegion for intra-node data sharing, ThreadLevel::Funneled constraint](../../images/d07-hybrid-parallelism.svg)

### 1.0a Single-Process Mode

For interface layers that do not use MPI (`cobre-mcp`, `cobre-python`, `cobre-tui`), Cobre operates in **single-process mode**: Rayon threads provide intra-process parallelism without any inter-process communication. In this mode, the `local` communication backend is selected (see [Local Backend](./backend-local.md)). No inter-process communication occurs. Specifically:

- There is exactly one "rank" (the process itself) with all scenarios assigned to it.
- All `Communicator` trait methods ([Communicator Trait §1.1](./communicator-trait.md)) are identity operations or no-ops, as specified in [Local Backend §2](./backend-local.md).
- `SharedMemoryProvider` uses `HeapFallback` -- shared read-only data (opening tree, case data) is allocated as regular per-process heap memory (see [Local Backend §3](./backend-local.md)).
- Rayon thread-level parallelism for LP solves remains fully active with the same thread pool configuration and NUMA policies.

Single-process mode is the required execution mode for Python bindings (GIL/MPI incompatibility -- see [Python Bindings §4](../interfaces/python-bindings.md)) and for the MCP server (long-lived server process incompatible with MPI launchers). See §6a below for the alternative initialization sequence.

### 1.1 Why a Communication Backend + Rayon

The communication backend covers inter-node communication, intra-node shared memory, and topology detection -- everything at the process level. For production HPC deployments, the **ferrompi backend** (see [Ferrompi Backend](./backend-ferrompi.md)) is the recommended choice because MPI provides battle-tested, vendor-optimized collectives and shared memory windows. The one thing the communication backend does not provide is **thread-level parallelism within a single rank**. To parallelize LP solves across cores within a NUMA domain, a threading model is needed. Rayon fills this gap, providing safe, ergonomic data parallelism that integrates naturally with Rust's ownership model and borrow checker.

SDDP's compute pattern — many independent small LP solves sharing large read-only data (scenarios, cuts) — maps naturally to this split:

- **Communication backend ranks** (MPI ranks under the ferrompi backend) provide memory isolation between NUMA domains and between nodes. Each rank has its own address space, avoiding false sharing across NUMA boundaries. Shared memory regions (via the `SharedMemoryProvider` trait) allow ranks on the same node to share large read-only data without replication, reducing memory footprint by an order of magnitude compared to per-rank replication.
- **Rayon threads** within each rank share the rank's address space, enabling fine-grained parallelism over LP solves without data replication within a NUMA domain.

The typical deployment is one MPI rank per NUMA domain. See [SLURM Deployment](./slurm-deployment.md) for concrete job configurations.

### 1.2 ferrompi Capabilities Used

The following table details the capabilities of the **ferrompi backend** (`FerrompiBackend`), which is the production communication backend for MPI-based HPC deployments. For the complete ferrompi backend specification, see [Ferrompi Backend](./backend-ferrompi.md). For alternative backends (local, TCP, shared-memory), see [Backend Selection §1.2](./backend-selection.md).

ferrompi provides safe, idiomatic Rust bindings for MPI. The capabilities used by Cobre:

| Capability                | ferrompi API                                                                                                     | SDDP Use Case                                                         |
| ------------------------- | ---------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| Thread-safe communicators | `Communicator` is `Send + Sync`                                                                                  | Hybrid MPI+threads without `unsafe` sharing of raw `MPI_Comm` handles |
| Shared memory windows     | `SharedWindow<T>` (`FerrompiRegion<T>` wrapping MPI windows -- see [Ferrompi Backend §3](./backend-ferrompi.md)) | Scenario storage and cut pool shared within a node                    |
| Intra-node communicator   | `split_local()` (for ferrompi, returns an intra-node MPI sub-communicator)                                       | Group co-located ranks for shared memory operations                   |
| Collectives               | `allreduce()`, `allgatherv()`, `broadcast()`                                                                     | Bound aggregation, cut synchronization, statistics                    |
| SLURM/NUMA detection      | Topology query APIs                                                                                              | Read resource allocations from scheduler environment                  |
| Threading level           | `Mpi::init_thread(ThreadLevel::Funneled)`                                                                        | Only the main thread makes MPI calls; Rayon workers never call MPI    |

### 1.3 Shared Memory Layout

Ranks on the same physical node share two large data regions via `SharedRegion<T>` (see [Communicator Trait §4.2](./communicator-trait.md)), provided by the `SharedMemoryProvider` trait. For the ferrompi backend, these are backed by MPI windows; for other backends, see the fallback semantics in [Communicator Trait §4.4](./communicator-trait.md). This eliminates per-rank replication:

| Region           | Contents                                                | Access Pattern                                                                 |
| ---------------- | ------------------------------------------------------- | ------------------------------------------------------------------------------ |
| Scenario storage | Pre-generated opening tree noise vectors for all stages | Read-only by all ranks on the node; written once during scenario generation    |
| Cut pool         | Accumulated Benders cuts for all stages                 | Written by the rank designated as the node-local aggregator; read by all ranks |

**Per-rank resources** (not shared):

- One LP solver instance per Rayon worker thread (LP solvers are not thread-safe — see [Solver Workspaces §1.1](../architecture/solver-workspaces.md))
- Thread-local solution buffers, RHS patch buffers, basis cache
- Per-thread state for forward pass trajectory tracking

> The exact memory sizing for shared regions and per-rank resources depends on the problem scale. See [Memory Architecture](./memory-architecture.md) for budget computation and [Production Scale Reference](../overview/production-scale-reference.md) for representative dimensions.

> **Minimal viable note.** The shared memory layout above describes the **target architecture**. In the minimal viable implementation (Phase 5), all data -- including the opening tree, input case data, and cut pool -- is replicated per rank via `HeapFallback` ([Communicator Trait §4.7](./communicator-trait.md)). No `SharedRegion<T>` instances backed by MPI windows are created. The memory overhead of per-rank replication is acceptable at initial scale and will be re-evaluated after profiling against the trigger conditions in [Communicator Trait §4.7](./communicator-trait.md).

## 2. Design Rationale: Why Rayon (OpenMP Deferred)

Rayon is used for intra-rank data parallelism. The decision is driven by four factors:

1. **Pure Rust, no FFI.** Rayon is a native Rust crate. It requires no C compiler, no `#pragma` wrappers, and no `extern "C"` callback trampolines. This eliminates an entire class of build complexity and `unsafe` code.

2. **Ecosystem standard.** Rayon is the de facto standard for data parallelism in the Rust ecosystem. Its `ParallelIterator` trait is well-documented, well-tested, and familiar to Rust developers. Libraries across the ecosystem (including numerical and scientific crates) integrate with Rayon's thread pool.

3. **Borrow checker integration.** Rayon's API is designed around Rust's ownership and borrowing model. The `par_iter()` family requires `&self` (shared reference) on the data being iterated, and closures must be `Send + Sync`. This means the compiler statically prevents data races — a guarantee that OpenMP's C FFI bridge cannot provide without manual `unsafe` reasoning.

4. **Sufficient for the initial implementation.** SDDP's intra-rank workload is embarrassingly parallel: independent LP solves across scenario trajectories with shared read-only data. Rayon's work-stealing scheduler handles this pattern efficiently. The one MPI rank per NUMA domain deployment (§8) confines each Rayon thread pool to a single NUMA domain, mitigating cross-NUMA work migration.

**OpenMP is deferred to post-profiling.** OpenMP would only be reconsidered if production profiling demonstrates that NUMA thread pinning or vendor-optimized scheduling (Intel/AMD runtime libraries) provides a measurable improvement that Rayon cannot match. The three scenarios where OpenMP might be needed are: (a) Rayon's work-stealing causes excessive cross-NUMA memory traffic within a rank, (b) vendor-tuned `schedule(guided)` outperforms work-stealing for the LP solve workload, or (c) HPC profiling tools (VTune, MAP) require OpenMP annotations for thread-level analysis.

**Escape hatch: `ThreadPoolBuilder::spawn_handler`.** If fine-grained thread placement becomes necessary without adopting OpenMP, Rayon's `ThreadPoolBuilder::spawn_handler` callback allows custom thread spawning logic. This callback receives a `ThreadBuilder` and can set CPU affinity (via `libc::sched_setaffinity` or `hwloc`) and NUMA memory policy (via `libnuma`) before the worker thread enters the Rayon run loop. This provides NUMA-aware thread placement within pure Rust, deferring the OpenMP dependency indefinitely.

## 3. MPI vs Rayon Responsibility Split

| Aspect            | MPI Ranks                                                                                         | Rayon Threads                                                                                         |
| ----------------- | ------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| **Purpose**       | Distributed memory, inter-node communication                                                      | Shared memory, intra-NUMA parallelism                                                                 |
| **Granularity**   | Coarse: contiguous blocks of scenario trajectories                                                | Fine: individual stage LP solves within assigned trajectories                                         |
| **Communication** | Explicit: cut synchronization (`MPI_Allgatherv`), bound aggregation (`MPI_Allreduce`), statistics | Implicit: shared read-only case data, scenario tree, cut pool                                         |
| **Memory model**  | Replicated (per-rank solver instances) or shared (MPI windows for scenarios/cuts)                 | Shared within rank (read-only data), thread-local (solver instances, buffers)                         |
| **Load balance**  | Static distribution: scenarios assigned in contiguous blocks to ranks                             | Work-stealing within rank: `par_iter()` dynamically distributes LP solves across Rayon worker threads |

**Thread-trajectory affinity**: Each Rayon worker thread owns one or more complete forward trajectories and also executes the backward pass for those trajectories. This preserves cache locality — the solver basis, scenario data, and LP coefficients remain warm in the thread's cache lines across stages. When the number of forward passes $M$ exceeds the total thread count $N$, threads process multiple trajectories in batches with state save/restore at stage boundaries. See [Training Loop §4.3](../architecture/training-loop.md) and [SDDP Algorithm §3.4](../math/sddp-algorithm.md).

**Forward pass**: Scenarios are distributed across MPI ranks in contiguous blocks. Within each rank, scenarios are parallelized across Rayon worker threads via `par_iter()`. No inter-rank synchronization during the forward pass — each rank processes its assigned trajectories independently. After all ranks complete, `MPI_Allreduce` aggregates lower bound and upper bound statistics.

**Backward pass**: Per-stage synchronization barrier — all threads across all ranks must complete cut generation at stage $t$ before proceeding to stage $t-1$. Within each stage:

| Level     | Operation                                                                     | Result                                                    |
| --------- | ----------------------------------------------------------------------------- | --------------------------------------------------------- |
| 1 — Rayon | Each worker evaluates its assigned trial points and all openings sequentially | Thread-local cut contributions                            |
| 2 — MPI   | `MPI_Allgatherv` collects new cuts from all ranks                             | All ranks have the complete set of new cuts for stage $t$ |

See [Training Loop §6.3](../architecture/training-loop.md) and [Synchronization](./synchronization.md).

## 4. Parallel Configuration

Parallel configuration follows the two-category separation established in [CLI and Lifecycle §6](../architecture/cli-and-lifecycle.md):

### 4.1 Resource Allocations (Read-Only from Environment)

These parameters are determined by the MPI launcher and job scheduler. The program reads them from the environment and **must not override them**.

| Parameter        | Source                                       | Description                   |
| ---------------- | -------------------------------------------- | ----------------------------- |
| MPI rank count   | MPI launcher (`mpiexec -n`, `srun`)          | Number of processes           |
| Threads per rank | `SLURM_CPUS_PER_TASK` or `RAYON_NUM_THREADS` | Rayon worker threads per rank |
| Memory per node  | `SLURM_MEM_PER_NODE`                         | Memory budget for pool sizing |

If `RAYON_NUM_THREADS` is not set and no scheduler is detected, the program defaults to 1 thread per rank.

### 4.2 Rayon Thread Pool Configuration

The Rayon thread pool is configured programmatically at startup using `rayon::ThreadPoolBuilder`. The thread count is resolved from the following sources in priority order:

1. Explicit value from `TrainingConfig` (set by the caller or CLI argument)
2. `RAYON_NUM_THREADS` environment variable
3. `SLURM_CPUS_PER_TASK` environment variable (when running under SLURM)
4. Default: 1 thread (conservative fallback)

The global thread pool is built once during initialization (see §6):

```rust
rayon::ThreadPoolBuilder::new()
    .num_threads(n_threads)
    .build_global()
    .expect("rayon global thread pool");
```

**NUMA-aware thread placement (deferred).** By default, Rayon's work-stealing scheduler does not pin threads to specific cores or NUMA domains. For the initial implementation, NUMA locality is achieved by deploying one MPI rank per NUMA domain (§8), which confines each Rayon pool to a single domain via the OS process affinity inherited from the MPI launcher. If profiling reveals that Rayon threads migrate across cores within the domain, `ThreadPoolBuilder::spawn_handler` can be used to set per-thread CPU affinity without introducing an OpenMP dependency (see §2).

### 4.3 LP Solver Threading Suppression

LP solvers must be forced to single-threaded mode because Cobre handles parallelism at the outer level (one solver per thread). The following environment variables are validated at startup:

| Variable          | Required Value | Affected Solver                                                               |
| ----------------- | -------------- | ----------------------------------------------------------------------------- |
| `HIGHS_PARALLEL`  | `false`        | HiGHS                                                                         |
| `MKL_NUM_THREADS` | `1`            | Any solver using MKL (CPLEX, etc.)                                            |
| `OMP_NUM_THREADS` | `1`            | Solvers that use OpenMP internally (prevents solver-internal thread spawning) |

See [Solver Abstraction §4](../architecture/solver-abstraction.md) for the solver interface contract requiring non-thread-safe solvers.

### 4.4 NUMA Allocation Policy

NUMA memory allocation is configured at startup via the Linux `libnuma` API:

| Policy           | Behavior                                                           | When to Use                                                                      |
| ---------------- | ------------------------------------------------------------------ | -------------------------------------------------------------------------------- |
| Local allocation | Memory allocated on the NUMA node of the calling thread            | Default — ensures LP solver data is NUMA-local                                   |
| First-touch      | Pages allocated on the NUMA node of the first thread to write them | Used for shared arrays during initialization (each thread touches its partition) |

See [Memory Architecture](./memory-architecture.md) for detailed NUMA-aware allocation strategy and budget computation.

## 5. Rayon Parallel Patterns

Rayon provides intra-rank data parallelism through its `ParallelIterator` trait. No C FFI wrapper, callback trampoline, or `unsafe` code is needed — the parallel iteration API is pure safe Rust.

### 5.1 Core Pattern: Parallel Scenario Solving

The fundamental parallel pattern in Cobre is distributing LP solves across Rayon worker threads. Each MPI rank receives a contiguous slice of scenario trajectories; within the rank, `par_iter()` distributes individual trajectory solves across the Rayon thread pool:

> **Decision [DEC-017](../overview/decision-log.md#dec-017) (active):** Communication-free parallel noise generation -- every rank and thread independently derives identical noise via deterministic SipHash-1-3 seed derivation, eliminating MPI broadcast or gather for scenario noise.

```rust
// Per-rank scenario slice -> parallel iteration -> per-worker LP solve
scenario_slice
    .par_iter()
    .map(|scenario| {
        let solver = workspace.acquire_solver();  // thread-local solver instance
        let result = solve_forward_trajectory(solver, scenario, &cuts, &system);
        workspace.release_solver(solver);
        result
    })
    .collect::<Vec<TrajectoryResult>>()
```

Each Rayon worker thread acquires a thread-local LP solver workspace (see [Solver Workspaces §1](../architecture/solver-workspaces.md)), solves the LP for its assigned scenario, and returns the result. Rayon's work-stealing scheduler dynamically balances the load — if one LP solve takes longer than others, idle workers steal remaining work items.

### 5.2 Backward Pass Pattern

The backward pass uses the same `par_iter()` pattern to distribute trial point evaluations across worker threads within each stage:

```rust
// Per-stage: parallel evaluation of trial points across all openings
trial_points
    .par_iter()
    .flat_map(|trial_point| {
        openings.iter().map(move |opening| {
            let solver = workspace.acquire_solver();
            let cut = evaluate_and_extract_cut(solver, trial_point, opening, &cuts);
            workspace.release_solver(solver);
            cut
        })
    })
    .collect::<Vec<Cut>>()
```

### 5.3 First-Touch Initialization

NUMA-aware first-touch initialization of large arrays uses Rayon's `par_chunks_mut()` to ensure each worker thread writes to its partition, placing the corresponding memory pages on the worker's NUMA node:

```rust
// First-touch: each worker initializes its chunk on the local NUMA node
buffer
    .par_chunks_mut(chunk_size)
    .for_each(|chunk| {
        for elem in chunk.iter_mut() {
            *elem = 0.0;
        }
    });
```

### 5.4 Encapsulation Boundary

Rayon is a `cobre-sddp` dependency only. No Rayon types (`ParallelIterator`, `ThreadPool`, `par_iter`) appear in public crate APIs. Other crates interact with the training loop through domain types (`&System`, `&TrainingConfig`, `&CutPool`, etc.). This ensures that the threading implementation is an internal detail of the SDDP training loop and can be changed without affecting downstream crates or interface layers.

## 6. Initialization Sequence

The parallel environment is initialized during the Startup phase (see [CLI and Lifecycle §5](../architecture/cli-and-lifecycle.md)). The ordering is critical -- the communication backend must be initialized before the Rayon thread pool because some backends (notably MPI) require initialization before any threading activity.

**Step 1 -- Backend initialization**: Call `create_communicator()` (see [Backend Selection §4](./backend-selection.md)) to initialize the selected communication backend. For the `mpi` feature, the factory internally calls `ferrompi::Mpi::init_thread(ThreadLevel::Funneled)` (see [Ferrompi Backend §2.1](./backend-ferrompi.md)) -- only the main thread makes MPI calls; Rayon worker threads perform LP solves but never invoke MPI directly. For other backends, initialization is backend-specific (TCP: coordinator handshake; shm: segment creation; local: no-op).

**Step 2 -- Topology detection**: Query `comm.rank()` and `comm.size()` (see [Communicator Trait §2.5](./communicator-trait.md)) to determine rank count and rank ID. Detect the scheduler environment (SLURM, PBS, or local) to read resource allocations. Validate rank count if the job script specifies an expected value.

**Step 3 -- Shared memory communicator**: Create an intra-node communicator via `comm.split_local()` (see [Communicator Trait §4.1](./communicator-trait.md)). For the ferrompi backend, this groups ranks on the same physical node. For local and TCP backends, this returns a `LocalBackend` (see [Local Backend §3.3](./backend-local.md)). For the shm backend, this returns a clone of the full communicator (see [Shm Backend §1.3](./backend-shm.md)).

> **Minimal viable note.** In the minimal viable implementation (Phase 5), Step 3 still calls `split_local()` for API consistency, but no shared memory regions are created in subsequent steps. All `SharedMemoryProvider` operations use `HeapFallback` semantics -- `create_shared_region` allocates per-rank heap memory, `is_leader()` returns `true` on every rank, and `fence()` is a no-op. The intra-node communicator returned by `split_local()` is not used for shared memory coordination; it is retained only so that the initialization sequence does not diverge between the minimal viable and target architectures. See [Communicator Trait §4.7](./communicator-trait.md) for the deferral rationale and trigger conditions.

**Step 4 — Rayon thread pool creation**: Build the global Rayon thread pool with the resolved thread count (see §4.2 for resolution order):

```rust
rayon::ThreadPoolBuilder::new()
    .num_threads(n_threads)
    .build_global()
    .expect("rayon global thread pool");
```

This must happen before any `par_iter()` call. The `build_global()` call is idempotent — if called more than once, subsequent calls return an error (which is why it is called exactly once during initialization).

**Step 5 — LP solver suppression**: Validate that LP solver threading environment variables are set correctly (`HIGHS_PARALLEL=false`, `MKL_NUM_THREADS=1`, `OMP_NUM_THREADS=1`). If not set, set them before any solver instance is created. These are process-level settings that prevent LP solvers from spawning their own internal threads — they do not conflict with Rayon (which manages its own thread pool independently of OpenMP).

**Step 6 — NUMA allocation policy**: On Linux, set the NUMA local allocation policy via `libnuma`. This must happen before any large memory allocation so that subsequent allocations respect NUMA locality.

**Step 7 — Workspace allocation**: Allocate thread-local solver workspaces on each thread's NUMA domain using first-touch initialization (see §5.3). See [Solver Workspaces §1](../architecture/solver-workspaces.md).

**Step 8 — Startup logging**: Rank 0 logs the parallel configuration summary (rank count, threads per rank, total cores, detected scheduler, NUMA topology).

### 6a. Alternative Initialization for Single-Process Mode

When operating in single-process mode (library-mode callers such as `cobre-mcp` and `cobre-python`), the `local` backend is selected via `create_communicator()` (either by explicit `COBRE_COMM_BACKEND=local` or by auto-detection -- see [Backend Selection §2](./backend-selection.md)). `LocalBackend` handles Steps 1-3 as no-ops: `rank()` returns 0, `size()` returns 1, `split_local()` returns `Box::new(LocalBackend)`, and `SharedMemoryProvider` uses `HeapFallback` (see [Local Backend](./backend-local.md)). The initialization sequence continues at Step 4:

**Step 4 -- Rayon thread pool creation**: Identical to the MPI mode. The global thread pool is built with the thread count from `RAYON_NUM_THREADS`, a caller-provided value, or the default. For Python bindings, the GIL is released before entering any Rayon parallel region (the `par_iter()` closures do not hold the GIL).

**Step 5 -- LP solver suppression**: Identical to the MPI mode. Solver threading environment variables are validated and set.

**Step 6 -- NUMA allocation policy**: Identical to the MPI mode on Linux. On platforms without `libnuma`, this step is a no-op.

**Step 7 -- Workspace allocation**: Identical to the MPI mode. Thread-local solver workspaces are allocated with first-touch initialization.

**Step 8 -- Startup logging**: In library mode, startup logging is optional. The library caller decides whether to register event consumers (text logger, JSON-lines writer, etc.) before training begins.

The key difference is that `LocalBackend` implements Steps 1-3 as identity/no-op operations, and `SharedMemoryProvider` uses `HeapFallback` -- shared read-only data is allocated as regular heap memory within the single process (see [Communicator Trait §4.4](./communicator-trait.md)).

## 7. Build Integration

Rayon is a pure Rust crate dependency. No C compiler detection, no wrapper compilation, and no special linker flags are needed for the threading layer.

| Concern                    | Status                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Rayon dependency**       | Added to `cobre-sddp/Cargo.toml` as a regular dependency. Cargo handles compilation and linking automatically.                                                                                                                                                                                                                                                                                                                                               |
| **C compiler requirement** | Not needed for threading. A C compiler is only required if the LP solver FFI bindings (e.g., HiGHS C API) need compilation, which is handled by the solver crate, not by Cobre's threading layer.                                                                                                                                                                                                                                                            |
| **MPI linking**            | Handled entirely by the ferrompi crate dependency -- no manual MPI link flags are needed. Communication backend linking is managed by the `cobre-comm` crate (see [cobre-comm](https://cobre-rs.github.io/cobre/crates/comm.html)). The `mpi` Cargo feature gates the `ferrompi` dependency; when disabled, no MPI headers or libraries are required at build time. See [Backend Selection §1](./backend-selection.md) for the complete feature flag matrix. |
| **LP solver suppression**  | `OMP_NUM_THREADS=1`, `HIGHS_PARALLEL=false`, and `MKL_NUM_THREADS=1` are runtime environment variables set in the job script or validated at startup (§4.3). They are not build-time concerns.                                                                                                                                                                                                                                                               |

## 8. Reference Deployment

Example deployment on a 2-socket AMD EPYC node (64 cores/socket, 4 NUMA domains/socket = 128 cores, 8 NUMA domains):

| Parameter            | Value | Rationale                                                                     |
| -------------------- | ----- | ----------------------------------------------------------------------------- |
| MPI ranks            | 8     | 1 per NUMA domain                                                             |
| Rayon threads / rank | 16    | All cores in the NUMA domain                                                  |
| Total cores          | 128   | Full node utilization                                                         |
| `RAYON_NUM_THREADS`  | `16`  | Matches cores per NUMA domain (or set via `ThreadPoolBuilder::num_threads()`) |
| `OMP_NUM_THREADS`    | `1`   | Suppress solver-internal OpenMP threading                                     |

For complete SLURM job scripts and multi-node deployment patterns, see [SLURM Deployment](./slurm-deployment.md).

## Cross-References

- [Backend Selection](./backend-selection.md) -- Feature flags, runtime selection, factory pattern for communication backend initialization
- [Communicator Trait §1](./communicator-trait.md) -- Communicator trait definition, method contracts, generic parameterization
- [Communicator Trait §4](./communicator-trait.md) -- SharedMemoryProvider trait, SharedRegion\<T\>, HeapFallback semantics
- [Ferrompi Backend](./backend-ferrompi.md) -- FerrompiBackend struct, MPI initialization wrapper, FerrompiRegion\<T\>
- [Local Backend](./backend-local.md) -- LocalBackend ZST, identity/no-op classification, HeapRegion\<T\>
- [cobre-comm](https://cobre-rs.github.io/cobre/crates/comm.html) -- Communication crate architecture, dependency graph, feature matrix
- [SDDP Algorithm §3.4](../math/sddp-algorithm.md) -- Thread-trajectory affinity, backward sync barriers, forward pass state saving
- [Training Loop §4.3](../architecture/training-loop.md) -- Forward pass parallel distribution: contiguous blocks to ranks, thread-trajectory affinity within rank
- [Training Loop §6.3](../architecture/training-loop.md) -- Backward pass parallel distribution: per-stage barrier, MPI_Allgatherv for cut synchronization
- [CLI and Lifecycle §6](../architecture/cli-and-lifecycle.md) -- Resource allocation sourcing (read-only from environment) and algorithm parameter hierarchy
- [Solver Abstraction §4](../architecture/solver-abstraction.md) -- Solver interface contract: solvers are not thread-safe, one instance per thread
- [Solver Workspaces §1](../architecture/solver-workspaces.md) -- Thread-local solver infrastructure, NUMA-local allocation, per-stage basis cache
- [Work Distribution](./work-distribution.md) -- Detailed forward/backward pass scenario distribution and load balancing
- [Synchronization](./synchronization.md) -- Sync points, thread synchronization, lock-free cut aggregation
- [Communication Patterns](./communication-patterns.md) -- ferrompi persistent collectives, SharedWindow\<T\>, async overlap
- [Memory Architecture](./memory-architecture.md) -- Memory budget computation, NUMA-aware allocation, shared memory sizing
- [Shared Memory Aggregation](./shared-memory-aggregation.md) -- Hierarchical cut aggregation, shared memory scenarios
- [Checkpointing](./checkpointing.md) -- Checkpoint strategy and warm-start across MPI ranks
- [SLURM Deployment](./slurm-deployment.md) -- Job scripts, multi-node configuration, performance monitoring
- [Design Principles](../overview/design-principles.md) -- Foundational design goals including distributed I/O
- [Python Bindings](../interfaces/python-bindings.md) -- Single-process mode requirement for Python (GIL/MPI incompatibility)
- [MCP Server](../interfaces/mcp-server.md) -- Single-process mode requirement for MCP server
- [Scenario Generation §2.2b](../architecture/scenario-generation.md) -- Communication-free noise generation work distribution; each Rayon worker thread generates noise independently using deterministic seeds
