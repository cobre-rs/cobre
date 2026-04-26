# Memory Architecture

## Purpose

This spec defines the memory architecture for Cobre: data ownership categories, per-rank memory budget derived from approved specifications, NUMA-aware allocation principles, and memory growth analysis across SDDP iterations. This spec aggregates memory-relevant requirements from [Solver Workspaces](../architecture/solver-workspaces.md), [Shared Memory Aggregation](./shared-memory-aggregation.md), and [Communication Patterns](./communication-patterns.md) into a unified memory perspective.

## 1. Data Ownership Model

### 1.1 Ownership Categories

All runtime data falls into one of four ownership categories, each with distinct allocation and access patterns:

| Category                 | Owner                           | Access During Training                           | Allocation Strategy                                    | Examples                                                                 |
| ------------------------ | ------------------------------- | ------------------------------------------------ | ------------------------------------------------------ | ------------------------------------------------------------------------ |
| **Shared read-only**     | Node (via SharedWindow) or rank | Read by all threads, never modified after init   | SharedWindow leader allocation or per-rank replication | Opening tree, input case data, PAR parameters, spectral factors          |
| **Thread-local mutable** | Rayon thread                    | Exclusive read/write by owning thread            | First-touch on owning thread's NUMA node               | Solver workspace, solution buffers, basis cache, cut accumulation buffer |
| **Rank-local growing**   | MPI rank                        | Read by all threads, written at stage boundaries | Pre-allocated with growth capacity                     | Cut pool (grows each iteration)                                          |
| **Temporary**            | Rayon thread                    | Allocated/freed within a single solve            | Pre-allocated in workspace, reused                     | RHS patch buffer, scratch arrays                                         |

**Single-process mode note**: In single-process mode (used by `cobre-python` and `cobre-mcp`), the "Shared read-only" category uses regular per-process heap allocation instead of `SharedWindow<T>`. MPI windows are not available without MPI initialization. The ownership semantics are otherwise identical -- data is still read-only during training, allocated once at initialization, and shared across all Rayon threads within the process. See [Hybrid Parallelism §1.0a](./hybrid-parallelism.md) for single-process mode details.

![Per-rank memory architecture — shared read-only (System, PAR, opening tree), thread-local mutable (solver workspaces), rank-local growing (cut pool), temporary (scratch buffers)](../../images/d08-memory-architecture.svg)

### 1.2 Concurrency Model

Cobre uses Rayon for intra-rank threading (see [Hybrid Parallelism §5](./hybrid-parallelism.md)). The data sharing model follows Rust's ownership and borrowing rules, enforced at compile time via `Send` and `Sync` trait bounds:

| Rust/Rayon Pattern                                 | Data Category        | Implication                                                                            |
| -------------------------------------------------- | -------------------- | -------------------------------------------------------------------------------------- |
| `&T` where `T: Sync`                               | Shared read-only     | All threads see the same reference; no synchronization needed for reads                |
| Indexed by `rayon::current_thread_index()`         | Thread-local mutable | Each thread accesses its own workspace by thread index                                 |
| Written between `par_iter` calls (single-threaded) | Rank-local growing   | Written between parallel regions (single-threaded merge), read during parallel regions |

No `Arc`, `RwLock`, or `Mutex` is needed — Rust's ownership model, Rayon's implicit join synchronization, and thread-index-indexed arrays provide the necessary semantics. The compiler enforces that shared data is `Sync` and that mutable data is exclusively owned.

## 2. Per-Rank Memory Budget

### 2.1 Derivable Components

The following memory estimates are derived from production-scale dimensions in approved specs. Reference configuration: 160 hydros, 130 thermals, 60 stages, 192 forward passes, 10 openings, 15K cut capacity, 48 threads per rank, 4 ranks per node.

Under Strategy 2+3 (StageLpCache), the memory model is a two-tier structure: thread-local solver workspaces plus a shared StageLpCache via `SharedRegion<T>`.

**Thread-local per rank (48 threads, NUMA-local, HiGHS):**

| Component                 |        Size | Derivation                                                                            | Source                                                         |
| ------------------------- | ----------: | ------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| Solver workspaces (×48)   |   ~1,737 MB | 48 threads × ~36 MB per workspace (solver instance + buffers + per-stage basis cache) | [Solver Workspaces §1.2](../architecture/solver-workspaces.md) |
| Forward pass state        |      ~18 MB | 192 trajectories × ~9 KB state vector (1,120 doubles + metadata) × ~1 stage buffered  | [Training Loop §5.1](../architecture/training-loop.md)         |
| MPI communication buffers |      ~10 MB | Send/receive buffers for `MPI_Allgatherv` (cuts: ~3.2 MB, states: ~1.72 MB per stage) | [Communication Patterns §2](./communication-patterns.md)       |
| **Thread-local per rank** | **~1.8 GB** |                                                                                       |                                                                |

**SharedRegion (1 copy per node, NUMA-interleaved):**

| Component              |         Size | Derivation                                                                   | Source                                                             |
| ---------------------- | -----------: | ---------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| StageLpCache           |   ~22,300 MB | 60 stages × ~378 MB per stage (structural CSC + 15K cut slots in CSC)        | [Solver Abstraction §11.4](../architecture/solver-abstraction.md)  |
| Cut pool metadata      |       ~12 MB | 60 stages × 15K cuts × (intercept + activity + metadata) ≈ ~200 bytes/cut    | [Cut Management Impl §1.3](../architecture/cut-management-impl.md) |
| Opening tree           |      ~0.8 MB | 10 openings × 60 stages × 160 entities × 8 bytes                             | [Scenario Generation §2.3](../architecture/scenario-generation.md) |
| Input case data        |       ~20 MB | System entities, PAR parameters, correlation factors, block/exchange factors | [Internal Structures](../data-model/internal-structures.md)        |
| **SharedRegion total** | **~22.3 GB** |                                                                              |                                                                    |

**Node total (4 ranks):**

| Component                       |         Size |
| ------------------------------- | -----------: |
| Thread-local (4 ranks × 1.8 GB) |      ~7.1 GB |
| SharedRegion (1 copy)           |     ~22.3 GB |
| **Node total (HiGHS)**          | **~27.7 GB** |
| **Node total (CLP)**            | **~26.0 GB** |

CLP workspaces are ~21 MB/thread (vs ~36 MB for HiGHS) due to 1-byte basis status codes, reducing thread-local memory by ~720 MB per rank.

### 2.2 SharedRegion Savings

Under Strategy 2+3, the dominant memory structure — the StageLpCache (~22.3 GB) — is shared across all ranks on the same node via `SharedRegion<T>` (see [Shared Memory Aggregation §1](./shared-memory-aggregation.md)). This is the primary memory optimization:

| Data                | Per-Rank (replicated) | SharedRegion (1 copy/node) | Savings (4 ranks/node) |
| ------------------- | --------------------: | -------------------------: | ---------------------: |
| StageLpCache        |            ~22,300 MB |                 ~22,300 MB |             ~66,900 MB |
| Cut pool metadata   |                ~12 MB |                     ~12 MB |                 ~36 MB |
| Opening tree        |               ~0.8 MB |                    ~0.8 MB |                ~2.4 MB |
| Input case data     |                ~20 MB |                     ~20 MB |                 ~60 MB |
| **Total shareable** |            ~22,333 MB |                 ~22,333 MB |             ~67,000 MB |

Without SharedRegion, each rank would replicate the entire StageLpCache (~22.3 GB × 4 ranks = ~89.2 GB), making the node total ~96 GB — still within the 384 GB node capacity but significantly more than the ~27.7 GB with sharing.

**Single-process mode note**: The SharedRegion savings table above applies only to multi-rank MPI deployments. In single-process mode (used by `cobre-python` and `cobre-mcp`), all data resides in a single process. The per-process memory footprint includes one copy of the StageLpCache (~22.3 GB) plus thread-local workspaces. At production scale with 48 threads, expect approximately 24 GB total memory usage.

### 2.3 Memory Growth

The StageLpCache is the dominant memory structure. It is physically pre-allocated at initialization (15K cut slots × 60 stages), so no dynamic allocation occurs during training. As cuts accumulate, StageLpCache CSC slots are populated with coefficient data — this is logical growth within the pre-allocated structure.

| Iteration | Active Cuts (approx) | StageLpCache Utilization | Notes                                   |
| --------- | -------------------: | -----------------------: | --------------------------------------- |
| 1         |                  192 |                    ~1.3% | $M$ cuts (one per forward pass)         |
| 50        |               ~5,000 |                     ~33% | Before cut selection starts pruning     |
| 100       |              ~10,000 |                     ~67% | Cut selection bounds active count       |
| 200       |              ~15,000 |                    ~100% | At capacity; dominated cuts deactivated |

The StageLpCache pre-allocates all 15K cut slots at initialization (per [Solver Abstraction §11.4](../architecture/solver-abstraction.md)). Physical memory is committed up front via the SharedRegion allocation. The stage transition time improves at lower utilization (fewer cut rows → smaller CSC → faster `passModel`).

### 2.4 Scaling with Problem Size

Memory scales primarily with the number of hydro plants (state dimension), the number of stages, and thread count:

| Dimension           | Effect on Memory                                                                                 |
| ------------------- | ------------------------------------------------------------------------------------------------ |
| Hydro count (×2)    | State dimension doubles → StageLpCache nnz per cut doubles → StageLpCache approximately doubles. |
| Stage count (×2)    | Per-stage basis cache doubles. StageLpCache doubles (one per stage). SharedRegion doubles.       |
| Forward passes (×2) | Forward state buffers double. StageLpCache growth rate doubles (more cuts per iteration).        |
| Threads (×2)        | Thread-local workspaces double (linear). SharedRegion unchanged (shared, not per-thread).        |
| Openings (×2)       | Opening tree doubles. Modest impact (~0.8 MB → ~1.6 MB at production scale).                     |

## 3. NUMA-Aware Allocation

### 3.1 Principles

> **Decision [DEC-011](../overview/decision-log.md#dec-011) (active):** One MPI rank per NUMA domain is the recommended deployment model; confines each Rayon thread pool to a single NUMA domain.

Modern HPC nodes have multiple NUMA domains. Memory access latency varies significantly between local and remote NUMA domains (typical: 1.5-3× slower for remote access). Cobre follows three NUMA principles:

**Principle 1 — Thread-owns-workspace**: Each solver workspace is allocated by the thread that will use it, ensuring first-touch allocation on the thread's local NUMA node. The workspace is never accessed by other threads.

**Principle 2 — One rank per NUMA domain**: The recommended deployment is one MPI rank per NUMA domain (see [Hybrid Parallelism §4.4](./hybrid-parallelism.md) and [SLURM Deployment](./slurm-deployment.md)). This ensures that all threads within a rank share the same NUMA domain, making shared read-only data (case data, opening tree) local to all threads.

**Principle 3 — First-touch initialization**: Large arrays (solution buffers, basis cache) are initialized by the owning thread within a Rayon parallel scope, not by the main thread. This ensures the OS places memory pages on the NUMA node where they will be accessed.

### 3.2 NUMA Initialization Sequence

The initialization of per-thread resources follows the sequence documented in [Solver Workspaces §1.3](../architecture/solver-workspaces.md):

1. Main thread determines NUMA topology via `cobre_comm::slurm` helpers (see [Hybrid Parallelism §1.2](./hybrid-parallelism.md))
2. Rayon parallel scope is entered (via `rayon::scope` or `par_iter`)
3. Each thread creates its solver instance (first-touch allocates solver internals on local NUMA)
4. Each thread initializes its per-stage basis cache, solution buffers, and scratch arrays
5. Parallel scope completes (implicit join)
6. Main thread verifies all workspaces are initialized

### 3.3 Cache Line Alignment

All per-thread data structures must be padded to cache line boundaries (64 bytes) to prevent false sharing between adjacent threads' workspaces. This applies to:

- Solver workspace array entries (indexed by thread ID)
- Cut accumulation buffers (see [Synchronization §3.1](./synchronization.md))
- Per-thread solve statistics counters

### 3.4 NUMA Topology (AMD EPYC 9R14)

The production reference hardware (AMD EPYC 9R14, 192 cores) has the following NUMA characteristics:

| Property                  | Value                                                                                      |
| ------------------------- | ------------------------------------------------------------------------------------------ |
| NUMA domains (NPS4)       | 4                                                                                          |
| CCDs per NUMA domain      | 3                                                                                          |
| Physical cores per domain | 24 (48 hardware threads with SMT)                                                          |
| L3 cache per CCD          | 32 MB (NOT unified within NUMA — 3 separate 32 MB caches per domain, 96 MB total per NUMA) |
| Local bandwidth           | ~60–80 GB/s sustained per NUMA domain (3 DDR5-4800 channels)                               |
| Cross-NUMA bandwidth      | ~30–40 GB/s per direction (Infinity Fabric)                                                |

### 3.5 NUMA Latency Reference

Representative NUMA latency characteristics:

| Access Pattern             | Typical Latency | Impact on LP Solve   |
| -------------------------- | --------------- | -------------------- |
| Local NUMA                 | ~80 ns          | Baseline             |
| Adjacent NUMA              | ~120 ns (1.5×)  | Moderate slowdown    |
| Remote NUMA (cross-socket) | ~200 ns (2.5×)  | Significant slowdown |

These values are representative of AMD EPYC and Intel Xeon platforms. Actual latencies vary by hardware. The key insight is that remote NUMA access can slow LP solves by 2-3× when solver working data (LU factorization, pricing vectors) is allocated on the wrong NUMA node — motivating Principle 1 (thread-owns-workspace).

### 3.6 NUMA-Interleaved Allocation for SharedRegion

> **Decision [DEC-010](../overview/decision-log.md#dec-010) (active):** NUMA-interleaved allocation (`mbind(MPOL_INTERLEAVE)`) for the SharedRegion holding the StageLpCache; distributes pages round-robin across all NUMA domains.

The StageLpCache (~22.3 GB) is shared across all ranks on the same node via `SharedRegion<T>`. If allocated entirely on the leader rank's NUMA domain, all other ranks incur cross-NUMA latency for every `passModel` read — creating a memory controller bottleneck.

**Solution**: Allocate the SharedRegion with NUMA-interleaved page placement (`mbind(MPOL_INTERLEAVE)` or `numactl --interleave=all`). This distributes memory pages round-robin across all 4 NUMA domains.

| Approach                   | Effective Bandwidth | Stage Transition Time | Node Memory |
| -------------------------- | ------------------: | --------------------: | ----------: |
| Leader-only placement      |         ~15–20 GB/s |           ~18.9–25 ms |    ~27.7 GB |
| NUMA-interleaved (adopted) |            ~44 GB/s |               ~8.6 ms |    ~27.7 GB |
| Per-rank replication       |         ~60–80 GB/s |               ~5.4 ms |    ~92.0 GB |

**Recommendation**: NUMA-interleaved SharedRegion is the default. It provides ~2.5× speedup over leader-only placement at no additional memory cost. Per-rank replication provides a further ~1.6× speedup but triples memory usage; this is a profile-guided optimization for memory-rich nodes.

**L3 cache impact**: The `passModel` operation reads ~378 MB per stage — far exceeding the 32 MB L3 per CCD (or even the 96 MB total per NUMA domain). This is a DRAM-bandwidth-bound streaming read, not a cache-friendly working-set operation. In contrast, the LP solve working set (~15 MB for solver internals + LU factorization) fits within a single CCD's 32 MB L3. Strategy 2+3 improves cache behavior: the per-thread footprint (~36 MB) doesn't pollute L3/TLB, whereas the per-thread CSR assembly approach required ~375 MB per thread for on-the-fly LP construction.

## 4. Hot-Path Allocation Avoidance

### 4.1 Requirement

No heap allocation (`malloc`/`new`/`Vec::push` beyond capacity) is permitted during the SDDP hot path — the forward pass LP solves, backward pass LP solves, and cut accumulation. All buffers are pre-allocated at initialization and reused.

### 4.2 Pre-Allocated Components

| Component                | Pre-allocated At | Reused During                               | Reference                                                         |
| ------------------------ | ---------------- | ------------------------------------------- | ----------------------------------------------------------------- |
| StageLpCache (CSC)       | Initialization   | Read every stage transition via `passModel` | [Solver Abstraction §11.4](../architecture/solver-abstraction.md) |
| Solver instance          | Initialization   | Every LP solve (all iterations)             | [Solver Workspaces §1.2](../architecture/solver-workspaces.md)    |
| Primal/dual buffers      | Initialization   | Solution extraction after each solve        | [Solver Workspaces §1.2](../architecture/solver-workspaces.md)    |
| RHS patch buffer         | Initialization   | Scenario patching before each solve         | [Solver Workspaces §1.2](../architecture/solver-workspaces.md)    |
| Cut pool metadata slots  | Initialization   | Cut metadata insertion via bitmap           | [Solver Abstraction §5](../architecture/solver-abstraction.md)    |
| Cut accumulation buffers | Initialization   | Per-thread cut collection each stage        | [Synchronization §3.1](./synchronization.md)                      |
| MPI send/recv buffers    | Initialization   | `MPI_Allgatherv` each stage                 | [Communication Patterns §2](./communication-patterns.md)          |

### 4.3 Allocation Monitoring

In debug/test builds, an allocation tracker can detect unexpected hot-path allocations by hooking the global allocator and flagging any allocation between the start and end of an LP solve. This is a development tool, not a production feature.

## Cross-References

- [Solver Workspaces §1](../architecture/solver-workspaces.md) — Detailed workspace contents, sizing, NUMA-aware initialization, basis cache
- [Solver Abstraction §5](../architecture/solver-abstraction.md) — Cut pool preallocation, slot assignment, capacity management
- [Solver Abstraction §11.4](../architecture/solver-abstraction.md) — StageLpCache design, sizing, ownership, update and read contracts
- [Shared Memory Aggregation §1](./shared-memory-aggregation.md) — SharedRegion allocation, StageLpCache as primary candidate, NUMA-interleaved placement
- [Communication Patterns §2](./communication-patterns.md) — Data payload sizes for MPI buffers
- [Communication Patterns §5](./communication-patterns.md) — SharedWindow capabilities and shared data candidates
- [Synchronization §3](./synchronization.md) — Cut accumulation buffers, cache line alignment
- [Hybrid Parallelism §1.2](./hybrid-parallelism.md) — ferrompi capabilities, SLURM/NUMA detection
- [Hybrid Parallelism §4.4](./hybrid-parallelism.md) — NUMA binding policy
- [Scenario Generation §2.3](../architecture/scenario-generation.md) — Opening tree size and memory layout
- [Cut Management Impl §4.2](../architecture/cut-management-impl.md) — Per-cut wire size and cut pool sizing
- [Training Loop §5.1](../architecture/training-loop.md) — State vector dimensions
- [Internal Structures](../data-model/internal-structures.md) — In-memory data model for system entities
