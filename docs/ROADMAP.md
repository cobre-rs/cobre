# Cobre -- Post-v0.1.0 Roadmap

This document tracks implementation-level work items planned for releases after
v0.1.0. It covers deferred features, HPC optimizations, and post-MVP crates.
For methodology-level roadmap pages (algorithm theory, spec evolution), see the
[cobre-docs roadmap](https://cobre-rs.github.io/cobre-docs/roadmap/).

---

## Inflow Truncation Methods

**Status**: `Truncation` was delivered in v0.1.1. Only `TruncationWithPenalty` remains deferred.

PAR(p) models can produce negative inflow realisations. The penalty method
(implemented in v0.1.0) handles this by adding a high-cost slack variable to
the water balance row. The `Truncation` method (implemented in v0.1.1) clamps
negative AR model draws to zero before LP patching. One additional method from
the literature is planned:

- **Truncation with penalty** -- combine modified bounds with a bounded slack
  variable, matching the SPTcpp reference implementation most closely.

The `InflowNonNegativityMethod` enum will gain a `TruncationWithPenalty { cost }`
variant when this is implemented. The `Truncation` variant was added in v0.1.1.
Existing `"penalty"` and `"none"` configs remain unchanged.

**Full design**: `docs/deferred-truncation-design.md`
**Paper reference**: Oliveira et al. (2022), _Energies_ 15(3):1115.

---

## HPC Optimizations

**Status**: Inter-rank (MPI) parallelism is implemented. Intra-rank thread
parallelism is the baseline for v0.1.0: rayon-based work-stealing controlled
via the `--threads` flag and `COBRE_THREADS` environment variable (default: 1
thread per rank). The optimizations below are deferred to post-v0.1.0 releases.

The items are grouped by expected implementation horizon.

### Near-term (v0.1.x / v0.2.x)

These optimizations address correctness prerequisites and high-return
single-node improvements that build directly on the rayon baseline.

- **NUMA-aware thread/memory placement** -- The current rayon thread pool does
  not constrain threads to NUMA nodes, which causes cross-socket memory traffic
  on multi-socket servers (2-, 4-, and 8-socket configurations are common in HPC
  clusters used for energy planning). Pin thread pools to NUMA domains using
  `hwloc` or `libnuma`, and allocate per-thread workspaces from the local NUMA
  node's memory. This is the highest single-node performance lever available
  after the rayon baseline.
  - _Expected impact_: 20--40% reduction in memory-bandwidth-bound LP solve time
    on dual-socket servers; highly case-size-dependent.
  - _Prerequisites_: Stable rayon thread-pool partitioning from v0.1.0; requires
    `hwloc` system library or a Rust binding (`hwloc2` crate).

- **Work-stealing tuning** -- The v0.1.0 rayon partition divides scenarios
  uniformly across threads. Real case scenario counts are not always divisible
  evenly, and forward-pass LP solve times vary per scenario due to basis quality.
  Profile rayon task granularity against representative case sizes and consider
  hybrid chunking (coarse initial partition, fine-grained stealing) to reduce
  thread idle time.
  - _Expected impact_: 5--15% throughput improvement on irregular scenario
    distributions; larger gains when scenario solve times are highly variable.
  - _Prerequisites_: Access to representative benchmark cases; requires Criterion
    or a custom timing harness to measure per-thread utilisation.

- **Memory pool / arena allocation** -- The training loop allocates LP workspaces,
  cut coefficient buffers, and scenario trajectory vectors on every iteration.
  Replacing per-iteration heap allocation with a thread-local arena (e.g., using
  the `bumpalo` crate) eliminates allocator contention between rayon threads and
  reduces jitter in iteration timing.
  - _Expected impact_: 5--20% reduction in iteration wall time on large cases
    (hundreds of stages, thousands of scenarios); allocator contention is the
    dominant source of variance at high thread counts.
  - _Prerequisites_: Profiling to confirm allocation pressure is a bottleneck;
    requires identifying all hot-path allocation sites in `cobre-sddp`.

- **Profile-guided optimization (PGO) builds** -- Standard `cargo build --release`
  uses static heuristics for inlining and branch prediction. A PGO build uses
  representative training runs to collect execution profiles, then feeds them back
  to the compiler for profile-specific optimizations. This is a compiler-side
  technique requiring no code changes.
  - _Expected impact_: 5--15% throughput improvement "for free" on the LP solve
    and cut evaluation hot paths; gains are compiler-version-dependent.
  - _Prerequisites_: A stable representative benchmark case; CI integration for
    the two-stage PGO build workflow (instrument build → profile collection →
    optimized build).

- **MPI + threads interaction validation** -- The `FerrompiBackend` uses
  `MPI_Init_thread` with `MPI_THREAD_MULTIPLE` to allow rayon threads to call MPI
  functions concurrently. The actual thread safety level granted by OpenMPI and
  MPICH at various versions is not fully characterized. Document the minimum thread
  support level required, add a startup assertion that the granted level meets the
  requirement, and validate under both OpenMPI 4.x/5.x and MPICH 4.x.
  - _Expected impact_: Correctness prerequisite for MPI + rayon combinations;
    prevents silent data corruption on MPI implementations that downgrade the
    thread support level.
  - _Prerequisites_: Test matrix across OpenMPI and MPICH versions; access to
    a multi-rank CI configuration.

### Longer-term (v0.3+)

These optimizations require new infrastructure, external dependencies, or
significant algorithmic changes. They are tracked here to preserve design intent
but are not planned for the v0.1.x/v0.2.x windows.

- **SIMD vectorization for cut evaluation hot paths** -- The backward pass
  evaluates all active cuts at each stage to compute the value function
  approximation. The inner product `coefficients · x` is computed for every cut
  in the pool, making it a natural SIMD target. Explicit SIMD (via `std::simd`
  or `wide`) or auto-vectorization hints can accelerate this loop, particularly
  when cut pool sizes reach the thousands.
  - _Expected impact_: 2--4x speedup on cut evaluation inner loops for large cut
    pools (> 1000 cuts per stage); smaller gains when cut selection keeps pools
    small.
  - _Prerequisites_: Stable `std::simd` API (currently nightly-only); profiling
    to confirm cut evaluation is a dominant cost at target case sizes.

- **Asynchronous cut exchange** -- Currently, the backward pass performs a
  blocking `allgatherv` at each stage to synchronize cuts across MPI ranks before
  proceeding to the next stage. Overlapping this communication with computation
  from the previous stage can hide network latency on distributed clusters.
  Requires restructuring the backward pass to pipeline cut exchange with LP solves.
  - _Expected impact_: Near-linear scaling improvement on clusters with high
    inter-node latency (> 10 µs); minimal gain on shared-memory or low-latency
    InfiniBand fabrics.
  - _Prerequisites_: Non-blocking MPI collectives (`MPI_Iallgatherv`) support
    in `ferrompi`; significant refactor of the backward-pass stage loop.

- **GPU acceleration** -- GPU solver backends (e.g., cuOSQP, NVIDIA cuPDLP, or
  Gurobi GPU) can accelerate individual LP solves, particularly for large
  network topologies with many buses and lines. Integrating a GPU backend requires
  a new `SolverInterface` variant and a device memory management layer.
  - _Expected impact_: Order-of-magnitude LP solve speedup for large network
    cases (> 500 buses); negligible or negative impact for small cases due to
    PCIe transfer overhead.
  - _Prerequisites_: A GPU-capable LP solver with a Rust-compatible C API;
    CUDA or ROCm toolchain in CI; new ADR for GPU backend selection.

- **MPI one-sided communication (MPI_Put / MPI_Get)** -- The current cut exchange
  uses `allgatherv` (collective, two-sided). MPI one-sided operations allow a
  rank to read remote memory directly without the target rank actively
  participating, which can reduce synchronization overhead when ranks finish
  their backward stage at different times.
  - _Expected impact_: Reduced tail latency in cut exchange when rank completion
    times are skewed; benefit depends heavily on the MPI implementation's
    one-sided performance characteristics.
  - _Prerequisites_: One-sided MPI primitives in `ferrompi`; careful window
    synchronization design to avoid data races; performance characterization on
    target cluster hardware.

- **OpenMP interop evaluation** -- Some LP solver backends (HiGHS, CLP) use
  OpenMP internally for parallelism within a single solve. When Cobre's rayon
  thread pool and the solver's OpenMP pool compete for the same CPU cores, thread
  oversubscription degrades performance. Evaluate whether disabling solver-level
  OpenMP (setting `OMP_NUM_THREADS=1`) and relying entirely on Cobre's rayon
  partitioning produces better overall throughput, or whether a co-scheduling
  strategy is needed.
  - _Expected impact_: Correctness of performance (avoiding oversubscription);
    may improve or degrade throughput depending on case size and thread count.
  - _Prerequisites_: Ability to control OpenMP thread count per solver backend;
    benchmark suite covering a range of case sizes and thread configurations.

**Spec references**: `cobre-docs/src/specs/hpc/parallelism-model.md`,
`cobre-docs/src/specs/hpc/memory-model.md`.

---

## Post-MVP Crates

Three crates are stubbed in the workspace but not yet implemented.

### `cobre-mcp`

A standalone MCP (Model Context Protocol) server that exposes Cobre case
management and result queries to AI coding assistants. Runs as a
single-process binary; does not use MPI. Depends only on `cobre-core` and
`cobre-io`.

### `cobre-tui`

A `ratatui`-based terminal UI library consumed by `cobre-cli`. Will provide
an interactive dashboard showing live convergence plots, scenario statistics,
and solver timing during a run. Depends only on `cobre-core`; the rendering
loop reads events published by `cobre-sddp`.

---

**Methodology reference**: `cobre-docs/src/roadmap/` for the full algorithm
extension roadmap and theoretical grounding.
