# Shared Memory and Aggregation

## Purpose

This spec defines the intra-node shared memory usage patterns, reproducibility guarantees, and performance monitoring for Cobre. It extends [Communication Patterns §5](./communication-patterns.md) with detailed `SharedRegion<T>` allocation and access patterns, consolidates reproducibility requirements from across the approved specs, and defines performance monitoring interpretation. For the collective operations and their semantics, see [Communication Patterns](./communication-patterns.md) and [Synchronization](./synchronization.md).

## 1. Intra-Node Shared Memory

### 1.1 SharedRegion\<T\> Usage Model

The `SharedMemoryProvider` trait ([Communicator Trait SS4](./communicator-trait.md)) allows ranks on the same physical node to share memory regions via `SharedRegion<T>` ([Communicator Trait SS4.2](./communicator-trait.md)) handles. Cobre uses a **leader allocation pattern**: one rank per node (the leader, determined by `is_leader()` from the `SharedMemoryProvider` trait, which returns `true` for rank 0 within the intra-node communicator from `split_local()` ([Communicator Trait SS4.1](./communicator-trait.md))) allocates the shared region; other ranks on the same node access it via the region handle.

| Role     | Allocation                                                   | Read Access                     | Write Access                                   |
| -------- | ------------------------------------------------------------ | ------------------------------- | ---------------------------------------------- |
| Leader   | Calls `create_shared_region(count)` -- allocates full region | `region.as_slice()` (zero-copy) | `region.as_mut_slice()`, then `region.fence()` |
| Follower | Receives handle to leader's region (size 0 local allocation) | `region.as_slice()` (zero-copy) | Not permitted (leader writes on behalf)        |

The `SharedRegion<T>` type provides RAII semantics — `Drop` automatically frees the underlying shared memory resource (MPI window for ferrompi, OS shared segment for shm, `Vec<T>` for HeapFallback). See [Communicator Trait SS4.2](./communicator-trait.md).

### 1.2 Shared Data: StageLpCache (Primary)

The StageLpCache ([Solver Abstraction SS11.4](../architecture/solver-abstraction.md)) is the **primary candidate** for `SharedRegion<T>` — it is the dominant memory structure at ~22.3 GB and provides the largest sharing benefit:

| Property        | Value                                                                                                                                                                                                                                                                                                                                 |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Size            | ~22.3 GB (60 stages × ~378 MB per stage at 15K cut capacity)                                                                                                                                                                                                                                                                          |
| Access pattern  | Read-only during forward/backward passes; updated between iterations by leader rank                                                                                                                                                                                                                                                   |
| Sharing benefit | Eliminates ~67 GB replication (4 ranks × 22.3 GB → 1 copy)                                                                                                                                                                                                                                                                            |
| Write phase     | Between iterations: leader rank writes new cuts and deactivation bounds, `region.fence()` + barrier. Cut selection computation is distributed across all ranks ([Cut Selection Strategy Trait SS2.2a](../architecture/cut-selection-trait.md)), but SharedRegion writes remain leader-only — the single-writer contract is preserved. |
| Memory layout   | Per-stage CSC arrays (structural template + cut slots) — sequential access during `passModel`                                                                                                                                                                                                                                         |

**NUMA-interleaved allocation** is mandatory for StageLpCache. At >5 GB, leader-only placement creates a memory controller bottleneck. `mbind(MPOL_INTERLEAVE)` distributes pages round-robin across all NUMA domains, providing ~44 GB/s effective read bandwidth (vs ~15–20 GB/s with leader-only). See [Memory Architecture §3.6](./memory-architecture.md).

**Allocation protocol:**

1. Create intra-node communicator via `comm.split_local()` ([Communicator Trait SS4.1](./communicator-trait.md))
2. Leader calls `create_shared_region::<u8>(total_bytes)` with NUMA-interleaved allocation hint; followers allocate size 0
3. Leader assembles initial StageLpCache from StageTemplate CSC + empty cut slots for each stage
4. `region.fence()` ensures initial data is visible to all ranks
5. During training passes: all ranks read `StageLpCache[t]` via `region.as_slice()` (zero-copy)
6. Between iterations: leader writes new cut data and deactivation bounds, then `region.fence()` + barrier

> **Decision [DEC-016](../overview/decision-log.md#dec-016) (active):** Cut selection uses deferred parallel execution — stages distributed across ranks and threads, with DeactivationSet allgatherv and leader-only SharedRegion write.

**Update path** (between iterations, leader-only writes):

- Write new cut coefficients into pre-allocated CSC slots (~300 MB/iteration at ~5 ms) — every iteration
- Apply deactivations from the parallel cut selection phase: set deactivated cut row bounds to $-\infty$ (non-binding, no structural change) — selection iterations only. The deactivation decisions are computed in parallel by all ranks ([Cut Management Implementation SS7.1a](../architecture/cut-management-impl.md)), but the SharedRegion writes are performed exclusively by the leader rank ([Cut Management Implementation SS7.1b](../architecture/cut-management-impl.md))
- `region.fence()` + barrier after all writes
- Read path: `passModel(StageLpCache[t])` = sequential bulk read at ~44 GB/s → **~8.6 ms** per stage transition

### 1.3 Shared Data: Cut Pool Metadata

Cut pool metadata (intercepts, activity bitmap, iteration/forward_pass indices, binding counts) is included in the SharedRegion alongside StageLpCache. Total: ~12 MB across 60 stages at 15K capacity — negligible relative to StageLpCache.

### 1.4 Shared Data: Opening Tree (Secondary)

The opening tree (fixed noise vectors for the backward pass — see [Scenario Generation §2.3](../architecture/scenario-generation.md)) is a secondary candidate for `SharedRegion<T>`:

| Property        | Value                                                                                                  |
| --------------- | ------------------------------------------------------------------------------------------------------ |
| Size            | $N_{\text{openings}} \times T \times N_{\text{entities}} \times 8$ bytes (~0.8 MB at production scale) |
| Access pattern  | Read-only during training (generated once before first iteration)                                      |
| Sharing benefit | Avoid replicating ~0.8 MB per rank on same node (4 ranks = ~2.4 MB saved)                              |
| Write phase     | Leader generates during initialization, `region.fence()` ensures visibility                            |
| Memory layout   | Opening-major ordering for backward pass locality (see scenario-generation.md §2.3)                    |

> **Decision [DEC-017](../overview/decision-log.md#dec-017) (active):** Communication-free parallel noise generation -- every rank and thread independently derives identical noise via deterministic SipHash-1-3 seed derivation, eliminating MPI broadcast or gather for scenario noise.

**Generation protocol:**

1. Create intra-node communicator via `comm.split_local()` ([Communicator Trait SS4.1](./communicator-trait.md))
2. Leader calls `create_shared_region::<f64>(count)` ([Communicator Trait SS4.1](./communicator-trait.md)) with total opening tree size; followers allocate size 0
3. Distribute generation work across ranks within the node (each rank generates its assigned portion via contiguous block assignment)
4. Each rank writes its portion directly to the shared region at the correct offset
5. `region.fence()` ensures all writes are visible to all ranks on the node
6. All ranks read any opening via `region.as_slice()` (zero-copy)

### 1.5 Shared Data: Input Case Data (Secondary)

Static input data (hydro parameters, thermal parameters, system topology) loaded during initialization is read-only during training. Sharing via `SharedRegion<T>` avoids per-rank replication:

| Data                   | Approximate Size | Shareable? | Notes                             |
| ---------------------- | ---------------- | ---------- | --------------------------------- |
| System entity data     | ~10 MB           | Yes        | Read-only after initialization    |
| PAR model parameters   | ~5 MB            | Yes        | Preprocessed contiguous arrays    |
| Correlation factors    | ~2 MB            | Yes        | Spectral factors, one per profile |
| Block/exchange factors | ~1 MB            | Yes        | Loaded once, read-only            |

Whether these smaller data structures justify `SharedRegion<T>` overhead (allocation, fence, pointer indirection) depends on the number of ranks per node. With 4+ ranks per node, the savings compound, but the absolute benefit (~60 MB) is negligible relative to StageLpCache (~67 GB saved).

## 2. Intra-Node Cut Aggregation

### 2.1 Baseline: Flat allgatherv

The approved architecture uses flat `allgatherv` for cut distribution: every rank sends its cuts and every rank receives all cuts. There is no hierarchical aggregation, no root-based reduction, and no point-to-point messaging. See [Synchronization §1.1](./synchronization.md) and [Communication Patterns §1.2](./communication-patterns.md).

### 2.2 Optimization: Two-Level Aggregation

For deployments with many ranks per node (8+), a two-level aggregation can reduce inter-node `allgatherv` volume by first aggregating cuts within a node:

1. **Level 1 (Intra-node)**: Ranks on the same node contribute cuts to shared memory via `SharedRegion<T>` (via the `SharedMemoryProvider` trait). The node leader collects and deduplicates if applicable.
2. **Level 2 (Inter-node)**: Node leaders participate in `allgatherv` across nodes, exchanging the aggregated per-node cut sets.
3. **Distribute within node**: Node leaders write received remote cuts to shared memory; `region.fence()` ensures visibility.

This reduces the `allgatherv` participant count from $R$ total ranks to $R / R_{\text{node}}$ node leaders, reducing collective latency for large rank counts. However, it adds intra-node synchronization overhead.

> **Design decision**: Two-level aggregation is an optimization for large-scale deployments (64+ ranks). For the initial implementation with moderate rank counts (4-16 ranks), flat `allgatherv` is sufficient. Profiling should guide whether to enable this optimization. See [Communication Patterns §3.2](./communication-patterns.md) for bandwidth analysis showing communication fraction < 2% even on Ethernet.

## 3. Reproducibility Guarantees

### 3.1 Requirement

Given the same inputs and random seed, Cobre must produce **bit-for-bit identical** results regardless of:

| Must Be Independent Of           | Mechanism                                                                     |
| -------------------------------- | ----------------------------------------------------------------------------- |
| Number of MPI ranks              | Deterministic seeding, contiguous block distribution, deterministic cut slots |
| Number of Rayon threads per rank | Thread-local accumulation, Rayon join, single-threaded merge                  |
| Execution timing/ordering        | Identity-based seeding, deterministic `allgatherv` rank ordering              |

### 3.2 Component-Level Reproducibility

Each component of the SDDP pipeline has specific reproducibility mechanisms:

| Component                | Mechanism                                                                   | Reference                                                                    |
| ------------------------ | --------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| Scenario noise           | Deterministic seed from `(base_seed, iteration, scenario, stage)`           | [Scenario Generation §2.2](../architecture/scenario-generation.md)           |
| Opening tree             | Deterministic seed from `(base_seed, opening_index, stage)`                 | [Scenario Generation §2.3](../architecture/scenario-generation.md)           |
| Forward pass assignment  | Contiguous block distribution by rank index                                 | [Work Distribution §3.1](./work-distribution.md)                             |
| Backward pass assignment | Contiguous block distribution by rank index                                 | [Work Distribution §2.1](./work-distribution.md)                             |
| Cut slot positions       | Deterministic from `(iteration, forward_pass_index)`                        | [Cut Management Implementation §4.3](../architecture/cut-management-impl.md) |
| Cut synchronization      | `allgatherv` receives in rank order (0, 1, ..., R-1)                        | [Communication Patterns §6.1](./communication-patterns.md)                   |
| Cut selection            | Deterministic algorithm on identical data (all ranks compute independently) | [Cut Management Implementation §6](../architecture/cut-management-impl.md)   |
| LP constraint order      | Cuts added by slot index (iteration order)                                  | [Solver Abstraction §5](../architecture/solver-abstraction.md)               |

### 3.3 Floating-Point Considerations

**Reductions**: `allreduce` with `ReduceOp::Sum` may produce different results depending on the reduction tree shape (non-associativity of floating-point addition). For convergence statistics this is acceptable — the upper bound is a statistical estimate. See [Communication Patterns §6.2](./communication-patterns.md).

**Intra-rank reductions**: Thread-local accumulation followed by single-threaded merge (see [Synchronization §3](./synchronization.md)) produces deterministic results because the merge order is fixed (thread 0, thread 1, ..., thread $N-1$). Rayon's `par_iter` with `collect()` or indexed `par_chunks()` preserves deterministic aggregation order, avoiding the non-determinism that would arise from unordered parallel reductions.

**Cut coefficient aggregation**: For single-cut formulation, per-opening results are aggregated within each thread (sequential — deterministic), then merged across threads (fixed order — deterministic), then exchanged via `allgatherv` (rank order — deterministic). The final averaging is performed locally by each rank on the full set of results (identical data — deterministic). No floating-point non-determinism is introduced at any step.

### 3.4 Verification

Reproducibility can be verified by running the same case with different rank/thread configurations and comparing:

| Quantity          | Comparison                 | Expected Result        |
| ----------------- | -------------------------- | ---------------------- |
| Lower bound trace | Bit-for-bit across configs | Identical              |
| Upper bound mean  | Within FP tolerance        | ~1e-12 relative error  |
| Final cut pool    | Bit-for-bit across configs | Identical coefficients |
| Policy output     | Bit-for-bit across configs | Identical              |

Upper bound mean may have small FP differences due to `allreduce` tree shape variation, but the lower bound and cut pool (which determine the policy) must be exact.

## 4. Performance Monitoring

### 4.1 Approved Timing Outputs

Cobre writes per-iteration and per-rank timing data to Parquet files during training. The schemas are defined in [Output Schemas §6.2-§6.3](../data-model/output-schemas.md):

**Per-iteration** (`training/timing/iterations.parquet`): forward solve, backward solve, cut computation, cut selection, MPI communication, I/O, overhead.

**Per-rank** (`training/timing/mpi_ranks.parquet`): forward time, backward time, communication time, idle time, LP solve count, scenarios processed.

### 4.2 Diagnostic Interpretation

| Symptom                                      | Metric to Check                         | Likely Cause                                    | Mitigation                                                       |
| -------------------------------------------- | --------------------------------------- | ----------------------------------------------- | ---------------------------------------------------------------- |
| Idle time >> 0 on some ranks                 | `idle_time_ms` variance across ranks    | Load imbalance                                  | Check LP iteration count variance; consider NUMA placement       |
| Communication time growing across iterations | `communication_time_ms` trend           | Cut count growth increases `allgatherv` payload | Enable cut selection to bound active cut count                   |
| Forward time >> backward time                | `forward_time_ms` vs `backward_time_ms` | Many forward passes, few trial points per rank  | Expected for high $M / R$ ratio                                  |
| Backward time growing across iterations      | `backward_solve_ms` trend               | More cuts in FCF → larger LPs → longer solves   | Enable cut selection; check for dominated cuts                   |
| Communication fraction > 10%                 | `communication_time_ms / total`         | Insufficient compute per rank                   | Reduce rank count (fewer, larger ranks); check network bandwidth |

### 4.3 Load Balance Assessment

The primary load balance indicator is the ratio of maximum to minimum `forward_time_ms` (or `backward_time_ms`) across ranks within an iteration. With static contiguous block distribution and homogeneous LP structure, this ratio should be close to 1.0:

| Imbalance Ratio | Assessment  | Action                                          |
| --------------- | ----------- | ----------------------------------------------- |
| < 1.1           | Excellent   | None needed                                     |
| 1.1 - 1.2       | Acceptable  | Monitor — may improve with NUMA tuning          |
| 1.2 - 1.5       | Notable     | Check NUMA placement, check LP iteration counts |
| > 1.5           | Significant | Investigate — likely NUMA or hardware issue     |

## Cross-References

- [Communication Patterns §5](./communication-patterns.md) — `SharedRegion<T>` capabilities and shared data candidates
- [Communication Patterns §6](./communication-patterns.md) — Deterministic communication, floating-point reduction
- [Communication Patterns §3](./communication-patterns.md) — Communication volume analysis, bandwidth requirements
- [Synchronization §1.1](./synchronization.md) — Three collective operations per iteration
- [Synchronization §3](./synchronization.md) — Thread-local cut accumulation pattern
- [Communicator Trait SS4](./communicator-trait.md) — SharedMemoryProvider trait, SharedRegion\<T\>, leader/follower pattern
- [Local Backend SS3](./backend-local.md) — HeapFallback implementation for backends without shared memory
- [Scenario Generation §2.2, §2.2b, §2.3, §2.3c](../architecture/scenario-generation.md) -- Deterministic seed derivation (§2.2), communication-free noise work distribution (§2.2b), fixed opening tree generation and memory layout (§2.3), parallel opening tree generation partitioning (§2.3c)
- [Cut Management Implementation §4](../architecture/cut-management-impl.md) — Wire format, deterministic slot assignment
- [Cut Management Implementation §7](../architecture/cut-management-impl.md) — StageLpCache update flow, MPI→CSC data path, parallel selection phase (SS7.1a), leader-only update phase (SS7.1b)
- [Cut Selection Strategy Trait SS2.2a](../architecture/cut-selection-trait.md) — Parallel work distribution for cut selection (distributed computation, leader-only SharedRegion write)
- [Synchronization §1.4a](./synchronization.md) — DeactivationSet allgatherv wire format and payload sizing
- [Work Distribution §3.1](./work-distribution.md) — Contiguous block distribution arithmetic
- [Solver Abstraction SS5](../architecture/solver-abstraction.md) — Cut pool preallocation, slot assignment
- [Solver Abstraction SS11.4](../architecture/solver-abstraction.md) — StageLpCache design, sizing, ownership, update/read contracts
- [Memory Architecture §2](./memory-architecture.md) — Memory budget, two-tier model (thread-local + SharedRegion)
- [Memory Architecture §3.6](./memory-architecture.md) — NUMA-interleaved allocation for SharedRegion
- [Output Schemas §6.2-§6.3](../data-model/output-schemas.md) — Timing output Parquet schemas
