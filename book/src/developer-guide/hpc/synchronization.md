# Synchronization

## Purpose

This spec defines the synchronization architecture for Cobre: the complete set of communication synchronization points through the `Communicator` trait ([Communicator Trait SS1](./communicator-trait.md)) during SDDP iterations, the forward-to-backward transition, per-stage barrier semantics in the backward pass, and thread coordination within a rank. This spec details the synchronization mechanics that [Training Loop](../architecture/training-loop.md) and [Work Distribution](./work-distribution.md) describe at the algorithmic and distribution levels.

## 1. Communication Synchronization Points

### 1.1 Synchronization Summary

The following table lists all communication synchronization points in a single SDDP iteration. On normal iterations there are four types of collective calls: one post-forward `allgatherv` for visited states, one `allgatherv` per backward stage ($T-1$ calls total), one post-forward `allreduce` for UB statistics, and one post-backward `broadcast` for the LB — totaling $T+2$ collective operations. On selection iterations (when `should_run(iteration)` returns `true`), an additional `allgatherv` gathers DeactivationSets from the parallel cut selection phase, bringing the total to $T+3$ collective operations. For method contracts and determinism guarantees, see [Communicator Trait SS2](./communicator-trait.md).

| Phase                 | Operation    | Data Exchanged                                              | Direction       | Frequency                        |
| --------------------- | ------------ | ----------------------------------------------------------- | --------------- | -------------------------------- |
| Forward → Backward    | `allgatherv` | Visited states (trial points) from all forward trajectories | All ranks ↔ all | Every iteration                  |
| Post-forward          | `allreduce`  | UB statistics (3 scalars — see §1.3)                        | All ranks ↔ all | Every iteration                  |
| Backward stage $t$    | `allgatherv` | New cuts generated at stage $t$ by each rank                | All ranks ↔ all | Every iteration ($T-1$ per iter) |
| Cut selection (§1.4a) | `allgatherv` | DeactivationSets from parallel selection across stages      | All ranks ↔ all | Selection iterations only        |
| Post-backward         | `broadcast`  | Lower bound (1 scalar — rank 0 root)                        | Rank 0 → all    | Every iteration                  |

### 1.2 Forward Pass: No Per-Stage Synchronization

The forward pass has **no per-stage synchronization barrier**. Each thread solves its assigned trajectories independently from stage $t = 1$ to $T$. No MPI communication occurs during the forward pass itself — each rank processes its contiguous block of scenarios without coordinating with other ranks.

### 1.3 Forward-to-Backward Transition

After all forward trajectories complete within a rank, the rank contributes its visited states to an `allgatherv` call. After this call, every rank has the complete set of trial points from all forward trajectories across all ranks. This is the **only** synchronization point between the forward and backward passes.

The post-forward `allreduce` aggregates upper bound statistics:

| Quantity                            | Reduction       | Purpose                          |
| ----------------------------------- | --------------- | -------------------------------- |
| Total forward cost (sum)            | `ReduceOp::Sum` | Upper bound mean computation     |
| Total forward cost (sum of squares) | `ReduceOp::Sum` | Upper bound variance computation |
| Trajectory count                    | `ReduceOp::Sum` | Denominator for mean/variance    |

The lower bound is evaluated separately after the backward pass: rank 0 solves all stage-0 openings with the latest FCF cuts, applies the risk measure, and broadcasts the scalar result to all ranks via `comm.broadcast()`. See [Training Loop SS4.3b](../architecture/training-loop.md) and [Convergence Monitoring SS3.2](../architecture/convergence-monitoring.md).

### 1.4 Backward Pass: Per-Stage Barrier

The backward pass has a **hard synchronization barrier at each stage boundary**. At each stage $t$ (walking from $T$ down to 2):

1. All ranks evaluate their assigned trial points and generate cuts (parallel across ranks and threads)
2. `allgatherv` collects all new cuts from all ranks — after this call, every rank has the complete set of new cuts for stage $t$
3. All ranks add the new cuts to stage $t-1$'s cut pool
4. Only then do ranks proceed to stage $t-1$

The `allgatherv` acts as an implicit barrier — no rank can proceed to stage $t-1$ until all ranks have contributed their cuts for stage $t$. This barrier is mandatory because the cuts generated at stage $t$ must be available when solving backward LPs at stage $t-1$ (the backward pass uses $V_{t+1}^k$, the current iteration's approximation). See [Training Loop §6.3](../architecture/training-loop.md) and [Work Distribution §2.2](./work-distribution.md).

### 1.4a Cut Selection Synchronization

> **Decision [DEC-016](../overview/decision-log.md#dec-016) (active):** Cut selection uses deferred parallel execution — stages distributed across ranks and threads, with DeactivationSet allgatherv and leader-only SharedRegion write.

On selection iterations (when `should_run(iteration)` returns `true` — see [Cut Selection Strategy Trait SS2.1](../architecture/cut-selection-trait.md)), an additional `allgatherv` gathers the `DeactivationSet` results from the parallel cut selection phase. This occurs between the backward pass (§1.4) and the convergence check (§1.5), as step 4a in the iteration lifecycle ([Training Loop SS2.1](../architecture/training-loop.md)).

**Context.** After the backward pass completes, every rank holds identical cut pool data (metadata, activity bitmap, visited states) because the per-stage `allgatherv` in §1.4 ensures all ranks have all cuts. The cut selection computation distributes stages across ranks ([Cut Selection Strategy Trait SS2.2a](../architecture/cut-selection-trait.md)) — each rank runs `select` on its assigned stages and produces `DeactivationSet` results for those stages only. The `allgatherv` in this subsection gathers these partial results so that every rank (and specifically the leader rank, which writes to the SharedRegion) has the complete deactivation picture.

**Wire format.** The `DeactivationSet` payload for each stage uses a compact binary format:

```
Per stage: stage_index (u32) | count (u32) | indices ([u32; count])
```

| Field         | Type       | Size              | Description                              |
| ------------- | ---------- | ----------------- | ---------------------------------------- |
| `stage_index` | `u32`      | 4 bytes           | Stage number (2..T)                      |
| `count`       | `u32`      | 4 bytes           | Number of deactivated cuts at this stage |
| `indices`     | `[u32; N]` | `count × 4` bytes | Slot indices of deactivated cuts         |

Each rank's send buffer contains one record per assigned stage, concatenated. Ranks with zero assigned stages contribute a zero-length buffer.

**Payload sizing.**

| Scenario                                                      | Payload                                                |
| ------------------------------------------------------------- | ------------------------------------------------------ |
| Typical (200 deactivations/stage, 59 stages, 16 ranks)        | 59 × (8 + 200 × 4) ≈ **~47 KB** total across all ranks |
| Worst case (15K deactivations/stage, 59 stages)               | 59 × (8 + 15,000 × 4) ≈ **~3.4 MB** total              |
| Best case (0 deactivations, selection runs but finds nothing) | 59 × 8 = **472 bytes** (headers only)                  |

This is negligible relative to the backward pass cut `allgatherv` payload (~3.2 MB per stage × 59 stages ≈ 189 MB total).

**Serialization convention.** Like the backward pass cut wire format ([Cut Management Implementation SS4.2a](../architecture/cut-management-impl.md)), the DeactivationSet payload uses raw `#[repr(C)]` byte reinterpretation — no structured serialization, no version byte, native endianness. The `allgatherv` type parameter is `T = u8` (raw bytes) with per-rank byte counts and displacements.

**Post-gather sequence.** After the `allgatherv`, the leader rank applies all deactivations to the SharedRegion StageLpCache ([Cut Management Implementation SS7.1b](../architecture/cut-management-impl.md)), then `fence()` + barrier before the next forward pass.

### 1.5 Iteration Boundary

No explicit communication synchronization is required between iterations. The convergence monitor evaluates stopping rules locally on each rank using the aggregated statistics from §1.3. All ranks reach the same termination decision deterministically (same data, same rules).

An optional checkpoint barrier (`barrier`) may occur every $N$ iterations if checkpointing is enabled — see [Checkpointing](./checkpointing.md).

## 2. Thread Coordination Within Rank

Thread coordination within a rank uses Rayon's implicit synchronization, described in [Hybrid Parallelism §5](./hybrid-parallelism.md).

### 2.1 Forward Pass Thread Coordination

No explicit thread synchronization during the forward pass. Each thread owns complete trajectories (thread-trajectory affinity) and solves them independently. Rayon's `par_iter_mut` completion ensures all threads have finished before the rank proceeds to the `allgatherv` for trial point collection.

### 2.2 Backward Pass Thread Coordination

At each backward stage $t$, threads within a rank coordinate in two phases:

**Phase 1 — Parallel evaluation**: Each thread evaluates its assigned trial points, solving all openings sequentially per trial point. Each thread accumulates its generated cuts in a thread-local buffer (no shared writes, no contention). See §3.

**Phase 2 — Collection and communication**: The Rayon parallel iterator completes (implicit join ensures all threads have finished). The rank's main thread collects cuts from all thread-local buffers and participates in the inter-rank `allgatherv`.

This two-phase pattern repeats for each stage from $T$ down to 2.

### 2.3 Synchronization Primitives

| Primitive                      | Usage                                                    | Source                                                            |
| ------------------------------ | -------------------------------------------------------- | ----------------------------------------------------------------- |
| Rayon `par_iter` completion    | End of parallel iteration — ensures all threads complete | Implicit when `par_iter`/`par_iter_mut` returns                   |
| Rayon `join()` barrier         | Fork-join synchronization point                          | Implicit when `rayon::join()` returns                             |
| Rust ownership + `Send`/`Sync` | Per-thread cut buffers, solver workspaces                | `rayon::current_thread_index()` indexing into pre-allocated array |

The approved architecture does not require custom spin barriers or lock-free data structures for thread synchronization. Rayon's implicit synchronization at parallel iterator completion provides the necessary synchronization point between the parallel evaluation phase and the single-threaded MPI collection phase. Rust's ownership model and `Send`/`Sync` trait bounds enforce thread safety at compile time.

## 3. Cut Accumulation Pattern

### 3.1 Thread-Local Accumulation

During the backward pass parallel evaluation phase, each thread accumulates cuts in its own buffer. The buffers are indexed by Rayon thread index and pre-allocated at initialization. This design has zero contention during the hot loop — each thread writes exclusively to its own buffer.

| Property          | Value                                                                                 |
| ----------------- | ------------------------------------------------------------------------------------- |
| Buffer allocation | Pre-allocated per thread at initialization                                            |
| Buffer indexing   | Rayon thread index (`rayon::current_thread_index()`)                                  |
| Contention        | None — pure thread-local writes                                                       |
| Cache alignment   | Each buffer starts on a cache line boundary (64 bytes) to prevent false sharing       |
| Capacity          | Sized for expected cuts per thread per stage ($\lceil M / N_{\text{threads}} \rceil$) |

### 3.2 Collection and Merge

After the Rayon parallel iterator completes (implicit join), the rank's main thread collects all cuts from the per-thread buffers into a single contiguous array. This array is then used as the send buffer for `allgatherv`.

### 3.3 False Sharing Prevention

Adjacent thread buffers must not share cache lines. If two threads' buffers share a cache line, a write by one thread invalidates the cache line for the other, causing expensive cache coherency traffic. Pre-allocating each buffer at a cache-line-aligned address (64-byte boundary) eliminates this. See [Memory Architecture](./memory-architecture.md) for cache-line alignment conventions.

## 4. Asymmetry Summary

| Property                  | Forward Pass                             | Backward Pass                                                |
| ------------------------- | ---------------------------------------- | ------------------------------------------------------------ |
| Per-stage synchronization | None                                     | `allgatherv` at every stage                                  |
| Thread coordination       | None (independent trajectories)          | Implicit Rayon join between evaluation and collection        |
| Data flow direction       | Each thread accumulates independently    | Thread-local → rank-local merge → inter-rank `allgatherv`    |
| Parallelism grain         | Trajectory (coarse)                      | Trial point (coarse), openings sequential within trial point |
| Post-phase aggregation    | `allgatherv` (states) + `allreduce` (UB) | `broadcast` (LB from rank 0) + convergence check             |

## Cross-References

- [Training Loop §4.3](../architecture/training-loop.md) — Forward pass parallel distribution, post-forward `allreduce`
- [Training Loop §5.2](../architecture/training-loop.md) — State extraction and `allgatherv` for trial point collection
- [Training Loop §6.3](../architecture/training-loop.md) — Backward pass stage synchronization barrier, `allgatherv` for cuts
- [Training Loop §6.4a](../architecture/training-loop.md) — Cut selection step in iteration lifecycle (step 4a)
- [SDDP Algorithm §3.4](../math/sddp-algorithm.md) — Backward pass hard barrier, forward pass no barrier, thread-trajectory affinity
- [Cut Selection Strategy Trait SS2.2a](../architecture/cut-selection-trait.md) — Parallel work distribution for cut selection, stage partitioning formula
- [Cut Management Implementation SS7.1a-SS7.1b](../architecture/cut-management-impl.md) — Parallel selection phase and StageLpCache update phase
- [Work Distribution §1.4](./work-distribution.md) — Post-forward `allreduce` with 4 convergence quantities
- [Work Distribution §2.2](./work-distribution.md) — Per-stage backward pass execution (6 steps including barrier)
- [Hybrid Parallelism §5](./hybrid-parallelism.md) — Rayon parallel patterns, implicit synchronization
- [Convergence Monitoring](../architecture/convergence-monitoring.md) — Stopping rule evaluation using aggregated statistics
- [Communication Patterns](./communication-patterns.md) — collective operations via Communicator trait: `allgatherv`, `allreduce`
- [Communicator Trait SS2](./communicator-trait.md) — Method contracts for allgatherv, allreduce, barrier
- [Shared Memory Aggregation §1.2](./shared-memory-aggregation.md) — SharedRegion StageLpCache write model, single-writer contract
- [Memory Architecture](./memory-architecture.md) — Cache-line alignment conventions, NUMA-local allocation
