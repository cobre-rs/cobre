# Cut Management Implementation

## Purpose

This spec defines how the mathematical cut management concepts from [Cut Management](../math/cut-management.md) are implemented in the Cobre architecture: the Future Cost Function (FCF) runtime structure, how cut selection strategies operate on the pre-allocated cut pool, cut serialization for checkpoint/resume via FlatBuffers, cross-rank cut synchronization via MPI, and cut activity tracking.

For the mathematical foundations (cut definition, dual extraction, aggregation, validity, convergence), see [Cut Management](../math/cut-management.md). For the cut pool memory layout requirements and FlatBuffers schema, see [Binary Formats SS3](../data-model/binary-formats.md). For how cuts enter the solver LP, see [Solver Abstraction SS5](./solver-abstraction.md).

## 1. Future Cost Function Structure

The FCF is the runtime data structure that holds the piecewise-linear outer approximation of the cost-to-go function. It consists of one cut pool per stage, each using the pre-allocated layout defined in [Binary Formats SS3.4](../data-model/binary-formats.md).

### 1.1 Per-Stage Cut Pool

Each stage's cut pool is a pre-allocated, fixed-capacity collection of Benders cuts. The pool is shared read-only across threads during the forward pass and written by a single thread per stage during the backward pass (with a synchronization barrier at each stage boundary — see [Training Loop SS6.3](./training-loop.md)).

| Component           | Description                                                                                                                                                                                         |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Coefficient storage | Contiguous dense `[f64]` arrays, one per cut, of length `state_dimension`. Stored in a flat buffer indexed by slot index for O(1) access. Cache-line aligned (64 bytes) for efficient CSR assembly. |
| Intercept array     | Separate contiguous `[f64]` array of intercepts (α values), one per slot. Separated from coefficients to avoid polluting cache during coefficient copy loops.                                       |
| Activity bitmap     | Bit array tracking which slots contain active cuts. An active count is maintained alongside the bitmap to avoid scanning.                                                                           |
| Populated count     | Number of slots that contain valid cuts (active or inactive). Slots beyond this count are empty (pre-allocated but unused).                                                                         |
| Cut metadata        | Per-slot metadata: iteration generated, forward pass index, last-active iteration (for LML1), cumulative active count (for Level-1 and dominated detection).                                        |

### 1.2 Deterministic Slot Assignment

Cut slots are computed, not allocated at runtime. The slot for a new cut is determined by a pure function:

```
slot = warm_start_count + iteration × forward_passes + forward_pass_index
```

This eliminates thread-safety concerns and non-determinism — the same inputs always produce the same slot. Combined with RNG state serialization, this enables bit-for-bit reproducible checkpoint/resume.

### 1.3 Capacity and Sizing

The pool is fully pre-allocated at initialization:

| Parameter              | Formula / Value                                               |
| ---------------------- | ------------------------------------------------------------- |
| Capacity per stage     | `warm_start_cuts + max_iterations × forward_passes`           |
| Production example     | 5,000 + 50 × 192 = 14,600 (rounded up to 15,000 for headroom) |
| Per-cut memory         | `state_dimension × 8` bytes (coefficients) + metadata         |
| Per-stage total (prod) | 15,000 × 2,080 × 8 ≈ 238 MB coefficients + ~1 MB metadata     |
| All stages (prod)      | 60 × 238 MB ≈ 14.3 GB per rank                                |

See [Binary Formats SS4.3](../data-model/binary-formats.md) for the full sizing breakdown.

> **StageLpCache note**: Under the adopted StageLpCache architecture ([Solver Abstraction SS11.4](./solver-abstraction.md)), the cut pool's coefficient storage is absorbed into the StageLpCache CSC. The cut pool retains metadata only: intercepts (`f64` per cut), activity bitmap, iteration/forward_pass indices, binding count history. Metadata total: ~12 MB across 60 stages at 15K capacity. Coefficient data continues to exist in the StageLpCache CSC and in the MPI wire format, but is not accessed from the cut pool during passes.

> **Precondition:** `forward_passes` is immutable after initialization. The cut pool
> capacity formula depends on this invariant — a runtime change to `forward_passes`
> would invalidate the pre-allocated slot assignment. A dynamic forward-passes scheduler
> is deferred (see [Deferred Features SSC.19](../deferred.md)).

### 1.4 FCF-Level Operations

The FCF provides the following operations to the training loop:

| Operation         | Description                                                                                                                              |
| ----------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| Add cut           | Write a new cut's coefficients, intercept, and metadata into the deterministic slot. Set the slot as active in the bitmap. O(1).         |
| Get active cuts   | Return the active cut indices and coefficient/intercept data for a given stage. Used by the solver workspace for `addRows` CSR assembly. |
| Evaluate at state | Compute max over all active cuts: $\max_k \{ \alpha_k + \pi_k^\top x \}$. Used for upper bound evaluation and dominated cut detection.   |
| Run selection     | Apply the configured cut selection strategy (SS2) to deactivate cuts. Updates the activity bitmap.                                       |
| Aggregate stats   | Return total cuts, active cuts per stage, cuts added this iteration — for logging and convergence monitoring.                            |

## 2. Cut Selection Strategies

Three cut selection strategies are implemented, corresponding to the three strategies defined in [Cut Management SS7](../math/cut-management.md). Selection runs periodically (every `check_frequency` iterations) and only deactivates cuts — it never deletes them, preserving slot indices for reproducibility.

### 2.1 Level-1

A cut is **Level-1 active** if it was binding at least once during the entire algorithm execution. At each selection check, cuts with a cumulative active count of zero are deactivated.

| Tracking data  | Description                                                                 |
| -------------- | --------------------------------------------------------------------------- |
| `active_count` | Per-cut counter, incremented each time the cut is binding at an LP solution |

This is the least aggressive strategy — it retains any cut that was ever useful. Convergence guarantee is preserved (see [Cut Management SS8](../math/cut-management.md)).

### 2.2 Limited Memory Level-1 (LML1)

Each cut is timestamped with the most recent iteration at which it was binding. At each selection check, cuts whose timestamp is older than `memory_window` iterations are deactivated.

| Tracking data      | Description                                                          |
| ------------------ | -------------------------------------------------------------------- |
| `last_active_iter` | Per-cut timestamp, updated each time the cut is binding              |
| `memory_window`    | Configurable parameter: number of iterations to retain inactive cuts |

More aggressive than Level-1 — forgets cuts that haven't been active recently. Convergence guarantee is preserved.

### 2.3 Dominated Cut Detection

A cut is dominated if at every visited state, some other cut achieves a higher value. At each selection check:

1. For each active cut $k$, evaluate $\alpha_k + \pi_k^\top \hat{x}$ at all visited states $\hat{x}$
2. For each visited state, compute the maximum value across all other active cuts
3. If cut $k$ is dominated at every visited state (within tolerance $\epsilon$), deactivate it

| Tracking data      | Description                                                                       |
| ------------------ | --------------------------------------------------------------------------------- |
| `domination_count` | Per-cut counter: number of visited states where this cut was dominated            |
| Visited states     | The set of trial points from forward passes, stored in the policy's `StageStates` |

This is the most aggressive strategy — directly identifies cuts providing no value at any known operating point. Computational cost is $\mathcal{O}(|\text{active cuts}| \times |\text{visited states}|)$ per stage per check.

### 2.4 Selection Configuration

| Parameter         | Description                                                 | Applies to |
| ----------------- | ----------------------------------------------------------- | ---------- |
| `method`          | Selection strategy: `"level1"`, `"lml1"`, or `"domination"` | All        |
| `threshold`       | Activity threshold $\epsilon$ (recommended: 0)              | All        |
| `check_frequency` | Iterations between selection runs                           | All        |
| `memory_window`   | Iterations to retain inactive cuts                          | LML1 only  |

See [Configuration Reference](../configuration/configuration-reference.md) for the JSON schema.

## 3. Cut Serialization

Cut persistence uses FlatBuffers, as defined in [Binary Formats SS3.1](../data-model/binary-formats.md). The serialization supports three execution modes with different reproducibility guarantees:

| Mode       | What is serialized                                   | What is restored                             | Reproducibility         |
| ---------- | ---------------------------------------------------- | -------------------------------------------- | ----------------------- |
| Fresh      | Nothing — training starts empty                      | —                                            | Deterministic from seed |
| Warm-start | All cuts (active + inactive), slot indices, metadata | Cut pool populated from policy; new RNG seed | Different from original |
| Resume     | All cuts + activity flags + RNG state + basis        | Exact state at checkpoint iteration          | Bit-for-bit identical   |

### 3.1 Checkpoint Write

At checkpoint boundaries (configured interval or graceful shutdown), the following data is serialized per stage:

| Data                         | FlatBuffers Table       | Rationale                                       |
| ---------------------------- | ----------------------- | ----------------------------------------------- |
| All cuts (active + inactive) | `StageCuts`             | LP row structure must be reconstructable        |
| Activity flags               | `BendersCut.is_active`  | Which cuts to load into solver on resume        |
| Slot indices                 | `BendersCut.slot_index` | Row mapping for reproducibility                 |
| Cut metadata                 | `BendersCut` fields     | Iteration, forward pass index, domination count |
| Visited states               | `StageStates`           | Needed for dominated cut detection on resume    |
| Solver basis                 | `StageBasis`            | Exact warm-start on resume                      |

**Write strategy**: Rank 0 serializes the shared cut pool to the policy directory ([Binary Formats SS3.2](../data-model/binary-formats.md)). Each stage is written as a separate `.bin` file. Compression (Zstd) is optional for archival.

### 3.2 Checkpoint Read (Resume / Warm-Start)

On resume, the FlatBuffers data is deserialized into the runtime cut pool layout:

1. Read the `StageCuts` FlatBuffer for each stage
2. Copy coefficient vectors from the FlatBuffers `[double]` arrays into the runtime contiguous layout (cache-line aligned)
3. Reconstruct the activity bitmap from `is_active` flags
4. Set `populated_count` and `warm_start_count`
5. Restore RNG state from `PolicyMetadata` (resume mode only)
6. Load cached basis from `StageBasis` into per-stage basis cache (see [Solver Workspaces SS1.2](./solver-workspaces.md))

The FlatBuffers zero-copy property means step 2 can read coefficient data directly from the memory-mapped file without parsing — but the data must still be copied into the NUMA-local, cache-aligned runtime layout for hot-path performance.

## 4. Cross-Rank Cut Synchronization

After each backward pass stage, new cuts generated by each MPI rank must be distributed to all other ranks so that every rank has the complete FCF for the next iteration's forward pass.

### 4.1 Synchronization Protocol

At each stage boundary during the backward pass:

1. **Count** — Each rank determines how many new cuts it generated at this stage
2. **Size exchange** — `MPI_Allgather` of cut counts (one integer per rank)
3. **Data exchange** — `MPI_Allgatherv` of serialized cut data (coefficients + intercepts + metadata)
4. **Integration** — Each rank deserializes received cuts and writes them into the local cut pool at the deterministic slot positions

### 4.2 Serialization for MPI

The MPI wire format is a compact binary representation optimized for bandwidth, not the full FlatBuffers format (which includes schema metadata overhead). For each cut:

| Field              | Type    | Size (production)        |
| ------------------ | ------- | ------------------------ |
| Slot index         | `u32`   | 4 bytes                  |
| Iteration          | `u32`   | 4 bytes                  |
| Forward pass index | `u32`   | 4 bytes                  |
| Intercept          | `f64`   | 8 bytes                  |
| Coefficients       | `[f64]` | 2,080 × 8 = 16,640 bytes |
| **Total per cut**  |         | **~16,660 bytes**        |

At production scale with 192 forward passes per iteration and 16 ranks, each rank generates 12 cuts per stage. The `MPI_Allgatherv` payload is ~192 cuts × 16,660 bytes ≈ 3.2 MB per stage — modest for InfiniBand interconnects.

### 4.2a Wire Format Specification

This subsection specifies the **exact byte-level layout** for cut data exchanged via `allgatherv` ([Communicator Trait SS2.1](../hpc/communicator-trait.md)) during the backward pass cut synchronization (SS4.1 step 3). It resolves the ambiguity in SS4.2 regarding endianness, alignment, and serialization strategy.

**Serialization: raw `#[repr(C)]` reinterpretation.** Cut records are transmitted as raw bytes obtained by reinterpreting a `#[repr(C)]` struct as `&[u8]` — **not** serialized via `postcard`, FlatBuffers, or any structured format. This is the same hot-path convention used for state vector exchange ([Training Loop SS5.4a](./training-loop.md)): per-iteration collective operations use raw reinterpretation for zero overhead. The `postcard` serialization path ([Input Loading Pipeline SS6](./input-loading-pipeline.md)) is reserved for initialization-time broadcast of heterogeneous structures.

**Endianness: native, with same-architecture assumption.** All ranks in an MPI job run the same compiled binary on nodes with the same CPU architecture. This is standard MPI practice — heterogeneous-endianness MPI jobs are not supported. No byte-swapping is performed. This assumption is shared with the state vector wire format ([Training Loop SS5.4a](./training-loop.md)).

**No version byte.** The wire format does not include a version tag, magic number, or schema identifier. All ranks run the same binary and therefore agree on the struct layout at compile time. Adding a version byte would waste one byte per cut record (multiplied by thousands of cuts per iteration across up to 119 stages at worst-case $T = 120$) for a scenario that cannot occur within a single MPI job.

**Struct layout.** Each cut is represented as a `CutWireRecord` with the following `#[repr(C)]` field ordering:

```rust
/// Wire format for a single cut transmitted via allgatherv.
///
/// The struct uses #[repr(C)] to guarantee field ordering and
/// platform-standard alignment. The explicit `_padding` field
/// ensures that the `intercept` field (and the variable-length
/// `coefficients` tail) are 8-byte aligned regardless of the
/// preceding u32 fields.
///
/// This type is NOT constructed directly — it is a layout
/// specification for raw byte reinterpretation.
#[repr(C)]
struct CutWireRecord {
    /// Deterministic slot index (SS1.2): warm_start_count + iteration * forward_passes + forward_pass_index
    slot_index: u32,         // offset  0, size 4
    /// Iteration at which this cut was generated
    iteration: u32,          // offset  4, size 4
    /// Forward pass index within the iteration
    forward_pass_index: u32, // offset  8, size 4
    /// Explicit padding to align intercept to 8-byte boundary
    _padding: u32,           // offset 12, size 4
    /// Cut intercept: α_k
    intercept: f64,          // offset 16, size 8
    /// Cut coefficients: β_k (variable-length, n_state elements)
    /// Declared as zero-length array — actual length is n_state.
    coefficients: [f64; 0],  // offset 24, size 0 (variable-length tail)
}
```

**Alignment.** The 4-byte `_padding` field between `forward_pass_index` (offset 12) and `intercept` (offset 16) ensures that `intercept` and all subsequent `coefficients` elements are 8-byte aligned. This is critical for correctness on architectures that require aligned `f64` loads (e.g., some ARM targets) and for performance on x86-64 (unaligned `f64` loads may cross cache line boundaries). The `#[repr(C)]` layout guarantee ensures no compiler-inserted padding beyond the explicit `_padding` field.

**Per-cut size.** The fixed header occupies 24 bytes (3 × `u32` + 1 × `u32` padding + 1 × `f64`). The variable-length coefficient tail adds $n_{state} \times 8$ bytes, where $n_{state} = N \cdot (1 + L)$ is the state dimension ([Solver Abstraction SS2.1](./solver-abstraction.md)):

$$\text{cut\_size} = 24 + n_{state} \times 8 \text{ bytes}$$

At production scale ($N = 160$ hydros, $L = 12$ AR lags, $n_{state} = 160 \times 13 = 2{,}080$):

| Component    | Size                                 |
| ------------ | ------------------------------------ |
| Fixed header | 24 bytes                             |
| Coefficients | $2{,}080 \times 8 = 16{,}640$ bytes  |
| **Total**    | **$24 + 16{,}640 = 16{,}664$ bytes** |

This corrects the approximate "~16,660 bytes" estimate in SS4.2 — the exact value is **16,664 bytes** per cut.

**allgatherv type parameter.** Because `CutWireRecord` has variable length (the coefficient tail depends on $n_{state}$, which is a runtime value), the `allgatherv` call uses `T = u8` (raw bytes) rather than `T = CutWireRecord`. Each rank serializes its local cuts into a contiguous `Vec<u8>` send buffer and provides byte counts and displacements to the collective operation:

| Parameter    | Formula                             | Description                              |
| ------------ | ----------------------------------- | ---------------------------------------- |
| `counts[r]`  | $M_r \times \text{cut\_size}$       | Byte count contributed by rank $r$       |
| `displs[r]`  | $\sum_{j=0}^{r-1} \text{counts}[j]$ | Byte offset where rank $r$'s data begins |
| `send.len()` | $M_r \times \text{cut\_size}$       | This rank's send buffer length in bytes  |
| `recv.len()` | $M \times \text{cut\_size}$         | Total receive buffer length in bytes     |

where $M_r$ is the number of forward passes assigned to rank $r$ ([Work Distribution SS3.1](../hpc/work-distribution.md)) and $M = \sum_{r=0}^{R-1} M_r$ is the total forward passes per iteration.

**Serialization (send).** Each rank constructs the send buffer by writing each local cut as a contiguous byte sequence:

1. Write the 24-byte header (`slot_index`, `iteration`, `forward_pass_index`, `_padding = 0`, `intercept`)
2. Copy $n_{state}$ `f64` coefficient values (16,640 bytes at production scale)
3. Advance write pointer by `cut_size` bytes
4. Repeat for each local cut at this stage

**Deserialization (receive).** Each rank reads received cuts by reinterpreting the byte buffer:

1. Read the 24-byte header from the current offset
2. Read $n_{state}$ `f64` values from offset + 24
3. Write the cut into the local cut pool at the slot position indicated by `slot_index`
4. Advance read pointer by `cut_size` bytes
5. Repeat for each received cut

Both serialization and deserialization are simple `memcpy`-equivalent operations — no parsing, no allocation, no schema lookup. The `unsafe` boundary for byte reinterpretation is encapsulated in a helper function that enforces the alignment and size invariants.

### 4.3 Deterministic Integration

Because slot indices are computed from `(iteration, forward_pass_index)` and forward passes are distributed to ranks in contiguous blocks, each rank can compute the correct slot position for every received cut without negotiation. All ranks end up with identical cut pools after synchronization.

**Invariant**: After `MPI_Allgatherv` at stage $t$, every rank has the same set of active cuts at stage $t$, in the same slot positions, with the same coefficients. This is necessary for the forward pass at the next iteration to produce identical results regardless of which rank evaluates which trajectory.

## 5. Cut Coefficient Extraction

Cut coefficient extraction requires no dual preprocessing or mapping. Each state variable (storage and inflow lag) has a dedicated fixing constraint in the LP, and the fixing constraint's dual gives the cut coefficient directly. The dual-extraction region contains exactly `n_state` rows — the first $N$ are storage fixing duals and the remaining $N \cdot L$ are lag fixing duals — so cut coefficient extraction is a single contiguous slice read: `cut_coefficients[0..n_state] = dual[0..n_state]`.

This design eliminates the need for precomputed dual-to-cut mappings for FPHA hyperplane duals and generic constraint duals. The LP solver resolves all downstream effects (water balance, FPHA, generic constraints) automatically through the fixing constraint dual, by the LP envelope theorem. See [Cut Management §2](../math/cut-management.md) for the mathematical derivation and [Solver Abstraction SS2.2](./solver-abstraction.md) for the row layout.

## 6. Cut Activity Tracking

After each LP solve, the training loop must update cut activity counters for the selection strategies to function. This happens in step 6 of the stage solve workflow ([Solver Workspaces SS1.4](./solver-workspaces.md)).

### 6.1 Binding Detection

A cut is binding if its dual multiplier (shadow price) in the LP solution is positive. After solving the LP:

1. Extract the dual values for the cut rows (the bottom region of the LP, per [Solver Abstraction SS2.2](./solver-abstraction.md))
2. For each cut row with a positive dual (above a configurable tolerance), mark the corresponding cut as active

### 6.2 Counter Updates

| Strategy  | Update per binding cut                                                 |
| --------- | ---------------------------------------------------------------------- |
| Level-1   | Increment `active_count` for the cut                                   |
| LML1      | Set `last_active_iter` to the current iteration                        |
| Dominated | Reset `domination_count` to 0 (the cut is not dominated at this state) |

These updates happen on the thread-local solver's cut rows. Since each thread processes its own trajectories and the backward pass has per-stage synchronization barriers, no locking is needed — each thread updates the activity data for the cuts it evaluated, and the results are reconciled during the cut synchronization step (SS4).

## 7. StageLpCache Update Flow

Between training iterations, the StageLpCache is updated to reflect newly generated cuts and any cuts deactivated by the selection strategy. The update has two phases: a distributed parallel selection phase where all ranks participate, followed by a leader-only StageLpCache write phase that preserves the single-writer SharedRegion contract.

### 7.1a Parallel Cut Selection Phase

> **Decision [DEC-016](../overview/decision-log.md#dec-016) (active):** Cut selection uses deferred parallel execution — stages distributed across ranks and threads, with DeactivationSet allgatherv and leader-only SharedRegion write.

When `should_run(iteration)` returns `true` ([Cut Selection Strategy Trait SS2.1](./cut-selection-trait.md)), the cut selection computation is distributed across ranks and threads:

1. **Stage partitioning** — Stages $\{2, \ldots, T\}$ are distributed across $R$ ranks via static contiguous block assignment ([Cut Selection Strategy Trait SS2.2a](./cut-selection-trait.md)). Each rank receives $\lceil (T-1) / R \rceil$ stages.
2. **Parallel selection** — Each rank runs `select` on its assigned stages, distributing stages across threads via Rayon work-stealing. Each `select` call reads the stage's cut pool metadata (identical across all ranks after the backward pass `allgatherv`) and produces a `DeactivationSet`.
3. **Result gathering** — `allgatherv` ([Communicator Trait SS2.1](../hpc/communicator-trait.md)) gathers the per-stage `DeactivationSet` payloads from all ranks. After this call, every rank has the complete set of deactivations across all stages. The wire format is specified in [Synchronization §1.4a](../hpc/synchronization.md).

On non-selection iterations (`should_run` returns `false`), this phase is skipped entirely — no partitioning, no communication, no selection computation.

### 7.1b StageLpCache Update Phase

After the parallel selection phase (or immediately after the backward pass on non-selection iterations), the leader rank updates the SharedRegion StageLpCache:

1. **New cut insertion** — For each new cut generated in the preceding backward pass (typically ~200 cuts × 59 stages):
   - Write the cut's coefficients into the pre-allocated CSC slot in `StageLpCache[t]`. The CSC slot positions are determined by the LP row layout: each cut occupies one row, with dense entries in the $n_{state}$ state columns plus the $\theta$ column (2,081 entries per cut).
   - Write the cut's intercept as the row lower bound.
2. **Cut deactivation** — For each cut in the `DeactivationSet` gathered in SS7.1a, set its row lower bound to $-\infty$ in the StageLpCache CSC. This makes the constraint non-binding without modifying the CSC structure (no row removal, no column recount).
3. **Fence and barrier** — `region.fence()` ensures all writes are visible, followed by an MPI barrier to synchronize all ranks before the next iteration's forward pass begins.

The single-writer SharedRegion contract ([Shared Memory Aggregation §1.2](../hpc/shared-memory-aggregation.md)) is preserved: only the leader rank writes to the SharedRegion, while all other ranks wait at the barrier.

### 7.2 MPI→StageLpCache Data Path

Cuts arrive from other ranks via `MPI_Allgatherv` (SS4.1) as `CutWireRecord` byte sequences. The deserialization path writes cut data into two destinations:

1. **Cut pool metadata** — Intercept, activity flag, iteration/forward_pass metadata, slot index. This is the permanent metadata store.
2. **StageLpCache CSC** — Coefficients are written directly into the pre-allocated CSC slot. This is the solver-facing representation consumed by `passModel`/`loadProblem`.

The coefficient write to StageLpCache ensures the per-stage CSC representation stays current as new cuts are generated. Coefficients are written to both the cut pool metadata (intercepts, activity bitmap) and the StageLpCache CSC column data.

### 7.3 Performance

**Selection compute.** The selection phase distributes $T-1$ stages across $R$ ranks. Each stage's `select` call is dominated by the cut pool scan (Level1/LML1: linear in `populated_count`; Dominated: quadratic in active cuts × visited states). At production scale with 16 ranks, each rank processes ~4 stages. The selection compute time scales as $\sim 5 \text{ ms} / N_{\text{ranks}}$ — effectively < 1 ms per rank for the Level1/LML1 variants.

**DeactivationSet allgatherv payload.** Each `DeactivationSet` contains a `stage_index` (4 bytes) + `count` (4 bytes) + deactivation indices (4 bytes each). At production scale with ~200 deactivations per stage across 59 stages:

| Component              | Size                          |
| ---------------------- | ----------------------------- |
| Per-stage header       | 8 bytes (stage_index + count) |
| Per-stage indices      | ~200 × 4 = 800 bytes          |
| Total across 59 stages | 59 × 808 ≈ **~47 KB**         |

This is negligible relative to the backward pass cut `allgatherv` payload (~3.2 MB per stage). Worst case (all 15K cuts deactivated at all stages): 59 × (8 + 15,000 × 4) ≈ **~3.4 MB** — still modest.

**New cut insertion.** Per-iteration cost: ~200 new cuts × 59 stages × 2,081 entries × 12 bytes (4 row_index + 8 value) ≈ 300 MB written. At ~60 GB/s DRAM write bandwidth, this takes ~5 ms.

**Total between-iterations update.** Selection compute (< 1 ms per rank) + DeactivationSet `allgatherv` (< 0.1 ms) + StageLpCache write (~5 ms) + `fence()` + barrier (< 0.5 ms) = **~5–7 ms**, which is < 0.1% of the per-iteration compute time (~16.5 s).

## Cross-References

- [Cut Management (Math)](../math/cut-management.md) — Cut definition, dual extraction, aggregation, validity, selection theory, convergence guarantee
- [Cut Selection Strategy Trait](./cut-selection-trait.md) — Parallel calling convention (SS2.2, SS2.2a), stage partitioning formula, conditional execution
- [Solver Abstraction](./solver-abstraction.md) — Cut pool design (SS5), LP layout convention (SS2), how active cuts enter the solver LP (SS5.4), StageLpCache design (SS11.4)
- [Solver Workspaces](./solver-workspaces.md) — Stage solve workflow (SS1.4) where cuts are loaded and activity is tracked, cut loading cost analysis (SS1.10)
- [Shared Memory Aggregation](../hpc/shared-memory-aggregation.md) — SharedRegion for StageLpCache allocation and NUMA-interleaved placement; single-writer contract (§1.2)
- [Binary Formats](../data-model/binary-formats.md) — FlatBuffers schema (SS3.1), policy directory structure (SS3.2), cut pool memory layout requirements (SS3.4), checkpoint reproducibility (SS4)
- [Training Loop](./training-loop.md) — Forward/backward pass that drives cut generation (SS6), dual extraction for cut coefficients (SS7), cut selection step in iteration lifecycle (SS2.1 step 4a)
- [Convergence Monitoring](./convergence-monitoring.md) — Uses FCF statistics (lower bound from stage 1 cuts)
- [Risk Measures](../math/risk-measures.md) — CVaR modifies aggregation weights during cut generation
- [LP Formulation](../math/lp-formulation.md) — Storage fixing constraints (§4a) that produce cut coefficients directly; generic constraints (§10) whose effects are captured via fixing constraint duals
- [Scenario Generation](./scenario-generation.md) — Fixed opening tree (SS2.3) that defines backward pass branchings
- [Hybrid Parallelism](../hpc/hybrid-parallelism.md) — MPI rank topology for cut synchronization
- [Synchronization](../hpc/synchronization.md) — DeactivationSet allgatherv wire format (§1.4a), per-stage backward pass barrier (§1.4)
- [Communicator Trait](../hpc/communicator-trait.md) — allgatherv contract (SS2.1) for DeactivationSet gathering
- [Work Distribution](../hpc/work-distribution.md) — Contiguous block assignment formula (§3.1) for stage partitioning
- [Checkpointing](../hpc/checkpointing.md) — Checkpoint trigger logic and graceful shutdown that invoke cut serialization
- [Configuration Reference](../configuration/configuration-reference.md) — `cut_selection` JSON schema parameters
