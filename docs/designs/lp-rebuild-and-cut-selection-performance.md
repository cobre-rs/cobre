# Design: LP Rebuild and Cut Selection Performance

## Status

Discussion — 2026-03-23

## Context

As the SDDP solver moves toward production-scale benchmarks (100+ hydros, 120
stages, 2000 scenarios, 300+ iterations), the cost of LP preparation becomes a
significant fraction of total training time. The current architecture rebuilds
the LP from scratch for every `(scenario, stage)` solve: load the static
template via `load_model`, then re-append all active Benders cuts via
`add_rows`. This is correct but increasingly expensive as the cut pool grows.

This document catalogs the bottlenecks identified during code review, proposes
mitigation strategies, and outlines the instrumentation needed to measure them.

---

## 1. Current LP Solve Lifecycle

Each `(scenario, stage)` solve in the forward and backward passes follows:

```
1. solver.load_model(&template)        — FFI: full structural LP reset (CSC bulk load)
2. solver.add_rows(&cut_batch)         — FFI: append active Benders cuts (CSR)
3. patch_buf.fill_*_patches(...)       — Rust: pre-allocated buffer fills
4. solver.set_row_bounds(...)          — FFI: patch scenario-specific row bounds
5. solver.set_col_bounds(...)          — FFI: patch NCS + terminal theta bounds
6. solver.solve_with_basis(...)        — FFI: simplex solve with warm-start
```

`build_cut_row_batch()` is called once per stage per iteration (outside the
scenario loop) to assemble the CSR `RowBatch` from all active cuts.

### Why rebuild every scenario?

The static template (structural constraints) is identical across all scenarios at
a stage. Only bounds change per scenario. However, HiGHS's `pass_lp` replaces
the entire model — there is no "keep the matrix, reset bounds only" API. Since
the number of active cuts can change between iterations (after cut selection),
the LP row count is not constant, requiring a full rebuild.

---

## 2. Identified Bottlenecks

### B1: Repeated `load_model` + `add_rows` per scenario

**Impact:** O(scenarios × stages × iterations) FFI calls, each transmitting the
full structural LP and all active cut rows.

For a production run (200 scenarios × 120 stages × 300 iterations):

- Forward pass: 7.2M `load_model` + `add_rows` calls per rank
- Backward pass: same order of magnitude
- At 0.5ms per pair → ~2 hours of pure LP setup overhead

**Root cause:** No model persistence between scenarios at the same stage.

### B2: `add_rows` cost grows with cut count

Each `add_rows` call transmits `active_cuts × (n_state + 1)` non-zero entries
through FFI. By iteration 300 with 1000+ active cuts and `n_state = 50`:

- 51K NNZ per call
- Repeated for every scenario at every stage

**Root cause:** All active cuts are re-appended from scratch rather than
incrementally.

### B3: `build_cut_row_batch` allocates every iteration

Called once per stage per iteration, but allocates fresh `Vec`s for
`row_starts`, `col_indices`, `values`, `row_lower`, `row_upper`. For 120
stages × 300 iterations = 36K allocations, each potentially 50K+ elements.

**Root cause:** No buffer reuse across iterations.

### B4: No incremental cut addition

When 1–2 new cuts are added per stage per iteration, the system rebuilds the
entire `RowBatch` from all active cuts (potentially 1000+) rather than appending
just the new ones.

**Root cause:** `build_cut_row_batch` iterates the full pool each time.

### B5: Active cut counting requires full pool scan

`fcf.active_cuts(stage).count()` at the start of `build_cut_row_batch` iterates
the entire pool (including inactive slots) to count active entries for
`Vec::with_capacity`. A cached `active_count` field would eliminate this O(pool_size)
scan.

**Root cause:** No cached active count in `CutPool`.

---

## 3. Mitigation Strategies

### S1: Model persistence across scenarios (eliminate B1)

**Idea:** At the start of each stage's scenario loop, load the model and add
cuts once. For subsequent scenarios, only patch bounds via `set_row_bounds` and
`set_col_bounds`.

**Requirements:**

- `set_row_bounds` must be able to patch ALL rows (template + cut rows)
- Need a way to "reset" bounds to template defaults between scenarios, or
  patch all scenario-dependent rows each time (which we already do)

**HiGHS feasibility:** `Highs_changeRowBoundsBySet` can target any row index,
including dynamically added rows. The patch buffer already tracks all
scenario-dependent rows. The question is whether we can patch the
state-fixing rows (Categories 1-2), noise rows (Category 3), load rows
(Category 4), and z-inflow rows (Category 5) without touching the structural
matrix.

**Expected gain:** Eliminates ~(S-1)/S of all `load_model` + `add_rows` calls.
For S=200 scenarios, that's 99.5% reduction in LP rebuild cost.

**Considerations:**

- The backward pass already reloads per trial point (not per opening), so it
  partially does this — openings within a trial point share the same loaded model.
- Basis management: warm-starting from a previous scenario's basis at the same
  stage may actually improve convergence since the LP structure is identical.
- Need to verify that HiGHS internal state (presolve cache, factorization) is
  properly invalidated by bound changes alone.

### S2: Incremental cut management (eliminate B2 + B4)

**Idea:** Instead of rebuild-from-scratch, maintain a persistent LP across
iterations and:

- After backward pass: append only new cuts (1–2 per stage) via `add_rows`
- After cut selection: remove deactivated cuts (if HiGHS supports efficient row
  deletion)

**HiGHS feasibility:** HiGHS has `Highs_deleteRows` which accepts a sorted
index set. If row deletion is O(deleted_rows) rather than O(total_rows), this
is viable. Need to benchmark.

**Alternative:** If row deletion is expensive, use "bound zeroing" — set
deactivated cut bounds to `(-inf, +inf)`, effectively making them non-binding
without changing the matrix structure. This avoids deletion entirely.

**Expected gain:** Reduces `add_rows` from O(all_active_cuts × n_state) to
O(new_cuts × n_state) per iteration. For 1–2 new cuts vs 1000 active, that's
500–1000× reduction.

**Considerations:**

- Row indices shift after deletion. Need a mapping layer between pool slots
  and LP row indices.
- "Bound zeroing" is simpler but grows the LP over time (deactivated cuts
  remain as slack rows). May affect simplex performance.
- Incremental addition is compatible with S1 (model persistence).

### S3: `RowBatch` buffer reuse (eliminate B3)

**Idea:** Pre-allocate `RowBatch` buffers with max capacity once and reuse
across iterations. Clear and refill instead of allocate and drop.

**Implementation:**

- Add `clear()` + `push_cut()` methods to a reusable `CutRowBuilder`
- Allocate with `capacity = max_iterations * forward_passes * (n_state + 1)`
- Reuse across `build_cut_row_batch` calls

**Expected gain:** Eliminates 36K heap allocations per training run. Minor
compared to B1/B2 but improves cache locality and reduces allocator pressure.

### S4: Sparse cut coefficients (reduce B2 NNZ)

**Idea:** Many cut coefficients are near-zero (especially for hydros not
hydraulically connected to the trial point's binding constraints). Store only
non-zero coefficients.

**Implementation:**

- Change `CutPool::coefficients` from `Vec<Vec<f64>>` (dense) to sparse
  representation (indices + values)
- `build_cut_row_batch` emits only non-zero entries

**Expected gain:** For systems with 100 hydros and sparse cascade topology,
70–90% of cut coefficients may be near-zero. NNZ reduction from 50K to 5–15K
per `add_rows` call.

**Considerations:**

- Needs a threshold for "near-zero" (e.g., `|c| < 1e-12`)
- Changes the cut wire format for MPI exchange
- Must preserve determinism (canonical ordering of non-zero entries)
- Evaluation at `evaluate_at_state` must handle sparse dot product

### S5: Cut pool compaction (reduce B5 + improve cache)

**Idea:** After deactivation, compact active cuts into a contiguous sub-array.
Cache `active_count` to avoid full pool scans.

**Implementation:**

- Add `active_count: usize` field to `CutPool`, updated on
  `add_cut`/`deactivate`
- Optionally maintain a contiguous `active_indices: Vec<usize>` for fast
  iteration
- `build_cut_row_batch` iterates `active_indices` instead of scanning the
  full pool

**Expected gain:** Eliminates O(pool_size) scan per `build_cut_row_batch`.
Improves cache locality for large pools with many inactive cuts.

---

## 4. Instrumentation Gaps

The current timing infrastructure measures phases coarsely but does not separate
LP setup from LP solve within each phase.

### What IS measured today

| Metric                   | Granularity            | Output                      |
| ------------------------ | ---------------------- | --------------------------- |
| Forward pass elapsed     | Per iteration          | `convergence.parquet`       |
| Backward pass elapsed    | Per iteration          | `convergence.parquet`       |
| LP solve time            | Per phase (fwd/bwd/lb) | `solver/iterations.parquet` |
| Simplex iterations       | Per phase              | `solver/iterations.parquet` |
| Basis offer/rejection    | Per phase              | `solver/iterations.parquet` |
| Cut selection time       | Per iteration          | `convergence.parquet`       |
| Forward sync (allreduce) | Per iteration          | `convergence.parquet`       |
| Iteration total time     | Per iteration          | `convergence.parquet`       |

### What is NOT measured (gaps)

| Missing metric             | Why it matters                                       |
| -------------------------- | ---------------------------------------------------- |
| `load_model` time          | Dominant in B1 — need to quantify template load cost |
| `add_rows` time            | Dominant in B2 — need to see scaling with cut count  |
| `build_cut_row_batch` time | B3/B4 — CSR assembly cost per stage                  |
| `set_row_bounds` time      | Quantify patch cost (should be small)                |
| `set_col_bounds` time      | Quantify NCS patch cost                              |
| LP setup vs solve split    | Critical: is the bottleneck in setup or simplex?     |
| Per-stage forward timing   | Currently only aggregate — need per-stage breakdown  |
| Cut sync allgatherv time   | Placeholder 0 in output                              |

### Existing Parquet schema placeholder fields

The `IterationRecord` schema already has columns for finer-grained timing that
are currently hardcoded to zero:

- `time_forward_sample_ms` — always 0
- `time_backward_cut_ms` — always 0
- `time_mpi_broadcast_ms` — always 0
- `time_io_write_ms` — always 0
- `time_overhead_ms` — computed as residual (total − attributed)

The `time_overhead_ms` residual currently absorbs all LP setup cost, cut
assembly, MPI state exchange, and other unattributed work. Breaking this down
is essential for identifying the actual bottleneck.

### Proposed instrumentation

To validate which bottleneck dominates before implementing mitigations:

**Phase 1: Split LP setup from solve in SolverStatistics**

Add to `SolverStatistics`:

```
total_load_model_time_seconds: f64
total_add_rows_time_seconds: f64
total_set_bounds_time_seconds: f64
```

This gives us the setup-vs-solve split per phase without changing the Parquet
schema (delta computation already works).

**Phase 2: Add `build_cut_row_batch` timing**

Capture CSR assembly time per stage per iteration. Report as part of the
backward/forward phase breakdown.

**Phase 3: Per-stage forward pass timing**

Currently only the backward pass reports per-stage `SolverStatsDelta`. Adding
the same for the forward pass would reveal stage-specific bottlenecks (e.g.,
stages with more active cuts taking disproportionately longer).

---

## 5. Implementation Priority

Based on expected impact and implementation complexity:

| Priority | Strategy                  | Expected gain        | Complexity  | Dependencies       |
| -------- | ------------------------- | -------------------- | ----------- | ------------------ |
| **1**    | S5: Cache active count    | Minor but free       | Trivial     | None               |
| **2**    | Instrumentation Phase 1   | Enables measurement  | Low         | None               |
| **3**    | S1: Model persistence     | 100× fewer rebuilds  | Medium      | HiGHS verification |
| **4**    | S3: RowBatch buffer reuse | Allocation reduction | Low         | None               |
| **5**    | S2: Incremental cuts      | 500× less cut data   | Medium-High | S1, HiGHS row API  |
| **6**    | S4: Sparse coefficients   | NNZ reduction        | High        | Wire format change |

The critical first step is **instrumentation** — we need to measure the
actual split between LP setup and LP solve before committing to any
mitigation strategy. The `time_overhead_ms` residual in the current output
suggests significant unattributed time, but we cannot tell whether it's
dominated by `load_model`, `add_rows`, `build_cut_row_batch`, or MPI
communication without proper measurement.

---

## 6. Related Specifications

- **Solver Abstraction SS2**: LP column/row layout conventions
- **Solver Interface SS4**: `SolverInterface` trait contract
- **Training Loop SS4**: Forward/backward pass structure
- **DEC-001**: Enum dispatch for closed variant sets
- **DEC-002**: Compile-time monomorphization for `SolverInterface`
