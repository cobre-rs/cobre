# Backward Pass Performance & Correctness — Implementation Spec

**Date**: 2026-03-24
**Status**: Draft
**Derived from**: [ONS teste analysis report](../2026-03-24-ons-teste-analysis.md)
**Affected crates**: `cobre-sddp`, `cobre-solver`, `cobre-core`

---

## 1. Problem Statement

The ONS teste case (158 hydros, 104 thermals, 118 stages, 1,106-dimensional state)
reveals that the backward pass is the dominant bottleneck in training. Cut accumulation
without pruning causes superlinear degradation: LP solve time grows 6.4x over 5
iterations, cost-per-simplex-pivot grows 3.9x, and extrapolation to 100 iterations
projects ~2,000 hours of cumulative training time.

A correctness bug (cut sync placement) additionally prevents valid multi-rank MPI runs.

This spec defines six implementation phases to address correctness, instrumentation,
and performance, ordered by dependency and impact.

---

## 2. Phase Overview

| Phase | Scope                                 | Items            | Expected Impact        | Prerequisite             |
| ----- | ------------------------------------- | ---------------- | ---------------------- | ------------------------ |
| 0     | Instrumentation + trivial correctness | BUG-2, INST-1..4 | Baseline measurement   | None                     |
| 1     | Backward pass structural fix          | BUG-1            | Multi-rank correctness | None                     |
| 2     | Approved quick wins                   | P3b, P3          | ~1.3-1.5x              | Phase 1 (P3b), None (P3) |
| 3     | Cut selection enablement              | P1               | 10-50x at scale        | Phase 0 (BUG-2)          |
| 4     | Incremental cut injection             | P2               | 2-3x                   | Phase 3                  |
| 5     | Solver tuning + buffers               | P4, P5           | 1.2-1.5x               | Phase 0 (INST-\*)        |

**Critical path**: Phase 0 → Phase 1 → Phase 3.

Phases 2, 4, and 5 are valuable but secondary to getting cut selection working on a
correct backward pass with proper instrumentation.

---

## 3. Phase 0: Instrumentation + Trivial Correctness

### 3.1 BUG-2: Wire `cut_activity_tolerance` to binding check

**Problem**: `backward.rs:394` uses `cut_duals[cut_idx] > 0.0` (strict positive).
The `cut_activity_tolerance` config field exists but is unused. Cuts with near-zero
positive duals (numerical noise) incorrectly count as binding, keeping useless cuts
alive under Level1.

**Fix**: Replace the strict comparison with `cut_duals[cut_idx] > cut_activity_tolerance`.
Thread the tolerance value from the config through to the binding check site.

**Complexity**: Trivial (one comparison change + plumb config value).

**Validation**: Unit test — inject a cut with dual = 1e-12, verify it is NOT marked
as binding when tolerance = 1e-8.

### 3.2 INST-1: Instrument `solve_with_basis` pre-solve overhead

**Problem**: `solve_time_ms` only measures `run_once()`. The time spent in
`cobre_highs_set_basis` (highs.rs:1150-1156) is uncaptured, contributing to the
38% uninstrumented overhead.

**Fix**: Add a `basis_set_time_ms` field to `SolverStatistics`. Capture elapsed time
around the `set_basis` FFI call in `solve_with_basis`.

### 3.3 INST-2: Instrument cut sync timing

**Problem**: `training.rs:541` hardcodes `sync_time_ms: 0`. For multi-rank MPI runs,
the per-stage allgatherv sends `local_cuts * 8,888 bytes` per rank per stage.

**Fix**: Capture wall-clock around the post-backward cut sync loop. Replace the
hardcoded zero with the measured value.

### 3.4 INST-3: Instrument state exchange timing

**Problem**: Per-stage state exchange (`exchange.exchange(records, t, comm)`) is
lumped into `backward_elapsed_ms` with no separate measurement.

**Fix**: Add `state_exchange_time_ms` to the backward pass timing breakdown. Capture
per-stage and aggregate.

### 3.5 INST-4: Instrument per-opening non-solve time in backward pass

**Problem**: The 38% uninstrumented overhead includes cut batch assembly, coefficient
computation, noise patching, and thread synchronization, but none are separately timed.

**Fix**: Add timing captures for:

- Cut batch build (`build_cut_row_batch_into`) — per-stage
- Coefficient extraction (dual copy + Vec alloc) — per-LP aggregate
- Rayon barrier wait time — per-stage (measure wall-clock minus useful work)

### 3.6 Validation checkpoint

Re-run ONS teste single-rank with instrumentation. Verify that instrumented
components sum to >90% of wall-clock (down from the current 62%). The timing
breakdown becomes the baseline for all subsequent phases.

---

## 4. Phase 1: Backward Pass Structural Fix

### 4.1 BUG-1: Move cut sync inside the per-stage backward loop

**Problem**: Cut synchronization across MPI ranks happens after the entire backward
sweep (training.rs:522-531), not per-stage inside the loop. This violates DEC-009
and means each rank only sees its own current-iteration cuts when building the cut
batch for stage t-1. Results depend on rank count, breaking bit-for-bit reproducibility.

Single-rank runs are unaffected. The bug only manifests with 2+ MPI ranks.

**Current flow**:

```
run_backward_pass():
  for t in (T-2 .. 0):
    1. exchange states for stage t          // allgatherv ✓
    2. build_cut_row_batch(fcf[t+1])        // uses LOCAL-ONLY cuts ✗
    3. solve LPs, generate cuts             // rayon parallel
    4. fcf.add_cut(t, local cuts)           // local only

// AFTER backward pass returns:
for stage in 0..T-1:
  5. sync_cuts(stage, comm)                 // allgatherv — too late
6. cut_selection()
```

**Correct flow (per DEC-009)**:

```
barrier (after forward pass)
for t in (T-2 .. 0):
  1. build_cut_row_batch(fcf[t+1])          // all ranks have same FCF ✓
  2. solve backward LPs                     // rayon parallel
  3. allgatherv: sync NEW cuts for stage t  // immediate, inside loop
  // FCF at stage t now complete for all ranks

// after loop:
4. cut_selection (parallel, all ranks have identical pools)
```

**Key structural changes**:

- The allgatherv for cuts moves from post-sweep to inside the per-stage loop, between
  step 2 (solve) and the next iteration's step 1 (build batch).
- The state exchange (step 1 in current code) remains as-is — it already happens
  per-stage inside the loop.
- Cut selection remains after the loop. Since all ranks now have identical FCF pools,
  selection is a deterministic local operation.

**Deferred: BUG-1b (visited state sync)**. The Dominated cut selection variant needs
accumulated visited states across all iterations. The allgatherv call structure should
accommodate this extension point (reserve a slot for visited state data in the
per-stage sync), but the actual accumulation buffer (~11.5 GB for 50 iter × 200 passes
× 118 stages × 1,106 elements) is a separate design decision. Do not implement the
accumulation in this phase.

**Complexity**: Medium. The backward pass loop structure changes, the cut sync function
moves, and the timing instrumentation from Phase 0 must be adapted.

**Validation**:

- Single-rank: results must be bit-for-bit identical to pre-fix (no behavioral change).
- Two-rank: run a small test case with 1 rank and 2 ranks. Results must be identical.

---

## 5. Phase 2: Approved Quick Wins

### 5.1 P3b: Skip refactorization in backward opening loop

**Problem**: The backward pass opening loop (backward.rs:345-405) calls
`solve_with_basis(working_basis)` for ALL 20 openings. For openings 1..19, only
noise-dependent bounds change — LP structure is identical. Calling `solve_with_basis`
forces HiGHS to discard its internal LU factorization and refactorize from scratch,
even though the basis status codes are identical to what it already has internally.

**Background**: HiGHS has two warm-start paths:

1. **Internal hot-start**: After `run()`, HiGHS retains the factorization. If only
   bounds change and `run()` is called again, it reuses the factorization — zero
   refactorization cost.
2. **External basis** (`setBasis` → `run`): HiGHS treats the basis as new and
   refactorizes from scratch, even if the statuses are identical.

The current code uses path 2 for all openings, wasting 19/20 factorizations.

**Fix**: In the opening loop, use `solve_with_basis` only for opening 0 (which needs
the external basis from BasisStore or from the cut-batch rebuild). For openings 1..19,
call `solve()` directly — HiGHS uses internal hot-start.

Remove the `get_basis` / `working_basis` extraction for openings 0..18. Only extract
the basis after the final opening if needed (currently it is discarded — the working
basis goes out of scope at line 411).

```rust
for omega in 0..n_openings {
    patch_opening_bounds(...);
    if omega == 0 {
        solver.solve_with_basis(external_basis);  // external, refactorize ✓
    } else {
        solver.solve();                           // internal hot-start ✓
    }
    // extract duals and primals for cut coefficients
}
```

**Solver trait implications**: `solve()` (without basis argument) must already exist
or be added to the `SolverInterface` trait. It calls `Highs_run()` directly without
`Highs_setBasis()`. Verify this method exists; if not, add it.

**Complexity**: Trivial.

**Validation**: Bit-for-bit identical results (LP solution is the same regardless of
warm-start path). Compare solver statistics: simplex iteration counts should be
identical or very close, but `basis_set_time_ms` (from INST-1) should drop ~95%.

**Sequencing**: Depends on Phase 1 (BUG-1) because BUG-1 restructures the backward
pass loop. Implementing P3b on the pre-BUG-1 loop would require rework.

### 5.2 P3: Sparse cut injection — drop structural zeros

**Problem**: The state indexer uses uniform stride `max_par_order=6` for all 158 hydros.
Hydros with AR order < 6 have padded lag slots whose cut coefficients are structurally
zero. Currently `build_cut_row_batch_into()` injects all 1,107 entries per cut row
including the 326 zeros (29.5% of the row).

**Fix**: Precompute a nonzero state index mask once at initialization:

```
nonzero_state_indices: Vec<usize>  // len=780
// storage dims [0, N): always included
// lag dims [N, N*(1+L)): included only if lag < actual_ar_order[hydro]
```

In `build_cut_row_batch_into()`, iterate only over the mask instead of the full state
range. The LP injection path uses CSR format (explicit column indices), so skipping
zeros is pure savings with no indirection penalty.

**Impact**:

| Metric                   | Dense (current) | Sparse | Reduction |
| ------------------------ | --------------- | ------ | --------- |
| NNZ per cut row          | 1,107           | 781    | 29.5%     |
| `add_rows` data volume   | 100%            | 70.5%  | 29.5%     |
| Basis factorization work | 100%            | ~70%   | ~30%      |

The LP is mathematically identical — only the representation changes. The savings
propagate to every `add_rows()` call, every basis factorization, and every RowBatch
assembly.

**Complexity**: Low.

**Sequencing**: Independent of all other phases. Can be implemented at any time.

**Validation**: Bit-for-bit identical results. LP dimensions: same row count, fewer
NNZ. Compare `add_rows` timing in solver statistics.

---

## 6. Phase 3: Cut Selection Enablement

### 6.1 P1: Enable and validate Level1/Lml1 cut selection

**Problem**: Cut accumulation without pruning is the dominant bottleneck. LP NNZ grows
159x over 5 iterations. All 112,320 cuts remain active — many are redundant or dominated.
Extrapolation to production iteration counts is prohibitive.

**Current state**: Two functional variants exist but are disabled in test configs:

- **Level1** (`threshold=0`): Deactivates cuts where `active_count == 0` (never binding).
- **Lml1** (`memory_window=N`): Deactivates cuts not binding within the last N iterations.
- **Dominated**: Stub, returns empty set. Blocked on visited state accumulation (BUG-1b).

The infrastructure is in place: `CutPool.deactivate()`, `CutMetadata` with
`active_count` and `last_active_iter`, `CutRowMap` with bound-zeroing deactivation.

**Implementation**:

1. Enable Level1 in the ONS teste config with `threshold=0`, `check_frequency=1`.
2. With BUG-2 fixed (Phase 0), the binding tolerance correctly distinguishes real
   binding from numerical noise.
3. Run ONS teste and observe LP dimension growth — should plateau instead of growing
   linearly.
4. Benchmark Lml1 with `memory_window` values of 3, 5, and 10. Find the sweet spot
   between pruning aggressiveness and convergence impact.

**Basis reuse interaction (Finding 9)**: Basis row statuses are positional. When cut
selection changes which cuts are active, the RowBatch ordering changes and stored
basis row statuses no longer correspond to the correct cuts.

Three options:

1. **Accept degraded warm-start** after selection changes. Only cut row statuses become
   meaningless; template row statuses remain valid. Simplest, zero code.
2. **Remap basis statuses** via CutRowMap's cut slot → LP row correspondence. Medium
   complexity, preserves warm-start quality.
3. **Discard cut row statuses** after selection, keeping only template row basis.
   Explicit version of option 1, slightly cleaner semantics.

**Recommendation**: Start with option 1. If benchmarking shows the warm-start
degradation is costly (measurable via simplex iteration increase in the first opening
after selection), implement option 3 as a clean fallback.

**Complexity**: Medium (mostly validation and benchmarking, not new code).

**Validation**: Run ONS teste with 20+ iterations. Key metrics:

- Cuts/stage should plateau (e.g., 200-500 under Level1 instead of unbounded growth)
- Backward wall-clock per iteration should stabilize
- Lower bound must still converge (no divergence from over-pruning)
- Compare final lower bound at iteration 20 with and without selection — should be
  within statistical noise

---

## 7. Phase 4: Incremental Cut Injection

### 7.1 P2: Persist solver workspaces, use incremental add_rows

**Problem**: `load_backward_lp()` and `load_forward_lp()` do `load_model() + add_rows()`
— full LP rebuild at every stage for every worker. The incremental CutRowMap exists
but is only used for the lower bound solver.

**Fix**: Persist solver workspace LPs between iterations. On subsequent iterations:

- `add_rows()` only the NEW cuts since last iteration
- Deactivate removed cuts via bound-zeroing (CutRowMap already supports this)
- Skip `load_model()` entirely for repeated stages

**Why Phase 3 is prerequisite**: Without cut selection, cuts only accumulate — there
are no deactivations and the incremental path still faces unbounded LP growth. With
selection, the delta between iterations is small: a few hundred new cuts minus a few
hundred deactivated. The incremental path then avoids rebuilding ~2,000+ row LPs from
scratch each time.

**Design considerations**:

- Each rayon thread needs its own persisted solver workspace per stage (or a pool of
  workspaces). Memory cost: `n_threads * n_stages * LP_memory`.
- The forward pass has `n_openings * n_stages` LPs but threads process different
  scenarios. Workspace reuse across scenarios at the same stage is possible if the
  cut set hasn't changed.
- The backward pass has `n_stages` LPs per thread, each reused across openings within
  a stage (openings only differ in bounds, not structure).

**Complexity**: Medium.

**Validation**: Bit-for-bit identical results. Compare `load_model` and `add_rows`
timing in solver statistics. The `load_model` count should drop to near-zero after
the first iteration.

---

## 8. Phase 5: Solver Tuning + Buffer Pre-allocation

### 8.1 P4: Benchmark dual vs primal simplex strategy

**Problem**: The backward pass adds cut rows then warm-starts. With primal simplex
(`strategy=4`, current default), the warm-start basis after row additions is typically
dual-feasible but primal-infeasible, requiring Phase-1 work. Dual simplex (`strategy=1`)
would start from dual feasibility and only pivot to fix primal infeasibilities.

However, between openings within a trial point, only bounds change (no row additions).
Primal simplex may be better for this case.

**Benchmark matrix**:

| Option | Strategy                                                              | When     |
| ------ | --------------------------------------------------------------------- | -------- |
| A      | Dual simplex always                                                   | All LPs  |
| B      | Dual for first opening (cross-iteration basis), primal for subsequent | Split    |
| C      | HiGHS auto-choose (`strategy=0` or `strategy=2`)                      | All LPs  |
| D      | Primal always (current)                                               | Baseline |

Run each on ONS teste with cut selection enabled. Compare total simplex iterations
and wall-clock. The answer may differ from the pre-cut-selection regime.

**Complexity**: Low (config change + benchmarking).

**Note**: With P3b implemented (Phase 2), openings 1..19 use internal hot-start. The
simplex strategy choice mainly affects opening 0 (external basis after cut changes).
This may reduce the impact of this optimization.

### 8.2 P5: Buffer pre-allocation and overhead reduction

**Problem**: Per-LP allocations in the backward pass inner loop:

- `Vec<f64>` allocation for cut coefficients (backward.rs:375,381) — 449K allocs ×
  8.8 KB = 3.9 GB allocation throughput per backward iteration
- `HashMap` for binding slot increments (backward.rs:467) — per-stage allocation with
  hashing overhead

**Fix**:

1. Replace per-LP `Vec<f64>` with a per-thread workspace buffer, sized to `n_state`
   at initialization and reused across LPs via `clear()` + indexed writes.
2. Replace the HashMap for binding slot increments with a `Vec<u32>` indexed by cut
   pool slot, zeroed at stage start.
3. Consider parallelizing `build_cut_row_batch_into()` — it runs serially before the
   parallel LP region. Could be pre-built per stage or built in parallel segments.

**Complexity**: Low.

**Validation**: Identical results. Measure allocation throughput reduction via
instrumentation or allocator profiling.

---

## 9. Deferred Items

These items are tracked but not scheduled:

| Item                               | Blocker                           | Notes                                                                                                                                                                                                                                    |
| ---------------------------------- | --------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| BUG-1b: Visited state accumulation | Design decision on memory budget  | ~11.5 GB for production case (50 iter × 200 passes × 118 stages × 1,106 elements). Needed for Dominated cut selection.                                                                                                                   |
| Dominated cut selection            | BUG-1b                            | Requires accumulated visited states across all iterations. O(active_cuts × visited_states) per stage.                                                                                                                                    |
| Cut selection parallelization      | P1 validated                      | Distribute stages across rayon threads. Trivial once selection is enabled and validated.                                                                                                                                                 |
| P6: NUMA-aware thread placement    | Algorithmic improvements complete | 192 threads on AMD EPYC spans multiple NUMA domains. Thread-to-core pinning and NUMA-local allocation. Requires platform-specific tuning and changes the per-thread work profile — better done after algorithmic improvements stabilize. |
| CLP solver evaluation              | Roadmap                           | Alternative LP solver benchmarking against HiGHS.                                                                                                                                                                                        |

---

## 10. Validation Protocol

Each phase must include a re-run of the ONS teste case to measure impact against the
Phase 0 instrumented baseline. Key metrics to track:

| Metric                            | Source                                                | Expected trend                      |
| --------------------------------- | ----------------------------------------------------- | ----------------------------------- |
| Backward wall-clock per iteration | `timing/iterations.parquet`                           | Decreasing or stabilizing           |
| LP solve time per LP              | `solver/iterations.parquet`                           | Decreasing                          |
| Cost per simplex pivot            | Derived (solve_time / simplex_iters)                  | Bounded after Phase 3               |
| Cuts per stage                    | `solver/iterations.parquet`                           | Plateau after Phase 3               |
| LP NNZ                            | Derived from LP dimensions                            | Bounded after Phase 3               |
| Simplex iterations per LP         | `solver/iterations.parquet`                           | Stable or decreasing                |
| Parallelism efficiency            | Derived (avg_thread / wall_clock)                     | Increasing                          |
| Lower bound                       | `convergence.parquet`                                 | Monotone non-decreasing, converging |
| Uninstrumented overhead %         | Derived (wall_clock - sum(instrumented)) / wall_clock | Decreasing toward <10%              |

**Bit-for-bit invariant**: Phases 0, 2, 4, and 5 must produce identical LP solutions
to the pre-change baseline (single-rank). Phase 1 changes multi-rank results (by
fixing a bug) but single-rank must remain identical. Phase 3 changes convergence
behavior (fewer cuts = different FCF) — validate that convergence is preserved, not
that results are identical.
