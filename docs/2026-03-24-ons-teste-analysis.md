# Performance Analysis Report — Cobre SDDP (ONS teste case)

**Date**: 2026-03-24
**Version**: v0.1.11
**Machine**: AMD EPYC 192 vCPUs
**Run**: 5 training iterations, 192 forward passes, 500 simulation scenarios, 192 threads (no MPI)

---

## Problem Dimensions

| Entity              | Count | Notes                                           |
| ------------------- | ----- | ----------------------------------------------- |
| Hydros              | 158   | 154 with evaporation, 114 cascaded, 1 diversion |
| Thermals            | 104   |                                                 |
| NCS                 | 32    | Stochastic availability                         |
| Buses               | 5     |                                                 |
| Lines               | 6     |                                                 |
| Generic constraints | 18    |                                                 |
| Stages              | 118   | 3 blocks each                                   |
| Openings/stage      | 20    |                                                 |
| State dimension     | 1,106 | 158 storage + 948 inflow lag slots (N\*L, L=6)  |
| Meaningful state    | 780   | 158 storage + 622 used lags (326 padded zeros)  |

**Base LP**: ~4,131 columns × ~1,637 rows × ~6,746 nnz

**Note on state dimension**: The indexer uses uniform stride `max_par_order=6` for all 158 hydros, yielding `N*(1+L) = 1106` state dimensions. Only 780 have nonzero PAR coefficients (AR orders range from 0 to 6). The remaining 326 dimensions are structural zeros in every cut — a 29.5% sparsity floor.

---

## Finding 1: Cut accumulation is the dominant bottleneck

**Zero cuts are being removed.** Active cuts grow linearly: 22,464 → 112,320 over 5 iterations. Each stage accumulates 192 new cuts per iteration with zero selection/pruning.

**Cut density is extreme.** Each cut row has 1,107 entries (1,106 state + theta), of which ~781 are nonzero (29.5% structural zeros from padded lag slots). The base LP row averages only ~4.1 nnz. Cut rows are **~269x denser** than average LP rows (or ~190x counting only nonzeros). After 5 iterations with 960 cuts/stage, the nnz count is **159x the base LP** despite rows only growing 1.6x.

| Metric                 | Iter 1 (192 cuts) | Iter 5 (960 cuts) | Growth   |
| ---------------------- | ----------------- | ----------------- | -------- |
| LP solve time/LP       | 46.3 ms           | 297.0 ms          | **6.4x** |
| Simplex iterations/LP  | 417               | 686               | 1.6x     |
| Cost per simplex pivot | 110 us            | 431 us            | **3.9x** |
| Backward wall-clock    | 228s              | 1,118s            | **4.9x** |

The 3.9x growth in cost-per-simplex-pivot comes purely from HiGHS operating on a larger/denser matrix (larger basis factorization, worse cache behavior).

**Extrapolation** (linear fit on observed data):

| Iterations | Cuts/stage | Projected backward wall-clock (per iter) |
| ---------- | ---------- | ---------------------------------------- |
| 10         | 1,920      | 53 min                                   |
| 20         | 3,840      | 2.8 hr                                   |
| 50         | 9,600      | 15.2 hr                                  |
| 100        | 19,200     | 57.6 hr                                  |

Projected cumulative training time at 100 iterations: **~2,000 hours (83 days)**.

---

## Finding 2: 38% uninstrumented overhead in backward pass

The backward pass wall-clock exceeds the sum of solver-instrumented times (solve + addrows + setbounds + loadmodel) by 38% in iteration 5 (422s out of 1,118s).

Sources:

1. **LIKELY DOMINANT: HiGHS basis installation** (`cobre_highs_set_basis` in `solve_with_basis`, highs.rs:1150-1156) — called 449,280 times per backward pass but NOT captured in `solve_time_ms` (which only measures `run_once()`). The per-LP uninstrumented gap grows with LP size: 51ms at 1,829 rows → 180ms at 2,597 rows. The cost-per-row also grows (28 → 69 us/row), consistent with superlinear basis factorization. This single uninstrumented call likely accounts for the **majority** of the 38% overhead.

   | Iter | Gap per LP | LP rows | us/row | Total gap (est.) |
   | ---- | ---------- | ------- | ------ | ---------------- |
   | 1    | 51 ms      | 1,829   | 27.7   | ~119s            |
   | 5    | 180 ms     | 2,597   | 69.5   | ~422s            |

   **Update (corrected):** Further investigation via [HiGHS issue #617](https://github.com/ERGO-Code/HiGHS/issues/617) and the [HiGHS documentation](https://ergo-code.github.io/HiGHS/dev/guide/further/) reveals that `Highs_setBasis()` does NOT trigger factorization — it only stores basis status codes. The factorization happens inside `Highs_run()`, which IS captured in `solve_time_ms`. The overhead attribution to `set_basis` was incorrect.

   However, this research revealed a much more significant finding — see Finding 11 below (unnecessary refactorizations in the backward opening loop).

   **Action still required:** Instrument `solve_with_basis` to separately capture pre-solve overhead. The 38% gap remains unexplained and is most likely dominated by work imbalance (estimated ~25-40%) amplified by 117 sequential rayon barriers, plus memory/NUMA effects from 192 threads with large LP working sets.

2. **Cut row batch assembly** (`build_cut_row_batch_into`) — serial, ~0.5s at iter 5.

3. **Cut coefficient computation** (backward.rs:368-388) — Vec alloc + dual copy, ~1s per thread.

4. **Thread synchronization + work imbalance** — 117 rayon barriers. Linear model: `overhead = 42s (fixed) + 0.39 × cuts_per_stage` (R²=0.99). The variable component scales with cuts because LP solve time variance increases with LP size → more imbalance at the max-of-192 barrier.

5. **Noise transformation + patch buffer filling** — constant, ~0.5s per thread.

Overhead growth across iterations:

| Iteration | Instrumented (per-thread avg) | Wall-clock | Overhead | Overhead % |
| --------- | ----------------------------- | ---------- | -------- | ---------- |
| 1         | 109s                          | 228s       | 119s     | 52%        |
| 2         | 240s                          | 443s       | 203s     | 46%        |
| 3         | 317s                          | 558s       | 242s     | 43%        |
| 4         | 542s                          | 883s       | 341s     | 39%        |
| 5         | 696s                          | 1,118s     | 422s     | 38%        |

---

## Finding 3: Backward pass parallelism efficiency is 48-62%

| Iteration | Avg thread time | Wall-clock | Efficiency |
| --------- | --------------- | ---------- | ---------- |
| 1         | 109s            | 228s       | 48%        |
| 2         | 240s            | 443s       | 53%        |
| 3         | 317s            | 558s       | 56%        |
| 4         | 542s            | 883s       | 61%        |
| 5         | 696s            | 1,118s     | 62%        |

Sources of inefficiency:

- **Sequential stage loop** with 117 synchronization barriers
- **Serial cut batch build** before each parallel region
- **Work imbalance** within stages (some trial points yield harder LPs)
- **NUMA effects** — 192 threads on AMD EPYC spans multiple NUMA domains

Forward pass efficiency is much better (84-96%) because forward passes are embarrassingly parallel with no inter-stage barriers.

---

## Finding 4: Simulation performance

- 500 scenarios × 118 LPs = 59,000 LP solves
- Aggregate solve time: 50,721s across all threads
- Wall-clock: 335s (5.6 min)
- **460 us/simplex iteration** — consistent with training iter 5 (960 cuts loaded)
- **79% parallelism efficiency** (better than backward: no barriers, scenarios independent)
- 2:1 spread in per-scenario solve time (67s to 141s)
- 17 LP retries across 59,000 solves (negligible)
- Basis warm-start: 100% offered, 0% rejected

---

## Finding 5: Convergence behavior

| Iteration | Lower bound | Upper bound mean | Gap   |
| --------- | ----------- | ---------------- | ----- |
| 1         | 34.69 B$    | 142.05 B$        | 75.6% |
| 2         | 34.70 B$    | 149.26 B$        | 76.7% |
| 3         | 122.41 B$   | 134.04 B$        | 8.7%  |
| 4         | 122.42 B$   | 143.31 B$        | 14.6% |
| 5         | 122.45 B$   | 134.69 B$        | 9.1%  |

- Lower bound stagnated for 2 iterations then jumped 3.5x at iteration 3
- Gap at iteration 5 is still 9.1% — many more iterations needed
- Upper bound std ~5.4B on 134.7B mean (4% CV)
- All 112,320 cuts remain active — many are likely redundant/dominated

---

## Finding 6: LP dimensions with cuts

Base LP density is 0.10%. Cut row density is 26.8% (1107/4131 columns). After cut accumulation:

| Iteration | Cuts/stage | LP rows | LP nnz (full) | NNZ growth | LP nnz (sparse) | Sparse growth |
| --------- | ---------- | ------- | ------------- | ---------- | --------------- | ------------- |
| 0 (base)  | 0          | 1,637   | 6,746         | 1x         | 6,746           | 1x            |
| 1         | 192        | 1,829   | 219,390       | 33x        | 156,698         | 23x           |
| 5         | 960        | 2,597   | 1,069,466     | 159x       | 756,506         | 112x          |
| 10        | 1,920      | 3,557   | 2,132,186     | 316x       | 1,506,266       | 223x          |
| 50        | 9,600      | 11,237  | 10,633,946    | 1576x      | 7,504,346       | 1112x         |
| 100       | 19,200     | 20,837  | 21,261,146    | 3152x      | 15,001,946      | 2224x         |

"Full" = current code (1107 entries/cut including padded zeros). "Sparse" = if zeros were dropped (781 nonzero entries/cut).

Stage 117 (terminal, no cuts) solves at 13.1 ms/LP vs 297 ms/LP for stages with 960 cuts — a **22.6x** slowdown from cuts alone.

---

## Optimization Recommendations

### P1: Cut Selection (Critical — 10-50x potential at scale)

This is overwhelmingly the highest-impact optimization. Without it, performance degrades superlinearly with iteration count.

The infrastructure is already in place:

- `CutPool.deactivate()` exists
- `CutMetadata` tracks `active_count` and `last_active_iter`
- `CutRowMap` supports bound-zeroing deactivation
- Level-1 cut selection is in the trait variant list

Implementation approach:

- **Binding frequency pruning**: Deactivate cuts that haven't been binding for N iterations. Even keeping only the last 3 iterations' binding cuts could reduce the pool from 960 to ~200-300 per stage.
- **Hard cap per stage**: Bound cut pool size (e.g., 500-1000), keeping most-recently-binding cuts.
- **Dominated cut removal**: If cut A has the same coefficients but lower intercept than cut B, remove A.

Limiting to ~300 active cuts/stage would keep LP nnz at ~3x base (vs 112x at iter 5, 2224x at iter 100), keeping cost/simplex-iteration bounded.

### P2: Incremental Cut Injection in Forward/Backward (2-3x potential)

Currently `load_backward_lp()` does `load_model() + add_rows()` — full LP rebuild at every stage for every worker. The incremental `CutRowMap` exists but is only used for the lower bound solver.

If solver workspaces persisted their LP model between iterations:

- Only `add_rows()` the NEW cuts (192 instead of 960+ at iter 5)
- Deactivate removed cuts via bound-zeroing
- Skip `load_model()` entirely at repeated stages

The `model_persistence` epic in recent commits suggests this is already in progress.

### P3: Sparse Cut Injection — Drop Structural Zeros (1.3-1.4x potential, APPROVED)

**Status: approved for implementation.**

The indexer uses uniform stride `max_par_order=6` for all 158 hydros. Hydros with AR order < 6 have padded lag slots whose cut coefficients are structurally zero (exactly 0.0 — the lag-fixing rows for unused slots have no PAR coupling, producing zero duals). Currently `build_cut_row_batch_into()` injects all 1,107 entries per cut row including the 326 zeros.

**Implementation**: Precompute a nonzero state index mask once at initialization (780 entries out of 1,106). In `build_cut_row_batch_into()`, iterate only over the mask instead of the full state range — no per-element comparison needed, just a tighter loop over known-nonzero positions.

```
// Precomputed once:
nonzero_state_indices: Vec<usize>  // len=780
// storage dims [0, N): always included
// lag dims [N, N*(1+L)): included only if lag < actual_ar_order[hydro]
```

Impact:

| Metric                             | Dense (current) | Sparse | Reduction |
| ---------------------------------- | --------------- | ------ | --------- |
| NNZ per cut row                    | 1,107           | 781    | 29.5%     |
| `add_rows` data volume             | 100%            | 70.5%  | 29.5%     |
| Basis factorization work per pivot | 100%            | ~70%   | ~30%      |

This reduces the **cost per simplex iteration** (the metric growing 3.9x over 5 iterations) without changing simplex iteration counts — the LP is mathematically identical. The savings propagate to every `add_rows()` call, every basis factorization, and every RowBatch assembly.

Note: This is distinct from the `SparseCut` storage tradeoff analyzed previously. The LP injection path uses CSR format (explicit column indices) — skipping zeros is pure savings with no indirection penalty.

### P4: HiGHS Simplex Strategy Investigation (unknown potential, needs benchmarking)

**Current config** (highs.rs:90-125, `default_options()`):

| Parameter                      | Value        | Rationale                                 |
| ------------------------------ | ------------ | ----------------------------------------- |
| `solver`                       | `"simplex"`  |                                           |
| `simplex_strategy`             | `4` (primal) | Chosen for between-openings bound changes |
| `simplex_scale_strategy`       | `0` (off)    | Cobre prescaler handles scaling           |
| `presolve`                     | `"off"`      | Warm-start makes presolve redundant       |
| `parallel`                     | `"off"`      | Parallelism is at rayon level             |
| `primal_feasibility_tolerance` | `1e-7`       |                                           |
| `dual_feasibility_tolerance`   | `1e-7`       |                                           |

**Potential conflict: primal simplex vs row additions.** The backward pass adds cut rows then warm-starts. When adding rows with BASIC slack (non-binding default), the warm-start basis is typically dual-feasible but may be primal-infeasible (if new cuts are violated). Primal simplex then needs Phase-1 work. Dual simplex (`strategy=1`) would start from dual feasibility and only pivot to fix primal — potentially fewer iterations for the cross-iteration warm-start.

However, between openings within a trial point, only bounds change (no row additions). Primal simplex may be better for this case. The tradeoff needs benchmarking:

- Option A: Dual simplex always (may hurt between-openings performance)
- Option B: Dual simplex for first opening (cross-iteration basis), primal for subsequent openings
- Option C: Let HiGHS auto-choose (`simplex_strategy=0` or `simplex_strategy=2`)

**Presolve is correctly disabled.** Confirmed at line 105-106. Warm-started LPs should not presolve — it re-analyzes structure the basis already captures.

**Scaling is correctly disabled.** Confirmed at line 101-102. Cobre's LP prescaler (row scaling + RHS prescaling + cost scaling) handles matrix conditioning. HiGHS internal scaling is redundant and was recently disabled (epic-03).

The retry escalation (12 levels, lines 708-1037) progressively enables presolve, switches to dual simplex, then IPM, with increasingly aggressive scaling as fallback. Only 17 retries across 59,000 simulation LPs confirms the defaults work well for this problem.

### P5: Buffer Pre-allocation and Overhead Reduction (1.1-1.3x potential)

1. **Reusable cut coefficient buffer** — replace per-LP `Vec<f64>` allocation (backward.rs:375,381) with a workspace buffer. Eliminates ~449K allocations × 8.8KB (1,106 elements) = 3.9GB allocation throughput per backward pass.
2. **Replace HashMap** for binding slot increments (backward.rs:467) with a fixed-size vec indexed by slot.
3. **Parallelize cut batch build** — `build_cut_row_batch_into()` runs serially before the parallel LP region. Could be pre-built or parallelized.

### P6: NUMA-Aware Thread Placement (1.1-1.3x potential)

192 threads on AMD EPYC spans multiple NUMA domains. Thread-to-core pinning and NUMA-local allocation for solver workspaces could improve the 48-62% backward pass parallelism efficiency.

---

## Finding 8: Forward Pass Efficiency Dip at Iteration 2

| Iteration | Wall  | Instrumented/thread | Overhead  | Simplex/LP | us/simplex | Basis offered |
| --------- | ----- | ------------------- | --------- | ---------- | ---------- | ------------- |
| 1         | 5.8s  | 5.65s               | 2.8%      | 2,312      | 20.4       | 0 (cold)      |
| 2         | 18.3s | 11.74s              | **35.7%** | 879        | 108.9      | 22,656        |
| 3         | 47.6s | 40.86s              | 14.2%     | 1,700      | 198.2      | 22,656        |
| 4         | 64.6s | 54.57s              | 15.5%     | 1,586      | 283.6      | 22,656        |
| 5         | 96.5s | 83.52s              | 13.4%     | 1,764      | 391.3      | 22,656        |

Iteration 2 shows an anomalous **35.7% overhead** (6.53s uninstrumented) vs 2.8% at iteration 1 and 13-15% at iterations 3-5. Contributing factors:

- **First iteration with cuts**: cut batch building runs for the first time (192 cuts × 781 nnz × 118 stages = 17.6M elements assembled serially before the parallel region)
- **First-time BasisStore population**: Iteration 1 forward had 0 basis offers (all cold-starts). The BasisStore is first populated at the end of iteration 1. Iteration 2 is the first to use warm-start, and the warm-start bases have NO cut rows (iteration 1 LP had zero cuts), causing dimension mismatch fill for all 192 cut rows per stage
- **Simplex iterations dropped 62%** (2312 → 879) thanks to warm-start, but us/simplex grew 5.3x (20.4 → 108.9) due to the larger LP from cuts

---

## Finding 9: Basis Reuse and Cut Selection Interaction

The BasisStore holds per-(scenario, stage) bases from the forward pass, including cut row statuses. The basis reuse flow is:

```
Forward iter k: solve with K cuts → store basis (template_rows + K rows) in BasisStore
Backward iter k: offer BasisStore basis for first opening, working_basis for subsequent
  → Dimension mismatch if backward has K+delta cuts: fill delta rows with BASIC
  → working_basis discarded after opening loop (NOT saved to BasisStore)
Forward iter k+1: offer BasisStore basis from iter k, which has K cut rows
  → LP now has K+192 cuts: first K row statuses reused, 192 new rows get BASIC
```

**Warning for cut selection**: Basis row statuses are positional. If cut selection changes which cuts are active between iterations, the RowBatch ordering changes (based on pool slot iteration order). Row position i in the new LP may correspond to a different cut than row position i in the stored basis. The reused cut row statuses become meaningless — only the template row statuses remain valid.

Options when cut selection is implemented:

1. Accept degraded warm-start quality after selection changes (simplest)
2. Track cut slot → LP row correspondence via CutRowMap and remap basis statuses
3. Discard cut row statuses entirely after selection, keeping only template row basis

---

## Finding 11: Unnecessary Basis Refactorization in Backward Opening Loop (APPROVED)

The backward pass opening loop (backward.rs:345-405) calls `solve_with_basis(working_basis)` for ALL 20 openings, including openings 1..19 where the LP structure hasn't changed — only noise-dependent bounds were patched.

### The problem

HiGHS has two warm-start paths ([ref: issue #617](https://github.com/ERGO-Code/HiGHS/issues/617)):

1. **Internal hot-start**: After a successful `run()`, HiGHS retains the optimal basis AND its LU factorization internally. If only bounds change (`set_row_bounds`/`set_col_bounds`) and you call `run()` again, HiGHS **reuses the existing factorization** — zero refactorization cost.

2. **External basis warm-start** (`setBasis` → `run`): HiGHS receives status codes from outside. Even if they are identical to its internal basis, it treats them as a new user-provided basis and **must refactorize from scratch**.

The current code does path 2 for ALL openings:

```rust
for omega in 0..20 {
    patch_opening_bounds(...);                    // only bounds change
    let warm = if omega == 0 {
        basis_store.get(m, s)                     // external: forward-pass basis ✓
    } else {
        working_basis.as_ref()                    // extracted from PREVIOUS solve
    };
    solver.solve_with_basis(warm);                // setBasis → destroys internal factorization → refactorize
    solver.get_basis(working_basis);              // extract basis back out (redundant for omega 1..18)
}
```

For openings 1..19, `working_basis` was just extracted via `get_basis()` — it contains the **exact same** status codes that HiGHS already has internally. Calling `setBasis` with them destroys the internal factorization and forces a redundant refactorization.

### The fix

```rust
for omega in 0..20 {
    patch_opening_bounds(...);
    if omega == 0 {
        solver.solve_with_basis(basis_store.get(m, s));  // first: external basis, refactorize ✓
    } else {
        solver.solve();                                   // subsequent: internal hot-start, NO refactorization
    }
}
// extract basis only once after last opening (if needed at all — currently discarded)
```

### Expected impact

The initial LU factorization of a ~2,600-row LP with dense cut rows is expensive — potentially a significant fraction of the per-LP solve time. By eliminating 19/20 factorizations (only the first opening needs external basis), the solve time for openings 1..19 could drop substantially.

With 449,280 backward LP solves per iteration, 19/20 = 427,416 of them currently do unnecessary refactorization. Even a modest 50us savings per LP from avoiding refactorization would save `427,416 × 50us = 21s` aggregate, or `0.1s` per thread — small. But if refactorization is a larger fraction of per-LP time (plausible for LPs where simplex converges quickly in few iterations), the savings scale up.

**This also eliminates 427,416 `get_basis` calls** (only 22,464 are needed — one per trial point's last opening), each copying ~6,700 status codes. And it eliminates 427,416 `setBasis` FFI calls.

### Additional benefit: enables HiGHS incremental factorization updates

When HiGHS uses internal hot-start after bound changes, it can perform incremental factorization updates (only modifying the parts affected by bound changes) rather than full refactorization. This is the fastest possible path for the between-openings case.

### Correctness

Zero risk. The LP solution is identical — only the warm-start path changes. The `view.dual` and `view.primal` outputs used for cut coefficient extraction come from the solve result regardless of warm-start method.

The `get_basis` call for `working_basis` can be removed entirely for openings 1..18 — the extracted basis was only used as input to the next opening's `solve_with_basis`, which is now replaced by `solve()`. The final opening's basis was already discarded (working_basis goes out of scope at line 411).

---

### Summary

| Priority | Optimization                            | Expected Impact       | Complexity  | Status      |
| -------- | --------------------------------------- | --------------------- | ----------- | ----------- |
| **P1**   | Cut selection (Level-1)                 | **10-50x at scale**   | Medium      | Planned     |
| **P2**   | Incremental cut injection (fwd/bwd)     | 2-3x                  | Medium      | Planned     |
| **P3**   | Sparse cut injection (drop zeros)       | 1.3-1.4x              | Low         | Approved    |
| **P3b**  | Skip refactorization in opening loop    | Unknown (needs bench) | **Trivial** | Approved    |
| **P4**   | HiGHS simplex strategy (dual vs primal) | Unknown (needs bench) | Low         | Investigate |
| **P5**   | Buffer pre-allocation                   | 1.1-1.3x              | Low         | Planned     |
| **P6**   | NUMA-aware threading                    | 1.1-1.3x              | Medium      | Planned     |

**P1 is the gating factor.** All other optimizations are secondary. Without cut selection, the LP grows without bound and every operation pays the full cost of the unbounded cut pool.

---

## Finding 7: Inflow Lags Dominate the State Dimension

The 1,106-dimensional state vector is 80% inflow lags (948 slots) and only 14% storage volumes (158 slots). This has profound consequences:

- Each cut coefficient vector is 1,107 entries wide (1,106 state + theta)
- The lag-fixing constraints generate nonzero duals for all 622 used lag dimensions
- Storage-only cuts (if theoretically possible) would be 159 entries — **7x narrower**

### State composition

| Component                         | Dimensions | % of n_state | In cuts?                  |
| --------------------------------- | ---------- | ------------ | ------------------------- |
| Storage volumes                   | 158        | 14.3%        | Always nonzero            |
| Used inflow lags (AR(1)-AR(6))    | 622        | 56.2%        | Nonzero (coupled via PAR) |
| Padded lag slots (AR < max_order) | 326        | 29.5%        | Structurally zero         |

### Production scenario projection (50 iter × 200 fwd)

With cut selection keeping N active cuts per stage:

| Active cuts/stage | Full nnz/cut | Sparse nnz/cut | LP nnz (sparse) | LP nnz growth |
| ----------------- | ------------ | -------------- | --------------- | ------------- |
| 500               | 1,107        | 781            | 397,246         | 59x           |
| 1,000             | 1,107        | 781            | 787,746         | 117x          |
| 2,000             | 1,107        | 781            | 1,568,746       | 233x          |

Even with aggressive cut selection at 500 cuts/stage, LP nnz is still 59x base — the per-simplex-iteration cost will remain elevated due to the dense cut rows.

---

## CRITICAL BUG: Backward Pass Cut Sync Is Outside the Per-Stage Loop

### The problem

The backward pass cut synchronization across MPI ranks happens **after** the entire backward sweep, not per-stage inside the loop. This breaks the invariant that all ranks see the same FCF when building the cut batch for the next stage.

**Current implementation** (training.rs:478-531, backward.rs:554-638):

```
run_backward_pass(...):                           // entire sweep in one call
  for t in (116, 115, ..., 0):
    1. exchange.exchange(records, t, comm)         // allgatherv: sync trial point STATES ✓
    2. build_cut_row_batch_into(batch[t+1], fcf)   // build cuts from LOCAL-ONLY FCF ✗
    3. process_stage_backward(...)                  // rayon: solve LPs, generate cuts
    4. fcf.add_cut(t, local cuts only)              // insert THIS rank's cuts only

// AFTER the backward pass returns (training.rs:522-531):
for stage in 0..117:
  5. sync_cuts(stage, local_cuts, fcf, comm)       // allgatherv: share cuts across ranks
6. cut_selection (serial, per-stage scan)
```

**The bug:** At step 2, `build_cut_row_batch_into` reads `fcf.pools[t+1]` which only contains this rank's current-iteration cuts for stages > t+1. Other ranks' cuts from this iteration are missing — they arrive at step 5 after the entire backward sweep. With R ranks each doing F/R forward passes, the stage t+1 LP has F/R cuts from this rank instead of F cuts from all ranks. **Results depend on rank count, violating bit-for-bit reproducibility.**

Single-rank runs (like this test) are unaffected — all cuts are local and immediately visible. The bug only manifests with 2+ MPI ranks.

### Correct implementation (per DEC-009)

```
barrier (after forward pass)
for t in (116, 115, ..., 0):
  1. solve backward LPs using local trial point states    // rayon parallel
  2. allgatherv: sync NEW cuts for stage t across ranks   // all ranks now have all cuts
  2b. allgatherv: sync visited states for stage t          // accumulate for dominated selection
  // FCF at stage t now complete — loop to t-1 is self-consistent

// after loop:
3. cut selection (parallel across stages — all ranks have identical pools)
```

Key changes:

- **Move cut sync inside the per-stage loop** (between step 1 and the next iteration of the loop). After generating cuts at stage t, allgatherv them immediately so all ranks have the full cut set before processing stage t-1.
- **Sync visited states together with cuts.** The per-stage allgatherv should also exchange this rank's forward pass states for the stage, accumulating them into a historical archive. This is necessary for the Dominated cut selection variant (which needs all visited states across all iterations to detect dominated cuts).
- **Cut selection runs after the loop, parallelizable.** Since all ranks have identical FCF pools after the synchronized backward sweep, cut selection is a deterministic local operation. It can be parallelized by distributing stages across MPI ranks and/or rayon threads. For computationally heavy strategies (e.g., Dominated with O(active_cuts × visited_states) per stage), this parallelization is essential. Each rank/thread processes a subset of stages independently, and the deactivation results are either applied locally (since all ranks compute identically) or synced via a lightweight allgatherv of DeactivationSets.

### Impact on this analysis

This test ran with a single rank (no MPI), so all data in this report is unaffected by the bug. But the fix is blocking for production multi-rank runs:

- Correctness: results must not depend on rank count
- Performance: per-stage allgatherv adds communication cost but is necessary
- Visited state accumulation: enables Dominated cut selection (currently stub)

---

## Finding 10: Cut Selection Implementation Status

Cut selection is **implemented but disabled by default** in the test config. Two functional variants exist:

- **Level1** (`threshold=0`, default): Deactivates cuts where `active_count == 0` (never binding). Conservative, preserves convergence.
- **Lml1** (`memory_window=N`): Deactivates cuts not binding within the last N iterations. More aggressive.
- **Dominated**: Stub (returns empty set). Requires accumulated visited states — not yet implemented.

### Post-backward sequence (training.rs:522-582, current)

```
5. Cut sync: for each stage, allgatherv to distribute cuts across MPI ranks
6. Cut selection: serial scan of metadata per stage, deactivate inactive cuts
7. Lower bound evaluation, convergence check
```

### Timing instrumentation gaps

| Operation                             | Instrumented? | Notes                           |
| ------------------------------------- | ------------- | ------------------------------- |
| Forward pass wall-clock               | Yes           | `forward_elapsed_ms`            |
| Forward sync (UB allreduce)           | Yes           | `sync_time_ms`                  |
| Backward pass wall-clock              | Yes           | `backward_elapsed_ms`           |
| State exchange (per-stage allgatherv) | **No**        | Lumped into backward_elapsed_ms |
| Cut sync (post-backward allgatherv)   | **No**        | Hardcoded `sync_time_ms: 0`     |
| Cut selection scan                    | Yes           | `selection_time_ms`             |
| Lower bound broadcast                 | **No**        | Not captured                    |

### Implementation issues found

1. **Binding tolerance not wired**: backward.rs:394 uses `cut_duals[cut_idx] > 0.0` (strict). The `cut_activity_tolerance` config field exists but is unused. Cuts with near-zero positive duals (numerical noise) incorrectly count as binding, keeping useless cuts alive under Level1.

2. **Visited states not accumulated**: Forward pass `records` buffer is overwritten each iteration (training.rs:287-293). No historical state archive exists. The Dominated variant cannot work without this. Accumulating states for 50 iterations × 200 passes × 118 stages × 1,106 elements ≈ 11.5 GB.

3. **Cut sync timing not measured**: training.rs:541 hardcodes `sync_time_ms: 0`. For multi-rank MPI runs, the per-stage allgatherv sends `local_cuts × 8,888 bytes` (40-byte header + 1,106 × 8 coefficient bytes) per rank per stage — potentially significant.

4. **Cut selection is serial but could be parallel**: The per-stage metadata scan (training.rs:552-564) is a serial for-loop. Since each stage's pool is independent, this could use rayon `par_iter` trivially.

5. **Multi-rank cut visibility**: During the backward sweep, each rank only sees its own current-iteration cuts. Other ranks' cuts arrive at step 5 (post-backward sync). This means the backward pass uses a "staler" FCF in multi-rank mode — valid but may slow convergence vs single-rank.

### Config to enable for next test

```json
"training": {
  "cut_selection": {
    "enabled": true,
    "method": "level1",
    "threshold": 0,
    "check_frequency": 1
  }
}
```

---

## Action Items Summary

### Critical (correctness)

| ID         | Item                                                             | Complexity |
| ---------- | ---------------------------------------------------------------- | ---------- |
| **BUG-1**  | Move cut sync inside backward per-stage loop (DEC-009 violation) | Medium     |
| **BUG-1b** | Sync visited states together with cuts (for Dominated selection) | Medium     |
| **BUG-2**  | Wire `cut_activity_tolerance` to binding check (backward.rs:394) | Trivial    |

### Approved (ready to implement)

| ID      | Item                                                                                              | Expected Impact | Complexity |
| ------- | ------------------------------------------------------------------------------------------------- | --------------- | ---------- |
| **P3**  | Sparse cut injection — precompute nonzero mask, skip padded zeros in RowBatch                     | 1.3-1.4x        | Low        |
| **P3b** | Skip `solve_with_basis`/`get_basis` for openings 1..N, use `solve()` for HiGHS internal hot-start | Unknown (bench) | Trivial    |

### Instrumentation (needed before further optimization)

| ID         | Item                                                                   |
| ---------- | ---------------------------------------------------------------------- |
| **INST-1** | Instrument `solve_with_basis` to split basis-set time from solve time  |
| **INST-2** | Instrument cut sync timing (replace hardcoded zero at training.rs:541) |
| **INST-3** | Instrument state exchange timing (separate from backward_elapsed_ms)   |
| **INST-4** | Instrument per-opening non-solve time in backward pass                 |

### Planned (need design or benchmarking)

| ID     | Item                                                                     | Expected Impact | Complexity |
| ------ | ------------------------------------------------------------------------ | --------------- | ---------- |
| **P1** | Cut selection — enable Level1/Lml1 and test                              | 10-50x at scale | Medium     |
| **P2** | Incremental cut injection in fwd/bwd (model persistence + CutRowMap)     | 2-3x            | Medium     |
| **P4** | Benchmark dual vs primal simplex strategy for cross-iteration warm-start | Unknown         | Low        |
| **P5** | Buffer pre-allocation (cut coefficients, HashMap → Vec)                  | 1.1-1.3x        | Low        |
| **P6** | NUMA-aware thread placement                                              | 1.1-1.3x        | Medium     |

### Deferred

| ID                            | Item                                                              | Blocker  |
| ----------------------------- | ----------------------------------------------------------------- | -------- |
| Dominated cut selection       | Requires visited state accumulation (~11.5GB for production case) | BUG-1b   |
| Cut selection parallelization | Distribute stages across MPI ranks/rayon threads                  | P1 first |
| CLP solver evaluation         | Alternative solver benchmarking                                   | Roadmap  |

---

## Raw Data Reference

- Training timing: `output/training/timing/iterations.parquet`
- Solver stats: `output/training/solver/iterations.parquet`
- Convergence: `output/training/convergence.parquet`
- Scaling report: `output/training/scaling_report.json`
- LP dictionaries: `output/training/dictionaries/`
- Simulation solver: `output/simulation/solver/iterations.parquet`
- Simulation manifest: `output/simulation/_manifest.json`
- Training metadata: `output/training/metadata.json`
