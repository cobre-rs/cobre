# LP Setup Profiling Results: Convertido 50-Iteration Run

## Status

Analysis — 2026-03-23

## Test System

| Parameter                        | Value                                             |
| -------------------------------- | ------------------------------------------------- |
| Case                             | Convertido (Newave-converted)                     |
| Stages                           | 117                                               |
| Forward passes                   | 5 (limited by core count)                         |
| Iterations                       | 50                                                |
| Backward openings × trial points | 100 per stage                                     |
| Forward LPs per iteration        | 590                                               |
| Backward LPs per iteration       | 11,700                                            |
| Cut selection                    | None active (cuts_removed = 0 for all iterations) |
| Final cut count                  | 250 per stage (29,250 total)                      |
| Parallelism                      | ~3.7x (rayon, inferred from solver/wall ratio)    |

---

## 1. Empirical Cost Models

All times are **thread-aggregated** (sum across rayon threads), not wall-clock.
Divide by the parallelism factor (~3.7x on this machine) for wall-clock estimates.

### 1.1 Costs that scale with cut count (per LP solve)

Measured via linear regression on forward pass data across 50 iterations.

| Component         | Model               |    R² | Unit      |
| ----------------- | ------------------- | ----: | --------- |
| `add_rows`        | `0.01278 × C`       | 0.996 | ms per LP |
| `solve` (simplex) | `8.60 + 0.1048 × C` | 0.988 | ms per LP |

Where `C` = active cuts per stage. Both are strongly linear. The key insight
is that **solve cost also grows linearly with cut count** because each cut
adds a row to the LP, increasing the simplex work per solve.

### 1.2 Costs independent of cut count (per LP solve)

| Component    |     Cost | Notes                             |
| ------------ | -------: | --------------------------------- |
| `load_model` | 0.130 ms | Constant across all 50 iterations |
| `set_bounds` | 0.128 ms | Constant across all 50 iterations |

These are **fixed per-LP-solve** costs: every scenario at every stage pays
them regardless of how many cuts exist.

### 1.3 Cut-count equivalence

The cost models depend only on `C` (cuts per stage), not on how that count
was reached. A run with 50 iterations × 5 forward passes produces the same
`C = 250` at the last iteration as 5 iterations × 50 forward passes. The
per-LP costs at that point are identical. What differs is:

- **Total LP solve count**: more forward passes = more LPs per iteration
- **Cumulative cost**: more iterations = more iterations at intermediate C values

This means we can extrapolate from our measured cost models to any
(iterations, forward_passes) configuration.

---

## 2. Observed Scaling

### 2.1 Per-LP cost breakdown at selected cut counts

| Cuts/stage | solve (ms) | add_rows (ms) | load (ms) | bounds (ms) | Setup % |
| ---------: | ---------: | ------------: | --------: | ----------: | ------: |
|          5 |       21.9 |         0.001 |      0.12 |        0.12 |    1.1% |
|         25 |       11.0 |          0.27 |      0.13 |        0.13 |    4.5% |
|         50 |       14.2 |          0.54 |      0.13 |        0.13 |    5.3% |
|        100 |       19.7 |          1.19 |      0.14 |        0.13 |    6.9% |
|        150 |       24.8 |          1.88 |      0.14 |        0.13 |    8.0% |
|        200 |       29.5 |          2.48 |      0.14 |        0.13 |    8.5% |
|        250 |       32.6 |          3.02 |      0.13 |        0.13 |    9.2% |

The iteration-1 solve time (21.9ms) is an outlier — the first iteration
has no cuts but also no warm-start basis, causing high simplex iteration
counts (2168 vs ~330 thereafter).

### 2.2 add_rows growth

In the forward pass, `add_rows` accounts for:

| Iteration | % of setup cost |
| --------: | --------------: |
|         1 |    0% (no cuts) |
|        10 |             68% |
|        25 |             85% |
|        50 |         **92%** |

`add_rows` starts negligible and becomes the dominant setup component
by iteration 10.

### 2.3 Backward pass: set_bounds dominance at low cut counts

The backward pass shows a different pattern. `set_bounds` accounts for
91% of backward setup at iteration 1 but only 38% at iteration 50, as
`add_rows` grows:

| Iter | add_rows | set_bounds | add_rows share |
| ---: | -------: | ---------: | -------------: |
|    1 |    65 ms |   1,469 ms |             4% |
|   10 |   431 ms |   1,523 ms |            21% |
|   30 | 1,451 ms |   1,577 ms |            46% |
|   50 | 2,393 ms |   1,525 ms |        **60%** |

The backward `set_bounds` is ~10x more expensive than the forward pass
(0.13 ms vs 0.013 ms per LP). This is because the backward pass patches
more rows per solve — the trial-point state-fixing rows change for each
opening, requiring more `changeRowBoundsBySet` entries.

The crossover (where `add_rows` exceeds `set_bounds` in backward) occurs
around **iteration 35** for this system.

---

## 3. Wall-Clock Context

Solver times are thread-aggregated. Wall-clock times reflect actual elapsed
time with ~3.7x parallelism.

| Iter | Wall (s) | Fwd wall (s) | Bwd wall (s) | Solver total (s) | Setup total (s) |
| ---: | -------: | -----------: | -----------: | ---------------: | --------------: |
|    1 |     14.6 |          2.6 |         11.6 |             60.5 |             1.7 |
|   10 |     26.3 |          2.0 |         23.2 |             96.1 |             2.5 |
|   20 |     40.1 |          3.0 |         35.4 |            142.1 |             3.5 |
|   30 |     51.9 |          4.0 |         46.2 |            186.0 |             4.4 |
|   40 |     68.4 |          5.0 |         60.5 |            243.1 |             5.2 |
|   50 |     72.3 |          5.3 |         65.0 |            257.2 |             5.9 |

At this scale and cut count (250), setup is ~2-3% of wall-clock time.
The simplex solver dominates. But `add_rows` is already the fastest-growing
component.

---

## 4. Extrapolation to Production

### 4.1 Last-iteration forward pass cost

Using the empirical cost models to project per-LP costs at the final
iteration, then multiplying by forward LPs per iteration.

| Configuration          |  C_end | FLP/iter | add_rows (s) | solve (s) | Setup % |
| ---------------------- | -----: | -------: | -----------: | --------: | ------: |
| This run (5F×50I)      |    250 |      590 |          1.9 |      20.5 |      9% |
| 50F×5I equivalent      |    250 |    5,850 |         18.7 |     203.6 |      9% |
| Prod small (20F×200I)  |  4,000 |    2,400 |        122.7 |   1,026.8 |     11% |
| Prod medium (50F×200I) | 10,000 |    6,000 |        766.8 |   6,340.2 |     11% |
| Prod large (200F×300I) | 60,000 |   24,000 |       18,403 |   151,133 |     11% |

- **C_end** = iterations × forward_passes = final cuts per stage
- **FLP/iter** = forward_passes × stages = LP solves per forward iteration

### 4.2 Cumulative training cost (forward pass only, all iterations)

| Configuration | Solve (h) | add_rows (h) | Setup % | S1 saves (h) |
| ------------- | --------: | -----------: | ------: | -----------: |
| This run      |      0.18 |        0.013 |    7.9% |        0.012 |
| Prod small    |      29.2 |          3.4 |   10.6% |          3.3 |
| Prod medium   |     178.4 |         21.4 |   10.8% |         21.0 |
| Prod large    |   6,326.8 |        769.4 |   10.8% |        765.8 |

The **cumulative** setup fraction stabilizes around **10.8%** at production
scale. This is because `add_rows` grows linearly with iteration while
`solve` also grows linearly (both scale with `C`), keeping their ratio
roughly constant.

### 4.3 Key scaling insight

The `add_rows`-to-`solve` ratio is governed by:

```
add_rows/solve = 0.01278 / (8.60/C + 0.1048) → 0.01278 / 0.1048 ≈ 12.2%
```

As `C → ∞`, the ratio converges to ~12.2%. This means **`add_rows` will
never exceed ~12% of total per-LP cost** with the current system
dimensions, because the solve cost also grows linearly with the same cut
count.

However, the absolute cost matters:

- At C=10,000 cuts: `add_rows` = 128 ms/LP, `solve` = 1,057 ms/LP
- At 6,000 forward LPs/iter: that's **12.8 minutes** of `add_rows` per
  iteration, or ~42.7 hours cumulative across 200 iterations

---

## 5. Strategy Impact Assessment (Revised)

### S1: Model persistence across scenarios

**Mechanism:** Load model + add cuts once per stage per iteration. Patch
bounds for each scenario.

**Projected savings at Prod medium (50F×200I):**

| Component       | Baseline/iter | After S1 |     Reduction |
| --------------- | ------------: | -------: | ------------: |
| add_rows        |        766.8s |    15.3s |       **98%** |
| load_model      |          0.8s |     0.0s |           98% |
| set_bounds      |          0.8s |     0.8s |            0% |
| **Total saved** |               |          | **752s/iter** |

Cumulative: **~21 hours saved** across 200 iterations.

S1 is high-impact because it eliminates the per-scenario multiplier.
Instead of `fwd_passes × stages` load+add_rows calls, you get `stages`
calls. The effect scales with scenario count.

### S2: Incremental cut addition (on top of S1)

**Mechanism:** After S1, the remaining `add_rows` cost is one call per
stage with all active cuts. S2 reduces this to appending only new cuts.

**Projected additional savings at Prod medium:**

| Component | After S1 | After S1+S2 | Additional |
| --------- | -------: | ----------: | ---------: |
| add_rows  |    15.3s |        0.1s | 15.2s/iter |

This is smaller than S1 because S1 already eliminated 98% of the cost.
S2 matters mainly at very high cut counts (C > 10,000) where even one
full `add_rows` per stage becomes expensive.

### S5: Cache active count

**Mechanism:** Maintain `active_count` field on `CutPool`.

**Impact:** Eliminates O(pool_size) scan in `build_cut_row_batch`. At
250 cuts this is negligible (<0.1ms). At 60,000 cuts it saves a
measurable amount in the per-stage assembly, but remains minor compared
to S1/S2.

### S3: RowBatch buffer reuse

**Impact:** Eliminates `stages × iterations` heap allocations. At
117 × 50 = 5,850 allocations with ~50KB each, this saves ~280MB of
allocation churn. The actual time savings are modest (~ms range) but
it reduces allocator pressure and improves cache behavior.

### S4: Sparse cut coefficients

**Impact:** Reduces NNZ per cut from `(n_state + 1)` to potentially
30-50% of that for systems with sparse hydraulic coupling. This directly
reduces `add_rows` cost per cut by the same factor. Effect compounds
with S1 and S2.

For this system, `n_state` appears to be moderate (~100-200 based on
117 hydros), so the potential is significant but requires analysis of
actual coefficient sparsity patterns.

---

## 6. Recommended Priority

| Priority | Strategy                  | Rationale                                                                                                                 |
| :------: | ------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
|  **1**   | **S1: Model persistence** | 98% reduction in `add_rows` + `load_model`. Scales with scenario count. By far the highest absolute impact.               |
|  **2**   | S3: RowBatch buffer reuse | Low complexity, eliminates allocation churn. Good hygiene before tackling S2.                                             |
|  **3**   | S5: Cache active count    | Trivial implementation, removes O(pool_size) scans.                                                                       |
|  **4**   | S2: Incremental cuts      | Needed only after S1 at extreme cut counts (C > 10,000). Medium complexity (requires LP row index tracking).              |
|  **5**   | S4: Sparse coefficients   | High complexity, moderate impact. Depends on system-specific sparsity. Defer until profiling shows NNZ is the bottleneck. |

### Why S1 is the clear winner

The data shows that **97-98% of `add_rows` cost comes from the per-scenario
multiplier**, not from the per-stage-per-iteration cost. At production
medium scale:

- Baseline: `add_rows` = 766.8s/iter (6,000 LPs × 128 ms each)
- After S1: `add_rows` = 15.3s/iter (120 stages × 128 ms each)
- Remaining after S1+S2: 0.1s/iter

S1 alone captures virtually all the `add_rows` savings. S2 provides
diminishing returns. The implementation complexity of S1 is moderate
(verify HiGHS bound patching works on dynamic rows, restructure the
forward/backward scenario loops), while S2 requires LP row index
tracking and careful interaction with cut deactivation.

---

## 7. What the data does NOT tell us

1. **`build_cut_row_batch` cost** — not separately timed. Hidden in the
   overhead residual (~0.4-2.8s wall-clock per iteration). Likely modest
   at 250 cuts but could matter at 10,000+.

2. **MPI communication cost** — this run is single-rank. Multi-rank runs
   will add allgatherv/allreduce overhead proportional to cut count × ranks.

3. **Memory pressure** — at 60,000 cuts/stage × 117 stages × ~200 state
   dimensions × 8 bytes, the cut pool alone is ~11 GB. Memory effects
   (cache misses, TLB pressure) may change the cost models.

4. **HiGHS internal overhead** — `set_bounds` in the backward pass
   costs 10x more than in the forward pass. This may be due to HiGHS
   invalidating presolve/factorization caches on bound changes. Needs
   investigation with HiGHS-level profiling.

5. **Simplex scaling** — the linear model `solve = 8.6 + 0.105 × C`
   was fit on C = 5..250. At C = 10,000+ the relationship may become
   superlinear as the LP gets denser and simplex pivoting degrades.

---

## 8. Memory Analysis

### 8.1 Convertido system dimensions

| Parameter                 |                               Value |
| ------------------------- | ----------------------------------: |
| Hydros (N)                |                                 158 |
| Max PAR order (L)         |                                   6 |
| n_state (LP) = N×(1+L)    |                               1,106 |
| n_state (semantic)        | 780 (158 storage + 622 actual lags) |
| Padding zeros             |        326 (29% of state dimension) |
| NNZ per cut row           |               1,107 (= n_state + 1) |
| LP columns (approx)       |                               2,371 |
| LP template rows (approx) |                               1,427 |
| Buses                     |                                   5 |
| Thermals                  |                                 104 |
| Lines                     |                                   6 |
| NCS                       |                                  32 |

The uniform-stride PAR layout uses L_max=6 for all hydros, but only
105 of 158 hydros actually have order >= 5. The remaining hydros have
order 1-4, so their higher lag slots are zero-padded. This creates
29% structural sparsity in cut coefficients.

### 8.2 Memory components

**Cut Pool** — stores coefficients (dense `Vec<f64>`, length n_state),
intercept, and metadata per cut per stage.

- Bytes per cut: 8,921 (8,848 coefficients + 8 intercept + 40 metadata + 25 overhead)
- Dominates memory at scale due to O(C × n_state × stages) growth.

**RowBatch** — CSR materialization of active cuts, rebuilt each iteration.
All n_stages batches exist simultaneously.

- Per-cut NNZ: 1,107 values (f64) + 1,107 col_indices (i32) + bounds (2 × f64)
- The col_indices are 100% redundant — same pattern `[0, 1, ..., n_state-1, theta]`
  repeated for every cut.
- RowBatch is ~1.5x the CutPool coefficient size due to the
  col_indices + values duplication in CSR format.

**Basis Store** — one basis per (scenario, stage) for both forward and
backward passes. Basis size grows with LP row count (template + cuts).

**Solver Instances** — one HiGHS instance per rayon thread.
Negligible per-instance, but HiGHS internal LP representation adds
~4-5x the buffer size.

### 8.3 Scaling table

| Config                 | Cuts/stage |    Pool | RowBatch |   Basis |        Total |
| ---------------------- | ---------: | ------: | -------: | ------: | -----------: |
| This run (5F×50I)      |        250 |  0.2 GB |   0.4 GB |  0.0 GB |   **0.6 GB** |
| Prod small (20F×200I)  |      4,000 |  3.9 GB |   5.8 GB |  0.1 GB |   **9.8 GB** |
| Prod medium (50F×200I) |     10,000 |  9.7 GB |  14.5 GB |  0.6 GB |  **24.8 GB** |
| Prod large (200F×300I) |     60,000 | 58.3 GB |  87.0 GB | 11.1 GB | **156.6 GB** |

### 8.4 Memory constraints assessment

**Prod small (4,000 cuts/stage): ~10 GB** — fits comfortably in any
modern workstation (typically 32-128 GB). No issue.

**Prod medium (10,000 cuts/stage): ~25 GB** — fits in a 32 GB machine
with some pressure, comfortable on 64 GB. Borderline on some HPC nodes
(many have 32 GB/node). Could be problematic if running alongside
other processes.

**Prod large (60,000 cuts/stage): ~157 GB** — exceeds typical
workstation memory. Requires a large-memory HPC node (256+ GB) or
distributed memory across MPI ranks. **This is a hard constraint.**

### 8.5 RowBatch is the largest single component

At every scale, RowBatch exceeds the CutPool by ~1.5x. This is
surprising because the RowBatch is a **derived view** — it merely
re-formats the CutPool data into CSR with negated/scaled coefficients.

The redundancy structure:

- `col_indices`: identical for every cut (always `[0, 1, ..., n_state, theta_col]`).
  At C=10,000: 42 MB/stage of repeated index patterns.
- `values`: negated and column-scaled copy of CutPool coefficients.
  Could be computed on-the-fly during `add_rows` instead of pre-materialized.

### 8.6 Basis Store becomes significant at production large

At 200 forward passes × 117 stages × 2 stores × 249 KB/entry = 11.1 GB.
The basis grows with row count (template + cuts), so it scales with C.
This is the third-largest component and can be mitigated by:

- Sharing basis across scenarios at the same stage (S1 enables this)
- Compressing basis (most entries are `BASIC` = 0, run-length compressible)
- Evicting old bases (LRU or generation-based)

### 8.7 Sparsity opportunity (S4)

With 29% structural sparsity from uniform PAR stride padding,
sparse cut coefficients would reduce memory by ~15%:

| Config      | Dense total | Sparse total | Savings |
| ----------- | ----------: | -----------: | ------: |
| Prod small  |      9.6 GB |       8.2 GB |  1.5 GB |
| Prod medium |     24.1 GB |      20.4 GB |  3.7 GB |
| Prod large  |    144.7 GB |     122.5 GB | 22.2 GB |

However, the 29% sparsity from PAR padding is only one source.
Additional sparsity from hydraulically disconnected state variables
(e.g., a cut generated at a trial point where only 20 of 158 hydros
are binding) could push sparsity much higher. Empirical measurement
of actual coefficient sparsity is needed.

### 8.8 Impact of strategies on memory

| Strategy                    |    Pool     |       RowBatch        |      Basis       |       Net       |
| --------------------------- | :---------: | :-------------------: | :--------------: | :-------------: |
| **S1** (model persist)      |    same     |         same          | slightly smaller |       ~0%       |
| **S2** (incremental cuts)   |    same     |    **eliminated**     |       same       |    **-60%**     |
| **S3** (buffer reuse)       |    same     | same (no alloc churn) |       same       |       ~0%       |
| **S4** (sparse coeffs)      | -15 to -70% |      -15 to -70%      |       same       | **-15 to -70%** |
| **S5** (cache active count) |    same     |         same          |       same       |       ~0%       |

**S2 is the memory-critical strategy.** If model persistence (S1) is
implemented with incremental cut addition (S2), the RowBatch is no
longer materialized at all — new cuts are appended directly to the
live LP. This eliminates the single largest memory component.

At prod large: RowBatch is 87 GB of the 157 GB total. Eliminating it
drops memory to ~70 GB — within reach of a 128 GB HPC node.

### 8.9 Revised priority considering memory

The time-based priority from Section 6 placed S1 first and S2 fourth.
Memory analysis changes this picture:

| Priority | Strategy                  |      Time impact       |              Memory impact              |
| :------: | ------------------------- | :--------------------: | :-------------------------------------: |
|  **1**   | **S1: Model persistence** | 98% add_rows reduction |                  none                   |
|  **2**   | **S2: Incremental cuts**  |   marginal after S1    | **eliminates RowBatch (60% of memory)** |
|  **3**   | S4: Sparse coefficients   |     NNZ reduction      |       15-70% pool+batch reduction       |
|  **4**   | S3: Buffer reuse          |      alloc churn       |                  none                   |
|  **5**   | S5: Cache active count    |         minor          |                  none                   |

S2 moves up to second priority because it addresses the **memory wall**
that would block production large runs. S1 alone saves time but does not
save memory — you still materialize the full RowBatch. S1+S2 together
save both time and memory.

For production medium (25 GB), S1 alone is sufficient if the machine has
64 GB. For production large (157 GB), S1+S2 is required to fit in a
128 GB node (drops to ~70 GB), and S4 may be needed to reach 32 GB nodes.
