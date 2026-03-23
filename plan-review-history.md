# LP Setup Optimization Plan Review

**Reviewed by:** Claude Opus 4.6 (1M context)
**Date:** 2026-03-23
**Profiling baseline:** `docs/designs/lp-setup-profiling-results.md`
**Plan location:** `plans/lp-setup-optimization/`

---

## Methodology

For each epic and ticket, the review follows the protocol from `plan-review.md`:

1. Read the ticket specification
2. Consult the relevant source code to verify claims about current behavior
3. Cross-reference with cobre-docs specs and cobre documentation
4. Evaluate against major requirements: bit-for-bit reproducibility, architectural quality, subcrate independence
5. Assess whether proposed actions target the measured bottlenecks
6. Document findings and improvement suggestions

---

## Master Plan Assessment

### Profiling Data Validation

The profiling results establish two empirical cost models via linear regression (R^2 > 0.98):

| Component         | Model               | Unit      |
| ----------------- | ------------------- | --------- |
| `add_rows`        | `0.01278 * C`       | ms per LP |
| `solve` (simplex) | `8.60 + 0.1048 * C` | ms per LP |

Where C = active cuts per stage. The asymptotic `add_rows/solve` ratio converges to `0.01278 / 0.1048 = 12.2%` as C grows. This means setup overhead stabilizes around 10-11% of total per-LP cost at production scale -- significant in absolute terms but bounded in relative terms.

**Verified:** The cost models are internally consistent. At C=250 (end of the profiled run), `add_rows = 3.2ms` and `solve = 34.8ms`, giving setup% = 9.2% -- matching the profiling table. The extrapolation to C=10,000 (prod medium) yields `add_rows = 128ms` and `solve = 1,057ms`, keeping the ratio at ~11%.

### Priority Ordering Validation

The revised priority (Section 8.9 of profiling results) factors both time and memory:

1. **S1 (Model Persistence):** 98% `add_rows` reduction. Justified.
2. **S2 (Incremental Cuts):** Eliminates RowBatch (60% of memory). Justified.
3. **S4 (Sparse Coefficients):** 15-70% memory reduction. Justified (deferred).
4. **S3 (Buffer Reuse):** Allocation churn only. Justified as low-priority.
5. **S5 (Cache Active Count):** O(1) vs O(N) scan. Trivial but useful.

The plan groups these into four epics, reordering S3/S5 before S2 because they are simpler and don't depend on S1. This is pragmatic.

### Dependency Graph Validation

```
Epic 1 (S1) ──> Epic 3 (S2) ──> Epic 4 (S4)
Epic 2 (S3 + S5)  [plan says parallel with Epic 1]
```

The plan states Epics 1 and 2 can execute in parallel. **This is technically correct but practically risky** -- see Improvement #1 below.

---

## Epic 1: Model Persistence (S1) -- Review

### Ticket 001: Restructure Forward Pass

**Code verification (forward.rs):**

- Lines 488-489: CONFIRMED. `ws.solver.load_model(&ctx.templates[t])` and `ws.solver.add_rows(&cut_batches[t])` are called inside `run_forward_stage`, which runs per `(scenario, stage)`.
- Lines 728-731: CONFIRMED. Cut batches are built once per `run_forward_pass` call (before the scenario loop), NOT per scenario. So the batch construction is already efficient -- only the LP loading/row addition is repeated.
- Lines 759-789: CONFIRMED. Current loop is scenario-outer (`for (local_m, m) in ...`), stage-inner (`for t in 0..num_stages`).

**Loop inversion analysis:**

The plan proposes inverting to stage-outer, scenario-inner:

```rust
for t in 0..num_stages {
    load_model + add_rows once
    for (local_m, m) in assigned_scenarios {
        restore state, patch bounds, solve
    }
}
```

This changes the processing order: currently scenario 0 processes stages 0..T, then scenario 1 processes stages 0..T. After the change, all scenarios process stage 0, then all process stage 1, etc.

**Correctness of state restoration:**

- `run_forward_stage` saves outgoing state to `rec.state` (lines 609-614) and applies lag shift (lines 617-628) to both `rec.state` and `ws.current_state`.
- The plan proposes restoring from `records[local_m * num_stages + (t - 1)].state` at each (scenario, stage) pair. This is CORRECT because `rec.state` contains the complete post-stage, post-lag-shift state vector.
- Overhead: one `copy_from_slice` of 1,106 floats (~8.8 KB) per (scenario, stage). NEGLIGIBLE compared to LP solve time (10-30ms).

**Noise sampling invariance:**

- Noise is sampled via `sample_forward(&tree_view, base_seed, i32, s32, t32, t)` (line 768). The seed depends on `(iteration, scenario, stage)`, not on execution order. VERIFIED: the loop inversion does not change noise samples.

**Basis store compatibility:**

- The basis store is split per-worker via `split_workers_mut` (line 745). Each worker writes basis at (m, t) during stage t's solve and reads it during stage (t+1)'s solve. In the stage-first loop, all writes at stage t complete before any reads at stage (t+1) begin. This is CORRECT and actually more orderly than the current scenario-first approach.

**VERDICT: SOUND. The restructuring produces bit-for-bit identical results.**

The plan's analysis in Steps 2-5 is thorough and correctly identifies the critical considerations (state restoration, lag shift path, sampling invariance). No changes needed.

---

### Ticket 002: Restructure Backward Pass

**Code verification (backward.rs):**

- Line 346: CONFIRMED. `load_backward_lp(ws, ctx, succ.cut_batch, s)` is called once per trial point in `process_trial_point_backward`.
- Lines 222-230: CONFIRMED. `load_backward_lp` calls `ws.solver.load_model(&ctx.templates[s])` + `ws.solver.add_rows(cut_batch)`.
- Lines 569-570: CONFIRMED. The cut_batch is built via `build_cut_row_batch` once per stage in the outer loop of `run_backward_pass`, then shared via `SuccessorSpec`.
- Lines 455-481: CONFIRMED. Parallel region uses `par_iter_mut().enumerate().map(...)` over workspaces.

**Bounds patching completeness:**

The plan correctly notes that `patch_opening_bounds` (lines 238-322) uses absolute writes via `set_row_bounds` and `set_col_bounds`. Between trial points, the first opening fully specifies all scenario-dependent bounds, including:

- State-fixing rows (Categories 1-2): patched with new `x_hat` per trial point
- Noise rows (Category 3): patched with opening noise
- Load rows (Category 4): patched with noise-derived load
- z-inflow rows (Category 5): patched with noise-derived inflows
- NCS column bounds: patched with noise-derived availability

**VERIFIED:** `fill_forward_patches` writes to ALL `N*(2+L)` state-fixing rows plus noise rows. There is no bound leakage between trial points.

**Basis management:**

- Lines 342, 352-356: `working_basis` is initialized to `None` at the start of each trial point. It does NOT carry over from the previous trial point. VERIFIED: basis is reset per trial point regardless of LP persistence.

**Edge case handling:**

- The `if start_m < end_m` guard correctly skips LP loading for empty workers.

**VERDICT: SOUND. The simplest of the three restructuring tickets because the backward pass already persists across openings -- this extends persistence to across trial points.**

---

### Ticket 003: Restructure Lower Bound Evaluation

**Code verification (lower_bound.rs):**

- Lines 166-169: CONFIRMED. `solver.load_model(template)` and `solver.add_rows(&cut_batch)` are inside the `for opening_idx in 0..n_openings` loop.
- Line 141: CONFIRMED. `cut_batch` is built once before the loop via `build_cut_row_batch`.

**Bounds patching per opening:**

- Lines 191-208: `fill_forward_patches` + `fill_z_inflow_patches` + `set_row_bounds` -- absolute writes.
- Lines 210-232: `transform_ncs_noise` + `set_col_bounds` -- absolute writes per opening.
- NCS column indices and lower bounds are pre-populated once before the loop (lines 149-163), only upper bounds change per opening. VERIFIED.

**Basis management:**

- Lower bound uses cold-start (`solver.solve()` with no basis argument, line 234). Moving model loading outside the loop has ZERO effect on basis behavior.

**Important note about the separate solver:**
The lower bound evaluation uses a SEPARATE solver instance (`solver` parameter in training.rs line 546), not the pool workspaces used by forward/backward passes. This matters for Epic 3 (cross-iteration persistence) -- see note under Epic 3 review.

**VERDICT: SOUND. The simplest ticket in Epic 1. Moving two lines before the loop.**

---

### Ticket 004: Integration Test

**Assessment:**

The test design is reasonable: run a small deterministic case, verify bit-for-bit convergence, and check solver statistics for reduced setup overhead.

**Concern:** The ticket notes that `SolverStatistics` currently tracks `total_load_model_time_seconds` (timing) but not call counts. The ticket suggests "you may need to add a `load_model_count` field." This is the right instinct -- timing comparisons are fragile (system load, thermal throttling), while call count assertions are deterministic. Adding `load_model_count` and `add_rows_count` to `SolverStatistics` would strengthen the test significantly.

**VERDICT: SOUND. Minor enhancement: recommend adding call count fields to `SolverStatistics` for deterministic assertions.**

---

## Epic 2: Buffer Reuse and Caching (S3 + S5) -- Review

### Ticket 001: Cache Active Count (S5)

**Code verification (cut/pool.rs):**

- Lines 276-282: CONFIRMED. `active_count()` scans `self.active[..self.populated_count]`.
- Lines 201-236: CONFIRMED. `add_cut` sets `self.active[slot] = true`.
- Lines 303-311: CONFIRMED. `deactivate` sets `self.active[i] = false` without counting.

**Slot uniqueness invariant:**
The plan correctly notes that each `(iteration, forward_pass_index)` pair maps to a unique slot (via `slot_index` at lines 155-172: `warm_start_count + iteration * forward_passes + forward_pass_index`). No slot is populated twice, so `add_cut` always transitions inactive -> active. The proposed `debug_assert!(!self.active[slot])` in `add_cut` correctly guards this invariant.

**Deactivation correctness:**
The plan correctly modifies `deactivate` to only decrement for slots transitioning active -> inactive (`if i < self.capacity && self.active[i]`). This handles the edge case where a slot might be deactivated multiple times (e.g., if the same index appears twice in the deactivation set, though this shouldn't happen in practice).

**Debug assertion:**
The `debug_assert_eq!` comparing cached vs computed count provides a safety net during development with zero production overhead. This is appropriate.

**VERDICT: SOUND. Trivially correct, well-guarded.**

---

### Ticket 002: Pre-Allocated CutRowBuilder (S3)

**Design evolution in the ticket:**

The ticket walks through several alternatives before settling on the final design:

1. `CutRowBuilder` with `as_row_batch()` via clone -- correctly rejected (defeats reuse purpose)
2. `take_row_batch()` via `mem::take` -- correctly rejected (loses capacity)
3. Borrowing view -- correctly rejected (`RowBatch` owns its vecs)
4. **`build_cut_row_batch_into(&mut RowBatch)`** -- final choice, correct

**Infrastructure crate impact:**
Adding `RowBatch::clear()` to `cobre-solver/src/types.rs` is appropriate. `RowBatch` is a generic LP data structure; `clear()` is a standard collection method. No algorithm-specific references are introduced. Passes the infrastructure crate genericity requirement.

**VERDICT: SOUND. The final design is correct. Minor suggestion: state the final design upfront in the ticket, with the design alternatives in a "Considered Alternatives" section, to reduce confusion for the implementer.**

---

### Ticket 003: Wire Builder Into Passes

**Assessment:**

This is straightforward plumbing: pre-allocate `Vec<RowBatch>` before the training loop, pass as `&mut [RowBatch]` to each pass function, use `build_cut_row_batch_into` inside.

**Signature change impact:**

- `run_forward_pass`: adds `cut_batches: &mut [RowBatch]` parameter
- `run_backward_pass`: adds equivalent parameter (or uses pre-allocated per-stage batch)
- `evaluate_lower_bound`: adds `&mut RowBatch` parameter

All test call sites must be updated. The ticket correctly identifies this.

**Interaction with Epic 1:**
If Epic 1 (loop restructuring) is done in parallel with Epic 2, this ticket creates merge conflicts: both change the signatures and internals of `run_forward_pass`, `run_backward_pass`, and `evaluate_lower_bound` in overlapping ways. This reinforces the recommendation to execute sequentially.

**VERDICT: SOUND. Straightforward plumbing change.**

---

## Epic 3: Incremental Cut Management (S2) -- Review

### Ticket 001: CutRowMap

**Assessment:**

The data structure design is sound. The open questions are well-identified:

1. **Vec vs HashMap:** The plan correctly leans toward `Vec<Option<usize>>`. The slot space is bounded by `capacity` (pre-allocated), making a flat vec optimal. Cache-friendly, O(1) lookup.

2. **Cut selection interaction:** Bound zeroing must be compatible with `CutSelectionStrategy::select_for_stage()` which returns `DeactivationSet { indices: Vec<u32> }`. The `CutRowMap` needs to translate slot indices to LP row indices for `set_row_bounds` calls. This is straightforward.

3. **HiGHS row index stability:** This is the CRITICAL open question. HiGHS appends rows at the end via `Highs_addRows`. After adding N rows to a model with R rows, the new rows are at indices R, R+1, ..., R+N-1. Subsequent `changeRowBoundsBySet` does NOT change indices. This is standard HiGHS behavior and is VERIFIED by the existing codebase (the forward/backward passes already rely on this when patching cut row duals at `succ.template_num_rows..template_num_rows + num_cuts_at_successor` in backward.rs lines 393-401).

**VERDICT: SOUND. The HiGHS row stability question can be resolved with confidence -- the existing code already depends on this behavior.**

---

### Ticket 002: Append New Cuts

**Assessment:**

The approach is correct: after the backward pass + cut sync, build a small RowBatch containing only the 1-50 new cuts per stage, call `solver.add_rows(&new_cut_batch)`, and update `CutRowMap`.

**First-iteration handling:** The plan correctly notes that iteration 1 must do a full build (no prior LP to append to). This is the initialization case.

**Memory savings:** With incremental addition, the full RowBatch materialization is eliminated. At prod medium (10,000 cuts, 117 stages), this saves 14.5 GB of RowBatch memory. The only RowBatch created is the small per-iteration new-cuts batch (~1-50 cuts x 1,107 NNZ x 12 bytes = ~50 KB). This is a 99.9%+ memory reduction for the RowBatch component.

**VERDICT: SOUND at outline level.**

---

### Ticket 003: Bound Zeroing for Cut Deactivation

**Assessment:**

Setting deactivated cut row bounds to `(-inf, +inf)` makes the constraint non-binding (it becomes a free row in LP terms). This avoids LP row index shifts that would occur with row deletion.

**Key concern: LP bloat from phantom rows.**

When cuts are deactivated via bound zeroing, they remain in the LP matrix as free rows. Over many iterations with active cut selection:

- If 10% of cuts are deactivated per selection cycle, after 200 iterations with cut selection every 10 iterations, the LP accumulates phantom rows.
- At prod large (60,000 active cuts), with 20 selection cycles deactivating 10% each: up to ~6,000 phantom rows per stage.
- These phantom rows have NNZ entries in the constraint matrix. HiGHS presolve would remove them, but presolve only runs on `load_model`, NOT on bound changes.

**The plan correctly identifies this risk** in the "Open Questions" section but doesn't resolve it.

**Resolution options:**

1. **Accept phantom rows:** At 6,000 phantom rows out of ~62,000 total rows (template + cuts), the overhead is ~10% extra simplex work. Acceptable for most cases.
2. **Periodic LP rebuild:** Every K iterations (e.g., every 50), do a full LP rebuild to purge phantom rows. This combines the benefits of incremental updates with periodic cleanup.
3. **Use `Highs_deleteRows`:** HiGHS supports row deletion, but it shifts indices. The `CutRowMap` would need a full rebuild after deletion. More complex but keeps the LP clean.

**Recommendation:** Option 2 (periodic rebuild) is the best balance. Add a configuration parameter `incremental_rebuild_interval` (default: every 50 iterations or after cut selection) that triggers a full LP rebuild. This is simple to implement and bounds the phantom row accumulation.

**VERDICT: SOUND with the caveat above. See Improvement #2.**

---

### Ticket 004: Cross-Iteration LP Persistence

**Assessment:**

This is the most complex ticket in the entire plan. It changes the training loop's fundamental assumption that each iteration starts with a fresh LP.

**Worker synchronization:**
After `allgatherv` in the backward pass, all ranks have the same set of new cuts. Each worker must independently append these cuts to its own solver instance. Since the cuts are identical across workers and appended in the same order, all workers end up with the same LP structure. This is CORRECT.

**Gap: Lower bound solver persistence.**
The training loop uses a SEPARATE solver for lower bound evaluation (see training.rs lines 545-553: `evaluate_lower_bound(solver, ...)` where `solver` is NOT from the workspace pool). Epic 3's cross-iteration persistence must handle this solver separately. The plan does not explicitly mention this. See Improvement #3.

**Basis extension for new rows:**
When new rows are appended, existing bases become too short (they have row statuses for R+C\_{i-1} rows, but the LP now has R+C_i rows). HiGHS handles this via `Highs_setBasis` which can accept a basis with fewer rows than the current LP -- it assigns BASIC status to unmatched rows. **VERIFIED:** The existing `solve_with_basis` implementation already handles basis size mismatches via HiGHS's internal basis padding.

**VERDICT: SOUND at outline level, with the gap noted above.**

---

### Ticket 005: Integration Test

**Assessment:**

Standard integration test. The open question about feature flags vs saved expected values is reasonable to defer. Given that D01-D15 already provide expected values, comparing against those is sufficient.

**VERDICT: SOUND.**

---

## Epic 4: Sparse Coefficients (S4) -- Review

### Ticket 001: Measure Sparsity

**Assessment:**

Correct approach: measure before implementing. The 29% structural sparsity from uniform PAR stride padding is already known. Additional sparsity from hydraulically disconnected state variables is the unknown.

**Important detail:** The threshold for "near-zero" directly affects bit-for-bit reproducibility. See Improvement #4.

**VERDICT: SOUND.**

---

### Ticket 002: Design Sparse Storage

**Assessment:**

Key design decision: CSR-style (shared `row_starts` + flat arrays) vs per-cut sparse vectors.

For the CutPool use case, cuts are inserted individually (one at a time from the backward pass) and iterated in bulk (during `build_cut_row_batch` and `evaluate_at_state`). Per-cut sparse vectors (`Vec<(usize, f64)>`) are simpler for insertion; CSR is more cache-friendly for iteration. Given that iteration is the hot path, CSR is likely better. But the plan correctly defers this to empirical measurement.

**VERDICT: SOUND at outline level.**

---

### Ticket 003-006: Implementation and Testing

**Assessment:**

The outline tickets cover the necessary scope: storage refactor, wire format update, sparse evaluation, and integration testing.

**Key concern for Ticket 005 (sparse evaluate): Floating-point summation order.**

The sparse dot product iterates only over non-zero indices. If the dense dot product iterates over ALL indices (including zeros), the summation order differs when zero entries are skipped. However, `0.0 * x = 0.0` in IEEE 754 (no NaN or signaling behavior for finite inputs), so skipping exact zeros produces bit-identical results.

The risk arises with NEAR-zero thresholds (e.g., |c| < 1e-12). Dropping a coefficient of 1e-13 changes the sum by up to 1e-13 \* |x|, which could flip the last few bits. For bit-for-bit reproducibility, only EXACT zeros (0.0) should be dropped in the initial implementation.

**VERDICT: SOUND with the threshold caveat. See Improvement #4.**

---

## Improvements Made to the Plan

### Improvement #1: Recommend Sequential Execution of Epics 1 and 2

**What changed:** Updated the master plan's execution order recommendation.

**Reason:** Epics 1 and 2 both modify `forward.rs`, `backward.rs`, and `lower_bound.rs`. Epic 1 changes the loop structure (where `load_model`/`add_rows` are called). Epic 2, Ticket 003 changes the function signatures (adding `&mut [RowBatch]` parameters). These changes touch overlapping code regions and would create merge conflicts if developed in parallel.

**Recommended sequence:**

1. Epic 2, Ticket 001 (cache active count) -- independent quick win
2. Epic 1, all tickets -- highest impact, modifies loop structure
3. Epic 2, Tickets 002-003 (buffer reuse) -- builds on Epic 1's modified code
4. Epic 3 -- depends on Epic 1
5. Epic 4 -- depends on Epic 3

This avoids merge conflicts while preserving the quick win of S5 (cache active count) being done early.

### Improvement #2: Added Periodic LP Rebuild Strategy for Epic 3

**What changed:** Added a recommendation to Epic 3, Ticket 003 for handling LP bloat from bound-zeroed phantom rows.

**Reason:** Bound zeroing leaves deactivated cut rows in the LP as free constraints. Over many iterations with cut selection, these accumulate. Since HiGHS doesn't re-run presolve on bound changes, phantom rows add unnecessary simplex work.

**Recommendation:** Add a `periodic_rebuild_interval` (default: after each cut selection cycle, or every 50 iterations). When triggered, the training loop does a full LP rebuild (load_model + add_rows with all active cuts) to purge phantom rows. This bounds the accumulation while preserving the incremental update benefits between rebuilds.

### Improvement #3: Noted Lower Bound Solver Gap in Epic 3, Ticket 004

**What changed:** Documented that the lower bound evaluation uses a separate solver instance (not from the workspace pool) and that cross-iteration LP persistence must handle this solver separately.

**Reason:** The training loop (training.rs lines 545-553) calls `evaluate_lower_bound(solver, ...)` where `solver` is a standalone solver, not one of the pool workspaces. If Epic 3 persists LP state across iterations for the pool workspaces but rebuilds the lower bound solver from scratch each iteration, the lower bound misses the S2 optimization. Since lower bound runs once per iteration at stage 0 only, the impact is small (1 stage x 1 rebuild), but it should be documented for completeness.

### Improvement #4: Specified Exact-Zero Threshold for Epic 4

**What changed:** Added a recommendation that Epic 4 should use exact zero (0.0) as the sparsity threshold, not a near-zero tolerance.

**Reason:** For bit-for-bit reproducibility, the sparse dot product must produce the same result as the dense dot product. Skipping exact zeros is safe (0.0 \* x = 0.0 in IEEE 754). Skipping near-zero values (e.g., |c| < 1e-12) introduces a numerical approximation that breaks bit-for-bit equivalence. The 29% structural sparsity from PAR padding consists entirely of exact zeros, so this threshold captures the guaranteed sparsity.

If additional near-zero sparsity is desired (for memory savings), it should be a separate, optional enhancement with a relaxed tolerance explicitly documented as a trade-off.

---

## Requirements Compliance Check

| Requirement                  | Status | Notes                                                                                                                                                                                                                                                                     |
| ---------------------------- | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Bit-for-bit reproducibility  | PASS   | Every ticket requires D01-D15 regression pass. Noise sampling is seed-deterministic, independent of execution order. State restoration from TrajectoryRecords is exact.                                                                                                   |
| Architectural quality        | PASS   | Infrastructure crate genericity preserved (RowBatch::clear is generic). No algorithm-specific references in cobre-solver.                                                                                                                                                 |
| Subcrate independence        | PASS   | Changes contained to cobre-sddp (loop structure, cut pool) and cobre-solver (RowBatch::clear). No cross-crate coupling introduced.                                                                                                                                        |
| No unsafe code               | PASS   | No unsafe introduced. All changes are safe Rust.                                                                                                                                                                                                                          |
| No allocation on hot paths   | PASS   | S3 explicitly eliminates allocation churn. S1 reduces call frequency. S2 eliminates RowBatch materialization.                                                                                                                                                             |
| Declaration-order invariance | PASS   | Cut iteration order is determined by pool slot indices, which are deterministic from (iteration, forward_pass_index). Loop inversion in Epic 1 does not change the mathematical computation -- only the order in which independent (scenario, stage) pairs are processed. |
| No Box<dyn Trait>            | PASS   | Not introduced.                                                                                                                                                                                                                                                           |
| clippy + fmt                 | PASS   | Every ticket requires clippy + fmt checks.                                                                                                                                                                                                                                |

---

## Bottleneck-to-Solution Traceability

### Bottleneck: Per-scenario LP rebuild (load_model + add_rows)

**Measured:** At 250 cuts, `load_model` = 0.13ms and `add_rows` = 3.2ms per LP. With 5,850 forward LPs per iteration, total setup = 19.5s/iter (thread-aggregated). At prod medium (10,000 cuts, 6,000 FLPs/iter), `add_rows` alone = 768s/iter.

**Root cause:** The forward pass calls `load_model + add_rows` for every `(scenario, stage)` pair, even though the LP structure (template + cuts) is identical across scenarios at the same stage. Only the bounds differ.

**Solution (S1, Epic 1):** Restructure loops so `load_model + add_rows` is called once per stage per worker. Subsequent scenarios patch bounds only.

**Why it works:** `set_row_bounds` and `set_col_bounds` are absolute writes that fully specify the bound values for each scenario. No LP structure changes are needed between scenarios. HiGHS supports bound patching on a loaded model without re-loading.

**Expected savings:** At prod medium: from 768s/iter to 15.3s/iter for `add_rows` (98% reduction). Matches the profiling extrapolation because the per-scenario multiplier is eliminated, leaving only the per-stage cost.

### Bottleneck: RowBatch memory (60% of total at prod large)

**Measured:** At prod medium (10,000 cuts, 117 stages), RowBatch = 14.5 GB. At prod large (60,000 cuts), RowBatch = 87 GB.

**Root cause:** RowBatch materializes ALL active cuts into CSR format for `add_rows`. The `col_indices` array is 100% redundant (same pattern `[0, 1, ..., n_state, theta]` for every cut). The `values` array is a negated, scaled copy of CutPool coefficients.

**Solution (S2, Epic 3):** Incremental cut addition. Instead of materializing all active cuts into a RowBatch each iteration, only append new cuts (1-50 per iteration) directly to the live LP.

**Why it works:** After S1, the LP persists across scenarios within an iteration. S2 extends persistence across iterations. Only new cuts need a (tiny) RowBatch. Deactivated cuts are neutralized via bound zeroing without removing LP rows. The large RowBatch materialization is eliminated entirely.

**Expected savings:** RowBatch memory drops from 14.5 GB (prod medium) to ~50 KB (new-cuts-only batch). Total memory drops from 24.8 GB to ~10 GB.

### Bottleneck: Per-iteration RowBatch allocation churn

**Measured:** 5,850 allocations at ~50 KB each = 280 MB allocation churn per run.

**Root cause:** `build_cut_row_batch` allocates fresh `Vec`s every call.

**Solution (S3, Epic 2 Tickets 002-003):** Pre-allocate `Vec<RowBatch>` once, reuse via `build_cut_row_batch_into(&mut RowBatch)`.

**Why it works:** `Vec::clear()` resets length to 0 without deallocating. Subsequent `push` calls reuse the existing capacity. After the first iteration, no heap allocation occurs.

### Bottleneck: O(pool_size) active count scan

**Measured:** At 60,000 cuts, each `active_count()` call scans 60,000 boolean flags.

**Root cause:** `active_count()` iterates `self.active[..populated_count]` and counts `true` values.

**Solution (S5, Epic 2 Ticket 001):** Maintain a cached `active_count` field, updated incrementally by `add_cut` (increment) and `deactivate` (decrement). The method becomes O(1).

**Why it works:** The CutPool mutation surface is small and controlled: only `add_cut` and `deactivate` change the active flags. Both are called in well-defined points of the training loop (after backward pass and during cut selection). Maintaining the counter at these two points is trivial and correct.

### Bottleneck: Dense cut coefficient storage (29%+ exact zeros from PAR padding)

**Measured:** Convertido has 158 hydros with L_max=6, but only 105 have order >= 5. The remaining 53 hydros have zero-padded higher lag slots, creating 326 structural zeros per cut (29% of the 1,106-dimensional state vector).

**Solution (S4, Epic 4):** Store coefficients in sparse format, emitting only non-zero entries.

**Why it works:** Structural zeros from PAR padding are guaranteed exact zeros. Sparse storage saves both memory (coefficient arrays shrink by 29%+) and compute (fewer `add_rows` NNZ entries, fewer dot product operations in `evaluate_at_state`).

**Expected savings:** 15-29% at minimum (structural sparsity alone). Could reach 50-70% if hydraulic disconnection creates additional sparsity.

---

## Summary

The LP setup optimization plan is **well-founded, data-driven, and architecturally sound**. The profiling results clearly identify the bottlenecks, and each epic's proposed solution directly targets a measured cost component.

### Strengths

1. **Empirical cost models drive the priority ordering.** The linear regression fits (R^2 > 0.98) give confidence in the extrapolations.
2. **Bit-for-bit equivalence is the primary acceptance criterion** for every ticket, enforced by the D01-D15 regression suite.
3. **The dependency graph is correct:** S1 is prerequisite for S2, which is prerequisite for S4. The plan respects this.
4. **Epic 1 (detailed) and Epic 2 (detailed) tickets are implementation-ready.** Line-level code references, step-by-step instructions, and clear acceptance criteria.
5. **Epic 3 and 4 (outline) correctly identify the open questions** that need resolution before detailed ticketing.

### Risks

1. **Epic 1, Ticket 001 is the riskiest ticket** -- the forward pass loop inversion touches the core hot path. Mitigated by comprehensive regression suite.
2. **Epic 3, Ticket 003 (bound zeroing)** has an unresolved performance question about phantom row accumulation. Mitigated by the periodic rebuild recommendation (Improvement #2).
3. **Epic 4's bit-reproducibility** depends on using exact-zero thresholds. This is resolved by Improvement #4.

### Recommended Execution Order

1. **Epic 2, Ticket 001** -- Cache active count (quick win, no dependencies)
2. **Epic 1, Tickets 001-004** -- Model persistence (highest impact)
3. **Epic 2, Tickets 002-003** -- Buffer reuse (after Epic 1 to avoid conflicts)
4. **Epic 3, Tickets 001-005** -- Incremental cuts (after Epic 1)
5. **Epic 4, Tickets 001-006** -- Sparse coefficients (after Epic 3)

### Expected Cumulative Impact at Production Medium (50F x 200I)

| After             | `add_rows` /iter | RowBatch mem | Total mem | Notes                  |
| ----------------- | ---------------- | ------------ | --------- | ---------------------- |
| Baseline          | 766.8s           | 14.5 GB      | 24.8 GB   |                        |
| +S5 (cache)       | 766.8s           | 14.5 GB      | 24.8 GB   | O(1) active_count only |
| +S1 (persist)     | 15.3s            | 14.5 GB      | 24.8 GB   | 98% add_rows reduction |
| +S3 (reuse)       | 15.3s            | 14.5 GB      | 24.8 GB   | Zero alloc churn       |
| +S2 (incremental) | ~0.1s            | ~0 GB        | ~10.3 GB  | RowBatch eliminated    |
| +S4 (sparse)      | ~0.1s            | ~0 GB        | ~7-9 GB   | Pool compression       |
