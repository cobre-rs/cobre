# Design: Work-Stealing Parallelism Within MPI Ranks

## Context

The current SDDP training loop uses static partitioning to distribute scenarios
(forward pass) and trial points (backward pass) across worker threads within each
MPI rank. The `partition(n_items, n_workers, worker_id)` function assigns each
worker a contiguous, pre-determined range. This was chosen for deterministic
reproducibility: the same scenario always maps to the same worker.

However, static partitioning creates load imbalance when LP solve times vary
across scenarios. Workers with harder LPs (more simplex iterations, worse
warm-start acceptance) become bottlenecks while other threads idle at the rayon
barrier. The imbalance is amplified by the backward pass, where each trial point
requires `n_openings` LP solves — so a single slow trial point blocks other
threads for an extended period.

This design introduces dynamic work-stealing for the backward pass within each
MPI rank while preserving bit-for-bit reproducibility of final results. The
forward pass retains static partitioning for now, with future work-stealing
options documented for later evaluation.

### Why Work-Stealing Is Correct

The results of both passes are order-independent with respect to which thread
processes which work item:

1. **Forward pass:** Noise samples are derived from `(iteration, global_scenario,
stage)` — the seed is a function of the scenario index, not the worker.
   Trajectory records are indexed by scenario. Which thread solves which scenario
   does not affect the noise, state propagation, or stored records.

2. **Backward pass:** Each worker generates `StagedCut` objects into a thread-local
   buffer. After the parallel region, staged cuts are sorted by `trial_point_idx`
   (`backward.rs:746`) and inserted into the FCF in that deterministic order. The
   sort guarantees identical results regardless of which thread generated which cut.

3. **Synchronization:** The forward pass has a sync step (`sync_forward`) after all
   scenarios complete. The backward pass has per-stage synchronization (cut sync via
   `allgatherv`). Both operate on completed results, not on in-progress work.

4. **Independence:** Each forward scenario traverses all stages independently. Each
   backward trial point produces one cut per stage independently. There are no
   cross-scenario or cross-trial-point data dependencies within a pass.

### Scope

This design covers intra-rank parallelism only. MPI-level distribution (across
ranks) remains static — each rank receives `ceil(N / n_ranks)` forward passes
and processes the corresponding trial points in the backward pass.

### Relationship to LP Growth

Work-stealing addresses load imbalance (secondary performance factor). The
dominant performance issue — superlinear LP growth from `iteration × forward_passes`
accumulated cuts — requires separate mitigation (cut selection, model persistence).
Work-stealing and LP growth mitigation are orthogonal and complementary.

## Decision

### Backward Pass: Atomic Counter Work-Stealing

The backward pass adopts an atomic counter pattern for trial point distribution.
This is simpler than the forward pass because:

- `BasisStore` is read-only (`&BasisStore`) — no mutable splitting needed.
- Cuts are collected into per-worker `Vec<StagedCut>` — no shared write targets.
- Each worker owns its `SolverWorkspace` exclusively — no solver sharing.

**Mechanism.** Replace the static `partition()` call inside `process_stage_backward`
with a shared `AtomicUsize` counter. Each worker loops: fetch-and-increment the
counter, process the trial point if within range, repeat until exhausted.

```rust
// Current: static partition
let (start_m, end_m) = partition(local_work, n_workers, worker_id);
for m in start_m..end_m { ... }

// Proposed: atomic counter
loop {
    let m = next_trial.fetch_add(1, Ordering::Relaxed);
    if m >= local_work { break; }
    staged.push(process_trial_point_backward(ws, ..., m, ...)?);
}
```

**Properties:**

- **Correctness:** Each trial point is processed exactly once. The `fetch_add` is
  atomic, so no two workers receive the same index.
- **Reproducibility:** The FCF insertion order depends on `trial_point_idx`, not
  on which thread produced it. The existing sort at `backward.rs:746` is retained.
- **Load balancing:** Fast-finishing workers steal work from the shared counter,
  eliminating barrier stall from LP solve time variance.
- **No locks:** `Ordering::Relaxed` is sufficient because the counter is the only
  shared state and there is no dependent memory ordering requirement. Each worker's
  solver workspace, staged cut buffer, and accumulator are thread-local.
- **Overhead:** One atomic increment per trial point. With typical `local_work` of
  20–200 trial points, the overhead is negligible compared to LP solve time.

**Changes required:**

1. `process_stage_backward` in `backward.rs`: replace `partition()` + range loop
   with atomic counter loop.
2. The `TrialAccumulators` allocation moves inside the loop body (already per-stage)
   — no change needed.
3. The `load_backward_lp` call must happen before the first trial point. With
   work-stealing, every worker must call it unconditionally (currently guarded by
   `if start_m < end_m`). When `local_work < n_workers`, some workers will load the
   LP and then immediately find no work — acceptable overhead.

### Forward Pass: Deferred (Static Partitioning Retained)

The forward pass retains static partitioning. Work-stealing is deferred to a
future design cycle pending backward pass performance measurements.

**Rationale for deferral:**

1. **Backward pass dominates wall-clock time.** Each trial point in the backward
   pass requires `n_openings` LP solves (typically 20–100) per stage, versus one
   solve per scenario per stage in the forward pass. Load imbalance in the backward
   pass therefore has a proportionally larger wall-clock impact.

2. **Implementation complexity.** The forward pass has two Rust borrow-checker
   constraints that the backward pass avoids:
   - **TrajectoryRecord mutable slices** (`forward.rs:987-994`): records are
     pre-split into per-worker `&mut [TrajectoryRecord]` via `split_at_mut`.
     Work-stealing requires dynamic scenario-to-worker assignment, which cannot be
     verified by Rust's static borrow analysis. Any solution requires either
     `unsafe` (via `UnsafeCell`) or explicit synchronization (mutexes, which are
     unacceptable on the hot path).

   - **BasisStore mutable slices** (`workspace.rs:336-356`): the basis store is
     split via `split_workers_mut` into `BasisStoreSliceMut` per worker. Work-
     stealing requires any worker to write bases for any scenario, breaking the
     disjoint-slice model.

3. **Sum-of-maxima penalty.** The most natural work-stealing design for the
   forward pass — moving the stage loop outside the parallel region so each stage
   is a separate rayon `par_iter_mut` — introduces a per-stage barrier. The
   forward pass wall-clock time becomes `sum_t(max_w(T_w,t))` (sum of per-stage
   maxima) instead of the current `max_w(sum_t(T_w,t))` (max of per-worker
   totals). By the minimax inequality, `sum(max) >= max(sum)`, so the stage-
   external approach has a strictly worse theoretical bound when LP solve time
   variance is high across stages. This penalty can exceed the work-stealing
   benefit, particularly with many stages (120+) and high per-solve variance.

**Measure first:** After implementing backward pass work-stealing, measure:

- `rayon_overhead_time_ms` before/after (backward pass diagnostic).
- Forward pass wall-clock vs. backward pass wall-clock per iteration.
- Per-worker solve time variance in the forward pass (to quantify imbalance).

If forward pass load imbalance is significant after the backward pass improvement,
revisit the forward pass design using the options documented in the appendix.

## Forward Pass Options (Appendix — Future Reference)

The following options were evaluated during design and are preserved here for
future reference. None are selected for implementation at this time.

### Option A: Stage-External with Per-Stage Work-Stealing

Move the stage loop outside the parallel region. Each stage is a separate rayon
`par_iter_mut` with an atomic counter for scenario stealing.

```rust
for t in 0..num_stages {
    let counter = AtomicUsize::new(0);
    workspaces.par_iter_mut().for_each(|ws| {
        ws.solver.load_model(&templates[t]);
        ws.solver.add_rows(&cut_batches[t]);
        loop {
            let m = counter.fetch_add(1, Relaxed);
            if m >= forward_passes { break; }
            // read state from records[m, t-1]  (safe: barrier between stages)
            // patch bounds, solve, write record, write basis
        }
    });
    // implicit barrier: par_iter_mut completion = happens-before edge
}
```

**Pros:**

- Clean separation: one parallel region per stage.
- Free barrier between stages via rayon completion semantics — guarantees state
  visibility across workers without explicit synchronization.
- Model persistence preserved (load once per worker per stage).
- Workspace access is safe (`par_iter_mut` assigns one workspace per thread).

**Cons:**

- `num_stages` rayon barriers → sum-of-maxima penalty (see rationale above).
- Still needs `UnsafeCell`-based wrapper (`SlottedArray`) for records and basis
  store, since any worker might write to any scenario's slot within a stage.
- Overhead: `num_stages` barrier entries (~1.2ms for 120 stages at ~10μs each).
  Negligible in absolute terms, but the sum-of-maxima penalty is the real cost.

**When to revisit:** If forward pass load imbalance is measured to be significant
AND the per-stage variance is low (meaning the sum-of-maxima penalty is small).

### Option B: Batched Work-Stealing (Epoch-Based)

Compromise between stage-external and static: process K stages per batch, with
work-stealing at batch boundaries. Workers claim scenarios via atomic counter
before each batch, then process them in stage-major order within the batch.

```rust
workspaces.par_iter_mut().for_each(|ws| {
    for batch in 0..num_batches {
        let t_start = batch * batch_size;
        let t_end = min(t_start + batch_size, num_stages);

        // Phase 1: claim scenarios (fast)
        let mut my_scenarios: Vec<usize> = Vec::new();
        loop {
            let m = batch_counters[batch].fetch_add(1, Relaxed);
            if m >= forward_passes { break; }
            my_scenarios.push(m);
        }

        // Phase 2: barrier (wait for previous batch to complete)
        if batch > 0 { batch_barriers[batch - 1].wait(); }

        // Phase 3: stage-major processing (model persistence preserved)
        for t in t_start..t_end {
            ws.solver.load_model(&templates[t]);
            ws.solver.add_rows(&cut_batches[t]);
            for &m in &my_scenarios {
                // solve(m, t), write record, write basis
            }
        }
    }
});
```

**Pros:**

- Single `par_iter_mut` — one rayon parallel region for the entire forward pass.
- Only `T/K` barriers instead of `T` (e.g., 6 barriers for K=20, stages=120).
- Emergent load balancing: fast workers from batch B reach batch B+1's atomic
  counter first and claim more scenarios; slow workers claim fewer.
- Model persistence preserved within each batch.

**Cons:**

- Requires `std::sync::Barrier` for explicit synchronization within the parallel
  region (rayon's implicit barrier only applies at `par_iter_mut` completion).
- Still needs `SlottedArray` for records and basis store (scenarios may move
  between workers across batches).
- Added complexity: batch size tuning, per-batch counters, explicit barriers.
- The `batch_size` parameter (K) controls the rebalancing/penalty tradeoff:
  - K = num_stages → degenerates to static partitioning (no rebalancing).
  - K = 1 → degenerates to stage-external (full sum-of-maxima penalty).
  - K = 10–20 → moderate rebalancing with few barriers.

**When to revisit:** If backward pass work-stealing is insufficient and forward
pass measurements show persistent load imbalance that varies across stage batches.

### Option C: SlottedArray (Encapsulated UnsafeCell)

A shared utility for any forward pass work-stealing option that requires dynamic
scenario-to-worker assignment. Encapsulates `UnsafeCell` in a small, auditable
abstraction.

```rust
/// A flat array that allows concurrent mutable access to disjoint slots.
///
/// The `unsafe` is confined to this module. Callers use safe APIs only.
/// The safety invariant — each slot is accessed by at most one thread at
/// a time — is enforced by the atomic work counter, not by this type.
struct SlottedArray<T> {
    slots: Box<[UnsafeCell<T>]>,
}
unsafe impl<T: Send> Sync for SlottedArray<T> {}
```

This is a building block, not a standalone option. Both Option A and Option B
would use `SlottedArray` (or equivalent) for records and basis store access.

Requires `#[allow(unsafe_code)]` on the containing module, with `// SAFETY:`
comments documenting the disjoint-access invariant.

## Consequences

### Benefits

- Eliminates load-imbalance barrier stall in the backward pass (immediate win).
- The backward pass dominates training wall-clock time due to the `n_openings`
  multiplier, so even a modest imbalance reduction yields significant speedup.
- No change to MPI-level distribution or algorithm semantics.
- Bit-for-bit reproducibility preserved (backward pass sort guarantees FCF
  insertion order).

### Costs

- Backward pass: minimal — one `AtomicUsize` per stage, `Ordering::Relaxed`,
  no locks, no additional `unsafe`.
- The `rayon_overhead_ms` diagnostic (F2-001) should be updated to measure only
  the parallel region, excluding post-parallel sequential work, so the load-
  balancing improvement is accurately reflected.

### Migration

1. Implement backward pass work-stealing (this design).
2. Measure forward vs. backward pass wall-clock times and per-worker variance.
3. If forward pass imbalance is significant, revisit the appendix options with
   measurement data to guide the design choice.

## Status

- Backward pass: **ready for implementation**.
- Forward pass: **deferred** — static partitioning retained; options documented
  in appendix for future evaluation pending performance measurements.
