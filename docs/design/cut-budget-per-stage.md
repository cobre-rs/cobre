# Design: Cut Budget Per Stage

## Implementation Status

**IMPLEMENTED** — v0.5.0, branch `feat/tier1-tier2-correctness-and-performance`

| Component | Status |
| --- | --- |
| Config: `max_active_per_stage: Option<u32>` in `CutSelectionConfig` | DONE |
| `CutPool::enforce_budget` with eviction by `(last_active_iter ASC, active_count ASC)` | DONE |
| Current-iteration cut protection | DONE |
| Training loop integration (runs every iteration, after strategy selection) | DONE |

---


## Problem

When `forward_passes` is large relative to the number of iterations, active
cuts accumulate faster than cut selection can prune them. Each SDDP iteration
generates `forward_passes * world_size` new cuts per stage, all initially
active. Cut selection can only deactivate cuts that have been observed across
multiple iterations — it deactivates nothing in the first 1–2 iterations.

Empirical measurement (118-stage hydrothermal case, 2 ranks × 3 threads):

| Configuration | Iterations | Final active cuts | Avg active/stage | Avg ms/solve (backward) |
| ------------- | ---------- | ----------------- | ---------------- | ----------------------- |
| 5 fw × 50 it  | 50         | 3,125             | 26.7             | 7.74                    |
| 50 fw × 5 it  | 5          | 14,621            | 125.0            | 13.34                   |

Despite doing 17% fewer backward LP solves, the 50fw case spends 44% more
total CPU time in the backward pass because each solve is 72% more expensive.
Each simplex pivot costs 81 μs vs 46 μs due to the denser constraint matrix.

The solver's cut selection strategies (Level1, LML1, Dominated) control
_which_ cuts to deactivate, but none of them control _how many_ active cuts
to retain. When the generation rate exceeds the pruning rate, LP size grows
without bound.

## Decision

Add an optional **per-stage active cut budget** that enforces a hard cap on
the number of active cuts from previous iterations. When the active count
exceeds the budget after normal cut selection, the least valuable cuts are
force-evicted until the budget is satisfied.

### Configuration

```jsonc
// config.json → training.cut_selection
{
  "enabled": true,
  "method": "lml1",
  "threshold": 1,
  "check_frequency": 1,
  "max_active_per_stage": 80, // NEW — null or absent = no budget (default)
}
```

- **`max_active_per_stage`**: maximum number of active cuts retained per
  stage after budget enforcement. `null` (or omitted) disables the budget.
  The budget is a best-effort target: cuts from the current iteration are
  protected from eviction, so the active count may temporarily exceed the
  budget by up to `forward_passes * world_size`.

- The budget is orthogonal to the selection method. Any method (Level1,
  LML1, Dominated) can be combined with a budget. The method determines
  _which_ cuts are candidates for regular deactivation; the budget
  determines the ceiling after regular deactivation has run.

- **Requires `enabled: true`**. Activity metadata (`last_active_iter`,
  `active_count`, `domination_count`) is only maintained when cut selection
  is enabled. The budget relies on this metadata for eviction ordering.

### Validation

At config parse time:

- `max_active_per_stage` must be `null` or a positive integer.
- If `max_active_per_stage < forward_passes * world_size`, emit a warning:
  the budget can never be satisfied because each iteration's protected cuts
  already exceed it. The system will still function correctly — it simply
  evicts all non-protected cuts every iteration.

### Eviction Priority

When `active_count > budget` after regular selection, cuts are sorted by
eviction priority (lowest priority evicted first):

1. **Primary**: `last_active_iter` ascending — cuts that have not been
   binding for the longest time are evicted first. This is the most
   universal signal across all selection methods: a cut that hasn't been
   binding in many iterations is unlikely to contribute to the policy
   regardless of how it was originally generated.

2. **Secondary** (tie-breaker): `active_count` ascending — among cuts with
   the same `last_active_iter`, those that have been binding fewer times
   overall are evicted first.

Cuts where `iteration_generated == current_iteration` are excluded from
eviction. This is consistent with the existing behavior of all selection
methods, which skip current-iteration cuts to avoid discarding untested
information.

### Enforcement Timing

The budget is enforced once per iteration, after both regular cut selection
and cut synchronization have completed. The sequence within the training
loop is:

```
Per iteration:
  1. Forward pass  → generates trial points
  2. Backward pass → generates cuts, per-stage allgatherv sync
  3. Cut selection → strategy-based deactivation (if should_run)
  4. Budget enforcement → force-evict if active > budget  ← NEW
  5. Lower bound evaluation
  6. Convergence check
```

Step 4 runs every iteration, not gated by `check_frequency`. The budget is
a safety net — it must fire every iteration to prevent LP growth between
selection rounds.

During the backward pass (step 2), the LP may temporarily exceed the budget
because new cuts are added at each stage. This is acceptable: the budget
controls LP size at the _start_ of each iteration (steps 1–2 use the FCF
from the end of the previous iteration), not during mid-iteration
computation.

### Steady-State Behavior

After enough iterations, the system reaches a steady state where:

```
active_per_stage ≈ min(budget, natural_active_count)
```

If the natural active count (what the selection method would retain) is
already below the budget, the budget is never triggered. The budget only
activates when the selection method's pruning rate is insufficient to keep
up with the generation rate — exactly the scenario measured above.

With a budget of 80 and 50 forward passes across 2 ranks:

- Each iteration adds 100 new cuts per stage (protected)
- Budget enforcement evicts old cuts until non-protected active ≤ 80
- Effective LP size: ≤ 180 active cuts per stage (80 old + 100 new)
- Next iteration: the 100 "new" cuts become candidates; enforcement
  evicts down to 80 again

The LP size is bounded by `budget + forward_passes * world_size` per stage,
rather than growing linearly with `iterations * forward_passes * world_size`.

## Implementation

### 1. Config — `cobre-io/src/config.rs`

Add the field to `CutSelectionConfig`:

```rust
pub struct CutSelectionConfig {
    // ... existing fields ...

    /// Maximum active cuts per stage. `None` = no budget (default).
    #[serde(default)]
    pub max_active_per_stage: Option<u32>,
}
```

### 2. Strategy — `cobre-sddp/src/cut_selection.rs`

Add a budget field to each variant (or as a wrapper). The simplest approach
is a separate struct since the budget is orthogonal to the strategy:

```rust
/// Optional hard cap on active cuts per stage.
#[derive(Debug, Clone, Copy)]
pub struct CutBudget {
    pub max_active_per_stage: usize,
}
```

Add an `enforce_budget` method on `CutPool`:

```rust
impl CutPool {
    /// Force-evict the least valuable cuts until `active_count() <= budget`.
    ///
    /// Cuts from `current_iteration` are protected and cannot be evicted.
    /// Returns the number of cuts evicted.
    pub fn enforce_budget(
        &mut self,
        budget: usize,
        current_iteration: u64,
    ) -> usize {
        if self.cached_active_count <= budget {
            return 0;
        }
        let excess = self.cached_active_count - budget;

        // Collect eviction candidates: active, not from current iteration
        let mut candidates: Vec<(usize, u64, u64)> = Vec::new();
        for i in 0..self.populated_count {
            if self.active[i]
                && self.metadata[i].iteration_generated < current_iteration
            {
                candidates.push((
                    i,
                    self.metadata[i].last_active_iter,
                    self.metadata[i].active_count,
                ));
            }
        }

        // Sort by eviction priority: oldest last_active_iter first,
        // then lowest active_count first
        candidates.sort_unstable_by(|a, b| {
            a.1.cmp(&b.1).then(a.2.cmp(&b.2))
        });

        let to_evict = excess.min(candidates.len());
        for &(slot, _, _) in &candidates[..to_evict] {
            self.active[slot] = false;
            self.cached_active_count -= 1;
        }
        to_evict
    }
}
```

### 3. Training loop — `cobre-sddp/src/training.rs`

After the existing cut selection block (~line 830), add budget enforcement:

```rust
// Step 4: Budget enforcement (every iteration, independent of check_frequency)
if let Some(budget) = cut_budget {
    let mut budget_evictions = 0u32;
    for stage in 0..num_stages.saturating_sub(1) {
        let evicted = fcf.pools[stage].enforce_budget(
            budget.max_active_per_stage,
            iteration,
        );
        budget_evictions += evicted as u32;
    }
    if budget_evictions > 0 {
        // Emit event or log — budget enforcement triggered
    }
}
```

### 4. Output — `training_output.rs`

Extend the `CutSelectionComplete` event (or add a new `BudgetEnforced`
event) to report per-stage evictions from budget enforcement separately
from strategy-based deactivations. This allows diagnosing whether the
budget is actively constraining the LP.

Add a column to `cut_selection/iterations.parquet`:

| Column           | Type | Description                        |
| ---------------- | ---- | ---------------------------------- |
| `budget_evicted` | i32  | Cuts evicted by budget enforcement |

### 5. Validation — regression tests

All 26 deterministic regression tests (D01–D26) must pass unchanged when
`max_active_per_stage` is `null` (the default). A new test case should
exercise the budget with a small value (e.g., `max_active_per_stage: 3`)
and verify:

- Active cuts per stage never exceed `budget + forward_passes * world_size`
- The final policy is still valid (LP feasibility, no solver failures)
- Results differ from the uncapped case (confirming the budget is active)

## Consequences

### Benefits

- Bounds LP size to `O(budget + forward_passes * world_size)` per stage
  regardless of iteration count.
- Keeps simplex pivot cost bounded — the per-iteration cost stabilizes
  rather than growing superlinearly.
- Makes "many forward passes, few iterations" configurations viable by
  preventing LP growth from overwhelming parallelization efficiency gains.
- Zero impact when not configured: `None` means no budget, no enforcement
  step, no overhead.

### Costs

- The `enforce_budget` method is `O(populated * log(populated))` per stage
  due to sorting. With typical populated counts of 200–2000 and 117 stages,
  this is ~0.1ms total — negligible compared to LP solve time.
- Aggressive budgets discard cuts that might have been useful, potentially
  requiring more iterations to converge. The budget should be set above the
  natural active count of a well-converged run (e.g., 2–3× the steady-state
  active count observed with many iterations).
- The eviction priority (oldest `last_active_iter`) is a heuristic. It may
  not be optimal for all cases — a cut that was inactive for several
  iterations might become relevant again as the policy evolves. The LML1
  strategy already makes this tradeoff; the budget is a harder version.

### Interaction with Cut Selection Methods

| Method     | Budget behavior                                           |
| ---------- | --------------------------------------------------------- |
| Level1     | Budget catches cuts that Level1 retains indefinitely      |
| LML1       | Budget provides a hard cap; LML1 provides the soft window |
| Dominated  | Budget provides a fast fallback when domination is slow   |
| (disabled) | Budget is not available — requires `enabled: true`        |

### Sizing Guidelines

- **Conservative**: `max_active_per_stage = 3 * forward_passes * world_size`.
  Allows 3 iterations of cuts to coexist. Unlikely to affect convergence.
- **Moderate**: `max_active_per_stage = forward_passes * world_size`.
  Only one iteration of old cuts survives. More aggressive pruning.
- **Production**: measure the steady-state active count from a long run
  (e.g., 50+ iterations) and set the budget to 1.5× that value.

## Status

**IMPLEMENTED** (v0.5.0). See `lp-scalability-cut-management.md` for the
composite pipeline context (S2 runs after S1 angular pruning).
