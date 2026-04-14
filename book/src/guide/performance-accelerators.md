# Performance Accelerators

This chapter documents the performance optimization techniques built into
Cobre's SDDP solver. These accelerators are the result of systematic
profiling and are active by default unless noted otherwise. Understanding
them helps users interpret timing statistics, configure cut management
strategies, and diagnose performance regressions.

---

## LP Setup Optimizations

Each SDDP iteration requires solving hundreds to thousands of LP
subproblems. Minimizing per-solve overhead is critical.

### Model Persistence

The structural LP for each stage (the constraint matrix, variable bounds,
and objective coefficients) is assembled once at initialization into a
`StageTemplate`. During the training loop, the solver loads the template
once per `(worker, stage)` pair and then only patches the scenario-dependent
row bounds for each forward-pass scenario. This avoids rebuilding the
entire LP from scratch at every scenario evaluation.

The simulation pipeline uses the same pattern: a **stage-major loop** loads
the LP once per `(worker, stage)` and then iterates over scenarios, patching
bounds only. This reduces LP assembly overhead from `O(scenarios x stages)`
to `O(workers x stages)`.

### Incremental Cut Injection

Benders cuts are appended to the LP via `add_rows` without rebuilding the
structural model. A `CutRowMap` provides O(1) bidirectional mapping between
cut pool slot indices and LP row indices.

When cut selection deactivates a cut, rather than rebuilding the LP, the
cut's row bounds are zeroed out (creating a "phantom row" that is present
in the LP but non-binding). A periodic full rebuild is triggered when
phantom rows exceed 20% of total rows or 50 iterations have elapsed since
the last rebuild.

### Sparse Cut Representation

Cut coefficient vectors are stored in sparse format: only non-zero entries
are kept as `(dimension_index, value)` pairs, sorted ascending for
reproducible dot products. This is motivated by PAR lag padding, which
produces 29%+ exact zeros in coefficient vectors for systems with mixed
AR orders. Only exact zeros are dropped, preserving bit-for-bit
reproducibility with the dense representation.

### PatchBuffer Pre-Allocation

The `PatchBuffer` holds three parallel arrays (`indices`, `lower`, `upper`)
consumed by the solver's `set_row_bounds` call. It is sized once at
construction for the maximum number of patches across all stages:

| Category | Range | Content |
| --- | --- | --- |
| 1 | `[0, N)` | Storage-fixing: equality constraint at incoming storage |
| 2 | `[N, N*(1+L))` | Lag-fixing: equality constraint at AR lagged inflows |
| 3 | `[N*(1+L), N*(2+L))` | Noise-fixing: equality constraint at scenario noise |
| 4 | `[N*(2+L), N*(2+L) + M*B)` | Load balance: stochastic load demand per bus per block |
| 5 | `[N*(2+L) + M*B, ...)` | z-inflow RHS: inflow variable bounds |

Where N = hydro plants, L = max PAR order, M = stochastic load buses,
B = max blocks per stage. The buffer is reused across all iterations and
scenarios with zero hot-path allocation.

---

## Solver Safeguards

When HiGHS returns a non-terminal error (`SOLVE_ERROR` or `UNKNOWN`),
the solver automatically escalates through a **12-level retry sequence**
organized in two phases, with per-level and overall wall-clock budgets.
The caller never sees intermediate failures — only the final
`Ok(solution)` or `Err(SolverError)`.

### Phase 1 (levels 0--4): Cumulative Sequence

Each level stacks on top of the previous:

| Level | Action |
| --- | --- |
| 0 | Clear cached basis and factorization |
| 1 | Enable presolve |
| 2 | Switch to dual simplex |
| 3 | Relax feasibility tolerances (1e-6) |
| 4 | Switch to interior point method (IPM) |

### Phase 2 (levels 5--11): Extended Strategies

Each level starts from restored defaults with presolve and iteration
limits, then applies level-specific options:

| Level | Action |
| --- | --- |
| 5 | Scale strategy 3 |
| 6 | Primal simplex + scale strategy 4 |
| 7 | Scale strategy 3 + relaxed tolerances |
| 8 | Objective scale (-10) |
| 9 | Primal simplex + objective scale (-10) + bound scale (-5) |
| 10 | Objective scale (-13) + bound scale (-8) + relaxed tolerances |
| 11 | IPM + objective/bound scaling + relaxed tolerances |

**Budgets:** 15 seconds per level in Phase 1, 30 seconds per level in
Phase 2, 120 seconds overall. Iteration limits are set to
`max(100_000, 50 x num_cols)` for simplex and 10,000 for IPM.

Default solver settings are restored unconditionally after the retry loop,
regardless of outcome. The per-level retry histogram is recorded in
`SolverStatistics.retry_level_histogram` and written to
`training/solver/retry_histogram.parquet` for post-run analysis.

---

## LP Scaling

Before each stage's LP template is built, a prescaler normalizes the
constraint matrix coefficients toward 1.0, improving numerical conditioning
and reducing the need for HiGHS's internal scaling.

### Column Scaling

For each column j, the scale factor is
`1 / sqrt(max|A_ij| * min|A_ij|)` over non-zero entries. The matrix
values, objective coefficients, and column bounds are scaled in-place.
After solving, primal values are unscaled: `x_original[j] = col_scale[j] * x_scaled[j]`.

### Row Scaling

Applied after column scaling with the same geometric-mean formula per row.
After solving, duals are unscaled: `dual_original[i] = row_scale[i] * dual_scaled[i]`.

### Cost Scale Factor

A constant `COST_SCALE_FACTOR = 1000` is applied to all objective
coefficients to reduce typical monetary values from ~1e11 to ~1e8,
improving simplex numerical stability.

Because the prescaler normalizes matrix entries toward 1.0, HiGHS's
internal scaling (`simplex_scale_strategy`) is disabled by default
(set to 0). Retry levels 5+ re-enable it as a fallback.

The scaling diagnostics are written to `training/scaling_report.json`
after template construction, documenting the coefficient range before
and after scaling for each stage.

---

## Cut Management Pipeline

As training progresses, the cut pool grows and LP solve times increase.
Cobre provides a three-stage cut management pipeline to control this
growth while preserving convergence guarantees.

The pipeline runs after each iteration's backward pass and cut
synchronization:

```text
Stage 1: Strategy-based selection  (check_frequency gated)
    |
    v
Stage 2: Angular diversity pruning (check_frequency gated)
    |
    v
Stage 3: Budget enforcement        (every iteration)
```

### Stage 1: Strategy-Based Selection

Three strategies are available, configured via
[`cut_selection`](./configuration.md#cut_selection) in `config.json`:

| Strategy | Deactivation Condition | Aggressiveness |
| --- | --- | --- |
| `level1` | `active_count <= threshold` (never-binding cuts) | Least |
| `lml1` | `iteration - last_active_iter > memory_window` | Medium |
| `domination` | Dominated at all visited forward-pass trial points | Most |

All strategies respect `check_frequency`: selection runs only at
iterations that are multiples of `check_frequency`. Stage 0 is always
exempt (its cuts drive the lower bound and are never backward-pass
successors). Selection runs in parallel across stages via `rayon`.

**Dominated** selection performs `O(|active cuts| x |visited states|)` work
per stage per check. It deactivates cuts that are pointwise dominated
at every visited forward-pass state, using the visited-states archive
that is always collected during training. The `domination_epsilon`
parameter controls the tolerance for domination comparisons.

### Stage 2: Angular Diversity Pruning

A computational accelerator for geometric redundancy detection, enabled
via the `angular_pruning` configuration block:

1. **Cluster phase**: Groups active cuts by cosine similarity of their
   coefficient vectors using greedy single-linkage clustering. Cuts whose
   coefficient vectors have cosine similarity above `cosine_threshold`
   (default 0.999) are placed in the same cluster.

2. **Dominance phase**: Within each cluster, performs pairwise pointwise
   dominance verification at all visited trial points. A cut is removed
   only if it is dominated at **every** trial point.

This preserves Assumption (H2) from Guigues (2017) — near-parallel cuts
with different intercepts create crossing hyperplanes where dominance
is state-dependent, so the per-point check is essential for convergence.

**Configuration:**

```json
{
  "training": {
    "cut_selection": {
      "enabled": true,
      "method": "level1",
      "threshold": 0,
      "check_frequency": 5,
      "angular_pruning": {
        "enabled": true,
        "cosine_threshold": 0.999,
        "check_frequency": 5
      }
    }
  }
}
```

### Stage 3: Budget Enforcement

A hard-cap safety net on LP size, enabled via `max_active_per_stage`.
When the number of active cuts exceeds the budget after Stages 1 and 2,
the pool evicts cuts sorted by staleness (`last_active_iter` ascending,
then `active_count` ascending). Cuts from the current iteration are
always protected.

Unlike Stages 1 and 2, budget enforcement runs **every iteration**
(not gated by `check_frequency`).

**Configuration:**

```json
{
  "training": {
    "cut_selection": {
      "enabled": true,
      "method": "level1",
      "threshold": 0,
      "check_frequency": 5,
      "max_active_per_stage": 500
    }
  }
}
```

**Why it matters:** Empirical data from a 118-stage case (2 ranks x 3
threads) shows that high-parallelism configurations (50 forward passes x
5 iterations) accumulate 4.7x more active cuts than low-parallelism
configurations (5 forward passes x 50 iterations), making each backward
LP solve 72% more expensive. Bounding LP size makes high-parallelism
configurations viable.

### Observability

The cut management pipeline writes per-stage statistics to
`training/cut_selection/iterations.parquet` with 10 columns:

| Column | Description |
| --- | --- |
| `iteration` | Training iteration |
| `stage` | Stage index |
| `cuts_populated` | Total slots with cuts |
| `cuts_active_before` | Active cuts before selection |
| `cuts_deactivated` | Cuts deactivated by Stage 1 |
| `cuts_active_after` | Active cuts after Stage 1 |
| `selection_time_ms` | Wall-clock time for the selection |
| `active_after_angular` | Active cuts after Stage 2 (null if disabled) |
| `budget_evicted` | Cuts evicted by Stage 3 (null if disabled) |
| `active_after_budget` | Active cuts after Stage 3 (null if disabled) |

---

## Basis Warm-Start

Reusing the LP simplex basis from the previous solve dramatically reduces
the number of simplex pivots needed for subsequent solves.

### BasisStore

The `BasisStore` holds one `Basis` per `(scenario, stage)` pair in a flat
array indexed as `bases[scenario * num_stages + stage]`. Before the parallel
forward pass, the store is split into disjoint per-worker sub-views
(`split_workers_mut`) so no synchronization is needed during writes.

The `Basis` struct stores solver-native `i32` status codes directly,
enabling zero-copy warm-starts via `memcpy` — no per-element enum
translation is needed.

### Simulation Basis Broadcast

When running with MPI, rank 0's scenario-0 basis is broadcast to all
ranks before the simulation phase. This ensures all ranks warm-start
simulation from the same LP vertex, regardless of rank count.

### Basis-Aware Padding (Strategy S3)

When new cut rows are added between iterations, the cached basis has fewer
row entries than the LP requires. By default, HiGHS fills new entries with
`BASIC`, which may require discovery pivots to find the correct status.

When `basis_padding` is enabled, each new cut is evaluated at the
warm-start state. Tight or violated cuts (slack <= tolerance) are assigned
`NONBASIC_LOWER`, while slack cuts are assigned `BASIC`. This reduces
simplex discovery work after cut additions.

**Configuration:**

```json
{
  "training": {
    "cut_selection": {
      "basis_padding": true
    }
  }
}
```

This feature is disabled by default. When enabled, the counters
`basis_padding_tight` and `basis_padding_slack` in the solver statistics
track how many cuts were assigned each status.

---

## Parallel Execution

### Backward Pass Work-Stealing

The backward pass parallelizes the inner trial-point loop using atomic
counter work-stealing: each worker claims the next available trial-point
index via `AtomicUsize::fetch_add(1, Relaxed)`. This keeps all threads
busy even when trial points solve in variable time.

After the parallel region, staged cuts are sorted by `trial_point_idx`
and inserted into the FCF in deterministic order, guaranteeing bit-for-bit
identical results regardless of thread count or completion order.

### Forward Pass and Simulation

Scenarios are statically partitioned across solver workspace instances
(not rayon's default work-stealing), making the scenario-to-worker
assignment deterministic. Within each scenario, the LP is loaded once
per stage and only row bounds are patched per scenario.

### Lower Bound Parallel Evaluation

The lower bound evaluation (solving stage-0 LPs for every opening in the
tree) is parallelized across the rayon thread pool using the same atomic
work-stealing pattern as the backward pass.

### Communication-Free Seed Derivation

Forward pass noise is generated without inter-rank communication. Each
rank independently derives its noise seed from
`(base_seed, iteration, scenario, stage)` using deterministic SipHash-1-3
seed derivation. The opening tree is pre-generated once before training
and shared read-only.

---

## Memory Efficiency

### Pre-Allocation Discipline

The training loop makes no heap allocations on the hot path inside the
iteration loop. All workspace buffers are allocated once before the loop:

| Buffer | Size |
| --- | --- |
| `TrajectoryRecord` flat vec | `forward_passes x num_stages` records |
| `PatchBuffer` | `N*(2+L) + M*max_blocks` entries |
| `ExchangeBuffers` (state allgatherv) | `local_count x num_ranks x n_state` floats |
| `CutSyncBuffers` (cut allgatherv) | `max_cuts_per_rank x num_ranks x cut_wire_size` bytes |
| `ScratchBuffers` per worker | noise, inflow, lag matrix, PAR, eta, load, z-inflow buffers |
| `Basis` per worker | pre-allocated with `template_rows + max_cut_rows` entries |

### CutPool Flat Coefficient Storage

Cut coefficients are stored as a single contiguous `Vec<f64>` of size
`capacity x state_dimension` rather than a `Vec<Vec<f64>>`. This provides
cache-friendly sequential access during batch iteration (cut evaluation,
dominance checks, angular pruning) and eliminates per-cut heap allocation.

### Lazy FCF Growth

The `CutPool` grows its coefficient storage on demand using a doubling
strategy (minimum 16 slots) rather than pre-allocating to the theoretical
maximum capacity. This prevents memory exhaustion on pathological parameter
combinations (e.g., 1000 iterations x 1000 forward passes x 50 states x
120 stages would require 48 GB with eager pre-allocation).

### O(1) Active Cut Count

`CutPool` maintains a `cached_active_count` that is updated incrementally
on each activation/deactivation, making `active_count()` O(1) instead of
requiring a scan of the entire pool.

### Compile-Time Solver Dispatch

`SolverInterface` is resolved as a generic type parameter at compile time,
not as `Box<dyn SolverInterface>`. All solver calls monomorphize to direct
function calls with no virtual dispatch overhead — critical when tens of
millions of LP solves occur per training run.

---

## See Also

- [Configuration](./configuration.md#cut_selection) — cut selection and cut management configuration
- [Output Format](../reference/output-format.md) — timing, solver statistics, and cut selection output schemas
- [cobre-solver](../crates/solver.md) — solver interface and retry escalation details
- [cobre-sddp](../crates/sddp.md) — training loop architecture and data structures
