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

Benders cuts are appended to the persistent lower-bound LP via `add_rows`
without rebuilding the structural model. A `CutRowMap` provides O(1)
slot-to-row lookup so the incremental append skips cuts that are already
present.

The LB LP is strictly append-only: rows generated during training are
appended and never removed, which keeps the lower bound monotonically
non-decreasing across iterations. Row selection in the shared row pool
still affects the forward and backward passes — pool-deactivated rows
remain as LP rows in the LB solver but are not re-evaluated, so they
contribute only their binding value at the trial point.

### PatchBuffer Pre-Allocation

The `PatchBuffer` holds three parallel arrays (`indices`, `lower`, `upper`)
consumed by the solver's `set_row_bounds` call. It is sized once at
construction for the maximum number of patches across all stages:

| Category | Range                      | Content                                                 |
| -------- | -------------------------- | ------------------------------------------------------- |
| 1        | `[0, N)`                   | Storage-fixing: equality constraint at incoming storage |
| 2        | `[N, N*(1+L))`             | Lag-fixing: equality constraint at AR lagged inflows    |
| 3        | `[N*(1+L), N*(2+L))`       | Noise-fixing: equality constraint at scenario noise     |
| 4        | `[N*(2+L), N*(2+L) + M*B)` | Load balance: stochastic load demand per bus per block  |
| 5        | `[N*(2+L) + M*B, ...)`     | z-inflow RHS: inflow variable bounds                    |

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

| Level | Action                                |
| ----- | ------------------------------------- |
| 0     | Clear cached basis and factorization  |
| 1     | Enable presolve                       |
| 2     | Switch to dual simplex                |
| 3     | Relax feasibility tolerances (1e-6)   |
| 4     | Switch to interior point method (IPM) |

### Phase 2 (levels 5--11): Extended Strategies

Each level starts from restored defaults with presolve and iteration
limits, then applies level-specific options:

| Level | Action                                                        |
| ----- | ------------------------------------------------------------- |
| 5     | Scale strategy 3                                              |
| 6     | Primal simplex + scale strategy 4                             |
| 7     | Scale strategy 3 + relaxed tolerances                         |
| 8     | Objective scale (-10)                                         |
| 9     | Primal simplex + objective scale (-10) + bound scale (-5)     |
| 10    | Objective scale (-13) + bound scale (-8) + relaxed tolerances |
| 11    | IPM + objective/bound scaling + relaxed tolerances            |

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

As training progresses, the row pool grows and LP solve times increase.
Cobre provides a two-stage row management pipeline to control this
growth while preserving convergence guarantees.

The pipeline runs after each iteration's backward pass and cut
synchronization:

```text
Stage 1: Strategy-based selection  (check_frequency gated)
    |
    v
Stage 2: Budget enforcement        (every iteration)
```

### Stage 1: Strategy-Based Selection

Three strategies are available, configured via
[`cut_selection`](./configuration.md#cut_selection) in `config.json`:

| Strategy     | Deactivation Condition                             | Aggressiveness |
| ------------ | -------------------------------------------------- | -------------- |
| `level1`     | `active_count <= threshold` (never-binding rows)   | Least          |
| `lml1`       | `iteration - last_active_iter > memory_window`     | Medium         |
| `domination` | Dominated at all visited forward-pass trial points | Most           |

All strategies respect `check_frequency`: selection runs only at
iterations that are multiples of `check_frequency`. Stage 0 is always
exempt (its rows drive the lower bound and are never backward-pass
successors). Selection runs in parallel across stages via `rayon`.

**Dominated** selection performs `O(|active rows| x |visited states|)` work
per stage per check. It deactivates rows that are pointwise dominated
at every visited forward-pass state, using the visited-states archive
that is always collected during training. The `domination_epsilon`
parameter controls the tolerance for domination comparisons.

### Stage 2: Budget Enforcement

A hard-cap safety net on LP size, enabled via `max_active_per_stage`.
When the number of active rows exceeds the budget after Stage 1, the
pool evicts rows sorted by staleness (`last_active_iter` ascending,
then `active_count` ascending). Rows from the current iteration are
always protected.

Unlike Stage 1, budget enforcement runs **every iteration** (not gated
by `check_frequency`).

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
5 iterations) accumulate 4.7x more active rows than low-parallelism
configurations (5 forward passes x 50 iterations), making each backward
LP solve 72% more expensive. Bounding LP size makes high-parallelism
configurations viable.

### Observability

The row management pipeline writes per-stage statistics to
`training/cut_selection/iterations.parquet` with 9 columns:

| Column                | Description                                  |
| --------------------- | -------------------------------------------- |
| `iteration`           | Training iteration                           |
| `stage`               | Stage index                                  |
| `cuts_populated`      | Total row slots populated                    |
| `cuts_active_before`  | Active rows before selection                 |
| `cuts_deactivated`    | Rows deactivated by Stage 1                  |
| `cuts_active_after`   | Active rows after Stage 1                    |
| `selection_time_ms`   | Wall-clock time for the selection            |
| `budget_evicted`      | Rows evicted by Stage 2 (null if disabled)   |
| `active_after_budget` | Active rows after Stage 2 (null if disabled) |

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

### Basis Reconstruction

Each stored warm-start basis is wrapped in a `CapturedBasis { basis,
base_row_count, cut_row_slots, state_at_capture }` struct that records
the LP row count and the ordered list of row-pool slot indices at
capture time, alongside the state vector at which the basis was
captured. The `reconstruct_basis` function in
`cobre-sddp::basis_reconstruct` is the sole entry point for applying a
stored basis across row-set churn on the forward pass, backward pass,
and simulation pipeline.

When a stored basis is applied to an LP whose appended rows have changed,
`reconstruct_basis` walks the current LP's appended rows, looks each slot
up in an O(1) scratch map built from `cut_row_slots`, and classifies
each row into one of two paths:

- **Preserved** (slot present in the stored basis): the original status
  is copied verbatim.
- **New** (slot not present — a row added since capture): the classifier
  consults the row's sliding bitmap of recent binding observations. If
  any bit within the `basis_activity_window` mask is set, or if the row
  was generated in the current iteration, the row is assigned
  `NONBASIC_LOWER` (tight guess); otherwise `BASIC` (slack guess).

Each `NONBASIC_LOWER` classification on a new row requires a
compensating demotion on a preserved row to keep HiGHS's
column-basic + row-basic invariant. The stalest preserved-`LOWER`
candidate is promoted, ranked lexicographically by recent-activity
popcount, last-active iteration, and insertion order. When
new-`LOWER` classifications outnumber preserved-`LOWER` candidates,
a tail fallback flips the most recent new-`LOWER` rows back to
`BASIC` until the invariant holds.

Reconstruction is always active when a stored basis exists — there is
no configuration flag.

The `basis_reconstructions` counter in
`training/solver/iterations.parquet` and
`simulation/solver/iterations.parquet` tracks how often
`reconstruct_basis` was invoked with a non-empty stored basis.

### Backward-Pass Basis Cache

During training, rank 0's ω=0 backward-pass worker captures a fresh
basis for every stage into a per-iteration backward cache. At end of
iteration the cache is broadcast to all ranks, and on the next
iteration's backward pass every rank's ω=0 solve warm-starts from the
cached basis instead of falling back to the forward-pass `BasisStore`.
The first iteration has no backward cache yet, so it uses the forward
cache exclusively.

The backward cache matters because rows added earlier in the current
iteration's backward walk are **new** relative to the previous
iteration's stored basis — so the classifier fires frequently on
backward solves, while the forward pass sees mostly preserved slots
and the classifier rarely runs. A warm-start at ω=0 also cascades
through the remaining openings (ω=1..`n_openings`-1) via HiGHS's
retained factorization, amplifying the per-solve impact.

---

## Parallel Execution

### Backward Pass Work-Stealing

The backward pass parallelizes the inner trial-point loop using atomic
counter work-stealing: each worker claims the next available trial-point
index via `AtomicUsize::fetch_add(1, Relaxed)`. This keeps all threads
busy even when trial points solve in variable time.

After the parallel region, staged rows are sorted by `trial_point_idx`
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

| Buffer                                 | Size                                                        |
| -------------------------------------- | ----------------------------------------------------------- |
| `TrajectoryRecord` flat vec            | `forward_passes x num_stages` records                       |
| `PatchBuffer`                          | `N*(2+L) + M*max_blocks` entries                            |
| `ExchangeBuffers` (state allgatherv)   | `local_count x num_ranks x n_state` floats                  |
| `CutSyncBuffers` (row-sync allgatherv) | `max_cuts_per_rank x num_ranks x cut_wire_size` bytes       |
| `ScratchBuffers` per worker            | noise, inflow, lag matrix, PAR, eta, load, z-inflow buffers |
| `Basis` per worker                     | pre-allocated with `template_rows + max_cut_rows` entries   |

### CutPool Flat Coefficient Storage

Row coefficients are stored as a single contiguous `Vec<f64>` of size
`capacity x state_dimension` rather than a `Vec<Vec<f64>>`. This provides
cache-friendly sequential access during batch iteration (row evaluation,
dominance checks) and eliminates per-row heap allocation.

### Lazy FCF Growth

The `CutPool` grows its coefficient storage on demand using a doubling
strategy (minimum 16 slots) rather than pre-allocating to the theoretical
maximum capacity. This prevents memory exhaustion on pathological parameter
combinations (e.g., 1000 iterations x 1000 forward passes x 50 states x
120 stages would require 48 GB with eager pre-allocation).

### O(1) Active Row Count

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

- [Configuration](./configuration.md#cut_selection) — row-selection and row management configuration
- [Output Format](../reference/output-format.md) — timing, solver statistics, and row-selection output schemas
- [cobre-solver](../crates/solver.md) — solver interface and retry escalation details
- [cobre-sddp](../crates/sddp.md) — training loop architecture and data structures
