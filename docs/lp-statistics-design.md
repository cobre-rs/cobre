# LP Statistics Collection — Design Specification

> **Date:** 2026-03-22
> **Status:** Implemented (v0.1.9)
> **Context:** After implementing column scaling, row scaling, cost scaling, and
> a 12-level retry policy, we need instrumentation to answer: "Did it help?",
> "Where do problems remain?", and "What's the solver bottleneck?"

---

## 1. Overview

Three new output channels, each serving a distinct purpose:

| Channel           | Format   | When                       | Purpose                         |
| ----------------- | -------- | -------------------------- | ------------------------------- |
| Scaling Report    | JSON     | Once, after template build | LP conditioning quality         |
| Solver Statistics | Parquet  | Per iteration              | Runtime solver behaviour trends |
| Enhanced Display  | CLI text | Live + final summary       | Human-readable operational view |

All three channels are additive — no existing output schemas or event types are
modified. The existing `training/convergence.parquet` and
`training/timing/iterations.parquet` remain unchanged.

---

## 2. Channel 1: Scaling Report (JSON)

### Output path

`training/scaling_report.json`

Written once after `build_stage_templates` and the column/row scaling pass in
`setup.rs`, before the training loop begins.

### Purpose

Captures the static LP conditioning state so that operators can assess whether
the scaling factors are effective without running a full training. This is a
diagnostic artifact — it does not affect solver behaviour.

### Schema

```json
{
  "cost_scale_factor": 1000.0,
  "stages": [
    {
      "stage_id": 0,
      "dimensions": {
        "num_cols": 3042,
        "num_rows": 1587,
        "num_nz": 18234
      },
      "pre_scaling": {
        "matrix_coeff_range": [0.003, 54400.0],
        "matrix_coeff_ratio": 1.81e7,
        "objective_range": [0.01, 8291.0],
        "objective_ratio": 8.29e5
      },
      "post_scaling": {
        "matrix_coeff_range": [0.12, 8.7],
        "matrix_coeff_ratio": 72.5,
        "objective_range": [0.00001, 8.291],
        "objective_ratio": 8.29e5
      },
      "col_scale": {
        "min": 0.0001,
        "max": 316.2,
        "median": 1.0,
        "count": 3042
      },
      "row_scale": {
        "min": 0.014,
        "max": 47.1,
        "median": 1.0,
        "count": 1587
      }
    }
  ],
  "summary": {
    "worst_pre_scaling_matrix_ratio": 1.81e7,
    "worst_post_scaling_matrix_ratio": 72.5,
    "improvement_factor": 2.49e5,
    "num_stages": 118
  }
}
```

### Field definitions

**Per-stage `pre_scaling`** (computed BEFORE `apply_col_scale` / `apply_row_scale`):

- `matrix_coeff_range`: `[min|A_ij|, max|A_ij|]` over all nonzero entries in
  the structural constraint matrix (excluding zeros).
- `matrix_coeff_ratio`: `max / min`. The key conditioning indicator.
- `objective_range`: `[min|c_j|, max|c_j|]` over nonzero objective coefficients.
- `objective_ratio`: `max / min`.

**Per-stage `post_scaling`** (computed AFTER both column and row scaling are
applied, and after cost scaling divides the objective by `COST_SCALE_FACTOR`):

Same four fields, recomputed on the scaled matrix and objective.

**Per-stage `col_scale` / `row_scale`**:

- `min`, `max`: extremes of the scale factor vectors.
- `median`: middle value (sort the vector, take the midpoint).
- `count`: vector length (= `num_cols` or `num_rows`).

**`summary`**:

- `worst_pre_scaling_matrix_ratio`: max of `pre_scaling.matrix_coeff_ratio`
  across all stages.
- `worst_post_scaling_matrix_ratio`: max of `post_scaling.matrix_coeff_ratio`
  across all stages.
- `improvement_factor`: `worst_pre / worst_post`.
- `num_stages`: number of study stages.

### Implementation notes

- The pre-scaling statistics must be captured between matrix assembly and the
  call to `apply_col_scale`. This requires a snapshot of the coefficient range
  before scaling is applied.
- The post-scaling statistics are computed on the final template after both
  column scaling, row scaling, and cost scaling have been applied.
- Computing the median requires sorting a copy of the scale factor vector.
  This is O(N log N) per stage but runs once at startup — negligible cost.
- The `objective_range` / `objective_ratio` in `post_scaling` reflects cost
  scaling. If all costs are divided by 1000, the ratio stays the same but the
  range shifts down by 3 orders of magnitude.

### Future extension (not in scope now)

- **Condition number**: Computing the actual condition number (κ = ‖A‖·‖A⁻¹‖)
  requires a matrix factorization and is expensive. The coefficient range ratio
  is a cheap upper bound that is sufficient for diagnostics. If precise
  conditioning metrics are needed in the future, this report is the natural
  place to add them.
- **Cut row conditioning**: As Benders cuts accumulate, the effective coefficient
  range grows. A future extension could sample the augmented matrix conditioning
  periodically (e.g., every N iterations). For now, the per-iteration retry rate
  in Channel 2 serves as a proxy for cut-induced conditioning degradation.

---

## 3. Channel 2: Solver Statistics (Parquet)

### Output path

`training/solver/iterations.parquet`

Written at the end of training alongside the existing convergence and timing
Parquet files.

### Purpose

Per-iteration, per-phase solver statistics that reveal trends over the training
run. Enables post-hoc analysis of questions like "do retries increase as cuts
accumulate?" and "which phase is the solver bottleneck?".

### Schema

One row per (iteration, phase) pair. Typically 2 rows per iteration (forward,
backward), plus 1 row for lower_bound evaluations when they occur.

| Column               | Type | Description                                                              |
| -------------------- | ---- | ------------------------------------------------------------------------ |
| `iteration`          | u32  | 1-based iteration number                                                 |
| `phase`              | utf8 | `"forward"`, `"backward"`, or `"lower_bound"`                            |
| `stage`              | i32  | Stage index for backward phase (-1 for forward/LB which span all stages) |
| `lp_solves`          | u32  | Number of LP solves in this phase                                        |
| `lp_successes`       | u32  | Solves that returned optimal on first attempt                            |
| `lp_retries`         | u32  | Solves that required retry escalation                                    |
| `lp_failures`        | u32  | Solves that exhausted all retry levels                                   |
| `retry_attempts`     | u32  | Total retry attempts (sum across all retried solves)                     |
| `basis_offered`      | u32  | Number of `solve_with_basis` calls                                       |
| `basis_rejections`   | u32  | Times the basis was rejected (cold-start fallback)                       |
| `simplex_iterations` | u64  | Total simplex iterations across all solves                               |
| `solve_time_ms`      | f64  | Cumulative wall-clock solve time in milliseconds                         |

### Derived metrics (computed by consumers, not stored)

- `avg_solve_time_us = solve_time_ms * 1000 / lp_solves`
- `retry_rate = lp_retries / lp_solves`
- `first_try_rate = lp_successes / lp_solves`
- `basis_hit_rate = 1 - (basis_rejections / basis_offered)`

### Backward pass stage tracking

The backward pass processes successor stages sequentially. Each stage's solves
(across all openings and scenarios) are tracked separately, producing one row
per (iteration, "backward", stage) triple. This enables identification of
which stages are numerically hardest.

The forward pass and lower bound evaluation span all stages in a single sweep,
so they produce one row each with `stage = -1`.

### Data flow

```
SolverStatistics (per-workspace, monotonic counters)
    ↓
Snapshot before/after each phase
    ↓
Delta computation (per phase, per iteration)
    ↓
New event variant or return struct field
    ↓
Collected in training_output bridge
    ↓
Written to Parquet at training end
```

### New fields needed on SolverStatistics

The current `SolverStatistics` struct needs two additions to support
first-try vs retry separation:

```rust
pub struct SolverStatistics {
    // existing
    pub solve_count: u64,
    pub success_count: u64,
    pub failure_count: u64,
    pub total_iterations: u64,
    pub retry_count: u64,
    pub total_solve_time_seconds: f64,
    pub basis_rejections: u64,
    // new
    pub first_try_successes: u64,   // solves optimal on first attempt
    pub basis_offered: u64,         // total solve_with_basis calls
}
```

- `first_try_successes`: incremented when the initial solve (before any retry)
  returns optimal. `lp_retries = success_count - first_try_successes`.
- `basis_offered`: incremented at entry to `solve_with_basis`. Enables
  `basis_hit_rate = 1 - basis_rejections / basis_offered`.

### Backward pass instrumentation

The backward pass loop in `backward.rs` iterates over successor stages.
To track per-stage statistics:

1. Snapshot `solver.statistics()` before the per-stage opening loop.
2. Snapshot again after all openings for that stage are solved.
3. Compute delta and store in a `Vec<SolverStatsDelta>` keyed by stage.
4. Return these deltas in `BackwardResult`.

This does NOT change the solve hot path — `statistics()` returns a clone of
a struct with 9 integer/float fields, which is negligible.

---

## 4. Channel 3: Enhanced CLI Display

### Progress bar (live, per-iteration)

Current:

```
Training   ████████░░░░░░░░░░░░░░░░░░░░░░ 42/100 iter  LB: 9.99e4  UB: 1.01e5  gap: 1.0%
```

Proposed:

```
Training   ████████░░░░░░░░░░░░░░░░░░░░░░ 42/100 iter  LB: 9.99e4  UB: 1.01e5  gap: 1.0%  LP: 0.8ms  [2m 15s < ~3m 05s]
```

Added fields:

- `LP: {avg}ms` — average LP solve time for this iteration
  (`solve_time_ms / lp_solves` from the `IterationSummary`).
- `[elapsed < eta]` — cumulative elapsed time and estimated time remaining.

The `LP` field is added to the `{msg}` portion. The `[elapsed < eta]` is
provided by indicatif's built-in `{elapsed_precise}` and `{eta_precise}`
template tokens, matching the simulation bar's existing format.

#### Implementation

Change the template constant and extend the message:

```rust
const TRAINING_TEMPLATE: &str =
    "Training   {bar:40} {pos}/{len} iter  {msg}  [{elapsed_precise} < {eta_precise}]";
```

The `{msg}` is set per-iteration from `IterationSummary` fields:

```rust
bar.set_message(format!(
    "LB: {}  UB: {}  gap: {gap_pct:.1}%  LP: {avg_lp:.1}ms",
    fmt_sci(lower_bound),
    fmt_sci(upper_bound),
));
```

This requires adding `solve_time_ms` to the `IterationSummary` event variant
(or computing the average from `solve_time_ms / lp_solves`).

#### ETA accuracy note

Indicatif's `{eta_precise}` uses a linear estimate: `elapsed / completed *
remaining`. SDDP iterations are **not** uniform — early iterations are fast
(few cuts, small LPs) and later iterations are slower (more cuts, larger LPs,
more retries). The ETA will be optimistic in early iterations and converge to
reality as iteration time stabilizes.

**Future improvement:** Replace the linear ETA with a custom estimate based on
an exponential moving average (EMA) of recent iteration times, or fit a simple
growth model to the observed `iteration_time_ms` series. The `IterationSummary`
already carries `iteration_time_ms` per iteration, so the data is available.
This is not in scope for the initial implementation.

### Final training summary

Current:

```
  LP solves:    12,000
```

Proposed:

```
  LP solves:    12,000 (11,988 first-try, 12 retried, 0 failed)
  LP time:      45.2s total, 0.8ms avg
  Basis reuse:  98.2% hit (214 rejections / 12,000 offered)
  Simplex iter: 1,234,567
```

New fields required in `TrainingSummary` (or a new `SolverSummary` substruct):

- `total_lp_solves` (already exists)
- `total_first_try` (new)
- `total_retried` (new: `success_count - first_try_successes`)
- `total_failed` (new)
- `total_solve_time_seconds` (new)
- `total_basis_offered` (new)
- `total_basis_rejections` (new)
- `total_simplex_iterations` (new)

These are aggregated from the final `solver.statistics()` snapshot across all
workspaces and (for multi-rank) via `allreduce(Sum)`.

---

## 5. Data Flow Summary

```
                    ┌─────────────────────────────────┐
                    │   build_stage_templates()        │
                    │   + apply_col_scale/row_scale    │
                    │   + cost scaling                 │
                    └──────────────┬──────────────────┘
                                  │
                    ┌─────────────▼──────────────────┐
                    │  Channel 1: Scaling Report      │
                    │  → training/scaling_report.json  │
                    └─────────────┬──────────────────┘
                                  │
                    ┌─────────────▼──────────────────┐
                    │      Training Loop              │
                    │  ┌──────────────────────────┐   │
                    │  │ Forward Pass              │   │
                    │  │  snapshot stats before    │   │
                    │  │  ... solve N LPs ...      │   │
                    │  │  snapshot stats after     │   │
                    │  │  → delta (1 row)          │   │
                    │  └──────────────────────────┘   │
                    │  ┌──────────────────────────┐   │
                    │  │ Backward Pass             │   │
                    │  │  for each successor stage │   │
                    │  │    snapshot before        │   │
                    │  │    ... solve openings ... │   │
                    │  │    snapshot after         │   │
                    │  │    → delta (1 row/stage)  │   │
                    │  └──────────────────────────┘   │
                    │  ┌──────────────────────────┐   │
                    │  │ Lower Bound Eval          │   │
                    │  │  snapshot before/after    │   │
                    │  │  → delta (1 row)          │   │
                    │  └──────────────────────────┘   │
                    │         │                       │
                    │    ┌────▼───────────────┐       │
                    │    │ Channel 3: Live     │       │
                    │    │ progress bar update │       │
                    │    └────────────────────┘       │
                    └──────────────┬──────────────────┘
                                  │
                    ┌─────────────▼──────────────────┐
                    │  Channel 2: Solver Stats        │
                    │  → training/solver/iterations   │
                    │    .parquet                      │
                    └─────────────┬──────────────────┘
                                  │
                    ┌─────────────▼──────────────────┐
                    │  Channel 3: Final Summary       │
                    │  → stderr (CLI text)            │
                    └────────────────────────────────┘
```

---

## 6. Crate Responsibilities

| Crate          | Changes                                                                                                                                                                     |
| -------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `cobre-solver` | Add `first_try_successes` and `basis_offered` to `SolverStatistics`. Increment in `HighsSolver`.                                                                            |
| `cobre-sddp`   | Snapshot stats before/after each phase. Per-stage tracking in backward pass. New fields on `ForwardResult`, `BackwardResult`. Scaling report data collection in `setup.rs`. |
| `cobre-core`   | Extend `IterationSummary` event with `solve_time_ms`. Add `SolverSummary` struct (or extend `TrainingEvent::TrainingFinished`).                                             |
| `cobre-io`     | New Parquet schema for `solver/iterations.parquet`. New JSON writer for `scaling_report.json`.                                                                              |
| `cobre-cli`    | Update progress bar format. Expand final summary display.                                                                                                                   |

---

## 7. Implementation Order

1. **`SolverStatistics` extension** — add 2 fields, wire in `HighsSolver`
2. **Scaling report data collection** — capture pre/post ranges in `setup.rs`
3. **Scaling report JSON writer** — new writer in `cobre-io`
4. **Per-phase stats delta** — snapshot before/after in forward, backward, LB
5. **Backward per-stage tracking** — inner loop instrumentation
6. **Solver stats Parquet schema + writer** — new file in `cobre-io`
7. **IterationSummary extension** — add solve_time_ms field
8. **Progress bar update** — add LP avg time
9. **Final summary expansion** — aggregate and display full stats
10. **Integration test** — verify outputs on a deterministic case
