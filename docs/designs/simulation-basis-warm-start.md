# Design: Simulation Basis Warm-Start from Training Checkpoint

## Status

Proposal — 2026-03-23

## Context

The SDDP simulation phase evaluates the trained policy by solving one LP per
(scenario, stage) pair. Currently, every simulation LP is cold-started — the
solver begins from scratch with no initial basis. This is deliberate: the
original design guaranteed thread-count-independent determinism by avoiding any
basis state that could depend on thread execution order.

However, simulation LPs are structurally identical to the final training
iteration's forward-pass LPs. The same template, the same cut pool, the same
variable layout — only the stochastic RHS patches differ across scenarios. The
training phase already produces and persists an optimal basis per stage in the
policy checkpoint. This basis is a fixed artifact of training, independent of
simulation thread count or scenario assignment.

This document proposes using the training checkpoint basis as a deterministic
warm-start for simulation, preserving reproducibility while reducing simplex
iterations.

---

## 1. Problem Statement

### 1.1 Simulation LP Cost

For a production-scale run (2000 scenarios, 120 stages), the simulation phase
solves 240,000 LPs. Each LP is cold-started via `ws.solver.solve()`, meaning
the simplex algorithm starts from a logical basis and must discover feasibility
before optimizing. For LPs with hundreds of Benders cuts, this can require
significantly more simplex iterations than a warm-started solve.

### 1.2 Why Cross-Scenario Basis Reuse Breaks Determinism

If scenario A's solved basis were used to warm-start scenario B, the result
would depend on which scenario was solved first — which depends on thread
assignment and completion order. This is why simulation currently avoids all
basis reuse.

### 1.3 Why Cross-Stage Chaining Is Not Viable

Each stage has a different LP template (different variable count, different
constraint count, different cut pool). The simulation rebuilds the LP from
template at each stage via `load_model(&ctx.templates[t])`, destroying all
solver state. A basis from stage `t-1` has wrong dimensions for stage `t` and
cannot be injected.

---

## 2. Proposed Solution

### 2.1 Core Idea

Use the per-stage basis from the training checkpoint as a read-only warm-start
for all simulation scenarios at that stage. The basis is:

- **Fixed**: produced once at training end, serialized to the policy checkpoint.
- **Per-stage**: one basis per stage, matching the LP dimensions exactly.
- **Read-only**: shared immutably across all threads — no mutation, no ordering
  dependency, no cross-scenario contamination.
- **Deterministic**: every scenario at stage `t` receives the identical basis,
  regardless of thread count, rank assignment, or execution order.

### 2.2 Why Dimensions Match

The training basis comes from the last forward pass of the last iteration. At
that point, the LP at stage `t` consists of:

1. The structural template `ctx.templates[t]` (same in simulation).
2. All active Benders cuts from the final FCF (same in simulation — the
   simulation uses the converged policy's full cut pool).

Since simulation builds the LP from the same template and the same cuts, the
column count and row count are identical. The `solve_with_basis()` method
already handles minor row-count mismatches (extending with BASIC status), but
in this case no extension is needed.

### 2.3 Solve Sequence (Per Scenario, Per Stage)

Current:

```
1. solver.load_model(&templates[t])
2. solver.add_rows(&cut_batch[t])
3. fill patches (inflow, load, NCS)
4. solver.set_row_bounds(...)
5. apply_ncs_col_bounds(...)
6. solver.solve()                    ← cold-start
7. extract results
```

Proposed:

```
1. solver.load_model(&templates[t])
2. solver.add_rows(&cut_batch[t])
3. fill patches (inflow, load, NCS)
4. solver.set_row_bounds(...)
5. apply_ncs_col_bounds(...)
6. solver.solve_with_basis(&stage_bases[t])   ← warm-start
7. extract results
```

The only change is step 6. If `stage_bases[t]` is `None` (e.g., the terminal
stage had no basis in the checkpoint), the fallback is `solver.solve()`.

### 2.4 Fallback on Basis Rejection

`solve_with_basis()` already handles basis rejection gracefully: it increments
`stats.basis_rejections`, triggers a `debug_assert!`, and falls back to
`solve()`. No special error handling is needed in the simulation code.

---

## 3. Determinism Argument

The simulation result for scenario `s` at stage `t` depends on:

| Input                      | Source                       | Thread-dependent?          |
| -------------------------- | ---------------------------- | -------------------------- |
| LP template                | `ctx.templates[t]`           | No (shared)                |
| Cut batch                  | `fcf.active_cuts(t)`         | No (shared)                |
| Stochastic patches (RHS)   | Deterministic SipHash seed   | No                         |
| NCS column bounds          | Deterministic SipHash seed   | No                         |
| Initial state              | `training_ctx.initial_state` | No (shared)                |
| **Warm-start basis** (new) | `stage_bases[t]`             | **No (shared, read-only)** |

Every input is either shared immutable data or derived from a deterministic
seed. The warm-start basis adds no thread-dependent state. The simplex algorithm
is deterministic given the same LP and starting basis: identical input produces
identical output regardless of which thread executes the solve.

**Guarantee preserved**: simulation results are bit-for-bit identical across
any thread count, rank count, or execution order.

---

## 4. Data Flow

### 4.1 Training → Basis Cache

Already implemented in `training.rs:699-706`:

```
train() completes
  → basis_store.get(last_scenario, t).cloned() for each stage
  → TrainingResult.basis_cache: Vec<Option<Basis>>
```

### 4.2 Basis Cache → Policy Checkpoint

Already implemented in `policy_io.rs:87-147`:

```
TrainingResult.basis_cache
  → i32 → u8 conversion (HiGHS codes 0-4)
  → PolicyBasisRecord per stage
  → FlatBuffers serialization to basis/stage_NNN.bin
```

### 4.3 Policy Checkpoint → Simulation (new)

```
TrainingResult.basis_cache
  ─or─
read_policy_checkpoint().basis_records → Vec<OwnedPolicyBasisRecord>
  → u8 → i32 conversion back to Basis
  → &[Option<Basis>] passed to simulate()
```

In the normal `cobre run` flow, training and simulation happen in the same
process. The `TrainingResult.basis_cache` is already in memory — no
deserialization needed. The basis vector is passed by shared reference to the
simulation phase.

For checkpoint-resume flows (future), the basis is deserialized from the policy
checkpoint's `OwnedPolicyBasisRecord` and converted back to `Basis`.

### 4.4 Simulation Internal Flow

```
simulate()
  receives &[Option<Basis>] as new parameter
  passes reference to each parallel worker

process_scenario_stages()
  receives &[Option<Basis>]
  passes to solve_simulation_stage() per stage

solve_simulation_stage()
  receives Option<&Basis> for current stage
  calls solve_with_basis(basis) or solve() based on Some/None
```

---

## 5. Implementation Plan

### 5.1 Changes to `simulate()` Signature

**File**: `crates/cobre-sddp/src/simulation/pipeline.rs`

Add `stage_bases: &[Option<Basis>]` parameter:

```rust
pub fn simulate<S: SolverInterface + Send, C: Communicator>(
    workspaces: &mut [SolverWorkspace<S>],
    ctx: &StageContext<'_>,
    fcf: &FutureCostFunction,
    training_ctx: &TrainingContext<'_>,
    config: &SimulationConfig,
    output: SimulationOutputSpec<'_>,
    stage_bases: &[Option<Basis>],   // NEW
    comm: &C,
) -> Result<SimulationRunResult, SimulationError>
```

The caller is responsible for providing a vector of length `num_stages`. An
empty slice or a slice of all `None` values falls back to cold-start behavior.

### 5.2 Changes to `solve_simulation_stage()`

**File**: `crates/cobre-sddp/src/simulation/pipeline.rs`

Add `warm_basis: Option<&Basis>` parameter. Replace the solve call:

```rust
// Before:
let view = ws.solver.solve().map_err(/* ... */)?;

// After:
let view = if let Some(basis) = warm_basis {
    ws.solver.solve_with_basis(basis)
} else {
    ws.solver.solve()
}.map_err(/* ... */)?;
```

### 5.3 Changes to `process_scenario_stages()`

**File**: `crates/cobre-sddp/src/simulation/pipeline.rs`

Thread `stage_bases: &[Option<Basis>]` through and pass `stage_bases[t].as_ref()`
to `solve_simulation_stage()` at each stage.

### 5.4 Changes to `StudySetup::simulate()`

**File**: `crates/cobre-sddp/src/setup.rs`

Add `stage_bases: &[Option<Basis>]` parameter and forward it to the inner
`crate::simulate()` call.

### 5.5 Changes to `run.rs` (CLI Orchestration)

**File**: `crates/cobre-cli/src/commands/run.rs`

After training completes, pass `&training_result.basis_cache` to
`setup.simulate()`. The basis cache is already in memory from the
`TrainingResult`.

### 5.6 Update Module Documentation

Update the doc comment at the top of `simulation/pipeline.rs` that currently
says "Each stage LP is cold-started to guarantee thread-count-independent
determinism" to reflect the new warm-start behavior and explain why determinism
is still guaranteed.

---

## 6. Observability

The existing solver statistics infrastructure already captures warm-start
effectiveness with no additional code:

| Metric               | Source                                      | Meaning                                            |
| -------------------- | ------------------------------------------- | -------------------------------------------------- |
| `basis_offered`      | `SolverStatistics.basis_offered`            | Number of `solve_with_basis` calls                 |
| `basis_rejections`   | `SolverStatistics.basis_rejections`         | Times the basis was rejected (cold-start fallback) |
| `total_iterations`   | `SolverStatistics.total_iterations`         | Simplex iterations (should decrease)               |
| `solve_time_seconds` | `SolverStatistics.total_solve_time_seconds` | Wall-clock solve time (should decrease)            |

These are already captured per-scenario in `SimulationRunResult.solver_stats`
and written to `simulation/solver/iterations.parquet`. After this change:

- `basis_offered` should equal `num_stages × 1` per scenario (one offer per
  stage, minus any stages with `None` basis).
- `basis_rejections` should be near zero (the basis dimensions match exactly).
- `total_iterations` and `solve_time_seconds` should decrease compared to
  cold-start baseline — this is the primary performance metric.

No new counters, output columns, or diagnostic files are needed.

---

## 7. Testing Strategy

### 7.1 Determinism Regression

Run the D01-D15 deterministic regression suite with warm-start enabled. Results
must be **bit-for-bit identical** to the cold-start baseline. This validates
that the warm-start basis does not alter the optimal solution.

### 7.2 Thread-Count Independence

Run a multi-scenario stochastic case with 1, 2, and 4 threads. Verify that
simulation costs are identical across all thread counts. This is already tested
by the existing simulation determinism tests, but should be explicitly verified
after the change.

### 7.3 Unit Test: Warm-Start vs Cold-Start Equivalence

Add a unit test in `simulation/pipeline.rs` that solves the same (scenario,
stage) LP twice — once with `solve()` and once with `solve_with_basis()` using
the training basis — and asserts that the optimal objective and primal solution
are identical.

### 7.4 Statistics Validation

After running a stochastic case, verify in the Parquet output that:

- `basis_offered > 0` for simulation rows.
- `basis_rejections` is zero (or near-zero).
- `simplex_iterations` decreased compared to the cold-start baseline.

---

## 8. Scope and Non-Goals

### In Scope

- Thread `&[Option<Basis>]` from `TrainingResult.basis_cache` into the
  simulation pipeline.
- Replace `solve()` with `solve_with_basis()` in `solve_simulation_stage()`.
- Update function signatures in the call chain (`simulate`, `StudySetup::simulate`,
  `process_scenario_stages`, `solve_simulation_stage`).
- Update documentation comments.
- Regression testing.

### Not in Scope

- **Per-scenario basis saving during simulation**: no `get_basis()` calls in
  simulation. The training basis is read-only.
- **Cross-stage basis chaining**: not viable due to LP dimension differences
  between stages.
- **Checkpoint-resume warm-start**: the `OwnedPolicyBasisRecord → Basis`
  conversion path for loading from disk. This can be added later when
  checkpoint-resume is implemented for simulation.
- **Multiple representative bases per stage**: e.g., saving one basis per
  cluster of scenarios. Adds complexity with unclear benefit over a single
  representative.
- **Lower-bound evaluation basis as alternative source**: the training
  forward-pass basis is already the best candidate (solved with the full cut
  pool under realistic scenario conditions).

---

## 9. Risk Assessment

| Risk                                    | Likelihood | Impact                               | Mitigation                                                                                       |
| --------------------------------------- | ---------- | ------------------------------------ | ------------------------------------------------------------------------------------------------ |
| Basis rejection by HiGHS                | Low        | None — automatic cold-start fallback | `solve_with_basis` already handles this                                                          |
| Numerical differences from warm-start   | None       | N/A                                  | Simplex is deterministic; same LP + same basis = same solution                                   |
| Dimension mismatch                      | None       | N/A                                  | Training and simulation build identical LPs from same template + same cuts                       |
| Performance regression (basis overhead) | Very low   | Negligible                           | `solve_with_basis` adds one `memcpy` + one FFI call; cost is dwarfed by simplex iterations saved |

---

## 10. Expected Performance Impact

The magnitude depends on case size and cut count. Warm-start benefit scales
with LP complexity:

- **Small cases** (D01-D15, <50 rows): minimal benefit — cold-start simplex
  converges in few iterations anyway.
- **Medium cases** (100-500 rows, 50-200 cuts): moderate benefit — warm-start
  should save 30-60% of simplex iterations per solve.
- **Production cases** (1000+ rows, 300+ cuts): significant benefit — the
  training basis provides a near-optimal starting point, potentially reducing
  iteration count by 50-80%.

The benefit is multiplicative: `(iterations saved per solve) × (num_scenarios × num_stages)`.
For a production run with 2000 scenarios × 120 stages = 240,000 solves, even a
modest 40% iteration reduction translates to substantial wall-clock savings.
