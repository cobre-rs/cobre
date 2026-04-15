# Basis Padding Audit Report

**Date:** 2026-04-14
**Feature:** Basis-aware warm-start padding (Strategy S3, Epic 05)
**Status:** Non-functional due to config wiring bug

---

## Summary

The basis padding feature (`pad_basis_for_cuts`) was implemented to reduce
simplex discovery pivots by assigning informed basis statuses to newly added
cut rows during training. The core algorithm is correct, but a config wiring
bug prevents the feature from ever being enabled in production. Additionally,
several architectural limitations constrain its effectiveness even after the
bug is fixed.

---

## Root Cause: Config Wiring Bug (SHOW-STOPPER)

**The feature cannot be enabled.** The config value `basis_padding: true` is
correctly parsed but silently dropped during `StudySetup` construction.

The gap is in `crates/cobre-sddp/src/setup.rs`, `StudySetup::new` (lines
430-466):

```rust
let params = StudyParams::from_config(config)?;  // parses basis_padding_enabled = true
let budget = params.budget;                       // budget IS extracted
let mut setup = Self::from_broadcast_params(      // basis_padding NOT passed
    /* 14 params -- none is basis_padding_enabled */
);
setup.budget = budget;                            // budget IS wired
// MISSING: setup.basis_padding_enabled = params.basis_padding_enabled;
Ok(setup)
```

- `StudyParams::from_config` correctly reads `config.training.cut_selection.basis_padding`
  and stores it as `basis_padding_enabled` (line 248).
- `from_broadcast_params` does not accept `basis_padding_enabled` in its
  signature (line 496). Its constructor hardcodes `basis_padding_enabled: false`
  (line 1319).
- No post-construction assignment exists (`setup.basis_padding_enabled` is
  never written after construction -- zero grep matches for
  `setup.basis_padding`).
- Neither the CLI (`cobre-cli`) nor the Python bindings (`cobre-python`)
  reference `basis_padding` at all.

**Result:** Even with `"basis_padding": true` in config.json, the training
loop always sees `false`. The padding code path at `forward.rs:808` is never
entered. The feature has never executed in any production run or integration
test.

### Fix

In `StudySetup::new` (`setup.rs`), add after `setup.budget = budget;`:

```rust
setup.basis_padding_enabled = params.basis_padding_enabled;
```

---

## Secondary Issues

These limit the feature's effectiveness even after the wiring bug is fixed.
They are ordered by impact, from most to least significant.

### 1. Backward pass not covered

Basis padding is only applied in `forward.rs:808-817`. The backward pass
(`backward.rs:452-454`) calls `solve_with_basis` at opening 0 for each
(trial-point, successor-stage) pair WITHOUT padding. The backward pass
typically dominates computation:

| Pass     | Warm-start calls per iteration              |
| -------- | ------------------------------------------- |
| Forward  | `scenarios x stages`                        |
| Backward | `trial_points x stages` (at opening 0 only) |

Since the number of openings per stage is often 20-100+, the backward pass
has comparable or greater warm-start volume. None of these benefit from
padding.

**Complication:** The backward pass solves at the SUCCESSOR stage, using the
forward-pass basis extracted at the same (scenario, stage). Extending padding
to the backward pass requires access to the successor stage's cut pool and
the trial-point state, which are available in `process_trial_point_backward`
but not currently wired to the padding function.

### 2. Cut selection causes basis row misalignment

The forward pass rebuilds the LP from scratch for every (scenario, stage) pair
(`forward.rs:1111-1113`). Both `build_cut_row_batch_into` and `active_cuts()`
iterate active cuts in ascending slot order. When cut selection deactivates
cuts between iterations, the stored basis row statuses become misaligned with
the new LP rows:

**Example:**

```
Iteration N:   active slots = [0, 1, 2, 3, 4]
                basis row 0 -> slot 0, row 1 -> slot 1, row 2 -> slot 2, ...

Cut selection deactivates slot 1.
Backward pass adds slots 5, 6.

Iteration N+1: active slots = [0, 2, 3, 4, 5, 6]
                LP row 0 -> slot 0 (ok)
                LP row 1 -> slot 2 (basis row 1 was for slot 1!)
                LP row 2 -> slot 3 (basis row 2 was for slot 2!)
                ...
```

All basis rows after the first deactivated slot are applied to the wrong LP
rows. Padding correctly extends the basis at the END with informed statuses
for the newest cuts, but cannot fix the misaligned interior rows. HiGHS may
reject the entire misaligned basis, negating both the warm-start AND the
padding.

**Note:** This is a pre-existing warm-start issue, not specific to the padding
feature. Without cut selection (or when the active set only grows), alignment
is preserved and the padding works correctly.

### 3. Cut evaluation at incoming state

`pad_basis_for_cuts` evaluates cuts at `ws.current_state` (the incoming state,
`forward.rs:809`). The cuts constrain theta as a function of the outgoing
state variables, which are LP decision variables determined by the solve.

The incoming state is the best available proxy before the solve, but the
tightness classification can be inaccurate: a cut that appears slack at the
incoming state may be tight at the outgoing state, and vice versa. The
severity depends on how much the state variables change within a single stage.

### 4. theta estimate is a heuristic

`pool.evaluate_at_state()` (`pool.rs:387-400`) returns `max(intercept_i +
coeff_i . state)` -- the FCF lower bound at the given state. This is used as
the theta estimate for slack computation:

```
slack = theta_max - cut_value
```

The LP-optimal theta may differ from the FCF bound (e.g., when the LP
objective pushes theta higher than the tightest cut at the warm-start state).
This introduces additional classification noise beyond issue 3.

### 5. Marginal benefit per solve is small

In SDDP, typically 1-3 cuts are binding at any given state. The remaining
cuts are slack. Padding correctly assigns:

- `NONBASIC_LOWER` to 1-3 tight cuts (saves discovery pivots)
- `BASIC` to the rest (matches the default fill)

This saves 1-3 simplex pivots per solve -- negligible compared to total
solve effort, especially for larger LPs with hundreds of constraints. The
benefit is proportional to the number of NEW cuts added per iteration, not
the total cut count.

### 6. Simulation phase explicitly disabled

`setup.rs:1625` hardcodes `basis_padding_enabled: false` for the simulation
context. This is intentional (simulation uses the training basis cache
directly), but means the feature only helps training wall-clock time.

---

## What Works Correctly

Despite the wiring bug preventing activation, the core algorithm is sound:

- **Sign convention:** The slack computation `theta_value - cut_value` matches
  the LP row formulation `-coeff . x + theta >= intercept` after accounting for
  the coefficient negation in `build_cut_row_batch_into` (`push_scaled_coefficient`
  at `forward.rs:300` applies `-coeff * d`).

- **Dimension consistency:** When the LP is rebuilt from scratch (no deactivated
  cuts), `pad_basis_for_cuts` targets `base_row_count + pool.active_count()`,
  which equals `solver.num_rows` after `load_model + add_rows`. No fallback
  fill occurs in `solve_with_basis`.

- **Scaling correctness:** The pool stores coefficients and intercepts in scaled
  cost units (inheriting the LP's `COST_SCALE_FACTOR = 1000`). The
  `evaluate_at_state` function and the padding function both operate in the
  same scaled space. Column scaling preserves row slack values.

- **Edge cases:** Empty pools trigger an early return (no padding). Single-cut
  pools correctly yield `slack = 0` (tight). The tolerance of `1e-7` is
  appropriate for scaled cost units in the 1-1000 range.

- **Unit tests:** The 6 tests in `basis_padding.rs:150-310` cover mixed
  tight/slack, exactly-tight, empty pool, already-padded, all-slack, and
  violated cuts. All pass.

---

## Test Coverage Gaps

- **Zero integration tests with padding enabled.** `basis_padding_enabled: true`
  has zero occurrences across all test files. The wiring bug makes it impossible
  to test end-to-end without fixing the assignment.

- **No regression test verifying numerical equivalence.** After enabling, the
  deterministic test suite (D01-D28) should produce bit-identical results. This
  needs explicit verification since the padding changes the simplex path (not
  the optimal solution).

- **No basis rejection rate test.** After enabling, `SolverStatistics.basis_rejections`
  should not increase. A test should verify that padded bases are accepted by
  HiGHS at least as often as unpadded bases.

---

## Data Flow Diagram

```
config.json
  |
  v
CutSelectionConfig.basis_padding: Option<bool>        [cobre-io/config.rs:256]
  |
  v
StudyParams.basis_padding_enabled: bool               [setup.rs:131]
  |
  X  <-- WIRING BUG: value dropped here
  |
StudySetup.basis_padding_enabled: false (hardcoded)   [setup.rs:1319]
  |
  v
TrainingContext.basis_padding_enabled: false            [setup.rs:1566]
  |
  v
forward.rs:808: if basis_padding_enabled {             [NEVER ENTERED]
  |-- pool.evaluate_at_state(incoming_state) -> theta
  |-- pad_basis_for_cuts(basis, pool, state, theta, base_rows, 1e-7)
  |     |-- Evaluate each new active cut: slack = theta - cut_value
  |     |-- slack <= 1e-7 -> NONBASIC_LOWER (tight)
  |     +-- slack >  1e-7 -> BASIC (slack, matches default fill)
  +-- ws.solver.solve_with_basis(padded_basis)
        |-- Copy basis row/col status to HiGHS buffers  [highs.rs:1186-1196]
        |-- cobre_highs_set_basis()                      [highs.rs:1206-1211]
        +-- solve()                                      [highs.rs:1222]
```

---

## Recommendations

1. **Fix the wiring bug.** Add `setup.basis_padding_enabled =
params.basis_padding_enabled;` in `StudySetup::new`. This is the
   prerequisite for all other improvements.

2. **Add integration test with padding enabled.** Run at least one
   deterministic case (e.g. D01) with `basis_padding_enabled: true` to verify
   numerical equivalence and zero basis rejections.

3. **Instrument effectiveness.** After enabling, log the
   `basis_padding_tight` / `basis_padding_slack` counters alongside
   `basis_rejections` per iteration in the solver stats output. This provides
   visibility into whether the classifications are reasonable and whether
   HiGHS accepts the padded bases.

4. **Extend to backward pass.** The backward pass at opening 0 already calls
   `solve_with_basis`. Padding the stored basis with the successor stage's
   cut pool would cover the backward warm-starts. The state for evaluation is
   the trial point `x_hat`, available in `process_trial_point_backward`.

5. **Remap basis rows after cut selection.** When cuts are deactivated, build
   a mapping from old active-slot order to new active-slot order and permute
   the stored basis row statuses accordingly. This preserves warm-start
   quality across iterations with active cut selection and benefits both
   padding and the pre-existing warm-start mechanism.

6. **Benchmark with a production-scale case.** Small test cases (few hydros,
   few stages) may not show measurable improvement because the per-solve
   savings (1-3 pivots) are dwarfed by fixed overheads. A case with 50+
   hydros, 100+ stages, and thousands of accumulated cuts is the realistic
   target for this optimization.
