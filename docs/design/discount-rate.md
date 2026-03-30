# Design: Discount Rate Implementation

## Context

The SDDP formulation supports an annual discount rate that converts future costs
to present value. The rate is already loaded from `stages.json` and stored in
`PolicyGraph::annual_discount_rate` (with optional per-transition overrides in
`Transition::annual_discount_rate_override`), but **no code in `cobre-sddp`
reads or applies it**. All discount factors are hardcoded to `1.0`.

A converted newave case with 12% annual discount rate produces output where
late-year costs are ~19% higher than early-year costs -- the opposite of what
discounting should produce.

## Mathematical Formulation

In stage-wise SDDP with discounting, the Bellman recursion is:

```
Q_t(x_{t-1}) = min  c_t'x_t  +  d_t * Q_{t+1}(x_t)
                s.t. constraints
```

where `d_t` is the one-step discount factor for the transition departing
stage `t`:

```
d_t = 1 / (1 + r_t)^(Dt / 365.25)
```

- `r_t` is the annual discount rate for the transition (either the global
  `annual_discount_rate` or the per-transition override).
- `Dt` is the duration of stage `t` in days (`end_date - start_date`).
- `365.25` converts to years (consistent with financial day-count conventions).

The LP at stage `t` (for non-terminal stages) becomes:

```
min  c_t'x_t  +  d_t * theta_t
s.t. A_t x_t  = b_t
     -pi_i * x_i  +  theta_t  >=  alpha_k     (Benders cuts)
     theta_t  >=  0
```

The only change to the LP is the **objective coefficient of theta**: it
becomes `d_t` instead of `1.0`.

### Why only the theta coefficient changes

The Benders cuts are derived from the subproblem duals. At stage `t`, we
solve the successor LP at `t+1` and obtain:

```
pi[i]  = dual of state-fixing constraint i
Q      = optimal objective of successor LP
alpha  = Q - sum_i(pi[i] * x_hat[i])
```

The cut added to stage `t` is:

```
theta_t >= alpha + sum_i(pi[i] * x_i)
```

(stored as `-pi[i] * x_i + theta_t >= alpha` in our convention.)

The discount factor `d_t` multiplies theta in the objective, not in the cuts.
This means the LP "sees" future cost as `d_t * theta_t`, which correctly
values the discounted future cost. The cut rows themselves remain unchanged --
they approximate `Q_{t+1}(x_t)` in undiscounted successor-stage units, and
the discount factor in the objective handles the present-value conversion.

### Cumulative discount factor (for reporting)

For output/reporting, the cumulative discount factor at stage `t` is:

```
D_0 = 1.0
D_t = D_{t-1} * d_{t-1}   for t >= 1
```

The present value of stage `t` immediate cost is `D_t * immediate_cost_t`.

## Current State (all gaps)

| Component                        | File                             | Status                                         |
| -------------------------------- | -------------------------------- | ---------------------------------------------- |
| Data model                       | `cobre-core/src/temporal.rs:482` | Stored, never consumed                         |
| LP objective (theta coeff)       | `lp_builder/template.rs:171-174` | Always `1.0`                                   |
| Backward pass (cut formula)      | `backward.rs:22-38`              | No discount (correct -- cuts are undiscounted) |
| Forward pass cost tracking       | `forward.rs`                     | No discount reference                          |
| Lower bound evaluation           | `lower_bound.rs`                 | No discount reference                          |
| Simulation pipeline              | `simulation/pipeline.rs`         | No discount reference                          |
| Simulation extraction            | `simulation/extraction.rs:985`   | Hardcoded `discount_factor: 1.0`               |
| `HorizonMode::discount_factor()` | `horizon_mode.rs:157-160`        | Always returns `1.0`                           |
| Tests                            | `deterministic.rs`               | All cases use `annual_discount_rate: 0.0`      |

## Proposed Changes

### 1. Compute per-stage discount factors at setup time

In `cobre-sddp/src/setup.rs`, after constructing `HorizonMode`, compute a
`Vec<f64>` of per-stage discount factors from the `PolicyGraph`:

```rust
/// Compute the one-step discount factor for each stage transition.
/// Returns a Vec of length `n_stages` where entry `t` is the discount
/// factor for the transition departing stage `t`.
/// Terminal stages get `0.0` (no theta variable exists).
fn compute_discount_factors(stages: &[Stage], policy_graph: &PolicyGraph) -> Vec<f64> {
    let annual_rate = policy_graph.annual_discount_rate;
    stages.iter().enumerate().map(|(t, stage)| {
        // Find the transition departing this stage (if any).
        let rate = policy_graph.transitions.iter()
            .find(|tr| tr.source_id == stage.id)
            .and_then(|tr| tr.annual_discount_rate_override)
            .unwrap_or(annual_rate);
        if rate == 0.0 {
            1.0
        } else {
            let dt_days = (stage.end_date - stage.start_date).num_days() as f64;
            1.0 / (1.0 + rate).powf(dt_days / 365.25)
        }
    }).collect()
}
```

Store this in `StageTemplates` (or a sibling struct) so it is available to
all solver phases.

### 2. Apply discount factor to theta objective coefficient

In `build_single_stage_template` (`lp_builder/template.rs`), after the
cost-scaling loop, multiply the theta column's objective coefficient by the
stage's discount factor:

```rust
// Apply discount factor to theta (future cost) coefficient.
// For terminal stages, theta_col is absent, so this is a no-op.
objective[theta_col] *= discount_factors[stage_idx];
```

Since templates are built once at startup and reused, this is zero runtime
cost.

**Important**: the current scaling comment (lines 156-170) documents that
theta must NOT be divided by `COST_SCALE_FACTOR`. The discount factor is
orthogonal to cost scaling -- it multiplies the already-correct `1.0`
coefficient. The comment should be updated to note this.

### 3. Lower bound evaluation

`evaluate_lower_bound` in `lower_bound.rs` solves the stage-0 LP and reads
its objective. Since the stage-0 template already has `d_0 * theta` in the
objective, the lower bound is automatically in present-value terms. **No
code change needed** beyond the template fix.

### 4. Forward pass

The forward pass solves each stage LP and records the objective. Since the
LP already includes `d_t * theta` in the objective:

- `view.objective * COST_SCALE_FACTOR` gives the total stage cost including
  discounted future cost -- this is correct for computing the upper bound
  trajectory cost.
- **No code change needed** in the forward pass itself.

However, the upper bound trajectory cost (sum of immediate costs across
stages) needs clarification. If the upper bound is computed as the sum of
stage immediate costs without discounting, it represents the undiscounted
total. If it is computed from the stage-0 LP objective (which includes
recursively discounted future costs), it is already in present value.
Verify the current convention and document it.

### 5. Simulation extraction

In `simulation/extraction.rs`, populate the `discount_factor` field with
the cumulative discount factor:

```rust
SimulationCostResult {
    // ...
    discount_factor: cumulative_discount_factors[stage_idx],
    // ...
}
```

The cumulative factors are precomputed:

```rust
let mut cumulative = vec![1.0; n_stages];
for t in 1..n_stages {
    cumulative[t] = cumulative[t - 1] * discount_factors[t - 1];
}
```

This is purely for reporting -- the LP optimization already handles
discounting via the theta coefficient.

### 6. `HorizonMode::discount_factor`

Two options:

**Option A**: Remove `HorizonMode::discount_factor()` entirely since
discount factors are now computed from `PolicyGraph` at setup time and
stored in `StageTemplates`. The method is currently unused in any
meaningful code path.

**Option B**: Keep it but have it accept the precomputed factors. This
adds complexity for no benefit since callers can index the `Vec` directly.

Recommendation: **Option A** -- remove the dead method.

### 7. Backward pass

No change needed. The cut coefficients (`pi`, `alpha`) approximate the
undiscounted successor cost-to-go `Q_{t+1}(x_t)`. The discount factor
`d_t` in the stage-`t` objective handles the present-value conversion.
This is the standard SDDP formulation.

### 8. New deterministic regression test

Add at least one test case (e.g., `D25`) with a non-zero discount rate
and verify:

- The lower bound is smaller than the undiscounted lower bound.
- Stage-0 LP objective reflects present-value cost.
- Simulation `discount_factor` values match expected cumulative factors.
- The solution satisfies Bellman optimality with discounted recursion.

A simple 2-stage, single-hydro case with known analytical solution would
suffice (extend the existing `d02-single-hydro` pattern).

## Files to Modify

| File                                      | Change                                                  |
| ----------------------------------------- | ------------------------------------------------------- |
| `cobre-sddp/src/setup.rs`                 | Compute `discount_factors: Vec<f64>` from `PolicyGraph` |
| `cobre-sddp/src/lp_builder/template.rs`   | Multiply theta coeff by `discount_factors[stage_idx]`   |
| `cobre-sddp/src/lp_builder/template.rs`   | Pass discount factors into `build_stage_templates`      |
| `cobre-sddp/src/simulation/extraction.rs` | Populate `discount_factor` from cumulative factors      |
| `cobre-sddp/src/horizon_mode.rs`          | Remove dead `discount_factor()` method                  |
| `cobre-sddp/tests/deterministic.rs`       | Add D25 regression test with non-zero discount          |
| `examples/`                               | Add or update example with non-zero discount rate       |

## Non-Changes

- **Backward pass**: no change (cuts are in undiscounted successor units).
- **Forward pass**: no change (LP objective already correct after template fix).
- **Lower bound**: no change (reads stage-0 LP objective which is correct).
- **`cobre-core`**: no change (data model is already correct).
- **`cobre-io`**: no change (loading and validation already work).
- **Cost scaling**: orthogonal -- discount factor multiplies the `1.0` theta
  coefficient, not `COST_SCALE_FACTOR`.

## Risk

Low. The change is a single multiplicative factor on one LP column's
objective coefficient, computed once at startup. All existing tests use
`annual_discount_rate: 0.0`, so `d_t = 1.0` and behavior is unchanged.
The only risk is getting the day-count convention wrong (365 vs 365.25 vs
actual year length), which the regression test will catch.
