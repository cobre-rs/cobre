# Fix: Apply Inflow Truncation in Lower Bound Evaluation

## Context

`evaluate_lower_bound` (lower_bound.rs) computes the stage-0 lower bound by
iterating over all openings, patching the LP with per-opening noise, and
solving. It builds the noise RHS inline (lines 200-215) using raw eta
directly, bypassing `transform_inflow_noise`.

When `InflowNonNegativityMethod::Truncation` is active:

- The LP template has **no inflow slack columns** (`has_slack_columns() = false`)
- `transform_inflow_noise` clamps negative PAR inflows to zero before patching
- `evaluate_lower_bound` does NOT clamp — raw negative eta reaches the LP
- Result: **LP infeasibility** at openings with sufficiently negative noise

This also affects `TruncationWithPenalty`: the LP has slack columns (so no
crash), but the noise is not clamped, producing a different subproblem than the
one whose cuts were generated. The lower bound is mathematically incorrect.

### Evidence

- **Paper**: Larroyd et al. (Energies 2022, SS2.5) requires that lower and
  upper bounds be evaluated using the truncation-modified master problem for
  convergence.
- **SPTcpp**: `atualizarModeloOtimizacaoComVariavelRealizacaoInterna()` applies
  truncation before **every** LP solve (forward, backward, lower bound) via
  the shared problem-solve path (C_ModeloOtimizacao.cpp:5526).
- **Cobre**: `transform_inflow_noise` is called in forward.rs:658,
  backward.rs:316, simulation/pipeline.rs:595. Not called in lower_bound.rs.

### Symptoms

```
error: training error: infeasible subproblem at stage 0, iteration 0, scenario 8
```

"scenario 8" is `opening_idx = 8` in the lower bound opening loop
(lower_bound.rs:261-264). Fails on the first lower bound evaluation with any
case that has negative PAR noise at stage 0 under `Truncation` method.

---

## Design

### Approach: Extract shared helper in noise.rs

Extract the core eta-clamping logic from `transform_inflow_noise` into a
reusable helper function. Both `transform_inflow_noise` and
`evaluate_lower_bound` call this helper, eliminating the class of bugs where
one call site is fixed and others are forgotten.

The helper operates on primitive parameters (no context structs) so it can be
called from both the hot-path workspace code and the standalone lower bound
evaluator.

### Key Observation: Precomputation for Stage 0

In `evaluate_lower_bound`, the lag matrix and eta floor are **constant across
all openings** (same `initial_state`, same stage 0). The expensive
`solve_par_noise_batch` call happens once before the opening loop, not per
opening. Only `evaluate_par_batch` (to check for negatives) runs per opening.

---

## Changes

### 1. New helper: `noise.rs::compute_effective_eta`

```rust
/// Compute effective (possibly clamped) eta for each hydro.
///
/// When the inflow method requires truncation and any PAR(p) inflow is
/// negative, clamps each negative hydro's eta upward to the precomputed
/// floor that produces zero inflow. Otherwise writes raw eta unchanged.
///
/// Writes `n_hydros` values into `effective_eta` (cleared on entry).
///
/// The caller must precompute:
/// - `eta_floor` via `solve_par_noise_batch(par_lp, stage, lag_matrix, zeros, out)`
/// - `par_inflow` via `evaluate_par_batch(par_lp, stage, lag_matrix, raw_noise, out)`
///
/// For non-truncation methods (`None`, `Penalty`), writes raw eta directly
/// (no PAR evaluation needed — caller may skip precomputation).
pub(crate) fn compute_effective_eta(
    raw_noise: &[f64],
    n_hydros: usize,
    inflow_method: &InflowNonNegativityMethod,
    par_inflows: &[f64],
    eta_floor: &[f64],
    effective_eta: &mut Vec<f64>,
)
```

**Behaviour:**

- `None` / `Penalty`: copy `raw_noise[..n_hydros]` into `effective_eta`
- `Truncation` / `TruncationWithPenalty`: for each hydro h:
  - if `par_inflows[h] < 0.0`: `effective_eta[h] = raw_noise[h].max(eta_floor[h])`
  - else: `effective_eta[h] = raw_noise[h]`

The `has_negative` fast path (skip all clamping when no hydro is negative) is
kept inside this function.

### 2. Refactor `transform_inflow_noise` to call the helper

`transform_inflow_noise` (noise.rs:52-153) currently has the truncation logic
inline. Refactor:

1. Keep lag extraction from `current_state` (lines 79-88) — unchanged
2. Keep `evaluate_par_batch` call (lines 92-98) — unchanged
3. Keep `solve_par_noise_batch` call (lines 101-112) — unchanged
4. **Replace lines 114-133** (the per-hydro clamping + RHS loop) with:
   ```rust
   compute_effective_eta(
       raw_noise, n_hydros, inflow_method,
       &scratch.par_inflow_buf, &scratch.eta_floor_buf,
       &mut scratch.effective_eta_buf,  // new scratch field
   );
   for (h, &eta_eff) in scratch.effective_eta_buf.iter().enumerate() {
       let base_rhs = template_row_lower[base_row + h];
       scratch.noise_buf.push(base_rhs + noise_scale[stage_offset + h] * eta_eff);
       if has_par {
           scratch.z_inflow_rhs_buf.push(
               par_lp.deterministic_base(stage, h) + par_lp.sigma(stage, h) * eta_eff,
           );
       } else {
           scratch.z_inflow_rhs_buf.push(0.0);
       }
   }
   ```
5. The `None`/`Penalty` branch (lines 135-151) also calls `compute_effective_eta`
   (which just copies raw eta), then the same RHS loop.

This merges the two match arms into a single RHS loop after the
`compute_effective_eta` call, simplifying `transform_inflow_noise`.

**New scratch field:** `effective_eta_buf: Vec<f64>` in `ScratchBuffers`
(workspace.rs). Pre-allocated to `hydro_count` capacity, same as
`par_inflow_buf`.

### 3. Add `inflow_method` to `LbEvalSpec`

```rust
pub struct LbEvalSpec<'a> {
    // ... existing fields ...

    /// Inflow non-negativity treatment method.
    ///
    /// When `Truncation` or `TruncationWithPenalty`, the opening loop clamps
    /// negative PAR(p) inflows to zero before patching the LP. When `None` or
    /// `Penalty`, raw noise is used directly.
    pub inflow_method: &'a InflowNonNegativityMethod,

    /// Stage indexer for LP layout (needed for lag extraction in truncation).
    pub indexer: &'a StageIndexer,
}
```

Note: `indexer` is already a parameter of `evaluate_lower_bound` but not in
`LbEvalSpec`. Adding it to the spec reduces parameter count on the function
signature. Alternatively, keep it as a separate parameter. Choose during
implementation.

### 4. Update `evaluate_lower_bound` opening loop

**Before the opening loop** (after cut batch setup, ~line 195):

```rust
// Truncation precomputation — constant across openings.
let needs_truncation = matches!(
    spec.inflow_method,
    InflowNonNegativityMethod::Truncation
    | InflowNonNegativityMethod::TruncationWithPenalty { .. }
);

let par_lp = spec.stochastic.map(|s| s.par());
let has_par = par_lp
    .map_or(false, |p| p.n_stages() > 0 && p.n_hydros() == n_hydros);

let mut lag_matrix_buf = Vec::new();
let mut eta_floor_buf = Vec::new();
let mut par_inflow_buf = vec![0.0_f64; n_hydros];
let mut effective_eta_buf = Vec::with_capacity(n_hydros);

if needs_truncation && has_par {
    let max_order = indexer.max_par_order;
    let lag_len = max_order * n_hydros;
    lag_matrix_buf.resize(lag_len, 0.0);
    for h in 0..n_hydros {
        for l in 0..max_order {
            lag_matrix_buf[l * n_hydros + h] =
                initial_state[indexer.inflow_lags.start + l * n_hydros + h];
        }
    }

    // eta_floor is constant: depends only on lags (initial_state) and
    // stage (0), not on the opening noise.
    eta_floor_buf.resize(n_hydros, f64::NEG_INFINITY);
    let zero_targets = vec![0.0_f64; n_hydros];
    solve_par_noise_batch(
        par_lp.unwrap(), 0, &lag_matrix_buf, &zero_targets, &mut eta_floor_buf,
    );
}
```

**Inside the opening loop** (replacing lines 198-215):

```rust
for opening_idx in 0..n_openings {
    let raw_noise = opening_tree.opening(0, opening_idx);
    noise_buf.clear();
    z_inflow_rhs_buf.clear();

    // Per-opening: evaluate PAR inflows (only when truncation active).
    if needs_truncation && has_par {
        evaluate_par_batch(
            par_lp.unwrap(), 0, &lag_matrix_buf, raw_noise, &mut par_inflow_buf,
        );
    }

    // Compute effective eta (clamped or raw).
    compute_effective_eta(
        raw_noise, n_hydros, spec.inflow_method,
        &par_inflow_buf, &eta_floor_buf, &mut effective_eta_buf,
    );

    // Build noise_buf and z_inflow_rhs_buf from effective eta.
    for (h, &eta_eff) in effective_eta_buf.iter().enumerate() {
        noise_buf.push(template.row_lower[base_row + h] + noise_scale[h] * eta_eff);
        if has_par {
            let par = par_lp.unwrap();
            z_inflow_rhs_buf.push(
                par.deterministic_base(0, h) + par.sigma(0, h) * eta_eff,
            );
        } else {
            z_inflow_rhs_buf.push(0.0);
        }
    }

    // ... rest of opening loop unchanged (patch_buf, NCS, solve) ...
}
```

### 5. Update call site in training.rs

`training.rs:659-670` — add `inflow_method` to the `LbEvalSpec` construction:

```rust
let lb_spec = LbEvalSpec {
    template: &stage_ctx.templates[0],
    base_row: stage_ctx.base_rows[0],
    noise_scale: stage_ctx.noise_scale,
    n_hydros: stage_ctx.n_hydros,
    opening_tree,
    risk_measure: &risk_measures[0],
    stochastic: Some(training_ctx.stochastic),
    n_load_buses: stage_ctx.n_load_buses,
    ncs_max_gen: stage_ctx.ncs_max_gen,
    block_count: stage_ctx.block_counts_per_stage[0],
    ncs_generation: indexer.ncs_generation.clone(),
    inflow_method: training_ctx.inflow_method,  // NEW
};
```

### 6. Update lower_bound.rs imports

Add to the `use crate::...` block:

```rust
use crate::{
    InflowNonNegativityMethod,
    noise::compute_effective_eta,
    // ... existing imports ...
};
use cobre_stochastic::{evaluate_par_batch, solve_par_noise_batch};
```

---

## Files Changed

| File                                   | Change                                                                                                        |
| -------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| `crates/cobre-sddp/src/noise.rs`       | Add `compute_effective_eta`. Refactor `transform_inflow_noise` to call it.                                    |
| `crates/cobre-sddp/src/workspace.rs`   | Add `effective_eta_buf: Vec<f64>` to `ScratchBuffers`.                                                        |
| `crates/cobre-sddp/src/lower_bound.rs` | Add `inflow_method` to `LbEvalSpec`. Add truncation precomputation + `compute_effective_eta` in opening loop. |
| `crates/cobre-sddp/src/training.rs`    | Pass `inflow_method` in `LbEvalSpec` construction (~line 659).                                                |

---

## Tests

### Unit test: `compute_effective_eta` (noise.rs)

| Test                             | Input                                                   | Expected                              |
| -------------------------------- | ------------------------------------------------------- | ------------------------------------- |
| `none_method_passes_through`     | `None`, eta=[-2, 1]                                     | effective = [-2, 1]                   |
| `penalty_method_passes_through`  | `Penalty{100}`, eta=[-2, 1]                             | effective = [-2, 1]                   |
| `truncation_clamps_negative`     | `Truncation`, par_inflows=[-5, 3], eta_floor=[-1, -inf] | effective = [max(-2,-1), 1] = [-1, 1] |
| `truncation_passes_positive`     | `Truncation`, par_inflows=[3, 5]                        | effective = raw eta (no clamping)     |
| `truncation_with_penalty_clamps` | `TruncationWithPenalty{100}`, par_inflows=[-5, 3]       | same as Truncation                    |

### Existing tests: `transform_inflow_noise` (noise.rs)

All existing tests (lines 639-810) must pass unchanged. The refactoring
replaces internal implementation but preserves identical outputs.

### Unit test: `evaluate_lower_bound` with truncation (lower_bound.rs)

| Test                                          | Setup                                                                                              | Expected                                 |
| --------------------------------------------- | -------------------------------------------------------------------------------------------------- | ---------------------------------------- |
| `lb_truncation_no_infeasibility`              | MockSolver (feasible), 1 hydro, 2 openings, 1 opening has negative PAR inflow, `Truncation` method | Returns `Ok(lb)` — no `Infeasible` error |
| `lb_truncation_with_penalty_no_infeasibility` | Same but `TruncationWithPenalty`                                                                   | Returns `Ok(lb)`                         |
| `lb_none_method_unchanged`                    | `None` method, raw noise applied directly                                                          | Same result as before (regression)       |

### Integration test: real case

```bash
# Must complete without infeasibility error:
target/release/cobre run ~/git/cobre-bridge/example/convertido
```

The case config uses `"method": "truncation"`. Before the fix, it fails at
iteration 0 with infeasibility. After the fix, training completes.

### Regression: deterministic test suite

```bash
cargo test --workspace --all-features
```

All D01-D25 regression tests use `Penalty` (default) and must pass unchanged.

---

## Verification

```bash
# 1. All tests pass
cargo test --workspace --all-features

# 2. Clippy clean
cargo clippy --workspace --all-targets --all-features -- -D warnings

# 3. Format
cargo fmt --all

# 4. Suppression count unchanged
python3 scripts/check_suppressions.py --max 10

# 5. Real case completes with truncation
target/release/cobre run ~/git/cobre-bridge/example/convertido

# 6. Verify convergence looks reasonable (LB > 0, gap shrinking)
# (manual check of output)
```
