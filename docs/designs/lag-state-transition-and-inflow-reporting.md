# Lag State Transition & Inflow Reporting — Design Report

> **Date:** 2026-03-22
> **Status:** Analysis complete, ready for implementation
> **Context:** Bug report in `cobre-bridge/docs/cobre-water-balance-bug.md` —
> simulation output shows water balance that doesn't close with the reported
> `inflow_m3s` values.

---

## 1. Root Cause Summary

Two distinct issues were identified:

| #   | Issue                                                                   | Severity     | Scope                  |
| --- | ----------------------------------------------------------------------- | ------------ | ---------------------- |
| 1   | `inflow_m3s` output omits PAR lag contribution and subtracts withdrawal | **High**     | Output reporting       |
| 2   | Lag state vector never shifts between stages                            | **Critical** | SDDP model correctness |

Both stem from the same code path: the state extraction after each LP solve
in `forward.rs` and `simulation/pipeline.rs`.

---

## 2. LP Structure (What's Already Correct)

The LP builder constructs the correct constraint structure. The column layout:

```text
[0, N)               storage (v_out) — outgoing storage, FREE
[N, N*(1+L))         inflow_lags    — AR lag variables, FIXED by lag-fixing rows
[N*(1+L), N*(2+L))   storage_in     — incoming storage, FIXED by storage-fixing rows
N*(2+L)              theta          — future cost variable
```

The fixing constraints (rows `[0, n_state)`):

- **Rows [0, N)**: `v_in[h] = state[h]` — fix incoming storage
- **Rows [N, N\*(1+L))**: `lag[l, h] = state[N + l·N + h]` — fix lag inputs

The water balance (rows `[n_state, n_state + N)`):

```
v_out[h] - v_in[h] + Σ_b[τ_b · (q_b + s_b)] - cascade
    - ζ · Σ_l[ψ_l · lag[l,h]] + ζ · evap - ζ · slack
    = ζ · (base_h - withdrawal_h + σ_h · η_h)
```

The lag-fixing rows are **strategically positioned** to enable the state
transition: their RHS is set by the Category 2 patches in `PatchBuffer`,
which read from `current_state`. If `current_state` contains correctly
shifted lags, the LP automatically receives the right values.

---

## 3. Issue 1: Inflow Reporting Bug

### Current behavior

**File:** `simulation/pipeline.rs:389-394`

```rust
for &rhs_hm3 in &scratch.noise_buf {
    scratch.inflow_m3s_buf.push(rhs_hm3 / zeta);
}
```

This computes:

```
inflow_m3s = noise_buf[h] / ζ
           = (ζ · (base - withdrawal + σ·η)) / ζ
           = base - withdrawal + σ·η
```

### What's wrong

The reported value is the **RHS-derived incremental inflow**, which omits:

1. **PAR lag contribution**: `Σ_l[ψ_l · lag_in_l]` — these terms are on the
   LHS of the water balance as coefficients on LP variables, not in the RHS.
2. **Withdrawal subtraction**: `withdrawal` is subtracted from the RHS to
   account for scheduled withdrawals, but inflow should be reported gross.

The **actual total inflow** used by the LP for the water balance is:

```
Z_t = base + Σ_l[ψ_l · lag_in_l] + σ · η
```

### Impact (PIMENTAL example)

```
reported inflow_m3s  = 1538 m³/s  (base + σ·η - withdrawal)
actual LP inflow     ≈ 2737 m³/s  (base + Σ[ψ·lag] + σ·η)
missing lag contrib  ≈ 1196 m³/s
missing withdrawal   ≈ 3 m³/s
```

Users checking the water balance with the reported inflow get a 3210 hm³
discrepancy, concluding the LP is broken. The LP is actually correct — the
report is wrong.

### Fix

Compute total inflow from noise_buf, withdrawal, and lag contribution:

```
total_inflow_m3s[h] = noise_buf[h]/ζ + withdrawal[h] + Σ_l[ψ_l · lag_in_l_h]
```

All components are available at extraction time:

- `noise_buf[h]` — already in scratch
- `zeta` — in `output.zeta_per_stage`
- `withdrawal` — needs to be passed through (from hydro bounds or template)
- `ψ_l` — from `stochastic.par().psi_slice(t, h)`
- `lag_in_l_h` — from `unscaled_primal[indexer.inflow_lags.start + l·N + h]`
  (the lag columns are fixed to incoming values)

---

## 4. Issue 2: Lag State Never Shifts

### Current behavior

**File:** `forward.rs:594-599`

```rust
rec.state.clear();
rec.state.extend_from_slice(&unscaled_primal[..indexer.n_state]);
ws.current_state.clear();
ws.current_state.extend_from_slice(&unscaled_primal[..indexer.n_state]);
```

The outgoing state copies `primal[0..n_state]` verbatim:

- `primal[0..N]` = v_out values (free LP variables) — **correct**
- `primal[N..N*(1+L)]` = lag column values (fixed to incoming) — **stale**

The lag columns are identity-constrained to the incoming state by the
lag-fixing rows. So `primal[N + l·N + h] = state_in[N + l·N + h]` always.
The outgoing lag state equals the incoming lag state — **no shift occurs.**

### What should happen

The PAR model convention: `lag[l]` in the state represents Z\_{t-1-l}, the
inflow from (l+1) stages ago. After solving stage t, the outgoing state must
shift:

```
lag_out[0, h] = Z_t_h          (realized inflow at stage t)
lag_out[l, h] = lag_in[l-1, h]  for l > 0   (shift down)
```

Where the realized inflow at stage t is:

```
Z_t_h = base_t_h + Σ_l[ψ_l · lag_in_l_h] + σ_t_h · η_t_h
```

### Impact

Without lag shifting, **every stage uses the initial lag values** from
`past_inflows` in `initial_conditions.json`. The PAR model never conditions
on the realized inflow history. For a 118-stage horizon, stages 1–117 all
see the same lag state as stage 0.

This means:

- The AR serial correlation across stages is lost
- The stochastic inflow model degenerates to i.i.d. noise around a
  seasonal mean adjusted by fixed initial lags
- Cut coefficients for lag state variables are near-zero (since changing
  the lag value at stage t has no effect on future stages — the same fixed
  value is used everywhere)
- SDDP convergence is slower and the policy is suboptimal for systems
  with strong inflow autocorrelation

The deterministic test suite (D01–D15) did not catch this because those
cases use PAR order 0 or have contrived initial conditions where the
lag values happen to be reasonable.

### Fix

After solving each stage, compute Z_t and shift the lag vector in the
outgoing state. This affects:

1. **`forward.rs` — `run_forward_stage`**: after unscaling primal, shift
   `ws.current_state` lags and `rec.state` lags.
2. **`simulation/pipeline.rs` — `solve_simulation_stage`**: same shift on
   `ws.current_state`.
3. **`backward.rs` — `process_trial_point_backward`**: the backward pass
   reads `basis_store.get(m, s)` which uses the forward trial point state.
   The trial point state in `exchange.state_at(rank, m)` must also contain
   shifted lags so the backward LP sees correct lag values.

The cut framework remains valid: the cut coefficients (duals of the
lag-fixing constraints) represent ∂Q/∂lag_l, which correctly measures the
marginal value of the incoming lag. What changes is the **state values**
used to evaluate the cuts, which are now correctly shifted.

### Computing Z_t

The realized inflow `Z_t_h` can be computed from data already available
after the noise transform:

```
Z_t_h = noise_buf[h] / ζ + withdrawal_h + Σ_l[ψ_l · lag_in_l_h]
```

This formula works for all inflow non-negativity methods because
`noise_buf[h]` already incorporates the clamped noise (for truncation)
or the raw noise (for none/penalty).

Alternatively, call `evaluate_par_batch` which does the same computation.

---

## 5. Implementation Plan

### Phase A: Fix inflow reporting (Issue 1)

Modify `extract_sim_stage_result` (or `solve_simulation_stage`) to compute
total inflow including lag contribution. Pass withdrawal values and PAR
coefficients to the extraction point.

**Files:** `simulation/pipeline.rs`
**Risk:** Low — output-only change, no effect on LP or convergence.

### Phase B: Fix lag state transition (Issue 2)

Add lag shifting to the state extraction after each LP solve. Compute
Z_t from the noise transform and shift the lag vector before storing
the outgoing state.

**Files:** `forward.rs`, `simulation/pipeline.rs`, possibly `backward.rs`
**Risk:** Medium — changes the SDDP state transition. Existing cuts from
a training run become inconsistent if mixed with new cuts. Clean restart
required after the fix.

### Phase C: Verification

1. Add a deterministic test case with PAR order > 0 and verify that lag
   values shift correctly between stages.
2. Verify that the reported `inflow_m3s` matches the water balance for
   the PIMENTAL case.
3. Run the full convertido case and compare convergence with/without the
   lag shift fix.

---

## 6. Summary of Affected Code Paths

| File                         | Function                   | Current Bug                                                | Fix                               |
| ---------------------------- | -------------------------- | ---------------------------------------------------------- | --------------------------------- |
| `simulation/pipeline.rs:389` | `extract_sim_stage_result` | `inflow_m3s = noise_buf/ζ` (missing lags, has -withdrawal) | Add lag contribution + withdrawal |
| `forward.rs:594`             | `run_forward_stage`        | `current_state = primal[0..n_state]` (no lag shift)        | Shift lag vector, set lag_0 = Z_t |
| `forward.rs:594`             | `run_forward_stage`        | `rec.state = primal[0..n_state]` (no lag shift)            | Same shift for trajectory record  |
| `simulation/pipeline.rs:367` | `solve_simulation_stage`   | `current_state = primal[0..n_state]` (no lag shift)        | Shift lag vector, set lag_0 = Z_t |
