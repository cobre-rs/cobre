# Lag Shift — Revised Implementation Approach

> **Date:** 2026-03-22
> **Status:** Design decision, pending plan revision
> **Context:** During implementation of the original plan
> (`plans/lag-state-transition-and-inflow-reporting/`), the reverse-engineering
> approach for recovering realized inflow was identified as fragile and
> unnecessarily complex. This document describes the original approach, its
> problems, and the replacement design.

---

## 1. The Two Bugs

Both bugs stem from the state extraction after each LP solve in `forward.rs`
and `simulation/pipeline.rs`.

### Bug 1 (Critical): Lag state never shifts

After solving stage t, the code copies `primal[0..n_state]` into the outgoing
state. The lag columns in the primal are identity-constrained to the incoming
values (the lag-fixing rows pin them), so the outgoing lag state equals the
incoming lag state. The PAR model never conditions on realized inflow history —
every stage sees the initial `past_inflows` lags.

### Bug 2 (High): Incorrect `inflow_m3s` reporting

The simulation reports `inflow_m3s = noise_buf[h] / zeta`, which equals
`base - withdrawal + sigma * eta`. This omits the PAR lag contribution
(`sum_l[psi_l * lag_in_l]`) and subtracts withdrawal. Users checking the water
balance against reported inflow see a large discrepancy.

---

## 2. Original Plan: Reverse-Engineer Z_t from noise_buf

The original plan (`plans/lag-state-transition-and-inflow-reporting/`) solved
both bugs by recovering the realized inflow Z_t from the noise transform
output:

```
Z_t_h = noise_buf[h] / zeta + withdrawal[h] + sum_l[psi_l * lag_in_l_h]
```

This required:

1. **Pre-extracting withdrawal** into a flat array on `StageContext`
   (epic-01/ticket-001) so it was accessible at LP solve time.
2. **A `compute_realized_inflow` helper** (epic-01/ticket-002) that performed
   the reverse formula.
3. **Threading zeta** through `StageContext` (epic-01/ticket-003).
4. **Saving incoming lags** in scratch buffers before overwriting the state
   (epic-01/ticket-003).
5. **Inline lag shift logic** in `forward.rs` and `simulation/pipeline.rs`
   using the recovered Z_t and saved incoming lags (epic-01/ticket-003).
6. **Removing the old `inflow_m3s`** computation in `extract_sim_stage_result`
   and replacing it with the recovered Z_t (epic-02/ticket-001).

### Problems with this approach

- **Fragile coupling:** The formula reconstructs Z_t by undoing what the LP
  builder did to the RHS. If the LP builder changes (e.g., a new term enters
  the water balance), the reverse formula silently breaks.
- **Withdrawal scaffolding:** Pre-extracting `withdrawal_per_stage` and
  `zeta_per_stage` into `StageContext` adds fields that exist solely to undo
  the LP builder's own subtraction — a code smell.
- **Divergence risk:** The externally computed Z_t may differ from what the LP
  actually used, especially under truncation or future inflow modifications.
- **Hot-path complexity:** Saving incoming lags into scratch buffers and
  performing the shift with inline loops adds code to the most
  performance-sensitive path.

---

## 3. Revised Approach: Z_t as a Primal Variable

Instead of recovering Z_t externally, make it a first-class LP variable. Add
one column `z_h` and one equality constraint per hydro:

```
z_h - sum_l[psi_l * lag_in[h,l]] = base_h + sigma_h * eta_h
```

The RHS is exactly the value that gets noise-patched already (the same patch
that writes to the water balance rows). After solving, `primal[col_z_h]` gives
Z_t directly.

### Why this is better

1. **Single source of truth.** The LP itself defines Z_t. If truncation clamps
   eta, if future features modify the inflow computation, the primal variable
   automatically reflects what the LP actually used.
2. **No reverse-engineering.** No withdrawal extraction, no zeta threading, no
   `noise_buf / zeta + withdrawal` formula.
3. **Trivial inflow reporting.** The simulation extraction reads
   `primal[col_z_h]` — the correct total natural inflow, including lag
   contribution, gross of withdrawal. Bug 2 is solved as a side effect.
4. **Clean state transfer.** After solving, the lag shift reads Z_t from the
   primal and writes it into the outgoing state. No scratch buffers needed for
   incoming lags beyond what slice operations provide.

### Cost

- N extra columns (free, unbounded) and N extra equality rows per stage.
- For typical N = 50–200 hydros, this is negligible (< 1% LP size increase).
- Each new row has 1 + L nonzeros (z_h plus L lag coefficients). Trivially
  sparse.

---

## 4. Solving Both Bugs with the Revised Approach

### Bug 1: Lag state shift

After solving stage t:

1. Read `Z_t_h = primal[col_z_h]` for each hydro.
2. For each hydro h, the outgoing lag block is:
   - `lag_out[1..L] = lag_in[0..L-1]` — a single `copy_from_slice` per hydro
   - `lag_out[0] = Z_t_h` — one scalar write

The LP builder already uses hydro-major contiguous layout for lags:
`state[N + h*L + l]`. The shift exploits this — each hydro's lags are a
contiguous slice, and the slide-down is a memcpy of L-1 elements.

### Bug 2: Inflow reporting

The simulation extraction reads `primal[col_z_h]` directly. No formula, no
post-processing. The reported `inflow_m3s` is exactly what the LP used.

---

## 5. LP Builder Changes

### New columns

Add N columns after the existing layout (or in a designated region). Each
`z_h` column is free (lower = -inf, upper = +inf) with zero objective cost.

### New constraint rows

Add N equality rows. For hydro h:

```
row: 1.0 * z_h - psi[0] * lag_in[h,0] - psi[1] * lag_in[h,1] - ... = RHS
```

Where `RHS = base_h + sigma_h * eta_h` is noise-patched at solve time, using
the same patch mechanism as the water balance rows.

### Layout alignment for state transfer

The `z_h` columns must be positioned in the state vector so that the lag shift
is efficient. The ideal layout places them at the start of the lag block or
adjacent to it, enabling contiguous reads. The exact column placement should
be aligned with the `StageIndexer` conventions.

Alternatively, the `z_h` values can be read from arbitrary primal positions
(they don't need to be state variables themselves — they're auxiliary). The
state transfer writes them into `state[N + h*L + 0]` regardless of where
`col_z_h` lives in the LP.

### Patch buffer

The RHS of the new rows needs noise patching. This is a new patch category:

```
patch z_inflow_row(h) = base_h + noise_scale[h] * eta_h
```

This uses the same `noise_scale` values as the water balance patch, but
WITHOUT the withdrawal subtraction (since Z_t is the gross natural inflow).

---

## 6. What Must Be Reverted

The partial implementation of the original plan added:

- `withdrawal_per_stage` field to `StageTemplates` and `StageContext`
- `zeta_per_stage` field to `StageContext`
- `compute_realized_inflow` and `shift_lag_state` functions in `noise.rs`
- Inline lag shift logic in `forward.rs` and `simulation/pipeline.rs`
- Modified `extract_sim_stage_result` in `simulation/pipeline.rs`
- ~80 test construction site updates for the new fields

All of this should be reverted to restore a clean baseline before implementing
the revised approach.
