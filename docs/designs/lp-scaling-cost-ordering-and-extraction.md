# Design: LP Scaling — Cost Ordering Preservation and Extraction Correctness

## Status

Proposed — 2026-03-23

## Context

The SDDP LP prescaler improves solver conditioning by applying geometric mean
column and row scaling (`D_r × A × D_c` form) before passing the problem to
HiGHS. This scaling successfully reduces the condition number of the constraint
matrix, but introduces three bugs that manifest in production-scale cases
(Brazilian NEWAVE PMO, 158 hydros, 118 stages):

1. **Evaporation dump valve** — the LP inflates evaporation 1,000–350,000× above
   physical values for large reservoirs, using evaporation violation as a cheaper
   alternative to spillage for dumping excess water.
2. **`load_mw` unit mismatch** — the simulation output `load_mw` is in row-scaled
   units rather than MW, so the power balance `hydro + thermal + NCS + imports -
exports + deficit - excess = load_mw` does not close.
3. **Per-variable cost distortion** — the cost breakdown (`spillage_cost`,
   `generation_cost`, `exchange_cost`, etc.) has a stray `col_scale` factor,
   so per-component costs do not sum to the aggregate `immediate_cost`.

All three stem from the same architectural gap: the prescaler correctly transforms
the LP into scaled space for the solver, but the inverse transform at the
extraction boundary is incomplete, and the scaling itself does not account for the
solver's own internal scaling.

---

## 1. The Scaling Pipeline (Current State)

### 1.1 Three Layers of Scaling

The LP goes through three scaling layers before the simplex method executes:

| Layer              | Location             | Transform                                         | Target                                          |
| ------------------ | -------------------- | ------------------------------------------------- | ----------------------------------------------- |
| **Cost scaling**   | `lp_builder.rs:2298` | `c[j] /= COST_SCALE_FACTOR` (K=1000)              | Normalizes monetary objective coefficients      |
| **Our prescaling** | `setup.rs:362–374`   | Column scaling, then row scaling (geometric mean) | Normalizes constraint matrix entries toward 1.0 |
| **HiGHS internal** | `highs.rs:99`        | `simplex_scale_strategy = 2` (equilibration)      | HiGHS applies its own scaling on top of ours    |

After column scaling (`compute_col_scale`, `apply_col_scale`):

- Matrix: `A[i,j] *= col_scale[j]`
- Objective: `c[j] *= col_scale[j]`
- Column bounds: `lo[j] /= col_scale[j]`, `hi[j] /= col_scale[j]`

After row scaling (`compute_row_scale` on already col-scaled matrix, `apply_row_scale`):

- Matrix: `A[i,j] *= row_scale[i]`
- Row bounds: `lo[i] *= row_scale[i]`, `hi[i] *= row_scale[i]`
- Objective and column bounds: **untouched** by row scaling

The solver operates in triply-scaled space. HiGHS returns solutions in our
prescaled space (HiGHS unscales its own layer internally). We then unscale
primal and dual values.

### 1.2 What Gets Unscaled at the Extraction Boundary

| Quantity             | Unscaled? | Code location           | Status                                 |
| -------------------- | --------- | ----------------------- | -------------------------------------- |
| Primal (x)           | Yes       | `pipeline.rs:328–337`   | `x_original = col_scale × x̃` ✓         |
| Dual (y)             | Yes       | `pipeline.rs:344–358`   | `y_original = row_scale × ỹ` ✓         |
| Cut coefficients     | Yes       | `forward.rs:311–337`    | `col_scale` applied to cut entries ✓   |
| Noise perturbation   | Yes       | `setup.rs:397–406`      | Pre-multiplied by `row_scale` ✓        |
| Load patches         | Yes       | `lp_builder.rs:553–556` | Multiplied by `row_scale` for solver ✓ |
| **Row bounds (RHS)** | **No**    | `extraction.rs:595`     | Read directly from scaled template ✗   |
| **Objective coeffs** | **No**    | `extraction.rs:711`     | Read directly from scaled template ✗   |

### 1.3 The Double-Scaling Problem

Our geometric mean scaling normalizes the **matrix entries** toward 1.0, but does
**not** normalize the RHS. For a large reservoir like TUCURUI:

- Matrix entries after prescaling: ≈ 1.0 (good)
- Water balance RHS after prescaling: `55,000 × row_scale[wb]` ≈ 34,000 (bad)

HiGHS receives a matrix with entries near 1.0 but RHS in the tens of thousands.
Its equilibration scaler then applies secondary row scaling to bring the RHS
closer to 1.0 — dividing all water-balance-row entries by ~34,000 and
compensating with column scaling that further inflates the effective cost of
variables in that row.

The cumulative (our prescaling × HiGHS equilibration) distortion is what
produces the 1,000×+ cost ordering inversion observed in the cobre-bridge
evaporation report. Our prescaling alone creates a ~2.6× col_scale ratio
between spillage and evaporation violation (from the `τ_b ≈ 2.6` entry
difference); the double-scaling amplifies this to >1,000×.

---

## 2. Bug Analysis

### 2.1 Bug 1: Evaporation Used as Dump Valve

**Observed**: For 21 of 150 hydro plants with evaporation, the LP produces
`evaporation_m3s` values 10× to 350,000× larger than the physical linearized
estimate. All affected plants are large reservoirs during wet-season stages
when excess inflow must be dumped.

**Mechanism**: The LP needs to remove excess water from the water balance. Two
options exist:

- **Spillage** (per-block): `spill_b` appears in the water balance with
  coefficient `τ_b`. Cost: `spillage_cost × block_hours / K`.
- **Evaporation inflation** (stage-level): `Q_ev` appears in the water balance
  with coefficient `ζ` and in the evaporation constraint with coefficient `1.0`.
  Inflating `Q_ev` beyond the linearized value activates `f_minus` (the
  over-evaporation violation slack). Cost: `evap_violation_cost × total_stage_hours / K`.

When `spillage_cost == evap_violation_cost`, the LP is indifferent in exact
arithmetic — both cost the same per hm³ of water removed. After double-scaling,
the effective cost of spillage is amplified because it lives in the water balance
row (subject to aggressive secondary scaling due to large RHS), while evaporation
violation lives in the evaporation constraint (small RHS, mild scaling). The
solver's simplex iterations, operating in doubly-scaled space, break the tie
in favor of evaporation violation.

**Critical**: `Q_ev` has no upper bound (`col_upper = +∞` at `lp_builder.rs:1432`).
There is no physical constraint preventing the LP from setting `Q_ev` to any
non-negative value. This is the enabler — without a bound, the dump mechanism
has unlimited capacity.

**Why the cobre-bridge partial fix (evap cost 0.1 → 100) is insufficient**: A
1,000:1 cost ratio should make spillage clearly cheaper. But for the largest
reservoirs, the cumulative scaling distortion exceeds 1,000×, so even this
ratio is inverted in the solver's effective cost space.

### 2.2 Bug 2: `load_mw` in Wrong Units

**Code path** (`simulation/pipeline.rs:449–457`, `extraction.rs:595`):

```rust
let row_lower_ref = build_row_lower_ref(
    &ctx.templates[t].row_lower,  // row-scaled
    &scratch.load_rhs_buf,         // unscaled (from transform_load_noise)
    ...
);
// ...
load_mw: view.row_lower[load_row],  // reads from row_lower_ref
```

`build_row_lower_ref` has two paths:

| Condition              | Load rows source             | Other rows source     | load_mw units                 |
| ---------------------- | ---------------------------- | --------------------- | ----------------------------- |
| Stochastic load active | `load_rhs_buf` (unscaled MW) | Template (row-scaled) | **MW** (accidentally correct) |
| Deterministic load     | Template (row-scaled)        | Template (row-scaled) | **row-scaled** (wrong)        |

Even in the stochastic case, if `load_bus_indices` does not cover all buses
(only buses with stochastic load are patched), unpatched bus load rows remain
in row-scaled units.

### 2.3 Bug 3: Per-Variable Cost Extraction

**Code path** (`extraction.rs:711`):

```rust
let col_cost = |col: usize| view.primal[col] * view.objective_coeffs[col];
```

Used at:

- `extraction.rs:404` — `spillage_cost`
- `extraction.rs:483` — thermal `generation_cost`
- `extraction.rs:535–537` — `exchange_cost`
- `extraction.rs:911` — NCS `curtailment_cost`
- `extraction.rs:703–750` — full `compute_cost_result` breakdown

The math:

- `view.primal[col]` = `x_original[col]` (correctly unscaled) ✓
- `view.objective_coeffs[col]` = `c_original[col] / K × col_scale[col]` (from template) ✗
- Product = `x_original × c_original × col_scale[col] / K` — stray `col_scale[col]`

The correct per-variable cost is `x_original × c_original / K`. Each extracted
cost is wrong by a factor of `col_scale[col]`.

The **aggregate** `immediate_cost` (`extraction.rs:715`) is correct because it
uses `view.objective - view.primal[theta]` which is computed entirely in scaled
space where the col_scale factors cancel.

Consequence: the per-component cost breakdown does not sum to `immediate_cost`
when `col_scale ≠ 1.0`. This affects all simulation cost output fields.

---

## 3. What Is Correct (No Changes Needed)

These components correctly handle the scaling:

- **Cut construction** (`forward.rs:311–340`): Cut coefficients are multiplied by
  `col_scale[j]` before insertion, producing the correct Benders cut
  `-c_j × Dc[j] × x̃[j] + Dc[θ] × x̃[θ] ≥ intercept`. Cut rows have implicit
  `row_scale = 1.0` (beyond the template row_scale vector length), consistent
  with the dual unscaling for cut rows.

- **Primal unscaling** (`pipeline.rs:328–337`, `forward.rs:732–741`):
  `x_original[j] = col_scale[j] × x̃[j]`.

- **Dual unscaling** (`pipeline.rs:344–358`):
  `y_original[i] = row_scale[i] × ỹ[i]`. For cut rows (`i ≥ row_scale.len()`),
  implicit scale = 1.0.

- **Inflow noise** (`setup.rs:397–406`): `noise_scale` is pre-multiplied by
  `row_scale` so the perturbation `noise_scale × η` is in scaled RHS space.

- **Load patches** (`lp_builder.rs:553–556`): `fill_load_patches` multiplies
  stochastic load values by `row_scale[row]` before writing to the patch buffer.

- **Aggregate cost** (`pipeline.rs:424`, `forward.rs:743`):
  `immediate_cost = (objective - primal[theta]) × K`.

---

## 4. Fix Design

The fix has three tiers, ordered by urgency and independence. Each tier is
implementable and testable on its own.

### 4.1 Tier 1: Physical Evaporation Bounds (Prevents Abuse Regardless of Scaling)

This is the primary defense. Regardless of what the scaling does to cost
ordering, the LP must not be able to produce physically impossible evaporation.

#### 4.1.1 Upper Bound on Q_ev

Add a physical upper bound on the evaporation flow variable.

**Location**: `lp_builder.rs`, in the evaporation column setup (currently line 1432).

**Change**:

```rust
// BEFORE:
col_upper[col_q_ev] = f64::INFINITY;

// AFTER:
// Physical upper bound: linearized evaporation at maximum storage.
// Q_ev = k_evap0 + k_evap_v × v_max
// Safety margin of 2× accounts for linearization approximation error
// (the actual area-volume curve may exceed the linear estimate near v_max).
let q_ev_max = (coeff.k_evap0 + coeff.k_evap_v * vol_max_hm3).max(0.0);
col_upper[col_q_ev] = q_ev_max * Q_EV_SAFETY_MARGIN;
```

where `Q_EV_SAFETY_MARGIN = 2.0`. The evaporation coefficients `k_evap0` and
`k_evap_v` are already computed per (hydro, stage) during template construction
via `compute_evaporation_coefficients`. The `vol_max_hm3` is available from
the hydro entity.

This bound is tight: for TUCURUI at stage 0, `k_evap0 + k_evap_v × v_max ≈ 4 m3/s`.
The LP currently produces 5,144 m3/s — the bound eliminates 99.8% of the abuse.

**Edge case**: When the evaporation coefficient is negative (net condensation,
e.g., JIRAU with coeff = -93 mm), `k_evap0 + k_evap_v × v_max` may be negative.
The `max(0.0)` clamp handles this — Q_ev is non-negative by construction
(`col_lower = 0.0`), so the upper bound is also non-negative.

#### 4.1.2 Asymmetric Evaporation Violation Penalties

The evaporation constraint `Q_ev + f_plus - f_minus = k_evap0 + k_evap_v/2 × (v_in + v_out)`
uses two slack variables:

- `f_plus` (under-evaporation): the linearization overestimates evaporation.
  This is a real approximation error — moderate penalty is appropriate.
- `f_minus` (over-evaporation): the LP pushes Q_ev above the linearized value.
  With the Q_ev upper bound in place, this should rarely activate. When it does,
  it means the linearization is significantly wrong — high penalty is appropriate.

**Change**: Apply an asymmetric cost multiplier to `f_minus`.

```rust
// BEFORE:
objective[col_f_plus] = obj;
objective[col_f_minus] = obj;

// AFTER:
objective[col_f_plus] = obj;
objective[col_f_minus] = obj * OVER_EVAPORATION_COST_MULTIPLIER;
```

where `OVER_EVAPORATION_COST_MULTIPLIER = 100.0`. This makes over-evaporation
100× more expensive than under-evaporation, providing defense-in-depth behind
the Q_ev upper bound.

**Why 100× and not more**: The multiplier must be large enough to deter abuse but
not so large that it creates numerical conditioning issues. At 100×, even if the
base evaporation violation cost is 0.1 $/mm, the effective `f_minus` cost
(`0.1 × 100 × 720 / 1000 = 7.2` in cost-scaled units) remains well within the
typical LP coefficient range.

#### 4.1.3 Data Flow

The `vol_max_hm3` value is needed during template construction to compute the
Q_ev upper bound. This value is already available in `TemplateBuildCtx` via
`ctx.hydros[h_idx].vol_max_hm3`. The evaporation coefficients `k_evap0` and
`k_evap_v` are already computed per `(hydro, stage)` in the evaporation
linearization code (`fill_evaporation_entries`).

The Q_ev bound computation should be placed next to the existing `col_upper`
assignment (currently `lp_builder.rs:1432`), reading from the same
`EvaporationCoefficients` struct that `fill_evaporation_entries` uses.

### 4.2 Tier 2: Extraction Boundary Correctness (Fixes Bugs 2 and 3)

The extraction `SolutionView` must operate entirely in original units. Currently
it mixes unscaled primal/dual with scaled row_lower/objective_coeffs.

#### 4.2.1 Approach: Pass Scale Factors to Extraction

Add `col_scale` and `row_scale` references to `StageExtractionSpec`:

```rust
pub struct StageExtractionSpec<'a> {
    // ... existing fields ...

    /// Column scaling factors from the stage template. Empty when no scaling.
    pub col_scale: &'a [f64],

    /// Row scaling factors from the stage template. Empty when no scaling.
    pub row_scale: &'a [f64],
}
```

This approach is preferred over pre-computing unscaled copies because:

- No additional memory allocation on the extraction path
- The scaling is explicit at each read site — easier to audit
- Consistent with how primal/dual unscaling is already done (multiply/divide
  at the point of use)

#### 4.2.2 Fix `load_mw` Extraction

In `bus_results` (`extraction.rs:595`):

```rust
// BEFORE:
load_mw: view.row_lower[load_row],

// AFTER:
load_mw: {
    let raw = view.row_lower[load_row];
    let scale = spec.row_scale.get(load_row).copied().unwrap_or(1.0);
    if scale != 0.0 { raw / scale } else { raw }
},
```

This produces the correct value in both the stochastic case (where load rows
were already overwritten with unscaled values — dividing by `row_scale` and
then dividing by `row_scale` would double-unscale) and the deterministic case.

**Wait — stochastic load path**: `build_row_lower_ref` patches load rows with
unscaled values from `load_rhs_buf`, but the rest of `row_lower_ref` is from
the row-scaled template. Dividing ALL entries by `row_scale` would correctly
unscale the template portions but would INCORRECTLY unscale the stochastic
load rows (which are already in MW).

**Resolution**: Change `build_row_lower_ref` to always produce fully unscaled
output. This is the cleaner fix because it makes the extraction path
scaling-unaware:

```rust
fn build_row_lower_unscaled<'a>(
    template_row_lower: &[f64],
    row_scale: &[f64],
    load_rhs_buf: &[f64],
    scratch_buf: &'a mut Vec<f64>,
    n_load_buses: usize,
    load_balance_row_start: usize,
    n_blks: usize,
    load_bus_indices: &[usize],
) -> &'a [f64] {
    scratch_buf.clear();
    scratch_buf.reserve(template_row_lower.len());

    // Unscale template row_lower
    for (i, &val) in template_row_lower.iter().enumerate() {
        let scale = if i < row_scale.len() && row_scale[i] != 0.0 {
            row_scale[i]
        } else {
            1.0
        };
        scratch_buf.push(val / scale);
    }

    // Overwrite stochastic load rows (already unscaled from transform_load_noise)
    if n_load_buses > 0 && !load_rhs_buf.is_empty() {
        let mut rhs_idx = 0;
        for &bus_pos in load_bus_indices {
            for blk in 0..n_blks {
                scratch_buf[load_balance_row_start + bus_pos * n_blks + blk] =
                    load_rhs_buf[rhs_idx];
                rhs_idx += 1;
            }
        }
    }

    &scratch_buf[..template_row_lower.len()]
}
```

With this change, `view.row_lower` is always in original units, and the
extraction reads `load_mw: view.row_lower[load_row]` without any per-site
unscaling.

**Performance note**: The old deterministic path returned a direct reference to
`template_row_lower` (zero copy). The new path always copies and unscales.
This is acceptable: extraction runs once per stage per scenario in simulation
(not on the training hot path), and the copy is O(num_rows) ≈ O(1000) — negligible
versus the LP solve time.

#### 4.2.3 Fix Per-Variable Cost Extraction

In `compute_cost_result` and per-entity extraction (`extraction.rs:711`):

```rust
// BEFORE:
let col_cost = |col: usize| view.primal[col] * view.objective_coeffs[col];

// AFTER:
let col_cost = |col: usize| {
    let c_scaled = view.objective_coeffs[col];
    let d = if col < spec.col_scale.len() { spec.col_scale[col] } else { 1.0 };
    // c_scaled = c_original / K × col_scale[col]
    // Divide by col_scale[col] to recover c_original / K
    view.primal[col] * c_scaled / d
};
```

Apply the same fix to all per-variable cost extraction sites:

- `extraction.rs:404` — `spillage_cost: spillage * view.objective_coeffs[s_col] * COST_SCALE_FACTOR`
- `extraction.rs:483` — `generation_cost: gen_mw * view.objective_coeffs[col] * COST_SCALE_FACTOR`
- `extraction.rs:535–537` — `exchange_cost`
- `extraction.rs:911` — NCS `col_cost`

After this fix, `Σ per_variable_cost == immediate_cost` holds exactly.

#### 4.2.4 Wire Scale Factors Through

The `col_scale` and `row_scale` vectors are already stored in `StageTemplate`
(`lp_builder.rs:2353–2354`). Pass references when constructing
`StageExtractionSpec` in `extract_sim_stage_result`:

```rust
&StageExtractionSpec {
    // ... existing fields ...
    col_scale: &ctx.templates[t].col_scale,
    row_scale: &ctx.templates[t].row_scale,
}
```

### 4.3 Tier 3: Eliminate the Double-Scaling Cascade

Tiers 1 and 2 fix the symptoms. Tier 3 addresses the root cause: the
interaction between our prescaling and HiGHS's internal scaling.

#### 4.3.1 Options Considered

**Option A: Disable HiGHS internal scaling**

Set `simplex_scale_strategy = 0` (off). Our prescaling already normalizes
matrix entries toward 1.0. HiGHS's secondary scaling is redundant and, for
large-RHS rows, harmful.

| Aspect           | Assessment                                                                                                                                 |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| Implementation   | One-line change in `highs.rs:99`                                                                                                           |
| Correctness      | No change to LP solution in exact arithmetic                                                                                               |
| Risk             | Our scaling may be insufficient for poorly conditioned cut rows that accumulate over many iterations; requires benchmarking                |
| Retry escalation | Retry levels 5 and 7 set `simplex_scale_strategy = 3` (forced equilibration); these should remain as-is since they are emergency fallbacks |

**Option B: Remove our prescaling, defer to HiGHS**

Remove `compute_col_scale`/`apply_col_scale`/`compute_row_scale`/`apply_row_scale`.
Keep only `COST_SCALE_FACTOR`.

| Aspect              | Assessment                                                                                                                                                                                  |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Implementation      | Large refactor: must rework noise perturbation path (currently pre-multiplied by `row_scale`), cut construction (currently applies `col_scale`), state unscaling, extraction, and all tests |
| Correctness         | HiGHS handles all scaling internally; extraction reads original-space values directly                                                                                                       |
| Risk                | HiGHS's general-purpose scaling may not handle the SDDP-specific structure (mixed hm³/m3s/MW units, large RHS, dynamic cut rows) as well as a structure-aware scaler                        |
| Code simplification | Eliminates ~250 lines of scaling code, `col_scale`/`row_scale` fields, unscaling logic                                                                                                      |

**Option C: RHS-aware prescaling + disable HiGHS**

Modify our row scaling to include the RHS magnitude in the normalization:

```rust
fn compute_row_scale_rhs_aware(
    num_rows: usize,
    num_cols: usize,
    col_starts: &[i32],
    row_indices: &[i32],
    values: &[f64],
    row_lower: &[f64],
    row_upper: &[f64],
) -> Vec<f64> {
    for i in 0..num_rows {
        // Include max(|row_lower|, |row_upper|) in the row statistics
        // so the row_scale also normalizes the RHS magnitude.
        let rhs_abs = row_lower[i].abs().max(row_upper[i].abs());
        if rhs_abs > 0.0 && rhs_abs.is_finite() {
            row_max[i] = row_max[i].max(rhs_abs);
            row_min[i] = row_min[i].min(rhs_abs);
        }
    }
    // ... rest same as current compute_row_scale
}
```

| Aspect         | Assessment                                                                                                                                                                                                      |
| -------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Implementation | Moderate: modify `compute_row_scale`, disable HiGHS scaling, update tests                                                                                                                                       |
| Correctness    | Normalizes both matrix entries AND RHS, eliminating the large-RHS problem                                                                                                                                       |
| Risk           | For rows where the RHS is much smaller than the matrix entries (e.g., evaporation constraint with `k_evap0 ≈ 4`), the RHS dominates the geometric mean and may over-scale the row, reducing matrix conditioning |
| Mitigation     | Clamp the row_scale factor: `scale = clamp(1/sqrt(max*min), 1/MAX_SCALE, MAX_SCALE)`                                                                                                                            |

**Option D: Objective-aware column scaling**

Include the objective coefficient in the geometric mean computation:

```rust
fn compute_col_scale_objective_aware(
    num_cols: usize,
    col_starts: &[i32],
    values: &[f64],
    objective: &[f64],
) -> Vec<f64> {
    for j in 0..num_cols {
        let obj_abs = objective[j].abs();
        if obj_abs > 0.0 {
            max_abs = max_abs.max(obj_abs);
            min_abs = min_abs.min(obj_abs);
        }
        // ... rest same
    }
}
```

| Aspect         | Assessment                                                                                                                    |
| -------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| Implementation | Small: add `objective` parameter to `compute_col_scale`                                                                       |
| Correctness    | Constrains `col_scale` to not push effective costs far from original values, reducing cost ordering distortion                |
| Risk           | May worsen matrix conditioning for columns where the objective coefficient is very different from the constraint coefficients |
| Scope          | Addresses cost ordering only, not the RHS normalization gap                                                                   |

#### 4.3.2 Recommended Approach

**Option A (disable HiGHS internal scaling)**, combined with Tiers 1 and 2.

Rationale:

1. **Tiers 1 + 2 handle correctness.** Physical Q_ev bounds prevent evaporation
   abuse. Extraction unscaling fixes output units. These are correct regardless
   of scaling strategy.

2. **Option A eliminates the cascade.** With HiGHS scaling off, the only scaling
   is ours (which produces ~2.6× col_scale ratio, not 1,000×). Combined with
   Q_ev bounds, this is more than sufficient.

3. **Option A is low-risk and reversible.** If benchmarking shows regressions,
   the one-line change can be reverted. The retry escalation (levels 5/7)
   already uses more aggressive HiGHS scaling as a fallback for hard problems.

4. **Option B is too disruptive.** Removing our prescaling requires reworking
   the noise perturbation path, cut construction, and all scaling-related tests.
   The benefit (code simplification) does not justify the risk for v0.2.0.

5. **Options C and D can be evaluated later.** If Option A causes regression on
   specific cases, Option C (RHS-aware scaling) is the natural next step. Option D
   is orthogonal and can be layered on top of any other option.

#### 4.3.3 Fallback Plan

If benchmarking of Option A reveals performance regression (more simplex
iterations, slower convergence):

1. First, try Option A + Option D (disable HiGHS + objective-aware col_scale).
   This gives our prescaler awareness of costs without adding complexity.

2. If still insufficient, implement Option C (RHS-aware row scaling + clamped
   scale factors) with HiGHS scaling off. This is the most comprehensive
   single-scaler approach.

3. As a last resort, re-enable HiGHS scaling (`simplex_scale_strategy = 2`) but
   keep Tiers 1 + 2. The Q_ev bounds prevent evaporation abuse even under
   double-scaling; the extraction fixes ensure correct output.

---

## 5. Impact on Existing Code

### 5.1 Files Modified

| File                                      | Change                                                                                     | Tier |
| ----------------------------------------- | ------------------------------------------------------------------------------------------ | ---- |
| `cobre-sddp/src/lp_builder.rs`            | Q_ev upper bound, asymmetric violation cost, `build_row_lower_unscaled`                    | 1, 2 |
| `cobre-sddp/src/simulation/extraction.rs` | Add `col_scale`/`row_scale` to `StageExtractionSpec`, fix `col_cost` lambda, fix `load_mw` | 2    |
| `cobre-sddp/src/simulation/pipeline.rs`   | Pass `row_scale` to `build_row_lower_unscaled`, pass scale factors to extraction spec      | 2    |
| `cobre-solver/src/highs.rs`               | Set `simplex_scale_strategy = 0` in `default_options`                                      | 3    |

### 5.2 Training Loop Impact

**None.** The training forward/backward passes are unaffected:

- Tier 1 adds bounds and modifies objective coefficients in the template —
  these propagate automatically through `load_model`.
- Tier 2 modifies only the simulation extraction path, not training.
- Tier 3 changes HiGHS options, which are set at solver construction.

The training cost computation uses `(view.objective - primal[theta]) × K`
(aggregate, no per-variable breakdown), which is already correct.

### 5.3 State Propagation Impact

**None.** State variables (storage, lags) are propagated via
`unscaled_primal[0..n_state]`, which is correctly unscaled by `col_scale`.
The Q_ev column is not a state variable.

### 5.4 Cut Construction Impact

**None.** Cut coefficients come from unscaled duals, and `build_cut_row_batch_into`
applies `col_scale` correctly. The Q_ev bound does not affect cuts (Q_ev is not
a state variable and does not appear in cuts).

---

## 6. Testing Strategy

### 6.1 Regression: Deterministic Suite D01–D15

All 15 cases must pass with identical expected costs. The Q_ev upper bound
should not bind for these small cases (evaporation values are already correct).
Disabling HiGHS scaling may produce slightly different simplex iteration counts
but the same optimal solution.

**Action**: Run full `cargo test --workspace --all-features` after each tier.

### 6.2 New Test: Evaporation Bound Enforcement

Create a deterministic test case with:

- One large reservoir (v_max = 50,000 hm³, area = 2,000 km²)
- High inflow that forces the LP to dump water
- Evaporation coefficient = 4 mm/month

Assert:

- `Q_ev ≤ Q_ev_max × Q_EV_SAFETY_MARGIN`
- `evaporation_violation_m3s` is near zero (the bound prevents abuse)
- Spillage > 0 (the LP correctly uses spillage to dump excess water)

### 6.3 New Test: Power Balance Consistency

After Tier 2, create a simulation test that verifies:

```
∀ (bus, block):
  hydro_gen + thermal_gen + ncs_gen + imports - exports + deficit - excess == load_mw
```

within floating-point tolerance (1e-6 MW).

### 6.4 New Test: Cost Breakdown Consistency

After Tier 2, verify in simulation output:

```
∀ stage:
  thermal_cost + spillage_cost + deficit_cost + excess_cost
  + exchange_cost + generic_violation_cost + fpha_turbined_cost
  + ncs_curtailment_cost == immediate_cost
```

within floating-point tolerance (1e-6 $).

### 6.5 Benchmark: Scaling Strategy Comparison

Run the Brazilian PMO case (or a representative large case) with three
configurations:

| Config       | Our prescaling | HiGHS scaling    | Expected                              |
| ------------ | -------------- | ---------------- | ------------------------------------- |
| A (current)  | On             | On (strategy=2)  | Baseline                              |
| B (proposed) | On             | Off (strategy=0) | Similar iterations, no cost inversion |
| C (fallback) | Off            | On (strategy=2)  | Reference for pure-HiGHS scaling      |

Compare: iteration counts, wall-clock time, lower/upper bound convergence,
evaporation values, cost breakdown consistency.

---

## 7. Implementation Order

```
Phase 1 (Tier 1):  Q_ev bounds + asymmetric violation
  └── Test: D01–D15 regression + new evaporation test
Phase 2 (Tier 2):  Extraction unscaling
  └── Test: D01–D15 + power balance + cost breakdown tests
Phase 3 (Tier 3):  Disable HiGHS scaling
  └── Test: D01–D15 + benchmark comparison
```

Phases are independent and can be shipped separately. Phase 1 is the highest
priority because it fixes the physically observable bug (evaporation values).
Phase 2 fixes silent data correctness issues in simulation output. Phase 3 is
an optimization that eliminates the root cause.

---

## 8. Future Considerations

### 8.1 Scaling for Dynamic Cut Rows

Cut rows are added dynamically and have implicit `row_scale = 1.0`. As cuts
accumulate, their coefficients (from unscaled duals × `col_scale`) may have
very different magnitudes from the template rows. This could worsen conditioning
over long training runs (hundreds of iterations). A future improvement could
apply row scaling to cut rows at insertion time.

### 8.2 Column Scaling Stability

The current `compute_col_scale` is called once during setup on the template
matrix. As cuts are added, the effective column statistics change (cuts add new
entries to state columns). The col_scale computed on the template may become
stale. A future improvement could periodically recompute scaling factors and
rebuild the LP.

### 8.3 Solver-Aware Prescaling

Instead of a generic geometric-mean scaler, a solver-aware prescaler could:

- Query HiGHS's internal scaling factors after a warmup solve
- Use the combined (our × HiGHS) scale as the prescaler for subsequent solves
- This requires HiGHS API support for reading internal scale factors

These are optimizations for v0.3+, not blockers for the current fix.
