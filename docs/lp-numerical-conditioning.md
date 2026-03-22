# LP Numerical Conditioning — Problem Report

> **Date:** 2026-03-21
> **Context:** First full-scale NEWAVE-converted case (`examples/convertido/`) exposed
> systematic numerical difficulties in HiGHS LP solves during SDDP training.
> Small deterministic test cases (D01–D15) are unaffected.

---

## 1. Problem Description

Training on the `convertido` case (158 hydros, 104 thermals, 5 buses, 6 lines,
118 monthly stages, 3 load blocks) fails with `NumericalDifficulty` after 2–3
SDDP iterations. The LP solver (HiGHS) cannot reach optimality on certain
scenario/stage subproblems, even after exhausting all 12 retry escalation levels.

The failure is **not** caused by an infeasible or unbounded LP — the subproblem
is structurally sound. The root cause is **ill-conditioned coefficient matrices**
produced by the LP builder when real-world Brazilian power system data is used.

---

## 2. Coefficient Range Analysis

The LP has ~3,000 columns and ~1,500 rows per stage (before Benders cuts).
The coefficient ranges span multiple orders of magnitude:

### 2.1 Constraint matrix (A)

| Row type      | Variable             | Coefficient magnitude | Source                      |
| ------------- | -------------------- | --------------------- | --------------------------- |
| Water balance | Storage (hm³)        | ~1.0                  | Direct                      |
| Water balance | Turbined flow (m³/s) | ~0.003                | `zeta = hours × 3600 / 1e6` |
| Water balance | Spillage (m³/s)      | ~0.003                | Same zeta factor            |
| Water balance | Evaporation (mm)     | ~1e-4 to 1e-3         | `zeta × c_ev × dA/dv`       |
| Load balance  | Generation (MW)      | ~1.0                  | Direct                      |
| Load balance  | Exchange (MW)        | ~1.0                  | Direct                      |
| Benders cuts  | State variables      | ~1e4 to 1e10          | Dual × storage range        |

**Matrix coefficient ratio:** ~10⁹ (9 orders of magnitude) after cuts accumulate.

### 2.2 Objective coefficients (c)

| Variable type       | Cost range ($/unit) |
| ------------------- | ------------------- |
| Excess slack        | 0.01                |
| Spillage penalty    | 0.01                |
| Evaporation penalty | 0.1                 |
| Thermal generation  | 7–3,000             |
| Deficit penalty     | 8,291               |

**Objective coefficient ratio:** ~8.3 × 10⁶.

### 2.3 Right-hand side (b)

| Source                 | RHS range       |
| ---------------------- | --------------- |
| Inflow (hm³)           | 0.001 – 50,000  |
| Storage bounds (hm³)   | 0 – 54,400      |
| Generation bounds (MW) | 0 – 8,370       |
| Benders cut intercepts | 3.9e10 – 5.1e10 |

**RHS range:** ~10¹⁴.

### 2.4 Why small cases work

The deterministic regression suite (D01–D15) uses 1–4 hydros with narrow
parameter ranges. The coefficient ratio stays below 10³, well within HiGHS's
comfort zone. The problem only manifests at real-world scale.

---

## 3. What We Have Tried

### 3.1 HiGHS retry escalation (12 levels)

We implemented a 12-level retry escalation that tries independent solver
strategy combinations when the initial solve fails:

| Level | Strategy       | Scaling              | Tolerances | Solver  | User Scale        |
| ----- | -------------- | -------------------- | ---------- | ------- | ----------------- |
| 0     | Primal simplex | presolve             | 1e-7       | simplex | —                 |
| 1     | Dual simplex   | presolve             | 1e-7       | simplex | —                 |
| 2     | Primal         | forced equilibration | 1e-7       | simplex | —                 |
| 3     | Dual           | forced equilibration | 1e-7       | simplex | —                 |
| 4     | Primal         | max-value scaling    | 1e-7       | simplex | —                 |
| 5     | Primal         | presolve             | 1e-6       | simplex | —                 |
| 6     | —              | presolve             | 1e-7       | IPM     | —                 |
| 7     | —              | presolve             | 1e-6       | IPM     | —                 |
| 8     | Primal         | presolve             | 1e-7       | simplex | obj 2⁻¹⁰          |
| 9     | Dual           | presolve             | 1e-7       | simplex | obj 2⁻¹⁰, bnd 2⁻⁵ |
| 10    | Primal         | presolve             | 1e-6       | simplex | obj 2⁻¹³, bnd 2⁻⁸ |
| 11    | —              | presolve             | 1e-6       | IPM     | obj 2⁻¹⁰, bnd 2⁻⁵ |

**Key design decisions:**

- Each level resets to defaults first (independent, non-cumulative).
- `clear_solver` only at level 0 — higher levels preserve internal state.
- Levels 0–1 run without time cap; levels 2–11 have 30s cap.
- UNBOUNDED and TIME_LIMIT during retry continue to next level.
- `user_objective_scale` and `user_bound_scale` (power-of-two HiGHS options)
  are used at levels 8–11 for uniform problem scaling.

**Result:** Helps with some LPs (iterations 1–2 complete), but as Benders cuts
accumulate the coefficient ranges grow and eventually no retry level succeeds.
The `user_*_scale` options apply uniform scaling to the entire problem, which
cannot address the per-column/per-row heterogeneity.

### 3.2 Default solver configuration

Changed default from `simplex_strategy=4, presolve=off` to
`simplex_strategy=4, presolve=off` with presolve enabled in all retry levels.
Presolve is the single most effective setting for numerical recovery but cannot
be enabled by default without performance impact on the millions of LP solves
that succeed on the first try.

### 3.3 Validation alignment

Aligned the `estimation_active` detection in `cobre-io` validation with the
runtime path matrix in `cobre-sddp`, ensuring `cobre validate` catches data
issues (missing `season_definitions`, missing geometry for evaporation) before
`cobre run` hits them at runtime.

---

## 4. Root Cause

The fundamental issue is that the LP builder in `cobre-sddp/src/lp_builder.rs`
constructs the constraint matrix with **heterogeneous physical units baked into
the coefficients**:

1. **Water balance rows** mix hm³ (storage, ~10⁴) with m³/s (flow, ~10¹)
   through the `zeta` conversion factor (~0.003). This creates a 1000:1 spread
   within a single constraint row.

2. **Objective coefficients** range from 0.01 (excess cost) to 8,291 (deficit
   penalty), a 10⁶ spread.

3. **Benders cut rows** inherit the dual values from previous iterations. Since
   duals are proportional to the constraint matrix conditioning, poorly
   conditioned base rows produce poorly conditioned cuts, creating a
   **feedback loop** where each iteration makes the next harder to solve.

HiGHS's internal scaling (equilibration, max-value) and the `user_*_scale`
options apply **uniform** transformations to the entire problem. They cannot
fix the **per-column heterogeneity** where one column's coefficients span
[0.003, 1.0, 50000] while another spans [1.0, 1.0, 1.0].

---

## 5. What Needs to Be Implemented

### 5.1 Column and row scaling at LP construction time

The standard approach for LP conditioning is to apply diagonal scaling matrices
D_r (row) and D_c (column) such that the scaled problem:

```
min  (D_c c)ᵀ x̃
s.t. (D_r A D_c) x̃ ∈ [D_r l, D_r u]
     x̃ ∈ [D_c⁻¹ x_lo, D_c⁻¹ x_hi]
```

has coefficients closer to unity. The original solution is recovered as
`x = D_c x̃` and the duals as `λ = D_r λ̃`.

#### 5.1.1 Column scaling factors

Each column j gets a scale factor `d_j` chosen to normalize the column's
coefficient range. A common choice is geometric mean scaling:

```
d_j = 1 / sqrt(max|A_ij| × min|A_ij|)   (over nonzero entries in column j)
```

For our LP:

- Storage columns (hm³): `d ≈ 1` (coefficients already ~1.0)
- Flow columns (m³/s): `d ≈ 300` (brings 0.003 up toward 1.0)
- Generation columns (MW): `d ≈ 1`
- Theta (future cost): `d ≈ 1e-10` (brings cut coefficients down)

#### 5.1.2 Row scaling factors

Each row i gets a scale factor `r_i`:

```
r_i = 1 / sqrt(max|A_ij| × min|A_ij|)   (over nonzero entries in row i)
```

#### 5.1.3 Scope of changes

| Component       | File                            | Change                                                               |
| --------------- | ------------------------------- | -------------------------------------------------------------------- |
| `StageTemplate` | `cobre-solver/src/types.rs`     | Add `col_scale: Vec<f64>` and `row_scale: Vec<f64>` fields           |
| LP builder      | `cobre-sddp/src/lp_builder.rs`  | Compute scaling factors after matrix assembly, apply to A, c, bounds |
| Forward pass    | `cobre-sddp/src/forward.rs`     | Unscale primal solution: `x = D_c × x̃`                               |
| Backward pass   | `cobre-sddp/src/backward.rs`    | Unscale duals: `λ = D_r × λ̃`; cut coefficients need correction       |
| Lower bound     | `cobre-sddp/src/lower_bound.rs` | Same unscaling as forward pass                                       |
| Cut building    | `cobre-sddp/src/backward.rs`    | Cut coefficients = `D_r[cut_row] × raw_dual × D_c[state_col]`        |
| Simulation      | `cobre-sddp/src/simulation/`    | Unscale primal for output extraction                                 |
| Noise patching  | `cobre-sddp/src/noise.rs`       | RHS patches must be pre-scaled: `patch × r_i`                        |

#### 5.1.4 Key invariant

The scaling must be **transparent** to all code outside the solver interface:

- Cut coefficients stored in the FCF must be in **original (unscaled) units**
- State values passed between stages must be in **original units**
- Simulation output must be in **original units**
- Only the LP template and the solver see scaled values

This means scaling is applied once during `build_stage_templates` and the
inverse is applied every time we read back from the solver.

### 5.2 Implementation strategy

**Phase 1 — Column scaling only (lowest risk, highest impact):**
Column scaling alone addresses the dominant source of ill-conditioning (the
m³/s vs hm³ spread in water balance rows). It requires:

- Computing `col_scale` from the assembled matrix
- Scaling `objective`, `col_lower`, `col_upper`, and matrix `values`
- Unscaling `primal` after every solve
- Adjusting `dual` readback (row scaling not applied, so duals are already
  in original row units; only need `D_c` correction on cut coefficients)

**Phase 2 — Row scaling (if column scaling alone is insufficient):**
Row scaling is more invasive because it changes the dual space and requires
corrections to every RHS patch (inflow noise, load noise, NCS bounds, cut
intercepts).

**Phase 3 — Iterative refinement (if static scaling is insufficient):**
Re-compute scaling factors periodically as cuts accumulate. This is the most
complex option and should only be considered if static scaling doesn't solve
the problem.

### 5.3 Alternative: unit-system redesign

A more radical approach is to reformulate the LP in a normalized unit system
from the start (e.g., express all water quantities in the same unit as flow,
eliminating the zeta conversion). This would require changes to the entire
data model and is not recommended as a first step, but would produce the
cleanest long-term solution.

---

## 6. Files Reference

| File                                           | Role                                            |
| ---------------------------------------------- | ----------------------------------------------- |
| `crates/cobre-solver/src/types.rs:176-238`     | `StageTemplate` struct (add scale vectors here) |
| `crates/cobre-solver/src/highs.rs`             | HiGHS interface + retry escalation              |
| `crates/cobre-sddp/src/lp_builder.rs`          | LP construction (apply scaling here)            |
| `crates/cobre-sddp/src/forward.rs`             | Forward pass (unscale primal here)              |
| `crates/cobre-sddp/src/backward.rs`            | Backward pass (unscale duals here)              |
| `crates/cobre-sddp/src/lower_bound.rs`         | Lower bound evaluation                          |
| `crates/cobre-sddp/src/noise.rs`               | RHS patching (pre-scale patches)                |
| `crates/cobre-sddp/src/simulation/pipeline.rs` | Simulation (unscale output)                     |
