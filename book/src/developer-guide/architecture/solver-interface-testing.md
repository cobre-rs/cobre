# Solver Interface Testing and Conformance

## Purpose

This spec defines the conformance test suite for the `SolverInterface` trait and its 10 methods (`load_model`, `add_rows`, `set_row_bounds`, `set_col_bounds`, `solve`, `solve_with_basis`, `reset`, `get_basis`, `statistics`, `name`), as specified in [Solver Interface Trait](./solver-interface-trait.md). The suite verifies that both reference solver backends (HiGHS and CLP) satisfy the same behavioral contract, produce equivalent solutions within tolerance, and correctly implement the dual normalization convention from [Solver Abstraction SS8](./solver-abstraction.md). All test cases use a shared LP fixture with 3 variables and 2 structural constraints, hand-computable optimal solutions, and explicit CSC/CSR data so that expected primal values, dual values, objective values, and basis statuses can be verified by manual calculation.

Test cases reference the method contracts from [Solver Interface Trait SS2](./solver-interface-trait.md), the LP layout convention from [Solver Abstraction SS2](./solver-abstraction.md), the dual normalization convention from [Solver Abstraction SS8](./solver-abstraction.md), and the basis storage convention from [Solver Abstraction SS9](./solver-abstraction.md).

> **Note:** CLP test variants are forward-looking — the CLP solver backend is not yet implemented. Only HiGHS test variants are currently runnable. CLP variants document the intended conformance surface for when additional backends are added.

## SS1. Conformance Test Suite

**Test naming convention:** `test_solver_{variant}_{method}_{scenario}` where `{variant}` is `highs` or `clp`, `{method}` is the `SolverInterface` method under test, and `{scenario}` describes the test case.

**Shared test fixture:** Unless otherwise noted, tests use the following LP with 3 variables and 2 structural constraints, designed to follow the LP layout convention from [Solver Abstraction SS2](./solver-abstraction.md).

### SS1.1 Shared LP Fixture

**Problem description:** A minimal hydrothermal stage problem with one reservoir (state variable), one future cost variable (theta), and one thermal generator (decision variable). Demand is 14 MW, hydro productivity is 2 MW/(m3/s), and the incoming state (reservoir volume) is fixed at 6 hm3.

$$\min_{x_0, x_1, x_2} \quad 0 \cdot x_0 + 1 \cdot x_1 + 50 \cdot x_2$$

subject to:

$$
\begin{aligned}
\text{Row 0 (state-fixing):} \quad & x_0 = 6 \\
\text{Row 1 (power balance):} \quad & 2 x_0 + x_2 = 14
\end{aligned}
$$

with column bounds:

$$x_0 \in [0, 10], \quad x_1 \in [0, +\infty), \quad x_2 \in [0, 8]$$

**LP layout (per [Solver Abstraction SS2](./solver-abstraction.md)):**

| Index | Column | Description       | Lower Bound | Upper Bound | Objective |
| ----- | ------ | ----------------- | ----------- | ----------- | --------- |
| 0     | $x_0$  | State (volume)    | 0.0         | 10.0        | 0.0       |
| 1     | $x_1$  | Future cost theta | 0.0         | $+\infty$   | 1.0       |
| 2     | $x_2$  | Thermal gen       | 0.0         | 8.0         | 50.0      |

| Index | Row           | Type     | Lower Bound | Upper Bound |
| ----- | ------------- | -------- | ----------- | ----------- |
| 0     | State-fixing  | Equality | 6.0         | 6.0         |
| 1     | Power balance | Equality | 14.0        | 14.0        |

Layout parameters: `n_state = 1` (column 0), `n_dual_relevant = 1` (row 0), `num_rows = 2`.

**Constraint matrix in CSC format:**

The constraint matrix $A$ is:

$$A = \begin{pmatrix} 1 & 0 & 0 \\ 2 & 0 & 1 \end{pmatrix}$$

CSC arrays (column-major):

| Array         | Values          | Description                                            |
| ------------- | --------------- | ------------------------------------------------------ |
| `col_starts`  | [0, 2, 2, 3]    | Column 0 has 2 entries, column 1 has 0, column 2 has 1 |
| `row_indices` | [0, 1, 1]       | Column 0 appears in rows 0 and 1; column 2 in row 1    |
| `values`      | [1.0, 2.0, 1.0] | Corresponding non-zero values                          |
| `num_nz`      | 3               | Total non-zeros                                        |

**Hand-computed optimal solution (no cuts):**

From Row 0: $x_0 = 6$. From Row 1: $2(6) + x_2 = 14 \Rightarrow x_2 = 2$. Minimizing objective with no cuts: $x_1 = 0$ (at lower bound).

| Quantity | Value |
| -------- | ----- |
| $x_0^*$  | 6.0   |
| $x_1^*$  | 0.0   |
| $x_2^*$  | 2.0   |
| $z^*$    | 100.0 |

**Hand-computed dual values:**

Sensitivity analysis for the equality constraints:

- $\pi_0$ (state-fixing): Increasing RHS from 6 to $6 + \varepsilon$. From Row 1: $x_2 = 14 - 2(6+\varepsilon) = 2 - 2\varepsilon$. Objective change: $50(-2\varepsilon) = -100\varepsilon$. Therefore $\pi_0 = -100.0$.
- $\pi_1$ (power balance): Increasing RHS from 14 to $14 + \varepsilon$. From Row 1: $x_2 = (14+\varepsilon) - 12 = 2 + \varepsilon$. Objective change: $50\varepsilon$. Therefore $\pi_1 = 50.0$.

**Canonical sign convention ([Solver Abstraction SS8](./solver-abstraction.md)):** A positive dual on a $\leq$ constraint means increasing RHS increases the objective. For equality constraints, the dual is unrestricted. The values above follow the canonical convention: $\pi_0 = -100$ (increasing the state-fixing RHS decreases the objective through reduced thermal need) and $\pi_1 = 50$ (increasing demand increases thermal cost).

**Hand-computed basis:**

With 3 columns and 2 rows, there are 2 basic variables. $x_0$ and $x_2$ are basic (between bounds or uniquely determined). $x_1$ is at its lower bound.

| Element | Status  | Rationale                         |
| ------- | ------- | --------------------------------- |
| Col 0   | Basic   | $x_0 = 6$, uniquely determined    |
| Col 1   | AtLower | $x_1 = 0$, at lower bound         |
| Col 2   | Basic   | $x_2 = 2$, between bounds [0, 8]  |
| Row 0   | AtLower | Equality: slack = 0 = lower bound |
| Row 1   | AtLower | Equality: slack = 0 = lower bound |

### SS1.2 Benders Cut Fixture

Two Benders cuts are provided for `add_rows` tests. Each cut has the form $x_1 \geq \alpha + \beta \cdot x_0$, which in LP row form is $-\beta \cdot x_0 + x_1 \geq \alpha$ (coefficients on the theta column and state column).

**Cut 1:** $x_1 \geq 20 + 5 x_0$

Row form: $-5 x_0 + x_1 \geq 20$. With $x_0 = 6$: $x_1 \geq 50$.

**Cut 2:** $x_1 \geq 80 - 3 x_0$

Row form: $3 x_0 + x_1 \geq 80$. With $x_0 = 6$: $x_1 \geq 62$.

**RowBatch CSR data (both cuts):**

| Array         | Values                 | Description                                      |
| ------------- | ---------------------- | ------------------------------------------------ |
| `num_rows`    | 2                      | Two cuts                                         |
| `row_starts`  | [0, 2, 4]              | Each cut has 2 non-zeros (state col + theta col) |
| `col_indices` | [0, 1, 0, 1]           | Cut 1: cols 0, 1; Cut 2: cols 0, 1               |
| `values`      | [-5.0, 1.0, 3.0, 1.0]  | Cut 1: -5*x0 + 1*x1; Cut 2: 3*x0 + 1*x1          |
| `row_lower`   | [20.0, 80.0]           | Intercepts (alpha values)                        |
| `row_upper`   | [$+\infty$, $+\infty$] | Cuts are $\geq$ constraints                      |

**Hand-computed solution with both cuts:**

The binding constraint is Cut 2: $x_1 = 80 - 3(6) = 62$. Cut 1 is non-binding: $50 < 62$.

| Quantity | Value |
| -------- | ----- |
| $x_0^*$  | 6.0   |
| $x_1^*$  | 62.0  |
| $x_2^*$  | 2.0   |
| $z^*$    | 162.0 |

**Dual values with both cuts:**

- $\pi_0$ (state-fixing): Increase RHS from 6 to $6+\varepsilon$. Row 1: $x_2 = 2 - 2\varepsilon$ (cost $-100\varepsilon$). Cut 2: $x_1 \geq 80 - 3(6+\varepsilon) = 62 - 3\varepsilon$ (cost $-3\varepsilon$). Total: $\pi_0 = -103.0$.
- $\pi_1$ (power balance): Same as before, $\pi_1 = 50.0$.
- $\pi_2$ (Cut 1, non-binding): $\pi_2 = 0.0$.
- $\pi_3$ (Cut 2, binding): Increase RHS from 80 to $80+\varepsilon$. $x_1 = 62 + \varepsilon$. Objective change: $\varepsilon$. Therefore $\pi_3 = 1.0$.

### SS1.3 Patched RHS Fixture

For `set_row_bounds` tests, the state-fixing constraint (Row 0) RHS is changed from 6.0 to 4.0, simulating a different incoming state $x_{t-1} = 4$.

**Hand-computed solution after RHS patch (with both cuts active):**

- $x_0 = 4$. Row 1: $x_2 = 14 - 8 = 6$. Cut 1: $x_1 \geq 20 + 20 = 40$. Cut 2: $x_1 \geq 80 - 12 = 68$. Binding: Cut 2, $x_1 = 68$.

| Quantity | Value |
| -------- | ----- |
| $x_0^*$  | 4.0   |
| $x_1^*$  | 68.0  |
| $x_2^*$  | 6.0   |
| $z^*$    | 368.0 |

### SS1.4 load_model Conformance

| Test Name                                        | Input Scenario                                                                                                                                          | Expected Observable Behavior                                                                                                                                                        | Variant |
| ------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| `test_solver_highs_load_model_and_solve`         | Load shared fixture StageTemplate (SS1.1). Solve without cuts.                                                                                          | Solve returns `Ok`. Objective = 100.0 within 1e-8 relative tolerance. Primal: $x_0 = 6.0$, $x_1 = 0.0$, $x_2 = 2.0$ within 1e-8 absolute tolerance.                                 | HiGHS   |
| `test_solver_clp_load_model_and_solve`           | Load shared fixture StageTemplate (SS1.1). Solve without cuts.                                                                                          | Solve returns `Ok`. Objective = 100.0 within 1e-8 relative tolerance. Primal: $x_0 = 6.0$, $x_1 = 0.0$, $x_2 = 2.0$ within 1e-8 absolute tolerance.                                 | CLP     |
| `test_solver_highs_load_model_replaces_previous` | Load shared fixture. Solve (objective = 100.0). Load a different StageTemplate with objective coefficients [0, 1, 25] (half thermal cost). Solve again. | Second solve returns `Ok`. Objective = 50.0 (= 0 + 0 + 25\*2). Loading a new model fully replaces the previous one per [Solver Interface Trait SS2.1](./solver-interface-trait.md). | HiGHS   |
| `test_solver_clp_load_model_replaces_previous`   | Same as above.                                                                                                                                          | Second solve returns `Ok`. Objective = 50.0. Model fully replaced.                                                                                                                  | CLP     |

### SS1.5 add_rows Conformance

| Test Name                               | Input Scenario                                                                                             | Expected Observable Behavior                                                                                           | Variant |
| --------------------------------------- | ---------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- | ------- |
| `test_solver_highs_add_rows_tightens`   | Load shared fixture. Add both cuts from SS1.2 via `add_rows`. Solve.                                       | Objective = 162.0. Primal: $x_0 = 6.0$, $x_1 = 62.0$, $x_2 = 2.0$. The cuts tighten the objective from 100.0 to 162.0. | HiGHS   |
| `test_solver_clp_add_rows_tightens`     | Same as above.                                                                                             | Objective = 162.0. Primal: $x_0 = 6.0$, $x_1 = 62.0$, $x_2 = 2.0$.                                                     | CLP     |
| `test_solver_highs_add_rows_single_cut` | Load shared fixture. Add only Cut 1 ($x_1 \geq 20 + 5 x_0$) via `add_rows` with a 1-row `RowBatch`. Solve. | Objective = 150.0 (= 0 + 50 + 100). Primal: $x_0 = 6.0$, $x_1 = 50.0$, $x_2 = 2.0$.                                    | HiGHS   |
| `test_solver_clp_add_rows_single_cut`   | Same as above.                                                                                             | Objective = 150.0. Primal: $x_0 = 6.0$, $x_1 = 50.0$, $x_2 = 2.0$.                                                     | CLP     |

### SS1.6 set_row_bounds Conformance

| Test Name                                       | Input Scenario                                                                                            | Expected Observable Behavior                                                   | Variant |
| ----------------------------------------------- | --------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ | ------- |
| `test_solver_highs_set_row_bounds_state_change` | Load shared fixture. Add both cuts. Patch Row 0 RHS from 6.0 to 4.0 (both lower and upper bounds). Solve. | Objective = 368.0. Primal: $x_0 = 4.0$, $x_1 = 68.0$, $x_2 = 6.0$ (per SS1.3). | HiGHS   |
| `test_solver_clp_set_row_bounds_state_change`   | Same as above.                                                                                            | Objective = 368.0. Primal: $x_0 = 4.0$, $x_1 = 68.0$, $x_2 = 6.0$.             | CLP     |

### SS1.6a set_col_bounds Conformance

**Test naming convention:** `test_solver_{variant}_set_col_bounds_{scenario}` where `{variant}` is `highs` or `clp` and `{scenario}` describes the column-bound patching test case.

These tests verify the `set_col_bounds` method contract from [Solver Interface Trait SS2.3a](./solver-interface-trait.md). Each test starts from the shared LP fixture (SS1.1) and applies column bound updates via three parallel slices: `indices`, `lower`, and `upper` (SoA parameter style). The method is infallible (panics on invalid column index) and preserves the solver basis.

| Test Name                                   | Input Scenario                                                                                                                                                                                                                                                                                                                 | Expected Observable Behavior                                                                                                                                                                                                                                                                                                                                                                               | Variant |
| ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| `test_solver_highs_set_col_bounds_basic`    | Load shared fixture. Add both cuts (SS1.2). Update column 2 ($x_2$, thermal gen) bounds from $[0, 8]$ to $[0, 3]$ via `set_col_bounds(&[2], &[0.0], &[3.0])`. Solve.                                                                                                                                                           | Solve returns `Ok`. Objective = 162.0. Primal: $x_0 = 6.0$, $x_1 = 62.0$, $x_2 = 2.0$. The tighter upper bound ($x_2 \leq 3$) does not bind because the optimal $x_2 = 2.0 < 3.0$. Solution is unchanged from the unpatched case (SS1.2). Non-updated column bounds ($x_0$, $x_1$) are unchanged. Row bounds are unchanged.                                                                                | HiGHS   |
| `test_solver_clp_set_col_bounds_basic`      | Same as above.                                                                                                                                                                                                                                                                                                                 | Same: objective = 162.0, primal = $(6.0, 62.0, 2.0)$. Column bound update does not bind.                                                                                                                                                                                                                                                                                                                   | CLP     |
| `test_solver_highs_set_col_bounds_tightens` | Load shared fixture (no cuts). Update column 1 ($x_1$, future cost theta) bounds from $[0, +\infty)$ to $[10, +\infty)$ via `set_col_bounds(&[1], &[10.0], &[f64::INFINITY])`. Solve.                                                                                                                                          | Solve returns `Ok`. Objective = 110.0. Primal: $x_0 = 6.0$, $x_1 = 10.0$, $x_2 = 2.0$. Without the bound update, $x_1 = 0$ (at original lower bound 0). With the tighter lower bound $x_1 \geq 10$, the theta variable is pushed to its new lower bound $x_1 = 10.0$. Objective: $0(6) + 1(10) + 50(2) = 110.0$, up from 100.0. Non-updated column bounds ($x_0$, $x_2$) and all row bounds are unchanged. | HiGHS   |
| `test_solver_clp_set_col_bounds_tightens`   | Same as above.                                                                                                                                                                                                                                                                                                                 | Same: objective = 110.0, primal = $(6.0, 10.0, 2.0)$. Tighter lower bound on $x_1$ is active.                                                                                                                                                                                                                                                                                                              | CLP     |
| `test_solver_highs_set_col_bounds_repatch`  | Load shared fixture (no cuts). Solve (objective = 100.0, $x_1 = 0.0$). Update column 1 bounds to $[10, +\infty)$ via `set_col_bounds(&[1], &[10.0], &[f64::INFINITY])`. Solve (objective = 110.0, $x_1 = 10.0$). Re-update column 1 bounds back to $[0, +\infty)$ via `set_col_bounds(&[1], &[0.0], &[f64::INFINITY])`. Solve. | Third solve returns `Ok`. Objective = 100.0. Primal: $x_0 = 6.0$, $x_1 = 0.0$, $x_2 = 2.0$. Re-applying column bound updates restores the original feasible region. The solver correctly applies each successive column bound update before the next solve. Basis is preserved across updates.                                                                                                             | HiGHS   |
| `test_solver_clp_set_col_bounds_repatch`    | Same as above.                                                                                                                                                                                                                                                                                                                 | Same: three solves produce objectives 100.0, 110.0, 100.0. Column bounds are correctly re-applied at each step.                                                                                                                                                                                                                                                                                            | CLP     |

### SS1.7 solve Conformance

| Test Name                                       | Input Scenario                             | Expected Observable Behavior                                                                                                                                          | Variant |
| ----------------------------------------------- | ------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| `test_solver_highs_solve_dual_values`           | Load shared fixture. Solve without cuts.   | Duals: $\pi_0 = -100.0$, $\pi_1 = 50.0$ within 1e-6 absolute tolerance. Dual sign follows canonical convention per [Solver Abstraction SS8](./solver-abstraction.md). | HiGHS   |
| `test_solver_clp_solve_dual_values`             | Same as above.                             | Duals: $\pi_0 = -100.0$, $\pi_1 = 50.0$ within 1e-6. Sign normalized to canonical convention by CLP implementation.                                                   | CLP     |
| `test_solver_highs_solve_dual_values_with_cuts` | Load shared fixture. Add both cuts. Solve. | Duals: $\pi_0 = -103.0$, $\pi_1 = 50.0$, $\pi_2 = 0.0$ (Cut 1, non-binding), $\pi_3 = 1.0$ (Cut 2, binding). All within 1e-6 (per SS1.2).                             | HiGHS   |
| `test_solver_clp_solve_dual_values_with_cuts`   | Same as above.                             | Same dual values as HiGHS within 1e-6. Both backends produce identical normalized duals.                                                                              | CLP     |
| `test_solver_highs_solve_reduced_costs`         | Load shared fixture. Solve without cuts.   | Reduced cost of $x_1$ is 1.0 (at lower bound, positive reduced cost confirms non-basic optimality). `LpSolution.reduced_costs` has length 3.                          | HiGHS   |
| `test_solver_clp_solve_reduced_costs`           | Same as above.                             | Same reduced costs as HiGHS within 1e-6.                                                                                                                              | CLP     |
| `test_solver_highs_solve_iterations_reported`   | Load shared fixture. Solve.                | `LpSolution.iterations >= 1`. `LpSolution.solve_time_seconds >= 0.0`. Iterations and timing are non-negative.                                                         | HiGHS   |
| `test_solver_clp_solve_iterations_reported`     | Same as above.                             | `LpSolution.iterations >= 1`. `LpSolution.solve_time_seconds >= 0.0`.                                                                                                 | CLP     |

### SS1.8 solve_with_basis Conformance

| Test Name                                          | Input Scenario                                                                                                                                                                                                                          | Expected Observable Behavior                                                                                                                                                                                 | Variant |
| -------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------- |
| `test_solver_highs_solve_with_basis_warm_start`    | Load shared fixture. Cold solve (record iterations as $I_{\text{cold}}$). Extract basis via `get_basis`. Reset. Load same fixture again. Call `solve_with_basis` with the cached basis (record $I_{\text{warm}}$).                      | Warm-start solve returns `Ok` with same objective (100.0) and same primal values. $I_{\text{warm}} \leq I_{\text{cold}}$. Warm-starting from the optimal basis of the same LP should require 0-1 iterations. | HiGHS   |
| `test_solver_clp_solve_with_basis_warm_start`      | Same as above.                                                                                                                                                                                                                          | Same: objective = 100.0, $I_{\text{warm}} \leq I_{\text{cold}}$.                                                                                                                                             | CLP     |
| `test_solver_highs_solve_with_basis_cut_extension` | Load shared fixture. Solve. Extract basis (2 row statuses). Load same fixture. Add both cuts (now 4 rows). Extend basis: structural row statuses from cache, new dynamic constraint rows initialized as Basic. Call `solve_with_basis`. | Solve returns `Ok`. Objective = 162.0 (cuts active). The basis extension per [Solver Abstraction SS2.3](./solver-abstraction.md) allows warm-starting even when cuts are added.                              | HiGHS   |
| `test_solver_clp_solve_with_basis_cut_extension`   | Same as above.                                                                                                                                                                                                                          | Same: objective = 162.0.                                                                                                                                                                                     | CLP     |

### SS1.9 reset Conformance

| Test Name                                      | Input Scenario                                                                                                                  | Expected Observable Behavior                                                                                                                                                                                                               | Variant |
| ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------- |
| `test_solver_highs_reset_clears_state`         | Load shared fixture. Solve. Call `reset`. Attempt to solve without loading a model.                                             | After reset, solver holds no model. Attempting to solve without a prior `load_model` is a precondition violation (panic or error, implementation-defined). State is clean per [Solver Interface Trait SS2.6](./solver-interface-trait.md). | HiGHS   |
| `test_solver_clp_reset_clears_state`           | Same as above.                                                                                                                  | Same behavior: no model after reset.                                                                                                                                                                                                       | CLP     |
| `test_solver_highs_reset_preserves_statistics` | Load shared fixture. Solve twice (2 solves). Record `statistics().solve_count`. Call `reset`. Check `statistics().solve_count`. | `solve_count` after reset equals `solve_count` before reset (= 2). Statistics are preserved across reset per [Solver Interface Trait SS2.6](./solver-interface-trait.md).                                                                  | HiGHS   |
| `test_solver_clp_reset_preserves_statistics`   | Same as above.                                                                                                                  | Same: `solve_count = 2` is preserved after reset.                                                                                                                                                                                          | CLP     |

### SS1.10 get_basis Conformance

| Test Name                                | Input Scenario                                                                                                                         | Expected Observable Behavior                                                                                                                                                                                        | Variant |
| ---------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| `test_solver_highs_get_basis_dimensions` | Load shared fixture. Solve. Call `get_basis`.                                                                                          | `basis.col_status.len() == 3` (3 columns). `basis.row_status.len() == 2` (2 static rows, no dynamic constraints). Basis dimensions match loaded LP per [Solver Interface Trait SS2.7](./solver-interface-trait.md). | HiGHS   |
| `test_solver_clp_get_basis_dimensions`   | Same as above.                                                                                                                         | Same dimensions: 3 column statuses, 2 row statuses.                                                                                                                                                                 | CLP     |
| `test_solver_highs_get_basis_roundtrip`  | Load shared fixture. Solve. Extract basis via `get_basis`. Reset. Load same fixture. Call `solve_with_basis` with the extracted basis. | Solve returns `Ok` with 0 or 1 simplex iterations. The basis round-trips through extract-reload without information loss.                                                                                           | HiGHS   |
| `test_solver_clp_get_basis_roundtrip`    | Same as above.                                                                                                                         | Same: 0 or 1 iterations on basis reload.                                                                                                                                                                            | CLP     |
| `test_solver_highs_get_basis_with_cuts`  | Load shared fixture. Add both cuts (4 total rows). Solve. Call `get_basis`.                                                            | `basis.col_status.len() == 3`. `basis.row_status.len() == 4` (2 structural + 2 dynamic constraint rows). Basis includes dynamic constraint row statuses per [Solver Abstraction SS9](./solver-abstraction.md).      | HiGHS   |
| `test_solver_clp_get_basis_with_cuts`    | Same as above.                                                                                                                         | Same dimensions: 3 column statuses, 4 row statuses.                                                                                                                                                                 | CLP     |

### SS1.11 statistics Conformance

| Test Name                                | Input Scenario                                                          | Expected Observable Behavior                                                                                                                                                                | Variant |
| ---------------------------------------- | ----------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| `test_solver_highs_statistics_initial`   | Construct a fresh solver instance. Call `statistics` before any solves. | `solve_count = 0`, `success_count = 0`, `failure_count = 0`, `total_iterations = 0`, `retry_count = 0`, `total_solve_time_seconds = 0.0`. All counters start at zero.                       | HiGHS   |
| `test_solver_clp_statistics_initial`     | Same as above.                                                          | Same: all counters zero.                                                                                                                                                                    | CLP     |
| `test_solver_highs_statistics_increment` | Load shared fixture. Solve 3 times (all successful). Call `statistics`. | `solve_count = 3`. `success_count = 3`. `failure_count = 0`. `total_iterations >= 3` (at least 1 iteration per solve). `total_solve_time_seconds > 0.0`. Counters accumulate monotonically. | HiGHS   |
| `test_solver_clp_statistics_increment`   | Same as above.                                                          | Same: `solve_count = 3`, `success_count = 3`, `failure_count = 0`.                                                                                                                          | CLP     |

### SS1.12 name Conformance

| Test Name                                   | Input Scenario                                  | Expected Observable Behavior                                                 | Variant |
| ------------------------------------------- | ----------------------------------------------- | ---------------------------------------------------------------------------- | ------- |
| `test_solver_highs_name_returns_identifier` | Construct a HiGHS solver instance. Call `name`. | Returns `"highs"`. Non-empty `&'static str`. Value is constant across calls. | HiGHS   |
| `test_solver_clp_name_returns_identifier`   | Construct a CLP solver instance. Call `name`.   | Returns `"clp"`. Non-empty `&'static str`. Value is constant across calls.   | CLP     |

## SS2. Cross-Solver Equivalence Tests

These tests verify that both solver backends produce equivalent results when given the same LP input. Equivalence is the central claim of the solver abstraction: the SDDP algorithm produces the same policy regardless of which backend is active. Each test loads the same LP into both HiGHS and CLP, solves, and compares results.

**Tolerances:**

| Quantity         | Tolerance | Type     | Rationale                                                  |
| ---------------- | --------- | -------- | ---------------------------------------------------------- |
| Objective value  | $10^{-8}$ | Relative | Both solvers minimize the same LP                          |
| Primal values    | $10^{-8}$ | Absolute | State transfer requires tight agreement                    |
| Dual values      | $10^{-6}$ | Absolute | Duals feed cut coefficients; moderate tolerance sufficient |
| Warm-start iters | $2\times$ | Ratio    | Implementation-specific iteration counts may differ        |

| Test Name                                           | Input Scenario                                                                                                       | Expected Observable Behavior                                                                                                                                                                          |
| --------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `test_solver_cross_solve_objective_agreement`       | Load shared fixture into both HiGHS and CLP. Solve both (no cuts).                                                   | Relative objective difference $\lvert z^*_{\text{highs}} - z^*_{\text{clp}} \rvert / \lvert z^*_{\text{highs}} \rvert < 10^{-8}$. Both return 100.0.                                                  |
| `test_solver_cross_solve_primal_agreement`          | Same LP, both solvers.                                                                                               | $\lvert x^*_{i,\text{highs}} - x^*_{i,\text{clp}} \rvert < 10^{-8}$ for all $i \in \{0, 1, 2\}$. Both return $(6.0, 0.0, 2.0)$.                                                                       |
| `test_solver_cross_solve_dual_agreement`            | Same LP, both solvers. No cuts.                                                                                      | $\lvert \pi_{j,\text{highs}} - \pi_{j,\text{clp}} \rvert < 10^{-6}$ for all $j \in \{0, 1\}$. Both return $(-100.0, 50.0)$ after normalization per [Solver Abstraction SS8](./solver-abstraction.md). |
| `test_solver_cross_add_cuts_objective_agreement`    | Load shared fixture into both. Add both cuts (SS1.2). Solve.                                                         | Relative objective difference $\lvert z^*_{\text{highs}} - z^*_{\text{clp}} \rvert / \lvert z^*_{\text{highs}} \rvert < 10^{-8}$. Both return 162.0.                                                  |
| `test_solver_cross_add_cuts_dual_agreement`         | Load shared fixture into both. Add both cuts. Solve.                                                                 | $\lvert \pi_{j,\text{highs}} - \pi_{j,\text{clp}} \rvert < 10^{-6}$ for all $j \in \{0, 1, 2, 3\}$. Both return $(-103.0, 50.0, 0.0, 1.0)$.                                                           |
| `test_solver_cross_patch_rhs_objective_agreement`   | Load shared fixture into both. Add both cuts. Patch Row 0 RHS to 4.0. Solve.                                         | Relative objective difference $\lvert z^*_{\text{highs}} - z^*_{\text{clp}} \rvert / \lvert z^*_{\text{highs}} \rvert < 10^{-8}$. Both return 368.0.                                                  |
| `test_solver_cross_warm_start_iteration_comparison` | Load shared fixture into both. Cold solve. Extract basis. Reset. Reload. Warm-start solve. Compare iteration counts. | Both backends achieve reduced iterations with warm start. $I_{\text{warm,highs}} \leq 2 \times I_{\text{warm,clp}}$ and $I_{\text{warm,clp}} \leq 2 \times I_{\text{warm,highs}}$ (within 2x).        |

## SS3. Error Path Tests

These tests verify that solver backends correctly identify and report infeasible and unbounded LPs with the appropriate `SolverError` variants from [Solver Interface Trait SS3](./solver-interface-trait.md).

### SS3.1 Infeasible LP

**LP construction:** A 1-variable LP with contradictory bounds.

$$\min \; x_0$$

subject to no constraints, with column bounds $x_0 \in [5, 3]$ (lower bound > upper bound, infeasible).

**StageTemplate data:**

| Field             | Value  |
| ----------------- | ------ |
| `num_cols`        | 1      |
| `num_rows`        | 0      |
| `num_nz`          | 0      |
| `col_starts`      | [0, 0] |
| `row_indices`     | []     |
| `values`          | []     |
| `col_lower`       | [5.0]  |
| `col_upper`       | [3.0]  |
| `objective`       | [1.0]  |
| `row_lower`       | []     |
| `row_upper`       | []     |
| `n_state`         | 1      |
| `n_dual_relevant` | 0      |

| Test Name                            | Input Scenario                     | Expected Observable Behavior                                                                                                                    | Variant |
| ------------------------------------ | ---------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| `test_solver_highs_solve_infeasible` | Load infeasible LP (SS3.1). Solve. | Returns `Err(SolverError::Infeasible)`. The LP has no feasible point because $x_0$ cannot simultaneously satisfy $x_0 \geq 5$ and $x_0 \leq 3$. | HiGHS   |
| `test_solver_clp_solve_infeasible`   | Same as above.                     | Returns `Err(SolverError::Infeasible)`.                                                                                                         | CLP     |

### SS3.2 Unbounded LP

**LP construction:** A 1-variable LP with no lower bound and a minimization objective that drives the variable to $-\infty$.

$$\min \; -x_0$$

subject to no constraints, with column bounds $x_0 \in (-\infty, +\infty)$.

**StageTemplate data:**

| Field             | Value       |
| ----------------- | ----------- |
| `num_cols`        | 1           |
| `num_rows`        | 0           |
| `num_nz`          | 0           |
| `col_starts`      | [0, 0]      |
| `row_indices`     | []          |
| `values`          | []          |
| `col_lower`       | [$-\infty$] |
| `col_upper`       | [$+\infty$] |
| `objective`       | [-1.0]      |
| `row_lower`       | []          |
| `row_upper`       | []          |
| `n_state`         | 1           |
| `n_dual_relevant` | 0           |

| Test Name                           | Input Scenario                    | Expected Observable Behavior                                                                                     | Variant |
| ----------------------------------- | --------------------------------- | ---------------------------------------------------------------------------------------------------------------- | ------- |
| `test_solver_highs_solve_unbounded` | Load unbounded LP (SS3.2). Solve. | Returns `Err(SolverError::Unbounded)`. The objective $-x_0$ is unbounded below because $x_0$ has no upper bound. | HiGHS   |
| `test_solver_clp_solve_unbounded`   | Same as above.                    | Returns `Err(SolverError::Unbounded)`.                                                                           | CLP     |

## SS4. LP Lifecycle Tests

These tests verify the full operational lifecycle of a solver instance: load, add cuts, patch, solve, extract basis, warm-start solve, reset, and reload. The lifecycle test ensures that the methods compose correctly in the sequence prescribed by [Solver Abstraction SS11.2](./solver-abstraction.md).

| Test Name                                          | Lifecycle Steps                                                                                                                                                                                                                                                                                                                                          | Expected Observable Behavior at Each Step                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | Variant |
| -------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| `test_solver_highs_lifecycle_full_cycle`           | (1) Construct solver. (2) `load_model` with shared fixture. (3) `solve` cold. (4) `add_rows` with both cuts. (5) `solve` cold. (6) `get_basis`. (7) `set_row_bounds` Row 0 RHS to 4.0. (8) `solve_with_basis` using basis from step 6. (9) Verify `statistics`. (10) `reset`. (11) `load_model` with shared fixture (original RHS=6). (12) `solve` cold. | (1) Fresh instance. (2) Model loaded. (3) Objective = 100.0, primal = (6, 0, 2). (4) Two cuts appended at rows 2-3. (5) Objective = 162.0, primal = (6, 62, 2). (6) Basis has 3 col statuses + 4 row statuses. (7) Row 0 RHS changed to 4.0. (8) Objective = 368.0, primal = (4, 68, 6); warm-start used. (9) `solve_count >= 3`, `success_count >= 3`, `failure_count = 0`, `total_solve_time_seconds > 0`. (10) Clean state; statistics preserved. (11) Fresh model loaded (no cuts). (12) Objective = 100.0, primal = (6, 0, 2); confirms clean state after reset. | HiGHS   |
| `test_solver_clp_lifecycle_full_cycle`             | Same steps as above.                                                                                                                                                                                                                                                                                                                                     | Same expected behavior at each step.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | CLP     |
| `test_solver_highs_lifecycle_repeated_patch_solve` | (1) `load_model` with shared fixture. (2) `add_rows` with both cuts. (3) `solve` with RHS=6 (objective=162.0). (4) `set_row_bounds` Row 0 to 4.0. (5) `solve` (objective=368.0). (6) `set_row_bounds` Row 0 to 8.0. (7) `solve` (objective=?).                                                                                                           | (3) Objective = 162.0. (5) Objective = 368.0. (7) With $x_0=8$: Row 1 gives $x_2 = 14 - 16 = -2$, but $x_2 \geq 0$, so $x_2 = 0$ is infeasible for equality. This means the equality power balance cannot be satisfied with $x_0 = 8$ and $x_2 \geq 0$ when demand is 14 and productivity is 2, because $2(8) = 16 > 14$. The LP becomes infeasible: returns `Err(SolverError::Infeasible { .. })`.                                                                                                                                                                   | HiGHS   |
| `test_solver_clp_lifecycle_repeated_patch_solve`   | Same steps as above.                                                                                                                                                                                                                                                                                                                                     | Same: step 7 returns `Err(SolverError::Infeasible { .. })`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | CLP     |

## SS5. Dual Normalization Verification

These tests specifically target the dual normalization contract from [Solver Abstraction SS8](./solver-abstraction.md) and [Solver Interface Trait SS7](./solver-interface-trait.md). The canonical sign convention states that a positive dual on a $\leq$ constraint means increasing the RHS increases the objective ($\partial z^* / \partial b > 0$). Both backends must produce identical normalized dual values regardless of their native sign convention.

**Verification approach:** The shared fixture has two equality constraints with hand-computed dual values derived from sensitivity analysis. The cut coefficient formula $\beta_t^k = W_t^\top \pi_t^*$ requires correct dual signs for the cut-relevant rows `[0, n_dual_relevant)`.

| Test Name                                                | Input Scenario                                                                                                        | Expected Observable Behavior                                                                                                                                                                           |
| -------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `test_solver_highs_dual_normalization_dual_relevant_row` | Load shared fixture. Solve. Extract dual for Row 0 (dual-relevant, state-fixing).                                     | $\pi_0 = -100.0$ within 1e-6. The state-fixing row dual propagates as the cut coefficient for the state variable: $\beta = W_t^\top \pi^* = (-100.0)$. A sign error here would produce divergent cuts. |
| `test_solver_clp_dual_normalization_dual_relevant_row`   | Same as above.                                                                                                        | $\pi_0 = -100.0$ within 1e-6. Same normalized value as HiGHS regardless of CLP's native dual convention.                                                                                               |
| `test_solver_highs_dual_normalization_sensitivity_check` | Load shared fixture. Solve to get $z^* = 100.0$ with RHS_0 = 6. Patch Row 0 RHS to 6.01. Solve again to get $z^{**}$. | Finite-difference check: $(z^{**} - z^*) / 0.01 \approx \pi_0 = -100.0$ within 1e-2 tolerance. The dual accurately predicts the objective sensitivity to RHS perturbation.                             |
| `test_solver_clp_dual_normalization_sensitivity_check`   | Same as above.                                                                                                        | Same finite-difference result within 1e-2. Both backends' duals predict the correct sensitivity.                                                                                                       |
| `test_solver_highs_dual_normalization_with_binding_cut`  | Load shared fixture. Add both cuts. Solve. Extract dual for Cut 2 (Row 3, binding).                                   | $\pi_3 = 1.0$ within 1e-6. The binding cut dual is positive, confirming that tightening the cut RHS (increasing $\alpha$) increases the objective.                                                     |
| `test_solver_clp_dual_normalization_with_binding_cut`    | Same as above.                                                                                                        | $\pi_3 = 1.0$ within 1e-6. Same normalized value.                                                                                                                                                      |

## Cross-References

- [Solver Interface Trait](./solver-interface-trait.md) -- Trait definition (SS1), method contracts for `load_model` (SS2.1), `add_rows` (SS2.2), `set_row_bounds` (SS2.3), `set_col_bounds` (SS2.3a), `solve` (SS2.4), `solve_with_basis` (SS2.5), `reset` (SS2.6), `get_basis` (SS2.7), `statistics` (SS2.8), `name` (SS2.9), `SolverError` enum (SS3), `LpSolution` type (SS4.1), `Basis` type (SS4.2), `SolverStatistics` type (SS4.3), `StageTemplate` type (SS4.4), `RowBatch` type (SS4.5), dual normalization contract (SS7)
- [Solver Abstraction](./solver-abstraction.md) -- LP layout convention (SS2), column layout (SS2.1), row layout with dual-extraction region (SS2.2), basis persistence with cut boundary (SS2.3), solver interface contract (SS4), cut pool design (SS5), error categories (SS6), retry logic contract (SS7), dual normalization canonical convention (SS8), basis storage (SS9), compile-time solver selection (SS10), stage template and rebuild strategy (SS11)
- [HiGHS Implementation](./solver-highs-impl.md) -- HiGHS-specific API mapping, retry strategy, dual sign convention before normalization
- [CLP Implementation](./solver-clp-impl.md) -- CLP-specific API mapping, C++ wrapper strategy, native dual sign convention
- [Backend Testing](../hpc/backend-testing.md) -- Conformance test suite structure, parameterized-by-backend pattern, interchangeability verification approach
- [Risk Measure Testing](./risk-measure-testing.md) -- Sibling conformance test spec: shared fixture design, requirements tables, variant equivalence tests
- [Cut Selection Testing](./cut-selection-testing.md) -- Sibling conformance test spec: hand-computable fixtures, cross-variant comparison
- [Training Loop](./training-loop.md) -- Forward pass (SS4) and backward pass (SS6) that drive the load-patch-solve-basis lifecycle tested in SS4
- [Cut Management Implementation](./cut-management-impl.md) -- Cut pool activity bitmap (SS1.1), CSR assembly for `addRows` (SS1) that produces the `RowBatch` input tested in SS1.5
- [LP Formulation](../math/lp-formulation.md) -- Constraint structure defining which row duals are cut-relevant; the fixture Row 0 models the state-linking constraint type
- [Binary Formats](../data-model/binary-formats.md) -- Cut pool memory layout (SS3.4) and CSC/CSR format conventions used in the fixture data
- [Solver Workspaces](./solver-workspaces.md) -- Thread-local solver infrastructure (SS1), per-stage basis cache (SS1.5) that the lifecycle tests in SS4 exercise
