# Solver Interface Trait

## Purpose

This spec defines the `SolverInterface` trait -- the backend abstraction through which optimization algorithms perform LP operations: loading models, adding constraint rows, updating row and column bounds, solving, warm-starting from a cached basis, and extracting solution data. The solver is resolved as a generic type parameter at compile time, following the same monomorphization pattern used by the Communicator trait ([Communicator Trait §3](../hpc/communicator-trait.md)). Because LP solvers wrap FFI calls to C libraries (HiGHS, CLP) on a per-thread exclusive-ownership basis, the trait requires `Send` but not `Sync`. The operations contract originates from [Solver Abstraction SS4](./solver-abstraction.md), which defines the behavioral requirements validated against both reference solver APIs.

> **Convention: Rust traits as specification guidelines.** The Rust trait definitions, method signatures, and struct declarations throughout this specification corpus serve as _guidelines for implementation_, not as absolute source-of-truth contracts that must be reproduced verbatim. Their purpose is twofold: (1) to express behavioral contracts, preconditions, postconditions, and type-level invariants more precisely than prose alone, and (2) to anchor conformance test suites that verify backend interchangeability (see [Backend Testing §1](../hpc/backend-testing.md)). Implementation may diverge in naming, parameter ordering, error representation, or internal organization when practical considerations demand it -- provided the behavioral contracts and conformance tests continue to pass. When a trait signature and a prose description conflict, the prose description (which captures the domain intent) takes precedence; the conflict should be resolved by updating the trait signature. This convention applies to all trait-bearing specification documents in `src/specs/`.

## 1. Trait Definition

The solver interface is modeled as a Rust trait with 10 methods. Each method corresponds to one operation from the solver interface contract ([Solver Abstraction SS4.1](./solver-abstraction.md)), plus a `name()` method for diagnostics.

```rust
/// Backend abstraction for LP solver operations in optimization algorithms.
///
/// Implementations wrap a single solver instance (HiGHS `void*` handle,
/// CLP `Clp_Simplex*` handle, etc.) and encapsulate all solver-specific
/// behavior: API calling conventions, retry logic, dual normalization,
/// and basis format translation.
///
/// The trait requires `Send` (transferable between threads) but NOT `Sync`.
/// Each solver instance is exclusively owned by one thread at a time --
/// LP solvers are not thread-safe. The thread-local workspace pattern
/// (see Solver Workspaces SS1.1) ensures exclusive ownership without
/// runtime synchronization.
pub trait SolverInterface: Send {
    /// Bulk-load a pre-assembled structural LP into the solver.
    ///
    /// The stage template contains the LP matrix in CSC form, column and
    /// row bounds, and objective coefficients. This operation replaces
    /// any previously loaded model. The LP layout follows the convention
    /// defined in [Solver Abstraction SS2](./solver-abstraction.md).
    ///
    /// Maps to `Highs_passLp` (HiGHS) or `Clp_loadProblem` (CLP).
    fn load_model(&mut self, template: &StageTemplate);

    /// Batch-add constraint rows to the LP (dynamic constraint region).
    ///
    /// The cut batch contains active cuts in CSR format, ready for a
    /// single `addRows` call. Cuts are appended at the bottom of the
    /// constraint matrix per [Solver Abstraction SS2.2](./solver-abstraction.md).
    ///
    /// Maps to `Highs_addRows` (HiGHS) or `Clp_addRows` (CLP).
    fn add_rows(&mut self, cuts: &RowBatch);

    /// Update row bounds (constraint RHS values).
    ///
    /// Takes three parallel slices: `indices` (row indices to patch),
    /// `lower` (new lower bounds), and `upper` (new upper bounds).
    /// Updates inflow RHS, state-fixing constraints, and noise-fixing
    /// values without modifying the structural LP. For equality
    /// constraints, set lower[i] = upper[i] = value.
    /// This is the primary modification performed between successive
    /// solves at the same stage (within-stage incremental updates per
    /// [Solver Abstraction SS11.4](./solver-abstraction.md)).
    ///
    /// Maps to `Highs_changeRowsBoundsBySet` (HiGHS) or mutable pointer
    /// access via `Clp_rowLower()`/`Clp_rowUpper()` (CLP).
    fn set_row_bounds(&mut self, indices: &[usize], lower: &[f64], upper: &[f64]);

    /// Update column bounds (variable lower/upper bounds).
    ///
    /// Takes three parallel slices: `indices` (column indices to patch),
    /// `lower` (new lower bounds), and `upper` (new upper bounds).
    /// Updates variable bounds without modifying the structural LP.
    /// This method is not used in minimal viable SDDP but is included
    /// for completeness — future extensions (e.g., thermal unit
    /// commitment bounds, battery state-of-charge limits) may require
    /// per-scenario column bound updates.
    ///
    /// Maps to `Highs_changeColsBoundsBySet` (HiGHS) or mutable pointer
    /// access via `Clp_colLower()`/`Clp_colUpper()` (CLP).
    fn set_col_bounds(&mut self, indices: &[usize], lower: &[f64], upper: &[f64]);

    /// Solve the loaded LP, returning a zero-copy view or terminal error.
    ///
    /// Hot-path method encapsulating internal retry logic (see SS6).
    /// Returns either a valid solution view with normalized duals
    /// (see SS7) or a terminal error. The caller never sees intermediate
    /// retry attempts. The returned `SolutionView` borrows solver-internal
    /// buffers and is valid until the next `&mut self` call. Call
    /// `SolutionView::to_owned()` when the solution must outlive the borrow.
    ///
    /// Maps to `Highs_run` (HiGHS) or `Clp_dual`/`Clp_initialDualSolve`
    /// (CLP).
    fn solve(&mut self) -> Result<SolutionView<'_>, SolverError>;

    /// Inject a basis and solve, returning a zero-copy `SolutionView`.
    ///
    /// Loads the provided basis into the solver before invoking the
    /// solve sequence. Status codes in `basis` are injected directly
    /// without per-element enum translation. The basis structure splits
    /// at the cut boundary: static rows are reused directly, new dynamic
    /// constraint rows are initialized as Basic per
    /// [Solver Abstraction SS2.3](./solver-abstraction.md).
    /// On success the returned view borrows solver-internal buffers and
    /// is valid until the next `&mut self` call. Call
    /// `SolutionView::to_owned()` when the solution must outlive the borrow.
    ///
    /// Maps to `Highs_setBasis` + `Highs_run` (HiGHS) or
    /// `Clp_copyinStatus` + `Clp_dual` (CLP).
    fn solve_with_basis(
        &mut self,
        basis: &Basis,
    ) -> Result<SolutionView<'_>, SolverError>;

    /// Clear all internal solver state.
    ///
    /// Resets the solver to a clean state: clears cached basis,
    /// factorization workspace, and any accumulated internal state.
    /// After reset, the solver is ready for a fresh `load_model` call.
    ///
    /// Maps to `Highs_clearSolver` (HiGHS) or model reconstruction
    /// (CLP).
    fn reset(&mut self);

    /// Write solver-native i32 status codes into a caller-owned Basis buffer.
    ///
    /// The caller pre-allocates a `Basis` with `Basis::new` and reuses it
    /// across iterations, eliminating per-element enum translation overhead.
    /// The buffer is not resized by this method. The implementation writes
    /// into the first `num_cols` entries of `out.col_status` and the first
    /// `num_rows` entries of `out.row_status`. Panics if no model is loaded.
    /// Stored in the original problem space (not presolved) per
    /// [Solver Abstraction SS9](./solver-abstraction.md).
    ///
    /// Maps to `Highs_getBasis` (HiGHS) or `Clp_statusArray` (CLP).
    fn get_basis(&mut self, out: &mut Basis);

    /// Return accumulated solve metrics.
    ///
    /// Provides total solve count, total simplex iterations, retry
    /// count, failure count, and cumulative wall-clock time. Counters
    /// accumulate across all solves performed by this instance since
    /// construction.
    fn statistics(&self) -> SolverStatistics;

    /// Return the solver backend name.
    ///
    /// Used for logging, diagnostics, and checkpoint metadata.
    /// Returns a static string such as `"highs"` or `"clp"`.
    fn name(&self) -> &'static str;
}
```

**Thread safety model:** The `Send` bound allows solver instances to be transferred between threads (e.g., during thread pool initialization), but the absence of `Sync` prevents concurrent access. This matches the reality of C-library solver handles, which maintain mutable internal state (factorization workspace, working arrays) that is not safe to share. The thread-local workspace pattern in [Solver Workspaces SS1.1](./solver-workspaces.md) ensures each OpenMP thread owns exactly one solver instance for the entire training run.

**Mutability:** All methods that modify solver state or write to internal buffers (`load_model`, `add_rows`, `set_row_bounds`, `set_col_bounds`, `solve`, `solve_with_basis`, `reset`, `get_basis`) take `&mut self`. `get_basis` requires `&mut self` because it writes to internal scratch buffers during extraction; it also takes a `&mut Basis` output parameter so the caller can pre-allocate and reuse the buffer across iterations without per-solve allocation. Read-only accessors (`statistics`, `name`) take `&self`.

## 2. Method Contracts

### 2.1 load_model

`load_model` bulk-loads a pre-assembled structural LP into the solver instance. This is the first step of the LP rebuild sequence at each stage transition ([Solver Abstraction SS11.2](./solver-abstraction.md), step 1). The stage template is built once at initialization and shared read-only across all threads within an MPI rank.

**Preconditions:**

| Condition                                           | Description                                                                            |
| --------------------------------------------------- | -------------------------------------------------------------------------------------- |
| `template` contains a valid CSC matrix              | Column starts, row indices, and values arrays are consistent; no out-of-bounds indices |
| `template` follows the LP layout convention         | Column and row ordering per [Solver Abstraction SS2](./solver-abstraction.md)          |
| `template.num_cols > 0` and `template.num_rows > 0` | Non-empty LP                                                                           |

**Postconditions:**

| Condition                                      | Description                                                                                                |
| ---------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| Solver holds the structural LP from `template` | Previous model (if any) is fully replaced                                                                  |
| No cuts are present                            | The loaded model contains only structural constraints; constraint rows are added separately via `add_rows` |
| Solver basis is cleared                        | Any cached basis from a previous model is invalidated                                                      |

**Infallibility:** This method does not return `Result`. The stage template is validated during initialization ([Solver Abstraction SS11.1](./solver-abstraction.md)); passing an invalid template is a programming error (panic on violation).

### 2.2 add_rows

`add_rows` appends constraint rows to the dynamic constraint region in a single batch call. In SDDP, this is used to add Benders cuts. This is step 2 of the LP rebuild sequence ([Solver Abstraction SS11.2](./solver-abstraction.md)). The row batch is assembled from the cut pool's activity bitmap for the current stage.

**Preconditions:**

| Condition                                                      | Description                                                          |
| -------------------------------------------------------------- | -------------------------------------------------------------------- |
| `load_model` has been called                                   | A structural LP must be loaded before adding cuts                    |
| `cuts` contains valid CSR row data                             | Row starts, column indices, values, and bounds arrays are consistent |
| Cut column indices reference valid columns in the loaded model | Indices within `[0, num_cols)`                                       |

**Postconditions:**

| Condition                                                                  | Description                                                                                |
| -------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| Active cuts are appended as rows at `[n_static, n_static + cuts.num_rows)` | Cut row positions follow [Solver Abstraction SS2.2](./solver-abstraction.md) bottom region |
| Structural rows `[0, n_static)` are unchanged                              | Adding cuts does not modify the structural LP                                              |
| Solver basis is not automatically set                                      | Caller must use `solve_with_basis` to apply a cached basis                                 |

**Infallibility:** This method does not return `Result`. The cut batch is assembled from the pre-validated cut pool ([Solver Abstraction SS5](./solver-abstraction.md)); invalid CSR data is a programming error (panic on violation).

### 2.3 set_row_bounds

`set_row_bounds` updates row bounds (constraint RHS values) without structural LP changes. This is step 3 of the LP rebuild sequence and the primary modification between successive solves at the same stage ([Solver Abstraction SS11.4](./solver-abstraction.md)). Takes three parallel slices: `indices` (row indices to patch), `lower` (new lower bounds), and `upper` (new upper bounds). All three slices must have equal length. For equality constraints (water balance, lag fixing, noise fixing), set `lower[i] = upper[i] = value`.

**Preconditions:**

| Condition                                    | Description                                           |
| -------------------------------------------- | ----------------------------------------------------- |
| `load_model` has been called                 | A model must be loaded before patching                |
| All indices in `indices` are valid           | Each index references a valid row in the loaded model |
| All values in `lower` and `upper` are finite | No NaN or infinity                                    |
| `lower[i] <= upper[i]` for each `i`          | Lower bound does not exceed upper bound               |

**Postconditions:**

| Condition                                                    | Description                                         |
| ------------------------------------------------------------ | --------------------------------------------------- |
| Row lower and upper bounds at each patched index are updated | The LP reflects the current scenario realization    |
| Non-patched rows are unchanged                               | Only the specified row indices are modified         |
| Column bounds are unchanged                                  | Row patching does not affect column bounds          |
| Solver basis is preserved                                    | Patching does not invalidate a previously set basis |

**Infallibility:** This method does not return `Result`. Patch indices are computed from the LP layout convention ([Solver Abstraction SS2](./solver-abstraction.md)); out-of-bounds indices are a programming error (panic on violation).

**Solver API mapping:**

| Solver | API Call                                                             |
| ------ | -------------------------------------------------------------------- |
| HiGHS  | `Highs_changeRowsBoundsBySet(model, num_set, indices, lower, upper)` |
| CLP    | Mutable `double*` via `Clp_rowLower()` / `Clp_rowUpper()`            |

### 2.3a set_col_bounds

`set_col_bounds` updates column bounds (variable lower/upper bounds) without structural LP changes. Takes three parallel slices: `indices` (column indices to patch), `lower` (new lower bounds), and `upper` (new upper bounds). All three slices must have equal length. This method is not used in minimal viable SDDP but is included for completeness -- future extensions (e.g., thermal unit commitment bounds, battery state-of-charge limits) may require per-scenario column bound updates.

**Preconditions:**

| Condition                                    | Description                                              |
| -------------------------------------------- | -------------------------------------------------------- |
| `load_model` has been called                 | A model must be loaded before patching                   |
| All indices in `indices` are valid           | Each index references a valid column in the loaded model |
| All values in `lower` and `upper` are finite | No NaN or infinity                                       |
| `lower[i] <= upper[i]` for each `i`          | Lower bound does not exceed upper bound                  |

**Postconditions:**

| Condition                                                       | Description                                         |
| --------------------------------------------------------------- | --------------------------------------------------- |
| Column lower and upper bounds at each patched index are updated | The LP reflects the current scenario realization    |
| Non-patched columns are unchanged                               | Only the specified column indices are modified      |
| Row bounds are unchanged                                        | Column patching does not affect row bounds          |
| Solver basis is preserved                                       | Patching does not invalidate a previously set basis |

**Infallibility:** This method does not return `Result`. Patch indices are computed from the LP layout convention ([Solver Abstraction SS2](./solver-abstraction.md)); out-of-bounds indices are a programming error (panic on violation).

**Solver API mapping:**

| Solver | API Call                                                             |
| ------ | -------------------------------------------------------------------- |
| HiGHS  | `Highs_changeColsBoundsBySet(model, num_set, indices, lower, upper)` |
| CLP    | Mutable `double*` via `Clp_colLower()` / `Clp_colUpper()`            |

### 2.4 solve

`solve` invokes the LP solver with its internal retry logic and returns either a valid solution or a terminal error. This is the primary hot-path method -- called millions of times during a training run.

**Preconditions:**

| Condition                                             | Description              |
| ----------------------------------------------------- | ------------------------ |
| `load_model` has been called                          | A model must be loaded   |
| Cuts and scenario patches have been applied as needed | The LP is ready to solve |

**Postconditions (on `Ok`):**

| Condition                                                  | Description                                                                    |
| ---------------------------------------------------------- | ------------------------------------------------------------------------------ |
| `SolutionView.objective` is the optimal objective value    | Minimization sense                                                             |
| `SolutionView.primal` contains optimal primal values       | Length equals `num_cols`                                                       |
| `SolutionView.dual` contains normalized dual values        | Sign convention per [Solver Abstraction SS8](./solver-abstraction.md); see SS7 |
| `SolutionView.dual.len() == num_rows`                      | One dual per constraint (structural + cuts)                                    |
| `SolutionView` borrows solver-internal buffers (zero-copy) | Valid until the next `&mut self` call; call `to_owned()` to persist            |
| Solver basis reflects the optimal solution                 | Available via `get_basis()` after a successful solve                           |
| `SolverStatistics` counters are incremented                | Solve count, iteration count, and timing updated                               |

**Postconditions (on `Err`):**

| Condition                                              | Description                                                |
| ------------------------------------------------------ | ---------------------------------------------------------- |
| `SolverError` variant identifies the terminal failure  | After all retry attempts are exhausted (see SS6)           |
| Solver state is unspecified                            | The caller should call `reset()` before reusing the solver |
| `SolverStatistics.retry_count` reflects retry attempts | Retry attempts are tracked even on failure                 |

**Fallibility:** This method returns `Result<SolutionView<'_>, SolverError>`. LP solves wrap FFI calls to C libraries that may encounter numerical difficulties, infeasibility, or other solver-internal failures that cannot be prevented by precondition checks. `SolutionView` is a zero-copy borrow of solver-internal buffers; call `SolutionView::to_owned()` to convert to an owned `LpSolution` when the data must outlive the solver borrow.

### 2.5 solve_with_basis

`solve_with_basis` sets a cached basis for warm-starting before solving. This combines the `set basis` and `solve` operations into a single method to ensure the basis is applied atomically with the solve. Warm-starting from a cached basis typically reduces simplex iterations by 80-95% compared to cold starts.

**Preconditions:**

| Condition                                                                         | Description                                   |
| --------------------------------------------------------------------------------- | --------------------------------------------- |
| `load_model` has been called                                                      | A model must be loaded                        |
| `basis.col_status.len()` matches the loaded model's column count                  | Basis dimension matches current LP            |
| `basis.row_status.len()` matches the loaded model's row count (structural + cuts) | Basis covers all rows including appended cuts |

**Postconditions (on `Ok`):**

| Condition                                               | Description                                                   |
| ------------------------------------------------------- | ------------------------------------------------------------- |
| Same as `solve()` `Ok` postconditions                   | Valid `SolutionView` with normalized duals (zero-copy borrow) |
| Simplex iterations are typically reduced vs. cold start | Warm-start benefit is observable in `SolverStatistics`        |

**Postconditions (on `Err`):**

| Condition                                               | Description                                      |
| ------------------------------------------------------- | ------------------------------------------------ |
| Same as `solve()` `Err` postconditions                  | Terminal error after retry exhaustion            |
| Implementation may fall back to cold start during retry | Basis rejection is a valid retry escalation step |

**Fallibility:** Same as `solve()` -- returns `Result<SolutionView<'_>, SolverError>`.

**Basis dimension mismatch handling:** If the provided basis dimensions do not match the current LP (e.g., because cuts were added since the basis was saved), the solver implementation must handle this gracefully. Per [Solver Abstraction SS2.3](./solver-abstraction.md), the static portion of the basis is position-stable; only the dynamic constraint portion needs extension (new dynamic constraint rows initialized as Basic) or truncation.

### 2.6 reset

`reset` clears all internal solver state, returning the instance to a clean state equivalent to a freshly constructed solver. This is used for error recovery (after a terminal `SolverError`) or when switching between fundamentally different LP structures.

**Preconditions:** None. `reset` can be called at any time.

**Postconditions:**

| Condition                                     | Description                                               |
| --------------------------------------------- | --------------------------------------------------------- |
| Solver state is clean                         | No loaded model, no cached basis, no factorization        |
| `load_model` must be called before next solve | The solver cannot solve without a loaded model            |
| Statistics are preserved                      | `reset` does not zero the accumulated statistics counters |

**Infallibility:** This method does not return `Result`. Clearing solver state is a local operation with no failure modes.

### 2.7 get_basis

`get_basis` writes solver-native `i32` status codes into a caller-owned `Basis` buffer. The caller pre-allocates a `Basis` with `Basis::new` and reuses it across iterations, eliminating per-solve allocation and per-element enum translation overhead on the hot path. The basis is stored in the original problem space (not presolved) to ensure portability across solver versions and presolve strategies ([Solver Abstraction SS9](./solver-abstraction.md)).

**Preconditions:**

| Condition                                                | Description                                                               |
| -------------------------------------------------------- | ------------------------------------------------------------------------- |
| A successful `solve` or `solve_with_basis` has completed | A basis exists only after a successful solve                              |
| `out` is a pre-allocated `Basis`                         | Created via `Basis::new(num_cols, num_rows)` and reused across iterations |

**Postconditions:**

| Condition                                                  | Description                                                                                             |
| ---------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| `out.col_status[0..num_cols]` contains column status codes | Written in place; buffer is not resized                                                                 |
| `out.row_status[0..num_rows]` contains row status codes    | Includes both structural and dynamic constraint rows                                                    |
| Status values are in the canonical set                     | `AtLower`, `Basic`, `AtUpper`, `Free`, or `Fixed` per [Solver Abstraction SS9](./solver-abstraction.md) |

**Infallibility:** This method does not return `Result`. After a successful solve, the basis always exists and can be extracted. Calling `get_basis` without a prior successful solve is a programming error (panic on violation).

### 2.8 statistics

`statistics` returns accumulated solve metrics for this solver instance. The counters grow monotonically across all solves performed since construction (or since the last construction -- `reset` does not clear statistics).

**Preconditions:** None. Can be called at any time, including before any solves.

**Postconditions:**

| Condition                                      | Description                                            |
| ---------------------------------------------- | ------------------------------------------------------ |
| All fields are non-negative                    | Counters start at zero and only increment              |
| `solve_count >= success_count + failure_count` | Total solves decompose into successes and failures     |
| `retry_count` counts individual retry attempts | One retry attempt per escalation step per failed solve |

**Infallibility:** This method does not return `Result`. It reads internal counters with no failure modes.

### 2.9 name

`name` returns a static string identifying the solver backend. Used for logging, diagnostics, and checkpoint metadata.

**Preconditions:** None.

**Postconditions:**

| Condition                                | Description                                   |
| ---------------------------------------- | --------------------------------------------- |
| Returns a non-empty `&'static str`       | Lifetime is `'static` -- no allocation needed |
| Value is constant for the implementation | `"highs"`, `"clp"`, etc.                      |

**Infallibility:** This method does not return `Result`. It returns a compile-time constant.

## 3. Error Type

The `SolverError` enum categorizes terminal LP solve failures. These are the errors that reach the SDDP algorithm after all solver-internal retry logic has been exhausted. The six variants correspond to the error categories defined in [Solver Abstraction SS6](./solver-abstraction.md).

```rust
/// Terminal LP solve error returned after all retry attempts are exhausted.
///
/// The calling algorithm uses the variant to determine its response:
/// hard stop (`Infeasible`, `Unbounded`, `NumericalDifficulty`,
/// `InternalError`) or terminate with a diagnostic error
/// (`TimeLimitExceeded`, `IterationLimit`).
#[derive(Debug)]
pub enum SolverError {
    /// The LP has no feasible solution.
    ///
    /// Indicates a data error (inconsistent bounds or constraints) or
    /// a modeling error. The calling algorithm performs a hard stop.
    Infeasible,

    /// The LP objective is unbounded below.
    ///
    /// Indicates a modeling error (missing bounds, incorrect objective
    /// sign). The calling algorithm performs a hard stop.
    Unbounded,

    /// Solver encountered numerical difficulties that persisted through
    /// all retry attempts.
    ///
    /// The calling algorithm should log the error and perform a hard stop.
    NumericalDifficulty {
        /// Human-readable description of the numerical issue from the solver.
        message: String,
    },

    /// Per-solve wall-clock time budget exhausted.
    TimeLimitExceeded {
        /// Elapsed wall-clock time in seconds.
        elapsed_seconds: f64,
    },

    /// Solver simplex iteration limit reached.
    IterationLimit {
        /// Number of iterations performed.
        iterations: u64,
    },

    /// Unrecoverable solver-internal failure.
    ///
    /// Covers FFI panics, memory allocation failures within the solver,
    /// corrupted internal state, or any error not classifiable into the
    /// above categories. The calling algorithm logs and performs a hard stop.
    InternalError {
        /// Human-readable error description.
        message: String,
        /// Solver-specific error code, if available.
        error_code: Option<i32>,
    },
}
```

**Error-to-response mapping:**

| Variant               | Hard Stop | Diagnostic |
| --------------------- | :-------: | :--------: |
| `Infeasible`          |    Yes    |     No     |
| `Unbounded`           |    Yes    |     No     |
| `NumericalDifficulty` |    Yes    |     No     |
| `TimeLimitExceeded`   |    No     |    Yes     |
| `IterationLimit`      |    No     |    Yes     |
| `InternalError`       |    Yes    |     No     |

## 4. Supporting Types

### 4.1 SolutionView and LpSolution

`SolutionView<'a>` is the primary return type of `solve()` and `solve_with_basis()`. It borrows directly from solver-internal buffers (zero-copy), avoiding per-solve heap allocation on the hot path. The lifetime `'a` ties the view to the solver instance, enforced at compile time by the Rust borrow checker: the view is valid until the next `&mut self` call on the solver. Call `SolutionView::to_owned()` to convert to an owned `LpSolution` when the data must outlive the current borrow or survive a subsequent solver call.

```rust
/// Zero-copy view of an LP solution, borrowing directly from
/// solver-internal buffers.
///
/// Valid until the next mutating method call on the solver (any
/// `&mut self` call). Use `to_owned()` to convert to an owned
/// `LpSolution` when the solution data must outlive the current borrow.
///
/// All values are in the original (unscaled) problem space. Dual values
/// are normalized to the canonical sign convention per
/// [Solver Abstraction SS8](./solver-abstraction.md) -- see SS7.
#[derive(Debug, Clone, Copy)]
pub struct SolutionView<'a> {
    /// Optimal objective value (minimization sense).
    pub objective: f64,

    /// Primal variable values, indexed by column.
    /// Length equals `num_cols`. State variables occupy the contiguous
    /// prefix `[0, n_state)` per [Solver Abstraction SS2.1](./solver-abstraction.md).
    pub primal: &'a [f64],

    /// Dual multipliers (shadow prices), indexed by row.
    /// Length equals `num_rows` (structural + cuts). Cut-relevant
    /// constraint duals occupy the contiguous prefix `[0, n_dual_relevant)`
    /// per [Solver Abstraction SS2.2](./solver-abstraction.md).
    ///
    /// Sign convention: normalized per SS7 before returning.
    pub dual: &'a [f64],

    /// Reduced costs, indexed by column.
    /// Length equals `num_cols`.
    pub reduced_costs: &'a [f64],

    /// Number of simplex iterations performed for this solve.
    pub iterations: u64,

    /// Wall-clock solve time in seconds (excluding retry overhead).
    pub solve_time_seconds: f64,
}

/// Complete owned solution from a successful LP solve.
///
/// Produced by `SolutionView::to_owned()`. All values are in the
/// original (unscaled) problem space. Dual values are normalized to
/// the canonical sign convention per
/// [Solver Abstraction SS8](./solver-abstraction.md) -- see SS7.
pub struct LpSolution {
    /// Optimal objective value (minimization sense).
    pub objective: f64,

    /// Primal variable values, indexed by column.
    /// Length equals `num_cols`.
    pub primal: Vec<f64>,

    /// Dual multipliers (shadow prices), indexed by row.
    /// Length equals `num_rows` (structural + cuts).
    /// Sign convention: normalized per SS7 before returning.
    pub dual: Vec<f64>,

    /// Reduced costs, indexed by column.
    /// Length equals `num_cols`.
    pub reduced_costs: Vec<f64>,

    /// Number of simplex iterations performed for this solve.
    pub iterations: u64,

    /// Wall-clock solve time in seconds (excluding retry overhead).
    pub solve_time_seconds: f64,
}
```

### 4.2 Basis

```rust
/// Raw simplex basis stored as solver-native i32 status codes.
///
/// Each element is a solver-specific status code (e.g., HiGHS uses
/// 0=AtLower, 1=Basic, 2=AtUpper, 3=Free, 4=Fixed). The codes are
/// opaque to the calling algorithm — they are extracted from one solve
/// and passed back to the next via `solve_with_basis` for warm-starting.
///
/// Stored in the original problem space (not presolved) to ensure
/// portability across solver versions and presolve strategies
/// ([Solver Abstraction SS9](./solver-abstraction.md)).
pub struct Basis {
    /// Basis status codes for each column (variable), in solver-native encoding.
    pub col_status: Vec<i32>,

    /// Basis status codes for each row (constraint), in solver-native encoding.
    /// Includes both static rows and dynamic constraint rows.
    pub row_status: Vec<i32>,
}
```

### 4.3 SolverStatistics

```rust
/// Accumulated solve metrics for a single solver instance.
///
/// Counters grow monotonically from construction. Thread-local --
/// each thread owns one solver instance and accumulates its own
/// statistics. Aggregated across threads via reduction after training
/// completes.
///
/// `reset()` does **not** zero statistics counters. They persist across
/// model reloads for the lifetime of the solver instance.
pub struct SolverStatistics {
    /// Total number of `solve` and `solve_with_basis` calls.
    pub solve_count: u64,

    /// Number of solves that returned `Ok` (optimal solution found).
    pub success_count: u64,

    /// Number of solves that returned `Err` (terminal failure after retries).
    pub failure_count: u64,

    /// Total simplex iterations summed across all solves.
    pub total_iterations: u64,

    /// Total retry attempts summed across all failed solves.
    pub retry_count: u64,

    /// Cumulative wall-clock time spent in solver calls, in seconds.
    pub total_solve_time_seconds: f64,

    /// Number of times `solve_with_basis` fell back to cold-start due to
    /// basis rejection.
    pub basis_rejections: u64,

    /// Number of solves that returned optimal on the first attempt
    /// (before any retry). Enables first-try rate computation:
    /// `first_try_rate = first_try_successes / solve_count`.
    /// The complement `success_count - first_try_successes` gives the
    /// number of retried solves.
    pub first_try_successes: u64,

    /// Total number of `solve_with_basis` calls (basis offers).
    /// Combined with `basis_rejections`, enables basis hit rate computation:
    /// `basis_hit_rate = 1 - basis_rejections / basis_offered`.
    pub basis_offered: u64,

    /// Total number of `load_model` calls.
    pub load_model_count: u64,

    /// Total number of `add_rows` calls.
    pub add_rows_count: u64,

    /// Cumulative wall-clock time spent in `load_model` calls, in seconds.
    pub total_load_model_time_seconds: f64,

    /// Cumulative wall-clock time spent in `add_rows` calls, in seconds.
    pub total_add_rows_time_seconds: f64,

    /// Cumulative wall-clock time spent in `set_row_bounds` and
    /// `set_col_bounds` calls, in seconds.
    pub total_set_bounds_time_seconds: f64,

    /// Cumulative wall-clock time spent in `set_basis` FFI calls, in seconds.
    /// Accumulated by `solve_with_basis` around the basis installation step.
    /// `solve()` (without basis) does not increment this counter.
    pub total_basis_set_time_seconds: f64,

    /// Per-level retry success histogram (12 levels, indexed 0..11).
    /// `retry_level_histogram[k]` counts how many solves were recovered at
    /// retry level `k`. The sum equals `success_count - first_try_successes`.
    pub retry_level_histogram: Vec<u64>,
}
```

### 4.4 StageTemplate

The `StageTemplate` holds the pre-assembled structural LP for one stage in solver-ready CSC form. It is built once at initialization and shared read-only across all threads. The full specification of its contents, lifecycle, and memory layout is in [Solver Abstraction SS11.1](./solver-abstraction.md). This spec defines only the type signature for use in the `SolverInterface` trait.

**Construction ownership**: The `cobre-sddp` crate owns `StageTemplate` construction. A builder function in `cobre-sddp` takes a reference to the resolved `System` struct ([Internal Structures SS1](../data-model/internal-structures.md)) and the `StageDefinition` for the target stage, and produces a `StageTemplate`. The solver crate (`cobre-solver`) receives `StageTemplate` as an opaque data holder and does not interpret its contents -- it bulk-loads the CSC arrays into the underlying LP solver without understanding what the columns or rows represent. This separation ensures that LP modeling concerns (variable ordering, constraint structure, state dimension) remain in `cobre-sddp`, while solver concerns (API calls, retry logic, basis management) remain in `cobre-solver`.

```rust
// Construction function signature (in cobre-sddp, not cobre-solver):
//
// pub fn build_stage_template(
//     system: &System,
//     stage_def: &StageDefinition,
//     stage_index: usize,
// ) -> StageTemplate;
```

```rust
/// Pre-assembled structural LP for one stage, in CSC (column-major) form.
///
/// Built once at initialization from resolved internal structures
/// ([Internal Structures](../data-model/internal-structures.md)).
/// Shared read-only across all threads within an MPI rank.
/// Column and row ordering follows [Solver Abstraction SS2](./solver-abstraction.md).
pub struct StageTemplate {
    /// Number of columns (variables).
    pub num_cols: usize,
    /// Number of static rows (constraints, excluding dynamic constraints).
    pub num_rows: usize,
    /// Number of non-zero entries in the structural matrix.
    pub num_nz: usize,

    /// CSC column start offsets (`i32` for HiGHS FFI compatibility).
    /// Length: `num_cols + 1`; `col_starts[num_cols] == num_nz`.
    pub col_starts: Vec<i32>,
    /// CSC row indices (`i32` for HiGHS FFI compatibility). Length: `num_nz`.
    pub row_indices: Vec<i32>,
    /// CSC non-zero values. Length: `num_nz`.
    pub values: Vec<f64>,

    /// Column lower bounds. Length: `num_cols`.
    pub col_lower: Vec<f64>,
    /// Column upper bounds. Length: `num_cols`.
    pub col_upper: Vec<f64>,
    /// Objective coefficients. Length: `num_cols`.
    pub objective: Vec<f64>,

    /// Row lower bounds. Length: `num_rows`.
    pub row_lower: Vec<f64>,
    /// Row upper bounds. Length: `num_rows`.
    pub row_upper: Vec<f64>,

    /// Number of state variables (contiguous prefix of columns).
    /// Equal to N * (1 + L) per [Solver Abstraction SS2.1](./solver-abstraction.md).
    pub n_state: usize,
    /// Number of state values to transfer between stages.
    /// Equal to N * L per [Solver Abstraction SS2.1](./solver-abstraction.md)
    /// (storage + all lags except the oldest).
    pub n_transfer: usize,
    /// Number of dual-relevant constraint rows (contiguous prefix of rows).
    /// Currently equal to `n_state` (= `N + N*L` where `N` is the number
    /// of hydros and `L` is the maximum PAR lag order). FPHA and generic
    /// variable constraint rows are structural and not included in the
    /// dual-relevant set. Cut coefficients are extracted from
    /// `dual[0..n_dual_relevant]`.
    pub n_dual_relevant: usize,
    /// Number of operating hydros at this stage.
    pub n_hydro: usize,
    /// Maximum PAR order across all operating hydros at this stage.
    /// Determines the uniform lag stride: all hydros store `max_par_order`
    /// lag values regardless of their individual PAR order, enabling SIMD
    /// vectorization with a single contiguous state stride.
    pub max_par_order: usize,

    /// Per-column scaling factors for numerical conditioning.
    /// When non-empty (length `num_cols`), the constraint matrix, objective
    /// coefficients, and column bounds have been pre-scaled by these factors.
    /// The calling algorithm is responsible for unscaling primal values after
    /// each solve: `x_original[j] = col_scale[j] * x_scaled[j]`.
    /// When empty, no column scaling has been applied and solver results are
    /// used directly.
    pub col_scale: Vec<f64>,
    /// Per-row scaling factors for numerical conditioning.
    /// When non-empty (length `num_rows`), the constraint matrix and row
    /// bounds have been pre-scaled by these factors. The calling algorithm
    /// is responsible for unscaling dual values after each solve:
    /// `dual_original[i] = row_scale[i] * dual_scaled[i]`.
    /// When empty, no row scaling has been applied and solver results are
    /// used directly.
    pub row_scale: Vec<f64>,
}
```

### 4.5 RowBatch

The `RowBatch` holds constraint rows for batch addition in CSR (row-major) form, ready for a single `add_rows` call. In SDDP, it is assembled from the cut pool's activity bitmap before each LP rebuild.

```rust
/// Batch of constraint rows for addition, in CSR (row-major) form.
///
/// In SDDP, assembled from the cut pool activity bitmap for the current stage.
/// Passed to `add_rows` for a single batch row-addition call.
/// See [Solver Abstraction SS5.4](./solver-abstraction.md) for the
/// assembly protocol.
pub struct RowBatch {
    /// Number of active cuts in this batch.
    pub num_rows: usize,

    /// CSR row start offsets (`i32` for HiGHS FFI compatibility).
    /// Length: `num_rows + 1`. `row_starts[num_rows]` equals the total
    /// number of non-zeros.
    pub row_starts: Vec<i32>,
    /// CSR column indices (`i32` for HiGHS FFI compatibility).
    /// Length: total non-zeros across all cuts.
    pub col_indices: Vec<i32>,
    /// CSR non-zero values. Length: total non-zeros across all cuts.
    pub values: Vec<f64>,

    /// Row lower bounds (cut intercepts alpha). Length: `num_rows`.
    pub row_lower: Vec<f64>,
    /// Row upper bounds (all +infinity). Length: `num_rows`.
    pub row_upper: Vec<f64>,
}
```

## 5. Dispatch Mechanism

> **Decision [DEC-006](../overview/decision-log.md#dec-006) (active):** `Box<dyn Trait>` rejected for all closed variant sets; enum dispatch used for algorithm variants (`RiskMeasure`, `HorizonMode`, `SamplingScheme`, `CutSelectionStrategy`, `StoppingRuleSet`); compile-time monomorphization reserved for FFI-wrapping traits.

The `SolverInterface` trait uses **compile-time monomorphization** -- the training loop is generic over the solver type, and the concrete implementation is resolved at compile time. This is the same pattern used by the `Communicator` trait ([Communicator Trait §3](../hpc/communicator-trait.md)) and documented as the solver selection strategy in [Solver Abstraction SS10](./solver-abstraction.md).

```rust
/// Train the SDDP policy using the provided solver and communicator backends.
///
/// Both generic parameters are resolved at compile time -- no trait object
/// indirection, no vtable lookup, no dynamic dispatch on the hot path.
pub fn train<S: SolverInterface, C: Communicator>(
    solver_factory: impl Fn() -> S,
    comm: &C,
    config: &TrainingConfig,
    stages: &[StageTemplate],
    // ... other parameters
) -> TrainingResult {
    // Each OpenMP thread creates its own solver instance via solver_factory
    // Thread-local workspace holds the solver for the entire training run
    todo!()
}
```

**Why compile-time monomorphization (not enum dispatch):** The solver interface is fundamentally different from the five algorithm-variant abstractions (risk measure, cut formulation, horizon mode, sampling scheme, cut selection strategy) that use enum dispatch:

| Aspect                      | SolverInterface                    | Algorithm-Variant Traits                              |
| --------------------------- | ---------------------------------- | ----------------------------------------------------- |
| **Variation scope**         | Global -- one solver per build     | May vary per stage (risk measure) or per run (others) |
| **Variant count**           | One per binary (feature-gated)     | 2-3 variants coexist in the same binary               |
| **Call frequency**          | Millions of times (every LP solve) | Hundreds to thousands per iteration                   |
| **FFI boundary**            | Wraps C library calls              | Pure Rust computation                                 |
| **Performance sensitivity** | Extremely high -- on the hot path  | Low to moderate -- dominated by LP solve cost         |
| **Dispatch pattern**        | Compile-time monomorphization      | Enum match at call site                               |

The key distinction is that exactly one solver backend is active per build (selected via Cargo feature flags per [Solver Abstraction SS10](./solver-abstraction.md)). The binary never needs to dispatch between HiGHS and CLP at runtime. Monomorphization eliminates virtual dispatch overhead entirely, enables the compiler to inline solver-specific code paths, and is consistent with the established `Communicator` trait pattern.

**Contrast with enum dispatch:** The algorithm-variant traits (e.g., `RiskMeasure`, `HorizonMode`) use enum dispatch because they have a small, fixed variant set where multiple variants may coexist in the same binary (e.g., per-stage risk measure variation). The solver interface has neither characteristic: there is exactly one active solver per binary, and the call frequency is orders of magnitude higher.

## 6. Retry Logic Encapsulation

Retry logic for numerical difficulties is **encapsulated within each `SolverInterface` implementation**. The `solve()` and `solve_with_basis()` methods handle retries internally and return only the final result (success or terminal error) to the caller. The calling algorithm never sees intermediate retry attempts. The `retry_max_attempts` and `retry_time_budget_seconds` parameters are sourced from `config.json` (see [Configuration Reference section 3.5](../configuration/configuration-reference.md)).

The retry behavioral contract is defined in [Solver Abstraction SS7](./solver-abstraction.md):

- **Maximum attempts**: Configurable upper bound (default: 5)
- **Time budget**: Configurable wall-clock budget for all attempts combined
- **Escalating strategies**: Least-disruptive (clear basis) to most-disruptive (switch algorithm)
- **Final disposition**: Terminal `SolverError` with best partial solution if available
- **Logging**: Each retry attempt logged at debug level

For solver-specific retry escalation sequences, see [HiGHS Implementation](./solver-highs-impl.md) and [CLP Implementation](./solver-clp-impl.md). The implementation specs define which HiGHS/CLP API calls correspond to each escalation step (clear basis, disable presolve, switch to interior point, relax tolerances).

## 7. Dual Normalization Contract

All dual multipliers in `SolutionView.dual` (and consequently `LpSolution.dual` via `to_owned()`) are **pre-normalized** to the canonical sign convention defined in [Solver Abstraction SS8](./solver-abstraction.md) before the solution is returned to the caller. Solver-specific sign differences are resolved within the `SolverInterface` implementation.

**Canonical convention:** A positive dual on a $\leq$ constraint means that increasing the RHS increases the objective ($\partial z^* / \partial b > 0$).

**Why this matters:** The cut coefficient computation in the backward pass extracts dual multipliers from the cut-relevant constraint rows (`[0, n_dual_relevant)`) and uses them directly as cut gradients:

$$\beta_t^k = W_t^\top \pi_t^*$$

A sign error in $\pi_t^*$ produces cuts that point in the wrong direction, leading to divergence of the outer approximation. By normalizing duals inside the solver implementation, the cut generation logic is solver-agnostic and provably correct regardless of which backend is active.

**Implementation responsibility:** Each solver backend must know its native dual sign convention and apply the appropriate transformation. For example, if a solver reports duals with the opposite sign for $\geq$ constraints, the implementation negates those duals before populating `SolutionView.dual`. This transformation is applied once per solve, adding negligible overhead to the solution extraction step.

## Cross-References

- [Solver Abstraction](./solver-abstraction.md) -- Operations contract (SS4), LP layout convention (SS2), cut pool design (SS5), error categories (SS6), retry logic contract (SS7), dual normalization (SS8), basis storage (SS9), compile-time selection (SS10), stage templates and rebuild strategy (SS11)
- [Solver Workspaces](./solver-workspaces.md) -- Thread-local solver infrastructure (SS1), per-stage basis cache (SS1.5), pre-allocated solution buffers (SS1.2), solve statistics (SS1.6)
- [HiGHS Implementation](./solver-highs-impl.md) -- HiGHS-specific API mapping (SS2), retry strategy, batch operations, memory footprint
- [CLP Implementation](./solver-clp-impl.md) -- CLP-specific API mapping (SS2), C++ wrapper strategy, mutable pointer optimization, cloning path
- [Communicator Trait](../hpc/communicator-trait.md) -- Compile-time monomorphization pattern (SS3) that the solver interface follows; convention blockquote source
- [Backend Testing](../hpc/backend-testing.md) -- Conformance test methodology applicable to solver backend testing
- [Extension Points](./extension-points.md) -- Dispatch mechanism analysis (SS7) contrasting compile-time monomorphization (solver) with enum dispatch (algorithm variants); variant selection pipeline (SS6)
- [Training Loop](./training-loop.md) -- Forward pass (SS4) and backward pass (SS6) that drive solver invocations; abstraction points (SS3) parameterizing the training loop
- [Cut Management Implementation](./cut-management-impl.md) -- Cut pool activity bitmap (SS1.1), CSR assembly for `addRows` (SS1), cut coefficient computation using duals from `LpSolution`
- [LP Formulation](../math/lp-formulation.md) -- Constraint structure that defines which duals are cut-relevant and feed into the cut coefficient formula
- [Binary Formats](../data-model/binary-formats.md) -- Cut pool memory layout (SS3.4) that produces `RowBatch` inputs
- [Internal Structures](../data-model/internal-structures.md) -- Logical in-memory data model from which `StageTemplate` is built
- [Hybrid Parallelism](../hpc/hybrid-parallelism.md) -- OpenMP threading model (SS3) requiring thread-local solvers with `Send` but not `Sync`
- [Memory Architecture](../hpc/memory-architecture.md) -- NUMA-aware allocation (SS2) for solver workspaces
