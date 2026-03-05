//! The [`SolverInterface`] trait definition.
//!
//! This module defines the central abstraction through which optimization
//! algorithms interact with LP solvers.

use crate::types::{Basis, LpSolution, RowBatch, SolverError, SolverStatistics, StageTemplate};

/// Backend-agnostic interface for LP solver instances.
///
/// # Design
///
/// The trait is resolved as a **generic type parameter at compile time**
/// (DEC-002), not as `dyn SolverInterface`. This monomorphization approach
/// eliminates virtual dispatch overhead on the hot path, where tens of millions
/// of LP solves occur during a single training run. The training loop is
/// parameterized as `fn train<S: SolverInterface>(solver_factory: impl Fn() -> S, ...)`.
///
/// # Thread Safety
///
/// The trait requires `Send` but not `Sync`. `Send` allows solver instances to
/// be transferred to worker threads during thread pool initialization. The
/// absence of `Sync` prevents concurrent access, which matches the reality of
/// C-library solver handles (`HiGHS`, CLP): they maintain mutable internal state
/// (factorization workspace, working arrays) that is not thread-safe. Each
/// worker thread owns exactly one solver instance for the duration of the
/// training run, following the thread-local workspace pattern described in
/// Solver Workspaces SS1.1.
///
/// # Mutability Convention
///
/// - Mutating methods (`load_model`, `add_rows`, `set_row_bounds`,
///   `set_col_bounds`, `solve`, `solve_with_basis`, `reset`) take `&mut self`.
/// - Read-only query methods (`get_basis`, `statistics`, `name`) take `&self`.
///
/// # Error Recovery Contract
///
/// When `solve` or `solve_with_basis` returns `Err`, the solver's internal
/// state is unspecified. The **caller** is responsible for calling `reset()`
/// before reusing the instance for another solve sequence. Failing to call
/// `reset()` after an error may produce incorrect results or panics.
///
/// # Usage as a Generic Bound
///
/// ```rust
/// use cobre_solver::{SolverInterface, LpSolution, SolverError};
///
/// fn run_solve<S: SolverInterface>(solver: &mut S) -> Result<LpSolution, SolverError> {
///     solver.solve()
/// }
/// ```
///
/// See [Solver Interface Trait SS1](../../../cobre-docs/src/specs/architecture/solver-interface-trait.md)
/// and [Solver Interface Trait SS5](../../../cobre-docs/src/specs/architecture/solver-interface-trait.md)
/// for the dispatch mechanism rationale.
pub trait SolverInterface: Send {
    /// Bulk-loads a pre-assembled structural LP into the solver instance.
    ///
    /// This is the first step of the LP rebuild sequence at each stage
    /// transition. The stage template is built once at initialization and
    /// shared read-only across all threads within an MPI rank.
    ///
    /// **Preconditions:**
    /// - `template` contains a valid CSC matrix: column starts, row indices,
    ///   and values arrays are consistent and contain no out-of-bounds indices.
    /// - `template` follows the LP layout convention (column and row ordering
    ///   per Solver Abstraction SS2).
    /// - `template.num_cols > 0` and `template.num_rows > 0`.
    ///
    /// **Postconditions:**
    /// - The solver holds the structural LP from `template`; any previous model
    ///   is fully replaced.
    /// - No cuts are present: the loaded model contains only structural
    ///   constraints, which are added separately via [`add_rows`](Self::add_rows).
    /// - Any cached basis from a previous model is invalidated.
    ///
    /// **Infallibility:** This method does not return `Result`. The stage
    /// template is validated during initialization; passing an invalid template
    /// is a programming error (panic on violation).
    ///
    /// See [Solver Interface Trait SS2.1](../../../cobre-docs/src/specs/architecture/solver-interface-trait.md).
    fn load_model(&mut self, template: &StageTemplate);

    /// Appends constraint rows to the dynamic constraint region in a single batch call.
    ///
    /// Used to add dynamically generated constraint rows (e.g., Benders cuts)
    /// assembled from a cut pool. This is step 2 of the LP rebuild sequence.
    ///
    /// **Preconditions:**
    /// - [`load_model`](Self::load_model) has been called; a structural LP must
    ///   be loaded before adding cuts.
    /// - `cuts` contains valid CSR row data: row starts, column indices, values,
    ///   and bounds arrays are consistent.
    /// - Cut column indices reference valid columns in the loaded model
    ///   (indices within `[0, num_cols)`).
    ///
    /// **Postconditions:**
    /// - Active cuts are appended as rows at `[n_static, n_static + cuts.num_rows)`,
    ///   following the dynamic constraint region per Solver Abstraction SS2.2.
    /// - Structural rows `[0, n_static)` are unchanged.
    /// - The solver basis is not automatically set; the caller must use
    ///   [`solve_with_basis`](Self::solve_with_basis) to apply a cached basis.
    ///
    /// **Infallibility:** This method does not return `Result`. The cut batch
    /// is assembled from the pre-validated cut pool; invalid CSR data is a
    /// programming error (panic on violation).
    ///
    /// See [Solver Interface Trait SS2.2](../../../cobre-docs/src/specs/architecture/solver-interface-trait.md).
    fn add_rows(&mut self, cuts: &RowBatch);

    /// Updates row bounds (constraint RHS values) without structural LP changes.
    ///
    /// `indices` contains the row indices to update. `lower` and `upper` contain
    /// the new lower and upper bounds for each index, respectively. All three
    /// slices must have the same length. This is step 3 of the LP rebuild
    /// sequence and the primary modification between successive solves at the
    /// same stage. For equality constraints (water balance, lag fixing, noise
    /// fixing), set the corresponding entries in `lower` and `upper` equal.
    ///
    /// **Preconditions:**
    /// - [`load_model`](Self::load_model) has been called.
    /// - All row indices in `indices` reference valid rows in the loaded model.
    /// - All bound values are finite (no NaN or infinity).
    /// - `lower[i] <= upper[i]` for each entry.
    /// - `indices`, `lower`, and `upper` have equal length.
    ///
    /// **Postconditions:**
    /// - Row lower and upper bounds at each index are updated; the LP
    ///   reflects the current scenario realization.
    /// - Non-patched rows are unchanged.
    /// - Column bounds are unchanged.
    /// - The solver basis is preserved; patching does not invalidate a
    ///   previously set basis.
    ///
    /// **Infallibility:** This method does not return `Result`. Patch indices
    /// are computed from the LP layout convention; out-of-bounds indices are a
    /// programming error (panic on violation).
    ///
    /// # Panics
    ///
    /// Panics if `indices`, `lower`, and `upper` do not have equal length.
    ///
    /// See [Solver Interface Trait SS2.3](../../../cobre-docs/src/specs/architecture/solver-interface-trait.md).
    fn set_row_bounds(&mut self, indices: &[usize], lower: &[f64], upper: &[f64]);

    /// Updates column bounds (variable lower/upper bounds) without structural LP changes.
    ///
    /// `indices` contains the column indices to update. `lower` and `upper`
    /// contain the new lower and upper bounds for each index, respectively. All
    /// three slices must have the same length. Useful for per-scenario variable
    /// bound updates such as thermal unit commitment bounds or battery
    /// state-of-charge limits.
    ///
    /// **Preconditions:**
    /// - [`load_model`](Self::load_model) has been called.
    /// - All column indices in `indices` reference valid columns in the loaded model.
    /// - All bound values are finite (no NaN or infinity).
    /// - `lower[i] <= upper[i]` for each entry.
    /// - `indices`, `lower`, and `upper` have equal length.
    ///
    /// **Postconditions:**
    /// - Column lower and upper bounds at each index are updated.
    /// - Non-patched columns are unchanged.
    /// - Row bounds are unchanged.
    /// - The solver basis is preserved; patching does not invalidate a
    ///   previously set basis.
    ///
    /// **Infallibility:** This method does not return `Result`. Patch indices
    /// are computed from the LP layout convention; out-of-bounds indices are a
    /// programming error (panic on violation).
    ///
    /// # Panics
    ///
    /// Panics if `indices`, `lower`, and `upper` do not have equal length.
    ///
    /// See [Solver Interface Trait SS2.3a](../../../cobre-docs/src/specs/architecture/solver-interface-trait.md).
    fn set_col_bounds(&mut self, indices: &[usize], lower: &[f64], upper: &[f64]);

    /// Invokes the LP solver and returns either a valid solution or a terminal error.
    ///
    /// This is the primary hot-path method, called millions of times during a
    /// training run. It encapsulates internal retry logic (e.g., clearing basis,
    /// disabling presolve, switching algorithm) before returning a terminal
    /// [`SolverError`]. The caller never sees intermediate retry failures.
    ///
    /// **Preconditions:**
    /// - [`load_model`](Self::load_model) has been called.
    /// - Cuts and scenario patches have been applied as needed.
    ///
    /// **Postconditions (on `Ok`):**
    /// - `LpSolution.objective` is the optimal objective value (minimization sense).
    /// - `LpSolution.primal` contains optimal primal values; length equals `num_cols`.
    /// - `LpSolution.dual` contains normalized dual values (sign convention per
    ///   Solver Abstraction SS8); length equals `num_rows` (structural + cuts).
    /// - The solver basis reflects the optimal solution; available via
    ///   [`get_basis`](Self::get_basis) after this call.
    /// - [`SolverStatistics`] counters are incremented (solve count, iterations, timing).
    ///
    /// **Postconditions (on `Err`):**
    /// - The [`SolverError`] variant identifies the terminal failure after all
    ///   retry attempts are exhausted.
    /// - Solver state is unspecified; the caller **must** call
    ///   [`reset`](Self::reset) before reusing this instance.
    /// - `SolverStatistics.retry_count` reflects retry attempts made.
    ///
    /// **Fallibility:** Returns `Result<LpSolution, SolverError>` because LP
    /// solves wrap FFI calls that may encounter numerical difficulties,
    /// infeasibility, or other solver-internal failures not preventable by
    /// precondition checks.
    ///
    /// # Errors
    ///
    /// Returns `Err(SolverError)` when all internal retry attempts are exhausted.
    /// Possible variants: [`SolverError::Infeasible`], [`SolverError::Unbounded`],
    /// [`SolverError::NumericalDifficulty`], [`SolverError::TimeLimitExceeded`],
    /// [`SolverError::IterationLimit`], or [`SolverError::InternalError`].
    /// After an `Err`, the caller must call [`reset`](Self::reset) before reusing
    /// this instance.
    ///
    /// See [Solver Interface Trait SS2.4](../../../cobre-docs/src/specs/architecture/solver-interface-trait.md).
    fn solve(&mut self) -> Result<LpSolution, SolverError>;

    /// Sets a cached basis for warm-starting, then solves the LP.
    ///
    /// Combines the set-basis and solve operations atomically to ensure the
    /// basis is applied before solving. Warm-starting from a cached basis
    /// typically reduces simplex iterations by 80-95% compared to cold starts.
    ///
    /// If the provided basis dimensions do not match the current LP (e.g.,
    /// because cuts were added since the basis was saved), the implementation
    /// handles this gracefully: the static portion of the basis is position-stable
    /// and reused directly; new dynamic constraint rows are initialized as
    /// [`BasisStatus::Basic`](crate::BasisStatus::Basic) per Solver Abstraction SS2.3.
    ///
    /// **Preconditions:**
    /// - [`load_model`](Self::load_model) has been called.
    /// - `basis.col_status.len()` matches the loaded model's column count.
    /// - `basis.row_status.len()` matches the loaded model's row count
    ///   (structural + cuts).
    ///
    /// **Postconditions (on `Ok`):**
    /// - Same as [`solve`](Self::solve) `Ok` postconditions; valid solution with
    ///   normalized duals.
    /// - Simplex iterations are typically reduced compared to a cold start.
    ///
    /// **Postconditions (on `Err`):**
    /// - Same as [`solve`](Self::solve) `Err` postconditions; terminal error after
    ///   retry exhaustion.
    /// - The implementation may fall back to a cold start during retry; basis
    ///   rejection is a valid retry escalation step.
    ///
    /// **Fallibility:** Returns `Result<LpSolution, SolverError>`, same as
    /// [`solve`](Self::solve).
    ///
    /// # Errors
    ///
    /// Returns `Err(SolverError)` for the same reasons as [`solve`](Self::solve):
    /// all internal retry attempts exhausted. The implementation may fall back to
    /// a cold start during retry; basis rejection is a valid escalation step. After
    /// an `Err`, the caller must call [`reset`](Self::reset) before reusing this
    /// instance.
    ///
    /// See [Solver Interface Trait SS2.5](../../../cobre-docs/src/specs/architecture/solver-interface-trait.md).
    fn solve_with_basis(&mut self, basis: &Basis) -> Result<LpSolution, SolverError>;

    /// Clears all internal solver state, returning the instance to a clean state.
    ///
    /// Used for error recovery after a terminal [`SolverError`], or when
    /// switching between fundamentally different LP structures. After `reset`,
    /// [`load_model`](Self::load_model) must be called before the next solve.
    ///
    /// **Preconditions:** None. `reset` can be called at any time.
    ///
    /// **Postconditions:**
    /// - Solver state is clean: no loaded model, no cached basis, no factorization.
    /// - [`load_model`](Self::load_model) must be called before the next solve.
    /// - Accumulated [`SolverStatistics`] counters are **preserved**; `reset` does
    ///   not zero them. Statistics accumulate for the lifetime of the instance.
    ///
    /// **Infallibility:** This method does not return `Result`. Clearing solver
    /// state is a local operation with no failure modes.
    ///
    /// See [Solver Interface Trait SS2.6](../../../cobre-docs/src/specs/architecture/solver-interface-trait.md).
    fn reset(&mut self);

    /// Extracts the current simplex basis from the solver.
    ///
    /// The basis is stored in the original problem space (not presolved) to
    /// ensure portability across solver versions and presolve strategies. Use
    /// the returned [`Basis`] with [`solve_with_basis`](Self::solve_with_basis)
    /// to warm-start subsequent solves.
    ///
    /// **Preconditions:**
    /// - A successful [`solve`](Self::solve) or
    ///   [`solve_with_basis`](Self::solve_with_basis) has completed. A basis
    ///   exists only after a successful solve. Calling `get_basis` without a
    ///   prior successful solve is a programming error (panic on violation).
    ///
    /// **Postconditions:**
    /// - `Basis.col_status.len() == num_cols`: one status per variable.
    /// - `Basis.row_status.len() == num_rows`: one status per constraint
    ///   (structural + appended cuts).
    /// - Status values are in the canonical set: `AtLower`, `Basic`, `AtUpper`,
    ///   `Free`, or `Fixed` per Solver Abstraction SS9.
    ///
    /// **Infallibility:** This method does not return `Result`. After a
    /// successful solve, the basis always exists and can be extracted.
    ///
    /// See [Solver Interface Trait SS2.7](../../../cobre-docs/src/specs/architecture/solver-interface-trait.md).
    fn get_basis(&self) -> Basis;

    /// Returns accumulated solve metrics for this solver instance.
    ///
    /// Returns a snapshot of the monotonically increasing counters. Statistics
    /// accumulate across all solves since construction; [`reset`](Self::reset)
    /// does not zero them.
    ///
    /// **Preconditions:** None. Can be called at any time, including before any
    /// solves have been performed.
    ///
    /// **Postconditions:**
    /// - All fields are non-negative; counters start at zero and only increment.
    /// - `solve_count >= success_count + failure_count`.
    /// - `retry_count` counts individual retry escalation steps across all
    ///   failed solves.
    ///
    /// **Infallibility:** This method does not return `Result`. It reads internal
    /// counters with no failure modes.
    ///
    /// See [Solver Interface Trait SS2.8](../../../cobre-docs/src/specs/architecture/solver-interface-trait.md).
    fn statistics(&self) -> SolverStatistics;

    /// Returns a static string identifying the solver backend.
    ///
    /// Used for logging, diagnostics, and checkpoint metadata. Example values:
    /// `"HiGHS"`, `"CLP"`.
    ///
    /// **Preconditions:** None.
    ///
    /// **Postconditions:**
    /// - The returned string is non-empty.
    /// - The string uniquely identifies the backend within the Cobre ecosystem.
    ///
    /// **Infallibility:** This method does not return `Result`. It returns a
    /// compile-time constant with no failure modes.
    ///
    /// See [Solver Interface Trait SS2.9](../../../cobre-docs/src/specs/architecture/solver-interface-trait.md).
    fn name(&self) -> &'static str;
}

#[cfg(test)]
mod tests {
    use super::SolverInterface;

    // Verify trait is usable as a generic bound (compile-time monomorphization
    // per DEC-002, not dyn dispatch).
    fn accepts_solver<S: SolverInterface>(_: &S) {}

    // Test double implementing SolverInterface for compile-time verification.
    struct NoopSolver;

    impl SolverInterface for NoopSolver {
        fn load_model(&mut self, _template: &crate::types::StageTemplate) {}

        fn add_rows(&mut self, _cuts: &crate::types::RowBatch) {}

        fn set_row_bounds(&mut self, _indices: &[usize], _lower: &[f64], _upper: &[f64]) {}

        fn set_col_bounds(&mut self, _indices: &[usize], _lower: &[f64], _upper: &[f64]) {}

        fn solve(&mut self) -> Result<crate::types::LpSolution, crate::types::SolverError> {
            Err(crate::types::SolverError::InternalError {
                message: "noop".to_string(),
                error_code: None,
            })
        }

        fn solve_with_basis(
            &mut self,
            _basis: &crate::types::Basis,
        ) -> Result<crate::types::LpSolution, crate::types::SolverError> {
            Err(crate::types::SolverError::InternalError {
                message: "noop".to_string(),
                error_code: None,
            })
        }

        fn reset(&mut self) {}

        fn get_basis(&self) -> crate::types::Basis {
            crate::types::Basis {
                col_status: vec![],
                row_status: vec![],
            }
        }

        fn statistics(&self) -> crate::types::SolverStatistics {
            crate::types::SolverStatistics::default()
        }

        fn name(&self) -> &'static str {
            "Noop"
        }
    }

    // Verify Send bound is satisfied.
    fn assert_send<T: Send>() {}

    #[test]
    fn test_trait_compiles_as_generic_bound() {
        accepts_solver(&NoopSolver);
    }

    #[test]
    fn test_solver_interface_send_bound() {
        assert_send::<NoopSolver>();
    }

    #[test]
    fn test_noop_solver_name() {
        let name = NoopSolver.name();
        assert_eq!(name, "Noop");
        assert!(!name.is_empty());
    }

    #[test]
    fn test_noop_solver_statistics_initial() {
        let stats = NoopSolver.statistics();
        assert_eq!(stats.solve_count, 0);
        assert_eq!(stats.success_count, 0);
        assert_eq!(stats.failure_count, 0);
        assert_eq!(stats.total_iterations, 0);
        assert_eq!(stats.retry_count, 0);
        assert_eq!(stats.total_solve_time_seconds, 0.0);
    }

    #[test]
    fn test_noop_solver_get_basis_empty() {
        let basis = NoopSolver.get_basis();
        assert!(basis.col_status.is_empty());
        assert!(basis.row_status.is_empty());
    }
}
