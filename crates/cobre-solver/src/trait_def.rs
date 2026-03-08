//! The [`SolverInterface`] trait definition.
//!
//! This module defines the central abstraction through which optimization
//! algorithms interact with LP solvers.

use crate::types::{
    Basis, LpSolution, RawBasis, RowBatch, SolutionView, SolverError, SolverStatistics,
    StageTemplate,
};

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
/// - Methods that write to internal scratch buffers (`get_basis`) take `&mut self`.
/// - Read-only query methods (`statistics`, `name`) take `&self`.
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
    /// Bulk-loads a pre-assembled structural LP (first step of rebuild sequence).
    ///
    /// Replaces any previous model. Validates template is a valid CSC matrix
    /// with `num_cols > 0` and `num_rows > 0` (panic on violation).
    ///
    /// See Solver Interface Trait SS2.1.
    fn load_model(&mut self, template: &StageTemplate);

    /// Appends constraint rows to the dynamic constraint region (step 2 of rebuild).
    ///
    /// Requires [`load_model`](Self::load_model) called first and `cuts` to have
    /// valid CSR data with column indices in `[0, num_cols)` (panic on violation).
    ///
    /// See Solver Interface Trait SS2.2.
    fn add_rows(&mut self, cuts: &RowBatch);

    /// Updates row bounds (step 3 of rebuild; patching for scenario realization).
    ///
    /// `indices`, `lower`, and `upper` must have equal length, with all indices
    /// referencing valid rows and bounds finite. For equality constraints, set
    /// `lower[i] == upper[i]`. Panics if lengths differ or indices are out-of-bounds.
    ///
    /// See Solver Interface Trait SS2.3.
    fn set_row_bounds(&mut self, indices: &[usize], lower: &[f64], upper: &[f64]);

    /// Updates column bounds (per-scenario variable bound patching).
    ///
    /// `indices`, `lower`, and `upper` must have equal length, with all indices
    /// referencing valid columns and bounds finite. Panics if lengths differ or
    /// indices are out-of-bounds.
    ///
    /// See Solver Interface Trait SS2.3a.
    fn set_col_bounds(&mut self, indices: &[usize], lower: &[f64], upper: &[f64]);

    /// Solves the LP, returning solution or terminal error after retry exhaustion.
    ///
    /// Hot-path method encapsulating internal retry logic. Requires [`load_model`]
    /// called first and scenario patches applied. On error, caller must call
    /// [`reset`](Self::reset) before reusing.
    ///
    /// # Errors
    ///
    /// Returns `Err(SolverError)` when all internal retry attempts exhausted.
    /// Possible variants: [`SolverError::Infeasible`], [`SolverError::Unbounded`],
    /// [`SolverError::NumericalDifficulty`], [`SolverError::TimeLimitExceeded`],
    /// [`SolverError::IterationLimit`], or [`SolverError::InternalError`].
    ///
    /// See Solver Interface Trait SS2.4.
    fn solve(&mut self) -> Result<LpSolution, SolverError> {
        self.solve_view().map(|v| v.to_owned())
    }

    /// Zero-copy variant of [`solve`]: returns [`SolutionView`] borrowing buffers.
    ///
    /// Same preconditions, postconditions, and errors as [`solve`]. Valid until
    /// the next `&mut self` call. Use on hot path to avoid cloning; call
    /// [`SolutionView::to_owned`] when solution must outlive the borrow.
    ///
    /// # Errors
    ///
    /// Same as [`solve`].
    ///
    /// See Solver Interface Trait SS2.4.
    fn solve_view(&mut self) -> Result<SolutionView<'_>, SolverError>;

    /// Warm-starts from a basis, then solves. Basis dimensions must match LP.
    ///
    /// New dynamic rows (from cuts) are initialized as [`BasisStatus::Basic`].
    /// Typically reduces iterations 80-95% vs cold start. May fall back to cold
    /// start during retry if basis rejected. Same errors as [`solve`]; requires
    /// [`reset`] after error.
    ///
    /// # Errors
    ///
    /// Same as [`solve`].
    ///
    /// See Solver Interface Trait SS2.5.
    fn solve_with_basis(&mut self, basis: &Basis) -> Result<LpSolution, SolverError> {
        self.solve_with_basis_view(basis).map(|v| v.to_owned())
    }

    /// Zero-copy variant of [`solve_with_basis`]: returns [`SolutionView`] borrowing buffers.
    ///
    /// Same preconditions, postconditions, and errors as [`solve_with_basis`].
    /// Valid until the next `&mut self` call. Use on hot path to avoid cloning.
    ///
    /// # Errors
    ///
    /// Same as [`solve_with_basis`].
    ///
    /// See Solver Interface Trait SS2.5.
    fn solve_with_basis_view(&mut self, basis: &Basis) -> Result<SolutionView<'_>, SolverError>;

    /// Clears internal solver state for error recovery or LP structure change.
    ///
    /// Requires [`load_model`] before next solve. Preserves `SolverStatistics`
    /// counters; does not zero them.
    ///
    /// See Solver Interface Trait SS2.6.
    fn reset(&mut self);

    /// Extracts the current simplex basis after a successful solve (panic if none).
    ///
    /// Basis is in original (unscaled) space for portability. Use with
    /// [`solve_with_basis`] to warm-start subsequent solves.
    ///
    /// See Solver Interface Trait SS2.7.
    fn get_basis(&mut self) -> Basis;

    /// Writes solver-native `i32` status codes into a caller-owned [`RawBasis`] buffer.
    ///
    /// This is the zero-copy counterpart of [`get_basis`](Self::get_basis). The
    /// caller pre-allocates a [`RawBasis`] with [`RawBasis::new`] and reuses it
    /// across iterations, eliminating per-element [`BasisStatus`] enum translation.
    ///
    /// The buffer is not resized by this method. The implementation writes into
    /// the first `num_cols` entries of `out.col_status` and the first `num_rows`
    /// entries of `out.row_status`. Panics if no model is loaded (same as
    /// [`get_basis`](Self::get_basis)).
    ///
    /// See Solver Interface Trait SS2.7.
    fn get_raw_basis(&mut self, out: &mut RawBasis);

    /// Injects a raw basis and solves, returning a zero-copy [`SolutionView`].
    ///
    /// The zero-copy counterpart of [`solve_with_basis_view`](Self::solve_with_basis_view).
    /// Status codes in `basis` are injected directly without per-element enum
    /// translation. On success the returned view borrows solver-internal buffers
    /// and is valid until the next `&mut self` call.
    ///
    /// # Errors
    ///
    /// Same error contract as [`solve_with_basis_view`](Self::solve_with_basis_view).
    ///
    /// See Solver Interface Trait SS2.5.
    fn solve_with_raw_basis_view(
        &mut self,
        basis: &RawBasis,
    ) -> Result<SolutionView<'_>, SolverError>;

    /// Warm-starts from a raw basis, then solves, returning an owned [`LpSolution`].
    ///
    /// Default implementation delegates to
    /// [`solve_with_raw_basis_view`](Self::solve_with_raw_basis_view) and calls
    /// [`SolutionView::to_owned`] on the result. Prefer
    /// [`solve_with_raw_basis_view`](Self::solve_with_raw_basis_view) on the hot
    /// path to avoid the allocation.
    ///
    /// # Errors
    ///
    /// Same error contract as [`solve_with_raw_basis_view`](Self::solve_with_raw_basis_view).
    ///
    /// See Solver Interface Trait SS2.5.
    fn solve_with_raw_basis(&mut self, basis: &RawBasis) -> Result<LpSolution, SolverError> {
        self.solve_with_raw_basis_view(basis).map(|v| v.to_owned())
    }

    /// Returns accumulated solve metrics (snapshot of monotonically increasing counters).
    ///
    /// Statistics accumulate since construction; [`reset`] does not zero them.
    /// All fields non-negative.
    ///
    /// See Solver Interface Trait SS2.8.
    fn statistics(&self) -> SolverStatistics;

    /// Returns a static string identifying the solver backend (e.g., `"HiGHS"`).
    ///
    /// Used for logging, diagnostics, and checkpoint metadata.
    ///
    /// See Solver Interface Trait SS2.9.
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

        fn solve_view(
            &mut self,
        ) -> Result<crate::types::SolutionView<'_>, crate::types::SolverError> {
            Err(crate::types::SolverError::InternalError {
                message: "noop".to_string(),
                error_code: None,
            })
        }

        fn solve_with_basis_view(
            &mut self,
            _basis: &crate::types::Basis,
        ) -> Result<crate::types::SolutionView<'_>, crate::types::SolverError> {
            Err(crate::types::SolverError::InternalError {
                message: "noop".to_string(),
                error_code: None,
            })
        }

        fn reset(&mut self) {}

        fn get_basis(&mut self) -> crate::types::Basis {
            crate::types::Basis {
                col_status: vec![],
                row_status: vec![],
            }
        }

        fn get_raw_basis(&mut self, _out: &mut crate::types::RawBasis) {}

        fn solve_with_raw_basis_view(
            &mut self,
            _basis: &crate::types::RawBasis,
        ) -> Result<crate::types::SolutionView<'_>, crate::types::SolverError> {
            Err(crate::types::SolverError::InternalError {
                message: "noop".to_string(),
                error_code: None,
            })
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
        let mut solver = NoopSolver;
        let basis = solver.get_basis();
        assert!(basis.col_status.is_empty());
        assert!(basis.row_status.is_empty());
    }

    #[test]
    fn test_noop_solver_get_raw_basis_noop() {
        use crate::types::RawBasis;

        let mut solver = NoopSolver;
        let mut raw = RawBasis::new(3, 2);
        // Pre-fill to detect any inadvertent writes
        raw.col_status.iter_mut().for_each(|v| *v = 99_i32);
        raw.row_status.iter_mut().for_each(|v| *v = 99_i32);
        solver.get_raw_basis(&mut raw);
        // NoopSolver does not modify the buffer
        assert!(raw.col_status.iter().all(|&v| v == 99_i32));
        assert!(raw.row_status.iter().all(|&v| v == 99_i32));
    }

    #[test]
    fn test_noop_solver_solve_with_raw_basis_view_returns_internal_error() {
        use crate::types::{RawBasis, SolverError};

        let mut solver = NoopSolver;
        let raw = RawBasis::new(0, 0);
        assert!(matches!(
            solver.solve_with_raw_basis_view(&raw),
            Err(SolverError::InternalError { .. })
        ));
    }

    #[test]
    fn test_noop_solver_solve_with_raw_basis_returns_internal_error() {
        use crate::types::{RawBasis, SolverError};

        let mut solver = NoopSolver;
        let raw = RawBasis::new(0, 0);
        assert!(matches!(
            solver.solve_with_raw_basis(&raw),
            Err(SolverError::InternalError { .. })
        ));
    }

    #[test]
    fn test_noop_solver_all_methods() {
        use crate::types::{Basis, RowBatch, SolverError, StageTemplate};

        let template = StageTemplate {
            num_cols: 1,
            num_rows: 0,
            num_nz: 0,
            col_starts: vec![0_i32, 0],
            row_indices: vec![],
            values: vec![],
            col_lower: vec![0.0],
            col_upper: vec![1.0],
            objective: vec![1.0],
            row_lower: vec![],
            row_upper: vec![],
            n_state: 0,
            n_transfer: 0,
            n_dual_relevant: 0,
            n_hydro: 0,
            max_par_order: 0,
        };

        let batch = RowBatch {
            num_rows: 0,
            row_starts: vec![0_i32],
            col_indices: vec![],
            values: vec![],
            row_lower: vec![],
            row_upper: vec![],
        };

        let basis = Basis {
            col_status: vec![],
            row_status: vec![],
        };

        let mut solver = NoopSolver;
        solver.load_model(&template);
        solver.add_rows(&batch);
        solver.set_row_bounds(&[], &[], &[]);
        solver.set_col_bounds(&[], &[], &[]);

        // Both solve and solve_with_basis return InternalError for NoopSolver
        // (via their default implementations that delegate to solve_view /
        // solve_with_basis_view).
        assert!(matches!(
            solver.solve(),
            Err(SolverError::InternalError { .. })
        ));
        assert!(matches!(
            solver.solve_with_basis(&basis),
            Err(SolverError::InternalError { .. })
        ));

        // solve_view and solve_with_basis_view also return InternalError.
        assert!(matches!(
            solver.solve_view(),
            Err(SolverError::InternalError { .. })
        ));
        assert!(matches!(
            solver.solve_with_basis_view(&basis),
            Err(SolverError::InternalError { .. })
        ));

        solver.reset();
    }
}
