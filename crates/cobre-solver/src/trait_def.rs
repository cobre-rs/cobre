//! The [`SolverInterface`] trait definition.
//!
//! This module defines the central abstraction through which optimization
//! algorithms interact with LP solvers.

use crate::types::{Basis, RowBatch, SolutionView, SolverError, SolverStatistics, StageTemplate};

/// Backend-agnostic interface for LP solver instances.
///
/// # Design
///
/// The trait is resolved as a **generic type parameter at compile time**
/// (compile-time monomorphization for FFI-wrapping trait),
/// not as `dyn SolverInterface`. This monomorphization approach
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
/// When `solve` or `solve_with_basis` returns `Err`, the solver's
/// internal state is unspecified. The **caller** is responsible for calling
/// `reset()` before reusing the instance for another solve sequence. Failing to
/// call `reset()` after an error may produce incorrect results or panics.
///
/// # Usage as a Generic Bound
///
/// ```rust
/// use cobre_solver::{SolverInterface, SolutionView, SolverError};
///
/// fn run_solve<S: SolverInterface>(solver: &mut S) -> Result<SolutionView<'_>, SolverError> {
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

    /// Solves the LP, returning a zero-copy view or terminal error after retry exhaustion.
    ///
    /// Hot-path method encapsulating internal retry logic. Requires [`Self::load_model`]
    /// called first and scenario patches applied. On error, caller must call
    /// [`Self::reset`] before reusing. The returned [`SolutionView`] borrows
    /// solver-internal buffers and is valid until the next `&mut self` call. Call
    /// [`SolutionView::to_owned`] when the solution must outlive the borrow.
    ///
    /// # Errors
    ///
    /// Returns `Err(SolverError)` when all internal retry attempts exhausted.
    /// Possible variants: [`SolverError::Infeasible`], [`SolverError::Unbounded`],
    /// [`SolverError::NumericalDifficulty`], [`SolverError::TimeLimitExceeded`],
    /// [`SolverError::IterationLimit`], or [`SolverError::InternalError`].
    ///
    /// See Solver Interface Trait SS2.4.
    fn solve(&mut self) -> Result<SolutionView<'_>, SolverError>;

    /// Clears internal solver state for error recovery or LP structure change.
    ///
    /// Requires [`Self::load_model`] before next solve. Preserves `SolverStatistics`
    /// counters; does not zero them.
    ///
    /// See Solver Interface Trait SS2.6.
    fn reset(&mut self);

    /// Clears the solver's derived state (factorization, warm-start weights,
    /// PRNG state, cycle-avoidance taboos, all simplex status flags) while
    /// keeping the loaded LP intact. After this call the next solve behaves
    /// as if it were the first solve on a fresh instance with the same LP.
    ///
    /// This is the deterministic-reset primitive for warm-start chains that
    /// span work-distribution variations — specifically, the backward pass
    /// reusing a solver across trial points at a single stage.
    ///
    /// Contract: caller does NOT need to call `load_model` before the next
    /// solve. Bounds should be set (via `set_row_bounds` /
    /// `set_col_bounds`) and a warm-start basis may be installed via
    /// `solve_with_basis`.
    ///
    /// # Errors
    /// `SolverError::Unsupported` on backends without an equivalent cheap
    /// reset. Such backends should be invoked via the `reset` +
    /// `load_model` path instead.
    fn clear_solver_state(&mut self) -> Result<(), SolverError> {
        Err(SolverError::Unsupported(
            "clear_solver_state not implemented for this backend",
        ))
    }

    /// Writes solver-native `i32` status codes into a caller-owned [`Basis`] buffer.
    ///
    /// The caller pre-allocates a [`Basis`] with [`Basis::new`] and reuses it
    /// across iterations, eliminating per-element enum translation overhead.
    ///
    /// The buffer is not resized by this method. The implementation writes into
    /// the first `num_cols` entries of `out.col_status` and the first `num_rows`
    /// entries of `out.row_status`. Panics if no model is loaded.
    ///
    /// See Solver Interface Trait SS2.7.
    fn get_basis(&mut self, out: &mut Basis);

    /// Injects a basis and solves, returning a zero-copy [`SolutionView`].
    ///
    /// Status codes in `basis` are injected directly without per-element enum
    /// translation. On success the returned view borrows solver-internal buffers
    /// and is valid until the next `&mut self` call. Call [`SolutionView::to_owned`]
    /// when the solution must outlive the borrow.
    ///
    /// # Errors
    ///
    /// Same error contract as [`solve`](Self::solve).
    ///
    /// See Solver Interface Trait SS2.5.
    fn solve_with_basis(&mut self, basis: &Basis) -> Result<SolutionView<'_>, SolverError>;

    /// Returns accumulated solve metrics (snapshot of monotonically increasing counters).
    ///
    /// Statistics accumulate since construction; [`Self::reset`] does not zero them.
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

    /// Returns the solver name and version as a human-readable string.
    ///
    /// Example: `"HiGHS 1.8.0"`
    ///
    /// See Solver Interface Trait SS2.9.
    fn solver_name_version(&self) -> String;

    /// Record slot-tracked basis reconstruction statistics.
    ///
    /// Called after each `reconstruct_basis` call on the forward or backward
    /// path. Default implementation is a no-op; `HighsSolver` overrides to
    /// accumulate into `SolverStatistics` fields.
    ///
    /// - `preserved`: cut rows whose slot identity was found in the stored
    ///   basis and whose status was copied directly.
    /// - `new_tight`: new cut rows (slot not in stored basis) evaluated as
    ///   tight or violated at the padding state.
    /// - `new_slack`: new cut rows evaluated as slack at the padding state.
    /// - `demotions`: BASIC row statuses demoted to LOWER by
    ///   `enforce_basic_count_invariant` on the forward path (ticket-009).
    ///   Pass `0` on the backward path where no demotion pass is applied.
    fn record_reconstruction_stats(
        &mut self,
        _preserved: u32,
        _new_tight: u32,
        _new_slack: u32,
        _demotions: u32,
    ) {
    }
}

#[cfg(test)]
mod tests {
    use super::SolverInterface;

    // Verify trait is usable as a generic bound (compile-time monomorphization).
    fn accepts_solver<S: SolverInterface>(_: &S) {}

    struct NoopSolver;

    impl SolverInterface for NoopSolver {
        fn load_model(&mut self, _template: &crate::types::StageTemplate) {}

        fn add_rows(&mut self, _cuts: &crate::types::RowBatch) {}

        fn set_row_bounds(&mut self, _indices: &[usize], _lower: &[f64], _upper: &[f64]) {}

        fn set_col_bounds(&mut self, _indices: &[usize], _lower: &[f64], _upper: &[f64]) {}

        fn solve(&mut self) -> Result<crate::types::SolutionView<'_>, crate::types::SolverError> {
            Err(crate::types::SolverError::InternalError {
                message: "noop".to_string(),
                error_code: None,
            })
        }

        fn reset(&mut self) {}

        fn get_basis(&mut self, _out: &mut crate::types::Basis) {}

        fn solve_with_basis(
            &mut self,
            _basis: &crate::types::Basis,
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

        fn solver_name_version(&self) -> String {
            "NoopSolver 0.0.0".to_string()
        }
    }

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
    fn test_noop_solver_get_basis_noop() {
        use crate::types::Basis;

        let mut solver = NoopSolver;
        let mut raw = Basis::new(3, 2);
        raw.col_status.iter_mut().for_each(|v| *v = 99_i32);
        raw.row_status.iter_mut().for_each(|v| *v = 99_i32);
        solver.get_basis(&mut raw);
        assert!(raw.col_status.iter().all(|&v| v == 99_i32));
        assert!(raw.row_status.iter().all(|&v| v == 99_i32));
    }

    #[test]
    fn test_noop_solver_solve_with_basis_returns_internal_error() {
        use crate::types::{Basis, SolverError};

        let mut solver = NoopSolver;
        let raw = Basis::new(0, 0);
        let result = solver.solve_with_basis(&raw);
        assert!(matches!(result, Err(SolverError::InternalError { .. })));
    }

    #[test]
    fn test_noop_solver_clear_solver_state_returns_unsupported() {
        use crate::types::SolverError;
        let mut solver = NoopSolver;
        let result = solver.clear_solver_state();
        match result {
            Err(SolverError::Unsupported(msg)) => {
                assert!(msg.contains("clear_solver_state"));
            }
            _ => panic!("expected Unsupported, got {result:?}"),
        }
    }

    #[test]
    fn test_unsupported_display_format() {
        use crate::types::SolverError;
        let err = SolverError::Unsupported("test message");
        let formatted = format!("{err}");
        assert!(formatted.contains("unsupported"), "got {formatted}");
        assert!(formatted.contains("test message"), "got {formatted}");
    }

    #[test]
    fn test_noop_solver_all_methods() {
        use crate::types::{RowBatch, SolverError, StageTemplate};

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
            col_scale: Vec::new(),
            row_scale: Vec::new(),
        };

        let batch = RowBatch {
            num_rows: 0,
            row_starts: vec![0_i32],
            col_indices: vec![],
            values: vec![],
            row_lower: vec![],
            row_upper: vec![],
        };

        let mut solver = NoopSolver;
        solver.load_model(&template);
        solver.add_rows(&batch);
        solver.set_row_bounds(&[], &[], &[]);
        solver.set_col_bounds(&[], &[], &[]);

        let result = solver.solve();
        assert!(matches!(result, Err(SolverError::InternalError { .. })));

        solver.reset();
    }
}
