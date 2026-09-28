//! The [`SolverInterface`] trait definition.
//!
//! This module defines the central abstraction through which optimization
//! algorithms interact with LP solvers.

use crate::types::{Basis, RowBatch, SolutionView, SolverError, SolverStatistics, StageTemplate};

/// Backend-agnostic interface for LP solver instances.
///
/// # Design
///
/// Resolved as a generic type parameter (compile-time monomorphization), **not**
/// as `dyn SolverInterface`, to keep virtual dispatch off the hot path.
///
/// # Thread Safety
///
/// Requires `Send` but **not** `Sync`: a C-library solver handle (`HiGHS`, CLP)
/// holds mutable internal state (factorization workspace, working arrays) that is
/// not thread-safe, so each worker thread owns exactly one instance. Adding `Sync`
/// would permit unsound concurrent access. See Solver Workspaces SS1.1.
///
/// # Solve-to-solve Contract
///
/// Implementations MAY retain internal state between consecutive `solve` calls;
/// callers needing a reproducible reset must call `load_model` or pass an explicit
/// `Basis`. See [`SolverInterface::solve`] for the full contract.
///
/// # Usage as a Generic Bound
///
/// ```rust
/// use cobre_solver::{SolverInterface, SolutionView, SolverError};
///
/// fn run_solve<S: SolverInterface>(solver: &mut S) -> Result<SolutionView<'_>, SolverError> {
///     solver.solve(None)
/// }
/// ```
///
/// See [Solver Interface Trait SS1](../../../cobre-docs/src/specs/architecture/solver-interface-trait.md)
/// and [Solver Interface Trait SS5](../../../cobre-docs/src/specs/architecture/solver-interface-trait.md)
/// for the dispatch mechanism rationale.
pub trait SolverInterface: Send {
    /// The solver-specific profile type — each backend's full tunable-option
    /// surface (test mocks use a fieldless placeholder). The bounds let
    /// `ProfiledSolver` delta-track fields and default-construct without a factory.
    ///
    /// Backend-generic consumers should treat `Profile` as opaque (construct via
    /// `Default`, pass through `apply_profile`) and not name a concrete profile
    /// type; cross-backend tuning intent belongs in a backend-agnostic hint.
    type Profile: Copy + PartialEq + Default + Send;

    /// Apply all profile options to the underlying solver.
    ///
    /// Called by `ProfiledSolver` before each solve. Implementations MUST
    /// configure ALL fields of the profile in a single call, so any internal
    /// option reset is overridden by the active profile.
    fn apply_profile(&mut self, profile: &Self::Profile);

    /// Bulk-loads a pre-assembled structural LP, replacing any previous model.
    ///
    /// Validates the template is a valid CSC matrix with `num_cols > 0` and
    /// `num_rows > 0` (panic on violation).
    ///
    /// See Solver Interface Trait SS2.1.
    fn load_model(&mut self, template: &StageTemplate);

    /// Append constraint rows to the dynamic constraint region.
    ///
    /// Requires [`load_model`](Self::load_model) called first and `rows` to have
    /// valid CSR data with column indices in `[0, num_cols)` (panic on violation).
    ///
    /// See Solver Interface Trait SS2.2.
    fn add_rows(&mut self, rows: &RowBatch);

    /// Updates row bounds.
    ///
    /// `indices`, `lower`, and `upper` must have equal length, with all indices
    /// referencing valid rows and bounds finite. For equality constraints, set
    /// `lower[i] == upper[i]`. Panics if lengths differ or indices are out-of-bounds.
    ///
    /// See Solver Interface Trait SS2.3.
    fn set_row_bounds(&mut self, indices: &[usize], lower: &[f64], upper: &[f64]);

    /// Updates column bounds.
    ///
    /// `indices`, `lower`, and `upper` must have equal length, with all indices
    /// referencing valid columns and bounds finite. Panics if lengths differ or
    /// indices are out-of-bounds.
    ///
    /// See Solver Interface Trait SS2.3a.
    fn set_col_bounds(&mut self, indices: &[usize], lower: &[f64], upper: &[f64]);

    /// Solve the LP currently loaded on the backend.
    ///
    /// Hot-path method encapsulating internal retry logic and optional warm-start.
    /// Requires [`Self::load_model`] called first and any pending row/column
    /// bound updates applied.
    /// The returned [`SolutionView`] borrows solver-internal buffers and is valid
    /// until the next `&mut self` call. Call [`SolutionView::to_owned`] when the
    /// solution must outlive the borrow.
    ///
    /// # Contract — solve-to-solve behavior
    ///
    /// `solve` returns the optimum of the LP currently loaded on the backend,
    /// subject to the current column/row bounds. If `basis` is `Some(&b)`, the
    /// solver attempts to warm-start from `b`; a basis that fails
    /// `isBasisConsistent` returns [`SolverError::BasisInconsistent`].
    ///
    /// `basis = Some(&b)` installs `b` before running the simplex.
    /// `basis = None` warm-starts from whatever basis this instance currently
    /// holds (itself determined by prior `solve` history on the same instance).
    ///
    /// Implementations MAY retain internal state (factorization, simplex basis)
    /// between consecutive `solve` calls on the same instance as a performance
    /// optimization. This means the result of a cold-start `solve(None)` can
    /// depend on prior `solve` history on the same instance through the retained
    /// internal basis. Callers that need a reproducible reset between runs must
    /// either call `load_model` (which resets topology) or pass an explicit
    /// `Basis` via `solve(Some(&b))`.
    ///
    /// # Errors
    ///
    /// Returns `Err(SolverError)` after internal retry exhaustion.
    /// Variants:
    /// - [`SolverError::Infeasible`] — LP has no feasible solution.
    /// - [`SolverError::Unbounded`] — objective is unbounded below.
    /// - [`SolverError::NumericalDifficulty`] — retry sequence exhausted without
    ///   convergence.
    /// - [`SolverError::TimeLimitExceeded`] — wall-clock budget exceeded.
    /// - [`SolverError::IterationLimit`] — simplex iteration budget exceeded
    ///   across all retry levels.
    /// - [`SolverError::InternalError`] — FFI layer returned an error.
    /// - [`SolverError::BasisInconsistent`] — ONLY when `basis = Some(&b)` and
    ///   `b` fails the solver's consistency check.
    /// - [`SolverError::BasisRowCountMismatch`] — ONLY when `basis = Some(&b)`
    ///   and `b.row_status.len()` is smaller than the current LP's row count.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # #[cfg(feature = "highs")] {
    /// use cobre_solver::{Basis, HighsSolver, SolverInterface};
    ///
    /// let mut solver = HighsSolver::new().expect("HiGHS init");
    /// # let template = unimplemented!();
    /// solver.load_model(&template);
    ///
    /// // Cold-start solve: no stored basis.
    /// let cold = solver.solve(None).expect("cold solve");
    /// let cold_obj = cold.objective;
    ///
    /// // Warm-start solve: reinstall a previously captured basis.
    /// let basis: Basis = unimplemented!("previously captured");
    /// let warm = solver.solve(Some(&basis)).expect("warm solve");
    /// assert!((warm.objective - cold_obj).abs() < 1e-9);
    /// # }
    /// ```
    ///
    /// See [Solver Interface Trait SS2.4] for the post-conditions on
    /// [`SolutionView`] lifetime and the thread-safety constraints inherited
    /// from the trait's `Send` bound.
    fn solve(&mut self, basis: Option<&Basis>) -> Result<SolutionView<'_>, SolverError>;

    /// Writes canonical [`BasisStatus`](crate::BasisStatus) values into a caller-owned [`Basis`] buffer.
    ///
    /// The buffer (from [`Basis::new`], reused across iterations) is **not** resized;
    /// writes go into the first `num_cols` entries of `out.col_status` and the first
    /// `num_rows` entries of `out.row_status`. Panics if no model is loaded.
    ///
    /// See Solver Interface Trait SS2.7.
    fn get_basis(&mut self, out: &mut Basis);

    /// Returns accumulated solve metrics; counters accumulate since construction
    /// and are never zeroed.
    ///
    /// See Solver Interface Trait SS2.8.
    fn statistics(&self) -> SolverStatistics;

    /// Copy accumulated solve metrics into a caller-owned buffer, reusing its
    /// `retry_level_histogram` allocation.
    ///
    /// Equivalent to `*out = self.statistics()` but performs no heap allocation
    /// when `out.retry_level_histogram` already has sufficient capacity.
    fn statistics_into(&self, out: &mut SolverStatistics);

    /// Returns a static string identifying the solver backend (e.g., `"HiGHS"`).
    ///
    /// See Solver Interface Trait SS2.9.
    fn name(&self) -> &'static str;

    /// Returns the solver name and version as a human-readable string.
    ///
    /// Example: `"HiGHS 1.8.0"`
    ///
    /// See Solver Interface Trait SS2.9.
    fn solver_name_version(&self) -> String;

    /// Record that `reconstruct_basis` applied a stored basis via slot reconciliation.
    /// Default no-op; `HighsSolver` overrides to increment
    /// `SolverStatistics::basis_reconstructions`.
    fn record_reconstruction_stats(&mut self) {}

    /// Reset the solver's internal working state to a clean baseline between
    /// independent solve sequences (e.g. between unrelated batches of models on
    /// the same handle), discarding any simplex state — factorization and,
    /// crucially, the pricing/edge-weight reference frame — carried over from
    /// prior solves.
    ///
    /// This is a **determinism** hook, not a performance one: it ensures a
    /// solve's result cannot depend on which models the same handle solved
    /// before it, so output stays bit-identical across thread/rank counts.
    ///
    /// Default: no-op. Override when your backend retains solver state across
    /// `load_model` calls (e.g., factorization, pricing weights) that would make
    /// results depend on prior solve history.
    fn reset_solver_state(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::SolverInterface;
    use crate::profile::MockProfile;
    use crate::types::{
        Basis, RowBatch, SolutionView, SolverError, SolverStatistics, StageTemplate,
    };

    fn accepts_solver<S: SolverInterface>(_: &S) {}

    struct NoopSolver;

    impl SolverInterface for NoopSolver {
        type Profile = MockProfile;

        fn apply_profile(&mut self, _profile: &MockProfile) {}

        fn load_model(&mut self, _template: &StageTemplate) {}

        fn add_rows(&mut self, _rows: &RowBatch) {}

        fn set_row_bounds(&mut self, _indices: &[usize], _lower: &[f64], _upper: &[f64]) {}

        fn set_col_bounds(&mut self, _indices: &[usize], _lower: &[f64], _upper: &[f64]) {}

        fn solve(&mut self, _basis: Option<&Basis>) -> Result<SolutionView<'_>, SolverError> {
            Err(SolverError::InternalError {
                message: "noop".to_string(),
                error_code: None,
            })
        }

        fn get_basis(&mut self, _out: &mut Basis) {}

        fn statistics(&self) -> SolverStatistics {
            SolverStatistics::default()
        }

        fn statistics_into(&self, out: &mut SolverStatistics) {
            out.copy_from(&SolverStatistics::default());
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
    fn test_noop_solver_statistics_into_initial() {
        use crate::types::SolverStatistics;

        let mut buf = SolverStatistics {
            solve_count: 7,
            success_count: 5,
            failure_count: 2,
            total_iterations: 99,
            retry_count: 3,
            total_solve_time_seconds: 12.5,
            ..SolverStatistics::default()
        };
        NoopSolver.statistics_into(&mut buf);
        assert_eq!(buf.solve_count, 0);
        assert_eq!(buf.success_count, 0);
        assert_eq!(buf.failure_count, 0);
        assert_eq!(buf.total_iterations, 0);
        assert_eq!(buf.retry_count, 0);
        assert_eq!(buf.total_solve_time_seconds, 0.0);
    }

    #[test]
    fn test_noop_solver_get_basis_noop() {
        use crate::BasisStatus;
        use crate::types::Basis;

        let mut solver = NoopSolver;
        let mut raw = Basis::new(3, 2);
        raw.col_status
            .iter_mut()
            .for_each(|v| *v = BasisStatus::Upper);
        raw.row_status
            .iter_mut()
            .for_each(|v| *v = BasisStatus::Upper);
        solver.get_basis(&mut raw);
        assert!(raw.col_status.iter().all(|&v| v == BasisStatus::Upper));
        assert!(raw.row_status.iter().all(|&v| v == BasisStatus::Upper));
    }

    #[test]
    fn test_noop_solver_solve_with_optional_basis_returns_internal_error() {
        use crate::types::{Basis, SolverError};

        let mut solver = NoopSolver;
        let raw = Basis::new(0, 0);
        let result = solver.solve(Some(&raw));
        assert!(matches!(result, Err(SolverError::InternalError { .. })));
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

        let result = solver.solve(None);
        assert!(matches!(result, Err(SolverError::InternalError { .. })));
    }
}
