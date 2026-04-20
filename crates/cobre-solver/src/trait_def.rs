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
///   `set_col_bounds`, `solve`) take `&mut self`.
/// - Methods that write to internal scratch buffers (`get_basis`) take `&mut self`.
/// - Read-only query methods (`statistics`, `name`) take `&self`.
///
/// # Solve-to-solve Contract
///
/// Implementations MAY retain internal state (factorization, simplex basis)
/// between consecutive `solve` calls on the same instance as a performance
/// optimization. Callers that need a reproducible reset between runs must
/// either call `load_model` (which resets topology) or pass an explicit
/// `Basis` via `solve(Some(&b))`. See [`SolverInterface::solve`] for the
/// full solve-to-solve contract.
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
    /// Bulk-loads a pre-assembled structural LP (first step of rebuild sequence).
    ///
    /// Replaces any previous model. Validates template is a valid CSC matrix
    /// with `num_cols > 0` and `num_rows > 0` (panic on violation).
    ///
    /// See Solver Interface Trait SS2.1.
    fn load_model(&mut self, template: &StageTemplate);

    /// Append constraint rows to the dynamic constraint region.
    ///
    /// Requires [`load_model`](Self::load_model) called first and
    /// `cuts` to have valid CSR data with column indices in
    /// `[0, num_cols)` (panic on violation).
    ///
    /// # Caller patterns
    ///
    /// In a baked-template architecture, the primary LP solve path
    /// loads pre-materialized templates that already contain all
    /// active rows as structural rows; that path does not call
    /// `add_rows`. Three legitimate caller patterns survive:
    ///
    /// 1. **Per-iteration delta append**: when a downstream pass
    ///    needs to extend a previously-baked template with rows
    ///    generated mid-iteration, `add_rows` appends those delta
    ///    rows on top of the baked template rather than triggering
    ///    a re-bake at every stage.
    /// 2. **Append-only LP managers**: an LP that grows
    ///    monotonically across iterations (cuts only added, never
    ///    removed) keeps cumulative setup cost at `O(n)` by
    ///    appending rather than re-baking. Re-baking the template
    ///    each iteration would be `O(n^2)` and is not pursued.
    /// 3. **Test-only fallback**: a no-tracking-map branch in some
    ///    test contexts performs a full rebuild via
    ///    [`load_model`](Self::load_model) followed by `add_rows`.
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

    /// Solve the LP currently loaded on the backend.
    ///
    /// Hot-path method encapsulating internal retry logic and optional warm-start.
    /// Requires [`Self::load_model`] called first and scenario patches applied.
    /// The returned [`SolutionView`] borrows solver-internal buffers and is valid
    /// until the next `&mut self` call. Call [`SolutionView::to_owned`] when the
    /// solution must outlive the borrow.
    ///
    /// # Contract — solve-to-solve behavior (revised 2026-04-19)
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
    /// [`crate::HighsSolver`] retains its internal simplex basis and
    /// factorization across consecutive `solve` calls as a warm-start
    /// optimization. This is the primary warm-start mechanism for backward-pass
    /// workloads where the LP shape is constant across trial points at the same
    /// (stage, opening). Callers that need solve-independence must pass an
    /// explicit `Basis` (or call `load_model` to reset topology). The performance
    /// fix in commit `25f1351` (April 2026) removed an unconditional
    /// `Highs_clearSolver` call that defeated this optimization;
    /// cross-sampled-state reproducibility concerns raised during that fix are
    /// documented at the plan level and deferred to a follow-up design
    /// (see known-concerns).
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
    ///
    /// # Examples
    ///
    /// ```no_run
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
    /// ```
    ///
    /// See [Solver Interface Trait SS2.4] for the post-conditions on
    /// [`SolutionView`] lifetime and the thread-safety constraints inherited
    /// from the trait's `Send` bound.
    fn solve(&mut self, basis: Option<&Basis>) -> Result<SolutionView<'_>, SolverError>;

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

    /// Returns accumulated solve metrics (snapshot of monotonically increasing counters).
    ///
    /// Statistics accumulate since construction; they are never zeroed.
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

    /// Record that `reconstruct_basis` applied a stored basis via slot reconciliation.
    /// Default implementation is a no-op; `HighsSolver` overrides to increment
    /// `SolverStatistics::basis_reconstructions` by 1.
    /// A non-zero count indicates basis reconstruction is active on this solver instance.
    fn record_reconstruction_stats(&mut self) {}
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

        fn solve(
            &mut self,
            _basis: Option<&crate::types::Basis>,
        ) -> Result<crate::types::SolutionView<'_>, crate::types::SolverError> {
            Err(crate::types::SolverError::InternalError {
                message: "noop".to_string(),
                error_code: None,
            })
        }

        fn get_basis(&mut self, _out: &mut crate::types::Basis) {}

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

        let result = solver.solve(None);
        assert!(matches!(result, Err(SolverError::InternalError { .. })));
    }
}
