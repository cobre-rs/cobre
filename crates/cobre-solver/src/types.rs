//! Core types for the solver abstraction layer.
//!
//! Defines the canonical representations of LP solutions, basis management,
//! and terminal solver errors used throughout the solver interface.

use core::fmt;

use crate::BasisStatus;

/// Simplex basis storing canonical [`BasisStatus`] values, backend-neutral across
/// `HiGHS` and CLP. Each backend maps to/from its native status codes at the FFI
/// capture/install boundary (`to_highs_code`/`from_highs_code`,
/// `to_clp_code`/`from_clp_code`).
///
/// The caller must match the buffer dimensions to the LP model before use.
///
/// See Solver Abstraction SS9.
#[derive(Debug, Clone)]
pub struct Basis {
    /// Canonical basis status for each column (length must equal `num_cols`).
    pub col_status: Vec<BasisStatus>,

    /// Canonical basis status for each row, including structural and dynamic rows.
    pub row_status: Vec<BasisStatus>,
}

impl Basis {
    /// Creates a `Basis` with status buffers of the requested lengths, filled with
    /// [`BasisStatus::Lower`] (`to_highs_code() == 0`, matching the historical
    /// zero-fill default), for reuse across solves via
    /// [`crate::SolverInterface::get_basis`].
    #[must_use]
    pub fn new(num_cols: usize, num_rows: usize) -> Self {
        Self {
            col_status: vec![BasisStatus::Lower; num_cols],
            row_status: vec![BasisStatus::Lower; num_rows],
        }
    }
}

/// Complete solution from a successful LP solve.
///
/// All values are in the original (unscaled) problem space. Dual values
/// are pre-normalized to the canonical sign convention defined in
/// [Solver Abstraction SS8](../../../cobre-docs/src/specs/architecture/solver-abstraction.md)
/// before this struct is returned -- solver-specific sign differences are
/// resolved within the [`crate::SolverInterface`] implementation.
///
/// See [Solver Interface Trait SS4.1](../../../cobre-docs/src/specs/architecture/solver-interface-trait.md).
#[derive(Debug, Clone)]
pub struct LpSolution {
    /// Optimal objective value (minimization sense).
    pub objective: f64,

    /// Primal variable values, indexed by column (length equals `num_cols`).
    pub primal: Vec<f64>,

    /// Dual multipliers (shadow prices), indexed by row (length equals `num_rows`).
    /// Normalized to canonical sign convention.
    pub dual: Vec<f64>,

    /// Reduced costs, indexed by column (length equals `num_cols`).
    pub reduced_costs: Vec<f64>,

    /// Number of simplex iterations performed for this solve.
    pub iterations: u64,

    /// Wall-clock solve time in seconds (excluding retry overhead).
    pub solve_time_seconds: f64,
}

/// Zero-copy view of an LP solution, borrowing directly from solver-internal buffers.
///
/// Valid until the next mutating method call on the solver (any `&mut self` call).
/// This is enforced at compile time by the Rust borrow checker: the lifetime `'a`
/// ties the view to the solver instance that produced it.
///
/// Use [`SolutionView::to_owned`] to convert to an owned [`LpSolution`] when the
/// solution data must outlive the current borrow, or when the same data will be
/// accessed after a subsequent solver call.
///
/// See [Solver Interface Trait SS4.1](../../../cobre-docs/src/specs/architecture/solver-interface-trait.md).
#[derive(Debug, Clone, Copy)]
pub struct SolutionView<'a> {
    /// Optimal objective value (minimization sense).
    pub objective: f64,

    /// Primal variable values, indexed by column (length equals `num_cols`).
    pub primal: &'a [f64],

    /// Dual multipliers (shadow prices), indexed by row (length equals `num_rows`).
    /// Normalized to canonical sign convention.
    pub dual: &'a [f64],

    /// Reduced costs, indexed by column (length equals `num_cols`).
    pub reduced_costs: &'a [f64],

    /// Number of simplex iterations performed for this solve.
    pub iterations: u64,

    /// Wall-clock solve time in seconds (excluding retry overhead).
    pub solve_time_seconds: f64,
}

impl SolutionView<'_> {
    /// Clones the borrowed slices into owned [`Vec`]s, producing an [`LpSolution`].
    #[must_use]
    pub fn to_owned(&self) -> LpSolution {
        LpSolution {
            objective: self.objective,
            primal: self.primal.to_vec(),
            dual: self.dual.to_vec(),
            reduced_costs: self.reduced_costs.to_vec(),
            iterations: self.iterations,
            solve_time_seconds: self.solve_time_seconds,
        }
    }
}

/// Accumulated solve metrics for a single solver instance.
///
/// Counters grow monotonically from construction. They are thread-local --
/// each thread owns one solver instance and accumulates its own statistics.
/// Statistics are aggregated across threads via reduction after training
/// completes.
///
/// Statistics counters persist across model reloads for the lifetime of the
/// solver instance.
///
/// See [Solver Interface Trait SS4.3](../../../cobre-docs/src/specs/architecture/solver-interface-trait.md).
#[derive(Debug, Clone, Default)]
pub struct SolverStatistics {
    /// Total number of `solve` calls (cold-start and warm-start).
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

    /// Number of warm-start `solve(Some(&basis))` calls in which
    /// `cobre_highs_set_basis_non_alien` rejected the offered basis because
    /// `isBasisConsistent` returned false (incremented once per rejected offer).
    pub basis_consistency_failures: u64,

    /// Number of solves that returned optimal on the first attempt (before any retry).
    ///
    /// Enables first-try rate computation: `first_try_rate = first_try_successes / solve_count`.
    /// The complement `success_count - first_try_successes` gives the number of retried solves.
    pub first_try_successes: u64,

    /// Total number of warm-start `solve(Some(&basis))` calls (basis offers).
    ///
    /// Combined with `basis_consistency_failures`, enables acceptance-rate computation:
    /// `basis_acceptance_rate = 1 - basis_consistency_failures / basis_offered`.
    pub basis_offered: u64,

    /// Total number of `load_model` calls.
    pub load_model_count: u64,

    /// Cumulative wall-clock time spent in `load_model` calls, in seconds.
    pub total_load_model_time_seconds: f64,

    /// Cumulative wall-clock time spent in `set_row_bounds` and `set_col_bounds` calls, in seconds.
    pub total_set_bounds_time_seconds: f64,

    /// Cumulative wall-clock time spent in `set_basis` FFI calls, in seconds.
    ///
    /// Accumulated by `solve(Some(&basis))` around the basis installation step.
    /// Cold-start `solve(None)` does not increment this counter.
    pub total_basis_set_time_seconds: f64,

    /// Number of `reconstruct_basis` invocations with a non-empty stored basis.
    /// Incremented via `record_reconstruction_stats`. A non-zero value indicates
    /// basis reconstruction is active on this solver instance.
    pub basis_reconstructions: u64,

    /// Per-level retry success histogram. Length depends on the solver backend
    /// (e.g. 12 for `HiGHS`). `retry_level_histogram[k]` counts how many solves
    /// were recovered at retry level `k`. The sum equals
    /// `success_count - first_try_successes`.
    pub retry_level_histogram: Vec<u64>,
}

impl SolverStatistics {
    /// Overwrite `self` from `src` in place, reusing `self.retry_level_histogram`'s
    /// allocation so no heap allocation occurs when its capacity suffices. After
    /// this call `*self` equals `src.clone()` field-for-field.
    pub fn copy_from(&mut self, src: &SolverStatistics) {
        self.solve_count = src.solve_count;
        self.success_count = src.success_count;
        self.failure_count = src.failure_count;
        self.total_iterations = src.total_iterations;
        self.retry_count = src.retry_count;
        self.total_solve_time_seconds = src.total_solve_time_seconds;
        self.basis_consistency_failures = src.basis_consistency_failures;
        self.first_try_successes = src.first_try_successes;
        self.basis_offered = src.basis_offered;
        self.load_model_count = src.load_model_count;
        self.total_load_model_time_seconds = src.total_load_model_time_seconds;
        self.total_set_bounds_time_seconds = src.total_set_bounds_time_seconds;
        self.total_basis_set_time_seconds = src.total_basis_set_time_seconds;
        self.basis_reconstructions = src.basis_reconstructions;
        self.retry_level_histogram
            .resize(src.retry_level_histogram.len(), 0);
        self.retry_level_histogram
            .copy_from_slice(&src.retry_level_histogram);
    }
}

/// Pre-assembled structural LP for one stage, in CSC (column-major) form.
///
/// Built once at initialization from resolved internal structures.
/// Shared read-only across all threads within an MPI rank.
/// Passed to [`crate::SolverInterface::load_model`] to bulk-load the LP.
///
/// Column and row ordering follows the LP layout convention defined in
/// [Solver Abstraction SS2](../../../cobre-docs/src/specs/architecture/solver-abstraction.md).
/// The calling algorithm crate owns construction of this type; `cobre-solver`
/// treats it as an opaque data holder and does not interpret the LP structure.
///
/// See [Solver Interface Trait SS4.4](../../../cobre-docs/src/specs/architecture/solver-interface-trait.md)
/// and [Solver Abstraction SS11.1](../../../cobre-docs/src/specs/architecture/solver-abstraction.md).
#[derive(Debug, Clone)]
pub struct StageTemplate {
    /// Number of columns (decision variables) in the structural LP.
    pub num_cols: usize,

    /// Number of static rows (structural constraints, excluding dynamic rows).
    pub num_rows: usize,

    /// Number of non-zero entries in the structural constraint matrix.
    pub num_nz: usize,

    /// CSC column start offsets (length: `num_cols + 1`; `col_starts[num_cols] == num_nz`).
    pub col_starts: Vec<i32>,

    /// CSC row indices for each non-zero entry (length: `num_nz`).
    pub row_indices: Vec<i32>,

    /// CSC non-zero values (length: `num_nz`).
    pub values: Vec<f64>,

    /// Column lower bounds (length: `num_cols`; use `f64::NEG_INFINITY` for unbounded).
    pub col_lower: Vec<f64>,

    /// Column upper bounds (length: `num_cols`; use `f64::INFINITY` for unbounded).
    pub col_upper: Vec<f64>,

    /// Objective coefficients, minimization sense (length: `num_cols`).
    pub objective: Vec<f64>,

    /// Row lower bounds (length: `num_rows`; set equal to `row_upper` for equality).
    pub row_lower: Vec<f64>,

    /// Row upper bounds (length: `num_rows`; set equal to `row_lower` for equality).
    pub row_upper: Vec<f64>,

    /// Number of state variables (contiguous prefix of columns).
    pub n_state: usize,

    /// Per-column scaling factors for numerical conditioning.
    ///
    /// When non-empty (length `num_cols`), the constraint matrix, objective
    /// coefficients, and column bounds have been pre-scaled by these factors.
    /// The calling algorithm is responsible for unscaling primal values after
    /// each solve: `x_original[j] = col_scale[j] * x_scaled[j]`.
    ///
    /// When empty, no column scaling has been applied and solver results are
    /// used directly.
    pub col_scale: Vec<f64>,

    /// Per-row scaling factors for numerical conditioning.
    ///
    /// When non-empty (length `num_rows`), the constraint matrix and row bounds
    /// have been pre-scaled by these factors. The calling algorithm is responsible
    /// for unscaling dual values after each solve:
    /// `dual_original[i] = row_scale[i] * dual_scaled[i]`.
    ///
    /// When empty, no row scaling has been applied and solver results are
    /// used directly.
    pub row_scale: Vec<f64>,
}

/// Batch of constraint rows for addition to a loaded LP, in CSR (row-major) form.
///
/// Assembled from the row-pool activity bitmap before each LP rebuild
/// and passed to [`crate::SolverInterface::add_rows`] for a single batch call.
/// Rows are appended at the bottom of the constraint matrix in the dynamic
/// constraint region per
/// [Solver Abstraction SS2.2](../../../cobre-docs/src/specs/architecture/solver-abstraction.md).
///
/// See [Solver Interface Trait SS4.5](../../../cobre-docs/src/specs/architecture/solver-interface-trait.md)
/// and the row-pool assembly protocol in
/// [Solver Abstraction SS5.4](../../../cobre-docs/src/specs/architecture/solver-abstraction.md).
#[derive(Debug, Clone)]
pub struct RowBatch {
    /// Number of active constraint rows in this batch.
    pub num_rows: usize,

    /// CSR row start offsets (`i32` for `HiGHS` FFI compatibility).
    ///
    /// Length: `num_rows + 1`. Entry `row_starts[i]` is the index into
    /// `col_indices` and `values` where row `i` begins.
    /// `row_starts[num_rows]` equals the total number of non-zeros.
    pub row_starts: Vec<i32>,

    /// CSR column indices for each non-zero entry (`i32` for `HiGHS` FFI compatibility).
    ///
    /// Length: total non-zeros across all rows. Entry `col_indices[k]` is the
    /// column of the `k`-th non-zero value.
    pub col_indices: Vec<i32>,

    /// CSR non-zero values.
    ///
    /// Length: total non-zeros across all rows. Entry `values[k]` is the
    /// coefficient at column `col_indices[k]` in its row.
    pub values: Vec<f64>,

    /// Row lower bounds (RHS lower bounds for `>=` constraints).
    ///
    /// Length: `num_rows`. For `>=` constraints, this is the RHS lower bound.
    pub row_lower: Vec<f64>,

    /// Row upper bounds.
    ///
    /// Length: `num_rows`. Use `f64::INFINITY` for `>=` constraints (no finite upper bound).
    pub row_upper: Vec<f64>,
}

impl StageTemplate {
    /// Creates an empty [`StageTemplate`], a reusable output buffer for
    /// [`crate::freeze::freeze_rows_into_template`].
    ///
    /// An empty template is **not** a valid model for `load_model` (`num_cols == 0`,
    /// `num_rows == 0`); only pass it there after a `freeze_rows_into_template` call
    /// has populated it. A `Default` impl is intentionally omitted: an empty
    /// template is a surprising default that invites misuse.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            num_cols: 0,
            num_rows: 0,
            num_nz: 0,
            col_starts: Vec::new(),
            row_indices: Vec::new(),
            values: Vec::new(),
            col_lower: Vec::new(),
            col_upper: Vec::new(),
            objective: Vec::new(),
            row_lower: Vec::new(),
            row_upper: Vec::new(),
            n_state: 0,
            col_scale: Vec::new(),
            row_scale: Vec::new(),
        }
    }
}

impl RowBatch {
    /// Reset all buffers to empty without deallocating.
    pub fn clear(&mut self) {
        self.num_rows = 0;
        self.row_starts.clear();
        self.col_indices.clear();
        self.values.clear();
        self.row_lower.clear();
        self.row_upper.clear();
    }
}

/// Terminal LP solve error returned after all retry attempts are exhausted.
///
/// The calling algorithm uses the variant to determine its response: a hard stop
/// or termination with a diagnostic. Solver-internal errors (e.g., factorization
/// failures) are resolved by retry logic before reaching this level. See the error
/// categories in Solver Abstraction SS6.
#[derive(Debug)]
pub enum SolverError {
    /// The LP has no feasible solution.
    ///
    /// Indicates a data error (inconsistent bounds or constraints) or a
    /// modeling error. The calling algorithm should perform a hard stop.
    Infeasible,

    /// The LP objective is unbounded below.
    ///
    /// Indicates a modeling error (missing bounds, incorrect objective sign).
    /// The calling algorithm should perform a hard stop.
    Unbounded,

    /// Solver encountered numerical difficulties that persisted through all
    /// retry attempts.
    ///
    /// The calling algorithm should log the error and perform a hard stop.
    NumericalDifficulty {
        /// Human-readable description of the numerical issue from the solver.
        message: String,
    },

    /// Per-solve wall-clock time budget exhausted.
    TimeLimitExceeded {
        /// Elapsed wall-clock time in seconds at the point of termination.
        elapsed_seconds: f64,
    },

    /// Solver simplex iteration limit reached.
    IterationLimit {
        /// Number of simplex iterations performed before the limit was hit.
        iterations: u64,
    },

    /// Unrecoverable solver-internal failure.
    ///
    /// Covers FFI panics, memory allocation failures within the solver,
    /// corrupted internal state, or any error not classifiable into the above
    /// categories. The calling algorithm should log the error and perform a hard stop.
    InternalError {
        /// Human-readable error description.
        message: String,
        /// Solver-specific error code, if available.
        error_code: Option<i32>,
    },

    /// The backend does not implement the requested operation.
    ///
    /// The caller should fall back to an alternate code path (e.g.,
    /// `reset` + `load_model`).
    Unsupported(&'static str),

    /// The offered basis was rejected by the solver because the total
    /// number of basic variables did not match the row count.
    ///
    /// Indicates that the reconstructed basis violates the fundamental LP
    /// basis consistency invariant (`col_basic + row_basic == num_row`).
    /// The calling algorithm should perform a hard stop; this is not a
    /// recoverable solver-internal condition.
    BasisInconsistent {
        /// The LP row count at the point of rejection.
        num_row: i64,
        /// The total basic-variable count in the offered basis (`col_basic + row_basic`).
        total_basic: i64,
        /// Number of basic columns in the offered basis.
        col_basic: i64,
        /// Number of basic rows in the offered basis.
        row_basic: i64,
    },

    /// The offered warm-start basis has fewer row entries than the loaded LP
    /// has rows. The basis predates an `add_rows` growth and cannot be padded
    /// soundly (a BASIC pad is wrong for inequality-row slacks), so it is
    /// rejected; the caller should fall back to a cold solve.
    BasisRowCountMismatch {
        /// The LP row count at the point of rejection.
        lp_rows: usize,
        /// The row-entry count in the offered basis.
        basis_rows: usize,
    },
}

impl fmt::Display for SolverError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Infeasible => write!(f, "LP is infeasible"),
            Self::Unbounded => write!(f, "LP is unbounded"),
            Self::NumericalDifficulty { message } => {
                write!(f, "numerical difficulty: {message}")
            }
            Self::TimeLimitExceeded { elapsed_seconds } => {
                write!(f, "time limit exceeded after {elapsed_seconds:.3}s")
            }
            Self::IterationLimit { iterations } => {
                write!(f, "iteration limit reached after {iterations} iterations")
            }
            Self::InternalError {
                message,
                error_code,
            } => match error_code {
                Some(code) => write!(f, "internal solver error (code {code}): {message}"),
                None => write!(f, "internal solver error: {message}"),
            },
            Self::Unsupported(msg) => write!(f, "unsupported operation: {msg}"),
            Self::BasisInconsistent {
                num_row,
                total_basic,
                col_basic,
                row_basic,
            } => write!(
                f,
                "basis inconsistent: num_row={num_row}, total_basic={total_basic} (col_basic={col_basic}, row_basic={row_basic})"
            ),
            Self::BasisRowCountMismatch {
                lp_rows,
                basis_rows,
            } => write!(
                f,
                "basis row count mismatch: lp_rows={lp_rows}, basis_rows={basis_rows}"
            ),
        }
    }
}

impl std::error::Error for SolverError {}

#[cfg(test)]
mod tests {
    use super::{Basis, RowBatch, SolutionView, SolverError, SolverStatistics, StageTemplate};
    use crate::BasisStatus;

    #[test]
    fn test_basis_new_dimensions_and_zero_fill() {
        let rb = Basis::new(3, 2);
        assert_eq!(rb.col_status.len(), 3);
        assert_eq!(rb.row_status.len(), 2);
        assert!(rb.col_status.iter().all(|&v| v == BasisStatus::Lower));
        assert!(rb.row_status.iter().all(|&v| v == BasisStatus::Lower));
    }

    #[test]
    fn test_basis_new_empty() {
        let rb = Basis::new(0, 0);
        assert!(rb.col_status.is_empty());
        assert!(rb.row_status.is_empty());
    }

    #[test]
    fn test_basis_debug_and_clone() {
        let rb = Basis::new(2, 1);
        assert!(!format!("{rb:?}").is_empty());
        let cloned = rb.clone();
        assert_eq!(cloned.col_status, rb.col_status);
        assert_eq!(cloned.row_status, rb.row_status);
        let mut cloned2 = rb.clone();
        cloned2.col_status[0] = BasisStatus::Basic;
        assert_eq!(rb.col_status[0], BasisStatus::Lower);
    }

    #[test]
    fn test_solver_error_display_infeasible() {
        let msg = format!("{}", SolverError::Infeasible);
        assert!(msg.contains("infeasible"));
    }

    #[test]
    fn test_solver_error_display_all_variants() {
        let variants = [
            SolverError::Infeasible,
            SolverError::Unbounded,
            SolverError::NumericalDifficulty {
                message: "factorization failed".to_string(),
            },
            SolverError::TimeLimitExceeded {
                elapsed_seconds: 60.0,
            },
            SolverError::IterationLimit { iterations: 10_000 },
            SolverError::InternalError {
                message: "segfault in HiGHS".to_string(),
                error_code: Some(-1),
            },
            SolverError::BasisInconsistent {
                num_row: 2,
                total_basic: 5,
                col_basic: 3,
                row_basic: 2,
            },
            SolverError::BasisRowCountMismatch {
                lp_rows: 3,
                basis_rows: 2,
            },
        ];

        let messages: Vec<String> = variants.iter().map(|err| format!("{err}")).collect();
        for i in 0..messages.len() {
            for j in (i + 1)..messages.len() {
                assert_ne!(messages[i], messages[j]);
            }
        }
    }

    #[test]
    fn test_solver_error_is_std_error() {
        let err = SolverError::InternalError {
            message: "test".to_string(),
            error_code: None,
        };
        let _: &dyn std::error::Error = &err;
    }

    #[test]
    fn test_solver_statistics_default_all_zero() {
        let stats = SolverStatistics::default();
        assert_eq!(stats.solve_count, 0);
        assert_eq!(stats.success_count, 0);
        assert_eq!(stats.failure_count, 0);
        assert_eq!(stats.total_iterations, 0);
        assert_eq!(stats.retry_count, 0);
        assert_eq!(stats.total_solve_time_seconds, 0.0);
        assert_eq!(stats.basis_consistency_failures, 0);
        assert_eq!(stats.first_try_successes, 0);
        assert_eq!(stats.basis_offered, 0);
        assert_eq!(stats.total_load_model_time_seconds, 0.0);
        assert_eq!(stats.total_set_bounds_time_seconds, 0.0);
        assert_eq!(stats.basis_reconstructions, 0);
        assert!(stats.retry_level_histogram.is_empty());
    }

    fn make_fixture_statistics() -> SolverStatistics {
        SolverStatistics {
            solve_count: 11,
            success_count: 9,
            failure_count: 2,
            total_iterations: 3456,
            retry_count: 5,
            total_solve_time_seconds: 1.25,
            basis_consistency_failures: 1,
            first_try_successes: 7,
            basis_offered: 8,
            load_model_count: 4,
            total_load_model_time_seconds: 0.5,
            total_set_bounds_time_seconds: 0.25,
            total_basis_set_time_seconds: 0.125,
            basis_reconstructions: 6,
            retry_level_histogram: vec![1, 0, 2, 0, 3, 0, 0, 0, 4, 0, 0, 5],
        }
    }

    fn assert_statistics_eq(a: &SolverStatistics, b: &SolverStatistics) {
        assert_eq!(a.solve_count, b.solve_count);
        assert_eq!(a.success_count, b.success_count);
        assert_eq!(a.failure_count, b.failure_count);
        assert_eq!(a.total_iterations, b.total_iterations);
        assert_eq!(a.retry_count, b.retry_count);
        assert_eq!(a.total_solve_time_seconds, b.total_solve_time_seconds);
        assert_eq!(a.basis_consistency_failures, b.basis_consistency_failures);
        assert_eq!(a.first_try_successes, b.first_try_successes);
        assert_eq!(a.basis_offered, b.basis_offered);
        assert_eq!(a.load_model_count, b.load_model_count);
        assert_eq!(
            a.total_load_model_time_seconds,
            b.total_load_model_time_seconds
        );
        assert_eq!(
            a.total_set_bounds_time_seconds,
            b.total_set_bounds_time_seconds
        );
        assert_eq!(
            a.total_basis_set_time_seconds,
            b.total_basis_set_time_seconds
        );
        assert_eq!(a.basis_reconstructions, b.basis_reconstructions);
        assert_eq!(a.retry_level_histogram, b.retry_level_histogram);
    }

    #[test]
    fn test_solver_statistics_copy_from_equals_clone() {
        let src = make_fixture_statistics();

        // Buffer starts empty: first copy resizes the histogram from scratch.
        let mut buf = SolverStatistics::default();
        buf.copy_from(&src);
        assert_statistics_eq(&buf, &src);

        // Buffer with a differently-sized histogram is also overwritten exactly.
        let mut buf2 = SolverStatistics {
            retry_level_histogram: vec![99; 5],
            ..Default::default()
        };
        buf2.copy_from(&src);
        assert_statistics_eq(&buf2, &src);
    }

    #[test]
    fn test_solver_statistics_copy_from_no_realloc_second_call() {
        let src = make_fixture_statistics();

        // Pre-size the buffer's histogram to the source length so the first
        // copy stabilizes capacity; the second copy must not reallocate.
        let mut buf = SolverStatistics {
            retry_level_histogram: vec![0; src.retry_level_histogram.len()],
            ..Default::default()
        };
        buf.copy_from(&src);

        let ptr_before = buf.retry_level_histogram.as_ptr();
        buf.copy_from(&src);
        let ptr_after = buf.retry_level_histogram.as_ptr();

        assert_eq!(
            ptr_before, ptr_after,
            "second copy_from must not reallocate the histogram"
        );
        assert_statistics_eq(&buf, &src);
    }

    fn make_fixture_stage_template() -> StageTemplate {
        StageTemplate {
            num_cols: 3,
            num_rows: 2,
            num_nz: 3,
            col_starts: vec![0_i32, 2, 2, 3],
            row_indices: vec![0_i32, 1, 1],
            values: vec![1.0, 2.0, 1.0],
            col_lower: vec![0.0, 0.0, 0.0],
            col_upper: vec![10.0, f64::INFINITY, 8.0],
            objective: vec![0.0, 1.0, 50.0],
            row_lower: vec![6.0, 14.0],
            row_upper: vec![6.0, 14.0],
            n_state: 1,
            col_scale: Vec::new(),
            row_scale: Vec::new(),
        }
    }

    #[test]
    fn test_stage_template_construction() {
        let tmpl = make_fixture_stage_template();

        assert_eq!(tmpl.num_cols, 3);
        assert_eq!(tmpl.num_rows, 2);
        assert_eq!(tmpl.num_nz, 3);
        assert_eq!(tmpl.col_starts, vec![0_i32, 2, 2, 3]);
        assert_eq!(tmpl.row_indices, vec![0_i32, 1, 1]);
        assert_eq!(tmpl.values, vec![1.0, 2.0, 1.0]);

        assert_eq!(tmpl.col_lower, vec![0.0, 0.0, 0.0]);
        assert_eq!(tmpl.col_upper[0], 10.0);
        assert!(tmpl.col_upper[1].is_infinite() && tmpl.col_upper[1] > 0.0);
        assert_eq!(tmpl.col_upper[2], 8.0);

        assert_eq!(tmpl.objective, vec![0.0, 1.0, 50.0]);
        assert_eq!(tmpl.row_lower, vec![6.0, 14.0]);
        assert_eq!(tmpl.row_upper, vec![6.0, 14.0]);

        assert_eq!(tmpl.n_state, 1);
    }

    #[test]
    fn test_solver_error_display_all_branches() {
        let cases = vec![
            ("Infeasible", SolverError::Infeasible, "infeasible"),
            ("Unbounded", SolverError::Unbounded, "unbounded"),
            (
                "NumericalDifficulty",
                SolverError::NumericalDifficulty {
                    message: "singular matrix".to_string(),
                },
                "singular matrix",
            ),
            (
                "TimeLimitExceeded",
                SolverError::TimeLimitExceeded {
                    elapsed_seconds: 60.0,
                },
                "60.000s",
            ),
            (
                "IterationLimit",
                SolverError::IterationLimit { iterations: 10_000 },
                "10000 iterations",
            ),
            (
                "InternalError/None",
                SolverError::InternalError {
                    message: "unknown failure".to_string(),
                    error_code: None,
                },
                "unknown failure",
            ),
            (
                "InternalError/Some",
                SolverError::InternalError {
                    message: "segfault in HiGHS".to_string(),
                    error_code: Some(-1),
                },
                "code -1",
            ),
            (
                "BasisInconsistent",
                SolverError::BasisInconsistent {
                    num_row: 2,
                    total_basic: 5,
                    col_basic: 3,
                    row_basic: 2,
                },
                "num_row=2",
            ),
            (
                "BasisRowCountMismatch",
                SolverError::BasisRowCountMismatch {
                    lp_rows: 3,
                    basis_rows: 2,
                },
                "lp_rows=3",
            ),
        ];

        for (name, err, expected_text) in cases {
            let msg = format!("{err}");
            assert!(!msg.is_empty());
            assert!(
                msg.contains(expected_text),
                "{name} missing '{expected_text}'"
            );
        }
    }

    #[test]
    fn test_solver_error_is_std_error_all_variants() {
        let errors: Vec<SolverError> = vec![
            SolverError::Infeasible,
            SolverError::Unbounded,
            SolverError::NumericalDifficulty {
                message: "test".to_string(),
            },
            SolverError::TimeLimitExceeded {
                elapsed_seconds: 1.0,
            },
            SolverError::IterationLimit { iterations: 1 },
            SolverError::InternalError {
                message: "test".to_string(),
                error_code: None,
            },
            SolverError::InternalError {
                message: "test".to_string(),
                error_code: Some(-1),
            },
            SolverError::BasisInconsistent {
                num_row: 2,
                total_basic: 5,
                col_basic: 3,
                row_basic: 2,
            },
            SolverError::BasisRowCountMismatch {
                lp_rows: 3,
                basis_rows: 2,
            },
        ];

        for err in &errors {
            let _: &dyn std::error::Error = err;
        }
    }

    #[test]
    fn test_solution_view_to_owned() {
        let primal = [1.0, 2.0];
        let dual = [3.0];
        let rc = [4.0, 5.0];
        let view = SolutionView {
            objective: 42.0,
            primal: &primal,
            dual: &dual,
            reduced_costs: &rc,
            iterations: 7,
            solve_time_seconds: 0.5,
        };
        let owned = view.to_owned();
        assert_eq!(owned.objective, 42.0);
        assert_eq!(owned.primal, vec![1.0, 2.0]);
        assert_eq!(owned.dual, vec![3.0]);
        assert_eq!(owned.reduced_costs, vec![4.0, 5.0]);
        assert_eq!(owned.iterations, 7);
        assert_eq!(owned.solve_time_seconds, 0.5);
    }

    #[test]
    fn test_solution_view_is_copy() {
        let primal = [1.0];
        let dual = [2.0];
        let rc = [3.0];
        let view = SolutionView {
            objective: 0.0,
            primal: &primal,
            dual: &dual,
            reduced_costs: &rc,
            iterations: 0,
            solve_time_seconds: 0.0,
        };
        let copy = view;
        assert_eq!(view.objective, copy.objective);
    }

    #[test]
    fn test_row_batch_construction() {
        let batch = RowBatch {
            num_rows: 2,
            row_starts: vec![0_i32, 2, 4],
            col_indices: vec![0_i32, 1, 0, 1],
            values: vec![-5.0, 1.0, 3.0, 1.0],
            row_lower: vec![20.0, 80.0],
            row_upper: vec![f64::INFINITY, f64::INFINITY],
        };

        assert_eq!(batch.num_rows, 2);
        assert_eq!(batch.row_starts.len(), 3);
        assert_eq!(batch.row_starts, vec![0_i32, 2, 4]);
        assert_eq!(batch.col_indices, vec![0_i32, 1, 0, 1]);
        assert_eq!(batch.values, vec![-5.0, 1.0, 3.0, 1.0]);
        assert_eq!(batch.row_lower, vec![20.0, 80.0]);
        assert!(batch.row_upper[0].is_infinite() && batch.row_upper[0] > 0.0);
        assert!(batch.row_upper[1].is_infinite() && batch.row_upper[1] > 0.0);
    }
}
