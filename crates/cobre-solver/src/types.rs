//! Core types for the solver abstraction layer.
//!
//! Defines the canonical representations of LP solutions, basis management,
//! and terminal solver errors used throughout the solver interface.

use core::fmt;

/// Simplex basis storing solver-native `i32` status codes for zero-copy round-trip
/// basis management.
///
/// `Basis` stores the raw solver `i32` status codes directly, enabling zero-copy
/// round-trip warm-starting via `copy_from_slice` (memcpy). This avoids per-element
/// translation overhead when the caller only needs to save and restore the basis
/// without inspecting individual statuses.
///
/// `HiGHS` uses `HighsInt` (4 bytes) for status codes; CLP uses `unsigned char`
/// (1 byte, widened to `i32` in this representation). The caller is responsible
/// for matching the buffer dimensions to the LP model before use.
///
/// See Solver Abstraction SS9.
#[derive(Debug, Clone)]
pub struct Basis {
    /// Solver-native `i32` status codes for each column (length must equal `num_cols`).
    pub col_status: Vec<i32>,

    /// Solver-native `i32` status codes for each row, including structural and dynamic rows.
    pub row_status: Vec<i32>,
}

impl Basis {
    /// Creates a new `Basis` with pre-allocated, zero-filled status code buffers.
    ///
    /// Both `col_status` and `row_status` are allocated to the requested lengths
    /// and filled with `0_i32`. The caller reuses this buffer across solves by
    /// passing it to [`crate::SolverInterface::get_basis`] on each iteration.
    #[must_use]
    pub fn new(num_cols: usize, num_rows: usize) -> Self {
        Self {
            col_status: vec![0_i32; num_cols],
            row_status: vec![0_i32; num_rows],
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
    ///
    /// Use this when the solution data must outlive the current solver borrow,
    /// or when the same solution will be read after a subsequent solver call.
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
/// `reset()` does **not** zero statistics counters. They persist across
/// model reloads for the lifetime of the solver instance.
///
/// See [Solver Interface Trait SS4.3](../../../cobre-docs/src/specs/architecture/solver-interface-trait.md).
#[derive(Debug, Clone, Default)]
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

    /// Number of times `solve_with_basis` fell back to cold-start due to basis rejection.
    pub basis_rejections: u64,
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

    /// Number of state values transferred between consecutive stages.
    ///
    /// Equal to `N * L` per
    /// [Solver Abstraction SS2.1](../../../cobre-docs/src/specs/architecture/solver-abstraction.md).
    /// This is the storage volumes plus all AR lags except the oldest
    /// (which ages out of the lag window).
    pub n_transfer: usize,

    /// Number of dual-relevant constraint rows (contiguous prefix of rows).
    ///
    /// Equal to `N + N*L + n_fpha + n_gvc` per
    /// [Solver Abstraction SS2.2](../../../cobre-docs/src/specs/architecture/solver-abstraction.md).
    /// For constant-productivity-only hydros (no FPHA), this equals `n_state`.
    /// Extracting cut coefficients reads `dual[0..n_dual_relevant]`.
    pub n_dual_relevant: usize,

    /// Number of operating hydros at this stage.
    pub n_hydro: usize,

    /// Maximum PAR order across all operating hydros at this stage.
    ///
    /// Determines the uniform lag stride: all hydros store `max_par_order`
    /// lag values regardless of their individual PAR order, enabling SIMD
    /// vectorization with a single contiguous state stride.
    pub max_par_order: usize,
}

/// Batch of constraint rows for addition to a loaded LP, in CSR (row-major) form.
///
/// Assembled from the cut pool activity bitmap before each LP rebuild
/// and passed to [`crate::SolverInterface::add_rows`] for a single batch call.
/// Cuts are appended at the bottom of the constraint matrix in the dynamic
/// constraint region per
/// [Solver Abstraction SS2.2](../../../cobre-docs/src/specs/architecture/solver-abstraction.md).
///
/// See [Solver Interface Trait SS4.5](../../../cobre-docs/src/specs/architecture/solver-interface-trait.md)
/// and the cut pool assembly protocol in
/// [Solver Abstraction SS5.4](../../../cobre-docs/src/specs/architecture/solver-abstraction.md).
#[derive(Debug, Clone)]
pub struct RowBatch {
    /// Number of active constraint rows (cuts) in this batch.
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

    /// Row lower bounds (cut intercepts alpha for Benders cuts).
    ///
    /// Length: `num_rows`. For `>=` cuts, this is the RHS lower bound.
    pub row_lower: Vec<f64>,

    /// Row upper bounds.
    ///
    /// Length: `num_rows`. Use `f64::INFINITY` for `>=` cuts (Benders cuts
    /// have no finite upper bound).
    pub row_upper: Vec<f64>,
}

/// Terminal LP solve error returned after all retry attempts are exhausted.
///
/// The calling algorithm uses the variant to determine its response:
/// hard stop (`Infeasible`, `Unbounded`, `InternalError`) or log and
/// proceed (`NumericalDifficulty`, `TimeLimitExceeded`, `IterationLimit`).
///
/// The six variants correspond to the error categories defined in
/// Solver Abstraction SS6. Solver-internal errors (e.g., factorization
/// failures) are resolved by retry logic before reaching this level.
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
            } => {
                if let Some(code) = error_code {
                    write!(f, "internal solver error (code {code}): {message}")
                } else {
                    write!(f, "internal solver error: {message}")
                }
            }
        }
    }
}

impl std::error::Error for SolverError {}

#[cfg(test)]
mod tests {
    use super::{Basis, RowBatch, SolutionView, SolverError, SolverStatistics, StageTemplate};

    #[test]
    fn test_basis_new_dimensions_and_zero_fill() {
        let rb = Basis::new(3, 2);
        assert_eq!(rb.col_status.len(), 3);
        assert_eq!(rb.row_status.len(), 2);
        assert!(rb.col_status.iter().all(|&v| v == 0_i32));
        assert!(rb.row_status.iter().all(|&v| v == 0_i32));
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
        cloned2.col_status[0] = 1_i32;
        assert_eq!(rb.col_status[0], 0_i32);
    }

    #[test]
    fn test_solver_error_display_infeasible() {
        let msg = format!("{}", SolverError::Infeasible);
        assert!(msg.contains("infeasible"));
    }

    #[test]
    fn test_solver_error_display_all_variants() {
        let variants = vec![
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
        assert_eq!(stats.basis_rejections, 0);
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
            n_transfer: 0,
            n_dual_relevant: 1,
            n_hydro: 1,
            max_par_order: 0,
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
        assert_eq!(tmpl.n_transfer, 0);
        assert_eq!(tmpl.n_dual_relevant, 1);
        assert_eq!(tmpl.n_hydro, 1);
        assert_eq!(tmpl.max_par_order, 0);
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
