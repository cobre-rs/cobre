//! Core types for the solver abstraction layer.
//!
//! Defines the canonical representations of basis status codes, LP solutions,
//! and terminal solver errors used throughout the solver interface.

use core::fmt;

/// Simplex basis status for a single variable or constraint.
///
/// Maps to solver-specific integer codes:
/// `HiGHS` uses `HighsInt` (4 bytes), CLP uses `unsigned char` (1 byte).
/// The implementation translates between this canonical representation
/// and the solver-specific encoding when calling [`crate::SolverInterface::get_basis`]
/// or [`crate::SolverInterface::solve_with_basis`].
///
/// See [Solver Abstraction SS9](../../../cobre-docs/src/specs/architecture/solver-abstraction.md).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BasisStatus {
    /// Variable or constraint is at its lower bound (non-basic).
    AtLower,
    /// Variable or constraint is in the basis (between its bounds).
    Basic,
    /// Variable or constraint is at its upper bound (non-basic).
    AtUpper,
    /// Variable or constraint is free (superbasic, not at any bound).
    Free,
    /// Variable or constraint is fixed (lower bound equals upper bound).
    Fixed,
}

/// Simplex basis for warm-starting a subsequent LP solve.
///
/// Stores one [`BasisStatus`] per column (variable) and one per row
/// (constraint), capturing the full simplex basis state. The basis is stored
/// in the original problem space (not presolved) to ensure portability across
/// solver versions and presolve strategies.
///
/// When warm-starting after adding new dynamic constraint rows, the static
/// portion of the basis (`[0, n_static)`) is reused directly and new rows are
/// initialized as [`BasisStatus::Basic`]. See Solver Abstraction SS2.3 and SS9.
#[derive(Debug, Clone)]
pub struct Basis {
    /// Basis status for each column (decision variable).
    ///
    /// Length must equal the number of LP columns (`num_cols`). State
    /// variables occupy the contiguous prefix `[0, n_state)` per
    /// Solver Abstraction SS2.1.
    pub col_status: Vec<BasisStatus>,

    /// Basis status for each row (constraint), including both static
    /// structural rows and dynamic constraint rows (e.g., Benders cuts).
    ///
    /// Length must equal the total number of LP rows at the time the basis
    /// was extracted. Dynamic constraint rows appended via
    /// [`crate::SolverInterface::add_rows`] are included at the end.
    pub row_status: Vec<BasisStatus>,
}

/// Complete solution from a successful LP solve.
///
/// All values are in the original (unscaled) problem space. Dual values
/// are pre-normalized to the canonical sign convention defined in
/// [Solver Abstraction SS8](../../../cobre-docs/src/specs/architecture/solver-abstraction.md)
/// before this struct is returned -- solver-specific sign differences are
/// resolved within the [`crate::SolverInterface`] implementation.
///
/// This struct is also embedded in [`SolverError`] variants that may carry a
/// partial solution when the solve terminates prematurely.
///
/// See [Solver Interface Trait SS4.1](../../../cobre-docs/src/specs/architecture/solver-interface-trait.md).
#[derive(Debug, Clone)]
pub struct LpSolution {
    /// Optimal objective value (minimization sense).
    pub objective: f64,

    /// Primal variable values, indexed by column.
    ///
    /// Length equals `num_cols`. State variables occupy the contiguous prefix
    /// `[0, n_state)` per
    /// [Solver Abstraction SS2.1](../../../cobre-docs/src/specs/architecture/solver-abstraction.md).
    /// Transferring state from stage $t$ to stage $t+1$ copies
    /// `primal[0..n_transfer]`.
    pub primal: Vec<f64>,

    /// Dual multipliers (shadow prices), indexed by row.
    ///
    /// Length equals `num_rows` (structural rows + appended dynamic constraint
    /// rows). Cut-relevant constraint duals occupy the contiguous prefix
    /// `[0, n_dual_relevant)` per
    /// [Solver Abstraction SS2.2](../../../cobre-docs/src/specs/architecture/solver-abstraction.md).
    /// Extracting cut coefficients is a single contiguous slice read:
    /// `cut_coefficients = dual[0..n_dual_relevant]`.
    ///
    /// Sign convention: normalized to the canonical convention (positive dual
    /// on a `<=` constraint means increasing RHS increases the objective)
    /// before returning.
    pub dual: Vec<f64>,

    /// Reduced costs, indexed by column.
    ///
    /// Length equals `num_cols`.
    pub reduced_costs: Vec<f64>,

    /// Number of simplex iterations performed for this solve.
    pub iterations: u64,

    /// Wall-clock solve time in seconds (excluding retry overhead).
    pub solve_time_seconds: f64,
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
    ///
    /// One retry attempt corresponds to one escalation step in the retry
    /// sequence (e.g., clear basis, disable presolve, switch algorithm).
    pub retry_count: u64,

    /// Cumulative wall-clock time spent in solver calls, in seconds.
    ///
    /// Excludes retry overhead between attempts; includes time for all
    /// individual solve invocations.
    pub total_solve_time_seconds: f64,

    /// Number of times `solve_with_basis` fell back to cold-start because the
    /// solver rejected the supplied basis.
    ///
    /// A non-zero value indicates that warm-starting failed for some solves,
    /// which has a direct performance impact. Investigate if this counter
    /// grows — common causes are singular basis matrices or factorization
    /// failures after LP structure changes.
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

    /// Number of static rows (structural constraints, excluding dynamic
    /// constraints added at runtime via `add_rows`).
    pub num_rows: usize,

    /// Number of non-zero entries in the structural constraint matrix.
    pub num_nz: usize,

    /// CSC column start offsets.
    ///
    /// Length: `num_cols + 1`. Entry `col_starts[j]` is the index into
    /// `row_indices` and `values` where column `j` begins.
    /// `col_starts[num_cols]` equals `num_nz`.
    pub col_starts: Vec<usize>,

    /// CSC row indices for each non-zero entry.
    ///
    /// Length: `num_nz`. Entry `row_indices[k]` is the row of the `k`-th
    /// non-zero value.
    pub row_indices: Vec<usize>,

    /// CSC non-zero values.
    ///
    /// Length: `num_nz`. Entry `values[k]` is the coefficient of the `k`-th
    /// non-zero, at row `row_indices[k]` in its column.
    pub values: Vec<f64>,

    /// Column lower bounds.
    ///
    /// Length: `num_cols`. Use `f64::NEG_INFINITY` for unbounded below.
    pub col_lower: Vec<f64>,

    /// Column upper bounds.
    ///
    /// Length: `num_cols`. Use `f64::INFINITY` for unbounded above.
    pub col_upper: Vec<f64>,

    /// Objective coefficients (minimization sense).
    ///
    /// Length: `num_cols`.
    pub objective: Vec<f64>,

    /// Row lower bounds.
    ///
    /// Length: `num_rows`. For equality constraints, set equal to `row_upper`.
    /// Use `f64::NEG_INFINITY` for constraints with no lower bound.
    pub row_lower: Vec<f64>,

    /// Row upper bounds.
    ///
    /// Length: `num_rows`. For equality constraints, set equal to `row_lower`.
    /// Use `f64::INFINITY` for constraints with no upper bound.
    pub row_upper: Vec<f64>,

    /// Number of state variables (contiguous prefix of columns).
    ///
    /// Equal to `N * (1 + L)` where `N` is the number of operating hydros
    /// and `L` is the maximum PAR order across all operating hydros, per
    /// [Solver Abstraction SS2.1](../../../cobre-docs/src/specs/architecture/solver-abstraction.md).
    /// State transfer copies `primal[0..n_transfer]`.
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

    /// CSR row start offsets.
    ///
    /// Length: `num_rows + 1`. Entry `row_starts[i]` is the index into
    /// `col_indices` and `values` where row `i` begins.
    /// `row_starts[num_rows]` equals the total number of non-zeros.
    pub row_starts: Vec<usize>,

    /// CSR column indices for each non-zero entry.
    ///
    /// Length: total non-zeros across all rows. Entry `col_indices[k]` is the
    /// column of the `k`-th non-zero value.
    pub col_indices: Vec<usize>,

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
/// hard stop (`Infeasible`, `Unbounded`, `InternalError`) or proceed with
/// degraded quality (`NumericalDifficulty`, `TimeLimitExceeded`,
/// `IterationLimit`) when a partial solution is available.
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
    Infeasible {
        /// Infeasibility ray (proof of infeasibility), if available from the
        /// solver. Not all solver backends provide this.
        ray: Option<Vec<f64>>,
    },

    /// The LP objective is unbounded below.
    ///
    /// Indicates a modeling error (missing bounds, incorrect objective sign).
    /// The calling algorithm should perform a hard stop.
    Unbounded {
        /// Unbounded direction certificate, if available from the solver.
        /// Not all solver backends provide this.
        direction: Option<Vec<f64>>,
    },

    /// Solver encountered numerical difficulties that persisted through all
    /// retry attempts.
    ///
    /// May have a partial (non-optimal) solution. The calling algorithm may
    /// log a warning and proceed if the partial solution is usable.
    NumericalDifficulty {
        /// Best solution found before the numerical difficulty, if any.
        partial_solution: Option<LpSolution>,
        /// Human-readable description of the numerical issue from the solver.
        message: String,
    },

    /// Per-solve wall-clock time budget exhausted.
    ///
    /// May have a partial solution from the best iteration reached within the
    /// budget.
    TimeLimitExceeded {
        /// Best solution found within the time budget, if any.
        partial_solution: Option<LpSolution>,
        /// Elapsed wall-clock time in seconds at the point of termination.
        elapsed_seconds: f64,
    },

    /// Solver simplex iteration limit reached.
    ///
    /// May have a partial solution from the last completed iteration.
    IterationLimit {
        /// Best solution found within the iteration budget, if any.
        partial_solution: Option<LpSolution>,
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
            Self::Infeasible { ray } => {
                if ray.is_some() {
                    write!(f, "LP is infeasible (infeasibility ray available)")
                } else {
                    write!(f, "LP is infeasible")
                }
            }
            Self::Unbounded { direction } => {
                if direction.is_some() {
                    write!(f, "LP is unbounded (unbounded direction available)")
                } else {
                    write!(f, "LP is unbounded")
                }
            }
            Self::NumericalDifficulty {
                partial_solution,
                message,
            } => {
                if partial_solution.is_some() {
                    write!(
                        f,
                        "numerical difficulty (partial solution available): {message}"
                    )
                } else {
                    write!(f, "numerical difficulty (no partial solution): {message}")
                }
            }
            Self::TimeLimitExceeded {
                partial_solution,
                elapsed_seconds,
            } => {
                if partial_solution.is_some() {
                    write!(
                        f,
                        "time limit exceeded after {elapsed_seconds:.3}s (partial solution available)"
                    )
                } else {
                    write!(
                        f,
                        "time limit exceeded after {elapsed_seconds:.3}s (no partial solution)"
                    )
                }
            }
            Self::IterationLimit {
                partial_solution,
                iterations,
            } => {
                if partial_solution.is_some() {
                    write!(
                        f,
                        "iteration limit reached after {iterations} iterations (partial solution available)"
                    )
                } else {
                    write!(
                        f,
                        "iteration limit reached after {iterations} iterations (no partial solution)"
                    )
                }
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
    use super::{BasisStatus, LpSolution, RowBatch, SolverError, SolverStatistics, StageTemplate};

    #[test]
    fn test_basis_status_clone_copy() {
        let original = BasisStatus::AtLower;
        let copied = original;
        let vec_of_status = vec![original, BasisStatus::Basic];
        let cloned_vec = vec_of_status.clone();
        assert_eq!(original, copied);
        assert_eq!(vec_of_status, cloned_vec);
        // Verify variants are comparable
        assert_ne!(BasisStatus::AtLower, BasisStatus::Basic);
        assert_ne!(BasisStatus::AtUpper, BasisStatus::Free);
        assert_ne!(BasisStatus::Fixed, BasisStatus::Basic);
    }

    #[test]
    fn test_solver_error_display_infeasible() {
        let err = SolverError::Infeasible { ray: None };
        let msg = format!("{err}");
        assert!(!msg.is_empty(), "display must be non-empty");
        assert!(
            msg.contains("infeasible"),
            "display must mention infeasible: {msg}"
        );
    }

    #[test]
    fn test_solver_error_display_all_variants() {
        let partial = Some(LpSolution {
            objective: 0.0,
            primal: vec![],
            dual: vec![],
            reduced_costs: vec![],
            iterations: 0,
            solve_time_seconds: 0.0,
        });

        let variants: Vec<(&str, SolverError)> = vec![
            (
                "Infeasible",
                SolverError::Infeasible {
                    ray: Some(vec![1.0, 0.0]),
                },
            ),
            (
                "Unbounded",
                SolverError::Unbounded {
                    direction: Some(vec![0.0, 1.0]),
                },
            ),
            (
                "NumericalDifficulty",
                SolverError::NumericalDifficulty {
                    partial_solution: partial,
                    message: "factorization failed".to_string(),
                },
            ),
            (
                "TimeLimitExceeded",
                SolverError::TimeLimitExceeded {
                    partial_solution: None,
                    elapsed_seconds: 60.0,
                },
            ),
            (
                "IterationLimit",
                SolverError::IterationLimit {
                    partial_solution: None,
                    iterations: 10_000,
                },
            ),
            (
                "InternalError",
                SolverError::InternalError {
                    message: "segfault in HiGHS".to_string(),
                    error_code: Some(-1),
                },
            ),
        ];

        let mut messages: Vec<String> = Vec::new();
        for (name, err) in &variants {
            let msg = format!("{err}");
            assert!(!msg.is_empty(), "display for {name} must be non-empty");
            messages.push(msg);
        }

        // Verify all messages are distinct
        for i in 0..messages.len() {
            for j in (i + 1)..messages.len() {
                assert_ne!(
                    messages[i], messages[j],
                    "display for variant {} and {} must be distinct",
                    variants[i].0, variants[j].0
                );
            }
        }
    }

    #[test]
    fn test_solver_error_is_std_error() {
        let err = SolverError::InternalError {
            message: "test".to_string(),
            error_code: None,
        };
        // Verify SolverError can be used as &dyn std::error::Error
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
    }

    // Shared fixture from Solver Interface Testing SS1.1:
    // 3 variables, 2 structural constraints, 3 non-zeros.
    //
    //   min  0*x0 + 1*x1 + 50*x2
    //   s.t. x0            = 6   (state-fixing)
    //        2*x0 + x2     = 14  (power balance)
    //   x0 in [0, 10], x1 in [0, +inf), x2 in [0, 8]
    //
    // CSC matrix A = [[1, 0, 0], [2, 0, 1]]:
    //   col_starts  = [0, 2, 2, 3]
    //   row_indices = [0, 1, 1]
    //   values      = [1.0, 2.0, 1.0]
    fn make_fixture_stage_template() -> StageTemplate {
        StageTemplate {
            num_cols: 3,
            num_rows: 2,
            num_nz: 3,
            col_starts: vec![0, 2, 2, 3],
            row_indices: vec![0, 1, 1],
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
        assert_eq!(tmpl.col_starts, vec![0, 2, 2, 3]);
        assert_eq!(tmpl.row_indices, vec![0, 1, 1]);
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
    fn test_row_batch_construction() {
        // Benders cut fixture from Solver Interface Testing SS1.2:
        // Cut 1: -5*x0 + x1 >= 20  (col_indices [0,1], values [-5, 1])
        // Cut 2:  3*x0 + x1 >= 80  (col_indices [0,1], values [ 3, 1])
        let batch = RowBatch {
            num_rows: 2,
            row_starts: vec![0, 2, 4],
            col_indices: vec![0, 1, 0, 1],
            values: vec![-5.0, 1.0, 3.0, 1.0],
            row_lower: vec![20.0, 80.0],
            row_upper: vec![f64::INFINITY, f64::INFINITY],
        };

        assert_eq!(batch.num_rows, 2);
        assert_eq!(batch.row_starts.len(), 3);
        assert_eq!(batch.row_starts, vec![0, 2, 4]);
        assert_eq!(batch.col_indices, vec![0, 1, 0, 1]);
        assert_eq!(batch.values, vec![-5.0, 1.0, 3.0, 1.0]);
        assert_eq!(batch.row_lower, vec![20.0, 80.0]);
        assert!(batch.row_upper[0].is_infinite() && batch.row_upper[0] > 0.0);
        assert!(batch.row_upper[1].is_infinite() && batch.row_upper[1] > 0.0);
    }
}
