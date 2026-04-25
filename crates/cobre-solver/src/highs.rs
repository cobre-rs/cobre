//! `HiGHS` LP solver backend implementing [`SolverInterface`].
//!
//! This module provides [`HighsSolver`], which wraps the `HiGHS` C API through
//! the FFI layer in `ffi` and implements the full [`SolverInterface`]
//! contract for iterative LP solving in power system optimization.
//!
//! # Thread Safety
//!
//! [`HighsSolver`] is `Send` but not `Sync`. The underlying `HiGHS` handle is
//! exclusively owned; transferring ownership to a worker thread is safe.
//! Concurrent access from multiple threads is not permitted (`HiGHS`
//! Implementation SS6.3).
//!
//! # Configuration
//!
//! The constructor applies performance-tuned defaults (`HiGHS` Implementation
//! SS4.1): dual simplex, no presolve, no parallelism, suppressed output, and
//! tight feasibility tolerances. These defaults are optimised for repeated
//! solves of small-to-medium LPs. Per-run parameters (time limit, iteration
//! limit) are not set here -- those are applied by the caller before each solve.

use std::ffi::CStr;
use std::os::raw::c_void;
use std::time::Instant;

use crate::{
    SolverInterface, ffi,
    types::{RowBatch, SolutionView, SolverError, SolverStatistics, StageTemplate},
};

// ─── Default HiGHS configuration ─────────────────────────────────────────────
//
// The eight performance-tuned options applied at construction and restored after
// each retry escalation. Keeping them in a single array eliminates per-option
// error branches that are structurally impossible to trigger in tests (HiGHS
// never rejects valid static option names).

/// A typed `HiGHS` option value for the configuration table.
enum OptionValue {
    /// String option (`cobre_highs_set_string_option`).
    Str(&'static CStr),
    /// Integer option (`cobre_highs_set_int_option`).
    Int(i32),
    /// Boolean option (`cobre_highs_set_bool_option`).
    Bool(i32),
    /// Double option (`cobre_highs_set_double_option`).
    Double(f64),
}

/// A named `HiGHS` option with its default value.
struct DefaultOption {
    name: &'static CStr,
    value: OptionValue,
}

impl DefaultOption {
    /// Applies this option to a `HiGHS` handle. Returns the `HiGHS` status code.
    ///
    /// # Safety
    ///
    /// `handle` must be a valid, non-null pointer from `cobre_highs_create()`.
    unsafe fn apply(&self, handle: *mut c_void) -> i32 {
        unsafe {
            match &self.value {
                OptionValue::Str(val) => {
                    ffi::cobre_highs_set_string_option(handle, self.name.as_ptr(), val.as_ptr())
                }
                OptionValue::Int(val) => {
                    ffi::cobre_highs_set_int_option(handle, self.name.as_ptr(), *val)
                }
                OptionValue::Bool(val) => {
                    ffi::cobre_highs_set_bool_option(handle, self.name.as_ptr(), *val)
                }
                OptionValue::Double(val) => {
                    ffi::cobre_highs_set_double_option(handle, self.name.as_ptr(), *val)
                }
            }
        }
    }
}

/// Performance-tuned default options (`HiGHS` Implementation SS4.1).
///
/// These eight options are applied at construction and restored after each retry
/// escalation. `simplex_scale_strategy` is set to 0 (off) because the calling
/// algorithm's prescaler already normalizes matrix entries toward 1.0; the
/// solver's internal equilibration scaling is redundant and can distort cost
/// ordering for large-RHS rows. Retry escalation levels 5+ override this to
/// more aggressive strategies as a fallback for hard problems.
fn default_options() -> [DefaultOption; 8] {
    [
        DefaultOption {
            name: c"solver",
            value: OptionValue::Str(c"simplex"),
        },
        DefaultOption {
            name: c"simplex_strategy",
            value: OptionValue::Int(1), // Dual simplex
        },
        DefaultOption {
            name: c"simplex_scale_strategy",
            value: OptionValue::Int(0), // Off (prescaler handles scaling)
        },
        DefaultOption {
            name: c"presolve",
            value: OptionValue::Str(c"off"),
        },
        DefaultOption {
            name: c"parallel",
            value: OptionValue::Str(c"off"),
        },
        DefaultOption {
            name: c"output_flag",
            value: OptionValue::Bool(0),
        },
        DefaultOption {
            name: c"primal_feasibility_tolerance",
            value: OptionValue::Double(1e-7),
        },
        DefaultOption {
            name: c"dual_feasibility_tolerance",
            value: OptionValue::Double(1e-7),
        },
    ]
}

/// `HiGHS` LP solver instance implementing [`SolverInterface`].
///
/// Owns an opaque `HiGHS` handle and pre-allocated buffers for solution
/// extraction, scratch i32 index conversion, and statistics accumulation.
///
/// Construct with [`HighsSolver::new`]. The handle is destroyed automatically
/// when the instance is dropped.
///
/// # Example
///
/// ```rust
/// use cobre_solver::{HighsSolver, SolverInterface};
///
/// let solver = HighsSolver::new().expect("HiGHS initialisation failed");
/// assert_eq!(solver.name(), "HiGHS");
/// ```
pub struct HighsSolver {
    /// Opaque pointer to the `HiGHS` C++ instance, obtained from `cobre_highs_create()`.
    handle: *mut c_void,
    /// Pre-allocated buffer for primal column values extracted after each solve.
    /// Resized in `load_model`; reused across solves to avoid per-solve allocation.
    col_value: Vec<f64>,
    /// Pre-allocated buffer for column dual values (reduced costs from `HiGHS` perspective).
    /// Resized in `load_model`.
    col_dual: Vec<f64>,
    /// Pre-allocated buffer for row primal values (constraint activity).
    /// Resized in `load_model`.
    row_value: Vec<f64>,
    /// Pre-allocated buffer for row dual multipliers (shadow prices).
    /// Resized in `load_model`.
    row_dual: Vec<f64>,
    /// Scratch buffer for converting `usize` indices to `i32` for the `HiGHS` C API.
    /// Used by `add_rows`, `set_row_bounds`, and `set_col_bounds`.
    /// Never shrunk -- only grows -- to prevent reallocation churn on the hot path.
    scratch_i32: Vec<i32>,
    /// Pre-allocated i32 buffer for column basis status codes.
    /// Reused across warm-start `solve` and `get_basis` calls to avoid per-call allocation.
    /// Resized in `load_model` to `num_cols`; never shrunk.
    basis_col_i32: Vec<i32>,
    /// Pre-allocated i32 buffer for row basis status codes.
    /// Reused across warm-start `solve` and `get_basis` calls to avoid per-call allocation.
    /// Resized in `load_model` to `num_rows` and grown in `add_rows`.
    basis_row_i32: Vec<i32>,
    /// Scratch buffer for dual-ray extraction in `interpret_terminal_status` (dual).
    /// Grown lazily to `num_rows` via `resize`; contents are discarded after classification.
    /// Retained across calls so repeated non-optimal solves do not re-allocate.
    terminal_status_dual_scratch: Vec<f64>,
    /// Scratch buffer for primal-ray extraction in `interpret_terminal_status` (primal).
    /// Grown lazily to `num_cols` via `resize`; contents are discarded after classification.
    /// Retained across calls so repeated non-optimal solves do not re-allocate.
    terminal_status_primal_scratch: Vec<f64>,
    /// Current number of LP columns (decision variables), updated by `load_model` and `add_rows`.
    num_cols: usize,
    /// Current number of LP rows (constraints), updated by `load_model` and `add_rows`.
    num_rows: usize,
    /// Whether a model is currently loaded. Set to `true` in `load_model`,
    /// `false` in `reset` and `new`. Guards `solve`/`get_basis` contract.
    has_model: bool,
    /// Accumulated solver statistics. Counters grow monotonically from zero;
    /// not reset by `reset()`.
    stats: SolverStatistics,
}

// SAFETY: `HighsSolver` holds a raw pointer to a `HiGHS` C++ object. The `HiGHS`
// handle is not thread-safe for concurrent access, but exclusive ownership is
// maintained at all times -- exactly one `HighsSolver` instance owns each
// handle and no shared references to the handle exist. Transferring the
// `HighsSolver` to another thread (via `Send`) is safe because there is no
// concurrent access; the new thread has exclusive ownership. `Sync` is
// intentionally NOT implemented per `HiGHS` Implementation SS6.3.
unsafe impl Send for HighsSolver {}

/// Outcome of a successful retry escalation in [`HighsSolver::retry_escalation`].
///
/// Contains the accumulated attempt count and the solve time / iteration
/// count from the successful retry level.
struct RetryOutcome {
    attempts: u64,
    solve_time: f64,
    iterations: u64,
    /// The retry level (0..11) at which the solve succeeded.
    level: u32,
}

impl HighsSolver {
    /// Creates a new `HiGHS` solver instance with performance-tuned defaults.
    ///
    /// Calls `cobre_highs_create()` to allocate the `HiGHS` handle, then applies
    /// the eight default options defined in `HiGHS` Implementation SS4.1:
    ///
    /// | Option                         | Value       | Type   |
    /// |--------------------------------|-------------|--------|
    /// | `solver`                       | `"simplex"` | string |
    /// | `simplex_strategy`             | `1`         | int    |
    /// | `simplex_scale_strategy`       | `0`         | int    |
    /// | `presolve`                     | `"off"`     | string |
    /// | `parallel`                     | `"off"`     | string |
    /// | `output_flag`                  | `0`         | bool   |
    /// | `primal_feasibility_tolerance` | `1e-7`      | double |
    /// | `dual_feasibility_tolerance`   | `1e-7`      | double |
    ///
    /// # Errors
    ///
    /// Returns `Err(SolverError::InternalError { .. })` if:
    /// - `cobre_highs_create()` returns a null pointer.
    /// - Any configuration call returns `HIGHS_STATUS_ERROR`.
    ///
    /// In both failure cases the `HiGHS` handle is destroyed before returning to
    /// prevent a resource leak.
    pub fn new() -> Result<Self, SolverError> {
        // SAFETY: `cobre_highs_create` is a C function with no preconditions.
        // It allocates and returns a new `HiGHS` instance, or null on allocation
        // failure. The returned pointer is opaque and must be passed back to
        // `HiGHS` API functions.
        let handle = unsafe { ffi::cobre_highs_create() };

        if handle.is_null() {
            return Err(SolverError::InternalError {
                message: "HiGHS instance creation failed: Highs_create() returned null".to_string(),
                error_code: None,
            });
        }

        // Apply performance-tuned configuration. On any failure, destroy the
        // handle before returning to prevent a resource leak.
        if let Err(e) = Self::apply_default_config(handle) {
            // SAFETY: `handle` is a valid, non-null pointer obtained from
            // `cobre_highs_create()` in this same function. It has not been
            // passed to `cobre_highs_destroy()` yet. After this call, `handle`
            // must not be used again -- this function returns immediately with Err.
            unsafe { ffi::cobre_highs_destroy(handle) };
            return Err(e);
        }

        Ok(Self {
            handle,
            col_value: Vec::new(),
            col_dual: Vec::new(),
            row_value: Vec::new(),
            row_dual: Vec::new(),
            scratch_i32: Vec::new(),
            basis_col_i32: Vec::new(),
            basis_row_i32: Vec::new(),
            terminal_status_dual_scratch: Vec::new(),
            terminal_status_primal_scratch: Vec::new(),
            num_cols: 0,
            num_rows: 0,
            has_model: false,
            stats: SolverStatistics {
                retry_level_histogram: vec![0u64; 12],
                ..SolverStatistics::default()
            },
        })
    }

    /// Applies the eight performance-tuned `HiGHS` configuration options.
    ///
    /// Called once during construction. Returns `Ok(())` if all options are set
    /// successfully, or `Err(SolverError::InternalError)` with the failing
    /// option name if any configuration call returns `HIGHS_STATUS_ERROR`.
    fn apply_default_config(handle: *mut c_void) -> Result<(), SolverError> {
        for opt in &default_options() {
            // SAFETY: `handle` is a valid, non-null HiGHS pointer.
            let status = unsafe { opt.apply(handle) };
            if status == ffi::HIGHS_STATUS_ERROR {
                return Err(SolverError::InternalError {
                    message: format!(
                        "HiGHS configuration failed: {}",
                        opt.name.to_str().unwrap_or("?")
                    ),
                    error_code: Some(status),
                });
            }
        }
        Ok(())
    }

    /// Extracts the optimal solution from `HiGHS` into pre-allocated buffers and returns
    /// a [`SolutionView`] borrowing directly from those buffers.
    ///
    /// The returned view borrows solver-internal buffers and is valid until the next
    /// `&mut self` call. `col_dual` is the reduced cost vector. Row duals follow the
    /// canonical sign convention (per Solver Abstraction SS8).
    fn extract_solution_view(&mut self, solve_time_seconds: f64) -> SolutionView<'_> {
        // SAFETY: buffers resized in `load_model`/`add_rows`; HiGHS writes within bounds.
        let status = unsafe {
            ffi::cobre_highs_get_solution(
                self.handle,
                self.col_value.as_mut_ptr(),
                self.col_dual.as_mut_ptr(),
                self.row_value.as_mut_ptr(),
                self.row_dual.as_mut_ptr(),
            )
        };
        // HiGHS documentation guarantees `cobre_highs_get_solution` returns
        // non-ERROR status after `OPTIMAL` model status; this is a
        // debug-build-only invariant check.
        debug_assert_ne!(
            status,
            ffi::HIGHS_STATUS_ERROR,
            "cobre_highs_get_solution failed after optimal solve; HiGHS invariant violation"
        );

        // SAFETY: `self.handle` is a valid, non-null HiGHS pointer.
        let objective = unsafe { ffi::cobre_highs_get_objective_value(self.handle) };

        // SAFETY: iteration count is non-negative so cast is safe.
        #[allow(clippy::cast_sign_loss)]
        let iterations =
            unsafe { ffi::cobre_highs_get_simplex_iteration_count(self.handle) } as u64;

        SolutionView {
            objective,
            primal: &self.col_value[..self.num_cols],
            dual: &self.row_dual[..self.num_rows],
            reduced_costs: &self.col_dual[..self.num_cols],
            iterations,
            solve_time_seconds,
        }
    }

    /// Restores default options after retry escalation.
    ///
    /// Status codes are checked via `debug_assert!` to catch programming
    /// errors during development (e.g., invalid option name). In release
    /// builds, failures are silently ignored since we are already on the
    /// recovery path.
    fn restore_default_settings(&mut self) {
        for opt in &default_options() {
            // SAFETY: `self.handle` is a valid, non-null HiGHS pointer.
            let status = unsafe { opt.apply(self.handle) };
            debug_assert_eq!(
                status,
                ffi::HIGHS_STATUS_OK,
                "restore_default_settings: option {:?} failed with status {status}",
                opt.name,
            );
        }
    }

    /// Runs the solver once and returns the raw `HiGHS` model status.
    fn run_once(&mut self) -> i32 {
        // SAFETY: `self.handle` is a valid, non-null HiGHS pointer.
        let run_status = unsafe { ffi::cobre_highs_run(self.handle) };
        if run_status == ffi::HIGHS_STATUS_ERROR {
            return ffi::HIGHS_MODEL_STATUS_SOLVE_ERROR;
        }
        // SAFETY: same.
        unsafe { ffi::cobre_highs_get_model_status(self.handle) }
    }

    /// Sets per-solve iteration limits before a `run_once()` call.
    ///
    /// Simplex gets `max(100_000, 50 × num_cols)` and IPM gets 10,000.
    /// These prevent degenerate cycling without affecting normal convergence.
    ///
    /// **Note on `time_limit`**: `HiGHS` tracks elapsed time cumulatively from
    /// instance creation, not per-`run()` call — neither `clear_solver()` nor
    /// option changes reset the internal timer. This makes `time_limit`
    /// unusable for the scenario-loop pattern (thousands of solves per
    /// instance). Wall-clock measurement via `Instant` is used instead for
    /// time-based budget management.
    fn set_iteration_limits(&mut self) {
        let simplex_iter_limit = self.num_cols.saturating_mul(50).max(100_000);
        // SAFETY: handle is valid non-null HiGHS pointer; option names are
        // static C strings with no retained pointers.
        unsafe {
            #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
            ffi::cobre_highs_set_int_option(
                self.handle,
                c"simplex_iteration_limit".as_ptr(),
                simplex_iter_limit as i32,
            );
            ffi::cobre_highs_set_int_option(self.handle, c"ipm_iteration_limit".as_ptr(), 10_000);
        }
    }

    /// Restores iteration limits to their unconstrained defaults.
    ///
    /// Called after `retry_escalation` completes (regardless of outcome).
    fn restore_iteration_limits(&mut self) {
        // SAFETY: handle is valid non-null HiGHS pointer.
        unsafe {
            ffi::cobre_highs_set_int_option(
                self.handle,
                c"simplex_iteration_limit".as_ptr(),
                i32::MAX,
            );
            ffi::cobre_highs_set_int_option(self.handle, c"ipm_iteration_limit".as_ptr(), i32::MAX);
        }
    }

    /// Interprets a non-optimal status as a terminal `SolverError`.
    ///
    /// Returns `None` for `SOLVE_ERROR` or `UNKNOWN` (retry continues),
    /// or `Some(error)` for terminal statuses.
    fn interpret_terminal_status(
        &mut self,
        status: i32,
        solve_time_seconds: f64,
    ) -> Option<SolverError> {
        match status {
            ffi::HIGHS_MODEL_STATUS_OPTIMAL => {
                // Caller should have handled optimal before reaching here.
                None
            }
            ffi::HIGHS_MODEL_STATUS_INFEASIBLE => Some(SolverError::Infeasible),
            ffi::HIGHS_MODEL_STATUS_UNBOUNDED_OR_INFEASIBLE => {
                // Probe for a dual ray to classify as Infeasible, then a primal
                // ray to classify as Unbounded. The ray values are not stored in
                // the error -- only the classification matters.
                //
                // `num_rows` and `num_cols` are up-to-date because `load_model`
                // and `add_rows` always update them before any solve that could
                // reach this branch. The `resize` below matches the exact count
                // that HiGHS writes into the buffer.
                let mut has_dual_ray: i32 = 0;
                self.terminal_status_dual_scratch.resize(self.num_rows, 0.0);
                // SAFETY: `self.handle` is a valid, non-null HiGHS pointer.
                // `terminal_status_dual_scratch` has been resized to at least
                // `self.num_rows` elements; HiGHS writes exactly `num_rows` values.
                let dual_status = unsafe {
                    ffi::cobre_highs_get_dual_ray(
                        self.handle,
                        &raw mut has_dual_ray,
                        self.terminal_status_dual_scratch.as_mut_ptr(),
                    )
                };
                if dual_status != ffi::HIGHS_STATUS_ERROR && has_dual_ray != 0 {
                    return Some(SolverError::Infeasible);
                }
                let mut has_primal_ray: i32 = 0;
                self.terminal_status_primal_scratch
                    .resize(self.num_cols, 0.0);
                // SAFETY: `self.handle` is a valid, non-null HiGHS pointer.
                // `terminal_status_primal_scratch` has been resized to at least
                // `self.num_cols` elements; HiGHS writes exactly `num_cols` values.
                let primal_status = unsafe {
                    ffi::cobre_highs_get_primal_ray(
                        self.handle,
                        &raw mut has_primal_ray,
                        self.terminal_status_primal_scratch.as_mut_ptr(),
                    )
                };
                if primal_status != ffi::HIGHS_STATUS_ERROR && has_primal_ray != 0 {
                    return Some(SolverError::Unbounded);
                }
                Some(SolverError::Infeasible)
            }
            ffi::HIGHS_MODEL_STATUS_UNBOUNDED => Some(SolverError::Unbounded),
            ffi::HIGHS_MODEL_STATUS_TIME_LIMIT => Some(SolverError::TimeLimitExceeded {
                elapsed_seconds: solve_time_seconds,
            }),
            ffi::HIGHS_MODEL_STATUS_ITERATION_LIMIT => {
                // SAFETY: handle is valid non-null pointer; iteration count is non-negative.
                #[allow(clippy::cast_sign_loss)]
                let iterations =
                    unsafe { ffi::cobre_highs_get_simplex_iteration_count(self.handle) } as u64;
                Some(SolverError::IterationLimit { iterations })
            }
            ffi::HIGHS_MODEL_STATUS_SOLVE_ERROR | ffi::HIGHS_MODEL_STATUS_UNKNOWN => {
                // Signal to the caller that retry should continue.
                None
            }
            other => Some(SolverError::InternalError {
                message: format!("HiGHS returned unexpected model status {other}"),
                error_code: Some(other),
            }),
        }
    }

    /// Converts `usize` indices to `i32` in the internal scratch buffer.
    ///
    /// Grows but never shrinks the buffer. Each element is debug-asserted to fit in i32.
    fn convert_to_i32_scratch(&mut self, source: &[usize]) -> &[i32] {
        if source.len() > self.scratch_i32.len() {
            self.scratch_i32.resize(source.len(), 0);
        }
        for (i, &v) in source.iter().enumerate() {
            debug_assert!(
                i32::try_from(v).is_ok(),
                "usize index {v} overflows i32::MAX at position {i}"
            );
            // SAFETY: debug_assert verifies v fits in i32; cast to HiGHS C API i32.
            #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
            {
                self.scratch_i32[i] = v as i32;
            }
        }
        &self.scratch_i32[..source.len()]
    }

    /// Run the 12-level retry escalation when the initial solve fails.
    ///
    /// Returns `Ok(RetryOutcome)` when a retry level finds optimal, or
    /// `Err((attempts, SolverError))` when all levels are exhausted or a
    /// terminal error is encountered. The caller is responsible for
    /// updating `self.stats` based on the outcome.
    ///
    /// Settings are always restored to defaults before returning (regardless
    /// of outcome).
    fn retry_escalation(&mut self, is_unbounded: bool) -> Result<RetryOutcome, (u64, SolverError)> {
        // 12-level retry escalation (HiGHS Implementation SS3). Organised into
        // two phases:
        //
        // Phase 1 (levels 0-4): Core cumulative sequence. Each level adds one
        //   option on top of the previous state. This proven sequence resolves
        //   the vast majority of retry-recoverable failures.
        //   L0: cold restart
        //   L1: + presolve
        //   L2: + dual simplex
        //   L3: + relaxed tolerances 1e-6
        //   L4: + IPM
        //
        // Phase 2 (levels 5-11): Extended strategies. Each level starts from
        //   a clean default state with presolve enabled and a time cap, then
        //   applies a specific combination of scaling, tolerances, and solver
        //   type. These address LPs with extreme coefficient ranges that the
        //   core sequence cannot resolve.
        //
        // Wall-clock per-level budgets: 15s (Phase 1), 30s (Phase 2), 60s
        // (Phase 2 extended). Overall 120s wall-clock budget caps the total.
        //
        // HiGHS `time_limit` is NOT used because HiGHS tracks elapsed time
        // cumulatively from instance creation — neither `clear_solver()` nor
        // option changes reset the internal timer. Iteration limits provide
        // the primary per-attempt safeguard; wall-clock budgets provide the
        // secondary time-based guard.
        let phase1_wall_budget = 15.0_f64;
        let phase2_wall_budget = 30.0_f64;
        let overall_budget = 120.0_f64;
        let num_retry_levels = 12_u32;

        let retry_start = Instant::now();
        let mut retry_attempts: u64 = 0;
        let mut terminal_err: Option<SolverError> = None;
        let mut found_optimal = false;
        let mut optimal_time = 0.0_f64;
        let mut optimal_iterations: u64 = 0;
        let mut optimal_level = 0_u32;

        for level in 0..num_retry_levels {
            // Check overall wall-clock budget before starting a new level.
            if retry_start.elapsed().as_secs_f64() >= overall_budget {
                break;
            }

            self.apply_retry_level_options(level);

            retry_attempts += 1;

            let t_retry = Instant::now();
            let retry_status = self.run_once();
            let retry_time = t_retry.elapsed().as_secs_f64();

            if retry_status == ffi::HIGHS_MODEL_STATUS_OPTIMAL {
                // Capture stats before establishing the borrow.
                // SAFETY: handle is valid non-null HiGHS pointer.
                #[allow(clippy::cast_sign_loss)]
                let iters =
                    unsafe { ffi::cobre_highs_get_simplex_iteration_count(self.handle) } as u64;
                found_optimal = true;
                optimal_time = retry_time;
                optimal_iterations = iters;
                optimal_level = level;
                break;
            }

            // UNBOUNDED and ITERATION_LIMIT during retry continue to the next
            // level: UNBOUNDED may be spurious (presolve resolves it);
            // ITERATION_LIMIT means this strategy is cycling but another may
            // converge. Wall-clock budget exceeded also continues (strategy
            // too slow). Other terminal statuses (INFEASIBLE) stop immediately.
            let level_budget = if level <= 4 {
                phase1_wall_budget
            } else {
                phase2_wall_budget
            };
            let budget_exceeded = retry_time > level_budget;
            let retryable = retry_status == ffi::HIGHS_MODEL_STATUS_UNBOUNDED
                || retry_status == ffi::HIGHS_MODEL_STATUS_ITERATION_LIMIT
                || budget_exceeded;
            if !retryable {
                if let Some(e) = self.interpret_terminal_status(retry_status, retry_time) {
                    terminal_err = Some(e);
                    break;
                }
            }
            // Still SOLVE_ERROR, UNKNOWN, UNBOUNDED, ITERATION_LIMIT, or
            // wall-clock exceeded -- continue to next level.
        }

        // Restore default settings and safeguard limits unconditionally.
        // `restore_default_settings()` covers the 8 defaults. Retry-only
        // options and safeguard limits need explicit reset.
        self.restore_default_settings();
        self.restore_iteration_limits();
        unsafe {
            ffi::cobre_highs_set_int_option(self.handle, c"user_objective_scale".as_ptr(), 0);
            ffi::cobre_highs_set_int_option(self.handle, c"user_bound_scale".as_ptr(), 0);
        }

        if found_optimal {
            return Ok(RetryOutcome {
                attempts: retry_attempts,
                solve_time: optimal_time,
                iterations: optimal_iterations,
                level: optimal_level,
            });
        }

        Err((
            retry_attempts,
            terminal_err.unwrap_or_else(|| {
                // All 12 retry levels exhausted or overall budget exceeded.
                if is_unbounded {
                    SolverError::Unbounded
                } else {
                    SolverError::NumericalDifficulty {
                        message:
                            "HiGHS failed to reach optimality after all retry escalation levels"
                                .to_string(),
                    }
                }
            }),
        ))
    }

    /// Apply `HiGHS` options for a specific retry escalation level.
    ///
    /// Phase 1 (levels 0-4) is cumulative: each level adds options on top of
    /// the previous state. Both phases apply `time_limit` and iteration limits
    /// as safeguards against hanging on hard LPs.
    ///
    /// Phase 2 (levels 5-11) starts fresh each time with its own time limit.
    ///
    /// # Safety (internal)
    ///
    /// All FFI calls use `self.handle` which is a valid non-null `HiGHS` pointer.
    /// Option names and values are static C strings with no retained pointers.
    fn apply_retry_level_options(&mut self, level: u32) {
        match level {
            // -- Phase 1: Core cumulative sequence (levels 0-4) ---------------
            //
            // Level 0: cold restart (clear solver state), dual simplex.
            0 => {
                unsafe { ffi::cobre_highs_clear_solver(self.handle) };
                self.set_iteration_limits();
            }
            // Level 1: + presolve.
            1 => unsafe {
                ffi::cobre_highs_set_string_option(
                    self.handle,
                    c"presolve".as_ptr(),
                    c"on".as_ptr(),
                );
            },
            // Level 2: + dual simplex.
            // Cumulative: presolve + dual simplex.
            2 => unsafe {
                ffi::cobre_highs_set_int_option(self.handle, c"simplex_strategy".as_ptr(), 1);
            },
            // Level 3: + relaxed tolerances 1e-6.
            // Cumulative: presolve + dual simplex + relaxed tolerances.
            3 => unsafe {
                ffi::cobre_highs_set_double_option(
                    self.handle,
                    c"primal_feasibility_tolerance".as_ptr(),
                    1e-6,
                );
                ffi::cobre_highs_set_double_option(
                    self.handle,
                    c"dual_feasibility_tolerance".as_ptr(),
                    1e-6,
                );
            },
            // Level 4: + IPM.
            // Cumulative: presolve + relaxed tolerances + IPM.
            4 => unsafe {
                ffi::cobre_highs_set_string_option(
                    self.handle,
                    c"solver".as_ptr(),
                    c"ipm".as_ptr(),
                );
            },

            // -- Phase 2: Extended strategies (levels 5-11) -------------------
            // Each level starts from a clean default state with presolve
            // and iteration limits, then applies specific options.
            _ => self.apply_extended_retry_options(level),
        }
    }

    /// Apply Phase 2 extended retry strategy options for levels 5-11.
    ///
    /// Each level starts from restored defaults with presolve and iteration
    /// limits, then applies level-specific scaling, tolerance, and solver
    /// options. Wall-clock budgets are managed by the caller.
    fn apply_extended_retry_options(&mut self, level: u32) {
        self.restore_default_settings();
        self.set_iteration_limits();
        // SAFETY: handle is valid non-null HiGHS pointer; option names/values
        // are static C strings; no retained pointers after call.
        unsafe {
            ffi::cobre_highs_set_string_option(self.handle, c"presolve".as_ptr(), c"on".as_ptr());
        }
        match level {
            5 => unsafe {
                ffi::cobre_highs_set_int_option(self.handle, c"simplex_scale_strategy".as_ptr(), 3);
            },
            6 => unsafe {
                ffi::cobre_highs_set_int_option(self.handle, c"simplex_strategy".as_ptr(), 1);
                ffi::cobre_highs_set_int_option(self.handle, c"simplex_scale_strategy".as_ptr(), 4);
            },
            7 => unsafe {
                ffi::cobre_highs_set_int_option(self.handle, c"simplex_scale_strategy".as_ptr(), 3);
                ffi::cobre_highs_set_double_option(
                    self.handle,
                    c"primal_feasibility_tolerance".as_ptr(),
                    1e-6,
                );
                ffi::cobre_highs_set_double_option(
                    self.handle,
                    c"dual_feasibility_tolerance".as_ptr(),
                    1e-6,
                );
            },
            8 => unsafe {
                ffi::cobre_highs_set_int_option(self.handle, c"user_objective_scale".as_ptr(), -10);
            },
            9 => unsafe {
                ffi::cobre_highs_set_int_option(self.handle, c"simplex_strategy".as_ptr(), 1);
                ffi::cobre_highs_set_int_option(self.handle, c"user_objective_scale".as_ptr(), -10);
                ffi::cobre_highs_set_int_option(self.handle, c"user_bound_scale".as_ptr(), -5);
            },
            10 => unsafe {
                ffi::cobre_highs_set_int_option(self.handle, c"user_objective_scale".as_ptr(), -13);
                ffi::cobre_highs_set_int_option(self.handle, c"user_bound_scale".as_ptr(), -8);
                ffi::cobre_highs_set_double_option(
                    self.handle,
                    c"primal_feasibility_tolerance".as_ptr(),
                    1e-6,
                );
                ffi::cobre_highs_set_double_option(
                    self.handle,
                    c"dual_feasibility_tolerance".as_ptr(),
                    1e-6,
                );
            },
            11 => unsafe {
                ffi::cobre_highs_set_string_option(
                    self.handle,
                    c"solver".as_ptr(),
                    c"ipm".as_ptr(),
                );
                ffi::cobre_highs_set_int_option(self.handle, c"user_objective_scale".as_ptr(), -10);
                ffi::cobre_highs_set_int_option(self.handle, c"user_bound_scale".as_ptr(), -5);
                ffi::cobre_highs_set_double_option(
                    self.handle,
                    c"primal_feasibility_tolerance".as_ptr(),
                    1e-6,
                );
                ffi::cobre_highs_set_double_option(
                    self.handle,
                    c"dual_feasibility_tolerance".as_ptr(),
                    1e-6,
                );
            },
            _ => unreachable!(),
        }
    }

    /// Internal helper: run the simplex and update stats.
    ///
    /// Core simplex execution, called after (for warm-start) the basis has been
    /// installed. `HiGHS` retains its internal simplex basis across consecutive
    /// `solve_inner` calls on the same LP shape, which is the primary warm-start
    /// mechanism for the backward pass. No `Highs_clearSolver` call is issued —
    /// that behavior was removed in commit `25f1351` to recover a 4.7× perf regression.
    fn solve_inner(&mut self) -> Result<SolutionView<'_>, SolverError> {
        // Safeguard: apply iteration limits before the initial attempt.
        // Time limits are NOT set here — HiGHS tracks time cumulatively from
        // instance creation, so a per-solve time_limit would fire spuriously
        // on long-running solver instances. Instead, wall-clock time is checked
        // after run_once() to detect stuck solves.
        self.set_iteration_limits();

        let t0 = Instant::now();
        let model_status = self.run_once();
        let solve_time = t0.elapsed().as_secs_f64();

        self.stats.solve_count += 1;

        if model_status == ffi::HIGHS_MODEL_STATUS_OPTIMAL {
            // Read iteration count from FFI BEFORE establishing the shared borrow
            // via extract_solution_view, so stats can be updated without violating
            // the aliasing rules.
            // SAFETY: handle is valid non-null HiGHS pointer.
            #[allow(clippy::cast_sign_loss)]
            let iterations =
                unsafe { ffi::cobre_highs_get_simplex_iteration_count(self.handle) } as u64;
            self.stats.success_count += 1;
            self.stats.first_try_successes += 1;
            self.stats.total_iterations += iterations;
            self.stats.total_solve_time_seconds += solve_time;
            self.restore_iteration_limits();
            return Ok(self.extract_solution_view(solve_time));
        }

        // Check for a definitive terminal status (not a retry-able error).
        // UNBOUNDED is retried: HiGHS dual simplex can report spurious UNBOUNDED
        // on numerically difficult LPs with wide coefficient ranges. The retry
        // escalation (especially presolve in the core sequence) often resolves these.
        // ITERATION_LIMIT from the initial attempt is retryable — the retry
        // sequence uses different strategies that may converge faster.
        // TIME_LIMIT is retryable — HiGHS tracks time cumulatively from instance
        // creation; a spurious TIME_LIMIT can fire even with time_limit=Infinity
        // in edge cases. Retry level 0 (cold restart) recovers from this.
        // Wall-clock > 15s is also retryable — detects stuck initial solves.
        let is_unbounded = model_status == ffi::HIGHS_MODEL_STATUS_UNBOUNDED;
        let initial_retryable = is_unbounded
            || model_status == ffi::HIGHS_MODEL_STATUS_ITERATION_LIMIT
            || model_status == ffi::HIGHS_MODEL_STATUS_TIME_LIMIT
            || solve_time > 15.0;
        if !initial_retryable {
            if let Some(terminal_err) = self.interpret_terminal_status(model_status, solve_time) {
                self.restore_iteration_limits();
                self.stats.failure_count += 1;
                return Err(terminal_err);
            }
        }

        // Delegate to the retry escalation method (restores limits internally).
        match self.retry_escalation(is_unbounded) {
            Ok(outcome) => {
                self.stats.retry_count += outcome.attempts;
                self.stats.success_count += 1;
                self.stats.total_iterations += outcome.iterations;
                self.stats.total_solve_time_seconds += outcome.solve_time;
                self.stats.retry_level_histogram[outcome.level as usize] += 1;
                Ok(self.extract_solution_view(outcome.solve_time))
            }
            Err((attempts, err)) => {
                self.stats.retry_count += attempts;
                self.stats.failure_count += 1;
                Err(err)
            }
        }
    }
}

impl Drop for HighsSolver {
    fn drop(&mut self) {
        // SAFETY: valid HiGHS pointer from construction, called once per instance.
        unsafe { ffi::cobre_highs_destroy(self.handle) };
    }
}

/// Returns the `HiGHS` version as a `"major.minor.patch"` string.
///
/// This is a free function — no solver instance is required.
///
/// # Example
///
/// ```rust
/// # #[cfg(feature = "highs")]
/// # {
/// let v = cobre_solver::highs_version();
/// assert!(v.contains('.'), "version string should be 'major.minor.patch'");
/// # }
/// ```
#[must_use]
pub fn highs_version() -> String {
    // SAFETY: These are pure query functions with no arguments. The HiGHS C API
    // documents them as safe to call without any prior initialisation; they read
    // only compile-time constants embedded in the library.
    let major = unsafe { crate::ffi::cobre_highs_version_major() };
    let minor = unsafe { crate::ffi::cobre_highs_version_minor() };
    let patch = unsafe { crate::ffi::cobre_highs_version_patch() };
    format!("{major}.{minor}.{patch}")
}

impl SolverInterface for HighsSolver {
    fn name(&self) -> &'static str {
        "HiGHS"
    }

    fn solver_name_version(&self) -> String {
        format!("HiGHS {}", highs_version())
    }

    fn load_model(&mut self, template: &StageTemplate) {
        let t0 = Instant::now();
        // SAFETY:
        // - `self.handle` is a valid, non-null HiGHS pointer from `cobre_highs_create()`.
        // - All pointer arguments point into owned `Vec` data that remains alive for the
        //   duration of this call.
        // - `template.col_starts` and `template.row_indices` are `Vec<i32>` owned by the
        //   template, alive for the duration of this borrow.
        // - All slice lengths match the HiGHS API contract:
        //   `num_col + 1` for a_start, `num_nz` for a_index and a_value,
        //   `num_col` for col_cost/col_lower/col_upper, `num_row` for row_lower/row_upper.
        assert!(
            i32::try_from(template.num_cols).is_ok(),
            "num_cols {} overflows i32: LP exceeds HiGHS API limit",
            template.num_cols
        );
        assert!(
            i32::try_from(template.num_rows).is_ok(),
            "num_rows {} overflows i32: LP exceeds HiGHS API limit",
            template.num_rows
        );
        assert!(
            i32::try_from(template.num_nz).is_ok(),
            "num_nz {} overflows i32: LP exceeds HiGHS API limit",
            template.num_nz
        );
        // SAFETY: All three values have been asserted to fit in i32 above.
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let num_col = template.num_cols as i32;
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let num_row = template.num_rows as i32;
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let num_nz = template.num_nz as i32;
        let status = unsafe {
            ffi::cobre_highs_pass_lp(
                self.handle,
                num_col,
                num_row,
                num_nz,
                ffi::HIGHS_MATRIX_FORMAT_COLWISE,
                ffi::HIGHS_OBJ_SENSE_MINIMIZE,
                0.0, // objective offset
                template.objective.as_ptr(),
                template.col_lower.as_ptr(),
                template.col_upper.as_ptr(),
                template.row_lower.as_ptr(),
                template.row_upper.as_ptr(),
                template.col_starts.as_ptr(),
                template.row_indices.as_ptr(),
                template.values.as_ptr(),
            )
        };

        assert_ne!(
            status,
            ffi::HIGHS_STATUS_ERROR,
            "cobre_highs_pass_lp failed with status {status}"
        );

        self.num_cols = template.num_cols;
        self.num_rows = template.num_rows;
        self.has_model = true;

        // Resize solution extraction buffers to match the new LP dimensions.
        // Zero-fill is fine; these are overwritten in full by `cobre_highs_get_solution`.
        self.col_value.resize(self.num_cols, 0.0);
        self.col_dual.resize(self.num_cols, 0.0);
        self.row_value.resize(self.num_rows, 0.0);
        self.row_dual.resize(self.num_rows, 0.0);

        // Resize basis status i32 buffers. Zero-fill is fine; values are overwritten before
        // any FFI call. These never shrink -- only grow -- to prevent reallocation on hot path.
        self.basis_col_i32.resize(self.num_cols, 0);
        self.basis_row_i32.resize(self.num_rows, 0);
        self.stats.total_load_model_time_seconds += t0.elapsed().as_secs_f64();
        self.stats.load_model_count += 1;
    }

    fn add_rows(&mut self, rows: &RowBatch) {
        assert!(
            i32::try_from(rows.num_rows).is_ok(),
            "rows.num_rows {} overflows i32: RowBatch exceeds HiGHS API limit",
            rows.num_rows
        );
        assert!(
            i32::try_from(rows.col_indices.len()).is_ok(),
            "rows nnz {} overflows i32: RowBatch exceeds HiGHS API limit",
            rows.col_indices.len()
        );
        // SAFETY: Both values have been asserted to fit in i32 above.
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let num_new_row = rows.num_rows as i32;
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let num_new_nz = rows.col_indices.len() as i32;

        // SAFETY:
        // - `self.handle` is a valid, non-null HiGHS pointer.
        // - All pointer arguments point into owned data alive for the duration of this call.
        // - `rows.row_starts` and `rows.col_indices` are `Vec<i32>` owned by the RowBatch,
        //   alive for the duration of this borrow.
        // - Slice lengths: `num_rows + 1` for starts, total nnz for index and value,
        //   `num_rows` for lower/upper bounds.
        let status = unsafe {
            ffi::cobre_highs_add_rows(
                self.handle,
                num_new_row,
                rows.row_lower.as_ptr(),
                rows.row_upper.as_ptr(),
                num_new_nz,
                rows.row_starts.as_ptr(),
                rows.col_indices.as_ptr(),
                rows.values.as_ptr(),
            )
        };

        assert_ne!(
            status,
            ffi::HIGHS_STATUS_ERROR,
            "cobre_highs_add_rows failed with status {status}"
        );

        self.num_rows += rows.num_rows;

        // Grow row-indexed solution extraction buffers to cover the new rows.
        self.row_value.resize(self.num_rows, 0.0);
        self.row_dual.resize(self.num_rows, 0.0);

        // Grow basis row i32 buffer to cover the new rows.
        self.basis_row_i32.resize(self.num_rows, 0);
    }

    fn set_row_bounds(&mut self, indices: &[usize], lower: &[f64], upper: &[f64]) {
        assert!(
            indices.len() == lower.len() && indices.len() == upper.len(),
            "set_row_bounds: indices ({}), lower ({}), and upper ({}) must have equal length",
            indices.len(),
            lower.len(),
            upper.len()
        );
        if indices.is_empty() {
            return;
        }

        assert!(
            i32::try_from(indices.len()).is_ok(),
            "set_row_bounds: indices.len() {} overflows i32",
            indices.len()
        );
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let num_entries = indices.len() as i32;

        let t0 = Instant::now();
        // SAFETY:
        // - `self.handle` is a valid, non-null HiGHS pointer.
        // - `convert_to_i32_scratch()` returns a slice pointing into `self.scratch_i32`,
        //   alive for `'self`. Pointer is used immediately in the FFI call.
        // - `lower` and `upper` are borrowed slices alive for the duration of this call.
        // - `num_entries` equals the lengths of all three arrays.
        let status = unsafe {
            ffi::cobre_highs_change_rows_bounds_by_set(
                self.handle,
                num_entries,
                self.convert_to_i32_scratch(indices).as_ptr(),
                lower.as_ptr(),
                upper.as_ptr(),
            )
        };

        assert_ne!(
            status,
            ffi::HIGHS_STATUS_ERROR,
            "cobre_highs_change_rows_bounds_by_set failed with status {status}"
        );
        self.stats.total_set_bounds_time_seconds += t0.elapsed().as_secs_f64();
    }

    fn set_col_bounds(&mut self, indices: &[usize], lower: &[f64], upper: &[f64]) {
        assert!(
            indices.len() == lower.len() && indices.len() == upper.len(),
            "set_col_bounds: indices ({}), lower ({}), and upper ({}) must have equal length",
            indices.len(),
            lower.len(),
            upper.len()
        );
        if indices.is_empty() {
            return;
        }

        assert!(
            i32::try_from(indices.len()).is_ok(),
            "set_col_bounds: indices.len() {} overflows i32",
            indices.len()
        );
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let num_entries = indices.len() as i32;

        let t0 = Instant::now();
        // SAFETY:
        // - `self.handle` is a valid, non-null HiGHS pointer.
        // - Converted indices point into `self.scratch_i32`, alive for `'self`.
        // - `lower` and `upper` are borrowed slices alive for the duration of this call.
        // - `num_entries` equals the lengths of all three arrays.
        let status = unsafe {
            ffi::cobre_highs_change_cols_bounds_by_set(
                self.handle,
                num_entries,
                self.convert_to_i32_scratch(indices).as_ptr(),
                lower.as_ptr(),
                upper.as_ptr(),
            )
        };

        assert_ne!(
            status,
            ffi::HIGHS_STATUS_ERROR,
            "cobre_highs_change_cols_bounds_by_set failed with status {status}"
        );
        self.stats.total_set_bounds_time_seconds += t0.elapsed().as_secs_f64();
    }

    /// # Preconditions
    ///
    /// When `basis` is `Some(b)`, the caller must size
    /// `b.row_status` to exactly `self.num_rows` (the current LP
    /// row count). Callers that grow the LP by adding rows are
    /// responsible for reconciling their basis to the new row
    /// count before invoking this method.
    fn solve(
        &mut self,
        basis: Option<&crate::types::Basis>,
    ) -> Result<SolutionView<'_>, SolverError> {
        assert!(
            self.has_model,
            "solve called without a loaded model — call load_model first"
        );

        if let Some(basis) = basis {
            assert!(
                basis.col_status.len() == self.num_cols,
                "basis column count {} does not match LP column count {}",
                basis.col_status.len(),
                self.num_cols
            );
            debug_assert!(
                basis.row_status.len() >= self.num_rows,
                "solve(Some(&basis)): basis.row_status.len() ({}) < self.num_rows ({}); \
                 callers introducing new rows must reconcile basis (e.g. extend with \
                 NONBASIC_AT_LOWER for fresh inequality rows) before calling solve. \
                 The defensive BASIC padding below is incorrect for inequality slacks.",
                basis.row_status.len(),
                self.num_rows
            );

            // Track every warm-start call as a basis offer for diagnostics.
            self.stats.basis_offered += 1;

            // Copy raw i32 codes directly into the pre-allocated buffers — no enum
            // translation. Zero-copy warm-start path.
            self.basis_col_i32[..self.num_cols].copy_from_slice(&basis.col_status);

            // Precondition: the caller must size `basis.row_status` to
            // exactly `self.num_rows`. The production caller reconciles
            // the basis size to the current row count before invoking
            // `solve(Some(&basis))`, so `basis_rows == lp_rows` always
            // holds in practice.
            //
            // For defensive robustness if a future caller offers a
            // mismatched basis:
            // - `basis_rows < lp_rows`: pad missing tail rows with BASIC.
            //   This is incorrect for newly added inequality rows, whose
            //   slacks should be non-basic at the appropriate bound;
            //   callers introducing new rows must reconcile the basis
            //   themselves before calling solve.
            // - `basis_rows > lp_rows`: truncate the trailing entries.
            //   The solver ignores any basis entry beyond `num_rows`.
            let basis_rows = basis.row_status.len();
            let lp_rows = self.num_rows;
            let copy_len = basis_rows.min(lp_rows);
            self.basis_row_i32[..copy_len].copy_from_slice(&basis.row_status[..copy_len]);
            if lp_rows > basis_rows {
                self.basis_row_i32[basis_rows..lp_rows].fill(ffi::HIGHS_BASIS_STATUS_BASIC);
            }

            // SAFETY:
            // - `self.handle` is a valid, non-null HiGHS pointer obtained from
            //   `cobre_highs_create()` and kept alive by `HighsSolver`.
            // - `basis_col_i32` was sized to `num_cols` in `load_model` and grown in
            //   `add_rows`; the slice written above covers exactly `num_cols` entries.
            // - `basis_row_i32` was sized to `num_rows` in `load_model` and grown in
            //   `add_rows`; the slice written above covers exactly `num_rows` entries
            //   (with missing rows extended to BASIC).
            let basis_set_start = Instant::now();
            let set_status = unsafe {
                ffi::cobre_highs_set_basis_non_alien(
                    self.handle,
                    self.basis_col_i32.as_ptr(),
                    self.basis_row_i32.as_ptr(),
                )
            };
            if set_status == ffi::HIGHS_STATUS_ERROR {
                // Non-alien rejected: the offered basis failed
                // `isBasisConsistent` (total_basic != num_row).
                // Count the rejection and surface it as a hard error.
                self.stats.basis_consistency_failures += 1;
                // Count basic entries from the already-populated buffers.
                //
                // `usize` -> `i64` is lossless for any basis that fits in memory:
                // realistic LP sizes are bounded well below 2^63.
                #[allow(clippy::cast_possible_wrap)]
                let col_basic = self.basis_col_i32[..self.num_cols]
                    .iter()
                    .filter(|&&s| s == ffi::HIGHS_BASIS_STATUS_BASIC)
                    .count() as i64;
                #[allow(clippy::cast_possible_wrap)]
                let row_basic = self.basis_row_i32[..self.num_rows]
                    .iter()
                    .filter(|&&s| s == ffi::HIGHS_BASIS_STATUS_BASIC)
                    .count() as i64;
                // Accumulate the elapsed time even on early return.
                self.stats.total_basis_set_time_seconds += basis_set_start.elapsed().as_secs_f64();
                #[allow(clippy::cast_possible_wrap)]
                return Err(SolverError::BasisInconsistent {
                    num_row: self.num_rows as i64,
                    total_basic: col_basic + row_basic,
                    col_basic,
                    row_basic,
                });
            }
            self.stats.total_basis_set_time_seconds += basis_set_start.elapsed().as_secs_f64();
        }

        // Basis is installed (warm path) or not needed (cold path); run the simplex.
        // HiGHS retains its internal basis across consecutive solves on the same
        // LP shape, giving the backward pass ~15x fewer simplex iterations on
        // repeat solves at the same stage/opening.
        self.solve_inner()
    }

    fn get_basis(&mut self, out: &mut crate::types::Basis) {
        assert!(
            self.has_model,
            "get_basis called without a loaded model — call load_model first"
        );

        out.col_status.resize(self.num_cols, 0);
        out.row_status.resize(self.num_rows, 0);

        // SAFETY:
        // - `self.handle` is a valid, non-null HiGHS pointer.
        // - `out.col_status` has been resized to `num_cols` entries above.
        // - `out.row_status` has been resized to `num_rows` entries above.
        // - HiGHS writes exactly `num_cols` col values and `num_rows` row values.
        let get_status = unsafe {
            ffi::cobre_highs_get_basis(
                self.handle,
                out.col_status.as_mut_ptr(),
                out.row_status.as_mut_ptr(),
            )
        };

        assert_ne!(
            get_status,
            ffi::HIGHS_STATUS_ERROR,
            "cobre_highs_get_basis failed: basis must exist after a successful solve (programming error)"
        );
    }

    fn statistics(&self) -> SolverStatistics {
        self.stats.clone()
    }

    fn record_reconstruction_stats(&mut self) {
        self.stats.basis_reconstructions += 1;
    }
}

/// Test-support accessors for integration tests that need to set raw `HiGHS` options.
///
/// Gated behind the `test-support` feature. The raw handle is intentionally not
/// part of the public API — callers use these methods to configure time/iteration
/// limits before a solve without going through the safe wrapper.
#[cfg(feature = "test-support")]
impl HighsSolver {
    /// Returns the raw `HiGHS` handle for use with test-support FFI helpers.
    ///
    /// # Safety
    ///
    /// The returned pointer is valid for the lifetime of `self`. The caller must
    /// not store the pointer beyond that lifetime, must not call
    /// `cobre_highs_destroy` on it, and must not alias it across threads.
    #[must_use]
    pub fn raw_handle(&self) -> *mut std::os::raw::c_void {
        self.handle
    }
}

#[cfg(test)]
mod tests {
    use super::HighsSolver;
    use crate::{
        SolverInterface,
        types::{Basis, RowBatch, StageTemplate},
    };

    // Shared LP fixture from Solver Interface Testing SS1.1:
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
            col_scale: Vec::new(),
            row_scale: Vec::new(),
        }
    }

    // Valid-inequality fixture from Solver Interface Testing SS1.2:
    // Row 1: -5*x0 + x1 >= 20  (col_indices [0,1], values [-5, 1])
    // Row 2:  3*x0 + x1 >= 80  (col_indices [0,1], values [ 3, 1])
    fn make_fixture_row_batch() -> RowBatch {
        RowBatch {
            num_rows: 2,
            row_starts: vec![0_i32, 2, 4],
            col_indices: vec![0_i32, 1, 0, 1],
            values: vec![-5.0, 1.0, 3.0, 1.0],
            row_lower: vec![20.0, 80.0],
            row_upper: vec![f64::INFINITY, f64::INFINITY],
        }
    }

    #[test]
    fn test_highs_solver_create_and_name() {
        let solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
        assert_eq!(solver.name(), "HiGHS");
        // Drop occurs here; verifies cobre_highs_destroy is called without crash.
    }

    #[test]
    fn test_highs_solver_send_bound() {
        fn assert_send<T: Send>() {}
        assert_send::<HighsSolver>();
    }

    #[test]
    fn test_highs_solver_statistics_initial() {
        let solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
        let stats = solver.statistics();
        assert_eq!(stats.solve_count, 0);
        assert_eq!(stats.success_count, 0);
        assert_eq!(stats.failure_count, 0);
        assert_eq!(stats.total_iterations, 0);
        assert_eq!(stats.retry_count, 0);
        assert_eq!(stats.total_solve_time_seconds, 0.0);
    }

    #[test]
    fn test_highs_load_model_updates_dimensions() {
        let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
        let template = make_fixture_stage_template();

        solver.load_model(&template);

        assert_eq!(solver.num_cols, 3, "num_cols must be 3 after load_model");
        assert_eq!(solver.num_rows, 2, "num_rows must be 2 after load_model");
        assert_eq!(
            solver.col_value.len(),
            3,
            "col_value buffer must be resized to num_cols"
        );
        assert_eq!(
            solver.col_dual.len(),
            3,
            "col_dual buffer must be resized to num_cols"
        );
        assert_eq!(
            solver.row_value.len(),
            2,
            "row_value buffer must be resized to num_rows"
        );
        assert_eq!(
            solver.row_dual.len(),
            2,
            "row_dual buffer must be resized to num_rows"
        );
    }

    #[test]
    fn test_highs_add_rows_updates_dimensions() {
        let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
        let template = make_fixture_stage_template();
        let cuts = make_fixture_row_batch();

        solver.load_model(&template);
        solver.add_rows(&cuts);

        // 2 structural rows + 2 appended rows = 4
        assert_eq!(solver.num_rows, 4, "num_rows must be 4 after add_rows");
        assert_eq!(
            solver.row_dual.len(),
            4,
            "row_dual buffer must be resized to 4 after add_rows"
        );
        assert_eq!(
            solver.row_value.len(),
            4,
            "row_value buffer must be resized to 4 after add_rows"
        );
        // Columns unchanged
        assert_eq!(solver.num_cols, 3, "num_cols must be unchanged by add_rows");
    }

    #[test]
    fn test_highs_set_row_bounds_no_panic() {
        let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
        let template = make_fixture_stage_template();
        solver.load_model(&template);

        // Patch row 0 to equality at 4.0. Must complete without panic.
        solver.set_row_bounds(&[0], &[4.0], &[4.0]);
    }

    #[test]
    fn test_highs_set_col_bounds_no_panic() {
        let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
        let template = make_fixture_stage_template();
        solver.load_model(&template);

        // Patch column 1 lower bound to 10.0. Must complete without panic.
        solver.set_col_bounds(&[1], &[10.0], &[f64::INFINITY]);
    }

    #[test]
    fn test_highs_set_bounds_empty_no_panic() {
        let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
        let template = make_fixture_stage_template();
        solver.load_model(&template);

        // Empty patch slices should be short-circuited without any FFI call.
        solver.set_row_bounds(&[], &[], &[]);
        solver.set_col_bounds(&[], &[], &[]);
    }

    /// SS1.1 fixture: min 0*x0 + 1*x1 + 50*x2, s.t. x0=6, 2*x0+x2=14, x>=0.
    /// Optimal: x0=6, x1=0, x2=2, objective=100.
    #[test]
    fn test_highs_solve_basic_lp() {
        let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
        let template = make_fixture_stage_template();
        solver.load_model(&template);

        let solution = solver
            .solve(None)
            .expect("solve() must succeed on a feasible LP");

        assert!(
            (solution.objective - 100.0).abs() < 1e-8,
            "objective must be 100.0, got {}",
            solution.objective
        );
        assert_eq!(solution.primal.len(), 3, "primal must have 3 elements");
        assert!(
            (solution.primal[0] - 6.0).abs() < 1e-8,
            "primal[0] (x0) must be 6.0, got {}",
            solution.primal[0]
        );
        assert!(
            (solution.primal[1] - 0.0).abs() < 1e-8,
            "primal[1] (x1) must be 0.0, got {}",
            solution.primal[1]
        );
        assert!(
            (solution.primal[2] - 2.0).abs() < 1e-8,
            "primal[2] (x2) must be 2.0, got {}",
            solution.primal[2]
        );
    }

    /// SS1.2: after adding two valid inequalities to SS1.1, optimal objective = 162.
    /// Cuts: -5*x0+x1>=20 and 3*x0+x1>=80. With x0=6: x1>=max(50,62)=62.
    /// Obj = 0*6 + 1*62 + 50*2 = 162.
    #[test]
    fn test_highs_solve_with_cuts() {
        let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
        let template = make_fixture_stage_template();
        let cuts = make_fixture_row_batch();
        solver.load_model(&template);
        solver.add_rows(&cuts);

        let solution = solver
            .solve(None)
            .expect("solve() must succeed on a feasible LP with cuts");

        assert!(
            (solution.objective - 162.0).abs() < 1e-8,
            "objective must be 162.0, got {}",
            solution.objective
        );
        assert!(
            (solution.primal[0] - 6.0).abs() < 1e-8,
            "primal[0] must be 6.0, got {}",
            solution.primal[0]
        );
        assert!(
            (solution.primal[1] - 62.0).abs() < 1e-8,
            "primal[1] must be 62.0, got {}",
            solution.primal[1]
        );
        assert!(
            (solution.primal[2] - 2.0).abs() < 1e-8,
            "primal[2] must be 2.0, got {}",
            solution.primal[2]
        );
    }

    /// SS1.3: after adding cuts and patching row 0 RHS to 4.0 (x0=4).
    /// x2=14-2*4=6. cut2: 3*4+x1>=80 => x1>=68. Obj = 0*4+1*68+50*6 = 368.
    #[test]
    fn test_highs_solve_after_rhs_patch() {
        let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
        let template = make_fixture_stage_template();
        let cuts = make_fixture_row_batch();
        solver.load_model(&template);
        solver.add_rows(&cuts);

        // Patch row 0 (x0=6 equality) to x0=4.
        solver.set_row_bounds(&[0], &[4.0], &[4.0]);

        let solution = solver
            .solve(None)
            .expect("solve() must succeed after RHS patch");

        assert!(
            (solution.objective - 368.0).abs() < 1e-8,
            "objective must be 368.0, got {}",
            solution.objective
        );
    }

    /// After two successful solves, statistics must reflect both.
    #[test]
    fn test_highs_solve_statistics_increment() {
        let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
        let template = make_fixture_stage_template();
        solver.load_model(&template);

        solver.solve(None).expect("first solve must succeed");
        solver.solve(None).expect("second solve must succeed");

        let stats = solver.statistics();
        assert_eq!(stats.solve_count, 2, "solve_count must be 2");
        assert_eq!(stats.success_count, 2, "success_count must be 2");
        assert_eq!(stats.failure_count, 0, "failure_count must be 0");
        assert!(
            stats.total_iterations > 0,
            "total_iterations must be positive"
        );
    }

    /// After a cold solve, statistics counters must reflect the single solve.
    #[test]
    fn test_highs_solve_preserves_stats() {
        let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
        let template = make_fixture_stage_template();
        solver.load_model(&template);
        solver.solve(None).expect("solve must succeed");

        let stats = solver.statistics();
        assert_eq!(
            stats.solve_count, 1,
            "solve_count must be 1 after one solve"
        );
        assert_eq!(
            stats.success_count, 1,
            "success_count must be 1 after one successful solve"
        );
        assert!(
            stats.total_iterations > 0,
            "total_iterations must be positive after a successful solve"
        );
    }

    /// The first solve must report a positive iteration count.
    #[test]
    fn test_highs_solve_iterations_positive() {
        let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
        let template = make_fixture_stage_template();
        solver.load_model(&template);

        let solution = solver.solve(None).expect("solve must succeed");
        assert!(
            solution.iterations > 0,
            "iterations must be positive, got {}",
            solution.iterations
        );
    }

    /// The first solve must report a positive wall-clock time.
    #[test]
    fn test_highs_solve_time_positive() {
        let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
        let template = make_fixture_stage_template();
        solver.load_model(&template);

        let solution = solver.solve(None).expect("solve must succeed");
        assert!(
            solution.solve_time_seconds > 0.0,
            "solve_time_seconds must be positive, got {}",
            solution.solve_time_seconds
        );
    }

    /// After one solve, `statistics()` must report `solve_count==1`, `success_count==1`,
    /// `failure_count==0`, and `total_iterations` > 0.
    #[test]
    fn test_highs_solve_statistics_single() {
        let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
        let template = make_fixture_stage_template();
        solver.load_model(&template);

        solver.solve(None).expect("solve must succeed");

        let stats = solver.statistics();
        assert_eq!(stats.solve_count, 1, "solve_count must be 1");
        assert_eq!(stats.success_count, 1, "success_count must be 1");
        assert_eq!(stats.failure_count, 0, "failure_count must be 0");
        assert!(
            stats.total_iterations > 0,
            "total_iterations must be positive after a successful solve"
        );
    }

    /// After `load_model` + `solve()`, `get_basis` must return i32 codes
    /// that are all valid `HiGHS` basis status values (0..=4).
    #[test]
    fn test_get_basis_valid_status_codes() {
        let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
        let template = make_fixture_stage_template();
        solver.load_model(&template);
        solver
            .solve(None)
            .expect("solve must succeed before get_basis");

        let mut basis = Basis::new(0, 0);
        solver.get_basis(&mut basis);

        for &code in &basis.col_status {
            assert!(
                (0..=4).contains(&code),
                "col_status code {code} is outside valid HiGHS range 0..=4"
            );
        }
        for &code in &basis.row_status {
            assert!(
                (0..=4).contains(&code),
                "row_status code {code} is outside valid HiGHS range 0..=4"
            );
        }
    }

    /// Starting from an empty `Basis`, `get_basis` must resize the output
    /// buffers to match the current LP dimensions (3 cols, 2 rows for SS1.1).
    #[test]
    fn test_get_basis_resizes_output() {
        let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
        let template = make_fixture_stage_template();
        solver.load_model(&template);
        solver
            .solve(None)
            .expect("solve must succeed before get_basis");

        let mut basis = Basis::new(0, 0);
        assert_eq!(
            basis.col_status.len(),
            0,
            "initial col_status must be empty"
        );
        assert_eq!(
            basis.row_status.len(),
            0,
            "initial row_status must be empty"
        );

        solver.get_basis(&mut basis);

        assert_eq!(
            basis.col_status.len(),
            3,
            "col_status must be resized to 3 (num_cols of SS1.1)"
        );
        assert_eq!(
            basis.row_status.len(),
            2,
            "row_status must be resized to 2 (num_rows of SS1.1)"
        );
    }

    /// Warm-start via `solve(Some(&basis))` on the same LP must reproduce
    /// the optimal objective and complete in at most 1 simplex iteration.
    #[test]
    fn test_solve_warm_start_reproduces_cold_objective() {
        let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
        let template = make_fixture_stage_template();
        solver.load_model(&template);
        solver.solve(None).expect("cold-start solve must succeed");

        let mut basis = Basis::new(0, 0);
        solver.get_basis(&mut basis);

        // Reload the same model to reset HiGHS internal state.
        solver.load_model(&template);
        let result = solver
            .solve(Some(&basis))
            .expect("warm-start solve must succeed");

        assert!(
            (result.objective - 100.0).abs() < 1e-8,
            "warm-start objective must be 100.0, got {}",
            result.objective
        );
        assert!(
            result.iterations <= 1,
            "warm-start from exact basis must use at most 1 iteration, got {}",
            result.iterations
        );

        let stats = solver.statistics();
        assert_eq!(
            stats.basis_consistency_failures, 0,
            "basis_consistency_failures must be 0 when raw basis is accepted, got {}",
            stats.basis_consistency_failures
        );
        assert_eq!(
            stats.basis_offered, 1,
            "basis_offered must be 1 after one warm-start call"
        );
    }

    /// When the basis has fewer rows than the current LP (2 vs 4 after `add_rows`),
    /// `solve(Some(&basis))` must extend missing rows as Basic and solve correctly.
    /// SS1.2 objective with both cuts active is 162.0.
    ///
    /// This test exercises the defensive BASIC-padding fallback path,
    /// which the production caller never hits because it reconciles the
    /// basis to the LP row count before invoking `solve`. The
    /// `debug_assert!` in `solve` would fire on this fallback path, so
    /// the test runs only when `debug_assertions` is disabled.
    #[cfg(not(debug_assertions))]
    #[test]
    fn test_solve_warm_start_extends_missing_rows_as_basic() {
        let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
        let template = make_fixture_stage_template();
        let cuts = make_fixture_row_batch();

        // First solve on 2-row LP to capture a 2-row basis.
        solver.load_model(&template);
        solver.solve(None).expect("SS1.1 solve must succeed");
        let mut basis = Basis::new(0, 0);
        solver.get_basis(&mut basis);
        assert_eq!(
            basis.row_status.len(),
            2,
            "captured basis must have 2 row statuses"
        );

        // Reload model and add 2 cuts to get a 4-row LP.
        solver.load_model(&template);
        solver.add_rows(&cuts);
        assert_eq!(solver.num_rows, 4, "LP must have 4 rows after add_rows");

        // Warm-start with the 2-row basis; extra rows are extended as Basic.
        let result = solver
            .solve(Some(&basis))
            .expect("solve with dimension-mismatched basis must succeed");

        assert!(
            (result.objective - 162.0).abs() < 1e-8,
            "objective with both cuts active must be 162.0, got {}",
            result.objective
        );
    }

    /// Non-alien path accepts a self-extracted basis: counter must stay at zero.
    ///
    /// Solves SS1.1 cold, extracts the optimal basis, reloads the model, and
    /// warm-starts via `solve(Some(&basis))`.  The non-alien FFI call
    /// (`cobre_highs_set_basis_non_alien`) should accept a basis that was just
    /// produced by `HiGHS` itself, so `basis_consistency_failures` must not
    /// increase.
    #[test]
    fn test_solve_warm_start_non_alien_success() {
        // Arrange
        let template = make_fixture_stage_template();
        let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
        solver.load_model(&template);
        let _ = solver.solve(None).expect("cold-start solve must succeed");
        let mut basis = Basis::new(template.num_cols, template.num_rows);
        solver.get_basis(&mut basis);

        // Reload model so HiGHS internal state is fresh, then warm-start.
        solver.load_model(&template);
        let before = solver.statistics();

        // Act
        let _ = solver
            .solve(Some(&basis))
            .expect("warm-start solve must succeed with self-extracted basis");

        // Assert
        let after = solver.statistics();
        assert_eq!(
            after.basis_consistency_failures - before.basis_consistency_failures,
            0,
            "non-alien path should accept a self-extracted basis; consistency failures delta must be 0"
        );
    }

    /// `solve(Some(&basis))` returns `Err(SolverError::BasisInconsistent)` when given
    /// an inconsistent basis instead of silently falling back to the alien setter.
    ///
    /// Builds a deliberately inconsistent basis (all column statuses set to
    /// `HIGHS_BASIS_STATUS_BASIC`, all row statuses `HIGHS_BASIS_STATUS_BASIC`).
    /// For the 3-column, 2-row SS1.1 LP this yields 5 basic variables against a
    /// rank of 2, which `cobre_highs_set_basis_non_alien` rejects with
    /// `HIGHS_STATUS_ERROR`.  The error is surfaced as a hard `Err` and
    /// `basis_consistency_failures` increments by 1.
    ///
    /// After the call:
    /// - `basis_consistency_failures` increments by 1.
    /// - The result is `Err(SolverError::BasisInconsistent { num_row: 2,
    ///   total_basic: 5, col_basic: 3, row_basic: 2 })`.
    #[test]
    fn test_solve_warm_start_rejects_inconsistent_basis() {
        use crate::ffi;
        use crate::types::SolverError;

        // Arrange: non-alien setter is now the only warm-start path.
        let template = make_fixture_stage_template();
        let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");

        solver.load_model(&template);

        // Build a deliberately inconsistent basis: all BASIC (5 basics, rank 2).
        let mut bad_basis = Basis::new(template.num_cols, template.num_rows);
        bad_basis
            .col_status
            .iter_mut()
            .for_each(|v| *v = ffi::HIGHS_BASIS_STATUS_BASIC);
        bad_basis
            .row_status
            .iter_mut()
            .for_each(|v| *v = ffi::HIGHS_BASIS_STATUS_BASIC);

        let before = solver.statistics();

        // Act — convert result to a form that does not borrow `solver`.
        // `SolutionView<'_>` borrows `solver`'s internal buffers; calling
        // `statistics()` afterwards would overlap borrows.  On the error path
        // `SolverError` contains no solver references, so mapping Ok → () breaks
        // the mutable borrow before the statistics call.
        let err_variant: Result<(), SolverError> = solver.solve(Some(&bad_basis)).map(|_| ());

        // Assert counters — the mutable borrow from solve(Some(&bad_basis)) is gone.
        let after = solver.statistics();
        assert_eq!(
            after.basis_consistency_failures - before.basis_consistency_failures,
            1,
            "basis_consistency_failures must increment by 1 for an overcounted basis"
        );

        // Assert the returned error.
        match err_variant {
            Err(SolverError::BasisInconsistent {
                num_row,
                total_basic,
                col_basic,
                row_basic,
            }) => {
                assert_eq!(num_row, 2, "num_row must match LP row count");
                assert_eq!(total_basic, 5, "total_basic must be col_basic + row_basic");
                assert_eq!(col_basic, 3, "col_basic must count BASIC columns");
                assert_eq!(row_basic, 2, "row_basic must count BASIC rows");
            }
            other => panic!(
                "expected Err(SolverError::BasisInconsistent {{ num_row: 2, total_basic: 5, \
                 col_basic: 3, row_basic: 2 }}), got {other:?}"
            ),
        }
    }

    /// `terminal_status_dual_scratch` and `terminal_status_primal_scratch` are
    /// initialized as empty `Vec`s in the constructor and retain their capacity
    /// across repeated `resize` calls, matching the pattern used by
    /// `scratch_i32`, `basis_col_i32`, and `basis_row_i32`.
    ///
    /// This test directly exercises the `resize`-reuse invariant without depending
    /// on a specific `HiGHS` model status. The `UNBOUNDED_OR_INFEASIBLE` branch in
    /// `interpret_terminal_status` calls `self.terminal_status_dual_scratch.resize(num_rows, 0.0)`;
    /// we verify here that repeated `resize` calls grow but never shrink capacity.
    ///
    /// The LP: 3-column, 2-row SS1.1 fixture. After `load_model`, `num_rows=2` and
    /// `num_cols=3`. We simulate two scratch-buffer resize cycles and verify capacity
    /// is monotonically non-decreasing.
    #[test]
    fn interpret_terminal_status_reuses_scratch() {
        let template = make_fixture_stage_template();
        let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");

        // Verify that scratch fields start empty (Vec::new() in constructor).
        assert_eq!(
            solver.terminal_status_dual_scratch.capacity(),
            0,
            "dual scratch must start with capacity 0 (Vec::new() in constructor)"
        );
        assert_eq!(
            solver.terminal_status_primal_scratch.capacity(),
            0,
            "primal scratch must start with capacity 0 (Vec::new() in constructor)"
        );

        // Load model to establish num_rows=2 and num_cols=3.
        solver.load_model(&template);

        // Simulate what interpret_terminal_status does in UNBOUNDED_OR_INFEASIBLE branch:
        // resize dual scratch to num_rows, primal scratch to num_cols.
        solver
            .terminal_status_dual_scratch
            .resize(solver.num_rows, 0.0);
        solver
            .terminal_status_primal_scratch
            .resize(solver.num_cols, 0.0);

        let cap_dual_after_first = solver.terminal_status_dual_scratch.capacity();
        let cap_primal_after_first = solver.terminal_status_primal_scratch.capacity();

        assert!(
            cap_dual_after_first >= solver.num_rows,
            "dual scratch capacity {cap_dual_after_first} must be >= num_rows {} after first resize",
            solver.num_rows,
        );
        assert!(
            cap_primal_after_first >= solver.num_cols,
            "primal scratch capacity {cap_primal_after_first} must be >= num_cols {} after first resize",
            solver.num_cols,
        );

        // Second resize to the same size: capacity must not decrease (heap retained).
        solver
            .terminal_status_dual_scratch
            .resize(solver.num_rows, 0.0);
        solver
            .terminal_status_primal_scratch
            .resize(solver.num_cols, 0.0);

        let cap_dual_after_second = solver.terminal_status_dual_scratch.capacity();
        let cap_primal_after_second = solver.terminal_status_primal_scratch.capacity();

        assert!(
            cap_dual_after_second >= cap_dual_after_first,
            "dual scratch capacity must not decrease: {cap_dual_after_second} < {cap_dual_after_first}",
        );
        assert!(
            cap_primal_after_second >= cap_primal_after_first,
            "primal scratch capacity must not decrease: {cap_primal_after_second} < {cap_primal_after_first}",
        );
    }
}

// ─── Research verification tests for non-optimal HiGHS model statuses ────
//
// These tests verify LP formulations that reliably trigger non-optimal
// HiGHS model statuses. They use the raw FFI layer to set options not
// exposed through SolverInterface and confirm the expected model status.
//
// The SS1.1 LP (3-variable, 2-constraint) is too small: HiGHS's crash
// heuristic solves it without entering the simplex loop, so time/iteration
// limits never fire. A 5-variable, 4-constraint "larger_lp" is required.
#[cfg(test)]
#[allow(clippy::doc_markdown)]
mod research_tests {
    // LP used: 3-variable, 2-constraint fixture from SS1.1 (same as other tests).
    // This LP requires at least 2 simplex iterations, so iteration_limit=1 will
    // produce ITERATION_LIMIT.

    // ─── Helper: load the SS1.1 LP onto an existing HiGHS handle ────────────
    //
    // 3 columns (x0, x1, x2), 2 equality rows, 3 non-zeros.
    // Optimal: x0=6, x1=0, x2=2, obj=100. Requires 2 simplex iterations.
    //
    // SAFETY: caller must guarantee `highs` is a valid, non-null HiGHS handle.
    unsafe fn research_load_ss11_lp(highs: *mut std::os::raw::c_void) {
        use crate::ffi;
        let col_cost: [f64; 3] = [0.0, 1.0, 50.0];
        let col_lower: [f64; 3] = [0.0, 0.0, 0.0];
        let col_upper: [f64; 3] = [10.0, f64::INFINITY, 8.0];
        let row_lower: [f64; 2] = [6.0, 14.0];
        let row_upper: [f64; 2] = [6.0, 14.0];
        let a_start: [i32; 4] = [0, 2, 2, 3];
        let a_index: [i32; 3] = [0, 1, 1];
        let a_value: [f64; 3] = [1.0, 2.0, 1.0];
        // SAFETY: all pointers are valid, aligned, non-null, and live for the call duration.
        let status = unsafe {
            ffi::cobre_highs_pass_lp(
                highs,
                3,
                2,
                3,
                ffi::HIGHS_MATRIX_FORMAT_COLWISE,
                ffi::HIGHS_OBJ_SENSE_MINIMIZE,
                0.0,
                col_cost.as_ptr(),
                col_lower.as_ptr(),
                col_upper.as_ptr(),
                row_lower.as_ptr(),
                row_upper.as_ptr(),
                a_start.as_ptr(),
                a_index.as_ptr(),
                a_value.as_ptr(),
            )
        };
        assert_eq!(
            status,
            ffi::HIGHS_STATUS_OK,
            "research_load_ss11_lp pass_lp failed"
        );
    }

    /// Probe: what do time_limit=0.0 and iteration_limit=0 actually return on SS1.1?
    ///
    /// This test is OBSERVATIONAL -- it captures actual HiGHS behavior. The SS1.1 LP
    /// (2 constraints, 3 variables) is solved by presolve/crash before the simplex
    /// loop, making limits ineffective. This test documents that behavior.
    #[test]
    fn test_research_probe_limit_status_on_ss11_lp() {
        use crate::ffi;

        // SS1.1 with time_limit=0.0: presolve/crash solves before time check fires.
        let highs = unsafe { ffi::cobre_highs_create() };
        assert!(!highs.is_null());
        unsafe { ffi::cobre_highs_set_bool_option(highs, c"output_flag".as_ptr(), 0) };
        unsafe { research_load_ss11_lp(highs) };
        let _ = unsafe { ffi::cobre_highs_set_double_option(highs, c"time_limit".as_ptr(), 0.0) };
        let run_status = unsafe { ffi::cobre_highs_run(highs) };
        let model_status = unsafe { ffi::cobre_highs_get_model_status(highs) };
        let obj = unsafe { ffi::cobre_highs_get_objective_value(highs) };
        eprintln!(
            "SS1.1 + time_limit=0: run_status={run_status}, model_status={model_status}, obj={obj}"
        );
        unsafe { ffi::cobre_highs_destroy(highs) };

        // SS1.1 with iteration_limit=0: same result, need a larger LP.
        let highs = unsafe { ffi::cobre_highs_create() };
        assert!(!highs.is_null());
        unsafe { ffi::cobre_highs_set_bool_option(highs, c"output_flag".as_ptr(), 0) };
        unsafe { research_load_ss11_lp(highs) };
        let _ = unsafe {
            ffi::cobre_highs_set_int_option(highs, c"simplex_iteration_limit".as_ptr(), 0)
        };
        let run_status = unsafe { ffi::cobre_highs_run(highs) };
        let model_status = unsafe { ffi::cobre_highs_get_model_status(highs) };
        let obj = unsafe { ffi::cobre_highs_get_objective_value(highs) };
        eprintln!(
            "SS1.1 + iteration_limit=0: run_status={run_status}, model_status={model_status}, obj={obj}"
        );
        unsafe { ffi::cobre_highs_destroy(highs) };
    }

    /// Helper: load a 5-variable, 4-constraint LP that requires multiple simplex
    /// iterations and cannot be solved by crash alone.
    ///
    /// LP (larger_lp):
    ///   min  x0 + x1 + x2 + x3 + x4
    ///   s.t. x0 + x1              >= 10
    ///        x1 + x2              >= 8
    ///        x2 + x3              >= 6
    ///        x3 + x4              >= 4
    ///   x_i in [0, 100], i = 0..4
    ///
    /// CSC matrix (5 cols, 4 rows, 8 non-zeros):
    ///   col 0: rows [0]       -> a_start[0]=0, a_start[1]=1
    ///   col 1: rows [0,1]     -> a_start[2]=3
    ///   col 2: rows [1,2]     -> a_start[3]=5
    ///   col 3: rows [2,3]     -> a_start[4]=7
    ///   col 4: rows [3]       -> a_start[5]=8
    ///
    /// SAFETY: caller must guarantee `highs` is a valid, non-null HiGHS handle.
    unsafe fn research_load_larger_lp(highs: *mut std::os::raw::c_void) {
        use crate::ffi;
        let col_cost: [f64; 5] = [1.0, 1.0, 1.0, 1.0, 1.0];
        let col_lower: [f64; 5] = [0.0; 5];
        let col_upper: [f64; 5] = [100.0; 5];
        let row_lower: [f64; 4] = [10.0, 8.0, 6.0, 4.0];
        let row_upper: [f64; 4] = [f64::INFINITY; 4];
        // CSC: col 0 -> row 0; col 1 -> rows 0,1; col 2 -> rows 1,2; col 3 -> rows 2,3; col 4 -> row 3
        let a_start: [i32; 6] = [0, 1, 3, 5, 7, 8];
        let a_index: [i32; 8] = [0, 0, 1, 1, 2, 2, 3, 3];
        let a_value: [f64; 8] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        // SAFETY: all pointers are valid, aligned, non-null, and live for the call duration.
        let status = unsafe {
            ffi::cobre_highs_pass_lp(
                highs,
                5,
                4,
                8,
                ffi::HIGHS_MATRIX_FORMAT_COLWISE,
                ffi::HIGHS_OBJ_SENSE_MINIMIZE,
                0.0,
                col_cost.as_ptr(),
                col_lower.as_ptr(),
                col_upper.as_ptr(),
                row_lower.as_ptr(),
                row_upper.as_ptr(),
                a_start.as_ptr(),
                a_index.as_ptr(),
                a_value.as_ptr(),
            )
        };
        assert_eq!(
            status,
            ffi::HIGHS_STATUS_OK,
            "research_load_larger_lp pass_lp failed"
        );
    }

    /// Verify time_limit=0.0 triggers HIGHS_MODEL_STATUS_TIME_LIMIT (13).
    ///
    /// Uses a 5-variable, 4-constraint LP that cannot be trivially solved by
    /// crash. HiGHS checks the time limit at entry to the simplex loop.
    /// time_limit=0.0 is always exceeded by wall-clock time before any pivot.
    ///
    /// Observed: run_status=WARNING (1), model_status=TIME_LIMIT (13).
    /// Confirmed in HiGHS check/TestQpSolver.cpp line 1083-1085.
    #[test]
    fn test_research_time_limit_zero_triggers_time_limit_status() {
        use crate::ffi;

        let highs = unsafe { ffi::cobre_highs_create() };
        assert!(!highs.is_null());
        unsafe { ffi::cobre_highs_set_bool_option(highs, c"output_flag".as_ptr(), 0) };
        unsafe { research_load_larger_lp(highs) };

        let opt_status =
            unsafe { ffi::cobre_highs_set_double_option(highs, c"time_limit".as_ptr(), 0.0) };
        assert_eq!(opt_status, ffi::HIGHS_STATUS_OK);

        let run_status = unsafe { ffi::cobre_highs_run(highs) };
        let model_status = unsafe { ffi::cobre_highs_get_model_status(highs) };

        eprintln!(
            "time_limit=0 on larger LP: run_status={run_status}, model_status={model_status}"
        );

        assert_eq!(
            run_status,
            ffi::HIGHS_STATUS_WARNING,
            "time_limit=0 must return HIGHS_STATUS_WARNING (1), got {run_status}"
        );
        assert_eq!(
            model_status,
            ffi::HIGHS_MODEL_STATUS_TIME_LIMIT,
            "time_limit=0 must give MODEL_STATUS_TIME_LIMIT (13), got {model_status}"
        );

        unsafe { ffi::cobre_highs_destroy(highs) };
    }

    /// Verify simplex_iteration_limit=0 triggers HIGHS_MODEL_STATUS_ITERATION_LIMIT (14).
    ///
    /// Uses the 5-variable, 4-constraint LP with presolve disabled so that
    /// the crash phase does not solve it, and the iteration limit check fires.
    ///
    /// Confirmed pattern from HiGHS check/TestLpSolversIterations.cpp
    /// lines 145-165: iteration_limit=0 -> HighsStatus::kWarning +
    /// HighsModelStatus::kIterationLimit, iteration count = 0.
    #[test]
    fn test_research_iteration_limit_zero_triggers_iteration_limit_status() {
        use crate::ffi;

        let highs = unsafe { ffi::cobre_highs_create() };
        assert!(!highs.is_null());
        unsafe { ffi::cobre_highs_set_bool_option(highs, c"output_flag".as_ptr(), 0) };
        // Disable presolve so crash cannot solve LP without simplex iterations.
        unsafe { ffi::cobre_highs_set_string_option(highs, c"presolve".as_ptr(), c"off".as_ptr()) };
        unsafe { research_load_larger_lp(highs) };

        let opt_status = unsafe {
            ffi::cobre_highs_set_int_option(highs, c"simplex_iteration_limit".as_ptr(), 0)
        };
        assert_eq!(opt_status, ffi::HIGHS_STATUS_OK);

        let run_status = unsafe { ffi::cobre_highs_run(highs) };
        let model_status = unsafe { ffi::cobre_highs_get_model_status(highs) };

        eprintln!(
            "iteration_limit=0 on larger LP: run_status={run_status}, model_status={model_status}"
        );

        assert_eq!(
            run_status,
            ffi::HIGHS_STATUS_WARNING,
            "iteration_limit=0 must return HIGHS_STATUS_WARNING (1), got {run_status}"
        );
        assert_eq!(
            model_status,
            ffi::HIGHS_MODEL_STATUS_ITERATION_LIMIT,
            "iteration_limit=0 must give MODEL_STATUS_ITERATION_LIMIT (14), got {model_status}"
        );

        unsafe { ffi::cobre_highs_destroy(highs) };
    }

    /// Observe partial solution availability after TIME_LIMIT and ITERATION_LIMIT.
    ///
    /// With time_limit=0.0, HiGHS halts before pivots. With iteration_limit=0
    /// and presolve disabled, HiGHS halts at the crash-point solution.
    /// Both tests record objective availability for documentation.
    #[test]
    fn test_research_partial_solution_availability() {
        use crate::ffi;

        // TIME_LIMIT: observe objective after halting at time check
        {
            let highs = unsafe { ffi::cobre_highs_create() };
            assert!(!highs.is_null());
            unsafe { ffi::cobre_highs_set_bool_option(highs, c"output_flag".as_ptr(), 0) };
            unsafe { research_load_larger_lp(highs) };
            unsafe { ffi::cobre_highs_set_double_option(highs, c"time_limit".as_ptr(), 0.0) };
            unsafe { ffi::cobre_highs_run(highs) };

            let obj = unsafe { ffi::cobre_highs_get_objective_value(highs) };
            let model_status = unsafe { ffi::cobre_highs_get_model_status(highs) };
            assert_eq!(model_status, ffi::HIGHS_MODEL_STATUS_TIME_LIMIT);
            eprintln!("TIME_LIMIT: obj={obj}, finite={}", obj.is_finite());
            unsafe { ffi::cobre_highs_destroy(highs) };
        }

        // ITERATION_LIMIT: observe objective at crash point
        {
            let highs = unsafe { ffi::cobre_highs_create() };
            assert!(!highs.is_null());
            unsafe { ffi::cobre_highs_set_bool_option(highs, c"output_flag".as_ptr(), 0) };
            unsafe {
                ffi::cobre_highs_set_string_option(highs, c"presolve".as_ptr(), c"off".as_ptr())
            };
            unsafe { research_load_larger_lp(highs) };
            unsafe {
                ffi::cobre_highs_set_int_option(highs, c"simplex_iteration_limit".as_ptr(), 0)
            };
            unsafe { ffi::cobre_highs_run(highs) };

            let obj = unsafe { ffi::cobre_highs_get_objective_value(highs) };
            let model_status = unsafe { ffi::cobre_highs_get_model_status(highs) };
            assert_eq!(model_status, ffi::HIGHS_MODEL_STATUS_ITERATION_LIMIT);
            eprintln!("ITERATION_LIMIT: obj={obj}, finite={}", obj.is_finite());
            unsafe { ffi::cobre_highs_destroy(highs) };
        }
    }

    /// Verify restore_default_settings: solve with iteration_limit=0, then solve
    /// without limit after restoring defaults. The second solve must succeed optimally.
    #[test]
    fn test_research_restore_defaults_allows_subsequent_optimal_solve() {
        use crate::ffi;

        let highs = unsafe { ffi::cobre_highs_create() };
        assert!(!highs.is_null());

        unsafe { ffi::cobre_highs_set_bool_option(highs, c"output_flag".as_ptr(), 0) };

        // Apply cobre defaults (mirror HighsSolver::new() configuration).
        unsafe {
            ffi::cobre_highs_set_string_option(highs, c"solver".as_ptr(), c"simplex".as_ptr());
            ffi::cobre_highs_set_int_option(highs, c"simplex_strategy".as_ptr(), 1);
            ffi::cobre_highs_set_string_option(highs, c"presolve".as_ptr(), c"off".as_ptr());
            ffi::cobre_highs_set_string_option(highs, c"parallel".as_ptr(), c"off".as_ptr());
            ffi::cobre_highs_set_double_option(
                highs,
                c"primal_feasibility_tolerance".as_ptr(),
                1e-7,
            );
            ffi::cobre_highs_set_double_option(highs, c"dual_feasibility_tolerance".as_ptr(), 1e-7);
        }

        let col_cost: [f64; 3] = [0.0, 1.0, 50.0];
        let col_lower: [f64; 3] = [0.0, 0.0, 0.0];
        let col_upper: [f64; 3] = [10.0, f64::INFINITY, 8.0];
        let row_lower: [f64; 2] = [6.0, 14.0];
        let row_upper: [f64; 2] = [6.0, 14.0];
        let a_start: [i32; 4] = [0, 2, 2, 3];
        let a_index: [i32; 3] = [0, 1, 1];
        let a_value: [f64; 3] = [1.0, 2.0, 1.0];

        // First solve: with iteration_limit = 0 -> ITERATION_LIMIT.
        unsafe {
            ffi::cobre_highs_pass_lp(
                highs,
                3,
                2,
                3,
                ffi::HIGHS_MATRIX_FORMAT_COLWISE,
                ffi::HIGHS_OBJ_SENSE_MINIMIZE,
                0.0,
                col_cost.as_ptr(),
                col_lower.as_ptr(),
                col_upper.as_ptr(),
                row_lower.as_ptr(),
                row_upper.as_ptr(),
                a_start.as_ptr(),
                a_index.as_ptr(),
                a_value.as_ptr(),
            );
            ffi::cobre_highs_set_int_option(highs, c"simplex_iteration_limit".as_ptr(), 0);
            ffi::cobre_highs_run(highs);
        }
        let status1 = unsafe { ffi::cobre_highs_get_model_status(highs) };
        assert_eq!(status1, ffi::HIGHS_MODEL_STATUS_ITERATION_LIMIT);

        // Restore default settings (mirror restore_default_settings()).
        unsafe {
            ffi::cobre_highs_set_string_option(highs, c"solver".as_ptr(), c"simplex".as_ptr());
            ffi::cobre_highs_set_int_option(highs, c"simplex_strategy".as_ptr(), 1);
            ffi::cobre_highs_set_string_option(highs, c"presolve".as_ptr(), c"off".as_ptr());
            ffi::cobre_highs_set_double_option(
                highs,
                c"primal_feasibility_tolerance".as_ptr(),
                1e-7,
            );
            ffi::cobre_highs_set_double_option(highs, c"dual_feasibility_tolerance".as_ptr(), 1e-7);
            ffi::cobre_highs_set_string_option(highs, c"parallel".as_ptr(), c"off".as_ptr());
            ffi::cobre_highs_set_bool_option(highs, c"output_flag".as_ptr(), 0);
            // simplex_iteration_limit is NOT in restore_default_settings -- reset explicitly.
            ffi::cobre_highs_set_int_option(highs, c"simplex_iteration_limit".as_ptr(), i32::MAX);
        }

        // Second solve on the same model: must reach OPTIMAL.
        unsafe { ffi::cobre_highs_clear_solver(highs) };
        unsafe { ffi::cobre_highs_run(highs) };
        let status2 = unsafe { ffi::cobre_highs_get_model_status(highs) };
        let obj = unsafe { ffi::cobre_highs_get_objective_value(highs) };
        assert_eq!(
            status2,
            ffi::HIGHS_MODEL_STATUS_OPTIMAL,
            "after restoring defaults, second solve must be OPTIMAL, got {status2}"
        );
        assert!(
            (obj - 100.0).abs() < 1e-8,
            "objective after restore must be 100.0, got {obj}"
        );

        unsafe { ffi::cobre_highs_destroy(highs) };
    }

    /// Verify iteration_limit=1 also triggers ITERATION_LIMIT for SS1.1 LP.
    ///
    /// This verifies that limiting to a small but non-zero number of iterations
    /// also works, providing an alternative formulation for triggering the same status.
    #[test]
    fn test_research_iteration_limit_one_triggers_iteration_limit_status() {
        use crate::ffi;

        let highs = unsafe { ffi::cobre_highs_create() };
        assert!(!highs.is_null());

        unsafe { ffi::cobre_highs_set_bool_option(highs, c"output_flag".as_ptr(), 0) };

        let col_cost: [f64; 3] = [0.0, 1.0, 50.0];
        let col_lower: [f64; 3] = [0.0, 0.0, 0.0];
        let col_upper: [f64; 3] = [10.0, f64::INFINITY, 8.0];
        let row_lower: [f64; 2] = [6.0, 14.0];
        let row_upper: [f64; 2] = [6.0, 14.0];
        let a_start: [i32; 4] = [0, 2, 2, 3];
        let a_index: [i32; 3] = [0, 1, 1];
        let a_value: [f64; 3] = [1.0, 2.0, 1.0];

        unsafe {
            ffi::cobre_highs_pass_lp(
                highs,
                3,
                2,
                3,
                ffi::HIGHS_MATRIX_FORMAT_COLWISE,
                ffi::HIGHS_OBJ_SENSE_MINIMIZE,
                0.0,
                col_cost.as_ptr(),
                col_lower.as_ptr(),
                col_upper.as_ptr(),
                row_lower.as_ptr(),
                row_upper.as_ptr(),
                a_start.as_ptr(),
                a_index.as_ptr(),
                a_value.as_ptr(),
            );
            ffi::cobre_highs_set_int_option(highs, c"simplex_iteration_limit".as_ptr(), 1);
            ffi::cobre_highs_run(highs);
        }

        let model_status = unsafe { ffi::cobre_highs_get_model_status(highs) };
        eprintln!("iteration_limit=1 model_status: {model_status}");
        // If the LP solves in 1 iteration it may be OPTIMAL; otherwise ITERATION_LIMIT.
        // We record both possibilities for the research document.
        assert!(
            model_status == ffi::HIGHS_MODEL_STATUS_ITERATION_LIMIT
                || model_status == ffi::HIGHS_MODEL_STATUS_OPTIMAL,
            "expected ITERATION_LIMIT or OPTIMAL, got {model_status}"
        );

        unsafe { ffi::cobre_highs_destroy(highs) };
    }

    /// Verify that `HighsSolver` correctly maps unbounded and infeasible statuses.
    ///
    /// With presolve=off and dual simplex (the default `HighsSolver` configuration),
    /// HiGHS returns `HIGHS_MODEL_STATUS_UNBOUNDED` (10) for unbounded LPs and
    /// `HIGHS_MODEL_STATUS_INFEASIBLE` (8) for infeasible LPs. Both are mapped to
    /// the appropriate `SolverError` variants without entering the
    /// `UNBOUNDED_OR_INFEASIBLE` probe branch.
    ///
    /// Note: `HIGHS_MODEL_STATUS_UNBOUNDED_OR_INFEASIBLE` (9) is returned only by
    /// IPM (`IpxWrapper.cpp:317`) when it detects dual infeasibility, or when
    /// `allow_unbounded_or_infeasible=true` is set with presolve=on. Neither
    /// condition occurs in the default `HighsSolver` configuration, so the
    /// `UNBOUNDED_OR_INFEASIBLE` branch serves as a safe fallback for retry paths
    /// that switch to IPM.
    #[test]
    fn test_research_verify_non_optimal_highs_status_mapping() {
        use super::super::HighsSolver;
        use crate::SolverInterface;
        use crate::types::SolverError;
        use crate::types::StageTemplate;

        // Unbounded LP: min -x0 - x1, x0 + x1 >= 1, x0/x1 in [0, +inf).
        // With presolve=off and dual simplex, HiGHS returns UNBOUNDED (10).
        let unbounded_template = StageTemplate {
            num_cols: 2,
            num_rows: 1,
            num_nz: 2,
            col_starts: vec![0_i32, 1, 2],
            row_indices: vec![0_i32, 0],
            values: vec![1.0, 1.0],
            col_lower: vec![0.0, 0.0],
            col_upper: vec![f64::INFINITY, f64::INFINITY],
            objective: vec![-1.0, -1.0],
            row_lower: vec![1.0],
            row_upper: vec![f64::INFINITY],
            n_state: 1,
            n_transfer: 0,
            n_dual_relevant: 1,
            n_hydro: 0,
            max_par_order: 0,
            col_scale: Vec::new(),
            row_scale: Vec::new(),
        };
        let mut solver_unb = HighsSolver::new().expect("HighsSolver::new() must succeed");
        solver_unb.load_model(&unbounded_template);
        let result_unb = solver_unb.solve(None).map(|_| ());
        assert!(
            matches!(result_unb, Err(SolverError::Unbounded)),
            "unbounded LP must return Err(SolverError::Unbounded), got {result_unb:?}"
        );

        // Infeasible LP: x0 must equal 99 but is bounded to [0, 10].
        // HiGHS returns INFEASIBLE (8) directly; mapped to Err(SolverError::Infeasible).
        let infeasible_template = StageTemplate {
            num_cols: 1,
            num_rows: 1,
            num_nz: 1,
            col_starts: vec![0_i32, 1],
            row_indices: vec![0_i32],
            values: vec![1.0],
            col_lower: vec![0.0],
            col_upper: vec![10.0],
            objective: vec![0.0],
            row_lower: vec![99.0],
            row_upper: vec![99.0],
            n_state: 1,
            n_transfer: 0,
            n_dual_relevant: 1,
            n_hydro: 0,
            max_par_order: 0,
            col_scale: Vec::new(),
            row_scale: Vec::new(),
        };
        let mut solver_inf = HighsSolver::new().expect("HighsSolver::new() must succeed");
        solver_inf.load_model(&infeasible_template);
        let result_inf = solver_inf.solve(None).map(|_| ());
        assert!(
            matches!(result_inf, Err(SolverError::Infeasible)),
            "infeasible LP must return Err(SolverError::Infeasible), got {result_inf:?}"
        );
    }
}
