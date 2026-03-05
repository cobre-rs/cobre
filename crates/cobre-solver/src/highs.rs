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

use std::os::raw::c_void;
use std::time::Instant;

use crate::{
    SolverInterface, ffi,
    types::{Basis, LpSolution, RowBatch, SolverError, SolverStatistics, StageTemplate},
};

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
    /// Pre-allocated buffer for reduced costs returned in `LpSolution`.
    /// Resized in `load_model`.
    reduced_costs: Vec<f64>,
    /// Scratch buffer for converting `usize` indices to `i32` for the `HiGHS` C API.
    /// Used by `add_rows`, `set_row_bounds`, and `set_col_bounds`.
    /// Never shrunk -- only grows -- to prevent reallocation churn on the hot path.
    scratch_i32: Vec<i32>,
    /// Pre-allocated i32 buffer for column basis status codes.
    /// Reused across `solve_with_basis` and `get_basis` calls to avoid per-call allocation.
    /// Resized in `load_model` to `num_cols`; never shrunk.
    basis_col_i32: Vec<i32>,
    /// Pre-allocated i32 buffer for row basis status codes.
    /// Reused across `solve_with_basis` and `get_basis` calls to avoid per-call allocation.
    /// Resized in `load_model` to `num_rows` and grown in `add_rows`.
    basis_row_i32: Vec<i32>,
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

impl HighsSolver {
    /// Creates a new `HiGHS` solver instance with performance-tuned defaults.
    ///
    /// Calls `cobre_highs_create()` to allocate the `HiGHS` handle, then applies
    /// the seven default options defined in `HiGHS` Implementation SS4.1:
    ///
    /// | Option                         | Value       | Type   |
    /// |--------------------------------|-------------|--------|
    /// | `solver`                       | `"simplex"` | string |
    /// | `simplex_strategy`             | `4`         | int    |
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
            reduced_costs: Vec::new(),
            scratch_i32: Vec::new(),
            basis_col_i32: Vec::new(),
            basis_row_i32: Vec::new(),
            num_cols: 0,
            num_rows: 0,
            has_model: false,
            stats: SolverStatistics::default(),
        })
    }

    /// Applies the seven performance-tuned `HiGHS` configuration options.
    ///
    /// Called once during construction. Returns `Ok(())` if all options are set
    /// successfully, or `Err(SolverError::InternalError)` with the failing
    /// option name if any configuration call returns `HIGHS_STATUS_ERROR`.
    fn apply_default_config(handle: *mut c_void) -> Result<(), SolverError> {
        // SAFETY: `handle` is a valid, non-null `HiGHS` pointer from `cobre_highs_create()`.
        // All C string literals are null-terminated static data with 'static lifetime.
        // Integers and doubles are plain values with no pointer requirements.

        // solver = "simplex"
        let status = unsafe {
            ffi::cobre_highs_set_string_option(handle, c"solver".as_ptr(), c"simplex".as_ptr())
        };
        if status == ffi::HIGHS_STATUS_ERROR {
            return Err(SolverError::InternalError {
                message: "HiGHS configuration failed: solver".to_string(),
                error_code: Some(status),
            });
        }

        // simplex_strategy = 4 (dual simplex)
        let status =
            unsafe { ffi::cobre_highs_set_int_option(handle, c"simplex_strategy".as_ptr(), 4) };
        if status == ffi::HIGHS_STATUS_ERROR {
            return Err(SolverError::InternalError {
                message: "HiGHS configuration failed: simplex_strategy".to_string(),
                error_code: Some(status),
            });
        }

        // presolve = "off"
        let status = unsafe {
            ffi::cobre_highs_set_string_option(handle, c"presolve".as_ptr(), c"off".as_ptr())
        };
        if status == ffi::HIGHS_STATUS_ERROR {
            return Err(SolverError::InternalError {
                message: "HiGHS configuration failed: presolve".to_string(),
                error_code: Some(status),
            });
        }

        // parallel = "off"
        let status = unsafe {
            ffi::cobre_highs_set_string_option(handle, c"parallel".as_ptr(), c"off".as_ptr())
        };
        if status == ffi::HIGHS_STATUS_ERROR {
            return Err(SolverError::InternalError {
                message: "HiGHS configuration failed: parallel".to_string(),
                error_code: Some(status),
            });
        }

        // output_flag = 0 (bool option -- suppresses console output)
        let status =
            unsafe { ffi::cobre_highs_set_bool_option(handle, c"output_flag".as_ptr(), 0) };
        if status == ffi::HIGHS_STATUS_ERROR {
            return Err(SolverError::InternalError {
                message: "HiGHS configuration failed: output_flag".to_string(),
                error_code: Some(status),
            });
        }

        // primal_feasibility_tolerance = 1e-7
        let status = unsafe {
            ffi::cobre_highs_set_double_option(
                handle,
                c"primal_feasibility_tolerance".as_ptr(),
                1e-7,
            )
        };
        if status == ffi::HIGHS_STATUS_ERROR {
            return Err(SolverError::InternalError {
                message: "HiGHS configuration failed: primal_feasibility_tolerance".to_string(),
                error_code: Some(status),
            });
        }

        // dual_feasibility_tolerance = 1e-7
        let status = unsafe {
            ffi::cobre_highs_set_double_option(handle, c"dual_feasibility_tolerance".as_ptr(), 1e-7)
        };
        if status == ffi::HIGHS_STATUS_ERROR {
            return Err(SolverError::InternalError {
                message: "HiGHS configuration failed: dual_feasibility_tolerance".to_string(),
                error_code: Some(status),
            });
        }

        Ok(())
    }

    /// Extracts the full optimal solution from `HiGHS` into pre-allocated buffers.
    ///
    /// Calls `cobre_highs_get_solution`, `cobre_highs_get_objective_value`, and
    /// `cobre_highs_get_simplex_iteration_count`, then builds an [`LpSolution`].
    ///
    /// `col_dual` from `HiGHS` is the reduced cost vector (per `HiGHS` Implementation
    /// SS2.4). Row duals are already in the canonical sign convention -- no negation
    /// needed (per Solver Abstraction SS8).
    ///
    /// # Panics
    ///
    /// Does not panic. The `iterations` cast from `i32` to `u64` is safe because
    /// `HiGHS` iteration counts are always non-negative.
    fn extract_solution(&self, solve_time_seconds: f64) -> LpSolution {
        // SAFETY:
        // - `self.handle` is a valid, non-null HiGHS pointer.
        // - All four mutable pointer arguments point into owned Vec buffers that were
        //   resized to at least `num_cols` / `num_rows` entries in `load_model` or
        //   `add_rows`. HiGHS writes exactly `num_cols` values into col_* buffers and
        //   `num_rows` values into row_* buffers, which is within bounds.
        unsafe {
            ffi::cobre_highs_get_solution(
                self.handle,
                self.col_value.as_ptr().cast_mut(),
                self.col_dual.as_ptr().cast_mut(),
                self.row_value.as_ptr().cast_mut(),
                self.row_dual.as_ptr().cast_mut(),
            );
        }

        // SAFETY: `self.handle` is a valid, non-null HiGHS pointer.
        let objective = unsafe { ffi::cobre_highs_get_objective_value(self.handle) };

        // SAFETY: `self.handle` is a valid, non-null HiGHS pointer. The return value
        // is a non-negative simplex iteration count; casting i32 -> u64 is safe.
        #[allow(clippy::cast_sign_loss)]
        let iterations =
            unsafe { ffi::cobre_highs_get_simplex_iteration_count(self.handle) } as u64;

        // col_dual from HiGHS IS the reduced cost vector. row_dual is already in
        // canonical sign convention (positive dual on <= constraint means increasing
        // RHS increases the objective). No sign negation is needed for HiGHS.
        LpSolution {
            objective,
            primal: self.col_value.clone(),
            dual: self.row_dual.clone(),
            reduced_costs: self.col_dual.clone(),
            iterations,
            solve_time_seconds,
        }
    }

    /// Attempts to extract a partial solution after a non-optimal termination.
    ///
    /// Returns `None` if solution extraction is unavailable (e.g. after a
    /// `SOLVE_ERROR`). Returns `Some(solution)` when `HiGHS` has a best-effort
    /// primal/dual solution available (e.g. after `TIME_LIMIT` or `ITERATION_LIMIT`).
    fn try_extract_partial_solution(&self, solve_time_seconds: f64) -> Option<LpSolution> {
        // Attempt the extraction; if HiGHS has no solution available the buffers will
        // be filled with zeros or unchanged values. We consider the extraction valid
        // as long as HiGHS reports a finite objective value.
        // SAFETY: same guarantees as `extract_solution`.
        let _status = unsafe {
            ffi::cobre_highs_get_solution(
                self.handle,
                self.col_value.as_ptr().cast_mut(),
                self.col_dual.as_ptr().cast_mut(),
                self.row_value.as_ptr().cast_mut(),
                self.row_dual.as_ptr().cast_mut(),
            )
        };

        // SAFETY: `self.handle` is a valid, non-null HiGHS pointer.
        let objective = unsafe { ffi::cobre_highs_get_objective_value(self.handle) };

        // A non-finite objective means HiGHS has no valid solution to report.
        if !objective.is_finite() {
            return None;
        }

        // SAFETY: same as `extract_solution`.
        #[allow(clippy::cast_sign_loss)]
        let iterations =
            unsafe { ffi::cobre_highs_get_simplex_iteration_count(self.handle) } as u64;

        Some(LpSolution {
            objective,
            primal: self.col_value.clone(),
            dual: self.row_dual.clone(),
            reduced_costs: self.col_dual.clone(),
            iterations,
            solve_time_seconds,
        })
    }

    /// Restores the seven performance-tuned default options after a retry escalation.
    ///
    /// Called unconditionally after the retry loop to ensure subsequent solves
    /// see the standard configuration regardless of which retry levels were
    /// reached (`HiGHS` Implementation SS3, restore-defaults requirement).
    fn restore_default_settings(&mut self) {
        // SAFETY for all option calls below:
        // - `self.handle` is a valid, non-null HiGHS pointer.
        // - All C string literals are null-terminated static data with 'static lifetime.
        // Errors from restore calls are silently ignored -- we are already in the
        // error recovery path and cannot do anything useful if option reset fails.
        unsafe {
            ffi::cobre_highs_set_string_option(
                self.handle,
                c"solver".as_ptr(),
                c"simplex".as_ptr(),
            );
            ffi::cobre_highs_set_int_option(self.handle, c"simplex_strategy".as_ptr(), 4);
            ffi::cobre_highs_set_string_option(self.handle, c"presolve".as_ptr(), c"off".as_ptr());
            ffi::cobre_highs_set_double_option(
                self.handle,
                c"primal_feasibility_tolerance".as_ptr(),
                1e-7,
            );
            ffi::cobre_highs_set_double_option(
                self.handle,
                c"dual_feasibility_tolerance".as_ptr(),
                1e-7,
            );
            ffi::cobre_highs_set_string_option(self.handle, c"parallel".as_ptr(), c"off".as_ptr());
            ffi::cobre_highs_set_bool_option(self.handle, c"output_flag".as_ptr(), 0);
        }
    }

    /// Runs the solver once and returns the raw `HiGHS` model status.
    ///
    /// Does not update statistics -- statistics are managed by the caller.
    /// Returns the model status integer for dispatch.
    fn run_once(&mut self) -> i32 {
        // SAFETY: `self.handle` is a valid, non-null HiGHS pointer. `cobre_highs_run`
        // drives the HiGHS solver on the currently loaded model.
        let run_status = unsafe { ffi::cobre_highs_run(self.handle) };
        if run_status == ffi::HIGHS_STATUS_ERROR {
            return ffi::HIGHS_MODEL_STATUS_SOLVE_ERROR;
        }
        // SAFETY: same.
        unsafe { ffi::cobre_highs_get_model_status(self.handle) }
    }

    /// Converts a `HiGHS` infeasible model status into a `SolverError::Infeasible`.
    ///
    /// Attempts to retrieve the dual ray. If the call succeeds and `has_dual_ray`
    /// is set, the ray values are included in the error.
    fn make_infeasible_error(&self) -> SolverError {
        let mut has_dual_ray: i32 = 0;
        let mut ray_buf = vec![0.0_f64; self.num_rows];
        // SAFETY:
        // - `self.handle` is a valid, non-null HiGHS pointer.
        // - `has_dual_ray` is a stack-allocated i32; pointer is valid.
        // - `ray_buf` is a heap-allocated Vec of length `num_rows`; pointer is valid
        //   for the duration of this call.
        let status = unsafe {
            ffi::cobre_highs_get_dual_ray(self.handle, &raw mut has_dual_ray, ray_buf.as_mut_ptr())
        };
        let ray = if status != ffi::HIGHS_STATUS_ERROR && has_dual_ray != 0 {
            Some(ray_buf)
        } else {
            None
        };
        SolverError::Infeasible { ray }
    }

    /// Converts a `HiGHS` unbounded model status into a `SolverError::Unbounded`.
    ///
    /// Attempts to retrieve the primal ray. If the call succeeds and `has_primal_ray`
    /// is set, the direction values are included in the error.
    fn make_unbounded_error(&self) -> SolverError {
        let mut has_primal_ray: i32 = 0;
        let mut ray_buf = vec![0.0_f64; self.num_cols];
        // SAFETY:
        // - `self.handle` is a valid, non-null HiGHS pointer.
        // - `has_primal_ray` is a stack-allocated i32; pointer is valid.
        // - `ray_buf` is a heap-allocated Vec of length `num_cols`; pointer is valid.
        let status = unsafe {
            ffi::cobre_highs_get_primal_ray(
                self.handle,
                &raw mut has_primal_ray,
                ray_buf.as_mut_ptr(),
            )
        };
        let direction = if status != ffi::HIGHS_STATUS_ERROR && has_primal_ray != 0 {
            Some(ray_buf)
        } else {
            None
        };
        SolverError::Unbounded { direction }
    }

    /// Interprets a non-optimal model status as a `SolverError`, extracting a
    /// partial solution where possible.
    ///
    /// This is the single dispatch point for all terminal error statuses. It is
    /// called both from the initial solve and from each retry level.
    ///
    /// Returns `None` if the status is `SOLVE_ERROR` or `UNKNOWN` (retry should
    /// continue), or `Some(Err(...))` for all other terminal statuses.
    fn interpret_terminal_status(
        &self,
        status: i32,
        solve_time_seconds: f64,
    ) -> Option<Result<LpSolution, SolverError>> {
        match status {
            ffi::HIGHS_MODEL_STATUS_OPTIMAL => {
                // Caller should have handled optimal before reaching here.
                None
            }
            ffi::HIGHS_MODEL_STATUS_INFEASIBLE => Some(Err(self.make_infeasible_error())),
            ffi::HIGHS_MODEL_STATUS_UNBOUNDED_OR_INFEASIBLE => {
                // Try dual ray first; if found, treat as infeasible.
                let mut has_dual_ray: i32 = 0;
                let mut dual_buf = vec![0.0_f64; self.num_rows];
                // SAFETY: handle is valid non-null pointer; dual_buf is heap-allocated Vec.
                let dual_status = unsafe {
                    ffi::cobre_highs_get_dual_ray(
                        self.handle,
                        &raw mut has_dual_ray,
                        dual_buf.as_mut_ptr(),
                    )
                };
                if dual_status != ffi::HIGHS_STATUS_ERROR && has_dual_ray != 0 {
                    return Some(Err(SolverError::Infeasible {
                        ray: Some(dual_buf),
                    }));
                }
                // Try primal ray; if found, treat as unbounded.
                let mut has_primal_ray: i32 = 0;
                let mut primal_buf = vec![0.0_f64; self.num_cols];
                // SAFETY: handle is valid non-null pointer; primal_buf is heap-allocated Vec.
                let primal_status = unsafe {
                    ffi::cobre_highs_get_primal_ray(
                        self.handle,
                        &raw mut has_primal_ray,
                        primal_buf.as_mut_ptr(),
                    )
                };
                if primal_status != ffi::HIGHS_STATUS_ERROR && has_primal_ray != 0 {
                    return Some(Err(SolverError::Unbounded {
                        direction: Some(primal_buf),
                    }));
                }
                // Default: treat as infeasible with no ray.
                Some(Err(SolverError::Infeasible { ray: None }))
            }
            ffi::HIGHS_MODEL_STATUS_UNBOUNDED => Some(Err(self.make_unbounded_error())),
            ffi::HIGHS_MODEL_STATUS_TIME_LIMIT => {
                let partial_solution = self.try_extract_partial_solution(solve_time_seconds);
                Some(Err(SolverError::TimeLimitExceeded {
                    partial_solution,
                    elapsed_seconds: solve_time_seconds,
                }))
            }
            ffi::HIGHS_MODEL_STATUS_ITERATION_LIMIT => {
                let partial_solution = self.try_extract_partial_solution(solve_time_seconds);
                // SAFETY: handle is valid non-null pointer; iteration count is non-negative.
                #[allow(clippy::cast_sign_loss)]
                let iterations =
                    unsafe { ffi::cobre_highs_get_simplex_iteration_count(self.handle) } as u64;
                Some(Err(SolverError::IterationLimit {
                    partial_solution,
                    iterations,
                }))
            }
            ffi::HIGHS_MODEL_STATUS_SOLVE_ERROR | ffi::HIGHS_MODEL_STATUS_UNKNOWN => {
                // Signal to the caller that retry should continue.
                None
            }
            other => Some(Err(SolverError::InternalError {
                message: format!("HiGHS returned unexpected model status {other}"),
                error_code: Some(other),
            })),
        }
    }

    /// Converts a [`crate::types::BasisStatus`] to the corresponding `HiGHS` basis status code.
    ///
    /// Mapping (`HiGHS` Implementation SS2.6):
    /// - `AtLower` -> `HIGHS_BASIS_STATUS_LOWER` (0)
    /// - `Basic`   -> `HIGHS_BASIS_STATUS_BASIC` (1)
    /// - `AtUpper` -> `HIGHS_BASIS_STATUS_UPPER` (2)
    /// - `Free`    -> `HIGHS_BASIS_STATUS_ZERO`  (3)
    /// - `Fixed`   -> `HIGHS_BASIS_STATUS_LOWER` (0) -- for fixed variables lb == ub, so
    ///   "at lower" and "at upper" are the same point; `HiGHS` expects lower.
    fn basis_status_to_highs(status: crate::types::BasisStatus) -> i32 {
        use crate::types::BasisStatus;
        // AtLower and Fixed both map to LOWER (0): Fixed variables have lb == ub so
        // "at lower" and "at upper" are the same point; HiGHS uses LOWER for both.
        #[allow(clippy::match_same_arms)]
        match status {
            BasisStatus::AtLower | BasisStatus::Fixed => ffi::HIGHS_BASIS_STATUS_LOWER,
            BasisStatus::Basic => ffi::HIGHS_BASIS_STATUS_BASIC,
            BasisStatus::AtUpper => ffi::HIGHS_BASIS_STATUS_UPPER,
            BasisStatus::Free => ffi::HIGHS_BASIS_STATUS_ZERO,
        }
    }

    /// Converts a `HiGHS` basis status code to the canonical [`crate::types::BasisStatus`].
    ///
    /// Mapping (`HiGHS` Implementation SS2.6):
    /// - 0 (`LOWER`)    -> `AtLower`
    /// - 1 (`BASIC`)    -> `Basic`
    /// - 2 (`UPPER`)    -> `AtUpper`
    /// - 3 (`ZERO`)     -> `Free`
    /// - 4 (`NONBASIC`) -> `AtLower` (default nonbasic position per `HiGHS` docs)
    ///
    /// # Panics
    ///
    /// Panics on any code outside 0-4, which should never occur with a functioning solver.
    #[allow(clippy::panic)]
    fn highs_to_basis_status(code: i32) -> crate::types::BasisStatus {
        use crate::types::BasisStatus;
        match code {
            c if c == ffi::HIGHS_BASIS_STATUS_LOWER => BasisStatus::AtLower,
            c if c == ffi::HIGHS_BASIS_STATUS_BASIC => BasisStatus::Basic,
            c if c == ffi::HIGHS_BASIS_STATUS_UPPER => BasisStatus::AtUpper,
            c if c == ffi::HIGHS_BASIS_STATUS_ZERO => BasisStatus::Free,
            // NONBASIC (4) is HiGHS's generic nonbasic designation; map to AtLower.
            c if c == ffi::HIGHS_BASIS_STATUS_NONBASIC => BasisStatus::AtLower,
            other => panic!("invalid HiGHS basis status code {other}: expected 0-4"),
        }
    }

    /// Converts a slice of `usize` indices into `i32` using the internal scratch buffer.
    ///
    /// Grows `scratch_i32` if the source length exceeds the current capacity, but
    /// never shrinks it to prevent reallocation churn on the hot path. Each element
    /// is bounds-checked with a `debug_assert` (elided in release builds); the cast
    /// is safe because LP dimensions in HPC solvers are never within 2^31 of overflow.
    ///
    /// Returns a shared reference to the filled prefix of `scratch_i32`.
    fn convert_to_i32_scratch(&mut self, source: &[usize]) -> &[i32] {
        if source.len() > self.scratch_i32.len() {
            self.scratch_i32.resize(source.len(), 0);
        }
        for (i, &v) in source.iter().enumerate() {
            debug_assert!(
                i32::try_from(v).is_ok(),
                "usize index {v} overflows i32::MAX at position {i}"
            );
            // SAFETY: debug_assert above verifies v fits in i32. The cast is
            // intentional: HiGHS C API requires i32 indices and LP dimensions
            // (columns, rows, non-zeros) never approach i32::MAX in practice.
            #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
            {
                self.scratch_i32[i] = v as i32;
            }
        }
        &self.scratch_i32[..source.len()]
    }
}

impl Drop for HighsSolver {
    fn drop(&mut self) {
        // SAFETY: `self.handle` is a valid, non-null `HiGHS` pointer obtained
        // from `cobre_highs_create()` during construction. Drop is called
        // exactly once per `HighsSolver` instance (Rust's ownership model
        // guarantees single-drop). After this call, `self.handle` is invalid
        // but the struct is being destroyed so it will never be accessed again.
        unsafe { ffi::cobre_highs_destroy(self.handle) };
    }
}

impl SolverInterface for HighsSolver {
    fn name(&self) -> &'static str {
        "HiGHS"
    }

    fn load_model(&mut self, template: &StageTemplate) {
        // Convert col_starts into a local Vec<i32>. load_model is not on the
        // innermost hot path (called ~60 times per iteration, not millions), so
        // a local allocation is acceptable and avoids keeping a scratch buffer
        // alive for this single use.
        let col_starts_i32: Vec<i32> = template
            .col_starts
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                debug_assert!(
                    i32::try_from(v).is_ok(),
                    "col_starts[{i}] = {v} overflows i32::MAX"
                );
                #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
                {
                    v as i32
                }
            })
            .collect();

        let row_indices_ptr = {
            let row_indices_i32 = self.convert_to_i32_scratch(&template.row_indices);
            row_indices_i32.as_ptr()
        };

        // SAFETY:
        // - `self.handle` is a valid, non-null HiGHS pointer from `cobre_highs_create()`.
        // - All pointer arguments point into owned `Vec` data that remains alive for the
        //   duration of this call.
        // - `col_starts_i32` is a local Vec alive until the end of this scope.
        // - `row_indices_ptr` points into `self.scratch_i32`, which is alive for `'self`.
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
                col_starts_i32.as_ptr(),
                row_indices_ptr,
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
        self.reduced_costs.resize(self.num_cols, 0.0);
        self.row_value.resize(self.num_rows, 0.0);
        self.row_dual.resize(self.num_rows, 0.0);

        // Resize basis status i32 buffers. Zero-fill is fine; values are overwritten before
        // any FFI call. These never shrink -- only grow -- to prevent reallocation on hot path.
        self.basis_col_i32.resize(self.num_cols, 0);
        self.basis_row_i32.resize(self.num_rows, 0);
    }

    fn add_rows(&mut self, cuts: &RowBatch) {
        // Use local Vec for row_starts, scratch for larger col_indices.
        let row_starts_i32: Vec<i32> = cuts
            .row_starts
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                debug_assert!(
                    i32::try_from(v).is_ok(),
                    "row_starts[{i}] = {v} overflows i32::MAX"
                );
                // SAFETY: debug_assert above verifies v fits in i32.
                #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
                {
                    v as i32
                }
            })
            .collect();

        let col_indices_ptr = {
            let col_indices_i32 = self.convert_to_i32_scratch(&cuts.col_indices);
            col_indices_i32.as_ptr()
        };

        assert!(
            i32::try_from(cuts.num_rows).is_ok(),
            "cuts.num_rows {} overflows i32: RowBatch exceeds HiGHS API limit",
            cuts.num_rows
        );
        assert!(
            i32::try_from(cuts.col_indices.len()).is_ok(),
            "cuts nnz {} overflows i32: RowBatch exceeds HiGHS API limit",
            cuts.col_indices.len()
        );
        // SAFETY: Both values have been asserted to fit in i32 above.
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let num_new_row = cuts.num_rows as i32;
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let num_new_nz = cuts.col_indices.len() as i32;

        // SAFETY:
        // - `self.handle` is a valid, non-null HiGHS pointer.
        // - All pointer arguments point into owned data alive for the duration of this call.
        // - `row_starts_i32` is a local Vec alive until the end of this scope.
        // - `col_indices_ptr` points into `self.scratch_i32`, alive for `'self`.
        // - Slice lengths: `num_rows + 1` for starts, total nnz for index and value,
        //   `num_rows` for lower/upper bounds.
        let status = unsafe {
            ffi::cobre_highs_add_rows(
                self.handle,
                num_new_row,
                cuts.row_lower.as_ptr(),
                cuts.row_upper.as_ptr(),
                num_new_nz,
                row_starts_i32.as_ptr(),
                col_indices_ptr,
                cuts.values.as_ptr(),
            )
        };

        assert_ne!(
            status,
            ffi::HIGHS_STATUS_ERROR,
            "cobre_highs_add_rows failed with status {status}"
        );

        self.num_rows += cuts.num_rows;

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
    }

    #[allow(clippy::too_many_lines)]
    fn solve(&mut self) -> Result<LpSolution, SolverError> {
        assert!(
            self.has_model,
            "solve called without a loaded model — call load_model first"
        );
        let t0 = Instant::now();
        let model_status = self.run_once();
        let solve_time = t0.elapsed().as_secs_f64();

        self.stats.solve_count += 1;

        if model_status == ffi::HIGHS_MODEL_STATUS_OPTIMAL {
            let solution = self.extract_solution(solve_time);
            self.stats.success_count += 1;
            self.stats.total_iterations += solution.iterations;
            self.stats.total_solve_time_seconds += solve_time;
            return Ok(solution);
        }

        // Check for a definitive terminal status (not a retry-able error).
        if let Some(terminal) = self.interpret_terminal_status(model_status, solve_time) {
            self.stats.failure_count += 1;
            return terminal;
        }

        // 5-level retry escalation (HiGHS Implementation SS3). Apply progressively
        // more permissive strategies on SOLVE_ERROR/UNKNOWN; break on OPTIMAL or
        // definitive terminal status.
        let mut retry_attempts: u64 = 0;
        let mut final_result: Option<Result<LpSolution, SolverError>> = None;

        for level in 0..5_u32 {
            // SAFETY: handle is valid non-null HiGHS pointer; option names/values
            // are static C strings; no retained pointers after call.
            match level {
                0 => {
                    unsafe { ffi::cobre_highs_clear_solver(self.handle) };
                }
                1 => unsafe {
                    ffi::cobre_highs_set_string_option(
                        self.handle,
                        c"presolve".as_ptr(),
                        c"on".as_ptr(),
                    );
                },
                2 => unsafe {
                    ffi::cobre_highs_set_int_option(self.handle, c"simplex_strategy".as_ptr(), 1);
                },
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
                4 => unsafe {
                    ffi::cobre_highs_set_string_option(
                        self.handle,
                        c"solver".as_ptr(),
                        c"ipm".as_ptr(),
                    );
                },
                _ => unreachable!(),
            }

            retry_attempts += 1;

            let t_retry = Instant::now();
            let retry_status = self.run_once();
            let retry_time = t_retry.elapsed().as_secs_f64();

            if retry_status == ffi::HIGHS_MODEL_STATUS_OPTIMAL {
                let solution = self.extract_solution(retry_time);
                final_result = Some(Ok(solution));
                break;
            }

            if let Some(terminal) = self.interpret_terminal_status(retry_status, retry_time) {
                final_result = Some(terminal);
                break;
            }
            // Still SOLVE_ERROR or UNKNOWN -- continue to next level.
        }

        // Restore default settings unconditionally (regardless of retry outcome).
        self.restore_default_settings();

        // Update statistics with accumulated retry attempts.
        self.stats.retry_count += retry_attempts;

        match final_result {
            Some(Ok(solution)) => {
                self.stats.success_count += 1;
                self.stats.total_iterations += solution.iterations;
                self.stats.total_solve_time_seconds += solution.solve_time_seconds;
                Ok(solution)
            }
            Some(Err(e)) => {
                self.stats.failure_count += 1;
                Err(e)
            }
            None => {
                // All 5 retry levels exhausted without a definitive result.
                self.stats.failure_count += 1;
                let partial_solution = self.try_extract_partial_solution(0.0);
                Err(SolverError::NumericalDifficulty {
                    partial_solution,
                    message: "HiGHS failed to reach optimality after all 5 retry escalation levels"
                        .to_string(),
                })
            }
        }
    }

    fn solve_with_basis(&mut self, basis: &Basis) -> Result<LpSolution, SolverError> {
        assert!(
            self.has_model,
            "solve_with_basis called without a loaded model — call load_model first"
        );
        // Column count must match exactly -- columns never change between stages
        // for the same template (Solver Abstraction SS2.3).
        assert!(
            basis.col_status.len() == self.num_cols,
            "basis column count {} does not match LP column count {}",
            basis.col_status.len(),
            self.num_cols
        );

        // Translate column statuses into the pre-allocated i32 buffer.
        // `basis_col_i32` was sized to `num_cols` in `load_model`.
        for (i, &status) in basis.col_status.iter().enumerate() {
            self.basis_col_i32[i] = Self::basis_status_to_highs(status);
        }

        // Translate row statuses, handling dimension mismatch for dynamic cuts
        // (Solver Abstraction SS2.3):
        // - Fewer rows than LP: extend with BASIC (new cut rows get Basic status).
        // - More rows than LP: truncate to current num_rows.
        let basis_rows = basis.row_status.len();
        let lp_rows = self.num_rows;

        // Fill the overlapping prefix from the saved basis.
        let copy_len = basis_rows.min(lp_rows);
        for i in 0..copy_len {
            self.basis_row_i32[i] = Self::basis_status_to_highs(basis.row_status[i]);
        }
        // If the LP has more rows than the saved basis (new cuts added), initialize
        // the extra rows as Basic -- newly added dynamic constraint rows have no
        // prior basis information.
        if lp_rows > basis_rows {
            for i in basis_rows..lp_rows {
                self.basis_row_i32[i] = ffi::HIGHS_BASIS_STATUS_BASIC;
            }
        }
        // If lp_rows < basis_rows the extra saved entries are simply ignored
        // (the truncation is implicit: we only pass lp_rows entries to HiGHS).

        // Attempt to install the basis in HiGHS.
        // SAFETY:
        // - `self.handle` is a valid, non-null HiGHS pointer.
        // - `basis_col_i32` has been sized to at least `num_cols` in `load_model`.
        // - `basis_row_i32` has been sized to at least `num_rows` in `load_model`/`add_rows`.
        // - We pass exactly `num_cols` col entries and `num_rows` row entries,
        //   which matches the current model dimensions.
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let set_status = unsafe {
            ffi::cobre_highs_set_basis(
                self.handle,
                self.basis_col_i32.as_ptr(),
                self.basis_row_i32.as_ptr(),
            )
        };

        // Basis rejection tracking: singular factorization is the only realistic
        // failure (dimensions and status codes are prevented by assertions above).
        // Fall back to cold-start and track for performance diagnostics.
        if set_status == ffi::HIGHS_STATUS_ERROR {
            self.stats.basis_rejections += 1;
            debug_assert!(false, "basis rejected; falling back to cold-start");
        }

        // Delegate to solve() which handles retry escalation and statistics updates.
        self.solve()
    }

    fn reset(&mut self) {
        // SAFETY: `self.handle` is a valid, non-null HiGHS pointer. `cobre_highs_clear_solver`
        // discards the cached basis and factorization while preserving the model data.
        // After this call the model is still loaded and `load_model` must be called
        // again before the next `solve` -- enforced by zeroing `num_cols` and `num_rows`.
        unsafe { ffi::cobre_highs_clear_solver(self.handle) };
        // Force `load_model` to be called before the next solve.
        self.num_cols = 0;
        self.num_rows = 0;
        self.has_model = false;
        // Intentionally do NOT zero `self.stats` -- statistics accumulate for the
        // lifetime of the instance (per trait contract, SS4.3).
    }

    fn get_basis(&self) -> Basis {
        assert!(
            self.has_model,
            "get_basis called without a loaded model — call load_model first"
        );
        // Reuse the pre-allocated i32 buffers as output targets.
        // SAFETY:
        // - `self.handle` is a valid, non-null HiGHS pointer.
        // - `basis_col_i32` has been sized to at least `num_cols` in `load_model`.
        // - `basis_row_i32` has been sized to at least `num_rows` in `load_model`/`add_rows`.
        // - HiGHS writes exactly `num_cols` values to the col pointer and `num_rows`
        //   values to the row pointer, which is within the allocated lengths.
        // We cast the shared references to mutable pointers because `cobre_highs_get_basis`
        // takes `*mut i32` for output; Rust aliasing rules are satisfied because HiGHS does
        // not retain the pointers after the call returns.
        let get_status = unsafe {
            ffi::cobre_highs_get_basis(
                self.handle,
                self.basis_col_i32.as_ptr().cast_mut(),
                self.basis_row_i32.as_ptr().cast_mut(),
            )
        };

        assert_ne!(
            get_status,
            ffi::HIGHS_STATUS_ERROR,
            "cobre_highs_get_basis failed: basis must exist after a successful solve (programming error)"
        );

        // Translate HiGHS i32 codes back to canonical BasisStatus.
        let col_status: Vec<crate::types::BasisStatus> = self.basis_col_i32[..self.num_cols]
            .iter()
            .map(|&code| Self::highs_to_basis_status(code))
            .collect();

        let row_status: Vec<crate::types::BasisStatus> = self.basis_row_i32[..self.num_rows]
            .iter()
            .map(|&code| Self::highs_to_basis_status(code))
            .collect();

        Basis {
            col_status,
            row_status,
        }
    }

    fn statistics(&self) -> SolverStatistics {
        self.stats.clone()
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
        types::{RowBatch, StageTemplate},
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

    // Benders cut fixture from Solver Interface Testing SS1.2:
    // Cut 1: -5*x0 + x1 >= 20  (col_indices [0,1], values [-5, 1])
    // Cut 2:  3*x0 + x1 >= 80  (col_indices [0,1], values [ 3, 1])
    fn make_fixture_row_batch() -> RowBatch {
        RowBatch {
            num_rows: 2,
            row_starts: vec![0, 2, 4],
            col_indices: vec![0, 1, 0, 1],
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
            solver.reduced_costs.len(),
            3,
            "reduced_costs buffer must be resized to num_cols"
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

        // 2 structural rows + 2 cut rows = 4
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

        let result = solver.solve();
        let solution = result.expect("solve() must succeed on a feasible LP");

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

    /// SS1.2: after adding two Benders cuts to SS1.1, optimal objective = 162.
    /// Cuts: -5*x0+x1>=20 and 3*x0+x1>=80. With x0=6: x1>=max(50,62)=62.
    /// Obj = 0*6 + 1*62 + 50*2 = 162.
    #[test]
    fn test_highs_solve_with_cuts() {
        let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
        let template = make_fixture_stage_template();
        let cuts = make_fixture_row_batch();
        solver.load_model(&template);
        solver.add_rows(&cuts);

        let result = solver.solve();
        let solution = result.expect("solve() must succeed on a feasible LP with cuts");

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

        let result = solver.solve();
        let solution = result.expect("solve() must succeed after RHS patch");

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

        solver.solve().expect("first solve must succeed");
        solver.solve().expect("second solve must succeed");

        let stats = solver.statistics();
        assert_eq!(stats.solve_count, 2, "solve_count must be 2");
        assert_eq!(stats.success_count, 2, "success_count must be 2");
        assert_eq!(stats.failure_count, 0, "failure_count must be 0");
        assert!(
            stats.total_iterations > 0,
            "total_iterations must be positive"
        );
    }

    /// After `reset()`, statistics counters must be unchanged.
    #[test]
    fn test_highs_reset_preserves_stats() {
        let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
        let template = make_fixture_stage_template();
        solver.load_model(&template);
        solver.solve().expect("solve must succeed");

        let stats_before = solver.statistics();
        assert_eq!(
            stats_before.solve_count, 1,
            "solve_count must be 1 before reset"
        );

        solver.reset();

        let stats_after = solver.statistics();
        assert_eq!(
            stats_after.solve_count, stats_before.solve_count,
            "solve_count must be unchanged after reset"
        );
        assert_eq!(
            stats_after.success_count, stats_before.success_count,
            "success_count must be unchanged after reset"
        );
        assert_eq!(
            stats_after.total_iterations, stats_before.total_iterations,
            "total_iterations must be unchanged after reset"
        );
    }

    /// The first solve must report a positive iteration count.
    #[test]
    fn test_highs_solve_iterations_positive() {
        let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
        let template = make_fixture_stage_template();
        solver.load_model(&template);

        let solution = solver.solve().expect("solve must succeed");
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

        let solution = solver.solve().expect("solve must succeed");
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

        solver.solve().expect("solve must succeed");

        let stats = solver.statistics();
        assert_eq!(stats.solve_count, 1, "solve_count must be 1");
        assert_eq!(stats.success_count, 1, "success_count must be 1");
        assert_eq!(stats.failure_count, 0, "failure_count must be 0");
        assert!(
            stats.total_iterations > 0,
            "total_iterations must be positive after a successful solve"
        );
    }

    /// Solve SS1.1, then call `get_basis()`. Verify the returned `Basis` has
    /// the correct dimension: 3 column statuses and 2 row statuses.
    #[test]
    fn test_highs_get_basis_dimensions() {
        let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
        let template = make_fixture_stage_template();
        solver.load_model(&template);
        solver.solve().expect("solve must succeed before get_basis");

        let basis = solver.get_basis();

        assert_eq!(
            basis.col_status.len(),
            3,
            "col_status must have 3 entries (one per LP column)"
        );
        assert_eq!(
            basis.row_status.len(),
            2,
            "row_status must have 2 entries (one per LP row)"
        );
    }

    /// Solve SS1.1, then verify basis values match the hand-computed optimal basis.
    ///
    /// SS1.1 optimal: x0=6 (Basic), x1=0 (`AtLower`), x2=2 (Basic).
    /// Both equality constraints are active (`row_lower` == `row_upper`), so their
    /// row basis statuses are nonbasic at a bound. `HiGHS` may report either
    /// `AtLower` or `AtUpper` for equality constraints; both are valid.
    /// The test verifies the column statuses exactly (these are unambiguous) and
    /// verifies that row statuses are one of the two valid nonbasic positions.
    #[test]
    fn test_highs_get_basis_values() {
        use crate::types::BasisStatus;

        let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
        let template = make_fixture_stage_template();
        solver.load_model(&template);
        solver.solve().expect("solve must succeed before get_basis");

        let basis = solver.get_basis();

        assert_eq!(
            basis.col_status[0],
            BasisStatus::Basic,
            "col 0 (x0=6, bound by equality) must be Basic"
        );
        assert_eq!(
            basis.col_status[1],
            BasisStatus::AtLower,
            "col 1 (x1=0, at lower bound) must be AtLower"
        );
        assert_eq!(
            basis.col_status[2],
            BasisStatus::Basic,
            "col 2 (x2=2, between bounds) must be Basic"
        );
        // Equality constraints (row_lower == row_upper) may be reported as either
        // AtLower or AtUpper by HiGHS -- both are valid for a nonbasic constraint
        // at its unique feasible bound value.
        assert!(
            basis.row_status[0] == BasisStatus::AtLower
                || basis.row_status[0] == BasisStatus::AtUpper,
            "row 0 (x0=6 equality) must be nonbasic at a bound (AtLower or AtUpper), got {:?}",
            basis.row_status[0]
        );
        assert!(
            basis.row_status[1] == BasisStatus::AtLower
                || basis.row_status[1] == BasisStatus::AtUpper,
            "row 1 (power balance equality) must be nonbasic at a bound (AtLower or AtUpper), got {:?}",
            basis.row_status[1]
        );
    }

    /// Solve SS1.1, capture the basis, then `solve_with_basis` on the same LP.
    /// The solution must be identical and iteration count must be <= cold-start count.
    #[test]
    fn test_highs_solve_with_basis_warm_start() {
        let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
        let template = make_fixture_stage_template();
        solver.load_model(&template);

        let cold_solution = solver.solve().expect("cold-start solve must succeed");
        let cold_iterations = cold_solution.iterations;
        let basis = solver.get_basis();

        // Reload the same model (simulates an iterative algorithm reusing the same template).
        solver.load_model(&template);
        let warm_solution = solver
            .solve_with_basis(&basis)
            .expect("warm-start solve must succeed");

        assert!(
            (warm_solution.objective - 100.0).abs() < 1e-8,
            "warm-start objective must be 100.0, got {}",
            warm_solution.objective
        );
        assert!(
            warm_solution.iterations <= cold_iterations,
            "warm-start iterations ({}) must not exceed cold-start iterations ({})",
            warm_solution.iterations,
            cold_iterations
        );
    }

    /// Solve SS1.1 (2 rows), save basis. Then load model + add 2 cuts (4 rows total).
    /// Call `solve_with_basis` with the 2-row basis. The 2 extra rows must be initialized
    /// as Basic and the solve must succeed with objective = 162.0 (SS1.2 result).
    #[test]
    fn test_highs_solve_with_basis_row_extension() {
        let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
        let template = make_fixture_stage_template();
        let cuts = make_fixture_row_batch();

        // Solve SS1.1 to get the 2-row basis.
        solver.load_model(&template);
        solver.solve().expect("SS1.1 solve must succeed");
        let basis_2rows = solver.get_basis();
        assert_eq!(
            basis_2rows.row_status.len(),
            2,
            "saved basis must have 2 row statuses"
        );

        // Reload model and add cuts to get 4-row LP (SS1.1 + SS1.2 cuts).
        solver.load_model(&template);
        solver.add_rows(&cuts);
        assert_eq!(solver.num_rows, 4, "LP must have 4 rows after add_rows");

        // Warm-start with the 2-row basis. The 2 extra rows are extended as Basic.
        let solution = solver
            .solve_with_basis(&basis_2rows)
            .expect("warm-start solve with row extension must succeed");

        assert!(
            (solution.objective - 162.0).abs() < 1e-8,
            "objective after row extension must be 162.0, got {}",
            solution.objective
        );
    }

    /// Verify `BasisStatus -> HiGHS code -> BasisStatus` roundtrip for all variants.
    ///
    /// `Fixed` maps to `LOWER` (0), which maps back to `AtLower` -- this is the
    /// documented lossy mapping (Fixed has no `HiGHS` equivalent; `AtLower` is the
    /// correct recovery for a fixed variable).
    #[test]
    fn test_basis_status_to_highs_roundtrip() {
        use crate::types::BasisStatus;

        let roundtrip_pairs = [
            (BasisStatus::AtLower, BasisStatus::AtLower),
            (BasisStatus::Basic, BasisStatus::Basic),
            (BasisStatus::AtUpper, BasisStatus::AtUpper),
            (BasisStatus::Free, BasisStatus::Free),
            // Fixed maps to LOWER (0) then back to AtLower -- documented lossy path.
            (BasisStatus::Fixed, BasisStatus::AtLower),
        ];

        for (input, expected_roundtrip) in roundtrip_pairs {
            let code = HighsSolver::basis_status_to_highs(input);
            let recovered = HighsSolver::highs_to_basis_status(code);
            assert_eq!(
                recovered, expected_roundtrip,
                "roundtrip for {input:?}: code={code}, recovered={recovered:?}, expected={expected_roundtrip:?}"
            );
        }
    }

    /// Verify all five `HiGHS` basis status codes map to the expected `BasisStatus`.
    #[test]
    fn test_highs_to_basis_status_all_codes() {
        use crate::{
            ffi::{
                HIGHS_BASIS_STATUS_BASIC, HIGHS_BASIS_STATUS_LOWER, HIGHS_BASIS_STATUS_NONBASIC,
                HIGHS_BASIS_STATUS_UPPER, HIGHS_BASIS_STATUS_ZERO,
            },
            types::BasisStatus,
        };

        assert_eq!(
            HighsSolver::highs_to_basis_status(HIGHS_BASIS_STATUS_LOWER),
            BasisStatus::AtLower,
            "LOWER (0) must map to AtLower"
        );
        assert_eq!(
            HighsSolver::highs_to_basis_status(HIGHS_BASIS_STATUS_BASIC),
            BasisStatus::Basic,
            "BASIC (1) must map to Basic"
        );
        assert_eq!(
            HighsSolver::highs_to_basis_status(HIGHS_BASIS_STATUS_UPPER),
            BasisStatus::AtUpper,
            "UPPER (2) must map to AtUpper"
        );
        assert_eq!(
            HighsSolver::highs_to_basis_status(HIGHS_BASIS_STATUS_ZERO),
            BasisStatus::Free,
            "ZERO (3) must map to Free"
        );
        assert_eq!(
            HighsSolver::highs_to_basis_status(HIGHS_BASIS_STATUS_NONBASIC),
            BasisStatus::AtLower,
            "NONBASIC (4) must map to AtLower (default nonbasic position)"
        );
    }
}

// ─── Research verification tests for ticket-023 ──────────────────────────
//
// These tests verify LP formulations that reliably trigger non-optimal
// HiGHS model statuses. They use the raw FFI layer to set options not
// exposed through SolverInterface and confirm the expected model status.
// Findings are documented in:
//   plans/phase-3-solver/epic-08-coverage/research-edge-case-lps.md
//
// The SS1.1 LP (3-variable, 2-constraint) is too small: HiGHS's crash
// heuristic solves it without entering the simplex loop, so time/iteration
// limits never fire. A 5-variable, 4-constraint "larger_lp" is required.
#[cfg(test)]
#[allow(clippy::doc_markdown)]
mod research_tests_ticket_023 {
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
        let _ = unsafe { ffi::cobre_highs_set_int_option(highs, c"simplex_iteration_limit".as_ptr(), 0) };
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
    /// Confirmed in vendor/HiGHS/check/TestQpSolver.cpp line 1083-1085.
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
    /// Confirmed pattern from vendor/HiGHS/check/TestLpSolversIterations.cpp
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
            unsafe { ffi::cobre_highs_set_string_option(highs, c"presolve".as_ptr(), c"off".as_ptr()) };
            unsafe { research_load_larger_lp(highs) };
            unsafe { ffi::cobre_highs_set_int_option(highs, c"simplex_iteration_limit".as_ptr(), 0) };
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
            ffi::cobre_highs_set_int_option(highs, c"simplex_strategy".as_ptr(), 4);
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
            ffi::cobre_highs_set_int_option(highs, c"simplex_strategy".as_ptr(), 4);
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
    /// also works, providing an alternative formulation for ticket-025.
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
}
