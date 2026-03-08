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
    types::{LpSolution, RowBatch, SolutionView, SolverError, SolverStatistics, StageTemplate},
};

// ─── Default HiGHS configuration ─────────────────────────────────────────────
//
// The seven performance-tuned options applied at construction and restored after
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

/// The seven performance-tuned default options (`HiGHS` Implementation SS4.1).
fn default_options() -> [DefaultOption; 7] {
    [
        DefaultOption {
            name: c"solver",
            value: OptionValue::Str(c"simplex"),
        },
        DefaultOption {
            name: c"simplex_strategy",
            value: OptionValue::Int(4),
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
        assert_ne!(
            status,
            ffi::HIGHS_STATUS_ERROR,
            "cobre_highs_get_solution failed after optimal solve"
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
    /// Errors are silently ignored — already in recovery path.
    fn restore_default_settings(&mut self) {
        for opt in &default_options() {
            // SAFETY: `self.handle` is a valid, non-null HiGHS pointer.
            unsafe { opt.apply(self.handle) };
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

    /// Interprets a non-optimal status as a `SolverError`.
    ///
    /// Returns `None` for `SOLVE_ERROR` or `UNKNOWN` (retry continues),
    /// or `Some(Err(...))` for terminal statuses.
    fn interpret_terminal_status(
        &mut self,
        status: i32,
        solve_time_seconds: f64,
    ) -> Option<Result<LpSolution, SolverError>> {
        match status {
            ffi::HIGHS_MODEL_STATUS_OPTIMAL => {
                // Caller should have handled optimal before reaching here.
                None
            }
            ffi::HIGHS_MODEL_STATUS_INFEASIBLE => Some(Err(SolverError::Infeasible)),
            ffi::HIGHS_MODEL_STATUS_UNBOUNDED_OR_INFEASIBLE => {
                // Probe for a dual ray to classify as Infeasible, then a primal
                // ray to classify as Unbounded. The ray values are not stored in
                // the error -- only the classification matters.
                let mut has_dual_ray: i32 = 0;
                // A scratch buffer is needed for the HiGHS API even though the
                // values are discarded after classification.
                let mut dual_buf = vec![0.0_f64; self.num_rows];
                // SAFETY: valid non-null HiGHS pointer; buffers are valid.
                let dual_status = unsafe {
                    ffi::cobre_highs_get_dual_ray(
                        self.handle,
                        &raw mut has_dual_ray,
                        dual_buf.as_mut_ptr(),
                    )
                };
                if dual_status != ffi::HIGHS_STATUS_ERROR && has_dual_ray != 0 {
                    return Some(Err(SolverError::Infeasible));
                }
                let mut has_primal_ray: i32 = 0;
                let mut primal_buf = vec![0.0_f64; self.num_cols];
                // SAFETY: valid non-null HiGHS pointer; buffers are valid.
                let primal_status = unsafe {
                    ffi::cobre_highs_get_primal_ray(
                        self.handle,
                        &raw mut has_primal_ray,
                        primal_buf.as_mut_ptr(),
                    )
                };
                if primal_status != ffi::HIGHS_STATUS_ERROR && has_primal_ray != 0 {
                    return Some(Err(SolverError::Unbounded));
                }
                Some(Err(SolverError::Infeasible))
            }
            ffi::HIGHS_MODEL_STATUS_UNBOUNDED => Some(Err(SolverError::Unbounded)),
            ffi::HIGHS_MODEL_STATUS_TIME_LIMIT => Some(Err(SolverError::TimeLimitExceeded {
                elapsed_seconds: solve_time_seconds,
            })),
            ffi::HIGHS_MODEL_STATUS_ITERATION_LIMIT => {
                // SAFETY: handle is valid non-null pointer; iteration count is non-negative.
                #[allow(clippy::cast_sign_loss)]
                let iterations =
                    unsafe { ffi::cobre_highs_get_simplex_iteration_count(self.handle) } as u64;
                Some(Err(SolverError::IterationLimit { iterations }))
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
}

impl Drop for HighsSolver {
    fn drop(&mut self) {
        // SAFETY: valid HiGHS pointer from construction, called once per instance.
        unsafe { ffi::cobre_highs_destroy(self.handle) };
    }
}

impl SolverInterface for HighsSolver {
    fn name(&self) -> &'static str {
        "HiGHS"
    }

    fn load_model(&mut self, template: &StageTemplate) {
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
    }

    fn add_rows(&mut self, cuts: &RowBatch) {
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
        // - `cuts.row_starts` and `cuts.col_indices` are `Vec<i32>` owned by the RowBatch,
        //   alive for the duration of this borrow.
        // - Slice lengths: `num_rows + 1` for starts, total nnz for index and value,
        //   `num_rows` for lower/upper bounds.
        let status = unsafe {
            ffi::cobre_highs_add_rows(
                self.handle,
                num_new_row,
                cuts.row_lower.as_ptr(),
                cuts.row_upper.as_ptr(),
                num_new_nz,
                cuts.row_starts.as_ptr(),
                cuts.col_indices.as_ptr(),
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
    fn solve(&mut self) -> Result<SolutionView<'_>, SolverError> {
        assert!(
            self.has_model,
            "solve called without a loaded model — call load_model first"
        );
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
            self.stats.total_iterations += iterations;
            self.stats.total_solve_time_seconds += solve_time;
            return Ok(self.extract_solution_view(solve_time));
        }

        // Check for a definitive terminal status (not a retry-able error).
        // `interpret_terminal_status` only returns `Some(Err(...))` for non-OPTIMAL
        // statuses; it never returns `Some(Ok(...))`.
        if let Some(Err(terminal_err)) = self.interpret_terminal_status(model_status, solve_time) {
            self.stats.failure_count += 1;
            return Err(terminal_err);
        }

        // 5-level retry escalation (HiGHS Implementation SS3). Apply progressively
        // more permissive strategies on SOLVE_ERROR/UNKNOWN; break on OPTIMAL or
        // definitive terminal status.
        let mut retry_attempts: u64 = 0;
        // None = retry loop exhausted without success; Some(Err) = terminal failure.
        // We accumulate the error, then after restoring settings we either return
        // it or return Ok(view).
        let mut terminal_err: Option<SolverError> = None;
        let mut found_optimal = false;
        let mut optimal_time = 0.0_f64;
        let mut optimal_iterations: u64 = 0;

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
                // Capture stats before establishing the borrow.
                // SAFETY: handle is valid non-null HiGHS pointer.
                #[allow(clippy::cast_sign_loss)]
                let iters =
                    unsafe { ffi::cobre_highs_get_simplex_iteration_count(self.handle) } as u64;
                found_optimal = true;
                optimal_time = retry_time;
                optimal_iterations = iters;
                break;
            }

            if let Some(Err(e)) = self.interpret_terminal_status(retry_status, retry_time) {
                terminal_err = Some(e);
                break;
            }
            // Still SOLVE_ERROR or UNKNOWN -- continue to next level.
        }

        // Restore default settings unconditionally (regardless of retry outcome).
        self.restore_default_settings();

        // Update statistics with accumulated retry attempts.
        self.stats.retry_count += retry_attempts;

        if found_optimal {
            self.stats.success_count += 1;
            self.stats.total_iterations += optimal_iterations;
            self.stats.total_solve_time_seconds += optimal_time;
            return Ok(self.extract_solution_view(optimal_time));
        }

        self.stats.failure_count += 1;
        Err(terminal_err.unwrap_or_else(|| {
            // All 5 retry levels exhausted without a definitive result.
            SolverError::NumericalDifficulty {
                message: "HiGHS failed to reach optimality after all 5 retry escalation levels"
                    .to_string(),
            }
        }))
    }

    fn reset(&mut self) {
        // SAFETY: `self.handle` is a valid, non-null HiGHS pointer. `cobre_highs_clear_solver`
        // discards the cached basis and factorization. HiGHS preserves the model data
        // internally, but Cobre's `reset` contract requires `load_model` before the
        // next solve — enforced by setting `has_model = false`.
        let status = unsafe { ffi::cobre_highs_clear_solver(self.handle) };
        debug_assert_ne!(
            status,
            ffi::HIGHS_STATUS_ERROR,
            "cobre_highs_clear_solver failed — HiGHS internal state may be inconsistent"
        );
        // Force `load_model` to be called before the next solve.
        self.num_cols = 0;
        self.num_rows = 0;
        self.has_model = false;
        // Intentionally do NOT zero `self.stats` -- statistics accumulate for the
        // lifetime of the instance (per trait contract, SS4.3).
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

    fn solve_with_basis(
        &mut self,
        basis: &crate::types::Basis,
    ) -> Result<crate::types::SolutionView<'_>, SolverError> {
        assert!(
            self.has_model,
            "solve_with_basis called without a loaded model — call load_model first"
        );
        assert!(
            basis.col_status.len() == self.num_cols,
            "basis column count {} does not match LP column count {}",
            basis.col_status.len(),
            self.num_cols
        );

        // Copy raw i32 codes directly into the pre-allocated buffers — no enum
        // translation. Zero-copy warm-start path.
        self.basis_col_i32[..self.num_cols].copy_from_slice(&basis.col_status);

        // Handle dimension mismatch for dynamic cuts:
        // - Fewer rows than LP: extend with BASIC.
        // - More rows than LP: truncate (extra entries ignored).
        let basis_rows = basis.row_status.len();
        let lp_rows = self.num_rows;
        let copy_len = basis_rows.min(lp_rows);
        self.basis_row_i32[..copy_len].copy_from_slice(&basis.row_status[..copy_len]);
        if lp_rows > basis_rows {
            self.basis_row_i32[basis_rows..lp_rows].fill(ffi::HIGHS_BASIS_STATUS_BASIC);
        }

        // Attempt to install the basis in HiGHS.
        // SAFETY:
        // - `self.handle` is a valid, non-null HiGHS pointer.
        // - `basis_col_i32` has been sized to at least `num_cols` in `load_model`.
        // - `basis_row_i32` has been sized to at least `num_rows` in `load_model`/`add_rows`.
        // - We pass exactly `num_cols` col entries and `num_rows` row entries.
        let set_status = unsafe {
            ffi::cobre_highs_set_basis(
                self.handle,
                self.basis_col_i32.as_ptr(),
                self.basis_row_i32.as_ptr(),
            )
        };

        // Basis rejection tracking: fall back to cold-start and track for diagnostics.
        if set_status == ffi::HIGHS_STATUS_ERROR {
            self.stats.basis_rejections += 1;
            debug_assert!(false, "raw basis rejected; falling back to cold-start");
        }

        // Delegate to solve() which handles retry escalation and statistics updates.
        self.solve()
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
        }
    }

    // Benders cut fixture from Solver Interface Testing SS1.2:
    // Cut 1: -5*x0 + x1 >= 20  (col_indices [0,1], values [-5, 1])
    // Cut 2:  3*x0 + x1 >= 80  (col_indices [0,1], values [ 3, 1])
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

        let solution = solver
            .solve()
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

        let solution = solver
            .solve()
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
            .solve()
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

    /// After `load_model` + `solve()`, `get_basis` must return i32 codes
    /// that are all valid `HiGHS` basis status values (0..=4).
    #[test]
    fn test_get_basis_valid_status_codes() {
        let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
        let template = make_fixture_stage_template();
        solver.load_model(&template);
        solver.solve().expect("solve must succeed before get_basis");

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
        solver.solve().expect("solve must succeed before get_basis");

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

    /// Warm-start via `solve_with_basis` on the same LP must reproduce
    /// the optimal objective and complete in at most 1 simplex iteration.
    #[test]
    fn test_solve_with_basis_warm_start() {
        let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
        let template = make_fixture_stage_template();
        solver.load_model(&template);
        solver.solve().expect("cold-start solve must succeed");

        let mut basis = Basis::new(0, 0);
        solver.get_basis(&mut basis);

        // Reload the same model to reset HiGHS internal state.
        solver.load_model(&template);
        let result = solver
            .solve_with_basis(&basis)
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
            stats.basis_rejections, 0,
            "basis_rejections must be 0 when raw basis is accepted, got {}",
            stats.basis_rejections
        );
    }

    /// When the basis has fewer rows than the current LP (2 vs 4 after `add_rows`),
    /// `solve_with_basis` must extend missing rows as Basic and solve correctly.
    /// SS1.2 objective with both cuts active is 162.0.
    #[test]
    fn test_solve_with_basis_dimension_mismatch() {
        let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
        let template = make_fixture_stage_template();
        let cuts = make_fixture_row_batch();

        // First solve on 2-row LP to capture a 2-row basis.
        solver.load_model(&template);
        solver.solve().expect("SS1.1 solve must succeed");
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
            .solve_with_basis(&basis)
            .expect("solve with dimension-mismatched basis must succeed");

        assert!(
            (result.objective - 162.0).abs() < 1e-8,
            "objective with both cuts active must be 162.0, got {}",
            result.objective
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
