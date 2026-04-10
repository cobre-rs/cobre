//! Raw FFI bindings to the `HiGHS` C wrapper layer.
//!
//! These are low-level unsafe functions that map 1:1 to the `cobre_highs_*`
//! functions declared in `csrc/highs_wrapper.h`.  Use the safe wrappers in
//! the parent module rather than calling these directly.

#![allow(dead_code)]
#![allow(non_camel_case_types)]

use std::os::raw::{c_char, c_double, c_int, c_void};

/// C `int32_t` mapped to Rust `i32`.
pub type int32_t = i32;

// ============================================================
// HiGHS status constants (kHighsStatus*)
// ============================================================

/// `kHighsStatusError` — the call failed with an error.
pub const HIGHS_STATUS_ERROR: i32 = -1;
/// `kHighsStatusOk` — the call succeeded.
pub const HIGHS_STATUS_OK: i32 = 0;
/// `kHighsStatusWarning` — the call succeeded with a warning.
pub const HIGHS_STATUS_WARNING: i32 = 1;

// ============================================================
// HiGHS model status constants (kHighsModelStatus*)
// ============================================================

/// `kHighsModelStatusSolveError` — the solver encountered an internal error.
pub const HIGHS_MODEL_STATUS_SOLVE_ERROR: i32 = 4;
/// `kHighsModelStatusOptimal` — the model was solved to optimality.
pub const HIGHS_MODEL_STATUS_OPTIMAL: i32 = 7;
/// `kHighsModelStatusInfeasible` — the model is infeasible.
pub const HIGHS_MODEL_STATUS_INFEASIBLE: i32 = 8;
/// `kHighsModelStatusUnboundedOrInfeasible` — the model is unbounded or infeasible.
pub const HIGHS_MODEL_STATUS_UNBOUNDED_OR_INFEASIBLE: i32 = 9;
/// `kHighsModelStatusUnbounded` — the model is unbounded.
pub const HIGHS_MODEL_STATUS_UNBOUNDED: i32 = 10;
/// `kHighsModelStatusTimeLimit` — the time limit was reached.
pub const HIGHS_MODEL_STATUS_TIME_LIMIT: i32 = 13;
/// `kHighsModelStatusIterationLimit` — the iteration limit was reached.
pub const HIGHS_MODEL_STATUS_ITERATION_LIMIT: i32 = 14;
/// `kHighsModelStatusUnknown` — the model status is unknown.
pub const HIGHS_MODEL_STATUS_UNKNOWN: i32 = 15;

// ============================================================
// HiGHS basis status constants (kHighsBasisStatus*)
// ============================================================

/// `kHighsBasisStatusLower` — variable is at its lower bound.
pub const HIGHS_BASIS_STATUS_LOWER: i32 = 0;
/// `kHighsBasisStatusBasic` — variable is basic.
pub const HIGHS_BASIS_STATUS_BASIC: i32 = 1;
/// `kHighsBasisStatusUpper` — variable is at its upper bound.
pub const HIGHS_BASIS_STATUS_UPPER: i32 = 2;
/// `kHighsBasisStatusZero` — variable is free and zero.
pub const HIGHS_BASIS_STATUS_ZERO: i32 = 3;
/// `kHighsBasisStatusNonbasic` — variable is nonbasic.
pub const HIGHS_BASIS_STATUS_NONBASIC: i32 = 4;

// ============================================================
// HiGHS matrix format constants (kHighsMatrixFormat*)
// ============================================================

/// `kHighsMatrixFormatColwise` — constraint matrix stored column-wise (CSC).
pub const HIGHS_MATRIX_FORMAT_COLWISE: i32 = 1;
/// `kHighsMatrixFormatRowwise` — constraint matrix stored row-wise (CSR).
pub const HIGHS_MATRIX_FORMAT_ROWWISE: i32 = 2;

// ============================================================
// HiGHS objective sense constant (kHighsObjSense*)
// ============================================================

/// `kHighsObjSenseMinimize` — minimize the objective.
pub const HIGHS_OBJ_SENSE_MINIMIZE: i32 = 1;

unsafe extern "C" {
    // ============================================================
    // Lifecycle
    // ============================================================

    /// Create a `HiGHS` instance. Wraps `Highs_create()`.
    pub fn cobre_highs_create() -> *mut c_void;

    /// Destroy a `HiGHS` instance and free all associated memory.
    /// Wraps `Highs_destroy()`.
    pub fn cobre_highs_destroy(highs: *mut c_void);

    // ============================================================
    // Model Loading
    // ============================================================

    /// Pass a complete LP to `HiGHS` in a single call.
    /// Wraps `Highs_passLp()`.
    pub fn cobre_highs_pass_lp(
        highs: *mut c_void,
        num_col: int32_t,
        num_row: int32_t,
        num_nz: int32_t,
        a_format: int32_t,
        sense: int32_t,
        offset: c_double,
        col_cost: *const c_double,
        col_lower: *const c_double,
        col_upper: *const c_double,
        row_lower: *const c_double,
        row_upper: *const c_double,
        a_start: *const int32_t,
        a_index: *const int32_t,
        a_value: *const c_double,
    ) -> c_int;

    // ============================================================
    // Row / Column Modification
    // ============================================================

    /// Add rows to the incumbent model. Wraps `Highs_addRows()`.
    pub fn cobre_highs_add_rows(
        highs: *mut c_void,
        num_new_row: int32_t,
        lower: *const c_double,
        upper: *const c_double,
        num_new_nz: int32_t,
        starts: *const int32_t,
        index: *const int32_t,
        value: *const c_double,
    ) -> c_int;

    /// Change bounds of rows identified by an index set.
    /// Wraps `Highs_changeRowsBoundsBySet()`.
    pub fn cobre_highs_change_rows_bounds_by_set(
        highs: *mut c_void,
        num_set_entries: int32_t,
        set: *const int32_t,
        lower: *const c_double,
        upper: *const c_double,
    ) -> c_int;

    /// Change bounds of columns identified by an index set.
    /// Wraps `Highs_changeColsBoundsBySet()`.
    pub fn cobre_highs_change_cols_bounds_by_set(
        highs: *mut c_void,
        num_set_entries: int32_t,
        set: *const int32_t,
        lower: *const c_double,
        upper: *const c_double,
    ) -> c_int;

    // ============================================================
    // Solving
    // ============================================================

    /// Run the solver on the incumbent model. Wraps `Highs_run()`.
    pub fn cobre_highs_run(highs: *mut c_void) -> c_int;

    // ============================================================
    // Solution Extraction
    // ============================================================

    /// Get the primal and dual solution arrays. Wraps `Highs_getSolution()`.
    pub fn cobre_highs_get_solution(
        highs: *const c_void,
        col_value: *mut c_double,
        col_dual: *mut c_double,
        row_value: *mut c_double,
        row_dual: *mut c_double,
    ) -> c_int;

    /// Get the primal objective value. Wraps `Highs_getObjectiveValue()`.
    pub fn cobre_highs_get_objective_value(highs: *const c_void) -> c_double;

    /// Get the model status after solving. Wraps `Highs_getModelStatus()`.
    pub fn cobre_highs_get_model_status(highs: *const c_void) -> c_int;

    /// Get the simplex iteration count from the most recent solve.
    /// Wraps `Highs_getSimplexIterationCount()`.
    pub fn cobre_highs_get_simplex_iteration_count(highs: *const c_void) -> c_int;

    // ============================================================
    // Basis Management
    // ============================================================

    /// Set the basis using column and row status arrays. Wraps `Highs_setBasis()`.
    pub fn cobre_highs_set_basis(
        highs: *mut c_void,
        col_status: *const int32_t,
        row_status: *const int32_t,
    ) -> c_int;

    /// Get the current basis into caller-allocated arrays. Wraps `Highs_getBasis()`.
    pub fn cobre_highs_get_basis(
        highs: *const c_void,
        col_status: *mut int32_t,
        row_status: *mut int32_t,
    ) -> c_int;

    // ============================================================
    // Reset
    // ============================================================

    /// Clear the solver state while preserving the model.
    /// Wraps `Highs_clearSolver()`.
    pub fn cobre_highs_clear_solver(highs: *mut c_void) -> c_int;

    // ============================================================
    // Configuration
    // ============================================================

    /// Set a string-valued `HiGHS` option. Wraps `Highs_setStringOptionValue()`.
    pub fn cobre_highs_set_string_option(
        highs: *mut c_void,
        option: *const c_char,
        value: *const c_char,
    ) -> c_int;

    /// Set a boolean-valued `HiGHS` option. Wraps `Highs_setBoolOptionValue()`.
    pub fn cobre_highs_set_bool_option(
        highs: *mut c_void,
        option: *const c_char,
        value: int32_t,
    ) -> c_int;

    /// Set an integer-valued `HiGHS` option. Wraps `Highs_setIntOptionValue()`.
    pub fn cobre_highs_set_int_option(
        highs: *mut c_void,
        option: *const c_char,
        value: int32_t,
    ) -> c_int;

    /// Set a double-valued `HiGHS` option. Wraps `Highs_setDoubleOptionValue()`.
    pub fn cobre_highs_set_double_option(
        highs: *mut c_void,
        option: *const c_char,
        value: c_double,
    ) -> c_int;

    // ============================================================
    // Diagnostics
    // ============================================================

    /// Check whether a dual ray exists and retrieve it.
    /// Wraps `Highs_getDualRay()`.
    pub fn cobre_highs_get_dual_ray(
        highs: *const c_void,
        has_dual_ray: *mut int32_t,
        dual_ray_value: *mut c_double,
    ) -> c_int;

    /// Check whether a primal ray exists and retrieve it.
    /// Wraps `Highs_getPrimalRay()`.
    pub fn cobre_highs_get_primal_ray(
        highs: *const c_void,
        has_primal_ray: *mut int32_t,
        primal_ray_value: *mut c_double,
    ) -> c_int;

    // ============================================================
    // Info
    // ============================================================

    /// Return the number of columns in the incumbent model.
    /// Wraps `Highs_getNumCol()`.
    pub fn cobre_highs_get_num_col(highs: *const c_void) -> c_int;

    /// Return the number of rows in the incumbent model.
    /// Wraps `Highs_getNumRow()`.
    pub fn cobre_highs_get_num_row(highs: *const c_void) -> c_int;

    // ============================================================
    // Version query (no solver instance required)
    // ============================================================

    /// Return the `HiGHS` major version number. Wraps `Highs_versionMajor()`.
    pub fn cobre_highs_version_major() -> c_int;

    /// Return the `HiGHS` minor version number. Wraps `Highs_versionMinor()`.
    pub fn cobre_highs_version_minor() -> c_int;

    /// Return the `HiGHS` patch version number. Wraps `Highs_versionPatch()`.
    pub fn cobre_highs_version_patch() -> c_int;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Smoke test: create a `HiGHS` instance, load a trivial 1-variable LP
    /// (minimize x, x ∈ [0, 10], no constraints), solve it, verify optimality
    /// and objective value, then destroy the instance.
    ///
    /// This validates the full FFI pipeline: C wrapper compiles and links against
    /// `HiGHS`, Rust declarations match C signatures, and a basic solve works
    /// end-to-end.
    #[test]
    fn test_ffi_smoke_create_solve_destroy() {
        let highs = unsafe { cobre_highs_create() };
        assert!(!highs.is_null(), "cobre_highs_create() returned null");

        // output_flag is bool-typed; use bool setter, not int setter.
        let status = unsafe { cobre_highs_set_bool_option(highs, c"output_flag".as_ptr(), 0) };
        assert_eq!(
            status, HIGHS_STATUS_OK,
            "cobre_highs_set_bool_option(output_flag) returned status {status}"
        );

        // Minimize x where x ∈ [0, 10] with no constraints. Expected: x* = 0, obj = 0.
        let col_cost: [f64; 1] = [1.0];
        let col_lower: [f64; 1] = [0.0];
        let col_upper: [f64; 1] = [10.0];
        let a_start: [i32; 2] = [0, 0];

        let status = unsafe {
            cobre_highs_pass_lp(
                highs,
                1, // num_col
                0, // num_row
                0, // num_nz
                HIGHS_MATRIX_FORMAT_COLWISE,
                HIGHS_OBJ_SENSE_MINIMIZE,
                0.0, // offset
                col_cost.as_ptr(),
                col_lower.as_ptr(),
                col_upper.as_ptr(),
                std::ptr::null(), // row_lower (empty)
                std::ptr::null(), // row_upper (empty)
                a_start.as_ptr(),
                std::ptr::null(), // a_index (empty)
                std::ptr::null(), // a_value (empty)
            )
        };
        assert_eq!(
            status, HIGHS_STATUS_OK,
            "cobre_highs_pass_lp() returned status {status}"
        );

        let status = unsafe { cobre_highs_run(highs) };
        assert_eq!(
            status, HIGHS_STATUS_OK,
            "cobre_highs_run() returned status {status}"
        );

        let model_status = unsafe { cobre_highs_get_model_status(highs) };
        assert_eq!(
            model_status, HIGHS_MODEL_STATUS_OPTIMAL,
            "expected Optimal model status, got {model_status}"
        );

        let obj = unsafe { cobre_highs_get_objective_value(highs) };
        assert!(
            (obj - 0.0_f64).abs() < 1e-10,
            "objective value {obj} is not within 1e-10 of expected 0.0"
        );

        unsafe { cobre_highs_destroy(highs) };
    }
}
