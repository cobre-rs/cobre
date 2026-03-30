/* Thin C wrapper around the HiGHS C API for use by cobre-solver FFI bindings.
 *
 * Each function is a direct call-through to the corresponding Highs_* function
 * with no additional logic.  The wrapper uses int32_t consistently; HighsInt
 * is typedef'd to `int` in the default (non-HIGHSINT64) build, so the two
 * types are ABI-compatible on all supported platforms.
 */

#include "highs_wrapper.h"
#include <interfaces/highs_c_api.h>
#include <stdint.h>

/* Compile-time guard: cobre-solver assumes HighsInt is 32-bit (i32 on the
 * Rust side). If HiGHS was built with -DHIGHSINT64=ON, all FFI calls would
 * silently corrupt memory. Fail the build early instead. */
_Static_assert(sizeof(HighsInt) == sizeof(int32_t),
    "cobre-solver assumes HighsInt is 32-bit; rebuild HiGHS without HIGHSINT64");

/* =========================================================================
 * Lifecycle
 * ========================================================================= */

void* cobre_highs_create(void) {
    return Highs_create();
}

void cobre_highs_destroy(void* highs) {
    Highs_destroy(highs);
}

/* =========================================================================
 * Model Loading
 * ========================================================================= */

int32_t cobre_highs_pass_lp(
    void*           highs,
    int32_t         num_col,
    int32_t         num_row,
    int32_t         num_nz,
    int32_t         a_format,
    int32_t         sense,
    double          offset,
    const double*   col_cost,
    const double*   col_lower,
    const double*   col_upper,
    const double*   row_lower,
    const double*   row_upper,
    const int32_t*  a_start,
    const int32_t*  a_index,
    const double*   a_value
) {
    return (int32_t)Highs_passLp(
        highs,
        (HighsInt)num_col,
        (HighsInt)num_row,
        (HighsInt)num_nz,
        (HighsInt)a_format,
        (HighsInt)sense,
        offset,
        col_cost,
        col_lower,
        col_upper,
        row_lower,
        row_upper,
        (const HighsInt*)a_start,
        (const HighsInt*)a_index,
        a_value
    );
}

/* =========================================================================
 * Row / Column Modification
 * ========================================================================= */

int32_t cobre_highs_add_rows(
    void*           highs,
    int32_t         num_new_row,
    const double*   lower,
    const double*   upper,
    int32_t         num_new_nz,
    const int32_t*  starts,
    const int32_t*  index,
    const double*   value
) {
    return (int32_t)Highs_addRows(
        highs,
        (HighsInt)num_new_row,
        lower,
        upper,
        (HighsInt)num_new_nz,
        (const HighsInt*)starts,
        (const HighsInt*)index,
        value
    );
}

int32_t cobre_highs_change_rows_bounds_by_set(
    void*           highs,
    int32_t         num_set_entries,
    const int32_t*  set,
    const double*   lower,
    const double*   upper
) {
    return (int32_t)Highs_changeRowsBoundsBySet(
        highs,
        (HighsInt)num_set_entries,
        (const HighsInt*)set,
        lower,
        upper
    );
}

int32_t cobre_highs_change_cols_bounds_by_set(
    void*           highs,
    int32_t         num_set_entries,
    const int32_t*  set,
    const double*   lower,
    const double*   upper
) {
    return (int32_t)Highs_changeColsBoundsBySet(
        highs,
        (HighsInt)num_set_entries,
        (const HighsInt*)set,
        lower,
        upper
    );
}

/* =========================================================================
 * Solving
 * ========================================================================= */

int32_t cobre_highs_run(void* highs) {
    return (int32_t)Highs_run(highs);
}

/* =========================================================================
 * Solution Extraction
 * ========================================================================= */

int32_t cobre_highs_get_solution(
    const void* highs,
    double*     col_value,
    double*     col_dual,
    double*     row_value,
    double*     row_dual
) {
    return (int32_t)Highs_getSolution(highs, col_value, col_dual, row_value, row_dual);
}

double cobre_highs_get_objective_value(const void* highs) {
    return Highs_getObjectiveValue(highs);
}

int32_t cobre_highs_get_model_status(const void* highs) {
    return (int32_t)Highs_getModelStatus(highs);
}

int32_t cobre_highs_get_simplex_iteration_count(const void* highs) {
    return (int32_t)Highs_getSimplexIterationCount(highs);
}

/* =========================================================================
 * Basis Management
 * ========================================================================= */

int32_t cobre_highs_set_basis(
    void*           highs,
    const int32_t*  col_status,
    const int32_t*  row_status
) {
    return (int32_t)Highs_setBasis(
        highs,
        (const HighsInt*)col_status,
        (const HighsInt*)row_status
    );
}

int32_t cobre_highs_get_basis(
    const void* highs,
    int32_t*    col_status,
    int32_t*    row_status
) {
    return (int32_t)Highs_getBasis(
        highs,
        (HighsInt*)col_status,
        (HighsInt*)row_status
    );
}

/* =========================================================================
 * Reset
 * ========================================================================= */

int32_t cobre_highs_clear_solver(void* highs) {
    return (int32_t)Highs_clearSolver(highs);
}

/* =========================================================================
 * Configuration
 * ========================================================================= */

int32_t cobre_highs_set_string_option(
    void*       highs,
    const char* option,
    const char* value
) {
    return (int32_t)Highs_setStringOptionValue(highs, option, value);
}

int32_t cobre_highs_set_bool_option(
    void*       highs,
    const char* option,
    int32_t     value
) {
    return (int32_t)Highs_setBoolOptionValue(highs, option, (HighsInt)value);
}

int32_t cobre_highs_set_int_option(
    void*       highs,
    const char* option,
    int32_t     value
) {
    return (int32_t)Highs_setIntOptionValue(highs, option, (HighsInt)value);
}

int32_t cobre_highs_set_double_option(
    void*       highs,
    const char* option,
    double      value
) {
    return (int32_t)Highs_setDoubleOptionValue(highs, option, value);
}

/* =========================================================================
 * Diagnostics
 * ========================================================================= */

int32_t cobre_highs_get_dual_ray(
    const void* highs,
    int32_t*    has_dual_ray,
    double*     dual_ray_value
) {
    return (int32_t)Highs_getDualRay(
        highs,
        (HighsInt*)has_dual_ray,
        dual_ray_value
    );
}

int32_t cobre_highs_get_primal_ray(
    const void* highs,
    int32_t*    has_primal_ray,
    double*     primal_ray_value
) {
    return (int32_t)Highs_getPrimalRay(
        highs,
        (HighsInt*)has_primal_ray,
        primal_ray_value
    );
}

/* =========================================================================
 * Info
 * ========================================================================= */

int32_t cobre_highs_get_num_col(const void* highs) {
    return (int32_t)Highs_getNumCol(highs);
}

int32_t cobre_highs_get_num_row(const void* highs) {
    return (int32_t)Highs_getNumRow(highs);
}
