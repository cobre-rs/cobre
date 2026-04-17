#ifndef COBRE_HIGHS_WRAPPER_H
#define COBRE_HIGHS_WRAPPER_H

/* Thin C wrapper around the HiGHS C API for use by cobre-solver FFI bindings.
 *
 * All functions use the `cobre_highs_` prefix and fixed-width types (int32_t,
 * double) for FFI safety.  Each function maps 1:1 to the corresponding
 * HiGHS C API call with no additional logic.
 *
 * This header is intentionally self-contained: it does NOT include
 * <Highs_c_api.h> so that Rust-facing code never needs to pull in the HiGHS
 * internal headers.
 */

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* =========================================================================
 * Lifecycle
 * ========================================================================= */

/** Create a HiGHS instance.  Returns an opaque pointer; caller owns it.
 *  Wraps Highs_create(). */
void* cobre_highs_create(void);

/** Destroy a HiGHS instance and free all associated memory.
 *  Wraps Highs_destroy(). */
void cobre_highs_destroy(void* highs);

/* =========================================================================
 * Model Loading
 * ========================================================================= */

/** Pass a complete LP to HiGHS in a single call.
 *  Wraps Highs_passLp().
 *  Returns a kHighsStatus constant (0 = OK, -1 = Error, 1 = Warning). */
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
);

/* =========================================================================
 * Row / Column Modification
 * ========================================================================= */

/** Add rows to the incumbent model.
 *  Wraps Highs_addRows().
 *  Returns a kHighsStatus constant. */
int32_t cobre_highs_add_rows(
    void*           highs,
    int32_t         num_new_row,
    const double*   lower,
    const double*   upper,
    int32_t         num_new_nz,
    const int32_t*  starts,
    const int32_t*  index,
    const double*   value
);

/** Change bounds of rows identified by an index set.
 *  Wraps Highs_changeRowsBoundsBySet().
 *  Returns a kHighsStatus constant. */
int32_t cobre_highs_change_rows_bounds_by_set(
    void*           highs,
    int32_t         num_set_entries,
    const int32_t*  set,
    const double*   lower,
    const double*   upper
);

/** Change bounds of columns identified by an index set.
 *  Wraps Highs_changeColsBoundsBySet().
 *  Returns a kHighsStatus constant. */
int32_t cobre_highs_change_cols_bounds_by_set(
    void*           highs,
    int32_t         num_set_entries,
    const int32_t*  set,
    const double*   lower,
    const double*   upper
);

/* =========================================================================
 * Solving
 * ========================================================================= */

/** Run the solver on the incumbent model.
 *  Wraps Highs_run().
 *  Returns a kHighsStatus constant. */
int32_t cobre_highs_run(void* highs);

/* =========================================================================
 * Solution Extraction
 * ========================================================================= */

/** Get the primal and dual solution arrays.
 *  Wraps Highs_getSolution().
 *  Returns a kHighsStatus constant. */
int32_t cobre_highs_get_solution(
    const void* highs,
    double*     col_value,
    double*     col_dual,
    double*     row_value,
    double*     row_dual
);

/** Get the primal objective value.
 *  Wraps Highs_getObjectiveValue().
 *  Returns the primal objective function value. */
double cobre_highs_get_objective_value(const void* highs);

/** Get the model status after solving.
 *  Wraps Highs_getModelStatus().
 *  Returns a kHighsModelStatus constant. */
int32_t cobre_highs_get_model_status(const void* highs);

/** Get the simplex iteration count from the most recent solve.
 *  Wraps Highs_getSimplexIterationCount().
 *  Returns the iteration count. */
int32_t cobre_highs_get_simplex_iteration_count(const void* highs);

/* =========================================================================
 * Basis Management
 * ========================================================================= */

/** Set the basis using column and row status arrays.
 *  Wraps Highs_setBasis().
 *  Returns a kHighsStatus constant. */
int32_t cobre_highs_set_basis(
    void*           highs,
    const int32_t*  col_status,
    const int32_t*  row_status
);

/** Set the basis with alien = false, skipping HiGHS's throwaway
 *  rank-detection LU factorization. Caller guarantees basis
 *  consistency (total basic count equals num_rows).
 *  Wraps Highs::setBasis(const HighsBasis&) with basis.alien = false.
 *  Returns a kHighsStatus constant. kError indicates basis rejection
 *  (isBasisConsistent failed); caller should fall back to the alien
 *  path. */
int32_t cobre_highs_set_basis_non_alien(
    void*           highs,
    const int32_t*  col_status,
    const int32_t*  row_status
);

/** Get the current basis into caller-allocated column and row status arrays.
 *  Wraps Highs_getBasis().
 *  Returns a kHighsStatus constant. */
int32_t cobre_highs_get_basis(
    const void* highs,
    int32_t*    col_status,
    int32_t*    row_status
);

/* =========================================================================
 * Reset
 * ========================================================================= */

/** Clear the solver state while preserving the model.
 *  Wraps Highs_clearSolver().
 *  Returns a kHighsStatus constant. */
int32_t cobre_highs_clear_solver(void* highs);

/* =========================================================================
 * Configuration
 * ========================================================================= */

/** Set a string-valued HiGHS option.
 *  Wraps Highs_setStringOptionValue().
 *  Returns a kHighsStatus constant. */
int32_t cobre_highs_set_string_option(
    void*       highs,
    const char* option,
    const char* value
);

/** Set a boolean-valued HiGHS option.
 *  Wraps Highs_setBoolOptionValue().
 *  Returns a kHighsStatus constant. */
int32_t cobre_highs_set_bool_option(
    void*       highs,
    const char* option,
    int32_t     value
);

/** Set an integer-valued HiGHS option.
 *  Wraps Highs_setIntOptionValue().
 *  Returns a kHighsStatus constant. */
int32_t cobre_highs_set_int_option(
    void*       highs,
    const char* option,
    int32_t     value
);

/** Set a double-valued HiGHS option.
 *  Wraps Highs_setDoubleOptionValue().
 *  Returns a kHighsStatus constant. */
int32_t cobre_highs_set_double_option(
    void*       highs,
    const char* option,
    double      value
);

/* =========================================================================
 * Diagnostics
 * ========================================================================= */

/** Check whether a dual ray (certificate of primal infeasibility) exists, and
 *  retrieve it.  Wraps Highs_getDualRay().
 *  Returns a kHighsStatus constant. */
int32_t cobre_highs_get_dual_ray(
    const void* highs,
    int32_t*    has_dual_ray,
    double*     dual_ray_value
);

/** Check whether a primal ray (certificate of primal unboundedness) exists, and
 *  retrieve it.  Wraps Highs_getPrimalRay().
 *  Returns a kHighsStatus constant. */
int32_t cobre_highs_get_primal_ray(
    const void* highs,
    int32_t*    has_primal_ray,
    double*     primal_ray_value
);

/* =========================================================================
 * Info
 * ========================================================================= */

/** Return the number of columns in the incumbent model.
 *  Wraps Highs_getNumCol(). */
int32_t cobre_highs_get_num_col(const void* highs);

/** Return the number of rows in the incumbent model.
 *  Wraps Highs_getNumRow(). */
int32_t cobre_highs_get_num_row(const void* highs);

/* =========================================================================
 * Version query (no instance required)
 * ========================================================================= */

/** Return the HiGHS major version number.
 *  Wraps Highs_versionMajor(). */
int32_t cobre_highs_version_major(void);

/** Return the HiGHS minor version number.
 *  Wraps Highs_versionMinor(). */
int32_t cobre_highs_version_minor(void);

/** Return the HiGHS patch version number.
 *  Wraps Highs_versionPatch(). */
int32_t cobre_highs_version_patch(void);

#ifdef __cplusplus
}
#endif

#endif /* COBRE_HIGHS_WRAPPER_H */
