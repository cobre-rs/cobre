//! `impl SolverInterface for HighsSolver`.

use std::time::Instant;

use super::config::HighsProfile;
use super::solver::{HighsSolver, highs_version};
use crate::types::Basis;
use crate::{
    BasisStatus, SolverInterface, ffi,
    types::{RowBatch, SolutionView, SolverError, SolverStatistics, StageTemplate},
};

impl SolverInterface for HighsSolver {
    type Profile = HighsProfile;

    fn apply_profile(&mut self, profile: &HighsProfile) {
        // SAFETY: `self.handle` is a valid, non-null HiGHS pointer obtained
        // from `cobre_highs_create()`. The option name is a static C string
        // literal with no retained pointer after the call returns.
        unsafe {
            ffi::cobre_highs_set_double_option(
                self.handle,
                c"primal_feasibility_tolerance".as_ptr(),
                profile.primal_feasibility_tolerance,
            );
            ffi::cobre_highs_set_double_option(
                self.handle,
                c"dual_feasibility_tolerance".as_ptr(),
                profile.dual_feasibility_tolerance,
            );
        }
        // No FFI for the iteration-limit fields: `set_iteration_limits` computes
        // those caps per solve from the cached `current_profile`.
        // SAFETY: self.handle is a valid HiGHS pointer; ffi setters accept any i32.
        unsafe {
            ffi::cobre_highs_set_int_option(
                self.handle,
                c"simplex_dual_edge_weight_strategy".as_ptr(),
                profile.simplex_dual_edge_weight_strategy,
            );
            ffi::cobre_highs_set_int_option(
                self.handle,
                c"simplex_scale_strategy".as_ptr(),
                profile.simplex_scale_strategy,
            );
            ffi::cobre_highs_set_int_option(
                self.handle,
                c"simplex_price_strategy".as_ptr(),
                profile.simplex_price_strategy,
            );
        }
        // SAFETY: self.handle is a valid, non-null HiGHS pointer; option names
        // are static C strings with no retained pointer after the call
        // returns; `simplex_update_limit` is clamped to `i32::MAX` before the
        // u32 -> i32 cast so the cast cannot wrap.
        unsafe {
            ffi::cobre_highs_set_string_option(
                self.handle,
                c"presolve".as_ptr(),
                profile.presolve.as_option().as_ptr(),
            );
            ffi::cobre_highs_set_bool_option(
                self.handle,
                c"use_warm_start".as_ptr(),
                i32::from(profile.use_warm_start),
            );
            #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
            let simplex_update_limit = profile.simplex_update_limit.min(i32::MAX as u32) as i32;
            ffi::cobre_highs_set_int_option(
                self.handle,
                c"simplex_update_limit".as_ptr(),
                simplex_update_limit,
            );
            ffi::cobre_highs_set_double_option(
                self.handle,
                c"dual_simplex_cost_perturbation_multiplier".as_ptr(),
                profile.cost_perturbation,
            );
            ffi::cobre_highs_set_double_option(
                self.handle,
                c"rebuild_refactor_solution_error_tolerance".as_ptr(),
                profile.refactor_error_tolerance,
            );
            ffi::cobre_highs_set_double_option(
                self.handle,
                c"factor_pivot_threshold".as_ptr(),
                profile.factor_pivot_threshold,
            );
            ffi::cobre_highs_set_double_option(
                self.handle,
                c"dual_steepest_edge_weight_log_error_threshold".as_ptr(),
                profile.steepest_edge_devex_fallback_threshold,
            );
        }
        self.current_profile = *profile;
    }

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
        // - All slice lengths match the HiGHS API contract (`StageTemplate::assert_ffi_shape`):
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
        template.assert_ffi_shape();
        // Scale vectors are optional: empty means "no scaling", otherwise they must be
        // keyed by the matching dimension.
        debug_assert!(
            template.col_scale.is_empty() || template.col_scale.len() == template.num_cols,
            "col_scale len {} != num_cols {} (and is non-empty)",
            template.col_scale.len(),
            template.num_cols
        );
        debug_assert!(
            template.row_scale.is_empty() || template.row_scale.len() == template.num_rows,
            "row_scale len {} != num_rows {} (and is non-empty)",
            template.row_scale.len(),
            template.num_rows
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

        self.col_value.resize(self.num_cols, 0.0);
        self.col_dual.resize(self.num_cols, 0.0);
        self.row_value.resize(self.num_rows, 0.0);
        self.row_dual.resize(self.num_rows, 0.0);

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
        rows.assert_ffi_shape();
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
        // - Slice lengths (`RowBatch::assert_ffi_shape`): `num_rows + 1` for starts,
        //   total nnz for index and value, `num_rows` for lower/upper bounds.
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

        self.row_value.resize(self.num_rows, 0.0);
        self.row_dual.resize(self.num_rows, 0.0);
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

    /// Under a profile with `use_warm_start == false` the offered `basis` is
    /// ignored and the solve runs cold: nothing is installed, validated, or
    /// counted in `basis_offered`.
    ///
    /// # Preconditions
    ///
    /// When `basis` is `Some(b)`, the caller should size `b.row_status` to at
    /// least `self.num_rows` (the current LP row count). A basis with **fewer**
    /// row entries than the LP (e.g. one captured before `add_rows` grew the LP)
    /// cannot be padded soundly — a BASIC pad is wrong for inequality-row slacks
    /// — so it is rejected with `Err(SolverError::BasisRowCountMismatch)` and
    /// `basis_consistency_failures` is incremented; the caller should fall back
    /// to a cold solve. A basis with **more** row entries is tolerated: the
    /// trailing entries beyond `self.num_rows` are ignored. The column count
    /// must match exactly (hard `assert!`).
    ///
    /// # Errors
    ///
    /// Returns `Err(SolverError::BasisRowCountMismatch { lp_rows, basis_rows })`
    /// when the offered basis has fewer row entries than the LP has rows, and
    /// `Err(SolverError::BasisInconsistent { .. })` when `HiGHS` rejects the
    /// offered basis via `isBasisConsistent`.
    fn solve(&mut self, basis: Option<&Basis>) -> Result<SolutionView<'_>, SolverError> {
        assert!(
            self.has_model,
            "solve called without a loaded model — call load_model first"
        );

        // HiGHS clears an installed basis when `use_warm_start` is off, so
        // installing and counting it would report a warm start that never ran.
        let basis = basis.filter(|_| self.current_profile.use_warm_start);
        if let Some(basis) = basis {
            assert!(
                basis.col_status.len() == self.num_cols,
                "basis column count {} does not match LP column count {}",
                basis.col_status.len(),
                self.num_cols
            );
            // Runs before `basis_offered` increments: a rejected basis was never
            // offered.
            if basis.row_status.len() < self.num_rows {
                self.stats.basis_consistency_failures += 1;
                return Err(SolverError::BasisRowCountMismatch {
                    lp_rows: self.num_rows,
                    basis_rows: basis.row_status.len(),
                });
            }

            self.stats.basis_offered += 1;

            for (dst, status) in self.basis_col_i32[..self.num_cols]
                .iter_mut()
                .zip(&basis.col_status)
            {
                *dst = status.to_highs_code();
            }

            // Undersized is rejected above, so `basis_rows >= lp_rows` here.
            let basis_rows = basis.row_status.len();
            let lp_rows = self.num_rows;
            let copy_len = basis_rows.min(lp_rows);
            for (dst, status) in self.basis_row_i32[..copy_len]
                .iter_mut()
                .zip(&basis.row_status[..copy_len])
            {
                *dst = status.to_highs_code();
            }

            // SAFETY:
            // - `self.handle` is a valid, non-null HiGHS pointer obtained from
            //   `cobre_highs_create()` and kept alive by `HighsSolver`.
            // - `basis_col_i32` was sized to `num_cols` in `load_model` and grown in
            //   `add_rows`; the slice written above covers exactly `num_cols` entries.
            // - `basis_row_i32` was sized to `num_rows` in `load_model` and grown in
            //   `add_rows`; the slice written above covers exactly `num_rows` entries
            //   (an undersized basis is rejected before reaching this point).
            let basis_set_start = Instant::now();
            let set_status = unsafe {
                ffi::cobre_highs_set_basis_non_alien(
                    self.handle,
                    self.basis_col_i32.as_ptr(),
                    self.basis_row_i32.as_ptr(),
                )
            };
            if set_status == ffi::HIGHS_STATUS_ERROR {
                // Non-alien rejected: the basis failed `isBasisConsistent`
                // (total_basic != num_row). Surface it as a hard error.
                self.stats.basis_consistency_failures += 1;
                // `usize` -> `i64` is lossless for any basis that fits in memory.
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
                // Accumulate the elapsed time even on this early return.
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

        self.solve_inner()
    }

    fn get_basis(&mut self, out: &mut Basis) {
        assert!(
            self.has_model,
            "get_basis called without a loaded model — call load_model first"
        );

        // SAFETY:
        // - `self.handle` is a valid, non-null HiGHS pointer.
        // - `basis_col_i32`/`basis_row_i32` are sized to `num_cols`/`num_rows` by
        //   `load_model`/`add_rows`.
        // - HiGHS writes exactly `num_cols` col values and `num_rows` row values.
        let get_status = unsafe {
            ffi::cobre_highs_get_basis(
                self.handle,
                self.basis_col_i32.as_mut_ptr(),
                self.basis_row_i32.as_mut_ptr(),
            )
        };

        assert_ne!(
            get_status,
            ffi::HIGHS_STATUS_ERROR,
            "cobre_highs_get_basis failed: basis must exist after a successful solve (programming error)"
        );

        out.col_status.resize(self.num_cols, BasisStatus::Lower);
        for (dst, &code) in out.col_status.iter_mut().zip(&self.basis_col_i32) {
            *dst = BasisStatus::from_highs_code(code);
        }
        out.row_status.resize(self.num_rows, BasisStatus::Lower);
        for (dst, &code) in out.row_status.iter_mut().zip(&self.basis_row_i32) {
            *dst = BasisStatus::from_highs_code(code);
        }
    }

    fn statistics(&self) -> SolverStatistics {
        self.stats.clone()
    }

    fn statistics_into(&self, out: &mut SolverStatistics) {
        out.copy_from(&self.stats);
    }

    fn record_reconstruction_stats(&mut self) {
        self.stats.basis_reconstructions += 1;
    }
}
