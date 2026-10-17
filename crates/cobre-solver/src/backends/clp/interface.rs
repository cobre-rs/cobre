use std::time::Instant;

use super::config::{ClpAlgorithm, ClpProfile};
use super::retry::LADDER_RUNGS;
use super::solver::{ClpSolver, clp_version, i32_from_usize};
use crate::{
    BasisStatus, SolverInterface, clp_ffi,
    types::{Basis, RowBatch, SolutionView, SolverError, SolverStatistics, StageTemplate},
};

impl SolverInterface for ClpSolver {
    type Profile = ClpProfile;

    /// Applies every [`ClpProfile`] field to the underlying CLP model, then
    /// caches the profile in `current_profile`. Perturbation `102` (not CLP's
    /// own default `100`) is required for bit-for-bit reproducible re-solves.
    fn apply_profile(&mut self, profile: &ClpProfile) {
        // Cache the profile first so `resolve_simplex_cap` reads the new
        // simplex iteration limit when computing the FFI cap below.
        self.current_profile = *profile;
        let cap = self.resolve_simplex_cap();
        // SAFETY: `self.handle` is a valid, non-null CLP pointer obtained from
        // `cobre_clp_create()`. Each `cobre_clp_set_*` setter accepts any
        // `i32`/`f64` value, retains no pointer after the call returns, and
        // cannot fail on a valid handle. `cap` is the resolved iteration limit.
        unsafe {
            clp_ffi::cobre_clp_set_perturbation(self.handle, profile.perturbation);
            clp_ffi::cobre_clp_scaling(self.handle, profile.scaling);
            clp_ffi::cobre_clp_set_primal_tolerance(
                self.handle,
                profile.primal_feasibility_tolerance,
            );
            clp_ffi::cobre_clp_set_dual_tolerance(self.handle, profile.dual_feasibility_tolerance);
            clp_ffi::cobre_clp_set_maximum_iterations(self.handle, cap);
        }

        // Skip mode 3 (CLP's steepest-edge ctor default) so the default profile
        // issues no shim call and stays byte-identical.
        if profile.dual_pricing_mode != 3 {
            self.set_dual_row_steepest(profile.dual_pricing_mode);
        }

        if profile.factorization_frequency != 0 {
            // SAFETY: `self.handle` is a valid, non-null CLP pointer from
            // `cobre_clp_create()`. The shim reaches the live `ClpSimplex`
            // through the wrapper's `model_` member and calls
            // `setFactorizationFrequency`, which stores the cadence on the
            // factorization object; it retains no pointer and cannot fail on a
            // valid handle.
            unsafe {
                clp_ffi::cobre_clp_set_factorization_frequency(
                    self.handle,
                    profile.factorization_frequency,
                );
            }
        }
    }

    /// Fully reset the CLP simplex state by recreating the underlying model.
    ///
    /// `Clp_loadProblem` swaps the model data but does NOT heal the
    /// `ClpSimplex`-level rim/pricing state, so stale steepest-edge reference
    /// weights persist and make the landed vertex on alternative-optima LPs
    /// depend on which models this handle solved before it — breaking
    /// thread/rank-count determinism. Recreating the `ClpSimplex` discards that
    /// state entirely; the cached profile is re-applied so configuration
    /// survives the swap.
    fn reset_solver_state(&mut self) {
        // SAFETY: `cobre_clp_create` has no preconditions; it allocates a new
        // empty CLP model or returns null on allocation failure.
        let new_handle = unsafe { clp_ffi::cobre_clp_create() };
        if new_handle.is_null() {
            // Allocation failed: keep the existing handle rather than abort;
            // determinism degrades but the run continues.
            return;
        }
        // SAFETY: `self.handle` is the valid handle from construction (or a prior
        // reset); `cobre_clp_destroy` frees it. It is immediately replaced by the
        // freshly created, non-null `new_handle` before any further use.
        unsafe { clp_ffi::cobre_clp_destroy(self.handle) };
        self.handle = new_handle;
        self.has_model = false;
        // SAFETY: `self.handle` is the just-created non-null model; mirror
        // `new()` by silencing CLP's per-solve logging.
        unsafe { clp_ffi::cobre_clp_set_log_level(self.handle, 0) };
        let profile = self.current_profile;
        self.apply_profile(&profile);
    }

    /// Loads a complete LP into the CLP model from column-major (CSC) data.
    ///
    /// The C wrapper owns the ±IEEE-inf → ±`DBL_MAX` bound translation and fixes
    /// the objective sense to minimize, so the bound slices are forwarded
    /// verbatim (`f64::INFINITY` is **not** pre-translated here). Re-calling
    /// replaces the prior model and resizes the solution buffers.
    ///
    /// # Panics
    ///
    /// Panics if `template.num_cols`, `template.num_rows`, or `template.num_nz`
    /// does not fit in `i32` (the LP exceeds the CLP C API limit).
    /// Also panics if a slice length disagrees with those counts.
    fn load_model(&mut self, template: &StageTemplate) {
        let t0 = Instant::now();
        assert!(
            i32::try_from(template.num_cols).is_ok(),
            "num_cols {} overflows i32: LP exceeds CLP API limit",
            template.num_cols
        );
        assert!(
            i32::try_from(template.num_rows).is_ok(),
            "num_rows {} overflows i32: LP exceeds CLP API limit",
            template.num_rows
        );
        assert!(
            i32::try_from(template.num_nz).is_ok(),
            "num_nz {} overflows i32: LP exceeds CLP API limit",
            template.num_nz
        );
        template.assert_ffi_shape();
        // Rationale: the values below were asserted to fit in i32 above.
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let num_col = template.num_cols as i32;
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let num_row = template.num_rows as i32;
        // SAFETY:
        // - `self.handle` is a valid, non-null CLP pointer from `cobre_clp_create()`.
        // - `num_col`/`num_row` fit in i32 (asserted above).
        // - All pointer arguments point into owned `Vec` data on `template` that
        //   remains alive for the duration of this call.
        // - Slice lengths match the CLP `Clp_loadProblem` contract
        //   (`StageTemplate::assert_ffi_shape`):
        //   `num_cols + 1` for col_starts, `num_nz` for row_indices and values,
        //   `num_cols` for col_lower/col_upper/objective, `num_rows` for
        //   row_lower/row_upper.
        // - Bounds are forwarded verbatim; the C wrapper owns the
        //   ±IEEE-inf → ±DBL_MAX translation and sets the objective sense.
        unsafe {
            clp_ffi::cobre_clp_load_problem(
                self.handle,
                num_col,
                num_row,
                template.col_starts.as_ptr(),
                template.row_indices.as_ptr(),
                template.values.as_ptr(),
                template.col_lower.as_ptr(),
                template.col_upper.as_ptr(),
                template.objective.as_ptr(),
                template.row_lower.as_ptr(),
                template.row_upper.as_ptr(),
            );
        }

        self.num_cols = template.num_cols;
        self.num_rows = template.num_rows;
        self.has_model = true;

        self.col_value.resize(self.num_cols, 0.0);
        self.col_dual.resize(self.num_cols, 0.0);
        self.row_dual.resize(self.num_rows, 0.0);

        // Clone the template CSC/bounds into the retained buffers — the
        // canonical, declaration-ordered mirror that `add_rows`/`set_*_bounds`
        // patch and reconcile into CLP natively.
        self.col_starts.clear();
        self.col_starts.extend_from_slice(&template.col_starts);
        self.row_indices.clear();
        self.row_indices.extend_from_slice(&template.row_indices);
        self.values.clear();
        self.values.extend_from_slice(&template.values);
        self.col_lower.clear();
        self.col_lower.extend_from_slice(&template.col_lower);
        self.col_upper.clear();
        self.col_upper.extend_from_slice(&template.col_upper);
        self.row_lower.clear();
        self.row_lower.extend_from_slice(&template.row_lower);
        self.row_upper.clear();
        self.row_upper.extend_from_slice(&template.row_upper);
        self.num_nz = template.num_nz;

        self.stats.total_load_model_time_seconds += t0.elapsed().as_secs_f64();
        self.stats.load_model_count += 1;
    }

    /// Appends a batch of constraint rows to the loaded LP.
    ///
    /// `rows` is CSR; the retained model is CSC. `cobre_clp_add_rows` takes the
    /// CSR batch directly, so the CSC transpose feeds only the retained mirror,
    /// not the FFI call. The native append preserves CLP's persistent simplex
    /// basis (no full rebuild).
    ///
    /// # Panics
    ///
    /// Panics if `rows.num_rows` or the batch nnz does not fit in `i32`, or if a
    /// slice length disagrees with the batch's CSR shape.
    fn add_rows(&mut self, rows: &RowBatch) {
        assert!(
            i32::try_from(rows.num_rows).is_ok(),
            "rows.num_rows {} overflows i32: RowBatch exceeds CLP API limit",
            rows.num_rows
        );
        assert!(
            i32::try_from(rows.col_indices.len()).is_ok(),
            "rows nnz {} overflows i32: RowBatch exceeds CLP API limit",
            rows.col_indices.len()
        );

        rows.assert_ffi_shape();

        let new_nz = rows.col_indices.len();
        if rows.num_rows == 0 {
            return;
        }

        let mut per_col_count = vec![0_usize; self.num_cols];
        for &col in &rows.col_indices {
            #[allow(clippy::cast_sign_loss)]
            let col = col as usize;
            debug_assert!(
                col < self.num_cols,
                "RowBatch column index {col} out of range [0, {})",
                self.num_cols
            );
            per_col_count[col] += 1;
        }

        let merged_nz = self.num_nz + new_nz;
        let mut new_col_starts = Vec::with_capacity(self.num_cols + 1);
        let mut new_row_indices = vec![0_i32; merged_nz];
        let mut new_values = vec![0.0_f64; merged_nz];

        let mut write_cursor = Vec::with_capacity(self.num_cols);
        let mut acc = 0_usize;
        for c in 0..self.num_cols {
            new_col_starts.push(i32_from_usize(acc));
            write_cursor.push(acc);
            #[allow(clippy::cast_sign_loss)]
            let old_start = self.col_starts[c] as usize;
            #[allow(clippy::cast_sign_loss)]
            let old_end = self.col_starts[c + 1] as usize;
            for k in old_start..old_end {
                new_row_indices[acc] = self.row_indices[k];
                new_values[acc] = self.values[k];
                acc += 1;
            }
            write_cursor[c] = acc;
            acc += per_col_count[c];
        }
        new_col_starts.push(i32_from_usize(acc));
        debug_assert_eq!(acc, merged_nz);

        // Appended rows occupy global row indices [num_rows, num_rows + n).
        for r in 0..rows.num_rows {
            #[allow(clippy::cast_sign_loss)]
            let start = rows.row_starts[r] as usize;
            #[allow(clippy::cast_sign_loss)]
            let end = rows.row_starts[r + 1] as usize;
            let global_row = self.num_rows + r;
            for k in start..end {
                #[allow(clippy::cast_sign_loss)]
                let col = rows.col_indices[k] as usize;
                let pos = write_cursor[col];
                new_row_indices[pos] = i32_from_usize(global_row);
                new_values[pos] = rows.values[k];
                write_cursor[col] += 1;
            }
        }

        self.col_starts = new_col_starts;
        self.row_indices = new_row_indices;
        self.values = new_values;
        self.row_lower.extend_from_slice(&rows.row_lower);
        self.row_upper.extend_from_slice(&rows.row_upper);
        self.num_rows += rows.num_rows;
        self.num_nz = merged_nz;

        // Rationale: `rows.num_rows` was asserted to fit in i32 above.
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let number = rows.num_rows as i32;
        // SAFETY:
        // - `self.handle` is a valid, non-null CLP pointer from
        //   `cobre_clp_create()` with a model loaded.
        // - `number` (== `rows.num_rows`) is non-negative and fits in i32
        //   (asserted at the top of this method).
        // - The pointer arguments point into the caller's `rows` CSR slices,
        //   which outlive this call: `row_lower`/`row_upper` have `num_rows`
        //   entries, `row_starts` has `num_rows + 1` entries, and
        //   `col_indices`/`values` have `row_starts[num_rows]` entries
        //   (`RowBatch::assert_ffi_shape`). The batch nnz was asserted to fit in i32.
        // - Row bounds are forwarded verbatim; the C wrapper owns the
        //   ±IEEE-inf → ±DBL_MAX translation.
        unsafe {
            clp_ffi::cobre_clp_add_rows(
                self.handle,
                number,
                rows.row_lower.as_ptr(),
                rows.row_upper.as_ptr(),
                rows.row_starts.as_ptr(),
                rows.col_indices.as_ptr(),
                rows.values.as_ptr(),
            );
        }

        self.row_dual.resize(self.num_rows, 0.0);
    }

    /// Patches the bounds of an arbitrary set of rows in place.
    ///
    /// An empty `indices` slice is a no-op. The retained `row_lower`/`row_upper`
    /// (the canonical mirror) is patched, then the **full** bound vectors are
    /// pushed into CLP via `cobre_clp_chg_row_lower`/`cobre_clp_chg_row_upper`,
    /// which take the whole array, not an index subset. This preserves CLP's
    /// factorization/basis across the patch.
    ///
    /// # Panics
    ///
    /// Panics if the three slices differ in length, or if any index is out of
    /// range (debug builds).
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

        let t0 = Instant::now();
        for (i, &row) in indices.iter().enumerate() {
            debug_assert!(
                row < self.num_rows,
                "set_row_bounds: index {row} out of range [0, {})",
                self.num_rows
            );
            self.row_lower[row] = lower[i];
            self.row_upper[row] = upper[i];
        }
        // SAFETY:
        // - `self.handle` is a valid, non-null CLP pointer with a model loaded.
        // - `self.row_lower`/`self.row_upper` each have exactly `self.num_rows`
        //   entries (maintained by `load_model`/`add_rows`), matching the model's
        //   current row count that the C wrapper queries to size its translation.
        // - The pointers reference owned `Vec` data alive for the call.
        // - Bounds are forwarded verbatim; the C wrapper owns the
        //   ±IEEE-inf → ±DBL_MAX translation.
        unsafe {
            clp_ffi::cobre_clp_chg_row_lower(self.handle, self.row_lower.as_ptr());
            clp_ffi::cobre_clp_chg_row_upper(self.handle, self.row_upper.as_ptr());
        }
        self.stats.total_set_bounds_time_seconds += t0.elapsed().as_secs_f64();
    }

    /// Patches the bounds of an arbitrary set of columns in place.
    ///
    /// Symmetric to [`Self::set_row_bounds`], patching the retained
    /// `col_lower`/`col_upper` and pushing them into CLP via
    /// `cobre_clp_chg_column_lower`/`cobre_clp_chg_column_upper`.
    ///
    /// # Panics
    ///
    /// Panics if the three slices differ in length, or if any index is out of
    /// range (debug builds).
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

        let t0 = Instant::now();
        for (i, &col) in indices.iter().enumerate() {
            debug_assert!(
                col < self.num_cols,
                "set_col_bounds: index {col} out of range [0, {})",
                self.num_cols
            );
            self.col_lower[col] = lower[i];
            self.col_upper[col] = upper[i];
        }

        // SAFETY:
        // - `self.handle` is a valid, non-null CLP pointer with a model loaded.
        // - `self.col_lower`/`self.col_upper` each have exactly `self.num_cols`
        //   entries (maintained by `load_model`), matching the model's current
        //   column count that the C wrapper queries to size its translation.
        // - The pointers reference owned `Vec` data alive for the call.
        // - Bounds are forwarded verbatim; the C wrapper owns the
        //   ±IEEE-inf → ±DBL_MAX translation.
        unsafe {
            clp_ffi::cobre_clp_chg_column_lower(self.handle, self.col_lower.as_ptr());
            clp_ffi::cobre_clp_chg_column_upper(self.handle, self.col_upper.as_ptr());
        }
        self.stats.total_set_bounds_time_seconds += t0.elapsed().as_secs_f64();
    }

    /// Solves the loaded LP and returns the optimal solution as a
    /// [`SolutionView`] borrowing the solver's owned buffers.
    ///
    /// On `CLP_STATUS_OPTIMAL` the three CLP-owned solution pointers are copied
    /// **immediately** into `col_value`/`col_dual`/`row_dual` (valid only until
    /// the next solve).
    ///
    /// # Escalation ladder
    ///
    /// CLP's bare dual simplex can spuriously report `PRIMAL_INFEASIBLE` on
    /// numerically delicate feasible LPs. On `PRIMAL_INFEASIBLE` or `STOPPED`
    /// the failure path runs `escalate_solve`. A solve recovered by the ladder
    /// returns `Ok` and counts as a retried success (`success_count` +
    /// `retry_count`, never `first_try_successes`). `DUAL_INFEASIBLE`, `ERRORS`,
    /// and unexpected statuses stay terminal. The floor (deterministic) settings
    /// are re-applied after the ladder runs regardless of outcome.
    ///
    /// # Warm-start basis
    ///
    /// When `basis = Some(b)`, `b` is reinstalled via `install_basis` before the
    /// solve. An undersized row basis (`b.row_status.len() < self.num_rows`) is
    /// rejected with `Err(SolverError::BasisRowCountMismatch)` rather than
    /// silently short-copied; the caller should fall back to a cold solve.
    ///
    /// # Errors
    ///
    /// Returns `Err(SolverError::Infeasible)` on `PRIMAL_INFEASIBLE` that the
    /// escalation ladder could not recover, `Err(SolverError::Unbounded)` on
    /// `DUAL_INFEASIBLE`, `Err(SolverError::IterationLimit { .. })` on `STOPPED`
    /// the ladder could not recover, and `Err(SolverError::InternalError { .. })`
    /// on `ERRORS` or any unexpected status int. Returns
    /// `Err(SolverError::BasisRowCountMismatch { .. })` when an offered warm-start
    /// basis has fewer row entries than the LP has rows.
    ///
    /// # Panics
    ///
    /// Panics if no model is loaded (`!self.has_model`).
    fn solve(&mut self, basis: Option<&Basis>) -> Result<SolutionView<'_>, SolverError> {
        assert!(self.has_model, "solve called without a loaded model");

        if let Some(b) = basis {
            self.install_basis(b)?;
        }

        let t0 = Instant::now();
        // Both dual and primal return the same `Clp_status` int space, so the
        // status mapping below is shared.
        let status = match self.current_profile.algorithm {
            ClpAlgorithm::Dual => {
                // SAFETY: `self.handle` is a valid, non-null CLP pointer from
                // `cobre_clp_create()` with a model loaded (asserted via
                // `has_model`). `if_values_pass = 0` requests a cold solve (no
                // values pass). The returned int is the CLP solve status.
                unsafe { clp_ffi::cobre_clp_dual(self.handle, 0) }
            }
            ClpAlgorithm::Primal => {
                // SAFETY: `self.handle` is a valid, non-null CLP pointer from
                // `cobre_clp_create()` with a model loaded (asserted via
                // `has_model`). `if_values_pass = 0` requests a cold solve (no
                // values pass). The returned int is the CLP solve status.
                unsafe { clp_ffi::cobre_clp_primal(self.handle, 0) }
            }
        };
        let solve_time = t0.elapsed().as_secs_f64();

        self.stats.solve_count += 1;

        if status == clp_ffi::CLP_STATUS_OPTIMAL {
            // Read iterations/objective BEFORE the shared borrow via the returned
            // `SolutionView`, so stats can be updated without violating aliasing.
            // SAFETY: `self.handle` is a valid, non-null CLP pointer that has
            // just been solved; iteration count is non-negative so the cast is
            // safe.
            #[allow(clippy::cast_sign_loss)]
            let iterations = unsafe { clp_ffi::cobre_clp_number_iterations(self.handle) } as u64;
            // SAFETY: `self.handle` is a valid, non-null CLP pointer that has
            // just been solved. Objective is already in minimize sense (the
            // wrapper set the optimization direction at load); returned as-is.
            let objective = unsafe { clp_ffi::cobre_clp_objective_value(self.handle) };

            self.copy_solution();

            self.stats.success_count += 1;
            self.stats.first_try_successes += 1;
            self.stats.total_iterations += iterations;
            self.stats.total_solve_time_seconds += solve_time;

            return Ok(SolutionView {
                objective,
                primal: &self.col_value[..self.num_cols],
                dual: &self.row_dual[..self.num_rows],
                reduced_costs: &self.col_dual[..self.num_cols],
                iterations,
                solve_time_seconds: solve_time,
            });
        }

        // `failure_count` is deliberately NOT incremented before the ladder: a
        // solve recovered by escalation counts only as a (retried) success. It is
        // incremented on the terminal paths below and on ladder exhaustion.
        if status == clp_ffi::CLP_STATUS_PRIMAL_INFEASIBLE || status == clp_ffi::CLP_STATUS_STOPPED
        {
            let outcome = self.escalate_solve();

            // Restore the floor (deterministic) settings unconditionally —
            // success OR exhaustion — so the NEXT `solve` starts from the clean
            // config the happy path depends on. Re-applying `current_profile`
            // turns perturbation and scaling back off; it reassigns
            // `self.current_profile` to the value it already holds, so the floor
            // is byte-identical to a build that never escalated.
            let profile = self.current_profile;
            self.apply_profile(&profile);

            if let Some(escalation) = outcome {
                self.stats.success_count += 1;
                self.stats.retry_count += escalation.attempts;
                self.stats.total_iterations += escalation.iterations;
                self.stats.total_solve_time_seconds += escalation.solve_time;

                return Ok(SolutionView {
                    objective: escalation.objective,
                    primal: &self.col_value[..self.num_cols],
                    dual: &self.row_dual[..self.num_rows],
                    reduced_costs: &self.col_dual[..self.num_cols],
                    iterations: escalation.iterations,
                    solve_time_seconds: escalation.solve_time,
                });
            }

            // Ladder exhausted: surface the ORIGINAL error, charging the
            // attempted rungs and the final failure. `LADDER_RUNGS` (<= 5) widens
            // losslessly to the `u64` `retry_count`.
            self.stats.retry_count += LADDER_RUNGS as u64;
            self.stats.failure_count += 1;
            if status == clp_ffi::CLP_STATUS_PRIMAL_INFEASIBLE {
                return Err(SolverError::Infeasible);
            }
            // STOPPED: map to IterationLimit using the last solve's iteration count.
            // SAFETY: `self.handle` is a valid, non-null CLP pointer; iteration
            // count is non-negative so the cast is safe.
            #[allow(clippy::cast_sign_loss)]
            let iterations = unsafe { clp_ffi::cobre_clp_number_iterations(self.handle) } as u64;
            return Err(SolverError::IterationLimit { iterations });
        }

        self.stats.failure_count += 1;
        match status {
            clp_ffi::CLP_STATUS_DUAL_INFEASIBLE => Err(SolverError::Unbounded),
            clp_ffi::CLP_STATUS_ERRORS => Err(SolverError::InternalError {
                message: "CLP solve failed (simplex returned ERRORS status)".to_string(),
                error_code: Some(4),
            }),
            other => Err(SolverError::InternalError {
                message: format!("CLP returned unexpected status {other}"),
                error_code: Some(other),
            }),
        }
    }

    /// Extracts the current simplex basis into `out`, element-by-element.
    ///
    /// CLP reports basis status one element at a time (no bulk array in the
    /// wrapper). Each native `CLP_BASIS_*` code is mapped via
    /// [`BasisStatus::from_clp_code`] into the canonical status stored in `out`.
    ///
    /// # Panics
    ///
    /// Panics if no model is loaded (`!self.has_model`).
    fn get_basis(&mut self, out: &mut Basis) {
        assert!(
            self.has_model,
            "get_basis called without a loaded model — call load_model first"
        );

        out.col_status.resize(self.num_cols, BasisStatus::Lower);
        out.row_status.resize(self.num_rows, BasisStatus::Lower);

        // Loop indices are bounded by `num_cols`/`num_rows`, both asserted to
        // fit in i32 by `load_model`; the casts cannot truncate or wrap.
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        for c in 0..self.num_cols {
            // SAFETY: `self.handle` is a valid, non-null CLP pointer with a model
            // loaded (asserted via `has_model`); `c` is in `0..num_cols`, a valid
            // column sequence index, and fits in i32. The getter reads a single
            // status byte and returns it widened to i32.
            let code = unsafe { clp_ffi::cobre_clp_get_column_status(self.handle, c as i32) };
            out.col_status[c] = BasisStatus::from_clp_code(code);
        }
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        for r in 0..self.num_rows {
            // SAFETY: `self.handle` is a valid, non-null CLP pointer with a model
            // loaded; `r` is in `0..num_rows`, a valid row sequence index, and fits
            // in i32. The getter reads a single status byte and returns it widened.
            let code = unsafe { clp_ffi::cobre_clp_get_row_status(self.handle, r as i32) };
            out.row_status[r] = BasisStatus::from_clp_code(code);
        }
    }

    /// Returns a snapshot of the accumulated solver statistics.
    fn statistics(&self) -> SolverStatistics {
        self.stats.clone()
    }

    fn statistics_into(&self, out: &mut SolverStatistics) {
        out.copy_from(&self.stats);
    }

    /// Returns a static string identifying the solver backend.
    fn name(&self) -> &'static str {
        "CLP"
    }

    /// Returns the solver name and version as a human-readable string.
    fn solver_name_version(&self) -> String {
        format!("CLP {}", clp_version())
    }

    // `record_reconstruction_stats` is intentionally NOT overridden — CLP has no
    // slot-reconciliation basis reconstruction, so the trait default no-op holds.
}
