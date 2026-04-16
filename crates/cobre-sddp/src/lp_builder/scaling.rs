use cobre_core::Stage;
use cobre_solver::StageTemplate;
use cobre_stochastic::par::precompute::PrecomputedPar;

use super::M3S_TO_HM3;

/// Compute per-column geometric-mean scaling factors from a CSC constraint matrix.
///
/// For each column `j`, the scale factor is `1 / sqrt(max|A_ij| * min|A_ij|)` over
/// nonzero entries. Columns with no nonzero entries receive a scale factor of 1.0.
///
/// The returned vector has length `num_cols`. Applying column scaling transforms the
/// LP: multiply each column's matrix entries, objective coefficient, and column bounds
/// by the corresponding scale factor.
#[must_use]
#[allow(clippy::cast_sign_loss)] // col_starts are non-negative by CSC construction
pub(crate) fn compute_col_scale(num_cols: usize, col_starts: &[i32], values: &[f64]) -> Vec<f64> {
    let mut scale = vec![1.0_f64; num_cols];
    for j in 0..num_cols {
        let start = col_starts[j] as usize;
        let end = col_starts[j + 1] as usize;
        if start == end {
            // No nonzero entries in this column.
            continue;
        }
        let mut max_abs = 0.0_f64;
        let mut min_abs = f64::INFINITY;
        for &v in &values[start..end] {
            let abs_val = v.abs();
            if abs_val > 0.0 {
                max_abs = max_abs.max(abs_val);
                min_abs = min_abs.min(abs_val);
            }
        }
        if max_abs > 0.0 && min_abs < f64::INFINITY {
            let d = 1.0 / (max_abs * min_abs).sqrt();
            scale[j] = d;
        }
        // Otherwise keep 1.0 (all structural zeros or defensive fallback).
    }
    scale
}

/// Apply column scaling to a stage template's matrix, objective, and bounds.
///
/// Modifies the template in-place. After this call:
/// - `values[k]` has been multiplied by `col_scale[col_of(k)]`
/// - `objective[j]` has been multiplied by `col_scale[j]`
/// - `col_lower[j]` has been divided by `col_scale[j]`
/// - `col_upper[j]` has been divided by `col_scale[j]`
///
/// Infinite bounds remain infinite (dividing infinity by a finite positive
/// scale factor yields infinity).
pub(crate) fn apply_col_scale(template: &mut StageTemplate, col_scale: &[f64]) {
    let num_cols = template.num_cols;
    debug_assert_eq!(col_scale.len(), num_cols);

    // Scale matrix values (CSC: iterate columns).
    #[allow(clippy::needless_range_loop, clippy::cast_sign_loss)]
    // j+1 access; col_starts non-negative by construction
    for j in 0..num_cols {
        let start = template.col_starts[j] as usize;
        let end = template.col_starts[j + 1] as usize;
        let d = col_scale[j];
        for v in &mut template.values[start..end] {
            *v *= d;
        }
    }

    // Scale objective coefficients.
    for (obj, &d) in template.objective.iter_mut().zip(col_scale) {
        *obj *= d;
    }

    // Inverse-scale column bounds.
    // The scaled variable is x_tilde = x / d_j, so bounds become [lo/d, hi/d].
    // For d > 0 this preserves bound ordering.
    for ((lo, hi), &d) in template
        .col_lower
        .iter_mut()
        .zip(template.col_upper.iter_mut())
        .zip(col_scale)
    {
        *lo /= d;
        *hi /= d;
    }
}

/// Compute per-row geometric-mean scaling factors from a CSC constraint matrix.
///
/// For each row `i`, the scale factor is `1 / sqrt(max|A_ij| * min|A_ij|)` over
/// all nonzero entries in that row. Rows with no nonzero entries receive a scale
/// factor of 1.0.
///
/// The matrix is given in CSC (column-major) form; row statistics are accumulated
/// by iterating all nonzeros once in O(nnz). This function should be called on
/// the already column-scaled matrix to obtain the standard `D_r * A * D_c` form.
///
/// The returned vector has length `num_rows`. Applying row scaling transforms the
/// LP: multiply each row's matrix entries, row lower bound, and row upper bound
/// by the corresponding scale factor.
#[must_use]
#[allow(clippy::cast_sign_loss)] // col_starts and row_indices are non-negative by CSC construction
pub(crate) fn compute_row_scale(
    num_rows: usize,
    num_cols: usize,
    col_starts: &[i32],
    row_indices: &[i32],
    values: &[f64],
) -> Vec<f64> {
    let mut row_max = vec![0.0_f64; num_rows];
    let mut row_min = vec![f64::INFINITY; num_rows];

    #[allow(clippy::needless_range_loop)] // j+1 access on col_starts requires index
    for j in 0..num_cols {
        let start = col_starts[j] as usize;
        let end = col_starts[j + 1] as usize;
        for k in start..end {
            let row = row_indices[k] as usize;
            let abs_val = values[k].abs();
            if abs_val > 0.0 {
                row_max[row] = row_max[row].max(abs_val);
                row_min[row] = row_min[row].min(abs_val);
            }
        }
    }

    let mut scale = vec![1.0_f64; num_rows];
    for (s, (&rmax, &rmin)) in scale.iter_mut().zip(row_max.iter().zip(row_min.iter())) {
        if rmax > 0.0 && rmin < f64::INFINITY {
            *s = 1.0 / (rmax * rmin).sqrt();
        }
        // Otherwise keep 1.0 (empty row or all structural zeros).
    }
    scale
}

/// Apply row scaling to a stage template's matrix and row bounds.
///
/// Modifies the template in-place. After this call:
/// - `values[k]` has been multiplied by `row_scale[row_of(k)]`
/// - `row_lower[i]` has been multiplied by `row_scale[i]`
/// - `row_upper[i]` has been multiplied by `row_scale[i]`
///
/// Infinite bounds remain infinite (multiplying infinity by a finite positive
/// scale factor yields infinity).
///
/// The objective and column bounds are not modified — those are column-domain
/// quantities already handled by column scaling.
pub(crate) fn apply_row_scale(template: &mut StageTemplate, row_scale: &[f64]) {
    let num_rows = template.num_rows;
    debug_assert_eq!(row_scale.len(), num_rows);

    // Scale matrix values (CSC: iterate columns, apply per-row factor).
    let num_cols = template.num_cols;
    #[allow(clippy::needless_range_loop, clippy::cast_sign_loss)]
    // j+1 access; values non-negative by construction
    for j in 0..num_cols {
        let start = template.col_starts[j] as usize;
        let end = template.col_starts[j + 1] as usize;
        for k in start..end {
            let row = template.row_indices[k] as usize;
            template.values[k] *= row_scale[row];
        }
    }

    // Scale row bounds.
    for ((lo, hi), &d) in template
        .row_lower
        .iter_mut()
        .zip(template.row_upper.iter_mut())
        .zip(row_scale)
    {
        *lo *= d;
        *hi *= d;
    }
}

/// Pre-compute `ζ * σ` per `(stage, hydro)` for noise transformation.
///
/// Returns `(noise_scale, zeta_per_stage, block_hours_per_stage)`.  The
/// `noise_scale` flat vector has layout `[s_idx * n_hydros + h_idx]` so that
/// the forward pass can index it without branching.
pub(super) fn compute_noise_scale(
    study_stages: &[&Stage],
    n_hydros: usize,
    par_lp: &PrecomputedPar,
) -> (Vec<f64>, Vec<f64>, Vec<Vec<f64>>) {
    let n = study_stages.len();
    let mut noise_scale = vec![0.0_f64; n * n_hydros];
    let mut zeta_per_stage = Vec::with_capacity(n);
    let mut block_hours_per_stage = Vec::with_capacity(n);

    for (s_idx, stage) in study_stages.iter().enumerate() {
        let total_hours: f64 = stage.blocks.iter().map(|b| b.duration_hours).sum();
        let zeta_s = total_hours * M3S_TO_HM3;
        zeta_per_stage.push(zeta_s);
        block_hours_per_stage.push(stage.blocks.iter().map(|b| b.duration_hours).collect());
        for h_idx in 0..n_hydros {
            let sigma = if par_lp.n_stages() > 0 && par_lp.n_hydros() == n_hydros {
                par_lp.sigma(s_idx, h_idx)
            } else {
                0.0
            };
            noise_scale[s_idx * n_hydros + h_idx] = zeta_s * sigma;
        }
    }

    (noise_scale, zeta_per_stage, block_hours_per_stage)
}

#[cfg(test)]
#[allow(
    clippy::doc_markdown,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation
)]
mod tests {
    use cobre_solver::StageTemplate;

    // =========================================================================
    // Row scaling tests (ticket E3-001)
    // =========================================================================

    /// Build a minimal `StageTemplate` for row-scaling unit tests.
    ///
    /// The matrix is given in CSC form.  All non-LP-semantic fields are zeroed
    /// so the helpers under test only touch the fields they care about.
    fn minimal_template(
        num_rows: usize,
        num_cols: usize,
        col_starts: Vec<i32>,
        row_indices: Vec<i32>,
        values: Vec<f64>,
        row_lower: Vec<f64>,
        row_upper: Vec<f64>,
    ) -> StageTemplate {
        let num_nz = values.len();
        StageTemplate {
            num_cols,
            num_rows,
            num_nz,
            col_starts,
            row_indices,
            values,
            col_lower: vec![0.0; num_cols],
            col_upper: vec![f64::INFINITY; num_cols],
            objective: vec![0.0; num_cols],
            row_lower,
            row_upper,
            n_transfer: 0,
            n_dual_relevant: 0,
            n_hydro: 0,
            max_par_order: 0,
            col_scale: Vec::new(),
            row_scale: Vec::new(),
            n_state: 0,
        }
    }

    /// AC E3-001-1: a matrix where every row has min_abs == max_abs gives scale 1.0.
    ///
    /// Matrix (2 rows × 2 cols, column-major):
    ///
    /// ```text
    /// col 0: row 0 → 3.0, row 1 → 3.0
    /// col 1: row 0 → 3.0, row 1 → 3.0
    /// ```
    ///
    /// For each row: min_abs = max_abs = 3.0 → scale = 1/sqrt(3*3) = 1/3.
    /// Wait — "uniform" means min == max, so scale = 1/sqrt(max*min) = 1/max.
    /// With all values 1.0: scale = 1/sqrt(1*1) = 1.0.
    #[test]
    fn row_scale_identity_for_uniform_matrix() {
        // All nonzeros have |value| = 1.0.  min_abs = max_abs = 1.0.
        // scale[i] = 1/sqrt(1.0 * 1.0) = 1.0 for every row.
        let col_starts = vec![0, 2, 4];
        let row_indices = vec![0, 1, 0, 1];
        let values = vec![1.0, 1.0, 1.0, 1.0];
        let scale = super::compute_row_scale(2, 2, &col_starts, &row_indices, &values);
        assert_eq!(scale.len(), 2);
        assert!(
            (scale[0] - 1.0).abs() < 1e-15,
            "row 0 scale should be 1.0, got {}",
            scale[0]
        );
        assert!(
            (scale[1] - 1.0).abs() < 1e-15,
            "row 1 scale should be 1.0, got {}",
            scale[1]
        );
    }

    /// AC E3-001-2: geometric-mean scale matches expected value for known matrix.
    ///
    /// Matrix (2 rows × 2 cols):
    ///
    /// ```text
    /// col 0: row 0 → 1.0
    /// col 1: row 0 → 100.0, row 1 → 4.0
    /// ```
    ///
    /// Row 0: min_abs = 1.0, max_abs = 100.0 → scale = 1/sqrt(100) = 0.1
    /// Row 1: min_abs = max_abs = 4.0         → scale = 1/sqrt(16)  = 0.25
    #[test]
    fn row_scale_geometric_mean() {
        let col_starts = vec![0, 1, 3];
        let row_indices = vec![0, 0, 1];
        let values = vec![1.0, 100.0, 4.0];
        let scale = super::compute_row_scale(2, 2, &col_starts, &row_indices, &values);
        assert_eq!(scale.len(), 2);
        let expected_row0 = 1.0_f64 / (1.0_f64 * 100.0_f64).sqrt(); // 0.1
        let expected_row1 = 1.0_f64 / (4.0_f64 * 4.0_f64).sqrt(); // 0.25
        assert!(
            (scale[0] - expected_row0).abs() < 1e-14,
            "row 0 scale: expected {expected_row0}, got {}",
            scale[0]
        );
        assert!(
            (scale[1] - expected_row1).abs() < 1e-14,
            "row 1 scale: expected {expected_row1}, got {}",
            scale[1]
        );
    }

    /// AC E3-001-3: `apply_row_scale` multiplies matrix values and row bounds.
    ///
    /// Uses the same 2×2 matrix as `row_scale_geometric_mean` so the expected
    /// values are easily verified by hand.
    #[test]
    fn apply_row_scale_scales_values_and_bounds() {
        // CSC: col 0 has one nonzero (row 0, val 1.0); col 1 has two (row 0→100.0, row 1→4.0).
        let col_starts = vec![0_i32, 1, 3];
        let row_indices = vec![0_i32, 0, 1];
        let values = vec![1.0_f64, 100.0, 4.0];
        let row_lower = vec![-5.0_f64, 7.0];
        let row_upper = vec![f64::INFINITY, 7.0];

        let mut tmpl =
            minimal_template(2, 2, col_starts, row_indices, values, row_lower, row_upper);

        // Row 0: scale = 1/sqrt(1*100) = 0.1
        // Row 1: scale = 1/sqrt(4*4)   = 0.25
        let row_scale = vec![0.1_f64, 0.25];
        super::apply_row_scale(&mut tmpl, &row_scale);

        // Matrix values: entry (row 0, col 0) = 1.0 * 0.1 = 0.1
        assert!((tmpl.values[0] - 0.1).abs() < 1e-15, "value[0] wrong");
        // Entry (row 0, col 1) = 100.0 * 0.1 = 10.0
        assert!((tmpl.values[1] - 10.0).abs() < 1e-15, "value[1] wrong");
        // Entry (row 1, col 1) = 4.0 * 0.25 = 1.0
        assert!((tmpl.values[2] - 1.0).abs() < 1e-15, "value[2] wrong");

        // Row bounds: row 0 lower = -5.0 * 0.1 = -0.5
        assert!(
            (tmpl.row_lower[0] - (-0.5)).abs() < 1e-15,
            "row_lower[0] wrong"
        );
        // Row 0 upper is INFINITY — must remain INFINITY after scaling.
        assert!(
            tmpl.row_upper[0].is_infinite() && tmpl.row_upper[0] > 0.0,
            "row_upper[0] must remain +inf"
        );
        // Row 1 lower = 7.0 * 0.25 = 1.75
        assert!(
            (tmpl.row_lower[1] - 1.75).abs() < 1e-15,
            "row_lower[1] wrong"
        );
        // Row 1 upper = 7.0 * 0.25 = 1.75 (equality constraint: lower == upper after scaling)
        assert!(
            (tmpl.row_upper[1] - 1.75).abs() < 1e-15,
            "row_upper[1] wrong"
        );

        // Column bounds and objective must be untouched.
        assert_eq!(tmpl.col_lower, vec![0.0; 2]);
        assert!(tmpl.col_upper[0].is_infinite());
        assert!(tmpl.col_upper[1].is_infinite());
        assert_eq!(tmpl.objective, vec![0.0; 2]);
    }

    /// AC E3-001-4: a row with no nonzeros receives scale factor 1.0.
    ///
    /// Matrix (3 rows × 1 col): only row 1 has a nonzero.
    /// Rows 0 and 2 are structurally empty → scale = 1.0.
    #[test]
    fn row_scale_empty_row_gets_one() {
        // col 0 has one nonzero: (row 1, val 8.0)
        let col_starts = vec![0_i32, 1];
        let row_indices = vec![1_i32];
        let values = vec![8.0_f64];
        let scale = super::compute_row_scale(3, 1, &col_starts, &row_indices, &values);
        assert_eq!(scale.len(), 3);
        // Rows 0 and 2 are empty → scale 1.0
        assert!(
            (scale[0] - 1.0).abs() < 1e-15,
            "empty row 0 scale should be 1.0"
        );
        // Row 1 has min_abs = max_abs = 8.0 → scale = 1/8
        let expected = 1.0_f64 / 8.0;
        assert!(
            (scale[1] - expected).abs() < 1e-15,
            "row 1 scale: expected {expected}, got {}",
            scale[1]
        );
        assert!(
            (scale[2] - 1.0).abs() < 1e-15,
            "empty row 2 scale should be 1.0"
        );
    }
}
