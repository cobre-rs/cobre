//! Offline LP prescaling: geometric-mean column/row scale factors applied to
//! stage templates for numerical conditioning (`D_r * A * D_c` form), plus the
//! noise pre-scaling helper. Invoked from `setup/template_postprocess::postprocess_templates`.

use cobre_core::Stage;
use cobre_solver::StageTemplate;

use crate::indexer::StateSpace;

/// Per-column geometric-mean scaling factors from a CSC matrix:
/// `1 / sqrt(max|A_ij| * min|A_ij|)` over nonzeros, `1.0` for an empty column.
/// Length `num_cols`.
#[must_use]
#[expect(
    clippy::cast_sign_loss,
    reason = "CSC column starts are non-negative by construction"
)]
pub(crate) fn compute_col_scale(num_cols: usize, col_starts: &[i32], values: &[f64]) -> Vec<f64> {
    let mut scale = vec![1.0_f64; num_cols];
    for j in 0..num_cols {
        let start = col_starts[j] as usize;
        let end = col_starts[j + 1] as usize;
        if start == end {
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
            scale[j] = 1.0 / (max_abs * min_abs).sqrt();
        }
    }
    scale
}

/// Apply column scaling in-place. After this call, per column `j`: `values` and
/// `objective` are MULTIPLIED by `col_scale[j]`, while `col_lower`/`col_upper` are
/// DIVIDED by it (the scaled variable is `x̃ = x / d_j`).
pub(crate) fn apply_col_scale(template: &mut StageTemplate, col_scale: &[f64]) {
    let num_cols = template.num_cols;
    debug_assert_eq!(col_scale.len(), num_cols);

    #[expect(
        clippy::needless_range_loop,
        clippy::cast_sign_loss,
        reason = "the loop reads col_starts[j + 1], and CSC offsets are non-negative by construction"
    )]
    for j in 0..num_cols {
        let start = template.col_starts[j] as usize;
        let end = template.col_starts[j + 1] as usize;
        let d = col_scale[j];
        for v in &mut template.values[start..end] {
            *v *= d;
        }
    }

    for (obj, &d) in template.objective.iter_mut().zip(col_scale) {
        *obj *= d;
    }

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

/// Force `col_scale = 1.0` on the merged commitment-hold region's outgoing and
/// incoming ranges (`commit_out`, `commit_in`), overriding whatever
/// `compute_col_scale` derived there. Every hold slot round-trips through a
/// pin-then-read — the in-study ring's deposit/fishing coupling, the
/// post-horizon lane's stage-to-stage carry — that is exact only at
/// `col_scale = 1.0`; any other factor can drift a commitment sitting exactly
/// at its delivery-stage generation cap a sub-ULP outside the bound. No-op
/// when `commit_out` is empty (no anticipated thermals, no post-horizon
/// commitments).
///
/// # Panics (debug builds only)
///
/// Panics if `col_scale` does not cover every state column including `theta` —
/// the `col_scale.len() > theta_col` contract every render/patch call site
/// relies on.
pub(crate) fn apply_commitment_hold_col_scale_unscale(
    col_scale: &mut [f64],
    state_layout: &StateSpace,
) {
    debug_assert!(
        col_scale.len() > state_layout.theta,
        "col_scale must cover every state column including theta ({}); got len {}",
        state_layout.theta,
        col_scale.len()
    );
    for c in state_layout
        .commit_out
        .clone()
        .chain(state_layout.commit_in.clone())
    {
        col_scale[c] = 1.0;
    }
}

/// Per-row geometric-mean scaling factors from a CSC matrix:
/// `1 / sqrt(max|A_ij| * min|A_ij|)` over a row's nonzeros, `1.0` for an empty row.
/// Length `num_rows`.
///
/// MUST be called on the ALREADY column-scaled matrix to obtain the standard
/// `D_r * A * D_c` form (column scaling before row scaling).
#[must_use]
#[expect(
    clippy::cast_sign_loss,
    reason = "CSC column starts and row indices are non-negative by construction"
)]
pub(crate) fn compute_row_scale(
    num_rows: usize,
    num_cols: usize,
    col_starts: &[i32],
    row_indices: &[i32],
    values: &[f64],
) -> Vec<f64> {
    let mut row_max = vec![0.0_f64; num_rows];
    let mut row_min = vec![f64::INFINITY; num_rows];

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
    }
    scale
}

/// Apply row scaling in-place: per row `i`, `values` and `row_lower`/`row_upper`
/// are MULTIPLIED by `row_scale[i]`. Objective and column bounds are NOT touched —
/// those are column-domain quantities handled by [`apply_col_scale`].
pub(crate) fn apply_row_scale(template: &mut StageTemplate, row_scale: &[f64]) {
    let num_rows = template.num_rows;
    debug_assert_eq!(row_scale.len(), num_rows);

    let num_cols = template.num_cols;
    #[expect(
        clippy::cast_sign_loss,
        reason = "CSC row indices are non-negative by construction"
    )]
    for j in 0..num_cols {
        let start = template.col_starts[j] as usize;
        let end = template.col_starts[j + 1] as usize;
        for k in start..end {
            let row = template.row_indices[k] as usize;
            template.values[k] *= row_scale[row];
        }
    }

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

/// Pre-compute the per-stage block-hours table.
pub(super) fn compute_stage_hours(study_stages: &[&Stage]) -> Vec<Vec<f64>> {
    study_stages
        .iter()
        .map(|stage| stage.blocks.iter().map(|b| b.duration_hours).collect())
        .collect()
}

#[cfg(test)]
#[expect(
    clippy::doc_markdown,
    reason = "test docs name LP symbols that are not code identifiers"
)]
mod tests {
    use cobre_solver::StageTemplate;

    // =========================================================================
    // Row scaling tests
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
            col_scale: Vec::new(),
            row_scale: Vec::new(),
            n_state: 0,
        }
    }

    /// A matrix where every row has min_abs == max_abs gives scale 1.0.
    ///
    /// Matrix (2 rows × 2 cols, column-major), all nonzeros |value| = 1.0:
    ///
    /// ```text
    /// col 0: row 0 → 1.0, row 1 → 1.0
    /// col 1: row 0 → 1.0, row 1 → 1.0
    /// ```
    ///
    /// For each row: min_abs = max_abs = 1.0 → scale = 1/sqrt(1*1) = 1.0.
    #[test]
    fn row_scale_identity_for_uniform_matrix() {
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

    /// Geometric-mean scale matches expected value for known matrix.
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

    /// `apply_row_scale` multiplies matrix values and row bounds.
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

    /// A row with no nonzeros receives scale factor 1.0.
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

    use crate::indexer::StateSpace;
    use crate::lead_time::AnticipatedResolution;
    use crate::test_support::constant_lead_resolution;

    // =========================================================================
    // Commitment-hold col_scale=1.0 override
    // =========================================================================

    /// `N=2` hydros, `L=0`, `A=1` anticipated plant, `K=2` slots: seeds every
    /// `col_scale` entry with a distinct non-1.0 value, then asserts the
    /// override forces exactly the `commit_out`/`commit_in` indices to `1.0`
    /// while leaving every other index byte-identical to its seed.
    #[test]
    fn apply_commitment_hold_col_scale_unscale_forces_hold_to_one() {
        let resolution = constant_lead_resolution(&[2], 4);
        let state_layout = StateSpace::new(2, 0, vec![], vec![2], resolution, &[0, 0]);

        assert_eq!(state_layout.commit_out, 2..4);
        assert_eq!(state_layout.commit_in, 8..10);
        assert_eq!(state_layout.theta, 10);

        let mut col_scale: Vec<f64> = (2..13).map(f64::from).collect();
        assert_eq!(col_scale.len(), state_layout.theta + 1);
        let before = col_scale.clone();

        super::apply_commitment_hold_col_scale_unscale(&mut col_scale, &state_layout);

        for c in state_layout.commit_out.clone() {
            assert_eq!(col_scale[c], 1.0, "commit_out index {c}");
        }
        for c in state_layout.commit_in.clone() {
            assert_eq!(col_scale[c], 1.0, "commit_in index {c}");
        }
        for (i, (&got, &want)) in col_scale.iter().zip(before.iter()).enumerate() {
            if state_layout.commit_out.contains(&i) || state_layout.commit_in.contains(&i) {
                continue;
            }
            assert_eq!(got, want, "non-commitment-hold index {i} must be untouched");
        }
    }

    /// `n_anticipated * k_max == 0`: `commit_out`/`commit_in` both collapse to
    /// `0..0`, so the override loop touches no column — `col_scale` is left
    /// exactly as the generic computation produced it.
    #[test]
    fn apply_commitment_hold_col_scale_unscale_is_noop_when_empty() {
        let state_layout = StateSpace::new(
            2,
            0,
            vec![],
            vec![],
            AnticipatedResolution::default(),
            &[0, 0],
        );
        let mut col_scale = vec![1.0_f64; state_layout.theta + 1];
        col_scale[0] = 3.0;

        let before = col_scale.clone();
        super::apply_commitment_hold_col_scale_unscale(&mut col_scale, &state_layout);

        assert_eq!(
            col_scale, before,
            "an empty CommitmentHold region must leave col_scale untouched"
        );
    }
}
