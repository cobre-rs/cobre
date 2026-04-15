//! Basis-aware warm-start padding for new cut rows (Strategy S3).
//!
//! When the LP for a stage gains new rows (cuts) between iterations, the
//! cached basis from the previous solve has fewer row entries than the
//! current LP. The naive approach fills all new entries with
//! `HIGHS_BASIS_STATUS_BASIC`, forcing `HiGHS` to discover the correct status
//! through simplex pivots.
//!
//! Strategy S3 replaces this blanket fill with *informed padding*: each new
//! cut is evaluated at the warm-start state `x̂`. If the cut is tight or
//! violated at `x̂` (i.e. `θ̂ - cut_value ≤ tolerance`), the cut is an active
//! constraint and gets `NONBASIC_LOWER`. If it is slack, it is an inactive
//! constraint and gets `BASIC`. This reduces the number of discovery pivots
//! that `HiGHS` needs after each cut addition.
//!
//! ## Usage
//!
//! ```rust
//! use cobre_sddp::basis_padding::pad_basis_for_cuts;
//! use cobre_sddp::cut::pool::CutPool;
//! use cobre_solver::Basis;
//!
//! let mut pool = CutPool::new(10, 1, 1, 0);
//! pool.add_cut(0, 0, 10.0, &[1.0]);
//!
//! let mut basis = Basis::new(3, 2);
//! let (tight, slack) = pad_basis_for_cuts(&mut basis, &pool, &[5.0], 25.0, 2, 1e-7);
//! assert_eq!(tight, 0);
//! assert_eq!(slack, 1);
//! assert_eq!(basis.row_status.len(), 3);
//! ```

pub use cobre_solver::ffi::{HIGHS_BASIS_STATUS_BASIC, HIGHS_BASIS_STATUS_LOWER};

use cobre_solver::Basis;

use crate::cut::pool::CutPool;

/// Extend `basis.row_status` with informed status codes for new cut rows.
///
/// For each active cut in `pool` whose row is not yet covered by the existing
/// `basis.row_status`, evaluates the cut at the warm-start state and assigns:
///
/// - `NONBASIC_LOWER` if `theta_value - cut_value <= tolerance` (tight or violated),
/// - `BASIC` if `theta_value - cut_value > tolerance` (slack).
///
/// # Parameters
///
/// - `basis`: the cached basis to extend. Only `row_status` is modified;
///   `col_status` is untouched.
/// - `pool`: the active cut pool for the current stage.
/// - `state`: warm-start state vector, length `pool.state_dimension`. Provides
///   the state-variable values `x̂` at which each cut is evaluated.
/// - `theta_value`: the future-cost variable value from the warm-start solution,
///   already unscaled.
/// - `base_row_count`: number of structural (template) rows. The basis is
///   expected to already have entries for these rows plus any previously
///   appended cut rows from earlier iterations.
/// - `tolerance`: slack threshold. Cuts with `theta_value - cut_value <= tolerance`
///   are assigned `NONBASIC_LOWER`.
///
/// # Returns
///
/// `(tight_count, slack_count)` — the number of new rows assigned
/// `NONBASIC_LOWER` and `BASIC` respectively. Both are 0 when no new rows
/// are added.
///
/// # Panics
///
/// Panics (in debug builds only) if `state.len() != pool.state_dimension` or
/// if `basis.row_status.len() < base_row_count`.
pub fn pad_basis_for_cuts(
    basis: &mut Basis,
    pool: &CutPool,
    state: &[f64],
    theta_value: f64,
    base_row_count: usize,
    tolerance: f64,
) -> (usize, usize) {
    debug_assert!(
        state.len() == pool.state_dimension,
        "state length {} != pool.state_dimension {}",
        state.len(),
        pool.state_dimension,
    );
    debug_assert!(
        basis.row_status.len() >= base_row_count,
        "basis.row_status.len() {} < base_row_count {}",
        basis.row_status.len(),
        base_row_count,
    );

    let target_len = base_row_count + pool.active_count();

    if basis.row_status.len() >= target_len {
        return (0, 0);
    }

    let old_len = basis.row_status.len();

    // Resize with conservative default BASIC. Entries for previously seen cuts
    // (between old_len and target_len) will be overwritten below; the resize
    // ensures the Vec has the correct final length before the loop writes into it.
    basis
        .row_status
        .resize(target_len, HIGHS_BASIS_STATUS_BASIC);

    // The offset of the first cut row from base_row_count in the basis vector.
    // Previously appended cut rows occupy [base_row_count, old_len).
    // New cut rows occupy [old_len, target_len).
    //
    // We iterate all active cuts in slot order (matching LP row order) and
    // assign statuses only for those that fall in the new region.
    let already_padded_cuts = old_len.saturating_sub(base_row_count);

    let mut tight_count: usize = 0;
    let mut slack_count: usize = 0;
    let mut cut_index: usize = 0;

    for (_slot, intercept, coefficients) in pool.active_cuts() {
        if cut_index < already_padded_cuts {
            // This cut row was already in the basis from a previous padding call.
            cut_index += 1;
            continue;
        }

        let cut_value: f64 = intercept
            + coefficients
                .iter()
                .zip(state)
                .map(|(c, x)| c * x)
                .sum::<f64>();
        let slack = theta_value - cut_value;

        let row_idx = base_row_count + cut_index;
        if slack <= tolerance {
            basis.row_status[row_idx] = HIGHS_BASIS_STATUS_LOWER;
            tight_count += 1;
        } else {
            // Already set to BASIC by the resize fill — increment counter only.
            slack_count += 1;
        }

        cut_index += 1;
    }

    (tight_count, slack_count)
}

#[cfg(test)]
mod tests {
    use cobre_solver::Basis;

    use super::{pad_basis_for_cuts, HIGHS_BASIS_STATUS_BASIC, HIGHS_BASIS_STATUS_LOWER};
    use crate::cut::pool::CutPool;

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn make_pool_with_cuts(cuts: &[(f64, Vec<f64>)], state_dim: usize) -> CutPool {
        let mut pool = CutPool::new(cuts.len().max(1) * 10, state_dim, 1, 0);
        for (i, (intercept, coeffs)) in cuts.iter().enumerate() {
            pool.add_cut(i as u64, 0, *intercept, coeffs);
        }
        pool
    }

    // -----------------------------------------------------------------------
    // AC 1: mixed tight/slack cuts get correct statuses and counts
    // -----------------------------------------------------------------------

    #[test]
    fn test_tight_and_slack_cuts_get_correct_status() {
        // Pool: 3 cuts
        //   cut 0: intercept=10, coeff=[1.0] → value = 10 + 1*5 = 15, slack = 25-15 = 10 → BASIC
        //   cut 1: intercept=20, coeff=[2.0] → value = 20 + 2*5 = 30, slack = 25-30 = -5 → NONBASIC_LOWER
        //   cut 2: intercept=30, coeff=[3.0] → value = 30 + 3*5 = 45, slack = 25-45 = -20 → NONBASIC_LOWER
        let pool = make_pool_with_cuts(
            &[(10.0, vec![1.0]), (20.0, vec![2.0]), (30.0, vec![3.0])],
            1,
        );

        let mut basis = Basis::new(5, 2); // 5 cols, 2 structural rows
        let (tight, slack) = pad_basis_for_cuts(&mut basis, &pool, &[5.0], 25.0, 2, 1e-7);

        assert_eq!(basis.row_status.len(), 5, "basis must grow to base+active");
        assert_eq!(
            basis.row_status[2], HIGHS_BASIS_STATUS_BASIC,
            "cut 0 slack=10 → BASIC"
        );
        assert_eq!(
            basis.row_status[3], HIGHS_BASIS_STATUS_LOWER,
            "cut 1 slack=-5 → NONBASIC_LOWER"
        );
        assert_eq!(
            basis.row_status[4], HIGHS_BASIS_STATUS_LOWER,
            "cut 2 slack=-20 → NONBASIC_LOWER"
        );
        assert_eq!(tight, 2, "two tight/violated cuts");
        assert_eq!(slack, 1, "one slack cut");
    }

    // -----------------------------------------------------------------------
    // AC 2: exactly-tight cut (slack == 0.0) gets NONBASIC_LOWER
    // -----------------------------------------------------------------------

    #[test]
    fn test_exactly_tight_cut_is_nonbasic_lower() {
        // cut: intercept=5, coeff=[1.0, 2.0], state=[1.0, 1.0]
        // cut_value = 5 + 1 + 2 = 8, slack = 8 - 8 = 0.0 ≤ 1e-7 → NONBASIC_LOWER
        let pool = make_pool_with_cuts(&[(5.0, vec![1.0, 2.0])], 2);

        let mut basis = Basis::new(3, 0);
        let (tight, slack) = pad_basis_for_cuts(&mut basis, &pool, &[1.0, 1.0], 8.0, 0, 1e-7);

        assert_eq!(basis.row_status.len(), 1);
        assert_eq!(basis.row_status[0], HIGHS_BASIS_STATUS_LOWER);
        assert_eq!(tight, 1);
        assert_eq!(slack, 0);
    }

    // -----------------------------------------------------------------------
    // AC 3: empty pool is a no-op
    // -----------------------------------------------------------------------

    #[test]
    fn test_empty_pool_is_noop() {
        let pool = CutPool::new(10, 2, 1, 0); // no cuts added
        let mut basis = Basis::new(3, 2);
        basis.row_status[0] = HIGHS_BASIS_STATUS_LOWER;
        basis.row_status[1] = HIGHS_BASIS_STATUS_BASIC;

        let (tight, slack) = pad_basis_for_cuts(&mut basis, &pool, &[1.0, 1.0], 5.0, 2, 1e-7);

        assert_eq!(basis.row_status.len(), 2, "row_status unchanged");
        assert_eq!(basis.row_status[0], HIGHS_BASIS_STATUS_LOWER);
        assert_eq!(basis.row_status[1], HIGHS_BASIS_STATUS_BASIC);
        assert_eq!(tight, 0);
        assert_eq!(slack, 0);
    }

    // -----------------------------------------------------------------------
    // AC 4: already fully padded basis is a no-op
    // -----------------------------------------------------------------------

    #[test]
    fn test_already_padded_basis_is_noop() {
        let pool = make_pool_with_cuts(&[(10.0, vec![1.0]), (20.0, vec![2.0])], 1);

        // Basis already has base_row_count + active_count = 2 + 2 = 4 rows.
        let mut basis = Basis::new(3, 4);
        basis.row_status[2] = HIGHS_BASIS_STATUS_LOWER;
        basis.row_status[3] = HIGHS_BASIS_STATUS_LOWER;

        let (tight, slack) = pad_basis_for_cuts(&mut basis, &pool, &[1.0], 5.0, 2, 1e-7);

        assert_eq!(basis.row_status.len(), 4, "row_status unchanged");
        assert_eq!(
            basis.row_status[2], HIGHS_BASIS_STATUS_LOWER,
            "prior status preserved"
        );
        assert_eq!(
            basis.row_status[3], HIGHS_BASIS_STATUS_LOWER,
            "prior status preserved"
        );
        assert_eq!(tight, 0);
        assert_eq!(slack, 0);
    }

    // -----------------------------------------------------------------------
    // AC 5 (test 5): all slack cuts → all BASIC, counts are (0, n)
    // -----------------------------------------------------------------------

    #[test]
    fn test_all_slack_cuts_get_basic() {
        // cut 0: value = 1 + 1*1 = 2, slack = 1000 - 2 = 998 → BASIC
        // cut 1: value = 2 + 2*1 = 4, slack = 1000 - 4 = 996 → BASIC
        let pool = make_pool_with_cuts(&[(1.0, vec![1.0]), (2.0, vec![2.0])], 1);

        let mut basis = Basis::new(3, 2);
        let (tight, slack) = pad_basis_for_cuts(&mut basis, &pool, &[1.0], 1000.0, 2, 1e-7);

        assert_eq!(basis.row_status.len(), 4);
        assert_eq!(basis.row_status[2], HIGHS_BASIS_STATUS_BASIC);
        assert_eq!(basis.row_status[3], HIGHS_BASIS_STATUS_BASIC);
        assert_eq!(tight, 0);
        assert_eq!(slack, 2);
    }

    // -----------------------------------------------------------------------
    // AC 6 (test 6): violated cut (value exceeds theta) → NONBASIC_LOWER
    // -----------------------------------------------------------------------

    #[test]
    fn test_negative_slack_is_tight() {
        // cut: intercept=100, coeff=[10.0], state=[5.0]
        // cut_value = 100 + 50 = 150, theta_value = 1.0
        // slack = 1.0 - 150.0 = -149.0 ≤ 1e-7 → NONBASIC_LOWER
        let pool = make_pool_with_cuts(&[(100.0, vec![10.0])], 1);

        let mut basis = Basis::new(3, 1);
        let (tight, slack) = pad_basis_for_cuts(&mut basis, &pool, &[5.0], 1.0, 1, 1e-7);

        assert_eq!(basis.row_status.len(), 2);
        assert_eq!(basis.row_status[1], HIGHS_BASIS_STATUS_LOWER);
        assert_eq!(tight, 1);
        assert_eq!(slack, 0);
    }
}
