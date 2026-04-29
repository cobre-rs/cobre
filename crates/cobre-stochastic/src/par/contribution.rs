//! Contribution analysis for Periodic Autoregressive (PAR) models.
//!
//! Implements the recursive contribution composition algorithm that converts
//! dimensionless AR coefficients into physically meaningful transfer factors.
//! The contribution of each lag captures both the direct effect and the
//! indirect effects that propagate through the periodic seasonal chain.
//!
//! A negative contribution indicates that a lag's cumulative influence opposes
//! the expected persistence direction, signaling potential model instability
//! and explosive oscillation in generated values.

/// Compute the denormalized contribution for a single (season, lag) pair.
///
/// Converts the dimensionless AR coefficient at `lag` for the given `season`
/// into a physically scaled transfer factor by multiplying with the ratio of
/// the current season's standard deviation to the lagged season's standard
/// deviation.
///
/// Returns `0.0` when:
/// - The `lag` exceeds the number of coefficients available for the season.
/// - The standard deviation of the lagged season is zero.
fn denormalized_contribution(
    season: usize,
    lag: usize,
    n_seasons: usize,
    coefficients_by_season: &[&[f64]],
    std_by_season: &[f64],
) -> f64 {
    let coeffs = coefficients_by_season[season];
    if lag >= coeffs.len() {
        return 0.0;
    }
    let lagged_season = (season + n_seasons - (lag + 1) % n_seasons) % n_seasons;
    let s_current = std_by_season[season];
    let s_lagged = std_by_season[lagged_season];
    if s_lagged == 0.0 {
        0.0
    } else {
        coeffs[lag] * s_current / s_lagged
    }
}

/// Compute the recursively-composed contribution vector for a periodic
/// autoregressive model at a given season.
///
/// The contribution of lag k (1-indexed) captures the cumulative influence
/// of the value k periods ago on the current period, accounting for both
/// direct effects and indirect effects that propagate through the periodic
/// seasonal chain of the model.
///
/// # Algorithm
///
/// 1. **Denormalize** each AR coefficient into a physical transfer factor
///    `fi(p, k) = phi(p, k) * std(p) / std(p - k)`.
/// 2. **Compose recursively** via a matrix `A` where:
///    - `A[0][j] = fi(season, j)` for all j (direct effects).
///    - `A[i][j] = A[i-1][0] * fi(season - i, j) + A[i-1][j+1]` (indirect
///      effects through the periodic chain).
/// 3. **Extract contributions**: the contribution of lag `k+1` is `A[k][0]`.
///
/// # Parameters
///
/// - `season_index` -- the season (0-based) being analyzed.
/// - `n_seasons` -- total number of seasons in the periodic cycle.
/// - `order` -- the AR order for this (entity, season) pair.
/// - `coefficients_by_season` -- AR coefficients for each season, indexed by
///   season id. `coefficients_by_season[m]` is the coefficient vector for
///   season `m`. Under the PAR(p)-A extension the effective coefficient vector
///   may have length 12 (annual ψ̂/12 contribution added to each lag slot);
///   the function operates correctly on that vector without modification.
/// - `std_by_season` -- standard deviations indexed by season id.
///
/// # Returns
///
/// A `Vec<f64>` of length `order` where entry `k` is the contribution of
/// lag `k + 1`. Returns an empty vector when `order == 0`.
#[must_use]
pub fn compute_contributions(
    season_index: usize,
    n_seasons: usize,
    order: usize,
    coefficients_by_season: &[&[f64]],
    std_by_season: &[f64],
) -> Vec<f64> {
    if order == 0 {
        return Vec::new();
    }

    let mut prev_row = vec![0.0; order];
    let mut curr_row = vec![0.0; order];
    let mut contributions = Vec::with_capacity(order);

    // Base row: direct denormalized contributions for the target season.
    for (j, slot) in prev_row.iter_mut().enumerate() {
        *slot = denormalized_contribution(
            season_index,
            j,
            n_seasons,
            coefficients_by_season,
            std_by_season,
        );
    }
    contributions.push(prev_row[0]);

    // Recursive rows: compose through the periodic chain.
    for i in 1..order {
        let prev_season = (season_index + n_seasons - (i % n_seasons)) % n_seasons;
        for j in 0..(order - i) {
            curr_row[j] = prev_row[0]
                * denormalized_contribution(
                    prev_season,
                    j,
                    n_seasons,
                    coefficients_by_season,
                    std_by_season,
                )
                + prev_row[j + 1];
        }
        contributions.push(curr_row[0]);
        std::mem::swap(&mut prev_row, &mut curr_row);
    }

    contributions
}

/// Check whether any contribution in a periodic autoregressive model is negative.
///
/// A negative contribution indicates that the lag's influence, when composed
/// through the periodic seasonal chain, opposes the expected persistence
/// direction. This violates the physical structure of most seasonal time
/// series and can cause explosive oscillation in generated values.
///
/// Returns `true` if any entry in `contributions` is strictly negative.
/// Returns `false` for empty slices (an order-0 model has no contributions).
#[must_use]
pub fn check_negative_contributions(contributions: &[f64]) -> bool {
    contributions.iter().any(|&c| c < 0.0)
}

/// Check whether the first AR coefficient (`phi_1`) is negative.
///
/// A negative `phi_1` indicates that the AR model contradicts expected
/// hydrological persistence: higher inflow in the previous period implies
/// lower inflow in the current period. This is physically unrealistic for
/// most seasonal inflow series.
///
/// Returns `true` when `coefficients` is non-empty and `coefficients[0] < 0.0`.
/// Returns `false` when `coefficients` is empty (order-0 model has no `phi_1`).
#[must_use]
pub fn has_negative_phi1(coefficients: &[f64]) -> bool {
    coefficients.first().is_some_and(|&c| c < 0.0)
}

/// Find the maximum lag order with no negative contributions.
///
/// Scans `contributions` from lag 1 (index 0) forward and returns the index
/// of the first negative entry, which represents the maximum lag order
/// where all contributions are non-negative.
///
/// Returns `contributions.len()` when all contributions are non-negative
/// (the full order is valid). Returns `0` when the first contribution is
/// negative (no autoregressive dependence is stable).
///
/// Under PAR(p)-A, the input may be a length-12 effective contribution vector;
/// the return value is then in `[0, 12]`.
#[must_use]
pub fn find_max_valid_order(contributions: &[f64]) -> usize {
    contributions
        .iter()
        .position(|&c| c < 0.0)
        .unwrap_or(contributions.len())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-10;

    fn assert_close(a: f64, b: f64, label: &str) {
        let diff = (a - b).abs();
        assert!(diff < TOL, "{label}: expected {b}, got {a}, diff {diff}");
    }

    // -----------------------------------------------------------------------
    // Tests for compute_contributions
    // -----------------------------------------------------------------------

    #[test]
    fn order_zero_returns_empty_vec() {
        let coeffs: &[&[f64]] = &[];
        let stds: &[f64] = &[];
        let result = compute_contributions(0, 12, 0, coeffs, stds);
        assert!(result.is_empty());
    }

    #[test]
    fn order_one_single_season_uniform_std() {
        // phi = [0.5], std = [10.0], 1 season
        let coeffs: &[&[f64]] = &[&[0.5]];
        let stds: &[f64] = &[10.0];
        let result = compute_contributions(0, 1, 1, coeffs, stds);
        assert_eq!(result.len(), 1);
        // fi = 0.5 * 10.0 / 10.0 = 0.5
        assert_close(result[0], 0.5, "lag 1");
    }

    #[test]
    fn order_one_two_seasons_different_std() {
        // Season 0, phi_0 = [0.3], std = [30.0, 25.0]
        // Lag 1 maps to season 1.
        // fi = 0.3 * 30.0 / 25.0 = 0.36
        let coeffs: &[&[f64]] = &[&[0.3], &[]];
        let stds: &[f64] = &[30.0, 25.0];
        let result = compute_contributions(0, 2, 1, coeffs, stds);
        assert_eq!(result.len(), 1);
        assert_close(result[0], 0.36, "lag 1");
    }

    #[test]
    fn order_two_single_season_recursive_composition() {
        // phi = [0.4, 0.2], std = [10.0], 1 season
        // fi(0, 0) = 0.4, fi(0, 1) = 0.2
        // A[0] = [0.4, 0.2], contribution of lag 1 = 0.4
        // A[1][0] = A[0][0] * fi(0, 0) + A[0][1] = 0.4 * 0.4 + 0.2 = 0.36
        let coeffs: &[&[f64]] = &[&[0.4, 0.2]];
        let stds: &[f64] = &[10.0];
        let result = compute_contributions(0, 1, 2, coeffs, stds);
        assert_eq!(result.len(), 2);
        assert_close(result[0], 0.4, "lag 1");
        assert_close(result[1], 0.36, "lag 2");
    }

    #[test]
    fn order_two_two_seasons_recursive_composition() {
        // Season 0, phi_0 = [0.4, 0.2], phi_1 = [0.3], std = [30.0, 25.0]
        // fi(0, 0) = 0.4 * 30/25 = 0.48
        // fi(0, 1) = 0.2 * 30/30 = 0.2  (lag 2 wraps to season 0)
        // fi(1, 0) = 0.3 * 25/30 = 0.25
        // A[0] = [0.48, 0.2], contribution of lag 1 = 0.48
        // prev_season = (0 + 2 - 1) % 2 = 1
        // A[1][0] = A[0][0] * fi(1, 0) + A[0][1] = 0.48 * 0.25 + 0.2 = 0.32
        let coeffs: &[&[f64]] = &[&[0.4, 0.2], &[0.3]];
        let stds: &[f64] = &[30.0, 25.0];
        let result = compute_contributions(0, 2, 2, coeffs, stds);
        assert_eq!(result.len(), 2);
        assert_close(result[0], 0.48, "lag 1");
        assert_close(result[1], 0.32, "lag 2");
    }

    #[test]
    fn negative_contribution_detection() {
        // phi = [0.3, -0.8], std = [10.0], 1 season
        // fi(0, 0) = 0.3, fi(0, 1) = -0.8
        // A[0] = [0.3, -0.8], contribution of lag 1 = 0.3
        // A[1][0] = 0.3 * 0.3 + (-0.8) = -0.71
        let coeffs: &[&[f64]] = &[&[0.3, -0.8]];
        let stds: &[f64] = &[10.0];
        let result = compute_contributions(0, 1, 2, coeffs, stds);
        assert_eq!(result.len(), 2);
        assert_close(result[0], 0.3, "lag 1");
        assert_close(result[1], -0.71, "lag 2");
        assert!(check_negative_contributions(&result));
    }

    #[test]
    fn pimental_like_explosive_scenario() {
        // Construct a 12-season model where one season (e.g., August = season 7)
        // has AR(2) with a large coefficient at lag 2 scaled by std ratio.
        // Most seasons have no AR dependence (order 0 / empty coeffs).
        let mut coeffs_data: Vec<Vec<f64>> = vec![vec![]; 12];
        let mut stds = vec![100.0; 12];

        // Season 7 (August): AR(2) with large coefficient at lag 2
        // phi = [0.5, 48.9], std_7 = 5.0, std_6 = 200.0, std_5 = 200.0
        coeffs_data[7] = vec![0.5, 48.9];
        stds[7] = 5.0;
        stds[6] = 200.0;
        stds[5] = 200.0;

        let coeffs_refs: Vec<&[f64]> = coeffs_data.iter().map(Vec::as_slice).collect();
        let result = compute_contributions(7, 12, 2, &coeffs_refs, &stds);

        assert_eq!(result.len(), 2);
        // fi(7, 0) = 0.5 * 5.0 / 200.0 = 0.0125
        // fi(7, 1) = 48.9 * 5.0 / 200.0 = 1.2225
        // fi(6, 0) = 0.0 (empty coefficients for season 6)
        // A[0] = [0.0125, 1.2225]
        // A[1][0] = 0.0125 * fi(6, 0) + 1.2225 = 0.0125 * 0.0 + 1.2225 = 1.2225
        assert_close(result[0], 0.0125, "lag 1");
        assert_close(result[1], 1.2225, "lag 2");

        // Now make the coefficient negative to trigger explosive behavior
        coeffs_data[7] = vec![0.5, -48.9];
        let coeffs_refs2: Vec<&[f64]> = coeffs_data.iter().map(Vec::as_slice).collect();
        let result2 = compute_contributions(7, 12, 2, &coeffs_refs2, &stds);

        assert_close(result2[0], 0.0125, "lag 1 negative case");
        assert_close(result2[1], -1.2225, "lag 2 negative case");
        assert!(check_negative_contributions(&result2));
    }

    #[test]
    fn zero_std_for_lagged_season_produces_zero_contribution() {
        // Season 0, phi_0 = [0.5], std = [10.0, 0.0]
        // Lag 1 maps to season 1 which has std = 0.0
        // fi = 0.0 (division by zero guard)
        let coeffs: &[&[f64]] = &[&[0.5], &[]];
        let stds: &[f64] = &[10.0, 0.0];
        let result = compute_contributions(0, 2, 1, coeffs, stds);
        assert_eq!(result.len(), 1);
        assert_close(result[0], 0.0, "lag 1 with zero std");
    }

    // -----------------------------------------------------------------------
    // Tests for check_negative_contributions
    // -----------------------------------------------------------------------

    #[test]
    fn check_empty_contributions_returns_false() {
        assert!(!check_negative_contributions(&[]));
    }

    #[test]
    fn check_all_positive_returns_false() {
        assert!(!check_negative_contributions(&[0.3, 0.2, 0.1]));
    }

    #[test]
    fn check_one_negative_returns_true() {
        assert!(check_negative_contributions(&[0.3, -0.1, 0.2]));
    }

    #[test]
    fn check_zero_is_not_negative() {
        assert!(!check_negative_contributions(&[0.3, 0.0, 0.1]));
    }

    // -----------------------------------------------------------------------
    // Tests for find_max_valid_order
    // -----------------------------------------------------------------------

    #[test]
    fn find_max_order_empty_returns_zero() {
        assert_eq!(find_max_valid_order(&[]), 0);
    }

    #[test]
    fn find_max_order_all_positive_returns_full_length() {
        assert_eq!(find_max_valid_order(&[0.3, 0.2, 0.1]), 3);
    }

    #[test]
    fn find_max_order_negative_at_index_two() {
        assert_eq!(find_max_valid_order(&[0.3, 0.2, -0.1]), 2);
    }

    #[test]
    fn find_max_order_negative_at_index_zero() {
        assert_eq!(find_max_valid_order(&[-0.1, 0.2, 0.3]), 0);
    }

    // -----------------------------------------------------------------------
    // Additional edge-case tests
    // -----------------------------------------------------------------------

    #[test]
    fn order_three_single_season_full_recursion() {
        // phi = [0.5, 0.2, 0.1], std = [10.0], 1 season
        // fi(0, 0) = 0.5, fi(0, 1) = 0.2, fi(0, 2) = 0.1
        // A[0] = [0.5, 0.2, 0.1], contribution of lag 1 = 0.5
        //
        // A[1][0] = 0.5 * fi(0, 0) + 0.2 = 0.5 * 0.5 + 0.2 = 0.45
        // A[1][1] = 0.5 * fi(0, 1) + 0.1 = 0.5 * 0.2 + 0.1 = 0.2
        //
        // A[2][0] = 0.45 * fi(0, 0) + 0.2 = 0.45 * 0.5 + 0.2 = 0.425
        let coeffs: &[&[f64]] = &[&[0.5, 0.2, 0.1]];
        let stds: &[f64] = &[10.0];
        let result = compute_contributions(0, 1, 3, coeffs, stds);
        assert_eq!(result.len(), 3);
        assert_close(result[0], 0.5, "lag 1");
        assert_close(result[1], 0.45, "lag 2");
        assert_close(result[2], 0.425, "lag 3");
    }

    // -----------------------------------------------------------------------
    // Tests for has_negative_phi1
    // -----------------------------------------------------------------------

    #[test]
    fn phi1_negative_returns_true() {
        assert!(has_negative_phi1(&[-0.5, 0.3]));
    }

    #[test]
    fn phi1_positive_returns_false() {
        assert!(!has_negative_phi1(&[0.5, -0.3]));
    }

    #[test]
    fn phi1_zero_returns_false() {
        assert!(!has_negative_phi1(&[0.0, 0.3]));
    }

    #[test]
    fn phi1_empty_returns_false() {
        assert!(!has_negative_phi1(&[]));
    }

    #[test]
    fn phi1_near_zero_negative_returns_true() {
        assert!(has_negative_phi1(&[-0.001]));
    }

    // -----------------------------------------------------------------------
    // Tests for the length-12 effective polynomial produced by the annual extension
    // -----------------------------------------------------------------------

    #[test]
    fn compute_contributions_length_12_par_a_polynomial() {
        // Fixture: single season (n_seasons = 1), order = 12.
        // coefficients[0] = [0.5, 0.05, 0.05, ..., 0.05] (12 entries).
        // std = [10.0] — uniform, so all std ratios are 1 and
        // fi(0, j) == coefficients[0][j].
        //
        // The recursion for a single season with uniform std reduces to:
        //   A[0] = phi           (base row)
        //   A[i][j] = A[i-1][0] * phi[j] + A[i-1][j+1]
        //   contribution[i] = A[i][0]
        //
        // Hand-computed reference values (phi = [0.5, 0.05 * 11]):
        //   lag 1  = phi[0]                              = 0.5
        //   lag 2  = 0.5 * 0.5 + 0.05                   = 0.300_000_000_000_000
        //   lag 3  = 0.30 * 0.5 + (0.5 * 0.05 + 0.05)  = 0.225_000_000_000_000
        //   lag 4  = 0.225 * 0.5 + (0.30 * 0.05 + 0.05)= 0.202_500_000_000_000
        //   lag 5  = 0.2025 * 0.5 + ...                 = 0.202_500_000_000_000
        //   (values determined by running the recursion offline)
        //   lag 6  = 0.212_625_000_000_000
        //   lag 7  = 0.227_812_500_000_000
        //   lag 8  = 0.246_037_500_000_000
        //   lag 9  = 0.266_540_625_000_000
        //   lag 10 = 0.289_094_062_500_000
        //   lag 11 = 0.313_697_812_500_000
        //   lag 12 = 0.340_454_390_625_000
        let phi: Vec<f64> = std::iter::once(0.5_f64)
            .chain(std::iter::repeat_n(0.05_f64, 11))
            .collect();
        let coeffs: &[&[f64]] = &[phi.as_slice()];
        let stds: &[f64] = &[10.0];
        let result = compute_contributions(0, 1, 12, coeffs, stds);
        assert_eq!(result.len(), 12, "contribution vector must have length 12");

        let expected = [
            0.5_f64,
            0.300_000_000_000_000,
            0.225_000_000_000_000,
            0.202_500_000_000_000,
            0.202_500_000_000_000,
            0.212_625_000_000_000,
            0.227_812_500_000_000,
            0.246_037_500_000_000,
            0.266_540_625_000_000,
            0.289_094_062_500_000,
            0.313_697_812_500_000,
            0.340_454_390_625_000,
        ];
        for (k, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
            assert_close(got, exp, &format!("lag {}", k + 1));
        }
    }

    #[test]
    fn find_max_valid_order_all_positive_length_12() {
        // A length-12 contribution vector with all non-negative entries must
        // return 12 (the full order is valid).
        let contributions = [
            0.3_f64, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
        ];
        assert_eq!(find_max_valid_order(&contributions), 12);
    }

    #[test]
    fn find_max_valid_order_first_negative_at_seven_in_length_12() {
        // A length-12 contribution vector with the first negative entry at
        // index 7 (lag 8) must return 7.
        let contributions = [
            0.3_f64, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, -0.01, 0.05, 0.05, 0.05, 0.05,
        ];
        assert_eq!(find_max_valid_order(&contributions), 7);
    }
}
