//! Conditional FACP for the PAR-A model via the partitioned-covariance approach.

use super::yw_matrices::{
    cross_correlation_a_z_neg1, cross_correlation_z_a, periodic_autocorrelation,
    solve_linear_system,
};

/// The three sub-matrices of the conditional FACP formula
/// `Σ̄ = Σ_11 − Σ_12 · Σ_22⁻¹ · Σ_21` for one `(season, k)`. All row-major.
// Rationale: the sigma_11/sigma_12/sigma_22 names mirror the partitioned-covariance
// notation (Σ_11, Σ_12, Σ_22); dropping the common prefix to silence the lint would
// break the correspondence with the algorithm documentation.
#[allow(clippy::struct_field_names)]
pub(crate) struct PartitionedCov {
    /// 2×2 auto-covariance of `(Z_t, Z_{t−k})`.
    pub(super) sigma_11: [f64; 4],
    /// 2×k cross-covariance between `(Z_t, Z_{t−k})` and the conditioning set
    /// `(Z_{t−1}, …, Z_{t−k+1}, A_{t−1})`.
    pub(super) sigma_12: Vec<f64>,
    /// k×k auto-covariance of the conditioning set.
    pub(super) sigma_22: Vec<f64>,
}

/// Assemble the three sub-matrices of the partitioned covariance for the
/// conditional FACP at lag `k` from `season`.
///
/// # Matrix layout
///
/// The conditioning set at lag `k` is:
/// `(Z_{t−1}, Z_{t−2}, …, Z_{t−k+1}, A_{t−1})` — that is, `k−1` lagged
/// `Z` values followed by the annual component at season `m−1`.
///
/// **`sigma_11`** (2×2): auto-covariance of `(Z_t, Z_{t−k})`.
/// - `[0,0] = 1` (unit variance of standardised `Z_t`)
/// - `[0,1] = [1,0] = ρ^{season}(k)`
/// - `[1,1] = 1`
///
/// **`sigma_22`** (k×k): auto-covariance of the conditioning set.
/// - Rows/cols `i,j < k−1`: `ρ^{season−1}(|i−j|)` — periodic autocorrelation
///   of the `Z` block at season `m−1`. Diagonal entries are `1.0`.
/// - Row/col `k−1` (the `A_{t−1}` entry):
///   - Off-diagonal: `cross_correlation_z_a(season−1, i, …)` for `i < k−1`.
///   - Diagonal `[k−1, k−1] = 1.0` (unit variance of standardised `A_{t−1}`).
///
/// **`sigma_12`** (2×k): cross-covariance between `(Z_t, Z_{t−k})` and the
/// conditioning set.
/// - Row 0 (from `Z_t`), column `j < k−1`: `ρ^{season}(j+1)`.
/// - Row 0, column `k−1`: `cross_correlation_a_z_neg1(season−1, …)`.
/// - Row 1 (from `Z_{t−k}`), column `j < k−1`: `ρ^{season−k}(k−1−j)`.
/// - Row 1, column `k−1`: `cross_correlation_z_a(season−1, k−1, …)`.
pub(crate) fn assemble_partitioned_covariance(
    season: usize,
    k: usize,
    n_seasons: usize,
    obs_z: &[&[f64]],
    stats_z: &[(f64, f64)],
    z_year_starts: &[i32],
    obs_a: &[&[f64]],
    stats_a: &[(f64, f64)],
    a_year_starts: &[i32],
) -> PartitionedCov {
    let prev_season = (season + n_seasons - 1) % n_seasons;

    let rho_k = periodic_autocorrelation(season, k, n_seasons, obs_z, stats_z, z_year_starts);
    let sigma_11 = [1.0, rho_k, rho_k, 1.0];

    let mut sigma_22 = vec![0.0_f64; k * k];

    for i in 0..k.saturating_sub(1) {
        sigma_22[i * k + i] = 1.0;
        let ref_month = (prev_season + n_seasons - i % n_seasons) % n_seasons;
        for j in (i + 1)..k.saturating_sub(1) {
            let lag = j - i;
            let rho =
                periodic_autocorrelation(ref_month, lag, n_seasons, obs_z, stats_z, z_year_starts);
            sigma_22[i * k + j] = rho;
            sigma_22[j * k + i] = rho;
        }
    }

    // sigma_22[i, k−1] = Corr(Z_{t−1−i}, A_{t−1}): Z_{t−1−i} is `i` steps older
    // than A_{t−1}, so lag = i — not k−2−i, which only coincides at k=2.
    for i in 0..k.saturating_sub(1) {
        let lag = i;
        let rho = cross_correlation_z_a(
            prev_season,
            lag,
            n_seasons,
            obs_z,
            stats_z,
            z_year_starts,
            obs_a,
            stats_a,
            a_year_starts,
        );
        sigma_22[i * k + (k - 1)] = rho;
        sigma_22[(k - 1) * k + i] = rho;
    }

    sigma_22[(k - 1) * k + (k - 1)] = 1.0;

    let mut sigma_12 = vec![0.0_f64; 2 * k];

    for (j, entry) in sigma_12[..k.saturating_sub(1)].iter_mut().enumerate() {
        let rho = periodic_autocorrelation(season, j + 1, n_seasons, obs_z, stats_z, z_year_starts);
        *entry = rho;
    }
    sigma_12[k - 1] = cross_correlation_a_z_neg1(
        prev_season,
        n_seasons,
        obs_z,
        stats_z,
        z_year_starts,
        obs_a,
        stats_a,
        a_year_starts,
    );

    // Row 1, position j = Corr(Z_{t−k}, Z_{t−1−j}): Z_{t−1−j} is the newer of the
    // two, so ρ_periodic is anchored at its season with lag = (k−1) − j.
    // `(1 + j) % n_seasons` prevents underflow when a lag exceeds the season
    // cycle (k ≥ n_seasons + 2), matching the conditioning-set walk above.
    for (j, entry) in sigma_12[k..k + k.saturating_sub(1)].iter_mut().enumerate() {
        let ref_season = (season + n_seasons - (1 + j) % n_seasons) % n_seasons;
        let lag = k.saturating_sub(1).saturating_sub(j);
        let rho =
            periodic_autocorrelation(ref_season, lag, n_seasons, obs_z, stats_z, z_year_starts);
        *entry = rho;
    }
    sigma_12[k + (k - 1)] = cross_correlation_z_a(
        prev_season,
        k - 1,
        n_seasons,
        obs_z,
        stats_z,
        z_year_starts,
        obs_a,
        stats_a,
        a_year_starts,
    );

    PartitionedCov {
        sigma_11,
        sigma_12,
        sigma_22,
    }
}

/// Compute the conditional FACP for the PAR-A model up to `max_order`.
///
/// For each lag `k`, the conditioning set is `(Z_{t−1}, …, Z_{t−k+1}, A_{t−1})`
/// and the conditional FACP is `Σ̄[0,1] / √(Σ̄[0,0] · Σ̄[1,1])` clamped to
/// `[−1, 1]`, where `Σ̄ = Σ_11 − Σ_12 · Σ_22⁻¹ · Σ_21` and `Σ_21 = Σ_12ᵀ`.
///
/// # Parameters
///
/// - `season` — 0-based target season `m`.
/// - `max_order` — maximum lag to evaluate; `0` returns `Vec::new()`.
/// - `n_seasons` — total number of seasons in the periodic cycle.
/// - `observations_by_season` — standardised inflow series `Z`, `[s][y]` grouped
///   by season.
/// - `stats_by_season` — `(mean, std)` for each `Z` season.
/// - `z_year_starts` — first PDF year of each `Z` bucket, used to align `A` and
///   `Z` by absolute PDF year, not bucket index (required for monthly data where
///   `A` buckets start a year after `Z` for most seasons).
/// - `annual_observations_by_season` — annual component `A`, grouped by season.
/// - `annual_stats_by_season` — `(mean, std)` for each `A` season.
/// - `a_year_starts` — first PDF year of each `A` bucket, indexed by season.
///
/// # Returns
///
/// Entry `i` is the conditional FACP at lag `i+1`. Shorter than `max_order` when
/// `Σ_22` is singular at some lag `k` (loop breaks; lags `≥ k` omitted); an entry
/// is `0.0` when `Σ̄[0,0] · Σ̄[1,1] ≤ 0` (numerical degeneracy).
#[must_use]
pub fn conditional_facp_partitioned(
    season: usize,
    max_order: usize,
    n_seasons: usize,
    observations_by_season: &[&[f64]],
    stats_by_season: &[(f64, f64)],
    z_year_starts: &[i32],
    annual_observations_by_season: &[&[f64]],
    annual_stats_by_season: &[(f64, f64)],
    a_year_starts: &[i32],
) -> Vec<f64> {
    if max_order == 0 {
        return Vec::new();
    }

    let mut facp_values = Vec::with_capacity(max_order);

    // Reused working copies of Σ_22 and the RHS column so each solve avoids a
    // fresh allocation.
    let mut matrix_buf: Vec<f64> = Vec::new();
    let mut rhs_col: Vec<f64> = Vec::new();

    for k in 1..=max_order {
        let cov = assemble_partitioned_covariance(
            season,
            k,
            n_seasons,
            observations_by_season,
            stats_by_season,
            z_year_starts,
            annual_observations_by_season,
            annual_stats_by_season,
            a_year_starts,
        );

        // Solve Σ_22 · X = Σ_21 = Σ_12ᵀ column-by-column, so column c of the RHS
        // is row c of Σ_12. solve_linear_system mutates in place, hence the copies.
        let mut x_cols: [Vec<f64>; 2] = [Vec::new(), Vec::new()];
        let mut singular = false;

        for (col_idx, x_col) in x_cols.iter_mut().enumerate() {
            matrix_buf.clear();
            matrix_buf.extend_from_slice(&cov.sigma_22);

            rhs_col.clear();
            for row in 0..k {
                rhs_col.push(cov.sigma_12[col_idx * k + row]);
            }

            if let Some(sol) = solve_linear_system(&mut matrix_buf, &mut rhs_col, k) {
                *x_col = sol;
            } else {
                singular = true;
                break;
            }
        }

        if singular {
            break;
        }

        // Σ̄ = Σ_11 − Σ_12 · X, row-major 2×2.
        let mut sigma_bar = cov.sigma_11;
        for r in 0..2 {
            for c in 0..2 {
                let correction: f64 = (0..k).map(|j| cov.sigma_12[r * k + j] * x_cols[c][j]).sum();
                sigma_bar[r * 2 + c] -= correction;
            }
        }

        let denom_sq = sigma_bar[0] * sigma_bar[3];
        let facp = if denom_sq <= 0.0 {
            0.0
        } else {
            (sigma_bar[1] / denom_sq.sqrt()).clamp(-1.0, 1.0)
        };

        facp_values.push(facp);
    }

    facp_values
}
