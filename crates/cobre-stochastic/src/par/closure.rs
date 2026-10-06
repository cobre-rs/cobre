//! Classical periodic-ACF closure for the standardized PAR process.
//!
//! The standardized coefficients `ψ*` plus the unit-marginal-variance contract
//! (`Var(z_m) = 1` per season) pin every residual std ratio `r_m`: solve the
//! periodic-ACF closure for the implied autocorrelations `ρ_m(k)` (the model's
//! own second-moment recursion, `ρ_m(0) = 1` imposed), then set
//! `r_m² = 1 − Σ_j ψ*_{m,j} ρ_m(j)`. See [`implied_periodic_acf`],
//! [`derive_residual_std_ratios`], and [`check_stationarity`] (the gate that
//! combines the closure with a stationarity test).
//!
//! ## PAR(p)-A extension
//!
//! The annual term `A_{t-1} = (1/12) Σ_{j=1..12} a_{t-j}` is a linear
//! functional of the last 12 inflows, so a PAR(p)-A model is an effective
//! 12-lag periodic AR with computable extra coefficients. [`AnnualParams`]
//! carries the standardized annual coefficient and `σ^A` per season;
//! [`derive_residual_std_ratios_annual`] and [`check_stationarity_annual`]
//! extend the classical closure to the effective system with `K = 12` (see
//! `fill_stage_arrays` in `par::precompute` for the ground-truth
//! original-unit construction these reproduce in standardized terms).

use std::cmp::Ordering;

use super::fitting::solve_linear_system;

/// Solves the periodic-ACF closure for the implied autocorrelations `ρ_m(l)`,
/// `l = 1..=k`, of the standardized coefficients `psi_by_season`.
///
/// Returns `rho[m] = [1.0, ρ_m(1), …, ρ_m(k)]` per season, or `None` when the
/// closure's `n = n_seasons * k` linear system is singular.
///
/// Season indices are positions in the season cycle: index `(m + n − 1) % n` is
/// season `m`'s predecessor.
#[must_use]
pub fn implied_periodic_acf(
    psi_by_season: &[Vec<f64>],
    orders: &[usize],
    n_seasons: usize,
    k: usize,
) -> Option<Vec<Vec<f64>>> {
    let n = n_seasons * k;
    let idx = |m: usize, l: usize| m * k + (l - 1);

    let mut a = vec![0.0_f64; n * n];
    let mut b = vec![0.0_f64; n];

    for m in 0..n_seasons {
        let psi = &psi_by_season[m];
        for kp in 1..=k {
            let row = idx(m, kp);
            a[row * n + idx(m, kp)] += 1.0;
            for j in 1..=orders[m] {
                let coeff = psi[j - 1];
                // `j % n_seasons` / `kp % n_seasons` prevent underflow when an
                // order exceeds n_seasons (matches `build_periodic_yw_matrix`'s guard).
                match j.cmp(&kp) {
                    Ordering::Equal => b[row] += coeff,
                    Ordering::Less => {
                        let col = idx((m + n_seasons - j % n_seasons) % n_seasons, kp - j);
                        a[row * n + col] -= coeff;
                    }
                    Ordering::Greater => {
                        let col = idx((m + n_seasons - kp % n_seasons) % n_seasons, j - kp);
                        a[row * n + col] -= coeff;
                    }
                }
            }
        }
    }

    let solved = solve_linear_system(&mut a, &mut b, n)?;

    Some(
        (0..n_seasons)
            .map(|m| {
                let mut row = Vec::with_capacity(k + 1);
                row.push(1.0);
                row.extend_from_slice(&solved[m * k..(m + 1) * k]);
                row
            })
            .collect(),
    )
}

/// Derives the residual std ratio `r_m` for each season from the standardized
/// coefficients alone, via [`implied_periodic_acf`] and the unit-marginal-
/// variance contract `r_m² = 1 − Σ_j ψ*_{m,j} ρ_m(j)`.
///
/// Short-circuits to `Some(vec![1.0; n_seasons])` without solving when every
/// season has order 0. Returns `None` when the underlying closure system is
/// singular.
#[must_use]
pub fn derive_residual_std_ratios(
    psi_by_season: &[Vec<f64>],
    orders: &[usize],
    n_seasons: usize,
) -> Option<Vec<f64>> {
    let k = orders.iter().copied().max().unwrap_or(0);
    if k == 0 {
        return Some(vec![1.0; n_seasons]);
    }

    let rho = implied_periodic_acf(psi_by_season, orders, n_seasons, k)?;

    Some(
        (0..n_seasons)
            .map(|m| {
                if orders[m] == 0 {
                    return 1.0;
                }
                let acc: f64 = (1..=orders[m])
                    .map(|j| psi_by_season[m][j - 1] * rho[m][j])
                    .sum();
                (1.0 - acc).sqrt()
            })
            .collect(),
    )
}

/// Reasons [`check_stationarity`] rejects a standardized coefficient set.
#[derive(Debug, Clone, PartialEq)]
pub enum ClosureRejection {
    /// The periodic-ACF closure (see [`implied_periodic_acf`]) solved to an
    /// implied residual variance `r_m²` at or below the numerical floor.
    NonPositiveResidualVariance {
        /// Season index (0-based) where `r_m²` failed the floor.
        season: usize,
        /// The offending `r_m²` value.
        r_squared: f64,
    },
    /// An implied autocorrelation `ρ_m(lag)` escaped `[-1, 1]` — the
    /// coefficient set is explosive even though the closure solved uniquely.
    AutocorrelationOutOfRange {
        /// Season index (0-based) of the offending autocorrelation.
        season: usize,
        /// Lag at which `|ρ_m(lag)|` exceeded the tolerance.
        lag: usize,
        /// The offending `ρ_m(lag)` value.
        rho: f64,
    },
    /// The periodic monodromy's Gelfand spectral-radius estimate reached the
    /// unit circle.
    NonStationaryMonodromy {
        /// The Frobenius–Gelfand spectral-radius estimate (an upper bound on
        /// the monodromy's true spectral radius).
        spectral_radius: f64,
    },
    /// The periodic-ACF closure system (see [`implied_periodic_acf`]) is
    /// singular.
    SingularClosure,
}

/// Gates a standardized coefficient set for stationarity, returning the
/// closure-derived residual std ratios `r = [r_0, …, r_{n_seasons-1}]` (see
/// [`derive_residual_std_ratios`]) on success.
///
/// Runs four checks against the periodic-ACF closure (see
/// [`implied_periodic_acf`]), in order, returning on the first failure:
///
/// 1. The closure system must solve —
///    [`ClosureRejection::SingularClosure`] otherwise.
/// 2. Every implied autocorrelation must satisfy `|ρ_m(l)| ≤ 1 + ε`
///    (`ε = 1e-9`) — [`ClosureRejection::AutocorrelationOutOfRange`]
///    otherwise. An explosive AR polynomial can still solve the closure
///    uniquely (its condition number stays near 1), so this check — not
///    closure singularity — is what catches it.
/// 3. Every implied residual variance `r_m² = 1 − Σ_j ψ*_{m,j} ρ_m(j)` must
///    exceed the floor `ε = 1e-12` —
///    [`ClosureRejection::NonPositiveResidualVariance`] otherwise.
/// 4. The periodic monodromy `M = C_{n_seasons-1}·…·C_0` — per-season
///    first-row companion matrices `C_m` (the coefficient row over a shifted
///    identity), current season multiplying on the left — must have a
///    Gelfand spectral-radius estimate `< 1 − ε` (`ε = 1e-9`) —
///    [`ClosureRejection::NonStationaryMonodromy`] otherwise. Running the
///    gate on standardized `ψ*` is exact: the standardized and physical
///    monodromies are related by a fixed diagonal similarity, so their
///    spectral radii coincide (a single-season `ψ* = [φ]` reduces this check
///    to the ordinary AR(1) condition `|φ| < 1`).
///
/// The spectral radius is estimated by repeated squaring
/// (`‖M^64‖_F^{1/64}`, the Frobenius–Gelfand estimate) rather than an
/// eigensolver, since the monodromy's eigenvalues can be complex and the
/// crate carries no linear-algebra dependency. The estimate is an upper
/// bound on the true spectral radius at every depth, so this gate never
/// accepts a truly explosive monodromy — its only error mode is rejecting a
/// stationary one whose radius sits within roughly `n^{1/128}` of the unit
/// circle (for `n = 12`, that ceiling is `12^{-1/128} ≈ 0.981`).
///
/// # Errors
///
/// Returns the first-triggered [`ClosureRejection`] variant from the checks
/// above.
pub fn check_stationarity(
    psi_by_season: &[Vec<f64>],
    orders: &[usize],
    n_seasons: usize,
) -> Result<Vec<f64>, ClosureRejection> {
    const AUTOCORRELATION_TOLERANCE: f64 = 1e-9;
    const RESIDUAL_VARIANCE_FLOOR: f64 = 1e-12;
    const SPECTRAL_RADIUS_TOLERANCE: f64 = 1e-9;

    let max_order = orders.iter().copied().max().unwrap_or(0);

    let implied_acf = implied_periodic_acf(psi_by_season, orders, n_seasons, max_order)
        .ok_or(ClosureRejection::SingularClosure)?;

    for (season, acf_row) in implied_acf.iter().enumerate() {
        if let Some((lag, &rho)) = acf_row
            .iter()
            .enumerate()
            .skip(1)
            .find(|&(_, &value)| value.abs() > 1.0 + AUTOCORRELATION_TOLERANCE)
        {
            return Err(ClosureRejection::AutocorrelationOutOfRange { season, lag, rho });
        }
    }

    let mut residual_std_ratio = vec![1.0_f64; n_seasons];
    for season in 0..n_seasons {
        let order = orders[season];
        if order == 0 {
            continue;
        }
        let explained: f64 = (1..=order)
            .map(|lag| psi_by_season[season][lag - 1] * implied_acf[season][lag])
            .sum();
        let r_squared = 1.0 - explained;
        if r_squared <= RESIDUAL_VARIANCE_FLOOR {
            return Err(ClosureRejection::NonPositiveResidualVariance { season, r_squared });
        }
        residual_std_ratio[season] = r_squared.sqrt();
    }

    let monodromy = monodromy_matrix(psi_by_season, orders, n_seasons, max_order);
    let spectral_radius = spectral_radius_gelfand(&monodromy, max_order);
    if spectral_radius >= 1.0 - SPECTRAL_RADIUS_TOLERANCE {
        return Err(ClosureRejection::NonStationaryMonodromy { spectral_radius });
    }

    Ok(residual_std_ratio)
}

// ---------------------------------------------------------------------------
// PAR(p)-A extension
// ---------------------------------------------------------------------------

/// Standardized PAR(p)-A annual-component parameters for a single season,
/// feeding [`derive_residual_std_ratios_annual`] and
/// [`check_stationarity_annual`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AnnualParams {
    /// Standardized annual coefficient `ψ` (dimensionless, direct
    /// Yule-Walker output — `precompute.rs`'s `ann.coefficient`).
    pub coefficient: f64,

    /// Sample std `σ^A` of the rolling 12-month average, in the same units as
    /// `seasonal_std`. `0.0` degenerates the season's annual term to
    /// classical (no divide-by-zero), matching `fill_stage_arrays`.
    pub sigma_a: f64,
}

/// Whether any season's annual term actually contributes (`σ^A ≠ 0`) — the
/// "engaged" state `fill_stage_arrays` gates on. `false` means the
/// annual term is inert (absent, or every present `σ^A == 0`) everywhere, so
/// the PAR(p)-A closure degenerates to the classical one bit-for-bit.
fn annual_engaged(annual: &[Option<AnnualParams>]) -> bool {
    annual
        .iter()
        .any(|entry| matches!(entry, Some(params) if params.sigma_a != 0.0))
}

/// Builds the per-season effective standardized coefficient vectors for the
/// PAR(p)-A model, reproducing `fill_stage_arrays`'s annual spread in
/// standardized terms.
///
/// The stride `k` widens to 12 whenever any season carries `annual:
/// Some(_)` at all — matching `precompute.rs`'s `any_annual` gate on the
/// shared array stride — even a season whose own `σ^A == 0` shares the wider
/// stride (its own annual term still contributes `0.0`). Returns
/// `(effective_psi, k)` with `effective_psi[m].len() == k` for every season;
/// `k == 0` when no season has a classical order or an annual component.
///
/// The effective coefficient at lag `τ` for season `m` is
/// `ψ*ᵉᶠᶠ_{m,τ} = (ψ*_{m,τ} if τ ≤ orders[m] else 0) + c_τ · ψ`, where
/// `c_τ = s_{m−τ} / (12 σ^A)` reproduces the original-unit `ψ̂/12` spread
/// (`fill_stage_arrays`) after the standardized↔original rescale
/// `ψ = ψ*ᵉᶠᶠ · s_m / s_{m−τ}`.
fn effective_psi_by_season(
    psi_by_season: &[Vec<f64>],
    orders: &[usize],
    annual: &[Option<AnnualParams>],
    seasonal_std: &[f64],
    n_seasons: usize,
) -> (Vec<Vec<f64>>, usize) {
    let classical_max_order = orders.iter().copied().max().unwrap_or(0);
    let any_annual = annual.iter().any(Option::is_some);
    let k = if any_annual {
        classical_max_order.max(12)
    } else {
        classical_max_order
    };

    if k == 0 {
        return (vec![Vec::new(); n_seasons], 0);
    }

    let effective_psi = (0..n_seasons)
        .map(|m| {
            (1..=k)
                .map(|tau| {
                    let classical_term = if tau <= orders[m] {
                        psi_by_season[m][tau - 1]
                    } else {
                        0.0
                    };
                    // Annual term reproduces precompute.rs's `ψ̂/12` spread,
                    // added at every lag up to the (possibly annual-widened)
                    // stride `k`, not just the first 12 — matching
                    // `precompute.rs`'s `for lag in 0..max_order` loop
                    // verbatim.
                    let annual_term = if k < 12 {
                        0.0
                    } else {
                        match &annual[m] {
                            Some(params) if params.sigma_a != 0.0 => {
                                let lag_season = (m + n_seasons - tau % n_seasons) % n_seasons;
                                params.coefficient * seasonal_std[lag_season]
                                    / (12.0 * params.sigma_a)
                            }
                            _ => 0.0,
                        }
                    };
                    classical_term + annual_term
                })
                .collect()
        })
        .collect();

    (effective_psi, k)
}

/// Derives the residual std ratio `r_m` for each season of a PAR(p)-A model
/// from the standardized coefficients alone, extending
/// [`derive_residual_std_ratios`] with the annual component's effective
/// contribution to the closure (see `effective_psi_by_season`).
///
/// Delegates to [`derive_residual_std_ratios`] outright when no season's
/// annual term is engaged (see [`AnnualParams`]) — for such a model this
/// returns the exact classical result. Otherwise solves
/// [`implied_periodic_acf`] on the effective 12-lag (or wider) system and
/// applies the same unit-marginal-variance contract
/// `r_m² = 1 − Σ_j ψ*ᵉᶠᶠ_{m,j} ρ_m(j)`.
///
/// Returns `None` when the underlying closure system is singular. Does not
/// itself guard against a non-stationary effective system (a negative
/// `1 − Σ…` yields `NaN`) — use [`check_stationarity_annual`] to gate that.
#[must_use]
pub fn derive_residual_std_ratios_annual(
    psi_by_season: &[Vec<f64>],
    orders: &[usize],
    annual: &[Option<AnnualParams>],
    seasonal_std: &[f64],
    n_seasons: usize,
) -> Option<Vec<f64>> {
    if !annual_engaged(annual) {
        return derive_residual_std_ratios(psi_by_season, orders, n_seasons);
    }

    let (effective_psi, k) =
        effective_psi_by_season(psi_by_season, orders, annual, seasonal_std, n_seasons);
    let orders_effective = vec![k; n_seasons];

    let rho = implied_periodic_acf(&effective_psi, &orders_effective, n_seasons, k)?;

    Some(
        (0..n_seasons)
            .map(|m| {
                let acc: f64 = (1..=k).map(|j| effective_psi[m][j - 1] * rho[m][j]).sum();
                (1.0 - acc).sqrt()
            })
            .collect(),
    )
}

/// Gates a PAR(p)-A standardized coefficient set for stationarity, extending
/// [`check_stationarity`] with the annual component's effective contribution
/// to the closure (see `effective_psi_by_season`).
///
/// Delegates to [`check_stationarity`] outright when no season's annual term
/// is engaged (see [`AnnualParams`]) — for such a model this returns exactly
/// what [`check_stationarity`] returns. Otherwise builds the per-season
/// effective standardized coefficient vectors (effective order 12, or wider
/// if the classical order exceeds it) and re-runs the full
/// [`check_stationarity`] gate — the `|ρ| > 1`, `r² ≤ 0`, and monodromy
/// checks — on that effective system, so an annual coefficient large enough
/// to make the effective system explosive is caught even when the classical
/// part alone is stationary.
///
/// # Errors
///
/// Returns the first-triggered [`ClosureRejection`] from
/// [`check_stationarity`]'s checks, run on the effective system.
pub fn check_stationarity_annual(
    psi_by_season: &[Vec<f64>],
    orders: &[usize],
    annual: &[Option<AnnualParams>],
    seasonal_std: &[f64],
    n_seasons: usize,
) -> Result<Vec<f64>, ClosureRejection> {
    if !annual_engaged(annual) {
        return check_stationarity(psi_by_season, orders, n_seasons);
    }

    let (effective_psi, k) =
        effective_psi_by_season(psi_by_season, orders, annual, seasonal_std, n_seasons);
    let orders_effective = vec![k; n_seasons];

    check_stationarity(&effective_psi, &orders_effective, n_seasons)
}

/// Periodic monodromy `M = C_{n_seasons-1}·…·C_0` at companion dimension
/// `dim`: each season's first-row companion matrix multiplies on the left.
/// Cyclic rotations of this product preserve `M`'s spectral radius;
/// reversing the order does not for `n_seasons > 2`.
fn monodromy_matrix(
    psi_by_season: &[Vec<f64>],
    orders: &[usize],
    n_seasons: usize,
    dim: usize,
) -> Vec<f64> {
    let mut product = identity_matrix(dim);
    for season in 0..n_seasons {
        let companion = companion_matrix(&psi_by_season[season], orders[season], dim);
        product = square_matrix_product(&companion, &product, dim);
    }
    product
}

/// First-row companion matrix at dimension `dim`: row 0 holds `psi`
/// zero-padded past `order`, rows `1..dim` shift the identity down
/// (`matrix[row][row - 1] = 1`).
fn companion_matrix(psi: &[f64], order: usize, dim: usize) -> Vec<f64> {
    let mut matrix = vec![0.0_f64; dim * dim];
    for (col, coeff) in matrix.iter_mut().take(dim).enumerate() {
        *coeff = if col < order { psi[col] } else { 0.0 };
    }
    for row in 1..dim {
        matrix[row * dim + row - 1] = 1.0;
    }
    matrix
}

fn identity_matrix(dim: usize) -> Vec<f64> {
    let mut matrix = vec![0.0_f64; dim * dim];
    for diag in 0..dim {
        matrix[diag * dim + diag] = 1.0;
    }
    matrix
}

/// Flat row-major `dim × dim` product `lhs · rhs`.
fn square_matrix_product(lhs: &[f64], rhs: &[f64], dim: usize) -> Vec<f64> {
    let mut product = vec![0.0_f64; dim * dim];
    for row in 0..dim {
        for col in 0..dim {
            let mut acc = 0.0_f64;
            for mid in 0..dim {
                acc += lhs[row * dim + mid] * rhs[mid * dim + col];
            }
            product[row * dim + col] = acc;
        }
    }
    product
}

/// Frobenius–Gelfand spectral-radius estimate `‖M^64‖_F^{1/64}`: six
/// repeated squarings (fixed for determinism), avoiding an eigensolver for
/// the monodromy's possibly-complex eigenvalues.
fn spectral_radius_gelfand(matrix: &[f64], dim: usize) -> f64 {
    let mut power = matrix.to_vec();
    for _ in 0..6 {
        power = square_matrix_product(&power, &power, dim);
    }
    let frobenius_norm = power.iter().map(|value| value * value).sum::<f64>().sqrt();
    frobenius_norm.powf(1.0 / 64.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    const N_SEASONS: usize = 4;

    /// Test-local port of the prototype's classical periodic YW solve
    /// (`par-noise-closure-check.py::yw_solve`): predicts season `m` from a
    /// `rho` table directly, mirroring `build_periodic_yw_matrix` but
    /// reading `rho` instead of raw observations.
    fn yw_solve_from_rho(m: usize, p: usize, rho: &[Vec<f64>]) -> (Vec<f64>, f64) {
        if p == 0 {
            return (Vec::new(), 1.0);
        }

        let mut r_mat = vec![0.0_f64; p * p];
        let mut rhs = vec![0.0_f64; p];

        for i in 0..p {
            r_mat[i * p + i] = 1.0;
            let ref_month = (m + N_SEASONS - (i + 1) % N_SEASONS) % N_SEASONS;
            for j in (i + 1)..p {
                let lag = j - i;
                let val = rho[ref_month][lag];
                r_mat[i * p + j] = val;
                r_mat[j * p + i] = val;
            }
        }
        for (i, entry) in rhs.iter_mut().enumerate().take(p) {
            *entry = rho[m][i + 1];
        }

        let mut a = r_mat;
        let mut b = rhs.clone();
        let psi = solve_linear_system(&mut a, &mut b, p).expect("test YW matrix invertible");
        let dot: f64 = psi.iter().zip(rhs.iter()).map(|(pi, ri)| pi * ri).sum();
        (psi, (1.0 - dot).sqrt())
    }

    /// Test-local port of the prototype's analytic stationary-variance solve
    /// (`par-noise-closure-check.py::stationary_variance`, lines 156-186):
    /// solves for the TRUE per-season stationary variance `γ_m(0)` (not pinned
    /// to 1) given `r_by_season`, as a cross-check independent of the closure
    /// under test.
    fn stationary_variance(
        psi_by_season: &[Vec<f64>],
        orders: &[usize],
        r_by_season: &[f64],
        k: usize,
    ) -> Vec<f64> {
        let n = N_SEASONS * (k + 1);
        let idx = |m: usize, l: usize| m * (k + 1) + l;

        let mut a = vec![0.0_f64; n * n];
        let mut b = vec![0.0_f64; n];

        for m in 0..N_SEASONS {
            let psi = &psi_by_season[m];

            let row0 = idx(m, 0);
            a[row0 * n + idx(m, 0)] += 1.0;
            for j in 1..=orders[m] {
                a[row0 * n + idx(m, j)] -= psi[j - 1];
            }
            b[row0] = r_by_season[m] * r_by_season[m];

            for kp in 1..=k {
                let row = idx(m, kp);
                a[row * n + idx(m, kp)] += 1.0;
                for j in 1..=orders[m] {
                    let coeff = psi[j - 1];
                    let col = match j.cmp(&kp) {
                        Ordering::Equal => idx((m + N_SEASONS - kp % N_SEASONS) % N_SEASONS, 0),
                        Ordering::Less => idx((m + N_SEASONS - j % N_SEASONS) % N_SEASONS, kp - j),
                        Ordering::Greater => {
                            idx((m + N_SEASONS - kp % N_SEASONS) % N_SEASONS, j - kp)
                        }
                    };
                    a[row * n + col] -= coeff;
                }
            }
        }

        let gamma0 =
            solve_linear_system(&mut a, &mut b, n).expect("test variance matrix invertible");
        (0..N_SEASONS).map(|m| gamma0[idx(m, 0)]).collect()
    }

    /// A "sample-like" per-season autocorrelation table: plausible in
    /// magnitude and decay, but not exactly consistent with any model — like a
    /// real sample ACF (prototype `make_rhohat`).
    fn rho_hat() -> Vec<Vec<f64>> {
        vec![
            vec![1.0, 0.42, 0.21, 0.08],
            vec![1.0, 0.35, 0.17, 0.06],
            vec![1.0, 0.30, 0.12, 0.03],
            vec![1.0, 0.38, 0.19, 0.07],
        ]
    }

    #[test]
    fn t1_uniform_orders_exact() {
        let orders = [3, 3, 3, 3];
        let rho_hat = rho_hat();

        let (psi, r_fit): (Vec<_>, Vec<_>) = orders
            .iter()
            .enumerate()
            .map(|(m, &order)| yw_solve_from_rho(m, order, &rho_hat))
            .unzip();

        let rho_imp = implied_periodic_acf(&psi, &orders, N_SEASONS, 3).unwrap();
        let r_imp = derive_residual_std_ratios(&psi, &orders, N_SEASONS).unwrap();

        let gap_rho = (0..N_SEASONS)
            .flat_map(|m| (1..=3).map(move |l| (m, l)))
            .map(|(m, l)| (rho_imp[m][l] - rho_hat[m][l]).abs())
            .fold(0.0_f64, f64::max);
        let gap_r = (0..N_SEASONS)
            .map(|m| (r_imp[m] - r_fit[m]).abs())
            .fold(0.0_f64, f64::max);

        assert!(gap_rho < 1e-12, "max|rho_implied - rho_hat| = {gap_rho:e}");
        assert!(gap_r < 1e-12, "max|r_derived - r_fitted| = {gap_r:e}");
    }

    #[test]
    fn t3_round_trip() {
        let orders = [3, 1, 2, 1];
        let rho_hat = rho_hat();

        let psi: Vec<Vec<f64>> = (0..N_SEASONS)
            .map(|m| yw_solve_from_rho(m, orders[m], &rho_hat).0)
            .collect();

        let rho_model = implied_periodic_acf(&psi, &orders, N_SEASONS, 3).unwrap();

        let psi_rt: Vec<Vec<f64>> = (0..N_SEASONS)
            .map(|m| yw_solve_from_rho(m, orders[m], &rho_model).0)
            .collect();

        let gap_psi = (0..N_SEASONS)
            .map(|m| {
                psi_rt[m]
                    .iter()
                    .zip(psi[m].iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0_f64, f64::max)
            })
            .fold(0.0_f64, f64::max);

        assert!(gap_psi < 1e-12, "max|psi_recovered - psi| = {gap_psi:e}");
    }

    #[test]
    fn t4_analytic_unit_variance() {
        let orders = [3, 1, 2, 1];
        let rho_hat = rho_hat();

        let psi: Vec<Vec<f64>> = (0..N_SEASONS)
            .map(|m| yw_solve_from_rho(m, orders[m], &rho_hat).0)
            .collect();

        let r_derived = derive_residual_std_ratios(&psi, &orders, N_SEASONS).unwrap();
        let var_analytic = stationary_variance(&psi, &orders, &r_derived, 3);

        for (m, v) in var_analytic.iter().enumerate() {
            assert!((v - 1.0).abs() < 1e-10, "season {m}: variance = {v}");
        }
    }

    #[test]
    fn order_zero_season_r_is_one() {
        let orders = [3, 3, 0, 3];
        let rho_hat = rho_hat();

        let psi: Vec<Vec<f64>> = (0..N_SEASONS)
            .map(|m| yw_solve_from_rho(m, orders[m], &rho_hat).0)
            .collect();

        let r = derive_residual_std_ratios(&psi, &orders, N_SEASONS).unwrap();
        assert_eq!(r[2], 1.0);

        let rho = implied_periodic_acf(&psi, &orders, N_SEASONS, 3).unwrap();
        for (l, &val) in rho[2].iter().enumerate().skip(1) {
            assert_eq!(val, 0.0, "rho[2][{l}] should be exactly 0.0");
        }
    }

    #[test]
    fn explosive_ar1_rejected() {
        let orders = [1];
        let psi = vec![vec![1.2]];

        let err = check_stationarity(&psi, &orders, 1).unwrap_err();
        assert!(
            matches!(
                err,
                ClosureRejection::AutocorrelationOutOfRange { .. }
                    | ClosureRejection::NonPositiveResidualVariance { .. }
            ),
            "expected AutocorrelationOutOfRange or NonPositiveResidualVariance, got {err:?}"
        );
    }

    #[test]
    fn t2_mixed_orders_gate_passes() {
        let orders = [3, 1, 2, 1];
        let rho_hat = rho_hat();

        let (psi, r_fit): (Vec<_>, Vec<_>) = orders
            .iter()
            .enumerate()
            .map(|(m, &order)| yw_solve_from_rho(m, order, &rho_hat))
            .unzip();

        let r = check_stationarity(&psi, &orders, N_SEASONS).expect("stationary mixed-order set");

        for (season, (&derived, &fitted)) in r.iter().zip(r_fit.iter()).enumerate() {
            let gap = (derived - fitted).abs();
            if season == 0 {
                assert!(
                    (1e-5..1e-3).contains(&gap),
                    "season {season}: expected a ~1e-4-scale gap, got {gap:e}"
                );
            } else {
                assert!(
                    gap < 1e-9,
                    "season {season}: expected an exact match, got {gap:e}"
                );
            }
        }
    }

    #[test]
    fn gelfand_matches_known_radius() {
        let matrix = [0.5, 0.0, 0.0, 0.05];

        let radius = spectral_radius_gelfand(&matrix, 2);

        assert!(
            (radius - 0.5).abs() < 1e-6,
            "expected radius near 0.5, got {radius}"
        );
    }

    #[test]
    fn stationary_ar1_accepted() {
        let orders = [1];
        let psi = vec![vec![0.5]];

        let r = check_stationarity(&psi, &orders, 1).expect("stationary AR(1) accepted");

        let expected = (1.0_f64 - 0.25).sqrt();
        assert!(
            (r[0] - expected).abs() < 1e-12,
            "expected r[0] ~ {expected}, got {}",
            r[0]
        );
    }

    // -----------------------------------------------------------------------
    // PAR(p)-A extension tests
    // -----------------------------------------------------------------------

    use chrono::NaiveDate;
    use cobre_core::{
        EntityId,
        scenario::{AnnualComponent, InflowModel},
        temporal::{
            Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
            StageStateConfig,
        },
    };

    use crate::par::precompute::PrecomputedPar;

    fn par_a_stage(index: usize, id: i32, season_id: usize) -> Stage {
        Stage {
            index,
            id,
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            season_id: Some(season_id),
            blocks: vec![Block {
                index: 0,
                name: "SINGLE".to_string(),
                duration_hours: 744.0,
            }],
            block_mode: BlockMode::Parallel,
            state_config: StageStateConfig {
                storage: true,
                inflow_lags: false,
            },
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor: 10,
                noise_method: NoiseMethod::Saa,
            },
        }
    }

    /// 12-month PAR(p)-A fixture over one stage per season:
    /// `mean = 100 + 10*stage_id`, `std = 20 + 2*stage_id` for every month —
    /// all strictly positive (the bit-parity requirement needs every
    /// `s_{m-τ} > 0`). Per-month classical order and annual component come
    /// from a [`MonthSpec`] pattern.
    struct ParAPrecomputeFixture {
        psi_by_season: Vec<Vec<f64>>,
        orders: Vec<usize>,
        annual: Vec<Option<AnnualParams>>,
        seasonal_std: Vec<f64>,
        models: Vec<InflowModel>,
        stages: Vec<Stage>,
    }

    /// Per-month fixture spec: classical AR order (coefficients generated
    /// from [`PSI_BASE`], month-scaled) + optional annual
    /// `(coefficient, sigma_a)`.
    type MonthSpec = (usize, Option<(f64, f64)>);

    const PSI_BASE: [f64; 16] = [
        0.30, -0.15, 0.10, -0.05, 0.08, -0.04, 0.06, -0.03, 0.05, -0.02, 0.04, -0.02, 0.03, -0.01,
        0.02, -0.01,
    ];

    fn par_a_precompute_fixture_with(pattern: &[MonthSpec; 12]) -> ParAPrecomputeFixture {
        let mut stages = Vec::with_capacity(12);
        let mut seasonal_std = Vec::with_capacity(12);
        let mut psi_by_season = vec![Vec::new(); 12];
        let mut orders = vec![0usize; 12];
        let mut annual = vec![None; 12];
        let mut models = Vec::with_capacity(12);

        for stage_id in 0..12_i32 {
            let m = usize::try_from(stage_id).unwrap();
            let fi = f64::from(stage_id);
            let mean = 100.0 + fi * 10.0;
            let std = 20.0 + fi * 2.0;
            let (order, month_annual) = pattern[m];

            stages.push(par_a_stage(m, stage_id, m));
            seasonal_std.push(std);

            let coeffs: Vec<f64> = PSI_BASE[..order]
                .iter()
                .map(|&base| base * fi.mul_add(0.02, 1.0))
                .collect();
            psi_by_season[m].clone_from(&coeffs);
            orders[m] = order;
            annual[m] = month_annual.map(|(coefficient, sigma_a)| AnnualParams {
                coefficient,
                sigma_a,
            });

            models.push(InflowModel {
                hydro_id: EntityId(1),
                stage_id,
                mean_m3s: mean,
                std_m3s: std,
                ar_coefficients: coeffs,
                residual_std_ratio: 0.9,
                annual: month_annual.map(|(coefficient, sigma_a)| AnnualComponent {
                    coefficient,
                    mean_m3s: mean,
                    std_m3s: sigma_a,
                }),
            });
        }

        ParAPrecomputeFixture {
            psi_by_season,
            orders,
            annual,
            seasonal_std,
            models,
            stages,
        }
    }

    /// `precompute.rs`'s own `par_a_on_12_seasons_with_season_fallback`
    /// pattern: only month 0 carries an AR(1) coefficient and an annual
    /// component; the other months are AR(0), no annual.
    fn par_a_precompute_fixture() -> ParAPrecomputeFixture {
        let mut pattern: [MonthSpec; 12] = [(0, None); 12];
        pattern[0] = (1, Some((0.4, 18.0)));
        par_a_precompute_fixture_with(&pattern)
    }

    /// The cross-module agreement pin: [`effective_psi_by_season`]'s
    /// standardized coefficients, rescaled to original units, must equal the
    /// actual `PrecomputedPar` psi buffer for every `(season, lag)` — the
    /// closure derives the innovation scale for exactly the coefficients
    /// precompute generates scenarios with, and nothing but this agreement
    /// binds the two formulas.
    fn assert_effective_matches_precompute(fixture: &ParAPrecomputeFixture) {
        let expected_stride = fixture.orders.iter().copied().max().unwrap_or(0).max(
            if fixture.annual.iter().any(Option::is_some) {
                12
            } else {
                0
            },
        );

        let lp = PrecomputedPar::build(&fixture.models, &fixture.stages, &[EntityId(1)], None)
            .expect("precompute build succeeds");
        assert_eq!(lp.max_order(), expected_stride, "precompute stride");

        let (effective_psi, k) = effective_psi_by_season(
            &fixture.psi_by_season,
            &fixture.orders,
            &fixture.annual,
            &fixture.seasonal_std,
            12,
        );
        assert_eq!(k, expected_stride, "closure stride");

        for (m, effective_psi_m) in effective_psi.iter().enumerate() {
            let psi_slice = lp.psi_slice(m, 0);
            for tau in 1..=k {
                let lag_season = (m + 12 - tau % 12) % 12;
                let reconstructed = effective_psi_m[tau - 1] * fixture.seasonal_std[m]
                    / fixture.seasonal_std[lag_season];
                let gap = (reconstructed - psi_slice[tau - 1]).abs();
                assert!(
                    gap < 1e-12,
                    "season {m} lag {tau}: reconstructed={reconstructed}, \
                     precompute={}, gap={gap:e}",
                    psi_slice[tau - 1]
                );
            }
        }
    }

    /// Pins [`effective_psi_by_season`]'s reconstruction against the actual
    /// `PrecomputedPar` build (the real precompute code, not a
    /// hand-transcribed port) on the single-annual-month pattern.
    #[test]
    fn par_a_effective_coeffs_match_precompute() {
        assert_effective_matches_precompute(&par_a_precompute_fixture());
    }

    /// The same agreement pinned across a deliberately diverse pattern
    /// matrix — annual on every month with mixed signs and orders,
    /// present-but-inert (`σ^A == 0`) months beside engaged ones, a
    /// classical order exceeding 12 (stride widened past the annual
    /// window), and annual-only (order 0 everywhere) — the shapes the
    /// single-pattern pin above cannot catch a formula desync on.
    #[test]
    fn par_a_effective_coeffs_match_precompute_pattern_sweep() {
        let all_annual_mixed_orders: [MonthSpec; 12] = [
            (1, Some((0.35, 12.0))),
            (0, Some((-0.22, 15.0))),
            (2, Some((0.31, 9.0))),
            (3, Some((-0.18, 20.0))),
            (0, Some((0.27, 11.0))),
            (1, Some((-0.33, 14.0))),
            (2, Some((0.15, 17.0))),
            (1, Some((-0.29, 8.0))),
            (3, Some((0.24, 13.0))),
            (0, Some((-0.12, 19.0))),
            (2, Some((0.38, 10.0))),
            (1, Some((-0.21, 16.0))),
        ];
        let inert_mix: [MonthSpec; 12] = [
            (3, Some((0.40, 18.0))),
            (1, None),
            (2, Some((-0.25, 12.0))),
            (1, None),
            (0, Some((0.55, 0.0))),
            (2, None),
            (1, Some((0.19, 21.0))),
            (3, None),
            (0, Some((-0.31, 15.0))),
            (1, None),
            (2, Some((0.62, 0.0))),
            (1, None),
        ];
        let wide_stride: [MonthSpec; 12] = [
            (0, None),
            (1, None),
            (0, None),
            (13, Some((0.28, 14.0))),
            (0, None),
            (2, None),
            (0, None),
            (1, Some((-0.17, 11.0))),
            (0, None),
            (0, None),
            (1, None),
            (0, None),
        ];
        let annual_only: [MonthSpec; 12] = [
            (0, Some((0.30, 10.0))),
            (0, Some((-0.20, 12.0))),
            (0, Some((0.25, 14.0))),
            (0, Some((-0.15, 16.0))),
            (0, Some((0.35, 18.0))),
            (0, Some((-0.10, 20.0))),
            (0, Some((0.28, 9.0))),
            (0, Some((-0.24, 11.0))),
            (0, Some((0.18, 13.0))),
            (0, Some((-0.32, 15.0))),
            (0, Some((0.22, 17.0))),
            (0, Some((-0.26, 19.0))),
        ];

        for (name, pattern) in [
            ("all_annual_mixed_orders", &all_annual_mixed_orders),
            ("inert_mix", &inert_mix),
            ("wide_stride", &wide_stride),
            ("annual_only", &annual_only),
        ] {
            println!("pattern: {name}");
            assert_effective_matches_precompute(&par_a_precompute_fixture_with(pattern));
        }
    }

    #[test]
    fn par_a_none_equals_classical() {
        let orders = [3, 1, 2, 1];
        let rho_hat = rho_hat();
        let psi: Vec<Vec<f64>> = (0..N_SEASONS)
            .map(|m| yw_solve_from_rho(m, orders[m], &rho_hat).0)
            .collect();
        let seasonal_std = vec![25.0, 22.0, 28.0, 24.0];
        let annual: Vec<Option<AnnualParams>> = vec![None; N_SEASONS];

        let r_classical =
            derive_residual_std_ratios(&psi, &orders, N_SEASONS).expect("classical closure solves");
        let r_annual =
            derive_residual_std_ratios_annual(&psi, &orders, &annual, &seasonal_std, N_SEASONS)
                .expect("annual closure solves");

        for (m, (&c, &a)) in r_classical.iter().zip(r_annual.iter()).enumerate() {
            assert!(
                (c - a).abs() < 1e-12,
                "season {m}: classical r={c}, annual r={a}"
            );
        }
    }

    #[test]
    fn par_a_zero_sigma_degenerates_classical() {
        let orders = [3, 1, 2, 1];
        let rho_hat = rho_hat();
        let psi: Vec<Vec<f64>> = (0..N_SEASONS)
            .map(|m| yw_solve_from_rho(m, orders[m], &rho_hat).0)
            .collect();
        let seasonal_std = vec![25.0, 22.0, 28.0, 24.0];
        // Every present annual entry has σ^A == 0 — the whole model is
        // inert, so the closure must degenerate to classical exactly (no
        // divide-by-zero), matching `fill_stage_arrays`.
        let annual: Vec<Option<AnnualParams>> = vec![
            Some(AnnualParams {
                coefficient: 0.7,
                sigma_a: 0.0,
            }),
            None,
            Some(AnnualParams {
                coefficient: -0.3,
                sigma_a: 0.0,
            }),
            None,
        ];

        let r_classical =
            derive_residual_std_ratios(&psi, &orders, N_SEASONS).expect("classical closure solves");
        let r_annual =
            derive_residual_std_ratios_annual(&psi, &orders, &annual, &seasonal_std, N_SEASONS)
                .expect("annual closure solves");

        for (m, (&c, &a)) in r_classical.iter().zip(r_annual.iter()).enumerate() {
            assert!(a.is_finite(), "season {m}: r_annual is not finite ({a})");
            assert!(
                (c - a).abs() < 1e-12,
                "season {m}: classical r={c}, annual r={a}"
            );
        }
    }

    #[test]
    fn par_a_explosive_effective_rejected() {
        let orders = [1];
        let psi = vec![vec![0.3]];
        let seasonal_std = vec![20.0];
        let annual = vec![Some(AnnualParams {
            coefficient: 5.0,
            sigma_a: 1.0,
        })];

        check_stationarity(&psi, &orders, 1).expect("classical part alone is stationary");

        let err = check_stationarity_annual(&psi, &orders, &annual, &seasonal_std, 1)
            .expect_err("explosive effective annual system must be rejected");
        assert!(
            !matches!(err, ClosureRejection::SingularClosure),
            "expected a typed rejection other than SingularClosure, got {err:?}"
        );
    }

    #[test]
    fn par_a_gate_none_equals_classical() {
        let orders = [3, 1, 2, 1];
        let rho_hat = rho_hat();
        let psi: Vec<Vec<f64>> = (0..N_SEASONS)
            .map(|m| yw_solve_from_rho(m, orders[m], &rho_hat).0)
            .collect();
        let seasonal_std = vec![25.0, 22.0, 28.0, 24.0];
        let annual: Vec<Option<AnnualParams>> = vec![None; N_SEASONS];

        let expected = check_stationarity(&psi, &orders, N_SEASONS);
        let actual = check_stationarity_annual(&psi, &orders, &annual, &seasonal_std, N_SEASONS);
        assert_eq!(actual, expected, "Ok case must be bit-equal");

        let explosive_orders = [1];
        let explosive_psi = vec![vec![1.2]];
        let explosive_std = vec![20.0];
        let explosive_annual: Vec<Option<AnnualParams>> = vec![None];

        let expected_err = check_stationarity(&explosive_psi, &explosive_orders, 1);
        let actual_err = check_stationarity_annual(
            &explosive_psi,
            &explosive_orders,
            &explosive_annual,
            &explosive_std,
            1,
        );
        assert_eq!(
            actual_err, expected_err,
            "rejection variant must be identical"
        );
    }
}
