//! AR-order selection via the Akaike Information Criterion and the periodic
//! partial autocorrelation function.

use super::yw_matrices::{build_periodic_yw_matrix_into, solve_linear_system};

/// Result of AIC-based AR order selection.
#[must_use]
#[derive(Debug, Clone, PartialEq)]
pub struct AicSelectionResult {
    /// Selected AR order; `0` when white noise is optimal or `sigma2_per_order`
    /// is empty.
    pub selected_order: usize,
    /// AIC per candidate order `0..=p_max`; `aic_values[0]` is the order-0
    /// white-noise baseline (`0.0`).
    pub aic_values: Vec<f64>,
}

/// Select the AR order that minimises the Akaike Information Criterion
/// `AIC(p) = N * ln(σ²_p) + 2p` over `p` in `1..=p_max`.
///
/// The white-noise baseline (order 0) is `AIC(0) = 0.0` by convention
/// (`σ²_0 = 1.0` in the normalised Yule-Walker formulation). On ties the lower
/// order wins (parsimony). Non-positive `sigma2` values are excluded via
/// `AIC = f64::INFINITY`.
///
/// # Parameters
///
/// - `sigma2_per_order` — prediction error variances; `sigma2_per_order[k]`
///   corresponds to AR order `k+1`.
/// - `n_observations` — number of historical observations for this season.
///
/// # Examples
///
/// ```
/// use cobre_stochastic::par::fitting::select_order_aic;
///
/// // A variance drop at order 1 that outweighs the penalty selects order 1.
/// let result = select_order_aic(&[0.3], 100);
/// assert_eq!(result.selected_order, 1);
/// assert_eq!(result.aic_values.len(), 2);
///
/// // Empty sigma2 always selects white noise.
/// let result = select_order_aic(&[], 50);
/// assert_eq!(result.selected_order, 0);
/// assert_eq!(result.aic_values, vec![0.0]);
/// ```
pub fn select_order_aic(sigma2_per_order: &[f64], n_observations: usize) -> AicSelectionResult {
    let p_max = sigma2_per_order.len();

    let mut aic_values = Vec::with_capacity(p_max + 1);
    aic_values.push(0.0_f64);

    #[allow(clippy::cast_precision_loss)]
    let n = n_observations as f64;

    for (k, &sigma2) in sigma2_per_order.iter().enumerate() {
        #[allow(clippy::cast_precision_loss)]
        let order = (k + 1) as f64;
        let aic = if sigma2 <= 0.0 {
            f64::INFINITY
        } else {
            n * sigma2.ln() + 2.0 * order
        };
        aic_values.push(aic);
    }

    // Strict `<` in the fold resolves ties to the first (lower-order) index —
    // the parsimony tie-break; `min_by` would not guarantee it.
    let selected_order = aic_values
        .iter()
        .enumerate()
        .fold(
            (0usize, f64::INFINITY),
            |(best_idx, best_val), (idx, &val)| {
                if val < best_val {
                    (idx, val)
                } else {
                    (best_idx, best_val)
                }
            },
        )
        .0;

    AicSelectionResult {
        selected_order,
        aic_values,
    }
}

/// Result of PACF-based AR order selection.
#[must_use]
#[derive(Debug, Clone, PartialEq)]
pub struct PacfSelectionResult {
    /// Selected AR order; `0` when no lag exceeds the significance threshold.
    /// The order names the outermost significant lag, never a subset, so the
    /// fitted coefficient vector carries one entry per lag `1..=selected_order`,
    /// where position `k` is lag `k + 1`.
    pub selected_order: usize,
    /// PACF values for lags `1..=p_max`; `pacf_values[k]` is the PACF at lag `k+1`.
    pub pacf_values: Vec<f64>,
    /// Significance threshold `z_alpha / sqrt(n_observations)`.
    pub threshold: f64,
}

/// Select the AR order by PACF significance testing: the **maximum** lag whose
/// `|PACF|` exceeds `z_alpha / sqrt(N)`, or order 0 (white noise) when none do.
///
/// # Parameters
///
/// - `parcor` -- partial autocorrelation coefficients; `parcor[k]` is the PACF
///   at lag `k+1`.
/// - `n_observations` -- number of historical observations for this season.
/// - `z_alpha` -- z-score for the desired confidence level (e.g., `1.96`
///   for 95% two-sided).
///
/// # Examples
///
/// ```
/// use cobre_stochastic::par::fitting::select_order_pacf;
///
/// // PACF at lag 1 = 0.5 exceeds 1.96/sqrt(100) = 0.196; lag 2 = 0.1 does not.
/// let result = select_order_pacf(&[0.5, 0.1], 100, 1.96);
/// assert_eq!(result.selected_order, 1);
///
/// // No significant PACF values -> order 0.
/// let result = select_order_pacf(&[0.05, 0.03], 100, 1.96);
/// assert_eq!(result.selected_order, 0);
/// ```
pub fn select_order_pacf(
    parcor: &[f64],
    n_observations: usize,
    z_alpha: f64,
) -> PacfSelectionResult {
    #[allow(clippy::cast_precision_loss)]
    let threshold = if n_observations > 0 {
        z_alpha / (n_observations as f64).sqrt()
    } else {
        f64::INFINITY
    };

    let selected_order = parcor
        .iter()
        .enumerate()
        .rev()
        .find(|&(_, p)| p.abs() > threshold)
        .map_or(0, |(k, _)| k + 1);

    PacfSelectionResult {
        selected_order,
        pacf_values: parcor.to_vec(),
        threshold,
    }
}

/// Select the AR order for a PAR(p)-A model from the conditional FACP, with two
/// extensions over the classical "largest lag exceeding `z_alpha / sqrt(N)`"
/// rule for cases the classical rule leaves undefined:
///
/// 1. **Structural-zero short-circuit at lag 1.** `conditional_facp[0] == 0.0`
///    exactly forces order 0: it marks a degenerate (Z, A) bucket, and the
///    convention refuses any AR structure on top of it. A structural zero at a
///    **higher** lag does not short-circuit — lag 1 being non-degenerate keeps
///    the AR(1) base (e.g. `[+0.37, 0, 0, 0, 0, 0]` -> order 1, not 0).
/// 2. **Minimum order 1 when lag 1 is non-zero.** Selected order is
///    `max(1, max_significant_lag)`, so a well-defined lag 1 defaults to AR(1)
///    even when no lag exceeds the threshold.
///
/// The order returned is tentative: the Maceira-Damazio iterative order
/// reduction runs across the whole periodic cycle and is the caller's
/// responsibility.
///
/// `pacf_values[k]` is the conditional FACP at lag `k+1`.
///
/// # Parameters
///
/// - `conditional_facp` -- conditional FACP coefficients from
///   [`conditional_facp_partitioned`](super::conditional_facp_partitioned). `conditional_facp[k]` is the
///   conditional FACP at lag `k+1`.
/// - `n_observations` -- number of historical observations for the given
///   (hydro, season) pair.
/// - `z_alpha` -- z-score for the desired confidence level (e.g., `1.96`
///   for 95% two-sided).
///
/// # Examples
///
/// ```
/// use cobre_stochastic::par::fitting::select_order_pacf_annual;
///
/// // Conditional FACP at lag 1 = 0.5 exceeds 1.96/sqrt(100) = 0.196; lag 2 = 0.1 does not.
/// let result = select_order_pacf_annual(&[0.5, 0.1], 100, 1.96);
/// assert_eq!(result.selected_order, 1);
///
/// // Lag 1 is non-zero (just small) -> min-order-1 rule kicks in.
/// let result = select_order_pacf_annual(&[0.05, 0.03], 100, 1.96);
/// assert_eq!(result.selected_order, 1);
///
/// // Structural zero at lag 1 -> order 0 (degenerate bucket).
/// let result = select_order_pacf_annual(&[0.0, 0.5], 100, 1.96);
/// assert_eq!(result.selected_order, 0);
/// ```
pub fn select_order_pacf_annual(
    conditional_facp: &[f64],
    n_observations: usize,
    z_alpha: f64,
) -> PacfSelectionResult {
    #[allow(clippy::cast_precision_loss)]
    let threshold = if n_observations > 0 {
        z_alpha / (n_observations as f64).sqrt()
    } else {
        f64::INFINITY
    };

    if conditional_facp.first().copied() == Some(0.0) {
        return PacfSelectionResult {
            selected_order: 0,
            pacf_values: conditional_facp.to_vec(),
            threshold,
        };
    }

    let max_significant = conditional_facp
        .iter()
        .enumerate()
        .rev()
        .find(|&(_, p)| p.abs() > threshold)
        .map_or(0, |(k, _)| k + 1);

    let selected_order = match conditional_facp.first() {
        Some(&p1) if p1 != 0.0 => max_significant.max(1),
        _ => max_significant,
    };

    PacfSelectionResult {
        selected_order,
        pacf_values: conditional_facp.to_vec(),
        threshold,
    }
}

/// Compute the periodic PACF for a given season up to `max_order`, extracting
/// `PACF(k) = phi[k-1]` from the order-`k` periodic Yule-Walker solve.
///
/// Accounts for the non-Toeplitz covariance of periodic processes — the
/// stationary Levinson-Durbin reflection coefficients assume a Toeplitz matrix
/// and are wrong here.
///
/// # Parameters
///
/// - `season` -- 0-based target season.
/// - `max_order` -- maximum lag to compute PACF for.
/// - `n_seasons` -- total number of seasons in the periodic cycle.
/// - `observations_by_season` -- observations grouped by season.
/// - `stats_by_season` -- `(mean, std)` for each season.
/// - `year_starts` -- first year of each season's bucket (see
///   [`periodic_autocorrelation`](super::periodic_autocorrelation)).
///
/// # Returns
///
/// A `Vec<f64>` of length <= `max_order`. Entry `k` is `PACF(k+1)`.
/// The vector may be shorter than `max_order` if a system at some order
/// is singular (remaining orders are skipped).
#[must_use]
pub fn periodic_pacf(
    season: usize,
    max_order: usize,
    n_seasons: usize,
    observations_by_season: &[&[f64]],
    stats_by_season: &[(f64, f64)],
    year_starts: &[i32],
) -> Vec<f64> {
    let mut pacf_values = Vec::with_capacity(max_order);

    // Reused across the loop so build_periodic_yw_matrix_into resizes in place
    // rather than allocating a Vec pair per order k.
    let mut matrix_buf: Vec<f64> = Vec::new();
    let mut rhs_buf: Vec<f64> = Vec::new();

    for k in 1..=max_order {
        build_periodic_yw_matrix_into(
            season,
            k,
            n_seasons,
            observations_by_season,
            stats_by_season,
            year_starts,
            &mut matrix_buf,
            &mut rhs_buf,
        );

        match solve_linear_system(&mut matrix_buf, &mut rhs_buf, k) {
            Some(phi) => pacf_values.push(phi[k - 1]),
            None => break,
        }
    }

    pacf_values
}
