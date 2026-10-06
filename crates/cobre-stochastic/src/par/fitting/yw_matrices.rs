//! Periodic Yule-Walker matrix construction and the dense linear solver.
//!
//! Provides the periodic autocorrelation primitive, the classical and extended
//! Yule-Walker matrix builders, the annual cross-correlation primitives, and the
//! small Gaussian-elimination solver shared across the fitter.

/// Compute the periodic normalised autocorrelation `rho(p, k)` for reference
/// season `p` and lag `k`, using a population (1/N) divisor and cross-year lag
/// adjustment.
///
/// Unlike a stationary autocorrelation, the reference season fixes the "current"
/// observations and stats while the lag selects the "lagged" ones.
///
/// Season indices are positions in the season cycle: index `(m + n − 1) % n` is
/// season `m`'s predecessor.
///
/// # Cross-year alignment
///
/// Entry `i` of season `s`'s bucket is year `year_starts[s] + i`, and buckets
/// pair by that absolute year, not by index: season `p` in year `Y` pairs with
/// the lagged season in year `Y − b`, where `b` is the number of cycle
/// boundaries the lag crosses (`⌈(k − p) / n⌉` when `k > p`, else `0`).
///
/// # Parameters
///
/// - `ref_season` -- 0-based season index of the reference month `p`.
/// - `lag` -- lag in seasonal periods (1-based: lag=1 is one season back).
/// - `n_seasons` -- total number of seasons in the periodic cycle.
/// - `observations_by_season` -- chronological values grouped by season.
/// - `stats_by_season` -- `(mean, std)` per season.
/// - `year_starts` -- first year of each season's bucket; equal entries recover
///   the by-index pairing.
///
/// # Returns
///
/// `rho(ref_season, lag)` clamped to `[-1, 1]`; `0.0` when either season has
/// zero std or insufficient paired observations exist.
#[must_use]
pub fn periodic_autocorrelation(
    ref_season: usize,
    lag: usize,
    n_seasons: usize,
    observations_by_season: &[&[f64]],
    stats_by_season: &[(f64, f64)],
    year_starts: &[i32],
) -> f64 {
    if lag == 0 {
        return 1.0;
    }

    let lag_season = (ref_season + n_seasons - lag % n_seasons) % n_seasons;

    let (mu_ref, std_ref) = stats_by_season[ref_season];
    let (mu_lag, std_lag) = stats_by_season[lag_season];

    // Zero std ⇒ autocorrelation undefined.
    if std_ref.abs() < f64::EPSILON || std_lag.abs() < f64::EPSILON {
        return 0.0;
    }

    let ref_obs = observations_by_season[ref_season];
    let lag_obs = observations_by_season[lag_season];

    #[allow(clippy::cast_possible_wrap)]
    let (ref_start, lag_start) = bucket_offsets(
        year_starts[ref_season],
        year_starts[lag_season],
        lag_years_back(ref_season, lag, n_seasons) as i64,
    );
    let n_pairs = ref_obs
        .len()
        .saturating_sub(ref_start)
        .min(lag_obs.len().saturating_sub(lag_start));

    if n_pairs == 0 {
        return 0.0;
    }

    // Z⊗Z divides by n_pairs — the max-bucket-size divisor of
    // cross_correlation_z_a / _a_z_neg1 applies only to Z⊗A cross-terms, whose
    // buckets differ in length.
    let mut gamma = 0.0_f64;
    for i in 0..n_pairs {
        gamma += (ref_obs[ref_start + i] - mu_ref) * (lag_obs[lag_start + i] - mu_lag);
    }
    #[allow(clippy::cast_precision_loss)]
    {
        gamma /= n_pairs as f64;
    }

    let rho = gamma / (std_ref * std_lag);
    rho.clamp(-1.0, 1.0)
}

/// The number of cycle boundaries between season `ref_season` and the season
/// `lag` steps before it.
fn lag_years_back(ref_season: usize, lag: usize, n_seasons: usize) -> usize {
    if lag > ref_season {
        (lag - ref_season).div_ceil(n_seasons)
    } else {
        0
    }
}

/// `(ref_offset, lag_offset)`: the leading entries the reference and lagged
/// buckets skip so that index `i` of each names years `years_back` apart.
// Rationale: each cast takes a non-negative difference of `i32` years.
#[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
fn bucket_offsets(ref_year_start: i32, lag_year_start: i32, years_back: i64) -> (usize, usize) {
    let shift = i64::from(ref_year_start) - i64::from(lag_year_start) - years_back;
    if shift >= 0 {
        (0, shift as usize)
    } else {
        ((-shift) as usize, 0)
    }
}

/// Build the **forward-prediction** periodic Yule-Walker matrix and RHS:
/// predict `z_m` from `{z_{m-1}, ..., z_{m-p}}`.
///
/// `R[i,j] = rho(season - (i+1), |j - i|)`, with the reference month shifting
/// per row — symmetric but NOT Toeplitz, since the autocorrelation varies with
/// the reference period. `rhs[i] = rho(season, i+1)`.
///
/// # Parameters
///
/// - `season` -- 0-based target season.
/// - `order` -- AR order; matrix dimension is `order x order`.
/// - `n_seasons` -- total number of seasons in the periodic cycle.
/// - `observations_by_season` -- chronological observations grouped by season.
/// - `stats_by_season` -- `(mean, std)` per season.
/// - `year_starts` -- first year of each season's bucket (see
///   [`periodic_autocorrelation`]).
///
/// # Returns
///
/// `(matrix, rhs)` with `matrix` flat row-major (`R[i][j]` at `i * order + j`),
/// or `(vec![], vec![])` when `order == 0`.
#[must_use]
pub fn build_periodic_yw_matrix(
    season: usize,
    order: usize,
    n_seasons: usize,
    observations_by_season: &[&[f64]],
    stats_by_season: &[(f64, f64)],
    year_starts: &[i32],
) -> (Vec<f64>, Vec<f64>) {
    #[cfg(test)]
    BUILD_PERIODIC_YW_MATRIX_CALL_COUNT.with(|c| {
        *c.borrow_mut() += 1;
    });

    if order == 0 {
        return (Vec::new(), Vec::new());
    }

    let mut matrix = vec![0.0_f64; order * order];
    let mut rhs = vec![0.0_f64; order];

    for i in 0..order {
        matrix[i * order + i] = 1.0;
        // `(i + 1) % n_seasons` prevents underflow when order > n_seasons.
        let ref_month = (season + n_seasons - (i + 1) % n_seasons) % n_seasons;
        for j in (i + 1)..order {
            let lag = j - i;
            let rho = periodic_autocorrelation(
                ref_month,
                lag,
                n_seasons,
                observations_by_season,
                stats_by_season,
                year_starts,
            );
            matrix[i * order + j] = rho;
            matrix[j * order + i] = rho;
        }
    }

    for (i, rhs_entry) in rhs.iter_mut().enumerate().take(order) {
        *rhs_entry = periodic_autocorrelation(
            season,
            i + 1,
            n_seasons,
            observations_by_season,
            stats_by_season,
            year_starts,
        );
    }

    (matrix, rhs)
}

/// Write the periodic Yule-Walker matrix and RHS into caller-supplied flat
/// buffers (resized to `order * order` and `order`), reusing allocations across
/// the increasing-dimension solves of `periodic_pacf`.
///
/// See [`build_periodic_yw_matrix`] for the matrix/RHS layout.
///
/// # Parameters
///
/// - `season` -- 0-based target season.
/// - `order` -- AR order; `matrix_out` holds `order * order` entries.
/// - `n_seasons` -- total number of seasons.
/// - `observations_by_season` -- observations grouped by season.
/// - `stats_by_season` -- `(mean, std)` per season.
/// - `year_starts` -- first year of each season's bucket (see
///   [`periodic_autocorrelation`]).
/// - `matrix_out` / `rhs_out` -- caller buffers; resized and overwritten.
///
/// No-op when `order == 0` (buffers cleared to empty).
pub fn build_periodic_yw_matrix_into(
    season: usize,
    order: usize,
    n_seasons: usize,
    observations_by_season: &[&[f64]],
    stats_by_season: &[(f64, f64)],
    year_starts: &[i32],
    matrix_out: &mut Vec<f64>,
    rhs_out: &mut Vec<f64>,
) {
    #[cfg(test)]
    BUILD_PERIODIC_YW_MATRIX_CALL_COUNT.with(|c| {
        *c.borrow_mut() += 1;
    });

    if order == 0 {
        matrix_out.clear();
        rhs_out.clear();
        return;
    }

    matrix_out.resize(order * order, 0.0_f64);
    // Must clear: resizing down from a larger prior order leaves stale leading
    // entries that resize alone keeps.
    matrix_out.fill(0.0);
    rhs_out.resize(order, 0.0_f64);

    for i in 0..order {
        matrix_out[i * order + i] = 1.0;
        // `(i + 1) % n_seasons` prevents underflow when order > n_seasons.
        let ref_month = (season + n_seasons - (i + 1) % n_seasons) % n_seasons;
        for j in (i + 1)..order {
            let lag = j - i;
            let rho = periodic_autocorrelation(
                ref_month,
                lag,
                n_seasons,
                observations_by_season,
                stats_by_season,
                year_starts,
            );
            matrix_out[i * order + j] = rho;
            matrix_out[j * order + i] = rho;
        }
    }

    for (i, rhs_entry) in rhs_out.iter_mut().enumerate().take(order) {
        *rhs_entry = periodic_autocorrelation(
            season,
            i + 1,
            n_seasons,
            observations_by_season,
            stats_by_season,
            year_starts,
        );
    }
}

// ---------------------------------------------------------------------------
// Extended periodic Yule-Walker matrix (annual component)
// ---------------------------------------------------------------------------

/// Compute `ρ_{Z,A}^{ref_season}(lag)` from the PAR-A extended Yule-Walker
/// system — the cross-correlation between annual component `A` at `ref_season`
/// and periodic series `Z` at `ref_season - lag` (wrapping), population divisor.
///
/// # Cross-year alignment
///
/// Two year offsets compose:
///
/// 1. **Bucket year offset**: `A` and `Z` buckets start at different PDF years
///    per season (the rolling 12-month `A` window needs a full year of
///    look-back), so pairing aligns by absolute PDF year via `z_year_starts` /
///    `a_year_starts`, not bucket index.
/// 2. **Lag year count**: `Z`'s PDF year is earlier than `A`'s by the number of
///    cycle boundaries the lag crosses, as in [`periodic_autocorrelation`].
///
/// # Parameters
///
/// - `ref_season` — 0-based season index for the `A` series.
/// - `lag` — non-negative lag; `lag = 0` pairs each `A_i` with `Z_i`.
/// - `n_seasons` — total number of seasons in the periodic cycle.
/// - `observations_by_season` — chronological `Z` observations grouped by season.
/// - `stats_by_season` — `(mean, std)` for each `Z` season.
/// - `z_year_starts` — first PDF year of each `Z` bucket; equal entries recover
///   the legacy by-index pairing.
/// - `annual_observations_by_season` — `A` observations grouped by season.
/// - `annual_stats_by_season` — `(mean, std)` for each `A` season.
/// - `a_year_starts` — first PDF year of each `A` bucket, indexed by season.
///
/// # Returns
///
/// The normalised cross-correlation clamped to `[-1.0, 1.0]`; `0.0` when either
/// std is below [`f64::EPSILON`] or insufficient paired observations exist.
#[must_use]
pub fn cross_correlation_z_a(
    ref_season: usize,
    lag: usize,
    n_seasons: usize,
    observations_by_season: &[&[f64]],
    stats_by_season: &[(f64, f64)],
    z_year_starts: &[i32],
    annual_observations_by_season: &[&[f64]],
    annual_stats_by_season: &[(f64, f64)],
    a_year_starts: &[i32],
) -> f64 {
    let (mu_a, std_a) = annual_stats_by_season[ref_season];
    let lag_season = (ref_season + n_seasons - lag % n_seasons) % n_seasons;
    let (mu_z, std_z) = stats_by_season[lag_season];

    // Zero std ⇒ cross-correlation undefined.
    if std_a.abs() < f64::EPSILON || std_z.abs() < f64::EPSILON {
        return 0.0;
    }

    let a_obs = annual_observations_by_season[ref_season];
    let z_obs = observations_by_season[lag_season];

    #[allow(clippy::cast_possible_wrap)]
    let (a_start, z_start) = bucket_offsets(
        a_year_starts[ref_season],
        z_year_starts[lag_season],
        lag_years_back(ref_season, lag, n_seasons) as i64,
    );

    let n_pairs = a_obs
        .len()
        .saturating_sub(a_start)
        .min(z_obs.len().saturating_sub(z_start));

    if n_pairs == 0 {
        return 0.0;
    }

    // Divide by the max bucket size, NOT n_pairs: it equals padding the missing-A
    // years with the sample mean (zero cross-product contribution), whereas
    // n_pairs would overstate ρ̂(Z, A) by max_len / n_pairs and tilt downstream
    // FACPs across the significance threshold.
    let mut gamma = 0.0_f64;
    for i in 0..n_pairs {
        gamma += (a_obs[a_start + i] - mu_a) * (z_obs[z_start + i] - mu_z);
    }
    let denom_n = a_obs.len().max(z_obs.len());
    #[allow(clippy::cast_precision_loss)]
    {
        gamma /= denom_n as f64;
    }

    let rho = gamma / (std_a * std_z);
    rho.clamp(-1.0, 1.0)
}

/// Compute `ρ_{Z,A}^{ref_season}(-1)`: the cross-correlation between annual
/// component `A` at `ref_season` and periodic series `Z` at the **next** season
/// `(ref_season + 1) % n_seasons`.
///
/// In the PAR-A extended Yule-Walker RHS this pairs `A_{t-1}` (season `m-1`)
/// with `Z_t` (season `m`), so `Z` is one step **ahead** of `A`.
///
/// # Cross-year alignment
///
/// Same two composing offsets as [`cross_correlation_z_a`] (bucket year offset
/// via `z_year_starts` / `a_year_starts`), except the lag wrap is **forward**:
/// when `z_season` wraps to 0, `Z` is in the next regression year.
///
/// # Parameters
///
/// - `ref_season` — 0-based season index for the `A` series.
/// - `n_seasons` — total number of seasons in the periodic cycle.
/// - `observations_by_season` — chronological `Z` observations grouped by season.
/// - `stats_by_season` — `(mean, std)` for each `Z` season.
/// - `z_year_starts` — first PDF year of each `Z` bucket, indexed by season.
/// - `annual_observations_by_season` — `A` observations grouped by season.
/// - `annual_stats_by_season` — `(mean, std)` for each `A` season.
/// - `a_year_starts` — first PDF year of each `A` bucket, indexed by season.
///
/// # Returns
///
/// The normalised cross-correlation clamped to `[-1.0, 1.0]`; `0.0` when either
/// std is below [`f64::EPSILON`] or insufficient paired observations exist.
#[must_use]
pub fn cross_correlation_a_z_neg1(
    ref_season: usize,
    n_seasons: usize,
    observations_by_season: &[&[f64]],
    stats_by_season: &[(f64, f64)],
    z_year_starts: &[i32],
    annual_observations_by_season: &[&[f64]],
    annual_stats_by_season: &[(f64, f64)],
    a_year_starts: &[i32],
) -> f64 {
    let (mu_a, std_a) = annual_stats_by_season[ref_season];
    let z_season = (ref_season + 1) % n_seasons;
    let (mu_z, std_z) = stats_by_season[z_season];

    // Zero std ⇒ cross-correlation undefined.
    if std_a.abs() < f64::EPSILON || std_z.abs() < f64::EPSILON {
        return 0.0;
    }

    let a_obs = annual_observations_by_season[ref_season];
    let z_obs = observations_by_season[z_season];

    let (a_start, z_start) = bucket_offsets(
        a_year_starts[ref_season],
        z_year_starts[z_season],
        -i64::from(z_season == 0),
    );

    let n_pairs = a_obs
        .len()
        .saturating_sub(a_start)
        .min(z_obs.len().saturating_sub(z_start));

    if n_pairs == 0 {
        return 0.0;
    }

    // Max-bucket-size divisor, not n_pairs — see [`cross_correlation_z_a`].
    let mut gamma = 0.0_f64;
    for i in 0..n_pairs {
        gamma += (a_obs[a_start + i] - mu_a) * (z_obs[z_start + i] - mu_z);
    }
    let denom_n = a_obs.len().max(z_obs.len());
    #[allow(clippy::cast_precision_loss)]
    {
        gamma /= denom_n as f64;
    }

    let rho = gamma / (std_a * std_z);
    rho.clamp(-1.0, 1.0)
}

/// Build the extended periodic Yule-Walker matrix and RHS for the PAR-A model,
/// augmenting the classical `order × order` system with one row/column for the
/// annual coefficient `ψ`:
///
/// ```text
/// [ classical_yw_matrix  |  cross_col ]
/// [ cross_row            |  1.0       ]
/// ```
///
/// The cross row/column are `cross_correlation_z_a(prev_season, i, …)`,
/// `rhs[0..order]` mirrors the classical RHS, and
/// `rhs[order] = cross_correlation_a_z_neg1(prev_season, …)`, where
/// `prev_season = (season + n_seasons − 1) % n_seasons`. The solution
/// `[φ_1, …, φ_p, ψ]` is **standardised**; the caller applies unit conversion.
///
/// # Parameters
///
/// - `season` — 0-based target season.
/// - `order` — AR order `p`; matrix dimension is `(order+1) × (order+1)`.
/// - `n_seasons` — total number of seasons in the periodic cycle.
/// - `observations_by_season` — chronological `Z` observations grouped by season.
/// - `stats_by_season` — `(mean, std)` for each `Z` season.
/// - `z_year_starts` — first PDF year of each `Z` bucket; aligns every pairing
///   with `Z` by absolute year (see [`periodic_autocorrelation`] and
///   [`cross_correlation_z_a`]).
/// - `annual_observations_by_season` — `A` observations grouped by season.
/// - `annual_stats_by_season` — `(mean, std)` for each `A` season.
/// - `a_year_starts` — first PDF year of each `A` bucket, indexed by season.
///
/// # Returns
///
/// `(matrix, rhs)` with `matrix` flat row-major (`R[i][j]` at `i * (order+1) + j`).
/// `order == 0` returns `(vec![1.0], vec![rhs_annual_neg1])` — a 1×1 system for `ψ`.
#[must_use]
pub fn build_extended_periodic_yw_matrix(
    season: usize,
    order: usize,
    n_seasons: usize,
    observations_by_season: &[&[f64]],
    stats_by_season: &[(f64, f64)],
    z_year_starts: &[i32],
    annual_observations_by_season: &[&[f64]],
    annual_stats_by_season: &[(f64, f64)],
    a_year_starts: &[i32],
) -> (Vec<f64>, Vec<f64>) {
    let dim = order + 1;
    let prev_season = (season + n_seasons - 1) % n_seasons;

    let mut matrix = vec![0.0_f64; dim * dim];
    let mut rhs = vec![0.0_f64; dim];

    for i in 0..order {
        matrix[i * dim + i] = 1.0;
        // `(i + 1) % n_seasons` prevents underflow when order > n_seasons.
        let ref_month = (season + n_seasons - (i + 1) % n_seasons) % n_seasons;
        for j in (i + 1)..order {
            let lag = j - i;
            let rho = periodic_autocorrelation(
                ref_month,
                lag,
                n_seasons,
                observations_by_season,
                stats_by_season,
                z_year_starts,
            );
            matrix[i * dim + j] = rho;
            matrix[j * dim + i] = rho;
        }
    }

    for (i, rhs_entry) in rhs.iter_mut().enumerate().take(order) {
        *rhs_entry = periodic_autocorrelation(
            season,
            i + 1,
            n_seasons,
            observations_by_season,
            stats_by_season,
            z_year_starts,
        );
    }

    for i in 0..order {
        let rho = cross_correlation_z_a(
            prev_season,
            i,
            n_seasons,
            observations_by_season,
            stats_by_season,
            z_year_starts,
            annual_observations_by_season,
            annual_stats_by_season,
            a_year_starts,
        );
        matrix[i * dim + order] = rho;
        matrix[order * dim + i] = rho;
    }

    matrix[order * dim + order] = 1.0;

    rhs[order] = cross_correlation_a_z_neg1(
        prev_season,
        n_seasons,
        observations_by_season,
        stats_by_season,
        z_year_starts,
        annual_observations_by_season,
        annual_stats_by_season,
        a_year_starts,
    );

    (matrix, rhs)
}

/// Solve a dense linear system `A * x = b` via Gaussian elimination with
/// partial pivoting.
///
/// Sized for the small (n <= 10) Yule-Walker systems of PAR fitting, where the
/// O(n³) cost is negligible.
///
/// # Parameters
///
/// - `a` -- flat row-major `n x n` matrix (length `n * n`), modified in place.
/// - `b` -- right-hand side of length `n`, modified in place.
/// - `n` -- system dimension.
///
/// # Returns
///
/// `Some(x)` of length `n`, or `None` when the matrix is singular (pivot
/// magnitude below `f64::EPSILON`).
pub fn solve_linear_system(a: &mut [f64], b: &mut [f64], n: usize) -> Option<Vec<f64>> {
    debug_assert_eq!(a.len(), n * n, "matrix must have n*n elements");
    debug_assert_eq!(b.len(), n, "rhs must have n elements");

    if n == 0 {
        return Some(Vec::new());
    }

    for k in 0..n {
        let mut max_val = a[k * n + k].abs();
        let mut max_row = k;
        for row in (k + 1)..n {
            let val = a[row * n + k].abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }

        if max_val < f64::EPSILON {
            return None;
        }

        if max_row != k {
            for col in 0..n {
                a.swap(k * n + col, max_row * n + col);
            }
            b.swap(k, max_row);
        }

        let pivot = a[k * n + k];
        for i in (k + 1)..n {
            let factor = a[i * n + k] / pivot;
            a[i * n + k] = 0.0;
            for j in (k + 1)..n {
                a[i * n + j] -= factor * a[k * n + j];
            }
            b[i] -= factor * b[k];
        }
    }

    let mut x = vec![0.0_f64; n];
    for k in (0..n).rev() {
        let mut sum = b[k];
        for j in (k + 1)..n {
            sum -= a[k * n + j] * x[j];
        }
        x[k] = sum / a[k * n + k];
    }

    Some(x)
}

// Counts build_periodic_yw_matrix calls for
// `estimate_periodic_ar_coefficients_calls_build_once_per_order` (one per order, not two).
#[cfg(test)]
thread_local! {
    pub(super) static BUILD_PERIODIC_YW_MATRIX_CALL_COUNT: std::cell::RefCell<usize> =
        const { std::cell::RefCell::new(0) };
}
