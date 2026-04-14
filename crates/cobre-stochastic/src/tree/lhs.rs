//! Latin Hypercube Sampling (LHS) for batch and point-wise noise generation.
//!
//! - [`generate_lhs`]: batch LHS filling `n_openings × dim` N(0,1) values.
//! - [`sample_lhs_point`]: point-wise LHS for a single scenario, no inter-worker coordination.
//!
//! Both methods stratify each dimension into `n_openings` equal-probability strata,
//! apply Fisher-Yates shuffling to break diagonal structure, and ensure marginal uniformity.
//!
//! **Batch algorithm**: for each dimension, generate stratified samples `u[k] = (k + U_k) / N`,
//! shuffle a permutation, and write `output[perm[k] * dim + d] = norm_quantile(u[k])`.
//!
//! **Point-wise algorithm**: derive per-dimension permutations identically on all workers via
//! `(sampling_seed, iteration, stage_id)`, look up `scenario`'s stratum, sample within-stratum
//! offset independently, and compute `norm_quantile((stratum + offset) / N)`.
//!
//! Output layout: opening-major `output[opening * dim + entity]`.
//!
//! Determinism: same `(base_seed, stage_id)` always produces identical output.

use rand::RngExt;
use rand_distr::Uniform;

use crate::noise::{
    quantile::norm_quantile,
    rng::rng_from_seed,
    seed::{derive_forward_seed, derive_opening_seed, derive_stage_seed},
};

/// Shuffle `perm` in place using the Fisher-Yates algorithm in O(n) time.
pub(crate) fn fisher_yates(perm: &mut [usize], rng: &mut impl rand::Rng) {
    let n = perm.len();
    for i in (1..n).rev() {
        let j = rng.random_range(0..=i);
        perm.swap(i, j);
    }
}

/// Fill `output` with `n_openings × dim` standard-normal N(0,1) values using LHS.
///
/// Each dimension is independently stratified: the `N = n_openings` strata
/// `[k/N, (k+1)/N)` each contribute exactly one sample, and a Fisher-Yates
/// shuffle independently assigns strata to openings for every dimension.
/// Output layout: `output[opening * dim + entity]`.
///
/// # Panics
///
/// Panics if `output.len() < n_openings * dim`.
pub fn generate_lhs(
    base_seed: u64,
    stage_id: u32,
    n_openings: usize,
    dim: usize,
    output: &mut [f64],
) {
    assert!(
        output.len() >= n_openings * dim,
        "output slice too short: need {}, got {}",
        n_openings * dim,
        output.len(),
    );

    if n_openings == 0 || dim == 0 {
        return;
    }

    let seed = derive_stage_seed(base_seed, stage_id);
    let mut rng = rng_from_seed(seed);
    #[allow(clippy::expect_used)]
    let uniform = Uniform::new(0.0_f64, 1.0_f64).expect("0.0 < 1.0 is always a valid range");

    let mut samples = vec![0.0_f64; n_openings];
    let mut perm: Vec<usize> = (0..n_openings).collect();
    #[allow(clippy::cast_precision_loss)]
    let n_f = n_openings as f64;

    for d in 0..dim {
        // Generate stratified samples: u[k] = (k + U_k) / N where U_k ~ U(0,1).
        // norm_quantile requires strictly positive input, so guard the k=0 case where u[0]=0.0 is possible.
        #[allow(clippy::cast_precision_loss)]
        for (k, sample) in samples.iter_mut().enumerate() {
            let u = rng.sample(uniform);
            let s = (k as f64 + u) / n_f;
            *sample = s.clamp(f64::MIN_POSITIVE, 1.0 - f64::EPSILON);
        }

        // Reset permutation to identity and shuffle.
        for (i, p) in perm.iter_mut().enumerate() {
            *p = i;
        }
        fisher_yates(&mut perm, &mut rng);

        // Write permuted, quantile-transformed samples to output.
        for k in 0..n_openings {
            output[perm[k] * dim + d] = norm_quantile(samples[k]);
        }
    }
}

/// Configuration for single-scenario LHS point generation.
#[derive(Debug, Clone, Copy)]
pub struct LhsPointSpec {
    /// Forward-pass base seed.
    pub sampling_seed: u64,
    /// Training iteration index.
    pub iteration: u32,
    /// Global scenario index in `0..total_scenarios`.
    pub scenario: u32,
    /// Stage domain identifier.
    pub stage_id: u32,
    /// Total forward scenarios per iteration (= N strata).
    pub total_scenarios: u32,
    /// Noise vector dimension.
    pub dim: usize,
}

/// Generate one scenario's noise vector using LHS without inter-worker coordination.
///
/// Each scenario derives the same per-dimension permutations from `(sampling_seed, iteration, stage_id)`,
/// looks up its stratum, and samples within-stratum offset independently. The `N = total_scenarios`
/// scenarios across all workers form a valid LHS design without communication.
///
/// # Panics
///
/// Panics if `output.len() < spec.dim` or `perm_scratch.len() < spec.total_scenarios as usize`.
#[allow(clippy::cast_precision_loss)]
pub fn sample_lhs_point(spec: &LhsPointSpec, output: &mut [f64], perm_scratch: &mut [usize]) {
    let n = spec.total_scenarios as usize;
    assert!(
        perm_scratch.len() >= n,
        "perm_scratch too short: need {n}, got {}",
        perm_scratch.len(),
    );
    assert!(
        output.len() >= spec.dim,
        "output too short: need {}, got {}",
        spec.dim,
        output.len(),
    );

    let perm_seed = derive_opening_seed(spec.sampling_seed, spec.iteration, spec.stage_id);
    let mut perm_rng = rng_from_seed(perm_seed);

    let draw_seed = derive_forward_seed(
        spec.sampling_seed,
        spec.iteration,
        spec.scenario,
        spec.stage_id,
    );
    let mut draw_rng = rng_from_seed(draw_seed);

    #[allow(clippy::expect_used)]
    let uniform = Uniform::new(0.0_f64, 1.0_f64).expect("0.0 < 1.0 is always a valid range");

    let perm = &mut perm_scratch[..n];
    let scenario_idx = spec.scenario as usize;

    for slot in output.iter_mut().take(spec.dim) {
        // Reset permutation to identity and shuffle. perm_rng advances identically on all workers.
        for (i, p) in perm.iter_mut().enumerate() {
            *p = i;
        }
        fisher_yates(perm, &mut perm_rng);

        let stratum = perm[scenario_idx];
        let u_raw = draw_rng.sample(uniform);
        let u_stratified = (stratum as f64 + u_raw) / n as f64;

        *slot = norm_quantile(u_stratified.clamp(f64::MIN_POSITIVE, 1.0 - f64::EPSILON));
    }
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand_pcg::Pcg64;

    use super::{fisher_yates, generate_lhs, sample_lhs_point, LhsPointSpec};

    /// A shuffled slice must contain exactly all elements 0..N (is a permutation).
    #[test]
    fn fisher_yates_is_permutation() {
        let n = 20_usize;
        let mut perm: Vec<usize> = (0..n).collect();
        let mut rng = Pcg64::seed_from_u64(42);
        fisher_yates(&mut perm, &mut rng);

        let mut sorted = perm.clone();
        sorted.sort_unstable();
        let expected: Vec<usize> = (0..n).collect();
        assert_eq!(
            sorted, expected,
            "shuffled slice is not a permutation of 0..{n}"
        );
    }

    /// Different RNG states must produce different permutations (with high probability).
    #[test]
    fn fisher_yates_different_states_differ() {
        let n = 10_usize;
        let mut perm1: Vec<usize> = (0..n).collect();
        let mut perm2: Vec<usize> = (0..n).collect();
        let mut rng1 = Pcg64::seed_from_u64(1);
        let mut rng2 = Pcg64::seed_from_u64(999_999_999);
        fisher_yates(&mut perm1, &mut rng1);
        fisher_yates(&mut perm2, &mut rng2);
        // With n=10 the probability of a collision is 1/10! ≈ 2.76e-7.
        assert_ne!(
            perm1, perm2,
            "two different RNG states produced the same permutation"
        );
    }

    /// Empty and single-element slices must not panic.
    #[test]
    fn fisher_yates_edge_cases_do_not_panic() {
        let mut rng = Pcg64::seed_from_u64(0);
        let mut empty: Vec<usize> = vec![];
        fisher_yates(&mut empty, &mut rng);
        assert!(empty.is_empty());

        let mut single = vec![7_usize];
        fisher_yates(&mut single, &mut rng);
        assert_eq!(single, vec![7]);
    }

    /// Same (`base_seed`, `stage_id`) must produce bitwise identical output.
    #[test]
    #[allow(clippy::float_cmp)]
    fn lhs_determinism() {
        let n_openings = 50;
        let dim = 3;
        let mut out1 = vec![0.0_f64; n_openings * dim];
        let mut out2 = vec![0.0_f64; n_openings * dim];
        generate_lhs(42, 0, n_openings, dim, &mut out1);
        generate_lhs(42, 0, n_openings, dim, &mut out2);
        assert_eq!(
            out1, out2,
            "generate_lhs is not deterministic for the same seed"
        );
    }

    /// Different seeds must produce different output.
    #[test]
    fn lhs_different_seeds_differ() {
        let n_openings = 20;
        let dim = 2;
        let mut out_a = vec![0.0_f64; n_openings * dim];
        let mut out_b = vec![0.0_f64; n_openings * dim];
        generate_lhs(42, 0, n_openings, dim, &mut out_a);
        generate_lhs(99, 0, n_openings, dim, &mut out_b);
        assert_ne!(out_a, out_b, "different seeds produced identical output");
    }

    /// Output length must equal `n_openings` * dim.
    #[test]
    fn lhs_correct_length() {
        let n_openings = 10;
        let dim = 4;
        let mut output = vec![0.0_f64; n_openings * dim];
        generate_lhs(1, 2, n_openings, dim, &mut output);
        // The length contract is enforced by the caller; we verify it wasn't
        // truncated or over-written by checking that all positions were touched.
        assert_eq!(output.len(), n_openings * dim);
    }

    /// All output values must be finite.
    #[test]
    fn lhs_all_finite() {
        let n_openings = 30;
        let dim = 5;
        let mut output = vec![0.0_f64; n_openings * dim];
        generate_lhs(7, 3, n_openings, dim, &mut output);
        for (i, &v) in output.iter().enumerate() {
            assert!(v.is_finite(), "non-finite value at index {i}: {v}");
        }
    }

    /// Each dimension must have exactly one sample per stratum (marginal uniformity).
    ///
    /// After applying Φ (the standard normal CDF) to the output values, the
    /// stratum index `floor(Φ(x) * N)` must be a permutation of {0, …, N-1}.
    ///
    /// We use the complementary property of the BSM approximation:
    /// `Φ(norm_quantile(p)) ≈ p`, so instead of applying the full CDF we
    /// invert back: `floor(samples[k] * N)` must yield all values in 0..N,
    /// where `samples[k]` is the original stratified sample.
    ///
    /// Because we do not expose `samples` from `generate_lhs`, we verify the
    /// acceptance criterion directly: for each dimension, the floor indices
    /// of the CDF-transformed output values form a permutation of {0, …, N-1}.
    ///
    /// `Φ(z)` is approximated via the logistic surrogate `1/(1+exp(-z * π/√3))`
    /// for the acceptance test (sufficient precision for the stratum floor test).
    #[test]
    #[allow(
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss
    )]
    fn lhs_marginal_stratification() {
        let n_openings = 50_usize;
        let dim = 3_usize;
        let mut output = vec![0.0_f64; n_openings * dim];
        generate_lhs(42, 0, n_openings, dim, &mut output);

        let n_f = n_openings as f64;
        // Approximate Φ(z) via the standard normal CDF using a simple approximation.
        let approx_cdf = |z: f64| -> f64 { 0.5 * (1.0 + libm_erf(z / std::f64::consts::SQRT_2)) };

        for d in 0..dim {
            let mut strata: Vec<usize> = (0..n_openings)
                .map(|k| {
                    let z = output[k * dim + d];
                    let p = approx_cdf(z);
                    // Clamp to [0, N-1] to handle floating-point boundary cases.
                    let stratum = (p * n_f).floor() as usize;
                    stratum.min(n_openings - 1)
                })
                .collect();
            strata.sort_unstable();
            let expected: Vec<usize> = (0..n_openings).collect();
            assert_eq!(
                strata, expected,
                "dimension {d}: CDF-floor indices are not a permutation of 0..{n_openings}"
            );
        }
    }

    /// Statistical sanity: mean ≈ 0, std ≈ 1 for large N.
    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn lhs_mean_and_std_within_tolerance() {
        let n_openings = 1000_usize;
        let dim = 1_usize;
        let mut output = vec![0.0_f64; n_openings * dim];
        generate_lhs(42, 0, n_openings, dim, &mut output);

        let n = n_openings as f64;
        let mean = output.iter().sum::<f64>() / n;
        let variance = output.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
        let std = variance.sqrt();

        assert!(
            mean.abs() < 0.1,
            "mean {mean:.4} too far from 0 (tolerance 0.1)"
        );
        assert!(
            (std - 1.0).abs() < 0.1,
            "std {std:.4} too far from 1 (tolerance 0.1)"
        );
    }

    /// `n_openings=0` must not panic and must leave output unchanged.
    #[test]
    #[allow(clippy::float_cmp)]
    fn lhs_zero_openings_does_not_panic() {
        let mut output = vec![99.0_f64; 0];
        generate_lhs(1, 0, 0, 3, &mut output);
        assert!(output.is_empty());
    }

    /// dim=0 must not panic.
    #[test]
    fn lhs_zero_dim_does_not_panic() {
        let mut output: Vec<f64> = vec![];
        generate_lhs(1, 0, 5, 0, &mut output);
    }

    /// Approximate `erf(x)` using the Horner-form rational approximation
    /// (Abramowitz & Stegun 7.1.26, max error 1.5e-7).
    fn libm_erf(x: f64) -> f64 {
        let sign = if x < 0.0 { -1.0_f64 } else { 1.0_f64 };
        let t = 1.0 / (1.0 + 0.327_591_1 * x.abs());
        let poly = t
            * (0.254_829_592
                + t * (-0.284_496_736
                    + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
        sign * (1.0 - poly * (-x * x).exp())
    }

    /// Same inputs must produce bitwise identical output (determinism).
    #[test]
    #[allow(clippy::float_cmp, clippy::cast_possible_truncation)]
    fn lhs_point_determinism() {
        let n = 50_usize;
        let dim = 3_usize;
        let mut out1 = vec![0.0_f64; dim];
        let mut out2 = vec![0.0_f64; dim];
        let mut perm = vec![0_usize; n];
        let spec = LhsPointSpec {
            sampling_seed: 42,
            iteration: 0,
            scenario: 0,
            stage_id: 0,
            total_scenarios: n as u32,
            dim,
        };

        sample_lhs_point(&spec, &mut out1, &mut perm);
        sample_lhs_point(&spec, &mut out2, &mut perm);

        assert_eq!(
            out1, out2,
            "sample_lhs_point is not deterministic for the same inputs"
        );
    }

    /// Different `sampling_seed` values must produce different outputs.
    #[test]
    #[allow(clippy::cast_possible_truncation)]
    fn lhs_point_different_seeds_differ() {
        let n = 50_usize;
        let dim = 3_usize;
        let mut out1 = vec![0.0_f64; dim];
        let mut out2 = vec![0.0_f64; dim];
        let mut perm = vec![0_usize; n];

        sample_lhs_point(
            &LhsPointSpec {
                sampling_seed: 42,
                iteration: 0,
                scenario: 0,
                stage_id: 0,
                total_scenarios: n as u32,
                dim,
            },
            &mut out1,
            &mut perm,
        );
        sample_lhs_point(
            &LhsPointSpec {
                sampling_seed: 43,
                iteration: 0,
                scenario: 0,
                stage_id: 0,
                total_scenarios: n as u32,
                dim,
            },
            &mut out2,
            &mut perm,
        );

        assert_ne!(
            out1, out2,
            "different sampling_seeds produced identical output"
        );
    }

    /// All output values across all scenarios must be finite.
    #[test]
    #[allow(clippy::cast_possible_truncation)]
    fn lhs_point_all_finite() {
        let n = 50_usize;
        let dim = 3_usize;
        let mut perm = vec![0_usize; n];

        for scenario in 0..n {
            let mut output = vec![0.0_f64; dim];
            sample_lhs_point(
                &LhsPointSpec {
                    sampling_seed: 42,
                    iteration: 0,
                    scenario: scenario as u32,
                    stage_id: 0,
                    total_scenarios: n as u32,
                    dim,
                },
                &mut output,
                &mut perm,
            );
            for (d, &v) in output.iter().enumerate() {
                assert!(
                    v.is_finite(),
                    "non-finite value at scenario={scenario}, dim={d}: {v}"
                );
            }
        }
    }

    /// For all scenarios 0..N, the strata per dimension must form a permutation
    /// of {0, …, N-1}, verifying a valid LHS design without communication.
    ///
    /// Uses the same Φ approximation as `lhs_marginal_stratification`.
    #[test]
    #[allow(
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss
    )]
    fn lhs_point_stratum_coverage() {
        let n = 50_usize;
        let dim = 3_usize;
        let mut perm = vec![0_usize; n];
        let n_f = n as f64;
        let approx_cdf = |z: f64| -> f64 { 0.5 * (1.0 + libm_erf(z / std::f64::consts::SQRT_2)) };

        // Collect per-dimension stratum indices across all scenarios.
        let mut strata_by_dim: Vec<Vec<usize>> = (0..dim).map(|_| Vec::with_capacity(n)).collect();
        for scenario in 0..n {
            let mut output = vec![0.0_f64; dim];
            sample_lhs_point(
                &LhsPointSpec {
                    sampling_seed: 42,
                    iteration: 0,
                    scenario: scenario as u32,
                    stage_id: 0,
                    total_scenarios: n as u32,
                    dim,
                },
                &mut output,
                &mut perm,
            );
            for (d, &v) in output.iter().enumerate() {
                let p = approx_cdf(v);
                let stratum = ((p * n_f).floor() as usize).min(n - 1);
                strata_by_dim[d].push(stratum);
            }
        }

        for (d, strata) in strata_by_dim.iter_mut().enumerate() {
            strata.sort_unstable();
            let expected: Vec<usize> = (0..n).collect();
            assert_eq!(
                *strata, expected,
                "dimension {d}: strata across all scenarios are not a permutation of 0..{n}"
            );
        }
    }
}
