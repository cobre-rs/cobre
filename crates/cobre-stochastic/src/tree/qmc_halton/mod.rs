//! Halton quasi-Monte Carlo sequence building blocks and generators.
//!
//! The Halton sequence is a low-discrepancy sequence that assigns each
//! dimension a distinct prime base. Dimension `d` (1-indexed) uses the
//! `d`-th prime: 2, 3, 5, 7, 11, … The coordinate of point `n` in
//! dimension `d` is `radical_inverse(n, p_d)`.
//!
//! ## Dimension numbering
//!
//! - Dimension 1 uses base 2 (van der Corput sequence).
//! - Dimension 2 uses base 3.
//! - Dimension `d` uses `sieve_primes(d)[d-1]`.
//!
//! ## Entry points
//!
//! - `sieve_primes`: compute the first `count` primes at generator
//!   initialisation time using the sieve of Eratosthenes.
//! - `radical_inverse`: compute the base-`b` radical inverse of integer
//!   `n` (pure digit-reflection, no scrambling).
//! - [`generate_qmc_halton`]: batch generation for all openings of a stage,
//!   with Owen-style random digit scrambling for decorrelation.
//! - [`scrambled_halton_point`]: single-scenario point-wise generation for
//!   the out-of-sample forward pass, independent of all other scenarios.
//!
//! ## Scrambling
//!
//! The plain Halton sequence suffers from correlation artifacts in high
//! dimensions (the "Halton curse"). Owen-style random digit scrambling
//! breaks these correlations by applying a random permutation to each
//! digit position in each dimension. For dimension `d` with prime base
//! `p_d` and digit position `j`, a permutation table `pi[d][j]` of size
//! `p_d` is applied: `scrambled_digit = pi[d][j][original_digit]`.
//! Permutation tables are derived deterministically from the stage seed,
//! ensuring reproducibility.

use crate::noise::{
    quantile::norm_quantile,
    rng::rng_from_seed,
    seed::{derive_opening_seed, derive_stage_seed},
};

use super::lhs::fisher_yates;

/// Return the first `count` prime numbers in ascending order.
///
/// Uses the sieve of Eratosthenes. For `count == 0`, returns an empty
/// vector. This function runs once per generator initialisation and is
/// never called on a hot path.
///
/// # Examples
///
/// ```no_run
/// // sieve_primes(5) returns [2, 3, 5, 7, 11]
/// ```
#[must_use]
pub(crate) fn sieve_primes(count: usize) -> Vec<u32> {
    if count == 0 {
        return Vec::new();
    }

    // Upper bound for the n-th prime via the prime number theorem:
    //   p_n < n * (ln(n) + ln(ln(n))) + 2  for n >= 6.
    // For small n we use a hard floor of 30 to avoid negative or zero bounds.
    let upper_bound: usize = if count < 6 {
        30
    } else {
        // Precision loss is acceptable: this is an upper-bound estimate.
        // The mantissa of f64 (52 bits) is sufficient for sieve sizes that
        // fit in practical memory. count values near 2^52 are unreachable.
        #[allow(clippy::cast_precision_loss)]
        let n = count as f64;
        let ln_n = n.ln();
        let ln_ln_n = ln_n.ln();
        // The expression is always positive and finite for n >= 6 (ln(ln(6)) > 0).
        // Truncation to usize is safe: the result is a small positive integer.
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let bound = (n * (ln_n + ln_ln_n) + 4.0) as usize + 4;
        bound
    };

    // Boolean sieve: `composite[i]` is true if `i` is not prime.
    let mut composite = vec![false; upper_bound + 1];
    composite[0] = true;
    composite[1] = true;

    let mut p = 2usize;
    while p * p <= upper_bound {
        if !composite[p] {
            let mut multiple = p * p;
            while multiple <= upper_bound {
                composite[multiple] = true;
                multiple += p;
            }
        }
        p += 1;
    }

    // Sieve indices are bounded by upper_bound, which the prime number theorem
    // guarantees holds all `count` primes; all such primes fit in u32 for any
    // practical dimension count (the 1,000,000th prime is 15,485,863).
    #[allow(clippy::cast_possible_truncation)]
    composite
        .iter()
        .enumerate()
        .filter(|&(_, &is_composite)| !is_composite)
        .map(|(i, _)| i as u32)
        .take(count)
        .collect()
}

/// Compute the base-`b` radical inverse of integer `n`.
///
/// The radical inverse reflects the base-`b` digits of `n` about the
/// decimal point. If `n = d_k d_{k-1} … d_1 d_0` in base `b`, then
/// `radical_inverse(n, b) = 0.d_0 d_1 … d_{k-1} d_k` as a floating-point
/// number.
///
/// Returns `0.0` for `n == 0`. Returns a value in `[0.0, 1.0)` for all
/// valid inputs.
///
/// # Preconditions
///
/// `base` must be `>= 2`. In debug builds a `debug_assert!` enforces this.
/// In release mode, `base < 2` causes the loop to not execute and returns
/// `0.0` without panicking.
///
/// # Examples
///
/// ```no_run
/// // radical_inverse(0, 2) == 0.0
/// // radical_inverse(1, 2) == 0.5   (binary 1 -> 0.1 = 0.5)
/// // radical_inverse(5, 2) == 0.625 (binary 101 -> 0.101 = 5/8)
/// ```
// `radical_inverse` is a module primitive used in unit tests and available
// for downstream integration. It is not yet called from production code outside
// `#[cfg(test)]` because the generators use `scrambled_radical_inverse`.
#[allow(dead_code)]
#[must_use]
pub(crate) fn radical_inverse(n: u32, base: u32) -> f64 {
    debug_assert!(base >= 2, "radical_inverse requires base >= 2, got {base}");

    let mut result = 0.0_f64;
    let mut inv_base = 1.0 / f64::from(base);
    let mut n = n;

    while n > 0 {
        let digit = n % base;
        result += f64::from(digit) * inv_base;
        n /= base;
        inv_base /= f64::from(base);
    }

    result
}

/// Build Owen-style random digit scramble tables for all dimensions.
///
/// Returns `tables[d][j][digit]` — for dimension `d` with prime base
/// `primes[d]`, digit position `j` (0-indexed from least significant),
/// the scrambled value of `digit` in `0..primes[d]`.
///
/// The number of digit positions for dimension `d` is
/// `ceil(log_{p_d}(max_n))` with a minimum of 1.
///
/// All permutations are generated from a single RNG seeded with `seed`,
/// advancing sequentially across all dimensions and digit positions to
/// ensure independence between dimensions.
fn build_scramble_tables(seed: u64, primes: &[u32], max_n: usize) -> Vec<Vec<Vec<u32>>> {
    let mut rng = rng_from_seed(seed);
    let mut tables = Vec::with_capacity(primes.len());

    for &base in primes {
        // Compute the number of digit positions needed to represent max_n in base p.
        // max_digits = max(1, ceil(log_base(max_n))).
        let max_digits = if max_n <= 1 {
            1
        } else {
            // Count digits by repeatedly dividing: how many times can we divide
            // (max_n - 1) by base before reaching 0? That equals floor(log_base(max_n - 1)) + 1.
            let mut digits = 0usize;
            let mut val = max_n - 1;
            while val > 0 {
                val /= base as usize;
                digits += 1;
            }
            digits.max(1)
        };

        // For each digit position, generate a random permutation of 0..base.
        let base_usize = base as usize;
        let mut dim_table: Vec<Vec<u32>> = Vec::with_capacity(max_digits);
        let mut work: Vec<usize> = (0..base_usize).collect();

        for _ in 0..max_digits {
            // Reset work buffer to identity permutation.
            for (i, slot) in work.iter_mut().enumerate() {
                *slot = i;
            }
            fisher_yates(&mut work, &mut rng);
            // Cast from usize to u32 is safe: values are in 0..base where base is u32.
            #[allow(clippy::cast_possible_truncation)]
            let perm: Vec<u32> = work.iter().map(|&v| v as u32).collect();
            dim_table.push(perm);
        }

        tables.push(dim_table);
    }

    tables
}

/// Compute the scrambled base-`base` radical inverse of integer `n`.
///
/// Applies the Owen-style random digit permutation `perm_table[j][digit]`
/// at each digit position `j` before accumulating the radical inverse.
/// Returns a value in `[0.0, 1.0)`.
fn scrambled_radical_inverse(n: u32, base: u32, perm_table: &[Vec<u32>]) -> f64 {
    debug_assert!(base >= 2, "base must be >= 2, got {base}");

    let mut result = 0.0_f64;
    let mut inv_base = 1.0 / f64::from(base);
    let mut n = n;
    let mut j = 0usize;

    while n > 0 {
        let digit = n % base;
        // Apply the permutation for digit position j if available; fall back to
        // identity if the table has fewer positions than the number of digits in n.
        let scrambled = if j < perm_table.len() {
            perm_table[j][digit as usize]
        } else {
            digit
        };
        result += f64::from(scrambled) * inv_base;
        n /= base;
        inv_base /= f64::from(base);
        j += 1;
    }

    result
}

/// Fill `output` with `n_openings × dim` standard-normal N(0,1) values
/// using scrambled Halton QMC with Owen-style random digit permutations.
///
/// Output layout: opening-major `output[opening * dim + entity]`.
///
/// Each dimension uses a different prime base and the same number of
/// scrambling tables built once from the stage-derived seed. Different
/// stages produce different scrambling and therefore different samples.
///
/// # Panics
///
/// Panics if `output.len() < n_openings * dim`.
pub fn generate_qmc_halton(
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
    let primes = sieve_primes(dim);
    let tables = build_scramble_tables(seed, &primes, n_openings);

    for n in 0..n_openings {
        // Cast to u32 is safe in practice: opening counts never exceed u32::MAX.
        #[allow(clippy::cast_possible_truncation)]
        let n_u32 = n as u32;
        for d in 0..dim {
            let u = scrambled_radical_inverse(n_u32, primes[d], &tables[d]);
            let u = u.clamp(f64::MIN_POSITIVE, 1.0 - f64::EPSILON);
            output[n * dim + d] = norm_quantile(u);
        }
    }
}

/// Configuration for single-scenario Halton point generation.
///
/// Bundles the parameters needed by [`scrambled_halton_point`] to generate
/// one scenario's noise vector without inter-worker coordination.
#[derive(Debug, Clone, Copy)]
pub struct HaltonPointSpec {
    /// Forward-pass base seed.
    pub sampling_seed: u64,
    /// Training iteration index.
    pub iteration: u32,
    /// Global scenario index in `0..total_scenarios`.
    pub scenario: u32,
    /// Stage domain identifier.
    pub stage_id: u32,
    /// Total forward scenarios per iteration.
    pub total_scenarios: u32,
    /// Noise vector dimension.
    pub dim: usize,
}

/// Generate one scenario's noise vector using scrambled Halton QMC,
/// independent of all other scenarios.
///
/// The scramble tables are derived from `(sampling_seed, iteration, stage_id)`
/// and are identical for all scenarios in the same iteration and stage.
/// Each scenario uses its own Halton index `spec.scenario`.
///
/// # Panics
///
/// Panics if `output.len() < spec.dim`.
pub fn scrambled_halton_point(spec: &HaltonPointSpec, output: &mut [f64]) {
    assert!(
        output.len() >= spec.dim,
        "output too short: need {}, got {}",
        spec.dim,
        output.len(),
    );

    debug_assert!(
        spec.scenario < spec.total_scenarios,
        "scenario {} out of range 0..{}",
        spec.scenario,
        spec.total_scenarios,
    );

    if spec.dim == 0 {
        return;
    }

    let seed = derive_opening_seed(spec.sampling_seed, spec.iteration, spec.stage_id);
    let primes = sieve_primes(spec.dim);
    let tables = build_scramble_tables(seed, &primes, spec.total_scenarios as usize);

    for d in 0..spec.dim {
        let u = scrambled_radical_inverse(spec.scenario, primes[d], &tables[d]);
        let u = u.clamp(f64::MIN_POSITIVE, 1.0 - f64::EPSILON);
        output[d] = norm_quantile(u);
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::float_cmp,
    clippy::panic
)]
mod tests {
    use super::{
        HaltonPointSpec, generate_qmc_halton, radical_inverse, scrambled_halton_point, sieve_primes,
    };

    #[test]
    fn test_sieve_first_10_primes() {
        assert_eq!(sieve_primes(10), vec![2, 3, 5, 7, 11, 13, 17, 19, 23, 29]);
    }

    #[test]
    fn test_sieve_zero_returns_empty() {
        assert_eq!(sieve_primes(0), Vec::<u32>::new());
    }

    #[test]
    fn test_sieve_one_returns_two() {
        assert_eq!(sieve_primes(1), vec![2]);
    }

    #[test]
    fn test_sieve_100_primes_count() {
        let primes = sieve_primes(100);
        assert_eq!(primes.len(), 100);
        assert_eq!(*primes.last().unwrap(), 541, "100th prime must be 541");
    }

    #[test]
    fn test_radical_inverse_base2_known_values() {
        // Van der Corput sequence in base 2: n -> radical_inverse(n, 2).
        let expected = [
            (0_u32, 0.0_f64),
            (1, 0.5),
            (2, 0.25),
            (3, 0.75),
            (4, 0.125),
            (5, 0.625),
            (6, 0.375),
            (7, 0.875),
        ];
        for (n, exp) in expected {
            let got = radical_inverse(n, 2);
            assert!(
                (got - exp).abs() < 1e-15,
                "radical_inverse({n}, 2): got {got}, expected {exp}"
            );
        }
    }

    #[test]
    fn test_radical_inverse_base3_known_values() {
        // Base-3 radical inverse first 4 values.
        // n=0 -> 0.0
        // n=1 -> 1/3
        // n=2 -> 2/3
        // n=3 -> ternary "10" -> 0.01 in ternary = 1/9
        let expected = [
            (0_u32, 0.0_f64),
            (1, 1.0 / 3.0),
            (2, 2.0 / 3.0),
            (3, 1.0 / 9.0),
        ];
        for (n, exp) in expected {
            let got = radical_inverse(n, 3);
            assert!(
                (got - exp).abs() < 1e-15,
                "radical_inverse({n}, 3): got {got}, expected {exp}"
            );
        }
    }

    #[test]
    fn test_radical_inverse_base5_n1() {
        // radical_inverse(1, 5) = 1/5 = 0.2
        let got = radical_inverse(1, 5);
        assert!(
            (got - 0.2_f64).abs() < 1e-15,
            "radical_inverse(1, 5): got {got}, expected 0.2"
        );
    }

    #[test]
    fn test_radical_inverse_range() {
        // For bases 2, 3, 5, 7 and n in 1..100, all results must be in (0.0, 1.0).
        for base in [2_u32, 3, 5, 7] {
            for n in 1_u32..100 {
                let v = radical_inverse(n, base);
                assert!(
                    v > 0.0 && v < 1.0,
                    "radical_inverse({n}, {base}) = {v} is not in (0.0, 1.0)"
                );
            }
        }
    }

    // --- Batch generator tests (ticket-020) ---

    /// Same inputs produce bitwise identical output (determinism).
    #[test]
    fn test_halton_batch_determinism() {
        let n_openings = 64;
        let dim = 2;
        let mut out1 = vec![0.0_f64; n_openings * dim];
        let mut out2 = vec![0.0_f64; n_openings * dim];
        generate_qmc_halton(42, 0, n_openings, dim, &mut out1);
        generate_qmc_halton(42, 0, n_openings, dim, &mut out2);
        assert_eq!(out1, out2, "generate_qmc_halton is not deterministic");
    }

    /// Different `base_seed` values must produce different output.
    #[test]
    fn test_halton_batch_different_seeds_differ() {
        let n_openings = 64;
        let dim = 2;
        let mut out1 = vec![0.0_f64; n_openings * dim];
        let mut out2 = vec![0.0_f64; n_openings * dim];
        generate_qmc_halton(42, 0, n_openings, dim, &mut out1);
        generate_qmc_halton(99, 0, n_openings, dim, &mut out2);
        assert_ne!(out1, out2, "different seeds produced identical output");
    }

    /// All output values must be finite for N=64, dim=5.
    #[test]
    fn test_halton_batch_all_finite() {
        let n_openings = 64;
        let dim = 5;
        let mut output = vec![0.0_f64; n_openings * dim];
        generate_qmc_halton(7, 3, n_openings, dim, &mut output);
        for (i, &v) in output.iter().enumerate() {
            assert!(v.is_finite(), "non-finite value at index {i}: {v}");
        }
    }

    /// All output values must be in the finite range expected for N(0,1).
    ///
    /// The BSM approximation clamps at ±8.22; all values must be within
    /// that range and finite.
    #[test]
    fn test_halton_batch_values_in_range() {
        let n_openings = 64;
        let dim = 2;
        let mut output = vec![0.0_f64; n_openings * dim];
        generate_qmc_halton(7, 3, n_openings, dim, &mut output);
        for (i, &v) in output.iter().enumerate() {
            assert!(v.is_finite(), "non-finite at index {i}");
            assert!(
                v > -8.22 && v < 8.22,
                "value {v} out of range (-8.22, 8.22) at index {i}"
            );
        }
    }

    /// Output must fill exactly `n_openings * dim` elements.
    #[test]
    fn test_halton_batch_correct_length() {
        let n_openings = 32;
        let dim = 4;
        let mut output = vec![0.0_f64; n_openings * dim];
        generate_qmc_halton(1, 2, n_openings, dim, &mut output);
        assert_eq!(output.len(), n_openings * dim);
    }

    /// `n_openings == 0` must not panic and must be a no-op.
    #[test]
    fn test_halton_batch_zero_openings() {
        let mut output: Vec<f64> = vec![];
        generate_qmc_halton(42, 0, 0, 3, &mut output);
        assert!(output.is_empty());
    }

    /// `dim == 0` must not panic and must be a no-op.
    #[test]
    fn test_halton_batch_zero_dim() {
        let mut output: Vec<f64> = vec![];
        generate_qmc_halton(42, 0, 5, 0, &mut output);
        assert!(output.is_empty());
    }

    // --- Point-wise generator tests (ticket-020) ---

    /// Same `HaltonPointSpec` produces bitwise identical output (determinism).
    #[test]
    fn test_halton_point_determinism() {
        let dim = 2;
        let spec = HaltonPointSpec {
            sampling_seed: 42,
            iteration: 0,
            scenario: 0,
            stage_id: 0,
            total_scenarios: 64,
            dim,
        };
        let mut out1 = vec![0.0_f64; dim];
        let mut out2 = vec![0.0_f64; dim];
        scrambled_halton_point(&spec, &mut out1);
        scrambled_halton_point(&spec, &mut out2);
        assert_eq!(out1, out2, "scrambled_halton_point is not deterministic");
    }

    /// Different `sampling_seed` values must produce different output.
    #[test]
    fn test_halton_point_different_seeds_differ() {
        let dim = 3;
        let base_spec = HaltonPointSpec {
            sampling_seed: 42,
            iteration: 0,
            scenario: 5,
            stage_id: 0,
            total_scenarios: 64,
            dim,
        };
        let mut out1 = vec![0.0_f64; dim];
        let mut out2 = vec![0.0_f64; dim];
        scrambled_halton_point(&base_spec, &mut out1);
        scrambled_halton_point(
            &HaltonPointSpec {
                sampling_seed: 43,
                ..base_spec
            },
            &mut out2,
        );
        assert_ne!(
            out1, out2,
            "different sampling_seeds produced identical output"
        );
    }

    /// All values must be finite for all scenarios 0..N.
    #[test]
    fn test_halton_point_all_finite() {
        let n = 64_usize;
        let dim = 4;
        for scenario in 0..n {
            let spec = HaltonPointSpec {
                sampling_seed: 42,
                iteration: 0,
                #[allow(clippy::cast_possible_truncation)]
                scenario: scenario as u32,
                stage_id: 1,
                #[allow(clippy::cast_possible_truncation)]
                total_scenarios: n as u32,
                dim,
            };
            let mut output = vec![0.0_f64; dim];
            scrambled_halton_point(&spec, &mut output);
            for (d, &v) in output.iter().enumerate() {
                assert!(
                    v.is_finite(),
                    "non-finite at scenario={scenario}, dim={d}: {v}"
                );
            }
        }
    }
}
