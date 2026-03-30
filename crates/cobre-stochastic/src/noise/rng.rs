//! PRNG wrapper for noise sampling.
//!
//! Initialises a `Pcg64` generator from a derived seed and samples
//! vectors of independent standard-normal (`N(0,1)`) variates. The
//! samples are consumed by the correlation module to produce spatially
//! correlated noise vectors for each scenario and stage.

use rand::SeedableRng;
use rand_pcg::Pcg64;

/// Initialize a `Pcg64` RNG from a derived 64-bit seed.
///
/// Expands the 64-bit seed to the 256-bit state required by `Pcg64`
/// using the [`SeedableRng::seed_from_u64`] method, which applies a
/// deterministic expansion algorithm. Calling this function twice with
/// the same seed produces two independent generators that yield
/// identical sequences.
///
/// # Resume invariant
///
/// The training pipeline derives per-draw seeds from
/// `(base_seed, iteration, scenario, stage)` via `derive_forward_seed`.
/// Because `iteration` is the absolute iteration number (not a counter
/// from zero), a resumed training run at iteration K+1 produces the
/// same seed — and therefore the same noise — as a continuous run.
/// This makes explicit RNG state serialization unnecessary for resume.
///
/// # Examples
///
/// ```
/// use rand::RngExt;
/// use cobre_stochastic::noise::rng::rng_from_seed;
///
/// let mut rng1 = rng_from_seed(12345);
/// let mut rng2 = rng_from_seed(12345);
///
/// // Both generators produce the same sequence.
/// assert_eq!(rng1.random::<f64>(), rng2.random::<f64>());
/// ```
#[must_use]
pub fn rng_from_seed(seed: u64) -> Pcg64 {
    Pcg64::seed_from_u64(seed)
}

#[cfg(test)]
mod tests {
    use rand::RngExt;

    use super::rng_from_seed;

    #[test]
    #[allow(clippy::float_cmp)]
    fn rng_from_seed_is_deterministic() {
        let mut rng1 = rng_from_seed(12345);
        let mut rng2 = rng_from_seed(12345);
        // Exact bitwise equality is intentional: we are testing that two RNGs
        // seeded identically reproduce the same bit pattern, not that two
        // computed floating-point results are approximately equal.
        for _ in 0..10 {
            assert_eq!(rng1.random::<f64>(), rng2.random::<f64>());
        }
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn rng_from_seed_differs_for_different_seeds() {
        let mut rng1 = rng_from_seed(0);
        let mut rng2 = rng_from_seed(1);
        // It is astronomically unlikely for the first f64 to coincide.
        assert_ne!(rng1.random::<f64>(), rng2.random::<f64>());
    }

    #[test]
    fn rng_from_seed_zero_is_valid() {
        // Seed 0 must not panic or produce all-zeros.
        let mut rng = rng_from_seed(0);
        let v: f64 = rng.random();
        assert!(v.is_finite());
    }

    #[test]
    fn rng_from_seed_max_u64_is_valid() {
        let mut rng = rng_from_seed(u64::MAX);
        let v: f64 = rng.random();
        assert!(v.is_finite());
    }
}
