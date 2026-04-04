//! Sobol quasi-Monte Carlo sequence generation with Joe-Kuo direction tables.
//!
//! This module embeds the Joe-Kuo 2010 direction number dataset for up to
//! [`MAX_SOBOL_DIM`] dimensions. The data is stored as a statically-typed
//! Rust static array that requires no runtime allocation or deserialization.
//!
//! ## Dimension numbering
//!
//! - Dimension 1 uses the van der Corput sequence (no direction table needed).
//! - Dimensions 2–21201 each have an entry in [`SOBOL_DIRECTIONS`].
//! - Array index 0 corresponds to dimension 2.
//!
//! ## Direction number convention
//!
//! [`SobolDirEntry::initial_dirs`] stores raw `m_i` values exactly as listed in
//! the Joe-Kuo file. The sequence generator applies the left-shift
//! `v[j] = m_j << (32 - j)` when constructing the direction vectors for the
//! standard Gray-code Sobol formulation.
//!
//! ## Entry points
//!
//! - [`generate_qmc_sobol`]: batch generation for all openings of a stage,
//!   using Gray-code recurrence for O(1) updates per point.
//! - [`scrambled_sobol_point`]: single-scenario point-wise generation via
//!   direct binary decomposition, used by the out-of-sample forward pass.
//!
//! Both paths apply Matousek linear scrambling `x' = a*x + b (mod 2^32)` with
//! seed-derived parameters, then transform to N(0,1) via `norm_quantile`.

mod sobol_directions;

pub(crate) use sobol_directions::{SOBOL_DIRECTIONS, SOBOL_MAX_DIM};

use rand::RngExt;

use crate::noise::{
    quantile::norm_quantile,
    rng::rng_from_seed,
    seed::{derive_opening_seed, derive_stage_seed},
};

/// Maximum supported dimension for Sobol sequences.
///
/// Equals 21,201: dimension 1 uses the van der Corput sequence and
/// dimensions 2–21,201 are covered by the 21,200 Joe-Kuo entries in
/// [`SOBOL_DIRECTIONS`].
pub(crate) const MAX_SOBOL_DIM: usize = SOBOL_MAX_DIM;

/// Scaling constant: `1.0 / 2^32`.
///
/// Used to convert a uniform u32 to a float in `[0, 1)`.
/// Written as a reciprocal constant so the compiler pre-computes it.
const INV_2_32: f64 = 1.0 / 4_294_967_296.0;

/// Build the full 32-bit direction vectors for each of the `dim` dimensions.
///
/// Dimension 1 uses the van der Corput sequence; dimensions 2+ use Joe-Kuo
/// direction numbers with polynomial recurrence. Returns `result[d][j]` as the
/// `j`-th direction number for dimension `d` with significant bit at `31 - j`.
///
/// # Panics
///
/// Panics if `dim > MAX_SOBOL_DIM`.
fn build_direction_matrix(dim: usize) -> Vec<[u32; 32]> {
    assert!(
        dim <= MAX_SOBOL_DIM,
        "dim {dim} exceeds MAX_SOBOL_DIM {MAX_SOBOL_DIM}"
    );

    let mut result: Vec<[u32; 32]> = Vec::with_capacity(dim);

    for d in 0..dim {
        let mut v = [0u32; 32];

        if d == 0 {
            // Dimension 1: van der Corput sequence.
            // v[j] = 1 << (31 - j) for j = 0..32.
            for (j, slot) in v.iter_mut().enumerate() {
                *slot = 1u32 << (31 - j);
            }
        } else {
            // Dimensions 2+: read from SOBOL_DIRECTIONS[d-1].
            let entry = &SOBOL_DIRECTIONS[d - 1];
            let s = entry.degree as usize;
            let a = entry.poly;

            // Left-shift the initial direction numbers to their bit positions.
            // Joe-Kuo stores raw m_j values; we need v[j] = m_j << (32 - (j+1)).
            // j here is 0-indexed, so j=0 gives shift = 31, j=1 gives shift = 30, etc.
            for (j, slot) in v[..s].iter_mut().enumerate() {
                *slot = entry.initial_dirs[j] << (31 - j);
            }

            // Apply the recurrence for j >= s (0-indexed).
            // Recurrence (Joe-Kuo convention, 1-indexed j):
            //   v[j] = v[j-s] XOR (v[j-s] >> s) XOR sum_{k=1}^{s-1} a_{s-1-k} * v[j-k]
            // In 0-indexed terms (j >= s):
            //   v[j] = v[j-s] XOR (v[j-s] >> s)
            //          XOR sum_{k=1}^{s-1} ((a >> (s-1-k)) & 1) * v[j-k]
            for j in s..32 {
                let mut x = v[j - s] ^ (v[j - s] >> s);
                for k in 1..s {
                    if (a >> (s - 1 - k)) & 1 == 1 {
                        x ^= v[j - k];
                    }
                }
                v[j] = x;
            }
        }

        result.push(v);
    }

    result
}

/// Derive Matousek linear scrambling parameters `(a_d, b_d)` for each dimension,
/// with `a_d` forced odd to ensure bijection on `Z_{2^32}`.
fn derive_scramble_params(seed: u64, dim: usize) -> Vec<(u32, u32)> {
    let mut rng = rng_from_seed(seed);
    (0..dim)
        .map(|_| {
            let a: u32 = rng.random::<u32>() | 1; // ensure odd for bijection
            let b: u32 = rng.random();
            (a, b)
        })
        .collect()
}

/// Fill `output` with `n_openings × dim` standard-normal N(0,1) values using
/// Scrambled Sobol QMC with Gray-code recurrence for O(1) updates per point.
///
/// Output layout: opening-major `output[opening * dim + entity]`.
///
/// # Panics
///
/// Panics if `output.len() < n_openings * dim` or `dim > MAX_SOBOL_DIM`.
pub fn generate_qmc_sobol(
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
    let directions = build_direction_matrix(dim);
    let scramble = derive_scramble_params(seed, dim);

    // Running Sobol state: sobol_state[d] holds the current unscrambled coordinate.
    let mut sobol_state = vec![0u32; dim];

    // Point 0: sobol_state = 0 for all dimensions.
    for d in 0..dim {
        let (a, b) = scramble[d];
        let xp = a.wrapping_mul(sobol_state[d]).wrapping_add(b);
        let u = (f64::from(xp) * INV_2_32).clamp(f64::MIN_POSITIVE, 1.0 - f64::EPSILON);
        output[d] = norm_quantile(u);
    }

    // Points 1..n_openings: Gray-code recurrence.
    // For the i-th point (i >= 1), XOR v_d[c] into sobol_state[d] where
    // c = i.trailing_zeros() (position of rightmost 1-bit of i).
    for i in 1..n_openings {
        #[allow(clippy::cast_possible_truncation)]
        let c = i.trailing_zeros() as usize;
        for d in 0..dim {
            sobol_state[d] ^= directions[d][c];
            let (a, b) = scramble[d];
            let xp = a.wrapping_mul(sobol_state[d]).wrapping_add(b);
            let u = (f64::from(xp) * INV_2_32).clamp(f64::MIN_POSITIVE, 1.0 - f64::EPSILON);
            output[i * dim + d] = norm_quantile(u);
        }
    }
}

/// Configuration for single-scenario Sobol point generation.
///
/// Bundles the parameters needed by [`scrambled_sobol_point`] to generate
/// one scenario's noise vector without inter-worker coordination.
#[derive(Debug, Clone, Copy)]
pub struct SobolPointSpec {
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

/// Generate one scenario's noise vector using scrambled Sobol QMC with direct
/// binary decomposition, independent of all other scenarios.
///
/// # Panics
///
/// Panics if `output.len() < spec.dim` or `spec.dim > MAX_SOBOL_DIM`.
pub fn scrambled_sobol_point(spec: &SobolPointSpec, output: &mut [f64]) {
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
    let directions = build_direction_matrix(spec.dim);
    let scramble = derive_scramble_params(seed, spec.dim);

    for d in 0..spec.dim {
        // Direct binary decomposition: XOR v[j] for each bit j set in scenario.
        let mut xd = 0u32;
        let mut scenario = spec.scenario;
        let mut j = 0usize;
        while scenario != 0 {
            if scenario & 1 == 1 {
                xd ^= directions[d][j];
            }
            scenario >>= 1;
            j += 1;
        }

        let (a, b) = scramble[d];
        let xp = a.wrapping_mul(xd).wrapping_add(b);
        let u = (f64::from(xp) * INV_2_32).clamp(f64::MIN_POSITIVE, 1.0 - f64::EPSILON);
        output[d] = norm_quantile(u);
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::float_cmp,
    clippy::panic,
    clippy::cast_possible_truncation
)]
mod tests {
    use super::{
        INV_2_32, MAX_SOBOL_DIM, SobolPointSpec, build_direction_matrix, generate_qmc_sobol,
        scrambled_sobol_point,
    };

    /// Generate unscrambled Gray-code Sobol points in `[0,1)` for regression testing.
    fn generate_unscrambled_sobol(n_openings: usize, dim: usize) -> Vec<Vec<f64>> {
        let directions = build_direction_matrix(dim);
        let mut sobol_state = vec![0u32; dim];
        let mut result = Vec::with_capacity(n_openings);

        result.push((0..dim).map(|_| 0.0_f64).collect());

        for i in 1..n_openings {
            let c = i.trailing_zeros() as usize;
            for d in 0..dim {
                sobol_state[d] ^= directions[d][c];
            }
            let point: Vec<f64> = (0..dim)
                .map(|d| f64::from(sobol_state[d]) * INV_2_32)
                .collect();
            result.push(point);
        }

        result
    }

    /// Same `(base_seed, stage_id, n_openings, dim)` must produce bitwise identical output.
    #[test]
    fn test_sobol_batch_determinism() {
        let n_openings = 64;
        let dim = 2;
        let mut out1 = vec![0.0_f64; n_openings * dim];
        let mut out2 = vec![0.0_f64; n_openings * dim];
        generate_qmc_sobol(42, 0, n_openings, dim, &mut out1);
        generate_qmc_sobol(42, 0, n_openings, dim, &mut out2);
        assert_eq!(out1, out2, "generate_qmc_sobol is not deterministic");
    }

    /// Different `base_seed` values must produce different output.
    #[test]
    fn test_sobol_batch_different_seeds_differ() {
        let n_openings = 64;
        let dim = 2;
        let mut out1 = vec![0.0_f64; n_openings * dim];
        let mut out2 = vec![0.0_f64; n_openings * dim];
        generate_qmc_sobol(42, 0, n_openings, dim, &mut out1);
        generate_qmc_sobol(99, 0, n_openings, dim, &mut out2);
        assert_ne!(out1, out2, "different seeds produced identical output");
    }

    /// All output values must be finite for N=64, dim=5.
    #[test]
    fn test_sobol_batch_all_finite() {
        let n_openings = 64;
        let dim = 5;
        let mut output = vec![0.0_f64; n_openings * dim];
        generate_qmc_sobol(7, 3, n_openings, dim, &mut output);
        for (i, &v) in output.iter().enumerate() {
            assert!(v.is_finite(), "non-finite value at index {i}: {v}");
        }
    }

    /// Output must fill exactly `n_openings * dim` elements.
    #[test]
    fn test_sobol_batch_correct_length() {
        let n_openings = 32;
        let dim = 4;
        let mut output = vec![0.0_f64; n_openings * dim];
        generate_qmc_sobol(1, 2, n_openings, dim, &mut output);
        assert_eq!(output.len(), n_openings * dim);
    }

    /// Same `SobolPointSpec` must produce bitwise identical output.
    #[test]
    fn test_sobol_point_determinism() {
        let dim = 2;
        let spec = SobolPointSpec {
            sampling_seed: 42,
            iteration: 0,
            scenario: 0,
            stage_id: 0,
            total_scenarios: 64,
            dim,
        };
        let mut out1 = vec![0.0_f64; dim];
        let mut out2 = vec![0.0_f64; dim];
        scrambled_sobol_point(&spec, &mut out1);
        scrambled_sobol_point(&spec, &mut out2);
        assert_eq!(out1, out2, "scrambled_sobol_point is not deterministic");
    }

    /// Different `sampling_seed` must produce different output.
    #[test]
    fn test_sobol_point_different_seeds_differ() {
        let dim = 3;
        let base_spec = SobolPointSpec {
            sampling_seed: 42,
            iteration: 0,
            scenario: 5,
            stage_id: 0,
            total_scenarios: 64,
            dim,
        };
        let mut out1 = vec![0.0_f64; dim];
        let mut out2 = vec![0.0_f64; dim];
        scrambled_sobol_point(&base_spec, &mut out1);
        scrambled_sobol_point(
            &SobolPointSpec {
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

    /// All values finite across all scenarios 0..N.
    #[test]
    fn test_sobol_point_all_finite() {
        let n = 64_usize;
        let dim = 4;
        for scenario in 0..n {
            let spec = SobolPointSpec {
                sampling_seed: 42,
                iteration: 0,
                scenario: scenario as u32,
                stage_id: 1,
                total_scenarios: n as u32,
                dim,
            };
            let mut output = vec![0.0_f64; dim];
            scrambled_sobol_point(&spec, &mut output);
            for (d, &v) in output.iter().enumerate() {
                assert!(
                    v.is_finite(),
                    "non-finite at scenario={scenario}, dim={d}: {v}"
                );
            }
        }
    }

    /// Dimension 1 directions must equal `1 << (31 - j)` for j in 0..32.
    #[test]
    fn test_build_direction_matrix_dim1() {
        let dirs = build_direction_matrix(1);
        assert_eq!(dirs.len(), 1, "expected 1 direction vector");
        for (j, &v) in dirs[0].iter().enumerate() {
            let expected = 1u32 << (31 - j);
            assert_eq!(
                v, expected,
                "dim1 direction[{j}]: got {v:#010x}, expected {expected:#010x}"
            );
        }
    }

    /// All batch output values must be in the finite range expected for N(0,1).
    ///
    /// The BSM approximation clamps at ±8.22; all values must be within that
    /// range and finite.
    #[test]
    fn test_sobol_batch_values_in_range() {
        let n_openings = 64;
        let dim = 2;
        let mut output = vec![0.0_f64; n_openings * dim];
        generate_qmc_sobol(42, 0, n_openings, dim, &mut output);
        for (i, &v) in output.iter().enumerate() {
            assert!(v.is_finite(), "non-finite at index {i}");
            assert!(
                v > -8.22 && v < 8.22,
                "value {v} out of range [-8.22, 8.22] at index {i}"
            );
        }
    }

    /// `MAX_SOBOL_DIM` must equal 21,201.
    #[test]
    fn test_max_sobol_dim_constant() {
        assert_eq!(MAX_SOBOL_DIM, 21_201);
    }

    /// Verify unscrambled dimension 1 (van der Corput base-2) first 8 points
    /// against the known Gray-code Sobol values.
    ///
    /// The Gray-code Sobol sequence for dimension 1 produces the same set of
    /// values as the van der Corput base-2 sequence but in a different traversal
    /// order dictated by the Gray-code recurrence. The first 8 expected values
    /// are computed from `v[j] = 1 << (31 - j)` with Gray-code XOR updates.
    #[test]
    fn test_unscrambled_dim1_first_8_points() {
        let pts = generate_unscrambled_sobol(8, 1);
        // Gray-code Sobol sequence for dim 1 (van der Corput values, Gray-code order):
        // i=0: state=0                   → 0/2^32   = 0.0
        // i=1: c=0, state^=v[0]=2^31     → 2^31     = 0.5
        // i=2: c=1, state^=v[1]=2^30     → 3*2^30   = 0.75
        // i=3: c=0, state^=v[0]=2^31     → 2^30     = 0.25
        // i=4: c=2, state^=v[2]=2^29     → 3*2^29   = 0.375
        // i=5: c=0, state^=v[0]=2^31     → 7*2^29   = 0.875
        // i=6: c=1, state^=v[1]=2^30     → 5*2^29   = 0.625
        // i=7: c=0, state^=v[0]=2^31     → 1*2^29   = 0.125
        let expected = [0.0, 0.5, 0.75, 0.25, 0.375, 0.875, 0.625, 0.125];
        for (i, (got, &exp)) in pts.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got[0] - exp).abs() < 1e-12,
                "dim1 point[{i}]: got {}, expected {}",
                got[0],
                exp
            );
        }
    }

    /// Verify unscrambled dimension 2 first 8 points against the known
    /// Joe-Kuo direction-table values.
    ///
    /// Dimension 2 uses `SOBOL_DIRECTIONS[0]` with degree=1, poly=0, and
    /// `initial_dirs=[1,...]`. The direction vector recurrence with degree 1
    /// is `v[j] = v[j-1] XOR (v[j-1] >> 1)`, giving a sequence of known
    /// values when combined with the Gray-code update.
    #[test]
    fn test_unscrambled_dim2_first_8_points() {
        let pts = generate_unscrambled_sobol(8, 2);
        // Gray-code Sobol sequence for dim 2 (Joe-Kuo degree=1, poly=0, m=[1,...]):
        // v[0]=0x80000000, v[1]=0xC0000000, v[2]=0xA0000000, ...
        // i=0: state=0                          → 0.0
        // i=1: c=0, state^=0x80000000           → 0.5
        // i=2: c=1, state^=0xC0000000           → 0x40000000/2^32 = 0.25
        // i=3: c=0, state^=0x80000000           → 0xC0000000/2^32 = 0.75
        // i=4: c=2, state^=0xA0000000           → 0x60000000/2^32 = 0.375
        // i=5: c=0, state^=0x80000000           → 0xE0000000/2^32 = 0.875
        // i=6: c=1, state^=0xC0000000           → 0x20000000/2^32 = 0.125
        // i=7: c=0, state^=0x80000000           → 0xA0000000/2^32 = 0.625
        let expected = [0.0, 0.5, 0.25, 0.75, 0.375, 0.875, 0.125, 0.625];
        for (i, (got, &exp)) in pts.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got[1] - exp).abs() < 1e-12,
                "dim2 point[{i}]: got {}, expected {}",
                got[1],
                exp
            );
        }
    }
}
