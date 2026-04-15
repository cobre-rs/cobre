//! Out-of-sample forward-pass noise generation.
//!
//! Generates fresh N(0,1) noise on-the-fly during forward pass by:
//! 1. Dispatching on [`NoiseMethod`] to fill output with independent N(0,1) samples
//! 2. Applying spatial spectral correlation in-place
//!
//! Supported methods: SAA, LHS, QMC (Sobol/Halton), Selective (falls back to SAA).

use cobre_core::temporal::NoiseMethod;
use rand::RngExt;
use rand_distr::StandardNormal;

use crate::{
    StochasticError,
    noise::{rng::rng_from_seed, seed::derive_forward_seed_grouped},
    tree::{
        lhs::{LhsPointSpec, sample_lhs_point},
        qmc_halton::{HaltonPointSpec, scrambled_halton_point},
        qmc_sobol::{
            MAX_SOBOL_DIM, SobolPointSpec, SobolPrecomputed, scrambled_sobol_point,
            scrambled_sobol_point_precomputed,
        },
    },
};

/// Parameters for a single out-of-sample noise draw.
///
/// Bundles common seed/dimension fields to stay within the argument budget.
#[derive(Debug, Clone, Copy)]
pub(crate) struct FreshNoiseSpec {
    pub forward_seed: u64,
    pub noise_method: NoiseMethod,
    pub iteration: u32,
    pub scenario: u32,
    pub stage_id: u32,
    /// Noise group identifier used for seed derivation (Pattern C sharing).
    ///
    /// Stages within the same `(season_id, year)` bucket share the same
    /// `noise_group_id` so that their noise draws are identical.
    pub noise_group_id: u32,
    pub dim: usize,
    pub total_scenarios: u32,
}

/// Generate fresh correlated N(0,1) noise for a single `(iteration, scenario, stage)` triple.
///
/// Fills `output[0..spec.dim]` with independent N(0,1) samples, then applies
/// spatial spectral correlation in-place. No heap allocation inside this function.
///
/// # Errors
///
/// Returns [`StochasticError::DimensionExceedsCapacity`] when `QmcSobol`
/// and `spec.dim > MAX_SOBOL_DIM`.
///
/// # Panics
///
/// Panics if `output.len() < spec.dim` or (for LHS)
/// `perm_scratch.len() < spec.total_scenarios`.
#[cfg(test)]
pub(crate) fn sample_fresh(
    spec: FreshNoiseSpec,
    output: &mut [f64],
    perm_scratch: &mut [usize],
    correlation: &crate::correlation::resolve::DecomposedCorrelation,
    entity_order: &[cobre_core::EntityId],
) -> Result<(), StochasticError> {
    fill_uncorrelated(spec, None, output, perm_scratch)?;
    #[allow(clippy::cast_possible_wrap)]
    correlation.apply_correlation(spec.stage_id as i32, &mut output[..spec.dim], entity_order);
    Ok(())
}

/// Fill `output[0..spec.dim]` with independent N(0,1) noise without applying correlation.
///
/// Dispatches on [`NoiseMethod`] exactly as [`sample_fresh`] does for the
/// uncorrelated phase, but omits the spectral correlation step. Correlation is
/// applied externally by the composite `ForwardSampler` after all class segments
/// have been filled.
///
/// # Errors
///
/// Returns [`StochasticError::DimensionExceedsCapacity`] when `QmcSobol`
/// and `spec.dim > MAX_SOBOL_DIM`.
///
/// # Panics
///
/// Panics if `output.len() < spec.dim` or (for LHS)
/// `perm_scratch.len() < spec.total_scenarios`.
pub(crate) fn fill_uncorrelated(
    spec: FreshNoiseSpec,
    sobol_ctx: Option<&SobolPrecomputed>,
    output: &mut [f64],
    perm_scratch: &mut [usize],
) -> Result<(), StochasticError> {
    match spec.noise_method {
        NoiseMethod::Saa => {
            fill_saa(spec, output);
        }
        NoiseMethod::Lhs => {
            let lhs_spec = LhsPointSpec {
                sampling_seed: spec.forward_seed,
                iteration: spec.iteration,
                scenario: spec.scenario,
                stage_id: spec.noise_group_id,
                total_scenarios: spec.total_scenarios,
                dim: spec.dim,
            };
            sample_lhs_point(&lhs_spec, output, perm_scratch);
        }
        NoiseMethod::QmcSobol => {
            if spec.dim > MAX_SOBOL_DIM {
                return Err(StochasticError::DimensionExceedsCapacity {
                    dim: spec.dim,
                    max_dim: MAX_SOBOL_DIM,
                    method: "sobol".to_string(),
                });
            }
            let sobol_spec = SobolPointSpec {
                sampling_seed: spec.forward_seed,
                iteration: spec.iteration,
                scenario: spec.scenario,
                stage_id: spec.noise_group_id,
                total_scenarios: spec.total_scenarios,
                dim: spec.dim,
            };
            if let Some(ctx) = sobol_ctx {
                scrambled_sobol_point_precomputed(&sobol_spec, ctx, output);
            } else {
                scrambled_sobol_point(&sobol_spec, output);
            }
        }
        NoiseMethod::QmcHalton => {
            let halton_spec = HaltonPointSpec {
                sampling_seed: spec.forward_seed,
                iteration: spec.iteration,
                scenario: spec.scenario,
                stage_id: spec.noise_group_id,
                total_scenarios: spec.total_scenarios,
                dim: spec.dim,
            };
            scrambled_halton_point(&halton_spec, output);
        }
        NoiseMethod::Selective => {
            tracing::warn!(
                stage_id = spec.stage_id,
                "selective noise method not supported in forward pass; falling back to SAA at stage {}",
                spec.stage_id,
            );
            fill_saa(spec, output);
        }
        NoiseMethod::HistoricalResiduals => {
            tracing::warn!(
                stage_id = spec.stage_id,
                "historical_residuals noise method not yet wired in forward pass; falling back to SAA at stage {} (see ticket-009)",
                spec.stage_id,
            );
            fill_saa(spec, output);
        }
    }
    Ok(())
}

/// Fill `output[0..spec.dim]` with independent N(0,1) draws using SAA.
fn fill_saa(spec: FreshNoiseSpec, output: &mut [f64]) {
    let seed = derive_forward_seed_grouped(
        spec.forward_seed,
        spec.iteration,
        spec.scenario,
        spec.noise_group_id,
    );
    let mut rng = rng_from_seed(seed);
    for slot in output.iter_mut().take(spec.dim) {
        *slot = rng.sample(StandardNormal);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp
)]
mod tests {
    use std::collections::BTreeMap;

    use cobre_core::{
        EntityId,
        scenario::{CorrelationEntity, CorrelationGroup, CorrelationModel, CorrelationProfile},
        temporal::NoiseMethod,
    };

    use crate::{StochasticError, correlation::resolve::DecomposedCorrelation};

    use super::{FreshNoiseSpec, fill_uncorrelated, sample_fresh};

    fn identity_correlation(entity_ids: &[i32]) -> DecomposedCorrelation {
        let n = entity_ids.len();
        let matrix: Vec<Vec<f64>> = (0..n)
            .map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
            .collect();
        let mut profiles = BTreeMap::new();
        profiles.insert(
            "default".to_string(),
            CorrelationProfile {
                groups: vec![CorrelationGroup {
                    name: "g1".to_string(),
                    entities: entity_ids
                        .iter()
                        .map(|&id| CorrelationEntity {
                            entity_type: "inflow".to_string(),
                            id: EntityId(id),
                        })
                        .collect(),
                    matrix,
                }],
            },
        );
        DecomposedCorrelation::build(&CorrelationModel {
            method: "spectral".to_string(),
            profiles,
            schedule: vec![],
        })
        .unwrap()
    }

    fn make_entity_order(ids: &[i32]) -> Vec<EntityId> {
        ids.iter().map(|&id| EntityId(id)).collect()
    }

    fn base_spec(noise_method: NoiseMethod) -> FreshNoiseSpec {
        FreshNoiseSpec {
            forward_seed: 42,
            noise_method,
            iteration: 0,
            scenario: 0,
            stage_id: 0,
            noise_group_id: 0,
            dim: 3,
            total_scenarios: 10,
        }
    }

    #[test]
    fn test_saa_determinism() {
        let corr = identity_correlation(&[1, 2, 3]);
        let entity_order = make_entity_order(&[1, 2, 3]);
        let spec = base_spec(NoiseMethod::Saa);

        let mut out_a = vec![0.0f64; spec.dim];
        let mut out_b = vec![0.0f64; spec.dim];
        let mut perm = vec![0usize; spec.total_scenarios as usize];

        sample_fresh(spec, &mut out_a, &mut perm, &corr, &entity_order).unwrap();
        sample_fresh(spec, &mut out_b, &mut perm, &corr, &entity_order).unwrap();

        assert_eq!(
            out_a, out_b,
            "SAA with identical inputs must produce bitwise-identical output"
        );
    }

    #[test]
    fn test_saa_different_seeds_differ() {
        let corr_a = identity_correlation(&[1, 2, 3]);
        let corr_b = identity_correlation(&[1, 2, 3]);
        let entity_order = make_entity_order(&[1, 2, 3]);

        let spec_a = FreshNoiseSpec {
            forward_seed: 42,
            ..base_spec(NoiseMethod::Saa)
        };
        let spec_b = FreshNoiseSpec {
            forward_seed: 99,
            ..base_spec(NoiseMethod::Saa)
        };

        let mut out_a = vec![0.0f64; spec_a.dim];
        let mut out_b = vec![0.0f64; spec_b.dim];
        let mut perm = vec![0usize; 10];

        sample_fresh(spec_a, &mut out_a, &mut perm, &corr_a, &entity_order).unwrap();
        sample_fresh(spec_b, &mut out_b, &mut perm, &corr_b, &entity_order).unwrap();

        assert_ne!(
            out_a, out_b,
            "SAA with different forward_seed must produce different noise"
        );
    }

    #[test]
    fn test_lhs_produces_finite_noise() {
        let corr = identity_correlation(&[1, 2, 3]);
        let entity_order = make_entity_order(&[1, 2, 3]);
        let spec = FreshNoiseSpec {
            scenario: 5,
            ..base_spec(NoiseMethod::Lhs)
        };

        let mut output = vec![0.0f64; spec.dim];
        let mut perm = vec![0usize; spec.total_scenarios as usize];

        let result = sample_fresh(spec, &mut output, &mut perm, &corr, &entity_order);

        assert!(result.is_ok(), "LHS must return Ok(()), got {result:?}");
        for (i, &v) in output.iter().enumerate() {
            assert!(v.is_finite(), "LHS output[{i}] is not finite: {v}");
        }
    }

    #[test]
    fn test_sobol_produces_finite_noise() {
        let corr = identity_correlation(&[1, 2, 3]);
        let entity_order = make_entity_order(&[1, 2, 3]);
        let spec = base_spec(NoiseMethod::QmcSobol);

        let mut output = vec![0.0f64; spec.dim];
        let mut perm = vec![0usize; spec.total_scenarios as usize];

        let result = sample_fresh(spec, &mut output, &mut perm, &corr, &entity_order);

        assert!(
            result.is_ok(),
            "QmcSobol must return Ok(()), got {result:?}"
        );
        for (i, &v) in output.iter().enumerate() {
            assert!(v.is_finite(), "QmcSobol output[{i}] is not finite: {v}");
        }
    }

    #[test]
    fn test_sobol_dim_exceeds_capacity() {
        let corr = identity_correlation(&[1]);
        let entity_order = make_entity_order(&[1]);
        let spec = FreshNoiseSpec {
            dim: 21_202, // one above MAX_SOBOL_DIM = 21_201
            ..base_spec(NoiseMethod::QmcSobol)
        };

        let mut output = vec![0.0f64; spec.dim];
        let mut perm = vec![0usize; spec.total_scenarios as usize];

        let result = sample_fresh(spec, &mut output, &mut perm, &corr, &entity_order);

        match result {
            Err(StochasticError::DimensionExceedsCapacity {
                dim: got_dim,
                max_dim,
                method,
            }) => {
                assert_eq!(got_dim, 21_202, "dim field");
                assert_eq!(max_dim, 21_201, "max_dim field");
                assert!(
                    method.contains("sobol"),
                    "method must contain 'sobol', got: {method}"
                );
            }
            Ok(()) => panic!("expected Err(DimensionExceedsCapacity) but got Ok"),
            Err(other) => panic!("expected DimensionExceedsCapacity, got {other:?}"),
        }
    }

    #[test]
    fn test_halton_produces_finite_noise() {
        let corr = identity_correlation(&[1, 2, 3]);
        let entity_order = make_entity_order(&[1, 2, 3]);
        let spec = base_spec(NoiseMethod::QmcHalton);

        let mut output = vec![0.0f64; spec.dim];
        let mut perm = vec![0usize; spec.total_scenarios as usize];

        let result = sample_fresh(spec, &mut output, &mut perm, &corr, &entity_order);

        assert!(
            result.is_ok(),
            "QmcHalton must return Ok(()), got {result:?}"
        );
        for (i, &v) in output.iter().enumerate() {
            assert!(v.is_finite(), "QmcHalton output[{i}] is not finite: {v}");
        }
    }

    #[test]
    fn test_selective_falls_back_to_saa() {
        let corr = identity_correlation(&[1, 2, 3]);
        let entity_order = make_entity_order(&[1, 2, 3]);
        let spec = base_spec(NoiseMethod::Selective);

        let mut output = vec![0.0f64; spec.dim];
        let mut perm = vec![0usize; spec.total_scenarios as usize];

        let result = sample_fresh(spec, &mut output, &mut perm, &corr, &entity_order);

        assert!(
            result.is_ok(),
            "Selective fallback must return Ok(()), got {result:?}"
        );
        for (i, &v) in output.iter().enumerate() {
            assert!(v.is_finite(), "Selective output[{i}] is not finite: {v}");
        }
    }

    #[test]
    fn test_selective_matches_saa() {
        let corr_a = identity_correlation(&[1, 2, 3]);
        let corr_b = identity_correlation(&[1, 2, 3]);
        let entity_order = make_entity_order(&[1, 2, 3]);

        let spec = FreshNoiseSpec {
            iteration: 1,
            scenario: 2,
            stage_id: 3,
            ..base_spec(NoiseMethod::Selective)
        };
        let spec_saa = FreshNoiseSpec {
            noise_method: NoiseMethod::Saa,
            ..spec
        };

        let mut out_selective = vec![0.0f64; spec.dim];
        let mut out_saa = vec![0.0f64; spec.dim];
        let mut perm = vec![0usize; spec.total_scenarios as usize];

        sample_fresh(spec, &mut out_selective, &mut perm, &corr_a, &entity_order).unwrap();
        sample_fresh(spec_saa, &mut out_saa, &mut perm, &corr_b, &entity_order).unwrap();

        assert_eq!(
            out_selective, out_saa,
            "Selective fallback must produce the same output as Saa with the same inputs"
        );
    }

    // -----------------------------------------------------------------------
    // Tests for fill_uncorrelated
    // -----------------------------------------------------------------------

    #[test]
    fn test_fill_uncorrelated_saa_deterministic() {
        let spec = base_spec(NoiseMethod::Saa);
        let mut out_a = vec![0.0f64; spec.dim];
        let mut out_b = vec![0.0f64; spec.dim];
        let mut perm = vec![0usize; spec.total_scenarios as usize];

        fill_uncorrelated(spec, None, &mut out_a, &mut perm).unwrap();
        fill_uncorrelated(spec, None, &mut out_b, &mut perm).unwrap();

        assert_eq!(
            out_a, out_b,
            "fill_uncorrelated with identical SAA inputs must produce bit-identical output"
        );
    }

    #[test]
    fn test_fill_uncorrelated_sobol_dim_exceeds_capacity() {
        let spec = FreshNoiseSpec {
            dim: 21_202, // one above MAX_SOBOL_DIM = 21_201
            ..base_spec(NoiseMethod::QmcSobol)
        };
        let mut output = vec![0.0f64; spec.dim];
        let mut perm = vec![0usize; spec.total_scenarios as usize];

        let result = fill_uncorrelated(spec, None, &mut output, &mut perm);

        match result {
            Err(StochasticError::DimensionExceedsCapacity {
                dim: got_dim,
                max_dim,
                method,
            }) => {
                assert_eq!(got_dim, 21_202, "dim field");
                assert_eq!(max_dim, 21_201, "max_dim field");
                assert!(
                    method.contains("sobol"),
                    "method must contain 'sobol', got: {method}"
                );
            }
            Ok(()) => panic!("expected Err(DimensionExceedsCapacity) but got Ok"),
            Err(other) => panic!("expected DimensionExceedsCapacity, got {other:?}"),
        }
    }

    #[test]
    fn test_fill_uncorrelated_produces_finite_values() {
        let methods = [
            NoiseMethod::Saa,
            NoiseMethod::Lhs,
            NoiseMethod::QmcSobol,
            NoiseMethod::QmcHalton,
            NoiseMethod::Selective,
        ];

        for method in methods {
            let spec = FreshNoiseSpec {
                scenario: 5,
                ..base_spec(method)
            };
            let mut output = vec![0.0f64; spec.dim];
            let mut perm = vec![0usize; spec.total_scenarios as usize];

            let result = fill_uncorrelated(spec, None, &mut output, &mut perm);

            assert!(
                result.is_ok(),
                "{method:?}: fill_uncorrelated must return Ok(()), got {result:?}"
            );
            for (i, &v) in output.iter().enumerate() {
                assert!(v.is_finite(), "{method:?}: output[{i}] is not finite: {v}");
            }
        }
    }
}
