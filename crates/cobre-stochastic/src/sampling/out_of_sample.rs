//! Out-of-sample forward-pass noise generation: fresh independent N(0,1) noise
//! per draw, optionally followed by in-place spatial spectral correlation.
//!
//! Methods: SAA, LHS, QMC (Sobol/Halton); Selective and `HistoricalResiduals`
//! fall back to SAA in the forward pass.

use cobre_core::temporal::NoiseMethod;
use rand::RngExt;
use rand_distr::StandardNormal;

#[cfg(test)]
use crate::{ClassNoiseTables, DecomposedCorrelation, EntityClass};
use crate::{
    NoisePointSpec, NoiseTable, StochasticError,
    noise::{rng::rng_from_seed, seed::derive_forward_seed_grouped},
    tree::{
        lhs::sample_lhs_point, qmc_halton::scrambled_halton_point, qmc_sobol::scrambled_sobol_point,
    },
};

/// Parameters for a single out-of-sample noise draw.
#[derive(Debug, Clone, Copy)]
pub(crate) struct FreshNoiseSpec {
    pub forward_seed: u64,
    pub noise_method: NoiseMethod,
    pub iteration: u32,
    pub scenario: u32,
    /// Seed-derivation identifier: stages sharing a `(season_id, year)` bucket
    /// share a `noise_group_id` so their noise draws are identical.
    pub noise_group_id: u32,
    pub dim: usize,
    pub total_scenarios: u32,
}

/// Fill `output[0..spec.dim]` with fresh N(0,1) noise, then apply spatial
/// spectral correlation in-place for the inflow class — this helper's fixtures
/// never populate load or NCS segments.
///
/// # Panics
///
/// Panics if `output.len() < spec.dim`.
#[cfg(test)]
pub(crate) fn sample_fresh(
    spec: FreshNoiseSpec,
    output: &mut [f64],
    correlation: &DecomposedCorrelation,
    stage_id: i32,
) -> Result<(), StochasticError> {
    let table = table_for_spec(spec)?;
    fill_uncorrelated(spec, &table, output)?;
    let groups = correlation.groups_for_stage(stage_id);
    let mut scratch = vec![0.0_f64; 2 * spec.dim];
    DecomposedCorrelation::apply_groups_for_class(
        groups,
        EntityClass::Inflow,
        &mut output[..spec.dim],
        &mut scratch,
    );
    Ok(())
}

/// Build the [`NoiseTable`] a single `spec` would resolve to under
/// [`ClassNoiseTables::refill`], for tests that call [`fill_uncorrelated`]
/// directly instead of through a driver's rebuilt tables.
#[cfg(test)]
fn table_for_spec(spec: FreshNoiseSpec) -> Result<NoiseTable, StochasticError> {
    let mut tables = ClassNoiseTables::default();
    tables.refill(
        spec.forward_seed,
        spec.dim,
        spec.iteration,
        spec.total_scenarios,
        &[spec.noise_group_id],
        &[spec.noise_method],
    )?;
    Ok(tables.table_at(0).cloned().unwrap_or(NoiseTable::Direct))
}

/// Fill `output[0..spec.dim]` with independent, uncorrelated N(0,1) noise —
/// the composite `ForwardSampler` correlates afterward, once all class
/// segments are filled.
///
/// # Errors
///
/// Returns [`StochasticError::InsufficientData`] when `table`'s variant does
/// not match `spec.noise_method`.
///
/// # Panics
///
/// Panics if `output.len() < spec.dim`.
pub(crate) fn fill_uncorrelated(
    spec: FreshNoiseSpec,
    table: &NoiseTable,
    output: &mut [f64],
) -> Result<(), StochasticError> {
    let point_spec = NoisePointSpec {
        sampling_seed: spec.forward_seed,
        iteration: spec.iteration,
        scenario: spec.scenario,
        stream_id: spec.noise_group_id,
        total_scenarios: spec.total_scenarios,
        dim: spec.dim,
    };
    match spec.noise_method {
        NoiseMethod::Saa => fill_saa(spec, output),
        NoiseMethod::Lhs => {
            let NoiseTable::Lhs(ctx) = table else {
                return Err(StochasticError::InsufficientData {
                    context: format!(
                        "fill_uncorrelated: noise_group_id {} expected NoiseTable::Lhs, got {table:?}",
                        spec.noise_group_id,
                    ),
                });
            };
            sample_lhs_point(&point_spec, ctx, output);
        }
        NoiseMethod::QmcSobol => {
            let NoiseTable::Sobol(ctx) = table else {
                return Err(StochasticError::InsufficientData {
                    context: format!(
                        "fill_uncorrelated: noise_group_id {} expected NoiseTable::Sobol, got {table:?}",
                        spec.noise_group_id,
                    ),
                });
            };
            scrambled_sobol_point(&point_spec, ctx, output);
        }
        NoiseMethod::QmcHalton => {
            let NoiseTable::Halton(ctx) = table else {
                return Err(StochasticError::InsufficientData {
                    context: format!(
                        "fill_uncorrelated: noise_group_id {} expected NoiseTable::Halton, got {table:?}",
                        spec.noise_group_id,
                    ),
                });
            };
            scrambled_halton_point(&point_spec, ctx, output);
        }
        NoiseMethod::Selective | NoiseMethod::HistoricalResiduals => {
            fill_saa(spec, output);
        }
    }
    Ok(())
}

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

    use crate::{ClassDimensions, StochasticError, correlation::resolve::DecomposedCorrelation};

    use super::{FreshNoiseSpec, fill_uncorrelated, sample_fresh, table_for_spec};

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
        let entity_order: Vec<EntityId> = entity_ids.iter().map(|&id| EntityId(id)).collect();
        DecomposedCorrelation::build(
            &CorrelationModel {
                method: "spectral".to_string(),
                profiles,
                schedule: vec![],
            },
            &entity_order,
            ClassDimensions {
                n_hydros: entity_ids.len(),
                n_load_buses: 0,
                n_ncs: 0,
            },
        )
        .unwrap()
    }

    fn base_spec(noise_method: NoiseMethod) -> FreshNoiseSpec {
        FreshNoiseSpec {
            forward_seed: 42,
            noise_method,
            iteration: 0,
            scenario: 0,
            noise_group_id: 0,
            dim: 3,
            total_scenarios: 10,
        }
    }

    #[test]
    fn test_saa_determinism() {
        let corr = identity_correlation(&[1, 2, 3]);
        let spec = base_spec(NoiseMethod::Saa);

        let mut out_a = vec![0.0f64; spec.dim];
        let mut out_b = vec![0.0f64; spec.dim];

        sample_fresh(spec, &mut out_a, &corr, 0).unwrap();
        sample_fresh(spec, &mut out_b, &corr, 0).unwrap();

        assert_eq!(
            out_a, out_b,
            "SAA with identical inputs must produce bitwise-identical output"
        );
    }

    #[test]
    fn test_saa_different_seeds_differ() {
        let corr_a = identity_correlation(&[1, 2, 3]);
        let corr_b = identity_correlation(&[1, 2, 3]);

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

        sample_fresh(spec_a, &mut out_a, &corr_a, 0).unwrap();
        sample_fresh(spec_b, &mut out_b, &corr_b, 0).unwrap();

        assert_ne!(
            out_a, out_b,
            "SAA with different forward_seed must produce different noise"
        );
    }

    #[test]
    fn test_lhs_produces_finite_noise() {
        let corr = identity_correlation(&[1, 2, 3]);
        let spec = FreshNoiseSpec {
            scenario: 5,
            ..base_spec(NoiseMethod::Lhs)
        };

        let mut output = vec![0.0f64; spec.dim];

        let result = sample_fresh(spec, &mut output, &corr, 0);

        assert!(result.is_ok(), "LHS must return Ok(()), got {result:?}");
        for (i, &v) in output.iter().enumerate() {
            assert!(v.is_finite(), "LHS output[{i}] is not finite: {v}");
        }
    }

    #[test]
    fn test_sobol_produces_finite_noise() {
        let corr = identity_correlation(&[1, 2, 3]);
        let spec = base_spec(NoiseMethod::QmcSobol);

        let mut output = vec![0.0f64; spec.dim];

        let result = sample_fresh(spec, &mut output, &corr, 0);

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
        let spec = FreshNoiseSpec {
            dim: 21_202, // one above the crate's Sobol dimension cap (21_201)
            ..base_spec(NoiseMethod::QmcSobol)
        };

        let mut output = vec![0.0f64; spec.dim];

        let result = sample_fresh(spec, &mut output, &corr, 0);

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
        let spec = base_spec(NoiseMethod::QmcHalton);

        let mut output = vec![0.0f64; spec.dim];

        let result = sample_fresh(spec, &mut output, &corr, 0);

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
        let spec = base_spec(NoiseMethod::Selective);

        let mut output = vec![0.0f64; spec.dim];

        let result = sample_fresh(spec, &mut output, &corr, 0);

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

        let spec = FreshNoiseSpec {
            iteration: 1,
            scenario: 2,
            ..base_spec(NoiseMethod::Selective)
        };
        let spec_saa = FreshNoiseSpec {
            noise_method: NoiseMethod::Saa,
            ..spec
        };

        let mut out_selective = vec![0.0f64; spec.dim];
        let mut out_saa = vec![0.0f64; spec.dim];

        sample_fresh(spec, &mut out_selective, &corr_a, 3).unwrap();
        sample_fresh(spec_saa, &mut out_saa, &corr_b, 3).unwrap();

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
        let table = table_for_spec(spec).expect("Saa never exceeds the Sobol dimension cap");

        fill_uncorrelated(spec, &table, &mut out_a).unwrap();
        fill_uncorrelated(spec, &table, &mut out_b).unwrap();

        assert_eq!(
            out_a, out_b,
            "fill_uncorrelated with identical SAA inputs must produce bit-identical output"
        );
    }

    #[test]
    fn test_fill_uncorrelated_sobol_dim_exceeds_capacity() {
        let spec = FreshNoiseSpec {
            dim: 21_202, // one above the crate's Sobol dimension cap (21_201)
            ..base_spec(NoiseMethod::QmcSobol)
        };

        let result = table_for_spec(spec);

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
            Ok(_) => panic!("expected Err(DimensionExceedsCapacity) but got Ok"),
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
            let table = table_for_spec(spec).expect("dim=3 never exceeds the Sobol dimension cap");

            let result = fill_uncorrelated(spec, &table, &mut output);

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
