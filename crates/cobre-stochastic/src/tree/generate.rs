//! Opening scenario tree generation from pre-decomposed Cholesky factors
//! and deterministic per-opening seeds. Each `(opening_index, stage)` pair
//! receives independent noise with spatial correlation applied in-place.

use cobre_core::{temporal::NoiseMethod, EntityId, Stage};
use rand::RngExt;
use rand_distr::StandardNormal;

use crate::{
    correlation::resolve::DecomposedCorrelation,
    noise::{rng::rng_from_seed, seed::derive_opening_seed},
    tree::{
        lhs::generate_lhs,
        opening_tree::OpeningTree,
        qmc_halton::generate_qmc_halton,
        qmc_sobol::{generate_qmc_sobol, MAX_SOBOL_DIM},
    },
    StochasticError,
};

/// Fill all `n_openings` noise vectors for one stage using SAA (pure Monte Carlo).
fn generate_saa(base_seed: u64, stage: &Stage, n_openings: usize, dim: usize, output: &mut [f64]) {
    for opening_idx in 0..n_openings {
        let start = opening_idx * dim;
        let noise_slice = &mut output[start..start + dim];
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let seed = derive_opening_seed(base_seed, opening_idx as u32, stage.id as u32);
        let mut rng = rng_from_seed(seed);
        for sample in noise_slice.iter_mut() {
            *sample = rng.sample(StandardNormal);
        }
    }
}

/// Generate a fixed opening tree with correlated noise realisations.
///
/// For each stage, all openings are generated together using the configured noise method.
/// SAA, LHS, QMC-Sobol, and QMC-Halton are fully implemented.
/// The `Selective` method returns an error.
///
/// Generation order is stage-major (outer: stages, inner: openings) to support batch methods
/// like LHS that require all openings for a stage simultaneously. Cholesky correlation
/// is applied in-place after noise generation.
///
/// # Errors
///
/// Returns [`StochasticError::UnsupportedNoiseMethod`] if any stage uses [`NoiseMethod::Selective`].
pub fn generate_opening_tree(
    base_seed: u64,
    stages: &[Stage],
    dim: usize,
    correlation: &mut DecomposedCorrelation,
    entity_order: &[EntityId],
) -> Result<OpeningTree, StochasticError> {
    let n_stages = stages.len();
    correlation.resolve_positions(entity_order);

    let openings_per_stage: Vec<usize> = stages
        .iter()
        .map(|s| s.scenario_config.branching_factor)
        .collect();

    let mut stage_offsets = Vec::with_capacity(n_stages);
    let mut running_offset = 0usize;
    for &n_openings in &openings_per_stage {
        stage_offsets.push(running_offset);
        running_offset += n_openings * dim;
    }
    let total_len = running_offset;

    let mut data = vec![0.0f64; total_len];

    for (stage_idx, stage) in stages.iter().enumerate() {
        let n_openings = openings_per_stage[stage_idx];
        let offset = stage_offsets[stage_idx];
        let stage_slice = &mut data[offset..offset + n_openings * dim];

        match stage.scenario_config.noise_method {
            NoiseMethod::Saa => {
                generate_saa(base_seed, stage, n_openings, dim, stage_slice);
            }
            NoiseMethod::Lhs => {
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                generate_lhs(base_seed, stage.id as u32, n_openings, dim, stage_slice);
            }
            NoiseMethod::QmcSobol => {
                if dim > MAX_SOBOL_DIM {
                    return Err(StochasticError::DimensionExceedsCapacity {
                        dim,
                        max_dim: MAX_SOBOL_DIM,
                        method: "sobol".to_string(),
                    });
                }
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                generate_qmc_sobol(base_seed, stage.id as u32, n_openings, dim, stage_slice);
            }
            NoiseMethod::QmcHalton => {
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                generate_qmc_halton(base_seed, stage.id as u32, n_openings, dim, stage_slice);
            }
            NoiseMethod::Selective => {
                return Err(StochasticError::UnsupportedNoiseMethod {
                    method: "selective".to_string(),
                    stage_id: stage.id,
                    reason: "selective/representative sampling is not supported by the opening tree generator; provide a pre-built tree instead".to_string(),
                });
            }
        }

        for opening_idx in 0..n_openings {
            let start = opening_idx * dim;
            let noise = &mut stage_slice[start..start + dim];
            correlation.apply_correlation(stage.id, noise, entity_order);
        }
    }

    Ok(OpeningTree::from_parts(data, openings_per_stage, dim))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use chrono::NaiveDate;
    use cobre_core::{
        scenario::{CorrelationEntity, CorrelationGroup, CorrelationModel, CorrelationProfile},
        temporal::{
            BlockMode, NoiseMethod, ScenarioSourceConfig, StageRiskConfig, StageStateConfig,
        },
        EntityId, Stage,
    };

    use crate::{correlation::resolve::DecomposedCorrelation, StochasticError};

    use super::generate_opening_tree;

    fn make_stage(index: usize, id: i32, branching_factor: usize) -> Stage {
        make_stage_with_method(index, id, branching_factor, NoiseMethod::Saa)
    }

    fn make_stage_with_method(
        index: usize,
        id: i32,
        branching_factor: usize,
        noise_method: NoiseMethod,
    ) -> Stage {
        Stage {
            index,
            id,
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            season_id: Some(0),
            blocks: vec![],
            block_mode: BlockMode::Parallel,
            state_config: StageStateConfig {
                storage: true,
                inflow_lags: false,
            },
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor,
                noise_method,
            },
        }
    }

    fn identity_correlation(entity_ids: &[i32]) -> DecomposedCorrelation {
        let n = entity_ids.len();
        let matrix = (0..n)
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
            method: "cholesky".to_string(),
            profiles,
            schedule: vec![],
        })
        .unwrap()
    }

    fn correlated_correlation(entity_ids: &[i32], rho: f64) -> DecomposedCorrelation {
        let n = entity_ids.len();
        let matrix = (0..n)
            .map(|i| (0..n).map(|j| if i == j { 1.0 } else { rho }).collect())
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
            method: "cholesky".to_string(),
            profiles,
            schedule: vec![],
        })
        .unwrap()
    }

    #[test]
    fn determinism_same_inputs_produce_identical_trees() {
        let stages = vec![make_stage(0, 0, 3), make_stage(1, 1, 3)];
        let mut corr = identity_correlation(&[1, 2]);
        let mut corr2 = identity_correlation(&[1, 2]);
        let entity_order = vec![EntityId(1), EntityId(2)];

        let tree1 = generate_opening_tree(42, &stages, 2, &mut corr, &entity_order).unwrap();
        let tree2 = generate_opening_tree(42, &stages, 2, &mut corr2, &entity_order).unwrap();

        assert_eq!(tree1.len(), tree2.len());
        for s in 0..tree1.n_stages() {
            for o in 0..tree1.n_openings(s) {
                assert_eq!(
                    tree1.opening(s, o),
                    tree2.opening(s, o),
                    "mismatch at stage={s} opening={o}"
                );
            }
        }
    }

    #[test]
    fn opening_0_0_has_correct_length_and_finite_values() {
        let stages = vec![make_stage(0, 0, 3), make_stage(1, 1, 3)];
        let mut corr = identity_correlation(&[1, 2]);
        let entity_order = vec![EntityId(1), EntityId(2)];

        let tree = generate_opening_tree(42, &stages, 2, &mut corr, &entity_order).unwrap();

        let slice = tree.opening(0, 0);
        assert_eq!(slice.len(), 2);
        assert!(
            slice.iter().all(|v| v.is_finite()),
            "non-finite values: {slice:?}"
        );
    }

    #[test]
    fn seed_sensitivity_different_seeds_produce_different_trees() {
        let stages = vec![make_stage(0, 0, 3), make_stage(1, 1, 3)];
        let mut corr = identity_correlation(&[1, 2]);
        let entity_order = vec![EntityId(1), EntityId(2)];

        let tree_a = generate_opening_tree(42, &stages, 2, &mut corr, &entity_order).unwrap();
        let tree_b = generate_opening_tree(99, &stages, 2, &mut corr, &entity_order).unwrap();

        // At least one element must differ; with high probability all will differ.
        let any_differ = (0..tree_a.n_stages()).any(|s| {
            (0..tree_a.n_openings(s)).any(|o| tree_a.opening(s, o) != tree_b.opening(s, o))
        });
        assert!(any_differ, "trees with different seeds should differ");
    }

    #[test]
    fn variable_branching_factors_correct_dimensions() {
        // branching_factors = [2, 3, 1], dim = 2
        // expected total = (2 + 3 + 1) * 2 = 12
        let stages = vec![
            make_stage(0, 0, 2),
            make_stage(1, 1, 3),
            make_stage(2, 2, 1),
        ];
        let mut corr = identity_correlation(&[1, 2]);
        let entity_order = vec![EntityId(1), EntityId(2)];

        let tree = generate_opening_tree(42, &stages, 2, &mut corr, &entity_order).unwrap();

        assert_eq!(tree.n_openings(0), 2, "stage 0");
        assert_eq!(tree.n_openings(1), 3, "stage 1");
        assert_eq!(tree.n_openings(2), 1, "stage 2");
        assert_eq!(tree.len(), 12, "total elements");
    }

    /// Verify `n_stages`, dim, and len for a uniform branching tree.
    #[test]
    fn correct_dimensions_uniform_branching() {
        let stages = vec![
            make_stage(0, 0, 5),
            make_stage(1, 1, 5),
            make_stage(2, 2, 5),
        ];
        let mut corr = identity_correlation(&[1, 2, 3]);
        let entity_order = vec![EntityId(1), EntityId(2), EntityId(3)];

        let tree = generate_opening_tree(7, &stages, 3, &mut corr, &entity_order).unwrap();

        assert_eq!(tree.n_stages(), 3);
        assert_eq!(tree.dim(), 3);
        assert_eq!(tree.len(), 3 * 5 * 3); // n_stages * branching * dim
    }

    /// Identity correlation: each noise vector is an independent N(0,1) sample.
    /// Verify statistical properties over many openings: mean ≈ 0, std ≈ 1.
    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn identity_correlation_noise_has_normal_statistics() {
        let n_openings = 500;
        let stages = vec![make_stage(0, 0, n_openings)];
        let mut corr = identity_correlation(&[1]);
        let entity_order = vec![EntityId(1)];

        let tree = generate_opening_tree(12345, &stages, 1, &mut corr, &entity_order).unwrap();

        // Collect all dim=1 noise values.
        let values: Vec<f64> = (0..n_openings).map(|o| tree.opening(0, o)[0]).collect();

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance =
            values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64;
        let std = variance.sqrt();

        // With 500 samples from N(0,1), |mean| < 0.15 and |std - 1| < 0.15
        // are generous statistical bounds that should hold with overwhelming probability.
        assert!(
            mean.abs() < 0.15,
            "mean too far from 0: {mean:.4} (expected N(0,1))"
        );
        assert!(
            (std - 1.0).abs() < 0.15,
            "std too far from 1: {std:.4} (expected N(0,1))"
        );
    }

    /// All generated values are finite.
    #[test]
    fn all_generated_values_are_finite() {
        let stages = vec![
            make_stage(0, 0, 10),
            make_stage(1, 1, 8),
            make_stage(2, 2, 12),
        ];
        let mut corr = identity_correlation(&[1, 2, 3, 4]);
        let entity_order = vec![EntityId(1), EntityId(2), EntityId(3), EntityId(4)];

        let tree = generate_opening_tree(99, &stages, 4, &mut corr, &entity_order).unwrap();

        for s in 0..tree.n_stages() {
            for o in 0..tree.n_openings(s) {
                for &v in tree.opening(s, o) {
                    assert!(v.is_finite(), "non-finite value at stage={s} opening={o}");
                }
            }
        }
    }

    /// Non-identity correlation: sample correlation should approximate the target.
    ///
    /// With rho=0.8 and a 2x2 correlation matrix, the sample correlation
    /// across many openings should be close to 0.8.
    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn correlated_noise_matches_target_correlation() {
        let n_openings = 2000;
        let stages = vec![make_stage(0, 0, n_openings)];
        let mut corr = correlated_correlation(&[1, 2], 0.8);
        let entity_order = vec![EntityId(1), EntityId(2)];

        let tree = generate_opening_tree(54321, &stages, 2, &mut corr, &entity_order).unwrap();

        // Collect paired samples.
        let pairs: Vec<(f64, f64)> = (0..n_openings)
            .map(|o| {
                let s = tree.opening(0, o);
                (s[0], s[1])
            })
            .collect();

        let n = pairs.len() as f64;
        let mean_x = pairs.iter().map(|(x, _)| x).sum::<f64>() / n;
        let mean_y = pairs.iter().map(|(_, y)| y).sum::<f64>() / n;

        let cov_xy = pairs
            .iter()
            .map(|(x, y)| (x - mean_x) * (y - mean_y))
            .sum::<f64>()
            / (n - 1.0);
        let var_x = pairs.iter().map(|(x, _)| (x - mean_x).powi(2)).sum::<f64>() / (n - 1.0);
        let var_y = pairs.iter().map(|(_, y)| (y - mean_y).powi(2)).sum::<f64>() / (n - 1.0);

        let sample_corr = cov_xy / (var_x.sqrt() * var_y.sqrt());

        // With 2000 samples, the sample correlation should be within ±0.1 of 0.8.
        assert!(
            (sample_corr - 0.8).abs() < 0.1,
            "sample correlation {sample_corr:.4} too far from target 0.8"
        );
    }

    /// Generation order is stage-major: verify that stage 0 opening 0 and
    /// stage 1 opening 0 use different seeds (different `stage.id`), while
    /// stage 0 opening 0 and stage 0 opening 1 also differ (different `opening_idx`).
    #[test]
    #[allow(clippy::float_cmp)]
    fn different_openings_and_stages_produce_different_noise() {
        let stages = vec![make_stage(0, 0, 4), make_stage(1, 1, 4)];
        let mut corr = identity_correlation(&[1]);
        let entity_order = vec![EntityId(1)];

        let tree = generate_opening_tree(0, &stages, 1, &mut corr, &entity_order).unwrap();

        let s0_o0 = tree.opening(0, 0)[0];
        let s0_o1 = tree.opening(0, 1)[0];
        let s1_o0 = tree.opening(1, 0)[0];

        assert_ne!(s0_o0, s0_o1, "same stage, different openings should differ");
        assert_ne!(
            s0_o0, s1_o0,
            "same opening index, different stages should differ"
        );
    }

    /// SAA bitwise compatibility: the stage-major refactor must produce
    /// bit-for-bit identical output for SAA stages.
    ///
    /// Golden values captured from the pre-refactor opening-major implementation
    /// with seed=42, 3 stages (Saa, bf=3), dim=2, identity correlation on [1, 2].
    #[test]
    #[allow(clippy::float_cmp)]
    fn saa_bitwise_compatible_with_pre_refactor_golden_values() {
        let stages = vec![
            make_stage(0, 0, 3),
            make_stage(1, 1, 3),
            make_stage(2, 2, 3),
        ];
        let mut corr = identity_correlation(&[1, 2]);
        let entity_order = vec![EntityId(1), EntityId(2)];

        let tree = generate_opening_tree(42, &stages, 2, &mut corr, &entity_order).unwrap();

        // Golden values from pre-refactor opening-major implementation.
        assert_eq!(
            tree.opening(0, 0)[0],
            4.009_893_649_649_564_6e-1,
            "stage=0 opening=0 dim=0"
        );
        assert_eq!(
            tree.opening(0, 0)[1],
            2.279_255_881_585_980_4e-1,
            "stage=0 opening=0 dim=1"
        );
        assert_eq!(
            tree.opening(0, 1)[0],
            -1.395_412_177_608_524_4,
            "stage=0 opening=1 dim=0"
        );
        assert_eq!(
            tree.opening(0, 1)[1],
            -2.693_936_692_173_674_6e-1,
            "stage=0 opening=1 dim=1"
        );
        assert_eq!(
            tree.opening(0, 2)[0],
            8.337_031_709_056_368_4e-1,
            "stage=0 opening=2 dim=0"
        );
        assert_eq!(
            tree.opening(0, 2)[1],
            -1.619_991_803_182_488_7,
            "stage=0 opening=2 dim=1"
        );
    }

    /// `NoiseMethod::Selective` returns `Err(StochasticError::UnsupportedNoiseMethod)`
    /// with `method == "selective"` and the correct `stage_id`.
    #[test]
    fn selective_returns_error() {
        let stages = vec![
            make_stage(0, 0, 3),
            make_stage_with_method(1, 7, 3, NoiseMethod::Selective),
        ];
        let mut corr = identity_correlation(&[1, 2]);
        let entity_order = vec![EntityId(1), EntityId(2)];

        let result = generate_opening_tree(42, &stages, 2, &mut corr, &entity_order);

        match result {
            Err(StochasticError::UnsupportedNoiseMethod {
                method,
                stage_id,
                reason: _,
            }) => {
                assert_eq!(method, "selective");
                assert_eq!(stage_id, 7);
            }
            Ok(_) => panic!("expected Err but got Ok"),
            Err(other) => panic!("expected UnsupportedNoiseMethod, got {other:?}"),
        }
    }

    /// A stage with `NoiseMethod::Lhs` produces a valid opening tree.
    ///
    /// Verifies that `generate_opening_tree` returns `Ok`, the tree has the
    /// correct number of stages and openings, and all noise values are finite.
    #[test]
    fn test_lhs_stage_produces_tree() {
        let n_openings = 50;
        let dim = 3;
        let stages = vec![make_stage_with_method(0, 0, n_openings, NoiseMethod::Lhs)];
        let mut corr = identity_correlation(&[1, 2, 3]);
        let entity_order = vec![EntityId(1), EntityId(2), EntityId(3)];

        let tree = generate_opening_tree(42, &stages, dim, &mut corr, &entity_order).unwrap();

        assert_eq!(tree.n_stages(), 1, "tree must have 1 stage");
        assert_eq!(tree.n_openings(0), n_openings, "stage 0 opening count");
        assert_eq!(tree.len(), n_openings * dim, "total element count");
        for o in 0..n_openings {
            for &v in tree.opening(0, o) {
                assert!(v.is_finite(), "non-finite value at opening={o}");
            }
        }
    }

    /// A mixed system (stage 0 = Lhs, stage 1 = Saa) produces valid noise for
    /// both stages.
    ///
    /// Also verifies the LHS marginal-uniformity property for stage 0: for each
    /// dimension, `floor(Φ(x_k) * N)` is a permutation of `{0..N-1}`.
    #[test]
    #[allow(
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss
    )]
    fn test_per_stage_method_mixing() {
        let n_openings = 50;
        let dim = 3;
        let stages = vec![
            make_stage_with_method(0, 0, n_openings, NoiseMethod::Lhs),
            make_stage_with_method(1, 1, n_openings, NoiseMethod::Saa),
        ];
        let mut corr = identity_correlation(&[1, 2, 3]);
        let entity_order = vec![EntityId(1), EntityId(2), EntityId(3)];

        let tree = generate_opening_tree(42, &stages, dim, &mut corr, &entity_order).unwrap();

        assert_eq!(tree.n_stages(), 2, "tree must have 2 stages");
        assert_eq!(tree.n_openings(0), n_openings, "stage 0 opening count");
        assert_eq!(tree.n_openings(1), n_openings, "stage 1 opening count");

        // All values for both stages must be finite.
        for s in 0..2 {
            for o in 0..n_openings {
                for &v in tree.opening(s, o) {
                    assert!(v.is_finite(), "non-finite at stage={s} opening={o}");
                }
            }
        }

        // Marginal-uniformity property for the LHS stage (stage 0).
        // Approximate Φ(z) via the Abramowitz & Stegun rational approximation.
        let approx_erf = |x: f64| -> f64 {
            let sign = if x < 0.0 { -1.0_f64 } else { 1.0_f64 };
            let t = 1.0 / (1.0 + 0.3275911 * x.abs());
            let poly = t
                * (0.254_829_592
                    + t * (-0.284_496_736
                        + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
            sign * (1.0 - poly * (-x * x).exp())
        };
        let approx_cdf = |z: f64| -> f64 { 0.5 * (1.0 + approx_erf(z / std::f64::consts::SQRT_2)) };
        let n_f = n_openings as f64;

        for d in 0..dim {
            let mut strata: Vec<usize> = (0..n_openings)
                .map(|k| {
                    let z = tree.opening(0, k)[d];
                    let p = approx_cdf(z);
                    ((p * n_f).floor() as usize).min(n_openings - 1)
                })
                .collect();
            strata.sort_unstable();
            let expected: Vec<usize> = (0..n_openings).collect();
            assert_eq!(
                strata, expected,
                "stage 0 dim {d}: CDF-floor indices not a permutation of 0..{n_openings}"
            );
        }
    }

    /// A stage with `NoiseMethod::QmcSobol` produces a valid opening tree.
    ///
    /// Verifies that `generate_opening_tree` returns `Ok`, the tree has the
    /// correct number of openings and dimensions, and all noise values are finite.
    #[test]
    fn test_sobol_stage_produces_tree() {
        let n_openings = 64;
        let dim = 3;
        let stages = vec![make_stage_with_method(
            0,
            0,
            n_openings,
            NoiseMethod::QmcSobol,
        )];
        let mut corr = identity_correlation(&[1, 2, 3]);
        let entity_order = vec![EntityId(1), EntityId(2), EntityId(3)];

        let tree = generate_opening_tree(42, &stages, dim, &mut corr, &entity_order).unwrap();

        assert_eq!(tree.n_stages(), 1, "tree must have 1 stage");
        assert_eq!(tree.n_openings(0), n_openings, "stage 0 opening count");
        assert_eq!(tree.len(), n_openings * dim, "total element count");
        for o in 0..n_openings {
            for &v in tree.opening(0, o) {
                assert!(v.is_finite(), "non-finite value at opening={o}");
            }
        }
    }

    /// A stage with `NoiseMethod::QmcSobol` and `dim > MAX_SOBOL_DIM` returns
    /// `Err(StochasticError::DimensionExceedsCapacity)` with the correct fields.
    #[test]
    fn test_sobol_dimension_exceeds_capacity() {
        let dim_over = 21_202; // one above MAX_SOBOL_DIM = 21_201
        let stages = vec![make_stage_with_method(0, 0, 4, NoiseMethod::QmcSobol)];
        // Build a minimal identity correlation with a single entity; `dim` is passed
        // separately to `generate_opening_tree`.
        let mut corr = identity_correlation(&[1]);
        let entity_order = vec![EntityId(1)];

        let result = generate_opening_tree(42, &stages, dim_over, &mut corr, &entity_order);

        match result {
            Err(StochasticError::DimensionExceedsCapacity {
                dim,
                max_dim,
                method,
            }) => {
                assert_eq!(dim, 21_202, "dim field");
                assert_eq!(max_dim, 21_201, "max_dim field");
                assert_eq!(method, "sobol", "method field");
            }
            Ok(_) => panic!("expected Err but got Ok"),
            Err(other) => panic!("expected DimensionExceedsCapacity, got {other:?}"),
        }
    }

    /// A mixed system (stage 0 = QmcSobol, stage 1 = Saa) produces valid noise
    /// for both stages with the correct dimensions.
    #[test]
    fn test_sobol_saa_mixing() {
        let n_openings = 32;
        let dim = 2;
        let stages = vec![
            make_stage_with_method(0, 0, n_openings, NoiseMethod::QmcSobol),
            make_stage_with_method(1, 1, n_openings, NoiseMethod::Saa),
        ];
        let mut corr = identity_correlation(&[1, 2]);
        let entity_order = vec![EntityId(1), EntityId(2)];

        let tree = generate_opening_tree(42, &stages, dim, &mut corr, &entity_order).unwrap();

        assert_eq!(tree.n_stages(), 2, "tree must have 2 stages");
        assert_eq!(tree.n_openings(0), n_openings, "stage 0 opening count");
        assert_eq!(tree.n_openings(1), n_openings, "stage 1 opening count");

        for s in 0..2 {
            for o in 0..n_openings {
                for &v in tree.opening(s, o) {
                    assert!(v.is_finite(), "non-finite at stage={s} opening={o}");
                }
            }
        }
    }

    /// A stage with `NoiseMethod::QmcHalton` produces a valid opening tree.
    ///
    /// Verifies that `generate_opening_tree` returns `Ok`, the tree has the
    /// correct number of stages and openings, and all noise values are finite.
    #[test]
    fn test_halton_stage_produces_tree() {
        let n_openings = 64;
        let dim = 3;
        let stages = vec![make_stage_with_method(
            0,
            0,
            n_openings,
            NoiseMethod::QmcHalton,
        )];
        let mut corr = identity_correlation(&[1, 2, 3]);
        let entity_order = vec![EntityId(1), EntityId(2), EntityId(3)];

        let tree = generate_opening_tree(42, &stages, dim, &mut corr, &entity_order).unwrap();

        assert_eq!(tree.n_stages(), 1, "tree must have 1 stage");
        assert_eq!(tree.n_openings(0), n_openings, "stage 0 opening count");
        assert_eq!(tree.len(), n_openings * dim, "total element count");
        for o in 0..n_openings {
            for &v in tree.opening(0, o) {
                assert!(v.is_finite(), "non-finite value at opening={o}");
            }
        }
    }

    /// A mixed system (stage 0 = QmcHalton, stage 1 = Saa) produces valid noise
    /// for both stages with the correct dimensions.
    #[test]
    fn test_halton_saa_mixing() {
        let n_openings = 32;
        let dim = 2;
        let stages = vec![
            make_stage_with_method(0, 0, n_openings, NoiseMethod::QmcHalton),
            make_stage_with_method(1, 1, n_openings, NoiseMethod::Saa),
        ];
        let mut corr = identity_correlation(&[1, 2]);
        let entity_order = vec![EntityId(1), EntityId(2)];

        let tree = generate_opening_tree(42, &stages, dim, &mut corr, &entity_order).unwrap();

        assert_eq!(tree.n_stages(), 2, "tree must have 2 stages");
        assert_eq!(tree.n_openings(0), n_openings, "stage 0 opening count");
        assert_eq!(tree.n_openings(1), n_openings, "stage 1 opening count");

        for s in 0..2 {
            for o in 0..n_openings {
                for &v in tree.opening(s, o) {
                    assert!(v.is_finite(), "non-finite at stage={s} opening={o}");
                }
            }
        }
    }
}
