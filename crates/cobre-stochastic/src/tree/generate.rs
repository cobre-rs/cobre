//! Opening scenario tree generation from pre-decomposed spectral factors
//! and deterministic per-opening seeds. Each `(opening_index, stage)` pair
//! receives independent noise with spatial correlation applied in-place.

use cobre_core::{EntityId, Stage, temporal::NoiseMethod};
use rand::RngExt;
use rand_distr::StandardNormal;

use crate::{
    StochasticError,
    correlation::resolve::DecomposedCorrelation,
    noise::{rng::rng_from_seed, seed::derive_opening_seed},
    tree::{
        lhs::generate_lhs,
        opening_tree::OpeningTree,
        qmc_halton::generate_qmc_halton,
        qmc_sobol::{MAX_SOBOL_DIM, generate_qmc_sobol},
    },
};

/// Per-class entity counts for the noise dimension.
///
/// The canonical noise vector layout is `[hydros | load buses | NCS entities]`.
/// These counts split the flat noise vector into per-class segments for independent
/// spectral correlation application within each entity class.
///
/// # Invariant
///
/// `n_hydros + n_load_buses + n_ncs` must equal the `dim` argument passed to
/// `generate_opening_tree`.
#[derive(Debug, Clone, Copy)]
pub struct ClassDimensions {
    /// Number of hydro entities (inflow class) in the noise vector.
    pub n_hydros: usize,
    /// Number of stochastic load bus entities (load class) in the noise vector.
    pub n_load_buses: usize,
    /// Number of stochastic NCS entities (ncs class) in the noise vector.
    pub n_ncs: usize,
}

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
/// like LHS that require all openings for a stage simultaneously. spectral correlation
/// is applied per class (inflow, load, ncs) in-place after noise generation.
///
/// The `entity_order` slice must have layout `[hydros | load buses | NCS entities]` and
/// `dims.n_hydros + dims.n_load_buses + dims.n_ncs` must equal `dim`.
///
/// # Errors
///
/// Returns [`StochasticError::UnsupportedNoiseMethod`] if any stage uses [`NoiseMethod::Selective`].
pub fn generate_opening_tree(
    base_seed: u64,
    stages: &[Stage],
    dim: usize,
    correlation: &DecomposedCorrelation,
    entity_order: &[EntityId],
    dims: ClassDimensions,
) -> Result<OpeningTree, StochasticError> {
    let n_stages = stages.len();

    let inflow_order = &entity_order[..dims.n_hydros];
    let load_order = &entity_order[dims.n_hydros..dims.n_hydros + dims.n_load_buses];
    let ncs_order = &entity_order[dims.n_hydros + dims.n_load_buses..];

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
            let (inflow_noise, rest) = noise.split_at_mut(dims.n_hydros);
            let (load_noise, ncs_noise) = rest.split_at_mut(dims.n_load_buses);
            correlation.apply_correlation_for_class(stage.id, inflow_noise, inflow_order, "inflow");
            correlation.apply_correlation_for_class(stage.id, load_noise, load_order, "load");
            correlation.apply_correlation_for_class(stage.id, ncs_noise, ncs_order, "ncs");
        }
    }

    Ok(OpeningTree::from_parts(data, openings_per_stage, dim))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use chrono::NaiveDate;
    use cobre_core::{
        EntityId, Stage,
        scenario::{CorrelationEntity, CorrelationGroup, CorrelationModel, CorrelationProfile},
        temporal::{
            BlockMode, NoiseMethod, ScenarioSourceConfig, StageRiskConfig, StageStateConfig,
        },
    };

    use crate::{StochasticError, correlation::resolve::DecomposedCorrelation};

    use super::{ClassDimensions, generate_opening_tree};

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
            method: "spectral".to_string(),
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
            method: "spectral".to_string(),
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

        let dims = ClassDimensions {
            n_hydros: 2,
            n_load_buses: 0,
            n_ncs: 0,
        };
        let tree1 = generate_opening_tree(42, &stages, 2, &mut corr, &entity_order, dims).unwrap();
        let tree2 = generate_opening_tree(42, &stages, 2, &mut corr2, &entity_order, dims).unwrap();

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
        let dims = ClassDimensions {
            n_hydros: 2,
            n_load_buses: 0,
            n_ncs: 0,
        };

        let tree = generate_opening_tree(42, &stages, 2, &mut corr, &entity_order, dims).unwrap();

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
        let dims = ClassDimensions {
            n_hydros: 2,
            n_load_buses: 0,
            n_ncs: 0,
        };

        let tree_a = generate_opening_tree(42, &stages, 2, &mut corr, &entity_order, dims).unwrap();
        let tree_b = generate_opening_tree(99, &stages, 2, &mut corr, &entity_order, dims).unwrap();

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
        let dims = ClassDimensions {
            n_hydros: 2,
            n_load_buses: 0,
            n_ncs: 0,
        };

        let tree = generate_opening_tree(42, &stages, 2, &mut corr, &entity_order, dims).unwrap();

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
        let dims = ClassDimensions {
            n_hydros: 3,
            n_load_buses: 0,
            n_ncs: 0,
        };

        let tree = generate_opening_tree(7, &stages, 3, &mut corr, &entity_order, dims).unwrap();

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
        let dims = ClassDimensions {
            n_hydros: 1,
            n_load_buses: 0,
            n_ncs: 0,
        };

        let tree =
            generate_opening_tree(12345, &stages, 1, &mut corr, &entity_order, dims).unwrap();

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
        let dims = ClassDimensions {
            n_hydros: 4,
            n_load_buses: 0,
            n_ncs: 0,
        };

        let tree = generate_opening_tree(99, &stages, 4, &mut corr, &entity_order, dims).unwrap();

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
        let dims = ClassDimensions {
            n_hydros: 2,
            n_load_buses: 0,
            n_ncs: 0,
        };

        let tree =
            generate_opening_tree(54321, &stages, 2, &mut corr, &entity_order, dims).unwrap();

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
        let dims = ClassDimensions {
            n_hydros: 1,
            n_load_buses: 0,
            n_ncs: 0,
        };

        let tree = generate_opening_tree(0, &stages, 1, &mut corr, &entity_order, dims).unwrap();

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
        let dims = ClassDimensions {
            n_hydros: 2,
            n_load_buses: 0,
            n_ncs: 0,
        };

        let tree = generate_opening_tree(42, &stages, 2, &mut corr, &entity_order, dims).unwrap();

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
            8.337_031_709_056_368e-1,
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
        let dims = ClassDimensions {
            n_hydros: 2,
            n_load_buses: 0,
            n_ncs: 0,
        };

        let result = generate_opening_tree(42, &stages, 2, &mut corr, &entity_order, dims);

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
        let dims = ClassDimensions {
            n_hydros: 3,
            n_load_buses: 0,
            n_ncs: 0,
        };

        let tree = generate_opening_tree(42, &stages, dim, &mut corr, &entity_order, dims).unwrap();

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
        let dims = ClassDimensions {
            n_hydros: 3,
            n_load_buses: 0,
            n_ncs: 0,
        };

        let tree = generate_opening_tree(42, &stages, dim, &mut corr, &entity_order, dims).unwrap();

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
            let t = 1.0 / (1.0 + 0.327_591_1 * x.abs());
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
        let dims = ClassDimensions {
            n_hydros: 3,
            n_load_buses: 0,
            n_ncs: 0,
        };

        let tree = generate_opening_tree(42, &stages, dim, &mut corr, &entity_order, dims).unwrap();

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
        let dims = ClassDimensions {
            n_hydros: 1,
            n_load_buses: 0,
            n_ncs: 0,
        };

        let result = generate_opening_tree(42, &stages, dim_over, &mut corr, &entity_order, dims);

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

    /// A mixed system (stage 0 = `QmcSobol`, stage 1 = Saa) produces valid noise
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
        let dims = ClassDimensions {
            n_hydros: 2,
            n_load_buses: 0,
            n_ncs: 0,
        };

        let tree = generate_opening_tree(42, &stages, dim, &mut corr, &entity_order, dims).unwrap();

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
        let dims = ClassDimensions {
            n_hydros: 3,
            n_load_buses: 0,
            n_ncs: 0,
        };

        let tree = generate_opening_tree(42, &stages, dim, &mut corr, &entity_order, dims).unwrap();

        assert_eq!(tree.n_stages(), 1, "tree must have 1 stage");
        assert_eq!(tree.n_openings(0), n_openings, "stage 0 opening count");
        assert_eq!(tree.len(), n_openings * dim, "total element count");
        for o in 0..n_openings {
            for &v in tree.opening(0, o) {
                assert!(v.is_finite(), "non-finite value at opening={o}");
            }
        }
    }

    /// A mixed system (stage 0 = `QmcHalton`, stage 1 = Saa) produces valid noise
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
        let dims = ClassDimensions {
            n_hydros: 2,
            n_load_buses: 0,
            n_ncs: 0,
        };

        let tree = generate_opening_tree(42, &stages, dim, &mut corr, &entity_order, dims).unwrap();

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

    /// Per-class spectral application produces bit-identical results to full-vector
    /// spectral when the correlation groups are all same-type (block-diagonal L).
    ///
    /// Generates a tree with 2 hydros (rho=0.8 inflow group) and 0 load/NCS
    /// using the new per-class path, then verifies the output against expected
    /// values that were confirmed identical to the old full-vector path.
    ///
    /// Acceptance criterion: both produce the same correlated noise because L is
    /// block-diagonal under same-type groups (ticket-010).
    #[test]
    #[allow(clippy::float_cmp)]
    fn test_per_class_tree_matches_full_vector_tree() {
        use cobre_core::scenario::{
            CorrelationEntity, CorrelationGroup, CorrelationModel, CorrelationProfile,
        };
        use std::collections::BTreeMap;

        let rho = 0.8_f64;
        let entity_ids = [EntityId(1), EntityId(2)];
        let n = entity_ids.len();
        let matrix: Vec<Vec<f64>> = (0..n)
            .map(|i| (0..n).map(|j| if i == j { 1.0 } else { rho }).collect())
            .collect();
        let mut profiles = BTreeMap::new();
        profiles.insert(
            "default".to_string(),
            CorrelationProfile {
                groups: vec![CorrelationGroup {
                    name: "hydro_group".to_string(),
                    entities: entity_ids
                        .iter()
                        .map(|&id| CorrelationEntity {
                            entity_type: "inflow".to_string(),
                            id,
                        })
                        .collect(),
                    matrix,
                }],
            },
        );
        let corr_model = CorrelationModel {
            method: "spectral".to_string(),
            profiles,
            schedule: vec![],
        };

        let stages = vec![make_stage(0, 0, 5), make_stage(1, 1, 5)];
        let entity_order = vec![EntityId(1), EntityId(2)];
        // All entities are inflow: n_hydros=2, n_load_buses=0, n_ncs=0.
        let dims = ClassDimensions {
            n_hydros: 2,
            n_load_buses: 0,
            n_ncs: 0,
        };

        // Build two independent DecomposedCorrelation instances from the same model.
        let mut corr_per_class = DecomposedCorrelation::build(&corr_model).unwrap();
        let mut corr_full = DecomposedCorrelation::build(&corr_model).unwrap();

        // Generate tree via the new per-class path (the only path in generate_opening_tree).
        let tree_per_class =
            generate_opening_tree(77, &stages, 2, &mut corr_per_class, &entity_order, dims)
                .unwrap();

        // Reproduce the old full-vector path manually: generate noise then call
        // apply_correlation on the full vector (not per-class).
        use crate::noise::{rng::rng_from_seed, seed::derive_opening_seed};
        use rand::RngExt;
        use rand_distr::StandardNormal;

        corr_full.resolve_positions(&entity_order);

        let n_stages = stages.len();
        let dim = 2_usize;
        let base_seed = 77_u64;
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
        let mut data_full = vec![0.0f64; running_offset];
        for (stage_idx, stage) in stages.iter().enumerate() {
            let n_openings = openings_per_stage[stage_idx];
            let offset = stage_offsets[stage_idx];
            let stage_slice = &mut data_full[offset..offset + n_openings * dim];
            for opening_idx in 0..n_openings {
                let start = opening_idx * dim;
                let noise = &mut stage_slice[start..start + dim];
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let seed = derive_opening_seed(base_seed, opening_idx as u32, stage.id as u32);
                let mut rng = rng_from_seed(seed);
                for sample in noise.iter_mut() {
                    *sample = rng.sample(StandardNormal);
                }
                corr_full.apply_correlation(stage.id, noise, &entity_order);
            }
        }

        // Compare every value: per-class and full-vector must be bit-identical.
        for stage_idx in 0..n_stages {
            let n_openings = openings_per_stage[stage_idx];
            for opening_idx in 0..n_openings {
                let per_class = tree_per_class.opening(stage_idx, opening_idx);
                let offset = stage_offsets[stage_idx] + opening_idx * dim;
                let full = &data_full[offset..offset + dim];
                assert_eq!(
                    per_class, full,
                    "stage={stage_idx} opening={opening_idx}: per-class differs from full-vector"
                );
            }
        }
    }

    /// Same as [`test_per_class_tree_matches_full_vector_tree`] but with a
    /// multi-class layout: 2 hydros (correlated at rho=0.8) and 1 load bus
    /// (identity group). Verifies that the per-class spectral path produces
    /// bit-identical results to the full-vector path for mixed-class scenarios.
    #[test]
    fn test_per_class_tree_matches_full_vector_multi_class() {
        let rho = 0.8_f64;
        let cholesky = vec![vec![1.0, rho], vec![rho, 1.0]];
        let inflow_group = CorrelationGroup {
            name: "hydro_inflow".to_string(),
            entities: vec![
                CorrelationEntity {
                    id: EntityId(1),
                    entity_type: "inflow".to_string(),
                },
                CorrelationEntity {
                    id: EntityId(2),
                    entity_type: "inflow".to_string(),
                },
            ],
            matrix: cholesky,
        };
        let load_group = CorrelationGroup {
            name: "load_bus".to_string(),
            entities: vec![CorrelationEntity {
                id: EntityId(3),
                entity_type: "load".to_string(),
            }],
            matrix: vec![vec![1.0]],
        };
        let profiles = BTreeMap::from([(
            "default".to_string(),
            CorrelationProfile {
                groups: vec![inflow_group, load_group],
            },
        )]);
        let corr_model = CorrelationModel {
            method: "spectral".to_string(),
            profiles,
            schedule: vec![],
        };

        let stages = vec![make_stage(0, 0, 5), make_stage(1, 1, 5)];
        let entity_order = vec![EntityId(1), EntityId(2), EntityId(3)];
        let dims = ClassDimensions {
            n_hydros: 2,
            n_load_buses: 1,
            n_ncs: 0,
        };

        let mut corr_per_class = DecomposedCorrelation::build(&corr_model).unwrap();
        let mut corr_full = DecomposedCorrelation::build(&corr_model).unwrap();

        let tree_per_class =
            generate_opening_tree(77, &stages, 3, &mut corr_per_class, &entity_order, dims)
                .unwrap();

        use crate::noise::{rng::rng_from_seed, seed::derive_opening_seed};
        use rand::RngExt;
        use rand_distr::StandardNormal;

        corr_full.resolve_positions(&entity_order);

        let n_stages = stages.len();
        let dim = 3_usize;
        let base_seed = 77_u64;
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
        let mut data_full = vec![0.0f64; running_offset];
        for (stage_idx, stage) in stages.iter().enumerate() {
            let n_openings = openings_per_stage[stage_idx];
            let offset = stage_offsets[stage_idx];
            let stage_slice = &mut data_full[offset..offset + n_openings * dim];
            for opening_idx in 0..n_openings {
                let start = opening_idx * dim;
                let noise = &mut stage_slice[start..start + dim];
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let seed = derive_opening_seed(base_seed, opening_idx as u32, stage.id as u32);
                let mut rng = rng_from_seed(seed);
                for sample in noise.iter_mut() {
                    *sample = rng.sample(StandardNormal);
                }
                corr_full.apply_correlation(stage.id, noise, &entity_order);
            }
        }

        for stage_idx in 0..n_stages {
            let n_openings = openings_per_stage[stage_idx];
            for opening_idx in 0..n_openings {
                let per_class = tree_per_class.opening(stage_idx, opening_idx);
                let offset = stage_offsets[stage_idx] + opening_idx * dim;
                let full = &data_full[offset..offset + dim];
                assert_eq!(
                    per_class, full,
                    "stage={stage_idx} opening={opening_idx}: per-class differs from full-vector (multi-class)"
                );
            }
        }
    }

    /// Per-class vs full-vector equivalence for `NoiseMethod::Lhs`.
    ///
    /// Generates a tree with 2 correlated hydros (rho=0.8) using LHS noise via
    /// `generate_opening_tree`, then manually reproduces the full-vector path
    /// (call `generate_lhs` on the full stage batch, then apply full-vector
    /// spectral per opening) and asserts bit-identical results.
    ///
    /// The LHS batch generator fills the entire `[n_openings × dim]` buffer in
    /// one call, after which each opening's noise slice is transformed by the
    /// spectral factor. This mirrors the per-class path, which splits the noise
    /// into class segments and applies each class's spectral factor separately.
    /// Under a same-type correlation group (block-diagonal L) the two paths must
    /// produce identical output (ticket-m9).
    #[test]
    #[allow(clippy::float_cmp)]
    fn test_per_class_tree_matches_full_vector_tree_lhs() {
        let rho = 0.8_f64;
        let entity_ids = [EntityId(1), EntityId(2)];
        let n = entity_ids.len();
        let matrix: Vec<Vec<f64>> = (0..n)
            .map(|i| (0..n).map(|j| if i == j { 1.0 } else { rho }).collect())
            .collect();
        let mut profiles = BTreeMap::new();
        profiles.insert(
            "default".to_string(),
            CorrelationProfile {
                groups: vec![CorrelationGroup {
                    name: "hydro_group".to_string(),
                    entities: entity_ids
                        .iter()
                        .map(|&id| CorrelationEntity {
                            entity_type: "inflow".to_string(),
                            id,
                        })
                        .collect(),
                    matrix,
                }],
            },
        );
        let corr_model = CorrelationModel {
            method: "spectral".to_string(),
            profiles,
            schedule: vec![],
        };

        let n_openings = 8;
        let dim = 2_usize;
        let base_seed = 77_u64;
        let stages = vec![
            make_stage_with_method(0, 0, n_openings, NoiseMethod::Lhs),
            make_stage_with_method(1, 1, n_openings, NoiseMethod::Lhs),
        ];
        let entity_order = vec![EntityId(1), EntityId(2)];
        let dims = ClassDimensions {
            n_hydros: 2,
            n_load_buses: 0,
            n_ncs: 0,
        };

        let mut corr_per_class = DecomposedCorrelation::build(&corr_model).unwrap();
        let mut corr_full = DecomposedCorrelation::build(&corr_model).unwrap();

        // Generate tree via the per-class path.
        let tree_per_class = generate_opening_tree(
            base_seed,
            &stages,
            dim,
            &mut corr_per_class,
            &entity_order,
            dims,
        )
        .unwrap();

        // Reproduce the full-vector path: generate all openings for a stage
        // in one LHS batch, then apply full-vector spectral per opening.
        use crate::tree::lhs::generate_lhs;
        corr_full.resolve_positions(&entity_order);

        let n_stages = stages.len();
        let openings_per_stage: Vec<usize> = stages
            .iter()
            .map(|s| s.scenario_config.branching_factor)
            .collect();
        let mut stage_offsets = Vec::with_capacity(n_stages);
        let mut running_offset = 0usize;
        for &n in &openings_per_stage {
            stage_offsets.push(running_offset);
            running_offset += n * dim;
        }
        let mut data_full = vec![0.0f64; running_offset];

        for (stage_idx, stage) in stages.iter().enumerate() {
            let n = openings_per_stage[stage_idx];
            let offset = stage_offsets[stage_idx];
            let stage_slice = &mut data_full[offset..offset + n * dim];
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            generate_lhs(base_seed, stage.id as u32, n, dim, stage_slice);
            for opening_idx in 0..n {
                let start = opening_idx * dim;
                let noise = &mut stage_slice[start..start + dim];
                corr_full.apply_correlation(stage.id, noise, &entity_order);
            }
        }

        for stage_idx in 0..n_stages {
            let n = openings_per_stage[stage_idx];
            for opening_idx in 0..n {
                let per_class = tree_per_class.opening(stage_idx, opening_idx);
                let offset = stage_offsets[stage_idx] + opening_idx * dim;
                let full = &data_full[offset..offset + dim];
                assert_eq!(
                    per_class, full,
                    "Lhs: stage={stage_idx} opening={opening_idx}: per-class differs from full-vector"
                );
            }
        }
    }

    /// Per-class vs full-vector equivalence for `NoiseMethod::QmcHalton`.
    ///
    /// Same structure as `test_per_class_tree_matches_full_vector_tree_lhs` but
    /// uses `generate_qmc_halton` for the full-vector reproduction path and
    /// `NoiseMethod::QmcHalton` in the stage configuration.
    #[test]
    #[allow(clippy::float_cmp)]
    fn test_per_class_tree_matches_full_vector_tree_halton() {
        let rho = 0.8_f64;
        let entity_ids = [EntityId(1), EntityId(2)];
        let n = entity_ids.len();
        let matrix: Vec<Vec<f64>> = (0..n)
            .map(|i| (0..n).map(|j| if i == j { 1.0 } else { rho }).collect())
            .collect();
        let mut profiles = BTreeMap::new();
        profiles.insert(
            "default".to_string(),
            CorrelationProfile {
                groups: vec![CorrelationGroup {
                    name: "hydro_group".to_string(),
                    entities: entity_ids
                        .iter()
                        .map(|&id| CorrelationEntity {
                            entity_type: "inflow".to_string(),
                            id,
                        })
                        .collect(),
                    matrix,
                }],
            },
        );
        let corr_model = CorrelationModel {
            method: "spectral".to_string(),
            profiles,
            schedule: vec![],
        };

        let n_openings = 8;
        let dim = 2_usize;
        let base_seed = 77_u64;
        let stages = vec![
            make_stage_with_method(0, 0, n_openings, NoiseMethod::QmcHalton),
            make_stage_with_method(1, 1, n_openings, NoiseMethod::QmcHalton),
        ];
        let entity_order = vec![EntityId(1), EntityId(2)];
        let dims = ClassDimensions {
            n_hydros: 2,
            n_load_buses: 0,
            n_ncs: 0,
        };

        let mut corr_per_class = DecomposedCorrelation::build(&corr_model).unwrap();
        let mut corr_full = DecomposedCorrelation::build(&corr_model).unwrap();

        // Generate tree via the per-class path.
        let tree_per_class = generate_opening_tree(
            base_seed,
            &stages,
            dim,
            &mut corr_per_class,
            &entity_order,
            dims,
        )
        .unwrap();

        // Reproduce the full-vector path: generate all openings for a stage
        // in one QMC-Halton batch, then apply full-vector spectral per opening.
        use crate::tree::qmc_halton::generate_qmc_halton;
        corr_full.resolve_positions(&entity_order);

        let n_stages = stages.len();
        let openings_per_stage: Vec<usize> = stages
            .iter()
            .map(|s| s.scenario_config.branching_factor)
            .collect();
        let mut stage_offsets = Vec::with_capacity(n_stages);
        let mut running_offset = 0usize;
        for &n in &openings_per_stage {
            stage_offsets.push(running_offset);
            running_offset += n * dim;
        }
        let mut data_full = vec![0.0f64; running_offset];

        for (stage_idx, stage) in stages.iter().enumerate() {
            let n = openings_per_stage[stage_idx];
            let offset = stage_offsets[stage_idx];
            let stage_slice = &mut data_full[offset..offset + n * dim];
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            generate_qmc_halton(base_seed, stage.id as u32, n, dim, stage_slice);
            for opening_idx in 0..n {
                let start = opening_idx * dim;
                let noise = &mut stage_slice[start..start + dim];
                corr_full.apply_correlation(stage.id, noise, &entity_order);
            }
        }

        for stage_idx in 0..n_stages {
            let n = openings_per_stage[stage_idx];
            for opening_idx in 0..n {
                let per_class = tree_per_class.opening(stage_idx, opening_idx);
                let offset = stage_offsets[stage_idx] + opening_idx * dim;
                let full = &data_full[offset..offset + dim];
                assert_eq!(
                    per_class, full,
                    "QmcHalton: stage={stage_idx} opening={opening_idx}: per-class differs from full-vector"
                );
            }
        }
    }
}
