//! Opening scenario tree generation.
//!
//! Constructs the opening scenario tree from pre-decomposed Cholesky
//! correlation factors and deterministic per-opening seeds. Each
//! `(opening_index, stage)` pair receives an independent noise vector
//! whose spatial correlation is applied in-place via the Cholesky transform.
//!
//! Deterministic seeds are derived via SipHash-1-3, enabling communication-free
//! parallel generation where each compute node independently generates its
//! assigned subset of openings without synchronisation (deterministic SipHash-1-3
//! seeds for communication-free parallel noise generation).

use cobre_core::{EntityId, Stage};
use rand::RngExt;
use rand_distr::StandardNormal;

use crate::{
    correlation::resolve::DecomposedCorrelation,
    noise::{rng::rng_from_seed, seed::derive_opening_seed},
    tree::opening_tree::OpeningTree,
};

/// Generate a fixed opening tree with correlated noise realisations.
///
/// For each `(opening_index, stage)` pair:
/// 1. Derive a deterministic seed via SipHash-1-3.
/// 2. Initialise a `Pcg64` RNG from the derived seed.
/// 3. Draw `dim` independent N(0,1) samples.
/// 4. Apply the Cholesky correlation transform for the active profile.
///
/// The generation order is opening-major (outer loop: openings, inner
/// loop: stages) to align with the parallel generation model where each
/// rank generates a contiguous block of openings.
///
/// # Arguments
///
/// * `base_seed` — Base seed from scenario source configuration.
/// * `stages` — Study stages (provides branching factors and stage IDs).
/// * `dim` — Number of entities (random variables per noise vector).
/// * `correlation` — Pre-decomposed correlation data (Cholesky factors).
/// * `entity_order` — Canonical entity IDs for correlation mapping.
///
/// # Returns
///
/// An [`OpeningTree`] with all noise values populated.
///
/// # Panics
///
/// Panics if `stages` is empty or if `dim` is zero.
///
/// # Examples
///
/// ```
/// use std::collections::BTreeMap;
/// use cobre_core::{EntityId, Stage, scenario::{
///     CorrelationEntity, CorrelationGroup, CorrelationModel, CorrelationProfile,
/// }};
/// use cobre_stochastic::correlation::resolve::DecomposedCorrelation;
/// use cobre_stochastic::tree::generate::generate_opening_tree;
///
/// # fn make_stage(index: usize, id: i32, branching_factor: usize) -> Stage {
/// #     use chrono::NaiveDate;
/// #     use cobre_core::temporal::{
/// #         BlockMode, NoiseMethod, ScenarioSourceConfig, StageRiskConfig, StageStateConfig,
/// #     };
/// #     Stage {
/// #         index,
/// #         id,
/// #         start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
/// #         end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
/// #         season_id: Some(0),
/// #         blocks: vec![],
/// #         block_mode: BlockMode::Parallel,
/// #         state_config: StageStateConfig { storage: true, inflow_lags: false },
/// #         risk_config: StageRiskConfig::Expectation,
/// #         scenario_config: ScenarioSourceConfig {
/// #             branching_factor,
/// #             noise_method: NoiseMethod::Saa,
/// #         },
/// #     }
/// # }
/// let stages = vec![make_stage(0, 0, 3), make_stage(1, 1, 3)];
///
/// let mut profiles = BTreeMap::new();
/// profiles.insert("default".to_string(), CorrelationProfile {
///     groups: vec![CorrelationGroup {
///         name: "g1".to_string(),
///         entities: vec![
///             CorrelationEntity { entity_type: "inflow".to_string(), id: EntityId(1) },
///             CorrelationEntity { entity_type: "inflow".to_string(), id: EntityId(2) },
///         ],
///         matrix: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
///     }],
/// });
/// let model = CorrelationModel {
///     method: "cholesky".to_string(), profiles, schedule: vec![],
/// };
/// let mut correlation = DecomposedCorrelation::build(&model).unwrap();
/// let entity_order = vec![EntityId(1), EntityId(2)];
///
/// let tree = generate_opening_tree(42, &stages, 2, &mut correlation, &entity_order);
/// assert_eq!(tree.n_stages(), 2);
/// assert_eq!(tree.n_openings(0), 3);
/// ```
#[must_use]
pub fn generate_opening_tree(
    base_seed: u64,
    stages: &[Stage],
    dim: usize,
    correlation: &mut DecomposedCorrelation,
    entity_order: &[EntityId],
) -> OpeningTree {
    let n_stages = stages.len();

    // Pre-compute entity position indices once, eliminating per-call O(n)
    // linear scans and Vec allocations inside apply_correlation.
    correlation.resolve_positions(entity_order);

    // Extract per-stage branching factors.
    let openings_per_stage: Vec<usize> = stages
        .iter()
        .map(|s| s.scenario_config.branching_factor)
        .collect();

    // Maximum number of openings across all stages (needed for the outer loop).
    let max_openings = openings_per_stage.iter().copied().max().unwrap_or(0);

    // Compute per-stage offsets into the flat data array (stage-major layout).
    // stage_offsets[t] is the index of the first element of stage t.
    let mut stage_offsets = Vec::with_capacity(n_stages);
    let mut running_offset = 0usize;
    for &n_openings in &openings_per_stage {
        stage_offsets.push(running_offset);
        running_offset += n_openings * dim;
    }
    let total_len = running_offset;

    // Allocate the backing array in one shot.
    let mut data = vec![0.0f64; total_len];

    // Generate noise: opening-major order (outer: openings, inner: stages).
    // This matches the parallel model where each rank owns a contiguous block
    // of opening indices and iterates stages in the inner loop.
    for opening_idx in 0..max_openings {
        for (stage_idx, stage) in stages.iter().enumerate() {
            // Variable branching: skip openings beyond this stage's branching factor.
            if opening_idx >= openings_per_stage[stage_idx] {
                continue;
            }

            let offset = stage_offsets[stage_idx] + opening_idx * dim;
            let noise_slice = &mut data[offset..offset + dim];

            // Derive seed using stage.id (stable identifier, not array position).
            // u32 casts: opening_idx is << 2^32; stage.id is reinterpreted as domain ID by SipHash-1-3.
            // Pre-study stages (negative id) are excluded by the caller.
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let seed = derive_opening_seed(base_seed, opening_idx as u32, stage.id as u32);

            let mut rng = rng_from_seed(seed);
            for sample in noise_slice.iter_mut() {
                *sample = rng.sample(StandardNormal);
            }
            correlation.apply_correlation(stage.id, noise_slice, entity_order);
        }
    }

    OpeningTree::from_parts(data, openings_per_stage, dim)
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

    use crate::correlation::resolve::DecomposedCorrelation;

    use super::generate_opening_tree;

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn make_stage(index: usize, id: i32, branching_factor: usize) -> Stage {
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
                noise_method: NoiseMethod::Saa,
            },
        }
    }

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
        let model = CorrelationModel {
            method: "cholesky".to_string(),
            profiles,
            schedule: vec![],
        };
        DecomposedCorrelation::build(&model).unwrap()
    }

    fn correlated_correlation(entity_ids: &[i32], rho: f64) -> DecomposedCorrelation {
        let n = entity_ids.len();
        let matrix: Vec<Vec<f64>> = (0..n)
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
        let model = CorrelationModel {
            method: "cholesky".to_string(),
            profiles,
            schedule: vec![],
        };
        DecomposedCorrelation::build(&model).unwrap()
    }

    // -----------------------------------------------------------------------
    // Acceptance criteria tests
    // -----------------------------------------------------------------------

    /// AC1: determinism — same inputs produce bit-for-bit identical output.
    #[test]
    fn determinism_same_inputs_produce_identical_trees() {
        let stages = vec![make_stage(0, 0, 3), make_stage(1, 1, 3)];
        let mut corr = identity_correlation(&[1, 2]);
        let mut corr2 = identity_correlation(&[1, 2]);
        let entity_order = vec![EntityId(1), EntityId(2)];

        let tree1 = generate_opening_tree(42, &stages, 2, &mut corr, &entity_order);
        let tree2 = generate_opening_tree(42, &stages, 2, &mut corr2, &entity_order);

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

    /// AC2: the returned slice for opening(0, 0) has length 2 and finite values.
    #[test]
    fn opening_0_0_has_correct_length_and_finite_values() {
        let stages = vec![make_stage(0, 0, 3), make_stage(1, 1, 3)];
        let mut corr = identity_correlation(&[1, 2]);
        let entity_order = vec![EntityId(1), EntityId(2)];

        let tree = generate_opening_tree(42, &stages, 2, &mut corr, &entity_order);

        let slice = tree.opening(0, 0);
        assert_eq!(slice.len(), 2);
        assert!(
            slice.iter().all(|v| v.is_finite()),
            "non-finite values: {slice:?}"
        );
    }

    /// AC3: different base seeds produce different trees.
    #[test]
    fn seed_sensitivity_different_seeds_produce_different_trees() {
        let stages = vec![make_stage(0, 0, 3), make_stage(1, 1, 3)];
        let mut corr = identity_correlation(&[1, 2]);
        let entity_order = vec![EntityId(1), EntityId(2)];

        let tree_a = generate_opening_tree(42, &stages, 2, &mut corr, &entity_order);
        let tree_b = generate_opening_tree(99, &stages, 2, &mut corr, &entity_order);

        // At least one element must differ; with high probability all will differ.
        let any_differ = (0..tree_a.n_stages()).any(|s| {
            (0..tree_a.n_openings(s)).any(|o| tree_a.opening(s, o) != tree_b.opening(s, o))
        });
        assert!(any_differ, "trees with different seeds should differ");
    }

    /// AC4: variable branching — correct `n_openings` per stage and total len.
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

        let tree = generate_opening_tree(42, &stages, 2, &mut corr, &entity_order);

        assert_eq!(tree.n_openings(0), 2, "stage 0");
        assert_eq!(tree.n_openings(1), 3, "stage 1");
        assert_eq!(tree.n_openings(2), 1, "stage 2");
        assert_eq!(tree.len(), 12, "total elements");
    }

    // -----------------------------------------------------------------------
    // Additional unit tests
    // -----------------------------------------------------------------------

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

        let tree = generate_opening_tree(7, &stages, 3, &mut corr, &entity_order);

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

        let tree = generate_opening_tree(12345, &stages, 1, &mut corr, &entity_order);

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

        let tree = generate_opening_tree(99, &stages, 4, &mut corr, &entity_order);

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

        let tree = generate_opening_tree(54321, &stages, 2, &mut corr, &entity_order);

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

    /// Generation order is opening-major: verify that stage 0 opening 0 and
    /// stage 1 opening 0 use different seeds (different `stage.id`), while
    /// stage 0 opening 0 and stage 0 opening 1 also differ (different `opening_idx`).
    #[test]
    #[allow(clippy::float_cmp)]
    fn different_openings_and_stages_produce_different_noise() {
        let stages = vec![make_stage(0, 0, 4), make_stage(1, 1, 4)];
        let mut corr = identity_correlation(&[1]);
        let entity_order = vec![EntityId(1)];

        let tree = generate_opening_tree(0, &stages, 1, &mut corr, &entity_order);

        let s0_o0 = tree.opening(0, 0)[0];
        let s0_o1 = tree.opening(0, 1)[0];
        let s1_o0 = tree.opening(1, 0)[0];

        assert_ne!(s0_o0, s0_o1, "same stage, different openings should differ");
        assert_ne!(
            s0_o0, s1_o0,
            "same opening index, different stages should differ"
        );
    }
}
