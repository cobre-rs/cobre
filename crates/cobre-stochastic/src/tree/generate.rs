//! Opening scenario tree generation from pre-decomposed spectral factors
//! and deterministic per-opening seeds. Each `(opening_index, stage)` pair
//! receives independent noise with spatial correlation applied in-place.

use std::ops::Range;

use cobre_core::{EntityId, Stage, temporal::NoiseMethod};
use rand::RngExt;
use rand_distr::StandardNormal;

use crate::{
    StochasticError,
    correlation::resolve::{DecomposedCorrelation, EntityClass},
    noise::{rng::rng_from_seed, seed::derive_opening_seed},
    sampling::historical::HistoricalScenarioLibrary,
    tree::{
        lhs::generate_lhs,
        opening_tree::OpeningTree,
        qmc_halton::generate_qmc_halton,
        qmc_sobol::{MAX_SOBOL_DIM, generate_qmc_sobol},
    },
};

/// Per-class entity counts splitting the flat noise vector into segments for
/// independent spectral correlation. Layout is `[hydros | load buses | NCS]`.
#[derive(Debug, Clone, Copy)]
pub struct ClassDimensions {
    /// Hydro (inflow) entities.
    pub n_hydros: usize,
    /// Stochastic load-bus (load) entities.
    pub n_load_buses: usize,
    /// Stochastic NCS entities.
    pub n_ncs: usize,
}

impl ClassDimensions {
    /// Sum of the three per-class entity counts.
    #[must_use]
    #[inline]
    pub fn total(&self) -> usize {
        self.n_hydros + self.n_load_buses + self.n_ncs
    }

    /// The hydro segment's range within the noise vector.
    #[must_use]
    #[inline]
    pub fn hydro_range(&self) -> Range<usize> {
        0..self.n_hydros
    }

    /// The load-bus segment's range within the noise vector.
    #[must_use]
    #[inline]
    pub fn load_bus_range(&self) -> Range<usize> {
        self.n_hydros..self.n_hydros + self.n_load_buses
    }

    /// The NCS segment's range within the noise vector.
    #[must_use]
    #[inline]
    pub fn ncs_range(&self) -> Range<usize> {
        let start = self.load_bus_range().end;
        start..start + self.n_ncs
    }

    /// Splits `noise` into its `[hydros | load buses | NCS]` segments.
    ///
    /// # Panics
    ///
    /// Panics if `noise.len() < self.n_hydros + self.n_load_buses`.
    #[inline]
    pub fn split_segments_mut<'b>(
        &self,
        noise: &'b mut [f64],
    ) -> (&'b mut [f64], &'b mut [f64], &'b mut [f64]) {
        let (hydro, rest) = noise.split_at_mut(self.n_hydros);
        let (load, ncs) = rest.split_at_mut(self.n_load_buses);
        (hydro, load, ncs)
    }

    /// Asserts `entity_order` observes the `[hydros | load buses | NCS]`
    /// partition contract shared by every noise-generation entry point.
    ///
    /// # Panics
    ///
    /// Panics if `entity_order.len() != self.total()`.
    pub fn assert_partitions(&self, entity_order: &[EntityId]) {
        assert_eq!(
            self.total(),
            entity_order.len(),
            "entity_order length ({}) must equal dims.n_hydros + dims.n_load_buses + dims.n_ncs ({})",
            entity_order.len(),
            self.total(),
        );
    }
}

/// Optional input bundle for [`generate_opening_tree`]. When every field is
/// `None`, the tree is generated from each stage's `scenario_config.noise_method`.
#[derive(Debug, Default, Clone, Copy)]
pub struct OpeningTreeGenerationInputs<'a> {
    /// Required when any stage uses [`cobre_core::temporal::NoiseMethod::HistoricalResiduals`].
    pub historical_library: Option<&'a HistoricalScenarioLibrary>,
    /// Per-stage external scenario count clamping opening counts where the
    /// external library was padded from fewer raw scenarios. `Some` length must
    /// equal the stage count.
    pub external_scenario_counts: Option<&'a [usize]>,
    /// Per-stage (stage-array-indexed) group IDs; consecutive stages sharing an
    /// ID share one noise draw, else each draws independently. `Some` length
    /// must equal the stage count.
    pub noise_group_ids: Option<&'a [u32]>,
}

/// Effective opening count per stage: `branching_factor` clamped by the
/// `HistoricalResiduals` window count and by `external_scenario_counts`. When both
/// clamps apply, the tighter one wins.
fn compute_effective_opening_counts(
    stages: &[Stage],
    historical_library: Option<&HistoricalScenarioLibrary>,
    external_scenario_counts: Option<&[usize]>,
) -> Vec<usize> {
    stages
        .iter()
        .enumerate()
        .map(|(stage_idx, s)| {
            let mut effective = s.scenario_config.branching_factor;

            if s.scenario_config.noise_method == NoiseMethod::HistoricalResiduals
                && let Some(lib) = historical_library
            {
                effective = effective.min(lib.n_windows());
            }

            if let Some(counts) = external_scenario_counts {
                let raw = counts[stage_idx];
                if raw < s.scenario_config.branching_factor && raw <= effective {
                    tracing::warn!(
                        stage_id = s.id,
                        "External scenarios: {} raw scenarios < branching_factor {}; \
                         opening tree clamped to {} openings",
                        raw,
                        s.scenario_config.branching_factor,
                        raw,
                    );
                }
                effective = effective.min(raw);
            }

            effective
        })
        .collect()
}

/// Copy already-correlated noise from the preceding stage when `stage_idx > 0`
/// and both share a group ID. Returns `true` when the copy was performed (caller
/// should `continue`), `false` when no copy is needed.
fn try_copy_noise_group(
    stage_idx: usize,
    noise_group_ids: Option<&[u32]>,
    openings_per_stage: &[usize],
    stage_offsets: &[usize],
    dim: usize,
    data: &mut [f64],
) -> bool {
    let Some(ids) = noise_group_ids else {
        return false;
    };
    if stage_idx == 0 || ids[stage_idx] != ids[stage_idx - 1] {
        return false;
    }
    let src_n_openings = openings_per_stage[stage_idx - 1];
    let src_offset = stage_offsets[stage_idx - 1];
    let n_openings = openings_per_stage[stage_idx];
    let offset = stage_offsets[stage_idx];
    let copy_openings = src_n_openings.min(n_openings);
    let copy_len = copy_openings * dim;
    data.copy_within(src_offset..src_offset + copy_len, offset);
    true
}

fn opening_slice_mut(data: &mut [f64], opening_idx: usize, dim: usize) -> &mut [f64] {
    let start = opening_idx * dim;
    &mut data[start..start + dim]
}

/// Generate raw noise for one stage into `stage_slice` using its configured method.
///
/// Returns `Ok(true)` when the method embeds its own correlation
/// (`HistoricalResiduals`) so the caller must skip spectral correlation;
/// `Ok(false)` when spectral correlation must follow.
///
/// # Errors
///
/// Returns [`StochasticError::UnsupportedNoiseMethod`] for `Selective`, or for
/// `HistoricalResiduals` when `historical_library` is `None`; and
/// [`StochasticError::DimensionExceedsCapacity`] when `QmcSobol` exceeds
/// `MAX_SOBOL_DIM`.
fn generate_stage_raw_noise(
    base_seed: u64,
    stage: &Stage,
    stage_idx: usize,
    n_openings: usize,
    dims: ClassDimensions,
    historical_library: Option<&HistoricalScenarioLibrary>,
    stage_slice: &mut [f64],
) -> Result<bool, StochasticError> {
    let dim = dims.total();
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
        NoiseMethod::HistoricalResiduals => {
            let lib =
                historical_library.ok_or_else(|| StochasticError::UnsupportedNoiseMethod {
                    method: "historical_residuals".to_string(),
                    stage_id: stage.id,
                    reason:
                        "HistoricalResiduals noise method requires a HistoricalScenarioLibrary \
                             but none was provided"
                            .to_string(),
                })?;
            let n_windows = lib.n_windows();

            if n_windows < stage.scenario_config.branching_factor {
                tracing::warn!(
                    stage_id = stage.id,
                    "HistoricalResiduals: {} historical windows < branching_factor {}; \
                     opening tree clamped to {} openings",
                    n_windows,
                    stage.scenario_config.branching_factor,
                    n_windows,
                );
            }

            for opening_idx in 0..n_openings {
                let noise_slice = opening_slice_mut(stage_slice, opening_idx, dim);
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let seed = derive_opening_seed(base_seed, opening_idx as u32, stage.id as u32);
                #[allow(clippy::cast_possible_truncation)]
                let window_idx = (seed % (n_windows as u64)) as usize;
                let eta = lib.eta_slice(window_idx, stage_idx);
                noise_slice[dims.hydro_range()].copy_from_slice(eta);
            }
            return Ok(true);
        }
    }
    Ok(false)
}

/// Fill all `n_openings` noise vectors for one stage using SAA (pure Monte Carlo).
fn generate_saa(base_seed: u64, stage: &Stage, n_openings: usize, dim: usize, output: &mut [f64]) {
    for opening_idx in 0..n_openings {
        let noise_slice = opening_slice_mut(output, opening_idx, dim);
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
/// Generation is stage-major (outer: stages, inner: openings) so batch methods
/// like LHS see all of a stage's openings at once. Spectral correlation is then
/// applied per class in-place, except for `HistoricalResiduals` stages whose
/// residuals already embed empirical cross-entity correlation. `Selective`
/// returns an error; the other methods are supported. Per-stage opening counts
/// and `noise_group_ids` sharing are described on [`OpeningTreeGenerationInputs`].
///
/// `entity_order` must have layout `[hydros | load buses | NCS]`.
///
/// # Errors
///
/// Returns [`StochasticError::UnsupportedNoiseMethod`] if any stage uses
/// [`NoiseMethod::Selective`], or [`NoiseMethod::HistoricalResiduals`] with no
/// `historical_library`; and [`StochasticError::DimensionExceedsCapacity`] when a
/// `QmcSobol` stage exceeds `MAX_SOBOL_DIM`.
///
/// # Panics
///
/// Panics if `external_scenario_counts` is `Some` with length `!= stages.len()`,
/// or if `entity_order` fails [`ClassDimensions::assert_partitions`]; debug-only
/// for the same `noise_group_ids` mismatch.
pub fn generate_opening_tree<'a>(
    base_seed: u64,
    stages: &'a [Stage],
    correlation: &'a DecomposedCorrelation,
    entity_order: &'a [EntityId],
    dims: ClassDimensions,
    inputs: &OpeningTreeGenerationInputs<'a>,
) -> Result<OpeningTree, StochasticError> {
    let dim = dims.total();
    let historical_library = inputs.historical_library;
    let external_scenario_counts = inputs.external_scenario_counts;
    let noise_group_ids = inputs.noise_group_ids;

    if let Some(counts) = external_scenario_counts {
        assert_eq!(
            counts.len(),
            stages.len(),
            "external_scenario_counts length ({}) must equal stages length ({})",
            counts.len(),
            stages.len(),
        );
    }
    debug_assert!(
        noise_group_ids.is_none_or(|ids| ids.len() == stages.len()),
        "noise_group_ids length ({}) must equal stages length ({})",
        noise_group_ids.map_or(0, <[u32]>::len),
        stages.len(),
    );

    let n_stages = stages.len();

    dims.assert_partitions(entity_order);

    let openings_per_stage =
        compute_effective_opening_counts(stages, historical_library, external_scenario_counts);

    let mut stage_offsets = Vec::with_capacity(n_stages);
    let mut running_offset = 0usize;
    for &n_openings in &openings_per_stage {
        stage_offsets.push(running_offset);
        running_offset += n_openings * dim;
    }
    let total_len = running_offset;

    let mut data = vec![0.0f64; total_len];
    let mut corr_scratch = vec![0.0f64; 2 * dim];

    for (stage_idx, stage) in stages.iter().enumerate() {
        let n_openings = openings_per_stage[stage_idx];
        let offset = stage_offsets[stage_idx];

        if try_copy_noise_group(
            stage_idx,
            noise_group_ids,
            &openings_per_stage,
            &stage_offsets,
            dim,
            &mut data,
        ) {
            continue;
        }

        let stage_slice = &mut data[offset..offset + n_openings * dim];

        let skip_spectral = generate_stage_raw_noise(
            base_seed,
            stage,
            stage_idx,
            n_openings,
            dims,
            historical_library,
            stage_slice,
        )?;

        if skip_spectral {
            continue;
        }

        let groups = correlation.groups_for_stage(stage.id);

        for opening_idx in 0..n_openings {
            let noise = opening_slice_mut(stage_slice, opening_idx, dim);
            let (inflow_noise, load_noise, ncs_noise) = dims.split_segments_mut(noise);
            for (class, class_noise) in [
                (EntityClass::Inflow, inflow_noise),
                (EntityClass::Load, load_noise),
                (EntityClass::Ncs, ncs_noise),
            ] {
                DecomposedCorrelation::apply_groups_for_class(
                    groups,
                    class,
                    class_noise,
                    &mut corr_scratch,
                );
            }
        }
    }

    Ok(OpeningTree::from_parts(data, openings_per_stage, dim))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use cobre_core::{
        EntityId, Stage,
        scenario::{CorrelationEntity, CorrelationGroup, CorrelationModel, CorrelationProfile},
        temporal::{NoiseMethod, ScenarioSourceConfig},
        test_support::StageSpec,
    };

    use crate::{
        StochasticError, correlation::resolve::DecomposedCorrelation,
        sampling::historical::HistoricalScenarioLibrary,
    };

    use super::{ClassDimensions, OpeningTreeGenerationInputs, generate_opening_tree};

    fn make_stage(index: usize, id: i32, branching_factor: usize) -> Stage {
        make_stage_with_method(index, id, branching_factor, NoiseMethod::Saa)
    }

    fn make_stage_with_method(
        index: usize,
        id: i32,
        branching_factor: usize,
        noise_method: NoiseMethod,
    ) -> Stage {
        cobre_core::test_support::make_stage(StageSpec {
            id,
            index: Some(index),
            season_id: Some(0),
            blocks: Vec::new(),
            scenario_config: ScenarioSourceConfig {
                branching_factor,
                noise_method,
            },
            ..Default::default()
        })
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

    #[test]
    fn class_dimensions_ranges_and_split_match_the_layout() {
        let dims = ClassDimensions {
            n_hydros: 2,
            n_load_buses: 0,
            n_ncs: 3,
        };
        assert_eq!(dims.hydro_range(), 0..2);
        assert_eq!(dims.load_bus_range(), 2..2);
        assert_eq!(dims.ncs_range(), 2..5);
        assert_eq!(dims.total(), 5);

        let mut noise = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (hydro, load, ncs) = dims.split_segments_mut(&mut noise);
        assert_eq!(hydro, &[1.0, 2.0]);
        assert!(load.is_empty());
        assert_eq!(ncs, &[3.0, 4.0, 5.0]);
    }

    #[test]
    fn determinism_same_inputs_produce_identical_trees() {
        let stages = vec![make_stage(0, 0, 3), make_stage(1, 1, 3)];
        let corr = identity_correlation(&[1, 2]);
        let corr2 = identity_correlation(&[1, 2]);
        let entity_order = vec![EntityId(1), EntityId(2)];

        let dims = ClassDimensions {
            n_hydros: 2,
            n_load_buses: 0,
            n_ncs: 0,
        };
        let tree1 = generate_opening_tree(
            42,
            &stages,
            &corr,
            &entity_order,
            dims,
            &OpeningTreeGenerationInputs::default(),
        )
        .unwrap();
        let tree2 = generate_opening_tree(
            42,
            &stages,
            &corr2,
            &entity_order,
            dims,
            &OpeningTreeGenerationInputs::default(),
        )
        .unwrap();

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
        let corr = identity_correlation(&[1, 2]);
        let entity_order = vec![EntityId(1), EntityId(2)];
        let dims = ClassDimensions {
            n_hydros: 2,
            n_load_buses: 0,
            n_ncs: 0,
        };

        let tree = generate_opening_tree(
            42,
            &stages,
            &corr,
            &entity_order,
            dims,
            &OpeningTreeGenerationInputs::default(),
        )
        .unwrap();

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
        let corr = identity_correlation(&[1, 2]);
        let entity_order = vec![EntityId(1), EntityId(2)];
        let dims = ClassDimensions {
            n_hydros: 2,
            n_load_buses: 0,
            n_ncs: 0,
        };

        let tree_a = generate_opening_tree(
            42,
            &stages,
            &corr,
            &entity_order,
            dims,
            &OpeningTreeGenerationInputs::default(),
        )
        .unwrap();
        let tree_b = generate_opening_tree(
            99,
            &stages,
            &corr,
            &entity_order,
            dims,
            &OpeningTreeGenerationInputs::default(),
        )
        .unwrap();

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
        let corr = identity_correlation(&[1, 2]);
        let entity_order = vec![EntityId(1), EntityId(2)];
        let dims = ClassDimensions {
            n_hydros: 2,
            n_load_buses: 0,
            n_ncs: 0,
        };

        let tree = generate_opening_tree(
            42,
            &stages,
            &corr,
            &entity_order,
            dims,
            &OpeningTreeGenerationInputs::default(),
        )
        .unwrap();

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
        let corr = identity_correlation(&[1, 2, 3]);
        let entity_order = vec![EntityId(1), EntityId(2), EntityId(3)];
        let dims = ClassDimensions {
            n_hydros: 3,
            n_load_buses: 0,
            n_ncs: 0,
        };

        let tree = generate_opening_tree(
            7,
            &stages,
            &corr,
            &entity_order,
            dims,
            &OpeningTreeGenerationInputs::default(),
        )
        .unwrap();

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
        let corr = identity_correlation(&[1]);
        let entity_order = vec![EntityId(1)];
        let dims = ClassDimensions {
            n_hydros: 1,
            n_load_buses: 0,
            n_ncs: 0,
        };

        let tree = generate_opening_tree(
            12345,
            &stages,
            &corr,
            &entity_order,
            dims,
            &OpeningTreeGenerationInputs::default(),
        )
        .unwrap();

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
        let corr = identity_correlation(&[1, 2, 3, 4]);
        let entity_order = vec![EntityId(1), EntityId(2), EntityId(3), EntityId(4)];
        let dims = ClassDimensions {
            n_hydros: 4,
            n_load_buses: 0,
            n_ncs: 0,
        };

        let tree = generate_opening_tree(
            99,
            &stages,
            &corr,
            &entity_order,
            dims,
            &OpeningTreeGenerationInputs::default(),
        )
        .unwrap();

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
        let corr = correlated_correlation(&[1, 2], 0.8);
        let entity_order = vec![EntityId(1), EntityId(2)];
        let dims = ClassDimensions {
            n_hydros: 2,
            n_load_buses: 0,
            n_ncs: 0,
        };

        let tree = generate_opening_tree(
            54321,
            &stages,
            &corr,
            &entity_order,
            dims,
            &OpeningTreeGenerationInputs::default(),
        )
        .unwrap();

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
        let corr = identity_correlation(&[1]);
        let entity_order = vec![EntityId(1)];
        let dims = ClassDimensions {
            n_hydros: 1,
            n_load_buses: 0,
            n_ncs: 0,
        };

        let tree = generate_opening_tree(
            0,
            &stages,
            &corr,
            &entity_order,
            dims,
            &OpeningTreeGenerationInputs::default(),
        )
        .unwrap();

        let s0_o0 = tree.opening(0, 0)[0];
        let s0_o1 = tree.opening(0, 1)[0];
        let s1_o0 = tree.opening(1, 0)[0];

        assert_ne!(s0_o0, s0_o1, "same stage, different openings should differ");
        assert_ne!(
            s0_o0, s1_o0,
            "same opening index, different stages should differ"
        );
    }

    /// SAA stages must produce bit-for-bit stable output.
    ///
    /// Golden values pinned for seed=42, 3 stages (Saa, bf=3), dim=2, identity
    /// correlation on [1, 2].
    #[test]
    #[allow(clippy::float_cmp)]
    fn saa_bitwise_compatible_with_pre_refactor_golden_values() {
        let stages = vec![
            make_stage(0, 0, 3),
            make_stage(1, 1, 3),
            make_stage(2, 2, 3),
        ];
        let corr = identity_correlation(&[1, 2]);
        let entity_order = vec![EntityId(1), EntityId(2)];
        let dims = ClassDimensions {
            n_hydros: 2,
            n_load_buses: 0,
            n_ncs: 0,
        };

        let tree = generate_opening_tree(
            42,
            &stages,
            &corr,
            &entity_order,
            dims,
            &OpeningTreeGenerationInputs::default(),
        )
        .unwrap();

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
        let corr = identity_correlation(&[1, 2]);
        let entity_order = vec![EntityId(1), EntityId(2)];
        let dims = ClassDimensions {
            n_hydros: 2,
            n_load_buses: 0,
            n_ncs: 0,
        };

        let result = generate_opening_tree(
            42,
            &stages,
            &corr,
            &entity_order,
            dims,
            &OpeningTreeGenerationInputs::default(),
        );

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
        let corr = identity_correlation(&[1, 2, 3]);
        let entity_order = vec![EntityId(1), EntityId(2), EntityId(3)];
        let dims = ClassDimensions {
            n_hydros: 3,
            n_load_buses: 0,
            n_ncs: 0,
        };

        let tree = generate_opening_tree(
            42,
            &stages,
            &corr,
            &entity_order,
            dims,
            &OpeningTreeGenerationInputs::default(),
        )
        .unwrap();

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
        let corr = identity_correlation(&[1, 2, 3]);
        let entity_order = vec![EntityId(1), EntityId(2), EntityId(3)];
        let dims = ClassDimensions {
            n_hydros: 3,
            n_load_buses: 0,
            n_ncs: 0,
        };

        let tree = generate_opening_tree(
            42,
            &stages,
            &corr,
            &entity_order,
            dims,
            &OpeningTreeGenerationInputs::default(),
        )
        .unwrap();

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
        let corr = identity_correlation(&[1, 2, 3]);
        let entity_order = vec![EntityId(1), EntityId(2), EntityId(3)];
        let dims = ClassDimensions {
            n_hydros: 3,
            n_load_buses: 0,
            n_ncs: 0,
        };

        let tree = generate_opening_tree(
            42,
            &stages,
            &corr,
            &entity_order,
            dims,
            &OpeningTreeGenerationInputs::default(),
        )
        .unwrap();

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
        let stages = vec![make_stage_with_method(0, 0, 4, NoiseMethod::QmcSobol)];
        // Correlation is never read on this early-return path, so a
        // one-entity correlation stands in for the 21_202-entity dims below.
        let corr = identity_correlation(&[1]);
        let entity_order: Vec<EntityId> = (1..=21_202).map(EntityId).collect();
        let dims = ClassDimensions {
            n_hydros: 21_202, // one above MAX_SOBOL_DIM = 21_201
            n_load_buses: 0,
            n_ncs: 0,
        };

        let result = generate_opening_tree(
            42,
            &stages,
            &corr,
            &entity_order,
            dims,
            &OpeningTreeGenerationInputs::default(),
        );

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
        let stages = vec![
            make_stage_with_method(0, 0, n_openings, NoiseMethod::QmcSobol),
            make_stage_with_method(1, 1, n_openings, NoiseMethod::Saa),
        ];
        let corr = identity_correlation(&[1, 2]);
        let entity_order = vec![EntityId(1), EntityId(2)];
        let dims = ClassDimensions {
            n_hydros: 2,
            n_load_buses: 0,
            n_ncs: 0,
        };

        let tree = generate_opening_tree(
            42,
            &stages,
            &corr,
            &entity_order,
            dims,
            &OpeningTreeGenerationInputs::default(),
        )
        .unwrap();

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
        let corr = identity_correlation(&[1, 2, 3]);
        let entity_order = vec![EntityId(1), EntityId(2), EntityId(3)];
        let dims = ClassDimensions {
            n_hydros: 3,
            n_load_buses: 0,
            n_ncs: 0,
        };

        let tree = generate_opening_tree(
            42,
            &stages,
            &corr,
            &entity_order,
            dims,
            &OpeningTreeGenerationInputs::default(),
        )
        .unwrap();

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
        let stages = vec![
            make_stage_with_method(0, 0, n_openings, NoiseMethod::QmcHalton),
            make_stage_with_method(1, 1, n_openings, NoiseMethod::Saa),
        ];
        let corr = identity_correlation(&[1, 2]);
        let entity_order = vec![EntityId(1), EntityId(2)];
        let dims = ClassDimensions {
            n_hydros: 2,
            n_load_buses: 0,
            n_ncs: 0,
        };

        let tree = generate_opening_tree(
            42,
            &stages,
            &corr,
            &entity_order,
            dims,
            &OpeningTreeGenerationInputs::default(),
        )
        .unwrap();

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

    // -------------------------------------------------------------------------
    // HistoricalResiduals tests
    // -------------------------------------------------------------------------

    /// Helper: build a `HistoricalScenarioLibrary` with known, distinct eta values.
    ///
    /// Each eta entry is set to `(window * 100 + stage) as f64` for all hydros,
    /// making it easy to verify which (window, stage) pair was selected.
    #[allow(
        clippy::cast_possible_wrap,
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        clippy::cast_lossless
    )]
    fn make_test_library(
        n_windows: usize,
        n_stages: usize,
        n_hydros: usize,
    ) -> HistoricalScenarioLibrary {
        let window_years: Vec<i32> = (0..n_windows).map(|w| 1990 + w as i32).collect();
        let mut lib =
            HistoricalScenarioLibrary::new(n_windows, n_stages, n_hydros, 1, window_years);
        for w in 0..n_windows {
            for s in 0..n_stages {
                let val = (w * 100 + s) as f64;
                lib.eta_slice_mut(w, s).fill(val);
            }
        }
        lib
    }

    /// Calling `generate_opening_tree` with `HistoricalResiduals` and `library = None`
    /// returns `Err(StochasticError::UnsupportedNoiseMethod)` with `method ==
    /// "historical_residuals"`.
    #[test]
    fn test_historical_residuals_none_library_returns_error() {
        let stages = vec![
            make_stage(0, 0, 3),
            make_stage_with_method(1, 9, 3, NoiseMethod::HistoricalResiduals),
        ];
        let corr = identity_correlation(&[1, 2]);
        let entity_order = vec![EntityId(1), EntityId(2)];
        let dims = ClassDimensions {
            n_hydros: 2,
            n_load_buses: 0,
            n_ncs: 0,
        };

        let result = generate_opening_tree(
            42,
            &stages,
            &corr,
            &entity_order,
            dims,
            &OpeningTreeGenerationInputs::default(),
        );

        match result {
            Err(StochasticError::UnsupportedNoiseMethod {
                method,
                stage_id,
                reason: _,
            }) => {
                assert_eq!(method, "historical_residuals");
                assert_eq!(stage_id, 9);
            }
            Ok(_) => panic!("expected Err but got Ok"),
            Err(other) => panic!("expected UnsupportedNoiseMethod, got {other:?}"),
        }
    }

    /// With a library of 5 windows and `branching_factor = 4`, the tree has 4 openings
    /// per stage. The hydro segment of each opening matches
    /// `library.eta_slice(hash(seed, opening, stage) % 5, stage_idx)`.
    #[test]
    #[allow(clippy::float_cmp)]
    fn test_historical_residuals_copies_eta_from_library() {
        use crate::noise::seed::derive_opening_seed;

        let n_hydros = 3_usize;
        let n_stages = 2_usize;
        let n_windows = 5_usize;
        let branching_factor = 4_usize;
        let base_seed = 77_u64;

        let lib = make_test_library(n_windows, n_stages, n_hydros);

        let stages = vec![
            make_stage_with_method(0, 10, branching_factor, NoiseMethod::HistoricalResiduals),
            make_stage_with_method(1, 20, branching_factor, NoiseMethod::HistoricalResiduals),
        ];
        let corr = identity_correlation(&[1, 2, 3]);
        let entity_order = vec![EntityId(1), EntityId(2), EntityId(3)];
        let dims = ClassDimensions {
            n_hydros,
            n_load_buses: 0,
            n_ncs: 0,
        };

        let tree = generate_opening_tree(
            base_seed,
            &stages,
            &corr,
            &entity_order,
            dims,
            &OpeningTreeGenerationInputs {
                historical_library: Some(&lib),
                ..OpeningTreeGenerationInputs::default()
            },
        )
        .unwrap();

        assert_eq!(tree.n_stages(), 2);
        assert_eq!(tree.n_openings(0), branching_factor);
        assert_eq!(tree.n_openings(1), branching_factor);

        for (stage_idx, stage) in stages.iter().enumerate() {
            for opening_idx in 0..branching_factor {
                #[allow(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    clippy::cast_possible_truncation
                )]
                let seed = derive_opening_seed(base_seed, opening_idx as u32, stage.id as u32);
                #[allow(clippy::cast_possible_truncation)]
                let expected_window = (seed % (n_windows as u64)) as usize;
                let expected_eta = lib.eta_slice(expected_window, stage_idx);
                let actual = tree.opening(stage_idx, opening_idx);
                assert_eq!(
                    actual, expected_eta,
                    "stage={stage_idx} opening={opening_idx}: hydro segment mismatch"
                );
            }
        }
    }

    /// With `dim = n_hydros + n_load + n_ncs`, indices `[n_hydros..dim]` of every
    /// opening produced by `HistoricalResiduals` must be `0.0`.
    #[test]
    #[allow(clippy::float_cmp)]
    fn test_historical_residuals_zeros_non_hydro_slots() {
        let n_hydros = 2_usize;
        let n_load = 1_usize;
        let n_ncs = 1_usize;
        let dim = n_hydros + n_load + n_ncs;
        let n_windows = 3_usize;
        let n_stages = 1_usize;
        let branching_factor = 3_usize;

        let lib = make_test_library(n_windows, n_stages, n_hydros);

        let stages = vec![make_stage_with_method(
            0,
            0,
            branching_factor,
            NoiseMethod::HistoricalResiduals,
        )];
        // Use 4 entity ids: 2 hydros + 1 load + 1 ncs.
        let corr = identity_correlation(&[1, 2, 3, 4]);
        let entity_order = vec![EntityId(1), EntityId(2), EntityId(3), EntityId(4)];
        let dims = ClassDimensions {
            n_hydros,
            n_load_buses: n_load,
            n_ncs,
        };

        let tree = generate_opening_tree(
            42,
            &stages,
            &corr,
            &entity_order,
            dims,
            &OpeningTreeGenerationInputs {
                historical_library: Some(&lib),
                ..OpeningTreeGenerationInputs::default()
            },
        )
        .unwrap();

        for opening_idx in 0..branching_factor {
            let slice = tree.opening(0, opening_idx);
            assert_eq!(slice.len(), dim);
            for (i, &val) in slice.iter().enumerate().skip(n_hydros) {
                assert_eq!(
                    val, 0.0,
                    "opening={opening_idx} index={i}: expected 0.0 for non-hydro slot"
                );
            }
        }
    }

    /// When `n_windows < branching_factor`, the opening tree's opening count for
    /// that stage is clamped to `n_windows`.
    #[test]
    fn test_historical_residuals_clamps_openings_when_windows_lt_branching() {
        let n_windows = 2_usize;
        let branching_factor = 5_usize;
        let n_hydros = 2_usize;
        let n_stages = 1_usize;

        let lib = make_test_library(n_windows, n_stages, n_hydros);

        let stages = vec![make_stage_with_method(
            0,
            0,
            branching_factor,
            NoiseMethod::HistoricalResiduals,
        )];
        let corr = identity_correlation(&[1, 2]);
        let entity_order = vec![EntityId(1), EntityId(2)];
        let dims = ClassDimensions {
            n_hydros,
            n_load_buses: 0,
            n_ncs: 0,
        };

        let tree = generate_opening_tree(
            42,
            &stages,
            &corr,
            &entity_order,
            dims,
            &OpeningTreeGenerationInputs {
                historical_library: Some(&lib),
                ..OpeningTreeGenerationInputs::default()
            },
        )
        .unwrap();

        assert_eq!(
            tree.n_openings(0),
            n_windows,
            "opening count must be clamped to n_windows when n_windows < branching_factor"
        );
    }

    /// Two calls with the same seed and library produce bit-identical output.
    #[test]
    fn test_historical_residuals_deterministic_window_selection() {
        let n_windows = 4_usize;
        let n_stages = 2_usize;
        let n_hydros = 2_usize;
        let branching_factor = 3_usize;
        let base_seed = 12_345_u64;

        let lib = make_test_library(n_windows, n_stages, n_hydros);

        let stages = vec![
            make_stage_with_method(0, 1, branching_factor, NoiseMethod::HistoricalResiduals),
            make_stage_with_method(1, 2, branching_factor, NoiseMethod::HistoricalResiduals),
        ];
        let corr = identity_correlation(&[1, 2]);
        let entity_order = vec![EntityId(1), EntityId(2)];
        let dims = ClassDimensions {
            n_hydros,
            n_load_buses: 0,
            n_ncs: 0,
        };

        let tree1 = generate_opening_tree(
            base_seed,
            &stages,
            &corr,
            &entity_order,
            dims,
            &OpeningTreeGenerationInputs {
                historical_library: Some(&lib),
                ..OpeningTreeGenerationInputs::default()
            },
        )
        .unwrap();
        let tree2 = generate_opening_tree(
            base_seed,
            &stages,
            &corr,
            &entity_order,
            dims,
            &OpeningTreeGenerationInputs {
                historical_library: Some(&lib),
                ..OpeningTreeGenerationInputs::default()
            },
        )
        .unwrap();

        for s in 0..n_stages {
            for o in 0..branching_factor {
                assert_eq!(
                    tree1.opening(s, o),
                    tree2.opening(s, o),
                    "non-deterministic output at stage={s} opening={o}"
                );
            }
        }
    }

    /// External clamping reduces opening count when raw < `branching_factor`.
    ///
    /// 3 stages with `branching_factor=10`, `external_scenario_counts=[2, 10, 5]`.
    /// Stage 0 clamps to 2, stage 1 stays at 10 (no clamping), stage 2 clamps to 5.
    #[test]
    fn test_external_clamping_reduces_openings() {
        let stages = vec![
            make_stage(0, 0, 10),
            make_stage(1, 1, 10),
            make_stage(2, 2, 10),
        ];
        let corr = identity_correlation(&[1]);
        let entity_order = vec![EntityId(1)];
        let dims = ClassDimensions {
            n_hydros: 1,
            n_load_buses: 0,
            n_ncs: 0,
        };
        let external_counts = [2usize, 10, 5];

        let tree = generate_opening_tree(
            42,
            &stages,
            &corr,
            &entity_order,
            dims,
            &OpeningTreeGenerationInputs {
                external_scenario_counts: Some(&external_counts),
                ..OpeningTreeGenerationInputs::default()
            },
        )
        .unwrap();

        assert_eq!(tree.n_openings(0), 2, "stage 0 must be clamped to 2");
        assert_eq!(tree.n_openings(1), 10, "stage 1 must remain at 10");
        assert_eq!(tree.n_openings(2), 5, "stage 2 must be clamped to 5");
    }

    /// Passing `external_scenario_counts = None` leaves opening counts unchanged
    /// (backward-compatible behaviour).
    #[test]
    fn test_external_clamping_none_no_effect() {
        let stages = vec![make_stage(0, 0, 5), make_stage(1, 1, 5)];
        let corr = identity_correlation(&[1]);
        let entity_order = vec![EntityId(1)];
        let dims = ClassDimensions {
            n_hydros: 1,
            n_load_buses: 0,
            n_ncs: 0,
        };

        let tree = generate_opening_tree(
            42,
            &stages,
            &corr,
            &entity_order,
            dims,
            &OpeningTreeGenerationInputs::default(),
        )
        .unwrap();

        assert_eq!(tree.n_openings(0), 5, "stage 0 must remain at 5");
        assert_eq!(tree.n_openings(1), 5, "stage 1 must remain at 5");
    }

    /// Both Historical and External clamping apply simultaneously; the tighter
    /// clamp wins.
    ///
    /// `branching_factor=10`, `n_windows=5`, `external_count=3` -> effective=3.
    #[test]
    fn test_external_and_historical_clamping_combined() {
        let n_windows = 5_usize;
        let n_hydros = 2_usize;
        let branching_factor = 10_usize;

        let lib = make_test_library(n_windows, 1, n_hydros);

        let stages = vec![make_stage_with_method(
            0,
            1,
            branching_factor,
            NoiseMethod::HistoricalResiduals,
        )];
        let corr = identity_correlation(&[1, 2]);
        let entity_order = vec![EntityId(1), EntityId(2)];
        let dims = ClassDimensions {
            n_hydros,
            n_load_buses: 0,
            n_ncs: 0,
        };
        let external_counts = [3usize];

        let tree = generate_opening_tree(
            42,
            &stages,
            &corr,
            &entity_order,
            dims,
            &OpeningTreeGenerationInputs {
                historical_library: Some(&lib),
                external_scenario_counts: Some(&external_counts),
                ..OpeningTreeGenerationInputs::default()
            },
        )
        .unwrap();

        assert_eq!(
            tree.n_openings(0),
            3,
            "external clamp (3) is tighter than historical (5); effective must be 3"
        );
    }

    /// Passing `external_scenario_counts` with wrong length panics with a
    /// descriptive message.
    #[test]
    #[should_panic(expected = "external_scenario_counts length")]
    fn test_external_clamping_counts_length_mismatch_panics() {
        let stages = vec![make_stage(0, 0, 5), make_stage(1, 1, 5)];
        let corr = identity_correlation(&[1]);
        let entity_order = vec![EntityId(1)];
        let dims = ClassDimensions {
            n_hydros: 1,
            n_load_buses: 0,
            n_ncs: 0,
        };
        // 2 stages but only 1 count — must panic.
        let external_counts = [1usize];

        let _ = generate_opening_tree(
            42,
            &stages,
            &corr,
            &entity_order,
            dims,
            &OpeningTreeGenerationInputs {
                external_scenario_counts: Some(&external_counts),
                ..OpeningTreeGenerationInputs::default()
            },
        );
    }

    // -----------------------------------------------------------------------
    // noise_group_ids tests
    // -----------------------------------------------------------------------

    /// Calling `generate_opening_tree` with `noise_group_ids = None` produces the
    /// same output as calling it with all-unique group IDs (backward compatibility).
    #[test]
    fn test_opening_tree_noise_group_none_backward_compat() {
        let stages = vec![
            make_stage(0, 0, 3),
            make_stage(1, 1, 3),
            make_stage(2, 2, 3),
            make_stage(3, 3, 3),
        ];
        let corr = identity_correlation(&[1, 2]);
        let entity_order = vec![EntityId(1), EntityId(2)];
        let dims = ClassDimensions {
            n_hydros: 2,
            n_load_buses: 0,
            n_ncs: 0,
        };

        let tree_none = generate_opening_tree(
            42,
            &stages,
            &corr,
            &entity_order,
            dims,
            &OpeningTreeGenerationInputs::default(),
        )
        .unwrap();

        let unique_ids = [0u32, 1, 2, 3];
        let tree_unique = generate_opening_tree(
            42,
            &stages,
            &corr,
            &entity_order,
            dims,
            &OpeningTreeGenerationInputs {
                noise_group_ids: Some(&unique_ids),
                ..OpeningTreeGenerationInputs::default()
            },
        )
        .unwrap();

        assert_eq!(tree_none.n_stages(), tree_unique.n_stages());
        for s in 0..tree_none.n_stages() {
            assert_eq!(tree_none.n_openings(s), tree_unique.n_openings(s));
            for o in 0..tree_none.n_openings(s) {
                assert_eq!(
                    tree_none.opening(s, o),
                    tree_unique.opening(s, o),
                    "stage={s} opening={o}: None and unique-IDs trees must be bit-identical"
                );
            }
        }
    }

    /// Four stages all in the same noise group (group 0): stages 1, 2, and 3 must
    /// have noise vectors bit-identical to stage 0 for every opening.
    #[test]
    fn test_opening_tree_same_group_copies_noise() {
        let stages = vec![
            make_stage(0, 0, 3),
            make_stage(1, 1, 3),
            make_stage(2, 2, 3),
            make_stage(3, 3, 3),
        ];
        let corr = identity_correlation(&[1, 2]);
        let entity_order = vec![EntityId(1), EntityId(2)];
        let dims = ClassDimensions {
            n_hydros: 2,
            n_load_buses: 0,
            n_ncs: 0,
        };
        let all_same_group = [0u32, 0, 0, 0];

        let tree = generate_opening_tree(
            42,
            &stages,
            &corr,
            &entity_order,
            dims,
            &OpeningTreeGenerationInputs {
                noise_group_ids: Some(&all_same_group),
                ..OpeningTreeGenerationInputs::default()
            },
        )
        .unwrap();

        for s in 0..4 {
            assert_eq!(tree.n_openings(s), 3, "stage {s} must have 3 openings");
        }

        for s in 1..4 {
            for o in 0..3 {
                assert_eq!(
                    tree.opening(0, o),
                    tree.opening(s, o),
                    "stage {s} opening {o} must be identical to stage 0 opening {o}"
                );
            }
        }
    }

    /// Four stages with groups [0, 0, 1, 1]: stages 0-1 share identical noise,
    /// stages 2-3 share identical noise, but the two groups differ from each other.
    #[test]
    fn test_opening_tree_two_groups() {
        let stages = vec![
            make_stage(0, 0, 3),
            make_stage(1, 1, 3),
            make_stage(2, 2, 3),
            make_stage(3, 3, 3),
        ];
        let corr = identity_correlation(&[1, 2]);
        let entity_order = vec![EntityId(1), EntityId(2)];
        let dims = ClassDimensions {
            n_hydros: 2,
            n_load_buses: 0,
            n_ncs: 0,
        };
        let two_groups = [0u32, 0, 1, 1];

        let tree = generate_opening_tree(
            42,
            &stages,
            &corr,
            &entity_order,
            dims,
            &OpeningTreeGenerationInputs {
                noise_group_ids: Some(&two_groups),
                ..OpeningTreeGenerationInputs::default()
            },
        )
        .unwrap();

        for o in 0..3 {
            assert_eq!(
                tree.opening(0, o),
                tree.opening(1, o),
                "group 0: stage 1 opening {o} must match stage 0 opening {o}"
            );
        }

        for o in 0..3 {
            assert_eq!(
                tree.opening(2, o),
                tree.opening(3, o),
                "group 1: stage 3 opening {o} must match stage 2 opening {o}"
            );
        }

        // Inter-group: group 0 noise must differ from group 1 noise (with overwhelming
        // probability given the random seed; zero probability of accidental equality).
        let group0_any = (0..3).map(|o| tree.opening(0, o)).collect::<Vec<_>>();
        let group1_any = (0..3).map(|o| tree.opening(2, o)).collect::<Vec<_>>();
        assert_ne!(
            group0_any, group1_any,
            "group 0 and group 1 noise must differ"
        );
    }

    /// 12 stages with unique group IDs (monthly study): no copying occurs and the
    /// output is bit-identical to the `noise_group_ids = None` case.
    #[test]
    fn test_opening_tree_monthly_no_copy() {
        let stages: Vec<_> = (0..12_usize)
            .map(|i| make_stage(i, i32::try_from(i).unwrap(), 4))
            .collect();
        let entity_ids: Vec<i32> = (1..=3).collect();
        let corr = identity_correlation(&entity_ids);
        let entity_order: Vec<EntityId> = entity_ids.iter().map(|&id| EntityId(id)).collect();
        let dims = ClassDimensions {
            n_hydros: 3,
            n_load_buses: 0,
            n_ncs: 0,
        };

        let tree_none = generate_opening_tree(
            999,
            &stages,
            &corr,
            &entity_order,
            dims,
            &OpeningTreeGenerationInputs::default(),
        )
        .unwrap();

        // Unique IDs: 0..12 — no stage shares a group with its predecessor.
        let unique_ids: Vec<u32> = (0..12).collect();
        let tree_unique = generate_opening_tree(
            999,
            &stages,
            &corr,
            &entity_order,
            dims,
            &OpeningTreeGenerationInputs {
                noise_group_ids: Some(&unique_ids),
                ..OpeningTreeGenerationInputs::default()
            },
        )
        .unwrap();

        assert_eq!(tree_none.n_stages(), 12);
        assert_eq!(tree_unique.n_stages(), 12);
        for s in 0..12 {
            for o in 0..tree_none.n_openings(s) {
                assert_eq!(
                    tree_none.opening(s, o),
                    tree_unique.opening(s, o),
                    "monthly study stage={s} opening={o}: None and unique-IDs must be identical"
                );
            }
        }
    }
}
