//! Pre-computation of normal noise model parameter arrays for LP RHS patching.
//!
//! This module provides [`PrecomputedNormalLp`], the performance-adapted cache built
//! once during initialization from raw normal noise model parameters. It exposes flat,
//! contiguous arrays in stage-major layout so the calling algorithm can patch LP
//! right-hand sides without per-scenario recomputation.
//!
//! ## Array layout
//!
//! Two-dimensional arrays (mean, std) use **stage-major** layout:
//! `array[stage * n_entities + entity_idx]`.
//!
//! The three-dimensional factor array uses **stage-major, entity-minor, block-innermost**:
//! `factors[stage * n_entities * max_blocks + entity_idx * max_blocks + block_idx]`.
//!
//! This layout is optimal for sequential stage iteration within a scenario trajectory:
//! all per-stage data for every entity is contiguous in memory, maximizing cache
//! utilization during forward/backward LP passes.
//!
//! ## Normal noise model
//!
//! Each entity follows: `x = μ_{e,m} + σ_{e,m} · f_{e,m,b} · ε`, where:
//!
//! - `μ_{e,m}` is the stage-level mean for entity `e` at stage `m`
//! - `σ_{e,m}` is the stage-level standard deviation for entity `e` at stage `m`
//! - `f_{e,m,b}` is the block scaling factor for entity `e` at stage `m`, block `b`
//! - `ε ~ N(0, 1)` is i.i.d. standard normal noise

use std::collections::HashMap;

use cobre_core::{scenario::LoadModel, temporal::Stage, EntityId};

use crate::StochasticError;

// ---------------------------------------------------------------------------
// Helper types
// ---------------------------------------------------------------------------

/// A `(block_id, factor)` pair used as input to [`PrecomputedNormalLp::build`].
///
/// `block_id` is the 0-based block index; `factor` is the multiplicative
/// scaling applied to the stage-level noise realization for that block.
pub type BlockFactorPair = (i32, f64);

/// A single entity-stage factor entry used as input to [`PrecomputedNormalLp::build`].
///
/// The tuple contains `(entity_id, stage_id, block_factors)` where
/// `block_factors` is a slice of [`BlockFactorPair`] values.
pub type EntityFactorEntry<'a> = (EntityId, i32, &'a [BlockFactorPair]);

// ---------------------------------------------------------------------------
// PrecomputedNormalLp
// ---------------------------------------------------------------------------

/// Cache-friendly normal noise model data for LP RHS patching.
///
/// Built once during initialization from raw normal noise model parameters.
/// Consumed read-only during iterative optimization.
///
/// All two-dimensional arrays use stage-major layout: outer dimension is stage
/// index, inner dimension is entity index (sorted by canonical entity ID order).
/// The three-dimensional factor array is additionally indexed by block within
/// each (stage, entity) pair.
///
/// See the [module documentation](self) for the array layout and noise model
/// description.
///
/// # Examples
///
/// ```
/// use cobre_core::{EntityId, scenario::LoadModel, temporal::{Stage, Block, BlockMode, StageStateConfig, StageRiskConfig, ScenarioSourceConfig, NoiseMethod}};
/// use cobre_stochastic::normal::precompute::PrecomputedNormalLp;
/// use chrono::NaiveDate;
///
/// let date = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();
/// let stage = Stage {
///     index: 0,
///     id: 0,
///     start_date: date,
///     end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
///     season_id: Some(0),
///     blocks: vec![Block { index: 0, name: "SINGLE".to_string(), duration_hours: 744.0 }],
///     block_mode: BlockMode::Parallel,
///     state_config: StageStateConfig { storage: true, inflow_lags: false },
///     risk_config: StageRiskConfig::Expectation,
///     scenario_config: ScenarioSourceConfig { branching_factor: 10, noise_method: NoiseMethod::Saa },
/// };
///
/// let model = LoadModel {
///     bus_id: EntityId(1),
///     stage_id: 0,
///     mean_mw: 100.0,
///     std_mw: 20.0,
/// };
///
/// let lp = PrecomputedNormalLp::build(&[model], &[], &[stage], &[EntityId(1)], 1).unwrap();
/// assert_eq!(lp.n_stages(), 1);
/// assert_eq!(lp.n_entities(), 1);
/// assert!((lp.mean(0, 0) - 100.0).abs() < f64::EPSILON);
/// assert!((lp.std(0, 0) - 20.0).abs() < f64::EPSILON);
/// assert!((lp.block_factor(0, 0, 0) - 1.0).abs() < f64::EPSILON);
/// ```
#[must_use]
#[derive(Debug)]
pub struct PrecomputedNormalLp {
    /// Stage-level mean per (stage, entity).
    /// Flat array indexed as `[stage * n_entities + entity_idx]`.
    /// Length: `n_stages * n_entities`.
    mean: Box<[f64]>,

    /// Stage-level standard deviation per (stage, entity).
    /// Flat array indexed as `[stage * n_entities + entity_idx]`.
    /// Length: `n_stages * n_entities`.
    std: Box<[f64]>,

    /// Per-block scaling factor per (stage, entity, block).
    /// Flat array indexed as `[stage * n_entities * max_blocks + entity_idx * max_blocks + block_idx]`.
    /// Length: `n_stages * n_entities * max_blocks`.
    /// Defaults to `1.0` for entries not present in the input factor data.
    factors: Box<[f64]>,

    /// Number of study stages.
    n_stages: usize,

    /// Number of entities.
    n_entities: usize,

    /// Maximum block count across all stages.
    max_blocks: usize,
}

impl PrecomputedNormalLp {
    // -----------------------------------------------------------------------
    // Builder
    // -----------------------------------------------------------------------

    /// Build a [`PrecomputedNormalLp`] from raw normal noise model parameters.
    ///
    /// # Parameters
    ///
    /// - `models`: raw stage-level mean and standard deviation parameters,
    ///   sorted by `(entity_id, stage_id)`. May contain entries for entity IDs
    ///   not present in `entity_ids`; those entries are silently ignored.
    /// - `factors`: per-(entity, stage, block) scaling factors. Each entry is
    ///   `(entity_id, stage_id, block_factors)` where `block_factors` is a
    ///   slice of `(block_id, factor)` pairs. Missing entries default to `1.0`.
    /// - `study_stages`: study stages in order (non-negative IDs).
    /// - `entity_ids`: canonical sorted entity IDs (determines entity array index order).
    /// - `max_blocks`: maximum block count across all stages.
    ///
    /// # Errors
    ///
    /// Currently returns `Ok` for all valid input combinations; missing
    /// (entity, stage) pairs default to zero mean/std and unit factors.
    /// The `Result` return type is retained for API consistency and to allow
    /// future validation without a breaking change.
    ///
    /// # Panics
    ///
    /// Does not panic for valid inputs. All indexing is bounds-checked during build.
    pub fn build(
        models: &[LoadModel],
        factors: &[EntityFactorEntry<'_>],
        study_stages: &[Stage],
        entity_ids: &[EntityId],
        max_blocks: usize,
    ) -> Result<Self, StochasticError> {
        let n_stages = study_stages.len();
        let n_entities = entity_ids.len();

        // Map entity EntityId → canonical index (0-based, canonical sorted order).
        let entity_index: HashMap<EntityId, usize> = entity_ids
            .iter()
            .enumerate()
            .map(|(i, &id)| (id, i))
            .collect();

        // Map stage_id → stage index.
        let stage_index: HashMap<i32, usize> = study_stages
            .iter()
            .enumerate()
            .map(|(i, s)| (s.id, i))
            .collect();

        // Allocate flat output arrays.
        let n2 = n_stages * n_entities;
        let n3 = n_stages * n_entities * max_blocks.max(1);
        let mut mean_arr = vec![0.0f64; n2];
        let mut std_arr = vec![0.0f64; n2];
        let mut factors_arr = vec![1.0f64; n3];

        // Populate mean and std from models.
        for model in models {
            let Some(&e_idx) = entity_index.get(&model.bus_id) else {
                continue;
            };
            let Some(&s_idx) = stage_index.get(&model.stage_id) else {
                continue;
            };
            let flat2 = s_idx * n_entities + e_idx;
            mean_arr[flat2] = model.mean_mw;
            std_arr[flat2] = model.std_mw;
        }

        // Populate block factors. Missing (entity, stage, block) combinations
        // remain at the default value of 1.0.
        if max_blocks > 0 {
            for (entity_id, stage_id, block_factors) in factors {
                let Some(&e_idx) = entity_index.get(entity_id) else {
                    continue;
                };
                let Some(&s_idx) = stage_index.get(stage_id) else {
                    continue;
                };
                for &(block_id, factor) in *block_factors {
                    let b_idx = usize::try_from(block_id).unwrap_or(usize::MAX);
                    if b_idx < max_blocks {
                        let flat3 = s_idx * n_entities * max_blocks + e_idx * max_blocks + b_idx;
                        factors_arr[flat3] = factor;
                    }
                }
            }
        }

        Ok(Self {
            mean: mean_arr.into_boxed_slice(),
            std: std_arr.into_boxed_slice(),
            factors: factors_arr.into_boxed_slice(),
            n_stages,
            n_entities,
            max_blocks,
        })
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    /// Stage-level mean for the given stage and entity indices.
    ///
    /// # Panics
    ///
    /// Panics if `stage >= n_stages` or `entity >= n_entities`.
    #[must_use]
    pub fn mean(&self, stage: usize, entity: usize) -> f64 {
        assert!(
            stage < self.n_stages,
            "stage index {stage} is out of bounds (n_stages = {})",
            self.n_stages
        );
        assert!(
            entity < self.n_entities,
            "entity index {entity} is out of bounds (n_entities = {})",
            self.n_entities
        );
        self.mean[stage * self.n_entities + entity]
    }

    /// Stage-level standard deviation for the given stage and entity indices.
    ///
    /// # Panics
    ///
    /// Panics if `stage >= n_stages` or `entity >= n_entities`.
    #[must_use]
    pub fn std(&self, stage: usize, entity: usize) -> f64 {
        assert!(
            stage < self.n_stages,
            "stage index {stage} is out of bounds (n_stages = {})",
            self.n_stages
        );
        assert!(
            entity < self.n_entities,
            "entity index {entity} is out of bounds (n_entities = {})",
            self.n_entities
        );
        self.std[stage * self.n_entities + entity]
    }

    /// Per-block scaling factor for the given stage, entity, and block indices.
    ///
    /// Returns `1.0` for any (stage, entity, block) combination that was not
    /// explicitly provided in the input factor data.
    ///
    /// # Panics
    ///
    /// Panics if `stage >= n_stages`, `entity >= n_entities`, or
    /// `block >= max_blocks`.
    #[must_use]
    pub fn block_factor(&self, stage: usize, entity: usize, block: usize) -> f64 {
        assert!(
            stage < self.n_stages,
            "stage index {stage} is out of bounds (n_stages = {})",
            self.n_stages
        );
        assert!(
            entity < self.n_entities,
            "entity index {entity} is out of bounds (n_entities = {})",
            self.n_entities
        );
        assert!(
            block < self.max_blocks,
            "block index {block} is out of bounds (max_blocks = {})",
            self.max_blocks
        );
        self.factors[stage * self.n_entities * self.max_blocks + entity * self.max_blocks + block]
    }

    /// Number of study stages.
    #[must_use]
    pub fn n_stages(&self) -> usize {
        self.n_stages
    }

    /// Number of entities.
    #[must_use]
    pub fn n_entities(&self) -> usize {
        self.n_entities
    }

    /// Maximum block count across all stages.
    #[must_use]
    pub fn max_blocks(&self) -> usize {
        self.max_blocks
    }
}

impl Default for PrecomputedNormalLp {
    /// Returns an empty [`PrecomputedNormalLp`] with zero stages, entities, and blocks.
    ///
    /// Useful as a sentinel value for callers that do not use the normal noise
    /// model (e.g., test fixtures for systems with no stochastic entities of
    /// this type).
    fn default() -> Self {
        Self {
            mean: Box::new([]),
            std: Box::new([]),
            factors: Box::new([]),
            n_stages: 0,
            n_entities: 0,
            max_blocks: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use chrono::NaiveDate;
    use cobre_core::{
        scenario::LoadModel,
        temporal::{
            Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
            StageStateConfig,
        },
        EntityId,
    };

    use super::{BlockFactorPair, EntityFactorEntry, PrecomputedNormalLp};

    fn dummy_date(year: i32, month: u32, day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(year, month, day).unwrap()
    }

    fn make_stage(index: usize, id: i32) -> Stage {
        Stage {
            index,
            id,
            start_date: dummy_date(2024, 1, 1),
            end_date: dummy_date(2024, 2, 1),
            season_id: Some(0),
            blocks: vec![Block {
                index: 0,
                name: "SINGLE".to_string(),
                duration_hours: 744.0,
            }],
            block_mode: BlockMode::Parallel,
            state_config: StageStateConfig {
                storage: true,
                inflow_lags: false,
            },
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor: 10,
                noise_method: NoiseMethod::Saa,
            },
        }
    }

    fn make_model(entity_id: i32, stage_id: i32, mean: f64, std: f64) -> LoadModel {
        LoadModel {
            bus_id: EntityId(entity_id),
            stage_id,
            mean_mw: mean,
            std_mw: std,
        }
    }

    // -----------------------------------------------------------------------
    // build_empty_returns_zero_mean_std_one_factors
    // -----------------------------------------------------------------------

    #[test]
    fn build_empty_returns_zero_mean_std_one_factors() {
        let stages = [make_stage(0, 0)];
        let entity_ids = [EntityId(1)];

        let lp = PrecomputedNormalLp::build(&[], &[], &stages, &entity_ids, 2).unwrap();

        assert_eq!(lp.n_stages(), 1);
        assert_eq!(lp.n_entities(), 1);
        assert_eq!(lp.max_blocks(), 2);

        assert!(
            (lp.mean(0, 0)).abs() < f64::EPSILON,
            "empty models → mean = 0.0"
        );
        assert!(
            (lp.std(0, 0)).abs() < f64::EPSILON,
            "empty models → std = 0.0"
        );
        assert!(
            (lp.block_factor(0, 0, 0) - 1.0).abs() < f64::EPSILON,
            "empty factors → factor = 1.0"
        );
        assert!(
            (lp.block_factor(0, 0, 1) - 1.0).abs() < f64::EPSILON,
            "empty factors → factor = 1.0 for block 1"
        );
    }

    // -----------------------------------------------------------------------
    // build_with_models_populates_mean_std
    // -----------------------------------------------------------------------

    #[test]
    fn build_with_models_populates_mean_std() {
        let stages = [make_stage(0, 0), make_stage(1, 1), make_stage(2, 2)];
        let entity_ids = [EntityId(10), EntityId(20)];

        let models = vec![
            make_model(10, 0, 100.0, 15.0),
            make_model(10, 1, 110.0, 18.0),
            make_model(10, 2, 95.0, 12.0),
            make_model(20, 0, 200.0, 30.0),
            make_model(20, 1, 210.0, 35.0),
            make_model(20, 2, 195.0, 28.0),
        ];

        let lp = PrecomputedNormalLp::build(&models, &[], &stages, &entity_ids, 1).unwrap();

        assert_eq!(lp.n_stages(), 3);
        assert_eq!(lp.n_entities(), 2);

        // entity 0 = EntityId(10), entity 1 = EntityId(20)
        assert!((lp.mean(0, 0) - 100.0).abs() < f64::EPSILON);
        assert!((lp.std(0, 0) - 15.0).abs() < f64::EPSILON);
        assert!((lp.mean(1, 0) - 110.0).abs() < f64::EPSILON);
        assert!((lp.std(1, 0) - 18.0).abs() < f64::EPSILON);
        assert!((lp.mean(2, 0) - 95.0).abs() < f64::EPSILON);
        assert!((lp.std(2, 0) - 12.0).abs() < f64::EPSILON);

        assert!((lp.mean(0, 1) - 200.0).abs() < f64::EPSILON);
        assert!((lp.std(0, 1) - 30.0).abs() < f64::EPSILON);
        assert!((lp.mean(2, 1) - 195.0).abs() < f64::EPSILON);
        assert!((lp.std(2, 1) - 28.0).abs() < f64::EPSILON);
    }

    // -----------------------------------------------------------------------
    // build_with_factors_populates_block_factors
    // -----------------------------------------------------------------------

    #[test]
    fn build_with_factors_populates_block_factors() {
        let stages = [make_stage(0, 0)];
        let entity_ids = [EntityId(1), EntityId(2)];

        let block_factors_entity2: &[BlockFactorPair] = &[(0, 0.85), (1, 1.15)];
        let factors: &[EntityFactorEntry<'_>] = &[(EntityId(2), 0, block_factors_entity2)];

        let lp = PrecomputedNormalLp::build(&[], factors, &stages, &entity_ids, 2).unwrap();

        // entity_ids[0] = EntityId(1), entity_ids[1] = EntityId(2)
        // entity 1 (EntityId(2)) has factors for stage 0
        assert!(
            (lp.block_factor(0, 1, 0) - 0.85).abs() < f64::EPSILON,
            "block 0 factor for entity 1 should be 0.85"
        );
        assert!(
            (lp.block_factor(0, 1, 1) - 1.15).abs() < f64::EPSILON,
            "block 1 factor for entity 1 should be 1.15"
        );

        // entity 0 (EntityId(1)) has no factors → defaults to 1.0
        assert!(
            (lp.block_factor(0, 0, 0) - 1.0).abs() < f64::EPSILON,
            "missing factor → 1.0"
        );
        assert!(
            (lp.block_factor(0, 0, 1) - 1.0).abs() < f64::EPSILON,
            "missing factor → 1.0"
        );
    }

    // -----------------------------------------------------------------------
    // build_with_missing_entity_stage_defaults_to_zero
    // -----------------------------------------------------------------------

    #[test]
    fn build_with_missing_entity_stage_defaults_to_zero() {
        let stages = [make_stage(0, 0), make_stage(1, 1)];
        let entity_ids = [EntityId(1), EntityId(2)];

        // Only entity 1 at stage 0 has a model
        let models = [make_model(1, 0, 150.0, 25.0)];

        let lp = PrecomputedNormalLp::build(&models, &[], &stages, &entity_ids, 1).unwrap();

        // Present entry
        assert!((lp.mean(0, 0) - 150.0).abs() < f64::EPSILON);
        assert!((lp.std(0, 0) - 25.0).abs() < f64::EPSILON);

        // Missing: entity 1 at stage 1
        assert!(
            (lp.mean(1, 0)).abs() < f64::EPSILON,
            "missing stage → mean = 0.0"
        );
        assert!(
            (lp.std(1, 0)).abs() < f64::EPSILON,
            "missing stage → std = 0.0"
        );

        // Missing: entity 2 at all stages
        assert!(
            (lp.mean(0, 1)).abs() < f64::EPSILON,
            "missing entity → mean = 0.0"
        );
        assert!(
            (lp.std(0, 1)).abs() < f64::EPSILON,
            "missing entity → std = 0.0"
        );
        assert!(
            (lp.mean(1, 1)).abs() < f64::EPSILON,
            "missing (entity, stage) → mean = 0.0"
        );
        assert!(
            (lp.std(1, 1)).abs() < f64::EPSILON,
            "missing (entity, stage) → std = 0.0"
        );
    }

    // -----------------------------------------------------------------------
    // build_with_missing_factor_defaults_to_one
    // -----------------------------------------------------------------------

    #[test]
    fn build_with_missing_factor_defaults_to_one() {
        let stages = [make_stage(0, 0), make_stage(1, 1)];
        let entity_ids = [EntityId(1), EntityId(2)];

        // Only entity 1 at stage 0, block 0 has a factor
        let present_factors: &[BlockFactorPair] = &[(0, 0.9)];
        let factors: &[EntityFactorEntry<'_>] = &[(EntityId(1), 0, present_factors)];

        let lp = PrecomputedNormalLp::build(&[], factors, &stages, &entity_ids, 3).unwrap();

        // Present entry
        assert!((lp.block_factor(0, 0, 0) - 0.9).abs() < f64::EPSILON);

        // Missing blocks for the same (stage, entity) → 1.0
        assert!((lp.block_factor(0, 0, 1) - 1.0).abs() < f64::EPSILON);
        assert!((lp.block_factor(0, 0, 2) - 1.0).abs() < f64::EPSILON);

        // Entirely missing (entity, stage) combinations → 1.0
        assert!((lp.block_factor(0, 1, 0) - 1.0).abs() < f64::EPSILON);
        assert!((lp.block_factor(1, 0, 0) - 1.0).abs() < f64::EPSILON);
        assert!((lp.block_factor(1, 1, 0) - 1.0).abs() < f64::EPSILON);
    }

    // -----------------------------------------------------------------------
    // accessor_consistency_across_stages
    // -----------------------------------------------------------------------

    #[test]
    fn accessor_consistency_across_stages() {
        // 2 entities, 3 stages, 2 blocks — the acceptance criterion scenario.
        let stages = [make_stage(0, 0), make_stage(1, 1), make_stage(2, 2)];
        let entity_ids = [EntityId(5), EntityId(7)];

        let models = vec![
            // EntityId(5) = entity_idx 0
            make_model(5, 0, 50.0, 5.0),
            make_model(5, 1, 55.0, 6.0),
            make_model(5, 2, 48.0, 4.0),
            // EntityId(7) = entity_idx 1
            make_model(7, 0, 70.0, 8.0),
            make_model(7, 1, 75.0, 9.0),
            make_model(7, 2, 68.0, 7.0),
        ];

        let bf_e0_s1: &[BlockFactorPair] = &[(0, 0.8), (1, 1.2)];
        let bf_e1_s2: &[BlockFactorPair] = &[(0, 0.95), (1, 1.05)];
        let factors: &[EntityFactorEntry<'_>] =
            &[(EntityId(5), 1, bf_e0_s1), (EntityId(7), 2, bf_e1_s2)];

        let lp = PrecomputedNormalLp::build(&models, factors, &stages, &entity_ids, 2).unwrap();

        assert_eq!(lp.n_stages(), 3);
        assert_eq!(lp.n_entities(), 2);
        assert_eq!(lp.max_blocks(), 2);

        // mean(1, 0): stage_idx=1 maps to stage id=1, entity_idx=0 maps to EntityId(5)
        assert!(
            (lp.mean(1, 0) - 55.0).abs() < f64::EPSILON,
            "mean(1, 0) should be 55.0 (stage id=1, EntityId(5))"
        );
        assert!((lp.std(1, 0) - 6.0).abs() < f64::EPSILON);

        // Check factors: EntityId(5) at stage 1 → block 0=0.8, block 1=1.2
        assert!((lp.block_factor(1, 0, 0) - 0.8).abs() < f64::EPSILON);
        assert!((lp.block_factor(1, 0, 1) - 1.2).abs() < f64::EPSILON);

        // EntityId(7) at stage 2 → block 0=0.95, block 1=1.05
        assert!((lp.block_factor(2, 1, 0) - 0.95).abs() < f64::EPSILON);
        assert!((lp.block_factor(2, 1, 1) - 1.05).abs() < f64::EPSILON);

        // EntityId(5) at stage 0 → no factors → 1.0
        assert!((lp.block_factor(0, 0, 0) - 1.0).abs() < f64::EPSILON);
        assert!((lp.block_factor(0, 0, 1) - 1.0).abs() < f64::EPSILON);

        // All stage/entity combinations have correct mean/std
        assert!((lp.mean(0, 0) - 50.0).abs() < f64::EPSILON);
        assert!((lp.mean(2, 0) - 48.0).abs() < f64::EPSILON);
        assert!((lp.mean(0, 1) - 70.0).abs() < f64::EPSILON);
        assert!((lp.mean(1, 1) - 75.0).abs() < f64::EPSILON);
        assert!((lp.mean(2, 1) - 68.0).abs() < f64::EPSILON);
    }

    // -----------------------------------------------------------------------
    // Acceptance criterion: mean(1, 0) returns correct value for 2 entities,
    // 3 stages setup
    // -----------------------------------------------------------------------

    #[test]
    fn acceptance_criterion_mean_stage1_entity0() {
        let stages = [make_stage(0, 0), make_stage(1, 1), make_stage(2, 2)];
        let entity_ids = [EntityId(3), EntityId(9)];

        let models = vec![
            make_model(3, 0, 120.0, 10.0),
            make_model(3, 1, 130.0, 11.0), // stage_ids[1] = id 1, entity_ids[0] = EntityId(3)
            make_model(3, 2, 115.0, 9.0),
            make_model(9, 0, 220.0, 20.0),
            make_model(9, 1, 230.0, 22.0),
            make_model(9, 2, 210.0, 18.0),
        ];

        let lp = PrecomputedNormalLp::build(&models, &[], &stages, &entity_ids, 1).unwrap();

        // mean(1, 0): stage_idx=1 corresponds to study_stages[1] (id=1),
        //             entity_idx=0 corresponds to entity_ids[0] = EntityId(3)
        assert!(
            (lp.mean(1, 0) - 130.0).abs() < f64::EPSILON,
            "mean(1, 0) should equal the LoadModel with EntityId(3) at stage_id=1: \
             expected 130.0, got {}",
            lp.mean(1, 0)
        );
    }

    // -----------------------------------------------------------------------
    // Declaration-order invariance
    // -----------------------------------------------------------------------

    #[test]
    fn declaration_order_invariance() {
        let stage = make_stage(0, 0);
        let entity_ids = [EntityId(3), EntityId(7)];

        // Models provided in reverse entity order
        let models = vec![make_model(7, 0, 200.0, 30.0), make_model(3, 0, 100.0, 15.0)];

        let lp = PrecomputedNormalLp::build(&models, &[], &[stage], &entity_ids, 1).unwrap();

        // entity_ids[0] = EntityId(3) → mean 100.0
        // entity_ids[1] = EntityId(7) → mean 200.0
        assert!(
            (lp.mean(0, 0) - 100.0).abs() < f64::EPSILON,
            "entity index 0 should be EntityId(3) with mean=100, got {}",
            lp.mean(0, 0)
        );
        assert!(
            (lp.mean(0, 1) - 200.0).abs() < f64::EPSILON,
            "entity index 1 should be EntityId(7) with mean=200, got {}",
            lp.mean(0, 1)
        );
    }

    // -----------------------------------------------------------------------
    // Bounds-checking panics
    // -----------------------------------------------------------------------

    #[test]
    #[should_panic(expected = "stage index 1 is out of bounds")]
    fn mean_out_of_bounds_panics() {
        let stage = make_stage(0, 0);
        let model = make_model(1, 0, 100.0, 20.0);
        let lp = PrecomputedNormalLp::build(&[model], &[], &[stage], &[EntityId(1)], 1).unwrap();
        let _ = lp.mean(1, 0);
    }

    #[test]
    #[should_panic(expected = "entity index 1 is out of bounds")]
    fn mean_entity_out_of_bounds_panics() {
        let stage = make_stage(0, 0);
        let model = make_model(1, 0, 100.0, 20.0);
        let lp = PrecomputedNormalLp::build(&[model], &[], &[stage], &[EntityId(1)], 1).unwrap();
        let _ = lp.mean(0, 1);
    }

    #[test]
    #[should_panic(expected = "stage index 1 is out of bounds")]
    fn std_out_of_bounds_panics() {
        let stage = make_stage(0, 0);
        let model = make_model(1, 0, 100.0, 20.0);
        let lp = PrecomputedNormalLp::build(&[model], &[], &[stage], &[EntityId(1)], 1).unwrap();
        let _ = lp.std(1, 0);
    }

    #[test]
    #[should_panic(expected = "block index 1 is out of bounds")]
    fn block_factor_out_of_bounds_panics() {
        let stage = make_stage(0, 0);
        let lp = PrecomputedNormalLp::build(&[], &[], &[stage], &[EntityId(1)], 1).unwrap();
        let _ = lp.block_factor(0, 0, 1);
    }

    // -----------------------------------------------------------------------
    // Default is empty
    // -----------------------------------------------------------------------

    #[test]
    fn default_is_empty() {
        let lp = PrecomputedNormalLp::default();
        assert_eq!(lp.n_stages(), 0);
        assert_eq!(lp.n_entities(), 0);
        assert_eq!(lp.max_blocks(), 0);
    }
}
