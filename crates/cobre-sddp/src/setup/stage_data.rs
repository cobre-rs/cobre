//! Stage-indexed data sub-struct extracted from [`super::StudySetup`].

use cobre_core::{Stage, temporal::StageLagTransition};

use crate::{
    indexer::StageIndexer, lp_builder::StageTemplates, scaling_report::ScalingReport,
    simulation::EntityCounts,
};

/// All per-stage and stage-indexed data owned by [`super::StudySetup`].
///
/// Groups the eight fields that describe the study's temporal and stage
/// structure. Constructed once during [`super::StudySetup::from_broadcast_params`]
/// and borrowed for hot-path context construction.
#[derive(Debug)]
pub struct StageData {
    /// LP skeleton templates, one per study stage.
    pub stage_templates: StageTemplates,

    /// Stage indexer: LP column/row offsets and equipment counts.
    pub(crate) indexer: StageIndexer,

    /// Study stages (id >= 0) in index order.
    ///
    /// Borrowed by `TrainingContext` so that
    /// [`cobre_stochastic::build_forward_sampler`] can read per-stage noise
    /// methods when constructing an `OutOfSample` sampler.
    pub(crate) stages: Vec<Stage>,

    /// Entity IDs and productivities for all dispatch entities.
    pub(crate) entity_counts: EntityCounts,

    /// Number of blocks per stage.
    pub(crate) block_counts_per_stage: Vec<usize>,

    /// Precomputed lag accumulation weights and period-finalization flags,
    /// one entry per study stage. Indexed by stage: `stage_lag_transitions[t]`.
    ///
    /// Computed once at setup time by
    /// [`crate::lag_transition::precompute_stage_lag_transitions`].
    pub(crate) stage_lag_transitions: Vec<StageLagTransition>,

    /// Pre-computed noise group assignments for noise sharing via noise-group sharing.
    ///
    /// Stages with the same `(season_id, year)` share a noise group. Computed at
    /// setup time by `precompute_noise_groups`.
    pub noise_group_ids: Vec<u32>,

    /// LP scaling report captured during template build.
    pub scaling_report: ScalingReport,
}
