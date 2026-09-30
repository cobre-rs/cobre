//! Stage-indexed data sub-struct of [`super::SolveInputs`].

use cobre_core::{ContractType, Stage, temporal::StageLagTransition};

use crate::{
    lp::builder::StageTemplates,
    lp::indexer::{HydroCellIndex, StateSpace, StudyDimensions},
    scaling_report::ScalingReport,
    simulation::EntityCounts,
    time_value::TimeValue,
};

/// All per-stage and stage-indexed data owned by [`super::SolveInputs`],
/// constructed once during [`super::StudySetup::from_broadcast_params`] and
/// borrowed for hot-path context construction.
#[derive(Debug)]
pub struct StageData {
    /// LP skeleton templates, one per study stage.
    pub stage_templates: StageTemplates,

    /// Present-value discounting and delivery hours/ids/post-study calendar,
    /// resolved once in `build_energy_and_templates` — the single owner every
    /// LP-build and reporting reader borrows from.
    pub(crate) time_value: TimeValue,

    /// Canonical stage-invariant state / cut column ranges and layout-derived
    /// caches — the single owner of these ("role (a)"); per-stage equipment
    /// geometry ("role (b)") lives on [`crate::lp::builder::StageGeometry`].
    pub(crate) state: StateSpace,

    /// Single owner of the study-invariant, non-state LP shape: non-state entity
    /// counts, optional-column presence flags, and the anticipated-thermal
    /// identity list. State-defining dims live on [`Self::state`], per-stage
    /// `n_blks` on the per-stage geometry.
    pub(crate) study_dims: StudyDimensions,

    /// Single owner of each plant's `unit_groups` partition into `bus_id`
    /// cells, built once from `System::hydros` and stage-invariant like
    /// [`Self::study_dims`]. The partition is the identity map for every
    /// study without multi-bus groups (see [`HydroCellIndex`] module docs).
    pub(crate) hydro_cell_index: HydroCellIndex,

    /// Study stages (id >= 0) in index order.
    pub(crate) stages: Vec<Stage>,

    /// Entity IDs and productivities for all dispatch entities.
    pub(crate) entity_counts: EntityCounts,

    /// Per-station pumping power-consumption rate \[MW/(m³/s)\], ID-sorted to
    /// match `entity_counts.pumping_station_ids` (positionally aligned).
    pub(crate) pumping_consumption_mw_per_m3s: Vec<f64>,

    /// Per-stage RESOLVED contract price \[$/`MWh`\]: one inner `Vec` per study
    /// stage, flat with the per-stage stride `stage_templates.geometry_per_stage[t].n_blks`
    /// — index `c * n_blks + blk`, `c` ID-sorted to match `entity_counts.contract_ids`. The
    /// resolved, possibly block-overridden `contract_bounds_at_block(c, t, blk).price_per_mwh`.
    pub(crate) contract_prices_per_stage: Vec<Vec<f64>>,

    /// Per-contract `(ContractType, per-family slot)`, ID-sorted to match
    /// `entity_counts.contract_ids`. Stage-invariant.
    pub(crate) contract_slots: Vec<(ContractType, usize)>,

    /// Precomputed lag accumulation weights and period-finalization flags,
    /// one entry per study stage.
    pub(crate) stage_lag_transitions: Vec<StageLagTransition>,

    /// Noise group assignments: stages with the same `(season_id, year)` share a
    /// noise group.
    pub noise_group_ids: Vec<u32>,

    /// LP scaling report captured during template build.
    pub scaling_report: ScalingReport,
}
