//! [`LpBuildInputs`]: the builder's resolved study inputs, each resolved once
//! by setup's `resolve_lp_build_inputs`.

use std::collections::{BTreeMap, HashMap};

use cobre_core::EntityId;
use cobre_core::scenario::LoadModel;

use crate::indexer::{EntityPositions, HydroCellIndex, StudyDimensions};
use crate::resolved_parameters::ResolvedParameters;
use crate::time_value::TimeValue;

/// Resolved study inputs [`build_stage_templates`](super::build_stage_templates)
/// consumes by value; every field is the single derivation of its fact, either
/// owned here or borrowed from the same setup `resolve_stage_data` step that
/// resolves the other borrowed fields.
pub(crate) struct LpBuildInputs<'a> {
    /// Canonical entity-id → slot maps for every position-addressed family.
    pub(crate) positions: EntityPositions,
    /// Per-stage minimum target-storage trajectory, keyed `(hydro_idx,
    /// stage_id) → V_target` \[hm³\].
    pub(crate) filling_v_target: BTreeMap<(usize, i32), f64>,
    /// Declared load-balance models for buses outside the stochastic
    /// load-noise membership.
    pub(crate) deterministic_load_models: Vec<LoadModel>,
    /// Bus-slice positions of the stochastic load-noise-member buses.
    pub(crate) load_bus_indices: Vec<usize>,
    /// Target hydro ID → system indices of hydros diverting to it.
    pub(crate) diversion_upstream: HashMap<EntityId, Vec<usize>>,
    /// Per-stage hydro productivities (MW per m³/s); FPHA hydros carry `0.0`.
    pub(crate) hydro_productivities_per_stage: Vec<Vec<f64>>,
    /// Study-invariant, non-state LP shape, from setup's
    /// `build_study_dimensions`.
    pub(crate) study_dims: &'a StudyDimensions,
    /// Present-value discounting and delivery hours/ids, from
    /// [`crate::time_value::TimeValue::from_system`].
    pub(crate) time_value: &'a TimeValue,
    /// Study-scope hydro-cell partition, from [`HydroCellIndex::build`].
    pub(crate) hydro_cell_index: &'a HydroCellIndex,
    /// Resolved `(parameter_id, stage_idx, block_idx)` table, from
    /// [`crate::resolved_parameters::build_resolved_parameters`].
    pub(crate) resolved_parameters: &'a ResolvedParameters,
}
