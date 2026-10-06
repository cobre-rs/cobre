//! The [`StudyDimensions`] single owner of the study-invariant, non-state LP
//! shape: the maximum deficit-segment count, the inflow non-negativity
//! method, the anticipated-plant set, and the downstream PAR order —
//! constant across every stage and block of a study and not part of the
//! state vector. No other long-lived type holds these facts.

use super::AnticipatedPlants;
use crate::inflow_method::InflowNonNegativityMethod;

/// Study-invariant, non-state LP shape for an SDDP study.
///
/// The exclusions are deliberate — duplicating a dim owned elsewhere would
/// reintroduce the multi-owner drift this type exists to remove:
///
/// - **State-defining dims** (`hydro_count`, `max_par_order`, `n_anticipated`,
///   `k_max`, `anticipated_lead_stages`) are owned solely by
///   [`StateSpace`](super::StateSpace): a count is either state-defining (→
///   `StateSpace`) or non-state study shape (→ here), never both.
/// - **`n_blks`** is *per-stage*, read from a stage's own template. A persisted
///   global `n_blks` is the footgun that mis-strides equipment columns at any
///   stage whose block count differs from stage 0's.
///
/// `anticipated_plants` is study-invariant, so it is owned here; the
/// per-stage FPHA / evaporation identity lists vary by stage and are owned by
/// the per-stage geometry.
#[derive(Debug, Clone)]
pub struct StudyDimensions {
    /// Maximum number of deficit segments across all buses (S).
    pub max_deficit_segments: usize,
    /// Inflow non-negativity enforcement method.
    pub inflow_method: InflowNonNegativityMethod,
    /// The study's anticipated-plant set.
    pub anticipated_plants: AnticipatedPlants,
    /// PAR order of the downstream (coarser) resolution model. Non-zero only when
    /// the study steps from a month-long season to a quarter-long one
    /// (`derive_downstream_par_order`); zero otherwise.
    pub downstream_par_order: usize,
}

impl Default for StudyDimensions {
    fn default() -> Self {
        Self {
            max_deficit_segments: 0,
            inflow_method: InflowNonNegativityMethod::None,
            anticipated_plants: AnticipatedPlants::default(),
            downstream_par_order: 0,
        }
    }
}
