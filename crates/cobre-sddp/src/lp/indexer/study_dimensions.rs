//! The [`StudyDimensions`] single owner of the study-invariant, non-state LP
//! shape: the scalar entity counts, presence flags, and anticipated-plant set
//! constant across every stage and block of a study and not part of the state
//! vector. No other long-lived type holds these facts.

use super::AnticipatedPlants;

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
// Rationale: the four bool fields are independent presence flags for optional
// column groups, not states of one machine; the lint's suggested enum/state-
// machine refactor would obscure that they vary independently.
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone, Default)]
pub struct StudyDimensions {
    /// Number of thermal units (T).
    pub n_thermals: usize,
    /// Number of transmission lines (`L_n`).
    pub n_lines: usize,
    /// Number of buses (B).
    pub n_buses: usize,
    /// Maximum number of deficit segments across all buses (S).
    pub max_deficit_segments: usize,
    /// Whether the study has NCS generation columns; only presence lives here,
    /// the per-`(ncs, block)` column base is addressed per stage.
    pub has_ncs: bool,
    /// Whether inflow non-negativity penalty slack columns are present.
    pub has_inflow_penalty: bool,
    /// Whether withdrawal slack columns are present (`hydro_count > 0`).
    pub has_withdrawal: bool,
    /// Whether operational violation slack columns are present.
    pub has_operational_violations: bool,
    /// The study's anticipated-plant set.
    pub anticipated_plants: AnticipatedPlants,
    /// Number of pumping stations.
    pub n_pumping: usize,
    /// PAR order of the downstream (coarser) resolution model. Non-zero only when
    /// the study includes stages with `season_id >= 12` (a monthly-to-quarterly
    /// transition); zero for uniform-resolution studies.
    pub downstream_par_order: usize,
}
