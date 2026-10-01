//! Stage LP patch buffer and stage template builder for SDDP subproblems.
//!
//! - [`PatchBuffer`]: parallel arrays consumed by `set_row_bounds` /
//!   `set_col_bounds`, filled with scenario-dependent values before each LP solve.
//!   Allocated once at training start and reused across every iteration/stage — the
//!   training loop fills it millions of times.
//! - [`build_stage_templates`]: one `StageTemplate` per stage encoding the full
//!   structural LP (CSC matrix, bounds, objective), built once and shared read-only.
//!
//! The column/row geometry — regions, ordering, offset arithmetic — is owned per
//! stage by `StageLayout` (state-vector region on [`crate::indexer::StateSpace`],
//! non-state shape on [`crate::indexer::StudyDimensions`]); this module documents
//! only the per-solve patch sequence layered on top.
//!
//! ## State pinning
//!
//! State pinning lives on **incoming-state columns**, not rows: the LP has no
//! state-fixing row range. Both forward-pass pinning (`set_col_bounds`) and
//! backward-pass cut-subgradient extraction resolve the same column via
//! [`crate::indexer::StateSpace::state_to_lp_incoming_column`].
//!
//! ## Patch sequence
//!
//! Each forward-pass solve writes the row buffer (load balance when
//! `n_load_buses > 0`, z-inflow) via `fill_load_patches` /
//! `fill_z_inflow_patches`, and the column buffer (incoming storage,
//! AR lags, travel-time buckets, anticipated state) via `fill_col_state_patches`.
//! The backward pass writes only the column buffer; noise comes from the fixed
//! opening tree through `fill_z_inflow_patches` with the opening-specific vector.
//!
//! ## Commissioning window
//!
//! [`cobre_core::commissioning::commissioning_active`] returns `true`/`false`, not
//! an active-subset index: under the dense layout an inactive entity keeps its LP
//! column (callers force its bounds to `[0, 0]`), so the column position is the
//! entity's system index at every stage and no per-stage active-set remap is
//! needed. Every per-phase gating site derived from
//! [`cobre_core::commissioning::filling_phase`] — column bounds, row emission, FPHA
//! exclusion — recomputes the phase by calling it; no caller may cache a per-stage
//! [`cobre_core::commissioning::Phase`] mask.

mod build_inputs;
mod columns;
pub(crate) mod delivery_ring;
mod entries;
mod fpha_cursor;
mod generic_constraints;
mod hydro_state;
mod layout;
mod patch;
mod rows;
mod scaling;
pub(crate) mod state_box;
mod template;

#[cfg(test)]
mod test_support;

// --- Public re-exports (stable API) ---
#[cfg(any(test, feature = "test-support"))]
pub use delivery_ring::DeliveryRing;
pub use layout::StageGeometry;
pub use patch::PatchBuffer;
pub use state_box::StateBox;
pub use template::StageTemplates;

// --- Crate-internal re-exports ---
pub(crate) use build_inputs::LpBuildInputs;
#[cfg(any(test, feature = "test-support"))]
pub(crate) use layout::ResolvedTables;
#[cfg(any(test, feature = "test-support"))]
pub(crate) use layout::{StageLayout, TemplateBuildCtx};
pub(crate) use layout::{contract_family_slot, evaporation_slot, evaporation_slot_count};
pub(crate) use scaling::{
    apply_col_scale, apply_commitment_hold_col_scale_unscale, apply_row_scale, compute_col_scale,
    compute_row_scale,
};
pub(crate) use state_box::build_state_box;
#[cfg(any(test, feature = "test-support"))]
pub(crate) use template::canonical::{
    FactGroups, encode_stage_templates_facts, encode_time_value_facts,
};
pub(crate) use template::{build_stage_templates, models_from_normal};

// ---------------------------------------------------------------------------
// Shared constants
// ---------------------------------------------------------------------------

/// Margin on the symmetric magnitude bound `[-q_max, +q_max]` of the evaporation
/// outflow variable, absorbing linearization error where the area-volume curve
/// exceeds the linear estimate near `v_max`. Symmetric because that error runs both
/// directions: a negative evaporation-outflow value is net rainfall input (inflow),
/// a positive one is evaporative outflow.
pub(crate) const EVAPORATION_FLOW_SAFETY_MARGIN: f64 = 2.0;

/// Number of LP columns per `(evaporating hydro, slot)` triple: evaporation
/// outflow, `f_evap_plus`, `f_evap_minus`. `StageLayout::evap_triple_base` is
/// the single owner of the base-column address; this const only fixes the
/// per-triple column width the indexer's `EvaporationIndices` constructor and
/// `StageLayout`'s evaporation accessors both multiply by.
pub(crate) const EVAP_COLS_PER_HYDRO: usize = 3;

/// Offset of the signed evaporation-outflow column within a hydro's evaporation
/// block (a negative value reads as net rainfall input).
pub(crate) const EVAP_FLOW_OFFSET: usize = 0;

/// Offset of the `f_evap_plus` (under-evaporation) slack column. Swapping with
/// [`EVAP_F_MINUS_OFFSET`] compiles and silently misplaces the directional
/// evaporation-violation slacks onto each other's columns.
pub(crate) const EVAP_F_PLUS_OFFSET: usize = 1;

/// Offset of the `f_evap_minus` (over-evaporation) slack column. Swapping with
/// [`EVAP_F_PLUS_OFFSET`] compiles and silently misplaces the directional
/// evaporation-violation slacks onto each other's columns.
pub(crate) const EVAP_F_MINUS_OFFSET: usize = 2;

// ---------------------------------------------------------------------------
// Shared types
// ---------------------------------------------------------------------------

/// Per-row metadata for one active generic constraint row at a single stage, used
/// by the LP builder (CSC entries, bounds, objective) and the simulation extraction
/// pipeline (LP index → constraint identity + block).
///
/// One entry per active `(constraint, block)` pair, except: a `block_id = None`
/// bound whose expression is **block-independent** collapses to a *single*
/// stage-level entry (`is_stage_level = true`), since the replicated rows would be
/// identical. A `block_id = None` bound on a block-level expression still generates
/// one entry per block; a `block_id = Some(k)` bound generates exactly one.
#[derive(Debug, Clone)]
pub struct GenericConstraintRowEntry {
    /// Index into `System::generic_constraints()` for the parent constraint.
    pub constraint_idx: usize,
    /// Entity ID of the parent constraint (copied from `GenericConstraint::id`).
    pub entity_id: i32,
    /// Block index within the stage (0-indexed); the sentinel `0` for a collapsed
    /// stage-level row (`is_stage_level = true`), which resolves the same column for
    /// any block.
    pub block_idx: usize,
    /// Whether this row is a collapsed stage-level row; when `true` the slack column
    /// is priced by the stage's total hours, not `block_idx`'s block hours.
    pub is_stage_level: bool,
    /// Lower right-hand-side endpoint; `None` when the row is upper-only.
    pub bound_lower: Option<f64>,
    /// Upper right-hand-side endpoint; `None` when the row is lower-only.
    pub bound_upper: Option<f64>,
    /// Whether slack is enabled for this constraint.
    pub slack_enabled: bool,
    /// Penalty cost per unit of slack violation (`0.0` when slack is disabled).
    pub slack_penalty: f64,
    /// Positive-violation slack (`slack_plus`) column; `None` when slack is disabled.
    pub slack_plus_col: Option<usize>,
    /// Negative-violation slack (`slack_minus`) column, present only when slack is
    /// enabled and the row is two-sided (both `bound_lower` and `bound_upper`
    /// present); `None` otherwise.
    pub slack_minus_col: Option<usize>,
}
