//! LP layout index map for SDDP stage subproblems.
//!
//! The state-vector column layout is owned by [`StateSpace`]; the per-stage
//! equipment column/row geometry is owned by
//! [`StageLayout`](crate::lp::builder)/[`StageGeometry`](crate::lp::builder::StageGeometry);
//! the non-state study shape is owned by [`StudyDimensions`]. The authoritative
//! ranges live on the owning types.
//!
//! ## Column layout (Solver Abstraction SS2.1)
//!
//! The stage-invariant state-vector column ranges (storage, AR lags,
//! travel-time buckets, the anticipated ring's outgoing and incoming blocks,
//! `z_inflow`, `theta`) are owned entirely by [`StateSpace`] — see its own
//! module doc for the authoritative diagram; this file does not re-derive it,
//! to avoid the two copies drifting apart.
//!
//! The equipment, slack, generic-constraint, and filling-phase column and row
//! ranges that follow `theta` — allocated in that equipment -> slack ->
//! generic -> filling family order — are owned entirely by
//! [`StageLayout`](crate::lp::builder).
//!
//! The `anticipated_decision` block is stage-level (one column per anticipated
//! plant, NOT per-block) and has length `A = n_anticipated`. The block collapses
//! to length 0 when `n_anticipated == 0`, leaving the rest of the layout
//! byte-identical to the pre-anticipated form. The control region runs
//! `anticipated_decision` then `line_fwd` directly — the anticipated ring's
//! outgoing slots (`StateSpace::commit_out`) do NOT live here:
//! they sit in the stage-invariant state region owned by [`StateSpace`], so
//! their address never depends on `n_blks`. An `anticipated_state_out_def`
//! equality row pins the plant's own newest ring slot to its
//! `anticipated_decision` column.
//!
//! ## Row layout (Solver Abstraction SS2.2)
//!
//! State pinning uses column bounds (`set_col_bounds`) on the incoming-state
//! columns, so the LP has no state-fixing row range. z-inflow rows start at
//! row 0.
//!
//! The per-solve patch sequence layered on top of this geometry is documented in
//! [`crate::lp::builder`].
//!
//! # Submodule layout
//!
//! - `anticipated_gate` — the anticipated-decision temporal-gating free
//!   functions (`is_anticipated_decision_active_for_delivery`,
//!   `anticipated_resolution_for`), plus the anticipated ring's
//!   delivery-axis → ring-slot sweep (`for_each_ring_residue`) and its
//!   readiness-filtered form (`for_each_live_commitment_slot`).
//! - `anticipated_plants` — the [`AnticipatedPlants`] typed owner of the
//!   anticipated-plant set.
//! - `entity_positions` — the [`EntityPositions`] typed owner: canonical
//!   `EntityId -> slot` for every position-addressed entity family.
//! - `layout` — the per-stage geometry satellite type [`EvaporationIndices`]
//!   (locating one hydro's evaporation columns/row within a stage LP).
//! - `index` — the base typed vocabulary: [`StateDim`], [`InCol`]/[`OutCol`]
//!   (incoming/outgoing column roles), [`BlockIdx`] (block operand), and
//!   [`Boundary`] (chronological storage-boundary operand).
//! - `block_grid` — the [`BlockGrid`] typed block-stride address primitive and
//!   its three shape methods ([`BlockGrid::flat`], [`BlockGrid::fpha_plane`],
//!   [`BlockGrid::deficit`]).
//! - `block_row_family` — the [`BlockRowFamily`] typed block-major row-family
//!   address primitive, the single owner of the one-row-per-entity vs.
//!   per-block collapse.
//! - `range_cursor` — the `RangeCursor` running column/row offset allocator
//!   shared by [`StageLayout`](crate::lp::builder)'s per-stage equipment chains
//!   and [`StateSpace`]'s stage-invariant state-vector chain.
//! - `storage_boundary_grid` — the [`StorageBoundaryGrid`] typed
//!   storage-boundary address primitive ([`StorageBoundaryGrid::col`]), the
//!   single owner of `block_storage_col`'s endpoint/interior split.
//! - `state_space` — the [`StateSpace`] type, the sole owner of the
//!   state-vector concern: the stage-invariant state-vector column ranges, the
//!   two layout-derived caches, and the resolver / mask methods
//!   ([`StateSpace::state_to_lp_column`],
//!   [`StateSpace::state_to_lp_incoming_column`],
//!   [`StateSpace::lp_column_for_state`], [`StateSpace::set_nonzero_mask`]). It
//!   finalizes both caches in its single constructor; downstream code threads a
//!   handle to it.
//! - `study_dimensions` — the [`StudyDimensions`] type, the single owner of the
//!   study-invariant non-state LP shape.
//! - `cut_state_projection` — the [`CutStateProjection`] type, a storage-scoped
//!   projection of [`StateSpace`] exposing only the cut-state dimensions a
//!   stage's `StageStateConfig` enables (anticipated state always included),
//!   delegating each column to [`StateSpace::state_to_lp_incoming_column`].
//! - `entity_index` — entity system/local index vocabulary
//!   ([`HydroSys`]/[`ThermalSys`]/[`LineSys`]/[`BusSys`]/[`NcsSys`]/
//!   [`PumpingSys`], [`FphaLocal`]/[`EvapLocal`]/[`FillingTargetLocal`]/
//!   [`FloorLocal`]/[`AnticipatedLocal`], [`HydroCell`]/[`FphaCellLocal`]).
//! - `hydro_cell` — the [`HydroCellIndex`] partition and [`HydroCell`] type.
//!
//! Every public symbol is re-exported here so the `cobre_sddp::indexer::Symbol`
//! and `crate::indexer::Symbol` module paths resolve to the same item regardless
//! of which submodule owns it.

mod anticipated_gate;
mod anticipated_plants;
mod block_grid;
mod block_row_family;
mod cut_state_projection;
mod entity_index;
mod entity_positions;
mod hydro_cell;
mod index;
mod layout;
mod range_cursor;
mod state_space;
mod storage_boundary_grid;
mod study_dimensions;

pub(crate) use anticipated_gate::{
    anticipated_resolution_for, for_each_live_commitment_slot, for_each_ring_residue,
    is_anticipated_decision_active_for_delivery,
};
pub use anticipated_plants::AnticipatedPlants;
pub use block_grid::BlockGrid;
pub use block_row_family::BlockRowFamily;
pub use cut_state_projection::CutStateProjection;
pub use entity_index::{
    AnticipatedLocal, BusSys, EvapLocal, FillingTargetLocal, FloorLocal, FphaCellLocal, FphaLocal,
    HydroCell, HydroSys, LineSys, NcsSys, PumpingSys, ThermalSys,
};
pub(crate) use entity_positions::EntityPositions;
pub use hydro_cell::HydroCellIndex;
pub use index::{BlockIdx, Boundary, CutSlot, InCol, OutCol, StateDim};
pub use layout::EvaporationIndices;
pub(crate) use range_cursor::RangeCursor;
pub use state_space::StateSpace;
pub(crate) use state_space::{REGION_ORDER, StateRegion};
pub use storage_boundary_grid::StorageBoundaryGrid;
pub use study_dimensions::StudyDimensions;
