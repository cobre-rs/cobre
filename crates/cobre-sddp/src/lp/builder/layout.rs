use std::collections::{BTreeMap, HashMap};
use std::ops::Range;

use cobre_core::commissioning::Phase;
use cobre_core::{
    AffineBound, BlockMode, Bus, CascadeTopology, CoefficientRef, ConstraintExpression,
    ContractType, EnergyContract, EntityId, GenericConstraint, Hydro, Line, LoadModel,
    NonControllableSource, PumpingStation, ResolvedBounds, ResolvedGenericConstraintBounds,
    ResolvedLoadFactors, ResolvedNcsBounds, ResolvedNcsFactors, ResolvedPenalties, SlackConfig,
    Stage, Thermal, VariableRef,
};
use cobre_stochastic::par::precompute::PrecomputedPar;

use crate::bucket_topology::TransitBucketTopology;
use crate::hydro_models::{
    EvaporationModel, EvaporationModelSet, ProductionModelSet, ResolvedProductionModel,
};
use crate::indexer::{
    AnticipatedLocal, BlockGrid, BlockIdx, BlockRowFamily, Boundary, BusSys, EntityPositions,
    EvapLocal, EvaporationIndices, FillingTargetLocal, FloorLocal, FphaCellLocal, FphaLocal,
    HydroCell, HydroCellIndex, HydroSys, LineSys, NcsSys, PumpingSys, RangeCursor, StateSpace,
    StorageBoundaryGrid, StudyDimensions, ThermalSys, anticipated_resolution_for,
    for_each_live_commitment_slot, is_anticipated_decision_active_for_delivery,
};
use crate::time_value::TimeValue;

use super::hydro_state::hydro_phase;
use super::{
    EVAP_COLS_PER_HYDRO, EVAP_F_MINUS_OFFSET, EVAP_F_PLUS_OFFSET, EVAP_FLOW_OFFSET,
    GenericConstraintRowEntry,
};
use crate::block_clock::BlockClock;
use crate::resolved_parameters::ResolvedParameters;

/// Pre-resolved bound, penalty, and factor tables shared across all stages.
pub(crate) struct ResolvedTables<'a> {
    /// Resolved per-stage entity bounds.
    pub(crate) bounds: &'a ResolvedBounds,
    /// Resolved per-stage penalties.
    pub(crate) penalties: &'a ResolvedPenalties,
    /// `(constraint_idx, stage_id)` → active bound entries.
    pub(crate) resolved_generic_bounds: &'a ResolvedGenericConstraintBounds,
    /// Per-block load scaling factors.
    pub(crate) resolved_load_factors: &'a ResolvedLoadFactors,
    /// Per-stage NCS available generation bounds.
    pub(crate) resolved_ncs_bounds: &'a ResolvedNcsBounds,
    /// Per-block NCS generation scaling factors.
    pub(crate) resolved_ncs_factors: &'a ResolvedNcsFactors,
    /// `(parameter_id, stage_idx, block_idx)` → resolved `f64`, queried for a
    /// [`cobre_core::CoefficientRef::Parameter`] term.
    pub(crate) resolved_parameters: &'a ResolvedParameters,
}

/// System-level context shared across all stages during template construction.
pub(crate) struct TemplateBuildCtx<'a> {
    pub(crate) hydros: &'a [Hydro],
    pub(crate) thermals: &'a [Thermal],
    pub(crate) lines: &'a [Line],
    pub(crate) buses: &'a [Bus],
    pub(crate) load_models: &'a [LoadModel],
    pub(crate) cascade: &'a CascadeTopology,
    /// Study-scope partition of each hydro plant's unit groups into `bus_id`
    /// cells (built once — never cloned or rebuilt per stage).
    pub(crate) hydro_cell_index: &'a HydroCellIndex,
    /// Pre-resolved bound, penalty, and factor tables.
    pub(crate) resolved: ResolvedTables<'a>,
    /// Canonical entity-id → slot maps for every position-addressed family.
    /// Declaration-order bit-determinism (`csc_byte_identical_under_permuted_multi_entity_order`)
    /// depends on every fill iterating a slice, never this map.
    pub(crate) positions: &'a EntityPositions,
    pub(crate) par_lp: &'a PrecomputedPar,
    /// Resolved production models for all (hydro, stage) pairs.
    pub(crate) production_models: &'a ProductionModelSet,
    /// Resolved evaporation models for all hydro plants.
    pub(crate) evaporation_models: &'a EvaporationModelSet,
    /// Generic constraint definitions (expression, slack config).
    pub(crate) generic_constraints: &'a [GenericConstraint],
    /// Non-controllable source entities, id-sorted.
    pub(crate) non_controllable_sources: &'a [NonControllableSource],
    /// Pumping station entities, id-sorted (canonical slot order).
    pub(crate) pumping_stations: &'a [PumpingStation],
    /// Energy contract entities, id-sorted (canonical slot order). One slice for
    /// both directions; the import/export split is derived at fill time from
    /// `contract_type`, not pre-partitioned.
    pub(crate) contracts: &'a [EnergyContract],
    /// Target hydro ID → system indices of hydros diverting to it (each hydro `d`
    /// with `diversion.downstream_id == target_id`). Borrowed from setup's
    /// single `resolve_lp_build_inputs` resolution
    /// (`LpBuildInputs::diversion_upstream`).
    pub(crate) diversion_upstream: &'a HashMap<EntityId, Vec<usize>>,
    /// The role-(a) state layout, from setup's single owner
    /// (`resolve_state_layout`), which owns `anticipated_lead_stages` and
    /// `anticipated_resolution`.
    pub(crate) state: &'a StateSpace,
    /// Study-invariant, non-state LP shape (`inflow_method`,
    /// `max_deficit_segments`, `anticipated_plants`), threaded from setup's
    /// single owner (`build_study_dimensions`).
    pub(crate) study_dims: &'a StudyDimensions,
    /// Present-value discounting and delivery hours/ids at each DELIVERY
    /// stage, length `n_study_stages + n_post` — the study's own per-stage
    /// values concatenated with the post-study continuation, the first
    /// cumulative-discount entry exactly `1.0`. The strict predicate
    /// `stage_idx + K_i < n_stages` keeps every delivery lookup in range.
    /// Borrowed from setup's single `StageData` owner.
    pub(crate) time_value: &'a TimeValue,
    /// Per-stage minimum target-storage trajectory, keyed `(hydro_idx, stage_id)
    /// → V_target` \[hm³\]. Computed once by a backward fold from the dead volume
    /// because the fold needs the full per-stage ζ·rate schedule across a hydro's
    /// Filling stages; the forbidden alternative — recomputing inside the per-stage
    /// `fill_filling_target_rows` (which sees one stage) — is wrong or re-walks the
    /// schedule on the hot path. `BTreeMap` for determinism (canonical iteration
    /// order, not `HashMap`'s). Empty for a non-filling build (parity-neutral).
    /// Borrowed from setup's single `resolve_lp_build_inputs` resolution
    /// (`LpBuildInputs::filling_v_target`, computed by setup's
    /// `build_filling_v_target`).
    pub(crate) filling_v_target: &'a BTreeMap<(usize, i32), f64>,
    /// The resolved bucket topology (canonical column order, per-stage
    /// reachability mask, and the three resolved arc tables — stage-clock
    /// weights, chronological spread, arrival density), threaded from
    /// setup's single owner (`crate::bucket_topology::build_transit_bucket_topology`).
    /// This ctx used to carry four of its tables as its own clones; see
    /// [`crate::bucket_topology::TransitBucketTopology`].
    pub(crate) topology: &'a TransitBucketTopology,
}

/// Column/row offsets for one stage's in-study anticipated-ring layout
/// (latch/carry/fish, modular-slot-addressed), carved from
/// [`StateSpace::commit_out`]/[`StateSpace::commit_in`]. There is no separate
/// block-layout struct — [`StageLayout::new`] allocates both columns and rows
/// in one pass.
pub(crate) struct AnticipatedLayout {
    /// The `anticipated_state_out_def` equality row block: one row per plant
    /// with a genuine, ACTIVE decision this stage
    /// (`PointResolution::genuine_decisions_at(stage_idx).next()`, AND the
    /// delivery stage's commissioning window), pinning that decision's ring
    /// slot (`ring_index(delivery_stage) mod k_max`) to its decision column.
    /// Immediately after [`Self::fishing_rows`].
    pub(crate) state_out_def_rows: Range<usize>,
    /// For each plant (local order), this stage's compact row position
    /// within the deposit-row family, or `None` when the plant has no
    /// genuine decision this stage (`PointResolution::genuine_decisions_at`)
    /// or the delivery is commissioning-inactive. Length `n_anticipated`.
    pub(crate) anticipated_decision_row_pos: Vec<Option<usize>>,
    /// The commitment-MATURITY rows: one per anticipated plant whose
    /// delivery matures THIS stage (`PointResolution::is_anticipated_at`,
    /// `false` at a `K = 0` self-delivery). Every such plant gets exactly one
    /// row here regardless of commissioning activeness — maturity always
    /// fishes, via [`super::entries::fill_anticipated_fishing_entries`].
    /// After operational-violation rows.
    pub(crate) fishing_rows: Range<usize>,
    /// For each anticipated plant (local order), this stage's compact row
    /// position within the maturity-row family, or `None` when no delivery
    /// matures this stage (including a `K = 0` self-delivery, which never
    /// matures through the ring at all). Length `n_anticipated`.
    pub(crate) anticipated_fishing_row_pos: Vec<Option<usize>>,
    /// The future-window commitment-carry equality rows (same-slot hold,
    /// `slot^out − slot^in = 0`, routed by
    /// `fill_anticipated_slot_definition_entries` via
    /// [`super::delivery_ring::DeliveryRing::emit_carry_rows`]): every
    /// STRICTLY FUTURE, not-yet-due in-study slot, modular-addressed
    /// (`ring_index(delivery_target) mod k_max`). The commitment maturing THIS
    /// stage is never here — it always fishes through the maturity row above;
    /// carry-to-terminal belongs to the post-study-targeted slot alone, so this
    /// family and [`Self::fishing_rows`] never double-book the same delivery.
    /// Immediately after [`Self::state_out_def_rows`].
    pub(crate) slot_definition_rows: Range<usize>,
    /// For each GLOBAL in-study commitment-hold slot (`(ring_index(m) mod k_max) *
    /// n_anticipated + plant`, modular slot-major/plant-minor —
    /// [`StateSpace::commitment_hold_in_study_offset`]'s own addressing), this
    /// stage's compact row position within the future-window carry-row
    /// family, or `None` when the slot's target is this stage's own latch
    /// ([`Self::state_out_def_rows`] owns it), matures THIS stage (always
    /// fished instead, [`Self::fishing_rows`] owns it), is beyond the study
    /// horizon, or is not yet ready (`PointResolution::is_ready_at`). Length
    /// `n_anticipated * k_max`.
    pub(crate) anticipated_slot_row_pos: Vec<Option<usize>>,
}

impl AnticipatedLayout {
    /// Allocate the commitment-maturity rows, then the deposit-row family,
    /// then the future-window carry rows, in that order: reordering these
    /// three `row.alloc` calls would shift every family after them.
    fn new(row: &mut RangeCursor, ctx: &TemplateBuildCtx<'_>, stage_idx: usize) -> Self {
        let state = ctx.state;
        // A `K = 0` self-delivery excludes a plant's row this stage, so the
        // maturity-row family is sparse like the deposit-row family below, not
        // the dense `state.n_anticipated` count.
        let n_stages = ctx.resolved.bounds.n_stages();
        let (anticipated_fishing_row_pos, n_fishing_rows) =
            build_anticipated_fishing_row_pos(state, n_stages, stage_idx);
        let fishing_rows = row.alloc(n_fishing_rows);

        let (anticipated_decision_row_pos, n_state_out_def_rows) =
            build_anticipated_decision_row_pos(
                state,
                stage_idx,
                ctx.study_dims.anticipated_plants.windows(),
                ctx.time_value.delivery_stage_ids(),
            );
        let state_out_def_rows = row.alloc(n_state_out_def_rows);

        let (anticipated_slot_row_pos, n_slot_definition_rows) =
            build_anticipated_slot_row_pos(state, stage_idx);
        let slot_definition_rows = row.alloc(n_slot_definition_rows);

        Self {
            state_out_def_rows,
            anticipated_decision_row_pos,
            fishing_rows,
            anticipated_fishing_row_pos,
            slot_definition_rows,
            anticipated_slot_row_pos,
        }
    }
}

/// Equipment column facts [`StageGeometry`] does not itself address.
pub(crate) struct EquipmentColumns {
    /// Maximum deficit segments across buses (`S`); the deficit-stride constant.
    pub(crate) max_deficit_segments: usize,
    /// Column-block cursor at which the evaporation block begins, even when empty
    /// (`generation.end`).
    pub(crate) evap_col_start: usize,
}

/// Row ranges for the four operational-violation slack families
/// (below-min-outflow, above-max-outflow, below-min-turbine,
/// below-min-generation). The two flow families are sized `n_h * n_blks`
/// (non-empty only when `n_h > 0`); the two power families are sized
/// `n_cells * n_blks` (non-empty only when `n_cells > 0`) — a cell's own
/// min-turbine/min-generation floor is the sum of ITS OWN member groups, never
/// the plant's aggregate, so each cell gets its own row and its own slack
/// column. See the min-floor contract. Their paired slack columns live on
/// [`StageGeometry`]; see [`allocate_oper_violation_slack_columns`] for why
/// the two halves are allocated separately.
pub(crate) struct OperViolationRanges {
    /// Row range for min-outflow constraints (one per hydro per block).
    pub(crate) min_outflow: Range<usize>,
    /// Row range for max-outflow constraints (one per hydro per block).
    pub(crate) max_outflow: Range<usize>,
    /// Row range for min-turbine constraints (one per hydro CELL per block).
    pub(crate) min_turbine: Range<usize>,
    /// Row range for min-generation constraints (one per hydro CELL per block).
    pub(crate) min_generation: Range<usize>,
}

impl OperViolationRanges {
    /// Allocate the four row families, contiguously in this order: reordering
    /// these `alloc` calls would shift every downstream row. `n_op_hydro`
    /// sizes the two flow families; `n_op_cell` sizes the two power
    /// families — they diverge the moment any plant declares groups on more
    /// than one bus. The caller allocates the paired slack columns
    /// (`allocate_oper_violation_slack_columns`) immediately before this, in
    /// the same order.
    fn new(row: &mut RangeCursor, n_op_hydro: usize, n_op_cell: usize) -> Self {
        Self {
            min_outflow: row.alloc(n_op_hydro),
            max_outflow: row.alloc(n_op_hydro),
            min_turbine: row.alloc(n_op_cell),
            min_generation: row.alloc(n_op_cell),
        }
    }
}

/// Allocate the four operational-violation slack columns (below-min-outflow,
/// above-max-outflow, below-min-turbine, below-min-generation), in the order
/// their paired rows follow in [`OperViolationRanges::new`].
fn allocate_oper_violation_slack_columns(
    col: &mut RangeCursor,
    n_op_hydro: usize,
    n_op_cell: usize,
) -> (Range<usize>, Range<usize>, Range<usize>, Range<usize>) {
    (
        col.alloc(n_op_hydro),
        col.alloc(n_op_hydro),
        col.alloc(n_op_cell),
        col.alloc(n_op_cell),
    )
}

/// Constraint row ranges shared by every stage's LP that [`StageGeometry`]
/// does not itself address: travel-time buckets, the generic-constraint
/// cursor, and the structural row-count scalars.
pub(crate) struct ConstraintRows {
    /// Row range for travel-time bucket definition rows: `b_d^out − b_{d+1}^in
    /// − deposit_d = 0`, one row per (plant, lag) bucket REACHABLE at this
    /// stage (`state.transit_bucket_column_order[slot]`'s lag within this stage's
    /// `per_stage_mask` cap for that plant — see [`Self::transit_bucket_row_pos`]);
    /// unlike `commit_in`'s active-plant sparseness, a lag beyond the cap gets
    /// no row at this stage — absent a boundary FCF the cap only shrinks toward
    /// the horizon end (Terminal credit deferred); with one present the
    /// terminal cap un-caps instead (Delivery-family right-boundary pricing).
    /// Placed immediately after the water-balance rows ([`StageGeometry::water_balance`]),
    /// so the load-balance rows and every row cursor after it shift by this
    /// stage's reachable count (`<= state.n_buckets`). Empty `start..start`
    /// when `state.n_buckets == 0` (the B==0 byte-identity anchor: the
    /// load-balance rows collapse back onto the water-balance end).
    pub(crate) transit_bucket_definition: Range<usize>,
    /// For each GLOBAL bucket index (`state.transit_bucket_column_order`'s index),
    /// this stage's compact row position within [`Self::transit_bucket_definition`], or
    /// `None` when its lag is beyond this stage's reachable cap (no row; the
    /// matching deposit in [`super::entries`]'s arc-release fill is dropped
    /// there, not misdirected to another row). Length `state.n_buckets`.
    pub(crate) transit_bucket_row_pos: Vec<Option<usize>>,
    /// Start of generic constraint rows (one per active `(constraint, block)` pair),
    /// after operational-violation rows.
    pub(crate) row_generic_start: usize,
    /// Total row count.
    pub(crate) num_rows: usize,
    /// Generic constraint row count.
    pub(crate) n_generic_rows: usize,
}

impl ConstraintRows {
    /// Pure assembly: every row family is already allocated by the caller.
    /// `generic_rows` is still last in the row chain, so its `.start`/`.len()`
    /// are `row_generic_start`/`n_generic_rows`.
    fn new(
        transit: (Vec<Option<usize>>, Range<usize>),
        generic_rows: Range<usize>,
        num_rows: usize,
    ) -> Self {
        let (transit_bucket_row_pos, transit_bucket_definition) = transit;
        Self {
            transit_bucket_definition,
            transit_bucket_row_pos,
            row_generic_start: generic_rows.start,
            num_rows,
            n_generic_rows: generic_rows.len(),
        }
    }
}

/// Pre-computed column and row layout offsets for a single stage LP.
///
/// Allocates the role-(b) geometry directly into its [`Self::geometry`] field,
/// built in [`StageLayout::new`] anchored at the handle's
/// [`StateSpace::control_region_start`]. Every other field holds a
/// construction-only fact `StageGeometry` does not itself address — never a
/// second copy of one of its ranges or counts. The stage-invariant role-(a)
/// state region is NOT duplicated here either — it is read through the
/// borrowed [`Self::state`] handle. The control region begins at
/// `state.control_region_start()` (`theta + 1`), so the two regions meet
/// there with no overlap.
pub(crate) struct StageLayout<'a> {
    /// Borrowed handle to the stage-invariant role-(a) state layout; the role-(a)
    /// accessors read through it rather than re-deriving offsets per stage. The
    /// dependency is one-directional (geometry → `StateSpace`), never the reverse.
    pub(crate) state: &'a StateSpace,
    /// The single address map for this stage's equipment/slack/row ranges and
    /// their entity counts; the runtime and output decoding read this same
    /// type. Construction-only facts that duplicate it are not kept here.
    pub(crate) geometry: StageGeometry,
    /// In-study anticipated-ring column/row offsets (see [`AnticipatedLayout`]).
    pub(crate) anticipated: AnticipatedLayout,
    /// Equipment column facts [`StageGeometry`] does not address (see
    /// [`EquipmentColumns`]).
    pub(crate) equipment: EquipmentColumns,
    /// Operational-violation constraint rows (see [`OperViolationRanges`]).
    pub(crate) oper_violation: OperViolationRanges,
    /// Constraint row ranges [`StageGeometry`] does not address (see
    /// [`ConstraintRows`]).
    pub(crate) rows: ConstraintRows,
    /// Total column count.
    pub(crate) num_cols: usize,
    /// This stage's block-hours owner; the water-balance noise/inflow scale.
    pub(crate) clock: BlockClock<'a>,
    /// Inverse of `geometry.fpha_hydro_indices`: system hydro index → FPHA-local
    /// index, length `n_h` (`None` at non-FPHA hydros). Single owner of the
    /// reverse map, read by the matrix-fill helpers in place of rebuilding it
    /// per call.
    pub(crate) fpha_local_index: Vec<Option<FphaLocal>>,
    /// FPHA-local index → that plant's first cell's FPHA-cell-local index,
    /// length `n_fpha_hydros` (parallel to `geometry.fpha_hydro_indices`); the
    /// identity (`[0, 1, 2, ...]`) while every FPHA plant has one cell. Single
    /// owner of the FPHA-cell prefix sum, read by [`Self::fpha_local_first_cell`].
    pub(crate) fpha_cell_local_start: Vec<usize>,
    /// Evaporation slots per evaporating hydro at this stage: single-owner result
    /// of [`evaporation_slot_count`] (`1` on a parallel stage, `n_blks` on a
    /// chronological one) — the stride every evaporation column/row family uses.
    pub(crate) n_evap_slots: usize,
    /// Per-row metadata for active generic constraint rows, one per active
    /// `(constraint, block)` pair in constraint-index-major order.
    pub(crate) generic_constraint_rows: Vec<GenericConstraintRowEntry>,
}

// ── Private helper return structs ─────────────────────────────────────────────

/// Layout metadata for all active generic constraint rows and slack columns.
struct GenericConstraintLayout {
    n_generic_rows: usize,
    n_generic_slack_cols: usize,
    generic_constraint_rows: Vec<GenericConstraintRowEntry>,
}

/// For each entry of `column_order` (global bucket index `slot`, `(plant, lag)`),
/// this stage's compact position within [`StageLayout::transit_bucket_definition_row`]'s
/// row family, or
/// `None` when `lag` exceeds `per_stage_mask[stage_idx]`'s max reachable lag
/// for that plant. Plant groups come from [`StateSpace::transit_bucket_plants`],
/// in the SAME discovery order `per_stage_mask` indexes
/// ([`crate::bucket_topology::build_transit_bucket_topology`]). Returns the
/// mapping and the reachable count (`transit_bucket_definition`'s row length).
fn build_transit_bucket_row_pos(
    state: &StateSpace,
    per_stage_mask: &[Vec<usize>],
    stage_idx: usize,
) -> (Vec<Option<usize>>, usize) {
    if state.transit_bucket_column_order.is_empty() {
        // B==0 byte-identity anchor: no declared bucket, so no per-stage mask
        // entry is required (`per_stage_mask` may be empty in fixtures that
        // never build one).
        return (Vec::new(), 0);
    }
    let stage_mask = &per_stage_mask[stage_idx];
    let mut transit_bucket_row_pos = Vec::with_capacity(state.transit_bucket_column_order.len());
    let mut n_reachable = 0_usize;
    for (plant_group, (_, local)) in state.transit_bucket_plants().enumerate() {
        for &(_, lag) in &state.transit_bucket_column_order[local] {
            if lag <= stage_mask[plant_group] {
                transit_bucket_row_pos.push(Some(n_reachable));
                n_reachable += 1;
            } else {
                transit_bucket_row_pos.push(None);
            }
        }
    }
    (transit_bucket_row_pos, n_reachable)
}

/// For each GLOBAL in-study commitment-hold slot (`(r mod k_max) *
/// n_anticipated + plant`, modular slot-major/plant-minor — mirroring
/// [`build_transit_bucket_row_pos`]'s role for buckets), this stage's compact
/// row position within the future-window carry-row family, or `None` when the
/// slot's physical delivery target `m` is a genuine fresh decision this stage
/// (`decider[m] == Some(stage_idx)`, the deposit-row family
/// [`AnticipatedLayout::state_out_def_rows`] owns it instead), beyond the
/// EXTENDED delivery calendar (`m >= state.n_delivery()`),
/// or not yet ready ([`for_each_live_commitment_slot`]'s own filter,
/// structural padding). Masking on the study horizon (`m >= n_stages`)
/// instead is the wrong-but-compiling alternative: it would freeze `[0, 0]`
/// a slot the terminal boundary must carry, zeroing a commitment the FCF
/// prices. This covers only STRICTLY FUTURE, not-yet-due deliveries; the
/// commitment maturing EXACTLY this stage (`m == stage_idx`) always fishes,
/// owned by [`build_anticipated_fishing_row_pos`] — never duplicated here.
///
/// The strictly-future ring-window sweep, its readiness filter, and its
/// per-plant physical-target resolution are owned by
/// [`for_each_live_commitment_slot`]; this builder only classifies each
/// visited (already-live) residue as carry or deposit. Returns the mapping
/// and the reachable count.
fn build_anticipated_slot_row_pos(
    state: &StateSpace,
    stage_idx: usize,
) -> (Vec<Option<usize>>, usize) {
    let n_anticipated = state.n_anticipated;
    let mut row_pos = vec![None; n_anticipated * state.k_max];
    let mut n_reachable = 0_usize;
    for_each_live_commitment_slot(state, stage_idx, |res, point| {
        let is_deposit = point.decider.get(res.target).copied().flatten() == Some(stage_idx);
        if !is_deposit {
            row_pos[res.slot * n_anticipated + res.plant] = Some(n_reachable);
            n_reachable += 1;
        }
    });
    debug_assert_eq!(
        row_pos.iter().filter(|pos| pos.is_some()).count(),
        n_reachable,
        "n_reachable must equal the count of Some positions in row_pos"
    );
    (row_pos, n_reachable)
}

/// For each plant (local order), this stage's compact row position within
/// the deposit-row family, or `None` when the plant has no genuine decision
/// this stage (`PointResolution::genuine_decisions_at(stage_idx).next()`) or
/// the delivery is commissioning-inactive
/// (`is_anticipated_decision_active_for_delivery`). Empty (`(Vec::new(), 0)`)
/// when `n_anticipated == 0 || k_max == 0`, mirroring
/// [`build_anticipated_slot_row_pos`] — a genuine decision implies a carried
/// in-flight delivery, so an empty ring can hold none. Returns the
/// mapping and the active count.
fn build_anticipated_decision_row_pos(
    state: &StateSpace,
    stage_idx: usize,
    anticipated_windows: &[(Option<i32>, Option<i32>)],
    delivery_stage_ids: &[i32],
) -> (Vec<Option<usize>>, usize) {
    let n_anticipated = state.n_anticipated;
    let k_max = state.k_max;
    if n_anticipated == 0 || k_max == 0 {
        return (Vec::new(), 0);
    }
    let n_delivery = state.n_delivery();
    let mut row_pos = vec![None; n_anticipated];
    let mut n_active = 0_usize;
    for (plant, pos) in row_pos.iter_mut().enumerate() {
        let plant = AnticipatedLocal::new(plant);
        let point = anticipated_resolution_for(state, plant);
        let Some(m) = point.genuine_decisions_at(stage_idx).next() else {
            continue;
        };
        debug_assert_ne!(
            m, stage_idx,
            "a K=0 self-delivery (decider[m] == m) must never reach the anticipated \
             ring's deposit-row fill"
        );
        if is_anticipated_decision_active_for_delivery(
            plant,
            m,
            n_delivery,
            anticipated_windows,
            delivery_stage_ids,
        ) {
            *pos = Some(n_active);
            n_active += 1;
        }
    }
    (row_pos, n_active)
}

/// For each anticipated plant (local order), this stage's compact row
/// position within the commitment-MATURITY family, or `None` when the
/// delivery maturing this stage is a `K = 0` self-delivery
/// (`PointResolution::is_anticipated_at`, exclude-with-advisory) — no
/// anticipation binds, so the plant's ordinary thermal generation is
/// unconstrained by any fishing coupling. A `Some` position means this plant
/// fishes this stage: every plant with a delivery maturing this stage gets
/// exactly one row regardless of commissioning activeness, and
/// [`super::entries::fill_anticipated_fishing_entries`] ALWAYS renders the
/// must-generate fish coupling for it (reading `commit_in`, never writing
/// `commit_out`) — a commissioning-inactive delivery's dormant `commit_in`
/// simply carries 0, pinning that stage's generation to 0. Empty
/// (`(Vec::new(), 0)`) when `n_anticipated == 0 || k_max == 0`, mirroring
/// [`build_anticipated_slot_row_pos`]: `is_anticipated_at` is `true` for a
/// pre-study (`None`) decider, so gating on `n_anticipated` alone would let a
/// pre-study-only plant reach a fishing row on an empty ring.
///
/// STUDY-only domain, deliberately NOT generalized to the extended calendar:
/// `stage_idx` must itself be in-study (`stage_idx < n_stages`, checked
/// explicitly here rather than trusted from the caller), because a
/// post-study-targeted slot has no stage LP to couple generation into — it
/// carries to the terminal instead ([`build_anticipated_slot_row_pos`]),
/// never fishes. Returns the mapping and the active count.
fn build_anticipated_fishing_row_pos(
    state: &StateSpace,
    n_stages: usize,
    stage_idx: usize,
) -> (Vec<Option<usize>>, usize) {
    let n_anticipated = state.n_anticipated;
    let k_max = state.k_max;
    if n_anticipated == 0 || k_max == 0 || stage_idx >= n_stages {
        return (Vec::new(), 0);
    }
    let mut row_pos = vec![None; n_anticipated];
    let mut n_active = 0_usize;
    for (plant, pos) in row_pos.iter_mut().enumerate() {
        if anticipated_resolution_for(state, AnticipatedLocal::new(plant))
            .is_anticipated_at(stage_idx)
        {
            *pos = Some(n_active);
            n_active += 1;
        }
    }
    (row_pos, n_active)
}

/// Evaporation slots per evaporating hydro: one stage-level slot on a parallel
/// stage (its blocks share the stage endpoints), one per block on a
/// chronological stage.
pub(crate) fn evaporation_slot_count(block_mode: BlockMode, n_blks: usize) -> usize {
    match block_mode {
        BlockMode::Parallel => 1,
        BlockMode::Chronological => n_blks,
    }
}

/// The slot block `blk` reads, given the stage's slot count.
pub(crate) fn evaporation_slot(slot_count: usize, blk: BlockIdx) -> BlockIdx {
    if slot_count == 1 {
        BlockIdx::new(0)
    } else {
        blk
    }
}

/// Evaporation column/row indices per `(evaporation hydro, slot)`, slot-major
/// (`local * n_evap_slots + slot`) to mirror the block-strided generation
/// columns. Within-triple columns at [`EVAP_FLOW_OFFSET`] / [`EVAP_F_PLUS_OFFSET`] /
/// [`EVAP_F_MINUS_OFFSET`], strided by [`EVAP_COLS_PER_HYDRO`]; one row per
/// `(hydro, slot)`.
fn build_evap_indices(
    n_evap_hydros: usize,
    n_evap_slots: usize,
    col_start: usize,
    row_start: usize,
) -> Vec<EvaporationIndices> {
    let mut out = Vec::with_capacity(n_evap_hydros * n_evap_slots);
    for i in 0..n_evap_hydros {
        for slot in 0..n_evap_slots {
            let flat = evap_slot_flat(i, slot, n_evap_slots);
            let triple_base = col_start + flat * EVAP_COLS_PER_HYDRO;
            out.push(EvaporationIndices {
                evaporation_flow_col: triple_base + EVAP_FLOW_OFFSET,
                f_evap_plus_col: triple_base + EVAP_F_PLUS_OFFSET,
                f_evap_minus_col: triple_base + EVAP_F_MINUS_OFFSET,
                evap_row: row_start + flat,
            });
        }
    }
    out
}

// ── Private helper functions ───────────────────────────────────────────────────

/// Collect the FPHA hydro indices and per-hydro plane counts for this stage.
///
/// A filling hydro is dropped from the FPHA set in `PreFilling` **or** `Filling`:
/// a non-operating plant has zero productivity, and the operating-range hyperplane
/// fit is invalid below `min_storage` where a filling reservoir sits. Because the
/// generation column block is densely packed by FPHA-local index, dropping a hydro
/// here removes its column entirely — no orphaned `[0, max]` column for an
/// unconstrained solve to exploit. `stage_id` is the study `stage.id`, not the
/// stage index ([`filling_phase`](cobre_core::commissioning::filling_phase) keys
/// on the commissioning id). A
/// commissioning-dormant non-filling hydro is `PreFilling` and is dropped here too;
/// a non-filling hydro with no window is `Operating` at every stage (parity-neutral).
fn identify_fpha_hydros(
    ctx: &TemplateBuildCtx<'_>,
    stage_idx: usize,
    stage_id: i32,
) -> Vec<HydroSys> {
    let mut fpha_hydro_indices: Vec<HydroSys> = Vec::new();
    for h_idx in 0..ctx.hydros.len() {
        let hydro = &ctx.hydros[h_idx];
        if matches!(
            hydro_phase(hydro, stage_id),
            Phase::PreFilling | Phase::Filling
        ) {
            continue;
        }
        if matches!(
            ctx.production_models.model(h_idx, stage_idx),
            ResolvedProductionModel::Fpha { .. }
        ) {
            fpha_hydro_indices.push(HydroSys::new(h_idx));
        }
    }
    fpha_hydro_indices
}

/// Collect the indices of hydros with linearized evaporation at this stage.
///
/// A hydro is dropped from the evaporation set only in `PreFilling` (before
/// `start_stage_id`, or while a non-filling hydro is commissioning-dormant, the dam
/// and hence the reservoir surface does not exist). Evaporation is **kept** during
/// `Filling` — the opposite of the FPHA rule (excluded in `PreFilling` *and*
/// `Filling`); the two must not be unified. A non-filling hydro with no window is
/// `Operating` at every stage (parity-neutral).
fn identify_evap_hydros(ctx: &TemplateBuildCtx<'_>, stage_id: i32) -> Vec<HydroSys> {
    (0..ctx.hydros.len())
        .filter(|&h_idx| {
            let hydro = &ctx.hydros[h_idx];
            if matches!(hydro_phase(hydro, stage_id), Phase::PreFilling) {
                return false;
            }
            matches!(
                ctx.evaporation_models.model(h_idx),
                EvaporationModel::Linearized { .. }
            )
        })
        .map(HydroSys::new)
        .collect()
}

/// Collect the indices of hydros emitting a per-stage `σ_fill` target at this
/// stage: the filling hydros (`filling.is_some()`) in [`Phase::Filling`].
///
/// EVERY Filling stage carries a floor, NOT only the terminal stage at `entry −
/// 1`: the per-stage trajectory `V_target[t]` requires one soft floor `v_out[t] +
/// σ_fill[t] ≥ V_target[t]` at each. The wrong-but-compiling alternative —
/// restricting membership to `entry − 1 == stage_id` (the v1 terminal-only rule) —
/// drops every intermediate floor. `PreFilling`/`Operating` are excluded by
/// [`filling_phase`](cobre_core::commissioning::filling_phase) (`filled_min_storage_floor`
/// takes over at/after `entry`). A
/// non-filling hydro is `Operating` at every stage (parity-neutral).
fn identify_filling_target_hydros(ctx: &TemplateBuildCtx<'_>, stage_id: i32) -> Vec<HydroSys> {
    (0..ctx.hydros.len())
        .filter(|&h_idx| {
            let hydro = &ctx.hydros[h_idx];
            hydro.filling.is_some() && matches!(hydro_phase(hydro, stage_id), Phase::Filling)
        })
        .map(HydroSys::new)
        .collect()
}

/// Collect the indices of hydros emitting a soft `σ^{v-}` operating-floor at this
/// stage: the filling hydros (`filling.is_some()`) in [`Phase::Operating`].
///
/// DISTINCT from [`identify_filling_target_hydros`] (`σ_fill`): `σ^{v-}` fires at
/// EVERY Operating stage, `σ_fill` at EVERY Filling stage; the two never overlap
/// and carry different costs.
///
/// The soft floor is scoped to filling hydros DELIBERATELY — a non-filling
/// `Operating` hydro keeps its hard `min_storage` floor (same gate as the relax in
/// `columns::fill_storage_columns`). The wrong-but-compiling alternative —
/// a GLOBAL soft floor matching every Operating hydro regardless of `filling` —
/// would let the optimizer cheaply violate dead volume system-wide. Empty for a
/// non-filling build (parity-neutral).
fn identify_filled_min_storage_floor_hydros(
    ctx: &TemplateBuildCtx<'_>,
    stage_id: i32,
) -> Vec<HydroSys> {
    (0..ctx.hydros.len())
        .filter(|&h_idx| {
            let hydro = &ctx.hydros[h_idx];
            matches!(hydro_phase(hydro, stage_id), Phase::Operating) && hydro.filling.is_some()
        })
        .map(HydroSys::new)
        .collect()
}

/// Per-direction contract counts, in `contracts`' own (id-sorted) slice
/// order — the dense per-stage import/export column strides.
pub(super) fn contract_direction_counts(contracts: &[EnergyContract]) -> (usize, usize) {
    let n_import = contracts
        .iter()
        .filter(|c| c.contract_type == ContractType::Import)
        .count();
    let n_export = contracts
        .iter()
        .filter(|c| c.contract_type == ContractType::Export)
        .count();
    (n_import, n_export)
}

/// Allocate the slack column index/indices for one generic-constraint row,
/// advancing `n_slack_cols`: zero columns when slack is disabled, one for a
/// one-sided row, two (plus then minus) for a two-sided row — a two-sided
/// bound pair needs both directions of slack to relax either endpoint
/// independently.
///
/// The two-sided test derives from the row's OWN endpoint pair
/// (`bound_lower.is_some() && bound_upper.is_some()`), not the constraint —
/// shape is a per-row property of the resolved bound entry, never a
/// constraint-level label.
fn allocate_generic_slack_cols(
    slack: &SlackConfig,
    bound_lower: Option<f64>,
    bound_upper: Option<f64>,
    col_generic_slack_start: usize,
    n_slack_cols: &mut usize,
) -> (Option<usize>, Option<usize>) {
    if !slack.enabled {
        return (None, None);
    }
    let plus_col = col_generic_slack_start + *n_slack_cols;
    *n_slack_cols += 1;
    let minus_col = if bound_lower.is_some() && bound_upper.is_some() {
        let mc = col_generic_slack_start + *n_slack_cols;
        *n_slack_cols += 1;
        Some(mc)
    } else {
        None
    };
    (Some(plus_col), minus_col)
}

/// Whether a single [`VariableRef`] resolves to the *same* LP column(s) regardless
/// of `block_idx` — **block-independent** ("stock"). Seven kinds qualify:
/// [`VariableRef::HydroStorage`] (stage-final alias `Sᴷ`),
/// [`VariableRef::AnticipatedDecision`], [`VariableRef::HydroEvaporation`] (a fixed
/// single-block column or the all-block sum — both `block_idx`-independent), and the
/// four storage/useful-volume-boundary variants [`VariableRef::HydroStorageInitial`] /
/// [`VariableRef::HydroStorageFinal`] / [`VariableRef::HydroUsefulVolumeInitial`] /
/// [`VariableRef::HydroUsefulVolumeFinal`], each resolving to a fixed boundary column
/// (`Sᵏ` / `S⁰` / `Sᴷ`) that does not follow the materialized row's block.
///
/// [`VariableRef::HydroInflow`] is block-DEPENDENT: its upstream-release terms are
/// per-block columns. Classifying it "stock" would collapse a multi-block expression
/// to one mis-priced stage-level row reading upstream columns at a single arbitrary
/// block, silently dropping the other blocks. [`VariableRef::PumpingFlow`] /
/// [`VariableRef::PumpingPower`] are block-level for the same reason. The stub kinds
/// (withdrawal, contracts, non-controllable) resolve to no columns and are
/// conservatively block-level, so only *provably* stock variables enable the
/// single-row collapse.
///
/// The match is exhaustive (no wildcard): a future variant forces a compile error
/// here, defaulting nothing to "stock" by omission.
#[must_use]
pub(super) fn variable_ref_is_block_independent(var_ref: &VariableRef) -> bool {
    match var_ref {
        VariableRef::HydroStorage { .. }
        | VariableRef::HydroStorageInitial { .. }
        | VariableRef::HydroStorageFinal { .. }
        | VariableRef::HydroUsefulVolumeInitial { .. }
        | VariableRef::HydroUsefulVolumeFinal { .. }
        | VariableRef::HydroEvaporation { .. }
        | VariableRef::AnticipatedDecision { .. } => true,
        VariableRef::HydroInflow { .. }
        | VariableRef::HydroTurbined { .. }
        | VariableRef::HydroSpillage { .. }
        | VariableRef::HydroDiversion { .. }
        | VariableRef::HydroOutflow { .. }
        | VariableRef::HydroGeneration { .. }
        | VariableRef::ThermalGeneration { .. }
        | VariableRef::LineDirect { .. }
        | VariableRef::LineReverse { .. }
        | VariableRef::LineExchange { .. }
        | VariableRef::BusDeficit { .. }
        | VariableRef::BusExcess { .. }
        | VariableRef::HydroWithdrawal { .. }
        | VariableRef::PumpingFlow { .. }
        | VariableRef::PumpingPower { .. }
        | VariableRef::ContractImport { .. }
        | VariableRef::ContractExport { .. }
        | VariableRef::NonControllableGeneration { .. }
        | VariableRef::NonControllableCurtailment { .. } => false,
    }
}

/// Whether **every** term of a generic-constraint expression is block-independent
/// (see [`variable_ref_is_block_independent`]), letting a `block_id = None` bound
/// collapse its per-block replication into one stage-level row. Any block-level term
/// forces `false`. An empty expression is vacuously true.
#[must_use]
fn expression_is_block_independent(expression: &ConstraintExpression) -> bool {
    expression
        .terms
        .iter()
        .all(|term| variable_ref_is_block_independent(&term.variable))
}

/// Whether a `block_id = None` bound over `expression` collapses to a single
/// stage-level row: only when every term is block-independent in BOTH its variable
/// ([`expression_is_block_independent`]) AND its coefficient. A term whose
/// coefficient references a block-varying (`PerStageBlock`) parameter makes the
/// expression block-dependent, so the collapsed single row cannot stand in for one
/// arbitrary block's coefficient — it stays a per-block row set.
fn expression_collapses_to_stage_level(
    expression: &ConstraintExpression,
    resolved: &ResolvedParameters,
) -> bool {
    expression_is_block_independent(expression)
        && !expression.terms.iter().any(|term| match term.coefficient {
            CoefficientRef::Parameter(id) => resolved.is_block_varying(id),
            CoefficientRef::Literal(_) => false,
        })
}

/// Resolve an affine bound remainder to `f64`: `bound.constant` plus the sum of
/// each term's coefficient times its parameter's resolved value at
/// `(stage_idx, block_idx)`. `AffineBound::single(id)` resolves to exactly
/// `resolved.get(id, stage_idx, block_idx)` (`0.0 + 1.0 * x == x` in `f64`).
fn resolve_affine(
    bound: &AffineBound,
    resolved: &ResolvedParameters,
    stage_idx: usize,
    block_idx: usize,
) -> f64 {
    bound.terms.iter().fold(bound.constant, |acc, &(coef, id)| {
        acc + coef * resolved.get(id, stage_idx, block_idx)
    })
}

/// Fold a generic-constraint endpoint's parquet base with its affine remainder:
/// a present remainder SHIFTS the base by `resolve_affine`'s value rather than
/// replacing it, so `(Some(base), Some(bound))` folds to `base +
/// resolve_affine(bound, ...)`, never `resolve_affine(bound, ...)` alone. A
/// `(None, None)` endpoint is untargeted and stays `None` (the open LP
/// direction), never shifted.
fn fold_endpoint(
    parquet: Option<f64>,
    affine: Option<&AffineBound>,
    resolved: &ResolvedParameters,
    stage_idx: usize,
    block_idx: usize,
) -> Option<f64> {
    match (parquet, affine) {
        (None, None) => None,
        (Some(base), None) => Some(base),
        (None, Some(bound)) => Some(resolve_affine(bound, resolved, stage_idx, block_idx)),
        (Some(base), Some(bound)) => {
            Some(base + resolve_affine(bound, resolved, stage_idx, block_idx))
        }
    }
}

/// Sum of `resolved_coeff * V_lo` over `constraint`'s `HydroUsefulVolume{Initial,
/// Final}` terms, or `None` when none are present (the no-term path must leave the
/// folded endpoints untouched, never add `0.0`). A useful-volume term resolves to
/// the absolute storage column, so its dead volume shifts onto the bound instead of
/// the column; `V_lo` is the entity-level physical `Hydro.min_storage_hm3`, never
/// the per-stage resolved `HydroStageBounds.min_storage_hm3`.
fn useful_volume_bound_shift(
    constraint: &GenericConstraint,
    ctx: &TemplateBuildCtx<'_>,
    stage_idx: usize,
    block_idx: usize,
) -> Option<f64> {
    let resolved_parameters = ctx.resolved.resolved_parameters;
    let mut shift = 0.0;
    let mut found = false;
    for term in &constraint.expression.terms {
        let (VariableRef::HydroUsefulVolumeInitial { hydro_id, .. }
        | VariableRef::HydroUsefulVolumeFinal { hydro_id, .. }) = term.variable
        else {
            continue;
        };
        found = true;
        // A dangling hydro_id is unreachable past referential validation
        // (`validate_variable_ref_entity`); mirrors `ResolvedParameters::get`'s
        // test-loud, production-safe miss handling.
        let Some(h_idx) = ctx.positions.hydro(hydro_id) else {
            debug_assert!(
                false,
                "generic constraint {:?} useful-volume term references unknown hydro {hydro_id:?}",
                constraint.id
            );
            continue;
        };
        let coef = match term.coefficient {
            CoefficientRef::Literal(v) => v,
            CoefficientRef::Parameter(param_id) => {
                resolved_parameters.get(param_id, stage_idx, block_idx)
            }
        };
        let v_lo = ctx.hydros[h_idx].min_storage_hm3;
        shift += coef * term.scale * v_lo;
    }
    found.then_some(shift)
}

/// Whether either affine bound on `constraint` references a block-varying
/// (`PerStageBlock`) parameter. When true, the stage-level collapse is suppressed:
/// a single collapsed row would resolve one arbitrary block's bound value, losing
/// the per-block variation.
fn bound_affine_is_block_varying(
    constraint: &GenericConstraint,
    resolved: &ResolvedParameters,
) -> bool {
    [
        &constraint.bound_lower_affine,
        &constraint.bound_upper_affine,
    ]
    .into_iter()
    .flatten()
    .flat_map(AffineBound::params)
    .any(|id| resolved.is_block_varying(id))
}

/// Enumerate active generic constraint rows and assign their slack column indices.
///
/// One [`GenericConstraintRowEntry`] per active `(constraint, block)` pair, except
/// a `block_id = None` bound over a block-independent expression, which collapses
/// to a single stage-level row.
fn enumerate_generic_constraint_rows(
    ctx: &TemplateBuildCtx<'_>,
    stage: &Stage,
    stage_idx: usize,
    n_blks: usize,
    col_generic_slack_start: usize,
) -> GenericConstraintLayout {
    let mut n_generic_rows: usize = 0;
    let mut n_generic_slack_cols: usize = 0;
    let mut generic_constraint_rows: Vec<GenericConstraintRowEntry> = Vec::new();
    let resolved_parameters = ctx.resolved.resolved_parameters;

    for (constraint_idx, constraint) in ctx.generic_constraints.iter().enumerate() {
        if !ctx
            .resolved
            .resolved_generic_bounds
            .is_active(constraint_idx, stage.id)
        {
            continue;
        }

        let bound_entries = ctx
            .resolved
            .resolved_generic_bounds
            .bounds_for_stage(constraint_idx, stage.id);

        let collapse_stage_level =
            expression_collapses_to_stage_level(&constraint.expression, resolved_parameters)
                && !bound_affine_is_block_varying(constraint, resolved_parameters);

        for entry in bound_entries {
            #[expect(
                clippy::cast_sign_loss,
                reason = "block ids are validated non-negative before the layout is built"
            )]
            let (block_start, block_count, is_stage_level) = match entry.block_id {
                None if collapse_stage_level => (0, 1, true),
                None => (0, n_blks, false),
                Some(blk_id) => (blk_id as usize, 1, false),
            };
            for block_idx in block_start..block_start + block_count {
                // The folded pair drives both the row bound and, below, the
                // two-sided slack shape.
                let mut effective_lower = fold_endpoint(
                    entry.bound_lower,
                    constraint.bound_lower_affine.as_ref(),
                    resolved_parameters,
                    stage_idx,
                    block_idx,
                );
                let mut effective_upper = fold_endpoint(
                    entry.bound_upper,
                    constraint.bound_upper_affine.as_ref(),
                    resolved_parameters,
                    stage_idx,
                    block_idx,
                );
                if let Some(shift) =
                    useful_volume_bound_shift(constraint, ctx, stage_idx, block_idx)
                {
                    effective_lower = effective_lower.map(|v| v + shift);
                    effective_upper = effective_upper.map(|v| v + shift);
                }
                let (slack_plus_col, slack_minus_col) = allocate_generic_slack_cols(
                    &constraint.slack,
                    effective_lower,
                    effective_upper,
                    col_generic_slack_start,
                    &mut n_generic_slack_cols,
                );
                n_generic_rows += 1;
                generic_constraint_rows.push(GenericConstraintRowEntry {
                    constraint_idx,
                    entity_id: constraint.id.0,
                    block_idx,
                    is_stage_level,
                    bound_lower: effective_lower,
                    bound_upper: effective_upper,
                    slack_enabled: constraint.slack.enabled,
                    slack_penalty: constraint.slack.penalty.unwrap_or(0.0),
                    slack_plus_col,
                    slack_minus_col,
                });
            }
        }
    }

    GenericConstraintLayout {
        n_generic_rows,
        n_generic_slack_cols,
        generic_constraint_rows,
    }
}

/// One hydro's resolved LP production role at one stage — the single
/// classifier [`StageLayout::stage_production_role`] resolves, so
/// `fill_load_balance_entries` and `fill_operational_violation_entries` can
/// no longer disagree about a plant's role the way two independent
/// `ProductionModelSet::model()` re-queries once did.
#[derive(Debug, Clone, Copy)]
pub(super) enum StageProductionRole {
    /// Prices through the plant's FPHA generation column(s), at this
    /// FPHA-local index.
    Fpha(FphaLocal),
    /// Prices through the plant's turbine column at this shared productivity.
    Constant(f64),
    /// A commissioning-dormant (`PreFilling`/`Filling`) `Fpha`-resolved plant:
    /// gated out of `fpha_local_index` by `identify_fpha_hydros`, so it has no
    /// generation column, and its turbine column is frozen `[0, 0]`. It
    /// contributes nothing to either consumer's row — never priced as
    /// `ConstantProductivity`, which it has no productivity for.
    Dormant,
}

fn fpha_cell_offsets(
    ctx: &TemplateBuildCtx<'_>,
    fpha_hydro_indices: &[HydroSys],
    stage_idx: usize,
    n_h: usize,
) -> (Vec<Option<FphaLocal>>, Vec<usize>, usize, usize) {
    let mut fpha_local_index: Vec<Option<FphaLocal>> = vec![None; n_h];
    let mut fpha_cell_local_start: Vec<usize> = Vec::with_capacity(fpha_hydro_indices.len());
    let mut n_fpha_cells = 0_usize;
    let mut total_fpha_rows = 0_usize;
    for (local_idx, &h) in fpha_hydro_indices.iter().enumerate() {
        fpha_local_index[h.get()] = Some(FphaLocal::new(local_idx));
        fpha_cell_local_start.push(n_fpha_cells);
        let n_cells_h = ctx.hydro_cell_index.cells_of(h).len();
        n_fpha_cells += n_cells_h;
        let n_planes = match ctx.production_models.model(h.get(), stage_idx) {
            ResolvedProductionModel::Fpha { planes, .. } => planes.len(),
            ResolvedProductionModel::ConstantProductivity { .. } => {
                debug_assert!(
                    false,
                    "fpha_hydro_indices contains hydro {} but model is ConstantProductivity",
                    h.get()
                );
                0
            }
        };
        total_fpha_rows += n_cells_h * n_planes;
    }
    (
        fpha_local_index,
        fpha_cell_local_start,
        n_fpha_cells,
        total_fpha_rows,
    )
}

fn allocate_hydro_columns(
    col: &mut RangeCursor,
    n_h: usize,
    n_cells: usize,
    block_mode: BlockMode,
    n_blks: usize,
) -> (usize, Range<usize>, Range<usize>, Range<usize>) {
    let n_interior = match block_mode {
        BlockMode::Chronological => n_blks.saturating_sub(1),
        BlockMode::Parallel => 0,
    };
    let storage_internal_start = col.alloc(n_h * n_interior).start;
    let turbine = col.alloc(n_cells * n_blks);
    let spillage = col.alloc(n_h * n_blks);
    let diversion = col.alloc(n_h * n_blks);
    (storage_internal_start, turbine, spillage, diversion)
}

fn allocate_network_columns(
    col: &mut RangeCursor,
    n_lines: usize,
    n_buses: usize,
    max_deficit_segments: usize,
    n_blks: usize,
) -> (Range<usize>, Range<usize>, Range<usize>, Range<usize>) {
    let line_fwd = col.alloc(n_lines * n_blks);
    let line_rev = col.alloc(n_lines * n_blks);
    let deficit = col.alloc(n_buses * max_deficit_segments * n_blks);
    let excess = col.alloc(n_buses * n_blks);
    (line_fwd, line_rev, deficit, excess)
}

fn allocate_water_balance_rows(
    row: &mut RangeCursor,
    block_mode: BlockMode,
    n_h: usize,
    n_blks: usize,
) -> BlockRowFamily {
    match block_mode {
        BlockMode::Chronological => BlockRowFamily::per_block(row.alloc(n_h * n_blks)),
        BlockMode::Parallel => BlockRowFamily::one_per_entity(row.alloc(n_h)),
    }
}

fn allocate_transit_bucket_rows(
    row: &mut RangeCursor,
    ctx: &TemplateBuildCtx<'_>,
    stage_idx: usize,
) -> (Vec<Option<usize>>, Range<usize>) {
    let (transit_bucket_row_pos, n_transit_bucket_rows) =
        build_transit_bucket_row_pos(ctx.state, &ctx.topology.per_stage_mask, stage_idx);
    let transit_bucket_definition = row.alloc(n_transit_bucket_rows);
    (transit_bucket_row_pos, transit_bucket_definition)
}

fn allocate_fpha(
    col: &mut RangeCursor,
    row: &mut RangeCursor,
    n_fpha_cells: usize,
    total_fpha_rows: usize,
    n_blks: usize,
) -> (Range<usize>, Range<usize>) {
    let generation = col.alloc(n_fpha_cells * n_blks);
    let fpha_rows = row.alloc(n_blks * total_fpha_rows);
    (generation, fpha_rows)
}

fn allocate_evaporation(
    col: &mut RangeCursor,
    row: &mut RangeCursor,
    ctx: &TemplateBuildCtx<'_>,
    stage_id: i32,
    n_evap_slots: usize,
) -> (Vec<HydroSys>, Vec<EvaporationIndices>) {
    let evap_hydro_indices = identify_evap_hydros(ctx, stage_id);
    let n_evap_hydros = evap_hydro_indices.len();
    let cols = col.alloc(n_evap_hydros * n_evap_slots * EVAP_COLS_PER_HYDRO);
    let rows = row.alloc(n_evap_hydros * n_evap_slots);
    let evap_indices = build_evap_indices(n_evap_hydros, n_evap_slots, cols.start, rows.start);
    (evap_hydro_indices, evap_indices)
}

/// Read the slack-column start, enumerate the active generic-constraint rows
/// and their slack columns, then allocate the generic slack columns and the
/// generic rows — still last in each of their respective cursor chains.
fn allocate_generic_constraints(
    col: &mut RangeCursor,
    row: &mut RangeCursor,
    ctx: &TemplateBuildCtx<'_>,
    stage: &Stage,
    stage_idx: usize,
    n_blks: usize,
) -> (GenericConstraintLayout, Range<usize>) {
    let col_generic_slack_start = col.pos();
    let generic =
        enumerate_generic_constraint_rows(ctx, stage, stage_idx, n_blks, col_generic_slack_start);
    col.alloc(generic.n_generic_slack_cols);
    let generic_rows = row.alloc(generic.n_generic_rows);
    (generic, generic_rows)
}

impl StageGeometry {
    /// The start value [`StageLayout::new`] allocates into: every range
    /// empty, every `Vec` empty.
    fn unallocated(block_mode: BlockMode, n_blks: usize) -> Self {
        Self {
            turbine: 0..0,
            spillage: 0..0,
            diversion: 0..0,
            thermal: 0..0,
            anticipated_decision: 0..0,
            line_fwd: 0..0,
            line_rev: 0..0,
            deficit: 0..0,
            excess: 0..0,
            generation: 0..0,
            ncs_generation: 0..0,
            pumping_flow: 0..0,
            evap_indices: Vec::new(),
            inflow_slack: 0..0,
            withdrawal_slack_neg: 0..0,
            withdrawal_slack_pos: 0..0,
            outflow_below_slack: 0..0,
            outflow_above_slack: 0..0,
            turbine_below_slack: 0..0,
            generation_below_slack: 0..0,
            contract_import: 0..0,
            contract_export: 0..0,
            water_balance: BlockRowFamily::one_per_entity(0..0),
            load_balance: BlockRowFamily::one_per_entity(0..0),
            fpha: 0..0,
            filling_target: 0..0,
            filling_target_col: 0..0,
            filled_min_storage_floor: 0..0,
            filled_min_storage_floor_col: 0..0,
            n_blks,
            storage_internal_start: 0,
            block_mode,
            fpha_hydro_indices: Vec::new(),
            evap_hydro_indices: Vec::new(),
            filling_target_hydro_indices: Vec::new(),
            filled_min_storage_floor_hydro_indices: Vec::new(),
        }
    }
}

impl<'a> StageLayout<'a> {
    pub(crate) fn new(ctx: &TemplateBuildCtx<'a>, stage: &'a Stage, stage_idx: usize) -> Self {
        let state_layout = ctx.state;
        let clock = BlockClock::new(stage);
        let n_blks = clock.n_blks();
        let n_h = state_layout.hydro_count;
        let n_cells = ctx.hydro_cell_index.n_cells();

        let fpha_hydro_indices = identify_fpha_hydros(ctx, stage_idx, stage.id);
        let filling_target_hydro_indices = identify_filling_target_hydros(ctx, stage.id);
        let filled_min_storage_floor_hydro_indices =
            identify_filled_min_storage_floor_hydros(ctx, stage.id);

        // `n_fpha_cells` (the running total) sizes the FPHA generation column
        // family; `total_fpha_rows` sizes its plane rows.
        let (fpha_local_index, fpha_cell_local_start, n_fpha_cells, total_fpha_rows) =
            fpha_cell_offsets(ctx, &fpha_hydro_indices, stage_idx, n_h);
        let n_evap_slots = evaporation_slot_count(stage.block_mode, n_blks);

        let mut geometry = StageGeometry::unallocated(stage.block_mode, n_blks);
        geometry.fpha_hydro_indices = fpha_hydro_indices;

        // ── Role-(b) equipment column ranges ─────────────────────────────────
        // Anchored at the handle's `control_region_start()` (the role-(a)/role-(b)
        // seam); `col` allocates every family through `RangeCursor::alloc`, strided
        // by THIS stage's `n_blks` (the per-stage authority over the stage-0 global
        // stride). Adjacency between consecutive families is structural, never a
        // hand-copied `.end`.
        let mut col = RangeCursor::new(state_layout.control_region_start());
        (
            geometry.storage_internal_start,
            geometry.turbine,
            geometry.spillage,
            geometry.diversion,
        ) = allocate_hydro_columns(&mut col, n_h, n_cells, stage.block_mode, n_blks);
        geometry.thermal = col.alloc(ctx.thermals.len() * n_blks);
        let anticipated_decision_cols = col.alloc(state_layout.n_anticipated);
        // `0..0`, not `anticipated_decision_cols` itself, when `n_anticipated == 0` —
        // the empty-case value a byte-identity oracle test pins.
        geometry.anticipated_decision = if state_layout.n_anticipated > 0 {
            anticipated_decision_cols
        } else {
            0..0
        };
        (
            geometry.line_fwd,
            geometry.line_rev,
            geometry.deficit,
            geometry.excess,
        ) = allocate_network_columns(
            &mut col,
            ctx.lines.len(),
            ctx.buses.len(),
            ctx.study_dims.max_deficit_segments,
            n_blks,
        );

        let has_inflow_slack_columns = ctx.study_dims.inflow_method.has_slack_columns();
        geometry.inflow_slack = col.alloc(if has_inflow_slack_columns { n_h } else { 0 });

        // ── Role-(b) constraint row ranges ───────────────────────────────────
        // The builder's own rows start immediately after `StateSpace::z_inflow_rows()`,
        // the sole owner of that leading row range. `row` allocates every family
        // through `RangeCursor::alloc`, mirroring `col` above.
        let mut row = RangeCursor::new(state_layout.z_inflow_rows().end);
        geometry.water_balance =
            allocate_water_balance_rows(&mut row, stage.block_mode, n_h, n_blks);
        // Sized from this stage's reachable count, not the stage-invariant
        // `state_layout.n_buckets`: `build_transit_bucket_row_pos` masks a lag beyond
        // `ctx.topology.per_stage_mask[stage_idx]`'s per-plant cap out of the row
        // range entirely — the cap itself is `build_transit_bucket_topology`'s,
        // gated on `boundary_present`.
        let (transit_bucket_row_pos, transit_bucket_definition) =
            allocate_transit_bucket_rows(&mut row, ctx, stage_idx);
        geometry.load_balance = BlockRowFamily::per_block(row.alloc(ctx.buses.len() * n_blks));

        // Sized by FPHA CELL, not FPHA plant (`n_fpha_cells` == `fpha_hydro_indices.len()`
        // while every FPHA plant has one cell). `total_fpha_rows` sums `n_cells(plant) *
        // n_planes(plant)`, not `Σ n_planes(plant)`: each cell owns its own
        // `n_blks * n_planes` row block (`for_each_fpha_plane`'s per-cell advance).
        // The plant-only sum undersizes a multi-bus plant's row range, aliasing
        // rows across cells.
        (geometry.generation, geometry.fpha) =
            allocate_fpha(&mut col, &mut row, n_fpha_cells, total_fpha_rows, n_blks);

        // One `EVAP_COLS_PER_HYDRO` triple and one row per `(evap hydro, slot)`,
        // strided by `n_evap_slots` (`evaporation_slot_count`).
        (geometry.evap_hydro_indices, geometry.evap_indices) =
            allocate_evaporation(&mut col, &mut row, ctx, stage.id, n_evap_slots);

        geometry.withdrawal_slack_neg = col.alloc(n_h);
        geometry.withdrawal_slack_pos = col.alloc(n_h);
        // `n_h * n_blks`/`n_cells * n_blks` are `0` when `n_h`/`n_cells == 0`, so
        // `alloc(0)` collapses every family onto the post-equipment cursor with
        // no branch.
        (
            geometry.outflow_below_slack,
            geometry.outflow_above_slack,
            geometry.turbine_below_slack,
            geometry.generation_below_slack,
        ) = allocate_oper_violation_slack_columns(&mut col, n_h * n_blks, n_cells * n_blks);
        let oper_violation = OperViolationRanges::new(&mut row, n_h * n_blks, n_cells * n_blks);

        geometry.ncs_generation = col.alloc(ctx.non_controllable_sources.len() * n_blks);

        // σ_fill then σ^{v-} rows, in the pre-cut region after the
        // operational-violation rows. Both MUST stay strictly below `num_rows`: a
        // row at index `>= num_rows` aliases the append-only cut rows (slot-identity
        // warm-start matches cut rows from `num_rows`) and corrupts every cut.
        geometry.filling_target = row.alloc(filling_target_hydro_indices.len());
        geometry.filled_min_storage_floor = row.alloc(filled_min_storage_floor_hydro_indices.len());

        let anticipated = AnticipatedLayout::new(&mut row, ctx, stage_idx);

        geometry.pumping_flow = col.alloc(ctx.pumping_stations.len() * n_blks);

        // Import then export contract block; both empty leaves
        // col_generic_slack_start at col_pumping_end (parity-neutral).
        let (n_contract_import, n_contract_export) = contract_direction_counts(ctx.contracts);
        geometry.contract_import = col.alloc(n_contract_import * n_blks);
        geometry.contract_export = col.alloc(n_contract_export * n_blks);

        let (generic, generic_rows) =
            allocate_generic_constraints(&mut col, &mut row, ctx, stage, stage_idx, n_blks);

        // σ_fill then σ^{v-} are the last two per-stage column families; σ^{v-}
        // last so its presence cannot shift any other family's start.
        geometry.filling_target_col = col.alloc(filling_target_hydro_indices.len());
        geometry.filled_min_storage_floor_col =
            col.alloc(filled_min_storage_floor_hydro_indices.len());
        geometry.filling_target_hydro_indices = filling_target_hydro_indices;
        geometry.filled_min_storage_floor_hydro_indices = filled_min_storage_floor_hydro_indices;

        let num_cols = col.pos();
        let num_rows = row.pos();

        Self {
            state: state_layout,
            equipment: EquipmentColumns {
                max_deficit_segments: ctx.study_dims.max_deficit_segments,
                evap_col_start: geometry.generation.end,
            },
            geometry,
            anticipated,
            oper_violation,
            rows: ConstraintRows::new(
                (transit_bucket_row_pos, transit_bucket_definition),
                generic_rows,
                num_rows,
            ),
            num_cols,
            clock,
            fpha_local_index,
            fpha_cell_local_start,
            n_evap_slots,
            generic_constraint_rows: generic.generic_constraint_rows,
        }
    }

    /// Resolve a block-major LP row or column address: `start + entity * n_blks + blk`
    /// (entity is the OUTER stride factor, block the INNER offset). The transposed
    /// `blk * n_entities + entity` is the wrong-but-compiling alternative — same
    /// length, but it interleaves columns across entities and silently misbuilds the
    /// LP. Delegates to [`BlockGrid::flat`](crate::indexer::BlockGrid::flat), the
    /// single owner of the stride arithmetic.
    #[inline]
    pub(crate) fn block_flat(&self, start: usize, entity: usize, blk: BlockIdx) -> usize {
        self.block_grid().flat(start, entity, blk)
    }

    /// The [`BlockGrid`] address primitive for this stage's LP, carrying this
    /// stage's own `n_blks` and `max_deficit_segments`.
    #[inline]
    #[must_use]
    pub(crate) fn block_grid(&self) -> BlockGrid {
        BlockGrid::new(self.clock.n_blks(), self.equipment.max_deficit_segments)
    }
}

/// Entity `i`'s index in a one-per-entity family `family`.
#[inline]
#[must_use]
pub(super) fn entity_flat(family: &Range<usize>, i: usize) -> usize {
    let idx = family.start + i;
    debug_assert!(idx < family.end, "index {idx} outside {family:?}");
    idx
}

/// Entity `i`'s row within a sparse row family, or `None` when `i` is absent
/// or the family masks it out — the position table's own `None` behavior, not
/// a range-membership check.
#[inline]
#[must_use]
pub(super) fn position_table_row(
    row_start: usize,
    row_pos: &[Option<usize>],
    i: usize,
) -> Option<usize> {
    row_pos.get(i).copied().flatten().map(|pos| row_start + pos)
}

/// Flat, slot-major `(evap hydro local_idx, slot)` stride: single owner of the
/// evaporation stride every column and row family built from it shares.
#[inline]
fn evap_slot_flat(local_idx: usize, slot: usize, n_evap_slots: usize) -> usize {
    debug_assert!(slot < n_evap_slots);
    local_idx * n_evap_slots + slot
}

impl StageLayout<'_> {
    /// FPHA-local plant `local_idx`'s first cell, as an [`FphaCellLocal`]. This is
    /// the plant's *base*, not its only cell: callers add the cell's offset within
    /// the plant, so it is exact at any cell count.
    #[inline]
    pub(crate) fn fpha_local_first_cell(&self, local_idx: FphaLocal) -> FphaCellLocal {
        FphaCellLocal::new(self.fpha_cell_local_start[local_idx.get()])
    }

    /// Hydro `h_idx`'s [`StageProductionRole`] at `stage_idx`: `Fpha` when
    /// `identify_fpha_hydros` admitted it into `fpha_local_index`, else the
    /// resolved model's `Constant` productivity or, for an `Fpha`-resolved
    /// model excluded by the phase gate, `Dormant`.
    #[inline]
    pub(super) fn stage_production_role(
        &self,
        production_models: &ProductionModelSet,
        h_idx: usize,
        stage_idx: usize,
    ) -> StageProductionRole {
        if let Some(local_idx) = self.fpha_local_index[h_idx] {
            debug_assert!(
                matches!(
                    production_models.model(h_idx, stage_idx),
                    ResolvedProductionModel::Fpha { .. }
                ),
                "FPHA local-index table inconsistent with production model for hydro {h_idx}"
            );
            return StageProductionRole::Fpha(local_idx);
        }
        match production_models.model(h_idx, stage_idx) {
            ResolvedProductionModel::ConstantProductivity { productivity } => {
                StageProductionRole::Constant(*productivity)
            }
            ResolvedProductionModel::Fpha { .. } => StageProductionRole::Dormant,
        }
    }

    #[inline]
    pub(crate) fn min_outflow_row(&self, h: HydroSys, blk: BlockIdx) -> usize {
        self.block_flat(self.oper_violation.min_outflow.start, h.get(), blk)
    }

    #[inline]
    pub(crate) fn max_outflow_row(&self, h: HydroSys, blk: BlockIdx) -> usize {
        self.block_flat(self.oper_violation.max_outflow.start, h.get(), blk)
    }

    #[inline]
    pub(crate) fn min_turbine_row(&self, c: HydroCell, blk: BlockIdx) -> usize {
        self.block_flat(self.oper_violation.min_turbine.start, c.get(), blk)
    }

    #[inline]
    pub(crate) fn min_generation_row(&self, c: HydroCell, blk: BlockIdx) -> usize {
        self.block_flat(self.oper_violation.min_generation.start, c.get(), blk)
    }

    /// Anticipated-local `local`'s commitment-maturity row, or `None` when no
    /// delivery matures this stage (including a `K = 0` self-delivery).
    #[inline]
    pub(crate) fn anticipated_fishing_row(&self, local: AnticipatedLocal) -> Option<usize> {
        position_table_row(
            self.anticipated.fishing_rows.start,
            &self.anticipated.anticipated_fishing_row_pos,
            local.get(),
        )
    }

    /// Anticipated-local `local`'s deposit-definition row, or `None` when the
    /// plant has no genuine, active decision this stage.
    #[inline]
    pub(crate) fn anticipated_state_out_def_row(&self, local: AnticipatedLocal) -> Option<usize> {
        position_table_row(
            self.anticipated.state_out_def_rows.start,
            &self.anticipated.anticipated_decision_row_pos,
            local.get(),
        )
    }

    /// Transit-bucket definition row for plant-local `slot` within `plant`'s
    /// contiguous bucket sub-range, or `None` when that lag is beyond this
    /// stage's reachable cap.
    #[inline]
    pub(crate) fn transit_bucket_definition_row(
        &self,
        plant: &Range<usize>,
        slot: usize,
    ) -> Option<usize> {
        position_table_row(
            self.rows.transit_bucket_definition.start,
            &self.rows.transit_bucket_row_pos[plant.clone()],
            slot,
        )
    }
}

/// A contract's [`ContractType`] and its PER-FAMILY slot — the count of
/// same-direction contracts that precede `c_sys` in the id-sorted `contracts` slice.
///
/// The dense column layout addresses each family by this per-family slot (the running
/// position within its own direction), NOT the combined slot, so both the LP-column
/// fill and the resolver must agree on it; sharing this one derivation keeps them
/// consistent.
pub(crate) fn contract_family_slot(
    contracts: &[EnergyContract],
    c_sys: usize,
) -> (ContractType, usize) {
    let contract_type = contracts[c_sys].contract_type;
    let family_slot = contracts[..c_sys]
        .iter()
        .filter(|c| c.contract_type == contract_type)
        .count();
    (contract_type, family_slot)
}

impl StageLayout<'_> {
    /// Base column of the `(evap hydro local_idx, slot)` triple, slot-major
    /// (`(local_idx * n_evap_slots + slot) * EVAP_COLS_PER_HYDRO`). Single owner of
    /// the evaporation block stride; the three offset accessors add their offset to
    /// it. The transposed `slot * n_evap_hydros + local_idx` stride compiles and
    /// silently aliases one hydro's slot onto another's.
    #[inline]
    fn evap_triple_base(&self, local_idx: usize, slot: BlockIdx) -> usize {
        self.equipment.evap_col_start
            + evap_slot_flat(local_idx, slot.get(), self.n_evap_slots) * EVAP_COLS_PER_HYDRO
    }

    /// Evaporation-outflow column for `(evap hydro local_idx, block blk)` (the
    /// [`EVAP_FLOW_OFFSET`] column of the block's triple).
    #[inline]
    pub(crate) fn evap_flow_col(&self, local_idx: EvapLocal, blk: BlockIdx) -> usize {
        self.evap_triple_base(local_idx.get(), blk) + EVAP_FLOW_OFFSET
    }

    /// `f_evap_plus` (under-evaporation slack) column for `(evap hydro local_idx,
    /// block blk)` (the [`EVAP_F_PLUS_OFFSET`] column of the block's triple).
    #[inline]
    pub(crate) fn evap_f_plus_col(&self, local_idx: EvapLocal, blk: BlockIdx) -> usize {
        self.evap_triple_base(local_idx.get(), blk) + EVAP_F_PLUS_OFFSET
    }

    /// `f_evap_minus` (over-evaporation slack) column for `(evap hydro local_idx,
    /// block blk)` (the [`EVAP_F_MINUS_OFFSET`] column of the block's triple).
    #[inline]
    pub(crate) fn evap_f_minus_col(&self, local_idx: EvapLocal, blk: BlockIdx) -> usize {
        self.evap_triple_base(local_idx.get(), blk) + EVAP_F_MINUS_OFFSET
    }

    /// Deficit column for bus `bus`, segment `seg_idx`, block `blk`. Three-term
    /// stride owned by [`BlockGrid::deficit`](crate::indexer::BlockGrid::deficit).
    #[inline]
    pub(crate) fn deficit_col(&self, bus: BusSys, seg_idx: usize, blk: BlockIdx) -> usize {
        self.geometry
            .deficit_col(bus, seg_idx, blk, self.equipment.max_deficit_segments)
    }

    /// Storage column at chronological `boundary` for hydro `h`; delegates to
    /// [`StorageBoundaryGrid::col`], the single owner of the endpoints-vs-interior
    /// split. At `n_blks = 1` only the two endpoints resolve (no interior).
    #[inline]
    pub(crate) fn block_storage_col(&self, h: HydroSys, boundary: Boundary) -> usize {
        self.geometry.block_storage_col(self.state, h, boundary)
    }

    // ── Role-(a) accessors (read through the borrowed StateSpace handle) ─────────

    /// Theta (future-cost) column; reads `self.state.theta`.
    #[inline]
    #[must_use]
    pub(crate) fn col_theta(&self) -> usize {
        self.state.theta
    }

    /// Column-side state dimension; reads `self.state.n_state`.
    #[inline]
    #[must_use]
    pub(crate) fn n_state(&self) -> usize {
        self.state.n_state
    }

    // ── Role-(b) accessors (read StageLayout's own fields) ───────────────────────

    /// Start of evaporation constraint rows, one per `(evap hydro, slot)`; see
    /// [`Self::evap_row`]. The evaporation row block follows the FPHA rows even
    /// when empty — reads `self.geometry.fpha.end`.
    #[inline]
    #[must_use]
    pub(crate) fn row_evap_start(&self) -> usize {
        self.geometry.fpha.end
    }

    /// Evaporation-equality row for `(evap hydro local, slot)`, slot-major over
    /// [`Self::row_evap_start`] — the row-side sibling of [`Self::evap_flow_col`].
    #[inline]
    #[must_use]
    pub(crate) fn evap_row(&self, local: EvapLocal, slot: BlockIdx) -> usize {
        self.row_evap_start() + evap_slot_flat(local.get(), slot.get(), self.n_evap_slots)
    }

    /// Filling-target-local `local`'s soft `σ_fill` row, over [`StageGeometry::filling_target`].
    #[inline]
    #[must_use]
    pub(crate) fn filling_target_row(&self, local: FillingTargetLocal) -> usize {
        entity_flat(&self.geometry.filling_target, local.get())
    }

    /// Floor-local `local`'s soft `σ^{v-}` operating-floor row, over
    /// [`StageGeometry::filled_min_storage_floor`].
    #[inline]
    #[must_use]
    pub(crate) fn filled_min_storage_floor_row(&self, local: FloorLocal) -> usize {
        entity_flat(&self.geometry.filled_min_storage_floor, local.get())
    }

    /// Generic constraint row `entry_idx`'s row, over
    /// `row_generic_start..row_generic_start + n_generic_rows`.
    #[inline]
    #[must_use]
    pub(crate) fn generic_row(&self, entry_idx: usize) -> usize {
        entity_flat(
            &(self.rows.row_generic_start..self.rows.row_generic_start + self.rows.n_generic_rows),
            entry_idx,
        )
    }

    /// Hydro `h`'s z-inflow definition row.
    #[inline]
    #[must_use]
    pub(crate) fn z_inflow_row(&self, h: HydroSys) -> usize {
        self.state.z_inflow_row(h)
    }
}

/// Per-stage equipment geometry for simulation extraction: the stage-correct
/// column/row `Range`s, identity lists, and block count for every block-major
/// family, each computed from **this** stage's `StageLayout`.
///
/// A single global stage-0 geometry is the bug this struct forbids: every family
/// after `turbine` has a base `turbine.start + Σ(prior)·n_blks` and length
/// `count·n_blks`, both striped by stage 0's block count, so at any stage with a
/// differing block count the stage-0 base/length addresses the WRONG primal
/// columns. The per-stage `n_blks` stride was already correct; this closes the
/// matching base/length gap. Uniform-block studies coincide with stage 0.
#[derive(Debug, Clone)]
pub struct StageGeometry {
    /// Turbined-flow column range (one per hydro per block). `turbine.start` is
    /// `theta + 1` and stage-invariant, but `turbine.end` is `n_blks`-dependent,
    /// so the cost-breakdown `range_sum` still needs the per-stage range.
    pub turbine: Range<usize>,
    /// Spillage column range (one per hydro per block).
    pub spillage: Range<usize>,
    /// Diversion-flow column range (one per hydro per block).
    pub diversion: Range<usize>,
    /// Thermal-generation column range (one per thermal per block).
    pub thermal: Range<usize>,
    /// Anticipated-decision column range (one per anticipated thermal,
    /// stage-level). Starts at `thermal.end`, which is `n_blks`-dependent, so the
    /// cost-breakdown `range_sum` needs the per-stage base.
    pub anticipated_decision: Range<usize>,
    /// Forward line-flow column range (one per line per block).
    pub line_fwd: Range<usize>,
    /// Reverse line-flow column range (one per line per block).
    pub line_rev: Range<usize>,
    /// Bus-deficit column range (`B · S · K` columns).
    pub deficit: Range<usize>,
    /// Bus-excess column range (one per bus per block).
    pub excess: Range<usize>,
    /// FPHA-generation column range (one per FPHA hydro per block).
    pub generation: Range<usize>,
    /// Dense, system-indexed, block-major NCS generation column family; reached
    /// through [`StageGeometry::ncs_generation_col`].
    pub ncs_generation: Range<usize>,
    /// Dense, system-indexed, block-major pumping-flow column family; reached
    /// through [`StageGeometry::pumping_flow_col`].
    pub pumping_flow: Range<usize>,
    /// Per-`(evaporation hydro, slot)` column/row indices, slot-major
    /// (`local_evap_idx * slots + slot`) — one slot per evaporating hydro on a
    /// parallel stage, one per block on a chronological stage
    /// (`evaporation_slot_count`). Anchored at the `n_blks`-dependent
    /// FPHA-generation-block end, so they shift under a non-uniform schedule —
    /// this per-stage copy carries the stage-correct columns.
    pub evap_indices: Vec<EvaporationIndices>,
    /// Inflow non-negativity slack column range (one per hydro, stage-level).
    pub inflow_slack: Range<usize>,
    /// Under-withdrawal slack column range (one per hydro, stage-level).
    pub withdrawal_slack_neg: Range<usize>,
    /// Over-withdrawal slack column range (one per hydro, stage-level).
    pub withdrawal_slack_pos: Range<usize>,
    /// Outflow-below-minimum slack column range (one per hydro per block).
    pub outflow_below_slack: Range<usize>,
    /// Outflow-above-maximum slack column range (one per hydro per block).
    pub outflow_above_slack: Range<usize>,
    /// Turbine-below-minimum slack column range (one per hydro CELL per block).
    pub turbine_below_slack: Range<usize>,
    /// Generation-below-minimum slack column range (one per hydro CELL per block).
    pub generation_below_slack: Range<usize>,
    /// Import-contract column range (one per import contract per block); empty
    /// `start..start` (not `0..0`) at the pumping-end column when there are none.
    pub contract_import: Range<usize>,
    /// Export-contract column range (one per export contract per block); empty
    /// `start..start` at the import-end column when there are none.
    pub contract_export: Range<usize>,

    // ── Per-stage row ranges, identity lists, and block count ────────────────
    /// Water-balance row family, strided by [`Self::n_blks`]; owns the shape
    /// (one row per hydro on a parallel stage, one per hydro per block on a
    /// chronological stage). Address a row through
    /// [`StageGeometry::water_balance_row`], never `.range()` arithmetic.
    pub water_balance: BlockRowFamily,
    /// Load-balance row family (one row per bus per block; `n_buses · n_blks`),
    /// strided by [`Self::n_blks`]. Address a row through
    /// [`StageGeometry::load_balance_row`].
    pub load_balance: BlockRowFamily,
    /// FPHA hyperplane row range, immediately following `load_balance`. Length
    /// varies per stage: `for_each_fpha_plane` sums plane counts that differ per
    /// hydro (`fpha_hydro_indices.len() * n_blks` is NOT the row count).
    pub fpha: Range<usize>,
    /// Per-stage `σ_fill`-target row range (one row per Filling-phase hydro); empty
    /// `start..start` (not `0..0`) at every non-Filling stage.
    pub filling_target: Range<usize>,
    /// Per-stage `σ_fill`-target slack column range (one column per Filling-phase
    /// hydro); empty `start..start` at every non-Filling stage. Simulation
    /// extraction reads the `σ_fill` primal at `start + local_idx`, resolving
    /// `local_idx` via `filling_target_hydro_indices`.
    pub filling_target_col: Range<usize>,
    /// Soft `σ^{v-}` operating-floor row range (one row per Operating-phase filling
    /// hydro); empty `start..start` (not `0..0`) at every non-operating stage.
    pub filled_min_storage_floor: Range<usize>,
    /// Soft `σ^{v-}` operating-floor slack column range (one column per
    /// Operating-phase filling hydro); empty `start..start` at every non-operating
    /// stage. Simulation extraction reads the `σ^{v-}` primal at `start + local_idx`,
    /// resolving `local_idx` via `filled_min_storage_floor_hydro_indices`.
    pub filled_min_storage_floor_col: Range<usize>,
    /// Number of operating blocks (K) at this stage — the block-major stride for
    /// every equipment family.
    pub n_blks: usize,
    /// Interior storage-boundary anchor for this stage, mirroring
    /// `StageLayout`'s own `equipment.storage_internal_start`; feeds
    /// [`StageGeometry::storage_boundary_grid`].
    pub storage_internal_start: usize,
    /// Block formulation mode at this stage. Selects per-block storage extraction
    /// (`Chronological` reads each block's own `(Sᵇ, Sᵇ⁺¹)` boundary) versus the
    /// stage-level `(S⁰, Sᴷ)` pair (`Parallel`); defaults to `Parallel`.
    pub block_mode: BlockMode,
    /// System hydro indices using FPHA at this stage, in slot order. FPHA
    /// membership is per `(hydro, stage)`, so this is the stage-correct list.
    pub fpha_hydro_indices: Vec<HydroSys>,
    /// System hydro indices with linearized evaporation at this stage, in slot
    /// order. Parallel to `evap_indices`.
    pub evap_hydro_indices: Vec<HydroSys>,
    /// System hydro indices owning a `σ_fill`-target slack column at this stage (the
    /// Filling-phase hydros), in slot order. Parallel to `filling_target_col` (slot
    /// `i` → `filling_target_col.start + i`). The family is SPARSE — one column per
    /// filling hydro — so extraction resolves a system hydro's column via this
    /// system→slot list, never by the dense system index `h`.
    pub filling_target_hydro_indices: Vec<HydroSys>,
    /// System hydro indices owning a `σ^{v-}` operating-floor slack column at this
    /// stage (the Operating-phase filling hydros), in slot order. Parallel to
    /// `filled_min_storage_floor_col`; SPARSE like `filling_target_hydro_indices`,
    /// resolved the same way.
    pub filled_min_storage_floor_hydro_indices: Vec<HydroSys>,
}

impl StageGeometry {
    /// Maximum block count across every stage's geometry; `0` for an empty slice.
    /// The sole max-over-stages block count in the crate.
    #[inline]
    #[must_use]
    pub(crate) fn max_blocks(per_stage: &[StageGeometry]) -> usize {
        per_stage.iter().map(|g| g.n_blks).max().unwrap_or(0)
    }

    /// Storage column at chronological `boundary` for hydro `h`, so the
    /// simulation read-path resolves per-block boundaries without a
    /// `StageLayout`; delegates to
    /// [`StorageBoundaryGrid::col`](crate::lp::indexer::StorageBoundaryGrid::col),
    /// the single owner of the endpoints-vs-interior split.
    #[inline]
    #[must_use]
    pub fn block_storage_col(&self, state: &StateSpace, h: HydroSys, boundary: Boundary) -> usize {
        self.storage_boundary_grid().col(state, h, boundary)
    }

    /// The [`StorageBoundaryGrid`] address primitive for this stage's LP,
    /// carrying its interior anchor.
    #[inline]
    #[must_use]
    pub fn storage_boundary_grid(&self) -> StorageBoundaryGrid {
        StorageBoundaryGrid::new(self.storage_internal_start, self.n_blks)
    }

    /// Resolve a block-major column within `family` and debug-assert it stays
    /// inside it — the single home for the bounds check every accessor below
    /// shares.
    #[inline]
    fn block_flat(&self, family: &Range<usize>, entity: usize, blk: BlockIdx) -> usize {
        let col = BlockGrid::new(self.n_blks, 0).flat(family.start, entity, blk);
        debug_assert!(col < family.end, "column {col} outside {family:?}");
        col
    }

    /// Turbine-flow column for cell `c`, block `blk`.
    #[inline]
    #[must_use]
    pub fn turbine_col(&self, c: HydroCell, blk: BlockIdx) -> usize {
        self.block_flat(&self.turbine, c.get(), blk)
    }

    /// Spillage column for hydro `h`, block `blk`.
    #[inline]
    #[must_use]
    pub fn spillage_col(&self, h: HydroSys, blk: BlockIdx) -> usize {
        self.block_flat(&self.spillage, h.get(), blk)
    }

    /// Diversion-flow column for hydro `h`, block `blk`.
    #[inline]
    #[must_use]
    pub fn diversion_col(&self, h: HydroSys, blk: BlockIdx) -> usize {
        self.block_flat(&self.diversion, h.get(), blk)
    }

    /// Outflow-below-minimum slack column for hydro `h`, block `blk`.
    #[inline]
    #[must_use]
    pub fn outflow_below_col(&self, h: HydroSys, blk: BlockIdx) -> usize {
        self.block_flat(&self.outflow_below_slack, h.get(), blk)
    }

    /// Outflow-above-maximum slack column for hydro `h`, block `blk`.
    #[inline]
    #[must_use]
    pub fn outflow_above_col(&self, h: HydroSys, blk: BlockIdx) -> usize {
        self.block_flat(&self.outflow_above_slack, h.get(), blk)
    }

    /// FPHA generation column for FPHA-cell-local index `c`, block `blk`.
    #[inline]
    #[must_use]
    pub fn generation_col(&self, c: FphaCellLocal, blk: BlockIdx) -> usize {
        self.block_flat(&self.generation, c.get(), blk)
    }

    /// Thermal-generation column for thermal `t`, block `blk`.
    #[inline]
    #[must_use]
    pub fn thermal_col(&self, t: ThermalSys, blk: BlockIdx) -> usize {
        self.block_flat(&self.thermal, t.get(), blk)
    }

    /// Forward line-flow column for line `l`, block `blk`.
    #[inline]
    #[must_use]
    pub fn line_fwd_col(&self, l: LineSys, blk: BlockIdx) -> usize {
        self.block_flat(&self.line_fwd, l.get(), blk)
    }

    /// Reverse line-flow column for line `l`, block `blk`.
    #[inline]
    #[must_use]
    pub fn line_rev_col(&self, l: LineSys, blk: BlockIdx) -> usize {
        self.block_flat(&self.line_rev, l.get(), blk)
    }

    /// Bus-excess column for bus `bus`, block `blk`.
    #[inline]
    #[must_use]
    pub fn excess_col(&self, bus: BusSys, blk: BlockIdx) -> usize {
        self.block_flat(&self.excess, bus.get(), blk)
    }

    /// Turbine-below-minimum slack column for cell `c`, block `blk`.
    #[inline]
    #[must_use]
    pub fn turbine_below_col(&self, c: HydroCell, blk: BlockIdx) -> usize {
        self.block_flat(&self.turbine_below_slack, c.get(), blk)
    }

    /// Generation-below-minimum slack column for cell `c`, block `blk`.
    #[inline]
    #[must_use]
    pub fn generation_below_col(&self, c: HydroCell, blk: BlockIdx) -> usize {
        self.block_flat(&self.generation_below_slack, c.get(), blk)
    }

    /// `contract_type`'s contract column at per-direction slot `family_slot`
    /// (from [`contract_family_slot`]) for block `blk`.
    #[inline]
    #[must_use]
    pub fn contract_col(
        &self,
        contract_type: ContractType,
        family_slot: usize,
        blk: BlockIdx,
    ) -> usize {
        let family = match contract_type {
            ContractType::Import => &self.contract_import,
            ContractType::Export => &self.contract_export,
        };
        self.block_flat(family, family_slot, blk)
    }

    /// Deficit column for bus `bus`, segment `seg`, block `blk`, given the
    /// study's `max_segments` ([`StudyDimensions::max_deficit_segments`](crate::lp::indexer::StudyDimensions::max_deficit_segments)).
    #[inline]
    #[must_use]
    pub fn deficit_col(
        &self,
        bus: BusSys,
        seg: usize,
        blk: BlockIdx,
        max_segments: usize,
    ) -> usize {
        let col = BlockGrid::new(self.n_blks, max_segments).deficit(
            self.deficit.start,
            bus.get(),
            seg,
            blk,
        );
        debug_assert!(
            col < self.deficit.end,
            "deficit column {col} outside {:?}",
            self.deficit
        );
        col
    }

    /// Hydro `h`'s water-balance row for block `blk`: its own block row on a
    /// chronological stage, its single stage row on a parallel stage (every block
    /// reads the same row).
    #[inline]
    #[must_use]
    pub fn water_balance_row(&self, h: HydroSys, blk: BlockIdx) -> usize {
        self.water_balance.row(h.get(), blk, self.n_blks)
    }

    /// Bus `bus`'s load-balance row for block `blk` (`n_buses · n_blks` rows,
    /// strided by [`Self::n_blks`]).
    #[inline]
    #[must_use]
    pub fn load_balance_row(&self, bus: BusSys, blk: BlockIdx) -> usize {
        self.load_balance.row(bus.get(), blk, self.n_blks)
    }

    /// NCS entity `ncs_sys`'s generation column for block `blk`.
    #[inline]
    #[must_use]
    pub fn ncs_generation_col(&self, ncs_sys: NcsSys, blk: BlockIdx) -> usize {
        self.block_flat(&self.ncs_generation, ncs_sys.get(), blk)
    }

    /// Pumping station `pumping_sys`'s flow column for block `blk`.
    #[inline]
    #[must_use]
    pub fn pumping_flow_col(&self, pumping_sys: PumpingSys, blk: BlockIdx) -> usize {
        self.block_flat(&self.pumping_flow, pumping_sys.get(), blk)
    }

    /// Anticipated-local `local`'s ring decision column.
    #[inline]
    #[must_use]
    pub fn anticipated_decision_col(&self, local: AnticipatedLocal) -> usize {
        entity_flat(&self.anticipated_decision, local.get())
    }

    /// Hydro `h`'s inflow-penalty slack column.
    #[inline]
    #[must_use]
    pub fn inflow_slack_col(&self, h: HydroSys) -> usize {
        entity_flat(&self.inflow_slack, h.get())
    }

    /// Hydro `h`'s below-withdrawal-target slack column.
    #[inline]
    #[must_use]
    pub fn withdrawal_slack_neg_col(&self, h: HydroSys) -> usize {
        entity_flat(&self.withdrawal_slack_neg, h.get())
    }

    /// Hydro `h`'s above-withdrawal-target slack column.
    #[inline]
    #[must_use]
    pub fn withdrawal_slack_pos_col(&self, h: HydroSys) -> usize {
        entity_flat(&self.withdrawal_slack_pos, h.get())
    }

    /// Filling-target-local `local`'s `σ_fill` slack column.
    #[inline]
    #[must_use]
    pub fn filling_target_slack_col(&self, local: FillingTargetLocal) -> usize {
        entity_flat(&self.filling_target_col, local.get())
    }

    /// Floor-local `local`'s `σ^{v-}` operating-floor slack column.
    #[inline]
    #[must_use]
    pub fn filled_min_storage_floor_slack_col(&self, local: FloorLocal) -> usize {
        entity_flat(&self.filled_min_storage_floor_col, local.get())
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod collapse_stage_level_tests {
    use super::*;
    use cobre_core::{LinearTerm, VariableRef};

    fn expr(term: LinearTerm) -> ConstraintExpression {
        ConstraintExpression { terms: vec![term] }
    }

    fn hydro_storage() -> VariableRef {
        VariableRef::HydroStorage {
            hydro_id: EntityId(1),
        }
    }

    /// Slot 42 stores two block values at stage 0 (block-varying); slot 43 stores a
    /// length-1 inner (block-invariant broadcast).
    fn resolved() -> ResolvedParameters {
        ResolvedParameters {
            per_param: vec![vec![vec![1.0, 2.0]], vec![vec![5.0]]],
            id_to_slot: vec![(42, 0), (43, 1)],
            ..Default::default()
        }
    }

    #[test]
    fn block_varying_coefficient_suppresses_collapse() {
        let e = expr(LinearTerm::parameter(EntityId(42), 1.0, hydro_storage()));
        assert!(
            !expression_collapses_to_stage_level(&e, &resolved()),
            "a block-varying coefficient over a block-independent variable must not collapse"
        );
    }

    #[test]
    fn block_invariant_coefficient_still_collapses() {
        let param = expr(LinearTerm::parameter(EntityId(43), 1.0, hydro_storage()));
        let literal = expr(LinearTerm::literal(1.0, hydro_storage()));
        let r = resolved();
        assert!(expression_collapses_to_stage_level(&param, &r));
        assert!(expression_collapses_to_stage_level(&literal, &r));
    }

    #[test]
    fn block_dependent_variable_never_collapses() {
        let e = expr(LinearTerm::literal(
            1.0,
            VariableRef::ThermalGeneration {
                thermal_id: EntityId(0),
                block_id: None,
            },
        ));
        assert!(!expression_collapses_to_stage_level(&e, &resolved()));
    }

    fn constraint_with_refs(
        lower_ref: Option<EntityId>,
        upper_ref: Option<EntityId>,
    ) -> GenericConstraint {
        GenericConstraint {
            id: EntityId(0),
            name: "c".to_string(),
            description: None,
            expression: expr(LinearTerm::literal(1.0, hydro_storage())),
            slack: cobre_core::SlackConfig {
                enabled: false,
                penalty: None,
            },
            bound_lower_affine: lower_ref.map(AffineBound::single),
            bound_upper_affine: upper_ref.map(AffineBound::single),
        }
    }

    #[test]
    fn bound_affine_block_varying_truth_table() {
        let r = resolved();
        // Slot 42 is block-varying, slot 43 broadcasts.
        assert!(bound_affine_is_block_varying(
            &constraint_with_refs(None, Some(EntityId(42))),
            &r
        ));
        assert!(bound_affine_is_block_varying(
            &constraint_with_refs(Some(EntityId(42)), None),
            &r
        ));
        assert!(!bound_affine_is_block_varying(
            &constraint_with_refs(None, Some(EntityId(43))),
            &r
        ));
        assert!(!bound_affine_is_block_varying(
            &constraint_with_refs(None, None),
            &r
        ));
    }

    /// A block-varying parameter reached through a multi-term affine bound (not
    /// just the `single` special case) still suppresses the collapse.
    #[test]
    fn bound_affine_block_varying_detects_multi_term_reference() {
        let r = resolved();
        let mut constraint = constraint_with_refs(None, None);
        constraint.bound_upper_affine = Some(AffineBound {
            constant: 10.0,
            terms: vec![(2.0, EntityId(43)), (0.5, EntityId(42))],
        });
        assert!(bound_affine_is_block_varying(&constraint, &r));
    }

    #[test]
    fn resolve_affine_of_single_equals_get() {
        let r = resolved();
        let bound = AffineBound::single(EntityId(42));
        assert_eq!(resolve_affine(&bound, &r, 0, 1), r.get(EntityId(42), 0, 1));
    }

    #[test]
    fn resolve_affine_of_two_term_remainder_sums_constant_and_terms() {
        let r = resolved();
        let bound = AffineBound {
            constant: 100.0,
            terms: vec![(2.0, EntityId(42)), (-1.0, EntityId(43))],
        };
        let expected = 100.0 + 2.0 * r.get(EntityId(42), 0, 1) - 1.0 * r.get(EntityId(43), 0, 1);
        assert_eq!(resolve_affine(&bound, &r, 0, 1), expected);
    }
}
