use std::collections::{BTreeMap, HashMap};
use std::ops::Range;

use cobre_core::commissioning::{Phase, filling_phase};
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

use super::template::StageGeometry;
use super::{
    EVAP_COLS_PER_HYDRO, EVAP_F_MINUS_OFFSET, EVAP_F_PLUS_OFFSET, EVAP_FLOW_OFFSET,
    GenericConstraintRowEntry,
};
use crate::block_clock::BlockClock;
use crate::generic_constraints::expression_is_block_independent;
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
    /// The role-(a) state layout, threaded from setup's single owner
    /// (`resolve_state_layout`) — owns `anticipated_lead_stages`
    /// and `anticipated_resolution`, which this ctx used to carry as its own
    /// copies.
    // Rationale: read only by tests and fixtures so far; production call sites
    // still thread the state layout as their own separate parameter alongside
    // this ctx.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "read only by tests and fixtures so far")
    )]
    pub(crate) state: &'a StateSpace,
    /// Study-invariant, non-state LP shape (`has_inflow_penalty`,
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
    /// Start of the anticipated-decision column block: `n_anticipated`
    /// columns (`col_anticipated_decision_start + local_idx`). Equals
    /// `col_thermal_end`.
    pub(crate) col_anticipated_decision_start: usize,
    /// Start of the `anticipated_state_out_def` equality row block: one row
    /// per plant with a genuine, ACTIVE decision this stage
    /// (`PointResolution::genuine_decisions_at(stage_idx).next()`, AND the
    /// delivery stage's commissioning window), pinning that decision's ring
    /// slot (`ring_index(delivery_stage) mod k_max`) to its decision column. Immediately
    /// after `row_anticipated_fishing_start`.
    pub(crate) row_anticipated_state_out_def_start: usize,
    /// Count of genuine, active decisions this stage (`Some` count of
    /// `anticipated_decision_row_pos`); drives the active-row iteration.
    pub(crate) n_anticipated_state_out_def_rows: usize,
    /// For each plant (local order), this stage's compact row position
    /// within the deposit-row family, or `None` when the plant has no
    /// genuine decision this stage (`PointResolution::genuine_decisions_at`)
    /// or the delivery is commissioning-inactive. Length `n_anticipated`.
    pub(crate) anticipated_decision_row_pos: Vec<Option<usize>>,
    /// Start of the commitment-MATURITY rows: one per anticipated plant
    /// whose delivery matures THIS stage (`PointResolution::is_anticipated_at`,
    /// `false` at a `K = 0` self-delivery). Every such plant gets exactly one
    /// row here regardless of commissioning activeness — maturity always
    /// fishes, via [`super::entries::fill_anticipated_fishing_entries`].
    /// After operational-violation rows.
    pub(crate) row_anticipated_fishing_start: usize,
    /// Commitment-maturity row count this stage (`Some` count of
    /// `anticipated_fishing_row_pos`).
    pub(crate) n_anticipated_fishing_rows: usize,
    /// For each anticipated plant (local order), this stage's compact row
    /// position within the maturity-row family, or `None` when no delivery
    /// matures this stage (including a `K = 0` self-delivery, which never
    /// matures through the ring at all). Length `n_anticipated`.
    pub(crate) anticipated_fishing_row_pos: Vec<Option<usize>>,
    /// Start of the future-window commitment-carry equality rows (same-slot
    /// hold, `slot^out − slot^in = 0`, routed by
    /// `fill_anticipated_slot_definition_entries` via
    /// [`super::delivery_ring::DeliveryRing::emit_carry_rows`]): every
    /// STRICTLY FUTURE, not-yet-due in-study slot, modular-addressed
    /// (`ring_index(delivery_target) mod k_max`). The commitment maturing THIS
    /// stage is never here — it always fishes through the maturity row above;
    /// carry-to-terminal belongs to the post-study-targeted slot alone, so this
    /// family and `row_anticipated_fishing_start` never double-book the same
    /// delivery. Immediately after `row_anticipated_state_out_def_start`.
    pub(crate) row_anticipated_slot_definition_start: usize,
    /// Count of future-window carrying slots this stage
    /// (`anticipated_slot_row_pos`'s `Some` count).
    pub(crate) n_anticipated_slot_definition_rows: usize,
    /// For each GLOBAL in-study commitment-hold slot (`(ring_index(m) mod k_max) *
    /// n_anticipated + plant`, modular slot-major/plant-minor —
    /// [`StateSpace::commitment_hold_in_study_offset`]'s own addressing), this
    /// stage's compact row position within the future-window carry-row
    /// family, or `None` when the slot's target is this stage's own latch
    /// (`row_anticipated_state_out_def_start` owns it), matures THIS stage
    /// (always fished instead, `row_anticipated_fishing_start` owns it), is
    /// beyond the study horizon, or is not yet ready
    /// (`PointResolution::is_ready_at`). Length `n_anticipated * k_max`.
    pub(crate) anticipated_slot_row_pos: Vec<Option<usize>>,
}

/// Equipment column ranges and their block-start cursors: every dispatchable
/// piece of equipment (storage/turbine/spillage/diversion/thermal/lines/
/// deficit/excess/generation/evaporation/NCS/pumping/contracts), anchored at
/// the handle's [`StateSpace::control_region_start`].
pub(crate) struct EquipmentColumns {
    /// Control-region anchor for the interior storage boundaries `S¹ … Sᴷ⁻¹`
    /// (= `control_region_start()`), read even when the family reserves no
    /// columns (parallel mode, or `K = 1`). Within-family address is
    /// `storage_internal_start + h * (n_blks − 1) + (k − 1)` for interior boundary
    /// `k ∈ 1..n_blks` — stride `n_blks − 1`, not `n_blks`.
    pub(crate) storage_internal_start: usize,
    /// Column range for turbined flow (one per partition **cell** per block, not
    /// per hydro — a plant whose unit groups span two buses owns two).
    pub(crate) turbine: Range<usize>,
    /// Column range for spillage (one per hydro per block).
    pub(crate) spillage: Range<usize>,
    /// Column range for diversion flow (one per hydro per block).
    pub(crate) diversion: Range<usize>,
    /// Column range for thermal generation (one per thermal per block).
    pub(crate) thermal: Range<usize>,
    /// Column range for forward line flow (one per line per block).
    pub(crate) line_fwd: Range<usize>,
    /// Column range for reverse line flow (one per line per block).
    pub(crate) line_rev: Range<usize>,
    /// Column range for bus deficit variables (`B * S * K` columns).
    pub(crate) deficit: Range<usize>,
    /// Maximum deficit segments across buses (`S`); the deficit-stride constant.
    pub(crate) max_deficit_segments: usize,
    /// Column range for bus excess variables (one per bus per block).
    pub(crate) excess: Range<usize>,
    /// Column-block cursor at which the FPHA generation block begins, even when
    /// that block is empty. Always `inflow_slack.end` — `RangeCursor::alloc(0)`
    /// leaves the cursor at `excess.end` when the penalty is inactive, which is
    /// also what `inflow_slack.end` reads there.
    pub(crate) generation_col_start: usize,
    /// Column range for FPHA generation (one per FPHA **cell** per block, not per
    /// FPHA hydro).
    pub(crate) generation: Range<usize>,
    /// Column-block cursor at which the evaporation block begins, even when empty
    /// (`generation_col_start + n_fpha_cells * n_blks`).
    pub(crate) evap_col_start: usize,
    /// Start of NCS generation columns (one per NCS per block, dense and
    /// system-indexed): `col_ncs_start + ncs_sys_idx * n_blks + blk`. A
    /// commissioning-dormant NCS keeps its column zeroed to `[0, 0]`, so the
    /// position is the entity's system index, not an active-local index.
    pub(crate) col_ncs_start: usize,
    /// Full NCS count (identical at every stage).
    pub(crate) n_ncs: usize,
    /// Start of pumping-flow columns (one per station per block, dense and
    /// system-indexed, block-major): `col_pumping_start + p_sys * n_blks + blk`. A
    /// dormant station keeps its column zeroed to `[0, 0]`; with `n_pumping == 0`
    /// the block is empty and `col_pumping_start == col_ncs_end`.
    pub(crate) col_pumping_start: usize,
    /// Full station count (identical at every stage); contributes `n_blks` columns
    /// each.
    pub(crate) n_pumping: usize,
    /// Full import-contract count (identical at every stage).
    pub(crate) n_contract_import: usize,
    /// Full export-contract count (identical at every stage).
    pub(crate) n_contract_export: usize,
    /// Column range for import-contract variables (one per import contract per
    /// block); empty `start..start` at `col_pumping_end` with no import contracts.
    pub(crate) contract_import: Range<usize>,
    /// Column range for export-contract variables (one per export contract per
    /// block); empty `start..start` at the import-block end with no export contracts.
    pub(crate) contract_export: Range<usize>,
}

/// Column and row ranges for the four operational-violation slack families
/// (below-min-outflow, above-max-outflow, below-min-turbine,
/// below-min-generation). The two flow families are sized `n_h * n_blks`
/// (non-empty only when `n_h > 0`); the two power families are sized
/// `n_cells * n_blks` (non-empty only when `n_cells > 0`) — a cell's own
/// min-turbine/min-generation floor is the sum of ITS OWN member groups, never
/// the plant's aggregate, so each cell gets its own row and its own slack
/// column. See the min-floor contract. Slack
/// columns follow the withdrawal slacks; constraint rows follow the
/// evaporation rows. Kept as one nested struct (not destructured) because the
/// column and row halves are allocated as two back-to-back `RangeCursor` runs
/// — see [`Self::new`].
pub(crate) struct OperViolationRanges {
    /// Column range for outflow-below-minimum slack (one per hydro per block).
    pub(crate) outflow_below_slack: Range<usize>,
    /// Column range for outflow-above-maximum slack (one per hydro per block).
    pub(crate) outflow_above_slack: Range<usize>,
    /// Column range for turbine-below-minimum slack (one per hydro CELL per block).
    pub(crate) turbine_below_slack: Range<usize>,
    /// Column range for generation-below-minimum slack (one per hydro CELL per block).
    pub(crate) generation_below_slack: Range<usize>,
    /// Row range for min-outflow constraints (one per hydro per block).
    pub(crate) min_outflow_rows: Range<usize>,
    /// Row range for max-outflow constraints (one per hydro per block).
    pub(crate) max_outflow_rows: Range<usize>,
    /// Row range for min-turbine constraints (one per hydro CELL per block).
    pub(crate) min_turbine_rows: Range<usize>,
    /// Row range for min-generation constraints (one per hydro CELL per block).
    pub(crate) min_generation_rows: Range<usize>,
}

impl OperViolationRanges {
    /// Allocate the four column families then the four row families,
    /// contiguously in that order: reordering these eight `alloc` calls would
    /// shift every downstream column/row, so `col`/`row` are threaded through
    /// and consumed in exactly this order. `n_op_hydro` sizes the two flow
    /// families; `n_op_cell` sizes the two power families — they diverge the
    /// moment any plant declares groups on more than one bus.
    fn new(
        col: &mut RangeCursor,
        row: &mut RangeCursor,
        n_op_hydro: usize,
        n_op_cell: usize,
    ) -> Self {
        Self {
            outflow_below_slack: col.alloc(n_op_hydro),
            outflow_above_slack: col.alloc(n_op_hydro),
            turbine_below_slack: col.alloc(n_op_cell),
            generation_below_slack: col.alloc(n_op_cell),
            min_outflow_rows: row.alloc(n_op_hydro),
            max_outflow_rows: row.alloc(n_op_hydro),
            min_turbine_rows: row.alloc(n_op_cell),
            min_generation_rows: row.alloc(n_op_cell),
        }
    }
}

/// Slack columns: inflow non-negativity, under/over-withdrawal, and the four
/// operational-violation slacks (nested via [`OperViolationRanges`], which also
/// carries their paired constraint rows — see that type's doc for why the
/// pairing is not split across this struct and [`ConstraintRows`]).
pub(crate) struct SlackColumns {
    /// Column range for inflow non-negativity slack (one per hydro, stage-level);
    /// empty `start..start` without the penalty or hydros. Stored first-class so
    /// the per-stage simulation geometry reads the stage-correct range — a single
    /// global stage-0 range would shift under a non-uniform block schedule.
    pub(crate) inflow_slack: Range<usize>,
    /// Column range for under-withdrawal slack (one per hydro); empty
    /// `start..start` with no hydros.
    pub(crate) withdrawal_slack_neg: Range<usize>,
    /// Column range for over-withdrawal slack (one per hydro).
    pub(crate) withdrawal_slack_pos: Range<usize>,
    /// The four operational-violation slack column ranges and their paired
    /// constraint-row ranges.
    pub(crate) oper_violation: OperViolationRanges,
}

/// Constraint row ranges shared by every stage's LP: z-inflow, water balance,
/// travel-time buckets, load balance, the FPHA/evaporation row cursor, and the
/// structural row-count scalars.
pub(crate) struct ConstraintRows {
    /// Water balance row family: `n_h` rows in parallel mode, `n_h * n_blks` in
    /// chronological mode (the `K` chained per-hydro rows), addressed through
    /// [`StageLayout::water_balance_row`].
    pub(crate) water_balance: BlockRowFamily,
    /// Row range for travel-time bucket definition rows: `b_d^out − b_{d+1}^in
    /// − deposit_d = 0`, one row per (plant, lag) bucket REACHABLE at this
    /// stage (`state.transit_bucket_column_order[slot]`'s lag within this stage's
    /// `per_stage_mask` cap for that plant — see [`Self::transit_bucket_row_pos`]);
    /// unlike `commit_in`'s active-plant sparseness, a lag beyond the cap gets
    /// no row at this stage — absent a boundary FCF the cap only shrinks toward
    /// the horizon end (Terminal credit deferred); with one present the
    /// terminal cap un-caps instead (Delivery-family right-boundary pricing).
    /// Placed immediately after
    /// [`Self::water_balance`], so `load_balance` and every row cursor after it
    /// shift by this stage's reachable count (`<= state.n_buckets`). Empty
    /// `start..start` when `state.n_buckets == 0` (the B==0 byte-identity
    /// anchor: `load_balance` collapses back onto `water_balance.end`).
    pub(crate) transit_bucket_definition: Range<usize>,
    /// For each GLOBAL bucket index (`state.transit_bucket_column_order`'s index),
    /// this stage's compact row position within [`Self::transit_bucket_definition`], or
    /// `None` when its lag is beyond this stage's reachable cap (no row; the
    /// matching deposit in [`super::entries`]'s arc-release fill is dropped
    /// there, not misdirected to another row). Length `state.n_buckets`.
    pub(crate) transit_bucket_row_pos: Vec<Option<usize>>,
    /// Load balance row family (one per bus per block), addressed through
    /// [`StageLayout::load_balance_row`].
    pub(crate) load_balance: BlockRowFamily,
    /// Row cursor at which the evaporation row block begins (`fpha_rows_end`),
    /// even when the FPHA block is empty.
    pub(crate) fpha_rows_end: usize,
    /// Start of generic constraint rows (one per active `(constraint, block)` pair),
    /// after operational-violation rows.
    pub(crate) row_generic_start: usize,
    /// Total row count.
    pub(crate) num_rows: usize,
    /// Generic constraint row count.
    pub(crate) n_generic_rows: usize,
}

/// Per-stage filling-phase row/column families: the `σ_fill` target (Filling
/// phase) and the soft `σ^{v-}` operating floor (Operating phase), each with
/// its paired hydro-index satellite vector.
pub(crate) struct FillingLayout {
    /// First per-stage `σ_fill`-target row (one per Filling-phase filling hydro);
    /// after the operational-violation rows, in the pre-cut region. Empty at every
    /// non-Filling stage. MUST stay strictly below `num_rows`: a row at index
    /// `>= num_rows` aliases the append-only cut rows (slot-identity warm-start
    /// matches cut rows from `num_rows`) and corrupts every cut.
    pub(crate) row_filling_target_start: usize,
    /// First `σ_fill` slack column (one per Filling-phase filling hydro); the
    /// second-to-last per-stage column family, after generic-slack and before
    /// `filled_min_storage_floor`. Empty for a non-filling system, leaving prior
    /// `col_*_start` and `num_cols` byte-identical.
    pub(crate) col_filling_target_start: usize,
    /// System hydro indices emitting a `σ_fill` target at this stage, ascending.
    /// Parallel to both the `filling_target` row and `σ_fill` column blocks: local
    /// index `i` → row `row_filling_target_start + i`, column
    /// `col_filling_target_start + i`.
    pub(crate) filling_target_hydro_indices: Vec<HydroSys>,
    /// First soft `σ^{v-}` operating-floor row (one per Operating-phase filling
    /// hydro); sibling to `filling_target` in the pre-cut region. Same
    /// `row >= num_rows` aliasing invariant as `row_filling_target_start`.
    pub(crate) row_filled_min_storage_floor_start: usize,
    /// First soft `σ^{v-}` slack column (one per Operating-phase filling hydro); the
    /// LAST per-stage column family, so its presence cannot shift any other family.
    /// Empty for a non-filling system, leaving other `col_*_start`/`num_cols`
    /// byte-identical.
    pub(crate) col_filled_min_storage_floor_start: usize,
    /// System hydro indices emitting a `σ^{v-}` floor at this stage, ascending.
    /// Parallel to both the `filled_min_storage_floor` row and column blocks. DISTINCT
    /// from `filling_target_hydro_indices` (`σ_fill`, Filling phase); the two
    /// never overlap (Operating vs Filling).
    pub(crate) filled_min_storage_floor_hydro_indices: Vec<HydroSys>,
}

/// Pre-computed column and row layout offsets for a single stage LP.
///
/// Owns the role-(b) geometry (per-stage equipment / slack / row ranges and the
/// entity counts that stride them) as its own fields, computed in
/// [`StageLayout::new`] anchored at the handle's
/// [`StateSpace::control_region_start`]. The stage-invariant role-(a) state
/// region is NOT duplicated here — it is read through the borrowed [`Self::state`]
/// handle. The control region begins at `state.control_region_start()`
/// (`theta + 1`), so the two regions meet there with no overlap.
pub(crate) struct StageLayout<'a> {
    /// Borrowed handle to the stage-invariant role-(a) state layout; the role-(a)
    /// accessors read through it rather than re-deriving offsets per stage. The
    /// dependency is one-directional (geometry → `StateSpace`), never the reverse.
    pub(crate) state: &'a StateSpace,
    /// In-study anticipated-ring column/row offsets (see [`AnticipatedLayout`]).
    pub(crate) anticipated: AnticipatedLayout,
    /// Equipment column ranges (see [`EquipmentColumns`]).
    pub(crate) equipment: EquipmentColumns,
    /// Slack columns, including the paired operational-violation rows (see
    /// [`SlackColumns`]).
    pub(crate) slack: SlackColumns,
    /// Constraint row ranges (see [`ConstraintRows`]).
    pub(crate) rows: ConstraintRows,
    /// Filling-phase row/column families (see [`FillingLayout`]).
    pub(crate) filling: FillingLayout,
    /// Total column count.
    pub(crate) num_cols: usize,
    /// This stage's block-hours owner; the water-balance noise/inflow scale.
    pub(crate) clock: BlockClock<'a>,
    /// Indices (into `ctx.hydros`) of hydros using FPHA at this stage.
    pub(crate) fpha_hydro_indices: Vec<HydroSys>,
    /// Inverse of `fpha_hydro_indices`: system hydro index → FPHA-local index,
    /// length `n_h` (`None` at non-FPHA hydros). Single owner of the reverse map,
    /// read by the matrix-fill helpers in place of rebuilding it per call.
    pub(crate) fpha_local_index: Vec<Option<FphaLocal>>,
    /// FPHA-local index → that plant's first cell's FPHA-cell-local index,
    /// length `n_fpha_hydros` (parallel to `fpha_hydro_indices`); the identity
    /// (`[0, 1, 2, ...]`) while every FPHA plant has one cell. Single owner of
    /// the FPHA-cell prefix sum, read by [`Self::fpha_local_first_cell`].
    pub(crate) fpha_cell_local_start: Vec<usize>,
    /// Hyperplane count per FPHA hydro at this stage.
    pub(crate) fpha_planes_per_hydro: Vec<usize>,
    /// Evaporation slots per evaporating hydro at this stage: single-owner result
    /// of [`evaporation_slot_count`] (`1` on a parallel stage, `n_blks` on a
    /// chronological one) — the stride every evaporation column/row family uses.
    pub(crate) n_evap_slots: usize,
    /// Indices (into `ctx.hydros`) of hydros with linearized evaporation at this stage.
    pub(crate) evap_hydro_indices: Vec<HydroSys>,
    /// Per-`(evaporation hydro, slot)` column/row indices, slot-major
    /// (`local * n_evap_slots + slot`), parallel to `evap_hydro_indices`.
    pub(crate) evap_indices: Vec<EvaporationIndices>,
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
/// for that plant. `column_order` groups contiguously by plant in the SAME
/// discovery order `per_stage_mask` indexes
/// ([`crate::bucket_topology::build_transit_bucket_topology`]), so a plant
/// transition in the scan advances the mask index. Returns the mapping and the
/// reachable count (`transit_bucket_definition`'s row length).
fn build_transit_bucket_row_pos(
    column_order: &[(usize, usize)],
    per_stage_mask: &[Vec<usize>],
    stage_idx: usize,
) -> (Vec<Option<usize>>, usize) {
    if column_order.is_empty() {
        // B==0 byte-identity anchor: no declared bucket, so no per-stage mask
        // entry is required (`per_stage_mask` may be empty in fixtures that
        // never build one).
        return (Vec::new(), 0);
    }
    let stage_mask = &per_stage_mask[stage_idx];
    let mut transit_bucket_row_pos = Vec::with_capacity(column_order.len());
    let mut plant_group = 0_usize;
    let mut prev_plant: Option<usize> = None;
    let mut n_reachable = 0_usize;
    for &(plant_idx, lag) in column_order {
        if prev_plant != Some(plant_idx) {
            if prev_plant.is_some() {
                plant_group += 1;
            }
            prev_plant = Some(plant_idx);
        }
        if lag <= stage_mask[plant_group] {
            transit_bucket_row_pos.push(Some(n_reachable));
            n_reachable += 1;
        } else {
            transit_bucket_row_pos.push(None);
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
/// `row_anticipated_state_out_def_start` owns it instead), beyond the
/// EXTENDED delivery calendar (`m >= state.delivery_stage_count(n_stages)`),
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
    n_stages: usize,
    stage_idx: usize,
) -> (Vec<Option<usize>>, usize) {
    let n_anticipated = state.n_anticipated;
    let mut row_pos = vec![None; n_anticipated * state.k_max];
    let mut n_reachable = 0_usize;
    for_each_live_commitment_slot(state, n_stages, stage_idx, |res, point| {
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
    n_stages: usize,
    stage_idx: usize,
    anticipated_windows: &[(Option<i32>, Option<i32>)],
    delivery_stage_ids: &[i32],
) -> (Vec<Option<usize>>, usize) {
    let n_anticipated = state.n_anticipated;
    let k_max = state.k_max;
    if n_anticipated == 0 || k_max == 0 {
        return (Vec::new(), 0);
    }
    let n_delivery = state.delivery_stage_count(n_stages);
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
            state,
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

fn hydro_phase(hydro: &Hydro, stage_id: i32) -> Phase {
    filling_phase(
        hydro.filling.as_ref(),
        hydro.entry_stage_id,
        hydro.exit_stage_id,
        stage_id,
    )
}

/// Collect the FPHA hydro indices and per-hydro plane counts for this stage.
///
/// A filling hydro is dropped from the FPHA set in `PreFilling` **or** `Filling`:
/// a non-operating plant has zero productivity, and the operating-range hyperplane
/// fit is invalid below `min_storage` where a filling reservoir sits. Because the
/// generation column block is densely packed by FPHA-local index, dropping a hydro
/// here removes its column entirely — no orphaned `[0, max]` column for an
/// unconstrained solve to exploit. `stage_id` is the study `stage.id`, not the
/// stage index ([`filling_phase`] keys on the commissioning id). A
/// commissioning-dormant non-filling hydro is `PreFilling` and is dropped here too;
/// a non-filling hydro with no window is `Operating` at every stage (parity-neutral).
fn identify_fpha_hydros(
    ctx: &TemplateBuildCtx<'_>,
    stage_idx: usize,
    stage_id: i32,
) -> (Vec<HydroSys>, Vec<usize>) {
    let mut fpha_hydro_indices: Vec<HydroSys> = Vec::new();
    let mut fpha_planes_per_hydro: Vec<usize> = Vec::new();
    for h_idx in 0..ctx.hydros.len() {
        let hydro = &ctx.hydros[h_idx];
        if matches!(
            hydro_phase(hydro, stage_id),
            Phase::PreFilling | Phase::Filling
        ) {
            continue;
        }
        if let ResolvedProductionModel::Fpha { planes, .. } =
            ctx.production_models.model(h_idx, stage_idx)
        {
            fpha_hydro_indices.push(HydroSys::new(h_idx));
            fpha_planes_per_hydro.push(planes.len());
        }
    }
    (fpha_hydro_indices, fpha_planes_per_hydro)
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
/// [`filling_phase`] (`filled_min_storage_floor` takes over at/after `entry`). A
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
fn contract_direction_counts(contracts: &[EnergyContract]) -> (usize, usize) {
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

impl<'a> StageLayout<'a> {
    #[expect(
        clippy::too_many_lines,
        clippy::similar_names,
        reason = "each range starts at the previous range's end, so the offset chain stays one linear read beside the established state/stage names"
    )]
    pub(crate) fn new(
        ctx: &TemplateBuildCtx<'_>,
        state: &'a StateSpace,
        stage: &'a Stage,
        stage_idx: usize,
    ) -> Self {
        let clock = BlockClock::new(stage);
        let n_blks = clock.n_blks();
        let n_h = state.hydro_count;

        let (fpha_hydro_indices, fpha_planes_per_hydro) =
            identify_fpha_hydros(ctx, stage_idx, stage.id);
        let evap_hydro_indices = identify_evap_hydros(ctx, stage.id);
        let filling_target_hydro_indices = identify_filling_target_hydros(ctx, stage.id);
        let filled_min_storage_floor_hydro_indices =
            identify_filled_min_storage_floor_hydros(ctx, stage.id);

        let mut fpha_local_index: Vec<Option<FphaLocal>> = vec![None; n_h];
        for (local_idx, &h) in fpha_hydro_indices.iter().enumerate() {
            fpha_local_index[h.get()] = Some(FphaLocal::new(local_idx));
        }

        // FPHA-cell-local start per FPHA-local plant (plant-major, matching
        // `fpha_hydro_indices`'s own order): the cumulative cell count over
        // preceding FPHA plants. `n_fpha_cells` (the running total) sizes the
        // generation family below.
        let mut fpha_cell_local_start: Vec<usize> = Vec::with_capacity(fpha_hydro_indices.len());
        let mut n_fpha_cells = 0_usize;
        let mut total_fpha_rows = 0_usize;
        for (local_idx, &h) in fpha_hydro_indices.iter().enumerate() {
            fpha_cell_local_start.push(n_fpha_cells);
            let n_cells_h = ctx.hydro_cell_index.cells_of(h).len();
            n_fpha_cells += n_cells_h;
            total_fpha_rows += n_cells_h * fpha_planes_per_hydro[local_idx];
        }

        let max_deficit_segments = ctx.study_dims.max_deficit_segments;

        // ── Role-(b) equipment column ranges ─────────────────────────────────
        // Anchored at the handle's `control_region_start()` (the role-(a)/role-(b)
        // seam); `col` allocates every family through `RangeCursor::alloc`, strided
        // by THIS stage's `n_blks` (the per-stage authority over the stage-0 global
        // stride). Adjacency between consecutive families is structural, never a
        // hand-copied `.end`.
        let n_interior = match stage.block_mode {
            BlockMode::Chronological => n_blks.saturating_sub(1),
            BlockMode::Parallel => 0,
        };
        let mut col = RangeCursor::new(state.control_region_start());
        let storage_internal_start = col.alloc(n_h * n_interior).start;
        let n_cells = ctx.hydro_cell_index.n_cells();
        let turbine = col.alloc(n_cells * n_blks);
        let spillage = col.alloc(n_h * n_blks);
        let diversion = col.alloc(n_h * n_blks);
        let thermal = col.alloc(ctx.thermals.len() * n_blks);
        let thermal_end = thermal.end;
        col.alloc(state.n_anticipated);
        let line_fwd = col.alloc(ctx.lines.len() * n_blks);
        let line_rev = col.alloc(ctx.lines.len() * n_blks);
        let deficit = col.alloc(ctx.buses.len() * max_deficit_segments * n_blks);
        let excess = col.alloc(ctx.buses.len() * n_blks);

        let has_inflow_penalty = ctx.study_dims.has_inflow_penalty;
        let inflow_slack = col.alloc(if has_inflow_penalty { n_h } else { 0 });

        // `generation_col_start` is the empty-block cursor `col_generation_start`
        // reads; `col.pos()` already carries the correct value whether or not the
        // inflow-penalty family above was empty. Sized by FPHA CELL, not FPHA
        // plant: `n_fpha_cells` is the identity (`== fpha_hydro_indices.len()`)
        // while every FPHA plant has one cell.
        let generation_col_start = col.pos();
        let generation = col.alloc(n_fpha_cells * n_blks);

        // `evap_col_start` is the empty-block cursor `col_evap_start` reads; one
        // `EVAP_COLS_PER_HYDRO` triple per `(evap hydro, slot)`, strided by
        // `n_evap_slots` (`evaporation_slot_count`).
        let n_evap_hydros = evap_hydro_indices.len();
        let n_evap_slots = evaporation_slot_count(stage.block_mode, n_blks);
        let evap_col_start = col.pos();
        col.alloc(n_evap_hydros * n_evap_slots * EVAP_COLS_PER_HYDRO);

        // ── Role-(b) constraint row ranges ───────────────────────────────────
        // The builder's own rows start immediately after `StateSpace::z_inflow_rows()`,
        // the sole owner of that leading row range. `row` allocates every family
        // through `RangeCursor::alloc`, mirroring `col` above.
        let mut row = RangeCursor::new(state.z_inflow_rows().end);
        let water_balance = match stage.block_mode {
            BlockMode::Chronological => BlockRowFamily::per_block(row.alloc(n_h * n_blks)),
            BlockMode::Parallel => BlockRowFamily::one_per_entity(row.alloc(n_h)),
        };
        // Sized from this stage's reachable count, not the stage-invariant
        // `state.n_buckets`: `build_transit_bucket_row_pos` masks a lag beyond
        // `ctx.topology.per_stage_mask[stage_idx]`'s per-plant cap out of the row
        // range entirely — the cap itself is `build_transit_bucket_topology`'s,
        // gated on `boundary_present`.
        let (transit_bucket_row_pos, n_transit_bucket_rows) = build_transit_bucket_row_pos(
            &state.transit_bucket_column_order,
            &ctx.topology.per_stage_mask,
            stage_idx,
        );
        let transit_bucket_definition = row.alloc(n_transit_bucket_rows);
        let load_balance = BlockRowFamily::per_block(row.alloc(ctx.buses.len() * n_blks));

        // Only the end cursor is kept here (the per-hydro ranges live on
        // `StageData.indexer`); `fpha_rows_end` is the evaporation-row start even
        // when the FPHA block is empty. `total_fpha_rows` sums `n_cells(plant) *
        // n_planes(plant)`, not `Σ n_planes(plant)`: each cell owns its own
        // `n_blks * n_planes` row block (`for_each_fpha_plane`'s per-cell advance).
        // The plant-only sum undersizes a multi-bus plant's row range, aliasing
        // rows across cells.
        let fpha_rows_end = row.alloc(n_blks * total_fpha_rows).end;

        // One row per `(evap hydro, slot)`, so the row block grows by `n_evap_slots`
        // — the cursor chain below MUST stay in lockstep or every downstream row shifts.
        let evap_indices =
            build_evap_indices(n_evap_hydros, n_evap_slots, evap_col_start, fpha_rows_end);
        row.alloc(n_evap_hydros * n_evap_slots);

        // Withdrawal slacks + the four operational-violation slack families (after
        // the evaporation columns) and their matching rows (after the evaporation
        // rows). `n_op_hydro`/`n_op_cell` are `0` when `n_h`/`n_cells == 0`, so
        // `alloc(0)` collapses every family onto the post-equipment cursor with no
        // branch.
        let withdrawal_slack_neg = col.alloc(n_h);
        let withdrawal_slack_pos = col.alloc(n_h);
        let n_op_hydro = n_h * n_blks;
        let n_op_cell = n_cells * n_blks;
        let oper_violation = OperViolationRanges::new(&mut col, &mut row, n_op_hydro, n_op_cell);

        // NCS follows the last operational-violation slack family; `col.pos()`
        // already equals the post-equipment cursor when `n_h == 0`, so no fallback
        // branch is needed.
        let n_ncs = ctx.non_controllable_sources.len();
        let col_ncs_start = col.alloc(n_ncs * n_blks).start;

        // σ_fill then σ^{v-} rows, in the pre-cut region after the
        // operational-violation rows. Both MUST stay strictly below `num_rows`: a
        // row at index `>= num_rows` aliases the append-only cut rows (slot-identity
        // warm-start matches cut rows from `num_rows`) and corrupts every cut.
        let n_filling_target_rows = filling_target_hydro_indices.len();
        let row_filling_target_start = row.alloc(n_filling_target_rows).start;
        let n_filled_min_storage_floor_rows = filled_min_storage_floor_hydro_indices.len();
        let row_filled_min_storage_floor_start = row.alloc(n_filled_min_storage_floor_rows).start;

        // Commitment-MATURITY rows: one per GENUINELY anticipated plant whose
        // delivery matures this stage (`build_anticipated_fishing_row_pos`) —
        // a `K = 0` self-delivery excludes a plant's row this stage, so the
        // row family is sparse like the deposit family below, not the dense
        // `state.n_anticipated` count.
        let n_stages = ctx.resolved.bounds.n_stages();
        let (anticipated_fishing_row_pos, n_anticipated_fishing_rows) =
            build_anticipated_fishing_row_pos(state, n_stages, stage_idx);
        let row_anticipated_fishing_start = row.alloc(n_anticipated_fishing_rows).start;

        // Anticipated-state-out (latch/deposit) definition rows
        // (`build_anticipated_decision_row_pos`).
        let (anticipated_decision_row_pos, n_anticipated_state_out_def_rows) =
            build_anticipated_decision_row_pos(
                state,
                n_stages,
                stage_idx,
                ctx.study_dims.anticipated_plants.windows(),
                ctx.time_value.delivery_stage_ids(),
            );
        let row_anticipated_state_out_def_start = row.alloc(n_anticipated_state_out_def_rows).start;

        // Future-window commitment-carry rows, modular-addressed
        // (`build_anticipated_slot_row_pos`) — strictly future, not-yet-due
        // deliveries only; the commitment maturing this stage is fished by
        // the maturity row above instead.
        let (anticipated_slot_row_pos, n_anticipated_slot_definition_rows) =
            build_anticipated_slot_row_pos(state, n_stages, stage_idx);
        let row_anticipated_slot_definition_start =
            row.alloc(n_anticipated_slot_definition_rows).start;

        // Peeked before `generic` below is computed: the generic row block's
        // length depends on `col_generic_slack_start` (the column axis), but its
        // own start does not depend on that length.
        let row_generic_start = row.pos();

        let n_pumping = ctx.pumping_stations.len();
        let col_pumping_start = col.alloc(n_pumping * n_blks).start;

        // Import then export contract block; both empty leaves
        // col_generic_slack_start at col_pumping_end (parity-neutral).
        let (n_contract_import, n_contract_export) = contract_direction_counts(ctx.contracts);
        let contract_import = col.alloc(n_contract_import * n_blks);
        let contract_export = col.alloc(n_contract_export * n_blks);

        let col_generic_slack_start = col.pos();
        let generic = enumerate_generic_constraint_rows(
            ctx,
            stage,
            stage_idx,
            n_blks,
            col_generic_slack_start,
        );
        col.alloc(generic.n_generic_slack_cols);

        // σ_fill then σ^{v-} are the last two per-stage column families; σ^{v-}
        // last so its presence cannot shift any other family's start.
        let col_filling_target_start = col.alloc(filling_target_hydro_indices.len()).start;
        let col_filled_min_storage_floor_start = col
            .alloc(filled_min_storage_floor_hydro_indices.len())
            .start;
        let num_cols = col.pos();
        row.alloc(generic.n_generic_rows);
        let num_rows = row.pos();

        let anticipated = AnticipatedLayout {
            col_anticipated_decision_start: thermal_end,
            row_anticipated_state_out_def_start,
            n_anticipated_state_out_def_rows,
            anticipated_decision_row_pos,
            row_anticipated_fishing_start,
            n_anticipated_fishing_rows,
            anticipated_fishing_row_pos,
            row_anticipated_slot_definition_start,
            n_anticipated_slot_definition_rows,
            anticipated_slot_row_pos,
        };

        let equipment = EquipmentColumns {
            storage_internal_start,
            turbine,
            spillage,
            diversion,
            thermal,
            line_fwd,
            line_rev,
            deficit,
            max_deficit_segments,
            excess,
            generation_col_start,
            generation,
            evap_col_start,
            col_ncs_start,
            n_ncs,
            col_pumping_start,
            n_pumping,
            n_contract_import,
            n_contract_export,
            contract_import,
            contract_export,
        };
        let slack = SlackColumns {
            inflow_slack,
            withdrawal_slack_neg,
            withdrawal_slack_pos,
            oper_violation,
        };
        let rows = ConstraintRows {
            water_balance,
            transit_bucket_definition,
            transit_bucket_row_pos,
            load_balance,
            fpha_rows_end,
            row_generic_start,
            num_rows,
            n_generic_rows: generic.n_generic_rows,
        };
        let filling = FillingLayout {
            row_filling_target_start,
            col_filling_target_start,
            filling_target_hydro_indices,
            row_filled_min_storage_floor_start,
            col_filled_min_storage_floor_start,
            filled_min_storage_floor_hydro_indices,
        };

        Self {
            state,
            anticipated,
            equipment,
            slack,
            rows,
            filling,
            num_cols,
            clock,
            fpha_hydro_indices,
            fpha_local_index,
            fpha_cell_local_start,
            fpha_planes_per_hydro,
            n_evap_slots,
            evap_hydro_indices,
            evap_indices,
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
    /// Turbine-flow column for cell `c`, block `blk`.
    #[inline]
    pub(crate) fn turbine_col(&self, c: HydroCell, blk: BlockIdx) -> usize {
        self.block_flat(self.equipment.turbine.start, c.get(), blk)
    }

    /// Spillage column for hydro `h`, block `blk`.
    #[inline]
    pub(crate) fn spillage_col(&self, h: HydroSys, blk: BlockIdx) -> usize {
        self.block_flat(self.equipment.spillage.start, h.get(), blk)
    }

    /// Diversion-flow column for hydro `h`, block `blk`.
    #[inline]
    pub(crate) fn diversion_col(&self, h: HydroSys, blk: BlockIdx) -> usize {
        self.block_flat(self.equipment.diversion.start, h.get(), blk)
    }

    /// FPHA generation column for FPHA-cell-local index `c`, block `blk`.
    #[inline]
    pub(crate) fn generation_col(&self, c: FphaCellLocal, blk: BlockIdx) -> usize {
        self.block_flat(self.equipment.generation_col_start, c.get(), blk)
    }

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

    /// Forward line-flow column for line `l`, block `blk`.
    #[inline]
    pub(crate) fn line_fwd_col(&self, l: LineSys, blk: BlockIdx) -> usize {
        self.block_flat(self.equipment.line_fwd.start, l.get(), blk)
    }

    /// Reverse line-flow column for line `l`, block `blk`.
    #[inline]
    pub(crate) fn line_rev_col(&self, l: LineSys, blk: BlockIdx) -> usize {
        self.block_flat(self.equipment.line_rev.start, l.get(), blk)
    }

    /// Outflow-below-minimum slack column for hydro `h`, block `blk`.
    #[inline]
    pub(crate) fn outflow_below_col(&self, h: HydroSys, blk: BlockIdx) -> usize {
        self.block_flat(
            self.slack.oper_violation.outflow_below_slack.start,
            h.get(),
            blk,
        )
    }

    /// Outflow-above-maximum slack column for hydro `h`, block `blk`.
    #[inline]
    pub(crate) fn outflow_above_col(&self, h: HydroSys, blk: BlockIdx) -> usize {
        self.block_flat(
            self.slack.oper_violation.outflow_above_slack.start,
            h.get(),
            blk,
        )
    }

    /// Turbine-below-minimum slack column for cell `c`, block `blk`.
    #[inline]
    pub(crate) fn turbine_below_col(&self, c: HydroCell, blk: BlockIdx) -> usize {
        self.block_flat(
            self.slack.oper_violation.turbine_below_slack.start,
            c.get(),
            blk,
        )
    }

    /// Generation-below-minimum slack column for cell `c`, block `blk`.
    #[inline]
    pub(crate) fn generation_below_col(&self, c: HydroCell, blk: BlockIdx) -> usize {
        self.block_flat(
            self.slack.oper_violation.generation_below_slack.start,
            c.get(),
            blk,
        )
    }

    #[inline]
    pub(crate) fn min_outflow_row(&self, h: HydroSys, blk: BlockIdx) -> usize {
        self.block_flat(
            self.slack.oper_violation.min_outflow_rows.start,
            h.get(),
            blk,
        )
    }

    #[inline]
    pub(crate) fn max_outflow_row(&self, h: HydroSys, blk: BlockIdx) -> usize {
        self.block_flat(
            self.slack.oper_violation.max_outflow_rows.start,
            h.get(),
            blk,
        )
    }

    #[inline]
    pub(crate) fn min_turbine_row(&self, c: HydroCell, blk: BlockIdx) -> usize {
        self.block_flat(
            self.slack.oper_violation.min_turbine_rows.start,
            c.get(),
            blk,
        )
    }

    #[inline]
    pub(crate) fn min_generation_row(&self, c: HydroCell, blk: BlockIdx) -> usize {
        self.block_flat(
            self.slack.oper_violation.min_generation_rows.start,
            c.get(),
            blk,
        )
    }

    /// Hydro `h`'s inflow-penalty slack column.
    #[inline]
    pub(crate) fn inflow_slack_col(&self, h: HydroSys) -> usize {
        entity_flat(&self.slack.inflow_slack, h.get())
    }

    /// Hydro `h`'s below-withdrawal-target slack column.
    #[inline]
    pub(crate) fn withdrawal_slack_neg_col(&self, h: HydroSys) -> usize {
        entity_flat(&self.slack.withdrawal_slack_neg, h.get())
    }

    /// Hydro `h`'s above-withdrawal-target slack column.
    #[inline]
    pub(crate) fn withdrawal_slack_pos_col(&self, h: HydroSys) -> usize {
        entity_flat(&self.slack.withdrawal_slack_pos, h.get())
    }

    /// Anticipated-local `local`'s ring decision column.
    #[inline]
    pub(crate) fn anticipated_decision_col(&self, local: AnticipatedLocal) -> usize {
        entity_flat(&self.anticipated_decision(), local.get())
    }

    /// Anticipated-local `local`'s commitment-maturity row, or `None` when no
    /// delivery matures this stage (including a `K = 0` self-delivery).
    #[inline]
    pub(crate) fn anticipated_fishing_row(&self, local: AnticipatedLocal) -> Option<usize> {
        position_table_row(
            self.anticipated.row_anticipated_fishing_start,
            &self.anticipated.anticipated_fishing_row_pos,
            local.get(),
        )
    }

    /// Anticipated-local `local`'s deposit-definition row, or `None` when the
    /// plant has no genuine, active decision this stage.
    #[inline]
    pub(crate) fn anticipated_state_out_def_row(&self, local: AnticipatedLocal) -> Option<usize> {
        position_table_row(
            self.anticipated.row_anticipated_state_out_def_start,
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

    /// Filling-target-local `local`'s `σ_fill` slack column.
    #[inline]
    pub(crate) fn filling_target_slack_col(&self, local: FillingTargetLocal) -> usize {
        entity_flat(&self.filling_target_col(), local.get())
    }

    /// Floor-local `local`'s `σ^{v-}` operating-floor slack column.
    #[inline]
    pub(crate) fn filled_min_storage_floor_slack_col(&self, local: FloorLocal) -> usize {
        entity_flat(&self.filled_min_storage_floor_col(), local.get())
    }

    /// NCS entity `ncs_sys`'s generation column for block `blk`.
    #[inline]
    pub(crate) fn ncs_generation_col(&self, ncs_sys: NcsSys, blk: BlockIdx) -> usize {
        self.block_flat(self.equipment.col_ncs_start, ncs_sys.get(), blk)
    }

    /// Pumping station `pumping_sys`'s flow column for block `blk`.
    #[inline]
    pub(crate) fn pumping_flow_col(&self, pumping_sys: PumpingSys, blk: BlockIdx) -> usize {
        self.block_flat(self.equipment.col_pumping_start, pumping_sys.get(), blk)
    }

    /// `contract_type`'s contract column at per-direction slot `family_slot`
    /// (from [`contract_family_slot`](crate::generic_constraints::contract_family_slot))
    /// for block `blk`.
    #[inline]
    pub(crate) fn contract_col(
        &self,
        contract_type: ContractType,
        family_slot: usize,
        blk: BlockIdx,
    ) -> usize {
        let family = match contract_type {
            ContractType::Import => &self.equipment.contract_import,
            ContractType::Export => &self.equipment.contract_export,
        };
        self.block_flat(family.start, family_slot, blk)
    }

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
        self.block_grid()
            .deficit(self.equipment.deficit.start, bus.get(), seg_idx, blk)
    }

    #[inline]
    pub(crate) fn thermal_col(&self, t: ThermalSys, blk: BlockIdx) -> usize {
        self.block_flat(self.equipment.thermal.start, t.get(), blk)
    }

    #[inline]
    pub(crate) fn excess_col(&self, bus: BusSys, blk: BlockIdx) -> usize {
        self.block_flat(self.equipment.excess.start, bus.get(), blk)
    }

    /// The [`StorageBoundaryGrid`] address primitive for this stage's LP,
    /// carrying its interior anchor.
    #[inline]
    #[must_use]
    pub(crate) fn storage_boundary_grid(&self) -> StorageBoundaryGrid {
        StorageBoundaryGrid::new(self.equipment.storage_internal_start, self.clock.n_blks())
    }

    /// Storage column at chronological `boundary` for hydro `h`; delegates to
    /// [`StorageBoundaryGrid::col`], the single owner of the endpoints-vs-interior
    /// split. At `n_blks = 1` only the two endpoints resolve (no interior).
    #[inline]
    pub(crate) fn block_storage_col(&self, h: HydroSys, boundary: Boundary) -> usize {
        self.storage_boundary_grid().col(self.state, h, boundary)
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

    /// First FPHA row; the FPHA block follows the load-balance rows, so this is
    /// the load-balance end cursor — reads `self.rows.load_balance.end()`.
    #[inline]
    #[must_use]
    pub(crate) fn row_fpha_start(&self) -> usize {
        self.rows.load_balance.end()
    }

    /// Start of evaporation constraint rows, one per `(evap hydro, slot)`; see
    /// [`Self::evap_row`]. The evaporation row block follows the FPHA rows even
    /// when empty — reads `self.rows.fpha_rows_end`.
    #[inline]
    #[must_use]
    pub(crate) fn row_evap_start(&self) -> usize {
        self.rows.fpha_rows_end
    }

    /// Evaporation-equality row for `(evap hydro local, slot)`, slot-major over
    /// [`Self::row_evap_start`] — the row-side sibling of [`Self::evap_flow_col`].
    #[inline]
    #[must_use]
    pub(crate) fn evap_row(&self, local: EvapLocal, slot: BlockIdx) -> usize {
        self.row_evap_start() + evap_slot_flat(local.get(), slot.get(), self.n_evap_slots)
    }

    /// Filling-target-local `local`'s soft `σ_fill` row, over [`Self::filling_target`].
    #[inline]
    #[must_use]
    pub(crate) fn filling_target_row(&self, local: FillingTargetLocal) -> usize {
        entity_flat(&self.filling_target(), local.get())
    }

    /// Floor-local `local`'s soft `σ^{v-}` operating-floor row, over
    /// [`Self::filled_min_storage_floor`].
    #[inline]
    #[must_use]
    pub(crate) fn filled_min_storage_floor_row(&self, local: FloorLocal) -> usize {
        entity_flat(&self.filled_min_storage_floor(), local.get())
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

    /// Hydro `h`'s water-balance row for block `blk`, striding by `self.clock.n_blks()`
    /// per [`BlockRowFamily::row`]: its own block row in chronological mode, its
    /// single stage row in parallel mode (every block collapses to that row).
    #[inline]
    #[must_use]
    pub(crate) fn water_balance_row(&self, h: HydroSys, blk: BlockIdx) -> usize {
        self.rows
            .water_balance
            .row(h.get(), blk, self.clock.n_blks())
    }

    /// Bus `bus`'s load-balance row for block `blk`, striding by `self.clock.n_blks()`.
    #[inline]
    #[must_use]
    pub(crate) fn load_balance_row(&self, bus: BusSys, blk: BlockIdx) -> usize {
        self.rows
            .load_balance
            .row(bus.get(), blk, self.clock.n_blks())
    }

    /// Hydro `h`'s z-inflow definition row.
    #[inline]
    #[must_use]
    pub(crate) fn z_inflow_row(&self, h: HydroSys) -> usize {
        self.state.z_inflow_row(h)
    }

    // ── Range accessors mirrored onto `StageGeometry` (own fields) ──────────────
    // `StageLayout::new` only ever needs each family's *length* (to derive the
    // next family's start), never its full range, so these are the sole place the
    // `start..start + len` arithmetic is expressed; `Self::geometry` is the only
    // consumer.

    /// Per-stage `σ_fill`-target row range: empty `start..start` (not `0..0`) at
    /// every non-Filling stage.
    #[inline]
    #[must_use]
    pub(crate) fn filling_target(&self) -> Range<usize> {
        self.filling.row_filling_target_start
            ..self.filling.row_filling_target_start
                + self.filling.filling_target_hydro_indices.len()
    }

    /// Per-stage `σ_fill`-target slack column range, parallel to
    /// [`Self::filling_target`].
    #[inline]
    #[must_use]
    pub(crate) fn filling_target_col(&self) -> Range<usize> {
        self.filling.col_filling_target_start
            ..self.filling.col_filling_target_start
                + self.filling.filling_target_hydro_indices.len()
    }

    /// Soft `σ^{v-}` operating-floor row range: empty `start..start` (not `0..0`)
    /// at every non-operating-filling stage.
    #[inline]
    #[must_use]
    pub(crate) fn filled_min_storage_floor(&self) -> Range<usize> {
        self.filling.row_filled_min_storage_floor_start
            ..self.filling.row_filled_min_storage_floor_start
                + self.filling.filled_min_storage_floor_hydro_indices.len()
    }

    /// Soft `σ^{v-}` operating-floor slack column range, parallel to
    /// [`Self::filled_min_storage_floor`].
    #[inline]
    #[must_use]
    pub(crate) fn filled_min_storage_floor_col(&self) -> Range<usize> {
        self.filling.col_filled_min_storage_floor_start
            ..self.filling.col_filled_min_storage_floor_start
                + self.filling.filled_min_storage_floor_hydro_indices.len()
    }

    /// Anticipated-decision column range (one per anticipated thermal,
    /// stage-level): `col_anticipated_decision_start .. + n_anticipated`. `0..0`
    /// (not `col_anticipated_decision_start..col_anticipated_decision_start`) when
    /// `n_anticipated == 0` — the empty-case value a byte-identity oracle test
    /// pins; do not align this to the `start..start` convention the sibling
    /// filling-family accessors use.
    #[inline]
    #[must_use]
    pub(crate) fn anticipated_decision(&self) -> Range<usize> {
        if self.state.n_anticipated > 0 {
            let s = self.anticipated.col_anticipated_decision_start;
            s..s + self.state.n_anticipated
        } else {
            0..0
        }
    }

    /// Owned per-stage equipment-geometry snapshot: every field is a clone or
    /// range accessor of `self`, so `StageLayout` alone owns each family's
    /// start/end arithmetic. Must stay OWNED — the result is cloned into
    /// `StageTemplates.geometry_per_stage`, which outlives this `StageLayout`
    /// (rebuilt per MPI rank, never serialized).
    #[must_use]
    pub(crate) fn geometry(&self, block_mode: BlockMode) -> StageGeometry {
        debug_assert_eq!(
            self.rows.water_balance.rows_per_entity(self.clock.n_blks()),
            match block_mode {
                BlockMode::Parallel => 1,
                BlockMode::Chronological => self.clock.n_blks(),
            }
        );
        StageGeometry {
            turbine: self.equipment.turbine.clone(),
            spillage: self.equipment.spillage.clone(),
            diversion: self.equipment.diversion.clone(),
            thermal: self.equipment.thermal.clone(),
            anticipated_decision: self.anticipated_decision(),
            line_fwd: self.equipment.line_fwd.clone(),
            line_rev: self.equipment.line_rev.clone(),
            deficit: self.equipment.deficit.clone(),
            excess: self.equipment.excess.clone(),
            generation: self.equipment.generation.clone(),
            ncs_generation: self.equipment.col_ncs_start
                ..self.equipment.col_ncs_start + self.equipment.n_ncs * self.clock.n_blks(),
            pumping_flow: self.equipment.col_pumping_start
                ..self.equipment.col_pumping_start + self.equipment.n_pumping * self.clock.n_blks(),
            evap_indices: self.evap_indices.clone(),
            inflow_slack: self.slack.inflow_slack.clone(),
            withdrawal_slack_neg: self.slack.withdrawal_slack_neg.clone(),
            withdrawal_slack_pos: self.slack.withdrawal_slack_pos.clone(),
            outflow_below_slack: self.slack.oper_violation.outflow_below_slack.clone(),
            outflow_above_slack: self.slack.oper_violation.outflow_above_slack.clone(),
            turbine_below_slack: self.slack.oper_violation.turbine_below_slack.clone(),
            generation_below_slack: self.slack.oper_violation.generation_below_slack.clone(),
            contract_import: self.equipment.contract_import.clone(),
            contract_export: self.equipment.contract_export.clone(),
            water_balance: self.rows.water_balance,
            load_balance: self.rows.load_balance,
            fpha: self.row_fpha_start()..self.rows.fpha_rows_end,
            filling_target: self.filling_target(),
            filling_target_col: self.filling_target_col(),
            filled_min_storage_floor: self.filled_min_storage_floor(),
            filled_min_storage_floor_col: self.filled_min_storage_floor_col(),
            n_blks: self.clock.n_blks(),
            storage_internal_start: self.equipment.storage_internal_start,
            block_mode,
            fpha_hydro_indices: self.fpha_hydro_indices.clone(),
            evap_hydro_indices: self.evap_hydro_indices.clone(),
            filling_target_hydro_indices: self.filling.filling_target_hydro_indices.clone(),
            filled_min_storage_floor_hydro_indices: self
                .filling
                .filled_min_storage_floor_hydro_indices
                .clone(),
        }
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
