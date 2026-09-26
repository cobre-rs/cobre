use std::collections::{BTreeMap, HashMap, HashSet};
use std::ops::Range;

use cobre_core::scenario::{LoadModel, SamplingScheme};
use cobre_core::{BlockMode, ContractType, EntityId, Hydro, ResolvedBounds, Stage, System};
use cobre_io::StageIdResolver;
use cobre_solver::StageTemplate;
use cobre_stochastic::normal::precompute::PrecomputedNormal;
use cobre_stochastic::par::precompute::PrecomputedPar;

use crate::block_clock::BlockClock;
#[cfg(any(test, feature = "test-support"))]
use crate::error::SddpError;
use crate::hydro_models::{EvaporationModelSet, ProductionModelSet, ResolvedProductionModel};
use crate::inflow_method::InflowNonNegativityMethod;
use crate::lead_time::{AnticipatedResolution, SpreadResolution};
use crate::resolved_parameters::ResolvedParameters;
use crate::time_value::TimeValue;

use super::layout::{ResolvedTables, StageLayout, TemplateBuildCtx};
use super::{GenericConstraintRowEntry, StateBox, columns, entries, rows, scaling};
use crate::lp::indexer::{
    AnticipatedPlants, BlockGrid, BlockIdx, BlockRowFamily, Boundary, BusSys, EvaporationIndices,
    HydroCellIndex, HydroSys, StateSpace, StorageBoundaryGrid,
};
#[cfg(any(test, feature = "test-support"))]
use crate::setup::bucket_topology::build_transit_bucket_topology;
#[cfg(any(test, feature = "test-support"))]
use crate::setup::resolve_state_layout;

#[cfg(any(test, feature = "test-support"))]
pub(crate) mod canonical;

/// Outcome of [`build_stage_templates`]: one [`StageTemplate`] per study stage
/// plus the per-stage offsets and counts the forward/backward/simulation passes
/// need. The per-stage `Vec`s are parallel — index `s` of each refers to stage `s`.
#[derive(Debug, Clone)]
pub struct StageTemplates {
    /// One structural LP template per study stage, in stage order.
    pub templates: Vec<StageTemplate>,
    /// Per-stage admissible box for every outgoing state dimension, populated by
    /// `postprocess_templates` after scaling. Length equals `templates.len()`.
    pub(crate) state_boxes: Vec<StateBox>,
    /// Per-stage time-conversion factor `ζ = total_hours * M3S_TO_HM3`, length
    /// `templates.len()`: converts an m³/s flow rate to hm³ over the stage
    /// (`volume_hm3 = flow_m3s * zeta_per_stage[stage]`), the coefficient a
    /// parallel-mode water row's `z_inflow` coupling applies
    /// (`push_z_inflow_coupling` in `lp/builder/entries.rs`).
    pub zeta_per_stage: Vec<f64>,
    /// Per-stage block durations in hours (`block_hours_per_stage[stage]` is length
    /// `n_blocks`). Converts load-balance duals $/MW → $/`MWh`:
    /// `spot_price = dual / block_hours`.
    pub block_hours_per_stage: Vec<Vec<f64>>,
    /// Number of hydro plants (N).
    pub n_hydros: usize,
    /// Resolved objective cost-scale factor (`modeling.cost_scale_factor`,
    /// [`ResolvedParameters::cost_scale_factor`]). Every non-theta objective
    /// coefficient was divided by this at template build time; cost-domain
    /// reporting boundaries multiply back by it.
    pub cost_scale_factor: f64,
    /// Per-stage row index of the first load-balance constraint.
    ///
    /// `load_balance_row_starts[s]` is `StageLayout::row_load_balance_start()`
    /// for stage `s` — NOT a hand-derived `row_water_balance_start + n_hydros`
    /// offset, which only held before the chronological per-block water rows
    /// and the travel-time bucket-definition rows sat between the two.
    /// Length equals `templates.len()`.
    pub load_balance_row_starts: Vec<usize>,
    /// Number of buses with stochastic load noise (`std_mw > 0`); equals
    /// `normal_lp.n_entities()`. Load noise occupies opening-tree noise-vector
    /// indices `[n_hydros, n_hydros + n_load_buses)`.
    pub n_load_buses: usize,
    /// Position in the `buses` slice for each stochastic load bus, length
    /// `n_load_buses`, sorted by [`cobre_core::EntityId`] for declaration-order
    /// invariance. Bus `i`'s load-balance base row is
    /// `load_balance_row_start + load_bus_indices[i] * n_blks + blk`.
    pub load_bus_indices: Vec<usize>,
    /// Per-stage metadata for active generic constraint rows: one
    /// [`GenericConstraintRowEntry`] per active `(constraint, block)` pair at
    /// stage `s`. Empty for stages with no active generic constraints.
    pub generic_constraint_row_entries: Vec<Vec<GenericConstraintRowEntry>>,
    /// Per-stage NCS column start indices.
    ///
    /// `ncs_col_starts[stage_idx]` is the column index of the first NCS generation
    /// variable for that stage. The base shifts per stage with `n_blks`, so it is
    /// legitimate per-stage geometry; the COUNT it strides is the scalar
    /// [`Self::n_ncs`].
    pub ncs_col_starts: Vec<usize>,
    /// NCS column count — the full system NCS count, identical at every stage.
    ///
    /// Under the dense layout every NCS keeps a column at every stage, so the count
    /// is a single scalar, not a per-stage Vec; a commissioning-dormant NCS keeps
    /// its column (pinned to `[0, 0]`).
    pub n_ncs: usize,
    /// Per-stage pumping-flow column start indices.
    ///
    /// `pumping_col_starts[stage_idx]` is the column index of the first
    /// pumping-flow variable for that stage, sourced from
    /// `StageLayout::col_pumping_start`, the sole owner of the pumping-flow column
    /// base. Pumping columns are block-major over ALL system stations (dense):
    /// `pumping_col_starts[stage_idx] + p_sys * n_blks + blk`, where `p_sys` is the
    /// SYSTEM station index. The base shifts per stage with `n_blks`, so it is
    /// legitimate per-stage geometry; the COUNT it strides is the scalar
    /// [`Self::n_pumping`]. A commissioning-dormant station keeps its column
    /// (pinned to `[0, 0]`).
    pub pumping_col_starts: Vec<usize>,
    /// Pumping-station column count — the full system station count, identical at
    /// every stage (dense). A commissioning-dormant station keeps its column.
    pub n_pumping: usize,
    /// Per-stage equipment geometry for simulation extraction.
    ///
    /// `geometry_per_stage[stage_idx]` holds the stage-correct column and row
    /// ranges for every block-major equipment family at that stage, sourced from
    /// the per-stage `StageLayout`. A single global stage-0 geometry would carry
    /// `n_blks`-striped bases/lengths that misread any stage with a differing
    /// block count. Length equals `templates.len()`. Threaded into
    /// `StageExtractionSpec` so the simulation read-path addresses the columns the
    /// solved primal occupies at the stage being extracted.
    pub geometry_per_stage: Vec<StageGeometry>,
    /// Mapping from target hydro ID to source hydro indices that divert to it.
    ///
    /// Used by the simulation extraction pipeline to compute `diverted_inflow_m3s`.
    /// Empty when no hydros have diversion.
    pub diversion_upstream: HashMap<EntityId, Vec<usize>>,
    /// Per-stage hydro productivities (MW per m³/s) for simulation extraction.
    ///
    /// `hydro_productivities_per_stage[stage][h]` is the productivity of hydro `h`
    /// at stage `stage`, accounting for per-stage overrides.  FPHA hydros have 0.0.
    pub hydro_productivities_per_stage: Vec<Vec<f64>>,
}

impl StageTemplates {
    /// All-empty [`StageTemplates`] for a study with zero stages. `n_hydros` and
    /// `cost_scale_factor` carry through; both are system-level values
    /// well-defined even with no stages.
    #[must_use]
    pub(crate) fn empty(n_hydros: usize, cost_scale_factor: f64) -> Self {
        Self {
            templates: Vec::new(),
            state_boxes: Vec::new(),
            zeta_per_stage: Vec::new(),
            block_hours_per_stage: Vec::new(),
            n_hydros,
            cost_scale_factor,
            load_balance_row_starts: Vec::new(),
            n_load_buses: 0,
            load_bus_indices: Vec::new(),
            generic_constraint_row_entries: Vec::new(),
            ncs_col_starts: Vec::new(),
            n_ncs: 0,
            pumping_col_starts: Vec::new(),
            n_pumping: 0,
            geometry_per_stage: Vec::new(),
            diversion_upstream: HashMap::new(),
            hydro_productivities_per_stage: Vec::new(),
        }
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
///
/// [`Default`] is the all-`0..0` geometry — every extraction read it gates returns
/// zero — the safe fallback when no per-stage geometry is available, matching the
/// sibling `ncs_col_starts` / `pumping_col_starts` empty-slice fallbacks.
#[derive(Debug, Clone, Default)]
pub struct StageGeometry {
    /// Future-cost epigraph (θ) column index — the authoritative value from
    /// `StageLayout::col_theta`, which accounts for the commitment-hold region's
    /// `commit_out`/`commit_in` column offset when `n_anticipated > 0`.
    /// Single source of truth for code that must address θ outside the builder
    /// (e.g. discount-factor postprocessing); do not re-derive the index from
    /// `n_state`/`n_hydros` by hand.
    pub theta_col: usize,
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
    /// `n_hydros` rows on a parallel stage, `n_hydros * n_blks` block-major rows on
    /// a chronological stage; address a row through
    /// [`StageGeometry::water_balance_row`].
    pub water_balance: Range<usize>,
    /// Load-balance row range (one row per bus per block; `n_buses · n_blks`);
    /// address a row through [`StageGeometry::load_balance_row`].
    pub load_balance: Range<usize>,
    /// FPHA hyperplane row range, immediately following `load_balance`. Length
    /// varies per stage: `for_each_fpha_plane` sums plane counts that differ per
    /// hydro (`fpha_hydro_indices.len() * n_blks` is NOT the row count).
    pub fpha: Range<usize>,
    /// Per-stage `σ_fill`-target row range (one row per Filling-phase hydro); empty
    /// `start..start` (not `0..0`) at every non-Filling stage.
    // Rationale (dead_code): no read site consumes this yet — the per-stage carrier
    // that keeps `StageGeometry` a faithful mirror of the row shape, the seam the
    // sibling `σ^{v-}` family extends.
    #[allow(dead_code)]
    pub filling_target: Range<usize>,
    /// Per-stage `σ_fill`-target slack column range (one column per Filling-phase
    /// hydro); empty `start..start` at every non-Filling stage. Simulation
    /// extraction reads the `σ_fill` primal at `start + local_idx`, resolving
    /// `local_idx` via `filling_target_hydro_indices`.
    pub filling_target_col: Range<usize>,
    /// Soft `σ^{v-}` operating-floor row range (one row per Operating-phase filling
    /// hydro); empty `start..start` (not `0..0`) at every non-operating stage.
    // Rationale (dead_code): no read site consumes this yet — the per-stage row-shape
    // carrier mirroring the `filling_target` seam.
    #[allow(dead_code)]
    pub filled_min_storage_floor: Range<usize>,
    /// Soft `σ^{v-}` operating-floor slack column range (one column per
    /// Operating-phase filling hydro); empty `start..start` at every non-operating
    /// stage. Simulation extraction reads the `σ^{v-}` primal at `start + local_idx`,
    /// resolving `local_idx` via `filled_min_storage_floor_hydro_indices`.
    pub filled_min_storage_floor_col: Range<usize>,
    /// Row index of the first z-inflow definition constraint. Always `0`: state
    /// pinning uses column bounds, so no state-fixing rows precede the z-inflow
    /// block. Carried per stage to mirror `StageLayout::z_inflow_row_start`.
    pub z_inflow_row_start: usize,
    /// Number of operating blocks (K) at this stage — the block-major stride for
    /// every equipment family.
    pub n_blks: usize,
    /// Storage-boundary address primitive for this stage, carrying both state
    /// bases plus the interior control-region anchor mirroring
    /// `StageLayout::storage_boundary_grid`; feeds [`StageGeometry::block_storage_col`].
    pub storage_boundary_grid: StorageBoundaryGrid,
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
    /// Storage column at chronological `boundary` for hydro `h`, so the
    /// simulation read-path resolves per-block boundaries without a
    /// `StageLayout`; delegates to
    /// [`StorageBoundaryGrid::col`](crate::lp::indexer::StorageBoundaryGrid::col),
    /// the single owner of the endpoints-vs-interior split.
    #[inline]
    #[must_use]
    pub fn block_storage_col(&self, h: HydroSys, boundary: Boundary) -> usize {
        self.storage_boundary_grid.col(h.get(), boundary)
    }

    fn water_balance_rows(&self) -> BlockRowFamily {
        match self.block_mode {
            BlockMode::Parallel => BlockRowFamily::one_per_entity(self.water_balance.clone()),
            BlockMode::Chronological => BlockRowFamily::per_block(self.water_balance.clone()),
        }
    }

    pub(crate) fn load_balance_rows(&self) -> BlockRowFamily {
        BlockRowFamily::per_block(self.load_balance.clone())
    }

    /// Hydro `h`'s water-balance row for block `blk`: its own block row on a
    /// chronological stage, its single stage row on a parallel stage (every block
    /// reads the same row).
    #[inline]
    #[must_use]
    pub fn water_balance_row(&self, h: HydroSys, blk: BlockIdx) -> usize {
        self.water_balance_rows().row(h.get(), blk, self.n_blks)
    }

    /// Bus `bus`'s load-balance row for block `blk` (`n_buses · n_blks` rows,
    /// strided by [`Self::n_blks`]).
    #[inline]
    #[must_use]
    pub fn load_balance_row(&self, bus: BusSys, blk: BlockIdx) -> usize {
        self.load_balance_rows().row(bus.get(), blk, self.n_blks)
    }

    /// NCS entity `ncs_sys`'s generation column for block `blk`.
    #[inline]
    #[must_use]
    pub fn ncs_generation_col(&self, ncs_sys: usize, blk: BlockIdx) -> usize {
        let col = BlockGrid::new(self.n_blks, 0).flat(self.ncs_generation.start, ncs_sys, blk);
        debug_assert!(
            col < self.ncs_generation.end,
            "NCS column {col} outside {:?}",
            self.ncs_generation
        );
        col
    }

    /// Pumping station `pumping_sys`'s flow column for block `blk`.
    #[inline]
    #[must_use]
    pub fn pumping_flow_col(&self, pumping_sys: usize, blk: BlockIdx) -> usize {
        let col = BlockGrid::new(self.n_blks, 0).flat(self.pumping_flow.start, pumping_sys, blk);
        debug_assert!(
            col < self.pumping_flow.end,
            "pumping column {col} outside {:?}",
            self.pumping_flow
        );
        col
    }
}

/// Per-stage outputs of [`build_single_stage_template`], transposed by
/// [`assemble_stage_templates_output`] into the parallel per-stage `Vec`s of
/// [`StageTemplates`]. Adding a per-stage datum is one field here plus one
/// transpose line in the assembler.
pub(super) struct StageBuildOutput {
    /// Structural LP template for the stage.
    pub template: StageTemplate,
    /// Row index of the first load-balance constraint (load-noise patches).
    pub load_balance_row_start: usize,
    /// Active generic-constraint row metadata for the stage.
    pub gc_entries: Vec<GenericConstraintRowEntry>,
    /// Column index of the first NCS generation variable.
    pub ncs_col_start: usize,
    /// Number of NCS entities at the stage — the full system count (dense).
    pub ncs_count: usize,
    /// Column index of the first pumping-flow variable (sourced from
    /// `StageLayout::col_pumping_start`).
    pub pumping_col_start: usize,
    /// Number of pumping stations ACTIVE (contributing columns) at the stage
    /// (the commissioning-gated count, sourced from [`StageLayout::n_pumping`]).
    pub n_pumping: usize,
    /// Stage-correct equipment column ranges for simulation extraction, computed
    /// from this stage's [`StageLayout`].
    pub equipment_geometry: StageGeometry,
}

/// Construct the [`StageBuildOutput`] for a single study stage.
// Rationale: `clippy::similar_names` flags the `state` handle next to `stage`/`stage_idx`;
// both names are established (the `StageLayout`/`StageData` field is `state`, the per-stage
// inputs are `stage`/`stage_idx`), so renaming either to satisfy the heuristic would obscure
// intent rather than clarify it.
#[allow(clippy::similar_names)]
pub(super) fn build_single_stage_template(
    ctx: &TemplateBuildCtx<'_>,
    state: &StateSpace,
    stage: &Stage,
    stage_idx: usize,
) -> StageBuildOutput {
    let layout = StageLayout::new(ctx, state, stage, stage_idx);
    let load_balance_row_start = layout.rows.load_balance.start();

    let (col_lower, mut col_upper, mut objective) =
        columns::fill_stage_columns(ctx, stage, stage_idx, &layout);
    let (mut row_lower, mut row_upper) = rows::fill_stage_rows(ctx, stage, stage_idx, &layout);
    let mut col_entries = entries::build_stage_matrix_entries(ctx, stage, stage_idx, &layout);

    let mut buffers = entries::LpMatrixBuffers {
        col_entries: &mut col_entries,
        col_upper: &mut col_upper,
        objective: &mut objective,
        row_lower: &mut row_lower,
        row_upper: &mut row_upper,
    };
    entries::fill_generic_constraint_entries(ctx, stage_idx, &layout, &mut buffers);

    // Scale every monetary objective coefficient by 1/K for numerical
    // conditioning; outputs are unscaled at the reporting boundary.
    //
    // Theta must NOT be divided: the Benders cuts already enforce
    // `theta >= Q_successor / K`, so theta holds the SCALED future cost. Dividing
    // it too would make the LP `stage_cost/K + (1/K)*theta`, which recovers
    // `stage_cost + future_cost/K` at the boundary — wrong. `layout.col_theta()`
    // reads the correct index even when `n_anticipated > 0` shifts theta.
    let theta_col = layout.col_theta();
    let cost_scale_factor = ctx.resolved.resolved_parameters.cost_scale_factor;
    for (i, coeff) in objective.iter_mut().enumerate() {
        if i != theta_col {
            *coeff /= cost_scale_factor;
        }
    }

    // CSC invariant: each column's entries must be row-sorted.
    for col_entry_vec in &mut col_entries {
        col_entry_vec.sort_unstable_by_key(|&(row, _)| row);
    }

    let (col_starts, row_indices, values) = entries::assemble_csc(&col_entries);

    let n_transfer = ctx.n_hydros * ctx.max_par_order;

    let template = StageTemplate {
        num_cols: layout.num_cols,
        num_rows: layout.rows.num_rows,
        num_nz: col_entries.iter().map(Vec::len).sum(),
        col_starts,
        row_indices,
        values,
        col_lower,
        col_upper,
        objective,
        row_lower,
        row_upper,
        n_state: layout.n_state(),
        n_transfer,
        n_dual_relevant: layout.rows.n_dual_relevant,
        n_hydro: layout.n_h,
        max_par_order: layout.lag_order,
        col_scale: Vec::new(),
        row_scale: Vec::new(),
    };

    // Snapshot the per-stage equipment geometry BEFORE moving `layout`'s owned
    // `generic_constraint_rows` Vec into the output: `geometry` only borrows
    // `layout`, so it must run while `layout` is intact.
    let equipment_geometry = layout.geometry(stage.block_mode);

    StageBuildOutput {
        template,
        load_balance_row_start,
        gc_entries: layout.generic_constraint_rows,
        ncs_col_start: layout.equipment.col_ncs_start,
        ncs_count: layout.equipment.n_ncs,
        pumping_col_start: layout.equipment.col_pumping_start,
        n_pumping: layout.equipment.n_pumping,
        equipment_geometry,
    }
}

/// Bus-slice positions of every load-noise-member bus
/// ([`System::load_noise_member_bus_ids`] — the single membership authority
/// `noise_entity_order` and the external library builders also route
/// through), sorted by `EntityId` for declaration-order invariance.
fn collect_load_bus_indices(
    system: &System,
    bus_pos: &BTreeMap<EntityId, usize>,
    load_scheme: SamplingScheme,
) -> Vec<usize> {
    system
        .load_noise_member_bus_ids(load_scheme)
        .iter()
        .filter_map(|id| bus_pos.get(id).copied())
        .collect()
}

/// Synthesize one entity-model per `(entity, stage)` for every entity in
/// `entity_ids`, reading `(mean, std)` from `normal_lp` at its own canonical
/// position. `entity_ids`/`study_stages` MUST be exactly the shape `normal_lp`
/// was built over (a `debug_assert` enforces it) — this is a pure positional
/// read, never a re-derivation from raw rows. Shared by the external library
/// builders' standardization-moment derivation
/// (`setup::scenario_libraries::build_external_load_library` /
/// `build_external_ncs_library`), so a library's standardization and
/// `cobre_stochastic::context`'s reconstruction read the identical moments
/// rather than each re-deriving independently.
pub(crate) fn models_from_normal<M>(
    normal_lp: &PrecomputedNormal,
    entity_ids: &[EntityId],
    study_stages: &[&Stage],
    constructor: impl Fn(EntityId, i32, f64, f64) -> M,
) -> Vec<M> {
    debug_assert_eq!(
        normal_lp.n_entities(),
        entity_ids.len(),
        "normal_lp must be built over exactly entity_ids"
    );
    debug_assert_eq!(
        normal_lp.n_stages(),
        study_stages.len(),
        "normal_lp must be built over exactly study_stages"
    );
    let mut models = Vec::with_capacity(study_stages.len() * entity_ids.len());
    for (stage_idx, stage) in study_stages.iter().enumerate() {
        for (entity_idx, &entity_id) in entity_ids.iter().enumerate() {
            models.push(constructor(
                entity_id,
                stage.id,
                normal_lp.mean(stage_idx, entity_idx),
                normal_lp.std(stage_idx, entity_idx),
            ));
        }
    }
    models
}

/// Declared load-balance rows for buses outside
/// [`System::load_noise_member_bus_ids`]. A member bus has no model here:
/// its template row is `0` (`fill_load_balance_rows`) and every solve
/// patches it through `StageSolvePrep`'s load patch.
fn deterministic_load_models(system: &System, load_scheme: SamplingScheme) -> Vec<LoadModel> {
    let member_ids: HashSet<EntityId> = system
        .load_noise_member_bus_ids(load_scheme)
        .into_iter()
        .collect();
    system
        .load_models()
        .iter()
        .filter(|lm| !member_ids.contains(&lm.bus_id))
        .cloned()
        .collect()
}

/// Build one [`StageTemplate`] per study stage from a fully loaded [`System`].
///
/// The templates encode the complete structural LP for each SDDP subproblem
/// in CSC format, ready for bulk-loading via `SolverInterface::load_model`.
/// They are constructed once at solver initialisation and shared read-only
/// across all solver threads.
///
/// ## Column and row layout
///
/// See the module-level documentation for the full LP layout.
/// Key dimensions for a stage with N hydros, T thermals, Lines lines,
/// B buses, K blocks per stage, and F FPHA hydros each with M planes:
///
/// - `num_cols` and `num_rows` are computed by `layout::StageLayout` —
///   see `layout.rs` for the authoritative column and row counts
/// - `n_state  = N*(1+L)`
/// - `n_transfer = N*L`  (storage + all lags except the oldest)
/// - `n_dual_relevant = 0`  (state pinning uses column bounds, not state-fixing rows, so no
///   structural row contributes to cut gradients; the cut path reads `view.reduced_costs`)
///
/// ## PAR order and `max_par_order`
///
/// `max_par_order` is the maximum of (a) the maximum AR coefficient count
/// across all hydro inflow models and (b) `par_lp.max_order()`.  The latter
/// is non-classical only when an annual component is present, in which case
/// the precompute widens the lag stride to 12 and the LP must allocate
/// matching column and row slots.  All hydros use the same uniform lag stride
/// `max_par_order` to enable SIMD-friendly contiguous access.
///
/// ## Objective coefficients
///
/// Costs are expressed in `$/MWh` (thermal, deficit, excess, lines) multiplied
/// by the block duration in hours so they integrate to $/block.  Storage, lag,
/// incoming-storage, theta, turbine, and spillage columns carry zero or small
/// regularization costs drawn from the resolved penalty tables.
///
/// When the penalty method is active, each inflow slack column `sigma_inf_h`
/// carries objective coefficient `penalty_cost * total_stage_hours`.
///
/// FPHA generation columns carry objective coefficient 0.0 by default.
///
/// ## Inflow non-negativity
///
/// When `inflow_method.has_slack_columns()` is `true` (i.e., the `Penalty`
/// variant), `N` slack columns `sigma_inf_h >= 0`
/// are appended at the end of the column layout.  Each slack enters the water
/// balance row for hydro `h` with coefficient `+tau_total * M3S_TO_HM3`,
/// acting as virtual inflow that prevents infeasibility when the PAR(p) noise
/// is sufficiently negative.
///
/// ## FPHA hydros
///
/// For hydros whose resolved production model at a given stage is
/// [`ResolvedProductionModel::Fpha`], generation becomes a free variable
/// `g_{h,k} ∈ [0, max_generation_mw]` bounded by M hyperplane constraints:
///
/// ```text
/// g_{h,k} - gamma_v/2*v - gamma_v/2*v_in - gamma_q*q_{h,k} - gamma_s*s_{h,k} <= gamma_0
/// ```
///
/// The `v_in` contribution propagates through the LP via the matrix coefficient
/// `-gamma_v/2` on the incoming-storage column; when `v_in` is pinned by that
/// column's bounds its value automatically enters the FPHA constraint
/// right-hand side.
///
/// Returns empty templates for a system with zero stages.  All entity counts
/// may be zero (valid for degenerate test systems).
///
/// ## Evaporation hydros
///
/// For hydros whose evaporation model is
/// `EvaporationModel::Linearized`,
/// three stage-level columns are added per hydro (evaporation outflow,
/// `f_evap_plus`, `f_evap_minus`).  The evaporation-outflow column is bounded
/// symmetrically `[-q_max, +q_max]` so a negative value can absorb net rainfall
/// input on the lake surface; `f_evap_plus` and `f_evap_minus` are bounded
/// `[0, +inf)`.  The evaporation-outflow column carries objective coefficient
/// 0.0; the violation slacks carry the evaporation penalty.  One equality
/// constraint row is added per evaporation hydro with
/// `row_lower == row_upper == intercept_m3s`.
///
// Rationale (too_many_arguments): each of the three arc-table parameters threads
// the single setup-owned derivation (`build_transit_bucket_topology`) through, the
// same coupling `per_stage_mask` already threads; a wrapper struct used at this one
// signature would rename the coupling, not remove it.
// implicit_hasher: callers pass a concrete `HashMap`; a `BuildHasher` generic buys
// nothing.
#[allow(clippy::too_many_arguments, clippy::implicit_hasher)]
#[expect(
    private_interfaces,
    reason = "time_value borrows the crate-private owner until this function's visibility narrows"
)]
#[must_use]
pub fn build_stage_templates(
    system: &System,
    inflow_method: InflowNonNegativityMethod,
    par_lp: &PrecomputedPar,
    normal_lp: &PrecomputedNormal,
    production_models: &ProductionModelSet,
    evaporation_models: &EvaporationModelSet,
    resolved_parameters: &ResolvedParameters,
    state_layout: &StateSpace,
    anticipated_plants: &AnticipatedPlants,
    per_stage_mask: &[Vec<usize>],
    arc_stage_weights: &HashMap<usize, Vec<Vec<f64>>>,
    arc_spread_chrono: &HashMap<usize, Vec<Option<SpreadResolution>>>,
    arc_arrival_density: &HashMap<usize, Vec<Option<Vec<f64>>>>,
    hydro_cell_index: &HydroCellIndex,
    load_scheme: SamplingScheme,
    time_value: &TimeValue,
) -> StageTemplates {
    let study_stages: Vec<_> = system.stages().iter().filter(|s| s.id >= 0).collect();
    let n_hydros = system.hydros().len();

    debug_assert!(
        par_lp.n_stages() == 0
            || (par_lp.n_stages() == study_stages.len() && par_lp.n_hydros() == n_hydros),
        "PrecomputedPar has {} stages x {} hydros but system has {} stages x {} hydros",
        par_lp.n_stages(),
        par_lp.n_hydros(),
        study_stages.len(),
        n_hydros
    );

    if study_stages.is_empty() {
        return StageTemplates::empty(n_hydros, resolved_parameters.cost_scale_factor);
    }

    let load_models = deterministic_load_models(system, load_scheme);
    let (ctx, load_bus_indices, diversion_upstream_output) = build_template_build_ctx(
        system,
        inflow_method,
        par_lp,
        &load_models,
        production_models,
        evaporation_models,
        resolved_parameters,
        state_layout.anticipated_resolution.clone(),
        state_layout.anticipated_lead_stages.clone(),
        anticipated_plants,
        per_stage_mask.to_vec(),
        arc_stage_weights.clone(),
        arc_spread_chrono.clone(),
        arc_arrival_density.clone(),
        state_layout.max_par_order,
        hydro_cell_index,
        load_scheme,
        time_value,
    );
    let n_load_buses = load_bus_indices.len();
    debug_assert!(
        normal_lp.n_entities() == 0 || normal_lp.n_entities() == n_load_buses,
        "PrecomputedNormal has {} entities but system has {} stochastic load buses",
        normal_lp.n_entities(),
        n_load_buses
    );
    debug_assert_eq!(
        ctx.anticipated_resolution, state_layout.anticipated_resolution,
        "ctx's threaded anticipated_resolution must match the state_layout it was built from"
    );
    debug_assert_eq!(
        ctx.anticipated_lead_stages, state_layout.anticipated_lead_stages,
        "ctx's threaded anticipated_lead_stages must match the state_layout it was built from"
    );
    debug_assert_eq!(
        ctx.max_par_order, state_layout.max_par_order,
        "ctx's threaded max_par_order must match the state_layout it was built from"
    );

    let n_study = study_stages.len();
    let mut stage_outputs = Vec::with_capacity(n_study);
    for (stage_idx, stage) in study_stages.iter().enumerate() {
        stage_outputs.push(build_single_stage_template(
            &ctx,
            state_layout,
            stage,
            stage_idx,
        ));
    }

    assemble_stage_templates_output(
        stage_outputs,
        load_bus_indices,
        diversion_upstream_output,
        &study_stages,
        &ctx,
        n_hydros,
        n_load_buses,
        n_study,
    )
}

/// Test/integration-only convenience wrapper over [`build_stage_templates`]:
/// resolves the state layout and bucket topology from `system`/`par_lp`
/// through the same setup entry point production uses
/// (`crate::setup::resolve_state_layout`), then delegates. Production
/// (`StudySetup`) always threads its own already-resolved
/// `StateSpace`/`per_stage_mask` directly through [`build_stage_templates`]
/// instead — this wrapper exists so test call sites that build templates from
/// a bare system do not each need to resolve the layout themselves.
///
/// # Errors
///
/// Propagates `crate::setup::resolve_state_layout`'s `LeadTime` fan-out
/// rejection.
#[cfg(any(test, feature = "test-support"))]
pub fn build_stage_templates_resolving_layout(
    system: &System,
    inflow_method: InflowNonNegativityMethod,
    par_lp: &PrecomputedPar,
    normal_lp: &PrecomputedNormal,
    production_models: &ProductionModelSet,
    evaporation_models: &EvaporationModelSet,
    resolved_parameters: &ResolvedParameters,
) -> Result<StageTemplates, SddpError> {
    let topology = build_transit_bucket_topology(system, false);
    let (state_layout, _, anticipated_plants) =
        resolve_state_layout(system, par_lp, &topology, None)?;
    let hydro_cell_index = HydroCellIndex::build(system.hydros());
    let study_stages: Vec<_> = system.stages().iter().filter(|s| s.id >= 0).collect();
    let study_total_hours: Vec<f64> = study_stages
        .iter()
        .map(|s| BlockClock::new(s).total_hours())
        .collect();
    let time_value = TimeValue::from_system(system, &anticipated_plants, &study_total_hours);
    Ok(build_stage_templates(
        system,
        inflow_method,
        par_lp,
        normal_lp,
        production_models,
        evaporation_models,
        resolved_parameters,
        &state_layout,
        &anticipated_plants,
        &topology.per_stage_mask,
        &topology.arc_stage_weights,
        &topology.arc_spread_chrono,
        &topology.arc_arrival_density,
        &hydro_cell_index,
        SamplingScheme::InSample,
        &time_value,
    ))
}

/// Precompute the per-stage minimum target-storage trajectory `V_target[t]` for
/// every filling hydro, keyed `(hydro_idx, stage_id) → V_target` \[hm³\].
///
/// Computed ONCE here, where the full per-stage ζ·rate schedule is available (a
/// per-stage row-fill helper sees one stage and cannot reconstruct the fold). With
/// `L = entry_stage_id − 1` the last Filling stage, anchored on the dead volume and
/// folded backward:
///
/// ```text
/// V_target[L] = min_storage_hm3                          (at L's stage_idx)
/// V_target[t] = min( V_target[t+1] − ζ_{t+1}·rate[t+1], min_storage_hm3 )
/// ```
///
/// `ζ_t = stage_zetas[stage_idx]`; `rate`/`min_storage` are
/// the RESOLVED per-stage bounds. The clip at `min_storage` enforces that no floor
/// exceeds the dead volume — dropping it would let an over-provisioned schedule
/// demand a floor ABOVE the dead volume — the forbidden alternative.
/// The fold runs on the UNCLIPPED running value, clipping each stored `V_target[t]`
/// independently to mirror the closed form.
///
/// Hydros are iterated in canonical slot order into a `BTreeMap`, so the result is
/// declaration-order-invariant; a non-filling system yields an empty map.
///
/// `pub(super)` so the sibling builder test modules can exercise it against
/// single-stage fixtures; production reaches it via `build_template_build_ctx`.
pub(super) fn build_filling_v_target(
    hydros: &[Hydro],
    bounds: &ResolvedBounds,
    stage_zetas: &[f64],
    stage_id_to_idx: &HashMap<i32, usize>,
) -> BTreeMap<(usize, i32), f64> {
    let mut v_target: BTreeMap<(usize, i32), f64> = BTreeMap::new();
    for (h_idx, hydro) in hydros.iter().enumerate() {
        let (Some(filling), Some(entry)) = (hydro.filling.as_ref(), hydro.entry_stage_id) else {
            continue;
        };
        let start = filling.start_stage_id;
        let last = entry - 1;
        // Guard a hypothetical inverted config (`start < entry` is validated
        // upstream) into an empty trajectory rather than a malformed loop.
        if last < start {
            continue;
        }
        let Some(&last_idx) = stage_id_to_idx.get(&last) else {
            continue;
        };
        let min_storage_at_last = bounds.hydro_bounds(h_idx, last_idx).min_storage_hm3;
        v_target.insert((h_idx, last), min_storage_at_last);
        let mut running = min_storage_at_last;
        let mut t = last;
        while t > start {
            if let Some(&t_idx) = stage_id_to_idx.get(&t) {
                let zeta_t = stage_zetas[t_idx];
                let rate_t = bounds.hydro_bounds(h_idx, t_idx).filling_min_rate_m3s;
                running -= zeta_t * rate_t;
            }
            let prev = t - 1;
            if let Some(&prev_idx) = stage_id_to_idx.get(&prev) {
                let min_storage_prev = bounds.hydro_bounds(h_idx, prev_idx).min_storage_hm3;
                v_target.insert((h_idx, prev), running.min(min_storage_prev));
            }
            t = prev;
        }
    }
    v_target
}

/// Build the [`TemplateBuildCtx`] and ancillary data needed by the stage loop.
///
/// Constructs position maps (hydro/thermal/line/bus), the diversion-upstream
/// map, and the `TemplateBuildCtx` that is shared across all per-stage builds.
/// Also returns `load_bus_indices` (the bus-slice positions of stochastic load
/// buses) and `diversion_upstream_output` (the clone of the diversion map
/// preserved for the final `StageTemplates` output field).
///
/// Called once per `build_stage_templates` invocation, after the early-return
/// guard for empty systems.
// Rationale: too_many_lines — one linear pass of per-entity prep blocks (position
// maps, anticipated metadata, contracts, discount factors) feeding a single
// `TemplateBuildCtx` literal; splitting it would scatter the construction the
// literal reads back, without removing any branching.
// Rationale: too_many_arguments — each parameter threads a single-owner value
// from `StateSpace` into the shared `TemplateBuildCtx` (mirroring the existing
// `anticipated_resolution`/`anticipated_lead_stages` threads); a wrapper struct
// used nowhere else would rename the coupling, not remove it.
#[allow(clippy::too_many_lines, clippy::too_many_arguments)]
fn build_template_build_ctx<'a>(
    system: &'a System,
    inflow_method: InflowNonNegativityMethod,
    par_lp: &'a PrecomputedPar,
    load_models: &'a [LoadModel],
    production_models: &'a ProductionModelSet,
    evaporation_models: &'a EvaporationModelSet,
    resolved_parameters: &'a ResolvedParameters,
    anticipated_resolution: AnticipatedResolution,
    anticipated_lead_stages: Vec<usize>,
    anticipated_plants: &'a AnticipatedPlants,
    per_stage_mask: Vec<Vec<usize>>,
    arc_stage_weights: HashMap<usize, Vec<Vec<f64>>>,
    arc_spread_chrono: HashMap<usize, Vec<Option<SpreadResolution>>>,
    arc_arrival_density: HashMap<usize, Vec<Option<Vec<f64>>>>,
    max_par_order: usize,
    hydro_cell_index: &'a HydroCellIndex,
    load_scheme: SamplingScheme,
    time_value: &'a TimeValue,
) -> (
    TemplateBuildCtx<'a>,
    Vec<usize>,
    HashMap<EntityId, Vec<usize>>,
) {
    let hydros = system.hydros();
    let buses = system.buses();
    let n_hydros = hydros.len();

    let hydro_pos: BTreeMap<EntityId, usize> =
        hydros.iter().enumerate().map(|(i, h)| (h.id, i)).collect();
    let thermal_pos: BTreeMap<EntityId, usize> = system
        .thermals()
        .iter()
        .enumerate()
        .map(|(i, t)| (t.id, i))
        .collect();
    let line_pos: BTreeMap<EntityId, usize> = system
        .lines()
        .iter()
        .enumerate()
        .map(|(i, l)| (l.id, i))
        .collect();
    let bus_pos: BTreeMap<EntityId, usize> =
        buses.iter().enumerate().map(|(i, b)| (b.id, i)).collect();

    // Iterate the (ID-sorted) station slice in slot order, NOT declaration order,
    // to uphold the declaration-order bit-determinism rule.
    let pumping_stations = system.pumping_stations();
    let pumping_pos: BTreeMap<EntityId, usize> = pumping_stations
        .iter()
        .enumerate()
        .map(|(i, p)| (p.id, i))
        .collect();
    let n_pumping = pumping_stations.len();
    // Fail fast on a station-count divergence rather than silently reserving the
    // wrong number of pumping-flow columns.
    debug_assert_eq!(
        n_pumping,
        system.bounds().n_pumping(),
        "pumping_stations.len() ({}) != bounds.n_pumping() ({}): resolved-bounds \
         station count disagrees with the entity slice",
        n_pumping,
        system.bounds().n_pumping()
    );

    // One id-sorted slice for both directions; the import/export split is derived
    // here as counts (the dense per-direction column strides) by `contract_type`.
    let contracts = system.contracts();
    let contract_pos: BTreeMap<EntityId, usize> = contracts
        .iter()
        .enumerate()
        .map(|(i, c)| (c.id, i))
        .collect();
    let n_contract_import = contracts
        .iter()
        .filter(|c| c.contract_type == ContractType::Import)
        .count();
    let n_contract_export = contracts
        .iter()
        .filter(|c| c.contract_type == ContractType::Export)
        .count();
    // Fail fast on a contract-count divergence rather than silently reserving the
    // wrong number of contract columns.
    debug_assert_eq!(
        contracts.len(),
        system.bounds().n_contracts(),
        "contracts.len() ({}) != bounds.n_contracts() ({}): resolved-bounds \
         contract count disagrees with the entity slice",
        contracts.len(),
        system.bounds().n_contracts()
    );

    let load_bus_indices = collect_load_bus_indices(system, &bus_pos, load_scheme);

    let anticipated_windows: Vec<(Option<i32>, Option<i32>)> = anticipated_plants
        .thermals()
        .map(|t| {
            let thermal = &system.thermals()[t.get()];
            (thermal.entry_stage_id, thermal.exit_stage_id)
        })
        .collect();
    let n_anticipated = anticipated_plants.len();

    // Cloned so the map serves both LP construction (ctx) and the simulation
    // extraction output.
    let mut diversion_upstream: HashMap<EntityId, Vec<usize>> = HashMap::new();
    for (h_idx, hydro) in hydros.iter().enumerate() {
        if let Some(ref div) = hydro.diversion {
            diversion_upstream
                .entry(div.downstream_id)
                .or_default()
                .push(h_idx);
        }
    }
    let diversion_upstream_output = diversion_upstream.clone();

    // Computed before the per-stage loop so `fill_anticipated_columns` can read the
    // discount factors and stage hours from the ctx at LP build time (before
    // postprocess runs).
    let study_stages: Vec<_> = system.stages().iter().filter(|s| s.id >= 0).collect();
    let block_clocks: Vec<BlockClock<'_>> =
        study_stages.iter().map(|s| BlockClock::new(s)).collect();

    debug_assert_eq!(
        block_clocks.len(),
        study_stages.len(),
        "block_clocks length must equal n_study_stages"
    );

    // Study-stage ids by study stage index: the decision gate keys its
    // operation-window clause on the DELIVERY stage's `stage.id`, mapping the
    // delivery index `t + K_i` to its id through this slice.
    let study_stage_ids: Vec<i32> = study_stages.iter().map(|s| s.id).collect();

    let stage_resolver = StageIdResolver::from_study_stage_ids(&study_stage_ids);

    let stage_zetas: Vec<f64> = block_clocks.iter().map(|c| c.zeta()).collect();
    let filling_v_target = build_filling_v_target(
        hydros,
        system.bounds(),
        &stage_zetas,
        stage_resolver.index_map(),
    );

    let ctx = TemplateBuildCtx {
        hydros,
        thermals: system.thermals(),
        lines: system.lines(),
        buses,
        load_models,
        cascade: system.cascade(),
        hydro_cell_index,
        resolved: ResolvedTables {
            bounds: system.bounds(),
            penalties: system.penalties(),
            resolved_generic_bounds: system.resolved_generic_bounds(),
            resolved_load_factors: system.resolved_load_factors(),
            resolved_ncs_bounds: system.resolved_ncs_bounds(),
            resolved_ncs_factors: system.resolved_ncs_factors(),
            resolved_parameters,
        },
        hydro_pos,
        thermal_pos,
        line_pos,
        bus_pos,
        par_lp,
        production_models,
        evaporation_models,
        generic_constraints: system.generic_constraints(),
        non_controllable_sources: system.non_controllable_sources(),
        pumping_stations,
        pumping_pos,
        n_pumping,
        contracts,
        contract_pos,
        n_contract_import,
        n_contract_export,
        diversion_upstream,
        n_hydros,
        n_thermals: system.thermals().len(),
        n_lines: system.lines().len(),
        n_buses: buses.len(),
        max_par_order,
        n_anticipated,
        anticipated_lead_stages,
        anticipated_plants,
        anticipated_windows,
        anticipated_resolution,
        study_stage_ids,
        has_penalty: n_hydros > 0 && inflow_method.has_slack_columns(),
        time_value,
        filling_v_target,
        arc_stage_weights,
        arc_spread_chrono,
        arc_arrival_density,
        per_stage_mask,
    };

    (ctx, load_bus_indices, diversion_upstream_output)
}

/// Transpose the per-stage `Vec<StageBuildOutput>` into the parallel per-stage
/// `Vec`s of [`StageTemplates`], computing the noise-scale, zeta, block-hour,
/// and hydro-productivity arrays.
// Rationale: the args have distinct lifetimes and ownership (some borrowed, some
// owned), so bundling them into one struct would buy nothing on this single-call
// cold path while obscuring the transpose inputs.
#[allow(clippy::too_many_arguments)]
fn assemble_stage_templates_output(
    stage_outputs: Vec<StageBuildOutput>,
    load_bus_indices: Vec<usize>,
    diversion_upstream_output: HashMap<EntityId, Vec<usize>>,
    study_stages: &[&Stage],
    ctx: &TemplateBuildCtx<'_>,
    n_hydros: usize,
    n_load_buses: usize,
    n_study: usize,
) -> StageTemplates {
    // Index `s` of every parallel Vec must refer to the same stage, so preserve the
    // per-stage push order.
    let mut templates = Vec::with_capacity(n_study);
    let mut load_balance_row_starts = Vec::with_capacity(n_study);
    let mut generic_constraint_row_entries = Vec::with_capacity(n_study);
    let mut ncs_col_starts = Vec::with_capacity(n_study);
    let mut pumping_col_starts = Vec::with_capacity(n_study);
    let mut geometry_per_stage = Vec::with_capacity(n_study);
    // The dense NCS/pumping counts are constant across stages, so they collapse to
    // scalars (column STARTS stay per-stage, riding `n_blks`): the first output
    // seeds the scalars, later outputs must agree.
    let mut n_ncs: usize = 0;
    let mut n_pumping: usize = 0;
    for (s, out) in stage_outputs.into_iter().enumerate() {
        templates.push(out.template);
        load_balance_row_starts.push(out.load_balance_row_start);
        generic_constraint_row_entries.push(out.gc_entries);
        ncs_col_starts.push(out.ncs_col_start);
        pumping_col_starts.push(out.pumping_col_start);
        if s == 0 {
            n_ncs = out.ncs_count;
            n_pumping = out.n_pumping;
        } else {
            debug_assert_eq!(
                out.ncs_count, n_ncs,
                "dense NCS count must be constant across stages",
            );
            debug_assert_eq!(
                out.n_pumping, n_pumping,
                "dense pumping count must be constant across stages",
            );
        }
        debug_assert_eq!(
            out.equipment_geometry.ncs_generation.start, out.ncs_col_start,
            "geometry.ncs_generation must start at the legacy ncs_col_start",
        );
        debug_assert_eq!(
            out.equipment_geometry.ncs_generation.len(),
            out.ncs_count * out.equipment_geometry.n_blks,
            "geometry.ncs_generation must span ncs_count * n_blks columns",
        );
        debug_assert_eq!(
            out.equipment_geometry.pumping_flow.start, out.pumping_col_start,
            "geometry.pumping_flow must start at the legacy pumping_col_start",
        );
        debug_assert_eq!(
            out.equipment_geometry.pumping_flow.len(),
            out.n_pumping * out.equipment_geometry.n_blks,
            "geometry.pumping_flow must span n_pumping * n_blks columns",
        );
        geometry_per_stage.push(out.equipment_geometry);
    }

    let (zeta_per_stage, block_hours_per_stage) = scaling::compute_stage_hours(study_stages);

    let hydro_productivities_per_stage: Vec<Vec<f64>> = (0..n_study)
        .map(|s| {
            (0..n_hydros)
                .map(|h| match ctx.production_models.model(h, s) {
                    ResolvedProductionModel::ConstantProductivity { productivity } => *productivity,
                    ResolvedProductionModel::Fpha { .. } => 0.0,
                })
                .collect()
        })
        .collect();

    StageTemplates {
        templates,
        state_boxes: Vec::new(),
        zeta_per_stage,
        block_hours_per_stage,
        n_hydros,
        cost_scale_factor: ctx.resolved.resolved_parameters.cost_scale_factor,
        load_balance_row_starts,
        n_load_buses,
        load_bus_indices,
        generic_constraint_row_entries,
        ncs_col_starts,
        n_ncs,
        pumping_col_starts,
        n_pumping,
        geometry_per_stage,
        diversion_upstream: diversion_upstream_output,
        hydro_productivities_per_stage,
    }
}

#[cfg(test)]
mod tests;
