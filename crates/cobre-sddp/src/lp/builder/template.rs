use std::collections::HashMap;
use std::ops::Range;

use cobre_core::{BlockMode, EntityId, Stage, System};
use cobre_solver::StageTemplate;
use cobre_stochastic::normal::precompute::PrecomputedNormal;
use cobre_stochastic::par::precompute::PrecomputedPar;

use crate::bucket_topology::TransitBucketTopology;
#[cfg(any(test, feature = "test-support"))]
use crate::error::SddpError;
use crate::hydro_models::{EvaporationModelSet, ProductionModelSet};
#[cfg(any(test, feature = "test-support"))]
use crate::inflow_method::InflowNonNegativityMethod;
#[cfg(any(test, feature = "test-support"))]
use crate::resolved_parameters::ResolvedParameters;
#[cfg(any(test, feature = "test-support"))]
use crate::time_value::DeliveryCalendar;
#[cfg(any(test, feature = "test-support"))]
use crate::time_value::TimeValue;

use super::layout::{ResolvedTables, StageLayout, TemplateBuildCtx, entity_flat};
use super::{GenericConstraintRowEntry, LpBuildInputs, StateBox, columns, entries, rows, scaling};
#[cfg(any(test, feature = "test-support"))]
use crate::bucket_topology::build_transit_bucket_topology;
#[cfg(any(test, feature = "test-support"))]
use crate::lp::indexer::HydroCellIndex;
use crate::lp::indexer::{
    AnticipatedLocal, BlockGrid, BlockIdx, BlockRowFamily, Boundary, BusSys, EvaporationIndices,
    FillingTargetLocal, FloorLocal, HydroSys, NcsSys, PumpingSys, StateSpace, StorageBoundaryGrid,
};
#[cfg(any(test, feature = "test-support"))]
use crate::setup::{
    build_study_dimensions, resolve_lp_build_inputs, resolve_stage_lag_transitions,
    resolve_state_layout,
};

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
    /// Per-stage block durations in hours (`block_hours_per_stage[stage]` is length
    /// `n_blocks`). Converts load-balance duals $/MW → $/`MWh`:
    /// `spot_price = dual / block_hours`.
    pub block_hours_per_stage: Vec<Vec<f64>>,
    /// Resolved objective cost-scale factor (`modeling.cost_scale_factor`,
    /// [`ResolvedParameters::cost_scale_factor`]). Every non-theta objective
    /// coefficient was divided by this at template build time; cost-domain
    /// reporting boundaries multiply back by it.
    pub cost_scale_factor: f64,
    /// Position in the `buses` slice for each stochastic load bus, sorted by
    /// [`cobre_core::EntityId`] for declaration-order invariance. Bus `i`'s
    /// load-balance base row is [`StageGeometry::load_balance_row`].
    pub load_bus_indices: Vec<usize>,
    /// Per-stage metadata for active generic constraint rows: one
    /// [`GenericConstraintRowEntry`] per active `(constraint, block)` pair at
    /// stage `s`. Empty for stages with no active generic constraints.
    pub generic_constraint_row_entries: Vec<Vec<GenericConstraintRowEntry>>,
    /// NCS column count — the full system NCS count, identical at every stage.
    ///
    /// Under the dense layout every NCS keeps a column at every stage, so the count
    /// is a single scalar, not a per-stage Vec; a commissioning-dormant NCS keeps
    /// its column (pinned to `[0, 0]`).
    pub n_ncs: usize,
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
    /// All-empty [`StageTemplates`] for a study with zero stages.
    /// `cost_scale_factor` carries through — a system-level value well-defined
    /// even with no stages.
    #[must_use]
    pub(crate) fn empty(cost_scale_factor: f64) -> Self {
        Self {
            templates: Vec::new(),
            state_boxes: Vec::new(),
            block_hours_per_stage: Vec::new(),
            cost_scale_factor,
            load_bus_indices: Vec::new(),
            generic_constraint_row_entries: Vec::new(),
            n_ncs: 0,
            n_pumping: 0,
            geometry_per_stage: Vec::new(),
            diversion_upstream: HashMap::new(),
            hydro_productivities_per_stage: Vec::new(),
        }
    }

    /// Buses with stochastic load noise.
    #[inline]
    #[must_use]
    pub fn n_load_buses(&self) -> usize {
        self.load_bus_indices.len()
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
    /// Storage-boundary address primitive for this stage, carrying the
    /// interior control-region anchor mirroring
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
        self.storage_boundary_grid.col(state, h, boundary)
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
        let col =
            BlockGrid::new(self.n_blks, 0).flat(self.ncs_generation.start, ncs_sys.get(), blk);
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
    pub fn pumping_flow_col(&self, pumping_sys: PumpingSys, blk: BlockIdx) -> usize {
        let col =
            BlockGrid::new(self.n_blks, 0).flat(self.pumping_flow.start, pumping_sys.get(), blk);
        debug_assert!(
            col < self.pumping_flow.end,
            "pumping column {col} outside {:?}",
            self.pumping_flow
        );
        col
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

/// Per-stage outputs of [`build_single_stage_template`], transposed by
/// [`assemble_stage_templates_output`] into the parallel per-stage `Vec`s of
/// [`StageTemplates`]. Adding a per-stage datum is one field here plus one
/// transpose line in the assembler.
pub(super) struct StageBuildOutput {
    /// Structural LP template for the stage.
    pub template: StageTemplate,
    /// Active generic-constraint row metadata for the stage.
    pub gc_entries: Vec<GenericConstraintRowEntry>,
    /// Number of NCS entities at the stage — the full system count (dense).
    pub ncs_count: usize,
    /// Number of pumping stations ACTIVE (contributing columns) at the stage
    /// (the commissioning-gated count, sourced from
    /// [`super::layout::EquipmentColumns::n_pumping`]).
    pub n_pumping: usize,
    /// Stage-correct equipment column ranges for simulation extraction, computed
    /// from this stage's [`StageLayout`].
    pub equipment_geometry: StageGeometry,
}

/// Construct the [`StageBuildOutput`] for a single study stage.
#[expect(
    clippy::similar_names,
    reason = "state is the StageData field and stage/stage_idx are the per-stage inputs, so renaming either would obscure it"
)]
pub(super) fn build_single_stage_template(
    ctx: &TemplateBuildCtx<'_>,
    state: &StateSpace,
    stage: &Stage,
    stage_idx: usize,
) -> StageBuildOutput {
    let layout = StageLayout::new(ctx, state, stage, stage_idx);

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
        col_scale: Vec::new(),
        row_scale: Vec::new(),
    };

    // Snapshot the per-stage equipment geometry BEFORE moving `layout`'s owned
    // `generic_constraint_rows` Vec into the output: `geometry` only borrows
    // `layout`, so it must run while `layout` is intact.
    let equipment_geometry = layout.geometry(stage.block_mode);

    StageBuildOutput {
        template,
        gc_entries: layout.generic_constraint_rows,
        ncs_count: layout.equipment.n_ncs,
        n_pumping: layout.equipment.n_pumping,
        equipment_geometry,
    }
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
/// For hydros whose resolved production model at a given stage is FPHA,
/// generation becomes a free variable `g_{h,k} ∈ [0, max_generation_mw]`
/// bounded by M hyperplane constraints:
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
#[expect(
    private_interfaces,
    reason = "time_value borrows the crate-private owner until this function's visibility narrows"
)]
#[must_use]
pub fn build_stage_templates(
    system: &System,
    par_lp: &PrecomputedPar,
    production_models: &ProductionModelSet,
    evaporation_models: &EvaporationModelSet,
    state_layout: &StateSpace,
    topology: &TransitBucketTopology,
    inputs: LpBuildInputs<'_>,
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
        return StageTemplates::empty(inputs.resolved_parameters.cost_scale_factor);
    }

    let ctx = build_template_build_ctx(
        system,
        par_lp,
        production_models,
        evaporation_models,
        state_layout,
        topology,
        &inputs,
    );

    let mut stage_outputs = Vec::with_capacity(study_stages.len());
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
        inputs.load_bus_indices,
        inputs.diversion_upstream,
        inputs.hydro_productivities_per_stage,
        &study_stages,
        inputs.resolved_parameters.cost_scale_factor,
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
    let calendar = DeliveryCalendar::from_system(system);
    let topology = build_transit_bucket_topology(system, &calendar, false);
    let layout = resolve_state_layout(system, &calendar, par_lp, &topology, None)?;
    let hydro_cell_index = HydroCellIndex::build(system.hydros());
    let stages: Vec<Stage> = system
        .stages()
        .iter()
        .filter(|s| s.id >= 0)
        .cloned()
        .collect();
    let (downstream_par_order, _) =
        resolve_stage_lag_transitions(&stages, par_lp, system.policy_graph().season_map.as_ref());
    let study_dims = build_study_dimensions(
        system,
        inflow_method,
        layout.state.hydro_count,
        layout.anticipated_plants.clone(),
        downstream_par_order,
    );
    let time_value = TimeValue::from_system(system, &study_dims.anticipated_plants, calendar);
    let inputs = resolve_lp_build_inputs(
        system,
        &crate::setup::lp_build_inputs::resolve_in_sample_load_bus_ids(system),
        production_models,
        &study_dims,
        &time_value,
        &hydro_cell_index,
        resolved_parameters,
    );
    debug_assert_eq!(
        normal_lp.n_entities(),
        inputs.load_bus_indices.len(),
        "load noise model and LP disagree on the stochastic load buses"
    );
    Ok(build_stage_templates(
        system,
        par_lp,
        production_models,
        evaporation_models,
        &layout.state,
        &topology,
        inputs,
    ))
}

/// Build the [`TemplateBuildCtx`] shared across all per-stage builds, from
/// `system`'s own slices plus every field it borrows from `inputs` (the
/// resolved positions, load models, filling target, diversion map, study
/// dimensions, time value, hydro-cell index, and resolved parameters).
///
/// Called once per `build_stage_templates` invocation, after the early-return
/// guard for empty systems.
fn build_template_build_ctx<'a>(
    system: &'a System,
    par_lp: &'a PrecomputedPar,
    production_models: &'a ProductionModelSet,
    evaporation_models: &'a EvaporationModelSet,
    state: &'a StateSpace,
    topology: &'a TransitBucketTopology,
    inputs: &'a LpBuildInputs<'a>,
) -> TemplateBuildCtx<'a> {
    TemplateBuildCtx {
        hydros: system.hydros(),
        thermals: system.thermals(),
        lines: system.lines(),
        buses: system.buses(),
        load_models: &inputs.deterministic_load_models,
        cascade: system.cascade(),
        hydro_cell_index: inputs.hydro_cell_index,
        resolved: ResolvedTables {
            bounds: system.bounds(),
            penalties: system.penalties(),
            resolved_generic_bounds: system.resolved_generic_bounds(),
            resolved_load_factors: system.resolved_load_factors(),
            resolved_ncs_bounds: system.resolved_ncs_bounds(),
            resolved_ncs_factors: system.resolved_ncs_factors(),
            resolved_parameters: inputs.resolved_parameters,
        },
        positions: &inputs.positions,
        par_lp,
        production_models,
        evaporation_models,
        generic_constraints: system.generic_constraints(),
        non_controllable_sources: system.non_controllable_sources(),
        // Iterate the (ID-sorted) station slice in slot order, NOT declaration
        // order, to uphold the declaration-order bit-determinism rule.
        pumping_stations: system.pumping_stations(),
        contracts: system.contracts(),
        diversion_upstream: &inputs.diversion_upstream,
        state,
        study_dims: inputs.study_dims,
        time_value: inputs.time_value,
        filling_v_target: &inputs.filling_v_target,
        topology,
    }
}

/// Transpose the per-stage `Vec<StageBuildOutput>` into the parallel per-stage
/// `Vec`s of [`StageTemplates`], moving in the resolved load-bus indices,
/// diversion map, and hydro productivities.
fn assemble_stage_templates_output(
    stage_outputs: Vec<StageBuildOutput>,
    load_bus_indices: Vec<usize>,
    diversion_upstream: HashMap<EntityId, Vec<usize>>,
    hydro_productivities_per_stage: Vec<Vec<f64>>,
    study_stages: &[&Stage],
    cost_scale_factor: f64,
) -> StageTemplates {
    let n_study = stage_outputs.len();
    // Index `s` of every parallel Vec must refer to the same stage, so preserve the
    // per-stage push order.
    let mut templates = Vec::with_capacity(n_study);
    let mut generic_constraint_row_entries = Vec::with_capacity(n_study);
    let mut geometry_per_stage = Vec::with_capacity(n_study);
    // The dense NCS/pumping counts are constant across stages, so they collapse to
    // scalars: the first output seeds the scalars, later outputs must agree.
    let mut n_ncs: usize = 0;
    let mut n_pumping: usize = 0;
    for (s, out) in stage_outputs.into_iter().enumerate() {
        templates.push(out.template);
        generic_constraint_row_entries.push(out.gc_entries);
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
            out.equipment_geometry.ncs_generation.len(),
            out.ncs_count * out.equipment_geometry.n_blks,
            "geometry.ncs_generation must span ncs_count * n_blks columns",
        );
        debug_assert_eq!(
            out.equipment_geometry.pumping_flow.len(),
            out.n_pumping * out.equipment_geometry.n_blks,
            "geometry.pumping_flow must span n_pumping * n_blks columns",
        );
        geometry_per_stage.push(out.equipment_geometry);
    }

    let block_hours_per_stage = scaling::compute_stage_hours(study_stages);

    StageTemplates {
        templates,
        state_boxes: Vec::new(),
        block_hours_per_stage,
        cost_scale_factor,
        load_bus_indices,
        generic_constraint_row_entries,
        n_ncs,
        n_pumping,
        geometry_per_stage,
        diversion_upstream,
        hydro_productivities_per_stage,
    }
}

#[cfg(test)]
mod tests;
