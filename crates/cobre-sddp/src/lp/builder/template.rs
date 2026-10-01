use std::collections::HashMap;

use cobre_core::{EntityId, Stage, System};
use cobre_solver::StageTemplate;
use cobre_stochastic::normal::precompute::PrecomputedNormal;
use cobre_stochastic::par::precompute::PrecomputedPar;

use crate::bucket_topology::TransitBucketTopology;
use crate::hydro_models::{EvaporationModelSet, ProductionModelSet};

use super::layout::{ResolvedTables, StageGeometry, StageLayout, TemplateBuildCtx};
use super::{GenericConstraintRowEntry, LpBuildInputs, StateBox, columns, entries, rows, scaling};
use crate::lp::indexer::StateSpace;

#[cfg(any(test, feature = "test-support"))]
pub(crate) mod canonical;

/// Outcome of [`build_stage_templates`]: one [`StageTemplate`] per study stage
/// plus the per-stage offsets and counts the forward/backward/simulation passes
/// need. The per-stage `Vec`s are parallel — index `s` of each refers to stage `s`.
#[derive(Debug, Clone)]
pub struct StageTemplates {
    /// One structural LP template per study stage, in stage order.
    pub templates: Vec<StageTemplate>,
    /// Per-stage admissible box for every outgoing state dimension, built by
    /// `postprocess_templates` from the physical column bounds before column
    /// scaling. Empty until then; read it through `state_boxes()`.
    state_boxes: Vec<StateBox>,
    /// Per-stage block durations in hours (`block_hours_per_stage[stage]` is length
    /// `n_blocks`). Converts load-balance duals $/MW → $/`MWh`:
    /// `spot_price = dual / block_hours`.
    pub block_hours_per_stage: Vec<Vec<f64>>,
    /// Resolved objective cost-scale factor (`modeling.cost_scale_factor`,
    /// the resolved `cost_scale_factor` scalar). Every non-theta objective
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

    /// The per-stage state boxes, one per template.
    ///
    /// # Panics
    ///
    /// In debug builds, when read before `postprocess_templates` fills them.
    pub(crate) fn state_boxes(&self) -> &[StateBox] {
        debug_assert_eq!(
            self.state_boxes.len(),
            self.templates.len(),
            "state_boxes read before postprocess_templates filled them"
        );
        &self.state_boxes
    }

    /// Fills the per-stage state boxes, one per template.
    pub(crate) fn set_state_boxes(&mut self, state_boxes: Vec<StateBox>) {
        debug_assert_eq!(
            state_boxes.len(),
            self.templates.len(),
            "set_state_boxes needs one state box per stage"
        );
        self.state_boxes = state_boxes;
    }
}

/// Per-stage outputs of [`build_single_stage_template`], transposed by
/// [`assemble_stage_templates_output`] into the parallel per-stage `Vec`s of
/// [`StageTemplates`]. Adding a per-stage datum is one field here plus one
/// transpose line in the assembler.
pub(super) struct StageBuildOutput {
    pub template: StageTemplate,
    pub gc_entries: Vec<GenericConstraintRowEntry>,
    /// Stage-correct equipment column ranges for simulation extraction, computed
    /// from this stage's [`StageLayout`].
    pub equipment_geometry: StageGeometry,
}

/// Construct the [`StageBuildOutput`] for a single study stage.
pub(super) fn build_single_stage_template(
    ctx: &TemplateBuildCtx<'_>,
    stage: &Stage,
    stage_idx: usize,
) -> StageBuildOutput {
    let layout = StageLayout::new(ctx, stage, stage_idx);

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

    finalize_stage_objective(ctx, stage_idx, &layout, &mut objective);

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

    StageBuildOutput {
        template,
        gc_entries: layout.generic_constraint_rows,
        equipment_geometry: layout.geometry,
    }
}

/// Turn the stage's raw costs into its final objective: every coefficient but
/// θ's is divided by the cost scale factor, and θ takes the stage's one-step
/// discount factor, which cascades every later stage's cost to the root exactly
/// once.
///
/// θ is not divided because the Benders cuts already bound it by the scaled
/// future cost.
fn finalize_stage_objective(
    ctx: &TemplateBuildCtx<'_>,
    stage_idx: usize,
    layout: &StageLayout,
    objective: &mut [f64],
) {
    debug_assert_eq!(
        ctx.time_value.discount_factors().len(),
        ctx.resolved.bounds.n_stages(),
        "time_value.discount_factors must have length n_stages"
    );
    let theta_col = layout.col_theta();
    let cost_scale_factor = ctx.resolved.resolved_parameters.cost_scale_factor;
    for (i, coeff) in objective.iter_mut().enumerate() {
        if i != theta_col {
            *coeff /= cost_scale_factor;
        }
    }
    objective[theta_col] = ctx.time_value.discount_factors()[stage_idx];
}

/// Synthesize one entity-model per `(entity, stage)` for every entity in
/// `entity_ids`, reading `(mean, std)` from `normal_lp` at its own canonical
/// position. `entity_ids`/`study_stages` MUST be exactly the shape `normal_lp`
/// was built over (a `debug_assert` enforces it) — this is a pure positional
/// read, never a re-derivation from raw rows. Shared by the external library
/// builders' standardization-moment derivation
/// (`build_external_load_library` / `build_external_ncs_library`), so a
/// library's standardization and `cobre_stochastic::context`'s
/// reconstruction read the identical moments rather than each re-deriving
/// independently.
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
#[must_use]
pub(crate) fn build_stage_templates(
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

    let stage_outputs = study_stages
        .iter()
        .enumerate()
        .map(|(stage_idx, stage)| build_single_stage_template(&ctx, stage, stage_idx))
        .collect();

    assemble_stage_templates_output(
        stage_outputs,
        inputs.load_bus_indices,
        inputs.diversion_upstream,
        inputs.hydro_productivities_per_stage,
        &study_stages,
        inputs.resolved_parameters.cost_scale_factor,
    )
}

/// Build the [`TemplateBuildCtx`] shared across all per-stage builds from
/// `system`'s slices and the fields it borrows from `inputs`.
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
    let mut templates = Vec::with_capacity(n_study);
    let mut generic_constraint_row_entries = Vec::with_capacity(n_study);
    let mut geometry_per_stage = Vec::with_capacity(n_study);
    for out in stage_outputs {
        templates.push(out.template);
        generic_constraint_row_entries.push(out.gc_entries);
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
        geometry_per_stage,
        diversion_upstream,
        hydro_productivities_per_stage,
    }
}

#[cfg(test)]
mod tests;
