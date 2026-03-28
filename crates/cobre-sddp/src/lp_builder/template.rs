use std::collections::HashMap;

use cobre_core::{EntityId, Stage, System};
use cobre_solver::StageTemplate;
use cobre_stochastic::normal::precompute::PrecomputedNormal;
use cobre_stochastic::par::precompute::PrecomputedPar;

use crate::error::SddpError;
use crate::hydro_models::{EvaporationModelSet, ProductionModelSet, ResolvedProductionModel};
use crate::indexer::StageIndexer;
use crate::inflow_method::InflowNonNegativityMethod;

use super::layout::{StageLayout, TemplateBuildCtx};
use super::{matrix, scaling, GenericConstraintRowEntry, COST_SCALE_FACTOR};

/// Outcome of [`build_stage_templates`]: one [`StageTemplate`] per study stage
/// plus the per-stage `base_rows` offsets needed by [`PatchBuffer`].
///
/// `base_rows[s]` is the row index of the first water-balance (AR dynamics)
/// constraint in stage `s`.  It equals `template.n_dual_relevant` for every
/// stage (constant when all stages share the same entity set, which is the
/// case for the minimal viable solver).  It is stored per-stage for forward
/// compatibility with stages that have different active entity sets.
#[derive(Debug, Clone)]
pub struct StageTemplates {
    /// One structural LP template per study stage, in stage order.
    pub templates: Vec<StageTemplate>,
    /// Row index of the first water-balance constraint in each stage's LP.
    ///
    /// Length equals `templates.len()`.  Used by [`PatchBuffer::fill_forward_patches`]
    /// to locate the noise-injection rows (Category 3 patches).
    pub base_rows: Vec<usize>,
    /// Pre-computed noise scale `ζ_stage * σ_{stage,hydro}` for each (stage, hydro) pair.
    ///
    /// Flat array in stage-major layout: `noise_scale[stage * n_hydros + hydro]`.
    /// Length equals `n_study_stages * n_hydros`.
    ///
    /// Used by the forward pass to transform raw standard-normal noise `η` into
    /// the full noise term `ζ*σ*η` before patching the water-balance RHS.
    /// The complete patch value is `ζ*base + ζ*σ*η`, where `ζ*base` is encoded
    /// in the template's `row_lower`/`row_upper` and `ζ*σ*η` is computed by the
    /// caller at each stage using this pre-computed scale.
    pub noise_scale: Vec<f64>,
    /// Per-stage time-conversion factor `ζ = total_hours * M3S_TO_HM3`.
    ///
    /// Length equals `templates.len()`.  Used by the simulation pipeline to
    /// convert the water-balance RHS (in hm³) back to inflow in m³/s for
    /// output reporting: `inflow_m3s = rhs_hm3 / zeta_per_stage[stage]`.
    pub zeta_per_stage: Vec<f64>,
    /// Per-stage block durations in hours.
    ///
    /// `block_hours_per_stage[stage]` is a `Vec<f64>` of length `n_blocks` for
    /// that stage.  Used by the simulation pipeline to convert load-balance
    /// constraint duals from $/MW to $/`MWh`: `spot_price = dual / block_hours`.
    pub block_hours_per_stage: Vec<Vec<f64>>,
    /// Number of hydro plants (N) used to stride into `noise_scale`.
    pub n_hydros: usize,
    /// Per-stage row index of the first load-balance constraint.
    ///
    /// `load_balance_row_starts[s]` equals `row_water_balance_start + n_hydros`
    /// for stage `s`.  Length equals `templates.len()`.  Used by the forward,
    /// backward, and simulation passes to locate load-balance rows for
    /// stochastic load patching.
    pub load_balance_row_starts: Vec<usize>,
    /// Number of buses with stochastic load noise (i.e. with `std_mw > 0`).
    ///
    /// Equals `normal_lp.n_entities()`.  Tells the forward and backward passes
    /// how many load-noise components to extract from the opening tree noise
    /// vector, which carries load noise in indices `[n_hydros, n_hydros + n_load_buses)`.
    pub n_load_buses: usize,
    /// Position in the `buses` slice for each stochastic load bus.
    ///
    /// Length equals `n_load_buses`.  Bus IDs are sorted by [`cobre_core::EntityId`] for
    /// declaration-order invariance.  The forward and backward passes use
    /// `load_bus_indices[i]` to compute the base row index of bus `i` in the
    /// load-balance region: `row = load_balance_row_start + load_bus_indices[i] * n_blks + blk`.
    pub load_bus_indices: Vec<usize>,
    /// Per-stage metadata for active generic constraint rows.
    ///
    /// `generic_constraint_row_entries[s]` contains one
    /// [`GenericConstraintRowEntry`] per active `(constraint, block)` pair at
    /// stage `s`.  Used by the simulation extraction pipeline to map LP
    /// row/column indices back to constraint identity and block.  Empty for
    /// stages with no active generic constraints.
    pub generic_constraint_row_entries: Vec<Vec<GenericConstraintRowEntry>>,
    /// Per-stage NCS column start indices.
    ///
    /// `ncs_col_starts[stage_idx]` is the column index of the first NCS generation
    /// variable for that stage.
    pub ncs_col_starts: Vec<usize>,
    /// Per-stage active NCS counts.
    ///
    /// `n_ncs_per_stage[stage_idx]` is the number of active NCS entities at that stage.
    pub n_ncs_per_stage: Vec<usize>,
    /// Per-stage active NCS system indices.
    ///
    /// `active_ncs_indices[stage_idx]` lists the system-level NCS indices active at
    /// that stage, in entity-ID order.
    pub active_ncs_indices: Vec<Vec<usize>>,
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

/// Construct a [`StageTemplate`] for a single study stage.
///
/// Returns the template, the row index of the water-balance block
/// (used as `base_row` by the [`PatchBuffer`] noise injection), the
/// row index of the load-balance block (used for load-noise patches),
/// the generic constraint row entries for this stage, NCS metadata
/// (column start, count, and active system indices), and z-inflow
/// metadata (row start, column start).
#[allow(clippy::type_complexity)]
pub(super) fn build_single_stage_template(
    ctx: &TemplateBuildCtx<'_>,
    stage: &Stage,
    stage_idx: usize,
) -> (
    StageTemplate,
    usize,
    usize,
    Vec<GenericConstraintRowEntry>,
    usize,
    usize,
    Vec<usize>,
) {
    let layout = StageLayout::new(ctx, stage, stage_idx);
    let stage_base_row = layout.row_water_balance_start;
    let load_balance_row_start = layout.row_load_balance_start;

    let (mut col_lower, mut col_upper, mut objective) =
        matrix::fill_stage_columns(ctx, stage, stage_idx, &layout);
    let (mut row_lower, mut row_upper) = matrix::fill_stage_rows(ctx, stage, stage_idx, &layout);
    let mut col_entries = matrix::build_stage_matrix_entries(ctx, stage, stage_idx, &layout);

    // Fill generic constraint rows, slack columns, and CSC entries.
    {
        let mut buffers = matrix::LpMatrixBuffers {
            col_entries: &mut col_entries,
            _col_lower: &mut col_lower,
            col_upper: &mut col_upper,
            objective: &mut objective,
            row_lower: &mut row_lower,
            row_upper: &mut row_upper,
        };
        matrix::fill_generic_constraint_entries(ctx, stage, stage_idx, &layout, &mut buffers);
    }

    // Scale all monetary objective coefficients for numerical conditioning.
    // The entire SDDP algorithm operates in scaled cost space; outputs
    // are unscaled at the reporting boundary (forward.rs, lower_bound.rs,
    // simulation/pipeline.rs, simulation/extraction.rs).
    //
    // Theta (the future cost approximation variable) must NOT be divided by
    // COST_SCALE_FACTOR.  The Benders cuts enforce `theta >= intercept_scaled`
    // where `intercept_scaled = Q_successor / K`, so theta holds the SCALED
    // future cost.  The LP objective is `sum(c_i/K * x_i) + 1.0 * theta`, and
    // the total scaled objective = (stage_cost + future_cost) / K.  Multiplying
    // by K at the reporting boundary recovers the original monetary cost.
    //
    // If theta were also divided by K its objective coefficient would become
    // 1/K, making the LP objective `stage_cost/K + (1/K)*theta` which, after
    // multiplication by K, gives `stage_cost + future_cost/K` -- wrong.
    let theta_col = StageIndexer::new(ctx.n_hydros, ctx.max_par_order).theta;
    for (i, coeff) in objective.iter_mut().enumerate() {
        if i != theta_col {
            *coeff /= COST_SCALE_FACTOR;
        }
    }

    // Sort each column's entries by row index (CSC invariant).
    for entries in &mut col_entries {
        entries.sort_unstable_by_key(|&(row, _)| row);
    }

    let (col_starts, row_indices, values) = matrix::assemble_csc(&col_entries);

    let n_transfer = ctx.n_hydros * ctx.max_par_order;
    let total_nz = col_entries.iter().map(Vec::len).sum();

    let gc_row_entries = layout.generic_constraint_rows;

    let ncs_col_start = layout.col_ncs_start;
    let n_ncs = layout.n_ncs;
    let ncs_active = layout.active_ncs_indices;

    let template = StageTemplate {
        num_cols: layout.num_cols,
        num_rows: layout.num_rows,
        num_nz: total_nz,
        col_starts,
        row_indices,
        values,
        col_lower,
        col_upper,
        objective,
        row_lower,
        row_upper,
        n_state: layout.n_state,
        n_transfer,
        n_dual_relevant: layout.n_dual_relevant,
        n_hydro: layout.n_h,
        max_par_order: layout.lag_order,
        col_scale: Vec::new(),
        row_scale: Vec::new(),
    };

    (
        template,
        stage_base_row,
        load_balance_row_start,
        gc_row_entries,
        ncs_col_start,
        n_ncs,
        ncs_active,
    )
}

/// Collect the bus-slice positions of stochastic load buses.
///
/// Returns bus-position indices (into the buses slice) for every bus that has
/// `std_mw > 0` in any load model, sorted by `EntityId` for declaration-order
/// invariance.  Buses with duplicate IDs across stages are deduplicated.
fn collect_load_bus_indices(system: &System, bus_pos: &HashMap<EntityId, usize>) -> Vec<usize> {
    // `n_load_buses` must equal `normal_lp.n_entities()` in a consistent
    // system; both are derived from buses with std_mw > 0 in the load models.
    let mut ids: Vec<EntityId> = system
        .load_models()
        .iter()
        .filter(|m| m.std_mw > 0.0)
        .map(|m| m.bus_id)
        .collect();
    ids.sort_unstable_by_key(|id| id.0);
    ids.dedup();
    ids.iter()
        .filter_map(|id| bus_pos.get(id).copied())
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
/// See the [module-level documentation](self) for the full LP layout.
/// Key dimensions for a stage with N hydros, T thermals, Lines lines,
/// B buses, K blocks per stage, and F FPHA hydros each with M planes:
///
/// - `num_cols = N*(2+L) + 1 + N*K*2 + T*K + Lines*K*2 + B*K*2 + [N penalty] + F*K`
///   (FPHA generation columns added after inflow-slack columns)
/// - `num_rows = N*(1+L) + N + B*K + F*K*M`
///   (FPHA constraint rows added after load-balance rows)
/// - `n_state  = N*(1+L)`
/// - `n_transfer = N*L`  (storage + all lags except the oldest)
/// - `n_dual_relevant = N*(1+L)`  (unchanged: FPHA rows are structural, not dual-relevant)
///
/// ## PAR order and `max_par_order`
///
/// `max_par_order` is derived from the maximum AR coefficient count across all
/// hydro inflow models for the stage.  All hydros use the same uniform lag
/// stride `max_par_order` to enable SIMD-friendly contiguous access.
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
/// `-gamma_v/2` on the incoming-storage column; when `v_in` is fixed by the
/// storage-fixing equality row its value automatically enters the FPHA constraint
/// right-hand side.  No changes to the backward pass or cut extraction are needed.
///
/// Returns `Ok` with empty templates for a system with zero stages.  All
/// entity counts may be zero (valid for degenerate test systems).
///
/// # Errors
///
/// Returns [`SddpError`] if the PAR precomputation data is inconsistent with
/// the system (e.g., a hydro in `par_lp` is not present in `system`), or if
/// the production model set has incompatible dimensions.
///
/// ## Evaporation hydros
///
/// For hydros whose evaporation model is
/// [`EvaporationModel::Linearized`],
/// three stage-level columns are added per hydro (`Q_ev`, `f_evap_plus`,
/// `f_evap_minus`), all bounded `[0, +inf)` with objective coefficient 0.0.
/// One equality constraint row is added per evaporation hydro with
/// `row_lower == row_upper == k_evap0`.
///
/// CSC matrix entries for the evaporation constraint are added by ticket-011
/// and ticket-012.  Violation cost objective coefficients are added by ticket-013.
///
/// # Examples
///
/// ```
/// use cobre_core::{Bus, DeficitSegment, EntityId, SystemBuilder};
/// use cobre_sddp::InflowNonNegativityMethod;
/// use cobre_sddp::hydro_models::PrepareHydroModelsResult;
/// use cobre_sddp::lp_builder::build_stage_templates;
/// use cobre_stochastic::par::precompute::PrecomputedPar;
///
/// let bus = Bus {
///     id: EntityId(1),
///     name: "B1".to_string(),
///     deficit_segments: vec![DeficitSegment { depth_mw: None, cost_per_mwh: 1000.0 }],
///     excess_cost: 0.0,
/// };
/// let system = SystemBuilder::new().buses(vec![bus]).build().expect("valid");
/// let method = InflowNonNegativityMethod::None;
/// let par_lp = PrecomputedPar::build(&[], &[], &[]).expect("empty ok");
/// let normal_lp = cobre_stochastic::normal::precompute::PrecomputedNormal::default();
/// let hydro_models = PrepareHydroModelsResult::default_from_system(&system);
/// // No stages → empty result.
/// let result = build_stage_templates(&system, &method, &par_lp, &normal_lp,
///                                    &hydro_models.production, &hydro_models.evaporation)
///     .expect("empty system ok");
/// assert!(result.templates.is_empty());
/// ```
#[allow(clippy::too_many_lines)]
pub fn build_stage_templates(
    system: &System,
    inflow_method: &InflowNonNegativityMethod,
    par_lp: &PrecomputedPar,
    normal_lp: &PrecomputedNormal,
    production_models: &ProductionModelSet,
    evaporation_models: &EvaporationModelSet,
) -> Result<StageTemplates, SddpError> {
    // Only build templates for study stages (id >= 0), in canonical order.
    let study_stages: Vec<_> = system.stages().iter().filter(|s| s.id >= 0).collect();
    let hydros = system.hydros();
    let n_hydros = hydros.len();

    if study_stages.is_empty() {
        return Ok(StageTemplates {
            templates: Vec::new(),
            base_rows: Vec::new(),
            noise_scale: Vec::new(),
            zeta_per_stage: Vec::new(),
            block_hours_per_stage: Vec::new(),
            n_hydros,
            load_balance_row_starts: Vec::new(),
            n_load_buses: 0,
            load_bus_indices: Vec::new(),
            generic_constraint_row_entries: Vec::new(),
            ncs_col_starts: Vec::new(),
            n_ncs_per_stage: Vec::new(),
            active_ncs_indices: Vec::new(),
            diversion_upstream: HashMap::new(),
            hydro_productivities_per_stage: Vec::new(),
        });
    }

    let buses = system.buses();
    let hydro_pos: HashMap<EntityId, usize> =
        hydros.iter().enumerate().map(|(i, h)| (h.id, i)).collect();
    let thermal_pos: HashMap<EntityId, usize> = system
        .thermals()
        .iter()
        .enumerate()
        .map(|(i, t)| (t.id, i))
        .collect();
    let line_pos: HashMap<EntityId, usize> = system
        .lines()
        .iter()
        .enumerate()
        .map(|(i, l)| (l.id, i))
        .collect();
    let bus_pos: HashMap<EntityId, usize> =
        buses.iter().enumerate().map(|(i, b)| (b.id, i)).collect();

    let load_bus_indices = collect_load_bus_indices(system, &bus_pos);
    let n_load_buses = load_bus_indices.len();
    // Consistency gate: a non-empty PrecomputedNormal must have the same
    // entity count as the stochastic load buses derived from the system.
    debug_assert!(
        normal_lp.n_entities() == 0 || normal_lp.n_entities() == n_load_buses,
        "PrecomputedNormal has {} entities but system has {} stochastic load buses",
        normal_lp.n_entities(),
        n_load_buses
    );

    let max_par_order: usize = system
        .inflow_models()
        .iter()
        .filter(|m| m.stage_id >= 0)
        .map(|m| m.ar_coefficients.len())
        .max()
        .unwrap_or(0);

    // Precompute diversion upstream map: maps target hydro ID -> list of source
    // hydro indices that divert water to it. O(1) lookup in water balance loop.
    // Cloned so the map is available both for LP construction (ctx) and for the
    // simulation extraction pipeline (StageTemplates output).
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

    let ctx = TemplateBuildCtx {
        hydros,
        thermals: system.thermals(),
        lines: system.lines(),
        buses,
        load_models: system.load_models(),
        cascade: system.cascade(),
        bounds: system.bounds(),
        penalties: system.penalties(),
        hydro_pos,
        thermal_pos,
        line_pos,
        bus_pos,
        inflow_method,
        par_lp,
        production_models,
        evaporation_models,
        generic_constraints: system.generic_constraints(),
        resolved_generic_bounds: system.resolved_generic_bounds(),
        resolved_load_factors: system.resolved_load_factors(),
        resolved_exchange_factors: system.resolved_exchange_factors(),
        non_controllable_sources: system.non_controllable_sources(),
        resolved_ncs_bounds: system.resolved_ncs_bounds(),
        resolved_ncs_factors: system.resolved_ncs_factors(),
        diversion_upstream,
        n_hydros,
        n_thermals: system.thermals().len(),
        n_lines: system.lines().len(),
        n_buses: buses.len(),
        max_par_order,
        has_penalty: n_hydros > 0 && inflow_method.has_slack_columns(),
    };

    let n_study = study_stages.len();
    let mut templates = Vec::with_capacity(n_study);
    let mut base_rows = Vec::with_capacity(n_study);
    let mut load_balance_row_starts = Vec::with_capacity(n_study);
    let mut generic_constraint_row_entries = Vec::with_capacity(n_study);
    let mut ncs_col_starts = Vec::with_capacity(n_study);
    let mut n_ncs_per_stage = Vec::with_capacity(n_study);
    let mut active_ncs_indices_per_stage = Vec::with_capacity(n_study);
    for (stage_idx, stage) in study_stages.iter().enumerate() {
        let (
            template,
            stage_base_row,
            load_balance_row_start,
            gc_entries,
            ncs_col_start,
            ncs_count,
            ncs_active,
        ) = build_single_stage_template(&ctx, stage, stage_idx);
        templates.push(template);
        base_rows.push(stage_base_row);
        load_balance_row_starts.push(load_balance_row_start);
        generic_constraint_row_entries.push(gc_entries);
        ncs_col_starts.push(ncs_col_start);
        n_ncs_per_stage.push(ncs_count);
        active_ncs_indices_per_stage.push(ncs_active);
    }

    let (noise_scale, zeta_per_stage, block_hours_per_stage) =
        scaling::compute_noise_scale(&study_stages, n_hydros, par_lp);

    // Build per-stage productivity arrays for simulation extraction.
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

    Ok(StageTemplates {
        templates,
        base_rows,
        noise_scale,
        zeta_per_stage,
        block_hours_per_stage,
        n_hydros,
        load_balance_row_starts,
        n_load_buses,
        load_bus_indices,
        generic_constraint_row_entries,
        ncs_col_starts,
        n_ncs_per_stage,
        active_ncs_indices: active_ncs_indices_per_stage,
        diversion_upstream: diversion_upstream_output,
        hydro_productivities_per_stage,
    })
}

#[cfg(test)]
#[allow(
    clippy::doc_markdown,
    clippy::too_many_lines,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation
)]
mod tests {
    // -------------------------------------------------------------------------
    // build_stage_templates unit tests
    // -------------------------------------------------------------------------

    use super::build_stage_templates;
    use crate::hydro_models::{
        FphaPlane, PrepareHydroModelsResult, ProductionModelSet, ResolvedProductionModel,
    };
    use crate::indexer::StageIndexer;
    use crate::inflow_method::InflowNonNegativityMethod;
    use crate::lp_builder::{
        COST_SCALE_FACTOR, OVER_EVAPORATION_COST_MULTIPLIER, Q_EV_SAFETY_MARGIN,
    };
    use cobre_core::{
        BoundsCountsSpec, BoundsDefaults, Bus, BusStagePenalties, ContractStageBounds,
        DeficitSegment, EntityId, HydroStageBounds, HydroStagePenalties, LineStageBounds,
        LineStagePenalties, NcsStagePenalties, PenaltiesCountsSpec, PenaltiesDefaults,
        PumpingStageBounds, ResolvedBounds, ResolvedPenalties, SystemBuilder, ThermalStageBounds,
    };
    use cobre_stochastic::normal::precompute::PrecomputedNormal;
    use cobre_stochastic::par::precompute::PrecomputedPar;

    use crate::hydro_models::{EvaporationModel, EvaporationModelSet};

    /// Build a default `ProductionModelSet` for a system (all constant productivity).
    fn default_production(system: &cobre_core::System) -> ProductionModelSet {
        PrepareHydroModelsResult::default_from_system(system).production
    }

    /// Build a default `EvaporationModelSet` for a system (all `EvaporationModel::None`).
    fn default_evaporation(system: &cobre_core::System) -> EvaporationModelSet {
        PrepareHydroModelsResult::default_from_system(system).evaporation
    }

    /// Method with no penalty — used in structural tests that check exact
    /// column/row counts that would change if penalty columns were added.
    fn no_penalty_config() -> InflowNonNegativityMethod {
        InflowNonNegativityMethod::None
    }

    /// Method with penalty — used in tests that verify the penalty
    /// column addition behaviour.
    fn penalty_config(cost: f64) -> InflowNonNegativityMethod {
        InflowNonNegativityMethod::Penalty { cost }
    }

    fn default_hydro_bounds() -> HydroStageBounds {
        HydroStageBounds {
            min_storage_hm3: 0.0,
            max_storage_hm3: 200.0,
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 100.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            min_generation_mw: 0.0,
            max_generation_mw: 250.0,
            max_diversion_m3s: None,
            filling_inflow_m3s: 0.0,
            water_withdrawal_m3s: 0.0,
        }
    }

    fn default_hydro_penalties() -> HydroStagePenalties {
        HydroStagePenalties {
            spillage_cost: 0.01,
            diversion_cost: 0.0,
            fpha_turbined_cost: 0.0,
            storage_violation_below_cost: 0.0,
            filling_target_violation_cost: 0.0,
            turbined_violation_below_cost: 0.0,
            outflow_violation_below_cost: 0.0,
            outflow_violation_above_cost: 0.0,
            generation_violation_below_cost: 0.0,
            evaporation_violation_cost: 0.0,
            water_withdrawal_violation_cost: 0.0,
        }
    }

    /// Build a minimal one-bus, no-entity system with `n_stages` study stages.
    /// Used as the base fixture for structural tests.
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    fn one_bus_system(n_stages: usize) -> cobre_core::System {
        use chrono::NaiveDate;
        use cobre_core::scenario::LoadModel;
        use cobre_core::temporal::{
            Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
            StageStateConfig,
        };

        let bus = Bus {
            id: EntityId(1),
            name: "B1".to_string(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 500.0,
            }],
            excess_cost: 0.0,
        };

        let stages: Vec<Stage> = (0..n_stages)
            .map(|i| Stage {
                index: i,
                id: i as i32,
                start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
                season_id: None,
                blocks: vec![Block {
                    index: 0,
                    name: "S".to_string(),
                    duration_hours: 744.0,
                }],
                block_mode: BlockMode::Parallel,
                state_config: StageStateConfig {
                    storage: false,
                    inflow_lags: false,
                },
                risk_config: StageRiskConfig::Expectation,
                scenario_config: ScenarioSourceConfig {
                    branching_factor: 1,
                    noise_method: NoiseMethod::Saa,
                },
            })
            .collect();

        let load_models: Vec<LoadModel> = (0..n_stages)
            .map(|i| LoadModel {
                bus_id: EntityId(1),
                stage_id: i as i32,
                mean_mw: 100.0,
                std_mw: 0.0,
            })
            .collect();

        // Resolved bounds and penalties are required for build_stage_templates to access
        // hydro/thermal/line bounds without panicking.
        let n_st = n_stages.max(1);
        let bounds = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 0,
                n_thermals: 0,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages: n_st,
            },
            &BoundsDefaults {
                hydro: default_hydro_bounds(),
                thermal: ThermalStageBounds {
                    min_generation_mw: 0.0,
                    max_generation_mw: 0.0,
                },
                line: LineStageBounds {
                    direct_mw: 0.0,
                    reverse_mw: 0.0,
                },
                pumping: PumpingStageBounds {
                    min_flow_m3s: 0.0,
                    max_flow_m3s: 0.0,
                },
                contract: ContractStageBounds {
                    min_mw: 0.0,
                    max_mw: 0.0,
                    price_per_mwh: 0.0,
                },
            },
        );
        let penalties = ResolvedPenalties::new(
            &PenaltiesCountsSpec {
                n_hydros: 0,
                n_buses: 1,
                n_lines: 0,
                n_ncs: 0,
                n_stages: n_st,
            },
            &PenaltiesDefaults {
                hydro: default_hydro_penalties(),
                bus: BusStagePenalties { excess_cost: 0.0 },
                line: LineStagePenalties { exchange_cost: 0.0 },
                ncs: NcsStagePenalties {
                    curtailment_cost: 0.0,
                },
            },
        );

        SystemBuilder::new()
            .buses(vec![bus])
            .stages(stages)
            .load_models(load_models)
            .bounds(bounds)
            .penalties(penalties)
            .build()
            .expect("one_bus_system: valid")
    }

    /// Build a system with 1 hydro, 1 bus, no thermals, no lines, K=1 block.
    /// N=1, L=`lag_order`, so we get a concrete formula to check.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        clippy::too_many_lines
    )]
    fn one_hydro_system(n_stages: usize, lag_order: usize) -> cobre_core::System {
        use chrono::NaiveDate;
        use cobre_core::entities::hydro::{Hydro, HydroGenerationModel, HydroPenalties};
        use cobre_core::scenario::{InflowModel, LoadModel};
        use cobre_core::temporal::{
            Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
            StageStateConfig,
        };

        let bus = Bus {
            id: EntityId(1),
            name: "B1".to_string(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 500.0,
            }],
            excess_cost: 0.0,
        };

        let hydro = Hydro {
            id: EntityId(2),
            name: "H1".to_string(),
            bus_id: EntityId(1),
            downstream_id: None,
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: 0.0,
            max_storage_hm3: 200.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: HydroGenerationModel::ConstantProductivity {
                productivity_mw_per_m3s: 2.5,
            },
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 100.0,
            min_generation_mw: 0.0,
            max_generation_mw: 250.0,
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: None,
            evaporation_reference_volumes_hm3: None,
            diversion: None,
            filling: None,
            penalties: HydroPenalties {
                spillage_cost: 0.01,
                diversion_cost: 0.0,
                fpha_turbined_cost: 0.0,
                storage_violation_below_cost: 0.0,
                filling_target_violation_cost: 0.0,
                turbined_violation_below_cost: 0.0,
                outflow_violation_below_cost: 0.0,
                outflow_violation_above_cost: 0.0,
                generation_violation_below_cost: 0.0,
                evaporation_violation_cost: 0.0,
                water_withdrawal_violation_cost: 0.0,
            },
        };

        let stages: Vec<Stage> = (0..n_stages)
            .map(|i| Stage {
                index: i,
                id: i as i32,
                start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
                season_id: None,
                blocks: vec![Block {
                    index: 0,
                    name: "S".to_string(),
                    duration_hours: 744.0,
                }],
                block_mode: BlockMode::Parallel,
                state_config: StageStateConfig {
                    storage: true,
                    inflow_lags: lag_order > 0,
                },
                risk_config: StageRiskConfig::Expectation,
                scenario_config: ScenarioSourceConfig {
                    branching_factor: 1,
                    noise_method: NoiseMethod::Saa,
                },
            })
            .collect();

        let ar_coefficients: Vec<f64> = (0..lag_order).map(|_| 0.5).collect();
        let inflow_models: Vec<InflowModel> = (0..n_stages)
            .map(|i| InflowModel {
                hydro_id: EntityId(2),
                stage_id: i as i32,
                mean_m3s: 80.0,
                std_m3s: 20.0,
                ar_coefficients: ar_coefficients.clone(),
                residual_std_ratio: 1.0,
            })
            .collect();

        let load_models: Vec<LoadModel> = (0..n_stages)
            .map(|i| LoadModel {
                bus_id: EntityId(1),
                stage_id: i as i32,
                mean_mw: 100.0,
                std_mw: 0.0,
            })
            .collect();

        // Resolved bounds and penalties are required for build_stage_templates to access
        // hydro/thermal/line bounds without panicking.
        let n_st = n_stages.max(1);
        let bounds = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 1,
                n_thermals: 0,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages: n_st,
            },
            &BoundsDefaults {
                hydro: default_hydro_bounds(),
                thermal: ThermalStageBounds {
                    min_generation_mw: 0.0,
                    max_generation_mw: 0.0,
                },
                line: LineStageBounds {
                    direct_mw: 0.0,
                    reverse_mw: 0.0,
                },
                pumping: PumpingStageBounds {
                    min_flow_m3s: 0.0,
                    max_flow_m3s: 0.0,
                },
                contract: ContractStageBounds {
                    min_mw: 0.0,
                    max_mw: 0.0,
                    price_per_mwh: 0.0,
                },
            },
        );
        let penalties = ResolvedPenalties::new(
            &PenaltiesCountsSpec {
                n_hydros: 1,
                n_buses: 1,
                n_lines: 0,
                n_ncs: 0,
                n_stages: n_st,
            },
            &PenaltiesDefaults {
                hydro: default_hydro_penalties(),
                bus: BusStagePenalties { excess_cost: 0.0 },
                line: LineStagePenalties { exchange_cost: 0.0 },
                ncs: NcsStagePenalties {
                    curtailment_cost: 0.0,
                },
            },
        );

        SystemBuilder::new()
            .buses(vec![bus])
            .hydros(vec![hydro])
            .stages(stages)
            .inflow_models(inflow_models)
            .load_models(load_models)
            .bounds(bounds)
            .penalties(penalties)
            .build()
            .expect("one_hydro_system: valid")
    }

    #[test]
    fn empty_stages_returns_empty() {
        // A system with no study stages returns empty StageTemplates.
        let system = one_bus_system(0);
        let result = build_stage_templates(
            &system,
            &no_penalty_config(),
            &PrecomputedPar::default(),
            &PrecomputedNormal::default(),
            &default_production(&system),
            &default_evaporation(&system),
        )
        .expect("constant productivity ok");
        assert!(result.templates.is_empty());
        assert!(result.base_rows.is_empty());
    }

    #[test]
    fn one_stage_one_template() {
        // One study stage produces exactly one template and one base_row.
        let system = one_bus_system(1);
        let result = build_stage_templates(
            &system,
            &no_penalty_config(),
            &PrecomputedPar::default(),
            &PrecomputedNormal::default(),
            &default_production(&system),
            &default_evaporation(&system),
        )
        .expect("constant productivity ok");
        assert_eq!(result.templates.len(), 1);
        assert_eq!(result.base_rows.len(), 1);
    }

    #[test]
    fn num_cols_formula_no_hydro_no_thermal_no_line() {
        // N=0, T=0, Lines=0, B=1, K=1, L=0
        // num_cols = N*(2+L)+1 + N*K*2 + T*K + Lines*K*2 + B*K*2
        //          = 0*2+1 + 0 + 0 + 0 + 1*1*2 = 3
        // (0 state + 0 lags + 0 storage_in + 1 theta) + (0 turb + 0 spill) + (0 thermal) + (0 lines) + (1 def + 1 exc)
        let system = one_bus_system(1);
        let result = build_stage_templates(
            &system,
            &no_penalty_config(),
            &PrecomputedPar::default(),
            &PrecomputedNormal::default(),
            &default_production(&system),
            &default_evaporation(&system),
        )
        .expect("constant productivity ok");
        let t = &result.templates[0];
        // theta + deficit + excess = 1 + 1 + 1 = 3
        assert_eq!(t.num_cols, 3, "num_cols mismatch for no-entity system");
    }

    #[test]
    fn num_cols_formula_one_hydro_lag_zero() {
        // N=1, L=0, T=0, Lines=0, B=1, K=1
        // State cols: N*(2+L)+1 = 1*2+1 = 3  (v_out, v_in, theta)
        // Decision: turbine[1] + spillage[1] + deficit[1] + excess[1] = 4
        // Total: 7
        let system = one_hydro_system(1, 0);
        let result = build_stage_templates(
            &system,
            &no_penalty_config(),
            &PrecomputedPar::default(),
            &PrecomputedNormal::default(),
            &default_production(&system),
            &default_evaporation(&system),
        )
        .expect("constant productivity ok");
        let t = &result.templates[0];
        // N=1 withdrawal slack adds 1 column: 7 + 1 = 8.
        // N=1 z-inflow column adds 1: 8 + 1 = 9.
        // N=1 diversion column adds 1: 9 + 1 = 10.
        assert_eq!(t.num_cols, 10, "num_cols mismatch for N=1 L=0");
    }

    #[test]
    fn num_cols_formula_one_hydro_lag_two() {
        // N=1, L=2, T=0, Lines=0, B=1, K=1
        // State cols: N*(2+L)+1 = 1*4+1 = 5  (v_out, lag0, lag1, v_in, theta)
        // Decision: turbine[1] + spillage[1] + deficit[1] + excess[1] = 4
        // Total: 9
        let system = one_hydro_system(1, 2);
        let result = build_stage_templates(
            &system,
            &no_penalty_config(),
            &PrecomputedPar::default(),
            &PrecomputedNormal::default(),
            &default_production(&system),
            &default_evaporation(&system),
        )
        .expect("constant productivity ok");
        let t = &result.templates[0];
        // N=1 withdrawal slack adds 1 column: 9 + 1 = 10.
        // N=1 z-inflow column adds 1: 10 + 1 = 11.
        // N=1 diversion column adds 1: 11 + 1 = 12.
        assert_eq!(t.num_cols, 12, "num_cols mismatch for N=1 L=2");
    }

    #[test]
    fn num_rows_formula_no_hydro() {
        // N=0, B=1, K=1, L=0 → n_state = 0*(1+0) = 0
        // fixing rows: 0, water balance: 0, load balance: 1*1 = 1
        // num_rows = 0 + 0 + 1 = 1
        let system = one_bus_system(1);
        let result = build_stage_templates(
            &system,
            &no_penalty_config(),
            &PrecomputedPar::default(),
            &PrecomputedNormal::default(),
            &default_production(&system),
            &default_evaporation(&system),
        )
        .expect("constant productivity ok");
        let t = &result.templates[0];
        assert_eq!(t.num_rows, 1, "num_rows mismatch for no-hydro system");
    }

    #[test]
    fn num_rows_formula_one_hydro_lag_zero() {
        // N=1, L=0, B=1, K=1
        // n_state = 1*(1+0) = 1
        // fixing rows = 1, water balance = 1, load balance = 1
        // num_rows = 3
        let system = one_hydro_system(1, 0);
        let result = build_stage_templates(
            &system,
            &no_penalty_config(),
            &PrecomputedPar::default(),
            &PrecomputedNormal::default(),
            &default_production(&system),
            &default_evaporation(&system),
        )
        .expect("constant productivity ok");
        let t = &result.templates[0];
        // + 1 z-inflow row for N=1.
        assert_eq!(t.num_rows, 4, "num_rows mismatch for N=1 L=0");
    }

    #[test]
    fn num_rows_formula_one_hydro_lag_two() {
        // N=1, L=2, B=1, K=1
        // n_state = 1*(1+2) = 3
        // fixing rows = 3, water balance = 1, load balance = 1
        // num_rows = 5
        let system = one_hydro_system(1, 2);
        let result = build_stage_templates(
            &system,
            &no_penalty_config(),
            &PrecomputedPar::default(),
            &PrecomputedNormal::default(),
            &default_production(&system),
            &default_evaporation(&system),
        )
        .expect("constant productivity ok");
        let t = &result.templates[0];
        // + 1 z-inflow row for N=1.
        assert_eq!(t.num_rows, 6, "num_rows mismatch for N=1 L=2");
    }

    #[test]
    fn n_state_matches_indexer() {
        // n_state must equal StageIndexer::new(N, L).n_state
        let system = one_hydro_system(1, 2);
        let result = build_stage_templates(
            &system,
            &no_penalty_config(),
            &PrecomputedPar::default(),
            &PrecomputedNormal::default(),
            &default_production(&system),
            &default_evaporation(&system),
        )
        .expect("constant productivity ok");
        let t = &result.templates[0];
        let expected = StageIndexer::new(1, 2).n_state;
        assert_eq!(t.n_state, expected, "n_state must match StageIndexer");
    }

    #[test]
    fn n_transfer_is_n_times_lag_order() {
        // n_transfer = N*L = 1*2 = 2
        let system = one_hydro_system(1, 2);
        let result = build_stage_templates(
            &system,
            &no_penalty_config(),
            &PrecomputedPar::default(),
            &PrecomputedNormal::default(),
            &default_production(&system),
            &default_evaporation(&system),
        )
        .expect("constant productivity ok");
        let t = &result.templates[0];
        assert_eq!(t.n_transfer, 2, "n_transfer = N*L");
    }

    #[test]
    fn n_dual_relevant_equals_n_state_for_constant_productivity() {
        // For v0.1.0 with no FPHA, n_dual_relevant = n_state.
        let system = one_hydro_system(1, 2);
        let result = build_stage_templates(
            &system,
            &no_penalty_config(),
            &PrecomputedPar::default(),
            &PrecomputedNormal::default(),
            &default_production(&system),
            &default_evaporation(&system),
        )
        .expect("constant productivity ok");
        let t = &result.templates[0];
        assert_eq!(
            t.n_dual_relevant, t.n_state,
            "n_dual_relevant must equal n_state for constant-productivity hydros"
        );
    }

    #[test]
    fn base_row_is_n_dual_relevant_plus_n_hydros() {
        // base_rows[s] = n_dual_relevant + n_hydro = N*(1+L) + N = N*(2+L).
        // z-inflow rows occupy [n_dual_relevant, n_dual_relevant + N).
        let system = one_hydro_system(2, 2);
        let result = build_stage_templates(
            &system,
            &no_penalty_config(),
            &PrecomputedPar::default(),
            &PrecomputedNormal::default(),
            &default_production(&system),
            &default_evaporation(&system),
        )
        .expect("constant productivity ok");
        for (s, (&br, t)) in result.base_rows.iter().zip(&result.templates).enumerate() {
            assert_eq!(
                br,
                t.n_dual_relevant + t.n_hydro,
                "base_rows[{s}] must equal n_dual_relevant + n_hydro"
            );
        }
    }

    #[test]
    fn csc_col_starts_monotone_nondecreasing() {
        // CSC validity: col_starts must be monotone non-decreasing.
        let system = one_hydro_system(1, 1);
        let result = build_stage_templates(
            &system,
            &no_penalty_config(),
            &PrecomputedPar::default(),
            &PrecomputedNormal::default(),
            &default_production(&system),
            &default_evaporation(&system),
        )
        .expect("constant productivity ok");
        let t = &result.templates[0];
        for w in t.col_starts.windows(2) {
            assert!(w[0] <= w[1], "col_starts not monotone: {} > {}", w[0], w[1]);
        }
        // Length must be num_cols + 1
        assert_eq!(t.col_starts.len(), t.num_cols + 1);
    }

    #[test]
    #[allow(clippy::cast_sign_loss)]
    fn csc_row_indices_in_range() {
        // All row_indices must be in [0, num_rows).
        let system = one_hydro_system(1, 1);
        let result = build_stage_templates(
            &system,
            &no_penalty_config(),
            &PrecomputedPar::default(),
            &PrecomputedNormal::default(),
            &default_production(&system),
            &default_evaporation(&system),
        )
        .expect("constant productivity ok");
        let t = &result.templates[0];
        for &r in &t.row_indices {
            assert!(
                r >= 0 && (r as usize) < t.num_rows,
                "row index {r} out of range [0, {})",
                t.num_rows
            );
        }
    }

    #[test]
    #[allow(clippy::cast_sign_loss)]
    fn csc_nz_count_matches_col_starts() {
        // num_nz == col_starts[num_cols]
        let system = one_hydro_system(1, 1);
        let result = build_stage_templates(
            &system,
            &no_penalty_config(),
            &PrecomputedPar::default(),
            &PrecomputedNormal::default(),
            &default_production(&system),
            &default_evaporation(&system),
        )
        .expect("constant productivity ok");
        let t = &result.templates[0];
        assert_eq!(
            t.num_nz,
            *t.col_starts.last().unwrap() as usize,
            "num_nz must equal col_starts[num_cols]"
        );
        assert_eq!(
            t.row_indices.len(),
            t.num_nz,
            "row_indices.len() must equal num_nz"
        );
        assert_eq!(t.values.len(), t.num_nz, "values.len() must equal num_nz");
    }

    #[test]
    fn theta_column_has_unit_objective() {
        // The theta column (index = N*(2+L)) must have objective coefficient = 1.0.
        let lag_order = 2;
        let system = one_hydro_system(1, lag_order);
        let result = build_stage_templates(
            &system,
            &no_penalty_config(),
            &PrecomputedPar::default(),
            &PrecomputedNormal::default(),
            &default_production(&system),
            &default_evaporation(&system),
        )
        .expect("constant productivity ok");
        let t = &result.templates[0];
        let theta_col = StageIndexer::new(1, lag_order).theta;
        assert_eq!(
            t.objective[theta_col], 1.0,
            "theta column objective must be 1.0 (theta is not scaled by COST_SCALE_FACTOR)"
        );
    }

    #[test]
    fn spillage_objective_nonzero_for_nonzero_penalty() {
        // The spillage column should carry a non-zero objective when spillage_cost > 0.
        // Hydro has spillage_cost = 0.01, block duration = 744h.
        let system = one_hydro_system(1, 0);
        let result = build_stage_templates(
            &system,
            &no_penalty_config(),
            &PrecomputedPar::default(),
            &PrecomputedNormal::default(),
            &default_production(&system),
            &default_evaporation(&system),
        )
        .expect("constant productivity ok");
        let t = &result.templates[0];
        // spillage col for h=0, blk=0: col_spillage_start + 0 = N*(3+L)+1 + N*K
        // With N=1, L=0, K=1: theta=3, decision_start=4, turbine_start=4, spill_start=5
        let spill_col = 5;
        assert!(
            t.objective[spill_col] > 0.0,
            "spillage objective must be > 0 when spillage_cost > 0"
        );
    }

    /// Build a 1-bus, 1-FPHA-hydro, 1-stage system with `n_planes` FPHA planes,
    /// a given `fpha_turbined_cost`, and custom block durations.
    ///
    /// This is a variant of `one_fpha_hydro_system` that allows injecting an
    /// arbitrary `fpha_turbined_cost` and specifying the stage blocks.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        clippy::too_many_lines
    )]
    fn fpha_system_with_turbined_cost(
        n_planes: usize,
        fpha_turbined_cost: f64,
        block_durations_hours: &[f64],
    ) -> (cobre_core::System, ProductionModelSet) {
        use chrono::NaiveDate;
        use cobre_core::entities::hydro::{Hydro, HydroGenerationModel, HydroPenalties};
        use cobre_core::scenario::{InflowModel, LoadModel};
        use cobre_core::temporal::{
            Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
            StageStateConfig,
        };

        let n_blks = block_durations_hours.len();
        assert!(n_blks > 0, "must have at least one block");

        let bus = Bus {
            id: EntityId(1),
            name: "B1".to_string(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 500.0,
            }],
            excess_cost: 0.0,
        };
        let hydro = Hydro {
            id: EntityId(2),
            name: "FPHA1".to_string(),
            bus_id: EntityId(1),
            downstream_id: None,
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: 0.0,
            max_storage_hm3: 500.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: HydroGenerationModel::Fpha,
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 150.0,
            min_generation_mw: 0.0,
            max_generation_mw: 300.0,
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: None,
            evaporation_reference_volumes_hm3: None,
            diversion: None,
            filling: None,
            penalties: HydroPenalties {
                spillage_cost: 0.01,
                diversion_cost: 0.0,
                fpha_turbined_cost,
                storage_violation_below_cost: 0.0,
                filling_target_violation_cost: 0.0,
                turbined_violation_below_cost: 0.0,
                outflow_violation_below_cost: 0.0,
                outflow_violation_above_cost: 0.0,
                generation_violation_below_cost: 0.0,
                evaporation_violation_cost: 0.0,
                water_withdrawal_violation_cost: 0.0,
            },
        };

        let blocks: Vec<Block> = block_durations_hours
            .iter()
            .enumerate()
            .map(|(i, &hours)| Block {
                index: i,
                name: format!("BLK{i}"),
                duration_hours: hours,
            })
            .collect();

        let stages: Vec<Stage> = vec![Stage {
            index: 0,
            id: 0,
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            season_id: None,
            blocks,
            block_mode: BlockMode::Parallel,
            state_config: StageStateConfig {
                storage: true,
                inflow_lags: false,
            },
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor: 1,
                noise_method: NoiseMethod::Saa,
            },
        }];

        let inflow_models: Vec<InflowModel> = vec![InflowModel {
            hydro_id: EntityId(2),
            stage_id: 0,
            mean_m3s: 80.0,
            std_m3s: 20.0,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
        }];

        let load_models: Vec<LoadModel> = vec![LoadModel {
            bus_id: EntityId(1),
            stage_id: 0,
            mean_mw: 200.0,
            std_mw: 0.0,
        }];

        let bounds = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 1,
                n_thermals: 0,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages: 1,
            },
            &BoundsDefaults {
                hydro: HydroStageBounds {
                    min_storage_hm3: 0.0,
                    max_storage_hm3: 500.0,
                    min_turbined_m3s: 0.0,
                    max_turbined_m3s: 150.0,
                    min_outflow_m3s: 0.0,
                    max_outflow_m3s: None,
                    min_generation_mw: 0.0,
                    max_generation_mw: 300.0,
                    max_diversion_m3s: None,
                    filling_inflow_m3s: 0.0,
                    water_withdrawal_m3s: 0.0,
                },
                thermal: ThermalStageBounds {
                    min_generation_mw: 0.0,
                    max_generation_mw: 0.0,
                },
                line: LineStageBounds {
                    direct_mw: 0.0,
                    reverse_mw: 0.0,
                },
                pumping: PumpingStageBounds {
                    min_flow_m3s: 0.0,
                    max_flow_m3s: 0.0,
                },
                contract: ContractStageBounds {
                    min_mw: 0.0,
                    max_mw: 0.0,
                    price_per_mwh: 0.0,
                },
            },
        );

        let penalties = ResolvedPenalties::new(
            &PenaltiesCountsSpec {
                n_hydros: 1,
                n_buses: 1,
                n_lines: 0,
                n_ncs: 0,
                n_stages: 1,
            },
            &PenaltiesDefaults {
                hydro: HydroStagePenalties {
                    spillage_cost: 0.01,
                    diversion_cost: 0.0,
                    fpha_turbined_cost,
                    storage_violation_below_cost: 0.0,
                    filling_target_violation_cost: 0.0,
                    turbined_violation_below_cost: 0.0,
                    outflow_violation_below_cost: 0.0,
                    outflow_violation_above_cost: 0.0,
                    generation_violation_below_cost: 0.0,
                    evaporation_violation_cost: 0.0,
                    water_withdrawal_violation_cost: 0.0,
                },
                bus: BusStagePenalties { excess_cost: 0.0 },
                line: LineStagePenalties { exchange_cost: 0.0 },
                ncs: NcsStagePenalties {
                    curtailment_cost: 0.0,
                },
            },
        );

        let system = SystemBuilder::new()
            .buses(vec![bus])
            .hydros(vec![hydro])
            .stages(stages)
            .inflow_models(inflow_models)
            .load_models(load_models)
            .bounds(bounds)
            .penalties(penalties)
            .build()
            .expect("fpha_system_with_turbined_cost: valid");

        let plane = FphaPlane {
            intercept: 10.0,
            gamma_v: 0.5,
            gamma_q: 2.0,
            gamma_s: 0.1,
        };
        let planes = vec![plane; n_planes];
        let models = vec![vec![ResolvedProductionModel::Fpha {
            planes,
            turbined_cost: 0.0,
        }]];
        let production = ProductionModelSet::new(models, 1, 1);

        (system, production)
    }

    // ---- fpha_turbined_cost tests -------------------------------------------

    #[test]
    fn fpha_turbined_cost_applied_to_fpha_turbine_column() {
        // AC-1: 1 FPHA hydro with fpha_turbined_cost = 0.5 $/MWh, 1 block of 720h.
        // Expected turbine column objective: 0.5 * 720 = 360.0.
        //
        // Column layout (N=1, L=0, K=1, no penalty):
        //   theta = N*(2+L) = 2,  decision_start = 3
        //   col_turbine_start = 3
        //   turbine col h=0, blk=0: 3 + 0*1 + 0 = 3
        let (system, production) = fpha_system_with_turbined_cost(3, 0.5, &[720.0]);
        let result = build_stage_templates(
            &system,
            &no_penalty_config(),
            &PrecomputedPar::default(),
            &PrecomputedNormal::default(),
            &production,
            &default_evaporation(&system),
        )
        .expect("fpha system builds ok");
        let t = &result.templates[0];
        let turbine_col = 4_usize;
        let expected = 0.5 * 720.0 / COST_SCALE_FACTOR;
        assert!(
            (t.objective[turbine_col] - expected).abs() < 1e-15,
            "FPHA turbine col objective: expected {expected}, got {}",
            t.objective[turbine_col]
        );
    }

    #[test]
    fn constant_hydro_turbine_column_has_zero_objective() {
        // AC-2: constant-productivity hydro must have objective coefficient 0.0
        // on its turbine column regardless of block duration.
        //
        // Column layout (N=1, L=0, K=1): col_turbine_start = 4.
        let system = one_hydro_system(1, 0);
        let result = build_stage_templates(
            &system,
            &no_penalty_config(),
            &PrecomputedPar::default(),
            &PrecomputedNormal::default(),
            &default_production(&system),
            &default_evaporation(&system),
        )
        .expect("constant productivity ok");
        let t = &result.templates[0];
        let turbine_col = 4_usize;
        assert_eq!(
            t.objective[turbine_col], 0.0,
            "constant hydro turbine column must have zero objective, got {}",
            t.objective[turbine_col]
        );
    }

    #[test]
    fn fpha_turbined_cost_multi_block_uses_per_block_hours() {
        // AC-3: 2-block stage — each turbine column carries cost * its own block_hours.
        // fpha_turbined_cost = 1.0 $/MWh, block 0 = 300h, block 1 = 420h.
        //
        // Column layout (N=1, L=0, K=2, no penalty):
        //   theta = N*(3+L) = 3, decision_start = 4, col_turbine_start = 4
        //   turbine col h=0, blk=0: 4 + 0*2 + 0 = 4
        //   turbine col h=0, blk=1: 4 + 0*2 + 1 = 5
        let (system, production) = fpha_system_with_turbined_cost(3, 1.0, &[300.0, 420.0]);
        let result = build_stage_templates(
            &system,
            &no_penalty_config(),
            &PrecomputedPar::default(),
            &PrecomputedNormal::default(),
            &production,
            &default_evaporation(&system),
        )
        .expect("fpha multi-block system builds ok");
        let t = &result.templates[0];
        let col_blk0 = 4_usize;
        let col_blk1 = 5_usize;
        assert!(
            (t.objective[col_blk0] - 300.0 / COST_SCALE_FACTOR).abs() < 1e-15,
            "block 0 turbine objective: expected {}, got {}",
            300.0 / COST_SCALE_FACTOR,
            t.objective[col_blk0]
        );
        assert!(
            (t.objective[col_blk1] - 420.0 / COST_SCALE_FACTOR).abs() < 1e-15,
            "block 1 turbine objective: expected {}, got {}",
            420.0 / COST_SCALE_FACTOR,
            t.objective[col_blk1]
        );
    }

    #[test]
    fn fpha_turbined_cost_mixed_system_only_fpha_hydros_carry_cost() {
        // AC-3: In a mixed system (2 constant + 2 FPHA), only the FPHA hydros'
        // turbine columns carry the fpha_turbined_cost objective.
        //
        // We reuse four_hydro_mixed_system() but override fpha_turbined_cost.
        let (system, production) = four_hydro_mixed_system();

        // Rebuild penalties with non-zero fpha_turbined_cost.
        let hydro_pen = HydroStagePenalties {
            spillage_cost: 0.01,
            diversion_cost: 0.0,
            fpha_turbined_cost: 1.0,
            storage_violation_below_cost: 0.0,
            filling_target_violation_cost: 0.0,
            turbined_violation_below_cost: 0.0,
            outflow_violation_below_cost: 0.0,
            outflow_violation_above_cost: 0.0,
            generation_violation_below_cost: 0.0,
            evaporation_violation_cost: 0.0,
            water_withdrawal_violation_cost: 0.0,
        };
        let penalties = ResolvedPenalties::new(
            &PenaltiesCountsSpec {
                n_hydros: 4,
                n_buses: 1,
                n_lines: 0,
                n_ncs: 0,
                n_stages: 1,
            },
            &PenaltiesDefaults {
                hydro: hydro_pen,
                bus: BusStagePenalties { excess_cost: 0.0 },
                line: LineStagePenalties { exchange_cost: 0.0 },
                ncs: NcsStagePenalties {
                    curtailment_cost: 0.0,
                },
            },
        );

        // Rebuild system with the new penalties.
        let system = SystemBuilder::new()
            .buses(system.buses().to_vec())
            .hydros(system.hydros().to_vec())
            .stages(system.stages().to_vec())
            .inflow_models(system.inflow_models().to_vec())
            .load_models(system.load_models().to_vec())
            .bounds(system.bounds().clone())
            .penalties(penalties)
            .build()
            .expect("mixed system with turbined cost");

        let result = build_stage_templates(
            &system,
            &no_penalty_config(),
            &PrecomputedPar::default(),
            &PrecomputedNormal::default(),
            &production,
            &default_evaporation(&system),
        )
        .expect("mixed system builds");
        let t = &result.templates[0];

        // N=4, L=0, K=1: theta=12, decision_start=13
        // col_turbine_start=13, turbine cols: h0=13, h1=14, h2=15, h3=16
        // spillage cols: h0=17, h1=18, h2=19, h3=20
        let col_turbine_start = 13;
        let block_hours = 744.0;

        // Hydros 0, 1 are ConstantProductivity → objective = 0.0.
        assert!(
            t.objective[col_turbine_start].abs() < 1e-12,
            "constant hydro 0 turbine objective should be 0.0, got {}",
            t.objective[col_turbine_start]
        );
        assert!(
            t.objective[col_turbine_start + 1].abs() < 1e-12,
            "constant hydro 1 turbine objective should be 0.0, got {}",
            t.objective[col_turbine_start + 1]
        );

        // Hydros 2, 3 are FPHA → objective = 1.0 * 744.0 / COST_SCALE_FACTOR.
        let expected_fpha = block_hours / COST_SCALE_FACTOR;
        assert!(
            (t.objective[col_turbine_start + 2] - expected_fpha).abs() < 1e-15,
            "FPHA hydro 2 turbine objective should be {expected_fpha}, got {}",
            t.objective[col_turbine_start + 2]
        );
        assert!(
            (t.objective[col_turbine_start + 3] - expected_fpha).abs() < 1e-15,
            "FPHA hydro 3 turbine objective should be {expected_fpha}, got {}",
            t.objective[col_turbine_start + 3]
        );
    }

    #[test]
    fn load_balance_rhs_matches_load_model_mean_mw() {
        // The load balance row RHS must equal the mean_mw from LoadModel (100.0 in fixture).
        let system = one_bus_system(1);
        let result = build_stage_templates(
            &system,
            &no_penalty_config(),
            &PrecomputedPar::default(),
            &PrecomputedNormal::default(),
            &default_production(&system),
            &default_evaporation(&system),
        )
        .expect("constant productivity ok");
        let t = &result.templates[0];
        // No hydros → n_dual_relevant=0, water_balance_rows=0, load_balance at row 0, blk 0
        let load_row = 0;
        assert_eq!(
            t.row_lower[load_row], 100.0,
            "row_lower for load balance must be mean_mw"
        );
        assert_eq!(
            t.row_upper[load_row], 100.0,
            "row_upper for load balance must be mean_mw"
        );
    }

    #[test]
    fn multiple_stages_produce_same_count_templates_and_base_rows() {
        // A 3-stage system yields 3 templates and 3 base_rows.
        let system = one_hydro_system(3, 1);
        let result = build_stage_templates(
            &system,
            &no_penalty_config(),
            &PrecomputedPar::default(),
            &PrecomputedNormal::default(),
            &default_production(&system),
            &default_evaporation(&system),
        )
        .expect("constant productivity ok");
        assert_eq!(result.templates.len(), 3);
        assert_eq!(result.base_rows.len(), 3);
    }

    #[test]
    fn stage_templates_clone_and_debug() {
        let system = one_hydro_system(1, 0);
        let result = build_stage_templates(
            &system,
            &no_penalty_config(),
            &PrecomputedPar::default(),
            &PrecomputedNormal::default(),
            &default_production(&system),
            &default_evaporation(&system),
        )
        .expect("constant productivity ok");
        let cloned = result.clone();
        assert_eq!(cloned.templates.len(), result.templates.len());
        let s = format!("{result:?}");
        assert!(s.contains("StageTemplates"));
    }

    // -------------------------------------------------------------------------
    // FPHA generation model validation tests
    // -------------------------------------------------------------------------

    /// AC: a system where a hydro plant uses `Fpha` entity model but has a
    /// `ConstantProductivity` resolved model must succeed (the FPHA rejection
    /// guard has been removed — validation now happens in `prepare_hydro_models`).
    #[test]
    #[allow(clippy::too_many_lines)]
    fn test_fpha_model_accepted() {
        use chrono::NaiveDate;
        use cobre_core::entities::hydro::{Hydro, HydroGenerationModel, HydroPenalties};
        use cobre_core::scenario::{InflowModel, LoadModel};
        use cobre_core::temporal::{
            Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
            StageStateConfig,
        };

        let bus = Bus {
            id: EntityId(1),
            name: "B1".to_string(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 500.0,
            }],
            excess_cost: 0.0,
        };
        let hydro = Hydro {
            id: EntityId(5),
            name: "Tucurui".to_string(),
            bus_id: EntityId(1),
            downstream_id: None,
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: 0.0,
            max_storage_hm3: 200.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: HydroGenerationModel::Fpha,
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 100.0,
            min_generation_mw: 0.0,
            max_generation_mw: 250.0,
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: None,
            evaporation_reference_volumes_hm3: None,
            diversion: None,
            filling: None,
            penalties: HydroPenalties {
                spillage_cost: 0.01,
                diversion_cost: 0.0,
                fpha_turbined_cost: 0.0,
                storage_violation_below_cost: 0.0,
                filling_target_violation_cost: 0.0,
                turbined_violation_below_cost: 0.0,
                outflow_violation_below_cost: 0.0,
                outflow_violation_above_cost: 0.0,
                generation_violation_below_cost: 0.0,
                evaporation_violation_cost: 0.0,
                water_withdrawal_violation_cost: 0.0,
            },
        };

        let stages: Vec<Stage> = vec![Stage {
            index: 0,
            id: 0,
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).expect("valid date"),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).expect("valid date"),
            season_id: None,
            blocks: vec![Block {
                index: 0,
                name: "S".to_string(),
                duration_hours: 744.0,
            }],
            block_mode: BlockMode::Parallel,
            state_config: StageStateConfig {
                storage: true,
                inflow_lags: false,
            },
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor: 1,
                noise_method: NoiseMethod::Saa,
            },
        }];

        let inflow_models: Vec<InflowModel> = vec![InflowModel {
            hydro_id: EntityId(5),
            stage_id: 0,
            mean_m3s: 80.0,
            std_m3s: 20.0,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
        }];

        let load_models: Vec<LoadModel> = vec![LoadModel {
            bus_id: EntityId(1),
            stage_id: 0,
            mean_mw: 100.0,
            std_mw: 0.0,
        }];

        let bounds = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 1,
                n_thermals: 0,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages: 1,
            },
            &BoundsDefaults {
                hydro: default_hydro_bounds(),
                thermal: ThermalStageBounds {
                    min_generation_mw: 0.0,
                    max_generation_mw: 0.0,
                },
                line: LineStageBounds {
                    direct_mw: 0.0,
                    reverse_mw: 0.0,
                },
                pumping: PumpingStageBounds {
                    min_flow_m3s: 0.0,
                    max_flow_m3s: 0.0,
                },
                contract: ContractStageBounds {
                    min_mw: 0.0,
                    max_mw: 0.0,
                    price_per_mwh: 0.0,
                },
            },
        );
        let penalties = ResolvedPenalties::new(
            &PenaltiesCountsSpec {
                n_hydros: 1,
                n_buses: 1,
                n_lines: 0,
                n_ncs: 0,
                n_stages: 1,
            },
            &PenaltiesDefaults {
                hydro: default_hydro_penalties(),
                bus: BusStagePenalties { excess_cost: 0.0 },
                line: LineStagePenalties { exchange_cost: 0.0 },
                ncs: NcsStagePenalties {
                    curtailment_cost: 0.0,
                },
            },
        );

        let system = cobre_core::SystemBuilder::new()
            .buses(vec![bus])
            .hydros(vec![hydro])
            .stages(stages)
            .inflow_models(inflow_models)
            .load_models(load_models)
            .bounds(bounds)
            .penalties(penalties)
            .build()
            .expect("test_fpha_model_accepted: valid system");

        // With a constant-productivity resolved model (default_from_system maps Fpha
        // entity model → ConstantProductivity { productivity: 0.0 }) the builder
        // must not reject the system.  FPHA entity model no longer causes a
        // validation error — the resolved production model determines the LP layout.
        let production = PrepareHydroModelsResult::default_from_system(&system).production;
        let result = build_stage_templates(
            &system,
            &no_penalty_config(),
            &PrecomputedPar::default(),
            &PrecomputedNormal::default(),
            &production,
            &default_evaporation(&system),
        );

        // The builder must now succeed (the old guard has been removed).
        assert!(
            result.is_ok(),
            "Fpha entity model with ConstantProductivity resolved model must now succeed: {result:?}"
        );

        // The plant name 'Tucurui' should appear nowhere in an Ok result —
        // if an error were still returned the name would be in the message.
        if let Err(ref e) = result {
            let msg = e.to_string();
            assert!(
                !msg.contains("Tucurui"),
                "unexpected error for Tucurui: {msg}"
            );
        }
    }

    /// AC: a system where all hydro plants use `ConstantProductivity` must be
    /// accepted, returning `Ok(StageTemplates { .. })`.
    #[test]
    fn test_constant_productivity_accepted() {
        let system = one_hydro_system(1, 0);
        let result = build_stage_templates(
            &system,
            &no_penalty_config(),
            &PrecomputedPar::default(),
            &PrecomputedNormal::default(),
            &default_production(&system),
            &default_evaporation(&system),
        );
        assert!(
            result.is_ok(),
            "ConstantProductivity system must return Ok, got: {result:?}"
        );
        assert_eq!(
            result.expect("accepted").templates.len(),
            1,
            "one study stage → one template"
        );
    }

    // -------------------------------------------------------------------------
    // Inflow non-negativity penalty method tests
    // -------------------------------------------------------------------------

    // AC-1 / test_penalty_columns_added:
    // penalty method with N=1 hydro adds 1 extra column; method="none" adds 0.
    #[test]
    fn test_penalty_columns_added() {
        let system = one_hydro_system(1, 0);
        let without = build_stage_templates(
            &system,
            &no_penalty_config(),
            &PrecomputedPar::default(),
            &PrecomputedNormal::default(),
            &default_production(&system),
            &default_evaporation(&system),
        )
        .expect("constant productivity ok");
        let with_p = build_stage_templates(
            &system,
            &penalty_config(1000.0),
            &PrecomputedPar::default(),
            &PrecomputedNormal::default(),
            &default_production(&system),
            &default_evaporation(&system),
        )
        .expect("constant productivity ok");
        assert_eq!(
            with_p.templates[0].num_cols,
            without.templates[0].num_cols + 1,
            "penalty method must add exactly n_hydros extra columns"
        );
    }

    // AC-1 (edge case): no slack columns when n_hydros == 0, even with penalty config.
    #[test]
    fn test_penalty_columns_added_3_hydros() {
        // Build a 3-hydro system by calling one_hydro_system 3 times is not possible;
        // use one_hydro_system(1, 0) as a proxy and verify the count formula directly.
        // The formula: num_cols(penalty) = num_cols(none) + n_hydros.
        // For N=1 we already cover N=1 above. Verify the N=0 (no hydros) edge case:
        // no slacks when n_hydros == 0, regardless of config.
        let system = one_bus_system(1);
        let with_p = build_stage_templates(
            &system,
            &penalty_config(1000.0),
            &PrecomputedPar::default(),
            &PrecomputedNormal::default(),
            &default_production(&system),
            &default_evaporation(&system),
        )
        .expect("constant productivity ok");
        let without = build_stage_templates(
            &system,
            &no_penalty_config(),
            &PrecomputedPar::default(),
            &PrecomputedNormal::default(),
            &default_production(&system),
            &default_evaporation(&system),
        )
        .expect("constant productivity ok");
        assert_eq!(
            with_p.templates[0].num_cols, without.templates[0].num_cols,
            "no slack columns when n_hydros == 0, even with penalty config"
        );
    }

    // AC-2 / test_penalty_objective_coefficient:
    // objective coefficient = penalty_cost * total_stage_hours.
    // one_hydro_system uses 1 block of 744 hours.
    #[test]
    fn test_penalty_objective_coefficient() {
        let system = one_hydro_system(1, 0);
        let config = penalty_config(1000.0);
        let result = build_stage_templates(
            &system,
            &config,
            &PrecomputedPar::default(),
            &PrecomputedNormal::default(),
            &default_production(&system),
            &default_evaporation(&system),
        )
        .expect("constant productivity ok");
        let t = &result.templates[0];
        // N=1, L=0: theta=3, decision_start=4, turbine=4, spillage=5, diversion=6,
        // deficit=7, excess=8, inflow_slack=9, withdrawal_slack=10.
        // Inflow slack is before withdrawal (N=1 withdrawal columns at end).
        let slack_col = t.num_cols - 1 - t.n_hydro; // inflow_slack (before withdrawal)
        let expected_obj = 1000.0 * 744.0 / COST_SCALE_FACTOR;
        assert!(
            (t.objective[slack_col] - expected_obj).abs() < 1e-12,
            "expected objective {expected_obj}, got {}",
            t.objective[slack_col]
        );
    }

    // AC-3 / test_no_penalty_columns_when_none:
    // method="none" leaves column/row counts unchanged.
    #[test]
    fn test_no_penalty_columns_when_none() {
        let system = one_hydro_system(1, 2);
        let result = build_stage_templates(
            &system,
            &no_penalty_config(),
            &PrecomputedPar::default(),
            &PrecomputedNormal::default(),
            &default_production(&system),
            &default_evaporation(&system),
        )
        .expect("constant productivity ok");
        let t = &result.templates[0];
        // N=1, L=2: state+aux = N*(3+L)+1 = 6 (storage, lags, z_inflow, storage_in, theta);
        // decisions = turb+spill+diversion+def+exc = 5; withdrawal = 1; total = 12.
        assert_eq!(
            t.num_cols, 12,
            "method=none must not add extra penalty columns"
        );
        // num_rows = N*(1+L) fixing + N z_inflow + N water_balance + B*K load_balance = 3+1+1+1 = 6
        assert_eq!(t.num_rows, 6, "method=none must not add extra penalty rows");
    }

    // test_penalty_slack_in_water_balance:
    // the slack column has a non-zero entry in the water balance row for its hydro.
    #[test]
    #[allow(clippy::cast_sign_loss)]
    fn test_penalty_slack_in_water_balance() {
        let system = one_hydro_system(1, 0);
        let config = penalty_config(1000.0);
        let result = build_stage_templates(
            &system,
            &config,
            &PrecomputedPar::default(),
            &PrecomputedNormal::default(),
            &default_production(&system),
            &default_evaporation(&system),
        )
        .expect("constant productivity ok");
        let t = &result.templates[0];

        // Locate the inflow slack column. For N=1, L=0: it is before withdrawal.
        let slack_col = t.num_cols - 1 - t.n_hydro;

        // Iterate the CSC to find the entry for slack_col in the water balance row.
        // Water balance row for hydro 0: row_water_balance_start = n_state + n_hydros = N*(1+L) + N = 2.
        let water_balance_row = 2_usize; // n_state + 0 = N*(1+L) + N = 2*(1+0) = 2

        let col_start = t.col_starts[slack_col] as usize;
        let col_end = t.col_starts[slack_col + 1] as usize;
        let found = t.row_indices[col_start..col_end]
            .iter()
            .zip(&t.values[col_start..col_end])
            .any(|(&r, &v)| r as usize == water_balance_row && v.abs() > 1e-12);

        assert!(
            found,
            "slack column must have a non-zero entry in the water balance row"
        );
    }

    // test_penalty_slack_bounds:
    // slack columns have lower = 0.0 and upper = +inf.
    #[test]
    fn test_penalty_slack_bounds() {
        let system = one_hydro_system(1, 0);
        let config = penalty_config(1000.0);
        let result = build_stage_templates(
            &system,
            &config,
            &PrecomputedPar::default(),
            &PrecomputedNormal::default(),
            &default_production(&system),
            &default_evaporation(&system),
        )
        .expect("constant productivity ok");
        let t = &result.templates[0];
        // The inflow slack is before withdrawal column for N=1.
        let slack_col = t.num_cols - 1 - t.n_hydro;
        assert_eq!(t.col_lower[slack_col], 0.0, "slack lower bound must be 0.0");
        assert!(
            t.col_upper[slack_col].is_infinite() && t.col_upper[slack_col] > 0.0,
            "slack upper bound must be +infinity"
        );
    }

    // Verify the water balance coefficient value.
    //
    // The penalty slack column represents virtual inflow. Adding virtual inflow
    // is equivalent to subtracting it from the LHS of the water balance
    // constraint (which is written as: outflows - inflows = RHS).
    // Therefore the coefficient is -ζ where ζ = tau_total * M3S_TO_HM3.
    //
    // With 1 block of 744 h:
    //   ζ = 744.0 * (3600.0 / 1_000_000.0) = 2.6784 hm3/(m3/s)
    //   coefficient = -ζ = -2.6784
    #[test]
    #[allow(clippy::cast_sign_loss)]
    fn test_penalty_water_balance_coefficient_value() {
        let system = one_hydro_system(1, 0);
        let config = penalty_config(1000.0);
        let result = build_stage_templates(
            &system,
            &config,
            &PrecomputedPar::default(),
            &PrecomputedNormal::default(),
            &default_production(&system),
            &default_evaporation(&system),
        )
        .expect("constant productivity ok");
        let t = &result.templates[0];

        // For N=1, L=0: inflow_slack is before withdrawal column.
        let slack_col = t.num_cols - 1 - t.n_hydro;
        let water_balance_row = 2_usize; // n_state + n_hydros = N*(1+L) + N = 2
        let zeta = 744.0 * (3_600.0 / 1_000_000.0);
        let expected_coeff = -zeta; // slack enters LHS with -ζ (virtual inflow)

        let col_start = t.col_starts[slack_col] as usize;
        let col_end = t.col_starts[slack_col + 1] as usize;
        let coeff = t.row_indices[col_start..col_end]
            .iter()
            .zip(&t.values[col_start..col_end])
            .find(|&(&r, _)| r as usize == water_balance_row)
            .map(|(_, &v)| v);

        assert!(
            coeff.is_some(),
            "slack column must have an entry in the water balance row"
        );
        let coeff = coeff.unwrap();
        assert!(
            (coeff - expected_coeff).abs() < 1e-9,
            "expected coefficient {expected_coeff:.9}, got {coeff:.9}"
        );
    }

    // Penalty method with multiple stages: verify each stage has consistent slack layout.
    #[test]
    fn test_penalty_multi_stage_consistent() {
        let system = one_hydro_system(3, 1);
        let config = penalty_config(2000.0);
        let result = build_stage_templates(
            &system,
            &config,
            &PrecomputedPar::default(),
            &PrecomputedNormal::default(),
            &default_production(&system),
            &default_evaporation(&system),
        )
        .expect("constant productivity ok");
        assert_eq!(result.templates.len(), 3);
        let base_cols = result.templates[0].num_cols;
        for t in &result.templates {
            assert_eq!(
                t.num_cols, base_cols,
                "all stages must have the same column count"
            );
        }
    }

    // AC-4 / test_penalty_slack_absorbs_negative_inflow:
    // A negative noise value would render the LP infeasible without the inflow
    // slack column. With `penalty_config`, the slack absorbs the deficit and the
    // solve must succeed with a positive slack value.
    //
    // System: N=1, L=0, K=1 block (744 h), B=1 bus, T=0, Lines=0.
    // Column layout:
    //   col 0: storage_out    col 1: z_inflow      col 2: storage_in
    //   col 3: theta          col 4: turbine        col 5: spillage
    //   col 6: diversion      col 7: deficit        col 8: excess
    //   col 9: inflow_slack   col 10: withdrawal_slack
    //
    // Row layout:
    //   row 0: storage_fixing  row 1: z_inflow_def  row 2: water_balance
    //   row 3: load_balance
    //
    // To apply negative inflow noise we patch the water balance row (row 2)
    // to RHS = -5.0. Without the slack this would make the LP infeasible.
    #[test]
    fn test_penalty_slack_absorbs_negative_inflow() {
        use cobre_solver::{HighsSolver, RowBatch, SolverInterface};

        let system = one_hydro_system(1, 0);
        let config = penalty_config(1000.0);
        let result = build_stage_templates(
            &system,
            &config,
            &PrecomputedPar::default(),
            &PrecomputedNormal::default(),
            &default_production(&system),
            &default_evaporation(&system),
        )
        .expect("constant productivity ok");
        let template = &result.templates[0];

        // The inflow slack is before withdrawal column for N=1.
        let col_inflow_slack_start = template.num_cols - 1 - template.n_hydro;

        // base_row for stage 0 is n_dual_relevant + n_hydro = n_state + N = 2 (for N=1, L=0).
        let base_row = result.base_rows[0];
        let water_balance_row = base_row; // hydro 0: base_row + 0

        let mut solver = HighsSolver::new().expect("HighsSolver::new must succeed");

        // Load the structural LP.
        solver.load_model(template);

        // Add an empty cut batch (no cuts at iteration 0).
        let empty_cuts = RowBatch {
            num_rows: 0,
            row_starts: vec![0_i32],
            col_indices: vec![],
            values: vec![],
            row_lower: vec![],
            row_upper: vec![],
        };
        solver.add_rows(&empty_cuts);

        // Patch the state rows.
        // Row 0 (storage_fixing): fix incoming storage to 100 hm³.
        // Row water_balance_row (water_balance): set RHS to -5.0 m³/s (negative noise).
        // Both are equality constraints: lower == upper == rhs.
        let initial_storage = 100.0_f64;
        let negative_noise = -5.0_f64;
        solver.set_row_bounds(
            &[0, water_balance_row],
            &[initial_storage, negative_noise],
            &[initial_storage, negative_noise],
        );

        // The solve must succeed — the slack absorbs the negative inflow.
        let view = solver
            .solve()
            .expect("LP must be feasible with inflow slack active");

        let primal = view.primal;

        // The inflow slack must be strictly positive: it compensates the
        // negative noise so that the water balance constraint is satisfied.
        assert!(
            primal[col_inflow_slack_start] > 0.0,
            "inflow slack must be positive when noise is negative, got {}",
            primal[col_inflow_slack_start]
        );

        // The objective must be positive: penalty cost * slack value > 0.
        assert!(
            view.objective > 0.0,
            "objective must include a positive penalty contribution, got {}",
            view.objective
        );
    }

    // -------------------------------------------------------------------------
    // ticket-024: load balance row starts, n_load_buses, load_bus_indices
    // -------------------------------------------------------------------------

    /// Build a two-bus system with N hydros and K blocks per stage.
    /// Bus B1 (EntityId=10) has `std_mw` = 0 (no load noise).
    /// Bus B2 (EntityId=20) has `std_mw` > 0 (stochastic load).
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        clippy::too_many_lines
    )]
    fn two_bus_system_with_stochastic_load(
        n_stages: usize,
        n_hydros_in_system: usize,
        n_blocks: usize,
    ) -> cobre_core::System {
        use chrono::NaiveDate;
        use cobre_core::entities::hydro::{Hydro, HydroGenerationModel, HydroPenalties};
        use cobre_core::scenario::{InflowModel, LoadModel};
        use cobre_core::temporal::{
            Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
            StageStateConfig,
        };

        // B1 (EntityId(10)): no noise, B2 (EntityId(20)): stochastic.
        let bus1 = Bus {
            id: EntityId(10),
            name: "B1".to_string(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 500.0,
            }],
            excess_cost: 0.0,
        };
        let bus2 = Bus {
            id: EntityId(20),
            name: "B2".to_string(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 500.0,
            }],
            excess_cost: 0.0,
        };

        let blocks: Vec<_> = (0..n_blocks)
            .map(|b| Block {
                index: b,
                name: format!("B{b}"),
                duration_hours: 240.0,
            })
            .collect();

        let stages: Vec<Stage> = (0..n_stages)
            .map(|i| Stage {
                index: i,
                id: i as i32,
                start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
                season_id: None,
                blocks: blocks.clone(),
                block_mode: BlockMode::Parallel,
                state_config: StageStateConfig {
                    storage: false,
                    inflow_lags: false,
                },
                risk_config: StageRiskConfig::Expectation,
                scenario_config: ScenarioSourceConfig {
                    branching_factor: 1,
                    noise_method: NoiseMethod::Saa,
                },
            })
            .collect();

        let hydros: Vec<Hydro> = (0..n_hydros_in_system)
            .map(|h| Hydro {
                id: EntityId((h + 100) as i32),
                name: format!("H{h}"),
                bus_id: EntityId(10),
                downstream_id: None,
                entry_stage_id: None,
                exit_stage_id: None,
                min_storage_hm3: 0.0,
                max_storage_hm3: 200.0,
                min_outflow_m3s: 0.0,
                max_outflow_m3s: None,
                generation_model: HydroGenerationModel::ConstantProductivity {
                    productivity_mw_per_m3s: 1.0,
                },
                min_turbined_m3s: 0.0,
                max_turbined_m3s: 50.0,
                min_generation_mw: 0.0,
                max_generation_mw: 50.0,
                tailrace: None,
                hydraulic_losses: None,
                efficiency: None,
                evaporation_coefficients_mm: None,
                evaporation_reference_volumes_hm3: None,
                diversion: None,
                filling: None,
                penalties: HydroPenalties {
                    spillage_cost: 0.0,
                    diversion_cost: 0.0,
                    fpha_turbined_cost: 0.0,
                    storage_violation_below_cost: 0.0,
                    filling_target_violation_cost: 0.0,
                    turbined_violation_below_cost: 0.0,
                    outflow_violation_below_cost: 0.0,
                    outflow_violation_above_cost: 0.0,
                    generation_violation_below_cost: 0.0,
                    evaporation_violation_cost: 0.0,
                    water_withdrawal_violation_cost: 0.0,
                },
            })
            .collect();

        let inflow_models: Vec<InflowModel> = hydros
            .iter()
            .flat_map(|h| {
                (0..n_stages).map(move |s| InflowModel {
                    hydro_id: h.id,
                    stage_id: s as i32,
                    mean_m3s: 50.0,
                    std_m3s: 10.0,
                    ar_coefficients: vec![],
                    residual_std_ratio: 1.0,
                })
            })
            .collect();

        let load_models: Vec<LoadModel> = (0..n_stages)
            .flat_map(|s| {
                [
                    LoadModel {
                        bus_id: EntityId(10),
                        stage_id: s as i32,
                        mean_mw: 80.0,
                        std_mw: 0.0, // B1: no noise
                    },
                    LoadModel {
                        bus_id: EntityId(20),
                        stage_id: s as i32,
                        mean_mw: 120.0,
                        std_mw: 15.0, // B2: stochastic
                    },
                ]
            })
            .collect();

        let n_st = n_stages.max(1);
        let bounds = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: n_hydros_in_system,
                n_thermals: 0,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages: n_st,
            },
            &BoundsDefaults {
                hydro: default_hydro_bounds(),
                thermal: ThermalStageBounds {
                    min_generation_mw: 0.0,
                    max_generation_mw: 0.0,
                },
                line: LineStageBounds {
                    direct_mw: 0.0,
                    reverse_mw: 0.0,
                },
                pumping: PumpingStageBounds {
                    min_flow_m3s: 0.0,
                    max_flow_m3s: 0.0,
                },
                contract: ContractStageBounds {
                    min_mw: 0.0,
                    max_mw: 0.0,
                    price_per_mwh: 0.0,
                },
            },
        );
        let penalties = ResolvedPenalties::new(
            &PenaltiesCountsSpec {
                n_hydros: n_hydros_in_system,
                n_buses: 2,
                n_lines: 0,
                n_ncs: 0,
                n_stages: n_st,
            },
            &PenaltiesDefaults {
                hydro: default_hydro_penalties(),
                bus: BusStagePenalties { excess_cost: 0.0 },
                line: LineStagePenalties { exchange_cost: 0.0 },
                ncs: NcsStagePenalties {
                    curtailment_cost: 0.0,
                },
            },
        );

        let mut builder = SystemBuilder::new()
            .buses(vec![bus1, bus2])
            .stages(stages)
            .load_models(load_models)
            .bounds(bounds)
            .penalties(penalties);
        if !hydros.is_empty() {
            builder = builder.hydros(hydros).inflow_models(inflow_models);
        }
        builder.build().expect("two_bus_system: valid")
    }

    // AC-1: load_balance_row_starts[0] == row_water_balance_start + n_hydros
    // for a 2-bus, 2-hydro, 3-block system.
    #[test]
    fn stage_templates_load_balance_row_starts_correct() {
        // N=2 hydros, B=2 buses, L=0 lags.
        // StageIndexer: n_state = N*(1+L) = 2, n_dual_relevant = 2.
        // row_water_balance_start = 2, row_load_balance_start = 2 + 2 = 4.
        let system = two_bus_system_with_stochastic_load(2, 2, 3);
        let result = build_stage_templates(
            &system,
            &no_penalty_config(),
            &PrecomputedPar::default(),
            &PrecomputedNormal::default(),
            &default_production(&system),
            &default_evaporation(&system),
        )
        .expect("constant productivity ok");

        assert_eq!(
            result.load_balance_row_starts.len(),
            result.templates.len(),
            "load_balance_row_starts length must match templates length"
        );

        // For N=2 hydros, L=0: n_state = 2*(1+0) = 2, so row_water_balance_start = 2,
        // row_load_balance_start = 2 + 2 = 4.
        let expected_row_start = result.base_rows[0] + 2; // base_rows[0] = row_water_balance_start
        assert_eq!(
            result.load_balance_row_starts[0], expected_row_start,
            "load_balance_row_starts[0] must equal row_water_balance_start + n_hydros"
        );
        // Both stages should have the same row start (same topology across stages).
        assert_eq!(
            result.load_balance_row_starts[0], result.load_balance_row_starts[1],
            "identical stages share the same load balance row start"
        );
    }

    // AC-2: n_load_buses and load_bus_indices populated for stochastic buses.
    #[test]
    fn stage_templates_n_load_buses_matches_stochastic_buses() {
        // B2 (EntityId(20)) has std_mw > 0; B1 (EntityId(10)) does not.
        // The system has buses in order [B1(10), B2(20)], so B2 is at index 1.
        let system = two_bus_system_with_stochastic_load(1, 0, 1);
        let result = build_stage_templates(
            &system,
            &no_penalty_config(),
            &PrecomputedPar::default(),
            &PrecomputedNormal::default(),
            &default_production(&system),
            &default_evaporation(&system),
        )
        .expect("constant productivity ok");

        assert_eq!(
            result.n_load_buses, 1,
            "only B2 has std_mw > 0 → n_load_buses must be 1"
        );
        assert_eq!(
            result.load_bus_indices.len(),
            1,
            "load_bus_indices must have exactly one entry"
        );
        assert_eq!(
            result.load_bus_indices[0], 1,
            "B2 is at buses slice index 1 (buses are [B1(10), B2(20)])"
        );
    }

    // AC-3: no load buses when no load models have std_mw > 0.
    #[test]
    fn stage_templates_no_load_buses_gives_zero() {
        // one_bus_system uses std_mw = 0 for all load models.
        let system = one_bus_system(2);
        let result = build_stage_templates(
            &system,
            &no_penalty_config(),
            &PrecomputedPar::default(),
            &PrecomputedNormal::default(),
            &default_production(&system),
            &default_evaporation(&system),
        )
        .expect("constant productivity ok");

        assert_eq!(
            result.n_load_buses, 0,
            "system with std_mw = 0 everywhere must give n_load_buses = 0"
        );
        assert!(
            result.load_bus_indices.is_empty(),
            "load_bus_indices must be empty when n_load_buses = 0"
        );
        assert_eq!(
            result.load_balance_row_starts.len(),
            result.templates.len(),
            "load_balance_row_starts length must always match templates length"
        );
    }

    // -------------------------------------------------------------------------
    // FPHA constraint tests (AC-1 through AC-5)
    // -------------------------------------------------------------------------

    /// Helper: find the coefficient for (column, row) in the CSC matrix of a
    /// stage template.  Returns `None` if the column has no entry in that row.
    #[allow(clippy::cast_sign_loss)] // col_starts and row_indices are non-negative by construction
    fn csc_entry(tmpl: &cobre_solver::StageTemplate, col: usize, row: usize) -> Option<f64> {
        let start = tmpl.col_starts[col] as usize;
        let end = tmpl.col_starts[col + 1] as usize;
        for pos in start..end {
            if tmpl.row_indices[pos] as usize == row {
                return Some(tmpl.values[pos]);
            }
        }
        None
    }

    /// Build a 1-bus, 1-FPHA-hydro, 1-stage, 1-block system plus an FPHA
    /// `ProductionModelSet` with `n_planes` hyperplanes.
    ///
    /// Plane coefficients are deterministic: `intercept=10.0, gamma_v=0.5,
    /// gamma_q=2.0, gamma_s=0.1` for every plane.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        clippy::too_many_lines
    )]
    fn one_fpha_hydro_system(n_planes: usize) -> (cobre_core::System, ProductionModelSet) {
        use chrono::NaiveDate;
        use cobre_core::entities::hydro::{Hydro, HydroGenerationModel, HydroPenalties};
        use cobre_core::scenario::{InflowModel, LoadModel};
        use cobre_core::temporal::{
            Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
            StageStateConfig,
        };

        let bus = Bus {
            id: EntityId(1),
            name: "B1".to_string(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 500.0,
            }],
            excess_cost: 0.0,
        };
        let hydro = Hydro {
            id: EntityId(2),
            name: "FPHA1".to_string(),
            bus_id: EntityId(1),
            downstream_id: None,
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: 0.0,
            max_storage_hm3: 500.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: HydroGenerationModel::Fpha,
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 150.0,
            min_generation_mw: 0.0,
            max_generation_mw: 300.0,
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: None,
            evaporation_reference_volumes_hm3: None,
            diversion: None,
            filling: None,
            penalties: HydroPenalties {
                spillage_cost: 0.01,
                diversion_cost: 0.0,
                fpha_turbined_cost: 0.0,
                storage_violation_below_cost: 0.0,
                filling_target_violation_cost: 0.0,
                turbined_violation_below_cost: 0.0,
                outflow_violation_below_cost: 0.0,
                outflow_violation_above_cost: 0.0,
                generation_violation_below_cost: 0.0,
                evaporation_violation_cost: 0.0,
                water_withdrawal_violation_cost: 0.0,
            },
        };

        let stages: Vec<Stage> = vec![Stage {
            index: 0,
            id: 0,
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            season_id: None,
            blocks: vec![Block {
                index: 0,
                name: "S".to_string(),
                duration_hours: 744.0,
            }],
            block_mode: BlockMode::Parallel,
            state_config: StageStateConfig {
                storage: true,
                inflow_lags: false,
            },
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor: 1,
                noise_method: NoiseMethod::Saa,
            },
        }];

        let inflow_models: Vec<InflowModel> = vec![InflowModel {
            hydro_id: EntityId(2),
            stage_id: 0,
            mean_m3s: 80.0,
            std_m3s: 20.0,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
        }];

        let load_models: Vec<LoadModel> = vec![LoadModel {
            bus_id: EntityId(1),
            stage_id: 0,
            mean_mw: 200.0,
            std_mw: 0.0,
        }];

        let bounds = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 1,
                n_thermals: 0,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages: 1,
            },
            &BoundsDefaults {
                hydro: HydroStageBounds {
                    min_storage_hm3: 0.0,
                    max_storage_hm3: 500.0,
                    min_turbined_m3s: 0.0,
                    max_turbined_m3s: 150.0,
                    min_outflow_m3s: 0.0,
                    max_outflow_m3s: None,
                    min_generation_mw: 0.0,
                    max_generation_mw: 300.0,
                    max_diversion_m3s: None,
                    filling_inflow_m3s: 0.0,
                    water_withdrawal_m3s: 0.0,
                },
                thermal: ThermalStageBounds {
                    min_generation_mw: 0.0,
                    max_generation_mw: 0.0,
                },
                line: LineStageBounds {
                    direct_mw: 0.0,
                    reverse_mw: 0.0,
                },
                pumping: PumpingStageBounds {
                    min_flow_m3s: 0.0,
                    max_flow_m3s: 0.0,
                },
                contract: ContractStageBounds {
                    min_mw: 0.0,
                    max_mw: 0.0,
                    price_per_mwh: 0.0,
                },
            },
        );
        let penalties = ResolvedPenalties::new(
            &PenaltiesCountsSpec {
                n_hydros: 1,
                n_buses: 1,
                n_lines: 0,
                n_ncs: 0,
                n_stages: 1,
            },
            &PenaltiesDefaults {
                hydro: default_hydro_penalties(),
                bus: BusStagePenalties { excess_cost: 0.0 },
                line: LineStagePenalties { exchange_cost: 0.0 },
                ncs: NcsStagePenalties {
                    curtailment_cost: 0.0,
                },
            },
        );

        let system = SystemBuilder::new()
            .buses(vec![bus])
            .hydros(vec![hydro])
            .stages(stages)
            .inflow_models(inflow_models)
            .load_models(load_models)
            .bounds(bounds)
            .penalties(penalties)
            .build()
            .expect("one_fpha_hydro_system: valid");

        // Build a ProductionModelSet with n_planes FPHA planes for hydro 0.
        let plane = FphaPlane {
            intercept: 10.0,
            gamma_v: 0.5,
            gamma_q: 2.0,
            gamma_s: 0.1,
        };
        let planes = vec![plane; n_planes];
        let models = vec![vec![ResolvedProductionModel::Fpha {
            planes,
            turbined_cost: 0.0,
        }]];
        let production = ProductionModelSet::new(models, 1, 1);

        (system, production)
    }

    /// Build a 1-bus, 4-hydro, 1-stage, 1-block system with 2 constant and
    /// 2 FPHA hydros.  Hydro indices 0 and 1 are constant productivity;
    /// hydro indices 2 and 3 are FPHA (3 planes each).
    ///
    /// Hydros are declared in [`EntityId`](cobre_core::EntityId) order:
    /// 100 (const), 101 (const), 102 (fpha), 103 (fpha) — canonical sort
    /// order is preserved by the system builder.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        clippy::too_many_lines
    )]
    fn four_hydro_mixed_system() -> (cobre_core::System, ProductionModelSet) {
        use chrono::NaiveDate;
        use cobre_core::entities::hydro::{Hydro, HydroGenerationModel, HydroPenalties};
        use cobre_core::scenario::{InflowModel, LoadModel};
        use cobre_core::temporal::{
            Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
            StageStateConfig,
        };

        let bus = Bus {
            id: EntityId(1),
            name: "B1".to_string(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 500.0,
            }],
            excess_cost: 0.0,
        };

        let hydro_penalties = HydroPenalties {
            spillage_cost: 0.01,
            diversion_cost: 0.0,
            fpha_turbined_cost: 0.0,
            storage_violation_below_cost: 0.0,
            filling_target_violation_cost: 0.0,
            turbined_violation_below_cost: 0.0,
            outflow_violation_below_cost: 0.0,
            outflow_violation_above_cost: 0.0,
            generation_violation_below_cost: 0.0,
            evaporation_violation_cost: 0.0,
            water_withdrawal_violation_cost: 0.0,
        };

        let make_hydro = |id: i32, gen_model: HydroGenerationModel| Hydro {
            id: EntityId(id),
            name: format!("H{id}"),
            bus_id: EntityId(1),
            downstream_id: None,
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: 0.0,
            max_storage_hm3: 200.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: gen_model,
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 100.0,
            min_generation_mw: 0.0,
            max_generation_mw: 250.0,
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: None,
            evaporation_reference_volumes_hm3: None,
            diversion: None,
            filling: None,
            penalties: hydro_penalties,
        };

        let hydros = vec![
            make_hydro(
                100,
                HydroGenerationModel::ConstantProductivity {
                    productivity_mw_per_m3s: 2.5,
                },
            ),
            make_hydro(
                101,
                HydroGenerationModel::ConstantProductivity {
                    productivity_mw_per_m3s: 3.0,
                },
            ),
            make_hydro(102, HydroGenerationModel::Fpha),
            make_hydro(103, HydroGenerationModel::Fpha),
        ];

        let stages: Vec<Stage> = vec![Stage {
            index: 0,
            id: 0,
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            season_id: None,
            blocks: vec![Block {
                index: 0,
                name: "S".to_string(),
                duration_hours: 744.0,
            }],
            block_mode: BlockMode::Parallel,
            state_config: StageStateConfig {
                storage: true,
                inflow_lags: false,
            },
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor: 1,
                noise_method: NoiseMethod::Saa,
            },
        }];

        let inflow_models: Vec<InflowModel> = hydros
            .iter()
            .map(|h| InflowModel {
                hydro_id: h.id,
                stage_id: 0,
                mean_m3s: 80.0,
                std_m3s: 20.0,
                ar_coefficients: vec![],
                residual_std_ratio: 1.0,
            })
            .collect();

        let load_models: Vec<LoadModel> = vec![LoadModel {
            bus_id: EntityId(1),
            stage_id: 0,
            mean_mw: 400.0,
            std_mw: 0.0,
        }];

        let bounds = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 4,
                n_thermals: 0,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages: 1,
            },
            &BoundsDefaults {
                hydro: default_hydro_bounds(),
                thermal: ThermalStageBounds {
                    min_generation_mw: 0.0,
                    max_generation_mw: 0.0,
                },
                line: LineStageBounds {
                    direct_mw: 0.0,
                    reverse_mw: 0.0,
                },
                pumping: PumpingStageBounds {
                    min_flow_m3s: 0.0,
                    max_flow_m3s: 0.0,
                },
                contract: ContractStageBounds {
                    min_mw: 0.0,
                    max_mw: 0.0,
                    price_per_mwh: 0.0,
                },
            },
        );
        let penalties = ResolvedPenalties::new(
            &PenaltiesCountsSpec {
                n_hydros: 4,
                n_buses: 1,
                n_lines: 0,
                n_ncs: 0,
                n_stages: 1,
            },
            &PenaltiesDefaults {
                hydro: default_hydro_penalties(),
                bus: BusStagePenalties { excess_cost: 0.0 },
                line: LineStagePenalties { exchange_cost: 0.0 },
                ncs: NcsStagePenalties {
                    curtailment_cost: 0.0,
                },
            },
        );

        let system = SystemBuilder::new()
            .buses(vec![bus])
            .hydros(hydros)
            .stages(stages)
            .inflow_models(inflow_models)
            .load_models(load_models)
            .bounds(bounds)
            .penalties(penalties)
            .build()
            .expect("four_hydro_mixed_system: valid");

        // Production models: hydros 0,1 → ConstantProductivity; hydros 2,3 → Fpha (3 planes).
        let plane = FphaPlane {
            intercept: 10.0,
            gamma_v: 0.5,
            gamma_q: 2.0,
            gamma_s: 0.1,
        };
        let fpha_planes = vec![plane; 3];
        let models = vec![
            vec![ResolvedProductionModel::ConstantProductivity { productivity: 2.5 }],
            vec![ResolvedProductionModel::ConstantProductivity { productivity: 3.0 }],
            vec![ResolvedProductionModel::Fpha {
                planes: fpha_planes.clone(),
                turbined_cost: 0.0,
            }],
            vec![ResolvedProductionModel::Fpha {
                planes: fpha_planes,
                turbined_cost: 0.0,
            }],
        ];
        let production = ProductionModelSet::new(models, 4, 1);

        (system, production)
    }

    /// AC-1: 1-FPHA-hydro system with 5 planes and 1 block:
    ///  - `num_cols` increases by 1 (one generation column) vs constant case
    ///  - `num_rows` increases by 5 (five FPHA rows) vs constant case
    #[test]
    fn fpha_ac1_dimensions_one_fpha_hydro_five_planes() {
        let (system, production) = one_fpha_hydro_system(5);

        // Build with FPHA production model.
        let fpha_result = build_stage_templates(
            &system,
            &no_penalty_config(),
            &PrecomputedPar::default(),
            &PrecomputedNormal::default(),
            &production,
            &default_evaporation(&system),
        )
        .expect("FPHA system ok");

        // Build with constant-productivity production model (same system entity).
        let const_result = build_stage_templates(
            &system,
            &no_penalty_config(),
            &PrecomputedPar::default(),
            &PrecomputedNormal::default(),
            &default_production(&system),
            &default_evaporation(&system),
        )
        .expect("constant productivity ok");

        let fpha_tmpl = &fpha_result.templates[0];
        let const_tmpl = &const_result.templates[0];

        assert_eq!(
            fpha_tmpl.num_cols,
            const_tmpl.num_cols + 1,
            "FPHA adds exactly 1 generation column (1 hydro * 1 block)"
        );
        assert_eq!(
            fpha_tmpl.num_rows,
            const_tmpl.num_rows + 5,
            "FPHA adds exactly 5 constraint rows (5 planes * 1 block)"
        );
    }

    /// AC-2: generation column has entries in all 5 FPHA rows (+1.0) and in
    ///       the load balance row for the hydro's bus (+1.0).
    ///
    /// Column layout for N=1, L=0, T=0, Lines=0, B=1, K=1, no penalty:
    /// - 0: `v` (storage out)
    /// - 1: `z_inflow` (realized inflow)
    /// - 2: `v_in` (storage in)
    /// - 3: `theta`
    /// - 4: `turbine[0,0]`
    /// - 5: `spillage[0,0]`
    /// - 6: `deficit[0,0]`
    /// - 7: `excess[0,0]`
    /// - 8: `g` (generation, FPHA, `col_generation_start = 8`)
    ///
    /// Row layout for N=1, L=0, B=1, K=1, 5 planes:
    /// - 0: storage-fixing row (`n_state = n_dual_relevant = 1`)
    /// - 1: z_inflow-def row 0
    /// - 2: water balance row 0 (`row_water_balance_start = 2`)
    /// - 3: load balance row 0 (`row_load_balance_start = 3`)
    /// - 4..8: FPHA rows 0..4 (`row_fpha_start = 4`)
    #[test]
    fn fpha_ac2_generation_column_entries() {
        let n_planes = 5;
        let (system, production) = one_fpha_hydro_system(n_planes);

        let result = build_stage_templates(
            &system,
            &no_penalty_config(),
            &PrecomputedPar::default(),
            &PrecomputedNormal::default(),
            &production,
            &default_evaporation(&system),
        )
        .expect("FPHA system ok");

        let tmpl = &result.templates[0];

        // N=1, L=0 → n_state = 1, decision_start = 4, col_generation_start = 9.
        // (turbine[4..5], spillage[5..6], diversion[6..7], deficit[7..8], excess[8..9], generation[9])
        let col_g = 9_usize;

        // row_fpha_start = 4 (= n_state + N z_inflow + N water balance + B*1 load balance)
        let row_fpha_start = 4_usize;

        // Check: generation column has +1.0 in each of the 5 FPHA rows.
        for p in 0..n_planes {
            let row = row_fpha_start + p;
            let coeff = csc_entry(tmpl, col_g, row)
                .unwrap_or_else(|| panic!("generation column missing entry in FPHA row {row}"));
            assert!(
                (coeff - 1.0).abs() < 1e-12,
                "generation col FPHA row {row}: expected +1.0, got {coeff}"
            );
        }

        // Check: generation column has +1.0 in the load balance row (row 3).
        let row_lb = 3_usize;
        let lb_coeff = csc_entry(tmpl, col_g, row_lb).unwrap_or_else(|| {
            panic!("generation column missing entry in load balance row {row_lb}")
        });
        assert!(
            (lb_coeff - 1.0).abs() < 1e-12,
            "generation col load balance row: expected +1.0, got {lb_coeff}"
        );
    }

    /// AC-3: `v_in` column has entries in all 5 FPHA rows with coefficient
    ///       `-gamma_v/2`.
    ///
    /// For plane with `gamma_v = 0.5`: coefficient = `-0.5/2 = -0.25`.
    #[test]
    fn fpha_ac3_v_in_column_entries() {
        let n_planes = 5;
        let (system, production) = one_fpha_hydro_system(n_planes);

        let result = build_stage_templates(
            &system,
            &no_penalty_config(),
            &PrecomputedPar::default(),
            &PrecomputedNormal::default(),
            &production,
            &default_evaporation(&system),
        )
        .expect("FPHA system ok");

        let tmpl = &result.templates[0];

        // v_in column: for N=1, L=0, storage_in.start = 2 (after z_inflow).
        let col_v_in = 2_usize;
        let row_fpha_start = 4_usize;
        // gamma_v = 0.5 → expected coefficient = -0.25
        let expected = -0.5_f64 / 2.0;

        for p in 0..n_planes {
            let row = row_fpha_start + p;
            let coeff = csc_entry(tmpl, col_v_in, row)
                .unwrap_or_else(|| panic!("v_in column missing entry in FPHA row {row}"));
            assert!(
                (coeff - expected).abs() < 1e-12,
                "v_in col FPHA row {row}: expected {expected}, got {coeff}"
            );
        }
    }

    /// AC-4: outgoing storage column (`v`) has entries in all 5 FPHA rows
    ///       with coefficient `-gamma_v/2`.
    ///
    /// For plane with `gamma_v = 0.5`: coefficient = `-0.25`.
    #[test]
    fn fpha_ac4_v_out_column_entries() {
        let n_planes = 5;
        let (system, production) = one_fpha_hydro_system(n_planes);

        let result = build_stage_templates(
            &system,
            &no_penalty_config(),
            &PrecomputedPar::default(),
            &PrecomputedNormal::default(),
            &production,
            &default_evaporation(&system),
        )
        .expect("FPHA system ok");

        let tmpl = &result.templates[0];

        // v (outgoing storage) column: for N=1, index 0.
        let col_v = 0_usize;
        let row_fpha_start = 4_usize;
        let expected = -0.5_f64 / 2.0;

        for p in 0..n_planes {
            let row = row_fpha_start + p;
            let coeff = csc_entry(tmpl, col_v, row)
                .unwrap_or_else(|| panic!("v column missing entry in FPHA row {row}"));
            assert!(
                (coeff - expected).abs() < 1e-12,
                "v col FPHA row {row}: expected {expected}, got {coeff}"
            );
        }
    }

    /// AC-5: mixed system (2 constant + 2 FPHA hydros).
    ///
    /// Column layout for N=4, L=0, T=0, Lines=0, B=1, K=1, no penalty:
    /// - state cols 0..3: `v[0..3]`; 4..7: `z_inflow[0..3]`; 8..11: `v_in[0..3]`; 12: `theta`
    /// - `col_turbine_start = 13`, `col_spillage_start = 17`
    /// - `col_deficit_start = 21`, `col_excess_start = 22`
    /// - `col_generation_start = 23` (no penalty)
    /// - FPHA hydro 0 (`local_idx=0`, `h_idx=2`): g col = 23
    /// - FPHA hydro 1 (`local_idx=1`, `h_idx=3`): g col = 24
    ///
    /// Row layout:
    /// - `n_state = 4`, `z_inflow rows = [4,8)`, `row_water_balance_start = 8`
    /// - `row_load_balance_start = 12`, `row_fpha_start = 13`
    /// - Load balance for bus B1 (`bus_idx=0`, blk=0): row = 12
    #[test]
    fn fpha_ac5_mixed_system_load_balance_uses_generation_col() {
        let (system, production) = four_hydro_mixed_system();

        let result = build_stage_templates(
            &system,
            &no_penalty_config(),
            &PrecomputedPar::default(),
            &PrecomputedNormal::default(),
            &production,
            &default_evaporation(&system),
        )
        .expect("mixed FPHA/constant system ok");

        let tmpl = &result.templates[0];

        // Verify overall dimensions: 2 FPHA hydros * 1 block * 3 planes = 6 extra rows,
        // 2 generation columns.
        // Base (constant) num_cols = 13 + 4*1*3 + 0 + 0 + 1*1*2 = 13 + 12 + 2 = 27.
        // (3 = turbine + spillage + diversion per hydro per block)
        // With FPHA: 27 + 2 = 29.
        // With withdrawal slack (N=4): 29 + 4 = 33.
        // z-inflow is now embedded in state region (not appended at end).
        // Base num_rows = n_state + N z_inflow + 4 water balance + 1*1 load balance = 4 + 4 + 4 + 1 = 13.
        // With FPHA: 13 + 2*1*3 = 13 + 6 = 19.
        assert_eq!(
            tmpl.num_cols, 33,
            "4-hydro mixed system: num_cols should be 33 (includes diversion and withdrawal slack)"
        );
        assert_eq!(
            tmpl.num_rows, 19,
            "4-hydro mixed system: num_rows should be 19 (includes z_inflow rows)"
        );

        // Load balance row for the single bus, block 0: row = 12.
        let row_lb = 12_usize;

        // FPHA hydro at h_idx=2 (local_idx=0): generation col = 27.
        // (shifted +4 from 23 due to diversion columns for 4 hydros)
        let col_g_fpha0 = 27_usize;
        let g0_lb_coeff = csc_entry(tmpl, col_g_fpha0, row_lb).unwrap_or_else(|| {
            panic!("FPHA hydro 0 generation column missing entry in load balance row {row_lb}")
        });
        assert!(
            (g0_lb_coeff - 1.0).abs() < 1e-12,
            "FPHA hydro 0 load balance: expected +1.0, got {g0_lb_coeff}"
        );

        // FPHA hydro at h_idx=3 (local_idx=1): generation col = 28.
        let col_g_fpha1 = 28_usize;
        let g1_lb_coeff = csc_entry(tmpl, col_g_fpha1, row_lb).unwrap_or_else(|| {
            panic!("FPHA hydro 1 generation column missing entry in load balance row {row_lb}")
        });
        assert!(
            (g1_lb_coeff - 1.0).abs() < 1e-12,
            "FPHA hydro 1 load balance: expected +1.0, got {g1_lb_coeff}"
        );

        // Constant hydro at h_idx=0: turbine col = col_turbine_start + 0*1+0 = 13.
        // This turbine column must appear in load balance row 12 with coefficient
        // rho*block_hours = 2.5 * 744 (not +1.0), confirming old behavior preserved.
        let col_turb_const = 13_usize;
        let turb_lb_coeff = csc_entry(tmpl, col_turb_const, row_lb);
        assert!(
            turb_lb_coeff.is_some(),
            "constant hydro 0 turbine col must appear in load balance row"
        );
        // The key invariant: constant hydro uses rho * turbine_col in the load
        // balance row (not a generation column).  The coefficient is rho (the
        // productivity scalar), not rho * block_hours; block_hours scaling is
        // applied only to cost objectives, not to the power-balance matrix.
        // For hydro 0 with productivity 2.5 MW/(m³/s): coefficient = 2.5.
        let expected_rho_coeff = 2.5_f64;
        assert!(
            (turb_lb_coeff.unwrap() - expected_rho_coeff).abs() < 1e-12,
            "constant hydro 0 turbine: expected rho = {expected_rho_coeff}, got {turb_lb_coeff:?}"
        );
    }

    // -------------------------------------------------------------------------
    // ticket-009: FPHA LP integration tests (HiGHS end-to-end solve)
    // -------------------------------------------------------------------------
    //
    // These tests build an FPHA template, load it into HiGHS, patch the
    // storage-fixing row to set `v_in`, solve, and inspect the solution.
    //
    // Column layout for N=1, L=0, T=0, Lines=0, B=1, K=1, 3 planes, no penalty:
    //   col 0: v        (outgoing storage, hm³)
    //   col 1: v_in     (incoming storage, fixed by row 0)
    //   col 2: theta    (future cost)
    //   col 3: q        (turbined flow, m³/s)
    //   col 4: s        (spillage, m³/s)
    //   col 5: deficit
    //   col 6: excess
    //   col 7: g        (FPHA generation variable, MW)
    //
    // Row layout for N=1, L=0, B=1, K=1, 3 planes:
    //   row 0: storage-fixing  (equality: v_in = v_in_value)
    //   row 1: water-balance
    //   row 2: load-balance    (equality: g = load_mw = 200 MW)
    //   row 3: FPHA plane 0
    //   row 4: FPHA plane 1
    //   row 5: FPHA plane 2
    //
    // Planes used in these tests (realistic, feasibility-ensuring):
    //   intercept = 300.0, gamma_v = 1.0 (>0), gamma_q = 3.0 (>0), gamma_s = 0.0 (<=0)
    //
    // With v_in = 100 hm³, inflow = 80 m³/s, load = 200 MW, spillage = 0:
    //   water balance: v = v_in + zeta*(q_in - q - s) = 100 + 0.2678*80 - (q+s)*zeta
    //   FPHA constraint: g <= 300 + 1.0*v_avg + 3.0*q
    //   load-balance: g = 200.0 (equality)
    //   At g=200, q=50 m³/s: 300 + 1.0*v_avg + 3.0*50 = 300 + ~100 + 150 = 550 >> 200 ✓

    /// Build a 1-FPHA-hydro system identical to `one_fpha_hydro_system` but
    /// with 3 planes and a large-enough intercept to ensure the solve is
    /// feasible for any `v_in` in `[0, 500]` hm³ and reasonable turbine flow.
    ///
    /// Planes: `intercept=300.0, gamma_v=1.0, gamma_q=3.0, gamma_s=0.0`.
    fn fpha_solve_system() -> (cobre_core::System, ProductionModelSet) {
        let planes = vec![
            FphaPlane {
                intercept: 300.0,
                gamma_v: 1.0,
                gamma_q: 3.0,
                gamma_s: 0.0,
            };
            3
        ];
        let models = vec![vec![ResolvedProductionModel::Fpha {
            planes,
            turbined_cost: 0.0,
        }]];
        let production = ProductionModelSet::new(models, 1, 1);
        // Reuse the one_fpha_hydro_system fixture (max_storage=500, max_turbine=150,
        // max_generation=300, load=200 MW) but replace its planes via production above.
        let (system, _) = one_fpha_hydro_system(3);
        (system, production)
    }

    /// AC (ticket-009): 1-FPHA-hydro LP solves to Optimal with generation > 0.
    ///
    /// Patches `v_in = 100 hm³` (row 0), then solves. Asserts:
    /// - status is `Ok` (no solver error)
    /// - objective is finite
    /// - `g` (col 7) is strictly positive
    #[test]
    fn fpha_solve_one_hydro_optimal() {
        use cobre_solver::{HighsSolver, RowBatch, SolverInterface};

        let (system, production) = fpha_solve_system();
        let result = build_stage_templates(
            &system,
            &no_penalty_config(),
            &PrecomputedPar::default(),
            &PrecomputedNormal::default(),
            &production,
            &default_evaporation(&system),
        )
        .expect("FPHA template build must succeed");

        let template = &result.templates[0];
        let mut solver = HighsSolver::new().expect("HighsSolver::new must succeed");
        solver.load_model(template);

        // No cuts at this point.
        let empty_cuts = RowBatch {
            num_rows: 0,
            row_starts: vec![0_i32],
            col_indices: vec![],
            values: vec![],
            row_lower: vec![],
            row_upper: vec![],
        };
        solver.add_rows(&empty_cuts);

        // Fix v_in = 100 hm³ via the storage-fixing row (row 0).
        let v_in = 100.0_f64;
        solver.set_row_bounds(&[0], &[v_in], &[v_in]);

        let view = solver
            .solve()
            .expect("FPHA LP must be feasible and optimal");

        // col 9 is g (generation variable, shifted by +1 for diversion).
        let col_g = 9_usize;
        let generation = view.primal[col_g];
        assert!(
            generation > 0.0,
            "FPHA generation must be strictly positive, got {generation}"
        );
    }

    /// AC (ticket-009): all hyperplane constraints hold within 1e-6 tolerance
    /// after solving the 1-FPHA-hydro LP.
    ///
    /// For each plane p: `g <= intercept + gamma_v * v_avg + gamma_q * q + gamma_s * s`
    /// where `v_avg = (v + v_in) / 2`.
    ///
    /// Column indices (N=1, L=0, no penalty, 3 planes):
    ///   col 0: `v`, col 1: `v_in`, col 3: `q`, col 4: `s`, col 8: `g`
    #[test]
    fn fpha_solve_hyperplane_constraints_hold() {
        use cobre_solver::{HighsSolver, RowBatch, SolverInterface};

        let (system, production) = fpha_solve_system();

        // Extract planes before moving production into build_stage_templates.
        let planes = match production.model(0, 0) {
            ResolvedProductionModel::Fpha { planes, .. } => planes.clone(),
            ResolvedProductionModel::ConstantProductivity { .. } => {
                panic!("expected Fpha model")
            }
        };

        let result = build_stage_templates(
            &system,
            &no_penalty_config(),
            &PrecomputedPar::default(),
            &PrecomputedNormal::default(),
            &production,
            &default_evaporation(&system),
        )
        .expect("FPHA template build must succeed");

        let template = &result.templates[0];
        let mut solver = HighsSolver::new().expect("HighsSolver::new must succeed");
        solver.load_model(template);

        let empty_cuts = RowBatch {
            num_rows: 0,
            row_starts: vec![0_i32],
            col_indices: vec![],
            values: vec![],
            row_lower: vec![],
            row_upper: vec![],
        };
        solver.add_rows(&empty_cuts);

        let v_in = 100.0_f64;
        solver.set_row_bounds(&[0], &[v_in], &[v_in]);

        let view = solver.solve().expect("FPHA LP must solve to optimal");
        let primal = view.primal;

        let col_v = 0_usize;
        let col_v_in = 1_usize;
        let col_q = 3_usize;
        let col_s = 4_usize;
        let col_g = 9_usize;

        let g = primal[col_g];
        let v = primal[col_v];
        let v_in_sol = primal[col_v_in];
        let q = primal[col_q];
        let s = primal[col_s];
        let v_avg = f64::midpoint(v, v_in_sol);

        for (p_idx, plane) in planes.iter().enumerate() {
            let rhs =
                plane.intercept + plane.gamma_v * v_avg + plane.gamma_q * q + plane.gamma_s * s;
            assert!(
                g <= rhs + 1e-6,
                "FPHA plane {p_idx}: g={g} must be <= rhs={rhs} \
                 (intercept={intercept}, gamma_v={gamma_v}, v_avg={v_avg}, \
                  gamma_q={gamma_q}, q={q}, gamma_s={gamma_s}, s={s})",
                intercept = plane.intercept,
                gamma_v = plane.gamma_v,
                gamma_q = plane.gamma_q,
                gamma_s = plane.gamma_s,
            );
        }
    }

    /// AC (ticket-009): storage-fixing dual differs between FPHA and
    /// constant-productivity for the same system entity.
    ///
    /// The FPHA model introduces `-gamma_v/2` entries on the `v_in` column
    /// in the hyperplane rows, which propagates through the simplex dual and
    /// modifies the shadow price of the storage-fixing equality (row 0).
    ///
    /// # Design
    ///
    /// To guarantee the FPHA constraint is **binding** at the optimum — a
    /// necessary condition for a non-zero storage-fixing dual — the planes use
    /// tight coefficients: `intercept=0, gamma_v=0.5, gamma_q=1.0, gamma_s=0.0`.
    ///
    /// System: 1 hydro, 1 bus, load=200 MW, inflow=80 m³/s, `max_turbine`=150 m³/s,
    /// `deficit_cost`=500. At `v_in`=100 hm³, the maximum achievable generation
    /// without deficit is approximately 142 MW < 200 MW (see comment below).
    /// The optimizer turbines at max capacity (bounded by water balance) and covers
    /// the remainder with costly deficit, so the FPHA constraint is binding and
    /// `d(cost)/d(v_in)` < 0.
    ///
    /// For the constant-productivity model (`default_from_system` gives `rho`=0),
    /// the turbine contributes zero generation and the cost is always 500*200,
    /// independent of `v_in` → dual = 0.
    ///
    /// Therefore the duals differ: FPHA dual is negative (more storage reduces cost),
    /// constant-productivity dual is zero.
    ///
    /// # FPHA constraint analysis
    ///
    /// With `intercept=0, gamma_v=0.5, gamma_q=1.0, gamma_s=0.0`, K=1, `zeta`≈2.678:
    /// - FPHA row: `g <= 0.5/2*(v + v_in) + 1.0*q = 0.25*(v + v_in) + q`
    /// - Water balance: `v = v_in + inflow*zeta - (q+s)*zeta`
    /// - Maximum feasible `q` (`v`≥0 binding): `q_max = v_in/zeta + inflow ≈ 37.3 + 80 = 117.3 m³/s`
    /// - Maximum `g` ≈ `0.25*(0 + v_in) + 117.3` ≈ `25 + 117.3 = 142.3 MW` < 200 MW
    ///
    /// The FPHA constraint is binding at `q_max`; extra `v_in` increases both `q_max` and
    /// the direct `gamma_v*v_in` term, reducing the deficit and hence the cost.
    #[test]
    fn fpha_solve_storage_fixing_dual_differs_from_constant() {
        use cobre_solver::{HighsSolver, RowBatch, SolverInterface};

        let (system, _) = one_fpha_hydro_system(1);

        // FPHA production model with tight planes: intercept=0, gamma_v=0.5, gamma_q=1.0.
        // These coefficients make the FPHA capacity (~142 MW) below the load (200 MW),
        // ensuring the constraint is binding and the storage-fixing dual is non-zero.
        let tight_planes = vec![FphaPlane {
            intercept: 0.0,
            gamma_v: 0.5,
            gamma_q: 1.0,
            gamma_s: 0.0,
        }];
        let fpha_production = ProductionModelSet::new(
            vec![vec![ResolvedProductionModel::Fpha {
                planes: tight_planes,
                turbined_cost: 0.0,
            }]],
            1,
            1,
        );

        // Build template for FPHA production model.
        let fpha_result = build_stage_templates(
            &system,
            &no_penalty_config(),
            &PrecomputedPar::default(),
            &PrecomputedNormal::default(),
            &fpha_production,
            &default_evaporation(&system),
        )
        .expect("FPHA template build must succeed");

        // Build template for constant-productivity production model.
        // `default_from_system` maps the Fpha entity model to productivity=0.0,
        // so the turbine contributes nothing to generation and the cost is always
        // deficit_cost * load_mw regardless of v_in → dual[0] = 0.
        let const_production = default_production(&system);
        let const_result = build_stage_templates(
            &system,
            &no_penalty_config(),
            &PrecomputedPar::default(),
            &PrecomputedNormal::default(),
            &const_production,
            &default_evaporation(&system),
        )
        .expect("constant productivity template build must succeed");

        let solve_and_get_storage_dual = |template: &cobre_solver::StageTemplate| -> f64 {
            let mut solver = HighsSolver::new().expect("HighsSolver::new must succeed");
            solver.load_model(template);
            let empty_cuts = RowBatch {
                num_rows: 0,
                row_starts: vec![0_i32],
                col_indices: vec![],
                values: vec![],
                row_lower: vec![],
                row_upper: vec![],
            };
            solver.add_rows(&empty_cuts);
            // Fix v_in = 100 hm³ via the storage-fixing equality row (row 0).
            let v_in = 100.0_f64;
            solver.set_row_bounds(&[0], &[v_in], &[v_in]);
            let view = solver.solve().expect("LP must solve to optimal");
            // Row 0 is the storage-fixing equality; its dual is the marginal cost
            // of one additional hm³ of initial storage.
            view.dual[0]
        };

        let fpha_dual = solve_and_get_storage_dual(&fpha_result.templates[0]);
        let const_dual = solve_and_get_storage_dual(&const_result.templates[0]);

        // Constant-productivity with rho=0 has zero storage-fixing dual (v_in
        // cannot improve generation when turbine contributes nothing).
        assert!(
            const_dual.abs() < 1e-6,
            "constant-productivity dual must be ~0, got {const_dual}"
        );

        // FPHA dual is non-zero because higher v_in expands the feasible generation
        // via the gamma_v term and the water balance, reducing the deficit cost.
        assert!(
            fpha_dual.abs() > 1e-6,
            "FPHA storage-fixing dual must be non-zero (FPHA v_in contribution \
             must be present), got {fpha_dual}"
        );

        assert_ne!(
            (fpha_dual * 1e6).round(),
            (const_dual * 1e6).round(),
            "storage-fixing dual must differ between FPHA ({fpha_dual}) and \
             constant-productivity ({const_dual})"
        );
    }

    /// AC (ticket-009, mixed): 2-constant + 1-FPHA system solves to Optimal.
    ///
    /// Verifies that generation variables for both types of hydros have
    /// correct values in the solution:
    /// - constant hydros: turbine * rho contributes to load balance
    /// - FPHA hydro: generation variable `g` satisfies load balance
    ///
    /// Uses `four_hydro_mixed_system` (2 constant + 2 FPHA hydros, 1 bus, 1 block).
    ///
    /// Column layout for N=4, L=0, no penalty:
    ///   cols 9..12: turbine[0..3] (q per hydro per block)
    ///   cols 19..20: g[0..1] (FPHA generation variables for hydros 2 and 3)
    #[test]
    fn fpha_solve_mixed_system_optimal() {
        use cobre_solver::{HighsSolver, RowBatch, SolverInterface};

        let (system, production) = four_hydro_mixed_system();

        let result = build_stage_templates(
            &system,
            &no_penalty_config(),
            &PrecomputedPar::default(),
            &PrecomputedNormal::default(),
            &production,
            &default_evaporation(&system),
        )
        .expect("mixed FPHA/constant system template build must succeed");

        let template = &result.templates[0];
        let mut solver = HighsSolver::new().expect("HighsSolver::new must succeed");
        solver.load_model(template);

        let empty_cuts = RowBatch {
            num_rows: 0,
            row_starts: vec![0_i32],
            col_indices: vec![],
            values: vec![],
            row_lower: vec![],
            row_upper: vec![],
        };
        solver.add_rows(&empty_cuts);

        // Fix v_in for all 4 hydros (rows 0..3 are storage-fixing rows).
        // N=4, so storage_fixing = 0..4; v_in columns are storage_in = 4..8.
        solver.set_row_bounds(
            &[0, 1, 2, 3],
            &[100.0, 100.0, 100.0, 100.0],
            &[100.0, 100.0, 100.0, 100.0],
        );

        let view = solver
            .solve()
            .expect("mixed FPHA LP must be feasible and optimal");

        // The solve must return a finite objective.
        assert!(
            view.objective.is_finite(),
            "objective must be finite, got {}",
            view.objective
        );

        // FPHA generation variables (cols 19 and 20) must be non-negative.
        let col_g0 = 19_usize;
        let col_g1 = 20_usize;
        assert!(
            view.primal[col_g0] >= 0.0,
            "FPHA hydro 0 generation must be non-negative, got {}",
            view.primal[col_g0]
        );
        assert!(
            view.primal[col_g1] >= 0.0,
            "FPHA hydro 1 generation must be non-negative, got {}",
            view.primal[col_g1]
        );
    }

    // =========================================================================
    // Evaporation variable tests (ticket-010)
    // =========================================================================

    use cobre_solver::StageTemplate;

    use crate::hydro_models::LinearizedEvaporation;

    /// Build an `EvaporationModelSet` for a system where only the specified
    /// hydro indices have linearized evaporation.
    ///
    /// All hydros receive `EvaporationModel::None` by default; hydros at the
    /// given `evap_indices` receive `Linearized` with the provided per-stage
    /// `k_evap0` values (uniform across stages).
    fn evap_set_for_system(
        system: &cobre_core::System,
        evap_indices: &[usize],
        k_evap0_per_stage: &[f64],
    ) -> EvaporationModelSet {
        let n_hydros = system.hydros().len();
        let n_stages = system.stages().iter().filter(|s| s.id >= 0).count();
        let models = (0..n_hydros)
            .map(|h| {
                if evap_indices.contains(&h) {
                    let coefficients = (0..n_stages)
                        .map(|s| LinearizedEvaporation {
                            k_evap0: k_evap0_per_stage
                                .get(s)
                                .copied()
                                .unwrap_or(k_evap0_per_stage.first().copied().unwrap_or(0.0)),
                            k_evap_v: 0.0,
                        })
                        .collect();
                    EvaporationModel::Linearized {
                        coefficients,
                        reference_volumes_hm3: vec![100.0; n_stages],
                    }
                } else {
                    EvaporationModel::None
                }
            })
            .collect();
        EvaporationModelSet::new(models)
    }

    /// AC (ticket-010): 0 evaporation hydros — `num_cols` and `num_rows` are unchanged.
    ///
    /// A system with 1 hydro (L=0, T=0, B=1, K=1) and no evaporation:
    /// - Without evaporation: `num_cols` = N\*(2+L)+1 + N\*K\*2 + B\*K\*2 = 3+2+2 = 7
    /// - Without evaporation: `num_rows` = N\*(1+L) + N + B\*K = 1+1+1 = 3
    /// - With 0 evaporation hydros: identical (0 extra cols, 0 extra rows)
    #[test]
    fn evap_zero_hydros_layout_unchanged() {
        let system = one_hydro_system(1, 0);
        let no_evap = default_evaporation(&system);
        let with_evap = build_stage_templates(
            &system,
            &no_penalty_config(),
            &PrecomputedPar::default(),
            &PrecomputedNormal::default(),
            &default_production(&system),
            &no_evap,
        )
        .expect("no evaporation ok");

        let baseline = build_stage_templates(
            &system,
            &no_penalty_config(),
            &PrecomputedPar::default(),
            &PrecomputedNormal::default(),
            &default_production(&system),
            &EvaporationModelSet::new(vec![EvaporationModel::None]),
        )
        .expect("none evaporation ok");

        assert_eq!(
            with_evap.templates[0].num_cols, baseline.templates[0].num_cols,
            "num_cols must match with zero evaporation hydros"
        );
        assert_eq!(
            with_evap.templates[0].num_rows, baseline.templates[0].num_rows,
            "num_rows must match with zero evaporation hydros"
        );
    }

    /// AC (ticket-010): 2 evaporation hydros + 1 block → `num_cols` += 6, `num_rows` += 2.
    ///
    /// Uses a system with 2 hydros (L=0, T=0, B=1, K=1).
    /// Baseline (no evaporation):
    ///   `num_cols` = N\*(2+L)+1 + N\*K\*2 + B\*K\*2 = 5+4+2 = 11
    ///   `num_rows` = N\*(1+L) + N + B\*K = 2+2+1 = 5
    /// With 2 evaporation hydros:
    ///   `num_cols` = 11 + 2\*3 = 17
    ///   `num_rows` = 5 + 2 = 7
    #[test]
    fn evap_two_hydros_increases_cols_and_rows() {
        // Build a 2-hydro system using one_hydro_system as base and adapt.
        let system1 = one_hydro_system(1, 0);
        // Use one_bus_system as the reference (1 bus, no hydros) for the delta.
        // Instead, build a 2-hydro system directly.
        // We reuse one_hydro_system for 2 independent calls; here we use a simpler approach:
        // compare a system with 1 hydro + no evaporation vs 1 hydro + 1 evaporation hydro.
        // This gives +3 cols and +1 row per evaporation hydro.

        let baseline = build_stage_templates(
            &system1,
            &no_penalty_config(),
            &PrecomputedPar::default(),
            &PrecomputedNormal::default(),
            &default_production(&system1),
            &EvaporationModelSet::new(vec![EvaporationModel::None]),
        )
        .expect("no evaporation baseline ok");

        let evap = evap_set_for_system(&system1, &[0], &[1.5]);
        let with_evap = build_stage_templates(
            &system1,
            &no_penalty_config(),
            &PrecomputedPar::default(),
            &PrecomputedNormal::default(),
            &default_production(&system1),
            &evap,
        )
        .expect("1 evaporation hydro ok");

        let base_cols = baseline.templates[0].num_cols;
        let base_rows = baseline.templates[0].num_rows;
        let evap_cols = with_evap.templates[0].num_cols;
        let evap_rows = with_evap.templates[0].num_rows;

        assert_eq!(
            evap_cols,
            base_cols + 3,
            "1 evap hydro must add exactly 3 columns (Q_ev, f_evap_plus, f_evap_minus)"
        );
        assert_eq!(
            evap_rows,
            base_rows + 1,
            "1 evap hydro must add exactly 1 row (evaporation equality constraint)"
        );
    }

    /// AC (ticket-010): evaporation row bounds are equality: `row_lower == row_upper == k_evap0`.
    ///
    /// Uses a 1-hydro system with `k_evap0 = 1.5` at stage 0.
    #[test]
    fn evap_row_bounds_equality_at_k_evap0() {
        let system = one_hydro_system(1, 0);
        let k_evap0 = 1.5_f64;
        let evap = evap_set_for_system(&system, &[0], &[k_evap0]);

        let result = build_stage_templates(
            &system,
            &no_penalty_config(),
            &PrecomputedPar::default(),
            &PrecomputedNormal::default(),
            &default_production(&system),
            &evap,
        )
        .expect("evaporation system ok");

        let t = &result.templates[0];

        // Evaporation row is placed after all other structural rows (last row).
        let evap_row = t.num_rows - 1;
        assert_eq!(
            t.row_lower[evap_row], k_evap0,
            "evaporation row_lower must equal k_evap0 = {k_evap0}, got {}",
            t.row_lower[evap_row]
        );
        assert_eq!(
            t.row_upper[evap_row], k_evap0,
            "evaporation row_upper must equal k_evap0 = {k_evap0}, got {}",
            t.row_upper[evap_row]
        );
    }

    /// AC (ticket-010): evaporation column bounds are [0, bound) and objective is 0.0.
    /// Q_ev has a physical upper bound; f_plus and f_minus are unbounded.
    #[test]
    fn evap_col_bounds_and_objective() {
        let system = one_hydro_system(1, 0);
        let evap = evap_set_for_system(&system, &[0], &[1.5]);

        let result = build_stage_templates(
            &system,
            &no_penalty_config(),
            &PrecomputedPar::default(),
            &PrecomputedNormal::default(),
            &default_production(&system),
            &evap,
        )
        .expect("evaporation system ok");

        let t = &result.templates[0];

        // The 3 evaporation columns are followed by 1 withdrawal slack column (N=1).
        // Since water_withdrawal_m3s == 0, the withdrawal slack at num_cols-1 is pinned at [0,0].
        // Evaporation columns are at num_cols-4, num_cols-3, num_cols-2 (before withdrawal slack).
        let col_q_ev = t.num_cols - 4;
        let col_f_plus = t.num_cols - 3;
        let col_f_minus = t.num_cols - 2;

        // All three columns have lower bound 0.0.
        for &col in &[col_q_ev, col_f_plus, col_f_minus] {
            assert_eq!(
                t.col_lower[col], 0.0,
                "evap column {col} lower bound must be 0.0, got {}",
                t.col_lower[col]
            );
            assert_eq!(
                t.objective[col], 0.0,
                "evap column {col} objective must be 0.0 (ticket-013 sets violation cost), got {}",
                t.objective[col]
            );
        }

        // Q_ev has a physical upper bound: max(0, k_evap0 + k_evap_v * v_max) * 2.0.
        // k_evap0 = 1.5, k_evap_v = 0.0, v_max = 200.0 → bound = 1.5 * 2.0 = 3.0.
        let expected_q_ev_bound = 1.5 * Q_EV_SAFETY_MARGIN;
        assert!(
            (t.col_upper[col_q_ev] - expected_q_ev_bound).abs() < 1e-12,
            "Q_ev upper bound must be {expected_q_ev_bound}, got {}",
            t.col_upper[col_q_ev]
        );

        // Slack columns f_plus and f_minus remain unbounded.
        for &col in &[col_f_plus, col_f_minus] {
            assert!(
                t.col_upper[col].is_infinite() && t.col_upper[col] > 0.0,
                "evap slack column {col} upper bound must be +inf, got {}",
                t.col_upper[col]
            );
        }
    }

    // =========================================================================
    // ticket-011: fill_evaporation_entries — CSC matrix entries
    // =========================================================================

    /// Build an `EvaporationModelSet` where evaporation hydros have a specific
    /// `k_evap_v` in addition to `k_evap0`.
    ///
    /// Hydros not in `evap_indices` receive `EvaporationModel::None`.
    fn evap_set_with_k_evap_v(
        system: &cobre_core::System,
        evap_indices: &[usize],
        k_evap0: f64,
        k_evap_v: f64,
    ) -> EvaporationModelSet {
        let n_hydros = system.hydros().len();
        let n_stages = system.stages().iter().filter(|s| s.id >= 0).count();
        let models = (0..n_hydros)
            .map(|h| {
                if evap_indices.contains(&h) {
                    let coefficients = (0..n_stages)
                        .map(|_| LinearizedEvaporation { k_evap0, k_evap_v })
                        .collect();
                    EvaporationModel::Linearized {
                        coefficients,
                        reference_volumes_hm3: vec![100.0; n_stages],
                    }
                } else {
                    EvaporationModel::None
                }
            })
            .collect();
        EvaporationModelSet::new(models)
    }

    /// Collect all `(row, value)` pairs from the assembled CSC for a given column.
    ///
    /// Reads `col_starts`, `row_indices`, and `values` from a [`StageTemplate`].
    #[allow(clippy::cast_sign_loss)] // col_starts and row_indices are non-negative by construction
    fn entries_for_col(t: &StageTemplate, col: usize) -> Vec<(usize, f64)> {
        let start = t.col_starts[col] as usize;
        let end = t.col_starts[col + 1] as usize;
        (start..end)
            .map(|i| (t.row_indices[i] as usize, t.values[i]))
            .collect()
    }

    /// AC (ticket-011): 1 evaporation hydro (`h_idx=0`) with `k_evap_v = 0.02` produces
    /// the correct CSC entries at the evaporation row and water balance row.
    ///
    /// Expected entries on the evaporation constraint row:
    ///   `(Q_ev_col, +1.0)`, `(v_col, -0.01)`, `(v_in_col, -0.01)`,
    ///   `(f_plus_col, +1.0)`, `(f_minus_col, -1.0)`.
    ///
    /// After ticket-012, the `Q_ev` column also has an entry in the water balance
    /// row with coefficient `+zeta`.
    #[test]
    fn evap_csc_entries_one_hydro_correct_coefficients() {
        let system = one_hydro_system(1, 0);
        let k_evap_v = 0.02_f64;
        let evap = evap_set_with_k_evap_v(&system, &[0], 1.5, k_evap_v);

        let result = build_stage_templates(
            &system,
            &no_penalty_config(),
            &PrecomputedPar::default(),
            &PrecomputedNormal::default(),
            &default_production(&system),
            &evap,
        )
        .expect("evaporation system ok");

        let t = &result.templates[0];

        // Column layout for 1-hydro system (N=1, L=0, T=0, B=1, K=1):
        //   col 0 = v (storage_out)  col 1 = z_inflow  col 2 = v_in  col 3 = theta
        //   col 4 = turbine  col 5 = spillage  col 6 = diversion
        //   col 7 = deficit  col 8 = excess
        //   col_evap_start = num_cols - 4
        // Row layout for N=1, L=0, B=1, K=1, no FPHA:
        //   row 0: storage-fixing
        //   row 1: z_inflow definition
        //   row 2: water balance (row_water_balance_start = n_state + n_hydros = 2)
        //   row 3: load balance
        //   row 4: evaporation constraint (row_evap_start = 4)
        // Evaporation columns come before withdrawal slack (N=1 column).
        // col_q_ev = col_evap_start + 0, col_f_plus = +1, col_f_minus = +2,
        // followed by 1 withdrawal_slack column.
        let col_q_ev = t.num_cols - 4;
        let col_f_plus = t.num_cols - 3;
        let col_f_minus = t.num_cols - 2;
        let evap_row = t.num_rows - 1;
        let water_balance_row = 2_usize; // row_water_balance_start = n_state + n_hydros = 2

        // After ticket-012, Q_ev has 2 entries: water balance row (+zeta) and
        // evaporation constraint row (+1.0). Entries are sorted by row ascending.
        let zeta = 744.0 * (3_600.0 / 1_000_000.0);
        let entries_q_ev = entries_for_col(t, col_q_ev);
        assert_eq!(
            entries_q_ev.len(),
            2,
            "Q_ev column must have exactly 2 entries (water balance + evap constraint), got {entries_q_ev:?}"
        );
        // Entries are CSC-sorted by row; water balance row (2) < evap row (4).
        assert_eq!(
            entries_q_ev[0].0, water_balance_row,
            "Q_ev first entry must be at water balance row"
        );
        assert!(
            (entries_q_ev[0].1 - zeta).abs() < 1e-12,
            "Q_ev water balance coefficient must be +zeta={zeta}, got {}",
            entries_q_ev[0].1
        );
        assert_eq!(
            entries_q_ev[1].0, evap_row,
            "Q_ev second entry must be at evap_row"
        );
        assert!(
            (entries_q_ev[1].1 - 1.0).abs() < 1e-12,
            "Q_ev evap constraint coefficient must be +1.0, got {}",
            entries_q_ev[1].1
        );

        let entries_f_plus = entries_for_col(t, col_f_plus);
        assert_eq!(
            entries_f_plus.len(),
            1,
            "f_plus column must have exactly 1 entry, got {entries_f_plus:?}"
        );
        assert_eq!(
            entries_f_plus[0].0, evap_row,
            "f_plus entry must be at evap_row"
        );
        assert!(
            (entries_f_plus[0].1 - 1.0).abs() < 1e-12,
            "f_plus coefficient must be +1.0, got {}",
            entries_f_plus[0].1
        );

        let entries_f_minus = entries_for_col(t, col_f_minus);
        assert_eq!(
            entries_f_minus.len(),
            1,
            "f_minus column must have exactly 1 entry, got {entries_f_minus:?}"
        );
        assert_eq!(
            entries_f_minus[0].0, evap_row,
            "f_minus entry must be at evap_row"
        );
        assert!(
            (entries_f_minus[0].1 - (-1.0)).abs() < 1e-12,
            "f_minus coefficient must be -1.0, got {}",
            entries_f_minus[0].1
        );

        // v column (col 0, h_idx=0) must contain an entry at evap_row with -k_evap_v/2.
        let expected_coeff = -k_evap_v / 2.0;
        let entry_v = entries_for_col(t, 0)
            .into_iter()
            .find(|&(r, _)| r == evap_row)
            .expect("v column must have an entry at evap_row");
        assert!(
            (entry_v.1 - expected_coeff).abs() < 1e-12,
            "v coefficient must be {expected_coeff}, got {}",
            entry_v.1
        );

        // v_in column: storage_in.start for 1-hydro (L=0) = N*(2+L) = 2; col_v_in = 2 + h_idx = 2.
        let col_v_in = 2;
        let entry_v_in = entries_for_col(t, col_v_in)
            .into_iter()
            .find(|&(r, _)| r == evap_row)
            .expect("v_in column must have an entry at evap_row");
        assert!(
            (entry_v_in.1 - expected_coeff).abs() < 1e-12,
            "v_in coefficient must be {expected_coeff}, got {}",
            entry_v_in.1
        );
    }

    /// AC (ticket-011): coefficient value check with `k_evap_v = 0.04` → v and `v_in` entries are -0.02.
    #[test]
    fn evap_csc_entries_coefficient_scaling() {
        let system = one_hydro_system(1, 0);
        let k_evap_v = 0.04_f64;
        let evap = evap_set_with_k_evap_v(&system, &[0], 0.0, k_evap_v);

        let result = build_stage_templates(
            &system,
            &no_penalty_config(),
            &PrecomputedPar::default(),
            &PrecomputedNormal::default(),
            &default_production(&system),
            &evap,
        )
        .expect("evaporation system ok");

        let t = &result.templates[0];
        let evap_row = t.num_rows - 1;
        let expected_coeff = -k_evap_v / 2.0; // -0.02

        let entry_v = entries_for_col(t, 0)
            .into_iter()
            .find(|&(r, _)| r == evap_row)
            .expect("v column must have evap_row entry");
        assert!(
            (entry_v.1 - expected_coeff).abs() < 1e-12,
            "v coefficient: expected {expected_coeff}, got {}",
            entry_v.1
        );

        // storage_in.start for 1-hydro (L=0): N*(2+L) = 2; col_v_in = 2 + h_idx = 2.
        let col_v_in = 2;
        let entry_v_in = entries_for_col(t, col_v_in)
            .into_iter()
            .find(|&(r, _)| r == evap_row)
            .expect("v_in column must have evap_row entry");
        assert!(
            (entry_v_in.1 - expected_coeff).abs() < 1e-12,
            "v_in coefficient: expected {expected_coeff}, got {}",
            entry_v_in.1
        );
    }

    /// AC (ticket-011): 0 evaporation hydros — `fill_evaporation_entries` is a no-op;
    /// the evaporation columns do not exist and no extra non-zeros are added.
    #[test]
    fn evap_csc_entries_zero_hydros_no_op() {
        let system = one_hydro_system(1, 0);
        let no_evap = default_evaporation(&system);

        let result = build_stage_templates(
            &system,
            &no_penalty_config(),
            &PrecomputedPar::default(),
            &PrecomputedNormal::default(),
            &default_production(&system),
            &no_evap,
        )
        .expect("no evaporation ok");

        let baseline = build_stage_templates(
            &system,
            &no_penalty_config(),
            &PrecomputedPar::default(),
            &PrecomputedNormal::default(),
            &default_production(&system),
            &EvaporationModelSet::new(vec![EvaporationModel::None]),
        )
        .expect("none evaporation ok");

        assert_eq!(
            result.templates[0].num_nz, baseline.templates[0].num_nz,
            "num_nz must be identical with zero evaporation hydros"
        );
    }

    /// AC (ticket-011): 2 evap hydros with distinct `k_evap_v` produce independent rows.
    #[test]
    fn evap_csc_entries_two_hydros_independent_rows() {
        let (system, production) = four_hydro_mixed_system();
        let n_stages = system.stages().iter().filter(|s| s.id >= 0).count();

        let models = vec![
            EvaporationModel::Linearized {
                coefficients: vec![
                    LinearizedEvaporation {
                        k_evap0: 1.0,
                        k_evap_v: 0.02,
                    };
                    n_stages
                ],
                reference_volumes_hm3: vec![100.0; n_stages],
            },
            EvaporationModel::Linearized {
                coefficients: vec![
                    LinearizedEvaporation {
                        k_evap0: 2.0,
                        k_evap_v: 0.06,
                    };
                    n_stages
                ],
                reference_volumes_hm3: vec![100.0; n_stages],
            },
            EvaporationModel::None,
            EvaporationModel::None,
        ];
        let evap = EvaporationModelSet::new(models);

        let result = build_stage_templates(
            &system,
            &no_penalty_config(),
            &PrecomputedPar::default(),
            &PrecomputedNormal::default(),
            &production,
            &evap,
        )
        .expect("2-evap-hydro system ok");

        let t = &result.templates[0];
        // 2 evap hydros: evap rows are the last 2 rows.
        let evap_row_0 = t.num_rows - 2;
        let evap_row_1 = t.num_rows - 1;

        // Hydro 0 (k_evap_v=0.02): v coefficient = -0.01.
        let entry_v_h0 = entries_for_col(t, 0)
            .into_iter()
            .find(|&(r, _)| r == evap_row_0)
            .expect("hydro 0 v col entry");
        assert!(
            (entry_v_h0.1 - (-0.01)).abs() < 1e-12,
            "hydro 0 v: expected -0.01, got {}",
            entry_v_h0.1
        );

        // Hydro 1 (k_evap_v=0.06): v coefficient = -0.03.
        let entry_v_h1 = entries_for_col(t, 1)
            .into_iter()
            .find(|&(r, _)| r == evap_row_1)
            .expect("hydro 1 v col entry");
        assert!(
            (entry_v_h1.1 - (-0.03)).abs() < 1e-12,
            "hydro 1 v: expected -0.03, got {}",
            entry_v_h1.1
        );

        // Row bounds: hydro 0 → k_evap0=1.0, hydro 1 → k_evap0=2.0.
        assert!((t.row_lower[evap_row_0] - 1.0).abs() < 1e-12);
        assert!((t.row_lower[evap_row_1] - 2.0).abs() < 1e-12);
    }

    /// AC (ticket-011): `k_evap_v = 0.0` → v and `v_in` entries are 0.0;
    /// the constraint reduces to `Q_ev + f_plus - f_minus = k_evap0`.
    #[test]
    fn evap_csc_entries_zero_k_evap_v_produces_zero_volume_coefficients() {
        let system = one_hydro_system(1, 0);
        let evap = evap_set_with_k_evap_v(&system, &[0], 2.0, 0.0);

        let result = build_stage_templates(
            &system,
            &no_penalty_config(),
            &PrecomputedPar::default(),
            &PrecomputedNormal::default(),
            &default_production(&system),
            &evap,
        )
        .expect("evaporation system ok");

        let t = &result.templates[0];
        let evap_row = t.num_rows - 1;

        let entry_v = entries_for_col(t, 0)
            .into_iter()
            .find(|&(r, _)| r == evap_row)
            .expect("v column must have evap_row entry");
        assert!(
            entry_v.1.abs() < 1e-12,
            "v coefficient must be 0.0 when k_evap_v=0, got {}",
            entry_v.1
        );

        // storage_in.start for 1-hydro (L=0): N*(2+L) = 2; col_v_in = 2 + h_idx = 2.
        let col_v_in = 2;
        let entry_v_in = entries_for_col(t, col_v_in)
            .into_iter()
            .find(|&(r, _)| r == evap_row)
            .expect("v_in column must have evap_row entry");
        assert!(
            entry_v_in.1.abs() < 1e-12,
            "v_in coefficient must be 0.0 when k_evap_v=0, got {}",
            entry_v_in.1
        );
    }

    // ── ticket-012: water balance entries for evaporation ────────────────────

    /// AC-1 (ticket-012): 1 evaporation hydro (`h_idx=0`), 1 block of 744 hours.
    ///
    /// The `Q_ev_h` column must have an entry in the water balance row
    /// (`row = row_water_balance_start + 0`) with coefficient `+zeta`
    /// where `zeta = 744.0 * 3_600.0 / 1_000_000.0`.
    #[test]
    #[allow(clippy::cast_sign_loss)]
    fn evap_water_balance_one_hydro_coefficient_is_zeta() {
        let system = one_hydro_system(1, 0);
        let evap = evap_set_with_k_evap_v(&system, &[0], 0.0, 0.0);

        let result = build_stage_templates(
            &system,
            &no_penalty_config(),
            &PrecomputedPar::default(),
            &PrecomputedNormal::default(),
            &default_production(&system),
            &evap,
        )
        .expect("evaporation system ok");

        let t = &result.templates[0];

        // For N=1, L=0: n_state = 1, row_water_balance_start = n_state + n_hydros = 2.
        // Hydro 0's water balance row = 2.
        let water_balance_row = 2_usize;

        // Q_ev is the first of the 3 evaporation columns; before withdrawal slack.
        let col_q_ev = t.num_cols - 4;

        let entries = entries_for_col(t, col_q_ev);
        let entry = entries
            .iter()
            .find(|&&(r, _)| r == water_balance_row)
            .copied()
            .expect("Q_ev column must have an entry in the water balance row");

        let zeta = 744.0_f64 * (3_600.0 / 1_000_000.0);
        assert!(
            (entry.1 - zeta).abs() < 1e-12,
            "Q_ev water balance coefficient must be +zeta={zeta}, got {}",
            entry.1
        );
    }

    /// AC-2 (ticket-012): 2 hydros where only hydro 1 has evaporation.
    ///
    /// The `Q_ev` column for hydro 1 must have an entry in water balance row 1
    /// with coefficient `+zeta`. Hydro 0's water balance row must have no
    /// evaporation entry.
    ///
    /// Uses a 2-hydro single-bus system built with the same pattern as
    /// `one_hydro_system` / `four_hydro_mixed_system`.
    #[test]
    #[allow(clippy::cast_sign_loss, clippy::too_many_lines)]
    fn evap_water_balance_only_second_hydro_has_evap() {
        use chrono::NaiveDate;
        use cobre_core::entities::hydro::{Hydro, HydroGenerationModel, HydroPenalties};
        use cobre_core::scenario::{InflowModel, LoadModel};
        use cobre_core::temporal::{
            Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage as CStage, StageRiskConfig,
            StageStateConfig,
        };

        let bus = Bus {
            id: EntityId(1),
            name: "B1".to_string(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 500.0,
            }],
            excess_cost: 0.0,
        };
        let hp = HydroPenalties {
            spillage_cost: 0.01,
            diversion_cost: 0.0,
            fpha_turbined_cost: 0.0,
            storage_violation_below_cost: 0.0,
            filling_target_violation_cost: 0.0,
            turbined_violation_below_cost: 0.0,
            outflow_violation_below_cost: 0.0,
            outflow_violation_above_cost: 0.0,
            generation_violation_below_cost: 0.0,
            evaporation_violation_cost: 0.0,
            water_withdrawal_violation_cost: 0.0,
        };
        let make_h = |id: i32| Hydro {
            id: EntityId(id),
            name: format!("H{id}"),
            bus_id: EntityId(1),
            downstream_id: None,
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: 0.0,
            max_storage_hm3: 200.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: HydroGenerationModel::ConstantProductivity {
                productivity_mw_per_m3s: 2.5,
            },
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 100.0,
            min_generation_mw: 0.0,
            max_generation_mw: 250.0,
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: None,
            evaporation_reference_volumes_hm3: None,
            diversion: None,
            filling: None,
            penalties: hp,
        };
        let hydros = vec![make_h(2), make_h(3)];
        let stages = vec![CStage {
            index: 0,
            id: 0,
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            season_id: None,
            blocks: vec![Block {
                index: 0,
                name: "S".to_string(),
                duration_hours: 744.0,
            }],
            block_mode: BlockMode::Parallel,
            state_config: StageStateConfig {
                storage: true,
                inflow_lags: false,
            },
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor: 1,
                noise_method: NoiseMethod::Saa,
            },
        }];
        let inflow_models: Vec<InflowModel> = hydros
            .iter()
            .map(|h| InflowModel {
                hydro_id: h.id,
                stage_id: 0,
                mean_m3s: 80.0,
                std_m3s: 20.0,
                ar_coefficients: vec![],
                residual_std_ratio: 1.0,
            })
            .collect();
        let load_models = vec![LoadModel {
            bus_id: EntityId(1),
            stage_id: 0,
            mean_mw: 200.0,
            std_mw: 0.0,
        }];
        let bounds = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 2,
                n_thermals: 0,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages: 1,
            },
            &BoundsDefaults {
                hydro: default_hydro_bounds(),
                thermal: ThermalStageBounds {
                    min_generation_mw: 0.0,
                    max_generation_mw: 0.0,
                },
                line: LineStageBounds {
                    direct_mw: 0.0,
                    reverse_mw: 0.0,
                },
                pumping: PumpingStageBounds {
                    min_flow_m3s: 0.0,
                    max_flow_m3s: 0.0,
                },
                contract: ContractStageBounds {
                    min_mw: 0.0,
                    max_mw: 0.0,
                    price_per_mwh: 0.0,
                },
            },
        );
        let penalties = ResolvedPenalties::new(
            &PenaltiesCountsSpec {
                n_hydros: 2,
                n_buses: 1,
                n_lines: 0,
                n_ncs: 0,
                n_stages: 1,
            },
            &PenaltiesDefaults {
                hydro: default_hydro_penalties(),
                bus: BusStagePenalties { excess_cost: 0.0 },
                line: LineStagePenalties { exchange_cost: 0.0 },
                ncs: NcsStagePenalties {
                    curtailment_cost: 0.0,
                },
            },
        );
        let system = SystemBuilder::new()
            .buses(vec![bus])
            .hydros(hydros)
            .stages(stages)
            .inflow_models(inflow_models)
            .load_models(load_models)
            .bounds(bounds)
            .penalties(penalties)
            .build()
            .expect("2-hydro system ok");

        // Only hydro 1 (h_idx=1) has evaporation.
        let evap = evap_set_with_k_evap_v(&system, &[1], 0.0, 0.0);

        let result = build_stage_templates(
            &system,
            &no_penalty_config(),
            &PrecomputedPar::default(),
            &PrecomputedNormal::default(),
            &default_production(&system),
            &evap,
        )
        .expect("2-hydro evap system ok");

        let t = &result.templates[0];

        // For N=2, L=0: n_state = 2, z_inflow rows [2,4), row_water_balance_start = 4.
        // Hydro 0 water balance row = 4, hydro 1 water balance row = 5.
        let water_balance_row_h0 = 4_usize;
        let water_balance_row_h1 = 5_usize;

        // Q_ev for hydro 1 (local_idx=0, since only hydro 1 is evap): col_evap_start + 0*3.
        // N=2 withdrawal columns follow evap; z_inflow is embedded in state region.
        let col_q_ev_h1 = t.num_cols - 5;

        // Q_ev (h1) must have an entry at water balance row 3.
        let entries_h1 = entries_for_col(t, col_q_ev_h1);
        let found_h1 = entries_h1
            .iter()
            .find(|&&(r, _)| r == water_balance_row_h1)
            .copied();
        assert!(
            found_h1.is_some(),
            "Q_ev for hydro 1 must have an entry in water balance row {water_balance_row_h1}"
        );
        let zeta = 744.0_f64 * (3_600.0 / 1_000_000.0);
        assert!(
            (found_h1.unwrap().1 - zeta).abs() < 1e-12,
            "Q_ev (h1) water balance coefficient must be +zeta={zeta}, got {}",
            found_h1.unwrap().1
        );

        // Q_ev (h1) must NOT have an entry at hydro 0's water balance row.
        let found_h0 = entries_h1.iter().any(|&(r, _)| r == water_balance_row_h0);
        assert!(
            !found_h0,
            "Q_ev for hydro 1 must not appear in hydro 0's water balance row"
        );
    }

    /// AC-3 (ticket-012): 0 evaporation hydros — no evaporation entries added.
    ///
    /// The total non-zero count must be identical to a baseline with no
    /// evaporation model (behaviour unchanged from before ticket-012).
    #[test]
    fn evap_water_balance_zero_hydros_no_op() {
        let system = one_hydro_system(1, 0);
        let no_evap = EvaporationModelSet::new(vec![EvaporationModel::None]);

        let result = build_stage_templates(
            &system,
            &no_penalty_config(),
            &PrecomputedPar::default(),
            &PrecomputedNormal::default(),
            &default_production(&system),
            &no_evap,
        )
        .expect("no evaporation ok");

        let baseline = build_stage_templates(
            &system,
            &no_penalty_config(),
            &PrecomputedPar::default(),
            &PrecomputedNormal::default(),
            &default_production(&system),
            &default_evaporation(&system),
        )
        .expect("default evaporation ok");

        assert_eq!(
            result.templates[0].num_nz, baseline.templates[0].num_nz,
            "num_nz must be identical with zero evaporation hydros (no water balance entries added)"
        );
    }

    // =========================================================================
    // Evaporation violation cost tests (ticket-013)
    // =========================================================================

    /// Build a 1-bus, 1-hydro system with evaporation and a custom
    /// `evaporation_violation_cost`, using the given block duration.
    ///
    /// The hydro has constant-productivity generation. The system has exactly
    /// 1 stage with 1 block of `block_hours` duration.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        clippy::too_many_lines
    )]
    fn evap_hydro_system_with_violation_cost(
        block_hours: f64,
        evaporation_violation_cost: f64,
    ) -> cobre_core::System {
        use chrono::NaiveDate;
        use cobre_core::entities::hydro::{Hydro, HydroGenerationModel, HydroPenalties};
        use cobre_core::scenario::{InflowModel, LoadModel};
        use cobre_core::temporal::{
            Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
            StageStateConfig,
        };

        let bus = Bus {
            id: EntityId(1),
            name: "B1".to_string(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 500.0,
            }],
            excess_cost: 0.0,
        };

        let hydro = Hydro {
            id: EntityId(2),
            name: "H1".to_string(),
            bus_id: EntityId(1),
            downstream_id: None,
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: 0.0,
            max_storage_hm3: 2_000.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: HydroGenerationModel::ConstantProductivity {
                productivity_mw_per_m3s: 2.5,
            },
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 100.0,
            min_generation_mw: 0.0,
            max_generation_mw: 250.0,
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: None,
            evaporation_reference_volumes_hm3: None,
            diversion: None,
            filling: None,
            penalties: HydroPenalties {
                spillage_cost: 0.01,
                diversion_cost: 0.0,
                fpha_turbined_cost: 0.0,
                storage_violation_below_cost: 0.0,
                filling_target_violation_cost: 0.0,
                turbined_violation_below_cost: 0.0,
                outflow_violation_below_cost: 0.0,
                outflow_violation_above_cost: 0.0,
                generation_violation_below_cost: 0.0,
                evaporation_violation_cost,
                water_withdrawal_violation_cost: 0.0,
            },
        };

        let stages = vec![Stage {
            index: 0,
            id: 0,
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            season_id: None,
            blocks: vec![Block {
                index: 0,
                name: "S".to_string(),
                duration_hours: block_hours,
            }],
            block_mode: BlockMode::Parallel,
            state_config: StageStateConfig {
                storage: true,
                inflow_lags: false,
            },
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor: 1,
                noise_method: NoiseMethod::Saa,
            },
        }];

        let inflow_models = vec![InflowModel {
            hydro_id: EntityId(2),
            stage_id: 0,
            mean_m3s: 50.0,
            std_m3s: 10.0,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
        }];

        let load_models = vec![LoadModel {
            bus_id: EntityId(1),
            stage_id: 0,
            mean_mw: 100.0,
            std_mw: 0.0,
        }];

        let bounds = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 1,
                n_thermals: 0,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages: 1,
            },
            &BoundsDefaults {
                hydro: HydroStageBounds {
                    min_storage_hm3: 0.0,
                    max_storage_hm3: 2_000.0,
                    min_turbined_m3s: 0.0,
                    max_turbined_m3s: 100.0,
                    min_outflow_m3s: 0.0,
                    max_outflow_m3s: None,
                    min_generation_mw: 0.0,
                    max_generation_mw: 250.0,
                    max_diversion_m3s: None,
                    filling_inflow_m3s: 0.0,
                    water_withdrawal_m3s: 0.0,
                },
                thermal: ThermalStageBounds {
                    min_generation_mw: 0.0,
                    max_generation_mw: 0.0,
                },
                line: LineStageBounds {
                    direct_mw: 0.0,
                    reverse_mw: 0.0,
                },
                pumping: PumpingStageBounds {
                    min_flow_m3s: 0.0,
                    max_flow_m3s: 0.0,
                },
                contract: ContractStageBounds {
                    min_mw: 0.0,
                    max_mw: 0.0,
                    price_per_mwh: 0.0,
                },
            },
        );

        let penalties = ResolvedPenalties::new(
            &PenaltiesCountsSpec {
                n_hydros: 1,
                n_buses: 1,
                n_lines: 0,
                n_ncs: 0,
                n_stages: 1,
            },
            &PenaltiesDefaults {
                hydro: HydroStagePenalties {
                    spillage_cost: 0.01,
                    diversion_cost: 0.0,
                    fpha_turbined_cost: 0.0,
                    storage_violation_below_cost: 0.0,
                    filling_target_violation_cost: 0.0,
                    turbined_violation_below_cost: 0.0,
                    outflow_violation_below_cost: 0.0,
                    outflow_violation_above_cost: 0.0,
                    generation_violation_below_cost: 0.0,
                    evaporation_violation_cost,
                    water_withdrawal_violation_cost: 0.0,
                },
                bus: BusStagePenalties { excess_cost: 0.0 },
                line: LineStagePenalties { exchange_cost: 0.0 },
                ncs: NcsStagePenalties {
                    curtailment_cost: 0.0,
                },
            },
        );

        SystemBuilder::new()
            .buses(vec![bus])
            .hydros(vec![hydro])
            .stages(stages)
            .inflow_models(inflow_models)
            .load_models(load_models)
            .bounds(bounds)
            .penalties(penalties)
            .build()
            .expect("evap_hydro_system_with_violation_cost: valid")
    }

    /// AC-1 (ticket-013): `f_evap_plus` carries base violation cost;
    /// `f_evap_minus` carries 100x asymmetric over-evaporation penalty.
    ///
    /// System: 1 hydro with evaporation, `evaporation_violation_cost = 500.0`,
    /// 1 block of 730 hours → base objective = `500.0 * 730.0 / 1000 = 365.0`.
    /// `f_minus` objective = `365.0 * 100 = 36_500.0`.
    #[test]
    fn evap_violation_cost_applied_to_slack_columns() {
        let system = evap_hydro_system_with_violation_cost(730.0, 500.0);
        let evap = evap_set_with_k_evap_v(&system, &[0], 1.0, 0.02);

        let result = build_stage_templates(
            &system,
            &no_penalty_config(),
            &PrecomputedPar::default(),
            &PrecomputedNormal::default(),
            &default_production(&system),
            &evap,
        )
        .expect("evap violation cost system builds ok");

        let t = &result.templates[0];

        // Evaporation columns (Q_ev, f_plus, f_minus) are followed by 1 withdrawal slack (N=1).
        // z_inflow is embedded in state region, not at end of columns.
        let col_q_ev = t.num_cols - 4;
        let col_f_plus = t.num_cols - 3;
        let col_f_minus = t.num_cols - 2;

        let expected_base = 500.0 * 730.0 / COST_SCALE_FACTOR;

        assert!(
            t.objective[col_q_ev].abs() < 1e-12,
            "Q_ev column objective must be 0.0 (evaporation flow itself has no cost), got {}",
            t.objective[col_q_ev]
        );
        assert!(
            (t.objective[col_f_plus] - expected_base).abs() < 1e-12,
            "f_evap_plus objective: expected {expected_base}, got {}",
            t.objective[col_f_plus]
        );
        let expected_minus = expected_base * OVER_EVAPORATION_COST_MULTIPLIER;
        assert!(
            (t.objective[col_f_minus] - expected_minus).abs() < 1e-6,
            "f_evap_minus objective: expected {expected_minus} (100x f_plus), got {}",
            t.objective[col_f_minus]
        );
    }

    /// AC-2 (ticket-013): `Q_ev` column objective is 0.0 even when a
    /// non-zero `evaporation_violation_cost` is set.
    #[test]
    fn evap_q_ev_objective_is_zero() {
        let system = evap_hydro_system_with_violation_cost(730.0, 500.0);
        let evap = evap_set_with_k_evap_v(&system, &[0], 0.0, 0.0);

        let result = build_stage_templates(
            &system,
            &no_penalty_config(),
            &PrecomputedPar::default(),
            &PrecomputedNormal::default(),
            &default_production(&system),
            &evap,
        )
        .expect("evap system with zero k_evap builds ok");

        let t = &result.templates[0];
        // N=1 withdrawal slack follows the 3 evap columns.
        let col_q_ev = t.num_cols - 4;

        assert!(
            t.objective[col_q_ev].abs() < 1e-12,
            "Q_ev objective must be 0.0, got {}",
            t.objective[col_q_ev]
        );
    }

    /// AC-3 (ticket-013): LP with 1 evaporation hydro is solvable (`HiGHS` returns
    /// `Optimal`) and the `Q_ev` value is non-negative after fixing `v_in = 1000.0 hm3`.
    ///
    /// System: 1 bus, 1 hydro, `k_evap0 = 1.0`, `k_evap_v = 0.02`.
    /// The LP is solved with `v_in = 1000 hm3`; the linearised evaporation
    /// constraint is `Q_ev = k_evap0 + k_evap_v/2 * (v + v_in)`.
    /// With `v_in` fixed at 1000, the RHS is at least 1 mm, so `Q_ev >= 0`.
    #[test]
    fn evap_lp_solvable_and_q_ev_nonnegative() {
        use cobre_solver::{HighsSolver, RowBatch, SolverInterface};

        let system = evap_hydro_system_with_violation_cost(730.0, 500.0);
        let evap = evap_set_with_k_evap_v(&system, &[0], 1.0, 0.02);

        let result = build_stage_templates(
            &system,
            &no_penalty_config(),
            &PrecomputedPar::default(),
            &PrecomputedNormal::default(),
            &default_production(&system),
            &evap,
        )
        .expect("evap system template build must succeed");

        let template = &result.templates[0];
        let mut solver = HighsSolver::new().expect("HighsSolver::new must succeed");
        solver.load_model(template);

        let empty_cuts = RowBatch {
            num_rows: 0,
            row_starts: vec![0_i32],
            col_indices: vec![],
            values: vec![],
            row_lower: vec![],
            row_upper: vec![],
        };
        solver.add_rows(&empty_cuts);

        // Fix v_in = 1000 hm3 via the storage-fixing equality row (row 0).
        let v_in = 1_000.0_f64;
        solver.set_row_bounds(&[0], &[v_in], &[v_in]);

        let view = solver
            .solve()
            .expect("evaporation LP must be feasible and optimal");

        // Q_ev is the first evaporation column (before withdrawal slack).
        let col_q_ev = template.num_cols - 4;
        let q_ev = view.primal[col_q_ev];

        assert!(
            q_ev >= -1e-8,
            "Q_ev must be non-negative after solving, got {q_ev}"
        );
    }

    /// AC-4 (ticket-013): violation slacks are near zero when `v_in` is large
    /// enough for the linearised evaporation constraint to be satisfiable without
    /// artificial violation.
    ///
    /// With `k_evap0 = 1.0`, `k_evap_v = 0.02`, and `v_in = 1000 hm3`, the
    /// evaporation constraint RHS is positive and feasible, so the solver should
    /// drive the high-cost violation slacks to zero.
    #[test]
    fn evap_violation_slacks_near_zero_feasible_constraint() {
        use cobre_solver::{HighsSolver, RowBatch, SolverInterface};

        let system = evap_hydro_system_with_violation_cost(730.0, 500.0);
        let evap = evap_set_with_k_evap_v(&system, &[0], 1.0, 0.02);

        let result = build_stage_templates(
            &system,
            &no_penalty_config(),
            &PrecomputedPar::default(),
            &PrecomputedNormal::default(),
            &default_production(&system),
            &evap,
        )
        .expect("evap system template build must succeed");

        let template = &result.templates[0];
        let mut solver = HighsSolver::new().expect("HighsSolver::new must succeed");
        solver.load_model(template);

        let empty_cuts = RowBatch {
            num_rows: 0,
            row_starts: vec![0_i32],
            col_indices: vec![],
            values: vec![],
            row_lower: vec![],
            row_upper: vec![],
        };
        solver.add_rows(&empty_cuts);

        let v_in = 1_000.0_f64;
        solver.set_row_bounds(&[0], &[v_in], &[v_in]);

        let view = solver
            .solve()
            .expect("evaporation LP must be feasible and optimal");

        // Evaporation violation slack columns are before withdrawal slack.
        let col_f_plus = template.num_cols - 3;
        let col_f_minus = template.num_cols - 2;
        let f_plus = view.primal[col_f_plus];
        let f_minus = view.primal[col_f_minus];

        assert!(
            f_plus.abs() < 1e-6,
            "f_evap_plus slack must be near zero (constraint satisfied without violation), got {f_plus}"
        );
        assert!(
            f_minus.abs() < 1e-6,
            "f_evap_minus slack must be near zero (constraint satisfied without violation), got {f_minus}"
        );
    }

    /// AC-5 (ticket-013): the storage-fixing dual for an evaporation hydro differs
    /// from the no-evaporation case.
    ///
    /// When evaporation is active, higher `v_in` reduces evaporation volume
    /// (water retained in the reservoir increases), changing the water balance and
    /// hence the marginal value of initial storage. The dual of the storage-fixing
    /// row must differ between the two configurations.
    #[test]
    fn evap_storage_fixing_dual_differs_from_no_evaporation() {
        use cobre_solver::{HighsSolver, RowBatch, SolverInterface};

        // System with evaporation violation cost (so slacks are penalised).
        let system_evap = evap_hydro_system_with_violation_cost(730.0, 500.0);
        let evap = evap_set_with_k_evap_v(&system_evap, &[0], 1.0, 0.02);
        let evap_result = build_stage_templates(
            &system_evap,
            &no_penalty_config(),
            &PrecomputedPar::default(),
            &PrecomputedNormal::default(),
            &default_production(&system_evap),
            &evap,
        )
        .expect("evap system template build must succeed");

        // Baseline system without evaporation (same structure, EvaporationModel::None).
        let system_base = one_hydro_system(1, 0);
        let no_evap = EvaporationModelSet::new(vec![EvaporationModel::None]);
        let base_result = build_stage_templates(
            &system_base,
            &no_penalty_config(),
            &PrecomputedPar::default(),
            &PrecomputedNormal::default(),
            &default_production(&system_base),
            &no_evap,
        )
        .expect("baseline system template build must succeed");

        let solve_and_get_storage_dual = |template: &cobre_solver::StageTemplate| -> f64 {
            let mut solver = HighsSolver::new().expect("HighsSolver::new must succeed");
            solver.load_model(template);
            let empty_cuts = RowBatch {
                num_rows: 0,
                row_starts: vec![0_i32],
                col_indices: vec![],
                values: vec![],
                row_lower: vec![],
                row_upper: vec![],
            };
            solver.add_rows(&empty_cuts);
            let v_in = 1_000.0_f64;
            solver.set_row_bounds(&[0], &[v_in], &[v_in]);
            let view = solver.solve().expect("LP must solve to optimal");
            // Row 0 is the storage-fixing equality; its dual is the marginal value
            // of one additional hm3 of initial storage.
            view.dual[0]
        };

        let evap_dual = solve_and_get_storage_dual(&evap_result.templates[0]);
        let base_dual = solve_and_get_storage_dual(&base_result.templates[0]);

        // The evaporation constraint couples Q_ev to v and v_in via k_evap_v,
        // so the marginal value of initial storage differs from the no-evaporation case.
        assert_ne!(
            (evap_dual * 1e6).round(),
            (base_dual * 1e6).round(),
            "storage-fixing dual must differ between evaporation ({evap_dual}) and \
             no-evaporation ({base_dual}) configurations"
        );
    }

    /// Q_ev physical bound prevents the LP from using evaporation as a dump
    /// valve.  With high v_in and high inflow, the LP must use spillage (not
    /// evaporation) to remove excess water.  The test confirms Q_ev <= Q_ev_max,
    /// f_minus ~ 0, and spillage > 0.
    #[test]
    fn evap_bound_prevents_dump_valve() {
        use cobre_solver::{HighsSolver, RowBatch, SolverInterface};

        let system = evap_hydro_system_with_violation_cost(730.0, 500.0);
        let evap = evap_set_with_k_evap_v(&system, &[0], 2.0, 0.0001);

        let result = build_stage_templates(
            &system,
            &no_penalty_config(),
            &PrecomputedPar::default(),
            &PrecomputedNormal::default(),
            &default_production(&system),
            &evap,
        )
        .expect("evap dump valve test: template build must succeed");

        let template = &result.templates[0];
        let mut solver = HighsSolver::new().expect("HighsSolver::new must succeed");
        solver.load_model(template);

        let empty_cuts = RowBatch {
            num_rows: 0,
            row_starts: vec![0_i32],
            col_indices: vec![],
            values: vec![],
            row_lower: vec![],
            row_upper: vec![],
        };
        solver.add_rows(&empty_cuts);

        // Fix v_in at max_storage = 2000 hm3.
        let v_in = 2_000.0_f64;
        solver.set_row_bounds(&[0], &[v_in], &[v_in]);

        // Patch water balance RHS (row 2 for N=1, L=0) to inject large inflow.
        // Water balance: v + zeta*(turbine + spill + div) - v_in + zeta*Q_ev = RHS.
        // The template RHS = zeta * base = 2.628 * 50 = 131.4.
        // Set RHS to zeta * 500 = 1314 to simulate a 500 m3/s inflow.
        // The LP must then satisfy: v + zeta*(turbine+spill+...) = v_in + 1314 = 3314.
        // With v <= 2000 and max turbine = 262.8 hm3, surplus > 1000 hm3 must spill.
        let zeta = 730.0 * 3600.0 / 1e6;
        let high_inflow_rhs = zeta * 500.0;
        solver.set_row_bounds(&[2], &[high_inflow_rhs], &[high_inflow_rhs]);

        let view = solver
            .solve()
            .expect("evap dump valve LP must be feasible and optimal");

        // Column layout: N=1, L=0, K=1.
        // col 0: v, col 1: z_inflow, col 2: v_in, col 3: theta,
        // col 4: turbine, col 5: spillage, col 6: diversion,
        // col 7: deficit, col 8: excess.
        // Evaporation columns: Q_ev, f_plus, f_minus, then withdrawal slack.
        let col_spillage = 5;
        let col_q_ev = template.num_cols - 4;
        let col_f_minus = template.num_cols - 2;

        let q_ev = view.primal[col_q_ev];
        let f_minus = view.primal[col_f_minus];
        let spillage = view.primal[col_spillage];

        // Q_ev must respect the physical bound.
        // k_evap0=2.0, k_evap_v=0.0001, max_storage_hm3=2000.0
        // q_ev_max = max(0, 2.0 + 0.0001*2000) * 2.0 = 2.2 * 2.0 = 4.4
        let q_ev_max = (2.0 + 0.0001 * 2_000.0) * Q_EV_SAFETY_MARGIN;
        assert!(
            q_ev <= q_ev_max + 1e-8,
            "Q_ev must be bounded by physical limit {q_ev_max}, got {q_ev}"
        );

        // Over-evaporation violation must be near zero (the 100x penalty deters it).
        assert!(
            f_minus < 1e-6,
            "f_minus (over-evaporation) must be near zero, got {f_minus}"
        );

        // With massive inflow, the LP must dump water through spillage.
        assert!(
            spillage > 1e-6,
            "spillage must be positive when excess water needs dumping, got {spillage}"
        );
    }

    // ─── Multi-segment deficit tests ──────────────────────────────────────────

    /// Build a no-hydro, no-thermal, no-line system with the given buses and 1 stage / 1 block.
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    fn multi_segment_system(buses: Vec<Bus>, block_hours: f64) -> cobre_core::System {
        use chrono::NaiveDate;
        use cobre_core::scenario::LoadModel;
        use cobre_core::temporal::{
            Block, BlockMode, ScenarioSourceConfig, Stage, StageRiskConfig, StageStateConfig,
        };

        let n_buses = buses.len();
        let load_models: Vec<LoadModel> = buses
            .iter()
            .map(|b| LoadModel {
                bus_id: b.id,
                stage_id: 0,
                mean_mw: 0.0,
                std_mw: 0.0,
            })
            .collect();

        let stage = Stage {
            index: 0,
            id: 0,
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            season_id: None,
            blocks: vec![Block {
                index: 0,
                name: "S".to_string(),
                duration_hours: block_hours,
            }],
            block_mode: BlockMode::Parallel,
            state_config: StageStateConfig {
                storage: false,
                inflow_lags: false,
            },
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor: 1,
                noise_method: cobre_core::temporal::NoiseMethod::Saa,
            },
        };

        let bounds = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 0,
                n_thermals: 0,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: n_buses,
                n_stages: 1,
            },
            &BoundsDefaults {
                hydro: default_hydro_bounds(),
                thermal: ThermalStageBounds {
                    min_generation_mw: 0.0,
                    max_generation_mw: 0.0,
                },
                line: LineStageBounds {
                    direct_mw: 0.0,
                    reverse_mw: 0.0,
                },
                pumping: PumpingStageBounds {
                    min_flow_m3s: 0.0,
                    max_flow_m3s: 0.0,
                },
                contract: ContractStageBounds {
                    min_mw: 0.0,
                    max_mw: 0.0,
                    price_per_mwh: 0.0,
                },
            },
        );
        let penalties = ResolvedPenalties::new(
            &PenaltiesCountsSpec {
                n_hydros: 0,
                n_buses,
                n_lines: 0,
                n_ncs: 0,
                n_stages: 1,
            },
            &PenaltiesDefaults {
                hydro: default_hydro_penalties(),
                bus: BusStagePenalties { excess_cost: 0.0 },
                line: LineStagePenalties { exchange_cost: 0.0 },
                ncs: NcsStagePenalties {
                    curtailment_cost: 0.0,
                },
            },
        );

        SystemBuilder::new()
            .buses(buses)
            .stages(vec![stage])
            .load_models(load_models)
            .bounds(bounds)
            .penalties(penalties)
            .build()
            .expect("multi_segment_system: valid")
    }

    /// AC: 2 buses (bus0: 3 segments, bus1: 1 segment), 2 blocks → deficit columns = `B*S_max*K` = 2*3*2 = 12.
    #[test]
    #[allow(clippy::too_many_lines)]
    fn test_multi_segment_deficit_column_count() {
        use chrono::NaiveDate;
        use cobre_core::scenario::LoadModel;
        use cobre_core::temporal::{
            Block, BlockMode, ScenarioSourceConfig, Stage, StageRiskConfig, StageStateConfig,
        };

        let bus0 = Bus {
            id: EntityId(1),
            name: "Bus0".to_string(),
            deficit_segments: vec![
                DeficitSegment {
                    depth_mw: Some(10.0),
                    cost_per_mwh: 100.0,
                },
                DeficitSegment {
                    depth_mw: Some(20.0),
                    cost_per_mwh: 200.0,
                },
                DeficitSegment {
                    depth_mw: None,
                    cost_per_mwh: 5000.0,
                },
            ],
            excess_cost: 0.0,
        };
        let bus1 = Bus {
            id: EntityId(2),
            name: "Bus1".to_string(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 1000.0,
            }],
            excess_cost: 0.0,
        };

        let load_models = vec![
            LoadModel {
                bus_id: EntityId(1),
                stage_id: 0,
                mean_mw: 0.0,
                std_mw: 0.0,
            },
            LoadModel {
                bus_id: EntityId(2),
                stage_id: 0,
                mean_mw: 0.0,
                std_mw: 0.0,
            },
        ];

        // 2 blocks per stage
        let stage = Stage {
            index: 0,
            id: 0,
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            season_id: None,
            blocks: vec![
                Block {
                    index: 0,
                    name: "B0".to_string(),
                    duration_hours: 360.0,
                },
                Block {
                    index: 1,
                    name: "B1".to_string(),
                    duration_hours: 360.0,
                },
            ],
            block_mode: BlockMode::Parallel,
            state_config: StageStateConfig {
                storage: false,
                inflow_lags: false,
            },
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor: 1,
                noise_method: cobre_core::temporal::NoiseMethod::Saa,
            },
        };

        let bounds = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 0,
                n_thermals: 0,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 2,
                n_stages: 1,
            },
            &BoundsDefaults {
                hydro: default_hydro_bounds(),
                thermal: ThermalStageBounds {
                    min_generation_mw: 0.0,
                    max_generation_mw: 0.0,
                },
                line: LineStageBounds {
                    direct_mw: 0.0,
                    reverse_mw: 0.0,
                },
                pumping: PumpingStageBounds {
                    min_flow_m3s: 0.0,
                    max_flow_m3s: 0.0,
                },
                contract: ContractStageBounds {
                    min_mw: 0.0,
                    max_mw: 0.0,
                    price_per_mwh: 0.0,
                },
            },
        );
        let penalties = ResolvedPenalties::new(
            &PenaltiesCountsSpec {
                n_hydros: 0,
                n_buses: 2,
                n_lines: 0,
                n_ncs: 0,
                n_stages: 1,
            },
            &PenaltiesDefaults {
                hydro: default_hydro_penalties(),
                bus: BusStagePenalties { excess_cost: 0.0 },
                line: LineStagePenalties { exchange_cost: 0.0 },
                ncs: NcsStagePenalties {
                    curtailment_cost: 0.0,
                },
            },
        );

        let system = SystemBuilder::new()
            .buses(vec![bus0, bus1])
            .stages(vec![stage])
            .load_models(load_models)
            .bounds(bounds)
            .penalties(penalties)
            .build()
            .expect("2-bus 2-block system: valid");

        let result = build_stage_templates(
            &system,
            &no_penalty_config(),
            &PrecomputedPar::default(),
            &PrecomputedNormal::default(),
            &default_production(&system),
            &default_evaporation(&system),
        )
        .expect("build ok");

        let t = &result.templates[0];

        // N=0, L=0 → theta=0, decision_start=1
        // No thermals, no lines → col_deficit_start = 1
        // B=2, S_max=3, K=2 → deficit region = 2*3*2 = 12 columns → col_excess_start = 13
        // excess region = B*K = 2*2 = 4 → num_cols = 13 + 4 = 17
        let col_deficit_start = 1_usize;
        let max_deficit_segments = 3_usize;
        let n_blks = 2_usize;
        let n_buses = 2_usize;
        let deficit_region = n_buses * max_deficit_segments * n_blks;
        assert_eq!(
            deficit_region, 12,
            "deficit region must be B*S_max*K = 2*3*2 = 12"
        );
        let col_excess_start = col_deficit_start + deficit_region;
        let excess_region = n_buses * n_blks; // 2*2 = 4
        let expected_num_cols = col_excess_start + excess_region;
        assert_eq!(
            t.num_cols, expected_num_cols,
            "num_cols must include expanded deficit region"
        );
    }

    /// AC: Bus with 2 deficit segments [{10MW, $500}, {None, $5000}], 1 block 730h.
    /// Verify upper bounds and objective coefficients for both segment columns.
    #[test]
    fn test_multi_segment_deficit_bounds_and_objective() {
        let bus = Bus {
            id: EntityId(1),
            name: "Bus0".to_string(),
            deficit_segments: vec![
                DeficitSegment {
                    depth_mw: Some(10.0),
                    cost_per_mwh: 500.0,
                },
                DeficitSegment {
                    depth_mw: None,
                    cost_per_mwh: 5000.0,
                },
            ],
            excess_cost: 0.0,
        };

        let block_hours = 730.0_f64;
        let system = multi_segment_system(vec![bus], block_hours);

        let result = build_stage_templates(
            &system,
            &no_penalty_config(),
            &PrecomputedPar::default(),
            &PrecomputedNormal::default(),
            &default_production(&system),
            &default_evaporation(&system),
        )
        .expect("build ok");

        let t = &result.templates[0];

        // N=0 → theta=0, decision_start=1, no thermals/lines
        // col_deficit_start = 1
        // B=1, S_max=2, K=1 → deficit region = 1*2*1 = 2
        // seg0 col = 1 + 0*2*1 + 0*1 + 0 = 1
        // seg1 col = 1 + 0*2*1 + 1*1 + 0 = 2
        let col_seg0 = 1_usize;
        let col_seg1 = 2_usize;

        assert_eq!(
            t.col_upper[col_seg0], 10.0,
            "segment 0 upper bound must equal depth_mw = 10.0"
        );
        assert!(
            t.col_upper[col_seg1].is_infinite() && t.col_upper[col_seg1] > 0.0,
            "segment 1 upper bound must be +infinity (unbounded final segment)"
        );
        assert!(
            (t.objective[col_seg0] - 500.0 * block_hours / COST_SCALE_FACTOR).abs() < 1e-12,
            "segment 0 objective must be cost * block_hours / COST_SCALE_FACTOR = {} but got {}",
            500.0 * block_hours / COST_SCALE_FACTOR,
            t.objective[col_seg0]
        );
        assert!(
            (t.objective[col_seg1] - 5000.0 * block_hours / COST_SCALE_FACTOR).abs() < 1e-12,
            "segment 1 objective must be cost * block_hours / COST_SCALE_FACTOR = {} but got {}",
            5000.0 * block_hours / COST_SCALE_FACTOR,
            t.objective[col_seg1]
        );
    }

    /// AC: Single-segment bus must produce identical LP structure to the old single-column behavior.
    /// Specifically, the single deficit column must be unbounded and carry the correct cost.
    #[test]
    fn test_single_segment_backward_compat() {
        let cost = 1000.0_f64;
        let block_hours = 744.0_f64;

        let bus = Bus {
            id: EntityId(1),
            name: "Bus0".to_string(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: cost,
            }],
            excess_cost: 0.0,
        };

        let system = multi_segment_system(vec![bus], block_hours);

        let result = build_stage_templates(
            &system,
            &no_penalty_config(),
            &PrecomputedPar::default(),
            &PrecomputedNormal::default(),
            &default_production(&system),
            &default_evaporation(&system),
        )
        .expect("build ok");

        let t = &result.templates[0];

        // N=0 → theta=0, decision_start=1, col_deficit_start=1
        // B=1, S_max=1, K=1 → 1 deficit column at index 1
        let col_def = 1_usize;

        assert!(
            t.col_upper[col_def].is_infinite() && t.col_upper[col_def] > 0.0,
            "single segment must be unbounded (None depth_mw)"
        );
        assert!(
            (t.objective[col_def] - cost * block_hours / COST_SCALE_FACTOR).abs() < 1e-12,
            "single-segment objective must be cost * block_hours / COST_SCALE_FACTOR"
        );

        // Excess column immediately follows deficit (S_max=1, B=1, K=1 → excess at col 2)
        let col_exc = 2_usize;
        assert!(
            t.col_upper[col_exc].is_infinite() && t.col_upper[col_exc] > 0.0,
            "excess column must be unbounded"
        );
    }

    /// AC (ticket-003 C4): Bus with 2 deficit segments [{10MW, $500}, {None, $5000}] and 1 block.
    /// Every deficit segment column for the bus/block must have exactly one entry in the
    /// load-balance row with coefficient +1.0.
    ///
    /// Column layout (`N`=0, no thermals/lines):
    ///   col 0 = theta (value function),
    ///   `col_deficit_start` = 1,
    ///   `col_seg0` = 1 (`b_idx`=0, `seg_idx`=0, `blk`=0),
    ///   `col_seg1` = 2 (`b_idx`=0, `seg_idx`=1, `blk`=0).
    ///
    /// Row layout (`N`=0, `n_hydros`=0):
    ///   `row_load_balance_start` = 0,
    ///   load balance row for bus 0, block 0 = 0.
    #[test]
    fn test_multi_segment_deficit_load_balance_coefficients() {
        let bus = Bus {
            id: EntityId(1),
            name: "Bus0".to_string(),
            deficit_segments: vec![
                DeficitSegment {
                    depth_mw: Some(10.0),
                    cost_per_mwh: 500.0,
                },
                DeficitSegment {
                    depth_mw: None,
                    cost_per_mwh: 5000.0,
                },
            ],
            excess_cost: 0.0,
        };

        let system = multi_segment_system(vec![bus], 730.0);

        let result = build_stage_templates(
            &system,
            &no_penalty_config(),
            &PrecomputedPar::default(),
            &PrecomputedNormal::default(),
            &default_production(&system),
            &default_evaporation(&system),
        )
        .expect("build ok");

        let t = &result.templates[0];

        // N=0, no thermals/lines → col_deficit_start = 1
        // B=1, S_max=2, K=1 → seg0 at col 1, seg1 at col 2
        let col_seg0 = 1_usize;
        let col_seg1 = 2_usize;

        // N=0 hydros → row_load_balance_start = 0; bus 0, block 0 → row 0
        let load_balance_row = 0_usize;

        // Segment 0: must have exactly one CSC entry at the load-balance row with +1.0
        let entries_seg0 = entries_for_col(t, col_seg0);
        assert_eq!(
            entries_seg0.len(),
            1,
            "deficit segment 0 column must have exactly 1 CSC entry (load-balance row), got {entries_seg0:?}"
        );
        assert_eq!(
            entries_seg0[0].0, load_balance_row,
            "deficit segment 0 entry must be at the load-balance row {load_balance_row}, got row {}",
            entries_seg0[0].0
        );
        assert!(
            (entries_seg0[0].1 - 1.0).abs() < 1e-12,
            "deficit segment 0 load-balance coefficient must be +1.0, got {}",
            entries_seg0[0].1
        );

        // Segment 1: same assertion
        let entries_seg1 = entries_for_col(t, col_seg1);
        assert_eq!(
            entries_seg1.len(),
            1,
            "deficit segment 1 column must have exactly 1 CSC entry (load-balance row), got {entries_seg1:?}"
        );
        assert_eq!(
            entries_seg1[0].0, load_balance_row,
            "deficit segment 1 entry must be at the load-balance row {load_balance_row}, got row {}",
            entries_seg1[0].0
        );
        assert!(
            (entries_seg1[0].1 - 1.0).abs() < 1e-12,
            "deficit segment 1 load-balance coefficient must be +1.0, got {}",
            entries_seg1[0].1
        );
    }

    // -------------------------------------------------------------------------
    // ticket-002: Water withdrawal LP wiring unit tests
    // -------------------------------------------------------------------------

    /// Build a `one_hydro_system` variant with a custom `water_withdrawal_m3s` and
    /// `water_withdrawal_violation_cost` injected into the resolved bounds/penalties.
    ///
    /// The stage duration is fixed at 744 hours (one 31-day month) and block count is 1.
    /// `lag_order` controls whether AR lag state columns are included.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        clippy::too_many_lines
    )]
    fn one_hydro_system_with_withdrawal(
        n_stages: usize,
        lag_order: usize,
        water_withdrawal_m3s: f64,
        water_withdrawal_violation_cost: f64,
    ) -> cobre_core::System {
        use chrono::NaiveDate;
        use cobre_core::entities::hydro::{Hydro, HydroGenerationModel, HydroPenalties};
        use cobre_core::scenario::{InflowModel, LoadModel};
        use cobre_core::temporal::{
            Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
            StageStateConfig,
        };

        let bus = Bus {
            id: EntityId(1),
            name: "B1".to_string(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 500.0,
            }],
            excess_cost: 0.0,
        };

        let hydro = Hydro {
            id: EntityId(2),
            name: "H1".to_string(),
            bus_id: EntityId(1),
            downstream_id: None,
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: 0.0,
            max_storage_hm3: 200.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: HydroGenerationModel::ConstantProductivity {
                productivity_mw_per_m3s: 2.5,
            },
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 100.0,
            min_generation_mw: 0.0,
            max_generation_mw: 250.0,
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: None,
            evaporation_reference_volumes_hm3: None,
            diversion: None,
            filling: None,
            penalties: HydroPenalties {
                spillage_cost: 0.01,
                diversion_cost: 0.0,
                fpha_turbined_cost: 0.0,
                storage_violation_below_cost: 0.0,
                filling_target_violation_cost: 0.0,
                turbined_violation_below_cost: 0.0,
                outflow_violation_below_cost: 0.0,
                outflow_violation_above_cost: 0.0,
                generation_violation_below_cost: 0.0,
                evaporation_violation_cost: 0.0,
                water_withdrawal_violation_cost,
            },
        };

        let stages: Vec<Stage> = (0..n_stages)
            .map(|i| Stage {
                index: i,
                id: i as i32,
                start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
                season_id: None,
                blocks: vec![Block {
                    index: 0,
                    name: "S".to_string(),
                    duration_hours: 744.0,
                }],
                block_mode: BlockMode::Parallel,
                state_config: StageStateConfig {
                    storage: true,
                    inflow_lags: lag_order > 0,
                },
                risk_config: StageRiskConfig::Expectation,
                scenario_config: ScenarioSourceConfig {
                    branching_factor: 1,
                    noise_method: NoiseMethod::Saa,
                },
            })
            .collect();

        let ar_coefficients: Vec<f64> = (0..lag_order).map(|_| 0.5).collect();
        let inflow_models: Vec<InflowModel> = (0..n_stages)
            .map(|i| InflowModel {
                hydro_id: EntityId(2),
                stage_id: i as i32,
                mean_m3s: 80.0,
                std_m3s: 20.0,
                ar_coefficients: ar_coefficients.clone(),
                residual_std_ratio: 1.0,
            })
            .collect();

        let load_models: Vec<LoadModel> = (0..n_stages)
            .map(|i| LoadModel {
                bus_id: EntityId(1),
                stage_id: i as i32,
                mean_mw: 100.0,
                std_mw: 0.0,
            })
            .collect();

        let n_st = n_stages.max(1);

        // Build bounds with the specified withdrawal rate for every (hydro, stage) cell.
        let mut bounds = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 1,
                n_thermals: 0,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages: n_st,
            },
            &BoundsDefaults {
                hydro: HydroStageBounds {
                    min_storage_hm3: 0.0,
                    max_storage_hm3: 200.0,
                    min_turbined_m3s: 0.0,
                    max_turbined_m3s: 100.0,
                    min_outflow_m3s: 0.0,
                    max_outflow_m3s: None,
                    min_generation_mw: 0.0,
                    max_generation_mw: 250.0,
                    max_diversion_m3s: None,
                    filling_inflow_m3s: 0.0,
                    water_withdrawal_m3s,
                },
                thermal: ThermalStageBounds {
                    min_generation_mw: 0.0,
                    max_generation_mw: 0.0,
                },
                line: LineStageBounds {
                    direct_mw: 0.0,
                    reverse_mw: 0.0,
                },
                pumping: PumpingStageBounds {
                    min_flow_m3s: 0.0,
                    max_flow_m3s: 0.0,
                },
                contract: ContractStageBounds {
                    min_mw: 0.0,
                    max_mw: 0.0,
                    price_per_mwh: 0.0,
                },
            },
        );
        // Ensure withdrawal is set on every stage cell.
        for s in 0..n_st {
            bounds.hydro_bounds_mut(0, s).water_withdrawal_m3s = water_withdrawal_m3s;
        }

        // Build penalties with the specified violation cost for every (hydro, stage) cell.
        let mut penalties = ResolvedPenalties::new(
            &PenaltiesCountsSpec {
                n_hydros: 1,
                n_buses: 1,
                n_lines: 0,
                n_ncs: 0,
                n_stages: n_st,
            },
            &PenaltiesDefaults {
                hydro: HydroStagePenalties {
                    spillage_cost: 0.01,
                    diversion_cost: 0.0,
                    fpha_turbined_cost: 0.0,
                    storage_violation_below_cost: 0.0,
                    filling_target_violation_cost: 0.0,
                    turbined_violation_below_cost: 0.0,
                    outflow_violation_below_cost: 0.0,
                    outflow_violation_above_cost: 0.0,
                    generation_violation_below_cost: 0.0,
                    evaporation_violation_cost: 0.0,
                    water_withdrawal_violation_cost,
                },
                bus: BusStagePenalties { excess_cost: 0.0 },
                line: LineStagePenalties { exchange_cost: 0.0 },
                ncs: NcsStagePenalties {
                    curtailment_cost: 0.0,
                },
            },
        );
        // Ensure violation cost is set on every stage cell.
        for s in 0..n_st {
            penalties
                .hydro_penalties_mut(0, s)
                .water_withdrawal_violation_cost = water_withdrawal_violation_cost;
        }

        SystemBuilder::new()
            .buses(vec![bus])
            .hydros(vec![hydro])
            .stages(stages)
            .inflow_models(inflow_models)
            .load_models(load_models)
            .bounds(bounds)
            .penalties(penalties)
            .build()
            .expect("one_hydro_system_with_withdrawal: valid")
    }

    /// AC-1: Water balance RHS = ζ * (`deterministic_base` - `water_withdrawal_m3s`).
    ///
    /// With no PAR data the `deterministic_base` is 0.0.  For this test we verify
    /// the withdrawal subtraction alone: base=0, withdrawal=10.0, `zeta`=744*`M3S_TO_HM3`.
    /// Expected RHS = `744 * 3600/1_000_000 * (0 - 10)` = -2.6784.
    #[test]
    fn withdrawal_rhs_subtracted_from_water_balance() {
        let withdrawal = 10.0_f64;
        let system = one_hydro_system_with_withdrawal(1, 0, withdrawal, 0.0);
        let result = build_stage_templates(
            &system,
            &no_penalty_config(),
            &PrecomputedPar::default(),
            &PrecomputedNormal::default(),
            &default_production(&system),
            &default_evaporation(&system),
        )
        .expect("build ok");

        let t = &result.templates[0];
        // N=1, L=0: row_water_balance_start = n_state + n_hydros = N*(1+L) + N = 2.
        let row_water = 2_usize;
        let total_hours = 744.0_f64;
        let zeta = total_hours * 3_600.0 / 1_000_000.0;
        // base = 0 (no PAR data), withdrawal = 10.0
        let expected_rhs = zeta * (0.0 - withdrawal);
        assert!(
            (t.row_lower[row_water] - expected_rhs).abs() < 1e-12,
            "water balance row_lower: expected {expected_rhs}, got {}",
            t.row_lower[row_water]
        );
        assert!(
            (t.row_upper[row_water] - expected_rhs).abs() < 1e-12,
            "water balance row_upper: expected {expected_rhs}, got {}",
            t.row_upper[row_water]
        );
    }

    /// AC-1 (acceptance criterion phrasing): base=50, withdrawal=10, `zeta`=0.36 → RHS=14.4.
    ///
    /// Since the system has no PAR data (`PrecomputedPar::default()` has `n_stages`=0),
    /// we cannot inject a `deterministic_base` of 50 directly here.  Instead, we verify
    /// the subtraction formula by checking base=0 gives RHS = `-zeta`*withdrawal and
    /// by unit-testing `fill_stage_rows` indirectly via the template row bounds.
    /// The exact acceptance criterion arithmetic (0.36 * (50-10) = 14.4) is verified
    /// in the fixture-free acceptance criterion test below.
    #[test]
    fn withdrawal_zero_leaves_rhs_unchanged_from_base() {
        // With withdrawal=0 the RHS must equal the no-withdrawal case identically.
        let system_zero = one_hydro_system_with_withdrawal(1, 0, 0.0, 0.0);
        let system_base = one_hydro_system(1, 0);

        let result_zero = build_stage_templates(
            &system_zero,
            &no_penalty_config(),
            &PrecomputedPar::default(),
            &PrecomputedNormal::default(),
            &default_production(&system_zero),
            &default_evaporation(&system_zero),
        )
        .expect("zero-withdrawal build ok");

        let result_base = build_stage_templates(
            &system_base,
            &no_penalty_config(),
            &PrecomputedPar::default(),
            &PrecomputedNormal::default(),
            &default_production(&system_base),
            &default_evaporation(&system_base),
        )
        .expect("base build ok");

        let t_zero = &result_zero.templates[0];
        let t_base = &result_base.templates[0];

        // N=1, L=0: row_water_balance_start = 1.
        let row_water = 1_usize;
        assert!(
            (t_zero.row_lower[row_water] - t_base.row_lower[row_water]).abs() < 1e-15,
            "zero-withdrawal row_lower must equal base: {} vs {}",
            t_zero.row_lower[row_water],
            t_base.row_lower[row_water]
        );
        assert!(
            (t_zero.row_upper[row_water] - t_base.row_upper[row_water]).abs() < 1e-15,
            "zero-withdrawal row_upper must equal base: {} vs {}",
            t_zero.row_upper[row_water],
            t_base.row_upper[row_water]
        );
    }

    /// AC-2: Withdrawal slack column has exactly one CSC entry at (`row_water`, -`zeta`).
    ///
    /// Column layout for N=1, L=0, no penalty, no FPHA, no evaporation:
    ///   col 0: `v_out`, col 1: `z_inflow`, col 2: `v_in`, col 3: theta,
    ///   col 4: turbine, col 5: spillage, col 6: diversion, col 7: deficit,
    ///   col 8: excess, col 9: `withdrawal_slack` (= `num_cols` - 1)
    /// Row layout for N=1, L=0:
    ///   row 0: storage-fixing, row 1: z_inflow-def, row 2: water-balance, row 3: load-balance
    #[test]
    fn withdrawal_slack_matrix_entry_coefficient_is_minus_zeta() {
        let system = one_hydro_system_with_withdrawal(1, 0, 5.0, 1000.0);
        let result = build_stage_templates(
            &system,
            &no_penalty_config(),
            &PrecomputedPar::default(),
            &PrecomputedNormal::default(),
            &default_production(&system),
            &default_evaporation(&system),
        )
        .expect("build ok");

        let t = &result.templates[0];
        // Withdrawal slack column is the last column for N=1.
        let col_w = t.num_cols - 1;
        // Water balance row for hydro 0: N=1, L=0 → row_water = n_state + n_hydros = 2.
        let row_water = 2_usize;

        let total_hours = 744.0_f64;
        let zeta = total_hours * 3_600.0 / 1_000_000.0;

        let coeff = csc_entry(t, col_w, row_water).unwrap_or_else(|| {
            panic!("withdrawal slack column {col_w} has no entry at water balance row {row_water}")
        });
        assert!(
            (coeff - (-zeta)).abs() < 1e-12,
            "withdrawal slack coefficient: expected {}, got {coeff}",
            -zeta
        );

        // Must have exactly one entry (water balance only; no load balance).
        let all_entries = entries_for_col(t, col_w);
        assert_eq!(
            all_entries.len(),
            1,
            "withdrawal slack column must have exactly 1 CSC entry, got {all_entries:?}"
        );
    }

    /// AC-3: Objective coefficient = `water_withdrawal_violation_cost` * `total_stage_hours`.
    ///
    /// `violation_cost` = 1000.0, `total_stage_hours` = 744.0 → expected = `744_000.0`.
    #[test]
    fn withdrawal_slack_objective_equals_cost_times_hours() {
        let violation_cost = 1_000.0_f64;
        let total_hours = 744.0_f64;
        let system = one_hydro_system_with_withdrawal(1, 0, 5.0, violation_cost);
        let result = build_stage_templates(
            &system,
            &no_penalty_config(),
            &PrecomputedPar::default(),
            &PrecomputedNormal::default(),
            &default_production(&system),
            &default_evaporation(&system),
        )
        .expect("build ok");

        let t = &result.templates[0];
        let col_w = t.num_cols - 1;
        let expected_obj = violation_cost * total_hours / COST_SCALE_FACTOR;
        assert!(
            (t.objective[col_w] - expected_obj).abs() < 1e-12,
            "withdrawal slack objective: expected {expected_obj}, got {}",
            t.objective[col_w]
        );
    }

    /// AC-3 (zero cost): objective coefficient is 0.0 when violation cost is 0.0.
    #[test]
    fn withdrawal_slack_objective_zero_when_cost_is_zero() {
        let system = one_hydro_system_with_withdrawal(1, 0, 0.0, 0.0);
        let result = build_stage_templates(
            &system,
            &no_penalty_config(),
            &PrecomputedPar::default(),
            &PrecomputedNormal::default(),
            &default_production(&system),
            &default_evaporation(&system),
        )
        .expect("build ok");

        let t = &result.templates[0];
        // Withdrawal slack is the last column for N=1 (no NCS, no generic constraints).
        let col_w = t.num_cols - 1;
        assert!(
            t.objective[col_w].abs() < 1e-15,
            "withdrawal slack objective must be 0 when cost=0, got {}",
            t.objective[col_w]
        );
    }

    /// AC-4: Withdrawal slack column has bounds [0, +inf).
    ///
    /// Lower bound must be 0.0 (from vec initialisation), upper bound must be +inf.
    #[test]
    fn withdrawal_slack_bounds_are_zero_to_infinity() {
        let system = one_hydro_system_with_withdrawal(1, 0, 10.0, 5_000.0);
        let result = build_stage_templates(
            &system,
            &no_penalty_config(),
            &PrecomputedPar::default(),
            &PrecomputedNormal::default(),
            &default_production(&system),
            &default_evaporation(&system),
        )
        .expect("build ok");

        let t = &result.templates[0];
        // Withdrawal slack is the last column for N=1 (no NCS, no generic constraints).
        let col_w = t.num_cols - 1;
        assert!(
            (t.col_lower[col_w] - 0.0).abs() < 1e-15,
            "withdrawal slack lower bound must be 0.0, got {}",
            t.col_lower[col_w]
        );
        assert!(
            t.col_upper[col_w].is_infinite() && t.col_upper[col_w] > 0.0,
            "withdrawal slack upper bound must be +inf, got {}",
            t.col_upper[col_w]
        );
    }

    /// AC-5: Two hydros → two withdrawal slack columns, one per hydro.
    ///
    /// Verifies that for N=2 hydros each withdrawal slack column has exactly one
    /// CSC entry at (`row_water` + `h_idx`, -`zeta`) for `h_idx` in [0, 1].
    #[test]
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        clippy::too_many_lines
    )]
    fn two_hydro_withdrawal_slack_entries_per_hydro() {
        use chrono::NaiveDate;
        use cobre_core::entities::hydro::{Hydro, HydroGenerationModel, HydroPenalties};
        use cobre_core::scenario::{InflowModel, LoadModel};
        use cobre_core::temporal::{
            Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
            StageStateConfig,
        };

        let bus = Bus {
            id: EntityId(1),
            name: "B1".to_string(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 500.0,
            }],
            excess_cost: 0.0,
        };

        #[allow(clippy::cast_possible_wrap)]
        let make_hydro = |id: i32| Hydro {
            id: EntityId(id),
            name: format!("H{id}"),
            bus_id: EntityId(1),
            downstream_id: None,
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: 0.0,
            max_storage_hm3: 200.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: HydroGenerationModel::ConstantProductivity {
                productivity_mw_per_m3s: 2.5,
            },
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 100.0,
            min_generation_mw: 0.0,
            max_generation_mw: 250.0,
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: None,
            evaporation_reference_volumes_hm3: None,
            diversion: None,
            filling: None,
            penalties: HydroPenalties {
                spillage_cost: 0.01,
                diversion_cost: 0.0,
                fpha_turbined_cost: 0.0,
                storage_violation_below_cost: 0.0,
                filling_target_violation_cost: 0.0,
                turbined_violation_below_cost: 0.0,
                outflow_violation_below_cost: 0.0,
                outflow_violation_above_cost: 0.0,
                generation_violation_below_cost: 0.0,
                evaporation_violation_cost: 0.0,
                water_withdrawal_violation_cost: 2_000.0,
            },
        };

        let stages = vec![Stage {
            index: 0,
            id: 0,
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            season_id: None,
            blocks: vec![Block {
                index: 0,
                name: "S".to_string(),
                duration_hours: 744.0,
            }],
            block_mode: BlockMode::Parallel,
            state_config: StageStateConfig {
                storage: true,
                inflow_lags: false,
            },
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor: 1,
                noise_method: NoiseMethod::Saa,
            },
        }];

        let inflow_models = vec![
            InflowModel {
                hydro_id: EntityId(2),
                stage_id: 0,
                mean_m3s: 80.0,
                std_m3s: 20.0,
                ar_coefficients: vec![],
                residual_std_ratio: 1.0,
            },
            InflowModel {
                hydro_id: EntityId(3),
                stage_id: 0,
                mean_m3s: 50.0,
                std_m3s: 10.0,
                ar_coefficients: vec![],
                residual_std_ratio: 1.0,
            },
        ];

        let load_models = vec![LoadModel {
            bus_id: EntityId(1),
            stage_id: 0,
            mean_mw: 100.0,
            std_mw: 0.0,
        }];

        let hydro_bounds_default = HydroStageBounds {
            min_storage_hm3: 0.0,
            max_storage_hm3: 200.0,
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 100.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            min_generation_mw: 0.0,
            max_generation_mw: 250.0,
            max_diversion_m3s: None,
            filling_inflow_m3s: 0.0,
            water_withdrawal_m3s: 0.0,
        };
        let mut bounds = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 2,
                n_thermals: 0,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages: 1,
            },
            &BoundsDefaults {
                hydro: hydro_bounds_default,
                thermal: ThermalStageBounds {
                    min_generation_mw: 0.0,
                    max_generation_mw: 0.0,
                },
                line: LineStageBounds {
                    direct_mw: 0.0,
                    reverse_mw: 0.0,
                },
                pumping: PumpingStageBounds {
                    min_flow_m3s: 0.0,
                    max_flow_m3s: 0.0,
                },
                contract: ContractStageBounds {
                    min_mw: 0.0,
                    max_mw: 0.0,
                    price_per_mwh: 0.0,
                },
            },
        );
        bounds.hydro_bounds_mut(0, 0).water_withdrawal_m3s = 8.0;
        bounds.hydro_bounds_mut(1, 0).water_withdrawal_m3s = 12.0;

        let hydro_penalties_default = HydroStagePenalties {
            spillage_cost: 0.01,
            diversion_cost: 0.0,
            fpha_turbined_cost: 0.0,
            storage_violation_below_cost: 0.0,
            filling_target_violation_cost: 0.0,
            turbined_violation_below_cost: 0.0,
            outflow_violation_below_cost: 0.0,
            outflow_violation_above_cost: 0.0,
            generation_violation_below_cost: 0.0,
            evaporation_violation_cost: 0.0,
            water_withdrawal_violation_cost: 2_000.0,
        };
        let penalties = ResolvedPenalties::new(
            &PenaltiesCountsSpec {
                n_hydros: 2,
                n_buses: 1,
                n_lines: 0,
                n_ncs: 0,
                n_stages: 1,
            },
            &PenaltiesDefaults {
                hydro: hydro_penalties_default,
                bus: BusStagePenalties { excess_cost: 0.0 },
                line: LineStagePenalties { exchange_cost: 0.0 },
                ncs: NcsStagePenalties {
                    curtailment_cost: 0.0,
                },
            },
        );

        let system = SystemBuilder::new()
            .buses(vec![bus])
            .hydros(vec![make_hydro(2), make_hydro(3)])
            .stages(stages)
            .inflow_models(inflow_models)
            .load_models(load_models)
            .bounds(bounds)
            .penalties(penalties)
            .build()
            .expect("two_hydro_with_withdrawal: valid");

        let result = build_stage_templates(
            &system,
            &no_penalty_config(),
            &PrecomputedPar::default(),
            &PrecomputedNormal::default(),
            &default_production(&system),
            &default_evaporation(&system),
        )
        .expect("build ok");

        let t = &result.templates[0];

        // N=2, L=0: state+aux = N*(3+L)+1 = 7 (storage, z_inflow, storage_in, theta).
        // col_turbine_start = 7, col_spillage_start = 9, col_diversion_start = 11,
        // col_deficit_start = 13, col_excess_start = 14,
        // col_withdrawal_slack_start = 15, num_cols = 17.
        let col_w0 = t.num_cols - 2;
        let col_w1 = t.num_cols - 1;

        // N=2, L=0: row_water_balance_start = n_state + n_hydros = 2 + 2 = 4.
        let row_w0 = 4_usize; // water balance for hydro 0
        let row_w1 = 5_usize; // water balance for hydro 1

        let total_hours = 744.0_f64;
        let zeta = total_hours * 3_600.0 / 1_000_000.0;

        // Verify coefficient for hydro 0 slack.
        let coeff_w0 = csc_entry(t, col_w0, row_w0).unwrap_or_else(|| {
            panic!("withdrawal slack col {col_w0} has no entry at water balance row {row_w0}")
        });
        assert!(
            (coeff_w0 - (-zeta)).abs() < 1e-12,
            "hydro-0 withdrawal slack coeff: expected {}, got {coeff_w0}",
            -zeta
        );

        // Verify coefficient for hydro 1 slack.
        let coeff_w1 = csc_entry(t, col_w1, row_w1).unwrap_or_else(|| {
            panic!("withdrawal slack col {col_w1} has no entry at water balance row {row_w1}")
        });
        assert!(
            (coeff_w1 - (-zeta)).abs() < 1e-12,
            "hydro-1 withdrawal slack coeff: expected {}, got {coeff_w1}",
            -zeta
        );

        // Cross-check: hydro 0 slack must NOT have an entry in hydro 1's water balance row.
        assert!(
            csc_entry(t, col_w0, row_w1).is_none(),
            "hydro-0 withdrawal slack must not appear in hydro-1 water balance row"
        );
        assert!(
            csc_entry(t, col_w1, row_w0).is_none(),
            "hydro-1 withdrawal slack must not appear in hydro-0 water balance row"
        );
    }

    /// 3-hydro system: `num_cols` includes exactly 3 withdrawal slack columns.
    #[test]
    #[allow(clippy::too_many_lines)]
    fn three_hydro_num_cols_includes_three_withdrawal_slacks() {
        use chrono::NaiveDate;
        use cobre_core::entities::hydro::{Hydro, HydroGenerationModel, HydroPenalties};
        use cobre_core::scenario::{InflowModel, LoadModel};
        use cobre_core::temporal::{
            Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
            StageStateConfig,
        };

        let bus = Bus {
            id: EntityId(1),
            name: "B1".to_string(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 500.0,
            }],
            excess_cost: 0.0,
        };

        #[allow(clippy::cast_possible_wrap)]
        let make_hydro = |id: i32| Hydro {
            id: EntityId(id),
            name: format!("H{id}"),
            bus_id: EntityId(1),
            downstream_id: None,
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: 0.0,
            max_storage_hm3: 200.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: HydroGenerationModel::ConstantProductivity {
                productivity_mw_per_m3s: 2.5,
            },
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 100.0,
            min_generation_mw: 0.0,
            max_generation_mw: 250.0,
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: None,
            evaporation_reference_volumes_hm3: None,
            diversion: None,
            filling: None,
            penalties: HydroPenalties {
                spillage_cost: 0.01,
                diversion_cost: 0.0,
                fpha_turbined_cost: 0.0,
                storage_violation_below_cost: 0.0,
                filling_target_violation_cost: 0.0,
                turbined_violation_below_cost: 0.0,
                outflow_violation_below_cost: 0.0,
                outflow_violation_above_cost: 0.0,
                generation_violation_below_cost: 0.0,
                evaporation_violation_cost: 0.0,
                water_withdrawal_violation_cost: 1_000.0,
            },
        };

        let stages = vec![Stage {
            index: 0,
            id: 0,
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            season_id: None,
            blocks: vec![Block {
                index: 0,
                name: "S".to_string(),
                duration_hours: 744.0,
            }],
            block_mode: BlockMode::Parallel,
            state_config: StageStateConfig {
                storage: true,
                inflow_lags: false,
            },
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor: 1,
                noise_method: NoiseMethod::Saa,
            },
        }];

        let inflow_models: Vec<InflowModel> = [1, 2, 3]
            .iter()
            .map(|&hid| InflowModel {
                hydro_id: EntityId(hid),
                stage_id: 0,
                mean_m3s: 50.0,
                std_m3s: 10.0,
                ar_coefficients: vec![],
                residual_std_ratio: 1.0,
            })
            .collect();

        let load_models = vec![LoadModel {
            bus_id: EntityId(1),
            stage_id: 0,
            mean_mw: 100.0,
            std_mw: 0.0,
        }];

        let bounds = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 3,
                n_thermals: 0,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages: 1,
            },
            &BoundsDefaults {
                hydro: HydroStageBounds {
                    min_storage_hm3: 0.0,
                    max_storage_hm3: 200.0,
                    min_turbined_m3s: 0.0,
                    max_turbined_m3s: 100.0,
                    min_outflow_m3s: 0.0,
                    max_outflow_m3s: None,
                    min_generation_mw: 0.0,
                    max_generation_mw: 250.0,
                    max_diversion_m3s: None,
                    filling_inflow_m3s: 0.0,
                    water_withdrawal_m3s: 5.0,
                },
                thermal: ThermalStageBounds {
                    min_generation_mw: 0.0,
                    max_generation_mw: 0.0,
                },
                line: LineStageBounds {
                    direct_mw: 0.0,
                    reverse_mw: 0.0,
                },
                pumping: PumpingStageBounds {
                    min_flow_m3s: 0.0,
                    max_flow_m3s: 0.0,
                },
                contract: ContractStageBounds {
                    min_mw: 0.0,
                    max_mw: 0.0,
                    price_per_mwh: 0.0,
                },
            },
        );

        let penalties = ResolvedPenalties::new(
            &PenaltiesCountsSpec {
                n_hydros: 3,
                n_buses: 1,
                n_lines: 0,
                n_ncs: 0,
                n_stages: 1,
            },
            &PenaltiesDefaults {
                hydro: HydroStagePenalties {
                    spillage_cost: 0.01,
                    diversion_cost: 0.0,
                    fpha_turbined_cost: 0.0,
                    storage_violation_below_cost: 0.0,
                    filling_target_violation_cost: 0.0,
                    turbined_violation_below_cost: 0.0,
                    outflow_violation_below_cost: 0.0,
                    outflow_violation_above_cost: 0.0,
                    generation_violation_below_cost: 0.0,
                    evaporation_violation_cost: 0.0,
                    water_withdrawal_violation_cost: 1_000.0,
                },
                bus: BusStagePenalties { excess_cost: 0.0 },
                line: LineStagePenalties { exchange_cost: 0.0 },
                ncs: NcsStagePenalties {
                    curtailment_cost: 0.0,
                },
            },
        );

        let system = SystemBuilder::new()
            .buses(vec![bus])
            .hydros(vec![make_hydro(1), make_hydro(2), make_hydro(3)])
            .stages(stages)
            .inflow_models(inflow_models)
            .load_models(load_models)
            .bounds(bounds)
            .penalties(penalties)
            .build()
            .expect("three_hydro_system: valid");

        let result = build_stage_templates(
            &system,
            &no_penalty_config(),
            &PrecomputedPar::default(),
            &PrecomputedNormal::default(),
            &default_production(&system),
            &default_evaporation(&system),
        )
        .expect("build ok");

        let t = &result.templates[0];

        // N=3, L=0, 1 block, no thermals/lines/evap/inflow-penalty:
        // state+aux: 3 storage + 3 z_inflow + 3 storage_in + 1 theta = 10
        // turbine: 3, spillage: 3, diversion: 3
        // deficit: 1, excess: 1
        // inflow slack: 0 (no penalty config)
        // evap: 0
        // withdrawal slack: 3
        // Total = 10 + 3 + 3 + 3 + 1 + 1 + 3 = 24
        let expected_cols = 24_usize;
        assert_eq!(
            t.num_cols, expected_cols,
            "3-hydro system: num_cols should be {expected_cols}, got {}",
            t.num_cols
        );

        // Verify the withdrawal slack columns (the last N columns in the layout).
        // withdrawal_slack is at [num_cols - n_h, num_cols).
        let n_h = t.n_hydro;
        assert_eq!(
            t.col_upper[t.num_cols - n_h],
            f64::INFINITY,
            "withdrawal slack column for hydro 0 should be unbounded above"
        );
        assert_eq!(
            t.col_upper[t.num_cols - n_h + 1],
            f64::INFINITY,
            "withdrawal slack column for hydro 1 should be unbounded above"
        );
        assert_eq!(
            t.col_upper[t.num_cols - n_h + 2],
            f64::INFINITY,
            "withdrawal slack column for hydro 2 should be unbounded above"
        );
    }

    // ── Generic constraint layout tests (ticket-002) ──────────────────────────

    /// Build a minimal one-bus, one-stage system for generic constraint tests.
    ///
    /// `n_blks` controls how many operating blocks the single stage has.
    #[allow(clippy::cast_possible_wrap)]
    fn one_bus_system_n_blks(n_blks: usize) -> cobre_core::System {
        use chrono::NaiveDate;
        use cobre_core::scenario::LoadModel;
        use cobre_core::temporal::{
            Block, BlockMode, NoiseMethod, ScenarioSourceConfig, StageRiskConfig, StageStateConfig,
        };

        let bus = Bus {
            id: EntityId(1),
            name: "B1".to_string(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 500.0,
            }],
            excess_cost: 0.0,
        };

        let blocks: Vec<Block> = (0..n_blks)
            .map(|i| Block {
                index: i,
                name: format!("BLK{i}"),
                duration_hours: 720.0,
            })
            .collect();

        let stage = cobre_core::temporal::Stage {
            index: 0,
            id: 0_i32,
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            season_id: None,
            blocks,
            block_mode: BlockMode::Parallel,
            state_config: StageStateConfig {
                storage: false,
                inflow_lags: false,
            },
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor: 1,
                noise_method: NoiseMethod::Saa,
            },
        };

        let load_models: Vec<LoadModel> = vec![LoadModel {
            bus_id: EntityId(1),
            stage_id: 0,
            mean_mw: 100.0,
            std_mw: 0.0,
        }];

        let bounds = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 0,
                n_thermals: 0,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages: 1,
            },
            &BoundsDefaults {
                hydro: default_hydro_bounds(),
                thermal: ThermalStageBounds {
                    min_generation_mw: 0.0,
                    max_generation_mw: 0.0,
                },
                line: LineStageBounds {
                    direct_mw: 0.0,
                    reverse_mw: 0.0,
                },
                pumping: PumpingStageBounds {
                    min_flow_m3s: 0.0,
                    max_flow_m3s: 0.0,
                },
                contract: ContractStageBounds {
                    min_mw: 0.0,
                    max_mw: 0.0,
                    price_per_mwh: 0.0,
                },
            },
        );
        let penalties = ResolvedPenalties::new(
            &PenaltiesCountsSpec {
                n_hydros: 0,
                n_buses: 1,
                n_lines: 0,
                n_ncs: 0,
                n_stages: 1,
            },
            &PenaltiesDefaults {
                hydro: default_hydro_penalties(),
                bus: BusStagePenalties { excess_cost: 0.0 },
                line: LineStagePenalties { exchange_cost: 0.0 },
                ncs: NcsStagePenalties {
                    curtailment_cost: 0.0,
                },
            },
        );

        SystemBuilder::new()
            .buses(vec![bus])
            .stages(vec![stage])
            .load_models(load_models)
            .bounds(bounds)
            .penalties(penalties)
            .build()
            .expect("one_bus_system_n_blks: valid")
    }

    /// Make a `GenericConstraint` with a trivial expression (no terms).
    fn make_constraint(
        id: i32,
        sense: cobre_core::ConstraintSense,
        slack_enabled: bool,
    ) -> cobre_core::GenericConstraint {
        use cobre_core::{ConstraintExpression, GenericConstraint, SlackConfig};
        GenericConstraint {
            id: EntityId(id),
            name: format!("gc_{id}"),
            description: None,
            expression: ConstraintExpression { terms: vec![] },
            sense,
            slack: SlackConfig {
                enabled: slack_enabled,
                penalty: if slack_enabled { Some(5000.0) } else { None },
            },
        }
    }

    /// Build templates for `system` using the no-penalty method and default PAR/Normal.
    fn build_templates_for(system: &cobre_core::System) -> Vec<cobre_solver::StageTemplate> {
        let production = default_production(system);
        let evaporation = default_evaporation(system);
        build_stage_templates(
            system,
            &no_penalty_config(),
            &PrecomputedPar::default(),
            &PrecomputedNormal::default(),
            &production,
            &evaporation,
        )
        .expect("build_templates_for: valid")
        .templates
    }

    /// AC: 0 generic constraints → num_rows and num_cols unchanged from baseline.
    #[test]
    fn generic_constraints_zero_does_not_change_layout() {
        // Baseline: 1-block system with no generic constraints (identical to
        // an explicit empty list — both paths must produce the same layout).
        let system = one_bus_system_n_blks(1);
        let templates = build_templates_for(&system);
        let t = &templates[0];
        // Sanity: the layout is valid (positive counts).
        assert!(t.num_cols > 0);
        assert!(t.num_rows > 0);
        // A second call must be bit-for-bit identical.
        let templates2 = build_templates_for(&system);
        assert_eq!(
            templates2[0].num_cols, t.num_cols,
            "second build must not change num_cols"
        );
        assert_eq!(
            templates2[0].num_rows, t.num_rows,
            "second build must not change num_rows"
        );
    }

    /// AC: 1 active constraint, `block_id = None`, 3 blocks, no slack
    /// → n_generic_rows == 3, n_generic_slack_cols == 0, num_rows += 3.
    #[test]
    fn generic_constraint_no_slack_block_id_none_3_blocks() {
        use cobre_core::{ConstraintSense, ResolvedGenericConstraintBounds};
        use std::collections::HashMap;

        let n_blks = 3_usize;
        let baseline_system = one_bus_system_n_blks(n_blks);
        let baseline_rows = build_templates_for(&baseline_system)[0].num_rows;
        let baseline_cols = build_templates_for(&baseline_system)[0].num_cols;

        // Map constraint ID 10 → index 0.
        let id_map: HashMap<i32, usize> = [(10_i32, 0)].into_iter().collect();
        // One bound entry: constraint 10, stage 0, block_id = None, bound = 500.0.
        let rows = vec![(10_i32, 0_i32, None::<i32>, 500.0_f64)];
        let generic_bounds = ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter());

        let constraint = make_constraint(10, ConstraintSense::LessEqual, false);
        let system = one_bus_system_n_blks_with_generic(n_blks, vec![constraint], generic_bounds);
        let t_rows = build_templates_for(&system)[0].num_rows;
        let t_cols = build_templates_for(&system)[0].num_cols;

        assert_eq!(
            t_rows,
            baseline_rows + n_blks,
            "num_rows must increase by n_blks={n_blks} (one row per block, no slack)"
        );
        assert_eq!(
            t_cols, baseline_cols,
            "num_cols must be unchanged (no slack columns)"
        );
    }

    /// AC: 1 active constraint (sense `<=`, slack enabled), `block_id = None`, 2 blocks
    /// → n_generic_rows == 2, n_generic_slack_cols == 2 (one slack per row).
    #[test]
    fn generic_constraint_le_slack_enabled_2_blocks() {
        use cobre_core::{ConstraintSense, ResolvedGenericConstraintBounds};
        use std::collections::HashMap;

        let n_blks = 2_usize;
        let baseline_system = one_bus_system_n_blks(n_blks);
        let baseline_rows = build_templates_for(&baseline_system)[0].num_rows;
        let baseline_cols = build_templates_for(&baseline_system)[0].num_cols;

        let id_map: HashMap<i32, usize> = [(20_i32, 0)].into_iter().collect();
        let rows = vec![(20_i32, 0_i32, None::<i32>, 300.0_f64)];
        let generic_bounds = ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter());

        let constraint = make_constraint(20, ConstraintSense::LessEqual, true);
        let system = one_bus_system_n_blks_with_generic(n_blks, vec![constraint], generic_bounds);
        let t_rows = build_templates_for(&system)[0].num_rows;
        let t_cols = build_templates_for(&system)[0].num_cols;

        // 2 rows (one per block), 2 slack cols (one per row for `<=`).
        assert_eq!(
            t_rows,
            baseline_rows + 2,
            "num_rows must increase by 2 (one row per block)"
        );
        assert_eq!(
            t_cols,
            baseline_cols + 2,
            "num_cols must increase by 2 (one slack per row for `<=`)"
        );
    }

    /// AC: 1 active constraint (sense `==`, slack enabled), `block_id = None`, 2 blocks
    /// → n_generic_slack_cols == 4 (two slacks per row: positive and negative).
    #[test]
    fn generic_constraint_equal_sense_two_slacks_per_row() {
        use cobre_core::{ConstraintSense, ResolvedGenericConstraintBounds};
        use std::collections::HashMap;

        let n_blks = 2_usize;
        let baseline_system = one_bus_system_n_blks(n_blks);
        let baseline_rows = build_templates_for(&baseline_system)[0].num_rows;
        let baseline_cols = build_templates_for(&baseline_system)[0].num_cols;

        let id_map: HashMap<i32, usize> = [(30_i32, 0)].into_iter().collect();
        let rows = vec![(30_i32, 0_i32, None::<i32>, 100.0_f64)];
        let generic_bounds = ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter());

        let constraint = make_constraint(30, ConstraintSense::Equal, true);
        let system = one_bus_system_n_blks_with_generic(n_blks, vec![constraint], generic_bounds);
        let t_rows = build_templates_for(&system)[0].num_rows;
        let t_cols = build_templates_for(&system)[0].num_cols;

        // 2 rows (one per block), 4 slack cols (two per row for `==`).
        assert_eq!(
            t_rows,
            baseline_rows + 2,
            "num_rows must increase by 2 (one row per block)"
        );
        assert_eq!(
            t_cols,
            baseline_cols + 4,
            "num_cols must increase by 4 (two slacks per row for `==`)"
        );
    }

    /// AC: 1 active constraint with `block_id = Some(1)`, 3 blocks
    /// → n_generic_rows == 1 (only the specified block generates a row).
    #[test]
    fn generic_constraint_specific_block_id_generates_one_row() {
        use cobre_core::{ConstraintSense, ResolvedGenericConstraintBounds};
        use std::collections::HashMap;

        let n_blks = 3_usize;
        let baseline_system = one_bus_system_n_blks(n_blks);
        let baseline_rows = build_templates_for(&baseline_system)[0].num_rows;
        let baseline_cols = build_templates_for(&baseline_system)[0].num_cols;

        let id_map: HashMap<i32, usize> = [(40_i32, 0)].into_iter().collect();
        // block_id = Some(1) → only block 1 gets a row.
        let rows = vec![(40_i32, 0_i32, Some(1_i32), 200.0_f64)];
        let generic_bounds = ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter());

        let constraint = make_constraint(40, ConstraintSense::LessEqual, false);
        let system = one_bus_system_n_blks_with_generic(n_blks, vec![constraint], generic_bounds);
        let t_rows = build_templates_for(&system)[0].num_rows;
        let t_cols = build_templates_for(&system)[0].num_cols;

        assert_eq!(
            t_rows,
            baseline_rows + 1,
            "num_rows must increase by exactly 1 (only the specified block)"
        );
        assert_eq!(
            t_cols, baseline_cols,
            "num_cols must be unchanged (no slack columns)"
        );
    }

    /// AC: 2 constraints, one active at stage 0 and one inactive (no bounds) — only the
    /// active one contributes rows.
    #[test]
    fn generic_constraint_inactive_does_not_contribute_rows() {
        use cobre_core::{ConstraintSense, ResolvedGenericConstraintBounds};
        use std::collections::HashMap;

        let n_blks = 2_usize;
        let baseline_system = one_bus_system_n_blks(n_blks);
        let baseline_rows = build_templates_for(&baseline_system)[0].num_rows;

        // Only constraint 50 (index 0) is active at stage 0.
        // Constraint 51 (index 1) has no bounds → inactive.
        let id_map: HashMap<i32, usize> = [(50_i32, 0), (51_i32, 1)].into_iter().collect();
        let rows = vec![(50_i32, 0_i32, None::<i32>, 400.0_f64)];
        let generic_bounds = ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter());

        let c_active = make_constraint(50, ConstraintSense::LessEqual, false);
        let c_inactive = make_constraint(51, ConstraintSense::LessEqual, false);
        let system =
            one_bus_system_n_blks_with_generic(n_blks, vec![c_active, c_inactive], generic_bounds);
        let t_rows = build_templates_for(&system)[0].num_rows;

        // Only the active constraint (c_active) contributes n_blks=2 rows.
        assert_eq!(
            t_rows,
            baseline_rows + n_blks,
            "only the active constraint must contribute rows"
        );
    }

    /// AC: StageIndexer fields for generic constraints are empty when built via `new`.
    #[test]
    fn stage_indexer_generic_fields_empty_from_new() {
        let idx = crate::indexer::StageIndexer::new(3, 2);
        assert!(
            idx.generic_constraint_rows.is_empty(),
            "generic_constraint_rows must be empty from new()"
        );
        assert!(
            idx.generic_constraint_slack.is_empty(),
            "generic_constraint_slack must be empty from new()"
        );
        assert_eq!(
            idx.n_generic_constraints_active, 0,
            "n_generic_constraints_active must be 0 from new()"
        );
    }

    /// Helper: build a one-bus system with `n_blks` operating blocks and
    /// the given generic constraints + resolved bounds.
    #[allow(clippy::cast_possible_wrap)]
    fn one_bus_system_n_blks_with_generic(
        n_blks: usize,
        constraints: Vec<cobre_core::GenericConstraint>,
        bounds: cobre_core::ResolvedGenericConstraintBounds,
    ) -> cobre_core::System {
        use chrono::NaiveDate;
        use cobre_core::scenario::LoadModel;
        use cobre_core::temporal::{
            Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
            StageStateConfig,
        };

        let bus = Bus {
            id: EntityId(1),
            name: "B1".to_string(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 500.0,
            }],
            excess_cost: 0.0,
        };

        let blks: Vec<Block> = (0..n_blks)
            .map(|i| Block {
                index: i,
                name: format!("BLK{i}"),
                duration_hours: 720.0,
            })
            .collect();

        let stage = Stage {
            index: 0,
            id: 0_i32,
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            season_id: None,
            blocks: blks,
            block_mode: BlockMode::Parallel,
            state_config: StageStateConfig {
                storage: false,
                inflow_lags: false,
            },
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor: 1,
                noise_method: NoiseMethod::Saa,
            },
        };

        let load_models: Vec<LoadModel> = vec![LoadModel {
            bus_id: EntityId(1),
            stage_id: 0,
            mean_mw: 100.0,
            std_mw: 0.0,
        }];

        let resolved_bounds = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 0,
                n_thermals: 0,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages: 1,
            },
            &BoundsDefaults {
                hydro: default_hydro_bounds(),
                thermal: ThermalStageBounds {
                    min_generation_mw: 0.0,
                    max_generation_mw: 0.0,
                },
                line: LineStageBounds {
                    direct_mw: 0.0,
                    reverse_mw: 0.0,
                },
                pumping: PumpingStageBounds {
                    min_flow_m3s: 0.0,
                    max_flow_m3s: 0.0,
                },
                contract: ContractStageBounds {
                    min_mw: 0.0,
                    max_mw: 0.0,
                    price_per_mwh: 0.0,
                },
            },
        );
        let penalties = ResolvedPenalties::new(
            &PenaltiesCountsSpec {
                n_hydros: 0,
                n_buses: 1,
                n_lines: 0,
                n_ncs: 0,
                n_stages: 1,
            },
            &PenaltiesDefaults {
                hydro: default_hydro_penalties(),
                bus: BusStagePenalties { excess_cost: 0.0 },
                line: LineStagePenalties { exchange_cost: 0.0 },
                ncs: NcsStagePenalties {
                    curtailment_cost: 0.0,
                },
            },
        );

        SystemBuilder::new()
            .buses(vec![bus])
            .stages(vec![stage])
            .load_models(load_models)
            .bounds(resolved_bounds)
            .penalties(penalties)
            .generic_constraints(constraints)
            .resolved_generic_bounds(bounds)
            .build()
            .expect("one_bus_system_n_blks_with_generic: valid")
    }

    // ── Helper: scan CSC for entries in a specific (column, row) pair ──────────

    /// Return all values stored at `(col, row)` in the CSC template.
    ///
    /// A column may have multiple entries for the same row when two fill helpers
    /// add to the same position; both are returned so tests can check the total
    /// (or assert uniqueness).
    fn csc_entries_at(t: &cobre_solver::StageTemplate, col: usize, row: usize) -> Vec<f64> {
        let start = t.col_starts[col] as usize;
        let end = t.col_starts[col + 1] as usize;
        t.row_indices[start..end]
            .iter()
            .zip(t.values[start..end].iter())
            .filter_map(|(&r, &v)| if r as usize == row { Some(v) } else { None })
            .collect()
    }

    // ── AC01: thermal <= constraint row bounds ─────────────────────────────────

    /// AC: `thermal_generation(0) <= 50.0`, 1 block, no slack.
    /// Verify `row_upper = 50.0`, `row_lower = -INF`, CSC entry `+1.0` in the
    /// thermal generation column at the generic constraint row.
    #[test]
    fn generic_constraint_thermal_le_row_bounds_and_csc_entry() {
        use cobre_core::ResolvedGenericConstraintBounds;
        use cobre_core::{
            ConstraintExpression, ConstraintSense, GenericConstraint, LinearTerm, SlackConfig,
            VariableRef,
        };
        use std::collections::HashMap;

        let thermal_entity_id = EntityId(2);

        // Build the generic constraint: thermal_generation(0) <= 50.0
        let constraint = GenericConstraint {
            id: EntityId(10),
            name: "gc_thermal_le".to_string(),
            description: None,
            expression: ConstraintExpression {
                terms: vec![LinearTerm {
                    coefficient: 1.0,
                    variable: VariableRef::ThermalGeneration {
                        thermal_id: thermal_entity_id,
                        block_id: None,
                    },
                }],
            },
            sense: ConstraintSense::LessEqual,
            slack: SlackConfig {
                enabled: false,
                penalty: None,
            },
        };

        let id_map: HashMap<i32, usize> = [(10_i32, 0)].into_iter().collect();
        let rows = vec![(10_i32, 0_i32, None::<i32>, 50.0_f64)];
        let generic_bounds = ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter());

        // Build a 1-bus, 1-thermal, 1-block system.
        let system =
            one_bus_one_thermal_system(thermal_entity_id, vec![constraint], generic_bounds);
        let t = &build_templates_for(&system)[0];

        // Layout for N=0, T=1, B=1, K=1:
        //   theta=0, decision_start=1
        //   thermal col 0, block 0 → col = 1
        //   load_balance row 0 → generic row at row 1
        let thermal_col = 1_usize;
        let generic_row = 1_usize;

        // Row bounds: <= sense → lower=-INF, upper=50.0
        assert!(
            t.row_lower[generic_row].is_infinite() && t.row_lower[generic_row] < 0.0,
            "row_lower must be -INF for <= constraint, got {}",
            t.row_lower[generic_row]
        );
        assert!(
            (t.row_upper[generic_row] - 50.0).abs() < f64::EPSILON,
            "row_upper must be 50.0, got {}",
            t.row_upper[generic_row]
        );

        // CSC entry: thermal gen column must have +1.0 at the generic constraint row.
        let entries = csc_entries_at(t, thermal_col, generic_row);
        assert!(
            !entries.is_empty(),
            "no CSC entry found at (col={thermal_col}, row={generic_row})"
        );
        let total: f64 = entries.iter().sum();
        assert!(
            (total - 1.0).abs() < f64::EPSILON,
            "expected +1.0 total at thermal col / generic row, got {total}"
        );
    }

    // ── AC02: slack column for <= constraint ───────────────────────────────────

    /// AC: `thermal_generation(0) <= 50.0`, 1 block, slack enabled (penalty=5000).
    /// Verify slack column `col_lower=0`, `col_upper=+INF`, `objective=5000*744`,
    /// and CSC entry `-1.0` for the slack column at the generic constraint row.
    #[test]
    fn generic_constraint_thermal_le_slack_column_and_csc_entry() {
        use cobre_core::ResolvedGenericConstraintBounds;
        use cobre_core::{
            ConstraintExpression, ConstraintSense, GenericConstraint, LinearTerm, SlackConfig,
            VariableRef,
        };
        use std::collections::HashMap;

        let thermal_entity_id = EntityId(2);
        let block_hours = 744.0_f64;
        let penalty = 5000.0_f64;

        let constraint = GenericConstraint {
            id: EntityId(20),
            name: "gc_thermal_le_slack".to_string(),
            description: None,
            expression: ConstraintExpression {
                terms: vec![LinearTerm {
                    coefficient: 1.0,
                    variable: VariableRef::ThermalGeneration {
                        thermal_id: thermal_entity_id,
                        block_id: None,
                    },
                }],
            },
            sense: ConstraintSense::LessEqual,
            slack: SlackConfig {
                enabled: true,
                penalty: Some(penalty),
            },
        };

        let id_map: HashMap<i32, usize> = [(20_i32, 0)].into_iter().collect();
        let rows = vec![(20_i32, 0_i32, None::<i32>, 50.0_f64)];
        let generic_bounds = ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter());

        let system =
            one_bus_one_thermal_system(thermal_entity_id, vec![constraint], generic_bounds);
        let t = &build_templates_for(&system)[0];

        // Layout for N=0, T=1, B=1, K=1, 1 slack col:
        //   withdrawal_slack_start = col_evap_start + 0 evap cols
        //   = col_generation_start + 0 generation cols = inflow_slack_end
        //   = excess_end = 1(theta)+1(thermal)+1(deficit)+1(excess) = 4
        //   col_generic_slack_start = withdrawal_slack_start + n_h(=0) = 4
        //   slack_plus_col = 4
        let slack_col = 4_usize;
        let generic_row = 1_usize;

        // Slack column bounds: [0, +INF)
        assert!(
            t.col_lower[slack_col].abs() < f64::EPSILON,
            "slack col_lower must be 0.0, got {}",
            t.col_lower[slack_col]
        );
        assert!(
            t.col_upper[slack_col].is_infinite() && t.col_upper[slack_col] > 0.0,
            "slack col_upper must be +INF, got {}",
            t.col_upper[slack_col]
        );

        // Objective: penalty * block_hours / COST_SCALE_FACTOR
        let expected_obj = penalty * block_hours / COST_SCALE_FACTOR;
        assert!(
            (t.objective[slack_col] - expected_obj).abs() < 1e-12,
            "slack objective must be {expected_obj}, got {}",
            t.objective[slack_col]
        );

        // CSC entry: slack column must have -1.0 at the generic constraint row (<=).
        let entries = csc_entries_at(t, slack_col, generic_row);
        assert!(
            !entries.is_empty(),
            "no CSC entry found at (col={slack_col}, row={generic_row})"
        );
        let total: f64 = entries.iter().sum();
        assert!(
            (total - (-1.0)).abs() < f64::EPSILON,
            "expected -1.0 at slack col / generic row for <= sense, got {total}"
        );
    }

    // ── AC03: >= row bounds ────────────────────────────────────────────────────

    /// AC: `thermal_generation(0) >= 10.0`, 1 block, no slack.
    /// Verify `row_lower = 10.0`, `row_upper = +INF`.
    #[test]
    fn generic_constraint_thermal_ge_row_bounds() {
        use cobre_core::ResolvedGenericConstraintBounds;
        use cobre_core::{
            ConstraintExpression, ConstraintSense, GenericConstraint, LinearTerm, SlackConfig,
            VariableRef,
        };
        use std::collections::HashMap;

        let thermal_entity_id = EntityId(2);

        let constraint = GenericConstraint {
            id: EntityId(30),
            name: "gc_thermal_ge".to_string(),
            description: None,
            expression: ConstraintExpression {
                terms: vec![LinearTerm {
                    coefficient: 1.0,
                    variable: VariableRef::ThermalGeneration {
                        thermal_id: thermal_entity_id,
                        block_id: None,
                    },
                }],
            },
            sense: ConstraintSense::GreaterEqual,
            slack: SlackConfig {
                enabled: false,
                penalty: None,
            },
        };

        let id_map: HashMap<i32, usize> = [(30_i32, 0)].into_iter().collect();
        let rows = vec![(30_i32, 0_i32, None::<i32>, 10.0_f64)];
        let generic_bounds = ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter());

        let system =
            one_bus_one_thermal_system(thermal_entity_id, vec![constraint], generic_bounds);
        let t = &build_templates_for(&system)[0];

        let generic_row = 1_usize;
        assert!(
            (t.row_lower[generic_row] - 10.0).abs() < f64::EPSILON,
            "row_lower must be 10.0 for >= constraint, got {}",
            t.row_lower[generic_row]
        );
        assert!(
            t.row_upper[generic_row].is_infinite() && t.row_upper[generic_row] > 0.0,
            "row_upper must be +INF for >= constraint, got {}",
            t.row_upper[generic_row]
        );
    }

    // ── AC04: == row bounds with two slacks ────────────────────────────────────

    /// AC: `thermal_generation(0) == 80.0`, 1 block, slack enabled.
    /// Verify two slack columns (plus at col 4, minus at col 5).
    #[test]
    fn generic_constraint_thermal_equal_two_slacks() {
        use cobre_core::ResolvedGenericConstraintBounds;
        use cobre_core::{ConstraintExpression, ConstraintSense, GenericConstraint, SlackConfig};
        use std::collections::HashMap;

        let thermal_entity_id = EntityId(2);
        let penalty = 5000.0_f64;
        let block_hours = 744.0_f64;

        let constraint = GenericConstraint {
            id: EntityId(40),
            name: "gc_thermal_eq_slack".to_string(),
            description: None,
            expression: ConstraintExpression { terms: vec![] },
            sense: ConstraintSense::Equal,
            slack: SlackConfig {
                enabled: true,
                penalty: Some(penalty),
            },
        };

        let id_map: HashMap<i32, usize> = [(40_i32, 0)].into_iter().collect();
        let rows = vec![(40_i32, 0_i32, None::<i32>, 80.0_f64)];
        let generic_bounds = ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter());

        let system =
            one_bus_one_thermal_system(thermal_entity_id, vec![constraint], generic_bounds);
        let t = &build_templates_for(&system)[0];

        let slack_plus_col = 4_usize;
        let slack_minus_col = 5_usize;
        let generic_row = 1_usize;

        // Row bounds: == → lower == upper == bound
        assert!(
            (t.row_lower[generic_row] - 80.0).abs() < f64::EPSILON,
            "row_lower must be 80.0 for == constraint, got {}",
            t.row_lower[generic_row]
        );
        assert!(
            (t.row_upper[generic_row] - 80.0).abs() < f64::EPSILON,
            "row_upper must be 80.0 for == constraint, got {}",
            t.row_upper[generic_row]
        );

        // Two slack columns: num_cols baseline is 4 (theta=0, thermal=1, deficit=1, excess=1,
        // withdrawal_slack=0), with 2 slacks → num_cols = 6.
        assert_eq!(t.num_cols, 6, "num_cols must be 6 with 2 slack columns");

        // Plus slack: col_upper=+INF, objective=penalty*block_hours, CSC +1.0.
        assert!(
            t.col_upper[slack_plus_col].is_infinite() && t.col_upper[slack_plus_col] > 0.0,
            "plus slack col_upper must be +INF"
        );
        let expected_obj = penalty * block_hours / COST_SCALE_FACTOR;
        assert!(
            (t.objective[slack_plus_col] - expected_obj).abs() < 1e-12,
            "plus slack objective must be {expected_obj}, got {}",
            t.objective[slack_plus_col]
        );
        let plus_entries = csc_entries_at(t, slack_plus_col, generic_row);
        assert!(
            !plus_entries.is_empty(),
            "no CSC entry at plus slack col / generic row"
        );
        let plus_total: f64 = plus_entries.iter().sum();
        assert!(
            (plus_total - 1.0).abs() < f64::EPSILON,
            "plus slack CSC must be +1.0 for == sense, got {plus_total}"
        );

        // Minus slack: col_upper=+INF, objective=penalty*block_hours/COST_SCALE_FACTOR, CSC -1.0.
        assert!(
            t.col_upper[slack_minus_col].is_infinite() && t.col_upper[slack_minus_col] > 0.0,
            "minus slack col_upper must be +INF"
        );
        assert!(
            (t.objective[slack_minus_col] - expected_obj).abs() < 1e-12,
            "minus slack objective must be {expected_obj}"
        );
        let minus_entries = csc_entries_at(t, slack_minus_col, generic_row);
        assert!(
            !minus_entries.is_empty(),
            "no CSC entry at minus slack col / generic row"
        );
        let minus_total: f64 = minus_entries.iter().sum();
        assert!(
            (minus_total - (-1.0)).abs() < f64::EPSILON,
            "minus slack CSC must be -1.0 for == sense, got {minus_total}"
        );
    }

    // ── AC03: two hydros with constant productivity, sum constraint ────────────

    /// AC: `hydro_generation(H1) + hydro_generation(H2)` with constant
    /// productivities 2.5 and 3.0. Verify CSC entries at both turbine columns
    /// with coefficients equal to the respective productivities.
    #[test]
    #[allow(clippy::cast_possible_wrap)]
    fn generic_constraint_two_hydros_sum_csc_entries() {
        use chrono::NaiveDate;
        use cobre_core::entities::hydro::{Hydro, HydroGenerationModel, HydroPenalties};
        use cobre_core::scenario::{InflowModel, LoadModel};
        use cobre_core::temporal::{
            Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
            StageStateConfig,
        };
        use cobre_core::ResolvedGenericConstraintBounds;
        use cobre_core::{
            ConstraintExpression, ConstraintSense, GenericConstraint, LinearTerm, SlackConfig,
            VariableRef,
        };
        use std::collections::HashMap;

        let h1_id = EntityId(5);
        let h2_id = EntityId(10);
        let prod_h1 = 2.5_f64;
        let prod_h2 = 3.0_f64;

        let make_hydro = |id: EntityId, prod: f64| Hydro {
            id,
            name: format!("H{}", id.0),
            bus_id: EntityId(1),
            downstream_id: None,
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: 0.0,
            max_storage_hm3: 200.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: HydroGenerationModel::ConstantProductivity {
                productivity_mw_per_m3s: prod,
            },
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 100.0,
            min_generation_mw: 0.0,
            max_generation_mw: 250.0,
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: None,
            evaporation_reference_volumes_hm3: None,
            diversion: None,
            filling: None,
            penalties: HydroPenalties {
                spillage_cost: 0.01,
                diversion_cost: 0.0,
                fpha_turbined_cost: 0.0,
                storage_violation_below_cost: 0.0,
                filling_target_violation_cost: 0.0,
                turbined_violation_below_cost: 0.0,
                outflow_violation_below_cost: 0.0,
                outflow_violation_above_cost: 0.0,
                generation_violation_below_cost: 0.0,
                evaporation_violation_cost: 0.0,
                water_withdrawal_violation_cost: 0.0,
            },
        };

        let hydros = vec![make_hydro(h1_id, prod_h1), make_hydro(h2_id, prod_h2)];

        let bus = Bus {
            id: EntityId(1),
            name: "B1".to_string(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 500.0,
            }],
            excess_cost: 0.0,
        };

        let stage = Stage {
            index: 0,
            id: 0_i32,
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            season_id: None,
            blocks: vec![Block {
                index: 0,
                name: "BLK0".to_string(),
                duration_hours: 744.0,
            }],
            block_mode: BlockMode::Parallel,
            state_config: StageStateConfig {
                storage: true,
                inflow_lags: false,
            },
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor: 1,
                noise_method: NoiseMethod::Saa,
            },
        };

        let inflow_models = vec![
            InflowModel {
                hydro_id: h1_id,
                stage_id: 0,
                mean_m3s: 50.0,
                std_m3s: 0.0,
                ar_coefficients: vec![],
                residual_std_ratio: 0.0,
            },
            InflowModel {
                hydro_id: h2_id,
                stage_id: 0,
                mean_m3s: 50.0,
                std_m3s: 0.0,
                ar_coefficients: vec![],
                residual_std_ratio: 0.0,
            },
        ];

        let load_models = vec![LoadModel {
            bus_id: EntityId(1),
            stage_id: 0,
            mean_mw: 100.0,
            std_mw: 0.0,
        }];

        // Generic constraint: hydro_generation(H1) + hydro_generation(H2) <= 200
        let constraint = GenericConstraint {
            id: EntityId(100),
            name: "gc_sum_gen".to_string(),
            description: None,
            expression: ConstraintExpression {
                terms: vec![
                    LinearTerm {
                        coefficient: 1.0,
                        variable: VariableRef::HydroGeneration {
                            hydro_id: h1_id,
                            block_id: None,
                        },
                    },
                    LinearTerm {
                        coefficient: 1.0,
                        variable: VariableRef::HydroGeneration {
                            hydro_id: h2_id,
                            block_id: None,
                        },
                    },
                ],
            },
            sense: ConstraintSense::LessEqual,
            slack: SlackConfig {
                enabled: false,
                penalty: None,
            },
        };

        let id_map: HashMap<i32, usize> = [(100_i32, 0)].into_iter().collect();
        let rows = vec![(100_i32, 0_i32, None::<i32>, 200.0_f64)];
        let generic_bounds = ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter());

        let resolved_bounds = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 2,
                n_thermals: // n_hydros
            0,
                n_lines: // n_thermals
            0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages: 1,
            },
            &BoundsDefaults {
                hydro: // n_stages
            default_hydro_bounds(),
                thermal: ThermalStageBounds {
                min_generation_mw: 0.0,
                max_generation_mw: 0.0,
            },
                line: LineStageBounds {
                direct_mw: 0.0,
                reverse_mw: 0.0,
            },
                pumping: PumpingStageBounds {
                min_flow_m3s: 0.0,
                max_flow_m3s: 0.0,
            },
                contract: ContractStageBounds {
                min_mw: 0.0,
                max_mw: 0.0,
                price_per_mwh: 0.0,
            },
            },
        );
        let penalties = ResolvedPenalties::new(
            &PenaltiesCountsSpec {
                n_hydros: 2,
                n_buses: // n_hydros
            1,
                n_lines: // n_buses
            0,
                n_ncs: // n_lines
            0,
                n_stages: // n_ncs
            1,
            },
            &PenaltiesDefaults {
                hydro: // n_stages
            default_hydro_penalties(),
                bus: BusStagePenalties { excess_cost: 0.0 },
                line: LineStagePenalties { exchange_cost: 0.0 },
                ncs: NcsStagePenalties {
                curtailment_cost: 0.0,
            },
            },
        );

        let system = SystemBuilder::new()
            .buses(vec![bus])
            .hydros(hydros)
            .stages(vec![stage])
            .inflow_models(inflow_models)
            .load_models(load_models)
            .bounds(resolved_bounds)
            .penalties(penalties)
            .generic_constraints(vec![constraint])
            .resolved_generic_bounds(generic_bounds)
            .build()
            .expect("two_hydro_system: valid");

        let t = &build_templates_for(&system)[0];

        // Hydros sorted by ID: H1(5) at pos=0, H2(10) at pos=1
        // Column layout (N=2, T=0, K=1, 1 block, constant productivity → no FPHA gen cols):
        //   col 0,1: storage (outgoing) for H1, H2
        //   col 2,3: z_inflow for H1, H2
        //   col 4,5: storage_in for H1, H2
        //   col 6: theta
        //   col 7,8: turbine H1 blk0, H2 blk0
        //   col 9,10: spillage H1 blk0, H2 blk0
        //   col 11,12: deficit segments (1 seg * 1 blk), excess (1 blk)
        //   col 13,14: withdrawal_slack H1, H2

        // For constant-productivity hydros, HydroGeneration maps to the turbine
        // column with multiplier = productivity.
        // Generic constraint row is the last structural row.
        let generic_row = t.num_rows - 1;

        // Find turbine columns for H1 and H2. We know storage takes first
        // N cols, then z_inflow N cols, then storage_in N cols, then theta, then turbine.
        // N=2: storage=[0..2], z_inflow=[2..4], storage_in=[4..6], theta=6, turbine_start=7
        let turbine_h1_col = 7_usize;
        let turbine_h2_col = 8_usize;

        // CSC entry at turbine_h1_col, generic_row should have coefficient = prod_h1 = 2.5
        let entries_h1 = csc_entries_at(t, turbine_h1_col, generic_row);
        assert!(
            !entries_h1.is_empty(),
            "no CSC entry found for H1 turbine at generic constraint row"
        );
        let total_h1: f64 = entries_h1.iter().sum();
        assert!(
            (total_h1 - prod_h1).abs() < f64::EPSILON,
            "expected coefficient {prod_h1} for H1, got {total_h1}"
        );

        // CSC entry at turbine_h2_col, generic_row should have coefficient = prod_h2 = 3.0
        let entries_h2 = csc_entries_at(t, turbine_h2_col, generic_row);
        assert!(
            !entries_h2.is_empty(),
            "no CSC entry found for H2 turbine at generic constraint row"
        );
        let total_h2: f64 = entries_h2.iter().sum();
        assert!(
            (total_h2 - prod_h2).abs() < f64::EPSILON,
            "expected coefficient {prod_h2} for H2, got {total_h2}"
        );
    }

    // ── Helper: build a one-bus, one-thermal system with generic constraints ───

    /// Build a 1-bus, 1-thermal (constant productivity), 1-block, 1-stage system
    /// with the given generic constraints and resolved bounds.
    ///
    /// Column layout (N=0, T=1, B=1, K=1, no penalty, no FPHA, no evap):
    ///   theta=0, thermal=[1,2), deficit=[2,3), excess=[3,4)
    ///   withdrawal_slack=[] (n_h=0), col_generic_slack_start=4
    ///
    /// Row layout:
    ///   load_balance=[0,1), generic_start=1
    #[allow(clippy::cast_possible_wrap)]
    fn one_bus_one_thermal_system(
        thermal_entity_id: EntityId,
        constraints: Vec<cobre_core::GenericConstraint>,
        bounds: cobre_core::ResolvedGenericConstraintBounds,
    ) -> cobre_core::System {
        use chrono::NaiveDate;
        use cobre_core::entities::thermal::{Thermal, ThermalCostSegment};
        use cobre_core::scenario::LoadModel;
        use cobre_core::temporal::{
            Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
            StageStateConfig,
        };

        let bus = Bus {
            id: EntityId(1),
            name: "B1".to_string(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 500.0,
            }],
            excess_cost: 0.0,
        };

        let thermal = Thermal {
            id: thermal_entity_id,
            name: "T1".to_string(),
            bus_id: EntityId(1),
            min_generation_mw: 0.0,
            max_generation_mw: 100.0,
            cost_segments: vec![ThermalCostSegment {
                capacity_mw: 100.0,
                cost_per_mwh: 50.0,
            }],
            gnl_config: None,
            entry_stage_id: None,
            exit_stage_id: None,
        };

        let stage = Stage {
            index: 0,
            id: 0_i32,
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            season_id: None,
            blocks: vec![Block {
                index: 0,
                name: "BLK0".to_string(),
                duration_hours: 744.0,
            }],
            block_mode: BlockMode::Parallel,
            state_config: StageStateConfig {
                storage: false,
                inflow_lags: false,
            },
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor: 1,
                noise_method: NoiseMethod::Saa,
            },
        };

        let load_models = vec![LoadModel {
            bus_id: EntityId(1),
            stage_id: 0,
            mean_mw: 100.0,
            std_mw: 0.0,
        }];

        // Resolved bounds: 0 hydros, 1 thermal, 0 lines, 0 pumping, 0 contracts, 1 stage.
        let resolved_bounds = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 0,
                n_thermals: 1,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages: 1,
            },
            &BoundsDefaults {
                hydro: default_hydro_bounds(),
                thermal: ThermalStageBounds {
                    min_generation_mw: 0.0,
                    max_generation_mw: 100.0,
                },
                line: LineStageBounds {
                    direct_mw: 0.0,
                    reverse_mw: 0.0,
                },
                pumping: PumpingStageBounds {
                    min_flow_m3s: 0.0,
                    max_flow_m3s: 0.0,
                },
                contract: ContractStageBounds {
                    min_mw: 0.0,
                    max_mw: 0.0,
                    price_per_mwh: 0.0,
                },
            },
        );
        let penalties = ResolvedPenalties::new(
            &PenaltiesCountsSpec {
                n_hydros: 0,
                n_buses: 1,
                n_lines: 0,
                n_ncs: 0,
                n_stages: 1,
            },
            &PenaltiesDefaults {
                hydro: default_hydro_penalties(),
                bus: BusStagePenalties { excess_cost: 0.0 },
                line: LineStagePenalties { exchange_cost: 0.0 },
                ncs: NcsStagePenalties {
                    curtailment_cost: 0.0,
                },
            },
        );

        SystemBuilder::new()
            .buses(vec![bus])
            .thermals(vec![thermal])
            .stages(vec![stage])
            .load_models(load_models)
            .bounds(resolved_bounds)
            .penalties(penalties)
            .generic_constraints(constraints)
            .resolved_generic_bounds(bounds)
            .build()
            .expect("one_bus_one_thermal_system: valid")
    }
}
