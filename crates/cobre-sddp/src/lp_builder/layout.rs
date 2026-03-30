use std::collections::HashMap;

use cobre_core::{
    Bus, CascadeTopology, ConstraintSense, EntityId, GenericConstraint, Hydro, Line, LoadModel,
    NonControllableSource, ResolvedBounds, ResolvedExchangeFactors,
    ResolvedGenericConstraintBounds, ResolvedLoadFactors, ResolvedNcsBounds, ResolvedNcsFactors,
    ResolvedPenalties, Stage, Thermal,
};
use cobre_stochastic::par::precompute::PrecomputedPar;

use crate::hydro_models::{
    EvaporationModel, EvaporationModelSet, ProductionModelSet, ResolvedProductionModel,
};
use crate::indexer::StageIndexer;
use crate::inflow_method::InflowNonNegativityMethod;

use super::{GenericConstraintRowEntry, M3S_TO_HM3};

/// System-level context shared across all stages during template construction.
///
/// Bundles the references extracted from a [`System`] before the per-stage
/// loop begins. Constructed once in [`build_stage_templates`] and borrowed by
/// [`build_single_stage_template`] for each study stage.
pub(crate) struct TemplateBuildCtx<'a> {
    pub(crate) hydros: &'a [Hydro],
    pub(crate) thermals: &'a [Thermal],
    pub(crate) lines: &'a [Line],
    pub(crate) buses: &'a [Bus],
    pub(crate) load_models: &'a [LoadModel],
    pub(crate) cascade: &'a CascadeTopology,
    pub(crate) bounds: &'a ResolvedBounds,
    pub(crate) penalties: &'a ResolvedPenalties,
    pub(crate) hydro_pos: HashMap<EntityId, usize>,
    pub(crate) thermal_pos: HashMap<EntityId, usize>,
    pub(crate) line_pos: HashMap<EntityId, usize>,
    pub(crate) bus_pos: HashMap<EntityId, usize>,
    pub(crate) inflow_method: &'a InflowNonNegativityMethod,
    pub(crate) par_lp: &'a PrecomputedPar,
    /// Resolved production models for all (hydro, stage) pairs.
    pub(crate) production_models: &'a ProductionModelSet,
    /// Resolved evaporation models for all hydro plants.
    pub(crate) evaporation_models: &'a EvaporationModelSet,
    /// Generic constraint definitions (expression, sense, slack config).
    pub(crate) generic_constraints: &'a [GenericConstraint],
    /// Pre-resolved table mapping `(constraint_idx, stage_id)` to active bound entries.
    pub(crate) resolved_generic_bounds: &'a ResolvedGenericConstraintBounds,
    /// Pre-resolved per-block load scaling factors.
    pub(crate) resolved_load_factors: &'a ResolvedLoadFactors,
    /// Pre-resolved per-block exchange capacity factors.
    pub(crate) resolved_exchange_factors: &'a ResolvedExchangeFactors,
    /// Non-controllable source entities sorted by ID.
    pub(crate) non_controllable_sources: &'a [NonControllableSource],
    /// Pre-resolved per-stage NCS available generation bounds.
    pub(crate) resolved_ncs_bounds: &'a ResolvedNcsBounds,
    /// Pre-resolved per-block NCS generation scaling factors.
    pub(crate) resolved_ncs_factors: &'a ResolvedNcsFactors,
    /// Mapping from target hydro ID to source hydro indices that divert to it.
    ///
    /// For each hydro `d` with `diversion.downstream_id == target_id`, the map
    /// contains `d`'s system-level hydro index in the vec for `target_id`.
    /// Built once in `build_stage_templates()`.
    pub(crate) diversion_upstream: HashMap<EntityId, Vec<usize>>,
    pub(crate) n_hydros: usize,
    pub(crate) n_thermals: usize,
    pub(crate) n_lines: usize,
    pub(crate) n_buses: usize,
    pub(crate) max_par_order: usize,
    pub(crate) has_penalty: bool,
}

/// Pre-computed column and row layout offsets for a single stage LP.
///
/// Centralises the arithmetic that derives column-start and row-start indices
/// from entity counts and block count so that the filling helpers do not need
/// to recompute them independently.
pub(crate) struct StageLayout {
    pub(crate) n_blks: usize,
    pub(crate) n_h: usize,
    pub(crate) lag_order: usize,
    // Column regions
    pub(crate) col_turbine_start: usize,
    pub(crate) col_spillage_start: usize,
    /// Start of diversion flow columns (one per hydro per block).
    ///
    /// Layout within this region: `col_diversion_start + h_idx * n_blks + blk`.
    /// Hydros without diversion have bounds [0, 0]; presolve eliminates them.
    pub(crate) col_diversion_start: usize,
    pub(crate) col_thermal_start: usize,
    pub(crate) col_line_fwd_start: usize,
    pub(crate) col_line_rev_start: usize,
    pub(crate) col_deficit_start: usize,
    /// Maximum number of deficit segments across all buses (S).
    ///
    /// The deficit region spans `n_buses * max_deficit_segments * n_blks` columns.
    pub(crate) max_deficit_segments: usize,
    pub(crate) col_excess_start: usize,
    pub(crate) col_inflow_slack_start: usize,
    /// Start of FPHA generation columns (one per FPHA hydro per block).
    ///
    /// Layout within this region: `col_generation_start + local_fpha_idx * n_blks + blk`.
    pub(crate) col_generation_start: usize,
    // Row regions
    pub(crate) row_water_balance_start: usize,
    pub(crate) row_load_balance_start: usize,
    /// Start of FPHA constraint rows (after load-balance rows).
    ///
    /// Layout: `row_fpha_start + local_fpha_idx * n_blks * n_planes + blk * n_planes + plane_idx`.
    pub(crate) row_fpha_start: usize,
    /// Start of evaporation constraint rows (after FPHA rows).
    ///
    /// One equality row per evaporation hydro.
    /// Layout: `row_evap_start + local_evap_idx`.
    pub(crate) row_evap_start: usize,
    /// Start of evaporation columns (after FPHA generation columns).
    ///
    /// 3 stage-level columns per evaporation hydro (`Q_ev`, `f_evap_plus`, `f_evap_minus`).
    /// Layout: `col_evap_start + local_evap_idx * 3 + {0, 1, 2}`.
    pub(crate) col_evap_start: usize,
    /// Start of withdrawal slack columns (after evaporation columns).
    ///
    /// One stage-level column per operating hydro (`sigma^r_h`).
    /// Layout: `col_withdrawal_slack_start + h`.
    /// Zero when `n_h == 0`.
    pub(crate) col_withdrawal_slack_start: usize,
    /// Start of outflow-below-minimum slack columns (one per hydro per block).
    ///
    /// Inserted after withdrawal slack columns.
    /// Layout: `col_outflow_below_start + h_idx * n_blks + blk`.
    pub(crate) col_outflow_below_start: usize,
    /// Start of outflow-above-maximum slack columns (one per hydro per block).
    ///
    /// Layout: `col_outflow_above_start + h_idx * n_blks + blk`.
    pub(crate) col_outflow_above_start: usize,
    /// Start of turbine-below-minimum slack columns (one per hydro per block).
    ///
    /// Layout: `col_turbine_below_start + h_idx * n_blks + blk`.
    pub(crate) col_turbine_below_start: usize,
    /// Start of generation-below-minimum slack columns (one per hydro per block).
    ///
    /// Layout: `col_generation_below_start + h_idx * n_blks + blk`.
    pub(crate) col_generation_below_start: usize,
    /// Start of NCS generation columns (after operational violation slack columns).
    ///
    /// One column per active NCS per block.
    /// Layout: `col_ncs_start + ncs_local_idx * n_blks + blk`.
    pub(crate) col_ncs_start: usize,
    /// Number of active NCS entities at this stage.
    pub(crate) n_ncs: usize,
    /// Indices (into `ctx.non_controllable_sources`) of NCS active at this stage.
    pub(crate) active_ncs_indices: Vec<usize>,
    pub(crate) num_cols: usize,
    /// Start of minimum-outflow constraint rows (one per hydro per block, after evaporation rows).
    ///
    /// Layout: `row_min_outflow_start + h_idx * n_blks + blk`.
    pub(crate) row_min_outflow_start: usize,
    /// Start of maximum-outflow constraint rows (one per hydro per block).
    ///
    /// Layout: `row_max_outflow_start + h_idx * n_blks + blk`.
    pub(crate) row_max_outflow_start: usize,
    /// Start of minimum-turbine constraint rows (one per hydro per block).
    ///
    /// Layout: `row_min_turbine_start + h_idx * n_blks + blk`.
    pub(crate) row_min_turbine_start: usize,
    /// Start of minimum-generation constraint rows (one per hydro per block).
    ///
    /// Layout: `row_min_generation_start + h_idx * n_blks + blk`.
    pub(crate) row_min_generation_start: usize,
    /// Start of generic constraint rows (after operational violation rows).
    ///
    /// One row per active `(constraint, block)` pair.
    /// Equals `num_rows_before_generic` when no generic constraints are active.
    pub(crate) row_generic_start: usize,
    pub(crate) num_rows: usize,
    /// Total number of generic constraint rows for this stage.
    ///
    /// Zero when no generic constraints are active.
    pub(crate) n_generic_rows: usize,
    /// Start of z-inflow definition rows (after generic constraint rows).
    ///
    /// One equality row per hydro, defining `z_h = base_h + sigma_h * eta_h + sum_l[psi_l * lag_in[h,l]]`.
    pub(crate) row_z_inflow_start: usize,
    /// Start of z-inflow columns (after generic constraint slack columns).
    ///
    /// One free column per hydro (`z_h`, lower = -inf, upper = +inf, cost = 0.0).
    pub(crate) col_z_inflow_start: usize,
    // Template metadata
    pub(crate) n_state: usize,
    pub(crate) n_dual_relevant: usize,
    // Scalar derived quantities used by row-bound and matrix helpers
    pub(crate) zeta: f64,
    // FPHA hydro information for this stage
    /// Indices (into `ctx.hydros`) of hydros using FPHA at this stage.
    pub(crate) fpha_hydro_indices: Vec<usize>,
    /// Number of hyperplane planes per FPHA hydro at this stage.
    pub(crate) fpha_planes_per_hydro: Vec<usize>,
    // Evaporation hydro information for this stage
    /// Indices (into `ctx.hydros`) of hydros with linearized evaporation at this stage.
    pub(crate) evap_hydro_indices: Vec<usize>,
    /// Per-row metadata for active generic constraint rows at this stage.
    ///
    /// One entry per active `(constraint, block)` pair, in constraint-index-major
    /// order within each constraint's bound entries. Consumed by ticket-004 for
    /// CSC matrix construction, row bound filling, and objective coefficient filling.
    pub(crate) generic_constraint_rows: Vec<GenericConstraintRowEntry>,
}

impl StageLayout {
    #[allow(clippy::too_many_lines)]
    pub(crate) fn new(ctx: &TemplateBuildCtx<'_>, stage: &Stage, stage_idx: usize) -> Self {
        let n_blks = stage.blocks.len();
        let idx = StageIndexer::new(ctx.n_hydros, ctx.max_par_order);
        let decision_start = idx.theta + 1;

        let col_turbine_start = decision_start;
        let col_spillage_start = col_turbine_start + ctx.n_hydros * n_blks;
        let col_diversion_start = col_spillage_start + ctx.n_hydros * n_blks;
        let col_thermal_start = col_diversion_start + ctx.n_hydros * n_blks;
        let col_line_fwd_start = col_thermal_start + ctx.n_thermals * n_blks;
        let col_line_rev_start = col_line_fwd_start + ctx.n_lines * n_blks;
        let max_deficit_segments = ctx
            .buses
            .iter()
            .map(|b| b.deficit_segments.len())
            .max()
            .unwrap_or(0);
        let col_deficit_start = col_line_rev_start + ctx.n_lines * n_blks;
        let col_excess_start = col_deficit_start + ctx.n_buses * max_deficit_segments * n_blks;
        let col_excess_end = col_excess_start + ctx.n_buses * n_blks;
        let col_inflow_slack_start = col_excess_end;
        let n_slack_cols = if ctx.has_penalty { ctx.n_hydros } else { 0 };

        // ── FPHA: identify which hydros use FPHA at this stage ────────────────
        let mut fpha_hydro_indices: Vec<usize> = Vec::new();
        let mut fpha_planes_per_hydro: Vec<usize> = Vec::new();
        for h_idx in 0..ctx.n_hydros {
            if let ResolvedProductionModel::Fpha { planes, .. } =
                ctx.production_models.model(h_idx, stage_idx)
            {
                fpha_hydro_indices.push(h_idx);
                fpha_planes_per_hydro.push(planes.len());
            }
        }
        let n_fpha_hydros = fpha_hydro_indices.len();

        // Generation columns: one per FPHA hydro per block, placed after inflow slacks.
        let col_generation_start = col_inflow_slack_start + n_slack_cols;
        let n_generation_cols = n_fpha_hydros * n_blks;
        let col_generation_end = col_generation_start + n_generation_cols;

        // ── Evaporation: identify which hydros have linearized evaporation ────
        let mut evap_hydro_indices: Vec<usize> = Vec::new();
        for h_idx in 0..ctx.n_hydros {
            if matches!(
                ctx.evaporation_models.model(h_idx),
                EvaporationModel::Linearized { .. }
            ) {
                evap_hydro_indices.push(h_idx);
            }
        }
        let n_evap_hydros = evap_hydro_indices.len();

        // Evaporation columns: 3 per evaporation hydro (stage-level, not per-block),
        // placed after FPHA generation columns.
        let col_evap_start = col_generation_end;
        let n_evap_cols = n_evap_hydros * 3;

        // Withdrawal slack columns: one per operating hydro, placed after evaporation columns.
        let col_withdrawal_slack_start = col_evap_start + n_evap_cols;
        let withdrawal_slack_end = col_withdrawal_slack_start + ctx.n_hydros;

        // Operational violation slack columns: 4 families * (n_hydros * n_blks), placed after withdrawal slack.
        let n_op_slack = ctx.n_hydros * n_blks;
        let col_outflow_below_start = withdrawal_slack_end;
        let col_outflow_above_start = col_outflow_below_start + n_op_slack;
        let col_turbine_below_start = col_outflow_above_start + n_op_slack;
        let col_generation_below_start = col_turbine_below_start + n_op_slack;
        let operational_slack_end = col_generation_below_start + n_op_slack;

        let n_state = idx.n_state;
        let n_dual_relevant = n_state;
        // z_inflow rows occupy [n_dual_relevant, n_dual_relevant + n_hydros),
        // so water balance rows start after them.
        let row_water_balance_start = n_dual_relevant + ctx.n_hydros;
        let row_load_balance_start = row_water_balance_start + ctx.n_hydros;
        let row_load_balance_end = row_load_balance_start + ctx.n_buses * n_blks;

        // FPHA rows: one per FPHA hydro per block per plane, placed after load balance.
        let row_fpha_start = row_load_balance_end;
        let n_fpha_rows: usize = fpha_planes_per_hydro.iter().map(|&p| p * n_blks).sum();

        // Evaporation rows: 1 per evaporation hydro, placed after FPHA rows.
        let row_evap_start = row_fpha_start + n_fpha_rows;
        let evap_rows_end = row_evap_start + n_evap_hydros;

        // ── NCS: identify active NCS entities at this stage ─────────────────────
        let mut active_ncs_indices: Vec<usize> = Vec::new();
        for (ncs_idx, ncs) in ctx.non_controllable_sources.iter().enumerate() {
            let entered = ncs.entry_stage_id.is_none_or(|entry| entry <= stage.id);
            let not_exited = ncs.exit_stage_id.is_none_or(|exit| stage.id < exit);
            if entered && not_exited {
                active_ncs_indices.push(ncs_idx);
            }
        }
        let n_active_ncs = active_ncs_indices.len();

        // NCS generation columns: one per active NCS per block, placed after
        // operational violation slack columns (before generic constraint slack).
        let col_ncs_start = operational_slack_end;
        let n_ncs_cols = n_active_ncs * n_blks;
        let col_ncs_end = col_ncs_start + n_ncs_cols;

        // Operational violation rows: 4 families * (n_hydros * n_blks), placed after evaporation rows.
        let n_op_rows = ctx.n_hydros * n_blks;
        let row_min_outflow_start = evap_rows_end;
        let row_max_outflow_start = row_min_outflow_start + n_op_rows;
        let row_min_turbine_start = row_max_outflow_start + n_op_rows;
        let row_min_generation_start = row_min_turbine_start + n_op_rows;
        let operational_rows_end = row_min_generation_start + n_op_rows;

        // ── Generic constraints: identify active rows and slack columns ─────────
        // Generic constraint rows are placed after operational violation rows (last row region).
        // Generic constraint slack columns are placed after NCS columns (last col region).
        let row_generic_start = operational_rows_end;
        let col_generic_slack_start = col_ncs_end;

        let mut n_generic_rows: usize = 0;
        let mut n_generic_slack_cols: usize = 0;
        let mut generic_constraint_rows: Vec<GenericConstraintRowEntry> = Vec::new();

        for (constraint_idx, constraint) in ctx.generic_constraints.iter().enumerate() {
            if !ctx
                .resolved_generic_bounds
                .is_active(constraint_idx, stage.id)
            {
                continue;
            }

            let bound_entries = ctx
                .resolved_generic_bounds
                .bounds_for_stage(constraint_idx, stage.id);

            for &(block_id, bound) in bound_entries {
                match block_id {
                    None => {
                        // One row per block.
                        for block_idx in 0..n_blks {
                            let row_offset = row_generic_start + n_generic_rows;
                            let (slack_plus_col, slack_minus_col) = if constraint.slack.enabled {
                                let plus_col = col_generic_slack_start + n_generic_slack_cols;
                                n_generic_slack_cols += 1;
                                let minus_col = if constraint.sense == ConstraintSense::Equal {
                                    let mc = col_generic_slack_start + n_generic_slack_cols;
                                    n_generic_slack_cols += 1;
                                    Some(mc)
                                } else {
                                    None
                                };
                                (Some(plus_col), minus_col)
                            } else {
                                (None, None)
                            };
                            let _ = row_offset; // used indirectly via n_generic_rows
                            n_generic_rows += 1;
                            generic_constraint_rows.push(GenericConstraintRowEntry {
                                constraint_idx,
                                entity_id: constraint.id.0,
                                block_idx,
                                bound,
                                sense: constraint.sense,
                                slack_enabled: constraint.slack.enabled,
                                slack_penalty: constraint.slack.penalty.unwrap_or(0.0),
                                slack_plus_col,
                                slack_minus_col,
                            });
                        }
                    }
                    Some(blk_id) => {
                        // One row for the specific block (0-indexed from the block_id value).
                        // block_id in bounds is a non-negative 0-indexed block position;
                        // upstream validation ensures it is non-negative.
                        #[allow(clippy::cast_sign_loss)]
                        let block_idx = blk_id as usize;
                        let (slack_plus_col, slack_minus_col) = if constraint.slack.enabled {
                            let plus_col = col_generic_slack_start + n_generic_slack_cols;
                            n_generic_slack_cols += 1;
                            let minus_col = if constraint.sense == ConstraintSense::Equal {
                                let mc = col_generic_slack_start + n_generic_slack_cols;
                                n_generic_slack_cols += 1;
                                Some(mc)
                            } else {
                                None
                            };
                            (Some(plus_col), minus_col)
                        } else {
                            (None, None)
                        };
                        n_generic_rows += 1;
                        generic_constraint_rows.push(GenericConstraintRowEntry {
                            constraint_idx,
                            entity_id: constraint.id.0,
                            block_idx,
                            bound,
                            sense: constraint.sense,
                            slack_enabled: constraint.slack.enabled,
                            slack_penalty: constraint.slack.penalty.unwrap_or(0.0),
                            slack_plus_col,
                            slack_minus_col,
                        });
                    }
                }
            }
        }

        // z-inflow columns and rows are at fixed offset N*(1+L), from the indexer.
        // They are embedded inside the layout (between lags and storage_in for columns,
        // between lag-fixing and water balance for rows), NOT at the end.
        let col_z_inflow_start = idx.z_inflow.start; // = N*(1+L)
        let row_z_inflow_start = idx.z_inflow_row_start; // = N*(1+L)

        // num_cols and num_rows: generic slack/rows are now the last variable-size regions.
        let num_cols = col_generic_slack_start + n_generic_slack_cols;
        let num_rows = row_generic_start + n_generic_rows;

        let total_stage_hours: f64 = stage.blocks.iter().map(|b| b.duration_hours).sum();
        let zeta = total_stage_hours * M3S_TO_HM3;

        Self {
            n_blks,
            n_h: ctx.n_hydros,
            lag_order: ctx.max_par_order,
            col_turbine_start,
            col_spillage_start,
            col_diversion_start,
            col_thermal_start,
            col_line_fwd_start,
            col_line_rev_start,
            col_deficit_start,
            max_deficit_segments,
            col_excess_start,
            col_inflow_slack_start,
            col_generation_start,
            col_evap_start,
            col_withdrawal_slack_start,
            col_outflow_below_start,
            col_outflow_above_start,
            col_turbine_below_start,
            col_generation_below_start,
            col_ncs_start,
            n_ncs: n_active_ncs,
            active_ncs_indices,
            num_cols,
            row_water_balance_start,
            row_load_balance_start,
            row_fpha_start,
            row_evap_start,
            row_min_outflow_start,
            row_max_outflow_start,
            row_min_turbine_start,
            row_min_generation_start,
            row_generic_start,
            num_rows,
            n_generic_rows,
            row_z_inflow_start,
            col_z_inflow_start,
            n_state,
            n_dual_relevant,
            zeta,
            fpha_hydro_indices,
            fpha_planes_per_hydro,
            evap_hydro_indices,
            generic_constraint_rows,
        }
    }
}
