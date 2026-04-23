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
    /// Start of under-withdrawal slack columns (after evaporation columns).
    ///
    /// One stage-level column per operating hydro.
    /// Layout: `col_withdrawal_neg_start + h`.
    /// Zero when `n_h == 0`.
    pub(crate) col_withdrawal_neg_start: usize,
    /// Start of over-withdrawal slack columns (after under-withdrawal slacks).
    ///
    /// One stage-level column per operating hydro.
    /// Layout: `col_withdrawal_pos_start + h`.
    /// Zero when `n_h == 0`.
    pub(crate) col_withdrawal_pos_start: usize,
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
    /// order within each constraint's bound entries. Used for CSC matrix construction,
    /// row bound filling, and objective coefficient filling.
    pub(crate) generic_constraint_rows: Vec<GenericConstraintRowEntry>,
}

// ── Private helper return structs ─────────────────────────────────────────────

/// Column offsets for the decision variable region of the LP.
struct DecisionColumnOffsets {
    col_turbine_start: usize,
    col_spillage_start: usize,
    col_diversion_start: usize,
    col_thermal_start: usize,
    col_line_fwd_start: usize,
    col_line_rev_start: usize,
    col_deficit_start: usize,
    col_excess_start: usize,
    col_inflow_slack_start: usize,
    max_deficit_segments: usize,
    n_slack_cols: usize,
}

/// Column offsets for the FPHA generation variable region of the LP.
struct FphaColumnOffsets {
    col_generation_start: usize,
    col_generation_end: usize,
}

/// Column offsets for the evaporation variable region of the LP.
struct EvapColumnOffsets {
    col_evap_start: usize,
    n_evap_cols: usize,
}

/// Column offsets for all operational slack variable regions of the LP.
struct OperationalSlackColumnOffsets {
    col_withdrawal_neg_start: usize,
    col_withdrawal_pos_start: usize,
    col_outflow_below_start: usize,
    col_outflow_above_start: usize,
    col_turbine_below_start: usize,
    col_generation_below_start: usize,
    operational_slack_end: usize,
}

/// Layout metadata for all active generic constraint rows and slack columns.
struct GenericConstraintLayout {
    n_generic_rows: usize,
    n_generic_slack_cols: usize,
    generic_constraint_rows: Vec<GenericConstraintRowEntry>,
}

// ── Private helper functions ───────────────────────────────────────────────────

/// Compute column offsets for the core decision variable region.
///
/// Covers turbine, spillage, diversion, thermal, `line_fwd`, `line_rev`, deficit,
/// excess, and inflow slack columns in that order.
fn compute_decision_column_offsets(
    ctx: &TemplateBuildCtx<'_>,
    n_blks: usize,
    decision_start: usize,
) -> DecisionColumnOffsets {
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
    let col_inflow_slack_start = col_excess_start + ctx.n_buses * n_blks;
    let n_slack_cols = if ctx.has_penalty { ctx.n_hydros } else { 0 };

    DecisionColumnOffsets {
        col_turbine_start,
        col_spillage_start,
        col_diversion_start,
        col_thermal_start,
        col_line_fwd_start,
        col_line_rev_start,
        col_deficit_start,
        col_excess_start,
        col_inflow_slack_start,
        max_deficit_segments,
        n_slack_cols,
    }
}

/// Identify which hydros use FPHA at this stage and compute their column offsets.
///
/// Returns the FPHA column offsets along with the hydro index and plane-count
/// vectors needed for row layout and matrix construction.
fn identify_and_layout_fpha_columns(
    ctx: &TemplateBuildCtx<'_>,
    stage_idx: usize,
    n_blks: usize,
    col_inflow_slack_start: usize,
    n_slack_cols: usize,
) -> (FphaColumnOffsets, Vec<usize>, Vec<usize>) {
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
    let col_generation_start = col_inflow_slack_start + n_slack_cols;
    let col_generation_end = col_generation_start + n_fpha_hydros * n_blks;

    (
        FphaColumnOffsets {
            col_generation_start,
            col_generation_end,
        },
        fpha_hydro_indices,
        fpha_planes_per_hydro,
    )
}

/// Identify which hydros have linearized evaporation and compute their column offsets.
///
/// Returns the evaporation column offsets along with the hydro index vector
/// needed for row layout and matrix construction.
fn identify_and_layout_evap_columns(
    ctx: &TemplateBuildCtx<'_>,
    col_generation_end: usize,
) -> (EvapColumnOffsets, Vec<usize>) {
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
    let col_evap_start = col_generation_end;
    let n_evap_cols = n_evap_hydros * 3;

    (
        EvapColumnOffsets {
            col_evap_start,
            n_evap_cols,
        },
        evap_hydro_indices,
    )
}

/// Compute column offsets for all operational slack variable families.
///
/// Covers withdrawal (neg/pos) slacks and the four per-hydro-per-block
/// operational violation slacks (`outflow_below`, `outflow_above`, `turbine_below`,
/// `generation_below`), placed after the evaporation columns.
fn compute_operational_slack_column_offsets(
    ctx: &TemplateBuildCtx<'_>,
    n_blks: usize,
    withdrawal_slack_start: usize,
) -> OperationalSlackColumnOffsets {
    let col_withdrawal_neg_start = withdrawal_slack_start;
    let col_withdrawal_pos_start = col_withdrawal_neg_start + ctx.n_hydros;

    let n_op_slack = ctx.n_hydros * n_blks;
    let col_outflow_below_start = col_withdrawal_pos_start + ctx.n_hydros;
    let col_outflow_above_start = col_outflow_below_start + n_op_slack;
    let col_turbine_below_start = col_outflow_above_start + n_op_slack;
    let col_generation_below_start = col_turbine_below_start + n_op_slack;
    let operational_slack_end = col_generation_below_start + n_op_slack;

    OperationalSlackColumnOffsets {
        col_withdrawal_neg_start,
        col_withdrawal_pos_start,
        col_outflow_below_start,
        col_outflow_above_start,
        col_turbine_below_start,
        col_generation_below_start,
        operational_slack_end,
    }
}

/// Collect indices of NCS entities that are active at this stage.
///
/// An NCS is active when the stage is at or after the entry stage (if any) and
/// strictly before the exit stage (if any).
fn identify_active_ncs(ctx: &TemplateBuildCtx<'_>, stage: &Stage) -> Vec<usize> {
    ctx.non_controllable_sources
        .iter()
        .enumerate()
        .filter_map(|(i, ncs)| {
            let ok = ncs.entry_stage_id.is_none_or(|e| e <= stage.id)
                && ncs.exit_stage_id.is_none_or(|e| stage.id < e);
            ok.then_some(i)
        })
        .collect()
}

/// Enumerate active generic constraint rows and assign their slack column indices.
///
/// For each active `(constraint, block)` pair at this stage, one
/// [`GenericConstraintRowEntry`] is produced. Slack columns are allocated
/// sequentially from `col_generic_slack_start` — first the plus-slack, then
/// (for equality constraints) the minus-slack.
fn enumerate_generic_constraint_rows(
    ctx: &TemplateBuildCtx<'_>,
    stage: &Stage,
    n_blks: usize,
    row_generic_start: usize,
    col_generic_slack_start: usize,
) -> GenericConstraintLayout {
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

    GenericConstraintLayout {
        n_generic_rows,
        n_generic_slack_cols,
        generic_constraint_rows,
    }
}

impl StageLayout {
    pub(crate) fn new(ctx: &TemplateBuildCtx<'_>, stage: &Stage, stage_idx: usize) -> Self {
        let n_blks = stage.blocks.len();
        let idx = StageIndexer::new(ctx.n_hydros, ctx.max_par_order);
        let decision_start = idx.theta + 1;

        // Column offsets: decision, FPHA, evaporation, operational slacks.
        let dec = compute_decision_column_offsets(ctx, n_blks, decision_start);
        let (fpha_cols, fpha_hydro_indices, fpha_planes_per_hydro) =
            identify_and_layout_fpha_columns(
                ctx,
                stage_idx,
                n_blks,
                dec.col_inflow_slack_start,
                dec.n_slack_cols,
            );
        let (evap_cols, evap_hydro_indices) =
            identify_and_layout_evap_columns(ctx, fpha_cols.col_generation_end);
        let op_slack = compute_operational_slack_column_offsets(
            ctx,
            n_blks,
            evap_cols.col_evap_start + evap_cols.n_evap_cols,
        );

        // NCS: identify active entities and compute their column region.
        let active_ncs_indices = identify_active_ncs(ctx, stage);
        let n_active_ncs = active_ncs_indices.len();
        let col_ncs_start = op_slack.operational_slack_end;
        let col_ncs_end = col_ncs_start + n_active_ncs * n_blks;

        // Row offsets: state, water balance, load balance, FPHA, evap, operational, generic.
        let n_state = idx.n_state;
        let n_dual_relevant = n_state;
        let row_water_balance_start = n_dual_relevant + ctx.n_hydros;
        let row_load_balance_start = row_water_balance_start + ctx.n_hydros;
        let row_fpha_start = row_load_balance_start + ctx.n_buses * n_blks;
        let n_fpha_rows: usize = fpha_planes_per_hydro.iter().map(|&p| p * n_blks).sum();
        let row_evap_start = row_fpha_start + n_fpha_rows;
        let n_op_rows = ctx.n_hydros * n_blks;
        let row_min_outflow_start = row_evap_start + evap_hydro_indices.len();
        let row_max_outflow_start = row_min_outflow_start + n_op_rows;
        let row_min_turbine_start = row_max_outflow_start + n_op_rows;
        let row_min_generation_start = row_min_turbine_start + n_op_rows;
        let row_generic_start = row_min_generation_start + n_op_rows;

        // Generic constraints: active rows and slack columns.
        let col_generic_slack_start = col_ncs_end;
        let generic = enumerate_generic_constraint_rows(
            ctx,
            stage,
            n_blks,
            row_generic_start,
            col_generic_slack_start,
        );

        // z-inflow columns and rows: fixed positions from the indexer.
        let col_z_inflow_start = idx.z_inflow.start;
        let row_z_inflow_start = idx.z_inflow_row_start;
        let num_cols = col_generic_slack_start + generic.n_generic_slack_cols;
        let num_rows = row_generic_start + generic.n_generic_rows;
        let zeta = stage.blocks.iter().map(|b| b.duration_hours).sum::<f64>() * M3S_TO_HM3;

        Self {
            n_blks,
            n_h: ctx.n_hydros,
            lag_order: ctx.max_par_order,
            col_turbine_start: dec.col_turbine_start,
            col_spillage_start: dec.col_spillage_start,
            col_diversion_start: dec.col_diversion_start,
            col_thermal_start: dec.col_thermal_start,
            col_line_fwd_start: dec.col_line_fwd_start,
            col_line_rev_start: dec.col_line_rev_start,
            col_deficit_start: dec.col_deficit_start,
            max_deficit_segments: dec.max_deficit_segments,
            col_excess_start: dec.col_excess_start,
            col_inflow_slack_start: dec.col_inflow_slack_start,
            col_generation_start: fpha_cols.col_generation_start,
            col_evap_start: evap_cols.col_evap_start,
            col_withdrawal_neg_start: op_slack.col_withdrawal_neg_start,
            col_withdrawal_pos_start: op_slack.col_withdrawal_pos_start,
            col_outflow_below_start: op_slack.col_outflow_below_start,
            col_outflow_above_start: op_slack.col_outflow_above_start,
            col_turbine_below_start: op_slack.col_turbine_below_start,
            col_generation_below_start: op_slack.col_generation_below_start,
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
            n_generic_rows: generic.n_generic_rows,
            row_z_inflow_start,
            col_z_inflow_start,
            n_state,
            n_dual_relevant,
            zeta,
            fpha_hydro_indices,
            fpha_planes_per_hydro,
            evap_hydro_indices,
            generic_constraint_rows: generic.generic_constraint_rows,
        }
    }
}
