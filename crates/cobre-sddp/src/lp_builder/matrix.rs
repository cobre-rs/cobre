use cobre_core::entities::hydro::HydroGenerationModel;
use cobre_core::{ConstraintSense, Stage};

use crate::generic_constraints::resolve_variable_ref;
use crate::hydro_models::{EvaporationModel, ResolvedProductionModel};
use crate::indexer::StageIndexer;

use super::layout::{StageLayout, TemplateBuildCtx};
use super::{M3S_TO_HM3, Q_EV_SAFETY_MARGIN};

/// Fill column lower/upper bounds and objective coefficients for one stage.
///
/// Returns `(col_lower, col_upper, objective)` vectors of length `layout.num_cols`.
#[allow(clippy::too_many_lines)]
pub(super) fn fill_stage_columns(
    ctx: &TemplateBuildCtx<'_>,
    stage: &Stage,
    stage_idx: usize,
    layout: &StageLayout,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let idx = StageIndexer::new(ctx.n_hydros, ctx.max_par_order);
    let mut col_lower = vec![0.0_f64; layout.num_cols];
    let mut col_upper = vec![f64::INFINITY; layout.num_cols];
    let mut objective = vec![0.0_f64; layout.num_cols];

    // Outgoing and incoming storage columns.
    for (h_idx, _hydro) in ctx.hydros.iter().enumerate() {
        let hb = ctx.bounds.hydro_bounds(h_idx, stage_idx);
        col_lower[h_idx] = hb.min_storage_hm3;
        col_upper[h_idx] = hb.max_storage_hm3;
        col_lower[idx.storage_in.start + h_idx] = f64::NEG_INFINITY;
        col_upper[idx.storage_in.start + h_idx] = f64::INFINITY;
    }

    // AR lag columns: unconstrained (signed).
    for lag_col in idx.inflow_lags.clone() {
        col_lower[lag_col] = f64::NEG_INFINITY;
        col_upper[lag_col] = f64::INFINITY;
    }

    // Theta: bounded below by zero so iteration-1 LPs with empty cut pools
    // are bounded rather than unbounded.
    col_lower[idx.theta] = 0.0;
    col_upper[idx.theta] = f64::INFINITY;
    objective[idx.theta] = 1.0;

    // Turbine columns per hydro per block.
    for (h_idx, _hydro) in ctx.hydros.iter().enumerate() {
        let hb = ctx.bounds.hydro_bounds(h_idx, stage_idx);
        let hp = ctx.penalties.hydro_penalties(h_idx, stage_idx);
        let model = ctx.production_models.model(h_idx, stage_idx);
        let is_fpha = matches!(model, ResolvedProductionModel::Fpha { .. });
        // For constant-productivity hydros, cap turbine flow so that
        // productivity * turbined <= max_generation_mw (derated capacity).
        let turb_upper = match model {
            ResolvedProductionModel::ConstantProductivity { productivity }
                if *productivity > 0.0 =>
            {
                hb.max_turbined_m3s.min(hb.max_generation_mw / productivity)
            }
            _ => hb.max_turbined_m3s,
        };
        for blk in 0..layout.n_blks {
            let col = layout.col_turbine_start + h_idx * layout.n_blks + blk;
            col_lower[col] = 0.0;
            col_upper[col] = turb_upper;
            if is_fpha {
                let block_hours = stage.blocks[blk].duration_hours;
                objective[col] = hp.fpha_turbined_cost * block_hours;
            }
        }
    }

    // Spillage columns per hydro per block.
    for (h_idx, _hydro) in ctx.hydros.iter().enumerate() {
        let hp = ctx.penalties.hydro_penalties(h_idx, stage_idx);
        for blk in 0..layout.n_blks {
            let col = layout.col_spillage_start + h_idx * layout.n_blks + blk;
            col_upper[col] = f64::INFINITY;
            let block_hours = stage.blocks[blk].duration_hours;
            objective[col] = hp.spillage_cost * block_hours;
        }
    }

    // Diversion columns per hydro per block.
    // Dense allocation: all hydros get columns; those without diversion have
    // bounds [0, 0] and are eliminated by presolve.
    for (h_idx, hydro) in ctx.hydros.iter().enumerate() {
        let hp = ctx.penalties.hydro_penalties(h_idx, stage_idx);
        let max_div = hydro.diversion.as_ref().map_or(0.0, |d| d.max_flow_m3s);
        for blk in 0..layout.n_blks {
            let col = layout.col_diversion_start + h_idx * layout.n_blks + blk;
            col_lower[col] = 0.0;
            col_upper[col] = max_div;
            if max_div > 0.0 {
                let block_hours = stage.blocks[blk].duration_hours;
                objective[col] = hp.diversion_cost * block_hours;
            }
        }
    }

    // Thermal columns per thermal per block.
    for (t_idx, _thermal) in ctx.thermals.iter().enumerate() {
        let tb = ctx.bounds.thermal_bounds(t_idx, stage_idx);
        let marginal_cost_per_mwh = tb.cost_per_mwh;
        for blk in 0..layout.n_blks {
            let col = layout.col_thermal_start + t_idx * layout.n_blks + blk;
            col_lower[col] = tb.min_generation_mw;
            col_upper[col] = tb.max_generation_mw;
            let block_hours = stage.blocks[blk].duration_hours;
            objective[col] = marginal_cost_per_mwh * block_hours;
        }
    }

    // Line columns per line per block (forward and reverse).
    // Exchange factors from `exchange_factors.json` scale the stage-level
    // capacity bounds per block. Default factor is (1.0, 1.0) (no scaling).
    for (l_idx, line) in ctx.lines.iter().enumerate() {
        let lb = ctx.bounds.line_bounds(l_idx, stage_idx);
        let lp = ctx.penalties.line_penalties(l_idx, stage_idx);
        for blk in 0..layout.n_blks {
            let (df, rf) = ctx.resolved_exchange_factors.factors(l_idx, stage_idx, blk);
            let col_fwd = layout.col_line_fwd_start + l_idx * layout.n_blks + blk;
            let col_rev = layout.col_line_rev_start + l_idx * layout.n_blks + blk;
            col_upper[col_fwd] = lb.direct_mw * df;
            col_upper[col_rev] = lb.reverse_mw * rf;
            let block_hours = stage.blocks[blk].duration_hours;
            objective[col_fwd] = lp.exchange_cost * block_hours;
            objective[col_rev] = lp.exchange_cost * block_hours;
            let _ = line;
        }
    }

    // Deficit and excess columns per bus per block.
    //
    // The deficit region uses a uniform stride of `max_deficit_segments` segments
    // per bus.  For bus `b_idx`, segment `seg_idx`, block `blk`:
    //   col = col_deficit_start + b_idx * max_deficit_segments * n_blks + seg_idx * n_blks + blk
    //
    // Buses with fewer than `max_deficit_segments` segments leave the trailing
    // slots at [lower=0, upper=0, objective=0] (from vec initialisation), which
    // the HiGHS presolver eliminates before the simplex phase.
    for (b_idx, bus) in ctx.buses.iter().enumerate() {
        let bp = ctx.penalties.bus_penalties(b_idx, stage_idx);
        for (seg_idx, segment) in bus.deficit_segments.iter().enumerate() {
            for blk in 0..layout.n_blks {
                let col_def = layout.col_deficit_start
                    + b_idx * layout.max_deficit_segments * layout.n_blks
                    + seg_idx * layout.n_blks
                    + blk;
                let block_hours = stage.blocks[blk].duration_hours;
                col_upper[col_def] = segment.depth_mw.unwrap_or(f64::INFINITY);
                objective[col_def] = segment.cost_per_mwh * block_hours;
            }
        }
        for blk in 0..layout.n_blks {
            let col_exc = layout.col_excess_start + b_idx * layout.n_blks + blk;
            let block_hours = stage.blocks[blk].duration_hours;
            col_upper[col_exc] = f64::INFINITY;
            objective[col_exc] = bp.excess_cost * block_hours;
        }
    }

    // Inflow non-negativity slack columns (sigma_inf_h), one per hydro.
    // Bounds [0, +inf) come from vec initialisation; only objective needs writing.
    // Per-plant cost from the penalty cascade (T-010).
    if ctx.has_penalty {
        let total_stage_hours: f64 = stage.blocks.iter().map(|b| b.duration_hours).sum();
        for h_idx in 0..layout.n_h {
            let col = layout.col_inflow_slack_start + h_idx;
            let hp = ctx.penalties.hydro_penalties(h_idx, stage_idx);
            objective[col] = hp.inflow_nonnegativity_cost * total_stage_hours;
        }
    }

    // FPHA generation columns (g_{h,k}): one per FPHA hydro per block.
    // Bounds: [0, max_generation_mw].  Objective: 0.0 (fpha_turbined_cost goes on
    // the turbine column).
    for (local_idx, &h_idx) in layout.fpha_hydro_indices.iter().enumerate() {
        let hb = ctx.bounds.hydro_bounds(h_idx, stage_idx);
        for blk in 0..layout.n_blks {
            let col = layout.col_generation_start + local_idx * layout.n_blks + blk;
            col_lower[col] = 0.0;
            col_upper[col] = hb.max_generation_mw;
        }
    }

    // Evaporation columns: 3 per evaporation hydro (stage-level, not per-block).
    // Q_ev_h, f_evap_plus_h, f_evap_minus_h — all in [0, +inf).
    // Q_ev carries zero objective cost (evaporation flow itself is not penalised).
    // f_evap_plus and f_evap_minus carry evaporation_violation_cost * total_stage_hours
    // so that the solver is penalised for violating the linearised evaporation constraint.
    let total_stage_hours: f64 = stage.blocks.iter().map(|b| b.duration_hours).sum();
    for (local_idx, &h_idx) in layout.evap_hydro_indices.iter().enumerate() {
        let col_q_ev = layout.col_evap_start + local_idx * 3;
        let col_f_plus = layout.col_evap_start + local_idx * 3 + 1;
        let col_f_minus = layout.col_evap_start + local_idx * 3 + 2;
        col_lower[col_q_ev] = 0.0;
        // Physical upper bound: linearized evaporation at maximum storage.
        // Q_ev_max = max(0, k_evap0 + k_evap_v * v_max) * safety_margin.
        // Negative coefficients (net condensation) are clamped to zero.
        if let EvaporationModel::Linearized { coefficients, .. } =
            ctx.evaporation_models.model(h_idx)
        {
            let coeff = &coefficients[stage_idx];
            let hb = ctx.bounds.hydro_bounds(h_idx, stage_idx);
            let q_ev_max = (coeff.k_evap0 + coeff.k_evap_v * hb.max_storage_hm3).max(0.0);
            col_upper[col_q_ev] = q_ev_max * Q_EV_SAFETY_MARGIN;
        } else {
            col_upper[col_q_ev] = 0.0;
        }
        col_lower[col_f_plus] = 0.0;
        col_upper[col_f_plus] = f64::INFINITY;
        col_lower[col_f_minus] = 0.0;
        col_upper[col_f_minus] = f64::INFINITY;
        // Violation cost: read directional costs from resolved penalties.
        // Q_ev (offset 0) keeps objective = 0.0 (already the vec initialisation default).
        // f_evap_plus = under-evaporation (evaporated less than target).
        // f_evap_minus = over-evaporation (evaporated more than target).
        let hp = ctx.penalties.hydro_penalties(h_idx, stage_idx);
        objective[col_f_plus] = hp.evaporation_violation_neg_cost * total_stage_hours;
        objective[col_f_minus] = hp.evaporation_violation_pos_cost * total_stage_hours;
    }

    // Withdrawal violation slack columns — neg (under-withdrawal), one per hydro.
    // Bounds [0, +inf) when water_withdrawal_m3s > 0; pinned to [0, 0] otherwise.
    // Pinning to zero when there is no scheduled withdrawal ensures the column has
    // no LP effect (it is presolve-eliminated), preserving identical behaviour to
    // the pre-withdrawal implementation for cases where withdrawal is absent.
    for h_idx in 0..layout.n_h {
        let col = layout.col_withdrawal_neg_start + h_idx;
        let hb = ctx.bounds.hydro_bounds(h_idx, stage_idx);
        if hb.water_withdrawal_m3s > 0.0 {
            col_upper[col] = f64::INFINITY;
        } else {
            col_upper[col] = 0.0;
        }
        let hp = ctx.penalties.hydro_penalties(h_idx, stage_idx);
        objective[col] = hp.water_withdrawal_violation_neg_cost * total_stage_hours;
    }

    // Withdrawal violation slack columns — pos (over-withdrawal), one per hydro.
    // Same activation logic as neg: active only when withdrawal target > 0.
    for h_idx in 0..layout.n_h {
        let col = layout.col_withdrawal_pos_start + h_idx;
        let hb = ctx.bounds.hydro_bounds(h_idx, stage_idx);
        if hb.water_withdrawal_m3s > 0.0 {
            col_upper[col] = f64::INFINITY;
        } else {
            col_upper[col] = 0.0;
        }
        let hp = ctx.penalties.hydro_penalties(h_idx, stage_idx);
        objective[col] = hp.water_withdrawal_violation_pos_cost * total_stage_hours;
    }

    // Operational violation slack columns: 4 regions of n_h * n_blks columns each.
    // Bounds [0, +inf) when the corresponding bound is active (positive for
    // min constraints, Some for max outflow); pinned to [0, 0] when inactive.
    // Objective cost: resolved penalty * block_hours.

    // Outflow-below-minimum slacks (sigma_outflow_below_{h,k}).
    for h_idx in 0..layout.n_h {
        let hb = ctx.bounds.hydro_bounds(h_idx, stage_idx);
        let hp = ctx.penalties.hydro_penalties(h_idx, stage_idx);
        for blk in 0..layout.n_blks {
            let col = layout.col_outflow_below_start + h_idx * layout.n_blks + blk;
            if hb.min_outflow_m3s > 0.0 {
                col_upper[col] = f64::INFINITY;
            } else {
                col_upper[col] = 0.0;
            }
            let block_hours = stage.blocks[blk].duration_hours;
            objective[col] = hp.outflow_violation_below_cost * block_hours;
        }
    }

    // Outflow-above-maximum slacks (sigma_outflow_above_{h,k}).
    for h_idx in 0..layout.n_h {
        let hb = ctx.bounds.hydro_bounds(h_idx, stage_idx);
        let hp = ctx.penalties.hydro_penalties(h_idx, stage_idx);
        for blk in 0..layout.n_blks {
            let col = layout.col_outflow_above_start + h_idx * layout.n_blks + blk;
            if hb.max_outflow_m3s.is_some() {
                col_upper[col] = f64::INFINITY;
            } else {
                col_upper[col] = 0.0;
            }
            let block_hours = stage.blocks[blk].duration_hours;
            objective[col] = hp.outflow_violation_above_cost * block_hours;
        }
    }

    // Turbine-below-minimum slacks (sigma_turbine_below_{h,k}).
    for h_idx in 0..layout.n_h {
        let hb = ctx.bounds.hydro_bounds(h_idx, stage_idx);
        let hp = ctx.penalties.hydro_penalties(h_idx, stage_idx);
        for blk in 0..layout.n_blks {
            let col = layout.col_turbine_below_start + h_idx * layout.n_blks + blk;
            if hb.min_turbined_m3s > 0.0 {
                col_upper[col] = f64::INFINITY;
            } else {
                col_upper[col] = 0.0;
            }
            let block_hours = stage.blocks[blk].duration_hours;
            objective[col] = hp.turbined_violation_below_cost * block_hours;
        }
    }

    // Generation-below-minimum slacks (sigma_generation_below_{h,k}).
    for h_idx in 0..layout.n_h {
        let hb = ctx.bounds.hydro_bounds(h_idx, stage_idx);
        let hp = ctx.penalties.hydro_penalties(h_idx, stage_idx);
        for blk in 0..layout.n_blks {
            let col = layout.col_generation_below_start + h_idx * layout.n_blks + blk;
            if hb.min_generation_mw > 0.0 {
                col_upper[col] = f64::INFINITY;
            } else {
                col_upper[col] = 0.0;
            }
            let block_hours = stage.blocks[blk].duration_hours;
            objective[col] = hp.generation_violation_below_cost * block_hours;
        }
    }

    // NCS generation columns: one per active NCS per block.
    // col_lower[col] = 0.0 (from vec initialisation).
    // col_upper[col] = available_gen * ncs_factor.
    // objective[col] = -curtailment_cost * block_hours (negative incentivises generation).
    for (ncs_local, &ncs_sys_idx) in layout.active_ncs_indices.iter().enumerate() {
        let ncs = &ctx.non_controllable_sources[ncs_sys_idx];
        let avail_gen = ctx
            .resolved_ncs_bounds
            .available_generation(ncs_sys_idx, stage_idx);
        for blk in 0..layout.n_blks {
            let col = layout.col_ncs_start + ncs_local * layout.n_blks + blk;
            let factor = ctx.resolved_ncs_factors.factor(ncs_sys_idx, stage_idx, blk);
            col_upper[col] = avail_gen * factor;
            let block_hours = stage.blocks[blk].duration_hours;
            objective[col] = -ncs.curtailment_cost * block_hours;
        }
    }

    // Z-inflow columns: free variables for realized total inflow per hydro.
    // col_lower = -inf, col_upper = +inf, objective = 0.0 (no direct cost).
    for h_idx in 0..layout.n_h {
        let col = layout.col_z_inflow_start + h_idx;
        col_lower[col] = f64::NEG_INFINITY;
        col_upper[col] = f64::INFINITY;
        // objective[col] = 0.0 — already zero from vec initialisation.
    }

    (col_lower, col_upper, objective)
}

/// Fill row lower/upper bounds for one stage.
///
/// Returns `(row_lower, row_upper)` vectors of length `layout.num_rows`.
pub(super) fn fill_stage_rows(
    ctx: &TemplateBuildCtx<'_>,
    stage: &Stage,
    stage_idx: usize,
    layout: &StageLayout,
) -> (Vec<f64>, Vec<f64>) {
    let mut row_lower = vec![0.0_f64; layout.num_rows];
    let mut row_upper = vec![0.0_f64; layout.num_rows];

    // Water balance rows: static RHS = ζ * (deterministic_base_h - water_withdrawal_m3s_h).
    // The withdrawal is a fixed schedule that reduces the effective inflow available
    // to the reservoir. Subtracting it from the base keeps the row bound correct for
    // the stage template; the PAR(p) noise innovation is added at solve time via patches.
    for h_idx in 0..layout.n_h {
        let row = layout.row_water_balance_start + h_idx;
        let base = if ctx.par_lp.n_stages() > 0 && ctx.par_lp.n_hydros() == layout.n_h {
            ctx.par_lp.deterministic_base(stage_idx, h_idx)
        } else {
            0.0
        };
        let withdrawal = ctx
            .bounds
            .hydro_bounds(h_idx, stage_idx)
            .water_withdrawal_m3s;
        let rhs = layout.zeta * (base - withdrawal);
        row_lower[row] = rhs;
        row_upper[row] = rhs;
    }

    // Load balance rows: static RHS = mean_mw * block_factor.
    // Block factors from `load_factors.json` scale the mean load per block
    // (e.g., heavy/medium/light blocks). Default factor is 1.0 (no scaling).
    for (b_idx, bus) in ctx.buses.iter().enumerate() {
        let mean_mw = ctx
            .load_models
            .iter()
            .find(|lm| lm.bus_id == bus.id && lm.stage_id == stage.id)
            .map_or(0.0, |lm| lm.mean_mw);
        for blk in 0..layout.n_blks {
            let factor = ctx.resolved_load_factors.factor(b_idx, stage_idx, blk);
            let row = layout.row_load_balance_start + b_idx * layout.n_blks + blk;
            let rhs = mean_mw * factor;
            row_lower[row] = rhs;
            row_upper[row] = rhs;
        }
    }

    // FPHA constraint rows: g_{h,k} - gamma_v/2*v - gamma_v/2*v_in
    //                            - gamma_q*q - gamma_s*s <= gamma_0
    // Row lower = -INF, row upper = gamma_0 (pre-scaled intercept).
    // The v_in contribution is encoded in the matrix entry on the v_in column,
    // so the static upper bound only needs the intercept term.
    let n_blks = layout.n_blks;
    for (local_idx, &h_idx) in layout.fpha_hydro_indices.iter().enumerate() {
        if let ResolvedProductionModel::Fpha { planes, .. } =
            ctx.production_models.model(h_idx, stage_idx)
        {
            let n_planes = planes.len();
            debug_assert_eq!(
                n_planes, layout.fpha_planes_per_hydro[local_idx],
                "plane count mismatch for FPHA hydro {h_idx} at stage {stage_idx}"
            );
            for blk in 0..n_blks {
                for (p_idx, plane) in planes.iter().enumerate() {
                    let row = layout.row_fpha_start
                        + local_idx * n_blks * n_planes
                        + blk * n_planes
                        + p_idx;
                    row_lower[row] = f64::NEG_INFINITY;
                    row_upper[row] = plane.intercept;
                }
            }
        }
    }

    // Evaporation constraint rows: Q_ev = k_evap0 + k_evap_v/2*(v + v_in - 2*V_ref).
    // The volume-dependent term `k_evap_v/2 * v` is added via the CSC matrix entry
    // on the outgoing-storage column, so the static row bounds only encode the
    // constant term `k_evap0`.  The constraint is an equality:
    // row_lower == row_upper == k_evap0.
    for (local_idx, &h_idx) in layout.evap_hydro_indices.iter().enumerate() {
        if let EvaporationModel::Linearized { coefficients, .. } =
            ctx.evaporation_models.model(h_idx)
        {
            debug_assert!(
                stage_idx < coefficients.len(),
                "stage index {stage_idx} out of bounds for evaporation coefficients (len = {})",
                coefficients.len()
            );
            let k_evap0 = coefficients[stage_idx].k_evap0;
            let row = layout.row_evap_start + local_idx;
            row_lower[row] = k_evap0;
            row_upper[row] = k_evap0;
        }
    }

    // Operational violation constraint rows (min/max outflow, min turbine, min generation).
    fill_operational_violation_rows(
        ctx,
        stage,
        stage_idx,
        layout,
        &mut row_lower,
        &mut row_upper,
    );

    // Z-inflow definition rows: equality constraints with RHS = base_h (m3/s).
    // The base is the deterministic PAR base inflow (before noise), NOT multiplied
    // by zeta and NOT reduced by withdrawal. The noise component (sigma * eta) is
    // added at solve time via PatchBuffer Category 5.
    for h_idx in 0..layout.n_h {
        let row = layout.row_z_inflow_start + h_idx;
        let base = if ctx.par_lp.n_stages() > 0 && ctx.par_lp.n_hydros() == layout.n_h {
            ctx.par_lp.deterministic_base(stage_idx, h_idx)
        } else {
            0.0
        };
        row_lower[row] = base;
        row_upper[row] = base;
    }

    (row_lower, row_upper)
}

/// Fill row bounds for the 4 operational violation constraint families.
///
/// Per-block formulation: one row per hydro per block. RHS is in rate units
/// (m3/s for flow families, MW for generation).
///
/// - **Min outflow** (`>=`): `row_lower = min_outflow_m3s`, `row_upper = +INF`.
/// - **Max outflow** (`<=`): `row_lower = -INF`, `row_upper = max_outflow_m3s`
///   (or `+INF` when the bound is absent, making the row non-binding).
/// - **Min turbine** (`>=`): `row_lower = min_turbined_m3s`, `row_upper = +INF`.
/// - **Min generation** (`>=`): `row_lower = min_generation_mw`, `row_upper = +INF`.
fn fill_operational_violation_rows(
    ctx: &TemplateBuildCtx<'_>,
    _stage: &Stage,
    stage_idx: usize,
    layout: &StageLayout,
    row_lower: &mut [f64],
    row_upper: &mut [f64],
) {
    // Min outflow rows (>= constraint): LHS + sigma >= min_outflow_m3s
    for h_idx in 0..layout.n_h {
        let hb = ctx.bounds.hydro_bounds(h_idx, stage_idx);
        for blk in 0..layout.n_blks {
            let row = layout.row_min_outflow_start + h_idx * layout.n_blks + blk;
            row_lower[row] = hb.min_outflow_m3s;
            row_upper[row] = f64::INFINITY;
        }
    }

    // Max outflow rows (<= constraint): LHS - sigma <= max_outflow_m3s
    for h_idx in 0..layout.n_h {
        let hb = ctx.bounds.hydro_bounds(h_idx, stage_idx);
        for blk in 0..layout.n_blks {
            let row = layout.row_max_outflow_start + h_idx * layout.n_blks + blk;
            row_lower[row] = f64::NEG_INFINITY;
            row_upper[row] = hb.max_outflow_m3s.unwrap_or(f64::INFINITY);
        }
    }

    // Min turbine flow rows (>= constraint): LHS + sigma >= min_turbined_m3s
    for h_idx in 0..layout.n_h {
        let hb = ctx.bounds.hydro_bounds(h_idx, stage_idx);
        for blk in 0..layout.n_blks {
            let row = layout.row_min_turbine_start + h_idx * layout.n_blks + blk;
            row_lower[row] = hb.min_turbined_m3s;
            row_upper[row] = f64::INFINITY;
        }
    }

    // Min generation rows (>= constraint): LHS + sigma >= min_generation_mw
    for h_idx in 0..layout.n_h {
        let hb = ctx.bounds.hydro_bounds(h_idx, stage_idx);
        for blk in 0..layout.n_blks {
            let row = layout.row_min_generation_start + h_idx * layout.n_blks + blk;
            row_lower[row] = hb.min_generation_mw;
            row_upper[row] = f64::INFINITY;
        }
    }
}

/// Build the CSC matrix entries for one stage.
///
/// Returns one `Vec<(row, value)>` per column. Entries within each column are
/// sorted by row index before return (CSC invariant).
/// Fill state-region and water-balance entries into `col_entries`.
///
/// Writes entries for storage-fixing rows, AR lag-fixing rows,
/// and the water-balance rows (outgoing/incoming storage, turbine, spillage,
/// upstream cascade, and AR lag dynamics).
pub(super) fn fill_state_and_water_entries(
    ctx: &TemplateBuildCtx<'_>,
    stage: &Stage,
    stage_idx: usize,
    layout: &StageLayout,
    col_entries: &mut [Vec<(usize, f64)>],
) {
    let idx = StageIndexer::new(ctx.n_hydros, ctx.max_par_order);
    let n_h = layout.n_h;
    let n_blks = layout.n_blks;
    let lag_order = layout.lag_order;
    let zeta = layout.zeta;
    let row_water = layout.row_water_balance_start;

    // State rows: storage-fixing (incoming storage column → row h).
    for h in 0..n_h {
        let col = idx.storage_in.start + h;
        col_entries[col].push((h, 1.0));
    }

    // State rows: lag-fixing (lag column → diagonal row).
    for lag in 0..lag_order {
        for h in 0..n_h {
            let col = idx.inflow_lags.start + lag * n_h + h;
            let row = n_h + lag * n_h + h;
            col_entries[col].push((row, 1.0));
        }
    }

    // Water balance: outgoing storage (+1), incoming storage (-1),
    // turbine/spillage (+tau), upstream turbine/spillage (-tau),
    // and AR lag dynamics (-ζ*ψ).
    for h_idx in 0..n_h {
        let hydro = &ctx.hydros[h_idx];
        let row = row_water + h_idx;
        col_entries[h_idx].push((row, 1.0));
        col_entries[idx.storage_in.start + h_idx].push((row, -1.0));
        for blk in 0..n_blks {
            let tau_h = stage.blocks[blk].duration_hours * M3S_TO_HM3;
            let col_turbine = layout.col_turbine_start + h_idx * n_blks + blk;
            col_entries[col_turbine].push((row, tau_h));
            let col_spillage = layout.col_spillage_start + h_idx * n_blks + blk;
            col_entries[col_spillage].push((row, tau_h));
            // Diversion outflow: this hydro's diversion column enters its own
            // water balance with +tau (outflow), same sign as turbine/spillage.
            let col_diversion = layout.col_diversion_start + h_idx * n_blks + blk;
            col_entries[col_diversion].push((row, tau_h));
            // Cascade inflow: upstream turbine/spillage enter with -tau.
            for &up_id in ctx.cascade.upstream(hydro.id) {
                if let Some(&u_idx) = ctx.hydro_pos.get(&up_id) {
                    col_entries[layout.col_turbine_start + u_idx * n_blks + blk]
                        .push((row, -tau_h));
                    col_entries[layout.col_spillage_start + u_idx * n_blks + blk]
                        .push((row, -tau_h));
                }
            }
            // Diversion inflow: for each hydro that diverts TO this hydro, its
            // diversion column enters this hydro's water balance with -tau.
            if let Some(sources) = ctx.diversion_upstream.get(&hydro.id) {
                for &d_idx in sources {
                    let col_div = layout.col_diversion_start + d_idx * n_blks + blk;
                    col_entries[col_div].push((row, -tau_h));
                }
            }
        }
        if ctx.par_lp.n_stages() > 0 && ctx.par_lp.n_hydros() == n_h {
            let psi = ctx.par_lp.psi_slice(stage_idx, h_idx);
            for (lag, &psi_val) in psi.iter().enumerate() {
                if psi_val != 0.0 && lag < lag_order {
                    let col = idx.inflow_lags.start + lag * n_h + h_idx;
                    col_entries[col].push((row, -zeta * psi_val));
                }
            }
        }
    }

    // Inflow non-negativity slack: sigma_inf_h enters water balance with -ζ.
    if ctx.has_penalty {
        for h_idx in 0..n_h {
            let col = layout.col_inflow_slack_start + h_idx;
            let row = row_water + h_idx;
            col_entries[col].push((row, -zeta));
        }
    }

    // Evaporation: Q_ev_h enters water balance with +ζ.
    // Evaporation is an outflow (water leaving the reservoir), so its
    // coefficient matches the turbine/spillage sign convention (positive).
    for (local_idx, &h_idx) in layout.evap_hydro_indices.iter().enumerate() {
        let col_q_ev = layout.col_evap_start + local_idx * 3;
        let row = row_water + h_idx;
        col_entries[col_q_ev].push((row, zeta));
    }

    // Under-withdrawal slack (neg): adds water back to the balance.
    // When the reservoir cannot sustain the full scheduled withdrawal, the neg slack
    // absorbs the difference, reducing the effective withdrawal in that stage.
    for h_idx in 0..n_h {
        let col = layout.col_withdrawal_neg_start + h_idx;
        let row = row_water + h_idx;
        col_entries[col].push((row, -zeta));
    }

    // Over-withdrawal slack (pos): removes additional water from the balance.
    // When the solver withdraws more than the target, the pos slack accounts for
    // the excess withdrawal at a penalty cost.
    for h_idx in 0..n_h {
        let col = layout.col_withdrawal_pos_start + h_idx;
        let row = row_water + h_idx;
        col_entries[col].push((row, zeta));
    }
}

/// Fill load-balance entries into `col_entries`.
///
/// Writes entries for hydro turbine generation, thermal generation,
/// line forward/reverse flows, and deficit/excess slacks.
///
/// For FPHA hydros the generation variable `g_{h,k}` (in `col_generation_start`)
/// enters the load balance with coefficient +1.0 instead of `rho * turbine_col`.
/// For constant-productivity hydros the original `rho * turbine_col` behavior is unchanged.
pub(super) fn fill_load_balance_entries(
    ctx: &TemplateBuildCtx<'_>,
    stage_idx: usize,
    layout: &StageLayout,
    col_entries: &mut [Vec<(usize, f64)>],
) {
    let n_blks = layout.n_blks;
    let row_load = layout.row_load_balance_start;

    // Build a quick lookup from hydro index to FPHA local index.
    // `fpha_local[h_idx]` is `Some(local_idx)` if hydro `h_idx` uses FPHA at this stage.
    let mut fpha_local: Vec<Option<usize>> = vec![None; ctx.n_hydros];
    for (local_idx, &h_idx) in layout.fpha_hydro_indices.iter().enumerate() {
        fpha_local[h_idx] = Some(local_idx);
    }

    // Hydro contribution to load balance.
    // - FPHA hydros: g_{h,k} column with coefficient +1.0
    // - Constant-productivity hydros: rho * turbine_col (unchanged)
    for (h_idx, hydro) in ctx.hydros.iter().enumerate() {
        if let Some(local_idx) = fpha_local[h_idx] {
            // FPHA: use the generation variable column.
            // The resolved model for this stage must be Fpha.
            debug_assert!(
                matches!(
                    ctx.production_models.model(h_idx, stage_idx),
                    ResolvedProductionModel::Fpha { .. }
                ),
                "FPHA local-index table inconsistent with production model for hydro {h_idx}"
            );
            if let Some(&b_idx) = ctx.bus_pos.get(&hydro.bus_id) {
                for blk in 0..n_blks {
                    let row = row_load + b_idx * n_blks + blk;
                    let col = layout.col_generation_start + local_idx * n_blks + blk;
                    col_entries[col].push((row, 1.0));
                }
            }
        } else {
            // Constant productivity: use the resolved per-stage production model,
            // which accounts for hydro_production_models.json overrides.
            let rho = match ctx.production_models.model(h_idx, stage_idx) {
                ResolvedProductionModel::ConstantProductivity { productivity } => *productivity,
                ResolvedProductionModel::Fpha { .. } => {
                    // This branch should not be reached because FPHA hydros are handled
                    // by the `if let Some(local_idx) = fpha_local[h_idx]` branch above.
                    // If we get here, the FPHA local-index table is inconsistent.
                    unreachable!(
                        "non-FPHA branch reached for FPHA resolved model at hydro {h_idx}"
                    );
                }
            };
            if let Some(&b_idx) = ctx.bus_pos.get(&hydro.bus_id) {
                for blk in 0..n_blks {
                    let row = row_load + b_idx * n_blks + blk;
                    let col = layout.col_turbine_start + h_idx * n_blks + blk;
                    col_entries[col].push((row, rho));
                }
            }
        }
    }

    // Thermal generation.
    for (t_idx, thermal) in ctx.thermals.iter().enumerate() {
        if let Some(&b_idx) = ctx.bus_pos.get(&thermal.bus_id) {
            for blk in 0..n_blks {
                let row = row_load + b_idx * n_blks + blk;
                let col = layout.col_thermal_start + t_idx * n_blks + blk;
                col_entries[col].push((row, 1.0));
            }
        }
    }

    // Line flows (+1 at target, -1 at source for forward; reversed for reverse).
    for (l_idx, line) in ctx.lines.iter().enumerate() {
        let src_idx = ctx.bus_pos.get(&line.source_bus_id).copied();
        let tgt_idx = ctx.bus_pos.get(&line.target_bus_id).copied();
        for blk in 0..n_blks {
            let col_fwd = layout.col_line_fwd_start + l_idx * n_blks + blk;
            let col_rev = layout.col_line_rev_start + l_idx * n_blks + blk;
            if let Some(tgt) = tgt_idx {
                let row = row_load + tgt * n_blks + blk;
                col_entries[col_fwd].push((row, 1.0));
                col_entries[col_rev].push((row, -1.0));
            }
            if let Some(src) = src_idx {
                let row = row_load + src * n_blks + blk;
                col_entries[col_fwd].push((row, -1.0));
                col_entries[col_rev].push((row, 1.0));
            }
        }
    }

    // Deficit (+1 for every segment) and excess (-1).
    //
    // All deficit segment columns for bus `b_idx` at block `blk` enter the same
    // load-balance row with coefficient +1.0 (total deficit = sum of segments).
    for (b_idx, bus) in ctx.buses.iter().enumerate() {
        for blk in 0..n_blks {
            let row = row_load + b_idx * n_blks + blk;
            for seg_idx in 0..bus.deficit_segments.len() {
                let col_def = layout.col_deficit_start
                    + b_idx * layout.max_deficit_segments * n_blks
                    + seg_idx * n_blks
                    + blk;
                col_entries[col_def].push((row, 1.0));
            }
            let col_exc = layout.col_excess_start + b_idx * n_blks + blk;
            col_entries[col_exc].push((row, -1.0));
        }
    }
}

/// Fill FPHA hyperplane constraint entries into `col_entries`.
///
/// For each FPHA hydro `h` at this stage, for each block `k`, for each
/// hyperplane `m`, adds matrix entries to FPHA row `r(h,k,m)`:
///
/// ```text
/// g_{h,k}  column:  +1.0              (generation variable)
/// v        column:  -gamma_v/2         (outgoing storage)
/// v_in     column:  -gamma_v/2         (incoming storage; fixed by storage-fixing row)
/// q_{h,k}  column:  -gamma_q           (turbined flow)
/// s_{h,k}  column:  -gamma_s           (spillage)
/// ```
///
/// These entries implement `g - gamma_v/2*v - gamma_v/2*v_in - gamma_q*q - gamma_s*s <= gamma_0`,
/// where `gamma_0` is already encoded in the row upper bound set by `fill_stage_rows`.
pub(super) fn fill_fpha_entries(
    ctx: &TemplateBuildCtx<'_>,
    stage_idx: usize,
    layout: &StageLayout,
    col_entries: &mut [Vec<(usize, f64)>],
) {
    let idx = StageIndexer::new(ctx.n_hydros, ctx.max_par_order);
    let n_blks = layout.n_blks;

    for (local_idx, &h_idx) in layout.fpha_hydro_indices.iter().enumerate() {
        let model = ctx.production_models.model(h_idx, stage_idx);
        let planes = match model {
            ResolvedProductionModel::Fpha { planes, .. } => planes,
            ResolvedProductionModel::ConstantProductivity { .. } => {
                // Should never happen: fpha_hydro_indices only contains FPHA hydros.
                debug_assert!(
                    false,
                    "fpha_hydro_indices contains hydro {h_idx} but model is ConstantProductivity"
                );
                continue;
            }
        };
        let n_planes = planes.len();

        for blk in 0..n_blks {
            // Column indices for this hydro/block.
            let col_v = h_idx; // outgoing storage column
            let col_v_in = idx.storage_in.start + h_idx; // incoming storage column
            let col_q = layout.col_turbine_start + h_idx * n_blks + blk;
            let col_s = layout.col_spillage_start + h_idx * n_blks + blk;
            let col_g = layout.col_generation_start + local_idx * n_blks + blk;

            for (p_idx, plane) in planes.iter().enumerate() {
                let row =
                    layout.row_fpha_start + local_idx * n_blks * n_planes + blk * n_planes + p_idx;

                // g_{h,k} column: +1.0
                col_entries[col_g].push((row, 1.0));
                // v (outgoing storage): -gamma_v/2
                col_entries[col_v].push((row, -plane.gamma_v / 2.0));
                // v_in (incoming storage, fixed by storage-fixing row): -gamma_v/2
                col_entries[col_v_in].push((row, -plane.gamma_v / 2.0));
                // q_{h,k} (turbine): -gamma_q
                col_entries[col_q].push((row, -plane.gamma_q));
                // s_{h,k} (spillage): -gamma_s
                col_entries[col_s].push((row, -plane.gamma_s));
            }
        }
    }
}

/// Fill CSC matrix entries for the evaporation constraint rows.
///
/// For each evaporation hydro `h` at local position `local_idx`, the equality row
/// `row_evap_start + local_idx` encodes:
///
/// ```text
/// Q_ev_h  column:  +1.0
/// v_h     column:  -k_evap_v / 2      (outgoing storage)
/// v_in_h  column:  -k_evap_v / 2      (incoming storage; fixed by storage-fixing row)
/// f_plus  column:  +1.0
/// f_minus column:  -1.0
/// ```
///
/// These entries implement `Q_ev - k_evap_v/2*v - k_evap_v/2*v_in + f_plus - f_minus = k_evap0`,
/// where `k_evap0` is already encoded in the row bounds set by `fill_stage_rows`.
/// When `v_in` is fixed to value `V`, the effective RHS becomes `k_evap0 + k_evap_v/2 * V`,
/// which matches the linearized evaporation at the average volume `(v + V) / 2`.
pub(super) fn fill_evaporation_entries(
    ctx: &TemplateBuildCtx<'_>,
    stage_idx: usize,
    layout: &StageLayout,
    col_entries: &mut [Vec<(usize, f64)>],
) {
    let idx = StageIndexer::new(ctx.n_hydros, ctx.max_par_order);

    for (local_idx, &h_idx) in layout.evap_hydro_indices.iter().enumerate() {
        let coeff = match ctx.evaporation_models.model(h_idx) {
            EvaporationModel::Linearized { coefficients, .. } => {
                debug_assert!(
                    stage_idx < coefficients.len(),
                    "evap_hydro_indices contains hydro {h_idx} but coefficients length {} \
                     is less than stage_idx {}",
                    coefficients.len(),
                    stage_idx
                );
                match coefficients.get(stage_idx) {
                    Some(c) => *c,
                    None => continue,
                }
            }
            EvaporationModel::None => {
                // Should never happen: evap_hydro_indices only contains linearized hydros.
                debug_assert!(
                    false,
                    "evap_hydro_indices contains hydro {h_idx} but model is None"
                );
                continue;
            }
        };

        let col_q_ev = layout.col_evap_start + local_idx * 3;
        let col_f_plus = layout.col_evap_start + local_idx * 3 + 1;
        let col_f_minus = layout.col_evap_start + local_idx * 3 + 2;
        let col_v = h_idx; // outgoing storage column
        let col_v_in = idx.storage_in.start + h_idx; // incoming storage column

        let row = layout.row_evap_start + local_idx;

        // Q_ev_h: +1.0
        col_entries[col_q_ev].push((row, 1.0));
        // v_h (outgoing storage): -k_evap_v / 2
        col_entries[col_v].push((row, -coeff.k_evap_v / 2.0));
        // v_in_h (incoming storage, fixed by storage-fixing row): -k_evap_v / 2
        col_entries[col_v_in].push((row, -coeff.k_evap_v / 2.0));
        // f_evap_plus: +1.0
        col_entries[col_f_plus].push((row, 1.0));
        // f_evap_minus: -1.0
        col_entries[col_f_minus].push((row, -1.0));
    }
}

/// Fill CSC matrix entries, row bounds, and slack column data for all active
/// generic constraint rows at this stage.
///
/// For each active `(constraint, block)` pair recorded in
/// `layout.generic_constraint_rows`:
///
/// 1. Sets `row_lower` / `row_upper` for the generic constraint row according
///    to the constraint sense:
///    - `<=`: `row_lower = -INF`, `row_upper = bound`
///    - `>=`: `row_lower = bound`, `row_upper = +INF`
///    - `==`: `row_lower = bound`, `row_upper = bound`
///
/// 2. Iterates over the constraint expression terms, calls
///    `resolve_variable_ref` for each [`LinearTerm`], and pushes
///    `(row_index, coefficient * multiplier)` entries into `col_entries`.
///
/// 3. When `slack.enabled = true`, sets slack column bounds to `[0, +INF)` and
///    objective to `penalty * block_hours`:
///    - `<=`: one slack column `s_g` with CSC entry `(row, -1.0)`.
///    - `>=`: one slack column `s_g` with CSC entry `(row, +1.0)`.
///    - `==`: two slack columns `s_g_plus` and `s_g_minus` with CSC entries
///      `(row, +1.0)` and `(row, -1.0)` respectively.
///
/// Unknown entity IDs in variable refs produce zero contributions (the empty
/// vec returned by `resolve_variable_ref` is skipped), which is the
/// defense-in-depth fallback for referential validation gaps.
/// Mutable LP matrix buffers for stage template construction.
///
/// Groups the column and row arrays that are filled during template building.
pub(super) struct LpMatrixBuffers<'a> {
    /// CSC column entries (column index -> list of (row, coefficient)).
    pub(super) col_entries: &'a mut [Vec<(usize, f64)>],
    /// Column lower bounds (currently unused for generic constraints).
    pub(super) _col_lower: &'a mut [f64],
    /// Column upper bounds.
    pub(super) col_upper: &'a mut [f64],
    /// Objective function coefficients.
    pub(super) objective: &'a mut [f64],
    /// Row lower bounds.
    pub(super) row_lower: &'a mut [f64],
    /// Row upper bounds.
    pub(super) row_upper: &'a mut [f64],
}

pub(super) fn fill_generic_constraint_entries(
    ctx: &TemplateBuildCtx<'_>,
    stage: &Stage,
    stage_idx: usize,
    layout: &StageLayout,
    buffers: &mut LpMatrixBuffers<'_>,
) {
    let col_entries = &mut *buffers.col_entries;
    let col_upper = &mut *buffers.col_upper;
    let objective = &mut *buffers.objective;
    let row_lower = &mut *buffers.row_lower;
    let row_upper = &mut *buffers.row_upper;
    if layout.n_generic_rows == 0 {
        return;
    }

    // Build a full StageIndexer so that resolve_variable_ref can map any
    // VariableRef to the correct column index for this stage.
    let indexer = crate::indexer::StageIndexer::with_equipment_and_evaporation(
        &crate::indexer::EquipmentCounts {
            hydro_count: ctx.n_hydros,
            max_par_order: ctx.max_par_order,
            n_thermals: ctx.n_thermals,
            n_lines: ctx.n_lines,
            n_buses: ctx.n_buses,
            n_blks: layout.n_blks,
            has_inflow_penalty: ctx.has_penalty,
            max_deficit_segments: layout.max_deficit_segments,
        },
        &crate::indexer::FphaColumnLayout {
            hydro_indices: layout.fpha_hydro_indices.clone(),
            planes_per_hydro: layout.fpha_planes_per_hydro.clone(),
        },
        &crate::indexer::EvapConfig {
            hydro_indices: layout.evap_hydro_indices.clone(),
        },
    );

    let positions = crate::generic_constraints::EntityPositionMaps {
        hydro: &ctx.hydro_pos,
        thermal: &ctx.thermal_pos,
        bus: &ctx.bus_pos,
        line: &ctx.line_pos,
    };

    for (entry_idx, entry) in layout.generic_constraint_rows.iter().enumerate() {
        let row = layout.row_generic_start + entry_idx;
        let constraint = &ctx.generic_constraints[entry.constraint_idx];
        let block_hours = stage.blocks[entry.block_idx].duration_hours;

        // 1. Set row bounds from sense and RHS bound value.
        match entry.sense {
            ConstraintSense::LessEqual => {
                row_lower[row] = f64::NEG_INFINITY;
                row_upper[row] = entry.bound;
            }
            ConstraintSense::GreaterEqual => {
                row_lower[row] = entry.bound;
                row_upper[row] = f64::INFINITY;
            }
            ConstraintSense::Equal => {
                row_lower[row] = entry.bound;
                row_upper[row] = entry.bound;
            }
        }

        // 2. Fill CSC matrix entries for each expression term.
        for term in &constraint.expression.terms {
            let pairs = resolve_variable_ref(
                &term.variable,
                entry.block_idx,
                layout.n_blks,
                stage_idx,
                &indexer,
                ctx.production_models,
                &positions,
            );
            for (col, multiplier) in pairs {
                col_entries[col].push((row, term.coefficient * multiplier));
            }
        }

        // 3. Set slack column bounds and CSC entries when slack is enabled.
        if let Some(plus_col) = entry.slack_plus_col {
            let penalty = constraint.slack.penalty.unwrap_or(0.0);
            let obj_coeff = penalty * block_hours;

            // plus slack: [0, +INF), penalised in objective.
            // col_lower is already 0.0 from vec initialisation.
            col_upper[plus_col] = f64::INFINITY;
            objective[plus_col] = obj_coeff;

            // CSC entry for plus slack depends on sense.
            match entry.sense {
                ConstraintSense::LessEqual => {
                    // LHS - s_g <= bound  →  slack enters with -1.0
                    col_entries[plus_col].push((row, -1.0));
                }
                ConstraintSense::GreaterEqual => {
                    // LHS + s_g >= bound  →  slack enters with +1.0
                    col_entries[plus_col].push((row, 1.0));
                }
                ConstraintSense::Equal => {
                    // LHS + s_g_plus - s_g_minus == bound  →  plus slack with +1.0
                    col_entries[plus_col].push((row, 1.0));
                }
            }

            // minus slack: only for equality constraints.
            if let Some(minus_col) = entry.slack_minus_col {
                // col_lower is already 0.0 from vec initialisation.
                col_upper[minus_col] = f64::INFINITY;
                objective[minus_col] = obj_coeff;
                // LHS + s_g_plus - s_g_minus == bound  →  minus slack with -1.0
                col_entries[minus_col].push((row, -1.0));
            }
        }
    }
}

/// Fill NCS generation entries into the load balance constraint rows.
///
/// For each active NCS `r` at block `k`, injects `+1.0` at the load balance
/// row of the connected bus, identical to thermal generation injection.
pub(super) fn fill_ncs_load_balance_entries(
    ctx: &TemplateBuildCtx<'_>,
    layout: &StageLayout,
    col_entries: &mut [Vec<(usize, f64)>],
) {
    for (ncs_local, &ncs_sys_idx) in layout.active_ncs_indices.iter().enumerate() {
        let ncs = &ctx.non_controllable_sources[ncs_sys_idx];
        let Some(&bus_idx) = ctx.bus_pos.get(&ncs.bus_id) else {
            // Unknown bus — should not happen with valid data, but defensive skip.
            continue;
        };
        for blk in 0..layout.n_blks {
            let col = layout.col_ncs_start + ncs_local * layout.n_blks + blk;
            let row = layout.row_load_balance_start + bus_idx * layout.n_blks + blk;
            col_entries[col].push((row, 1.0));
        }
    }
}

/// Fill z-inflow definition constraint entries into `col_entries`.
///
/// For each hydro h, the z-inflow constraint is:
///   `z_h - sum_l[psi_l * lag_in[h,l]] = base_h + sigma_h * eta_h`
///
/// Matrix entries:
/// - Column `z_h`: coefficient `+1.0` in row `row_z_inflow_start + h`
/// - For each lag l with nonzero `psi_l`: column `inflow_lags.start + lag * n_h + h`
///   gets coefficient `-psi_l` in row `row_z_inflow_start + h`
///
/// Note: the lag column layout matches the LP builder convention (lag-major):
/// the column at `inflow_lags.start + lag * n_h + h` stores lag `l` of hydro `h`.
pub(super) fn fill_z_inflow_entries(
    ctx: &TemplateBuildCtx<'_>,
    stage_idx: usize,
    layout: &StageLayout,
    col_entries: &mut [Vec<(usize, f64)>],
) {
    let n_h = layout.n_h;
    let lag_order = layout.lag_order;
    let idx = StageIndexer::new(n_h, lag_order);

    for h_idx in 0..n_h {
        let row = layout.row_z_inflow_start + h_idx;

        // z_h column: coefficient +1.0
        let col_z = layout.col_z_inflow_start + h_idx;
        col_entries[col_z].push((row, 1.0));

        // Lag columns: coefficient -psi_l for each nonzero psi.
        // Uses lag-major layout (lag * n_h + h) matching the water-balance
        // AR dynamics entries in fill_state_and_water_entries.
        if ctx.par_lp.n_stages() > 0 && ctx.par_lp.n_hydros() == n_h {
            let psi = ctx.par_lp.psi_slice(stage_idx, h_idx);
            for (lag, &psi_val) in psi.iter().enumerate() {
                if psi_val != 0.0 && lag < lag_order {
                    let col = idx.inflow_lags.start + lag * n_h + h_idx;
                    col_entries[col].push((row, -psi_val));
                }
            }
        }
    }
}

/// Fill CSC matrix entries for the 4 operational violation constraint families.
///
/// Each constraint links decision variables (turbine, spillage, diversion, generation)
/// to their respective slack columns via the constraint rows allocated in
/// [`StageLayout`].
///
/// - **Min outflow** (`>=`): `sum_blk[tau * (q + s + d)] + sigma_below >= RHS`
/// - **Max outflow** (`<=`): `sum_blk[tau * (q + s + d)] - sigma_above <= RHS`
/// - **Min turbine** (`>=`): `sum_blk[tau * q] + sigma_below >= RHS`
/// - **Min generation** (`>=`): `sum_blk[coeff * var * hours] + sigma_below >= RHS`
///   where `coeff * var` is `rho * q` for constant-productivity hydros or `g` for FPHA.
pub(super) fn fill_operational_violation_entries(
    ctx: &TemplateBuildCtx<'_>,
    _stage: &Stage,
    stage_idx: usize,
    layout: &StageLayout,
    col_entries: &mut [Vec<(usize, f64)>],
) {
    let n_blks = layout.n_blks;

    // Build FPHA local-index lookup (same pattern as fill_load_balance_entries).
    let mut fpha_local: Vec<Option<usize>> = vec![None; ctx.n_hydros];
    for (local_idx, &h_idx) in layout.fpha_hydro_indices.iter().enumerate() {
        fpha_local[h_idx] = Some(local_idx);
    }

    for (h_idx, fpha_local_entry) in fpha_local.iter().enumerate() {
        // ── Min outflow (per block): q + s + d + sigma >= min_outflow_m3s ───
        for blk in 0..n_blks {
            let row = layout.row_min_outflow_start + h_idx * n_blks + blk;
            let col_q = layout.col_turbine_start + h_idx * n_blks + blk;
            col_entries[col_q].push((row, 1.0));
            let col_s = layout.col_spillage_start + h_idx * n_blks + blk;
            col_entries[col_s].push((row, 1.0));
            let col_d = layout.col_diversion_start + h_idx * n_blks + blk;
            col_entries[col_d].push((row, 1.0));
            let col_slack = layout.col_outflow_below_start + h_idx * n_blks + blk;
            col_entries[col_slack].push((row, 1.0));
        }

        // ── Max outflow (per block): q + s + d - sigma <= max_outflow_m3s ───
        for blk in 0..n_blks {
            let row = layout.row_max_outflow_start + h_idx * n_blks + blk;
            let col_q = layout.col_turbine_start + h_idx * n_blks + blk;
            col_entries[col_q].push((row, 1.0));
            let col_s = layout.col_spillage_start + h_idx * n_blks + blk;
            col_entries[col_s].push((row, 1.0));
            let col_d = layout.col_diversion_start + h_idx * n_blks + blk;
            col_entries[col_d].push((row, 1.0));
            let col_slack = layout.col_outflow_above_start + h_idx * n_blks + blk;
            col_entries[col_slack].push((row, -1.0));
        }

        // ── Min turbine flow (per block): q + sigma >= min_turbined_m3s ─────
        for blk in 0..n_blks {
            let row = layout.row_min_turbine_start + h_idx * n_blks + blk;
            let col_q = layout.col_turbine_start + h_idx * n_blks + blk;
            col_entries[col_q].push((row, 1.0));
            let col_slack = layout.col_turbine_below_start + h_idx * n_blks + blk;
            col_entries[col_slack].push((row, 1.0));
        }

        // ── Min generation (per block): g + sigma >= min_generation_mw ──────
        if let Some(&local_fpha_idx) = fpha_local_entry.as_ref() {
            // FPHA: generation variable g_{h,blk} (already in MW).
            for blk in 0..n_blks {
                let row = layout.row_min_generation_start + h_idx * n_blks + blk;
                let col_g = layout.col_generation_start + local_fpha_idx * n_blks + blk;
                col_entries[col_g].push((row, 1.0));
                let col_slack = layout.col_generation_below_start + h_idx * n_blks + blk;
                col_entries[col_slack].push((row, 1.0));
            }
        } else {
            // Constant productivity: gen_k = rho * q_k (MW).
            let rho = match &ctx.hydros[h_idx].generation_model {
                HydroGenerationModel::ConstantProductivity {
                    productivity_mw_per_m3s,
                }
                | HydroGenerationModel::LinearizedHead {
                    productivity_mw_per_m3s,
                } => *productivity_mw_per_m3s,
                HydroGenerationModel::Fpha => {
                    // Entity model is Fpha but resolved model at this stage is
                    // ConstantProductivity (fallback). Extract rho from the resolved model.
                    if let ResolvedProductionModel::ConstantProductivity { productivity } =
                        ctx.production_models.model(h_idx, stage_idx)
                    {
                        *productivity
                    } else {
                        debug_assert!(
                            false,
                            "Fpha entity model with non-Fpha resolved model and no local index for hydro {h_idx}"
                        );
                        0.0
                    }
                }
            };
            for blk in 0..n_blks {
                let row = layout.row_min_generation_start + h_idx * n_blks + blk;
                let col_q = layout.col_turbine_start + h_idx * n_blks + blk;
                col_entries[col_q].push((row, rho));
                let col_slack = layout.col_generation_below_start + h_idx * n_blks + blk;
                col_entries[col_slack].push((row, 1.0));
            }
        }
    }
}

/// Build the unsorted CSC matrix entries for one stage.
///
/// Returns one `Vec<(row, value)>` per column. Entries are in insertion
/// order; the caller is responsible for sorting by row index before
/// assembling the final CSC arrays (see [`build_single_stage_template`]).
pub(super) fn build_stage_matrix_entries(
    ctx: &TemplateBuildCtx<'_>,
    stage: &Stage,
    stage_idx: usize,
    layout: &StageLayout,
) -> Vec<Vec<(usize, f64)>> {
    let mut col_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); layout.num_cols];

    fill_state_and_water_entries(ctx, stage, stage_idx, layout, &mut col_entries);
    fill_load_balance_entries(ctx, stage_idx, layout, &mut col_entries);
    fill_ncs_load_balance_entries(ctx, layout, &mut col_entries);
    fill_fpha_entries(ctx, stage_idx, layout, &mut col_entries);
    fill_evaporation_entries(ctx, stage_idx, layout, &mut col_entries);
    fill_z_inflow_entries(ctx, stage_idx, layout, &mut col_entries);
    fill_operational_violation_entries(ctx, stage, stage_idx, layout, &mut col_entries);

    col_entries
}

/// Assemble CSC arrays from per-column entry lists.
///
/// Returns `(col_starts, row_indices, values)` in the format required by
/// `SolverInterface::load_model`.
pub(super) fn assemble_csc(col_entries: &[Vec<(usize, f64)>]) -> (Vec<i32>, Vec<i32>, Vec<f64>) {
    let num_cols = col_entries.len();
    let total_nz: usize = col_entries.iter().map(Vec::len).sum();
    let mut col_starts = Vec::with_capacity(num_cols + 1);
    let mut row_indices = Vec::with_capacity(total_nz);
    let mut values = Vec::with_capacity(total_nz);

    let mut offset: i32 = 0;
    for entries in col_entries {
        col_starts.push(offset);
        for &(row, val) in entries {
            #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
            row_indices.push(row as i32);
            values.push(val);
        }
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        {
            offset += entries.len() as i32;
        }
    }
    col_starts.push(offset);

    (col_starts, row_indices, values)
}
