use cobre_core::commissioning::Phase;
use cobre_core::{BlockMode, Stage};

use crate::hydro_models::EvaporationModel;
use crate::indexer::{
    BlockIdx, BusSys, EvapLocal, FillingTargetLocal, FloorLocal, HydroCell, HydroSys,
};

use super::fpha_cursor::for_each_fpha_plane;
use super::hydro_state::{
    GroupBoundLookup, cell_min_generation, cell_min_turbined, hydro_phase,
    resolve_shortcircuit_target,
};
use super::layout::{StageLayout, TemplateBuildCtx, position_table_row};

/// Fill row lower/upper bounds for one stage.
///
/// Returns `(row_lower, row_upper)` vectors of length `layout.rows.num_rows`.
pub(super) fn fill_stage_rows(
    ctx: &TemplateBuildCtx<'_>,
    stage: &Stage,
    stage_idx: usize,
    layout: &StageLayout,
) -> (Vec<f64>, Vec<f64>) {
    let mut row_lower = vec![0.0_f64; layout.rows.num_rows];
    let mut row_upper = vec![0.0_f64; layout.rows.num_rows];

    fill_water_balance_rows(
        ctx,
        stage,
        stage_idx,
        layout,
        &mut row_lower,
        &mut row_upper,
    );
    fill_transit_bucket_definition_rows(layout, &mut row_lower, &mut row_upper);
    fill_filling_target_rows(ctx, stage.id, layout, &mut row_lower, &mut row_upper);
    fill_filled_min_storage_floor_rows(ctx, stage_idx, layout, &mut row_lower, &mut row_upper);
    fill_load_balance_rows(
        ctx,
        stage,
        stage_idx,
        layout,
        &mut row_lower,
        &mut row_upper,
    );
    fill_fpha_rows(ctx, stage_idx, layout, &mut row_lower, &mut row_upper);
    fill_evaporation_rows(ctx, stage_idx, layout, &mut row_lower, &mut row_upper);
    fill_operational_violation_rows(ctx, stage_idx, layout, &mut row_lower, &mut row_upper);
    fill_anticipated_fishing_rows(layout, &mut row_lower, &mut row_upper);
    fill_anticipated_state_out_def_rows(layout, &mut row_lower, &mut row_upper);
    fill_anticipated_slot_definition_rows(layout, &mut row_lower, &mut row_upper);
    fill_z_inflow_rows(ctx, stage_idx, layout, &mut row_lower, &mut row_upper);

    (row_lower, row_upper)
}

/// Fill water-balance row bounds: static RHS = `−(ζ · water_withdrawal_m3s_h)`.
/// The realized inflow (deterministic base + noise) enters through the water
/// row's own `z_h` coupling entry (`entries::push_z_inflow_coupling`),
/// never the RHS.
///
/// A `PreFilling` hydro's row is the frozen identity `v_h − v_h_in = 0` (matrix
/// entries by [`super::entries::fill_state_and_water_entries`]), so its RHS is
/// `0`: its own row carries no `z_h` coupling (the coupling routes to the
/// short-circuit target instead), and its withdrawal DEMAND transfers to that
/// target's RHS below.
///
/// In `BlockMode::Chronological` the single per-hydro RHS splits into `K` per-block
/// row bounds `−(τ_k·withdrawal)` (block-major, mirroring the entries side);
/// summing them recovers the parallel `−(ζ·withdrawal)` since `Σ_k τ_k = ζ`.
fn fill_water_balance_rows(
    ctx: &TemplateBuildCtx<'_>,
    stage: &Stage,
    stage_idx: usize,
    layout: &StageLayout,
    row_lower: &mut [f64],
    row_upper: &mut [f64],
) {
    match stage.block_mode {
        BlockMode::Parallel => {
            fill_parallel_water_rows(ctx, stage, stage_idx, layout, row_lower, row_upper);
        }
        BlockMode::Chronological => {
            fill_chronological_water_rows(ctx, stage, stage_idx, layout, row_lower, row_upper);
        }
    }
}

fn fill_parallel_water_rows(
    ctx: &TemplateBuildCtx<'_>,
    stage: &Stage,
    stage_idx: usize,
    layout: &StageLayout,
    row_lower: &mut [f64],
    row_upper: &mut [f64],
) {
    for h_idx in 0..layout.state.hydro_count {
        let row = layout
            .geometry
            .water_balance_row(HydroSys::new(h_idx), BlockIdx::new(0));
        if matches!(hydro_phase(&ctx.hydros[h_idx], stage.id), Phase::PreFilling) {
            row_lower[row] = 0.0;
            row_upper[row] = 0.0;
            continue;
        }
        let withdrawal = ctx
            .resolved
            .bounds
            .hydro_bounds(h_idx, stage_idx)
            .water_withdrawal_m3s;
        let rhs = -(layout.clock.zeta() * withdrawal);
        row_lower[row] = rhs;
        row_upper[row] = rhs;
    }

    // Transfer each PreFilling hydro's withdrawal DEMAND (`ζ·withdrawal_h`) to its
    // short-circuit target `d`'s RHS, absorbed by `d`'s withdrawal slacks. `d` MUST
    // be the SAME target `fill_prefilling_shortcircuit` routed the matrix terms to,
    // so the RHS transfer and the matrix coupling agree; routing onto the immediate
    // `downstream(h)` would land it on a PreFilling downstream's frozen-identity RHS
    // (which must stay `0`). A second pass, since `d` may be filled before or after
    // `h` in index order; sink case transfers nothing.
    for h_idx in 0..layout.state.hydro_count {
        let Some(d_idx) =
            resolve_shortcircuit_target(ctx.hydros, ctx.cascade, ctx.positions, stage.id, h_idx)
        else {
            continue;
        };
        let withdrawal_h = ctx
            .resolved
            .bounds
            .hydro_bounds(h_idx, stage_idx)
            .water_withdrawal_m3s;
        let delta = layout.clock.zeta() * withdrawal_h;
        let row_d = layout
            .geometry
            .water_balance_row(HydroSys::new(d_idx), BlockIdx::new(0));
        row_lower[row_d] -= delta;
        row_upper[row_d] -= delta;
    }
}

/// Per-block water-balance RHS for chronological mode: each Operating/Filling hydro
/// gets `K` rows `−(τ_k·withdrawal)` (block-major `row_water + h·K + (k−1)`), with
/// `τ_k` replacing `ζ` so `Σ_k` recovers the parallel total. A `PreFilling`
/// hydro gets `K` frozen-identity rows with RHS `0` (block-major), and its
/// withdrawal transfers per block (`−τ_k·withdrawal_h`) to the short-circuit
/// target's block rows, mirroring the entries side.
fn fill_chronological_water_rows(
    ctx: &TemplateBuildCtx<'_>,
    stage: &Stage,
    stage_idx: usize,
    layout: &StageLayout,
    row_lower: &mut [f64],
    row_upper: &mut [f64],
) {
    let n_blks = layout.clock.n_blks();
    for h_idx in 0..layout.state.hydro_count {
        if matches!(hydro_phase(&ctx.hydros[h_idx], stage.id), Phase::PreFilling) {
            for blk in 0..n_blks {
                let row = layout
                    .geometry
                    .water_balance_row(HydroSys::new(h_idx), BlockIdx::new(blk));
                row_lower[row] = 0.0;
                row_upper[row] = 0.0;
            }
            continue;
        }
        let withdrawal = ctx
            .resolved
            .bounds
            .hydro_bounds(h_idx, stage_idx)
            .water_withdrawal_m3s;
        for blk in 0..n_blks {
            let row = layout
                .geometry
                .water_balance_row(HydroSys::new(h_idx), BlockIdx::new(blk));
            let tau_k = layout.clock.tau(BlockIdx::new(blk));
            let rhs = -(tau_k * withdrawal);
            row_lower[row] = rhs;
            row_upper[row] = rhs;
        }
    }

    for h_idx in 0..layout.state.hydro_count {
        let Some(d_idx) =
            resolve_shortcircuit_target(ctx.hydros, ctx.cascade, ctx.positions, stage.id, h_idx)
        else {
            continue;
        };
        let withdrawal_h = ctx
            .resolved
            .bounds
            .hydro_bounds(h_idx, stage_idx)
            .water_withdrawal_m3s;
        for blk in 0..n_blks {
            let tau_k = layout.clock.tau(BlockIdx::new(blk));
            let row_d = layout
                .geometry
                .water_balance_row(HydroSys::new(d_idx), BlockIdx::new(blk));
            let delta = tau_k * withdrawal_h;
            row_lower[row_d] -= delta;
            row_upper[row_d] -= delta;
        }
    }
}

/// Fill the travel-time bucket-definition equality row bounds (`0 == 0`), one
/// row per (plant, lag) bucket REACHABLE at this stage
/// (`layout.rows.transit_bucket_row_pos`, sparse like
/// [`fill_anticipated_state_out_def_rows`]'s `active_pos` offset, not
/// [`fill_anticipated_fishing_rows`]'s always-active dense one) — a lag beyond
/// this stage's horizon-reachable cap gets no row. Empty when
/// `layout.state.n_buckets == 0`.
fn fill_transit_bucket_definition_rows(
    layout: &StageLayout,
    row_lower: &mut [f64],
    row_upper: &mut [f64],
) {
    fill_zero_equality_rows(
        layout.rows.transit_bucket_definition.start,
        &layout.rows.transit_bucket_row_pos,
        row_lower,
        row_upper,
    );
}

/// Fill the soft filling-target row bounds (`v_h + σ_fill ≥ V_target[t]`, in hm³):
/// `row_lower = V_target[t]`, `row_upper = +∞`. LHS coefficients are emitted by
/// `entries::fill_filling_target_entries`.
///
/// **Contract — the RHS is the per-stage `V_target[t]` (backward-anchored), NOT
/// `min_storage` at every stage.** `V_target[t]` is the precomputed trajectory
/// folded backward from the dead volume in setup's `build_filling_v_target`
/// precompute, so only the LAST Filling stage's floor equals `min_storage_hm3`
/// and earlier floors are strictly lower. Writing `min_storage_hm3` at every
/// Filling stage would demand the full dead volume from the FIRST Filling
/// stage — an over-strict floor the soft slack absorbs at cost every stage.
/// A per-stage helper cannot see other stages' ζ·rate, so the trajectory MUST
/// come from the precompute.
///
/// SOFT `≥` (the `σ_fill` slack relaxes it), never a hard column bound on `v_h`, so
/// a hydro that fills short keeps a feasible LP.
fn fill_filling_target_rows(
    ctx: &TemplateBuildCtx<'_>,
    stage_id: i32,
    layout: &StageLayout,
    row_lower: &mut [f64],
    row_upper: &mut [f64],
) {
    for (local_idx, &h) in layout
        .geometry
        .filling_target_hydro_indices
        .iter()
        .enumerate()
    {
        // Fail loud on a lookup miss rather than writing a `0.0` floor, which would
        // make `v + σ_fill ≥ 0` trivially true and silently neutralize the
        // constraint. Membership and the `build_filling_v_target` precompute share
        // the Filling-phase predicate, so a miss is a construction bug between them.
        let Some(v_target) = ctx.filling_v_target.get(&(h.get(), stage_id)).copied() else {
            unreachable!(
                "no V_target for filling hydro {} at stage id {stage_id}: \
                 filling_target membership and the V_target precompute disagree",
                h.get()
            );
        };
        let row = layout.filling_target_row(FillingTargetLocal::new(local_idx));
        row_lower[row] = v_target;
        row_upper[row] = f64::INFINITY;
    }
}

/// Fill the soft operating-floor row bounds (`v_h + σ^{v-} ≥ min_storage_hm3`, in
/// hm³): `row_lower = min_storage_hm3`, `row_upper = +∞`. LHS coefficients are
/// emitted by `entries::fill_filled_min_storage_floor_entries`.
///
/// `min_storage_hm3` is the RESOLVED per-stage dead volume
/// (`hydro_bounds(h_idx, stage_idx).min_storage_hm3`); reading the raw
/// `Hydro.min_storage_hm3` entity field instead would drop any per-stage override.
///
/// SOFT `≥`, never a hard column bound: `fill_storage_columns` relaxes a filling
/// hydro's hard floor to `0` so a reservoir that finished filling short keeps a
/// feasible LP, with `storage_violation_below_cost` driving `σ^{v-} → 0`.
///
/// DISTINCT from `fill_filling_target_rows` (`σ_fill`): same `≥` shape but a
/// non-overlapping stage scope (Operating vs Filling), a different RHS
/// (`min_storage` here vs the per-stage `V_target[t]` there), and a different slack
/// column and cost.
fn fill_filled_min_storage_floor_rows(
    ctx: &TemplateBuildCtx<'_>,
    stage_idx: usize,
    layout: &StageLayout,
    row_lower: &mut [f64],
    row_upper: &mut [f64],
) {
    for (local_idx, &h) in layout
        .geometry
        .filled_min_storage_floor_hydro_indices
        .iter()
        .enumerate()
    {
        let min_storage = ctx
            .resolved
            .bounds
            .hydro_bounds(h.get(), stage_idx)
            .min_storage_hm3;
        let row = layout.filled_min_storage_floor_row(FloorLocal::new(local_idx));
        row_lower[row] = min_storage;
        row_upper[row] = f64::INFINITY;
    }
}

/// Fill load-balance row bounds: static RHS = `mean_mw · block_factor` (the
/// per-block load scaling from `load_factors.json`) for a bus whose load is
/// deterministic; `0` for a load-noise member, patched at solve time.
fn fill_load_balance_rows(
    ctx: &TemplateBuildCtx<'_>,
    stage: &Stage,
    stage_idx: usize,
    layout: &StageLayout,
    row_lower: &mut [f64],
    row_upper: &mut [f64],
) {
    for (b_idx, bus) in ctx.buses.iter().enumerate() {
        let mean_mw = ctx
            .load_models
            .iter()
            .find(|lm| lm.bus_id == bus.id && lm.stage_id == stage.id)
            .map_or(0.0, |lm| lm.mean_mw);
        for blk in 0..layout.clock.n_blks() {
            let factor = ctx
                .resolved
                .resolved_load_factors
                .factor(b_idx, stage_idx, blk);
            let row = layout
                .geometry
                .load_balance_row(BusSys::new(b_idx), BlockIdx::new(blk));
            let rhs = mean_mw * factor;
            row_lower[row] = rhs;
            row_upper[row] = rhs;
        }
    }
}

/// Fill FPHA hyperplane row bounds: `row_lower = -INF`, `row_upper = σ_c * gamma_0`
/// (the pre-scaled `intercept`, apportioned by the cell's turbine-capacity share —
/// see [`super::entries::fill_fpha_entries`]). The `v`/`v_in`/`q`/`s` contributions
/// live in the matrix entries, so the upper bound carries only the apportioned
/// intercept. Driven by [`for_each_fpha_plane`] so these bounds and the matrix
/// coefficients share one row cursor.
fn fill_fpha_rows(
    ctx: &TemplateBuildCtx<'_>,
    stage_idx: usize,
    layout: &StageLayout,
    row_lower: &mut [f64],
    row_upper: &mut [f64],
) {
    for_each_fpha_plane(ctx, stage_idx, layout, |visit, plane| {
        let sigma_c = ctx.hydro_cell_index.share_of(visit.cell);
        row_lower[visit.row] = f64::NEG_INFINITY;
        row_upper[visit.row] = sigma_c * plane.intercept;
    });
}

/// Fill evaporation row bounds: equality `row_lower == row_upper == intercept_m3s`,
/// one row per `(evap hydro, slot)`, addressed by [`StageLayout::evap_row`] in
/// lockstep with the entries side. The volume-dependent term lives in the matrix
/// entries ([`super::entries::fill_evaporation_entries`]), so the row bounds
/// encode only the constant intercept, replicated across the hydro's evaporation
/// slots.
fn fill_evaporation_rows(
    ctx: &TemplateBuildCtx<'_>,
    stage_idx: usize,
    layout: &StageLayout,
    row_lower: &mut [f64],
    row_upper: &mut [f64],
) {
    let n_evap_slots = layout.n_evap_slots;
    for (local_idx, &h) in layout.geometry.evap_hydro_indices.iter().enumerate() {
        match ctx.evaporation_models.model(h.get()) {
            EvaporationModel::Linearized { coefficients, .. } => {
                debug_assert!(
                    stage_idx < coefficients.len(),
                    "stage index {stage_idx} out of bounds for evaporation coefficients (len = {})",
                    coefficients.len()
                );
                let intercept_m3s = coefficients[stage_idx].intercept_m3s;
                let local = EvapLocal::new(local_idx);
                for slot in 0..n_evap_slots {
                    let row = layout.evap_row(local, BlockIdx::new(slot));
                    row_lower[row] = intercept_m3s;
                    row_upper[row] = intercept_m3s;
                }
            }
            EvaporationModel::None => {
                debug_assert!(
                    false,
                    "evap_hydro_indices contains hydro {} but model is None",
                    h.get()
                );
            }
        }
    }
}

/// Fill z-inflow definition row bounds: equality with RHS = `base_h` (m3/s).
///
/// The base is the deterministic PAR base inflow (before noise), NOT multiplied
/// by ζ and NOT reduced by withdrawal. The noise component (sigma · eta) is added
/// at solve time via [`super::PatchBuffer::fill_z_inflow_patches`].
fn fill_z_inflow_rows(
    ctx: &TemplateBuildCtx<'_>,
    stage_idx: usize,
    layout: &StageLayout,
    row_lower: &mut [f64],
    row_upper: &mut [f64],
) {
    let has_par = ctx.par_lp.n_stages() > 0;
    for h_idx in 0..layout.state.hydro_count {
        let row = layout.z_inflow_row(HydroSys::new(h_idx));
        let base = if has_par {
            ctx.par_lp.deterministic_base(stage_idx, h_idx)
        } else {
            0.0
        };
        row_lower[row] = base;
        row_upper[row] = base;
    }
}

/// Fill row bounds for the 4 operational violation constraint families.
///
/// The two flow families are per-hydro per-block; RHS is in rate units
/// (m3/s). The two power families (min-turbine, min-generation) are per-hydro
/// CELL per-block: each cell's RHS is the PLAIN SUM of its own member groups'
/// resolved minimum (`cell_min_turbined`/`cell_min_generation`), never the
/// plant's declared `min_turbined_m3s`/`min_generation_mw` — see the
/// min-floor contract.
///
/// - **Min outflow** (`>=`): `row_lower = min_outflow_m3s`, `row_upper = +INF`.
/// - **Max outflow** (`<=`): `row_lower = -INF`, `row_upper = max_outflow_m3s`
///   (or `+INF` when the bound is absent, making the row non-binding).
/// - **Min turbine** (`>=`, per cell): `row_lower = cell_min_turbined`, `row_upper = +INF`.
/// - **Min generation** (`>=`, per cell): `row_lower = cell_min_generation`, `row_upper = +INF`.
pub(super) fn fill_operational_violation_rows(
    ctx: &TemplateBuildCtx<'_>,
    stage_idx: usize,
    layout: &StageLayout,
    row_lower: &mut [f64],
    row_upper: &mut [f64],
) {
    // Each family writes its own computed row index, so the visit order does not
    // affect the result; the descriptor order is nonetheless pinned to the canonical
    // row-region order so the write order stays auditable against the layout.
    for h_idx in 0..layout.state.hydro_count {
        let hydro_sys = HydroSys::new(h_idx);
        for blk in 0..layout.clock.n_blks() {
            let b = BlockIdx::new(blk);
            let hb = ctx
                .resolved
                .bounds
                .hydro_bounds_at_block(h_idx, stage_idx, blk);
            let families = [
                (
                    layout.min_outflow_row(hydro_sys, b),
                    hb.min_outflow_m3s,
                    f64::INFINITY,
                ),
                (
                    layout.max_outflow_row(hydro_sys, b),
                    f64::NEG_INFINITY,
                    hb.max_outflow_m3s.unwrap_or(f64::INFINITY),
                ),
            ];
            for (row, lower, upper) in families {
                row_lower[row] = lower;
                row_upper[row] = upper;
            }
        }

        let hydro = &ctx.hydros[h_idx];
        for blk in 0..layout.clock.n_blks() {
            let b = BlockIdx::new(blk);
            let lookup =
                GroupBoundLookup::new(ctx.resolved.bounds.group_overlay(), h_idx, stage_idx, blk);
            for cell_idx in ctx.hydro_cell_index.cells_of(hydro_sys) {
                let cell = HydroCell::new(cell_idx);
                let positions = ctx.hydro_cell_index.groups_of(cell);

                let row_t = layout.min_turbine_row(cell, b);
                row_lower[row_t] = cell_min_turbined(&hydro.unit_groups, positions, lookup);
                row_upper[row_t] = f64::INFINITY;

                let row_g = layout.min_generation_row(cell, b);
                row_lower[row_g] = cell_min_generation(&hydro.unit_groups, positions, lookup);
                row_upper[row_g] = f64::INFINITY;
            }
        }
    }
}

/// Fill commitment-MATURITY equality row bounds: `0 == 0` per anticipated
/// plant whose delivery matures THIS stage
/// (`layout.anticipated.anticipated_fishing_row_pos`; a `K = 0`
/// self-delivery, or no maturing delivery, excludes a plant's row this
/// stage).
pub(super) fn fill_anticipated_fishing_rows(
    layout: &StageLayout,
    row_lower: &mut [f64],
    row_upper: &mut [f64],
) {
    let n_active = fill_zero_equality_rows(
        layout.anticipated.fishing_rows.start,
        &layout.anticipated.anticipated_fishing_row_pos,
        row_lower,
        row_upper,
    );
    debug_assert_eq!(
        n_active,
        layout.anticipated.fishing_rows.len(),
        "fill_anticipated_fishing_rows: active count mismatch"
    );
}

/// Fill the `anticipated_state_out` (deposit) definition equality row bounds
/// (`0 == 0`) for each plant with a genuine, ACTIVE decision this stage
/// (`layout.anticipated.anticipated_decision_row_pos`, the single
/// position-table owner). Inactive or no-genuine-decision plants emit no
/// row, so rows pack at the SPARSE position-table offset — unlike
/// [`fill_anticipated_fishing_rows`], which is dense per anticipated plant.
pub(super) fn fill_anticipated_state_out_def_rows(
    layout: &StageLayout,
    row_lower: &mut [f64],
    row_upper: &mut [f64],
) {
    let n_active = fill_zero_equality_rows(
        layout.anticipated.state_out_def_rows.start,
        &layout.anticipated.anticipated_decision_row_pos,
        row_lower,
        row_upper,
    );
    debug_assert_eq!(
        n_active,
        layout.anticipated.state_out_def_rows.len(),
        "fill_anticipated_state_out_def_rows: active count mismatch"
    );
}

/// Fill the future-window commitment-carry equality row bounds (`0 == 0`),
/// one row per (plant, modular slot) carrying a strictly future, not-yet-due
/// delivery this stage (`layout.anticipated.anticipated_slot_row_pos`,
/// sparse like [`fill_anticipated_state_out_def_rows`]'s `active_pos`
/// offset) — a slot beyond the study horizon, not yet ready, or claimed by
/// the latch/maturity rows gets no row. Empty when `n_anticipated * k_max ==
/// 0`.
fn fill_anticipated_slot_definition_rows(
    layout: &StageLayout,
    row_lower: &mut [f64],
    row_upper: &mut [f64],
) {
    fill_zero_equality_rows(
        layout.anticipated.slot_definition_rows.start,
        &layout.anticipated.anticipated_slot_row_pos,
        row_lower,
        row_upper,
    );
}

/// Write `0 == 0` bounds at each present position of a sparse row-position table.
fn fill_zero_equality_rows(
    row_start: usize,
    row_pos: &[Option<usize>],
    row_lower: &mut [f64],
    row_upper: &mut [f64],
) -> usize {
    let mut n_active = 0_usize;
    for row in (0..row_pos.len()).filter_map(|i| position_table_row(row_start, row_pos, i)) {
        row_lower[row] = 0.0;
        row_upper[row] = 0.0;
        n_active += 1;
    }
    n_active
}
