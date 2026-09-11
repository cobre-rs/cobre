use cobre_core::commissioning::{Phase, commissioning_active, filling_phase};
use cobre_core::{
    ContractType, HydroBlockBounds, HydroUnitGroup, ResolvedHydroUnitGroupBounds, Stage,
};

use crate::hydro_models::{EvaporationModel, ResolvedProductionModel};
use crate::indexer::{
    AnticipatedLocal, BlockIdx, Boundary, EvapLocal, FillingTargetLocal, FloorLocal, FphaCellLocal,
    FphaLocal, HydroCell, HydroSys, LineSys, anticipated_resolution_for,
    is_anticipated_decision_active_for_delivery,
};

use super::EVAPORATION_FLOW_SAFETY_MARGIN;
use super::layout::{StageLayout, TemplateBuildCtx};
use crate::generic_constraints::contract_family_slot;

/// Mutable column-bound and objective buffers shared by all fill helpers.
pub(super) struct ColumnBufs<'a> {
    pub(super) col_lower: &'a mut [f64],
    pub(super) col_upper: &'a mut [f64],
    pub(super) objective: &'a mut [f64],
}

/// Fill column lower/upper bounds and objective coefficients for one stage.
pub(super) fn fill_stage_columns(
    ctx: &TemplateBuildCtx<'_>,
    stage: &Stage,
    stage_idx: usize,
    layout: &StageLayout,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut col_lower = vec![0.0_f64; layout.num_cols];
    let mut col_upper = vec![f64::INFINITY; layout.num_cols];
    let mut objective = vec![0.0_f64; layout.num_cols];
    let total_stage_hours: f64 = stage.blocks.iter().map(|b| b.duration_hours).sum();
    let bufs = &mut ColumnBufs {
        col_lower: &mut col_lower,
        col_upper: &mut col_upper,
        objective: &mut objective,
    };

    fill_storage_columns(ctx, stage, stage_idx, layout, bufs);
    fill_transit_bucket_columns(layout, bufs);
    fill_anticipated_slot_columns(layout, bufs);
    fill_ar_lag_columns(layout, bufs);
    fill_anticipated_state_columns(layout, bufs);
    fill_theta_column(layout, bufs);
    fill_turbine_columns(ctx, stage, stage_idx, layout, bufs);
    fill_spillage_columns(ctx, stage, stage_idx, layout, bufs);
    fill_diversion_columns(ctx, stage, stage_idx, layout, bufs);
    fill_thermal_columns(ctx, stage, stage_idx, layout, bufs);
    fill_anticipated_columns(ctx, stage_idx, layout, bufs);
    fill_line_columns(ctx, stage, stage_idx, layout, bufs);
    fill_deficit_and_excess_columns(ctx, stage, stage_idx, layout, bufs);
    fill_inflow_slack_columns(ctx, stage_idx, layout, total_stage_hours, bufs);
    fill_fpha_generation_columns(ctx, stage_idx, layout, bufs);
    fill_evaporation_columns(ctx, stage, stage_idx, layout, bufs);
    fill_withdrawal_slack_columns(ctx, stage_idx, layout, total_stage_hours, bufs);
    fill_operational_slack_columns(ctx, stage, stage_idx, layout, bufs);
    fill_ncs_columns(ctx, stage, stage_idx, layout, bufs);
    fill_pumping_columns(ctx, stage, stage_idx, layout, bufs);
    fill_contract_columns(ctx, stage, stage_idx, layout, bufs);
    fill_filling_target_columns(ctx, stage_idx, layout, bufs);
    fill_filled_min_storage_floor_columns(ctx, stage_idx, layout, bufs);
    fill_z_inflow_columns(layout, bufs);

    (col_lower, col_upper, objective)
}

/// Outgoing and incoming storage columns.
///
/// Incoming storage `v_in_h` is left unconstrained here — pinned at solve time via
/// column bounds, not this site.
fn fill_storage_columns(
    ctx: &TemplateBuildCtx<'_>,
    stage: &Stage,
    stage_idx: usize,
    layout: &StageLayout,
    bufs: &mut ColumnBufs<'_>,
) {
    for h_idx in 0..layout.n_h {
        let hydro = &ctx.hydros[h_idx];
        // CONTRACT: `min_storage` is a HARD lower bound for every hydro EXCEPT (a) a
        // filling one (`hydro.filling.is_some()`), whose floor relaxes to `0` in ALL
        // phases and is re-imposed as a SOFT floor by
        // `fill_filled_min_storage_floor_columns`, and (b) a non-filling hydro while
        // `PreFilling` (commissioning-dormant), whose frozen-identity row pins `v_h`
        // to the inert IC storage — a hard floor above that IC value would reject the
        // pin and make the LP infeasible. FORBIDDEN — globalizing the relax to all
        // Operating hydros (makes dead volume soft system-wide); keeping it hard
        // through a dormant non-filling stage (rejects the IC pin). The dormant relax
        // disappears at `Operating`, restoring the hard floor.
        let floor_off = hydro.filling.is_some()
            || matches!(
                filling_phase(
                    hydro.filling.as_ref(),
                    hydro.entry_stage_id,
                    hydro.exit_stage_id,
                    stage.id,
                ),
                Phase::PreFilling
            );
        let hb = ctx.resolved.bounds.hydro_bounds(h_idx, stage_idx);
        let storage_lower = if floor_off { 0.0 } else { hb.min_storage_hm3 };
        bufs.col_lower[h_idx] = storage_lower;
        bufs.col_upper[h_idx] = hb.max_storage_hm3;
        let storage_in_col = layout.col_storage_in_start() + h_idx;
        bufs.col_lower[storage_in_col] = f64::NEG_INFINITY;
        bufs.col_upper[storage_in_col] = f64::INFINITY;
        // Interior Sᵏ reuse the outgoing column's EXACT bounds, floor_off included: the
        // frozen-identity chain pins each interior to the inert IC, so a hard floor above
        // IC would reject the pin.
        for k in 1..layout.n_blks {
            let col = layout.block_storage_col(HydroSys::new(h_idx), Boundary::Interior(k));
            bufs.col_lower[col] = storage_lower;
            bufs.col_upper[col] = hb.max_storage_hm3;
        }
    }
}

/// Travel-time bucket state columns.
///
/// A masked lag (no definition row, `entries::fill_transit_bucket_definition_entries`)
/// gets its outgoing column frozen `[0, 0]` here — the column-freeze half of the
/// two-sided masking contract; leaving it free would be a free column with no defining
/// constraint. Incoming buckets stay open, pinned every solve by `fill_col_state_patches`.
fn fill_transit_bucket_columns(layout: &StageLayout, bufs: &mut ColumnBufs<'_>) {
    let state = layout.state;
    for range in super::entries::transit_bucket_plant_ranges(state) {
        let col_base = state.transit_buckets_out.start + range.start;
        let ring = super::entries::transit_bucket_ring(state, range.clone());
        ring.freeze_masked_columns(
            &layout.rows.transit_bucket_row_pos[range],
            col_base,
            (0.0, f64::INFINITY),
            bufs,
        );
    }
}

/// Commitment-hold outgoing columns: the leading in-study slots keep the
/// two-sided reachability masking under the carry geometry — open
/// `(-inf, inf)` bounds (a committed MW value carries either sign, unlike
/// the water buckets' `[0, inf)`) for every reachable slot, frozen `[0, 0]`
/// otherwise (mirroring [`fill_transit_bucket_columns`]; a plant's own
/// latching slot is bounded later by [`fill_anticipated_columns`], which
/// overwrites this fill when active).
fn fill_anticipated_slot_columns(layout: &StageLayout, bufs: &mut ColumnBufs<'_>) {
    let base = layout.anticipated.col_anticipated_slots_out_start;
    let ring = super::entries::anticipated_ring(layout);
    ring.freeze_masked_columns(
        &layout.anticipated.anticipated_slot_row_pos,
        base,
        (f64::NEG_INFINITY, f64::INFINITY),
        bufs,
    );
}

/// AR lag columns: unconstrained (signed).
fn fill_ar_lag_columns(layout: &StageLayout, bufs: &mut ColumnBufs<'_>) {
    let n_lag_cols = layout.lag_order * layout.n_h;
    for lag_col in layout.col_inflow_lags_start()..layout.col_inflow_lags_start() + n_lag_cols {
        bufs.col_lower[lag_col] = f64::NEG_INFINITY;
        bufs.col_upper[lag_col] = f64::INFINITY;
    }
}

/// Incoming commitment-hold columns (in-study + post-horizon): open
/// `(-INF, +INF)` everywhere, left open because pinning is via
/// `set_col_bounds` at solve time (`fill_col_state_patches`), not an
/// equality row. Covers the whole merged `commit_in` range directly — the
/// post-horizon lanes need the identical open bound the in-study slots
/// already carry, so there is no separate masking to apply here.
fn fill_anticipated_state_columns(layout: &StageLayout, bufs: &mut ColumnBufs<'_>) {
    for col in layout.state.commit_in.clone() {
        bufs.col_lower[col] = f64::NEG_INFINITY;
        bufs.col_upper[col] = f64::INFINITY;
    }
}

/// Theta column: bounded below by zero so iteration-1 LPs with empty cut pools
/// are bounded rather than unbounded.
fn fill_theta_column(layout: &StageLayout, bufs: &mut ColumnBufs<'_>) {
    bufs.col_lower[layout.col_theta()] = 0.0;
    bufs.col_upper[layout.col_theta()] = f64::INFINITY;
    bufs.objective[layout.col_theta()] = 1.0;
}

/// Bundles the resolved group-bounds table with the three indices that are
/// constant across a cell's member groups, so `cell_max_turbined`/
/// `cell_max_generation` take one bundled parameter instead of four loose
/// ones that would cross `clippy::too_many_arguments`. `pub(super)` (plus the
/// `new` constructor and the `min_*` readers) so `rows.rs` can resolve the
/// same per-block group override when it fills the min-floor row RHS.
#[derive(Clone, Copy)]
pub(super) struct GroupBoundLookup<'a> {
    table: &'a ResolvedHydroUnitGroupBounds,
    hydro_idx: usize,
    stage_idx: usize,
    block_idx: usize,
}

impl<'a> GroupBoundLookup<'a> {
    pub(super) fn new(
        table: &'a ResolvedHydroUnitGroupBounds,
        hydro_idx: usize,
        stage_idx: usize,
        block_idx: usize,
    ) -> Self {
        Self {
            table,
            hydro_idx,
            stage_idx,
            block_idx,
        }
    }
}

impl GroupBoundLookup<'_> {
    /// Group `group_pos`'s resolved turbined-flow maximum — the override when
    /// the study supplies one, `group.max_turbined_m3s` otherwise.
    fn max_turbined(&self, group_pos: usize, group: &HydroUnitGroup) -> f64 {
        self.table
            .override_at_block(self.hydro_idx, group_pos, self.stage_idx, self.block_idx)
            .max_turbined_m3s
            .unwrap_or(group.max_turbined_m3s)
    }

    /// Group `group_pos`'s resolved generation maximum — the override when the
    /// study supplies one, `group.max_generation_mw` otherwise.
    fn max_generation(&self, group_pos: usize, group: &HydroUnitGroup) -> f64 {
        self.table
            .override_at_block(self.hydro_idx, group_pos, self.stage_idx, self.block_idx)
            .max_generation_mw
            .unwrap_or(group.max_generation_mw)
    }

    /// Group `group_pos`'s resolved turbined-flow minimum — the override when
    /// the study supplies one, `group.min_turbined_m3s` otherwise.
    pub(super) fn min_turbined(&self, group_pos: usize, group: &HydroUnitGroup) -> f64 {
        self.table
            .override_at_block(self.hydro_idx, group_pos, self.stage_idx, self.block_idx)
            .min_turbined_m3s
            .unwrap_or(group.min_turbined_m3s)
    }

    /// Group `group_pos`'s resolved generation minimum — the override when the
    /// study supplies one, `group.min_generation_mw` otherwise.
    pub(super) fn min_generation(&self, group_pos: usize, group: &HydroUnitGroup) -> f64 {
        self.table
            .override_at_block(self.hydro_idx, group_pos, self.stage_idx, self.block_idx)
            .min_generation_mw
            .unwrap_or(group.min_generation_mw)
    }
}

/// Cell `c`'s turbined-flow upper bound. A `ConstantProductivity` model folds
/// EACH member group's own MW cap into its own flow cap first, then sums —
/// summing the raw group boxes and folding the total instead overstates the
/// cell, since `min` does not distribute over a sum whose terms bind on
/// different sides (`test_same_bus_groups_sum_into_one_cell_box`). Any other
/// model (FPHA; a non-positive productivity) sums each group's flow cap
/// unfolded, exact because FPHA's turbine and generation columns are
/// independent.
///
/// Both terms of the closing `sum.min(fold(hb...))` are load-bearing, not a
/// group term guarded by an inert plant-side cap. Drop the plant term and a
/// lowering `hydro_bounds` override — the no-raising rule's own prescribed
/// remedy for a mid-horizon capacity cut — is silently discarded. Drop the
/// group term and a multi-cell plant can turbine past its declared capacity:
/// this helper and `cell_max_generation` are the ONLY readers of
/// `hb.max_turbined_m3s`/`hb.max_generation_mw` in the hydro LP path, so
/// nothing else would catch it. The plant term is a no-op only for a plant
/// with no declared groups (never a same-bus plant with several) — inert on
/// today's fixtures, not provably inert, since both admission rules allow an
/// envelope tolerance no shipped fixture exercises.
///
/// Each member group's own cap fed into the fold is its RESOLVED per-block
/// value — the override when the study supplies one, the declaration
/// otherwise (`test_cell_bound_takes_the_resolved_group_override`).
fn cell_max_turbined(
    groups: &[HydroUnitGroup],
    positions: &[usize],
    model: &ResolvedProductionModel,
    hb: HydroBlockBounds,
    lookup: GroupBoundLookup<'_>,
) -> f64 {
    let fold = |turbined: f64, generation: f64| match model {
        ResolvedProductionModel::ConstantProductivity { productivity } if *productivity > 0.0 => {
            turbined.min(generation / productivity)
        }
        _ => turbined,
    };
    let sum: f64 = positions
        .iter()
        .map(|&pos| {
            fold(
                lookup.max_turbined(pos, &groups[pos]),
                lookup.max_generation(pos, &groups[pos]),
            )
        })
        .sum();
    sum.min(fold(hb.max_turbined_m3s, hb.max_generation_mw))
}

/// Cell `c`'s min-turbine soft-floor RHS: the PLAIN SUM of the cell's own
/// member groups' resolved `min_turbined_m3s`, never a fold and never clamped
/// against the plant's declared minimum — see the min-floor contract. A floor
/// on a sum of variables (the cell's member groups all feed the same
/// aggregate turbine column) adds; it does not fold or clamp the way the
/// closing `MAX` bound does.
pub(super) fn cell_min_turbined(
    groups: &[HydroUnitGroup],
    positions: &[usize],
    lookup: GroupBoundLookup<'_>,
) -> f64 {
    positions
        .iter()
        .map(|&pos| lookup.min_turbined(pos, &groups[pos]))
        .sum()
}

/// Turbine columns per hydro cell per block.
///
/// A suspended hydro (`PreFilling`/`Filling`) forces BOTH bounds to `[0, 0]` on
/// EVERY cell: both must drop, or a positive `min_turbined_m3s` leaves the
/// infeasible `[min > 0, 0]`. `suspended`, `hp`, and `model` are read once per
/// plant, never per cell: filling/commissioning and the production model are
/// plant properties, not a group's or a bus's.
fn fill_turbine_columns(
    ctx: &TemplateBuildCtx<'_>,
    stage: &Stage,
    stage_idx: usize,
    layout: &StageLayout,
    bufs: &mut ColumnBufs<'_>,
) {
    for h_idx in 0..layout.n_h {
        let hydro = &ctx.hydros[h_idx];
        let suspended = matches!(
            filling_phase(
                hydro.filling.as_ref(),
                hydro.entry_stage_id,
                hydro.exit_stage_id,
                stage.id,
            ),
            Phase::PreFilling | Phase::Filling
        );
        let hp = ctx.resolved.penalties.hydro_penalties(h_idx, stage_idx);
        let model = ctx.production_models.model(h_idx, stage_idx);
        for blk in 0..layout.n_blks {
            let hb = ctx
                .resolved
                .bounds
                .hydro_bounds_at_block(h_idx, stage_idx, blk);
            let lookup = GroupBoundLookup {
                table: ctx.resolved.bounds.group_overlay(),
                hydro_idx: h_idx,
                stage_idx,
                block_idx: blk,
            };
            let block_hours = stage.blocks[blk].duration_hours;
            for cell_idx in ctx.hydro_cell_index.cells_of(HydroSys::new(h_idx)) {
                let cell = HydroCell::new(cell_idx);
                let turb_upper = cell_max_turbined(
                    &hydro.unit_groups,
                    ctx.hydro_cell_index.groups_of(cell),
                    model,
                    hb,
                    lookup,
                );
                let col = layout.turbine_col(cell, BlockIdx::new(blk));
                // Never a group's own min_turbined_m3s: the cell's floor is the soft
                // slack-backed min_turbine_rows row (this cell's own group-sum), not a
                // column floor, and a per-group hard floor would invent an asymmetry
                // with its own maximum.
                bufs.col_lower[col] = 0.0;
                bufs.col_upper[col] = if suspended { 0.0 } else { turb_upper };
                bufs.objective[col] = hp.turbined_cost * block_hours;
            }
        }
    }
}

/// Spillage columns per hydro per block.
///
/// CONTRACT: a `PreFilling` hydro's spillage is pinned `[0, 0]` (no dam exists yet
/// to spill from), gated on `Phase::PreFilling` ALONE, and the pin outranks any
/// resolved `min_spillage_m3s`/`max_spillage_m3s` — a user bound never lifts it.
/// Two forbidden alternatives: extending the freeze to `Filling` kills the
/// legitimate over-dam relief valve an impounding reservoir needs (D40); gating on
/// `filling.is_none()` leaves the phantom-spill hole open for a filling hydro in
/// its own `PreFilling` sub-phase (D38/D39), where a free spillage column
/// decoupled from frozen storage injects water it does not have onto the
/// downstream balance row.
fn fill_spillage_columns(
    ctx: &TemplateBuildCtx<'_>,
    stage: &Stage,
    stage_idx: usize,
    layout: &StageLayout,
    bufs: &mut ColumnBufs<'_>,
) {
    for h_idx in 0..layout.n_h {
        let hydro = &ctx.hydros[h_idx];
        let prefilling = matches!(
            filling_phase(
                hydro.filling.as_ref(),
                hydro.entry_stage_id,
                hydro.exit_stage_id,
                stage.id,
            ),
            Phase::PreFilling
        );
        let hp = ctx.resolved.penalties.hydro_penalties(h_idx, stage_idx);
        for blk in 0..layout.n_blks {
            let col = layout.spillage_col(HydroSys::new(h_idx), BlockIdx::new(blk));
            let (min_spill, max_spill) = if prefilling {
                (0.0, 0.0)
            } else {
                let hb = ctx
                    .resolved
                    .bounds
                    .hydro_bounds_at_block(h_idx, stage_idx, blk);
                (
                    hb.min_spillage_m3s.unwrap_or(0.0),
                    hb.max_spillage_m3s.unwrap_or(f64::INFINITY),
                )
            };
            bufs.col_lower[col] = min_spill;
            bufs.col_upper[col] = max_spill;
            let block_hours = stage.blocks[blk].duration_hours;
            bufs.objective[col] = hp.spillage_cost * block_hours;
        }
    }
}

/// Diversion columns per hydro per block (dense; non-diverting hydros get `[0, 0]`
/// and are presolve-eliminated).
///
/// A dormant hydro (`hydro.filling.is_some()` OR suspended) forces BOTH bounds to
/// `[0, 0]`: both must drop, or a positive `min_diversion_m3s` floor leaves the
/// infeasible `[min > 0, 0]`. The gate is `is_some()`, NOT the `Phase`, alone —
/// phase-gating would wrongly re-enable diversion at `Operating`, but a filling
/// hydro never diverts.
fn fill_diversion_columns(
    ctx: &TemplateBuildCtx<'_>,
    stage: &Stage,
    stage_idx: usize,
    layout: &StageLayout,
    bufs: &mut ColumnBufs<'_>,
) {
    for (h_idx, hydro) in ctx.hydros.iter().enumerate() {
        let hp = ctx.resolved.penalties.hydro_penalties(h_idx, stage_idx);
        let suspended = matches!(
            filling_phase(
                hydro.filling.as_ref(),
                hydro.entry_stage_id,
                hydro.exit_stage_id,
                stage.id,
            ),
            Phase::PreFilling | Phase::Filling
        );
        let dormant = hydro.filling.is_some() || suspended;
        for blk in 0..layout.n_blks {
            // CONTRACT: read the per-stage RESOLVED bounds, NOT the declaration-time
            // `hydro.diversion.max_flow_m3s` — the entity read silently drops any wired
            // per-stage (or per-block) override (mirrors every sibling column family).
            let (min_div, max_div) = if dormant {
                (0.0, 0.0)
            } else {
                let hb = ctx
                    .resolved
                    .bounds
                    .hydro_bounds_at_block(h_idx, stage_idx, blk);
                (
                    hb.min_diversion_m3s.unwrap_or(0.0),
                    hb.max_diversion_m3s.unwrap_or(0.0),
                )
            };
            let col = layout.diversion_col(HydroSys::new(h_idx), BlockIdx::new(blk));
            bufs.col_lower[col] = min_div;
            bufs.col_upper[col] = max_div;
            if max_div > 0.0 {
                let block_hours = stage.blocks[blk].duration_hours;
                bufs.objective[col] = hp.diversion_cost * block_hours;
            }
        }
    }
}

/// Thermal columns per thermal per block.
///
/// A genuinely anticipated thermal gets per-block bounds here but keeps objective
/// `0.0` — generation is priced once at the decision stage (`fill_anticipated_columns`);
/// pricing the delivery column too double-counts. "Genuinely anticipated" keys on
/// `layout.anticipated.anticipated_fishing_row_pos` (the fishing row's own gate), NOT a
/// static per-plant flag: a `K = 0` sub-stage-lead delivery has no fishing coupling and
/// prices normally, like an ordinary thermal.
///
/// A commissioning-dormant thermal (`commissioning_active == false`, keyed on `stage.id`)
/// forces BOTH bounds to `[0, 0]`: both must drop, or a `min_generation_mw` must-run floor
/// leaves the infeasible `[min > 0, 0]`. This generation column carries the operation-window
/// gate; the shifted gate (decision priced `K` stages early) lives on the decision column
/// in `fill_anticipated_columns`, not here.
///
/// The generation bound is read per block
/// ([`thermal_bounds_at_block`](cobre_core::ResolvedBounds::thermal_bounds_at_block)); the
/// cost stays stage-level (`thermal_bounds`) — `ThermalBlockOverride` has no cost field.
pub(super) fn fill_thermal_columns(
    ctx: &TemplateBuildCtx<'_>,
    stage: &Stage,
    stage_idx: usize,
    layout: &StageLayout,
    bufs: &mut ColumnBufs<'_>,
) {
    for (t_idx, thermal) in ctx.thermals.iter().enumerate() {
        let active = commissioning_active(thermal.entry_stage_id, thermal.exit_stage_id, stage.id);
        let marginal_cost_per_mwh = ctx
            .resolved
            .bounds
            .thermal_bounds(t_idx, stage_idx)
            .cost_per_mwh;
        // Indexed via `.get` rather than `[local_idx]`: `build_anticipated_fishing_row_pos`
        // returns an empty vec whenever `k_max == 0`, regardless of `n_anticipated`.
        let is_anticipated =
            layout
                .anticipated_local_by_sys_pos
                .get(&t_idx)
                .is_some_and(|&local_idx| {
                    layout
                        .anticipated
                        .anticipated_fishing_row_pos
                        .get(local_idx)
                        .copied()
                        .flatten()
                        .is_some()
                });
        for blk in 0..layout.n_blks {
            let tb = ctx
                .resolved
                .bounds
                .thermal_bounds_at_block(t_idx, stage_idx, blk);
            let col =
                layout
                    .block_grid()
                    .flat(layout.equipment.thermal.start, t_idx, BlockIdx::new(blk));
            if active {
                bufs.col_lower[col] = tb.min_generation_mw;
                bufs.col_upper[col] = tb.max_generation_mw;
            } else {
                bufs.col_lower[col] = 0.0;
                bufs.col_upper[col] = 0.0;
            }
            if !is_anticipated {
                let block_hours = stage.blocks[blk].duration_hours;
                bufs.objective[col] = marginal_cost_per_mwh * block_hours;
            }
        }
    }
}

/// Anticipated-plant decision columns: state-out bound, decision bound, and
/// decision objective for the plant's single genuine decision this stage.
///
/// The decision column is bound/costed at ITS OWN delivery stage — never the
/// decision stage `stage_idx`, never `stage_idx + constant` — split on
/// `delivery_stage < n_stages`. In study, `thermal_block_base` supplies
/// `[min, max]` and `thermal_bounds` supplies `cost_per_mwh`. Post study,
/// `ctx.post_study_resolved.anticipated_bound` supplies `(cost, min_mw,
/// max_mw)` from the post-study table ALONE — no intersection with a second
/// declaration, since `post_study_stages.json` is the sole post-horizon bound
/// surface. Both branches read delivery
/// hours/discount from the EXTENDED `ctx.delivery_total_hours`/
/// `ctx.delivery_cumulative_discount_factors` vectors and deposit into ring
/// slot `ring_index(delivery_stage) mod k_max`, the ring's modular delivery-target
/// mapping [`crate::indexer::StateSpace::commitment_hold_in_study_offset`]
/// addresses. A missing post-study cell — a deck error the loader rejects
/// upstream, never reported here — degrades to the same dormant `[0, 0]`
/// treatment an inactive plant gets.
///
/// Active (`is_anticipated_decision_active_for_delivery`) is evaluated at the decision's
/// OWN delivery stage; `delivery_stage == n_delivery` is INACTIVE (strict gate) — pricing
/// it would create a cost-only column with no delivery LP. The `anticipated_state_out_def`
/// row is emitted iff the decision is active (lockstep: zero-bound iff no def row),
/// regardless of whether the post-study branch below resolves a price.
///
/// The decision objective is the present-value commit cost UNSCALED — the caller divides
/// every non-theta entry by `COST_SCALE_FACTOR`.
pub(super) fn fill_anticipated_columns(
    ctx: &TemplateBuildCtx<'_>,
    stage_idx: usize,
    layout: &StageLayout,
    bufs: &mut ColumnBufs<'_>,
) {
    let n_stages = ctx.resolved.bounds.n_stages();
    let n_delivery = layout.state.delivery_stage_count(n_stages);
    let n_ant = ctx.n_anticipated;
    let decision_start = layout.anticipated.col_anticipated_decision_start;
    let ring = super::entries::anticipated_ring(layout);

    for local_idx in 0..n_ant {
        let col = decision_start + local_idx;
        bufs.col_lower[col] = 0.0;
        bufs.col_upper[col] = 0.0;
    }

    let mut active_count = 0_usize;
    for local_idx in 0..n_ant {
        let point =
            anticipated_resolution_for(layout.state, AnticipatedLocal::new(local_idx), n_stages);
        let Some(delivery_stage) = point.genuine_decisions_at(stage_idx).next() else {
            continue;
        };
        let decision_col = decision_start + local_idx;
        debug_assert!(
            delivery_stage > stage_idx,
            "a genuine decision's delivery stage must be strictly after the decision \
             stage (K=0 self-delivery must already be excluded)"
        );
        let slot = delivery_stage % layout.k_max;
        let state_out_col = ring.out_col(slot, local_idx);

        if is_anticipated_decision_active_for_delivery(
            layout.state,
            AnticipatedLocal::new(local_idx),
            delivery_stage,
            n_delivery,
            &ctx.anticipated_windows,
            &ctx.delivery_stage_ids,
        ) {
            active_count += 1;
            bufs.col_lower[state_out_col] = f64::NEG_INFINITY;
            bufs.col_upper[state_out_col] = f64::INFINITY;

            let bound = if delivery_stage < n_stages {
                let thermal_idx = ctx.anticipated_thermal_indices[local_idx];
                // Safe only because cobre-io's load-time validation rejects a
                // `block_id` bound row on an anticipated thermal, so the base is the
                // value at every block — a guarantee this type cannot see.
                let cap = ctx
                    .resolved
                    .bounds
                    .thermal_block_base(thermal_idx.get(), delivery_stage);
                let cost = ctx
                    .resolved
                    .bounds
                    .thermal_bounds(thermal_idx.get(), delivery_stage)
                    .cost_per_mwh;
                Some((cap.min_generation_mw, cap.max_generation_mw, cost))
            } else {
                ctx.post_study_resolved
                    .anticipated_bound(AnticipatedLocal::new(local_idx), delivery_stage - n_stages)
                    .map(|(cost, min_mw, max_mw)| (min_mw, max_mw, cost))
            };

            if let Some((min_mw, max_mw, cost)) = bound {
                bufs.col_lower[decision_col] = min_mw;
                bufs.col_upper[decision_col] = max_mw;

                let delivery_hours = ctx.delivery_total_hours[delivery_stage];
                let d_factor = ctx.delivery_cumulative_discount_factors[delivery_stage];
                bufs.objective[decision_col] = cost * delivery_hours * d_factor;
            } else {
                bufs.col_lower[decision_col] = 0.0;
                bufs.col_upper[decision_col] = 0.0;
            }
        }
    }
    debug_assert_eq!(
        active_count, layout.anticipated.n_anticipated_state_out_def_rows,
        "active state_out column count must match def-row count at stage {stage_idx}"
    );
}

/// Line columns per line per block (forward and reverse).
///
/// A commissioning-dormant line (`commissioning_active == false`, keyed on `stage.id`)
/// forces `col_upper` to `0` both directions; `col_lower` is already `0` (no
/// transmission floor), so only the cap drops.
fn fill_line_columns(
    ctx: &TemplateBuildCtx<'_>,
    stage: &Stage,
    stage_idx: usize,
    layout: &StageLayout,
    bufs: &mut ColumnBufs<'_>,
) {
    for (l_idx, line) in ctx.lines.iter().enumerate() {
        let active = commissioning_active(line.entry_stage_id, line.exit_stage_id, stage.id);
        let lp = ctx.resolved.penalties.line_penalties(l_idx, stage_idx);
        for blk in 0..layout.n_blks {
            let lb = ctx
                .resolved
                .bounds
                .line_bounds_at_block(l_idx, stage_idx, blk);
            let col_fwd = layout.line_fwd_col(LineSys::new(l_idx), BlockIdx::new(blk));
            let col_rev = layout.line_rev_col(LineSys::new(l_idx), BlockIdx::new(blk));
            if active {
                bufs.col_upper[col_fwd] = lb.direct_mw;
                bufs.col_upper[col_rev] = lb.reverse_mw;
            } else {
                bufs.col_upper[col_fwd] = 0.0;
                bufs.col_upper[col_rev] = 0.0;
            }
            let block_hours = stage.blocks[blk].duration_hours;
            bufs.objective[col_fwd] = lp.exchange_cost * block_hours;
            bufs.objective[col_rev] = lp.exchange_cost * block_hours;
        }
    }
}

/// Deficit and excess columns per bus per block.
///
/// The deficit region uses a uniform per-bus stride of `max_deficit_segments`
/// (column address owned by `deficit_col`). Buses with fewer segments leave the
/// trailing slots at the `[0, 0]` vec default, presolve-eliminated.
fn fill_deficit_and_excess_columns(
    ctx: &TemplateBuildCtx<'_>,
    stage: &Stage,
    stage_idx: usize,
    layout: &StageLayout,
    bufs: &mut ColumnBufs<'_>,
) {
    for (b_idx, bus) in ctx.buses.iter().enumerate() {
        let bp = ctx.resolved.penalties.bus_penalties(b_idx, stage_idx);
        for (seg_idx, segment) in bus.deficit_segments.iter().enumerate() {
            for blk in 0..layout.n_blks {
                let col_def = layout.deficit_col(b_idx, seg_idx, BlockIdx::new(blk));
                let block_hours = stage.blocks[blk].duration_hours;
                bufs.col_upper[col_def] = segment.depth_mw.unwrap_or(f64::INFINITY);
                bufs.objective[col_def] = segment.cost_per_mwh * block_hours;
            }
        }
        for blk in 0..layout.n_blks {
            let col_exc =
                layout
                    .block_grid()
                    .flat(layout.equipment.excess.start, b_idx, BlockIdx::new(blk));
            let block_hours = stage.blocks[blk].duration_hours;
            bufs.col_upper[col_exc] = f64::INFINITY;
            bufs.objective[col_exc] = bp.excess_cost * block_hours;
        }
    }
}

/// Inflow non-negativity slack columns (`sigma_inf_h`), one per hydro. Bounds
/// `[0, +inf)` are the vec default; only the objective is written.
fn fill_inflow_slack_columns(
    ctx: &TemplateBuildCtx<'_>,
    stage_idx: usize,
    layout: &StageLayout,
    total_stage_hours: f64,
    bufs: &mut ColumnBufs<'_>,
) {
    if ctx.has_penalty {
        for h_idx in 0..layout.n_h {
            let col = layout.slack.inflow_slack.start + h_idx;
            let hp = ctx.resolved.penalties.hydro_penalties(h_idx, stage_idx);
            bufs.objective[col] = hp.inflow_nonnegativity_cost * total_stage_hours;
        }
    }
}

/// Cell `c`'s FPHA generation-column upper bound. FPHA's turbine and generation
/// columns are independent (no productivity fold couples them), so summing
/// `max_generation_mw` over the cell's member groups directly is exact.
///
/// Both terms of `sum.min(hb.max_generation_mw)` are load-bearing — the same
/// two-term contract `cell_max_turbined` states in full. Drop the plant term
/// and a lowering `hydro_bounds` override is silently discarded; drop the
/// group term and a multi-cell plant can generate past its declared capacity,
/// since this helper is the ONLY reader of `hb.max_generation_mw` in the
/// hydro LP path.
///
/// Each member group's own cap fed into the sum is its RESOLVED per-block
/// value — the override when the study supplies one, the declaration
/// otherwise (`test_generation_cell_bound_takes_the_resolved_group_override`).
fn cell_max_generation(
    groups: &[HydroUnitGroup],
    positions: &[usize],
    hb: HydroBlockBounds,
    lookup: GroupBoundLookup<'_>,
) -> f64 {
    let sum: f64 = positions
        .iter()
        .map(|&pos| lookup.max_generation(pos, &groups[pos]))
        .sum();
    sum.min(hb.max_generation_mw)
}

/// Cell `c`'s min-generation soft-floor RHS: the PLAIN SUM of the cell's own
/// member groups' resolved `min_generation_mw` — never folded through a
/// productivity, never clamped against the plant's declared minimum. See
/// [`cell_min_turbined`] and the min-floor contract.
pub(super) fn cell_min_generation(
    groups: &[HydroUnitGroup],
    positions: &[usize],
    lookup: GroupBoundLookup<'_>,
) -> f64 {
    positions
        .iter()
        .map(|&pos| lookup.min_generation(pos, &groups[pos]))
        .sum()
}

/// FPHA generation columns (`g_{h,k}`): one per FPHA hydro CELL per block, bounds
/// `[0, max_generation_mw]`, objective `0.0` (turbined cost is on the turbine column).
///
/// `identify_fpha_hydros` excludes a filling hydro from `fpha_hydro_indices` during
/// `PreFilling`/`Filling` — the single owner of "no generation while filling"; this
/// loop must NOT re-gate by phase (the branch would be dead).
fn fill_fpha_generation_columns(
    ctx: &TemplateBuildCtx<'_>,
    stage_idx: usize,
    layout: &StageLayout,
    bufs: &mut ColumnBufs<'_>,
) {
    for (local_idx, &h) in layout.fpha_hydro_indices.iter().enumerate() {
        let local_idx = FphaLocal::new(local_idx);
        let hydro = &ctx.hydros[h.get()];
        let fpha_cell_base = layout.fpha_local_first_cell(local_idx).get();
        for blk in (0..layout.n_blks).map(BlockIdx::new) {
            let hb = ctx
                .resolved
                .bounds
                .hydro_bounds_at_block(h.get(), stage_idx, blk.get());
            let lookup = GroupBoundLookup {
                table: ctx.resolved.bounds.group_overlay(),
                hydro_idx: h.get(),
                stage_idx,
                block_idx: blk.get(),
            };
            for (offset, cell_idx) in ctx.hydro_cell_index.cells_of(h).enumerate() {
                let cell = HydroCell::new(cell_idx);
                let gen_upper = cell_max_generation(
                    &hydro.unit_groups,
                    ctx.hydro_cell_index.groups_of(cell),
                    hb,
                    lookup,
                );
                let col = layout.generation_col(FphaCellLocal::new(fpha_cell_base + offset), blk);
                // Never a group's own min_generation_mw: see fill_turbine_columns's
                // identical col_lower contract (min_generation_rows stays the sole
                // owner of the cell's soft floor).
                bufs.col_lower[col] = 0.0;
                bufs.col_upper[col] = gen_upper;
            }
        }
    }
}

/// Evaporation columns: one `EVAP_COLS_PER_HYDRO` triple (evaporation outflow,
/// `f_evap_plus`, `f_evap_minus`) per evaporation hydro per block.
///
/// The evaporation-outflow column is bounded symmetrically `[-q_max, +q_max]`, zero
/// objective. `f_evap_plus`/`f_evap_minus` are `[0, +inf)` and carry the directional
/// violation costs scaled by **that block's** `duration_hours`, not `total_stage_hours`
/// — the flow enters the water balance per block, so a stage-total factor inflates the
/// penalty `K`-fold at `K ≥ 2`.
fn fill_evaporation_columns(
    ctx: &TemplateBuildCtx<'_>,
    stage: &Stage,
    stage_idx: usize,
    layout: &StageLayout,
    bufs: &mut ColumnBufs<'_>,
) {
    for (local_idx, &h) in layout.evap_hydro_indices.iter().enumerate() {
        let local_idx = EvapLocal::new(local_idx);
        let (q_max_abs, hp) = match ctx.evaporation_models.model(h.get()) {
            EvaporationModel::Linearized { coefficients, .. } => {
                let coeff = &coefficients[stage_idx];
                let hb = ctx.resolved.bounds.hydro_bounds(h.get(), stage_idx);
                let q_max_abs = (coeff.intercept_m3s
                    + coeff.volume_slope_m3s_per_hm3 * hb.max_storage_hm3)
                    .abs()
                    * EVAPORATION_FLOW_SAFETY_MARGIN;
                (
                    q_max_abs,
                    ctx.resolved.penalties.hydro_penalties(h.get(), stage_idx),
                )
            }
            EvaporationModel::None => {
                debug_assert!(
                    false,
                    "evap_hydro_indices contains hydro {} but model is None",
                    h.get()
                );
                continue;
            }
        };
        for blk in 0..layout.n_blks {
            let col_evaporation_flow = layout.evap_flow_col(local_idx, BlockIdx::new(blk));
            let col_f_plus = layout.evap_f_plus_col(local_idx, BlockIdx::new(blk));
            let col_f_minus = layout.evap_f_minus_col(local_idx, BlockIdx::new(blk));
            // Signed: a negative outflow reads as net rainfall input (inflow).
            bufs.col_lower[col_evaporation_flow] = -q_max_abs;
            bufs.col_upper[col_evaporation_flow] = q_max_abs;
            bufs.col_lower[col_f_plus] = 0.0;
            bufs.col_upper[col_f_plus] = f64::INFINITY;
            bufs.col_lower[col_f_minus] = 0.0;
            bufs.col_upper[col_f_minus] = f64::INFINITY;
            // f_evap_plus = under-evaporation, f_evap_minus = over-evaporation.
            let block_hours = stage.blocks[blk].duration_hours;
            bufs.objective[col_f_plus] = hp.evaporation_violation_neg_cost * block_hours;
            bufs.objective[col_f_minus] = hp.evaporation_violation_pos_cost * block_hours;
        }
    }
}

/// Withdrawal violation slack columns — neg (under-withdrawal) and pos
/// (over-withdrawal), one stage-level column per hydro per direction.
///
/// Realized withdrawal is `R = T - neg + pos`, `T = water_withdrawal_m3s` (signed:
/// `T > 0` removal, `T < 0` inter-basin return). `R` must NOT flip sign, so the
/// *under-delivery* direction (the one dragging `R` toward zero) is capped at `|T|`
/// while *over-delivery* is left `+∞`:
/// - `T > 0`: cap `neg ≤ |T|` (floors `R ≥ 0`); `pos` is `+∞`.
/// - `T < 0`: cap `pos ≤ |T|` (floors `R ≤ 0`); `neg` is `+∞`.
/// - `T = 0`: both `0` (presolve-eliminated).
fn fill_withdrawal_slack_columns(
    ctx: &TemplateBuildCtx<'_>,
    stage_idx: usize,
    layout: &StageLayout,
    total_stage_hours: f64,
    bufs: &mut ColumnBufs<'_>,
) {
    for h_idx in 0..layout.n_h {
        let hb = ctx.resolved.bounds.hydro_bounds(h_idx, stage_idx);
        let hp = ctx.resolved.penalties.hydro_penalties(h_idx, stage_idx);
        let t = hb.water_withdrawal_m3s;

        let neg_col = layout.slack.withdrawal_slack_neg.start + h_idx;
        bufs.col_upper[neg_col] = if t > 0.0 {
            t
        } else if t < 0.0 {
            f64::INFINITY
        } else {
            0.0
        };
        bufs.objective[neg_col] = hp.water_withdrawal_violation_neg_cost * total_stage_hours;

        let pos_col = layout.slack.withdrawal_slack_pos.start + h_idx;
        bufs.col_upper[pos_col] = if t > 0.0 {
            f64::INFINITY
        } else if t < 0.0 {
            -t
        } else {
            0.0
        };
        bufs.objective[pos_col] = hp.water_withdrawal_violation_pos_cost * total_stage_hours;
    }
}

/// One hydro-keyed operational-violation slack family (min-outflow,
/// max-outflow), addressing a disjoint `n_h * n_blks` column range. Every
/// activation predicate reads the resolved per-block bound
/// (`hydro_bounds_at_block`) inside the `for blk` loop — never the entity
/// declaration on `ctx.hydros[h_idx]` (drops per-stage/per-block overrides) and
/// never a stage-level read hoisted above the loop (drops a block-only floor,
/// e.g. `min_outflow_m3s > 0.0` on one block only, leaving no slack column to
/// relax it); either alternative compiles and silently mis-activates the column.
/// The power-side families (min-turbine, min-generation) are CELL-keyed —
/// see [`CellSlackFamily`] — because a plant's floor is a per-cell sum, not a
/// plant-level aggregate; only the two flow families stay hydro-keyed here.
#[derive(Clone, Copy)]
enum BlockSlackFamily {
    /// Active iff resolved `min_outflow_m3s > 0.0`.
    OutflowBelow,
    /// Active iff resolved `max_outflow_m3s.is_some()` — NOT `> 0.0`: a `Some(0.0)`
    /// cap still activates the column.
    OutflowAbove,
}

/// The two cell-keyed operational-violation slack families, addressing a
/// disjoint `n_cells * n_blks` column range — the min-floor mirror of
/// [`BlockSlackFamily`]'s hydro-keyed families. See the min-floor contract.
#[derive(Clone, Copy)]
enum CellSlackFamily {
    /// Active iff the cell's resolved `cell_min_turbined` sum is `> 0.0`.
    TurbineBelow,
    /// Active iff the cell's resolved `cell_min_generation` sum is `> 0.0`.
    GenerationBelow,
}

/// Operational violation slack columns: the two hydro-keyed flow families
/// (`n_h * n_blks` each) and the two cell-keyed power families (`n_cells *
/// n_blks` each), all four disjoint ranges so call order does not affect the
/// result.
fn fill_operational_slack_columns(
    ctx: &TemplateBuildCtx<'_>,
    stage: &Stage,
    stage_idx: usize,
    layout: &StageLayout,
    bufs: &mut ColumnBufs<'_>,
) {
    for family in [
        BlockSlackFamily::OutflowBelow,
        BlockSlackFamily::OutflowAbove,
    ] {
        fill_block_family(ctx, stage, stage_idx, layout, bufs, family);
    }
    for family in [
        CellSlackFamily::TurbineBelow,
        CellSlackFamily::GenerationBelow,
    ] {
        fill_cell_block_family(ctx, stage, stage_idx, layout, bufs, family);
    }
}

/// Fill one hydro-keyed operational-violation slack family's `n_h * n_blks`
/// columns; `col_lower` stays at the `0.0` vec default.
fn fill_block_family(
    ctx: &TemplateBuildCtx<'_>,
    stage: &Stage,
    stage_idx: usize,
    layout: &StageLayout,
    bufs: &mut ColumnBufs<'_>,
    family: BlockSlackFamily,
) {
    for h_idx in 0..layout.n_h {
        let hp = ctx.resolved.penalties.hydro_penalties(h_idx, stage_idx);
        let cost = match family {
            BlockSlackFamily::OutflowBelow => hp.outflow_violation_below_cost,
            BlockSlackFamily::OutflowAbove => hp.outflow_violation_above_cost,
        };
        for blk in 0..layout.n_blks {
            let hb = ctx
                .resolved
                .bounds
                .hydro_bounds_at_block(h_idx, stage_idx, blk);
            let active = match family {
                BlockSlackFamily::OutflowBelow => hb.min_outflow_m3s > 0.0,
                BlockSlackFamily::OutflowAbove => hb.max_outflow_m3s.is_some(),
            };
            let col = match family {
                BlockSlackFamily::OutflowBelow => {
                    layout.outflow_below_col(HydroSys::new(h_idx), BlockIdx::new(blk))
                }
                BlockSlackFamily::OutflowAbove => {
                    layout.outflow_above_col(HydroSys::new(h_idx), BlockIdx::new(blk))
                }
            };
            bufs.col_upper[col] = if active { f64::INFINITY } else { 0.0 };
            bufs.objective[col] = cost * stage.blocks[blk].duration_hours;
        }
    }
}

/// Fill one cell-keyed operational-violation slack family's `n_cells * n_blks`
/// columns; `col_lower` stays at the `0.0` vec default. The cost is the
/// PLANT's own penalty (`HydroPenalties` is plant-level) at FULL magnitude —
/// never divided by the plant's cell count — replicated onto every one of the
/// plant's cells; see the min-floor contract.
fn fill_cell_block_family(
    ctx: &TemplateBuildCtx<'_>,
    stage: &Stage,
    stage_idx: usize,
    layout: &StageLayout,
    bufs: &mut ColumnBufs<'_>,
    family: CellSlackFamily,
) {
    for h_idx in 0..layout.n_h {
        let hydro = &ctx.hydros[h_idx];
        let hp = ctx.resolved.penalties.hydro_penalties(h_idx, stage_idx);
        let cost = match family {
            CellSlackFamily::TurbineBelow => hp.turbined_violation_below_cost,
            CellSlackFamily::GenerationBelow => hp.generation_violation_below_cost,
        };
        for blk in 0..layout.n_blks {
            let lookup =
                GroupBoundLookup::new(ctx.resolved.bounds.group_overlay(), h_idx, stage_idx, blk);
            let block_hours = stage.blocks[blk].duration_hours;
            for cell_idx in ctx.hydro_cell_index.cells_of(HydroSys::new(h_idx)) {
                let cell = HydroCell::new(cell_idx);
                let positions = ctx.hydro_cell_index.groups_of(cell);
                let cell_min = match family {
                    CellSlackFamily::TurbineBelow => {
                        cell_min_turbined(&hydro.unit_groups, positions, lookup)
                    }
                    CellSlackFamily::GenerationBelow => {
                        cell_min_generation(&hydro.unit_groups, positions, lookup)
                    }
                };
                let col = match family {
                    CellSlackFamily::TurbineBelow => {
                        layout.turbine_below_col(cell, BlockIdx::new(blk))
                    }
                    CellSlackFamily::GenerationBelow => {
                        layout.generation_below_col(cell, BlockIdx::new(blk))
                    }
                };
                bufs.col_upper[col] = if cell_min > 0.0 { f64::INFINITY } else { 0.0 };
                bufs.objective[col] = cost * block_hours;
            }
        }
    }
}

/// NCS generation columns: one per NCS per block, dense and system-indexed.
///
/// A commissioning-dormant NCS (`commissioning_active == false`) forces BOTH bounds to
/// `[0, 0]`: leaving the must-run lower bound at `upper > 0` would force generation from
/// a not-yet-commissioned source.
///
/// These template values govern only for non-stochastic NCS noise; with stochastic NCS,
/// `transform_ncs_noise` overwrites both bounds per scenario via `set_col_bounds`.
fn fill_ncs_columns(
    ctx: &TemplateBuildCtx<'_>,
    stage: &Stage,
    stage_idx: usize,
    layout: &StageLayout,
    bufs: &mut ColumnBufs<'_>,
) {
    for (ncs_sys_idx, ncs) in ctx.non_controllable_sources.iter().enumerate() {
        let active = commissioning_active(ncs.entry_stage_id, ncs.exit_stage_id, stage.id);
        let avail_gen = ctx
            .resolved
            .resolved_ncs_bounds
            .available_generation(ncs_sys_idx, stage_idx);
        let np = ctx.resolved.penalties.ncs_penalties(ncs_sys_idx, stage_idx);
        for blk in 0..layout.n_blks {
            let col = layout.block_grid().flat(
                layout.equipment.col_ncs_start,
                ncs_sys_idx,
                BlockIdx::new(blk),
            );
            if active {
                let factor = ctx
                    .resolved
                    .resolved_ncs_factors
                    .factor(ncs_sys_idx, stage_idx, blk);
                let upper = avail_gen * factor;
                bufs.col_upper[col] = upper;
                bufs.col_lower[col] = if ncs.allow_curtailment { 0.0 } else { upper };
            } else {
                bufs.col_upper[col] = 0.0;
                bufs.col_lower[col] = 0.0;
            }
            let block_hours = stage.blocks[blk].duration_hours;
            bufs.objective[col] = -np.curtailment_cost * block_hours;
        }
    }
}

/// Pumping-flow columns: one per station per block, dense and system-indexed.
/// Objective is zero — pumping's electrical cost enters through the bus load balance,
/// not here.
///
/// A commissioning-dormant station (`commissioning_active == false`) forces BOTH bounds
/// to `[0, 0]`: zeroing only `max` leaves the infeasible `[min > 0, 0]`.
pub(super) fn fill_pumping_columns(
    ctx: &TemplateBuildCtx<'_>,
    stage: &Stage,
    stage_idx: usize,
    layout: &StageLayout,
    bufs: &mut ColumnBufs<'_>,
) {
    for (p_sys, station) in ctx.pumping_stations.iter().enumerate() {
        let active = commissioning_active(station.entry_stage_id, station.exit_stage_id, stage.id);
        for blk in (0..layout.n_blks).map(BlockIdx::new) {
            let pb = ctx
                .resolved
                .bounds
                .pumping_bounds_at_block(p_sys, stage_idx, blk.get());
            let col = layout
                .block_grid()
                .flat(layout.equipment.col_pumping_start, p_sys, blk);
            if active {
                bufs.col_lower[col] = pb.min_flow_m3s;
                bufs.col_upper[col] = pb.max_flow_m3s;
            } else {
                bufs.col_lower[col] = 0.0;
                bufs.col_upper[col] = 0.0;
            }
        }
    }
}

/// Energy-contract columns. The family base (`col_contract_import_start` /
/// `col_contract_export_start`) is addressed by the per-family slot from
/// [`contract_family_slot`](crate::generic_constraints::contract_family_slot) — the
/// single owner the load-balance fill and the resolver also share — not by `c_sys`.
///
/// A commissioning-dormant contract has BOTH bounds forced to `[0, 0]`: zeroing only
/// `max` would leave the infeasible `[min > 0, 0]` for a take-or-pay floor.
///
/// The objective is `price_per_mwh * block_hours`, written UNSCALED and UNNEGATED for
/// both families and regardless of commissioning — the stored price sign carries
/// direction (import `> 0` cost, export `< 0` revenue) and the prescaling pass owns
/// `col_scale`.
///
/// `price_per_mwh` IS block-eligible, unlike [`fill_thermal_columns`]'s
/// `cost_per_mwh` — ratified, not an oversight.
fn fill_contract_columns(
    ctx: &TemplateBuildCtx<'_>,
    stage: &Stage,
    stage_idx: usize,
    layout: &StageLayout,
    bufs: &mut ColumnBufs<'_>,
) {
    let grid = layout.block_grid();
    for (c_sys, contract) in ctx.contracts.iter().enumerate() {
        let active =
            commissioning_active(contract.entry_stage_id, contract.exit_stage_id, stage.id);
        let (contract_type, family_slot) = contract_family_slot(ctx.contracts, c_sys);
        let (base, family_count) = match contract_type {
            ContractType::Import => (
                layout.equipment.col_contract_import_start,
                layout.equipment.n_contract_import,
            ),
            ContractType::Export => (
                layout.equipment.col_contract_export_start,
                layout.equipment.n_contract_export,
            ),
        };
        debug_assert!(
            family_slot < family_count,
            "contract family slot {family_slot} out of range {family_count} at stage {stage_idx}"
        );
        for blk in 0..layout.n_blks {
            let cb = ctx
                .resolved
                .bounds
                .contract_bounds_at_block(c_sys, stage_idx, blk);
            let col = grid.flat(base, family_slot, BlockIdx::new(blk));
            if active {
                bufs.col_lower[col] = cb.min_mw;
                bufs.col_upper[col] = cb.max_mw;
            } else {
                bufs.col_lower[col] = 0.0;
                bufs.col_upper[col] = 0.0;
            }
            let block_hours = stage.blocks[blk].duration_hours;
            bufs.objective[col] = cb.price_per_mwh * block_hours;
        }
    }
}

/// Per-stage `σ_fill`-target slack columns: one stage-level slack per Filling-phase
/// filling hydro. `[0, +∞)`, objective is the RESOLVED `filling_target_violation_cost`,
/// written UNSCALED (the caller divides non-theta entries by `COST_SCALE_FACTOR`).
///
/// CRITICAL — the cost is NOT multiplied by stage hours. `σ_fill` is a
/// STORAGE-VOLUME slack (hm³) and the cost is $/hm³, so `σ_fill · cost` is already
/// $. The wrong-but-compiling alternative — copying the `* total_stage_hours`
/// factor the flow/power-RATE slacks carry — is a $·h/hm³ units error that lets the
/// optimizer violate the target ~744× too cheaply. (`σ^{v-}` shares this convention.)
fn fill_filling_target_columns(
    ctx: &TemplateBuildCtx<'_>,
    stage_idx: usize,
    layout: &StageLayout,
    bufs: &mut ColumnBufs<'_>,
) {
    let col_start = layout.filling.col_filling_target_start;
    for (local_idx, &h) in layout
        .filling
        .filling_target_hydro_indices
        .iter()
        .enumerate()
    {
        let local_idx = FillingTargetLocal::new(local_idx);
        let col = col_start + local_idx.get();
        let hp = ctx.resolved.penalties.hydro_penalties(h.get(), stage_idx);
        bufs.col_lower[col] = 0.0;
        bufs.col_upper[col] = f64::INFINITY;
        bufs.objective[col] = hp.filling_target_violation_cost;
    }
}

/// Soft `σ^{v-}` operating-floor slack columns: one stage-level slack per
/// Operating-phase filling hydro. `[0, +∞)`, objective is the RESOLVED
/// `storage_violation_below_cost`, written UNSCALED with the SAME no-hours,
/// $/hm³ units contract as `fill_filling_target_columns` (`σ_fill`).
///
/// DISTINCT from `σ_fill`: `σ^{v-}` fires at EVERY Operating stage, `σ_fill` at
/// every Filling stage; separate columns, costs, and stage scopes (never overlap).
fn fill_filled_min_storage_floor_columns(
    ctx: &TemplateBuildCtx<'_>,
    stage_idx: usize,
    layout: &StageLayout,
    bufs: &mut ColumnBufs<'_>,
) {
    let col_start = layout.filling.col_filled_min_storage_floor_start;
    for (local_idx, &h) in layout
        .filling
        .filled_min_storage_floor_hydro_indices
        .iter()
        .enumerate()
    {
        let local_idx = FloorLocal::new(local_idx);
        let col = col_start + local_idx.get();
        let hp = ctx.resolved.penalties.hydro_penalties(h.get(), stage_idx);
        bufs.col_lower[col] = 0.0;
        bufs.col_upper[col] = f64::INFINITY;
        bufs.objective[col] = hp.storage_violation_below_cost;
    }
}

/// Z-inflow columns: free variables for realized total inflow per hydro.
fn fill_z_inflow_columns(layout: &StageLayout, bufs: &mut ColumnBufs<'_>) {
    for h_idx in 0..layout.n_h {
        let col = layout.col_z_inflow_start() + h_idx;
        bufs.col_lower[col] = f64::NEG_INFINITY;
        bufs.col_upper[col] = f64::INFINITY;
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::float_cmp,
    clippy::needless_range_loop,
    clippy::similar_names
)]
mod interior_storage_bound_tests {
    use std::collections::{BTreeMap, HashMap};

    use cobre_core::entities::hydro::HydroGenerationModel;
    use cobre_core::{
        Block, BlockMode, BoundsCountsSpec, BoundsDefaults, BusStagePenalties, CascadeTopology,
        ContractBlockBounds, EntityId, Hydro, HydroBlockBounds, HydroStageBounds,
        HydroStagePenalties, LineBlockBounds, LineStagePenalties, NcsStagePenalties, NoiseMethod,
        PenaltiesCountsSpec, PenaltiesDefaults, PumpingBlockBounds, ResolvedBounds,
        ResolvedGenericConstraintBounds, ResolvedLoadFactors, ResolvedNcsBounds,
        ResolvedNcsFactors, ResolvedPenalties, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig, ThermalBlockBounds, ThermalStageBounds,
    };
    use cobre_stochastic::par::precompute::PrecomputedPar;

    use crate::hydro_models::{
        EvaporationModel, EvaporationModelSet, ProductionModelSet, ResolvedProductionModel,
    };
    use crate::lead_time::AnticipatedResolution;
    use crate::resolved_parameters::ResolvedParameters;

    use super::super::layout::ResolvedTables;
    use super::super::test_support::state_layout_for;
    use super::{ColumnBufs, StageLayout, TemplateBuildCtx, fill_storage_columns};
    use crate::indexer::{Boundary, HydroCellIndex, HydroSys};

    const N_STAGES: usize = 1;
    const STAGE_IDX: usize = 0;
    const N_BLKS: usize = 3;
    const MIN_STORAGE_HM3: f64 = 50.0;
    const MAX_STORAGE_HM3: f64 = 175.0;

    /// One Operating run-of-river hydro (`ConstantProductivity`, no filling, no
    /// commissioning window ⇒ `floor_off == false`), so the endpoint outgoing
    /// storage column takes the hard `[MIN_STORAGE_HM3, MAX_STORAGE_HM3]` floor and
    /// interior == endpoint is the unambiguous expectation. No FPHA/evaporation, so
    /// the layout reserves only the structural storage families.
    fn operating_hydro() -> Hydro {
        let mut hydro = Hydro {
            unit_groups: Vec::new(),
            id: EntityId(1),
            name: "H1".to_string(),
            operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            downstream_id: None,
            travel_time_hours: None,
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: MIN_STORAGE_HM3,
            max_storage_hm3: MAX_STORAGE_HM3,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: HydroGenerationModel::ConstantProductivity,
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 100.0,
            specific_productivity_mw_per_m3s_per_m: None,
            min_generation_mw: 0.0,
            max_generation_mw: 250.0,
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: None,
            evaporation_reference_volumes_hm3: None,
            diversion: None,
            filling: None,
            penalties: super::super::test_support::zero_hydro_penalties(),
        };
        hydro.declare_mirror_unit_group(EntityId(1));
        hydro
    }

    /// `ResolvedBounds` for one hydro carrying the distinct `[MIN, MAX]` storage
    /// range so the assertions bite (a wrong cell read would not equal MIN/MAX).
    fn bounds_one_hydro() -> ResolvedBounds {
        ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 1,
                n_thermals: 0,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages: N_STAGES,
                k_max: 0,
            },
            &BoundsDefaults {
                hydro: HydroStageBounds {
                    min_storage_hm3: MIN_STORAGE_HM3,
                    max_storage_hm3: MAX_STORAGE_HM3,
                    filling_min_rate_m3s: 0.0,
                    water_withdrawal_m3s: 0.0,
                },
                hydro_block: HydroBlockBounds {
                    max_turbined_m3s: 100.0,
                    max_generation_mw: 250.0,
                    ..Default::default()
                },
                thermal: ThermalStageBounds { cost_per_mwh: 0.0 },
                thermal_block: ThermalBlockBounds {
                    min_generation_mw: 0.0,
                    max_generation_mw: 0.0,
                },
                line_block: LineBlockBounds {
                    direct_mw: 0.0,
                    reverse_mw: 0.0,
                },
                pumping_block: PumpingBlockBounds {
                    min_flow_m3s: 0.0,
                    max_flow_m3s: 0.0,
                },
                contract_block: ContractBlockBounds {
                    min_mw: 0.0,
                    max_mw: 0.0,
                    price_per_mwh: 0.0,
                },
            },
        )
    }

    /// All-zero penalties for one hydro across `N_STAGES` stages so no fixture-side
    /// cost contaminates the storage objective/scale assertions, and so the full
    /// template build's penalty reads land on a populated cell.
    fn penalties_one_hydro() -> ResolvedPenalties {
        ResolvedPenalties::new(
            &PenaltiesCountsSpec {
                n_hydros: 1,
                n_buses: 0,
                n_lines: 0,
                n_ncs: 0,
                n_stages: N_STAGES,
            },
            &PenaltiesDefaults {
                hydro: HydroStagePenalties {
                    spillage_cost: 0.0,
                    diversion_cost: 0.0,
                    turbined_cost: 0.0,
                    storage_violation_below_cost: 0.0,
                    filling_target_violation_cost: 0.0,
                    turbined_violation_below_cost: 0.0,
                    outflow_violation_below_cost: 0.0,
                    outflow_violation_above_cost: 0.0,
                    generation_violation_below_cost: 0.0,
                    evaporation_violation_cost: 0.0,
                    water_withdrawal_violation_cost: 0.0,
                    water_withdrawal_violation_pos_cost: 0.0,
                    water_withdrawal_violation_neg_cost: 0.0,
                    evaporation_violation_pos_cost: 0.0,
                    evaporation_violation_neg_cost: 0.0,
                    inflow_nonnegativity_cost: 0.0,
                },
                bus: BusStagePenalties { excess_cost: 0.0 },
                line: LineStagePenalties { exchange_cost: 0.0 },
                ncs: NcsStagePenalties {
                    curtailment_cost: 0.0,
                },
            },
        )
    }

    /// A `Stage` with `N_BLKS` equal-duration blocks under `block_mode`.
    fn stage_with_blocks(block_mode: BlockMode) -> Stage {
        Stage {
            index: STAGE_IDX,
            id: STAGE_IDX as i32,
            start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: chrono::NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            season_id: Some(0),
            blocks: (0..N_BLKS)
                .map(|index| Block {
                    index,
                    name: format!("BLK{index}"),
                    duration_hours: 248.0,
                })
                .collect(),
            block_mode,
            state_config: StageStateConfig {
                storage: false,
                inflow_lags: false,
            },
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor: 1,
                noise_method: NoiseMethod::Saa,
            },
        }
    }

    /// Owns the borrow targets for a one-hydro `TemplateBuildCtx`.
    struct InteriorStorageFixtures {
        par_lp: PrecomputedPar,
        hydros: Vec<Hydro>,
        hydro_cell_index: HydroCellIndex,
        cascade: CascadeTopology,
        bounds: ResolvedBounds,
        penalties: ResolvedPenalties,
        production_models: ProductionModelSet,
        evaporation_models: EvaporationModelSet,
        resolved_generic_bounds: ResolvedGenericConstraintBounds,
        resolved_load_factors: ResolvedLoadFactors,
        resolved_ncs_bounds: ResolvedNcsBounds,
        resolved_ncs_factors: ResolvedNcsFactors,
        resolved_parameters: ResolvedParameters,
    }

    impl InteriorStorageFixtures {
        fn new() -> Self {
            let hydros = vec![operating_hydro()];
            let cascade = CascadeTopology::build(&hydros);
            let hydro_cell_index = HydroCellIndex::build(&hydros);
            Self {
                par_lp: PrecomputedPar::default(),
                hydros,
                hydro_cell_index,
                cascade,
                bounds: bounds_one_hydro(),
                penalties: penalties_one_hydro(),
                production_models: ProductionModelSet::new(
                    vec![vec![
                        ResolvedProductionModel::ConstantProductivity {
                            productivity: 1.0
                        };
                        N_STAGES
                    ]],
                    1,
                    N_STAGES,
                ),
                evaporation_models: EvaporationModelSet::new(vec![EvaporationModel::None]),
                resolved_generic_bounds: ResolvedGenericConstraintBounds::empty(),
                resolved_load_factors: ResolvedLoadFactors::empty(),
                resolved_ncs_bounds: ResolvedNcsBounds::empty(),
                resolved_ncs_factors: ResolvedNcsFactors::empty(),
                resolved_parameters: ResolvedParameters {
                    per_param: vec![],
                    id_to_slot: vec![],
                    cost_scale_factor: 1_000_000.0,
                },
            }
        }

        fn make_ctx(&self) -> TemplateBuildCtx<'_> {
            let mut hydro_pos = BTreeMap::new();
            hydro_pos.insert(self.hydros[0].id, 0_usize);
            TemplateBuildCtx {
                hydros: &self.hydros,
                thermals: &[],
                lines: &[],
                buses: &[],
                load_models: &[],
                cascade: &self.cascade,
                hydro_cell_index: &self.hydro_cell_index,
                resolved: ResolvedTables {
                    bounds: &self.bounds,
                    penalties: &self.penalties,
                    resolved_generic_bounds: &self.resolved_generic_bounds,
                    resolved_load_factors: &self.resolved_load_factors,
                    resolved_ncs_bounds: &self.resolved_ncs_bounds,
                    resolved_ncs_factors: &self.resolved_ncs_factors,
                    resolved_parameters: &self.resolved_parameters,
                },
                hydro_pos,
                thermal_pos: BTreeMap::new(),
                line_pos: BTreeMap::new(),
                bus_pos: BTreeMap::new(),
                par_lp: &self.par_lp,
                production_models: &self.production_models,
                evaporation_models: &self.evaporation_models,
                generic_constraints: &[],
                non_controllable_sources: &[],
                pumping_stations: &[],
                pumping_pos: BTreeMap::new(),
                n_pumping: 0,
                contracts: &[],
                contract_pos: BTreeMap::new(),
                n_contract_import: 0,
                n_contract_export: 0,
                diversion_upstream: HashMap::new(),
                arc_stage_weights: HashMap::new(),
                arc_spread_chrono: HashMap::new(),
                arc_arrival_density: HashMap::new(),
                per_stage_mask: Vec::new(),
                post_study_resolved: crate::setup::PostStudyResolved::default(),
                n_hydros: 1,
                n_thermals: 0,
                n_lines: 0,
                n_buses: 0,
                max_par_order: 0,
                n_anticipated: 0,
                k_max: 0,
                anticipated_lead_stages: vec![],
                anticipated_thermal_indices: vec![],
                anticipated_windows: vec![],
                anticipated_resolution: AnticipatedResolution::default(),
                study_stage_ids: vec![],
                delivery_stage_ids: vec![],
                has_penalty: false,
                delivery_cumulative_discount_factors: vec![1.0],
                delivery_total_hours: vec![744.0],
                filling_v_target: BTreeMap::new(),
            }
        }
    }

    /// Run `fill_storage_columns` against raw, unscaled buffers for `stage`,
    /// returning the bound/objective buffers plus the resolved storage-column
    /// offsets by value (the borrowed `StateSpace` cannot escape).
    fn run_fill(fixtures: &InteriorStorageFixtures, stage: &Stage) -> RawFill {
        let ctx = fixtures.make_ctx();
        let state = state_layout_for(&ctx);
        let layout = StageLayout::new(&ctx, &state, stage, STAGE_IDX);
        let mut col_lower = vec![0.0_f64; layout.num_cols];
        let mut col_upper = vec![f64::INFINITY; layout.num_cols];
        let mut objective = vec![0.0_f64; layout.num_cols];
        let mut bufs = ColumnBufs {
            col_lower: &mut col_lower,
            col_upper: &mut col_upper,
            objective: &mut objective,
        };
        fill_storage_columns(&ctx, stage, STAGE_IDX, &layout, &mut bufs);
        // The actual interior columns are the `storage_internal` range members
        // (empty in parallel mode and at K = 1); `block_storage_col(0, k)` for
        // interior `k` resolves into this range only in chronological K ≥ 2.
        let interior: Vec<usize> = layout.equipment.storage_internal.clone().collect();
        RawFill {
            col_lower,
            col_upper,
            objective,
            endpoint: layout.block_storage_col(HydroSys::new(0), Boundary::Outgoing),
            interior,
            storage_internal_empty: layout.equipment.storage_internal.is_empty(),
        }
    }

    /// Raw `fill_storage_columns` output plus the storage column offsets the
    /// assertions read.
    struct RawFill {
        col_lower: Vec<f64>,
        col_upper: Vec<f64>,
        objective: Vec<f64>,
        endpoint: usize,
        interior: Vec<usize>,
        storage_internal_empty: bool,
    }

    /// Interior `Sᵏ` columns inherit the endpoint outgoing-storage bounds, carry a
    /// `0.0` objective, and take the matrix-derived empty-column scale `1.0` while
    /// they carry no row coefficients. Parallel mode produces no interior columns,
    /// with storage bounds and objective unchanged.
    #[test]
    fn interior_storage_columns_inherit_stage_bounds_objective_scale() {
        let fixtures = InteriorStorageFixtures::new();

        // Bounds + objective: raw, unscaled buffers in chronological K = 3.
        let chrono = run_fill(&fixtures, &stage_with_blocks(BlockMode::Chronological));
        assert!(
            !chrono.storage_internal_empty,
            "chronological K=3 must reserve interior storage columns"
        );
        assert_eq!(chrono.interior.len(), N_BLKS - 1, "K - 1 interior columns");
        let endpoint_lower = chrono.col_lower[chrono.endpoint];
        let endpoint_upper = chrono.col_upper[chrono.endpoint];
        assert_eq!(
            endpoint_lower, MIN_STORAGE_HM3,
            "endpoint floor is the hard min"
        );
        assert_eq!(endpoint_upper, MAX_STORAGE_HM3, "endpoint cap is the max");
        for &col in &chrono.interior {
            assert_eq!(
                chrono.col_lower[col], endpoint_lower,
                "interior col {col} lower bound must equal the endpoint storage lower bound"
            );
            assert_eq!(
                chrono.col_upper[col], endpoint_upper,
                "interior col {col} upper bound must equal the endpoint storage upper bound"
            );
            assert_eq!(
                chrono.objective[col], 0.0,
                "interior col {col} objective must stay 0.0"
            );
        }

        // Scale: the matrix-derived scale of a structurally-empty interior column is
        // 1.0 while it carries no row coefficients (later water-balance/FPHA fills add
        // them). Build the full chronological template and run the production scale
        // computation on its CSC.
        let ctx = fixtures.make_ctx();
        let state = state_layout_for(&ctx);
        let chrono_stage = stage_with_blocks(BlockMode::Chronological);
        let layout = StageLayout::new(&ctx, &state, &chrono_stage, STAGE_IDX);
        let template = super::super::template::build_single_stage_template(
            &ctx,
            &state,
            &chrono_stage,
            STAGE_IDX,
        )
        .template;
        let col_scale = super::super::compute_col_scale(
            template.num_cols,
            &template.col_starts,
            &template.values,
        );
        for k in 1..layout.n_blks {
            let col = layout.block_storage_col(HydroSys::new(0), Boundary::Interior(k));
            assert_eq!(
                col_scale[col], 1.0,
                "interior col {col} scale must be the empty-column scale 1.0 while it carries no row coefficients"
            );
        }

        // Parallel identity: the interior loop's range is empty in parallel mode, so
        // it touches no column. Two independent parallel builds must therefore be
        // bit-for-bit identical in bounds, objective, and dense matrix — and neither
        // reserves interior columns. (A change perturbing the inert loop into the
        // parallel column block would break this dense comparison.)
        let parallel = run_fill(&fixtures, &stage_with_blocks(BlockMode::Parallel));
        assert!(
            parallel.storage_internal_empty,
            "parallel storage_internal must be empty (no interior columns)"
        );
        assert_eq!(
            parallel.interior,
            Vec::<usize>::new(),
            "parallel mode resolves no interior storage columns"
        );

        let build_parallel = || {
            let par_ctx = fixtures.make_ctx();
            let par_state = state_layout_for(&par_ctx);
            let parallel_stage = stage_with_blocks(BlockMode::Parallel);
            super::super::template::build_single_stage_template(
                &par_ctx,
                &par_state,
                &parallel_stage,
                STAGE_IDX,
            )
            .template
        };
        let tpl_a = build_parallel();
        let tpl_b = build_parallel();
        assert_eq!(tpl_a.num_cols, tpl_b.num_cols, "parallel num_cols stable");
        let dense_a = csc_to_dense(&tpl_a);
        let dense_b = csc_to_dense(&tpl_b);
        for j in 0..tpl_a.num_cols {
            assert_eq!(
                tpl_a.col_lower[j].to_bits(),
                tpl_b.col_lower[j].to_bits(),
                "parallel col_lower differs at col {j}"
            );
            assert_eq!(
                tpl_a.col_upper[j].to_bits(),
                tpl_b.col_upper[j].to_bits(),
                "parallel col_upper differs at col {j}"
            );
            assert_eq!(
                tpl_a.objective[j].to_bits(),
                tpl_b.objective[j].to_bits(),
                "parallel objective differs at col {j}"
            );
        }
        for i in 0..tpl_a.num_rows {
            for j in 0..tpl_a.num_cols {
                assert_eq!(
                    dense_a[i][j].to_bits(),
                    dense_b[i][j].to_bits(),
                    "parallel matrix differs at row {i} col {j}"
                );
            }
        }

        let (endpoint, par_storage_internal_empty) = {
            let par_ctx = fixtures.make_ctx();
            let par_state = state_layout_for(&par_ctx);
            let stage = stage_with_blocks(BlockMode::Parallel);
            let l = StageLayout::new(&par_ctx, &par_state, &stage, STAGE_IDX);
            (
                l.block_storage_col(HydroSys::new(0), Boundary::Outgoing),
                l.equipment.storage_internal.is_empty(),
            )
        };
        assert!(
            par_storage_internal_empty,
            "parallel build reserves no interior storage columns"
        );
        assert_eq!(
            tpl_a.col_lower[endpoint], MIN_STORAGE_HM3,
            "parallel endpoint storage lower bound unchanged"
        );
        assert_eq!(
            tpl_a.col_upper[endpoint], MAX_STORAGE_HM3,
            "parallel endpoint storage upper bound unchanged"
        );
        assert_eq!(
            tpl_a.objective[endpoint], 0.0,
            "parallel endpoint storage objective unchanged"
        );
    }

    /// Expand a CSC `StageTemplate` to a dense `Vec<Vec<f64>>` (mirrors the
    /// `template/tests.rs` dense-comparison helper).
    #[allow(clippy::cast_sign_loss)]
    fn csc_to_dense(tpl: &cobre_solver::StageTemplate) -> Vec<Vec<f64>> {
        let mut dense = vec![vec![0.0_f64; tpl.num_cols]; tpl.num_rows];
        for j in 0..tpl.num_cols {
            let start = tpl.col_starts[j] as usize;
            let end = tpl.col_starts[j + 1] as usize;
            for nz in start..end {
                let row = tpl.row_indices[nz] as usize;
                dense[row][j] = tpl.values[nz];
            }
        }
        dense
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::float_cmp,
    clippy::similar_names
)]
mod diversion_bound_tests {
    use std::collections::{BTreeMap, HashMap};

    use cobre_core::entities::hydro::{DiversionChannel, HydroGenerationModel};
    use cobre_core::{
        BoundsCountsSpec, BoundsDefaults, BusStagePenalties, CascadeTopology, ContractBlockBounds,
        EntityId, Hydro, HydroBlockBounds, HydroStageBounds, HydroStagePenalties, LineBlockBounds,
        LineStagePenalties, NcsStagePenalties, PenaltiesCountsSpec, PenaltiesDefaults,
        PumpingBlockBounds, ResolvedBounds, ResolvedGenericConstraintBounds, ResolvedLoadFactors,
        ResolvedNcsBounds, ResolvedNcsFactors, ResolvedPenalties, ThermalBlockBounds,
        ThermalStageBounds,
    };
    use cobre_stochastic::par::precompute::PrecomputedPar;

    use crate::hydro_models::{
        EvaporationModel, EvaporationModelSet, ProductionModelSet, ResolvedProductionModel,
    };
    use crate::indexer::HydroCellIndex;
    use crate::lead_time::AnticipatedResolution;
    use crate::resolved_parameters::ResolvedParameters;

    use super::super::layout::ResolvedTables;
    use super::super::test_support::{state_layout_for, two_block_stage, zero_hydro_penalties};
    use super::{ColumnBufs, StageLayout, TemplateBuildCtx, fill_diversion_columns};

    // Declaration-time diversion capacity on the entity. The test makes the
    // resolved per-stage override distinct from this so the two reads disagree:
    // reading the entity directly (the pre-fix bug) would yield this value.
    const DECLARATION_MAX_FLOW_M3S: f64 = 200.0;
    const N_STAGES: usize = 1;
    const STAGE_IDX: usize = 0;

    /// One run-of-river hydro carrying a declaration-time diversion channel with
    /// `DECLARATION_MAX_FLOW_M3S` capacity and a `ConstantProductivity` generation
    /// model (so the layout reserves no FPHA/evaporation columns).
    fn diverting_hydro() -> Hydro {
        let mut hydro = Hydro {
            unit_groups: Vec::new(),
            id: EntityId(1),
            name: "H1".to_string(),
            operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            downstream_id: None,
            travel_time_hours: None,
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: 0.0,
            max_storage_hm3: 200.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: HydroGenerationModel::ConstantProductivity,
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 100.0,
            specific_productivity_mw_per_m3s_per_m: None,
            min_generation_mw: 0.0,
            max_generation_mw: 250.0,
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: None,
            evaporation_reference_volumes_hm3: None,
            diversion: Some(DiversionChannel {
                downstream_id: EntityId(2),
                max_flow_m3s: DECLARATION_MAX_FLOW_M3S,
            }),
            filling: None,
            penalties: zero_hydro_penalties(),
        };
        hydro.declare_mirror_unit_group(EntityId(1));
        hydro
    }

    /// Build a `ResolvedBounds` table for one hydro across `N_STAGES` stages. The
    /// fixture seeds `max_diversion_m3s` to `None`; both test bodies overwrite that
    /// cell before asserting, so the fixture default never affects the result.
    fn bounds_one_hydro() -> ResolvedBounds {
        ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 1,
                n_thermals: 0,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages: N_STAGES,
                k_max: 0,
            },
            &BoundsDefaults {
                hydro: HydroStageBounds {
                    min_storage_hm3: 0.0,
                    max_storage_hm3: 200.0,
                    filling_min_rate_m3s: 0.0,
                    water_withdrawal_m3s: 0.0,
                },
                hydro_block: HydroBlockBounds {
                    max_turbined_m3s: 100.0,
                    max_generation_mw: 250.0,
                    ..Default::default()
                },
                thermal: ThermalStageBounds { cost_per_mwh: 0.0 },
                thermal_block: ThermalBlockBounds {
                    min_generation_mw: 0.0,
                    max_generation_mw: 0.0,
                },
                line_block: LineBlockBounds {
                    direct_mw: 0.0,
                    reverse_mw: 0.0,
                },
                pumping_block: PumpingBlockBounds {
                    min_flow_m3s: 0.0,
                    max_flow_m3s: 0.0,
                },
                contract_block: ContractBlockBounds {
                    min_mw: 0.0,
                    max_mw: 0.0,
                    price_per_mwh: 0.0,
                },
            },
        )
    }

    fn penalties_one_hydro() -> ResolvedPenalties {
        ResolvedPenalties::new(
            &PenaltiesCountsSpec {
                n_hydros: 1,
                n_buses: 0,
                n_lines: 0,
                n_ncs: 0,
                n_stages: N_STAGES,
            },
            &PenaltiesDefaults {
                hydro: HydroStagePenalties {
                    spillage_cost: 0.0,
                    diversion_cost: 0.0,
                    turbined_cost: 0.0,
                    storage_violation_below_cost: 0.0,
                    filling_target_violation_cost: 0.0,
                    turbined_violation_below_cost: 0.0,
                    outflow_violation_below_cost: 0.0,
                    outflow_violation_above_cost: 0.0,
                    generation_violation_below_cost: 0.0,
                    evaporation_violation_cost: 0.0,
                    water_withdrawal_violation_cost: 0.0,
                    water_withdrawal_violation_pos_cost: 0.0,
                    water_withdrawal_violation_neg_cost: 0.0,
                    evaporation_violation_pos_cost: 0.0,
                    evaporation_violation_neg_cost: 0.0,
                    inflow_nonnegativity_cost: 0.0,
                },
                bus: BusStagePenalties { excess_cost: 0.0 },
                line: LineStagePenalties { exchange_cost: 0.0 },
                ncs: NcsStagePenalties {
                    curtailment_cost: 0.0,
                },
            },
        )
    }

    /// Owns the borrow targets for a one-hydro `TemplateBuildCtx`.
    struct DivFixtures {
        par_lp: PrecomputedPar,
        hydros: Vec<Hydro>,
        hydro_cell_index: HydroCellIndex,
        cascade: CascadeTopology,
        bounds: ResolvedBounds,
        penalties: ResolvedPenalties,
        production_models: ProductionModelSet,
        evaporation_models: EvaporationModelSet,
        resolved_generic_bounds: ResolvedGenericConstraintBounds,
        resolved_load_factors: ResolvedLoadFactors,
        resolved_ncs_bounds: ResolvedNcsBounds,
        resolved_ncs_factors: ResolvedNcsFactors,
        resolved_parameters: ResolvedParameters,
    }

    impl DivFixtures {
        fn new() -> Self {
            let hydros = vec![diverting_hydro()];
            let cascade = CascadeTopology::build(&hydros);
            let hydro_cell_index = HydroCellIndex::build(&hydros);
            Self {
                par_lp: PrecomputedPar::default(),
                hydros,
                hydro_cell_index,
                cascade,
                bounds: bounds_one_hydro(),
                penalties: penalties_one_hydro(),
                production_models: ProductionModelSet::new(
                    vec![vec![
                        ResolvedProductionModel::ConstantProductivity {
                            productivity: 1.0
                        };
                        N_STAGES
                    ]],
                    1,
                    N_STAGES,
                ),
                evaporation_models: EvaporationModelSet::new(vec![EvaporationModel::None]),
                resolved_generic_bounds: ResolvedGenericConstraintBounds::empty(),
                resolved_load_factors: ResolvedLoadFactors::empty(),
                resolved_ncs_bounds: ResolvedNcsBounds::empty(),
                resolved_ncs_factors: ResolvedNcsFactors::empty(),
                resolved_parameters: ResolvedParameters {
                    per_param: vec![],
                    id_to_slot: vec![],
                    cost_scale_factor: 1_000_000.0,
                },
            }
        }

        /// Set the per-stage resolved `max_diversion_m3s` override for hydro 0.
        fn set_resolved_diversion(&mut self, value: Option<f64>) {
            self.bounds
                .hydro_block_base_mut(0, STAGE_IDX)
                .max_diversion_m3s = value;
        }

        /// Set the per-stage resolved `min_diversion_m3s` override for hydro 0.
        fn set_resolved_diversion_min(&mut self, value: Option<f64>) {
            self.bounds
                .hydro_block_base_mut(0, STAGE_IDX)
                .min_diversion_m3s = value;
        }

        fn make_ctx(&self) -> TemplateBuildCtx<'_> {
            let mut hydro_pos = BTreeMap::new();
            hydro_pos.insert(self.hydros[0].id, 0_usize);
            TemplateBuildCtx {
                hydros: &self.hydros,
                thermals: &[],
                lines: &[],
                buses: &[],
                load_models: &[],
                cascade: &self.cascade,
                hydro_cell_index: &self.hydro_cell_index,
                resolved: ResolvedTables {
                    bounds: &self.bounds,
                    penalties: &self.penalties,
                    resolved_generic_bounds: &self.resolved_generic_bounds,
                    resolved_load_factors: &self.resolved_load_factors,
                    resolved_ncs_bounds: &self.resolved_ncs_bounds,
                    resolved_ncs_factors: &self.resolved_ncs_factors,
                    resolved_parameters: &self.resolved_parameters,
                },
                hydro_pos,
                thermal_pos: BTreeMap::new(),
                line_pos: BTreeMap::new(),
                bus_pos: BTreeMap::new(),
                par_lp: &self.par_lp,
                production_models: &self.production_models,
                evaporation_models: &self.evaporation_models,
                generic_constraints: &[],
                non_controllable_sources: &[],
                pumping_stations: &[],
                pumping_pos: BTreeMap::new(),
                n_pumping: 0,
                contracts: &[],
                contract_pos: BTreeMap::new(),
                n_contract_import: 0,
                n_contract_export: 0,
                diversion_upstream: HashMap::new(),
                arc_stage_weights: HashMap::new(),
                arc_spread_chrono: HashMap::new(),
                arc_arrival_density: HashMap::new(),
                per_stage_mask: Vec::new(),
                post_study_resolved: crate::setup::PostStudyResolved::default(),
                n_hydros: 1,
                n_thermals: 0,
                n_lines: 0,
                n_buses: 0,
                max_par_order: 0,
                n_anticipated: 0,
                k_max: 0,
                anticipated_lead_stages: vec![],
                anticipated_thermal_indices: vec![],
                anticipated_windows: vec![],
                anticipated_resolution: AnticipatedResolution::default(),
                study_stage_ids: vec![],
                delivery_stage_ids: vec![],
                has_penalty: false,
                delivery_cumulative_discount_factors: vec![1.0],
                delivery_total_hours: vec![744.0],
                filling_v_target: BTreeMap::new(),
            }
        }
    }

    /// Run `fill_diversion_columns` against the fixture and return `col_lower`,
    /// `col_upper`, plus the two layout offsets the assertions read.
    ///
    /// Returns the layout's `(n_blks, col_diversion_start)` by value rather than
    /// the `StageLayout` itself: the layout borrows the function-local
    /// `StateSpace`, so it cannot escape — the caller only needs these two
    /// offsets to index `col_lower`/`col_upper`.
    fn run_fill(fixtures: &DivFixtures) -> (Vec<f64>, Vec<f64>, usize, usize) {
        let stage = two_block_stage(STAGE_IDX, [372.0, 372.0]);
        let ctx = fixtures.make_ctx();
        let state = state_layout_for(&ctx);
        let layout = StageLayout::new(&ctx, &state, &stage, STAGE_IDX);
        let mut col_lower = vec![0.0_f64; layout.num_cols];
        let mut col_upper = vec![f64::INFINITY; layout.num_cols];
        let mut objective = vec![0.0_f64; layout.num_cols];
        let mut bufs = ColumnBufs {
            col_lower: &mut col_lower,
            col_upper: &mut col_upper,
            objective: &mut objective,
        };
        fill_diversion_columns(&ctx, &stage, STAGE_IDX, &layout, &mut bufs);
        (
            col_lower,
            col_upper,
            layout.n_blks,
            layout.equipment.diversion.start,
        )
    }

    /// A per-stage resolved override distinct from the declaration value flows
    /// into every diversion block's `col_upper`. Fails against the pre-fix code,
    /// which read `hydro.diversion.max_flow_m3s` (= `DECLARATION_MAX_FLOW_M3S`).
    #[test]
    fn resolved_diversion_override_flows_into_col_upper() {
        let override_value = 37.5;
        assert!(
            (override_value - DECLARATION_MAX_FLOW_M3S).abs() > f64::EPSILON,
            "override must differ from the declaration value for the test to bite"
        );
        let mut fixtures = DivFixtures::new();
        fixtures.set_resolved_diversion(Some(override_value));

        let (_col_lower, col_upper, n_blks, col_diversion_start) = run_fill(&fixtures);
        for blk in 0..n_blks {
            let col = col_diversion_start + blk;
            assert_eq!(
                col_upper[col], override_value,
                "blk {blk}: col_upper[{col}] must equal the resolved override {override_value}"
            );
        }
    }

    /// No per-stage override preserves the inert default: `col_upper` equals the
    /// declaration `hydro.diversion.max_flow_m3s`. At the resolved-table level
    /// "no override" is `Some(declaration)` — the resolver seeds the declaration
    /// value into the cell and only overwrites it when a per-stage row supplies a
    /// different `Some(v)`, so this models what the resolver actually produces.
    #[test]
    fn no_override_diversion_preserves_declaration_default() {
        let mut fixtures = DivFixtures::new();
        // Seed the resolved cell with the declaration default, exactly as the
        // resolver does for a diverting hydro with no per-stage override.
        fixtures.set_resolved_diversion(Some(DECLARATION_MAX_FLOW_M3S));

        let (_col_lower, col_upper, n_blks, col_diversion_start) = run_fill(&fixtures);
        for blk in 0..n_blks {
            let col = col_diversion_start + blk;
            assert_eq!(
                col_upper[col], DECLARATION_MAX_FLOW_M3S,
                "blk {blk}: col_upper[{col}] must equal the declaration default \
                 {DECLARATION_MAX_FLOW_M3S} when no override tightens it"
            );
        }
    }

    /// An active hydro with a resolved floor below its resolved max writes both:
    /// `col_lower` gets the floor, `col_upper` keeps the max — the floor is a
    /// genuine two-sided bound, not a repricing of the existing upper bound.
    #[test]
    fn active_floor_flows_into_col_lower_alongside_max() {
        let mut fixtures = DivFixtures::new();
        fixtures.set_resolved_diversion(Some(20.0));
        fixtures.set_resolved_diversion_min(Some(5.0));

        let (col_lower, col_upper, n_blks, col_diversion_start) = run_fill(&fixtures);
        for blk in 0..n_blks {
            let col = col_diversion_start + blk;
            assert_eq!(
                col_lower[col], 5.0,
                "blk {blk}: col_lower[{col}] must equal the resolved floor 5.0"
            );
            assert_eq!(
                col_upper[col], 20.0,
                "blk {blk}: col_upper[{col}] must equal the resolved max 20.0"
            );
        }
    }

    /// No resolved floor (`min_diversion_m3s = None`) preserves today's `col_lower`
    /// of `0.0` — bit-identical byte-neutrality for every study that never sets it.
    #[test]
    fn absent_floor_preserves_zero_col_lower() {
        let mut fixtures = DivFixtures::new();
        fixtures.set_resolved_diversion(Some(DECLARATION_MAX_FLOW_M3S));

        let (col_lower, _col_upper, n_blks, col_diversion_start) = run_fill(&fixtures);
        for blk in 0..n_blks {
            let col = col_diversion_start + blk;
            assert_eq!(
                col_lower[col], 0.0,
                "blk {blk}: col_lower[{col}] stays 0.0 with no resolved floor"
            );
        }
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::float_cmp,
    clippy::similar_names
)]
mod filling_phase_gating_tests {
    use std::collections::{BTreeMap, HashMap};

    use cobre_core::entities::hydro::{FillingConfig, HydroGenerationModel};
    use cobre_core::{
        BlockBoundsCountsSpec, BoundsCountsSpec, BoundsDefaults, BusStagePenalties,
        CascadeTopology, ContractBlockBounds, EntityId, Hydro, HydroBlockBounds, HydroStageBounds,
        HydroStagePenalties, LineBlockBounds, LineStagePenalties, NcsStagePenalties,
        PenaltiesCountsSpec, PenaltiesDefaults, PumpingBlockBounds, ResolvedBlockBounds,
        ResolvedBounds, ResolvedGenericConstraintBounds, ResolvedLoadFactors, ResolvedNcsBounds,
        ResolvedNcsFactors, ResolvedPenalties, Stage, ThermalBlockBounds, ThermalStageBounds,
    };
    use cobre_stochastic::par::precompute::PrecomputedPar;

    use crate::hydro_models::{
        EvaporationModel, EvaporationModelSet, FphaPlane, ProductionModelSet,
        ResolvedProductionModel,
    };
    use crate::indexer::{BlockIdx, FphaCellLocal, HydroCell, HydroCellIndex};
    use crate::lead_time::AnticipatedResolution;
    use crate::resolved_parameters::ResolvedParameters;

    use super::super::layout::ResolvedTables;
    use super::super::test_support::{
        state_layout_for, three_block_stage, two_block_stage, zero_hydro_penalties,
    };
    use super::{
        ColumnBufs, StageLayout, TemplateBuildCtx, fill_diversion_columns,
        fill_fpha_generation_columns, fill_spillage_columns, fill_turbine_columns,
    };

    const MAX_TURBINED_M3S: f64 = 100.0;
    const MAX_GENERATION_MW: f64 = 250.0;
    const MAX_DIVERSION_M3S: f64 = 60.0;
    const START_STAGE_ID: i32 = 2;
    const ENTRY_STAGE_ID: i32 = 4;
    // One bounds row suffices: the phase is keyed on `stage.id`, which the test
    // varies independently of the resolved-bound stage index (always 0).
    const N_STAGES: usize = 1;
    const STAGE_IDX: usize = 0;

    // Three representative stage ids, one per phase, for the
    // `start_stage_id = 2`, `entry_stage_id = 4` window:
    //   id 1 < start            -> PreFilling
    //   start <= id 3 < entry   -> Filling
    //   id 4 >= entry           -> Operating
    const PREFILLING_ID: i32 = 1;
    const FILLING_ID: i32 = 3;
    const OPERATING_ID: i32 = 4;

    /// One hydro, optionally filling, optionally FPHA. `ConstantProductivity`
    /// reserves no FPHA generation column; `Fpha` reserves one per block so the
    /// generation-column gate can be exercised.
    fn hydro(filling: Option<FillingConfig>, entry: Option<i32>, fpha: bool) -> Hydro {
        let mut hydro = Hydro {
            unit_groups: Vec::new(),
            id: EntityId(1),
            name: "H1".to_string(),
            operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            downstream_id: None,
            travel_time_hours: None,
            entry_stage_id: entry,
            exit_stage_id: None,
            min_storage_hm3: 0.0,
            max_storage_hm3: 200.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: if fpha {
                HydroGenerationModel::Fpha
            } else {
                HydroGenerationModel::ConstantProductivity
            },
            min_turbined_m3s: 0.0,
            max_turbined_m3s: MAX_TURBINED_M3S,
            specific_productivity_mw_per_m3s_per_m: None,
            min_generation_mw: 0.0,
            max_generation_mw: MAX_GENERATION_MW,
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: None,
            evaporation_reference_volumes_hm3: None,
            diversion: None,
            filling,
            penalties: zero_hydro_penalties(),
        };
        hydro.declare_mirror_unit_group(EntityId(1));
        hydro
    }

    fn bounds_one_hydro() -> ResolvedBounds {
        let mut bounds = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 1,
                n_thermals: 0,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages: N_STAGES,
                k_max: 0,
            },
            &BoundsDefaults {
                hydro: HydroStageBounds {
                    min_storage_hm3: 0.0,
                    max_storage_hm3: 200.0,
                    filling_min_rate_m3s: 0.0,
                    water_withdrawal_m3s: 0.0,
                },
                hydro_block: HydroBlockBounds {
                    max_turbined_m3s: MAX_TURBINED_M3S,
                    max_generation_mw: MAX_GENERATION_MW,
                    max_diversion_m3s: Some(MAX_DIVERSION_M3S),
                    ..Default::default()
                },
                thermal: ThermalStageBounds { cost_per_mwh: 0.0 },
                thermal_block: ThermalBlockBounds {
                    min_generation_mw: 0.0,
                    max_generation_mw: 0.0,
                },
                line_block: LineBlockBounds {
                    direct_mw: 0.0,
                    reverse_mw: 0.0,
                },
                pumping_block: PumpingBlockBounds {
                    min_flow_m3s: 0.0,
                    max_flow_m3s: 0.0,
                },
                contract_block: ContractBlockBounds {
                    min_mw: 0.0,
                    max_mw: 0.0,
                    price_per_mwh: 0.0,
                },
            },
        );
        bounds.hydro_block_base_mut(0, STAGE_IDX).max_diversion_m3s = Some(MAX_DIVERSION_M3S);
        bounds
    }

    /// One-hydro all-zero penalties table sized to `N_STAGES`. A properly sized
    /// table (not `ResolvedPenalties::empty()`) is required: the column fills read
    /// `hydro_penalties(0, 0)`, which indexes into the table and would panic on an
    /// empty one.
    fn penalties_one_hydro() -> ResolvedPenalties {
        ResolvedPenalties::new(
            &PenaltiesCountsSpec {
                n_hydros: 1,
                n_buses: 0,
                n_lines: 0,
                n_ncs: 0,
                n_stages: N_STAGES,
            },
            &PenaltiesDefaults {
                hydro: zero_hydro_stage_penalties(),
                bus: BusStagePenalties { excess_cost: 0.0 },
                line: LineStagePenalties { exchange_cost: 0.0 },
                ncs: NcsStagePenalties {
                    curtailment_cost: 0.0,
                },
            },
        )
    }

    /// All-zero `HydroStagePenalties` — the per-stage resolved analogue of
    /// `zero_hydro_penalties` (which produces a declaration-time `HydroPenalties`).
    fn zero_hydro_stage_penalties() -> HydroStagePenalties {
        HydroStagePenalties {
            spillage_cost: 0.0,
            diversion_cost: 0.0,
            turbined_cost: 0.0,
            storage_violation_below_cost: 0.0,
            filling_target_violation_cost: 0.0,
            turbined_violation_below_cost: 0.0,
            outflow_violation_below_cost: 0.0,
            outflow_violation_above_cost: 0.0,
            generation_violation_below_cost: 0.0,
            evaporation_violation_cost: 0.0,
            water_withdrawal_violation_cost: 0.0,
            water_withdrawal_violation_pos_cost: 0.0,
            water_withdrawal_violation_neg_cost: 0.0,
            evaporation_violation_pos_cost: 0.0,
            evaporation_violation_neg_cost: 0.0,
            inflow_nonnegativity_cost: 0.0,
        }
    }

    /// Owns the borrow targets for a one-hydro `TemplateBuildCtx`.
    struct Fixtures {
        par_lp: PrecomputedPar,
        hydros: Vec<Hydro>,
        hydro_cell_index: HydroCellIndex,
        cascade: CascadeTopology,
        bounds: ResolvedBounds,
        penalties: ResolvedPenalties,
        production_models: ProductionModelSet,
        evaporation_models: EvaporationModelSet,
        resolved_generic_bounds: ResolvedGenericConstraintBounds,
        resolved_load_factors: ResolvedLoadFactors,
        resolved_ncs_bounds: ResolvedNcsBounds,
        resolved_ncs_factors: ResolvedNcsFactors,
        resolved_parameters: ResolvedParameters,
    }

    impl Fixtures {
        fn new(filling: Option<FillingConfig>, entry: Option<i32>, fpha: bool) -> Self {
            let hydros = vec![hydro(filling, entry, fpha)];
            let cascade = CascadeTopology::build(&hydros);
            let hydro_cell_index = HydroCellIndex::build(&hydros);
            let model = if fpha {
                ResolvedProductionModel::Fpha {
                    planes: vec![FphaPlane {
                        intercept: 0.0,
                        gamma_v: 0.0,
                        gamma_q: 0.0,
                        gamma_s: 0.0,
                    }],
                }
            } else {
                ResolvedProductionModel::ConstantProductivity { productivity: 1.0 }
            };
            Self {
                par_lp: PrecomputedPar::default(),
                hydros,
                hydro_cell_index,
                cascade,
                bounds: bounds_one_hydro(),
                penalties: penalties_one_hydro(),
                production_models: ProductionModelSet::new(
                    vec![vec![model; N_STAGES]],
                    1,
                    N_STAGES,
                ),
                evaporation_models: EvaporationModelSet::new(vec![EvaporationModel::None]),
                resolved_generic_bounds: ResolvedGenericConstraintBounds::empty(),
                resolved_load_factors: ResolvedLoadFactors::empty(),
                resolved_ncs_bounds: ResolvedNcsBounds::empty(),
                resolved_ncs_factors: ResolvedNcsFactors::empty(),
                resolved_parameters: ResolvedParameters {
                    per_param: vec![],
                    id_to_slot: vec![],
                    cost_scale_factor: 1_000_000.0,
                },
            }
        }

        fn make_ctx(&self) -> TemplateBuildCtx<'_> {
            let mut hydro_pos = BTreeMap::new();
            hydro_pos.insert(self.hydros[0].id, 0_usize);
            TemplateBuildCtx {
                hydros: &self.hydros,
                thermals: &[],
                lines: &[],
                buses: &[],
                load_models: &[],
                cascade: &self.cascade,
                hydro_cell_index: &self.hydro_cell_index,
                resolved: ResolvedTables {
                    bounds: &self.bounds,
                    penalties: &self.penalties,
                    resolved_generic_bounds: &self.resolved_generic_bounds,
                    resolved_load_factors: &self.resolved_load_factors,
                    resolved_ncs_bounds: &self.resolved_ncs_bounds,
                    resolved_ncs_factors: &self.resolved_ncs_factors,
                    resolved_parameters: &self.resolved_parameters,
                },
                hydro_pos,
                thermal_pos: BTreeMap::new(),
                line_pos: BTreeMap::new(),
                bus_pos: BTreeMap::new(),
                par_lp: &self.par_lp,
                production_models: &self.production_models,
                evaporation_models: &self.evaporation_models,
                generic_constraints: &[],
                non_controllable_sources: &[],
                pumping_stations: &[],
                pumping_pos: BTreeMap::new(),
                n_pumping: 0,
                contracts: &[],
                contract_pos: BTreeMap::new(),
                n_contract_import: 0,
                n_contract_export: 0,
                diversion_upstream: HashMap::new(),
                arc_stage_weights: HashMap::new(),
                arc_spread_chrono: HashMap::new(),
                arc_arrival_density: HashMap::new(),
                per_stage_mask: Vec::new(),
                post_study_resolved: crate::setup::PostStudyResolved::default(),
                n_hydros: 1,
                n_thermals: 0,
                n_lines: 0,
                n_buses: 0,
                max_par_order: 0,
                n_anticipated: 0,
                k_max: 0,
                anticipated_lead_stages: vec![],
                anticipated_thermal_indices: vec![],
                anticipated_windows: vec![],
                anticipated_resolution: AnticipatedResolution::default(),
                study_stage_ids: vec![],
                delivery_stage_ids: vec![],
                has_penalty: false,
                delivery_cumulative_discount_factors: vec![1.0],
                delivery_total_hours: vec![744.0],
                filling_v_target: BTreeMap::new(),
            }
        }
    }

    /// Fresh `[0, +INF]`-initialised column buffers sized to `layout.num_cols`.
    fn fresh_bufs(num_cols: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        (
            vec![0.0_f64; num_cols],
            vec![f64::INFINITY; num_cols],
            vec![0.0_f64; num_cols],
        )
    }

    /// Run the three column fills against a fixture at the given `stage_id`,
    /// returning `(col_lower, col_upper)` and the column offsets the assertions
    /// read. The layout and resolved-bound lookups use `STAGE_IDX`; the phase is
    /// keyed on `stage_id` alone, so building the stage at `stage_id` (its `id`)
    /// while pinning the resolved-bound index lets one bounds row serve all phases.
    fn run_fills(fixtures: &Fixtures, stage_id: i32) -> (Vec<f64>, Vec<f64>, [usize; 3]) {
        let stage_index = usize::try_from(stage_id).expect("test stage ids are non-negative");
        let stage = two_block_stage(stage_index, [372.0, 372.0]);
        let ctx = fixtures.make_ctx();
        let state = state_layout_for(&ctx);
        let layout = StageLayout::new(&ctx, &state, &stage, STAGE_IDX);
        let (mut col_lower, mut col_upper, mut objective) = fresh_bufs(layout.num_cols);
        let mut bufs = ColumnBufs {
            col_lower: &mut col_lower,
            col_upper: &mut col_upper,
            objective: &mut objective,
        };
        fill_turbine_columns(&ctx, &stage, STAGE_IDX, &layout, &mut bufs);
        fill_diversion_columns(&ctx, &stage, STAGE_IDX, &layout, &mut bufs);
        fill_fpha_generation_columns(&ctx, STAGE_IDX, &layout, &mut bufs);
        let offsets = [
            layout.turbine_col(HydroCell::new(0), BlockIdx::new(0)),
            layout.equipment.diversion.start,
            // FPHA-local index 0 (the sole FPHA hydro); for a non-FPHA fixture
            // there is no generation column, so callers must not read this slot.
            if layout.fpha_hydro_indices.is_empty() {
                usize::MAX
            } else {
                layout.generation_col(FphaCellLocal::new(0), BlockIdx::new(0))
            },
        ];
        (col_lower, col_upper, offsets)
    }

    /// Run `fill_spillage_columns` against the fixture at `stage_id`, returning
    /// `(col_lower, col_upper)`, the spillage column start, and the block count.
    /// Isolated like `run_storage_fill`: the spillage freeze is independent of the
    /// turbine/diversion/generation gates, so it is exercised on its own.
    fn run_spillage_fill(fixtures: &Fixtures, stage_id: i32) -> (Vec<f64>, Vec<f64>, usize, usize) {
        let stage_index = usize::try_from(stage_id).expect("test stage ids are non-negative");
        let stage = two_block_stage(stage_index, [372.0, 372.0]);
        run_spillage_fill_at(fixtures, &stage)
    }

    /// [`run_spillage_fill`], but over a three-block stage — needed for the
    /// per-block override AC, whose `block_id = 2` target has no counterpart on a
    /// two-block stage.
    fn run_spillage_fill_three_block(
        fixtures: &Fixtures,
        stage_id: i32,
    ) -> (Vec<f64>, Vec<f64>, usize, usize) {
        let stage_index = usize::try_from(stage_id).expect("test stage ids are non-negative");
        let stage = three_block_stage(stage_index);
        run_spillage_fill_at(fixtures, &stage)
    }

    fn run_spillage_fill_at(
        fixtures: &Fixtures,
        stage: &Stage,
    ) -> (Vec<f64>, Vec<f64>, usize, usize) {
        let ctx = fixtures.make_ctx();
        let state = state_layout_for(&ctx);
        let layout = StageLayout::new(&ctx, &state, stage, STAGE_IDX);
        let (mut col_lower, mut col_upper, mut objective) = fresh_bufs(layout.num_cols);
        let mut bufs = ColumnBufs {
            col_lower: &mut col_lower,
            col_upper: &mut col_upper,
            objective: &mut objective,
        };
        fill_spillage_columns(&ctx, stage, STAGE_IDX, &layout, &mut bufs);
        (
            col_lower,
            col_upper,
            layout.equipment.spillage.start,
            layout.n_blks,
        )
    }

    fn filling_config() -> FillingConfig {
        FillingConfig {
            start_stage_id: START_STAGE_ID,
            filling_min_rate_m3s: 0.0,
        }
    }

    /// A filling hydro's spillage is pinned `[0, 0]` in `PreFilling` ONLY (no dam yet
    /// to spill from), but stays FREE `[0, +∞)` in `Filling` (a real impounding
    /// reservoir can spill over-dam excess) and `Operating`. The forbidden
    /// alternative — extending the freeze to `Filling` — removes that legitimate
    /// relief valve (D40).
    #[test]
    fn filling_hydro_spillage_frozen_in_prefilling_free_in_filling_and_operating() {
        let fixtures = Fixtures::new(Some(filling_config()), Some(ENTRY_STAGE_ID), false);
        let (_lower_pre, upper_pre, spill_start, n_blks) =
            run_spillage_fill(&fixtures, PREFILLING_ID);
        for blk in 0..n_blks {
            assert_eq!(
                upper_pre[spill_start + blk],
                0.0,
                "spillage col_upper frozen [0,0] in PreFilling, blk {blk}"
            );
        }
        for stage_id in [FILLING_ID, OPERATING_ID] {
            let (_lower, upper, start, n) = run_spillage_fill(&fixtures, stage_id);
            for blk in 0..n {
                assert_eq!(
                    upper[start + blk],
                    f64::INFINITY,
                    "spillage col_upper free [0,+∞) at stage {stage_id}, blk {blk}"
                );
            }
        }
    }

    /// A commissioning-dormant non-filling hydro (`filling = None`, `entry`) has its
    /// spillage pinned `[0, 0]` while `PreFilling` (the un-built dam spills nothing),
    /// regaining free spillage from `entry` onward (`Operating`).
    #[test]
    fn dormant_non_filling_hydro_spillage_frozen_before_entry_free_after() {
        let fixtures = Fixtures::new(None, Some(ENTRY_STAGE_ID), false);
        // With filling = None, both ids < entry are PreFilling (no Filling phase
        // exists for a non-filling hydro): FILLING_ID here is just a second dormant id.
        for stage_id in [PREFILLING_ID, FILLING_ID] {
            let (_lower, upper, start, n_blks) = run_spillage_fill(&fixtures, stage_id);
            for blk in 0..n_blks {
                assert_eq!(
                    upper[start + blk],
                    0.0,
                    "dormant spillage col_upper frozen [0,0] at PreFilling stage {stage_id}, blk {blk}"
                );
            }
        }
        let (_lower, upper, start, n_blks) = run_spillage_fill(&fixtures, OPERATING_ID);
        for blk in 0..n_blks {
            assert_eq!(
                upper[start + blk],
                f64::INFINITY,
                "spillage col_upper free [0,+∞) once Operating, blk {blk}"
            );
        }
    }

    /// A non-filling hydro with no commissioning window is `Operating` at every stage:
    /// its spillage stays free `[0, +∞)` at every stage id (parity-neutral — the
    /// freeze never fires).
    #[test]
    fn non_filling_hydro_spillage_free_at_every_stage() {
        let fixtures = Fixtures::new(None, None, false);
        for stage_id in [PREFILLING_ID, FILLING_ID, OPERATING_ID] {
            let (_lower, upper, start, n_blks) = run_spillage_fill(&fixtures, stage_id);
            for blk in 0..n_blks {
                assert_eq!(
                    upper[start + blk],
                    f64::INFINITY,
                    "non-filling spillage col_upper free at stage {stage_id}, blk {blk}"
                );
            }
        }
    }

    /// A resolved `min_spillage_m3s`/`max_spillage_m3s` band flows into both
    /// bounds while `Filling` — the `PreFilling` pin fires only in its own phase,
    /// so a filling hydro's `Filling`-phase spillage takes the ordinary two-sided
    /// resolved read.
    #[test]
    fn filling_phase_resolved_band_flows_into_both_bounds() {
        let mut fixtures = Fixtures::new(Some(filling_config()), Some(ENTRY_STAGE_ID), false);
        let hb = fixtures.bounds.hydro_block_base_mut(0, STAGE_IDX);
        hb.min_spillage_m3s = Some(2.0);
        hb.max_spillage_m3s = Some(30.0);

        let (lower, upper, start, n_blks) = run_spillage_fill(&fixtures, FILLING_ID);
        for blk in 0..n_blks {
            assert_eq!(
                lower[start + blk],
                2.0,
                "spillage col_lower at Filling, blk {blk}"
            );
            assert_eq!(
                upper[start + blk],
                30.0,
                "spillage col_upper at Filling, blk {blk}"
            );
        }
    }

    /// The `PreFilling` pin outranks a resolved spillage band: even with
    /// `min_spillage_m3s = Some(2.0)` and `max_spillage_m3s = Some(30.0)`
    /// resolved, both bounds are `[0, 0]` — the pin is a conservation contract,
    /// never a default a user bound can override.
    #[test]
    fn prefilling_pin_outranks_resolved_spillage_band() {
        let mut fixtures = Fixtures::new(Some(filling_config()), Some(ENTRY_STAGE_ID), false);
        let hb = fixtures.bounds.hydro_block_base_mut(0, STAGE_IDX);
        hb.min_spillage_m3s = Some(2.0);
        hb.max_spillage_m3s = Some(30.0);

        let (lower, upper, start, n_blks) = run_spillage_fill(&fixtures, PREFILLING_ID);
        for blk in 0..n_blks {
            assert_eq!(
                lower[start + blk],
                0.0,
                "PreFilling pin wins col_lower over the resolved floor, blk {blk}"
            );
            assert_eq!(
                upper[start + blk],
                0.0,
                "PreFilling pin wins col_upper over the resolved ceiling, blk {blk}"
            );
        }
    }

    /// With no resolved spillage bounds, an `Operating`-phase hydro's spillage
    /// stays `[0, +∞)` on both sides — extends
    /// `non_filling_hydro_spillage_free_at_every_stage`'s `col_upper` coverage to
    /// `col_lower`.
    #[test]
    fn operating_no_bounds_preserves_free_range_both_sides() {
        let fixtures = Fixtures::new(None, None, false);
        let (lower, upper, start, n_blks) = run_spillage_fill(&fixtures, OPERATING_ID);
        for blk in 0..n_blks {
            assert_eq!(
                lower[start + blk],
                0.0,
                "col_lower stays 0.0 with no resolved floor, blk {blk}"
            );
            assert_eq!(
                upper[start + blk],
                f64::INFINITY,
                "col_upper stays +∞ with no resolved ceiling, blk {blk}"
            );
        }
    }

    /// A three-block `Operating` stage with a stage-wide `max_spillage_m3s =
    /// Some(50.0)` and a `block_id = 2` override to `10.0` binds ONLY block 2 —
    /// the spillage analogue of the diversion/turbine per-block cap.
    #[test]
    fn per_block_spillage_cap_binds_only_its_own_block() {
        let mut fixtures = Fixtures::new(None, None, false);
        fixtures
            .bounds
            .hydro_block_base_mut(0, STAGE_IDX)
            .max_spillage_m3s = Some(50.0);
        fixtures
            .bounds
            .set_block_overlay(ResolvedBlockBounds::new(&BlockBoundsCountsSpec {
                n_hydros: 1,
                n_thermals: 0,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages: N_STAGES,
                max_blocks: 3,
            }));
        fixtures
            .bounds
            .block_overlay_mut()
            .hydro_override_mut(0, STAGE_IDX, 2)
            .expect("overlay cell must exist for a fixture-sized overlay")
            .max_spillage_m3s = Some(10.0);

        let (_lower, upper, start, n_blks) = run_spillage_fill_three_block(&fixtures, OPERATING_ID);
        assert_eq!(n_blks, 3, "three-block fixture");
        assert_eq!(
            [upper[start], upper[start + 1], upper[start + 2]],
            [50.0, 50.0, 10.0],
            "only block 2 is bound to the override"
        );
    }

    /// A filling FPHA hydro pins its turbine column to `[0, 0]` in `PreFilling`
    /// and `Filling` (turbines not installed). Both bounds drop — zeroing only
    /// `col_upper` would leave any `col_lower` floor live and make the LP
    /// infeasible during filling.
    ///
    /// The FPHA generation column has **no bound to zero** in these phases: the
    /// per-stage FPHA row exclusion (`identify_fpha_hydros`) drops the filling
    /// hydro from `fpha_hydro_indices`, so no generation column is allocated at
    /// all — the dense FPHA-local block omits it rather than emitting an open
    /// `[0, max]` column. `run_fills` therefore reports `usize::MAX` for the
    /// generation slot here (the no-generation-column sentinel), which is why
    /// this test asserts only the turbine bound. Generation re-enters the dense
    /// block once `Operating` — covered by
    /// `filling_hydro_turbine_and_generation_normal_in_operating`.
    #[test]
    fn filling_hydro_turbine_zeroed_and_no_generation_column_before_entry() {
        let fixtures = Fixtures::new(Some(filling_config()), Some(ENTRY_STAGE_ID), true);
        for stage_id in [PREFILLING_ID, FILLING_ID] {
            let (lower, upper, [turb, _div, gen_col]) = run_fills(&fixtures, stage_id);
            assert_eq!(lower[turb], 0.0, "turbine col_lower at stage {stage_id}");
            assert_eq!(upper[turb], 0.0, "turbine col_upper at stage {stage_id}");
            assert_eq!(
                gen_col,
                usize::MAX,
                "filling hydro has no FPHA generation column during filling (stage {stage_id})"
            );
        }
    }

    /// In Operating, a filling hydro's turbine and generation columns return to
    /// their normal operating bounds — only diversion stays gated all-phases.
    #[test]
    fn filling_hydro_turbine_and_generation_normal_in_operating() {
        let fixtures = Fixtures::new(Some(filling_config()), Some(ENTRY_STAGE_ID), true);
        let (lower, upper, [turb, _div, gen_col]) = run_fills(&fixtures, OPERATING_ID);
        assert_eq!(lower[turb], 0.0, "turbine col_lower");
        assert_eq!(upper[turb], MAX_TURBINED_M3S, "turbine col_upper");
        assert_eq!(lower[gen_col], 0.0, "generation col_lower");
        assert_eq!(upper[gen_col], MAX_GENERATION_MW, "generation col_upper");
    }

    /// A filling hydro's diversion column is `[0, 0]` in ALL three phases — gated
    /// on `filling.is_some()`, not on the phase, so entry does not re-enable it.
    #[test]
    fn filling_hydro_diversion_zeroed_in_all_phases() {
        let fixtures = Fixtures::new(Some(filling_config()), Some(ENTRY_STAGE_ID), true);
        for stage_id in [PREFILLING_ID, FILLING_ID, OPERATING_ID] {
            let (lower, upper, [_turb, div, _gen_col]) = run_fills(&fixtures, stage_id);
            for blk in 0..2 {
                let col = div + blk;
                assert_eq!(
                    lower[col], 0.0,
                    "diversion col_lower blk {blk} at {stage_id}"
                );
                assert_eq!(
                    upper[col], 0.0,
                    "diversion col_upper blk {blk} at {stage_id}"
                );
            }
        }
    }

    /// A filling hydro's diversion floor drops with the freeze: a resolved
    /// `min_diversion_m3s = Some(5.0)` at an `Operating`-phase stage still yields
    /// `[0, 0]`, never the infeasible `[5.0, 0.0]` — the dormant gate on
    /// `filling.is_some()` drops both bounds together regardless of a floor.
    #[test]
    fn filling_hydro_diversion_floor_dropped_at_operating() {
        let mut fixtures = Fixtures::new(Some(filling_config()), Some(ENTRY_STAGE_ID), true);
        fixtures
            .bounds
            .hydro_block_base_mut(0, STAGE_IDX)
            .min_diversion_m3s = Some(5.0);

        let (lower, upper, [_turb, div, _gen_col]) = run_fills(&fixtures, OPERATING_ID);
        for blk in 0..2 {
            let col = div + blk;
            assert_eq!(
                lower[col], 0.0,
                "diversion col_lower blk {blk} at Operating with a floor set"
            );
            assert_eq!(
                upper[col], 0.0,
                "diversion col_upper blk {blk} at Operating with a floor set"
            );
        }
    }

    /// A non-filling FPHA hydro is `Operating` at every stage: turbine,
    /// generation, and diversion columns keep their normal bounds at every
    /// stage id (the parity-neutrality contract — all three gates no-op).
    #[test]
    fn non_filling_hydro_unchanged_at_every_stage() {
        let fixtures = Fixtures::new(None, None, true);
        for stage_id in [PREFILLING_ID, FILLING_ID, OPERATING_ID] {
            let (lower, upper, [turb, div, gen_col]) = run_fills(&fixtures, stage_id);
            assert_eq!(lower[turb], 0.0, "turbine col_lower at {stage_id}");
            assert_eq!(
                upper[turb], MAX_TURBINED_M3S,
                "turbine col_upper at {stage_id}"
            );
            assert_eq!(lower[gen_col], 0.0, "generation col_lower at {stage_id}");
            assert_eq!(
                upper[gen_col], MAX_GENERATION_MW,
                "generation col_upper at {stage_id}"
            );
            for blk in 0..2 {
                let col = div + blk;
                assert_eq!(
                    lower[col], 0.0,
                    "diversion col_lower blk {blk} at {stage_id}"
                );
                assert_eq!(
                    upper[col], MAX_DIVERSION_M3S,
                    "diversion col_upper blk {blk} at {stage_id}"
                );
            }
        }
    }

    /// A commissioning-dormant non-filling hydro (`filling = None`,
    /// `entry = Some(4)`) is `PreFilling` before entry: turbine and diversion
    /// columns are `[0, 0]`, and the FPHA generation column is omitted from the
    /// dense block (`identify_fpha_hydros` drops it), exactly like a filling hydro's
    /// `PreFilling` stage.
    #[test]
    fn dormant_non_filling_hydro_zeroed_before_entry() {
        let fixtures = Fixtures::new(None, Some(ENTRY_STAGE_ID), true);
        for stage_id in [PREFILLING_ID, FILLING_ID] {
            let (lower, upper, [turb, div, gen_col]) = run_fills(&fixtures, stage_id);
            assert_eq!(lower[turb], 0.0, "turbine col_lower at stage {stage_id}");
            assert_eq!(upper[turb], 0.0, "turbine col_upper at stage {stage_id}");
            for blk in 0..2 {
                let col = div + blk;
                assert_eq!(
                    lower[col], 0.0,
                    "diversion col_lower blk {blk} at {stage_id}"
                );
                assert_eq!(
                    upper[col], 0.0,
                    "dormant diversion col_upper blk {blk} at {stage_id}"
                );
            }
            assert_eq!(
                gen_col,
                usize::MAX,
                "dormant non-filling hydro has no FPHA generation column before entry (stage {stage_id})"
            );
        }
    }

    /// From `entry` onward a commissioning-dormant non-filling hydro is `Operating`
    /// with NO intervening `Filling` phase: turbine, generation, and diversion
    /// return to their normal bounds at the first commissioned stage.
    #[test]
    fn dormant_non_filling_hydro_normal_from_entry() {
        let fixtures = Fixtures::new(None, Some(ENTRY_STAGE_ID), true);
        let (lower, upper, [turb, div, gen_col]) = run_fills(&fixtures, OPERATING_ID);
        assert_eq!(lower[turb], 0.0, "turbine col_lower");
        assert_eq!(upper[turb], MAX_TURBINED_M3S, "turbine col_upper");
        assert_eq!(lower[gen_col], 0.0, "generation col_lower");
        assert_eq!(upper[gen_col], MAX_GENERATION_MW, "generation col_upper");
        for blk in 0..2 {
            let col = div + blk;
            assert_eq!(
                upper[col], MAX_DIVERSION_M3S,
                "diversion col_upper blk {blk}"
            );
        }
    }

    // A non-zero dead volume so the storage-floor relax (floor → 0) is observable
    // against the hard `min_storage` floor.
    const MIN_STORAGE_HM3: f64 = 50.0;
    const MAX_STORAGE_HM3: f64 = 200.0;

    /// Run `fill_storage_columns` against the fixture at `stage_id` with the
    /// resolved dead volume overridden to `MIN_STORAGE_HM3`, returning
    /// `(col_lower[0], col_upper[0])` — the outgoing-storage column is always
    /// system index `0`.
    fn run_storage_fill(fixtures: &mut Fixtures, stage_id: i32) -> (f64, f64) {
        fixtures
            .bounds
            .hydro_bounds_mut(0, STAGE_IDX)
            .min_storage_hm3 = MIN_STORAGE_HM3;
        fixtures
            .bounds
            .hydro_bounds_mut(0, STAGE_IDX)
            .max_storage_hm3 = MAX_STORAGE_HM3;
        let stage_index = usize::try_from(stage_id).expect("test stage ids are non-negative");
        let stage = two_block_stage(stage_index, [372.0, 372.0]);
        let ctx = fixtures.make_ctx();
        let state = state_layout_for(&ctx);
        let layout = StageLayout::new(&ctx, &state, &stage, STAGE_IDX);
        let (mut col_lower, mut col_upper, mut objective) = fresh_bufs(layout.num_cols);
        let mut bufs = ColumnBufs {
            col_lower: &mut col_lower,
            col_upper: &mut col_upper,
            objective: &mut objective,
        };
        super::fill_storage_columns(&ctx, &stage, STAGE_IDX, &layout, &mut bufs);
        (col_lower[0], col_upper[0])
    }

    /// A filling hydro relaxes its storage FLOOR to `0` in `PreFilling` and
    /// `Filling` (the reservoir may sit below dead volume while filling); the
    /// upper bound stays `max_storage` in every phase. The forbidden alternative —
    /// keeping the hard `min_storage` floor — would make the LP infeasible whenever
    /// the filling reservoir is below dead volume.
    #[test]
    fn filling_hydro_storage_floor_relaxed_in_prefilling_and_filling() {
        for stage_id in [PREFILLING_ID, FILLING_ID] {
            let mut fixtures = Fixtures::new(Some(filling_config()), Some(ENTRY_STAGE_ID), false);
            let (lower, upper) = run_storage_fill(&mut fixtures, stage_id);
            assert_eq!(lower, 0.0, "storage col_lower relaxed to 0 at {stage_id}");
            assert_eq!(
                upper, MAX_STORAGE_HM3,
                "storage col_upper stays max_storage at {stage_id}"
            );
        }
    }

    /// A filling hydro relaxes its storage FLOOR to `0` in `Operating` too — the
    /// soft `σ^{v-}` operating-floor row supplies the economic floor so a hydro that
    /// finished filling short can recover without infeasibility. Combined with the
    /// `PreFilling`/`Filling` relax above, a filling hydro has `col_lower = 0` in ALL
    /// phases. The upper bound stays `max_storage`. The forbidden alternative —
    /// keeping the hard `min_storage` floor in Operating — would make the LP
    /// infeasible the moment a deficient-start reservoir sits below dead volume.
    #[test]
    fn filling_hydro_storage_floor_relaxed_in_operating() {
        let mut fixtures = Fixtures::new(Some(filling_config()), Some(ENTRY_STAGE_ID), false);
        let (lower, upper) = run_storage_fill(&mut fixtures, OPERATING_ID);
        assert_eq!(lower, 0.0, "storage col_lower relaxed to 0 in Operating");
        assert_eq!(upper, MAX_STORAGE_HM3, "storage col_upper in Operating");
    }

    /// A non-filling hydro keeps the hard `min_storage` floor at EVERY stage id
    /// (the parity-neutrality contract — the floor relax never fires). The
    /// forbidden alternative — relaxing the floor system-wide — would silently make
    /// dead volume soft for every reservoir.
    #[test]
    fn non_filling_hydro_storage_floor_hard_at_every_stage() {
        for stage_id in [PREFILLING_ID, FILLING_ID, OPERATING_ID] {
            let mut fixtures = Fixtures::new(None, None, false);
            let (lower, upper) = run_storage_fill(&mut fixtures, stage_id);
            assert_eq!(
                lower, MIN_STORAGE_HM3,
                "non-filling storage col_lower hard at {stage_id}"
            );
            assert_eq!(
                upper, MAX_STORAGE_HM3,
                "non-filling storage col_upper at {stage_id}"
            );
        }
    }

    /// A commissioning-dormant non-filling hydro relaxes its storage FLOOR to `0`
    /// while `PreFilling` (the frozen-identity row pins `v_h` to the inert IC
    /// storage, which a hard `min_storage` floor would reject), then RESTORES the
    /// hard `min_storage` floor from `entry` onward (`Operating` — a normal plant
    /// with no soft operating-floor row, unlike a filling hydro). The forbidden
    /// alternative — keeping the relax at `Operating` — would silently make this
    /// plant's dead volume soft once it commissions.
    #[test]
    fn dormant_non_filling_hydro_storage_floor_relaxed_then_restored() {
        for stage_id in [PREFILLING_ID, FILLING_ID] {
            let mut fixtures = Fixtures::new(None, Some(ENTRY_STAGE_ID), false);
            let (lower, upper) = run_storage_fill(&mut fixtures, stage_id);
            assert_eq!(
                lower, 0.0,
                "dormant storage col_lower relaxed to 0 at {stage_id}"
            );
            assert_eq!(upper, MAX_STORAGE_HM3, "storage col_upper at {stage_id}");
        }
        let mut fixtures = Fixtures::new(None, Some(ENTRY_STAGE_ID), false);
        let (lower, upper) = run_storage_fill(&mut fixtures, OPERATING_ID);
        assert_eq!(
            lower, MIN_STORAGE_HM3,
            "storage col_lower restored to hard min_storage at Operating"
        );
        assert_eq!(upper, MAX_STORAGE_HM3, "storage col_upper at Operating");
    }

    // ── σ_fill terminal-target column ────────────────────────────────────────

    /// A representative non-default `filling_target_violation_cost` so the
    /// objective coefficient is observable against the `0.0` default.
    const FILLING_TARGET_COST: f64 = 50_000.0;

    /// Run `fill_filling_target_columns` against the fixture at `stage_id` with the
    /// resolved `filling_target_violation_cost` overridden to
    /// `FILLING_TARGET_COST`, returning the whole column buffers and the `σ_fill`
    /// column start (which equals `num_cols` when the block is empty).
    fn run_filling_target_fill(
        fixtures: &mut Fixtures,
        stage_id: i32,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>, usize, usize, usize) {
        fixtures
            .penalties
            .hydro_penalties_mut(0, STAGE_IDX)
            .filling_target_violation_cost = FILLING_TARGET_COST;
        let stage_index = usize::try_from(stage_id).expect("test stage ids are non-negative");
        let stage = two_block_stage(stage_index, [372.0, 372.0]);
        let ctx = fixtures.make_ctx();
        let state = state_layout_for(&ctx);
        let layout = StageLayout::new(&ctx, &state, &stage, STAGE_IDX);
        let (mut col_lower, mut col_upper, mut objective) = fresh_bufs(layout.num_cols);
        let mut bufs = ColumnBufs {
            col_lower: &mut col_lower,
            col_upper: &mut col_upper,
            objective: &mut objective,
        };
        super::fill_filling_target_columns(&ctx, STAGE_IDX, &layout, &mut bufs);
        let n_targets = layout.filling.filling_target_hydro_indices.len();
        let col_start = layout.filling.col_filling_target_start;
        (
            col_lower,
            col_upper,
            objective,
            col_start,
            n_targets,
            layout.num_cols,
        )
    }

    /// At a Filling stage (every `id` in `[start, entry)`) a filling hydro gets
    /// exactly ONE `σ_fill` column: `col_lower = 0`, `col_upper = +∞`, and the
    /// objective coefficient is the RESOLVED `filling_target_violation_cost`
    /// UNSCALED — NOT multiplied by stage hours. The hours-multiplication
    /// alternative (copied from the flow/power-rate slacks) would be a $·h/hm³ units
    /// error on this storage-volume ($/hm³) penalty. Exercises BOTH Filling stages
    /// (ids 2 and 3) to pin per-stage membership, not the v1 terminal-only rule.
    #[test]
    fn filling_target_column_emitted_at_every_filling_stage() {
        // start = 2, entry = 4 ⇒ Filling stages are ids 2 and 3.
        for stage_id in [START_STAGE_ID, ENTRY_STAGE_ID - 1] {
            let mut fixtures = Fixtures::new(Some(filling_config()), Some(ENTRY_STAGE_ID), false);
            let (col_lower, col_upper, objective, col_start, n_targets, _num_cols) =
                run_filling_target_fill(&mut fixtures, stage_id);
            assert_eq!(
                n_targets, 1,
                "exactly one σ_fill column at Filling id {stage_id}"
            );
            let col = col_start;
            assert_eq!(col_lower[col], 0.0, "σ_fill col_lower = 0 at id {stage_id}");
            assert_eq!(
                col_upper[col],
                f64::INFINITY,
                "σ_fill col_upper = +∞ at id {stage_id}"
            );
            // Cost is UNSCALED here (the global /COST_SCALE_FACTOR pass runs later in
            // build_single_stage_template) and carries NO hours factor.
            assert_eq!(
                objective[col], FILLING_TARGET_COST,
                "σ_fill objective = filling_target_violation_cost (unscaled, no hours) at id {stage_id}"
            );
        }
    }

    /// No `σ_fill` column is emitted off the Filling phase — `PreFilling` (id 1) or
    /// `Operating` (id 4). Per-stage Filling membership, NOT the v1 terminal-only
    /// rule.
    ///
    /// At `PreFilling` the `σ_fill` block being empty means its cursor coincides
    /// with `num_cols` (`σ_fill` is the last occupied family there, since `σ^{v-}`
    /// is also empty off `Operating`). At `Operating` the `σ^{v-}` block
    /// legitimately occupies one column AFTER the (empty) `σ_fill` block, so the
    /// faithful empty-`σ_fill` check is the zero block width (`n_targets == 0`), not
    /// the `== num_cols` coincidence — which `σ^{v-}` breaks at `Operating`.
    #[test]
    fn filling_target_column_absent_off_filling_phase() {
        for stage_id in [PREFILLING_ID, OPERATING_ID] {
            let mut fixtures = Fixtures::new(Some(filling_config()), Some(ENTRY_STAGE_ID), false);
            let (_, _, _, col_start, n_targets, num_cols) =
                run_filling_target_fill(&mut fixtures, stage_id);
            assert_eq!(n_targets, 0, "no σ_fill column at id {stage_id}");
            if stage_id == OPERATING_ID {
                // σ^{v-} occupies exactly one column beyond the empty σ_fill block.
                assert_eq!(
                    col_start + 1,
                    num_cols,
                    "Operating: σ^{{v-}} column sits after the empty σ_fill block"
                );
            } else {
                assert_eq!(
                    col_start, num_cols,
                    "empty σ_fill block: col start coincides with num_cols at id {stage_id}"
                );
            }
        }
    }

    /// A non-filling hydro never gets a `σ_fill` column at any stage id
    /// (parity-neutral): the index list is empty and the column buffers are
    /// untouched by the fill.
    #[test]
    fn non_filling_hydro_no_filling_target_column() {
        for stage_id in [PREFILLING_ID, FILLING_ID, OPERATING_ID] {
            let mut fixtures = Fixtures::new(None, None, false);
            let (col_lower, col_upper, objective, col_start, n_targets, num_cols) =
                run_filling_target_fill(&mut fixtures, stage_id);
            assert_eq!(
                n_targets, 0,
                "non-filling: no σ_fill column at id {stage_id}"
            );
            assert_eq!(col_start, num_cols, "empty σ_fill block at id {stage_id}");
            // The fill wrote nothing: objective is all-zero, bounds are the fresh
            // `[0, +∞]` defaults (no column carries the penalty).
            assert!(
                objective.iter().all(|&c| c == 0.0),
                "non-filling: no objective entry written by σ_fill fill"
            );
            assert!(col_lower.iter().all(|&l| l == 0.0));
            assert!(col_upper.iter().all(|&u| u == f64::INFINITY));
        }
    }

    // ── σ^{v-} operating-floor column ────────────────────────────────────────

    /// A representative non-default `storage_violation_below_cost` so the objective
    /// coefficient is observable against the `0.0` default, and DISTINCT from
    /// `FILLING_TARGET_COST` so a test that conflates the two costs fails.
    const STORAGE_BELOW_COST: f64 = 12_345.0;

    /// Run `fill_filled_min_storage_floor_columns` against the fixture at `stage_id` with the
    /// resolved `storage_violation_below_cost` overridden to `STORAGE_BELOW_COST`,
    /// returning the column buffers and the `σ^{v-}` column start (which equals
    /// `num_cols` when the block is empty), the column count, and `num_cols`.
    fn run_filled_min_storage_floor_fill(
        fixtures: &mut Fixtures,
        stage_id: i32,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>, usize, usize, usize) {
        fixtures
            .penalties
            .hydro_penalties_mut(0, STAGE_IDX)
            .storage_violation_below_cost = STORAGE_BELOW_COST;
        let stage_index = usize::try_from(stage_id).expect("test stage ids are non-negative");
        let stage = two_block_stage(stage_index, [372.0, 372.0]);
        let ctx = fixtures.make_ctx();
        let state = state_layout_for(&ctx);
        let layout = StageLayout::new(&ctx, &state, &stage, STAGE_IDX);
        let (mut col_lower, mut col_upper, mut objective) = fresh_bufs(layout.num_cols);
        let mut bufs = ColumnBufs {
            col_lower: &mut col_lower,
            col_upper: &mut col_upper,
            objective: &mut objective,
        };
        super::fill_filled_min_storage_floor_columns(&ctx, STAGE_IDX, &layout, &mut bufs);
        let n_floors = layout.filling.filled_min_storage_floor_hydro_indices.len();
        let col_start = layout.filling.col_filled_min_storage_floor_start;
        (
            col_lower,
            col_upper,
            objective,
            col_start,
            n_floors,
            layout.num_cols,
        )
    }

    /// In `Operating` (`id == entry == 4`) a filling hydro gets exactly ONE
    /// `σ^{v-}` column: `col_lower = 0`, `col_upper = +∞`, and the objective
    /// coefficient is the RESOLVED `storage_violation_below_cost` UNSCALED — NOT
    /// multiplied by stage hours (the hours-multiplication alternative copied from
    /// the flow/power-rate slacks would be a $·h/hm³ units error on this
    /// storage-volume ($/hm³) penalty, identical to the `σ_fill` convention).
    #[test]
    fn filled_min_storage_floor_column_emitted_in_operating() {
        let mut fixtures = Fixtures::new(Some(filling_config()), Some(ENTRY_STAGE_ID), false);
        let (col_lower, col_upper, objective, col_start, n_floors, _num_cols) =
            run_filled_min_storage_floor_fill(&mut fixtures, OPERATING_ID);
        assert_eq!(n_floors, 1, "exactly one σ^{{v-}} column in Operating");
        let col = col_start;
        assert_eq!(col_lower[col], 0.0, "σ^{{v-}} col_lower = 0");
        assert_eq!(col_upper[col], f64::INFINITY, "σ^{{v-}} col_upper = +∞");
        // Cost is UNSCALED here (the global /COST_SCALE_FACTOR pass runs later in
        // build_single_stage_template) and carries NO hours factor.
        assert_eq!(
            objective[col], STORAGE_BELOW_COST,
            "σ^{{v-}} objective = storage_violation_below_cost (unscaled, no hours)"
        );
    }

    /// No `σ^{v-}` column is emitted at any non-operating stage of a filling hydro —
    /// `PreFilling` or `Filling`. `Operating`-only (the complement of `σ_fill`'s
    /// every-Filling-stage scope), so the `σ^{v-}` family never collides with `σ_fill`.
    #[test]
    fn filled_min_storage_floor_column_absent_in_prefilling_and_filling() {
        for stage_id in [PREFILLING_ID, FILLING_ID] {
            let mut fixtures = Fixtures::new(Some(filling_config()), Some(ENTRY_STAGE_ID), false);
            let (_, _, _, col_start, n_floors, num_cols) =
                run_filled_min_storage_floor_fill(&mut fixtures, stage_id);
            assert_eq!(n_floors, 0, "no σ^{{v-}} column at id {stage_id}");
            assert_eq!(
                col_start, num_cols,
                "empty σ^{{v-}} block: col start coincides with num_cols at id {stage_id}"
            );
        }
    }

    /// A non-filling hydro never gets a `σ^{v-}` column at any stage id
    /// (parity-neutral): the index list is empty and the column buffers are
    /// untouched by the fill. The forbidden GLOBAL soft floor — matching every
    /// Operating hydro regardless of `filling` — would softly floor every reservoir.
    #[test]
    fn non_filling_hydro_no_filled_min_storage_floor_column() {
        for stage_id in [PREFILLING_ID, FILLING_ID, OPERATING_ID] {
            let mut fixtures = Fixtures::new(None, None, false);
            let (col_lower, col_upper, objective, col_start, n_floors, num_cols) =
                run_filled_min_storage_floor_fill(&mut fixtures, stage_id);
            assert_eq!(
                n_floors, 0,
                "non-filling: no σ^{{v-}} column at id {stage_id}"
            );
            assert_eq!(col_start, num_cols, "empty σ^{{v-}} block at id {stage_id}");
            assert!(
                objective.iter().all(|&c| c == 0.0),
                "non-filling: no objective entry written by σ^{{v-}} fill"
            );
            assert!(col_lower.iter().all(|&l| l == 0.0));
            assert!(col_upper.iter().all(|&u| u == f64::INFINITY));
        }
    }

    /// The `σ^{v-}` (`Operating`) and `σ_fill` (`Filling`) families are MUTUALLY
    /// EXCLUSIVE per stage: at the LAST Filling stage (`entry − 1`, the boundary
    /// witness) a filling hydro has `σ_fill` but NOT `σ^{v-}`; at the first
    /// `Operating` stage (`entry`) it has `σ^{v-}` but NOT `σ_fill`. Two separate
    /// columns, two non-overlapping stage scopes (Filling vs Operating) — never
    /// conflated.
    #[test]
    fn filled_min_storage_floor_and_filling_target_are_mutually_exclusive_by_stage() {
        // Last Filling stage (entry − 1): σ_fill present, σ^{v-} absent.
        let mut f_terminal = Fixtures::new(Some(filling_config()), Some(ENTRY_STAGE_ID), false);
        let (_, _, _, _, n_targets_terminal, _) =
            run_filling_target_fill(&mut f_terminal, ENTRY_STAGE_ID - 1);
        let mut f_terminal2 = Fixtures::new(Some(filling_config()), Some(ENTRY_STAGE_ID), false);
        let (_, _, _, _, n_floors_terminal, _) =
            run_filled_min_storage_floor_fill(&mut f_terminal2, ENTRY_STAGE_ID - 1);
        assert_eq!(n_targets_terminal, 1, "σ_fill present at terminal stage");
        assert_eq!(n_floors_terminal, 0, "σ^{{v-}} absent at terminal stage");

        // Operating stage (entry): σ^{v-} present, σ_fill absent.
        let mut f_op = Fixtures::new(Some(filling_config()), Some(ENTRY_STAGE_ID), false);
        let (_, _, _, _, n_targets_op, _) = run_filling_target_fill(&mut f_op, OPERATING_ID);
        let mut f_op2 = Fixtures::new(Some(filling_config()), Some(ENTRY_STAGE_ID), false);
        let (_, _, _, _, n_floors_op, _) =
            run_filled_min_storage_floor_fill(&mut f_op2, OPERATING_ID);
        assert_eq!(n_targets_op, 0, "σ_fill absent in Operating");
        assert_eq!(n_floors_op, 1, "σ^{{v-}} present in Operating");
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::float_cmp,
    clippy::similar_names
)]
mod anticipated_objective_tests {
    use std::collections::{BTreeMap, HashMap};

    use cobre_core::entities::thermal::AnticipatedConfig;
    use cobre_core::{
        BoundsCountsSpec, BoundsDefaults, CascadeTopology, ContractBlockBounds, EntityId,
        HorizonGraph, HydroBlockBounds, HydroStageBounds, LineBlockBounds, PostStudyStage,
        PostStudyStages, PostStudyThermalBound, PumpingBlockBounds, ResolvedBounds,
        ResolvedGenericConstraintBounds, ResolvedLoadFactors, ResolvedNcsBounds,
        ResolvedNcsFactors, ResolvedPenalties, Thermal, ThermalBlockBounds, ThermalStageBounds,
    };
    use cobre_stochastic::par::precompute::PrecomputedPar;

    use crate::hydro_models::{EvaporationModelSet, ProductionModelSet};
    use crate::lead_time::{AnticipatedResolution, DeliveryAxis, LeadTime};
    use crate::resolved_parameters::ResolvedParameters;
    use crate::setup::PostStudyResolved;

    use super::super::layout::ResolvedTables;
    use super::super::test_support::{
        state_layout_for, state_layout_with_resolution, two_block_stage,
    };
    use super::{StageLayout, TemplateBuildCtx, fill_stage_columns};
    use crate::indexer::{HydroCellIndex, ThermalSys};

    const N_STAGES: usize = 6;
    const K_MAX: usize = 1;
    const DELIVERY_COST_PER_MWH: f64 = 30.0;
    const STD_COST_PER_MWH: f64 = 40.0;
    const MAX_GEN_MW: f64 = 100.0;
    const STAGE_IDX: usize = 0;
    const DELIVERY_STAGE: usize = STAGE_IDX + 1; // K_0 = 1.

    /// Owns the borrow targets for a one-anticipated-thermal `TemplateBuildCtx`.
    ///
    /// Thermal 0 is anticipated (`K=1`), thermal 1 is a standard thermal. Both
    /// carry a non-zero resolved `cost_per_mwh` so the skipped delivery objective
    /// and the NPV-priced decision column are both observable in the assertions.
    struct AntObjFixtures {
        par_lp: PrecomputedPar,
        thermals: Vec<Thermal>,
        hydro_cell_index: HydroCellIndex,
        cascade: CascadeTopology,
        bounds: ResolvedBounds,
        penalties: ResolvedPenalties,
        production_models: ProductionModelSet,
        evaporation_models: EvaporationModelSet,
        resolved_generic_bounds: ResolvedGenericConstraintBounds,
        resolved_load_factors: ResolvedLoadFactors,
        resolved_ncs_bounds: ResolvedNcsBounds,
        resolved_ncs_factors: ResolvedNcsFactors,
        resolved_parameters: ResolvedParameters,
    }

    impl AntObjFixtures {
        fn new() -> Self {
            let thermals = vec![
                Thermal {
                    id: EntityId(1),
                    name: "T_ant".to_string(),
                    operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                    bus_id: EntityId(1),
                    min_generation_mw: 0.0,
                    max_generation_mw: MAX_GEN_MW,
                    cost_per_mwh: DELIVERY_COST_PER_MWH,
                    // lead_stages == K_MAX; the entity field is u32 while K_MAX
                    // is the usize layout dimension, so write the value directly.
                    anticipated_config: Some(AnticipatedConfig::LeadStages(1)),
                    entry_stage_id: None,
                    exit_stage_id: None,
                },
                Thermal {
                    id: EntityId(2),
                    name: "T_std".to_string(),
                    operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                    bus_id: EntityId(1),
                    min_generation_mw: 0.0,
                    max_generation_mw: MAX_GEN_MW,
                    cost_per_mwh: STD_COST_PER_MWH,
                    anticipated_config: None,
                    entry_stage_id: None,
                    exit_stage_id: None,
                },
            ];
            let mut bounds = bounds_two_thermals();
            for stage in 0..N_STAGES {
                bounds.thermal_bounds_mut(0, stage).cost_per_mwh = DELIVERY_COST_PER_MWH;
                bounds.thermal_block_base_mut(0, stage).max_generation_mw = MAX_GEN_MW;
                bounds.thermal_bounds_mut(1, stage).cost_per_mwh = STD_COST_PER_MWH;
                bounds.thermal_block_base_mut(1, stage).max_generation_mw = MAX_GEN_MW;
            }
            Self {
                par_lp: PrecomputedPar::default(),
                thermals,
                cascade: CascadeTopology::build(&[]),
                hydro_cell_index: HydroCellIndex::build(&[]),
                bounds,
                penalties: ResolvedPenalties::empty(),
                production_models: ProductionModelSet::new(vec![], 0, 1),
                evaporation_models: EvaporationModelSet::new(vec![]),
                resolved_generic_bounds: ResolvedGenericConstraintBounds::empty(),
                resolved_load_factors: ResolvedLoadFactors::empty(),
                resolved_ncs_bounds: ResolvedNcsBounds::empty(),
                resolved_ncs_factors: ResolvedNcsFactors::empty(),
                resolved_parameters: ResolvedParameters {
                    per_param: vec![],
                    id_to_slot: vec![],
                    cost_scale_factor: 1_000_000.0,
                },
            }
        }

        fn make_ctx(&self) -> TemplateBuildCtx<'_> {
            TemplateBuildCtx {
                hydros: &[],
                thermals: &self.thermals,
                lines: &[],
                buses: &[],
                load_models: &[],
                cascade: &self.cascade,
                hydro_cell_index: &self.hydro_cell_index,
                resolved: ResolvedTables {
                    bounds: &self.bounds,
                    penalties: &self.penalties,
                    resolved_generic_bounds: &self.resolved_generic_bounds,
                    resolved_load_factors: &self.resolved_load_factors,
                    resolved_ncs_bounds: &self.resolved_ncs_bounds,
                    resolved_ncs_factors: &self.resolved_ncs_factors,
                    resolved_parameters: &self.resolved_parameters,
                },
                hydro_pos: BTreeMap::new(),
                thermal_pos: BTreeMap::new(),
                line_pos: BTreeMap::new(),
                bus_pos: BTreeMap::new(),
                par_lp: &self.par_lp,
                production_models: &self.production_models,
                evaporation_models: &self.evaporation_models,
                generic_constraints: &[],
                non_controllable_sources: &[],
                pumping_stations: &[],
                pumping_pos: BTreeMap::new(),
                n_pumping: 0,
                contracts: &[],
                contract_pos: BTreeMap::new(),
                n_contract_import: 0,
                n_contract_export: 0,
                diversion_upstream: HashMap::new(),
                arc_stage_weights: HashMap::new(),
                arc_spread_chrono: HashMap::new(),
                arc_arrival_density: HashMap::new(),
                per_stage_mask: Vec::new(),
                post_study_resolved: crate::setup::PostStudyResolved::default(),
                n_hydros: 0,
                n_thermals: 2,
                n_lines: 0,
                n_buses: 0,
                max_par_order: 0,
                n_anticipated: 1,
                k_max: K_MAX,
                anticipated_lead_stages: vec![K_MAX],
                anticipated_thermal_indices: vec![ThermalSys::new(0)],
                // Windowless single plant: the decision gate reduces to the
                // strict horizon clause. `study_stage_ids` lists the N_STAGES
                // study-stage ids so the in-range delivery lookup is safe.
                anticipated_windows: vec![(None, None)],
                anticipated_resolution: AnticipatedResolution::default(),
                study_stage_ids: (0..N_STAGES as i32).collect(),
                delivery_stage_ids: (0..N_STAGES as i32).collect(),
                has_penalty: false,
                delivery_cumulative_discount_factors: vec![1.0, 0.9, 0.81, 0.729, 0.6561, 0.59049],
                delivery_total_hours: vec![744.0; N_STAGES],
                filling_v_target: BTreeMap::new(),
            }
        }
    }

    /// `ResolvedBounds` table for two thermals across `N_STAGES` stages.
    fn bounds_two_thermals() -> ResolvedBounds {
        ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 0,
                n_thermals: 2,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages: N_STAGES,
                k_max: K_MAX,
            },
            &BoundsDefaults {
                hydro: HydroStageBounds {
                    min_storage_hm3: 0.0,
                    max_storage_hm3: 0.0,
                    filling_min_rate_m3s: 0.0,
                    water_withdrawal_m3s: 0.0,
                },
                hydro_block: HydroBlockBounds {
                    ..Default::default()
                },
                thermal: ThermalStageBounds { cost_per_mwh: 0.0 },
                thermal_block: ThermalBlockBounds {
                    min_generation_mw: 0.0,
                    max_generation_mw: 0.0,
                },
                line_block: LineBlockBounds {
                    direct_mw: 0.0,
                    reverse_mw: 0.0,
                },
                pumping_block: PumpingBlockBounds {
                    min_flow_m3s: 0.0,
                    max_flow_m3s: 0.0,
                },
                contract_block: ContractBlockBounds {
                    min_mw: 0.0,
                    max_mw: 0.0,
                    price_per_mwh: 0.0,
                },
            },
        )
    }

    /// After `fill_stage_columns`, the anticipated thermal's per-block delivery
    /// objective is `0.0` (`fill_thermal_columns` skips the objective write),
    /// while the standard thermal is priced normally; and the anticipated
    /// decision column carries the NPV-discounted commitment cost
    /// (`fill_anticipated_columns` writes `cost * hours * cumulative_discount`).
    #[test]
    fn anticipated_objective_skip_and_npv_after_fill_stage_columns() {
        let fixtures = AntObjFixtures::new();
        let ctx = fixtures.make_ctx();
        let stage = two_block_stage(STAGE_IDX, [372.0, 372.0]);
        let state = state_layout_for(&ctx);
        let layout = StageLayout::new(&ctx, &state, &stage, STAGE_IDX);

        let (_col_lower, col_upper, objective) =
            fill_stage_columns(&ctx, &stage, STAGE_IDX, &layout);

        let n_blks = layout.n_blks;
        // Anticipated thermal (t_idx 0) objective stays at the 0.0 default; its
        // per-block bounds are still written by fill_thermal_columns.
        for blk in 0..n_blks {
            let col = layout.equipment.thermal.start + blk;
            assert_eq!(
                objective[col], 0.0,
                "anticipated thermal objective must be 0.0 at col {col}",
            );
            assert_eq!(
                col_upper[col], MAX_GEN_MW,
                "anticipated thermal per-block bounds must still be set at col {col}",
            );
        }
        // Control: standard thermal (t_idx 1) is priced as cost * block_hours.
        for blk in 0..n_blks {
            let col = layout.equipment.thermal.start + n_blks + blk;
            let expected = STD_COST_PER_MWH * stage.blocks[blk].duration_hours;
            assert_eq!(
                objective[col], expected,
                "standard thermal objective must be priced at col {col}",
            );
        }
        // The anticipated decision column carries the NPV commitment cost
        // cost_per_mwh(delivery) * total_hours[delivery] * cumulative_discount[delivery].
        let decision_col = layout.anticipated.col_anticipated_decision_start;
        let expected_npv = DELIVERY_COST_PER_MWH
            * ctx.delivery_total_hours[DELIVERY_STAGE]
            * ctx.delivery_cumulative_discount_factors[DELIVERY_STAGE];
        assert_eq!(
            objective[decision_col], expected_npv,
            "anticipated decision objective must equal the NPV commitment cost",
        );
        // The active plant's newest ring slot is open (active), confirming the
        // merged fill ran the active branch. K_MAX == 1 here, so the newest
        // slot is the ring's own start (no per-plant offset needed).
        let state_out_col = layout.anticipated.col_anticipated_slots_out_start;
        assert_eq!(col_upper[state_out_col], f64::INFINITY);
    }

    /// Borrow-target owner for a one-anticipated-plant delivery-anchoring
    /// preservation `TemplateBuildCtx`. `per_stage[s] == (min_gen, max_gen,
    /// cost)` is the plant's stage-`s` `thermal_bounds`; the discount factor is
    /// stage-varying (`0.9^s`) too. Both are deliberately stage-varying so a
    /// DECISION-anchored read (the forbidden alternative) yields a provably
    /// different column than the shipped DELIVERY-anchored read.
    struct DeliveryAnchoredFixtures {
        thermals: Vec<Thermal>,
        hydro_cell_index: HydroCellIndex,
        cascade: CascadeTopology,
        bounds: ResolvedBounds,
        par_lp: PrecomputedPar,
        penalties: ResolvedPenalties,
        production_models: ProductionModelSet,
        evaporation_models: EvaporationModelSet,
        resolved_generic_bounds: ResolvedGenericConstraintBounds,
        resolved_load_factors: ResolvedLoadFactors,
        resolved_ncs_bounds: ResolvedNcsBounds,
        resolved_ncs_factors: ResolvedNcsFactors,
        resolved_parameters: ResolvedParameters,
        n_stages: usize,
        k_max: usize,
        discount: Vec<f64>,
        hours: Vec<f64>,
    }

    impl DeliveryAnchoredFixtures {
        fn new(
            n_stages: usize,
            k_max: usize,
            config: AnticipatedConfig,
            per_stage: &[(f64, f64, f64)],
        ) -> Self {
            let thermals = vec![Thermal {
                id: EntityId(1),
                name: "T_ant".to_string(),
                operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                bus_id: EntityId(1),
                min_generation_mw: 0.0,
                max_generation_mw: 100.0,
                cost_per_mwh: 0.0,
                anticipated_config: Some(config),
                entry_stage_id: None,
                exit_stage_id: None,
            }];
            let mut bounds = ResolvedBounds::new(
                &BoundsCountsSpec {
                    n_hydros: 0,
                    n_thermals: 1,
                    n_lines: 0,
                    n_pumping: 0,
                    n_contracts: 0,
                    n_stages,
                    k_max,
                },
                &BoundsDefaults {
                    hydro: HydroStageBounds {
                        min_storage_hm3: 0.0,
                        max_storage_hm3: 0.0,
                        filling_min_rate_m3s: 0.0,
                        water_withdrawal_m3s: 0.0,
                    },
                    hydro_block: HydroBlockBounds {
                        ..Default::default()
                    },
                    thermal: ThermalStageBounds { cost_per_mwh: 0.0 },
                    thermal_block: ThermalBlockBounds {
                        min_generation_mw: 0.0,
                        max_generation_mw: 0.0,
                    },
                    line_block: LineBlockBounds {
                        direct_mw: 0.0,
                        reverse_mw: 0.0,
                    },
                    pumping_block: PumpingBlockBounds {
                        min_flow_m3s: 0.0,
                        max_flow_m3s: 0.0,
                    },
                    contract_block: ContractBlockBounds {
                        min_mw: 0.0,
                        max_mw: 0.0,
                        price_per_mwh: 0.0,
                    },
                },
            );
            for (stage, &(min_g, max_g, cost)) in per_stage.iter().enumerate() {
                let tbb = bounds.thermal_block_base_mut(0, stage);
                tbb.min_generation_mw = min_g;
                tbb.max_generation_mw = max_g;
                bounds.thermal_bounds_mut(0, stage).cost_per_mwh = cost;
            }
            let mut discount = Vec::with_capacity(n_stages);
            let mut d = 1.0_f64;
            for _ in 0..n_stages {
                discount.push(d);
                d *= 0.9;
            }
            Self {
                thermals,
                cascade: CascadeTopology::build(&[]),
                hydro_cell_index: HydroCellIndex::build(&[]),
                bounds,
                par_lp: PrecomputedPar::default(),
                penalties: ResolvedPenalties::empty(),
                production_models: ProductionModelSet::new(vec![], 0, 1),
                evaporation_models: EvaporationModelSet::new(vec![]),
                resolved_generic_bounds: ResolvedGenericConstraintBounds::empty(),
                resolved_load_factors: ResolvedLoadFactors::empty(),
                resolved_ncs_bounds: ResolvedNcsBounds::empty(),
                resolved_ncs_factors: ResolvedNcsFactors::empty(),
                resolved_parameters: ResolvedParameters {
                    per_param: vec![],
                    id_to_slot: vec![],
                    cost_scale_factor: 1_000_000.0,
                },
                n_stages,
                k_max,
                discount,
                hours: vec![744.0; n_stages],
            }
        }

        fn make_ctx(&self) -> TemplateBuildCtx<'_> {
            TemplateBuildCtx {
                hydros: &[],
                thermals: &self.thermals,
                lines: &[],
                buses: &[],
                load_models: &[],
                cascade: &self.cascade,
                hydro_cell_index: &self.hydro_cell_index,
                resolved: ResolvedTables {
                    bounds: &self.bounds,
                    penalties: &self.penalties,
                    resolved_generic_bounds: &self.resolved_generic_bounds,
                    resolved_load_factors: &self.resolved_load_factors,
                    resolved_ncs_bounds: &self.resolved_ncs_bounds,
                    resolved_ncs_factors: &self.resolved_ncs_factors,
                    resolved_parameters: &self.resolved_parameters,
                },
                hydro_pos: BTreeMap::new(),
                thermal_pos: BTreeMap::new(),
                line_pos: BTreeMap::new(),
                bus_pos: BTreeMap::new(),
                par_lp: &self.par_lp,
                production_models: &self.production_models,
                evaporation_models: &self.evaporation_models,
                generic_constraints: &[],
                non_controllable_sources: &[],
                pumping_stations: &[],
                pumping_pos: BTreeMap::new(),
                n_pumping: 0,
                contracts: &[],
                contract_pos: BTreeMap::new(),
                n_contract_import: 0,
                n_contract_export: 0,
                diversion_upstream: HashMap::new(),
                arc_stage_weights: HashMap::new(),
                arc_spread_chrono: HashMap::new(),
                arc_arrival_density: HashMap::new(),
                per_stage_mask: Vec::new(),
                post_study_resolved: crate::setup::PostStudyResolved::default(),
                n_hydros: 0,
                n_thermals: 1,
                n_lines: 0,
                n_buses: 0,
                max_par_order: 0,
                n_anticipated: 1,
                k_max: self.k_max,
                anticipated_lead_stages: vec![self.k_max],
                anticipated_thermal_indices: vec![ThermalSys::new(0)],
                anticipated_windows: vec![(None, None)],
                anticipated_resolution: AnticipatedResolution::default(),
                study_stage_ids: (0..self.n_stages as i32).collect(),
                delivery_stage_ids: (0..self.n_stages as i32).collect(),
                has_penalty: false,
                delivery_cumulative_discount_factors: self.discount.clone(),
                delivery_total_hours: self.hours.clone(),
                filling_v_target: BTreeMap::new(),
            }
        }
    }

    /// Delivery-anchoring preservation contract: the anticipated decision
    /// column is bounded/costed at ITS OWN delivery stage `m`, never the
    /// decision stage —
    /// `col_upper == thermal_bounds(m).max_generation_mw`,
    /// `col_lower == thermal_bounds(m).min_generation_mw`,
    /// `objective == cost(m) * hours(m) * discount(m)`. STAGE-VARYING delivery
    /// bounds/cost so a decision-anchored read gives a provably different
    /// value (constant-across-lead bounds would make the test vacuous).
    ///
    /// Load-bearing verified by MUTATION: changing the production read at
    /// `fill_anticipated_columns` from `thermal_block_base(thermal_idx,
    /// delivery_stage)` to `thermal_block_base(thermal_idx, stage_idx)` (the
    /// decision stage) fails the first assertion — `left: 55.0, right: 100.0`
    /// on `col_upper[decision]` — reintroducing the capacity-drop
    /// infeasibility this contract forbids.
    #[test]
    fn test_anticipated_decision_delivery_anchored_bounds() {
        // Decision stage 0, delivery stage 1. Stage 0's bounds/cost differ
        // from stage 1's, so a decision-anchored read (stage 0) is
        // distinguishable from the delivery-anchored read.
        const N_STAGES: usize = 6;
        const K_MAX: usize = 1;
        // (min_gen, max_gen, cost) per stage; only stage 1 (delivery) is
        // read by the fill, stage 0 (decision) is the discriminating decoy.
        let per_stage = [
            (5.0, 55.0, 15.0),
            (11.0, 100.0, 30.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
        ];
        let fx = DeliveryAnchoredFixtures::new(
            N_STAGES,
            K_MAX,
            AnticipatedConfig::LeadStages(1),
            &per_stage,
        );
        let ctx = fx.make_ctx();
        let stage = two_block_stage(0, [372.0, 372.0]);
        let state = state_layout_for(&ctx);
        let layout = StageLayout::new(&ctx, &state, &stage, 0);

        let (col_lower, col_upper, objective) = fill_stage_columns(&ctx, &stage, 0, &layout);
        let decision_col = layout.anticipated.col_anticipated_decision_start;

        let delivery = 1_usize;
        let (min_g, max_g, cost) = per_stage[delivery];
        assert_eq!(
            col_upper[decision_col], max_g,
            "decision column must bound at its OWN delivery stage {delivery}'s \
             max_generation_mw, not the decision stage's",
        );
        assert_eq!(
            col_lower[decision_col], min_g,
            "decision column must bound at its OWN delivery stage {delivery}'s \
             min_generation_mw, not the decision stage's",
        );
        let expected_obj = cost
            * ctx.delivery_total_hours[delivery]
            * ctx.delivery_cumulative_discount_factors[delivery];
        assert_eq!(
            objective[decision_col], expected_obj,
            "decision objective must be priced at its OWN delivery stage \
             {delivery}'s cost/hours/discount",
        );
        // The decision stage's own bounds (stage 0) must be strictly
        // different, or the anchoring proof is vacuous.
        let (dec_min, dec_max, _) = per_stage[0];
        assert_ne!(max_g, dec_max, "delivery and decision max_gen must differ");
        assert_ne!(min_g, dec_min, "delivery and decision min_gen must differ");
    }

    const PSA_THERMAL_ID: EntityId = EntityId(7);
    const PSA_N_STAGES: usize = 3;
    const PSA_LEAD: u32 = 2;
    const PSA_STUDY_HOURS: f64 = 730.0;
    /// `(min, max, cost)` at each in-study stage. Only stage 2 (AC1's
    /// delivery target) carries non-default values; stage 1 (AC2-4's
    /// decision stage) is never read by those post-study cases.
    const PSA_IN_STUDY: [(f64, f64, f64); PSA_N_STAGES] =
        [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (5.0, 55.0, 555.0)];

    /// Borrow-target owner for a post-study anchor-read `TemplateBuildCtx`:
    /// one anticipated plant, `LeadTime::Stages(PSA_LEAD)`, `PSA_N_STAGES`
    /// study stages plus a configurable `n_post` post-study stages. The
    /// attached [`AnticipatedResolution`] spans the full extended delivery
    /// axis, so the state must be built through
    /// `state_layout_with_resolution` — the constant-lead fallback
    /// `state_layout_for` leaves active only covers the study-only axis.
    struct PostStudyAnchorFixtures {
        thermals: Vec<Thermal>,
        hydro_cell_index: HydroCellIndex,
        cascade: CascadeTopology,
        bounds: ResolvedBounds,
        par_lp: PrecomputedPar,
        penalties: ResolvedPenalties,
        production_models: ProductionModelSet,
        evaporation_models: EvaporationModelSet,
        resolved_generic_bounds: ResolvedGenericConstraintBounds,
        resolved_load_factors: ResolvedLoadFactors,
        resolved_ncs_bounds: ResolvedNcsBounds,
        resolved_ncs_factors: ResolvedNcsFactors,
        resolved_parameters: ResolvedParameters,
        post_study_resolved: PostStudyResolved,
        k_max: usize,
        resolution: AnticipatedResolution,
        delivery_hours: Vec<f64>,
        delivery_discount: Vec<f64>,
        delivery_stage_ids: Vec<i32>,
    }

    /// The fixture's `ResolvedBounds` table, sized `n_stages + k_max` per
    /// [`BoundsCountsSpec::k_max`]'s contract; only `PSA_IN_STUDY`'s in-study
    /// cells are populated.
    fn psa_bounds(k_max: usize) -> ResolvedBounds {
        let mut bounds = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 0,
                n_thermals: 1,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages: PSA_N_STAGES,
                k_max,
            },
            &BoundsDefaults {
                hydro: HydroStageBounds {
                    min_storage_hm3: 0.0,
                    max_storage_hm3: 0.0,
                    filling_min_rate_m3s: 0.0,
                    water_withdrawal_m3s: 0.0,
                },
                hydro_block: HydroBlockBounds {
                    ..Default::default()
                },
                thermal: ThermalStageBounds { cost_per_mwh: 0.0 },
                thermal_block: ThermalBlockBounds {
                    min_generation_mw: 0.0,
                    max_generation_mw: 0.0,
                },
                line_block: LineBlockBounds {
                    direct_mw: 0.0,
                    reverse_mw: 0.0,
                },
                pumping_block: PumpingBlockBounds {
                    min_flow_m3s: 0.0,
                    max_flow_m3s: 0.0,
                },
                contract_block: ContractBlockBounds {
                    min_mw: 0.0,
                    max_mw: 0.0,
                    price_per_mwh: 0.0,
                },
            },
        );
        for (stage, &(min_g, max_g, cost)) in PSA_IN_STUDY.iter().enumerate() {
            let tbb = bounds.thermal_block_base_mut(0, stage);
            tbb.min_generation_mw = min_g;
            tbb.max_generation_mw = max_g;
            bounds.thermal_bounds_mut(0, stage).cost_per_mwh = cost;
        }
        bounds
    }

    /// `n_post == 0` yields [`PostStudyResolved::default`] (inert, matching
    /// a study with no declared post-horizon commitment); otherwise resolves
    /// `post_study_cells` (`(post_study_stage_index, cost_per_mwh, min_mw,
    /// max_mw)`) through the production entry point, never hand-assembled.
    fn psa_post_study_resolved(
        n_post: usize,
        post_study_cells: &[(usize, f64, f64, f64)],
        last_real_cumulative: f64,
        last_real_per_stage: f64,
    ) -> PostStudyResolved {
        if n_post == 0 {
            return PostStudyResolved::default();
        }
        let post_study = PostStudyStages {
            stages: (0..n_post)
                .map(|_| PostStudyStage {
                    start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                    duration_hours: PSA_STUDY_HOURS,
                })
                .collect(),
            thermal_bounds: post_study_cells
                .iter()
                .map(
                    |&(post_study_stage_index, cost, min_mw, max_mw)| PostStudyThermalBound {
                        thermal_id: PSA_THERMAL_ID,
                        post_study_stage_index,
                        cost_per_mwh: cost,
                        min_mw,
                        max_mw,
                    },
                )
                .collect(),
        };
        crate::setup::resolve_post_study_artifacts(
            Some(&post_study),
            &[PSA_THERMAL_ID],
            &HorizonGraph::default(),
            last_real_cumulative,
            last_real_per_stage,
        )
    }

    impl PostStudyAnchorFixtures {
        /// `n_post` post-study stages; `post_study_cells` declares
        /// `(post_study_stage_index, cost_per_mwh, min_mw, max_mw)` cells in
        /// the post-study thermal-bounds table.
        fn new(n_post: usize, post_study_cells: &[(usize, f64, f64, f64)]) -> Self {
            let thermals = vec![Thermal {
                id: PSA_THERMAL_ID,
                name: "T_post_study_anchor".to_string(),
                operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                bus_id: EntityId(1),
                min_generation_mw: 0.0,
                max_generation_mw: 100.0,
                cost_per_mwh: 0.0,
                anticipated_config: Some(AnticipatedConfig::LeadStages(PSA_LEAD)),
                entry_stage_id: None,
                exit_stage_id: None,
            }];

            let resolution = AnticipatedResolution::resolve(
                &[LeadTime::Stages(PSA_LEAD)],
                DeliveryAxis {
                    stage_lengths_hours: &[],
                    n_decision: PSA_N_STAGES,
                    n_delivery: PSA_N_STAGES + n_post,
                },
            );
            let k_max = resolution.k_max;
            let bounds = psa_bounds(k_max);

            let study_hours = [PSA_STUDY_HOURS; PSA_N_STAGES];
            let mut study_discount = Vec::with_capacity(PSA_N_STAGES);
            let mut d = 1.0_f64;
            for _ in 0..PSA_N_STAGES {
                study_discount.push(d);
                d *= 0.9;
            }
            let last_real_cumulative = *study_discount.last().unwrap();
            let last_real_per_stage = 0.9;

            let post_study_resolved = psa_post_study_resolved(
                n_post,
                post_study_cells,
                last_real_cumulative,
                last_real_per_stage,
            );

            let delivery_hours: Vec<f64> = study_hours
                .iter()
                .copied()
                .chain(post_study_resolved.total_hours.iter().copied())
                .collect();
            let delivery_discount: Vec<f64> = study_discount
                .iter()
                .copied()
                .chain(
                    post_study_resolved
                        .cumulative_discount_factors
                        .iter()
                        .copied(),
                )
                .collect();
            let delivery_stage_ids: Vec<i32> =
                (0..i32::try_from(PSA_N_STAGES + n_post).unwrap()).collect();

            Self {
                thermals,
                cascade: CascadeTopology::build(&[]),
                hydro_cell_index: HydroCellIndex::build(&[]),
                bounds,
                par_lp: PrecomputedPar::default(),
                penalties: ResolvedPenalties::empty(),
                production_models: ProductionModelSet::new(vec![], 0, 1),
                evaporation_models: EvaporationModelSet::new(vec![]),
                resolved_generic_bounds: ResolvedGenericConstraintBounds::empty(),
                resolved_load_factors: ResolvedLoadFactors::empty(),
                resolved_ncs_bounds: ResolvedNcsBounds::empty(),
                resolved_ncs_factors: ResolvedNcsFactors::empty(),
                resolved_parameters: ResolvedParameters {
                    per_param: vec![],
                    id_to_slot: vec![],
                    cost_scale_factor: 1_000_000.0,
                },
                post_study_resolved,
                k_max,
                resolution,
                delivery_hours,
                delivery_discount,
                delivery_stage_ids,
            }
        }

        fn make_ctx(&self) -> TemplateBuildCtx<'_> {
            TemplateBuildCtx {
                hydros: &[],
                thermals: &self.thermals,
                lines: &[],
                buses: &[],
                load_models: &[],
                cascade: &self.cascade,
                hydro_cell_index: &self.hydro_cell_index,
                resolved: ResolvedTables {
                    bounds: &self.bounds,
                    penalties: &self.penalties,
                    resolved_generic_bounds: &self.resolved_generic_bounds,
                    resolved_load_factors: &self.resolved_load_factors,
                    resolved_ncs_bounds: &self.resolved_ncs_bounds,
                    resolved_ncs_factors: &self.resolved_ncs_factors,
                    resolved_parameters: &self.resolved_parameters,
                },
                hydro_pos: BTreeMap::new(),
                thermal_pos: BTreeMap::new(),
                line_pos: BTreeMap::new(),
                bus_pos: BTreeMap::new(),
                par_lp: &self.par_lp,
                production_models: &self.production_models,
                evaporation_models: &self.evaporation_models,
                generic_constraints: &[],
                non_controllable_sources: &[],
                pumping_stations: &[],
                pumping_pos: BTreeMap::new(),
                n_pumping: 0,
                contracts: &[],
                contract_pos: BTreeMap::new(),
                n_contract_import: 0,
                n_contract_export: 0,
                diversion_upstream: HashMap::new(),
                arc_stage_weights: HashMap::new(),
                arc_spread_chrono: HashMap::new(),
                arc_arrival_density: HashMap::new(),
                per_stage_mask: Vec::new(),
                post_study_resolved: self.post_study_resolved.clone(),
                n_hydros: 0,
                n_thermals: 1,
                n_lines: 0,
                n_buses: 0,
                max_par_order: 0,
                n_anticipated: 1,
                k_max: self.k_max,
                anticipated_lead_stages: vec![usize::try_from(PSA_LEAD).unwrap()],
                anticipated_thermal_indices: vec![ThermalSys::new(0)],
                anticipated_windows: vec![(None, None)],
                anticipated_resolution: self.resolution.clone(),
                study_stage_ids: (0..i32::try_from(PSA_N_STAGES).unwrap()).collect(),
                delivery_stage_ids: self.delivery_stage_ids.clone(),
                has_penalty: false,
                delivery_cumulative_discount_factors: self.delivery_discount.clone(),
                delivery_total_hours: self.delivery_hours.clone(),
                filling_v_target: BTreeMap::new(),
            }
        }
    }

    /// AC1: study-only axis (`n_post = 0`, so `n_delivery == n_stages`).
    /// Decision at `stage_idx = 0` delivers at `delivery_stage = 2` — the
    /// last in-study stage — so the generalized branch still takes the
    /// in-study arm, byte-identical to the pre-ticket read.
    #[test]
    fn post_study_anchor_study_only_axis_reads_thermal_block_base() {
        let fixtures = PostStudyAnchorFixtures::new(0, &[]);
        let ctx = fixtures.make_ctx();
        let state = state_layout_with_resolution(&ctx);
        let stage = two_block_stage(0, [PSA_STUDY_HOURS / 2.0, PSA_STUDY_HOURS / 2.0]);
        let layout = StageLayout::new(&ctx, &state, &stage, 0);

        let (col_lower, col_upper, objective) = fill_stage_columns(&ctx, &stage, 0, &layout);
        let decision_col = layout.anticipated.col_anticipated_decision_start;

        let delivery = 2_usize;
        let (min_g, max_g, cost) = PSA_IN_STUDY[delivery];
        assert_eq!(
            col_upper[decision_col], max_g,
            "col_upper must equal thermal_block_base's max_generation_mw at the delivery stage"
        );
        assert_eq!(
            col_lower[decision_col], min_g,
            "col_lower must equal thermal_block_base's min_generation_mw at the delivery stage"
        );
        let expected_obj = cost
            * ctx.delivery_total_hours[delivery]
            * ctx.delivery_cumulative_discount_factors[delivery];
        assert_eq!(
            objective[decision_col], expected_obj,
            "objective must equal cost * hours * discount at the delivery stage"
        );
    }

    /// AC2: extended axis (3 study + 2 post-study stages). Decision at
    /// `stage_idx = 1` delivers at `delivery_stage = 3` (post-study stage
    /// `0`), where the deck declares `(cost 42.0, min 10.0, max 50.0)`.
    #[test]
    fn post_study_anchor_extended_axis_reads_post_study_bound() {
        let fixtures = PostStudyAnchorFixtures::new(2, &[(0, 42.0, 10.0, 50.0)]);
        let ctx = fixtures.make_ctx();
        let state = state_layout_with_resolution(&ctx);
        let stage = two_block_stage(1, [PSA_STUDY_HOURS / 2.0, PSA_STUDY_HOURS / 2.0]);
        let layout = StageLayout::new(&ctx, &state, &stage, 1);

        let (col_lower, col_upper, objective) = fill_stage_columns(&ctx, &stage, 1, &layout);
        let decision_col = layout.anticipated.col_anticipated_decision_start;

        let delivery = 3_usize;
        assert_eq!(
            col_lower[decision_col], 10.0,
            "col_lower must equal the post-study cell's min_mw"
        );
        assert_eq!(
            col_upper[decision_col], 50.0,
            "col_upper must equal the post-study cell's max_mw"
        );
        let expected_obj = 42.0
            * ctx.delivery_total_hours[delivery]
            * ctx.delivery_cumulative_discount_factors[delivery];
        assert_eq!(
            objective[decision_col], expected_obj,
            "objective must equal the post-study cell's cost * delivery hours * delivery discount"
        );
    }

    /// AC3: same extended axis, no post-study cell declared for `(thermal,
    /// post-study stage 0)`. The decision column degrades to the dormant
    /// `[0, 0]` treatment, with no panic.
    #[test]
    fn post_study_anchor_missing_cell_leaves_decision_column_dormant() {
        let fixtures = PostStudyAnchorFixtures::new(2, &[]);
        let ctx = fixtures.make_ctx();
        let state = state_layout_with_resolution(&ctx);
        let stage = two_block_stage(1, [PSA_STUDY_HOURS / 2.0, PSA_STUDY_HOURS / 2.0]);
        let layout = StageLayout::new(&ctx, &state, &stage, 1);

        let (col_lower, col_upper, objective) = fill_stage_columns(&ctx, &stage, 1, &layout);
        let decision_col = layout.anticipated.col_anticipated_decision_start;

        assert_eq!(
            col_lower[decision_col], 0.0,
            "a missing cell must leave col_lower dormant"
        );
        assert_eq!(
            col_upper[decision_col], 0.0,
            "a missing cell must leave col_upper dormant"
        );
        assert_eq!(
            objective[decision_col], 0.0,
            "a missing cell must leave the objective zero"
        );
    }

    /// AC4: a fixed-replay post-study cell (`min_mw == max_mw`) pins the
    /// decision column to that value from the post-study table alone — no
    /// second interval intersected.
    #[test]
    fn post_study_anchor_min_equals_max_pins_the_decision_column() {
        let fixtures = PostStudyAnchorFixtures::new(2, &[(0, 42.0, 30.0, 30.0)]);
        let ctx = fixtures.make_ctx();
        let state = state_layout_with_resolution(&ctx);
        let stage = two_block_stage(1, [PSA_STUDY_HOURS / 2.0, PSA_STUDY_HOURS / 2.0]);
        let layout = StageLayout::new(&ctx, &state, &stage, 1);

        let (col_lower, col_upper, _objective) = fill_stage_columns(&ctx, &stage, 1, &layout);
        let decision_col = layout.anticipated.col_anticipated_decision_start;

        assert_eq!(
            col_lower[decision_col], 30.0,
            "the pinned cell must set col_lower"
        );
        assert_eq!(
            col_upper[decision_col], 30.0,
            "the pinned cell must set col_upper"
        );
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::float_cmp,
    clippy::similar_names
)]
mod block_family_slack_tests {
    use std::collections::{BTreeMap, HashMap};

    use cobre_core::entities::hydro::HydroGenerationModel;
    use cobre_core::{
        BoundsCountsSpec, BoundsDefaults, BusStagePenalties, CascadeTopology, ContractBlockBounds,
        EntityId, Hydro, HydroBlockBounds, HydroStageBounds, HydroStagePenalties, LineBlockBounds,
        LineStagePenalties, NcsStagePenalties, PenaltiesCountsSpec, PenaltiesDefaults,
        PumpingBlockBounds, ResolvedBounds, ResolvedGenericConstraintBounds, ResolvedLoadFactors,
        ResolvedNcsBounds, ResolvedNcsFactors, ResolvedPenalties, ThermalBlockBounds,
        ThermalStageBounds,
    };
    use cobre_stochastic::par::precompute::PrecomputedPar;

    use crate::hydro_models::{
        EvaporationModel, EvaporationModelSet, ProductionModelSet, ResolvedProductionModel,
    };
    use crate::indexer::{BlockIdx, HydroCell, HydroCellIndex, HydroSys};
    use crate::lead_time::AnticipatedResolution;
    use crate::resolved_parameters::ResolvedParameters;

    use super::super::layout::ResolvedTables;
    use super::super::test_support::{state_layout_for, two_block_stage, zero_hydro_penalties};
    use super::{ColumnBufs, StageLayout, TemplateBuildCtx, fill_operational_slack_columns};

    const N_STAGES: usize = 1;
    const STAGE_IDX: usize = 0;
    const N_HYDROS: usize = 6;
    const BLOCK_HOURS: [f64; 2] = [300.0, 444.0];

    /// One hydro's intended predicate state and the four distinct objective costs
    /// it carries. The costs differ per family and per hydro so a cost-field
    /// cross-wiring in the driver's `match` is observable in the assertions.
    struct HydroSpec {
        min_outflow_m3s: f64,
        max_outflow_m3s: Option<f64>,
        min_turbined_m3s: f64,
        min_generation_mw: f64,
        outflow_below_cost: f64,
        outflow_above_cost: f64,
        turbined_below_cost: f64,
        generation_below_cost: f64,
    }

    /// Six hydros spanning every predicate state plus an all-inert row. Hydro 2
    /// carries `max_outflow_m3s = Some(0.0)` to lock the `is_some()` semantics: a
    /// `> 0.0` comparison would wrongly deactivate its outflow-above column.
    fn hydro_specs() -> [HydroSpec; N_HYDROS] {
        [
            // H0: outflow-below active only.
            HydroSpec {
                min_outflow_m3s: 12.0,
                max_outflow_m3s: None,
                min_turbined_m3s: 0.0,
                min_generation_mw: 0.0,
                outflow_below_cost: 1.0,
                outflow_above_cost: 2.0,
                turbined_below_cost: 3.0,
                generation_below_cost: 4.0,
            },
            // H1: outflow-above active via Some(positive).
            HydroSpec {
                min_outflow_m3s: 0.0,
                max_outflow_m3s: Some(50.0),
                min_turbined_m3s: 0.0,
                min_generation_mw: 0.0,
                outflow_below_cost: 5.0,
                outflow_above_cost: 6.0,
                turbined_below_cost: 7.0,
                generation_below_cost: 8.0,
            },
            // H2: outflow-above active via Some(0.0) — the is_some() lock.
            HydroSpec {
                min_outflow_m3s: 0.0,
                max_outflow_m3s: Some(0.0),
                min_turbined_m3s: 0.0,
                min_generation_mw: 0.0,
                outflow_below_cost: 9.0,
                outflow_above_cost: 10.0,
                turbined_below_cost: 11.0,
                generation_below_cost: 12.0,
            },
            // H3: turbine-below active only.
            HydroSpec {
                min_outflow_m3s: 0.0,
                max_outflow_m3s: None,
                min_turbined_m3s: 7.0,
                min_generation_mw: 0.0,
                outflow_below_cost: 13.0,
                outflow_above_cost: 14.0,
                turbined_below_cost: 15.0,
                generation_below_cost: 16.0,
            },
            // H4: generation-below active only.
            HydroSpec {
                min_outflow_m3s: 0.0,
                max_outflow_m3s: None,
                min_turbined_m3s: 0.0,
                min_generation_mw: 9.0,
                outflow_below_cost: 17.0,
                outflow_above_cost: 18.0,
                turbined_below_cost: 19.0,
                generation_below_cost: 20.0,
            },
            // H5: all four inert.
            HydroSpec {
                min_outflow_m3s: 0.0,
                max_outflow_m3s: None,
                min_turbined_m3s: 0.0,
                min_generation_mw: 0.0,
                outflow_below_cost: 21.0,
                outflow_above_cost: 22.0,
                turbined_below_cost: 23.0,
                generation_below_cost: 24.0,
            },
        ]
    }

    /// Independent constant-productivity hydro (no FPHA/evaporation columns).
    fn fixture_hydro(id: i32) -> Hydro {
        let mut hydro = Hydro {
            unit_groups: Vec::new(),
            id: EntityId(id),
            name: format!("H{id}"),
            operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            downstream_id: None,
            travel_time_hours: None,
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: 0.0,
            max_storage_hm3: 100.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: HydroGenerationModel::ConstantProductivity,
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 50.0,
            specific_productivity_mw_per_m3s_per_m: None,
            min_generation_mw: 0.0,
            max_generation_mw: 45.0,
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: None,
            evaporation_reference_volumes_hm3: None,
            diversion: None,
            filling: None,
            penalties: zero_hydro_penalties(),
        };
        hydro.declare_mirror_unit_group(EntityId(1));
        hydro
    }

    fn zero_hydro_stage_bounds() -> HydroStageBounds {
        HydroStageBounds {
            min_storage_hm3: 0.0,
            max_storage_hm3: 100.0,
            filling_min_rate_m3s: 0.0,
            water_withdrawal_m3s: 0.0,
        }
    }

    fn zero_hydro_block_bounds() -> HydroBlockBounds {
        HydroBlockBounds {
            max_turbined_m3s: 50.0,
            max_generation_mw: 45.0,
            ..Default::default()
        }
    }

    fn zero_hydro_stage_penalties() -> HydroStagePenalties {
        HydroStagePenalties {
            spillage_cost: 0.0,
            diversion_cost: 0.0,
            turbined_cost: 0.0,
            storage_violation_below_cost: 0.0,
            filling_target_violation_cost: 0.0,
            turbined_violation_below_cost: 0.0,
            outflow_violation_below_cost: 0.0,
            outflow_violation_above_cost: 0.0,
            generation_violation_below_cost: 0.0,
            evaporation_violation_cost: 0.0,
            water_withdrawal_violation_cost: 0.0,
            water_withdrawal_violation_pos_cost: 0.0,
            water_withdrawal_violation_neg_cost: 0.0,
            evaporation_violation_pos_cost: 0.0,
            evaporation_violation_neg_cost: 0.0,
            inflow_nonnegativity_cost: 0.0,
        }
    }

    /// Build the resolved bound and penalty tables, writing each hydro's predicate
    /// state and four costs into its resolved cell (the driver reads the resolved
    /// table, so the per-stage cells — not the entity declarations — carry the state).
    fn resolved_tables(specs: &[HydroSpec; N_HYDROS]) -> (ResolvedBounds, ResolvedPenalties) {
        let mut bounds = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: N_HYDROS,
                n_thermals: 0,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages: N_STAGES,
                k_max: 0,
            },
            &BoundsDefaults {
                hydro: zero_hydro_stage_bounds(),
                hydro_block: zero_hydro_block_bounds(),
                thermal: ThermalStageBounds { cost_per_mwh: 0.0 },
                thermal_block: ThermalBlockBounds {
                    min_generation_mw: 0.0,
                    max_generation_mw: 0.0,
                },
                line_block: LineBlockBounds {
                    direct_mw: 0.0,
                    reverse_mw: 0.0,
                },
                pumping_block: PumpingBlockBounds {
                    min_flow_m3s: 0.0,
                    max_flow_m3s: 0.0,
                },
                contract_block: ContractBlockBounds {
                    min_mw: 0.0,
                    max_mw: 0.0,
                    price_per_mwh: 0.0,
                },
            },
        );
        let mut penalties = ResolvedPenalties::new(
            &PenaltiesCountsSpec {
                n_hydros: N_HYDROS,
                n_buses: 0,
                n_lines: 0,
                n_ncs: 0,
                n_stages: N_STAGES,
            },
            &PenaltiesDefaults {
                hydro: zero_hydro_stage_penalties(),
                bus: BusStagePenalties { excess_cost: 0.0 },
                line: LineStagePenalties { exchange_cost: 0.0 },
                ncs: NcsStagePenalties {
                    curtailment_cost: 0.0,
                },
            },
        );
        for (h_idx, spec) in specs.iter().enumerate() {
            let hb = bounds.hydro_block_base_mut(h_idx, STAGE_IDX);
            hb.min_outflow_m3s = spec.min_outflow_m3s;
            hb.max_outflow_m3s = spec.max_outflow_m3s;
            hb.min_turbined_m3s = spec.min_turbined_m3s;
            hb.min_generation_mw = spec.min_generation_mw;
            let hp = penalties.hydro_penalties_mut(h_idx, STAGE_IDX);
            hp.outflow_violation_below_cost = spec.outflow_below_cost;
            hp.outflow_violation_above_cost = spec.outflow_above_cost;
            hp.turbined_violation_below_cost = spec.turbined_below_cost;
            hp.generation_violation_below_cost = spec.generation_below_cost;
        }
        (bounds, penalties)
    }

    /// Owns the borrow targets for a multi-hydro `TemplateBuildCtx`.
    struct SlackFixtures {
        par_lp: PrecomputedPar,
        hydros: Vec<Hydro>,
        hydro_cell_index: HydroCellIndex,
        cascade: CascadeTopology,
        bounds: ResolvedBounds,
        penalties: ResolvedPenalties,
        production_models: ProductionModelSet,
        evaporation_models: EvaporationModelSet,
        resolved_generic_bounds: ResolvedGenericConstraintBounds,
        resolved_load_factors: ResolvedLoadFactors,
        resolved_ncs_bounds: ResolvedNcsBounds,
        resolved_ncs_factors: ResolvedNcsFactors,
        resolved_parameters: ResolvedParameters,
    }

    impl SlackFixtures {
        fn new(specs: &[HydroSpec; N_HYDROS]) -> Self {
            let mut hydros: Vec<Hydro> =
                (0..N_HYDROS).map(|i| fixture_hydro(i as i32 + 1)).collect();
            // `fixture_hydro` mirrors the entity's OWN (zero) min_turbined_m3s/
            // min_generation_mw into its single group at construction time, before
            // this spec's intended activation state is known; the cell-keyed
            // min-floor families now read the GROUP's own resolved minimum
            // (`cell_min_turbined`/`cell_min_generation`), so the mirror must be
            // overwritten here to carry each spec's value — otherwise every cell
            // reads back 0.0 regardless of `spec.min_turbined_m3s`/`min_generation_mw`.
            for (i, hydro) in hydros.iter_mut().enumerate() {
                hydro.unit_groups[0].min_turbined_m3s = specs[i].min_turbined_m3s;
                hydro.unit_groups[0].min_generation_mw = specs[i].min_generation_mw;
            }
            let cascade = CascadeTopology::build(&hydros);
            let hydro_cell_index = HydroCellIndex::build(&hydros);
            let (bounds, penalties) = resolved_tables(specs);
            Self {
                par_lp: PrecomputedPar::default(),
                hydros,
                hydro_cell_index,
                cascade,
                bounds,
                penalties,
                production_models: ProductionModelSet::new(
                    vec![
                        vec![
                            ResolvedProductionModel::ConstantProductivity { productivity: 1.0 };
                            N_STAGES
                        ];
                        N_HYDROS
                    ],
                    N_HYDROS,
                    N_STAGES,
                ),
                evaporation_models: EvaporationModelSet::new(vec![
                    EvaporationModel::None;
                    N_HYDROS
                ]),
                resolved_generic_bounds: ResolvedGenericConstraintBounds::empty(),
                resolved_load_factors: ResolvedLoadFactors::empty(),
                resolved_ncs_bounds: ResolvedNcsBounds::empty(),
                resolved_ncs_factors: ResolvedNcsFactors::empty(),
                resolved_parameters: ResolvedParameters {
                    per_param: vec![],
                    id_to_slot: vec![],
                    cost_scale_factor: 1_000_000.0,
                },
            }
        }

        fn make_ctx(&self) -> TemplateBuildCtx<'_> {
            let mut hydro_pos = BTreeMap::new();
            for (i, h) in self.hydros.iter().enumerate() {
                hydro_pos.insert(h.id, i);
            }
            TemplateBuildCtx {
                hydros: &self.hydros,
                thermals: &[],
                lines: &[],
                buses: &[],
                load_models: &[],
                cascade: &self.cascade,
                hydro_cell_index: &self.hydro_cell_index,
                resolved: ResolvedTables {
                    bounds: &self.bounds,
                    penalties: &self.penalties,
                    resolved_generic_bounds: &self.resolved_generic_bounds,
                    resolved_load_factors: &self.resolved_load_factors,
                    resolved_ncs_bounds: &self.resolved_ncs_bounds,
                    resolved_ncs_factors: &self.resolved_ncs_factors,
                    resolved_parameters: &self.resolved_parameters,
                },
                hydro_pos,
                thermal_pos: BTreeMap::new(),
                line_pos: BTreeMap::new(),
                bus_pos: BTreeMap::new(),
                par_lp: &self.par_lp,
                production_models: &self.production_models,
                evaporation_models: &self.evaporation_models,
                generic_constraints: &[],
                non_controllable_sources: &[],
                pumping_stations: &[],
                pumping_pos: BTreeMap::new(),
                n_pumping: 0,
                contracts: &[],
                contract_pos: BTreeMap::new(),
                n_contract_import: 0,
                n_contract_export: 0,
                diversion_upstream: HashMap::new(),
                arc_stage_weights: HashMap::new(),
                arc_spread_chrono: HashMap::new(),
                arc_arrival_density: HashMap::new(),
                per_stage_mask: Vec::new(),
                post_study_resolved: crate::setup::PostStudyResolved::default(),
                n_hydros: N_HYDROS,
                n_thermals: 0,
                n_lines: 0,
                n_buses: 0,
                max_par_order: 0,
                n_anticipated: 0,
                k_max: 0,
                anticipated_lead_stages: vec![],
                anticipated_thermal_indices: vec![],
                anticipated_windows: vec![],
                anticipated_resolution: AnticipatedResolution::default(),
                study_stage_ids: vec![],
                delivery_stage_ids: vec![],
                has_penalty: false,
                delivery_cumulative_discount_factors: vec![1.0],
                delivery_total_hours: vec![BLOCK_HOURS[0] + BLOCK_HOURS[1]],
                filling_v_target: BTreeMap::new(),
            }
        }
    }

    /// One hydro-keyed family's expected contract: its name, the activation
    /// predicate over a `HydroSpec`, the `StageLayout` column accessor, and the
    /// expected cost field.
    struct FamilyCheck<'b> {
        name: &'static str,
        predicate: fn(&HydroSpec) -> bool,
        accessor: fn(&StageLayout<'b>, HydroSys, BlockIdx) -> usize,
        cost_of: fn(&HydroSpec) -> f64,
    }

    /// One cell-keyed family's expected contract — the min-floor mirror of
    /// [`FamilyCheck`]: the `StageLayout` accessor now takes a [`HydroCell`],
    /// never a [`HydroSys`], since a plant's min-turbine/min-generation floor is
    /// now a per-cell sum, not a plant-level aggregate.
    struct CellFamilyCheck<'b> {
        name: &'static str,
        predicate: fn(&HydroSpec) -> bool,
        accessor: fn(&StageLayout<'b>, HydroCell, BlockIdx) -> usize,
        cost_of: fn(&HydroSpec) -> f64,
    }

    /// Building the operational-violation slack columns through the enum-dispatched
    /// driver reproduces the per-family contract for every (hydro, block): `col_upper`
    /// is `+∞` exactly when the family's resolved predicate holds (else `0.0`),
    /// `objective` is the family's resolved cost times the block duration, and
    /// `col_lower` stays at the `0.0` default. Hydro 2's `Some(0.0)` cap locks the
    /// `is_some()` outflow-above semantics against a `> 0.0` regression. Every
    /// fixture hydro carries exactly one (mirrored) group, so `HydroCell::new(h_idx)`
    /// is that hydro's own cell under the single-bus identity partition.
    #[test]
    fn block_family_driver_matches_legacy_slack_fills() {
        let specs = hydro_specs();
        let fixtures = SlackFixtures::new(&specs);
        let stage = two_block_stage(STAGE_IDX, [BLOCK_HOURS[0], BLOCK_HOURS[1]]);
        let ctx = fixtures.make_ctx();
        let state = state_layout_for(&ctx);
        let layout = StageLayout::new(&ctx, &state, &stage, STAGE_IDX);

        let mut col_lower = vec![0.0_f64; layout.num_cols];
        let mut col_upper = vec![f64::INFINITY; layout.num_cols];
        let mut objective = vec![0.0_f64; layout.num_cols];
        let mut bufs = ColumnBufs {
            col_lower: &mut col_lower,
            col_upper: &mut col_upper,
            objective: &mut objective,
        };
        fill_operational_slack_columns(&ctx, &stage, STAGE_IDX, &layout, &mut bufs);

        let hydro_families = [
            FamilyCheck {
                name: "outflow_below",
                predicate: |s| s.min_outflow_m3s > 0.0,
                accessor: StageLayout::outflow_below_col,
                cost_of: |s| s.outflow_below_cost,
            },
            FamilyCheck {
                name: "outflow_above",
                predicate: |s| s.max_outflow_m3s.is_some(),
                accessor: StageLayout::outflow_above_col,
                cost_of: |s| s.outflow_above_cost,
            },
        ];
        let cell_families = [
            CellFamilyCheck {
                name: "turbine_below",
                predicate: |s| s.min_turbined_m3s > 0.0,
                accessor: StageLayout::turbine_below_col,
                cost_of: |s| s.turbined_below_cost,
            },
            CellFamilyCheck {
                name: "generation_below",
                predicate: |s| s.min_generation_mw > 0.0,
                accessor: StageLayout::generation_below_col,
                cost_of: |s| s.generation_below_cost,
            },
        ];

        // The block loop iterates BLOCK_HOURS directly; assert the layout agrees so a
        // fixture/layout block-count drift cannot silently skip blocks.
        assert_eq!(layout.n_blks, BLOCK_HOURS.len());
        for family in &hydro_families {
            let name = family.name;
            for (h_idx, spec) in specs.iter().enumerate() {
                let active = (family.predicate)(spec);
                let cost = (family.cost_of)(spec);
                for (blk, &hours) in BLOCK_HOURS.iter().enumerate() {
                    let col = (family.accessor)(&layout, HydroSys::new(h_idx), BlockIdx::new(blk));
                    let expected_upper = if active { f64::INFINITY } else { 0.0 };
                    assert_eq!(
                        col_upper[col], expected_upper,
                        "{name} h{h_idx} blk{blk}: col_upper[{col}] expected {expected_upper}"
                    );
                    let expected_obj = cost * hours;
                    assert_eq!(
                        objective[col], expected_obj,
                        "{name} h{h_idx} blk{blk}: objective[{col}] expected {expected_obj}"
                    );
                    assert_eq!(
                        col_lower[col], 0.0,
                        "{name} h{h_idx} blk{blk}: col_lower[{col}] must stay at 0.0"
                    );
                }
            }
        }
        for family in &cell_families {
            let name = family.name;
            for (h_idx, spec) in specs.iter().enumerate() {
                let active = (family.predicate)(spec);
                let cost = (family.cost_of)(spec);
                let cell = HydroCell::new(h_idx);
                for (blk, &hours) in BLOCK_HOURS.iter().enumerate() {
                    let col = (family.accessor)(&layout, cell, BlockIdx::new(blk));
                    let expected_upper = if active { f64::INFINITY } else { 0.0 };
                    assert_eq!(
                        col_upper[col], expected_upper,
                        "{name} h{h_idx} blk{blk}: col_upper[{col}] expected {expected_upper}"
                    );
                    let expected_obj = cost * hours;
                    assert_eq!(
                        objective[col], expected_obj,
                        "{name} h{h_idx} blk{blk}: objective[{col}] expected {expected_obj}"
                    );
                    assert_eq!(
                        col_lower[col], 0.0,
                        "{name} h{h_idx} blk{blk}: col_lower[{col}] must stay at 0.0"
                    );
                }
            }
        }
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::float_cmp,
    clippy::similar_names
)]
mod evaporation_slack_objective_tests {
    use std::collections::{BTreeMap, HashMap};

    use cobre_core::entities::hydro::HydroGenerationModel;
    use cobre_core::{
        Block, BlockMode, BoundsCountsSpec, BoundsDefaults, BusStagePenalties, CascadeTopology,
        ContractBlockBounds, EntityId, Hydro, HydroBlockBounds, HydroStageBounds,
        HydroStagePenalties, LineBlockBounds, LineStagePenalties, NcsStagePenalties, NoiseMethod,
        PenaltiesCountsSpec, PenaltiesDefaults, PumpingBlockBounds, ResolvedBounds,
        ResolvedGenericConstraintBounds, ResolvedLoadFactors, ResolvedNcsBounds,
        ResolvedNcsFactors, ResolvedPenalties, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig, ThermalBlockBounds, ThermalStageBounds,
    };
    use cobre_stochastic::par::precompute::PrecomputedPar;

    use crate::hydro_models::{
        EvaporationModel, EvaporationModelSet, LinearizedEvaporation, ProductionModelSet,
        ResolvedProductionModel,
    };
    use crate::indexer::{BlockIdx, EvapLocal, HydroCellIndex};
    use crate::lead_time::AnticipatedResolution;
    use crate::resolved_parameters::ResolvedParameters;

    use super::super::layout::ResolvedTables;
    use super::super::test_support::{state_layout_for, zero_hydro_penalties};
    use super::{ColumnBufs, StageLayout, TemplateBuildCtx, fill_evaporation_columns};

    const N_STAGES: usize = 1;
    const STAGE_IDX: usize = 0;
    const NEG_COST: f64 = 3.0;
    const POS_COST: f64 = 7.0;

    /// One Operating hydro carrying a `Linearized` evaporation model, so
    /// `identify_evap_hydros` reserves the `EVAP_COLS_PER_HYDRO` triple per block.
    fn evaporating_hydro() -> Hydro {
        let mut hydro = Hydro {
            unit_groups: Vec::new(),
            id: EntityId(1),
            name: "H1".to_string(),
            operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            downstream_id: None,
            travel_time_hours: None,
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: 0.0,
            max_storage_hm3: 100.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: HydroGenerationModel::ConstantProductivity,
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 50.0,
            specific_productivity_mw_per_m3s_per_m: None,
            min_generation_mw: 0.0,
            max_generation_mw: 45.0,
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: None,
            evaporation_reference_volumes_hm3: None,
            diversion: None,
            filling: None,
            penalties: zero_hydro_penalties(),
        };
        hydro.declare_mirror_unit_group(EntityId(1));
        hydro
    }

    fn bounds_one_hydro() -> ResolvedBounds {
        ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 1,
                n_thermals: 0,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages: N_STAGES,
                k_max: 0,
            },
            &BoundsDefaults {
                hydro: HydroStageBounds {
                    min_storage_hm3: 0.0,
                    max_storage_hm3: 100.0,
                    filling_min_rate_m3s: 0.0,
                    water_withdrawal_m3s: 0.0,
                },
                hydro_block: HydroBlockBounds {
                    max_turbined_m3s: 50.0,
                    max_generation_mw: 45.0,
                    ..Default::default()
                },
                thermal: ThermalStageBounds { cost_per_mwh: 0.0 },
                thermal_block: ThermalBlockBounds {
                    min_generation_mw: 0.0,
                    max_generation_mw: 0.0,
                },
                line_block: LineBlockBounds {
                    direct_mw: 0.0,
                    reverse_mw: 0.0,
                },
                pumping_block: PumpingBlockBounds {
                    min_flow_m3s: 0.0,
                    max_flow_m3s: 0.0,
                },
                contract_block: ContractBlockBounds {
                    min_mw: 0.0,
                    max_mw: 0.0,
                    price_per_mwh: 0.0,
                },
            },
        )
    }

    /// One hydro's penalties with distinct nonzero evaporation-violation costs so a
    /// pos/neg cross-wire and a missing per-block weighting are both observable.
    fn penalties_one_hydro() -> ResolvedPenalties {
        let mut penalties = ResolvedPenalties::new(
            &PenaltiesCountsSpec {
                n_hydros: 1,
                n_buses: 0,
                n_lines: 0,
                n_ncs: 0,
                n_stages: N_STAGES,
            },
            &PenaltiesDefaults {
                hydro: HydroStagePenalties {
                    spillage_cost: 0.0,
                    diversion_cost: 0.0,
                    turbined_cost: 0.0,
                    storage_violation_below_cost: 0.0,
                    filling_target_violation_cost: 0.0,
                    turbined_violation_below_cost: 0.0,
                    outflow_violation_below_cost: 0.0,
                    outflow_violation_above_cost: 0.0,
                    generation_violation_below_cost: 0.0,
                    evaporation_violation_cost: 0.0,
                    water_withdrawal_violation_cost: 0.0,
                    water_withdrawal_violation_pos_cost: 0.0,
                    water_withdrawal_violation_neg_cost: 0.0,
                    evaporation_violation_pos_cost: 0.0,
                    evaporation_violation_neg_cost: 0.0,
                    inflow_nonnegativity_cost: 0.0,
                },
                bus: BusStagePenalties { excess_cost: 0.0 },
                line: LineStagePenalties { exchange_cost: 0.0 },
                ncs: NcsStagePenalties {
                    curtailment_cost: 0.0,
                },
            },
        );
        let hp = penalties.hydro_penalties_mut(0, STAGE_IDX);
        hp.evaporation_violation_neg_cost = NEG_COST;
        hp.evaporation_violation_pos_cost = POS_COST;
        penalties
    }

    /// A `Stage` with `block_durations.len()` blocks under `block_mode`. The
    /// durations differ per block so a per-block divisor confusion in the code under
    /// test cannot be masked by equal blocks.
    fn stage_with_blocks(block_mode: BlockMode, block_durations: &[f64]) -> Stage {
        Stage {
            index: STAGE_IDX,
            id: STAGE_IDX as i32,
            start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: chrono::NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            season_id: Some(0),
            blocks: block_durations
                .iter()
                .enumerate()
                .map(|(index, &duration_hours)| Block {
                    index,
                    name: format!("BLK{index}"),
                    duration_hours,
                })
                .collect(),
            block_mode,
            state_config: StageStateConfig {
                storage: false,
                inflow_lags: false,
            },
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor: 1,
                noise_method: NoiseMethod::Saa,
            },
        }
    }

    /// Owns the borrow targets for a one-evaporating-hydro `TemplateBuildCtx`.
    struct EvapFixtures {
        par_lp: PrecomputedPar,
        hydros: Vec<Hydro>,
        hydro_cell_index: HydroCellIndex,
        cascade: CascadeTopology,
        bounds: ResolvedBounds,
        penalties: ResolvedPenalties,
        production_models: ProductionModelSet,
        evaporation_models: EvaporationModelSet,
        resolved_generic_bounds: ResolvedGenericConstraintBounds,
        resolved_load_factors: ResolvedLoadFactors,
        resolved_ncs_bounds: ResolvedNcsBounds,
        resolved_ncs_factors: ResolvedNcsFactors,
        resolved_parameters: ResolvedParameters,
    }

    impl EvapFixtures {
        fn new() -> Self {
            let hydros = vec![evaporating_hydro()];
            let cascade = CascadeTopology::build(&hydros);
            let hydro_cell_index = HydroCellIndex::build(&hydros);
            let evaporation_models = EvaporationModelSet::new(vec![EvaporationModel::Linearized {
                coefficients: vec![
                    LinearizedEvaporation {
                        intercept_m3s: 1.0,
                        volume_slope_m3s_per_hm3: 0.0,
                    };
                    N_STAGES
                ],
                reference_volumes_hm3: vec![50.0; N_STAGES],
            }]);
            Self {
                par_lp: PrecomputedPar::default(),
                hydros,
                hydro_cell_index,
                cascade,
                bounds: bounds_one_hydro(),
                penalties: penalties_one_hydro(),
                production_models: ProductionModelSet::new(
                    vec![vec![
                        ResolvedProductionModel::ConstantProductivity {
                            productivity: 1.0
                        };
                        N_STAGES
                    ]],
                    1,
                    N_STAGES,
                ),
                evaporation_models,
                resolved_generic_bounds: ResolvedGenericConstraintBounds::empty(),
                resolved_load_factors: ResolvedLoadFactors::empty(),
                resolved_ncs_bounds: ResolvedNcsBounds::empty(),
                resolved_ncs_factors: ResolvedNcsFactors::empty(),
                resolved_parameters: ResolvedParameters {
                    per_param: vec![],
                    id_to_slot: vec![],
                    cost_scale_factor: 1_000_000.0,
                },
            }
        }

        fn make_ctx(&self) -> TemplateBuildCtx<'_> {
            let mut hydro_pos = BTreeMap::new();
            hydro_pos.insert(self.hydros[0].id, 0_usize);
            TemplateBuildCtx {
                hydros: &self.hydros,
                thermals: &[],
                lines: &[],
                buses: &[],
                load_models: &[],
                cascade: &self.cascade,
                hydro_cell_index: &self.hydro_cell_index,
                resolved: ResolvedTables {
                    bounds: &self.bounds,
                    penalties: &self.penalties,
                    resolved_generic_bounds: &self.resolved_generic_bounds,
                    resolved_load_factors: &self.resolved_load_factors,
                    resolved_ncs_bounds: &self.resolved_ncs_bounds,
                    resolved_ncs_factors: &self.resolved_ncs_factors,
                    resolved_parameters: &self.resolved_parameters,
                },
                hydro_pos,
                thermal_pos: BTreeMap::new(),
                line_pos: BTreeMap::new(),
                bus_pos: BTreeMap::new(),
                par_lp: &self.par_lp,
                production_models: &self.production_models,
                evaporation_models: &self.evaporation_models,
                generic_constraints: &[],
                non_controllable_sources: &[],
                pumping_stations: &[],
                pumping_pos: BTreeMap::new(),
                n_pumping: 0,
                contracts: &[],
                contract_pos: BTreeMap::new(),
                n_contract_import: 0,
                n_contract_export: 0,
                diversion_upstream: HashMap::new(),
                arc_stage_weights: HashMap::new(),
                arc_spread_chrono: HashMap::new(),
                arc_arrival_density: HashMap::new(),
                per_stage_mask: Vec::new(),
                post_study_resolved: crate::setup::PostStudyResolved::default(),
                n_hydros: 1,
                n_thermals: 0,
                n_lines: 0,
                n_buses: 0,
                max_par_order: 0,
                n_anticipated: 0,
                k_max: 0,
                anticipated_lead_stages: vec![],
                anticipated_thermal_indices: vec![],
                anticipated_windows: vec![],
                anticipated_resolution: AnticipatedResolution::default(),
                study_stage_ids: vec![],
                delivery_stage_ids: vec![],
                has_penalty: false,
                delivery_cumulative_discount_factors: vec![1.0],
                delivery_total_hours: vec![744.0],
                filling_v_target: BTreeMap::new(),
            }
        }
    }

    /// Per-block `f_evap_plus`/`f_evap_minus` objectives plus the layout's `n_blks`.
    struct EvapFill {
        f_plus: Vec<f64>,
        f_minus: Vec<f64>,
        n_blks: usize,
    }

    fn run_fill(fixtures: &EvapFixtures, stage: &Stage) -> EvapFill {
        let ctx = fixtures.make_ctx();
        let state = state_layout_for(&ctx);
        let layout = StageLayout::new(&ctx, &state, stage, STAGE_IDX);
        let mut col_lower = vec![0.0_f64; layout.num_cols];
        let mut col_upper = vec![f64::INFINITY; layout.num_cols];
        let mut objective = vec![0.0_f64; layout.num_cols];
        let mut bufs = ColumnBufs {
            col_lower: &mut col_lower,
            col_upper: &mut col_upper,
            objective: &mut objective,
        };
        fill_evaporation_columns(&ctx, stage, STAGE_IDX, &layout, &mut bufs);
        let f_plus = (0..layout.n_blks)
            .map(|blk| objective[layout.evap_f_plus_col(EvapLocal::new(0), BlockIdx::new(blk))])
            .collect();
        let f_minus = (0..layout.n_blks)
            .map(|blk| objective[layout.evap_f_minus_col(EvapLocal::new(0), BlockIdx::new(blk))])
            .collect();
        EvapFill {
            f_plus,
            f_minus,
            n_blks: layout.n_blks,
        }
    }

    /// In chronological K ≥ 2 each block's evaporation-violation slack objective is
    /// the directional cost times THAT block's `duration_hours` — not the stage-total
    /// hours on every block (the pre-fix inflation) — and the per-block sum telescopes
    /// to `cost * total_stage_hours` (the single-slack parallel total).
    #[test]
    fn chronological_evap_slack_objective_is_block_weighted() {
        let block_durations = [300.0, 444.0, 148.0];
        let total_hours: f64 = block_durations.iter().sum();
        let fixtures = EvapFixtures::new();
        let chrono = run_fill(
            &fixtures,
            &stage_with_blocks(BlockMode::Chronological, &block_durations),
        );

        assert_eq!(
            chrono.n_blks,
            block_durations.len(),
            "layout must reserve one evap triple per block"
        );
        for (blk, &hours) in block_durations.iter().enumerate() {
            assert_eq!(
                chrono.f_plus[blk],
                NEG_COST * hours,
                "blk {blk}: f_evap_plus objective must be neg_cost * this block's hours"
            );
            assert_eq!(
                chrono.f_minus[blk],
                POS_COST * hours,
                "blk {blk}: f_evap_minus objective must be pos_cost * this block's hours"
            );
        }
        let plus_sum: f64 = chrono.f_plus.iter().sum();
        let minus_sum: f64 = chrono.f_minus.iter().sum();
        assert_eq!(
            plus_sum,
            NEG_COST * total_hours,
            "Σ f_evap_plus over blocks must telescope to neg_cost * total_stage_hours"
        );
        assert_eq!(
            minus_sum,
            POS_COST * total_hours,
            "Σ f_evap_minus over blocks must telescope to pos_cost * total_stage_hours"
        );
    }

    /// Parallel (`n_blks == 1`): the single evaporation slack objective is
    /// `cost * total_stage_hours` (`blocks[0].duration_hours == total_stage_hours`),
    /// unchanged by the per-block weighting.
    #[test]
    fn parallel_evap_slack_objective_equals_total_stage_hours() {
        let total_hours = 744.0;
        let fixtures = EvapFixtures::new();
        let parallel = run_fill(
            &fixtures,
            &stage_with_blocks(BlockMode::Parallel, &[total_hours]),
        );

        assert_eq!(parallel.n_blks, 1, "parallel mode reserves one block");
        assert_eq!(
            parallel.f_plus[0],
            NEG_COST * total_hours,
            "parallel f_evap_plus objective must equal neg_cost * total_stage_hours"
        );
        assert_eq!(
            parallel.f_minus[0],
            POS_COST * total_hours,
            "parallel f_evap_minus objective must equal pos_cost * total_stage_hours"
        );
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::float_cmp,
    clippy::similar_names
)]
mod contract_column_tests {
    use std::collections::{BTreeMap, HashMap};

    use cobre_core::entities::energy_contract::{ContractType, EnergyContract};
    use cobre_core::{
        BoundsCountsSpec, BoundsDefaults, CascadeTopology, ContractBlockBounds, EntityId,
        HydroBlockBounds, HydroStageBounds, LineBlockBounds, PumpingBlockBounds, ResolvedBounds,
        ResolvedGenericConstraintBounds, ResolvedLoadFactors, ResolvedNcsBounds,
        ResolvedNcsFactors, ResolvedPenalties, ThermalBlockBounds, ThermalStageBounds,
    };
    use cobre_stochastic::par::precompute::PrecomputedPar;

    use crate::hydro_models::{EvaporationModelSet, ProductionModelSet};
    use crate::indexer::HydroCellIndex;
    use crate::lead_time::AnticipatedResolution;
    use crate::resolved_parameters::ResolvedParameters;

    use super::super::layout::ResolvedTables;
    use super::super::test_support::state_layout_for;
    use super::{ColumnBufs, StageLayout, TemplateBuildCtx, fill_contract_columns};

    const N_STAGES: usize = 1;
    const STAGE_IDX: usize = 0;
    const BLOCK_HOURS: f64 = 730.0;

    fn contract(
        id: i32,
        contract_type: ContractType,
        entry_stage_id: Option<i32>,
    ) -> EnergyContract {
        EnergyContract {
            id: EntityId(id),
            name: format!("C{id}"),
            operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId(1),
            contract_type,
            entry_stage_id,
            exit_stage_id: None,
            price_per_mwh: 0.0,
            min_mw: 0.0,
            max_mw: 0.0,
        }
    }

    fn bounds_with_contracts(n_contracts: usize) -> ResolvedBounds {
        ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 0,
                n_thermals: 0,
                n_lines: 0,
                n_pumping: 0,
                n_contracts,
                n_stages: N_STAGES,
                k_max: 0,
            },
            &BoundsDefaults {
                hydro: HydroStageBounds {
                    min_storage_hm3: 0.0,
                    max_storage_hm3: 0.0,
                    filling_min_rate_m3s: 0.0,
                    water_withdrawal_m3s: 0.0,
                },
                hydro_block: HydroBlockBounds {
                    ..Default::default()
                },
                thermal: ThermalStageBounds { cost_per_mwh: 0.0 },
                thermal_block: ThermalBlockBounds {
                    min_generation_mw: 0.0,
                    max_generation_mw: 0.0,
                },
                line_block: LineBlockBounds {
                    direct_mw: 0.0,
                    reverse_mw: 0.0,
                },
                pumping_block: PumpingBlockBounds {
                    min_flow_m3s: 0.0,
                    max_flow_m3s: 0.0,
                },
                contract_block: ContractBlockBounds {
                    min_mw: 0.0,
                    max_mw: 0.0,
                    price_per_mwh: 0.0,
                },
            },
        )
    }

    /// Owns the borrow targets for a contract-only `TemplateBuildCtx`.
    struct ContractFixtures {
        par_lp: PrecomputedPar,
        hydro_cell_index: HydroCellIndex,
        cascade: CascadeTopology,
        bounds: ResolvedBounds,
        penalties: ResolvedPenalties,
        production_models: ProductionModelSet,
        evaporation_models: EvaporationModelSet,
        resolved_generic_bounds: ResolvedGenericConstraintBounds,
        resolved_load_factors: ResolvedLoadFactors,
        resolved_ncs_bounds: ResolvedNcsBounds,
        resolved_ncs_factors: ResolvedNcsFactors,
        resolved_parameters: ResolvedParameters,
        contracts: Vec<EnergyContract>,
    }

    impl ContractFixtures {
        fn new(contracts: Vec<EnergyContract>) -> Self {
            Self {
                par_lp: PrecomputedPar::default(),
                cascade: CascadeTopology::build(&[]),
                hydro_cell_index: HydroCellIndex::build(&[]),
                bounds: bounds_with_contracts(contracts.len()),
                penalties: ResolvedPenalties::empty(),
                production_models: ProductionModelSet::new(vec![], 0, 1),
                evaporation_models: EvaporationModelSet::new(vec![]),
                resolved_generic_bounds: ResolvedGenericConstraintBounds::empty(),
                resolved_load_factors: ResolvedLoadFactors::empty(),
                resolved_ncs_bounds: ResolvedNcsBounds::empty(),
                resolved_ncs_factors: ResolvedNcsFactors::empty(),
                resolved_parameters: ResolvedParameters {
                    per_param: vec![],
                    id_to_slot: vec![],
                    cost_scale_factor: 1_000_000.0,
                },
                contracts,
            }
        }

        fn set_contract_bounds(&mut self, c_sys: usize, min_mw: f64, max_mw: f64, price: f64) {
            let cell = self.bounds.contract_bounds_mut(c_sys, STAGE_IDX);
            cell.min_mw = min_mw;
            cell.max_mw = max_mw;
            cell.price_per_mwh = price;
        }

        fn make_ctx(&self) -> TemplateBuildCtx<'_> {
            let n_contract_import = self
                .contracts
                .iter()
                .filter(|c| c.contract_type == ContractType::Import)
                .count();
            let n_contract_export = self
                .contracts
                .iter()
                .filter(|c| c.contract_type == ContractType::Export)
                .count();
            let contract_pos: BTreeMap<EntityId, usize> = self
                .contracts
                .iter()
                .enumerate()
                .map(|(i, c)| (c.id, i))
                .collect();
            TemplateBuildCtx {
                hydros: &[],
                thermals: &[],
                lines: &[],
                buses: &[],
                load_models: &[],
                cascade: &self.cascade,
                hydro_cell_index: &self.hydro_cell_index,
                resolved: ResolvedTables {
                    bounds: &self.bounds,
                    penalties: &self.penalties,
                    resolved_generic_bounds: &self.resolved_generic_bounds,
                    resolved_load_factors: &self.resolved_load_factors,
                    resolved_ncs_bounds: &self.resolved_ncs_bounds,
                    resolved_ncs_factors: &self.resolved_ncs_factors,
                    resolved_parameters: &self.resolved_parameters,
                },
                hydro_pos: BTreeMap::new(),
                thermal_pos: BTreeMap::new(),
                line_pos: BTreeMap::new(),
                bus_pos: BTreeMap::new(),
                par_lp: &self.par_lp,
                production_models: &self.production_models,
                evaporation_models: &self.evaporation_models,
                generic_constraints: &[],
                non_controllable_sources: &[],
                pumping_stations: &[],
                pumping_pos: BTreeMap::new(),
                n_pumping: 0,
                contracts: &self.contracts,
                contract_pos,
                n_contract_import,
                n_contract_export,
                diversion_upstream: HashMap::new(),
                arc_stage_weights: HashMap::new(),
                arc_spread_chrono: HashMap::new(),
                arc_arrival_density: HashMap::new(),
                per_stage_mask: Vec::new(),
                post_study_resolved: crate::setup::PostStudyResolved::default(),
                n_hydros: 0,
                n_thermals: 0,
                n_lines: 0,
                n_buses: 0,
                max_par_order: 0,
                n_anticipated: 0,
                k_max: 0,
                anticipated_lead_stages: vec![],
                anticipated_thermal_indices: vec![],
                anticipated_windows: vec![],
                anticipated_resolution: AnticipatedResolution::default(),
                study_stage_ids: vec![],
                delivery_stage_ids: vec![],
                has_penalty: false,
                delivery_cumulative_discount_factors: vec![1.0; N_STAGES],
                delivery_total_hours: vec![744.0; N_STAGES],
                filling_v_target: BTreeMap::new(),
            }
        }
    }

    /// Single-block stage at `STAGE_IDX` with `id = 0` and `BLOCK_HOURS` duration.
    fn one_block_stage() -> cobre_core::Stage {
        use chrono::NaiveDate;
        use cobre_core::{
            Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
            StageStateConfig,
        };
        Stage {
            index: STAGE_IDX,
            id: 0,
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            season_id: Some(0),
            blocks: vec![Block {
                index: 0,
                name: "BLK0".to_string(),
                duration_hours: BLOCK_HOURS,
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
        }
    }

    /// Run `fill_contract_columns` and return `(col_lower, col_upper, objective)`
    /// plus the two family-base offsets the assertions read.
    fn run_fill(fixtures: &ContractFixtures) -> (Vec<f64>, Vec<f64>, Vec<f64>, usize, usize) {
        let stage = one_block_stage();
        let ctx = fixtures.make_ctx();
        let state = state_layout_for(&ctx);
        let layout = StageLayout::new(&ctx, &state, &stage, STAGE_IDX);
        let mut col_lower = vec![0.0_f64; layout.num_cols];
        let mut col_upper = vec![f64::INFINITY; layout.num_cols];
        let mut objective = vec![0.0_f64; layout.num_cols];
        let mut bufs = ColumnBufs {
            col_lower: &mut col_lower,
            col_upper: &mut col_upper,
            objective: &mut objective,
        };
        fill_contract_columns(&ctx, &stage, STAGE_IDX, &layout, &mut bufs);
        (
            col_lower,
            col_upper,
            objective,
            layout.equipment.col_contract_import_start,
            layout.equipment.col_contract_export_start,
        )
    }

    /// An active import contract carries its resolved `[min_mw, max_mw]` bounds and a
    /// `price * block_hours` objective with the stored (positive = cost) sign.
    #[test]
    fn import_contract_bounds_and_objective() {
        let mut fixtures = ContractFixtures::new(vec![contract(1, ContractType::Import, None)]);
        fixtures.set_contract_bounds(0, 10.0, 100.0, 200.0);

        let (col_lower, col_upper, objective, import_start, _) = run_fill(&fixtures);
        assert_eq!(col_lower[import_start], 10.0);
        assert_eq!(col_upper[import_start], 100.0);
        assert_eq!(objective[import_start], 200.0 * BLOCK_HOURS);
    }

    /// An active export contract keeps the stored negative price sign in the
    /// objective — the wiring must NOT negate it.
    #[test]
    fn export_contract_objective_keeps_negative_sign() {
        let mut fixtures = ContractFixtures::new(vec![contract(1, ContractType::Export, None)]);
        fixtures.set_contract_bounds(0, 0.0, 500.0, -150.0);

        let (_, _, objective, _, export_start) = run_fill(&fixtures);
        assert_eq!(objective[export_start], -150.0 * BLOCK_HOURS);
    }

    /// A contract whose entry window opens after the current stage is dormant: BOTH
    /// bounds are pinned to zero at every block (never `[min > 0, 0]`).
    #[test]
    fn dormant_contract_zero_pins_both_bounds() {
        let mut fixtures = ContractFixtures::new(vec![contract(1, ContractType::Import, Some(2))]);
        fixtures.set_contract_bounds(0, 25.0, 100.0, 200.0);

        let (col_lower, col_upper, _, import_start, _) = run_fill(&fixtures);
        assert_eq!(col_lower[import_start], 0.0);
        assert_eq!(col_upper[import_start], 0.0);
    }

    /// A take-or-pay floor (`min_mw > 0`) lands on `col_lower` as a hard lower bound.
    #[test]
    fn take_or_pay_floor_sets_col_lower() {
        let mut fixtures = ContractFixtures::new(vec![contract(1, ContractType::Import, None)]);
        fixtures.set_contract_bounds(0, 50.0, 100.0, 200.0);

        let (col_lower, _, _, import_start, _) = run_fill(&fixtures);
        assert_eq!(col_lower[import_start], 50.0);
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::float_cmp,
    clippy::similar_names
)]
mod thermal_block_bound_tests {
    use std::collections::{BTreeMap, HashMap};

    use cobre_core::{
        BlockBoundsCountsSpec, BoundsCountsSpec, BoundsDefaults, CascadeTopology,
        ContractBlockBounds, EntityId, HydroBlockBounds, HydroStageBounds, LineBlockBounds,
        PumpingBlockBounds, ResolvedBlockBounds, ResolvedBounds, ResolvedGenericConstraintBounds,
        ResolvedLoadFactors, ResolvedNcsBounds, ResolvedNcsFactors, ResolvedPenalties, Thermal,
        ThermalBlockBounds, ThermalStageBounds,
    };
    use cobre_stochastic::par::precompute::PrecomputedPar;

    use crate::hydro_models::{EvaporationModelSet, ProductionModelSet};
    use crate::indexer::HydroCellIndex;
    use crate::lead_time::AnticipatedResolution;
    use crate::resolved_parameters::ResolvedParameters;

    use super::super::layout::ResolvedTables;
    use super::super::test_support::{BLOCK_HOURS, N_BLKS, state_layout_for, three_block_stage};
    use super::{ColumnBufs, StageLayout, TemplateBuildCtx, fill_thermal_columns};

    const N_STAGES: usize = 2;
    const STAGE_IDX: usize = 0;

    fn thermal(id: i32, entry_stage_id: Option<i32>, cost_per_mwh: f64) -> Thermal {
        Thermal {
            id: EntityId(id),
            name: format!("T{id}"),
            operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId(1),
            entry_stage_id,
            exit_stage_id: None,
            cost_per_mwh,
            min_generation_mw: 0.0,
            max_generation_mw: 0.0,
            anticipated_config: None,
        }
    }

    fn bounds_with_thermals(n_thermals: usize) -> ResolvedBounds {
        ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 0,
                n_thermals,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages: N_STAGES,
                k_max: 0,
            },
            &BoundsDefaults {
                hydro: HydroStageBounds {
                    min_storage_hm3: 0.0,
                    max_storage_hm3: 0.0,
                    filling_min_rate_m3s: 0.0,
                    water_withdrawal_m3s: 0.0,
                },
                hydro_block: HydroBlockBounds {
                    ..Default::default()
                },
                thermal: ThermalStageBounds { cost_per_mwh: 0.0 },
                thermal_block: ThermalBlockBounds {
                    min_generation_mw: 0.0,
                    max_generation_mw: 0.0,
                },
                line_block: LineBlockBounds {
                    direct_mw: 0.0,
                    reverse_mw: 0.0,
                },
                pumping_block: PumpingBlockBounds {
                    min_flow_m3s: 0.0,
                    max_flow_m3s: 0.0,
                },
                contract_block: ContractBlockBounds {
                    min_mw: 0.0,
                    max_mw: 0.0,
                    price_per_mwh: 0.0,
                },
            },
        )
    }

    /// Owns the borrow targets for a thermal-only `TemplateBuildCtx`.
    struct ThermalFixtures {
        par_lp: PrecomputedPar,
        hydro_cell_index: HydroCellIndex,
        cascade: CascadeTopology,
        bounds: ResolvedBounds,
        penalties: ResolvedPenalties,
        production_models: ProductionModelSet,
        evaporation_models: EvaporationModelSet,
        resolved_generic_bounds: ResolvedGenericConstraintBounds,
        resolved_load_factors: ResolvedLoadFactors,
        resolved_ncs_bounds: ResolvedNcsBounds,
        resolved_ncs_factors: ResolvedNcsFactors,
        resolved_parameters: ResolvedParameters,
        thermals: Vec<Thermal>,
    }

    impl ThermalFixtures {
        fn new(thermals: Vec<Thermal>) -> Self {
            let n_thermals = thermals.len();
            Self {
                par_lp: PrecomputedPar::default(),
                cascade: CascadeTopology::build(&[]),
                hydro_cell_index: HydroCellIndex::build(&[]),
                bounds: bounds_with_thermals(n_thermals),
                penalties: ResolvedPenalties::empty(),
                production_models: ProductionModelSet::new(vec![], 0, 1),
                evaporation_models: EvaporationModelSet::new(vec![]),
                resolved_generic_bounds: ResolvedGenericConstraintBounds::empty(),
                resolved_load_factors: ResolvedLoadFactors::empty(),
                resolved_ncs_bounds: ResolvedNcsBounds::empty(),
                resolved_ncs_factors: ResolvedNcsFactors::empty(),
                resolved_parameters: ResolvedParameters {
                    per_param: vec![],
                    id_to_slot: vec![],
                    cost_scale_factor: 1_000_000.0,
                },
                thermals,
            }
        }

        fn set_stage_bounds(
            &mut self,
            t_idx: usize,
            stage_idx: usize,
            min_mw: f64,
            max_mw: f64,
            cost: f64,
        ) {
            let cell = self.bounds.thermal_block_base_mut(t_idx, stage_idx);
            cell.min_generation_mw = min_mw;
            cell.max_generation_mw = max_mw;
            self.bounds
                .thermal_bounds_mut(t_idx, stage_idx)
                .cost_per_mwh = cost;
        }

        fn install_block_overlay(&mut self) {
            self.bounds
                .set_block_overlay(ResolvedBlockBounds::new(&BlockBoundsCountsSpec {
                    n_hydros: 0,
                    n_thermals: self.thermals.len(),
                    n_lines: 0,
                    n_pumping: 0,
                    n_contracts: 0,
                    n_stages: N_STAGES,
                    max_blocks: N_BLKS,
                }));
        }

        fn set_block_override(
            &mut self,
            t_idx: usize,
            stage_idx: usize,
            block_idx: usize,
            min_mw: Option<f64>,
            max_mw: Option<f64>,
        ) {
            let over = self
                .bounds
                .block_overlay_mut()
                .thermal_override_mut(t_idx, stage_idx, block_idx)
                .expect("overlay cell must exist for a fixture-sized overlay");
            over.min_generation_mw = min_mw;
            over.max_generation_mw = max_mw;
        }

        fn make_ctx(&self) -> TemplateBuildCtx<'_> {
            let thermal_pos: BTreeMap<EntityId, usize> = self
                .thermals
                .iter()
                .enumerate()
                .map(|(i, t)| (t.id, i))
                .collect();
            TemplateBuildCtx {
                hydros: &[],
                thermals: &self.thermals,
                lines: &[],
                buses: &[],
                load_models: &[],
                cascade: &self.cascade,
                hydro_cell_index: &self.hydro_cell_index,
                resolved: ResolvedTables {
                    bounds: &self.bounds,
                    penalties: &self.penalties,
                    resolved_generic_bounds: &self.resolved_generic_bounds,
                    resolved_load_factors: &self.resolved_load_factors,
                    resolved_ncs_bounds: &self.resolved_ncs_bounds,
                    resolved_ncs_factors: &self.resolved_ncs_factors,
                    resolved_parameters: &self.resolved_parameters,
                },
                hydro_pos: BTreeMap::new(),
                thermal_pos,
                line_pos: BTreeMap::new(),
                bus_pos: BTreeMap::new(),
                par_lp: &self.par_lp,
                production_models: &self.production_models,
                evaporation_models: &self.evaporation_models,
                generic_constraints: &[],
                non_controllable_sources: &[],
                pumping_stations: &[],
                pumping_pos: BTreeMap::new(),
                n_pumping: 0,
                contracts: &[],
                contract_pos: BTreeMap::new(),
                n_contract_import: 0,
                n_contract_export: 0,
                diversion_upstream: HashMap::new(),
                arc_stage_weights: HashMap::new(),
                arc_spread_chrono: HashMap::new(),
                arc_arrival_density: HashMap::new(),
                per_stage_mask: Vec::new(),
                post_study_resolved: crate::setup::PostStudyResolved::default(),
                n_hydros: 0,
                n_thermals: self.thermals.len(),
                n_lines: 0,
                n_buses: 0,
                max_par_order: 0,
                n_anticipated: 0,
                k_max: 0,
                anticipated_lead_stages: vec![],
                anticipated_thermal_indices: vec![],
                anticipated_windows: vec![],
                anticipated_resolution: AnticipatedResolution::default(),
                study_stage_ids: (0..N_STAGES as i32).collect(),
                delivery_stage_ids: (0..N_STAGES as i32).collect(),
                has_penalty: false,
                delivery_cumulative_discount_factors: vec![1.0; N_STAGES],
                delivery_total_hours: vec![BLOCK_HOURS.iter().sum(); N_STAGES],
                filling_v_target: BTreeMap::new(),
            }
        }
    }

    /// Run `fill_thermal_columns` at `stage_idx` against a three-block stage,
    /// returning `(col_lower, col_upper, objective)` and the thermal family's
    /// block-major column base (`thermal_start + t_idx * N_BLKS + blk`).
    fn run_fill(
        fixtures: &ThermalFixtures,
        stage_idx: usize,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>, usize) {
        let stage = three_block_stage(stage_idx);
        let ctx = fixtures.make_ctx();
        let state = state_layout_for(&ctx);
        let layout = StageLayout::new(&ctx, &state, &stage, stage_idx);
        let mut col_lower = vec![0.0_f64; layout.num_cols];
        let mut col_upper = vec![f64::INFINITY; layout.num_cols];
        let mut objective = vec![0.0_f64; layout.num_cols];
        let mut bufs = ColumnBufs {
            col_lower: &mut col_lower,
            col_upper: &mut col_upper,
            objective: &mut objective,
        };
        fill_thermal_columns(&ctx, &stage, stage_idx, &layout, &mut bufs);
        (
            col_lower,
            col_upper,
            objective,
            layout.equipment.thermal.start,
        )
    }

    /// Two thermals, three-block stage, empty overlay: every generation column
    /// reads bit-identical to the stage-level bound (`thermal_bounds_at_block`
    /// falls through to the stage cell). Expected values are explicit literals
    /// derived from the fixture's own bound writes, not recomputed via the
    /// production formula.
    #[test]
    fn test_thermal_columns_bit_identical_without_overlay() {
        let mut fixtures =
            ThermalFixtures::new(vec![thermal(1, None, 30.0), thermal(2, None, 45.0)]);
        fixtures.set_stage_bounds(0, STAGE_IDX, 10.0, 200.0, 30.0);
        fixtures.set_stage_bounds(1, STAGE_IDX, 5.0, 150.0, 45.0);

        let (col_lower, col_upper, objective, thermal_start) = run_fill(&fixtures, STAGE_IDX);

        // Thermal 0: min 10.0, max 200.0, cost 30.0 * [200.0, 300.0, 244.0] hours.
        let expected_0 = (
            [10.0_f64, 10.0, 10.0],
            [200.0_f64, 200.0, 200.0],
            [6000.0_f64, 9000.0, 7320.0],
        );
        // Thermal 1: min 5.0, max 150.0, cost 45.0 * [200.0, 300.0, 244.0] hours.
        let expected_1 = (
            [5.0_f64, 5.0, 5.0],
            [150.0_f64, 150.0, 150.0],
            [9000.0_f64, 13500.0, 10980.0],
        );
        for (t_idx, (exp_lower, exp_upper, exp_obj)) in
            [expected_0, expected_1].into_iter().enumerate()
        {
            for blk in 0..N_BLKS {
                let col = thermal_start + t_idx * N_BLKS + blk;
                assert_eq!(
                    col_lower[col].to_bits(),
                    exp_lower[blk].to_bits(),
                    "col_lower bit-identical, thermal {t_idx} blk {blk}"
                );
                assert_eq!(
                    col_upper[col].to_bits(),
                    exp_upper[blk].to_bits(),
                    "col_upper bit-identical, thermal {t_idx} blk {blk}"
                );
                assert_eq!(
                    objective[col].to_bits(),
                    exp_obj[blk].to_bits(),
                    "objective bit-identical, thermal {t_idx} blk {blk}"
                );
            }
        }
    }

    /// A thermal with stage-wide `max_generation_mw = 500.0` and a `block_id = 1`
    /// row overriding it to `100.0` on a three-block stage binds ONLY block 1;
    /// `col_lower` (no override written) and `objective` are unchanged from the
    /// no-override case.
    #[test]
    fn test_thermal_block_bound_binds_only_its_own_block() {
        let mut fixtures = ThermalFixtures::new(vec![thermal(1, None, 20.0)]);
        fixtures.set_stage_bounds(0, STAGE_IDX, 0.0, 500.0, 20.0);
        fixtures.install_block_overlay();
        fixtures.set_block_override(0, STAGE_IDX, 1, None, Some(100.0));

        let (col_lower, col_upper, objective, thermal_start) = run_fill(&fixtures, STAGE_IDX);

        assert_eq!(
            col_upper[thermal_start..thermal_start + N_BLKS],
            [500.0, 100.0, 500.0],
            "only block 1 is bound to the override"
        );
        assert_eq!(
            col_lower[thermal_start..thermal_start + N_BLKS],
            [0.0, 0.0, 0.0],
            "col_lower unaffected by the max-only override"
        );
        for blk in 0..N_BLKS {
            assert_eq!(
                objective[thermal_start + blk],
                20.0 * BLOCK_HOURS[blk],
                "objective unaffected by the block bound, blk {blk}"
            );
        }
    }

    /// An active thermal with stage-wide `min_generation_mw = 0.0` and a
    /// `block_id = 0` row overriding it to `300.0` on a three-block stage binds
    /// ONLY block 0's floor; `col_upper` (no override written) is unchanged.
    #[test]
    fn test_thermal_block_min_floor_binds_only_its_own_block() {
        let mut fixtures = ThermalFixtures::new(vec![thermal(1, None, 20.0)]);
        fixtures.set_stage_bounds(0, STAGE_IDX, 0.0, 500.0, 20.0);
        fixtures.install_block_overlay();
        fixtures.set_block_override(0, STAGE_IDX, 0, Some(300.0), None);

        let (col_lower, col_upper, _objective, thermal_start) = run_fill(&fixtures, STAGE_IDX);

        assert_eq!(
            col_lower[thermal_start..thermal_start + N_BLKS],
            [300.0, 0.0, 0.0],
            "only block 0's floor is bound to the override"
        );
        assert_eq!(
            col_upper[thermal_start..thermal_start + N_BLKS],
            [500.0, 500.0, 500.0],
            "col_upper unaffected by the min-only override"
        );
    }

    /// A commissioning-dormant thermal (`entry_stage_id` after the build stage)
    /// carrying a `block_id` row with `min_generation_mw = 300.0` still gets
    /// `[0, 0]` at every block — the dormant gate wins over the per-block floor.
    #[test]
    fn test_dormant_thermal_ignores_per_block_floor() {
        let mut fixtures = ThermalFixtures::new(vec![thermal(1, Some(1), 20.0)]);
        fixtures.set_stage_bounds(0, STAGE_IDX, 0.0, 500.0, 20.0);
        fixtures.install_block_overlay();
        fixtures.set_block_override(0, STAGE_IDX, 0, Some(300.0), None);

        let (col_lower, col_upper, _objective, thermal_start) = run_fill(&fixtures, STAGE_IDX);

        for blk in 0..N_BLKS {
            let col = thermal_start + blk;
            assert_eq!(col_lower[col], 0.0, "dormant col_lower blk {blk}");
            assert_eq!(col_upper[col], 0.0, "dormant col_upper blk {blk}");
        }
    }

    /// A thermal whose `cost_per_mwh` differs per stage and which carries a
    /// per-block generation-bound override at the build stage still prices
    /// `objective[col] == stage_cost_per_mwh * block_hours` for every block — the
    /// block bound never touches the objective.
    #[test]
    fn test_per_block_generation_bound_does_not_change_objective() {
        let build_stage_idx = 1;
        let mut fixtures = ThermalFixtures::new(vec![thermal(1, None, 30.0)]);
        fixtures.set_stage_bounds(0, 0, 0.0, 500.0, 30.0);
        fixtures.set_stage_bounds(0, build_stage_idx, 0.0, 500.0, 45.0);
        fixtures.install_block_overlay();
        fixtures.set_block_override(0, build_stage_idx, 1, None, Some(100.0));

        let (_col_lower, _col_upper, objective, thermal_start) =
            run_fill(&fixtures, build_stage_idx);

        for blk in 0..N_BLKS {
            assert_eq!(
                objective[thermal_start + blk],
                45.0 * BLOCK_HOURS[blk],
                "objective reads the stage-level (not per-block) cost, blk {blk}"
            );
        }
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::float_cmp,
    clippy::similar_names
)]
mod line_contract_pumping_block_bound_tests {
    use std::collections::{BTreeMap, HashMap};

    use cobre_core::entities::energy_contract::{ContractType, EnergyContract};
    use cobre_core::{
        BlockBoundsCountsSpec, BoundsCountsSpec, BoundsDefaults, BusStagePenalties,
        CascadeTopology, ContractBlockBounds, ContractBlockOverride, EntityId, HydroBlockBounds,
        HydroStageBounds, HydroStagePenalties, Line, LineBlockBounds, LineBlockOverride,
        LineStagePenalties, NcsStagePenalties, PenaltiesCountsSpec, PenaltiesDefaults,
        PumpingBlockBounds, PumpingBlockOverride, PumpingStation, ResolvedBlockBounds,
        ResolvedBounds, ResolvedGenericConstraintBounds, ResolvedLoadFactors, ResolvedNcsBounds,
        ResolvedNcsFactors, ResolvedPenalties, ThermalBlockBounds, ThermalStageBounds,
    };
    use cobre_stochastic::par::precompute::PrecomputedPar;

    use crate::hydro_models::{EvaporationModelSet, ProductionModelSet};
    use crate::indexer::HydroCellIndex;
    use crate::lead_time::AnticipatedResolution;
    use crate::resolved_parameters::ResolvedParameters;

    use super::super::layout::ResolvedTables;
    use super::super::test_support::{BLOCK_HOURS, N_BLKS, state_layout_for, three_block_stage};
    use super::{
        ColumnBufs, StageLayout, TemplateBuildCtx, fill_contract_columns, fill_line_columns,
        fill_pumping_columns,
    };

    const N_STAGES: usize = 1;
    const STAGE_IDX: usize = 0;

    fn line(id: i32, entry_stage_id: Option<i32>) -> Line {
        Line {
            id: EntityId(id),
            name: format!("L{id}"),
            operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            source_bus_id: EntityId(1),
            target_bus_id: EntityId(2),
            entry_stage_id,
            exit_stage_id: None,
            direct_capacity_mw: 0.0,
            reverse_capacity_mw: 0.0,
            losses_percent: 0.0,
            exchange_cost: 0.0,
        }
    }

    fn pumping_station(id: i32, entry_stage_id: Option<i32>) -> PumpingStation {
        PumpingStation {
            id: EntityId(id),
            name: format!("P{id}"),
            operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId(1),
            source_hydro_id: EntityId(1),
            destination_hydro_id: EntityId(2),
            entry_stage_id,
            exit_stage_id: None,
            consumption_mw_per_m3s: 0.0,
            min_flow_m3s: 0.0,
            max_flow_m3s: 0.0,
        }
    }

    fn contract(
        id: i32,
        contract_type: ContractType,
        entry_stage_id: Option<i32>,
    ) -> EnergyContract {
        EnergyContract {
            id: EntityId(id),
            name: format!("C{id}"),
            operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId(1),
            contract_type,
            entry_stage_id,
            exit_stage_id: None,
            price_per_mwh: 0.0,
            min_mw: 0.0,
            max_mw: 0.0,
        }
    }

    fn bounds_with(n_lines: usize, n_pumping: usize, n_contracts: usize) -> ResolvedBounds {
        ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 0,
                n_thermals: 0,
                n_lines,
                n_pumping,
                n_contracts,
                n_stages: N_STAGES,
                k_max: 0,
            },
            &BoundsDefaults {
                hydro: HydroStageBounds {
                    min_storage_hm3: 0.0,
                    max_storage_hm3: 0.0,
                    filling_min_rate_m3s: 0.0,
                    water_withdrawal_m3s: 0.0,
                },
                hydro_block: HydroBlockBounds {
                    ..Default::default()
                },
                thermal: ThermalStageBounds { cost_per_mwh: 0.0 },
                thermal_block: ThermalBlockBounds {
                    min_generation_mw: 0.0,
                    max_generation_mw: 0.0,
                },
                line_block: LineBlockBounds {
                    direct_mw: 0.0,
                    reverse_mw: 0.0,
                },
                pumping_block: PumpingBlockBounds {
                    min_flow_m3s: 0.0,
                    max_flow_m3s: 0.0,
                },
                contract_block: ContractBlockBounds {
                    min_mw: 0.0,
                    max_mw: 0.0,
                    price_per_mwh: 0.0,
                },
            },
        )
    }

    fn penalties_with(n_lines: usize) -> ResolvedPenalties {
        ResolvedPenalties::new(
            &PenaltiesCountsSpec {
                n_hydros: 0,
                n_buses: 0,
                n_lines,
                n_ncs: 0,
                n_stages: N_STAGES,
            },
            &PenaltiesDefaults {
                hydro: HydroStagePenalties {
                    spillage_cost: 0.0,
                    diversion_cost: 0.0,
                    turbined_cost: 0.0,
                    storage_violation_below_cost: 0.0,
                    filling_target_violation_cost: 0.0,
                    turbined_violation_below_cost: 0.0,
                    outflow_violation_below_cost: 0.0,
                    outflow_violation_above_cost: 0.0,
                    generation_violation_below_cost: 0.0,
                    evaporation_violation_cost: 0.0,
                    water_withdrawal_violation_cost: 0.0,
                    water_withdrawal_violation_pos_cost: 0.0,
                    water_withdrawal_violation_neg_cost: 0.0,
                    evaporation_violation_pos_cost: 0.0,
                    evaporation_violation_neg_cost: 0.0,
                    inflow_nonnegativity_cost: 0.0,
                },
                bus: BusStagePenalties { excess_cost: 0.0 },
                line: LineStagePenalties { exchange_cost: 0.0 },
                ncs: NcsStagePenalties {
                    curtailment_cost: 0.0,
                },
            },
        )
    }

    /// Owns the borrow targets for a line/pumping/contract-only `TemplateBuildCtx`.
    struct LcpFixtures {
        par_lp: PrecomputedPar,
        hydro_cell_index: HydroCellIndex,
        cascade: CascadeTopology,
        bounds: ResolvedBounds,
        penalties: ResolvedPenalties,
        production_models: ProductionModelSet,
        evaporation_models: EvaporationModelSet,
        resolved_generic_bounds: ResolvedGenericConstraintBounds,
        resolved_load_factors: ResolvedLoadFactors,
        resolved_ncs_bounds: ResolvedNcsBounds,
        resolved_ncs_factors: ResolvedNcsFactors,
        resolved_parameters: ResolvedParameters,
        lines: Vec<Line>,
        pumping_stations: Vec<PumpingStation>,
        contracts: Vec<EnergyContract>,
    }

    impl LcpFixtures {
        fn new(
            lines: Vec<Line>,
            pumping_stations: Vec<PumpingStation>,
            contracts: Vec<EnergyContract>,
        ) -> Self {
            let n_lines = lines.len();
            let n_pumping = pumping_stations.len();
            let n_contracts = contracts.len();
            Self {
                par_lp: PrecomputedPar::default(),
                cascade: CascadeTopology::build(&[]),
                hydro_cell_index: HydroCellIndex::build(&[]),
                bounds: bounds_with(n_lines, n_pumping, n_contracts),
                penalties: penalties_with(n_lines),
                production_models: ProductionModelSet::new(vec![], 0, 1),
                evaporation_models: EvaporationModelSet::new(vec![]),
                resolved_generic_bounds: ResolvedGenericConstraintBounds::empty(),
                resolved_load_factors: ResolvedLoadFactors::empty(),
                resolved_ncs_bounds: ResolvedNcsBounds::empty(),
                resolved_ncs_factors: ResolvedNcsFactors::empty(),
                resolved_parameters: ResolvedParameters {
                    per_param: vec![],
                    id_to_slot: vec![],
                    cost_scale_factor: 1_000_000.0,
                },
                lines,
                pumping_stations,
                contracts,
            }
        }

        fn set_line_bounds(&mut self, l_idx: usize, direct_mw: f64, reverse_mw: f64) {
            let cell = self.bounds.line_bounds_mut(l_idx, STAGE_IDX);
            cell.direct_mw = direct_mw;
            cell.reverse_mw = reverse_mw;
        }

        fn set_line_exchange_cost(&mut self, l_idx: usize, exchange_cost: f64) {
            self.penalties
                .line_penalties_mut(l_idx, STAGE_IDX)
                .exchange_cost = exchange_cost;
        }

        fn set_pumping_bounds(&mut self, p_idx: usize, min_flow_m3s: f64, max_flow_m3s: f64) {
            let cell = self.bounds.pumping_bounds_mut(p_idx, STAGE_IDX);
            cell.min_flow_m3s = min_flow_m3s;
            cell.max_flow_m3s = max_flow_m3s;
        }

        fn set_contract_bounds(&mut self, c_idx: usize, min_mw: f64, max_mw: f64, price: f64) {
            let cell = self.bounds.contract_bounds_mut(c_idx, STAGE_IDX);
            cell.min_mw = min_mw;
            cell.max_mw = max_mw;
            cell.price_per_mwh = price;
        }

        fn install_block_overlay(&mut self) {
            self.bounds
                .set_block_overlay(ResolvedBlockBounds::new(&BlockBoundsCountsSpec {
                    n_hydros: 0,
                    n_thermals: 0,
                    n_lines: self.lines.len(),
                    n_pumping: self.pumping_stations.len(),
                    n_contracts: self.contracts.len(),
                    n_stages: N_STAGES,
                    max_blocks: N_BLKS,
                }));
        }

        fn set_line_block_override(
            &mut self,
            l_idx: usize,
            block_idx: usize,
            over: LineBlockOverride,
        ) {
            *self
                .bounds
                .block_overlay_mut()
                .line_override_mut(l_idx, STAGE_IDX, block_idx)
                .expect("overlay cell must exist for a fixture-sized overlay") = over;
        }

        fn set_pumping_block_override(
            &mut self,
            p_idx: usize,
            block_idx: usize,
            over: PumpingBlockOverride,
        ) {
            *self
                .bounds
                .block_overlay_mut()
                .pumping_override_mut(p_idx, STAGE_IDX, block_idx)
                .expect("overlay cell must exist for a fixture-sized overlay") = over;
        }

        fn set_contract_block_override(
            &mut self,
            c_idx: usize,
            block_idx: usize,
            over: ContractBlockOverride,
        ) {
            *self
                .bounds
                .block_overlay_mut()
                .contract_override_mut(c_idx, STAGE_IDX, block_idx)
                .expect("overlay cell must exist for a fixture-sized overlay") = over;
        }

        fn make_ctx(&self) -> TemplateBuildCtx<'_> {
            let n_contract_import = self
                .contracts
                .iter()
                .filter(|c| c.contract_type == ContractType::Import)
                .count();
            let n_contract_export = self
                .contracts
                .iter()
                .filter(|c| c.contract_type == ContractType::Export)
                .count();
            let line_pos: BTreeMap<EntityId, usize> = self
                .lines
                .iter()
                .enumerate()
                .map(|(i, l)| (l.id, i))
                .collect();
            let pumping_pos: BTreeMap<EntityId, usize> = self
                .pumping_stations
                .iter()
                .enumerate()
                .map(|(i, p)| (p.id, i))
                .collect();
            let contract_pos: BTreeMap<EntityId, usize> = self
                .contracts
                .iter()
                .enumerate()
                .map(|(i, c)| (c.id, i))
                .collect();
            TemplateBuildCtx {
                hydros: &[],
                thermals: &[],
                lines: &self.lines,
                buses: &[],
                load_models: &[],
                cascade: &self.cascade,
                hydro_cell_index: &self.hydro_cell_index,
                resolved: ResolvedTables {
                    bounds: &self.bounds,
                    penalties: &self.penalties,
                    resolved_generic_bounds: &self.resolved_generic_bounds,
                    resolved_load_factors: &self.resolved_load_factors,
                    resolved_ncs_bounds: &self.resolved_ncs_bounds,
                    resolved_ncs_factors: &self.resolved_ncs_factors,
                    resolved_parameters: &self.resolved_parameters,
                },
                hydro_pos: BTreeMap::new(),
                thermal_pos: BTreeMap::new(),
                line_pos,
                bus_pos: BTreeMap::new(),
                par_lp: &self.par_lp,
                production_models: &self.production_models,
                evaporation_models: &self.evaporation_models,
                generic_constraints: &[],
                non_controllable_sources: &[],
                pumping_stations: &self.pumping_stations,
                pumping_pos,
                n_pumping: self.pumping_stations.len(),
                contracts: &self.contracts,
                contract_pos,
                n_contract_import,
                n_contract_export,
                diversion_upstream: HashMap::new(),
                arc_stage_weights: HashMap::new(),
                arc_spread_chrono: HashMap::new(),
                arc_arrival_density: HashMap::new(),
                per_stage_mask: Vec::new(),
                post_study_resolved: crate::setup::PostStudyResolved::default(),
                n_hydros: 0,
                n_thermals: 0,
                n_lines: self.lines.len(),
                n_buses: 0,
                max_par_order: 0,
                n_anticipated: 0,
                k_max: 0,
                anticipated_lead_stages: vec![],
                anticipated_thermal_indices: vec![],
                anticipated_windows: vec![],
                anticipated_resolution: AnticipatedResolution::default(),
                study_stage_ids: (0..N_STAGES as i32).collect(),
                delivery_stage_ids: (0..N_STAGES as i32).collect(),
                has_penalty: false,
                delivery_cumulative_discount_factors: vec![1.0; N_STAGES],
                delivery_total_hours: vec![BLOCK_HOURS.iter().sum(); N_STAGES],
                filling_v_target: BTreeMap::new(),
            }
        }
    }

    /// The per-family column start offsets `run_fill` resolves while the
    /// `StageLayout` is alive, plus `n_blks` — enough to reconstruct every
    /// block-major address (`start + entity_idx * n_blks + blk`) after the
    /// layout itself is dropped.
    struct FillOffsets {
        line_fwd_start: usize,
        line_rev_start: usize,
        pumping_start: usize,
        contract_import_start: usize,
        contract_export_start: usize,
        n_blks: usize,
    }

    impl FillOffsets {
        fn at(&self, start: usize, entity_idx: usize, blk: usize) -> usize {
            start + entity_idx * self.n_blks + blk
        }
    }

    struct FillResult {
        col_lower: Vec<f64>,
        col_upper: Vec<f64>,
        objective: Vec<f64>,
        offsets: FillOffsets,
    }

    /// Run the line, pumping, and contract column fills against `fixtures` at
    /// `stage_idx`, over a three-block stage.
    fn run_fill(fixtures: &LcpFixtures, stage_idx: usize) -> FillResult {
        let stage = three_block_stage(stage_idx);
        let ctx = fixtures.make_ctx();
        let state = state_layout_for(&ctx);
        let layout = StageLayout::new(&ctx, &state, &stage, stage_idx);

        let mut col_lower = vec![0.0_f64; layout.num_cols];
        let mut col_upper = vec![f64::INFINITY; layout.num_cols];
        let mut objective = vec![0.0_f64; layout.num_cols];
        let mut bufs = ColumnBufs {
            col_lower: &mut col_lower,
            col_upper: &mut col_upper,
            objective: &mut objective,
        };
        fill_line_columns(&ctx, &stage, stage_idx, &layout, &mut bufs);
        fill_pumping_columns(&ctx, &stage, stage_idx, &layout, &mut bufs);
        fill_contract_columns(&ctx, &stage, stage_idx, &layout, &mut bufs);

        let offsets = FillOffsets {
            line_fwd_start: layout.equipment.line_fwd.start,
            line_rev_start: layout.equipment.line_rev.start,
            pumping_start: layout.equipment.col_pumping_start,
            contract_import_start: layout.equipment.col_contract_import_start,
            contract_export_start: layout.equipment.col_contract_export_start,
            n_blks: layout.n_blks,
        };

        FillResult {
            col_lower,
            col_upper,
            objective,
            offsets,
        }
    }

    /// Two lines, one pumping station, two contracts (one import, one export),
    /// three blocks, empty overlay: every line forward/reverse, pumping, and
    /// contract import/export column reads bit-identical to the stage-level
    /// formula (`*_bounds_at_block` falls through to the stage cell at every
    /// block).
    #[test]
    fn test_line_contract_pumping_columns_bit_identical_without_overlay() {
        let lines = vec![line(1, None), line(2, None)];
        let pumping_stations = vec![pumping_station(1, None)];
        let contracts = vec![
            contract(1, ContractType::Import, None),
            contract(2, ContractType::Export, None),
        ];
        let mut fixtures = LcpFixtures::new(lines, pumping_stations, contracts);
        fixtures.set_line_bounds(0, 100.0, 80.0);
        fixtures.set_line_exchange_cost(0, 2.0);
        fixtures.set_line_bounds(1, 50.0, 40.0);
        fixtures.set_line_exchange_cost(1, 3.0);
        fixtures.set_pumping_bounds(0, 5.0, 60.0);
        fixtures.set_contract_bounds(0, 10.0, 90.0, 25.0);
        fixtures.set_contract_bounds(1, 0.0, 70.0, -15.0);

        let result = run_fill(&fixtures, STAGE_IDX);
        let off = &result.offsets;

        for (blk, &hours) in BLOCK_HOURS.iter().enumerate() {
            let l0_fwd = off.at(off.line_fwd_start, 0, blk);
            assert_eq!(result.col_upper[l0_fwd].to_bits(), 100.0_f64.to_bits());
            assert_eq!(result.objective[l0_fwd].to_bits(), (2.0 * hours).to_bits());
            let l0_rev = off.at(off.line_rev_start, 0, blk);
            assert_eq!(result.col_upper[l0_rev].to_bits(), 80.0_f64.to_bits());
            assert_eq!(result.objective[l0_rev].to_bits(), (2.0 * hours).to_bits());

            let l1_fwd = off.at(off.line_fwd_start, 1, blk);
            assert_eq!(result.col_upper[l1_fwd].to_bits(), 50.0_f64.to_bits());
            assert_eq!(result.objective[l1_fwd].to_bits(), (3.0 * hours).to_bits());
            let l1_rev = off.at(off.line_rev_start, 1, blk);
            assert_eq!(result.col_upper[l1_rev].to_bits(), 40.0_f64.to_bits());
            assert_eq!(result.objective[l1_rev].to_bits(), (3.0 * hours).to_bits());

            let pump = off.at(off.pumping_start, 0, blk);
            assert_eq!(result.col_lower[pump].to_bits(), 5.0_f64.to_bits());
            assert_eq!(result.col_upper[pump].to_bits(), 60.0_f64.to_bits());
            assert_eq!(result.objective[pump].to_bits(), 0.0_f64.to_bits());

            let import = off.at(off.contract_import_start, 0, blk);
            assert_eq!(result.col_lower[import].to_bits(), 10.0_f64.to_bits());
            assert_eq!(result.col_upper[import].to_bits(), 90.0_f64.to_bits());
            assert_eq!(result.objective[import].to_bits(), (25.0 * hours).to_bits());

            let export = off.at(off.contract_export_start, 0, blk);
            assert_eq!(result.col_lower[export].to_bits(), 0.0_f64.to_bits());
            assert_eq!(result.col_upper[export].to_bits(), 70.0_f64.to_bits());
            assert_eq!(
                result.objective[export].to_bits(),
                (-15.0 * hours).to_bits()
            );
        }
    }

    /// A line with stage-wide `direct_mw = 1000.0` / `reverse_mw = 500.0` and
    /// two separate block overrides — `direct_mw = 800.0` at block 2,
    /// `reverse_mw = 200.0` at block 0: each override caps only its own
    /// (block, direction) pair.
    #[test]
    fn test_per_block_line_cap_binds_only_its_own_block() {
        let mut fixtures = LcpFixtures::new(vec![line(1, None)], vec![], vec![]);
        fixtures.set_line_bounds(0, 1000.0, 500.0);
        fixtures.install_block_overlay();
        fixtures.set_line_block_override(
            0,
            2,
            LineBlockOverride {
                direct_mw: Some(800.0),
                ..Default::default()
            },
        );
        fixtures.set_line_block_override(
            0,
            0,
            LineBlockOverride {
                reverse_mw: Some(200.0),
                ..Default::default()
            },
        );

        let result = run_fill(&fixtures, STAGE_IDX);
        let off = &result.offsets;
        let fwd_upper: Vec<f64> = (0..N_BLKS)
            .map(|blk| result.col_upper[off.at(off.line_fwd_start, 0, blk)])
            .collect();
        assert_eq!(
            fwd_upper,
            vec![1000.0, 1000.0, 800.0],
            "only block 2 is bound to the direct override"
        );
        let rev_upper: Vec<f64> = (0..N_BLKS)
            .map(|blk| result.col_upper[off.at(off.line_rev_start, 0, blk)])
            .collect();
        assert_eq!(
            rev_upper,
            vec![200.0, 500.0, 500.0],
            "only block 0 is bound to the reverse override"
        );
    }

    /// A contract with stage-wide `price_per_mwh = 80.0` and TWO block overrides
    /// — block 0 sets `max_mw = 50.0`/`price_per_mwh = 120.0`, block 2 sets
    /// `min_mw = 30.0` — plus a pumping station with TWO block overrides —
    /// block 1 sets `max_flow_m3s = 20.0`, block 0 sets `min_flow_m3s = 15.0`:
    /// each override binds only its own (entity, block, column) triple; every
    /// other block keeps the stage-level value.
    #[test]
    fn test_per_block_contract_price_and_pumping_bounds_bind() {
        let mut fixtures = LcpFixtures::new(
            vec![],
            vec![pumping_station(1, None)],
            vec![contract(1, ContractType::Import, None)],
        );
        fixtures.set_contract_bounds(0, 0.0, 200.0, 80.0);
        fixtures.set_pumping_bounds(0, 0.0, 100.0);
        fixtures.install_block_overlay();
        fixtures.set_contract_block_override(
            0,
            0,
            ContractBlockOverride {
                max_mw: Some(50.0),
                price_per_mwh: Some(120.0),
                ..Default::default()
            },
        );
        fixtures.set_contract_block_override(
            0,
            2,
            ContractBlockOverride {
                min_mw: Some(30.0),
                ..Default::default()
            },
        );
        fixtures.set_pumping_block_override(
            0,
            1,
            PumpingBlockOverride {
                max_flow_m3s: Some(20.0),
                ..Default::default()
            },
        );
        fixtures.set_pumping_block_override(
            0,
            0,
            PumpingBlockOverride {
                min_flow_m3s: Some(15.0),
                ..Default::default()
            },
        );

        let result = run_fill(&fixtures, STAGE_IDX);
        let off = &result.offsets;

        let contract_upper: Vec<f64> = (0..N_BLKS)
            .map(|blk| result.col_upper[off.at(off.contract_import_start, 0, blk)])
            .collect();
        assert_eq!(
            contract_upper,
            vec![50.0, 200.0, 200.0],
            "only block 0 is bound to the contract max_mw override"
        );
        let contract_lower: Vec<f64> = (0..N_BLKS)
            .map(|blk| result.col_lower[off.at(off.contract_import_start, 0, blk)])
            .collect();
        assert_eq!(
            contract_lower,
            vec![0.0, 0.0, 30.0],
            "only block 2 is bound to the contract min_mw override"
        );
        let contract_obj: Vec<f64> = (0..N_BLKS)
            .map(|blk| result.objective[off.at(off.contract_import_start, 0, blk)])
            .collect();
        assert_eq!(
            contract_obj,
            vec![
                120.0 * BLOCK_HOURS[0],
                80.0 * BLOCK_HOURS[1],
                80.0 * BLOCK_HOURS[2],
            ],
            "only block 0 prices at the overridden price"
        );

        let pumping_upper: Vec<f64> = (0..N_BLKS)
            .map(|blk| result.col_upper[off.at(off.pumping_start, 0, blk)])
            .collect();
        assert_eq!(
            pumping_upper,
            vec![100.0, 20.0, 100.0],
            "only block 1 is bound to the pumping max_flow_m3s override"
        );
        let pumping_lower: Vec<f64> = (0..N_BLKS)
            .map(|blk| result.col_lower[off.at(off.pumping_start, 0, blk)])
            .collect();
        assert_eq!(
            pumping_lower,
            vec![15.0, 0.0, 0.0],
            "only block 0 is bound to the pumping min_flow_m3s override"
        );
    }

    /// A dormant line, pumping station, and contract (each `entry_stage_id`
    /// after the build stage) carrying a `block_id` override with a large
    /// override value still get `[0, 0]` at every block — the commissioning
    /// gate wins over the per-block bound.
    #[test]
    fn test_dormant_entity_ignores_per_block_bound() {
        let mut fixtures = LcpFixtures::new(
            vec![line(1, Some(1))],
            vec![pumping_station(1, Some(1))],
            vec![contract(1, ContractType::Import, Some(1))],
        );
        fixtures.set_line_bounds(0, 1000.0, 900.0);
        fixtures.set_pumping_bounds(0, 5.0, 60.0);
        fixtures.set_contract_bounds(0, 10.0, 90.0, 80.0);
        fixtures.install_block_overlay();
        fixtures.set_line_block_override(
            0,
            1,
            LineBlockOverride {
                direct_mw: Some(800.0),
                reverse_mw: Some(700.0),
            },
        );
        fixtures.set_pumping_block_override(
            0,
            1,
            PumpingBlockOverride {
                min_flow_m3s: Some(999.0),
                max_flow_m3s: Some(999.0),
            },
        );
        fixtures.set_contract_block_override(
            0,
            1,
            ContractBlockOverride {
                min_mw: Some(999.0),
                max_mw: Some(999.0),
                price_per_mwh: Some(999.0),
            },
        );

        let result = run_fill(&fixtures, STAGE_IDX);
        let off = &result.offsets;
        for blk in 0..N_BLKS {
            assert_eq!(
                result.col_upper[off.at(off.line_fwd_start, 0, blk)],
                0.0,
                "dormant line fwd, blk {blk}"
            );
            assert_eq!(
                result.col_upper[off.at(off.line_rev_start, 0, blk)],
                0.0,
                "dormant line rev, blk {blk}"
            );
            assert_eq!(
                result.col_lower[off.at(off.pumping_start, 0, blk)],
                0.0,
                "dormant pumping lower, blk {blk}"
            );
            assert_eq!(
                result.col_upper[off.at(off.pumping_start, 0, blk)],
                0.0,
                "dormant pumping upper, blk {blk}"
            );
            assert_eq!(
                result.col_lower[off.at(off.contract_import_start, 0, blk)],
                0.0,
                "dormant contract lower, blk {blk}"
            );
            assert_eq!(
                result.col_upper[off.at(off.contract_import_start, 0, blk)],
                0.0,
                "dormant contract upper, blk {blk}"
            );
        }
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::float_cmp,
    clippy::similar_names
)]
mod hydro_block_bound_tests {
    use std::collections::{BTreeMap, HashMap};

    use cobre_core::entities::hydro::HydroGenerationModel;
    use cobre_core::{
        BlockBoundsCountsSpec, BoundsCountsSpec, BoundsDefaults, BusStagePenalties,
        CascadeTopology, ContractBlockBounds, EntityId, Hydro, HydroBlockBounds,
        HydroBlockOverride, HydroStageBounds, HydroStagePenalties, HydroUnitGroupBoundsCountsSpec,
        HydroUnitGroupOverride, LineBlockBounds, LineStagePenalties, NcsStagePenalties,
        PenaltiesCountsSpec, PenaltiesDefaults, PumpingBlockBounds, ResolvedBlockBounds,
        ResolvedBounds, ResolvedGenericConstraintBounds, ResolvedHydroUnitGroupBounds,
        ResolvedLoadFactors, ResolvedNcsBounds, ResolvedNcsFactors, ResolvedPenalties,
        ThermalBlockBounds, ThermalStageBounds,
    };
    use cobre_stochastic::par::precompute::PrecomputedPar;

    use crate::hydro_models::{
        EvaporationModel, EvaporationModelSet, FphaPlane, ProductionModelSet,
        ResolvedProductionModel,
    };
    use crate::indexer::{BlockIdx, FphaCellLocal, HydroCellIndex};
    use crate::lead_time::AnticipatedResolution;
    use crate::resolved_parameters::ResolvedParameters;

    use super::super::layout::ResolvedTables;
    use super::super::rows::fill_operational_violation_rows;
    use super::super::test_support::{
        BLOCK_HOURS, N_BLKS, state_layout_for, three_block_stage, zero_hydro_penalties,
    };
    use super::{
        ColumnBufs, StageLayout, TemplateBuildCtx, fill_diversion_columns,
        fill_fpha_generation_columns, fill_operational_slack_columns, fill_turbine_columns,
    };

    const N_STAGES: usize = 1;
    const STAGE_IDX: usize = 0;

    fn fixture_hydro(id: i32, entry_stage_id: Option<i32>) -> Hydro {
        let mut hydro = Hydro {
            unit_groups: Vec::new(),
            id: EntityId(id),
            name: format!("H{id}"),
            operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            downstream_id: None,
            travel_time_hours: None,
            entry_stage_id,
            exit_stage_id: None,
            min_storage_hm3: 0.0,
            max_storage_hm3: 100.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: HydroGenerationModel::ConstantProductivity,
            min_turbined_m3s: 0.0,
            // Deliberately non-binding, not a realistic install capacity: this
            // plant's own bound becomes its mirrored group's box
            // (declare_mirror_unit_group), which every cell column bound now
            // sums against and would otherwise cap below the resolved bounds
            // this module's tests exercise — a bounds row raising a plant
            // above its declared capacity is rejected in a real deck, so this
            // stays generous rather than tracking each test's resolved value.
            max_turbined_m3s: 1_000_000.0,
            specific_productivity_mw_per_m3s_per_m: None,
            min_generation_mw: 0.0,
            max_generation_mw: 1_000_000.0,
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: None,
            evaporation_reference_volumes_hm3: None,
            diversion: None,
            filling: None,
            penalties: zero_hydro_penalties(),
        };
        hydro.declare_mirror_unit_group(EntityId(1));
        hydro
    }

    fn hydro_block_bounds(
        min_turbined_m3s: f64,
        max_turbined_m3s: f64,
        min_outflow_m3s: f64,
        max_outflow_m3s: Option<f64>,
        min_generation_mw: f64,
        max_generation_mw: f64,
        max_diversion_m3s: Option<f64>,
    ) -> HydroBlockBounds {
        HydroBlockBounds {
            min_turbined_m3s,
            max_turbined_m3s,
            min_outflow_m3s,
            max_outflow_m3s,
            min_generation_mw,
            max_generation_mw,
            max_diversion_m3s,
            ..Default::default()
        }
    }

    fn hydro_stage_penalties(
        turbined_cost: f64,
        diversion_cost: f64,
        outflow_violation_below_cost: f64,
        outflow_violation_above_cost: f64,
        turbined_violation_below_cost: f64,
        generation_violation_below_cost: f64,
    ) -> HydroStagePenalties {
        HydroStagePenalties {
            spillage_cost: 0.0,
            diversion_cost,
            turbined_cost,
            storage_violation_below_cost: 0.0,
            filling_target_violation_cost: 0.0,
            turbined_violation_below_cost,
            outflow_violation_below_cost,
            outflow_violation_above_cost,
            generation_violation_below_cost,
            evaporation_violation_cost: 0.0,
            water_withdrawal_violation_cost: 0.0,
            water_withdrawal_violation_pos_cost: 0.0,
            water_withdrawal_violation_neg_cost: 0.0,
            evaporation_violation_pos_cost: 0.0,
            evaporation_violation_neg_cost: 0.0,
            inflow_nonnegativity_cost: 0.0,
        }
    }

    fn bounds_with_hydros(n_hydros: usize) -> ResolvedBounds {
        ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros,
                n_thermals: 0,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages: N_STAGES,
                k_max: 0,
            },
            &BoundsDefaults {
                hydro: HydroStageBounds {
                    min_storage_hm3: 0.0,
                    max_storage_hm3: 100.0,
                    filling_min_rate_m3s: 0.0,
                    water_withdrawal_m3s: 0.0,
                },
                hydro_block: hydro_block_bounds(0.0, 0.0, 0.0, None, 0.0, 0.0, None),
                thermal: ThermalStageBounds { cost_per_mwh: 0.0 },
                thermal_block: ThermalBlockBounds {
                    min_generation_mw: 0.0,
                    max_generation_mw: 0.0,
                },
                line_block: LineBlockBounds {
                    direct_mw: 0.0,
                    reverse_mw: 0.0,
                },
                pumping_block: PumpingBlockBounds {
                    min_flow_m3s: 0.0,
                    max_flow_m3s: 0.0,
                },
                contract_block: ContractBlockBounds {
                    min_mw: 0.0,
                    max_mw: 0.0,
                    price_per_mwh: 0.0,
                },
            },
        )
    }

    fn penalties_with_hydros(n_hydros: usize) -> ResolvedPenalties {
        ResolvedPenalties::new(
            &PenaltiesCountsSpec {
                n_hydros,
                n_buses: 0,
                n_lines: 0,
                n_ncs: 0,
                n_stages: N_STAGES,
            },
            &PenaltiesDefaults {
                hydro: hydro_stage_penalties(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                bus: BusStagePenalties { excess_cost: 0.0 },
                line: LineStagePenalties { exchange_cost: 0.0 },
                ncs: NcsStagePenalties {
                    curtailment_cost: 0.0,
                },
            },
        )
    }

    /// Owns the borrow targets for a hydro-only `TemplateBuildCtx`, sized to
    /// however many hydros the test passes in.
    struct HydroBlockFixtures {
        par_lp: PrecomputedPar,
        hydros: Vec<Hydro>,
        hydro_cell_index: HydroCellIndex,
        cascade: CascadeTopology,
        bounds: ResolvedBounds,
        penalties: ResolvedPenalties,
        production_models: ProductionModelSet,
        evaporation_models: EvaporationModelSet,
        resolved_generic_bounds: ResolvedGenericConstraintBounds,
        resolved_load_factors: ResolvedLoadFactors,
        resolved_ncs_bounds: ResolvedNcsBounds,
        resolved_ncs_factors: ResolvedNcsFactors,
        resolved_parameters: ResolvedParameters,
    }

    impl HydroBlockFixtures {
        /// `productivities[i]` is hydro `i`'s constant-productivity model input.
        fn new(hydros: Vec<Hydro>, productivities: &[f64]) -> Self {
            let n_hydros = hydros.len();
            assert_eq!(
                productivities.len(),
                n_hydros,
                "one productivity per fixture hydro"
            );
            let cascade = CascadeTopology::build(&hydros);
            let hydro_cell_index = HydroCellIndex::build(&hydros);
            Self {
                par_lp: PrecomputedPar::default(),
                hydros,
                hydro_cell_index,
                cascade,
                bounds: bounds_with_hydros(n_hydros),
                penalties: penalties_with_hydros(n_hydros),
                production_models: ProductionModelSet::new(
                    productivities
                        .iter()
                        .map(|&productivity| {
                            vec![
                                ResolvedProductionModel::ConstantProductivity { productivity };
                                N_STAGES
                            ]
                        })
                        .collect(),
                    n_hydros,
                    N_STAGES,
                ),
                evaporation_models: EvaporationModelSet::new(vec![
                    EvaporationModel::None;
                    n_hydros
                ]),
                resolved_generic_bounds: ResolvedGenericConstraintBounds::empty(),
                resolved_load_factors: ResolvedLoadFactors::empty(),
                resolved_ncs_bounds: ResolvedNcsBounds::empty(),
                resolved_ncs_factors: ResolvedNcsFactors::empty(),
                resolved_parameters: ResolvedParameters {
                    per_param: vec![],
                    id_to_slot: vec![],
                    cost_scale_factor: 1_000_000.0,
                },
            }
        }

        /// Also syncs `hydro.unit_groups[0]`'s `min_turbined_m3s`/`min_generation_mw`
        /// to `hb`'s: `fixture_hydro`'s `declare_mirror_unit_group` mirrors the
        /// entity's (hard-coded zero) declared minimum at construction time, before
        /// this per-stage bound is known, and the min-floor rows now read the
        /// GROUP's own resolved minimum, never `HydroBlockBounds`. Every fixture
        /// hydro carries exactly one group at position 0, and this module's
        /// `N_STAGES == 1`, so overwriting it here tracks the single call's value
        /// with no cross-stage ambiguity.
        fn set_hydro_bounds(&mut self, h_idx: usize, stage_idx: usize, hb: HydroBlockBounds) {
            self.hydros[h_idx].unit_groups[0].min_turbined_m3s = hb.min_turbined_m3s;
            self.hydros[h_idx].unit_groups[0].min_generation_mw = hb.min_generation_mw;
            *self.bounds.hydro_block_base_mut(h_idx, stage_idx) = hb;
        }

        fn set_hydro_penalties(&mut self, h_idx: usize, stage_idx: usize, hp: HydroStagePenalties) {
            *self.penalties.hydro_penalties_mut(h_idx, stage_idx) = hp;
        }

        fn install_block_overlay(&mut self) {
            self.bounds
                .set_block_overlay(ResolvedBlockBounds::new(&BlockBoundsCountsSpec {
                    n_hydros: self.hydros.len(),
                    n_thermals: 0,
                    n_lines: 0,
                    n_pumping: 0,
                    n_contracts: 0,
                    n_stages: N_STAGES,
                    max_blocks: N_BLKS,
                }));
        }

        fn set_hydro_block_override(
            &mut self,
            h_idx: usize,
            stage_idx: usize,
            block_idx: usize,
            over: HydroBlockOverride,
        ) {
            *self
                .bounds
                .block_overlay_mut()
                .hydro_override_mut(h_idx, stage_idx, block_idx)
                .expect("overlay cell must exist for a fixture-sized overlay") = over;
        }

        /// Install the group-bounds overlay, sized from each hydro's own
        /// declared `unit_groups` count (every fixture hydro carries exactly one,
        /// via `declare_mirror_unit_group`).
        fn install_group_overlay(&mut self) {
            let groups_per_plant: Vec<usize> =
                self.hydros.iter().map(|h| h.unit_groups.len()).collect();
            self.bounds
                .set_group_overlay(ResolvedHydroUnitGroupBounds::new(
                    &HydroUnitGroupBoundsCountsSpec {
                        groups_per_plant: &groups_per_plant,
                        n_stages: N_STAGES,
                        max_blocks: N_BLKS,
                    },
                ));
        }

        fn set_group_block_override(
            &mut self,
            h_idx: usize,
            group_pos: usize,
            stage_idx: usize,
            block_idx: usize,
            over: HydroUnitGroupOverride,
        ) {
            *self
                .bounds
                .group_overlay_mut()
                .block_override_mut(h_idx, group_pos, stage_idx, block_idx)
                .expect("overlay cell must exist for a fixture-sized overlay") = over;
        }

        fn make_ctx(&self) -> TemplateBuildCtx<'_> {
            let mut hydro_pos = BTreeMap::new();
            for (i, h) in self.hydros.iter().enumerate() {
                hydro_pos.insert(h.id, i);
            }
            TemplateBuildCtx {
                hydros: &self.hydros,
                thermals: &[],
                lines: &[],
                buses: &[],
                load_models: &[],
                cascade: &self.cascade,
                hydro_cell_index: &self.hydro_cell_index,
                resolved: ResolvedTables {
                    bounds: &self.bounds,
                    penalties: &self.penalties,
                    resolved_generic_bounds: &self.resolved_generic_bounds,
                    resolved_load_factors: &self.resolved_load_factors,
                    resolved_ncs_bounds: &self.resolved_ncs_bounds,
                    resolved_ncs_factors: &self.resolved_ncs_factors,
                    resolved_parameters: &self.resolved_parameters,
                },
                hydro_pos,
                thermal_pos: BTreeMap::new(),
                line_pos: BTreeMap::new(),
                bus_pos: BTreeMap::new(),
                par_lp: &self.par_lp,
                production_models: &self.production_models,
                evaporation_models: &self.evaporation_models,
                generic_constraints: &[],
                non_controllable_sources: &[],
                pumping_stations: &[],
                pumping_pos: BTreeMap::new(),
                n_pumping: 0,
                contracts: &[],
                contract_pos: BTreeMap::new(),
                n_contract_import: 0,
                n_contract_export: 0,
                diversion_upstream: HashMap::new(),
                arc_stage_weights: HashMap::new(),
                arc_spread_chrono: HashMap::new(),
                arc_arrival_density: HashMap::new(),
                per_stage_mask: Vec::new(),
                post_study_resolved: crate::setup::PostStudyResolved::default(),
                n_hydros: self.hydros.len(),
                n_thermals: 0,
                n_lines: 0,
                n_buses: 0,
                max_par_order: 0,
                n_anticipated: 0,
                k_max: 0,
                anticipated_lead_stages: vec![],
                anticipated_thermal_indices: vec![],
                anticipated_windows: vec![],
                anticipated_resolution: AnticipatedResolution::default(),
                study_stage_ids: (0..N_STAGES as i32).collect(),
                delivery_stage_ids: (0..N_STAGES as i32).collect(),
                has_penalty: false,
                delivery_cumulative_discount_factors: vec![1.0; N_STAGES],
                delivery_total_hours: vec![BLOCK_HOURS.iter().sum(); N_STAGES],
                filling_v_target: BTreeMap::new(),
            }
        }
    }

    /// The per-family column/row starting offsets `run_fill` resolves while the
    /// `StageLayout` is alive, plus `n_blks` — enough for `at` to address every
    /// block-major cell after the layout itself is dropped.
    struct FillOffsets {
        turbine: usize,
        diversion: usize,
        outflow_below: usize,
        outflow_above: usize,
        turbine_below: usize,
        generation_below: usize,
        min_outflow_row: usize,
        max_outflow_row: usize,
        min_turbine_row: usize,
        min_generation_row: usize,
        n_blks: usize,
    }

    impl FillOffsets {
        fn at(&self, start: usize, h_idx: usize, blk: usize) -> usize {
            start + h_idx * self.n_blks + blk
        }
    }

    /// Hydro 0's per-block values of `buf` for the family starting at `start`.
    fn per_block(buf: &[f64], off: &FillOffsets, start: usize) -> Vec<f64> {
        (0..N_BLKS).map(|blk| buf[off.at(start, 0, blk)]).collect()
    }

    struct FillResult {
        col_lower: Vec<f64>,
        col_upper: Vec<f64>,
        objective: Vec<f64>,
        row_lower: Vec<f64>,
        row_upper: Vec<f64>,
        offsets: FillOffsets,
    }

    /// Run the turbine, diversion, and operational-slack column fills plus the
    /// operational-violation row fill against `fixtures` at `stage_idx`, over a
    /// three-block stage.
    fn run_fill(fixtures: &HydroBlockFixtures, stage_idx: usize) -> FillResult {
        let stage = three_block_stage(stage_idx);
        let ctx = fixtures.make_ctx();
        let state = state_layout_for(&ctx);
        let layout = StageLayout::new(&ctx, &state, &stage, stage_idx);

        let mut col_lower = vec![0.0_f64; layout.num_cols];
        let mut col_upper = vec![f64::INFINITY; layout.num_cols];
        let mut objective = vec![0.0_f64; layout.num_cols];
        let mut bufs = ColumnBufs {
            col_lower: &mut col_lower,
            col_upper: &mut col_upper,
            objective: &mut objective,
        };
        fill_turbine_columns(&ctx, &stage, stage_idx, &layout, &mut bufs);
        fill_diversion_columns(&ctx, &stage, stage_idx, &layout, &mut bufs);
        fill_operational_slack_columns(&ctx, &stage, stage_idx, &layout, &mut bufs);

        let mut row_lower = vec![0.0_f64; layout.rows.num_rows];
        let mut row_upper = vec![0.0_f64; layout.rows.num_rows];
        fill_operational_violation_rows(&ctx, stage_idx, &layout, &mut row_lower, &mut row_upper);

        let offsets = FillOffsets {
            turbine: layout.equipment.turbine.start,
            diversion: layout.equipment.diversion.start,
            outflow_below: layout.slack.oper_violation.outflow_below_slack.start,
            outflow_above: layout.slack.oper_violation.outflow_above_slack.start,
            turbine_below: layout.slack.oper_violation.turbine_below_slack.start,
            generation_below: layout.slack.oper_violation.generation_below_slack.start,
            min_outflow_row: layout.slack.oper_violation.min_outflow_rows.start,
            max_outflow_row: layout.slack.oper_violation.max_outflow_rows.start,
            min_turbine_row: layout.slack.oper_violation.min_turbine_rows.start,
            min_generation_row: layout.slack.oper_violation.min_generation_rows.start,
            n_blks: layout.n_blks,
        };

        FillResult {
            col_lower,
            col_upper,
            objective,
            row_lower,
            row_upper,
            offsets,
        }
    }

    /// One hydro's expected turbine/diversion/slack/row values for
    /// [`test_hydro_block_fill_sites_bit_identical_without_overlay`], read off the
    /// fixture's own stage-level bound/penalty writes (never recomputed via the
    /// production formula).
    struct Expected {
        turb_upper: f64,
        turbined_cost: f64,
        max_div: f64,
        diversion_cost: f64,
        active: [bool; 4],
        costs: [f64; 4],
        min_outflow_row_lower: f64,
        max_outflow_row_upper: f64,
        min_turbine_row_lower: f64,
        min_generation_row_lower: f64,
    }

    /// Assert every turbine, diversion, operational-slack, and operational-violation
    /// row value at `(h_idx, blk)` against `exp`, bit-for-bit.
    fn assert_block_bit_identical(
        result: &FillResult,
        off: &FillOffsets,
        h_idx: usize,
        blk: usize,
        hours: f64,
        exp: &Expected,
    ) {
        let turb_col = off.at(off.turbine, h_idx, blk);
        assert_eq!(result.col_lower[turb_col].to_bits(), 0.0_f64.to_bits());
        assert_eq!(
            result.col_upper[turb_col].to_bits(),
            exp.turb_upper.to_bits()
        );
        assert_eq!(
            result.objective[turb_col].to_bits(),
            (exp.turbined_cost * hours).to_bits()
        );

        let div_col = off.at(off.diversion, h_idx, blk);
        assert_eq!(result.col_lower[div_col].to_bits(), 0.0_f64.to_bits());
        assert_eq!(result.col_upper[div_col].to_bits(), exp.max_div.to_bits());
        let expected_div_obj = if exp.max_div > 0.0 {
            exp.diversion_cost * hours
        } else {
            0.0
        };
        assert_eq!(
            result.objective[div_col].to_bits(),
            expected_div_obj.to_bits()
        );

        let family_starts = [
            off.outflow_below,
            off.outflow_above,
            off.turbine_below,
            off.generation_below,
        ];
        for (fam_idx, &start) in family_starts.iter().enumerate() {
            let col = off.at(start, h_idx, blk);
            let expected_upper = if exp.active[fam_idx] {
                f64::INFINITY
            } else {
                0.0
            };
            assert_eq!(result.col_lower[col].to_bits(), 0.0_f64.to_bits());
            assert_eq!(result.col_upper[col].to_bits(), expected_upper.to_bits());
            assert_eq!(
                result.objective[col].to_bits(),
                (exp.costs[fam_idx] * hours).to_bits()
            );
        }

        let row_min_outflow = off.at(off.min_outflow_row, h_idx, blk);
        assert_eq!(
            result.row_lower[row_min_outflow].to_bits(),
            exp.min_outflow_row_lower.to_bits()
        );
        assert_eq!(
            result.row_upper[row_min_outflow].to_bits(),
            f64::INFINITY.to_bits()
        );

        let row_max_outflow = off.at(off.max_outflow_row, h_idx, blk);
        assert_eq!(
            result.row_lower[row_max_outflow].to_bits(),
            f64::NEG_INFINITY.to_bits()
        );
        assert_eq!(
            result.row_upper[row_max_outflow].to_bits(),
            exp.max_outflow_row_upper.to_bits()
        );

        let row_min_turbine = off.at(off.min_turbine_row, h_idx, blk);
        assert_eq!(
            result.row_lower[row_min_turbine].to_bits(),
            exp.min_turbine_row_lower.to_bits()
        );
        assert_eq!(
            result.row_upper[row_min_turbine].to_bits(),
            f64::INFINITY.to_bits()
        );

        let row_min_generation = off.at(off.min_generation_row, h_idx, blk);
        assert_eq!(
            result.row_lower[row_min_generation].to_bits(),
            exp.min_generation_row_lower.to_bits()
        );
        assert_eq!(
            result.row_upper[row_min_generation].to_bits(),
            f64::INFINITY.to_bits()
        );
    }

    /// Three hydros spanning every operational-slack predicate state (mirroring
    /// `block_family_slack_tests::hydro_specs`'s coverage, including hydro 2's
    /// `Some(0.0)` `is_some()` lock), three blocks, empty overlay: every turbine,
    /// diversion, and operational-slack column, and every operational-violation row,
    /// reads bit-identical to the stage-level formula — `hydro_bounds_at_block` falls
    /// through to the stage cell at every block.
    #[test]
    fn test_hydro_block_fill_sites_bit_identical_without_overlay() {
        let hydros = vec![
            fixture_hydro(1, None),
            fixture_hydro(2, None),
            fixture_hydro(3, None),
        ];
        let mut fixtures = HydroBlockFixtures::new(hydros, &[2.0, 1.0, 1.0]);
        fixtures.set_hydro_bounds(
            0,
            STAGE_IDX,
            hydro_block_bounds(10.0, 200.0, 12.0, None, 0.0, 190.0, None),
        );
        fixtures.set_hydro_penalties(
            0,
            STAGE_IDX,
            hydro_stage_penalties(1.0, 0.0, 2.0, 3.0, 4.0, 5.0),
        );
        fixtures.set_hydro_bounds(
            1,
            STAGE_IDX,
            hydro_block_bounds(0.0, 150.0, 0.0, Some(80.0), 20.0, 140.0, Some(30.0)),
        );
        fixtures.set_hydro_penalties(
            1,
            STAGE_IDX,
            hydro_stage_penalties(6.0, 7.0, 8.0, 9.0, 10.0, 11.0),
        );
        fixtures.set_hydro_bounds(
            2,
            STAGE_IDX,
            hydro_block_bounds(0.0, 0.0, 0.0, Some(0.0), 0.0, 0.0, None),
        );
        fixtures.set_hydro_penalties(
            2,
            STAGE_IDX,
            hydro_stage_penalties(12.0, 0.0, 13.0, 14.0, 15.0, 16.0),
        );

        let result = run_fill(&fixtures, STAGE_IDX);
        let off = &result.offsets;

        let expected = [
            Expected {
                turb_upper: 95.0,
                turbined_cost: 1.0,
                max_div: 0.0,
                diversion_cost: 0.0,
                active: [true, false, true, false],
                costs: [2.0, 3.0, 4.0, 5.0],
                min_outflow_row_lower: 12.0,
                max_outflow_row_upper: f64::INFINITY,
                min_turbine_row_lower: 10.0,
                min_generation_row_lower: 0.0,
            },
            Expected {
                turb_upper: 140.0,
                turbined_cost: 6.0,
                max_div: 30.0,
                diversion_cost: 7.0,
                active: [false, true, false, true],
                costs: [8.0, 9.0, 10.0, 11.0],
                min_outflow_row_lower: 0.0,
                max_outflow_row_upper: 80.0,
                min_turbine_row_lower: 0.0,
                min_generation_row_lower: 20.0,
            },
            Expected {
                turb_upper: 0.0,
                turbined_cost: 12.0,
                max_div: 0.0,
                diversion_cost: 0.0,
                // Hydro 2 locks the `max_outflow_m3s.is_some()` semantics: `Some(0.0)`
                // still activates outflow-above (a `> 0.0` regression would not).
                active: [false, true, false, false],
                costs: [13.0, 14.0, 15.0, 16.0],
                min_outflow_row_lower: 0.0,
                max_outflow_row_upper: 0.0,
                min_turbine_row_lower: 0.0,
                min_generation_row_lower: 0.0,
            },
        ];

        for (h_idx, exp) in expected.iter().enumerate() {
            for (blk, &hours) in BLOCK_HOURS.iter().enumerate() {
                assert_block_bit_identical(&result, off, h_idx, blk, hours, exp);
            }
        }
    }

    /// A hydro with stage-wide `max_turbined_m3s = 500.0` (`max_generation_mw`
    /// large enough that the constant-productivity cap never binds) and a
    /// `block_id = 1` override to `100.0` on a three-block stage binds ONLY block 1.
    #[test]
    fn test_per_block_turbined_cap_binds_only_its_own_block() {
        let hydros = vec![fixture_hydro(1, None)];
        let mut fixtures = HydroBlockFixtures::new(hydros, &[1.0]);
        fixtures.set_hydro_bounds(
            0,
            STAGE_IDX,
            hydro_block_bounds(0.0, 500.0, 0.0, None, 0.0, 1000.0, None),
        );
        fixtures.install_block_overlay();
        fixtures.set_hydro_block_override(
            0,
            STAGE_IDX,
            1,
            HydroBlockOverride {
                max_turbined_m3s: Some(100.0),
                ..Default::default()
            },
        );

        let result = run_fill(&fixtures, STAGE_IDX);
        let off = &result.offsets;
        let upper = per_block(&result.col_upper, off, off.turbine);
        assert_eq!(
            upper,
            vec![500.0, 100.0, 500.0],
            "only block 1 is bound to the override"
        );
        for blk in 0..N_BLKS {
            assert_eq!(
                result.col_lower[off.at(off.turbine, 0, blk)],
                0.0,
                "col_lower unaffected by the max-only override, blk {blk}"
            );
        }
    }

    /// A hydro with stage-wide `max_diversion_m3s = Some(20.0)` and a `block_id = 2`
    /// override to `75.0` on a three-block stage binds ONLY block 2 — the diversion
    /// analogue of the turbine per-block cap.
    #[test]
    fn test_per_block_diversion_cap_binds_only_its_own_block() {
        let hydros = vec![fixture_hydro(1, None)];
        let mut fixtures = HydroBlockFixtures::new(hydros, &[1.0]);
        fixtures.set_hydro_bounds(
            0,
            STAGE_IDX,
            hydro_block_bounds(0.0, 0.0, 0.0, None, 0.0, 0.0, Some(20.0)),
        );
        fixtures.install_block_overlay();
        fixtures.set_hydro_block_override(
            0,
            STAGE_IDX,
            2,
            HydroBlockOverride {
                max_diversion_m3s: Some(75.0),
                ..Default::default()
            },
        );

        let result = run_fill(&fixtures, STAGE_IDX);
        let off = &result.offsets;
        let upper = per_block(&result.col_upper, off, off.diversion);
        assert_eq!(
            upper,
            vec![20.0, 20.0, 75.0],
            "only block 2 is bound to the override"
        );
    }

    /// A hydro with stage-wide `min_diversion_m3s = Some(2.0)` and a `block_id = 1`
    /// override to `7.0` on a three-block stage binds ONLY block 1 — the floor
    /// analogue of [`test_per_block_diversion_cap_binds_only_its_own_block`].
    #[test]
    fn test_per_block_diversion_floor_binds_only_its_own_block() {
        let hydros = vec![fixture_hydro(1, None)];
        let mut fixtures = HydroBlockFixtures::new(hydros, &[1.0]);
        let mut hb = hydro_block_bounds(0.0, 0.0, 0.0, None, 0.0, 0.0, Some(20.0));
        hb.min_diversion_m3s = Some(2.0);
        fixtures.set_hydro_bounds(0, STAGE_IDX, hb);
        fixtures.install_block_overlay();
        fixtures.set_hydro_block_override(
            0,
            STAGE_IDX,
            1,
            HydroBlockOverride {
                min_diversion_m3s: Some(7.0),
                ..Default::default()
            },
        );

        let result = run_fill(&fixtures, STAGE_IDX);
        let off = &result.offsets;
        let lower = per_block(&result.col_lower, off, off.diversion);
        assert_eq!(
            lower,
            vec![2.0, 7.0, 2.0],
            "only block 1 is bound to the floor override"
        );
    }

    /// A filling-suspended hydro (`filling = None`, `entry_stage_id` after the
    /// build stage — `PreFilling`) carrying a `block_id` override setting
    /// `max_turbined_m3s = 400.0` still gets `[0, 0]` at every block: the
    /// suspension gate wins over the per-block cap.
    #[test]
    fn test_suspended_hydro_ignores_per_block_turbine_cap() {
        let hydros = vec![fixture_hydro(1, Some(3))];
        let mut fixtures = HydroBlockFixtures::new(hydros, &[1.0]);
        fixtures.set_hydro_bounds(
            0,
            STAGE_IDX,
            hydro_block_bounds(0.0, 500.0, 0.0, None, 0.0, 1000.0, None),
        );
        fixtures.install_block_overlay();
        fixtures.set_hydro_block_override(
            0,
            STAGE_IDX,
            1,
            HydroBlockOverride {
                max_turbined_m3s: Some(400.0),
                ..Default::default()
            },
        );

        let result = run_fill(&fixtures, STAGE_IDX);
        let off = &result.offsets;
        for blk in 0..N_BLKS {
            let col = off.at(off.turbine, 0, blk);
            assert_eq!(result.col_upper[col], 0.0, "suspended col_upper, blk {blk}");
            assert_eq!(result.col_lower[col], 0.0, "suspended col_lower, blk {blk}");
        }
    }

    /// A hydro with stage-wide `min_outflow_m3s = 0.0` (below-min-outflow slack
    /// inactive) and a `block_id = 2` override to `200.0` activates ONLY block 2's
    /// slack column, and the min-outflow row's `row_lower` reads `[0, 0, 200]` —
    /// a stage-level activation predicate hoisted above the block loop would leave
    /// this floor unenforceable (no slack column, hard infeasibility).
    #[test]
    fn test_block_only_floor_activates_its_slack_column() {
        let hydros = vec![fixture_hydro(1, None)];
        let mut fixtures = HydroBlockFixtures::new(hydros, &[1.0]);
        fixtures.set_hydro_bounds(
            0,
            STAGE_IDX,
            hydro_block_bounds(0.0, 50.0, 0.0, None, 0.0, 45.0, None),
        );
        fixtures.install_block_overlay();
        fixtures.set_hydro_block_override(
            0,
            STAGE_IDX,
            2,
            HydroBlockOverride {
                min_outflow_m3s: Some(200.0),
                ..Default::default()
            },
        );

        let result = run_fill(&fixtures, STAGE_IDX);
        let off = &result.offsets;

        let slack_upper = per_block(&result.col_upper, off, off.outflow_below);
        assert_eq!(
            slack_upper,
            vec![0.0, 0.0, f64::INFINITY],
            "only block 2's slack column is active"
        );

        let row_lower = per_block(&result.row_lower, off, off.min_outflow_row);
        assert_eq!(
            row_lower,
            vec![0.0, 0.0, 200.0],
            "min-outflow row_lower reads per block"
        );
    }

    /// A hydro with stage-wide `max_outflow_m3s = Some(300.0)` and a `block_id = 1`
    /// override to `50.0` binds ONLY block 1's max-outflow row upper bound;
    /// `row_lower` stays `-INF` at every block (a literal constant, not read from
    /// the resolved bound).
    #[test]
    fn test_per_block_max_outflow_row_bound() {
        let hydros = vec![fixture_hydro(1, None)];
        let mut fixtures = HydroBlockFixtures::new(hydros, &[1.0]);
        fixtures.set_hydro_bounds(
            0,
            STAGE_IDX,
            hydro_block_bounds(0.0, 50.0, 0.0, Some(300.0), 0.0, 45.0, None),
        );
        fixtures.install_block_overlay();
        fixtures.set_hydro_block_override(
            0,
            STAGE_IDX,
            1,
            HydroBlockOverride {
                max_outflow_m3s: Some(50.0),
                ..Default::default()
            },
        );

        let result = run_fill(&fixtures, STAGE_IDX);
        let off = &result.offsets;
        let row_upper = per_block(&result.row_upper, off, off.max_outflow_row);
        assert_eq!(
            row_upper,
            vec![300.0, 50.0, 300.0],
            "max-outflow row_upper reads per block"
        );
        for blk in 0..N_BLKS {
            assert_eq!(
                result.row_lower[off.at(off.max_outflow_row, 0, blk)],
                f64::NEG_INFINITY,
                "max-outflow row_lower stays -INF, blk {blk}"
            );
        }
    }

    /// A hydro with a declared group `min_turbined_m3s = 0.0` and a group
    /// `block_id = 1` override to `75.0` (`hydro_unit_group_bounds.parquet`'s
    /// per-cell floor overlay — BU-F1's previously-unconsumed override column,
    /// now the min-turbine row's sole RHS source) binds ONLY block 1's
    /// min-turbine row lower bound. The stage-wide `HydroBlockBounds.min_turbined_m3s`
    /// set below stays `0.0` throughout and is NOT what the row reads.
    #[test]
    fn test_per_block_min_turbine_row_bound() {
        let hydros = vec![fixture_hydro(1, None)];
        let mut fixtures = HydroBlockFixtures::new(hydros, &[1.0]);
        fixtures.set_hydro_bounds(
            0,
            STAGE_IDX,
            hydro_block_bounds(0.0, 50.0, 0.0, None, 0.0, 45.0, None),
        );
        fixtures.install_group_overlay();
        fixtures.set_group_block_override(
            0,
            0,
            STAGE_IDX,
            1,
            HydroUnitGroupOverride {
                min_turbined_m3s: Some(75.0),
                ..Default::default()
            },
        );

        let result = run_fill(&fixtures, STAGE_IDX);
        let off = &result.offsets;
        let row_lower = per_block(&result.row_lower, off, off.min_turbine_row);
        assert_eq!(
            row_lower,
            vec![0.0, 75.0, 0.0],
            "min-turbine row_lower reads the group's own per-block override"
        );
    }

    /// A hydro with a declared group `min_generation_mw = 0.0` and a group
    /// `block_id = 0` override to `50.0` binds ONLY block 0's min-generation
    /// row lower bound — the min-generation mirror of the min-turbine test above.
    #[test]
    fn test_per_block_min_generation_row_bound() {
        let hydros = vec![fixture_hydro(1, None)];
        let mut fixtures = HydroBlockFixtures::new(hydros, &[1.0]);
        fixtures.set_hydro_bounds(
            0,
            STAGE_IDX,
            hydro_block_bounds(0.0, 50.0, 0.0, None, 0.0, 45.0, None),
        );
        fixtures.install_group_overlay();
        fixtures.set_group_block_override(
            0,
            0,
            STAGE_IDX,
            0,
            HydroUnitGroupOverride {
                min_generation_mw: Some(50.0),
                ..Default::default()
            },
        );

        let result = run_fill(&fixtures, STAGE_IDX);
        let off = &result.offsets;
        let row_lower = per_block(&result.row_lower, off, off.min_generation_row);
        assert_eq!(
            row_lower,
            vec![50.0, 0.0, 0.0],
            "min-generation row_lower reads the group's own per-block override"
        );
    }

    /// An FPHA (non-`ConstantProductivity`) hydro with a stage-wide
    /// `max_generation_mw = 300.0` and a `block_id = 1` override to `120.0` on a
    /// three-block stage binds ONLY block 1 — the FPHA-generation-cap analogue of
    /// the turbine/diversion per-block caps above. `max_generation_mw` is
    /// block-eligible (`HydroBlockOverride`); reading `hydro_bounds` once above
    /// the `for blk` loop instead of `hydro_bounds_at_block` inside it would
    /// silently apply the stage-wide cap to every block.
    #[test]
    fn test_fpha_generation_cap_binds_only_its_own_block() {
        let hydros = vec![fixture_hydro(1, None)];
        let mut fixtures = HydroBlockFixtures::new(hydros, &[0.0]);
        fixtures.production_models = ProductionModelSet::new(
            vec![vec![
                ResolvedProductionModel::Fpha {
                    planes: vec![FphaPlane {
                        intercept: 0.0,
                        gamma_v: 0.0,
                        gamma_q: 0.0,
                        gamma_s: 0.0,
                    }],
                };
                N_STAGES
            ]],
            1,
            N_STAGES,
        );
        fixtures.set_hydro_bounds(
            0,
            STAGE_IDX,
            hydro_block_bounds(0.0, 0.0, 0.0, None, 0.0, 300.0, None),
        );
        fixtures.install_block_overlay();
        fixtures.set_hydro_block_override(
            0,
            STAGE_IDX,
            1,
            HydroBlockOverride {
                max_generation_mw: Some(120.0),
                ..Default::default()
            },
        );

        let stage = three_block_stage(STAGE_IDX);
        let ctx = fixtures.make_ctx();
        let state = state_layout_for(&ctx);
        let layout = StageLayout::new(&ctx, &state, &stage, STAGE_IDX);
        assert_eq!(
            layout.fpha_hydro_indices.len(),
            1,
            "fixture hydro must classify as FPHA"
        );

        let mut col_lower = vec![0.0_f64; layout.num_cols];
        let mut col_upper = vec![f64::INFINITY; layout.num_cols];
        let mut objective = vec![0.0_f64; layout.num_cols];
        let mut bufs = ColumnBufs {
            col_lower: &mut col_lower,
            col_upper: &mut col_upper,
            objective: &mut objective,
        };
        fill_fpha_generation_columns(&ctx, STAGE_IDX, &layout, &mut bufs);

        let upper: Vec<f64> = (0..N_BLKS)
            .map(|blk| col_upper[layout.generation_col(FphaCellLocal::new(0), BlockIdx::new(blk))])
            .collect();
        assert_eq!(
            upper,
            vec![300.0, 120.0, 300.0],
            "only block 1 is bound to the override"
        );
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::float_cmp,
    clippy::similar_names
)]
mod cell_column_bound_tests {
    use std::collections::{BTreeMap, HashMap};

    use cobre_core::entities::hydro::{FillingConfig, HydroGenerationModel};
    use cobre_core::{
        BlockBoundsCountsSpec, BoundsCountsSpec, BoundsDefaults, BusStagePenalties,
        CascadeTopology, ContractBlockBounds, EntityId, Hydro, HydroBlockBounds,
        HydroBlockOverride, HydroStageBounds, HydroStagePenalties, HydroUnitGroup,
        HydroUnitGroupBoundsCountsSpec, HydroUnitGroupOverride, LineBlockBounds,
        LineStagePenalties, NcsStagePenalties, PenaltiesCountsSpec, PenaltiesDefaults,
        PumpingBlockBounds, ResolvedBlockBounds, ResolvedBounds, ResolvedGenericConstraintBounds,
        ResolvedHydroUnitGroupBounds, ResolvedLoadFactors, ResolvedNcsBounds, ResolvedNcsFactors,
        ResolvedPenalties, ThermalBlockBounds, ThermalStageBounds,
    };
    use cobre_stochastic::par::precompute::PrecomputedPar;

    use crate::hydro_models::{
        EvaporationModel, EvaporationModelSet, FphaPlane, ProductionModelSet,
        ResolvedProductionModel,
    };
    use crate::indexer::{BlockIdx, FphaCellLocal, HydroCell, HydroCellIndex, HydroSys};
    use crate::lead_time::AnticipatedResolution;
    use crate::resolved_parameters::ResolvedParameters;
    use crate::test_support::make_unit_group;

    use super::super::layout::ResolvedTables;
    use super::super::test_support::{
        BLOCK_HOURS, state_layout_for, three_block_stage, zero_hydro_penalties,
    };
    use super::{
        ColumnBufs, GroupBoundLookup, StageLayout, TemplateBuildCtx, cell_min_generation,
        cell_min_turbined, fill_fpha_generation_columns, fill_turbine_columns,
    };

    const N_STAGES: usize = 1;
    const STAGE_IDX: usize = 0;
    const N_BLKS: usize = 3;

    const START_STAGE_ID: i32 = 2;
    const ENTRY_STAGE_ID: i32 = 4;
    const PREFILLING_ID: i32 = 1;

    /// The always-`Operating`, single-mirror-group padding plant ahead of a
    /// split plant in every fixture below, so the split plant's cells land at
    /// global index >= 1 — mutually distinct from `h_idx` and `block_idx`, the
    /// index-coincidence check.
    fn padding_hydro() -> Hydro {
        let mut hydro = Hydro {
            unit_groups: Vec::new(),
            id: EntityId(1),
            name: "H1".to_string(),
            operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            downstream_id: None,
            travel_time_hours: None,
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: 0.0,
            max_storage_hm3: 100.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: HydroGenerationModel::ConstantProductivity,
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 200.0,
            specific_productivity_mw_per_m3s_per_m: None,
            min_generation_mw: 0.0,
            max_generation_mw: 1_000_000.0,
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: None,
            evaporation_reference_volumes_hm3: None,
            diversion: None,
            filling: None,
            penalties: zero_hydro_penalties(),
        };
        hydro.declare_mirror_unit_group(EntityId(40));
        hydro
    }

    /// The two-bus FPHA plant `test_cell_columns_take_their_own_group_box` and
    /// `test_filling_suspension_pins_every_cell_of_a_split_plant` share: two
    /// UNEQUAL-maxima groups on distinct buses (200 < 300, so the low-bus
    /// group's cell sorts first). `filling`/`entry` let the suspension test
    /// suspend it while the collapse test leaves it `Operating`.
    fn split_plant(filling: Option<FillingConfig>, entry: Option<i32>) -> Hydro {
        let groups = vec![
            make_unit_group(EntityId(21), EntityId(200), 0.0, 7000.0, 0.0, 5250.0),
            make_unit_group(EntityId(22), EntityId(300), 0.0, 5000.0, 0.0, 3750.0),
        ];
        let mut hydro = Hydro {
            unit_groups: groups,
            id: EntityId(2),
            name: "H2".to_string(),
            operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            downstream_id: None,
            travel_time_hours: None,
            entry_stage_id: entry,
            exit_stage_id: None,
            min_storage_hm3: 0.0,
            max_storage_hm3: 100.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: HydroGenerationModel::Fpha,
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 9000.0,
            specific_productivity_mw_per_m3s_per_m: None,
            min_generation_mw: 0.0,
            max_generation_mw: 12000.0,
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: None,
            evaporation_reference_volumes_hm3: None,
            diversion: None,
            filling,
            penalties: zero_hydro_penalties(),
        };
        hydro.declare_mirror_unit_group(EntityId(99));
        hydro
    }

    /// A `ConstantProductivity` hydro at `id`/`bus_id` with explicit `groups`
    /// (empty for the mirrored-group comparison), declared turbined/generation
    /// maxima matching its resolved bounds (set separately via
    /// `Fixtures::set_hydro_bounds`) so an empty-groups plant's mirrored group
    /// carries them exactly.
    fn grouped_hydro(
        id: i32,
        bus_id: EntityId,
        groups: Vec<HydroUnitGroup>,
        max_turbined_m3s: f64,
        max_generation_mw: f64,
    ) -> Hydro {
        let mut hydro = Hydro {
            unit_groups: groups,
            id: EntityId(id),
            name: format!("H{id}"),
            operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            downstream_id: None,
            travel_time_hours: None,
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: 0.0,
            max_storage_hm3: 100.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: HydroGenerationModel::ConstantProductivity,
            min_turbined_m3s: 0.0,
            max_turbined_m3s,
            specific_productivity_mw_per_m3s_per_m: None,
            min_generation_mw: 0.0,
            max_generation_mw,
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: None,
            evaporation_reference_volumes_hm3: None,
            diversion: None,
            filling: None,
            penalties: zero_hydro_penalties(),
        };
        hydro.declare_mirror_unit_group(bus_id);
        hydro
    }

    /// A `HydroBlockBounds` row with only turbined/generation maxima set; every
    /// other bound is a generous or zero default irrelevant to the column
    /// fills under test.
    fn hydro_block_bounds(max_turbined_m3s: f64, max_generation_mw: f64) -> HydroBlockBounds {
        HydroBlockBounds {
            max_turbined_m3s,
            max_generation_mw,
            ..Default::default()
        }
    }

    fn zero_hydro_stage_penalties() -> HydroStagePenalties {
        HydroStagePenalties {
            spillage_cost: 0.0,
            diversion_cost: 0.0,
            turbined_cost: 0.0,
            storage_violation_below_cost: 0.0,
            filling_target_violation_cost: 0.0,
            turbined_violation_below_cost: 0.0,
            outflow_violation_below_cost: 0.0,
            outflow_violation_above_cost: 0.0,
            generation_violation_below_cost: 0.0,
            evaporation_violation_cost: 0.0,
            water_withdrawal_violation_cost: 0.0,
            water_withdrawal_violation_pos_cost: 0.0,
            water_withdrawal_violation_neg_cost: 0.0,
            evaporation_violation_pos_cost: 0.0,
            evaporation_violation_neg_cost: 0.0,
            inflow_nonnegativity_cost: 0.0,
        }
    }

    fn empty_bounds(n_hydros: usize, n_stages: usize) -> ResolvedBounds {
        ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros,
                n_thermals: 0,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages,
                k_max: 0,
            },
            &BoundsDefaults {
                hydro: HydroStageBounds {
                    min_storage_hm3: 0.0,
                    max_storage_hm3: 100.0,
                    filling_min_rate_m3s: 0.0,
                    water_withdrawal_m3s: 0.0,
                },
                hydro_block: hydro_block_bounds(0.0, 0.0),
                thermal: ThermalStageBounds { cost_per_mwh: 0.0 },
                thermal_block: ThermalBlockBounds {
                    min_generation_mw: 0.0,
                    max_generation_mw: 0.0,
                },
                line_block: LineBlockBounds {
                    direct_mw: 0.0,
                    reverse_mw: 0.0,
                },
                pumping_block: PumpingBlockBounds {
                    min_flow_m3s: 0.0,
                    max_flow_m3s: 0.0,
                },
                contract_block: ContractBlockBounds {
                    min_mw: 0.0,
                    max_mw: 0.0,
                    price_per_mwh: 0.0,
                },
            },
        )
    }

    fn zero_penalties(n_hydros: usize, n_stages: usize) -> ResolvedPenalties {
        ResolvedPenalties::new(
            &PenaltiesCountsSpec {
                n_hydros,
                n_buses: 0,
                n_lines: 0,
                n_ncs: 0,
                n_stages,
            },
            &PenaltiesDefaults {
                hydro: zero_hydro_stage_penalties(),
                bus: BusStagePenalties { excess_cost: 0.0 },
                line: LineStagePenalties { exchange_cost: 0.0 },
                ncs: NcsStagePenalties {
                    curtailment_cost: 0.0,
                },
            },
        )
    }

    /// Owns the borrow targets for a hydro-only `TemplateBuildCtx`. Unlike
    /// `hydro_block_bound_tests`'s all-`ConstantProductivity` fixture, `models`
    /// is per-hydro and mixes FPHA and `ConstantProductivity` across one fixture.
    struct Fixtures {
        par_lp: PrecomputedPar,
        hydros: Vec<Hydro>,
        hydro_cell_index: HydroCellIndex,
        cascade: CascadeTopology,
        bounds: ResolvedBounds,
        penalties: ResolvedPenalties,
        production_models: ProductionModelSet,
        evaporation_models: EvaporationModelSet,
        resolved_generic_bounds: ResolvedGenericConstraintBounds,
        resolved_load_factors: ResolvedLoadFactors,
        resolved_ncs_bounds: ResolvedNcsBounds,
        resolved_ncs_factors: ResolvedNcsFactors,
        resolved_parameters: ResolvedParameters,
        /// Stage count backing `bounds`/`penalties`/`production_models` — `N_STAGES`
        /// for every single-stage fixture, wider only for the group-override tests
        /// that need a real "another stage" to assert against.
        n_stages: usize,
    }

    impl Fixtures {
        fn new(hydros: Vec<Hydro>, models: Vec<ResolvedProductionModel>, n_stages: usize) -> Self {
            let n_hydros = hydros.len();
            assert_eq!(
                models.len(),
                n_hydros,
                "one production model per fixture hydro"
            );
            let cascade = CascadeTopology::build(&hydros);
            let hydro_cell_index = HydroCellIndex::build(&hydros);
            Self {
                par_lp: PrecomputedPar::default(),
                hydros,
                hydro_cell_index,
                cascade,
                bounds: empty_bounds(n_hydros, n_stages),
                penalties: zero_penalties(n_hydros, n_stages),
                production_models: ProductionModelSet::new(
                    models.into_iter().map(|m| vec![m; n_stages]).collect(),
                    n_hydros,
                    n_stages,
                ),
                evaporation_models: EvaporationModelSet::new(vec![
                    EvaporationModel::None;
                    n_hydros
                ]),
                resolved_generic_bounds: ResolvedGenericConstraintBounds::empty(),
                resolved_load_factors: ResolvedLoadFactors::empty(),
                resolved_ncs_bounds: ResolvedNcsBounds::empty(),
                resolved_ncs_factors: ResolvedNcsFactors::empty(),
                resolved_parameters: ResolvedParameters {
                    per_param: vec![],
                    id_to_slot: vec![],
                    cost_scale_factor: 1_000_000.0,
                },
                n_stages,
            }
        }

        fn set_hydro_bounds(&mut self, h_idx: usize, stage_idx: usize, hb: HydroBlockBounds) {
            *self.bounds.hydro_block_base_mut(h_idx, stage_idx) = hb;
        }

        fn install_block_overlay(&mut self) {
            self.bounds
                .set_block_overlay(ResolvedBlockBounds::new(&BlockBoundsCountsSpec {
                    n_hydros: self.hydros.len(),
                    n_thermals: 0,
                    n_lines: 0,
                    n_pumping: 0,
                    n_contracts: 0,
                    n_stages: self.n_stages,
                    max_blocks: N_BLKS,
                }));
        }

        fn set_hydro_block_override(
            &mut self,
            h_idx: usize,
            block_idx: usize,
            over: HydroBlockOverride,
        ) {
            *self
                .bounds
                .block_overlay_mut()
                .hydro_override_mut(h_idx, STAGE_IDX, block_idx)
                .expect("overlay cell must exist for a fixture-sized overlay") = over;
        }

        /// Install the group-bounds overlay, sized from each hydro's own
        /// declared `unit_groups` count (ragged per plant, matching
        /// [`ResolvedHydroUnitGroupBounds`]'s CSR group axis).
        fn install_group_overlay(&mut self) {
            let groups_per_plant: Vec<usize> =
                self.hydros.iter().map(|h| h.unit_groups.len()).collect();
            self.bounds
                .set_group_overlay(ResolvedHydroUnitGroupBounds::new(
                    &HydroUnitGroupBoundsCountsSpec {
                        groups_per_plant: &groups_per_plant,
                        n_stages: self.n_stages,
                        max_blocks: N_BLKS,
                    },
                ));
        }

        fn set_group_block_override(
            &mut self,
            h_idx: usize,
            group_pos: usize,
            stage_idx: usize,
            block_idx: usize,
            over: HydroUnitGroupOverride,
        ) {
            *self
                .bounds
                .group_overlay_mut()
                .block_override_mut(h_idx, group_pos, stage_idx, block_idx)
                .expect("overlay cell must exist for a fixture-sized overlay") = over;
        }

        fn make_ctx(&self) -> TemplateBuildCtx<'_> {
            let mut hydro_pos = BTreeMap::new();
            for (i, h) in self.hydros.iter().enumerate() {
                hydro_pos.insert(h.id, i);
            }
            TemplateBuildCtx {
                hydros: &self.hydros,
                thermals: &[],
                lines: &[],
                buses: &[],
                load_models: &[],
                cascade: &self.cascade,
                hydro_cell_index: &self.hydro_cell_index,
                resolved: ResolvedTables {
                    bounds: &self.bounds,
                    penalties: &self.penalties,
                    resolved_generic_bounds: &self.resolved_generic_bounds,
                    resolved_load_factors: &self.resolved_load_factors,
                    resolved_ncs_bounds: &self.resolved_ncs_bounds,
                    resolved_ncs_factors: &self.resolved_ncs_factors,
                    resolved_parameters: &self.resolved_parameters,
                },
                hydro_pos,
                thermal_pos: BTreeMap::new(),
                line_pos: BTreeMap::new(),
                bus_pos: BTreeMap::new(),
                par_lp: &self.par_lp,
                production_models: &self.production_models,
                evaporation_models: &self.evaporation_models,
                generic_constraints: &[],
                non_controllable_sources: &[],
                pumping_stations: &[],
                pumping_pos: BTreeMap::new(),
                n_pumping: 0,
                contracts: &[],
                contract_pos: BTreeMap::new(),
                n_contract_import: 0,
                n_contract_export: 0,
                diversion_upstream: HashMap::new(),
                arc_stage_weights: HashMap::new(),
                arc_spread_chrono: HashMap::new(),
                arc_arrival_density: HashMap::new(),
                per_stage_mask: Vec::new(),
                post_study_resolved: crate::setup::PostStudyResolved::default(),
                n_hydros: self.hydros.len(),
                n_thermals: 0,
                n_lines: 0,
                n_buses: 0,
                max_par_order: 0,
                n_anticipated: 0,
                k_max: 0,
                anticipated_lead_stages: vec![],
                anticipated_thermal_indices: vec![],
                anticipated_windows: vec![],
                anticipated_resolution: AnticipatedResolution::default(),
                study_stage_ids: (0..self.n_stages as i32).collect(),
                delivery_stage_ids: (0..self.n_stages as i32).collect(),
                has_penalty: false,
                delivery_cumulative_discount_factors: vec![1.0; self.n_stages],
                delivery_total_hours: vec![BLOCK_HOURS.iter().sum(); self.n_stages],
                filling_v_target: BTreeMap::new(),
            }
        }
    }

    /// Fresh `[0, +INF]`-initialised column buffers sized to `num_cols`.
    fn fresh_bufs(num_cols: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        (
            vec![0.0_f64; num_cols],
            vec![f64::INFINITY; num_cols],
            vec![0.0_f64; num_cols],
        )
    }

    /// Two unequal groups on two buses (bus 200: 7000 MW / 5250 m³/s; bus 300:
    /// 5000 MW / 3750 m³/s) against a plant resolved at 12000 MW / 9000 m³/s —
    /// neither group equals the other or the plant's own box, so an
    /// implementation that splits evenly or reads the plant's box for every
    /// cell is caught. A padding plant ahead of the split one puts its cells at
    /// global index 1 and 2, so cell 2's asserted column has `h_idx = 1`,
    /// `cell_idx = 2`, `block_idx = 0` mutually distinct. Block 2 carries a
    /// per-block override lowering the plant's resolved bound below cell 1's
    /// own sum, so the `min(..., hb.max_*)` cap actually binds there while
    /// staying slack for cell 2 — invisible at blocks 0/1, where neither
    /// cell's sum exceeds the plant's declared value.
    #[test]
    fn test_cell_columns_take_their_own_group_box() {
        let hydros = vec![padding_hydro(), split_plant(None, None)];
        let models = vec![
            ResolvedProductionModel::ConstantProductivity { productivity: 1.0 },
            ResolvedProductionModel::Fpha {
                planes: vec![FphaPlane {
                    intercept: 0.0,
                    gamma_v: 0.0,
                    gamma_q: 0.0,
                    gamma_s: 0.0,
                }],
            },
        ];
        let mut fixtures = Fixtures::new(hydros, models, N_STAGES);
        fixtures.set_hydro_bounds(0, STAGE_IDX, hydro_block_bounds(200.0, 1_000_000.0));
        fixtures.set_hydro_bounds(1, STAGE_IDX, hydro_block_bounds(9000.0, 12000.0));
        fixtures.install_block_overlay();
        fixtures.set_hydro_block_override(
            1,
            2,
            HydroBlockOverride {
                max_turbined_m3s: Some(4000.0),
                max_generation_mw: Some(6000.0),
                ..HydroBlockOverride::default()
            },
        );

        let ctx = fixtures.make_ctx();
        assert_eq!(
            ctx.hydro_cell_index.n_cells(),
            3,
            "padding plant (1 cell) + two-bus split plant (2 cells)"
        );
        assert_eq!(
            ctx.hydro_cell_index.cells_of(HydroSys::new(1)),
            1..3,
            "the split plant's cells must land at global index 1 and 2"
        );

        let stage = three_block_stage(STAGE_IDX);
        let state = state_layout_for(&ctx);
        let layout = StageLayout::new(&ctx, &state, &stage, STAGE_IDX);
        let (mut col_lower, mut col_upper, mut objective) = fresh_bufs(layout.num_cols);
        let mut bufs = ColumnBufs {
            col_lower: &mut col_lower,
            col_upper: &mut col_upper,
            objective: &mut objective,
        };
        fill_turbine_columns(&ctx, &stage, STAGE_IDX, &layout, &mut bufs);
        fill_fpha_generation_columns(&ctx, STAGE_IDX, &layout, &mut bufs);

        let cell_low = HydroCell::new(1);
        let cell_high = HydroCell::new(2);

        let pad_col = layout.turbine_col(HydroCell::new(0), BlockIdx::new(0));
        assert_eq!(
            col_upper[pad_col].to_bits(),
            200.0_f64.to_bits(),
            "the padding plant's single mirrored group must carry the plant's own box"
        );

        let turb_low_0 = col_upper[layout.turbine_col(cell_low, BlockIdx::new(0))];
        let turb_high_0 = col_upper[layout.turbine_col(cell_high, BlockIdx::new(0))];
        assert_eq!(turb_low_0.to_bits(), 5250.0_f64.to_bits());
        assert_eq!(turb_high_0.to_bits(), 3750.0_f64.to_bits());
        assert_eq!(
            col_lower[layout.turbine_col(cell_low, BlockIdx::new(0))],
            0.0
        );
        assert_eq!(
            col_lower[layout.turbine_col(cell_high, BlockIdx::new(0))],
            0.0
        );

        let gen_low_0 = col_upper[layout.generation_col(FphaCellLocal::new(0), BlockIdx::new(0))];
        let gen_high_0 = col_upper[layout.generation_col(FphaCellLocal::new(1), BlockIdx::new(0))];
        assert_eq!(gen_low_0.to_bits(), 7000.0_f64.to_bits());
        assert_eq!(gen_high_0.to_bits(), 5000.0_f64.to_bits());
        assert_eq!(
            col_lower[layout.generation_col(FphaCellLocal::new(0), BlockIdx::new(0))],
            0.0
        );
        assert_eq!(
            col_lower[layout.generation_col(FphaCellLocal::new(1), BlockIdx::new(0))],
            0.0
        );

        let turb_low_2 = col_upper[layout.turbine_col(cell_low, BlockIdx::new(2))];
        let turb_high_2 = col_upper[layout.turbine_col(cell_high, BlockIdx::new(2))];
        assert_eq!(
            turb_low_2.to_bits(),
            4000.0_f64.to_bits(),
            "the block-2 override must cap cell 1's turbined column below its own group sum"
        );
        assert_eq!(
            turb_high_2.to_bits(),
            3750.0_f64.to_bits(),
            "the block-2 override must stay slack for cell 2's own (lower) group sum"
        );

        let gen_low_2 = col_upper[layout.generation_col(FphaCellLocal::new(0), BlockIdx::new(2))];
        let gen_high_2 = col_upper[layout.generation_col(FphaCellLocal::new(1), BlockIdx::new(2))];
        assert_eq!(
            gen_low_2.to_bits(),
            6000.0_f64.to_bits(),
            "the block-2 override must cap cell 1's generation column below its own group sum"
        );
        assert_eq!(
            gen_high_2.to_bits(),
            5000.0_f64.to_bits(),
            "the block-2 override must stay slack for cell 2's own (lower) group sum"
        );
    }

    /// Three unequal same-bus groups (100/250/400) collapse into one cell whose
    /// box is their sum, byte-identical to the same plant declaring one group
    /// that mirrors it. A second same-bus pair binding
    /// on OPPOSITE sides (MW-bound then flow-bound) pins fold-then-sum against
    /// sum-then-fold: whenever every group's flow side binds, as in the first
    /// pair, the two orders coincide, so this divergent-binding pair is the
    /// only fixture that can tell them apart.
    ///
    /// Plant C's resolved bound is set to EXACTLY its declared value (rule-41
    /// equality, no override) — the raw group sum (100+10=110, 50+100=150)
    /// matches it bit-exactly, yet the folded group side (60) still binds
    /// under it. This is the sharper counterexample: a same-bus multi-group
    /// plant has one cell, but "one cell" is not "one group" — the group term
    /// of the outer `min` is load-bearing even here, where declared and
    /// resolved coincide and there is nothing for an override to lower.
    #[test]
    fn test_same_bus_groups_sum_into_one_cell_box() {
        let same_bus = EntityId(500);
        let unequal_thirds = vec![
            make_unit_group(EntityId(11), same_bus, 0.0, 1_000_000.0, 0.0, 100.0),
            make_unit_group(EntityId(12), same_bus, 0.0, 1_000_000.0, 0.0, 250.0),
            make_unit_group(EntityId(13), same_bus, 0.0, 1_000_000.0, 0.0, 400.0),
        ];
        let plant_a = grouped_hydro(1, EntityId(70), unequal_thirds, 750.0, 1_000_000.0);
        let plant_b = grouped_hydro(2, EntityId(71), Vec::new(), 750.0, 1_000_000.0);

        let opposite_sides = vec![
            make_unit_group(EntityId(31), EntityId(600), 0.0, 50.0, 0.0, 100.0),
            make_unit_group(EntityId(32), EntityId(600), 0.0, 100.0, 0.0, 10.0),
        ];
        let plant_c = grouped_hydro(3, EntityId(72), opposite_sides, 110.0, 150.0);

        let hydros = vec![plant_a, plant_b, plant_c];
        let models = vec![
            ResolvedProductionModel::ConstantProductivity { productivity: 1.0 },
            ResolvedProductionModel::ConstantProductivity { productivity: 1.0 },
            ResolvedProductionModel::ConstantProductivity { productivity: 1.0 },
        ];
        let mut fixtures = Fixtures::new(hydros, models, N_STAGES);
        fixtures.set_hydro_bounds(0, STAGE_IDX, hydro_block_bounds(750.0, 1_000_000.0));
        fixtures.set_hydro_bounds(1, STAGE_IDX, hydro_block_bounds(750.0, 1_000_000.0));
        // Exactly Plant C's declared (110.0, 150.0): rule 41 and the no-raising
        // rule both hold at equality, so this is fully valid, un-overridden input.
        fixtures.set_hydro_bounds(2, STAGE_IDX, hydro_block_bounds(110.0, 150.0));

        let ctx = fixtures.make_ctx();
        assert_eq!(
            ctx.hydro_cell_index.n_cells(),
            3,
            "each plant's same-bus groups must collapse to exactly one cell"
        );
        let cell_a = HydroCell::new(0);
        assert_eq!(
            ctx.hydro_cell_index.groups_of(cell_a).to_vec(),
            vec![0, 1, 2],
            "plant A's three same-bus groups must all land in cell 0, in declaration order"
        );

        let stage = three_block_stage(STAGE_IDX);
        let state = state_layout_for(&ctx);
        let layout = StageLayout::new(&ctx, &state, &stage, STAGE_IDX);
        let (mut col_lower, mut col_upper, mut objective) = fresh_bufs(layout.num_cols);
        let mut bufs = ColumnBufs {
            col_lower: &mut col_lower,
            col_upper: &mut col_upper,
            objective: &mut objective,
        };
        fill_turbine_columns(&ctx, &stage, STAGE_IDX, &layout, &mut bufs);

        let turb_a = col_upper[layout.turbine_col(cell_a, BlockIdx::new(0))];
        let turb_b = col_upper[layout.turbine_col(HydroCell::new(1), BlockIdx::new(0))];
        let turb_c = col_upper[layout.turbine_col(HydroCell::new(2), BlockIdx::new(0))];
        assert_eq!(
            turb_a.to_bits(),
            750.0_f64.to_bits(),
            "unequal thirds must sum to the plant's own box: 100 + 250 + 400 = 750"
        );
        assert_eq!(
            turb_b.to_bits(),
            turb_a.to_bits(),
            "declaring the groups vs. declaring none must be byte-identical"
        );
        assert_eq!(
            turb_c.to_bits(),
            60.0_f64.to_bits(),
            "fold-then-sum: min(100,50) + min(10,100) = 50 + 10 = 60, not \
             sum-then-fold's min(100+10, 50+100) = 110, and not the plant's \
             resolved 110 that a group-term-dropped cap would return"
        );
        assert_eq!(col_lower[layout.turbine_col(cell_a, BlockIdx::new(0))], 0.0);
    }

    /// A `PreFilling` split plant pins BOTH its cells' turbined columns to
    /// `[0, 0]` — the plant-scoped `suspended` gate applies once, to every
    /// cell, never re-evaluated per cell. The non-filling padding plant stays
    /// unaffected, pinning that the gate is scoped to the right plant.
    #[test]
    fn test_filling_suspension_pins_every_cell_of_a_split_plant() {
        let filling = FillingConfig {
            start_stage_id: START_STAGE_ID,
            filling_min_rate_m3s: 0.0,
        };
        let hydros = vec![
            padding_hydro(),
            split_plant(Some(filling), Some(ENTRY_STAGE_ID)),
        ];
        let models = vec![
            ResolvedProductionModel::ConstantProductivity { productivity: 1.0 },
            ResolvedProductionModel::Fpha {
                planes: vec![FphaPlane {
                    intercept: 0.0,
                    gamma_v: 0.0,
                    gamma_q: 0.0,
                    gamma_s: 0.0,
                }],
            },
        ];
        let mut fixtures = Fixtures::new(hydros, models, N_STAGES);
        fixtures.set_hydro_bounds(0, STAGE_IDX, hydro_block_bounds(200.0, 1_000_000.0));
        fixtures.set_hydro_bounds(1, STAGE_IDX, hydro_block_bounds(9000.0, 12000.0));

        let ctx = fixtures.make_ctx();
        // `stage.id = PREFILLING_ID` (< start_stage_id) resolves the split
        // plant's phase to PreFilling; STAGE_IDX (0) stays the resolved-bounds
        // row, decoupled from stage.id exactly as the filling-phase gating
        // tests decouple them.
        let stage = three_block_stage(usize::try_from(PREFILLING_ID).expect("non-negative"));
        let state = state_layout_for(&ctx);
        let layout = StageLayout::new(&ctx, &state, &stage, STAGE_IDX);
        assert!(
            layout.fpha_hydro_indices.is_empty(),
            "the suspended FPHA plant must be excluded from fpha_hydro_indices during PreFilling"
        );

        let (mut col_lower, mut col_upper, mut objective) = fresh_bufs(layout.num_cols);
        let mut bufs = ColumnBufs {
            col_lower: &mut col_lower,
            col_upper: &mut col_upper,
            objective: &mut objective,
        };
        fill_turbine_columns(&ctx, &stage, STAGE_IDX, &layout, &mut bufs);

        for cell_idx in [1, 2] {
            let cell = HydroCell::new(cell_idx);
            for blk in 0..N_BLKS {
                let col = layout.turbine_col(cell, BlockIdx::new(blk));
                assert_eq!(
                    col_upper[col], 0.0,
                    "cell {cell_idx} block {blk} must be pinned [0,0] while its plant is suspended"
                );
                assert_eq!(
                    col_lower[col], 0.0,
                    "cell {cell_idx} block {blk} col_lower must stay 0.0"
                );
            }
        }

        let pad_col = layout.turbine_col(HydroCell::new(0), BlockIdx::new(0));
        assert_eq!(
            col_upper[pad_col].to_bits(),
            200.0_f64.to_bits(),
            "the non-filling padding plant must stay unaffected: suspension is per-plant"
        );
    }

    /// With no `hydro_unit_group_bounds` rows, the overlay stays
    /// [`ResolvedHydroUnitGroupBounds::empty`] and every `GroupBoundLookup` read
    /// is `None.unwrap_or(declared)` — the same `f64` the non-overlay path
    /// reads directly off `HydroUnitGroup`. `mw_bind` binds on its MW cap
    /// (q̄=100, p̄=50, ρ=1 → fold 50.0), `flow_bind` binds on its flow cap
    /// (q̄=10, p̄=100, ρ=1 → fold 10.0); the padding/split pair reruns
    /// `test_cell_columns_take_their_own_group_box`'s FPHA numbers to cover
    /// `cell_max_generation`'s neutrality too.
    #[test]
    fn test_cell_bound_is_byte_neutral_without_group_rows() {
        let mw_bind = grouped_hydro(5, EntityId(500), Vec::new(), 100.0, 50.0);
        let flow_bind = grouped_hydro(6, EntityId(600), Vec::new(), 10.0, 100.0);
        let hydros = vec![mw_bind, flow_bind, padding_hydro(), split_plant(None, None)];
        let models = vec![
            ResolvedProductionModel::ConstantProductivity { productivity: 1.0 },
            ResolvedProductionModel::ConstantProductivity { productivity: 1.0 },
            ResolvedProductionModel::ConstantProductivity { productivity: 1.0 },
            ResolvedProductionModel::Fpha {
                planes: vec![FphaPlane {
                    intercept: 0.0,
                    gamma_v: 0.0,
                    gamma_q: 0.0,
                    gamma_s: 0.0,
                }],
            },
        ];
        let mut fixtures = Fixtures::new(hydros, models, N_STAGES);
        fixtures.set_hydro_bounds(0, STAGE_IDX, hydro_block_bounds(1_000_000.0, 1_000_000.0));
        fixtures.set_hydro_bounds(1, STAGE_IDX, hydro_block_bounds(1_000_000.0, 1_000_000.0));
        fixtures.set_hydro_bounds(2, STAGE_IDX, hydro_block_bounds(200.0, 1_000_000.0));
        fixtures.set_hydro_bounds(3, STAGE_IDX, hydro_block_bounds(9000.0, 12000.0));

        let ctx = fixtures.make_ctx();
        assert_eq!(
            ctx.hydro_cell_index.n_cells(),
            5,
            "mw_bind + flow_bind + padding (1 cell each) + split plant (2 cells)"
        );

        let stage = three_block_stage(STAGE_IDX);
        let state = state_layout_for(&ctx);
        let layout = StageLayout::new(&ctx, &state, &stage, STAGE_IDX);
        let (mut col_lower, mut col_upper, mut objective) = fresh_bufs(layout.num_cols);
        let mut bufs = ColumnBufs {
            col_lower: &mut col_lower,
            col_upper: &mut col_upper,
            objective: &mut objective,
        };
        fill_turbine_columns(&ctx, &stage, STAGE_IDX, &layout, &mut bufs);
        fill_fpha_generation_columns(&ctx, STAGE_IDX, &layout, &mut bufs);

        let turb_mw = col_upper[layout.turbine_col(HydroCell::new(0), BlockIdx::new(0))];
        assert_eq!(
            turb_mw.to_bits(),
            50.0_f64.to_bits(),
            "MW-binding plant: min(100, 50/1.0) = 50.0"
        );
        assert_eq!(
            col_lower[layout.turbine_col(HydroCell::new(0), BlockIdx::new(0))],
            0.0
        );

        let turb_flow = col_upper[layout.turbine_col(HydroCell::new(1), BlockIdx::new(0))];
        assert_eq!(
            turb_flow.to_bits(),
            10.0_f64.to_bits(),
            "flow-binding plant: min(10, 100/1.0) = 10.0"
        );
        assert_eq!(
            col_lower[layout.turbine_col(HydroCell::new(1), BlockIdx::new(0))],
            0.0
        );

        let turb_pad = col_upper[layout.turbine_col(HydroCell::new(2), BlockIdx::new(0))];
        assert_eq!(turb_pad.to_bits(), 200.0_f64.to_bits());

        let turb_split_low = col_upper[layout.turbine_col(HydroCell::new(3), BlockIdx::new(0))];
        let turb_split_high = col_upper[layout.turbine_col(HydroCell::new(4), BlockIdx::new(0))];
        assert_eq!(turb_split_low.to_bits(), 5250.0_f64.to_bits());
        assert_eq!(turb_split_high.to_bits(), 3750.0_f64.to_bits());

        let gen_split_low =
            col_upper[layout.generation_col(FphaCellLocal::new(0), BlockIdx::new(0))];
        let gen_split_high =
            col_upper[layout.generation_col(FphaCellLocal::new(1), BlockIdx::new(0))];
        assert_eq!(gen_split_low.to_bits(), 7000.0_f64.to_bits());
        assert_eq!(gen_split_high.to_bits(), 5000.0_f64.to_bits());
    }

    /// A four-plant fixture (three single-group fillers, then a two-bus main
    /// plant) makes `hydro_idx` (3), `cell_idx` (4), `group_pos` (0), `group_id`
    /// (77), `bus_idx` (900), `stage_idx` (2), and `block_idx` (1) mutually
    /// distinct on the overridden entry — an index coincidence has masked a
    /// swap on this axis before. Only group 0 (bus 900) is overridden, only at
    /// `(stage 2, block 1)`; the sibling group (bus 500), the same cell at
    /// block 0, and the same cell at stage 0 must all keep their declared
    /// value (100.0).
    #[test]
    fn test_cell_bound_takes_the_resolved_group_override() {
        const MULTI_STAGES: usize = 3;
        let filler = |id: i32, bus: i32| grouped_hydro(id, EntityId(bus), Vec::new(), 500.0, 500.0);
        let group_high_bus =
            make_unit_group(EntityId(77), EntityId(900), 0.0, 1_000_000.0, 0.0, 100.0);
        let group_low_bus =
            make_unit_group(EntityId(78), EntityId(500), 0.0, 1_000_000.0, 0.0, 60.0);
        let main_plant = grouped_hydro(
            90,
            EntityId(999),
            vec![group_high_bus, group_low_bus],
            160.0,
            2_000_000.0,
        );
        let hydros = vec![filler(10, 10), filler(11, 11), filler(12, 12), main_plant];
        let models = vec![ResolvedProductionModel::ConstantProductivity { productivity: 1.0 }; 4];
        let mut fixtures = Fixtures::new(hydros, models, MULTI_STAGES);
        fixtures.set_hydro_bounds(3, 0, hydro_block_bounds(1_000_000.0, 1_000_000.0));
        fixtures.set_hydro_bounds(3, 2, hydro_block_bounds(1_000_000.0, 1_000_000.0));
        fixtures.install_group_overlay();
        fixtures.set_group_block_override(
            3,
            0,
            2,
            1,
            HydroUnitGroupOverride {
                max_turbined_m3s: Some(30.0),
                ..HydroUnitGroupOverride::default()
            },
        );

        let ctx = fixtures.make_ctx();
        assert_eq!(ctx.hydro_cell_index.n_cells(), 5);
        assert_eq!(
            ctx.hydro_cell_index.cells_of(HydroSys::new(3)),
            3..5,
            "the main plant's two cells must land at global index 3 and 4"
        );
        let cell_sibling = HydroCell::new(3);
        let cell_overridden = HydroCell::new(4);
        assert_eq!(ctx.hydro_cell_index.bus_of(cell_sibling), EntityId(500));
        assert_eq!(ctx.hydro_cell_index.bus_of(cell_overridden), EntityId(900));
        assert_eq!(
            ctx.hydro_cell_index.groups_of(cell_sibling).to_vec(),
            vec![1]
        );
        assert_eq!(
            ctx.hydro_cell_index.groups_of(cell_overridden).to_vec(),
            vec![0]
        );

        let state = state_layout_for(&ctx);

        let stage2 = three_block_stage(2);
        let layout2 = StageLayout::new(&ctx, &state, &stage2, 2);
        let (mut col_lower2, mut col_upper2, mut objective2) = fresh_bufs(layout2.num_cols);
        let mut bufs2 = ColumnBufs {
            col_lower: &mut col_lower2,
            col_upper: &mut col_upper2,
            objective: &mut objective2,
        };
        fill_turbine_columns(&ctx, &stage2, 2, &layout2, &mut bufs2);

        let overridden_block1 = col_upper2[layout2.turbine_col(cell_overridden, BlockIdx::new(1))];
        assert_eq!(
            overridden_block1.to_bits(),
            30.0_f64.to_bits(),
            "(stage 2, block 1) must take the override"
        );
        let overridden_block0 = col_upper2[layout2.turbine_col(cell_overridden, BlockIdx::new(0))];
        assert_eq!(
            overridden_block0.to_bits(),
            100.0_f64.to_bits(),
            "the same cell at block 0 must keep the declared value"
        );
        let sibling_block1 = col_upper2[layout2.turbine_col(cell_sibling, BlockIdx::new(1))];
        assert_eq!(
            sibling_block1.to_bits(),
            60.0_f64.to_bits(),
            "the sibling cell at (stage 2, block 1) must keep its declared value"
        );

        let stage0 = three_block_stage(0);
        let layout0 = StageLayout::new(&ctx, &state, &stage0, 0);
        let (mut col_lower0, mut col_upper0, mut objective0) = fresh_bufs(layout0.num_cols);
        let mut bufs0 = ColumnBufs {
            col_lower: &mut col_lower0,
            col_upper: &mut col_upper0,
            objective: &mut objective0,
        };
        fill_turbine_columns(&ctx, &stage0, 0, &layout0, &mut bufs0);
        let other_stage_block1 = col_upper0[layout0.turbine_col(cell_overridden, BlockIdx::new(1))];
        assert_eq!(
            other_stage_block1.to_bits(),
            100.0_f64.to_bits(),
            "the same cell at another stage must keep the declared value even at block 1"
        );
    }

    /// Mirrors `test_cell_bound_takes_the_resolved_group_override` for the FPHA
    /// generation column family: that test only calls `fill_turbine_columns`,
    /// so it cannot exercise `cell_max_generation`'s override read. Three
    /// `ConstantProductivity` fillers keep the main plant the sole FPHA hydro
    /// (`layout.fpha_hydro_indices == [HydroSys::new(3)]`), reusing the same
    /// mutually-distinct-index fixture shape: `hydro_idx` (3), `cell_idx` (4),
    /// `group_pos` (0), `group_id` (77), `bus_idx` (900), `stage_idx` (2), and
    /// `block_idx` (1) are mutually distinct on the overridden entry. Only
    /// group 0 (bus 900) is overridden, only at `(stage 2, block 1)`; the
    /// sibling group (bus 500), the same cell at block 0, and the same cell at
    /// stage 0 must all keep their declared value (100.0).
    #[test]
    fn test_generation_cell_bound_takes_the_resolved_group_override() {
        const MULTI_STAGES: usize = 3;
        let filler = |id: i32, bus: i32| grouped_hydro(id, EntityId(bus), Vec::new(), 500.0, 500.0);
        let group_high_bus =
            make_unit_group(EntityId(77), EntityId(900), 0.0, 100.0, 0.0, 1_000_000.0);
        let group_low_bus =
            make_unit_group(EntityId(78), EntityId(500), 0.0, 60.0, 0.0, 1_000_000.0);
        let mut main_plant = grouped_hydro(
            90,
            EntityId(999),
            vec![group_high_bus, group_low_bus],
            160.0,
            2_000_000.0,
        );
        main_plant.generation_model = HydroGenerationModel::Fpha;
        let hydros = vec![filler(10, 10), filler(11, 11), filler(12, 12), main_plant];
        let models = vec![
            ResolvedProductionModel::ConstantProductivity { productivity: 1.0 },
            ResolvedProductionModel::ConstantProductivity { productivity: 1.0 },
            ResolvedProductionModel::ConstantProductivity { productivity: 1.0 },
            ResolvedProductionModel::Fpha {
                planes: vec![FphaPlane {
                    intercept: 0.0,
                    gamma_v: 0.0,
                    gamma_q: 0.0,
                    gamma_s: 0.0,
                }],
            },
        ];
        let mut fixtures = Fixtures::new(hydros, models, MULTI_STAGES);
        fixtures.set_hydro_bounds(3, 0, hydro_block_bounds(1_000_000.0, 1_000_000.0));
        fixtures.set_hydro_bounds(3, 2, hydro_block_bounds(1_000_000.0, 1_000_000.0));
        fixtures.install_group_overlay();
        fixtures.set_group_block_override(
            3,
            0,
            2,
            1,
            HydroUnitGroupOverride {
                max_generation_mw: Some(30.0),
                ..HydroUnitGroupOverride::default()
            },
        );

        let ctx = fixtures.make_ctx();
        assert_eq!(ctx.hydro_cell_index.n_cells(), 5);
        assert_eq!(
            ctx.hydro_cell_index.cells_of(HydroSys::new(3)),
            3..5,
            "the main plant's two cells must land at global index 3 and 4"
        );
        let cell_sibling = HydroCell::new(3);
        let cell_overridden = HydroCell::new(4);
        assert_eq!(ctx.hydro_cell_index.bus_of(cell_sibling), EntityId(500));
        assert_eq!(ctx.hydro_cell_index.bus_of(cell_overridden), EntityId(900));

        let state = state_layout_for(&ctx);

        let stage2 = three_block_stage(2);
        let layout2 = StageLayout::new(&ctx, &state, &stage2, 2);
        assert_eq!(
            layout2.fpha_hydro_indices,
            vec![HydroSys::new(3)],
            "only the main plant is FPHA; the three fillers stay ConstantProductivity"
        );
        let (mut col_lower2, mut col_upper2, mut objective2) = fresh_bufs(layout2.num_cols);
        let mut bufs2 = ColumnBufs {
            col_lower: &mut col_lower2,
            col_upper: &mut col_upper2,
            objective: &mut objective2,
        };
        fill_fpha_generation_columns(&ctx, 2, &layout2, &mut bufs2);

        let overridden_block1 =
            col_upper2[layout2.generation_col(FphaCellLocal::new(1), BlockIdx::new(1))];
        assert_eq!(
            overridden_block1.to_bits(),
            30.0_f64.to_bits(),
            "(stage 2, block 1) must take the override"
        );
        let overridden_block0 =
            col_upper2[layout2.generation_col(FphaCellLocal::new(1), BlockIdx::new(0))];
        assert_eq!(
            overridden_block0.to_bits(),
            100.0_f64.to_bits(),
            "the same cell at block 0 must keep the declared value"
        );
        let sibling_block1 =
            col_upper2[layout2.generation_col(FphaCellLocal::new(0), BlockIdx::new(1))];
        assert_eq!(
            sibling_block1.to_bits(),
            60.0_f64.to_bits(),
            "the sibling cell at (stage 2, block 1) must keep its declared value"
        );

        let stage0 = three_block_stage(0);
        let layout0 = StageLayout::new(&ctx, &state, &stage0, 0);
        let (mut col_lower0, mut col_upper0, mut objective0) = fresh_bufs(layout0.num_cols);
        let mut bufs0 = ColumnBufs {
            col_lower: &mut col_lower0,
            col_upper: &mut col_upper0,
            objective: &mut objective0,
        };
        fill_fpha_generation_columns(&ctx, 0, &layout0, &mut bufs0);
        let other_stage_block1 =
            col_upper0[layout0.generation_col(FphaCellLocal::new(1), BlockIdx::new(1))];
        assert_eq!(
            other_stage_block1.to_bits(),
            100.0_f64.to_bits(),
            "the same cell at another stage must keep the declared value even at block 1"
        );
    }

    /// `cell_min_turbined`/`cell_min_generation` are declaration-order
    /// invariant: a plant declaring the SAME three groups (one on bus 5, two on
    /// bus 6) in a different `Vec` order produces the IDENTICAL per-cell floor
    /// sum, bit-for-bit — the min-floor mirror of `HydroCellIndex`'s own
    /// declaration-order invariance
    /// (`test_cell_index_is_invariant_to_group_declaration_order_and_ids` in
    /// `lp/indexer/hydro_cell.rs`).
    #[test]
    fn test_cell_min_floor_is_invariant_to_group_declaration_order() {
        let group_a = make_unit_group(EntityId(10), EntityId(5), 8.0, 500.0, 5.0, 500.0);
        let group_b = make_unit_group(EntityId(11), EntityId(6), 4.0, 500.0, 3.0, 500.0);
        let group_c = make_unit_group(EntityId(12), EntityId(6), 5.0, 500.0, 4.0, 500.0);

        let forward = grouped_hydro(
            1,
            EntityId(999),
            vec![group_a.clone(), group_b.clone(), group_c.clone()],
            1000.0,
            1000.0,
        );
        let reversed = grouped_hydro(
            1,
            EntityId(999),
            vec![group_c, group_b, group_a],
            1000.0,
            1000.0,
        );

        let index_forward = HydroCellIndex::build(std::slice::from_ref(&forward));
        let index_reversed = HydroCellIndex::build(std::slice::from_ref(&reversed));
        assert_eq!(index_forward.n_cells(), index_reversed.n_cells());
        assert_eq!(
            index_forward.n_cells(),
            2,
            "bus 5 and bus 6 must partition into 2 cells"
        );

        let empty = ResolvedHydroUnitGroupBounds::empty();
        let lookup = GroupBoundLookup::new(&empty, 0, 0, 0);

        for c in 0..index_forward.n_cells() {
            let cell_f = HydroCell::new(c);
            let bus = index_forward.bus_of(cell_f);
            let cell_r = (0..index_reversed.n_cells())
                .map(HydroCell::new)
                .find(|&cr| index_reversed.bus_of(cr) == bus)
                .expect("both declaration orders must partition onto the same buses");

            let min_turb_f = cell_min_turbined(
                &forward.unit_groups,
                index_forward.groups_of(cell_f),
                lookup,
            );
            let min_turb_r = cell_min_turbined(
                &reversed.unit_groups,
                index_reversed.groups_of(cell_r),
                lookup,
            );
            assert_eq!(
                min_turb_f.to_bits(),
                min_turb_r.to_bits(),
                "bus {bus:?}: min-turbine floor must not depend on group declaration order"
            );

            let min_gen_f = cell_min_generation(
                &forward.unit_groups,
                index_forward.groups_of(cell_f),
                lookup,
            );
            let min_gen_r = cell_min_generation(
                &reversed.unit_groups,
                index_reversed.groups_of(cell_r),
                lookup,
            );
            assert_eq!(
                min_gen_f.to_bits(),
                min_gen_r.to_bits(),
                "bus {bus:?}: min-generation floor must not depend on group declaration order"
            );
        }
    }

    /// The same opposite-binding-sides pair `test_same_bus_groups_sum_into_one_cell_box`
    /// pins (`ρ=1`, groups `(q̄=100, p̄=50)` and `(q̄=10, p̄=100)`), except group A's
    /// resolved `q̄=100` is supplied by a block OVERRIDE over a deliberately wrong
    /// declaration (40.0) rather than by the declaration itself. Fold-then-sum on
    /// the resolved values still gives `min(100,50) + min(10,100) = 60`, not
    /// sum-then-fold's `min(110,150) = 110` — switching the input source did not
    /// reorder the fold.
    #[test]
    fn test_resolved_group_box_still_folds_before_summing() {
        let same_bus = EntityId(600);
        let group_a = make_unit_group(EntityId(31), same_bus, 0.0, 50.0, 0.0, 40.0);
        let group_b = make_unit_group(EntityId(32), same_bus, 0.0, 100.0, 0.0, 10.0);
        let plant = grouped_hydro(1, EntityId(72), vec![group_a, group_b], 150.0, 150.0);

        let hydros = vec![plant];
        let models = vec![ResolvedProductionModel::ConstantProductivity { productivity: 1.0 }];
        let mut fixtures = Fixtures::new(hydros, models, N_STAGES);
        fixtures.set_hydro_bounds(0, STAGE_IDX, hydro_block_bounds(1_000_000.0, 1_000_000.0));
        fixtures.install_group_overlay();
        fixtures.set_group_block_override(
            0,
            0,
            STAGE_IDX,
            0,
            HydroUnitGroupOverride {
                max_turbined_m3s: Some(100.0),
                ..HydroUnitGroupOverride::default()
            },
        );

        let ctx = fixtures.make_ctx();
        assert_eq!(
            ctx.hydro_cell_index.n_cells(),
            1,
            "the same-bus pair must collapse to one cell"
        );

        let stage = three_block_stage(STAGE_IDX);
        let state = state_layout_for(&ctx);
        let layout = StageLayout::new(&ctx, &state, &stage, STAGE_IDX);
        let (mut col_lower, mut col_upper, mut objective) = fresh_bufs(layout.num_cols);
        let mut bufs = ColumnBufs {
            col_lower: &mut col_lower,
            col_upper: &mut col_upper,
            objective: &mut objective,
        };
        fill_turbine_columns(&ctx, &stage, STAGE_IDX, &layout, &mut bufs);

        let turb = col_upper[layout.turbine_col(HydroCell::new(0), BlockIdx::new(0))];
        assert_eq!(
            turb.to_bits(),
            60.0_f64.to_bits(),
            "fold-then-sum on resolved values: min(100,50) + min(10,100) = 50 + 10 = 60, \
             not sum-then-fold's min(110,150) = 110"
        );
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::float_cmp,
    clippy::similar_names
)]
mod ncs_objective_tests {
    use std::collections::{BTreeMap, HashMap};

    use cobre_core::{
        BusStagePenalties, EntityId, HydroStagePenalties, LineStagePenalties, NcsStagePenalties,
        NonControllableSource, PenaltiesCountsSpec, PenaltiesDefaults, ResolvedBounds,
        ResolvedGenericConstraintBounds, ResolvedLoadFactors, ResolvedNcsBounds,
        ResolvedNcsFactors, ResolvedPenalties,
    };
    use cobre_stochastic::par::precompute::PrecomputedPar;

    use crate::hydro_models::{EvaporationModelSet, ProductionModelSet};
    use crate::indexer::HydroCellIndex;
    use crate::lead_time::AnticipatedResolution;
    use crate::resolved_parameters::ResolvedParameters;

    use super::super::layout::ResolvedTables;
    use super::super::test_support::{BLOCK_HOURS, N_BLKS, state_layout_for, three_block_stage};
    use super::{ColumnBufs, StageLayout, TemplateBuildCtx, fill_ncs_columns};

    const N_STAGES: usize = 1;
    const STAGE_IDX: usize = 0;
    const N_NCS: usize = 2;
    const NCS0_ENTITY_COST: f64 = 5.0;
    const NCS0_OVERRIDE_COST: f64 = 77.0;
    const NCS1_ENTITY_COST: f64 = 9.0;

    fn make_ncs(id: i32, curtailment_cost: f64) -> NonControllableSource {
        NonControllableSource {
            id: EntityId(id),
            name: format!("W{id}"),
            operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId(1),
            entry_stage_id: None,
            exit_stage_id: None,
            max_generation_mw: 100.0,
            allow_curtailment: true,
            curtailment_cost,
        }
    }

    fn penalties_with() -> ResolvedPenalties {
        ResolvedPenalties::new(
            &PenaltiesCountsSpec {
                n_hydros: 0,
                n_buses: 0,
                n_lines: 0,
                n_ncs: N_NCS,
                n_stages: N_STAGES,
            },
            &PenaltiesDefaults {
                hydro: HydroStagePenalties {
                    spillage_cost: 0.0,
                    diversion_cost: 0.0,
                    turbined_cost: 0.0,
                    storage_violation_below_cost: 0.0,
                    filling_target_violation_cost: 0.0,
                    turbined_violation_below_cost: 0.0,
                    outflow_violation_below_cost: 0.0,
                    outflow_violation_above_cost: 0.0,
                    generation_violation_below_cost: 0.0,
                    evaporation_violation_cost: 0.0,
                    water_withdrawal_violation_cost: 0.0,
                    water_withdrawal_violation_pos_cost: 0.0,
                    water_withdrawal_violation_neg_cost: 0.0,
                    evaporation_violation_pos_cost: 0.0,
                    evaporation_violation_neg_cost: 0.0,
                    inflow_nonnegativity_cost: 0.0,
                },
                bus: BusStagePenalties { excess_cost: 0.0 },
                line: LineStagePenalties { exchange_cost: 0.0 },
                ncs: NcsStagePenalties {
                    curtailment_cost: 0.0,
                },
            },
        )
    }

    /// Owns the borrow targets for an NCS-only `TemplateBuildCtx`. Seeds the
    /// resolved penalty cell for each entity from its own `curtailment_cost`,
    /// mirroring the entity-seeding step `resolve_penalties` performs before
    /// any override row is applied.
    struct NcsFixtures {
        par_lp: PrecomputedPar,
        hydro_cell_index: HydroCellIndex,
        cascade: cobre_core::CascadeTopology,
        bounds: ResolvedBounds,
        penalties: ResolvedPenalties,
        production_models: ProductionModelSet,
        evaporation_models: EvaporationModelSet,
        resolved_generic_bounds: ResolvedGenericConstraintBounds,
        resolved_load_factors: ResolvedLoadFactors,
        resolved_ncs_bounds: ResolvedNcsBounds,
        resolved_ncs_factors: ResolvedNcsFactors,
        resolved_parameters: ResolvedParameters,
        non_controllable_sources: Vec<NonControllableSource>,
    }

    impl NcsFixtures {
        fn new() -> Self {
            let non_controllable_sources =
                vec![make_ncs(1, NCS0_ENTITY_COST), make_ncs(2, NCS1_ENTITY_COST)];
            let mut penalties = penalties_with();
            for (ncs_sys_idx, ncs) in non_controllable_sources.iter().enumerate() {
                penalties
                    .ncs_penalties_mut(ncs_sys_idx, STAGE_IDX)
                    .curtailment_cost = ncs.curtailment_cost;
            }
            Self {
                par_lp: PrecomputedPar::default(),
                hydro_cell_index: HydroCellIndex::build(&[]),
                cascade: cobre_core::CascadeTopology::build(&[]),
                bounds: ResolvedBounds::empty(),
                penalties,
                production_models: ProductionModelSet::new(vec![], 0, 1),
                evaporation_models: EvaporationModelSet::new(vec![]),
                resolved_generic_bounds: ResolvedGenericConstraintBounds::empty(),
                resolved_load_factors: ResolvedLoadFactors::empty(),
                resolved_ncs_bounds: ResolvedNcsBounds::new(N_NCS, N_STAGES, &[100.0, 100.0]),
                resolved_ncs_factors: ResolvedNcsFactors::new(N_NCS, N_STAGES, N_BLKS),
                resolved_parameters: ResolvedParameters {
                    per_param: vec![],
                    id_to_slot: vec![],
                    cost_scale_factor: 1_000_000.0,
                },
                non_controllable_sources,
            }
        }

        fn set_ncs_override(&mut self, ncs_sys_idx: usize, curtailment_cost: f64) {
            *self.penalties.ncs_penalties_mut(ncs_sys_idx, STAGE_IDX) =
                NcsStagePenalties { curtailment_cost };
        }

        fn make_ctx(&self) -> TemplateBuildCtx<'_> {
            TemplateBuildCtx {
                hydros: &[],
                thermals: &[],
                lines: &[],
                buses: &[],
                load_models: &[],
                cascade: &self.cascade,
                hydro_cell_index: &self.hydro_cell_index,
                resolved: ResolvedTables {
                    bounds: &self.bounds,
                    penalties: &self.penalties,
                    resolved_generic_bounds: &self.resolved_generic_bounds,
                    resolved_load_factors: &self.resolved_load_factors,
                    resolved_ncs_bounds: &self.resolved_ncs_bounds,
                    resolved_ncs_factors: &self.resolved_ncs_factors,
                    resolved_parameters: &self.resolved_parameters,
                },
                hydro_pos: BTreeMap::new(),
                thermal_pos: BTreeMap::new(),
                line_pos: BTreeMap::new(),
                bus_pos: BTreeMap::new(),
                par_lp: &self.par_lp,
                production_models: &self.production_models,
                evaporation_models: &self.evaporation_models,
                generic_constraints: &[],
                non_controllable_sources: &self.non_controllable_sources,
                pumping_stations: &[],
                pumping_pos: BTreeMap::new(),
                n_pumping: 0,
                contracts: &[],
                contract_pos: BTreeMap::new(),
                n_contract_import: 0,
                n_contract_export: 0,
                diversion_upstream: HashMap::new(),
                arc_stage_weights: HashMap::new(),
                arc_spread_chrono: HashMap::new(),
                arc_arrival_density: HashMap::new(),
                per_stage_mask: Vec::new(),
                post_study_resolved: crate::setup::PostStudyResolved::default(),
                n_hydros: 0,
                n_thermals: 0,
                n_lines: 0,
                n_buses: 0,
                max_par_order: 0,
                n_anticipated: 0,
                k_max: 0,
                anticipated_lead_stages: vec![],
                anticipated_thermal_indices: vec![],
                anticipated_windows: vec![],
                anticipated_resolution: AnticipatedResolution::default(),
                study_stage_ids: (0..N_STAGES as i32).collect(),
                delivery_stage_ids: (0..N_STAGES as i32).collect(),
                has_penalty: false,
                delivery_cumulative_discount_factors: vec![1.0; N_STAGES],
                delivery_total_hours: vec![BLOCK_HOURS.iter().sum(); N_STAGES],
                filling_v_target: BTreeMap::new(),
            }
        }
    }

    /// The `col_ncs_start` offset `run_fill` resolves while the `StageLayout` is
    /// alive, plus `n_blks` — enough to reconstruct every block-major NCS
    /// address (`col_ncs_start + ncs_sys_idx * n_blks + blk`, the same formula
    /// [`BlockGrid::flat`](crate::indexer::BlockGrid::flat) applies) after the
    /// layout itself is dropped.
    struct FillOffsets {
        col_ncs_start: usize,
        n_blks: usize,
    }

    impl FillOffsets {
        fn at(&self, ncs_sys_idx: usize, blk: usize) -> usize {
            self.col_ncs_start + ncs_sys_idx * self.n_blks + blk
        }
    }

    struct FillResult {
        objective: Vec<f64>,
        offsets: FillOffsets,
    }

    /// Run the NCS column fill against `fixtures` at `stage_idx`, over a
    /// three-block stage.
    fn run_fill(fixtures: &NcsFixtures, stage_idx: usize) -> FillResult {
        let stage = three_block_stage(stage_idx);
        let ctx = fixtures.make_ctx();
        let state = state_layout_for(&ctx);
        let layout = StageLayout::new(&ctx, &state, &stage, stage_idx);

        let mut col_lower = vec![0.0_f64; layout.num_cols];
        let mut col_upper = vec![f64::INFINITY; layout.num_cols];
        let mut objective = vec![0.0_f64; layout.num_cols];
        let mut bufs = ColumnBufs {
            col_lower: &mut col_lower,
            col_upper: &mut col_upper,
            objective: &mut objective,
        };
        fill_ncs_columns(&ctx, &stage, stage_idx, &layout, &mut bufs);

        let offsets = FillOffsets {
            col_ncs_start: layout.equipment.col_ncs_start,
            n_blks: layout.n_blks,
        };

        FillResult { objective, offsets }
    }

    /// Two NCS entities allowing curtailment: entity 0's resolved cell is
    /// overridden to `NCS0_OVERRIDE_COST` while its entity constant stays
    /// `NCS0_ENTITY_COST`. Every one of entity 0's three block columns must
    /// price the objective from the override, not the entity constant.
    #[test]
    fn ncs_objective_reads_the_per_stage_override() {
        let mut fixtures = NcsFixtures::new();
        fixtures.set_ncs_override(0, NCS0_OVERRIDE_COST);

        let result = run_fill(&fixtures, STAGE_IDX);
        let off = &result.offsets;

        for (blk, &hours) in BLOCK_HOURS.iter().enumerate() {
            assert_eq!(
                result.objective[off.at(0, blk)].to_bits(),
                (-NCS0_OVERRIDE_COST * hours).to_bits(),
                "ncs 0 objective must read the per-stage override, blk {blk}"
            );
        }
    }

    /// The same fixture as above: entity 1's resolved cell is left untouched
    /// at the seeded default, which equals its own entity `curtailment_cost`
    /// (`NCS1_ENTITY_COST`). Every one of entity 1's three block columns must
    /// still price the objective from that entity constant.
    #[test]
    fn ncs_objective_falls_back_to_the_seeded_default() {
        let mut fixtures = NcsFixtures::new();
        fixtures.set_ncs_override(0, NCS0_OVERRIDE_COST);

        let result = run_fill(&fixtures, STAGE_IDX);
        let off = &result.offsets;

        for (blk, &hours) in BLOCK_HOURS.iter().enumerate() {
            assert_eq!(
                result.objective[off.at(1, blk)].to_bits(),
                (-NCS1_ENTITY_COST * hours).to_bits(),
                "ncs 1 objective must fall back to the seeded entity default, blk {blk}"
            );
        }
    }
}
