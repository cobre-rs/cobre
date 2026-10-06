//! Variable reference to LP column index mapping for generic constraints.
//!
//! `resolve_variable_ref` maps a [`VariableRef`] and block index to a list of
//! `(column_index, coefficient_multiplier)` pairs; the LP builder calls it for each
//! [`cobre_core::LinearTerm`] of a generic-constraint expression to produce CSC
//! entries. Column offsets come from the same two owners every other builder fill
//! function reads — [`TemplateBuildCtx`] for entity slices and position maps,
//! [`StageLayout`] for this stage's column/row ranges and the typed accessors over
//! them, with all block-stride arithmetic routed through the single-owner
//! stride primitive those accessors already carry. `HydroStorage`/`HydroInflow`
//! resolve through `layout.state`'s `StateSpace` handle, so a generic constraint
//! lands on the same column the cut path reads.
//!
//! For a block-level variable with `block_id = None`, the resolver returns the
//! column for the *current* `block_idx`; the caller loops over blocks and calls once
//! per block, so per-block expansion happens in the caller, not here.
//!
//! `PumpingPower` aliases the SAME flow column as `PumpingFlow`, scaled by the
//! station's `consumption_mw_per_m3s` — power is affine in flow, so a column of its
//! own would be an unconstrained free variable. Variables referencing entity types
//! with no LP columns (contracts, non-controllable sources, withdrawal) return an
//! empty vec.

use cobre_core::{ContractType, EntityId, PumpingStation, VariableRef};

use super::delivery_ring::{maturing_bucket_in_col, resolve_bucket_arrival_density};
use super::hydro_state::resolve_shortcircuit_target;
use super::layout::{StageLayout, TemplateBuildCtx, contract_family_slot, evaporation_slot};
use crate::hydro_models::ResolvedProductionModel;
use crate::indexer::{
    BlockIdx, Boundary, BusSys, EvapLocal, FphaCellLocal, HydroCell, HydroSys, LineSys, PumpingSys,
    ThermalSys,
};

/// Map a [`VariableRef`] and block index to LP column indices with multipliers.
///
/// Returns a `Vec<(column_index, coefficient_multiplier)>`; the caller scales each
/// entry by the `LinearTerm::coefficient` for the final CSC value. `block_idx` is
/// ignored for stage-level variables and overridden by `block_id = Some(b)` for
/// block-level ones.
///
/// # Returns
///
/// An empty vec when the entity ID is absent from the relevant position map
/// (defense-in-depth past referential validation), or when the variable references
/// a stub entity with no LP columns (contracts, non-controllable sources,
/// withdrawal).
#[must_use]
pub(super) fn resolve_variable_ref(
    var_ref: &VariableRef,
    block_idx: usize,
    stage_idx: usize,
    ctx: &TemplateBuildCtx<'_>,
    layout: &StageLayout<'_>,
) -> Vec<(usize, f64)> {
    let at = |block_id: Option<usize>| BlockIdx::new(block_id.unwrap_or(block_idx));
    match var_ref {
        VariableRef::HydroStorage { hydro_id } => resolve_hydro_storage(*hydro_id, ctx, layout),

        VariableRef::HydroStorageInitial { hydro_id, block_id }
        | VariableRef::HydroUsefulVolumeInitial { hydro_id, block_id } => {
            resolve_hydro_storage_boundary(*hydro_id, *block_id, 0, ctx, layout)
        }

        VariableRef::HydroStorageFinal { hydro_id, block_id }
        | VariableRef::HydroUsefulVolumeFinal { hydro_id, block_id } => {
            resolve_hydro_storage_boundary(*hydro_id, *block_id, 1, ctx, layout)
        }

        VariableRef::HydroEvaporation { hydro_id, block_id } => {
            resolve_hydro_evaporation(*hydro_id, *block_id, ctx, layout)
        }

        VariableRef::HydroInflow { hydro_id, block_id } => {
            resolve_hydro_inflow(*hydro_id, at(*block_id), stage_idx, ctx, layout)
        }

        VariableRef::HydroTurbined {
            hydro_id,
            block_id,
            bus_id,
        } => resolve_turbine_cells(*hydro_id, *bus_id, at(*block_id), 1.0, ctx, layout),

        VariableRef::HydroSpillage { hydro_id, block_id } => {
            resolve_hydro_spillage(*hydro_id, at(*block_id), ctx, layout)
        }

        VariableRef::HydroOutflow { hydro_id, block_id } => {
            resolve_hydro_outflow(*hydro_id, at(*block_id), ctx, layout)
        }

        VariableRef::HydroGeneration {
            hydro_id,
            block_id,
            bus_id,
        } => resolve_hydro_generation(*hydro_id, *bus_id, at(*block_id), stage_idx, ctx, layout),

        VariableRef::ThermalGeneration {
            thermal_id,
            block_id,
        } => resolve_thermal_generation(*thermal_id, at(*block_id), ctx, layout),

        VariableRef::LineDirect { line_id, block_id } => {
            resolve_line_direct(*line_id, at(*block_id), ctx, layout)
        }

        VariableRef::LineReverse { line_id, block_id } => {
            resolve_line_reverse(*line_id, at(*block_id), ctx, layout)
        }

        VariableRef::LineExchange { line_id, block_id } => {
            resolve_line_exchange(*line_id, at(*block_id), ctx, layout)
        }

        VariableRef::BusDeficit { bus_id, block_id } => {
            resolve_bus_deficit(*bus_id, at(*block_id), ctx, layout)
        }

        VariableRef::BusExcess { bus_id, block_id } => {
            resolve_bus_excess(*bus_id, at(*block_id), ctx, layout)
        }

        VariableRef::HydroDiversion { hydro_id, block_id } => {
            resolve_hydro_diversion(*hydro_id, at(*block_id), ctx, layout)
        }

        VariableRef::AnticipatedDecision { thermal_id } => {
            resolve_anticipated_decision(*thermal_id, ctx, layout)
        }

        VariableRef::PumpingFlow {
            station_id,
            block_id,
        } => resolve_pumping_column(*station_id, at(*block_id), ctx, layout, |_| 1.0),

        VariableRef::PumpingPower {
            station_id,
            block_id,
        } => resolve_pumping_column(*station_id, at(*block_id), ctx, layout, |station| {
            station.consumption_mw_per_m3s
        }),

        VariableRef::ContractImport {
            contract_id,
            block_id,
        } => resolve_contract_column(
            *contract_id,
            ContractType::Import,
            at(*block_id),
            ctx,
            layout,
        ),

        VariableRef::ContractExport {
            contract_id,
            block_id,
        } => resolve_contract_column(
            *contract_id,
            ContractType::Export,
            at(*block_id),
            ctx,
            layout,
        ),

        // Registered in the data model but no LP decision column: withdrawal is a
        // schedule fixed by bounds; non-controllable sources carry no decision column.
        VariableRef::HydroWithdrawal { .. }
        | VariableRef::NonControllableGeneration { .. }
        | VariableRef::NonControllableCurtailment { .. } => vec![],
    }
}

/// One `(column, multiplier)` pair per cell of the plant addressed by
/// `hydro_id` in the `Turbine` family, or exactly one pair when `bus_id` names
/// one of the plant's cells — the correct plant-level resolution now that the
/// family is sized by cell, not by plant: a `VariableRef` with `bus_id: None`
/// means the whole plant, i.e. every cell's column, matching today's single
/// pair under the identity partition. `Some(b)` resolves through
/// `ctx.hydro_cell_index.cell_of_bus`, never `Hydro::bus_id`: the cell's bus comes
/// from a unit group's `bus_id`, an independent value the plant's own field
/// need not match. Returns an empty vec on an `EntityPositions::hydro` miss (mirrors every
/// other resolver's guard) or a `bus_id` naming no cell of the plant.
fn resolve_turbine_cells(
    hydro_id: EntityId,
    bus_id: Option<EntityId>,
    blk: BlockIdx,
    multiplier: f64,
    ctx: &TemplateBuildCtx<'_>,
    layout: &StageLayout<'_>,
) -> Vec<(usize, f64)> {
    let Some(pos) = ctx.positions.hydro(hydro_id) else {
        return vec![];
    };
    let sys = HydroSys::new(pos);
    let flat = |c: usize| {
        (
            layout.geometry.turbine_col(HydroCell::new(c), blk),
            multiplier,
        )
    };
    match bus_id {
        Some(b) => ctx
            .hydro_cell_index
            .cell_of_bus(sys, b)
            .map(|cell| vec![flat(cell.get())])
            .unwrap_or_default(),
        None => ctx.hydro_cell_index.cells_of(sys).map(flat).collect(),
    }
}

/// Resolve `HydroStorage` to its stage-level outgoing storage column.
///
/// Role (a): the storage column is `layout.state.storage_outgoing_col(h)`, read
/// through the state handle. Returns empty vec when the hydro ID is not found
/// in `ctx.positions`.
fn resolve_hydro_storage(
    hydro_id: EntityId,
    ctx: &TemplateBuildCtx<'_>,
    layout: &StageLayout<'_>,
) -> Vec<(usize, f64)> {
    resolve_block_column(ctx.positions.hydro(hydro_id), |pos| {
        layout.state.storage_outgoing_col(HydroSys::new(pos)).get()
    })
}

/// Resolve `HydroStorageInitial`/`HydroStorageFinal` to a single fixed storage
/// boundary column via [`StageLayout::block_storage_col`].
/// `boundary_offset = 0` (initial): `Some(k)` → boundary `k`, `None` → stage-initial
/// `S⁰` (boundary `0`). `boundary_offset = 1` (final): `Some(k)` → boundary `k + 1`,
/// `None` → stage-final `Sᴷ` (boundary `K`). Both are stage-level stocks (fixed
/// column, no per-block expansion). Returns an empty vec on an `EntityPositions::hydro` miss
/// (mirrors [`resolve_hydro_storage`]).
fn resolve_hydro_storage_boundary(
    hydro_id: EntityId,
    block_id: Option<usize>,
    boundary_offset: usize,
    ctx: &TemplateBuildCtx<'_>,
    layout: &StageLayout<'_>,
) -> Vec<(usize, f64)> {
    resolve_block_column(ctx.positions.hydro(hydro_id), |pos| {
        let k = match block_id {
            Some(k) => k + boundary_offset,
            None => boundary_offset * layout.clock.n_blks(),
        };
        let boundary = Boundary::from_index(k, layout.clock.n_blks());
        layout.block_storage_col(HydroSys::new(pos), boundary)
    })
}

/// Resolve `HydroInflow` to the cascade total-inflow expression at `blk`: the
/// incremental (local) `z_inflow` column, plants diverting into `h`, each
/// upstream release weighted — for an operating reservoir — by the share the
/// downstream balance row credits to this block, and the maturing transit water
/// entering as a rate.
///
/// Each hydro `u` whose `PreFilling` short-circuit targets `h`
/// ([`resolve_shortcircuit_target`]) also adds its own local inflow, diverted
/// inflow, and upstream releases, each at `1.0` in `blk` — the same whole,
/// no-lag routing `fill_prefilling_shortcircuit` gives them on the
/// water-balance row — and its maturing bucket as a rate.
///
/// This is an instantaneous **rate** identity (m³/s), **not** the `−τ`-weighted (hm³)
/// storage-balance row — the `−τ` sign and `τ` weighting belong to storage balance
/// and must not be copied here. `h`'s own outflows, evaporation, withdrawal slacks,
/// AR-lag-`ψ`, and pumped transfer are excluded (storage-balance / loss / outflow
/// terms, or no LP column).
///
/// Upstream releases iterate `ctx.cascade.upstream(h)`; diverted inflow iterates
/// `ctx.diversion_upstream[h]` (values already system indices). Both are canonically
/// ordered at build time, so emitted pairs are input-ordering-independent with no
/// extra sort.
///
/// The `layout.state.z_inflow.is_empty()` guard is load-bearing: `z_inflow` is empty
/// when `hydro_count == 0` (unlike `storage`), so `z_inflow_col` would be
/// meaningless. Returns an empty vec when `hydro_count == 0` or `hydro_id` is unknown.
fn resolve_hydro_inflow(
    hydro_id: EntityId,
    blk: BlockIdx,
    stage_idx: usize,
    ctx: &TemplateBuildCtx<'_>,
    layout: &StageLayout<'_>,
) -> Vec<(usize, f64)> {
    if layout.state.z_inflow.is_empty() {
        return vec![];
    }
    let Some(pos_h) = ctx.positions.hydro(hydro_id) else {
        return vec![];
    };

    let upstream = ctx.cascade.upstream(hydro_id);
    let has_release_columns =
        !layout.geometry.turbine.is_empty() && !layout.geometry.spillage.is_empty();

    let mut result = Vec::with_capacity(2 + 2 * upstream.len());

    push_local_inflow_rate(pos_h, blk, ctx, layout, &mut result);

    if has_release_columns {
        for &up_id in upstream {
            if let Some(pos_up) = ctx.positions.hydro(up_id) {
                push_upstream_release_rate(pos_up, blk, stage_idx, ctx, layout, &mut result);
            }
        }
    }

    push_maturing_bucket_rate(pos_h, blk, stage_idx, ctx, layout, &mut result);

    let stage_id = ctx.time_value.delivery_stage_ids()[stage_idx];
    for u_idx in 0..ctx.hydros.len() {
        if resolve_shortcircuit_target(ctx.hydros, ctx.cascade, ctx.positions, stage_id, u_idx)
            != Some(pos_h)
        {
            continue;
        }
        push_local_inflow_rate(u_idx, blk, ctx, layout, &mut result);
        push_maturing_bucket_rate(u_idx, blk, stage_idx, ctx, layout, &mut result);
        if has_release_columns {
            for &w_id in ctx.cascade.upstream(ctx.hydros[u_idx].id) {
                if let Some(pos_w) = ctx.positions.hydro(w_id) {
                    push_release_columns(pos_w, blk, 1.0, ctx, layout, &mut result);
                }
            }
        }
    }

    result
}

/// Push plant `plant_idx`'s own local inflow rate onto `out`: its `z_inflow`
/// column, then each `diversion_upstream` source into it, both at `1.0`
/// (mirrors `push_z_inflow_coupling`'s z coupling and the diversion-inflow
/// loop in `fill_state_and_water_entries`, both `lp/builder/entries.rs`).
fn push_local_inflow_rate(
    plant_idx: usize,
    blk: BlockIdx,
    ctx: &TemplateBuildCtx<'_>,
    layout: &StageLayout<'_>,
    out: &mut Vec<(usize, f64)>,
) {
    out.push((
        layout.state.z_inflow_col(HydroSys::new(plant_idx)).get(),
        1.0,
    ));

    if !layout.geometry.diversion.is_empty() {
        let diversion_into = ctx
            .diversion_upstream
            .get(&ctx.hydros[plant_idx].id)
            .map_or(&[][..], Vec::as_slice);
        for &d_idx in diversion_into {
            out.push((
                layout.geometry.diversion_col(HydroSys::new(d_idx), blk),
                1.0,
            ));
        }
    }
}

/// Push plant `plant_idx`'s maturing bucket onto `out` as the rate
/// `arrival_density[blk] / τ(blk)` (mirrors `push_maturing_bucket_coupling`,
/// `lp/builder/entries.rs`). A no-op when the plant declares no incoming arc.
fn push_maturing_bucket_rate(
    plant_idx: usize,
    blk: BlockIdx,
    stage_idx: usize,
    ctx: &TemplateBuildCtx<'_>,
    layout: &StageLayout<'_>,
    out: &mut Vec<(usize, f64)>,
) {
    let Some(col) = maturing_bucket_in_col(layout.state, HydroSys::new(plant_idx)) else {
        return;
    };
    let rho = resolve_bucket_arrival_density(
        ctx,
        layout.clock,
        stage_idx,
        ctx.hydros[plant_idx].id,
        layout.clock.n_blks(),
    )[blk.get()];
    if rho != 0.0 {
        out.push((col, rho / layout.clock.tau(blk)));
    }
}

/// Push arc `u_idx → h`'s per-block release rate onto `out`, mirroring the water
/// balance's own branch (`fill_arc_release_block_entries` /
/// `fill_arc_release_chrono_block_entries`): a chronological spread entry routes
/// each source block's share; a stage-clock-weights entry uses its same-block share
/// `k_0` bare (the balance's own `τ_b · k_0` product, never re-multiplied); absent
/// either, the whole release lands at `blk` at `+1.0` (the zero-lag reduction).
fn push_upstream_release_rate(
    u_idx: usize,
    blk: BlockIdx,
    stage_idx: usize,
    ctx: &TemplateBuildCtx<'_>,
    layout: &StageLayout<'_>,
    out: &mut Vec<(usize, f64)>,
) {
    if let Some(res) = ctx
        .topology
        .arc_spread_chrono
        .get(&u_idx)
        .and_then(|by_stage| by_stage[stage_idx].as_ref())
    {
        for src in 0..=blk.get() {
            let j = blk.get() - src;
            let Some(&r) = res.within_stage_routing[src].get(j) else {
                continue;
            };
            if r == 0.0 {
                continue;
            }
            let coeff = r * layout.clock.tau(BlockIdx::new(src)) / layout.clock.tau(blk);
            push_release_columns(u_idx, BlockIdx::new(src), coeff, ctx, layout, out);
        }
        return;
    }

    let Some(k_by_stage) = ctx.topology.arc_stage_weights.get(&u_idx) else {
        push_release_columns(u_idx, blk, 1.0, ctx, layout, out);
        return;
    };
    let k0 = k_by_stage[stage_idx][0];
    if k0 != 0.0 {
        push_release_columns(u_idx, blk, k0, ctx, layout, out);
    }
}

/// Push plant `u_idx`'s release columns (every cell's turbine column, then
/// spillage) onto `out` at `coeff`: a plant's release is `Σ_c q_c + s` over a
/// disjoint cell partition, so `coeff` is replicated across cells, never divided
/// (mirrors `push_plant_release` in `lp/builder/entries.rs`).
fn push_release_columns(
    u_idx: usize,
    blk: BlockIdx,
    coeff: f64,
    ctx: &TemplateBuildCtx<'_>,
    layout: &StageLayout<'_>,
    out: &mut Vec<(usize, f64)>,
) {
    let sys_up = HydroSys::new(u_idx);
    for cell in ctx.hydro_cell_index.cells_of(sys_up) {
        out.push((
            layout.geometry.turbine_col(HydroCell::new(cell), blk),
            coeff,
        ));
    }
    out.push((layout.geometry.spillage_col(sys_up, blk), coeff));
}

/// Resolve `HydroEvaporation` to the evaporation-outflow column for the matching
/// hydro; empty vec when the hydro has no linearized evaporation at this stage, or
/// when `block_id` names a block `>= layout.clock.n_blks()`. `None` maps to block 0. On a
/// chronological stage each block resolves to its own slot; `None` in `K > 1`
/// (where blocks differ) is rejected upstream by generic-constraint validation, so
/// it is not reached here for a valid study. On a parallel stage `None`/`Some(0)`
/// resolve to the one stage-level slot; `Some(k >= 1)` is rejected by the same
/// validation, so the collapse below onto that slot for `k >= 1`
/// (`evaporation_slot`/`evaporation_slot_count`) is reached only by a study that
/// bypassed validation.
fn resolve_hydro_evaporation(
    hydro_id: EntityId,
    block_id: Option<usize>,
    ctx: &TemplateBuildCtx<'_>,
    layout: &StageLayout<'_>,
) -> Vec<(usize, f64)> {
    let Some(sys_pos) = ctx.positions.hydro(hydro_id) else {
        return vec![];
    };
    // Linear scan: cold template-build path over a handful of evap hydros, so an
    // O(1) reverse map is not warranted (unlike `resolve_anticipated_decision`).
    let Some(local_idx) = layout
        .geometry
        .evap_hydro_indices
        .iter()
        .position(|&p| p.get() == sys_pos)
    else {
        return vec![];
    };
    let blk = block_id.unwrap_or(0);
    if blk >= layout.clock.n_blks() {
        return vec![];
    }
    let slot = evaporation_slot(layout.n_evap_slots, BlockIdx::new(blk));
    vec![(layout.evap_flow_col(EvapLocal::new(local_idx), slot), 1.0)]
}

/// Resolve `HydroOutflow` to turbine (every cell of the plant, summed) plus
/// spillage (one plant-keyed column). An `EntityPositions::hydro` miss returns an empty vec,
/// never a partial reading.
fn resolve_hydro_outflow(
    hydro_id: EntityId,
    blk: BlockIdx,
    ctx: &TemplateBuildCtx<'_>,
    layout: &StageLayout<'_>,
) -> Vec<(usize, f64)> {
    let Some(pos) = ctx.positions.hydro(hydro_id) else {
        return vec![];
    };
    let mut result = resolve_turbine_cells(hydro_id, None, blk, 1.0, ctx, layout);
    result.push((layout.geometry.spillage_col(HydroSys::new(pos), blk), 1.0));
    result
}

/// Resolve `HydroGeneration` by dispatching on the production model.
///
/// - FPHA hydros: maps to the generation column at
///   `layout.geometry.generation_col(FphaCellLocal::new(first.get() + offset), blk)`, one
///   pair per cell, or exactly one when `bus_id` names a cell.
/// - Constant-productivity hydros: maps to the turbine column(s) scaled by
///   productivity, threading `bus_id` through [`resolve_turbine_cells`] — one
///   plant, one productivity (`ProductionModelSet::model` carries no group or
///   cell axis), so scaling the selected cell alone is exact.
fn resolve_hydro_generation(
    hydro_id: EntityId,
    bus_id: Option<EntityId>,
    blk: BlockIdx,
    stage_idx: usize,
    ctx: &TemplateBuildCtx<'_>,
    layout: &StageLayout<'_>,
) -> Vec<(usize, f64)> {
    let Some(sys_pos) = ctx.positions.hydro(hydro_id) else {
        return vec![];
    };
    match ctx.production_models.model(sys_pos, stage_idx) {
        ResolvedProductionModel::Fpha { .. } => {
            // `layout.fpha_local_index` is the reverse map's single owner (indexed
            // by system hydro position), an O(1) lookup unlike the evaporation scan.
            let Some(fpha_local) = layout.fpha_local_index[sys_pos] else {
                return vec![];
            };
            let sys = HydroSys::new(sys_pos);
            let fpha_cell_start = layout.fpha_local_first_cell(fpha_local);
            let flat = |fpha_idx: usize| {
                (
                    layout
                        .geometry
                        .generation_col(FphaCellLocal::new(fpha_idx), blk),
                    1.0,
                )
            };
            if let Some(b) = bus_id {
                ctx.hydro_cell_index
                    .cell_of_bus(sys, b)
                    .map(|cell| {
                        // Offset within the PLANT'S OWN cell range, never the absolute
                        // cell index: `fpha_local_first_cell` prefixes FPHA plants only,
                        // so a leading non-FPHA plant makes the two diverge.
                        let offset = cell.get() - ctx.hydro_cell_index.cells_of(sys).start;
                        vec![flat(fpha_cell_start.get() + offset)]
                    })
                    .unwrap_or_default()
            } else {
                let n_cells = ctx.hydro_cell_index.cells_of(sys).len();
                (0..n_cells)
                    .map(|i| flat(fpha_cell_start.get() + i))
                    .collect()
            }
        }
        ResolvedProductionModel::ConstantProductivity { productivity } => {
            resolve_turbine_cells(hydro_id, bus_id, blk, *productivity, ctx, layout)
        }
    }
}

/// Resolve `LineExchange` (net = forward − reverse) to two columns with signs.
fn resolve_line_exchange(
    line_id: EntityId,
    blk: BlockIdx,
    ctx: &TemplateBuildCtx<'_>,
    layout: &StageLayout<'_>,
) -> Vec<(usize, f64)> {
    let Some(pos) = ctx.positions.line(line_id) else {
        return vec![];
    };
    let sys = LineSys::new(pos);
    vec![
        (layout.geometry.line_fwd_col(sys, blk), 1.0),
        (layout.geometry.line_rev_col(sys, blk), -1.0),
    ]
}

/// Resolve `BusDeficit` to one column per deficit segment via
/// [`StageLayout::deficit_col`]. The segment count `S` comes from
/// `layout.equipment.max_deficit_segments`.
fn resolve_bus_deficit(
    bus_id: EntityId,
    blk: BlockIdx,
    ctx: &TemplateBuildCtx<'_>,
    layout: &StageLayout<'_>,
) -> Vec<(usize, f64)> {
    let Some(b_pos) = ctx.positions.bus(bus_id) else {
        return vec![];
    };
    (0..layout.equipment.max_deficit_segments)
        .map(|seg| (layout.deficit_col(BusSys::new(b_pos), seg, blk), 1.0))
        .collect()
}

/// Resolve `AnticipatedDecision` to `layout.geometry.anticipated_decision_col(local)`,
/// the per-plant stage-level decision column.
///
/// Returns an empty vec when `thermal_id` has no `ctx.positions.thermal` slot, or the
/// thermal is not in `ctx.study_dims.anticipated_plants` (`AnticipatedPlants::local_of`
/// returns `None`) — both defense-in-depth past semantic validation
/// (`check_anticipated_decision_target_is_anticipated`).
fn resolve_anticipated_decision(
    thermal_id: EntityId,
    ctx: &TemplateBuildCtx<'_>,
    layout: &StageLayout<'_>,
) -> Vec<(usize, f64)> {
    let Some(sys_pos) = ctx.positions.thermal(thermal_id) else {
        return vec![];
    };
    if let Some(local) = ctx
        .study_dims
        .anticipated_plants
        .local_of(ThermalSys::new(sys_pos))
    {
        vec![(layout.geometry.anticipated_decision_col(local), 1.0)]
    } else {
        vec![]
    }
}

/// Resolve `PumpingFlow`/`PumpingPower` to the block-major pumping-flow column.
///
/// Both variants resolve to the SAME flow column — `PumpingPower` has no column of
/// its own; resolving it to a separate column would create an unconstrained free
/// variable. The coefficient comes from `coeff_fn`: `|_| 1.0` for flow,
/// `|s| s.consumption_mw_per_m3s` for power (power is affine in flow).
///
/// Addressed by the station's SYSTEM index `p_idx`, not an active-local index:
/// under the dense layout the column block is system-indexed, so the system
/// index IS the correct column-block position at every stage (a dormant
/// station keeps its zeroed column).
///
/// Returns an empty vec on an unknown station or no stations (`EntityPositions::pumping` miss);
/// `n_pumping == 0` is handled by the same guard. No panic.
fn resolve_pumping_column(
    station_id: EntityId,
    blk: BlockIdx,
    ctx: &TemplateBuildCtx<'_>,
    layout: &StageLayout<'_>,
    coeff_fn: impl Fn(&PumpingStation) -> f64,
) -> Vec<(usize, f64)> {
    let Some(p_idx) = ctx.positions.pumping(station_id) else {
        return vec![];
    };
    // Guard rather than index to uphold no-panic if `EntityPositions::pumping` and
    // `pumping_stations` ever diverge (both built from the same ID-sorted slice).
    let Some(station) = ctx.pumping_stations.get(p_idx) else {
        return vec![];
    };
    let col = layout
        .geometry
        .pumping_flow_col(PumpingSys::new(p_idx), blk);
    vec![(col, coeff_fn(station))]
}

/// Resolve `ContractImport`/`ContractExport` to the block-major contract column via
/// [`StageGeometry::contract_col`](super::layout::StageGeometry::contract_col).
///
/// The injection/withdrawal LOAD-BALANCE sign is owned by the load-balance fill, not
/// here: the resolved coefficient is the variable's own unit `+1.0` (the
/// generic-constraint coefficient is the user's), matching `resolve_pumping_column`'s
/// `|_| 1.0`.
///
/// `contract_col` addresses by the contract's per-family slot
/// ([`contract_family_slot`]) so the import block precedes the export block under
/// the dense layout. A dormant (commissioning-window-inactive) contract keeps its
/// `[0, 0]` column, so the column always exists.
///
/// Returns an empty vec on an unknown contract id (`EntityPositions::contract` miss) or a
/// direction mismatch (the referenced family differs from the contract's
/// `contract_type` — a referential-validation gap), mirroring the pumping precedent.
/// No panic.
fn resolve_contract_column(
    contract_id: EntityId,
    family: ContractType,
    blk: BlockIdx,
    ctx: &TemplateBuildCtx<'_>,
    layout: &StageLayout<'_>,
) -> Vec<(usize, f64)> {
    let Some(c_sys) = ctx.positions.contract(contract_id) else {
        return vec![];
    };
    let Some(contract) = ctx.contracts.get(c_sys) else {
        return vec![];
    };
    if contract.contract_type != family {
        return vec![];
    }
    let (_, family_slot) = contract_family_slot(ctx.contracts, c_sys);
    let col = layout.geometry.contract_col(family, family_slot, blk);
    vec![(col, 1.0)]
}

/// Resolve a position lookup to its `(column_index, 1.0)` pair via `col`, the
/// single owner of the "position miss returns an empty vec" rule every
/// block-major single-column family shares.
fn resolve_block_column(pos: Option<usize>, col: impl Fn(usize) -> usize) -> Vec<(usize, f64)> {
    if let Some(pos) = pos {
        vec![(col(pos), 1.0)]
    } else {
        vec![]
    }
}

/// Resolve `HydroSpillage` via the single-column dispatcher.
fn resolve_hydro_spillage(
    hydro_id: EntityId,
    blk: BlockIdx,
    ctx: &TemplateBuildCtx<'_>,
    layout: &StageLayout<'_>,
) -> Vec<(usize, f64)> {
    resolve_block_column(ctx.positions.hydro(hydro_id), |pos| {
        layout.geometry.spillage_col(HydroSys::new(pos), blk)
    })
}

/// Resolve `HydroDiversion` via the single-column dispatcher.
fn resolve_hydro_diversion(
    hydro_id: EntityId,
    blk: BlockIdx,
    ctx: &TemplateBuildCtx<'_>,
    layout: &StageLayout<'_>,
) -> Vec<(usize, f64)> {
    resolve_block_column(ctx.positions.hydro(hydro_id), |pos| {
        layout.geometry.diversion_col(HydroSys::new(pos), blk)
    })
}

/// Resolve `ThermalGeneration` via the single-column dispatcher.
fn resolve_thermal_generation(
    thermal_id: EntityId,
    blk: BlockIdx,
    ctx: &TemplateBuildCtx<'_>,
    layout: &StageLayout<'_>,
) -> Vec<(usize, f64)> {
    resolve_block_column(ctx.positions.thermal(thermal_id), |pos| {
        layout.geometry.thermal_col(ThermalSys::new(pos), blk)
    })
}

/// Resolve `LineDirect` via the single-column dispatcher.
fn resolve_line_direct(
    line_id: EntityId,
    blk: BlockIdx,
    ctx: &TemplateBuildCtx<'_>,
    layout: &StageLayout<'_>,
) -> Vec<(usize, f64)> {
    resolve_block_column(ctx.positions.line(line_id), |pos| {
        layout.geometry.line_fwd_col(LineSys::new(pos), blk)
    })
}

/// Resolve `LineReverse` via the single-column dispatcher.
fn resolve_line_reverse(
    line_id: EntityId,
    blk: BlockIdx,
    ctx: &TemplateBuildCtx<'_>,
    layout: &StageLayout<'_>,
) -> Vec<(usize, f64)> {
    resolve_block_column(ctx.positions.line(line_id), |pos| {
        layout.geometry.line_rev_col(LineSys::new(pos), blk)
    })
}

/// Resolve `BusExcess` via the single-column dispatcher.
fn resolve_bus_excess(
    bus_id: EntityId,
    blk: BlockIdx,
    ctx: &TemplateBuildCtx<'_>,
    layout: &StageLayout<'_>,
) -> Vec<(usize, f64)> {
    resolve_block_column(ctx.positions.bus(bus_id), |pos| {
        layout.geometry.excess_col(BusSys::new(pos), blk)
    })
}

#[cfg(test)]
mod tests;
