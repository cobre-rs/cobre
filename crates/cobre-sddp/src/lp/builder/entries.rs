use cobre_core::commissioning::Phase;
use cobre_core::{BlockMode, CoefficientRef, ContractType, Stage};

use super::generic_constraints::resolve_variable_ref;
use crate::hydro_models::EvaporationModel;
use crate::indexer::{
    AnticipatedLocal, BlockIdx, Boundary, BusSys, EvapLocal, FillingTargetLocal, FloorLocal,
    FphaCellLocal, HydroCell, HydroSys, LineSys, NcsSys, PumpingSys, ThermalSys,
    for_each_ring_residue,
};

use super::delivery_ring::{DeliveryRing, maturing_bucket_in_col, resolve_bucket_arrival_density};
use super::fpha_cursor::for_each_fpha_plane;
use super::hydro_state::{hydro_phase, resolve_shortcircuit_target};
use super::layout::{StageLayout, StageProductionRole, TemplateBuildCtx, contract_family_slot};

/// Fishing (consumption) coupling: for every anticipated plant whose
/// delivery matures THIS stage
/// (`layout.anticipated.anticipated_fishing_row_pos`, `None` at a `K = 0`
/// self-delivery or when no delivery matures here), fish UNCONDITIONALLY —
/// active or commissioning-inactive alike. A commissioning-inactive delivery
/// was never latched (its decision column stays dormant `[0, 0]`,
/// `fill_anticipated_columns`), so its `in_col` carries `0` and this equality
/// pins that stage's thermal generation to `0` — the correct, harmless
/// outcome for a delivery the plant's window cannot receive. Carrying it
/// instead (the retired alternative) collided with the SAME stage's fresh
/// latch for the next delivery sharing the same modular residue whenever a
/// plant's own lead defines `k_max` (no other anticipated plant reaches
/// deeper): two definition rows on one `out_col` pinned a freshly-costed
/// decision to a stale carried value, producing a false `Infeasible` or a
/// silent zero-commit release-mode (`debug_assert` compiled out). Fishing
/// reads only `in_col` and never writes `out_col`, so it cannot collide with
/// the latch — sums per-block thermal energy against the maturing slot's
/// committed power scaled to `MWh` (`MW × block_hours`), preserving the
/// pre-migration `+h_b`/`−H` coefficient shape exactly; only its slot
/// addressing is modular (`stage_idx mod k_max`, via
/// [`crate::indexer::StateSpace::commitment_hold_in_study_offset`]).
/// `stage_idx` is in-study by construction
/// (`build_anticipated_fishing_row_pos` returns an empty mapping once
/// `stage_idx >= n_stages`), where [`crate::lead_time::PointResolution::ring_index`] is the
/// identity — so this slot needs no excision; applying one here would
/// double-shift a slot the plant already owns.
pub(super) fn fill_anticipated_fishing_entries(
    ctx: &TemplateBuildCtx<'_>,
    stage: &Stage,
    stage_idx: usize,
    layout: &StageLayout,
    col_entries: &mut [Vec<(usize, f64)>],
) {
    let n_blks = layout.clock.n_blks();
    let ring = DeliveryRing::anticipated(layout.state);
    let mut n_active = 0_usize;
    for local_idx in 0..ctx.study_dims.anticipated_plants.len() {
        let local = AnticipatedLocal::new(local_idx);
        let Some(row) = layout.anticipated_fishing_row(local) else {
            continue;
        };
        // Reachable only because `build_anticipated_fishing_row_pos` gates
        // this position on `k_max >= 1`, not `n_anticipated >= 1` alone —
        // `is_anticipated_at` is `true` for a pre-study (`None`) decider, so
        // an `n_anticipated`-only gate would reach this modulo on an empty
        // ring.
        let slot = stage_idx % layout.state.k_max;
        let thermal_idx = ctx.study_dims.anticipated_plants.thermal_of(local);
        let mut block_hours_total: f64 = 0.0;
        for blk in 0..n_blks {
            let col_gen = layout.geometry.thermal_col(thermal_idx, BlockIdx::new(blk));
            let block_hours = stage.blocks[blk].duration_hours;
            col_entries[col_gen].push((row, block_hours));
            block_hours_total += block_hours;
        }
        let col_state = ring.in_col(slot, local_idx);
        col_entries[col_state].push((row, -block_hours_total));
        n_active += 1;
    }
    debug_assert_eq!(
        n_active,
        layout.anticipated.fishing_rows.len(),
        "fill_anticipated_fishing_entries: active count mismatch"
    );
}

/// Encode the commitment-hold ring's delivery-decision LATCH row
/// `slot^out − decision_col = 0` for each plant with a genuine, active
/// decision this stage (`anticipated_decision_row_pos`). The decision's own
/// delivery target is latched into its ring slot through
/// [`for_each_ring_residue`], which owns the delivery-axis → ring-slot map;
/// the active-count assert below covers the case a genuine target were ever
/// excised (never latched, so short by one).
pub(super) fn fill_anticipated_state_out_def_entries(
    stage_idx: usize,
    layout: &StageLayout,
    col_entries: &mut [Vec<(usize, f64)>],
) {
    let ring = DeliveryRing::anticipated(layout.state);
    let mut n_active: usize = 0;
    for_each_ring_residue(layout.state, stage_idx, |res, point| {
        let Some(delivery_stage) = point.genuine_decisions_at(stage_idx).next() else {
            return;
        };
        // The walker visits every window residue for every plant; latch only at
        // the residue this plant's own genuine decision matures into.
        if delivery_stage != res.target {
            return;
        }
        let local = AnticipatedLocal::new(res.plant);
        let Some(row) = layout.anticipated_state_out_def_row(local) else {
            return;
        };
        debug_assert!(
            delivery_stage > stage_idx,
            "a genuine decision's delivery stage must be strictly after the decision \
             stage (K=0 self-delivery must already be excluded)"
        );
        let col_decision = layout.geometry.anticipated_decision_col(local);
        ring.emit_deposit(res.slot, res.plant, row, col_decision, col_entries);
        n_active += 1;
    });
    debug_assert_eq!(
        n_active,
        layout.anticipated.state_out_def_rows.len(),
        "fill_anticipated_state_out_def_entries: active count mismatch at stage {stage_idx}"
    );
}

/// Encode the future-window commitment-carry rows (same-slot hold,
/// `slot^out − slot^in = 0`) via [`DeliveryRing::emit_carry_rows`] — the
/// same-slot hold identity replacing the retired Markov-1
/// [`DeliveryRing::emit_shift_rows`] for the anticipated family (the water
/// travel-time ring keeps `emit_shift_rows` unchanged; its physics genuinely
/// shift). `anticipated_slot_row_pos` carries every STRICTLY FUTURE,
/// not-yet-due in-flight slot; the commitment maturing THIS stage is always
/// fished ([`fill_anticipated_fishing_entries`], never carried), so this
/// family never duplicates it. A slot with no row here is beyond the study
/// horizon, not yet ready, or handled by the latch/maturity rows.
fn fill_anticipated_slot_definition_entries(
    layout: &StageLayout,
    col_entries: &mut [Vec<(usize, f64)>],
) {
    let ring = DeliveryRing::anticipated(layout.state);
    let n_reachable = ring.emit_carry_rows(
        &layout.anticipated.anticipated_slot_row_pos,
        layout.anticipated.slot_definition_rows.start,
        col_entries,
    );
    debug_assert_eq!(
        n_reachable,
        layout.anticipated.slot_definition_rows.len(),
        "fill_anticipated_slot_definition_entries: reachable-slot count must match \
         slot_definition_rows"
    );
}

/// Fill water-balance row entries. Incoming state is pinned via column bounds, so no
/// row-equality state-fixing diagonals are written here.
///
/// A `PreFilling` hydro's row collapses to the frozen-storage identity `v_h − v_h_in = 0`,
/// its water interactions routed to the first non-`PreFilling` downstream by
/// [`fill_prefilling_shortcircuit`] (the contract home). The forbidden alternative —
/// zeroing `h`'s flow columns (turbine/spillage/diversion) while leaving its inflow on
/// this row untouched — traps the water and makes the LP infeasible whenever the site
/// has inflow.
///
/// In `BlockMode::Chronological` the single row becomes `K` chained rows
/// ([`fill_chronological_water_entries`]) that telescope to this parallel row, so
/// `K = 1` is byte-identical to parallel.
pub(super) fn fill_state_and_water_entries(
    ctx: &TemplateBuildCtx<'_>,
    stage: &Stage,
    stage_idx: usize,
    layout: &StageLayout,
    col_entries: &mut [Vec<(usize, f64)>],
) {
    // Mode-independent: the bucket ring-shift never depends on block_mode, so it runs
    // once outside the per-mode match.
    fill_transit_bucket_definition_entries(layout, col_entries);

    match stage.block_mode {
        BlockMode::Parallel => {
            fill_parallel_water_entries(ctx, stage, stage_idx, layout, col_entries);
        }
        BlockMode::Chronological => {
            fill_chronological_water_entries(ctx, stage, stage_idx, layout, col_entries);
        }
    }
}

/// Parallel single-row water-balance fill: one equality row per hydro summing all
/// blocks, with per-block flow terms scaled by `τ_k` and the inflow/AR-lag/
/// evaporation/withdrawal families scaled once by the stage total `ζ = Σ_k τ_k`.
fn fill_parallel_water_entries(
    ctx: &TemplateBuildCtx<'_>,
    stage: &Stage,
    stage_idx: usize,
    layout: &StageLayout,
    col_entries: &mut [Vec<(usize, f64)>],
) {
    let geom = &layout.geometry;
    let n_h = layout.state.hydro_count;
    let n_blks = layout.clock.n_blks();
    let zeta = layout.clock.zeta();

    for h_idx in 0..n_h {
        let hydro = &ctx.hydros[h_idx];
        let row = geom.water_balance_row(HydroSys::new(h_idx), BlockIdx::new(0));
        let storage_out_col = layout
            .state
            .storage_outgoing_col(HydroSys::new(h_idx))
            .get();
        let storage_in_col = layout
            .state
            .storage_incoming_col(HydroSys::new(h_idx))
            .get();

        col_entries[storage_out_col].push((row, 1.0));
        col_entries[storage_in_col].push((row, -1.0));

        if matches!(hydro_phase(hydro, stage.id), Phase::PreFilling) {
            // Frozen-storage identity `v_h − v_h_in = 0`: emit ONLY the two storage entries
            // above. Any inflow/upstream/AR-lag/withdrawal/evaporation coupling left here
            // makes `β_h` stale-nonzero — a wrong cut that still compiles.
            fill_prefilling_shortcircuit(ctx, stage, stage_idx, h_idx, layout, col_entries);
            continue;
        }

        push_maturing_bucket_coupling(ctx, stage, stage_idx, layout, h_idx, h_idx, col_entries);

        for blk in 0..n_blks {
            let tau_h = layout.clock.tau(BlockIdx::new(blk));
            for c in ctx.hydro_cell_index.cells_of(HydroSys::new(h_idx)) {
                let col_turbine = geom.turbine_col(HydroCell::new(c), BlockIdx::new(blk));
                col_entries[col_turbine].push((row, tau_h));
            }
            let col_spillage = geom.spillage_col(HydroSys::new(h_idx), BlockIdx::new(blk));
            col_entries[col_spillage].push((row, tau_h));
            let col_diversion = geom.diversion_col(HydroSys::new(h_idx), BlockIdx::new(blk));
            col_entries[col_diversion].push((row, tau_h));
            for &up_id in ctx.cascade.upstream(hydro.id) {
                if let Some(u_idx) = ctx.positions.hydro(up_id) {
                    fill_arc_release_block_entries(
                        ctx,
                        layout,
                        u_idx,
                        h_idx,
                        stage_idx,
                        blk,
                        tau_h,
                        row,
                        col_entries,
                    );
                }
            }
            if let Some(sources) = ctx.diversion_upstream.get(&hydro.id) {
                for &d_idx in sources {
                    let col_div = geom.diversion_col(HydroSys::new(d_idx), BlockIdx::new(blk));
                    col_entries[col_div].push((row, -tau_h));
                }
            }
        }
        push_z_inflow_coupling(stage, layout, h_idx, h_idx, col_entries);
    }

    // The PreFilling `continue` below keeps the frozen identity row free of slack/flow
    // terms (the contract above); `evap_hydro_indices` already excludes PreFilling hydros.
    for h_idx in 0..n_h {
        if matches!(hydro_phase(&ctx.hydros[h_idx], stage.id), Phase::PreFilling) {
            continue;
        }
        let row = geom.water_balance_row(HydroSys::new(h_idx), BlockIdx::new(0));
        if !geom.inflow_slack.is_empty() {
            col_entries[geom.inflow_slack_col(HydroSys::new(h_idx))].push((row, -zeta));
        }
        col_entries[geom.withdrawal_slack_neg_col(HydroSys::new(h_idx))].push((row, -zeta));
        col_entries[geom.withdrawal_slack_pos_col(HydroSys::new(h_idx))].push((row, zeta));
    }

    for (local_idx, &h) in geom.evap_hydro_indices.iter().enumerate() {
        let col_evaporation_flow =
            layout.evap_flow_col(EvapLocal::new(local_idx), BlockIdx::new(0));
        let row = geom.water_balance_row(h, BlockIdx::new(0));
        col_entries[col_evaporation_flow].push((row, zeta));
    }
}

/// Fill the travel-time bucket-definition ring-shift rows via
/// [`DeliveryRing::emit_shift_rows`], once per downstream plant. A masked-out bucket
/// gets no row — its outgoing column is frozen `[0, 0]` by `fill_transit_bucket_columns`
/// (the two-sided masking contract). Mode-independent (buckets are stage-level), so it
/// runs once per stage; the deposit terms are emitted separately by
/// [`fill_arc_release_block_entries`]. A no-op when `state.n_buckets == 0`.
fn fill_transit_bucket_definition_entries(
    layout: &StageLayout,
    col_entries: &mut [Vec<(usize, f64)>],
) {
    let row_start = layout.rows.transit_bucket_definition.start;
    for bucket in DeliveryRing::transit_buckets(layout.state) {
        bucket.ring.emit_shift_rows(
            &layout.rows.transit_bucket_row_pos[bucket.local],
            row_start,
            col_entries,
        );
    }
}

/// Push a plant's release onto `row`: every cell's turbine column plus the plant's
/// single spillage column, all at the SAME `coeff`. A plant's release is `Σ_c q_c + s`
/// over a disjoint cell partition, so a per-arc coefficient is REPLICATED across the
/// cells, never divided by cell count (k-factor conservation).
fn push_plant_release(
    ctx: &TemplateBuildCtx<'_>,
    layout: &StageLayout,
    u_idx: usize,
    blk: usize,
    row: usize,
    coeff: f64,
    col_entries: &mut [Vec<(usize, f64)>],
) {
    for c in ctx.hydro_cell_index.cells_of(HydroSys::new(u_idx)) {
        col_entries[layout
            .geometry
            .turbine_col(HydroCell::new(c), BlockIdx::new(blk))]
        .push((row, coeff));
    }
    col_entries[layout
        .geometry
        .spillage_col(HydroSys::new(u_idx), BlockIdx::new(blk))]
    .push((row, coeff));
}

/// One upstream release's per-block contribution to the downstream water balance
/// (arc `u_idx → h_idx`), split by the arc's resolved stage-clock weight `k`: same-stage
/// share `-k_0·τ_blk` on the balance row, and, for a multi-lag arc, deposits `-k_d·τ_blk`
/// (`d = 1..=depth`) into the plant's bucket-definition rows. The SAME release column
/// carries `k_0` on the balance row and `k_1..k_d` into the definition rows — never the
/// once-per-stage `ζ`-family.
///
/// `ctx.topology.arc_stage_weights` has no entry for an undeclared arc, so this emits exactly
/// today's `-τ_blk` and no deposit (the B==0 byte-identity anchor).
fn fill_arc_release_block_entries(
    ctx: &TemplateBuildCtx<'_>,
    layout: &StageLayout,
    u_idx: usize,
    h_idx: usize,
    stage_idx: usize,
    blk: usize,
    tau_h: f64,
    row_balance: usize,
    col_entries: &mut [Vec<(usize, f64)>],
) {
    let Some(stage_weights) = ctx
        .topology
        .arc_stage_weights
        .get(&u_idx)
        .map(|k_by_stage| &k_by_stage[stage_idx])
    else {
        push_plant_release(ctx, layout, u_idx, blk, row_balance, -tau_h, col_entries);
        return;
    };

    debug_assert!(
        (stage_weights.iter().sum::<f64>() - 1.0).abs() < 1e-9,
        "arc {u_idx} -> {h_idx} stage {stage_idx}: stage-clock weights must sum to 1.0, got \
         {stage_weights:?}"
    );

    if stage_weights[0] != 0.0 {
        push_plant_release(
            ctx,
            layout,
            u_idx,
            blk,
            row_balance,
            -stage_weights[0] * tau_h,
            col_entries,
        );
    }

    let depth = stage_weights.len() - 1;
    if depth == 0 {
        return;
    }
    let bucket =
        DeliveryRing::transit_bucket(layout.state, HydroSys::new(h_idx)).unwrap_or_else(|| {
            unreachable!(
                "hydro {h_idx} receives a depth-{depth} deposit at stage {stage_idx} but has no \
             bucket range (TransitBucketTopology/arc_stage_weights disagreement)"
            )
        });
    for (d, &stage_weight) in stage_weights.iter().enumerate().skip(1) {
        if stage_weight == 0.0 {
            continue;
        }
        let Some(row_def) =
            layout.transit_bucket_definition_row(&bucket.local, bucket.ring.slot_target(0, d))
        else {
            // A dropped lag targets only a stage past the horizon, unreachable once
            // `boundary_present` un-caps the mask (sddp.md "Terminal credit deferred").
            debug_assert!(
                ctx.topology
                    .arc_stage_weights
                    .get(&u_idx)
                    .is_some_and(|k_by_stage| stage_idx + d >= k_by_stage.len()),
                "arc {u_idx} -> {h_idx} stage {stage_idx}: lag-{d} deposit targeting inside the \
                 horizon must have a bucket-definition row"
            );
            continue;
        };
        push_plant_release(
            ctx,
            layout,
            u_idx,
            blk,
            row_def,
            -stage_weight * tau_h,
            col_entries,
        );
    }
}

/// Chronological per-block water-balance fill: each Operating/Filling hydro emits `K`
/// chained rows (block-major `row_water + h·K + (k−1)`), each the parallel row per block
/// with `τ_k` replacing the stage total `ζ` EVERYWHERE. A stray `ζ` double-applies
/// (`Σ_k τ_k = ζ`) and breaks the telescoping identity that recovers the parallel row.
/// The inflow STATE (`z_inflow` and its definition row) stays stage-level.
///
/// A `PreFilling` hydro emits `K` per-block frozen identities `Sᵏ − Sᵏ⁻¹ = 0` (any
/// coupling left on a frozen row makes `β_h` stale-nonzero — a wrong cut that compiles),
/// short-circuiting per block via [`fill_prefilling_shortcircuit`], maturing bucket
/// included.
fn fill_chronological_water_entries(
    ctx: &TemplateBuildCtx<'_>,
    stage: &Stage,
    stage_idx: usize,
    layout: &StageLayout,
    col_entries: &mut [Vec<(usize, f64)>],
) {
    let geom = &layout.geometry;
    let n_h = layout.state.hydro_count;
    let n_blks = layout.clock.n_blks();

    for h_idx in 0..n_h {
        let hydro = &ctx.hydros[h_idx];

        if matches!(hydro_phase(hydro, stage.id), Phase::PreFilling) {
            for k in 1..=n_blks {
                let row = geom.water_balance_row(HydroSys::new(h_idx), BlockIdx::new(k - 1));
                col_entries[layout
                    .block_storage_col(HydroSys::new(h_idx), Boundary::from_index(k, n_blks))]
                .push((row, 1.0));
                col_entries[layout
                    .block_storage_col(HydroSys::new(h_idx), Boundary::from_index(k - 1, n_blks))]
                .push((row, -1.0));
            }
            fill_prefilling_shortcircuit(ctx, stage, stage_idx, h_idx, layout, col_entries);
            continue;
        }

        push_maturing_bucket_coupling(ctx, stage, stage_idx, layout, h_idx, h_idx, col_entries);
        push_z_inflow_coupling(stage, layout, h_idx, h_idx, col_entries);

        for k in 1..=n_blks {
            let blk = k - 1;
            let row = geom.water_balance_row(HydroSys::new(h_idx), BlockIdx::new(blk));
            let tau_k = layout.clock.tau(BlockIdx::new(blk));

            col_entries
                [layout.block_storage_col(HydroSys::new(h_idx), Boundary::from_index(k, n_blks))]
            .push((row, 1.0));
            col_entries[layout
                .block_storage_col(HydroSys::new(h_idx), Boundary::from_index(k - 1, n_blks))]
            .push((row, -1.0));

            for c in ctx.hydro_cell_index.cells_of(HydroSys::new(h_idx)) {
                col_entries[geom.turbine_col(HydroCell::new(c), BlockIdx::new(blk))]
                    .push((row, tau_k));
            }
            col_entries[geom.spillage_col(HydroSys::new(h_idx), BlockIdx::new(blk))]
                .push((row, tau_k));
            col_entries[geom.diversion_col(HydroSys::new(h_idx), BlockIdx::new(blk))]
                .push((row, tau_k));
            for &up_id in ctx.cascade.upstream(hydro.id) {
                if let Some(u_idx) = ctx.positions.hydro(up_id) {
                    fill_arc_release_chrono_block_entries(
                        ctx,
                        layout,
                        u_idx,
                        h_idx,
                        stage_idx,
                        blk,
                        col_entries,
                    );
                }
            }
            if let Some(sources) = ctx.diversion_upstream.get(&hydro.id) {
                for &d_idx in sources {
                    col_entries[geom.diversion_col(HydroSys::new(d_idx), BlockIdx::new(blk))]
                        .push((row, -tau_k));
                }
            }

            if !geom.inflow_slack.is_empty() {
                col_entries[geom.inflow_slack_col(HydroSys::new(h_idx))].push((row, -tau_k));
            }
            col_entries[geom.withdrawal_slack_neg_col(HydroSys::new(h_idx))].push((row, -tau_k));
            col_entries[geom.withdrawal_slack_pos_col(HydroSys::new(h_idx))].push((row, tau_k));
        }
    }

    for (local_idx, &h) in geom.evap_hydro_indices.iter().enumerate() {
        let local_idx = EvapLocal::new(local_idx);
        for k in 1..=n_blks {
            let blk = k - 1;
            let tau_k = layout.clock.tau(BlockIdx::new(blk));
            let row = geom.water_balance_row(h, BlockIdx::new(blk));
            col_entries[layout.evap_flow_col(local_idx, BlockIdx::new(blk))].push((row, tau_k));
        }
    }
}

/// One upstream release's per-block contribution to the downstream chained rows
/// (arc `u_idx → h_idx`, block `blk`): same-stage routing `-κ·τ_blk` onto downstream
/// blocks and crossing deposits `-χ·τ_blk` into the plant's bucket-definition rows — the
/// SAME release column carrying both. Verifies the shared-density aggregation identity
/// `Σ_b w_b·χ_{b,d} == k_d` once per (arc, stage) (`blk == 0`).
///
/// `arc_spread_chrono` has no entry for an undeclared arc (or a `Parallel`-mode stage),
/// so this emits today's `-τ_blk` and no routing/deposit — the B==0/K==1 byte-identity
/// anchor.
fn fill_arc_release_chrono_block_entries(
    ctx: &TemplateBuildCtx<'_>,
    layout: &StageLayout,
    u_idx: usize,
    h_idx: usize,
    stage_idx: usize,
    blk: usize,
    col_entries: &mut [Vec<(usize, f64)>],
) {
    let tau_k = layout.clock.tau(BlockIdx::new(blk));

    let Some(resolution) = ctx
        .topology
        .arc_spread_chrono
        .get(&u_idx)
        .and_then(|by_stage| by_stage[stage_idx].as_ref())
    else {
        let row = layout
            .geometry
            .water_balance_row(HydroSys::new(h_idx), BlockIdx::new(blk));
        push_plant_release(ctx, layout, u_idx, blk, row, -tau_k, col_entries);
        return;
    };

    let block_routing = &resolution.within_stage_routing[blk];
    let block_deposit = &resolution.block_deposits[blk];
    debug_assert!(
        (block_routing.iter().sum::<f64>() + block_deposit[1..].iter().sum::<f64>() - 1.0).abs()
            < 1e-9,
        "arc {u_idx} block {blk} stage {stage_idx}: per-column conservation \
         sum(within_stage_routing) + sum(block_deposits[1..]) must equal 1.0"
    );
    if blk == 0 {
        debug_assert!(
            (resolution.stage_weights.iter().sum::<f64>() - 1.0).abs() < 1e-9,
            "arc {u_idx} stage {stage_idx}: stage-clock weights must sum to 1.0, got {:?}",
            resolution.stage_weights
        );
        for (d, &stage_weight) in resolution.stage_weights.iter().enumerate() {
            let aggregated: f64 = resolution
                .block_deposits
                .iter()
                .zip((0..layout.clock.n_blks()).map(BlockIdx::new))
                .map(|(deposit_row, k)| {
                    (layout.clock.tau(k) / layout.clock.zeta()) * deposit_row[d]
                })
                .sum();
            debug_assert!(
                (aggregated - stage_weight).abs() < 1e-9,
                "arc {u_idx} stage {stage_idx}: block deposits must aggregate to k_d (d={d})"
            );
        }
    }

    for (j, &routing_val) in block_routing.iter().enumerate() {
        if routing_val == 0.0 {
            continue;
        }
        let row = layout
            .geometry
            .water_balance_row(HydroSys::new(h_idx), BlockIdx::new(blk + j));
        push_plant_release(
            ctx,
            layout,
            u_idx,
            blk,
            row,
            -routing_val * tau_k,
            col_entries,
        );
    }

    let depth = block_deposit.len() - 1;
    if depth == 0 {
        return;
    }
    let bucket =
        DeliveryRing::transit_bucket(layout.state, HydroSys::new(h_idx)).unwrap_or_else(|| {
            unreachable!(
                "hydro {h_idx} receives a depth-{depth} deposit at stage {stage_idx} but has no \
             bucket range (TransitBucketTopology/arc_spread_chrono disagreement)"
            )
        });
    for (d, &deposit_d) in block_deposit.iter().enumerate().skip(1) {
        if deposit_d == 0.0 {
            continue;
        }
        let Some(row_def) =
            layout.transit_bucket_definition_row(&bucket.local, bucket.ring.slot_target(0, d))
        else {
            // A dropped block deposit targets only a lag past the horizon, unreachable
            // once `boundary_present` un-caps the mask (sddp.md "Terminal credit deferred").
            debug_assert!(
                ctx.topology
                    .arc_spread_chrono
                    .get(&u_idx)
                    .is_some_and(|by_stage| stage_idx + d >= by_stage.len()),
                "arc {u_idx} -> {h_idx} stage {stage_idx}: lag-{d} block deposit targeting \
                 inside the horizon must have a bucket-definition row"
            );
            continue;
        };
        push_plant_release(
            ctx,
            layout,
            u_idx,
            blk,
            row_def,
            -deposit_d * tau_k,
            col_entries,
        );
    }
}

/// Couple `z_h` (hydro `h_idx`'s realized-inflow column) onto `target_idx`'s water
/// row(s): the hydro's own route (`target_idx == h_idx`) or the
/// [`fill_prefilling_shortcircuit`] route (`target_idx == d_idx`). Emitted for every
/// non-`PreFilling` hydro regardless of a PAR model (`z_h` is then pinned to `0` by
/// its own row).
///
/// Parallel pushes the single stage-total `−ζ` (a `Σ_k −τ_k` loop would inflate
/// `z_h`'s routed-entry count, pinned by
/// `prefilling_upstream_inflow_lands_on_balance_row_only`); Chronological splits into
/// per-block `−τ_k` (`Σ_k τ_k = ζ`).
fn push_z_inflow_coupling(
    stage: &Stage,
    layout: &StageLayout,
    h_idx: usize,
    target_idx: usize,
    col_entries: &mut [Vec<(usize, f64)>],
) {
    let z_h = layout.state.z_inflow_col(HydroSys::new(h_idx)).get();
    let target = HydroSys::new(target_idx);
    for blk in 0..layout
        .geometry
        .water_balance
        .rows_per_entity(layout.clock.n_blks())
    {
        let coeff = match stage.block_mode {
            BlockMode::Parallel => -layout.clock.zeta(),
            BlockMode::Chronological => -layout.clock.tau(BlockIdx::new(blk)),
        };
        col_entries[z_h].push((
            layout
                .geometry
                .water_balance_row(target, BlockIdx::new(blk)),
            coeff,
        ));
    }
}

/// Couple hydro `h_idx`'s maturing bucket `b_1^in` onto `target_idx`'s water row(s):
/// the hydro's own route (`target_idx == h_idx`) or the [`fill_prefilling_shortcircuit`]
/// route (`target_idx == d_idx`). A no-op when `h_idx` declares no incoming arc.
///
/// Parallel pushes a single `−1.0` and never reads the density: the confluence sum over
/// every upstream arc lives in the state variable itself. Chronological spreads it over
/// the target's block rows by `h_idx`'s fixed `arrival_density` (fixed-delivery-density
/// contract).
fn push_maturing_bucket_coupling(
    ctx: &TemplateBuildCtx<'_>,
    stage: &Stage,
    stage_idx: usize,
    layout: &StageLayout,
    h_idx: usize,
    target_idx: usize,
    col_entries: &mut [Vec<(usize, f64)>],
) {
    let Some(col) = maturing_bucket_in_col(layout.state, HydroSys::new(h_idx)) else {
        return;
    };
    let target = HydroSys::new(target_idx);
    match stage.block_mode {
        BlockMode::Parallel => {
            let row = layout.geometry.water_balance_row(target, BlockIdx::new(0));
            col_entries[col].push((row, -1.0));
        }
        BlockMode::Chronological => {
            let n_blks = layout.clock.n_blks();
            let arrival_density = resolve_bucket_arrival_density(
                ctx,
                layout.clock,
                stage_idx,
                ctx.hydros[h_idx].id,
                n_blks,
            );
            debug_assert!(
                (arrival_density.iter().sum::<f64>() - 1.0).abs() < 1e-9,
                "hydro {h_idx} stage {stage_idx}: arrival_density must sum to 1.0"
            );
            for (k, &rho_k) in arrival_density.iter().enumerate() {
                if rho_k == 0.0 {
                    continue;
                }
                let row = layout.geometry.water_balance_row(target, BlockIdx::new(k));
                col_entries[col].push((row, -rho_k));
            }
        }
    }
}

/// Re-route an absent `PreFilling` hydro `h`'s water interactions onto the FIRST
/// non-`PreFilling` downstream hydro `d` ([`resolve_shortcircuit_target`]): in `Parallel`
/// onto `d`'s single row, in `Chronological` onto `d`'s block rows with the stage-total
/// `−ζ` on `z_h` split into `−τ_k` per block (`Σ_k τ_k = ζ`).
///
/// Route the REAL `ζ`/`τ_k` columns, never a synthesized coefficient: with the real
/// columns on `d`'s row, `rc / col_scale` of `d`'s pinned incoming-storage column is a
/// valid subgradient; a synthesized coefficient breaks that duality and produces an
/// invalid cut.
///
/// `z_h`'s own definition row and noise patch are left UNTOUCHED for `h`, so the routed
/// column is scenario-exact; the deterministic base rides on `z_h` and must NOT also be
/// added to `d`'s RHS (double-count). `h`'s withdrawal demand transfers via `d`'s RHS
/// (owned by `super::rows::fill_water_balance_rows`), which MUST resolve the SAME `d` so
/// matrix and RHS agree. A skipped intermediate `PreFilling` hydro contributes zero
/// releases, so a chain re-routes each link's inflow to `d` exactly once. Sink case (no
/// non-`PreFilling` downstream): nothing routed, no panic.
///
/// The `z_h` and maturing-bucket couplings are [`push_z_inflow_coupling`] and
/// [`push_maturing_bucket_coupling`], shared with both water writers' own-row route.
fn fill_prefilling_shortcircuit(
    ctx: &TemplateBuildCtx<'_>,
    stage: &Stage,
    stage_idx: usize,
    h_idx: usize,
    layout: &StageLayout,
    col_entries: &mut [Vec<(usize, f64)>],
) {
    let hydro = &ctx.hydros[h_idx];
    let Some(d_idx) =
        resolve_shortcircuit_target(ctx.hydros, ctx.cascade, ctx.positions, stage.id, h_idx)
    else {
        return;
    };
    let n_blks = layout.clock.n_blks();
    let target = HydroSys::new(d_idx);

    push_z_inflow_coupling(stage, layout, h_idx, d_idx, col_entries);
    push_maturing_bucket_coupling(ctx, stage, stage_idx, layout, h_idx, d_idx, col_entries);

    for blk in 0..n_blks {
        let tau_k = layout.clock.tau(BlockIdx::new(blk));
        let row_d = layout
            .geometry
            .water_balance_row(target, BlockIdx::new(blk));
        for &up_id in ctx.cascade.upstream(hydro.id) {
            if let Some(u_idx) = ctx.positions.hydro(up_id) {
                push_plant_release(ctx, layout, u_idx, blk, row_d, -tau_k, col_entries);
            }
        }
        if let Some(sources) = ctx.diversion_upstream.get(&hydro.id) {
            for &src_idx in sources {
                col_entries[layout
                    .geometry
                    .diversion_col(HydroSys::new(src_idx), BlockIdx::new(blk))]
                .push((row_d, -tau_k));
            }
        }
    }
}

/// Fill the LHS of the per-stage soft filling-target row `v_h + σ_fill ≥ V_target[t]`
/// for each Filling-phase hydro: `+1.0` on the outgoing storage column `v_h` and `+1.0`
/// on the `σ_fill` slack. The `≥` sense and RHS are set by
/// `rows::fill_filling_target_rows`.
///
/// Cut validity: LP duality folds the `σ_fill` soft-row dual into the incoming-storage
/// column's `rc / col_scale`. NEVER separately extract this row's dual and add it by hand
/// — that double-counts the soft floor (a guard test asserts no `lp/builder` file
/// references the dual-extraction entry point).
fn fill_filling_target_entries(layout: &StageLayout, col_entries: &mut [Vec<(usize, f64)>]) {
    for (local_idx, &h) in layout
        .geometry
        .filling_target_hydro_indices
        .iter()
        .enumerate()
    {
        let local = FillingTargetLocal::new(local_idx);
        let row = layout.filling_target_row(local);
        col_entries[layout.state.storage_outgoing_col(h).get()].push((row, 1.0));
        col_entries[layout.geometry.filling_target_slack_col(local)].push((row, 1.0));
    }
}

/// Fill the LHS of the soft operating-floor row `v_h + σ^{v-} ≥ min_storage_hm3` for
/// each Operating-phase filling hydro: `+1.0` on `v_h` and `+1.0` on the `σ^{v-}` slack.
/// The `≥` sense and RHS are set by `rows::fill_filled_min_storage_floor_rows`.
///
/// Same cut-validity contract as [`fill_filling_target_entries`]: never hand-extract this
/// row's dual. DISTINCT from that sibling — different slack, non-overlapping stage scope
/// (Operating vs Filling), different RHS and cost; never conflate them.
fn fill_filled_min_storage_floor_entries(
    layout: &StageLayout,
    col_entries: &mut [Vec<(usize, f64)>],
) {
    for (local_idx, &h) in layout
        .geometry
        .filled_min_storage_floor_hydro_indices
        .iter()
        .enumerate()
    {
        let local = FloorLocal::new(local_idx);
        let row = layout.filled_min_storage_floor_row(local);
        col_entries[layout.state.storage_outgoing_col(h).get()].push((row, 1.0));
        col_entries[layout.geometry.filled_min_storage_floor_slack_col(local)].push((row, 1.0));
    }
}

/// Fill pumping-flow water-balance entries: per block, the pumped-flow column enters
/// that block's water row of the SOURCE hydro with `+tau_h` (outflow sign) and of the
/// DESTINATION hydro with `−tau_h` (inflow sign). `tau_h` is the same [`BlockClock::tau`](crate::block_clock::BlockClock::tau)
/// value turbine/spillage use, so the coefficient stays bit-identical across sites.
/// Structural entries are written for every station: a dormant station's column is `[0, 0]`.
pub(super) fn fill_pumping_water_entries(
    ctx: &TemplateBuildCtx<'_>,
    layout: &StageLayout,
    col_entries: &mut [Vec<(usize, f64)>],
) {
    let geom = &layout.geometry;
    let n_blks = layout.clock.n_blks();
    for (p_sys, station) in ctx.pumping_stations.iter().enumerate() {
        // Per-side guards are defense-in-depth (`validate_pumping_station_refs` guarantees
        // resolution on a production `System`). Do NOT promote to an unconditional
        // index/expect — a one-sided resolve writes a feasible-but-wrong half coupling.
        let source = ctx.positions.hydro(station.source_hydro_id);
        let destination = ctx.positions.hydro(station.destination_hydro_id);
        for blk in 0..n_blks {
            let blk_idx = BlockIdx::new(blk);
            let tau_h = layout.clock.tau(blk_idx);
            let col = geom.pumping_flow_col(PumpingSys::new(p_sys), blk_idx);
            if let Some(s_idx) = source {
                col_entries[col]
                    .push((geom.water_balance_row(HydroSys::new(s_idx), blk_idx), tau_h));
            }
            if let Some(d_idx) = destination {
                col_entries[col].push((
                    geom.water_balance_row(HydroSys::new(d_idx), blk_idx),
                    -tau_h,
                ));
            }
        }
    }
}

/// Fill load-balance entries for hydro/thermal generation, line flows, pumping power,
/// and deficit/excess slacks. Each hydro CELL credits its own bus
/// (`HydroCellIndex::bus_of`) — `Hydro` carries no bus of its own, only its unit
/// groups do: FPHA cells enter with `g_c` at `+1.0`; constant-productivity cells
/// with `rho * turbine_col(cell)`, the plant's shared `rho`; a commissioning-dormant
/// FPHA plant ([`StageProductionRole::Dormant`]) credits nothing — it has no
/// generation column and no productivity to price its frozen turbine column at.
///
/// Pumping power is a negative injection: the `pumping_flow` column enters with
/// `−consumption_mw_per_m3s` (no separate power column). A positive coefficient would
/// credit the bus for power the station consumes.
pub(super) fn fill_load_balance_entries(
    ctx: &TemplateBuildCtx<'_>,
    stage_idx: usize,
    layout: &StageLayout,
    col_entries: &mut [Vec<(usize, f64)>],
) {
    let geom = &layout.geometry;
    let n_blks = layout.clock.n_blks();

    for h_idx in 0..ctx.hydros.len() {
        let h_sys = HydroSys::new(h_idx);
        match layout.stage_production_role(ctx.production_models, h_idx, stage_idx) {
            StageProductionRole::Fpha(local_idx) => {
                let cell_base = layout.fpha_cell_local_start[local_idx.get()];
                for (offset, c) in ctx.hydro_cell_index.cells_of(h_sys).enumerate() {
                    let cell = HydroCell::new(c);
                    if let Some(b_idx) = ctx.positions.bus(ctx.hydro_cell_index.bus_of(cell)) {
                        let cell_local = FphaCellLocal::new(cell_base + offset);
                        for blk in (0..n_blks).map(BlockIdx::new) {
                            let row = geom.load_balance_row(BusSys::new(b_idx), blk);
                            let col = geom.generation_col(cell_local, blk);
                            col_entries[col].push((row, 1.0));
                        }
                    }
                }
            }
            StageProductionRole::Constant(rho) => {
                for c in ctx.hydro_cell_index.cells_of(h_sys) {
                    let cell = HydroCell::new(c);
                    if let Some(b_idx) = ctx.positions.bus(ctx.hydro_cell_index.bus_of(cell)) {
                        for blk in (0..n_blks).map(BlockIdx::new) {
                            let row = geom.load_balance_row(BusSys::new(b_idx), blk);
                            let col = geom.turbine_col(cell, blk);
                            col_entries[col].push((row, rho));
                        }
                    }
                }
            }
            StageProductionRole::Dormant => {}
        }
    }

    for (t_idx, thermal) in ctx.thermals.iter().enumerate() {
        if let Some(b_idx) = ctx.positions.bus(thermal.bus_id) {
            for blk in (0..n_blks).map(BlockIdx::new) {
                let row = geom.load_balance_row(BusSys::new(b_idx), blk);
                let col = geom.thermal_col(ThermalSys::new(t_idx), blk);
                col_entries[col].push((row, 1.0));
            }
        }
    }

    for (l_idx, line) in ctx.lines.iter().enumerate() {
        let src_idx = ctx.positions.bus(line.source_bus_id);
        let tgt_idx = ctx.positions.bus(line.target_bus_id);
        for blk in (0..n_blks).map(BlockIdx::new) {
            let col_fwd = geom.line_fwd_col(LineSys::new(l_idx), blk);
            let col_rev = geom.line_rev_col(LineSys::new(l_idx), blk);
            if let Some(tgt) = tgt_idx {
                let row = geom.load_balance_row(BusSys::new(tgt), blk);
                col_entries[col_fwd].push((row, 1.0));
                col_entries[col_rev].push((row, -1.0));
            }
            if let Some(src) = src_idx {
                let row = geom.load_balance_row(BusSys::new(src), blk);
                col_entries[col_fwd].push((row, -1.0));
                col_entries[col_rev].push((row, 1.0));
            }
        }
    }

    // Written for every station: a dormant station's pumping column is `[0, 0]`.
    for (p_sys, station) in ctx.pumping_stations.iter().enumerate() {
        if let Some(b_idx) = ctx.positions.bus(station.bus_id) {
            for blk in (0..n_blks).map(BlockIdx::new) {
                let row = geom.load_balance_row(BusSys::new(b_idx), blk);
                let col = geom.pumping_flow_col(PumpingSys::new(p_sys), blk);
                col_entries[col].push((row, -station.consumption_mw_per_m3s));
            }
        }
    }

    // Import injects into its bus (`+1`), export withdraws (`−1`) — INDEPENDENT of the
    // stored price sign; flipping it would make an export feed the bus. Written for every
    // contract: a dormant contract's column is `[0, 0]`.
    for (c_sys, contract) in ctx.contracts.iter().enumerate() {
        let (contract_type, family_slot) = contract_family_slot(ctx.contracts, c_sys);
        let sign = match contract_type {
            ContractType::Import => 1.0,
            ContractType::Export => -1.0,
        };
        if let Some(b_idx) = ctx.positions.bus(contract.bus_id) {
            for blk in (0..n_blks).map(BlockIdx::new) {
                let row = geom.load_balance_row(BusSys::new(b_idx), blk);
                let col = geom.contract_col(contract_type, family_slot, blk);
                col_entries[col].push((row, sign));
            }
        }
    }

    for (b_idx, bus) in ctx.buses.iter().enumerate() {
        for blk in (0..n_blks).map(BlockIdx::new) {
            let row = geom.load_balance_row(BusSys::new(b_idx), blk);
            for seg_idx in 0..bus.deficit_segments.len() {
                let col_def = layout.deficit_col(BusSys::new(b_idx), seg_idx, blk);
                col_entries[col_def].push((row, 1.0));
            }
            let col_exc = geom.excess_col(BusSys::new(b_idx), blk);
            col_entries[col_exc].push((row, -1.0));
        }
    }
}

/// Fill FPHA hyperplane constraint entries, one row per `(FPHA cell, block, plane)`,
/// implementing `g_c − σ_c·γᵥ/2·v − σ_c·γᵥ/2·v_in − γ_q·q_c − σ_c·γ_s·s ≤ σ_c·γ₀`
/// (`σ_c·γ₀` in the row upper bound set by `rows::fill_fpha_rows`). `σ_c`
/// apportions the plane's flow-independent part by the cell's share of the plant's
/// declared turbine capacity; `γ_q` stays unscaled on the cell's own flow `q_c`.
///
/// FPHA uses AVERAGE storage `(V_in + V_out)/2`, so `−σ_c·γᵥ/2` lands on BOTH the
/// outgoing- and incoming-storage columns. Putting it on `V_out` alone compiles and
/// passes single-plane tests but understates generation by the `V_in` head term —
/// the wrong-but-compiling alternative deterministic case D06 pins against.
///
/// In `BlockMode::Chronological` block `k` averages the block-local boundaries
/// `(Sᵏ⁻¹, Sᵏ)`; `K = 1` resolves both back to `(S⁰, Sᴷ)`, byte-identical to parallel.
///
/// Driven by [`for_each_fpha_plane`] so entries and the row bounds set by
/// `rows::fill_fpha_rows` share one row cursor.
pub(super) fn fill_fpha_entries(
    ctx: &TemplateBuildCtx<'_>,
    stage: &Stage,
    stage_idx: usize,
    layout: &StageLayout,
    col_entries: &mut [Vec<(usize, f64)>],
) {
    let geom = &layout.geometry;
    for_each_fpha_plane(ctx, stage_idx, layout, |visit, plane| {
        let (col_v_in, col_v) = match stage.block_mode {
            BlockMode::Parallel => (
                layout.block_storage_col(visit.plant, Boundary::Incoming),
                layout.block_storage_col(visit.plant, Boundary::Outgoing),
            ),
            BlockMode::Chronological => (
                layout.block_storage_col(
                    visit.plant,
                    Boundary::from_index(visit.blk.get(), layout.clock.n_blks()),
                ),
                layout.block_storage_col(
                    visit.plant,
                    Boundary::from_index(visit.blk.get() + 1, layout.clock.n_blks()),
                ),
            ),
        };
        let col_q = geom.turbine_col(visit.cell, visit.blk);
        let col_s = geom.spillage_col(visit.plant, visit.blk);
        let col_g = geom.generation_col(visit.cell_local, visit.blk);
        // Apportion the plane's flow-independent part by this cell's share of the
        // plant's declared turbine capacity; γ_q stays unscaled on the cell's own
        // flow (only `A ≡ γ₀ + γ_V·V̄ + γ_s·s` fails the homogeneity that makes
        // same-bus aggregation exact — the cell partition's whole safety argument).
        let sigma_c = ctx.hydro_cell_index.share_of(visit.cell);

        col_entries[col_g].push((visit.row, 1.0));
        // Average storage: −γᵥ/2 on BOTH storage columns, not one alone (D06),
        // each apportioned by sigma_c like the other flow-independent terms.
        col_entries[col_v_in].push((visit.row, sigma_c * (-plane.gamma_v / 2.0)));
        col_entries[col_v].push((visit.row, sigma_c * (-plane.gamma_v / 2.0)));
        col_entries[col_q].push((visit.row, -plane.gamma_q));
        col_entries[col_s].push((visit.row, sigma_c * (-plane.gamma_s)));
    });
}

/// Fill the evaporation equality rows, one per `(evaporation hydro, slot)`, encoding
/// `evaporation_flow − slope/2·Sᵏ⁻¹ − slope/2·Sᵏ + f_plus − f_minus = intercept_m3s`
/// (`slope` = `volume_slope_m3s_per_hm3`; `intercept_m3s` set by `super::rows::fill_stage_rows`).
///
/// Like FPHA, `slope/2` lands on BOTH storage columns to average the block-local storage
/// `(Sᵏ⁻¹ + Sᵏ)/2`; chronological `K = 1` resolves both boundaries back to `(S⁰, Sᴷ)`,
/// byte-identical to parallel. A parallel stage has exactly one slot, on the stage
/// endpoints `(S⁰, Sᴷ)`; a chronological stage has one slot per block, each on that
/// block's own `(Sᵏ⁻¹, Sᵏ)`.
///
/// The evaporation flow's entry INTO the water-balance row lives with the water-balance
/// fill, not here.
pub(super) fn fill_evaporation_entries(
    ctx: &TemplateBuildCtx<'_>,
    stage: &Stage,
    stage_idx: usize,
    layout: &StageLayout,
    col_entries: &mut [Vec<(usize, f64)>],
) {
    let n_blks = layout.clock.n_blks();
    let n_evap_slots = layout.n_evap_slots;

    for (local_idx, &h) in layout.geometry.evap_hydro_indices.iter().enumerate() {
        let coeff = match ctx.evaporation_models.model(h.get()) {
            EvaporationModel::Linearized { coefficients, .. } => {
                debug_assert!(
                    stage_idx < coefficients.len(),
                    "evap_hydro_indices contains hydro {} but coefficients length {} \
                     is less than stage_idx {}",
                    h.get(),
                    coefficients.len(),
                    stage_idx
                );
                let Some(c) = coefficients.get(stage_idx) else {
                    continue;
                };
                *c
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

        let half_slope = coeff.volume_slope_m3s_per_hm3 / 2.0;
        for slot in 0..n_evap_slots {
            let (col_v_in, col_v) = match stage.block_mode {
                BlockMode::Parallel => (
                    layout.block_storage_col(h, Boundary::Incoming),
                    layout.block_storage_col(h, Boundary::Outgoing),
                ),
                BlockMode::Chronological => (
                    layout.block_storage_col(h, Boundary::from_index(slot, n_blks)),
                    layout.block_storage_col(h, Boundary::from_index(slot + 1, n_blks)),
                ),
            };
            let local = EvapLocal::new(local_idx);
            let col_evaporation_flow = layout.evap_flow_col(local, BlockIdx::new(slot));
            let col_f_plus = layout.evap_f_plus_col(local, BlockIdx::new(slot));
            let col_f_minus = layout.evap_f_minus_col(local, BlockIdx::new(slot));
            let row = layout.evap_row(local, BlockIdx::new(slot));

            col_entries[col_evaporation_flow].push((row, 1.0));
            col_entries[col_v_in].push((row, -half_slope));
            col_entries[col_v].push((row, -half_slope));
            col_entries[col_f_plus].push((row, 1.0));
            col_entries[col_f_minus].push((row, -1.0));
        }
    }
}

/// Mutable LP matrix buffers for stage template construction.
pub(super) struct LpMatrixBuffers<'a> {
    /// CSC column entries (column index -> list of (row, coefficient)).
    pub(super) col_entries: &'a mut [Vec<(usize, f64)>],
    pub(super) col_upper: &'a mut [f64],
    pub(super) objective: &'a mut [f64],
    pub(super) row_lower: &'a mut [f64],
    pub(super) row_upper: &'a mut [f64],
}

/// Fill matrix entries, row bounds, and slack columns for every active generic constraint
/// row at this stage, resolving each term via `resolve_variable_ref` and pricing enabled
/// slack at `penalty * block_hours`. Unknown entity IDs resolve to zero contributions
/// (the defense-in-depth fallback for referential-validation gaps).
pub(super) fn fill_generic_constraint_entries(
    ctx: &TemplateBuildCtx<'_>,
    stage_idx: usize,
    layout: &StageLayout,
    buffers: &mut LpMatrixBuffers<'_>,
) {
    if layout.rows.n_generic_rows == 0 {
        return;
    }
    let col_entries = &mut *buffers.col_entries;
    let col_upper = &mut *buffers.col_upper;
    let objective = &mut *buffers.objective;
    let row_lower = &mut *buffers.row_lower;
    let row_upper = &mut *buffers.row_upper;

    for (entry_idx, entry) in layout.generic_constraint_rows.iter().enumerate() {
        let row = layout.generic_row(entry_idx);
        let constraint = &ctx.generic_constraints[entry.constraint_idx];
        // A collapsed stage-level row is priced by the stage's total hours (it stands in
        // for one row per block); the total is penalty-conserving either way.
        let block_hours = if entry.is_stage_level {
            layout.clock.total_hours()
        } else {
            layout.clock.hours(BlockIdx::new(entry.block_idx))
        };

        // The interval IS the constraint: shape derives from the null-pattern.
        // A missing endpoint falls back to the unbounded direction (visibly open),
        // never a cross-fill from the other endpoint (would silently fabricate a
        // bound) — referential validation guarantees at least one is present.
        row_lower[row] = entry.bound_lower.unwrap_or(f64::NEG_INFINITY);
        row_upper[row] = entry.bound_upper.unwrap_or(f64::INFINITY);

        for term in &constraint.expression.terms {
            let pairs =
                resolve_variable_ref(&term.variable, entry.block_idx, stage_idx, ctx, layout);
            for (col, multiplier) in pairs {
                let coef = match term.coefficient {
                    CoefficientRef::Literal(v) => v,
                    CoefficientRef::Parameter(param_id) => {
                        ctx.resolved
                            .resolved_parameters
                            .get(param_id, stage_idx, entry.block_idx)
                    }
                };
                col_entries[col].push((row, coef * term.scale * multiplier));
            }
        }

        if let Some(plus_col) = entry.slack_plus_col {
            let penalty = constraint.slack.penalty.unwrap_or(0.0);
            let obj_coeff = penalty * block_hours;

            col_upper[plus_col] = f64::INFINITY;
            objective[plus_col] = obj_coeff;

            // Slack sign convention: `LHS - s_plus <= upper` (upper-only, relaxing
            // downward) or `LHS + s_plus >= lower` (lower-only or two-sided,
            // relaxing upward); a two-sided row additionally gets `s_minus` below,
            // relaxing the upper bound the same way `s_plus` relaxes the lower one.
            let plus_coeff = if entry.bound_upper.is_some() && entry.bound_lower.is_none() {
                -1.0
            } else {
                1.0
            };
            col_entries[plus_col].push((row, plus_coeff));

            if let Some(minus_col) = entry.slack_minus_col {
                col_upper[minus_col] = f64::INFINITY;
                objective[minus_col] = obj_coeff;
                col_entries[minus_col].push((row, -1.0));
            }
        }
    }
}

/// Inject `+1.0` for each NCS at its connected bus's load-balance row, per block. Written
/// for every NCS: a dormant NCS's generation column is `[0, 0]`.
pub(super) fn fill_ncs_load_balance_entries(
    ctx: &TemplateBuildCtx<'_>,
    layout: &StageLayout,
    col_entries: &mut [Vec<(usize, f64)>],
) {
    for (ncs_sys_idx, ncs) in ctx.non_controllable_sources.iter().enumerate() {
        let Some(bus_idx) = ctx.positions.bus(ncs.bus_id) else {
            continue;
        };
        for blk in (0..layout.clock.n_blks()).map(BlockIdx::new) {
            let col = layout
                .geometry
                .ncs_generation_col(NcsSys::new(ncs_sys_idx), blk);
            let row = layout.geometry.load_balance_row(BusSys::new(bus_idx), blk);
            col_entries[col].push((row, 1.0));
        }
    }
}

/// Fill the z-inflow definition row per hydro `z_h − Σ_l ψ_l·lag_in[h,l] = base_h + σ_h·η_h`:
/// `+1.0` on `z_h`, `−ψ_l` on each nonzero lag column, addressed via
/// [`StateSpace::lag_incoming_col`](crate::indexer::StateSpace::lag_incoming_col).
pub(super) fn fill_z_inflow_entries(
    ctx: &TemplateBuildCtx<'_>,
    stage_idx: usize,
    layout: &StageLayout,
    col_entries: &mut [Vec<(usize, f64)>],
) {
    let n_h = layout.state.hydro_count;
    let lag_order = layout.state.max_par_order;

    for h_idx in 0..n_h {
        let row = layout.z_inflow_row(HydroSys::new(h_idx));

        let col_z = layout.state.z_inflow_col(HydroSys::new(h_idx)).get();
        col_entries[col_z].push((row, 1.0));

        if ctx.par_lp.n_stages() > 0 {
            let psi = ctx.par_lp.psi_slice(stage_idx, h_idx);
            for (lag, &psi_val) in psi.iter().enumerate() {
                if psi_val != 0.0 && lag < lag_order {
                    let col = layout
                        .state
                        .lag_incoming_col(lag, HydroSys::new(h_idx))
                        .get();
                    col_entries[col].push((row, -psi_val));
                }
            }
        }
    }
}

/// Fill entries for the 4 operational-violation families, linking decision
/// variables to their slack columns:
///
/// Both outflow rows bind the NON-DIVERTED river-remnant flow `q + s`; the
/// diversion column `d` is EXCLUDED from both. Diversion routes to a different
/// downstream target (a separate flow path, capped by its own
/// `max_diversion_m3s` column bound), not the plant's natural reach. Coupling
/// `d` into either row is a wrong-but-compiling bound: on the minimum it lets
/// diverted water satisfy the floor (understating the mandated river release);
/// on the maximum it double-governs the diversion's own cap.
///
/// - **Min outflow** (`>=`, per hydro): `q + s + sigma_below`
/// - **Max outflow** (`<=`, per hydro): `q + s - sigma_above`
/// - **Min turbine** (`>=`, per CELL): `q_c + sigma_below_c`
/// - **Min generation** (`>=`, per CELL): `var_c + sigma_below_c`, where `var_c`
///   is `rho * q_c` for constant-productivity hydros, the cell's own generation
///   column `g_c` for FPHA, or absent (`sigma_below_c` alone absorbs the floor)
///   for a commissioning-dormant FPHA plant.
///
/// The two power-side families couple only the CELL's own columns to the
/// CELL's own slack — never summed across a plant's cells the way the two
/// flow families above are — see the min-floor contract.
pub(super) fn fill_operational_violation_entries(
    ctx: &TemplateBuildCtx<'_>,
    stage_idx: usize,
    layout: &StageLayout,
    col_entries: &mut [Vec<(usize, f64)>],
) {
    let geom = &layout.geometry;
    let n_blks = layout.clock.n_blks();

    for h_idx in 0..layout.state.hydro_count {
        let hydro = HydroSys::new(h_idx);
        for blk in (0..n_blks).map(BlockIdx::new) {
            let row = layout.min_outflow_row(hydro, blk);
            for c in ctx.hydro_cell_index.cells_of(hydro) {
                let col_q = geom.turbine_col(HydroCell::new(c), blk);
                col_entries[col_q].push((row, 1.0));
            }
            let col_s = geom.spillage_col(hydro, blk);
            col_entries[col_s].push((row, 1.0));
            // Diversion `d` is intentionally NOT coupled into either outflow row —
            // both bind the non-diverted `q + s` (see the fn doc); re-adding it is
            // the wrong-but-compiling bound.
            let col_slack = geom.outflow_below_col(hydro, blk);
            col_entries[col_slack].push((row, 1.0));
        }

        for blk in (0..n_blks).map(BlockIdx::new) {
            let row = layout.max_outflow_row(hydro, blk);
            for c in ctx.hydro_cell_index.cells_of(hydro) {
                let col_q = geom.turbine_col(HydroCell::new(c), blk);
                col_entries[col_q].push((row, 1.0));
            }
            let col_s = geom.spillage_col(hydro, blk);
            col_entries[col_s].push((row, 1.0));
            let col_slack = geom.outflow_above_col(hydro, blk);
            col_entries[col_slack].push((row, -1.0));
        }

        // Per-cell, not plant-keyed: each cell's own min-turbine row couples ONLY
        // its own turbine column to its own slack column — never the plant's other
        // cells (see fill_operational_violation_rows for the matching per-cell RHS).
        for c in ctx.hydro_cell_index.cells_of(hydro) {
            let cell = HydroCell::new(c);
            for blk in (0..n_blks).map(BlockIdx::new) {
                let row = layout.min_turbine_row(cell, blk);
                let col_q = geom.turbine_col(cell, blk);
                col_entries[col_q].push((row, 1.0));
                let col_slack = geom.turbine_below_col(cell, blk);
                col_entries[col_slack].push((row, 1.0));
            }
        }

        // Per-cell min-generation row: FPHA couples the cell's own generation
        // column; ConstantProductivity couples the cell's own turbine column at
        // rho. A commissioning-dormant FPHA plant (`StageProductionRole::Dormant`)
        // couples neither — it has no generation column and no productivity to
        // price its frozen turbine column at — leaving only the row's own slack
        // to absorb a nonzero floor, exactly as a dormant ConstantProductivity
        // plant's frozen (`rho * 0`) turbine term already does.
        match layout.stage_production_role(ctx.production_models, h_idx, stage_idx) {
            StageProductionRole::Fpha(local_fpha_idx) => {
                let fpha_base = layout.fpha_local_first_cell(local_fpha_idx).get();
                for (offset, c) in ctx.hydro_cell_index.cells_of(hydro).enumerate() {
                    let cell = HydroCell::new(c);
                    for blk in (0..n_blks).map(BlockIdx::new) {
                        let row = layout.min_generation_row(cell, blk);
                        let col_g =
                            geom.generation_col(FphaCellLocal::new(fpha_base + offset), blk);
                        col_entries[col_g].push((row, 1.0));
                        let col_slack = geom.generation_below_col(cell, blk);
                        col_entries[col_slack].push((row, 1.0));
                    }
                }
            }
            StageProductionRole::Constant(rho) => {
                for c in ctx.hydro_cell_index.cells_of(hydro) {
                    let cell = HydroCell::new(c);
                    for blk in (0..n_blks).map(BlockIdx::new) {
                        let row = layout.min_generation_row(cell, blk);
                        let col_q = geom.turbine_col(cell, blk);
                        col_entries[col_q].push((row, rho));
                        let col_slack = geom.generation_below_col(cell, blk);
                        col_entries[col_slack].push((row, 1.0));
                    }
                }
            }
            StageProductionRole::Dormant => {
                for c in ctx.hydro_cell_index.cells_of(hydro) {
                    let cell = HydroCell::new(c);
                    for blk in (0..n_blks).map(BlockIdx::new) {
                        let row = layout.min_generation_row(cell, blk);
                        let col_slack = geom.generation_below_col(cell, blk);
                        col_entries[col_slack].push((row, 1.0));
                    }
                }
            }
        }
    }
}

/// Build the unsorted CSC matrix entries for one stage: one `Vec<(row, value)>` per column
/// in insertion order. The caller must sort by row index before assembling (see
/// `build_single_stage_template`; `assemble_csc` asserts the sort).
pub(super) fn build_stage_matrix_entries(
    ctx: &TemplateBuildCtx<'_>,
    stage: &Stage,
    stage_idx: usize,
    layout: &StageLayout,
) -> Vec<Vec<(usize, f64)>> {
    let mut col_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); layout.num_cols];

    fill_state_and_water_entries(ctx, stage, stage_idx, layout, &mut col_entries);
    fill_filling_target_entries(layout, &mut col_entries);
    fill_filled_min_storage_floor_entries(layout, &mut col_entries);
    fill_pumping_water_entries(ctx, layout, &mut col_entries);
    fill_anticipated_state_out_def_entries(stage_idx, layout, &mut col_entries);
    fill_anticipated_slot_definition_entries(layout, &mut col_entries);
    fill_load_balance_entries(ctx, stage_idx, layout, &mut col_entries);
    fill_ncs_load_balance_entries(ctx, layout, &mut col_entries);
    fill_fpha_entries(ctx, stage, stage_idx, layout, &mut col_entries);
    fill_evaporation_entries(ctx, stage, stage_idx, layout, &mut col_entries);
    fill_z_inflow_entries(ctx, stage_idx, layout, &mut col_entries);
    fill_operational_violation_entries(ctx, stage_idx, layout, &mut col_entries);
    fill_anticipated_fishing_entries(ctx, stage, stage_idx, layout, &mut col_entries);

    col_entries
}

/// Assemble CSC arrays from per-column entry lists.
///
/// Returns `(col_starts, row_indices, values)` in the format required by
/// `SolverInterface::load_model`.
pub(super) fn assemble_csc(col_entries: &[Vec<(usize, f64)>]) -> (Vec<i32>, Vec<i32>, Vec<f64>) {
    // Caller owns the sort; `assemble_csc` does NOT sort. Unsorted `row_indices` within a
    // column can make HiGHS/CLP silently misfactorize, so this debug_assert surfaces a
    // missing caller-side sort rather than masking it with a re-sort.
    debug_assert!(
        col_entries
            .iter()
            .all(|c| c.windows(2).all(|w| w[0].0 <= w[1].0)),
        "assemble_csc: each column's entries must be row-sorted (caller-owned sort)"
    );
    let num_cols = col_entries.len();
    let total_nz: usize = col_entries.iter().map(Vec::len).sum();
    let mut col_starts = Vec::with_capacity(num_cols + 1);
    let mut row_indices = Vec::with_capacity(total_nz);
    let mut values = Vec::with_capacity(total_nz);

    let mut offset: i32 = 0;
    for entries in col_entries {
        col_starts.push(offset);
        for &(row, val) in entries {
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_possible_wrap,
                reason = "the solver C API takes i32 row indices, and a stage LP's row count is far below i32::MAX"
            )]
            row_indices.push(row as i32);
            values.push(val);
        }
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_possible_wrap,
            reason = "the solver C API takes i32 offsets, and a stage LP's nonzero count is far below i32::MAX"
        )]
        {
            offset += entries.len() as i32;
        }
    }
    col_starts.push(offset);

    (col_starts, row_indices, values)
}

#[cfg(test)]
mod assemble_csc_tests {
    use super::assemble_csc;

    /// Sorted columns assemble into the expected CSC triple without panicking.
    /// Columns: c0 = [(0, 1.0), (2, 2.0)], c1 = [], c2 = [(1, 3.0)].
    #[test]
    fn test_assemble_csc_sorted_columns_assemble_expected_csc() {
        let col_entries: Vec<Vec<(usize, f64)>> =
            vec![vec![(0, 1.0), (2, 2.0)], vec![], vec![(1, 3.0)]];

        let (col_starts, row_indices, values) = assemble_csc(&col_entries);

        // col_starts has num_cols + 1 entries; the prefix sum of per-column nnz.
        assert_eq!(col_starts, vec![0, 2, 2, 3]);
        assert_eq!(row_indices, vec![0, 2, 1]);
        assert_eq!(values, vec![1.0, 2.0, 3.0]);
    }

    /// A column whose `(row, _)` pairs are out of order trips the `debug_assert`,
    /// proving the caller-owned-sort precondition is guarded in debug/test builds.
    #[test]
    #[should_panic(expected = "each column's entries must be row-sorted")]
    fn test_assemble_csc_unsorted_column_panics() {
        // Column 0 has rows 2 then 1 — strictly decreasing, violating the contract.
        let col_entries: Vec<Vec<(usize, f64)>> = vec![vec![(2, 1.0), (1, 2.0)]];

        let _ = assemble_csc(&col_entries);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests: parameter resolution in the LP builder
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[expect(
    clippy::cast_sign_loss,
    reason = "the test reads non-negative CSC offsets"
)]
mod parameter_resolution_tests {
    use cobre_core::{
        BoundsCountsSpec, BoundsDefaults, Bus, BusStagePenalties, CoefficientRef,
        ConstraintExpression, ContractBlockBounds, DeficitSegment, EntityId, GenericConstraint,
        HydroBlockBounds, HydroPenalties, HydroStageBounds, LineBlockBounds, LineStagePenalties,
        NcsStagePenalties, ParameterKind, PenaltiesCountsSpec, PenaltiesDefaults,
        PumpingBlockBounds, ResolvedBounds, ResolvedGenericConstraintBounds, ResolvedPenalties,
        ScalarParameter, SlackConfig, StageId, SystemBuilder, ThermalBlockBounds,
        ThermalStageBounds,
    };
    use cobre_core::{LinearTerm, VariableRef};
    use cobre_stochastic::normal::precompute::PrecomputedNormal;
    use cobre_stochastic::par::precompute::PrecomputedPar;
    use std::collections::HashMap;

    use crate::build_stage_templates_resolving_layout;
    use crate::energy_conversion::{EnergyConversionSet, build_hydro_energy_productivity_override};
    use crate::hydro_models::PrepareHydroModelsResult;
    use crate::inflow_method::InflowNonNegativityMethod;
    use crate::resolved_parameters::{ResolvedParameters, build_resolved_parameters};

    /// `StageId(0)..StageId(n_stages - 1)`: the 0-based domain ids these
    /// fixtures use (no `Computed` parameter reads the override table here).
    fn stage_ids_0_based(n_stages: usize) -> Vec<StageId> {
        (0..n_stages)
            .map(|s| StageId(i32::try_from(s).expect("test stage count fits in i32")))
            .collect()
    }

    /// Return all CSC values stored at `(col, row)` in the template.
    fn csc_entries_at(t: &cobre_solver::StageTemplate, col: usize, row: usize) -> Vec<f64> {
        let start = t.col_starts[col] as usize;
        let end = t.col_starts[col + 1] as usize;
        t.row_indices[start..end]
            .iter()
            .zip(t.values[start..end].iter())
            .filter_map(|(&r, &v)| if r as usize == row { Some(v) } else { None })
            .collect()
    }

    fn default_hydro_bounds() -> HydroStageBounds {
        HydroStageBounds {
            min_storage_hm3: 0.0,
            max_storage_hm3: 200.0,
            filling_min_rate_m3s: 0.0,
            water_withdrawal_m3s: 0.0,
        }
    }

    fn default_hydro_block_bounds() -> HydroBlockBounds {
        HydroBlockBounds {
            max_turbined_m3s: 100.0,
            max_generation_mw: 250.0,
            ..Default::default()
        }
    }

    fn default_hydro_penalties() -> HydroPenalties {
        HydroPenalties {
            spillage_cost: 0.01,
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
            inflow_nonnegativity_cost: 1000.0,
        }
    }

    /// Build a one-bus, one-thermal system with `n_stages` stages and one
    /// generic constraint. Each stage has a single block of 744 hours.
    fn one_thermal_n_stages(
        n_stages: usize,
        thermal_entity_id: EntityId,
        constraints: Vec<GenericConstraint>,
        bounds: ResolvedGenericConstraintBounds,
    ) -> cobre_core::System {
        use chrono::NaiveDate;
        use cobre_core::entities::thermal::Thermal;
        use cobre_core::scenario::LoadModel;
        use cobre_core::temporal::{
            Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
            StageStateConfig,
        };

        let bus = Bus {
            id: EntityId(1),
            name: "B1".to_string(),
            operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 500.0,
            }],
            excess_cost: 0.0,
        };

        let thermal = Thermal {
            id: thermal_entity_id,
            name: "T1".to_string(),
            operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId(1),
            min_generation_mw: 0.0,
            max_generation_mw: 100.0,
            cost_per_mwh: 50.0,
            anticipated_config: None,
            entry_stage_id: None,
            exit_stage_id: None,
        };

        let stages: Vec<Stage> = (0..n_stages)
            .map(|i| Stage {
                index: i,
                id: i as i32,
                start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
                season_id: Some(0),
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
            })
            .collect();

        let load_models: Vec<_> = (0..n_stages)
            .map(|i| LoadModel {
                bus_id: EntityId(1),
                stage_id: i as i32,
                mean_mw: 100.0,
                std_mw: 0.0,
            })
            .collect();

        let resolved_bounds = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 0,
                n_thermals: 1,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages,
                k_max: 0,
            },
            &BoundsDefaults {
                hydro: default_hydro_bounds(),
                hydro_block: default_hydro_block_bounds(),
                thermal: ThermalStageBounds { cost_per_mwh: 0.0 },
                thermal_block: ThermalBlockBounds {
                    min_generation_mw: 0.0,
                    max_generation_mw: 100.0,
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
        let penalties = ResolvedPenalties::new(
            &PenaltiesCountsSpec {
                n_hydros: 0,
                n_buses: 1,
                n_lines: 0,
                n_ncs: 0,
                n_stages,
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
            .stages(stages)
            .load_models(load_models)
            .bounds(resolved_bounds)
            .penalties(penalties)
            .generic_constraints(constraints)
            .resolved_generic_bounds(bounds)
            .build()
            .expect("one_thermal_n_stages: valid system")
    }

    /// Build templates for the given system using the supplied `ResolvedParameters`.
    fn make_templates(
        system: &cobre_core::System,
        resolved_params: &ResolvedParameters,
    ) -> Vec<cobre_solver::StageTemplate> {
        let production = PrepareHydroModelsResult::default_from_system(system).production;
        let evaporation = PrepareHydroModelsResult::default_from_system(system).evaporation;
        build_stage_templates_resolving_layout(
            system,
            InflowNonNegativityMethod::None,
            &PrecomputedPar::default(),
            &PrecomputedNormal::default(),
            &production,
            &evaporation,
            resolved_params,
        )
        .expect("make_templates: valid")
        .templates
    }

    /// Build an empty `ResolvedParameters` table (no parameters) with a given
    /// stage-to-season mapping.
    fn empty_resolved_params(n_stages: usize) -> ResolvedParameters {
        let stage_to_season: Vec<i32> = vec![0; n_stages];
        let stage_ids = stage_ids_0_based(n_stages);
        let ec = EnergyConversionSet::new(vec![], vec![], &[], n_stages);
        let override_table =
            build_hydro_energy_productivity_override(&[]).expect("empty override table");
        build_resolved_parameters(
            &[],
            &ec,
            &override_table,
            &[],
            &stage_to_season,
            &stage_ids,
            &vec![1usize; n_stages],
            1_000_000.0,
        )
        .expect("empty_resolved_params: valid")
    }

    /// Build a `ResolvedParameters` table containing a single `Constant` parameter.
    fn constant_param_resolved(
        param_id: EntityId,
        value: f64,
        n_stages: usize,
    ) -> ResolvedParameters {
        let stage_to_season: Vec<i32> = vec![0; n_stages];
        let stage_ids = stage_ids_0_based(n_stages);
        let ec = EnergyConversionSet::new(vec![], vec![], &[], n_stages);
        let override_table =
            build_hydro_energy_productivity_override(&[]).expect("empty override table");
        let params = vec![ScalarParameter {
            id: param_id,
            name: format!("p{}", param_id.0),
            kind: ParameterKind::Constant { value },
        }];
        build_resolved_parameters(
            &params,
            &ec,
            &override_table,
            &[],
            &stage_to_season,
            &stage_ids,
            &vec![1usize; n_stages],
            1_000_000.0,
        )
        .expect("constant_param_resolved: valid")
    }

    /// Build a `ResolvedParameters` table containing a single `PerStage` parameter.
    fn per_stage_param_resolved(param_id: EntityId, values: Vec<f64>) -> ResolvedParameters {
        let n_stages = values.len();
        let stage_to_season: Vec<i32> = vec![0; n_stages];
        let stage_ids = stage_ids_0_based(n_stages);
        let ec = EnergyConversionSet::new(vec![], vec![], &[], n_stages);
        let override_table =
            build_hydro_energy_productivity_override(&[]).expect("empty override table");
        let params = vec![ScalarParameter {
            id: param_id,
            name: format!("p{}", param_id.0),
            kind: ParameterKind::PerStage { values },
        }];
        build_resolved_parameters(
            &params,
            &ec,
            &override_table,
            &[],
            &stage_to_season,
            &stage_ids,
            &vec![1usize; n_stages],
            1_000_000.0,
        )
        .expect("per_stage_param_resolved: valid")
    }

    /// Make a generic constraint with a single `Parameter` term over a thermal.
    fn parameter_constraint(
        constraint_id: EntityId,
        param_id: EntityId,
        scale: f64,
        thermal_id: EntityId,
    ) -> GenericConstraint {
        GenericConstraint {
            id: constraint_id,
            name: format!("gc_{}", constraint_id.0),
            description: None,
            expression: ConstraintExpression {
                terms: vec![LinearTerm::parameter(
                    param_id,
                    scale,
                    VariableRef::ThermalGeneration {
                        thermal_id,
                        block_id: None,
                    },
                )],
            },
            slack: SlackConfig {
                enabled: false,
                penalty: None,
            },
            bound_lower_affine: None,
            bound_upper_affine: None,
        }
    }

    /// Make a generic constraint with a single `Literal` term over a thermal.
    fn literal_constraint(
        constraint_id: EntityId,
        coef: f64,
        scale: f64,
        thermal_id: EntityId,
    ) -> GenericConstraint {
        GenericConstraint {
            id: constraint_id,
            name: format!("gc_lit_{}", constraint_id.0),
            description: None,
            expression: ConstraintExpression {
                terms: vec![LinearTerm {
                    coefficient: CoefficientRef::Literal(coef),
                    scale,
                    variable: VariableRef::ThermalGeneration {
                        thermal_id,
                        block_id: None,
                    },
                }],
            },
            slack: SlackConfig {
                enabled: false,
                penalty: None,
            },
            bound_lower_affine: None,
            bound_upper_affine: None,
        }
    }

    /// Build a `ResolvedGenericConstraintBounds` for a single constraint active
    /// at all `n_stages` stages (bound value `50.0`).
    fn bounds_for_n_stages(
        constraint_id: EntityId,
        n_stages: usize,
    ) -> ResolvedGenericConstraintBounds {
        let id_map: HashMap<i32, usize> = [(constraint_id.0, 0)].into_iter().collect();
        let rows = (0..n_stages)
            .map(|s| (constraint_id.0, s as i32, None, None, Some(50.0_f64)))
            .collect::<Vec<_>>();
        ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter())
    }

    // ── Layout constants for 1-bus, 1-thermal, 1-block, 0-hydro ──────────────
    //
    // theta=col 0, decision_start=col 1
    // thermal col 0 block 0 → col 1
    // load_balance row 0 → generic constraint row at row 1
    const THERMAL_COL: usize = 1;
    const GENERIC_ROW: usize = 1;

    // ─────────────────────────────────────────────────────────────────────────
    // Test 1: single stage, constant parameter, coefficient = param * scale * 1
    // ─────────────────────────────────────────────────────────────────────────

    /// A `CoefficientRef::Parameter` term is resolved against `ResolvedParameters`
    /// at LP-build time.
    ///
    /// Fixture: EntityId(7) → 3.0 (constant), scale 2.0.
    /// Expected CSC value: 3.0 * 2.0 * 1.0 = 6.0.
    #[test]
    fn parameter_coefficient_is_resolved_at_lp_build() {
        let param_id = EntityId(7);
        let thermal_id = EntityId(2);
        let constraint_id = EntityId(10);
        let scale = 2.0_f64;
        let param_value = 3.0_f64;

        let constraint = parameter_constraint(constraint_id, param_id, scale, thermal_id);
        let generic_bounds = bounds_for_n_stages(constraint_id, 1);

        let system = one_thermal_n_stages(1, thermal_id, vec![constraint], generic_bounds);
        let resolved = constant_param_resolved(param_id, param_value, 1);
        let templates = make_templates(&system, &resolved);

        let t = &templates[0];
        let entries = csc_entries_at(t, THERMAL_COL, GENERIC_ROW);
        assert!(
            !entries.is_empty(),
            "no CSC entry at (col={THERMAL_COL}, row={GENERIC_ROW}) for parameter term"
        );
        let total: f64 = entries.iter().sum();
        let expected = param_value * scale; // multiplier=1.0
        assert_eq!(
            total.to_bits(),
            expected.to_bits(),
            "expected {expected} (bit-exact), got {total}"
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 2: three stages, per-stage parameter, each stage gets its own value
    // ─────────────────────────────────────────────────────────────────────────

    /// When the parameter has per-stage values, each stage template receives
    /// the stage-specific resolved coefficient.
    ///
    /// Fixture: EntityId(7) → [3.0, 7.0, 11.0], scale 2.0.
    /// Expected CSC values: stage 0 → 6.0, stage 1 → 14.0, stage 2 → 22.0.
    #[test]
    fn parameter_coefficient_per_stage_changes_lp() {
        let param_id = EntityId(7);
        let thermal_id = EntityId(2);
        let constraint_id = EntityId(10);
        let scale = 2.0_f64;
        let per_stage_values = vec![3.0_f64, 7.0, 11.0];
        let n_stages = per_stage_values.len();

        let constraint = parameter_constraint(constraint_id, param_id, scale, thermal_id);
        let generic_bounds = bounds_for_n_stages(constraint_id, n_stages);

        let system = one_thermal_n_stages(n_stages, thermal_id, vec![constraint], generic_bounds);
        let resolved = per_stage_param_resolved(param_id, per_stage_values.clone());
        let templates = make_templates(&system, &resolved);

        for (stage_idx, &param_val) in per_stage_values.iter().enumerate() {
            let t = &templates[stage_idx];
            let entries = csc_entries_at(t, THERMAL_COL, GENERIC_ROW);
            assert!(
                !entries.is_empty(),
                "no CSC entry at (col={THERMAL_COL}, row={GENERIC_ROW}) for stage {stage_idx}"
            );
            let total: f64 = entries.iter().sum();
            let expected = param_val * scale; // multiplier=1.0
            assert_eq!(
                total.to_bits(),
                expected.to_bits(),
                "stage {stage_idx}: expected {expected} (bit-exact), got {total}"
            );
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 3: regression — Literal coefficients still work after wiring
    // ─────────────────────────────────────────────────────────────────────────

    /// `CoefficientRef::Literal(5.0)` with scale 3.0 must produce a CSC entry
    /// of exactly `5.0 * 3.0 * 1.0 = 15.0`, unchanged by the parameter-resolution
    /// wiring.
    #[test]
    fn literal_coefficient_still_works_after_wiring() {
        let thermal_id = EntityId(2);
        let constraint_id = EntityId(10);
        let coef = 5.0_f64;
        let scale = 3.0_f64;

        let constraint = literal_constraint(constraint_id, coef, scale, thermal_id);
        let generic_bounds = bounds_for_n_stages(constraint_id, 1);

        let system = one_thermal_n_stages(1, thermal_id, vec![constraint], generic_bounds);
        let resolved = empty_resolved_params(1);
        let templates = make_templates(&system, &resolved);

        let t = &templates[0];
        let entries = csc_entries_at(t, THERMAL_COL, GENERIC_ROW);
        assert!(
            !entries.is_empty(),
            "no CSC entry at (col={THERMAL_COL}, row={GENERIC_ROW}) for literal term"
        );
        let total: f64 = entries.iter().sum();
        let expected = coef * scale; // multiplier=1.0
        assert_eq!(
            total.to_bits(),
            expected.to_bits(),
            "expected {expected} (bit-exact), got {total}"
        );
    }
}

#[cfg(test)]
mod zero_cost_tests {

    use chrono::NaiveDate;
    use cobre_core::{
        BoundsCountsSpec, BoundsDefaults, ContractBlockBounds, EntityId, HydroBlockBounds,
        HydroStageBounds, LineBlockBounds, PumpingBlockBounds, ResolvedBounds, Stage, Thermal,
        ThermalBlockBounds, ThermalStageBounds,
    };

    use crate::indexer::BlockIdx;
    use crate::lead_time::{AnticipatedResolution, DeliveryAxis, LeadTime, PointResolution};

    use crate::test_support::anticipated_plants_at;
    use crate::test_support::ctx_fixture::CtxFixture;
    use crate::time_value::{PostStudyResolved, TimeValue};

    use super::super::columns::{fill_stage_columns, fill_thermal_columns};
    use super::super::delivery_ring::ColumnBufs;
    use super::super::layout::{StageLayout, TemplateBuildCtx, entity_flat};
    use super::super::rows::{
        fill_anticipated_fishing_rows, fill_anticipated_state_out_def_rows, fill_stage_rows,
    };
    use super::super::test_support::two_block_stage;
    use super::{
        DeliveryRing, build_stage_matrix_entries, fill_anticipated_fishing_entries,
        fill_anticipated_slot_definition_entries, fill_anticipated_state_out_def_entries,
    };

    /// Placeholder thermal at position `idx`, inert past its `id`:
    /// `make_ctx`'s `n_thermals` needs only the count `ctx.thermals.len()`
    /// reserves in `StageLayout`, never a bound or cost.
    fn dormant_thermal(idx: usize) -> Thermal {
        Thermal {
            id: EntityId(i32::try_from(idx).unwrap_or(i32::MAX)),
            name: String::new(),
            operational_start_date: NaiveDate::default(),
            bus_id: EntityId(0),
            entry_stage_id: None,
            exit_stage_id: None,
            cost_per_mwh: 0.0,
            min_generation_mw: 0.0,
            max_generation_mw: 0.0,
            anticipated_config: None,
        }
    }

    /// Owns data for a context with anticipated thermals and zero other entities.
    struct AntFixtures {
        base: CtxFixture,
    }

    impl AntFixtures {
        /// Build a minimal `ResolvedBounds` with the given thermal count,
        /// `n_stages`, and `k_max` (all other entity counts zero).
        ///
        /// `n_stages` must exceed the queried `stage_idx` so the anticipated
        /// layout the fixture builds places every plant inside the study horizon
        /// `[0, n_stages)`. `k_max` sizes the thermal-bounds stage axis: it must
        /// be large enough that every delivery stage `stage_idx + K_i` the test
        /// accesses falls within `[0, n_stages + k_max)` (callers whose
        /// deliveries all fall strictly inside `[0, n_stages)` pass `0`).
        /// `n_thermals` sizes the thermal slot axis: tests that read or mutate
        /// `thermal_bounds(t_idx, …)` for an active anticipated plant must size it
        /// to cover every `t_idx` they touch; pure state-out/row tests pass `0`.
        fn bounds_with_n_stages(
            n_stages: usize,
            k_max: usize,
            n_thermals: usize,
        ) -> ResolvedBounds {
            ResolvedBounds::new(
                &BoundsCountsSpec {
                    n_hydros: 0,
                    n_thermals,
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
                    hydro_block: HydroBlockBounds::default(),
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

        fn new() -> Self {
            Self {
                base: CtxFixture::default(),
            }
        }

        /// `anticipated_positions` must be strictly ascending
        /// (`test_support::anticipated_plants_at`).
        fn make_ctx(
            &mut self,
            k_max: usize,
            anticipated_lead_stages: Vec<usize>,
            anticipated_positions: &[usize],
            n_thermals: usize,
        ) -> TemplateBuildCtx<'_> {
            self.base.anticipated_plants = anticipated_plants_at(anticipated_positions);
            self.base.anticipated_lead_stages = anticipated_lead_stages;
            self.base.thermals = (0..n_thermals).map(dormant_thermal).collect();
            // Sized to cover every active plant's delivery stage
            // (`stage_idx + K_i < n_stages`); `fill_anticipated_columns`
            // indexes these by delivery stage when pricing the decision column.
            self.base.time_value = TimeValue::from_parts(
                vec![],
                vec![1.0; self.base.bounds.n_stages() + k_max],
                vec![744.0; self.base.bounds.n_stages() + k_max],
                (0..i32::try_from(self.base.bounds.n_stages() + k_max).unwrap_or(0)).collect(),
                PostStudyResolved::default(),
            );
            self.base.ctx()
        }
    }

    /// `fill_thermal_columns` skips the per-block objective for anticipated
    /// plants (leaving the `0.0` vec default) while still writing it for
    /// non-anticipated thermals — the order-independent replacement for the
    /// former write-then-zero pass. The anticipated plant's per-block **bounds**
    /// are still written.
    ///
    /// Fixture: two thermals, thermal 0 anticipated (`K=1`), thermal 1 standard,
    /// both with non-zero resolved `cost_per_mwh` so the skip is observable
    /// (a regression that priced anticipated plants would leave a non-zero
    /// objective on thermal 0).
    #[test]
    fn fill_thermal_columns_skips_objective_for_anticipated_plants() {
        use cobre_core::entities::thermal::AnticipatedConfig;
        use cobre_core::{EntityId, Thermal};

        const ANT_COST: f64 = 30.0;
        const STD_COST: f64 = 40.0;

        let thermals = vec![
            Thermal {
                id: EntityId(1),
                name: "T_ant".to_string(),
                operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                bus_id: EntityId(1),
                min_generation_mw: 0.0,
                max_generation_mw: 100.0,
                cost_per_mwh: ANT_COST,
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
                max_generation_mw: 100.0,
                cost_per_mwh: STD_COST,
                anticipated_config: None,
                entry_stage_id: None,
                exit_stage_id: None,
            },
        ];

        let mut fixtures = AntFixtures::new();
        // 10 stages, k_max=1; seed each thermal's resolved per-stage cost so the
        // objective write (when not skipped) is non-zero.
        fixtures.base.bounds = AntFixtures::bounds_with_n_stages(10, 1, 2);
        for stage in 0..10 {
            fixtures
                .base
                .bounds
                .thermal_bounds_mut(0, stage)
                .cost_per_mwh = ANT_COST;
            fixtures
                .base
                .bounds
                .thermal_block_base_mut(0, stage)
                .max_generation_mw = 100.0;
            fixtures
                .base
                .bounds
                .thermal_bounds_mut(1, stage)
                .cost_per_mwh = STD_COST;
            fixtures
                .base
                .bounds
                .thermal_block_base_mut(1, stage)
                .max_generation_mw = 100.0;
        }
        let mut ctx = fixtures.make_ctx(
            1,       // k_max
            vec![1], // anticipated_lead_stages: K_0 = 1
            &[0],    // anticipated_positions: thermal 0 is anticipated
            2,       // n_thermals
        );
        ctx.thermals = &thermals;

        let stage = two_block_stage(2, [372.0, 372.0]);
        let layout = StageLayout::new(&ctx, &stage, 2);

        let mut objective = vec![0.0_f64; layout.num_cols];
        let mut col_lower = vec![0.0_f64; layout.num_cols];
        let mut col_upper = vec![0.0_f64; layout.num_cols];
        let mut bufs = ColumnBufs {
            col_lower: &mut col_lower,
            col_upper: &mut col_upper,
            objective: &mut objective,
        };

        fill_thermal_columns(&ctx, &stage, 2, &layout, &mut bufs);

        let n_blks = layout.clock.n_blks();
        // Anticipated thermal 0: objective skipped (stays 0.0), bounds still set.
        // Thermal 0's block columns start at col_thermal_start (t_idx 0 offset).
        for blk in 0..n_blks {
            let col = layout.geometry.thermal.start + blk;
            assert_eq!(
                bufs.objective[col], 0.0,
                "anticipated thermal 0 objective must stay 0.0 at col {col}",
            );
            assert_eq!(
                bufs.col_upper[col], 100.0,
                "anticipated thermal 0 bounds must still be written at col {col}",
            );
        }
        // Standard thermal 1: objective priced as cost * block_hours.
        for blk in 0..n_blks {
            let col = layout.geometry.thermal.start + n_blks + blk;
            let expected = STD_COST * stage.blocks[blk].duration_hours;
            assert_eq!(
                bufs.objective[col], expected,
                "standard thermal 1 objective must be priced at col {col}",
            );
        }
    }

    /// Fills rows for all anticipated plants (always-active predicate).
    /// At stage 2 with `K_i=[1, 5]` and `n_anticipated=2`: both plants are active,
    /// so exactly two fishing rows are written.
    #[test]
    fn fishing_rows_fill_all_plants() {
        let mut fixtures = AntFixtures::new();
        fixtures.base.bounds = AntFixtures::bounds_with_n_stages(10, 0, 0);
        let ctx = fixtures.make_ctx(5, vec![1, 5], &[0, 1], 2);
        let stage = two_block_stage(2, [372.0, 372.0]);
        let layout = StageLayout::new(&ctx, &stage, 2);

        // Always-active: both plants active at stage 2 → two fishing rows.
        assert_eq!(
            layout.anticipated.fishing_rows.len(),
            2,
            "expected fishing_rows.len() == 2, got {}",
            layout.anticipated.fishing_rows.len()
        );

        let mut row_lower = vec![f64::NAN; layout.rows.num_rows];
        let mut row_upper = vec![f64::NAN; layout.rows.num_rows];

        fill_anticipated_fishing_rows(&layout, &mut row_lower, &mut row_upper);

        // Both plants write a row with (0.0, 0.0) bounds.
        for local_idx in 0..layout.anticipated.fishing_rows.len() {
            let row = entity_flat(&layout.anticipated.fishing_rows, local_idx);
            assert_eq!(
                row_lower[row], 0.0,
                "row_lower[{row}] (local_idx={local_idx}) expected 0.0, got {}",
                row_lower[row]
            );
            assert_eq!(
                row_upper[row], 0.0,
                "row_upper[{row}] (local_idx={local_idx}) expected 0.0, got {}",
                row_upper[row]
            );
        }
    }

    /// Always-active at `stage_idx = 0`: with `K = [1, 5]` and `n_anticipated = 2`,
    /// both plants are active even before their lead time elapses.
    /// Asserts `layout.anticipated.fishing_rows.len() == 2`, that both rows
    /// are filled with `(0.0, 0.0)` bounds, and that the anticipated-state
    /// slot-0 column carries the `-block_hours_total` coupling for both plants.
    #[test]
    fn fishing_rows_always_active_stage_zero() {
        let mut fixtures = AntFixtures::new();
        fixtures.base.bounds = AntFixtures::bounds_with_n_stages(10, 0, 0);
        let ctx = fixtures.make_ctx(5, vec![1, 5], &[0, 1], 2);
        let stage = two_block_stage(0, [372.0, 372.0]);
        let layout = StageLayout::new(&ctx, &stage, 0);

        // Always-active: both plants are active at stage 0 → two fishing rows.
        assert_eq!(
            layout.anticipated.fishing_rows.len(),
            2,
            "expected fishing_rows.len() == 2 at stage 0, got {}",
            layout.anticipated.fishing_rows.len()
        );

        let mut row_lower = vec![f64::NAN; layout.rows.num_rows];
        let mut row_upper = vec![f64::NAN; layout.rows.num_rows];

        fill_anticipated_fishing_rows(&layout, &mut row_lower, &mut row_upper);

        // Both plants write equality rows with (0.0, 0.0) bounds.
        for local_idx in 0..layout.anticipated.fishing_rows.len() {
            let row = entity_flat(&layout.anticipated.fishing_rows, local_idx);
            assert_eq!(
                row_lower[row], 0.0,
                "row_lower[{row}] (local_idx={local_idx}) expected 0.0, got {}",
                row_lower[row]
            );
            assert_eq!(
                row_upper[row], 0.0,
                "row_upper[{row}] (local_idx={local_idx}) expected 0.0, got {}",
                row_upper[row]
            );
        }

        // CSC coupling: the maturing slot's incoming (commit_in) column carries
        // (row, -block_hours_total) for each plant under the always-active
        // predicate. At stage 0 the maturing slot is 0 (0 mod k_max), so it is
        // commit_in slot 0 = ctx.state.commit_in.start + local_idx.
        let mut col_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); layout.num_cols];
        fill_anticipated_fishing_entries(&ctx, &stage, 0, &layout, &mut col_entries);

        let block_hours_total: f64 = stage.blocks.iter().map(|b| b.duration_hours).sum();
        let expected_neg = -block_hours_total;
        for local_idx in 0..layout.anticipated.fishing_rows.len() {
            let row = entity_flat(&layout.anticipated.fishing_rows, local_idx);
            let col_state = layout.state.commit_in.start + local_idx;
            let state_couplings: Vec<&(usize, f64)> = col_entries[col_state]
                .iter()
                .filter(|(r, _)| *r == row)
                .collect();
            assert_eq!(
                state_couplings.len(),
                1,
                "anticipated_state col {col_state} must carry exactly 1 coupling \
                 at fishing row {row} (plant local_idx={local_idx})"
            );
            let (_, coeff) = state_couplings[0];
            assert!(
                (coeff - expected_neg).abs() < 1e-12,
                "anticipated_state col {col_state} fishing-row coupling: \
                 expected {expected_neg}, got {coeff} (plant local_idx={local_idx})"
            );
        }
    }

    /// Regression: `build_anticipated_fishing_row_pos` gates a plant's
    /// fishing row on `k_max >= 1`, not merely `n_anticipated >= 1`. A lead-0
    /// plant (`K = 0` self-delivery) is the only state production can build
    /// with `k_max == 0` (`ring_size(&[0]) == 0`), and it reaches this same
    /// guard: `fill_anticipated_fishing_entries` must reach the final line
    /// without panicking and without writing any coupling.
    #[test]
    fn fishing_fill_on_an_empty_ring_does_not_divide_by_zero() {
        let mut fixtures = AntFixtures::new();
        fixtures.base.bounds = AntFixtures::bounds_with_n_stages(1, 0, 1);
        let ctx = fixtures.make_ctx(0, vec![0], &[0], 1);
        let stage = two_block_stage(0, [372.0, 372.0]);
        assert_eq!(
            ctx.state.n_anticipated, 1,
            "fixture sanity: one anticipated plant"
        );
        assert_eq!(
            ctx.state.k_max, 0,
            "fixture sanity: a lead-0 plant's ring is empty"
        );
        let layout = StageLayout::new(&ctx, &stage, 0);

        let mut col_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); layout.num_cols];
        fill_anticipated_fishing_entries(&ctx, &stage, 0, &layout, &mut col_entries);

        assert!(
            col_entries.iter().all(Vec::is_empty),
            "an empty ring must emit no anticipated fishing coupling entries"
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Fixture helper for anticipated state-out tests
    // ─────────────────────────────────────────────────────────────────────────

    /// Build a minimal fixture for anticipated state-out tests:
    /// `n_anticipated = 2`, `K = [2, 3]`, `n_stages = 6`.
    ///
    /// `ResolvedBounds` is constructed with the correct `n_stages` so that
    /// `ctx.resolved.bounds.n_stages()` returns 6, which is required by
    /// `fill_anticipated_columns`, `fill_anticipated_state_out_def_rows`,
    /// and `fill_anticipated_state_out_def_entries`.
    fn build_anticipated_ctx_n_stages_6() -> (AntFixtures, Stage) {
        let mut fixtures = AntFixtures::new();
        // Override bounds with a 6-stage, 2-thermal table; k_max = 3 matches the
        // anticipated config the state-out tests pass to `make_ctx`. The two
        // thermal slots back the active anticipated plants' delivery-stage bound
        // reads in `fill_anticipated_columns`; the row/def tests sharing this
        // fixture leave them unread.
        fixtures.base.bounds = AntFixtures::bounds_with_n_stages(6, 3, 2);
        let stage = two_block_stage(0, [372.0, 372.0]);
        (fixtures, stage)
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Tests for the anticipated state-out columns (fill_anticipated_columns)
    // ─────────────────────────────────────────────────────────────────────────

    /// Active plants (`stage_idx + K_i < n_stages`) get `[-INF, +INF]` bounds;
    /// inactive plants (`stage_idx + K_i >= n_stages`) get `[0, 0]` bounds.
    ///
    /// Fixture: `n_anticipated=2`, `K=[2, 3]`, `n_stages=6`.
    /// Stage 0: both plants active  (0+2=2 < 6, 0+3=3 < 6) → `[-INF, +INF]`.
    /// Stage 5: both plants inactive (5+2=7 >= 6, 5+3=8 >= 6) → `[0, 0]`.
    ///
    /// Runs the full `fill_stage_columns` pipeline (not `fill_anticipated_columns`
    /// alone): a stage with NO genuine decision at all (stage 5's hypothetical
    /// delivery is out of horizon, so `genuine_decisions_at` is empty) has its
    /// ring slot frozen by `fill_anticipated_slot_columns`'s masking, not by
    /// `fill_anticipated_columns` — the two functions collaborate on the
    /// dormant-column convention exactly as `fill_stage_columns` composes them.
    #[test]
    fn test_fill_anticipated_columns_state_out_active_and_inactive() {
        let (mut fixtures, _) = build_anticipated_ctx_n_stages_6();
        let ctx = fixtures.make_ctx(
            3,          // k_max
            vec![2, 3], // anticipated_lead_stages: K=[2,3]
            &[0, 1],    // anticipated_positions
            0,          // n_thermals
        );

        // Stage 0: both plants active.
        let stage0 = two_block_stage(0, [372.0, 372.0]);
        let layout0 = StageLayout::new(&ctx, &stage0, 0);
        let (col_lower, col_upper, _objective) = fill_stage_columns(&ctx, &stage0, 0, &layout0);
        let leads = [2_usize, 3];
        let k_max = 3_usize;
        // Hold deposit slot = delivery mod k_max (delivery = stage 0 + lead), the
        // modular slot fill_anticipated_columns re-frees for an active decision —
        // not the retired shift newest-slot K-1. plant 0 (K=2) -> slot 2, plant 1
        // (K=3) -> slot 0.
        let deposit_cols: Vec<usize> = leads
            .iter()
            .enumerate()
            .map(|(i, &lead)| {
                let slot = lead % k_max;
                DeliveryRing::anticipated(layout0.state).out_col(slot, i)
            })
            .collect();
        for (i, &col) in deposit_cols.iter().enumerate() {
            assert_eq!(
                col_lower[col],
                f64::NEG_INFINITY,
                "stage 0, plant {i}: deposit-slot col_lower expected -INF, got {}",
                col_lower[col]
            );
            assert_eq!(
                col_upper[col],
                f64::INFINITY,
                "stage 0, plant {i}: deposit-slot col_upper expected +INF, got {}",
                col_upper[col]
            );
        }

        // Stage 5: both plants inactive.
        let stage5 = two_block_stage(5, [372.0, 372.0]);
        let layout5 = StageLayout::new(&ctx, &stage5, 5);
        assert_eq!(
            layout5.anticipated.state_out_def_rows.len(),
            0,
            "stage 5 inactive: expected no def rows, got {}",
            layout5.anticipated.state_out_def_rows.len(),
        );
        let (col_lower5, col_upper5, _objective5) = fill_stage_columns(&ctx, &stage5, 5, &layout5);
        // At the inactive stage every anticipated slot is masked [0, 0] (no
        // interior carry, no active deposit), so the SAME physical deposit
        // columns that were free at stage 0 are now frozen. The ring's slot base
        // is stage-invariant, so reuse the stage-0 deposit columns directly.
        for (i, &col) in deposit_cols.iter().enumerate() {
            assert_eq!(
                col_lower5[col], 0.0,
                "stage 5, plant {i}: col_lower expected 0.0, got {}",
                col_lower5[col]
            );
            assert_eq!(
                col_upper5[col], 0.0,
                "stage 5, plant {i}: col_upper expected 0.0, got {}",
                col_upper5[col]
            );
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Tests for fill_anticipated_state_out_def_rows
    // ─────────────────────────────────────────────────────────────────────────

    /// At stage 0 with `K=[2,3]` and `n_stages=6`, both plants are active
    /// (0+2 < 6, 0+3 < 6), so `state_out_def_rows.len() == 2` and
    /// both definition rows must have equality bounds `[0.0, 0.0]`.
    #[test]
    fn test_fill_anticipated_state_out_def_rows_two_active_plants() {
        let (mut fixtures, stage) = build_anticipated_ctx_n_stages_6();
        let ctx = fixtures.make_ctx(3, vec![2, 3], &[0, 1], 0);
        let layout = StageLayout::new(&ctx, &stage, 0);

        assert_eq!(
            layout.anticipated.state_out_def_rows.len(),
            2,
            "expected state_out_def_rows.len() == 2, got {}",
            layout.anticipated.state_out_def_rows.len()
        );

        let mut row_lower = vec![f64::NEG_INFINITY; layout.rows.num_rows];
        let mut row_upper = vec![f64::INFINITY; layout.rows.num_rows];
        fill_anticipated_state_out_def_rows(&layout, &mut row_lower, &mut row_upper);

        for k in 0..2 {
            let row = entity_flat(&layout.anticipated.state_out_def_rows, k);
            assert_eq!(
                row_lower[row], 0.0,
                "def row {k}: row_lower expected 0.0, got {}",
                row_lower[row]
            );
            assert_eq!(
                row_upper[row], 0.0,
                "def row {k}: row_upper expected 0.0, got {}",
                row_upper[row]
            );
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Tests for fill_anticipated_state_out_def_entries
    // ─────────────────────────────────────────────────────────────────────────

    /// At stage 0 with `K=[2,3]` and `n_stages=6`, both plants are active.
    /// For each active plant `i`, the latch (deposit) CSC entry list must contain:
    /// - `(def_row_i, +1.0)` on plant `i`'s own delivery slot `delivery mod k_max`
    ///   (`delivery = 0 + K_i`: plant 0 -> slot 2, plant 1 -> slot 0), not the retired
    ///   shift newest-slot `K-1`
    /// - `(def_row_i, -1.0)` on `geometry.anticipated_decision.start + i`
    #[test]
    fn test_fill_anticipated_state_out_def_entries_two_active_plants() {
        let (mut fixtures, stage) = build_anticipated_ctx_n_stages_6();
        let ctx = fixtures.make_ctx(3, vec![2, 3], &[0, 1], 0);
        let layout = StageLayout::new(&ctx, &stage, 0);

        let mut col_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); layout.num_cols];
        fill_anticipated_state_out_def_entries(0, &layout, &mut col_entries);

        let leads = [2_usize, 3];
        let k_max = 3_usize;
        for (k, &lead) in leads.iter().enumerate() {
            let row = entity_flat(&layout.anticipated.state_out_def_rows, k);
            let slot = lead % k_max; // delivery (0 + lead) mod k_max
            let col_state_out = DeliveryRing::anticipated(layout.state).out_col(slot, k);
            let col_decision = layout.geometry.anticipated_decision.start + k;

            assert!(
                col_entries[col_state_out]
                    .iter()
                    .any(|&(r, v)| r == row && (v - 1.0).abs() < 1e-15),
                "plant {k}: expected (+1.0) entry at (col_state_out={col_state_out}, row={row}), \
                 got {:?}",
                col_entries[col_state_out]
            );
            assert!(
                col_entries[col_decision]
                    .iter()
                    .any(|&(r, v)| r == row && (v + 1.0).abs() < 1e-15),
                "plant {k}: expected (-1.0) entry at (col_decision={col_decision}, row={row}), \
                 got {:?}",
                col_entries[col_decision]
            );
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Two-sided masking: row cap and column freeze in one regression
    // ─────────────────────────────────────────────────────────────────────────

    /// A commitment-hold slot whose delivery target lands past the horizon gets
    /// BOTH sides of the masking contract together: no carry definition row AND a
    /// frozen `[0, 0]` outgoing column — never one without the other.
    ///
    /// Fixture: one plant, `K=3`, `n_stages=6`, evaluated at `stage_idx=4`. The
    /// depth window is delivery targets `m = 5, 6, 7`; only `m = 5` lands inside
    /// the horizon, at modular slot `5 mod 3 = 2` (an in-flight interior carry,
    /// decided at stage 2). Targets `m = 6, 7` map to slots `0` and `1` and are
    /// past the horizon, so those slots are masked. There is no active deposit at
    /// this stage (delivery `4 + 3 = 7 >= n_stages`).
    #[test]
    fn anticipated_slot_masking_ships_row_cap_and_column_freeze_together() {
        let mut fixtures = AntFixtures::new();
        fixtures.base.bounds = AntFixtures::bounds_with_n_stages(6, 3, 1);
        let ctx = fixtures.make_ctx(3, vec![3], &[0], 1);
        let stage = two_block_stage(4, [372.0, 372.0]);
        let layout = StageLayout::new(&ctx, &stage, 4);

        assert_eq!(
            layout.anticipated.anticipated_slot_row_pos,
            vec![None, None, Some(0)],
            "slot 2 reachable (m=5, row pos 0); slots 0 and 1 map to past-horizon \
             targets m=6, m=7 and are masked"
        );
        assert_eq!(layout.anticipated.slot_definition_rows.len(), 1);

        let (col_lower, col_upper, _objective) = fill_stage_columns(&ctx, &stage, 4, &layout);
        let (row_lower, row_upper) = fill_stage_rows(&ctx, &stage, 4, &layout);
        let col_entries = build_stage_matrix_entries(&ctx, &stage, 4, &layout);

        let base = layout.state.commit_out.start;
        let row_start = layout.anticipated.slot_definition_rows.start;

        // Reachable slot 2: free column, a defining row exists, and the CSC
        // carries the same-slot carry identity `out(2) - in(2) = 0`.
        let col2 = base + 2;
        assert_eq!(
            col_lower[col2],
            f64::NEG_INFINITY,
            "slot 2 column must be free"
        );
        assert_eq!(col_upper[col2], f64::INFINITY, "slot 2 column must be free");
        let row0 = row_start;
        assert_eq!(
            row_lower[row0], 0.0,
            "slot 2 definition row must be an equality"
        );
        assert_eq!(
            row_upper[row0], 0.0,
            "slot 2 definition row must be an equality"
        );
        assert!(
            col_entries[col2]
                .iter()
                .any(|&(r, v)| r == row0 && (v - 1.0).abs() < 1e-15),
            "slot 2 outgoing column must carry the +1.0 structural term at its \
             carry definition row; got {:?}",
            col_entries[col2]
        );
        // The carry pins the SAME slot's incoming column (in(2)), never in(slot+1).
        let incoming_slot2 = ctx.state.commit_in.start + 2;
        assert!(
            col_entries[incoming_slot2]
                .iter()
                .any(|&(r, v)| r == row0 && (v + 1.0).abs() < 1e-15),
            "slot 2's own incoming pin must carry the -1.0 structural term at slot \
             2's carry definition row (same slot); got {:?}",
            col_entries[incoming_slot2]
        );

        // Masked slots 0 and 1: frozen columns, no defining row, no CSC entry.
        for masked_slot in [0_usize, 1] {
            let col = base + masked_slot;
            assert_eq!(
                col_lower[col], 0.0,
                "masked slot {masked_slot} column must be frozen"
            );
            assert_eq!(
                col_upper[col], 0.0,
                "masked slot {masked_slot} column must be frozen"
            );
            assert!(
                col_entries[col].is_empty(),
                "masked slot {masked_slot} must carry no CSC entries; got {:?}",
                col_entries[col]
            );
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Equivalence pin: ring-routed entries vs the open-coded reference formula
    // ─────────────────────────────────────────────────────────────────────────

    /// Equivalence pin: `fill_anticipated_slot_definition_entries`'s
    /// `DeliveryRing::emit_carry_rows` routing reproduces the open-coded carry
    /// identity (`+1` on `commit_out.start + global_slot`, `-1` on
    /// `commit_in.start + global_slot` — the SAME slot, `out(slot) - in(slot) = 0`,
    /// never the retired shift target `in(slot + 1)`) for every reachable global
    /// slot on a fixed two-plant, heterogeneous-`K` fixture (`K = [3, 2]`).
    #[test]
    fn fill_anticipated_slot_definition_entries_matches_open_coded_carry_formula_across_heterogeneous_plants()
     {
        let (mut fixtures, stage) = build_anticipated_ctx_n_stages_6();
        let ctx = fixtures.make_ctx(3, vec![3, 2], &[0, 1], 2);
        let layout = StageLayout::new(&ctx, &stage, 0);

        let mut actual: Vec<Vec<(usize, f64)>> = vec![Vec::new(); layout.num_cols];
        fill_anticipated_slot_definition_entries(&layout, &mut actual);

        let mut expected: Vec<Vec<(usize, f64)>> = vec![Vec::new(); layout.num_cols];
        let mut n_expected_reachable = 0_usize;
        for (global_slot, pos) in layout
            .anticipated
            .anticipated_slot_row_pos
            .iter()
            .enumerate()
        {
            let Some(pos) = *pos else { continue };
            let row = entity_flat(&layout.anticipated.slot_definition_rows, pos);
            // Carry identity: +1 on the outgoing slot, -1 on the SAME incoming slot.
            expected[ctx.state.commit_out.start + global_slot].push((row, 1.0));
            expected[ctx.state.commit_in.start + global_slot].push((row, -1.0));
            n_expected_reachable += 1;
        }

        assert_eq!(
            n_expected_reachable,
            layout.anticipated.slot_definition_rows.len(),
            "fixture sanity: reachable count must match the layout's own count"
        );
        assert!(
            n_expected_reachable >= 3,
            "fixture must exercise multiple slots across both plants; got {n_expected_reachable}"
        );
        assert_eq!(
            actual, expected,
            "ring-routed entries must match the pre-migration open-coded formula"
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // K = 0 exclusion
    // ─────────────────────────────────────────────────────────────────────────

    /// `K = 0` exclusion: a `LeadTime(720.0)` plant on the uniform 31-day-month
    /// calendar `[744,744,744,744]` h resolves `depth == [0,0,0,0]` (every
    /// delivery self-delivers, hand-derived from the calendar). No
    /// anticipated slot, deposit row, interior carry row, or fishing row is
    /// ever emitted for this plant, at any stage — the stage LP dispatches its
    /// generation as ordinary, unconstrained thermal output (no fishing
    /// coupling), never an underflow.
    #[test]
    fn k0_sub_stage_lead_emits_no_anticipated_rows_or_fishing_coupling() {
        let mut fixtures = AntFixtures::new();
        fixtures.base.bounds = AntFixtures::bounds_with_n_stages(4, 0, 1);
        fixtures.base.anticipated_resolution = AnticipatedResolution::resolve(
            &[LeadTime::Time(720.0)],
            DeliveryAxis {
                study_stage_hours: &[744.0, 744.0, 744.0, 744.0],
                post_study_stage_hours: &[],
            },
        );
        let ctx = fixtures.make_ctx(0, vec![0], &[0], 1);

        for stage_idx in 0..4 {
            let stage = two_block_stage(stage_idx, [372.0, 372.0]);
            let layout = StageLayout::new(&ctx, &stage, stage_idx);
            assert_eq!(
                layout.anticipated.fishing_rows.len(),
                0,
                "stage {stage_idx}: K=0 must exclude the fishing row entirely"
            );
            assert_eq!(
                layout.anticipated.state_out_def_rows.len(),
                0,
                "stage {stage_idx}: K=0 must exclude the deposit row entirely"
            );
            assert_eq!(
                layout.anticipated.slot_definition_rows.len(),
                0,
                "stage {stage_idx}: K=0 must exclude every interior carry row"
            );
            assert_eq!(
                layout.geometry.anticipated_decision.len(),
                ctx.study_dims.anticipated_plants.len(),
                "stage {stage_idx}: the decision-column block stays uniformly \
                 n_anticipated wide even when every plant is K=0 (all rows excluded, \
                 the columns are not)"
            );

            let mut row_lower = vec![f64::NAN; layout.rows.num_rows];
            let mut row_upper = vec![f64::NAN; layout.rows.num_rows];
            fill_anticipated_fishing_rows(&layout, &mut row_lower, &mut row_upper);
            fill_anticipated_state_out_def_rows(&layout, &mut row_lower, &mut row_upper);

            let mut col_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); layout.num_cols];
            fill_anticipated_fishing_entries(&ctx, &stage, stage_idx, &layout, &mut col_entries);
            fill_anticipated_state_out_def_entries(stage_idx, &layout, &mut col_entries);

            // The plant's ordinary thermal generation columns carry no entry
            // at all from either anticipated row family — unconstrained by
            // any fishing coupling.
            for blk in 0..layout.clock.n_blks() {
                let col_gen =
                    layout
                        .block_grid()
                        .flat(layout.geometry.thermal.start, 0, BlockIdx::new(blk));
                assert!(
                    col_entries[col_gen].is_empty(),
                    "stage {stage_idx} blk {blk}: thermal generation column must carry no \
                     anticipated coupling entry; got {:?}",
                    col_entries[col_gen]
                );
            }
        }
    }

    /// Three-family collapse: a lead-0 plant's ring is the only
    /// `k_max == 0` state production can build. All three row
    /// families — fishing, deposit, and interior carry — collapse to zero,
    /// and none of the three entry-fill functions panics or writes a
    /// coupling.
    #[test]
    fn empty_ring_collapses_all_three_anticipated_row_families() {
        let mut fixtures = AntFixtures::new();
        fixtures.base.bounds = AntFixtures::bounds_with_n_stages(2, 0, 1);
        let ctx = fixtures.make_ctx(0, vec![0], &[0], 1);
        let stage = two_block_stage(0, [372.0, 372.0]);
        assert_eq!(
            ctx.state.n_anticipated, 1,
            "fixture sanity: one anticipated plant"
        );
        assert_eq!(
            ctx.state.k_max, 0,
            "fixture sanity: a lead-0 plant's ring is empty"
        );
        let layout = StageLayout::new(&ctx, &stage, 0);

        assert_eq!(
            layout.anticipated.fishing_rows.len(),
            0,
            "k_max == 0 must collapse the fishing-row family to zero"
        );
        assert_eq!(
            layout.anticipated.state_out_def_rows.len(),
            0,
            "k_max == 0 must collapse the deposit-row family to zero"
        );
        assert_eq!(
            layout.anticipated.slot_definition_rows.len(),
            0,
            "k_max == 0 must collapse the interior-carry-row family to zero"
        );

        let mut col_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); layout.num_cols];
        fill_anticipated_fishing_entries(&ctx, &stage, 0, &layout, &mut col_entries);
        fill_anticipated_state_out_def_entries(0, &layout, &mut col_entries);
        fill_anticipated_slot_definition_entries(&layout, &mut col_entries);

        assert!(
            col_entries.iter().all(Vec::is_empty),
            "an empty ring must emit no anticipated coupling entries from any of \
             the three families"
        );
    }

    /// Ring-undersizing collision reproduction: a `LeadTime(350.0)` plant on
    /// the uniform `[100.0; 4]` calendar has a real stage-0 in-flight set of
    /// three deliveries (`decider == [None, None, None, Some(0)]`), but
    /// `AnticipatedResolution::k_max` (derived from `PointResolution::depth`,
    /// which excludes pre-study occupancy) resolves a ring only `1` slot
    /// deep. The interior carry row (delivery 1, pre-study, still in flight)
    /// and the deposit row (delivery 3, decided this stage) both address
    /// modular slot `m % k_max == 0`, so both write the SAME outgoing column
    /// — a definition-row collision no column in the ring's outgoing block
    /// may exhibit.
    #[test]
    fn deposit_and_carry_never_share_an_outgoing_column() {
        let mut fixtures = AntFixtures::new();
        fixtures.base.bounds = AntFixtures::bounds_with_n_stages(4, 3, 1);
        fixtures.base.anticipated_resolution = AnticipatedResolution::resolve(
            &[LeadTime::Time(350.0)],
            DeliveryAxis {
                study_stage_hours: &[100.0; 4],
                post_study_stage_hours: &[],
            },
        );
        let ctx = fixtures.make_ctx(3, vec![3], &[0], 1);
        let stage = two_block_stage(0, [372.0, 372.0]);
        let layout = StageLayout::new(&ctx, &stage, 0);

        let mut col_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); layout.num_cols];
        fill_anticipated_slot_definition_entries(&layout, &mut col_entries);
        fill_anticipated_state_out_def_entries(0, &layout, &mut col_entries);

        let out_start = layout.state.commit_out.start;
        let n_ant_state = layout.state.n_anticipated * layout.state.k_max;
        for (offset, entries) in col_entries[out_start..out_start + n_ant_state]
            .iter()
            .enumerate()
        {
            let col = out_start + offset;
            let mut rows: Vec<usize> = entries.iter().map(|(r, _)| *r).collect();
            rows.sort_unstable();
            rows.dedup();
            assert!(
                rows.len() <= 1,
                "outgoing column {col} carries entries from {} distinct rows ({rows:?}); \
                 expected at most 1",
                rows.len()
            );
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Ring-axis excision: the deposit latch resolves through `ring_index`
    // ─────────────────────────────────────────────────────────────────────────

    /// Byte-identity anchor: with no fixed post-horizon window `ring_index` is
    /// the identity, so the deposit's ring-axis residue equals the raw-axis
    /// residue `delivery_stage % k_max` — unchanged from before the excision.
    /// Two-plant fixture (`K = [3, 4]`, `k_max = 4`, no attached
    /// `AnticipatedResolution`, so `anticipated_resolution_for` falls back to
    /// the on-the-fly constant-lead resolution, whose `n_delivery == n_decision`
    /// makes `ring_index` the identity everywhere): plant 0's decision at
    /// `stage_idx = 0` targets `delivery_stage = 3` (`3 < k_max`, so the raw and
    /// ring-axis residues coincide without wrapping).
    #[test]
    fn anticipated_deposit_targets_the_raw_residue_on_an_identity_axis() {
        let mut fixtures = AntFixtures::new();
        fixtures.base.bounds = AntFixtures::bounds_with_n_stages(8, 4, 0);
        let ctx = fixtures.make_ctx(4, vec![3, 4], &[0, 1], 0);
        let stage = two_block_stage(0, [372.0, 372.0]);
        let layout = StageLayout::new(&ctx, &stage, 0);

        let mut col_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); layout.num_cols];
        fill_anticipated_state_out_def_entries(0, &layout, &mut col_entries);

        let row = layout.anticipated.state_out_def_rows.start;
        let slot = 3_usize;
        let col_state_out = DeliveryRing::anticipated(layout.state).out_col(slot, 0);
        let col_decision = layout.geometry.anticipated_decision.start;

        assert!(
            col_entries[col_state_out]
                .iter()
                .any(|&(r, v)| r == row && (v - 1.0).abs() < 1e-15),
            "plant 0: expected (+1.0) entry at (col_state_out={col_state_out}, row={row}), \
             got {:?}",
            col_entries[col_state_out]
        );
        assert!(
            col_entries[col_decision]
                .iter()
                .any(|&(r, v)| r == row && (v + 1.0).abs() < 1e-15),
            "plant 0: expected (-1.0) entry at (col_decision={col_decision}, row={row}), \
             got {:?}",
            col_entries[col_decision]
        );
    }

    /// Plant 0's resolution for the ring-axis excision tests below: `k_max =
    /// 4`, `decision_sets.len() == 4` (`n_decision`), a 3-wide excised fixed
    /// post-horizon window `decider[4..7) == None` (`g == 3`), and a genuine
    /// decision at `stage_idx = 0` targeting `delivery_stage = 7`
    /// (`decider[7] == Some(0)`). `ring_index(7) == Some(4)` (`n_decision + g
    /// == 7`), so the deposit's ring-axis residue is `4 % 4 == 0` — not the
    /// raw-axis residue `7 % 4 == 3`, the slot the stage-3 seed (`decider[3]
    /// == None`, still in flight) already carries at.
    fn plant0_excised_window_g3_resolution() -> AnticipatedResolution {
        let point = PointResolution {
            decider: vec![
                None,
                None,
                None,
                None, // m = 0..3 (in-study, pre-study seeds)
                None,
                None,
                None, // m = 4..6 (excised window, g = 3)
                Some(0),
                Some(1),
                Some(2),
                Some(3), // m = 7..10
            ],
            decision_sets: vec![vec![7], vec![8], vec![9], vec![10]],
            depth: vec![1, 2, 3, 4],
            occupancy: vec![4, 4, 4, 4],
        };
        AnticipatedResolution {
            per_plant: vec![point],
        }
    }

    /// The deposit's slot resolves through the ring axis, not the raw
    /// delivery axis, across [`plant0_excised_window_g3_resolution`]'s excised
    /// window.
    #[test]
    fn anticipated_deposit_targets_the_ring_axis_residue_across_an_excised_window() {
        let mut fixtures = AntFixtures::new();
        fixtures.base.bounds = AntFixtures::bounds_with_n_stages(4, 4, 0);
        fixtures.base.anticipated_resolution = plant0_excised_window_g3_resolution();
        let mut ctx = fixtures.make_ctx(4, vec![4], &[0], 0);
        let time_value = TimeValue::from_parts(
            vec![],
            vec![1.0; 11],
            vec![744.0; 11],
            (0..11).collect(),
            PostStudyResolved::default(),
        );
        ctx.time_value = &time_value;
        let stage = two_block_stage(0, [372.0, 372.0]);
        let layout = StageLayout::new(&ctx, &stage, 0);

        let mut col_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); layout.num_cols];
        fill_anticipated_state_out_def_entries(0, &layout, &mut col_entries);

        let row = layout.anticipated.state_out_def_rows.start;
        let col_decision = layout.geometry.anticipated_decision.start;
        let ring = DeliveryRing::anticipated(layout.state);
        let col_slot0 = ring.out_col(0, 0);
        let col_slot3 = ring.out_col(3, 0);

        assert!(
            col_entries[col_slot0]
                .iter()
                .any(|&(r, v)| r == row && (v - 1.0).abs() < 1e-15),
            "expected (+1.0) entry at (col_slot0={col_slot0}, row={row}), got {:?}",
            col_entries[col_slot0]
        );
        assert!(
            col_entries[col_decision]
                .iter()
                .any(|&(r, v)| r == row && (v + 1.0).abs() < 1e-15),
            "expected (-1.0) entry at (col_decision={col_decision}, row={row}), got {:?}",
            col_entries[col_decision]
        );
        assert!(
            col_entries[col_slot3].iter().all(|&(r, _)| r != row),
            "the deposit row must not land on slot 3 — the slot the stage-3 seed holds; \
             got {:?}",
            col_entries[col_slot3]
        );
    }

    /// With both families filled for the same excised-window fixture, no
    /// outgoing ring column carries entries from two distinct definition rows
    /// — the collision the ring-axis excision removes.
    #[test]
    fn anticipated_deposit_and_carry_never_share_an_outgoing_column() {
        let mut fixtures = AntFixtures::new();
        fixtures.base.bounds = AntFixtures::bounds_with_n_stages(4, 4, 0);
        fixtures.base.anticipated_resolution = plant0_excised_window_g3_resolution();
        let mut ctx = fixtures.make_ctx(4, vec![4], &[0], 0);
        let time_value = TimeValue::from_parts(
            vec![],
            vec![1.0; 11],
            vec![744.0; 11],
            (0..11).collect(),
            PostStudyResolved::default(),
        );
        ctx.time_value = &time_value;
        let stage = two_block_stage(0, [372.0, 372.0]);
        let layout = StageLayout::new(&ctx, &stage, 0);

        let mut col_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); layout.num_cols];
        fill_anticipated_slot_definition_entries(&layout, &mut col_entries);
        fill_anticipated_state_out_def_entries(0, &layout, &mut col_entries);

        let out_start = layout.state.commit_out.start;
        let n_ant_state = layout.state.n_anticipated * layout.state.k_max;
        for (offset, entries) in col_entries[out_start..out_start + n_ant_state]
            .iter()
            .enumerate()
        {
            let col = out_start + offset;
            let mut rows: Vec<usize> = entries.iter().map(|(r, _)| *r).collect();
            rows.sort_unstable();
            rows.dedup();
            assert!(
                rows.len() <= 1,
                "outgoing column {col} carries entries from {} distinct rows ({rows:?}); \
                 expected at most 1",
                rows.len()
            );
        }
    }

    /// `fill_anticipated_fishing_entries` is unchanged by the excision — its
    /// `stage_idx` is in-study by construction, so `ring_index` is the
    /// identity there. Reuses
    /// [`plant0_excised_window_g3_resolution`]'s `g == 3` fixture at
    /// `stage_idx = 2` (in-study), where the maturing slot is
    /// `2 % k_max == 2`, unaffected by the plant's post-study excised window.
    #[test]
    fn anticipated_fishing_slot_is_unchanged_by_an_excised_window() {
        let mut fixtures = AntFixtures::new();
        fixtures.base.bounds = AntFixtures::bounds_with_n_stages(4, 4, 1);
        fixtures.base.anticipated_resolution = plant0_excised_window_g3_resolution();
        let mut ctx = fixtures.make_ctx(4, vec![4], &[0], 1);
        let time_value = TimeValue::from_parts(
            vec![],
            vec![1.0; 11],
            vec![744.0; 11],
            (0..11).collect(),
            PostStudyResolved::default(),
        );
        ctx.time_value = &time_value;
        let stage = two_block_stage(2, [372.0, 372.0]);
        let layout = StageLayout::new(&ctx, &stage, 2);

        let mut col_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); layout.num_cols];
        fill_anticipated_fishing_entries(&ctx, &stage, 2, &layout, &mut col_entries);

        let row = layout.anticipated.fishing_rows.start;
        let col_in_slot2 = layout.state.commit_in.start + 2;
        let block_hours_total: f64 = stage.blocks.iter().map(|b| b.duration_hours).sum();
        let expected_neg = -block_hours_total;

        let couplings: Vec<&(usize, f64)> = col_entries[col_in_slot2]
            .iter()
            .filter(|(r, _)| *r == row)
            .collect();
        assert_eq!(
            couplings.len(),
            1,
            "in_col(2, plant 0) must carry exactly 1 coupling at fishing row {row}; got {:?}",
            col_entries[col_in_slot2]
        );
        let (_, coeff) = couplings[0];
        assert!(
            (coeff - expected_neg).abs() < 1e-12,
            "in_col(2, plant 0) fishing-row coupling: expected {expected_neg}, got {coeff}"
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // State-fixing diagonals must be absent from the CSC output
    // ─────────────────────────────────────────────────────────────────────────

    /// Asserts `build_stage_matrix_entries` produces no state-fixing
    /// diagonals in the CSC output.
    ///
    /// Coverage strategy: storage-fixing and lag-fixing diagonals are absent
    /// by construction, because no loop in `fill_state_and_water_entries`
    /// emits them. Anticipated-state-fixing diagonals are checked dynamically:
    /// the test builds a fixture with `n_anticipated = 2, k_max = 3` and
    /// asserts every `(slot, plant)` column at
    /// `state.commit_in.start + slot*A + plant` has no entry at
    /// row `slot*A + plant` (the diagonal entry that existed in the
    /// pre-cutover layout, before state pinning moved to column bounds).
    ///
    /// The storage/lag assertions are included as zero-iteration loops in
    /// this fixture (`n_hydros` = 0) so the test documents the intent and
    /// would catch a future regression in any fixture that adds hydros.
    #[test]
    fn state_fixing_diagonals_absent_from_csc() {
        let (mut fixtures, stage) = build_anticipated_ctx_n_stages_6();
        let ctx = fixtures.make_ctx(
            3,          // k_max
            vec![2, 3], // anticipated_lead_stages
            &[0, 1],    // anticipated_positions
            2,          // n_thermals: must cover thermal indices 0 and 1 so the
                        // fishing-row entry resolves to a real thermal column.
        );

        let layout = StageLayout::new(&ctx, &stage, 0);
        let col_entries = build_stage_matrix_entries(&ctx, &stage, 0, &layout);

        let a = ctx.study_dims.anticipated_plants.len();
        let k = ctx
            .state
            .anticipated_resolution
            .ring_size(&ctx.state.anticipated_lead_stages);
        for slot in 0..k {
            for plant in 0..a {
                let col = layout.state.commit_in.start + slot * a + plant;
                let diag_row = slot * a + plant;
                let has_diag = col_entries[col]
                    .iter()
                    .any(|&(r, v)| r == diag_row && (v - 1.0).abs() < 1e-15);
                assert!(
                    !has_diag,
                    "anticipated-state-fixing diagonal (row={diag_row}, val=1.0) must be absent \
                     from col {col} (slot={slot}, plant={plant})"
                );
            }
        }

        // Storage-fixing and lag-fixing diagonal absence assertions. With
        // n_hydros = 0 in this fixture these loops execute zero iterations,
        // but the structure documents intent and the same assertion shape
        // catches a regression in any future fixture with non-zero hydros.
        let n_h = ctx.hydros.len();
        let lag_order = ctx.par_lp.max_order();
        for h in 0..n_h {
            let col = layout.state.storage_in.start + h;
            let has_diag = col_entries[col]
                .iter()
                .any(|&(r, v)| r == h && (v - 1.0).abs() < 1e-15);
            assert!(
                !has_diag,
                "storage-fixing diagonal (row={h}, val=1.0) must be absent from col {col}"
            );
        }
        for lag in 0..lag_order {
            for h in 0..n_h {
                let col = layout.state.inflow_lags.start + lag * n_h + h;
                let diag_row = n_h + lag * n_h + h;
                let has_diag = col_entries[col]
                    .iter()
                    .any(|&(r, v)| r == diag_row && (v - 1.0).abs() < 1e-15);
                assert!(
                    !has_diag,
                    "lag-fixing diagonal (row={diag_row}, val=1.0) must be absent from col {col} \
                     (lag={lag}, h={h})"
                );
            }
        }
    }
}

#[cfg(test)]
#[expect(
    clippy::similar_names,
    clippy::too_many_lines,
    reason = "test locals mirror the paired column and row names the assertions compare, and the fixture spells out one complete study inline so each assertion traces to a literal"
)]
mod pumping_water_tests {
    use std::collections::{BTreeMap, HashMap};

    use cobre_core::{
        BlockMode, BoundsCountsSpec, BoundsDefaults, Bus, BusStagePenalties, CascadeTopology,
        CoefficientRef, ConstraintExpression, ContractBlockBounds, ContractType, DeficitSegment,
        EnergyContract, EntityId, GenericConstraint, Hydro, HydroBlockBounds, HydroGenerationModel,
        HydroPenalties, HydroStageBounds, HydroUnitGroup, Line, LineBlockBounds,
        LineStagePenalties, LinearTerm, NcsStagePenalties, PenaltiesCountsSpec, PenaltiesDefaults,
        PumpingBlockBounds, PumpingStation, ResolvedBounds, ResolvedGenericConstraintBounds,
        ResolvedPenalties, SlackConfig, Stage, Thermal, ThermalBlockBounds, ThermalStageBounds,
        VariableRef,
    };
    use cobre_stochastic::par::precompute::PrecomputedPar;

    use crate::block_clock::{BlockClock, M3S_TO_HM3};
    use crate::hydro_models::{
        EvaporationModel, EvaporationModelSet, FphaPlane, ProductionModelSet,
        ResolvedProductionModel,
    };
    use crate::indexer::{
        AnticipatedPlants, BlockIdx, Boundary, BusSys, EvapLocal, FphaCellLocal, HydroCell,
        HydroCellIndex, HydroSys, LineSys, StateDim,
    };
    use crate::lead_time::{SpreadResolution, resolve_spread};

    use crate::test_support::ctx_fixture::CtxFixture;
    use crate::test_support::make_unit_group;
    use crate::time_value::{PostStudyResolved, TimeValue};

    use super::super::columns::{fill_pumping_columns, fill_stage_columns};
    use super::super::delivery_ring::ColumnBufs;
    use super::super::layout::{StageLayout, TemplateBuildCtx};
    use super::super::rows::fill_stage_rows;
    use super::super::test_support::{
        BLOCK_HOURS, three_block_stage, two_block_stage, zero_hydro_penalties,
    };
    use super::{
        LpMatrixBuffers, assemble_csc, build_stage_matrix_entries, fill_fpha_entries,
        fill_generic_constraint_entries, fill_load_balance_entries,
        fill_operational_violation_entries, fill_pumping_water_entries,
        fill_transit_bucket_definition_entries, resolve_bucket_arrival_density,
    };

    const N_STAGES: usize = 1;

    /// Minimal independent (no-downstream) constant-productivity hydro.
    fn fixture_hydro(id: i32) -> Hydro {
        fixture_hydro_ds(id, None)
    }

    /// `fixture_hydro` with a caller-chosen `downstream_id`, so a two-reservoir
    /// cascade can be built from the fixture hydros. `fixture_hydro` delegates
    /// here with `None`, keeping every existing caller's cascade empty.
    fn fixture_hydro_ds(id: i32, downstream_id: Option<i32>) -> Hydro {
        let mut hydro = Hydro {
            unit_groups: Vec::new(),
            id: EntityId(id),
            name: format!("H{id}"),
            operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            downstream_id: downstream_id.map(EntityId),
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

    fn default_bounds_defaults() -> BoundsDefaults {
        BoundsDefaults {
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
        }
    }

    /// A single bus with one unbounded deficit segment, on `EntityId(1)` (the bus
    /// the fixture hydros and `station` helper reference).
    fn fixture_bus(id: i32) -> Bus {
        Bus {
            id: EntityId(id),
            name: format!("B{id}"),
            operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 1000.0,
            }],
            excess_cost: 1.0,
        }
    }

    /// A pumping station on `EntityId(1)` with explicit source/destination/flow
    /// data and the default `consumption_mw_per_m3s` (0.5).
    fn station(
        id: i32,
        source: i32,
        destination: i32,
        min_flow: f64,
        max_flow: f64,
    ) -> PumpingStation {
        station_full(id, source, destination, min_flow, max_flow, 1, 0.5)
    }

    /// A pumping station with an explicit `bus_id` and consumption rate so the
    /// power-coupling tests can place a station on an unmapped bus and observe a
    /// distinct coefficient.
    fn station_full(
        id: i32,
        source: i32,
        destination: i32,
        min_flow: f64,
        max_flow: f64,
        bus_id: i32,
        consumption_mw_per_m3s: f64,
    ) -> PumpingStation {
        PumpingStation {
            id: EntityId(id),
            name: format!("P{id}"),
            operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId(bus_id),
            source_hydro_id: EntityId(source),
            destination_hydro_id: EntityId(destination),
            entry_stage_id: None,
            exit_stage_id: None,
            consumption_mw_per_m3s,
            min_flow_m3s: min_flow,
            max_flow_m3s: max_flow,
        }
    }

    /// A bus with a caller-chosen deficit-segment count and excess cost so the
    /// multi-entity permutation test can give each bus a distinct CSC footprint:
    /// `fill_load_balance_entries` emits one deficit column per
    /// `deficit_segments.len()`, so distinct segment counts make the deficit
    /// column block bus-position-dependent — a bus permutation that escaped the
    /// ID-sort would land the wrong segment count in the wrong column block.
    fn fixture_bus_with(id: i32, n_segments: usize, excess_cost: f64) -> Bus {
        Bus {
            id: EntityId(id),
            name: format!("B{id}"),
            operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            deficit_segments: (0..n_segments)
                .map(|_| DeficitSegment {
                    depth_mw: None,
                    cost_per_mwh: 1000.0,
                })
                .collect(),
            excess_cost,
        }
    }

    /// A thermal plant on `bus_id` with distinct generation bounds and cost so a
    /// permutation that mislabelled which thermal owns which column is observable.
    fn fixture_thermal(id: i32, bus_id: i32, min_gen: f64, max_gen: f64, cost: f64) -> Thermal {
        Thermal {
            id: EntityId(id),
            name: format!("T{id}"),
            operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId(bus_id),
            entry_stage_id: None,
            exit_stage_id: None,
            cost_per_mwh: cost,
            min_generation_mw: min_gen,
            max_generation_mw: max_gen,
            anticipated_config: None,
        }
    }

    /// A transmission line between `source_bus`/`target_bus` with distinct
    /// capacities. Distinct bus pairs across lines make the load-balance fill
    /// (±1.0 into source/target bus rows) line-position- and bus-position-
    /// dependent, so a permutation that escaped either ID-sort changes the CSC.
    fn fixture_line(id: i32, source_bus: i32, target_bus: i32, direct: f64, reverse: f64) -> Line {
        Line {
            id: EntityId(id),
            name: format!("L{id}"),
            operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            source_bus_id: EntityId(source_bus),
            target_bus_id: EntityId(target_bus),
            entry_stage_id: None,
            exit_stage_id: None,
            direct_capacity_mw: direct,
            reverse_capacity_mw: reverse,
            losses_percent: 0.0,
            exchange_cost: 0.0,
        }
    }

    /// Owns the data backing a two-hydro `TemplateBuildCtx` carrying pumping
    /// stations, thermals, and lines. Every entity slice is stored in canonical
    /// (ID-sorted) order; the position maps are derived from those sorted slices,
    /// exactly as `SystemBuilder::build` produces them in production. The ID-sort
    /// is what makes two declaration orders of the same entities converge to one
    /// ctx — the property the permutation tests assert.
    struct PumpFixtures {
        base: CtxFixture,
        /// Hydro id → sorted-slice position, mirroring `base.hydros`. Kept
        /// alongside `base` (rather than read off a built [`TemplateBuildCtx`])
        /// since many tests index it directly to address `arc_stage_weights`/
        /// water-balance fixtures before ever calling [`Self::make_ctx`].
        hydro_pos: BTreeMap<EntityId, usize>,
    }

    impl PumpFixtures {
        /// Build a fixture with a single bus (`EntityId(1)`) — the bus the fixture
        /// hydros and the `station` helper reference — so the load-balance row is
        /// present and the pumping-power coupling is exercised.
        fn new(hydros: Vec<Hydro>, stations: Vec<PumpingStation>) -> Self {
            Self::new_with_buses(hydros, stations, vec![fixture_bus(1)])
        }

        /// Build a fixture from hydros, stations, and buses supplied in arbitrary
        /// declaration order, with no thermals or lines. Delegates to
        /// [`PumpFixtures::new_full`], which sorts every slice by `id.0` (the
        /// canonical operation `SystemBuilder::build` performs) before deriving
        /// position maps, so the resulting ctx is declaration-order-invariant.
        fn new_with_buses(
            hydros: Vec<Hydro>,
            stations: Vec<PumpingStation>,
            buses: Vec<Bus>,
        ) -> Self {
            Self::new_full(hydros, stations, buses, Vec::new(), Vec::new())
        }

        /// Build a fixture with buses and energy contracts (no thermals/lines),
        /// for the contract load-balance-sign tests.
        fn new_with_contracts(
            hydros: Vec<Hydro>,
            buses: Vec<Bus>,
            contracts: Vec<EnergyContract>,
        ) -> Self {
            Self::new_full_with_contracts(
                hydros,
                Vec::new(),
                buses,
                Vec::new(),
                Vec::new(),
                contracts,
            )
        }

        /// Build a fixture carrying hydros, stations, buses, thermals, and lines in
        /// arbitrary declaration order. Every slice is sorted by `id.0` before
        /// deriving its position map and bounds table, mirroring
        /// `SystemBuilder::build`; this canonicalisation is what makes two
        /// declaration orders of the same entities converge to one ctx (the
        /// invariant the permutation tests assert). Thermal and line bounds are
        /// written per-entity from the sorted slices so a column/bound mislabel
        /// under permutation is observable.
        fn new_full(
            hydros: Vec<Hydro>,
            stations: Vec<PumpingStation>,
            buses: Vec<Bus>,
            thermals: Vec<Thermal>,
            lines: Vec<Line>,
        ) -> Self {
            Self::new_full_with_contracts(hydros, stations, buses, thermals, lines, Vec::new())
        }

        fn new_full_with_contracts(
            mut hydros: Vec<Hydro>,
            mut stations: Vec<PumpingStation>,
            mut buses: Vec<Bus>,
            mut thermals: Vec<Thermal>,
            mut lines: Vec<Line>,
            mut contracts: Vec<EnergyContract>,
        ) -> Self {
            hydros.sort_by_key(|h| h.id.0);
            stations.sort_by_key(|s| s.id.0);
            buses.sort_by_key(|b| b.id.0);
            thermals.sort_by_key(|t| t.id.0);
            lines.sort_by_key(|l| l.id.0);
            contracts.sort_by_key(|c| c.id.0);

            let hydro_pos: BTreeMap<EntityId, usize> =
                hydros.iter().enumerate().map(|(i, h)| (h.id, i)).collect();

            let mut bounds = ResolvedBounds::new(
                &BoundsCountsSpec {
                    n_hydros: hydros.len(),
                    n_thermals: thermals.len(),
                    n_lines: lines.len(),
                    n_pumping: stations.len(),
                    n_contracts: contracts.len(),
                    n_stages: N_STAGES,
                    k_max: 0,
                },
                &default_bounds_defaults(),
            );
            // Distinct per-contract bounds from the sorted slice so a column/bound
            // mislabel under permutation is observable.
            for (c_idx, c) in contracts.iter().enumerate() {
                for stage_idx in 0..N_STAGES {
                    *bounds.contract_bounds_mut(c_idx, stage_idx) = ContractBlockBounds {
                        min_mw: c.min_mw,
                        max_mw: c.max_mw,
                        price_per_mwh: c.price_per_mwh,
                    };
                }
            }
            // Distinct per-station bounds so a column/bound mismatch is observable.
            for (p_idx, s) in stations.iter().enumerate() {
                for stage_idx in 0..N_STAGES {
                    *bounds.pumping_bounds_mut(p_idx, stage_idx) = PumpingBlockBounds {
                        min_flow_m3s: s.min_flow_m3s,
                        max_flow_m3s: s.max_flow_m3s,
                    };
                }
            }
            // Distinct per-thermal generation bounds and cost, taken from the
            // sorted slice, so a permutation that mislabelled which thermal owns
            // which column would change the resolved bounds.
            for (t_idx, t) in thermals.iter().enumerate() {
                for stage_idx in 0..N_STAGES {
                    *bounds.thermal_bounds_mut(t_idx, stage_idx) = ThermalStageBounds {
                        cost_per_mwh: t.cost_per_mwh,
                    };
                    *bounds.thermal_block_base_mut(t_idx, stage_idx) = ThermalBlockBounds {
                        min_generation_mw: t.min_generation_mw,
                        max_generation_mw: t.max_generation_mw,
                    };
                }
            }
            // Distinct per-line capacities from the sorted slice, same rationale.
            for (l_idx, l) in lines.iter().enumerate() {
                for stage_idx in 0..N_STAGES {
                    *bounds.line_bounds_mut(l_idx, stage_idx) = LineBlockBounds {
                        direct_mw: l.direct_capacity_mw,
                        reverse_mw: l.reverse_capacity_mw,
                    };
                }
            }

            let production_models = ProductionModelSet::new(
                vec![
                    vec![
                        ResolvedProductionModel::ConstantProductivity { productivity: 1.0 };
                        N_STAGES
                    ];
                    hydros.len()
                ],
                &hydros,
                N_STAGES,
            );
            let evaporation_models =
                EvaporationModelSet::new(vec![EvaporationModel::None; hydros.len()]);

            // Built from the sorted hydros so a fixture carrying `downstream_id`
            // produces a real cascade; output-neutral for the `fixture_hydro`
            // callers, whose `downstream_id: None` yields the same empty cascade
            // `CascadeTopology::build(&[])` did.
            let cascade = CascadeTopology::build(&hydros);
            let hydro_cell_index = HydroCellIndex::build(&hydros);
            let anticipated_plants = AnticipatedPlants::build(&thermals);

            Self {
                base: CtxFixture {
                    hydros,
                    hydro_cell_index,
                    pumping_stations: stations,
                    buses,
                    thermals,
                    lines,
                    contracts,
                    cascade,
                    bounds,
                    production_models,
                    evaporation_models,
                    time_value: TimeValue::from_parts(
                        vec![],
                        vec![1.0; N_STAGES],
                        vec![744.0; N_STAGES],
                        (0..N_STAGES as i32).collect(),
                        PostStudyResolved::default(),
                    ),
                    anticipated_plants,
                    ..CtxFixture::default()
                },
                hydro_pos,
            }
        }

        /// Attach a generic constraint (and its active-at-stage-0 upper-only bound)
        /// so the LP builder resolves the constraint's expression against the
        /// pumping columns. Used by the end-to-end resolver-integration test.
        fn with_generic_constraint(
            mut self,
            constraint: GenericConstraint,
            bound_upper: f64,
        ) -> Self {
            let constraint_id = constraint.id.0;
            let id_map: HashMap<i32, usize> = [(constraint_id, 0)].into_iter().collect();
            let rows =
                (0..N_STAGES).map(|s| (constraint_id, s as i32, None, None, Some(bound_upper)));
            self.base.resolved_generic_bounds =
                ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter());
            self.base.generic_constraints = vec![constraint];
            self
        }

        /// Like [`Self::with_generic_constraint`], but sets BOTH endpoints — a
        /// two-sided row, whose slack allocation and net-report path differ from
        /// the one-sided row above.
        fn with_generic_constraint_range(
            mut self,
            constraint: GenericConstraint,
            bound_lower: f64,
            bound_upper: f64,
        ) -> Self {
            let constraint_id = constraint.id.0;
            let id_map: HashMap<i32, usize> = [(constraint_id, 0)].into_iter().collect();
            let rows = (0..N_STAGES).map(|s| {
                (
                    constraint_id,
                    s as i32,
                    None,
                    Some(bound_lower),
                    Some(bound_upper),
                )
            });
            self.base.resolved_generic_bounds =
                ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter());
            self.base.generic_constraints = vec![constraint];
            self
        }

        /// Inject a `PrecomputedPar`, whose order (read via `ctx.par_lp.max_order()`)
        /// sizes the inflow-lag columns so the `−ζ·ψ` AR-lag water term fires. The
        /// default fixture carries `PrecomputedPar::default()` (order 0), under
        /// which the AR-lag term is dormant.
        fn with_par_lp(mut self, par_lp: PrecomputedPar) -> Self {
            self.base.par_lp = par_lp;
            self
        }

        /// Replace the default all-`ConstantProductivity` production model set
        /// (one entry per hydro, matching `self.hydros`' length) with a
        /// caller-supplied one, e.g. an FPHA plant alongside constant-productivity
        /// ones.
        fn with_production_models(mut self, production_models: ProductionModelSet) -> Self {
            self.base.production_models = production_models;
            self
        }

        /// Replace the default empty penalty table with a properly-sized all-zero
        /// (plus a small spillage cost) `ResolvedPenalties` so the FULL template
        /// build (`build_single_stage_template`, which reads per-hydro penalties)
        /// does not index an empty table. The matrix-only tests use
        /// `ResolvedPenalties::empty()` because `build_stage_matrix_entries` never
        /// reads penalties; the solver-backed duals test needs this.
        fn with_resolved_penalties(mut self) -> Self {
            let hydro = HydroPenalties {
                spillage_cost: 0.01,
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
            };
            self.base.penalties = ResolvedPenalties::new(
                &PenaltiesCountsSpec {
                    n_hydros: self.base.hydros.len(),
                    n_buses: self.base.buses.len(),
                    n_lines: self.base.lines.len(),
                    n_ncs: 0,
                    n_stages: N_STAGES,
                },
                &PenaltiesDefaults {
                    hydro,
                    bus: BusStagePenalties { excess_cost: 0.0 },
                    line: LineStagePenalties { exchange_cost: 0.0 },
                    ncs: NcsStagePenalties {
                        curtailment_cost: 0.0,
                    },
                },
            );
            self
        }

        /// Size a `ResolvedPenalties` carrying nonzero directional evaporation
        /// violation costs so the per-block `f_evap_plus`/`f_evap_minus` slack
        /// objectives are observable in the column build.
        fn with_evap_penalties(mut self, neg_cost: f64, pos_cost: f64) -> Self {
            let hydro = HydroPenalties {
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
                evaporation_violation_pos_cost: pos_cost,
                evaporation_violation_neg_cost: neg_cost,
                inflow_nonnegativity_cost: 0.0,
            };
            self.base.penalties = ResolvedPenalties::new(
                &PenaltiesCountsSpec {
                    n_hydros: self.base.hydros.len(),
                    n_buses: self.base.buses.len(),
                    n_lines: self.base.lines.len(),
                    n_ncs: 0,
                    n_stages: N_STAGES,
                },
                &PenaltiesDefaults {
                    hydro,
                    bus: BusStagePenalties { excess_cost: 0.0 },
                    line: LineStagePenalties { exchange_cost: 0.0 },
                    ncs: NcsStagePenalties {
                        curtailment_cost: 0.0,
                    },
                },
            );
            self
        }

        fn make_ctx(&mut self) -> TemplateBuildCtx<'_> {
            // These single-stage fixtures decouple `stage.id` from
            // `stage_idx` (every phase is exercised at `stage_idx = 0` against
            // one bounds row), so the filling window's stage ids all resolve to
            // idx 0. The backward fold reads `stage_zetas[0]` and
            // `hydro_bounds(h, 0)` for every filling stage, matching how each
            // stage is built. Covers ids 0..=8 — wider than any filling window
            // under test (max entry = 4). Recomputed on every call (never cached
            // at construction) so a test's post-construction bounds mutation is
            // reflected.
            self.base.filling_v_target = crate::test_support::build_filling_v_target(
                &self.base.hydros,
                &self.base.bounds,
                &[744.0 * M3S_TO_HM3; N_STAGES],
                &(0..=8_i32).map(|id| (id, 0_usize)).collect(),
            );
            self.base.ctx()
        }
    }

    /// Column bounds = `[min_flow, max_flow]` and zero objective for every
    /// `(station, block)` pumping column.
    #[test]
    fn pumping_columns_get_flow_bounds_and_zero_cost() {
        let stations = vec![station(10, 1, 2, 5.0, 80.0), station(20, 2, 1, 0.0, 30.0)];
        let mut fixtures = PumpFixtures::new(vec![fixture_hydro(1), fixture_hydro(2)], stations);
        let ctx = fixtures.make_ctx();
        let stage = two_block_stage(0, [300.0, 444.0]);
        let layout = StageLayout::new(&ctx, &stage, 0);

        // Lower/upper start at a NaN sentinel so any column the helper fails to
        // bound is visible; objective starts at the production default (0.0), so
        // the post-fill zero assertion proves the helper writes no cost.
        let mut col_lower = vec![f64::NAN; layout.num_cols];
        let mut col_upper = vec![f64::NAN; layout.num_cols];
        let mut objective = vec![0.0_f64; layout.num_cols];
        let mut bufs = ColumnBufs {
            col_lower: &mut col_lower,
            col_upper: &mut col_upper,
            objective: &mut objective,
        };

        fill_pumping_columns(&ctx, &stage, 0, &layout, &mut bufs);

        let n_blks = layout.clock.n_blks();
        for (p_idx, s) in ctx.pumping_stations.iter().enumerate() {
            for blk in 0..n_blks {
                let col = layout.geometry.pumping_flow.start + p_idx * n_blks + blk;
                assert_eq!(
                    bufs.col_lower[col], s.min_flow_m3s,
                    "station {p_idx} blk {blk}: lower bound must be min_flow"
                );
                assert_eq!(
                    bufs.col_upper[col], s.max_flow_m3s,
                    "station {p_idx} blk {blk}: upper bound must be max_flow"
                );
                assert_eq!(
                    bufs.objective[col], 0.0,
                    "station {p_idx} blk {blk}: objective must be zero"
                );
            }
        }
    }

    /// Source water row gains `+tau_h`, destination water row gains `−tau_h`,
    /// with `tau_h == block.duration_hours * M3S_TO_HM3` per block.
    #[test]
    fn pumping_water_entries_source_plus_tau_destination_minus_tau() {
        // Station id 10: source hydro id 1 (pos 0), destination hydro id 2 (pos 1).
        let mut fixtures = PumpFixtures::new(
            vec![fixture_hydro(1), fixture_hydro(2)],
            vec![station(10, 1, 2, 0.0, 50.0)],
        );
        let ctx = fixtures.make_ctx();
        let stage = two_block_stage(0, [300.0, 444.0]);
        let layout = StageLayout::new(&ctx, &stage, 0);

        let mut col_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); layout.num_cols];
        fill_pumping_water_entries(&ctx, &layout, &mut col_entries);

        let n_blks = layout.clock.n_blks();
        let source_pos = ctx.positions.hydro(EntityId(1)).unwrap();
        let dest_pos = ctx.positions.hydro(EntityId(2)).unwrap();
        let row_source = layout.geometry.water_balance.start() + source_pos;
        let row_dest = layout.geometry.water_balance.start() + dest_pos;

        for blk in 0..n_blks {
            let tau_h = stage.blocks[blk].duration_hours * M3S_TO_HM3;
            let col = layout.geometry.pumping_flow.start + blk;
            assert_eq!(
                col_entries[col],
                vec![(row_source, tau_h), (row_dest, -tau_h)],
                "blk {blk}: source +tau_h then destination -tau_h"
            );
        }
    }

    /// On a Chronological stage, each station's per-block entries land on its
    /// own hydro's own block row (`start + hydro_pos * n_blks + blk`), never
    /// `start + hydro_pos`. Declaration order of the hydros and stations must
    /// not change the result.
    #[test]
    fn chronological_pumping_entries_land_on_each_hydros_own_block_rows() {
        let mut fixtures = PumpFixtures::new(
            vec![fixture_hydro(3), fixture_hydro(1), fixture_hydro(2)],
            vec![station(20, 3, 1, 0.0, 30.0), station(10, 2, 3, 0.0, 50.0)],
        );
        let ctx = fixtures.make_ctx();
        let stage = chronological_stage(0, &[300.0, 420.0, 24.0]);
        let layout = StageLayout::new(&ctx, &stage, 0);

        let mut col_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); layout.num_cols];
        fill_pumping_water_entries(&ctx, &layout, &mut col_entries);

        let n_blks = layout.clock.n_blks();
        let s_row = layout.geometry.water_balance.start();
        for (p_sys, st) in ctx.pumping_stations.iter().enumerate() {
            let src = ctx.positions.hydro(st.source_hydro_id).unwrap();
            let dst = ctx.positions.hydro(st.destination_hydro_id).unwrap();
            for blk in 0..n_blks {
                let tau = stage.blocks[blk].duration_hours * M3S_TO_HM3;
                let col = layout.geometry.pumping_flow.start + p_sys * n_blks + blk;
                assert_eq!(
                    col_entries[col],
                    vec![
                        (s_row + src * n_blks + blk, tau),
                        (s_row + dst * n_blks + blk, -tau),
                    ],
                    "station {p_sys} blk {blk}: must land on each hydro's own block row"
                );
            }
        }

        let mut reordered = PumpFixtures::new(
            vec![fixture_hydro(1), fixture_hydro(2), fixture_hydro(3)],
            vec![station(10, 2, 3, 0.0, 50.0), station(20, 3, 1, 0.0, 30.0)],
        );
        let reordered_ctx = reordered.make_ctx();
        let reordered_layout = StageLayout::new(&reordered_ctx, &stage, 0);
        let mut reordered_entries: Vec<Vec<(usize, f64)>> =
            vec![Vec::new(); reordered_layout.num_cols];
        fill_pumping_water_entries(&reordered_ctx, &reordered_layout, &mut reordered_entries);

        assert_eq!(
            col_entries, reordered_entries,
            "declaration order must not change the emitted entries"
        );
    }

    /// The pumped volume shares its block's water-balance row with that
    /// hydro's own spillage column, on both the source and the destination
    /// side, and touches no other water-balance row.
    #[test]
    fn chronological_pumping_column_shares_block_rows_with_its_hydros_spillage() {
        let mut fixtures = PumpFixtures::new(
            vec![fixture_hydro(3), fixture_hydro(1), fixture_hydro(2)],
            vec![station(20, 3, 1, 0.0, 30.0), station(10, 2, 3, 0.0, 50.0)],
        );
        let ctx = fixtures.make_ctx();
        let stage = chronological_stage(0, &[300.0, 420.0, 24.0]);
        let layout = StageLayout::new(&ctx, &stage, 0);
        let csc = build_sorted_csc(&ctx, &stage, 0, &layout);

        let n_blks = layout.clock.n_blks();
        let s_row = layout.geometry.water_balance.start();
        for (p_sys, st) in ctx.pumping_stations.iter().enumerate() {
            let src = ctx.positions.hydro(st.source_hydro_id).unwrap();
            let dst = ctx.positions.hydro(st.destination_hydro_id).unwrap();
            for blk in 0..n_blks {
                let tau = stage.blocks[blk].duration_hours * M3S_TO_HM3;
                let blk_idx = BlockIdx::new(blk);
                let pump_col = layout.geometry.pumping_flow.start + p_sys * n_blks + blk;
                let r_src = i32::try_from(s_row + src * n_blks + blk).unwrap();
                let r_dst = i32::try_from(s_row + dst * n_blks + blk).unwrap();

                assert_eq!(
                    coeff_at(&csc, pump_col, r_src),
                    tau,
                    "station {p_sys} blk {blk}: pumping column must carry +tau on the source's own block row"
                );
                assert_eq!(
                    coeff_at(
                        &csc,
                        layout.geometry.spillage_col(HydroSys::new(src), blk_idx),
                        r_src
                    ),
                    tau,
                    "station {p_sys} blk {blk}: the source's own spillage must share the same row"
                );
                assert_eq!(
                    coeff_at(&csc, pump_col, r_dst),
                    -tau,
                    "station {p_sys} blk {blk}: pumping column must carry -tau on the destination's own block row"
                );
                assert_eq!(
                    coeff_at(
                        &csc,
                        layout.geometry.spillage_col(HydroSys::new(dst), blk_idx),
                        r_dst
                    ),
                    tau,
                    "station {p_sys} blk {blk}: the destination's own spillage must share the same row"
                );

                for row in layout.geometry.water_balance.range() {
                    let row_i32 = i32::try_from(row).unwrap();
                    if row_i32 == r_src || row_i32 == r_dst {
                        continue;
                    }
                    assert_eq!(
                        coeff_at(&csc, pump_col, row_i32),
                        0.0,
                        "station {p_sys} blk {blk}: pumping column must carry no entry on row {row}"
                    );
                }
            }
        }
    }

    /// A station whose `source_hydro_id` is absent from `hydro_pos` skips only
    /// the source entry — the destination side is still written, no panic.
    #[test]
    fn pumping_water_entries_missing_source_skips_only_source() {
        // Source hydro id 99 does NOT exist; destination hydro id 2 (pos 1) does.
        let mut fixtures = PumpFixtures::new(
            vec![fixture_hydro(1), fixture_hydro(2)],
            vec![station(10, 99, 2, 0.0, 50.0)],
        );
        let ctx = fixtures.make_ctx();
        let stage = two_block_stage(0, [300.0, 444.0]);
        let layout = StageLayout::new(&ctx, &stage, 0);

        let mut col_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); layout.num_cols];
        fill_pumping_water_entries(&ctx, &layout, &mut col_entries);

        let n_blks = layout.clock.n_blks();
        let dest_pos = ctx.positions.hydro(EntityId(2)).unwrap();
        let row_dest = layout.geometry.water_balance.start() + dest_pos;
        for blk in 0..n_blks {
            let tau_h = stage.blocks[blk].duration_hours * M3S_TO_HM3;
            let col = layout.geometry.pumping_flow.start + blk;
            assert_eq!(
                col_entries[col],
                vec![(row_dest, -tau_h)],
                "blk {blk}: only the destination -tau_h entry survives"
            );
        }
    }

    /// A station whose `destination_hydro_id` is absent skips only the
    /// destination entry — the source side is still written, no panic.
    #[test]
    fn pumping_water_entries_missing_destination_skips_only_destination() {
        // Source hydro id 1 (pos 0) exists; destination hydro id 99 does NOT.
        let mut fixtures = PumpFixtures::new(
            vec![fixture_hydro(1), fixture_hydro(2)],
            vec![station(10, 1, 99, 0.0, 50.0)],
        );
        let ctx = fixtures.make_ctx();
        let stage = two_block_stage(0, [300.0, 444.0]);
        let layout = StageLayout::new(&ctx, &stage, 0);

        let mut col_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); layout.num_cols];
        fill_pumping_water_entries(&ctx, &layout, &mut col_entries);

        let n_blks = layout.clock.n_blks();
        let source_pos = ctx.positions.hydro(EntityId(1)).unwrap();
        let row_source = layout.geometry.water_balance.start() + source_pos;
        for blk in 0..n_blks {
            let tau_h = stage.blocks[blk].duration_hours * M3S_TO_HM3;
            let col = layout.geometry.pumping_flow.start + blk;
            assert_eq!(
                col_entries[col],
                vec![(row_source, tau_h)],
                "blk {blk}: only the source +tau_h entry survives"
            );
        }
    }

    /// The pumping flow column enters its bus load-balance row with
    /// `−consumption_mw_per_m3s` per block — a negative injection, the same sign a
    /// line carries into its source bus, NOT the `+1.0` of generation.
    #[test]
    fn pumping_power_enters_bus_row_with_negative_consumption() {
        // Station id 10 on bus id 1 (pos 0), consumption 0.75 MW per m³/s.
        let mut fixtures = PumpFixtures::new_with_buses(
            vec![fixture_hydro(1), fixture_hydro(2)],
            vec![station_full(10, 1, 2, 0.0, 50.0, 1, 0.75)],
            vec![fixture_bus(1)],
        );
        let ctx = fixtures.make_ctx();
        let stage = two_block_stage(0, [300.0, 444.0]);
        let layout = StageLayout::new(&ctx, &stage, 0);

        let mut col_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); layout.num_cols];
        fill_load_balance_entries(&ctx, 0, &layout, &mut col_entries);

        let n_blks = layout.clock.n_blks();
        let b_idx = ctx.positions.bus(EntityId(1)).unwrap();
        for blk in 0..n_blks {
            let row = layout.geometry.load_balance.start() + b_idx * n_blks + blk;
            let col = layout.geometry.pumping_flow.start + blk;
            assert!(
                col_entries[col].contains(&(row, -0.75)),
                "blk {blk}: pumping column {col} must carry (row {row}, -0.75); got {:?}",
                col_entries[col]
            );
            // The flow column carries the bus-power coupling and nothing else from
            // the load-balance fill (no positive generation-style entry).
            assert_eq!(
                col_entries[col],
                vec![(row, -0.75)],
                "blk {blk}: pumping column must carry only the negative-injection entry"
            );
        }
    }

    /// An import-only `EnergyContract` on `bus_id` carrying `id`/`bus_id`/
    /// `contract_type`; every other field inert.
    fn contract(id: i32, bus_id: i32, contract_type: ContractType) -> EnergyContract {
        EnergyContract {
            id: EntityId(id),
            name: format!("C{id}"),
            operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId(bus_id),
            contract_type,
            entry_stage_id: None,
            exit_stage_id: None,
            price_per_mwh: 0.0,
            min_mw: 0.0,
            max_mw: 100.0,
        }
    }

    /// An import contract enters its bus load-balance row with `+1.0` per block —
    /// injection into the bus. This sign is independent of the price sign.
    #[test]
    fn contract_import_enters_bus_row_with_plus_one() {
        let mut fixtures = PumpFixtures::new_with_contracts(
            vec![fixture_hydro(1), fixture_hydro(2)],
            vec![fixture_bus(1)],
            vec![contract(10, 1, ContractType::Import)],
        );
        let ctx = fixtures.make_ctx();
        let stage = two_block_stage(0, [300.0, 444.0]);
        let layout = StageLayout::new(&ctx, &stage, 0);

        let mut col_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); layout.num_cols];
        fill_load_balance_entries(&ctx, 0, &layout, &mut col_entries);

        let n_blks = layout.clock.n_blks();
        let b_idx = ctx.positions.bus(EntityId(1)).unwrap();
        for blk in 0..n_blks {
            let row = layout.geometry.load_balance.start() + b_idx * n_blks + blk;
            let col = layout.geometry.contract_import.start + blk;
            assert_eq!(
                col_entries[col],
                vec![(row, 1.0)],
                "blk {blk}: import contract column must carry only (row, +1.0)"
            );
        }
    }

    /// An export contract enters its bus load-balance row with `-1.0` per block —
    /// withdrawal from the bus. Flipping this would make an export feed the bus.
    #[test]
    fn contract_export_enters_bus_row_with_minus_one() {
        let mut fixtures = PumpFixtures::new_with_contracts(
            vec![fixture_hydro(1), fixture_hydro(2)],
            vec![fixture_bus(1)],
            vec![contract(10, 1, ContractType::Export)],
        );
        let ctx = fixtures.make_ctx();
        let stage = two_block_stage(0, [300.0, 444.0]);
        let layout = StageLayout::new(&ctx, &stage, 0);

        let mut col_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); layout.num_cols];
        fill_load_balance_entries(&ctx, 0, &layout, &mut col_entries);

        let n_blks = layout.clock.n_blks();
        let b_idx = ctx.positions.bus(EntityId(1)).unwrap();
        for blk in 0..n_blks {
            let row = layout.geometry.load_balance.start() + b_idx * n_blks + blk;
            let col = layout.geometry.contract_export.start + blk;
            assert_eq!(
                col_entries[col],
                vec![(row, -1.0)],
                "blk {blk}: export contract column must carry only (row, -1.0)"
            );
        }
    }

    /// Mixed import/export at distinct per-family slots land on the right column
    /// bases with the right signs: import at `contract_import.start`, export at
    /// `contract_export.start`.
    #[test]
    fn contract_mixed_import_export_use_per_family_bases() {
        let mut fixtures = PumpFixtures::new_with_contracts(
            vec![fixture_hydro(1), fixture_hydro(2)],
            vec![fixture_bus(1)],
            vec![
                contract(10, 1, ContractType::Import),
                contract(20, 1, ContractType::Export),
            ],
        );
        let ctx = fixtures.make_ctx();
        let stage = two_block_stage(0, [300.0, 444.0]);
        let layout = StageLayout::new(&ctx, &stage, 0);

        let mut col_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); layout.num_cols];
        fill_load_balance_entries(&ctx, 0, &layout, &mut col_entries);

        let n_blks = layout.clock.n_blks();
        let b_idx = ctx.positions.bus(EntityId(1)).unwrap();
        for blk in 0..n_blks {
            let row = layout.geometry.load_balance.start() + b_idx * n_blks + blk;
            let import_col = layout.geometry.contract_import.start + blk;
            let export_col = layout.geometry.contract_export.start + blk;
            assert_eq!(
                col_entries[import_col],
                vec![(row, 1.0)],
                "blk {blk}: import"
            );
            assert_eq!(
                col_entries[export_col],
                vec![(row, -1.0)],
                "blk {blk}: export"
            );
        }
    }

    /// Two imports on one bus address distinct per-family slots: the first
    /// (`family_slot` 0) lands on `contract_import.start + 0*n_blks + blk`, the
    /// second (`family_slot` 1) on `contract_import.start + 1*n_blks + blk`. A
    /// regression to using `c_sys` instead of `family_slot` would collide them.
    #[test]
    fn contract_second_import_uses_family_slot_one() {
        let mut fixtures = PumpFixtures::new_with_contracts(
            vec![fixture_hydro(1), fixture_hydro(2)],
            vec![fixture_bus(1)],
            vec![
                contract(10, 1, ContractType::Import),
                contract(20, 1, ContractType::Import),
            ],
        );
        let ctx = fixtures.make_ctx();
        let stage = two_block_stage(0, [300.0, 444.0]);
        let layout = StageLayout::new(&ctx, &stage, 0);

        let mut col_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); layout.num_cols];
        fill_load_balance_entries(&ctx, 0, &layout, &mut col_entries);

        let n_blks = layout.clock.n_blks();
        let b_idx = ctx.positions.bus(EntityId(1)).unwrap();
        for blk in 0..n_blks {
            let row = layout.geometry.load_balance.start() + b_idx * n_blks + blk;
            let slot0_col = layout.geometry.contract_import.start + blk;
            let slot1_col = layout.geometry.contract_import.start + n_blks + blk;
            assert_eq!(
                col_entries[slot0_col],
                vec![(row, 1.0)],
                "blk {blk}: first import (family_slot 0)"
            );
            assert_eq!(
                col_entries[slot1_col],
                vec![(row, 1.0)],
                "blk {blk}: second import (family_slot 1)"
            );
        }
    }

    /// A contract whose `bus_id` is absent from `bus_pos` writes no load-balance
    /// entry and does not panic.
    #[test]
    fn contract_missing_bus_skips_without_panic() {
        let mut fixtures = PumpFixtures::new_with_contracts(
            vec![fixture_hydro(1), fixture_hydro(2)],
            vec![fixture_bus(1)],
            vec![contract(10, 99, ContractType::Import)],
        );
        let ctx = fixtures.make_ctx();
        let stage = two_block_stage(0, [300.0, 444.0]);
        let layout = StageLayout::new(&ctx, &stage, 0);

        let mut col_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); layout.num_cols];
        fill_load_balance_entries(&ctx, 0, &layout, &mut col_entries);

        let n_blks = layout.clock.n_blks();
        for blk in 0..n_blks {
            let col = layout.geometry.contract_import.start + blk;
            assert!(
                col_entries[col].is_empty(),
                "blk {blk}: contract on an unmapped bus must write no load-balance entry"
            );
        }
    }

    /// AC: a contracts-bearing system with a generic constraint referencing
    /// `ContractImport` assembles its full stage matrix in a debug build with no
    /// `debug_assert!` firing — the mandatory tripwire's terminal condition.
    #[test]
    fn contracts_with_generic_constraint_assemble_without_debug_assert() {
        let import_term = LinearTerm {
            variable: VariableRef::ContractImport {
                contract_id: EntityId(10),
                block_id: None,
            },
            coefficient: CoefficientRef::Literal(1.0),
            scale: 1.0,
        };
        let constraint = GenericConstraint {
            id: EntityId(1),
            name: "c-import-cap".to_string(),
            description: None,
            expression: ConstraintExpression {
                terms: vec![import_term],
            },
            slack: SlackConfig {
                enabled: false,
                penalty: None,
            },
            bound_lower_affine: None,
            bound_upper_affine: None,
        };
        let mut fixtures = PumpFixtures::new_with_contracts(
            vec![fixture_hydro(1), fixture_hydro(2)],
            vec![fixture_bus(1)],
            vec![contract(10, 1, ContractType::Import)],
        )
        .with_generic_constraint(constraint, 50.0);
        let ctx = fixtures.make_ctx();
        let stage = two_block_stage(0, [300.0, 444.0]);
        let layout = StageLayout::new(&ctx, &stage, 0);

        // Exercises fill_load_balance_entries (contract block) and
        // fill_generic_constraint_entries (the resolved ContractImport arm). Under
        // debug_assertions every debug_assert! is live; reaching the assertion below
        // proves none fired.
        let col_entries = build_stage_matrix_entries(&ctx, &stage, 0, &layout);
        assert_eq!(col_entries.len(), layout.num_cols);
    }

    /// A station whose `bus_id` is absent from `bus_pos` writes no load-balance
    /// entry and does not panic.
    #[test]
    fn pumping_power_missing_bus_skips_without_panic() {
        // Station on bus id 99, which is NOT among the fixture buses (only id 1).
        let mut fixtures = PumpFixtures::new_with_buses(
            vec![fixture_hydro(1), fixture_hydro(2)],
            vec![station_full(10, 1, 2, 0.0, 50.0, 99, 0.5)],
            vec![fixture_bus(1)],
        );
        let ctx = fixtures.make_ctx();
        let stage = two_block_stage(0, [300.0, 444.0]);
        let layout = StageLayout::new(&ctx, &stage, 0);

        let mut col_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); layout.num_cols];
        fill_load_balance_entries(&ctx, 0, &layout, &mut col_entries);

        let n_blks = layout.clock.n_blks();
        for blk in 0..n_blks {
            let col = layout.geometry.pumping_flow.start + blk;
            assert!(
                col_entries[col].is_empty(),
                "blk {blk}: station on an unmapped bus must write no load-balance entry"
            );
        }
    }

    /// With no pumping stations, `fill_load_balance_entries` produces exactly the
    /// same entries it would without the pumping loop — the pumping path is inert.
    /// A 1-bus, 2-hydro system with one declared station is the baseline; removing
    /// the station must leave every column's load-balance entries unchanged.
    #[test]
    fn no_pumping_stations_leaves_load_balance_entries_identical() {
        let build = |stations: Vec<PumpingStation>| {
            let mut fixtures = PumpFixtures::new_with_buses(
                vec![fixture_hydro(1), fixture_hydro(2)],
                stations,
                vec![fixture_bus(1)],
            );
            let ctx = fixtures.make_ctx();
            let stage = two_block_stage(0, [300.0, 444.0]);
            let layout = StageLayout::new(&ctx, &stage, 0);
            let mut col_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); layout.num_cols];
            fill_load_balance_entries(&ctx, 0, &layout, &mut col_entries);
            // Truncate to the non-pumping column region: with zero stations the
            // layout has no pumping columns, so compare the shared prefix that both
            // layouts share (generation/thermal/line/deficit/excess columns are all
            // indexed before the pumping block).
            (layout.geometry.pumping_flow.start, col_entries)
        };

        let (pump_start_empty, entries_empty) = build(vec![]);
        let (_pump_start_one, entries_one) = build(vec![station(10, 1, 2, 0.0, 50.0)]);

        // Every column before the pumping block must carry identical load-balance
        // entries whether or not a station is present.
        for col in 0..pump_start_empty {
            assert_eq!(
                entries_empty[col], entries_one[col],
                "load-balance entries for column {col} must be pumping-independent"
            );
        }
    }

    /// Build the full CSC for a 2-reservoir + 1-bus system twice with the hydro,
    /// station, and bus declarations supplied in two DIFFERENT input orders, and
    /// assert the assembled CSC arrays are byte-identical. Determinism is a hard
    /// rule: the canonical ID-sort plus the per-column row-sort must erase all
    /// trace of the input declaration order.
    ///
    /// Two stations with opposite source/destination orientation are declared so
    /// the assertion is load-bearing on the pumping path: a single station would
    /// pass even if the pumping iteration were declaration-order-dependent (there
    /// is nothing to scramble), whereas permuting two stations exercises the
    /// per-column row-sort that decouples declaration order from CSC layout. Both
    /// stations sit on the single bus, so the `−consumption_mw_per_m3s` bus-power
    /// entries are part of the assembled CSC and the assertion covers them too.
    #[test]
    fn csc_byte_identical_under_permuted_declaration_order() {
        let assemble = |hydros: Vec<Hydro>, stations: Vec<PumpingStation>| {
            let mut fixtures = PumpFixtures::new(hydros, stations);
            let ctx = fixtures.make_ctx();
            let stage = two_block_stage(0, [300.0, 444.0]);
            let layout = StageLayout::new(&ctx, &stage, 0);
            let mut entries = build_stage_matrix_entries(&ctx, &stage, 0, &layout);
            // Mirror the production per-column row-sort (see build_single_stage_template).
            for col in &mut entries {
                col.sort_unstable_by_key(|&(row, _)| row);
            }
            assemble_csc(&entries)
        };

        // Two reservoirs (ids 1, 2) and two stations moving water in opposite
        // directions (10: 1 → 2, 20: 2 → 1), both on bus 1 with DISTINCT
        // consumption rates so a permutation that mislabels which station's
        // `−consumption_mw_per_m3s` lands on which pumping column would be caught.
        // Order A declares both ascending.
        let csc_a = assemble(
            vec![fixture_hydro(1), fixture_hydro(2)],
            vec![
                station_full(10, 1, 2, 5.0, 80.0, 1, 0.4),
                station_full(20, 2, 1, 0.0, 30.0, 1, 0.9),
            ],
        );
        // Order B declares the identical entities with both hydros and both
        // stations reversed.
        let csc_b = assemble(
            vec![fixture_hydro(2), fixture_hydro(1)],
            vec![
                station_full(20, 2, 1, 0.0, 30.0, 1, 0.9),
                station_full(10, 1, 2, 5.0, 80.0, 1, 0.4),
            ],
        );

        assert_eq!(csc_a.0, csc_b.0, "col_starts must be byte-identical");
        assert_eq!(csc_a.1, csc_b.1, "row_indices must be byte-identical");
        assert_eq!(csc_a.2, csc_b.2, "values must be byte-identical");
    }

    /// Build the full stage CSC for a multi-entity system (2 hydros, 3 buses, 2
    /// thermals on different buses, 2 lines on different bus pairs, 1 generic
    /// constraint over a thermal/line/bus triple) twice with EVERY entity family's
    /// declaration order reversed between the two builds, and assert the assembled
    /// CSC arrays are byte-identical. This is the determinism backstop for the
    /// bus/line/thermal/generic-constraint families the single-bus
    /// `csc_byte_identical_under_permuted_declaration_order` cannot scramble: a
    /// map-iterating fill that reintroduced declaration-order dependence on any of
    /// these families would diverge here.
    ///
    /// The constraint references a `ThermalGeneration`, a `LineExchange`, and a
    /// `BusDeficit` term so the resolver path through
    /// `thermal_pos`/`line_pos`/`bus_pos` participates in the asserted CSC, not
    /// only the load-balance fill. Distinct per-entity attributes (thermals on
    /// different buses with distinct bounds, lines on different bus pairs with
    /// distinct capacities, buses with distinct deficit-segment counts) make the
    /// assertion load-bearing: a permutation that mislabelled which entity owns
    /// which slot would change the CSC.
    #[test]
    fn csc_byte_identical_under_permuted_multi_entity_order() {
        // Generic constraint id 7:
        //   thermal_gen(10) + line_exchange(100) + bus_deficit(2) <= 100
        // All referenced ids ARE present in the position maps, so the resolver
        // contributes real entries rather than silently returning an empty vec.
        let make_constraint = || GenericConstraint {
            id: EntityId(7),
            name: "gc_multi".to_string(),
            description: None,
            expression: ConstraintExpression {
                terms: vec![
                    LinearTerm {
                        coefficient: CoefficientRef::Literal(1.0),
                        scale: 1.0,
                        variable: VariableRef::ThermalGeneration {
                            thermal_id: EntityId(10),
                            block_id: None,
                        },
                    },
                    LinearTerm {
                        coefficient: CoefficientRef::Literal(1.0),
                        scale: 1.0,
                        variable: VariableRef::LineExchange {
                            line_id: EntityId(100),
                            block_id: None,
                        },
                    },
                    LinearTerm {
                        coefficient: CoefficientRef::Literal(1.0),
                        scale: 1.0,
                        variable: VariableRef::BusDeficit {
                            bus_id: EntityId(2),
                            block_id: None,
                        },
                    },
                ],
            },
            slack: SlackConfig {
                enabled: false,
                penalty: None,
            },
            bound_lower_affine: None,
            bound_upper_affine: None,
        };

        // Run the production assemble sequence into a CSC for a fixture. Takes the
        // fixture by reference so the caller can keep it alive and build a layout
        // from it for the offset reads — the per-call `StageLayout` borrows the
        // function-local ctx/state and cannot escape the closure.
        let assemble = |fixtures: &mut PumpFixtures| {
            let ctx = fixtures.make_ctx();
            let stage = two_block_stage(0, [300.0, 444.0]);
            let layout = StageLayout::new(&ctx, &stage, 0);

            let mut entries = build_stage_matrix_entries(&ctx, &stage, 0, &layout);
            let mut col_upper = vec![f64::INFINITY; layout.num_cols];
            let mut objective = vec![0.0_f64; layout.num_cols];
            let mut row_lower = vec![f64::NEG_INFINITY; layout.rows.num_rows];
            let mut row_upper = vec![f64::INFINITY; layout.rows.num_rows];
            let mut buffers = LpMatrixBuffers {
                col_entries: &mut entries,
                col_upper: &mut col_upper,
                objective: &mut objective,
                row_lower: &mut row_lower,
                row_upper: &mut row_upper,
            };
            fill_generic_constraint_entries(&ctx, 0, &layout, &mut buffers);

            // Mirror the production per-column row-sort (see
            // build_single_stage_template) before assembling the CSC.
            for col in &mut entries {
                col.sort_unstable_by_key(|&(row, _)| row);
            }
            assemble_csc(&entries)
        };

        // Order A: every family declared ascending. The fixture is kept alive so
        // the order-A layout (for the offset reads below) is built from it.
        let mut fixtures_a = PumpFixtures::new_full(
            vec![fixture_hydro(1), fixture_hydro(2)],
            Vec::new(),
            vec![
                fixture_bus_with(1, 1, 1.0),
                fixture_bus_with(2, 2, 3.0),
                fixture_bus_with(3, 1, 5.0),
            ],
            vec![
                fixture_thermal(10, 1, 0.0, 30.0, 12.0),
                fixture_thermal(20, 3, 5.0, 45.0, 27.0),
            ],
            vec![
                fixture_line(100, 1, 2, 50.0, 40.0),
                fixture_line(200, 2, 3, 70.0, 60.0),
            ],
        )
        .with_generic_constraint(make_constraint(), 100.0);
        let csc_a = assemble(&mut fixtures_a);

        // Order B: the identical entities, every family declared in reverse.
        let mut fixtures_b = PumpFixtures::new_full(
            vec![fixture_hydro(2), fixture_hydro(1)],
            Vec::new(),
            vec![
                fixture_bus_with(3, 1, 5.0),
                fixture_bus_with(2, 2, 3.0),
                fixture_bus_with(1, 1, 1.0),
            ],
            vec![
                fixture_thermal(20, 3, 5.0, 45.0, 27.0),
                fixture_thermal(10, 1, 0.0, 30.0, 12.0),
            ],
            vec![
                fixture_line(200, 2, 3, 70.0, 60.0),
                fixture_line(100, 1, 2, 50.0, 40.0),
            ],
        )
        .with_generic_constraint(make_constraint(), 100.0);
        let csc_b = assemble(&mut fixtures_b);

        // Order-A layout (held by the test, owning its ctx/state) for the offset
        // reads below. The layout offsets are declaration-order-invariant, so this
        // matches the layout order A's CSC was assembled with.
        let ctx_a = fixtures_a.make_ctx();
        let stage_a = two_block_stage(0, [300.0, 444.0]);
        let layout_a = StageLayout::new(&ctx_a, &stage_a, 0);

        assert_eq!(csc_a.0, csc_b.0, "col_starts must be byte-identical");
        assert_eq!(csc_a.1, csc_b.1, "row_indices must be byte-identical");
        assert_eq!(csc_a.2, csc_b.2, "values must be byte-identical");

        // Criterion 3: prove the generic-constraint resolver path ran (not just
        // the load-balance fill) by reading the generic row's coefficients on the
        // resolved thermal/line/deficit columns from order A's CSC. The expression
        // is block-dependent (ThermalGeneration/LineExchange/BusDeficit), so it
        // expands to one generic row per block; probe every block.
        let n_blks = layout_a.clock.n_blks();
        assert_eq!(
            layout_a.rows.n_generic_rows, n_blks,
            "block-dependent generic constraint must expand to one row per block"
        );
        let grid = layout_a.block_grid();
        let t_pos = 0; // thermal id 10 sorts to position 0.
        let l_pos = 0; // line id 100 sorts to position 0.
        let b_pos = 1; // bus id 2 sorts to position 1 (buses 1,2,3).
        for blk in 0..n_blks {
            let row = i32::try_from(layout_a.rows.row_generic_start + blk).unwrap();
            let coeff_at = |col: usize| -> f64 {
                let start = usize::try_from(csc_a.0[col]).unwrap();
                let end = usize::try_from(csc_a.0[col + 1]).unwrap();
                csc_a.1[start..end]
                    .iter()
                    .zip(&csc_a.2[start..end])
                    .filter(|&(&r, _)| r == row)
                    .map(|(_, &v)| v)
                    .sum()
            };

            // ThermalGeneration(10): +1.0 on thermal 10's column.
            let thermal_col = grid.flat(layout_a.geometry.thermal.start, t_pos, BlockIdx::new(blk));
            assert_eq!(
                coeff_at(thermal_col),
                1.0,
                "blk {blk}: generic row must carry +1.0 on thermal 10's column \
                 (resolver path through thermal_pos)"
            );
            // LineExchange(100): +1.0 on the forward column, -1.0 on the reverse.
            assert_eq!(
                coeff_at(
                    layout_a
                        .geometry
                        .line_fwd_col(LineSys::new(l_pos), BlockIdx::new(blk))
                ),
                1.0,
                "blk {blk}: generic row must carry +1.0 on line 100's forward column \
                 (resolver path through line_pos)"
            );
            assert_eq!(
                coeff_at(
                    layout_a
                        .geometry
                        .line_rev_col(LineSys::new(l_pos), BlockIdx::new(blk))
                ),
                -1.0,
                "blk {blk}: generic row must carry -1.0 on line 100's reverse column"
            );
            // BusDeficit(2): +1.0 on each of bus 2's two deficit-segment columns.
            for seg in 0..2 {
                assert_eq!(
                    coeff_at(layout_a.deficit_col(BusSys::new(b_pos), seg, BlockIdx::new(blk))),
                    1.0,
                    "blk {blk}: generic row must carry +1.0 on bus 2 deficit segment {seg} \
                     (resolver path through bus_pos)"
                );
            }
        }
    }

    /// Declaration-order invariance on a FILLING system: build a two-filling-hydro
    /// system twice with the hydros declared in opposite orders, at a Filling stage,
    /// and assert BOTH the assembled CSC and the row bounds (including the per-stage
    /// `filling_target` `V_target[t]` RHS, which rides on the precomputed
    /// `ctx.filling_v_target`) are byte-identical. The two hydros carry DISTINCT
    /// resolved `min_storage_hm3` so a permutation that mislabelled which hydro owns
    /// which `σ_fill` floor would change the RHS and diverge here. This is the
    /// filling counterpart of `csc_byte_identical_under_permuted_declaration_order`.
    #[test]
    fn filling_csc_and_rows_byte_identical_under_permuted_declaration_order() {
        // Distinct dead volumes so a hydro mislabel changes the V_target RHS.
        const H1_MIN_STORAGE: f64 = 41.0;
        const H2_MIN_STORAGE: f64 = 58.0;

        // Build the fixture with the two filling hydros (ids 1, 2) declared in the
        // given order, set each hydro's resolved dead volume, and return the
        // assembled CSC plus the (row_lower, row_upper) bounds at a Filling stage.
        #[expect(
            clippy::type_complexity,
            reason = "each case returns the raw build tuple its assertions destructure"
        )]
        let build = |hydros: Vec<Hydro>| -> ((Vec<i32>, Vec<i32>, Vec<f64>), Vec<f64>, Vec<f64>) {
            let mut fixtures = PumpFixtures::new(hydros, Vec::new());
            let h1_idx = fixtures.hydro_pos[&EntityId(1)];
            let h2_idx = fixtures.hydro_pos[&EntityId(2)];
            fixtures
                .base
                .bounds
                .hydro_bounds_mut(h1_idx, 0)
                .min_storage_hm3 = H1_MIN_STORAGE;
            fixtures
                .base
                .bounds
                .hydro_bounds_mut(h2_idx, 0)
                .min_storage_hm3 = H2_MIN_STORAGE;
            let ctx = fixtures.make_ctx();
            // RET_FILLING_ID = 3 is a Filling stage for the start=2/entry=4 window.
            let stage = two_block_stage(usize::try_from(RET_FILLING_ID).unwrap(), [300.0, 444.0]);
            let layout = StageLayout::new(&ctx, &stage, 0);
            let (row_lower, row_upper) =
                super::super::rows::fill_stage_rows(&ctx, &stage, 0, &layout);
            let csc = {
                let mut entries = build_stage_matrix_entries(&ctx, &stage, 0, &layout);
                for col in &mut entries {
                    col.sort_unstable_by_key(|&(row, _)| row);
                }
                assemble_csc(&entries)
            };
            (csc, row_lower, row_upper)
        };

        // Both hydros are filling (start = 2, entry = 4); ret_hydro pins that window.
        let (csc_a, rl_a, ru_a) = build(vec![
            ret_hydro(1, Some(2), Some(RET_ENTRY_STAGE_ID), true),
            ret_hydro(2, None, Some(RET_ENTRY_STAGE_ID), true),
        ]);
        let (csc_b, rl_b, ru_b) = build(vec![
            ret_hydro(2, None, Some(RET_ENTRY_STAGE_ID), true),
            ret_hydro(1, Some(2), Some(RET_ENTRY_STAGE_ID), true),
        ]);

        assert_eq!(csc_a.0, csc_b.0, "col_starts must be byte-identical");
        assert_eq!(csc_a.1, csc_b.1, "row_indices must be byte-identical");
        assert_eq!(csc_a.2, csc_b.2, "values must be byte-identical");
        assert_eq!(
            rl_a, rl_b,
            "row_lower (incl. V_target RHS) must be byte-identical"
        );
        assert_eq!(ru_a, ru_b, "row_upper must be byte-identical");
    }

    /// Pin the two structural water-row coefficients `fill_state_and_water_entries`
    /// writes for a two-reservoir cascade: the cascade-upstream `−tau_h` and the
    /// `z`-inflow coupling `−ζ`. Both are the weakest-backstopped coefficients in
    /// the water row — a sign flip on either silently mis-routes water and produces
    /// wrong bounds, yet outside this test they are exercised only by a slow parity
    /// D-case whose hash-mismatch failure mode does not localize to the water row.
    ///
    /// Cascade `H_up`(id 1) → `H_down`(id 2), both constant-productivity. On the
    /// downstream hydro's water-balance row, per block, the upstream turbine and
    /// spillage columns carry `−tau_h` (the upstream release arriving as inflow)
    /// while the downstream's own turbine carries `+tau_h` (its own outflow). The
    /// `+τ`/`−τ` sign *contrast* is asserted, not just one magnitude: a global
    /// flip negating both would otherwise pass. `tau_h` is computed from
    /// `M3S_TO_HM3` (never a literal) so the assertion cannot drift from the
    /// production constant.
    #[test]
    fn cascade_upstream_tau_and_z_inflow_land_on_downstream_water_row() {
        use cobre_core::scenario::InflowModel;

        // Assemble the production water-row fill into a CSC. The generic-constraint
        // fill is omitted (no generic constraints here); the water row is reached
        // through `build_stage_matrix_entries` alone.
        // H_up id 1 sorts to position 0, H_down id 2 to position 1.
        let up = 1;
        let down = 2;
        let mut cascade_fixtures = PumpFixtures::new_full(
            vec![
                fixture_hydro_ds(up, Some(down)),
                fixture_hydro_ds(down, None),
            ],
            Vec::new(),
            vec![fixture_bus(1)],
            Vec::new(),
            Vec::new(),
        );

        // Run the production water-row fill into a CSC. `ctx`/`state`/`layout` are
        // held by the test (borrowing `cascade_fixtures`, which outlives them), so
        // the layout offsets read below stay valid.
        let ctx = cascade_fixtures.make_ctx();
        let stage = two_block_stage(0, [300.0, 444.0]);
        let layout = StageLayout::new(&ctx, &stage, 0);
        let csc = {
            let mut entries = build_stage_matrix_entries(&ctx, &stage, 0, &layout);
            // Mirror the production per-column row-sort (see
            // build_single_stage_template) before assembling the CSC.
            for col in &mut entries {
                col.sort_unstable_by_key(|&(row, _)| row);
            }
            assemble_csc(&entries)
        };

        // Sum the CSC values landing on `(col, row)` — mirrors the permutation
        // test's probe; multiple per-block pushes to one cell would otherwise hide.
        let coeff_at = |col: usize, row: i32| -> f64 {
            let start = usize::try_from(csc.0[col]).unwrap();
            let end = usize::try_from(csc.0[col + 1]).unwrap();
            csc.1[start..end]
                .iter()
                .zip(&csc.2[start..end])
                .filter(|&(&r, _)| r == row)
                .map(|(_, &v)| v)
                .sum()
        };

        let up_idx = 0; // H_up id 1.
        let down_idx = 1; // H_down id 2.
        let down_row = i32::try_from(layout.geometry.water_balance.start() + down_idx).unwrap();
        for blk in 0..layout.clock.n_blks() {
            // tau_h is the identical expression the production fill uses; the two
            // blocks carry distinct durations (300 vs 444), so a per-block divisor
            // confusion is observable.
            let tau_h = two_block_stage(0, [300.0, 444.0]).blocks[blk].duration_hours * M3S_TO_HM3;

            assert_eq!(
                coeff_at(
                    layout
                        .geometry
                        .turbine_col(HydroCell::new(up_idx), BlockIdx::new(blk)),
                    down_row
                ),
                -tau_h,
                "blk {blk}: upstream turbine column must carry -tau_h on the \
                 downstream water row (cascade-upstream inflow)"
            );
            assert_eq!(
                coeff_at(
                    layout
                        .geometry
                        .spillage_col(HydroSys::new(up_idx), BlockIdx::new(blk)),
                    down_row
                ),
                -tau_h,
                "blk {blk}: upstream spillage column must carry -tau_h on the \
                 downstream water row (cascade-upstream inflow)"
            );
            assert_eq!(
                coeff_at(
                    layout
                        .geometry
                        .turbine_col(HydroCell::new(down_idx), BlockIdx::new(blk)),
                    down_row
                ),
                tau_h,
                "blk {blk}: downstream's OWN turbine column must carry +tau_h on \
                 its water row (self-outflow) — the +τ/−τ sign contrast"
            );
        }

        // z-inflow coupling: self-contained block. An AR(1) PrecomputedPar carrying
        // a nonzero psi for the downstream hydro proves the water row carries no
        // lag entry EVEN when a PAR model is present (the default fixture has
        // psi == 0, which would leave the absence vacuous). psi[0] for the
        // downstream hydro is constructed to equal phi exactly:
        // the classical conversion psi = phi * s_m / s_lag collapses to phi when
        // both the study stage and its pre-study lag stage carry the same std.
        let phi = 0.6_f64;
        let inflow_models = vec![
            // Upstream (id 1): white noise — psi stays 0 at every lag.
            InflowModel {
                hydro_id: EntityId(up),
                stage_id: 0,
                mean_m3s: 50.0,
                std_m3s: 1.0,
                ar_coefficients: vec![],
                residual_std_ratio: 1.0,
                annual: None,
            },
            // Downstream (id 2) study stage: AR(1) with coefficient phi; std == 1.0
            // so the lag-stage ratio is exactly 1.0.
            InflowModel {
                hydro_id: EntityId(down),
                stage_id: 0,
                mean_m3s: 80.0,
                std_m3s: 1.0,
                ar_coefficients: vec![phi],
                residual_std_ratio: 1.0,
                annual: None,
            },
            // Downstream pre-study lag stage (stage_id -1) with the same std, so
            // the Tier-1 exact-match lag lookup yields s_lag == s_m and the
            // conversion collapses to psi[0] == phi bit-exactly.
            InflowModel {
                hydro_id: EntityId(down),
                stage_id: -1,
                mean_m3s: 80.0,
                std_m3s: 1.0,
                ar_coefficients: vec![phi],
                residual_std_ratio: 1.0,
                annual: None,
            },
        ];
        let par_stage = two_block_stage(0, [300.0, 444.0]);
        let par_lp = PrecomputedPar::build(
            &inflow_models,
            std::slice::from_ref(&par_stage),
            &[EntityId(up), EntityId(down)],
            None,
        )
        .expect("AR(1) PrecomputedPar build must succeed");
        let psi_val = par_lp.psi_slice(0, down_idx)[0];
        assert_eq!(psi_val, phi, "downstream psi[0] must equal phi exactly");

        let mut ar_fixtures = PumpFixtures::new_full(
            vec![
                fixture_hydro_ds(up, Some(down)),
                fixture_hydro_ds(down, None),
            ],
            Vec::new(),
            vec![fixture_bus(1)],
            Vec::new(),
            Vec::new(),
        )
        .with_par_lp(par_lp);
        let ar_ctx = ar_fixtures.make_ctx();
        let ar_stage = two_block_stage(0, [300.0, 444.0]);
        let ar_layout = StageLayout::new(&ar_ctx, &ar_stage, 0);
        let ar_csc = {
            let mut entries = build_stage_matrix_entries(&ar_ctx, &ar_stage, 0, &ar_layout);
            for col in &mut entries {
                col.sort_unstable_by_key(|&(row, _)| row);
            }
            assemble_csc(&entries)
        };
        let ar_coeff_at = |col: usize, row: i32| -> f64 {
            let start = usize::try_from(ar_csc.0[col]).unwrap();
            let end = usize::try_from(ar_csc.0[col + 1]).unwrap();
            ar_csc.1[start..end]
                .iter()
                .zip(&ar_csc.2[start..end])
                .filter(|&(&r, _)| r == row)
                .map(|(_, &v)| v)
                .sum()
        };
        // Lag column for (lag 0, downstream hydro): inflow_lags.start + 0*n_h + h.
        let lag_col = ar_layout.state.inflow_lags.start + down_idx;
        let ar_row = i32::try_from(ar_layout.geometry.water_balance.start() + down_idx).unwrap();
        assert_eq!(
            ar_coeff_at(lag_col, ar_row),
            0.0,
            "downstream inflow-lag column must carry no entry on its water row"
        );
        let z_col = ar_layout.state.z_inflow.start + down_idx;
        assert_eq!(
            ar_coeff_at(z_col, ar_row),
            -ar_layout.clock.zeta(),
            "downstream z-inflow column must carry -zeta on its own water row"
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Cell-summing: a split plant's turbined flow sums across HydroCellIndex
    // cells instead of reading only the first.
    // ─────────────────────────────────────────────────────────────────────────

    /// `fixture_hydro_ds` with two explicit unit groups, on `group_bus_a` and
    /// `group_bus_b`, so the plant splits into two `HydroCellIndex` cells
    /// instead of the single implicit one every other fixture in this module
    /// gets. Bounds differ between the two groups so a summed-vs-first-cell
    /// bug cannot hide behind a symmetric fixture.
    fn fixture_hydro_two_cells(
        id: i32,
        downstream_id: Option<i32>,
        group_bus_a: i32,
        group_bus_b: i32,
    ) -> Hydro {
        let mut hydro = fixture_hydro_ds(id, downstream_id);
        hydro.unit_groups = vec![
            make_unit_group(
                EntityId(id * 10),
                EntityId(group_bus_a),
                0.0,
                20.0,
                0.0,
                25.0,
            ),
            make_unit_group(
                EntityId(id * 10 + 1),
                EntityId(group_bus_b),
                0.0,
                25.0,
                0.0,
                25.0,
            ),
        ];
        hydro
    }

    /// Two-hydro fixture for the cell-summing water-balance and
    /// operational-violation tests: plant 0 (id 10) stays single-bus (cell 0);
    /// plant 1 (id 11) declares two groups on buses 20 and 21, landing on cells
    /// 1 and 2 (id-ascending after plant 0's cell 0 — `HydroCellIndex` orders a
    /// plant's cells after every earlier plant's). No cascade: each plant's own
    /// row is what these tests read.
    fn split_plant_fixture() -> PumpFixtures {
        PumpFixtures::new(
            vec![fixture_hydro(10), fixture_hydro_two_cells(11, None, 20, 21)],
            Vec::new(),
        )
    }

    /// Three-hydro cascade fixture for the cell-summing cascade test: plant 0
    /// (id 10) is the SPLIT, UPSTREAM plant (two groups on buses 20/21,
    /// cells 0 and 1) releasing into plant 2 (id 12, single-bus, downstream,
    /// cell 3); plant 1 (id 11, single-bus, cell 2) is an unrelated filler
    /// entity so "plant 2" lands on a third, independently-indexed hydro. The
    /// downstream plant is deliberately NOT split, so summing the wrong
    /// (downstream) plant's cells is distinguishable from summing the
    /// upstream's.
    fn split_upstream_cascade_fixture() -> PumpFixtures {
        PumpFixtures::new(
            vec![
                fixture_hydro_two_cells(10, Some(12), 20, 21),
                fixture_hydro(11),
                fixture_hydro(12),
            ],
            Vec::new(),
        )
    }

    /// Requirement 1: a split plant's own water-balance row sums every cell's
    /// turbined-flow column at the SAME `tau_h`, not just the first, while the
    /// plant-level spillage/diversion columns stay singly-pushed. Plant 0
    /// (single cell) is the byte-identity control; plant 1 (cells 1, 2) is
    /// what the mutation below breaks.
    #[test]
    fn test_water_balance_sums_every_cell_of_a_split_plant() {
        let mut fixtures = split_plant_fixture();
        let ctx = fixtures.make_ctx();
        let stage = three_block_stage(0);
        let layout = StageLayout::new(&ctx, &stage, 0);
        let entries = build_stage_matrix_entries(&ctx, &stage, 0, &layout);

        let plant1_idx = 1;
        let row0_u = layout.geometry.water_balance.start();
        let row1_u = layout.geometry.water_balance.start() + plant1_idx;
        let row0 = i32::try_from(row0_u).unwrap();
        let row1 = i32::try_from(row1_u).unwrap();

        let mut sorted = entries.clone();
        for col in &mut sorted {
            col.sort_unstable_by_key(|&(row, _)| row);
        }
        let csc = assemble_csc(&sorted);

        for blk in 0..layout.clock.n_blks() {
            let tau_h = stage.blocks[blk].duration_hours * M3S_TO_HM3;
            let blk_idx = BlockIdx::new(blk);
            let col_cell0 = layout.geometry.turbine_col(HydroCell::new(0), blk_idx);
            let col_cell1 = layout.geometry.turbine_col(HydroCell::new(1), blk_idx);
            let col_cell2 = layout.geometry.turbine_col(HydroCell::new(2), blk_idx);
            let col_spillage1 = layout
                .geometry
                .spillage_col(HydroSys::new(plant1_idx), blk_idx);
            let col_diversion1 = layout
                .geometry
                .diversion_col(HydroSys::new(plant1_idx), blk_idx);

            assert_eq!(
                coeff_at(&csc, col_cell1, row1),
                tau_h,
                "blk {blk}: plant 1's first cell must carry tau_h on its own water row"
            );
            assert_eq!(
                coeff_at(&csc, col_cell2, row1),
                tau_h,
                "blk {blk}: plant 1's second cell must carry tau_h on its own water row"
            );
            assert_eq!(
                entry_count_at(&entries, col_cell1, row1_u),
                1,
                "blk {blk}: plant 1's first cell must get exactly one turbined push on \
                 the water-balance row"
            );
            assert_eq!(
                entry_count_at(&entries, col_cell2, row1_u),
                1,
                "blk {blk}: plant 1's second cell must get exactly one turbined push on \
                 the water-balance row"
            );

            assert_eq!(
                coeff_at(&csc, col_cell0, row0),
                tau_h,
                "blk {blk}: single-cell plant 0's own turbine column must carry tau_h \
                 (plant 0's row carries exactly one turbined entry)"
            );
            assert_eq!(
                entry_count_at(&entries, col_cell0, row0_u),
                1,
                "blk {blk}: plant 0's turbine column must get exactly one push on the \
                 water-balance row"
            );

            assert_eq!(
                entry_count_at(&entries, col_spillage1, row1_u),
                1,
                "blk {blk}: plant 1's spillage column must not be pushed once per cell \
                 on the water-balance row"
            );
            assert_eq!(
                coeff_at(&csc, col_spillage1, row1),
                tau_h,
                "blk {blk}: plant 1's spillage coefficient must stay tau_h, not double"
            );
            assert_eq!(
                entry_count_at(&entries, col_diversion1, row1_u),
                1,
                "blk {blk}: plant 1's diversion column must not be pushed once per cell \
                 on the water-balance row"
            );
            assert_eq!(
                coeff_at(&csc, col_diversion1, row1),
                tau_h,
                "blk {blk}: plant 1's diversion coefficient must stay tau_h, not double"
            );
        }
    }

    /// Requirement 3: `fill_arc_release_block_entries` sums the UPSTREAM
    /// plant's cells with the SAME per-arc coefficient on each — never divides
    /// it across cells. Plant 0 (upstream) splits into cells 0 and 1; plant 2
    /// (downstream) stays single-cell.
    #[test]
    fn test_cascade_release_sums_the_upstream_plants_cells() {
        let mut fixtures = split_upstream_cascade_fixture();
        let ctx = fixtures.make_ctx();
        let stage = two_block_stage(0, [300.0, 444.0]);
        let layout = StageLayout::new(&ctx, &stage, 0);
        let csc = build_sorted_csc(&ctx, &stage, 0, &layout);

        // id order 10, 11, 12 -> positions 0, 1, 2. Plant 0's cells: 0, 1
        // (split); plant 1's cell: 2 (filler); plant 2's cell: 3 (downstream).
        let plant2_idx = 2;
        let down_row = i32::try_from(layout.geometry.water_balance.start() + plant2_idx).unwrap();

        for blk in 0..layout.clock.n_blks() {
            let tau_h = stage.blocks[blk].duration_hours * M3S_TO_HM3;
            let blk_idx = BlockIdx::new(blk);
            let coeff_cell0 = coeff_at(
                &csc,
                layout.geometry.turbine_col(HydroCell::new(0), blk_idx),
                down_row,
            );
            let coeff_cell1 = coeff_at(
                &csc,
                layout.geometry.turbine_col(HydroCell::new(1), blk_idx),
                down_row,
            );

            assert_eq!(
                coeff_cell0, -tau_h,
                "blk {blk}: plant 0's first cell must carry -tau_h on plant 2's water row"
            );
            assert_eq!(
                coeff_cell1, -tau_h,
                "blk {blk}: plant 0's second cell must carry -tau_h on plant 2's water \
                 row, the SAME magnitude as the first cell (never divided by cell count)"
            );
            assert_eq!(
                coeff_cell0 + coeff_cell1,
                2.0 * -tau_h,
                "blk {blk}: the total credited to plant 2's row from plant 0's two cells \
                 must be n_cells * -tau_h; a divide-by-cell-count implementation collapses \
                 this sum back to a single -tau_h"
            );
        }
    }

    /// The two power-side operational-violation families (min-turbine,
    /// min-generation) are now CELL-keyed — one row per `(cell, block)`, each
    /// row coupling ONLY that cell's own columns — while the two flow
    /// families (min/max-outflow) stay plant-keyed. Plant 1's second cell is
    /// `cell_idx = 2`; block 1 numerically coincides with plant 1's own
    /// index, so block 0 is also checked to give a genuinely three-way-distinct
    /// `(h_idx, cell_idx, blk_idx)` assertion point.
    #[test]
    fn test_operational_violation_power_rows_are_per_cell_not_plant() {
        let mut fixtures = split_plant_fixture();
        let ctx = fixtures.make_ctx();
        let stage = three_block_stage(0);
        let layout = StageLayout::new(&ctx, &stage, 0);
        let csc = build_sorted_csc(&ctx, &stage, 0, &layout);

        let n_h = layout.state.hydro_count;
        let n_blks = layout.clock.n_blks();
        let n_cells = ctx.hydro_cell_index.n_cells();
        assert_eq!(
            layout.oper_violation.min_outflow.len(),
            n_h * n_blks,
            "min_outflow family stays sized n_hydros * n_blks"
        );
        assert_eq!(
            layout.oper_violation.max_outflow.len(),
            n_h * n_blks,
            "max_outflow family stays sized n_hydros * n_blks"
        );
        assert_eq!(
            layout.oper_violation.min_turbine.len(),
            n_cells * n_blks,
            "min_turbine family is now sized n_cells * n_blks"
        );
        assert_eq!(
            layout.oper_violation.min_generation.len(),
            n_cells * n_blks,
            "min_generation family is now sized n_cells * n_blks"
        );

        let grid = layout.block_grid();
        let rho = 1.0; // PumpFixtures' default ConstantProductivity productivity.
        let cell1 = HydroCell::new(1);
        let cell2 = HydroCell::new(2);

        for &blk in &[0_usize, 1] {
            let blk_idx = BlockIdx::new(blk);
            let row_min_turbine_1 =
                i32::try_from(grid.flat(layout.oper_violation.min_turbine.start, 1, blk_idx))
                    .unwrap();
            let row_min_turbine_2 =
                i32::try_from(grid.flat(layout.oper_violation.min_turbine.start, 2, blk_idx))
                    .unwrap();
            assert_ne!(
                row_min_turbine_1, row_min_turbine_2,
                "blk {blk}: plant 1's two cells must own DISTINCT min_turbine rows"
            );

            assert_eq!(
                coeff_at(
                    &csc,
                    layout.geometry.turbine_col(cell1, blk_idx),
                    row_min_turbine_1
                ),
                1.0,
                "blk {blk}: cell 1's own turbine column must carry +1.0 on ITS OWN row"
            );
            assert_eq!(
                coeff_at(
                    &csc,
                    layout.geometry.turbine_col(cell2, blk_idx),
                    row_min_turbine_1
                ),
                0.0,
                "blk {blk}: cell 2's turbine column must NOT appear on cell 1's row"
            );
            assert_eq!(
                coeff_at(
                    &csc,
                    layout.geometry.turbine_col(cell2, blk_idx),
                    row_min_turbine_2
                ),
                1.0,
                "blk {blk}: cell 2's own turbine column must carry +1.0 on ITS OWN row"
            );
            assert_eq!(
                coeff_at(
                    &csc,
                    layout.geometry.turbine_below_col(cell1, blk_idx),
                    row_min_turbine_1
                ),
                1.0,
                "blk {blk}: cell 1's own turbine_below_slack column must carry +1.0 on ITS OWN row"
            );
            assert_eq!(
                coeff_at(
                    &csc,
                    layout.geometry.turbine_below_col(cell2, blk_idx),
                    row_min_turbine_2
                ),
                1.0,
                "blk {blk}: cell 2's own turbine_below_slack column must carry +1.0 on ITS OWN row"
            );
            assert_ne!(
                layout.geometry.turbine_below_col(cell1, blk_idx),
                layout.geometry.turbine_below_col(cell2, blk_idx),
                "blk {blk}: the two cells must own DISTINCT turbine_below_slack columns"
            );

            let row_min_gen_1 =
                i32::try_from(grid.flat(layout.oper_violation.min_generation.start, 1, blk_idx))
                    .unwrap();
            let row_min_gen_2 =
                i32::try_from(grid.flat(layout.oper_violation.min_generation.start, 2, blk_idx))
                    .unwrap();
            assert_ne!(
                row_min_gen_1, row_min_gen_2,
                "blk {blk}: plant 1's two cells must own DISTINCT min_generation rows"
            );
            assert_eq!(
                coeff_at(
                    &csc,
                    layout.geometry.turbine_col(cell1, blk_idx),
                    row_min_gen_1
                ),
                rho,
                "blk {blk}: cell 1's own turbine column must carry rho on ITS OWN min_generation row"
            );
            assert_eq!(
                coeff_at(
                    &csc,
                    layout.geometry.turbine_col(cell2, blk_idx),
                    row_min_gen_1
                ),
                0.0,
                "blk {blk}: cell 2's turbine column must NOT appear on cell 1's min_generation row"
            );
            assert_eq!(
                coeff_at(
                    &csc,
                    layout.geometry.turbine_col(cell2, blk_idx),
                    row_min_gen_2
                ),
                rho,
                "blk {blk}: cell 2's own turbine column must carry rho on ITS OWN min_generation row"
            );
        }
    }

    /// A single plant with a `ConstantProductivity` model and two cells: cell A
    /// (bus 30) owns one group; cell B (bus 31) owns two groups. Every group's
    /// `min_turbined_m3s * rho` is deliberately BELOW its own `min_generation_mw`
    /// (discriminates a rho-fold mutation on the generation floor), cell B's two
    /// groups have distinct minima (discriminates a max-fold mutation), and the
    /// plant's OWN declared `min_turbined_m3s`/`min_generation_mw` are set below
    /// EVERY cell's own group-sum (discriminates a `.min(plant)` clamp). Pins the
    /// min-floor contract: RHS = plain sum of the cell's own groups, never a
    /// fold, never a plant clamp, never apportioned.
    fn min_floor_fixture() -> Hydro {
        let mut hydro = fixture_hydro_ds(20, None);
        hydro.min_turbined_m3s = 1.0;
        hydro.min_generation_mw = 1.0;
        hydro.max_turbined_m3s = 1000.0;
        hydro.max_generation_mw = 1000.0;
        hydro.unit_groups = vec![
            make_unit_group(EntityId(200), EntityId(30), 8.0, 500.0, 5.0, 500.0),
            make_unit_group(EntityId(201), EntityId(31), 4.0, 500.0, 3.0, 500.0),
            make_unit_group(EntityId(202), EntityId(31), 5.0, 500.0, 4.0, 500.0),
        ];
        hydro
    }

    /// The min-floor contract, end to end: RHS is the plain sum of a cell's OWN
    /// member groups' resolved minima, the matrix couples only the cell's own
    /// columns to the cell's own slack, and the penalty is the plant's price at
    /// FULL magnitude on every cell. Mutation-verified (by hand-patching
    /// `cell_min_turbined`/`cell_min_generation`/`fill_cell_block_family` and
    /// re-running) to fail under: (a) a `.min(plant)` clamp — would read `1.0`
    /// on every row instead of `5.0`/`7.0`/`8.0`/`9.0`; (b) a `max`-fold over a
    /// cell's own groups — cell B's `min_turbined` row would read `4.0` instead
    /// of `7.0`; (c) summing over the WHOLE PLANT instead of the cell (the
    /// pre-reversal plant-keyed behavior) — every row would read `12.0`
    /// (turbine) or `17.0` (generation) instead of the per-cell values; (d)
    /// `1/|cells|` price apportionment — every slack objective would read half
    /// its expected value.
    #[test]
    fn test_min_floor_rhs_is_the_cells_own_group_sum() {
        let hydro = min_floor_fixture();
        let mut fixtures = PumpFixtures::new(vec![hydro], Vec::new());
        fixtures.base.penalties = ResolvedPenalties::new(
            &PenaltiesCountsSpec {
                n_hydros: 1,
                n_buses: fixtures.base.buses.len(),
                n_lines: 0,
                n_ncs: 0,
                n_stages: N_STAGES,
            },
            &PenaltiesDefaults {
                hydro: HydroPenalties {
                    spillage_cost: 0.0,
                    diversion_cost: 0.0,
                    turbined_cost: 0.0,
                    storage_violation_below_cost: 0.0,
                    filling_target_violation_cost: 0.0,
                    turbined_violation_below_cost: 7.0,
                    outflow_violation_below_cost: 0.0,
                    outflow_violation_above_cost: 0.0,
                    generation_violation_below_cost: 11.0,
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

        let ctx = fixtures.make_ctx();
        assert_eq!(
            ctx.hydro_cell_index.n_cells(),
            2,
            "bus 30 (1 group) and bus 31 (2 groups) must partition into 2 cells"
        );
        let cell_a = HydroCell::new(0);
        let cell_b = HydroCell::new(1);

        let stage = three_block_stage(0);
        let layout = StageLayout::new(&ctx, &stage, 0);
        let n_cells = ctx.hydro_cell_index.n_cells();
        let n_blks = layout.clock.n_blks();

        let (row_lower, _row_upper) = fill_stage_rows(&ctx, &stage, 0, &layout);
        let (_col_lower, col_upper, objective) = fill_stage_columns(&ctx, &stage, 0, &layout);
        let csc = build_sorted_csc(&ctx, &stage, 0, &layout);
        let grid = layout.block_grid();
        let blk0 = BlockIdx::new(0);

        assert_eq!(
            layout.oper_violation.min_turbine.len(),
            n_cells * n_blks,
            "min_turbine must be sized n_cells * n_blks"
        );
        assert_eq!(
            layout.oper_violation.min_generation.len(),
            n_cells * n_blks,
            "min_generation must be sized n_cells * n_blks"
        );

        let row_turb_a = grid.flat(layout.oper_violation.min_turbine.start, 0, blk0);
        let row_turb_b = grid.flat(layout.oper_violation.min_turbine.start, 1, blk0);
        let row_gen_a = grid.flat(layout.oper_violation.min_generation.start, 0, blk0);
        let row_gen_b = grid.flat(layout.oper_violation.min_generation.start, 1, blk0);

        // RHS: plain sum of the cell's OWN groups, never the plant's declared
        // 1.0, never a fold, never the whole-plant sum (12.0 / 17.0).
        assert_eq!(row_lower[row_turb_a], 5.0, "cell A min-turbine RHS");
        assert_eq!(
            row_lower[row_turb_b], 7.0,
            "cell B min-turbine RHS (3.0 + 4.0)"
        );
        assert_eq!(row_lower[row_gen_a], 8.0, "cell A min-generation RHS");
        assert_eq!(
            row_lower[row_gen_b], 9.0,
            "cell B min-generation RHS (4.0 + 5.0)"
        );

        // Coupling: each cell's own turbine column at +1.0 on its OWN row only.
        let row_turb_a_i32 = i32::try_from(row_turb_a).unwrap();
        let row_turb_b_i32 = i32::try_from(row_turb_b).unwrap();
        assert_eq!(
            coeff_at(
                &csc,
                layout.geometry.turbine_col(cell_a, blk0),
                row_turb_a_i32
            ),
            1.0,
            "cell A's turbine column must carry +1.0 on cell A's row"
        );
        assert_eq!(
            coeff_at(
                &csc,
                layout.geometry.turbine_col(cell_b, blk0),
                row_turb_a_i32
            ),
            0.0,
            "cell B's turbine column must NOT appear on cell A's row"
        );
        assert_eq!(
            coeff_at(
                &csc,
                layout.geometry.turbine_col(cell_b, blk0),
                row_turb_b_i32
            ),
            1.0,
            "cell B's turbine column must carry +1.0 on cell B's row"
        );
        assert_eq!(
            coeff_at(
                &csc,
                layout.geometry.turbine_below_col(cell_a, blk0),
                row_turb_a_i32
            ),
            1.0,
            "cell A's turbine_below_slack must carry +1.0 on cell A's row"
        );
        assert_eq!(
            coeff_at(
                &csc,
                layout.geometry.turbine_below_col(cell_b, blk0),
                row_turb_b_i32
            ),
            1.0,
            "cell B's turbine_below_slack must carry +1.0 on cell B's row"
        );

        // Coupling: ConstantProductivity's min-generation row reads the cell's
        // own turbine column at rho (here rho == 1.0, PumpFixtures' default).
        let rho = 1.0;
        let row_gen_a_i32 = i32::try_from(row_gen_a).unwrap();
        let row_gen_b_i32 = i32::try_from(row_gen_b).unwrap();
        assert_eq!(
            coeff_at(
                &csc,
                layout.geometry.turbine_col(cell_a, blk0),
                row_gen_a_i32
            ),
            rho,
            "cell A's turbine column must carry rho on cell A's min-generation row"
        );
        assert_eq!(
            coeff_at(
                &csc,
                layout.geometry.turbine_col(cell_b, blk0),
                row_gen_a_i32
            ),
            0.0,
            "cell B's turbine column must NOT appear on cell A's min-generation row"
        );
        assert_eq!(
            coeff_at(
                &csc,
                layout.geometry.turbine_col(cell_b, blk0),
                row_gen_b_i32
            ),
            rho,
            "cell B's turbine column must carry rho on cell B's min-generation row"
        );
        assert_eq!(
            coeff_at(
                &csc,
                layout.geometry.generation_below_col(cell_a, blk0),
                row_gen_a_i32
            ),
            1.0,
            "cell A's generation_below_slack must carry +1.0 on cell A's row"
        );
        assert_eq!(
            coeff_at(
                &csc,
                layout.geometry.generation_below_col(cell_b, blk0),
                row_gen_b_i32
            ),
            1.0,
            "cell B's generation_below_slack must carry +1.0 on cell B's row"
        );

        // Penalty: the PLANT's price at FULL magnitude on EVERY cell — never
        // divided by the plant's cell count (2).
        let hours0 = BLOCK_HOURS[0];
        let turb_below_a = layout.geometry.turbine_below_col(cell_a, blk0);
        let turb_below_b = layout.geometry.turbine_below_col(cell_b, blk0);
        let gen_below_a = layout.geometry.generation_below_col(cell_a, blk0);
        let gen_below_b = layout.geometry.generation_below_col(cell_b, blk0);
        assert_eq!(
            objective[turb_below_a],
            7.0 * hours0,
            "cell A turbine_below price"
        );
        assert_eq!(
            objective[turb_below_b],
            7.0 * hours0,
            "cell B turbine_below price (same, full magnitude)"
        );
        assert_eq!(
            objective[gen_below_a],
            11.0 * hours0,
            "cell A generation_below price"
        );
        assert_eq!(
            objective[gen_below_b],
            11.0 * hours0,
            "cell B generation_below price (same, full magnitude)"
        );

        // Activation: every cell's floor is > 0.0, so every slack is unbounded above.
        assert_eq!(col_upper[turb_below_a], f64::INFINITY);
        assert_eq!(col_upper[turb_below_b], f64::INFINITY);
        assert_eq!(col_upper[gen_below_a], f64::INFINITY);
        assert_eq!(col_upper[gen_below_b], f64::INFINITY);
    }

    /// Plant-total invariance: the SAME total upstream release, pinned through
    /// one cell or split evenly across two, must reach an IDENTICAL objective
    /// and an IDENTICAL dual on the downstream water-balance row. The per-cell
    /// magnitude and count assertions above inspect the MATRIX; this inspects
    /// the SOLVED LP's economic output, which is what a leftover or duplicated
    /// cell-column push (over-delivering at the arc's coefficient) would
    /// actually break, and which a divide-by-cell-count mutation does not
    /// exercise.
    #[test]
    fn test_plant_total_release_is_invariant_to_cell_partition() {
        use cobre_solver::{ActiveSolver, SolverInterface};

        fn solve_pinned_release(
            hydros: Vec<Hydro>,
            cells: &[usize],
            total_flow: f64,
        ) -> (f64, f64) {
            let mut fixtures = PumpFixtures::new(hydros, Vec::new()).with_resolved_penalties();
            let ctx = fixtures.make_ctx();
            let stage = two_block_stage(0, [300.0, 444.0]);
            let layout = StageLayout::new(&ctx, &stage, 0);

            // Parallel mode's water-balance row sums EVERY block, so block 1's
            // upstream/downstream turbine columns must be pinned too -- left free,
            // block 1 gives the solver an untested, zero-cost way to balance the row
            // that has nothing to do with how block 0's release is partitioned.
            let downstream_idx = 1;
            #[expect(
                clippy::cast_precision_loss,
                reason = "test counts are small, far inside f64's exact-integer range"
            )]
            let per_cell = total_flow / cells.len() as f64;
            let mut pin_cols = Vec::new();
            let mut pin_bounds = Vec::new();
            for blk in [BlockIdx::new(0), BlockIdx::new(1)] {
                let release = if blk.get() == 0 { per_cell } else { 0.0 };
                for &c in cells {
                    pin_cols.push(layout.geometry.turbine_col(HydroCell::new(c), blk));
                    pin_bounds.push(release);
                }
                // The downstream plant's own turbining and any net storage change are
                // free, zero-cost escape valves that would otherwise absorb the pinned
                // release -- pin them shut so spillage is the row's only relief.
                pin_cols.push(
                    layout.geometry.turbine_col(
                        ctx.hydro_cell_index
                            .first_cell_of(HydroSys::new(downstream_idx)),
                        blk,
                    ),
                );
                pin_bounds.push(0.0);
            }
            pin_cols.push(layout.state.storage_in.start + downstream_idx);
            pin_bounds.push(50.0);
            pin_cols.push(downstream_idx);
            pin_bounds.push(50.0);

            let down_row = layout.geometry.water_balance.start() + downstream_idx;

            let out = super::super::template::build_single_stage_template(&ctx, &stage, 0);
            let template = out.template;

            let mut solver = ActiveSolver::new().expect("ActiveSolver::new()");
            solver.load_model(&template);
            solver.set_col_bounds(&pin_cols, &pin_bounds, &pin_bounds);
            let view = solver
                .solve(None)
                .expect("pinned cell-partition invariance LP must be feasible");
            (view.objective, view.dual[down_row])
        }

        let total_flow = 20.0;
        let (obj_one, dual_one) = solve_pinned_release(
            vec![fixture_hydro_ds(10, Some(11)), fixture_hydro_ds(11, None)],
            &[0],
            total_flow,
        );
        let (obj_two, dual_two) = solve_pinned_release(
            vec![
                fixture_hydro_two_cells(10, Some(11), 20, 21),
                fixture_hydro_ds(11, None),
            ],
            &[0, 1],
            total_flow,
        );

        assert_eq!(
            obj_one, obj_two,
            "the same total pinned release, through one cell or evenly split across two, \
             must reach the same objective"
        );
        assert_eq!(
            dual_one, dual_two,
            "the same total pinned release must produce the same downstream water-balance \
             dual regardless of how the upstream plant partitions it"
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Water travel-time: parallel-mode arrival split + bucket definition rows
    // ─────────────────────────────────────────────────────────────────────────

    /// Sort each column's entries by row (mirroring `build_single_stage_template`)
    /// and assemble the CSC.
    fn build_sorted_csc(
        ctx: &TemplateBuildCtx<'_>,
        stage: &Stage,
        stage_idx: usize,
        layout: &StageLayout,
    ) -> (Vec<i32>, Vec<i32>, Vec<f64>) {
        let mut entries = build_stage_matrix_entries(ctx, stage, stage_idx, layout);
        for col in &mut entries {
            col.sort_unstable_by_key(|&(row, _)| row);
        }
        assemble_csc(&entries)
    }

    /// Sum the CSC values landing on `(col, row)`; multiple per-block pushes to
    /// one cell would otherwise hide behind a single positional read.
    fn coeff_at(csc: &(Vec<i32>, Vec<i32>, Vec<f64>), col: usize, row: i32) -> f64 {
        let start = usize::try_from(csc.0[col]).unwrap();
        let end = usize::try_from(csc.0[col + 1]).unwrap();
        csc.1[start..end]
            .iter()
            .zip(&csc.2[start..end])
            .filter(|&(&r, _)| r == row)
            .map(|(_, &v)| v)
            .sum()
    }

    /// Count the RAW (pre-sort, pre-`assemble_csc`) pushes a column carries at
    /// `row`. A column is shared by several unrelated row families (water
    /// balance, load balance, the operational-violation slack rows), so a
    /// whole-column `entries[col].len()` over-counts; this scopes the count to
    /// one row the way [`coeff_at`] scopes the value sum.
    fn entry_count_at(entries: &[Vec<(usize, f64)>], col: usize, row: usize) -> usize {
        entries[col].iter().filter(|&&(r, _)| r == row).count()
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Per-cell bus crediting and per-cell FPHA production rows
    // ─────────────────────────────────────────────────────────────────────────

    /// Sum the RAW (pre-sort, pre-`assemble_csc`) values a column pushes at
    /// `row` — [`coeff_at`]'s summing semantics without a CSC assembly first.
    fn raw_coeff_at(entries: &[Vec<(usize, f64)>], col: usize, row: usize) -> f64 {
        entries[col]
            .iter()
            .filter(|&&(r, _)| r == row)
            .map(|&(_, v)| v)
            .sum()
    }

    /// Split-plant fixture for the per-cell bus-crediting and production-row
    /// apportionment tests. Buses, ascending id: `bus_pad` (1, unreferenced by
    /// any entity — pushes every referenced bus's `bus_pos` position up by one),
    /// `bus_cell_a` (2), `bus_decoy` (3, the decoy plant's bus), `bus_cell_b` (4).
    ///
    /// Plant 0 (id 10, decoy) is single-bus/single-cell — cell 0. Plant 1 (id
    /// 11, split) declares two groups: `max_turbined_m3s` 9000.0 on
    /// `bus_cell_a` and 3000.0 on `bus_cell_b` (share 0.75/0.25) — landing on
    /// cells 1 and 2 (ascending bus id, after the decoy's cell 0). At
    /// `bus_pos`, `bus_cell_a` is position 1 and `bus_cell_b` is position 3, so
    /// `(hydro_idx=1, cell_idx=2, bus_idx=3)` are mutually distinct from each
    /// other and from every `block_idx` this fixture's 3-block stages use.
    struct SplitBusFixture {
        fixtures: PumpFixtures,
        bus_cell_a: EntityId,
        bus_cell_b: EntityId,
        bus_decoy: EntityId,
    }

    fn split_bus_fixture(production_models: ProductionModelSet) -> SplitBusFixture {
        let bus_pad = EntityId(1);
        let bus_cell_a = EntityId(2);
        let bus_decoy = EntityId(3);
        let bus_cell_b = EntityId(4);

        let mut decoy = fixture_hydro_ds(10, None);
        decoy.unit_groups = vec![make_unit_group(
            EntityId(30),
            bus_decoy,
            0.0,
            45.0,
            0.0,
            50.0,
        )];
        decoy.declare_mirror_unit_group(bus_decoy);

        let mut split = fixture_hydro_ds(11, None);
        split.unit_groups = vec![
            make_unit_group(EntityId(20), bus_cell_a, 0.0, 45.0, 0.0, 9000.0),
            make_unit_group(EntityId(21), bus_cell_b, 0.0, 45.0, 0.0, 3000.0),
        ];
        split.declare_mirror_unit_group(bus_decoy);

        let buses = vec![
            fixture_bus(bus_pad.0),
            fixture_bus(bus_cell_a.0),
            fixture_bus(bus_decoy.0),
            fixture_bus(bus_cell_b.0),
        ];
        let fixtures = PumpFixtures::new_with_buses(vec![decoy, split], Vec::new(), buses)
            .with_production_models(production_models);

        SplitBusFixture {
            fixtures,
            bus_cell_a,
            bus_cell_b,
            bus_decoy,
        }
    }

    /// `ConstantProductivity` at the decoy (hydro-local 0) and `Fpha` at the
    /// split plant (hydro-local 1), for `N_STAGES` stages — the production
    /// model set every [`split_bus_fixture`]-based FPHA test shares.
    fn split_bus_production_models(fpha_planes: Vec<FphaPlane>) -> ProductionModelSet {
        ProductionModelSet::new(
            vec![
                vec![ResolvedProductionModel::ConstantProductivity { productivity: 1.0 }; N_STAGES],
                vec![
                    ResolvedProductionModel::Fpha {
                        planes: fpha_planes
                    };
                    N_STAGES
                ],
            ],
            &crate::test_support::minimal_hydros(2),
            N_STAGES,
        )
    }

    /// Requirement 1: the FPHA branch credits each cell to its OWN bus, and
    /// neither cell's generation column appears on the other cell's — or the
    /// decoy plant's — row.
    #[test]
    fn test_fpha_cell_generation_is_credited_to_its_own_bus() {
        let plane = FphaPlane {
            intercept: 1000.0,
            gamma_v: 4.0,
            gamma_q: 0.6,
            gamma_s: 0.3,
        };
        let mut fixture = split_bus_fixture(split_bus_production_models(vec![plane]));
        let ctx = fixture.fixtures.make_ctx();
        let stage = three_block_stage(0);
        let layout = StageLayout::new(&ctx, &stage, 0);

        let mut col_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); layout.num_cols];
        fill_load_balance_entries(&ctx, 0, &layout, &mut col_entries);

        let split = HydroSys::new(1);
        let cells: Vec<HydroCell> = ctx
            .hydro_cell_index
            .cells_of(split)
            .map(HydroCell::new)
            .collect();
        assert_eq!(
            cells.len(),
            2,
            "the split plant must partition into two cells"
        );
        let (cell_a, cell_b) = (cells[0], cells[1]);
        assert_eq!(
            cell_b,
            HydroCell::new(2),
            "the split plant's second cell must be cell index 2"
        );
        assert_eq!(ctx.hydro_cell_index.bus_of(cell_a), fixture.bus_cell_a);
        assert_eq!(ctx.hydro_cell_index.bus_of(cell_b), fixture.bus_cell_b);

        let bus_pos_a = ctx.positions.bus(fixture.bus_cell_a).unwrap();
        let bus_pos_b = ctx.positions.bus(fixture.bus_cell_b).unwrap();
        let bus_pos_decoy = ctx.positions.bus(fixture.bus_decoy).unwrap();
        // The mutually-distinct index check the fixture's own doc promises:
        // hydro_idx=1, cell_idx=2, block_idx=0, bus_idx=3.
        assert_ne!(split.get(), cell_b.get());
        assert_ne!(split.get(), 0_usize);
        assert_ne!(split.get(), bus_pos_b);
        assert_ne!(cell_b.get(), 0_usize);
        assert_ne!(cell_b.get(), bus_pos_b);
        assert_ne!(0_usize, bus_pos_b);

        let fpha_base = layout.fpha_cell_local_start[0];
        let cell_local_a = FphaCellLocal::new(fpha_base);
        let cell_local_b = FphaCellLocal::new(fpha_base + 1);
        let grid = layout.block_grid();
        let row_load = layout.geometry.load_balance.start();

        for blk_idx in 0..layout.clock.n_blks() {
            let blk = BlockIdx::new(blk_idx);
            let col_a = layout.geometry.generation_col(cell_local_a, blk);
            let col_b = layout.geometry.generation_col(cell_local_b, blk);
            let row_a = grid.flat(row_load, bus_pos_a, blk);
            let row_b = grid.flat(row_load, bus_pos_b, blk);
            let row_decoy = grid.flat(row_load, bus_pos_decoy, blk);

            assert_eq!(
                raw_coeff_at(&col_entries, col_a, row_a),
                1.0,
                "blk {blk_idx}: cell A's generation column must carry +1.0 on its own bus row"
            );
            assert_eq!(entry_count_at(&col_entries, col_a, row_a), 1);
            assert_eq!(
                entry_count_at(&col_entries, col_a, row_b),
                0,
                "blk {blk_idx}: cell A must not appear on cell B's bus row"
            );
            assert_eq!(
                entry_count_at(&col_entries, col_a, row_decoy),
                0,
                "blk {blk_idx}: cell A must not credit the decoy bus"
            );

            assert_eq!(
                raw_coeff_at(&col_entries, col_b, row_b),
                1.0,
                "blk {blk_idx}: cell B's generation column must carry +1.0 on its own bus row"
            );
            assert_eq!(entry_count_at(&col_entries, col_b, row_b), 1);
            assert_eq!(
                entry_count_at(&col_entries, col_b, row_a),
                0,
                "blk {blk_idx}: cell B must not appear on cell A's bus row"
            );
            assert_eq!(
                entry_count_at(&col_entries, col_b, row_decoy),
                0,
                "blk {blk_idx}: cell B must not credit the decoy bus"
            );
        }
    }

    /// Requirement 2: the constant-productivity branch credits each cell's
    /// turbined column, at the plant's shared `rho`, to that cell's own bus.
    #[test]
    fn test_constant_productivity_cells_are_credited_per_bus() {
        let production_models = ProductionModelSet::new(
            vec![
                vec![ResolvedProductionModel::ConstantProductivity { productivity: 1.0 }; N_STAGES],
                vec![ResolvedProductionModel::ConstantProductivity { productivity: 0.4 }; N_STAGES],
            ],
            &crate::test_support::minimal_hydros(2),
            N_STAGES,
        );
        let mut fixture = split_bus_fixture(production_models);
        let ctx = fixture.fixtures.make_ctx();
        let stage = three_block_stage(0);
        let layout = StageLayout::new(&ctx, &stage, 0);

        let mut col_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); layout.num_cols];
        fill_load_balance_entries(&ctx, 0, &layout, &mut col_entries);

        let split = HydroSys::new(1);
        let cells: Vec<HydroCell> = ctx
            .hydro_cell_index
            .cells_of(split)
            .map(HydroCell::new)
            .collect();
        let (cell_a, cell_b) = (cells[0], cells[1]);
        let bus_pos_a = ctx.positions.bus(fixture.bus_cell_a).unwrap();
        let bus_pos_b = ctx.positions.bus(fixture.bus_cell_b).unwrap();
        let grid = layout.block_grid();
        let row_load = layout.geometry.load_balance.start();

        for blk_idx in 0..layout.clock.n_blks() {
            let blk = BlockIdx::new(blk_idx);
            let col_turbine_a = layout.geometry.turbine_col(cell_a, blk);
            let col_turbine_b = layout.geometry.turbine_col(cell_b, blk);
            let row_a = grid.flat(row_load, bus_pos_a, blk);
            let row_b = grid.flat(row_load, bus_pos_b, blk);

            assert_eq!(
                raw_coeff_at(&col_entries, col_turbine_a, row_a),
                0.4,
                "blk {blk_idx}: cell A's turbined column must carry the plant's rho on its own bus row"
            );
            assert_eq!(entry_count_at(&col_entries, col_turbine_a, row_a), 1);
            assert_eq!(
                entry_count_at(&col_entries, col_turbine_a, row_b),
                0,
                "blk {blk_idx}: cell A must not appear on cell B's bus row"
            );

            assert_eq!(
                raw_coeff_at(&col_entries, col_turbine_b, row_b),
                0.4,
                "blk {blk_idx}: cell B's turbined column must carry the plant's rho on its own bus row"
            );
            assert_eq!(entry_count_at(&col_entries, col_turbine_b, row_b), 1);
            assert_eq!(
                entry_count_at(&col_entries, col_turbine_b, row_a),
                0,
                "blk {blk_idx}: cell B must not appear on cell A's bus row"
            );
        }
    }

    /// Requirement 3: the production row apportions the plane's flow-independent
    /// part (`gamma_0`, `gamma_v`, `gamma_s`) by each cell's turbine-capacity
    /// share, and leaves `gamma_q` unscaled on the cell's own flow. Shares are
    /// unequal (0.75/0.25) so an even-split or unscaled-everywhere
    /// implementation is caught.
    #[test]
    fn test_production_rows_apportion_the_plane_intercept_by_cell_share() {
        let plane = FphaPlane {
            intercept: 1000.0,
            gamma_v: 4.0,
            gamma_q: 0.6,
            gamma_s: 0.3,
        };
        let mut fixture = split_bus_fixture(split_bus_production_models(vec![plane]));
        let ctx = fixture.fixtures.make_ctx();
        let stage = three_block_stage(0);
        let layout = StageLayout::new(&ctx, &stage, 0);

        let mut col_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); layout.num_cols];
        fill_fpha_entries(&ctx, &stage, 0, &layout, &mut col_entries);
        let (_row_lower, row_upper) = fill_stage_rows(&ctx, &stage, 0, &layout);

        // `assemble_csc` asserts each column is row-ascending, and the plant-level
        // storage and spillage columns take one push per cell here. This call is on
        // the RAW fill output — the higher-level tests sort first, so only this one
        // reaches the assertion. Each cell owns a contiguous row block, so pushes to
        // a shared column stay ascending only while cells are the outer loop; nesting
        // blocks outside cells interleaves the blocks and the sequence decreases.
        let _ = assemble_csc(&col_entries);

        let split = HydroSys::new(1);
        let cells: Vec<HydroCell> = ctx
            .hydro_cell_index
            .cells_of(split)
            .map(HydroCell::new)
            .collect();
        let (cell_a, cell_b) = (cells[0], cells[1]);
        let share_a = ctx.hydro_cell_index.share_of(cell_a);
        let share_b = ctx.hydro_cell_index.share_of(cell_b);
        assert!((share_a - 0.75).abs() < 1e-12);
        assert!((share_b - 0.25).abs() < 1e-12);

        let fpha_base = layout.fpha_cell_local_start[0];
        let cell_local_a = FphaCellLocal::new(fpha_base);
        let cell_local_b = FphaCellLocal::new(fpha_base + 1);
        let grid = layout.block_grid();
        let row_start = layout.geometry.fpha.start;
        let blk = BlockIdx::new(0);
        let row_a = grid.fpha_plane(row_start, blk, 0, 1);
        let row_b = grid.fpha_plane(row_start + layout.clock.n_blks(), blk, 0, 1);

        let col_g_a = layout.geometry.generation_col(cell_local_a, blk);
        let col_g_b = layout.geometry.generation_col(cell_local_b, blk);
        let col_q_a = layout.geometry.turbine_col(cell_a, blk);
        let col_q_b = layout.geometry.turbine_col(cell_b, blk);
        let col_v_in = layout.block_storage_col(split, Boundary::Incoming);
        let col_v_out = layout.block_storage_col(split, Boundary::Outgoing);
        let col_s = layout.geometry.spillage_col(split, blk);

        assert_eq!(
            raw_coeff_at(&col_entries, col_g_a, row_a),
            1.0,
            "cell A's own generation column: +1.0, unscaled"
        );
        assert_eq!(
            raw_coeff_at(&col_entries, col_q_a, row_a),
            -0.6,
            "cell A's own turbined column: -gamma_q, UNSCALED by share"
        );
        assert_eq!(
            raw_coeff_at(&col_entries, col_v_in, row_a),
            0.75 * (-4.0 / 2.0)
        );
        assert_eq!(entry_count_at(&col_entries, col_v_in, row_a), 1);
        assert_eq!(
            raw_coeff_at(&col_entries, col_v_out, row_a),
            0.75 * (-4.0 / 2.0)
        );
        assert_eq!(entry_count_at(&col_entries, col_v_out, row_a), 1);
        assert_eq!(raw_coeff_at(&col_entries, col_s, row_a), 0.75 * -0.3);
        assert_eq!(entry_count_at(&col_entries, col_s, row_a), 1);
        assert_eq!(row_upper[row_a], 0.75 * 1000.0);

        assert_eq!(
            raw_coeff_at(&col_entries, col_g_b, row_b),
            1.0,
            "cell B's own generation column: +1.0, unscaled"
        );
        assert_eq!(
            raw_coeff_at(&col_entries, col_q_b, row_b),
            -0.6,
            "cell B's own turbined column: -gamma_q, UNSCALED by share (same value as cell A's)"
        );
        assert_eq!(
            raw_coeff_at(&col_entries, col_v_in, row_b),
            0.25 * (-4.0 / 2.0)
        );
        assert_eq!(entry_count_at(&col_entries, col_v_in, row_b), 1);
        assert_eq!(
            raw_coeff_at(&col_entries, col_v_out, row_b),
            0.25 * (-4.0 / 2.0)
        );
        assert_eq!(entry_count_at(&col_entries, col_v_out, row_b), 1);
        assert_eq!(raw_coeff_at(&col_entries, col_s, row_b), 0.25 * -0.3);
        assert_eq!(entry_count_at(&col_entries, col_s, row_b), 1);
        assert_eq!(row_upper[row_b], 0.25 * 1000.0);
    }

    /// Build a single-plant FPHA fixture from `unit_groups`, fill its production
    /// row entries and bounds, and return the row block's implied bound on
    /// `Σ_c g_c` at `blk = 0`, `plane = 0`, for the given per-cell flows and
    /// shared `(v_in, v_out, spill)`. `q_per_cell` must have one entry per group.
    fn fpha_implied_aggregate_cap(
        unit_groups: Vec<HydroUnitGroup>,
        plane: FphaPlane,
        q_per_cell: &[f64],
        v_in: f64,
        v_out: f64,
        spill: f64,
    ) -> f64 {
        let n_cells = unit_groups.len();
        assert_eq!(q_per_cell.len(), n_cells);

        let mut hydro = fixture_hydro_ds(10, None);
        hydro.unit_groups = unit_groups;
        let buses: Vec<Bus> = hydro
            .unit_groups
            .iter()
            .map(|g| fixture_bus(g.bus_id.0))
            .collect();

        let production_models = ProductionModelSet::new(
            vec![vec![
                ResolvedProductionModel::Fpha {
                    planes: vec![plane]
                };
                N_STAGES
            ]],
            std::slice::from_ref(&hydro),
            N_STAGES,
        );
        let mut fixtures = PumpFixtures::new_with_buses(vec![hydro], Vec::new(), buses)
            .with_production_models(production_models);
        let ctx = fixtures.make_ctx();
        let stage = three_block_stage(0);
        let layout = StageLayout::new(&ctx, &stage, 0);

        let mut col_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); layout.num_cols];
        fill_fpha_entries(&ctx, &stage, 0, &layout, &mut col_entries);
        let (_row_lower, row_upper) = fill_stage_rows(&ctx, &stage, 0, &layout);

        let plant = HydroSys::new(0);
        let cells: Vec<HydroCell> = ctx
            .hydro_cell_index
            .cells_of(plant)
            .map(HydroCell::new)
            .collect();
        assert_eq!(cells.len(), n_cells);

        let blk = BlockIdx::new(0);
        let grid = layout.block_grid();
        let row_start = layout.geometry.fpha.start;
        let col_v_in = layout.block_storage_col(plant, Boundary::Incoming);
        let col_v_out = layout.block_storage_col(plant, Boundary::Outgoing);
        let col_s = layout.geometry.spillage_col(plant, blk);

        let mut total = 0.0_f64;
        for (i, &cell) in cells.iter().enumerate() {
            let row = grid.fpha_plane(row_start + i * layout.clock.n_blks(), blk, 0, 1);
            let col_q = layout.geometry.turbine_col(cell, blk);
            total += row_upper[row]
                - raw_coeff_at(&col_entries, col_v_in, row) * v_in
                - raw_coeff_at(&col_entries, col_v_out, row) * v_out
                - raw_coeff_at(&col_entries, col_s, row) * spill
                - raw_coeff_at(&col_entries, col_q, row) * q_per_cell[i];
        }
        total
    }

    /// Requirement 6 / the aggregate-cap parity gate: the FPHA row block's
    /// implied bound on `Σ_c g_c`, at fixed `(v_in, v_out, spill, Σ_c q_c)`, is
    /// independent of how the SAME plant's declared capacity partitions into
    /// cells. Tolerance is `n` ULP relative (requirement 6's plain-ratio
    /// residual), generously bounded to also absorb this test's own handful of
    /// floating-point operations — still far tighter than the encodings
    /// requirement 6 rejects (off by a fraction of `A`, not a few ULP of it).
    #[test]
    fn test_aggregate_cap_is_independent_of_the_cell_partition() {
        let plane = FphaPlane {
            intercept: 3000.0,
            gamma_v: 0.6,
            gamma_q: 0.9,
            gamma_s: 0.2,
        };
        let (v_in, v_out, spill) = (40.0, 60.0, 5.0);

        let one_cell = vec![make_unit_group(
            EntityId(600),
            EntityId(60),
            0.0,
            100.0,
            0.0,
            12000.0,
        )];
        let two_cell = vec![
            make_unit_group(EntityId(601), EntityId(61), 0.0, 100.0, 0.0, 9000.0),
            make_unit_group(EntityId(602), EntityId(62), 0.0, 100.0, 0.0, 3000.0),
        ];
        let three_cell = vec![
            make_unit_group(EntityId(603), EntityId(63), 0.0, 100.0, 0.0, 6000.0),
            make_unit_group(EntityId(604), EntityId(64), 0.0, 100.0, 0.0, 3000.0),
            make_unit_group(EntityId(605), EntityId(65), 0.0, 100.0, 0.0, 3000.0),
        ];

        let cap_one = fpha_implied_aggregate_cap(one_cell, plane, &[300.0], v_in, v_out, spill);
        let cap_two =
            fpha_implied_aggregate_cap(two_cell, plane, &[180.0, 120.0], v_in, v_out, spill);
        let cap_three = fpha_implied_aggregate_cap(
            three_cell,
            plane,
            &[100.0, 100.0, 100.0],
            v_in,
            v_out,
            spill,
        );

        let tol = 1.0e4 * f64::EPSILON * cap_one.abs();
        assert!(
            (cap_one - cap_two).abs() < tol,
            "one-cell vs two-cell implied aggregate cap must agree to {tol}: {cap_one} vs {cap_two}"
        );
        assert!(
            (cap_one - cap_three).abs() < tol,
            "one-cell vs three-cell implied aggregate cap must agree to {tol}: {cap_one} vs {cap_three}"
        );
    }

    /// One declared arc `k = [1/2, 1/2]` (depth 1, the plant's only bucket):
    /// the downstream balance row carries `-1/2 * tau_h` on the upstream's own
    /// turbine/spillage columns (NOT `-tau_h`) plus `-1.0` on the plant's
    /// `b_1^in` column, and the single bucket-definition row deposits
    /// `-1/2 * tau_h` into the same upstream columns with `+1.0` on `b_1^out` —
    /// `b_{L+1}^in` does not exist since `d == L_j`.
    #[test]
    fn declared_arc_arrival_split_and_single_definition_row() {
        let up = 1;
        let down = 2;
        let mut fixtures = PumpFixtures::new_full(
            vec![
                fixture_hydro_ds(up, Some(down)),
                fixture_hydro_ds(down, None),
            ],
            Vec::new(),
            vec![fixture_bus(1)],
            Vec::new(),
            Vec::new(),
        );
        let up_idx = fixtures.hydro_pos[&EntityId(up)];
        let down_idx = fixtures.hydro_pos[&EntityId(down)];

        let mut arc_stage_weights = HashMap::new();
        arc_stage_weights.insert(up_idx, vec![vec![0.5, 0.5]]);
        fixtures.base.topology.column_order = vec![(HydroSys::new(down_idx), 1)];
        fixtures.base.topology.arc_stage_weights = arc_stage_weights;
        fixtures.base.topology.per_stage_mask = vec![vec![1]];
        let ctx = fixtures.make_ctx();

        let stage = two_block_stage(0, [300.0, 444.0]);
        let layout = StageLayout::new(&ctx, &stage, 0);
        let csc = build_sorted_csc(&ctx, &stage, 0, &layout);

        let down_row = i32::try_from(layout.geometry.water_balance.start() + down_idx).unwrap();
        let def_row = i32::try_from(layout.rows.transit_bucket_definition.start).unwrap();
        let col_first_slot_in = ctx.state.transit_buckets_in.start;
        let col_first_slot_out = ctx.state.transit_buckets_out.start;

        for blk in 0..layout.clock.n_blks() {
            let tau_h = stage.blocks[blk].duration_hours * M3S_TO_HM3;
            assert_eq!(
                coeff_at(
                    &csc,
                    layout
                        .geometry
                        .turbine_col(HydroCell::new(up_idx), BlockIdx::new(blk)),
                    down_row
                ),
                -0.5 * tau_h,
                "blk {blk}: balance row must carry -k_0*tau_h, not -tau_h"
            );
            assert_eq!(
                coeff_at(
                    &csc,
                    layout
                        .geometry
                        .spillage_col(HydroSys::new(up_idx), BlockIdx::new(blk)),
                    down_row
                ),
                -0.5 * tau_h,
                "blk {blk}: balance row spillage must carry -k_0*tau_h"
            );
            assert_eq!(
                coeff_at(
                    &csc,
                    layout
                        .geometry
                        .turbine_col(HydroCell::new(up_idx), BlockIdx::new(blk)),
                    def_row
                ),
                -0.5 * tau_h,
                "blk {blk}: definition row must carry -k_1*tau_h on the SAME release column"
            );
            assert_eq!(
                coeff_at(
                    &csc,
                    layout
                        .geometry
                        .spillage_col(HydroSys::new(up_idx), BlockIdx::new(blk)),
                    def_row
                ),
                -0.5 * tau_h,
                "blk {blk}: definition row spillage must carry -k_1*tau_h"
            );
        }
        assert_eq!(
            coeff_at(&csc, col_first_slot_in, down_row),
            -1.0,
            "the maturing-now bucket b_1^in must carry -1.0 on the balance row"
        );
        assert_eq!(
            coeff_at(&csc, col_first_slot_out, def_row),
            1.0,
            "the definition row must carry +1.0 on b_1^out"
        );
        // b_{L+1}^in does not exist: the plant's only bucket is depth 1, so
        // transit_buckets_in has exactly one column (no b_2^in to reference).
        assert_eq!(ctx.state.transit_buckets_in.len(), 1);
    }

    /// Row 13 (Filling arm): a `Filling`-phase upstream (turbine/diversion
    /// frozen, spillage FREE — the D40 relief valve) still deposits its
    /// spillage share into the downstream balance row and the bucket
    /// definition row — `fill_arc_release_block_entries` never special-cases
    /// the RELEASING hydro's own commissioning phase, so the `(u+s)` deposit
    /// rides unchanged. Cross-checked against `columns.rs`: the coefficient
    /// carries real flow because the spillage column is actually free at this
    /// stage, not a coefficient on a column pinned to zero.
    #[test]
    fn filling_upstream_spillage_still_deposits_into_transit_bucket() {
        use cobre_core::entities::hydro::FillingConfig;

        let up = 1;
        let down = 2;
        let mut up_hydro = fixture_hydro_ds(up, Some(down));
        up_hydro.filling = Some(FillingConfig {
            start_stage_id: 0,
            filling_min_rate_m3s: 0.0,
        });
        up_hydro.entry_stage_id = Some(5);
        let mut fixtures = PumpFixtures::new_full(
            vec![up_hydro, fixture_hydro_ds(down, None)],
            Vec::new(),
            vec![fixture_bus(1)],
            Vec::new(),
            Vec::new(),
        )
        .with_resolved_penalties();
        let up_idx = fixtures.hydro_pos[&EntityId(up)];
        let down_idx = fixtures.hydro_pos[&EntityId(down)];

        let mut arc_stage_weights = HashMap::new();
        arc_stage_weights.insert(up_idx, vec![vec![0.5, 0.5]]);
        fixtures.base.topology.column_order = vec![(HydroSys::new(down_idx), 1)];
        fixtures.base.topology.arc_stage_weights = arc_stage_weights;
        fixtures.base.topology.per_stage_mask = vec![vec![1]];
        let ctx = fixtures.make_ctx();

        let stage = two_block_stage(0, [300.0, 444.0]);
        let layout = StageLayout::new(&ctx, &stage, 0);
        let csc = build_sorted_csc(&ctx, &stage, 0, &layout);

        let down_row = i32::try_from(layout.geometry.water_balance.start() + down_idx).unwrap();
        let def_row = i32::try_from(layout.rows.transit_bucket_definition.start).unwrap();

        for blk in 0..layout.clock.n_blks() {
            let tau_h = stage.blocks[blk].duration_hours * M3S_TO_HM3;
            assert_eq!(
                coeff_at(
                    &csc,
                    layout
                        .geometry
                        .spillage_col(HydroSys::new(up_idx), BlockIdx::new(blk)),
                    down_row
                ),
                -0.5 * tau_h,
                "blk {blk}: a Filling upstream's spillage must still carry -k_0*tau_h on the \
                 balance row"
            );
            assert_eq!(
                coeff_at(
                    &csc,
                    layout
                        .geometry
                        .spillage_col(HydroSys::new(up_idx), BlockIdx::new(blk)),
                    def_row
                ),
                -0.5 * tau_h,
                "blk {blk}: a Filling upstream's spillage must still deposit -k_1*tau_h into \
                 the bucket"
            );
        }

        let (_col_lower, col_upper, _objective) =
            super::super::columns::fill_stage_columns(&ctx, &stage, 0, &layout);
        for blk in 0..layout.clock.n_blks() {
            assert_eq!(
                col_upper[layout
                    .geometry
                    .spillage_col(HydroSys::new(up_idx), BlockIdx::new(blk))],
                f64::INFINITY,
                "blk {blk}: a Filling upstream's spillage column must stay free (D40), not frozen"
            );
        }
    }

    /// Row 13 (exit arm): an upstream past its own `exit_stage_id` still
    /// emits the SAME arc-release deposit coefficients as an active upstream
    /// — `fill_arc_release_block_entries` never special-cases the RELEASING
    /// hydro's own commissioning phase. Combined with the commissioning
    /// freeze pinning its turbine/spillage columns to `[0, 0]` post-exit
    /// (verified below via `columns.rs`), the realized deposit is zero with
    /// no special-cased code path, so the bucket drains through the pure
    /// ring shift (the state-assembly copy-gap) with no replenishment.
    #[test]
    fn exited_upstream_arc_deposit_coefficients_unchanged_no_special_case() {
        let up = 1;
        let down = 2;
        let mut up_hydro = fixture_hydro_ds(up, Some(down));
        up_hydro.exit_stage_id = Some(1);
        let mut fixtures = PumpFixtures::new_full(
            vec![up_hydro, fixture_hydro_ds(down, None)],
            Vec::new(),
            vec![fixture_bus(1)],
            Vec::new(),
            Vec::new(),
        )
        .with_resolved_penalties();
        let up_idx = fixtures.hydro_pos[&EntityId(up)];
        let down_idx = fixtures.hydro_pos[&EntityId(down)];

        let mut arc_stage_weights = HashMap::new();
        arc_stage_weights.insert(up_idx, vec![vec![0.5, 0.5]]);
        fixtures.base.topology.column_order = vec![(HydroSys::new(down_idx), 1)];
        fixtures.base.topology.arc_stage_weights = arc_stage_weights;
        fixtures.base.topology.per_stage_mask = vec![vec![1]];
        let ctx = fixtures.make_ctx();

        // `stage_idx` stays 0 for both builds (the single-stage fixture's
        // established decoupling from `stage.id`); only `stage.id` moves
        // across the upstream's `exit_stage_id` boundary.
        let mut stage_active = two_block_stage(0, [300.0, 444.0]);
        stage_active.id = 0;
        let layout_active = StageLayout::new(&ctx, &stage_active, 0);
        let csc_active = build_sorted_csc(&ctx, &stage_active, 0, &layout_active);

        let mut stage_exited = two_block_stage(0, [300.0, 444.0]);
        stage_exited.id = 1;
        let layout_exited = StageLayout::new(&ctx, &stage_exited, 0);
        let csc_exited = build_sorted_csc(&ctx, &stage_exited, 0, &layout_exited);

        let down_row_active =
            i32::try_from(layout_active.geometry.water_balance.start() + down_idx).unwrap();
        let def_row_active =
            i32::try_from(layout_active.rows.transit_bucket_definition.start).unwrap();
        let down_row_exited =
            i32::try_from(layout_exited.geometry.water_balance.start() + down_idx).unwrap();
        let def_row_exited =
            i32::try_from(layout_exited.rows.transit_bucket_definition.start).unwrap();

        for blk in 0..layout_active.clock.n_blks() {
            for (col_active, col_exited) in [
                (
                    layout_active
                        .geometry
                        .turbine_col(HydroCell::new(up_idx), BlockIdx::new(blk)),
                    layout_exited
                        .geometry
                        .turbine_col(HydroCell::new(up_idx), BlockIdx::new(blk)),
                ),
                (
                    layout_active
                        .geometry
                        .spillage_col(HydroSys::new(up_idx), BlockIdx::new(blk)),
                    layout_exited
                        .geometry
                        .spillage_col(HydroSys::new(up_idx), BlockIdx::new(blk)),
                ),
            ] {
                assert_eq!(
                    coeff_at(&csc_active, col_active, down_row_active),
                    coeff_at(&csc_exited, col_exited, down_row_exited),
                    "blk {blk}: the balance-row deposit coefficient must not depend on the \
                     upstream's own exit_stage_id"
                );
                assert_eq!(
                    coeff_at(&csc_active, col_active, def_row_active),
                    coeff_at(&csc_exited, col_exited, def_row_exited),
                    "blk {blk}: the bucket-definition deposit coefficient must not depend on \
                     the upstream's own exit_stage_id"
                );
            }
        }

        // The freeze that actually zeroes the realized deposit lives in
        // columns.rs, not here: post-exit, both columns are pinned [0, 0].
        let (_lo_active, col_upper_active, _obj_active) =
            super::super::columns::fill_stage_columns(&ctx, &stage_active, 0, &layout_active);
        let (_lo_exited, col_upper_exited, _obj_exited) =
            super::super::columns::fill_stage_columns(&ctx, &stage_exited, 0, &layout_exited);
        for blk in 0..layout_active.clock.n_blks() {
            assert!(
                col_upper_active[layout_active
                    .geometry
                    .turbine_col(HydroCell::new(up_idx), BlockIdx::new(blk))]
                    > 0.0,
                "blk {blk}: the active upstream's turbine column must be free before exit"
            );
            assert_eq!(
                col_upper_exited[layout_exited
                    .geometry
                    .turbine_col(HydroCell::new(up_idx), BlockIdx::new(blk))],
                0.0,
                "blk {blk}: the exited upstream's turbine column must be pinned to 0"
            );
            assert_eq!(
                col_upper_exited[layout_exited
                    .geometry
                    .spillage_col(HydroSys::new(up_idx), BlockIdx::new(blk))],
                0.0,
                "blk {blk}: the exited upstream's spillage column must be pinned to 0 \
                 (post-exit reverts to PreFilling, which freezes spillage too)"
            );
        }
    }

    /// Confluence: two upstreams feeding one downstream plant sum their
    /// travel-time deposits into the SAME per-plant bucket definition row,
    /// not one row per arc.
    #[test]
    fn confluence_two_upstreams_sum_deposits_into_single_definition_row() {
        let up_a = 1;
        let up_b = 2;
        let down = 3;
        let mut fixtures = PumpFixtures::new_full(
            vec![
                fixture_hydro_ds(up_a, Some(down)),
                fixture_hydro_ds(up_b, Some(down)),
                fixture_hydro_ds(down, None),
            ],
            Vec::new(),
            vec![fixture_bus(1)],
            Vec::new(),
            Vec::new(),
        );
        let up_a_idx = fixtures.hydro_pos[&EntityId(up_a)];
        let up_b_idx = fixtures.hydro_pos[&EntityId(up_b)];
        let down_idx = fixtures.hydro_pos[&EntityId(down)];

        let mut arc_stage_weights = HashMap::new();
        arc_stage_weights.insert(up_a_idx, vec![vec![0.5, 0.5]]);
        arc_stage_weights.insert(up_b_idx, vec![vec![0.25, 0.75]]);
        fixtures.base.topology.column_order = vec![(HydroSys::new(down_idx), 1)];
        fixtures.base.topology.arc_stage_weights = arc_stage_weights;
        fixtures.base.topology.per_stage_mask = vec![vec![1]];
        let ctx = fixtures.make_ctx();

        let stage = two_block_stage(0, [300.0, 444.0]);
        let layout = StageLayout::new(&ctx, &stage, 0);
        let csc = build_sorted_csc(&ctx, &stage, 0, &layout);

        let def_row = i32::try_from(layout.rows.transit_bucket_definition.start).unwrap();
        for blk in 0..layout.clock.n_blks() {
            let tau_h = stage.blocks[blk].duration_hours * M3S_TO_HM3;
            assert_eq!(
                coeff_at(
                    &csc,
                    layout
                        .geometry
                        .turbine_col(HydroCell::new(up_a_idx), BlockIdx::new(blk)),
                    def_row
                ),
                -0.5 * tau_h,
                "blk {blk}: upstream A's deposit must land on the shared definition row"
            );
            assert_eq!(
                coeff_at(
                    &csc,
                    layout
                        .geometry
                        .spillage_col(HydroSys::new(up_a_idx), BlockIdx::new(blk)),
                    def_row
                ),
                -0.5 * tau_h
            );
            assert_eq!(
                coeff_at(
                    &csc,
                    layout
                        .geometry
                        .turbine_col(HydroCell::new(up_b_idx), BlockIdx::new(blk)),
                    def_row
                ),
                -0.75 * tau_h,
                "blk {blk}: upstream B's deposit must land on the SAME shared definition row"
            );
            assert_eq!(
                coeff_at(
                    &csc,
                    layout
                        .geometry
                        .spillage_col(HydroSys::new(up_b_idx), BlockIdx::new(blk)),
                    def_row
                ),
                -0.75 * tau_h
            );
        }
        // A single aggregated bucket for the downstream plant, not one per arc.
        assert_eq!(ctx.state.n_buckets, 1);
        assert_eq!(ctx.state.transit_buckets_out.len(), 1);
    }

    /// Equivalence pin: `fill_transit_bucket_definition_entries`'s per-plant
    /// `DeliveryRing::emit_shift_rows` routing reproduces the open-coded reference
    /// formula (`+1` on `b_d^out`, `-1` on `b_{d+1}^in` only within the SAME plant's
    /// own contiguous group) on a fixed three-plant fixture: one plant with 3 lags,
    /// one with 1 lag, and one with no declared arc at all (absent from `column_order`).
    #[test]
    fn fill_transit_bucket_definition_entries_matches_pre_migration_formula_across_ragged_plants() {
        let h_down3 = 10;
        let h_down1 = 20;
        let h_none = 30;
        let mut fixtures = PumpFixtures::new_full(
            vec![
                fixture_hydro_ds(h_down3, None),
                fixture_hydro_ds(h_down1, None),
                fixture_hydro_ds(h_none, None),
            ],
            Vec::new(),
            vec![fixture_bus(1)],
            Vec::new(),
            Vec::new(),
        );
        let down3_idx = fixtures.hydro_pos[&EntityId(h_down3)];
        let down1_idx = fixtures.hydro_pos[&EntityId(h_down1)];

        fixtures.base.topology.column_order = vec![
            (HydroSys::new(down3_idx), 1),
            (HydroSys::new(down3_idx), 2),
            (HydroSys::new(down3_idx), 3),
            (HydroSys::new(down1_idx), 1),
        ];
        fixtures.base.topology.per_stage_mask = vec![vec![3, 1]];
        let ctx = fixtures.make_ctx();
        let stage = two_block_stage(0, [300.0, 444.0]);
        let layout = StageLayout::new(&ctx, &stage, 0);

        let mut col_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); layout.num_cols];
        fill_transit_bucket_definition_entries(&layout, &mut col_entries);

        let row_start = layout.rows.transit_bucket_definition.start;
        let out = ctx.state.transit_buckets_out.start;
        let inn = ctx.state.transit_buckets_in.start;

        // down3 slot 0 (lag 1): a deeper own-plant slot (lag 2) exists.
        assert_eq!(col_entries[out], vec![(row_start, 1.0)]);
        assert_eq!(col_entries[inn + 1], vec![(row_start, -1.0)]);
        // down3 slot 1 (lag 2): a deeper own-plant slot (lag 3) exists.
        assert_eq!(col_entries[out + 1], vec![(row_start + 1, 1.0)]);
        assert_eq!(col_entries[inn + 2], vec![(row_start + 1, -1.0)]);
        // down3 slot 2 (lag 3, its own last lag): the next global slot belongs
        // to down1, so no shift term crosses the plant boundary.
        assert_eq!(col_entries[out + 2], vec![(row_start + 2, 1.0)]);
        assert!(col_entries[inn + 3].is_empty());
        // down1 slot 3 (lag 1, its only lag, also the last global slot): no
        // deeper slot exists at all.
        assert_eq!(col_entries[out + 3], vec![(row_start + 3, 1.0)]);
    }

    /// `B == 0` (no declared arc, `state.n_buckets == 0`): the emitted water
    /// entries are byte-identical to today's shape — no bucket-definition rows
    /// exist at all (`load_balance` collapses back onto `water_balance.end`)
    /// and the upstream release carries exactly `-tau_h` (the `n_buckets` == 0
    /// byte-identity anchor).
    #[test]
    fn b_zero_water_entries_are_byte_identical_to_undeclared_arc() {
        let up = 1;
        let down = 2;
        let mut fixtures = PumpFixtures::new_full(
            vec![
                fixture_hydro_ds(up, Some(down)),
                fixture_hydro_ds(down, None),
            ],
            Vec::new(),
            vec![fixture_bus(1)],
            Vec::new(),
            Vec::new(),
        );
        let up_idx = fixtures.hydro_pos[&EntityId(up)];
        let down_idx = fixtures.hydro_pos[&EntityId(down)];

        let ctx = fixtures.make_ctx();
        let stage = two_block_stage(0, [300.0, 444.0]);
        assert_eq!(
            ctx.state.n_buckets, 0,
            "fixture must declare no travel-time arc"
        );
        let layout = StageLayout::new(&ctx, &stage, 0);

        // No bucket-row gap: load_balance starts exactly where water_balance
        // ends, and the bucket-definition row cursor collapses onto it.
        assert_eq!(
            layout.rows.transit_bucket_definition.start,
            layout.geometry.load_balance.start(),
            "B==0 must leave no bucket-definition rows between water_balance and load_balance"
        );
        assert_eq!(
            layout.geometry.load_balance.start(),
            layout.geometry.water_balance.start() + layout.state.hydro_count,
            "B==0 must reproduce today's row_water_balance_start + n_hydros offset"
        );

        let csc = build_sorted_csc(&ctx, &stage, 0, &layout);
        let down_row = i32::try_from(layout.geometry.water_balance.start() + down_idx).unwrap();
        for blk in 0..layout.clock.n_blks() {
            let tau_h = stage.blocks[blk].duration_hours * M3S_TO_HM3;
            assert_eq!(
                coeff_at(
                    &csc,
                    layout
                        .geometry
                        .turbine_col(HydroCell::new(up_idx), BlockIdx::new(blk)),
                    down_row
                ),
                -tau_h,
                "blk {blk}: undeclared arc must carry exactly -tau_h (today's shape)"
            );
            assert_eq!(
                coeff_at(
                    &csc,
                    layout
                        .geometry
                        .spillage_col(HydroSys::new(up_idx), BlockIdx::new(blk)),
                    down_row
                ),
                -tau_h,
                "blk {blk}: undeclared arc must carry exactly -tau_h (today's shape)"
            );
        }
    }

    /// The conservation `debug_assert` fires when a declared arc's `k` does not
    /// sum to 1.0 — the guard is real, not dead code.
    #[test]
    #[should_panic(expected = "stage-clock weights must sum to 1.0")]
    fn declared_arc_non_conserving_k_panics_in_debug() {
        let up = 1;
        let down = 2;
        let mut fixtures = PumpFixtures::new_full(
            vec![
                fixture_hydro_ds(up, Some(down)),
                fixture_hydro_ds(down, None),
            ],
            Vec::new(),
            vec![fixture_bus(1)],
            Vec::new(),
            Vec::new(),
        );
        let up_idx = fixtures.hydro_pos[&EntityId(up)];
        let down_idx = fixtures.hydro_pos[&EntityId(down)];

        let mut arc_stage_weights = HashMap::new();
        arc_stage_weights.insert(up_idx, vec![vec![0.5, 0.3]]); // sums to 0.8: violates conservation.
        fixtures.base.topology.column_order = vec![(HydroSys::new(down_idx), 1)];
        fixtures.base.topology.arc_stage_weights = arc_stage_weights;
        fixtures.base.topology.per_stage_mask = vec![vec![1]];
        let ctx = fixtures.make_ctx();

        let stage = two_block_stage(0, [300.0, 444.0]);
        let layout = StageLayout::new(&ctx, &stage, 0);

        let _ = build_stage_matrix_entries(&ctx, &stage, 0, &layout);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Water travel-time: chronological block-resolved deposits/routing/delivery
    // ─────────────────────────────────────────────────────────────────────────

    /// A chronological `Stage` with the given per-block durations, otherwise
    /// mirroring [`two_block_stage`]'s fixture defaults.
    fn chronological_stage(index: usize, block_hours: &[f64]) -> Stage {
        let mut stage = two_block_stage(
            index,
            [
                block_hours[0],
                block_hours.get(1).copied().unwrap_or(block_hours[0]),
            ],
        );
        stage.blocks = block_hours
            .iter()
            .enumerate()
            .map(|(i, &h)| cobre_core::Block {
                index: i,
                name: format!("B{i}"),
                duration_hours: h,
            })
            .collect();
        stage.block_mode = BlockMode::Chronological;
        stage
    }

    /// A `t_v = 250h` travel time against a 720h stage of 3×240h
    /// chronological blocks. Pins
    /// `κ_{B0→B1}=230/240, κ_{B0→B2}=10/240, κ_{B1→B2}=230/240,
    /// χ=(0, 10/240, 1)`-shaped on the emitted matrix entries.
    #[test]
    fn example_iii_kappa_and_chi_match_worked_numbers() {
        let up = 1;
        let down = 2;
        let mut fixtures = PumpFixtures::new_full(
            vec![
                fixture_hydro_ds(up, Some(down)),
                fixture_hydro_ds(down, None),
            ],
            Vec::new(),
            vec![fixture_bus(1)],
            Vec::new(),
            Vec::new(),
        );
        let up_idx = fixtures.hydro_pos[&EntityId(up)];
        let down_idx = fixtures.hydro_pos[&EntityId(down)];

        let t_v = 250.0_f64;
        let block_hours = [240.0, 240.0, 240.0];
        let resolution = resolve_spread(t_v, 0, &[720.0, 720.0], Some(&block_hours));

        assert!((resolution.within_stage_routing[0][1] - 230.0 / 240.0).abs() < 1e-9);
        assert!((resolution.within_stage_routing[0][2] - 10.0 / 240.0).abs() < 1e-9);
        assert!((resolution.within_stage_routing[1][1] - 230.0 / 240.0).abs() < 1e-9);
        assert!(resolution.block_deposits[0][1].abs() < 1e-9);
        assert!((resolution.block_deposits[1][1] - 10.0 / 240.0).abs() < 1e-9);
        assert!((resolution.block_deposits[2][1] - 1.0).abs() < 1e-9);

        let mut arc_spread_chrono = HashMap::new();
        arc_spread_chrono.insert(up_idx, vec![Some(resolution)]);
        fixtures.base.topology.column_order = vec![(HydroSys::new(down_idx), 1)];
        fixtures.base.topology.arc_spread_chrono = arc_spread_chrono;
        fixtures.base.topology.per_stage_mask = vec![vec![1]];
        let ctx = fixtures.make_ctx();

        let stage = chronological_stage(0, &block_hours);
        let layout = StageLayout::new(&ctx, &stage, 0);
        let csc = build_sorted_csc(&ctx, &stage, 0, &layout);

        let def_row = i32::try_from(layout.rows.transit_bucket_definition.start).unwrap();
        let row_water = layout.geometry.water_balance.start();
        let row_b1 = i32::try_from(row_water + down_idx * 3 + 1).unwrap();
        let row_b2 = i32::try_from(row_water + down_idx * 3 + 2).unwrap();
        let tau = |b: usize| stage.blocks[b].duration_hours * M3S_TO_HM3;

        // Block B0: routes 230/240 to B1, 10/240 to B2; no crossing deposit.
        assert_eq!(
            coeff_at(
                &csc,
                layout
                    .geometry
                    .turbine_col(HydroCell::new(up_idx), BlockIdx::new(0)),
                row_b1
            ),
            -(230.0 / 240.0) * tau(0)
        );
        assert_eq!(
            coeff_at(
                &csc,
                layout
                    .geometry
                    .spillage_col(HydroSys::new(up_idx), BlockIdx::new(0)),
                row_b1
            ),
            -(230.0 / 240.0) * tau(0)
        );
        assert_eq!(
            coeff_at(
                &csc,
                layout
                    .geometry
                    .turbine_col(HydroCell::new(up_idx), BlockIdx::new(0)),
                row_b2
            ),
            -(10.0 / 240.0) * tau(0)
        );
        assert_eq!(
            coeff_at(
                &csc,
                layout
                    .geometry
                    .turbine_col(HydroCell::new(up_idx), BlockIdx::new(0)),
                def_row
            ),
            0.0
        );

        // Block B1: routes 230/240 to B2; deposits 10/240 into the bucket.
        assert_eq!(
            coeff_at(
                &csc,
                layout
                    .geometry
                    .turbine_col(HydroCell::new(up_idx), BlockIdx::new(1)),
                row_b2
            ),
            -(230.0 / 240.0) * tau(1)
        );
        assert_eq!(
            coeff_at(
                &csc,
                layout
                    .geometry
                    .turbine_col(HydroCell::new(up_idx), BlockIdx::new(1)),
                def_row
            ),
            -(10.0 / 240.0) * tau(1)
        );

        // Block B2: nothing in-stage; deposits its full release into the bucket.
        assert_eq!(
            coeff_at(
                &csc,
                layout
                    .geometry
                    .turbine_col(HydroCell::new(up_idx), BlockIdx::new(2)),
                def_row
            ),
            -tau(2)
        );
        assert_eq!(
            coeff_at(
                &csc,
                layout
                    .geometry
                    .spillage_col(HydroSys::new(up_idx), BlockIdx::new(2)),
                def_row
            ),
            -tau(2)
        );
    }

    /// `K = 1` (a single block spanning the whole stage) with travel time ON is
    /// byte-identical to the parallel fill (the K=1 parity anchor):
    /// `χ_{0,d} = k_d` collapses κ to self-routing and χ to the parallel deposit.
    #[test]
    fn k1_chronological_with_travel_time_is_byte_identical_to_parallel() {
        let up = 1;
        let down = 2;
        let t_v = 250.0_f64;
        let stage_durations = [720.0, 720.0];

        let make_fixtures = || {
            PumpFixtures::new_full(
                vec![
                    fixture_hydro_ds(up, Some(down)),
                    fixture_hydro_ds(down, None),
                ],
                Vec::new(),
                vec![fixture_bus(1)],
                Vec::new(),
                Vec::new(),
            )
        };
        let up_idx = make_fixtures().hydro_pos[&EntityId(up)];
        let down_idx = make_fixtures().hydro_pos[&EntityId(down)];

        let resolution = resolve_spread(t_v, 0, &stage_durations, Some(&[720.0]));
        let stage_weights = resolution.stage_weights.clone();

        // Parallel build. `chronological_stage` (not `two_block_stage`, which
        // always carries 2 blocks) gives a single 720h block so `n_blks == 1`
        // on both sides of the comparison; the mode is then forced back to
        // `Parallel` for this side.
        let mut par_fixtures = make_fixtures();
        let mut arc_stage_weights = HashMap::new();
        arc_stage_weights.insert(up_idx, vec![stage_weights]);
        par_fixtures.base.topology.column_order = vec![(HydroSys::new(down_idx), 1)];
        par_fixtures.base.topology.arc_stage_weights = arc_stage_weights;
        par_fixtures.base.topology.per_stage_mask = vec![vec![1]];
        let par_ctx = par_fixtures.make_ctx();
        let mut par_stage = chronological_stage(0, &[720.0]);
        par_stage.block_mode = BlockMode::Parallel;
        let par_layout = StageLayout::new(&par_ctx, &par_stage, 0);
        let par_csc = build_sorted_csc(&par_ctx, &par_stage, 0, &par_layout);

        // Chronological build (K=1), same arc data via arc_spread_chrono.
        let mut chr_fixtures = make_fixtures();
        let mut arc_spread_chrono = HashMap::new();
        arc_spread_chrono.insert(up_idx, vec![Some(resolution)]);
        chr_fixtures.base.topology.column_order = vec![(HydroSys::new(down_idx), 1)];
        chr_fixtures.base.topology.arc_spread_chrono = arc_spread_chrono;
        chr_fixtures.base.topology.per_stage_mask = vec![vec![1]];
        let chr_ctx = chr_fixtures.make_ctx();
        let chr_stage = chronological_stage(0, &[720.0]);
        let chr_layout = StageLayout::new(&chr_ctx, &chr_stage, 0);
        let chr_csc = build_sorted_csc(&chr_ctx, &chr_stage, 0, &chr_layout);

        assert_eq!(
            par_layout.num_cols, chr_layout.num_cols,
            "K=1 must carry the same column count in both modes"
        );
        assert_eq!(
            par_layout.rows.num_rows, chr_layout.rows.num_rows,
            "K=1 must carry the same row count in both modes"
        );
        assert_eq!(
            par_csc, chr_csc,
            "K=1 chronological with travel time ON must be byte-identical to parallel"
        );
    }

    /// The `Σ_d k_d == 1.0` stage-clock conservation (row 8) fires at the
    /// CHRONOLOGICAL deposit site too, independently of the per-column and
    /// aggregation identities below it — mirrors the parallel-site guard
    /// (`declared_arc_non_conserving_k_panics_in_debug`), closing the gap
    /// where only the chrono-specific identities were fill-time-asserted.
    #[test]
    #[should_panic(expected = "stage-clock weights must sum to 1.0")]
    fn row_8_chrono_stage_clock_sum_panics_on_disagreement() {
        let up = 1;
        let down = 2;
        let mut fixtures = PumpFixtures::new_full(
            vec![
                fixture_hydro_ds(up, Some(down)),
                fixture_hydro_ds(down, None),
            ],
            Vec::new(),
            vec![fixture_bus(1)],
            Vec::new(),
            Vec::new(),
        );
        let up_idx = fixtures.hydro_pos[&EntityId(up)];
        let down_idx = fixtures.hydro_pos[&EntityId(down)];

        // stage_weights = [0.3, 0.3] sums to 0.6 — violates row 8's
        // stage-clock conservation, even though within_stage_routing's/
        // block_deposits's own per-column identity (0.5 + 0.5 == 1.0) holds.
        let bad_resolution = SpreadResolution {
            stage_reach: 1,
            stage_weights: vec![0.3, 0.3],
            block_deposits: vec![vec![0.5, 0.5]],
            within_stage_routing: vec![vec![0.5]],
            arrival_density: vec![vec![1.0]],
        };
        let mut arc_spread_chrono = HashMap::new();
        arc_spread_chrono.insert(up_idx, vec![Some(bad_resolution)]);
        fixtures.base.topology.column_order = vec![(HydroSys::new(down_idx), 1)];
        fixtures.base.topology.arc_spread_chrono = arc_spread_chrono;
        fixtures.base.topology.per_stage_mask = vec![vec![1]];
        let ctx = fixtures.make_ctx();

        let stage = chronological_stage(0, &[720.0]);
        let layout = StageLayout::new(&ctx, &stage, 0);

        let _ = build_stage_matrix_entries(&ctx, &stage, 0, &layout);
    }

    /// The `Σ_b w_b·χ_{b,d} == k_d` shared-density consistency (row 9) fires
    /// when a hand-built resolution's block deposits disagree with its own
    /// stage-level `stage_weights` — the guard is real, not dead code.
    #[test]
    #[should_panic(expected = "block deposits must aggregate to k_d")]
    fn row_9_shared_density_consistency_panics_on_disagreement() {
        let up = 1;
        let down = 2;
        let mut fixtures = PumpFixtures::new_full(
            vec![
                fixture_hydro_ds(up, Some(down)),
                fixture_hydro_ds(down, None),
            ],
            Vec::new(),
            vec![fixture_bus(1)],
            Vec::new(),
            Vec::new(),
        );
        let up_idx = fixtures.hydro_pos[&EntityId(up)];
        let down_idx = fixtures.hydro_pos[&EntityId(down)];

        // k_1 = 0.5 but the single block's block_deposits value is 0.0 at
        // lag 1 — violates the shared-density aggregation identity.
        let bad_resolution = SpreadResolution {
            stage_reach: 1,
            stage_weights: vec![0.5, 0.5],
            block_deposits: vec![vec![1.0, 0.0]],
            within_stage_routing: vec![vec![1.0]],
            arrival_density: vec![vec![1.0]],
        };
        let mut arc_spread_chrono = HashMap::new();
        arc_spread_chrono.insert(up_idx, vec![Some(bad_resolution)]);
        fixtures.base.topology.column_order = vec![(HydroSys::new(down_idx), 1)];
        fixtures.base.topology.arc_spread_chrono = arc_spread_chrono;
        fixtures.base.topology.per_stage_mask = vec![vec![1]];
        let ctx = fixtures.make_ctx();

        let stage = chronological_stage(0, &[720.0]);
        let layout = StageLayout::new(&ctx, &stage, 0);

        let _ = build_stage_matrix_entries(&ctx, &stage, 0, &layout);
    }

    /// `resolve_bucket_arrival_density` returns the precomputed arrival-frame
    /// `arc_arrival_density` table entry verbatim — a lookup, not a
    /// re-derivation from the sender's own lag-1 row.
    #[test]
    fn resolve_bucket_arrival_density_looks_up_arrival_frame_table() {
        let up = 1;
        let down = 2;
        let mut fixtures = PumpFixtures::new_full(
            vec![
                fixture_hydro_ds(up, Some(down)),
                fixture_hydro_ds(down, None),
            ],
            Vec::new(),
            vec![fixture_bus(1)],
            Vec::new(),
            Vec::new(),
        );
        let up_idx = fixtures.hydro_pos[&EntityId(up)];

        let table_density = vec![0.3, 0.7];
        let mut arc_arrival_density = HashMap::new();
        arc_arrival_density.insert(up_idx, vec![None, Some(table_density.clone())]);
        fixtures.base.topology.arc_arrival_density = arc_arrival_density;
        let ctx = fixtures.make_ctx();

        let stage = chronological_stage(1, &[300.0, 420.0]);
        let resolved =
            resolve_bucket_arrival_density(&ctx, BlockClock::new(&stage), 1, EntityId(down), 2);

        assert_eq!(
            resolved, table_density,
            "must return the precomputed arrival-frame table entry, not a re-derived density"
        );
    }

    /// The genuine no-arc/first-stage default: the study's first stage has no
    /// source stage to blend from, so `arc_arrival_density` carries no entry
    /// (mirrors the real setup precompute's `None` at stage 0) and the
    /// fallback is the duration-weighted uniform density.
    #[test]
    fn resolve_bucket_arrival_density_falls_back_to_uniform_when_table_entry_absent() {
        let up = 1;
        let down = 2;
        let mut fixtures = PumpFixtures::new_full(
            vec![
                fixture_hydro_ds(up, Some(down)),
                fixture_hydro_ds(down, None),
            ],
            Vec::new(),
            vec![fixture_bus(1)],
            Vec::new(),
            Vec::new(),
        );
        let ctx = fixtures.make_ctx();

        let stage = chronological_stage(0, &[300.0, 420.0]);
        let resolved =
            resolve_bucket_arrival_density(&ctx, BlockClock::new(&stage), 0, EntityId(down), 2);

        assert_eq!(
            resolved,
            vec![300.0 / 720.0, 420.0 / 720.0],
            "must fall back to the duration-weighted uniform density"
        );
    }

    /// A plain (no-travel-time) tributary into a bucketed confluence is EXCLUDED
    /// from the arrival-density resolution: `build_arc_arrival_density` inserts
    /// declared travel-time arcs only, so the plain tributary has no table entry
    /// and the resolver skips it, returning the sole travel-time arc's density —
    /// never a false heterogeneous-confluence panic (debug) nor a wrong uniform
    /// split (release). The plain tributary sorts first in `cascade.upstream`
    /// (id 0 < id 1), so the pre-fix code would seed `chosen` with the uniform
    /// fallback before the travel-time arc disagrees. The `cobre-io`
    /// `check_chronological_confluence_heterogeneous_travel_time` gate cannot
    /// catch this: it counts travel-time arcs only, so one arc plus one plain
    /// tributary is `< 2` and passes config validation.
    #[test]
    fn resolve_bucket_arrival_density_excludes_plain_tributary_from_confluence() {
        let plain = 0;
        let up = 1;
        let down = 2;
        let mut fixtures = PumpFixtures::new_full(
            vec![
                fixture_hydro_ds(plain, Some(down)),
                fixture_hydro_ds(up, Some(down)),
                fixture_hydro_ds(down, None),
            ],
            Vec::new(),
            vec![fixture_bus(1)],
            Vec::new(),
            Vec::new(),
        );
        let up_idx = fixtures.hydro_pos[&EntityId(up)];

        // Only the travel-time arc is in the arrival-frame table, and its entry
        // is deliberately non-uniform so the pre-fix uniform fallback disagrees.
        let arrival_density = vec![0.25, 0.75];
        let mut arc_arrival_density = HashMap::new();
        arc_arrival_density.insert(up_idx, vec![None, Some(arrival_density.clone())]);
        fixtures.base.topology.arc_arrival_density = arc_arrival_density;
        let ctx = fixtures.make_ctx();

        let stage = chronological_stage(1, &[300.0, 420.0]);
        let resolved =
            resolve_bucket_arrival_density(&ctx, BlockClock::new(&stage), 1, EntityId(down), 2);

        assert_eq!(
            resolved, arrival_density,
            "the plain tributary must be skipped; the travel-time arc's density wins"
        );
    }

    /// `fill_parallel_water_entries` never reads `arc_arrival_density`: the
    /// maturing bucket's parallel entry stays a single `-1.0` (the confluence
    /// sum lives in the state variable, not a density split) regardless of
    /// what the arrival-frame table holds for the arc at this stage.
    #[test]
    fn fill_parallel_water_entries_ignores_arc_arrival_density() {
        let up = 1;
        let down = 2;
        let mut fixtures = PumpFixtures::new_full(
            vec![
                fixture_hydro_ds(up, Some(down)),
                fixture_hydro_ds(down, None),
            ],
            Vec::new(),
            vec![fixture_bus(1)],
            Vec::new(),
            Vec::new(),
        );
        let up_idx = fixtures.hydro_pos[&EntityId(up)];
        let down_idx = fixtures.hydro_pos[&EntityId(down)];

        // A deliberately non-uniform entry: if the parallel fill read this
        // table at all, the bucket's maturing entry would carry something
        // other than -1.0.
        let mut arc_arrival_density = HashMap::new();
        arc_arrival_density.insert(up_idx, vec![Some(vec![0.9, 0.1])]);
        fixtures.base.topology.column_order = vec![(HydroSys::new(down_idx), 1)];
        fixtures.base.topology.arc_arrival_density = arc_arrival_density;
        fixtures.base.topology.per_stage_mask = vec![vec![1]];
        let ctx = fixtures.make_ctx();

        let stage = two_block_stage(0, [300.0, 420.0]);
        let layout = StageLayout::new(&ctx, &stage, 0);
        let csc = build_sorted_csc(&ctx, &stage, 0, &layout);

        let row_water = i32::try_from(layout.geometry.water_balance.start() + down_idx).unwrap();
        let col_first_slot_in = ctx.state.transit_buckets_in.start;
        assert_eq!(
            coeff_at(&csc, col_first_slot_in, row_water),
            -1.0,
            "the parallel maturing-bucket entry must stay a single -1.0, \
             independent of arc_arrival_density"
        );
    }

    /// The `Σ_b arrival_density_b == 1` conservation `debug_assert` in
    /// [`fill_chronological_water_entries`] still fires when a hand-built
    /// `arc_arrival_density` table entry violates conservation — the guard is
    /// real, not dead code, after the arrival-frame lookup swap.
    #[test]
    #[should_panic(expected = "arrival_density must sum to 1.0")]
    fn fill_chronological_water_entries_arrival_density_conservation_panics_on_disagreement() {
        let up = 1;
        let down = 2;
        let mut fixtures = PumpFixtures::new_full(
            vec![
                fixture_hydro_ds(up, Some(down)),
                fixture_hydro_ds(down, None),
            ],
            Vec::new(),
            vec![fixture_bus(1)],
            Vec::new(),
            Vec::new(),
        );
        let up_idx = fixtures.hydro_pos[&EntityId(up)];
        let down_idx = fixtures.hydro_pos[&EntityId(down)];

        // Deliberately non-conserving: sums to 0.6, not 1.0.
        let mut arc_arrival_density = HashMap::new();
        arc_arrival_density.insert(up_idx, vec![Some(vec![0.3, 0.3])]);
        fixtures.base.topology.column_order = vec![(HydroSys::new(down_idx), 1)];
        fixtures.base.topology.arc_arrival_density = arc_arrival_density;
        fixtures.base.topology.per_stage_mask = vec![vec![1]];
        let ctx = fixtures.make_ctx();

        let stage = chronological_stage(0, &[300.0, 420.0]);
        let layout = StageLayout::new(&ctx, &stage, 0);

        let _ = build_stage_matrix_entries(&ctx, &stage, 0, &layout);
    }

    /// End-to-end: a generic constraint referencing `pumping_flow` and
    /// `pumping_power` resolves to the REAL pumping column(s) through the
    /// resolver's sole caller (`fill_generic_constraint_entries`), and the
    /// constraint participates in the LP — its row carries CSC entries on the
    /// pumping columns. The `block_id = None` expression is block-dependent, so it
    /// expands to one generic row per block; each row's two terms (flow ×1.0 and
    /// power ×consumption) alias the SAME pumping column for that block, so the
    /// summed coefficient at `(pumping_col, generic_row)` is `1.0 + consumption`.
    #[test]
    fn b6b_generic_constraint_resolves_pumping_columns_in_lp() {
        let consumption = 0.5_f64;
        let constraint_id = EntityId(7);
        let station_id = EntityId(10);

        // pumping_flow(10) + pumping_power(10) <= 40 (block_id = None on both).
        let constraint = GenericConstraint {
            id: constraint_id,
            name: "gc_pump".to_string(),
            description: None,
            expression: ConstraintExpression {
                terms: vec![
                    LinearTerm {
                        coefficient: CoefficientRef::Literal(1.0),
                        scale: 1.0,
                        variable: VariableRef::PumpingFlow {
                            station_id,
                            block_id: None,
                        },
                    },
                    LinearTerm {
                        coefficient: CoefficientRef::Literal(1.0),
                        scale: 1.0,
                        variable: VariableRef::PumpingPower {
                            station_id,
                            block_id: None,
                        },
                    },
                ],
            },
            slack: SlackConfig {
                enabled: false,
                penalty: None,
            },
            bound_lower_affine: None,
            bound_upper_affine: None,
        };

        let mut fixtures = PumpFixtures::new(
            vec![fixture_hydro(1), fixture_hydro(2)],
            vec![station_full(station_id.0, 1, 2, 0.0, 50.0, 1, consumption)],
        )
        .with_generic_constraint(constraint, 40.0);
        let ctx = fixtures.make_ctx();
        let stage = two_block_stage(0, [300.0, 444.0]);
        let layout = StageLayout::new(&ctx, &stage, 0);

        // Block-dependent expression with block_id = None expands to one generic
        // row per block, so the constraint participates as `n_blks` rows.
        let n_blks = layout.clock.n_blks();
        assert_eq!(
            layout.rows.n_generic_rows, n_blks,
            "block-dependent pumping constraint must expand to one row per block"
        );

        let mut col_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); layout.num_cols];
        let mut col_upper = vec![f64::INFINITY; layout.num_cols];
        let mut objective = vec![0.0_f64; layout.num_cols];
        let mut row_lower = vec![f64::NEG_INFINITY; layout.rows.num_rows];
        let mut row_upper = vec![f64::INFINITY; layout.rows.num_rows];
        let mut buffers = LpMatrixBuffers {
            col_entries: &mut col_entries,
            col_upper: &mut col_upper,
            objective: &mut objective,
            row_lower: &mut row_lower,
            row_upper: &mut row_upper,
        };

        fill_generic_constraint_entries(&ctx, 0, &layout, &mut buffers);

        // Each generic row `blk` lands on the station's flow column for that block,
        // with the flow (1.0) and power (consumption) terms aliasing the SAME column.
        // p_idx = 0 (the only station), so col = col_pumping_start + blk.
        for blk in 0..n_blks {
            let row = layout.rows.row_generic_start + blk;
            let col = layout.geometry.pumping_flow.start + blk;
            let summed: f64 = col_entries[col]
                .iter()
                .filter(|&&(r, _)| r == row)
                .map(|&(_, v)| v)
                .sum();
            assert_eq!(
                summed,
                1.0 + consumption,
                "blk {blk}: pumping column {col} must carry flow(1.0) + power({consumption}) on generic row {row}"
            );
            // The row bound proves the constraint participates as an upper-only row.
            assert_eq!(
                row_upper[row], 40.0,
                "blk {blk}: upper-only row upper bound"
            );
            assert_eq!(
                row_lower[row],
                f64::NEG_INFINITY,
                "blk {blk}: upper-only row lower"
            );
        }
    }

    /// AC: a two-sided row (both `bound_lower` and `bound_upper` present) carries
    /// `row_lower = bound_lower` and `row_upper = bound_upper`, and — with slack
    /// enabled — TWO slack columns (plus then minus): the two-sidedness test in
    /// `allocate_generic_slack_cols` must derive from the row's own endpoint pair,
    /// not a constraint-level label, or a two-sided row's upper bound would carry
    /// no relaxation and no error would surface anywhere downstream.
    #[test]
    fn two_sided_row_bounds_and_two_slack_columns() {
        let station_id = EntityId(10);
        let penalty = 25.0_f64;
        let constraint = GenericConstraint {
            id: EntityId(7),
            name: "gc_range".to_string(),
            description: None,
            expression: ConstraintExpression {
                terms: vec![LinearTerm {
                    coefficient: CoefficientRef::Literal(1.0),
                    scale: 1.0,
                    variable: VariableRef::PumpingFlow {
                        station_id,
                        block_id: None,
                    },
                }],
            },
            slack: SlackConfig {
                enabled: true,
                penalty: Some(penalty),
            },
            bound_lower_affine: None,
            bound_upper_affine: None,
        };

        let mut fixtures = PumpFixtures::new(
            vec![fixture_hydro(1), fixture_hydro(2)],
            vec![station_full(station_id.0, 1, 2, 0.0, 50.0, 1, 0.5)],
        )
        .with_generic_constraint_range(constraint, 5.0, 20.0);
        let ctx = fixtures.make_ctx();
        let block_hours = [300.0, 444.0];
        let stage = two_block_stage(0, block_hours);
        let layout = StageLayout::new(&ctx, &stage, 0);

        let n_blks = layout.clock.n_blks();
        assert_eq!(
            layout.rows.n_generic_rows, n_blks,
            "block-dependent two-sided constraint must expand to one row per block"
        );
        assert_eq!(
            layout.generic_constraint_rows.len(),
            n_blks,
            "one GenericConstraintRowEntry per block"
        );

        for (blk, entry) in layout.generic_constraint_rows.iter().enumerate() {
            let plus_col = entry
                .slack_plus_col
                .unwrap_or_else(|| panic!("blk {blk}: a slack-enabled row must get a plus column"));
            let minus_col = entry.slack_minus_col.unwrap_or_else(|| {
                panic!(
                    "blk {blk}: a two-sided row with slack enabled must get a minus slack column"
                )
            });
            assert_eq!(
                minus_col,
                plus_col + 1,
                "blk {blk}: minus slack must be a DISTINCT column immediately after plus slack"
            );
        }

        let mut col_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); layout.num_cols];
        let mut col_upper = vec![f64::INFINITY; layout.num_cols];
        let mut objective = vec![0.0_f64; layout.num_cols];
        let mut row_lower = vec![f64::NEG_INFINITY; layout.rows.num_rows];
        let mut row_upper = vec![f64::INFINITY; layout.rows.num_rows];
        let mut buffers = LpMatrixBuffers {
            col_entries: &mut col_entries,
            col_upper: &mut col_upper,
            objective: &mut objective,
            row_lower: &mut row_lower,
            row_upper: &mut row_upper,
        };

        fill_generic_constraint_entries(&ctx, 0, &layout, &mut buffers);

        assert_eq!(
            block_hours.len(),
            n_blks,
            "fixture has exactly n_blks blocks"
        );
        for (blk, &hours) in block_hours.iter().enumerate() {
            let row = layout.rows.row_generic_start + blk;
            assert_eq!(
                row_lower[row], 5.0,
                "blk {blk}: two-sided row lower = bound_lower"
            );
            assert_eq!(
                row_upper[row], 20.0,
                "blk {blk}: two-sided row upper = bound_upper"
            );

            let entry = &layout.generic_constraint_rows[blk];
            let plus_col = entry.slack_plus_col.unwrap();
            let minus_col = entry.slack_minus_col.unwrap();
            assert!(
                col_entries[plus_col].contains(&(row, 1.0)),
                "blk {blk}: plus-slack coefficient must be +1.0 (relaxes the lower bound)"
            );
            assert!(
                col_entries[minus_col].contains(&(row, -1.0)),
                "blk {blk}: minus-slack coefficient must be -1.0 (relaxes the upper bound)"
            );
            let expected_obj = penalty * hours;
            assert_eq!(
                objective[plus_col], expected_obj,
                "blk {blk}: plus-slack objective coefficient"
            );
            assert_eq!(
                objective[minus_col], expected_obj,
                "blk {blk}: minus-slack objective coefficient"
            );
        }
    }

    // ── Filling-cascade test helpers (shared by the filling-row tests below) ─────

    use cobre_core::entities::hydro::FillingConfig;

    const RET_START_STAGE_ID: i32 = 2;
    const RET_ENTRY_STAGE_ID: i32 = 4;
    const RET_FILLING_ID: i32 = 3; // start <= 3 < entry  -> Filling
    const RET_OPERATING_ID: i32 = 4; // 4 >= entry         -> Operating

    /// A cascade hydro with caller-chosen `downstream_id`, `entry_stage_id`, and
    /// optional `FillingConfig`, so an upstream→downstream cascade with a filling
    /// downstream can be built. Constant productivity (no FPHA columns).
    fn ret_hydro(id: i32, downstream: Option<i32>, entry: Option<i32>, filling: bool) -> Hydro {
        let mut hydro = Hydro {
            unit_groups: Vec::new(),
            id: EntityId(id),
            name: format!("H{id}"),
            operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            downstream_id: downstream.map(EntityId),
            travel_time_hours: None,
            entry_stage_id: entry,
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
            filling: filling.then_some(FillingConfig {
                start_stage_id: RET_START_STAGE_ID,
                filling_min_rate_m3s: 0.0,
            }),
            penalties: zero_hydro_penalties(),
        };
        hydro.declare_mirror_unit_group(EntityId(1));
        hydro
    }

    /// Sum the CSC values stored at `(col, row)`.
    fn csc_at(csc: &(Vec<i32>, Vec<i32>, Vec<f64>), col: usize, row: usize) -> f64 {
        let start = usize::try_from(csc.0[col]).unwrap();
        let end = usize::try_from(csc.0[col + 1]).unwrap();
        csc.1[start..end]
            .iter()
            .zip(&csc.2[start..end])
            .filter(|(r, _)| usize::try_from(**r).is_ok_and(|r| r == row))
            .map(|(_, &v)| v)
            .sum()
    }

    // ── PreFilling upstream of a Filling downstream (balance-row routing) ────────

    /// Like [`ret_hydro`] (filling) but with a caller-chosen filling
    /// `start_stage_id`, so an upstream filling hydro can begin its `PreFilling`
    /// phase LATER than a downstream filling hydro — the topology where the
    /// downstream is Filling while the upstream is still `PreFilling` at the same
    /// stage.
    fn ret_hydro_start(
        id: i32,
        downstream: Option<i32>,
        entry: Option<i32>,
        start_stage_id: i32,
    ) -> Hydro {
        let mut h = ret_hydro(id, downstream, entry, false);
        h.filling = Some(FillingConfig {
            start_stage_id,
            filling_min_rate_m3s: 0.0,
        });
        h
    }

    struct PfuOffsets {
        zeta: f64,
        z_u: usize,
        water_row_u: usize,
        water_row_d: usize,
        z_inflow_row_u: usize,
        filling_target_row_d: usize,
        n_target_rows: usize,
        storage_in_u: usize,
    }

    /// Count the nonzero entries stored in CSC column `col`.
    fn csc_col_nnz(csc: &(Vec<i32>, Vec<i32>, Vec<f64>), col: usize) -> usize {
        let start = usize::try_from(csc.0[col]).unwrap();
        let end = usize::try_from(csc.0[col + 1]).unwrap();
        csc.2[start..end].iter().filter(|&&v| v != 0.0).count()
    }

    /// Build `U(H1) → D(H2)` where BOTH are filling but the upstream U commissions
    /// LATER than the downstream D, evaluated at stage 2 where D is Filling
    /// (`start_D = 2 ≤ 2 < entry_D = 4`) and U is still `PreFilling`
    /// (`2 < start_U = 3`).
    #[expect(
        clippy::type_complexity,
        reason = "each case returns the raw build tuple its assertions destructure"
    )]
    fn build_prefilling_upstream_of_filling_case() -> ((Vec<i32>, Vec<i32>, Vec<f64>), PfuOffsets) {
        let stage_id = 2;
        let mut fixtures = PumpFixtures::new(
            vec![
                ret_hydro_start(1, Some(2), Some(5), 3), // U: PreFilling at 0,1,2; Filling at 3,4
                ret_hydro_start(2, None, Some(4), 2),    // D: Filling at 2,3; downstream of U
            ],
            Vec::new(),
        );
        let u_idx = fixtures.hydro_pos[&EntityId(1)];
        let d_idx = fixtures.hydro_pos[&EntityId(2)];
        let ctx = fixtures.make_ctx();
        let stage_index = usize::try_from(stage_id).expect("non-negative");
        let stage = two_block_stage(stage_index, [300.0, 444.0]);
        let layout = StageLayout::new(&ctx, &stage, 0);
        let csc = {
            let mut entries = build_stage_matrix_entries(&ctx, &stage, 0, &layout);
            for col in &mut entries {
                col.sort_unstable_by_key(|&(row, _)| row);
            }
            assemble_csc(&entries)
        };
        // D is the only Filling hydro at this stage, so its σ_fill target row is the
        // single `filling_target` row — used below to prove routed inflow does NOT
        // land on it.
        let d_target_local = layout
            .geometry
            .filling_target_hydro_indices
            .iter()
            .position(|&h| h.get() == d_idx)
            .expect("D is Filling, so it carries a σ_fill target row");
        let offsets = PfuOffsets {
            zeta: layout.clock.zeta(),
            z_u: layout.state.z_inflow.start + u_idx,
            water_row_u: layout.geometry.water_balance.start() + u_idx,
            water_row_d: layout.geometry.water_balance.start() + d_idx,
            z_inflow_row_u: layout.z_inflow_row(HydroSys::new(u_idx)),
            filling_target_row_d: layout.geometry.filling_target.start + d_target_local,
            n_target_rows: layout.geometry.filling_target_hydro_indices.len(),
            storage_in_u: layout.state.storage_in.start + u_idx,
        };
        (csc, offsets)
    }

    /// Regression: when a `PreFilling` hydro U is the direct upstream of a Filling
    /// hydro D at the same stage, U's routed natural inflow (`z_U` at `−ζ`) lands on
    /// D's water-balance row and on NO OTHER constraint row. Filling is uncapped in
    /// the volume-target model — there is no impound-cap row — so the routed inflow
    /// has only the balance row to receive it. The forbidden alternative (the v1
    /// cap-row port) routed a SECOND `z_U` push onto a retention row; removing the
    /// balance-row push instead would strand U's water (a correctness regression), so
    /// this test pins both: routed onto the balance row, and onto no other row.
    ///
    /// The `z_U` column carries exactly TWO nonzero entries: its `+1.0` on U's own
    /// z-inflow DEFINITION row (untouched by the short-circuit) and the routed `−ζ`
    /// on D's water-balance row. A cap-row port would add a third entry.
    #[test]
    fn prefilling_upstream_inflow_lands_on_balance_row_only() {
        let (csc, off) = build_prefilling_upstream_of_filling_case();
        assert_eq!(
            off.n_target_rows, 1,
            "exactly one σ_fill target row (only D is Filling)"
        );

        // U's realized inflow z_U is routed onto D's standard water-balance row at −ζ.
        assert_eq!(
            csc_at(&csc, off.z_u, off.water_row_d),
            -off.zeta,
            "z_U is routed onto D's water-balance row at −ζ (short-circuit target)"
        );
        // …and on NO OTHER constraint row beyond its own z-inflow definition row:
        // exactly two nonzero entries (def row +1.0, routed balance-row −ζ). A
        // retention/cap-row port would push a third entry here.
        assert_eq!(
            csc_at(&csc, off.z_u, off.z_inflow_row_u),
            1.0,
            "z_U keeps its +1.0 on U's own z-inflow definition row"
        );
        assert_eq!(
            csc_col_nnz(&csc, off.z_u),
            2,
            "z_U has exactly two nonzero CSC entries (def row + balance-row push, no cap-row port)"
        );
        // Specifically, z_U is absent from D's σ_fill target row (the σ_fill row
        // couples v_h + slack, never the routed inflow) and from U's own frozen
        // water-balance row.
        assert_eq!(
            csc_at(&csc, off.z_u, off.filling_target_row_d),
            0.0,
            "z_U does not land on D's σ_fill target row"
        );
        assert_eq!(
            csc_at(&csc, off.z_u, off.water_row_u),
            0.0,
            "z_U is relocated off U's frozen-identity water-balance row (PreFilling)"
        );
        assert_eq!(
            csc_at(&csc, off.storage_in_u, off.water_row_u),
            -1.0,
            "U's frozen-identity row keeps its incoming-storage −1 entry"
        );
    }

    // ── σ_fill terminal target row + column ──────────────────────────────────

    // The terminal Filling stage of the `ret_hydro` window: entry − 1 = 3.
    const TARGET_TERMINAL_ID: i32 = RET_ENTRY_STAGE_ID - 1;
    // A non-zero resolved dead volume so the σ_fill row RHS is observable.
    const TARGET_MIN_STORAGE_HM3: f64 = 37.5;

    struct TargetOffsets {
        n_target_rows: usize,
        target_row: usize,
        sigma_fill_col: usize,
        v_h_col: usize,
        num_rows: usize,
        num_cols: usize,
    }

    /// Build the `H1 → H2` cascade (H2 the filling hydro, `entry = 4`) at the given
    /// `stage_id`, with H2's resolved per-stage `min_storage_hm3` set to
    /// `TARGET_MIN_STORAGE_HM3`. Returns the assembled CSC triple, the
    /// `(row_lower, row_upper)` vectors, and the `σ_fill` offsets the assertions read.
    #[expect(
        clippy::type_complexity,
        reason = "each case returns the raw build tuple its assertions destructure"
    )]
    fn build_target_case(
        stage_id: i32,
    ) -> (
        (Vec<i32>, Vec<i32>, Vec<f64>),
        Vec<f64>,
        Vec<f64>,
        TargetOffsets,
    ) {
        let mut fixtures = PumpFixtures::new(
            vec![
                ret_hydro(1, Some(2), None, false),
                ret_hydro(2, None, Some(RET_ENTRY_STAGE_ID), true),
            ],
            Vec::new(),
        );
        let h2_idx = fixtures.hydro_pos[&EntityId(2)];
        fixtures
            .base
            .bounds
            .hydro_bounds_mut(h2_idx, 0)
            .min_storage_hm3 = TARGET_MIN_STORAGE_HM3;
        let ctx = fixtures.make_ctx();
        let stage_index = usize::try_from(stage_id).expect("non-negative");
        let stage = two_block_stage(stage_index, [300.0, 444.0]);
        let layout = StageLayout::new(&ctx, &stage, 0);
        let (row_lower, row_upper) = super::super::rows::fill_stage_rows(&ctx, &stage, 0, &layout);
        let csc = {
            let mut entries = build_stage_matrix_entries(&ctx, &stage, 0, &layout);
            for col in &mut entries {
                col.sort_unstable_by_key(|&(row, _)| row);
            }
            assemble_csc(&entries)
        };
        let offsets = TargetOffsets {
            n_target_rows: layout.geometry.filling_target_hydro_indices.len(),
            target_row: layout.geometry.filling_target.start,
            sigma_fill_col: layout.geometry.filling_target_col.start,
            // The outgoing storage column v_h is the dense system index h2_idx.
            v_h_col: h2_idx,
            num_rows: layout.rows.num_rows,
            num_cols: layout.num_cols,
        };
        (csc, row_lower, row_upper, offsets)
    }

    /// At the last Filling stage (id == entry − 1 == 3) the `σ_fill` row exists with
    /// exactly `+1` on the outgoing storage column `v_h`, `+1` on the `σ_fill` slack
    /// column, `≥` sense, and RHS `V_target[3] == min_storage_hm3` (the backward-fold
    /// anchor; rate is 0 here so the flat trajectory pins every floor to the dead
    /// volume). Exactly one such row + column (the single filling hydro H2).
    #[test]
    fn sigma_fill_row_columns_coefficients_and_rhs_at_last_filling_stage() {
        let (csc, row_lower, row_upper, off) = build_target_case(TARGET_TERMINAL_ID);
        assert_eq!(
            off.n_target_rows, 1,
            "exactly one σ_fill row (H2 last Filling)"
        );
        let row = off.target_row;
        assert_eq!(
            csc_at(&csc, off.v_h_col, row),
            1.0,
            "+1 on the outgoing storage column v_h"
        );
        assert_eq!(
            csc_at(&csc, off.sigma_fill_col, row),
            1.0,
            "+1 on the σ_fill slack column"
        );
        assert_eq!(
            row_lower[row], TARGET_MIN_STORAGE_HM3,
            "σ_fill row_lower == V_target[L] == min_storage_hm3 (≥ RHS at the anchor)"
        );
        assert_eq!(
            row_upper[row],
            f64::INFINITY,
            "σ_fill row is a ≥ inequality (upper = +∞)"
        );
    }

    /// No `σ_fill` row or column is emitted OFF the Filling phase — `PreFilling`
    /// (id 1, before start = 2) or `Operating` (id 4, at entry). Per-stage Filling
    /// membership; the v1 terminal-only rule is gone.
    ///
    /// At `PreFilling` the `σ^{v-}` family is also empty, so the empty `σ_fill`
    /// block's cursor coincides with `num_rows`/`num_cols`. At `Operating` the
    /// `σ^{v-}` family legitimately adds exactly one row + column for the single
    /// filling hydro AFTER the empty `σ_fill` block, so the `σ_fill` cursor sits one
    /// short of `num_rows`/`num_cols` — the `σ_fill` block is still empty
    /// (`n_target_rows == 0`), it is simply no longer the last family.
    #[test]
    fn sigma_fill_absent_off_filling_phase() {
        // id 1 is PreFilling (start = 2 > 0, stage_id < start); id 4 is Operating.
        for stage_id in [RET_START_STAGE_ID - 1, RET_OPERATING_ID] {
            let (_csc, _rl, _ru, off) = build_target_case(stage_id);
            assert_eq!(
                off.n_target_rows, 0,
                "no σ_fill row at non-Filling id {stage_id}"
            );
            // σ^{v-} adds one row + column at Operating (the single filling hydro),
            // none at PreFilling.
            let sigma_minus_width = usize::from(stage_id == RET_OPERATING_ID);
            assert_eq!(
                off.target_row + sigma_minus_width,
                off.num_rows,
                "empty σ_fill row block at id {stage_id} (σ^{{v-}} occupies the tail in Operating)"
            );
            assert_eq!(
                off.sigma_fill_col + sigma_minus_width,
                off.num_cols,
                "empty σ_fill column block at id {stage_id} (σ^{{v-}} occupies the tail in Operating)"
            );
        }
    }

    /// At an EARLY Filling stage (id 2 = start, with the last Filling stage at id 3)
    /// the `σ_fill` row RHS is the backward-folded `V_target[2] == min_storage −
    /// ζ_3·rate_3`, strictly BELOW the dead volume — NOT `min_storage` at every
    /// stage. This is the per-stage trajectory: the reservoir is only required to
    /// hold the running minimum by stage 2, with the full dead volume due at stage 3.
    /// Uses the AC schedule (`min_storage = 60`, per-stage ζ = 2.592, rate = 5) so
    /// `V_target[3] = 60`, `V_target[2] = 60 − 2.592·5 = 47.04`.
    #[test]
    fn sigma_fill_row_rhs_is_backward_anchored_v_target_at_early_filling_stage() {
        // AC values. The fixture's `build_filling_v_target` maps ids 2 and 3 to
        // stage_idx 0; set H2's resolved dead volume and per-stage fill rate at
        // stage_idx 0 so the fold reads them.
        const AC_MIN_STORAGE: f64 = 60.0;
        const AC_RATE_M3S: f64 = 5.0;
        // ζ = total_hours · M3S_TO_HM3 = 720 · 0.0036 = 2.592 (the AC ζ).
        const AC_TOTAL_HOURS: f64 = 720.0;

        let mut fixtures = PumpFixtures::new(
            vec![
                ret_hydro(1, Some(2), None, false),
                ret_hydro(2, None, Some(RET_ENTRY_STAGE_ID), true),
            ],
            Vec::new(),
        );
        let h2_idx = fixtures.hydro_pos[&EntityId(2)];
        fixtures
            .base
            .bounds
            .hydro_bounds_mut(h2_idx, 0)
            .min_storage_hm3 = AC_MIN_STORAGE;
        fixtures
            .base
            .bounds
            .hydro_bounds_mut(h2_idx, 0)
            .filling_min_rate_m3s = AC_RATE_M3S;
        // The fixture-default stage_zetas is 744 · M3S_TO_HM3; rebuild the ctx's
        // V_target map with the AC ζ (720 h → ζ = 2.592) so the fold matches the AC.
        let ac_filling_v_target = crate::test_support::build_filling_v_target(
            &fixtures.base.hydros,
            &fixtures.base.bounds,
            &[AC_TOTAL_HOURS * M3S_TO_HM3],
            &(0..=8_i32).map(|id| (id, 0_usize)).collect(),
        );
        let mut ctx = fixtures.make_ctx();
        ctx.filling_v_target = &ac_filling_v_target;

        // V_target[3] (last Filling stage) == min_storage.
        let v_target_last = ctx.filling_v_target[&(h2_idx, RET_FILLING_ID)];
        assert!(
            (v_target_last - AC_MIN_STORAGE).abs() < 1e-9,
            "V_target[3] == min_storage (anchor), got {v_target_last}"
        );

        // V_target[2] (early Filling stage) == 60 − 2.592·5 == 47.04.
        let v_target_early = ctx.filling_v_target[&(h2_idx, RET_START_STAGE_ID)];
        let expected_early = AC_MIN_STORAGE - AC_TOTAL_HOURS * M3S_TO_HM3 * AC_RATE_M3S;
        assert!(
            (v_target_early - 47.04).abs() < 1e-9,
            "V_target[2] == 47.04 (backward fold), got {v_target_early}"
        );
        assert!(
            (v_target_early - expected_early).abs() < 1e-9,
            "V_target[2] == min_storage − ζ·rate, got {v_target_early} vs {expected_early}"
        );
        assert!(
            v_target_early < v_target_last,
            "early floor strictly below the dead-volume anchor (per-stage trajectory)"
        );

        // The row RHS at id 2 reads that V_target[2], not min_storage.
        let stage = two_block_stage(usize::try_from(RET_START_STAGE_ID).unwrap(), [360.0, 360.0]);
        let layout = StageLayout::new(&ctx, &stage, 0);
        let (row_lower, _row_upper) = super::super::rows::fill_stage_rows(&ctx, &stage, 0, &layout);
        let row = layout.geometry.filling_target.start;
        assert_eq!(
            layout.geometry.filling_target_hydro_indices.len(),
            1,
            "one σ_fill row at the early Filling stage (id 2)"
        );
        assert!(
            (row_lower[row] - 47.04).abs() < 1e-9,
            "σ_fill row_lower == V_target[2] == 47.04, got {}",
            row_lower[row]
        );
    }

    /// A non-filling system (no `FillingConfig` on any hydro) emits NO `σ_fill` row
    /// or column, and its `num_rows`/`num_cols` are unchanged at the would-be
    /// terminal stage — the parity-neutrality contract.
    #[test]
    fn non_filling_system_emits_no_sigma_fill() {
        let mut control = PumpFixtures::new(
            vec![
                ret_hydro(1, Some(2), None, false),
                ret_hydro(2, None, None, false),
            ],
            Vec::new(),
        );
        let ctx = control.make_ctx();
        let stage = two_block_stage(usize::try_from(TARGET_TERMINAL_ID).unwrap(), [300.0, 444.0]);
        let layout = StageLayout::new(&ctx, &stage, 0);
        assert_eq!(
            layout.geometry.filling_target_hydro_indices.len(),
            0,
            "control system has no filling hydro ⇒ no σ_fill row/column"
        );
        // The σ_fill row/column cursors degenerate to the structural bounds.
        assert_eq!(layout.geometry.filling_target.start, layout.rows.num_rows);
        assert_eq!(layout.geometry.filling_target_col.start, layout.num_cols);
    }

    /// Cut-validity guard (§4 trap 3): the `σ_fill` soft row couples to the storage
    /// state through the constraint matrix, so LP duality folds its dual into the
    /// SINGLE reduced cost of the incoming-storage column the cut already reads as
    /// `rc / col_scale`. The `σ_fill` builder code therefore must NEVER reference the
    /// dual-extraction entry point — doing so would signal a hand-combination of the
    /// soft-row dual that double-counts the floor and corrupts the cut. This test
    /// reads every `lp/builder/` source file and asserts none mentions that symbol,
    /// so a future "simplification" that wires the soft dual in by hand fails here.
    /// The dual-extraction function itself lives in
    /// `training/backward/duals_extraction.rs`, untouched by this family.
    ///
    /// The forbidden symbol is assembled from chars so this guard's own source text
    /// does not contain the literal (which would make the test flag itself).
    #[test]
    fn builder_never_references_dual_extraction() {
        use std::path::Path;
        // The dual-extraction symbol, assembled from fragments so the needle is
        // absent from this file's own bytes (else the guard would flag itself).
        let needle: String = ["extract", "_duals", "_from", "_view"].concat();
        let builder_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/lp/builder");
        assert!(
            builder_dir.is_dir(),
            "lp/builder source dir must exist at {}",
            builder_dir.display()
        );
        let mut offenders: Vec<String> = Vec::new();
        for entry in std::fs::read_dir(&builder_dir).expect("read lp/builder dir") {
            let path = entry.expect("dir entry").path();
            if path.extension().and_then(|e| e.to_str()) != Some("rs") {
                continue;
            }
            let src = std::fs::read_to_string(&path).expect("read builder source file");
            if src.contains(&needle) {
                offenders.push(path.display().to_string());
            }
        }
        assert!(
            offenders.is_empty(),
            "lp/builder must not reference the dual-extraction symbol (the σ_fill \
             soft dual is folded into the storage column reduced cost by LP duality, \
             never hand-combined); offending files: {offenders:?}"
        );
    }

    /// The `σ_fill` row lands STRICTLY BELOW `num_rows` (inside the structural
    /// region, ahead of the appended cut rows that start at `num_rows`), and the
    /// `σ_fill` column strictly below `num_cols`. A `σ_fill` row at index `>= num_rows`
    /// would alias a cut row and corrupt slot-identity warm-start reconstruction.
    #[test]
    fn sigma_fill_row_below_num_rows_and_col_below_num_cols() {
        let (_csc, _rl, _ru, off) = build_target_case(TARGET_TERMINAL_ID);
        assert!(off.n_target_rows > 0, "this case has a σ_fill row");
        assert!(
            off.target_row + off.n_target_rows <= off.num_rows,
            "σ_fill rows [{}, {}) must lie within [0, num_rows={})",
            off.target_row,
            off.target_row + off.n_target_rows,
            off.num_rows
        );
        assert!(
            off.sigma_fill_col + off.n_target_rows <= off.num_cols,
            "σ_fill cols [{}, {}) must lie within [0, num_cols={})",
            off.sigma_fill_col,
            off.sigma_fill_col + off.n_target_rows,
            off.num_cols
        );
    }

    // ── σ^{v-} operating-floor row + column ──────────────────────────────────

    struct FloorOffsets {
        n_floor_rows: usize,
        floor_row: usize,
        sigma_minus_col: usize,
        v_h_col: usize,
        num_rows: usize,
        num_cols: usize,
    }

    /// Build the `H1 → H2` cascade (H2 the filling hydro, `entry = 4`) at the given
    /// `stage_id`, with H2's resolved per-stage `min_storage_hm3` set to
    /// `TARGET_MIN_STORAGE_HM3`. Returns the assembled CSC triple, the
    /// `(row_lower, row_upper)` vectors, and the `σ^{v-}` offsets the assertions
    /// read. Mirrors `build_target_case` but reads the `filled_min_storage_floor` family.
    #[expect(
        clippy::type_complexity,
        reason = "each case returns the raw build tuple its assertions destructure"
    )]
    fn build_floor_case(
        stage_id: i32,
    ) -> (
        (Vec<i32>, Vec<i32>, Vec<f64>),
        Vec<f64>,
        Vec<f64>,
        FloorOffsets,
    ) {
        let mut fixtures = PumpFixtures::new(
            vec![
                ret_hydro(1, Some(2), None, false),
                ret_hydro(2, None, Some(RET_ENTRY_STAGE_ID), true),
            ],
            Vec::new(),
        );
        let h2_idx = fixtures.hydro_pos[&EntityId(2)];
        fixtures
            .base
            .bounds
            .hydro_bounds_mut(h2_idx, 0)
            .min_storage_hm3 = TARGET_MIN_STORAGE_HM3;
        let ctx = fixtures.make_ctx();
        let stage_index = usize::try_from(stage_id).expect("non-negative");
        let stage = two_block_stage(stage_index, [300.0, 444.0]);
        let layout = StageLayout::new(&ctx, &stage, 0);
        let (row_lower, row_upper) = super::super::rows::fill_stage_rows(&ctx, &stage, 0, &layout);
        let csc = {
            let mut entries = build_stage_matrix_entries(&ctx, &stage, 0, &layout);
            for col in &mut entries {
                col.sort_unstable_by_key(|&(row, _)| row);
            }
            assemble_csc(&entries)
        };
        let offsets = FloorOffsets {
            n_floor_rows: layout.geometry.filled_min_storage_floor_hydro_indices.len(),
            floor_row: layout.geometry.filled_min_storage_floor.start,
            sigma_minus_col: layout.geometry.filled_min_storage_floor_col.start,
            // The outgoing storage column v_h is the dense system index h2_idx.
            v_h_col: h2_idx,
            num_rows: layout.rows.num_rows,
            num_cols: layout.num_cols,
        };
        (csc, row_lower, row_upper, offsets)
    }

    /// At an Operating stage (id == entry == 4) the `σ^{v-}` row exists with exactly
    /// `+1` on the outgoing storage column `v_h`, `+1` on the `σ^{v-}` slack column,
    /// `≥` sense, and RHS `min_storage_hm3`. Exactly one such row + column (the
    /// single filling hydro H2). Same shape as the `σ_fill` row, different stage
    /// scope.
    #[test]
    fn sigma_minus_row_columns_coefficients_and_rhs_in_operating() {
        let (csc, row_lower, row_upper, off) = build_floor_case(RET_OPERATING_ID);
        assert_eq!(
            off.n_floor_rows, 1,
            "exactly one σ^{{v-}} row (H2 operating)"
        );
        let row = off.floor_row;
        assert_eq!(
            csc_at(&csc, off.v_h_col, row),
            1.0,
            "+1 on the outgoing storage column v_h"
        );
        assert_eq!(
            csc_at(&csc, off.sigma_minus_col, row),
            1.0,
            "+1 on the σ^{{v-}} slack column"
        );
        assert_eq!(
            row_lower[row], TARGET_MIN_STORAGE_HM3,
            "σ^{{v-}} row_lower == min_storage_hm3 (≥ RHS)"
        );
        assert_eq!(
            row_upper[row],
            f64::INFINITY,
            "σ^{{v-}} row is a ≥ inequality (upper = +∞)"
        );
    }

    /// No `σ^{v-}` row or column is emitted at a non-operating stage of a filling
    /// hydro — `PreFilling` (id 0) or a Filling stage (id 3). Operating-only.
    #[test]
    fn sigma_minus_absent_off_operating_stage() {
        for stage_id in [RET_PREFILLING_ID, TARGET_TERMINAL_ID] {
            let (_csc, _rl, _ru, off) = build_floor_case(stage_id);
            assert_eq!(
                off.n_floor_rows, 0,
                "no σ^{{v-}} row at id {stage_id} (only Operating carries it)"
            );
            assert_eq!(
                off.floor_row, off.num_rows,
                "empty σ^{{v-}} row block: cursor coincides with num_rows at id {stage_id}"
            );
            assert_eq!(
                off.sigma_minus_col, off.num_cols,
                "empty σ^{{v-}} column block: cursor coincides with num_cols at id {stage_id}"
            );
        }
    }

    /// At the terminal `Filling` stage (id == entry − 1 == 3) a filling hydro has
    /// the `σ_fill` family but NOT `σ^{v-}`; in `Operating` (id == entry == 4) it has
    /// `σ^{v-}` but NOT `σ_fill` — the two families are mutually exclusive by stage
    /// and must not be conflated.
    #[test]
    fn sigma_minus_and_sigma_fill_mutually_exclusive_by_stage() {
        let (_csc_t, _rl_t, _ru_t, target_at_terminal) = build_target_case(TARGET_TERMINAL_ID);
        let (_csc_tf, _rl_tf, _ru_tf, floor_at_terminal) = build_floor_case(TARGET_TERMINAL_ID);
        assert_eq!(
            target_at_terminal.n_target_rows, 1,
            "σ_fill present at terminal Filling stage"
        );
        assert_eq!(
            floor_at_terminal.n_floor_rows, 0,
            "σ^{{v-}} absent at terminal Filling stage"
        );

        let (_csc_o, _rl_o, _ru_o, target_in_op) = build_target_case(RET_OPERATING_ID);
        let (_csc_of, _rl_of, _ru_of, floor_in_op) = build_floor_case(RET_OPERATING_ID);
        assert_eq!(target_in_op.n_target_rows, 0, "σ_fill absent in Operating");
        assert_eq!(floor_in_op.n_floor_rows, 1, "σ^{{v-}} present in Operating");
    }

    /// A non-filling system (no `FillingConfig` on any hydro) emits NO `σ^{v-}` row
    /// or column, and its `num_rows`/`num_cols` are unchanged at the would-be
    /// Operating stage — the parity-neutrality contract.
    #[test]
    fn non_filling_system_emits_no_sigma_minus() {
        let mut control = PumpFixtures::new(
            vec![
                ret_hydro(1, Some(2), None, false),
                ret_hydro(2, None, None, false),
            ],
            Vec::new(),
        );
        let ctx = control.make_ctx();
        let stage = two_block_stage(usize::try_from(RET_OPERATING_ID).unwrap(), [300.0, 444.0]);
        let layout = StageLayout::new(&ctx, &stage, 0);
        assert_eq!(
            layout.geometry.filled_min_storage_floor_hydro_indices.len(),
            0,
            "control system has no filling hydro ⇒ no σ^{{v-}} row/column"
        );
        // The σ^{v-} row/column cursors degenerate to the structural bounds.
        assert_eq!(
            layout.geometry.filled_min_storage_floor.start,
            layout.rows.num_rows
        );
        assert_eq!(
            layout.geometry.filled_min_storage_floor_col.start,
            layout.num_cols
        );
    }

    /// The `σ^{v-}` row lands STRICTLY BELOW `num_rows` (inside the structural
    /// region, ahead of the appended cut rows that start at `num_rows`), and the
    /// `σ^{v-}` column strictly below `num_cols`. A `σ^{v-}` row at index
    /// `>= num_rows` would alias a cut row and corrupt slot-identity warm-start
    /// reconstruction.
    #[test]
    fn sigma_minus_row_below_num_rows_and_col_below_num_cols() {
        let (_csc, _rl, _ru, off) = build_floor_case(RET_OPERATING_ID);
        assert!(off.n_floor_rows > 0, "this case has a σ^{{v-}} row");
        assert!(
            off.floor_row + off.n_floor_rows <= off.num_rows,
            "σ^{{v-}} rows [{}, {}) must lie within [0, num_rows={})",
            off.floor_row,
            off.floor_row + off.n_floor_rows,
            off.num_rows
        );
        assert!(
            off.sigma_minus_col + off.n_floor_rows <= off.num_cols,
            "σ^{{v-}} cols [{}, {}) must lie within [0, num_cols={})",
            off.sigma_minus_col,
            off.sigma_minus_col + off.n_floor_rows,
            off.num_cols
        );
    }

    // ── PreFilling cascade short-circuit ─────────────────────────────────────

    // `RET_START_STAGE_ID = 2`, so a stage with id `< 2` is PreFilling for a
    // filling hydro built by `ret_hydro`. Stage id 0 is the canonical PreFilling
    // probe; the boundary stage id 2 is Filling (governed by the retention tests
    // above), confirming the short-circuit does NOT fire there.
    const RET_PREFILLING_ID: i32 = 0;

    /// Resolved offsets for the H1→H2→H3 routed short-circuit probe.
    struct ScOffsets {
        zeta: f64,
        n_blks: usize,
        h2_idx: usize,
        water_row_h2: usize,
        water_row_h3: usize,
        col_storage_in_h2: usize,
        z_h2: usize,
        h1_turbine: Vec<usize>,
        h1_spillage: Vec<usize>,
        h2_turbine: Vec<usize>,
        h2_spillage: Vec<usize>,
    }

    /// Build the cascade `H1(id 1) → H2(id 2, filling) → H3(id 3)` at `stage_id`,
    /// with H2's resolved per-stage withdrawal set to `withdrawal_h`. Returns the
    /// assembled CSC, the `(row_lower, row_upper)` vectors, and the offsets the
    /// short-circuit assertions read. H2 is the mid-cascade filling hydro, so the
    /// short-circuit routes to a REAL downstream (H3), not a sink.
    #[expect(
        clippy::type_complexity,
        reason = "each case returns the raw build tuple its assertions destructure"
    )]
    fn build_shortcircuit_case(
        stage_id: i32,
        withdrawal_h: f64,
    ) -> (
        (Vec<i32>, Vec<i32>, Vec<f64>),
        Vec<f64>,
        Vec<f64>,
        ScOffsets,
    ) {
        let mut fixtures = PumpFixtures::new(
            vec![
                ret_hydro(1, Some(2), None, false),
                ret_hydro(2, Some(3), Some(RET_ENTRY_STAGE_ID), true),
                ret_hydro(3, None, None, false),
            ],
            Vec::new(),
        );
        let h2_idx = fixtures.hydro_pos[&EntityId(2)];
        fixtures
            .base
            .bounds
            .hydro_bounds_mut(h2_idx, 0)
            .water_withdrawal_m3s = withdrawal_h;
        let h1_idx = fixtures.hydro_pos[&EntityId(1)];
        let h3_idx = fixtures.hydro_pos[&EntityId(3)];
        let ctx = fixtures.make_ctx();
        let stage_index = usize::try_from(stage_id).expect("non-negative");
        let stage = two_block_stage(stage_index, [300.0, 444.0]);
        let layout = StageLayout::new(&ctx, &stage, 0);
        let (row_lower, row_upper) = super::super::rows::fill_stage_rows(&ctx, &stage, 0, &layout);
        let csc = {
            let mut entries = build_stage_matrix_entries(&ctx, &stage, 0, &layout);
            for col in &mut entries {
                col.sort_unstable_by_key(|&(row, _)| row);
            }
            assemble_csc(&entries)
        };
        let offsets = ScOffsets {
            zeta: layout.clock.zeta(),
            n_blks: layout.clock.n_blks(),
            h2_idx,
            water_row_h2: layout.geometry.water_balance.start() + h2_idx,
            water_row_h3: layout.geometry.water_balance.start() + h3_idx,
            col_storage_in_h2: layout.state.storage_in.start + h2_idx,
            z_h2: layout.state.z_inflow.start + h2_idx,
            h1_turbine: (0..layout.clock.n_blks())
                .map(|blk| {
                    layout
                        .geometry
                        .turbine_col(HydroCell::new(h1_idx), BlockIdx::new(blk))
                })
                .collect(),
            h1_spillage: (0..layout.clock.n_blks())
                .map(|blk| {
                    layout
                        .geometry
                        .spillage_col(HydroSys::new(h1_idx), BlockIdx::new(blk))
                })
                .collect(),
            h2_turbine: (0..layout.clock.n_blks())
                .map(|blk| {
                    layout
                        .geometry
                        .turbine_col(HydroCell::new(h2_idx), BlockIdx::new(blk))
                })
                .collect(),
            h2_spillage: (0..layout.clock.n_blks())
                .map(|blk| {
                    layout
                        .geometry
                        .spillage_col(HydroSys::new(h2_idx), BlockIdx::new(blk))
                })
                .collect(),
        };
        (csc, row_lower, row_upper, offsets)
    }

    /// At a `PreFilling` stage the absent hydro `H2`'s four water interactions land
    /// on the DOWNSTREAM `H3`'s balance row in real `ζ`/`τ_k` coefficients:
    /// `H2`'s incremental inflow (`−ζ` on `z_{H2}`), `H2`'s upstream `H1` releases
    /// (`−τ` on `H1` turbine/spillage), and `H2`'s withdrawal demand (`−ζ·withdrawal`
    /// on `H3`'s RHS). `H2`'s evaporation is NOT transferred (no evap column exists
    /// — `PreFilling` is excluded upstream). The same columns carry NOTHING extra on
    /// `H2`'s own (frozen) row.
    #[test]
    fn prefilling_routes_inflow_upstream_releases_and_withdrawal_to_downstream() {
        let withdrawal_h = 17.5_f64;
        let (csc, _rl, row_upper, off) = build_shortcircuit_case(RET_PREFILLING_ID, withdrawal_h);
        let row_d = off.water_row_h3;
        let row_h = off.water_row_h2;

        // (1) Incremental inflow: z_{H2} carries −ζ on H3's row (relocated), and
        // NOTHING on H2's own frozen row.
        assert_eq!(
            csc_at(&csc, off.z_h2, row_d),
            -off.zeta,
            "z_{{H2}} carries −ζ on H3's water row (incremental inflow re-routed)"
        );
        assert_eq!(
            csc_at(&csc, off.z_h2, row_h),
            0.0,
            "z_{{H2}} carries nothing on H2's own frozen-identity row"
        );

        // (2) Upstream H1 releases re-routed upstream(H2)→H3 with −τ_h per block.
        // H1 is NOT a standard upstream of H3 (the standard cascade edge is
        // H1→H2→H3), so a nonzero −τ on H1's columns at H3's row can ONLY come from
        // the short-circuit re-route. (H2's own release columns DO appear on H3's
        // row via H3's standard upstream loop — H2 is structurally upstream of H3 —
        // but H2's turbine/spillage are pinned `[0,0]` in PreFilling, so that
        // standard coefficient multiplies a zero column and is harmless. The
        // re-route is what makes H1's water reach H3.)
        for blk in 0..off.n_blks {
            let tau_h = [300.0_f64, 444.0][blk] * M3S_TO_HM3;
            assert_eq!(
                csc_at(&csc, off.h1_turbine[blk], row_d),
                -tau_h,
                "blk {blk}: H1 turbine carries −τ_h on H3's row (cascade edge \
                 H1→H2→H3 re-routed to H1→H3 — H1 is NOT a standard upstream of H3)"
            );
            assert_eq!(
                csc_at(&csc, off.h1_spillage[blk], row_d),
                -tau_h,
                "blk {blk}: H1 spillage carries −τ_h on H3's row"
            );
        }

        // (3) Withdrawal demand transferred to H3's RHS: H3's row_upper drops by
        // ζ·withdrawal_h versus a no-withdrawal build (its own withdrawal is 0).
        let (_csc0, _rl0, row_upper0, off0) = build_shortcircuit_case(RET_PREFILLING_ID, 0.0);
        let delta = off.zeta * withdrawal_h;
        assert_eq!(
            row_upper0[off0.water_row_h3] - row_upper[row_d],
            delta,
            "H3's RHS drops by ζ·withdrawal_{{H2}} (the transferred withdrawal demand)"
        );
    }

    /// `H2`'s own (frozen) row is exactly `v_{H2} − v_{H2,in} = 0`: `+1.0` on its
    /// outgoing-storage column, `−1.0` on its incoming-storage column, and NOTHING
    /// else (no inflow/upstream/AR-lag/withdrawal/evaporation), with RHS 0. The
    /// storage column keeps its dense system index (the absent reservoir's column
    /// is never omitted or relocated — §4 trap 2).
    #[test]
    fn prefilling_h_row_is_frozen_identity_with_dense_storage_column() {
        let (csc, row_lower, row_upper, off) = build_shortcircuit_case(RET_PREFILLING_ID, 9.0);
        let row_h = off.water_row_h2;

        // Outgoing storage column index == H2's system hydro index (dense, unchanged
        // from an operating build — the storage column is never relocated).
        assert_eq!(
            csc_at(&csc, off.h2_idx, row_h),
            1.0,
            "outgoing storage v_{{H2}} carries +1.0 on the frozen-identity row, \
             at the dense column index == H2's system hydro index"
        );
        assert_eq!(
            csc_at(&csc, off.col_storage_in_h2, row_h),
            -1.0,
            "incoming storage v_{{H2,in}} carries −1.0 on the frozen-identity row"
        );

        // Nothing else on H2's row: own releases, z-coupling, withdrawal slacks.
        for blk in 0..off.n_blks {
            assert_eq!(
                csc_at(&csc, off.h2_turbine[blk], row_h),
                0.0,
                "blk {blk}: own turbine absent from the frozen row"
            );
            assert_eq!(
                csc_at(&csc, off.h2_spillage[blk], row_h),
                0.0,
                "blk {blk}: own spillage absent from the frozen row"
            );
            assert_eq!(
                csc_at(&csc, off.h1_turbine[blk], row_h),
                0.0,
                "blk {blk}: upstream H1 turbine absent from H2's frozen row \
                 (re-routed to H3, not left on H2 — §4 trap 4)"
            );
        }
        assert_eq!(
            csc_at(&csc, off.z_h2, row_h),
            0.0,
            "z_{{H2}} absent from the frozen row"
        );

        // RHS is exactly 0 (NOT ζ·(base − withdrawal)).
        assert_eq!(row_lower[row_h], 0.0, "frozen-identity row_lower == 0");
        assert_eq!(row_upper[row_h], 0.0, "frozen-identity row_upper == 0");
    }

    /// Sink case: a `PreFilling` hydro with `downstream_id == None` drops its water
    /// from the system (exactly as a terminal hydro's outflow today). No downstream
    /// row receives the inflow/upstream/withdrawal, and `H2`'s own row is still the
    /// frozen identity. Builds `H1 → H2(filling, terminal)`.
    #[test]
    fn prefilling_sink_drops_water_and_keeps_frozen_identity() {
        let mut fixtures = PumpFixtures::new(
            vec![
                ret_hydro(1, Some(2), None, false),
                ret_hydro(2, None, Some(RET_ENTRY_STAGE_ID), true),
            ],
            Vec::new(),
        );
        let h2_idx = fixtures.hydro_pos[&EntityId(2)];
        fixtures
            .base
            .bounds
            .hydro_bounds_mut(h2_idx, 0)
            .water_withdrawal_m3s = 13.0;
        let h1_idx = fixtures.hydro_pos[&EntityId(1)];
        let ctx = fixtures.make_ctx();
        let stage = two_block_stage(usize::try_from(RET_PREFILLING_ID).unwrap(), [300.0, 444.0]);
        let layout = StageLayout::new(&ctx, &stage, 0);
        let (row_lower, row_upper) = super::super::rows::fill_stage_rows(&ctx, &stage, 0, &layout);
        let csc = {
            let mut entries = build_stage_matrix_entries(&ctx, &stage, 0, &layout);
            for col in &mut entries {
                col.sort_unstable_by_key(|&(row, _)| row);
            }
            assemble_csc(&entries)
        };
        let row_h = layout.geometry.water_balance.start() + h2_idx;
        let z_h2 = layout.state.z_inflow.start + h2_idx;

        // Frozen identity intact on H2's own row.
        assert_eq!(csc_at(&csc, h2_idx, row_h), 1.0, "v_{{H2}} +1.0");
        assert_eq!(
            csc_at(&csc, layout.state.storage_in.start + h2_idx, row_h),
            -1.0,
            "v_{{H2,in}} −1.0"
        );
        assert_eq!(
            row_lower[row_h], 0.0,
            "frozen RHS 0 (no withdrawal folded in)"
        );
        assert_eq!(row_upper[row_h], 0.0, "frozen RHS 0");

        // H2's water exits the system: z_{H2} appears on NO water-balance row, and
        // H1's releases appear only on H1's own row (no downstream to feed).
        for h in 0..layout.state.hydro_count {
            let r = layout.geometry.water_balance.start() + h;
            assert_eq!(
                csc_at(&csc, z_h2, r),
                0.0,
                "z_{{H2}} must not land on any water row in the sink case (row {r})"
            );
        }
        let row_h1 = layout.geometry.water_balance.start() + h1_idx;
        for blk in 0..layout.clock.n_blks() {
            let tau_h = [300.0_f64, 444.0][blk] * M3S_TO_HM3;
            // H1's own +τ on its own row is unchanged; it lands on NO other water row.
            assert_eq!(
                csc_at(
                    &csc,
                    layout
                        .geometry
                        .turbine_col(HydroCell::new(h1_idx), BlockIdx::new(blk)),
                    row_h1
                ),
                tau_h,
                "blk {blk}: H1's own turbine still carries +τ on its own row"
            );
            assert_eq!(
                csc_at(
                    &csc,
                    layout
                        .geometry
                        .turbine_col(HydroCell::new(h1_idx), BlockIdx::new(blk)),
                    row_h
                ),
                0.0,
                "blk {blk}: H1's turbine does not feed the absent sink H2"
            );
        }
    }

    /// Duals test (real solver): at a `PreFilling` stage the incoming-storage column's
    /// reduced cost is 0 — the §4 trap-4 contract `∂Q/∂v̂_h = 0` (a valid flat cut).
    /// With `v_{H2,in}` relocated out of `H2`'s row (the row is the frozen identity
    /// and `v_{H2}` is a dead variable), perturbing the pinned `v̂_{H2}` changes
    /// neither the reduced cost nor the objective. A botched relocation that left
    /// `H2`'s row coupled to upstream would give a nonzero reduced cost
    /// (`β_{H2}` stale-nonzero).
    #[test]
    fn prefilling_incoming_storage_reduced_cost_is_zero() {
        use cobre_solver::{ActiveSolver, SolverInterface};

        let mut fixtures = PumpFixtures::new(
            vec![
                ret_hydro(1, Some(2), None, false),
                ret_hydro(2, Some(3), Some(RET_ENTRY_STAGE_ID), true),
                ret_hydro(3, None, None, false),
            ],
            Vec::new(),
        )
        .with_resolved_penalties();
        let h2_idx = fixtures.hydro_pos[&EntityId(2)];
        let ctx = fixtures.make_ctx();
        let stage = two_block_stage(usize::try_from(RET_PREFILLING_ID).unwrap(), [300.0, 444.0]);
        let out = super::super::template::build_single_stage_template(&ctx, &stage, 0);
        let template = out.template;

        let col_v_in = ctx
            .state
            .state_to_lp_incoming_column(StateDim::new(h2_idx))
            .get();

        // Solve twice with different pinned v̂_{H2}; assert the incoming-storage
        // reduced cost is 0 at both and the objective is unchanged. col_scale is
        // empty for this freshly built (unscaled) template, so the reduced cost is
        // already in original units (no /col_scale needed).
        let solve_at = |v_hat: f64| -> (f64, f64) {
            let mut solver = ActiveSolver::new().expect("ActiveSolver::new()");
            solver.load_model(&template);
            solver.set_col_bounds(&[col_v_in], &[v_hat], &[v_hat]);
            let view = solver.solve(None).expect("PreFilling LP must be feasible");
            (view.reduced_costs[col_v_in], view.objective)
        };

        let (rc_lo, obj_lo) = solve_at(10.0);
        let (rc_hi, obj_hi) = solve_at(70.0);

        assert!(
            rc_lo.abs() < 1e-9,
            "incoming-storage reduced cost must be 0 at v̂=10 (flat cut), got {rc_lo}"
        );
        assert!(
            rc_hi.abs() < 1e-9,
            "incoming-storage reduced cost must be 0 at v̂=70 (flat cut), got {rc_hi}"
        );
        assert!(
            (obj_lo - obj_hi).abs() < 1e-9,
            "objective must be invariant to v̂_{{H2}} (∂Q/∂v̂_h = 0): {obj_lo} vs {obj_hi}"
        );
    }

    /// Non-filling parity: building the SAME 3-hydro cascade with H2 NOT filling
    /// produces a bit-identical water-balance row for H2 (no short-circuit, no
    /// relocation) and identical `num_rows`. The `PreFilling` logic is a no-op for a
    /// hydro with no `FillingConfig`.
    #[test]
    fn non_filling_hydro_water_row_and_num_rows_bit_identical() {
        // Filling build at the PreFilling stage: H2 IS short-circuited.
        let (csc_f, rl_f, ru_f, off_f) = build_shortcircuit_case(RET_PREFILLING_ID, 0.0);

        // Control: same topology and stage, but H2 carries no filling ⇒ Operating
        // everywhere ⇒ standard balance row, no short-circuit.
        let mut control = PumpFixtures::new(
            vec![
                ret_hydro(1, Some(2), None, false),
                ret_hydro(2, Some(3), None, false),
                ret_hydro(3, None, None, false),
            ],
            Vec::new(),
        );
        let h2_idx_c = control.hydro_pos[&EntityId(2)];
        let ctx = control.make_ctx();
        let stage = two_block_stage(usize::try_from(RET_PREFILLING_ID).unwrap(), [300.0, 444.0]);
        let layout = StageLayout::new(&ctx, &stage, 0);
        let (rl_c, ru_c) = super::super::rows::fill_stage_rows(&ctx, &stage, 0, &layout);
        let csc_c = {
            let mut entries = build_stage_matrix_entries(&ctx, &stage, 0, &layout);
            for col in &mut entries {
                col.sort_unstable_by_key(|&(row, _)| row);
            }
            assemble_csc(&entries)
        };
        let row_h2_c = layout.geometry.water_balance.start() + h2_idx_c;

        // num_rows identical (no extra structural rows from the short-circuit; it
        // only moves coefficients, never adds rows).
        assert_eq!(
            layout.rows.num_rows,
            ru_f.len(),
            "num_rows identical whether H2 is PreFilling or non-filling"
        );

        // H2's OWN water row differs between the builds (frozen vs standard) — that
        // is the short-circuit. The control's H2 row carries its own +τ releases;
        // assert at least the own-turbine coefficient is present in the control and
        // ABSENT in the filling build (the observable short-circuit).
        let row_h2_f = off_f.water_row_h2;
        for blk in 0..off_f.n_blks {
            let tau_h = [300.0_f64, 444.0][blk] * M3S_TO_HM3;
            assert_eq!(
                csc_at(
                    &csc_c,
                    layout
                        .geometry
                        .turbine_col(HydroCell::new(h2_idx_c), BlockIdx::new(blk)),
                    row_h2_c
                ),
                tau_h,
                "blk {blk}: control (non-filling) H2 carries +τ on its own row"
            );
            assert_eq!(
                csc_at(&csc_f, off_f.h2_turbine[blk], row_h2_f),
                0.0,
                "blk {blk}: filling (PreFilling) H2 row is frozen — no own +τ"
            );
        }

        // The non-filling control's balance row is a non-zero equality RHS
        // (ζ·base, here base=0 with no PAR ⇒ 0) and an equality (lower == upper).
        assert_eq!(
            rl_c[row_h2_c], ru_c[row_h2_c],
            "control H2 row is an equality"
        );
        // Sanity: the filling build's H2 row is the frozen identity (0).
        assert_eq!(rl_f[row_h2_f], 0.0, "filling H2 frozen RHS == 0");
        assert_eq!(ru_f[row_h2_f], 0.0, "filling H2 frozen RHS == 0");
    }

    // ── Commissioning-dormant NON-filling hydro (reuses the PreFilling path) ─────

    /// A NON-filling cascade hydro with a caller-chosen `entry_stage_id` and no
    /// `FillingConfig`: commissioning-dormant (`PreFilling`) before `entry`, then
    /// `Operating` from `entry` with no intervening `Filling`.
    fn ret_hydro_nonfilling(id: i32, downstream: Option<i32>, entry: Option<i32>) -> Hydro {
        ret_hydro(id, downstream, entry, false)
    }

    /// Build `H1 → H2(non-filling, entry) → H3` at `stage_id`, with H2's resolved
    /// withdrawal set to `withdrawal_h`. Mirrors [`build_shortcircuit_case`] but H2
    /// carries a commissioning window instead of a `FillingConfig`.
    #[expect(
        clippy::type_complexity,
        reason = "each case returns the raw build tuple its assertions destructure"
    )]
    fn build_nonfilling_shortcircuit_case(
        stage_id: i32,
        entry: i32,
        withdrawal_h: f64,
    ) -> (
        (Vec<i32>, Vec<i32>, Vec<f64>),
        Vec<f64>,
        Vec<f64>,
        ScOffsets,
    ) {
        let mut fixtures = PumpFixtures::new(
            vec![
                ret_hydro_nonfilling(1, Some(2), None),
                ret_hydro_nonfilling(2, Some(3), Some(entry)),
                ret_hydro_nonfilling(3, None, None),
            ],
            Vec::new(),
        );
        let h2_idx = fixtures.hydro_pos[&EntityId(2)];
        fixtures
            .base
            .bounds
            .hydro_bounds_mut(h2_idx, 0)
            .water_withdrawal_m3s = withdrawal_h;
        let h1_idx = fixtures.hydro_pos[&EntityId(1)];
        let h3_idx = fixtures.hydro_pos[&EntityId(3)];
        let ctx = fixtures.make_ctx();
        let stage_index = usize::try_from(stage_id).expect("non-negative");
        let stage = two_block_stage(stage_index, [300.0, 444.0]);
        let layout = StageLayout::new(&ctx, &stage, 0);
        let (row_lower, row_upper) = super::super::rows::fill_stage_rows(&ctx, &stage, 0, &layout);
        let csc = {
            let mut entries = build_stage_matrix_entries(&ctx, &stage, 0, &layout);
            for col in &mut entries {
                col.sort_unstable_by_key(|&(row, _)| row);
            }
            assemble_csc(&entries)
        };
        let offsets = ScOffsets {
            zeta: layout.clock.zeta(),
            n_blks: layout.clock.n_blks(),
            h2_idx,
            water_row_h2: layout.geometry.water_balance.start() + h2_idx,
            water_row_h3: layout.geometry.water_balance.start() + h3_idx,
            col_storage_in_h2: layout.state.storage_in.start + h2_idx,
            z_h2: layout.state.z_inflow.start + h2_idx,
            h1_turbine: (0..layout.clock.n_blks())
                .map(|blk| {
                    layout
                        .geometry
                        .turbine_col(HydroCell::new(h1_idx), BlockIdx::new(blk))
                })
                .collect(),
            h1_spillage: (0..layout.clock.n_blks())
                .map(|blk| {
                    layout
                        .geometry
                        .spillage_col(HydroSys::new(h1_idx), BlockIdx::new(blk))
                })
                .collect(),
            h2_turbine: (0..layout.clock.n_blks())
                .map(|blk| {
                    layout
                        .geometry
                        .turbine_col(HydroCell::new(h2_idx), BlockIdx::new(blk))
                })
                .collect(),
            h2_spillage: (0..layout.clock.n_blks())
                .map(|blk| {
                    layout
                        .geometry
                        .spillage_col(HydroSys::new(h2_idx), BlockIdx::new(blk))
                })
                .collect(),
        };
        (csc, row_lower, row_upper, offsets)
    }

    /// A commissioning-dormant non-filling hydro H2 (entry 4, evaluated at stage 0)
    /// is `PreFilling`: its own water row is the frozen identity `v_{H2} −
    /// v_{H2,in} = 0` (RHS 0) and its incremental inflow `z_{H2}` lands on the
    /// downstream H3's balance row at `−ζ` and nowhere else — identical to a filling
    /// `PreFilling` hydro, proving the reformulation is reused, not reinvented.
    #[test]
    fn dormant_nonfilling_frozen_identity_and_inflow_routed_downstream() {
        let (csc, row_lower, row_upper, off) =
            build_nonfilling_shortcircuit_case(0, RET_ENTRY_STAGE_ID, 9.0);
        let row_h = off.water_row_h2;
        let row_d = off.water_row_h3;

        // Frozen identity on H2's own row.
        assert_eq!(
            csc_at(&csc, off.h2_idx, row_h),
            1.0,
            "v_{{H2}} +1.0 on the frozen-identity row (dense storage column)"
        );
        assert_eq!(
            csc_at(&csc, off.col_storage_in_h2, row_h),
            -1.0,
            "v_{{H2,in}} −1.0 on the frozen-identity row"
        );
        assert_eq!(row_lower[row_h], 0.0, "frozen RHS lower == 0");
        assert_eq!(row_upper[row_h], 0.0, "frozen RHS upper == 0");

        // z_{H2} routed onto H3's balance row at −ζ, and absent from H2's frozen row.
        assert_eq!(
            csc_at(&csc, off.z_h2, row_d),
            -off.zeta,
            "z_{{H2}} routed onto H3's water row at −ζ (river flows past the un-built dam)"
        );
        assert_eq!(
            csc_at(&csc, off.z_h2, row_h),
            0.0,
            "z_{{H2}} absent from H2's frozen-identity row (not trapped)"
        );
    }

    /// The cascade tail (sink) case: a dormant non-filling hydro with no downstream
    /// drops its water from the system and keeps its frozen identity — feasible, no
    /// trapped inflow.
    #[test]
    fn dormant_nonfilling_sink_drops_water_and_keeps_frozen_identity() {
        let mut fixtures = PumpFixtures::new(
            vec![
                ret_hydro_nonfilling(1, Some(2), None),
                ret_hydro_nonfilling(2, None, Some(RET_ENTRY_STAGE_ID)),
            ],
            Vec::new(),
        );
        let h2_idx = fixtures.hydro_pos[&EntityId(2)];
        fixtures
            .base
            .bounds
            .hydro_bounds_mut(h2_idx, 0)
            .water_withdrawal_m3s = 13.0;
        let ctx = fixtures.make_ctx();
        let stage = two_block_stage(0, [300.0, 444.0]);
        let layout = StageLayout::new(&ctx, &stage, 0);
        let (row_lower, row_upper) = super::super::rows::fill_stage_rows(&ctx, &stage, 0, &layout);
        let csc = {
            let mut entries = build_stage_matrix_entries(&ctx, &stage, 0, &layout);
            for col in &mut entries {
                col.sort_unstable_by_key(|&(row, _)| row);
            }
            assemble_csc(&entries)
        };
        let row_h = layout.geometry.water_balance.start() + h2_idx;
        let z_h2 = layout.state.z_inflow.start + h2_idx;

        assert_eq!(csc_at(&csc, h2_idx, row_h), 1.0, "v_{{H2}} +1.0");
        assert_eq!(
            csc_at(&csc, layout.state.storage_in.start + h2_idx, row_h),
            -1.0,
            "v_{{H2,in}} −1.0"
        );
        assert_eq!(row_lower[row_h], 0.0, "frozen RHS 0");
        assert_eq!(row_upper[row_h], 0.0, "frozen RHS 0");
        for h in 0..layout.state.hydro_count {
            let r = layout.geometry.water_balance.start() + h;
            assert_eq!(
                csc_at(&csc, z_h2, r),
                0.0,
                "z_{{H2}} on no water row in the sink case (row {r})"
            );
        }
    }

    /// From `entry` onward the same non-filling hydro is `Operating`: its water row
    /// regains the standard form (own +τ releases, no short-circuit), proving the
    /// window opens at `entry` with NO intervening `Filling` phase.
    #[test]
    fn dormant_nonfilling_operating_from_entry_has_standard_row() {
        // entry == 0 ⇒ H2 is Operating at stage 0 (commissioned immediately).
        let (csc, row_lower, row_upper, off) = build_nonfilling_shortcircuit_case(0, 0, 0.0);
        let row_h = off.water_row_h2;
        let row_d = off.water_row_h3;

        for blk in 0..off.n_blks {
            let tau_h = [300.0_f64, 444.0][blk] * M3S_TO_HM3;
            assert_eq!(
                csc_at(&csc, off.h2_turbine[blk], row_h),
                tau_h,
                "blk {blk}: Operating H2 carries +τ on its own row (standard fill)"
            );
        }
        // No short-circuit: z_{H2} is NOT routed onto H3's row.
        assert_eq!(
            csc_at(&csc, off.z_h2, row_d),
            0.0,
            "z_{{H2}} not routed downstream when Operating"
        );
        // Standard equality row, not the frozen identity (lower == upper, base 0).
        assert_eq!(row_lower[row_h], row_upper[row_h], "H2 row is an equality");
    }

    /// Resolved offsets for a CHAINED short-circuit probe `H1 → H2 → H3` where H1
    /// AND H2 are BOTH `PreFilling` at the probe stage and H3 is operating.
    struct ChainOffsets {
        zeta: f64,
        h1_idx: usize,
        h2_idx: usize,
        water_row_h1: usize,
        water_row_h2: usize,
        water_row_h3: usize,
        col_storage_in_h1: usize,
        col_storage_in_h2: usize,
        z_h1: usize,
        z_h2: usize,
    }

    /// Build the cascade `H1(id 1, filling) → H2(id 2, filling) → H3(id 3,
    /// operating)` at `stage_id`, with H1's and H2's resolved per-stage withdrawals
    /// set to `withdrawal_h1` / `withdrawal_h2`. At a `PreFilling` `stage_id` both H1
    /// and H2 are `PreFilling` (frozen rows), so H1 must cascade THROUGH H2 to H3.
    /// Returns the CSC, `(row_lower, row_upper)`, and the chained offsets.
    #[expect(
        clippy::type_complexity,
        reason = "each case returns the raw build tuple its assertions destructure"
    )]
    fn build_chained_shortcircuit_case(
        stage_id: i32,
        withdrawal_h1: f64,
        withdrawal_h2: f64,
    ) -> (
        (Vec<i32>, Vec<i32>, Vec<f64>),
        Vec<f64>,
        Vec<f64>,
        ChainOffsets,
    ) {
        let mut fixtures = PumpFixtures::new(
            vec![
                ret_hydro(1, Some(2), Some(RET_ENTRY_STAGE_ID), true),
                ret_hydro(2, Some(3), Some(RET_ENTRY_STAGE_ID), true),
                ret_hydro(3, None, None, false),
            ],
            Vec::new(),
        );
        let h1_idx = fixtures.hydro_pos[&EntityId(1)];
        let h2_idx = fixtures.hydro_pos[&EntityId(2)];
        fixtures
            .base
            .bounds
            .hydro_bounds_mut(h1_idx, 0)
            .water_withdrawal_m3s = withdrawal_h1;
        fixtures
            .base
            .bounds
            .hydro_bounds_mut(h2_idx, 0)
            .water_withdrawal_m3s = withdrawal_h2;
        let h3_idx = fixtures.hydro_pos[&EntityId(3)];
        let ctx = fixtures.make_ctx();
        let stage_index = usize::try_from(stage_id).expect("non-negative");
        let stage = two_block_stage(stage_index, [300.0, 444.0]);
        let layout = StageLayout::new(&ctx, &stage, 0);
        let (row_lower, row_upper) = super::super::rows::fill_stage_rows(&ctx, &stage, 0, &layout);
        let csc = {
            let mut entries = build_stage_matrix_entries(&ctx, &stage, 0, &layout);
            for col in &mut entries {
                col.sort_unstable_by_key(|&(row, _)| row);
            }
            assemble_csc(&entries)
        };
        let offsets = ChainOffsets {
            zeta: layout.clock.zeta(),
            h1_idx,
            h2_idx,
            water_row_h1: layout.geometry.water_balance.start() + h1_idx,
            water_row_h2: layout.geometry.water_balance.start() + h2_idx,
            water_row_h3: layout.geometry.water_balance.start() + h3_idx,
            col_storage_in_h1: layout.state.storage_in.start + h1_idx,
            col_storage_in_h2: layout.state.storage_in.start + h2_idx,
            z_h1: layout.state.z_inflow.start + h1_idx,
            z_h2: layout.state.z_inflow.start + h2_idx,
        };
        (csc, row_lower, row_upper, offsets)
    }

    /// CHAINED short-circuit: cascade `H1 → H2 → H3` with H1 AND H2 BOTH `PreFilling`
    /// at the same stage and H3 operating. Both H1's water (z_{H1}, withdrawal_{H1})
    /// and H2's water (z_{H2}, withdrawal_{H2}) land on H3's row (the first
    /// non-`PreFilling` downstream) — H1 cascades THROUGH the absent H2, never onto
    /// H2's frozen row.
    #[test]
    fn chained_prefilling_routes_both_links_to_first_operating_downstream() {
        let (w_h1, w_h2) = (11.0_f64, 23.0_f64);
        let (csc, _rl, row_upper, off) =
            build_chained_shortcircuit_case(RET_PREFILLING_ID, w_h1, w_h2);

        // (1) Incremental inflow of BOTH links lands on H3's row (−ζ each), and on
        // NO frozen row. H1 routing THROUGH H2 is the chained behaviour: routing to
        // the immediate downstream H2 would corrupt H2's frozen identity.
        assert_eq!(
            csc_at(&csc, off.z_h1, off.water_row_h3),
            -off.zeta,
            "z_{{H1}} carries −ζ on H3's row (H1 cascades THROUGH the PreFilling H2)"
        );
        assert_eq!(
            csc_at(&csc, off.z_h2, off.water_row_h3),
            -off.zeta,
            "z_{{H2}} carries −ζ on H3's row (first non-PreFilling downstream)"
        );
        // Neither z column touches H2's frozen-identity row.
        assert_eq!(
            csc_at(&csc, off.z_h1, off.water_row_h2),
            0.0,
            "z_{{H1}} must NOT land on H2's frozen row (the corruption this guards against)"
        );
        assert_eq!(
            csc_at(&csc, off.z_h2, off.water_row_h2),
            0.0,
            "z_{{H2}} must NOT land on H2's own frozen row"
        );
        // Nor H1's frozen row.
        assert_eq!(
            csc_at(&csc, off.z_h1, off.water_row_h1),
            0.0,
            "z_{{H1}} must NOT land on H1's own frozen row"
        );

        // (2) Withdrawal demand of BOTH links lands on H3's RHS: H3's row_upper
        // drops by ζ·withdrawal_{H1} + ζ·withdrawal_{H2} versus a zero-withdrawal
        // build. Both transfers resolve to H3 (the same short-circuit target),
        // NEVER onto H2's frozen RHS. The expected delta mirrors the code's two
        // independent per-link `-=` subtractions (rows.rs), NOT a single
        // ζ·(w_h1 + w_h2) product, so the FP rounding matches bit-for-bit.
        let (_csc0, _rl0, row_upper0, off0) =
            build_chained_shortcircuit_case(RET_PREFILLING_ID, 0.0, 0.0);
        let delta = off.zeta * w_h1 + off.zeta * w_h2;
        assert_eq!(
            row_upper0[off0.water_row_h3] - row_upper[off.water_row_h3],
            delta,
            "H3's RHS drops by ζ·withdrawal_{{H1}} + ζ·withdrawal_{{H2}}"
        );
    }

    /// CHAINED short-circuit, clean frozen identities: with H1 AND H2 BOTH
    /// `PreFilling`, EACH of H1's and H2's own water rows is exactly the frozen
    /// identity (`+1 v_h`, `−1 v_h_in`, RHS 0) and exactly 2 matrix entries — NOT
    /// corrupted by the other link's routed inflow/upstream/withdrawal terms.
    #[test]
    fn chained_prefilling_keeps_both_frozen_rows_clean() {
        let (csc, row_lower, row_upper, off) =
            build_chained_shortcircuit_case(RET_PREFILLING_ID, 7.0, 13.0);

        // Helper: count the matrix entries on a given water row and assert it is the
        // clean frozen identity (exactly +1 on v_h, −1 on v_h_in, nothing else).
        let assert_clean_frozen = |row: usize, col_v: usize, col_v_in: usize, label: &str| {
            assert_eq!(
                csc_at(&csc, col_v, row),
                1.0,
                "{label}: outgoing storage v carries +1.0 on the frozen row"
            );
            assert_eq!(
                csc_at(&csc, col_v_in, row),
                -1.0,
                "{label}: incoming storage v_in carries −1.0 on the frozen row"
            );
            // Exactly 2 entries on the whole row: +1 v, −1 v_in, nothing else.
            let n_entries: usize = (0..csc.0.len().saturating_sub(1))
                .map(|col| {
                    let start = usize::try_from(csc.0[col]).unwrap();
                    let end = usize::try_from(csc.0[col + 1]).unwrap();
                    csc.1[start..end]
                        .iter()
                        .filter(|&&r| usize::try_from(r).is_ok_and(|r| r == row))
                        .count()
                })
                .sum();
            assert_eq!(
                n_entries, 2,
                "{label}: frozen-identity row must have EXACTLY 2 entries \
                 (+1 v, −1 v_in) — found {n_entries}, indicating corruption by a \
                 chained link's routed terms"
            );
            assert_eq!(row_lower[row], 0.0, "{label}: frozen RHS lower == 0");
            assert_eq!(row_upper[row], 0.0, "{label}: frozen RHS upper == 0");
        };

        // H2's row clean (NOT corrupted by H1's terms — the bug this fix removes).
        assert_clean_frozen(off.water_row_h2, off.h2_idx, off.col_storage_in_h2, "H2");
        // H1's row likewise clean.
        assert_clean_frozen(off.water_row_h1, off.h1_idx, off.col_storage_in_h1, "H1");
    }

    /// CHAIN-to-sink: a cascade `H1 → H2` where H1 AND H2 are BOTH `PreFilling` and
    /// H2 is terminal (no downstream). The whole chain is `PreFilling`, so the
    /// resolved target is None (SINK): H1's and H2's water exits the system, and
    /// NEITHER frozen row is corrupted.
    #[test]
    fn chained_prefilling_all_the_way_down_is_sink_and_keeps_frozen_rows_clean() {
        let mut fixtures = PumpFixtures::new(
            vec![
                ret_hydro(1, Some(2), Some(RET_ENTRY_STAGE_ID), true),
                ret_hydro(2, None, Some(RET_ENTRY_STAGE_ID), true),
            ],
            Vec::new(),
        );
        let h1_idx = fixtures.hydro_pos[&EntityId(1)];
        let h2_idx = fixtures.hydro_pos[&EntityId(2)];
        fixtures
            .base
            .bounds
            .hydro_bounds_mut(h1_idx, 0)
            .water_withdrawal_m3s = 8.0;
        fixtures
            .base
            .bounds
            .hydro_bounds_mut(h2_idx, 0)
            .water_withdrawal_m3s = 19.0;
        let ctx = fixtures.make_ctx();
        let stage = two_block_stage(usize::try_from(RET_PREFILLING_ID).unwrap(), [300.0, 444.0]);
        let layout = StageLayout::new(&ctx, &stage, 0);
        let (row_lower, row_upper) = super::super::rows::fill_stage_rows(&ctx, &stage, 0, &layout);
        let csc = {
            let mut entries = build_stage_matrix_entries(&ctx, &stage, 0, &layout);
            for col in &mut entries {
                col.sort_unstable_by_key(|&(row, _)| row);
            }
            assemble_csc(&entries)
        };
        let z_h1 = layout.state.z_inflow.start + h1_idx;
        let z_h2 = layout.state.z_inflow.start + h2_idx;

        // Both links' inflow exits the system: neither z column lands on ANY water
        // row (no non-PreFilling downstream exists to receive it).
        for h in 0..layout.state.hydro_count {
            let r = layout.geometry.water_balance.start() + h;
            assert_eq!(
                csc_at(&csc, z_h1, r),
                0.0,
                "z_{{H1}} must not land on any water row in the all-PreFilling sink (row {r})"
            );
            assert_eq!(
                csc_at(&csc, z_h2, r),
                0.0,
                "z_{{H2}} must not land on any water row in the all-PreFilling sink (row {r})"
            );
        }

        // Both frozen rows are clean (exactly +1 v, −1 v_in, RHS 0) and no
        // withdrawal demand was folded onto any frozen RHS (the sink transfers
        // nothing).
        for (h_idx, label) in [(h1_idx, "H1"), (h2_idx, "H2")] {
            let row = layout.geometry.water_balance.start() + h_idx;
            assert_eq!(csc_at(&csc, h_idx, row), 1.0, "{label}: v +1.0");
            assert_eq!(
                csc_at(&csc, layout.state.storage_in.start + h_idx, row),
                -1.0,
                "{label}: v_in −1.0"
            );
            assert_eq!(
                row_lower[row], 0.0,
                "{label}: frozen RHS 0 (no withdrawal folded in)"
            );
            assert_eq!(row_upper[row], 0.0, "{label}: frozen RHS 0");
        }
    }

    // ── Chronological PreFilling per-block frozen identity + short-circuit ────────

    /// Resolved offsets for a chronological H1→H2→H3 short-circuit probe (`n_blks = 2`).
    struct ChrScOffsets {
        n_blks: usize,
        h2_idx: usize,
        d_idx: usize,
        z_h2: usize,
        col_storage_in_h2: usize,
        h1_turbine: Vec<usize>,
        h1_spillage: Vec<usize>,
    }

    /// Build the cascade `H1(id 1) → H2(id 2, filling) → H3(id 3)` at the canonical
    /// `PreFilling` stage under `BlockMode::Chronological`, with H2's resolved
    /// per-stage withdrawal set to `withdrawal_h`. Returns the assembled CSC, the
    /// `(row_lower, row_upper)` vectors, the resolved [`StageLayout`] (block-major
    /// addressing reads through its accessors), and the offsets the assertions read.
    #[expect(
        clippy::type_complexity,
        reason = "each case returns the raw build tuple its assertions destructure"
    )]
    fn build_chronological_shortcircuit_case(
        withdrawal_h: f64,
    ) -> (
        (Vec<i32>, Vec<i32>, Vec<f64>),
        Vec<f64>,
        Vec<f64>,
        StageLayout<'static>,
        ChrScOffsets,
    ) {
        let mut fixtures = PumpFixtures::new(
            vec![
                ret_hydro(1, Some(2), None, false),
                ret_hydro(2, Some(3), Some(RET_ENTRY_STAGE_ID), true),
                ret_hydro(3, None, None, false),
            ],
            Vec::new(),
        );
        let h2_idx = fixtures.hydro_pos[&EntityId(2)];
        fixtures
            .base
            .bounds
            .hydro_bounds_mut(h2_idx, 0)
            .water_withdrawal_m3s = withdrawal_h;
        let h1_idx = fixtures.hydro_pos[&EntityId(1)];
        let d_idx = fixtures.hydro_pos[&EntityId(3)];
        let fixtures = Box::leak(Box::new(fixtures));
        let ctx = Box::leak(Box::new(fixtures.make_ctx()));
        let mut stage =
            two_block_stage(usize::try_from(RET_PREFILLING_ID).unwrap(), [300.0, 444.0]);
        stage.block_mode = BlockMode::Chronological;
        let stage = Box::leak(Box::new(stage));
        let layout = StageLayout::new(ctx, stage, 0);
        let (row_lower, row_upper) = super::super::rows::fill_stage_rows(ctx, stage, 0, &layout);
        let csc = {
            let mut entries = build_stage_matrix_entries(ctx, stage, 0, &layout);
            for col in &mut entries {
                col.sort_unstable_by_key(|&(row, _)| row);
            }
            assemble_csc(&entries)
        };
        let offsets = ChrScOffsets {
            n_blks: layout.clock.n_blks(),
            h2_idx,
            d_idx,
            z_h2: layout.state.z_inflow.start + h2_idx,
            col_storage_in_h2: layout.state.storage_in.start + h2_idx,
            h1_turbine: (0..layout.clock.n_blks())
                .map(|blk| {
                    layout
                        .geometry
                        .turbine_col(HydroCell::new(h1_idx), BlockIdx::new(blk))
                })
                .collect(),
            h1_spillage: (0..layout.clock.n_blks())
                .map(|blk| {
                    layout
                        .geometry
                        .spillage_col(HydroSys::new(h1_idx), BlockIdx::new(blk))
                })
                .collect(),
        };
        (csc, row_lower, row_upper, layout, offsets)
    }

    /// AC#1: a chronological `K = 2` `PreFilling` hydro H2 emits `K` block-major
    /// frozen-identity rows `row_water + h·K + (k−1)`, each carrying EXACTLY two
    /// entries: `+1.0` on `Sᵏ` and `−1.0` on `Sᵏ⁻¹`, and nothing else (no
    /// flow/inflow/loss/withdrawal term — any coupling makes `β_h` stale-nonzero).
    /// The per-hydro `h·K` stride is what eliminates the single-row collision into a
    /// neighbour's block row.
    #[test]
    fn chronological_prefilling_frozen_identity_per_block() {
        let (csc, row_lower, row_upper, layout, off) = build_chronological_shortcircuit_case(9.0);
        let h = off.h2_idx;
        let n_blks = off.n_blks;

        for k in 1..=n_blks {
            let blk = k - 1;
            let row = layout.geometry.water_balance.start() + h * n_blks + blk;
            assert_eq!(
                csc_at(
                    &csc,
                    layout.block_storage_col(HydroSys::new(h), Boundary::from_index(k, n_blks)),
                    row
                ),
                1.0,
                "block {k}: Sᵏ carries +1.0 on the frozen-identity row"
            );
            assert_eq!(
                csc_at(
                    &csc,
                    layout.block_storage_col(HydroSys::new(h), Boundary::from_index(k - 1, n_blks)),
                    row
                ),
                -1.0,
                "block {k}: Sᵏ⁻¹ carries −1.0 on the frozen-identity row"
            );
            // No other column touches this block row (frozen identity Sᵏ − Sᵏ⁻¹ = 0):
            // count CSC entries landing on it across EVERY column.
            let entries_on_row = (0..csc.0.len() - 1)
                .filter(|&col| csc_at(&csc, col, row) != 0.0)
                .count();
            assert_eq!(
                entries_on_row, 2,
                "block {k}: frozen-identity row must have EXACTLY two entries (Sᵏ, Sᵏ⁻¹)"
            );
            assert_eq!(row_lower[row], 0.0, "block {k}: frozen RHS lower == 0");
            assert_eq!(row_upper[row], 0.0, "block {k}: frozen RHS upper == 0");
        }
    }

    /// AC#2: H2's per-block short-circuit lands `−τ_k` on the DOWNSTREAM target H3's
    /// block-`k` rows — H2's incremental inflow (`z_{H2}`) and H2's upstream H1
    /// releases — and NOTHING on H2's own (frozen) block rows. H1 is NOT a standard
    /// upstream of H3 (the cascade edge is H1→H2→H3), so a nonzero `−τ_k` on H1's
    /// columns at H3's block row can only come from the re-route.
    #[test]
    fn chronological_prefilling_shortcircuit_per_block() {
        let withdrawal_h = 17.5_f64;
        let (csc, _rl, row_upper, layout, off) =
            build_chronological_shortcircuit_case(withdrawal_h);
        let n_blks = off.n_blks;

        for k in 1..=n_blks {
            let blk = k - 1;
            let tau_k = [300.0_f64, 444.0][blk] * M3S_TO_HM3;
            let row_d = layout.geometry.water_balance.start() + off.d_idx * n_blks + blk;
            let row_h = layout.geometry.water_balance.start() + off.h2_idx * n_blks + blk;

            assert_eq!(
                csc_at(&csc, off.z_h2, row_d),
                -tau_k,
                "block {k}: z_{{H2}} carries −τ_k on H3's block row (re-routed inflow)"
            );
            assert_eq!(
                csc_at(&csc, off.z_h2, row_h),
                0.0,
                "block {k}: z_{{H2}} carries nothing on H2's own frozen block row"
            );
            assert_eq!(
                csc_at(&csc, off.h1_turbine[blk], row_d),
                -tau_k,
                "block {k}: H1 turbine carries −τ_k on H3's block row (cascade re-route)"
            );
            assert_eq!(
                csc_at(&csc, off.h1_spillage[blk], row_d),
                -tau_k,
                "block {k}: H1 spillage carries −τ_k on H3's block row"
            );
            assert_eq!(
                csc_at(&csc, off.h1_turbine[blk], row_h),
                0.0,
                "block {k}: H1 turbine absent from H2's own frozen block row"
            );
            assert_eq!(
                csc_at(&csc, off.col_storage_in_h2, row_d),
                0.0,
                "block {k}: H2's incoming storage stays on H2's row, not routed to H3"
            );
        }

        // Withdrawal demand transfers per block: H3's block-`k` row_upper drops by
        // τ_k·withdrawal_h versus a no-withdrawal build (H3's own withdrawal is 0).
        let (_csc0, _rl0, row_upper0, layout0, off0) = build_chronological_shortcircuit_case(0.0);
        for k in 1..=n_blks {
            let blk = k - 1;
            let tau_k = [300.0_f64, 444.0][blk] * M3S_TO_HM3;
            let row_d = layout.geometry.water_balance.start() + off.d_idx * n_blks + blk;
            let row_d0 = layout0.geometry.water_balance.start() + off0.d_idx * n_blks + blk;
            assert_eq!(
                row_upper0[row_d0] - row_upper[row_d],
                tau_k * withdrawal_h,
                "block {k}: H3's block RHS drops by τ_k·withdrawal_{{H2}}"
            );
        }
    }

    /// AC#3: a chronological `K = 1` `PreFilling` build is byte-identical to the
    /// parallel `PreFilling` build — `τ_1 = ζ`, no interior boundary, the single
    /// chained frozen row IS the parallel frozen row, and the per-block short-circuit
    /// collapses to the single-row `−ζ`/`−ζ·withdrawal` parallel form. Covers the CSC
    /// arrays and both RHS vectors.
    #[test]
    fn chronological_k1_prefilling_byte_identical() {
        let withdrawal_h = 11.0_f64;
        let build = |block_mode: cobre_core::BlockMode| {
            let mut fixtures = PumpFixtures::new(
                vec![
                    ret_hydro(1, Some(2), None, false),
                    ret_hydro(2, Some(3), Some(RET_ENTRY_STAGE_ID), true),
                    ret_hydro(3, None, None, false),
                ],
                Vec::new(),
            );
            let h2_idx = fixtures.hydro_pos[&EntityId(2)];
            fixtures
                .base
                .bounds
                .hydro_bounds_mut(h2_idx, 0)
                .water_withdrawal_m3s = withdrawal_h;
            let ctx = fixtures.make_ctx();
            let mut stage =
                two_block_stage(usize::try_from(RET_PREFILLING_ID).unwrap(), [372.0, 372.0]);
            stage.blocks.truncate(1);
            stage.block_mode = block_mode;
            let layout = StageLayout::new(&ctx, &stage, 0);
            let (rl, ru) = super::super::rows::fill_stage_rows(&ctx, &stage, 0, &layout);
            let csc = {
                let mut entries = build_stage_matrix_entries(&ctx, &stage, 0, &layout);
                for col in &mut entries {
                    col.sort_unstable_by_key(|&(row, _)| row);
                }
                assemble_csc(&entries)
            };
            (csc, rl, ru)
        };

        let (csc_p, rl_p, ru_p) = build(BlockMode::Parallel);
        let (csc_c, rl_c, ru_c) = build(BlockMode::Chronological);

        assert_eq!(
            csc_p.0, csc_c.0,
            "K=1 PreFilling col_starts must be byte-identical"
        );
        assert_eq!(
            csc_p.1, csc_c.1,
            "K=1 PreFilling row_indices must be byte-identical"
        );
        assert_eq!(
            csc_p.2, csc_c.2,
            "K=1 PreFilling values must be byte-identical"
        );
        assert_eq!(
            rl_p, rl_c,
            "K=1 PreFilling row_lower must be byte-identical"
        );
        assert_eq!(
            ru_p, ru_c,
            "K=1 PreFilling row_upper must be byte-identical"
        );
    }

    // ── A PreFilling plant's maturing transit bucket ─────────────────────────────

    const EXITED_UP_ID: i32 = 1;
    const EXITED_MID_ID: i32 = 2;
    const EXITED_DOWN_ID: i32 = 3;

    /// `up -> mid -> down` (`mid -> down` omitted when `!with_downstream`) with one
    /// depth-1 travel-time arc `up -> mid`. `mid.exit_stage_id = Some(1)` makes it
    /// `PreFilling` at `stage.id = 1` while its incoming bucket still matures.
    fn exited_mid_fixtures(
        with_downstream: bool,
        arrival_density: Option<Vec<f64>>,
    ) -> PumpFixtures {
        let mut mid = fixture_hydro_ds(EXITED_MID_ID, with_downstream.then_some(EXITED_DOWN_ID));
        mid.exit_stage_id = Some(1);
        let mut hydros = vec![fixture_hydro_ds(EXITED_UP_ID, Some(EXITED_MID_ID)), mid];
        if with_downstream {
            hydros.push(fixture_hydro_ds(EXITED_DOWN_ID, None));
        }
        let mut fixtures = PumpFixtures::new(hydros, Vec::new());
        let up_idx = fixtures.hydro_pos[&EntityId(EXITED_UP_ID)];
        let mid_idx = fixtures.hydro_pos[&EntityId(EXITED_MID_ID)];
        fixtures.base.topology.column_order = vec![(HydroSys::new(mid_idx), 1)];
        fixtures.base.topology.arc_stage_weights = HashMap::from([(up_idx, vec![vec![0.5, 0.5]])]);
        fixtures.base.topology.per_stage_mask = vec![vec![1]];
        if let Some(density) = arrival_density {
            fixtures.base.topology.arc_arrival_density =
                HashMap::from([(up_idx, vec![Some(density)])]);
        }
        fixtures
    }

    #[test]
    fn prefilling_plants_maturing_bucket_lands_on_the_short_circuit_target_row() {
        // Non-uniform on purpose: a parallel fill that read the density would not write -1.0.
        let mut fixtures = exited_mid_fixtures(true, Some(vec![0.9, 0.1]));
        let mid_idx = fixtures.hydro_pos[&EntityId(EXITED_MID_ID)];
        let down_idx = fixtures.hydro_pos[&EntityId(EXITED_DOWN_ID)];
        let ctx = fixtures.make_ctx();
        let mut stage = two_block_stage(0, [300.0, 444.0]);
        stage.id = 1;
        let layout = StageLayout::new(&ctx, &stage, 0);
        let csc = build_sorted_csc(&ctx, &stage, 0, &layout);

        let bucket_col = ctx.state.transit_buckets_in.start;
        let down_row = layout
            .geometry
            .water_balance_row(HydroSys::new(down_idx), BlockIdx::new(0));
        let mid_row = layout
            .geometry
            .water_balance_row(HydroSys::new(mid_idx), BlockIdx::new(0));
        assert_eq!(
            csc_at(&csc, bucket_col, down_row),
            -1.0,
            "mid's maturing bucket must land at -1.0 on the short-circuit target's row"
        );
        assert_eq!(
            csc_at(&csc, bucket_col, mid_row),
            0.0,
            "mid's frozen-identity row must carry nothing of its maturing bucket"
        );
    }

    #[test]
    fn prefilling_plants_maturing_bucket_spreads_by_arrival_density_on_the_target_block_rows() {
        let density = [0.25, 0.75];
        let mut fixtures = exited_mid_fixtures(true, Some(density.to_vec()));
        let mid_idx = fixtures.hydro_pos[&EntityId(EXITED_MID_ID)];
        let down_idx = fixtures.hydro_pos[&EntityId(EXITED_DOWN_ID)];
        let ctx = fixtures.make_ctx();
        let mut stage = chronological_stage(0, &[300.0, 444.0]);
        stage.id = 1;
        let layout = StageLayout::new(&ctx, &stage, 0);
        let csc = build_sorted_csc(&ctx, &stage, 0, &layout);

        let bucket_col = ctx.state.transit_buckets_in.start;
        for (k, &rho_k) in density.iter().enumerate() {
            let down_row = layout
                .geometry
                .water_balance_row(HydroSys::new(down_idx), BlockIdx::new(k));
            let mid_row = layout
                .geometry
                .water_balance_row(HydroSys::new(mid_idx), BlockIdx::new(k));
            assert_eq!(
                csc_at(&csc, bucket_col, down_row),
                -rho_k,
                "block {k}: mid's maturing bucket must land at -arrival_density[{k}] on the \
                 short-circuit target's block row"
            );
            assert_eq!(
                csc_at(&csc, bucket_col, mid_row),
                0.0,
                "block {k}: mid's frozen block row must carry nothing of its maturing bucket"
            );
        }
    }

    #[test]
    fn prefilling_plants_maturing_bucket_at_a_sink_lands_on_no_row() {
        for block_mode in [BlockMode::Parallel, BlockMode::Chronological] {
            let mut fixtures = exited_mid_fixtures(false, None);
            let ctx = fixtures.make_ctx();
            let mut stage = two_block_stage(0, [300.0, 444.0]);
            stage.id = 1;
            stage.block_mode = block_mode;
            let layout = StageLayout::new(&ctx, &stage, 0);
            let csc = build_sorted_csc(&ctx, &stage, 0, &layout);

            let bucket_col = ctx.state.transit_buckets_in.start;
            assert_eq!(
                csc.0[bucket_col],
                csc.0[bucket_col + 1],
                "{block_mode:?}: with no non-PreFilling downstream, mid's maturing bucket must \
                 land on no row (the water leaves at the system outlet)"
            );
        }
    }

    // ── Per-block FPHA & evaporation (block-local average storage) ───────────────

    use crate::hydro_models::LinearizedEvaporation;

    const FPHA_GAMMA_V: f64 = 0.2;
    const EVAP_SLOPE: f64 = 0.03;
    const EVAP_INTERCEPT: f64 = 1.5;

    /// One non-filling hydro with a single-plane FPHA production model and a
    /// linearized evaporation model, built under `block_mode` with two blocks
    /// `[300.0, 444.0]` (truncated to one for the `K = 1` cases). Returns the
    /// assembled CSC, the `(row_lower, row_upper)` vectors, the column
    /// `(col_lower, col_upper, objective)` vectors, and the resolved `StageLayout`.
    #[expect(
        clippy::type_complexity,
        reason = "each case returns the raw build tuple its assertions destructure"
    )]
    fn build_fpha_evap_case(
        block_mode: cobre_core::BlockMode,
        durations: &[f64],
    ) -> (
        (Vec<i32>, Vec<i32>, Vec<f64>),
        Vec<f64>,
        Vec<f64>,
        (Vec<f64>, Vec<f64>, Vec<f64>),
        StageLayout<'static>,
    ) {
        let mut fixtures = PumpFixtures::new(vec![ret_hydro(1, None, None, false)], Vec::new())
            .with_evap_penalties(7.0, 11.0);
        fixtures.base.production_models = ProductionModelSet::new(
            vec![vec![ResolvedProductionModel::Fpha {
                planes: vec![FphaPlane {
                    intercept: 1.0,
                    gamma_v: FPHA_GAMMA_V,
                    gamma_q: 0.5,
                    gamma_s: 0.05,
                }],
            }]],
            &fixtures.base.hydros,
            N_STAGES,
        );
        fixtures.base.evaporation_models =
            EvaporationModelSet::new(vec![EvaporationModel::Linearized {
                coefficients: vec![LinearizedEvaporation {
                    intercept_m3s: EVAP_INTERCEPT,
                    volume_slope_m3s_per_hm3: EVAP_SLOPE,
                }],
                reference_volumes_hm3: vec![0.0],
            }]);
        let fixtures = Box::leak(Box::new(fixtures));
        let ctx = Box::leak(Box::new(fixtures.make_ctx()));
        let mut stage = two_block_stage(0, [300.0, 444.0]);
        stage.blocks.truncate(durations.len());
        for (blk, &d) in durations.iter().enumerate() {
            stage.blocks[blk].duration_hours = d;
        }
        stage.block_mode = block_mode;
        let stage = Box::leak(Box::new(stage));
        let layout = StageLayout::new(ctx, stage, 0);
        let (row_lower, row_upper) = super::super::rows::fill_stage_rows(ctx, stage, 0, &layout);
        let cols = super::super::columns::fill_stage_columns(ctx, stage, 0, &layout);
        let csc = {
            let mut entries = build_stage_matrix_entries(ctx, stage, 0, &layout);
            for col in &mut entries {
                col.sort_unstable_by_key(|&(row, _)| row);
            }
            assemble_csc(&entries)
        };
        (csc, row_lower, row_upper, cols, layout)
    }

    /// AC#1: a chronological `K = 2` FPHA plane row for block `k` carries `−γᵥ/2`
    /// on BOTH `block_storage_col(h, k−1)` (Sᵏ⁻¹) and `block_storage_col(h, k)` (Sᵏ)
    /// — the block-local average storage, both columns (D06), never one.
    #[test]
    fn chronological_fpha_uses_block_local_average() {
        let (csc, _rl, _ru, _cols, layout) =
            build_fpha_evap_case(BlockMode::Chronological, &[300.0, 444.0]);
        let h = 0_usize;
        let half_gamma_v = -FPHA_GAMMA_V / 2.0;
        let n_blks = 2_usize;
        // One plane, so block `k`'s FPHA row is at row_fpha_start + blk.
        for k in 1..=n_blks {
            let blk = k - 1;
            let row = layout.geometry.fpha.start + blk;
            assert_eq!(
                csc_at(
                    &csc,
                    layout.block_storage_col(HydroSys::new(h), Boundary::from_index(k - 1, n_blks)),
                    row
                ),
                half_gamma_v,
                "block {k}: −γᵥ/2 on Sᵏ⁻¹"
            );
            assert_eq!(
                csc_at(
                    &csc,
                    layout.block_storage_col(HydroSys::new(h), Boundary::from_index(k, n_blks)),
                    row
                ),
                half_gamma_v,
                "block {k}: −γᵥ/2 on Sᵏ"
            );
        }
        // S⁰, the interior boundary S¹, and Sᴷ are three distinct columns, so the
        // two block rows average genuinely block-local storage (no aliasing).
        let s0 = layout.block_storage_col(HydroSys::new(h), Boundary::Incoming);
        let s1 = layout.block_storage_col(HydroSys::new(h), Boundary::Interior(1));
        let s2 = layout.block_storage_col(HydroSys::new(h), Boundary::Outgoing);
        assert_ne!(s0, s1, "S⁰ and S¹ distinct");
        assert_ne!(s1, s2, "S¹ and Sᴷ distinct");
        assert_ne!(s0, s2, "S⁰ and Sᴷ distinct");
    }

    /// AC#2: a chronological `K = 2` evaporating hydro emits `K` evaporation rows,
    /// each with `−slope/2` on its own `(Sᵏ⁻¹, Sᵏ)` pair; each block's evaporation
    /// flow appears in that block's water row with `+τ_k`; each block's flow column
    /// is BOUNDED `[−q_max, +q_max]` (the wrong-bounds bug: leaving the extra
    /// per-block flow columns unbounded); and each block's directional slack
    /// objective is the cost times THAT block's `duration_hours` (block-scoped like
    /// the flow's `+τ_k` term — not the stage-total hours on every block, which would
    /// inflate the penalty `K`-fold).
    #[test]
    fn chronological_evaporation_per_block() {
        let (csc, rl, ru, cols, layout) =
            build_fpha_evap_case(BlockMode::Chronological, &[300.0, 444.0]);
        let (col_lower, col_upper, objective) = cols;
        let h = 0_usize;
        let local = 0_usize;
        let n_blks = 2_usize;
        let half_slope = -EVAP_SLOPE / 2.0;
        let q_max = (EVAP_INTERCEPT + EVAP_SLOPE * 100.0).abs() * 2.0;

        for k in 1..=n_blks {
            let blk = k - 1;
            let evap_row = layout.row_evap_start() + local * n_blks + blk;
            let tau_k = [300.0_f64, 444.0][blk] * M3S_TO_HM3;

            // Block-local average storage on the evaporation row.
            assert_eq!(
                csc_at(
                    &csc,
                    layout.block_storage_col(HydroSys::new(h), Boundary::from_index(k - 1, n_blks)),
                    evap_row
                ),
                half_slope,
                "block {k}: −slope/2 on Sᵏ⁻¹"
            );
            assert_eq!(
                csc_at(
                    &csc,
                    layout.block_storage_col(HydroSys::new(h), Boundary::from_index(k, n_blks)),
                    evap_row
                ),
                half_slope,
                "block {k}: −slope/2 on Sᵏ"
            );
            // The equality-row intercept is replicated per block.
            assert_eq!(rl[evap_row], EVAP_INTERCEPT, "block {k}: evap RHS lower");
            assert_eq!(ru[evap_row], EVAP_INTERCEPT, "block {k}: evap RHS upper");

            let flow_col = layout.evap_flow_col(EvapLocal::new(local), BlockIdx::new(blk));
            // Flow enters block k's water row with +τ_k.
            let water_row = layout.geometry.water_balance.start() + h * n_blks + blk;
            assert_eq!(
                csc_at(&csc, flow_col, water_row),
                tau_k,
                "block {k}: evap flow carries +τ_k on its water row"
            );
            // Flow column bounded [−q_max, +q_max], NOT the default [0, +∞).
            assert_eq!(col_lower[flow_col], -q_max, "block {k}: flow lower −q_max");
            assert_eq!(col_upper[flow_col], q_max, "block {k}: flow upper +q_max");
            // Directional slacks carry nonzero objective, weighted by THIS block's
            // hours (not the stage-total, which would inflate the penalty K-fold).
            let block_hours = [300.0_f64, 444.0][blk];
            assert_eq!(
                objective[layout.evap_f_plus_col(EvapLocal::new(local), BlockIdx::new(blk))],
                7.0 * block_hours,
                "block {k}: f_plus objective"
            );
            assert_eq!(
                objective[layout.evap_f_minus_col(EvapLocal::new(local), BlockIdx::new(blk))],
                11.0 * block_hours,
                "block {k}: f_minus objective"
            );
        }
    }

    /// AC#3: a chronological `K = 1` FPHA + evaporation build is byte-identical to
    /// the parallel build — one water row, one FPHA plane row on `(S⁰, Sᴷ)`, one
    /// evaporation row/triple, and the flow's single `+ζ` water term. Covers the
    /// CSC arrays, both RHS vectors, and the column bound/objective vectors.
    #[test]
    fn chronological_k1_fpha_evap_byte_identical() {
        let (csc_p, rl_p, ru_p, (cl_p, cu_p, obj_p), _lp) =
            build_fpha_evap_case(BlockMode::Parallel, &[372.0]);
        let (csc_c, rl_c, ru_c, (cl_c, cu_c, obj_c), _lc) =
            build_fpha_evap_case(BlockMode::Chronological, &[372.0]);

        assert_eq!(csc_p.0, csc_c.0, "K=1 col_starts byte-identical");
        assert_eq!(csc_p.1, csc_c.1, "K=1 row_indices byte-identical");
        assert_eq!(csc_p.2, csc_c.2, "K=1 values byte-identical");
        assert_eq!(rl_p, rl_c, "K=1 row_lower byte-identical");
        assert_eq!(ru_p, ru_c, "K=1 row_upper byte-identical");
        assert_eq!(cl_p, cl_c, "K=1 col_lower byte-identical");
        assert_eq!(cu_p, cu_c, "K=1 col_upper byte-identical");
        assert_eq!(obj_p, obj_c, "K=1 objective byte-identical");
    }

    /// A parallel two-block FPHA + evaporation build is byte-identical to the
    /// pre-change parallel build in structure: the single FPHA row uses the stage
    /// endpoints `(S⁰, Sᴷ)` and the single evaporation row/triple with the flow's
    /// `+ζ` water term (the parallel path is unchanged by the per-block work).
    #[test]
    fn parallel_fpha_evap_uses_stage_endpoints() {
        let (csc, rl, _ru, cols, layout) =
            build_fpha_evap_case(BlockMode::Parallel, &[300.0, 444.0]);
        let (col_lower, col_upper, _obj) = cols;
        let h = 0_usize;
        let local = 0_usize;
        let col_s_in = layout.state.storage_in.start + h;
        let col_s_out = h;

        // Single FPHA row (one plane) on the stage endpoints, −γᵥ/2 on both.
        let fpha_row = layout.geometry.fpha.start;
        assert_eq!(csc_at(&csc, col_s_in, fpha_row), -FPHA_GAMMA_V / 2.0);
        assert_eq!(csc_at(&csc, col_s_out, fpha_row), -FPHA_GAMMA_V / 2.0);

        // Single evaporation row (the stage slot) on the stage endpoints.
        let evap_row = layout.row_evap_start();
        assert_eq!(csc_at(&csc, col_s_in, evap_row), -EVAP_SLOPE / 2.0);
        assert_eq!(csc_at(&csc, col_s_out, evap_row), -EVAP_SLOPE / 2.0);
        assert_eq!(rl[evap_row], EVAP_INTERCEPT);

        // Flow enters the single water row with +ζ (Σ_k τ_k).
        let flow_col = layout.evap_flow_col(EvapLocal::new(local), BlockIdx::new(0));
        let zeta = (300.0_f64 + 444.0) * M3S_TO_HM3;
        assert_eq!(
            csc_at(&csc, flow_col, layout.geometry.water_balance.start() + h),
            zeta,
            "parallel evap flow carries +ζ on the single water row"
        );
        let q_max = (EVAP_INTERCEPT + EVAP_SLOPE * 100.0).abs() * 2.0;
        assert_eq!(col_lower[flow_col], -q_max);
        assert_eq!(col_upper[flow_col], q_max);
    }

    /// A parallel multi-block stage evaporates as one stage-level quantity: one slot per
    /// evaporating hydro, its flow coupled into the single water row with `ζ`, and its
    /// violation slacks priced at the violation cost times the stage's total hours.
    #[test]
    fn parallel_multi_block_evap_slack_price_matches_water_it_moves() {
        let durations = [300.0_f64, 444.0];
        let (csc, _rl, _ru, (_cl, _cu, obj), layout) =
            build_fpha_evap_case(BlockMode::Parallel, &durations);
        let geometry = layout.geometry.clone();
        let total_hours: f64 = durations.iter().sum();
        let zeta = total_hours * M3S_TO_HM3;

        assert_eq!(
            geometry.evap_indices.len(),
            geometry.evap_hydro_indices.len(),
            "a parallel stage must reserve exactly one evaporation slot per evaporating hydro"
        );
        let slot = geometry.evap_indices[0];
        assert_eq!(
            csc_at(
                &csc,
                slot.evaporation_flow_col,
                layout.geometry.water_balance.start()
            ),
            zeta,
            "the stage-level evaporation flow must move ζ = {zeta} on the water row"
        );
        for (col, cost, name) in [
            (slot.f_evap_plus_col, 7.0, "f_evap_plus"),
            (slot.f_evap_minus_col, 11.0, "f_evap_minus"),
        ] {
            let expected = cost * total_hours;
            assert!(
                (obj[col] - expected).abs() <= 1e-12 * expected,
                "{name} must cost {cost} per stage hour ({total_hours} h): expected {expected}, got {}",
                obj[col]
            );
        }
    }

    // ── Commissioning-dormant FPHA plant (A.1 regression) ────────────────────

    /// A commissioning FPHA plant with a future `entry_stage_id`: resolution is
    /// phase-blind (`ResolvedProductionModel::Fpha` at every stage, including
    /// stage 0) while `identify_fpha_hydros` gates it `PreFilling` out of
    /// `fpha_local_index` — the fixture that reproduced the
    /// commissioning-dormant FPHA panic before `StageProductionRole::Dormant`.
    /// A nonzero group `min_generation_mw` exercises the min-generation floor
    /// while the plant is dormant, so the operational-violation test below can
    /// confirm the row's own slack — not a `var_c` term the plant has no
    /// column for — is what keeps it satisfiable.
    fn dormant_fpha_fixture() -> PumpFixtures {
        let mut dormant = fixture_hydro(1);
        dormant.entry_stage_id = Some(2);
        dormant.unit_groups[0].min_generation_mw = 5.0;
        let production_models = ProductionModelSet::new(
            vec![vec![
                ResolvedProductionModel::Fpha {
                    planes: vec![FphaPlane {
                        intercept: 1000.0,
                        gamma_v: 4.0,
                        gamma_q: 0.6,
                        gamma_s: 0.3,
                    }],
                };
                N_STAGES
            ]],
            std::slice::from_ref(&dormant),
            N_STAGES,
        );
        PumpFixtures::new(vec![dormant], Vec::new()).with_production_models(production_models)
    }

    /// AC1: the LP builds (no `unreachable!` panic) and the dormant plant is
    /// excluded from the FPHA index, exactly as a resolved-Fpha commissioning
    /// plant that never reaches a stage template today would be.
    #[test]
    fn test_dormant_fpha_plant_is_excluded_from_fpha_index() {
        let mut fixture = dormant_fpha_fixture();
        let ctx = fixture.make_ctx();
        let stage = three_block_stage(0);
        let layout = StageLayout::new(&ctx, &stage, 0);

        assert!(
            layout.fpha_local_index[0].is_none(),
            "the dormant plant must be gated out of the FPHA index by identify_fpha_hydros"
        );
        assert!(
            layout.geometry.fpha_hydro_indices.is_empty(),
            "a solely-dormant plant reserves no FPHA generation column region at all"
        );
    }

    /// AC2: `fill_load_balance_entries` does not panic on a dormant FPHA plant
    /// and credits its bus with zero generation — never priced as
    /// `ConstantProductivity` on its frozen turbine column.
    #[test]
    fn test_dormant_fpha_plant_load_balance_contributes_nothing() {
        let mut fixture = dormant_fpha_fixture();
        let ctx = fixture.make_ctx();
        let stage = three_block_stage(0);
        let layout = StageLayout::new(&ctx, &stage, 0);

        let mut col_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); layout.num_cols];
        fill_load_balance_entries(&ctx, 0, &layout, &mut col_entries);

        let bus_pos = ctx.positions.bus(EntityId(1)).unwrap();
        let grid = layout.block_grid();
        let row_load = layout.geometry.load_balance.start();
        let cell = HydroCell::new(
            ctx.hydro_cell_index
                .cells_of(HydroSys::new(0))
                .next()
                .unwrap(),
        );

        for blk_idx in 0..layout.clock.n_blks() {
            let blk = BlockIdx::new(blk_idx);
            let row = grid.flat(row_load, bus_pos, blk);
            let col_turbine = layout.geometry.turbine_col(cell, blk);
            assert_eq!(
                entry_count_at(&col_entries, col_turbine, row),
                0,
                "blk {blk_idx}: a dormant FPHA plant must not be priced as \
                 ConstantProductivity on its frozen turbine column"
            );
        }
    }

    /// AC3: `fill_operational_violation_entries` does not panic on a dormant
    /// FPHA plant; the min-generation row's own slack alone absorbs the
    /// plant's nonzero floor — the row stays satisfiable via the slack, not a
    /// `var_c` term coupling a column the plant has none of.
    #[test]
    fn test_dormant_fpha_plant_operational_violation_contributes_nothing() {
        let mut fixture = dormant_fpha_fixture();
        let ctx = fixture.make_ctx();
        let stage = three_block_stage(0);
        let layout = StageLayout::new(&ctx, &stage, 0);

        let mut col_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); layout.num_cols];
        fill_operational_violation_entries(&ctx, 0, &layout, &mut col_entries);
        let (row_lower, _row_upper) = fill_stage_rows(&ctx, &stage, 0, &layout);

        let grid = layout.block_grid();
        let cell_idx = ctx
            .hydro_cell_index
            .cells_of(HydroSys::new(0))
            .next()
            .unwrap();
        let cell = HydroCell::new(cell_idx);

        for blk_idx in 0..layout.clock.n_blks() {
            let blk = BlockIdx::new(blk_idx);
            let row = grid.flat(layout.oper_violation.min_generation.start, cell_idx, blk);
            assert_eq!(
                row_lower[row], 5.0,
                "blk {blk_idx}: the dormant plant's own group min_generation_mw still \
                 sets a nonzero floor"
            );
            let col_turbine = layout.geometry.turbine_col(cell, blk);
            assert_eq!(
                entry_count_at(&col_entries, col_turbine, row),
                0,
                "blk {blk_idx}: no rho-priced turbine term for a plant with no productivity"
            );
            let col_slack = layout.geometry.generation_below_col(cell, blk);
            assert_eq!(
                raw_coeff_at(&col_entries, col_slack, row),
                1.0,
                "blk {blk_idx}: the row's own slack must still absorb the floor, keeping \
                 it satisfiable rather than structurally infeasible"
            );
            assert_eq!(
                entry_count_at(&col_entries, col_slack, row),
                1,
                "blk {blk_idx}: exactly one slack entry, no duplicate or leftover var_c term"
            );
        }
    }
}
