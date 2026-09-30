//! Per-family structural pins for evaporation, FPHA and the delivery rings.
//!
//! Each test reads its family through the address owners (`StageGeometry`,
//! `StateSpace`, `DeliveryRing`, via `cobre_sddp::test_support::template_structure`),
//! never `start + h` arithmetic, over every committed deck and in-code study
//! (`common::for_each_study`).

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp
)]

mod common;

use std::collections::{HashMap, HashSet};

use cobre_core::BlockMode;
use cobre_sddp::indexer::{BlockIdx, Boundary, FphaCellLocal, HydroSys};
use cobre_sddp::test_support::decks::{SLOW_DECKS, committed_decks};
use cobre_sddp::test_support::template_structure::{
    RingLane, RingLaneKind, UnscaledMatrix, generation_column_owners, hours_to_hm3, ring_lanes,
    storage_column_owners, water_row_owners,
};

const TOL: f64 = 1e-12;

/// Asserts `common::for_each_study`'s visit count against the same formula
/// `cobre-cli`'s non-root-rebuild parity test uses, extended by the in-code
/// studies `for_each_study` also visits.
fn assert_full_sweep_count(count: usize) {
    let slow_tests_enabled = cfg!(feature = "slow-tests");
    let skipped = if slow_tests_enabled {
        0
    } else {
        committed_decks()
            .iter()
            .filter(|deck| SLOW_DECKS.contains(&deck.key.as_str()))
            .count()
    };
    assert_eq!(
        count,
        committed_decks().len() - skipped
            + common::in_code_studies::keyed_setups().len()
            + common::in_code_studies::structural_studies().len(),
        "every committed deck (minus SLOW_DECKS skips) plus every in-code and \
         structural study must be swept"
    );
}

#[test]
fn every_study_couples_each_evaporating_hydro_through_its_mode_slot_count() {
    let mut checked_slots = 0usize;
    let mut saw_parallel_multiblock = false;

    let count = common::for_each_study(|key, setup| {
        let state = setup.stage_state();
        let templates = &setup.inputs.stage_data.stage_templates;
        for (s, t) in templates.templates.iter().enumerate() {
            let geom = &templates.geometry_per_stage[s];
            if geom.evap_indices.is_empty() {
                continue;
            }
            let matrix = UnscaledMatrix::of(t);
            let storage_owners = storage_column_owners(geom, state);
            let water_owners = water_row_owners(geom, state.hydro_count);

            let mut per_hydro: HashMap<usize, Vec<(BlockIdx, f64)>> = HashMap::new();

            for e in &geom.evap_indices {
                let row = matrix.row(e.evap_row);
                let flow = row.iter().find(|&&(c, _)| c == e.evaporation_flow_col);
                assert_eq!(
                    flow.map(|&(_, v)| v),
                    Some(1.0),
                    "{key} stage {s}: evap row {} flow coefficient",
                    e.evap_row
                );
                let f_plus = row.iter().find(|&&(c, _)| c == e.f_evap_plus_col);
                assert_eq!(
                    f_plus.map(|&(_, v)| v),
                    Some(1.0),
                    "{key} stage {s}: evap row {} f_evap_plus coefficient",
                    e.evap_row
                );
                let f_minus = row.iter().find(|&&(c, _)| c == e.f_evap_minus_col);
                assert_eq!(
                    f_minus.map(|&(_, v)| v),
                    Some(-1.0),
                    "{key} stage {s}: evap row {} f_evap_minus coefficient",
                    e.evap_row
                );

                let storage: Vec<(HydroSys, Boundary, f64)> = row
                    .iter()
                    .filter_map(|&(c, v)| storage_owners.get(&c).map(|&(h, b)| (h, b, v)))
                    .collect();
                assert_eq!(
                    storage.len(),
                    2,
                    "{key} stage {s}: evap row {} storage column count",
                    e.evap_row
                );
                let h = storage[0].0;
                assert_eq!(
                    storage[1].0, h,
                    "{key} stage {s}: evap row {} storage columns span two hydros",
                    e.evap_row
                );
                let (v0, v1) = (storage[0].2, storage[1].2);
                assert!(
                    (v0 - v1).abs() <= TOL * v0.abs(),
                    "{key} stage {s}: evap row {} storage coefficients {v0} != {v1}",
                    e.evap_row
                );

                let flow_water_entries: Vec<(usize, f64)> = matrix
                    .col(e.evaporation_flow_col)
                    .iter()
                    .copied()
                    .filter(|&(r, _)| water_owners.contains_key(&r))
                    .collect();
                assert_eq!(
                    flow_water_entries.len(),
                    1,
                    "{key} stage {s}: evaporation_flow_col {} water-row entry count",
                    e.evaporation_flow_col
                );
                let (water_row, coeff) = flow_water_entries[0];
                let owner = water_owners.get(&water_row);
                let blk = match owner {
                    Some(&(h_water, blk)) if h_water == h => blk,
                    _ => panic!(
                        "{key} stage {s}: evaporation_flow_col {} water row {water_row} owner {owner:?} != hydro {h:?}",
                        e.evaporation_flow_col
                    ),
                };

                let boundaries: Vec<Boundary> = storage.iter().map(|&(_, b, _)| b).collect();
                let expected = match geom.block_mode {
                    BlockMode::Parallel => (Boundary::Incoming, Boundary::Outgoing),
                    BlockMode::Chronological => (
                        Boundary::from_index(blk.get(), geom.n_blks),
                        Boundary::from_index(blk.get() + 1, geom.n_blks),
                    ),
                };
                assert!(
                    boundaries.contains(&expected.0) && boundaries.contains(&expected.1),
                    "{key} stage {s}: evap row {} storage boundaries {boundaries:?} != {expected:?}",
                    e.evap_row
                );

                per_hydro.entry(h.get()).or_default().push((blk, coeff));
                checked_slots += 1;
            }

            let expected_slot_count = match geom.block_mode {
                BlockMode::Parallel => 1,
                BlockMode::Chronological => geom.n_blks,
            };
            let block_hours = &templates.block_hours_per_stage[s];
            let zeta = hours_to_hm3(block_hours.iter().sum());

            for (h_idx, slots) in &per_hydro {
                assert_eq!(
                    slots.len(),
                    expected_slot_count,
                    "{key} stage {s}: hydro sys {h_idx} evaporation slot count"
                );
                let sum: f64 = slots.iter().map(|&(_, v)| v).sum();
                assert!(
                    (sum - zeta).abs() <= TOL * zeta.abs(),
                    "{key} stage {s}: hydro sys {h_idx} evaporation coefficients sum {sum} != {zeta}"
                );
                if geom.block_mode == BlockMode::Chronological {
                    for &(blk, v) in slots {
                        let expected = hours_to_hm3(block_hours[blk.get()]);
                        assert!(
                            (v - expected).abs() <= TOL * expected.abs(),
                            "{key} stage {s}: hydro sys {h_idx} block {} coefficient {v} != {expected}",
                            blk.get()
                        );
                    }
                }
            }

            if geom.block_mode == BlockMode::Parallel
                && geom.n_blks > 1
                && !geom.evap_indices.is_empty()
            {
                saw_parallel_multiblock = true;
            }
        }
    });

    assert_full_sweep_count(count);
    assert!(checked_slots > 0, "no evaporation slot was checked");
    assert!(
        saw_parallel_multiblock,
        "no parallel multiblock evaporation slot was checked"
    );
}

#[test]
fn every_study_fpha_plane_row_averages_its_own_plants_storage() {
    let mut checked_rows = 0usize;
    let mut cells_per_hydro: HashMap<usize, HashSet<usize>> = HashMap::new();

    let count = common::for_each_study(|key, setup| {
        let state = setup.stage_state();
        let templates = &setup.inputs.stage_data.stage_templates;
        for (s, t) in templates.templates.iter().enumerate() {
            let geom = &templates.geometry_per_stage[s];
            if geom.fpha.is_empty() {
                continue;
            }
            let matrix = UnscaledMatrix::of(t);
            let generation_owners = generation_column_owners(geom);
            let storage_owners = storage_column_owners(geom, state);

            for r in geom.fpha.clone() {
                let row = matrix.row(r);

                let generation: Vec<(FphaCellLocal, BlockIdx, f64)> = row
                    .iter()
                    .filter_map(|&(c, v)| {
                        generation_owners.get(&c).map(|&(cell, blk)| (cell, blk, v))
                    })
                    .collect();
                assert_eq!(
                    generation.len(),
                    1,
                    "{key} stage {s}: fpha row {r} generation column count"
                );
                let (cell, blk, gen_v) = generation[0];
                assert_eq!(
                    gen_v, 1.0,
                    "{key} stage {s}: fpha row {r} generation coefficient"
                );

                let storage: Vec<(HydroSys, Boundary, f64)> = row
                    .iter()
                    .filter_map(|&(c, v)| storage_owners.get(&c).map(|&(h, b)| (h, b, v)))
                    .collect();
                assert_eq!(
                    storage.len(),
                    2,
                    "{key} stage {s}: fpha row {r} storage column count"
                );
                let h = storage[0].0;
                assert_eq!(
                    storage[1].0, h,
                    "{key} stage {s}: fpha row {r} storage columns span two hydros"
                );
                assert!(
                    geom.fpha_hydro_indices.contains(&h),
                    "{key} stage {s}: fpha row {r} hydro {h:?} not in fpha_hydro_indices"
                );
                let (v0, v1) = (storage[0].2, storage[1].2);
                assert!(
                    (v0 - v1).abs() <= TOL * v0.abs(),
                    "{key} stage {s}: fpha row {r} storage coefficients {v0} != {v1}"
                );
                let boundaries: Vec<Boundary> = storage.iter().map(|&(_, b, _)| b).collect();
                let expected = match geom.block_mode {
                    BlockMode::Parallel => (Boundary::Incoming, Boundary::Outgoing),
                    BlockMode::Chronological => (
                        Boundary::from_index(blk.get(), geom.n_blks),
                        Boundary::from_index(blk.get() + 1, geom.n_blks),
                    ),
                };
                assert!(
                    boundaries.contains(&expected.0) && boundaries.contains(&expected.1),
                    "{key} stage {s}: fpha row {r} storage boundaries {boundaries:?} != {expected:?}"
                );

                let expected_spill = geom.spillage_col(h, blk);
                assert!(
                    row.iter().any(|&(c, _)| c == expected_spill),
                    "{key} stage {s}: fpha row {r} missing spillage column {expected_spill}"
                );

                cells_per_hydro
                    .entry(h.get())
                    .or_default()
                    .insert(cell.get());
                checked_rows += 1;
            }
        }
    });

    assert_full_sweep_count(count);
    assert!(checked_rows > 0, "no fpha row was checked");
    assert!(
        cells_per_hydro.values().any(|cells| cells.len() >= 2),
        "no fpha plant with two or more generation cells was checked"
    );
}

/// This slot's definition row: `o` (`lane.out_cols[j]`) at `+1`, plus the
/// case-specific partner. A water lane's last slot has no partner column —
/// its selector instead requires the row carry no OTHER ring column at all.
fn definition_rows(
    matrix: &UnscaledMatrix,
    ring_col_owner: &HashMap<usize, usize>,
    o: usize,
    partner_candidates: &[usize],
) -> Vec<usize> {
    matrix
        .col(o)
        .iter()
        .filter(|&&(_, v)| v == 1.0)
        .map(|&(r, _)| r)
        .filter(|&r| {
            let entries = matrix.row(r);
            if partner_candidates.is_empty() {
                entries
                    .iter()
                    .all(|&(c, _)| c == o || !ring_col_owner.contains_key(&c))
            } else {
                partner_candidates
                    .iter()
                    .any(|&p| entries.iter().any(|&(c, v)| c == p && v == -1.0))
            }
        })
        .collect()
}

#[test]
fn every_study_ring_slot_has_one_signed_definition_row_or_is_frozen() {
    let mut saw_anticipated_definition = false;
    let mut saw_water_definition = false;

    let count = common::for_each_study(|key, setup| {
        let state = setup.stage_state();
        let templates = &setup.inputs.stage_data.stage_templates;
        for (s, t) in templates.templates.iter().enumerate() {
            let geom = &templates.geometry_per_stage[s];
            let lanes: Vec<RingLane> = ring_lanes(state, geom);
            if lanes.is_empty() {
                continue;
            }
            let matrix = UnscaledMatrix::of(t);

            let mut ring_col_owner: HashMap<usize, usize> = HashMap::new();
            for (lane_idx, lane) in lanes.iter().enumerate() {
                for &c in lane.out_cols.iter().chain(lane.in_cols.iter()) {
                    ring_col_owner.insert(c, lane_idx);
                }
            }

            let mut water_slot_total = 0usize;
            for (lane_idx, lane) in lanes.iter().enumerate() {
                let depth = lane.out_cols.len();
                match &lane.kind {
                    RingLaneKind::Anticipated { .. } => {
                        assert_eq!(
                            depth, state.k_max,
                            "{key} stage {s}: anticipated lane {lane_idx} out_cols.len() != state.k_max"
                        );
                    }
                    RingLaneKind::Water { .. } => water_slot_total += depth,
                }

                for j in 0..depth {
                    let o = lane.out_cols[j];
                    let masked = t.col_lower[o] == 0.0 && t.col_upper[o] == 0.0;

                    let partner_candidates: Vec<usize> = match &lane.kind {
                        RingLaneKind::Water { .. } => {
                            if j + 1 < depth {
                                vec![lane.in_cols[j + 1]]
                            } else {
                                vec![]
                            }
                        }
                        RingLaneKind::Anticipated { .. } => {
                            let mut candidates = vec![lane.in_cols[j]];
                            if let Some(decision_col) = lane.decision_col {
                                candidates.push(decision_col);
                            }
                            candidates
                        }
                    };

                    let rows = definition_rows(&matrix, &ring_col_owner, o, &partner_candidates);

                    if masked {
                        assert!(
                            rows.is_empty(),
                            "{key} stage {s}: masked ring slot {o} (lane {lane_idx}, slot {j}) has {} definition rows",
                            rows.len()
                        );
                        continue;
                    }
                    assert_eq!(
                        rows.len(),
                        1,
                        "{key} stage {s}: ring slot {o} (lane {lane_idx}, slot {j}) has {} definition rows, want 1",
                        rows.len()
                    );
                    for &(c, _) in matrix.row(rows[0]) {
                        if let Some(&owner) = ring_col_owner.get(&c) {
                            assert_eq!(
                                owner, lane_idx,
                                "{key} stage {s}: ring slot {o}'s definition row {} carries lane {owner}'s ring column {c}",
                                rows[0]
                            );
                        }
                    }

                    match &lane.kind {
                        RingLaneKind::Anticipated { .. } => saw_anticipated_definition = true,
                        RingLaneKind::Water { .. } => saw_water_definition = true,
                    }
                }
            }
            assert_eq!(
                water_slot_total, state.n_buckets,
                "{key} stage {s}: water lanes' total slot count != state.n_buckets"
            );
        }
    });

    assert_full_sweep_count(count);
    assert!(
        saw_anticipated_definition,
        "no anticipated ring definition row was checked"
    );
    assert!(
        saw_water_definition,
        "no water ring definition row was checked"
    );
}
