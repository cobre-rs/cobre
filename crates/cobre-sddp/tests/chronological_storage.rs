//! Solved per-block storage of the `chronological_storage` and `parallel_storage`
//! fixture decks, which differ only in `block_mode`: a chronological stage chains
//! storage from block to block and its blocks differ, a parallel stage holds
//! storage and evaporation constant across its blocks.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::path::PathBuf;

/// derived: the decks' `initial_conditions.json` storage of hydro 0.
const INITIAL_STORAGE_HM3: f64 = 100.0;
const N_BLOCKS: usize = 3;
const N_STAGES: usize = 2;

struct Block {
    storage_initial: f64,
    storage_final: f64,
    evaporation: f64,
}

fn deck_dir(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-6 * 1.0_f64.max(a.abs()).max(b.abs())
}

fn constant_across(blocks: &[Block], field: impl Fn(&Block) -> f64) -> bool {
    blocks
        .iter()
        .all(|block| close(field(block), field(&blocks[0])))
}

fn solved_blocks(deck: &str) -> Vec<Vec<Vec<Block>>> {
    let (_setup, mut results) = common::parity_hash::train_and_simulate_at_dir(
        &deck_dir(deck),
        cobre_solver::ActiveSolver::new,
    );
    assert!(!results.is_empty(), "{deck}: no simulated scenario");
    results.sort_by_key(|scenario| scenario.scenario_id);
    results
        .into_iter()
        .map(|mut scenario| {
            assert_eq!(scenario.stages.len(), N_STAGES, "{deck}: stages");
            scenario.stages.sort_by_key(|stage| stage.stage_id);
            scenario
                .stages
                .into_iter()
                .map(|stage| {
                    let mut rows: Vec<_> = stage
                        .hydros
                        .into_iter()
                        .filter(|row| row.hydro_id == 0)
                        .collect();
                    rows.sort_by_key(|row| row.block_id);
                    assert!(
                        rows.iter().map(|row| row.block_id).eq((0..N_BLOCKS)
                            .map(|b| Some(u32::try_from(b).expect("block index fits u32")))),
                        "{deck}: stage {} must have one row per block 0..{N_BLOCKS}",
                        stage.stage_id
                    );
                    rows.into_iter()
                        .map(|row| Block {
                            storage_initial: row.storage_initial_hm3,
                            storage_final: row.storage_final_hm3,
                            evaporation: row
                                .evaporation_m3s
                                .expect("hydro 0 evaporates in every block"),
                        })
                        .collect()
                })
                .collect()
        })
        .collect()
}

#[test]
fn chronological_storage_deck_chains_each_blocks_storage_through_the_stage() {
    let scenarios = solved_blocks("chronological_storage");

    assert!(
        scenarios.iter().flatten().any(|blocks| {
            !constant_across(blocks, |b| b.storage_final)
                && !constant_across(blocks, |b| b.evaporation)
        }),
        "no stage has blocks with differing final storage and evaporation"
    );

    for (s, stages) in scenarios.iter().enumerate() {
        assert!(
            close(stages[0][0].storage_initial, INITIAL_STORAGE_HM3),
            "scenario {s}: stage 0 block 0 must start at the initial storage"
        );
        for (t, blocks) in stages.iter().enumerate() {
            for b in 1..N_BLOCKS {
                assert!(
                    close(blocks[b].storage_initial, blocks[b - 1].storage_final),
                    "scenario {s} stage {t}: block {b} must start where block {} ended",
                    b - 1
                );
            }
        }
        assert!(
            close(
                stages[1][0].storage_initial,
                stages[0][N_BLOCKS - 1].storage_final
            ),
            "scenario {s}: stage 1 must start where stage 0's last block ended"
        );
    }
}

#[test]
fn parallel_storage_deck_holds_storage_and_evaporation_across_blocks() {
    let scenarios = solved_blocks("parallel_storage");

    for (s, stages) in scenarios.iter().enumerate() {
        for (t, blocks) in stages.iter().enumerate() {
            assert!(
                constant_across(blocks, |b| b.storage_initial),
                "scenario {s} stage {t}: storage_initial differs across blocks"
            );
            assert!(
                constant_across(blocks, |b| b.storage_final),
                "scenario {s} stage {t}: storage_final differs across blocks"
            );
            assert!(
                constant_across(blocks, |b| b.evaporation),
                "scenario {s} stage {t}: evaporation differs across blocks"
            );
        }
        assert!(
            stages[0]
                .iter()
                .all(|block| close(block.storage_initial, INITIAL_STORAGE_HM3)),
            "scenario {s}: stage 0 must start at the initial storage"
        );
        assert!(
            stages[1]
                .iter()
                .zip(&stages[0])
                .all(|(next, prev)| close(next.storage_initial, prev.storage_final)),
            "scenario {s}: stage 1 must start where stage 0 ended"
        );
    }
}
