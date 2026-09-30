//! Declaration-order invariance regression test for [`ResolvedParameters`]:
//! permuting the input [`ScalarParameter`] set leaves the values and the
//! canonical `id_to_slot` key order that [`build_resolved_parameters`] produces
//! unchanged.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp
)]

use cobre_core::{EntityId, ParameterKind, ScalarParameter, StageId};
use cobre_sddp::build_resolved_parameters;
use cobre_sddp::energy_conversion::{EnergyConversionSet, HydroEnergyProductivityOverride};

/// Return the four fixed [`ScalarParameter`] entries permuted by `order`. The
/// last is a `PerStageBlock` parameter (a per-`(stage, block)` value) so the
/// permutation exercises the block axis, not only the stage axis.
fn make_params(order: &[usize]) -> Vec<ScalarParameter> {
    let alpha = ScalarParameter {
        id: EntityId(10),
        name: "alpha".to_string(),
        kind: ParameterKind::Constant { value: 3.6 },
    };
    let beta = ScalarParameter {
        id: EntityId(20),
        name: "beta".to_string(),
        kind: ParameterKind::PerStage {
            values: vec![1.0, 1.5, 2.0, 2.5],
        },
    };
    let gamma = ScalarParameter {
        id: EntityId(30),
        name: "gamma".to_string(),
        kind: ParameterKind::Seasonal {
            values: vec![(0, 0.9), (1, 1.1)],
        },
    };
    // Distinct value per (stage, block) across 4 stages x 2 blocks; block 1 always
    // differs from block 0 within a stage.
    let delta = ScalarParameter {
        id: EntityId(40),
        name: "delta".to_string(),
        kind: ParameterKind::PerStageBlock {
            values: vec![
                (0, 0, 0.0),
                (0, 1, 10.0),
                (1, 0, 1.0),
                (1, 1, 11.0),
                (2, 0, 2.0),
                (2, 1, 12.0),
                (3, 0, 3.0),
                (3, 1, 13.0),
            ],
        },
    };
    let pool = [alpha, beta, gamma, delta];
    order.iter().map(|&i| pool[i].clone()).collect()
}

#[test]
fn scalar_parameters_resolution_is_declaration_order_invariant() {
    let stage_to_season: [i32; 4] = [0, 1, 0, 1];
    let stage_ids = [StageId(0), StageId(1), StageId(2), StageId(3)];
    let stage_block_counts: [usize; 4] = [2, 2, 2, 2];
    let n_stages = 4;
    let ec = EnergyConversionSet::new(vec![], vec![], &[], n_stages);
    let overrides = HydroEnergyProductivityOverride::default();
    let hydros: Vec<cobre_core::Hydro> = Vec::new();

    let order_a = make_params(&[0, 1, 2, 3]);
    let order_b = make_params(&[3, 1, 0, 2]);

    let resolved_a = build_resolved_parameters(
        &order_a,
        &ec,
        &overrides,
        &hydros,
        &stage_to_season,
        &stage_ids,
        &stage_block_counts,
        1_000_000.0,
    )
    .expect("ResolvedParameters builds for order_a");
    let resolved_b = build_resolved_parameters(
        &order_b,
        &ec,
        &overrides,
        &hydros,
        &stage_to_season,
        &stage_ids,
        &stage_block_counts,
        1_000_000.0,
    )
    .expect("ResolvedParameters builds for order_b");

    // Every id resolves bit-identically under both orders, at every (stage, block).
    for id_raw in [10_i32, 20, 30, 40] {
        for (stage_idx, &n_blocks) in stage_block_counts.iter().enumerate() {
            for block_idx in 0..n_blocks {
                let va = resolved_a.get(EntityId(id_raw), stage_idx, block_idx);
                let vb = resolved_b.get(EntityId(id_raw), stage_idx, block_idx);
                assert_eq!(
                    va.to_bits(),
                    vb.to_bits(),
                    "mismatch at id={id_raw}, stage={stage_idx}, block={block_idx}: a={va} b={vb}",
                );
            }
        }
    }

    // The block-varying parameter genuinely carries a different value per block.
    for stage_idx in 0..n_stages {
        assert_ne!(
            resolved_a.get(EntityId(40), stage_idx, 0).to_bits(),
            resolved_a.get(EntityId(40), stage_idx, 1).to_bits(),
            "delta must differ between block 0 and block 1 at stage {stage_idx}",
        );
    }

    assert_eq!(resolved_a.id_to_slot.len(), 4);
    assert_eq!(resolved_b.id_to_slot.len(), 4);

    let keys_a: Vec<i32> = resolved_a.id_to_slot.iter().map(|(k, _)| *k).collect();
    let keys_b: Vec<i32> = resolved_b.id_to_slot.iter().map(|(k, _)| *k).collect();
    assert_eq!(keys_a, vec![10, 20, 30, 40]);
    assert_eq!(keys_b, vec![10, 20, 30, 40]);
}
