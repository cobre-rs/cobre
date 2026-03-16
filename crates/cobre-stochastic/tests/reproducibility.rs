//! Reproducibility and invariance integration tests for `cobre-stochastic`.
//!
//! Verifies four cross-concern invariants that are not covered by individual
//! module unit tests:
//!
//! 1. **Deterministic reproducibility** — identical inputs and seed produce
//!    bit-identical opening trees and `sample_forward` results.
//! 2. **Declaration-order invariance** — reordering hydro entity IDs in the
//!    input produces identical opening trees, because the pipeline sorts
//!    entities by `EntityId` internally.
//! 3. **Seed sensitivity** — different base seeds produce different trees.
//! 4. **Infrastructure genericity** — the crate source contains zero
//!    algorithm-specific references.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp
)]

use std::collections::BTreeMap;

use chrono::NaiveDate;
use cobre_core::{
    Bus, DeficitSegment, EntityId, SystemBuilder,
    entities::hydro::{Hydro, HydroGenerationModel, HydroPenalties},
    scenario::{
        CorrelationEntity, CorrelationGroup, CorrelationModel, CorrelationProfile, InflowModel,
    },
    temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    },
};
use cobre_stochastic::{build_stochastic_context, sample_forward};

fn make_bus(id: i32) -> Bus {
    Bus {
        id: EntityId(id),
        name: format!("Bus{id}"),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 1000.0,
        }],
        excess_cost: 0.0,
    }
}

fn make_stage(index: usize, id: i32, branching_factor: usize) -> Stage {
    Stage {
        index,
        id,
        start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
        season_id: Some(0),
        blocks: vec![Block {
            index: 0,
            name: "SINGLE".to_string(),
            duration_hours: 744.0,
        }],
        block_mode: BlockMode::Parallel,
        state_config: StageStateConfig {
            storage: true,
            inflow_lags: false,
        },
        risk_config: StageRiskConfig::Expectation,
        scenario_config: ScenarioSourceConfig {
            branching_factor,
            noise_method: NoiseMethod::Saa,
        },
    }
}

fn make_hydro(id: i32) -> Hydro {
    Hydro {
        id: EntityId(id),
        name: format!("H{id}"),
        bus_id: EntityId(0),
        downstream_id: None,
        entry_stage_id: None,
        exit_stage_id: None,
        min_storage_hm3: 0.0,
        max_storage_hm3: 100.0,
        min_outflow_m3s: 0.0,
        max_outflow_m3s: None,
        generation_model: HydroGenerationModel::ConstantProductivity {
            productivity_mw_per_m3s: 1.0,
        },
        min_turbined_m3s: 0.0,
        max_turbined_m3s: 100.0,
        min_generation_mw: 0.0,
        max_generation_mw: 100.0,
        tailrace: None,
        hydraulic_losses: None,
        efficiency: None,
        evaporation_coefficients_mm: None,
        evaporation_reference_volumes_hm3: None,
        diversion: None,
        filling: None,
        penalties: HydroPenalties {
            spillage_cost: 0.0,
            diversion_cost: 0.0,
            fpha_turbined_cost: 0.0,
            storage_violation_below_cost: 0.0,
            filling_target_violation_cost: 0.0,
            turbined_violation_below_cost: 0.0,
            outflow_violation_below_cost: 0.0,
            outflow_violation_above_cost: 0.0,
            generation_violation_below_cost: 0.0,
            evaporation_violation_cost: 0.0,
            water_withdrawal_violation_cost: 0.0,
        },
    }
}

fn make_inflow_model(hydro_id: i32, stage_id: i32) -> InflowModel {
    InflowModel {
        hydro_id: EntityId(hydro_id),
        stage_id,
        mean_m3s: 100.0,
        std_m3s: 30.0,
        ar_coefficients: vec![],
        residual_std_ratio: 1.0,
    }
}

fn identity_correlation(entity_ids: &[i32]) -> CorrelationModel {
    let n = entity_ids.len();
    let matrix: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
        .collect();
    let mut profiles = BTreeMap::new();
    profiles.insert(
        "default".to_string(),
        CorrelationProfile {
            groups: vec![CorrelationGroup {
                name: "g1".to_string(),
                entities: entity_ids
                    .iter()
                    .map(|&id| CorrelationEntity {
                        entity_type: "inflow".to_string(),
                        id: EntityId(id),
                    })
                    .collect(),
                matrix,
            }],
        },
    );
    CorrelationModel {
        method: "cholesky".to_string(),
        profiles,
        schedule: vec![],
    }
}

fn build_fixture(hydros: Vec<Hydro>, base_seed: u64) -> cobre_stochastic::StochasticContext {
    let stages = vec![
        make_stage(0, 0, 5),
        make_stage(1, 1, 5),
        make_stage(2, 2, 5),
    ];
    let inflow_models = vec![
        make_inflow_model(1, 0),
        make_inflow_model(1, 1),
        make_inflow_model(1, 2),
        make_inflow_model(2, 0),
        make_inflow_model(2, 1),
        make_inflow_model(2, 2),
    ];

    let system = SystemBuilder::new()
        .buses(vec![make_bus(0)])
        .hydros(hydros)
        .stages(stages)
        .inflow_models(inflow_models)
        .correlation(identity_correlation(&[1, 2]))
        .build()
        .expect("build_fixture: system build must succeed");

    build_stochastic_context(&system, base_seed, &[], None)
        .expect("build_fixture: build_stochastic_context must succeed")
}

#[test]
fn deterministic_reproducibility() {
    let hydros = vec![make_hydro(1), make_hydro(2)];

    let ctx_a = build_fixture(hydros.clone(), 42);
    let ctx_b = build_fixture(hydros, 42);

    let tree_a = ctx_a.opening_tree();
    let tree_b = ctx_b.opening_tree();

    assert_eq!(
        tree_a.n_stages(),
        tree_b.n_stages(),
        "n_stages must match between two identical builds"
    );

    for stage in 0..tree_a.n_stages() {
        assert_eq!(
            tree_a.n_openings(stage),
            tree_b.n_openings(stage),
            "n_openings at stage={stage} must match"
        );
        for opening in 0..tree_a.n_openings(stage) {
            assert_eq!(
                tree_a.opening(stage, opening),
                tree_b.opening(stage, opening),
                "opening values at stage={stage} opening={opening} must be bit-identical"
            );
        }
    }

    let view_a = ctx_a.tree_view();
    let view_b = ctx_b.tree_view();
    let seed_a = ctx_a.base_seed();
    let seed_b = ctx_b.base_seed();

    for iteration in 0_u32..3 {
        for scenario in 0_u32..5 {
            for (stage_idx, stage_domain_id) in [(0usize, 0u32), (1, 1), (2, 2)] {
                let (idx_a, slice_a) = sample_forward(
                    &view_a,
                    seed_a,
                    iteration,
                    scenario,
                    stage_domain_id,
                    stage_idx,
                );
                let (idx_b, slice_b) = sample_forward(
                    &view_b,
                    seed_b,
                    iteration,
                    scenario,
                    stage_domain_id,
                    stage_idx,
                );

                assert_eq!(
                    idx_a, idx_b,
                    "sample_forward index must be identical for iteration={iteration} \
                     scenario={scenario} stage_idx={stage_idx}"
                );
                assert_eq!(
                    slice_a, slice_b,
                    "sample_forward slice must be bit-identical for iteration={iteration} \
                     scenario={scenario} stage_idx={stage_idx}"
                );
            }
        }
    }
}

/// Reversing the hydro entity list in the input produces an identical opening
/// tree, because `SystemBuilder` sorts hydros by `EntityId` internally.
#[test]
fn declaration_order_invariance() {
    // Forward order: [EntityId(1), EntityId(2)]
    let hydros_forward = vec![make_hydro(1), make_hydro(2)];
    // Reversed order: [EntityId(2), EntityId(1)]
    let hydros_reversed = vec![make_hydro(2), make_hydro(1)];

    let ctx_forward = build_fixture(hydros_forward, 42);
    let ctx_reversed = build_fixture(hydros_reversed, 42);

    let tree_fwd = ctx_forward.opening_tree();
    let tree_rev = ctx_reversed.opening_tree();

    assert_eq!(
        tree_fwd.n_stages(),
        tree_rev.n_stages(),
        "n_stages must match regardless of hydro insertion order"
    );

    for stage in 0..tree_fwd.n_stages() {
        assert_eq!(
            tree_fwd.n_openings(stage),
            tree_rev.n_openings(stage),
            "n_openings at stage={stage} must match regardless of hydro insertion order"
        );
        for opening in 0..tree_fwd.n_openings(stage) {
            assert_eq!(
                tree_fwd.opening(stage, opening),
                tree_rev.opening(stage, opening),
                "opening values at stage={stage} opening={opening} must be identical \
                 regardless of hydro insertion order"
            );
        }
    }
}

/// Different base seeds produce at least one differing noise value in the
/// opening tree.
#[test]
fn seed_sensitivity() {
    let hydros = vec![make_hydro(1), make_hydro(2)];

    let ctx_42 = build_fixture(hydros.clone(), 42);
    let ctx_99 = build_fixture(hydros, 99);

    let tree_42 = ctx_42.opening_tree();
    let tree_99 = ctx_99.opening_tree();

    let any_differ = (0..tree_42.n_stages()).any(|stage| {
        (0..tree_42.n_openings(stage)).any(|opening| {
            tree_42
                .opening(stage, opening)
                .iter()
                .zip(tree_99.opening(stage, opening).iter())
                .any(|(a, b)| a != b)
        })
    });

    assert!(
        any_differ,
        "expected at least one differing noise value between seed=42 and seed=99 trees"
    );
}

/// The crate source contains zero algorithm-specific references.
#[test]
fn infrastructure_genericity_no_sddp_references() {
    use std::path::Path;
    use std::process::Command;

    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let workspace_root = Path::new(manifest_dir).join("..").join("..");
    let src_path = Path::new(manifest_dir).join("src");

    let output = Command::new("grep")
        .args(["-riE", "sddp"])
        .arg(&src_path)
        .current_dir(&workspace_root)
        .output()
        .expect("infrastructure_genericity: failed to execute grep");

    assert_eq!(
        output.status.code(),
        Some(1),
        "grep found algorithm-specific references in cobre-stochastic/src/:\n{}",
        String::from_utf8_lossy(&output.stdout)
    );
}
