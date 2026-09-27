//! Golden-value regression guard for SAA opening tree generation: any change to
//! seed derivation, RNG, or loop order that alters the SAA output for the pinned
//! `base_seed = 42` configuration breaks the constants below.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp
)]

use cobre_core::{EntityId, NoiseMethod, ScenarioSourceConfig, Stage};
use cobre_stochastic::{ClassDimensions, generate_opening_tree, tree::OpeningTreeGenerationInputs};

mod common;
use common::{StageSpec, identity_correlation};

/// The blockless SAA stage this golden regression was produced from:
/// `generate_opening_tree` reads the branching factor, not the load blocks,
/// but a blocked stage is a different fixture and would move the constants
/// below.
fn make_stage(index: usize, id: i32, branching_factor: usize) -> Stage {
    common::make_stage(StageSpec {
        id,
        index: Some(index),
        season_id: Some(0),
        blocks: Vec::new(),
        scenario_config: ScenarioSourceConfig {
            branching_factor,
            noise_method: NoiseMethod::Saa,
        },
        ..Default::default()
    })
}

const GOLDEN_S0_O0_D0: f64 = 4.009_893_649_649_564_6e-1;
const GOLDEN_S0_O0_D1: f64 = 2.279_255_881_585_980_4e-1;
const GOLDEN_S0_O1_D0: f64 = -1.395_412_177_608_524_4;
const GOLDEN_S0_O1_D1: f64 = -2.693_936_692_173_674_6e-1;
const GOLDEN_S0_O2_D0: f64 = 8.337_031_709_056_368e-1;
const GOLDEN_S0_O2_D1: f64 = -1.619_991_803_182_488_7;

#[test]
fn saa_golden_value_regression() {
    let stages = vec![
        make_stage(0, 0, 3),
        make_stage(1, 1, 3),
        make_stage(2, 2, 3),
    ];
    let corr = identity_correlation(&[1, 2]);
    let entity_order = vec![EntityId(1), EntityId(2)];

    let dims = ClassDimensions {
        n_hydros: 2,
        n_load_buses: 0,
        n_ncs: 0,
    };
    let tree = generate_opening_tree(
        42,
        &stages,
        &corr,
        &entity_order,
        dims,
        &OpeningTreeGenerationInputs::default(),
    )
    .unwrap();

    assert_eq!(
        tree.opening(0, 0)[0],
        GOLDEN_S0_O0_D0,
        "stage=0 opening=0 dim=0"
    );
    assert_eq!(
        tree.opening(0, 0)[1],
        GOLDEN_S0_O0_D1,
        "stage=0 opening=0 dim=1"
    );
    assert_eq!(
        tree.opening(0, 1)[0],
        GOLDEN_S0_O1_D0,
        "stage=0 opening=1 dim=0"
    );
    assert_eq!(
        tree.opening(0, 1)[1],
        GOLDEN_S0_O1_D1,
        "stage=0 opening=1 dim=1"
    );
    assert_eq!(
        tree.opening(0, 2)[0],
        GOLDEN_S0_O2_D0,
        "stage=0 opening=2 dim=0"
    );
    assert_eq!(
        tree.opening(0, 2)[1],
        GOLDEN_S0_O2_D1,
        "stage=0 opening=2 dim=1"
    );
}
