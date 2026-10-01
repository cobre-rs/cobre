//! Gates every stored Benders cut with a validity, tightness, and mask-soundness oracle.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp
)]

mod common;

use cobre_core::System;
use cobre_io::Config;
use cobre_sddp::StudySetup;
use cobre_sddp::test_support::decks::committed_decks;

use common::cut_oracles::{apply_oracle_config, run_cut_oracles};
use common::in_code_studies::{
    ChronologicalNoiseSpec, chronological_noise_study, discounted_anticipated_study,
    mixed_lead_anticipated_study, parallel_multiblock_evaporation_study, stochastic_parallel_study,
};
use common::{build_setup_in_code, build_setup_in_code_with_models, fresh_setup_with};

fn oracle_setup(system: System, mut config: Config) -> StudySetup {
    apply_oracle_config(&mut config);
    build_setup_in_code(system, &config)
}

#[test]
fn cut_oracles_hold_on_the_discounted_anticipated_study() {
    let (system, config) = discounted_anticipated_study();
    let setup = oracle_setup(system, config);
    run_cut_oracles("discounted_anticipated_study", setup)
        .assert_sound("discounted_anticipated_study");
}

#[test]
fn cut_oracles_hold_on_the_parallel_multiblock_evaporation_study() {
    let (system, mut config, hydro_models) = parallel_multiblock_evaporation_study();
    apply_oracle_config(&mut config);
    let setup = build_setup_in_code_with_models(system, &config, hydro_models);
    run_cut_oracles("parallel_multiblock_evaporation_study", setup)
        .assert_sound("parallel_multiblock_evaporation_study");
}

#[test]
fn cut_oracles_hold_on_the_stochastic_parallel_study() {
    let (system, config) = stochastic_parallel_study();
    let setup = oracle_setup(system, config);
    run_cut_oracles("stochastic_parallel_study", setup).assert_sound("stochastic_parallel_study");
}

#[test]
fn cut_oracles_hold_on_the_chronological_noise_study() {
    let (system, config) = chronological_noise_study(&ChronologicalNoiseSpec::default());
    let setup = oracle_setup(system, config);
    run_cut_oracles("chronological_noise_study", setup).assert_sound("chronological_noise_study");
}

#[test]
fn cut_oracles_hold_on_the_mixed_lead_anticipated_study() {
    let (system, config) = mixed_lead_anticipated_study(false);
    let setup = oracle_setup(system, config);
    run_cut_oracles("mixed_lead_anticipated_study", setup)
        .assert_sound("mixed_lead_anticipated_study");
}

#[test]
fn cut_oracles_hold_on_the_chronological_pumping_study() {
    let spec = ChronologicalNoiseSpec {
        pumping_station: true,
        ..ChronologicalNoiseSpec::default()
    };
    let (system, config) = chronological_noise_study(&spec);
    let setup = oracle_setup(system, config);
    run_cut_oracles("chronological_pumping_study", setup)
        .assert_sound("chronological_pumping_study");
}

#[test]
fn cut_oracles_hold_on_the_chronological_and_parallel_storage_decks() {
    let decks = committed_decks();
    for key in [
        "crates/cobre-sddp/tests/fixtures/chronological_storage",
        "crates/cobre-sddp/tests/fixtures/parallel_storage",
    ] {
        let deck = decks
            .iter()
            .find(|deck| deck.key == key)
            .unwrap_or_else(|| panic!("committed deck {key} is absent"));
        let setup = fresh_setup_with(&deck.dir, apply_oracle_config);
        run_cut_oracles(&deck.key, setup).assert_sound(&deck.key);
    }
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "trains every committed deck that carries inflow-lag or anticipated state"
)]
fn cut_oracles_hold_on_every_committed_deck_with_lag_or_anticipated_state() {
    let mut selected: Vec<String> = Vec::new();
    let mut any_max_par_order = false;

    for deck in committed_decks() {
        let setup = fresh_setup_with(&deck.dir, apply_oracle_config);
        let (has_anticipated, has_lag) = {
            let state = setup.stage_state();
            (state.n_anticipated > 0, state.max_par_order > 0)
        };
        if !has_anticipated && !has_lag {
            continue;
        }
        if has_lag {
            any_max_par_order = true;
        }

        let report = run_cut_oracles(&deck.key, setup);
        eprintln!(
            "{}: mask drops at least one dimension = {}",
            deck.key,
            report.off_mask_slots_checked > 0
        );
        report.assert_sound(&deck.key);
        selected.push(deck.key.clone());
    }

    eprintln!("selected decks: {selected:?}");
    for required in [
        "examples/deterministic/d34-anticipated-varying-blocks",
        "examples/deterministic/d37-anticipated-commissioning",
        "examples/deterministic/d55-post-study-anticipated-lanes",
    ] {
        assert!(
            selected.iter().any(|k| k == required),
            "deck {required} must be selected by the anticipated/lag criterion"
        );
    }
    assert!(
        any_max_par_order,
        "at least one selected deck must have max_par_order > 0"
    );
}
