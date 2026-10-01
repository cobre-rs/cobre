//! Guard: pins the noise-vector segment layout — hydro, stochastic-load, and
//! NCS — to the LP entity sets it must equal, over every committed deck and
//! an in-code fixture exercising all three segments non-trivially.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use cobre_core::{EntityId, System};
use cobre_sddp::StudySetup;
use cobre_sddp::hydro_models::PrepareHydroModelsResult;
use cobre_sddp::test_support::decks::committed_decks;

fn assert_noise_segments_match_lp(label: &str, system: &System, setup: &StudySetup) {
    let stochastic = &setup.inputs.stochastic;
    let (n_h, n_l) = (stochastic.n_hydros(), stochastic.n_load_buses());
    let order = stochastic.entity_order();
    let hydro_ids: Vec<EntityId> = system.hydros().iter().map(|h| h.id).collect();
    let ctx = setup.stage_ctx();
    let lp_load_ids: Vec<EntityId> = ctx
        .load_bus_indices
        .iter()
        .map(|&b| system.buses()[b].id)
        .collect();
    assert_eq!(
        &order[..n_h],
        hydro_ids.as_slice(),
        "{label}: hydro segment"
    );
    assert_eq!(
        &order[n_h..n_h + n_l],
        lp_load_ids.as_slice(),
        "{label}: load segment"
    );
    assert_eq!(
        &order[n_h + n_l..],
        stochastic.ncs_entity_ids(),
        "{label}: NCS segment"
    );
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "builds a study setup for every committed deck"
)]
fn noise_segments_match_lp_entity_sets_on_committed_decks() {
    for deck in committed_decks() {
        let system = cobre_io::load_case(&deck.dir).expect("load_case must succeed");
        let setup = common::fresh_setup_with(&deck.dir, |_| {});
        assert_noise_segments_match_lp(&deck.key, &system, &setup);
    }
}

#[test]
fn noise_segments_match_lp_entity_sets_with_stochastic_load_and_ncs() {
    let (system, config) = common::in_code_studies::stochastic_parallel_study();
    let stochastic = common::stochastic_in_code(&system);
    let hydro_models = PrepareHydroModelsResult::default_from_system(&system);
    let setup = StudySetup::new(&system, &config, stochastic, hydro_models, Vec::new())
        .expect("StudySetup::new");

    assert!(
        setup.inputs.stochastic.n_hydros() > 0,
        "stochastic_parallel_study must have stochastic hydro noise"
    );
    assert!(
        setup.inputs.stochastic.n_load_buses() > 0,
        "stochastic_parallel_study must have stochastic load noise"
    );
    assert!(
        setup.inputs.stochastic.n_stochastic_ncs() > 0,
        "stochastic_parallel_study must have stochastic NCS"
    );

    assert_noise_segments_match_lp("in-code/stochastic-parallel", &system, &setup);
}
