//! The full-FCF load entry: directory lookup, check and apply against a
//! checkpoint written by the production writer.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp
)]

use std::path::{Path, PathBuf};

use cobre_core::System;
use cobre_core::scenario::ScenarioSource;
use cobre_io::config::StoppingRuleConfig;
use cobre_io::read_policy_checkpoint;
use cobre_sddp::StudySetup;
use cobre_sddp::hydro_models::prepare_hydro_models;
use cobre_sddp::policy::full_fcf_load::{
    FullFcfLoadError, FullFcfLoadKind, check_full_fcf_load, locate_policy_dir,
};
use cobre_sddp::policy::orchestration::{CheckpointParams, write_checkpoint};
use cobre_sddp::setup::prepare_stochastic;
use cobre_solver::ActiveSolver;
use tempfile::TempDir;

mod common;
use common::StubComm;

const ITERATIONS: u32 = 3;

fn d01_case_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("examples/deterministic/d01-thermal-dispatch")
}

fn build_setup() -> (StudySetup, System) {
    let case_dir = d01_case_dir();
    let mut config = cobre_io::parse_config(&case_dir.join("config.json")).expect("config");
    config.training.stopping_rules = Some(vec![StoppingRuleConfig::IterationLimit {
        limit: ITERATIONS,
    }]);
    let system = cobre_io::load_case(&case_dir).expect("load_case");
    let prep = prepare_stochastic(
        system,
        &case_dir,
        &config,
        42,
        &ScenarioSource::default(),
        None,
    )
    .expect("prepare_stochastic");
    let hydro_models =
        prepare_hydro_models(&prep.system, &case_dir, false).expect("prepare_hydro_models");
    let setup = StudySetup::new(
        &prep.system,
        &config,
        prep.stochastic,
        hydro_models,
        Vec::new(),
    )
    .expect("StudySetup::new");
    (setup, prep.system)
}

struct Trained {
    output: TempDir,
    completed_iterations: u64,
    final_lower_bound: f64,
}

fn train_and_write_checkpoint() -> Trained {
    let (mut setup, system) = build_setup();
    let mut solver = ActiveSolver::new().expect("solver");
    let outcome = setup
        .train(&mut solver, &StubComm, 1, ActiveSolver::new, None, None)
        .expect("train");
    assert!(outcome.error.is_none());

    let output = TempDir::new().expect("tempdir");
    let policy_dir = output.path().join(&setup.policy_path);
    std::fs::create_dir_all(&policy_dir).expect("policy dir");
    let params = CheckpointParams {
        max_iterations: setup.loop_params.max_iterations,
        forward_passes: setup.loop_params.forward_passes,
        seed: setup.loop_params.seed,
        export_states: false,
    };
    write_checkpoint(&policy_dir, &setup, &system, &outcome.result, &params)
        .expect("write_checkpoint");

    let producer = read_policy_checkpoint(&policy_dir)
        .expect("read checkpoint")
        .metadata
        .producer;
    Trained {
        output,
        completed_iterations: u64::from(producer.completed_iterations),
        final_lower_bound: producer.final_lower_bound,
    }
}

#[test]
fn missing_policy_directory_names_the_directory_and_what_the_load_needs() {
    let (setup, _system) = build_setup();
    let output = TempDir::new().expect("tempdir");
    let dir = output.path().join(&setup.policy_path);
    for (kind, sentence) in [
        (
            FullFcfLoadKind::WarmStart,
            "Cannot warm-start without a prior policy.",
        ),
        (
            FullFcfLoadKind::Resume,
            "Cannot resume without a prior checkpoint.",
        ),
        (
            FullFcfLoadKind::SimulationOnly,
            "Cannot run simulation-only mode without a trained policy.",
        ),
    ] {
        let err = locate_policy_dir(kind, output.path(), &setup).expect_err("no directory");
        assert!(matches!(
            err,
            FullFcfLoadError::MissingPolicyDirectory { .. }
        ));
        assert_eq!(
            err.to_string(),
            format!("Policy directory not found: {}. {sentence}", dir.display())
        );
    }
}

#[test]
fn resume_apply_installs_the_checkpoint_cuts_and_the_start_iteration() {
    let trained = train_and_write_checkpoint();
    for kind in [FullFcfLoadKind::Resume, FullFcfLoadKind::WarmStart] {
        let (mut setup, system) = build_setup();
        assert_eq!(setup.fcf.total_active_cuts(), 0);
        let policy_dir = locate_policy_dir(kind, trained.output.path(), &setup).expect("locate");
        let checked =
            check_full_fcf_load(kind, &policy_dir, &system, &setup, &mut |_| {}).expect("check");
        assert_eq!(checked.completed_iterations(), trained.completed_iterations);
        checked.apply_to_training(&mut setup);
        let loaded_cuts = setup.fcf.total_active_cuts();
        assert!(loaded_cuts > 0);
        let mut solver = ActiveSolver::new().expect("solver");
        let outcome = setup
            .train(&mut solver, &StubComm, 1, ActiveSolver::new, None, None)
            .expect("train");
        assert!(outcome.error.is_none());
        // Resume starts past the iteration limit, so it adds no cuts.
        assert_eq!(
            setup.fcf.total_active_cuts() == loaded_cuts,
            kind == FullFcfLoadKind::Resume
        );
    }
}

#[test]
fn simulation_only_check_returns_the_recorded_bounds_without_frozen_templates() {
    let trained = train_and_write_checkpoint();
    let (setup, system) = build_setup();
    let kind = FullFcfLoadKind::SimulationOnly;
    let policy_dir = locate_policy_dir(kind, trained.output.path(), &setup).expect("locate");
    let checked =
        check_full_fcf_load(kind, &policy_dir, &system, &setup, &mut |_| {}).expect("check");
    let (fcf, result) = checked.into_simulation_policy();
    assert_eq!(result.iterations, trained.completed_iterations);
    assert_eq!(result.final_lb, trained.final_lower_bound);
    assert!(result.frozen_templates.is_none());
    assert_eq!(fcf.state_dimension, setup.fcf.state_dimension);
}

fn relative_listing(root: &Path) -> Vec<PathBuf> {
    let mut listing = Vec::new();
    let mut pending = vec![root.to_path_buf()];
    while let Some(dir) = pending.pop() {
        for entry in std::fs::read_dir(&dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                pending.push(path.clone());
            }
            listing.push(path.strip_prefix(root).unwrap().to_path_buf());
        }
    }
    listing.sort();
    listing
}

#[test]
fn locate_policy_dir_accepts_a_policy_caught_between_renames() {
    let trained = train_and_write_checkpoint();
    let (setup, system) = build_setup();
    let output_dir = trained.output.path();
    let policy_dir = output_dir.join(&setup.policy_path);
    let mut staging = policy_dir.clone().into_os_string();
    staging.push(".staging");
    std::fs::rename(&policy_dir, &staging).unwrap();
    let before = relative_listing(output_dir);

    let kind = FullFcfLoadKind::SimulationOnly;
    let located = locate_policy_dir(kind, output_dir, &setup).expect("locate");
    assert_eq!(located, policy_dir);
    let checked = check_full_fcf_load(kind, &located, &system, &setup, &mut |_| {}).expect("check");
    let (_fcf, result) = checked.into_simulation_policy();
    assert_eq!(result.iterations, trained.completed_iterations);
    assert_eq!(relative_listing(output_dir), before);
}
