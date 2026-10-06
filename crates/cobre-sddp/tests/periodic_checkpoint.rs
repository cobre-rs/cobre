//! Checkpoints written by the training loop on the `policy.checkpointing`
//! schedule.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::path::{Path, PathBuf};
use std::sync::mpsc;

use cobre_core::{System, TrainingEvent};
use cobre_io::config::{CheckpointingConfig, StoppingRuleConfig};
use cobre_io::output::policy::{PolicyCheckpoint, read_policy_checkpoint};
use cobre_sddp::policy::orchestration::{CheckpointParams, write_checkpoint};
use cobre_sddp::{SddpError, StudySetup, TrainingResult};
use cobre_solver::ActiveSolver;

use common::{StubComm, fresh_system_and_setup_with};

fn toy_case() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples/1dtoy")
}

fn checkpointing_setup(
    iteration_limit: u32,
    initial_iteration: Option<u32>,
    interval_iterations: u32,
) -> (System, StudySetup) {
    fresh_system_and_setup_with(&toy_case(), |config| {
        config.training.stopping_rules = Some(vec![StoppingRuleConfig::IterationLimit {
            limit: iteration_limit,
        }]);
        config.exports.states = true;
        config.policy.checkpointing = CheckpointingConfig {
            enabled: Some(true),
            initial_iteration,
            interval_iterations: Some(interval_iterations),
            ..CheckpointingConfig::default()
        };
    })
}

fn train(
    setup: &mut StudySetup,
    event_sender: Option<mpsc::Sender<TrainingEvent>>,
) -> TrainingResult {
    let mut solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");
    let outcome = setup
        .train(
            &mut solver,
            &StubComm,
            1,
            ActiveSolver::new,
            event_sender,
            None,
        )
        .expect("train must return Ok");
    assert!(
        outcome.error.is_none(),
        "training error: {:?}",
        outcome.error
    );
    outcome.result
}

/// The N-run (`iteration_limit: 6`) leaves only its iteration-3 periodic
/// checkpoint in its policy directory; the k-run (`iteration_limit: 3`) is
/// written by the final-checkpoint path.
struct PeriodicAndFinal {
    periodic: PolicyCheckpoint,
    final_at_k: PolicyCheckpoint,
    n_run_pool_capacities: Vec<u32>,
}

fn periodic_at_3_and_final_at_3() -> PeriodicAndFinal {
    let n_out = tempfile::tempdir().expect("tempdir");
    let (system, mut n_setup) = checkpointing_setup(6, Some(3), 6);
    let n_run_pool_capacities = n_setup
        .fcf
        .pools
        .iter()
        .map(|p| u32::try_from(p.capacity).unwrap())
        .collect();
    n_setup.enable_periodic_checkpoints(&system, n_out.path());
    assert_eq!(train(&mut n_setup, None).iterations, 6);
    let periodic = read_policy_checkpoint(&n_out.path().join(&n_setup.policy_path))
        .expect("the periodic checkpoint must be readable");

    let (k_system, mut k_setup) = checkpointing_setup(3, Some(3), 6);
    let k_result = train(&mut k_setup, None);
    assert_eq!(k_result.iterations, 3);
    let k_out = tempfile::tempdir().expect("tempdir");
    let k_policy_dir = k_out.path().join(&k_setup.policy_path);
    write_checkpoint(
        &k_policy_dir,
        &k_setup,
        &k_system,
        &k_result,
        &CheckpointParams {
            max_iterations: k_setup.loop_params.max_iterations,
            forward_passes: k_setup.loop_params.forward_passes,
            seed: k_setup.loop_params.seed,
            export_states: true,
        },
    )
    .expect("write_checkpoint must succeed");
    let final_at_k = read_policy_checkpoint(&k_policy_dir).expect("read_policy_checkpoint");

    PeriodicAndFinal {
        periodic,
        final_at_k,
        n_run_pool_capacities,
    }
}

#[test]
fn periodic_checkpoint_equals_the_final_checkpoint_of_a_run_stopped_at_that_iteration() {
    let PeriodicAndFinal {
        periodic,
        final_at_k,
        ..
    } = periodic_at_3_and_final_at_3();
    assert_eq!(periodic.metadata.producer.completed_iterations, 3);
    assert_eq!(periodic.stage_cuts.len(), final_at_k.stage_cuts.len());
    assert!(
        !periodic.stage_states.is_empty(),
        "the case must export visited states"
    );

    let mut aligned = periodic.clone();
    aligned
        .metadata
        .created_at
        .clone_from(&final_at_k.metadata.created_at);
    aligned.metadata.producer.max_iterations = final_at_k.metadata.producer.max_iterations;
    for (pool, final_pool) in aligned.stage_cuts.iter_mut().zip(&final_at_k.stage_cuts) {
        pool.capacity = final_pool.capacity;
    }

    assert_eq!(format!("{aligned:?}"), format!("{final_at_k:?}"));
}

#[test]
fn periodic_checkpoint_records_the_running_configuration_as_provenance() {
    let PeriodicAndFinal {
        periodic,
        final_at_k,
        n_run_pool_capacities,
    } = periodic_at_3_and_final_at_3();

    assert_eq!(periodic.metadata.producer.max_iterations, 6);
    assert_eq!(final_at_k.metadata.producer.max_iterations, 3);

    let periodic_capacities: Vec<u32> = periodic.stage_cuts.iter().map(|p| p.capacity).collect();
    assert_eq!(periodic_capacities, n_run_pool_capacities);
    assert!(
        periodic
            .stage_cuts
            .iter()
            .zip(&final_at_k.stage_cuts)
            .any(|(p, k)| p.capacity != k.capacity),
        "the iteration limit must size at least one pool differently"
    );
}

#[test]
fn periodic_checkpoints_follow_the_schedule_and_skip_the_stop_iteration() {
    let out = tempfile::tempdir().expect("tempdir");
    let (system, mut setup) = checkpointing_setup(6, Some(2), 2);
    setup.enable_periodic_checkpoints(&system, out.path());
    let policy_dir = out.path().join(&setup.policy_path);

    let (tx, rx) = mpsc::channel();
    assert_eq!(train(&mut setup, Some(tx)).iterations, 6);

    let written: Vec<(u64, String)> = rx
        .try_iter()
        .filter_map(|event| match event {
            TrainingEvent::CheckpointComplete {
                iteration,
                checkpoint_path,
                ..
            } => Some((iteration, checkpoint_path)),
            _ => None,
        })
        .collect();
    let expected_path = policy_dir.display().to_string();
    assert_eq!(
        written,
        vec![(2, expected_path.clone()), (4, expected_path)]
    );

    let latest = read_policy_checkpoint(&policy_dir).expect("read_policy_checkpoint");
    assert_eq!(latest.metadata.producer.completed_iterations, 4);
}

#[test]
fn failed_periodic_write_ends_training_with_the_failing_iteration() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let blocker = tmp.path().join("blocker");
    let (system, mut setup) = checkpointing_setup(6, None, 2);
    setup.enable_periodic_checkpoints(&system, &blocker);
    std::fs::write(&blocker, b"a regular file, not a directory").expect("write blocker");

    let mut solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");
    let outcome = setup
        .train(&mut solver, &StubComm, 1, ActiveSolver::new, None, None)
        .expect("train must return Ok");

    assert!(
        matches!(
            outcome.error,
            Some(SddpError::CheckpointWrite { iteration: 2, .. })
        ),
        "{:?}",
        outcome.error
    );
    assert_eq!(outcome.result.iterations, 2);
}
