//! The policy checkpoint's record of the lower bound across training iterations.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::path::Path;

use cobre_io::config::StoppingRuleConfig;
use cobre_io::output::policy::read_policy_checkpoint;
use cobre_sddp::policy::orchestration::{CheckpointParams, write_checkpoint};
use cobre_solver::ActiveSolver;

use common::{StubComm, fresh_system_and_setup_with};

#[test]
fn checkpoint_records_the_lower_bound_of_every_completed_iteration() {
    let case_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples/1dtoy");
    let (system, mut setup) = fresh_system_and_setup_with(&case_dir, |config| {
        config.training.stopping_rules =
            Some(vec![StoppingRuleConfig::IterationLimit { limit: 4 }]);
    });

    let mut solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");
    let outcome = setup
        .train(&mut solver, &StubComm, 1, ActiveSolver::new, None, None)
        .expect("train must return Ok");
    assert!(
        outcome.error.is_none(),
        "training error: {:?}",
        outcome.error
    );
    let result = outcome.result;
    assert_eq!(result.iterations, 4);

    let tmpdir = tempfile::tempdir().expect("tempdir");
    let policy_dir = tmpdir.path().join("policy");
    write_checkpoint(
        &policy_dir,
        &setup,
        &system,
        &result,
        &CheckpointParams {
            max_iterations: setup.loop_params.max_iterations,
            forward_passes: setup.loop_params.forward_passes,
            seed: setup.loop_params.seed,
            export_states: false,
        },
    )
    .expect("write_checkpoint must succeed");

    let checkpoint = read_policy_checkpoint(&policy_dir).expect("read_policy_checkpoint");
    let producer = &checkpoint.metadata.producer;
    let recorded: Vec<u64> = producer
        .lower_bound_history
        .iter()
        .copied()
        .map(f64::to_bits)
        .collect();
    let trained: Vec<u64> = result
        .lower_bound_history
        .iter()
        .copied()
        .map(f64::to_bits)
        .collect();

    assert_eq!(recorded.len(), 4);
    assert_eq!(recorded, trained);
    assert_eq!(
        recorded.last().copied(),
        Some(producer.final_lower_bound.to_bits())
    );
}
