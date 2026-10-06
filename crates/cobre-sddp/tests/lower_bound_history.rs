//! The policy checkpoint's record of the lower bound across training iterations.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::any::TypeId;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use cobre_comm::{CommData, CommError, Communicator, ReduceOp};
use cobre_core::System;
use cobre_io::config::StoppingRuleConfig;
use cobre_io::output::policy::read_policy_checkpoint;
use cobre_sddp::policy::full_fcf_load::{FullFcfLoadKind, check_full_fcf_load, locate_policy_dir};
use cobre_sddp::policy::orchestration::{CheckpointParams, write_checkpoint};
use cobre_sddp::{SddpError, StopMask, StudySetup, TrainingOutcome, TrainingResult};
use cobre_solver::ActiveSolver;
use tempfile::TempDir;

use common::{StubComm, fresh_system_and_setup_with};

fn case_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples/1dtoy")
}

fn setup_under(rules: Vec<StoppingRuleConfig>) -> (System, StudySetup) {
    fresh_system_and_setup_with(&case_dir(), |config| {
        config.training.stopping_rules = Some(rules);
    })
}

fn train_with<C: Communicator>(setup: &mut StudySetup, comm: &C) -> TrainingOutcome {
    let mut solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");
    setup
        .train(&mut solver, comm, 1, ActiveSolver::new, None, None)
        .expect("train must return an outcome")
}

fn train_to_completion(setup: &mut StudySetup) -> TrainingResult {
    let outcome = train_with(setup, &StubComm);
    assert!(
        outcome.error.is_none(),
        "training error: {:?}",
        outcome.error
    );
    outcome.result
}

fn bits(series: &[f64]) -> Vec<u64> {
    series.iter().copied().map(f64::to_bits).collect()
}

/// Train `iterations` iterations and write the checkpoint under the study's
/// policy path; returns the output directory and the series the run recorded.
fn checkpoint_after(iterations: u32) -> (TempDir, Vec<f64>) {
    let (system, mut setup) = setup_under(vec![StoppingRuleConfig::IterationLimit {
        limit: iterations,
    }]);
    let result = train_to_completion(&mut setup);
    assert_eq!(result.iterations, u64::from(iterations));

    let output = TempDir::new().expect("tempdir");
    let policy_dir = output.path().join(&setup.policy_path);
    std::fs::create_dir_all(&policy_dir).expect("policy dir");
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
    (output, result.lower_bound_history)
}

fn resumed_setup(output: &TempDir, rules: Vec<StoppingRuleConfig>) -> StudySetup {
    let (system, mut setup) = setup_under(rules);
    let kind = FullFcfLoadKind::Resume;
    let policy_dir = locate_policy_dir(kind, output.path(), &setup).expect("locate");
    check_full_fcf_load(kind, &policy_dir, &system, &setup, &mut |_| {})
        .expect("check")
        .apply_to_training(&mut setup);
    setup
}

/// Fails the `lp_solves` reduction, the only single-element `f64` sum a
/// training iteration issues on one rank, on its `fail_on`-th call.
struct LpSolvesFailureComm {
    fail_on: usize,
    calls: AtomicUsize,
}

impl LpSolvesFailureComm {
    fn failing_on(fail_on: usize) -> Self {
        Self {
            fail_on,
            calls: AtomicUsize::new(0),
        }
    }
}

impl Communicator for LpSolvesFailureComm {
    fn allgatherv<T: CommData>(
        &self,
        send: &[T],
        recv: &mut [T],
        _counts: &[usize],
        _displs: &[usize],
    ) -> Result<(), CommError> {
        recv[..send.len()].clone_from_slice(send);
        Ok(())
    }

    fn allreduce<T: CommData>(
        &self,
        send: &[T],
        recv: &mut [T],
        op: ReduceOp,
    ) -> Result<(), CommError> {
        let is_lp_solves = matches!(op, ReduceOp::Sum)
            && send.len() == 1
            && TypeId::of::<T>() == TypeId::of::<f64>();
        if is_lp_solves && self.calls.fetch_add(1, Ordering::Relaxed) + 1 == self.fail_on {
            return Err(CommError::CollectiveFailed {
                operation: "allreduce",
                mpi_error_code: 1,
                message: "injected lp_solves failure".to_string(),
            });
        }
        recv.clone_from_slice(send);
        Ok(())
    }

    fn broadcast<T: CommData>(&self, _buf: &mut [T], _root: usize) -> Result<(), CommError> {
        Ok(())
    }

    fn barrier(&self) -> Result<(), CommError> {
        Ok(())
    }

    fn rank(&self) -> usize {
        0
    }

    fn size(&self) -> usize {
        1
    }

    fn abort(&self, error_code: i32) -> ! {
        std::process::exit(error_code)
    }
}

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

#[test]
fn resumed_run_stops_on_bound_stalling_without_reaccumulating_the_window() {
    let (output, _) = checkpoint_after(2);
    let mut setup = resumed_setup(
        &output,
        vec![
            StoppingRuleConfig::IterationLimit { limit: 10 },
            StoppingRuleConfig::BoundStalling {
                iterations: 3,
                tolerance: 1.0e6,
            },
        ],
    );

    let result = train_to_completion(&mut setup);

    assert_eq!(result.iterations, 3);
    assert_eq!(result.reason, "bound_stalling");
}

#[test]
fn resumed_run_stops_at_the_absolute_iteration_limit_through_the_stop_decision() {
    let (output, _) = checkpoint_after(2);
    let mut setup = resumed_setup(
        &output,
        vec![StoppingRuleConfig::IterationLimit { limit: 4 }],
    );

    let result = train_to_completion(&mut setup);

    assert_eq!(result.iterations, 4);
    assert!(result.stop_decision.configured_stop());
    assert!(
        result
            .stop_decision
            .mask()
            .contains(StopMask::ITERATION_LIMIT)
    );
}

#[test]
fn run_failing_after_the_stop_decision_records_only_the_committed_iterations() {
    let rules = vec![StoppingRuleConfig::IterationLimit { limit: 3 }];
    let reference = train_to_completion(&mut setup_under(rules.clone()).1);
    assert_eq!(reference.lower_bound_history.len(), 3);

    let outcome = train_with(
        &mut setup_under(rules).1,
        &LpSolvesFailureComm::failing_on(3),
    );

    assert!(matches!(outcome.error, Some(SddpError::Communication(_))));
    assert_eq!(outcome.result.iterations, 2);
    assert_eq!(
        bits(&outcome.result.lower_bound_history),
        bits(&reference.lower_bound_history[..2])
    );
}

#[test]
fn resumed_run_failing_after_the_stop_decision_records_only_the_committed_iterations() {
    let (output, recorded) = checkpoint_after(2);
    let mut setup = resumed_setup(
        &output,
        vec![StoppingRuleConfig::IterationLimit { limit: 4 }],
    );

    let outcome = train_with(&mut setup, &LpSolvesFailureComm::failing_on(1));

    assert!(matches!(outcome.error, Some(SddpError::Communication(_))));
    assert_eq!(outcome.result.iterations, 2);
    assert_eq!(bits(&outcome.result.lower_bound_history), bits(&recorded));
}
