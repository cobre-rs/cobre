//! Integration test for simulation-only round-trip.
//!
//! Exercises the full simulation-only pipeline: train a policy, write it to
//! disk, load it back, and verify that the reconstructed FCF evaluates
//! identically to the original.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation
)]

use std::path::Path;

use cobre_comm::{CommData, CommError, Communicator, ReduceOp};
use cobre_core::scenario::ScenarioSource;
use cobre_io::output::policy::{read_policy_checkpoint, write_policy_checkpoint};
use cobre_sddp::{
    FutureCostFunction, StudySetup, build_basis_cache_from_checkpoint,
    hydro_models::prepare_hydro_models,
    policy_export::{
        build_active_indices, build_stage_basis_records, build_stage_cut_records,
        build_stage_cuts_payloads, convert_basis_cache,
    },
    setup::prepare_stochastic,
};
use cobre_solver::highs::HighsSolver;

/// Single-rank communicator stub for testing.
struct StubComm;

impl Communicator for StubComm {
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
        _op: ReduceOp,
    ) -> Result<(), CommError> {
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

/// Train the D01 case, write the policy checkpoint, load it back, and verify
/// that the reconstructed FCF evaluates identically to the original.
#[test]
fn simulation_only_fcf_round_trip() {
    let case_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("examples/deterministic/d01-thermal-dispatch");

    let config_path = case_dir.join("config.json");
    let config = cobre_io::parse_config(&config_path).expect("config must parse");

    let system = cobre_io::load_case(&case_dir).expect("load_case must succeed");
    let prepare_result =
        prepare_stochastic(system, &case_dir, &config, 42, &ScenarioSource::default())
            .expect("prepare_stochastic");
    let system = prepare_result.system;
    let stochastic = prepare_result.stochastic;

    let hydro_models = prepare_hydro_models(&system, &case_dir).expect("prepare_hydro_models");

    let mut setup =
        StudySetup::new(&system, &config, stochastic, hydro_models).expect("StudySetup");

    let comm = StubComm;
    let mut solver = HighsSolver::new().expect("HighsSolver");

    let outcome = setup
        .train(&mut solver, &comm, 1, HighsSolver::new, None, None)
        .expect("train must return Ok");
    assert!(outcome.error.is_none(), "expected no training error");
    let training_result = outcome.result;

    // Capture original FCF state.
    let original_active_cuts = setup.fcf().total_active_cuts();
    assert!(original_active_cuts > 0, "training should produce cuts");

    let n_stages = setup.num_stages();
    let state_dim = setup.fcf().state_dimension;

    // Evaluate at a representative state point for each stage.
    let test_state: Vec<f64> = vec![50.0; state_dim];
    let mut original_evals = Vec::with_capacity(n_stages);
    for stage in 0..n_stages {
        original_evals.push(setup.fcf().evaluate_at_state(stage, &test_state));
    }

    // Write policy checkpoint to a temporary directory.
    let tmpdir = tempfile::tempdir().expect("tempdir");
    let policy_dir = tmpdir.path().join("policy");

    let fcf = setup.fcf();
    let stage_records = build_stage_cut_records(fcf);
    let stage_active_indices = build_active_indices(&stage_records);
    let stage_cuts = build_stage_cuts_payloads(fcf, &stage_records, &stage_active_indices);

    let (basis_col_u8, basis_row_u8) = convert_basis_cache(&training_result);
    let stage_bases =
        build_stage_basis_records(fcf, &training_result, &basis_col_u8, &basis_row_u8);

    let metadata = cobre_io::PolicyCheckpointMetadata {
        cobre_version: env!("CARGO_PKG_VERSION").to_string(),
        created_at: "2026-03-29T00:00:00Z".to_string(),
        completed_iterations: training_result.iterations as u32,
        final_lower_bound: training_result.final_lb,
        best_upper_bound: Some(training_result.final_ub),
        state_dimension: state_dim as u32,
        num_stages: n_stages as u32,
        max_iterations: setup.max_iterations() as u32,
        forward_passes: setup.forward_passes(),
        warm_start_cuts: 0,
        rng_seed: 42,
        total_visited_states: 0,
    };

    write_policy_checkpoint(&policy_dir, &stage_cuts, &stage_bases, &metadata, &[])
        .expect("write checkpoint");

    // Read policy checkpoint back.
    let checkpoint = read_policy_checkpoint(&policy_dir).expect("read checkpoint");

    // Verify metadata round-trip.
    assert_eq!(
        checkpoint.metadata.state_dimension, state_dim as u32,
        "state_dimension must round-trip"
    );
    assert_eq!(
        checkpoint.metadata.num_stages, n_stages as u32,
        "num_stages must round-trip"
    );
    assert_eq!(
        checkpoint.stage_cuts.len(),
        n_stages,
        "stage_cuts count must match"
    );

    // Reconstruct FCF from deserialized data.
    let loaded_fcf =
        FutureCostFunction::from_deserialized(&checkpoint.stage_cuts).expect("from_deserialized");

    // Verify active cut count matches.
    assert_eq!(
        loaded_fcf.total_active_cuts(),
        original_active_cuts,
        "active cut count must match after round-trip"
    );

    // Verify evaluate_at_state produces identical results.
    for (stage, &expected_eval) in original_evals.iter().enumerate().take(n_stages) {
        let loaded_eval = loaded_fcf.evaluate_at_state(stage, &test_state);
        assert_eq!(
            loaded_eval, expected_eval,
            "evaluate_at_state mismatch at stage {stage}"
        );
    }

    // Verify basis cache round-trip.
    let loaded_basis_cache = build_basis_cache_from_checkpoint(n_stages, &checkpoint.stage_bases);
    assert_eq!(
        loaded_basis_cache.len(),
        n_stages,
        "basis cache length must match"
    );
    // At least some stages should have basis data.
    let has_basis = loaded_basis_cache.iter().any(Option::is_some);
    assert!(has_basis, "at least one stage should have basis data");
}
