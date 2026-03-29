//! Integration test for warm-start training.
//!
//! Trains a policy, saves it, then warm-starts a new training run from the
//! saved cuts. Verifies that the warm-start FCF has non-zero warm_start_count
//! and that training produces correct slot layouts.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::too_many_lines
)]

use std::path::Path;

use cobre_comm::{CommData, CommError, Communicator, ReduceOp};
use cobre_io::output::policy::{read_policy_checkpoint, write_policy_checkpoint};
use cobre_sddp::{
    FutureCostFunction, StudySetup,
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
}

/// Train fresh, save policy, warm-start train, verify improvement and cut counts.
#[test]
fn warm_start_training_preserves_cuts_and_trains_further() {
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
        prepare_stochastic(system, &case_dir, &config, 42).expect("prepare_stochastic");
    let system = prepare_result.system;

    // Phase 1: Fresh training.
    let stochastic_fresh = {
        let prep = prepare_stochastic(
            cobre_io::load_case(&case_dir).unwrap(),
            &case_dir,
            &config,
            42,
        )
        .unwrap();
        prep.stochastic
    };
    let hydro_models_fresh =
        prepare_hydro_models(&system, &case_dir).expect("prepare_hydro_models");
    let mut setup_fresh =
        StudySetup::new(&system, &config, stochastic_fresh, hydro_models_fresh).expect("setup");
    let comm = StubComm;
    let mut solver = HighsSolver::new().expect("HighsSolver");
    let fresh_outcome = setup_fresh
        .train(&mut solver, &comm, 1, HighsSolver::new, None, None)
        .expect("train");
    assert!(fresh_outcome.error.is_none());
    let fresh_result = fresh_outcome.result;
    let fresh_lb = fresh_result.final_lb;
    let fresh_active = setup_fresh.fcf().total_active_cuts();

    // Write policy checkpoint.
    let tmpdir = tempfile::tempdir().expect("tempdir");
    let policy_dir = tmpdir.path().join("policy");
    let fcf = setup_fresh.fcf();
    let stage_records = build_stage_cut_records(fcf);
    let stage_active_indices = build_active_indices(&stage_records);
    let stage_cuts = build_stage_cuts_payloads(fcf, &stage_records, &stage_active_indices);
    let (basis_col, basis_row) = convert_basis_cache(&fresh_result);
    let stage_bases = build_stage_basis_records(fcf, &fresh_result, &basis_col, &basis_row);
    let metadata = cobre_io::PolicyCheckpointMetadata {
        version: "1.0.0".to_string(),
        cobre_version: env!("CARGO_PKG_VERSION").to_string(),
        created_at: "2026-03-29T00:00:00Z".to_string(),
        completed_iterations: fresh_result.iterations as u32,
        final_lower_bound: fresh_result.final_lb,
        best_upper_bound: Some(fresh_result.final_ub),
        state_dimension: fcf.state_dimension as u32,
        num_stages: fcf.pools.len() as u32,
        config_hash: String::new(),
        system_hash: String::new(),
        max_iterations: setup_fresh.max_iterations() as u32,
        forward_passes: setup_fresh.forward_passes(),
        warm_start_cuts: 0,
        rng_seed: 42,
    };
    write_policy_checkpoint(&policy_dir, &stage_cuts, &stage_bases, &metadata)
        .expect("write checkpoint");

    // Phase 2: Warm-start training.
    let checkpoint = read_policy_checkpoint(&policy_dir).expect("read checkpoint");

    let stochastic_warm = {
        let prep = prepare_stochastic(
            cobre_io::load_case(&case_dir).unwrap(),
            &case_dir,
            &config,
            42,
        )
        .unwrap();
        prep.stochastic
    };
    let hydro_models_warm = prepare_hydro_models(&system, &case_dir).expect("prepare_hydro_models");
    let mut setup_warm =
        StudySetup::new(&system, &config, stochastic_warm, hydro_models_warm).expect("setup");

    // Build warm-start FCF and replace the fresh one.
    // Use max_iterations + 1 for capacity, matching the original FCF constructor
    // in from_broadcast_params (which does saturating_add(1)).
    let warm_fcf = FutureCostFunction::new_with_warm_start(
        &checkpoint.stage_cuts,
        setup_warm.forward_passes(),
        setup_warm.max_iterations().saturating_add(1),
    )
    .expect("warm-start FCF");
    setup_warm.replace_fcf(warm_fcf);

    // Verify warm-start state before training.
    let warm_start_count = setup_warm.fcf().pools[0].warm_start_count;
    assert!(warm_start_count > 0, "warm_start_count should be > 0");
    assert_eq!(
        setup_warm.fcf().total_active_cuts(),
        fresh_active,
        "warm-start FCF should have same active cuts as fresh training"
    );

    // Train warm-start.
    let mut solver_warm = HighsSolver::new().expect("HighsSolver");
    let warm_outcome = setup_warm
        .train(&mut solver_warm, &comm, 1, HighsSolver::new, None, None)
        .expect("warm-start train");
    assert!(warm_outcome.error.is_none());
    let warm_result = warm_outcome.result;

    // Verify: warm-start final LB should be >= fresh final LB
    // (warm-start starts with all cuts from fresh, so it can only improve).
    assert!(
        warm_result.final_lb >= fresh_lb - 1e-6,
        "warm-start LB ({}) should be >= fresh LB ({})",
        warm_result.final_lb,
        fresh_lb
    );

    // Verify total cuts = warm_start_count + new training cuts.
    let total_active_after = setup_warm.fcf().total_active_cuts();
    assert!(
        total_active_after > fresh_active,
        "warm-start training should produce more total cuts"
    );
}
