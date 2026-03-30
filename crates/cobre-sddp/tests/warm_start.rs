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

/// Train for 5 iterations, save a checkpoint, resume from iteration 5 up to 10,
/// verify the resumed run reports 10 total iterations and a non-worse lower bound.
#[test]
fn resume_training_from_checkpoint() {
    let case_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("examples/deterministic/d01-thermal-dispatch");

    let config_path = case_dir.join("config.json");
    let config_full = cobre_io::parse_config(&config_path).expect("config must parse");

    // Build a 5-iteration config for phase 1 by overriding the stopping rules.
    let mut config_phase1 = config_full.clone();
    config_phase1.training.stopping_rules =
        Some(vec![cobre_io::config::StoppingRuleConfig::IterationLimit {
            limit: 5,
        }]);

    // Phase 1: Train for exactly 5 iterations.
    // Load fresh system and stochastic; hold `system` as an owned value so that
    // `&system` can be passed to prepare_hydro_models and StudySetup::new.
    let prep_phase1 = {
        let system = cobre_io::load_case(&case_dir).expect("load_case phase1");
        prepare_stochastic(system, &case_dir, &config_phase1, 42)
            .expect("prepare_stochastic phase1")
    };
    let system_phase1 = prep_phase1.system;
    let hydro_models_phase1 =
        prepare_hydro_models(&system_phase1, &case_dir).expect("prepare_hydro_models phase1");
    let mut setup_phase1 = StudySetup::new(
        &system_phase1,
        &config_phase1,
        prep_phase1.stochastic,
        hydro_models_phase1,
    )
    .expect("setup phase1");

    let comm = StubComm;
    let mut solver_phase1 = HighsSolver::new().expect("HighsSolver");
    let outcome_phase1 = setup_phase1
        .train(&mut solver_phase1, &comm, 1, HighsSolver::new, None, None)
        .expect("train phase1");
    assert!(outcome_phase1.error.is_none());
    let result_phase1 = outcome_phase1.result;
    assert_eq!(
        result_phase1.iterations, 5,
        "phase 1 must complete exactly 5 iterations"
    );
    let lb_phase1 = result_phase1.final_lb;

    // Save the checkpoint from phase 1.
    let tmpdir = tempfile::tempdir().expect("tempdir");
    let policy_dir = tmpdir.path().join("policy");
    let fcf_phase1 = setup_phase1.fcf();
    let stage_records = build_stage_cut_records(fcf_phase1);
    let stage_active_indices = build_active_indices(&stage_records);
    let stage_cuts = build_stage_cuts_payloads(fcf_phase1, &stage_records, &stage_active_indices);
    let (basis_col, basis_row) = convert_basis_cache(&result_phase1);
    let stage_bases = build_stage_basis_records(fcf_phase1, &result_phase1, &basis_col, &basis_row);
    let metadata = cobre_io::PolicyCheckpointMetadata {
        version: "1.0.0".to_string(),
        cobre_version: env!("CARGO_PKG_VERSION").to_string(),
        created_at: "2026-03-29T00:00:00Z".to_string(),
        completed_iterations: result_phase1.iterations as u32,
        final_lower_bound: result_phase1.final_lb,
        best_upper_bound: Some(result_phase1.final_ub),
        state_dimension: fcf_phase1.state_dimension as u32,
        num_stages: fcf_phase1.pools.len() as u32,
        config_hash: String::new(),
        system_hash: String::new(),
        max_iterations: setup_phase1.max_iterations() as u32,
        forward_passes: setup_phase1.forward_passes(),
        warm_start_cuts: 0,
        rng_seed: 42,
    };
    write_policy_checkpoint(&policy_dir, &stage_cuts, &stage_bases, &metadata)
        .expect("write checkpoint");

    // Phase 2: Resume from the checkpoint using the full 10-iteration config.
    let checkpoint = read_policy_checkpoint(&policy_dir).expect("read checkpoint");

    let prep_phase2 = {
        let system = cobre_io::load_case(&case_dir).expect("load_case phase2");
        prepare_stochastic(system, &case_dir, &config_full, 42).expect("prepare_stochastic phase2")
    };
    let system_phase2 = prep_phase2.system;
    let hydro_models_phase2 =
        prepare_hydro_models(&system_phase2, &case_dir).expect("prepare_hydro_models phase2");
    let mut setup_phase2 = StudySetup::new(
        &system_phase2,
        &config_full,
        prep_phase2.stochastic,
        hydro_models_phase2,
    )
    .expect("setup phase2");

    // Replace the FCF with a warm-start version loaded from the checkpoint, then
    // set the start iteration so the training loop begins at iteration 6.
    let warm_fcf = FutureCostFunction::new_with_warm_start(
        &checkpoint.stage_cuts,
        setup_phase2.forward_passes(),
        setup_phase2.max_iterations().saturating_add(1),
    )
    .expect("warm-start FCF");
    setup_phase2.replace_fcf(warm_fcf);
    setup_phase2.set_start_iteration(checkpoint.metadata.completed_iterations as u64);

    let mut solver_phase2 = HighsSolver::new().expect("HighsSolver");
    let outcome_phase2 = setup_phase2
        .train(&mut solver_phase2, &comm, 1, HighsSolver::new, None, None)
        .expect("train phase2");
    assert!(outcome_phase2.error.is_none());
    let result_phase2 = outcome_phase2.result;

    // The resumed run must report the absolute iteration count (10), not the delta (5).
    assert_eq!(
        result_phase2.iterations, 10,
        "resumed run must report 10 total iterations (not 5 delta)"
    );

    // The resumed run starts with all cuts from phase 1, so its final LB must be
    // at least as good as phase 1's final LB.
    assert!(
        result_phase2.final_lb >= lb_phase1 - 1e-6,
        "resumed LB ({}) must be >= phase-1 LB ({})",
        result_phase2.final_lb,
        lb_phase1
    );
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
