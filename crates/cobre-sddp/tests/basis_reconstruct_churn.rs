//! Cross-churn basis-reconstruction regression tests.
//!
//! Guards against regressions in [`cobre_sddp::basis_reconstruct`] that
//! would break slot reconciliation, `state_at_capture` routing, or the
//! iteration-wide hot-path allocation invariants introduced in Epic 01.
//!
//! ## Tests
//!
//! 1. [`basis_reconstruct_churn`] — cross-churn: all three churn types
//!    (LML1 deactivation, budget eviction, new cuts) active simultaneously
//!    on D03 (3 stages). Slot-tracked reconstruction is always active
//!    post-Epic-01.
//!
//! 2. [`test_basis_reconstruct_no_churn_full_preservation`] — happy-path:
//!    no cut selection, no eviction, cuts only grow.  By the third
//!    iteration the stored basis accumulates cut rows that are preserved by
//!    `reconstruct_basis`.
//!
//! 3. [`test_basis_reconstruct_full_churn_no_rows_preserved`] — edge-case:
//!    all iteration-1 cuts deactivated via direct pool mutation before
//!    iteration 2's forward pass.  Verifies the reconstruction path
//!    correctly reports 0 preserved rows.
//!
//! ## Regression sensitivity
//!
//! Injecting `preserved = 0` always (by reverting `reconstruct_basis`) will
//! cause test 2 to fail with a clear assertion message.  Injecting the
//! `padding_state = x_hat` regression from ticket-004 will cause test 1's
//! `simplex_iterations` to exceed the ±5 % tolerance band.
//!
//! ## Relation to Epic 01
//!
//! This is the closing test of Epic 01.  Epic 02's A/B benchmark depends
//! on this safety net being in place.
//!
//! ## Stage-count note
//!
//! The ticket spec calls for 5 study stages.  The repository's closest
//! clean deterministic case is D03 (3 stages, 2 cascaded hydros, 1
//! scenario per stage, no PAR estimation).  Three stages are sufficient to
//! exercise all three churn types when the cut budget is set to 6: after
//! iteration 2 (2 iterations × 3 forward-passes = 6 cuts per stage), any
//! new backward-pass cut triggers eviction.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::too_many_lines
)]

use std::path::Path;
use std::sync::mpsc;

use cobre_comm::{CommData, CommError, Communicator, ReduceOp};
use cobre_core::scenario::ScenarioSource;
use cobre_io::config::StoppingRuleConfig;
use cobre_sddp::{
    SolverStatsDelta, StudySetup, hydro_models::prepare_hydro_models, setup::prepare_stochastic,
};
use cobre_solver::highs::{HighsSolver, WarmStartBasisMode};

// ---------------------------------------------------------------------------
// StubComm — single-rank communicator for testing
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Path helpers
// ---------------------------------------------------------------------------

fn d03_case_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/<crate> must have a parent")
        .parent()
        .expect("crates/ must have a parent (repo root)")
        .join("examples/deterministic/d03-two-hydro-cascade")
}

fn d01_case_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/<crate> must have a parent")
        .parent()
        .expect("crates/ must have a parent (repo root)")
        .join("examples/deterministic/d01-thermal-dispatch")
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Aggregate `SolverStatsDelta` across all `"forward"` log entries.
fn sum_forward_deltas(log: &[(u64, &'static str, i32, SolverStatsDelta)]) -> SolverStatsDelta {
    SolverStatsDelta::aggregate(
        log.iter()
            .filter(|(_, phase, _, _)| *phase == "forward")
            .map(|(_, _, _, d)| d),
    )
}

// ---------------------------------------------------------------------------
// Test 1: Cross-churn — LML1 + budget + new cuts
// ---------------------------------------------------------------------------

/// End-to-end regression test exercising all three churn types simultaneously.
///
/// ## Churn sources
///
/// - **LML1 deactivation**: cuts whose `last_active_iter <
///   current_iteration - 2` are marked inactive every iteration
///   (`check_frequency = 1`, `memory_window = 2`).
/// - **Budget eviction**: `max_active_per_stage = 6`.  After iteration 2
///   (2 iters × 3 fwd-passes = 6 cuts per stage), each new backward cut
///   triggers eviction of the oldest cut.
/// - **New cuts**: every backward pass generates fresh cuts that extend the
///   active set.
///
/// ## What is pinned
///
/// - `simplex_iterations` total across all forward-pass log entries with a
///   ± 5 % tolerance band.  Run observed value; regenerate the pin only if
///   `HiGHS` major version changes.
/// - `final_lb` to float-exact equality (lower bound must be stable).
/// - `basis_rejections == 0` (reconstruction must never produce an invalid
///   warm-start basis).
///
/// ## Sensitivity
///
/// A regression that reverts `padding_state = x_hat` (ticket-004) will
/// increase `simplex_iterations` by roughly +6 %, which exceeds the ±5 %
/// band and causes the test to fail.
#[test]
fn basis_reconstruct_churn() {
    // Pinned regression values — declared first to satisfy `clippy::items_after_statements`.
    //
    // AC-5: simplex-iteration pin (±5 % tolerance).
    // Pinned from local run 2026-04-18 (ticket-009 enforce_basic_count_invariant).
    // The forward-path invariant fix reduces simplex iterations by ~9 % vs the
    // pre-fix pin of 198: demoting excess BASIC rows gives HiGHS a tighter warm
    // start, requiring fewer pivots.
    // Regenerate if `HiGHS` major version changes or the fixture parameters change.
    // Sensitivity: a padding_state = x_hat regression (ticket-004) raises this
    // count by ~+6 %, which exceeds the ±5 % band and trips this assertion.
    const PINNED_SIMPLEX_ITERS: u64 = 180;
    // ±5 % expressed as integer fractions to avoid `clippy::cast_sign_loss`.
    const LO_SIMPLEX: u64 = PINNED_SIMPLEX_ITERS * 95 / 100; // floor of 0.95×pin
    const HI_SIMPLEX: u64 = PINNED_SIMPLEX_ITERS * 105 / 100; // floor of 1.05×pin

    // AC-6: lower-bound pin (float-exact).
    // Pinned from local run 2026-04-17 (commit b866acb9).
    // The lower bound is deterministic for a fixed SDDP seed and scenario count.
    const PINNED_FINAL_LB: f64 = 1.390_333_333_333_335e6;

    let case_dir = d03_case_dir();
    let config_path = case_dir.join("config.json");
    let mut config = cobre_io::parse_config(&config_path).expect("config must parse");

    // Override training parameters for cross-churn fixture.
    config.training.forward_passes = Some(3);
    config.training.stopping_rules = Some(vec![StoppingRuleConfig::IterationLimit { limit: 8 }]);

    // LML1 cut selection: deactivate cuts unused for 2 iterations; check every iter.
    config.training.cut_selection.enabled = Some(true);
    config.training.cut_selection.method = Some("lml1".to_string());
    config.training.cut_selection.memory_window = Some(2);
    config.training.cut_selection.check_frequency = Some(1);

    // Cut budget: 6 active cuts per stage.
    // Eviction starts after iteration 2 (2 iters × 3 fwd-passes = 6 cuts/stage).
    config.training.cut_selection.max_active_per_stage = Some(6);

    let system = cobre_io::load_case(&case_dir).expect("load_case must succeed");
    let prepare_result =
        prepare_stochastic(system, &case_dir, &config, 42, &ScenarioSource::default())
            .expect("prepare_stochastic must succeed");
    let system = prepare_result.system;
    let stochastic = prepare_result.stochastic;

    let hydro_models =
        prepare_hydro_models(&system, &case_dir).expect("prepare_hydro_models must succeed");

    let mut setup =
        StudySetup::new(&system, &config, stochastic, hydro_models).expect("StudySetup must build");

    let comm = StubComm;
    let mut solver = HighsSolver::new().expect("HighsSolver::new must succeed");

    let outcome = setup
        .train(&mut solver, &comm, 1, HighsSolver::new, None, None)
        .expect("train must return Ok");
    assert!(
        outcome.error.is_none(),
        "basis_reconstruct_churn: expected no training error, got: {:?}",
        outcome.error
    );
    let result = outcome.result;
    assert_eq!(
        result.iterations, 8,
        "basis_reconstruct_churn: expected 8 iterations, got {}",
        result.iterations
    );

    // Aggregate forward-pass stats across all iterations.
    let fwd = sum_forward_deltas(&result.solver_stats_log);

    // AC-1: reconstruction preserved at least one row.
    // A regression that makes reconstruct_basis always return preserved=0
    // will cause this assertion to fail.
    assert!(
        fwd.basis_preserved > 0,
        "basis_reconstruct_churn: expected basis_preserved > 0, got {} \
         (reconstruction path must preserve at least one row across 8 iterations)",
        fwd.basis_preserved
    );

    // AC-2: at least one new cut was slack (BASIC) at the padding state.
    // After ticket-008, all new-cut rows are unconditionally assigned BASIC status
    // in reconstruct_basis, so basis_new_tight is always 0. The meaningful signal
    // is now basis_new_slack (new cuts classified as BASIC = not tight).
    assert!(
        fwd.basis_new_slack > 0,
        "basis_reconstruct_churn: expected basis_new_slack > 0, got {} \
         (at least one new cut must be classified as BASIC across 8 iterations)",
        fwd.basis_new_slack
    );

    // AC-3: at least one new cut was slack at the padding state.
    assert!(
        fwd.basis_new_slack > 0,
        "basis_reconstruct_churn: expected basis_new_slack > 0, got {} \
         (at least one new cut must be classified as BASIC across 8 iterations)",
        fwd.basis_new_slack
    );

    // AC-4: zero basis rejections — length-matching invariant must hold.
    // If reconstruction produces a basis with wrong row count, HiGHS rejects it.
    assert_eq!(
        fwd.basis_rejections, 0,
        "basis_reconstruct_churn: expected 0 basis rejections, got {} \
         (reconstructed bases must always be accepted by HiGHS)",
        fwd.basis_rejections
    );

    // AC-5: simplex-iteration pin (±5 % tolerance).
    let observed_simplex = fwd.simplex_iterations;
    assert!(
        (LO_SIMPLEX..=HI_SIMPLEX).contains(&observed_simplex),
        "basis_reconstruct_churn: simplex_iterations={observed_simplex} is outside \
         the ±5 % band [{LO_SIMPLEX}, {HI_SIMPLEX}] around pin={PINNED_SIMPLEX_ITERS} \
         (a padding_state=x_hat regression raises this by ~+6 %)"
    );

    assert_eq!(
        result.final_lb, PINNED_FINAL_LB,
        "basis_reconstruct_churn: final_lb={} does not match pin={PINNED_FINAL_LB:.15e} \
         (lower bound must be deterministic for fixed seed and scenario count)",
        result.final_lb
    );
}

// ---------------------------------------------------------------------------
// Test 2: No-churn — happy path, preservation grows across iterations
// ---------------------------------------------------------------------------

/// No-churn test: no cut selection, no budget, cuts only accumulate.
///
/// ## What this tests
///
/// After the first backward pass populates the FCF, subsequent forward
/// passes should see an increasing `basis_preserved` count as more cut rows
/// are available in the stored basis.  By iteration 3, cuts from iteration
/// 1's backward pass that appeared in iteration 2's LP are preserved in
/// iteration 3's warm-start.
///
/// ## Assertions
///
/// - `basis_new_tight + basis_new_slack > 0`: new cuts appear and are
///   correctly classified at the padding state.
/// - `basis_preserved > 0` by iteration 3: cuts from an earlier backward
///   pass are found in the stored basis and assigned the preserved status.
/// - `basis_rejections == 0`: no length-mismatch on the `HiGHS` side.
/// - Lower bound is positive and finite.
///
/// ## Failure mode
///
/// A regression that makes `reconstruct_basis` return `preserved = 0` for
/// every call will cause the `basis_preserved > 0` assertion to fail with
/// a message naming the column and value.
#[test]
fn test_basis_reconstruct_no_churn_full_preservation() {
    let case_dir = d03_case_dir();
    let config_path = case_dir.join("config.json");
    let mut config = cobre_io::parse_config(&config_path).expect("config must parse");

    // 2 forward passes, 3 iterations: enough for preservation to manifest.
    // Iteration 1: forward stores empty-cut basis; backward adds 2 cuts/stage.
    // Iteration 2: forward sees 2 cuts → new_tight/slack = 2; backward adds 2 more.
    // Iteration 3: forward sees 4 cuts, 2 from iter 1 backward are in stored basis
    //              → preserved = 2 per LP solve (≥ 1 stage × 2 fwd-passes).
    config.training.forward_passes = Some(2);
    config.training.stopping_rules = Some(vec![StoppingRuleConfig::IterationLimit { limit: 3 }]);

    // No cut selection — cuts only grow.
    config.training.cut_selection.enabled = Some(false);
    config.training.cut_selection.max_active_per_stage = None;

    let system = cobre_io::load_case(&case_dir).expect("load_case must succeed");
    let prepare_result =
        prepare_stochastic(system, &case_dir, &config, 42, &ScenarioSource::default())
            .expect("prepare_stochastic must succeed");
    let system = prepare_result.system;
    let stochastic = prepare_result.stochastic;

    let hydro_models =
        prepare_hydro_models(&system, &case_dir).expect("prepare_hydro_models must succeed");

    let mut setup =
        StudySetup::new(&system, &config, stochastic, hydro_models).expect("StudySetup must build");

    let comm = StubComm;
    let mut solver = HighsSolver::new().expect("HighsSolver::new must succeed");

    let outcome = setup
        .train(&mut solver, &comm, 1, HighsSolver::new, None, None)
        .expect("train must return Ok");
    assert!(
        outcome.error.is_none(),
        "test_basis_reconstruct_no_churn_full_preservation: unexpected training error: {:?}",
        outcome.error
    );
    let result = outcome.result;
    assert_eq!(
        result.iterations, 3,
        "no_churn: expected 3 iterations, got {}",
        result.iterations
    );

    // Aggregate forward stats across all iterations.
    let fwd = sum_forward_deltas(&result.solver_stats_log);

    // AC-A: new cuts appear and are classified.
    // Cuts from iteration 1's backward pass are "new" in iteration 2's forward.
    assert!(
        fwd.basis_new_tight + fwd.basis_new_slack > 0,
        "no_churn: expected basis_new_tight + basis_new_slack > 0, got tight={} slack={} \
         (iteration 2 forward must classify new cuts at the padding state)",
        fwd.basis_new_tight,
        fwd.basis_new_slack
    );

    // AC-B: preservation manifest from iteration 3 onward.
    // Cuts that appeared in iteration 2's LP are preserved in iteration 3's forward.
    assert!(
        fwd.basis_preserved > 0,
        "no_churn: expected basis_preserved > 0, got {} \
         (by iteration 3 the stored basis must contain cut rows to preserve; \
          a regression that always returns preserved=0 will trip this assertion)",
        fwd.basis_preserved
    );

    // AC-C: zero rejections.
    assert_eq!(
        fwd.basis_rejections, 0,
        "no_churn: expected 0 basis rejections, got {}",
        fwd.basis_rejections
    );

    // AC-D: lower bound is positive and finite.
    assert!(
        result.final_lb.is_finite() && result.final_lb > 0.0,
        "no_churn: expected finite positive lower bound, got {}",
        result.final_lb
    );
}

// ---------------------------------------------------------------------------
// Test 3: Full churn — all iteration-1 cuts deactivated before iteration 2
// ---------------------------------------------------------------------------

/// Full-churn edge case: deactivate every cut from iteration 1 before the
/// iteration-2 forward pass.
///
/// ## Setup
///
/// 1. Train D01 (2 stages, 1 scenario/stage) for 1 iteration.  This generates
///    2 cuts per stage (from 2 forward passes); slot-tracked reconstruction
///    is always active post-Epic-01.
/// 2. Deactivate ALL cuts in every pool via direct pool mutation.
/// 3. Resume training from iteration 2 using `set_start_iteration`.
///    Iteration 2's forward pass now sees an empty FCF.
///
/// ## Assertions
///
/// - `basis_preserved == 0` for iteration 2's forward: with no active cuts,
///   there is nothing to preserve.
/// - `basis_new_tight + basis_new_slack == 0` for iteration 2's forward:
///   with no active cuts, there are no new rows to classify.
/// - `preserved + new_tight + new_slack == 0 == active_cuts_at_forward_time`:
///   the accounting invariant holds for the degenerate (empty) case.
/// - `basis_rejections == 0`: an empty FCF must still produce a valid warm-start.
/// - The LP optimum of iteration 2 is finite (cold-start with no cuts).
///
/// ## Failure mode
///
/// If the reconstruction path panics or produces an invalid basis when the
/// FCF is empty, `basis_rejections` will be non-zero and the assertion will
/// fire.
#[test]
fn test_basis_reconstruct_full_churn_no_rows_preserved() {
    let case_dir = d01_case_dir();
    let config_path = case_dir.join("config.json");
    let mut config = cobre_io::parse_config(&config_path).expect("config must parse");

    // Phase 1: train for 1 iteration, basis padding enabled.
    //
    // Two IterationLimit rules in "any" mode (the default):
    //  - limit: 2 → sizes the FCF cut pool to (2+1) × forward_passes = 6 slots per stage
    //              so that the transplanted FCF has room for phase-2 cuts at slots 2,3.
    //  - limit: 1 → stops training after iteration 1 (fires first in "any" mode).
    //
    // If we used only limit:1 here, FCF capacity = (1+1)×2 = 4, and iter-2's backward
    // pass would panic at "cut slot 4 is out of bounds (capacity = 4)".
    config.training.forward_passes = Some(2);
    config.training.stopping_rules = Some(vec![
        StoppingRuleConfig::IterationLimit { limit: 2 }, // FCF capacity sizing
        StoppingRuleConfig::IterationLimit { limit: 1 }, // actual stop point
    ]);
    let system = cobre_io::load_case(&case_dir).expect("load_case phase1 must succeed");
    let prepare_result =
        prepare_stochastic(system, &case_dir, &config, 42, &ScenarioSource::default())
            .expect("prepare_stochastic phase1 must succeed");
    let system = prepare_result.system;
    let stochastic = prepare_result.stochastic;

    let hydro_models =
        prepare_hydro_models(&system, &case_dir).expect("prepare_hydro_models must succeed");

    let mut setup = StudySetup::new(&system, &config, stochastic, hydro_models)
        .expect("StudySetup phase1 must build");

    let comm = StubComm;
    let mut solver1 = HighsSolver::new().expect("HighsSolver phase1 must succeed");

    let outcome1 = setup
        .train(&mut solver1, &comm, 1, HighsSolver::new, None, None)
        .expect("phase1 train must return Ok");
    assert!(
        outcome1.error.is_none(),
        "full_churn: phase 1 unexpected error: {:?}",
        outcome1.error
    );
    assert_eq!(
        outcome1.result.iterations, 1,
        "full_churn: phase 1 must complete exactly 1 iteration"
    );

    // Verify that phase 1 generated at least some cuts (otherwise the test
    // is degenerate and cannot exercise the full-churn path).
    let cuts_after_iter1: Vec<usize> = setup
        .fcf()
        .pools
        .iter()
        .map(cobre_sddp::CutPool::active_count)
        .collect();
    let total_cuts_iter1: usize = cuts_after_iter1.iter().sum();
    assert!(
        total_cuts_iter1 > 0,
        "full_churn: phase 1 must generate at least 1 cut, got 0; \
         test cannot exercise the deactivation path otherwise"
    );

    // --- Direct pool mutation: deactivate ALL cuts in every stage pool ---
    {
        let fcf = setup.fcf_mut();
        for (stage, pool) in fcf.pools.iter_mut().enumerate() {
            let active_indices: Vec<u32> = (0..pool.populated_count)
                .filter(|&i| pool.active[i])
                .map(|i| i as u32)
                .collect();
            if !active_indices.is_empty() {
                pool.deactivate(&active_indices);
            }
            assert_eq!(
                pool.active_count(),
                0,
                "full_churn: stage {stage} pool must have 0 active cuts after deactivation, \
                 got {}",
                pool.active_count()
            );
        }
    }

    // --- Phase 2: resume from iteration 2 (after full deactivation) ---
    // Override stopping rule to run exactly 1 more iteration (iteration 2).
    {
        // Rebuild config for phase 2: limit to 2 iterations total (start at 1, run to 2).
        config.training.stopping_rules =
            Some(vec![StoppingRuleConfig::IterationLimit { limit: 2 }]);
    }

    // Update start_iteration so the training loop begins at iteration 2.
    setup.set_start_iteration(1);

    let mut solver2 = HighsSolver::new().expect("HighsSolver phase2 must succeed");

    // We cannot call `setup.train` again with the new stopping rule because
    // the config is baked into setup at construction time. Instead we use the
    // training loop directly by rebuilding a fresh setup from the same system
    // but with the updated iteration limit, then transplanting the FCF pool.
    //
    // The simplest approach: rebuild setup with the 2-iteration config and
    // replace its FCF with the (fully-deactivated) pool from phase 1.
    {
        let system2 = cobre_io::load_case(&case_dir).expect("load_case phase2 must succeed");
        let prepare2 =
            prepare_stochastic(system2, &case_dir, &config, 42, &ScenarioSource::default())
                .expect("prepare_stochastic phase2 must succeed");
        let system2 = prepare2.system;
        let stochastic2 = prepare2.stochastic;
        let hydro2 =
            prepare_hydro_models(&system2, &case_dir).expect("prepare_hydro_models phase2");

        let mut setup2 =
            StudySetup::new(&system2, &config, stochastic2, hydro2).expect("StudySetup phase2");

        // Transplant the deactivated FCF from phase 1.
        // Read the metadata we need to construct the placeholder FCF before
        // taking the mutable borrow, to satisfy the borrow checker.
        let n_stages = setup.fcf().pools.len();
        let state_dim = setup.fcf().state_dimension;
        let fwd_passes = setup.forward_passes();
        let max_iters = setup.max_iterations();
        let placeholder_fcf = cobre_sddp::FutureCostFunction::new(
            n_stages,
            state_dim,
            fwd_passes,
            max_iters,
            &vec![0u32; n_stages],
        );
        let deactivated_fcf = std::mem::replace(setup.fcf_mut(), placeholder_fcf);
        setup2.replace_fcf(deactivated_fcf);
        setup2.set_start_iteration(1);

        let outcome2 = setup2
            .train(&mut solver2, &comm, 1, HighsSolver::new, None, None)
            .expect("phase2 train must return Ok");
        assert!(
            outcome2.error.is_none(),
            "full_churn: phase 2 unexpected error: {:?}",
            outcome2.error
        );
        assert_eq!(
            outcome2.result.iterations, 2,
            "full_churn: phase 2 must report 2 total iterations, got {}",
            outcome2.result.iterations
        );

        let result2 = outcome2.result;

        // Filter to only iteration 2's forward log entry.
        let iter2_fwd: Vec<&SolverStatsDelta> = result2
            .solver_stats_log
            .iter()
            .filter(|(iter, phase, _, _)| *iter == 2 && *phase == "forward")
            .map(|(_, _, _, d)| d)
            .collect();

        assert!(
            !iter2_fwd.is_empty(),
            "full_churn: must have at least one forward log entry for iteration 2"
        );

        let iter2_fwd_agg = SolverStatsDelta::aggregate(iter2_fwd.into_iter());

        // AC-F1: zero preserved — no active cuts in the FCF at forward time.
        assert_eq!(
            iter2_fwd_agg.basis_preserved, 0,
            "full_churn: iteration 2 forward must have basis_preserved=0, got {} \
             (all iteration-1 cuts were deactivated before iteration 2's forward pass)",
            iter2_fwd_agg.basis_preserved
        );

        // AC-F2: zero new classified — no active cuts → nothing to classify.
        assert_eq!(
            iter2_fwd_agg.basis_new_tight + iter2_fwd_agg.basis_new_slack,
            0,
            "full_churn: iteration 2 forward must have new_tight=0 and new_slack=0, \
             got tight={} slack={} (no active cuts means no new rows to classify)",
            iter2_fwd_agg.basis_new_tight,
            iter2_fwd_agg.basis_new_slack
        );

        // AC-F3: accounting invariant — preserved + new_tight + new_slack == active_cuts == 0.
        let total_accounted = iter2_fwd_agg.basis_preserved
            + iter2_fwd_agg.basis_new_tight
            + iter2_fwd_agg.basis_new_slack;
        assert_eq!(
            total_accounted, 0,
            "full_churn: accounting invariant: preserved+new_tight+new_slack must equal \
             active_cuts_at_forward_time (0), got {total_accounted}"
        );

        // AC-F4: zero basis rejections — empty FCF must still produce a valid warm-start.
        assert_eq!(
            iter2_fwd_agg.basis_rejections, 0,
            "full_churn: iteration 2 forward must have 0 basis rejections, got {} \
             (empty FCF must not cause HiGHS to reject the warm-start basis)",
            iter2_fwd_agg.basis_rejections
        );

        // AC-F5: LP optimum is finite (cold-start with no cuts runs successfully).
        assert!(
            result2.final_lb.is_finite(),
            "full_churn: final_lb must be finite after full-churn iteration, got {}",
            result2.final_lb
        );
    }
}

// ---------------------------------------------------------------------------
// Test 4: Simulation smoke — basis_preserved > 0 after train + simulate
// ---------------------------------------------------------------------------

/// Smoke test: after 2-iteration training on D03, simulation with the trained
/// `basis_cache` produces `basis_preserved > 0` in the aggregated solver stats.
///
/// ## What this guards
///
/// This test verifies that the simulation warm-start path introduced in
/// ticket-010 actually reaches `reconstruct_basis` and records at least one
/// preserved row in the `SolverStatistics`.  A regression that accidentally
/// gates reconstruction off (e.g., always taking the cold-start branch) will
/// produce `basis_preserved == 0` and fail this assertion.
///
/// ## Setup
///
/// - D03 case (3 stages, 2 cascaded hydros).
/// - 2 training iterations with 2 forward passes: enough for the backward pass
///   to populate the FCF, and for the second iteration's forward pass to store
///   a basis with non-empty `cut_row_slots` in `basis_cache`.
/// - Simulation runs with the stored `basis_cache` from training.
///
/// ## Assertion
///
/// `sum(basis_preserved across all simulation scenarios) > 0`.
/// `basis_rejections == 0` (reconstructed bases must always be accepted).
#[test]
fn simulate_warm_start_basis_preserved_gt_zero() {
    let case_dir = d03_case_dir();
    let config_path = case_dir.join("config.json");
    let mut config = cobre_io::parse_config(&config_path).expect("config must parse");

    // 2 forward passes, 2 iterations: backward pass from iter 1 populates the FCF;
    // iter 2 forward pass captures a basis with non-empty cut_row_slots, which is
    // then available in basis_cache for the simulation warm-start.
    config.training.forward_passes = Some(2);
    config.training.stopping_rules = Some(vec![StoppingRuleConfig::IterationLimit { limit: 2 }]);

    // No cut selection — cuts only grow; keeps slot tracking simple.
    config.training.cut_selection.enabled = Some(false);
    config.training.cut_selection.max_active_per_stage = None;

    // Enable simulation so StudySetup::simulation_config() returns n_scenarios > 0.
    // 2 scenarios gives a meaningful basis_preserved count without being slow.
    config.simulation.enabled = true;
    config.simulation.num_scenarios = 2;

    let system = cobre_io::load_case(&case_dir).expect("load_case must succeed");
    let prepare_result =
        prepare_stochastic(system, &case_dir, &config, 42, &ScenarioSource::default())
            .expect("prepare_stochastic must succeed");
    let system = prepare_result.system;
    let stochastic = prepare_result.stochastic;

    let hydro_models =
        prepare_hydro_models(&system, &case_dir).expect("prepare_hydro_models must succeed");

    let mut setup =
        StudySetup::new(&system, &config, stochastic, hydro_models).expect("StudySetup must build");

    let comm = StubComm;
    let mut train_solver = HighsSolver::new().expect("HighsSolver for training must succeed");

    let outcome = setup
        .train(&mut train_solver, &comm, 1, HighsSolver::new, None, None)
        .expect("train must return Ok");
    assert!(
        outcome.error.is_none(),
        "simulate_warm_start: expected no training error, got: {:?}",
        outcome.error
    );
    assert_eq!(
        outcome.result.iterations, 2,
        "simulate_warm_start: expected 2 iterations, got {}",
        outcome.result.iterations
    );

    // Verify training produced non-empty basis_cache at stage 0 with slots.
    let basis_cache = &outcome.result.basis_cache;
    assert!(
        basis_cache
            .iter()
            .any(|cb| cb.as_ref().is_some_and(|b| !b.cut_row_slots.is_empty())),
        "simulate_warm_start: at least one stage must have a CapturedBasis with \
         non-empty cut_row_slots in basis_cache after 2 iterations"
    );

    // Diagnostic: inspect the FCF pool state after training.
    // Build a simulation workspace pool (1 thread, HighsSolver).
    let mut pool = setup
        .create_workspace_pool(1, HighsSolver::new)
        .expect("simulation workspace pool must build");

    let io_capacity = setup.io_channel_capacity().max(1);
    let (result_tx, result_rx) = mpsc::sync_channel(io_capacity);

    // Drain simulation results on a background thread to avoid channel backpressure.
    let drain_handle = std::thread::spawn(move || result_rx.into_iter().count());

    // Pass baked_templates=None to force the fallback (non-baked) reconstruction
    // path.  On the baked path reconstruct_basis receives an empty
    // current_cut_rows iterator (all cuts are structural), so basis_preserved is
    // always 0 there by design.  The fallback path passes pool.active_cuts() as
    // current_cut_rows; stored slots from basis_cache are matched against the
    // active pool slots and preserved, producing basis_preserved > 0.
    let sim_result = setup
        .simulate(
            &mut pool.workspaces,
            &comm,
            &result_tx,
            None,
            None, // force fallback path so basis_preserved > 0 is verifiable
            &outcome.result.basis_cache,
        )
        .expect("simulate must return Ok");

    drop(result_tx);
    drain_handle.join().expect("drain thread must not panic");

    // Aggregate basis_preserved and basis_rejections across all simulation scenarios.
    let total_preserved: u64 = sim_result
        .solver_stats
        .iter()
        .map(|(_, delta)| delta.basis_preserved)
        .sum();
    let total_rejections: u64 = sim_result
        .solver_stats
        .iter()
        .map(|(_, delta)| delta.basis_rejections)
        .sum();

    assert!(
        total_preserved > 0,
        "simulate_warm_start: expected basis_preserved > 0 in simulation, got 0 \
         (reconstruction path must be exercised when basis_cache has non-empty slots)"
    );
    assert_eq!(
        total_rejections, 0,
        "simulate_warm_start: expected 0 basis_rejections in simulation, got {total_rejections} \
         (reconstructed bases must always be accepted by HiGHS)"
    );
}

// ---------------------------------------------------------------------------
// Test 5: Simulation invariant fix — cut-set churn + enforce_basic_count_invariant
// ---------------------------------------------------------------------------

/// Runs a training + simulation cycle with cut-selection churn enabled under
/// both `WarmStartBasisMode` variants and asserts that the simulation pipeline
/// produces zero `basis_non_alien_rejections`.
///
/// ## What this guards
///
/// Two bugs from `docs/assessments/warm-start-observability-findings.md`:
///
/// 1. **Simulation warm-start wiring**: under `AlienOnly` the simulation must
///    never call the non-alien setter, so `basis_non_alien_rejections == 0`
///    is structural.  A regression that drops the `with_warm_start_mode`
///    threading from a workspace-pool factory will reach the non-alien path
///    and, on a large enough workload, record non-zero rejections.
///
/// 2. **`enforce_basic_count_invariant` in simulation**: under
///    `NonAlienFirst`, dropped non-BASIC cuts inflate `total_basic` beyond
///    `num_row` and trip `HiGHS`'s `isBasisConsistent` check.  The simulation
///    pipeline must mirror the forward-path demotion pass.
///
/// Sensitivity to a missing invariant pass is covered by the unit tests in
/// `basis_reconstruct.rs::tests::reconstructed_basis_preserves_basic_count_invariant_*`.
/// This integration test is the smoke-level guard that the simulation
/// pipeline actually reaches `enforce_basic_count_invariant` on both arms.
///
/// ## Setup
///
/// D03 with LML1 cut selection (`memory_window=2`, `check_frequency=1`) and a
/// `max_active_per_stage` budget of 6.  4 training iterations give three
/// rounds of cut-selection churn before simulation.
#[test]
fn simulate_invariant_zero_non_alien_rejections_under_churn() {
    for mode in [
        WarmStartBasisMode::NonAlienFirst,
        WarmStartBasisMode::AlienOnly,
    ] {
        let case_dir = d03_case_dir();
        let config_path = case_dir.join("config.json");
        let mut config = cobre_io::parse_config(&config_path).expect("config must parse");

        // Cut-set churn: LML1 + budget eviction.  4 iterations × 3 forward
        // passes produces enough churn to expose stale-basis behaviour in
        // simulation.
        config.training.forward_passes = Some(3);
        config.training.stopping_rules =
            Some(vec![StoppingRuleConfig::IterationLimit { limit: 4 }]);
        config.training.cut_selection.enabled = Some(true);
        config.training.cut_selection.method = Some("lml1".to_string());
        config.training.cut_selection.memory_window = Some(2);
        config.training.cut_selection.check_frequency = Some(1);
        config.training.cut_selection.max_active_per_stage = Some(6);

        // 2 simulation scenarios is enough to exercise the warm-start path.
        config.simulation.enabled = true;
        config.simulation.num_scenarios = 2;

        let system = cobre_io::load_case(&case_dir).expect("load_case must succeed");
        let prepare_result =
            prepare_stochastic(system, &case_dir, &config, 42, &ScenarioSource::default())
                .expect("prepare_stochastic must succeed");
        let system = prepare_result.system;
        let stochastic = prepare_result.stochastic;

        let hydro_models =
            prepare_hydro_models(&system, &case_dir).expect("prepare_hydro_models must succeed");

        let mut setup = StudySetup::new(&system, &config, stochastic, hydro_models)
            .expect("StudySetup must build");

        let comm = StubComm;
        let mut train_solver = HighsSolver::new()
            .expect("HighsSolver for training must succeed")
            .with_warm_start_mode(mode);

        let outcome = setup
            .train(
                &mut train_solver,
                &comm,
                1,
                move || HighsSolver::new().map(|s| s.with_warm_start_mode(mode)),
                None,
                None,
            )
            .expect("train must return Ok");
        assert!(
            outcome.error.is_none(),
            "simulate_invariant[{mode:?}]: unexpected training error: {:?}",
            outcome.error
        );

        let mut pool = setup
            .create_workspace_pool(1, move || {
                HighsSolver::new().map(|s| s.with_warm_start_mode(mode))
            })
            .expect("simulation workspace pool must build");

        let io_capacity = setup.io_channel_capacity().max(1);
        let (result_tx, result_rx) = mpsc::sync_channel(io_capacity);
        let drain_handle = std::thread::spawn(move || result_rx.into_iter().count());

        let sim_result = setup
            .simulate(
                &mut pool.workspaces,
                &comm,
                &result_tx,
                None,
                // Fallback (non-baked) path: exercises the delta-cut-row arm
                // where stale-basis inflation can occur.
                None,
                &outcome.result.basis_cache,
            )
            .expect("simulate must return Ok");

        drop(result_tx);
        drain_handle.join().expect("drain thread must not panic");

        let total_non_alien: u64 = sim_result
            .solver_stats
            .iter()
            .map(|(_, delta)| delta.basis_non_alien_rejections)
            .sum();
        let total_rejections: u64 = sim_result
            .solver_stats
            .iter()
            .map(|(_, delta)| delta.basis_rejections)
            .sum();

        assert_eq!(
            total_non_alien, 0,
            "simulate_invariant[{mode:?}]: expected 0 non-alien rejections, got \
             {total_non_alien} (invariant fix or warm-start-mode wiring regressed)"
        );
        assert_eq!(
            total_rejections, 0,
            "simulate_invariant[{mode:?}]: expected 0 basis rejections, got {total_rejections}"
        );
    }
}
