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
//!    all iteration-1 cuts deactivated before iteration 2's forward pass.
//!    Verifies the reconstruction path handles an empty FCF without panicking.
//!
//! ## Regression sensitivity
//!
//! Injecting the `padding_state = x_hat` regression from ticket-004 will
//! cause test 1's `simplex_iterations` to exceed the ±5 % tolerance band.
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
use cobre_solver::highs::HighsSolver;

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
fn sum_forward_deltas(
    log: &[(u64, &'static str, i32, i32, i32, i32, SolverStatsDelta)],
) -> SolverStatsDelta {
    SolverStatsDelta::aggregate(
        log.iter()
            .filter(|(_, phase, _, _, _, _, _)| *phase == "forward")
            .map(|(_, _, _, _, _, _, d)| d),
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
/// - `basis_consistency_failures == 0` (reconstruction must never produce an invalid
///   warm-start basis).
///
/// ## Always-baked reconstruction (ticket-002)
///
/// On the always-baked forward path, cuts are baked as structural rows into the
/// stage template. `reconstruct_basis` is called once per warm-start solve, so
/// `basis_reconstructions > 0` across all forward-pass log entries. The critical
/// invariant is that reconstruction is active and produces zero consistency
/// failures. Only `basis_consistency_failures` and `simplex_iterations` matter
/// for correctness pinning here.
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
    // Pinned 2026-04-22 from the transient-G1 design (Epic 06 T4 refinement):
    // `add_cut` seeds `SEED_BIT` (bit 31, outside RECENT_WINDOW_BITS) instead of
    // bit 0, and the end-of-iter shift clears SEED_BIT so the G1 signal does not
    // carry into the next iteration. Previous pin was 120 (pre-T4; classifier
    // dormant) and 123 (T4 cross-iter-persistent seed).
    //
    // D03 churn is an adversarial fixture for transient-G1: it has only 3 FP
    // and 1 scenario per stage, so iter-to-iter x̂ drift is tiny and cross-iter
    // G1 persistence yields a real 68-simplex win. Transient-G1 gives that up
    // in exchange for recovering the convertido (stochastic, 50 FP) wall-clock.
    //
    // Regenerate if `HiGHS` major version changes, the fixture parameters
    // change, or the G1 design changes again.
    //
    // Sensitivity: the `padding_state = x_hat` regression this test was
    // designed to catch raises this count to >>193, well outside the ±5 % band.
    const PINNED_SIMPLEX_ITERS: u64 = 184;
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

    // AC-1: On the always-baked forward path (ticket-002), reconstruct_basis is
    // called once per warm-start solve. The counter must be non-zero across all
    // forward-pass log entries, confirming basis reconstruction is active.
    assert!(
        fwd.basis_reconstructions > 0,
        "basis_reconstruct_churn: forward warm-start solves must increment \
         basis_reconstructions, got {}",
        fwd.basis_reconstructions
    );

    // AC-4: zero basis rejections — length-matching invariant must hold.
    // If reconstruction produces a basis with wrong row count, HiGHS rejects it.
    assert_eq!(
        fwd.basis_consistency_failures, 0,
        "basis_reconstruct_churn: expected 0 basis rejections, got {} \
         (reconstructed bases must always be accepted by HiGHS)",
        fwd.basis_consistency_failures
    );

    // AC-5: simplex-iteration pin (±5 % tolerance).
    let observed_simplex = fwd.simplex_iterations;
    assert!(
        (LO_SIMPLEX..=HI_SIMPLEX).contains(&observed_simplex),
        "basis_reconstruct_churn: simplex_iterations={observed_simplex} is outside \
         the ±5 % band [{LO_SIMPLEX}, {HI_SIMPLEX}] around pin={PINNED_SIMPLEX_ITERS}"
    );

    assert_eq!(
        result.final_lb, PINNED_FINAL_LB,
        "basis_reconstruct_churn: final_lb={} does not match pin={PINNED_FINAL_LB:.15e} \
         (lower bound must be deterministic for fixed seed and scenario count)",
        result.final_lb
    );
}

// ---------------------------------------------------------------------------
// Test 2: No-churn — happy path, reconstruction active across iterations
// ---------------------------------------------------------------------------

/// No-churn test: no cut selection, no budget, cuts only accumulate.
///
/// ## What this tests
///
/// Verifies the no-churn happy path on the always-baked forward path
/// (ticket-002). Cuts are baked as structural rows into the stage template;
/// `reconstruct_basis` is called once per warm-start solve so
/// `basis_reconstructions > 0` across all forward-pass log entries.
/// Warm-start is effective via `load_model(baked_template)`.
///
/// ## Assertions
///
/// - `basis_reconstructions > 0`: reconstruction is called on every warm-start
///   solve; a non-zero value confirms the always-baked path is active.
/// - `basis_consistency_failures == 0`: no length-mismatch on the `HiGHS` side.
/// - Lower bound is positive and finite.
///
/// ## Failure mode
///
/// A regression that accidentally routes the forward path away from
/// `baked_template: Some(...)` will change the lower bound or alter
/// the reconstruction call pattern.
#[test]
fn test_basis_reconstruct_no_churn_full_preservation() {
    let case_dir = d03_case_dir();
    let config_path = case_dir.join("config.json");
    let mut config = cobre_io::parse_config(&config_path).expect("config must parse");

    // 2 forward passes, 3 iterations: enough for reconstruction to accumulate.
    // Iteration 1: forward stores empty-cut basis; backward adds 2 cuts/stage.
    // Iteration 2: forward warm-starts from the stored basis (basis_reconstructions > 0).
    // Iteration 3: forward warm-starts again; basis_reconstructions keeps growing.
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

    // AC-A: On the always-baked forward path (ticket-002), reconstruct_basis is
    // called once per warm-start solve. The counter must be non-zero, confirming
    // basis reconstruction is active across all 3 iterations.
    assert!(
        fwd.basis_reconstructions > 0,
        "no_churn: forward warm-start solves must increment basis_reconstructions, \
         got {}",
        fwd.basis_reconstructions
    );

    // AC-C: zero rejections.
    assert_eq!(
        fwd.basis_consistency_failures, 0,
        "no_churn: expected 0 basis rejections, got {}",
        fwd.basis_consistency_failures
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
/// 3. Rebuild `setup2` — a fresh `StudySetup` with the deactivated FCF
///    transplanted in.  Because `setup2` is a cold-start (no prior
///    `record_reconstruction_stats` call), `basis_reconstructions == 0` for
///    phase 2 is correct: the solver never had a stored basis to reconstruct.
///
/// ## Cold-start semantics for phase 2
///
/// `setup2` is constructed fresh and has no stored basis in `basis_cache`.
/// The forward pass in iteration 2 therefore runs as a cold-start: `HiGHS` is
/// given no warm-start basis and `record_reconstruction_stats` is never called.
/// `basis_reconstructions == 0` is the expected value.  The classification
/// assertions (`basis_reconstructions > 0`) are dropped here on purpose; the
/// safety goal is that an empty FCF does not cause a panic or an invalid basis.
///
/// ## Assertions
///
/// - `basis_consistency_failures == 0`: an empty FCF must not produce an
///   invalid warm-start (or any other `HiGHS` rejection).
/// - The LP optimum of iteration 2 is finite (cold-start with no cuts
///   must still find a feasible solution).
///
/// ## Failure mode
///
/// If the reconstruction path panics or corrupts the LP when the FCF is
/// empty, `basis_consistency_failures` will be non-zero and the assertion will
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
            .filter(|(iter, phase, _, _, _, _, _)| *iter == 2 && *phase == "forward")
            .map(|(_, _, _, _, _, _, d)| d)
            .collect();

        assert!(
            !iter2_fwd.is_empty(),
            "full_churn: must have at least one forward log entry for iteration 2"
        );

        let iter2_fwd_agg = SolverStatsDelta::aggregate(iter2_fwd.into_iter());

        // Safety invariant: empty FCF must not cause HiGHS to reject the warm-start
        // basis (or panic).  Phase 2 is a cold-start (setup2 has no prior stored
        // basis), so basis_reconstructions == 0 is expected and not asserted here.
        // The classification assertion (`basis_reconstructions > 0`) is intentionally
        // absent — see the test docstring for the cold-start rationale.
        assert_eq!(
            iter2_fwd_agg.basis_consistency_failures, 0,
            "full_churn: iteration 2 forward must have 0 basis rejections, got {} \
             (empty FCF must not cause HiGHS to reject the warm-start basis)",
            iter2_fwd_agg.basis_consistency_failures
        );

        // Safety invariant: LP optimum must be finite even with a cold-start and
        // an empty FCF (no cuts from phase 1 are active).
        assert!(
            result2.final_lb.is_finite(),
            "full_churn: final_lb must be finite after full-churn iteration, got {}",
            result2.final_lb
        );
    }
}

// ---------------------------------------------------------------------------
// Test 4: Simulation smoke — baked-path simulate completes with zero failures
// ---------------------------------------------------------------------------

/// Smoke test: after 2-iteration training on D03, baked-path simulation
/// with the trained `basis_cache` produces zero `basis_consistency_failures`.
///
/// ## What this guards
///
/// This test verifies that the simulation baked path correctly runs
/// `reconstruct_basis` on every stage/scenario and that all reconstructed
/// bases are accepted by `HiGHS`.  A regression that delivers an inconsistent
/// warm-start basis will fail this assertion via a non-zero rejection count.
///
/// ## On `basis_reconstructions` in baked simulation
///
/// On the baked path all cut rows are structural rows embedded in
/// `StageTemplate`.  `reconstruct_basis` is called once per warm-start solve,
/// incrementing `basis_reconstructions` throughout the simulation.  The
/// simulation warms-starts correctly — the template's structural rows carry the
/// basis status implicitly through `CapturedBasis.basis`.
///
/// ## Setup
///
/// - D03 case (3 stages, 2 cascaded hydros).
/// - 2 training iterations with 2 forward passes.
/// - Simulation runs with the trained `baked_templates` and `basis_cache`.
///
/// ## Assertion
///
/// `basis_consistency_failures == 0` (reconstructed bases must always be
/// accepted by `HiGHS` on the baked path).
#[test]
fn simulate_baked_path_zero_consistency_failures() {
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
    // 2 scenarios is sufficient without being slow.
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

    // Verify training produced baked templates and a non-empty basis_cache.
    let baked_templates =
        outcome.result.baked_templates.as_deref().expect(
            "simulate_warm_start: training must produce baked_templates after >= 2 iterations",
        );
    let basis_cache = &outcome.result.basis_cache;
    assert!(
        basis_cache
            .iter()
            .any(|cb| cb.as_ref().is_some_and(|b| !b.cut_row_slots.is_empty())),
        "simulate_warm_start: at least one stage must have a CapturedBasis with \
         non-empty cut_row_slots in basis_cache after 2 iterations"
    );

    // Build a simulation workspace pool (1 thread, HighsSolver).
    let mut pool = setup
        .create_workspace_pool(&comm, 1, HighsSolver::new)
        .expect("simulation workspace pool must build");

    let io_capacity = setup.io_channel_capacity().max(1);
    let (result_tx, result_rx) = mpsc::sync_channel(io_capacity);

    // Drain simulation results on a background thread to avoid channel backpressure.
    let drain_handle = std::thread::spawn(move || result_rx.into_iter().count());

    // Use the baked templates from training.  All cuts are structural rows in the
    // template; `reconstruct_basis` is called once per warm-start solve on this path.
    // The critical assertion is that zero bases are rejected by HiGHS.
    let sim_result = setup
        .simulate(
            &mut pool.workspaces,
            &comm,
            &result_tx,
            None,
            Some(baked_templates),
            basis_cache,
        )
        .expect("simulate must return Ok");

    drop(result_tx);
    drain_handle.join().expect("drain thread must not panic");

    let total_rejections: u64 = sim_result
        .solver_stats
        .iter()
        .map(|(_, _, delta)| delta.basis_consistency_failures)
        .sum();

    assert_eq!(
        total_rejections, 0,
        "simulate_warm_start: expected 0 basis_consistency_failures in baked-path simulation, \
         got {total_rejections} (reconstructed bases must always be accepted by HiGHS)"
    );
}
