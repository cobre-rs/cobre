//! Backward-basis-cache hit-rate regression gate (Epic 02, ticket-004).
//!
//! Verifies that the backward-basis cache fires at an acceptable rate on D01
//! end-to-end: at least 95 % of iter ≥ 2 ω=0 backward rows must carry
//! `basis_source == BasisSource::Backward`.
//!
//! This test is the CI regression gate for the cache: if a future refactor
//! silently disables the backward store read path, the hit rate drops to near
//! zero and this test fails.
//!
//! ## What is tested
//!
//! - `cache_hit_rate >= 0.95` on all `(iter >= 2, phase == "backward",
//!   opening == 0)` rows in `TrainingResult::solver_stats_log`.
//! - Cold-start invariant: every `(iter == 1, phase == "backward",
//!   opening == 0)` row has `basis_source == BasisSource::Forward` (the cache
//!   is empty on the first iteration; every ω=0 solve falls back to the
//!   forward store).
//!
//! ## Fixture
//!
//! D01 (`examples/deterministic/d01-thermal-dispatch`) is the simplest
//! deterministic case.  It trains for a small number of iterations and
//! produces a small parquet — appropriate for a tight CI regression test.
//! The test runs single-rank via `StubComm` (no MPI dependency) and uses
//! the default D01 iteration count.
//!
//! ## Failure triage
//!
//! If `cache_hit_rate < 0.95`:
//!
//! 1. Check that `solver_stats_log` contains backward entries with `opening ==
//!    0` for D01's single opening — if the filter returns zero rows the
//!    cache-hit assertion will panic with a clear message.
//! 2. Confirm the stored backward basis returns `Some(...)` from
//!    iteration 2 onwards (add `RUST_LOG=debug` to training).
//! 3. Verify that the backward read site in `process_trial_point_backward`
//!    sets `delta.basis_source = BasisSource::Backward` when reading from the
//!    backward-pass basis cache.
//! 4. Confirm that the backward-pass basis cache was populated after iteration 1
//!    and the updated cache is visible before iteration 2's backward pass begins.
//!
//! If the cold-start assertion fires (`iter == 1` row has non-Forward source):
//!
//! 1. Verify that the backward-pass basis cache is empty before the first backward pass.
//! 2. Check that the stored backward basis returns `None` when the
//!    cache is freshly initialised.

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
use cobre_sddp::{
    StudySetup, hydro_models::prepare_hydro_models, setup::prepare_stochastic,
    solver_stats::BasisSource,
};
use cobre_solver::highs::HighsSolver;

// ---------------------------------------------------------------------------
// Case path constant
// ---------------------------------------------------------------------------

const D01_CASE: &str = "examples/deterministic/d01-thermal-dispatch";

// ---------------------------------------------------------------------------
// Cache-hit rate threshold
//
// A threshold of 0.95 is robust to the rare two-store miss (both backward and
// forward stores return None at iter >= 2, producing basis_source = None_).
// An exact threshold of 1.0 would be fragile; 0.95 tolerates occasional misses
// without masking a cache that has stopped firing entirely.
// ---------------------------------------------------------------------------

const CACHE_HIT_RATE_THRESHOLD: f64 = 0.95;

// ---------------------------------------------------------------------------
// Stub communicator (single-rank, no MPI)
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
// Path helper
// ---------------------------------------------------------------------------

fn d01_case_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/<crate> must have a parent")
        .parent()
        .expect("crates/ must have a parent (repo root)")
        .join(D01_CASE)
}

// ---------------------------------------------------------------------------
// Test
// ---------------------------------------------------------------------------

/// Asserts that the backward-basis cache fires at a >= 95 % hit rate on D01
/// iter >= 2 ω=0 backward rows, and that all iter=1 ω=0 backward rows carry
/// `basis_source == BasisSource::Forward` (cold-start invariant).
///
/// The cache (Epic 01) stores rank-0's m=0 backward-pass basis after each
/// iteration and broadcasts it to all ranks.  Starting from iteration 2, the
/// ω=0 backward solve warm-starts from this cached basis (`BasisSource::Backward`).
/// Iteration 1 always falls back to the forward store (`BasisSource::Forward`)
/// because the cache is empty until after the first backward pass completes.
///
/// ## `SolverStatsEntry` tuple layout
///
/// `(iteration: u64, phase: &'static str, stage: i32, opening: i32,
///  rank: i32, worker_id: i32, delta: SolverStatsDelta)`
///
/// - `phase == "backward"` and `opening == 0` selects ω=0 backward rows.
/// - `delta.basis_source` carries the `BasisSource` discriminant for that row.
#[test]
fn test_backward_cache_hit_rate() {
    let case_dir = d01_case_dir();

    assert!(
        case_dir.exists(),
        "D01 case directory not found at {}. \
         Run `cargo test` from the repo root so that `CARGO_MANIFEST_DIR` \
         resolves to `crates/cobre-sddp` and the relative path \
         `examples/deterministic/d01-thermal-dispatch` is reachable.",
        case_dir.display()
    );

    let config_path = case_dir.join("config.json");
    let mut config = cobre_io::parse_config(&config_path).expect("config must parse");

    // Use the default D01 config.  Disable cut selection so that the test
    // exercises the base training path rather than the selection variant.
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
    let (tx, _rx) = mpsc::channel();
    let mut solver = HighsSolver::new().expect("HighsSolver::new must succeed");

    let outcome = setup
        .train(&mut solver, &comm, 1, HighsSolver::new, Some(tx), None)
        .expect("train must succeed");

    assert!(
        outcome.error.is_none(),
        "test_backward_cache_hit_rate: unexpected training error: {:?}",
        outcome.error
    );

    let result = outcome.result;

    // ------------------------------------------------------------------
    // Assertion 1: cache-hit rate >= CACHE_HIT_RATE_THRESHOLD on iter >= 2
    // ------------------------------------------------------------------
    //
    // Filter to all backward ω=0 (opening == 0) rows at iteration >= 2.
    // SolverStatsEntry is a 7-tuple:
    //   (iteration, phase, stage, opening, rank, worker_id, delta)

    let total_iter2_backward_omega0 = result
        .solver_stats_log
        .iter()
        .filter(|(iter, phase, _stage, opening, _rank, _wid, _delta)| {
            *iter >= 2 && *phase == "backward" && *opening == 0
        })
        .count();

    assert!(
        total_iter2_backward_omega0 > 0,
        "test_backward_cache_hit_rate: no backward ω=0 iter >= 2 entries found \
         in solver_stats_log — check that the backward pass emits per-opening entries \
         and that D01 runs for at least 2 iterations"
    );

    let backward_hits = result
        .solver_stats_log
        .iter()
        .filter(|(iter, phase, _stage, opening, _rank, _wid, delta)| {
            *iter >= 2
                && *phase == "backward"
                && *opening == 0
                && delta.basis_source == BasisSource::Backward
        })
        .count();

    let forward_fallbacks = result
        .solver_stats_log
        .iter()
        .filter(|(iter, phase, _stage, opening, _rank, _wid, delta)| {
            *iter >= 2
                && *phase == "backward"
                && *opening == 0
                && delta.basis_source == BasisSource::Forward
        })
        .count();

    let none_count = result
        .solver_stats_log
        .iter()
        .filter(|(iter, phase, _stage, opening, _rank, _wid, delta)| {
            *iter >= 2
                && *phase == "backward"
                && *opening == 0
                && delta.basis_source == BasisSource::None_
        })
        .count();

    let cache_hit_rate = backward_hits as f64 / total_iter2_backward_omega0 as f64;

    assert!(
        cache_hit_rate >= CACHE_HIT_RATE_THRESHOLD,
        "test_backward_cache_hit_rate: cache hit rate too low on D01 iter >= 2 ω=0 backward rows.\n\
         \n\
         Actual:    {backward_hits}/{total_iter2_backward_omega0} = {cache_hit_rate:.3}\n\
         Threshold: {CACHE_HIT_RATE_THRESHOLD:.3}\n\
         \n\
         Breakdown:\n\
           backward (cache hit) = {backward_hits}\n\
           forward  (fallback)  = {forward_fallbacks}\n\
           none     (miss)      = {none_count}\n\
           total                = {total_iter2_backward_omega0}\n\
         \n\
         Triage:\n\
           (1) Confirm backward log entries have opening == 0 for D01's single opening.\n\
           (2) Verify the stored backward basis returns Some from iter 2 onwards.\n\
           (3) Check that the backward read site sets delta.basis_source = Backward.\n\
           (4) Verify the backward-pass basis cache is updated at end of iter 1."
    );

    // ------------------------------------------------------------------
    // Assertion 2: cold-start invariant — all iter=1 ω=0 backward rows
    // must have basis_source == BasisSource::Forward
    // ------------------------------------------------------------------

    let iter1_entries: Vec<_> = result
        .solver_stats_log
        .iter()
        .filter(|(iter, phase, _stage, opening, _rank, _wid, _delta)| {
            *iter == 1 && *phase == "backward" && *opening == 0
        })
        .collect();

    assert!(
        !iter1_entries.is_empty(),
        "test_backward_cache_hit_rate: cold-start assertion requires at least one \
         iter=1 ω=0 backward row; found zero. Either D01 training emitted no iter=1 \
         backward entries (possible regression), or the solver_stats_log filter \
         semantics changed."
    );

    for (iter, _phase, stage, _opening, _rank, _wid, delta) in &iter1_entries {
        assert_eq!(
            delta.basis_source,
            BasisSource::Forward,
            "test_backward_cache_hit_rate: cold-start invariant violated.\n\
             iter={iter} stage={stage} has basis_source={:?} (expected BasisSource::Forward).\n\
             The backward cache is empty on iter 1; every ω=0 backward solve \
             must fall back to the forward store.",
            delta.basis_source,
        );
    }
}
