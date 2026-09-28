//! Node-native enumerated backward gate suite: a deterministic-trunk +
//! terminal-fan enumerated study must do LINEAR backward work in the fan
//! width, never quadratic.
//!
//! [`trunk_fan_setup_enumerated`]'s `t_trunk` deterministic single-successor
//! trunk nodes feed one terminal fan of `k` distinct leaves off the last
//! trunk node. Every non-leaf node (`n_nonleaf_nodes`, one per trunk stage) is
//! solved and cut exactly once per iteration: `Σ backward lp_solves == k +
//! (t_trunk - 1)` (one solve per (non-leaf node, opening), never `k²`) and
//! `Σ cuts_added == n_nonleaf_nodes`. The bound closes to the extensive-form
//! optimum, and the result is bitwise reproducible across thread shapes.
//!
//! # Backward `lp_solves` reconciliation
//!
//! `solver_stats_log`'s `phase == "backward"` rows (summed) and
//! `TrainingEvent::IterationSummary::lp_solves - forward - lower_bound` are two
//! independently-derived views of the same count: the former reads
//! `BackwardResult::stage_stats` directly (one `(rank, worker_id, 0, delta)`
//! entry per visited successor stage on the node-native path — mirroring how
//! `tests/branching_value_oracle.rs`'s `bwd_solves += entry.delta.lp_solves`
//! accumulation reads the sampled path's own `stage_stats`), the latter derives
//! it from `TrainingEvent::IterationSummary::lp_solves` (
//! `forward_result.lp_solves + backward_result.lp_solves + lb_lp_solves`,
//! `TrainingSession::run_iteration`) minus the forward and lower-bound phases'
//! own directly-observed sums. `IterationAccounting::backward_lp_solves`
//! (indirect) and `IterationAccounting::backward_lp_solves_direct` must agree
//! every iteration — `enumerated_backward_is_linear_and_bound_closes` asserts
//! it.
//!
//! # Sampled sibling for the scheduler axis
//!
//! `Traversal::Enumerated` always dispatches to the node-native backward
//! regardless of `backward_scheduler` — `by_scenario`/`by_node` is a
//! `Traversal::Sampled`-only axis. [`trunk_fan_setup`] (the SAME graph,
//! trained `sampled`) gives the by-scenario/by-node bound-close comparison a
//! fixture it genuinely exercises, mirroring how
//! `tests/branching_value_oracle.rs`'s by-node cases already use the sampled
//! `k_fan_setup`, never `k_fan_setup_enumerated`.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::float_cmp,
    clippy::panic,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

mod common;

use std::collections::BTreeMap;
use std::sync::mpsc;

use cobre_core::{TrainingEvent, WorkerTimingPhase};
use cobre_io::config::BackwardScheduler;
use cobre_sddp::StudySetup;
use cobre_sddp::setup::NodePos;
use cobre_sddp::test_support::{
    extensive_form_optimum, trunk_fan_setup, trunk_fan_setup_enumerated,
};
use cobre_solver::ActiveSolver;

use common::StubComm;

/// Relative + absolute LP tolerance for oracle-vs-engine value comparison
/// (mirrors `tests/branching_value_oracle.rs`/`tests/extensive_form_oracle.rs`):
/// both LP backends' feasibility tolerances default to `≈ 1e-9`; scaled by the
/// objective magnitude this bounds the gap between two independent
/// formulations. Never bit-equality.
const REL_TOL: f64 = 1e-6;
const ABS_TOL: f64 = 1e-4;

/// `true` when `a` and `b` agree within [`REL_TOL`]·|scale| + [`ABS_TOL`].
fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= ABS_TOL + REL_TOL * a.abs().max(b.abs())
}

/// Trunk depth and fan width for every gate in this file — small enough for
/// the default suite's build cost, large enough that `k²` is a strong
/// separation from the node-native closed form (`k + (t_trunk - 1)`).
const T_TRUNK: usize = 4;
const FAN_K: usize = 8;
const MAX_ITERATIONS: u32 = 30;
const SAMPLED_FORWARD_PASSES: u32 = 6;

/// Per-iteration accounting, keyed by iteration, filled from one `train()`
/// call's event channel and `solver_stats_log`. `backward_lp_solves` is
/// derived; `backward_lp_solves_direct` reads `phase == "backward"` rows —
/// see this file's module doc. `forward_wall_ms`/`backward_wall_ms` sum every
/// `TrainingEvent::WorkerTiming` event's own phase wall for the iteration —
/// the same per-`(iteration, rank, worker_id)` summation
/// `build_worker_timing_records` performs when it fills
/// `training/timing/iterations.parquet`.
#[derive(Default, Clone, Copy)]
struct IterationAccounting {
    total_lp_solves: u64,
    forward_lp_solves: u64,
    lower_bound_lp_solves: u64,
    backward_lp_solves_direct: u64,
    cuts_added: u32,
    forward_wall_ms: f64,
    backward_wall_ms: f64,
}

impl IterationAccounting {
    fn backward_lp_solves(&self) -> u64 {
        self.total_lp_solves - self.forward_lp_solves - self.lower_bound_lp_solves
    }
}

/// Train `setup` single-rank with `threads` worker threads, returning
/// `(final_lb, final_ub, per_iteration_accounting)`.
fn train_and_collect(
    setup: &mut StudySetup,
    threads: usize,
) -> (f64, f64, BTreeMap<u64, IterationAccounting>) {
    let comm = StubComm;
    let mut solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");
    let (event_tx, event_rx) = mpsc::channel::<TrainingEvent>();
    let outcome = setup
        .train(
            &mut solver,
            &comm,
            threads,
            ActiveSolver::new,
            Some(event_tx),
            None,
        )
        .expect("training must return Ok");
    assert!(
        outcome.error.is_none(),
        "training must not error: {:?}",
        outcome.error
    );

    let mut by_iter: BTreeMap<u64, IterationAccounting> = BTreeMap::new();
    for event in event_rx.try_iter() {
        match event {
            TrainingEvent::IterationSummary {
                iteration,
                lp_solves,
                ..
            } => {
                by_iter.entry(iteration).or_default().total_lp_solves = lp_solves;
            }
            TrainingEvent::BackwardPassComplete {
                iteration,
                rows_generated,
                ..
            } => {
                by_iter.entry(iteration).or_default().cuts_added = rows_generated;
            }
            TrainingEvent::WorkerTiming {
                iteration,
                phase,
                timings,
                ..
            } => {
                let acc = by_iter.entry(iteration).or_default();
                match phase {
                    WorkerTimingPhase::Forward => acc.forward_wall_ms += timings.forward_wall_ms,
                    WorkerTimingPhase::Backward => {
                        acc.backward_wall_ms += timings.backward_wall_ms;
                    }
                }
            }
            _ => {}
        }
    }
    for entry in &outcome.result.solver_stats_log {
        let acc = by_iter.entry(entry.iteration).or_default();
        match entry.phase {
            "forward" => acc.forward_lp_solves += entry.delta.lp_solves,
            "lower_bound" => acc.lower_bound_lp_solves += entry.delta.lp_solves,
            "backward" => acc.backward_lp_solves_direct += entry.delta.lp_solves,
            _ => {}
        }
    }

    (outcome.result.final_lb, outcome.result.final_ub, by_iter)
}

// ── Fast, unignored: the fixture's own structural power self-check ──────────

#[test]
fn trunk_fan_fixture_has_the_declared_shape() {
    let fixture = trunk_fan_setup_enumerated(T_TRUNK, FAN_K, MAX_ITERATIONS);
    assert_eq!(fixture.t_trunk, T_TRUNK);
    assert_eq!(fixture.k, FAN_K);
    assert_eq!(
        fixture.n_nonleaf_nodes, T_TRUNK,
        "every trunk node must be non-leaf; the last one owns the terminal fan"
    );

    let g = &fixture.setup.inputs.node_graph;
    let leaf_count = (0..g.nodes.len())
        .map(NodePos)
        .filter(|&pos| g.successors[pos].is_empty())
        .count();
    assert_eq!(
        leaf_count, FAN_K,
        "the terminal fan must have exactly k leaves"
    );
}

// ── Enumerated leg: the headline linear-work + bound gate ────

/// The headline regression gate: trains the trunk+fan fixture `enumerated`
/// and asserts, per iteration, that backward `lp_solves` equals the
/// node-native closed form `k + (t_trunk - 1)` — exactly, never a hard-coded
/// literal — and stays strictly below `k²` (the O(fan²) regression this gate
/// exists to catch); `cuts_added` equals `n_nonleaf_nodes` exactly; and the
/// final bounds close to the extensive-form optimum.
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "trains the trunk+fan fixture to convergence; slow-tests only"
)]
#[test]
fn enumerated_backward_is_linear_and_bound_closes() {
    let fixture = trunk_fan_setup_enumerated(T_TRUNK, FAN_K, MAX_ITERATIONS);
    let optimum = extensive_form_optimum(&fixture.setup);
    let mut setup = fixture.setup;

    // Closed form, derived from the fixture's own T_TRUNK/K — never a
    // hard-coded literal.
    let expected_solves = fixture.k as u64 + (fixture.t_trunk as u64 - 1);
    let expected_cuts = fixture.n_nonleaf_nodes as u32;
    let fan_squared = (fixture.k as u64) * (fixture.k as u64);
    assert!(
        expected_solves < fan_squared,
        "power: the closed form itself must sit strictly below k² on this fixture \
         (expected {expected_solves} vs k² {fan_squared}), or the regression check below \
         is vacuous"
    );

    let (lb, ub, by_iter) = train_and_collect(&mut setup, 1);
    assert!(
        !by_iter.is_empty(),
        "power: training must run at least one iteration"
    );

    for (&iteration, acc) in &by_iter {
        let solves = acc.backward_lp_solves();
        assert_eq!(
            solves, expected_solves,
            "iteration {iteration}: backward lp_solves {solves} != k+(t_trunk-1) \
             {expected_solves}"
        );
        assert_eq!(
            acc.backward_lp_solves_direct, solves,
            "iteration {iteration}: direct backward-phase stage_stats sum {} must \
             reconcile with the indirect total-forward-lower_bound derivation {solves}",
            acc.backward_lp_solves_direct
        );
        assert!(
            solves < fan_squared,
            "iteration {iteration}: backward lp_solves {solves} must stay below k² \
             {fan_squared} — the O(fan²) regression this gate exists to catch"
        );
        assert_eq!(
            acc.cuts_added, expected_cuts,
            "iteration {iteration}: cuts_added {} != n_nonleaf_nodes {expected_cuts}",
            acc.cuts_added
        );
        // The enumerated engines must emit non-zero WorkerTiming walls — the
        // values `training/timing/iterations.parquet` recovers via its
        // per-(iteration, rank, worker_id) SUM — not the all-zero rows an
        // absent emit would leave.
        assert!(
            acc.forward_wall_ms > 0.0,
            "iteration {iteration}: summed WorkerTiming forward_wall_ms must be non-zero on \
             the enumerated forward, got {}",
            acc.forward_wall_ms
        );
        assert!(
            acc.backward_wall_ms > 0.0,
            "iteration {iteration}: summed WorkerTiming backward_wall_ms must be non-zero on \
             the enumerated backward, got {}",
            acc.backward_wall_ms
        );
    }

    assert!(
        close(lb, optimum),
        "enumerated trunk+fan final_lb {lb} must equal extensive-form optimum {optimum} \
         (gap {})",
        lb - optimum
    );
    assert!(
        close(ub, optimum),
        "enumerated trunk+fan final_ub {ub} must equal extensive-form optimum {optimum} \
         (gap {})",
        ub - optimum
    );
}

// ── Sampled leg: bound closes under both backward schedulers ──────────

/// `Traversal::Enumerated` ignores `backward_scheduler` entirely (it always
/// dispatches to the node-native backward), so the by-scenario/by-node
/// comparison runs on [`trunk_fan_setup`]'s sampled sibling — the same graph,
/// genuinely exercising both schedulers.
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "trains the sampled trunk+fan fixture twice to convergence; slow-tests only"
)]
#[test]
fn sampled_trunk_fan_bound_closes_under_by_scenario_and_by_node() {
    let by_scenario_fixture =
        trunk_fan_setup(T_TRUNK, FAN_K, SAMPLED_FORWARD_PASSES, MAX_ITERATIONS);
    // Power self-check: the terminal fan has >= 2 successors off one node, so
    // the by-node opening-block claim loop genuinely spans more than one child.
    assert!(
        by_scenario_fixture.k >= 2,
        "power: the terminal fan must have >= 2 leaves"
    );
    let optimum = extensive_form_optimum(&by_scenario_fixture.setup);
    let mut by_scenario_setup = by_scenario_fixture.setup;
    by_scenario_setup.set_scheduler(BackwardScheduler::ByScenario {});
    let (lb_by_scenario, ub_by_scenario, _) = train_and_collect(&mut by_scenario_setup, 1);

    let mut by_node_setup =
        trunk_fan_setup(T_TRUNK, FAN_K, SAMPLED_FORWARD_PASSES, MAX_ITERATIONS).setup;
    by_node_setup.set_scheduler(BackwardScheduler::ByNode { block_size: None });
    let (lb_by_node, ub_by_node, _) = train_and_collect(&mut by_node_setup, 1);

    assert!(
        close(lb_by_scenario, optimum),
        "by_scenario trunk+fan final_lb {lb_by_scenario} must equal extensive-form optimum \
         {optimum} (gap {})",
        lb_by_scenario - optimum
    );
    assert!(
        close(ub_by_scenario, optimum),
        "by_scenario trunk+fan final_ub {ub_by_scenario} must equal extensive-form optimum \
         {optimum} (gap {})",
        ub_by_scenario - optimum
    );
    assert!(
        close(lb_by_node, optimum),
        "by_node trunk+fan final_lb {lb_by_node} must equal extensive-form optimum {optimum} \
         (gap {})",
        lb_by_node - optimum
    );
    assert!(
        close(ub_by_node, optimum),
        "by_node trunk+fan final_ub {ub_by_node} must equal extensive-form optimum {optimum} \
         (gap {})",
        ub_by_node - optimum
    );
}

// ── Thread-shape reproducibility ────────────────────────────────────────

/// `final_lb`/`final_ub` must be bitwise identical across `--threads 1`/`2`/`4`
/// on the enumerated trunk+fan fixture (the satisfiable thread-shape axis;
/// genuine world>=2 rank-shape bit-invariance rides the real-MPI SLURM job —
/// in-process 2-rank genuine-fan invariance is structurally unsatisfiable, so
/// there is no `Rank0Of2` stub here).
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "trains the trunk+fan fixture three times to convergence; slow-tests only"
)]
#[test]
fn enumerated_final_bounds_are_thread_shape_invariant() {
    let mut one = trunk_fan_setup_enumerated(T_TRUNK, FAN_K, MAX_ITERATIONS).setup;
    let mut two = trunk_fan_setup_enumerated(T_TRUNK, FAN_K, MAX_ITERATIONS).setup;
    let mut four = trunk_fan_setup_enumerated(T_TRUNK, FAN_K, MAX_ITERATIONS).setup;

    let (lb1, ub1, _) = train_and_collect(&mut one, 1);
    let (lb2, ub2, _) = train_and_collect(&mut two, 2);
    let (lb4, ub4, _) = train_and_collect(&mut four, 4);

    for (label_a, lb_a, ub_a, label_b, lb_b, ub_b) in [
        ("1", lb1, ub1, "2", lb2, ub2),
        ("1", lb1, ub1, "4", lb4, ub4),
        ("2", lb2, ub2, "4", lb4, ub4),
    ] {
        assert_eq!(
            lb_a.to_bits(),
            lb_b.to_bits(),
            "final_lb must be bit-identical across --threads {label_a} ({lb_a}) and \
             --threads {label_b} ({lb_b})"
        );
        assert_eq!(
            ub_a.to_bits(),
            ub_b.to_bits(),
            "final_ub must be bit-identical across --threads {label_a} ({ub_a}) and \
             --threads {label_b} ({ub_b})"
        );
    }
}
