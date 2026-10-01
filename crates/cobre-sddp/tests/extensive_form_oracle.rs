//! Extensive-form LP oracle: closes node-native `enumerated` training's
//! `final_lb`/`final_ub` to the graph's true first-stage value on three
//! validation fixtures — (a) 2-stage K-fan (the DECOMP shape), (b) 3-stage
//! binary tree branching at two stages, (d) chain.
//!
//! A fourth fixture, a fan-then-recombine DAG (a node reached from two or more
//! parent nodes), is out of scope here: `setup::reject_recombining_node_enumeration`
//! hard-rejects any node graph with in-degree `>= 2` under `enumerated` selection
//! (`crates/cobre-sddp/src/setup/mod.rs`) — per-prefix state reconstruction for a
//! multi-parent node is a reserved seam, not built, and this binary's own
//! `extensive_form_optimum` adoption (below) would need the same prefix-aware
//! rewrite before it could expand one either. Tracked separately; not attempted.
//!
//! # The oracle
//!
//! [`extensive_form_optimum`] (`cobre_sddp::test_support`, adopted unchanged from
//! `branching_value_oracle.rs`'s harness — not re-derived here) expands one LP
//! column block per graph node. On these `|Ω| = 1`-per-node trees every node has
//! exactly one predecessor, so one column block per node IS one column block per
//! reachable `(node, realization)` prefix; [`assert_reachable_prefixes_match`]
//! checks that precondition before any bound comparison runs. Non-anticipativity is
//! structural: a parent's outgoing-state column is tied to its child's
//! incoming-state column by one equality row per state dimension, so no
//! non-anticipativity constraint row exists and none can be mis-signed.
//!
//! # Fixture (b): the headline
//!
//! [`branching_tree_setup_enumerated`] trains node-native `enumerated` over a
//! 3-stage binary tree branching at BOTH interior stages (root -> 2 -> 4 leaves)
//! under non-uniform weights — the shape a shape-based enumerated-execution
//! admission clause would have rejected outright. Its `final_lb`/`final_ub`
//! closing to the extensive-form optimum is the empirical discharge of that
//! claim, and it runs in the same (un-`#[ignore]`d) mode as fixtures (a)/(d).
//!
//! # LP tolerance
//!
//! [`REL_TOL`]/[`ABS_TOL`] mirror `branching_value_oracle.rs`: both LP backends'
//! `primal_feasibility_tolerance`/`dual_feasibility_tolerance` default to `1e-9`
//! (`cobre-solver`'s `clp`/`highs` backend configs); scaled by the objective
//! magnitude, this bounds the floating-point gap between the extensive-form LP and
//! the node-native engine's own accumulated solves — two independent
//! formulations, never bit equality.

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

use cobre_sddp::StudySetup;
use cobre_sddp::setup::{
    NodeGraph, NodeId, NodeOpenings, NodePos, NodeRuntime, NodeSuccessor, OpeningSource, StageIdx,
};
use cobre_sddp::test_support::{
    branching_tree_setup_enumerated, extensive_form_optimum, k_fan_setup_enumerated,
    node_prefix_counts, oracle_chain_setup, single_path_enumerated_setup,
};
use cobre_solver::ActiveSolver;

use common::StubComm;

/// Relative + absolute LP tolerance for oracle-vs-engine value comparison. The
/// backend primal/dual feasibility tolerance is `≈ 1e-9`; scaled by the objective
/// magnitude this bounds the gap between the two formulations. Never bit-equality —
/// this is a value oracle across two different LP encodings.
const REL_TOL: f64 = 1e-6;
const ABS_TOL: f64 = 1e-4;

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= ABS_TOL + REL_TOL * a.abs().max(b.abs())
}

/// Train `setup` single-rank/single-thread to convergence, returning
/// `(final_lb, final_ub)`.
fn train_bounds(setup: &mut StudySetup) -> (f64, f64) {
    let comm = StubComm;
    let mut solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");
    let outcome = setup
        .train(&mut solver, &comm, 1, ActiveSolver::new, None, None)
        .expect("training must return Ok");
    assert!(
        outcome.error.is_none(),
        "training must not error: {:?}",
        outcome.error
    );
    (outcome.result.final_lb, outcome.result.final_ub)
}

/// Assert `graph`'s reachable `(node, realization)` prefix set matches
/// [`extensive_form_optimum`]'s one-column-block-per-node expansion: every node
/// must be reached by EXACTLY one root-to-node prefix
/// ([`node_prefix_counts`] `== 1` everywhere), which is what makes "one LP column
/// block per node" the correct "one LP column block per reachable prefix"
/// expansion. Panics with a descriptive message on any mismatch, so a fixture
/// whose graph carries a recombination join (or any other multi-prefix node)
/// fails loudly rather than silently comparing the oracle's expansion against a
/// different problem.
///
/// # Panics
///
/// Panics if the prefix-count vector's length disagrees with the graph's node
/// count, if any node is reached by other than exactly one prefix, or if the
/// prefix-count total disagrees with the node count.
fn assert_reachable_prefixes_match(graph: &NodeGraph) {
    let n = graph.nodes.len();
    let prefix_counts = node_prefix_counts(graph)
        .expect("assert_reachable_prefixes_match: node prefix counts must not overflow");
    assert_eq!(
        prefix_counts.len(),
        n,
        "reachable-prefix count entries ({}) must equal the graph's node count ({n})",
        prefix_counts.len()
    );
    for (pos, &count) in prefix_counts.iter().enumerate() {
        assert!(
            count == 1,
            "node at canonical position {pos} is reached by {count} distinct \
             root-to-node prefixes, not exactly one: extensive_form_optimum \
             allocates a single LP column block per node, which expands the \
             correct problem only when every node has exactly one predecessor \
             (this graph carries a recombination join)"
        );
    }
    let prefix_total: u64 = prefix_counts.iter().sum();
    assert_eq!(
        prefix_total as usize, n,
        "reachable-prefix total ({prefix_total}) must equal the graph's node count \
         ({n}) — extensive_form_optimum's node-indexed column allocation is its \
         prefix allocation only under this equality"
    );
}

/// Cross-check the reachable prefixes, solve the extensive-form LP, train
/// `enumerated` to convergence, and assert both `final_lb` and `final_ub` close to
/// the LP optimum within [`REL_TOL`]/[`ABS_TOL`].
fn assert_closes_to_extensive_form(mut setup: StudySetup, fixture: &str) {
    assert_reachable_prefixes_match(&setup.inputs.node_graph);
    let optimum = extensive_form_optimum(&setup);
    let (lb, ub) = train_bounds(&mut setup);
    assert!(
        close(lb, optimum),
        "{fixture} final_lb {lb} must equal extensive-form optimum {optimum} (gap {})",
        lb - optimum
    );
    assert!(
        close(ub, optimum),
        "{fixture} final_ub {ub} must equal extensive-form optimum {optimum} (gap {})",
        ub - optimum
    );
}

// ── Three-fixture convergence gate ──────────────────────────────────────

#[test]
fn k_fan_matches_extensive_form() {
    let fixture = k_fan_setup_enumerated(3, 30);
    assert_closes_to_extensive_form(fixture.setup, "K-fan");
}

/// The headline: a 3-stage binary tree branching at TWO stages — the shape a
/// shape-based enumerated-execution admission clause would have rejected. Not
/// `#[ignore]`d; runs in the same mode as the other fixtures.
#[test]
fn branching_tree_matches_extensive_form() {
    let setup = branching_tree_setup_enumerated(30);
    assert_closes_to_extensive_form(setup, "3-stage binary tree");
}

#[test]
fn chain_matches_extensive_form() {
    let setup = oracle_chain_setup(30);
    assert_closes_to_extensive_form(setup, "chain");
}

// ── Reachable-prefix cross-check failure path ───────────────────────────

/// A 4-node fan-then-recombine `NodeGraph`, hand-built (bypassing `StudySetup`,
/// which `enumerated`'s own admission guard would reject before construction
/// completes — see this file's module doc): root (position 0) fans into two
/// nodes (1, 2), both of which point at a shared node (3) — a recombination
/// join, `node_prefix_counts()[3] == 2`.
fn recombination_join_graph() -> NodeGraph {
    let openings = NodeOpenings {
        source: OpeningSource::Generated,
        offset: 0,
        len: 1,
        q: 1.0,
    };
    let node = |stage: usize, pool_id: usize| NodeRuntime {
        stage: StageIdx(stage),
        pool_id,
        openings,
    };
    let succ = |child: usize, probability: f64| NodeSuccessor {
        child: NodePos(child),
        probability,
    };
    NodeGraph {
        node_ids: (0..4_i32).map(NodeId).collect(),
        nodes: vec![node(0, 0), node(1, 1), node(1, 2), node(2, 3)].into(),
        successors: vec![
            vec![succ(1, 0.5), succ(2, 0.5)],
            vec![succ(3, 1.0)],
            vec![succ(3, 1.0)],
            vec![],
        ]
        .into(),
        n_pools: 4,
        pool_stage: vec![StageIdx(0), StageIdx(1), StageIdx(1), StageIdx(2)],
    }
}

#[test]
#[should_panic(expected = "distinct root-to-node prefixes")]
fn reachable_prefix_cross_check_panics_on_recombination_join() {
    assert_reachable_prefixes_match(&recombination_join_graph());
}

// ── Unit test: hand-checkable tiny graph ─────────────────────────────────────

/// The oracle on a hand-checkable 2-stage chain
/// ([`single_path_enumerated_setup`]'s system, built via
/// `PrepareHydroModelsResult::default_from_system`, which resolves every hydro's
/// `ConstantProductivity` at the documented `0.0` placeholder — no
/// `hydro_production_models.json` backs this fixture, so the plant can never
/// generate): with generation always `0`, the `80` MW load is met entirely by
/// deficit at `500` `$/MWh` on both `744`-hour stages, and turbining any water
/// that would otherwise spill is free (`turbined_cost 0.0`) and always within
/// the `100` m³/s turbine cap, so no spillage cost accrues either — the
/// closed-form optimum is exactly `500.0 * 80.0 * 744.0 * 2.0 = 59_520_000.0`.
#[test]
fn two_stage_chain_extensive_form_matches_deficit_closed_form() {
    let setup = single_path_enumerated_setup(30);
    assert_reachable_prefixes_match(&setup.inputs.node_graph);
    let optimum = extensive_form_optimum(&setup);
    let closed_form = 500.0 * 80.0 * 744.0 * 2.0;
    assert!(
        close(optimum, closed_form),
        "hand-checkable 2-stage chain extensive-form optimum {optimum} must equal \
         the closed-form deficit cost {closed_form}"
    );
}
