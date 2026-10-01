//! Branching value oracle: a **value** instrument for graph branching.
//!
//! Every executed branching signal in the tree today is an *invariance* signal
//! (per-pool cut counts, `k_fan_*` bitwise thread/rank-shape equality). A
//! deterministic **wrong** value passes all of them. This binary adds the missing
//! obligation: it asserts a trained policy's `final_lb` equals the graph's **true**
//! first-stage value, computed independently by the extensive-form LP.
//!
//! On HEAD every branching shape that trains is value-**correct**: chain,
//! terminal-Generated fan, interior-sibling Generated fan, and DCS-arm Generated
//! fan all close `final_lb` to the extensive-form optimum (Generated siblings share
//! per-(hydro, stage) inflow, so they converge to identical pools and the
//! single-successor backward solve prices them correctly). These cases stand as
//! permanent value regression guards. The materially-different-successor shapes
//! (all-External distinct and root fans) now train and are covered here by
//! shape/train-smoke gates only: on HEAD they train to a zero value gap (inflow is
//! non-binding in these fixtures), so their value-red — the one that would expose
//! the child-0 collapse mispricing the fan — is deferred to the engine fix that
//! makes a leaf inflow-binding, not asserted here.
//!
//! # The oracle
//!
//! The extensive-form LP expands one subproblem copy per reachable
//! `(node, realization)` prefix into a single monolithic LP. Non-anticipativity is
//! **structural**: prefixes sharing a history share the same decision columns (on
//! these finite acyclic `|Ω| = 1` trees, one copy per node, the shared ancestor's
//! columns literally one set), so no non-anticipativity constraint row exists and
//! none can be mis-signed. Each node copy is the engine's own patched single-stage
//! LP (`capture_patched_node_template`); a parent's outgoing-state column is tied to
//! its child's incoming-state column by one equality row per state dimension; the
//! root's incoming state keeps the engine's pin to the initial state; each copy's
//! objective is weighted by its path probability. Solved once, this LP's optimum is
//! the graph's true first-stage value, which node-native training must close `LB`
//! (and the exact `UB`) to at LP tolerance.
//!
//! # Control cases prove the harness
//!
//! `chain` and `terminal_generated_fan` are the harness-validation cases (a single
//! successor; interchangeable leaves sharing one pool), and additionally pin
//! `final_lb == final_ub`. If a control fails, the oracle is wrong, not the engine —
//! which is what lets the interior-sibling and DCS-arm fan cases attribute a value
//! disagreement to the engine rather than the expander.
//!
//! # Hand cross-check (chain control)
//!
//! The 3-stage chain is single-hydro / single-bus, storage in `[0, 200]` hm³, start
//! `100` hm³, inflow `60` m³/s, load `80` MW, `ρ = 2.5` MW·s/m³, turbine cap `100`
//! m³/s so `g ≤ 250` MW covers the load with `q = 32` m³/s < inflow. The cheapest
//! feasible policy carries water forward and never deficits; its only cost is the
//! `0.01`/hm³ spillage penalty on water it cannot store (storage cap `200`, inflow
//! surplus `≈ 28` m³/s · `2.6784` hm³/(m³/s) per stage). The extensive-form LP is
//! solved here; this note pins the regime (near-zero cost, no deficit) so a future
//! expander regression that flips into a large-deficit optimum is attributable.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::float_cmp,
    clippy::panic,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::too_many_lines
)]

mod common;

use cobre_io::config::BackwardScheduler;
use cobre_sddp::CutPool;
use cobre_sddp::StudySetup;
use cobre_sddp::hydro_models::ResolvedProductionModel;
use cobre_sddp::setup::{NodePos, OpeningSource, StageIdx};
use cobre_sddp::test_support::{
    dcs_k_fan_setup, extensive_form_optimum, external_distinct_fan_setup,
    external_distinct_fan_setup_heterogeneous_cut_state, external_root_fan_setup, k_fan_setup,
    node_prefix_counts, node_scenario_count, node_visit_probabilities, oracle_chain_setup,
    pool_cut_state_dimensions, terminal_generated_fan_setup, try_k_fan_simulation_enumerated,
    water_binding_external_fan_setup, water_binding_external_fan_setup_reversed,
};
use cobre_solver::ActiveSolver;

use common::{StubComm, run_simulation};

/// Relative + absolute LP tolerance for oracle-vs-engine value comparison. The
/// backend primal/dual feasibility tolerance is `≈ 1e-9`; scaled by the objective
/// magnitude this bounds the gap between the two formulations. Never bit-equality —
/// this is a value oracle across two different LP encodings.
const REL_TOL: f64 = 1e-6;
const ABS_TOL: f64 = 1e-4;

/// `true` when `a` and `b` agree within [`REL_TOL`]·|scale| + [`ABS_TOL`].
fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= ABS_TOL + REL_TOL * a.abs().max(b.abs())
}

/// Train `setup` single-rank with `threads` worker threads, returning
/// `(final_lb, final_ub)`.
fn train_bounds_threads(setup: &mut StudySetup, threads: usize) -> (f64, f64) {
    let comm = StubComm;
    let mut solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");
    let outcome = setup
        .train(&mut solver, &comm, threads, ActiveSolver::new, None, None)
        .expect("training must return Ok");
    assert!(
        outcome.error.is_none(),
        "training must not error: {:?}",
        outcome.error
    );
    (outcome.result.final_lb, outcome.result.final_ub)
}

/// Train `setup` single-rank / single-thread, returning `(final_lb, final_ub)`.
fn train_bounds(setup: &mut StudySetup) -> (f64, f64) {
    train_bounds_threads(setup, 1)
}

/// Train `setup` single-rank / single-thread under the given backward scheduler,
/// returning `(final_lb, final_ub)`. The by-node scheduler prices each fan child
/// against its OWN LP; on these `|Ω| = 1`-per-child oracle trees each child run is
/// a single-opening solve loading its own basis — identical warm-start structure
/// to the by-scenario path — so the two schedulers produce a bit-identical cut set.
fn train_bounds_scheduler(setup: &mut StudySetup, scheduler: BackwardScheduler) -> (f64, f64) {
    setup.set_scheduler(scheduler);
    train_bounds_threads(setup, 1)
}

const BY_NODE: BackwardScheduler = BackwardScheduler::ByNode { block_size: None };

/// Train `setup` single-rank and additionally return the backward phase's total
/// LP solves and warm-basis offers from `solver_stats_log` — the runtime signal
/// that distinguishes the DCS (`Lazy`) backward, which loads a cut-free core and
/// offers no captured frozen basis, from the Frozen path, which warm-starts from a
/// captured basis on every cross-iteration solve.
fn train_backward_basis_signal(setup: &mut StudySetup) -> (f64, u64, u64) {
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
    let result = &outcome.result;
    let mut bwd_solves = 0_u64;
    let mut bwd_basis_offered = 0_u64;
    for entry in &result.solver_stats_log {
        if entry.phase == "backward" {
            bwd_solves += entry.delta.lp_solves;
            bwd_basis_offered += entry.delta.basis_offered;
        }
    }
    (result.final_lb, bwd_solves, bwd_basis_offered)
}

/// Assert the expander's own coverage: its one-column-block-per-node
/// construction matches the graph's own enumerated counts, and the leaf visit
/// probabilities sum to 1.
///
/// The expander emits exactly `g.nodes.len()` node copies. On a finite acyclic
/// `|Ω| = 1` tree every node is reached by exactly one prefix (`π(n) == 1`) and
/// carries one realization (`openings.len == 1`), so the reachable
/// `(node, realization)` prefix count is `Σ π(n) == g.nodes.len()` — one copy per
/// node is the correct expansion. `node_scenario_count` (root→leaf paths) then
/// equals the graph's leaf count. A fixture that violated `|Ω| = 1` or introduced
/// a recombination join would break these and make one-copy-per-node wrong.
fn assert_expander_self_consistent(setup: &StudySetup) {
    let g = &setup.inputs.node_graph;
    let n = g.nodes.len();

    let prefix_counts = node_prefix_counts(g).expect("node prefix counts must not overflow");
    assert_eq!(prefix_counts.len(), n, "one prefix count per node");
    assert!(
        prefix_counts.iter().all(|&c| c == 1),
        "|Ω|=1 tree: every node reached by exactly one prefix, got {prefix_counts:?}"
    );
    assert!(
        g.nodes.iter().all(|node| node.openings.len == 1),
        "|Ω|=1: every node carries exactly one realization, so one copy per node = one per prefix"
    );
    let prefix_total: u64 = prefix_counts.iter().sum();
    assert_eq!(
        prefix_total as usize, n,
        "reachable-prefix count must equal the expander's node-copy count"
    );

    let scenario_count = node_scenario_count(g).expect("scenario count must not overflow");
    let leaf_count = (0..n)
        .map(NodePos)
        .filter(|&pos| g.successors[pos].is_empty())
        .count();
    assert_eq!(
        scenario_count as usize, leaf_count,
        "enumerated scenario count must equal the graph's leaf count on an |Ω|=1 tree"
    );

    let prob = node_visit_probabilities(setup);
    let leaf_mass: f64 = (0..n)
        .map(NodePos)
        .filter(|&pos| g.successors[pos].is_empty())
        .map(|pos| prob[pos.0])
        .sum();
    assert!(
        (leaf_mass - 1.0).abs() < 1e-9,
        "leaf visit probabilities must sum to 1.0, got {leaf_mass}"
    );
}

// ── GREEN control cases (must pass un-ignored) ───────────────────────────────

#[test]
fn chain_control_matches_extensive_form_and_ub() {
    let mut setup = oracle_chain_setup(30);
    assert_expander_self_consistent(&setup);
    let optimum = extensive_form_optimum(&setup);
    let (lb, ub) = train_bounds(&mut setup);
    assert!(
        close(lb, optimum),
        "chain final_lb {lb} must equal extensive-form optimum {optimum} (gap {})",
        lb - optimum
    );
    assert!(
        close(lb, ub),
        "chain final_lb {lb} must equal final_ub {ub} (gap {})",
        lb - ub
    );
}

#[test]
fn terminal_generated_fan_control_matches_extensive_form_and_ub() {
    let k = 3;
    let mut setup = terminal_generated_fan_setup(k, 30);

    // Power self-check: interchangeable leaves share ONE pool.
    let g = &setup.inputs.node_graph;
    let leaf_pools: Vec<usize> = (0..g.nodes.len())
        .map(NodePos)
        .filter(|&pos| g.successors[pos].is_empty())
        .map(|pos| g.nodes[pos].pool_id)
        .collect();
    assert!(
        leaf_pools.windows(2).all(|w| w[0] == w[1]),
        "terminal-Generated fan leaves must share one pool, got {leaf_pools:?}"
    );

    assert_expander_self_consistent(&setup);
    let optimum = extensive_form_optimum(&setup);
    let (lb, ub) = train_bounds(&mut setup);
    assert!(
        close(lb, optimum),
        "terminal-fan final_lb {lb} must equal extensive-form optimum {optimum} (gap {})",
        lb - optimum
    );
    assert!(
        close(lb, ub),
        "terminal-fan final_lb {lb} must equal final_ub {ub} (gap {})",
        lb - ub
    );
}

// ── Generated-fan value cases (correct on HEAD; permanent regression guards) ──
//
// A Generated fan's interior siblings face identical per-(hydro, stage) inflow, so
// they converge to identical pools and the single-successor backward solve prices
// them correctly. These assert that value equality; they are NOT invariance gates.
// External-distinct fans (materially different successors) are the value-red case
// and land once the sampling-offset defect blocking them is fixed (see this
// module's header).

#[test]
fn interior_sibling_generated_fan_value_matches_oracle() {
    let k = 3;
    let mut fixture = k_fan_setup(k, 6, 25);

    // Power self-check: the fan children own DISTINCT pools.
    let g = &fixture.setup.inputs.node_graph;
    let fan_pools: Vec<usize> = (0..g.nodes.len())
        .map(NodePos)
        .filter(|&pos| !g.successors[pos].is_empty() && g.nodes[pos].stage == StageIdx(1))
        .map(|pos| g.nodes[pos].pool_id)
        .collect();
    assert!(
        fan_pools.len() >= 2,
        "need >= 2 fan nodes, got {fan_pools:?}"
    );
    let mut sorted = fan_pools.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(
        sorted.len(),
        fan_pools.len(),
        "interior-sibling fan children must own distinct pools, got {fan_pools:?}"
    );

    assert_expander_self_consistent(&fixture.setup);
    let optimum = extensive_form_optimum(&fixture.setup);
    let (lb, _ub) = train_bounds(&mut fixture.setup);
    assert!(
        close(lb, optimum),
        "interior-sibling fan final_lb {lb} != extensive-form optimum {optimum} (gap {})",
        lb - optimum
    );
}

#[test]
fn dcs_arm_generated_fan_value_matches_oracle() {
    let k = 3;
    let mut fixture = dcs_k_fan_setup(k, 6, 25);

    // Power self-check: the target fan node has pool_id != stage.
    let g = &fixture.setup.inputs.node_graph;
    let mismatched = (0..g.nodes.len())
        .map(NodePos)
        .filter(|&pos| !g.successors[pos].is_empty())
        .any(|pos| g.nodes[pos].pool_id != g.nodes[pos].stage.0);
    assert!(
        mismatched,
        "DCS arm fixture must carry a cut-generating node whose pool_id != stage"
    );

    assert_expander_self_consistent(&fixture.setup);
    let optimum = extensive_form_optimum(&fixture.setup);
    let (lb, bwd_solves, bwd_basis_offered) = train_backward_basis_signal(&mut fixture.setup);

    // Power self-check: DCS genuinely engaged its lazy backward path, not
    // merely configured it. The Lazy (DCS) backward loads a cut-free core and
    // captures/offers NO frozen warm basis; the Frozen (non-DCS) backward
    // warm-starts from a captured basis on every cross-iteration solve (hundreds
    // of offers on this fixture). Zero backward basis offers WITH the backward
    // genuinely solving is the fingerprint that the DCS lazy path ran on the
    // `pool_id != stage` node — a nonzero count would mean the run silently took
    // the Frozen path and the DCS-arm successor-pool resolution was never
    // exercised.
    assert!(
        bwd_solves > 0,
        "power: the backward pass must actually solve LPs, got {bwd_solves}"
    );
    assert_eq!(
        bwd_basis_offered, 0,
        "DCS Lazy backward offers no frozen warm basis; a nonzero count \
         ({bwd_basis_offered}) means the Frozen path ran, so DCS did not engage"
    );

    assert!(
        close(lb, optimum),
        "DCS-arm final_lb {lb} != extensive-form optimum {optimum} (gap {})",
        lb - optimum
    );
}

// ── All-External fan train-smoke gates ───────────────────────────────────────
//
// The all-External distinct and root fans train once the zero-entity in-sample
// skip unblocks the degenerate NCS class (a non-empty in-sample class alongside
// external-column nodes is rejected at setup). These are shape/train gates only —
// they make NO value comparison, since both train to a zero value gap on HEAD
// (inflow non-binding); the value-red that would expose the child-0 collapse is
// deferred. The distinct-column self-check keeps the gate non-vacuous.

/// The fan's terminal leaves each pin a DISTINCT external scenario column.
fn assert_distinct_external_leaf_columns(setup: &StudySetup) {
    let g = &setup.inputs.node_graph;
    let mut cols: Vec<usize> = (0..g.nodes.len())
        .map(NodePos)
        .filter(|&pos| g.successors[pos].is_empty())
        .map(|pos| {
            assert_eq!(
                g.nodes[pos].openings.source,
                OpeningSource::External,
                "fan leaf must be an external-column node"
            );
            g.nodes[pos].openings.offset
        })
        .collect();
    let n_leaves = cols.len();
    assert!(n_leaves >= 2, "need >= 2 fan leaves, got {n_leaves}");
    cols.sort_unstable();
    cols.dedup();
    assert_eq!(
        cols.len(),
        n_leaves,
        "fan leaves must pin distinct external columns, not {n_leaves} copies of one"
    );
}

#[test]
fn external_distinct_fan_trains_with_distinct_leaf_columns() {
    let mut setup = external_distinct_fan_setup(3, 30);
    assert_distinct_external_leaf_columns(&setup);
    let (lb, ub) = train_bounds(&mut setup);
    assert!(
        lb.is_finite() && ub.is_finite(),
        "training must produce finite bounds, got lb={lb} ub={ub}"
    );
}

#[test]
fn external_root_fan_trains_with_nonzero_root_column() {
    let mut setup = external_root_fan_setup(3, 30);
    assert_distinct_external_leaf_columns(&setup);

    let g = &setup.inputs.node_graph;
    let roots: Vec<NodePos> = (0..g.nodes.len())
        .map(NodePos)
        .filter(|&pos| g.nodes[pos].stage == StageIdx(0))
        .collect();
    assert_eq!(
        roots.len(),
        1,
        "root fixture must have exactly one stage-0 node"
    );
    let root = roots[0];
    assert_eq!(
        g.nodes[root].openings.source,
        OpeningSource::External,
        "root must be an external-column node"
    );
    assert!(
        g.nodes[root].openings.offset > 0,
        "root must pin a non-zero external column, got offset {}",
        g.nodes[root].openings.offset
    );

    let (lb, ub) = train_bounds(&mut setup);
    assert!(
        lb.is_finite() && ub.is_finite(),
        "training must produce finite bounds, got lb={lb} ub={ub}"
    );
}

// ── Water-binding External-distinct fan value-red (Gap 0) ────────────────────
//
// The genuine value-red: an all-external-inflow distinct fan whose hydro generates
// against a scarce reservoir, so each leaf's own inflow binds. The backward child-0
// collapse prices every leaf against column 0 (the low-inflow, most-expensive leaf),
// overstating future cost so `final_lb` overshoots the true optimum (final_lb >
// final_ub, an invalid lower bound). Reifying the successor outcome set — each child
// priced against its own LP — closes the bound to the extensive-form optimum.

#[test]
fn water_binding_external_fan_final_lb_matches_extensive_form() {
    let mut setup = water_binding_external_fan_setup(3, 30);

    // Power self-check (i): the injected productivity is ρ = 0.95 on every stage, not
    // `default_from_system`'s 0.0 placeholder (which generates 0 MW and hides the gap).
    for stage in 0..2 {
        match setup.hydro_models.production.model(0, stage) {
            ResolvedProductionModel::ConstantProductivity { productivity } => assert!(
                (productivity - 0.95).abs() < 1e-12,
                "stage {stage} productivity must be 0.95, got {productivity}"
            ),
            fpha @ ResolvedProductionModel::Fpha { .. } => {
                panic!("stage {stage} must be ConstantProductivity, got {fpha:?}")
            }
        }
    }

    // Power self-check (ii): the fan leaves pin DISTINCT external inflow columns, so
    // the successors are genuinely non-interchangeable.
    assert_distinct_external_leaf_columns(&setup);

    // Power self-check (iii): the load is deterministic (no stochastic load bus), so
    // the oracle — which applies external INFLOW only, leaving the load segment at
    // η = 0 — reproduces training's demand exactly. A stochastic (std > 0) or external
    // load would make the two disagree on demand and contaminate the value gap.
    assert_eq!(
        setup.stage_ctx().load_bus_indices.len(),
        0,
        "water-binding fan load must be deterministic (std = 0) so oracle == training on demand"
    );

    assert_expander_self_consistent(&setup);
    let optimum = extensive_form_optimum(&setup);
    let (lb, ub) = train_bounds(&mut setup);

    assert!(
        close(lb, optimum),
        "water-binding fan final_lb {lb} must equal extensive-form optimum {optimum} (gap {})",
        lb - optimum
    );
    assert!(
        close(lb, ub),
        "water-binding fan final_lb {lb} must equal final_ub {ub} (gap {})",
        lb - ub
    );
}

#[test]
fn water_binding_external_fan_final_lb_is_thread_shape_invariant() {
    // Power self-check: the fan genuinely branches (>= 2 distinct-column leaves), so
    // the per-child backward solve is actually exercised.
    let mut probe = water_binding_external_fan_setup(3, 30);
    assert_distinct_external_leaf_columns(&probe);
    let _ = train_bounds(&mut probe);

    let mut one = water_binding_external_fan_setup(3, 30);
    let mut many = water_binding_external_fan_setup(3, 30);
    let (lb1, _) = train_bounds_threads(&mut one, 1);
    let (lb4, _) = train_bounds_threads(&mut many, 4);
    assert_eq!(
        lb1.to_bits(),
        lb4.to_bits(),
        "water-binding fan final_lb must be bit-identical across --threads 1 ({lb1}) and \
         --threads 4 ({lb4}) — the reified per-child backward is thread-shape invariant"
    );
}

/// Declaration-order invariance on a genuine NON-interchangeable fan: the
/// water-binding fan trained with its nodes/transitions declared in canonical vs
/// reversed order must produce a bit-identical `final_lb`. Generated-fan children are
/// interchangeable, so the existing `k_fan_*` gates cannot catch a canonical-order
/// bug on a fan whose siblings differ in value — this fixture can.
#[test]
fn water_binding_external_fan_final_lb_is_declaration_order_invariant() {
    // Power self-check: both orderings are genuine multi-node fans (>= 2
    // distinct-column leaves), so the per-child backward is actually exercised.
    let mut canonical = water_binding_external_fan_setup(3, 30);
    let mut reversed = water_binding_external_fan_setup_reversed(3, 30);
    assert_distinct_external_leaf_columns(&canonical);
    assert_distinct_external_leaf_columns(&reversed);

    let (lb_canonical, _) = train_bounds(&mut canonical);
    let (lb_reversed, _) = train_bounds(&mut reversed);
    assert_eq!(
        lb_canonical.to_bits(),
        lb_reversed.to_bits(),
        "water-binding fan final_lb must be bit-identical whether the fan's nodes are declared \
         canonically ({lb_canonical}) or reversed ({lb_reversed})"
    );
}

// ── By-node scheduler on the branching graph ─────────────────────────────────
//
// The by-node (opening-block) scheduler now claims over the reified successor
// outcome set, pricing each fan child against its OWN LP instead of collapsing the
// whole set onto child 0 (which went out of bounds on a fan). These cases run the
// fan oracle fixtures under `by_node` and assert (a) it closes `final_lb` to the
// extensive-form optimum — value-correct on a fan, not merely non-crashing — and
// (b) on these `|Ω| = 1`-per-child trees it produces the SAME cut as `by_scenario`,
// legitimately bitwise because every child run is a single-opening solve loading its
// own basis (identical warm-start structure to the trial-point path).

/// The water-binding External-distinct fan — the genuine value-red — under
/// `by_node`: closes `final_lb` to the extensive-form optimum and to the exact UB,
/// and is bit-identical to the `by_scenario` run (`|Ω| = 1` per child).
#[test]
fn water_binding_external_fan_by_node_matches_extensive_form() {
    let mut setup = water_binding_external_fan_setup(3, 30);
    assert_distinct_external_leaf_columns(&setup);
    let optimum = extensive_form_optimum(&setup);

    let mut by_scenario = water_binding_external_fan_setup(3, 30);
    let (lb_by_scenario, _) =
        train_bounds_scheduler(&mut by_scenario, BackwardScheduler::ByScenario {});
    let (lb_by_node, ub_by_node) = train_bounds_scheduler(&mut setup, BY_NODE);

    assert!(
        close(lb_by_node, optimum),
        "by_node water-binding fan final_lb {lb_by_node} must equal extensive-form optimum \
         {optimum} (gap {})",
        lb_by_node - optimum
    );
    assert!(
        close(lb_by_node, ub_by_node),
        "by_node water-binding fan final_lb {lb_by_node} must equal final_ub {ub_by_node} (gap {})",
        lb_by_node - ub_by_node
    );
    assert_eq!(
        lb_by_node.to_bits(),
        lb_by_scenario.to_bits(),
        "by_node and by_scenario must produce a bit-identical final_lb on the |Ω|=1-per-child \
         water-binding fan ({lb_by_node} vs {lb_by_scenario})"
    );
}

/// The interior-sibling Generated fan under `by_node`: closes `final_lb` to the
/// extensive-form optimum and is bit-identical to `by_scenario` (`|Ω| = 1` per
/// child). Its root fans over `k` distinct-pool children, so the by-node claim loop
/// genuinely spans child boundaries within a block.
#[test]
fn interior_sibling_generated_fan_by_node_matches_oracle() {
    let k = 3;
    let mut fixture = k_fan_setup(k, 6, 25);

    // Power self-check: the fan children own DISTINCT pools, so the by-node root
    // backward genuinely fans over more than one child LP.
    let g = &fixture.setup.inputs.node_graph;
    let fan_pools: Vec<usize> = (0..g.nodes.len())
        .map(NodePos)
        .filter(|&pos| !g.successors[pos].is_empty() && g.nodes[pos].stage == StageIdx(1))
        .map(|pos| g.nodes[pos].pool_id)
        .collect();
    assert!(
        fan_pools.len() >= 2,
        "need >= 2 fan nodes, got {fan_pools:?}"
    );

    let optimum = extensive_form_optimum(&fixture.setup);

    let mut by_scenario = k_fan_setup(k, 6, 25);
    let (lb_by_scenario, _) =
        train_bounds_scheduler(&mut by_scenario.setup, BackwardScheduler::ByScenario {});
    let (lb_by_node, _) = train_bounds_scheduler(&mut fixture.setup, BY_NODE);

    assert!(
        close(lb_by_node, optimum),
        "by_node interior-sibling fan final_lb {lb_by_node} != extensive-form optimum {optimum} \
         (gap {})",
        lb_by_node - optimum
    );
    assert_eq!(
        lb_by_node.to_bits(),
        lb_by_scenario.to_bits(),
        "by_node and by_scenario must produce a bit-identical final_lb on the |Ω|=1-per-child \
         interior-sibling fan ({lb_by_node} vs {lb_by_scenario})"
    );
}

/// The all-External distinct fan under `by_node`: trains (the OOB the child-0
/// collapse hit on a fan is closed) and is bit-identical to `by_scenario`.
#[test]
fn external_distinct_fan_by_node_matches_by_scenario() {
    let mut setup = external_distinct_fan_setup(3, 30);
    assert_distinct_external_leaf_columns(&setup);

    let mut by_scenario = external_distinct_fan_setup(3, 30);
    let (lb_by_scenario, ub_by_scenario) =
        train_bounds_scheduler(&mut by_scenario, BackwardScheduler::ByScenario {});
    let (lb_by_node, ub_by_node) = train_bounds_scheduler(&mut setup, BY_NODE);

    assert!(
        lb_by_node.is_finite() && ub_by_node.is_finite(),
        "by_node training must produce finite bounds, got lb={lb_by_node} ub={ub_by_node}"
    );
    assert_eq!(
        lb_by_node.to_bits(),
        lb_by_scenario.to_bits(),
        "by_node and by_scenario must produce a bit-identical final_lb on the external-distinct \
         fan ({lb_by_node} vs {lb_by_scenario})"
    );
    assert_eq!(
        ub_by_node.to_bits(),
        ub_by_scenario.to_bits(),
        "by_node and by_scenario must produce a bit-identical final_ub on the external-distinct \
         fan ({ub_by_node} vs {ub_by_scenario})"
    );
}

// ── Terminal-fusion cut-state projection (forward capture) ──────────────────
//
// The fused forward capture must project a terminal leaf's dual with its
// CUT-GENERATING PARENT's cut-state layout (the backward's own
// `SuccessorSpec::cut_state`), never the leaf's own terminal pool
// (`training::forward::enumerated::solve_forward_node`). Every other
// enumerated fixture in this file declares no inflow-lag slot, so the leaf's
// always-full terminal pool coincides with its parent's successor-sized pool
// and a wrong-pool projection is dimensionally invisible; this fixture's
// active lag slot makes the two diverge, reproducing the invalid-lower-bound /
// `allgather_outcomes` failure a wrong-projection capture produces. There is
// no config path to force a non-fused cold solve of one specific leaf, so
// this does not compare against a cold-solved control cut directly — the
// extensive-form oracle below is the value-correctness check instead.

#[test]
fn external_distinct_fan_heterogeneous_cut_state_matches_extensive_form() {
    let mut setup = external_distinct_fan_setup_heterogeneous_cut_state(2, 30);
    assert_distinct_external_leaf_columns(&setup);

    // Power self-check: this fixture's whole point is that a leaf's own
    // terminal pool and its cut-generating parent's pool project DIFFERENT
    // cut-state dimensions — the coverage gap the other fixtures above lack.
    let g = &setup.inputs.node_graph;
    let root_pool = (0..g.nodes.len())
        .map(NodePos)
        .find(|&pos| g.nodes[pos].stage == StageIdx(0))
        .map(|pos| g.nodes[pos].pool_id)
        .expect("fixture must have a stage-0 root");
    let leaf_pools: Vec<usize> = (0..g.nodes.len())
        .map(NodePos)
        .filter(|&pos| g.successors[pos].is_empty())
        .map(|pos| g.nodes[pos].pool_id)
        .collect();
    let dims = pool_cut_state_dimensions(&setup);
    for &leaf_pool in &leaf_pools {
        assert_ne!(
            dims[leaf_pool], dims[root_pool],
            "fixture power check: leaf pool {leaf_pool} and root (cut-generating parent) pool \
             {root_pool} must project different cut-state dimensions ({dims:?}), or this test \
             cannot distinguish the fix from the wrong-projection bug"
        );
    }

    let optimum = extensive_form_optimum(&setup);
    let (lb, ub) = train_bounds(&mut setup);

    assert!(
        close(lb, optimum),
        "heterogeneous-cut-state fan final_lb {lb} must equal extensive-form optimum {optimum} \
         (gap {})",
        lb - optimum
    );
    assert!(
        close(lb, ub),
        "heterogeneous-cut-state fan final_lb {lb} must equal final_ub {ub} — no negative gap \
         (gap {})",
        lb - ub
    );
}

// ── Terminal-pool memory-shape regression guard (ignored) ────────────────────
//
// The static terminal pool removed the terminal
// pool's growable training capacity (fixed-capacity `CutPool::new_with_warm_start`
// construction and `grow_pools_for_next_iteration`'s terminal skip) and stopped
// re-baking its frozen LP template every iteration (the priming-only freeze). Both
// changes are structural and deterministic, so they are pinned below as an
// `#[ignore]`-gated capstone guard over a full training run on a wide terminal
// fan, rather than left to the empirical RSS spot-check documented here.
//
// Structural invariant: on a study whose leaves share one terminal `CutPool` —
// the trailing shared leaf pool the node graph assigns to every
// all-successor-less stage — training must leave the terminal pool's
// `capacity`, `warm_start_count`, `active_count()`, and `populated()` unchanged
// from their pre-training values: a leaf never receives a new cut
// (`enumerated_pool_cut_stride` sizes a leaf pool's per-iteration stride at
// zero), so there is no growable training slack to consume and no new cut to
// bake into a refrozen template. The two tests below capture that snapshot
// before training, train to completion via `train_bounds`, and assert the
// snapshot is bit-for-bit unchanged afterward — the non-growable and
// single-materialization invariants together.
//
// Per-worker RSS spot-measurement (manual; commits no number): the design
// doc's "Memory analysis" section asks for a per-worker RSS measurement on a
// DECOMP-like terminal fan as an implementation-time regression check,
// explicitly not a committed figure. The reproducible procedure:
//
// 1. Build a DECOMP-like terminal fan at production-representative scale —
//    `terminal_generated_fan_setup`/`external_distinct_fan_setup` (or an
//    equivalent DECOMP-format deck with a boundary policy loaded via
//    `inject_boundary_cuts`) sized to the target `k`, `max_iterations`, and
//    worker thread count (the `n_threads` argument to `StudySetup::train`).
// 2. On Linux, read this process's own `VmRSS:` line from `/proc/self/status`
//    (a text pseudo-file; the line reports resident memory in kB) immediately
//    before calling `train`, and again immediately after `train` returns.
// 3. Run the same procedure twice with every parameter held fixed: once
//    against a checkout at (or before) the commit predating Change 3, and
//    once against the tree under test.
// 4. Compare the two post-minus-pre `VmRSS` deltas. The tree under test
//    should show a smaller delta at higher thread counts, reflecting one
//    fewer full copy of the terminal pool's coefficients held per worker (the
//    growable pool plus its per-iteration refrozen template, versus the fixed
//    pool plus its bake-once template).
//
// This is a manual spot check, not an automated assertion: `VmRSS` is
// influenced by allocator behavior, resident page-cache state, and OS
// scheduling in ways that make it unsuitable as a deterministic CI gate, and
// no number from step 4 is recorded here or anywhere else in this file. The
// structural guard below is the automated regression protection; this
// procedure is its empirical companion, run by hand during implementation or
// optimization work.

/// Training-iteration budget for both terminal-pool fixtures below: enough
/// iterations that a per-iteration growth or re-materialization regression has
/// room to manifest, small enough to keep an `#[ignore]`-gated guard cheap to
/// run on demand.
const MAX_ITERATIONS: u32 = 30;

/// Fan-out width for [`terminal_generated_fan_setup`] below — `k >= 8` per the
/// wide-fan requirement this guard exists to cover.
const WIDE_FAN_K: usize = 8;

/// Fan-out width for [`external_distinct_fan_setup`] below, capped at its
/// fixture's own distinct-inflow-column limit.
const DISTINCT_FAN_K: usize = 3;

/// Snapshot of a terminal [`CutPool`]'s memory-shape fields. Comparing two
/// snapshots — one taken before training, one after — is the pre/post check
/// the non-growable and single-materialization invariants reduce to on a leaf
/// pool that never receives a new cut.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TerminalPoolShape {
    capacity: usize,
    warm_start_count: u32,
    active_count: usize,
    populated: usize,
}

impl TerminalPoolShape {
    fn capture(pool: &CutPool) -> Self {
        Self {
            capacity: pool.capacity,
            warm_start_count: pool.warm_start_count,
            active_count: pool.active_count(),
            populated: pool.populated(),
        }
    }
}

/// Asserts the fixed-capacity, no-boundary-cut shape both fixtures below start
/// from: no warm-started cuts, so `capacity == warm_start_count == 0`
/// (`CutPool::new_with_warm_start`'s `max_iterations = 0` construction leaves
/// no growable training slack), and no populated or active slot.
fn assert_fixed_no_boundary_shape(shape: TerminalPoolShape) {
    assert_eq!(
        shape.warm_start_count, 0,
        "fixture carries no boundary-injected cuts"
    );
    assert_eq!(
        shape.capacity, shape.warm_start_count as usize,
        "a leaf pool's capacity must equal its warm_start_count (0 here) -- no \
         growable training slack"
    );
    assert_eq!(
        shape.active_count, 0,
        "a leaf pool never receives a training cut"
    );
    assert_eq!(
        shape.populated, 0,
        "a leaf pool's populated high-water mark never advances"
    );
}

/// A wide (`k = 8`) terminal-Generated fan: the primary AC-mandated fixture.
/// Training must leave the shared terminal pool's capacity, warm-start count,
/// active-cut count, and populated high-water mark bit-for-bit unchanged.
#[test]
#[ignore = "trains a wide terminal fan; run with `-- --ignored`"]
fn terminal_generated_fan_pool_stays_fixed_and_single_materialized() {
    let mut setup = terminal_generated_fan_setup(WIDE_FAN_K, MAX_ITERATIONS);
    let terminal_idx = setup
        .inputs
        .node_graph
        .terminal_pool(setup.num_stages())
        .unwrap();

    let before = TerminalPoolShape::capture(&setup.fcf.pools[terminal_idx]);
    assert_fixed_no_boundary_shape(before);

    let _ = train_bounds(&mut setup);

    let after = TerminalPoolShape::capture(&setup.fcf.pools[terminal_idx]);
    assert_fixed_no_boundary_shape(after);
    assert_eq!(
        after, before,
        "the terminal pool's capacity/warm_start_count/active_count/populated must \
         be unchanged across the full training run -- no growth, no \
         re-materialization"
    );
}

/// A second, materially different terminal fan (distinct, non-interchangeable
/// per-leaf inflow columns rather than the Generated fan's interchangeable
/// leaves) exercising the same shared-terminal-pool invariant under a different
/// topology.
#[test]
#[ignore = "trains a wide terminal fan; run with `-- --ignored`"]
fn external_distinct_fan_terminal_pool_stays_fixed_and_single_materialized() {
    let mut setup = external_distinct_fan_setup(DISTINCT_FAN_K, MAX_ITERATIONS);
    let terminal_idx = setup
        .inputs
        .node_graph
        .terminal_pool(setup.num_stages())
        .unwrap();

    let before = TerminalPoolShape::capture(&setup.fcf.pools[terminal_idx]);
    assert_fixed_no_boundary_shape(before);

    let _ = train_bounds(&mut setup);

    let after = TerminalPoolShape::capture(&setup.fcf.pools[terminal_idx]);
    assert_fixed_no_boundary_shape(after);
    assert_eq!(
        after, before,
        "the terminal pool's capacity/warm_start_count/active_count/populated must \
         be unchanged across the full training run on a distinct-successor \
         terminal fan too"
    );
}

// ── Terminal boundary FCF bound-accounting (sddp.md "Terminal boundary FCF in the
//    reported total cost") ──────────────────────────────────────────────────────

/// Intercept (scaled cost units) of the constant boundary cut injected at the
/// terminal pool: a nonzero post-horizon value-to-go with no state gradient, so it
/// shifts every path's cost by a fixed discounted amount without perturbing the
/// in-horizon optimal policy.
const TERMINAL_BOUNDARY_INTERCEPT: f64 = 100.0;

/// Replace `setup`'s terminal FCF pool with a single constant boundary cut
/// (`θ_terminal ≥ intercept`, zero state coefficients), so the terminal θ prices a
/// nonzero post-horizon value-to-go — the terminal-boundary-policy shape no
/// existing fixture exercised. Mirrors `inject_boundary_cuts`, building the fixed
/// warm-start pool directly from a hand-constructed record (the validated
/// file-load path is unnecessary for a synthetic fixture).
fn inject_constant_terminal_boundary_fcf(setup: &mut StudySetup, intercept: f64, is_active: bool) {
    let terminal = setup
        .inputs
        .node_graph
        .terminal_pool(setup.num_stages())
        .unwrap();
    let state_dim = setup.fcf.state_dimension;
    let forward_passes = setup.fcf.forward_passes;
    let record = cobre_io::OwnedPolicyCutRecord {
        cut_id: 0,
        slot_index: 0,
        coefficients: vec![0.0; state_dim],
        intercept,
        iteration: 0,
        forward_pass_index: 0,
        is_active,
    };
    setup.fcf.pools[terminal] =
        CutPool::new_with_warm_start(state_dim, forward_passes, 0, &[record]);
}

/// A terminal boundary FCF (nonzero post-horizon value-to-go) must be booked
/// identically in the lower and upper bound, so a converged deterministic chain
/// still closes `final_lb == final_ub`. Before the terminal-θ-in-cost fix the UB
/// subtracted the terminal θ that the LB keeps, so `final_lb ≫ final_ub` — a
/// negative gap the stopping rule's `.max(0.0)` clamp falsely reads as converged.
/// The `!close(lb_b, lb_p)` power check proves the injected FCF actually reaches
/// the bound (a vacuous θ=0 fixture would pass `close(lb_b, ub_b)` trivially).
#[test]
fn terminal_boundary_fcf_training_gap_is_consistent() {
    let mut plain = oracle_chain_setup(30);
    let (lb_p, ub_p) = train_bounds(&mut plain);
    assert!(
        close(lb_p, ub_p),
        "control: plain chain must converge to final_lb == final_ub (lb {lb_p}, ub {ub_p})"
    );

    let mut boundary = oracle_chain_setup(30);
    inject_constant_terminal_boundary_fcf(&mut boundary, TERMINAL_BOUNDARY_INTERCEPT, true);
    let (lb_b, ub_b) = train_bounds(&mut boundary);

    assert!(
        !close(lb_b, lb_p),
        "power: the terminal boundary FCF must materially raise the LB above the plain \
         chain (lb_b {lb_b}, lb_p {lb_p}); a vacuous θ=0 fixture could not test the fix"
    );
    assert!(
        close(lb_b, ub_b),
        "terminal boundary FCF must be booked in BOTH bounds: final_lb {lb_b} must equal \
         final_ub {ub_b} (gap {}). A negative gap here is the terminal-θ subtraction bug",
        lb_b - ub_b
    );
}

/// Mean per-scenario `total_cost` over a simulation's returned scenarios.
fn mean_scenario_total_cost(sims: &[cobre_sddp::SimulationScenarioResult]) -> f64 {
    assert!(
        !sims.is_empty(),
        "simulation must produce at least one scenario result"
    );
    sims.iter().map(|s| s.total_cost).sum::<f64>() / sims.len() as f64
}

/// The simulation's reported per-scenario cost books the terminal boundary FCF
/// (post-horizon value-to-go) the same way the forward pass does. A constant
/// terminal FCF shifts every path's reported total by the same discounted amount,
/// so the boundary run's mean cost exceeds the plain run's by that amount. Before
/// the fix the census simulation subtracted the terminal θ, so the FCF never
/// reached the reported total and the two means coincided. Exercises the census
/// (`enumerated`) simulation engine, which shares `extract_sim_stage_result` — the
/// single fixed site — with the sampled engine.
#[test]
fn terminal_boundary_fcf_simulation_cost_includes_post_horizon() {
    let mut plain = try_k_fan_simulation_enumerated(2).expect("simulate-enabled fixture builds");
    let mean_plain = mean_scenario_total_cost(&run_simulation(&mut plain, 1));

    let mut boundary = try_k_fan_simulation_enumerated(2).expect("simulate-enabled fixture builds");
    inject_constant_terminal_boundary_fcf(&mut boundary, TERMINAL_BOUNDARY_INTERCEPT, true);
    let mean_boundary = mean_scenario_total_cost(&run_simulation(&mut boundary, 1));

    assert!(
        !close(mean_boundary, mean_plain),
        "simulation total_cost must carry the post-horizon FCF: boundary mean {mean_boundary} \
         must differ from plain mean {mean_plain}. Equal means are the terminal-θ subtraction bug"
    );
    assert!(
        mean_boundary > mean_plain,
        "the post-horizon FCF (a positive value-to-go) must strictly raise the reported \
         simulation cost: boundary mean {mean_boundary} must exceed plain mean {mean_plain}"
    );
}

/// Asserts an injected `is_active: false` terminal boundary record still
/// counts as warm-started but contributes no active cut, and that training
/// closes to the plain fixture's own bounds regardless — the shared body of
/// the chain and fan variants below. `topology` names the fixture in the
/// bound-mismatch panic message.
fn assert_inactive_boundary_record_leaves_bounds_unchanged(
    inactive: &StudySetup,
    lb_inactive: f64,
    ub_inactive: f64,
    lb_plain: f64,
    ub_plain: f64,
    topology: &str,
) {
    let terminal = inactive
        .inputs
        .node_graph
        .terminal_pool(inactive.num_stages())
        .unwrap();
    assert_eq!(
        inactive.fcf.pools[terminal].active_count(),
        0,
        "the injected record is inactive"
    );
    assert_eq!(
        inactive.fcf.pools[terminal].warm_start_count, 1,
        "the injected record still counts as a warm-started slot"
    );
    assert!(
        inactive.fcf.pools[terminal].has_warm_start_cuts(),
        "the owner must still read the pool as boundary-loaded despite the inactive record"
    );

    assert!(
        close(lb_inactive, lb_plain),
        "an inactive boundary record must not move the {topology}'s lower bound: \
         {lb_inactive} vs {lb_plain}"
    );
    assert!(
        close(ub_inactive, ub_plain),
        "an inactive boundary record must not move the {topology}'s upper bound: \
         {ub_inactive} vs {ub_plain}"
    );
}

/// The one input where the terminal-boundary flag's two formulas disagree
/// (`CutPool::active_count() > 0` vs `warm_start_count > 0`): a terminal pool
/// whose only boundary record is `is_active: false`. Both leave terminal θ at
/// 0 — the inactive record contributes no active LP row either way, and θ's
/// own lower bound is `0.0` — so training must close to the plain chain's
/// bounds regardless of which formula the engine reads.
#[test]
fn terminal_boundary_records_all_inactive_leave_the_plain_chain_bounds() {
    let mut plain = oracle_chain_setup(30);
    let (lb_plain, ub_plain) = train_bounds(&mut plain);

    let mut inactive = oracle_chain_setup(30);
    inject_constant_terminal_boundary_fcf(&mut inactive, TERMINAL_BOUNDARY_INTERCEPT, false);
    let (lb_inactive, ub_inactive) = train_bounds(&mut inactive);

    assert_inactive_boundary_record_leaves_bounds_unchanged(
        &inactive,
        lb_inactive,
        ub_inactive,
        lb_plain,
        ub_plain,
        "chain",
    );
}

/// The chain disagreement case above, on a terminal fan — the branching
/// topology [`terminal_boundary_fcf_training_gap_is_consistent`]'s chain
/// control does not cover. No existing fan fixture in this file takes an
/// injected boundary record through training, so this test builds its own
/// small fan directly from `test_support::terminal_generated_fan_setup`.
#[test]
fn terminal_boundary_records_all_inactive_leave_the_plain_fan_bounds() {
    let mut plain = terminal_generated_fan_setup(2, MAX_ITERATIONS);
    let (lb_plain, ub_plain) = train_bounds(&mut plain);

    let mut inactive = terminal_generated_fan_setup(2, MAX_ITERATIONS);
    inject_constant_terminal_boundary_fcf(&mut inactive, TERMINAL_BOUNDARY_INTERCEPT, false);
    let (lb_inactive, ub_inactive) = train_bounds(&mut inactive);

    assert_inactive_boundary_record_leaves_bounds_unchanged(
        &inactive,
        lb_inactive,
        ub_inactive,
        lb_plain,
        ub_plain,
        "fan",
    );
}

// ── Terminal-leaf fusion: analytical + behavioral verification capstone ─────
//
// Fusion reuses an eligible External terminal leaf's forward-captured
// `(objective, duals)` for its cut-generating parent's Benders cut instead of
// re-solving that leaf in the backward pass (`NodeGraph::is_external_terminal_leaf`,
// `training/backward/replicated.rs`'s `run_backward_node_replicated`). This module
// is the capstone: an analytical oracle proving the fused cut is a valid
// supporting hyperplane, the bound/thread-shape pins fusion must not perturb, a
// compute-win pin that the redundant External-leaf solve is gone, and the
// Generated-leaf mirror proving fusion did not leak past its `is_external_terminal_leaf`
// gate.
mod terminal_fusion {
    use std::collections::HashMap;
    use std::sync::mpsc;

    use cobre_core::TrainingEvent;
    use cobre_solver::SolverInterface;

    use super::{
        ActiveSolver, NodePos, StageIdx, StubComm, StudySetup,
        assert_distinct_external_leaf_columns, close, extensive_form_optimum,
        external_distinct_fan_setup, terminal_generated_fan_setup, train_bounds_threads,
    };
    use cobre_sddp::indexer::StateDim;
    use cobre_sddp::test_support::{
        capture_patched_node_template, water_binding_external_fan_setup,
    };

    /// Training budget for every External-fan behavioral/compute-win fixture
    /// in this module.
    const FUSION_MAX_ITERATIONS: u32 = 30;

    /// Fan-out width for every fixture in this module.
    const FUSION_FAN_K: usize = 3;

    /// Training budget for the analytical oracle: exactly one iteration,
    /// so the trial state driving the produced cut is deterministic. With no
    /// cuts yet on the root's pool, θ is unconstrained, so the root's own
    /// solve is a well-defined LP an independent probe solve can reproduce.
    const ANALYTICAL_ORACLE_ITERATIONS: u32 = 1;

    /// Trains `setup` single-rank/single-thread with an attached event
    /// channel, returning `(final_lb, final_ub, backward_lp_solves)`.
    /// `solver_stats_log`'s own `"backward"` phase stays empty for enumerated
    /// training — `BackwardResult::stage_stats` is unconditionally
    /// `Vec::new()` off the sampled/by-node path
    /// (`BackwardPassState::run_enumerated_backward`) — so the real backward
    /// solve count is recovered by subtracting this iteration's logged
    /// `"forward"` + `"lower_bound"` total from
    /// `TrainingEvent::IterationSummary::lp_solves`, which sums all three
    /// phases for that iteration. Filtering `solver_stats_log` by
    /// `phase == "backward"` here would silently read zero regardless of
    /// fusion — the wrong-but-compiling shortcut this helper exists to avoid.
    fn train_enumerated_backward_lp_solves(setup: &mut StudySetup) -> (f64, f64, u64) {
        let comm = StubComm;
        let mut solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");
        let (tx, rx) = mpsc::channel::<TrainingEvent>();
        let outcome = setup
            .train(&mut solver, &comm, 1, ActiveSolver::new, Some(tx), None)
            .expect("training must return Ok");
        assert!(
            outcome.error.is_none(),
            "training must not error: {:?}",
            outcome.error
        );

        let mut fwd_lb_by_iter: HashMap<u64, u64> = HashMap::new();
        for entry in &outcome.result.solver_stats_log {
            if entry.phase == "forward" || entry.phase == "lower_bound" {
                *fwd_lb_by_iter.entry(entry.iteration).or_default() += entry.delta.lp_solves;
            }
        }

        let mut backward_lp_solves = 0_u64;
        let mut iterations_seen = 0_u64;
        for event in rx.try_iter() {
            if let TrainingEvent::IterationSummary {
                iteration,
                lp_solves,
                ..
            } = event
            {
                let fwd_lb = fwd_lb_by_iter.get(&iteration).copied().unwrap_or(0);
                backward_lp_solves += lp_solves.saturating_sub(fwd_lb);
                iterations_seen += 1;
            }
        }
        assert!(
            iterations_seen > 0,
            "power: training must emit at least one IterationSummary event"
        );
        (
            outcome.result.final_lb,
            outcome.result.final_ub,
            backward_lp_solves,
        )
    }

    /// Analytical oracle: [`water_binding_external_fan_setup`]'s hydro
    /// genuinely generates against a scarce reservoir (ρ = 0.95), so each
    /// leaf's storage dual is a real, leaf-varying, nonzero number — unlike a
    /// zero-productivity fixture, where every leaf's dual collapses to zero
    /// regardless of a sign/scale/weighting bug. Trained for exactly
    /// [`ANALYTICAL_ORACLE_ITERATIONS`] iteration, the produced cut is
    /// independently re-derived via a code path that never touches the
    /// fused-capture machinery in `training/forward/enumerated.rs`:
    ///
    /// 1. solve the root's own fresh (zero-cut) template directly
    ///    (`capture_patched_node_template` + a bare solver call) to read
    ///    `x_bar` — the root's own outgoing-storage decision;
    /// 2. solve each leaf's own template with its incoming-storage column
    ///    FIXED to `x_bar` (`col_lower == col_upper == x_bar`, mirroring
    ///    production's own column-bound state pinning — sddp.md "State
    ///    pinning uses column bounds"), reading `(objective,
    ///    reduced_costs[in_col] / col_scale[in_col])` per leaf — the same
    ///    `rc_scaled / col_scale` formula `extract_state_duals_only` uses,
    ///    computed here from an independent solve;
    /// 3. combine the leaves' `(objective, dual)` by the fixture's own
    ///    probability weights (`i / total`, `i = 1..=k`) into
    ///    `hand_intercept = Σ wᵢ·(objᵢ − dualᵢ·x_bar)`,
    ///    `hand_coefficient = Σ wᵢ·dualᵢ`.
    ///
    /// A dropped, sign-flipped, mis-scaled, or misweighted dual in the
    /// production fused-capture path changes the TRAINED cut but not this
    /// independent re-derivation, so the two diverge — this oracle has power.
    #[test]
    fn water_binding_external_fan_fused_cut_matches_independently_derived_cut() {
        let k = FUSION_FAN_K;
        let mut setup = water_binding_external_fan_setup(k, ANALYTICAL_ORACLE_ITERATIONS);
        assert_distinct_external_leaf_columns(&setup);

        let state = setup.stage_state();
        let out_col = state.lp_column_for_state(StateDim::new(0)).get();
        let in_col = state.state_to_lp_incoming_column(StateDim::new(0)).get();

        let g = &setup.inputs.node_graph;
        let root_pos = (0..g.nodes.len())
            .map(NodePos)
            .find(|&pos| g.nodes[pos].stage == StageIdx(0))
            .expect("fixture must have a stage-0 root");
        let root_pool = g.nodes[root_pos].pool_id;
        let leaf_positions: Vec<NodePos> = (0..g.nodes.len())
            .map(NodePos)
            .filter(|&pos| g.successors[pos].is_empty())
            .collect();
        assert_eq!(
            leaf_positions.len(),
            k,
            "fixture must fan the root into exactly k leaves"
        );

        let root_template = capture_patched_node_template(&setup, root_pos);
        let mut root_solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");
        root_solver.load_model(&root_template);
        let root_view = root_solver
            .solve(None)
            .expect("root's fresh (zero-cut) LP must solve");
        let x_bar = root_view.primal[out_col];

        let mut leaf_objectives = Vec::with_capacity(k);
        let mut leaf_duals = Vec::with_capacity(k);
        for &leaf_pos in &leaf_positions {
            let mut leaf_template = capture_patched_node_template(&setup, leaf_pos);
            leaf_template.col_lower[in_col] = x_bar;
            leaf_template.col_upper[in_col] = x_bar;
            let mut solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");
            solver.load_model(&leaf_template);
            let view = solver
                .solve(None)
                .expect("leaf's own LP, pinned at x_bar, must solve");
            leaf_objectives.push(view.objective);
            leaf_duals.push(view.reduced_costs[in_col] / leaf_template.col_scale[in_col]);
        }

        let max_obj = leaf_objectives.iter().copied().fold(f64::MIN, f64::max);
        let min_obj = leaf_objectives.iter().copied().fold(f64::MAX, f64::min);
        assert!(
            max_obj - min_obj > 1.0,
            "power: leaf objectives must differ substantially so per-child weighting is \
             exercised, got {leaf_objectives:?}"
        );
        assert!(
            leaf_duals.iter().any(|d| d.abs() > 1e-3),
            "power: at least one leaf's storage dual must be substantially nonzero so sign/\
             scale extraction is exercised, got {leaf_duals:?}"
        );

        let total: f64 = (1..=k).map(|i| i as f64).sum();
        let weights: Vec<f64> = (1..=k).map(|i| i as f64 / total).collect();
        let hand_coefficient: f64 = weights.iter().zip(&leaf_duals).map(|(w, d)| w * d).sum();
        let hand_intercept: f64 = weights
            .iter()
            .zip(&leaf_objectives)
            .zip(&leaf_duals)
            .map(|((w, obj), dual)| w * (obj - dual * x_bar))
            .sum();

        let _ = train_bounds_threads(&mut setup, 1);

        let pool = &setup.fcf.pools[root_pool];
        let mut checked = 0_usize;
        for (slot, intercept, coeffs) in pool.active_cuts() {
            assert!(
                close(intercept, hand_intercept),
                "trained fused cut intercept at slot {slot} is {intercept}, independently \
                 derived is {hand_intercept} (gap {})",
                intercept - hand_intercept
            );
            assert_eq!(
                coeffs.len(),
                1,
                "fixture state dimension must be storage-only (1 dimension)"
            );
            assert!(
                close(coeffs[0], hand_coefficient),
                "trained fused cut coefficient at slot {slot} is {}, independently derived \
                 is {hand_coefficient} (gap {})",
                coeffs[0],
                coeffs[0] - hand_coefficient
            );
            checked += 1;
        }
        assert_eq!(
            checked, 1,
            "training for exactly one iteration must generate exactly one cut"
        );
    }

    /// Behavioral bound: fusion must not perturb the External fan's
    /// bound closure or its thread-shape invariance.
    #[test]
    fn external_distinct_fan_final_bounds_match_extensive_form_and_thread_shape_invariant() {
        let mut setup = external_distinct_fan_setup(FUSION_FAN_K, FUSION_MAX_ITERATIONS);
        assert_distinct_external_leaf_columns(&setup);
        let optimum = extensive_form_optimum(&setup);

        let (lb1, ub1) = train_bounds_threads(&mut setup, 1);
        assert!(
            close(lb1, optimum),
            "fused External fan final_lb {lb1} must equal extensive-form optimum {optimum} \
             (gap {})",
            lb1 - optimum
        );
        assert!(
            close(lb1, ub1),
            "fused External fan final_lb {lb1} must equal final_ub {ub1} (gap {})",
            lb1 - ub1
        );

        let mut setup4 = external_distinct_fan_setup(FUSION_FAN_K, FUSION_MAX_ITERATIONS);
        let (lb4, _) = train_bounds_threads(&mut setup4, 4);
        assert_eq!(
            lb1.to_bits(),
            lb4.to_bits(),
            "fused External fan final_lb must be bit-identical across threads=1 ({lb1}) and \
             threads=4 ({lb4})"
        );
    }

    /// Compute-win pin: fusion eliminates every terminal-leaf backward
    /// solve on the all-External fan.
    #[test]
    fn external_distinct_fan_backward_performs_zero_terminal_leaf_solves() {
        let mut setup = external_distinct_fan_setup(FUSION_FAN_K, FUSION_MAX_ITERATIONS);
        assert_distinct_external_leaf_columns(&setup);

        let (_, _, backward_lp_solves) = train_enumerated_backward_lp_solves(&mut setup);
        assert_eq!(
            backward_lp_solves, 0,
            "fusion must eliminate every terminal-leaf backward solve on the all-External \
             fan, got {backward_lp_solves} backward LP solves"
        );
    }

    /// Generated-leaf guard (paired with the compute-win mirror): a Generated
    /// terminal fan still integrates every opening exhaustively — its bounds
    /// still match the extensive-form optimum — and the backward still
    /// genuinely solves every Generated leaf every iteration (one solve per
    /// leaf per iteration, none skipped), proving fusion did not leak past its
    /// External-only gate.
    #[test]
    fn terminal_generated_fan_integrates_exhaustively_and_backward_solves_every_leaf() {
        let mut setup = terminal_generated_fan_setup(FUSION_FAN_K, FUSION_MAX_ITERATIONS);
        let optimum = extensive_form_optimum(&setup);

        let (lb, ub, backward_lp_solves) = train_enumerated_backward_lp_solves(&mut setup);

        assert!(
            close(lb, optimum),
            "Generated terminal fan final_lb {lb} must equal extensive-form optimum {optimum} \
             (gap {})",
            lb - optimum
        );
        assert!(
            close(lb, ub),
            "Generated terminal fan final_lb {lb} must equal final_ub {ub} (gap {})",
            lb - ub
        );

        let expected_backward_lp_solves =
            u64::from(u32::try_from(FUSION_FAN_K).unwrap()) * u64::from(FUSION_MAX_ITERATIONS);
        assert_eq!(
            backward_lp_solves, expected_backward_lp_solves,
            "the compute-win mirror: a Generated terminal leaf is never fused, so the backward \
             must solve every one of the {FUSION_FAN_K} leaves on every one of the \
             {FUSION_MAX_ITERATIONS} iterations, got {backward_lp_solves} backward LP solves"
        );
    }
}
