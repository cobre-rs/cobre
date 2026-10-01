//! Guard test: a single-node boundary-policy source loads into a fanned
//! terminal target without rejecting on node count or graph identity, and
//! `inject_boundary_cuts` populates the ONE pool every terminal fan leaf's
//! `NodeRuntime::pool_id` resolves to — never a per-leaf distinct policy. A
//! multi-node source is still rejected.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::doc_markdown,
    clippy::too_many_lines
)]

use std::path::Path;

use chrono::NaiveDate;
use cobre_io::{
    GraphManifest, PolicyCutRecord, STAGE_CUTS_NODE_ID_SENTINEL, StageCutsPayload,
    encode_slot_date, write_policy_checkpoint,
};
use cobre_sddp::setup::{NodeGraph, NodePos};
use cobre_sddp::test_support::k_fan_setup;
use cobre_sddp::{
    BoundaryLoadRequest, LEGACY_COST_SCALE_FACTOR, inject_boundary_cuts, load_boundary_cuts,
};

/// Pool `pool`'s fixture `priced_state_date`: `2030-01-01` plus `pool`
/// months.
fn fixture_priced_date(pool: u32) -> NaiveDate {
    cobre_sddp::test_support::fixture_priced_date(cobre_sddp::test_support::ymd(2030, 1, 1), pool)
}

/// Write a synthetic single-pool policy checkpoint whose pool's own
/// `graph_stage_id`/`node_id` self-describing facts are derived from
/// `graph_nodes` (`(node_id, stage_id, pool_id)` triples all naming
/// `pool_id`): `graph_stage_id` is their shared `stage_id`; `node_id` is the
/// sole owner's id, or [`STAGE_CUTS_NODE_ID_SENTINEL`] when more than one node
/// owns the pool — mirroring the real writer's `sole_pool_owner_node_id`, so
/// `load_boundary_cuts` resolves and single-node-checks this pool exactly as
/// it would a genuinely exported one. Carries no entity manifest (the
/// identity check is out of this probe's scope; covered by the
/// boundary-injection manifest-mismatch tests elsewhere).
fn write_synthetic_checkpoint(
    dir: &Path,
    graph_nodes: &[(i32, i32, u32)],
    pool_id: u32,
    intercepts: &[f64],
    state_dimension: u32,
) {
    let coefficients = vec![1.0_f64; state_dimension as usize];
    let cuts: Vec<PolicyCutRecord<'_>> = intercepts
        .iter()
        .enumerate()
        .map(|(i, &intercept)| PolicyCutRecord {
            cut_id: i as u64,
            slot_index: i as u32,
            iteration: 0,
            forward_pass_index: 0,
            intercept,
            coefficients: &coefficients,
            is_active: true,
        })
        .collect();
    let active_cut_indices: Vec<u32> = (0..intercepts.len() as u32).collect();

    let mut owner_node_id: Option<i32> = None;
    let mut owner_stage_id: Option<i32> = None;
    let mut shared = false;
    for &(id, stage_id, owning_pool) in graph_nodes {
        if owning_pool == pool_id {
            if owner_node_id.is_some() {
                shared = true;
            }
            owner_node_id = Some(id);
            owner_stage_id = Some(stage_id);
        }
    }
    let node_id = if shared {
        STAGE_CUTS_NODE_ID_SENTINEL
    } else {
        owner_node_id.unwrap_or(STAGE_CUTS_NODE_ID_SENTINEL)
    };
    let graph_stage_id = owner_stage_id.unwrap_or(-1);

    let payload = StageCutsPayload {
        stage_id: pool_id,
        state_dimension,
        capacity: intercepts.len() as u32,
        warm_start_count: 0,
        cuts: &cuts,
        active_cut_indices: &active_cut_indices,
        populated_count: intercepts.len() as u32,
        entity_manifest: &[],
        cost_scale_factor: 1_000_000.0,
        node_id,
        graph_stage_id,
        priced_state_date: encode_slot_date(fixture_priced_date(pool_id)),
    };
    let metadata = cobre_sddp::test_support::checkpoint_metadata(
        1,
        GraphManifest::default(),
        cobre_sddp::test_support::producer_block(),
    );
    write_policy_checkpoint(dir, &[payload], &[], &metadata, &[]).expect("write checkpoint");
}

/// Every leaf `NodePos` in `graph` — a node with no successors, the terminal
/// fan whose pool `inject_boundary_cuts` writes.
fn leaf_positions(graph: &NodeGraph) -> Vec<NodePos> {
    graph
        .nodes
        .iter_indexed()
        .filter(|&(pos, _)| graph.successors[pos].is_empty())
        .map(|(pos, _)| pos)
        .collect()
}

#[test]
fn single_node_source_injects_into_the_one_pool_every_terminal_fan_leaf_shares() {
    let fixture = k_fan_setup(3, 2, 5);
    let mut setup = fixture.setup;

    let leaves = leaf_positions(&setup.inputs.node_graph);
    assert_eq!(
        leaves.len(),
        fixture.k,
        "the fixture must declare a genuine multi-node terminal fan"
    );
    assert!(
        setup.inputs.node_graph.nodes.len() > setup.inputs.node_graph.n_pools,
        "the fan must have more nodes than pools, or leaf-pool-sharing has no work to do"
    );

    let terminal_idx = setup
        .inputs
        .node_graph
        .terminal_pool(setup.num_stages())
        .expect("the fanned target must carry a terminal pool");
    let leaf_pool_ids: Vec<usize> = leaves
        .iter()
        .map(|&pos| setup.inputs.node_graph.nodes[pos].pool_id)
        .collect();
    assert!(
        leaf_pool_ids.iter().all(|&p| p == terminal_idx),
        "every terminal fan leaf must resolve to the same pool inject_boundary_cuts writes: \
         {leaf_pool_ids:?} != {terminal_idx}"
    );

    let state_dimension = setup.fcf.state_dimension as u32;
    let tmp = tempfile::tempdir().expect("tempdir");
    let source_dir = tmp.path().join("single_node_source");
    // One declared node (id 100) at stage 0 -> pool 0: a single-node source,
    // structurally unrelated to the fanned target's own (larger) graph shape.
    write_synthetic_checkpoint(
        &source_dir,
        &[(100, 0, 0)],
        0,
        &[7.0, 11.0],
        state_dimension,
    );

    let boundary_cuts = load_boundary_cuts(&BoundaryLoadRequest::new(
        &source_dir,
        fixture_priced_date(0),
        state_dimension,
        &[],
        LEGACY_COST_SCALE_FACTOR,
    ))
    .expect(
        "a single-node source boundary must load into a fanned terminal target, not reject on \
         node-count or graph identity",
    );
    assert_eq!(boundary_cuts.len(), 2);

    let n_pools_before = setup.fcf.pools.len();
    inject_boundary_cuts(&mut setup, &boundary_cuts).unwrap();
    assert_eq!(
        setup.fcf.pools.len(),
        n_pools_before,
        "injection must not fan out into per-leaf pools"
    );
    assert!(
        setup.fcf.pools[terminal_idx].has_warm_start_cuts(),
        "inject_boundary_cuts must write into the pool NodeGraph::terminal_pool resolves"
    );

    for &pool_id in &leaf_pool_ids {
        let pool = &setup.fcf.pools[pool_id];
        assert_eq!(
            pool.warm_start_count as usize,
            boundary_cuts.len(),
            "every terminal fan leaf's own pool must carry the injected boundary cuts"
        );
        let loaded: Vec<(f64, Vec<f64>)> = pool
            .active_cuts()
            .map(|(_, intercept, coeffs)| (intercept, coeffs.to_vec()))
            .collect();
        let expected: Vec<(f64, Vec<f64>)> = boundary_cuts
            .iter()
            .map(|c| (c.intercept, c.coefficients.clone()))
            .collect();
        assert_eq!(
            loaded, expected,
            "the shared pool's cut content must match the loaded boundary cuts exactly — one \
             value function applied to every terminal fan node, not a per-leaf distinct policy"
        );
    }
}

#[test]
fn multi_node_shared_pool_is_rejected() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let source_dir = tmp.path().join("multi_node_source");
    // Two declared nodes (ids 200, 201) share pool 0: the pool's own node_id
    // reads the STAGE_CUTS_NODE_ID_SENTINEL rather than a single owner.
    write_synthetic_checkpoint(&source_dir, &[(200, 5, 0), (201, 5, 0)], 0, &[7.0, 11.0], 2);

    let result = load_boundary_cuts(&BoundaryLoadRequest::new(
        &source_dir,
        fixture_priced_date(0),
        2,
        &[],
        LEGACY_COST_SCALE_FACTOR,
    ));

    let err = result.expect_err(
        "a source pool shared by multiple nodes must be rejected, not silently resolved to one \
         of them",
    );
    let msg = err.to_string();
    assert!(
        msg.contains("single-node terminal pool") && msg.contains("shared by multiple nodes"),
        "rejection must name the shared-pool node_id guard: {msg}"
    );
    assert!(
        msg.contains(&source_dir.display().to_string()),
        "rejection must name the checkpoint path: {msg}"
    );
}
