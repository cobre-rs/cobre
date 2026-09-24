//! Generalized patched-template capture (`capture_patched_node_template_at`,
//! `raw_noise_len`, `node_opening_noise`) must agree with the existing
//! node-capture helpers and must produce the raw-noise length every opening
//! of a node expects.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::path::Path;

use cobre_sddp::setup::NodePos;
use cobre_sddp::test_support::{
    capture_patched_node_template, capture_patched_node_template_at, node_opening_noise,
    oracle_initial_state, raw_noise_len,
};

#[test]
fn capture_at_initial_state_matches_node_capture() {
    let case_dir =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples/deterministic/d02-single-hydro");
    let setup = common::fresh_setup_with(&case_dir, |_| {});
    let zero_noise = vec![0.0_f64; raw_noise_len(&setup)];
    let initial_state = oracle_initial_state(&setup);

    for pos in (0..setup.node_graph.nodes.len()).map(NodePos) {
        let at = capture_patched_node_template_at(&setup, pos, &zero_noise, &initial_state);
        let node = capture_patched_node_template(&setup, pos);

        assert_eq!(to_bits(&at.row_lower), to_bits(&node.row_lower));
        assert_eq!(to_bits(&at.row_upper), to_bits(&node.row_upper));
        assert_eq!(to_bits(&at.col_lower), to_bits(&node.col_lower));
        assert_eq!(to_bits(&at.col_upper), to_bits(&node.col_upper));
    }
}

#[test]
fn node_opening_noise_has_the_raw_noise_length() {
    let case_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples/1dtoy");
    let setup = common::fresh_setup_with(&case_dir, |_| {});
    let expected_len = raw_noise_len(&setup);

    for pos in (0..setup.node_graph.nodes.len()).map(NodePos) {
        let openings = setup.node_graph.nodes[pos].openings;
        for opening in 0..openings.len {
            assert_eq!(node_opening_noise(&setup, pos, opening).len(), expected_len);
        }
    }
}

fn to_bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}
