//! Generalized patched-template capture (`capture_patched_node_template_at`,
//! `raw_noise_len`, `node_opening_noise`) must agree with the existing
//! node-capture helpers and must produce the raw-noise length every opening
//! of a node expects.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::collections::BTreeMap;
use std::ops::Range;
use std::path::Path;

use cobre_core::BlockMode;
use cobre_sddp::StudySetup;
use cobre_sddp::lp::StageGeometry;
use cobre_sddp::setup::NodePos;
use cobre_sddp::test_support::{
    capture_patched_node_template, capture_patched_node_template_at, node_opening_noise,
    oracle_initial_state, raw_noise_len, stage_state_box_bounds,
};
use cobre_solver::StageTemplate;

use common::decks::{SLOW_DECKS, committed_decks};
use common::in_code_studies::stochastic_parallel_study;
use common::{build_setup_in_code, fresh_setup_with};

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

// ── One-hot patch-ownership sweep ────────────────────────────────────────────
//
// One address-arithmetic helper per family despite each having a single call
// site below: keeps a later change to an ownership set confined to one
// function instead of every check site.

fn water_chunk(geom: &StageGeometry, n_hydros: usize, h: usize) -> Range<usize> {
    assert_eq!(
        geom.water_balance.len() % n_hydros,
        0,
        "water_balance row count must be a multiple of the hydro count"
    );
    let per = geom.water_balance.len() / n_hydros;
    let start = geom.water_balance.start + h * per;
    start..start + per
}

fn z_row(geom: &StageGeometry, h: usize) -> usize {
    geom.z_inflow_row_start + h
}

fn load_chunk(geom: &StageGeometry, bus_pos: usize) -> Range<usize> {
    let start = geom.load_balance.start + bus_pos * geom.n_blks;
    start..start + geom.n_blks
}

fn ncs_chunk(ncs_col_start: usize, n_blks: usize, slot: usize) -> Range<usize> {
    let start = ncs_col_start + slot * n_blks;
    start..start + n_blks
}

/// LP row/column indices whose bounds differ between two templates (compared
/// by bit pattern, so a `-0.0`/`0.0` change is visible).
struct ChangedIndices {
    rows: Vec<usize>,
    cols: Vec<usize>,
}

fn changed_indices(base: &StageTemplate, patched: &StageTemplate) -> ChangedIndices {
    let rows = (0..base.row_lower.len())
        .filter(|&i| {
            base.row_lower[i].to_bits() != patched.row_lower[i].to_bits()
                || base.row_upper[i].to_bits() != patched.row_upper[i].to_bits()
        })
        .collect();
    let cols = (0..base.col_lower.len())
        .filter(|&i| {
            base.col_lower[i].to_bits() != patched.col_lower[i].to_bits()
                || base.col_upper[i].to_bits() != patched.col_upper[i].to_bits()
        })
        .collect();
    ChangedIndices { rows, cols }
}

fn block_mode_tag(mode: BlockMode) -> &'static str {
    match mode {
        BlockMode::Parallel => "Parallel",
        BlockMode::Chronological => "Chronological",
    }
}

fn check_inflow(
    deck: &str,
    pos: NodePos,
    dim: usize,
    h: usize,
    geom: &StageGeometry,
    n_hydros: usize,
    changed: &ChangedIndices,
    violations: &mut Vec<String>,
) {
    if !changed.cols.is_empty() {
        violations.push(format!(
            "{deck} node={pos} dim={dim} (inflow h={h}): unexpected column changes {:?}",
            changed.cols
        ));
    }
    let chunk = water_chunk(geom, n_hydros, h);
    let z = z_row(geom, h);
    for &row in &changed.rows {
        if !(chunk.contains(&row) || row == z) {
            violations.push(format!(
                "{deck} node={pos} dim={dim} (inflow h={h}): row {row} outside water chunk \
                 {chunk:?} and z row {z}"
            ));
        }
    }
}

fn check_load(
    deck: &str,
    pos: NodePos,
    dim: usize,
    bus_pos: usize,
    geom: &StageGeometry,
    changed: &ChangedIndices,
    violations: &mut Vec<String>,
) {
    if !changed.cols.is_empty() {
        violations.push(format!(
            "{deck} node={pos} dim={dim} (load bus_pos={bus_pos}): unexpected column changes {:?}",
            changed.cols
        ));
    }
    let chunk = load_chunk(geom, bus_pos);
    for &row in &changed.rows {
        if !chunk.contains(&row) {
            violations.push(format!(
                "{deck} node={pos} dim={dim} (load bus_pos={bus_pos}): row {row} outside load \
                 chunk {chunk:?}"
            ));
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn check_ncs(
    deck: &str,
    pos: NodePos,
    dim: usize,
    ncs_col_start: usize,
    n_blks: usize,
    n_ncs: usize,
    changed: &ChangedIndices,
    violations: &mut Vec<String>,
) {
    if !changed.rows.is_empty() {
        violations.push(format!(
            "{deck} node={pos} dim={dim} (ncs): unexpected row changes {:?}",
            changed.rows
        ));
    }
    let Some(&first) = changed.cols.first() else {
        return;
    };
    let Some(rel) = first.checked_sub(ncs_col_start) else {
        violations.push(format!(
            "{deck} node={pos} dim={dim} (ncs): column {first} precedes the NCS family start \
             {ncs_col_start}"
        ));
        return;
    };
    let slot = rel / n_blks;
    if slot >= n_ncs {
        violations.push(format!(
            "{deck} node={pos} dim={dim} (ncs): column {first} falls outside the {n_ncs}-slot \
             NCS family starting at {ncs_col_start}"
        ));
        return;
    }
    let chunk = ncs_chunk(ncs_col_start, n_blks, slot);
    for &col in &changed.cols {
        if !chunk.contains(&col) {
            violations.push(format!(
                "{deck} node={pos} dim={dim} (ncs): column {col} outside single ncs chunk \
                 {chunk:?}"
            ));
        }
    }
}

/// Sweep every node position and every raw-noise dimension of `setup`,
/// appending any cross-entity patch to `violations` and crediting each
/// nonvacuous `(deck, node, dimension)` triple to `vacuity`.
fn sweep_setup(
    deck_key: &str,
    setup: &StudySetup,
    violations: &mut Vec<String>,
    vacuity: &mut BTreeMap<(&'static str, &'static str), usize>,
) {
    let n_dims = raw_noise_len(setup);
    let n_load = setup.stage_data.stage_templates.n_load_buses;
    let n_ncs_stochastic = setup.stochastic.n_stochastic_ncs();
    let n_hydros = n_dims - n_load - n_ncs_stochastic;
    let initial_state = oracle_initial_state(setup);
    let zero_noise = vec![0.0_f64; n_dims];

    for pos in (0..setup.node_graph.nodes.len()).map(NodePos) {
        let stage = setup.node_graph.nodes[pos].stage.0;
        let geom = &setup.stage_data.stage_templates.geometry_per_stage[stage];
        let mode_tag = block_mode_tag(geom.block_mode);
        let ncs_col_start = setup.stage_data.stage_templates.ncs_col_starts[stage];
        let n_ncs_dense = setup.stage_data.stage_templates.n_ncs;

        let (lo, hi) = stage_state_box_bounds(setup, stage.saturating_sub(1));
        let incoming_state: Vec<f64> = initial_state
            .iter()
            .zip(&lo)
            .zip(&hi)
            .map(|((&x, &l), &h)| x.max(l).min(h))
            .collect();

        let base = capture_patched_node_template_at(setup, pos, &zero_noise, &incoming_state);

        for dim in 0..n_dims {
            let mut one_hot = zero_noise.clone();
            one_hot[dim] = 1.0;
            let patched = capture_patched_node_template_at(setup, pos, &one_hot, &incoming_state);
            let changed = changed_indices(&base, &patched);
            if changed.rows.is_empty() && changed.cols.is_empty() {
                continue;
            }

            if dim < n_hydros {
                *vacuity.entry((mode_tag, "inflow")).or_insert(0) += 1;
                check_inflow(
                    deck_key, pos, dim, dim, geom, n_hydros, &changed, violations,
                );
            } else if dim < n_hydros + n_load {
                *vacuity.entry((mode_tag, "load")).or_insert(0) += 1;
                let bus_pos = setup.stage_data.stage_templates.load_bus_indices[dim - n_hydros];
                check_load(deck_key, pos, dim, bus_pos, geom, &changed, violations);
            } else {
                *vacuity.entry((mode_tag, "ncs")).or_insert(0) += 1;
                check_ncs(
                    deck_key,
                    pos,
                    dim,
                    ncs_col_start,
                    geom.n_blks,
                    n_ncs_dense,
                    &changed,
                    violations,
                );
            }
        }
    }
}

/// On every deck, each noise dimension, on its own, changes only the bounds
/// the layout assigns to that dimension's entity. Ownership is defined by
/// row/column families (`water_chunk`/`z_row`/`load_chunk`/`ncs_chunk`
/// above), never by the patch buffers themselves.
#[test]
fn every_noise_dimension_patches_only_its_own_entity() {
    let slow_tests_enabled = cfg!(feature = "slow-tests");
    let mut violations: Vec<String> = Vec::new();
    let mut vacuity: BTreeMap<(&'static str, &'static str), usize> = BTreeMap::new();

    for deck in committed_decks() {
        if !slow_tests_enabled && SLOW_DECKS.contains(&deck.key.as_str()) {
            continue;
        }
        let setup = fresh_setup_with(&deck.dir, |_| {});
        sweep_setup(&deck.key, &setup, &mut violations, &mut vacuity);
    }

    let (stochastic_system, stochastic_config) = stochastic_parallel_study();
    let stochastic_setup = build_setup_in_code(stochastic_system, &stochastic_config);
    sweep_setup(
        "in-code/stochastic-parallel",
        &stochastic_setup,
        &mut violations,
        &mut vacuity,
    );

    eprintln!("patch-ownership vacuity counts, (block_mode, family) -> nonvacuous triples:");
    for (&(mode, family), count) in &vacuity {
        eprintln!("  ({mode}, {family}): {count}");
    }

    for family in ["inflow", "load", "ncs"] {
        let count = vacuity.get(&("Parallel", family)).copied().unwrap_or(0);
        assert!(
            count >= 1,
            "vacuity guard: (Parallel, {family}) has no nonvacuous (deck, node, dimension) \
             triple — the sweep has no power on this family"
        );
    }

    assert!(
        violations.is_empty(),
        "patch-ownership violations:\n{}",
        violations.join("\n")
    );
}
