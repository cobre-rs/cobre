//! Generalized patched-template capture (`capture_patched_node_template_at`,
//! `node_opening_noise`) must agree with the existing
//! node-capture helpers and must produce the raw-noise length every opening
//! of a node expects. The one-hot patch-ownership sweep
//! (`every_noise_dimension_patches_only_its_own_entity`) then uses that
//! capture to assert, over every committed deck plus a stochastic and a
//! chronological-noise in-code fixture, that each noise dimension patches
//! only the row/column family its own entity owns. A third test pins the
//! lower bound's root-opening LPs to the forward pass's own patched root
//! templates, bound by bound. A fourth test,
//! `every_deck_workspace_pool_is_sized_from_its_owners`, pins each deck's
//! workspace pool to the sizes its owners declare.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::collections::BTreeMap;
use std::ops::Range;
use std::path::Path;

use cobre_core::{BlockMode, EntityId};
use cobre_sddp::StudySetup;
use cobre_sddp::indexer::{BlockIdx, BusSys, HydroSys, NcsSys, StateSpace};
use cobre_sddp::lp::StageGeometry;
use cobre_sddp::setup::{NodePos, StageIdx};
use cobre_sddp::test_support::decks::{Deck, SLOW_DECKS, committed_decks};
use cobre_sddp::test_support::{
    capture_patched_node_template, capture_patched_node_template_at, lower_bound_root_templates,
    node_opening_noise, oracle_initial_state, stage_state_box_bounds, state_space,
};
use cobre_solver::{ActiveSolver, StageTemplate};

use common::in_code_studies::{
    ChronologicalNoiseSpec, chronological_noise_study, mixed_lead_anticipated_study,
    stochastic_parallel_study,
};
use common::{build_setup_in_code, fresh_setup_with};

#[test]
fn capture_at_initial_state_matches_node_capture() {
    let case_dir =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples/deterministic/d02-single-hydro");
    let setup = fresh_setup_with(&case_dir, |_| {});
    let zero_noise = vec![0.0_f64; setup.inputs.stochastic.dim()];
    let initial_state = oracle_initial_state(&setup);

    for pos in (0..setup.inputs.node_graph.nodes.len()).map(NodePos) {
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
    let setup = fresh_setup_with(&case_dir, |_| {});
    let expected_len = setup.inputs.stochastic.dim();

    for pos in (0..setup.inputs.node_graph.nodes.len()).map(NodePos) {
        let openings = setup.inputs.node_graph.nodes[pos].openings;
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

fn z_row(state: &StateSpace, h: usize) -> usize {
    state.z_inflow_row(HydroSys::new(h))
}

fn load_chunk(geom: &StageGeometry, bus_pos: usize) -> Range<usize> {
    let bus = BusSys::new(bus_pos);
    let first = geom.load_balance_row(bus, BlockIdx::new(0));
    let last = geom.load_balance_row(bus, BlockIdx::new(geom.n_blks - 1));
    first..last + 1
}

fn ncs_chunk(geom: &StageGeometry, sys_idx: usize) -> Range<usize> {
    let ncs = NcsSys::new(sys_idx);
    geom.ncs_generation_col(ncs, BlockIdx::new(0))
        ..geom.ncs_generation_col(ncs, BlockIdx::new(geom.n_blks - 1)) + 1
}

/// Maps stochastic NCS slot `r` (`setup.inputs.stochastic.ncs_entity_ids()[r]`) to its
/// dense system index — its position in `system_ncs_ids`, the deck's own
/// `non_controllable_sources()` in canonical order — the same lookup
/// `build_ncs_entity_data` uses to size the `pub(crate)`
/// `StudySetup::ncs_stochastic_dense_col`, unreachable from this integration
/// test, so this mirrors it through the public `System`/`StochasticContext`
/// accessors instead. Indexing `r` directly into `system_ncs_ids` is the
/// wrong-but-compiling alternative: `ncs_entity_ids` sorts by ID alone while
/// `non_controllable_sources` sorts by `(operational_start_date, id)`, so the
/// two orders diverge whenever the system's NCS entities disagree on start date.
fn ncs_dense_col_map(system_ncs_ids: &[EntityId], setup: &StudySetup) -> Vec<usize> {
    setup
        .inputs
        .stochastic
        .ncs_entity_ids()
        .iter()
        .map(|id| {
            system_ncs_ids
                .iter()
                .position(|sys_id| sys_id == id)
                .expect("stochastic NCS entity id must exist in the system's NCS list")
        })
        .collect()
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
    state: &StateSpace,
    changed: &ChangedIndices,
    violations: &mut Vec<String>,
) {
    if !changed.cols.is_empty() {
        violations.push(format!(
            "{deck} node={pos} dim={dim} (inflow h={h}): unexpected column changes {:?}",
            changed.cols
        ));
    }
    let z = z_row(state, h);
    for &row in &changed.rows {
        if row != z {
            violations.push(format!(
                "{deck} node={pos} dim={dim} (inflow h={h}): row {row} outside z row {z}"
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

fn check_ncs(
    deck: &str,
    pos: NodePos,
    dim: usize,
    sys_idx: usize,
    geom: &StageGeometry,
    changed: &ChangedIndices,
    violations: &mut Vec<String>,
) {
    if !changed.rows.is_empty() {
        violations.push(format!(
            "{deck} node={pos} dim={dim} (ncs sys_idx={sys_idx}): unexpected row changes {:?}",
            changed.rows
        ));
    }
    let chunk = ncs_chunk(geom, sys_idx);
    for &col in &changed.cols {
        if !chunk.contains(&col) {
            violations.push(format!(
                "{deck} node={pos} dim={dim} (ncs sys_idx={sys_idx}): column {col} outside \
                 single ncs chunk {chunk:?}"
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
    ncs_dense_col: &[usize],
    violations: &mut Vec<String>,
    vacuity: &mut BTreeMap<(&'static str, &'static str), usize>,
) {
    let dims = setup.inputs.stochastic.class_dimensions();
    let n_dims = dims.total();
    let initial_state = oracle_initial_state(setup);
    let zero_noise = vec![0.0_f64; n_dims];

    for pos in (0..setup.inputs.node_graph.nodes.len()).map(NodePos) {
        let stage = setup.inputs.node_graph.nodes[pos].stage.0;
        let geom = &setup.inputs.stage_data.stage_templates.geometry_per_stage[stage];
        let mode_tag = block_mode_tag(geom.block_mode);

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

            if dims.hydro_range().contains(&dim) {
                *vacuity.entry((mode_tag, "inflow")).or_insert(0) += 1;
                check_inflow(
                    deck_key,
                    pos,
                    dim,
                    dim,
                    state_space(setup),
                    &changed,
                    violations,
                );
            } else if dims.load_bus_range().contains(&dim) {
                *vacuity.entry((mode_tag, "load")).or_insert(0) += 1;
                let bus_pos = setup.inputs.stage_data.stage_templates.load_bus_indices
                    [dim - dims.load_bus_range().start];
                check_load(deck_key, pos, dim, bus_pos, geom, &changed, violations);
            } else {
                *vacuity.entry((mode_tag, "ncs")).or_insert(0) += 1;
                let r = dim - dims.ncs_range().start;
                let sys_idx = *ncs_dense_col.get(r).unwrap_or_else(|| {
                    panic!(
                        "{deck_key}: no dense-column mapping for stochastic NCS slot {r} \
                         (dim {dim}); pass an ncs_dense_col sized to n_stochastic_ncs"
                    )
                });
                check_ncs(deck_key, pos, dim, sys_idx, geom, &changed, violations);
            }
        }
    }
}

/// On every deck, each noise dimension, on its own, changes only the bounds
/// the layout assigns to that dimension's entity. Ownership is defined by
/// row/column families (`z_row`/`load_chunk`/`ncs_chunk`
/// above), never by the patch buffers themselves.
#[test]
fn every_noise_dimension_patches_only_its_own_entity() {
    let slow_tests_enabled = cfg!(feature = "slow-tests");
    let mut violations: Vec<String> = Vec::new();
    let mut vacuity: BTreeMap<(&'static str, &'static str), usize> = BTreeMap::new();
    let mut swept: Vec<String> = Vec::new();

    for deck in committed_decks() {
        if !slow_tests_enabled && SLOW_DECKS.contains(&deck.key.as_str()) {
            continue;
        }
        let setup = fresh_setup_with(&deck.dir, |_| {});
        swept.push(deck.key.clone());
        sweep_setup(&deck.key, &setup, &[], &mut violations, &mut vacuity);
    }

    let (stochastic_system, stochastic_config) = stochastic_parallel_study();
    let system_ncs_ids: Vec<EntityId> = stochastic_system
        .non_controllable_sources()
        .iter()
        .map(|n| n.id)
        .collect();
    let stochastic_setup = build_setup_in_code(stochastic_system, &stochastic_config);
    let ncs_dense_col = ncs_dense_col_map(&system_ncs_ids, &stochastic_setup);
    let label = "in-code/stochastic-parallel";
    swept.push(label.to_string());
    sweep_setup(
        label,
        &stochastic_setup,
        &ncs_dense_col,
        &mut violations,
        &mut vacuity,
    );

    let (chronological_system, chronological_config) =
        chronological_noise_study(&ChronologicalNoiseSpec::default());
    let chronological_setup = build_setup_in_code(chronological_system, &chronological_config);
    let label = "in-code/chronological-noise";
    swept.push(label.to_string());
    sweep_setup(
        label,
        &chronological_setup,
        &[],
        &mut violations,
        &mut vacuity,
    );

    let (mixed_lead_system, mixed_lead_config) = mixed_lead_anticipated_study(false);
    let mixed_lead_setup = build_setup_in_code(mixed_lead_system, &mixed_lead_config);
    let label = "in-code/mixed-lead-anticipated";
    swept.push(label.to_string());
    sweep_setup(label, &mixed_lead_setup, &[], &mut violations, &mut vacuity);

    let (chronological_pumping_system, chronological_pumping_config) =
        chronological_noise_study(&ChronologicalNoiseSpec {
            pumping_station: true,
            ..ChronologicalNoiseSpec::default()
        });
    let chronological_pumping_setup =
        build_setup_in_code(chronological_pumping_system, &chronological_pumping_config);
    let label = "in-code/chronological-pumping";
    swept.push(label.to_string());
    sweep_setup(
        label,
        &chronological_pumping_setup,
        &[],
        &mut violations,
        &mut vacuity,
    );

    eprintln!("swept labels: {swept:?}");
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
    let chronological_inflow_count = vacuity
        .get(&("Chronological", "inflow"))
        .copied()
        .unwrap_or(0);
    assert!(
        chronological_inflow_count >= 1,
        "vacuity guard: (Chronological, inflow) has no nonvacuous (deck, node, dimension) \
         triple — the sweep has no power on this family"
    );

    assert!(
        violations.is_empty(),
        "patch-ownership violations:\n{}",
        violations.join("\n")
    );
}

// ── Lower bound vs. forward root LP ──────────────────────────────────────────

fn root_node(setup: &StudySetup) -> NodePos {
    let graph = &setup.inputs.node_graph;
    (0..graph.nodes.len())
        .map(NodePos)
        .find(|&pos| graph.nodes[pos].stage == StageIdx(0))
        .expect("study must have a stage-0 node")
}

fn compare_bound_vec(
    deck: &str,
    opening: usize,
    kind: &str,
    lower_bound: &[f64],
    forward: &[f64],
    violations: &mut Vec<String>,
) {
    assert_eq!(
        lower_bound.len(),
        forward.len(),
        "{deck} opening={opening} {kind}: length mismatch"
    );
    for (i, (&lb, &fwd)) in lower_bound.iter().zip(forward).enumerate() {
        if lb.to_bits() != fwd.to_bits() {
            violations.push(format!(
                "{deck} opening={opening} {kind}[{i}]: lower_bound={lb} forward={fwd}"
            ));
        }
    }
}

/// Compares, for every root opening of `setup`, the lower bound's recorded
/// root template ([`lower_bound_root_templates`]) against the forward pass's
/// own patched root template at the same opening's draw
/// ([`node_opening_noise`] through `capture_patched_node_template_at`) on
/// `col_lower`/`col_upper`/`row_lower`/`row_upper` by `to_bits`. Appends one
/// line per mismatch to `violations`; returns the number of openings compared.
fn compare_lower_bound_to_forward_root_lp(
    deck: &str,
    setup: &StudySetup,
    violations: &mut Vec<String>,
) -> usize {
    let root = root_node(setup);
    let n_openings = setup.inputs.node_graph.nodes[root].openings.len;
    let initial_state = oracle_initial_state(setup);

    let recorded = lower_bound_root_templates(
        setup,
        ActiveSolver::new().expect("ActiveSolver::new must succeed"),
    )
    .expect("lower_bound_root_templates must succeed");
    assert_eq!(
        recorded.len(),
        n_openings,
        "{deck}: lower bound recorded {} root templates but the root has {n_openings} openings",
        recorded.len()
    );

    for (j, lb_template) in recorded.iter().enumerate() {
        let raw_noise = node_opening_noise(setup, root, j);
        let forward = capture_patched_node_template_at(setup, root, &raw_noise, &initial_state);

        for (kind, lb, fwd) in [
            ("col_lower", &lb_template.col_lower, &forward.col_lower),
            ("col_upper", &lb_template.col_upper, &forward.col_upper),
            ("row_lower", &lb_template.row_lower, &forward.row_lower),
            ("row_upper", &lb_template.row_upper, &forward.row_upper),
        ] {
            compare_bound_vec(deck, j, kind, lb, fwd, violations);
        }
    }
    n_openings
}

/// Runs [`compare_lower_bound_to_forward_root_lp`] over every committed deck
/// (skipping `SLOW_DECKS` unless `slow-tests`), `stochastic_parallel_study()`,
/// and the chronological-noise study; returns the total openings compared
/// and every mismatch line.
fn sweep_lower_bound_vs_forward_root_lp() -> (usize, Vec<String>) {
    let slow_tests_enabled = cfg!(feature = "slow-tests");
    let mut violations: Vec<String> = Vec::new();
    let mut n_compared = 0usize;

    for deck in committed_decks() {
        if !slow_tests_enabled && SLOW_DECKS.contains(&deck.key.as_str()) {
            continue;
        }
        let setup = fresh_setup_with(&deck.dir, |_| {});
        n_compared += compare_lower_bound_to_forward_root_lp(&deck.key, &setup, &mut violations);
    }

    let (stochastic_system, stochastic_config) = stochastic_parallel_study();
    let stochastic_setup = build_setup_in_code(stochastic_system, &stochastic_config);
    n_compared += compare_lower_bound_to_forward_root_lp(
        "in-code/stochastic-parallel",
        &stochastic_setup,
        &mut violations,
    );

    let (chronological_system, chronological_config) =
        chronological_noise_study(&ChronologicalNoiseSpec::default());
    let chronological_setup = build_setup_in_code(chronological_system, &chronological_config);
    n_compared += compare_lower_bound_to_forward_root_lp(
        "in-code/chronological-noise",
        &chronological_setup,
        &mut violations,
    );

    (n_compared, violations)
}

#[test]
fn lower_bound_root_lp_matches_the_forward_root_lp() {
    let (n_compared, violations) = sweep_lower_bound_vs_forward_root_lp();
    assert!(n_compared >= 1, "vacuity guard: no root opening compared");
    assert!(
        violations.is_empty(),
        "lower-bound vs forward root LP mismatches:\n{}",
        violations.join("\n")
    );
}

fn active_decks() -> Vec<Deck> {
    let slow_tests_enabled = cfg!(feature = "slow-tests");
    committed_decks()
        .into_iter()
        .filter(|deck| slow_tests_enabled || !SLOW_DECKS.contains(&deck.key.as_str()))
        .collect()
}

/// Build `deck`'s [`StudySetup`], panicking with the deck's key on any
/// build failure — no deck is ever skipped silently.
fn build_deck_or_panic(deck: &Deck) -> StudySetup {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        fresh_setup_with(&deck.dir, |_| {})
    }))
    .unwrap_or_else(|payload| {
        let msg = payload
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| payload.downcast_ref::<&str>().copied())
            .unwrap_or("<non-string panic payload>");
        panic!("deck {} failed to build: {msg}", deck.key);
    })
}

#[test]
fn every_deck_workspace_pool_is_sized_from_its_owners() {
    use cobre_comm::LocalBackend;
    use cobre_sddp::test_support::workspace_downstream_lag_shape;

    let mut any_downstream_par_order_positive = false;

    for deck in active_decks() {
        let setup = build_deck_or_panic(&deck);
        let stage_ctx = setup.stage_ctx();
        let training_ctx = setup.training_ctx();
        let state = training_ctx.state;
        let max_n_blks = stage_ctx
            .geometry_per_stage
            .iter()
            .map(|g| g.n_blks)
            .max()
            .unwrap_or(0);

        any_downstream_par_order_positive |= training_ctx.study_dims.downstream_par_order > 0;

        let comm = LocalBackend;
        let pool = setup
            .create_workspace_pool(&comm, 1, ActiveSolver::new)
            .unwrap_or_else(|e| panic!("deck {}: workspace pool: {e:?}", deck.key));

        for ws in &pool.workspaces {
            assert_eq!(
                ws.patch_buf.indices.len(),
                stage_ctx.load_bus_indices.len() * max_n_blks + state.hydro_count,
                "deck {}: patch_buf.indices length",
                deck.key
            );
            assert_eq!(
                ws.patch_buf.col_indices.len(),
                state.hydro_count * (1 + state.max_par_order)
                    + state.n_buckets
                    + state.n_anticipated * state.k_max,
                "deck {}: patch_buf.col_indices length",
                deck.key
            );
            assert!(
                ws.current_state.capacity() >= state.n_state,
                "deck {}: current_state capacity",
                deck.key
            );
            let (downstream_completed_lags_len, lag_accumulator_len) =
                workspace_downstream_lag_shape(ws);
            assert_eq!(
                downstream_completed_lags_len,
                lag_accumulator_len * training_ctx.study_dims.downstream_par_order,
                "deck {}: downstream_completed_lags length",
                deck.key
            );
        }

        assert_eq!(
            stage_ctx.templates.len(),
            training_ctx.horizon.num_stages(),
            "deck {}: stage template count",
            deck.key
        );
    }

    assert!(
        any_downstream_par_order_positive,
        "at least one swept deck must have downstream_par_order > 0"
    );
}
