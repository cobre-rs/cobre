//! Chronological inflow noise must reach the water balance of the hydro it was
//! drawn for, and only that hydro.
//!
//! Two independent hydros on a two-block chronological stage, neither able to
//! release water profitably (zero load, positive turbining and spillage costs, a
//! wide storage box), so each hydro's end-of-stage storage is its initial storage
//! plus the stage inflow volume. A one-hot standardized draw on hydro 1 must then
//! raise hydro 1's end storage by `ζ · σ₁` hm³ and leave hydro 0's untouched. The
//! assertions read the solved LP, not row positions, so they hold for any
//! encoding of the inflow in the water balance.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::too_many_lines
)]

mod common;

use cobre_sddp::StudySetup;
use cobre_sddp::indexer::StateDim;
use cobre_sddp::setup::{NodePos, StageIdx};
use cobre_sddp::test_support::{
    capture_patched_node_template_at, capture_patched_node_template_with_inflow_noise,
    no_cut_root_lower_bound, node_opening_noise, oracle_initial_state,
};
use cobre_solver::{ActiveSolver, SolverInterface};

use common::build_setup_in_code;
use common::in_code_studies::{
    CHRONOLOGICAL_NOISE_BLOCK_HOURS, CHRONOLOGICAL_NOISE_INFLOW_MEAN_M3S,
    CHRONOLOGICAL_NOISE_INFLOW_STD_M3S, CHRONOLOGICAL_NOISE_INITIAL_STORAGE_HM3,
    CHRONOLOGICAL_NOISE_RELEASE_COST, ChronologicalNoiseSpec, chronological_noise_study,
};

const M3S_TO_HM3: f64 = 0.0036;

fn root_node(setup: &StudySetup) -> NodePos {
    let graph = &setup.inputs.node_graph;
    (0..graph.nodes.len())
        .map(NodePos)
        .find(|&pos| graph.nodes[pos].stage == StageIdx(0))
        .expect("study must have a stage-0 node")
}

fn end_storage_hm3(setup: &StudySetup, inflow_eta: &[f64]) -> Vec<f64> {
    let template =
        capture_patched_node_template_with_inflow_noise(setup, root_node(setup), inflow_eta);
    let mut solver = ActiveSolver::new().expect("ActiveSolver::new");
    solver.load_model(&template);
    let view = solver.solve(None).expect("root stage LP must solve");
    let state = setup.stage_state();
    (0..CHRONOLOGICAL_NOISE_INFLOW_STD_M3S.len())
        .map(|h| {
            let col = state.lp_column_for_state(StateDim::new(h)).get();
            let scale = template.col_scale.get(col).copied().unwrap_or(1.0);
            view.primal[col] * scale
        })
        .collect()
}

#[test]
fn chronological_inflow_noise_moves_only_its_own_hydro() {
    let (system, config) = chronological_noise_study(&ChronologicalNoiseSpec::default());
    let setup = build_setup_in_code(system, &config);
    let zeta_hm3_per_m3s: f64 = CHRONOLOGICAL_NOISE_BLOCK_HOURS.iter().sum::<f64>() * M3S_TO_HM3;

    let baseline = end_storage_hm3(&setup, &[0.0, 0.0]);
    for (h, &storage) in baseline.iter().enumerate() {
        let expected = CHRONOLOGICAL_NOISE_INITIAL_STORAGE_HM3
            + zeta_hm3_per_m3s * CHRONOLOGICAL_NOISE_INFLOW_MEAN_M3S;
        assert!(
            (storage - expected).abs() < 1e-6,
            "hydro {h}: with a zero draw the stage must store its mean inflow, \
             expected {expected}, got {storage}"
        );
    }

    let shocked = end_storage_hm3(&setup, &[0.0, 1.0]);
    let delta: Vec<f64> = shocked.iter().zip(&baseline).map(|(s, b)| s - b).collect();
    let expected_delta = [
        0.0,
        zeta_hm3_per_m3s * CHRONOLOGICAL_NOISE_INFLOW_STD_M3S[1],
    ];
    for h in 0..CHRONOLOGICAL_NOISE_INFLOW_STD_M3S.len() {
        assert!(
            (delta[h] - expected_delta[h]).abs() < 1e-6,
            "a one-hot draw on hydro 1 must change hydro {h}'s end storage by \
             {} hm³, got {} (all deltas: {delta:?})",
            expected_delta[h],
            delta[h]
        );
    }
}

#[test]
fn chronological_noise_lower_bound_is_the_mean_root_objective() {
    let spec = ChronologicalNoiseSpec {
        max_storage_hm3: 1000.0,
        branching_factor: 3,
        ..ChronologicalNoiseSpec::default()
    };
    let (system, config) = chronological_noise_study(&spec);
    let setup = build_setup_in_code(system, &config);
    let root = root_node(&setup);
    let n = setup.inputs.node_graph.nodes[root].openings.len;
    let n_hydros = CHRONOLOGICAL_NOISE_INFLOW_STD_M3S.len();

    let etas: Vec<Vec<f64>> = (0..n)
        .map(|j| node_opening_noise(&setup, root, j))
        .collect();

    let mut sum_z = 0.0_f64;
    for (j, eta) in etas.iter().enumerate() {
        for h in 0..n_hydros {
            let z_hj = CHRONOLOGICAL_NOISE_INFLOW_MEAN_M3S
                + CHRONOLOGICAL_NOISE_INFLOW_STD_M3S[h] * eta[h];
            assert!(
                z_hj > 0.0,
                "opening {j} hydro {h}: the analytic release value assumes a positive \
                 realized inflow, got {z_hj} (eta {eta:?})"
            );
            sum_z += z_hj;
        }
    }

    let initial_state = oracle_initial_state(&setup);
    let mut probe_solver = ActiveSolver::new().expect("ActiveSolver::new");
    let sum_objective: f64 = etas
        .iter()
        .map(|eta| {
            let template = capture_patched_node_template_at(&setup, root, eta, &initial_state);
            probe_solver.load_model(&template);
            probe_solver
                .solve(None)
                .expect("root stage LP must solve")
                .objective
        })
        .sum();

    let n_f64 = n as f64;
    let cost_scale_factor = setup.inputs.stage_data.stage_templates.cost_scale_factor;
    let expected_from_solves = sum_objective / n_f64 * cost_scale_factor;

    let mut lb_solver = ActiveSolver::new().expect("ActiveSolver::new");
    let lb = no_cut_root_lower_bound(&setup, &mut lb_solver).expect("no_cut_root_lower_bound");

    assert!(
        (lb - expected_from_solves).abs() <= 1e-9 * lb.abs().max(1.0),
        "lower bound {lb} must equal the mean cold-solved root objective {expected_from_solves}"
    );

    let analytic = CHRONOLOGICAL_NOISE_RELEASE_COST
        * CHRONOLOGICAL_NOISE_BLOCK_HOURS.iter().sum::<f64>()
        * sum_z
        / n_f64;
    assert!(
        (lb - analytic).abs() <= 1e-6 * lb.abs(),
        "lower bound {lb} must equal the analytic inflow-release cost {analytic}"
    );
}
