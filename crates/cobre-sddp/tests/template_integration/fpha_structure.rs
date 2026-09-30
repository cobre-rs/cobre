//! `fpha_structure` section tests.

use super::*;
use cobre_solver::{ActiveSolver, RowBatch, SolverInterface};

fn empty_cut_batch() -> RowBatch {
    RowBatch {
        num_rows: 0,
        row_starts: vec![0_i32],
        col_indices: vec![],
        values: vec![],
        row_lower: vec![],
        row_upper: vec![],
    }
}

#[test]
fn fpha_ac1_dimensions_one_fpha_hydro_five_planes() {
    let (system, production) = one_fpha_hydro_system(5);

    let fpha_result = build_stage_templates_resolving_layout(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &production,
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("FPHA system ok");

    let const_result = build_stage_templates_resolving_layout(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("constant productivity ok");

    let fpha_tmpl = &fpha_result.templates[0];
    let const_tmpl = &const_result.templates[0];

    assert_eq!(
        fpha_tmpl.num_cols,
        const_tmpl.num_cols + 1,
        "FPHA adds exactly 1 generation column (1 hydro * 1 block)"
    );
    assert_eq!(
        fpha_tmpl.num_rows,
        const_tmpl.num_rows + 5,
        "FPHA adds exactly 5 constraint rows (5 planes * 1 block)"
    );
}

#[test]
fn fpha_ac2_generation_column_entries() {
    let n_planes = 5;
    let (system, production) = one_fpha_hydro_system(n_planes);

    let result = build_stage_templates_resolving_layout(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &production,
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("FPHA system ok");

    let tmpl = &result.templates[0];

    // N=1, L=0 → n_state = 1, decision_start = 4, col_generation_start = 9.
    // (turbine[4..5], spillage[5..6], diversion[6..7], deficit[7..8], excess[8..9], generation[9])
    let col_g = 9_usize;

    // row_fpha_start = 3 (= N z_inflow + N water balance + B*1 load balance)
    let row_fpha_start = 3_usize;

    for p in 0..n_planes {
        let row = row_fpha_start + p;
        let coeff = csc_entry(tmpl, col_g, row)
            .unwrap_or_else(|| panic!("generation column missing entry in FPHA row {row}"));
        assert!(
            (coeff - 1.0).abs() < 1e-12,
            "generation col FPHA row {row}: expected +1.0, got {coeff}"
        );
    }

    let row_lb = 2_usize; // load balance row
    let lb_coeff = csc_entry(tmpl, col_g, row_lb)
        .unwrap_or_else(|| panic!("generation column missing entry in load balance row {row_lb}"));
    assert!(
        (lb_coeff - 1.0).abs() < 1e-12,
        "generation col load balance row: expected +1.0, got {lb_coeff}"
    );
}

/// The `v_in` column carries `-gamma_v/2` in every FPHA row; gamma_v = 0.5 → -0.25.
#[test]
fn fpha_ac3_v_in_column_entries() {
    let n_planes = 5;
    let (system, production) = one_fpha_hydro_system(n_planes);

    let result = build_stage_templates_resolving_layout(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &production,
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("FPHA system ok");

    let tmpl = &result.templates[0];

    let col_v_in = 2_usize; // N=1, L=0: storage_in after z_inflow
    let row_fpha_start = 3_usize; // N z_inflow + N water balance + B*K load balance
    let expected = -0.5_f64 / 2.0;

    for p in 0..n_planes {
        let row = row_fpha_start + p;
        let coeff = csc_entry(tmpl, col_v_in, row)
            .unwrap_or_else(|| panic!("v_in column missing entry in FPHA row {row}"));
        assert!(
            (coeff - expected).abs() < 1e-12,
            "v_in col FPHA row {row}: expected {expected}, got {coeff}"
        );
    }
}

/// The outgoing-storage column `v` carries `-gamma_v/2` in every FPHA row; -0.25 here.
#[test]
fn fpha_ac4_v_out_column_entries() {
    let n_planes = 5;
    let (system, production) = one_fpha_hydro_system(n_planes);

    let result = build_stage_templates_resolving_layout(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &production,
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("FPHA system ok");

    let tmpl = &result.templates[0];

    let col_v = 0_usize; // outgoing storage is column 0 for N=1
    let row_fpha_start = 3_usize; // N z_inflow + N water balance + B*K load balance
    let expected = -0.5_f64 / 2.0;

    for p in 0..n_planes {
        let row = row_fpha_start + p;
        let coeff = csc_entry(tmpl, col_v, row)
            .unwrap_or_else(|| panic!("v column missing entry in FPHA row {row}"));
        assert!(
            (coeff - expected).abs() < 1e-12,
            "v col FPHA row {row}: expected {expected}, got {coeff}"
        );
    }
}

#[test]
fn fpha_ac5_mixed_system_load_balance_uses_generation_col() {
    let (system, production) = four_hydro_mixed_system();

    let result = build_stage_templates_resolving_layout(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &production,
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("mixed FPHA/constant system ok");

    let tmpl = &result.templates[0];

    // Base (constant) num_cols = 13 + 4*1*3 (turbine+spillage+diversion per hydro/block)
    // + 1*1*2 = 27; + 2 FPHA generation = 29; + 2*N=8 withdrawal = 37;
    // + 4*N=16 op-violation slacks = 53.
    // num_rows (no state-fixing rows) = N z_inflow(4) + N water balance(4) + load balance(1)
    // = 9; + 2*3 FPHA = 15; + 4*N=16 op-violation = 31.
    assert_eq!(
        tmpl.num_cols, 53,
        "4-hydro mixed system: num_cols should be 53 (includes diversion, 2*4 withdrawal, and operational slacks)"
    );
    assert_eq!(
        tmpl.num_rows, 31,
        "4-hydro mixed system: num_rows should be 31 (no state-fixing rows)"
    );

    let row_lb = 8_usize; // N z_inflow(4) + N water balance(4) + bus_blk_idx(0)

    // FPHA hydro h_idx=2 (local_idx=0): generation col 27 (= 23 + 4 diversion cols).
    let col_g_fpha0 = 27_usize;
    let g0_lb_coeff = csc_entry(tmpl, col_g_fpha0, row_lb).unwrap_or_else(|| {
        panic!("FPHA hydro 0 generation column missing entry in load balance row {row_lb}")
    });
    assert!(
        (g0_lb_coeff - 1.0).abs() < 1e-12,
        "FPHA hydro 0 load balance: expected +1.0, got {g0_lb_coeff}"
    );

    let col_g_fpha1 = 28_usize; // FPHA hydro h_idx=3 (local_idx=1)
    let g1_lb_coeff = csc_entry(tmpl, col_g_fpha1, row_lb).unwrap_or_else(|| {
        panic!("FPHA hydro 1 generation column missing entry in load balance row {row_lb}")
    });
    assert!(
        (g1_lb_coeff - 1.0).abs() < 1e-12,
        "FPHA hydro 1 load balance: expected +1.0, got {g1_lb_coeff}"
    );

    let col_turb_const = 13_usize; // constant hydro h_idx=0: col_turbine_start
    let turb_lb_coeff = csc_entry(tmpl, col_turb_const, row_lb);
    assert!(
        turb_lb_coeff.is_some(),
        "constant hydro 0 turbine col must appear in load balance row"
    );
    // Constant hydro enters the power-balance matrix with rho (the productivity
    // scalar), NOT rho * block_hours — block_hours scales cost objectives only.
    // Hydro 0 has productivity 2.5 MW/(m³/s), so the coefficient is 2.5.
    let expected_rho_coeff = 2.5_f64;
    assert!(
        (turb_lb_coeff.unwrap() - expected_rho_coeff).abs() < 1e-12,
        "constant hydro 0 turbine: expected rho = {expected_rho_coeff}, got {turb_lb_coeff:?}"
    );
}

#[test]
fn fpha_solve_one_hydro_optimal() {
    let (system, production) = fpha_solve_system();
    let result = build_stage_templates_resolving_layout(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &production,
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("FPHA template build must succeed");

    let template = &result.templates[0];
    let mut solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");
    solver.load_model(template);
    solver.add_rows(&empty_cut_batch());

    let v_in = 100.0_f64;
    solver.set_row_bounds(&[0], &[v_in], &[v_in]);

    let view = solver
        .solve(None)
        .expect("FPHA LP must be feasible and optimal");

    let col_g = 9_usize; // g column, shifted +1 by the diversion column
    let generation = view.primal[col_g];
    assert!(
        generation > 0.0,
        "FPHA generation must be strictly positive, got {generation}"
    );
}

/// After solving, every plane holds within 1e-6:
/// `g <= intercept + gamma_v * v_avg + gamma_q * q + gamma_s * s`,
/// `v_avg = (v + v_in) / 2`.
#[test]
fn fpha_solve_hyperplane_constraints_hold() {
    let (system, production) = fpha_solve_system();

    // Extract planes before moving production into build_stage_templates_resolving_layout.
    let planes = match production.model(0, 0) {
        ResolvedProductionModel::Fpha { planes, .. } => planes.clone(),
        ResolvedProductionModel::ConstantProductivity { .. } => {
            panic!("expected Fpha model")
        }
    };

    let result = build_stage_templates_resolving_layout(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &production,
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("FPHA template build must succeed");

    let template = &result.templates[0];
    let mut solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");
    solver.load_model(template);
    solver.add_rows(&empty_cut_batch());

    let v_in = 100.0_f64;
    solver.set_row_bounds(&[0], &[v_in], &[v_in]);

    let view = solver.solve(None).expect("FPHA LP must solve to optimal");
    let primal = view.primal;

    let col_v = 0_usize;
    let col_v_in = 1_usize;
    let col_q = 3_usize;
    let col_s = 4_usize;
    let col_g = 9_usize;

    let g = primal[col_g];
    let v = primal[col_v];
    let v_in_sol = primal[col_v_in];
    let q = primal[col_q];
    let s = primal[col_s];
    let v_avg = f64::midpoint(v, v_in_sol);

    for (p_idx, plane) in planes.iter().enumerate() {
        let rhs = plane.intercept + plane.gamma_v * v_avg + plane.gamma_q * q + plane.gamma_s * s;
        assert!(
            g <= rhs + 1e-6,
            "FPHA plane {p_idx}: g={g} must be <= rhs={rhs} \
             (intercept={intercept}, gamma_v={gamma_v}, v_avg={v_avg}, \
              gamma_q={gamma_q}, q={q}, gamma_s={gamma_s}, s={s})",
            intercept = plane.intercept,
            gamma_v = plane.gamma_v,
            gamma_q = plane.gamma_q,
            gamma_s = plane.gamma_s,
        );
    }
}

/// The reduced cost of the pinned `storage_in` column
/// differs between FPHA and constant productivity. The `-gamma_v/2` FPHA entries
/// on the `v_in` column propagate through the simplex dual to that reduced cost.
///
/// The tight planes (`intercept=0, gamma_v=0.5, gamma_q=1.0, gamma_s=0.0`) keep
/// the FPHA capacity below the 200 MW load so the constraint binds at the optimum
/// — necessary for a non-zero dual. With K=1, `zeta`≈2.678 and `v_in`=100 hm³:
/// - FPHA row: `g <= 0.25*(v + v_in) + q`
/// - Water balance: `v = v_in + inflow*zeta - (q+s)*zeta`
/// - `q_max = v_in/zeta + inflow ≈ 37.3 + 80 = 117.3 m³/s` (at `v`=0)
/// - `g_max ≈ 0.25*v_in + q_max ≈ 25 + 117.3 = 142.3 MW < 200 MW`
///
/// So the optimizer covers the shortfall with costly deficit and extra `v_in`
/// lowers cost → FPHA dual < 0. Constant productivity gives `rho`=0
/// (`default_from_system`), generation is `v_in`-independent → dual = 0.
#[test]
fn fpha_solve_incoming_storage_reduced_cost_differs_from_constant() {
    let (system, _) = one_fpha_hydro_system(1);

    let tight_planes = vec![FphaPlane {
        intercept: 0.0,
        gamma_v: 0.5,
        gamma_q: 1.0,
        gamma_s: 0.0,
    }];
    let fpha_production = ProductionModelSet::new(
        vec![vec![ResolvedProductionModel::Fpha {
            planes: tight_planes,
        }]],
        system.hydros(),
        1,
    );

    let fpha_result = build_stage_templates_resolving_layout(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &fpha_production,
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("FPHA template build must succeed");

    let const_production = default_production(&system);
    let const_result = build_stage_templates_resolving_layout(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &const_production,
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("constant productivity template build must succeed");

    let solve_and_get_storage_dual = |template: &cobre_solver::StageTemplate| -> f64 {
        let mut solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");
        solver.load_model(template);
        solver.add_rows(&empty_cut_batch());
        // Storage is pinned via column bounds: col 0 = storage_out, 1 = z_inflow, 2 = storage_in.
        let col_storage_in = 2_usize;
        let v_in = 100.0_f64;
        solver.set_col_bounds(&[col_storage_in], &[v_in], &[v_in]);
        let view = solver.solve(None).expect("LP must solve to optimal");
        view.reduced_costs[col_storage_in]
    };

    let fpha_dual = solve_and_get_storage_dual(&fpha_result.templates[0]);
    let const_dual = solve_and_get_storage_dual(&const_result.templates[0]);

    assert!(
        const_dual.abs() < 1e-6,
        "constant-productivity dual must be ~0, got {const_dual}"
    );

    assert!(
        fpha_dual.abs() > 1e-6,
        "FPHA incoming-storage reduced cost must be non-zero (FPHA v_in contribution \
         must be present), got {fpha_dual}"
    );

    assert_ne!(
        (fpha_dual * 1e6).round(),
        (const_dual * 1e6).round(),
        "incoming-storage reduced cost must differ between FPHA ({fpha_dual}) and \
         constant-productivity ({const_dual})"
    );
}

#[test]
fn fpha_solve_mixed_system_optimal() {
    let (system, production) = four_hydro_mixed_system();

    let result = build_stage_templates_resolving_layout(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &production,
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("mixed FPHA/constant system template build must succeed");

    let template = &result.templates[0];
    let mut solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");
    solver.load_model(template);
    solver.add_rows(&empty_cut_batch());

    solver.set_row_bounds(
        &[0, 1, 2, 3],
        &[100.0, 100.0, 100.0, 100.0],
        &[100.0, 100.0, 100.0, 100.0],
    );

    let view = solver
        .solve(None)
        .expect("mixed FPHA LP must be feasible and optimal");

    assert!(
        view.objective.is_finite(),
        "objective must be finite, got {}",
        view.objective
    );

    let col_g0 = 19_usize; // FPHA generation variables for hydros 2 and 3
    let col_g1 = 20_usize;
    assert!(
        view.primal[col_g0] >= 0.0,
        "FPHA hydro 0 generation must be non-negative, got {}",
        view.primal[col_g0]
    );
    assert!(
        view.primal[col_g1] >= 0.0,
        "FPHA hydro 1 generation must be non-negative, got {}",
        view.primal[col_g1]
    );
}
