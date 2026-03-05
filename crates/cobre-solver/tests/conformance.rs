//! Conformance tests for the `SolverInterface` trait (SS1).
//!
//! These are backend-agnostic integration tests that verify the `SolverInterface`
//! contract from the outside, using only the public API. The fixture functions
//! here duplicate the ones in `highs.rs` unit tests intentionally: integration
//! tests cannot access `#[cfg(test)]` module internals.
#![cfg_attr(
    test,
    allow(clippy::unwrap_used, clippy::expect_used, clippy::float_cmp)
)]

use cobre_solver::{HighsSolver, RowBatch, SolverInterface, StageTemplate};

// ─── SS1.1 Shared LP fixture ─────────────────────────────────────────────────
//
//   min  0*x0 + 1*x1 + 50*x2
//   s.t. x0            = 6   (state-fixing)
//        2*x0 + x2     = 14  (power balance)
//   x0 in [0, 10], x1 in [0, +inf), x2 in [0, 8]
//
// Optimal solution: x0=6, x1=0, x2=2, objective=100.0
//
// CSC matrix A = [[1, 0, 0], [2, 0, 1]]:
//   col_starts  = [0, 2, 2, 3]
//   row_indices = [0, 1, 1]
//   values      = [1.0, 2.0, 1.0]
fn make_fixture_stage_template() -> StageTemplate {
    StageTemplate {
        num_cols: 3,
        num_rows: 2,
        num_nz: 3,
        col_starts: vec![0, 2, 2, 3],
        row_indices: vec![0, 1, 1],
        values: vec![1.0, 2.0, 1.0],
        col_lower: vec![0.0, 0.0, 0.0],
        col_upper: vec![10.0, f64::INFINITY, 8.0],
        objective: vec![0.0, 1.0, 50.0],
        row_lower: vec![6.0, 14.0],
        row_upper: vec![6.0, 14.0],
        n_state: 1,
        n_transfer: 0,
        n_dual_relevant: 1,
        n_hydro: 1,
        max_par_order: 0,
    }
}

// ─── SS1.2 Benders cut fixture ────────────────────────────────────────────────
//
// Cut 1: -5*x0 + x1 >= 20  (col_indices [0,1], values [-5, 1])
// Cut 2:  3*x0 + x1 >= 80  (col_indices [0,1], values [ 3, 1])
fn make_fixture_row_batch() -> RowBatch {
    RowBatch {
        num_rows: 2,
        row_starts: vec![0, 2, 4],
        col_indices: vec![0, 1, 0, 1],
        values: vec![-5.0, 1.0, 3.0, 1.0],
        row_lower: vec![20.0, 80.0],
        row_upper: vec![f64::INFINITY, f64::INFINITY],
    }
}

// ─── SS1.4 load_model conformance tests ──────────────────────────────────────

/// SS1.4 row 1: load fixture, solve, verify objective and primals.
#[test]
fn test_solver_highs_load_model_and_solve() {
    let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
    let template = make_fixture_stage_template();

    solver.load_model(&template);
    let solution = solver.solve().expect("solve() must succeed on feasible LP");

    let obj = solution.objective;
    assert!(
        (obj - 100.0).abs() < 1e-8,
        "expected objective = 100.0, got {obj}"
    );

    let primals = &solution.primal;
    assert!(
        (primals[0] - 6.0).abs() < 1e-8,
        "expected x0 = 6.0, got {}",
        primals[0]
    );
    assert!(
        (primals[1] - 0.0).abs() < 1e-8,
        "expected x1 = 0.0, got {}",
        primals[1]
    );
    assert!(
        (primals[2] - 2.0).abs() < 1e-8,
        "expected x2 = 2.0, got {}",
        primals[2]
    );
}

/// SS1.4 row 3: load fixture, solve, reload with modified objective, solve again.
/// Verifies `load_model` fully replaces the previous model state.
#[test]
fn test_solver_highs_load_model_replaces_previous() {
    let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
    let template = make_fixture_stage_template();

    // First solve: objective = 100.0
    solver.load_model(&template);
    let solution1 = solver.solve().expect("first solve() must succeed");
    let obj1 = solution1.objective;
    assert!(
        (obj1 - 100.0).abs() < 1e-8,
        "expected first objective = 100.0, got {obj1}"
    );

    // Reload with modified objective coefficients
    let mut modified = make_fixture_stage_template();
    modified.objective = vec![0.0, 1.0, 25.0];
    solver.load_model(&modified);

    // Second solve: objective = 50.0 (same optimal point, lower cost per x2)
    let solution2 = solver.solve().expect("second solve() must succeed");
    let obj2 = solution2.objective;
    assert!(
        (obj2 - 50.0).abs() < 1e-8,
        "expected second objective = 50.0, got {obj2}"
    );
}

// ─── Fixture self-check (not a conformance test, validates fixture data) ──────

/// Verifies that `make_fixture_stage_template` produces exactly the SS1.1 data.
#[test]
fn test_fixture_stage_template_data() {
    let t = make_fixture_stage_template();
    assert_eq!(t.num_cols, 3);
    assert_eq!(t.num_rows, 2);
    assert_eq!(t.num_nz, 3);
    assert_eq!(t.col_starts, vec![0, 2, 2, 3]);
    assert_eq!(t.row_indices, vec![0, 1, 1]);
    assert_eq!(t.values, vec![1.0, 2.0, 1.0]);
    assert_eq!(t.col_lower, vec![0.0, 0.0, 0.0]);
    assert_eq!(t.col_upper[0], 10.0);
    assert!(t.col_upper[1].is_infinite() && t.col_upper[1].is_sign_positive());
    assert_eq!(t.col_upper[2], 8.0);
    assert_eq!(t.objective, vec![0.0, 1.0, 50.0]);
    assert_eq!(t.row_lower, vec![6.0, 14.0]);
    assert_eq!(t.row_upper, vec![6.0, 14.0]);
    assert_eq!(t.n_state, 1);
    assert_eq!(t.n_transfer, 0);
    assert_eq!(t.n_dual_relevant, 1);
    assert_eq!(t.n_hydro, 1);
    assert_eq!(t.max_par_order, 0);
}

/// Verifies that `make_fixture_row_batch` produces exactly the SS1.2 data.
#[test]
fn test_fixture_row_batch_data() {
    let b = make_fixture_row_batch();
    assert_eq!(b.num_rows, 2);
    assert_eq!(b.row_starts, vec![0, 2, 4]);
    assert_eq!(b.col_indices, vec![0, 1, 0, 1]);
    assert_eq!(b.values, vec![-5.0, 1.0, 3.0, 1.0]);
    assert_eq!(b.row_lower, vec![20.0, 80.0]);
    assert!(b.row_upper[0].is_infinite() && b.row_upper[0].is_sign_positive());
    assert!(b.row_upper[1].is_infinite() && b.row_upper[1].is_sign_positive());
}

// ─── SS1.5 add_rows conformance tests ────────────────────────────────────────

/// SS1.5 row 1: load fixture, add both cuts, solve. Optimal: x0=6, x1=62, x2=2, obj=162.0
#[test]
fn test_solver_highs_add_rows_tightens() {
    let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
    let template = make_fixture_stage_template();
    let cuts = make_fixture_row_batch();

    solver.load_model(&template);
    solver.add_rows(&cuts);
    let solution = solver
        .solve()
        .expect("solve() must succeed after adding both cuts");

    assert!(
        (solution.objective - 162.0).abs() < 1e-8,
        "expected objective = 162.0, got {}",
        solution.objective
    );
    let primals = &solution.primal;
    assert_eq!(primals.len(), 3);
    assert!(
        (primals[0] - 6.0).abs() < 1e-8
            && (primals[1] - 62.0).abs() < 1e-8
            && (primals[2] - 2.0).abs() < 1e-8,
        "expected [6.0, 62.0, 2.0], got [{}, {}, {}]",
        primals[0],
        primals[1],
        primals[2]
    );
}

/// SS1.5 row 3: load fixture, add single cut, solve. Optimal: x0=6, x1=50, x2=2, obj=150.0
#[test]
fn test_solver_highs_add_rows_single_cut() {
    let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
    let template = make_fixture_stage_template();

    // Construct a 1-row RowBatch containing only Cut 1: -5*x0 + x1 >= 20
    let single_cut = RowBatch {
        num_rows: 1,
        row_starts: vec![0, 2],
        col_indices: vec![0, 1],
        values: vec![-5.0, 1.0],
        row_lower: vec![20.0],
        row_upper: vec![f64::INFINITY],
    };

    solver.load_model(&template);
    solver.add_rows(&single_cut);
    let solution = solver
        .solve()
        .expect("solve() must succeed after adding single cut");

    let obj = solution.objective;
    assert!(
        (obj - 150.0).abs() < 1e-8,
        "expected objective = 150.0, got {obj}"
    );

    let primals = &solution.primal;
    assert!(
        (primals[1] - 50.0).abs() < 1e-8,
        "expected x1 = 50.0, got {}",
        primals[1]
    );
}

// ─── SS1.6 set_row_bounds conformance tests ───────────────────────────────────

/// SS1.6 row 1: load fixture, add cuts, patch Row 0 RHS to 4.0, solve. Optimal: obj=368.0
#[test]
fn test_solver_highs_set_row_bounds_state_change() {
    let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
    let template = make_fixture_stage_template();
    let cuts = make_fixture_row_batch();

    solver.load_model(&template);
    solver.add_rows(&cuts);
    // Patch Row 0 (state-fixing equality) from 6.0 to 4.0
    solver.set_row_bounds(&[(0, 4.0, 4.0)]);
    let solution = solver
        .solve()
        .expect("solve() must succeed after patching row bounds");

    let obj = solution.objective;
    assert!(
        (obj - 368.0).abs() < 1e-8,
        "expected objective = 368.0, got {obj}"
    );

    let primals = &solution.primal;
    assert!(
        (primals[0] - 4.0).abs() < 1e-8,
        "expected x0 = 4.0, got {}",
        primals[0]
    );
    assert!(
        (primals[1] - 68.0).abs() < 1e-8,
        "expected x1 = 68.0, got {}",
        primals[1]
    );
    assert!(
        (primals[2] - 6.0).abs() < 1e-8,
        "expected x2 = 6.0, got {}",
        primals[2]
    );
}

// ─── SS1.6a set_col_bounds conformance tests ──────────────────────────────────

/// SS1.6a row 1: load fixture, add cuts, patch col 2 bounds to [0, 3], solve. Obj=162.0
#[test]
fn test_solver_highs_set_col_bounds_basic() {
    let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
    let template = make_fixture_stage_template();
    let cuts = make_fixture_row_batch();

    solver.load_model(&template);
    solver.add_rows(&cuts);
    // Tighten col 2 upper bound from 8.0 to 3.0 (x2=2 still feasible)
    solver.set_col_bounds(&[(2, 0.0, 3.0)]);
    let solution = solver
        .solve()
        .expect("solve() must succeed after tightening col 2 bounds");

    let obj = solution.objective;
    assert!(
        (obj - 162.0).abs() < 1e-8,
        "expected objective = 162.0 (tighter bound does not bind), got {obj}"
    );
}

/// SS1.6a row 3: load fixture (no cuts), patch col 1 lower bound to 10.0, solve. Obj=110.0
#[test]
fn test_solver_highs_set_col_bounds_tightens() {
    let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
    let template = make_fixture_stage_template();

    solver.load_model(&template);
    // Force x1 >= 10.0
    solver.set_col_bounds(&[(1, 10.0, f64::INFINITY)]);
    let solution = solver
        .solve()
        .expect("solve() must succeed after patching col 1 lower bound");

    let obj = solution.objective;
    assert!(
        (obj - 110.0).abs() < 1e-8,
        "expected objective = 110.0, got {obj}"
    );

    let primals = &solution.primal;
    assert!(
        (primals[1] - 10.0).abs() < 1e-8,
        "expected x1 = 10.0, got {}",
        primals[1]
    );
}

/// SS1.6a row 5: load fixture, three solve cycles with col 1 bound changes. Verify restore.
#[test]
fn test_solver_highs_set_col_bounds_repatch() {
    let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
    let template = make_fixture_stage_template();

    // Cycle 1: original bounds
    solver.load_model(&template);
    let solution1 = solver
        .solve()
        .expect("first solve() must succeed with original bounds");
    let obj1 = solution1.objective;
    assert!(
        (obj1 - 100.0).abs() < 1e-8,
        "expected first objective = 100.0, got {obj1}"
    );

    // Cycle 2: patch col 1 lower bound to 10.0
    solver.set_col_bounds(&[(1, 10.0, f64::INFINITY)]);
    let solution2 = solver
        .solve()
        .expect("second solve() must succeed after tightening col 1");
    let obj2 = solution2.objective;
    assert!(
        (obj2 - 110.0).abs() < 1e-8,
        "expected second objective = 110.0, got {obj2}"
    );

    // Cycle 3: re-patch col 1 back to original [0, inf]
    solver.set_col_bounds(&[(1, 0.0, f64::INFINITY)]);
    let solution3 = solver
        .solve()
        .expect("third solve() must succeed after restoring col 1 bounds");
    let obj3 = solution3.objective;
    assert!(
        (obj3 - 100.0).abs() < 1e-8,
        "expected third objective = 100.0 (bounds restored), got {obj3}"
    );
}

// ─── SS1.7 solve dual values and reduced costs conformance tests ──────────────

/// SS1.7 row 1: load fixture, solve, verify dual values. Expected: `pi_0`=-100.0, `pi_1`=50.0
#[test]
fn test_solver_highs_solve_dual_values() {
    let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
    let template = make_fixture_stage_template();

    solver.load_model(&template);
    let solution = solver.solve().expect("solve() must succeed on feasible LP");

    assert_eq!(
        solution.dual.len(),
        2,
        "expected dual.len() = 2, got {}",
        solution.dual.len()
    );

    let pi_0 = solution.dual[0];
    assert!(
        (pi_0 - (-100.0)).abs() < 1e-6,
        "expected dual[0] = -100.0, got {pi_0}"
    );

    let pi_1 = solution.dual[1];
    assert!(
        (pi_1 - 50.0).abs() < 1e-6,
        "expected dual[1] = 50.0, got {pi_1}"
    );
}

/// SS1.7 row 3: load fixture, add cuts, solve, verify dual values with binding cut.
#[test]
fn test_solver_highs_solve_dual_values_with_cuts() {
    let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
    let template = make_fixture_stage_template();
    let cuts = make_fixture_row_batch();

    solver.load_model(&template);
    solver.add_rows(&cuts);
    let solution = solver
        .solve()
        .expect("solve() must succeed after adding both cuts");

    assert_eq!(
        solution.dual.len(),
        4,
        "expected dual.len() = 4, got {}",
        solution.dual.len()
    );

    let expected = [-103.0_f64, 50.0, 0.0, 1.0];
    for (i, &expected_pi) in expected.iter().enumerate() {
        let actual_pi = solution.dual[i];
        assert!(
            (actual_pi - expected_pi).abs() < 1e-6,
            "expected dual[{i}] = {expected_pi}, got {actual_pi}"
        );
    }
}

/// SS1.7 row 5: load fixture, solve, verify reduced costs. Expected: rc[1]=1.0
#[test]
fn test_solver_highs_solve_reduced_costs() {
    let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
    let template = make_fixture_stage_template();

    solver.load_model(&template);
    let solution = solver.solve().expect("solve() must succeed on feasible LP");

    assert_eq!(
        solution.reduced_costs.len(),
        3,
        "expected reduced_costs.len() = 3, got {}",
        solution.reduced_costs.len()
    );

    let rc_x1 = solution.reduced_costs[1];
    assert!(
        (rc_x1 - 1.0).abs() < 1e-6,
        "expected reduced_costs[1] = 1.0, got {rc_x1}"
    );
}

/// SS1.7 row 7: load fixture, solve, verify iterations >= 1 and solve time reported.
#[test]
fn test_solver_highs_solve_iterations_reported() {
    let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
    let template = make_fixture_stage_template();

    solver.load_model(&template);
    let solution = solver.solve().expect("solve() must succeed on feasible LP");

    assert!(
        solution.iterations >= 1,
        "expected iterations >= 1, got {}",
        solution.iterations
    );

    assert!(
        solution.solve_time_seconds >= 0.0,
        "expected solve_time_seconds >= 0.0, got {}",
        solution.solve_time_seconds
    );
}

// ─── SS5 Dual normalization conformance tests ─────────────────────────────────

/// SS5 row 1: load fixture, solve, verify cut-relevant row dual=-100.0 (canonical sign).
#[test]
fn test_solver_highs_dual_normalization_cut_relevant_row() {
    let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
    let template = make_fixture_stage_template();

    solver.load_model(&template);
    let solution = solver.solve().expect("solve() must succeed on feasible LP");

    // dual[0] is the cut-relevant state-fixing row dual (n_dual_relevant = 1)
    let pi_0 = solution.dual[0];
    assert!(
        (pi_0 - (-100.0)).abs() < 1e-6,
        "expected cut-relevant dual[0] = -100.0, got {pi_0}"
    );
}

/// SS5 row 3: finite-difference sensitivity check: dual sign convention via RHS perturbation.
#[test]
fn test_solver_highs_dual_normalization_sensitivity_check() {
    let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
    let template = make_fixture_stage_template();

    // Solve original problem (Row 0 RHS = 6.0, z* = 100.0)
    solver.load_model(&template);
    let solution_original = solver
        .solve()
        .expect("first solve() must succeed on original fixture");
    let z_original = solution_original.objective;
    assert!(
        (z_original - 100.0).abs() < 1e-8,
        "expected original objective = 100.0, got {z_original}"
    );

    // Patch Row 0 RHS to 6.01 (perturb state-fixing constraint by +0.01)
    solver.set_row_bounds(&[(0, 6.01, 6.01)]);
    let solution_perturbed = solver
        .solve()
        .expect("second solve() must succeed after patching Row 0 RHS");
    let z_perturbed = solution_perturbed.objective;

    // Finite-difference approximation of the sensitivity
    let finite_diff = (z_perturbed - z_original) / 0.01;
    assert!(
        (finite_diff - (-100.0)).abs() < 1e-2,
        "expected finite_diff = -100.0, got {finite_diff}"
    );
}

/// SS5 row 5: load fixture, add cuts, solve, verify binding cut dual=1.0 (canonical sign).
#[test]
fn test_solver_highs_dual_normalization_with_binding_cut() {
    let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
    let template = make_fixture_stage_template();
    let cuts = make_fixture_row_batch();

    solver.load_model(&template);
    solver.add_rows(&cuts);
    let solution = solver
        .solve()
        .expect("solve() must succeed after adding both cuts");

    // dual[3] is Cut 2 (the binding cut, appended as Row 3)
    let pi_3 = solution.dual[3];
    assert!(
        (pi_3 - 1.0).abs() < 1e-6,
        "expected binding cut dual[3] = 1.0, got {pi_3}"
    );
}

// ─── SS1.8 solve_with_basis conformance tests ─────────────────────────────────

/// SS1.8 row 1: cold solve, extract basis, reload, warm-start. Verify iterations <= cold.
#[test]
fn test_solver_highs_solve_with_basis_warm_start() {
    let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
    let template = make_fixture_stage_template();

    // Cold solve: record baseline iterations
    solver.load_model(&template);
    let cold_solution = solver
        .solve()
        .expect("cold solve() must succeed on feasible LP");
    let cold_iters = cold_solution.iterations;
    assert!(
        (cold_solution.objective - 100.0).abs() < 1e-8,
        "expected cold objective = 100.0, got {}",
        cold_solution.objective
    );

    // Extract basis from the cold solve
    let basis = solver.get_basis();

    // Reload identical fixture and warm-start
    solver.load_model(&template);
    let warm_solution = solver
        .solve_with_basis(&basis)
        .expect("solve_with_basis() must succeed on feasible LP");

    assert!(
        (warm_solution.objective - 100.0).abs() < 1e-8,
        "expected warm objective = 100.0, got {}",
        warm_solution.objective
    );
    assert!(
        warm_solution.iterations <= cold_iters,
        "warm-start iterations ({}) must be <= cold iterations ({})",
        warm_solution.iterations,
        cold_iters
    );

    // Basis was accepted: basis_rejections must remain at zero
    let stats = solver.statistics();
    assert_eq!(
        stats.basis_rejections, 0,
        "expected basis_rejections = 0 after successful warm start, got {}",
        stats.basis_rejections
    );
}

/// SS1.8 row 3: extract 2-row basis, reload, add cuts (4 rows), warm-start. Obj=162.0
#[test]
fn test_solver_highs_solve_with_basis_cut_extension() {
    let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
    let template = make_fixture_stage_template();
    let cuts = make_fixture_row_batch();

    // Solve to obtain a 2-row basis (structural rows only)
    solver.load_model(&template);
    solver
        .solve()
        .expect("initial solve() must succeed on feasible LP");
    let basis_2row = solver.get_basis();
    assert_eq!(
        basis_2row.col_status.len(),
        3,
        "expected 3 col statuses in 2-row basis"
    );
    assert_eq!(
        basis_2row.row_status.len(),
        2,
        "expected 2 row statuses in 2-row basis"
    );

    // Reload, add both cuts, warm-start with the 2-row basis
    solver.load_model(&template);
    solver.add_rows(&cuts);
    let solution = solver
        .solve_with_basis(&basis_2row)
        .expect("solve_with_basis() must succeed on 4-row LP with 2-row basis");

    assert!(
        (solution.objective - 162.0).abs() < 1e-8,
        "expected objective = 162.0 after cut extension warm-start, got {}",
        solution.objective
    );
}

// ─── SS1.9 reset conformance tests ────────────────────────────────────────────

/// SS1.9 row 3: solve twice, reset, verify statistics counters preserved.
#[test]
fn test_solver_highs_reset_preserves_statistics() {
    let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
    let template = make_fixture_stage_template();

    // Solve twice to accumulate non-trivial statistics
    solver.load_model(&template);
    solver.solve().expect("first solve() must succeed");
    solver.solve().expect("second solve() must succeed");

    let stats_before = solver.statistics();
    assert_eq!(
        stats_before.solve_count, 2,
        "expected 2 solves before reset"
    );
    assert_eq!(
        stats_before.success_count, 2,
        "expected 2 successes before reset"
    );

    // Reset clears the model but must preserve statistics
    solver.reset();

    let stats_after = solver.statistics();
    assert_eq!(
        stats_after.solve_count, stats_before.solve_count,
        "solve_count must be preserved across reset"
    );
    assert_eq!(
        stats_after.success_count, stats_before.success_count,
        "success_count must be preserved across reset"
    );
    assert_eq!(
        stats_after.failure_count, stats_before.failure_count,
        "failure_count must be preserved across reset"
    );
    assert_eq!(
        stats_after.total_iterations, stats_before.total_iterations,
        "total_iterations must be preserved across reset"
    );
    assert_eq!(
        stats_after.total_solve_time_seconds, stats_before.total_solve_time_seconds,
        "total_solve_time_seconds must be preserved across reset"
    );
}

// ─── SS1.10 get_basis conformance tests ───────────────────────────────────────

/// SS1.10 row 1: load fixture, solve, verify basis dimensions (3 cols, 2 rows).
#[test]
fn test_solver_highs_get_basis_dimensions() {
    let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
    let template = make_fixture_stage_template();

    solver.load_model(&template);
    solver.solve().expect("solve() must succeed on feasible LP");
    let basis = solver.get_basis();

    assert_eq!(
        basis.col_status.len(),
        3,
        "expected col_status.len() = 3 (num_cols), got {}",
        basis.col_status.len()
    );
    assert_eq!(
        basis.row_status.len(),
        2,
        "expected row_status.len() = 2 (num_rows), got {}",
        basis.row_status.len()
    );
}

/// SS1.10 row 3: basis roundtrip (extract → reset → reload → warm-start). Verify iterations <= 1.
#[test]
fn test_solver_highs_get_basis_roundtrip() {
    let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
    let template = make_fixture_stage_template();

    // Initial solve to extract a basis
    solver.load_model(&template);
    solver
        .solve()
        .expect("initial solve() must succeed on feasible LP");
    let basis = solver.get_basis();

    // Full reset then reload
    solver.reset();
    solver.load_model(&template);

    // Warm-start with the extracted basis
    let solution = solver
        .solve_with_basis(&basis)
        .expect("solve_with_basis() must succeed after roundtrip");

    assert!(
        (solution.objective - 100.0).abs() < 1e-8,
        "expected objective = 100.0 after roundtrip, got {}",
        solution.objective
    );
    assert!(
        solution.iterations <= 1,
        "expected iterations <= 1 after basis roundtrip, got {}",
        solution.iterations
    );
}

/// SS1.10 row 5: load fixture, add cuts, solve, verify basis dimensions (3 cols, 4 rows).
#[test]
fn test_solver_highs_get_basis_with_cuts() {
    let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
    let template = make_fixture_stage_template();
    let cuts = make_fixture_row_batch();

    solver.load_model(&template);
    solver.add_rows(&cuts);
    solver
        .solve()
        .expect("solve() must succeed after adding both cuts");
    let basis = solver.get_basis();

    assert_eq!(
        basis.col_status.len(),
        3,
        "expected col_status.len() = 3 (num_cols unchanged), got {}",
        basis.col_status.len()
    );
    assert_eq!(
        basis.row_status.len(),
        4,
        "expected row_status.len() = 4 (2 structural + 2 cuts), got {}",
        basis.row_status.len()
    );
}

// ─── SS1.11 statistics conformance tests ──────────────────────────────────────

/// SS1.11 row 1: fresh solver, call `statistics()`, verify all counters = 0.
#[test]
fn test_solver_highs_statistics_initial() {
    let solver = HighsSolver::new().expect("HighsSolver::new() must succeed");

    let stats = solver.statistics();
    assert_eq!(
        stats.solve_count, 0,
        "expected solve_count = 0 on fresh solver"
    );
    assert_eq!(
        stats.success_count, 0,
        "expected success_count = 0 on fresh solver"
    );
    assert_eq!(
        stats.failure_count, 0,
        "expected failure_count = 0 on fresh solver"
    );
    assert_eq!(
        stats.total_iterations, 0,
        "expected total_iterations = 0 on fresh solver"
    );
    assert_eq!(
        stats.retry_count, 0,
        "expected retry_count = 0 on fresh solver"
    );
    assert_eq!(
        stats.total_solve_time_seconds, 0.0,
        "expected total_solve_time_seconds = 0.0 on fresh solver"
    );
    assert_eq!(
        stats.basis_rejections, 0,
        "expected basis_rejections = 0 on fresh solver"
    );
}

/// SS1.11 row 3: load fixture, solve 3 times, verify statistics counters increment.
#[test]
fn test_solver_highs_statistics_increment() {
    let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
    let template = make_fixture_stage_template();

    solver.load_model(&template);
    solver.solve().expect("first solve() must succeed");
    solver.solve().expect("second solve() must succeed");
    solver.solve().expect("third solve() must succeed");

    let stats = solver.statistics();
    assert_eq!(
        stats.solve_count, 3,
        "expected solve_count = 3 after three solves, got {}",
        stats.solve_count
    );
    assert_eq!(
        stats.success_count, 3,
        "expected success_count = 3 after three successful solves, got {}",
        stats.success_count
    );
    assert_eq!(
        stats.failure_count, 0,
        "expected failure_count = 0, got {}",
        stats.failure_count
    );
    assert!(
        stats.total_iterations >= 1,
        "expected total_iterations >= 1, got {}",
        stats.total_iterations
    );
    assert!(
        stats.total_solve_time_seconds > 0.0,
        "expected total_solve_time_seconds > 0.0, got {}",
        stats.total_solve_time_seconds
    );
    assert_eq!(
        stats.basis_rejections, 0,
        "expected basis_rejections = 0 after cold solves, got {}",
        stats.basis_rejections
    );
}

// ─── SS1.12 name conformance tests ────────────────────────────────────────────

/// SS1.12 row 1: verify `name()` returns `"HiGHS"` and is non-empty.
#[test]
fn test_solver_highs_name_returns_identifier() {
    let solver = HighsSolver::new().expect("HighsSolver::new() must succeed");

    let name = solver.name();
    assert_eq!(name, "HiGHS", "expected name = \"HiGHS\", got \"{name}\"");
    assert!(!name.is_empty(), "name must be non-empty");
}

// ─── SS4 LP lifecycle conformance tests ───────────────────────────────────────

/// SS4 row 1: full 12-step lifecycle: new→load→solve→`add_rows`→solve→basis→patch→`solve_with_basis`→reset→reload→solve.
#[test]
fn test_solver_highs_lifecycle_full_cycle() {
    // Step 1: construct solver
    let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
    let template = make_fixture_stage_template();
    let cuts = make_fixture_row_batch();

    // Step 2: load structural LP
    solver.load_model(&template);

    // Step 3: cold solve → 100.0
    let sol3 = solver
        .solve()
        .expect("step 3 solve() must succeed on structural LP");
    assert!(
        (sol3.objective - 100.0).abs() < 1e-8,
        "step 3: expected objective = 100.0, got {}",
        sol3.objective
    );

    // Step 4: add both cuts
    solver.add_rows(&cuts);

    // Step 5: solve with cuts → 162.0
    let sol5 = solver
        .solve()
        .expect("step 5 solve() must succeed after adding both cuts");
    assert!(
        (sol5.objective - 162.0).abs() < 1e-8,
        "step 5: expected objective = 162.0, got {}",
        sol5.objective
    );

    // Step 6: extract basis, verify dimensions
    let basis = solver.get_basis();
    assert_eq!(
        basis.col_status.len(),
        3,
        "step 6: expected col_status.len() = 3, got {}",
        basis.col_status.len()
    );
    assert_eq!(
        basis.row_status.len(),
        4,
        "step 6: expected row_status.len() = 4, got {}",
        basis.row_status.len()
    );

    // Step 7: patch Row 0 state-fixing equality to 4.0
    solver.set_row_bounds(&[(0, 4.0, 4.0)]);

    // Step 8: warm-start with basis → 368.0
    let sol8 = solver
        .solve_with_basis(&basis)
        .expect("step 8 solve_with_basis() must succeed after row bound patch");
    assert!(
        (sol8.objective - 368.0).abs() < 1e-8,
        "step 8: expected objective = 368.0, got {}",
        sol8.objective
    );

    // Step 9: reset
    solver.reset();

    // Step 10: reload clean structural LP
    solver.load_model(&template);

    // Step 11: cold solve → 100.0 (back to baseline, cuts are gone)
    let sol11 = solver
        .solve()
        .expect("step 11 solve() must succeed after reset and reload");
    assert!(
        (sol11.objective - 100.0).abs() < 1e-8,
        "step 11: expected objective = 100.0 after reset+reload, got {}",
        sol11.objective
    );

    // Step 12: verify accumulated statistics after all 4 solves
    // (steps 3, 5, 8 via solve_with_basis→solve, and 11)
    let stats = solver.statistics();
    assert!(
        stats.solve_count >= 4,
        "step 12: expected solve_count >= 4, got {}",
        stats.solve_count
    );
    assert!(
        stats.success_count >= 4,
        "step 12: expected success_count >= 4, got {}",
        stats.success_count
    );
    assert_eq!(
        stats.failure_count, 0,
        "step 12: expected failure_count = 0, got {}",
        stats.failure_count
    );
    assert_eq!(
        stats.basis_rejections, 0,
        "step 12: expected basis_rejections = 0, got {}",
        stats.basis_rejections
    );
}

/// SS4 row 3: repeated RHS patching with infeasibility on the third patch.
///
/// Steps:
/// 1. Load SS1.1, add both cuts.
/// 2. Solve → objective = 162.0 (base + cuts, no patching).
/// 3. Patch Row 0 to 4.0, solve → objective = 368.0.
/// 4. Patch Row 0 to 8.0, solve → expect `Err(SolverError::Infeasible { .. })`.
///
/// The infeasibility at x0=8 arises because the power balance requires
/// x2 = 14 - 2*8 = -2, which violates x2 >= 0.
#[test]
fn test_solver_highs_lifecycle_repeated_patch_solve() {
    let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
    let template = make_fixture_stage_template();
    let cuts = make_fixture_row_batch();

    // Step 1: load and add cuts
    solver.load_model(&template);
    solver.add_rows(&cuts);

    // Step 2: solve without patching → 162.0
    let sol1 = solver
        .solve()
        .expect("step 2 solve() must succeed with base fixture + cuts");
    assert!(
        (sol1.objective - 162.0).abs() < 1e-8,
        "step 2: expected objective = 162.0, got {}",
        sol1.objective
    );

    // Step 3: patch Row 0 to 4.0, solve → 368.0
    solver.set_row_bounds(&[(0, 4.0, 4.0)]);
    let sol2 = solver
        .solve()
        .expect("step 3 solve() must succeed with x0=4.0");
    assert!(
        (sol2.objective - 368.0).abs() < 1e-8,
        "step 3: expected objective = 368.0, got {}",
        sol2.objective
    );

    // Step 4: patch Row 0 to 8.0 — this makes x2 = 14 - 16 = -2 < 0, infeasible
    solver.set_row_bounds(&[(0, 8.0, 8.0)]);
    let result = solver.solve();
    assert!(
        matches!(result, Err(cobre_solver::SolverError::Infeasible { .. })),
        "step 4: expected Err(SolverError::Infeasible {{ .. }}), got {:?}",
        result.map(|s| s.objective)
    );
}

// ─── SS3 Error path conformance tests ─────────────────────────────────────────

/// SS3.1: infeasible LP — contradictory column bounds.
///
/// A 1-variable LP with `col_lower = [5.0]` and `col_upper = [3.0]` has no
/// feasible point. `HiGHS` must report model status 8 (Infeasible), which
/// `interpret_terminal_status()` maps to `SolverError::Infeasible`.
///
/// Statistics after the failed solve:
/// - `solve_count` = 1
/// - `failure_count` = 1
/// - `success_count` = 0
#[test]
fn test_solver_highs_solve_infeasible() {
    let infeasible_template = StageTemplate {
        num_cols: 1,
        num_rows: 0,
        num_nz: 0,
        col_starts: vec![0, 0],
        row_indices: vec![],
        values: vec![],
        col_lower: vec![5.0],
        col_upper: vec![3.0],
        objective: vec![1.0],
        row_lower: vec![],
        row_upper: vec![],
        n_state: 1,
        n_transfer: 0,
        n_dual_relevant: 0,
        n_hydro: 0,
        max_par_order: 0,
    };

    let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
    solver.load_model(&infeasible_template);
    let result = solver.solve();

    assert!(
        matches!(result, Err(cobre_solver::SolverError::Infeasible { .. })),
        "expected Err(SolverError::Infeasible {{ .. }}), got {:?}",
        result.map(|s| s.objective)
    );

    let stats = solver.statistics();
    assert_eq!(
        stats.solve_count, 1,
        "expected solve_count = 1 after infeasible solve, got {}",
        stats.solve_count
    );
    assert_eq!(
        stats.failure_count, 1,
        "expected failure_count = 1 after infeasible solve, got {}",
        stats.failure_count
    );
    assert_eq!(
        stats.success_count, 0,
        "expected success_count = 0 after infeasible solve, got {}",
        stats.success_count
    );
}

/// SS3.2: unbounded LP — minimise a variable with no lower bound and negative
/// objective coefficient.
///
/// A 1-variable LP with `col_lower = [NEG_INFINITY]`, `col_upper = [INFINITY]`,
/// and `objective = [-1.0]` is unbounded below (driving the variable to
/// -infinity minimises the objective). `HiGHS` reports model status 10
/// (Unbounded) or 9 (`UnboundedOrInfeasible`); both map to
/// `SolverError::Unbounded` via `interpret_terminal_status()`.
///
/// Statistics after the failed solve:
/// - `solve_count` = 1
/// - `failure_count` = 1
/// - `success_count` = 0
#[test]
fn test_solver_highs_solve_unbounded() {
    let unbounded_template = StageTemplate {
        num_cols: 1,
        num_rows: 0,
        num_nz: 0,
        col_starts: vec![0, 0],
        row_indices: vec![],
        values: vec![],
        col_lower: vec![f64::NEG_INFINITY],
        col_upper: vec![f64::INFINITY],
        objective: vec![-1.0],
        row_lower: vec![],
        row_upper: vec![],
        n_state: 1,
        n_transfer: 0,
        n_dual_relevant: 0,
        n_hydro: 0,
        max_par_order: 0,
    };

    let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
    solver.load_model(&unbounded_template);
    let result = solver.solve();

    assert!(
        matches!(result, Err(cobre_solver::SolverError::Unbounded { .. })),
        "expected Err(SolverError::Unbounded {{ .. }}), got {:?}",
        result.map(|s| s.objective)
    );

    let stats = solver.statistics();
    assert_eq!(
        stats.solve_count, 1,
        "expected solve_count = 1 after unbounded solve, got {}",
        stats.solve_count
    );
    assert_eq!(
        stats.failure_count, 1,
        "expected failure_count = 1 after unbounded solve, got {}",
        stats.failure_count
    );
    assert_eq!(
        stats.success_count, 0,
        "expected success_count = 0 after unbounded solve, got {}",
        stats.success_count
    );
}
