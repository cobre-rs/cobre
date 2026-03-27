//! Conformance tests for the `SolverInterface` trait (SS1).
//!
//! These are backend-agnostic integration tests that verify the `SolverInterface`
//! contract from the outside, using only the public API. The fixture functions
//! here duplicate the ones in `highs.rs` unit tests intentionally: integration
//! tests cannot access `#[cfg(test)]` module internals.
#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::float_cmp,
        clippy::too_many_lines,
        clippy::panic
    )
)]

use cobre_solver::{
    Basis, HighsSolver, RowBatch, SolutionView, SolverError, SolverInterface, StageTemplate,
};

#[cfg(feature = "test-support")]
use cobre_solver::test_support;

fn make_fixture_stage_template() -> StageTemplate {
    StageTemplate {
        num_cols: 3,
        num_rows: 2,
        num_nz: 3,
        col_starts: vec![0_i32, 2, 2, 3],
        row_indices: vec![0_i32, 1, 1],
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
        col_scale: Vec::new(),
        row_scale: Vec::new(),
    }
}

fn make_fixture_row_batch() -> RowBatch {
    RowBatch {
        num_rows: 2,
        row_starts: vec![0_i32, 2, 4],
        col_indices: vec![0_i32, 1, 0, 1],
        values: vec![-5.0, 1.0, 3.0, 1.0],
        row_lower: vec![20.0, 80.0],
        row_upper: vec![f64::INFINITY, f64::INFINITY],
    }
}

// ─── SS1.4 load_model conformance tests ──────────────────────────────────────

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

// SS1.4 row 3: load_model replaces previous model completely
#[test]
fn test_solver_highs_load_model_replaces_previous() {
    let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
    let template = make_fixture_stage_template();

    solver.load_model(&template);
    let obj1 = solver
        .solve()
        .expect("first solve() must succeed")
        .objective;
    assert!(
        (obj1 - 100.0).abs() < 1e-8,
        "expected first objective = 100.0, got {obj1}"
    );

    let mut modified = make_fixture_stage_template();
    modified.objective = vec![0.0, 1.0, 25.0];
    solver.load_model(&modified);

    let obj2 = solver
        .solve()
        .expect("second solve() must succeed")
        .objective;
    assert!(
        (obj2 - 50.0).abs() < 1e-8,
        "expected second objective = 50.0, got {obj2}"
    );
}

// ─── Fixture self-check (not a conformance test, validates fixture data) ──────

#[test]
fn test_fixture_stage_template_data() {
    let t = make_fixture_stage_template();
    assert_eq!(t.num_cols, 3);
    assert_eq!(t.num_rows, 2);
    assert_eq!(t.num_nz, 3);
    assert_eq!(t.col_starts, vec![0_i32, 2, 2, 3]);
    assert_eq!(t.row_indices, vec![0_i32, 1, 1]);
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

#[test]
fn test_fixture_row_batch_data() {
    let b = make_fixture_row_batch();
    assert_eq!(b.num_rows, 2);
    assert_eq!(b.row_starts, vec![0_i32, 2, 4]);
    assert_eq!(b.col_indices, vec![0_i32, 1, 0, 1]);
    assert_eq!(b.values, vec![-5.0, 1.0, 3.0, 1.0]);
    assert_eq!(b.row_lower, vec![20.0, 80.0]);
    assert!(b.row_upper[0].is_infinite() && b.row_upper[0].is_sign_positive());
    assert!(b.row_upper[1].is_infinite() && b.row_upper[1].is_sign_positive());
}

// ─── SS1.5 add_rows conformance tests ────────────────────────────────────────

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

// SS1.5 row 3: add_rows with single cut
#[test]
fn test_solver_highs_add_rows_single_cut() {
    let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
    let template = make_fixture_stage_template();

    let single_cut = RowBatch {
        num_rows: 1,
        row_starts: vec![0_i32, 2],
        col_indices: vec![0_i32, 1],
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

#[test]
fn test_solver_highs_set_row_bounds_state_change() {
    let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
    let template = make_fixture_stage_template();
    let cuts = make_fixture_row_batch();

    solver.load_model(&template);
    solver.add_rows(&cuts);
    solver.set_row_bounds(&[0], &[4.0], &[4.0]);
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

#[test]
fn test_solver_highs_set_col_bounds_basic() {
    let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
    let template = make_fixture_stage_template();
    let cuts = make_fixture_row_batch();

    solver.load_model(&template);
    solver.add_rows(&cuts);
    solver.set_col_bounds(&[2], &[0.0], &[3.0]);
    let solution = solver
        .solve()
        .expect("solve() must succeed after tightening col 2 bounds");

    let obj = solution.objective;
    assert!(
        (obj - 162.0).abs() < 1e-8,
        "expected objective = 162.0 (tighter bound does not bind), got {obj}"
    );
}

// SS1.6a row 3: set_col_bounds tightens variable
#[test]
fn test_solver_highs_set_col_bounds_tightens() {
    let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
    let template = make_fixture_stage_template();

    solver.load_model(&template);
    // Force x1 >= 10.0
    solver.set_col_bounds(&[1], &[10.0], &[f64::INFINITY]);
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

// SS1.6a row 5: set_col_bounds: patch, re-patch, verify restore
#[test]
fn test_solver_highs_set_col_bounds_repatch() {
    let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
    let template = make_fixture_stage_template();

    solver.load_model(&template);
    let obj1 = solver
        .solve()
        .expect("first solve() must succeed with original bounds")
        .objective;
    assert!(
        (obj1 - 100.0).abs() < 1e-8,
        "expected first objective = 100.0, got {obj1}"
    );

    solver.set_col_bounds(&[1], &[10.0], &[f64::INFINITY]);
    let obj2 = solver
        .solve()
        .expect("second solve() must succeed after tightening col 1")
        .objective;
    assert!(
        (obj2 - 110.0).abs() < 1e-8,
        "expected second objective = 110.0, got {obj2}"
    );

    solver.set_col_bounds(&[1], &[0.0], &[f64::INFINITY]);
    let obj3 = solver
        .solve()
        .expect("third solve() must succeed after restoring col 1 bounds")
        .objective;
    assert!(
        (obj3 - 100.0).abs() < 1e-8,
        "expected third objective = 100.0 (bounds restored), got {obj3}"
    );
}

// ─── SS1.7 solve dual values and reduced costs conformance tests ──────────────

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

// SS1.7 row 3: solve() returns correct dual values with binding cuts
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

// SS1.7 row 5: solve() returns correct reduced costs
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

// SS1.7 row 7: solve() reports iteration count and solve time
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
    let z_original = solver
        .solve()
        .expect("first solve() must succeed on original fixture")
        .objective;
    assert!(
        (z_original - 100.0).abs() < 1e-8,
        "expected original objective = 100.0, got {z_original}"
    );

    // Patch Row 0 RHS to 6.01 (perturb state-fixing constraint by +0.01)
    solver.set_row_bounds(&[0], &[6.01], &[6.01]);
    let z_perturbed = solver
        .solve()
        .expect("second solve() must succeed after patching Row 0 RHS")
        .objective;

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

    solver.load_model(&template);
    solver.add_rows(&cuts);

    let obj1 = solver
        .solve()
        .expect("step 2 solve() must succeed with base fixture + cuts")
        .objective;
    assert!(
        (obj1 - 162.0).abs() < 1e-8,
        "step 2: expected objective = 162.0, got {obj1}"
    );

    solver.set_row_bounds(&[0], &[4.0], &[4.0]);
    let obj2 = solver
        .solve()
        .expect("step 3 solve() must succeed with x0=4.0")
        .objective;
    assert!(
        (obj2 - 368.0).abs() < 1e-8,
        "step 3: expected objective = 368.0, got {obj2}"
    );

    solver.set_row_bounds(&[0], &[8.0], &[8.0]);
    let result = solver.solve();
    assert!(
        matches!(result, Err(cobre_solver::SolverError::Infeasible)),
        "step 4: expected Err(SolverError::Infeasible), got {:?}",
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
        col_starts: vec![0_i32, 0],
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
        col_scale: Vec::new(),
        row_scale: Vec::new(),
    };

    let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
    solver.load_model(&infeasible_template);
    let result = solver.solve().map(|s| s.objective);

    assert!(
        matches!(result, Err(cobre_solver::SolverError::Infeasible)),
        "expected Err(SolverError::Infeasible), got {result:?}"
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
        col_starts: vec![0_i32, 0],
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
        col_scale: Vec::new(),
        row_scale: Vec::new(),
    };

    let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
    solver.load_model(&unbounded_template);
    let result = solver.solve().map(|s| s.objective);

    assert!(
        matches!(result, Err(cobre_solver::SolverError::Unbounded)),
        "expected Err(SolverError::Unbounded), got {result:?}"
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

// ─── Edge case: time and iteration limits ─────────────────────────────────────
//
// These tests exercise the time-limit and iteration-limit branches in
// `interpret_terminal_status`, which are never reached by tests that only
// use perfectly solvable LPs.
//
// The SS1.1 fixture (3 vars, 2 constraints) is too small to trigger these limits:
// HiGHS's crash heuristic produces an optimal starting point without entering the
// simplex loop, so per-iteration and per-second checks never fire. A larger LP
// is required (see research doc: plans/phase-3-solver/epic-08-coverage/research-edge-case-lps.md).
//
// ─── Larger LP fixture (5 vars, 4 constraints) ───────────────────────────────
//
//   Minimize:   x0 + x1 + x2 + x3 + x4
//   Subject to:
//     x0 + x1           >= 10  (row 0)
//          x1 + x2      >= 8   (row 1)
//               x2 + x3 >= 6   (row 2)
//                    x3 + x4 >= 4  (row 3)
//   0 <= xi <= 100, i = 0..4
//
// This LP cannot be solved at the crash point (all xi=0); it requires at least
// 4 simplex pivots. The chained structure means presolve cannot reduce it to
// a trivially optimal form in a single pass. This makes it reliable for
// triggering both TIME_LIMIT (time_limit=0.0) and ITERATION_LIMIT
// (presolve="off", simplex_iteration_limit=0).
//
// CSC format (column-wise sparse):
//   col_starts  = [0, 1, 3, 5, 7, 8]       len = num_cols + 1 = 6
//   row_indices = [0, 0, 1, 1, 2, 2, 3, 3] len = num_nz = 8
//   values      = [1.0; 8]                 len = num_nz = 8
//
// StageTemplate fields n_state, n_transfer, n_dual_relevant, n_hydro, max_par_order
// are set to 0/1 as needed to satisfy the struct; they do not affect LP solving.

/// Constructs the "larger LP" fixture used for time/iteration limit tests.
///
/// 5 variables, 4 chained >= constraints, all coefficients 1.0.
/// Cannot be solved at the crash point; requires >= 4 simplex pivots.
#[allow(dead_code)]
fn make_larger_lp_template() -> StageTemplate {
    StageTemplate {
        num_cols: 5,
        num_rows: 4,
        num_nz: 8,
        col_starts: vec![0_i32, 1, 3, 5, 7, 8],
        row_indices: vec![0_i32, 0, 1, 1, 2, 2, 3, 3],
        values: vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        col_lower: vec![0.0, 0.0, 0.0, 0.0, 0.0],
        col_upper: vec![100.0, 100.0, 100.0, 100.0, 100.0],
        objective: vec![1.0, 1.0, 1.0, 1.0, 1.0],
        row_lower: vec![10.0, 8.0, 6.0, 4.0],
        row_upper: vec![f64::INFINITY, f64::INFINITY, f64::INFINITY, f64::INFINITY],
        n_state: 1,
        n_transfer: 0,
        n_dual_relevant: 1,
        n_hydro: 0,
        max_par_order: 0,
        col_scale: Vec::new(),
        row_scale: Vec::new(),
    }
}

/// SS limit row 1: external `time_limit=0` causes graceful failure.
///
/// `HiGHS` tracks elapsed time cumulatively from instance creation —
/// `time_limit` is not used by the safeguard system (iteration limits
/// and wall-clock checks are used instead). An externally-set
/// `time_limit=0` causes immediate `TIME_LIMIT` on every `run_once()`,
/// exhausting all retry levels and returning `NumericalDifficulty`.
#[cfg(feature = "test-support")]
#[test]
fn test_solver_highs_solve_time_limit() {
    let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
    solver.load_model(&make_larger_lp_template());

    // External time_limit=0 persists through all retry levels.
    unsafe {
        test_support::cobre_highs_set_double_option(
            solver.raw_handle(),
            c"time_limit".as_ptr(),
            0.0,
        );
    }

    let result = solver.solve();
    assert!(result.is_err(), "time_limit=0 must exhaust all retries");

    let stats = solver.statistics();
    assert_eq!(stats.solve_count, 1);
    assert_eq!(stats.failure_count, 1);
    assert!(
        stats.retry_count > 0,
        "retry escalation must have been attempted"
    );
}

/// SS limit row 2: internal safeguard iteration limits override externally-set limits.
///
/// `solve()` applies its own `simplex_iteration_limit` (derived from LP dimensions)
/// before `run_once()`, overriding any externally-set `simplex_iteration_limit`.
/// This ensures the LP solves successfully even if an external caller sets
/// `simplex_iteration_limit=0`.
#[cfg(feature = "test-support")]
#[test]
fn test_solver_highs_solve_iteration_limit() {
    let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
    solver.load_model(&make_larger_lp_template());

    // Set an impossibly tight external limit — internal safeguards must override it.
    unsafe {
        test_support::cobre_highs_set_string_option(
            solver.raw_handle(),
            c"presolve".as_ptr(),
            c"off".as_ptr(),
        );
        test_support::cobre_highs_set_int_option(
            solver.raw_handle(),
            c"simplex_iteration_limit".as_ptr(),
            0,
        );
    }

    // Solve succeeds because internal safeguards override the external limit.
    let result = solver.solve();
    assert!(
        result.is_ok(),
        "internal safeguard limits must override external simplex_iteration_limit=0"
    );

    let stats = solver.statistics();
    assert_eq!(stats.solve_count, 1);
    assert_eq!(stats.success_count, 1);
}

/// SS limit row 3: internal safeguards ensure consistent solve across reset cycles.
///
/// Verifies that `solve()` applies and restores safeguard limits correctly
/// across multiple `load_model`/`solve`/`reset` cycles. External limit overrides
/// do not persist because `solve()` sets its own limits before each attempt
/// and restores them afterward.
#[cfg(feature = "test-support")]
#[test]
fn test_solver_highs_restore_defaults_after_limit() {
    let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");

    // First solve succeeds despite external iter_limit=0 (internal safeguards override).
    solver.load_model(&make_larger_lp_template());
    unsafe {
        test_support::cobre_highs_set_int_option(
            solver.raw_handle(),
            c"simplex_iteration_limit".as_ptr(),
            0,
        );
    }
    assert!(
        solver.solve().is_ok(),
        "internal safeguards must override external simplex_iteration_limit=0"
    );

    // Reset and reload a different LP — solve must still work because
    // safeguard limits are restored after each solve.
    solver.reset();
    solver.load_model(&make_fixture_stage_template());
    let objective = solver
        .solve()
        .expect("solve() must succeed after reset")
        .objective;
    assert!((objective - 100.0).abs() < 1e-8);

    let stats = solver.statistics();
    assert_eq!(stats.solve_count, 2);
    assert_eq!(stats.success_count, 2);
}

// ─── Retry escalation note ────────────────────────────────────────────────────
//
// The 5-level retry escalation loop in `highs.rs` lines 920-1006 is entered
// only when `run_once()` returns `HIGHS_MODEL_STATUS_SOLVE_ERROR` (4) or
// `HIGHS_MODEL_STATUS_UNKNOWN` (15). These statuses are NOT triggered by any
// pure LP formulation reliably across platforms:
//
// - SOLVE_ERROR requires `cobre_highs_run` to return `HIGHS_STATUS_ERROR` (-1),
//   which only happens for semi-continuous/semi-integer variables (not supported
//   by cobre-solver's LP-only interface) or LPs with HiGHS-internal-infinity
//   cost coefficients (not representable as IEEE 754 doubles).
// - UNKNOWN requires IPM crossover failure on a degenerate LP, which is
//   platform- and version-dependent.
//
// A mock-based test would require either:
//   a) Making `run_once` replaceable (trait object or function pointer), which
//      changes the production code structure; or
//   b) Injecting invalid model data through `unsafe` code, which violates the
//      workspace `unsafe_code = "forbid"` lint.
//
// Neither approach is acceptable without explicit user approval. The retry loop
// and `restore_default_settings` are therefore covered indirectly:
// `restore_default_settings` is exercised by `test_solver_highs_restore_defaults_after_limit`
// (which verifies that cleanup after a limit error allows a subsequent optimal solve),
// and the retry loop's coverage is deferred to a future ticket that adds a
// controllable error injection mechanism.
//
// Reference: research-edge-case-lps.md §3.3 "Retry Escalation" and §8 item 3.

// ─── Infeasible / unbounded ray extraction ───────────────────────────────────
//
// The existing infeasible and unbounded tests use trivial 0-row LPs where HiGHS
// detects the status through bound-checking alone (no simplex). For those LPs,
// HiGHS does not compute dual/primal rays. These tests use multi-row LPs that
// force simplex to discover infeasibility/unboundedness, producing rays.

/// SS3.3: infeasible LP with constraints — exercises the infeasible classification path.
///
/// A 2-variable LP with row constraints that cannot be simultaneously satisfied:
///   x0 + x1 >= 10   (row 0)
///   x0 + x1 <= 5    (row 1)
///   x0, x1 >= 0
///
/// `HiGHS` simplex discovers infeasibility and returns `SolverError::Infeasible`.
#[test]
fn test_solver_highs_infeasible_with_rows() {
    // CSC: 2 cols, 2 rows, 4 non-zeros
    // col 0: rows [0, 1] -> a_start = [0, 2, 4]
    // col 1: rows [0, 1]
    let infeasible_with_rows = StageTemplate {
        num_cols: 2,
        num_rows: 2,
        num_nz: 4,
        col_starts: vec![0_i32, 2, 4],
        row_indices: vec![0_i32, 1, 0, 1],
        values: vec![1.0, 1.0, 1.0, 1.0],
        col_lower: vec![0.0, 0.0],
        col_upper: vec![f64::INFINITY, f64::INFINITY],
        objective: vec![1.0, 1.0],
        row_lower: vec![10.0, f64::NEG_INFINITY],
        row_upper: vec![f64::INFINITY, 5.0],
        n_state: 1,
        n_transfer: 0,
        n_dual_relevant: 2,
        n_hydro: 0,
        max_par_order: 0,
        col_scale: Vec::new(),
        row_scale: Vec::new(),
    };

    let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
    solver.load_model(&infeasible_with_rows);
    let result = solver.solve();

    assert!(
        matches!(result, Err(SolverError::Infeasible)),
        "expected Err(SolverError::Infeasible), got {:?}",
        result.map(|_| ())
    );
}

/// SS3.3b: infeasible LP with presolve — exercises dual ray extraction with presolve on.
///
/// Some `HiGHS` solver paths only provide dual rays when presolve is enabled.
/// This test re-runs the infeasible LP with presolve=on to maximise the
/// chance of exercising the `Some(ray_buf)` branch.
#[cfg(feature = "test-support")]
#[test]
fn test_solver_highs_infeasible_with_presolve() {
    let infeasible_with_rows = StageTemplate {
        num_cols: 2,
        num_rows: 2,
        num_nz: 4,
        col_starts: vec![0_i32, 2, 4],
        row_indices: vec![0_i32, 1, 0, 1],
        values: vec![1.0, 1.0, 1.0, 1.0],
        col_lower: vec![0.0, 0.0],
        col_upper: vec![f64::INFINITY, f64::INFINITY],
        objective: vec![1.0, 1.0],
        row_lower: vec![10.0, f64::NEG_INFINITY],
        row_upper: vec![f64::INFINITY, 5.0],
        n_state: 1,
        n_transfer: 0,
        n_dual_relevant: 2,
        n_hydro: 0,
        max_par_order: 0,
        col_scale: Vec::new(),
        row_scale: Vec::new(),
    };

    let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");

    // Enable presolve — may help HiGHS compute infeasibility certificates.
    unsafe {
        test_support::cobre_highs_set_string_option(
            solver.raw_handle(),
            c"presolve".as_ptr(),
            c"on".as_ptr(),
        );
    }

    solver.load_model(&infeasible_with_rows);
    let result = solver.solve().map(|s| s.objective);

    assert!(
        matches!(result, Err(SolverError::Infeasible)),
        "expected Err(SolverError::Infeasible), got {result:?}"
    );
}

/// SS3.4: unbounded LP with primal ray — free variable driving objective to -∞.
///
/// A 2-variable LP where x1 is unconstrained and drives the objective:
///   min -x1
///   s.t. x0 <= 10    (row 0, only constrains x0)
///   x0 >= 0, x1 free
///
/// `HiGHS` simplex discovers unboundedness and returns `SolverError::Unbounded`.
#[test]
fn test_solver_highs_unbounded_with_primal_ray() {
    // CSC: 2 cols, 1 row, 1 non-zero
    // col 0: row [0] -> a_start = [0, 1, 1]
    // col 1: (empty)
    let unbounded_with_rows = StageTemplate {
        num_cols: 2,
        num_rows: 1,
        num_nz: 1,
        col_starts: vec![0_i32, 1, 1],
        row_indices: vec![0_i32],
        values: vec![1.0],
        col_lower: vec![0.0, f64::NEG_INFINITY],
        col_upper: vec![f64::INFINITY, f64::INFINITY],
        objective: vec![0.0, -1.0],
        row_lower: vec![f64::NEG_INFINITY],
        row_upper: vec![10.0],
        n_state: 1,
        n_transfer: 0,
        n_dual_relevant: 1,
        n_hydro: 0,
        max_par_order: 0,
        col_scale: Vec::new(),
        row_scale: Vec::new(),
    };

    let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
    solver.load_model(&unbounded_with_rows);
    let result = solver.solve();

    assert!(
        matches!(result, Err(SolverError::Unbounded)),
        "expected Err(SolverError::Unbounded), got {:?}",
        result.map(|_| ())
    );
}

/// SS3.5: unbounded-or-infeasible LP — presolve detects ambiguous status.
///
/// A 2-variable LP with contradictory constraints AND an unbounded free variable:
///   min -x1
///   s.t. x0 >= 10     (row 0)
///        x0 <= 5      (row 1, contradicts row 0)
///   x0 >= 0, x1 free (unbounded)
///
/// With presolve ON, `HiGHS` detects the contradiction during preprocessing and
/// may report model status 9 (`UNBOUNDED_OR_INFEASIBLE`) because the presence
/// of the free variable x1 makes the dual also infeasible. The
/// `interpret_terminal_status()` path for status 9 attempts ray extraction.
///
/// If `HiGHS` reports status 8 (INFEASIBLE) instead, the test still succeeds —
/// the ray extraction code for INFEASIBLE is exercised by the previous test.
#[cfg(feature = "test-support")]
#[test]
fn test_solver_highs_unbounded_or_infeasible() {
    // CSC: 2 cols, 2 rows, 2 non-zeros
    // col 0: rows [0, 1] -> a_start = [0, 2, 2]
    // col 1: (empty — free variable not in constraints)
    let ambiguous_template = StageTemplate {
        num_cols: 2,
        num_rows: 2,
        num_nz: 2,
        col_starts: vec![0_i32, 2, 2],
        row_indices: vec![0_i32, 1],
        values: vec![1.0, 1.0],
        col_lower: vec![0.0, f64::NEG_INFINITY],
        col_upper: vec![f64::INFINITY, f64::INFINITY],
        objective: vec![0.0, -1.0],
        row_lower: vec![10.0, f64::NEG_INFINITY],
        row_upper: vec![f64::INFINITY, 5.0],
        n_state: 1,
        n_transfer: 0,
        n_dual_relevant: 2,
        n_hydro: 0,
        max_par_order: 0,
        col_scale: Vec::new(),
        row_scale: Vec::new(),
    };

    let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");

    // Enable presolve to trigger UNBOUNDED_OR_INFEASIBLE detection.
    unsafe {
        test_support::cobre_highs_set_string_option(
            solver.raw_handle(),
            c"presolve".as_ptr(),
            c"on".as_ptr(),
        );
    }

    solver.load_model(&ambiguous_template);
    let result = solver.solve().map(|s| s.objective);

    // Accept either UNBOUNDED_OR_INFEASIBLE or INFEASIBLE — both are valid
    // responses for this LP depending on the HiGHS solver path taken.
    match &result {
        Err(SolverError::Infeasible | SolverError::Unbounded) => {
            // Both are acceptable outcomes.
        }
        other => panic!("expected Infeasible or Unbounded error, got {other:?}"),
    }
}

// ─── SolutionView conformance tests ──────────────────────────────────────────

/// `solve()` + `to_owned()` must be numerically identical to a second `solve()`.
///
/// Both calls read from the same `HiGHS` internal buffers on equivalent solvers;
/// values must be bitwise-equal (same IEEE 754 bits), not merely close.
#[test]
fn solve_equals_solve_owned() {
    // Solver A: view path converted to owned
    let mut solver_a = HighsSolver::new().expect("HighsSolver::new() must succeed");
    solver_a.load_model(&make_fixture_stage_template());
    let owned = solver_a.solve().expect("solve() must succeed").to_owned();

    // Solver B: zero-copy view path
    let mut solver_b = HighsSolver::new().expect("HighsSolver::new() must succeed");
    solver_b.load_model(&make_fixture_stage_template());
    let view = solver_b.solve().expect("solve() must succeed");
    let from_view = view.to_owned();

    assert_eq!(
        owned.objective, from_view.objective,
        "objectives must be bitwise equal"
    );
    assert_eq!(
        owned.primal, from_view.primal,
        "primals must be bitwise equal"
    );
    assert_eq!(owned.dual, from_view.dual, "duals must be bitwise equal");
    assert_eq!(
        owned.reduced_costs, from_view.reduced_costs,
        "reduced_costs must be bitwise equal"
    );
    assert_eq!(
        owned.iterations, from_view.iterations,
        "iterations must match"
    );
}

/// Calling `solve()` twice on the same loaded model (borrow-drop cycle) succeeds.
///
/// Verifies that: (a) the first view is correctly dropped at end of the scope,
/// (b) the second `solve()` call acquires the `&mut self` borrow without conflict,
/// and (c) both results are identical.
#[test]
fn solve_borrows_internal_buffers() {
    let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
    solver.load_model(&make_fixture_stage_template());

    // First solve — read a couple of values then let it go out of scope.
    let (obj1, primal0_first) = {
        let view = solver.solve().expect("first solve() must succeed");
        (view.objective, view.primal[0])
    };
    // view is dropped here; the &mut self borrow is released.

    // Second solve — model is unchanged, so results must be identical.
    let view2 = solver.solve().expect("second solve() must succeed");
    assert_eq!(
        obj1, view2.objective,
        "objective must be identical on both calls"
    );
    assert_eq!(
        primal0_first, view2.primal[0],
        "primal[0] must be identical on both calls"
    );
}

/// After `add_rows`, `solve()` must reflect the extended LP.
///
/// The fixture with two Benders cuts has an optimal objective of 162.0
/// (x0=6, x1=62, x2=2; the tighter cut forces x1 up to 62).
/// `view.dual.len()` must equal `template.num_rows + cuts.num_rows` (2 + 2 = 4).
#[test]
fn solve_after_add_rows() {
    let template = make_fixture_stage_template();
    let cuts = make_fixture_row_batch();

    let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
    solver.load_model(&template);
    solver.add_rows(&cuts);

    let view = solver.solve().expect("solve() after add_rows must succeed");

    assert!(
        (view.objective - 162.0).abs() < 1e-8,
        "objective must be 162.0 after adding Benders cuts, got {}",
        view.objective
    );
    assert_eq!(
        view.dual.len(),
        template.num_rows + cuts.num_rows,
        "dual length must equal template.num_rows ({}) + cuts.num_rows ({}) = {}",
        template.num_rows,
        cuts.num_rows,
        template.num_rows + cuts.num_rows,
    );
}

/// After `solve()`, `statistics().solve_count` and `success_count` must each be 1.
#[test]
fn solve_statistics_updated() {
    let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
    solver.load_model(&make_fixture_stage_template());

    let _view: SolutionView<'_> = solver.solve().expect("solve() must succeed");

    let stats = solver.statistics();
    assert_eq!(
        stats.solve_count, 1,
        "solve_count must be 1 after one solve call"
    );
    assert_eq!(
        stats.success_count, 1,
        "success_count must be 1 after a successful solve"
    );
}

// --- Basis conformance tests ---

/// `get_basis` must write exactly `num_cols` col statuses and `num_rows` row
/// statuses, each in the valid `HiGHS` range [0, 4].
#[test]
fn basis_dimensions_after_solve() {
    let mut solver = HighsSolver::new().expect("solver");
    let template = make_fixture_stage_template();
    solver.load_model(&template);
    solver.solve().expect("solve");

    let mut basis = Basis::new(template.num_cols, template.num_rows);
    solver.get_basis(&mut basis);

    assert_eq!(basis.col_status.len(), 3, "expected 3 col statuses");
    assert_eq!(basis.row_status.len(), 2, "expected 2 row statuses");

    for (i, &code) in basis.col_status.iter().enumerate() {
        assert!(
            (0..=4).contains(&code),
            "col_status[{i}] = {code} is not a valid HiGHS basis status (0..=4)"
        );
    }
    for (i, &code) in basis.row_status.iter().enumerate() {
        assert!(
            (0..=4).contains(&code),
            "row_status[{i}] = {code} is not a valid HiGHS basis status (0..=4)"
        );
    }
}

/// A basis extracted from a 2-row LP must remain valid after 2 Benders cuts are
/// added, and the warm-started objective must equal 162.0.
#[test]
fn basis_cut_extension() {
    let mut solver = HighsSolver::new().expect("solver");
    let template = make_fixture_stage_template();
    solver.load_model(&template);
    solver.solve().expect("cold solve");

    let mut basis = Basis::new(template.num_cols, template.num_rows);
    solver.get_basis(&mut basis);

    // Reload and add cuts (2 structural + 2 cuts = 4 rows)
    solver.load_model(&template);
    let cuts = make_fixture_row_batch();
    solver.add_rows(&cuts);

    let view = solver
        .solve_with_basis(&basis)
        .expect("warm-start with cuts");

    assert!(
        (view.objective - 162.0).abs() < 1e-8,
        "expected objective 162.0, got {}",
        view.objective
    );
}

/// A warm-start via `solve_with_basis` must not require more simplex
/// iterations than a cold-start, and `basis_rejections` must remain zero.
#[test]
fn basis_warm_start_iterations() {
    let mut solver = HighsSolver::new().expect("solver");
    let template = make_fixture_stage_template();
    solver.load_model(&template);
    let cold_view = solver.solve().expect("cold solve");
    let cold_iterations = cold_view.iterations;
    // cold_view is dropped here; the &mut self borrow on solver is released.

    let mut basis = Basis::new(template.num_cols, template.num_rows);
    solver.get_basis(&mut basis);

    solver.load_model(&template);
    let warm_view = solver.solve_with_basis(&basis).expect("warm-start");

    assert!(
        warm_view.iterations <= cold_iterations,
        "warm-start iterations ({}) must not exceed cold-start iterations ({})",
        warm_view.iterations,
        cold_iterations
    );
    // warm_view is dropped here; the &mut self borrow on solver is released.

    let stats = solver.statistics();
    assert_eq!(
        stats.basis_rejections, 0,
        "basis_rejections must be 0 after accepted basis, got {}",
        stats.basis_rejections
    );
}

/// Full basis round-trip: solve SS1.1, extract basis via `get_basis`,
/// reload the same model, warm-start via `solve_with_basis`, and verify
/// that the objective matches and the solver needs at most 1 simplex iteration.
#[test]
fn test_basis_roundtrip() {
    let mut solver = HighsSolver::new().expect("solver");
    let template = make_fixture_stage_template();

    // Cold-start solve to obtain the optimal basis.
    solver.load_model(&template);
    solver.solve().expect("cold solve must succeed");

    // Extract the basis into a pre-allocated buffer.
    let mut basis = Basis::new(template.num_cols, template.num_rows);
    solver.get_basis(&mut basis);

    // Reload the model to reset HiGHS internal state, then warm-start.
    solver.load_model(&template);
    let warm_view = solver
        .solve_with_basis(&basis)
        .expect("warm-start solve must succeed");

    assert!(
        (warm_view.objective - 100.0).abs() < 1e-8,
        "warm-start objective must equal 100.0, got {}",
        warm_view.objective
    );
    assert!(
        warm_view.iterations <= 1,
        "warm-start from exact basis must complete in at most 1 iteration, got {}",
        warm_view.iterations
    );
}
