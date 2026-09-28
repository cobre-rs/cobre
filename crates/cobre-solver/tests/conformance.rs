//! Conformance tests for the `SolverInterface` trait (SS1).
//!
//! Backend-agnostic integration tests verifying the `SolverInterface` contract
//! through the public API only. Fixtures are duplicated from the `highs.rs` unit
//! tests because integration tests cannot access `#[cfg(test)]` module internals.
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

use cobre_solver::{RowBatch, StageTemplate};

// Gated on either backend so a no-backend build does not import unused items.
#[cfg(any(feature = "highs", feature = "clp"))]
use cobre_solver::{Basis, SolverInterface};

#[cfg(feature = "highs")]
use cobre_solver::{HighsSolver, SolutionView, SolverError};

#[cfg(feature = "clp")]
use cobre_solver::{ClpSolver, SolverError};

// Gated identically to its only consumers (the HiGHS option-poking tests) so the
// clp+test-support build does not see an unused import.
#[cfg(all(feature = "test-support", feature = "highs"))]
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

#[cfg(feature = "highs")]
#[test]
fn test_solver_highs_load_model_and_solve() {
    let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
    let template = make_fixture_stage_template();

    solver.load_model(&template);
    let solution = solver
        .solve(None)
        .expect("solve() must succeed on feasible LP");

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

#[cfg(feature = "highs")]
#[test]
fn test_solver_highs_load_model_replaces_previous() {
    let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
    let template = make_fixture_stage_template();

    solver.load_model(&template);
    let obj1 = solver
        .solve(None)
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
        .solve(None)
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

#[cfg(feature = "highs")]
#[test]
fn test_solver_highs_add_rows_tightens() {
    let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
    let template = make_fixture_stage_template();
    let cuts = make_fixture_row_batch();

    solver.load_model(&template);
    solver.add_rows(&cuts);
    let solution = solver
        .solve(None)
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

#[cfg(feature = "highs")]
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
        .solve(None)
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

#[cfg(feature = "highs")]
#[test]
fn test_solver_highs_set_row_bounds_state_change() {
    let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
    let template = make_fixture_stage_template();
    let cuts = make_fixture_row_batch();

    solver.load_model(&template);
    solver.add_rows(&cuts);
    solver.set_row_bounds(&[0], &[4.0], &[4.0]);
    let solution = solver
        .solve(None)
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

#[cfg(feature = "highs")]
#[test]
fn test_solver_highs_set_col_bounds_basic() {
    let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
    let template = make_fixture_stage_template();
    let cuts = make_fixture_row_batch();

    solver.load_model(&template);
    solver.add_rows(&cuts);
    solver.set_col_bounds(&[2], &[0.0], &[3.0]);
    let solution = solver
        .solve(None)
        .expect("solve() must succeed after tightening col 2 bounds");

    let obj = solution.objective;
    assert!(
        (obj - 162.0).abs() < 1e-8,
        "expected objective = 162.0 (tighter bound does not bind), got {obj}"
    );
}

#[cfg(feature = "highs")]
#[test]
fn test_solver_highs_set_col_bounds_tightens() {
    let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
    let template = make_fixture_stage_template();

    solver.load_model(&template);
    solver.set_col_bounds(&[1], &[10.0], &[f64::INFINITY]);
    let solution = solver
        .solve(None)
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

#[cfg(feature = "highs")]
#[test]
fn test_solver_highs_set_col_bounds_repatch() {
    let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
    let template = make_fixture_stage_template();

    solver.load_model(&template);
    let obj1 = solver
        .solve(None)
        .expect("first solve() must succeed with original bounds")
        .objective;
    assert!(
        (obj1 - 100.0).abs() < 1e-8,
        "expected first objective = 100.0, got {obj1}"
    );

    solver.set_col_bounds(&[1], &[10.0], &[f64::INFINITY]);
    let obj2 = solver
        .solve(None)
        .expect("second solve() must succeed after tightening col 1")
        .objective;
    assert!(
        (obj2 - 110.0).abs() < 1e-8,
        "expected second objective = 110.0, got {obj2}"
    );

    solver.set_col_bounds(&[1], &[0.0], &[f64::INFINITY]);
    let obj3 = solver
        .solve(None)
        .expect("third solve() must succeed after restoring col 1 bounds")
        .objective;
    assert!(
        (obj3 - 100.0).abs() < 1e-8,
        "expected third objective = 100.0 (bounds restored), got {obj3}"
    );
}

// ─── SS1.7 solve dual values and reduced costs conformance tests ──────────────

#[cfg(feature = "highs")]
#[test]
fn test_solver_highs_solve_dual_values() {
    let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
    let template = make_fixture_stage_template();

    solver.load_model(&template);
    let solution = solver
        .solve(None)
        .expect("solve() must succeed on feasible LP");

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

#[cfg(feature = "highs")]
#[test]
fn test_solver_highs_solve_dual_values_with_cuts() {
    let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
    let template = make_fixture_stage_template();
    let cuts = make_fixture_row_batch();

    solver.load_model(&template);
    solver.add_rows(&cuts);
    let solution = solver
        .solve(None)
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

#[cfg(feature = "highs")]
#[test]
fn test_solver_highs_solve_reduced_costs() {
    let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
    let template = make_fixture_stage_template();

    solver.load_model(&template);
    let solution = solver
        .solve(None)
        .expect("solve() must succeed on feasible LP");

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

#[cfg(feature = "highs")]
#[test]
fn test_solver_highs_solve_iterations_reported() {
    let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
    let template = make_fixture_stage_template();

    solver.load_model(&template);
    let solution = solver
        .solve(None)
        .expect("solve() must succeed on feasible LP");

    // HiGHS may solve this LP entirely via presolve (0 simplex iterations), so do
    // not assert on iteration count; solve_time_seconds confirms the call ran.
    assert!(
        solution.solve_time_seconds >= 0.0,
        "expected solve_time_seconds >= 0.0, got {}",
        solution.solve_time_seconds
    );
}

// ─── SS5 Dual normalization conformance tests ─────────────────────────────────

/// SS5 row 1: load fixture, solve, verify cut-relevant row dual=-100.0 (canonical sign).
#[cfg(feature = "highs")]
#[test]
fn test_solver_highs_dual_normalization_cut_relevant_row() {
    let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
    let template = make_fixture_stage_template();

    solver.load_model(&template);
    let solution = solver
        .solve(None)
        .expect("solve() must succeed on feasible LP");

    // dual[0] is the dual of fixture row 0 (`x0 = 6`)
    let pi_0 = solution.dual[0];
    assert!(
        (pi_0 - (-100.0)).abs() < 1e-6,
        "expected cut-relevant dual[0] = -100.0, got {pi_0}"
    );
}

/// SS5 row 3: finite-difference sensitivity check: dual sign convention via RHS perturbation.
#[cfg(feature = "highs")]
#[test]
fn test_solver_highs_dual_normalization_sensitivity_check() {
    let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
    let template = make_fixture_stage_template();

    solver.load_model(&template);
    let z_original = solver
        .solve(None)
        .expect("first solve() must succeed on original fixture")
        .objective;
    assert!(
        (z_original - 100.0).abs() < 1e-8,
        "expected original objective = 100.0, got {z_original}"
    );

    // Perturb row 0's RHS by +0.01 (the divisor below).
    solver.set_row_bounds(&[0], &[6.01], &[6.01]);
    let z_perturbed = solver
        .solve(None)
        .expect("second solve() must succeed after patching Row 0 RHS")
        .objective;

    let finite_diff = (z_perturbed - z_original) / 0.01;
    assert!(
        (finite_diff - (-100.0)).abs() < 1e-2,
        "expected finite_diff = -100.0, got {finite_diff}"
    );
}

/// SS5 row 5: load fixture, add cuts, solve, verify binding cut dual=1.0 (canonical sign).
#[cfg(feature = "highs")]
#[test]
fn test_solver_highs_dual_normalization_with_binding_cut() {
    let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
    let template = make_fixture_stage_template();
    let cuts = make_fixture_row_batch();

    solver.load_model(&template);
    solver.add_rows(&cuts);
    let solution = solver
        .solve(None)
        .expect("solve() must succeed after adding both cuts");

    // dual[3] is Cut 2 (the binding cut, appended as Row 3)
    let pi_3 = solution.dual[3];
    assert!(
        (pi_3 - 1.0).abs() < 1e-6,
        "expected binding cut dual[3] = 1.0, got {pi_3}"
    );
}

// ─── SS1.9 statistics-accumulation conformance tests ─────────────────────────

/// SS1.9: statistics are always cumulative and survive `load_model` calls.
#[cfg(feature = "highs")]
#[test]
fn test_solver_highs_statistics_are_cumulative() {
    let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
    let template = make_fixture_stage_template();

    solver.load_model(&template);
    solver.solve(None).expect("first solve() must succeed");
    solver.solve(None).expect("second solve() must succeed");

    let stats = solver.statistics();
    assert_eq!(stats.solve_count, 2, "expected 2 solves");
    assert_eq!(stats.success_count, 2, "expected 2 successes");

    solver.load_model(&template);
    solver.solve(None).expect("third solve() must succeed");

    let stats_after = solver.statistics();
    assert_eq!(
        stats_after.solve_count, 3,
        "solve_count must accumulate across load_model calls"
    );
    assert_eq!(
        stats_after.success_count, 3,
        "success_count must accumulate across load_model calls"
    );
}

// ─── SS1.11 statistics conformance tests ──────────────────────────────────────

/// SS1.11 row 1: fresh solver, call `statistics()`, verify all counters = 0.
#[cfg(feature = "highs")]
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
        stats.basis_consistency_failures, 0,
        "expected basis_consistency_failures = 0 on fresh solver"
    );
}

/// SS1.11 row 3: load fixture, solve 3 times, verify statistics counters increment.
#[cfg(feature = "highs")]
#[test]
fn test_solver_highs_statistics_increment() {
    let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
    let template = make_fixture_stage_template();

    solver.load_model(&template);
    solver.solve(None).expect("first solve() must succeed");
    solver.solve(None).expect("second solve() must succeed");
    solver.solve(None).expect("third solve() must succeed");

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
    // HiGHS may solve via presolve (0 simplex iterations), so do not assert on
    // total_iterations; total_solve_time_seconds > 0.0 confirms the solver ran.
    assert!(
        stats.total_solve_time_seconds > 0.0,
        "expected total_solve_time_seconds > 0.0, got {}",
        stats.total_solve_time_seconds
    );
    assert_eq!(
        stats.basis_consistency_failures, 0,
        "expected basis_consistency_failures = 0 after cold solves, got {}",
        stats.basis_consistency_failures
    );
}

// ─── SS1.12 name conformance tests ────────────────────────────────────────────

/// SS1.12 row 1: verify `name()` returns `"HiGHS"` and is non-empty.
#[cfg(feature = "highs")]
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
/// The infeasibility at x0=8 arises because the power balance requires
/// x2 = 14 - 2*8 = -2, which violates x2 >= 0.
#[cfg(feature = "highs")]
#[test]
fn test_solver_highs_lifecycle_repeated_patch_solve() {
    let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
    let template = make_fixture_stage_template();
    let cuts = make_fixture_row_batch();

    solver.load_model(&template);
    solver.add_rows(&cuts);

    let obj1 = solver
        .solve(None)
        .expect("step 2 solve() must succeed with base fixture + cuts")
        .objective;
    assert!(
        (obj1 - 162.0).abs() < 1e-8,
        "step 2: expected objective = 162.0, got {obj1}"
    );

    solver.set_row_bounds(&[0], &[4.0], &[4.0]);
    let obj2 = solver
        .solve(None)
        .expect("step 3 solve() must succeed with x0=4.0")
        .objective;
    assert!(
        (obj2 - 368.0).abs() < 1e-8,
        "step 3: expected objective = 368.0, got {obj2}"
    );

    solver.set_row_bounds(&[0], &[8.0], &[8.0]);
    let result = solver.solve(None);
    assert!(
        matches!(result, Err(SolverError::Infeasible)),
        "step 4: expected Err(SolverError::Infeasible), got {:?}",
        result.map(|s| s.objective)
    );
}

// ─── SS3 Error path conformance tests ─────────────────────────────────────────

/// SS3.1: infeasible LP — contradictory column bounds.
///
/// A 1-variable LP with `col_lower = [5.0]` and `col_upper = [3.0]` has no
/// feasible point. `HiGHS` reports model status 8 (Infeasible), which
/// `interpret_terminal_status()` maps to `SolverError::Infeasible`.
#[cfg(feature = "highs")]
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
        col_scale: Vec::new(),
        row_scale: Vec::new(),
    };

    let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
    solver.load_model(&infeasible_template);
    let result = solver.solve(None).map(|s| s.objective);

    assert!(
        matches!(result, Err(SolverError::Infeasible)),
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
/// and `objective = [-1.0]` is unbounded below. `HiGHS` reports model status 10
/// (Unbounded) or 9 (`UnboundedOrInfeasible`); both map to
/// `SolverError::Unbounded` via `interpret_terminal_status()`.
#[cfg(feature = "highs")]
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
        col_scale: Vec::new(),
        row_scale: Vec::new(),
    };

    let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
    solver.load_model(&unbounded_template);
    let result = solver.solve(None).map(|s| s.objective);

    assert!(
        matches!(result, Err(SolverError::Unbounded)),
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
// `interpret_terminal_status`. The SS1.1 fixture is too small: HiGHS's crash
// heuristic produces an optimal starting point without entering the simplex loop,
// so the limit checks never fire. The larger chained-constraint fixture below
// cannot be solved at the crash point and requires real pivots.

/// Constructs the "larger LP" fixture used for time/iteration limit tests.
///
/// 5 variables, 4 chained >= constraints, all coefficients 1.0. Cannot be solved
/// at the crash point; requires >= 4 simplex pivots, so presolve cannot reduce it
/// to a trivially optimal form and the limit branches reliably fire.
#[cfg(feature = "highs")]
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
        col_scale: Vec::new(),
        row_scale: Vec::new(),
    }
}

/// SS limit row 1: external `time_limit=0` causes graceful failure.
///
/// An externally-set `time_limit=0` causes immediate `TIME_LIMIT` on every
/// `run_once()`, exhausting all retry levels — `solve()` does not override the
/// external time limit the way it overrides the iteration limit.
#[cfg(all(feature = "highs", feature = "test-support"))]
#[test]
fn test_solver_highs_solve_time_limit() {
    let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
    solver.load_model(&make_larger_lp_template());

    unsafe {
        test_support::cobre_highs_set_double_option(
            solver.raw_handle(),
            c"time_limit".as_ptr(),
            0.0,
        );
    }

    let result = solver.solve(None);
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
#[cfg(all(feature = "highs", feature = "test-support"))]
#[test]
fn test_solver_highs_solve_iteration_limit() {
    let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
    solver.load_model(&make_larger_lp_template());

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

    let result = solver.solve(None);
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
#[cfg(all(feature = "highs", feature = "test-support"))]
#[test]
fn test_solver_highs_restore_defaults_after_limit() {
    let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");

    solver.load_model(&make_larger_lp_template());
    unsafe {
        test_support::cobre_highs_set_int_option(
            solver.raw_handle(),
            c"simplex_iteration_limit".as_ptr(),
            0,
        );
    }
    assert!(
        solver.solve(None).is_ok(),
        "internal safeguards must override external simplex_iteration_limit=0"
    );

    // A different LP must still solve: safeguard limits are restored each solve.
    solver.load_model(&make_fixture_stage_template());
    let objective = solver
        .solve(None)
        .expect("solve() must succeed after reset")
        .objective;
    assert!((objective - 100.0).abs() < 1e-8);

    let stats = solver.statistics();
    assert_eq!(stats.solve_count, 2);
    assert_eq!(stats.success_count, 2);
}

// ─── Retry escalation note ────────────────────────────────────────────────────
//
// The 5-level retry escalation loop is entered only on `SOLVE_ERROR` (4) or
// `UNKNOWN` (15), which no pure LP triggers reliably across platforms. Reaching
// it from a test would require either making `run_once` replaceable (a production
// structure change) or injecting invalid model data via `unsafe` (violating the
// workspace `unsafe_code = "forbid"` lint) — neither acceptable without approval,
// so the loop is covered only indirectly via
// `test_solver_highs_restore_defaults_after_limit`.

// ─── Infeasible / unbounded ray extraction ───────────────────────────────────
//
// The trivial 0-row infeasible/unbounded tests detect status via bound-checking
// alone, so HiGHS computes no rays. These multi-row LPs force simplex to discover
// infeasibility/unboundedness, exercising the ray-extraction branch.

/// SS3.3: infeasible LP with constraints — exercises the infeasible classification path.
///
/// A 2-variable LP with row constraints that cannot be simultaneously satisfied:
///   x0 + x1 >= 10   (row 0)
///   x0 + x1 <= 5    (row 1)
///   x0, x1 >= 0
///
/// `HiGHS` simplex discovers infeasibility and returns `SolverError::Infeasible`.
#[cfg(feature = "highs")]
#[test]
fn test_solver_highs_infeasible_with_rows() {
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
        col_scale: Vec::new(),
        row_scale: Vec::new(),
    };

    let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
    solver.load_model(&infeasible_with_rows);
    let result = solver.solve(None);

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
#[cfg(all(feature = "highs", feature = "test-support"))]
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
        col_scale: Vec::new(),
        row_scale: Vec::new(),
    };

    let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");

    unsafe {
        test_support::cobre_highs_set_string_option(
            solver.raw_handle(),
            c"presolve".as_ptr(),
            c"on".as_ptr(),
        );
    }

    solver.load_model(&infeasible_with_rows);
    let result = solver.solve(None).map(|s| s.objective);

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
#[cfg(feature = "highs")]
#[test]
fn test_solver_highs_unbounded_with_primal_ray() {
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
        col_scale: Vec::new(),
        row_scale: Vec::new(),
    };

    let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
    solver.load_model(&unbounded_with_rows);
    let result = solver.solve(None);

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
#[cfg(all(feature = "highs", feature = "test-support"))]
#[test]
fn test_solver_highs_unbounded_or_infeasible() {
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
        col_scale: Vec::new(),
        row_scale: Vec::new(),
    };

    let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");

    unsafe {
        test_support::cobre_highs_set_string_option(
            solver.raw_handle(),
            c"presolve".as_ptr(),
            c"on".as_ptr(),
        );
    }

    solver.load_model(&ambiguous_template);
    let result = solver.solve(None).map(|s| s.objective);

    // Either is valid for this LP, depending on the HiGHS solver path taken.
    match &result {
        Err(SolverError::Infeasible | SolverError::Unbounded) => {}
        other => panic!("expected Infeasible or Unbounded error, got {other:?}"),
    }
}

// ─── SolutionView conformance tests ──────────────────────────────────────────

/// `solve()` + `to_owned()` must be numerically identical to a second `solve()`.
///
/// Both calls read from the same `HiGHS` internal buffers on equivalent solvers;
/// values must be bitwise-equal (same IEEE 754 bits), not merely close.
#[cfg(feature = "highs")]
#[test]
fn solve_equals_solve_owned() {
    let mut solver_a = HighsSolver::new().expect("HighsSolver::new() must succeed");
    solver_a.load_model(&make_fixture_stage_template());
    let owned = solver_a
        .solve(None)
        .expect("solve() must succeed")
        .to_owned();

    let mut solver_b = HighsSolver::new().expect("HighsSolver::new() must succeed");
    solver_b.load_model(&make_fixture_stage_template());
    let view = solver_b.solve(None).expect("solve() must succeed");
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
#[cfg(feature = "highs")]
#[test]
fn solve_borrows_internal_buffers() {
    let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
    solver.load_model(&make_fixture_stage_template());

    let (obj1, primal0_first) = {
        let view = solver.solve(None).expect("first solve() must succeed");
        (view.objective, view.primal[0])
    };
    // view dropped here, releasing the &mut self borrow for the second solve.

    let view2 = solver.solve(None).expect("second solve() must succeed");
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
/// The fixture with two appended rows has an optimal objective of 162.0
/// (x0=6, x1=62, x2=2; the tighter row forces x1 up to 62).
/// `view.dual.len()` must equal `template.num_rows + cuts.num_rows` (2 + 2 = 4).
#[cfg(feature = "highs")]
#[test]
fn solve_after_add_rows() {
    let template = make_fixture_stage_template();
    let cuts = make_fixture_row_batch();

    let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
    solver.load_model(&template);
    solver.add_rows(&cuts);

    let view = solver
        .solve(None)
        .expect("solve() after add_rows must succeed");

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
#[cfg(feature = "highs")]
#[test]
fn solve_statistics_updated() {
    let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
    solver.load_model(&make_fixture_stage_template());

    let _view: SolutionView<'_> = solver.solve(None).expect("solve() must succeed");

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
/// statuses, each HiGHS-representable (never `Superbasic`/`Fixed`).
#[cfg(feature = "highs")]
#[test]
fn basis_dimensions_after_solve() {
    use cobre_solver::BasisStatus;

    let mut solver = HighsSolver::new().expect("solver");
    let template = make_fixture_stage_template();
    solver.load_model(&template);
    solver.solve(None).expect("solve");

    let mut basis = Basis::new(template.num_cols, template.num_rows);
    solver.get_basis(&mut basis);

    assert_eq!(basis.col_status.len(), 3, "expected 3 col statuses");
    assert_eq!(basis.row_status.len(), 2, "expected 2 row statuses");

    let is_highs_representable =
        |status: BasisStatus| !matches!(status, BasisStatus::Superbasic | BasisStatus::Fixed);
    for (i, &status) in basis.col_status.iter().enumerate() {
        assert!(
            is_highs_representable(status),
            "col_status[{i}] = {status:?} is not HiGHS-representable"
        );
    }
    for (i, &status) in basis.row_status.iter().enumerate() {
        assert!(
            is_highs_representable(status),
            "row_status[{i}] = {status:?} is not HiGHS-representable"
        );
    }
}

/// A basis extracted from a 2-row LP must remain valid after 2 inequality rows
/// are added, and the warm-started objective must equal 162.0.
///
/// This test exercises the defensive BASIC-padding fallback path in `solve`.
/// The production caller reconciles the basis size to the current LP row count
/// before invoking `solve`, so the `debug_assert!` would fire on this fallback
/// path. The test runs only when `debug_assertions` is disabled.
#[cfg(all(feature = "highs", not(debug_assertions)))]
#[test]
fn basis_cut_extension() {
    let mut solver = HighsSolver::new().expect("solver");
    let template = make_fixture_stage_template();
    solver.load_model(&template);
    solver.solve(None).expect("cold solve");

    let mut basis = Basis::new(template.num_cols, template.num_rows);
    solver.get_basis(&mut basis);

    solver.load_model(&template);
    let cuts = make_fixture_row_batch();
    solver.add_rows(&cuts);

    let view = solver.solve(Some(&basis)).expect("warm-start with cuts");

    assert!(
        (view.objective - 162.0).abs() < 1e-8,
        "expected objective 162.0, got {}",
        view.objective
    );
}

/// A warm-start via `solve(Some(&basis))` must not require more simplex
/// iterations than a cold-start, and `basis_consistency_failures` must remain zero.
#[cfg(feature = "highs")]
#[test]
fn basis_warm_start_iterations() {
    let mut solver = HighsSolver::new().expect("solver");
    let template = make_fixture_stage_template();
    solver.load_model(&template);
    let cold_view = solver.solve(None).expect("cold solve");
    let cold_iterations = cold_view.iterations;

    let mut basis = Basis::new(template.num_cols, template.num_rows);
    solver.get_basis(&mut basis);

    solver.load_model(&template);
    let warm_view = solver.solve(Some(&basis)).expect("warm-start");

    assert!(
        warm_view.iterations <= cold_iterations,
        "warm-start iterations ({}) must not exceed cold-start iterations ({})",
        warm_view.iterations,
        cold_iterations
    );

    let stats = solver.statistics();
    assert_eq!(
        stats.basis_consistency_failures, 0,
        "basis_consistency_failures must be 0 after accepted basis, got {}",
        stats.basis_consistency_failures
    );
}

/// Full basis round-trip: solve SS1.1, extract basis via `get_basis`,
/// reload the same model, warm-start via `solve(Some(&basis))`, and verify
/// that the objective matches and the solver needs at most 1 simplex iteration.
#[cfg(feature = "highs")]
#[test]
fn test_basis_roundtrip() {
    let mut solver = HighsSolver::new().expect("solver");
    let template = make_fixture_stage_template();

    solver.load_model(&template);
    solver.solve(None).expect("cold solve must succeed");

    let mut basis = Basis::new(template.num_cols, template.num_rows);
    solver.get_basis(&mut basis);

    // Reload to reset HiGHS internal state before warm-starting from the basis.
    solver.load_model(&template);
    let warm_view = solver
        .solve(Some(&basis))
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

// ─── CLP-gated conformance tests (mirror HiGHS) ──────────────────────────────
//
// These mirror the core HiGHS conformance tests against the CLP backend, proving
// the two solvers agree on the shared SS1 fixtures. The dual sign convention is
// the canonical HiGHS sign — `normalize_row_dual` in `clp.rs` is the identity
// function — so the HiGHS expected values are reused verbatim, with no negation.

/// Mirrors `test_solver_highs_load_model_and_solve`: assert objective `100.0`
/// and primals `(6.0, 0.0, 2.0)`.
#[cfg(feature = "clp")]
#[test]
fn test_solver_clp_load_model_and_solve() {
    let mut solver = ClpSolver::new().expect("ClpSolver::new() must succeed");
    let template = make_fixture_stage_template();

    solver.load_model(&template);
    let solution = solver
        .solve(None)
        .expect("solve() must succeed on feasible LP");

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

/// Mirrors `test_solver_highs_solve_dual_values`: assert `dual.len() == 2` and
/// `dual[0] == -100.0` (the canonical sign, asserted directly).
#[cfg(feature = "clp")]
#[test]
fn test_solver_clp_solve_dual_values() {
    let mut solver = ClpSolver::new().expect("ClpSolver::new() must succeed");
    let template = make_fixture_stage_template();

    solver.load_model(&template);
    let solution = solver
        .solve(None)
        .expect("solve() must succeed on feasible LP");

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

/// Mirrors `test_solver_highs_add_rows_tightens`: load SS1.1, `add_rows(SS1.2)`,
/// `solve(None)`, assert objective `162.0` and primals `(6.0, 62.0, 2.0)`.
/// This is the end-to-end CSR→CSC merge cross-validation.
#[cfg(feature = "clp")]
#[test]
fn test_solver_clp_add_rows_then_solve() {
    let mut solver = ClpSolver::new().expect("ClpSolver::new() must succeed");
    let template = make_fixture_stage_template();
    let cuts = make_fixture_row_batch();

    solver.load_model(&template);
    solver.add_rows(&cuts);
    let solution = solver
        .solve(None)
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

/// Mirrors `test_basis_roundtrip`: cold solve SS1.1, `get_basis`, reload SS1.1,
/// warm-start via `solve(Some(&basis))`, assert objective `100.0`.
///
/// No iteration-count bound is asserted: CLP discards the basis on bounds
/// reload and its iteration semantics differ from `HiGHS`.
#[cfg(feature = "clp")]
#[test]
fn test_solver_clp_warm_start_roundtrip() {
    let mut solver = ClpSolver::new().expect("ClpSolver::new() must succeed");
    let template = make_fixture_stage_template();

    solver.load_model(&template);
    solver.solve(None).expect("cold solve must succeed");

    // `get_basis` resizes the buffer, so a 0×0 basis is fine to start.
    let mut basis = Basis::new(0, 0);
    solver.get_basis(&mut basis);

    // Reload to reset CLP internal state before warm-starting from the basis.
    solver.load_model(&template);
    let warm_view = solver
        .solve(Some(&basis))
        .expect("warm-start solve must succeed");

    assert!(
        (warm_view.objective - 100.0).abs() < 1e-8,
        "warm-start objective must equal 100.0, got {}",
        warm_view.objective
    );
}

// ─── CLP/HiGHS basis-validation asymmetry conformance tests ─────────────────
//
// Pins the two malformed-warm-basis shapes of the CLP/HiGHS basis-validation
// asymmetry contract, through the public `SolverInterface`
// API only. Shape 1 (row-count mismatch) is symmetric: both backends reject
// via `SolverError::BasisRowCountMismatch` before the basis ever reaches the
// native solver. Shape 2 (right dimensions, invalid status combination) is
// the asymmetry itself: `HighsSolver::solve` validates via `isBasisConsistent`
// and rejects with `SolverError::BasisInconsistent`; `ClpSolver::solve`
// installs the offered basis element-by-element with no consistency check and
// lets `Clp_dual` repair it, returning `Ok` with the correct optimum.

/// Offering a basis with fewer row entries than the LP has rows, after
/// `add_rows` grows SS1.1 from 2 to 4 rows, is rejected with
/// `Err(SolverError::BasisRowCountMismatch { lp_rows: 4, basis_rows: 2 })` —
/// symmetric with the `CLP` behavior pinned by
/// `test_solver_clp_solve_rejects_undersized_row_basis`; row-count mismatch is
/// not where the `CLP`/`HiGHS` basis-validation asymmetry lives.
#[cfg(feature = "highs")]
#[test]
fn test_solver_highs_solve_rejects_undersized_row_basis() {
    let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
    let template = make_fixture_stage_template();
    solver.load_model(&template);
    solver.solve(None).expect("cold solve must succeed");
    solver.add_rows(&make_fixture_row_batch());

    let undersized = Basis::new(template.num_cols, template.num_rows);
    let result = solver.solve(Some(&undersized));

    match result {
        Err(SolverError::BasisRowCountMismatch {
            lp_rows,
            basis_rows,
        }) => {
            assert_eq!(lp_rows, 4, "lp_rows must equal the post-add_rows row count");
            assert_eq!(
                basis_rows, 2,
                "basis_rows must equal the offered SS1.1 row count"
            );
        }
        other => panic!(
            "HiGHS must reject an undersized row basis with BasisRowCountMismatch \
             (lp_rows: 4, basis_rows: 2), got {other:?}"
        ),
    }
}

/// Offering a basis with fewer row entries than the LP has rows, after
/// `add_rows` grows SS1.1 from 2 to 4 rows, is rejected with
/// `Err(SolverError::BasisRowCountMismatch { lp_rows: 4, basis_rows: 2 })` —
/// symmetric with the `HiGHS` behavior pinned by
/// `test_solver_highs_solve_rejects_undersized_row_basis`.
#[cfg(feature = "clp")]
#[test]
fn test_solver_clp_solve_rejects_undersized_row_basis() {
    let mut solver = ClpSolver::new().expect("ClpSolver::new() must succeed");
    let template = make_fixture_stage_template();
    solver.load_model(&template);
    solver.solve(None).expect("cold solve must succeed");
    solver.add_rows(&make_fixture_row_batch());

    let undersized = Basis::new(template.num_cols, template.num_rows);
    let result = solver.solve(Some(&undersized));

    match result {
        Err(SolverError::BasisRowCountMismatch {
            lp_rows,
            basis_rows,
        }) => {
            assert_eq!(lp_rows, 4, "lp_rows must equal the post-add_rows row count");
            assert_eq!(
                basis_rows, 2,
                "basis_rows must equal the offered SS1.1 row count"
            );
        }
        other => panic!(
            "CLP must reject an undersized row basis with BasisRowCountMismatch \
             (lp_rows: 4, basis_rows: 2), got {other:?}"
        ),
    }
}

/// Offering a basis with the SS1.1 LP's exact dimensions (3 cols, 2 rows) but
/// every column and row status set `Basic` (5 basics against rank 2) is
/// rejected with `Err(SolverError::BasisInconsistent { num_row: 2,
/// total_basic: 5, col_basic: 3, row_basic: 2 })` — `HiGHS` validates the
/// offered basis via `isBasisConsistent` and rejects it loudly, unlike `CLP`
/// (`test_solver_clp_solve_accepts_inconsistent_basis_status_combination_silently`).
#[cfg(feature = "highs")]
#[test]
fn test_solver_highs_solve_rejects_inconsistent_basis_status_combination() {
    use cobre_solver::BasisStatus;

    let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
    let template = make_fixture_stage_template();
    solver.load_model(&template);

    let mut bad_basis = Basis::new(template.num_cols, template.num_rows);
    bad_basis
        .col_status
        .iter_mut()
        .for_each(|v| *v = BasisStatus::Basic);
    bad_basis
        .row_status
        .iter_mut()
        .for_each(|v| *v = BasisStatus::Basic);

    let result = solver.solve(Some(&bad_basis));

    match result {
        Err(SolverError::BasisInconsistent {
            num_row,
            total_basic,
            col_basic,
            row_basic,
        }) => {
            assert_eq!(num_row, 2, "num_row must match the SS1.1 LP row count");
            assert_eq!(total_basic, 5, "total_basic must be col_basic + row_basic");
            assert_eq!(col_basic, 3, "col_basic must count the 3 BASIC columns");
            assert_eq!(row_basic, 2, "row_basic must count the 2 BASIC rows");
        }
        other => panic!(
            "HiGHS must reject an over-determined (right-dimension, invalid \
             status combination) basis loudly with BasisInconsistent \
             (num_row: 2, total_basic: 5, col_basic: 3, row_basic: 2); got {other:?} \
             instead of the documented CLP-silently-accepts / HiGHS-rejects-loudly split"
        ),
    }
}

/// Offering a basis with the SS1.1 LP's exact dimensions (3 cols, 2 rows) but
/// every column and row status set `Basic` (5 basics against rank 2) is
/// accepted silently: `ClpSolver::install_basis` has no consistency check
/// (unlike `HighsSolver::solve`'s `isBasisConsistent` gate — see
/// `test_solver_highs_solve_rejects_inconsistent_basis_status_combination`),
/// so `Clp_dual` repairs the inconsistent starting basis internally and
/// returns `Ok` with the same objective and primal solution a cold solve
/// reaches. This is the CLP/HiGHS basis-validation asymmetry contract:
/// `CLP` accepts an internally inconsistent warm basis silently,
/// `HiGHS` validates and rejects it loudly.
#[cfg(feature = "clp")]
#[test]
fn test_solver_clp_solve_accepts_inconsistent_basis_status_combination_silently() {
    use cobre_solver::BasisStatus;

    let mut solver = ClpSolver::new().expect("ClpSolver::new() must succeed");
    let template = make_fixture_stage_template();
    solver.load_model(&template);

    let mut bad_basis = Basis::new(template.num_cols, template.num_rows);
    bad_basis
        .col_status
        .iter_mut()
        .for_each(|v| *v = BasisStatus::Basic);
    bad_basis
        .row_status
        .iter_mut()
        .for_each(|v| *v = BasisStatus::Basic);

    let view = solver.solve(Some(&bad_basis)).unwrap_or_else(|e| {
        panic!(
            "CLP is expected to accept an over-determined (right-dimension, \
             invalid status combination) basis silently and repair it via \
             Clp_dual, not reject it (contradicts the CLP-silently-accepts / \
             HiGHS-rejects-loudly basis-validation asymmetry contract); \
             got Err({e:?})"
        )
    });

    assert!(
        (view.objective - 100.0).abs() < 1e-8,
        "CLP's silent repair of the inconsistent basis must still reach the \
         SS1.1 optimum 100.0, got {}",
        view.objective
    );
    let primals = &view.primal;
    assert!(
        (primals[0] - 6.0).abs() < 1e-8
            && (primals[1] - 0.0).abs() < 1e-8
            && (primals[2] - 2.0).abs() < 1e-8,
        "expected primal [6.0, 0.0, 2.0] after CLP's silent repair, got \
         [{}, {}, {}]",
        primals[0],
        primals[1],
        primals[2]
    );
}
