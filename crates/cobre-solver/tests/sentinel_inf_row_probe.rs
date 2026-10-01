//! Probe: HiGHS row-bound sentinel semantics for cut deactivation.
//!
//! Validates that setting a row's bounds to `[-f64::INFINITY, +f64::INFINITY]`
//! makes the constraint trivially satisfied, and that this can be reversed
//! without corrupting LP state.

#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        clippy::doc_markdown,
        clippy::too_many_lines
    )
)]
#![cfg(feature = "highs")]

use cobre_solver::{HighsSolver, RowBatch, SolverError, SolverInterface, StageTemplate};

/// 2-column, 2-row LP:
/// - Col 0 (theta): free, objective 1.0; Col 1 (x): [0, 10], objective 0.0
/// - Row 0: x == 5; Row 1 (cut_row): theta - 2*x >= -3
fn build_two_row_template() -> StageTemplate {
    StageTemplate {
        num_cols: 2,
        num_rows: 2,
        num_nz: 3,
        col_starts: vec![0_i32, 1, 3],
        row_indices: vec![1_i32, 0, 1],
        values: vec![1.0_f64, 1.0, -2.0],
        col_lower: vec![-f64::INFINITY, 0.0],
        col_upper: vec![f64::INFINITY, 10.0],
        objective: vec![1.0, 0.0],
        row_lower: vec![5.0, -3.0],
        row_upper: vec![5.0, f64::INFINITY],
        n_state: 0,
        col_scale: Vec::new(),
        row_scale: Vec::new(),
    }
}

/// 2-column, 3-row LP:
/// - Col 0 (theta): free, objective 1.0; Col 1 (x): [0, 10], objective 0.0
/// - Row 0: x == 5; Row 1 (cut_row_A): theta - 2*x >= -3;
///   Row 2 (cut_row_B): theta >= -1000 (dominated by A)
fn build_three_row_template() -> StageTemplate {
    StageTemplate {
        num_cols: 2,
        num_rows: 3,
        num_nz: 4,
        col_starts: vec![0_i32, 2, 4],
        row_indices: vec![1_i32, 2, 0, 1],
        values: vec![1.0_f64, 1.0, 1.0, -2.0],
        col_lower: vec![-f64::INFINITY, 0.0],
        col_upper: vec![f64::INFINITY, 10.0],
        objective: vec![1.0, 0.0],
        row_lower: vec![5.0, -3.0, -1000.0],
        row_upper: vec![5.0, f64::INFINITY, f64::INFINITY],
        n_state: 0,
        col_scale: Vec::new(),
        row_scale: Vec::new(),
    }
}

fn make_empty_row_batch() -> RowBatch {
    RowBatch {
        num_rows: 0,
        row_starts: vec![0_i32],
        col_indices: Vec::new(),
        values: Vec::new(),
        row_lower: Vec::new(),
        row_upper: Vec::new(),
    }
}

/// When the only cut row is deactivated via `[-INF, +INF]` bounds, theta is unbounded
/// below (no lower bound on theta anywhere), so HiGHS must report `SolverError::Unbounded`.
#[test]
fn inactive_cut_row_with_inf_bounds_preserves_lp_value() {
    let template = build_two_row_template();
    let empty_rows = make_empty_row_batch();

    let mut solver = HighsSolver::new().expect("HiGHS init");
    solver.load_model(&template);
    solver.add_rows(&empty_rows);

    // First solve: cut_row is active (lower = -3, upper = +INF).
    // At x = 5, theta >= -3 + 2*5 = 7, so optimal theta = 7 and objective = 7.
    let view1 = solver
        .solve(None)
        .expect("first solve must be feasible and bounded");
    let obj1 = view1.objective;
    let theta1 = view1.primal[0];
    let x1 = view1.primal[1];

    println!("\n=== Test 1 — first solve (cut_row active) ===");
    println!("  objective = {obj1}");
    println!("  theta     = {theta1}");
    println!("  x         = {x1}");

    assert!(
        (obj1 - 7.0).abs() < 1e-9,
        "first solve objective must be 7.0, got {obj1}"
    );
    assert!(
        (theta1 - 7.0).abs() < 1e-9,
        "first solve theta must be 7.0, got {theta1}"
    );
    assert!(
        (x1 - 5.0).abs() < 1e-9,
        "first solve x must be 5.0, got {x1}"
    );

    // Deactivate cut_row (row 1) by setting bounds to [-INF, +INF].
    // With no remaining lower bound on theta, the LP is unbounded below.
    solver.set_row_bounds(&[1], &[-f64::INFINITY], &[f64::INFINITY]);

    let result2 = solver.solve(None);

    println!("\n=== Test 1 — second solve (cut_row deactivated) ===");
    println!("  result = {result2:?}");

    assert!(
        matches!(result2, Err(SolverError::Unbounded)),
        "second solve must be unbounded after deactivating the only cut row, got {result2:?}"
    );
}

/// Deactivating a dominated cut row leaves the LP optimum unchanged when another
/// active cut still bounds theta.
#[test]
fn inactive_cut_row_with_inf_bounds_still_bounded_when_other_cut_active() {
    let template = build_three_row_template();
    let empty_rows = make_empty_row_batch();

    let mut solver = HighsSolver::new().expect("HiGHS init");
    solver.load_model(&template);
    solver.add_rows(&empty_rows);

    // First solve: both cut rows active.
    // cut_row_A forces theta >= 7 (at x=5); cut_row_B forces theta >= -1000 (dominated).
    // Optimal theta = 7, objective = 7.
    let view1 = solver
        .solve(None)
        .expect("first solve must be feasible and bounded");
    let obj1 = view1.objective;
    let theta1 = view1.primal[0];
    let x1 = view1.primal[1];

    println!("\n=== Test 2 — first solve (both cuts active) ===");
    println!("  objective = {obj1}");
    println!("  theta     = {theta1}");
    println!("  x         = {x1}");

    assert!(
        (obj1 - 7.0).abs() < 1e-9,
        "first solve objective must be 7.0, got {obj1}"
    );

    // Deactivate cut_row_B (row 2) — the dominated cut.
    // cut_row_A (row 1) remains active and still forces theta >= 7.
    solver.set_row_bounds(&[2], &[-f64::INFINITY], &[f64::INFINITY]);

    let view2 = solver
        .solve(None)
        .expect("second solve must be feasible and bounded after deactivating dominated cut");
    let obj2 = view2.objective;
    let theta2 = view2.primal[0];
    let x2 = view2.primal[1];

    println!("\n=== Test 2 — second solve (cut_row_B deactivated) ===");
    println!("  objective = {obj2}");
    println!("  theta     = {theta2}");
    println!("  x         = {x2}");

    assert!(
        (obj2 - 7.0).abs() < 1e-9,
        "second solve objective must still be 7.0 after deactivating dominated cut, got {obj2}"
    );
    assert!(
        (theta2 - 7.0).abs() < 1e-9,
        "second solve theta must still be 7.0, got {theta2}"
    );
    assert!(
        (x2 - 5.0).abs() < 1e-9,
        "second solve x must still be 5.0, got {x2}"
    );
}

/// A deactivate-then-reactivate cycle on a dominated cut row must not corrupt LP state.
#[test]
fn reactivating_inactive_cut_row_restores_constraint() {
    let template = build_three_row_template();
    let empty_rows = make_empty_row_batch();

    let mut solver = HighsSolver::new().expect("HiGHS init");
    solver.load_model(&template);
    solver.add_rows(&empty_rows);

    // Deactivate cut_row_B (row 2) immediately.
    solver.set_row_bounds(&[2], &[-f64::INFINITY], &[f64::INFINITY]);

    // First solve after deactivation: cut_row_A still active, optimal theta = 7.
    let view1 = solver
        .solve(None)
        .expect("first solve must be feasible and bounded after deactivation");
    let obj1 = view1.objective;
    let theta1 = view1.primal[0];

    println!("\n=== Test 3 — first solve (cut_row_B deactivated) ===");
    println!("  objective = {obj1}");
    println!("  theta     = {theta1}");

    assert!(
        (obj1 - 7.0).abs() < 1e-9,
        "first solve objective must be 7.0 after deactivating dominated cut, got {obj1}"
    );

    // Reactivate cut_row_B with its original bounds.
    solver.set_row_bounds(&[2], &[-1000.0], &[f64::INFINITY]);

    // Second solve after reactivation: cut_row_B is dominated by cut_row_A,
    // so objective is still 7.0.  No state corruption from the deactivation cycle.
    let view2 = solver
        .solve(None)
        .expect("second solve must be feasible and bounded after reactivation");
    let obj2 = view2.objective;
    let theta2 = view2.primal[0];

    println!("\n=== Test 3 — second solve (cut_row_B reactivated) ===");
    println!("  objective = {obj2}");
    println!("  theta     = {theta2}");

    assert!(
        (obj2 - 7.0).abs() < 1e-9,
        "second solve objective must still be 7.0 after reactivating dominated cut, got {obj2}"
    );
    assert!(
        (theta2 - 7.0).abs() < 1e-9,
        "second solve theta must still be 7.0, got {theta2}"
    );
}
