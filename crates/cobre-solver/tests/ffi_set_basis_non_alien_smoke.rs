//! Smoke test for the warm-start basis path in `HighsSolver`.
//!
//! `cobre_highs_set_basis_non_alien` is the sole basis setter used at runtime.
//! It rejects bases where `col_basic + row_basic != num_row` by returning
//! `HIGHS_STATUS_ERROR`; `solve(Some(&basis))` maps that rejection to
//! `SolverError::BasisInconsistent`. No alien fallback exists.
//!
//! This test exercises the warm-start loop on a well-formed fixture and asserts
//! the self-extracted basis is accepted (near-zero `basis_consistency_failures`
//! relative to `basis_offered`). Driving it through `HighsSolver` directly
//! avoids a circular dev-dep on `cobre-sddp`.
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

use cobre_solver::{Basis, HighsSolver, SolverInterface, StageTemplate};

/// The SS1.1 fixture: 3 variables, 2 equality constraints.
///
/// Duplicated from `src/highs.rs` `#[cfg(test)] mod tests` because the private
/// test module is not accessible from integration test files.
///
///   min  0*x0 + 1*x1 + 50*x2
///   s.t. x0            = 6   (state-fixing)
///        2*x0 + x2     = 14  (power balance)
///   x0 in [0, 10], x1 in [0, +inf), x2 in [0, 8]
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

/// Simulated warm-start loop: verifies that `basis_consistency_failures` stays
/// near zero across many warm-start solve calls with a self-consistent basis.
///
/// Structure mirrors the baked-template backward pass:
///   1. Cold-solve once to obtain an optimal basis.
///   2. Loop N times: reload model → `solve(Some(&basis))` → `get_basis`.
///
/// The final `get_basis` in each iteration captures any updated basis for the
/// next iteration, matching how the SDDP pipeline propagates bases forward.
///
/// Assertions:
///   - `basis_offered > 0`
///   - `basis_consistency_failures / basis_offered < 0.01`
#[test]
fn non_alien_basis_loop_low_rejection_rate() {
    const ITERATIONS: usize = 60;

    // Arrange
    let template = make_fixture_stage_template();
    let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");

    // Cold-solve to get a consistent basis.
    solver.load_model(&template);
    solver.solve(None).expect("cold-start solve must succeed");

    let mut basis = Basis::new(template.num_cols, template.num_rows);
    solver.get_basis(&mut basis);

    let stats_before = solver.statistics();

    // Act — simulate the warm-start loop.
    for _ in 0..ITERATIONS {
        solver.load_model(&template);
        solver
            .solve(Some(&basis))
            .expect("warm-start solve must succeed");
        // Capture the updated basis for the next iteration, matching the
        // baked-template pipeline's basis propagation.
        solver.get_basis(&mut basis);
    }

    // Assert
    let stats_after = solver.statistics();
    let offered = stats_after.basis_offered - stats_before.basis_offered;
    let failures = stats_after.basis_consistency_failures - stats_before.basis_consistency_failures;

    assert!(
        offered > 0,
        "basis_offered must be > 0; got {offered} — the warm-start loop must have executed"
    );
    // Equivalent to failures/offered < 0.01 but avoids u64 → f64 precision-loss cast.
    // failures / offered < 1/100  ⟺  failures * 100 < offered
    assert!(
        failures.saturating_mul(100) < offered,
        "basis_consistency_failures / basis_offered must be < 0.01; \
         got {failures}/{offered}"
    );
}
