//! Smoke test for the warm-start basis path in `HighsSolver`.
//!
//! `cobre_highs_set_basis_non_alien` is the sole basis setter used at runtime; it
//! rejects bases where `col_basic + row_basic != num_row`, which
//! `solve(Some(&basis))` maps to `SolverError::BasisInconsistent`. No alien
//! fallback exists.
//!
//! This exercises the warm-start loop on a well-formed fixture and asserts the
//! self-extracted basis is accepted (near-zero `basis_consistency_failures`).
//! Driving it through `HighsSolver` directly avoids a circular dev-dep on
//! `cobre-sddp`.
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
#![cfg(feature = "highs")]

use cobre_solver::{Basis, HighsSolver, SolverInterface, StageTemplate};

/// The SS1.1 fixture: 3 variables, 2 equality constraints. Duplicated from the
/// `src/highs.rs` `#[cfg(test)]` module, which is not accessible from integration
/// test files.
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

/// Simulated warm-start loop: verifies that `basis_consistency_failures` stays
/// near zero across many warm-start solve calls with a self-consistent basis.
///
/// The reload → `solve(Some(&basis))` → `get_basis` loop mirrors how the pipeline
/// propagates a basis forward across solves.
#[test]
fn non_alien_basis_loop_low_rejection_rate() {
    const ITERATIONS: usize = 60;

    let template = make_fixture_stage_template();
    let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");

    solver.load_model(&template);
    solver.solve(None).expect("cold-start solve must succeed");

    let mut basis = Basis::new(template.num_cols, template.num_rows);
    solver.get_basis(&mut basis);

    let stats_before = solver.statistics();

    for _ in 0..ITERATIONS {
        solver.load_model(&template);
        solver
            .solve(Some(&basis))
            .expect("warm-start solve must succeed");
        solver.get_basis(&mut basis);
    }

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
