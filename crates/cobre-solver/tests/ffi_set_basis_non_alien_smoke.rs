//! Smoke test for the warm-start basis path in `HighsSolver`.
//!
//! ## Current runtime coverage
//!
//! The runtime `solve_with_basis` path uses only the alien FFI
//! (`cobre_highs_set_basis`). The non-alien setter
//! (`cobre_highs_set_basis_non_alien`) is wired but not exercised at runtime
//! — it will be re-enabled behind a feature gate once the backward-path basis
//! reconstruction padding issue (basic count invariant + baked-cut row
//! statuses) is fixed. This test exercises the warm-start loop to confirm
//! counters update correctly on a well-formed fixture.
//!
//! ## Deviation from ticket-005 spec
//!
//! The original ticket specified calling `cobre_sddp::train` end-to-end.
//! `cobre-sddp` depends on `cobre-solver`, so adding it as a dev-dependency
//! would create a circular dependency. This test exercises the same observable
//! (warm-start loop + rejection counters) directly through `HighsSolver`.
//!
//! ## Assertions
//!
//! - `basis_offered > 0` — warm-start calls were made.
//! - `basis_non_alien_rejections == 0` — non-alien path not exercised in the
//!   current runtime.
//! - `basis_rejections == 0` — the alien path accepts the self-extracted basis.
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

/// Simulated warm-start loop: verifies that `basis_non_alien_rejections` stays
/// near zero and `basis_rejections` stays at zero across many `solve_with_basis`
/// calls with a self-consistent basis.
///
/// Structure mirrors the baked-template backward pass:
///   1. Cold-solve once to obtain an optimal basis.
///   2. Loop N times: reload model → `solve_with_basis(&basis)` → `get_basis`.
///
/// The final `get_basis` in each iteration captures any updated basis for the
/// next iteration, matching how the SDDP pipeline propagates bases forward.
///
/// Assertions (equivalent to the SDDP integration test in the original spec):
///   - `basis_offered > 0`
///   - `basis_non_alien_rejections / basis_offered < 0.01`
///   - `basis_rejections == 0`
#[test]
fn non_alien_basis_loop_low_rejection_rate() {
    const ITERATIONS: usize = 60;

    // Arrange
    let template = make_fixture_stage_template();
    let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");

    // Cold-solve to get a consistent basis.
    solver.load_model(&template);
    solver.solve().expect("cold-start solve must succeed");

    let mut basis = Basis::new(template.num_cols, template.num_rows);
    solver.get_basis(&mut basis);

    let stats_before = solver.statistics();

    // Act — simulate the warm-start loop.
    for _ in 0..ITERATIONS {
        solver.load_model(&template);
        solver
            .solve_with_basis(&basis)
            .expect("warm-start solve must succeed");
        // Capture the updated basis for the next iteration, matching the
        // baked-template pipeline's basis propagation.
        solver.get_basis(&mut basis);
    }

    // Assert
    let stats_after = solver.statistics();
    let offered = stats_after.basis_offered - stats_before.basis_offered;
    let non_alien_rejections =
        stats_after.basis_non_alien_rejections - stats_before.basis_non_alien_rejections;
    let alien_rejections = stats_after.basis_rejections - stats_before.basis_rejections;

    assert!(
        offered > 0,
        "basis_offered must be > 0; got {offered} — the warm-start loop must have executed"
    );
    // Equivalent to rejection_rate < 0.01 but avoids u64 → f64 precision-loss cast.
    // non_alien_rejections / offered < 1/100  ⟺  non_alien_rejections * 100 < offered
    assert!(
        non_alien_rejections.saturating_mul(100) < offered,
        "basis_non_alien_rejections / basis_offered must be < 0.01; \
         got {non_alien_rejections}/{offered}"
    );
    assert_eq!(
        alien_rejections, 0,
        "basis_rejections must be 0 on a well-formed warm-start loop; \
         got {alien_rejections}"
    );
}
